//! `Driver::prepare`, [`DriverCaps`] and `compile_count` (Card 735, dserve.md section 3.1).
//!
//! `prepare` admits an explicit finite set of entries: the product of the row counts, the token
//! pieces, the heads, the verify windows and the adapter-set generations a deployment serves. Each
//! entry is traced, compiled, checked for the row-axis rule and retained under [`PreparedSetLimits`],
//! counting once. Capability is read from what compiled and is published as [`DriverCaps`]; nothing
//! reads a family or backend predicate. A step never compiles: a shape `prepare` did not admit is a
//! typed refusal, and admitting more is another `prepare` call, bounded by the same limits.

use std::collections::HashSet;
use std::num::NonZeroUsize;

use poot_graph_ir::op::SampleRule;
use poot_models::model::{KvLayout, LogitRows, StepShape};

use super::block_table::BLOCK_SIZE;
use super::chunks::Piece;
use super::error::{DriverError, InvalidOptions};
use super::lora::{AdapterSet, GenerationId};
use super::step::{Serving, serving_phase};
use super::suffix::Head;
use super::{Driver, EntryKey, Warm};

/// The paged pool a driver serves from: sized once, because state avals pin one pool per driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolShape {
    /// Physical KV blocks of [`BLOCK_SIZE`] tokens, shared by every open sequence and the prefix cache.
    pub blocks: NonZeroUsize,
    /// Sequences that may be open at once.
    pub max_seqs: NonZeroUsize,
}

impl PoolShape {
    pub(super) fn kv(self) -> KvLayout {
        KvLayout::Paged {
            pool_slots: NonZeroUsize::new(self.blocks.get() * BLOCK_SIZE)
                .expect("a pool of at least one block"),
        }
    }
}

/// How the KV cache of the admitted entries is laid out. A driver admits one layout for its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// One private cache for one sequence: no block table, no prefix reuse.
    Contiguous,
    /// The shared paged pool, a block table per sequence.
    Paged(PoolShape),
}

/// The shapes one `prepare` call admits. Every field is explicit: there is no default.
#[derive(Clone, Copy, Debug)]
pub struct ServingShapes<'a> {
    pub layout: Layout,
    /// Row counts to compile; one (`1`) under [`Layout::Contiguous`]. A step runs on the smallest
    /// admitted count holding its rows; the rows beyond the live ones are masked, so composition never
    /// recompiles.
    pub rows: &'a [NonZeroUsize],
    /// The heads to compile: sampling suffixes and/or the logits head the host picks from.
    pub heads: &'a [Head],
    /// Which token pieces to compile besides the decode step.
    pub warm: Warm<'a>,
    /// Verify-window widths (query token plus drafts, at least two): greedy suffix entries that return
    /// a pick at every window position.
    pub windows: &'a [NonZeroUsize],
    /// Adapter-set generations to compile entries for, besides the base weights.
    pub adapters: &'a [GenerationId],
}

/// Whether a model's carried state is only positional KV (a rejected token's write is harmless) or
/// folds every token into a value the driver cannot roll back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateResidency {
    KvOnly,
    Recurrent,
}

/// Which sampling heads compiled.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeadSet {
    /// The sampling-suffix rules with prepared entries (greedy, the Gumbel family). A step whose
    /// picking rows share one of these runs it; any other mix needs `with_logits`.
    pub suffix: Vec<SampleRule>,
    /// The logits head: rows that need the host path pick from its output.
    pub with_logits: bool,
}

/// One admitted verify window: `width` positions on an entry of `rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCap {
    pub rows: NonZeroUsize,
    pub width: NonZeroUsize,
}

/// Which inputs a step accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputSet {
    pub tokens: bool,
}

/// What the compiled adapter entries cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoraCaps {
    /// Adapter-set generations with prepared entries.
    pub generations: usize,
}

/// What `prepare` compiled: the only source `EnginePlan` reads capability, capacity and readiness from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriverCaps {
    /// The largest row count with prepared entries.
    pub rows: NonZeroUsize,
    /// Every row count with prepared entries: a step runs on the smallest one holding its live rows.
    pub row_counts: Vec<NonZeroUsize>,
    /// Every token width with prepared entries at the last-position logits (prefill pieces and the
    /// one-token decode step).
    pub widths: Vec<NonZeroUsize>,
    /// Positions a sequence may hold (prompt plus generated).
    pub max_seq: NonZeroUsize,
    pub kv: KvLayout,
    pub kv_blocks: usize,
    pub state: StateResidency,
    pub heads: HeadSet,
    pub lora: Option<LoraCaps>,
    /// Verify windows with prepared entries, per row count: a window is admitted only on the row
    /// counts a `prepare` call listed beside it.
    pub windows: Vec<WindowCap>,
    pub inputs: InputSet,
}

impl Driver {
    /// Admit `shapes`: compile every entry of its product that is not prepared yet, then publish the
    /// caps. Everything checkable is validated before anything compiles, and the layout and the paged
    /// pool are published last, so a refused call leaves the driver as it was except for the entries
    /// that compiled before the refusal, which stay prepared (an entry over the retention limits is a
    /// typed refusal; in-flight work on retained entries is untouched).
    pub fn prepare(&mut self, shapes: &ServingShapes<'_>) -> Result<&DriverCaps, DriverError> {
        if let Some(width) = shapes.windows.iter().find(|width| width.get() < 2) {
            return Err(InvalidOptions::WindowBelowTwo { width: width.get() }.into());
        }
        match (self.layout, shapes.layout) {
            (Some(held), asked) if held != asked => {
                return Err(InvalidOptions::LayoutChanged.into());
            }
            (_, Layout::Contiguous)
                if shapes.rows.iter().any(|rows| rows.get() != 1) || !shapes.windows.is_empty() =>
            {
                return Err(InvalidOptions::ContiguousServesOneRow.into());
            }
            _ => {}
        }
        let serves_one = |residency: StateResidency| {
            let seqs = match shapes.layout {
                Layout::Contiguous => 1,
                Layout::Paged(pool) => pool.max_seqs.get(),
            };
            residency == StateResidency::Recurrent
                && (seqs > 1
                    || shapes.rows.iter().any(|rows| rows.get() > 1)
                    || !shapes.windows.is_empty())
        };
        if serves_one(self.residency) {
            return Err(InvalidOptions::RecurrentServesOneSequence.into());
        }
        let kv = match shapes.layout {
            Layout::Contiguous => KvLayout::Contiguous,
            Layout::Paged(pool) => pool.kv(),
        };
        let mut pieces = match shapes.warm {
            Warm::Admitted => self.chunks.admitted_pieces(),
            Warm::Prompts(lens) => lens.iter().flat_map(|&len| self.chunks.plan(len)).collect(),
        };
        pieces.push(Piece::Decode);
        let mut seen = HashSet::new();
        pieces.retain(|piece| seen.insert(*piece));

        let adapters = std::iter::once(None).chain(shapes.adapters.iter().copied().map(Some));
        for adapters in adapters {
            for &rows in shapes.rows {
                for &head in shapes.heads {
                    for &piece in &pieces {
                        self.prepare_entry(kv, piece_key(piece), rows, head, adapters)?;
                    }
                }
                for &width in shapes.windows {
                    let key = (width.get(), LogitRows::All);
                    self.prepare_entry(kv, key, rows, Head::GREEDY, adapters)?;
                }
            }
        }
        // The residency of this model is known once an entry compiled.
        if serves_one(self.residency) {
            return Err(InvalidOptions::RecurrentServesOneSequence.into());
        }
        if self.serving.is_none() {
            self.serving = Some(Serving::new(shapes.layout));
        }
        self.layout = Some(shapes.layout);
        Ok(self.refresh_caps(shapes.layout))
    }

    /// Capture an adapter set: the generation already holding it, or a new one. Entries for it exist
    /// once a `prepare` names it. The serve lifecycle's LoRA admin route is the caller.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "held for POOT-589 (LoRA admin through the engine)"
        )
    )]
    pub(crate) fn capture_adapter_set(
        &mut self,
        set: AdapterSet,
    ) -> Result<GenerationId, DriverError> {
        Ok(self.lora.capture(set)?)
    }

    /// Unload `generation` and drop its prepared entries, returning their retained bytes. Refused
    /// while an open sequence holds it. The serve lifecycle's LoRA admin route is the caller.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "held for POOT-589 (LoRA admin through the engine)"
        )
    )]
    pub(crate) fn unload_adapter_set(
        &mut self,
        generation: GenerationId,
    ) -> Result<(), DriverError> {
        self.lora.check_unloadable(generation)?;
        let dropped: Vec<EntryKey> = self
            .entries
            .keys()
            .filter(|key| key.adapters == Some(generation))
            .copied()
            .collect();
        // The registry keeps the generation until every entry is gone, so a failed removal leaves a
        // loaded generation whose remaining entries a retry drops; the map never names an entry the
        // executor no longer has.
        let mut outcome = Ok(());
        for key in dropped {
            let id = self.entries[&key].id;
            if let Err(error) = self.executor.remove_entry(self.exe, id) {
                outcome = Err(error.into());
                break;
            }
            let entry = self.entries.remove(&key).expect("collected from the map");
            self.drop_holds(&entry.held);
        }
        if outcome.is_ok() {
            let identities = self.lora.identities_of(generation);
            self.lora.unload(generation)?;
            if let Some(serving) = self.serving.as_mut() {
                serving.evict_identities(&identities);
            }
        }
        if let Some(layout) = self.layout {
            self.refresh_caps(layout);
        }
        outcome
    }

    /// The published caps; `None` until `prepare` has admitted a shape.
    pub fn caps(&self) -> Option<&DriverCaps> {
        self.caps.as_ref()
    }

    /// Programs compiled so far: constant across steps drawn from the prepared set. Serve's
    /// composition-change row reads it.
    #[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-753"))]
    pub(crate) fn compile_count(&self) -> u64 {
        self.stats.compiles
    }

    fn prepare_entry(
        &mut self,
        kv: KvLayout,
        (tokens, logits): (usize, LogitRows),
        rows: NonZeroUsize,
        head: Head,
        adapters: Option<GenerationId>,
    ) -> Result<(), DriverError> {
        let key = EntryKey {
            phase: serving_phase(tokens, logits),
            shape: StepShape {
                rows,
                tokens: NonZeroUsize::new(tokens).expect("a piece carries at least one token"),
                capacity: self.options.capacity,
                kv,
                logits,
            },
            head,
            adapters,
        };
        self.prepared(key).map(|_| ())
    }

    /// Recompute the caps from the prepared entries of `layout`.
    fn refresh_caps(&mut self, layout: Layout) -> &DriverCaps {
        let kv = match layout {
            Layout::Contiguous => KvLayout::Contiguous,
            Layout::Paged(pool) => pool.kv(),
        };
        let mut row_counts = Vec::new();
        let mut widths = Vec::new();
        let mut heads = HeadSet::default();
        let mut windows = Vec::new();
        let mut generations = HashSet::new();
        for key in self.entries.keys().filter(|key| key.shape.kv == kv) {
            let StepShape { rows, tokens, .. } = key.shape;
            if !row_counts.contains(&rows) {
                row_counts.push(rows);
            }
            match key.head {
                Head::Logits => heads.with_logits = true,
                Head::Sample(rule) if !heads.suffix.contains(&rule) => heads.suffix.push(rule),
                Head::Sample(_) => {}
            }
            if key.shape.logits == LogitRows::All {
                let window = WindowCap {
                    rows,
                    width: tokens,
                };
                if !windows.contains(&window) {
                    windows.push(window);
                }
            } else if !widths.contains(&tokens) {
                widths.push(tokens);
            }
            generations.extend(key.adapters);
        }
        row_counts.sort();
        widths.sort();
        windows.sort_by_key(|w| (w.rows, w.width));
        self.caps.insert(DriverCaps {
            rows: row_counts.last().copied().unwrap_or(NonZeroUsize::MIN),
            row_counts,
            widths,
            max_seq: self.options.capacity,
            kv,
            kv_blocks: match layout {
                Layout::Contiguous => 0,
                Layout::Paged(pool) => pool.blocks.get(),
            },
            state: self.residency,
            heads,
            lora: (!generations.is_empty()).then_some(LoraCaps {
                generations: generations.len(),
            }),
            windows,
            inputs: InputSet { tokens: true },
        })
    }

    /// Blocks free in the paged pool (cached prefix blocks count as held); `None` before a paged
    /// `prepare`.
    pub fn free_blocks(&self) -> Option<usize> {
        self.serving.as_ref().and_then(Serving::free_blocks)
    }
}

fn piece_key(piece: Piece) -> (usize, LogitRows) {
    match piece {
        Piece::Prefill(n) => (n.get(), LogitRows::Last),
        Piece::Decode => (1, LogitRows::Last),
    }
}

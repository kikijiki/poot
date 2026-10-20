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

use poot_graph_ir::ops::sampling::{Adjust, SampleSpec};
use poot_models::io::{InputId, InputSet, RequestInputs};
use poot_models::model::{KvLayout, LogitRows, StepShape};

use super::bind::{self, RowFeed};
use super::block_table::BLOCK_SIZE;
use super::chunks::Piece;
use super::encode::{EncodeCaps, EncodeShapes};
use super::error::{DriverError, InvalidOptions};
use poot_load::lora::LoraAdapter;

use super::lora::{AdapterSet, AdapterStack, GenerationId};
use super::sequences::Serving;
use super::speculate::SpeculationCaps;
use super::step::{Lane, RowPlan};
use super::suffix::Head;
use super::{Driver, EntryKey, Feature, FeatureSet, ModelRole, Program, State, Unprepared, Warm};

/// The most carriable inputs one deployment prepares combinations of.
const MAX_CARRIED_INPUTS: usize = 4;

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
            seqs: self.max_seqs,
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
    /// The KV layout every admitted entry uses.
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
    /// The most decode steps a row chains for one host wait ([`RowWork::Chain`](super::RowWork::Chain)):
    /// for each row count and each sampling head a device chain can carry (a plain or adjusted draw with
    /// no logprobs), `depth - 1` more copies of the head's decode entry, one per step after the first
    /// (the first runs the ordinary entry). `None` prepares no chain.
    pub chain: Option<NonZeroUsize>,
}

/// Whether a model's carried state is only positional KV (a rejected token's write is harmless) or
/// folds every token into a value the driver cannot roll back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StateResidency {
    /// Carried state is positional KV only: a rejected token's write is overwritten and harmless.
    #[default]
    KvOnly,
    /// Some carried state folds every token into one value, so a rejected token must be rolled back
    /// by a snapshot.
    Recurrent,
}

/// Which sampling heads compiled.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeadSet {
    /// The sampling heads with prepared entries (the greedy argmax, the Gumbel family, an adjusted or
    /// logprob variant of each). A step runs the narrowest of them that serves all its picking rows;
    /// a request none serves is refused, never picked on the host.
    pub suffix: Vec<SampleSpec>,
    /// The logits head: the full logits for a caller that asked for them (a sampled verify window).
    pub with_logits: bool,
}

/// One admitted verify window: `width` positions on an entry of `rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCap {
    /// Rows of the entry the window runs on.
    pub rows: NonZeroUsize,
    /// Positions the window verifies: the query token plus its drafts.
    pub width: NonZeroUsize,
}

/// Which inputs a driver accepts beyond a token prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputCaps {
    /// The carriable request inputs (a sparse or per-row input the model declares) the target's `step`
    /// entries bind: a request carrying one of them is served only if it is listed
    /// ([`Driver::prepare_carried`]).
    pub carried: Vec<InputId>,
    /// [`Driver::encode`]: an `encode` entry is prepared.
    pub encode: bool,
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
    /// How the KV cache is laid out.
    pub kv: KvLayout,
    /// Blocks in the paged pool; 0 under the contiguous layout.
    pub kv_blocks: usize,
    /// Whether the carried state is positional KV only or recurrent.
    pub state: StateResidency,
    /// Which sampling heads have prepared entries.
    pub heads: HeadSet,
    /// Adapter coverage; `None` when no adapter generation is compiled.
    pub lora: Option<LoraCaps>,
    /// Verify windows with prepared entries, per row count: a window is admitted only on the row
    /// counts a `prepare` call listed beside it.
    pub windows: Vec<WindowCap>,
    /// The speculation the driver's draft policy serves; `None` under `DraftPolicy::Off`.
    pub speculation: Option<SpeculationCaps>,
    /// Which inputs the driver accepts beyond a token prompt.
    pub inputs: InputCaps,
    /// The token-row shapes [`Driver::prepare_encode`] admitted; `None` when none were.
    pub encode: Option<EncodeCaps>,
}

impl Driver<Unprepared> {
    /// A driver serving `layout`, with nothing prepared: [`Driver::generate`] prepares the entries its
    /// request needs, and [`Driver::extend`] admits more. The layout is the driver's for life.
    #[must_use]
    pub fn serve(self, layout: Layout) -> Driver<Serving> {
        let state = Serving::new(layout);
        Driver {
            handle: self.handle,
            runtime: self.runtime,
            settings: self.settings,
            ledger: self.ledger,
            speculation: self.speculation,
            lora: self.lora,
            stats: self.stats,
            state,
        }
    }

    /// Admit `shapes`: compile every entry of its product, then publish the caps and return the
    /// driver that serves them. Everything checkable is validated before anything compiles. A refused
    /// call gives the driver back with the refusal: the entries that compiled before it stay prepared
    /// (an entry over the retention limits is a typed refusal), and nothing is published.
    pub fn prepare(
        self,
        shapes: &ServingShapes<'_>,
    ) -> Result<Driver<Serving>, Box<PrepareRefused>> {
        let mut serving = self.serve(shapes.layout);
        match serving.extend(shapes) {
            Ok(_) => Ok(serving),
            Err(error) => Err(Box::new(PrepareRefused {
                driver: serving.unserve(),
                error,
            })),
        }
    }
}

/// A [`Driver::prepare`] the driver refused, with the driver it left behind.
pub struct PrepareRefused {
    /// The driver, still unprepared: entries compiled before the refusal stay prepared.
    pub driver: Driver<Unprepared>,
    /// Why the shapes were refused.
    pub error: DriverError,
}

impl std::fmt::Debug for PrepareRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepareRefused")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Display for PrepareRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for PrepareRefused {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl From<Box<PrepareRefused>> for DriverError {
    fn from(refused: Box<PrepareRefused>) -> Self {
        refused.error
    }
}

impl Driver<Serving> {
    /// Admit `shapes` beside what is prepared: compile every entry of its product that is not prepared
    /// yet, then publish the caps. `shapes` must name the layout the driver serves. Everything
    /// checkable is validated before anything compiles; a refused call leaves the entries that
    /// compiled before the refusal prepared (an entry over the retention limits is a typed refusal;
    /// in-flight work on retained entries is untouched).
    pub fn extend(&mut self, shapes: &ServingShapes<'_>) -> Result<&DriverCaps, DriverError> {
        if let Some(width) = shapes.windows.iter().find(|width| width.get() < 2) {
            return Err(InvalidOptions::WindowBelowTwo { width: width.get() }.into());
        }
        let held = self.state.layout();
        if held != shapes.layout {
            return Err(InvalidOptions::LayoutChanged.into());
        }
        if held == Layout::Contiguous && shapes.rows.iter().any(|rows| rows.get() != 1) {
            return Err(InvalidOptions::ContiguousServesOneRow.into());
        }
        self.check_draft()?;
        self.plan_hidden_carry();
        let kv = match shapes.layout {
            Layout::Contiguous => KvLayout::Contiguous,
            Layout::Paged(pool) => pool.kv(),
        };
        let mut pieces = match shapes.warm {
            Warm::Admitted => self.settings.chunks.admitted_pieces(),
            Warm::Prompts(lens) => lens
                .iter()
                .flat_map(|&len| self.settings.chunks.plan(len))
                .collect(),
        };
        pieces.push(Piece::Decode);
        let mut seen = HashSet::new();
        pieces.retain(|piece| seen.insert(*piece));

        let adapters = std::iter::once(None).chain(shapes.adapters.iter().copied().map(Some));
        for adapters in adapters {
            for &rows in shapes.rows {
                for &head in shapes.heads {
                    for &piece in &pieces {
                        let target = ModelRole::Target;
                        self.prepare_entry(target, kv, piece_key(piece), rows, head, adapters)?;
                    }
                }
            }
        }
        // The residency of this model is known once an entry compiled, before any window compiles.
        self.check_speculation_state()?;
        self.compose(FeatureSet::NONE.with_if(!shapes.windows.is_empty(), Feature::VerifyWindows))?;
        let adapters = std::iter::once(None).chain(shapes.adapters.iter().copied().map(Some));
        for adapters in adapters {
            for &rows in shapes.rows {
                for &width in shapes.windows {
                    let key = (width.get(), LogitRows::All);
                    self.prepare_entry(ModelRole::Target, kv, key, rows, Head::GREEDY, adapters)?;
                }
            }
        }
        self.prepare_speculation(shapes, kv, &pieces)?;
        if let Some(depth) = shapes.chain {
            self.prepare_chain(shapes, kv, depth)?;
        }
        Ok(self.refresh_caps())
    }

    /// The chained decode of `depth` steps for the heads of `shapes` a device chain can carry; see
    /// [`ServingShapes::chain`].
    fn prepare_chain(
        &mut self,
        shapes: &ServingShapes<'_>,
        kv: KvLayout,
        depth: NonZeroUsize,
    ) -> Result<(), DriverError> {
        let adapters = std::iter::once(None).chain(shapes.adapters.iter().copied().map(Some));
        for adapters in adapters {
            for &rows in shapes.rows {
                for &head in shapes.heads {
                    let Head::Sample(spec) = head else {
                        continue;
                    };
                    if spec.logprobs.is_some() || spec.adjust == Adjust::Guided {
                        continue;
                    }
                    for step in 1..depth.get() {
                        let key = self.step_key(
                            ModelRole::Target,
                            kv,
                            piece_key(Piece::Decode),
                            rows,
                            head,
                            adapters,
                            InputSet::EMPTY,
                        );
                        let chain = u8::try_from(step).map_err(|_| InvalidOptions::ChainTooLong)?;
                        self.prepared(EntryKey { chain, ..key })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// This driver as it was before its layout was fixed: the prepared entries stay, the open
    /// sequences and the pool go.
    fn unserve(self) -> Driver<Unprepared> {
        Driver {
            handle: self.handle,
            runtime: self.runtime,
            settings: self.settings,
            ledger: self.ledger,
            speculation: self.speculation,
            lora: self.lora,
            stats: self.stats,
            state: Unprepared,
        }
    }
}

impl<S: State> Driver<S> {
    /// Capture an adapter set: the generation already holding it, or a new one with `adapters` stacked
    /// against the model's weights. Entries for it exist once a `prepare` names it. The serve
    /// lifecycle's LoRA admin route is the caller.
    pub fn capture_adapter_set(
        &mut self,
        set: AdapterSet,
        adapters: &[LoraAdapter],
    ) -> Result<GenerationId, DriverError> {
        let stack = AdapterStack::build(adapters, self.handle.model().weights())?;
        Ok(self.lora.capture(set, stack)?)
    }

    /// Unload `generation` and drop its prepared entries, returning their retained bytes. Refused
    /// while an open sequence holds it. The serve lifecycle's LoRA admin route is the caller.
    pub fn unload_adapter_set(&mut self, generation: GenerationId) -> Result<(), DriverError> {
        self.lora.check_unloadable(generation)?;
        let dropped: Vec<EntryKey> = self
            .ledger
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
            let (id, exe) = (self.ledger.entries[&key].id, self.ledger.entries[&key].exe);
            if let Err(error) = self.runtime.executor.remove_entry(exe, id) {
                outcome = Err(error.into());
                break;
            }
            let entry = self
                .ledger
                .entries
                .remove(&key)
                .expect("collected from the map");
            self.ledger.drop_holds(&entry.held);
        }
        // The executable goes first: if it cannot unload, the generation stays registered and mapped, and a
        // retry unloads it. Only then does the map forget it and the registry drop the generation.
        if outcome.is_ok()
            && let Some(&exe) = self.runtime.adapter_exes.get(&generation)
        {
            match self.runtime.executor.unload(exe) {
                Ok(()) => {
                    self.runtime.adapter_exes.remove(&generation);
                }
                Err(error) => outcome = Err(error.into()),
            }
        }
        if outcome.is_ok() {
            let identities = self.lora.identities_of(generation);
            self.lora.unload(generation)?;
            self.state.evict_identities(&identities);
        }
        self.refresh_caps_if_published();
        outcome
    }

    /// The published caps; `None` until `prepare` has admitted a shape.
    pub fn caps(&self) -> Option<&DriverCaps> {
        self.ledger.caps.as_ref()
    }
}

impl Driver<Serving> {
    /// Admit the speculation of the driver's draft policy: a verify-window entry per row count beside
    /// decode, and under a draft model its own decode and prefill entries on a second executable.
    /// The window carries the greedy suffix, and the logits head when a request samples.
    fn prepare_speculation(
        &mut self,
        shapes: &ServingShapes<'_>,
        kv: KvLayout,
        pieces: &[Piece],
    ) -> Result<(), DriverError> {
        let Some(speculation) = self.speculation.policy.caps() else {
            return Ok(());
        };
        let sampled = shapes.heads.iter().any(|&head| head != Head::GREEDY);
        let heads: &[Head] = if sampled {
            &[Head::GREEDY, Head::Logits]
        } else {
            &[Head::GREEDY]
        };
        let window = (speculation.window.get(), LogitRows::All);
        for adapters in std::iter::once(None).chain(shapes.adapters.iter().copied().map(Some)) {
            for &rows in shapes.rows {
                for &head in heads {
                    self.prepare_entry(ModelRole::Target, kv, window, rows, head, adapters)?;
                }
            }
        }
        self.prepare_rollback(shapes, kv, pieces)?;
        self.load_draft()?;
        if self.speculation.draft.is_some() {
            // The draft's prompt pieces run unread and a proposal is one decode step: only that
            // step needs the logits head of a sampled row.
            for &rows in shapes.rows {
                for &piece in pieces {
                    let draft = ModelRole::Draft;
                    self.prepare_entry(draft, kv, piece_key(piece), rows, Head::GREEDY, None)?;
                    if sampled && piece == Piece::Decode {
                        self.prepare_entry(draft, kv, piece_key(piece), rows, Head::Logits, None)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl<S: State> Driver<S> {
    fn prepare_entry(
        &mut self,
        role: ModelRole,
        kv: KvLayout,
        (tokens, logits): (usize, LogitRows),
        rows: NonZeroUsize,
        head: Head,
        adapters: Option<GenerationId>,
    ) -> Result<(), DriverError> {
        let key = self.step_key(
            role,
            kv,
            (tokens, logits),
            rows,
            head,
            adapters,
            InputSet::EMPTY,
        );
        self.prepared(key).map(|_| ())
    }

    /// The key of the `step` entry of `tokens` tokens per row over `rows` rows that binds `carried`.
    #[expect(clippy::too_many_arguments)]
    fn step_key(
        &self,
        role: ModelRole,
        kv: KvLayout,
        (tokens, logits): (usize, LogitRows),
        rows: NonZeroUsize,
        head: Head,
        adapters: Option<GenerationId>,
        carried: InputSet,
    ) -> EntryKey {
        EntryKey {
            chain: 0,
            role,
            program: if role == ModelRole::Mtp {
                Program::Draft
            } else {
                Program::Step
            },
            shape: StepShape::new(
                rows,
                NonZeroUsize::new(tokens).expect("a piece carries at least one token"),
                self.settings.options.capacity,
            )
            .with_kv(kv)
            .with_logits(logits)
            .with_hidden(self.hidden_of(role)),
            head,
            adapters,
            carried,
        }
    }

    /// The request inputs among `inputs` the target's `step` program binds only on the steps that
    /// carry them (cadence `Covers` or `AtStart`).
    pub(super) fn carriable_in(&self, inputs: &RequestInputs) -> Vec<InputId> {
        let schema = self.handle.model().schema();
        let Some(program) = schema.step() else {
            return Vec::new();
        };
        let carriable = schema.carriable(program);
        let mut ids: Vec<InputId> = inputs
            .iter()
            .map(|(id, _)| id)
            .filter(|&id| {
                schema
                    .index_of(id)
                    .is_some_and(|index| carriable.contains(index))
            })
            .collect();
        ids.sort();
        ids.dedup();
        ids
    }
}

impl Driver<Serving> {
    /// The carriable inputs the step of `lanes` and `plans` binds: the entry it runs declares exactly
    /// those slots.
    pub(super) fn carried_by(&self, lanes: &[Lane<'_>], plans: &[RowPlan]) -> InputSet {
        let schema = self.handle.model().schema();
        let Some(program) = schema.step() else {
            return InputSet::EMPTY;
        };
        let serving = &self.state;
        let feeds: Vec<RowFeed<'_>> = lanes
            .iter()
            .zip(plans)
            .map(|(lane, plan)| RowFeed {
                inputs: &serving.seqs[&lane.seq].inputs,
                start: plan.start,
                fed: plan.feed.len(),
            })
            .collect();
        bind::carried(schema, program, &feeds)
    }

    /// Admit the entries a request with carriable inputs runs, beside the entries [`Driver::prepare`]
    /// admitted: for every non-empty combination of `inputs` (which the model's `step` program reads
    /// as carriable inputs) one entry per prefill piece `shapes.warm` plans, per head and row count.
    /// `shapes` must name the layout a `prepare` already published. A request that carries an input
    /// no prepared entry binds is refused at admission.
    pub fn prepare_carried(
        &mut self,
        shapes: &ServingShapes<'_>,
        inputs: &[InputId],
    ) -> Result<&DriverCaps, DriverError> {
        if self.state.layout() != shapes.layout {
            return Err(InvalidOptions::LayoutChanged.into());
        }
        let schema = self.handle.model().schema();
        let carriable = schema
            .step()
            .map_or(InputSet::EMPTY, |program| schema.carriable(program));
        let mut indices = Vec::with_capacity(inputs.len());
        for &id in inputs {
            match schema.index_of(id) {
                Some(index) if carriable.contains(index) => indices.push(index),
                _ => return Err(InvalidOptions::NotCarriable { input: id }.into()),
            }
        }
        indices.sort_unstable();
        indices.dedup();
        if indices.len() > MAX_CARRIED_INPUTS {
            return Err(InvalidOptions::TooManyCarried {
                count: indices.len(),
                max: MAX_CARRIED_INPUTS,
            }
            .into());
        }
        let combinations: Vec<InputSet> = (1u32..1 << indices.len())
            .map(|mask| {
                indices
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| mask >> bit & 1 == 1)
                    .fold(InputSet::EMPTY, |set, (_, &index)| set.with(index))
            })
            .collect();
        let kv = match shapes.layout {
            Layout::Contiguous => KvLayout::Contiguous,
            Layout::Paged(pool) => pool.kv(),
        };
        let pieces: Vec<Piece> = match shapes.warm {
            Warm::Admitted => self.settings.chunks.admitted_pieces(),
            Warm::Prompts(lens) => lens
                .iter()
                .flat_map(|&len| self.settings.chunks.plan(len))
                .collect(),
        };
        let mut seen = HashSet::new();
        for piece in pieces {
            let Piece::Prefill(tokens) = piece else {
                continue;
            };
            if !seen.insert(tokens) {
                continue;
            }
            for &rows in shapes.rows {
                for &head in shapes.heads {
                    for &carried in &combinations {
                        let key = self.step_key(
                            ModelRole::Target,
                            kv,
                            (tokens.get(), LogitRows::Last),
                            rows,
                            head,
                            None,
                            carried,
                        );
                        self.prepared(key)?;
                    }
                }
            }
        }
        Ok(self.refresh_caps())
    }

    /// Blocks free in the paged pool (cached prefix blocks count as held); `None` under the contiguous
    /// layout.
    pub fn free_blocks(&self) -> Option<usize> {
        self.state.free_blocks()
    }
}

impl<S: State> Driver<S> {
    /// Recompute the caps from the prepared entries of the layout the driver serves.
    pub(super) fn refresh_caps(&mut self) -> &DriverCaps {
        let layout = self.state.admitted_layout();
        let kv = match layout {
            None | Some(Layout::Contiguous) => KvLayout::Contiguous,
            Some(Layout::Paged(pool)) => pool.kv(),
        };
        let mut row_counts = Vec::new();
        let mut widths = Vec::new();
        let mut heads = HeadSet::default();
        let mut windows = Vec::new();
        let mut generations = HashSet::new();
        let target_entries = self.ledger.entries.keys().filter(|key| {
            key.role == ModelRole::Target
                && key.shape.kv == kv
                && key.program == Program::Step
                && key.carried.is_empty()
                && key.chain == 0
        });
        for key in target_entries {
            let StepShape { rows, tokens, .. } = key.shape;
            if !row_counts.contains(&rows) {
                row_counts.push(rows);
            }
            match key.head {
                Head::Logits => heads.with_logits = true,
                Head::Sample(spec) if !heads.suffix.contains(&spec) => heads.suffix.push(spec),
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
        heads.suffix.sort();
        windows.sort_by_key(|w| (w.rows, w.width));
        self.ledger.caps.insert(DriverCaps {
            rows: row_counts
                .last()
                .copied()
                .or(self.ledger.encode_shapes.map(|shapes| shapes.rows))
                .unwrap_or(NonZeroUsize::MIN),
            row_counts,
            widths,
            max_seq: self.settings.options.capacity,
            kv,
            kv_blocks: match layout {
                None | Some(Layout::Contiguous) => 0,
                Some(Layout::Paged(pool)) => pool.blocks.get(),
            },
            state: self.ledger.residency,
            heads,
            lora: (!generations.is_empty()).then_some(LoraCaps {
                generations: generations.len(),
            }),
            windows,
            speculation: self.speculation.policy.caps(),
            inputs: self.input_caps(),
            encode: self.ledger.encode_shapes.map(EncodeShapes::caps),
        })
    }

    /// The inputs beyond a token prompt the prepared entries accept.
    fn input_caps(&self) -> InputCaps {
        let schema = self.handle.model().schema();
        let target = |key: &&EntryKey| key.role == ModelRole::Target;
        let mut carried: Vec<InputId> = self
            .ledger
            .entries
            .keys()
            .filter(target)
            .filter(|key| key.program == Program::Step)
            .flat_map(|key| schema.ids_of(key.carried))
            .collect();
        carried.sort();
        carried.dedup();
        InputCaps {
            carried,
            encode: self
                .ledger
                .entries
                .keys()
                .filter(target)
                .any(|key| key.program == Program::Encode),
        }
    }

    /// Republish the caps when a layout has been admitted: an entry prepared outside `prepare`
    /// (an encode step's) changes what the driver accepts.
    pub(super) fn refresh_caps_if_published(&mut self) {
        if self.state.admitted_layout().is_some() || self.ledger.encode_shapes.is_some() {
            self.refresh_caps();
        }
    }
}

fn piece_key(piece: Piece) -> (usize, LogitRows) {
    match piece {
        Piece::Prefill(n) => (n.get(), LogitRows::Last),
        Piece::Decode => (1, LogitRows::Last),
    }
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    impl<S: State> Driver<S> {
        /// Programs compiled so far: constant across steps drawn from the prepared set. The serving rows read
        /// it through [`Driver::stats`].
        pub(crate) fn compile_count(&self) -> u64 {
            self.stats.compiles
        }
    }
}

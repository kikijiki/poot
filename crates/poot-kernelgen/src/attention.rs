//! The attention generator: one online-softmax body for every phase.
//!
//! An [`AttentionSpec`] describes the region (score transform, mask kind, head grouping, bound) and a
//! schedule; [`attention_body`] turns it into one kernel. `lq` and `lk` are independent: decode (`lq == 1`
//! against a capacity-long cache), prefill (`lq == lk`) and prefill into a cache (`1 < lq < lk`) differ only in
//! the extents they pass.
//!
//! The [`AttentionSchedule::Streaming`] schedule is one workgroup of `lanes` threads per `(batch, head,
//! query row)`. Each lane strides over the keys (`t = lane, lane + lanes, ..`), keeping its own running
//! `(m, l, o[dv])` triple: `m` the running score maximum, `l` the running sum of `exp(score - m)`, `o` the
//! running `exp(score - m)`-weighted sum of value rows. Admitting a key at score `s` moves the maximum to
//! `m' = max(m, s)` and rescales what the lane holds by `exp(m - m')` before adding the key's own term, so
//! the `[lq, lk]` scores are never materialized. The partials live in workgroup memory (`o` in array 0,
//! `lanes * dv` wide, lane `i` at `[i*dv, i*dv + dv)` with its channels XOR-swizzled by the lane (see
//! [`OSlots`]); `m` and `l` in arrays 1 and 2, `lanes` wide each),
//! because a dynamically indexed private array cannot be stated on SPIR-V. One barrier, on the straight-line
//! path after the key loops, separates the per-lane work (disjoint LDS regions, barrier-free) from the
//! combine. The combine then runs on every lane over a slice of the output channels: with `M` the maximum of
//! the lane maxima, `out[c] = sum_i o_i[c] * exp(m_i - M) / sum_i l_i * exp(m_i - M)`.
//!
//! [`AttentionSchedule::SplitKv`] is the same row split over the keys across `splits` workgroups (decode,
//! where `batch * heads` workgroups leave a many-CU device idle). Stage one is this body with each
//! workgroup visiting only its own contiguous chunk of the keys and publishing the combine's inputs
//! instead of dividing: `o[dv]` unnormalized, then `m` and `l`, at `[(row * splits + split) * (dv + 2)]` of
//! the scratch it writes. A split with no key to visit (a live length short of its chunk) publishes the
//! neutral `(o = 0, m = -inf, l = 0)` by the same arithmetic as a lane that saw nothing, so the scratch
//! needs no initialization. Stage two ([`AttentionCombineSpec`]) is the `lanes`-wide combine over the
//! splits.
//!
//! [`AttentionSchedule::Tiled`] is the multi-row schedule (FlashAttention-2): one workgroup per block of query
//! rows of a key/value head, the head's query heads packed into the block, each block of keys and values staged
//! in workgroup memory once for all of its rows and both products run as register tiles over the micro-tile
//! interface. Its body is in [`tiled`] and reads the same parameters.
//!
//! Operands are dense: `q [B, Hq, Lq, Dk]`, `k [B, Hkv, Lk, Dk]` and `v [B, Hkv, Lk, Dv]`, `mask [B, Hm, Lq, Lk]`
//! (additive), `out [B, Hq, Lq, Dv]`, GQA through `kv_head = head / n_rep`. A [`KeyLayout::Split`] key
//! is `k [B, Hkv, Lk, Dk - s]` plus `k_shared [B, 1, Lk, s]`, the shared part read by every key head, and
//! the dot product runs over the two parts in turn. A [`KeyLayout::Paged`] key and value are read in place
//! from `[S, Hkv, D]` pools through an `I32 [B, Lk]` slot map, the same loop with a different address. A
//! [`KeyBound::Live`] row visits only the first `len[b]`
//! keys, `len` an `I32 [B]` operand clamped into `0..=Lk` before it bounds the loop: a length outside that
//! range cannot reach memory past the buffers, and (like the decomposition) selects no key at all, or every
//! key. The grid and the recorded launch never see `len`. A position whose score is `-inf` (every
//! unmasked key so far is absent) contributes exactly zero: `exp(-inf - -inf)` would be NaN, so an empty
//! running state is routed around the subtraction. A row with no unmasked key at all divides zero by zero,
//! which is the NaN the oracle's softmax gives.

use poot_kernel_ir::{
    BinOp, Body, Constant, IndexAxis, LocalDecl, MathOp, Operand, Place, Rvalue, Ty,
    WorkgroupLocalDecl,
};

use crate::KernelGenError;
use crate::data_movement::Cfg;
use crate::emit::{copy as copy_local, cu};
use crate::fused::{FusedScalarOp, emit_scalar_op};
use crate::helpers::{copy, elem, ld, slice_dtype, slice_f32};

mod tiled;

/// What the planner asks the attention generator for: the region's shape, its score transform, mask and
/// bound, and the schedule that runs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionSpec {
    /// Batch rows `B`.
    pub batch: usize,
    /// Query heads `Hq`.
    pub hq: usize,
    /// Query heads per key/value head (`hq / hkv`).
    pub n_rep: usize,
    /// Query rows per batch row and head (`1` for decode).
    pub lq: usize,
    /// Key/value positions per batch row and head: the cache capacity for decode, `lq` for prefill.
    pub lk: usize,
    /// The query and key head width.
    pub dk: usize,
    /// The value and output head width. Independent of `dk` (MLA: `dk` 192, `dv` 128).
    pub dv: usize,
    /// How the key operand is laid out.
    pub key: KeyLayout,
    /// The transform from the raw dot product to a score.
    pub score: ScoreTransform,
    /// The mask operand's head axis.
    pub mask: MaskKind,
    /// Which keys each row visits.
    pub bound: KeyBound,
    /// Whether the row's softmax carries a per-head sink logit.
    pub sink: AttentionSink,
    /// How the keys are walked and across how many workgroups.
    pub schedule: AttentionSchedule,
    /// How the key and value elements (the shared key part too) are stored.
    pub kv: KvLane,
}

/// How an attention's key and value elements sit in their buffers (POOT-1395): f32, or a two-byte float in its
/// native lane that the body widens to f32 on load. Every score and sum is f32 whichever lane is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvLane {
    /// One f32 per element.
    F32,
    /// One native IEEE binary16 per element.
    F16,
    /// One native bfloat16 per element.
    Bf16,
}

impl KvLane {
    /// The key or value parameter's declaration.
    fn param(self) -> LocalDecl {
        let elem = match self {
            KvLane::F32 => return ld(slice_f32(false), false),
            KvLane::F16 => Ty::F16,
            KvLane::Bf16 => Ty::BF16,
        };
        ld(slice_dtype(elem, false), false)
    }

    /// Element `idx` of `buffer`, as an f32 local.
    fn read(
        self,
        e: &mut crate::emit::Emit,
        buffer: poot_kernel_ir::Local,
        idx: poot_kernel_ir::Local,
    ) -> poot_kernel_ir::Local {
        match self {
            KvLane::F32 => e.let_(Ty::F32, Rvalue::Use(copy(elem(buffer, idx)))),
            KvLane::F16 | KvLane::Bf16 => e.let_(
                Ty::F32,
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(buffer, idx)),
                },
            ),
        }
    }
}

impl AttentionSpec {
    /// Query rows: one per `(batch, head, query row)`.
    pub fn rows(&self) -> usize {
        self.batch * self.hq * self.lq
    }

    /// Query heads one workgroup serves: the `n_rep` heads of a key/value head under a grouped
    /// [`MaskKind::Blocks`] (they read the same selected keys, so the workgroup loads each key and value
    /// once for all of them), one otherwise.
    /// The tiled schedule always serves the `n_rep` heads of a key/value head, packed into its tile rows.
    pub fn heads_per_workgroup(&self) -> usize {
        match (self.schedule, self.mask) {
            (AttentionSchedule::Tiled { .. }, _) => self.n_rep,
            (_, MaskKind::Blocks { grouped: true, .. }) => self.n_rep,
            (
                _,
                MaskKind::Blocks { grouped: false, .. }
                | MaskKind::Broadcast
                | MaskKind::PerHead
                | MaskKind::Causal { .. },
            ) => 1,
        }
    }

    /// The query rows one workgroup serves: its heads' rows of one query position, or under the tiled schedule
    /// the `br` positions of a block for each of its heads (the last block of a head may hold fewer real ones).
    pub fn rows_per_workgroup(&self) -> usize {
        match self.schedule {
            AttentionSchedule::Tiled { br, .. } => br * self.n_rep,
            AttentionSchedule::Streaming { .. } | AttentionSchedule::SplitKv { .. } => {
                self.heads_per_workgroup()
            }
        }
    }

    /// Row groups: the rows one workgroup serves, `rows() / heads_per_workgroup()`, or under the tiled schedule
    /// one per `(batch, key/value head, block of br query positions)`.
    pub fn row_groups(&self) -> usize {
        match self.schedule {
            AttentionSchedule::Tiled { br, .. } => {
                self.batch * (self.hq / self.n_rep) * self.lq.div_ceil(br)
            }
            AttentionSchedule::Streaming { .. } | AttentionSchedule::SplitKv { .. } => {
                self.rows() / self.heads_per_workgroup()
            }
        }
    }

    /// Workgroups the schedule launches: one per row group, or one per `(row group, split)`.
    pub fn workgroups(&self) -> usize {
        match self.schedule {
            AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => {
                self.row_groups()
            }
            AttentionSchedule::SplitKv { splits, .. } => self.row_groups() * splits,
        }
    }

    /// The steps a row walks: its keys one at a time, or under [`MaskKind::Blocks`] runs of [`block_step`]
    /// consecutive keys of one block.
    pub fn walk_units(&self) -> usize {
        match self.mask {
            MaskKind::Blocks { ratio, k, grouped } => {
                (k + 1) * (ratio / block_step(ratio, grouped))
            }
            MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Causal { .. } => self.lk,
        }
    }

    /// The steps one workgroup walks: all of the row's, or one split's chunk.
    pub fn units_per_workgroup(&self) -> usize {
        match self.schedule {
            AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => {
                self.walk_units()
            }
            AttentionSchedule::SplitKv { splits, .. } => self.walk_units().div_ceil(splits),
        }
    }

    /// The key positions a row considers: all `lk`, or under [`MaskKind::Blocks`] the `(K + 1) * ratio` of its
    /// selected blocks and its unfinished block. The split schedule divides these.
    pub fn candidates(&self) -> usize {
        match self.mask {
            MaskKind::Blocks { ratio, k, .. } => (k + 1) * ratio,
            MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Causal { .. } => self.lk,
        }
    }

    /// The workgroup-memory bytes one workgroup of a [`AttentionSchedule::Tiled`] schedule declares, or the typed
    /// reason its tiles do not fit the shape (`None` for the other schedules, whose footprint is their lanes'
    /// partials).
    pub fn tiled_lds_bytes(&self) -> Option<Result<usize, KernelGenError>> {
        match self.schedule {
            AttentionSchedule::Tiled { .. } => Some(tiled::lds_bytes(self)),
            AttentionSchedule::Streaming { .. } | AttentionSchedule::SplitKv { .. } => None,
        }
    }

    /// The `f32` elements the split schedule's partial state occupies: `(o[dv], m, l)` per `(row,
    /// split)`. `None` for the streaming and tiled schedules, which have no intermediate.
    pub fn partial_elems(&self) -> Option<usize> {
        match self.schedule {
            AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => None,
            AttentionSchedule::SplitKv { splits, .. } => Some(self.rows() * splits * (self.dv + 2)),
        }
    }
}

/// A contiguous run of a streaming or tiled attention's row groups ([`AttentionSpec::row_groups`]: its rows, under
/// a grouped block selection the key/value heads' rows, under the tiled schedule its blocks of rows), launched as one dispatch of its own: `count` row groups
/// of `spec` starting at `first`. A prefill whose whole grid would run past the device's watchdog is several
/// of these over disjoint rows of one output, all this one body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionRows {
    /// The whole attention: its shape and its [`AttentionSchedule::Streaming`] or [`AttentionSchedule::Tiled`]
    /// schedule.
    pub spec: AttentionSpec,
    /// The first row group of the run, in the `(batch, head, query row)` order of the rows they serve.
    pub first: usize,
    /// Row groups in the run; at least 1, and `first + count` at most `spec.row_groups()`.
    pub count: usize,
}

/// The combine stage of a split-KV attention: one workgroup of `lanes` threads per row, each lane a strided
/// share of the `dv` output channels, reduces the row's `splits` partials stage one wrote into the scratch.
/// Params: the partials `[rows, splits, dv + 2]` (read), then `out [rows, dv]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionCombineSpec {
    /// `batch * hq * lq` rows.
    pub rows: usize,
    /// Splits per row that stage one wrote partials for.
    pub splits: usize,
    /// The value and output head width.
    pub dv: usize,
    /// The workgroup width, `lanes` threads striding over the `dv` channels.
    pub lanes: usize,
}

impl AttentionCombineSpec {
    /// The combine stage that reads the partials `spec`'s split schedule writes: the same rows, splits
    /// and value width, `None` for a streaming schedule.
    pub fn of(spec: &AttentionSpec) -> Option<Self> {
        match spec.schedule {
            AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => None,
            AttentionSchedule::SplitKv { lanes, splits } => Some(Self {
                rows: spec.rows(),
                splits,
                dv: spec.dv,
                lanes,
            }),
        }
    }
}

/// How a raw `q . k` dot product becomes a score before the mask is added. Scale and cap ride as `f32`
/// bits, so the request stays plain data with an injective `Debug`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreTransform {
    /// `score = dot * scale`.
    Scale {
        /// The `f32` bit pattern of the multiplier applied to the dot product.
        scale_bits: u32,
    },
    /// `score = cap * tanh(dot * scale / cap)`: Gemma2 and Grok attention-logit softcapping.
    ScaleSoftcap {
        /// The `f32` bit pattern of the multiplier applied to the dot product.
        scale_bits: u32,
        /// The `f32` bit pattern of the softcap `cap`, the transform's saturation magnitude.
        cap_bits: u32,
    },
}

/// The mask operand's head axis `Hm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskKind {
    /// `Hm == 1`: one `[Lq, Lk]` plane per batch row, shared by every head.
    Broadcast,
    /// `Hm == Hq`: one plane per head (ALiBi).
    PerHead,
    /// No mask tensor: the parameter that follows the value is the `I32 [B, Lq]` position of each query row,
    /// and key `t` is visible when `t <= position` and, under a window, `position - t < window`. The loop
    /// visits only those keys: it ends at the position and starts at the window's first key, so a hidden key
    /// costs nothing and the scores need no mask. A negative position (a padded query) is read as 0.
    Causal {
        /// The most recent keys a query sees; `None` is every earlier key. At least 1.
        window: Option<usize>,
    },
    /// No mask operand: the parameters that follow the value are the `I32 [B, Lq, K]` ids of the blocks each
    /// query selected (ascending, `-1` after the last) and the `I32 [B, Lq]` query positions. A row visits
    /// only the keys of its selected blocks of `ratio` positions and its own unfinished block, `(K + 1) *
    /// ratio` candidates in all, instead of the `lk` positions; the scores need no mask. Dense keys and the
    /// static bound only; the split schedule divides the candidates.
    Blocks {
        /// Positions per block.
        ratio: usize,
        /// Selected blocks per query: the width of `ids`.
        k: usize,
        /// Whether a workgroup serves the `n_rep` query heads of a key/value head together. They select the
        /// same blocks (the ids are per query, not per head), so the workgroup loads each key and value row
        /// once for all of them instead of once per head: a row group of `n_rep` head lanes times the
        /// schedule's slices, `lanes` a multiple of `n_rep`. Off, a workgroup is one head, as for a dense row.
        grouped: bool,
    },
}

/// How the key operand is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyLayout {
    /// One dense `[B, Hkv, Lk, Dk]` tensor.
    Single,
    /// `[B, Hkv, Lk, Dk - shared_dim]` followed by a `[B, 1, Lk, shared_dim]` tensor shared by every key
    /// head: the kernel's `k` parameter is the first part and `k_shared` follows `mask`.
    Split {
        /// Width `s` of the part shared by every key head; the first part is `Dk - s` wide.
        shared_dim: usize,
    },
    /// The keys and values stay in their paged pools: `k` is a `[S, Hkv, Dk]` pool, `v` a `[S, Hkv, Dv]` pool,
    /// and the `I32 [B, Lk]` `read` parameter that follows `mask` names the pool slot of each position. Key
    /// `t` of batch row `b` and key head `g` is row `(read[b * Lk + t] * Hkv + g)` of its pool. Every word of
    /// `read` must be a slot of the pool: the kernel does not clamp it.
    Paged,
}

/// Which keys a row visits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyBound {
    /// All `lk` positions, a loop bound known when the kernel is generated; the mask removes the ones that
    /// must not count.
    Static,
    /// The first `len[b]` positions of batch row `b`, read at run time from an `I32 [B]` parameter that
    /// follows the key parameters. The loop bound moves; the launch does not.
    Live,
}

/// Whether the softmax includes a sink: one learned logit per query head that joins the row maximum and the
/// normalizer and is no key, so it weighs nothing in the value sum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionSink {
    /// No sink.
    Absent,
    /// An `F32 [Hq]` parameter, the last before `out`, holds the sink logit of each head. It is the initial
    /// running state of the row's first lane (of its first split): `m = sink`, `l = 1` (`exp(sink - m)`),
    /// `o = 0`, so admitting the keys, the lane combine and the split combine need no sink case of their own.
    PerHead,
}

/// How the region's keys are walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionSchedule {
    /// One workgroup of `lanes` threads per `(batch, head, query row)`, lanes striding over the keys.
    Streaming {
        /// Threads per workgroup, each striding over the keys.
        lanes: usize,
    },
    /// Stage one of split-KV attention: one workgroup of `lanes` threads per `(batch, head, query row,
    /// split)`, each walking its own contiguous `ceil(lk / splits)`-key chunk and writing a partial state
    /// to a scratch instead of the output. [`AttentionCombineSpec`] reduces the partials.
    SplitKv {
        /// Threads per workgroup, each striding over its chunk's keys.
        lanes: usize,
        /// Workgroups the keys of one row are divided across.
        splits: usize,
    },
    /// FlashAttention-2 form (Dao, arXiv 2307.08691): one workgroup of `lanes` threads per `(batch, key/value
    /// head, block of br query positions)`, the `n_rep` query heads of the key/value head packed into the
    /// block's `br * n_rep` tile rows. The keys are walked in blocks of `bc`: each block's keys and values are
    /// staged in workgroup memory once for every row, the `rows x bc` score tile and the `rows x dv` output
    /// accumulator are register tiles ([`AttentionMicro`]), and the online softmax runs per row. See
    /// [`tiled`](self::tiled) for the body.
    Tiled {
        /// Query positions per workgroup; the block's tile rows are these for each of the `n_rep` heads.
        br: usize,
        /// Keys staged per block.
        bc: usize,
        /// Threads per workgroup.
        lanes: usize,
        /// The register tiles each lane accumulates.
        micro: AttentionMicro,
    },
}

impl AttentionSchedule {
    /// The workgroup width: the lanes that stride the keys.
    pub fn lanes(self) -> usize {
        match self {
            Self::Streaming { lanes } | Self::SplitKv { lanes, .. } | Self::Tiled { lanes, .. } => {
                lanes
            }
        }
    }
}

/// The per-lane register tiles of a [`AttentionSchedule::Tiled`] workgroup: one for the score tile of a key
/// block, one for the output accumulator. The lanes of a tile form a grid of `rows / tile.rows` by `cols /
/// tile.cols` lanes, which must be the workgroup's `lanes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttentionMicro {
    /// Each lane's part of the `(br * n_rep) x bc` score tile.
    pub score: LaneTile,
    /// Each lane's part of the `(br * n_rep) x dv` output accumulator.
    pub out: LaneTile,
}

/// A lane's `rows x cols` register tile, interleaved over its tile (see [`crate::microtile`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaneTile {
    /// Rows per lane.
    pub rows: usize,
    /// Columns per lane.
    pub cols: usize,
}

impl LaneTile {
    /// The f32 accumulators of the tile.
    pub const fn cells(self) -> usize {
        self.rows * self.cols
    }
}

const GENERATOR: &str = "attention_body";

/// The `(m, l, o)` partial arrays, in the workgroup-local declaration order the body uses.
const O_ARRAY: u8 = 0;
const M_ARRAY: u8 = 1;
const L_ARRAY: u8 = 2;

fn bad(what: &str, value: usize, min: usize) -> KernelGenError {
    KernelGenError::BelowMinimum {
        generator: GENERATOR,
        what: what.to_string(),
        value,
        min,
    }
}

/// Check that `rows` is a run of `spec`'s workgroups: nonempty, inside the grid, and the whole grid unless the
/// schedule is streaming (a split-KV stage one and its combine address every workgroup).
fn validate_rows(
    spec: &AttentionSpec,
    rows: &std::ops::Range<usize>,
) -> Result<(), KernelGenError> {
    let workgroups = spec.workgroups();
    if rows.is_empty() {
        return Err(bad("rows", 0, 1));
    }
    if rows.end > workgroups {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "rows".to_string(),
            value: rows.end,
            bound: workgroups,
        });
    }
    if matches!(spec.schedule, AttentionSchedule::SplitKv { .. }) && *rows != (0..workgroups) {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "rows of a split-KV schedule".to_string(),
            value: rows.len(),
            bound: workgroups,
        });
    }
    Ok(())
}

/// Check the spec's own preconditions: every extent nonzero, `hq` a multiple of `n_rep`, and a split key's
/// shared part a proper, nonempty slice of the key width.
fn validate(spec: &AttentionSpec) -> Result<usize, KernelGenError> {
    let lanes = spec.schedule.lanes();
    let splits = match spec.schedule {
        AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => 1,
        AttentionSchedule::SplitKv { splits, .. } => splits,
    };
    for (what, value) in [
        ("splits", splits),
        ("batch", spec.batch),
        ("hq", spec.hq),
        ("n_rep", spec.n_rep),
        ("lq", spec.lq),
        ("lk", spec.lk),
        ("dk", spec.dk),
        ("dv", spec.dv),
        ("lanes", lanes),
    ] {
        if value == 0 {
            return Err(bad(what, value, 1));
        }
    }
    if !spec.hq.is_multiple_of(spec.n_rep) {
        return Err(KernelGenError::NotDivisible {
            generator: GENERATOR,
            dim: "hq".to_string(),
            value: spec.hq,
            divisor: spec.n_rep,
        });
    }
    // A split with no key at all is a wasted workgroup, and `ceil(candidates / splits)` would leave it empty.
    if splits > spec.candidates() {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "splits".to_string(),
            value: splits,
            bound: spec.candidates(),
        });
    }
    if spec.mask == (MaskKind::Causal { window: Some(0) }) {
        return Err(bad("window", 0, 1));
    }
    if let MaskKind::Blocks { ratio, k, grouped } = spec.mask {
        for (what, value) in [("ratio", ratio), ("k", k)] {
            if value == 0 {
                return Err(bad(what, value, 1));
            }
        }
        if spec.key != KeyLayout::Single || spec.bound != KeyBound::Static {
            return Err(KernelGenError::Unsupported {
                request: "attention",
                reason: "a block-selected attention reads a dense key under the static bound",
            });
        }
        // The lanes of a grouped selection are `n_rep` head lanes per key-walking slice.
        if grouped && !lanes.is_multiple_of(spec.n_rep) {
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: "lanes".to_string(),
                value: lanes,
                divisor: spec.n_rep,
            });
        }
    }
    if let KeyLayout::Split { shared_dim } = spec.key {
        if shared_dim == 0 {
            return Err(bad("shared_dim", shared_dim, 1));
        }
        if shared_dim >= spec.dk {
            return Err(KernelGenError::ExceedsBound {
                generator: GENERATOR,
                what: "shared_dim".to_string(),
                value: shared_dim,
                bound: spec.dk - 1,
            });
        }
    }
    Ok(lanes)
}

pub(crate) fn f32_array(len: usize) -> WorkgroupLocalDecl {
    WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: len as u32,
    }
}

pub(crate) fn f32c(value: f32) -> Operand {
    Operand::Const(Constant::F32(value))
}

pub(crate) fn local_op(local: poot_kernel_ir::Local) -> Operand {
    copy_local(local)
}

/// A usize `a op b` of two locals.
pub(crate) fn ubin(
    c: &mut Cfg,
    op: BinOp,
    a: poot_kernel_ir::Local,
    b: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    c.bin(Ty::Usize, op, local_op(a), local_op(b))
}

/// A usize `a op n`.
pub(crate) fn uc(
    c: &mut Cfg,
    op: BinOp,
    a: poot_kernel_ir::Local,
    n: usize,
) -> poot_kernel_ir::Local {
    c.bin(Ty::Usize, op, local_op(a), cu(n))
}

/// An f32 `a op b` of two locals.
pub(crate) fn fbin(
    c: &mut Cfg,
    op: BinOp,
    a: poot_kernel_ir::Local,
    b: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    c.bin(Ty::F32, op, local_op(a), local_op(b))
}

fn exp(c: &mut Cfg, x: poot_kernel_ir::Local) -> poot_kernel_ir::Local {
    c.let_(Ty::F32, Rvalue::MathUnary(MathOp::Exp, local_op(x)))
}

pub(crate) fn lds_read(
    c: &mut Cfg,
    array: u8,
    idx: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    c.emit(|e| e.read_lds(Ty::F32, array, local_op(idx)))
}

pub(crate) fn lds_write(c: &mut Cfg, array: u8, idx: poot_kernel_ir::Local, value: Operand) {
    c.emit(|e| e.write_lds(array, local_op(idx), value));
}

/// `var <- var + 1`.
pub(crate) fn bump(c: &mut Cfg, var: poot_kernel_ir::Local, by: usize) {
    let next = uc(c, BinOp::Add, var, by);
    c.assign(Place::local(var), Rvalue::Use(local_op(next)));
}

/// `while counter < end { body; counter += step }` over a fresh usize counter starting at `start`.
pub(crate) fn count(
    c: &mut Cfg,
    start: Operand,
    end: usize,
    step: usize,
    body: impl FnOnce(&mut Cfg, poot_kernel_ir::Local),
) {
    count_to(c, start, cu(end), step, body);
}

/// [`count`] with an end known only at run time.
pub(crate) fn count_to(
    c: &mut Cfg,
    start: Operand,
    end: Operand,
    step: usize,
    body: impl FnOnce(&mut Cfg, poot_kernel_ir::Local),
) {
    let counter = c.var(Ty::Usize, Rvalue::Use(start));
    c.while_(
        |c| c.bin(Ty::Bool, BinOp::Lt, local_op(counter), end),
        |c| {
            body(c, counter);
            bump(c, counter, step);
        },
    );
}

/// The workgroup-memory banks (4-byte words wide) of the devices: consecutive words fall in consecutive banks.
const BANKS: usize = 32;

/// Where the channels of one lane's partial `o` sit in workgroup memory: lane `i` at `[i * dv, i * dv + dv)`,
/// channel `c` at `i * dv + (c ^ (i % 32))` when `dv` is a multiple of the bank count. Without the XOR every
/// lane of a warp touching one channel would hit the same bank (the stride `dv` is a multiple of 32), a
/// 32-way conflict on each access; with it the lanes of a warp land in 32 different banks, and so do the
/// lanes of a combine that each read their own channel of one lane's partial. The XOR permutes the channels of
/// a 32-aligned group among themselves, so the footprint is exactly `lanes * dv` words (a padded stride cost
/// an occupancy step at 128K on RTX PRO 6000). A `dv` that is not a multiple of 32 keeps the plain layout.
struct OSlots {
    start: poot_kernel_ir::Local,
    bank: Option<poot_kernel_ir::Local>,
}

impl OSlots {
    fn new(c: &mut Cfg, dv: usize, lane: poot_kernel_ir::Local) -> Self {
        Self {
            start: uc(c, BinOp::Mul, lane, dv),
            bank: dv
                .is_multiple_of(BANKS)
                .then(|| uc(c, BinOp::Rem, lane, BANKS)),
        }
    }

    /// The slot of `channel`.
    fn at(&self, c: &mut Cfg, channel: poot_kernel_ir::Local) -> poot_kernel_ir::Local {
        let within = match self.bank {
            Some(bank) => ubin(c, BinOp::BitXor, channel, bank),
            None => channel,
        };
        ubin(c, BinOp::Add, self.start, within)
    }
}

/// Where a row's keys and values start in their operands.
#[derive(Clone, Copy)]
enum KeyRows {
    /// Dense operands: the element offsets of the row's `(batch, head)` plane in `k`, `k_shared` and `v`.
    Dense {
        k: poot_kernel_ir::Local,
        shared: poot_kernel_ir::Local,
        v: poot_kernel_ir::Local,
    },
    /// Pools read through the slot map parameter `read`, whose words for this batch row start at `read_base`.
    Paged {
        read: poot_kernel_ir::Local,
        read_base: poot_kernel_ir::Local,
    },
}

/// A key a block-selected row considers: its position, and whether the row reads it.
struct Candidate {
    key: poot_kernel_ir::Local,
    valid: poot_kernel_ir::Local,
    /// The query's position (a negative one read as 0).
    position: poot_kernel_ir::Local,
}

/// The keys of one block a block-selected row reads in one step. A grouped workgroup (a prefill chunk, whose
/// rows fill the device) takes the largest of 4, 2 and 1 that divides the block, so a step never crosses a
/// block and its keys are consecutive rows. A per-head row (decode, which splits its few rows across the
/// device) takes one key, so its lanes stay as many as the candidates.
fn block_step(ratio: usize, grouped: bool) -> usize {
    if !grouped {
        return 1;
    }
    [4, 2, 1]
        .into_iter()
        .find(|step| ratio.is_multiple_of(*step))
        .expect("1 divides every block size")
}

/// Candidate `candidate` of query `query` (an index into `ids` and `pos`) among the `(k + 1) * ratio` a
/// block-selected row considers: the positions of its `k` selected blocks in order, then the positions of its
/// own unfinished block. `(k, ratio, lk)` are the selection width, the block size and the stored key
/// extent. A position is read when it is real (`id >= 0`, or inside the unfinished block), at or before
/// the query, and inside the stored keys: whatever `ids` holds, no address leaves the key buffers.
fn block_candidate(
    c: &mut Cfg,
    ids: poot_kernel_ir::Local,
    pos: poot_kernel_ir::Local,
    query: poot_kernel_ir::Local,
    (k, ratio, lk): (usize, usize, usize),
    candidate: poot_kernel_ir::Local,
) -> Candidate {
    let i32c = |value: i32| Operand::Const(Constant::I32(value));
    let widen = |c: &mut Cfg, word: poot_kernel_ir::Local| {
        c.let_(
            Ty::Usize,
            Rvalue::Cast {
                to: Ty::Usize,
                operand: local_op(word),
            },
        )
    };
    let raw = c.let_(Ty::I32, Rvalue::Use(copy(elem(pos, query))));
    let at = c.bin(Ty::I32, BinOp::Max, local_op(raw), i32c(0));
    let position = widen(c, at);
    let next = uc(c, BinOp::Add, position, 1);
    let complete = uc(c, BinOp::Div, next, ratio);
    let tail_start = uc(c, BinOp::Mul, complete, ratio);

    let slot = uc(c, BinOp::Div, candidate, ratio);
    let offset = uc(c, BinOp::Rem, candidate, ratio);
    let in_ids = c.bin(Ty::Bool, BinOp::Lt, local_op(slot), cu(k));
    // Read a real slot of `ids` even for a tail candidate: the value is discarded there.
    let last_slot = c.usize_const(k - 1);
    let slot_read =
        crate::block_top_k::select(c, Ty::Usize, in_ids, local_op(slot), local_op(last_slot));
    let id_at = {
        let row = uc(c, BinOp::Mul, query, k);
        ubin(c, BinOp::Add, row, slot_read)
    };
    let id = c.let_(Ty::I32, Rvalue::Use(copy(elem(ids, id_at))));
    let id_real = c.bin(Ty::Bool, BinOp::Ge, local_op(id), i32c(0));
    let id_clamped = c.bin(Ty::I32, BinOp::Max, local_op(id), i32c(0));
    let block = widen(c, id_clamped);
    let block_start = uc(c, BinOp::Mul, block, ratio);
    let selected_key = ubin(c, BinOp::Add, block_start, offset);
    let tail_key = ubin(c, BinOp::Add, tail_start, offset);
    let key = crate::block_top_k::select(
        c,
        Ty::Usize,
        in_ids,
        local_op(selected_key),
        local_op(tail_key),
    );
    let tail_real = c.bin(Ty::Bool, BinOp::Le, local_op(tail_key), local_op(position));
    let real =
        crate::block_top_k::select(c, Ty::Bool, in_ids, local_op(id_real), local_op(tail_real));
    let causal = c.bin(Ty::Bool, BinOp::Le, local_op(key), local_op(position));
    let stored = c.bin(Ty::Bool, BinOp::Lt, local_op(key), cu(lk));
    let readable = c.both(real, causal);
    let valid = c.both(readable, stored);
    Candidate {
        key,
        valid,
        position,
    }
}

/// Build the attention kernel `spec` describes over its workgroups `rows` (all of them, except for a streaming
/// schedule split into runs of rows). The launch is `rows.len()` workgroups of `lanes` threads laid over a
/// 2-D grid `x_groups` workgroups across: `GroupY * x_groups + GroupX` is the workgroup's place in the run,
/// `gi = rows.start +` that place is the flat `(batch, head, row)` index, and a workgroup past the run (the
/// fold's remainder) returns.
/// Parameters: `q`, `k`, `v`, `mask`, then `k_shared` for a split key or `read` for a paged one, then `len` for a
/// live bound, then `sinks` for a sink, then `out`
/// (the partial-state scratch under [`AttentionSchedule::SplitKv`]: the `x_groups`/`gi` index then runs over
/// `(row, split)`, `gi = row * splits + split`).
pub(crate) fn attention_body(
    name: &str,
    spec: &AttentionSpec,
    rows: std::ops::Range<usize>,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    let lanes = validate(spec)?;
    validate_rows(spec, &rows)?;
    if let AttentionSchedule::Tiled { .. } = spec.schedule {
        return tiled::tiled_body(name, spec, rows, x_groups);
    }
    let AttentionSpec {
        batch: _,
        hq,
        n_rep,
        lq,
        lk,
        dk,
        dv,
        key,
        score,
        mask: mask_kind,
        bound,
        sink,
        schedule,
        kv,
    } = *spec;
    let splits = match schedule {
        AttentionSchedule::Streaming { .. } => None,
        AttentionSchedule::SplitKv { splits, .. } => Some(splits),
        AttentionSchedule::Tiled { .. } => unreachable!("the tiled schedule has its own body"),
    };
    let chunk = spec.units_per_workgroup();
    let hkv = hq / n_rep;
    let mask_heads = match mask_kind {
        MaskKind::Broadcast | MaskKind::Causal { .. } | MaskKind::Blocks { .. } => 1,
        MaskKind::PerHead => hq,
    };

    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        kv.param(),                  // 2 k
        kv.param(),                  // 3 v
    ];
    // 4 the additive mask, the query positions, or the selected block ids and the query positions (5)
    match mask_kind {
        MaskKind::Broadcast | MaskKind::PerHead => params.push(ld(slice_f32(false), false)),
        MaskKind::Causal { .. } => params.push(ld(slice_dtype(Ty::I32, false), false)),
        MaskKind::Blocks { .. } => {
            params.push(ld(slice_dtype(Ty::I32, false), false));
            params.push(ld(slice_dtype(Ty::I32, false), false));
        }
    }
    let mut next_param = |decl| {
        params.push(decl);
        poot_kernel_ir::Local {
            index: params.len() as u32 - 1,
        }
    };
    let k_shared = match key {
        KeyLayout::Single | KeyLayout::Paged => None,
        KeyLayout::Split { .. } => Some(next_param(kv.param())),
    };
    let read = match key {
        KeyLayout::Single | KeyLayout::Split { .. } => None,
        KeyLayout::Paged => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
    };
    let len = match bound {
        KeyBound::Static => None,
        KeyBound::Live => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
    };
    let sinks = match sink {
        AttentionSink::Absent => None,
        AttentionSink::PerHead => Some(next_param(ld(slice_f32(false), false))),
    };
    let out = next_param(ld(slice_f32(true), true));
    let (q, k, v, mask) = (
        poot_kernel_ir::Local { index: 1 },
        poot_kernel_ir::Local { index: 2 },
        poot_kernel_ir::Local { index: 3 },
        poot_kernel_ir::Local { index: 4 },
    );
    let mut c = Cfg::new(params);
    let shared_dim = match key {
        KeyLayout::Single | KeyLayout::Paged => 0,
        KeyLayout::Split { shared_dim } => shared_dim,
    };
    let head_dim = dk - shared_dim;

    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let lane = c.index(IndexAxis::LocalX);
    // Under a block selection the lanes are `group` head lanes (the query heads of one key/value head) for each
    // of `slices` key-walking slices, lane `l` being head lane `l % group` of slice `l / group`; elsewhere every
    // lane is its own slice of the one head.
    let group = spec.heads_per_workgroup();
    let slices = lanes / group;
    let lane_split = (group > 1).then(|| {
        (
            uc(&mut c, BinOp::Rem, lane, group),
            uc(&mut c, BinOp::Div, lane, group),
        )
    });
    let slice = lane_split.map_or(lane, |(_, slice)| slice);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let place = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(place), cu(rows.len()));

    c.if_then(in_range, |c| {
        let gi = uc(c, BinOp::Add, place, rows.start);
        // unit = the row group the workgroup serves: the row (b * hq + h) * lq + i itself, or under a block
        // selection `(b * hkv + kv_head) * lq + i`, whose `n_rep` head lanes are the rows of the key/value
        // head's query heads. Under the split schedule gi = unit * splits + split.
        let (unit, split) = match splits {
            None => (gi, None),
            Some(splits) => (
                uc(c, BinOp::Div, gi, splits),
                Some(uc(c, BinOp::Rem, gi, splits)),
            ),
        };
        let i = uc(c, BinOp::Rem, unit, lq);
        let by_head = uc(c, BinOp::Div, unit, lq);
        let (b, h, row) = if let Some((head_lane, _)) = lane_split {
            let kv = uc(c, BinOp::Rem, by_head, hq / n_rep);
            let b = uc(c, BinOp::Div, by_head, hq / n_rep);
            let first_head = uc(c, BinOp::Mul, kv, n_rep);
            let h = ubin(c, BinOp::Add, first_head, head_lane);
            let by_batch = uc(c, BinOp::Mul, b, hq);
            let bh = ubin(c, BinOp::Add, by_batch, h);
            let by_head = uc(c, BinOp::Mul, bh, lq);
            (b, h, ubin(c, BinOp::Add, by_head, i))
        } else {
            let h = uc(c, BinOp::Rem, by_head, hq);
            (uc(c, BinOp::Div, by_head, hq), h, unit)
        };
        let kv_head = uc(c, BinOp::Div, h, n_rep);
        let qbase = uc(c, BinOp::Mul, row, dk);
        let kv_row = {
            let by_batch = uc(c, BinOp::Mul, b, hkv);
            ubin(c, BinOp::Add, by_batch, kv_head)
        };
        // A dense key's rows start at its `(batch, head)` plane; a paged key has no plane: its slot map starts
        // at the batch row and the pool row of key `t` comes from the slot it names.
        let rows = match read {
            None => KeyRows::Dense {
                k: uc(c, BinOp::Mul, kv_row, lk * head_dim),
                shared: uc(c, BinOp::Mul, b, lk * shared_dim),
                v: uc(c, BinOp::Mul, kv_row, lk * dv),
            },
            Some(read) => KeyRows::Paged {
                read,
                read_base: uc(c, BinOp::Mul, b, lk),
            },
        };
        // Where this row's additive mask starts; the other masks have no tensor to index.
        let maskbase = match mask_kind {
            MaskKind::Broadcast | MaskKind::PerHead => {
                let plane = match mask_kind {
                    MaskKind::PerHead => {
                        let by_batch = uc(c, BinOp::Mul, b, mask_heads);
                        ubin(c, BinOp::Add, by_batch, h)
                    }
                    _ => uc(c, BinOp::Mul, b, mask_heads),
                };
                let plane_start = uc(c, BinOp::Mul, plane, lq * lk);
                let row_start = uc(c, BinOp::Mul, i, lk);
                Some(ubin(c, BinOp::Add, plane_start, row_start))
            }
            MaskKind::Causal { .. } | MaskKind::Blocks { .. } => None,
        };
        let own = OSlots::new(c, dv, lane);
        // The query's index among the `B * Lq` of the selection operands.
        let qrow = {
            let by_batch = uc(c, BinOp::Mul, b, lq);
            ubin(c, BinOp::Add, by_batch, i)
        };

        // This lane's partial state: the running maximum and denominator in registers, `o` in LDS.
        let m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
        let l = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
        // The sink is the row's initial state, held by exactly one lane of one workgroup of the row.
        if let Some(sinks) = sinks {
            let holds_sink = {
                let first_lane = c.bin(Ty::Bool, BinOp::Eq, local_op(slice), cu(0));
                match split {
                    None => first_lane,
                    Some(split) => {
                        let first_split = c.bin(Ty::Bool, BinOp::Eq, local_op(split), cu(0));
                        c.both(first_lane, first_split)
                    }
                }
            };
            c.if_then(holds_sink, |c| {
                let logit = c.let_(Ty::F32, Rvalue::Use(copy(elem(sinks, h))));
                c.assign(Place::local(m), Rvalue::Use(local_op(logit)));
                c.assign(Place::local(l), Rvalue::Use(f32c(1.0)));
            });
        }
        count(c, cu(0), dv, 1, |c, channel| {
            let slot = own.at(c, channel);
            lds_write(c, O_ARRAY, slot, f32c(0.0));
        });

        // The keys this row visits: all `lk`, or the first `len[b]` clamped into `0..=lk`. A negative length
        // is a zero bound (the signed compare comes before the unsigned cast), a length past `lk` is `lk`.
        let key_end = match len {
            None => cu(spec.walk_units()),
            Some(len) => {
                let raw = c.let_(Ty::I32, Rvalue::Use(copy(elem(len, b))));
                let end = c.var(Ty::Usize, Rvalue::Use(cu(0)));
                let non_negative = c.bin(
                    Ty::Bool,
                    BinOp::Ge,
                    local_op(raw),
                    Operand::Const(Constant::I32(0)),
                );
                c.if_then(non_negative, |c| {
                    let widened = c.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: local_op(raw),
                        },
                    );
                    c.assign(Place::local(end), Rvalue::Use(local_op(widened)));
                });
                let past_end = c.bin(Ty::Bool, BinOp::Gt, local_op(end), cu(lk));
                c.if_then(past_end, |c| {
                    c.assign(Place::local(end), Rvalue::Use(cu(lk)));
                });
                local_op(end)
            }
        };
        // One workgroup's keys: all of them, or its split's contiguous chunk of the live range (empty when
        // the live length stops short of the chunk).
        let (first_key, key_end) = match split {
            None => (local_op(slice), key_end),
            Some(split) => {
                let begin = uc(c, BinOp::Mul, split, chunk);
                let end = c.var(Ty::Usize, Rvalue::Use(key_end));
                let chunk_end = uc(c, BinOp::Add, begin, chunk);
                let shorter = c.bin(Ty::Bool, BinOp::Lt, local_op(chunk_end), local_op(end));
                c.if_then(shorter, |c| {
                    c.assign(Place::local(end), Rvalue::Use(local_op(chunk_end)));
                });
                let first = ubin(c, BinOp::Add, begin, slice);
                (local_op(first), local_op(end))
            }
        };
        // A causal row walks only the keys it can see: it ends at its position and starts at its window's
        // first key, kept on this lane's stride.
        let (first_key, key_end) = match mask_kind {
            MaskKind::Causal { window } => {
                let raw = c.let_(Ty::I32, Rvalue::Use(copy(elem(mask, qrow))));
                // The position as an index; a negative one (a padded query) is 0.
                let position = c.var(Ty::Usize, Rvalue::Use(cu(0)));
                let non_negative = c.bin(
                    Ty::Bool,
                    BinOp::Ge,
                    local_op(raw),
                    Operand::Const(Constant::I32(0)),
                );
                c.if_then(non_negative, |c| {
                    let widened = c.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: local_op(raw),
                        },
                    );
                    c.assign(Place::local(position), Rvalue::Use(local_op(widened)));
                });
                let visible_end = uc(c, BinOp::Add, position, 1);
                let end = c.var(Ty::Usize, Rvalue::Use(key_end));
                let shorter = c.bin(Ty::Bool, BinOp::Lt, local_op(visible_end), local_op(end));
                c.if_then(shorter, |c| {
                    c.assign(Place::local(end), Rvalue::Use(local_op(visible_end)));
                });
                let first = c.var(Ty::Usize, Rvalue::Use(first_key));
                if let Some(window) = window {
                    // window_start = visible_end - window, or 0; move `first` up to the lane's first key at or
                    // after it: first + ceil((window_start - first) / slices) * slices.
                    let past_window = c.bin(Ty::Bool, BinOp::Gt, local_op(visible_end), cu(window));
                    c.if_then(past_window, |c| {
                        let window_start = uc(c, BinOp::Sub, visible_end, window);
                        let behind =
                            c.bin(Ty::Bool, BinOp::Lt, local_op(first), local_op(window_start));
                        c.if_then(behind, |c| {
                            let gap = ubin(c, BinOp::Sub, window_start, first);
                            let padded = uc(c, BinOp::Add, gap, slices - 1);
                            let steps = uc(c, BinOp::Div, padded, slices);
                            let jump = uc(c, BinOp::Mul, steps, slices);
                            let moved = ubin(c, BinOp::Add, first, jump);
                            c.assign(Place::local(first), Rvalue::Use(local_op(moved)));
                        });
                    });
                }
                (local_op(first), local_op(end))
            }
            _ => (first_key, key_end),
        };
        let visit = |c: &mut Cfg, t: poot_kernel_ir::Local| {
            // dot = q[row] . k[t], over the key's own part and then its shared part
            let dot = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
            let accumulate = |c: &mut Cfg,
                              part: poot_kernel_ir::Local,
                              row: poot_kernel_ir::Local,
                              width: usize,
                              q_offset: usize| {
                count(c, cu(0), width, 1, |c, channel| {
                    let qi = {
                        let at = ubin(c, BinOp::Add, qbase, channel);
                        uc(c, BinOp::Add, at, q_offset)
                    };
                    let ki = ubin(c, BinOp::Add, row, channel);
                    let qv = c.let_(Ty::F32, Rvalue::Use(copy(elem(q, qi))));
                    let kv_elem = c.emit(|e| kv.read(e, part, ki));
                    let term = fbin(c, BinOp::Mul, qv, kv_elem);
                    let sum = fbin(c, BinOp::Add, dot, term);
                    c.assign(Place::local(dot), Rvalue::Use(local_op(sum)));
                });
            };
            // The row of the key and of the value in their operands: dense rows follow the plane, a paged
            // key's operands share the pool row `slot * hkv + kv_head`.
            let (key_row, value_row) = match rows {
                KeyRows::Dense { k, v, .. } => {
                    let key = uc(c, BinOp::Mul, t, head_dim);
                    let value = uc(c, BinOp::Mul, t, dv);
                    (ubin(c, BinOp::Add, k, key), ubin(c, BinOp::Add, v, value))
                }
                KeyRows::Paged { read, read_base } => {
                    let at = ubin(c, BinOp::Add, read_base, t);
                    let word = c.let_(Ty::I32, Rvalue::Use(copy(elem(read, at))));
                    let slot = c.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: local_op(word),
                        },
                    );
                    let by_slot = uc(c, BinOp::Mul, slot, hkv);
                    let pool_row = ubin(c, BinOp::Add, by_slot, kv_head);
                    (
                        uc(c, BinOp::Mul, pool_row, head_dim),
                        uc(c, BinOp::Mul, pool_row, dv),
                    )
                }
            };
            accumulate(c, k, key_row, head_dim, 0);
            if let (Some(k_shared), KeyRows::Dense { shared, .. }) = (k_shared, rows) {
                let shared_row = {
                    let by_key = uc(c, BinOp::Mul, t, shared_dim);
                    ubin(c, BinOp::Add, shared, by_key)
                };
                accumulate(c, k_shared, shared_row, shared_dim, head_dim);
            }

            let s = score_of(c, score, dot);
            // A block-selected or causal row visits only keys it can see, so there is no mask to add.
            let s = match (mask_kind, maskbase) {
                (MaskKind::Broadcast | MaskKind::PerHead, Some(maskbase)) => {
                    let mi = ubin(c, BinOp::Add, maskbase, t);
                    let mask_value = c.let_(Ty::F32, Rvalue::Use(copy(elem(mask, mi))));
                    fbin(c, BinOp::Add, s, mask_value)
                }
                _ => s,
            };

            // The online-softmax step. `m_new == -inf` means every position seen so far, this one
            // included, is masked: the lane has contributed nothing, and `exp(-inf - -inf)` is NaN.
            let m_new = fbin(c, BinOp::Max, m, s);
            let corr = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
            let weight = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
            let seen = c.bin(
                Ty::Bool,
                BinOp::Ne,
                local_op(m_new),
                f32c(f32::NEG_INFINITY),
            );
            c.if_then(seen, |c| {
                let behind = fbin(c, BinOp::Sub, m, m_new);
                let rescale = exp(c, behind);
                c.assign(Place::local(corr), Rvalue::Use(local_op(rescale)));
                let below = fbin(c, BinOp::Sub, s, m_new);
                let own = exp(c, below);
                c.assign(Place::local(weight), Rvalue::Use(local_op(own)));
            });
            let decayed = fbin(c, BinOp::Mul, l, corr);
            let l_next = fbin(c, BinOp::Add, decayed, weight);
            c.assign(Place::local(l), Rvalue::Use(local_op(l_next)));

            count(c, cu(0), dv, 1, |c, channel| {
                let slot = own.at(c, channel);
                let vi = ubin(c, BinOp::Add, value_row, channel);
                let held = lds_read(c, O_ARRAY, slot);
                let rescaled = fbin(c, BinOp::Mul, held, corr);
                let vv = c.emit(|e| kv.read(e, v, vi));
                let added = fbin(c, BinOp::Mul, weight, vv);
                let sum = fbin(c, BinOp::Add, rescaled, added);
                lds_write(c, O_ARRAY, slot, local_op(sum));
            });
            c.assign(Place::local(m), Rvalue::Use(local_op(m_new)));
        };
        // One step of a block-selected row: `step` consecutive keys of a block, the first of which `head` has
        // checked. Their scores share each load of `q`, one running maximum and rescale cover all of them, and
        // `o` is read and written once per step instead of once per key. A key past the query or the stored
        // keys weighs zero and is read at position 0, so no address leaves the buffers.
        let visit_block = |c: &mut Cfg, head: &Candidate, step: usize| {
            let KeyRows::Dense {
                k: k_plane,
                v: v_plane,
                ..
            } = rows
            else {
                unreachable!("a block-selected key is dense")
            };
            let mut keys = Vec::with_capacity(step);
            for j in 0..step {
                let (key, valid) = if j == 0 {
                    (head.key, head.valid)
                } else {
                    let key = uc(c, BinOp::Add, head.key, j);
                    let causal = c.bin(Ty::Bool, BinOp::Le, local_op(key), local_op(head.position));
                    let stored = c.bin(Ty::Bool, BinOp::Lt, local_op(key), cu(lk));
                    let readable = c.both(causal, stored);
                    (key, c.both(head.valid, readable))
                };
                let safe = crate::block_top_k::select(c, Ty::Usize, valid, local_op(key), cu(0));
                let k_row = {
                    let at = uc(c, BinOp::Mul, safe, head_dim);
                    ubin(c, BinOp::Add, k_plane, at)
                };
                let v_row = {
                    let at = uc(c, BinOp::Mul, safe, dv);
                    ubin(c, BinOp::Add, v_plane, at)
                };
                keys.push((k_row, v_row, valid));
            }
            let dots: Vec<_> = (0..step)
                .map(|_| c.var(Ty::F32, Rvalue::Use(f32c(0.0))))
                .collect();
            count(c, cu(0), head_dim, 1, |c, channel| {
                let qi = ubin(c, BinOp::Add, qbase, channel);
                let qv = c.let_(Ty::F32, Rvalue::Use(copy(elem(q, qi))));
                for (&(k_row, _, _), &dot) in keys.iter().zip(&dots) {
                    let ki = ubin(c, BinOp::Add, k_row, channel);
                    let kv_elem = c.emit(|e| kv.read(e, k, ki));
                    let term = fbin(c, BinOp::Mul, qv, kv_elem);
                    let sum = fbin(c, BinOp::Add, dot, term);
                    c.assign(Place::local(dot), Rvalue::Use(local_op(sum)));
                }
            });
            let scores: Vec<_> = keys
                .iter()
                .zip(&dots)
                .map(|(&(_, _, valid), &dot)| {
                    let s = score_of(c, score, dot);
                    crate::block_top_k::select(
                        c,
                        Ty::F32,
                        valid,
                        local_op(s),
                        f32c(f32::NEG_INFINITY),
                    )
                })
                .collect();
            // The first key is read, so the step's maximum is finite.
            let step_max = scores[1..]
                .iter()
                .fold(scores[0], |max, &s| fbin(c, BinOp::Max, max, s));
            let m_new = fbin(c, BinOp::Max, m, step_max);
            let behind = fbin(c, BinOp::Sub, m, m_new);
            let corr = exp(c, behind);
            let weights: Vec<_> = scores
                .iter()
                .map(|&s| {
                    let below = fbin(c, BinOp::Sub, s, m_new);
                    exp(c, below)
                })
                .collect();
            let decayed = fbin(c, BinOp::Mul, l, corr);
            let l_next = weights
                .iter()
                .fold(decayed, |sum, &w| fbin(c, BinOp::Add, sum, w));
            c.assign(Place::local(l), Rvalue::Use(local_op(l_next)));
            count(c, cu(0), dv, 1, |c, channel| {
                let slot = own.at(c, channel);
                let held = lds_read(c, O_ARRAY, slot);
                let rescaled = fbin(c, BinOp::Mul, held, corr);
                let sum =
                    keys.iter()
                        .zip(&weights)
                        .fold(rescaled, |sum, (&(_, v_row, _), &weight)| {
                            let vi = ubin(c, BinOp::Add, v_row, channel);
                            let vv = c.emit(|e| kv.read(e, v, vi));
                            let added = fbin(c, BinOp::Mul, weight, vv);
                            fbin(c, BinOp::Add, sum, added)
                        });
                lds_write(c, O_ARRAY, slot, local_op(sum));
            });
            c.assign(Place::local(m), Rvalue::Use(local_op(m_new)));
        };
        match mask_kind {
            MaskKind::Blocks { ratio, k, grouped } => {
                let ids = poot_kernel_ir::Local { index: 4 };
                let pos = poot_kernel_ir::Local { index: 5 };
                let step = block_step(ratio, grouped);
                count_to(c, first_key, key_end, slices, |c, unit| {
                    let candidate = uc(c, BinOp::Mul, unit, step);
                    let head = block_candidate(c, ids, pos, qrow, (k, ratio, lk), candidate);
                    c.if_then(head.valid, |c| visit_block(c, &head, step));
                });
            }
            MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Causal { .. } => {
                count_to(c, first_key, key_end, slices, visit);
            }
        }

        // Publish this lane's `(m, l)`, then the one barrier: every lane reaches it exactly once.
        lds_write(c, M_ARRAY, lane, local_op(m));
        lds_write(c, L_ARRAY, lane, local_op(l));
        c.barrier();

        // The combine of this lane's head over its `slices` lanes. `big_m` is the maximum of their maxima; a
        // lane whose maximum is `-inf` saw no unmasked key and weighs zero, so no `-inf - -inf` is formed.
        let same_head = |c: &mut Cfg, other_slice: poot_kernel_ir::Local| match lane_split {
            None => other_slice,
            Some((head_lane, _)) => {
                let start = uc(c, BinOp::Mul, other_slice, group);
                ubin(c, BinOp::Add, start, head_lane)
            }
        };
        let big_m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
        count(c, cu(0), slices, 1, |c, other_slice| {
            let other = same_head(c, other_slice);
            let m_other = lds_read(c, M_ARRAY, other);
            let next = fbin(c, BinOp::Max, big_m, m_other);
            c.assign(Place::local(big_m), Rvalue::Use(local_op(next)));
        });
        let denominator = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
        count(c, cu(0), slices, 1, |c, other_slice| {
            let other = same_head(c, other_slice);
            let w = lane_weight(c, other, big_m);
            let l_other = lds_read(c, L_ARRAY, other);
            let part = fbin(c, BinOp::Mul, l_other, w);
            let next = fbin(c, BinOp::Add, denominator, part);
            c.assign(Place::local(denominator), Rvalue::Use(local_op(next)));
        });
        // The streaming schedule writes the normalized row; a split writes its unnormalized state.
        let out_base = match split {
            None => uc(c, BinOp::Mul, row, dv),
            Some(split) => {
                let partial = {
                    let first = uc(
                        c,
                        BinOp::Mul,
                        row,
                        splits.expect("a split schedule has splits"),
                    );
                    ubin(c, BinOp::Add, first, split)
                };
                uc(c, BinOp::Mul, partial, dv + 2)
            }
        };
        count(c, local_op(slice), dv, slices, |c, channel| {
            let acc = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
            count(c, cu(0), slices, 1, |c, other_slice| {
                let other = same_head(c, other_slice);
                let w = lane_weight(c, other, big_m);
                let slot = OSlots::new(c, dv, other).at(c, channel);
                let partial = lds_read(c, O_ARRAY, slot);
                let part = fbin(c, BinOp::Mul, partial, w);
                let next = fbin(c, BinOp::Add, acc, part);
                c.assign(Place::local(acc), Rvalue::Use(local_op(next)));
            });
            let result = match split {
                None => fbin(c, BinOp::Div, acc, denominator),
                Some(_) => acc,
            };
            let oi = ubin(c, BinOp::Add, out_base, channel);
            c.assign(elem(out, oi), Rvalue::Use(local_op(result)));
        });
        if split.is_some() {
            let first_lane = c.bin(Ty::Bool, BinOp::Eq, local_op(slice), cu(0));
            c.if_then(first_lane, |c| {
                let m_at = uc(c, BinOp::Add, out_base, dv);
                c.assign(elem(out, m_at), Rvalue::Use(local_op(big_m)));
                let l_at = uc(c, BinOp::Add, out_base, dv + 1);
                c.assign(elem(out, l_at), Rvalue::Use(local_op(denominator)));
            });
        }
    });

    let mut body = c.finish(name);
    body.workgroup_size = [lanes as u32, 1, 1];
    body.workgroup_locals = vec![f32_array(lanes * dv), f32_array(lanes), f32_array(lanes)];
    Ok(body)
}

/// The combine stage of split-KV attention (see [`AttentionCombineSpec`]). The launch is one workgroup of
/// `lanes` threads per row over the same 2-D `x_groups` fold as [`attention_body`]. Parameters: `partials`,
/// then `out`. A lane reads each split's `(m, l)` once, then every channel of its strided share; a split
/// whose maximum is `-inf` saw no unmasked key and weighs zero, so no `-inf - -inf` is formed.
pub(crate) fn attention_combine_body(
    name: &str,
    spec: &AttentionCombineSpec,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    for (what, value) in [
        ("rows", spec.rows),
        ("splits", spec.splits),
        ("dv", spec.dv),
        ("lanes", spec.lanes),
    ] {
        if value == 0 {
            return Err(bad(what, value, 1));
        }
    }
    let AttentionCombineSpec {
        rows,
        splits,
        dv,
        lanes,
    } = *spec;
    let params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 partials
        ld(slice_f32(true), true),   // 2 out
    ];
    let (partials, out) = (
        poot_kernel_ir::Local { index: 1 },
        poot_kernel_ir::Local { index: 2 },
    );
    let mut c = Cfg::new(params);
    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let lane = c.index(IndexAxis::LocalX);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let row = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(row), cu(rows));
    c.if_then(in_range, |c| {
        let row_base = uc(c, BinOp::Mul, row, splits * (dv + 2));
        // `(m, l)` of split `s` sit at `row_base + s * (dv + 2) + dv` and the element after it.
        let split_at = |c: &mut Cfg, s: poot_kernel_ir::Local, extra: usize| {
            let start = uc(c, BinOp::Mul, s, dv + 2);
            let at = ubin(c, BinOp::Add, row_base, start);
            uc(c, BinOp::Add, at, extra)
        };
        let big_m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
        count(c, cu(0), splits, 1, |c, s| {
            let at = split_at(c, s, dv);
            let m_s = c.let_(Ty::F32, Rvalue::Use(copy(elem(partials, at))));
            let next = fbin(c, BinOp::Max, big_m, m_s);
            c.assign(Place::local(big_m), Rvalue::Use(local_op(next)));
        });
        let denominator = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
        count(c, cu(0), splits, 1, |c, s| {
            let m_at = split_at(c, s, dv);
            let w = split_weight(c, partials, m_at, big_m);
            let l_at = split_at(c, s, dv + 1);
            let l_s = c.let_(Ty::F32, Rvalue::Use(copy(elem(partials, l_at))));
            let part = fbin(c, BinOp::Mul, l_s, w);
            let next = fbin(c, BinOp::Add, denominator, part);
            c.assign(Place::local(denominator), Rvalue::Use(local_op(next)));
        });
        let out_base = uc(c, BinOp::Mul, row, dv);
        count(c, local_op(lane), dv, lanes, |c, channel| {
            let acc = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
            count(c, cu(0), splits, 1, |c, s| {
                let m_at = split_at(c, s, dv);
                let w = split_weight(c, partials, m_at, big_m);
                let o_at = {
                    let start = split_at(c, s, 0);
                    ubin(c, BinOp::Add, start, channel)
                };
                let o_s = c.let_(Ty::F32, Rvalue::Use(copy(elem(partials, o_at))));
                let part = fbin(c, BinOp::Mul, o_s, w);
                let next = fbin(c, BinOp::Add, acc, part);
                c.assign(Place::local(acc), Rvalue::Use(local_op(next)));
            });
            let result = fbin(c, BinOp::Div, acc, denominator);
            let oi = ubin(c, BinOp::Add, out_base, channel);
            c.assign(elem(out, oi), Rvalue::Use(local_op(result)));
        });
    });
    let mut body = c.finish(name);
    body.workgroup_size = [lanes as u32, 1, 1];
    Ok(body)
}

/// `exp(m_s - big_m)` for the split whose maximum is at `partials[m_at]`, or zero for a split that saw no
/// unmasked key.
fn split_weight(
    c: &mut Cfg,
    partials: poot_kernel_ir::Local,
    m_at: poot_kernel_ir::Local,
    big_m: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let weight = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    let m_s = c.let_(Ty::F32, Rvalue::Use(copy(elem(partials, m_at))));
    let seen = c.bin(Ty::Bool, BinOp::Ne, local_op(m_s), f32c(f32::NEG_INFINITY));
    c.if_then(seen, |c| {
        let behind = fbin(c, BinOp::Sub, m_s, big_m);
        let w = exp(c, behind);
        c.assign(Place::local(weight), Rvalue::Use(local_op(w)));
    });
    weight
}

/// `exp(m_other - big_m)`, or zero for a lane that saw no unmasked key.
fn lane_weight(
    c: &mut Cfg,
    other: poot_kernel_ir::Local,
    big_m: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let weight = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    let m_other = lds_read(c, M_ARRAY, other);
    let seen = c.bin(
        Ty::Bool,
        BinOp::Ne,
        local_op(m_other),
        f32c(f32::NEG_INFINITY),
    );
    c.if_then(seen, |c| {
        let behind = fbin(c, BinOp::Sub, m_other, big_m);
        let w = exp(c, behind);
        c.assign(Place::local(weight), Rvalue::Use(local_op(w)));
    });
    weight
}

/// The score a raw dot product becomes: `dot * scale`, then the softcap `cap * tanh(. / cap)` when asked.
/// The softcap multiplies by the reciprocal `1 / cap`, as `ops::softcap` does, so the bits agree.
fn score_of(
    c: &mut Cfg,
    score: ScoreTransform,
    dot: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    c.emit(|e| emit_score(e, score, dot))
}

/// [`score_of`] into straight-line statements.
fn emit_score(
    e: &mut crate::emit::Emit,
    score: ScoreTransform,
    dot: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let (scale_bits, softcap) = match score {
        ScoreTransform::Scale { scale_bits } => (scale_bits, None),
        ScoreTransform::ScaleSoftcap {
            scale_bits,
            cap_bits,
        } => (scale_bits, Some(f32::from_bits(cap_bits))),
    };
    let scaled = crate::emit::bin(
        e,
        Ty::F32,
        BinOp::Mul,
        local_op(dot),
        f32c(f32::from_bits(scale_bits)),
    );
    let Some(cap) = softcap else {
        return scaled;
    };
    let unit = crate::emit::bin(e, Ty::F32, BinOp::Mul, local_op(scaled), f32c(1.0 / cap));
    let tanh = e.let_(Ty::F32, Rvalue::Use(f32c(0.0)));
    let (al, stmts) = e.parts();
    emit_scalar_op(tanh, FusedScalarOp::Tanh, &[local_op(unit)], al, stmts);
    crate::emit::bin(e, Ty::F32, BinOp::Mul, local_op(tanh), f32c(cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AttentionSpec {
        AttentionSpec {
            kv: crate::KvLane::F32,
            batch: 1,
            hq: 4,
            n_rep: 2,
            lq: 1,
            lk: 8,
            dk: 4,
            dv: 4,
            key: KeyLayout::Single,
            score: ScoreTransform::Scale {
                scale_bits: 0.5f32.to_bits(),
            },
            mask: MaskKind::Broadcast,
            bound: KeyBound::Static,
            sink: AttentionSink::Absent,
            schedule: AttentionSchedule::Streaming { lanes: 4 },
        }
    }

    /// The generator's own preconditions are typed errors: a head count the grouping does not divide, a zero
    /// extent and a split key whose shared part is empty or the whole key, each naming the generator.
    /// A value width that differs from the key width is not one of them.
    #[test]
    fn a_spec_outside_the_generators_contract_is_a_typed_error() {
        let wider_values = AttentionSpec { dv: 8, ..spec() };
        assert!(attention_body("a", &wider_values, 0..wider_values.workgroups(), 1).is_ok());
        let ragged = AttentionSpec { n_rep: 3, ..spec() };
        assert!(matches!(
            attention_body("a", &ragged, 0..ragged.workgroups(), 1),
            Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                divisor: 3,
                ..
            })
        ));
        let no_lanes = AttentionSpec {
            kv: crate::KvLane::F32,
            schedule: AttentionSchedule::Streaming { lanes: 0 },
            ..spec()
        };
        assert!(matches!(
            attention_body("a", &no_lanes, 0..no_lanes.workgroups(), 1),
            Err(KernelGenError::BelowMinimum {
                generator: GENERATOR,
                ..
            })
        ));
        let empty_shared = AttentionSpec {
            kv: crate::KvLane::F32,
            key: KeyLayout::Split { shared_dim: 0 },
            ..spec()
        };
        assert!(matches!(
            attention_body("a", &empty_shared, 0..empty_shared.workgroups(), 1),
            Err(KernelGenError::BelowMinimum {
                generator: GENERATOR,
                ..
            })
        ));
        let all_shared = AttentionSpec {
            kv: crate::KvLane::F32,
            key: KeyLayout::Split { shared_dim: 4 },
            ..spec()
        };
        assert!(matches!(
            attention_body("a", &all_shared, 0..all_shared.workgroups(), 1),
            Err(KernelGenError::ExceedsBound {
                generator: GENERATOR,
                value: 4,
                bound: 3,
                ..
            })
        ));
    }
}

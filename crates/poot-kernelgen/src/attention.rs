//! The attention generator: one online-softmax body for every phase.
//!
//! An [`AttentionSpec`] describes the region (score transform, mask kind, head grouping, bound) and a
//! schedule; [`attention_body`] turns it into one kernel. `lq` and `lk` are independent: decode (`lq == 1`
//! against a capacity-long cache), prefill (`lq == lk`) and prefill into a cache (`1 < lq < lk`) differ only in
//! the extents they pass.
//!
//! The [`AttentionSchedule::Streaming`] schedule is the clustered key walk (see [`clustered`]): one workgroup
//! of `lanes` threads per `(batch, key/value head, query row)`, serving all `n_rep` query heads of the key/value
//! head. The lanes form clusters of [`KeyWalk::cluster`] lanes and a cluster walks one key per step, each lane
//! holding a contiguous run of the key and value channels. Every lane keeps the heads' queries, running maxima
//! `m`, denominators `l` and weighted value sums `o` in registers: admitting a key at score `s` moves the maximum
//! to `m' = max(m, s)` and rescales what the lane holds by `exp(m - m')` before adding the key's own term, so the
//! `[lq, lk]` scores are never materialized. A dot product is a per-lane sum over the run and a subgroup xor
//! butterfly over the cluster. One barrier, on the straight-line path after the key loops, separates the walk
//! from the combine of the clusters' states in workgroup memory: with `M` the maximum of the cluster maxima,
//! `out[c] = sum_i o_i[c] * exp(m_i - M) / sum_i l_i * exp(m_i - M)`.
//!
//! [`AttentionSchedule::SplitKv`] is the same row group split over the keys across `splits` workgroups (decode,
//! where `batch * kv heads` workgroups leave a many-CU device idle). Stage one is this body with each
//! workgroup visiting only its own contiguous share of the keys and publishing the combine's inputs
//! instead of dividing: `o[dv]` unnormalized, then `m` and `l`, at `[(row * splits + split) * (dv + 2)]` of
//! the scratch it writes, for each of the group's rows. The grid is fixed by `splits`, but the shares are not: a
//! workgroup first works out the range `[lo, hi)` its row visits (live length, causal position, window), then
//! `active`, the number of shares that range is cut into (at most `splits`, each at least `min_units` steps,
//! balanced to whole cluster steps), and walks share `split` of it. A workgroup with `split >= active` returns
//! and writes nothing; split 0 publishes `active` in each row's word after the partials. Nothing in the
//! partition depends on the capacity `lk`, so a cache of any capacity partitions a live prefix identically.
//! Stage two ([`AttentionCombineSpec`]) is the `lanes`-wide combine over the `active` splits the row
//! published, so the scratch needs no initialization.
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
    BinOp, Body, COLLECTIVE_LANES, Constant, IndexAxis, LocalDecl, MathOp, Operand, Place, Rvalue,
    Ty, WorkgroupLocalDecl,
};

use crate::KernelGenError;
use crate::data_movement::Cfg;
use crate::emit::{copy as copy_local, cu};
use crate::fused::{FusedScalarOp, emit_scalar_op};
use crate::helpers::{copy, elem, ld, slice_dtype, slice_f32};
use crate::precision::require_exact;
use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_target::RoundingConversions;

mod clustered;
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
    /// How Q, K, V and the probabilities P enter the two products: every schedule rounds them as this says, P
    /// where it enters `P V`.
    pub precision: OperandPrecision,
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
    /// Bytes one element takes in its buffer.
    pub const fn bytes(self) -> usize {
        match self {
            KvLane::F32 => 4,
            KvLane::F16 | KvLane::Bf16 => 2,
        }
    }

    /// The fragment dtype this lane stores its elements in, `None` for f32: a stored element is exact in it.
    pub(crate) const fn fragment(self) -> Option<FragmentDtype> {
        match self {
            KvLane::F32 => None,
            KvLane::F16 => Some(FragmentDtype::F16),
            KvLane::Bf16 => Some(FragmentDtype::Bf16),
        }
    }

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

    /// Query heads one workgroup serves: the `n_rep` heads of a key/value head, which read the same keys and
    /// values, so a workgroup loads each key and value once for all of them. Every schedule groups the heads
    /// (the clustered walk holds them in registers, the tiled schedule packs them into its tile rows).
    pub fn heads_per_workgroup(&self) -> usize {
        self.n_rep
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

    /// Row groups: the rows one workgroup serves, `rows() / heads_per_workgroup()` (one per `(batch, key/value
    /// head, query row)`), or under the tiled schedule one per `(batch, key/value head, block of br query
    /// positions)`.
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

    /// The key positions a row considers (the steps its walk takes, a key each): all `lk`, or under
    /// [`MaskKind::Blocks`] the `(K + 1) * ratio` of its selected blocks and its unfinished block. The split
    /// schedule's grid is bounded by these, and its `min_units` floor counts them.
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

    /// Whether the schedule multiplies on matrix fragments, which need their subgroup of lanes ([`AttentionMicro::Fragment`]).
    pub fn uses_fragments(&self) -> bool {
        matches!(
            self.schedule,
            AttentionSchedule::Tiled {
                micro: AttentionMicro::Fragment { .. },
                ..
            }
        )
    }

    /// The `f32` elements the split schedule's partial state occupies: `(o[dv], m, l)` per `(row, split)`,
    /// then one word per row holding the number of splits that row's stage one made partials for (an exact
    /// small integer in f32). `None` for the streaming and tiled schedules, which have no intermediate.
    pub fn partial_elems(&self) -> Option<usize> {
        match self.schedule {
            AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => None,
            AttentionSchedule::SplitKv { splits, .. } => {
                Some(self.rows() * (splits * (self.dv + 2) + 1))
            }
        }
    }
}

/// A contiguous run of a streaming or tiled attention's row groups ([`AttentionSpec::row_groups`]: its
/// key/value heads' rows, under the tiled schedule its blocks of rows), launched as one dispatch of its own:
/// `count` row groups of `spec` starting at `first`. A prefill whose whole grid would run past the device's watchdog is several
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
            AttentionSchedule::SplitKv { lanes, splits, .. } => Some(Self {
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
    /// ratio` candidates in all, instead of the `lk` positions; the scores need no mask. The `n_rep` query
    /// heads of a key/value head select the same blocks (the ids are per query, not per head), so the
    /// workgroup loads each candidate once for all of them. Dense keys and the static bound only; the split
    /// schedule divides the candidates.
    Blocks {
        /// Positions per block.
        ratio: usize,
        /// Selected blocks per query: the width of `ids`.
        k: usize,
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

/// How the lanes of a clustered key walk ([`AttentionSchedule::Streaming`], [`AttentionSchedule::SplitKv`]) divide
/// the head dimension: a mechanism the planner chooses from the device, not a policy of the generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyWalk {
    /// Lanes of one cluster, `G`: a power of two, at most [`poot_kernel_ir::COLLECTIVE_LANES`]. A cluster walks
    /// one key per step, and its dot product is the butterfly over these lanes.
    pub cluster: usize,
    /// Channels each lane of a cluster owns, of the key and of the value: lane `j` reads and accumulates the
    /// contiguous channels `[j * run, (j + 1) * run)`, so a cluster reads one contiguous key row. The planner
    /// takes the widest run the device loads at once (16 bytes). It divides `dk` and `dv` (and a split key's
    /// parts), and `ceil(dk / run)` and `ceil(dv / run)` are at most `cluster`: a lane past a head's last run
    /// holds zeros.
    pub run: usize,
}

/// How the region's keys are walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionSchedule {
    /// The clustered key walk: one workgroup of `lanes` threads per `(batch, key/value head, query row)`,
    /// serving the head's `n_rep` query heads, the lanes in clusters of `walk.cluster` each walking a key per
    /// step (see [`clustered`](self::clustered) for the body).
    Streaming {
        /// Threads per workgroup: a multiple of 32, so a cluster never straddles an aligned 32-lane group.
        lanes: usize,
        /// How the lanes divide the head dimension.
        walk: KeyWalk,
    },
    /// Stage one of split-KV attention: one workgroup of `lanes` threads per `(batch, key/value head, query
    /// row, split)`, each walking its own contiguous share of the row's visited keys and writing the partial
    /// state of each of the group's heads to a scratch instead of the output. The shares are cut from the keys
    /// the row visits at run time (its live length, causal position and window), never from the cache capacity;
    /// `splits` only bounds the grid. [`AttentionCombineSpec`] reduces the partials.
    SplitKv {
        /// Threads per workgroup: a multiple of 32.
        lanes: usize,
        /// Workgroups of one row group: the most shares its visited keys are divided into.
        splits: usize,
        /// The fewest walk steps (keys, [`AttentionSpec::candidates`]) a share takes: a row with fewer visited
        /// steps than `splits * min_units` uses fewer shares, and the workgroups past them return.
        min_units: usize,
        /// How the lanes divide the head dimension.
        walk: KeyWalk,
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
    /// The workgroup width.
    pub fn lanes(self) -> usize {
        match self {
            Self::Streaming { lanes, .. }
            | Self::SplitKv { lanes, .. }
            | Self::Tiled { lanes, .. } => lanes,
        }
    }

    /// How the lanes divide the head dimension, for the schedules that walk keys in clusters.
    pub fn walk(self) -> Option<KeyWalk> {
        match self {
            Self::Streaming { walk, .. } | Self::SplitKv { walk, .. } => Some(walk),
            Self::Tiled { .. } => None,
        }
    }
}

/// The micro-tiles of a [`AttentionSchedule::Tiled`] workgroup: one for the score tile of a key block, one for
/// the output accumulator (see [`crate::microtile`]). Both products use the same kind of arithmetic unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttentionMicro {
    /// Each lane accumulates a register tile of f32 products. The lanes of a tile form a grid of `rows /
    /// tile.rows` by `cols / tile.cols` lanes, which must be the workgroup's `lanes`.
    Scalar {
        /// Each lane's part of the `(br * n_rep) x bc` score tile.
        score: LaneTile,
        /// Each lane's part of the `(br * n_rep) x dv` output accumulator.
        out: LaneTile,
    },
    /// Each wave of 32 lanes accumulates matrix fragments of 16x16 outputs from `dtype` operands staged in
    /// workgroup memory (POOT-1437). The waves of a tile form a `waves_m x waves_n` grid, and both grids hold
    /// the workgroup's `lanes / 32` waves. The schedule's operand precision must narrow to `dtype`.
    Fragment {
        /// The operand dtype of Q, K, P and V; the accumulators are f32.
        dtype: poot_target::FragmentDtype,
        /// Whether the accumulators' elements are addressable in place.
        layout: poot_target::FragmentLayout,
        /// The waves and fragments of the `(br * n_rep) x bc` score tile.
        score: WaveTile,
        /// The waves and fragments of the `(br * n_rep) x dv` output accumulator.
        out: WaveTile,
        /// Where the score product's Q operand lives during the key loop.
        q: QOperand,
    },
}

/// Where a fragment attention keeps the score product's Q operand during the key loop (POOT-1447).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QOperand {
    /// A tile in workgroup memory, staged once; every key block loads each wave's A fragments from it again.
    Staged,
    /// Each wave's A fragments, loaded once before the key loop and held in registers across it
    /// (`score.rows x dk / 16` fragments). Q takes no workgroup memory during the loop: it is staged through the
    /// key/value array in depth chunks before it.
    Held,
}

/// The wave grid over one product's tile and the fragments each wave accumulates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WaveTile {
    /// Waves along the tile's rows.
    pub waves_m: usize,
    /// Waves along the tile's columns.
    pub waves_n: usize,
    /// Fragments each wave accumulates along the rows.
    pub rows: u32,
    /// Fragments each wave accumulates along the columns.
    pub cols: u32,
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
fn validate(spec: &AttentionSpec) -> Result<(), KernelGenError> {
    let lanes = spec.schedule.lanes();
    let (splits, min_units) = match spec.schedule {
        AttentionSchedule::Streaming { .. } | AttentionSchedule::Tiled { .. } => (1, 1),
        AttentionSchedule::SplitKv {
            splits, min_units, ..
        } => (splits, min_units),
    };
    for (what, value) in [
        ("splits", splits),
        ("min_units", min_units),
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
    // More splits than candidate keys is a grid of workgroups that can never have a key to walk.
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
    if let Some(walk) = spec.schedule.walk() {
        validate_walk(spec, walk)?;
    }
    if let MaskKind::Blocks { ratio, k } = spec.mask {
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
    Ok(())
}

/// Check a clustered walk's geometry: a cluster is a power of two of at most [`COLLECTIVE_LANES`] lanes (its
/// butterfly masks stay inside one aligned run of that many lanes), the workgroup is whole such runs, and the
/// runs tile the heads: the run width divides both head widths, and a split key's two parts (so no lane's run
/// straddles them), and the lanes of a cluster cover each head.
fn validate_walk(spec: &AttentionSpec, walk: KeyWalk) -> Result<(), KernelGenError> {
    let KeyWalk { cluster, run } = walk;
    let lanes = spec.schedule.lanes();
    if cluster == 0 || !cluster.is_power_of_two() {
        return Err(KernelGenError::Unsupported {
            request: "attention",
            reason: "a key-walk cluster is a power of two of lanes",
        });
    }
    if cluster > COLLECTIVE_LANES as usize {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "cluster".to_string(),
            value: cluster,
            bound: COLLECTIVE_LANES as usize,
        });
    }
    if run == 0 {
        return Err(bad("run", run, 1));
    }
    if !lanes.is_multiple_of(COLLECTIVE_LANES as usize) {
        return Err(KernelGenError::NotDivisible {
            generator: GENERATOR,
            dim: "lanes".to_string(),
            value: lanes,
            divisor: COLLECTIVE_LANES as usize,
        });
    }
    let mut widths = vec![("dk", spec.dk), ("dv", spec.dv)];
    if let KeyLayout::Split { shared_dim } = spec.key {
        widths.push(("dk - shared_dim", spec.dk - shared_dim));
    }
    for (dim, width) in widths {
        if !width.is_multiple_of(run) {
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: dim.to_string(),
                value: width,
                divisor: run,
            });
        }
    }
    for (what, width) in [("dk", spec.dk), ("dv", spec.dv)] {
        if width / run > cluster {
            return Err(KernelGenError::ExceedsBound {
                generator: GENERATOR,
                what: format!("{what} / run"),
                value: width / run,
                bound: cluster,
            });
        }
    }
    Ok(())
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

/// Build the attention kernel `spec` describes over its workgroups `rows` (all of them, except for a streaming
/// or tiled schedule split into runs of row groups). The launch is `rows.len()` workgroups laid over a 2-D grid
/// `x_groups` workgroups across: `GroupY * x_groups + GroupX` is the workgroup's place in the run. The streaming
/// and split-KV schedules take their body from [`clustered`], the tiled schedule from [`tiled`].
pub(crate) fn attention_body(
    name: &str,
    spec: &AttentionSpec,
    rows: std::ops::Range<usize>,
    x_groups: usize,
    conversions: RoundingConversions,
) -> Result<Body, KernelGenError> {
    validate(spec)?;
    validate_rows(spec, &rows)?;
    if let AttentionSchedule::Tiled { .. } = spec.schedule {
        return tiled::tiled_body(name, spec, rows, x_groups, conversions);
    }
    // The clustered body multiplies the operands as stored: only a matrix unit narrows them.
    require_exact(spec.precision, GENERATOR)?;
    clustered::clustered_body(name, spec, rows, x_groups)
}

/// The combine stage of split-KV attention (see [`AttentionCombineSpec`]). The launch is one workgroup of
/// `lanes` threads per row over the same 2-D `x_groups` fold as [`attention_body`]. Parameters: `partials`,
/// then `out`. A row reduces only the `active` splits its stage one published (the word after the partials of
/// all rows); a split whose maximum is `-inf` saw no unmasked key and weighs zero, so no `-inf - -inf` is
/// formed.
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
        // The splits stage one made partials for; the others hold whatever the scratch held before.
        let active = {
            let count_at = uc(c, BinOp::Add, row, rows * splits * (dv + 2));
            let published = c.let_(Ty::F32, Rvalue::Use(copy(elem(partials, count_at))));
            c.let_(
                Ty::Usize,
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: local_op(published),
                },
            )
        };
        let big_m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
        count_to(c, cu(0), local_op(active), 1, |c, s| {
            let at = split_at(c, s, dv);
            let m_s = c.let_(Ty::F32, Rvalue::Use(copy(elem(partials, at))));
            let next = fbin(c, BinOp::Max, big_m, m_s);
            c.assign(Place::local(big_m), Rvalue::Use(local_op(next)));
        });
        let denominator = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
        count_to(c, cu(0), local_op(active), 1, |c, s| {
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
            count_to(c, cu(0), local_op(active), 1, |c, s| {
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
    // `Tanh` rounds nothing, so the conversions are moot.
    emit_scalar_op(
        tanh,
        FusedScalarOp::Tanh,
        &[local_op(unit)],
        RoundingConversions::NONE,
        al,
        stmts,
    );
    crate::emit::bin(e, Ty::F32, BinOp::Mul, local_op(tanh), f32c(cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AttentionSpec {
        AttentionSpec {
            kv: crate::KvLane::F32,
            precision: OperandPrecision::Exact,
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
            schedule: AttentionSchedule::Streaming {
                lanes: 32,
                walk: KeyWalk { cluster: 4, run: 1 },
            },
        }
    }

    /// The generator's own preconditions are typed errors: a head count the grouping does not divide, a zero
    /// extent and a split key whose shared part is empty or the whole key, each naming the generator.
    /// A value width that differs from the key width is not one of them.
    #[test]
    fn a_spec_outside_the_generators_contract_is_a_typed_error() {
        let wider_values = AttentionSpec {
            dv: 8,
            schedule: AttentionSchedule::Streaming {
                lanes: 32,
                walk: KeyWalk { cluster: 8, run: 1 },
            },
            ..spec()
        };
        assert!(
            attention_body(
                "a",
                &wider_values,
                0..wider_values.workgroups(),
                1,
                RoundingConversions::NONE
            )
            .is_ok()
        );
        let ragged = AttentionSpec { n_rep: 3, ..spec() };
        assert!(matches!(
            attention_body(
                "a",
                &ragged,
                0..ragged.workgroups(),
                1,
                RoundingConversions::NONE
            ),
            Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                divisor: 3,
                ..
            })
        ));
        let no_lanes = AttentionSpec {
            kv: crate::KvLane::F32,
            schedule: AttentionSchedule::Streaming {
                lanes: 0,
                walk: KeyWalk { cluster: 4, run: 1 },
            },
            ..spec()
        };
        assert!(matches!(
            attention_body(
                "a",
                &no_lanes,
                0..no_lanes.workgroups(),
                1,
                RoundingConversions::NONE
            ),
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
            attention_body(
                "a",
                &empty_shared,
                0..empty_shared.workgroups(),
                1,
                RoundingConversions::NONE
            ),
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
            attention_body(
                "a",
                &all_shared,
                0..all_shared.workgroups(),
                1,
                RoundingConversions::NONE
            ),
            Err(KernelGenError::ExceedsBound {
                generator: GENERATOR,
                value: 4,
                bound: 3,
                ..
            })
        ));
    }

    /// A clustered walk's geometry is checked: a cluster that is not a power of two or is wider than the 32 lanes a
    /// butterfly stays inside, a workgroup that is not whole runs of 32 lanes, a run that does not divide the head
    /// (or a split key's parts), and a cluster too narrow to cover a head are each a typed error.
    #[test]
    fn a_walk_outside_the_clusters_contract_is_a_typed_error() {
        let walked = |lanes, cluster, run| AttentionSpec {
            schedule: AttentionSchedule::Streaming {
                lanes,
                walk: KeyWalk { cluster, run },
            },
            ..spec()
        };
        let generate = |spec: &AttentionSpec| {
            attention_body(
                "a",
                spec,
                0..spec.workgroups(),
                1,
                RoundingConversions::NONE,
            )
        };
        assert!(generate(&walked(32, 4, 1)).is_ok());
        assert!(matches!(
            generate(&walked(32, 3, 1)),
            Err(KernelGenError::Unsupported { .. })
        ));
        assert!(matches!(
            generate(&walked(128, 64, 1)),
            Err(KernelGenError::ExceedsBound {
                what,
                value: 64,
                bound: 32,
                ..
            }) if what == "cluster"
        ));
        assert!(matches!(
            generate(&walked(48, 4, 1)),
            Err(KernelGenError::NotDivisible {
                dim,
                value: 48,
                divisor: 32,
                ..
            }) if dim == "lanes"
        ));
        assert!(matches!(
            generate(&walked(32, 4, 3)),
            Err(KernelGenError::NotDivisible {
                value: 4,
                divisor: 3,
                ..
            })
        ));
        assert!(matches!(
            generate(&walked(32, 2, 1)),
            Err(KernelGenError::ExceedsBound {
                value: 4,
                bound: 2,
                ..
            })
        ));
        assert!(matches!(
            generate(&walked(32, 4, 0)),
            Err(KernelGenError::BelowMinimum { .. })
        ));
        // A split key's run may not straddle its parts: head 6 + shared 2, run 4 divides the key (8) but not the first part.
        let split = AttentionSpec {
            dk: 8,
            key: KeyLayout::Split { shared_dim: 2 },
            ..walked(32, 4, 4)
        };
        assert!(matches!(
            generate(&split),
            Err(KernelGenError::NotDivisible {
                dim,
                value: 6,
                divisor: 4,
                ..
            }) if dim == "dk - shared_dim"
        ));
    }
}

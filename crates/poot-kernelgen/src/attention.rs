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
//! `lanes * dv` wide, lane `i` at `[i*dv, i*dv + dv)`; `m` and `l` in arrays 1 and 2, `lanes` wide each),
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
    BinOp, Body, Constant, IndexAxis, MathOp, Operand, Place, Rvalue, Ty, WorkgroupLocalDecl,
};

use crate::KernelGenError;
use crate::data_movement::Cfg;
use crate::emit::{copy as copy_local, cu};
use crate::fused::{FusedScalarOp, emit_scalar_op};
use crate::helpers::{copy, elem, ld, slice_dtype, slice_f32};
use crate::request::SerialWork;

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
    /// How the keys are walked and across how many workgroups.
    pub schedule: AttentionSchedule,
}

impl AttentionSpec {
    /// Query rows: one per `(batch, head, query row)`.
    pub fn rows(&self) -> usize {
        self.batch * self.hq * self.lq
    }

    /// Workgroups the schedule launches: one per row, or one per `(row, split)`.
    pub fn workgroups(&self) -> usize {
        match self.schedule {
            AttentionSchedule::Streaming { .. } => self.rows(),
            AttentionSchedule::SplitKv { splits, .. } => self.rows() * splits,
        }
    }

    /// The keys one workgroup walks: all `lk`, or one split's chunk.
    pub fn keys_per_workgroup(&self) -> usize {
        match self.schedule {
            AttentionSchedule::Streaming { .. } => self.lk,
            AttentionSchedule::SplitKv { splits, .. } => self.lk.div_ceil(splits),
        }
    }

    /// The `f32` elements the split schedule's partial state occupies: `(o[dv], m, l)` per `(row,
    /// split)`. `None` for the streaming schedule, which has no intermediate.
    pub fn partial_elems(&self) -> Option<usize> {
        match self.schedule {
            AttentionSchedule::Streaming { .. } => None,
            AttentionSchedule::SplitKv { splits, .. } => Some(self.rows() * splits * (self.dv + 2)),
        }
    }

    /// The serial work of one dispatch: `workgroups * lanes` threads, each stepping over its share of the
    /// keys and, per key, over the key width (the score dot product) and the value width (the output update).
    pub(crate) fn serial_work(&self) -> SerialWork {
        let lanes = self.schedule.lanes();
        SerialWork {
            threads: (self.workgroups() * lanes) as u64,
            steps_per_thread: (self.keys_per_workgroup().div_ceil(lanes) * (self.dk + self.dv))
                as u64,
        }
    }
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
            AttentionSchedule::Streaming { .. } => None,
            AttentionSchedule::SplitKv { lanes, splits } => Some(Self {
                rows: spec.rows(),
                splits,
                dv: spec.dv,
                lanes,
            }),
        }
    }

    pub(crate) fn serial_work(&self) -> SerialWork {
        SerialWork {
            threads: (self.rows * self.lanes) as u64,
            steps_per_thread: (self.dv.div_ceil(self.lanes) * self.splits) as u64,
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
}

impl AttentionSchedule {
    /// The workgroup width: the lanes that stride the keys.
    pub fn lanes(self) -> usize {
        match self {
            Self::Streaming { lanes } | Self::SplitKv { lanes, .. } => lanes,
        }
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

/// Check the spec's own preconditions: every extent nonzero, `hq` a multiple of `n_rep`, and a split key's
/// shared part a proper, nonempty slice of the key width.
fn validate(spec: &AttentionSpec) -> Result<usize, KernelGenError> {
    let lanes = spec.schedule.lanes();
    let splits = match spec.schedule {
        AttentionSchedule::Streaming { .. } => 1,
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
    // A split with no key at all is a wasted workgroup, and `ceil(lk / splits)` would leave it empty.
    if splits > spec.lk {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "splits".to_string(),
            value: splits,
            bound: spec.lk,
        });
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

/// Build the attention kernel `spec` describes. The launch is `spec.workgroups()` workgroups of `lanes`
/// threads laid over a 2-D grid `x_groups` workgroups across: `gi = GroupY * x_groups + GroupX` is the flat
/// `(batch, head, row)` index, and a workgroup past `workgroups()` (the fold's remainder) returns.
/// Parameters: `q`, `k`, `v`, `mask`, then `k_shared` for a split key or `read` for a paged one, then `len` for a
/// live bound, then `out`
/// (the partial-state scratch under [`AttentionSchedule::SplitKv`]: the `x_groups`/`gi` index then runs over
/// `(row, split)`, `gi = row * splits + split`).
pub(crate) fn attention_body(
    name: &str,
    spec: &AttentionSpec,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    let lanes = validate(spec)?;
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
        schedule,
    } = *spec;
    let splits = match schedule {
        AttentionSchedule::Streaming { .. } => None,
        AttentionSchedule::SplitKv { splits, .. } => Some(splits),
    };
    let chunk = spec.keys_per_workgroup();
    let hkv = hq / n_rep;
    let mask_heads = match mask_kind {
        MaskKind::Broadcast => 1,
        MaskKind::PerHead => hq,
    };

    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        ld(slice_f32(false), false), // 2 k
        ld(slice_f32(false), false), // 3 v
        ld(slice_f32(false), false), // 4 mask
    ];
    let mut next_param = |decl| {
        params.push(decl);
        poot_kernel_ir::Local {
            index: params.len() as u32 - 1,
        }
    };
    let k_shared = match key {
        KeyLayout::Single | KeyLayout::Paged => None,
        KeyLayout::Split { .. } => Some(next_param(ld(slice_f32(false), false))),
    };
    let read = match key {
        KeyLayout::Single | KeyLayout::Split { .. } => None,
        KeyLayout::Paged => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
    };
    let len = match bound {
        KeyBound::Static => None,
        KeyBound::Live => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
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
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let gi = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(gi), cu(spec.workgroups()));

    c.if_then(in_range, |c| {
        // row = (b * hq + h) * lq + i; under the split schedule gi = row * splits + split.
        let (row, split) = match splits {
            None => (gi, None),
            Some(splits) => (
                uc(c, BinOp::Div, gi, splits),
                Some(uc(c, BinOp::Rem, gi, splits)),
            ),
        };
        let i = uc(c, BinOp::Rem, row, lq);
        let bh = uc(c, BinOp::Div, row, lq);
        let h = uc(c, BinOp::Rem, bh, hq);
        let b = uc(c, BinOp::Div, bh, hq);
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
        let maskbase = {
            let plane = match mask_kind {
                MaskKind::Broadcast => uc(c, BinOp::Mul, b, mask_heads),
                MaskKind::PerHead => {
                    let by_batch = uc(c, BinOp::Mul, b, mask_heads);
                    ubin(c, BinOp::Add, by_batch, h)
                }
            };
            let plane_start = uc(c, BinOp::Mul, plane, lq * lk);
            let row_start = uc(c, BinOp::Mul, i, lk);
            ubin(c, BinOp::Add, plane_start, row_start)
        };
        let lane_slot = uc(c, BinOp::Mul, lane, dv);

        // This lane's partial state: the running maximum and denominator in registers, `o` in LDS.
        let m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
        let l = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
        count(c, cu(0), dv, 1, |c, channel| {
            let slot = ubin(c, BinOp::Add, lane_slot, channel);
            lds_write(c, O_ARRAY, slot, f32c(0.0));
        });

        // The keys this row visits: all `lk`, or the first `len[b]` clamped into `0..=lk`. A negative length
        // is a zero bound (the signed compare comes before the unsigned cast), a length past `lk` is `lk`.
        let key_end = match len {
            None => cu(lk),
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
            None => (local_op(lane), key_end),
            Some(split) => {
                let begin = uc(c, BinOp::Mul, split, chunk);
                let end = c.var(Ty::Usize, Rvalue::Use(key_end));
                let chunk_end = uc(c, BinOp::Add, begin, chunk);
                let shorter = c.bin(Ty::Bool, BinOp::Lt, local_op(chunk_end), local_op(end));
                c.if_then(shorter, |c| {
                    c.assign(Place::local(end), Rvalue::Use(local_op(chunk_end)));
                });
                let first = ubin(c, BinOp::Add, begin, lane);
                (local_op(first), local_op(end))
            }
        };
        count_to(c, first_key, key_end, lanes, |c, t| {
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
                    let kv_elem = c.let_(Ty::F32, Rvalue::Use(copy(elem(part, ki))));
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
            let mi = ubin(c, BinOp::Add, maskbase, t);
            let mask_value = c.let_(Ty::F32, Rvalue::Use(copy(elem(mask, mi))));
            let s = fbin(c, BinOp::Add, s, mask_value);

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
                let slot = ubin(c, BinOp::Add, lane_slot, channel);
                let vi = ubin(c, BinOp::Add, value_row, channel);
                let held = lds_read(c, O_ARRAY, slot);
                let rescaled = fbin(c, BinOp::Mul, held, corr);
                let vv = c.let_(Ty::F32, Rvalue::Use(copy(elem(v, vi))));
                let added = fbin(c, BinOp::Mul, weight, vv);
                let sum = fbin(c, BinOp::Add, rescaled, added);
                lds_write(c, O_ARRAY, slot, local_op(sum));
            });
            c.assign(Place::local(m), Rvalue::Use(local_op(m_new)));
        });

        // Publish this lane's `(m, l)`, then the one barrier: every lane reaches it exactly once.
        lds_write(c, M_ARRAY, lane, local_op(m));
        lds_write(c, L_ARRAY, lane, local_op(l));
        c.barrier();

        // The combine. `big_m` is the maximum of the lane maxima; a lane whose maximum is `-inf` saw no
        // unmasked key and weighs zero, so no `-inf - -inf` is formed.
        let big_m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
        count(c, cu(0), lanes, 1, |c, other| {
            let m_other = lds_read(c, M_ARRAY, other);
            let next = fbin(c, BinOp::Max, big_m, m_other);
            c.assign(Place::local(big_m), Rvalue::Use(local_op(next)));
        });
        let denominator = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
        count(c, cu(0), lanes, 1, |c, other| {
            let w = lane_weight(c, other, big_m);
            let l_other = lds_read(c, L_ARRAY, other);
            let part = fbin(c, BinOp::Mul, l_other, w);
            let next = fbin(c, BinOp::Add, denominator, part);
            c.assign(Place::local(denominator), Rvalue::Use(local_op(next)));
        });
        // The streaming schedule writes the normalized row; a split writes its unnormalized state.
        let out_base = match split {
            None => uc(c, BinOp::Mul, gi, dv),
            Some(_) => uc(c, BinOp::Mul, gi, dv + 2),
        };
        count(c, local_op(lane), dv, lanes, |c, channel| {
            let acc = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
            count(c, cu(0), lanes, 1, |c, other| {
                let w = lane_weight(c, other, big_m);
                let slot = {
                    let start = uc(c, BinOp::Mul, other, dv);
                    ubin(c, BinOp::Add, start, channel)
                };
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
            let first_lane = c.bin(Ty::Bool, BinOp::Eq, local_op(lane), cu(0));
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
    let (scale_bits, softcap) = match score {
        ScoreTransform::Scale { scale_bits } => (scale_bits, None),
        ScoreTransform::ScaleSoftcap {
            scale_bits,
            cap_bits,
        } => (scale_bits, Some(f32::from_bits(cap_bits))),
    };
    let scaled = c.bin(
        Ty::F32,
        BinOp::Mul,
        local_op(dot),
        f32c(f32::from_bits(scale_bits)),
    );
    let Some(cap) = softcap else {
        return scaled;
    };
    let unit = c.bin(Ty::F32, BinOp::Mul, local_op(scaled), f32c(1.0 / cap));
    let tanh = c.emit(|e| {
        let dst = e.let_(Ty::F32, Rvalue::Use(f32c(0.0)));
        let (al, stmts) = e.parts();
        emit_scalar_op(dst, FusedScalarOp::Tanh, &[local_op(unit)], al, stmts);
        dst
    });
    c.bin(Ty::F32, BinOp::Mul, local_op(tanh), f32c(cap))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> AttentionSpec {
        AttentionSpec {
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
            schedule: AttentionSchedule::Streaming { lanes: 4 },
        }
    }

    /// The generator's own preconditions are typed errors: a head count the grouping does not divide, a zero
    /// extent and a split key whose shared part is empty or the whole key, each naming the generator.
    /// A value width that differs from the key width is not one of them.
    #[test]
    fn a_spec_outside_the_generators_contract_is_a_typed_error() {
        let wider_values = AttentionSpec { dv: 8, ..spec() };
        assert!(attention_body("a", &wider_values, 1).is_ok());
        let ragged = AttentionSpec { n_rep: 3, ..spec() };
        assert!(matches!(
            attention_body("a", &ragged, 1),
            Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                divisor: 3,
                ..
            })
        ));
        let no_lanes = AttentionSpec {
            schedule: AttentionSchedule::Streaming { lanes: 0 },
            ..spec()
        };
        assert!(matches!(
            attention_body("a", &no_lanes, 1),
            Err(KernelGenError::BelowMinimum {
                generator: GENERATOR,
                ..
            })
        ));
        let empty_shared = AttentionSpec {
            key: KeyLayout::Split { shared_dim: 0 },
            ..spec()
        };
        assert!(matches!(
            attention_body("a", &empty_shared, 1),
            Err(KernelGenError::BelowMinimum {
                generator: GENERATOR,
                ..
            })
        ));
        let all_shared = AttentionSpec {
            key: KeyLayout::Split { shared_dim: 4 },
            ..spec()
        };
        assert!(matches!(
            attention_body("a", &all_shared, 1),
            Err(KernelGenError::ExceedsBound {
                generator: GENERATOR,
                value: 4,
                bound: 3,
                ..
            })
        ));
    }
}

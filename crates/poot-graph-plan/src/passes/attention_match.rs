use super::guard_witness::{first_unobserved_clamp, producer_map};
use super::{
    BinOp, DType, EditError, Eqn, Graph, HashMap, HashSet, NumericalRewriteDecline,
    NumericalRewriteReason, OpKind, Operand, RedOp, Scalar, Storage, UnOp, ValueId, with_eqns,
};
use poot_graph_ir::{
    AttentionAttrs, ComputedConst, IndexRange, KeyLayout, KvBound, MaskOperand, RangeFacts,
    SinkOperand, ValueTable, ops::MASK_HIDDEN, value_ranges,
};

/// A fusion the attention or rope matcher left alone on a near-miss chain, and the first check that
/// failed. [`Program::declines`](crate::Program::declines) publishes them; the original arithmetic stays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusionDecline {
    /// The result of the candidate equation, as the traced graph names it.
    pub value: ValueId,
    /// The check that failed.
    pub reason: FusionDeclineReason,
}

/// Which matcher declined a candidate, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionDeclineReason {
    /// The flash-attention matcher.
    Attention(AttentionDecline),
    /// The rope matcher.
    Rope(RopeDecline),
}

/// Why the attention matcher did not fuse a candidate equation, in the order the checks run. The pass
/// leaves the equation alone and records the first failing check (it never guesses); tests assert the
/// exact reason for a near-miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionDecline {
    /// The root is not `MatMul(p, V)`.
    RootNotMatmul,
    /// The chain is not F32: the generated attention kernel reads and writes F32 only, so a BF16 or F16
    /// chain stays the decomposition (every primitive of it has a lowering at its own dtype).
    NotF32,
    /// `p` is not `Div(e, denom)` (the two-pass softmax normalizer).
    NotDiv,
    /// `denom` is not a keepdim `Sum` over the same `e` on the last axis.
    Denom,
    /// `e` is not `Exp(shifted)`.
    NotExp,
    /// `shifted` is not `Sub(scores, m)`.
    NotSub,
    /// The normalizer adds a sink mass (or the row maximum folds a sink logit) but the sink spelling is not
    /// the one [`poot_graph_ir::ops::attention_masked_sinks`] emits: `m = Max(rowmax, sinks)`, `denom =
    /// Add(Sum(e), Exp(Sub(sinks, m)))`, both over the same `sinks`.
    SinkChain,
    /// The sink logits are not `Reshape(sinks [Hq], [1, Hq, 1, 1])`.
    SinkLogits,
    /// `m` is not a keepdim `Max` over the same `scores` on the last axis.
    RowMax,
    /// `scores` is not `Add(pre_mask, mask)` with one operand the scaled `Q K^T` product.
    ScoreAdd,
    /// The scaled `Q K^T` product is not `Mul(qk, scale)` or the softcap composition.
    ScoreProduct,
    /// A softcap chain was matched but its two constants are not reciprocals.
    SoftcapRecip,
    /// The softcap chain's shape is not the tanh composition `ops::softcap` emits.
    SoftcapShape,
    /// `qk` is not `MatMul(q, kt)`.
    QkMatmul,
    /// `kt` is not `Transpose(k, [0, 1, 3, 2])`.
    KeyTranspose,
    /// The key or value GQA repeat chains disagree (`n_rep_k != n_rep_v`).
    GqaRepeatMismatch,
    /// The scores pass through a `Select` that is not the canonical live-bound score guard: `Select(active,
    /// scores, -inf)` over the row's key positions, `active` the length comparison.
    LiveScoreGuard,
    /// A live-bound score guard without the matching guard on the keys and values: some source of the key (or
    /// the value) is not `Select(active, x, 0)` over the same length, so inactive K or V could still reach a
    /// matmul and the guarded semantics do not hold.
    LiveKeyGuard,
    /// The live length is not proved to lie in `1..=Lk`: it is not `guarded + 1` of a canonical index-bounds
    /// guard over `Lk` whose witness the graph declares as a validation output.
    LiveLengthUnproved,
    /// The fused op's own typing rule disagrees with the region it would replace (infer-and-compare).
    TypeMismatch,
    /// A prefill-shaped attention whose `Hq * Lq` workgroup count is over the target's grid cap: the fused
    /// kernel launches one workgroup per query row and the target has no plan-time fallback for a larger grid.
    PrefillGrid,
}

/// Why the rope matcher did not fuse a candidate equation. The same first-failing-check contract as
/// [`AttentionDecline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeDecline {
    /// The root is not `Add(xc, rs)`.
    RootNotAdd,
    /// Neither addend is `Mul(rotate_half, sin)` with a `Concat` rotate-half.
    NoRotateHalf,
    /// The rotate-half op is not a two-input `Concat` on `a`'s last axis.
    RotateHalfConcat,
    /// The first concat input is not `Neg(x2)`.
    RotatedNeg,
    /// `x2` is not `Slice(a, half..rot)` on the last axis.
    RotatedSlice,
    /// `x1` is not `Slice(a, 0..half)` on the same last axis.
    KeepSlice,
    /// The two slices do not split the same tensor at the same `half`, or `rot != 2 * half`.
    SliceMismatch,
    /// `a` (the sliced tensor) is not a factor of the `Mul(a, cos)` addend.
    RotatedFactor,
    /// `cos`/`sin` do not cover exactly the rotated width on their last axis.
    TableWidth,
}

/// The equation indices that consume each value.
fn consumer_map(g: &Graph) -> ValueTable<Vec<usize>> {
    let mut consumers = ValueTable::filled(g.values().len(), Vec::new());
    for (index, eqn) in g.eqns().iter().enumerate() {
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                consumers[*value].push(index);
            }
        }
    }
    consumers
}

fn is_liveness_root(g: &Graph, value: ValueId) -> bool {
    g.liveness_roots().any(|root| root == value)
}

/// `inputs` as `(value, scalar-literal)` in either order: commutative scalar placement (R467-005).
fn value_and_scalar(inputs: &[Operand]) -> Option<(ValueId, Scalar)> {
    match inputs {
        [Operand::Value(value), Operand::Lit(scalar)] => Some((*value, *scalar)),
        [Operand::Lit(scalar), Operand::Value(value)] => Some((*value, *scalar)),
        _ => None,
    }
}

/// `inputs` as `(value, value)`: both operands are values, in their original order.
fn two_values(inputs: &[Operand]) -> Option<(ValueId, ValueId)> {
    match inputs {
        [Operand::Value(a), Operand::Value(b)] => Some((*a, *b)),
        _ => None,
    }
}

#[cfg(any(test, feature = "test-support"))]
/// Canonicalize only proved identities. A scalar can be removed from a contraction operand when it
/// is exactly F32 +1 and the multiplication is F32. No general factoring or reciprocal-to-division
/// rewrite is valid without a range and rounding proof. Matchers detect operand roles separately.
pub fn canonicalize(g: &Graph) -> Result<Graph, EditError> {
    Ok(canonicalize_with_declines(g)?.0)
}

/// The canonical graph and typed explanations of numerical rewrites that were withheld. Value ids
/// identify the original candidate's result; a declined optimization does not refuse compilation.
pub(crate) fn canonicalize_with_declines(
    g: &Graph,
) -> Result<(Graph, Vec<NumericalRewriteDecline>), EditError> {
    let producers = producer_map(g);
    let consumers = consumer_map(g);
    let mut out = g.edit();
    let mut declines = Vec::new();
    for eqn in out.eqns_mut() {
        if matches!(eqn.op, OpKind::Binary(BinOp::Mul))
            && two_values(&eqn.inputs).is_some()
            && eqn.inputs.iter().any(|operand| {
                let Operand::Value(value) = operand else {
                    return false;
                };
                producers
                    .get(value)
                    .is_some_and(|&index| matches!(g.eqns()[index].op, OpKind::Unary(UnOp::Recip)))
            })
        {
            declines.push(NumericalRewriteDecline {
                value: eqn.out,
                reason: NumericalRewriteReason::ReciprocalRounding,
            });
        }
        if !matches!(eqn.op, OpKind::MatMul) {
            continue;
        }
        for operand in &mut eqn.inputs {
            let Operand::Value(side) = operand else {
                continue;
            };
            let Some(&index) = producers.get(side) else {
                continue;
            };
            let mul = &g.eqns()[index];
            if !matches!(mul.op, OpKind::Binary(BinOp::Mul))
                || consumers[*side].len() != 1
                || is_liveness_root(g, *side)
            {
                continue;
            }
            let Some((input, scalar)) = value_and_scalar(&mul.inputs) else {
                continue;
            };
            if matches!(scalar, Scalar::F32(1.0))
                && g.aval(input).dtype == DType::F32
                && g.aval(*side).dtype == DType::F32
            {
                // Multiplication by +1 is an identity, including subnormals and signed zero.
                // The contraction still computes every product in the same order.
                *operand = Operand::Value(input);
            } else {
                declines.push(NumericalRewriteDecline {
                    value: eqn.out,
                    reason: NumericalRewriteReason::ScalarAcrossContraction,
                });
            }
        }
    }
    Ok((out.commit()?, declines))
}
/// A matched materialized softmax-attention subgraph (the operands of the fused op it collapses to).
pub(super) struct AttnMatch {
    pub(super) q: ValueId,
    /// The pre-GQA-repeat key (the per-head part of a split key), or the key pool of a paged one.
    pub(super) k: ValueId,
    /// The part of a split key shared by every key head, before its repeat chain.
    pub(super) k_shared: Option<ValueId>,
    /// The `[B, Lk]` slot map of a paged key, which reads `k` and `v` as pools.
    pub(super) read: Option<ValueId>,
    pub(super) key: KeyLayout,
    /// The pre-GQA-repeat value, or the value pool of a paged one.
    pub(super) v: ValueId,
    /// The mask operand: the additive tensor, or the `[B, Lq]` positions of a causal mask.
    pub(super) mask: ValueId,
    pub(super) mask_kind: MaskOperand,
    /// The proved live length `[B]`, when the chain is the guarded live-capacity composite.
    pub(super) len: Option<ValueId>,
    /// The per-head sink logits `[Hq]`, when the softmax carries a sink.
    pub(super) sinks: Option<ValueId>,
    pub(super) n_rep: usize,
    pub(super) scale: f32,
    pub(super) decode: bool, // M==1 (decode) vs M>1 (prefill)
    /// `Some(c)` when the scaled `Q K^T` passes through a Gemma2/Grok attention-logit softcap
    /// (`c * tanh(scores / c)`, [`poot_graph_ir::ops::softcap`]) before the mask-add.
    pub(super) softcap: Option<f32>,
}

/// A key (or value) source as the live-bound composite guards it: the data and the condition that selects it.
struct Guarded {
    data: ValueId,
    active: ValueId,
}

/// `value`'s producing equation and its three value operands when it is a `Select`.
fn select_operands(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    value: ValueId,
) -> Option<[ValueId; 3]> {
    let eqn = prod.get(&value).map(|&i| &g.eqns()[i])?;
    if !matches!(eqn.op, OpKind::Select) {
        return None;
    }
    match eqn.inputs.as_slice() {
        [Operand::Value(cond), Operand::Value(t), Operand::Value(f)] => Some([*cond, *t, *f]),
        _ => None,
    }
}

/// `value` as the `[0, 1, .., len)` range of `dtype`, produced by an `Iota` equation or bound as the computed
/// constant `fold_iota` leaves.
fn iota_len(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    value: ValueId,
    dtype: DType,
) -> Option<usize> {
    match prod.get(&value).map(|&i| &g.eqns()[i].op) {
        Some(OpKind::Iota { len, dtype: d }) if *d == dtype => Some(*len),
        Some(_) => None,
        None => match g.meta(value).storage {
            Storage::Computed(ComputedConst::Iota { len, dtype: d }) if d == dtype => Some(len),
            _ => None,
        },
    }
}

/// The key positions `1..=lk` of the live bound: `Add(iota_i32(lk), 1)`, exact I32 `[lk]`.
fn is_key_ends(g: &Graph, prod: &HashMap<ValueId, usize>, ends: ValueId, lk: usize) -> bool {
    let Some(eqn) = prod.get(&ends).map(|&i| &g.eqns()[i]) else {
        return false;
    };
    matches!(eqn.op, OpKind::Binary(BinOp::Add))
        && matches!(
            value_and_scalar(&eqn.inputs),
            Some((iota, Scalar::I32(1))) if iota_len(g, prod, iota, DType::I32) == Some(lk)
        )
}

/// `value` as `reshape(source)` to exactly `shape`.
fn reshaped_from(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    value: ValueId,
    shape: &[usize],
) -> Option<ValueId> {
    let eqn = prod.get(&value).map(|&i| &g.eqns()[i])?;
    match (&eqn.op, eqn.inputs.as_slice()) {
        (OpKind::Reshape { shape: target }, [Operand::Value(source)]) if target == shape => {
            Some(*source)
        }
        _ => None,
    }
}

/// The length behind a live-bound activity condition `Ge(reshape(len, [B,1,1,1]), reshape(ends, along))`,
/// where `along` puts the `lk` key positions on the key-row axis (`[1,1,lk,1]`) or the score-row axis
/// (`[1,1,1,lk]`). `None` for any other comparison: an off-by-one key range, another operand order, a
/// different axis.
fn live_active_len(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    active: ValueId,
    lk: usize,
    along: [usize; 4],
) -> Option<ValueId> {
    let eqn = prod.get(&active).map(|&i| &g.eqns()[i])?;
    let (OpKind::Binary(BinOp::Ge), [Operand::Value(len_view), Operand::Value(ends_view)]) =
        (&eqn.op, eqn.inputs.as_slice())
    else {
        return None;
    };
    let len = reshaped_from(g, prod, *len_view, &[g.aval(*len_view).shape[0], 1, 1, 1])?;
    let ends = reshaped_from(g, prod, *ends_view, &along)?;
    is_key_ends(g, prod, ends, lk).then_some(len)
}

/// Which select fallback of the live bound a value must be.
#[derive(Clone, Copy)]
enum LiveFallback {
    /// The K/V rows' `Mul(iota(lk), 0.0)`.
    Zero,
    /// The scores' `Add(Mul(iota(lk), 0.0), -inf)`.
    NegInf,
}

/// `fallback` as the live bound's select fallback over key row `along`: `reshape` of the F32 zero range
/// `Mul(iota(lk), 0.0)`, or of that range plus `-inf`.
fn is_live_fallback(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    fallback: ValueId,
    lk: usize,
    along: [usize; 4],
    which: LiveFallback,
) -> bool {
    let Some(product) = reshaped_from(g, prod, fallback, &along) else {
        return false;
    };
    let zero_range = match which {
        LiveFallback::Zero => product,
        LiveFallback::NegInf => {
            let Some(eqn) = prod.get(&product).map(|&i| &g.eqns()[i]) else {
                return false;
            };
            match (&eqn.op, value_and_scalar(&eqn.inputs)) {
                (OpKind::Binary(BinOp::Add), Some((zero, Scalar::F32(inf))))
                    if inf == f32::NEG_INFINITY =>
                {
                    zero
                }
                _ => return false,
            }
        }
    };
    let Some(eqn) = prod.get(&zero_range).map(|&i| &g.eqns()[i]) else {
        return false;
    };
    matches!(eqn.op, OpKind::Binary(BinOp::Mul))
        && matches!(
            value_and_scalar(&eqn.inputs),
            Some((iota, Scalar::F32(zero)))
                if zero.to_bits() == 0.0f32.to_bits()
                    && iota_len(g, prod, iota, DType::F32) == Some(lk)
        )
}

/// `source` as a guarded K or V source: `Select(active, data, zeros)` with the live bound's zero fallback.
fn peel_live_rows(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    source: ValueId,
    lk: usize,
) -> Option<Guarded> {
    let [active, data, zeros] = select_operands(g, prod, source)?;
    is_live_fallback(g, prod, zeros, lk, [1, 1, lk, 1], LiveFallback::Zero)
        .then_some(Guarded { data, active })
}

/// A key as the matcher recovers it: the per-head part, the optional shared part, the GQA repeat of the whole
/// key, and the live-bound guards on its sources (empty for a capacity key).
struct PeeledKey {
    head: ValueId,
    shared: Option<ValueId>,
    n_rep: usize,
    guards: Vec<Guarded>,
}

/// Peel a key operand: its GQA repeat, then either a single source or the `concat(head, repeat(shared))` of a
/// split key, with each source behind the live bound's select when `live`. `None` when `live` and a source is
/// not guarded.
fn peel_key(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    krep: ValueId,
    live: bool,
) -> Option<PeeledKey> {
    let (k, n_rep) = peel_repeat_kv(g, prod, krep);
    let lk = g.aval(k).shape.get(2).copied()?;
    let unguard = |source: ValueId, guards: &mut Vec<Guarded>| -> Option<ValueId> {
        if !live {
            return Some(source);
        }
        let guarded = peel_live_rows(g, prod, source, lk)?;
        let data = guarded.data;
        guards.push(guarded);
        Some(data)
    };
    let mut guards = Vec::new();
    let split = prod.get(&k).map(|&i| &g.eqns()[i]).and_then(|eqn| {
        let OpKind::Concat { axis } = eqn.op else {
            return None;
        };
        let [Operand::Value(head), Operand::Value(tail)] = eqn.inputs.as_slice() else {
            return None;
        };
        let (shared, tail_rep) = peel_repeat_kv(g, prod, *tail);
        // The shared part is one key head repeated across every head of the per-head part.
        let heads = g.aval(*head).shape.get(1).copied()?;
        let one_head = g.aval(shared).shape.get(1) == Some(&1);
        (axis == 3 && one_head && tail_rep == heads).then_some((*head, shared))
    });
    match split {
        Some((head, shared)) => Some(PeeledKey {
            head: unguard(head, &mut guards)?,
            shared: Some(unguard(shared, &mut guards)?),
            n_rep,
            guards,
        }),
        None => Some(PeeledKey {
            head: unguard(k, &mut guards)?,
            shared: None,
            n_rep,
            guards,
        }),
    }
}

/// A key or value view of a paged pool, as [`poot_graph_ir::ops::paged_view`] traces it.
struct PagedView {
    pool: ValueId,
    read: ValueId,
}

/// Peel `view` as `transpose(reshape(gather(pool, 0, reshape(read))))`, the logical `[B, Hkv, Lk, D]` view of a
/// `[S, Hkv, D]` pool through the I32 `[B, Lk]` slot map `read`. `None` for any other chain: its shapes are
/// checked, not just its op sequence, so a lookalike movement chain is not a paged view.
fn peel_paged_view(g: &Graph, prod: &HashMap<ValueId, usize>, view: ValueId) -> Option<PagedView> {
    let eqn = prod.get(&view).map(|&i| &g.eqns()[i])?;
    let (OpKind::Transpose { perm }, [Operand::Value(rows_first)]) =
        (&eqn.op, eqn.inputs.as_slice())
    else {
        return None;
    };
    if perm.as_slice() != [0, 2, 1, 3] {
        return None;
    }
    let &[rows, hkv, cap, d] = g.aval(view).shape.as_slice() else {
        return None;
    };
    let gathered = reshaped_from(g, prod, *rows_first, &[rows, cap, hkv, d])?;
    let gather = prod.get(&gathered).map(|&i| &g.eqns()[i])?;
    let (OpKind::Gather { axis: 0 }, [Operand::Value(pool), Operand::Value(slots)]) =
        (&gather.op, gather.inputs.as_slice())
    else {
        return None;
    };
    let read = reshaped_from(g, prod, *slots, &[rows * cap])?;
    let read_type = g.aval(read);
    let pool_type = g.aval(*pool);
    (read_type.dtype == DType::I32
        && read_type.shape == [rows, cap]
        && pool_type.shape.len() == 3
        && pool_type.shape[1..] == [hkv, d])
    .then_some(PagedView { pool: *pool, read })
}

/// The key and value of a paged cache: both views of pools through one slot map, whose declared range proves
/// every word a slot of the pools. The proof is what the kernel's unchecked pool read stands on, so a map
/// that is not proved stays the gather it is.
fn peel_paged_kv(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    facts: &RangeFacts,
    k: ValueId,
    v: ValueId,
) -> Option<(PagedView, PagedView)> {
    let (k, v) = (peel_paged_view(g, prod, k)?, peel_paged_view(g, prod, v)?);
    let slots = g.aval(k.pool).shape[0];
    let in_pool = IndexRange::below(slots)
        .zip(facts[k.read])
        .is_some_and(|(pool, proved)| proved.within(&pool));
    (k.read == v.read && g.aval(v.pool).shape[0] == slots && in_pool).then_some((k, v))
}

/// Peel a `repeat_kv(x, n_rep)` chain (`reshape(broadcast(reshape(x)))`) back to `(x, n_rep)`. If `v` is not
/// produced by that exact 3-op chain (the `n_rep == 1` case emits nothing), return `(v, 1)`.
fn peel_repeat_kv(g: &Graph, prod: &HashMap<ValueId, usize>, v: ValueId) -> (ValueId, usize) {
    let pe = |x: ValueId| -> Option<&Eqn> { prod.get(&x).map(|&i| &g.eqns()[i]) };
    let val = |o: &Operand| -> Option<ValueId> {
        match o {
            Operand::Value(x) => Some(*x),
            Operand::Lit(_) => None,
        }
    };
    (|| {
        let outer = pe(v)?;
        if !matches!(outer.op, OpKind::Reshape { .. }) {
            return None;
        }
        let bc = val(outer.inputs.first()?)?;
        let bc_eqn = pe(bc)?;
        let n_rep = match &bc_eqn.op {
            OpKind::Broadcast { shape } => *shape.get(2)?, // repeat_kv broadcasts the inserted axis 2
            _ => return None,
        };
        let inner = val(bc_eqn.inputs.first()?)?;
        if !matches!(pe(inner)?.op, OpKind::Reshape { .. }) {
            return None;
        }
        let orig = val(pe(inner)?.inputs.first()?)?;
        // Verify this is `ops::repeat_kv`'s exact shape chain (`[B,Hkv,S,D] -> [B,Hkv,1,S,D] ->
        // [B,Hkv,n_rep,S,D] -> [B,Hkv*n_rep,S,D]`), not just any reshape(broadcast(reshape)) chain: a
        // structurally similar but different movement chain would yield a wrong `(orig, n_rep)` pair that
        // still passes the caller's only cross-check (`n_rep_k == n_rep_v`), giving a silently wrong GQA
        // mapping.
        let os = g.aval(orig).shape.clone();
        if os.len() != 4 {
            return None;
        }
        let (bsz, h_kv, seq, d) = (os[0], os[1], os[2], os[3]);
        if g.aval(inner).shape != [bsz, h_kv, 1, seq, d] {
            return None;
        }
        if g.aval(bc).shape != [bsz, h_kv, n_rep, seq, d] {
            return None;
        }
        if g.aval(v).shape != [bsz, h_kv * n_rep, seq, d] {
            return None;
        }
        Some((orig, n_rep))
    })()
    .unwrap_or((v, 1))
}

/// The scaled `Q Kᵀ` product (with optional Gemma2/Grok softcap) under the additive mask: either the
/// plain `Mul(qk, scale_lit)` or, card 198, the softcap chain `ops::softcap` inserts around it:
/// `Mul(Tanh(Mul(Mul(qk, scale_lit), 1/cap_lit)), cap_lit)`, i.e. `cap * tanh((qk*scale) / cap)`.
/// `ops::tanh` is the single `UnOp::Tanh` primitive, so a non-softcap, non-tanh chain never matches.
///
/// Both the scale and the `1/cap` constants are matched in either operand position (R467-005 scalar
/// placement). Returns `(qk, scale, softcap)`.
fn match_pre_mask(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    pre_mask: ValueId,
) -> Result<(ValueId, f32, Option<f32>), AttentionDecline> {
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns()[i]) };
    let eqn = pe(pre_mask).ok_or(AttentionDecline::ScoreProduct)?;
    if matches!(eqn.op, OpKind::MatMul) && g.aval(pre_mask).dtype == DType::F32 {
        // A pre-scaled query remains an input value, including its rounding boundary. The
        // fused definition's post-contraction multiplication by +1 is exact; no scale moves.
        return Ok((pre_mask, 1.0, None));
    }
    if !matches!(eqn.op, OpKind::Binary(BinOp::Mul)) {
        return Err(AttentionDecline::ScoreProduct);
    }
    let (product, scalar) = value_and_scalar(&eqn.inputs).ok_or(AttentionDecline::ScoreProduct)?;
    let Scalar::F32(scalar) = scalar else {
        return Err(AttentionDecline::ScoreProduct);
    };
    let product_eqn = pe(product).ok_or(AttentionDecline::ScoreProduct)?;
    if matches!(product_eqn.op, OpKind::MatMul) {
        // No softcap: pre_mask = Mul(qk, scale_lit); `product` IS the qk MatMul directly.
        return Ok((product, scalar, None));
    }
    // Softcap: pre_mask = Mul(Tanh(tanh_arg), cap_lit); tanh_arg = Mul(rawscore, inv_cap_lit);
    // rawscore = Mul(qk, scale_lit). `scalar` is `cap`. Verify `tanh_arg`'s literal is `1/cap` (the
    // exact reciprocal `ops::softcap` builds) so an unrelated tanh-shaped chain with different
    // constants never matches.
    let cap = scalar;
    if cap == 0.0 {
        return Err(AttentionDecline::SoftcapRecip);
    }
    if !matches!(product_eqn.op, OpKind::Unary(UnOp::Tanh)) {
        return Err(AttentionDecline::SoftcapShape);
    }
    let Some(Operand::Value(tanh_arg)) = product_eqn.inputs.first() else {
        return Err(AttentionDecline::SoftcapShape);
    };
    let tanh_arg = *tanh_arg;
    let arg_eqn = pe(tanh_arg).ok_or(AttentionDecline::SoftcapShape)?;
    if !matches!(arg_eqn.op, OpKind::Binary(BinOp::Mul)) {
        return Err(AttentionDecline::SoftcapShape);
    }
    let (rawscore, inv_cap) =
        value_and_scalar(&arg_eqn.inputs).ok_or(AttentionDecline::SoftcapShape)?;
    let Scalar::F32(inv_cap) = inv_cap else {
        return Err(AttentionDecline::SoftcapShape);
    };
    if !cap.is_finite() || inv_cap.to_bits() != (1.0 / cap).to_bits() {
        return Err(AttentionDecline::SoftcapRecip);
    }
    let rawscore_eqn = pe(rawscore).ok_or(AttentionDecline::SoftcapShape)?;
    if !matches!(rawscore_eqn.op, OpKind::Binary(BinOp::Mul)) {
        return Err(AttentionDecline::SoftcapShape);
    }
    let (qk, scale) =
        value_and_scalar(&rawscore_eqn.inputs).ok_or(AttentionDecline::SoftcapShape)?;
    let Scalar::F32(scale) = scale else {
        return Err(AttentionDecline::SoftcapShape);
    };
    if !matches!(pe(qk).map(|eqn| &eqn.op), Some(OpKind::MatMul)) {
        return Err(AttentionDecline::SoftcapShape);
    }
    Ok((qk, scale, Some(cap)))
}

/// Match the materialized `softmax(scale * Q Kᵀ + mask) @ V` chain (with GQA repeat) that
/// `ops::attention_masked` / `ops::attention_prefill` produce, rooted at the final `MatMul(p, V)`.
/// Returns the fused op's operands, or the first [`AttentionDecline`] check that failed (conservative:
/// a non-attention softmax never matches).
///
/// Commutative operand order (the mask add in either order, `Mul(scale, qk)`) is canonicalized here by
/// role detection rather than by position (R467-005). Pre-scaled operands remain inputs, and
/// reciprocal normalization remains primitive because this fused definition uses division. The final
/// infer-and-compare gate requires the fused op's type to reproduce the root's declared type.
pub(super) fn match_attention(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    facts: &RangeFacts,
    root: &Eqn,
) -> Result<AttnMatch, AttentionDecline> {
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns()[i]) };
    let val = |o: &Operand| -> Option<ValueId> {
        match o {
            Operand::Value(x) => Some(*x),
            Operand::Lit(_) => None,
        }
    };
    let last_axis = |v: ValueId| g.aval(v).rank().saturating_sub(1);

    // root: out = MatMul(p, v_rep)
    if !matches!(root.op, OpKind::MatMul) {
        return Err(AttentionDecline::RootNotMatmul);
    }
    if g.aval(root.out).dtype != DType::F32 {
        return Err(AttentionDecline::NotF32);
    }
    let p = val(root.inputs.first().ok_or(AttentionDecline::RootNotMatmul)?)
        .ok_or(AttentionDecline::RootNotMatmul)?;
    let vrep = val(root.inputs.get(1).ok_or(AttentionDecline::RootNotMatmul)?)
        .ok_or(AttentionDecline::RootNotMatmul)?;

    // p = Div(e, denom)
    let div = pe(p).ok_or(AttentionDecline::NotDiv)?;
    if !matches!(div.op, OpKind::Binary(BinOp::Div)) {
        return Err(AttentionDecline::NotDiv);
    }
    let e =
        val(div.inputs.first().ok_or(AttentionDecline::NotDiv)?).ok_or(AttentionDecline::NotDiv)?;
    let denom =
        val(div.inputs.get(1).ok_or(AttentionDecline::NotDiv)?).ok_or(AttentionDecline::NotDiv)?;

    // denom = ReduceSum(e, last, keepdim) over the SAME e; with a sink, denom = Add(that sum, the sink mass)
    let denom_eqn = pe(denom).ok_or(AttentionDecline::Denom)?;
    let (sum_eqn, sink_mass) = match (&denom_eqn.op, two_values(&denom_eqn.inputs)) {
        (OpKind::Binary(BinOp::Add), Some((a, b))) => {
            // The sum is the addend that reduces `e`; the other is the sink's mass.
            let reduces = |v: ValueId| {
                pe(v).filter(|eqn| matches!(eqn.op, OpKind::Reduce { op: RedOp::Sum, .. }))
            };
            match (reduces(a), reduces(b)) {
                (Some(sum), _) => (sum, Some(b)),
                (None, Some(sum)) => (sum, Some(a)),
                (None, None) => return Err(AttentionDecline::Denom),
            }
        }
        _ => (denom_eqn, None),
    };
    match &sum_eqn.op {
        OpKind::Reduce {
            op: RedOp::Sum,
            axis,
            keepdim: true,
        } if val(sum_eqn.inputs.first().ok_or(AttentionDecline::Denom)?) == Some(e)
            && *axis == last_axis(e) => {}
        _ => return Err(AttentionDecline::Denom),
    }

    // e = Exp(shifted); shifted = Sub(scores, m)
    let e_eqn = pe(e).ok_or(AttentionDecline::NotExp)?;
    if !matches!(e_eqn.op, OpKind::Unary(UnOp::Exp)) {
        return Err(AttentionDecline::NotExp);
    }
    let shifted = val(e_eqn.inputs.first().ok_or(AttentionDecline::NotExp)?)
        .ok_or(AttentionDecline::NotExp)?;
    let sub = pe(shifted).ok_or(AttentionDecline::NotSub)?;
    if !matches!(sub.op, OpKind::Binary(BinOp::Sub)) {
        return Err(AttentionDecline::NotSub);
    }
    let scores =
        val(sub.inputs.first().ok_or(AttentionDecline::NotSub)?).ok_or(AttentionDecline::NotSub)?;
    let m =
        val(sub.inputs.get(1).ok_or(AttentionDecline::NotSub)?).ok_or(AttentionDecline::NotSub)?;

    // m = ReduceMax(scores, last, keepdim) over the SAME scores; with a sink, m = Max(that, sinks)
    let m_eqn = pe(m).ok_or(AttentionDecline::RowMax)?;
    let (row_max, sink_logits) = match (&m_eqn.op, two_values(&m_eqn.inputs)) {
        (OpKind::Binary(BinOp::Max), Some((row_max, logits))) => (row_max, Some(logits)),
        _ => (m, None),
    };
    let row_max_eqn = pe(row_max).ok_or(AttentionDecline::RowMax)?;
    match &row_max_eqn.op {
        OpKind::Reduce {
            op: RedOp::Max,
            axis,
            keepdim: true,
        } if val(row_max_eqn.inputs.first().ok_or(AttentionDecline::RowMax)?) == Some(scores)
            && *axis == last_axis(scores) => {}
        _ => return Err(AttentionDecline::RowMax),
    }
    // The sink is either in both the maximum and the normalizer, over the same logits, or in neither.
    let sinks = match (sink_logits, sink_mass) {
        (None, None) => None,
        (Some(logits), Some(mass)) => {
            let mass_eqn = pe(mass).ok_or(AttentionDecline::SinkChain)?;
            let [Operand::Value(shifted_sink)] = mass_eqn.inputs.as_slice() else {
                return Err(AttentionDecline::SinkChain);
            };
            let shifted_sink = pe(*shifted_sink).ok_or(AttentionDecline::SinkChain)?;
            if !matches!(mass_eqn.op, OpKind::Unary(UnOp::Exp))
                || !matches!(shifted_sink.op, OpKind::Binary(BinOp::Sub))
                || two_values(&shifted_sink.inputs) != Some((logits, m))
            {
                return Err(AttentionDecline::SinkChain);
            }
            let heads = *g
                .aval(logits)
                .shape
                .get(1)
                .ok_or(AttentionDecline::SinkLogits)?;
            Some(
                reshaped_from(g, prod, logits, &[1, heads, 1, 1])
                    .ok_or(AttentionDecline::SinkLogits)?,
            )
        }
        _ => return Err(AttentionDecline::SinkChain),
    };

    // The live bound's score guard: `scores = Select(active, added, -inf)`. Without it `scores` is the
    // additive-mask add itself.
    let lk = g.aval(scores).shape.last().copied().unwrap_or(0);
    let live_scores = select_operands(g, prod, scores);
    let live_len = match live_scores {
        None => None,
        Some([active, _, neg_inf]) => {
            let len = live_active_len(g, prod, active, lk, [1, 1, 1, lk])
                .ok_or(AttentionDecline::LiveScoreGuard)?;
            if !is_live_fallback(g, prod, neg_inf, lk, [1, 1, 1, lk], LiveFallback::NegInf) {
                return Err(AttentionDecline::LiveScoreGuard);
            }
            Some(len)
        }
    };
    let added = live_scores.map_or(scores, |[_, added, _]| added);

    // added = Add(pre_mask, mask). Either operand may be the additive mask (R467-005): the scaled
    // product is the one that decomposes to a `Q K^T`, the other is the mask.
    let add = pe(added).ok_or(AttentionDecline::ScoreAdd)?;
    if !matches!(add.op, OpKind::Binary(BinOp::Add)) {
        return Err(AttentionDecline::ScoreAdd);
    }
    let lhs = val(add.inputs.first().ok_or(AttentionDecline::ScoreAdd)?)
        .ok_or(AttentionDecline::ScoreAdd)?;
    let rhs = val(add.inputs.get(1).ok_or(AttentionDecline::ScoreAdd)?)
        .ok_or(AttentionDecline::ScoreAdd)?;
    let (pre_mask, mask) = match match_pre_mask(g, prod, lhs) {
        Ok(product) => (product, rhs),
        Err(first) => match match_pre_mask(g, prod, rhs) {
            Ok(product) => (product, lhs),
            Err(_) => return Err(first),
        },
    };
    let (qk, scale, softcap) = pre_mask;

    // qk = MatMul(q, kt); kt = Transpose(krep, [0,1,3,2])
    let qk_eqn = pe(qk).ok_or(AttentionDecline::QkMatmul)?;
    if !matches!(qk_eqn.op, OpKind::MatMul) {
        return Err(AttentionDecline::QkMatmul);
    }
    let q = val(qk_eqn.inputs.first().ok_or(AttentionDecline::QkMatmul)?)
        .ok_or(AttentionDecline::QkMatmul)?;
    let kt = val(qk_eqn.inputs.get(1).ok_or(AttentionDecline::QkMatmul)?)
        .ok_or(AttentionDecline::QkMatmul)?;
    let kt_eqn = pe(kt).ok_or(AttentionDecline::KeyTranspose)?;
    match &kt_eqn.op {
        OpKind::Transpose { perm } if perm.as_slice() == [0, 1, 3, 2] => {}
        _ => return Err(AttentionDecline::KeyTranspose),
    }
    let krep = val(kt_eqn
        .inputs
        .first()
        .ok_or(AttentionDecline::KeyTranspose)?)
    .ok_or(AttentionDecline::KeyTranspose)?;

    let key = peel_key(g, prod, krep, live_len.is_some()).ok_or(AttentionDecline::LiveKeyGuard)?;
    let (v_source, n_rep_v) = peel_repeat_kv(g, prod, vrep);
    let (v, v_guard) = match live_len {
        None => (v_source, None),
        Some(_) => {
            let guarded =
                peel_live_rows(g, prod, v_source, lk).ok_or(AttentionDecline::LiveKeyGuard)?;
            (guarded.data, Some(guarded))
        }
    };
    if key.n_rep != n_rep_v {
        return Err(AttentionDecline::GqaRepeatMismatch);
    }
    // Every guarded source must be selected by this very length, and the length must be proved.
    if let Some(len) = live_len {
        let same_len = key.guards.iter().chain(&v_guard).all(|guarded| {
            live_active_len(g, prod, guarded.active, lk, [1, 1, lk, 1]) == Some(len)
        });
        if !same_len {
            return Err(AttentionDecline::LiveKeyGuard);
        }
        if !live_length_proved(g, prod, facts, len, lk) {
            return Err(AttentionDecline::LiveLengthUnproved);
        }
    }

    let decode = g.aval(q).shape.get(2).copied() == Some(1);
    // A key and value that are views of one paged pool read it in place.
    let paged = match key.shared {
        None => peel_paged_kv(g, prod, facts, key.head, v),
        Some(_) => None,
    };
    let (k, v, read) = match paged {
        Some((k, v)) => (k.pool, v.pool, Some(k.read)),
        None => (key.head, v, None),
    };
    // A mask built by `causal_mask_from_pos` over this score row is stated by its positions instead of
    // read from a tensor; any other mask (ALiBi's bias, a padding mask) stays the additive operand.
    let (mask, mask_kind) = match match_causal_mask(g, prod, mask, &g.aval(q).shape, lk) {
        Some((positions, window)) => (positions, MaskOperand::Causal { window }),
        None => (mask, MaskOperand::Additive),
    };
    let m = AttnMatch {
        q,
        k,
        k_shared: key.shared,
        read,
        key: match (key.shared, read) {
            (Some(shared), _) => KeyLayout::Split {
                shared_dim: g.aval(shared).shape[3],
            },
            (None, Some(_)) => KeyLayout::Paged,
            (None, None) => KeyLayout::Single,
        },
        v,
        mask,
        mask_kind,
        len: live_len,
        sinks,
        n_rep: key.n_rep,
        scale,
        decode,
        softcap,
    };
    // Shape proof (R467-001). The decomposition broadcasts its MatMul batch axes, so a valid chain can
    // feed K/V with a size-1 head or batch axis (MQA or a shared KV written by broadcasting instead of
    // `repeat_kv`), a broadcast mask, a batched prefill, a carried cache under a multi-row verify
    // (`trace_verify`, card 094 bug 2: `[L,cap]` mask, K longer than Q's L), or MLA's value width
    // differing from the QK width. The fused op indexes all of its operands densely, so it is only the
    // same computation when its own typing rule accepts the operands and reproduces the root's declared
    // type; anything else stays the decomposition.
    let operands: Vec<_> = m.operands().iter().map(|&x| g.aval(x).clone()).collect();
    if m.op().infer(&operands).ok().as_ref() != Some(g.aval(root.out)) {
        return Err(AttentionDecline::TypeMismatch);
    }
    Ok(m)
}

/// `mask` as `ops::causal_mask_from_pos(pos, lk, window)`, exactly as the helper emits it: the positions
/// `[B, Lq]` of the query `q_shape` and the window. `None` for any other mask, and for positions that are
/// not one row per query (a broadcast positions tensor is not the fused operand).
fn match_causal_mask(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    mask: ValueId,
    q_shape: &[usize],
    lk: usize,
) -> Option<(ValueId, Option<usize>)> {
    let producer = |v: ValueId| prod.get(&v).map(|&i| &g.eqns()[i]);
    let binary = |v: ValueId, op: BinOp| {
        producer(v)
            .filter(|eqn| matches!(&eqn.op, OpKind::Binary(o) if *o == op))
            .map(|eqn| eqn.inputs.as_slice())
    };
    // mask = Mul(Sub(visible, 1.0), -MASK_HIDDEN)
    let (bias, scalar) = value_and_scalar(binary(mask, BinOp::Mul)?)?;
    if scalar != Scalar::F32(-MASK_HIDDEN) {
        return None;
    }
    let (visible, scalar) = value_and_scalar(binary(bias, BinOp::Sub)?)?;
    if scalar != Scalar::F32(1.0) {
        return None;
    }
    // visible = Ge(pos_col, keys_row), times `Ge(keys_row, pos_col - (window - 1))` under a window
    let (causal, window) = match binary(visible, BinOp::Mul) {
        None => (visible, None),
        Some(inputs) => {
            let (causal, in_window) = two_values(inputs)?;
            let (keys_row, lower) = two_values(binary(in_window, BinOp::Ge)?)?;
            let (pos_col, scalar) = value_and_scalar(binary(lower, BinOp::Sub)?)?;
            let Scalar::F32(span) = scalar else {
                return None;
            };
            let causal_inputs = binary(causal, BinOp::Ge)?;
            if two_values(causal_inputs) != Some((pos_col, keys_row))
                || span < 0.0
                || span.fract() != 0.0
            {
                return None;
            }
            (causal, Some(span as usize + 1))
        }
    };
    let (pos_col, keys_row) = two_values(binary(causal, BinOp::Ge)?)?;
    // keys_row = reshape(iota_f32(lk), [1,1,1,lk]); pos_col = reshape(cast_f32(pos), [rows,1,tokens,1])
    let keys = reshaped_from(g, prod, keys_row, &[1, 1, 1, lk])?;
    if iota_len(g, prod, keys, DType::F32) != Some(lk) {
        return None;
    }
    let &[batch, _, tokens, _] = g.aval(pos_col).shape.as_slice() else {
        return None;
    };
    let cast = reshaped_from(g, prod, pos_col, &[batch, 1, tokens, 1])?;
    let cast = producer(cast).filter(|eqn| matches!(&eqn.op, OpKind::Cast { to: DType::F32 }))?;
    let [Operand::Value(positions)] = cast.inputs.as_slice() else {
        return None;
    };
    let positions_type = g.aval(*positions);
    (positions_type.dtype == DType::I32
        && positions_type.shape == [batch, tokens]
        && matches!(q_shape, [b, _, l, _] if (*b, *l) == (batch, tokens)))
    .then_some((*positions, window))
}

/// Whether the graph proves `len` in `1..=lk` on every output an executor publishes (POOT-726).
///
/// The range comes from [`value_ranges`] and two sources are told apart. A bound-checked declared slot
/// needs nothing more: the binder refuses a length outside it. A range that exists only because an index
/// guard clamped the length (`facts.clamps`) fuses only when each such guard's witness is a declared
/// validation output, so a bad length publishes nothing instead of turning silently into a valid-looking
/// answer (spec 375).
fn live_length_proved(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    facts: &RangeFacts,
    len: ValueId,
    lk: usize,
) -> bool {
    let live = i64::try_from(lk)
        .ok()
        .and_then(|lk| IndexRange::new(1, lk + 1));
    let in_range = live.is_some_and(|live| facts[len].is_some_and(|proved| proved.within(&live)));
    in_range && first_unobserved_clamp(g, prod, facts, len).is_none()
}

impl AttnMatch {
    /// The fused op this match collapses to.
    pub(super) fn op(&self) -> OpKind {
        OpKind::Attention(AttentionAttrs {
            key: self.key,
            bound: if self.len.is_some() {
                KvBound::Live
            } else {
                KvBound::Capacity
            },
            sink: if self.sinks.is_some() {
                SinkOperand::PerHead
            } else {
                SinkOperand::Absent
            },
            mask: self.mask_kind,
            ..AttentionAttrs::new(self.n_rep, self.scale, self.softcap)
        })
    }

    /// The fused op's operands: `[q, k, v, mask or positions]`, then the shared key part or the slot map,
    /// then the live length, then the sink logits.
    pub(super) fn operands(&self) -> Vec<ValueId> {
        [self.q, self.k, self.v, self.mask]
            .into_iter()
            .chain(self.k_shared)
            .chain(self.read)
            .chain(self.len)
            .chain(self.sinks)
            .collect()
    }
}

/// Flash-attention fusion (automatic fusion as a compiler pass), capped by `max_prefill_groups`: a
/// prefill-shaped `Attention` is emitted only when its `Hq*L` workgroup count is `<= max_prefill_groups`
/// (decode, `B*Hq` workgroups, is always emitted); `None` means no cap. Pattern-match the materialized
/// `softmax(scale * Q Kᵀ + mask) @ V` subgraph (with GQA repeat) that `ops::attention_masked` /
/// `ops::attention_prefill` produce, and replace its root with the fused `Attention` op, which the planner lowers to the generated
/// attention kernel (online softmax, never materializing the `[.,.,L,L]` scores). The replaced intermediates become
/// dead; run `dce` after. Conservative (only the exact chain matches) and result-preserving (one
/// semantic definition shared by tracing, the oracle and every lowering), so every attention tracer
/// gets flash without a separate `_flash` variant. Best run on a `cse`'d graph. `compile` passes its own
/// measured `target.caps.max_grid[0]` (wgpu's `gridDim.x` caps at 65535 and has no plan-time fallback
/// from the op to the decomposition).
///
/// Also returns the near-miss candidates it left alone, each with its first failing check. A candidate that is
/// not the softmax-normalized matmul shape at all (an ordinary matmul) is no near-miss and is not reported.
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn flash_attention_capped(
    g: &Graph,
    max_prefill_groups: Option<usize>,
) -> Result<(Graph, Vec<FusionDecline>), EditError> {
    let mut prod: HashMap<ValueId, usize> = HashMap::new();
    for (i, e) in g.eqns().iter().enumerate() {
        prod.insert(e.out, i);
    }
    // One range analysis per pass run: every live candidate reads its length from it.
    let facts = value_ranges(g);
    let fits = |m: &AttnMatch| -> bool {
        if m.decode {
            return true; // decode grid = Hq, always tiny
        }
        match max_prefill_groups {
            None => true,
            Some(cap) => {
                let s = &g.aval(m.q).shape; // [1, Hq, L, D]
                s.get(1).copied().unwrap_or(1) * s.get(2).copied().unwrap_or(1) <= cap
            }
        }
    };
    let mut declines = Vec::new();
    let eqns: Vec<Eqn> = g
        .eqns()
        .iter()
        .map(|e| match match_attention(g, &prod, &facts, e) {
            Ok(m) if fits(&m) => Eqn {
                op: m.op(),
                inputs: m.operands().into_iter().map(Operand::Value).collect(),
                out: e.out,
                layer: e.layer,
            },
            Ok(_) => {
                declines.push(FusionDecline {
                    value: e.out,
                    reason: FusionDeclineReason::Attention(AttentionDecline::PrefillGrid),
                });
                e.clone()
            }
            Err(reason) => {
                if attention_near_miss(g, &prod, e, reason) {
                    declines.push(FusionDecline {
                        value: e.out,
                        reason: FusionDeclineReason::Attention(reason),
                    });
                }
                e.clone()
            }
        })
        .collect();
    Ok((with_eqns(g, eqns)?, declines))
}

/// Whether `reason` is a failure of a softmax-weighted matmul: a contraction whose left operand is a `Div`
/// of an `Exp`. Anything else with a `Div` in front of a matmul (a sigmoid-gelu quotient, a plain
/// normalization) is no attention candidate and reports nothing. A BF16 or F16 chain fails `NotF32` before
/// the normalizer is checked, so it counts under the same shape test.
fn attention_near_miss(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    root: &Eqn,
    reason: AttentionDecline,
) -> bool {
    if matches!(
        reason,
        AttentionDecline::RootNotMatmul | AttentionDecline::NotDiv
    ) {
        return false;
    }
    let producer = |operand: Option<&Operand>| match operand {
        Some(Operand::Value(value)) => prod.get(value).map(|&i| &g.eqns()[i]),
        _ => None,
    };
    producer(root.inputs.first())
        .filter(|div| matches!(div.op, OpKind::Binary(BinOp::Div)))
        .and_then(|div| producer(div.inputs.first()))
        .is_some_and(|numerator| matches!(numerator.op, OpKind::Unary(UnOp::Exp)))
}

/// A matched RoPE rotate-half chain. `x` is the tensor the rotation applies to (width `rot` on the last
/// axis; for a partial rope `x` is the FULL head-dim tensor and `[rot, D)` passes through). `cos`/`sin` are
/// the width-`rot` tables (broadcast against `x`'s leading dims).
struct RopeMatch {
    x: ValueId,
    cos: ValueId,
    sin: ValueId,
    rot: usize,
}

/// The core of a rope match, rooted at the final `Add(xc, rs)` of `ops::rope_partial`'s rotated region:
/// `xc = Mul(a, cos)`, `rs = Mul(rotate_half, sin)`, `rotate_half = Concat(last, [Neg(Slice(a,
/// half..rot)), Slice(a, 0..half)])`. Returns `(a, cos, sin, rot)` where `a` is the rotated tensor
/// (`half = rot/2`), or the first [`RopeDecline`] check that failed.
///
/// The rotated tensor `a` is identified structurally (the tensor both slices split), and `cos` is
/// whichever factor of the `Mul` `a` is (R467-005: `Mul(cos, a)` and `Mul(a, cos)` are the same
/// rotation), so a non-rope `Add` of two `Mul`s never matches.
pub(super) fn match_rope_core(
    g: &Graph,
    prod: &HashMap<ValueId, usize>,
    add: &Eqn,
) -> Result<(ValueId, ValueId, ValueId, usize), RopeDecline> {
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns()[i]) };
    let val = |o: &Operand| -> Option<ValueId> {
        match o {
            Operand::Value(x) => Some(*x),
            Operand::Lit(_) => None,
        }
    };
    if !matches!(add.op, OpKind::Binary(BinOp::Add)) {
        return Err(RopeDecline::RootNotAdd);
    }
    let in0 =
        val(add.inputs.first().ok_or(RopeDecline::RootNotAdd)?).ok_or(RopeDecline::RootNotAdd)?;
    let in1 =
        val(add.inputs.get(1).ok_or(RopeDecline::RootNotAdd)?).ok_or(RopeDecline::RootNotAdd)?;
    let e0 = pe(in0).ok_or(RopeDecline::RootNotAdd)?;
    let e1 = pe(in1).ok_or(RopeDecline::RootNotAdd)?;
    // Both addends are Muls. rs is the one carrying the rotate-half Concat; xc is the other.
    let is_mul = |e: &Eqn| matches!(e.op, OpKind::Binary(BinOp::Mul));
    if !is_mul(e0) || !is_mul(e1) {
        return Err(RopeDecline::RootNotAdd);
    }
    let has_concat = |e: &Eqn| {
        e.inputs.iter().any(|operand| match operand {
            Operand::Value(value) => pe(*value)
                .map(|producer| matches!(producer.op, OpKind::Concat { .. }))
                .unwrap_or(false),
            Operand::Lit(_) => false,
        })
    };
    let (xc_eqn, rs_eqn) = if has_concat(e1) {
        (e0, e1)
    } else if has_concat(e0) {
        (e1, e0)
    } else {
        return Err(RopeDecline::NoRotateHalf);
    };

    // rs = Mul(rotate_half, sin): `rotate_half` is the Concat factor, `sin` the other.
    let (rs_a, rs_b) = two_values(&rs_eqn.inputs).ok_or(RopeDecline::NoRotateHalf)?;
    let rhs_concat = |value: ValueId| {
        pe(value)
            .map(|producer| matches!(producer.op, OpKind::Concat { .. }))
            .unwrap_or(false)
    };
    let (rh, sin) = if rhs_concat(rs_a) {
        (rs_a, rs_b)
    } else if rhs_concat(rs_b) {
        (rs_b, rs_a)
    } else {
        return Err(RopeDecline::NoRotateHalf);
    };

    // rotate_half = Concat(last, [neg_x2, x1])
    let rh_eqn = pe(rh).ok_or(RopeDecline::RotateHalfConcat)?;
    let concat_axis = match &rh_eqn.op {
        OpKind::Concat { axis } => *axis,
        _ => return Err(RopeDecline::RotateHalfConcat),
    };
    if rh_eqn.inputs.len() != 2 {
        return Err(RopeDecline::RotateHalfConcat);
    }
    let neg_x2 = val(rh_eqn.inputs.first().ok_or(RopeDecline::RotateHalfConcat)?)
        .ok_or(RopeDecline::RotateHalfConcat)?;
    let x1 = val(rh_eqn.inputs.get(1).ok_or(RopeDecline::RotateHalfConcat)?)
        .ok_or(RopeDecline::RotateHalfConcat)?;

    // neg_x2 = Neg(x2); x2 = Slice(a, last, half, rot)
    let neg_eqn = pe(neg_x2).ok_or(RopeDecline::RotatedNeg)?;
    if !matches!(neg_eqn.op, OpKind::Unary(UnOp::Neg)) {
        return Err(RopeDecline::RotatedNeg);
    }
    let x2 = val(neg_eqn.inputs.first().ok_or(RopeDecline::RotatedNeg)?)
        .ok_or(RopeDecline::RotatedNeg)?;
    let x2_eqn = pe(x2).ok_or(RopeDecline::RotatedSlice)?;
    let (a2, axis2, half2, rot) = match &x2_eqn.op {
        OpKind::Slice { axis, start, end } => (
            val(x2_eqn.inputs.first().ok_or(RopeDecline::RotatedSlice)?)
                .ok_or(RopeDecline::RotatedSlice)?,
            *axis,
            *start,
            *end,
        ),
        _ => return Err(RopeDecline::RotatedSlice),
    };
    // x1 = Slice(a, last, 0, half)
    let x1_eqn = pe(x1).ok_or(RopeDecline::KeepSlice)?;
    let (a1, axis1, half1) = match &x1_eqn.op {
        OpKind::Slice { axis, start, end } if *start == 0 => (
            val(x1_eqn.inputs.first().ok_or(RopeDecline::KeepSlice)?)
                .ok_or(RopeDecline::KeepSlice)?,
            *axis,
            *end,
        ),
        _ => return Err(RopeDecline::KeepSlice),
    };
    // Consistency: both slices split the SAME tensor `a` at `half`, and rot == 2*half.
    if a1 != a2 || axis1 != axis2 || half1 != half2 || rot != half1 * 2 {
        return Err(RopeDecline::SliceMismatch);
    }
    let a = a1;
    let half = half1;
    if half == 0 {
        return Err(RopeDecline::SliceMismatch);
    }
    let last = g
        .aval(a)
        .rank()
        .checked_sub(1)
        .ok_or(RopeDecline::SliceMismatch)?;
    if concat_axis != last || axis1 != last {
        return Err(RopeDecline::RotateHalfConcat);
    }

    // xc = Mul(a, cos): `cos` is whichever factor is not the rotated tensor.
    let (f0, f1) = two_values(&xc_eqn.inputs).ok_or(RopeDecline::RotatedFactor)?;
    let cos = if f0 == a {
        f1
    } else if f1 == a {
        f0
    } else {
        return Err(RopeDecline::RotatedFactor);
    };
    // cos/sin cover exactly the rotated width `rot` on their last axis.
    if g.aval(cos).shape.last().copied() != Some(rot)
        || g.aval(sin).shape.last().copied() != Some(rot)
    {
        return Err(RopeDecline::TableWidth);
    }
    Ok((a, cos, sin, rot))
}

/// Whether `Rope { rot }` over `[x, cos, sin]` is well-typed with exactly `out`'s declared type (R467-002).
/// The rotate-half chain broadcasts its `Mul`s, so a valid chain can grow `x` (e.g. `x[1,1,S,D]` against
/// per-head `cos[1,H,S,D]`), while `Rope` is shape-preserving: such a chain is not this op and stays the
/// decomposition.
fn rope_types_as(
    g: &Graph,
    x: ValueId,
    cos: ValueId,
    sin: ValueId,
    rot: usize,
    out: ValueId,
) -> bool {
    let operands = [x, cos, sin].map(|v| g.aval(v).clone());
    OpKind::Rope { rot }.infer(&operands).ok().as_ref() == Some(g.aval(out))
}

/// RoPE fusion (automatic fusion as a compiler pass). Pattern-match the
/// `ops::rope_partial` rotate-half chain (`slice/slice/neg/concat/mul/mul/add`, plus the outer
/// `slice/concat` for a partial rope) and replace its root with the fused `OpKind::Rope { rot }` op
/// (inputs `[x, cos, sin]`), which the executor lowers to `kernelgen::rope`: one dispatch instead of
/// ~6-8. Two forms are matched. Full (`rot == D`) is rooted at the `Add` with `x == a`; it also fires on
/// the rotated core of a partial rope (`x == Slice(x0, 0, rot)`), fusing 6-7 of its dispatches even if
/// the outer slice/concat are left. Partial (`rot < D`) is rooted at the outer
/// `Concat(last, [add, Slice(x0, rot, D)])` where `add`'s `a` is `Slice(x0, 0, rot)`; it is replaced
/// with `Rope { rot }` over the full `x0` (the kernel passes `[rot, D)` through), leaving the inner
/// `Add` core dead. Conservative and result-preserving (one semantic definition shared by tracing, the oracle and every lowering). Run on a `cse`'d graph, before
/// flash/fuse; the dead intermediates need a following `dce`. It only rewrites the q/k rope, so it is
/// independent of the flash-attention match.
///
/// Also returns the near-miss `Add`s it left alone, each with its first failing check. An `Add` that is not two
/// `Mul`s around a rotate-half `Concat` is no near-miss and is not reported.
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn rope_fusion(g: &Graph) -> Result<(Graph, Vec<FusionDecline>), EditError> {
    let mut prod: HashMap<ValueId, usize> = HashMap::new();
    for (i, e) in g.eqns().iter().enumerate() {
        prod.insert(e.out, i);
    }
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns()[i]) };
    let val = |o: &Operand| -> Option<ValueId> {
        match o {
            Operand::Value(x) => Some(*x),
            Operand::Lit(_) => None,
        }
    };

    // Pass 1: partial-rope matches (rooted at the outer Concat). Records the full-`x` rewrite for the
    // Concat's out, and marks the inner Add's out as consumed so Pass 2 does not also fuse it standalone
    // (its output becomes dead once the Concat is replaced).
    let mut partial: HashMap<ValueId, RopeMatch> = HashMap::new();
    let mut consumed: HashSet<ValueId> = HashSet::new();
    for e in g.eqns() {
        let axis = match &e.op {
            OpKind::Concat { axis } if e.inputs.len() == 2 => *axis,
            _ => continue,
        };
        let _ = (|| -> Option<()> {
            let p0 = val(e.inputs.first()?)?; // the rotated Add
            let p1 = val(e.inputs.get(1)?)?; // the passthrough Slice
            let add = pe(p0)?;
            if axis != g.aval(add.out).rank().checked_sub(1)? {
                return None;
            }
            let (a, cos, sin, rot) = match_rope_core(g, &prod, add).ok()?;
            // a must be Slice(x0, last, 0, rot); the passthrough p1 must be Slice(x0, last, rot, D).
            let a_eqn = pe(a)?;
            let (x0, alast) = match &a_eqn.op {
                OpKind::Slice { axis, start, end }
                    if *start == 0
                        && *end == rot
                        && *axis == g.aval(a).rank().checked_sub(1)? =>
                {
                    (val(a_eqn.inputs.first()?)?, *axis)
                }
                _ => return None,
            };
            let d = *g.aval(x0).shape.get(alast)?;
            if rot >= d {
                return None; // not a partial rope (no passthrough)
            }
            let p1_eqn = pe(p1)?;
            match &p1_eqn.op {
                OpKind::Slice { axis, start, end }
                    if *axis == alast
                        && *start == rot
                        && *end == d
                        && val(p1_eqn.inputs.first()?)? == x0 => {}
                _ => return None,
            }
            if !rope_types_as(g, x0, cos, sin, rot, e.out) {
                return None;
            }
            partial.insert(
                e.out,
                RopeMatch {
                    x: x0,
                    cos,
                    sin,
                    rot,
                },
            );
            consumed.insert(add.out);
            Some(())
        })();
    }

    // Pass 2: emit the fused ops. Partial matches replace their Concat; every remaining Add that matches the
    // core (and was not consumed by a partial) becomes a FULL rope over `a` (rot == a's last dim).
    let mut declines = Vec::new();
    let eqns: Vec<Eqn> = g
        .eqns()
        .iter()
        .map(|e| {
            if let Some(m) = partial.get(&e.out) {
                return Eqn {
                    op: OpKind::Rope { rot: m.rot },
                    inputs: vec![
                        Operand::Value(m.x),
                        Operand::Value(m.cos),
                        Operand::Value(m.sin),
                    ],
                    out: e.out,
                    layer: e.layer,
                };
            }
            // Full rope: a matched Add core (not consumed by a partial) whose `a` IS exactly the rotated
            // tensor (last dim == rot). Otherwise the Add is the core of an unmatched partial (its output
            // shape would be `rot`, not the full head dim) - leave it for the outer slice/concat.
            if !consumed.contains(&e.out) {
                match match_rope_core(g, &prod, e) {
                    Ok((a, cos, sin, rot))
                        if g.aval(a).shape.last().copied() == Some(rot)
                            && rope_types_as(g, a, cos, sin, rot, e.out) =>
                    {
                        return Eqn {
                            op: OpKind::Rope { rot },
                            inputs: vec![
                                Operand::Value(a),
                                Operand::Value(cos),
                                Operand::Value(sin),
                            ],
                            out: e.out,
                            layer: e.layer,
                        };
                    }
                    Ok(_) => {}
                    // `RootNotAdd` and `NoRotateHalf` are every other `Add` in the graph.
                    Err(RopeDecline::RootNotAdd | RopeDecline::NoRotateHalf) => {}
                    Err(reason) => declines.push(FusionDecline {
                        value: e.out,
                        reason: FusionDeclineReason::Rope(reason),
                    }),
                }
            }
            e.clone()
        })
        .collect();
    Ok((with_eqns(g, eqns)?, declines))
}

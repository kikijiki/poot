use super::*;
use poot_target::{FLASH_LDS_CAP, FLASH_PREFILL_LDS_CAP};

/// Why the attention matcher did not fuse a candidate equation, in the order the checks run. The pass
/// leaves the equation alone and records the first failing check (it never guesses); tests assert the
/// exact reason for a near-miss.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionDecline {
    /// The root is not `MatMul(p, V)`.
    RootNotMatmul,
    /// `p` is not `Div(e, denom)` (the two-pass softmax normalizer).
    NotDiv,
    /// `denom` is not a keepdim `Sum` over the same `e` on the last axis.
    Denom,
    /// `e` is not `Exp(shifted)`.
    NotExp,
    /// `shifted` is not `Sub(scores, m)`.
    NotSub,
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
    /// A decode-shaped (`M == 1`) match carries a softcap, which `FlashAttentionDecode` cannot express.
    DecodeSoftcap,
    /// The fused op's own typing rule disagrees with the region it would replace (infer-and-compare).
    TypeMismatch,
    /// A decode head dim exceeds the fused decode kernel's LDS cap.
    HeadDimExceedsCap,
}

/// Why the rope matcher did not fuse a candidate equation. The same first-failing-check contract as
/// [`AttentionDecline`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RopeDecline {
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

/// The producer index of every value defined by an equation.
pub(super) fn producer_map<V: ValidationChannel>(g: &Graph<V>) -> HashMap<ValueId, usize> {
    g.eqns
        .iter()
        .enumerate()
        .map(|(index, eqn)| (eqn.out, index))
        .collect()
}

/// The equation indices that consume each value.
fn consumer_map<V: ValidationChannel>(g: &Graph<V>) -> Vec<Vec<usize>> {
    let mut consumers = vec![Vec::new(); g.values.len()];
    for (index, eqn) in g.eqns.iter().enumerate() {
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                consumers[*value].push(index);
            }
        }
    }
    consumers
}

fn is_liveness_root<V: ValidationChannel>(g: &Graph<V>, value: ValueId) -> bool {
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
pub fn canonicalize<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    canonicalize_with_declines(g).0
}

/// The canonical graph and typed explanations of numerical rewrites that were withheld. Value ids
/// identify the original candidate's result; a declined optimization does not refuse compilation.
pub fn canonicalize_with_declines<V: ValidationChannel>(
    g: &Graph<V>,
) -> (Graph<V>, Vec<NumericalRewriteDecline>) {
    let producers = producer_map(g);
    let consumers = consumer_map(g);
    let mut out = g.clone();
    let mut declines = Vec::new();
    for eqn in &mut out.eqns {
        if matches!(eqn.op, OpKind::Binary(BinOp::Mul))
            && two_values(&eqn.inputs).is_some()
            && eqn.inputs.iter().any(|operand| {
                let Operand::Value(value) = operand else {
                    return false;
                };
                producers
                    .get(value)
                    .is_some_and(|&index| matches!(g.eqns[index].op, OpKind::Unary(UnOp::Recip)))
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
            let mul = &g.eqns[index];
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
    (out, declines)
}
/// A matched materialized softmax-attention subgraph (the operands of the fused op it collapses to).
pub(super) struct AttnMatch {
    q: ValueId,
    k: ValueId, // pre-GQA-repeat key
    v: ValueId, // pre-GQA-repeat value
    mask: ValueId,
    n_rep: usize,
    scale: f32,
    decode: bool, // M==1 (decode) vs M>1 (prefill)
    /// Card 198: `Some(c)` when the scaled `Q K^T` passes through a Gemma2/Grok attention-logit softcap
    /// (`c * tanh(scores / c)`, [`poot_graph_ir::ops::softcap`]) before the mask-add. Always `None` when `decode`
    /// is true: `FlashAttentionDecode` has no softcap parameter, so [`match_attention`] refuses to match
    /// a softcapped decode chain (falling back to the decomposition) rather than dropping the softcap.
    softcap: Option<f32>,
}

/// Peel a `repeat_kv(x, n_rep)` chain (`reshape(broadcast(reshape(x)))`) back to `(x, n_rep)`. If `v` is not
/// produced by that exact 3-op chain (the `n_rep == 1` case emits nothing), return `(v, 1)`.
fn peel_repeat_kv<V: ValidationChannel>(
    g: &Graph<V>,
    prod: &HashMap<ValueId, usize>,
    v: ValueId,
) -> (ValueId, usize) {
    let pe = |x: ValueId| -> Option<&Eqn> { prod.get(&x).map(|&i| &g.eqns[i]) };
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
fn match_pre_mask<V: ValidationChannel>(
    g: &Graph<V>,
    prod: &HashMap<ValueId, usize>,
    pre_mask: ValueId,
) -> Result<(ValueId, f32, Option<f32>), AttentionDecline> {
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns[i]) };
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
pub(super) fn match_attention<V: ValidationChannel>(
    g: &Graph<V>,
    prod: &HashMap<ValueId, usize>,
    root: &Eqn,
) -> Result<AttnMatch, AttentionDecline> {
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns[i]) };
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

    // denom = ReduceSum(e, last, keepdim) over the SAME e
    let denom_eqn = pe(denom).ok_or(AttentionDecline::Denom)?;
    match &denom_eqn.op {
        OpKind::Reduce {
            op: RedOp::Sum,
            axis,
            keepdim: true,
        } if val(denom_eqn.inputs.first().ok_or(AttentionDecline::Denom)?) == Some(e)
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

    // m = ReduceMax(scores, last, keepdim) over the SAME scores
    let m_eqn = pe(m).ok_or(AttentionDecline::RowMax)?;
    match &m_eqn.op {
        OpKind::Reduce {
            op: RedOp::Max,
            axis,
            keepdim: true,
        } if val(m_eqn.inputs.first().ok_or(AttentionDecline::RowMax)?) == Some(scores)
            && *axis == last_axis(scores) => {}
        _ => return Err(AttentionDecline::RowMax),
    }

    // scores = Add(pre_mask, mask). Either operand may be the additive mask (R467-005): the scaled
    // product is the one that decomposes to a `Q K^T`, the other is the mask.
    let add = pe(scores).ok_or(AttentionDecline::ScoreAdd)?;
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

    let (k, n_rep_k) = peel_repeat_kv(g, prod, krep);
    let (v, n_rep_v) = peel_repeat_kv(g, prod, vrep);
    if n_rep_k != n_rep_v {
        return Err(AttentionDecline::GqaRepeatMismatch);
    }

    let decode = g.aval(q).shape.get(2).copied() == Some(1);
    // Card 198: `FlashAttentionDecode` has no softcap parameter (see `AttnMatch::softcap`'s doc). A
    // softcapped decode-shaped chain must fall back to the decomposition rather than fuse with the
    // softcap dropped.
    if decode && softcap.is_some() {
        return Err(AttentionDecline::DecodeSoftcap);
    }
    // Head-dim awareness (R480-009): the fused decode kernel's LDS scratch is sized to `FLASH_LDS_CAP`
    // and the D>cap decode fallback is not portable, so a decode-shaped match wider than the cap stays
    // the decomposition instead of producing an op the planner refuses (or a kernel that crashes) on
    // some backend. Gemma4's per-layer shapes (local 256, global 512) then fuse where they fit.
    let head_dim = g.aval(q).shape.get(3).copied().unwrap_or(0);
    let head_cap = if decode {
        FLASH_LDS_CAP
    } else {
        FLASH_PREFILL_LDS_CAP
    };
    if head_dim > head_cap {
        return Err(AttentionDecline::HeadDimExceedsCap);
    }

    let m = AttnMatch {
        q,
        k,
        v,
        mask,
        n_rep: n_rep_k,
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
    let operands = [q, k, v, mask].map(|x| g.aval(x).clone());
    if m.op().infer(&operands).ok().as_ref() != Some(g.aval(root.out)) {
        return Err(AttentionDecline::TypeMismatch);
    }
    Ok(m)
}

impl AttnMatch {
    /// The fused op this match collapses to.
    fn op(&self) -> OpKind {
        if self.decode {
            OpKind::FlashAttentionDecode {
                n_rep: self.n_rep,
                scale: self.scale,
            }
        } else {
            OpKind::FlashAttentionPrefill {
                n_rep: self.n_rep,
                scale: self.scale,
                softcap: self.softcap,
            }
        }
    }
}

/// Flash-attention fusion (automatic fusion as a compiler pass), capped by `max_prefill_groups`: a
/// `FlashAttentionPrefill` is emitted only when its `Hq*L` workgroup count is `<= max_prefill_groups`
/// (decode, `Hq` workgroups, is always emitted); `None` means no cap. Pattern-match the materialized
/// `softmax(scale * Q Kᵀ + mask) @ V` subgraph (with GQA repeat) that `ops::attention_masked` /
/// `ops::attention_prefill` produce, and replace its root with the fused `FlashAttentionDecode` (M==1
/// queries) or `FlashAttentionPrefill` (M>1) op, which the executor lowers to the imported flash kernel
/// (online softmax, never materializing the `[.,.,L,L]` scores). The replaced intermediates become
/// dead; run `dce` after. Conservative (only the exact chain matches) and result-preserving (one
/// semantic definition shared by tracing, the oracle and every lowering), so every attention tracer
/// gets flash without a separate `_flash` variant. Best run on a `cse`'d graph. `compile` passes its own
/// measured `target.caps.max_grid[0]` (wgpu's `gridDim.x` caps at 65535 and has no plan-time fallback
/// from the op to the decomposition).
pub fn flash_attention_capped<V: ValidationChannel>(
    g: &Graph<V>,
    max_prefill_groups: Option<usize>,
) -> Graph<V> {
    let mut prod: HashMap<ValueId, usize> = HashMap::new();
    for (i, e) in g.eqns.iter().enumerate() {
        prod.insert(e.out, i);
    }
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
    let eqns: Vec<Eqn> = g
        .eqns
        .iter()
        .map(|e| match match_attention(g, &prod, e) {
            Ok(m) if fits(&m) => Eqn {
                op: m.op(),
                inputs: vec![
                    Operand::Value(m.q),
                    Operand::Value(m.k),
                    Operand::Value(m.v),
                    Operand::Value(m.mask),
                ],
                out: e.out,
                layer: e.layer,
            },
            _ => e.clone(),
        })
        .collect();
    Graph { eqns, ..g.clone() }
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
pub(super) fn match_rope_core<V: ValidationChannel>(
    g: &Graph<V>,
    prod: &HashMap<ValueId, usize>,
    add: &Eqn,
) -> Result<(ValueId, ValueId, ValueId, usize), RopeDecline> {
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns[i]) };
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
fn rope_types_as<V: ValidationChannel>(
    g: &Graph<V>,
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
pub fn rope_fusion<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let mut prod: HashMap<ValueId, usize> = HashMap::new();
    for (i, e) in g.eqns.iter().enumerate() {
        prod.insert(e.out, i);
    }
    let pe = |v: ValueId| -> Option<&Eqn> { prod.get(&v).map(|&i| &g.eqns[i]) };
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
    for e in &g.eqns {
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
    let eqns: Vec<Eqn> = g
        .eqns
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
            if !consumed.contains(&e.out)
                && let Ok((a, cos, sin, rot)) = match_rope_core(g, &prod, e)
                && g.aval(a).shape.last().copied() == Some(rot)
                && rope_types_as(g, a, cos, sin, rot, e.out)
            {
                return Eqn {
                    op: OpKind::Rope { rot },
                    inputs: vec![Operand::Value(a), Operand::Value(cos), Operand::Value(sin)],
                    out: e.out,
                    layer: e.layer,
                };
            }
            e.clone()
        })
        .collect();
    Graph { eqns, ..g.clone() }
}

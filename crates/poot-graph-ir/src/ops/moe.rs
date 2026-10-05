use super::*;

/// Canonicalize router scores before ranking or weighting. Every finite `f32` is unchanged, NaN and
/// negative infinity become `f32::MIN`, and positive infinity becomes `f32::MAX`. `BinOp::Max` has
/// max-number semantics in the eager evaluator and LLVM lowering, so this stays backend-neutral and makes
/// every score participating in the route finite.
pub fn canonical_router_scores(b: &Builder, x: Traced) -> Traced {
    let lower = b.binary_scalar(BinOp::Max, x, Scalar::F32(f32::MIN));
    let neg = b.binary_scalar(BinOp::Mul, lower, Scalar::F32(-1.0));
    let neg_capped = b.binary_scalar(BinOp::Max, neg, Scalar::F32(f32::MIN));
    b.binary_scalar(BinOp::Mul, neg_capped, Scalar::F32(-1.0))
}

/// Stable descending rank along the trailing axis, expressed only with primitive graph operations.
/// `rank[..,i]` counts scores strictly greater than score `i`, plus equal scores at lower expert indices.
/// The result is therefore a permutation of `0..E` for every row, including tied and canonicalized
/// non-finite inputs. An `iota` range (card 558a) supplies the index tie-break.
pub fn stable_descending_rank(b: &Builder, x: Traced) -> Traced {
    let shape = b.aval(x).shape;
    let e = *shape
        .last()
        .expect("stable_descending_rank input has an expert axis");
    assert!(e > 0, "stable_descending_rank needs at least one expert");
    let last = shape.len() - 1;
    let scores = canonical_router_scores(b, x);

    // Pairwise scores: P[..,i,j] = score_i, Q[..,i,j] = score_j.
    let mut pair_shape = shape.clone();
    pair_shape.push(e); // [.., E, E]
    let mut p_shape = shape.clone();
    p_shape.push(1); // [.., E, 1]
    let p = b.broadcast(b.reshape(scores, p_shape.clone()), pair_shape.clone());
    let mut q_shape = shape.clone();
    q_shape[last] = 1;
    q_shape.push(e); // [.., 1, E]
    let q = b.broadcast(b.reshape(scores, q_shape.clone()), pair_shape.clone());

    // q > p = (q >= p) and not (p >= q). Equality is both comparisons true.
    let pgeq = b.binary(BinOp::Ge, p, q);
    let qgep = b.binary(BinOp::Ge, q, p);
    let not_pgeq = b.binary_scalar(BinOp::Mul, pgeq, Scalar::F32(-1.0));
    let not_pgeq = b.binary_scalar(BinOp::Add, not_pgeq, Scalar::F32(1.0));
    let greater = b.binary(BinOp::Mul, qgep, not_pgeq);
    let equal = b.binary(BinOp::Mul, pgeq, qgep);

    // On equal scores, expert j precedes expert i iff j < i. Indices are exact, finite, and distinct.
    // Card 558a: compute the range with `iota` (no bound named-constant range); the F32 cast keeps
    // the rank arithmetic in f32 until the index migration of Card 558b.
    let iota = b.iota(e);
    let mut iota_shape = vec![1usize; shape.len()];
    iota_shape[last] = e;
    let iota = b.broadcast(b.reshape(iota, iota_shape), shape);
    let i_idx = b.broadcast(b.reshape(iota, p_shape), pair_shape.clone());
    let j_idx = b.broadcast(b.reshape(iota, q_shape), pair_shape);
    let jgei = b.binary(BinOp::Ge, j_idx, i_idx);
    let j_lt_i = b.binary_scalar(BinOp::Mul, jgei, Scalar::F32(-1.0));
    let j_lt_i = b.binary_scalar(BinOp::Add, j_lt_i, Scalar::F32(1.0));
    let earlier_equal = b.binary(BinOp::Mul, equal, j_lt_i);
    let precedes = b.binary(BinOp::Add, greater, earlier_equal);
    b.reduce(RedOp::Sum, precedes, last + 1, false)
}

/// Stable descending rank restricted by an exact `0.0`/`1.0` eligibility mask of the same shape.
/// Eligible entries always precede ineligible entries, then canonical score descending and lower index
/// break ties within each class. The result remains a permutation of `0..E`; this is the group-limited MoE
/// equivalent of setting excluded softmax probabilities to zero, without relying on an additive sentinel
/// that can collide with `f32::MIN`.
pub fn stable_descending_rank_masked(b: &Builder, x: Traced, eligible: Traced) -> Traced {
    let shape = b.aval(x).shape;
    assert_eq!(
        b.aval(eligible).shape,
        shape,
        "stable_descending_rank_masked needs score and mask shapes to match"
    );
    let e = *shape
        .last()
        .expect("stable_descending_rank_masked input has an expert axis");
    assert!(
        e > 0,
        "stable_descending_rank_masked needs at least one expert"
    );
    let last = shape.len() - 1;
    let scores = canonical_router_scores(b, x);

    let mut pair_shape = shape.clone();
    pair_shape.push(e);
    let mut p_shape = shape.clone();
    p_shape.push(1);
    let mut q_shape = shape.clone();
    q_shape[last] = 1;
    q_shape.push(e);

    let p = b.broadcast(b.reshape(scores, p_shape.clone()), pair_shape.clone());
    let q = b.broadcast(b.reshape(scores, q_shape.clone()), pair_shape.clone());
    let pgeq = b.binary(BinOp::Ge, p, q);
    let qgep = b.binary(BinOp::Ge, q, p);
    let not_pgeq = b.binary_scalar(BinOp::Mul, pgeq, Scalar::F32(-1.0));
    let not_pgeq = b.binary_scalar(BinOp::Add, not_pgeq, Scalar::F32(1.0));
    let greater = b.binary(BinOp::Mul, qgep, not_pgeq);
    let equal = b.binary(BinOp::Mul, pgeq, qgep);

    // Card 558a: computed via `iota`, not a bound named-constant range - see `stable_descending_rank`.
    let iota = b.iota(e);
    let mut iota_shape = vec![1usize; shape.len()];
    iota_shape[last] = e;
    let iota = b.broadcast(b.reshape(iota, iota_shape), shape.clone());
    let i_idx = b.broadcast(b.reshape(iota, p_shape.clone()), pair_shape.clone());
    let j_idx = b.broadcast(b.reshape(iota, q_shape.clone()), pair_shape.clone());
    let jgei = b.binary(BinOp::Ge, j_idx, i_idx);
    let j_lt_i = b.binary_scalar(BinOp::Mul, jgei, Scalar::F32(-1.0));
    let j_lt_i = b.binary_scalar(BinOp::Add, j_lt_i, Scalar::F32(1.0));
    let earlier_equal = b.binary(BinOp::Mul, equal, j_lt_i);
    let score_precedes = b.binary(BinOp::Add, greater, earlier_equal);

    let p_keep = b.broadcast(b.reshape(eligible, p_shape), pair_shape.clone());
    let q_keep = b.broadcast(b.reshape(eligible, q_shape), pair_shape);
    let pgeq = b.binary(BinOp::Ge, p_keep, q_keep);
    let qgep = b.binary(BinOp::Ge, q_keep, p_keep);
    let same_class = b.binary(BinOp::Mul, pgeq, qgep);
    let not_p_keep = b.binary_scalar(BinOp::Mul, p_keep, Scalar::F32(-1.0));
    let not_p_keep = b.binary_scalar(BinOp::Add, not_p_keep, Scalar::F32(1.0));
    let eligible_precedes = b.binary(BinOp::Mul, q_keep, not_p_keep);
    let within_class = b.binary(BinOp::Mul, same_class, score_precedes);
    let precedes = b.binary(BinOp::Add, eligible_precedes, within_class);
    b.reduce(RedOp::Sum, precedes, last + 1, false)
}

/// Exact `0.0`/`1.0` mask for the first `k` positions of a stable rank tensor.
pub fn top_k_keep_mask(b: &Builder, rank: Traced, k: usize) -> Traced {
    let rge = b.binary_scalar(BinOp::Ge, rank, Scalar::F32(k as f32));
    let keep = b.binary_scalar(BinOp::Mul, rge, Scalar::F32(-1.0));
    b.binary_scalar(BinOp::Add, keep, Scalar::F32(1.0))
}

pub(crate) fn assert_valid_top_k(e: usize, k: usize, op: &str) {
    assert!(
        (1..=e).contains(&k),
        "{op} needs 1 <= k <= E, got k={k}, E={e}"
    );
}

pub(crate) fn normalized_top_k_gate_from_rank(
    b: &Builder,
    x: Traced,
    rank: Traced,
    k: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let last = shape.len() - 1;
    let scores = canonical_router_scores(b, x);
    let keep = top_k_keep_mask(b, rank, k);

    // The global maximum is selected for every valid k. Multiplication by the exact mask makes dropped
    // terms exactly zero and the denominator contains exactly the selected k terms.
    let m = b.reduce(RedOp::Max, scores, last, true);
    let shifted = b.binary(BinOp::Sub, scores, m);
    let ex = b.unary(UnOp::Exp, shifted);
    let selected = b.binary(BinOp::Mul, ex, keep);
    let denom = b.reduce(RedOp::Sum, selected, last, true);
    b.binary(BinOp::Div, selected, denom)
}

/// MoE top-k gating as a primitive composition: router scores `[.., E]` -> per-row gate weights
/// `[.., E]`. Scores rank descending with lower expert index first on ties. Exactly `k` distinct experts
/// are selected for `1 <= k <= E`; only those terms enter the softmax denominator and every other weight
/// is exactly zero. See [`canonical_router_scores`] for the explicit non-finite policy.
pub fn top_k_gate(b: &Builder, x: Traced, k: usize) -> Traced {
    let shape = b.aval(x).shape;
    let e = *shape.last().expect("top_k_gate input has an expert axis");
    assert_valid_top_k(e, k, "top_k_gate");
    let rank = stable_descending_rank(b, x);
    normalized_top_k_gate_from_rank(b, x, rank, k)
}

/// Mixture-of-experts MLP. Decode (L=1) uses the sparse form ([`moe_sparse`], only the `k` selected
/// experts' FFNs); prefill (L>1) uses [`moe_grouped`] (spec 136 Tier 1: flatten to `M = L*k` rows and
/// reuse the gather-free `indexed_matmul`, `~E/k` fewer expert FLOPs than evaluating every expert
/// densely). `moe_sparse` is gather-free (card 088: `indexed_matmul` reads the selected expert's rows
/// in-kernel, no weight-gather copy). Measured on RADV STRIX_HALO (`moe_sparse_vs_dense_decode_timing`,
/// resident path, E=32/k=8/H=1024/I=512): sparse 1.49 vs dense 1.73 ms/fwd = 1.16x with identical
/// output (`moe_sparse_gpu_matches_dense`); the win grows with E/k. `indexed_matmul_dt` is still a
/// naive one-thread-per-output kernel for M=1; an indexed LDS-GEMV would widen the gap. One MoE
/// definition shared by every arch's tracer.
///
/// Shapes: `x[1, L, H]`; `router_w[H, E]`; `w_in[E, H, 2*I]` (fused gate||up, pre-transposed to `[in, out]`
/// at load); `w_out[E, I, H]` (per-expert down, pre-transposed). Returns `[1, L, H]`.
#[allow(clippy::too_many_arguments)]
pub fn moe(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    w_in: Traced,
    w_out: Traced,
    n_experts: usize,
    k: usize,
    inter: usize,
) -> Traced {
    // Decode (L=1): gather-free sparse form (card 088). Prefill (L>1): grouped form (spec 136),
    // flattening to `M = L*k` rows over the same `indexed_matmul`. `moe_dense` is kept only as the
    // CPU-eval correctness oracle. L is static here.
    let shape = b.aval(x).shape;
    let l = shape[shape.len() - 2];
    if l == 1 {
        moe_sparse(b, x, router_w, w_in, w_out, n_experts, k, inter)
    } else {
        moe_grouped(b, x, router_w, w_in, w_out, n_experts, k, inter)
    }
}

/// Dense MoE: every expert's swiglu FFN is evaluated and weighted by its gate (0 outside the top-k),
/// then summed over the expert axis. Shape-static (no data-dependent gather); `E/k` times the FLOPs
/// of [`moe_sparse`]. Not called by [`moe`] (spec 136 fork b); kept as the CPU-eval correctness
/// oracle that [`moe_grouped`] is checked against (`moe_grouped_matches_dense`), since the grouped
/// summation order differs (`~1e-5`, not bit-exact).
#[allow(clippy::too_many_arguments)]
pub fn moe_dense(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    w_in: Traced,
    w_out: Traced,
    n_experts: usize,
    k: usize,
    inter: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let (h, l) = (shape[shape.len() - 1], shape[shape.len() - 2]);
    let xm = b.reshape(x, vec![l, h]);

    // route: logits [L, E] -> per-row top-k softmax gate weights [L, E].
    let logits = linear(b, xm, router_w, None);
    let gate = top_k_gate(b, logits, k); // [L, E]

    // every expert's swiglu FFN, batched over the expert axis (broadcast the tokens across experts).
    let x_e = b.broadcast(b.reshape(xm, vec![1, l, h]), vec![n_experts, l, h]); // [E, L, H]
    let gu = b.matmul(x_e, w_in); // [E, L, 2I]
    let g_part = b.slice(gu, 2, 0, inter); // [E, L, I]
    let u_part = b.slice(gu, 2, inter, 2 * inter); // [E, L, I]
    let act = swiglu(b, g_part, u_part); // [E, L, I]
    let out = b.matmul(act, w_out); // [E, L, H]

    // weight each expert's output by its gate value and sum over the expert axis. Experts move to the
    // LAST axis first ([E,L,H] -> [L,H,E]) because the GPU executor only reduces the last axis; the
    // result equals an axis-0 reduce.
    let gate_w = b.reshape(b.transpose(gate, vec![1, 0]), vec![n_experts, l, 1]); // [E, L, 1]
    let weighted = b.binary(BinOp::Mul, out, gate_w); // [E, L, H]
    let weighted_t = b.transpose(weighted, vec![1, 2, 0]); // [L, H, E]
    let y = b.reduce(RedOp::Sum, weighted_t, 2, false); // [L, H]
    b.reshape(y, vec![1, l, h])
}

/// Sparse MoE decode (card 034): the FLOPs-reduced form of [`moe`] for one token (`x[1,1,H]`).
/// Computes only the token's `k` selected experts' FFNs. Output matches [`moe`] (the dense form
/// zero-weights the `E-k` others) at `~E/k` the expert FLOPs.
///
/// Routing is the `L=1` case of [`moe_grouped_prep`], so dense, sparse, grouped, router-only, and
/// pooled paths share the same stable rank, exact mask, and normalized selected weights. Needs the
/// shared `iota[E]` range for the lower-index tie-break and gate-slot reconstruction.
#[allow(clippy::too_many_arguments)]
pub fn moe_sparse(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    w_in: Traced,
    w_out: Traced,
    n_experts: usize,
    k: usize,
    inter: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let (h, l) = (shape[shape.len() - 1], shape[shape.len() - 2]);
    assert_eq!(
        l, 1,
        "moe_sparse is the decode (L=1) path; use moe for prefill"
    );
    let e = n_experts;
    let xm = b.reshape(x, vec![1, h]); // [1, H]
    let logits = linear(b, xm, router_w, None); // [1, E]
    let (x_rep, idx_k, gate_k) = moe_grouped_prep(b, xm, logits, e, k);

    // run the k FFNs: each of the k rows is the SAME token x, routed to its OWN selected expert via idx_k.
    let gu = b.indexed_matmul(x_rep, w_in, idx_k); // [k, 2I]: row t = x @ w_in[idx_k[t]]
    let g_part = b.slice(gu, 1, 0, inter); // [k, I]
    let u_part = b.slice(gu, 1, inter, 2 * inter); // [k, I]
    let act = swiglu(b, g_part, u_part); // [k, I]
    let out = b.indexed_matmul(act, w_out, idx_k); // [k, H]: row t = act[t] @ w_out[idx_k[t]]

    // weight by the selected gate values and sum over the k experts (move k to the last axis for the reduce).
    let gate_kw = b.reshape(gate_k, vec![k, 1]); // [k, 1]
    let weighted = b.binary(BinOp::Mul, out, gate_kw); // [k, H]
    let weighted_t = b.transpose(weighted, vec![1, 0]); // [H, k]
    let y = b.reduce(RedOp::Sum, weighted_t, 1, false); // [H]
    b.reshape(y, vec![1, 1, h])
}

/// Grouped MoE prefill routing (spec 136 Tier 1 + P4, FR-009): computes the per-token top-k routing
/// and flattens `(token, slot)` to `M = L*k` rows. Factored out of [`moe_grouped`] as the one
/// decomposition the f32 grouped path builds on; the expert matmuls (`indexed_matmul`) stay in each
/// caller.
///
/// Decomposition (all shapes static from `(L,E,k,H)`, no data-dependent shape - FR-007):
/// 1. `rank[L,E]`: [`stable_descending_rank`], shared with [`top_k_gate`]. Scores rank descending and lower
///    expert index breaks ties, so every row is a permutation of `0..E`.
/// 2. `idx[L,k] = ArgTopK(rank, k)`: the per-token top-k expert ids (F32, fork c), generalizing
///    `moe_sparse`'s `scatter(iota,rank)[0..k]` to the leading `L` axis (fork a: rank-in, not logits-in).
/// 3. `gate[L,E]`: [`top_k_gate`]'s exact selected-only normalization on the same shared rank.
/// 4. `gate_sorted[L,k]` (B1, FR-010): there is no batched take-along-axis primitive (`Gather` shares one
///    index set across all leading rows), so the per-slot gate value is selected via a one-hot over the
///    shared rank: `gate_sorted[l,r] = sum_e eq(rank[l,e],r) * gate[l,e]`, `eq(a,b) = ge(a,b)*ge(b,a)` (exact
///    for the integer-valued rank; only `Ge` exists in `BinOp`). The `r in 0..k` values reuse the existing
///    `iota[E]` range ([`moe_sparse`] already requires it) sliced to `[0..k]` - no new equation.
/// 5. Flatten to `M = L*k` rows, row `l*k+s` = `(token l, slot s)` consistently across all three flattened
///    tensors (row-major reshape of a shared `[L,k,..]` shape): `idx_flat[M]`, `x_flat[M,H]` (token `x`
///    broadcast over the `k` slots), `gate_flat[M]`.
///
/// `xm` is the already-flattened `[L, H]` token matrix; `logits` is `linear(xm, router_w)`, `[L, E]`.
/// Returns `(x_flat[M,H], idx_flat[M], gate_flat[M])`, `M = L*k`.
pub fn moe_grouped_prep(
    b: &Builder,
    xm: Traced,
    logits: Traced,
    n_experts: usize,
    k: usize,
) -> (Traced, Traced, Traced) {
    let xm_shape = b.aval(xm).shape;
    let (l, h) = (xm_shape[0], xm_shape[1]);
    let e = n_experts;
    assert_valid_top_k(e, k, "moe_grouped_prep");

    // 1. Stable descending rank shared by every normalized route.
    let rank = stable_descending_rank(b, logits); // [L, E]

    // 2. per-token top-k expert ids, rank-ordered (fork a: rank-in; fork c: F32 ids).
    let idx = b.arg_top_k(rank, k); // [L, k]

    // 3. Exact selected-only normalization on the same rank.
    let gate = normalized_top_k_gate_from_rank(b, logits, rank, k); // [L, E]

    // 4. B1: gate_sorted[l,r] = sum_e eq(rank[l,e],r) * gate[l,e], via a one-hot over the shared rank.
    //    `r` ranges over 0..k; compute it with `iota` (card 558a), sliced to its first k entries.
    let iota_e = b.iota(e);
    let r_iota = b.slice(iota_e, 0, 0, k); // [k]: 0..k
    let rank_b = b.broadcast(b.reshape(rank, vec![l, 1, e]), vec![l, k, e]); // [L,k,E]: rank[l,e]
    let r_b = b.broadcast(b.reshape(r_iota, vec![1, k, 1]), vec![l, k, e]); // [L,k,E]: r
    let ge1 = b.binary(BinOp::Ge, rank_b, r_b); // rank[l,e] >= r
    let ge2 = b.binary(BinOp::Ge, r_b, rank_b); // r >= rank[l,e]
    let eq = b.binary(BinOp::Mul, ge1, ge2); // eq(rank[l,e], r): exact for integer-valued rank
    let gate_b = b.broadcast(b.reshape(gate, vec![l, 1, e]), vec![l, k, e]); // [L,k,E]: gate[l,e]
    let sel = b.binary(BinOp::Mul, eq, gate_b); // [L,k,E]
    let gate_sorted = b.reduce(RedOp::Sum, sel, 2, false); // [L, k]: gate value at (token l, slot r)

    // 5. flatten (token, slot) -> M = L*k rows, row l*k+s consistently across idx/x/gate (all reshape a
    //    shared [L,k,..] leading shape row-major, so row l*k+s always means (token l, slot s)).
    let m_rows = l * k;
    let idx_flat = b.reshape(idx, vec![m_rows]); // [M]
    let x_rep = b.broadcast(b.reshape(xm, vec![l, 1, h]), vec![l, k, h]); // [L,k,H]: token l repeated k times
    let x_flat = b.reshape(x_rep, vec![m_rows, h]); // [M, H]
    let gate_flat = b.reshape(gate_sorted, vec![m_rows]); // [M]

    (x_flat, idx_flat, gate_flat)
}

/// Grouped MoE prefill (spec 136 Tier 1, card 146): the FLOPs-reduced form of [`moe`] for `L > 1`
/// tokens. Flattens `(token, slot)` pairs to `M = L*k` rows and reuses the gather-free
/// `indexed_matmul` (as [`moe_sparse`] does for one token) instead of evaluating every expert densely
/// (`moe_dense`, `~E/k` more FLOPs). Matches `moe_dense` up to summation order (dense reduces over
/// `E`, grouped over `k`): within `~1e-5`, not bit-exact (`moe_grouped_matches_dense`).
///
/// Routing + flatten (steps 1-5) are [`moe_grouped_prep`]. This adds:
/// 6. The two existing indexed matmuls + swiglu, exactly as `moe_sparse`'s per-token FFN, but over all `M`
///    rows at once: `gu = indexed_matmul(x_flat,w_in,idx_flat)` -> swiglu -> `indexed_matmul(_,w_out,idx_flat)`.
/// 7. Gate-weight, reshape to `[L,k,H]`, and reduce over `k` via TRANSPOSE-TO-LAST-AXIS (S3, since the GPU
///    executor only reduces the last axis): `[L,k,H] -> transpose [L,H,k] -> reduce(Sum,axis=2) -> [L,H]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    w_in: Traced,
    w_out: Traced,
    n_experts: usize,
    k: usize,
    inter: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let (h, l) = (shape[shape.len() - 1], shape[shape.len() - 2]);
    let xm = b.reshape(x, vec![l, h]); // [L, H]
    let logits = linear(b, xm, router_w, None); // [L, E]
    let (x_flat, idx_flat, gate_flat) = moe_grouped_prep(b, xm, logits, n_experts, k);

    // 6. the gather-free indexed FFN, over all M rows at once (same primitive as moe_sparse, batched).
    let gu = b.indexed_matmul(x_flat, w_in, idx_flat); // [M, 2I]: row m = x_flat[m] @ w_in[idx_flat[m]]
    let g_part = b.slice(gu, 1, 0, inter); // [M, I]
    let u_part = b.slice(gu, 1, inter, 2 * inter); // [M, I]
    let act = swiglu(b, g_part, u_part); // [M, I]
    let out = b.indexed_matmul(act, w_out, idx_flat); // [M, H]: row m = act[m] @ w_out[idx_flat[m]]

    // 7. weight by the selected gate, reshape to [L,k,H], and reduce over k via transpose-to-last-axis (S3)
    //    since the GPU executor only reduces the last axis.
    let m_rows = l * k;
    let gate_w = b.reshape(gate_flat, vec![m_rows, 1]); // [M, 1]
    let weighted = b.binary(BinOp::Mul, out, gate_w); // [M, H]
    let weighted3 = b.reshape(weighted, vec![l, k, h]); // [L, k, H]
    let weighted_t = b.transpose(weighted3, vec![0, 2, 1]); // [L, H, k]
    let y = b.reduce(RedOp::Sum, weighted_t, 2, false); // [L, H]
    b.reshape(y, vec![1, l, h])
}

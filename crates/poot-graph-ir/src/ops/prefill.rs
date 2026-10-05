use super::*;

// --- prefill variants (process the whole prompt in one forward; causal mask + per-position RoPE) ---

/// RoPE for a full sequence `x[.., L, D]` at positions `0..L`. cos/sin are `[max_pos, D]`; rows `0..L`
/// are sliced and broadcast over the head/batch axes.
pub fn rope_prefill(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    seq_len: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let d = *shape.last().expect("rope on a scalar");
    let last = shape.len() - 1;
    let rot = *b.aval(cos_table).shape.last().expect("rope table rank>=1");
    let cos = b.slice(cos_table, 0, 0, seq_len); // [L, rot]
    let sin = b.slice(sin_table, 0, 0, seq_len);
    rope_partial(b, x, cos, sin, d, rot, last)
}

/// Prefill attention over a full sequence: `q[1,Hq,L,D]`, `k`/`v[1,Hkv,L,D]`, additive causal `mask`
/// (`[1,1,L,L]`, 0 on/below the diagonal, large-negative above) added before softmax. Returns
/// `[1,Hq,L,D]`.
pub fn attention_prefill(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
) -> Traced {
    attention_prefill_softcap(b, q, k, v, n_rep, scale, mask, None)
}

/// [`attention_prefill`] with optional Gemma2/Grok attention-logit softcapping. `Some(c)` applies
/// `c * tanh(scores / c)` ([`softcap`]) to the scaled `Q K^T` scores BEFORE the causal mask and softmax;
/// `None` is byte-identical to [`attention_prefill`] (no extra eqns). One definition.
#[allow(clippy::too_many_arguments)]
pub fn attention_prefill_softcap(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    scale: f32,
    mask: Traced,
    attn_logit_softcap: Option<f32>,
) -> Traced {
    let k = repeat_kv(b, k, n_rep);
    let v = repeat_kv(b, v, n_rep);
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D,L]
    let scores = b.matmul(q, kt); // [1,Hq,L,L]
    let scores = b.binary_scalar(BinOp::Mul, scores, Scalar::F32(scale));
    let scores = match attn_logit_softcap {
        Some(c) => softcap(b, scores, c),
        None => scores,
    };
    let scores = b.binary(BinOp::Add, scores, mask); // causal mask, broadcast over Hq
    let last = b.aval(scores).rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true);
    let shifted = b.binary(BinOp::Sub, scores, m);
    let e = b.unary(UnOp::Exp, shifted);
    let denom = b.reduce(RedOp::Sum, e, last, true);
    let p = b.binary(BinOp::Div, e, denom);
    b.matmul(p, v) // [1,Hq,L,D]
}

/// Two-pass softmax over the LAST axis of `x` (card 099a reference): `exp(x - max) / sum(exp(x - max))`.
pub fn softmax(b: &Builder, x: Traced) -> Traced {
    let last = b.aval(x).rank() - 1;
    let m = b.reduce(RedOp::Max, x, last, true); // [.., 1]
    let e = b.unary(UnOp::Exp, b.binary(BinOp::Sub, x, m)); // [.., W]
    let denom = b.reduce(RedOp::Sum, e, last, true); // [.., 1]
    b.binary(BinOp::Div, e, denom)
}

/// In-graph additive causal/sliding-window attention mask from absolute positions (Card 550): `pos` is `[rows, tokens]` I32 absolute positions (`Slot::Pos`); `cap` is the key axis length
/// (the prompt length `L` for a one-shot prefill that starts at position 0, or a persistent cache's
/// capacity for decode/chunked continuation). Returns `[rows, 1, tokens, cap]` f32, broadcast over
/// heads: `0.0` where key `j` is visible to query `(r, i)` (`j <= pos[r,i]`, and `pos[r,i] - j < window`
/// when `window` is given), `-1.0e9` elsewhere. `-1.0e9` is the same magnitude the pre-card decode path's
/// `decode_mask_row` used; the pre-card prefill path's `prefill_causal_mask` used `-1.0e30` instead -
/// this helper unifies both call sites on one magnitude rather than carrying the old distinction
/// forward. Softmax flushes either choice to an exact-zero weight (`exp(mask - row_max)` underflows to
/// `0.0f32` for any row max a real attention score produces; see
/// `causal_mask_from_pos_prefill_shape_matches_visibility_pattern_and_flushes_either_magnitude` in
/// `poot-eval`'s test suite for the numeric proof), so the value is not load-bearing beyond "large
/// enough to underflow `exp` after the row max is subtracted".
///
/// Column `j`'s absolute key position is `j` itself (`iota(cap)`): every caller of this helper either
/// starts a one-shot prefill at position 0 (so `cap = L` and column `j` IS position `j`) or reads a
/// persistent `[.., cap, ..]` KV cache written at its absolute position (so column `j` holds the token
/// written at position `j`). A `Cast(I32 -> F32)` plus a signed `Ge` compare against `iota(cap)`, not a
/// gather: `pos` is cast once to F32 ([`Builder::cast`] lowers `I32 -> F32` on every backend) and
/// compared against `iota(cap)` ([`Builder::iota`], card 558a) with [`BinOp::Ge`].
pub fn causal_mask_from_pos(b: &Builder, pos: Traced, cap: usize, window: Option<usize>) -> Traced {
    let (visible, ..) = mask_visibility(b, pos, cap, window);
    let bias = b.binary_scalar(BinOp::Sub, visible, Scalar::F32(1.0)); // 0.0 visible, -1.0 masked
    b.binary_scalar(BinOp::Mul, bias, Scalar::F32(1.0e9)) // 0.0 visible, -1.0e9 masked
}

/// [`causal_mask_from_pos`] plus a per-head ALiBi (Press et al.) linear-distance bias, for the
/// architectures whose attention bias replaces rotary position embedding (BLOOM, MPT; spec 254).
/// `slopes` is the model's fixed `[n_heads]` per-head slope constant (a per-model, not per-call, value:
/// a pure function of `n_heads` that the Runner still binds as a named constant, same as `rope.cos`).
/// Returns `[rows, n_heads, tokens, cap]`: masked pairs keep the plain causal bias unmodified (the
/// ALiBi term is multiplied by `visible` before it is added, so it is exactly `0.0` wherever the causal
/// term is `-1.0e9`); visible pairs add `-slope[h] * (pos[r,i] - j)`.
pub fn alibi_mask_from_pos(
    b: &Builder,
    pos: Traced,
    cap: usize,
    window: Option<usize>,
    slopes: Traced,
    n_heads: usize,
) -> Traced {
    let (visible, pos_col, keys_row) = mask_visibility(b, pos, cap, window);
    let base = b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, visible, Scalar::F32(1.0)),
        Scalar::F32(1.0e9),
    ); // [rows,1,tokens,cap]: 0.0 visible, -1.0e9 masked
    let distance = b.binary(BinOp::Sub, pos_col, keys_row); // [rows,1,tokens,cap] = pos[r,i] - j
    let neg_slopes = b.reshape(b.unary(UnOp::Neg, slopes), vec![1, n_heads, 1, 1]);
    let term = b.binary(BinOp::Mul, neg_slopes, distance); // [rows,n_heads,tokens,cap]
    let term = b.binary(BinOp::Mul, term, visible); // zeroed wherever masked
    b.binary(BinOp::Add, base, term) // base broadcasts over the n_heads axis
}

/// Shared visibility core of [`causal_mask_from_pos`]/[`alibi_mask_from_pos`]: `1.0`/`0.0` visibility
/// plus the two broadcast-ready operands the ALiBi variant also needs (the query's absolute position,
/// reshaped to broadcast over `cap`, and the key axis's absolute positions). All `[rows, 1, tokens, cap]`
/// or broadcastable to it.
fn mask_visibility(
    b: &Builder,
    pos: Traced,
    cap: usize,
    window: Option<usize>,
) -> (Traced, Traced, Traced) {
    let pos_shape = b.aval(pos).shape;
    assert_eq!(
        pos_shape.len(),
        2,
        "causal_mask_from_pos: Slot::Pos must be [rows, tokens], got shape {pos_shape:?}"
    );
    let (rows, tokens) = (pos_shape[0], pos_shape[1]);
    let pos_f = b.cast(pos, DType::F32); // [rows, tokens]
    let pos_col = b.reshape(pos_f, vec![rows, 1, tokens, 1]); // query abs pos, broadcasts over cap
    let keys = b.iota(cap); // [cap] f32: 0..cap-1
    let keys_row = b.reshape(keys, vec![1, 1, 1, cap]); // key abs pos, broadcasts over rows/tokens
    let visible = b.binary(BinOp::Ge, pos_col, keys_row); // [rows,1,tokens,cap]: j <= pos
    let visible = match window {
        None => visible,
        Some(w) => {
            // also require pos - j < w, i.e. j > pos - w, i.e. j >= pos - (w - 1).
            let lower_bound = b.binary_scalar(BinOp::Sub, pos_col, Scalar::F32((w - 1) as f32));
            let in_window = b.binary(BinOp::Ge, keys_row, lower_bound);
            b.binary(BinOp::Mul, visible, in_window)
        }
    };
    (visible, pos_col, keys_row)
}

/// Prefill (whole-prompt) linear attention (card 038), the parallel quadratic form of the decode
/// recurrence [`linear_attention_decode`]. The recurrence `o_t = q_t @ sum_{j<=t} decay^(t-j) k_j^T v_j`
/// equals `o_t = sum_{j<=t} decay^(t-j) (q_t . k_j) v_j`, a causal, decay-weighted `(Q K^T) V` with no
/// softmax:
///
/// ```text
/// o = (mask (.) (Q @ K^T)) @ V        mask[t,j] = decay^(t-j) for j <= t, else 0  (multiplicative)
/// ```
///
/// `q[1,Hq,L,D_k]`, `k`/`v[1,Hkv,L,D_k|D_v]` (GQA), `mask` (the caller precomputes the lower-triangular
/// decay weights: 1 on the diagonal, `decay^(t-j)` below, 0 above). Returns `[1,Hq,L,D_v]`. Pure
/// primitive composition (two matmuls + a masked multiply). O(L^2) but fully parallel: one graph, no
/// scan.
///
/// The mask absorbs the gate, so per-head data-dependent gating is also covered: pass a per-head
/// `mask[1,Hq,L,L]` whose entries are the cumulative per-head gate product
/// `mask[h,t,j] = prod_{j<i<=t} g_h[i]` (bounded in `[0,1]` for decay gates, so numerically stable).
/// Per-channel gating (the GLA gate inside `D_k`) does not reduce to a scalar `(h,t,j)` mask and needs
/// a chunked scan.
pub fn linear_attention_prefill(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    mask: Traced,
) -> Traced {
    let k = repeat_kv(b, k, n_rep);
    let v = repeat_kv(b, v, n_rep);
    let kt = b.transpose(k, vec![0, 1, 3, 2]); // [1,Hq,D_k,L]
    let scores = b.matmul(q, kt); // [1,Hq,L,L] = Q K^T
    let weighted = b.binary(BinOp::Mul, scores, mask); // multiplicative decay-causal mask, broadcast over Hq
    b.matmul(weighted, v) // [1,Hq,L,D_v]
}

/// Prefill linear attention with per-channel data-dependent gating (GLA / gated-delta), card 038. The
/// gate `g_t[a]` lives inside the key dimension `D_k`, so it does not fold into a scalar `(h,t,j)`
/// mask. The cumulative per-channel gate product `A_t[a] = prod_{i<=t} g_i[a]` is factored out of the
/// pairwise term:
///
/// ```text
/// o_t[c] = sum_{j<=t} sum_a q_t[a] (A_t[a]/A_j[a]) k_j[a] v_j[c]
///        = sum_{j<=t} ( (q_t (.) A_t) . (k_j (/) A_j) ) v_j[c]
/// ```
///
/// so it is again a causal `(Q' K'^T) V` with a binary lower-triangular mask, where `Q' = Q (.) A` and
/// `K' = K (/) A`. The caller precomputes `cumgate = A [1,Hq,L,D_k]` and a binary causal
/// `mask[1,1,L,L]` (1 for `j<=t`, else 0). Pure primitives, exact, fully parallel.
///
/// Numerical range: `K' = K / A` divides by the whole-sequence cumulative product. f32's wide exponent
/// (plus denormals) keeps this stable even at `A ~ 1e-39`, but with strong decay over a long sequence
/// `A` underflows to zero and `K/A` becomes inf/NaN (e.g. `decay=0.1`, `L=64`: `A=1e-64 -> 0`). For
/// that regime use [`linear_attention_prefill_chunked`], which keeps the gate cumulative only within a
/// chunk and carries state across chunks. This O(L^2) form is the building block chunked-scan
/// composes.
pub fn linear_attention_prefill_gated(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    cumgate: Traced,
    mask: Traced,
) -> Traced {
    let k = repeat_kv(b, k, n_rep); // [1,Hq,L,D_k]
    let v = repeat_kv(b, v, n_rep);
    let q2 = b.binary(BinOp::Mul, q, cumgate); // Q' = Q (.) A
    let k2 = b.binary(BinOp::Div, k, cumgate); // K' = K (/) A
    let kt = b.transpose(k2, vec![0, 1, 3, 2]); // [1,Hq,D_k,L]
    let scores = b.matmul(q2, kt); // [1,Hq,L,L] = Q' K'^T
    let weighted = b.binary(BinOp::Mul, scores, mask); // binary causal mask (the gate is already in Q'/K')
    b.matmul(weighted, v) // [1,Hq,L,D_v]
}

/// Unit-lower-triangular `[C,C]` matrix inverse `T = (I + attn)^{-1}` (spec 139, FR-003 / SC-003): the
/// UT-transform that decouples the delta-rule's state-dependent intra-chunk correction (see
/// `gdn_prefill_chunked`, card 153 P2). `attn` is `[.., C, C]`, strictly lower-triangular (zero
/// diagonal, e.g. `tril(diag(beta).K.K^T, -1)`), so `M = I + attn` is unit-lower-triangular,
/// `det(M) = 1`, invertible without pivoting. Leading `..` dims are an arbitrary batch (e.g.
/// `[1,H,n_chunks]`); the inverse is over the last two axes only.
///
/// Uses block-recursive halving (spec 139 review C6; forward substitution is `C`-deep sequential and
/// would blow up the dispatch count). Partition `M` into 2x2 blocks of `[C/2,C/2]`:
///
/// ```text
/// M = [[A, 0],      M^{-1} = [[A^{-1},              0     ],
///      [B, D]]                [-D^{-1} @ B @ A^{-1}, D^{-1}]]
/// ```
///
/// `A`/`D` are the unit-lower-triangular diagonal blocks (recurse), `B` is the off-diagonal coupling
/// block (read directly from `attn`, since the identity contributes nothing off-diagonal). Recursing
/// on `A`/`D` halves `C` per level down to `1x1`, whose inverse is `[1]` (a strictly-lower `attn` has
/// a structurally zero diagonal), built as `zeros_like(attn_block) + 1` (`Sub` of the block with
/// itself for an exact zero of the right batch shape, then `+1`), no host constant. `C` must be a
/// power of two (asserted); the recursion unrolls at trace time, so `C=64` is 6 levels of matmuls,
/// not an IR-level loop.
///
/// Composes from `Slice` (partition into `A`/`B`/`D`), `MatMul` (`B@A^{-1}` then `D^{-1}@(B@A^{-1})`),
/// `Neg`, `Sub` (the zero trick), and `Concat` (the top-right block is zero, so
/// `out = concat_rows(concat_cols(A_inv, zeros), concat_cols(neg_DBA, D_inv))`). No new `OpKind`
/// (FR-002/FR-003).
pub fn unit_lower_triangular_inverse(b: &Builder, attn: Traced) -> Traced {
    let shape = b.aval(attn).shape;
    let rank = shape.len();
    assert!(
        rank >= 2,
        "unit_lower_triangular_inverse: attn must be at least rank 2, got shape {shape:?}"
    );
    let c = shape[rank - 1];
    assert_eq!(
        shape[rank - 2],
        c,
        "unit_lower_triangular_inverse: last two axes must be square (C,C), got shape {shape:?}"
    );
    ut_inverse_neumann_doubling(b, attn, c, rank)
}

/// Lean UT-transform via Neumann-series doubling (spec 139 P1, the default). `M = I + N` with
/// `N = attn` strictly lower-triangular is nilpotent (`N^C = 0` for a `[.., C, C]` block), so its
/// inverse is the finite Neumann series `T = M^{-1} = sum_{i=0}^{C-1} (-N)^i`, exact rather than
/// truncated. Instead of summing `C` terms one at a time, double the partial sum each step: with
/// `S_k = sum_{i<k} (-N)^i` and `P_k = (-N)^k`,
/// ```text
/// S_{2k} = S_k + P_k @ S_k      (split the sum: the top half factors as (-N)^k times the bottom half)
/// P_{2k} = P_k @ P_k
/// ```
///
/// After `log2(C)` doublings `S_C = T` exactly. This is `~3*log2(C)` matmuls versus block-recursive
/// halving's `~C` (`947` eqns at `C=64`); at `C=64` it emits ~45 eqns, the ~20x dispatch-count cut
/// the chunked-GDN prefill needs (card 158). Composes only from `MatMul`, `Neg`, `Add`, and
/// [`identity_like`] (no new `OpKind`, no host constant).
///
/// The first level is folded: `S_2 = I + P_1 @ I = I + P_1` (an `Add`, not a `MatMul`), and the last
/// level's `P_C` is never needed, so both are skipped.
pub(crate) fn ut_inverse_neumann_doubling(
    b: &Builder,
    attn: Traced,
    c: usize,
    rank: usize,
) -> Traced {
    if c == 1 {
        // a strictly-lower 1x1 block is structurally [0], so M = [1] and T = [1]. zero-of-the-right-batch
        // -shape via attn - attn, then + 1 (the same host-constant-free trick as identity_like's seed).
        let zero = b.binary(BinOp::Sub, attn, attn);
        return b.binary_scalar(BinOp::Add, zero, Scalar::F32(1.0));
    }
    assert_eq!(
        c & (c - 1),
        0,
        "unit_lower_triangular_inverse: Neumann doubling needs C a power of two (got block size {c})"
    );
    let ident = identity_like(b, attn, c, rank); // I of the same [.., C, C] shape/dtype as attn
    let p1 = b.unary(UnOp::Neg, attn); // P_1 = (-N)^1 = -attn
    // level 1 folded: S_2 = I + P_1 @ I = I + P_1; P_2 = P_1 @ P_1.
    let mut s = b.binary(BinOp::Add, ident, p1); // S_2
    let mut p = b.matmul(p1, p1); // P_2
    let mut k = 2usize;
    while k < c {
        let ps = b.matmul(p, s); // P_k @ S_k
        s = b.binary(BinOp::Add, s, ps); // S_{2k} = S_k + P_k @ S_k
        k *= 2;
        if k < c {
            p = b.matmul(p, p); // P_{2k} = P_k @ P_k (unused at the last level, so skipped there)
        }
    }
    s // S_C = T
}

/// Build a `[.., c, c]` identity matrix with the same batch dims and dtype as `like` (a `[.., C, C]`
/// tensor), with no host constant. Seeds a `[.., 1, 1]` one-block (slice `like` to a single cell,
/// subtract it from itself for an exact zero, `+ 1`), then grows it by `log2(c)` levels of
/// block-diagonal doubling: each level places the current `[.., s, s]` identity on both diagonal
/// corners of a `[.., 2s, 2s]` block with zero off-diagonal blocks (`zeros = ident - ident`),
/// assembled by two column-concats and one row-concat. `c` must be a power of two. ~`4*log2(c) + 4`
/// eqns (28 at `c=64`).
pub(crate) fn identity_like(b: &Builder, like: Traced, c: usize, rank: usize) -> Traced {
    let (ax0, ax1) = (rank - 2, rank - 1);
    // [.., 1, 1] one-block seed: reuse `like`'s batch shape/dtype, no external data.
    let cell = b.slice(b.slice(like, ax0, 0, 1), ax1, 0, 1); // [.., 1, 1]
    let zero1 = b.binary(BinOp::Sub, cell, cell); // exact 0
    let mut ident = b.binary_scalar(BinOp::Add, zero1, Scalar::F32(1.0)); // I_1 = [1]
    let mut s = 1usize;
    while s < c {
        let zeros = b.binary(BinOp::Sub, ident, ident); // [.., s, s] zero block
        let top = b.concat(ax1, &[ident, zeros]); // [.., s, 2s]
        let bottom = b.concat(ax1, &[zeros, ident]); // [.., s, 2s]
        ident = b.concat(ax0, &[top, bottom]); // [.., 2s, 2s]
        s *= 2;
    }
    ident
}

/// Chunked-scan prefill for per-channel gated linear attention (card 038): the numerically stable
/// form of [`linear_attention_prefill_gated`]. The naive form divides by the cumulative gate over the
/// whole sequence (`K/A`), which underflows; this splits the sequence into chunks of `chunk`
/// positions and keeps the gate cumulative only within a chunk (so `k/beta` stays bounded), carrying
/// the recurrent state `S [1,Hq,D_k,D_v]` across chunks. For each chunk `r` (with `beta` = the
/// per-channel gate product from the chunk start, and `Lambda_r` = `beta` at the chunk end):
/// ```text
/// Q' = Q (.) beta ;  K' = K (/) beta
/// o_chunk = (causal_mask (.) (Q' K'^T)) @ V    +    Q' @ S_r        (intra-chunk + carried state)
/// S_{r+1} = Lambda_r (.) (S_r + K'^T @ V)                          (state carry, gated)
/// ```
///
/// `q[1,Hq,L,D_k]`, `k`/`v[1,Hkv,L,D_k|D_v]` (GQA), `chunk_beta[1,Hq,L,D_k]` (the caller's per-chunk
/// cumulative gate product), `causal[1,1,chunk,chunk]` (binary lower-triangular for one chunk). `L`
/// must be a multiple of `chunk`. Returns `[1,Hq,L,D_v]`, equal to [`linear_attention_prefill_gated`]
/// but stable for long sequences. The chunk loop is unrolled at trace time (L is static).
#[allow(clippy::too_many_arguments)]
pub fn linear_attention_prefill_chunked(
    b: &Builder,
    q: Traced,
    k: Traced,
    v: Traced,
    n_rep: usize,
    chunk_beta: Traced,
    causal: Traced,
    chunk: usize,
) -> Traced {
    let k = repeat_kv(b, k, n_rep); // [1,Hq,L,D_k]
    let v = repeat_kv(b, v, n_rep); // [1,Hq,L,D_v]
    let qp = b.binary(BinOp::Mul, q, chunk_beta); // Q' = Q (.) beta
    let kp = b.binary(BinOp::Div, k, chunk_beta); // K' = K (/) beta
    let l = b.aval(q).shape[2];
    let n_chunks = l / chunk;
    let mut s: Option<Traced> = None; // carried state [1,Hq,D_k,D_v]; None before the first chunk (S_0 = 0)
    let mut outs: Vec<Traced> = Vec::with_capacity(n_chunks);
    for r in 0..n_chunks {
        let (lo, hi) = (r * chunk, (r + 1) * chunk);
        let qr = b.slice(qp, 2, lo, hi); // [1,Hq,chunk,D_k]
        let kr = b.slice(kp, 2, lo, hi);
        let vr = b.slice(v, 2, lo, hi); // [1,Hq,chunk,D_v]
        // intra-chunk: causal (Q' K'^T) V within the chunk (beta is small here, so K' is well-scaled).
        let krt = b.transpose(kr, vec![0, 1, 3, 2]); // [1,Hq,D_k,chunk]
        let scores = b.matmul(qr, krt); // [1,Hq,chunk,chunk]
        let masked = b.binary(BinOp::Mul, scores, causal);
        let intra = b.matmul(masked, vr); // [1,Hq,chunk,D_v]
        // inter-chunk: the carried state contributes Q' @ S_r (zero for the first chunk).
        let o_r = match &s {
            Some(state) => {
                let inter = b.matmul(qr, *state); // [1,Hq,chunk,D_v]
                b.binary(BinOp::Add, intra, inter)
            }
            None => intra,
        };
        outs.push(o_r);
        // state carry: S_{r+1} = Lambda_r (.) (S_r + K'^T V), Lambda_r = beta at the chunk's last position.
        let krt_v = b.matmul(krt, vr); // K'^T @ V = [1,Hq,D_k,D_v]
        let s_pre = match &s {
            Some(state) => b.binary(BinOp::Add, *state, krt_v),
            None => krt_v,
        };
        let lam = b.slice(chunk_beta, 2, hi - 1, hi); // beta at chunk end [1,Hq,1,D_k]
        let lam = b.transpose(lam, vec![0, 1, 3, 2]); // [1,Hq,D_k,1] (broadcasts over D_v)
        s = Some(b.binary(BinOp::Mul, lam, s_pre)); // gated carry
    }
    b.concat(2, &outs) // [1,Hq,L,D_v]
}

/// Zero-pad `x` along `axis` by `pad` positions (FR-005 / spec 139 review C9). There is no `Pad`
/// primitive, so this builds an exact zero block via the sub-self trick (as in
/// [`unit_lower_triangular_inverse`]'s base case): slice one position of `x`, subtract it from
/// itself, and [`Builder::broadcast`] the size-1 axis up to `pad`. Concat appends it after `x`. A
/// no-op when `pad == 0`.
pub(crate) fn zero_pad_seq(b: &Builder, x: Traced, axis: usize, pad: usize) -> Traced {
    if pad == 0 {
        return x;
    }
    let one = b.slice(x, axis, 0, 1); // [..,1,..] - reuses x's own shape/dtype, no external data needed
    let zero = b.binary(BinOp::Sub, one, one); // exact zero, same shape as `one`
    let mut target = b.aval(one).shape;
    target[axis] = pad;
    let zero_pad = b.broadcast(zero, target); // [..,pad,..]
    b.concat(axis, &[x, zero_pad])
}

/// Numerically stable within-chunk cumulative log-decay `G_i = cumsum(g)[i]` (i = 0..C-1), computed by
/// a Hillis-Steele doubling additive scan instead of a single `tril_incl @ g` matmul (card 158).
///
/// The real Qwen3-Next `g = softplus(dt)*ssm_a` reaches per-step magnitudes up to ~-92 (`ssm_a` down
/// to ~-72 on the real 35B checkpoint, `probe_qwen3next_real_g_range_layer0`), so the cumulative sum
/// over a C=64 chunk reaches `|G_i|` ~5900. Linear summation accumulates rounding error that grows
/// with the running-sum magnitude (roughly with chunk length squared). The doubling scan needs only
/// `ceil(log2(C))` rounds, so accumulated rounding is `O(log2(C))` additions deep instead of `O(C)`.
///
/// This stays in log space rather than computing a per-position decay from the chunk start and
/// dividing ratios (`A_i/A_j`): a single large-decay step makes `A_i` underflow to exact `0.0f32`, and
/// every later ratio, including the diagonal `A_i/A_i` (which must be exactly `1`), collapses to
/// `0/0`, irrecoverably losing the chunk-start-referenced magnitude (this regressed
/// `debug_gdn_prefill_chunked_single_head_varied_decay`, error ~1e-2 to ~35). In log space `G_i`
/// stays finite and `G_i - G_j` is a direct subtraction, so nearby `i`,`j` (including `i == j`,
/// giving exactly `0`) recover the correct local difference, which `gdn_prefill_chunked` relies on
/// for its masked-diagonal and state-carry terms. This function only tightens the precision of `G_i`;
/// `gdn_prefill_chunked` keeps its subtract-then-mask-then-exp-once structure.
pub fn cumulative_log_decay_scan(b: &Builder, g: Traced, chunk: usize) -> Traced {
    let axis = 2; // the C (position) axis, matching gdn_prefill_chunked's [1,H_v,C,1] convention
    let mut x = g;
    let mut d = 1usize;
    while d < chunk {
        // shifted[i] = x[i-d] for i>=d, else 0.0 (the sum identity - a no-op addend).
        let keep = b.slice(x, axis, 0, chunk - d); // [..,C-d,..]
        let one_slice = b.slice(x, axis, 0, 1); // reuses x's shape/dtype, no external data needed
        let zero = b.binary(BinOp::Sub, one_slice, one_slice); // exact 0, same shape as `one_slice`
        let mut pad_shape = b.aval(zero).shape;
        pad_shape[axis] = d;
        let pad = b.broadcast(zero, pad_shape); // [..,d,..] of exact 0.0
        let shifted = b.concat(axis, &[pad, keep]); // [..,C,..]
        x = b.binary(BinOp::Add, x, shifted);
        d *= 2;
    }
    x
}

/// Chunked prefill for Qwen3-Next's Gated-DeltaNet (GDN) delta-rule recurrence (spec 139 / card 153
/// P2). Reproduces [`gated_delta_net_decode`]'s per-position recurrence over all `L` positions; that
/// decode op is the llama.cpp-validated oracle (`gdn_prefill_chunked_matches_decode_recurrence`,
/// spec 139 SC-001/002). Unlike the additive GLA in [`linear_attention_prefill_chunked`], the delta
/// rule's `- k_t @ S_{t-1}` correction couples every position in a chunk to the running state, so the
/// intra-chunk term needs the UT-transform ([`unit_lower_triangular_inverse`], P1). Only the outer
/// chunk-loop/state-carry shape is shared with `linear_attention_prefill_chunked`.
///
/// Derivation (matches `gated_delta_net_decode`'s recurrence
/// `S_i = decay_i*(I - beta_i k_i k_i^T) @ S_{i-1} + beta_i k_i^T v_i`, unrolled and solved in closed
/// form via the UT-transform; cross-checked against llama.cpp `build_delta_net_chunking`'s GDA branch
/// in `delta-net-base.cpp`). Per chunk of `C` positions (local index `i`, `A_i = exp(G_i)` the
/// within-chunk cumulative decay through position `i` inclusive, `G_i = cumsum(g)[i]`, S3; `g` is
/// already log-domain, so no `Log` call):
/// ```text
/// attn        = tril_strict (.) ( beta_row (.) (decay_ratio_incl (.) (K @ K^T)) )   # 3rd product folds
///               decay_ratio_incl[i,j] = exp(G_i - G_j) for j<=i, else 0             # decays into K@K^T
/// T           = unit_lower_triangular_inverse(attn)                                 # the UT-transform (P1)
/// v_pseudo    = T @ (beta (.) V)                                                    # 1st product (B1)
/// k_cumdecay  = T @ (beta (.) A (.) K)                                              # T folded once (assoc.)
/// v_new       = v_pseudo - k_cumdecay @ S_chunk_in                                  # 2nd product (B1)
/// o_chunk     = (decay_ratio_incl (.) (Q @ K^T)) @ v_new + (Q (.) A) @ S_chunk_in    # 3rd product (output,
///                                                                                    #  S2 diag-inclusive)
/// S_chunk_out = S_chunk_in * exp(g_last) + (K (.) exp(g_last - G))^T @ v_new         # gated state carry
/// ```
///
/// `beta_row` means `beta_i` scales row `i` (the position receiving the correction, not the
/// contributing column); this follows directly from `gated_delta_net_decode`'s recurrence and matches
/// llama.cpp's per-row `k_b = k*beta` scaling.
///
/// S2 (two masks): `tril_strict` (`[1,1,C,C]`, strictly lower, zero diagonal) feeds the solve's
/// `attn`; the same `decay_ratio_incl` (lower-triangular including the diagonal, via `tril_incl`)
/// feeds the intra-chunk output score mask (`o_p` attends to itself, the post-update state). Swapping
/// them is an invisible bug (spec 139 "internal correctness" note).
///
/// `q`/`k` are `[1,H_k,L,D]` (L2-normed, unscaled; the `1/sqrt(D)` query scale is applied inside, S4);
/// `v`/`g`/`beta` are `[1,H_v,L,1|D]` (`g`/`beta` `[1,H_v,L,1]`, already the log-domain forget-gate
/// arg and delta weight); `s_in` is `[1,H_v,D,D]`. `q`/`k` are repeated `H_k -> H_v` via
/// [`repeat_kv_tiled`] inside this fn (FR-004: `h -> h % H_kv`, not the blocked [`repeat_kv`]; card
/// 140's tiled-vs-blocked landmine). `tril_incl`/`tril_strict` are `[1,1,C,C]` caller-supplied consts
/// at the static chunk size `C` (the `Iota` primitive of card 558a is F32-only and ranges over a
/// single axis; it does not displace these two hand-built 2-D triangular tables). `L` is zero-padded
/// to `ceil(L/C)*C`
/// (FR-005, [`zero_pad_seq`]) and the output truncated back to `L`; the padded tail is inert (zero
/// `k`/`v`/`beta` never contribute, zero `g` never changes cumulative decay). Returns
/// `(o [1,H_v,L,D], s_out [1,H_v,D,D])`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_chunked(
    b: &Builder,
    q: Traced,           // [1,H_k,L,D] L2-normed, UNSCALED
    k: Traced,           // [1,H_k,L,D] L2-normed
    v: Traced,           // [1,H_v,L,D]
    g: Traced,           // [1,H_v,L,1] already log-domain (negative)
    beta: Traced,        // [1,H_v,L,1] in (0,1)
    s_in: Traced,        // [1,H_v,D,D]
    tril_incl: Traced,   // [1,1,C,C] lower-triangular ones INCLUDING the diagonal
    tril_strict: Traced, // [1,1,C,C] STRICTLY lower-triangular ones (zero diagonal)
    chunk: usize,
) -> (Traced, Traced) {
    let qshape = b.aval(q).shape;
    let (h_k, l, d) = (qshape[1], qshape[2], qshape[3]);
    let h_v = b.aval(v).shape[1];
    assert_eq!(
        h_v % h_k,
        0,
        "gdn_prefill_chunked: H_v ({h_v}) must be a multiple of H_k ({h_k})"
    );
    let n_rep = h_v / h_k;

    // FR-004: tiled GQA repeat q/k from H_k to H_v (NOT the blocked repeat_kv, card 140).
    let q = repeat_kv_tiled(b, q, n_rep); // [1,H_v,L,D]
    let k = repeat_kv_tiled(b, k, n_rep); // [1,H_v,L,D]

    // S4: the query scale lives INSIDE this fn (the oracle applies it internally too).
    let q = b.binary_scalar(BinOp::Mul, q, Scalar::F32(1.0 / (d as f32).sqrt()));

    // FR-005: zero-pad every per-position input to a whole number of chunks.
    let n_chunks = l.div_ceil(chunk);
    let lp = n_chunks * chunk;
    let pad = lp - l;
    let q = zero_pad_seq(b, q, 2, pad);
    let k = zero_pad_seq(b, k, 2, pad);
    let v = zero_pad_seq(b, v, 2, pad);
    let g = zero_pad_seq(b, g, 2, pad);
    let beta = zero_pad_seq(b, beta, 2, pad);

    let mut s_state = s_in; // carried per-chunk state [1,H_v,D,D]
    let mut outs: Vec<Traced> = Vec::with_capacity(n_chunks);
    for r in 0..n_chunks {
        let (lo, hi) = (r * chunk, (r + 1) * chunk);
        let qr = b.slice(q, 2, lo, hi); // [1,H_v,C,D]
        let kr = b.slice(k, 2, lo, hi); // [1,H_v,C,D]
        let vr = b.slice(v, 2, lo, hi); // [1,H_v,C,D]
        let gr = b.slice(g, 2, lo, hi); // [1,H_v,C,1] already log-domain
        let br = b.slice(beta, 2, lo, hi); // [1,H_v,C,1]

        // S3 (card 158 fix): within-chunk cumulative log-decay G_i = cumsum(g)[i] via the Hillis-Steele
        // additive scan (see `cumulative_log_decay_scan` for why it stays in log space).
        let cumlog = cumulative_log_decay_scan(b, gr, chunk); // [1,H_v,C,1] = G_i
        let cum_row = b.reshape(cumlog, vec![1, h_v, 1, chunk]); // [1,H_v,1,C] = G_j (same values, moved axis)
        let diff = b.binary(BinOp::Sub, cumlog, cum_row); // diff[i,j] = G_i - G_j (exactly 0 on the diagonal)
        // Mask the exponent additively before exp, not the result after. In the upper triangle (i<j)
        // diff = G_i - G_j is positive and grows with the chunk's cumulative magnitude; past ~88,
        // exp(diff) overflows to +Inf and a post-exp mask multiply gives Inf * 0.0 = NaN, poisoning the
        // chunk and the carried GDN state. A large negative bias strictly above the diagonal drives
        // those entries to exp(-inf)=0, leaving the retained lower/diag entries (diff <= 0, exp in
        // (0,1]) untouched. `tril_incl` is 1 on/below the diagonal and 0 above, so
        // `(tril_incl - 1) * 1e30` is 0 on/below and -1e30 above.
        let upper_bias = b.binary_scalar(
            BinOp::Mul,
            b.binary_scalar(BinOp::Add, tril_incl, Scalar::F32(-1.0)),
            Scalar::F32(1.0e30),
        );
        let diff_masked = b.binary(BinOp::Add, diff, upper_bias);
        let ratio = b.unary(UnOp::Exp, diff_masked); // exp(G_i - G_j) on/below diag, 0 strictly above
        let decay_incl = b.binary(BinOp::Mul, ratio, tril_incl); // S2: lower-incl-diag (the OUTPUT mask)
        let decay_strict = b.binary(BinOp::Mul, decay_incl, tril_strict); // S2: strictly-lower (the SOLVE mask)
        let a_cum = b.unary(UnOp::Exp, cumlog); // [1,H_v,C,1] = A_i = exp(G_i)

        // B1 product 1/3: attn = tril_strict (.) (beta_row (.) (decay (.) (K @ K^T))).
        let krt = b.transpose(kr, vec![0, 1, 3, 2]); // [1,H_v,D,C]
        let kk = b.matmul(kr, krt); // [1,H_v,C,C] = K @ K^T
        let kk_decayed = b.binary(BinOp::Mul, kk, decay_strict); // decays folded, already strictly masked
        let attn = b.binary(BinOp::Mul, kk_decayed, br); // beta on the ROW index (broadcast over columns)
        let t = unit_lower_triangular_inverse(b, attn); // [1,H_v,C,C] the UT-transform

        // B1 product 2/3: v_pseudo = T @ (beta (.) V); k_cumdecay = T @ (beta (.) A (.) K) (T folded once,
        // valid by matmul associativity: T@(k_cumdecay_raw@S) == (T@k_cumdecay_raw)@S).
        let bv = b.binary(BinOp::Mul, vr, br); // [1,H_v,C,D]
        let v_pseudo = b.matmul(t, bv); // [1,H_v,C,D]
        let kg_raw = b.binary(BinOp::Mul, b.binary(BinOp::Mul, kr, br), a_cum); // beta*A*K, [1,H_v,C,D]
        let k_cumdecay = b.matmul(t, kg_raw); // [1,H_v,C,D]
        let kcd_s = b.matmul(k_cumdecay, s_state); // [1,H_v,C,D]
        let v_new = b.binary(BinOp::Sub, v_pseudo, kcd_s); // B1 product 3 input: the inter-chunk correction

        // B1 product 3/3 (the output term): intra-chunk (post-update, diag-inclusive) + inter-chunk (S2).
        let q_scores = b.matmul(qr, krt); // [1,H_v,C,C] = Q @ K^T
        let masked_scores = b.binary(BinOp::Mul, q_scores, decay_incl); // S2: diag-INCLUSIVE
        let intra_out = b.matmul(masked_scores, v_new); // [1,H_v,C,D]
        let q_scaled_a = b.binary(BinOp::Mul, qr, a_cum); // Q (.) A, [1,H_v,C,D]
        let inter_out = b.matmul(q_scaled_a, s_state); // [1,H_v,C,D]
        let o_chunk = b.binary(BinOp::Add, intra_out, inter_out);
        outs.push(o_chunk);

        // Gated state carry: S_out = S_in*exp(g_last) + k_gdiff^T @ v_new, k_gdiff = K (.) exp(g_last - G).
        let g_last = b.slice(cumlog, 2, chunk - 1, chunk); // [1,H_v,1,1] = G_{C-1}, the chunk's total log-decay
        let g_last_exp = b.unary(UnOp::Exp, g_last); // [1,H_v,1,1], broadcasts over s_state's [1,H_v,D,D]
        let g_diff = b.binary(BinOp::Sub, g_last, cumlog); // [1,H_v,C,1] = g_last - G_i, broadcast g_last over C
        let k_gdiff = b.binary(BinOp::Mul, kr, b.unary(UnOp::Exp, g_diff)); // [1,H_v,C,D]
        let k_gdiff_t = b.transpose(k_gdiff, vec![0, 1, 3, 2]); // [1,H_v,D,C]
        let kgv = b.matmul(k_gdiff_t, v_new); // [1,H_v,D,D]
        s_state = b.binary(BinOp::Add, b.binary(BinOp::Mul, s_state, g_last_exp), kgv);
    }

    let o_full = b.concat(2, &outs); // [1,H_v,Lp,D]
    let o = b.slice(o_full, 2, 0, l); // FR-005: truncate the padded tail back to L
    (o, s_state)
}

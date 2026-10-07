//! DeepSeek sparse attention (DSA, DeepSeek-V3.2): the lightning indexer and fine-grained token
//! selection as a mask over multi-head latent attention ([`super::mla`]).
//!
//! The indexer scores every cached key for each query token with
//! `I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])`: `q` per indexer head from the low-rank query,
//! `w` per head from the hidden state, and one key per position shared by the heads, LayerNorm'ed
//! and rotated (half-split, over the leading `qk_rope_head_dim` dims only) before it is cached. The
//! keys the step caches carry the position axis like attention's own ([`carried_cache`]).
//!
//! Selection keeps the `topk` best-scoring causal positions of each query token as an additive
//! mask on top of the causal one, so it composes with the attention core unchanged: ranking the
//! scores plus the causal mask keeps top-k from picking a position the query cannot see.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_quant::weights::{DsaRole, WeightId, WeightMap, WeightRole};

use super::attention::{carried_cache, first_writes};
use super::linear::{Weight, WeightError, linear};
use super::mla::Selection;
use super::rope::{RopeParams, apply_rope, rope_rows};
use super::sparse_top_k::{keep_top_k_mask, pairwise_rank};
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;

/// Additive penalty for a position the selection drops: not `f32::MIN` or `-inf`, so it zeroes
/// out after the softmax and summing it with the causal mask's own penalty does not overflow.
const DROPPED: f32 = -1e30;

/// The indexer's key LayerNorm epsilon (`Indexer.k_norm = nn.LayerNorm(head_dim, eps=1e-6)`).
const K_NORM_EPS: f32 = 1e-6;

/// Checked DSA parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DsaParams {
    heads: usize,
    dim: usize,
    topk: usize,
    rope: RopeParams,
}

impl DsaParams {
    /// `heads` indexer heads of `dim` width keeping the `topk` best positions per token; `rope`
    /// rotates the leading `rope.rotary()` dims of each head, which must not exceed `dim`.
    pub fn new(
        heads: usize,
        dim: usize,
        topk: usize,
        rope: RopeParams,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("index_n_heads", heads),
            ("index_head_dim", dim),
            ("index_topk", topk),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if rope.rotary() > dim {
            return Err(ParamError::new(
                "index_head_dim",
                ConfigReason::Exceeds { max: rope.rotary() },
            ));
        }
        Ok(Self {
            heads,
            dim,
            topk,
            rope,
        })
    }
}

/// A layer's indexer weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsaWeights {
    q: Weight,
    k: Weight,
    k_norm: Weight,
    k_norm_bias: Weight,
    weights: Weight,
}

impl DsaWeights {
    /// The indexer's weights for a model of `width` hidden units whose low-rank query is `q_rank`
    /// wide.
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        q_rank: usize,
        p: &DsaParams,
    ) -> Result<Self, WeightError> {
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Dsa(role)), shape)
        };
        Ok(Self {
            q: w(DsaRole::Q, &[p.heads * p.dim, q_rank])?,
            k: w(DsaRole::K, &[p.dim, width])?,
            k_norm: w(DsaRole::KNorm, &[p.dim])?,
            k_norm_bias: w(DsaRole::KNormBias, &[p.dim])?,
            weights: w(DsaRole::Weights, &[p.heads, width])?,
        })
    }
}

/// The indexer's score of every cached key for each query token of `layer`: `[rows, 1, tokens,
/// capacity]`, `sum_j w[t, j] * ReLU(q[t, j] . k[s])`. `x` is the layer's normalized hidden state
/// `[rows, tokens, width]` (the key and the weights read it) and `qr` the low-rank query `[rows,
/// tokens, q_rank]` (the query reads it). The layer's indexer keys (`kv.l{layer}.index`) are
/// carried by `step`; a position not yet written scores zero.
pub fn dsa_scores(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    qr: Traced,
    w: &DsaWeights,
    p: &DsaParams,
) -> Traced {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (hi, di) = (p.heads, p.dim);
    let rope = rope_rows(b, p.rope, step.pos(), cap);

    let q = linear(b, qr, &w.q, None);
    let q = b.transpose(b.reshape(q, vec![rows, t, hi, di]), vec![0, 2, 1, 3]); // [rows, hi, t, di]
    let q = apply_rope(b, q, &rope);

    let k = ops::layernorm(
        b,
        linear(b, x, &w.k, None),
        w.k_norm.declare_f32(b),
        w.k_norm_bias.declare_f32(b),
        K_NORM_EPS,
    );
    let k = apply_rope(b, b.reshape(k, vec![rows, 1, t, di]), &rope);
    let write_at = first_writes(b, step);
    let keys = carried_cache(b, step, &write_at, layer, "index", k, 1, di); // [rows, 1, cap, di]

    let weights = linear(b, x, &w.weights, None); // [rows, t, hi]
    let weights = b.reshape(b.transpose(weights, vec![0, 2, 1]), vec![rows, hi, t, 1]);

    // The head axis reduced, kept at size one so it adds onto the `[rows, 1, t, cap]` mask.
    let dots = b.matmul(q, b.transpose(keys, vec![0, 1, 3, 2])); // [rows, hi, t, cap]
    let scored = b.binary(BinOp::Mul, ops::relu(b, dots), weights);
    b.reduce(RedOp::Sum, scored, 1, true)
}

/// The mask of the sparse attention of `layer`: `sel.mask` plus a [`DROPPED`] penalty on every
/// position outside each query token's `topk` best by [`dsa_scores`]. `x` is the layer's normalized
/// hidden state `[rows, tokens, width]`; the indexer's query reads the low-rank query in `sel`.
pub fn dsa_select(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    sel: Selection,
    w: &DsaWeights,
    p: &DsaParams,
) -> Traced {
    let scores = dsa_scores(b, step, layer, x, sel.qr, w, p);
    let ranked = b.binary(BinOp::Add, scores, sel.mask);
    let keep = keep_top_k_mask(b, pairwise_rank(b, ranked), p.topk); // 1.0 kept, 0.0 dropped
    let drop = b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0)),
        Scalar::F32(-DROPPED),
    ); // 0 kept, DROPPED dropped
    b.binary(BinOp::Add, sel.mask, drop)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::rope_table::RopeFlavor;
    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const WIDTH: usize = 6;
    const Q_RANK: usize = 3;
    const HEADS: usize = 2;
    const DIM: usize = 4;
    const ROPE: usize = 2;
    const THETA: f32 = 10_000.0;
    const T: usize = 5;

    fn id(role: DsaRole) -> WeightId {
        WeightId::layer(0, WeightRole::Dsa(role))
    }

    fn params(topk: usize) -> DsaParams {
        let rope = RopeParams::new(ROPE, THETA, &RopeFlavor::Plain).unwrap();
        DsaParams::new(HEADS, DIM, topk, rope).unwrap()
    }

    fn fixture() -> Fx {
        let mut fx = Fx::new(13);
        fx.f32(id(DsaRole::Q), &[HEADS * DIM, Q_RANK]);
        fx.f32(id(DsaRole::K), &[DIM, WIDTH]);
        fx.set(id(DsaRole::KNorm), &[DIM], vec![1.1, 0.9, 1.2, 0.8]);
        fx.f32(id(DsaRole::KNormBias), &[DIM]);
        fx.f32(id(DsaRole::Weights), &[HEADS, WIDTH]);
        fx
    }

    /// The scores `[T, CAP]` (or, with `select`, the selection mask) of `T` tokens from position
    /// 0 over the zero cache.
    fn run(fx: &Fx, topk: usize, x: &[f32], qr: &[f32], select: bool) -> Vec<f32> {
        let p = params(topk);
        let w = DsaWeights::new(&fx.map(), 0, WIDTH, Q_RANK, &p).unwrap();
        let b = Builder::new();
        let step = Step::new(&b, step_shape(T, CAP, LogitRows::All));
        let xs = b.constant("x", TensorType::f32(vec![1, T, WIDTH]));
        let qrs = b.constant("qr", TensorType::f32(vec![1, T, Q_RANK]));
        let out = if select {
            let mask = ops::causal_mask_from_pos(&b, step.pos(), CAP, None);
            dsa_select(&b, &step, 0, xs, Selection { qr: qrs, mask }, &w, &p)
        } else {
            dsa_scores(&b, &step, 0, xs, qrs, &w, &p)
        };
        let g = step.finish(b, out);
        let zeros: Vec<Vec<f32>> = g
            .state
            .iter()
            .map(|&(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let ids = vec![0; T];
        let pos: Vec<i32> = (0..T as i32).collect();
        fx.eval(
            &g,
            &[("x", x), ("qr", qr)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &state,
        )
        .0
    }

    /// Half-split rotation of the leading `ROPE` dims of `row` at `pos` in f64: `(x[j], x[j +
    /// ROPE/2])` turn by `pos * theta^(-2j / ROPE)`.
    fn rotate(row: &[f64], pos: usize) -> Vec<f64> {
        let mut out = row.to_vec();
        let half = ROPE / 2;
        for j in 0..half {
            let angle = pos as f64 * f64::from(THETA).powf(-2.0 * j as f64 / ROPE as f64);
            let (s, c) = angle.sin_cos();
            out[j] = row[j] * c - row[j + half] * s;
            out[j + half] = row[j + half] * c + row[j] * s;
        }
        out
    }

    fn matvec(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        (0..rows)
            .map(|r| (0..k).map(|j| f64::from(w[r * k + j]) * x[j]).sum())
            .collect()
    }

    /// `I[t, s]` for every key the step wrote (`s < T`, causal or not: the causal mask is applied after;
    /// unwritten keys score zero), from the definition.
    fn reference_scores(fx: &Fx, x: &[f32], qr: &[f32]) -> Vec<f64> {
        let value = |role| fx.values(id(role));
        let (wk, kn, kb) = (
            value(DsaRole::K),
            value(DsaRole::KNorm),
            value(DsaRole::KNormBias),
        );
        let row = |data: &[f32], t: usize, n: usize| -> Vec<f64> {
            data[t * n..(t + 1) * n].iter().map(|&v| v.into()).collect()
        };
        let keys: Vec<Vec<f64>> = (0..T)
            .map(|s| {
                let raw = matvec(&wk, DIM, WIDTH, &row(x, s, WIDTH));
                let mean = raw.iter().sum::<f64>() / DIM as f64;
                let var = raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / DIM as f64;
                let normed: Vec<f64> = (0..DIM)
                    .map(|j| {
                        (raw[j] - mean) / (var + 1e-6).sqrt() * f64::from(kn[j]) + f64::from(kb[j])
                    })
                    .collect();
                let (head, tail) = normed.split_at(ROPE);
                [rotate(head, s), tail.to_vec()].concat()
            })
            .collect();
        let mut out = Vec::new();
        for t in 0..T {
            let q = matvec(&value(DsaRole::Q), HEADS * DIM, Q_RANK, &row(qr, t, Q_RANK));
            let w = matvec(&value(DsaRole::Weights), HEADS, WIDTH, &row(x, t, WIDTH));
            for s in 0..CAP {
                let mut score = 0.0;
                if let Some(key) = keys.get(s) {
                    for h in 0..HEADS {
                        let head = &q[h * DIM..(h + 1) * DIM];
                        let (lead, tail) = head.split_at(ROPE);
                        let head = [rotate(lead, t), tail.to_vec()].concat();
                        let dot: f64 = head.iter().zip(key).map(|(a, b)| a * b).sum();
                        score += w[h] * dot.max(0.0);
                    }
                }
                out.push(score);
            }
        }
        out
    }

    fn inputs() -> (Vec<f32>, Vec<f32>) {
        let scaled = |n, seed| {
            oracle::random(n, seed)
                .into_iter()
                .map(|v| 3.0 * v)
                .collect::<Vec<_>>()
        };
        (scaled(T * WIDTH, 21), scaled(T * Q_RANK, 22))
    }

    /// SC-002: the indexer's scores equal the f64 definition, each weight matters, and the
    /// selection keeps exactly the `topk` best causal positions of every query token (the others
    /// are pushed to the dropped penalty, the invisible ones were already masked). Mutation: an
    /// off-by-one in `keep_top_k_mask` (`>= k + 1`); a row keeps `topk + 1` positions and the
    /// selection assertion goes red.
    #[test]
    fn the_indexer_scores_and_the_selection_match_the_definition() {
        let fx = fixture();
        let (x, qr) = inputs();
        let scores = run(&fx, 2, &x, &qr, false);
        let want = reference_scores(&fx, &x, &qr);
        oracle::assert_matches_f64(&scores, &want, 1e-4);
        for role in [
            DsaRole::Q,
            DsaRole::K,
            DsaRole::KNorm,
            DsaRole::KNormBias,
            DsaRole::Weights,
        ] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id(role));
            oracle::assert_differs(&run(&perturbed, 2, &x, &qr, false), &scores, id(role));
        }

        // The positions a query cannot see were already at the causal penalty (the selection may
        // keep or drop them: they tie far below every visible score), so only the visible ones
        // are compared: the `topk` best of them stay, the rest drop to the penalty.
        let mask = run(&fx, 2, &x, &qr, true);
        for t in 0..T {
            let visible = &want[t * CAP..t * CAP + t + 1];
            let mut kept = 0;
            for s in 0..=t {
                let rank = visible.iter().filter(|&&v| v > visible[s]).count();
                let got = mask[t * CAP + s];
                if rank < 2 {
                    kept += 1;
                    assert!(got.abs() < 1e-3, "token {t} key {s} should stay: {got}");
                } else {
                    assert!(got < -1e29, "token {t} key {s} should drop: {got}");
                }
            }
            assert_eq!(kept, 2.min(t + 1), "token {t}");
        }
    }

    #[test]
    fn params_refuse_an_impossible_indexer() {
        let rope = RopeParams::new(ROPE, THETA, &RopeFlavor::Plain).unwrap();
        assert_eq!(
            DsaParams::new(0, DIM, 2, rope),
            Err(ParamError::new("index_n_heads", ConfigReason::Zero))
        );
        assert_eq!(
            DsaParams::new(HEADS, 1, 2, rope),
            Err(ParamError::new(
                "index_head_dim",
                ConfigReason::Exceeds { max: ROPE }
            ))
        );
    }
}

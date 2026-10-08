//! Manifold-constrained hyper-connections (mHC): a residual stream widened to `streams` parallel
//! copies per token, `[rows, tokens, streams, width]`, mixed around every block (DeepSeek-V4,
//! GLM-5.3-Flash).
//!
//! A site ([`mhc_pre`]) reads the whole widened stream of a token and produces three things: `pre`,
//! the weights collapsing the streams into the one vector the block reads; `post`, the weights
//! placing the block's output back onto each stream; and `comb`, a `streams x streams` mixer of the
//! old streams, projected onto the doubly stochastic matrices by Sinkhorn-Knopp. [`mhc_post`] joins
//! the block's output to the mixed streams. The model ends with [`mhc_head`], a `pre`-only collapse.
//!
//! The weights are read through [`Traced`] values in the contraction's own `[streams * width, mix]`
//! orientation and at F32 (the elementwise work is F32), so a family declares them however its
//! checkpoint stores them; [`MhcSiteWeights`] and [`MhcHeadWeights`] do that for a mapped weight.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_quant::weights::{MhcRole, WeightId, WeightMap, WeightRole};

use super::linear::{Weight, WeightError};
use super::standard::ParamError;
use crate::model::ConfigReason;

/// Checked mHC parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MhcParams {
    streams: usize,
    eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
}

impl MhcParams {
    /// `streams` parallel streams, the norm epsilon of the stream normalization, the floor `hc_eps`
    /// added after every sigmoid, softmax and Sinkhorn division, and `sinkhorn_iters` Sinkhorn
    /// rounds. Refused, naming the field, for no streams or rounds or a non-finite epsilon.
    pub fn new(
        streams: usize,
        eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<Self, ParamError> {
        for (field, value) in [("hc_mult", streams), ("hc_sinkhorn_iters", sinkhorn_iters)] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        for (field, value) in [("rms_norm_eps", eps), ("hc_eps", hc_eps)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(ParamError::new(field, ConfigReason::NotFinitePositive));
            }
        }
        Ok(Self {
            streams,
            eps,
            hc_eps,
            sinkhorn_iters,
        })
    }

    pub fn streams(&self) -> usize {
        self.streams
    }

    /// The width of a site's mix: `pre` and `post` (one weight per stream each) and `comb` (one per
    /// stream pair).
    pub fn mix(&self) -> usize {
        (2 + self.streams) * self.streams
    }
}

/// What a site computes from the widened streams: the weights that place the block's output and
/// mix the old streams, and the collapsed block input.
#[derive(Clone, Copy, Debug)]
pub struct MhcMix {
    /// `[rows, tokens, streams]`, in `[0, 2]`.
    pub post: Traced,
    /// `[rows, tokens, streams, streams]`, doubly stochastic.
    pub comb: Traced,
    /// `[rows, tokens, width]`: the streams weighted by `pre` and summed.
    pub collapsed: Traced,
}

/// `x` is `[rows, tokens, streams, width]`; the shape's extents.
fn dims(b: &Builder, x: Traced, p: &MhcParams) -> (usize, usize, usize) {
    let shape = b.aval(x).shape;
    assert_eq!(
        (shape.len(), shape.get(2).copied()),
        (4, Some(p.streams)),
        "mHC reads [rows, tokens, {}, width] streams, got {shape:?}",
        p.streams
    );
    (shape[0], shape[1], shape[3])
}

/// The projection of the normalized streams: `[rows, tokens, streams * width]` against the
/// `[streams * width, out]` weight.
fn mix_of(b: &Builder, x: Traced, projection: Traced, p: &MhcParams) -> Traced {
    let (rows, tokens, width) = dims(b, x, p);
    let flat = b.reshape(x, vec![rows, tokens, p.streams * width]);
    b.matmul(ops::rmsnorm_no_weight(b, flat, p.eps), projection)
}

/// `streams` weighted by `pre` (`[rows, tokens, streams]`) and summed over the streams.
fn collapse(b: &Builder, x: Traced, pre: Traced, p: &MhcParams) -> Traced {
    let (rows, tokens, _) = dims(b, x, p);
    let pre = b.reshape(pre, vec![rows, tokens, p.streams, 1]);
    b.reduce(RedOp::Sum, b.binary(BinOp::Mul, pre, x), 2, false)
}

/// Sum over the stream axis that precedes the last, keeping it as an axis of one.
fn sum_over_rows(b: &Builder, x: Traced) -> Traced {
    let rank = b.aval(x).shape.len();
    b.reduce(RedOp::Sum, x, rank - 2, true)
}

/// One site's mapping of the widened streams `x[rows, tokens, streams, width]`. `projection` is
/// `[streams * width, mix]`, `base` `[mix]` and `scale` `[3]` (the `pre`, `post` and `comb` scales),
/// all F32.
pub fn mhc_pre(
    b: &Builder,
    x: Traced,
    projection: Traced,
    base: Traced,
    scale: Traced,
    p: &MhcParams,
) -> MhcMix {
    let (rows, tokens, _) = dims(b, x, p);
    let (hc, eps) = (p.streams, p.hc_eps);
    let mix = mix_of(b, x, projection, p);

    let mixes = |lo: usize, hi: usize| b.slice(mix, 2, lo, hi);
    let bases = |lo: usize, hi: usize| b.slice(base, 0, lo, hi);
    let scaled = |w: Traced, scale_ix: usize, base: Traced| {
        let s = b.slice(scale, 0, scale_ix, scale_ix + 1);
        b.binary(BinOp::Add, b.binary(BinOp::Mul, w, s), base)
    };

    let pre_logits = scaled(mixes(0, hc), 0, bases(0, hc));
    let pre = b.binary_scalar(BinOp::Add, ops::sigmoid(b, pre_logits), Scalar::F32(eps));

    let post_logits = scaled(mixes(hc, 2 * hc), 1, bases(hc, 2 * hc));
    let post = b.binary_scalar(BinOp::Mul, ops::sigmoid(b, post_logits), Scalar::F32(2.0));

    let comb_w = b.reshape(mixes(2 * hc, (2 + hc) * hc), vec![rows, tokens, hc, hc]);
    let comb_b = b.reshape(bases(2 * hc, (2 + hc) * hc), vec![hc, hc]);
    let comb_logits = scaled(comb_w, 2, comb_b);
    let comb = b.binary_scalar(BinOp::Add, ops::softmax(b, comb_logits), Scalar::F32(eps));

    // One column normalization, then `sinkhorn_iters - 1` rounds of rows and columns.
    let divide_by = |m: Traced, sum: Traced| {
        b.binary(
            BinOp::Div,
            m,
            b.binary_scalar(BinOp::Add, sum, Scalar::F32(eps)),
        )
    };
    let mut comb = divide_by(comb, sum_over_rows(b, comb));
    for _ in 1..p.sinkhorn_iters {
        comb = divide_by(comb, b.reduce(RedOp::Sum, comb, 3, true));
        comb = divide_by(comb, sum_over_rows(b, comb));
    }

    MhcMix {
        post,
        comb,
        collapsed: collapse(b, x, pre, p),
    }
}

/// The widened streams after a block: `post * output` on every stream plus `comb`'s mix of the old
/// streams `residual` (`comb` is doubly stochastic but not symmetric, so the mix reads its
/// transpose). `post` and `comb` are the site's ([`MhcMix`]), `output` is `[rows, tokens, width]`
/// and `residual` `[rows, tokens, streams, width]`.
pub fn mhc_post(
    b: &Builder,
    post: Traced,
    comb: Traced,
    output: Traced,
    residual: Traced,
    p: &MhcParams,
) -> Traced {
    let (rows, tokens, width) = dims(b, residual, p);
    let post = b.reshape(post, vec![rows, tokens, p.streams, 1]);
    let output = b.reshape(output, vec![rows, tokens, 1, width]);
    let placed = b.binary(BinOp::Mul, post, output);
    let comb_t = b.transpose(comb, vec![0, 1, 3, 2]);
    b.binary(BinOp::Add, placed, b.matmul(comb_t, residual))
}

/// The model's closing collapse of the widened streams `x[rows, tokens, streams, width]` into
/// `[rows, tokens, width]`: a site's `pre` alone. `projection` is `[streams * width, streams]`,
/// `base` `[streams]` and `scale` `[1]`.
pub fn mhc_head(
    b: &Builder,
    x: Traced,
    projection: Traced,
    base: Traced,
    scale: Traced,
    p: &MhcParams,
) -> Traced {
    let mixes = mix_of(b, x, projection, p);
    let logits = b.binary(BinOp::Add, b.binary(BinOp::Mul, mixes, scale), base);
    let pre = b.binary_scalar(BinOp::Add, ops::sigmoid(b, logits), Scalar::F32(p.hc_eps));
    collapse(b, x, pre, p)
}

/// A mapped weight as the F32 `[in, out]` operand of [`mhc_pre`] and [`mhc_head`]: the stored
/// `[out, in]` projection widened and transposed, or an F32 vector as stored.
fn operand(b: &Builder, w: &Weight, transpose: bool) -> Traced {
    let value = w.declare_f32(b);
    if transpose {
        b.transpose(value, vec![1, 0])
    } else {
        value
    }
}

/// One layer site's weights, as the checkpoint stores them: the projection `[mix, streams * width]`,
/// the base `[mix]` and the scale `[3]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MhcSiteWeights {
    projection: Weight,
    base: Weight,
    scale: Weight,
}

/// Which of a layer's two sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MhcSite {
    Attn,
    Ffn,
}

impl MhcSiteWeights {
    pub fn new(
        map: &WeightMap,
        layer: usize,
        site: MhcSite,
        width: usize,
        p: &MhcParams,
    ) -> Result<Self, WeightError> {
        let (projection, base, scale) = match site {
            MhcSite::Attn => (MhcRole::AttnFn, MhcRole::AttnBase, MhcRole::AttnScale),
            MhcSite::Ffn => (MhcRole::FfnFn, MhcRole::FfnBase, MhcRole::FfnScale),
        };
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Mhc(role)), shape)
        };
        Ok(Self {
            projection: w(projection, &[p.mix(), p.streams * width])?,
            base: w(base, &[p.mix()])?,
            scale: w(scale, &[3])?,
        })
    }

    /// [`mhc_pre`] of `x` over these weights.
    pub fn pre(&self, b: &Builder, x: Traced, p: &MhcParams) -> MhcMix {
        mhc_pre(
            b,
            x,
            operand(b, &self.projection, true),
            operand(b, &self.base, false),
            operand(b, &self.scale, false),
            p,
        )
    }
}

/// The model's closing collapse weights: the projection `[streams, streams * width]`, the base
/// `[streams]` and the scale `[1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MhcHeadWeights {
    projection: Weight,
    base: Weight,
    scale: Weight,
}

impl MhcHeadWeights {
    pub fn new(map: &WeightMap, width: usize, p: &MhcParams) -> Result<Self, WeightError> {
        let w =
            |role, shape: &[usize]| Weight::new(map, WeightId::model(WeightRole::Mhc(role)), shape);
        Ok(Self {
            projection: w(MhcRole::HeadFn, &[p.streams, p.streams * width])?,
            base: w(MhcRole::HeadBase, &[p.streams])?,
            scale: w(MhcRole::HeadScale, &[1])?,
        })
    }

    /// [`mhc_head`] of `x` over these weights.
    pub fn collapse(&self, b: &Builder, x: Traced, p: &MhcParams) -> Traced {
        mhc_head(
            b,
            x,
            operand(b, &self.projection, true),
            operand(b, &self.base, false),
            operand(b, &self.scale, false),
            p,
        )
    }
}

/// The widened streams of `embedded` (`[rows, tokens, width]`): every stream starts as a copy.
pub fn widen(b: &Builder, embedded: Traced, p: &MhcParams) -> Traced {
    let shape = b.aval(embedded).shape;
    let (rows, tokens, width) = (shape[0], shape[1], shape[2]);
    let one = b.reshape(embedded, vec![rows, tokens, 1, width]);
    b.broadcast(one, vec![rows, tokens, p.streams, width])
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::TensorType;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    const ROWS: usize = 2;
    const TOKENS: usize = 3;
    const HC: usize = 3;
    const WIDTH: usize = 5;
    const EPS: f64 = 1e-5;
    const HC_EPS: f64 = 1e-6;

    fn params(iters: usize) -> MhcParams {
        MhcParams::new(HC, EPS as f32, HC_EPS as f32, iters).unwrap()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn rms(x: &[f64]) -> Vec<f64> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + EPS).sqrt();
        x.iter().map(|v| v * inv).collect()
    }

    /// `x[in] @ w[in][out]`.
    fn project(x: &[f64], w: &[f32], out: usize) -> Vec<f64> {
        (0..out)
            .map(|o| (0..x.len()).map(|i| x[i] * f64::from(w[i * out + o])).sum())
            .collect()
    }

    /// The independent reference of one site over one token's streams `[hc][width]`:
    /// `(pre, post, comb, collapsed)`, written from the HF `DeepseekV4HyperConnection.forward`:
    /// split `[pre, post, comb]` out of the projection of the unweighted-normalized streams,
    /// `sigmoid + eps`, `2 * sigmoid`, a row softmax plus `eps` and one column normalization, then
    /// `sinkhorn_iters - 1` rounds of rows and columns.
    fn site_ref(
        streams: &[Vec<f64>],
        projection: &[f32],
        base: &[f32],
        scale: &[f32],
        iters: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let flat: Vec<f64> = streams.iter().flatten().copied().collect();
        let mix = project(&rms(&flat), projection, (2 + HC) * HC);
        let (base, scale): (Vec<f64>, Vec<f64>) = (
            base.iter().map(|&v| v.into()).collect(),
            scale.iter().map(|&v| v.into()).collect(),
        );
        let pre: Vec<f64> = (0..HC)
            .map(|i| sigmoid(mix[i] * scale[0] + base[i]) + HC_EPS)
            .collect();
        let post: Vec<f64> = (0..HC)
            .map(|i| 2.0 * sigmoid(mix[HC + i] * scale[1] + base[HC + i]))
            .collect();
        let mut comb = vec![vec![0.0; HC]; HC];
        for j in 0..HC {
            let logits: Vec<f64> = (0..HC)
                .map(|k| mix[2 * HC + j * HC + k] * scale[2] + base[2 * HC + j * HC + k])
                .collect();
            let max = logits.iter().cloned().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for k in 0..HC {
                comb[j][k] = exps[k] / sum + HC_EPS;
            }
        }
        let columns = |m: &mut Vec<Vec<f64>>| {
            for k in 0..HC {
                let s: f64 = (0..HC).map(|j| m[j][k]).sum::<f64>() + HC_EPS;
                for row in m.iter_mut() {
                    row[k] /= s;
                }
            }
        };
        let rows = |m: &mut Vec<Vec<f64>>| {
            for row in m.iter_mut() {
                let s: f64 = row.iter().sum::<f64>() + HC_EPS;
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        };
        columns(&mut comb);
        for _ in 1..iters {
            rows(&mut comb);
            columns(&mut comb);
        }
        let mut collapsed = vec![0.0; WIDTH];
        for (s, stream) in streams.iter().enumerate() {
            for (c, v) in collapsed.iter_mut().zip(stream) {
                *c += pre[s] * v;
            }
        }
        (pre, post, comb, collapsed)
    }

    /// `post[k] * out + sum_j comb[j][k] * residual[j]` per stream `k`: HF
    /// `DeepseekV4DecoderLayer.forward`, `matmul(comb.transpose(-1, -2), residual)`.
    fn place_ref(
        post: &[f64],
        comb: &[Vec<f64>],
        out: &[f64],
        residual: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        (0..HC)
            .map(|k| {
                (0..WIDTH)
                    .map(|c| {
                        post[k] * out[c] + (0..HC).map(|j| comb[j][k] * residual[j][c]).sum::<f64>()
                    })
                    .collect()
            })
            .collect()
    }

    /// The final collapse of HF `DeepseekV4HyperHead.forward`: `sigmoid + eps` of the projection of
    /// the normalized streams, weights the streams.
    fn head_ref(streams: &[Vec<f64>], projection: &[f32], base: &[f32], scale: f32) -> Vec<f64> {
        let flat: Vec<f64> = streams.iter().flatten().copied().collect();
        let mixes = project(&rms(&flat), projection, HC);
        let mut out = vec![0.0; WIDTH];
        for (s, stream) in streams.iter().enumerate() {
            let pre = sigmoid(mixes[s] * f64::from(scale) + f64::from(base[s])) + HC_EPS;
            for (o, v) in out.iter_mut().zip(stream) {
                *o += pre * v;
            }
        }
        out
    }

    fn streams_of(row: usize, token: usize, x: &[f32]) -> Vec<Vec<f64>> {
        (0..HC)
            .map(|s| {
                let at = (((row * TOKENS) + token) * HC + s) * WIDTH;
                x[at..at + WIDTH].iter().map(|&v| v.into()).collect()
            })
            .collect()
    }

    fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mix = (2 + HC) * HC;
        let scaled = |n, seed, k: f32| -> Vec<f32> {
            oracle::random(n, seed).into_iter().map(|v| k * v).collect()
        };
        (
            scaled(ROWS * TOKENS * HC * WIDTH, 1, 4.0),
            scaled(HC * WIDTH * mix, 2, 2.0),
            scaled(mix, 3, 2.0),
            scaled(3, 4, 4.0),
        )
    }

    /// SC-008 (F10): `mhc_pre` equals the independent reference at every token of every row, for
    /// one, three and twenty Sinkhorn rounds: `pre` through the collapsed stream, `post`, and every
    /// element of `comb`, which is doubly stochastic. Mutation: one Sinkhorn round fewer
    /// (`1..iters` becomes `2..iters`); the row sums leave one and the three- and twenty-round
    /// cases go red.
    #[test]
    fn mhc_pre_matches_the_independent_reference_and_comb_is_doubly_stochastic() {
        let (x, projection, base, scale) = inputs();
        for iters in [1, 3, 20] {
            let p = params(iters);
            let run = |pick: fn(&MhcMix) -> Traced| {
                let b = Builder::new();
                let xs = b.constant("x", TensorType::f32(vec![ROWS, TOKENS, HC, WIDTH]));
                let mix = mhc_pre(
                    &b,
                    xs,
                    b.constant(
                        "projection",
                        TensorType::f32(vec![HC * WIDTH, (2 + HC) * HC]),
                    ),
                    b.constant("base", TensorType::f32(vec![(2 + HC) * HC])),
                    b.constant("scale", TensorType::f32(vec![3])),
                    &p,
                );
                let out = pick(&mix);
                Fx::new(0)
                    .eval(
                        &b.finish(out),
                        &[
                            ("x", &x),
                            ("projection", &projection),
                            ("base", &base),
                            ("scale", &scale),
                        ],
                        &[],
                        &[],
                    )
                    .0
            };
            let (post, comb, collapsed) = (run(|m| m.post), run(|m| m.comb), run(|m| m.collapsed));
            for row in 0..ROWS {
                for token in 0..TOKENS {
                    let (_, want_post, want_comb, want_collapsed) = site_ref(
                        &streams_of(row, token, &x),
                        &projection,
                        &base,
                        &scale,
                        iters,
                    );
                    let at = row * TOKENS + token;
                    oracle::assert_matches_f64(&post[at * HC..(at + 1) * HC], &want_post, 1e-5);
                    oracle::assert_matches_f64(
                        &collapsed[at * WIDTH..(at + 1) * WIDTH],
                        &want_collapsed,
                        1e-5,
                    );
                    let flat: Vec<f64> = want_comb.iter().flatten().copied().collect();
                    oracle::assert_matches_f64(
                        &comb[at * HC * HC..(at + 1) * HC * HC],
                        &flat,
                        1e-5,
                    );
                    // The last round normalizes the columns, so they sum to one; the rows only
                    // approach it as the rounds go on (the equality with the reference carries the
                    // rest).
                    for column in 0..HC {
                        let col_sum: f64 = want_comb.iter().map(|row| row[column]).sum();
                        assert!((col_sum - 1.0).abs() < 1e-4, "column sum {col_sum}");
                    }
                }
            }
        }
    }

    /// `mhc_post` places `post * output` on every stream and mixes the old streams through
    /// `comb` transposed: with an asymmetric `comb` the transposed mix differs from the plain one,
    /// and the result equals the reference. Mutation: mix through `comb` untransposed; red.
    #[test]
    fn mhc_post_mixes_the_residual_through_comb_transposed() {
        let (x, ..) = inputs();
        let post = oracle::random(ROWS * TOKENS * HC, 5);
        // Distinct entries, so the transpose is visible.
        let comb: Vec<f32> = (0..ROWS * TOKENS * HC * HC)
            .map(|i| 0.1 + 0.05 * (i % 7) as f32)
            .collect();
        let out = oracle::random(ROWS * TOKENS * WIDTH, 6);
        let b = Builder::new();
        let y = mhc_post(
            &b,
            b.constant("post", TensorType::f32(vec![ROWS, TOKENS, HC])),
            b.constant("comb", TensorType::f32(vec![ROWS, TOKENS, HC, HC])),
            b.constant("out", TensorType::f32(vec![ROWS, TOKENS, WIDTH])),
            b.constant("x", TensorType::f32(vec![ROWS, TOKENS, HC, WIDTH])),
            &params(2),
        );
        let got = Fx::new(0)
            .eval(
                &b.finish(y),
                &[("post", &post), ("comb", &comb), ("out", &out), ("x", &x)],
                &[],
                &[],
            )
            .0;
        for row in 0..ROWS {
            for token in 0..TOKENS {
                let at = row * TOKENS + token;
                let comb_rows: Vec<Vec<f64>> = (0..HC)
                    .map(|j| {
                        (0..HC)
                            .map(|k| f64::from(comb[at * HC * HC + j * HC + k]))
                            .collect()
                    })
                    .collect();
                let post_row: Vec<f64> = post[at * HC..(at + 1) * HC]
                    .iter()
                    .map(|&v| v.into())
                    .collect();
                let out_row: Vec<f64> = out[at * WIDTH..(at + 1) * WIDTH]
                    .iter()
                    .map(|&v| v.into())
                    .collect();
                let want = place_ref(&post_row, &comb_rows, &out_row, &streams_of(row, token, &x));
                let flat: Vec<f64> = want.iter().flatten().copied().collect();
                oracle::assert_matches_f64(
                    &got[at * HC * WIDTH..(at + 1) * HC * WIDTH],
                    &flat,
                    1e-5,
                );
            }
        }
    }

    /// `mhc_head` equals the reference collapse at every token.
    #[test]
    fn mhc_head_matches_the_independent_reference() {
        let (x, ..) = inputs();
        let projection = oracle::random(HC * WIDTH * HC, 7);
        let (base, scale) = (oracle::random(HC, 8), [0.7f32]);
        let b = Builder::new();
        let y = mhc_head(
            &b,
            b.constant("x", TensorType::f32(vec![ROWS, TOKENS, HC, WIDTH])),
            b.constant("projection", TensorType::f32(vec![HC * WIDTH, HC])),
            b.constant("base", TensorType::f32(vec![HC])),
            b.constant("scale", TensorType::f32(vec![1])),
            &params(1),
        );
        let got = Fx::new(0)
            .eval(
                &b.finish(y),
                &[
                    ("x", &x),
                    ("projection", &projection),
                    ("base", &base),
                    ("scale", &scale),
                ],
                &[],
                &[],
            )
            .0;
        for row in 0..ROWS {
            for token in 0..TOKENS {
                let want = head_ref(&streams_of(row, token, &x), &projection, &base, scale[0]);
                let at = row * TOKENS + token;
                oracle::assert_matches_f64(&got[at * WIDTH..(at + 1) * WIDTH], &want, 1e-5);
            }
        }
    }

    /// The site's weights read as stored: a BF16 `[mix, streams * width]` projection transposed and
    /// widened, the base and scale as F32, and `MhcParams` refuses a stream or round count of zero
    /// and a non-finite epsilon by field.
    #[test]
    fn site_weights_read_the_stored_orientation_and_params_are_checked() {
        let mix = (2 + HC) * HC;
        let id = |role| WeightId::layer(0, WeightRole::Mhc(role));
        let mut fx = Fx::new(9);
        fx.bf16(id(MhcRole::AttnFn), &[mix, HC * WIDTH]);
        fx.f32(id(MhcRole::AttnBase), &[mix]);
        fx.f32(id(MhcRole::AttnScale), &[3]);
        let p = params(3);
        let w = MhcSiteWeights::new(&fx.map(), 0, MhcSite::Attn, WIDTH, &p).unwrap();
        let (x, ..) = inputs();
        let b = Builder::new();
        let xs = b.constant("x", TensorType::f32(vec![ROWS, TOKENS, HC, WIDTH]));
        let collapsed = w.pre(&b, xs, &p).collapsed;
        let got = fx.eval(&b.finish(collapsed), &[("x", &x)], &[], &[]).0;
        let stored = fx.values(id(MhcRole::AttnFn)); // [mix][hc * width]
        let mut projection = vec![0.0f32; HC * WIDTH * mix]; // [in][out]
        for o in 0..mix {
            for i in 0..HC * WIDTH {
                projection[i * mix + o] = stored[o * HC * WIDTH + i];
            }
        }
        let (_, _, _, want) = site_ref(
            &streams_of(1, 2, &x),
            &projection,
            &fx.values(id(MhcRole::AttnBase)),
            &fx.values(id(MhcRole::AttnScale)),
            3,
        );
        let at = TOKENS + 2;
        oracle::assert_matches_f64(&got[at * WIDTH..(at + 1) * WIDTH], &want, 1e-5);

        for (args, field) in [
            ((0, 1e-5, 1e-6, 1), "hc_mult"),
            ((2, 1e-5, 1e-6, 0), "hc_sinkhorn_iters"),
            ((2, f32::NAN, 1e-6, 1), "rms_norm_eps"),
            ((2, 1e-5, 0.0, 1), "hc_eps"),
        ] {
            let err = MhcParams::new(args.0, args.1, args.2, args.3).unwrap_err();
            assert_eq!(err.field, field);
        }
    }
}

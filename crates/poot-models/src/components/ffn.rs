//! The gated (SwiGLU) feed-forward block: `down(silu(gate(x)) * up(x))`.

use poot_graph_ir::{BinOp, Builder, Scalar, Traced, UnOp, ops};
use poot_quant::weights::{FfnRole, WeightId, WeightMap, WeightRole};

use super::linear::{Weight, WeightError, linear};
use super::standard::ParamError;
use crate::model::ConfigReason;

/// A layer's `gate`/`up` (`[inter, width]`) and `down` (`[width, inter]`) projections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatedFfnWeights {
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

impl GatedFfnWeights {
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        inter: usize,
    ) -> Result<Self, WeightError> {
        let w = |role, shape: [usize; 2]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Ffn(role)), &shape)
        };
        Ok(Self {
            gate: w(FfnRole::Gate, [inter, width])?,
            up: w(FfnRole::Up, [inter, width])?,
            down: w(FfnRole::Down, [width, inter])?,
        })
    }
}

/// `down(silu(gate(x)) * up(x))` over `[.., width]`.
pub fn gated_ffn(b: &Builder, x: Traced, w: &GatedFfnWeights) -> Traced {
    let gate = linear(b, x, &w.gate, None);
    let up = linear(b, x, &w.up, None);
    linear(b, ops::swiglu(b, gate, up), &w.down, None)
}

/// `down(gelu_tanh(gate(x)) * up(x))` over `[.., width]` (the GeGLU block): the same weights as
/// [`gated_ffn`] with the tanh-approximated GELU as the gate.
pub fn geglu_ffn(b: &Builder, x: Traced, w: &GatedFfnWeights) -> Traced {
    let gate = linear(b, x, &w.gate, None);
    let up = linear(b, x, &w.up, None);
    linear(b, ops::geglu(b, gate, up), &w.down, None)
}

/// The gate of a gated feed-forward block: how `gate` and `up` combine before the down projection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Glu {
    /// `silu(gate) * up` (SwiGLU).
    Silu,
    /// `gelu_tanh(gate) * up` (GeGLU, Gemma).
    Gelu,
    /// gpt-oss's clamped swish GLU: `gate` clamped above at `limit`, `up` clamped to
    /// `[-limit, limit]`, then `gate * sigmoid(alpha * gate) * (up + 1)`.
    Clamped { alpha: f32, limit: f32 },
}

impl Glu {
    /// The clamped GLU with its `alpha` and `limit`, both finite and positive.
    pub fn clamped(alpha: f32, limit: f32) -> Result<Self, ParamError> {
        for (field, value) in [("glu_alpha", alpha), ("glu_limit", limit)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ParamError::new(field, ConfigReason::NotFinitePositive));
            }
        }
        Ok(Self::Clamped { alpha, limit })
    }
}

/// `gate` and `up` (same shape) combined by `glu`.
pub fn glu(b: &Builder, gate: Traced, up: Traced, glu: &Glu) -> Traced {
    match *glu {
        Glu::Silu => ops::swiglu(b, gate, up),
        Glu::Gelu => ops::geglu(b, gate, up),
        Glu::Clamped { alpha, limit } => {
            // `min(x, c) = -max(-x, -c)`: there is no Min or Clamp primitive.
            let below = |x: Traced| {
                let negated = b.unary(UnOp::Neg, x);
                let capped = b.binary_scalar(BinOp::Max, negated, Scalar::F32(-limit));
                b.unary(UnOp::Neg, capped)
            };
            let gate = below(gate);
            let up = below(b.binary_scalar(BinOp::Max, up, Scalar::F32(-limit)));
            let swish = b.binary(
                BinOp::Mul,
                gate,
                ops::sigmoid(b, b.binary_scalar(BinOp::Mul, gate, Scalar::F32(alpha))),
            );
            b.binary(
                BinOp::Mul,
                b.binary_scalar(BinOp::Add, up, Scalar::F32(1.0)),
                swish,
            )
        }
    }
}

/// The activation of a plain MLP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gelu {
    /// `x (1 + erf(x / sqrt 2)) / 2` (MPT).
    Erf,
    /// The tanh approximation (BLOOM).
    Tanh,
}

/// A layer's `up` (`[inter, width]`) and `down` (`[width, inter]`) projections of a plain MLP, and
/// their biases when the model has them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlainMlpWeights {
    pub up: Weight,
    pub down: Weight,
    pub bias: Option<[Weight; 2]>,
}

impl PlainMlpWeights {
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        inter: usize,
        biased: bool,
    ) -> Result<Self, WeightError> {
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Ffn(role)), shape)
        };
        let bias = if biased {
            Some([
                w(FfnRole::UpBias, &[inter])?,
                w(FfnRole::DownBias, &[width])?,
            ])
        } else {
            None
        };
        Ok(Self {
            up: w(FfnRole::Up, &[inter, width])?,
            down: w(FfnRole::Down, &[width, inter])?,
            bias,
        })
    }
}

/// The non-gated feed-forward block with the squared-ReLU activation, `down(relu(up(x))^2)` over
/// `[.., width]` (Nemotron-H's `relu2`): the same weights as [`plain_mlp`].
pub fn relu2_mlp(b: &Builder, x: Traced, w: &PlainMlpWeights) -> Traced {
    let bias = |i: usize| w.bias.as_ref().map(|biases| &biases[i]);
    let relu = ops::relu(b, linear(b, x, &w.up, bias(0)));
    linear(b, b.binary(BinOp::Mul, relu, relu), &w.down, bias(1))
}

/// The non-gated feed-forward block `down(gelu(up(x)))` over `[.., width]`.
pub fn plain_mlp(b: &Builder, x: Traced, w: &PlainMlpWeights, gelu: Gelu) -> Traced {
    let bias = |i: usize| w.bias.as_ref().map(|biases| &biases[i]);
    let up = linear(b, x, &w.up, bias(0));
    let act = match gelu {
        Gelu::Erf => ops::gelu_erf(b, up),
        Gelu::Tanh => ops::gelu(b, up),
    };
    linear(b, act, &w.down, bias(1))
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::TensorType;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    /// erf by its Maclaurin series: `2/sqrt(pi) * sum (-1)^n x^(2n+1) / (n! (2n+1))`, exact to f64
    /// precision for the |x| < 4 this test reaches.
    fn erf(x: f64) -> f64 {
        let (mut term, mut sum) = (x, x);
        for n in 1..80 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    /// `plain_mlp` against an f64 `down(gelu(up(x) + b_up)) + b_down`; every weight matters.
    fn assert_plain_mlp_matches(gelu: Gelu, biased: bool) {
        let (t, w, inter) = (2, 4, 6);
        let id = |role| WeightId::layer(1, WeightRole::Ffn(role));
        let mut fx = Fx::new(8);
        fx.bf16(id(FfnRole::Up), &[inter, w]);
        fx.f32(id(FfnRole::Down), &[w, inter]);
        if biased {
            fx.f32(id(FfnRole::UpBias), &[inter]);
            fx.bf16(id(FfnRole::DownBias), &[w]);
        }
        let x = oracle::random(t * w, 5)
            .into_iter()
            .map(|v| 4.0 * v)
            .collect::<Vec<_>>();
        let run = |fx: &Fx| {
            let weights = PlainMlpWeights::new(&fx.map(), 1, w, inter, biased).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = plain_mlp(&b, xs, &weights, gelu);
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let (u, d) = (fx.values(id(FfnRole::Up)), fx.values(id(FfnRole::Down)));
        let bias = |role, n: usize| {
            if biased {
                fx.values(id(role))
            } else {
                vec![0.0; n]
            }
        };
        let (bu, bd) = (bias(FfnRole::UpBias, inter), bias(FfnRole::DownBias, w));
        let act = |a: f64| match gelu {
            Gelu::Erf => a * 0.5 * (1.0 + erf(a / std::f64::consts::SQRT_2)),
            Gelu::Tanh => {
                let k = (2.0 / std::f64::consts::PI).sqrt();
                0.5 * a * (1.0 + (k * (a + 0.044715 * a * a * a)).tanh())
            }
        };
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let v: Vec<f64> = row.iter().map(|&x| x.into()).collect();
            let hidden: Vec<f64> = (0..inter)
                .map(|o| {
                    let a: f64 = (0..w).map(|j| v[j] * f64::from(u[o * w + j])).sum();
                    act(a + f64::from(bu[o]))
                })
                .collect();
            want.extend((0..w).map(|o| {
                (0..inter)
                    .map(|j| hidden[j] * f64::from(d[o * inter + j]))
                    .sum::<f64>()
                    + f64::from(bd[o])
            }));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        let mut roles = vec![FfnRole::Up, FfnRole::Down];
        if biased {
            roles.extend([FfnRole::UpBias, FfnRole::DownBias]);
        }
        for role in roles {
            let mut p = fx.clone();
            p.perturb(id(role));
            oracle::assert_differs(&run(&p), &got, id(role));
        }
    }

    /// SC-001: the exact-GELU plain MLP with no biases (MPT).
    #[test]
    fn plain_mlp_matches_an_f64_exact_gelu_mlp() {
        assert_plain_mlp_matches(Gelu::Erf, false);
    }

    /// SC-001: the tanh-GELU plain MLP with both biases (BLOOM); it differs from the exact one.
    #[test]
    fn plain_mlp_with_biases_and_tanh_gelu_matches_an_f64_reference() {
        assert_plain_mlp_matches(Gelu::Tanh, true);
    }

    /// SC-001: `relu2_mlp` against an f64 `down(relu(up(x) + b_up)^2) + b_down`, with and without
    /// biases; every weight matters.
    #[test]
    fn relu2_mlp_matches_an_f64_squared_relu_mlp() {
        let (t, w, inter) = (2, 4, 6);
        let id = |role| WeightId::layer(1, WeightRole::Ffn(role));
        for biased in [false, true] {
            let mut fx = Fx::new(12);
            fx.bf16(id(FfnRole::Up), &[inter, w]);
            fx.f32(id(FfnRole::Down), &[w, inter]);
            if biased {
                fx.f32(id(FfnRole::UpBias), &[inter]);
                fx.f32(id(FfnRole::DownBias), &[w]);
            }
            let x = oracle::random(t * w, 6)
                .into_iter()
                .map(|v| 4.0 * v)
                .collect::<Vec<_>>();
            let run = |fx: &Fx| {
                let weights = PlainMlpWeights::new(&fx.map(), 1, w, inter, biased).unwrap();
                let b = Builder::new();
                let xs = b.constant("x", TensorType::f32(vec![t, w]));
                let y = relu2_mlp(&b, xs, &weights);
                fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
            };
            let got = run(&fx);
            let (u, d) = (fx.values(id(FfnRole::Up)), fx.values(id(FfnRole::Down)));
            let bias = |role, n: usize| {
                if biased {
                    fx.values(id(role))
                } else {
                    vec![0.0; n]
                }
            };
            let (bu, bd) = (bias(FfnRole::UpBias, inter), bias(FfnRole::DownBias, w));
            let mut want = Vec::new();
            for row in x.chunks(w) {
                let hidden: Vec<f64> = (0..inter)
                    .map(|o| {
                        let a: f64 = (0..w)
                            .map(|j| f64::from(row[j]) * f64::from(u[o * w + j]))
                            .sum::<f64>()
                            + f64::from(bu[o]);
                        a.max(0.0).powi(2)
                    })
                    .collect();
                want.extend((0..w).map(|o| {
                    (0..inter)
                        .map(|j| hidden[j] * f64::from(d[o * inter + j]))
                        .sum::<f64>()
                        + f64::from(bd[o])
                }));
            }
            oracle::assert_matches_f64(&got, &want, 1e-5);
            let mut roles = vec![FfnRole::Up, FfnRole::Down];
            if biased {
                roles.extend([FfnRole::UpBias, FfnRole::DownBias]);
            }
            for role in roles {
                let mut p = fx.clone();
                p.perturb(id(role));
                oracle::assert_differs(&run(&p), &got, id(role));
            }
        }
    }

    /// SC-001: `geglu_ffn` against an f64 `down(gelu_tanh(gate(x)) * up(x))`; each of the three
    /// projections matters, and it differs from the SwiGLU block on the same weights.
    #[test]
    fn geglu_ffn_matches_an_f64_geglu_mlp() {
        let (t, w, inter) = (2, 4, 6);
        let id = |role| WeightId::layer(1, WeightRole::Ffn(role));
        let mut fx = Fx::new(16);
        fx.bf16(id(FfnRole::Gate), &[inter, w]);
        fx.f32(id(FfnRole::Up), &[inter, w]);
        fx.bf16(id(FfnRole::Down), &[w, inter]);
        let x = oracle::random(t * w, 13)
            .into_iter()
            .map(|v| 4.0 * v)
            .collect::<Vec<_>>();
        let run = |fx: &Fx, f: fn(&Builder, Traced, &GatedFfnWeights) -> Traced| {
            let weights = GatedFfnWeights::new(&fx.map(), 1, w, inter).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = f(&b, xs, &weights);
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx, geglu_ffn);
        let (g, u, d) = (
            fx.values(id(FfnRole::Gate)),
            fx.values(id(FfnRole::Up)),
            fx.values(id(FfnRole::Down)),
        );
        let proj = |m: &[f32], v: &[f64], out: usize, k: usize| -> Vec<f64> {
            (0..out)
                .map(|o| (0..k).map(|j| v[j] * f64::from(m[o * k + j])).sum())
                .collect()
        };
        let gelu = |a: f64| {
            0.5 * a
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (a + 0.044715 * a * a * a)).tanh())
        };
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let v: Vec<f64> = row.iter().map(|&x| x.into()).collect();
            let act: Vec<f64> = proj(&g, &v, inter, w)
                .iter()
                .zip(proj(&u, &v, inter, w))
                .map(|(&a, b)| gelu(a) * b)
                .collect();
            want.extend(proj(&d, &act, w, inter));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        assert_ne!(got, run(&fx, gated_ffn), "GeGLU is not SwiGLU");
        for role in [FfnRole::Gate, FfnRole::Up, FfnRole::Down] {
            let mut p = fx.clone();
            p.perturb(id(role));
            oracle::assert_differs(&run(&p, geglu_ffn), &got, id(role));
        }
    }

    /// SC-002: the clamped GLU against an f64 `min(gate, limit) * sigmoid(alpha * min(gate,
    /// limit)) * (clamp(up, -limit, limit) + 1)` over inputs reaching past the limit on both
    /// sides; a higher limit gives different values (the clamp is live), and the gate and the up
    /// input each change the output. Mutation: clamp `up` from above only; red.
    #[test]
    fn clamped_glu_matches_an_f64_reference() {
        let (alpha, limit) = (1.702f32, 3.0f32);
        let n = 24;
        let scaled = |seed| -> Vec<f32> {
            oracle::random(n, seed)
                .into_iter()
                .map(|v| 16.0 * v)
                .collect()
        };
        let (gate, up) = (scaled(21), scaled(22));
        let run = |gate: &[f32], up: &[f32], limit: f32| {
            let b = Builder::new();
            let g = b.constant("gate", TensorType::f32(vec![n]));
            let u = b.constant("up", TensorType::f32(vec![n]));
            let y = glu(&b, g, u, &Glu::clamped(alpha, limit).unwrap());
            Fx::new(1)
                .eval(&b.finish(y), &[("gate", gate), ("up", up)], &[], &[])
                .0
        };
        let got = run(&gate, &up, limit);
        let (alpha, limit) = (f64::from(alpha), f64::from(limit));
        let want: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let g = f64::from(g).min(limit);
                let u = f64::from(u).clamp(-limit, limit);
                (u + 1.0) * g / (1.0 + (-alpha * g).exp())
            })
            .collect();
        oracle::assert_matches_f64(&got, &want, 1e-5);
        assert!(
            gate.iter().any(|&g| f64::from(g) > limit)
                && up.iter().any(|&u| f64::from(u).abs() > limit),
            "the inputs must reach past the limit"
        );
        assert_ne!(run(&gate, &up, 100.0), got, "the clamp changes the output");
        let moved: Vec<f32> = gate.iter().map(|g| g - 0.5).collect();
        assert_ne!(run(&moved, &up, 3.0), got);
        let moved: Vec<f32> = up.iter().map(|u| u + 0.5).collect();
        assert_ne!(run(&gate, &moved, 3.0), got);
    }

    #[test]
    fn a_clamped_glu_refuses_a_non_positive_or_non_finite_parameter() {
        for (alpha, limit, field) in [
            (0.0, 7.0, "glu_alpha"),
            (f32::NAN, 7.0, "glu_alpha"),
            (1.7, 0.0, "glu_limit"),
            (1.7, f32::INFINITY, "glu_limit"),
        ] {
            assert_eq!(
                Glu::clamped(alpha, limit),
                Err(ParamError::new(field, ConfigReason::NotFinitePositive))
            );
        }
    }

    /// SC-001: `gated_ffn` against an f64 SwiGLU MLP; each of the three projections matters.
    #[test]
    fn gated_ffn_matches_an_f64_swiglu_mlp() {
        let (t, w, inter) = (2, 4, 6);
        let id = |role| WeightId::layer(1, WeightRole::Ffn(role));
        let mut fx = Fx::new(6);
        fx.bf16(id(FfnRole::Gate), &[inter, w]);
        fx.f32(id(FfnRole::Up), &[inter, w]);
        fx.bf16(id(FfnRole::Down), &[w, inter]);
        let x = oracle::random(t * w, 3)
            .into_iter()
            .map(|v| 4.0 * v)
            .collect::<Vec<_>>();
        let run = |fx: &Fx| {
            let weights = GatedFfnWeights::new(&fx.map(), 1, w, inter).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = gated_ffn(&b, xs, &weights);
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let (g, u, d) = (
            fx.values(id(FfnRole::Gate)),
            fx.values(id(FfnRole::Up)),
            fx.values(id(FfnRole::Down)),
        );
        let proj = |m: &[f32], v: &[f64], out: usize, k: usize| -> Vec<f64> {
            (0..out)
                .map(|o| (0..k).map(|j| v[j] * f64::from(m[o * k + j])).sum())
                .collect()
        };
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let v: Vec<f64> = row.iter().map(|&x| x.into()).collect();
            let act: Vec<f64> = proj(&g, &v, inter, w)
                .iter()
                .zip(proj(&u, &v, inter, w))
                .map(|(&a, b)| a / (1.0 + (-a).exp()) * b)
                .collect();
            want.extend(proj(&d, &act, w, inter));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for role in [FfnRole::Gate, FfnRole::Up, FfnRole::Down] {
            let mut p = fx.clone();
            p.perturb(id(role));
            oracle::assert_differs(&run(&p), &got, id(role));
        }
    }
}

//! The gated (SwiGLU) feed-forward block: `down(silu(gate(x)) * up(x))`.

use poot_graph_ir::{Builder, Traced, ops};
use poot_quant::weights::{FfnRole, WeightId, WeightMap, WeightRole};

use super::linear::{Weight, WeightError, linear};

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

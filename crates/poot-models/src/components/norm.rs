//! RMSNorm and LayerNorm over a mapped scale weight.

use poot_graph_ir::{BinOp, Builder, Traced, ops};

use super::linear::Weight;
use super::standard::ParamError;
use crate::model::ConfigReason;

/// Which normalization a layer applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    /// `x * rsqrt(mean(x^2) + eps) * w`.
    Rms,
    /// `(x - mean) / sqrt(var + eps) * w`, with no bias (MPT's profile).
    Layer,
    /// `x * rsqrt(mean(x^2) + eps) * (1 + w)`: the scale is stored as an offset from one (HF's
    /// Gemma checkpoints; llama.cpp's GGUFs fold the one in and use [`NormKind::Rms`]).
    RmsPlusOne,
}

/// The norm's kind and epsilon (finite and positive).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormParams {
    eps: f32,
    kind: NormKind,
}

impl NormParams {
    /// An RMSNorm.
    pub fn new(eps: f32) -> Result<Self, ParamError> {
        Self::of(NormKind::Rms, eps)
    }

    pub fn of(kind: NormKind, eps: f32) -> Result<Self, ParamError> {
        if !(eps.is_finite() && eps > 0.0) {
            return Err(ParamError::new("eps", ConfigReason::NotFinitePositive));
        }
        Ok(Self { eps, kind })
    }

    pub fn eps(self) -> f32 {
        self.eps
    }

    pub fn kind(self) -> NormKind {
        self.kind
    }
}

/// A norm's scale and, for a LayerNorm that has one, its bias (both `[width]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NormWeights {
    pub scale: Weight,
    pub bias: Option<Weight>,
}

/// `p`'s normalization of `x` over the last axis, scaled by `w.scale` and shifted by `w.bias`.
pub fn norm_with(b: &Builder, x: Traced, w: &NormWeights, p: NormParams) -> Traced {
    let y = norm(b, x, &w.scale, p);
    match &w.bias {
        Some(bias) => {
            let shape = b.aval(y).shape;
            b.binary(BinOp::Add, y, b.broadcast(bias.declare_f32(b), shape))
        }
        None => y,
    }
}

/// `p`'s normalization of `x` over the last axis, scaled by `w`.
pub fn norm(b: &Builder, x: Traced, w: &Weight, p: NormParams) -> Traced {
    let w = w.declare_f32(b);
    match p.kind {
        NormKind::Rms => ops::rmsnorm(b, x, w, p.eps),
        NormKind::Layer => ops::layernorm_no_bias(b, x, w, p.eps),
        NormKind::RmsPlusOne => {
            let w = b.binary_scalar(BinOp::Add, w, Builder::f32(1.0));
            ops::rmsnorm(b, x, w, p.eps)
        }
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::TensorType;
    use poot_quant::weights::{WeightId, WeightRole};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};

    /// SC-001: `rms_norm` with a BF16 scale against an f64 RMSNorm, and the scale matters.
    #[test]
    fn rms_norm_matches_an_f64_reference_and_depends_on_its_scale() {
        let (t, w) = (2, 8);
        let id = WeightId::model(WeightRole::FinalNorm);
        let mut fx = Fx::new(2);
        fx.bf16(id, &[w]);
        let x = oracle::random(t * w, 4);
        let eps = 1e-5f32;
        let run = |fx: &Fx| {
            let weight = Weight::new(&fx.map(), id, &[w]).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = norm(&b, xs, &weight, NormParams::new(eps).unwrap());
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx);
        let scale = fx.values(id);
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let ms = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / w as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            want.extend(
                row.iter()
                    .zip(&scale)
                    .map(|(&v, &s)| f64::from(v) * inv * f64::from(s)),
            );
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        let mut p = fx.clone();
        p.perturb(id);
        oracle::assert_differs(&run(&p), &got, id);
    }

    /// SC-001: the LayerNorm kind against an f64 `(x - mean) / sqrt(var + eps) * w`, which differs
    /// from the RMS kind on the same input (the mean matters), and the scale matters.
    #[test]
    fn layer_norm_matches_an_f64_reference_and_differs_from_rms() {
        let (t, w) = (2, 8);
        let id = WeightId::model(WeightRole::FinalNorm);
        let mut fx = Fx::new(3);
        fx.bf16(id, &[w]);
        let x: Vec<f32> = oracle::random(t * w, 6).iter().map(|v| v + 0.7).collect();
        let eps = 1e-5f32;
        let run = |fx: &Fx, kind| {
            let weight = Weight::new(&fx.map(), id, &[w]).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = norm(&b, xs, &weight, NormParams::of(kind, eps).unwrap());
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx, NormKind::Layer);
        let scale = fx.values(id);
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / w as f64;
            let var = row
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / w as f64;
            let inv = 1.0 / (var + f64::from(eps)).sqrt();
            want.extend(
                row.iter()
                    .zip(&scale)
                    .map(|(&v, &s)| (f64::from(v) - mean) * inv * f64::from(s)),
            );
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        assert_ne!(got, run(&fx, NormKind::Rms), "the kinds must differ");
        let mut p = fx.clone();
        p.perturb(id);
        oracle::assert_differs(&run(&p, NormKind::Layer), &got, id);
    }

    /// SC-001: the offset RMS kind against an f64 `x * rsqrt(ms + eps) * (1 + w)`; it differs from
    /// the plain kind, and the scale matters.
    #[test]
    fn rms_plus_one_matches_an_f64_reference_and_differs_from_rms() {
        let (t, w) = (2, 8);
        let id = WeightId::model(WeightRole::FinalNorm);
        let mut fx = Fx::new(5);
        fx.bf16(id, &[w]);
        let x = oracle::random(t * w, 8);
        let eps = 1e-5f32;
        let run = |fx: &Fx, kind| {
            let weight = Weight::new(&fx.map(), id, &[w]).unwrap();
            let b = Builder::new();
            let xs = b.constant("x", TensorType::f32(vec![t, w]));
            let y = norm(&b, xs, &weight, NormParams::of(kind, eps).unwrap());
            fx.eval(&b.finish(y), &[("x", &x)], &[], &[]).0
        };
        let got = run(&fx, NormKind::RmsPlusOne);
        let scale = fx.values(id);
        let mut want = Vec::new();
        for row in x.chunks(w) {
            let ms = row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / w as f64;
            let inv = 1.0 / (ms + f64::from(eps)).sqrt();
            want.extend(
                row.iter()
                    .zip(&scale)
                    .map(|(&v, &s)| f64::from(v) * inv * (1.0 + f64::from(s))),
            );
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        assert_ne!(got, run(&fx, NormKind::Rms));
        let mut p = fx.clone();
        p.perturb(id);
        oracle::assert_differs(&run(&p, NormKind::RmsPlusOne), &got, id);
    }

    #[test]
    fn a_non_positive_or_non_finite_epsilon_is_refused_by_field() {
        for eps in [0.0, -1e-6, f32::NAN, f32::INFINITY] {
            assert_eq!(
                NormParams::new(eps),
                Err(ParamError::new("eps", ConfigReason::NotFinitePositive)),
                "{eps}"
            );
        }
    }
}

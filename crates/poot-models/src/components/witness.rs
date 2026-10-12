//! Runtime witnesses of a routed layer: bounded F32 lanes that are zero while the layer is sound
//! and non-zero when it is not (spec 999, ADR-0101 decision 2). The executor checks every declared
//! lane before it reads any output back, so a fault stops the step instead of publishing garbage.
//!
//! A witness has a [`WitnessKind`] and a layer; its validation id is `layer * KINDS + kind`, so the
//! id alone names both ([`WitnessKind::decode`]) and a failure maps back to its condition without
//! parsing a name.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, UnOp, ValidationId};
use poot_tensor::DType;

use super::standard::Step;
use crate::model::TraceError;
use crate::model::WiringError;

/// The three conditions a router's run-time check observes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WitnessKind {
    /// A router logit is NaN or infinite.
    RouterLogitsFinite,
    /// The selected experts' score sum is not a finite positive number, so the normalization would
    /// publish NaN instead of a convex combination.
    SelectedSumUsable,
    /// A selected expert id fell outside `0..experts`.
    SelectorInRange,
}

impl WitnessKind {
    pub const ALL: [Self; 3] = [
        Self::RouterLogitsFinite,
        Self::SelectedSumUsable,
        Self::SelectorInRange,
    ];

    const fn ordinal(self) -> u32 {
        match self {
            Self::RouterLogitsFinite => 0,
            Self::SelectedSumUsable => 1,
            Self::SelectorInRange => 2,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::RouterLogitsFinite => "router_logits_finite",
            Self::SelectedSumUsable => "selected_sum_usable",
            Self::SelectorInRange => "selector_in_range",
        }
    }

    /// The validation id of this condition on `layer`.
    pub fn id(self, layer: usize) -> Result<ValidationId, WiringError> {
        let kinds = Self::ALL.len() as u32;
        let id = u32::try_from(layer)
            .ok()
            .and_then(|layer| layer.checked_mul(kinds))
            .and_then(|base| base.checked_add(self.ordinal()))
            .ok_or(WiringError::LayerIndexOutOfRange { layer })?;
        Ok(ValidationId(id))
    }

    /// `(layer, kind)` of an id [`Self::id`] minted; `None` for any other id.
    pub fn decode(id: ValidationId) -> Option<(usize, Self)> {
        let kinds = Self::ALL.len() as u32;
        let kind = Self::ALL
            .into_iter()
            .find(|k| k.ordinal() == id.0 % kinds)?;
        Some(((id.0 / kinds) as usize, kind))
    }
}

/// Declare `value` as the witness of `kind` on `layer` of `step`.
pub fn declare(
    step: &Step,
    layer: usize,
    kind: WitnessKind,
    value: Traced,
) -> Result<(), WiringError> {
    step.validate(kind.id(layer)?, format!("{}.l{layer}", kind.label()), value);
    Ok(())
}

/// Sum every axis of `x` away, so a witness costs one validation lane.
fn sum_to_scalar(b: &Builder, x: Traced) -> Result<Traced, TraceError> {
    let mut value = x;
    while !b.aval(value)?.shape.is_empty() {
        value = b.reduce(RedOp::Sum, value, 0, false)?;
    }
    Ok(value)
}

/// `1 - indicator`, for an indicator that is already 0.0 or 1.0.
fn logical_not(b: &Builder, indicator: Traced) -> Result<Traced, TraceError> {
    let negated = b.binary_scalar(BinOp::Mul, indicator, Scalar::F32(-1.0))?;
    Ok(b.binary_scalar(BinOp::Add, negated, Scalar::F32(1.0))?)
}

/// The number of non-finite values of `x`. `Ge(x, x)` is false exactly for NaN and `Ge(|x|, inf)`
/// true exactly for infinities; there is no `Abs`, so `Max(x, -x)` is the magnitude.
pub fn nonfinite_count(b: &Builder, x: Traced) -> Result<Traced, TraceError> {
    let nan = logical_not(b, b.binary(BinOp::Ge, x, x)?)?;
    let magnitude = b.binary(BinOp::Max, x, b.unary(UnOp::Neg, x)?)?;
    let infinite = b.binary_scalar(BinOp::Ge, magnitude, Scalar::F32(f32::INFINITY))?;
    sum_to_scalar(b, b.binary(BinOp::Add, nan, infinite)?)
}

/// The number of entries of `sum` that are not finite positive numbers. Comparisons keep the value
/// on the left (the planner rejects a literal left operand) and are false for NaN.
pub fn unusable_sum_count(b: &Builder, sum: Traced) -> Result<Traced, TraceError> {
    let negated = b.binary_scalar(BinOp::Mul, sum, Scalar::F32(-1.0))?;
    // sum <= f32::MAX is false for +inf, sum >= f32::MIN false for -inf, sum > 0 false at zero.
    let below_max = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(-f32::MAX))?;
    let above_min = b.binary_scalar(BinOp::Ge, sum, Scalar::F32(f32::MIN))?;
    let positive = logical_not(b, b.binary_scalar(BinOp::Ge, negated, Scalar::F32(0.0))?)?;
    let usable = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Mul, below_max, above_min)?,
        positive,
    )?;
    sum_to_scalar(b, logical_not(b, usable)?)
}

/// The number of selected ids outside `0..experts`. The ids are I32 out of `ArgTopK`; one unsigned compare
/// (`GeU`) is an exact I32 `0`/`1` indicator of an id at or past `experts` or negative, widened to F32, so this
/// is an ordinary F32 witness, not an in-graph clamp.
pub fn out_of_range_count(b: &Builder, ids: Traced, experts: usize) -> Result<Traced, TraceError> {
    let outside = b.binary_scalar(BinOp::GeU, ids, Scalar::I32(experts as i32))?;
    sum_to_scalar(b, b.cast(outside, DType::F32)?)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::Fx;

    /// Ids round-trip through [`WitnessKind::decode`] and are distinct per (layer, kind).
    #[test]
    fn ids_name_their_layer_and_kind() {
        let mut seen = std::collections::BTreeSet::new();
        for layer in [0, 1, 7, 1000] {
            for kind in WitnessKind::ALL {
                let id = kind.id(layer).unwrap();
                assert!(seen.insert(id), "{kind:?} on layer {layer} repeats an id");
                assert_eq!(WitnessKind::decode(id), Some((layer, kind)));
            }
        }
    }

    fn eval(build: impl FnOnce(&Builder, Traced) -> Result<Traced, TraceError>, x: &[f32]) -> f32 {
        let fx = Fx::new(1);
        let b = Builder::for_tests();
        let xs = b.constant("x", TensorType::f32(vec![x.len()])).unwrap();
        let w = build(&b, xs).unwrap();
        let out = fx.eval(&b.finish(w).unwrap(), &[("x", x)], &[], &[]).0;
        assert_eq!(out.len(), 1, "a witness is one lane");
        out[0]
    }

    /// Each witness counts exactly the values it guards: zero on sound input, the number of bad
    /// values otherwise (NaN and both infinities for finiteness; zero, negative, NaN and infinite
    /// sums; ids at or past `experts` and negative ids).
    #[test]
    fn witnesses_count_what_they_guard() {
        let nonfinite = |x: &[f32]| eval(nonfinite_count, x);
        assert_eq!(nonfinite(&[0.0, -3.5, 1e30]), 0.0);
        assert_eq!(
            nonfinite(&[f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY]),
            3.0
        );
        let unusable = |x: &[f32]| eval(unusable_sum_count, x);
        assert_eq!(unusable(&[0.5, 1.0, 4.0]), 0.0);
        assert_eq!(unusable(&[0.0, -1.0, f32::NAN, f32::INFINITY, 2.0]), 4.0);
        // The ids are I32, as `ArgTopK` produces them: a token slot is the oracle harness's I32 input.
        let out_of_range = |ids: &[i32]| {
            let fx = Fx::new(1);
            let b = Builder::for_tests();
            let slot = b
                .slot(Slot::Token, TensorType::new(vec![ids.len()], DType::I32))
                .unwrap();
            let w = out_of_range_count(&b, slot, 4).unwrap();
            let out = fx
                .eval(&b.finish(w).unwrap(), &[], &[(Slot::Token, ids)], &[])
                .0;
            assert_eq!(out.len(), 1, "a witness is one lane");
            out[0]
        };
        assert_eq!(out_of_range(&[0, 3, 2]), 0.0);
        assert_eq!(out_of_range(&[4, -1, 3, 9]), 3.0);
    }
}

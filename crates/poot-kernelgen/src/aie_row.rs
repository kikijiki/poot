//! The numerical contract of a row kernel on an AIE core (POOT-1174).
//!
//! A fused row kernel (a norm: reductions and the pointwise steps around them) is a composition of ops, so
//! its error is not one measured number but a function of the steps. The AIE2p adder is faithful but not
//! round-to-nearest, so an addition that cancels has no relative bound; the contract is therefore derived,
//! step by step, for the step set whose error is bounded relative to the result, and every other kernel is
//! refused with the step that has no bound. The derivation tracks, per value, a bound on its relative error
//! and a proved lower bound (a value known to be at least some non-negative number). Each op adds its own
//! relative error:
//!
//! - `mul` and `div` are bit-exact IEEE (round to nearest even): their sum of operand errors plus one
//!   rounding;
//! - `add` of two values that are provably non-negative cannot cancel, so its error against the result is the
//!   core adder's `2 * eps` plus the operands' larger error;
//! - `sqrt` is the core's software sequence ([`SQRT_REL`]), halving the operand's error, and needs an operand
//!   provably a positive normal number;
//! - a `sum` reduction adds its terms one after another (`cols` additions, the first against the zero
//!   accumulator), each non-cancelling because every term is provably non-negative;
//! - a `max` reduction selects one of its terms, so it keeps the term's error when the terms are exact.
//!
//! Every bound is first order in the operands' errors and keeps the factor of two between the round-to-nearest
//! unit (`eps / 2`) and `eps` as margin. It is promised over finite operands whose every intermediate is a
//! finite normal number. A kernel the hardware does not meet is a red row in `poot-npu/tests/aie_rows.rs`.

use crate::{
    ErrorBound, FusedInput, FusedScalarOp, NumericalContract, RowKernel, RowOp, RowReduce,
    RowValue, request::SQRT_REL,
};
use poot_kernel_ir::{BinOp, MathOp};

/// What the derivation knows of one value: a bound on its relative error and, when it is provably
/// non-negative, a lower bound it never goes below.
#[derive(Clone, Copy, Debug)]
struct Quantity {
    /// Upper bound of `|computed - exact| / |exact|`.
    rel: f32,
    /// A proved lower bound (at least zero), or `None` when the value may be negative.
    at_least: Option<f32>,
}

impl Quantity {
    const EXACT_ANY_SIGN: Self = Self {
        rel: 0.0,
        at_least: None,
    };
}

/// Why a row kernel has no derived contract.
pub(crate) const NO_BOUND: &str = "the row kernel has a step the AIE core has no derived error bound for \
     (only mul, div, non-cancelling add, sqrt of a positive value, recip, and sum or max reductions)";

/// The contract of `kernel` on an AIE core, or the reason it has none.
pub(crate) fn row_contract(kernel: &RowKernel) -> Result<NumericalContract, &'static str> {
    let mut steps: Vec<Quantity> = Vec::with_capacity(kernel.steps.len());
    let operand = |input: &FusedInput, steps: &[Quantity]| -> Option<Quantity> {
        match input {
            FusedInput::Leaf(_) => Some(Quantity::EXACT_ANY_SIGN),
            FusedInput::Step(s) => steps.get(s.index()).copied(),
            FusedInput::Lit(c) => Some(Quantity {
                rel: 0.0,
                at_least: (*c >= 0.0).then_some(*c),
            }),
            FusedInput::LitI32(_) | FusedInput::LeafLane { .. } => None,
        }
    };
    let same = |a: &FusedInput, b: &FusedInput| match (a, b) {
        (FusedInput::Leaf(i), FusedInput::Leaf(j)) => i == j,
        (FusedInput::Step(i), FusedInput::Step(j)) => i == j,
        _ => false,
    };
    for step in &kernel.steps {
        let quantity = match step {
            RowOp::Pointwise { op, inputs } => {
                let q = |i: usize| inputs.get(i).and_then(|input| operand(input, &steps));
                match (op, inputs.len()) {
                    (FusedScalarOp::Binary(BinOp::Mul), 2) => {
                        let (a, b) = (q(0).ok_or(NO_BOUND)?, q(1).ok_or(NO_BOUND)?);
                        Quantity {
                            rel: a.rel + b.rel + f32::EPSILON,
                            at_least: match (a.at_least, b.at_least) {
                                (Some(x), Some(y)) => Some(x * y),
                                _ if same(&inputs[0], &inputs[1]) => Some(0.0),
                                _ => None,
                            },
                        }
                    }
                    (FusedScalarOp::Binary(BinOp::Div), 2) => {
                        let (a, b) = (q(0).ok_or(NO_BOUND)?, q(1).ok_or(NO_BOUND)?);
                        Quantity {
                            rel: a.rel + b.rel + f32::EPSILON,
                            at_least: a.at_least.zip(b.at_least).map(|_| 0.0),
                        }
                    }
                    (FusedScalarOp::Binary(BinOp::Add), 2) => {
                        let (a, b) = (q(0).ok_or(NO_BOUND)?, q(1).ok_or(NO_BOUND)?);
                        let (x, y) = a.at_least.zip(b.at_least).ok_or(NO_BOUND)?;
                        Quantity {
                            rel: a.rel.max(b.rel) + 2.0 * f32::EPSILON,
                            at_least: Some(x + y),
                        }
                    }
                    (FusedScalarOp::Math(MathOp::Sqrt), 1) => {
                        let a = q(0).ok_or(NO_BOUND)?;
                        let floor = a.at_least.filter(|&m| m >= f32::MIN_POSITIVE);
                        let floor = floor.ok_or(NO_BOUND)?;
                        Quantity {
                            rel: a.rel / 2.0 + SQRT_REL,
                            at_least: Some(floor.sqrt()),
                        }
                    }
                    (FusedScalarOp::Recip, 1) => {
                        let a = q(0).ok_or(NO_BOUND)?;
                        Quantity {
                            rel: a.rel + f32::EPSILON,
                            at_least: a.at_least.map(|_| 0.0),
                        }
                    }
                    _ => return Err(NO_BOUND),
                }
            }
            RowOp::Reduce { op, input } => {
                let a = operand(input, &steps).ok_or(NO_BOUND)?;
                match op {
                    RowReduce::Sum => {
                        let floor = a.at_least.ok_or(NO_BOUND)?;
                        Quantity {
                            rel: a.rel + 2.0 * f32::EPSILON * kernel.n_cols as f32,
                            at_least: Some(floor * kernel.n_cols as f32),
                        }
                    }
                    RowReduce::Max => {
                        if a.rel != 0.0 {
                            return Err(NO_BOUND);
                        }
                        a
                    }
                }
            }
        };
        steps.push(quantity);
    }
    let output = match kernel.output {
        RowValue::Step(step) => *steps.get(step.index()).ok_or(NO_BOUND)?,
        RowValue::Leaf(_) => Quantity::EXACT_ANY_SIGN,
    };
    Ok(NumericalContract {
        id: "f32-row-derived-v1",
        bound: ErrorBound::Result {
            abs: 0.0,
            rel: output.rel,
        },
        domain: (f32::MIN, f32::MAX),
    })
}

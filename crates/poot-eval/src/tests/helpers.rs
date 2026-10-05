//! Shared helpers used by the test submodules.
//!
//! `i32_constant`/`finish_with_validations`/`finish_with_state_and_validations` reimplement
//! `poot-graph-ir`'s deleted `Builder` methods of the same name (card 671: dead in production,
//! moved to `poot_test_util::graph_fixtures`) from the same already-public primitives that crate
//! uses. Duplicated here rather than imported: `poot-test-util`'s `graph-fixtures` feature
//! regular-depends on `poot-eval` (for its own graph fixtures), so this crate's own `#[cfg(test)]`
//! tests dev-depending on it back would compile two mismatched instances of `poot-graph-ir`'s
//! `Builder`/`Traced`/`Graph`.

use poot_graph_ir::{
    Builder, BuilderAppendError, Graph, GraphValidationError, Storage, TensorType, Traced,
    ValidationId, ValidationOutput, ValidationOutputs,
};
use poot_tensor::DType;

pub(super) fn i32_constant(
    b: &Builder,
    name: &str,
    shape: Vec<usize>,
) -> Result<Traced, BuilderAppendError> {
    shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or_else(|| BuilderAppendError::ElementCountOverflow {
            name: name.to_string(),
            shape: shape.clone(),
        })?;
    let mut plan = b.append_plan(0);
    let value = plan.input_result(
        name.to_string(),
        TensorType::new(shape, DType::I32),
        Storage::Const,
    )?;
    let mut prepared = b.preflight_append(plan)?;
    let id = b.commit_append(&mut prepared)?;
    debug_assert_eq!(id, value.id);
    Ok(value)
}

pub(super) fn finish_with_validations(
    b: Builder,
    out: Traced,
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph<ValidationOutputs>, GraphValidationError> {
    finish_with_state_and_validations(b, out, &[], validations)
}

pub(super) fn finish_with_state_and_validations(
    b: Builder,
    out: Traced,
    state: &[(Traced, Traced)],
    validations: &[(ValidationId, &str, Traced)],
) -> Result<Graph<ValidationOutputs>, GraphValidationError> {
    let g = b.finish_with_state(out, state).with_validations(
        validations
            .iter()
            .map(|&(id, name, value)| ValidationOutput {
                id,
                name: name.to_owned(),
                value: value.id,
            })
            .collect(),
    );
    g.validate()?;
    Ok(g)
}

/// deterministic pseudo-random fill in [-1, 1), no rng dependency.
pub(super) fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Flat matmul: C[m, n] = A[m, k] @ B[k, n], all row-major.
pub(super) fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Softmax over a slice (numerically stable).
pub(super) fn softmax_ref(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// GELU (tanh approximation), matching `ops::gelu`'s oracle bits:
/// `v / (1 + exp(-2 * sqrt(2/pi) * (v + 0.044715 * v^3)))`.
pub(super) fn gelu_ref(v: f32) -> f32 {
    let z = (2.0_f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v);
    v / (1.0 + (-2.0 * z).exp())
}

/// Deterministic xorshift64 step for the fuzz loops (distinct from `fill`/`rng_stream` so fuzzers can
/// share one mutable seed across nested closures).
pub(super) fn fuzz_u64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

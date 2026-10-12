//! Build a small op graph (RMSNorm) with the `Builder`, watch it decompose into primitives, and evaluate it on
//! the CPU eager reference executor against a hand-computed reference. High-level ops are compositions of
//! orthogonal primitives (the IR is primitive, with one semantic definition shared by tracing, the
//! oracle and every lowering), and one executor runs the primitive graph.
//!
//! Run: `cargo run -p poot-llm --example build_and_eval` (CPU only).

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Builder, TensorType, TraceLimits, ops};
use poot_tensor::HostTensor;

fn main() {
    let n = 8usize;
    let eps = 1e-6f32;

    // A `Builder` charges every value, equation, scalar literal and owned record against the limits its
    // caller states; there is no unlimited builder. These are far above what one RMSNorm needs.
    let limits = TraceLimits {
        max_tokens: NonZeroUsize::MIN, // a model trace's per-step token bound; a bare builder never reads it
        max_values: NonZeroUsize::new(1 << 10).unwrap(),
        max_eqns: NonZeroUsize::new(1 << 10).unwrap(),
        max_literal_bytes: NonZeroU64::new(1 << 12).unwrap(),
        max_owned_bytes: NonZeroU64::new(1 << 20).unwrap(),
    };

    // trace `rmsnorm(x, w)` into the graph. `x`/`w` are graph inputs (bound at eval by their value id).
    // Every builder call is fallible: a shape the operation does not admit, or a request past the
    // limits, is a typed `BuilderError`.
    let b = Builder::new(limits);
    let x = b
        .constant("x", TensorType::f32(vec![1, 1, n]))
        .expect("within the limits");
    let w = b
        .constant("w", TensorType::f32(vec![n]))
        .expect("within the limits");
    let out = ops::rmsnorm(&b, x, w, eps).expect("a well-typed RMSNorm within the limits");
    let g = b.finish(out).expect("a well-formed graph");
    println!(
        "rmsnorm traced to {} primitive eqns (mul, reduce-sum, scalar mul/add, sqrt, div, mul)",
        g.eqns().len()
    );

    // evaluate with a sample input.
    let xv: Vec<f32> = (0..n).map(|i| i as f32 - 3.5).collect();
    let wv = vec![1.0f32; n];
    let mut inputs = HashMap::new();
    inputs.insert(
        x.id(),
        Value::from(HostTensor::f32(vec![1, 1, n], xv.clone())),
    );
    inputs.insert(w.id(), Value::from(HostTensor::f32(vec![n], wv)));
    let y = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("eval")
        .output
        .into_host()
        .expect("dense output");

    // hand-computed reference: x / sqrt(mean(x^2) + eps).
    let ms = xv.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let denom = (ms + eps).sqrt();
    let want: Vec<f32> = xv.iter().map(|v| v / denom).collect();

    println!("input    : {xv:?}");
    println!("rmsnorm  : {:?}", y.as_f32().unwrap());
    println!("reference: {want:?}");
    let max_err = y
        .as_f32()
        .unwrap()
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("max abs error vs reference: {max_err:.2e}");
    assert!(max_err <= 1e-5, "rmsnorm eval diverged from reference");
}

//! Build a small op graph (RMSNorm) with the `Builder`, watch it decompose into primitives, and evaluate it on
//! the CPU eager reference executor against a hand-computed reference. High-level ops are compositions of
//! orthogonal primitives (the IR is primitive, with one semantic definition shared by tracing, the
//! oracle and every lowering), and one executor runs the primitive graph.
//!
//! Run: `cargo run -p poot-llm --example build_and_eval` (CPU only).

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Builder, TensorType, ops};
use poot_tensor::HostTensor;

fn main() {
    let n = 8usize;
    let eps = 1e-6f32;

    // trace `rmsnorm(x, w)` into the graph. `x`/`w` are graph inputs (bound at eval by their value id).
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n]));
    let w = b.constant("w", TensorType::f32(vec![n]));
    let out = ops::rmsnorm(&b, x, w, eps);
    let g = b.finish(out);
    println!(
        "rmsnorm traced to {} primitive eqns (mul, reduce-sum, scalar mul/add, sqrt, div, mul)",
        g.eqns.len()
    );

    // evaluate with a sample input.
    let xv: Vec<f32> = (0..n).map(|i| i as f32 - 3.5).collect();
    let wv = vec![1.0f32; n];
    let mut inputs = HashMap::new();
    inputs.insert(
        x.id,
        Value::from(HostTensor::f32(vec![1, 1, n], xv.clone())),
    );
    inputs.insert(w.id, Value::from(HostTensor::f32(vec![n], wv)));
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

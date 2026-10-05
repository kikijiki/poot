//! Elementwise-chain and matmul-epilogue fusion equivalence fuzzers, moved from
//! `poot-eval/src/tests/fuzz_primitives.rs` with the `fuse` pass itself (card 626) - poot-eval must
//! never depend on poot-graph-plan (its own architecture test); this crate already dev-depends on
//! poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::Builder;
use poot_graph_ir::op::{BinOp, OpKind, UnOp};
use poot_graph_ir::types::TensorType;
use poot_graph_plan::{cse, fuse};
use poot_tensor::HostTensor;
use std::collections::HashMap;

/// Deterministic pseudo-random fill in [-1, 1), no rng dependency (poot-eval's own test helper of the
/// same name, duplicated: this is an external integration test, so it cannot reach poot-eval's
/// `pub(super)` test helpers).
fn fill(n: usize, seed: u64) -> Vec<f32> {
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

/// A tiny deterministic xorshift stream (poot-eval's own test helper, duplicated for the same reason
/// as `fill` above).
fn rng_stream(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed
        .wrapping_mul(0x100000001B3)
        .wrapping_add(0x9E3779B97F4A7C15)
        | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

#[test]
fn fused_equals_unfused_elementwise_chain_all_fusable() {
    let shape = vec![2usize, 8];
    let numel = 16usize;
    let mut total = 0usize;
    for seed in 0u64..300 {
        let mut rng = rng_stream(seed ^ 0xF00D);
        let b = Builder::new();
        // Two leaf constants. Binary steps use x1 as the "other" arg so each acc is consumed once.
        let x0 = b.constant("x0", TensorType::f32(shape.clone()));
        let x1 = b.constant("x1", TensorType::f32(shape.clone()));
        let leaf_ids = [x0.id, x1.id];
        // Build a deep single-use chain: every intermediate consumed exactly once.
        let depth = 3 + (rng() % 6) as usize; // 3..8
        let mut acc = x0;
        for _ in 0..depth {
            if rng().is_multiple_of(2) {
                // unary: all five are in is_fusable.
                let op = match rng() % 5 {
                    0 => UnOp::Neg,
                    1 => UnOp::Exp,
                    2 => UnOp::Log,
                    3 => UnOp::Sqrt,
                    _ => UnOp::Tanh,
                };
                acc = b.unary(op, acc);
            } else {
                // binary: acc is the left operand (single-use); x0/x1 are the right leaf.
                let other = if rng().is_multiple_of(2) { x0 } else { x1 };
                let op = match rng() % 3 {
                    0 => BinOp::Add,
                    1 => BinOp::Sub,
                    _ => BinOp::Mul,
                };
                acc = b.binary(op, acc, other);
            }
        }
        let g = b.finish(acc);
        let n_raw = g.eqns.len();
        // Inputs in [0.1, 1.1] so Log/Sqrt don't produce NaN.
        let safe = |v: f32| v.abs() + 0.1;
        let d0: Vec<f32> = fill(numel, seed * 3 + 1).into_iter().map(safe).collect();
        let d1: Vec<f32> = fill(numel, seed * 3 + 2).into_iter().map(safe).collect();
        let mut inputs = HashMap::new();
        inputs.insert(leaf_ids[0], Value::from(HostTensor::f32(shape.clone(), d0)));
        inputs.insert(leaf_ids[1], Value::from(HostTensor::f32(shape.clone(), d1)));

        let fg = fuse(&cse(&g));
        fg.validate().expect("fused graph valid");
        let n_fused = fg.eqns.len();

        // Assertion (a): fusion must fire.
        assert!(
            n_fused < n_raw,
            "seed {seed}: depth {depth} chain did not shrink eqn count ({n_raw} -> {n_fused})"
        );
        let has_fused_region = fg.eqns.iter().any(|e| matches!(e.op, OpKind::Fused(_)));
        assert!(
            has_fused_region,
            "seed {seed}: no Fused region in fused graph (eqns: {n_fused})"
        );

        // Assertion (b): bit-identical output.
        let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .expect("eval unfused");
        let fused_out = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .expect("eval fused");
        assert_eq!(
            unfused.shape(),
            fused_out.shape(),
            "seed {seed}: shape mismatch"
        );
        let ub: Vec<u32> = unfused
            .as_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let fb: Vec<u32> = fused_out
            .as_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(
            ub, fb,
            "seed {seed}: fused not bit-identical to unfused ({n_raw} -> {n_fused} eqns)"
        );
        total += 1;
    }
    eprintln!("elementwise chain fusion: {total}/300 graphs, fusion fired on all");
}

#[test]
fn fused_equals_unfused_matmul_epilogue() {
    // (M, K, N) triples: small so the CPU matmul oracle is fast.
    let configs: &[(usize, usize, usize)] =
        &[(4, 8, 16), (1, 4, 8), (3, 6, 5), (2, 3, 7), (8, 4, 4)];
    let mut fusion_fired = 0usize;
    let mut total = 0usize;

    for seed in 0u64..200 {
        let mut rng = rng_stream(seed ^ 0xBEEF);
        let (m, k, n) = configs[(seed as usize) % configs.len()];
        let out_shape = vec![m, n];
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        // matmul: the fusion barrier.
        let mm = b.matmul(a, w); // [m, n]
        // Epilogue: chain 2..5 fusable ops so there are always >= 2 members to fuse.
        let epilogue_len = 2 + (rng() % 4) as usize; // 2..5
        let bias = b.constant("bias", TensorType::f32(out_shape.clone()));
        let mut acc = b.binary(BinOp::Add, mm, bias); // bias-add (always first)
        for _ in 1..epilogue_len {
            if rng().is_multiple_of(2) {
                let op = match rng() % 3 {
                    0 => UnOp::Neg,
                    1 => UnOp::Tanh,
                    _ => UnOp::Exp,
                };
                acc = b.unary(op, acc);
            } else {
                // second leaf so the intermediate stays single-use.
                let other = bias;
                let op = match rng() % 3 {
                    0 => BinOp::Add,
                    1 => BinOp::Mul,
                    _ => BinOp::Sub,
                };
                acc = b.binary(op, acc, other);
            }
        }
        let g = b.finish(acc);
        let n_raw = g.eqns.len(); // matmul + epilogue_len eqns

        // Inputs.
        let numel_a = m * k;
        let numel_w = k * n;
        let numel_b = m * n;
        let mut inputs = HashMap::new();
        inputs.insert(
            a.id,
            Value::from(HostTensor::f32(vec![m, k], fill(numel_a, seed * 5 + 1))),
        );
        inputs.insert(
            w.id,
            Value::from(HostTensor::f32(vec![k, n], fill(numel_w, seed * 5 + 2))),
        );
        inputs.insert(
            bias.id,
            Value::from(HostTensor::f32(
                out_shape.clone(),
                fill(numel_b, seed * 5 + 3),
            )),
        );

        let fg = fuse(&cse(&g));
        fg.validate().expect("fused graph valid");
        let n_fused = fg.eqns.len();

        // Assertion (a): epilogue must fuse (at least 2 epilogue eqns -> 1 Fused).
        assert!(
            n_fused < n_raw,
            "seed {seed}: matmul+epilogue({epilogue_len}) did not reduce eqn count \
             ({n_raw} -> {n_fused})"
        );
        let has_fused_region = fg.eqns.iter().any(|e| matches!(e.op, OpKind::Fused(_)));
        assert!(
            has_fused_region,
            "seed {seed}: no Fused region after matmul epilogue (eqns: {n_fused})"
        );
        // The matmul itself must not have been absorbed.
        let has_matmul = fg.eqns.iter().any(|e| matches!(e.op, OpKind::MatMul));
        assert!(
            has_matmul,
            "seed {seed}: matmul disappeared - fusion crossed the matmul boundary"
        );
        fusion_fired += 1;

        // Assertion (b): bit-identical output (matmul is exact f32; epilogue is exact elementwise).
        let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .expect("eval unfused");
        let fused_out = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("fuzz_primitives tests evaluate dense graphs")
            })
            .expect("eval fused");
        assert_eq!(
            unfused.shape(),
            fused_out.shape(),
            "seed {seed}: shape mismatch"
        );
        let ub: Vec<u32> = unfused
            .as_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let fb: Vec<u32> = fused_out
            .as_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(
            ub, fb,
            "seed {seed}: matmul epilogue fused not bit-identical ({n_raw} -> {n_fused} eqns)"
        );
        total += 1;
    }
    eprintln!(
        "matmul epilogue fusion: {fusion_fired}/{total} fired fusion (all expected), configs {:?}",
        configs
    );
    assert_eq!(
        fusion_fired, total,
        "epilogue fusion must fire on every seed"
    );
}

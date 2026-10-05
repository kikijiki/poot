//! Pointwise/reduction fusion equivalence and rope-fusion equivalence, moved from
//! `poot-eval/src/tests/cast_quant_norm_rope.rs` with the passes themselves (card 626) - poot-eval
//! must never depend on poot-graph-plan (its own architecture test); this crate already dev-depends
//! on poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::op::{BinOp, UnOp};
use poot_graph_ir::ops::rope;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Builder, Slot, ValueId};
use poot_graph_plan::{cse, fuse};
use poot_tensor::DType;
use poot_tensor::HostTensor;
use poot_test_util::assert_close_rel;
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

use poot_graph_plan::passes_without_target as optimize;

#[test]
fn fused_pointwise_chain_is_bit_identical_to_unfused() {
    // out = exp(a - b) * c + a (4 pointwise ops, all [1,1,8]; a appears twice as a leaf). The fused
    // graph collapses the chain to one Fused eqn; eval must be bit-identical to the un-fused chain
    // (graph-architecture.md section 5 stage 2: same ops, same order).
    let b = Builder::new();
    let n = 8usize;
    let (a, bb, c) = (
        b.constant("a", TensorType::f32(vec![1, 1, n])),
        b.constant("b", TensorType::f32(vec![1, 1, n])),
        b.constant("c", TensorType::f32(vec![1, 1, n])),
    );
    let t1 = b.binary(BinOp::Sub, a, bb);
    let t2 = b.unary(UnOp::Exp, t1);
    let t3 = b.binary(BinOp::Mul, t2, c);
    let out = b.binary(BinOp::Add, t3, a);
    let (ai, bi, ci) = (a.id, bb.id, c.id);
    let g = b.finish(out);

    let fg = fuse(&g);
    fg.validate().expect("fused graph valid");
    assert_eq!(fg.eqns.len(), 1, "the 4-op chain fuses to one eqn");

    let mut inputs = HashMap::new();
    inputs.insert(ai, Value::from(HostTensor::f32(vec![1, 1, n], fill(n, 11))));
    inputs.insert(bi, Value::from(HostTensor::f32(vec![1, 1, n], fill(n, 22))));
    inputs.insert(ci, Value::from(HostTensor::f32(vec![1, 1, n], fill(n, 33))));

    let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    let fused = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_eq!(unfused.shape(), fused.shape());
    assert_eq!(
        unfused.as_f32().unwrap(),
        fused.as_f32().unwrap(),
        "fused eval must be bit-identical to un-fused"
    );
}

#[test]
fn fused_equals_unfused_over_random_pointwise_graphs() {
    // Randomized fusion equivalence: over procedurally generated pointwise graphs (unary + same-shape
    // binary, with shared subexpressions and multi-use values), eval(fuse(cse(g))) must be
    // bit-identical to eval(g) ("same ops, same order", graph-architecture.md section 5 stage 2).
    // Compared via `to_bits` so NaN/inf also match.
    let shape = vec![1usize, 4, 8];
    let numel = 32usize;
    let mut fused_count = 0usize;
    for seed in 0u64..400 {
        let mut rng = rng_stream(seed);
        let b = Builder::new();
        let inputs_t: Vec<_> = (0..3)
            .map(|i| b.constant(&format!("x{i}"), TensorType::f32(shape.clone())))
            .collect();
        let leaf_ids: Vec<ValueId> = inputs_t.iter().map(|t| t.id).collect();
        // `acc` is the running output (a deep dependent chain, so fusable); `pool` holds reused
        // intermediates (shared subexpressions, multi-use).
        let mut pool = inputs_t.clone();
        let mut acc = inputs_t[(rng() as usize) % 3];
        let steps = 5 + (rng() % 12) as usize;
        for _ in 0..steps {
            match rng() % 3 {
                0 => {
                    let op = [UnOp::Neg, UnOp::Tanh, UnOp::Erf, UnOp::Round][(rng() as usize) % 4];
                    acc = b.unary(op, acc);
                }
                1 => {
                    let other = pool[(rng() as usize) % pool.len()];
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Max][(rng() as usize) % 4];
                    acc = b.binary(op, acc, other);
                }
                _ => {
                    // grow the pool with a reused intermediate (creates multi-use values for cse/fusion).
                    let a = pool[(rng() as usize) % pool.len()];
                    let c = pool[(rng() as usize) % pool.len()];
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Max][(rng() as usize) % 4];
                    pool.push(b.binary(op, a, c));
                }
            }
        }
        let g = b.finish(acc);
        let mut inputs = HashMap::new();
        for (i, id) in leaf_ids.iter().enumerate() {
            inputs.insert(
                *id,
                Value::from(HostTensor::f32(
                    shape.clone(),
                    fill(numel, seed * 9 + i as u64 + 1),
                )),
            );
        }
        let fg = fuse(&cse(&g));
        fg.validate().expect("fused graph valid");
        if fg.eqns.len() < g.eqns.len() {
            fused_count += 1;
        }
        let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        let fused = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        assert_eq!(
            unfused.shape(),
            fused.shape(),
            "seed {seed}: shape mismatch"
        );
        let ub: Vec<u32> = unfused
            .as_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        let fb: Vec<u32> = fused
            .as_f32()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(
            ub,
            fb,
            "seed {seed}: fused eval not bit-identical to unfused ({} -> {} eqns)",
            g.eqns.len(),
            fg.eqns.len()
        );
    }
    eprintln!("pointwise fuzz: 400 graphs, {fused_count} exercised fusion");
    assert!(
        fused_count >= 320,
        "most random pointwise graphs should fuse (got {fused_count}/400)"
    );
}

#[test]
fn fused_equals_unfused_over_random_reduction_graphs() {
    // FusedRow analogue of the pointwise fuzzer: graphs that interleave reduce-along-the-last-axis (+
    // broadcast back) with pointwise ops (the rmsnorm/softmax shape). Fused and un-fused agree within a
    // tight f32 tolerance (summation order differs, so close, not bit-exact).
    use poot_graph_ir::op::RedOp;

    // Shape varies per seed (row counts + partial widths 7/33) and one leaf is a row-broadcast input
    // ([..,1]), exercising eval's FusedRow handler over broadcast strides and non-`[1,4,8]` widths.
    let shapes: [Vec<usize>; 5] = [
        vec![1, 4, 8],
        vec![2, 3, 7],
        vec![1, 8, 16],
        vec![1, 6, 33],
        vec![3, 5, 5],
    ];
    let mut had_reduction = 0usize;
    for seed in 0u64..300 {
        let mut rng = rng_stream(seed ^ 0xABCD);
        let shape = shapes[(seed as usize) % shapes.len()].clone();
        let axis = shape.len() - 1;
        let numel: usize = shape.iter().product();
        let mut bshape = shape.clone();
        bshape[axis] = 1;
        let bnumel: usize = bshape.iter().product();
        let b = Builder::new();
        let full: Vec<_> = (0..3)
            .map(|i| b.constant(&format!("x{i}"), TensorType::f32(shape.clone())))
            .collect();
        let bcast = b.constant("xb", TensorType::f32(bshape.clone()));
        let mut acc = full[(rng() as usize) % 3];
        let mut reduced = false;
        let steps = 5 + (rng() % 10) as usize;
        for _ in 0..steps {
            match rng() % 5 {
                0 => {
                    let op = [UnOp::Neg, UnOp::Tanh, UnOp::Erf][(rng() as usize) % 3];
                    acc = b.unary(op, acc);
                }
                1 | 2 => {
                    let other = full[(rng() as usize) % 3];
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][(rng() as usize) % 3];
                    acc = b.binary(op, acc, other);
                }
                3 => {
                    // binary against the row-broadcast leaf (broadcasts to the full row).
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][(rng() as usize) % 3];
                    acc = b.binary(op, acc, bcast);
                }
                _ => {
                    // reduction region: reduce last axis (keepdim) then broadcast back to the row shape.
                    let op = if rng().is_multiple_of(2) {
                        RedOp::Sum
                    } else {
                        RedOp::Max
                    };
                    let r = b.reduce(op, acc, axis, true);
                    acc = b.broadcast(r, shape.clone());
                    reduced = true;
                }
            }
        }
        if reduced {
            had_reduction += 1;
        }
        let g = b.finish(acc);
        let mut inputs = HashMap::new();
        for (i, t) in full.iter().enumerate() {
            inputs.insert(
                t.id,
                Value::from(HostTensor::f32(
                    shape.clone(),
                    fill(numel, seed * 13 + i as u64 + 1),
                )),
            );
        }
        inputs.insert(
            bcast.id,
            Value::from(HostTensor::f32(
                bshape.clone(),
                fill(bnumel, seed * 13 + 101),
            )),
        );
        let fg = fuse(&cse(&g));
        fg.validate().expect("fused graph valid");
        let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        let fused = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        assert_eq!(
            unfused.shape(),
            fused.shape(),
            "seed {seed}: shape mismatch"
        );
        for (i, (x, y)) in unfused
            .as_f32()
            .unwrap()
            .iter()
            .zip(fused.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                (x - y).abs() <= 1e-4 * (1.0 + x.abs()),
                "seed {seed} elem {i}: fused {y} vs unfused {x}"
            );
        }
    }
    eprintln!("reduction fuzz: 300 graphs, {had_reduction} contained a reduction region");
    assert!(
        had_reduction >= 200,
        "most graphs should contain a reduction (got {had_reduction}/300)"
    );
}

/// A multi-head rope graph and its inputs, shared by the fusion-equivalence tests below (cos/sin
/// broadcast against the head axis).
fn rope_graph_and_inputs(d: usize, rot: usize) -> (poot_graph_ir::Graph, HashMap<ValueId, Value>) {
    let (hq, max_pos) = (3usize, 16usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, hq, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, rot]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, rot]));
    let pos = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
    let out = rope(&b, x, cos, sin, pos);
    let (xi, ci, si, pi) = (x.id, cos.id, sin.id, pos.id);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![1, hq, 1, d], fill(hq * d, 5))),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, rot], fill(max_pos * rot, 6))),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, rot], fill(max_pos * rot, 7))),
    );
    inputs.insert(pi, Value::from(HostTensor::i32(vec![], vec![3])));
    (g, inputs)
}

#[test]
fn rope_fusion_eval_matches_unfused_full() {
    // full rotary: eval(rope_fusion(g)) == eval(g) approximately; checks match_rope fired without
    // changing numerics.
    let (g, inputs) = rope_graph_and_inputs(8, 8);
    let fused = poot_graph_plan::rope_fusion(&cse(&g));
    assert!(
        fused
            .eqns
            .iter()
            .any(|e| matches!(e.op, poot_graph_ir::op::OpKind::Rope { .. })),
        "rope_fusion should fire on the full-rotary chain"
    );
    let got_unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    let got_fused = eval(&fused, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_close_rel(
        got_fused.as_f32().unwrap(),
        got_unfused.as_f32().unwrap(),
        1e-6,
    );
}

#[test]
fn rope_fusion_eval_matches_unfused_partial() {
    // partial rotary (rot=4 of D=8): the fused Rope op passes [rot, D) through; eval stays bit-approximate.
    let (g, inputs) = rope_graph_and_inputs(8, 4);
    let fused = poot_graph_plan::rope_fusion(&cse(&g));
    let rope_eqn = fused
        .eqns
        .iter()
        .find(|e| matches!(e.op, poot_graph_ir::op::OpKind::Rope { .. }))
        .expect("rope_fusion should fire on the partial chain");
    assert!(matches!(
        rope_eqn.op,
        poot_graph_ir::op::OpKind::Rope { rot: 4 }
    ));
    let got_unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    let got_fused = eval(&fused, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("cast_quant_norm_rope tests evaluate dense graphs");
    assert_close_rel(
        got_fused.as_f32().unwrap(),
        got_unfused.as_f32().unwrap(),
        1e-6,
    );
}

#[test]
fn optimize_preserves_rope_numerics() {
    // the full pass pipeline must be eval-equivalent to the raw graph.
    for (d, rot) in [(8usize, 8usize), (8, 4)] {
        let (g, inputs) = rope_graph_and_inputs(d, rot);
        let opt = optimize(&g);
        let got_unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        let got_opt = eval(&opt, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("cast_quant_norm_rope tests evaluate dense graphs");
        assert_close_rel(
            got_opt.as_f32().unwrap(),
            got_unfused.as_f32().unwrap(),
            1e-6,
        );
    }
}

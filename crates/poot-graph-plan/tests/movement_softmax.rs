//! Reshape collapse, movement-graph fusion fuzz, the tiled-GEMM contraction choice.
//!
//! Card 626: moved here from `poot-eval/src/tests/movement_softmax.rs` with the passes themselves -
//! poot-eval must never depend on poot-graph-plan (its own architecture test); this crate already
//! dev-depends on poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, UnOp};
use poot_graph_ir::types::TensorType;
use poot_graph_plan::{collapse_reshape_chains, cse, fuse};
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

#[test]
fn reshape_collapse_is_value_identical() {
    // collapse_reshape_chains (and so fuse, which runs it) only reinterprets buffers, so the executor output must be
    // bit-identical to the un-collapsed graph. Covers a bare reshape chain and the decode-attention reshape -> (no-op)
    // transpose -> reshape run that elision exposes.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![2, 3, 4]));
    // a reshape chain feeding an elementwise op (so the result depends on layout, not just shape).
    let r = b.reshape(b.reshape(b.reshape(x, vec![6, 4]), vec![24]), vec![4, 6]);
    let scaled = b.binary(BinOp::Mul, r, r); // [4,6], pointwise over the reshaped layout
    // a separate decode-attention-style run: reshape -> no-op transpose -> reshape.
    let y = b.constant("y", TensorType::f32(vec![1, 1, 24]));
    let yt = b.transpose(b.reshape(y, vec![1, 1, 4, 6]), vec![0, 2, 1, 3]); // no-op at seq 1
    let yr = b.reshape(yt, vec![4, 6]);
    let out = b.binary(BinOp::Add, scaled, yr);
    let g = b.finish(out);

    let inputs = {
        let mut m = HashMap::new();
        m.insert(
            x.id,
            Value::from(HostTensor::f32(vec![2, 3, 4], fill(24, 1))),
        );
        m.insert(
            y.id,
            Value::from(HostTensor::f32(vec![1, 1, 24], fill(24, 2))),
        );
        m
    };
    let opts = || EvalOptions::new(EvalBudget::UNBOUNDED);
    let base = eval(&g, &inputs, opts())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let collapsed = eval(&collapse_reshape_chains(&g), &inputs, opts())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let fused = eval(&fuse(&cse(&g)), &inputs, opts())
        .unwrap()
        .output
        .into_host()
        .unwrap();

    assert_eq!(base.shape(), vec![4, 6]);
    let bits = |t: &HostTensor| {
        t.as_f32()
            .unwrap()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&collapsed), bits(&base), "collapse changed values");
    assert_eq!(bits(&fused), bits(&base), "fuse(collapse) changed values");
    // the collapse actually shrank the graph (the chain's two intermediate reshapes are gone).
    assert!(
        collapse_reshape_chains(&g).eqns.len() < g.eqns.len(),
        "collapse should remove dead intermediate reshapes"
    );
}

#[test]
fn fused_equals_unfused_over_random_movement_graphs() {
    // Property fuzzer over movement + pointwise graphs (the other fuzzers are pointwise/reduction only and never emit
    // reshape/transpose/broadcast/slice/concat). Each random graph is evaluated raw and after each value-preserving
    // transform (cse, elide_noop_transposes, collapse_reshape_chains, fuse(cse)); outputs must be bit-identical.
    // Deterministic, no rng dependency.
    use poot_graph_ir::builder::Traced;
    use poot_graph_plan::{collapse_reshape_chains, cse, elide_noop_transposes};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }
    fn numel(s: &[usize]) -> usize {
        s.iter().product()
    }
    // a random small starting shape (rank 1..3, dims 1..4), often with a size-1 axis (enables broadcast).
    fn rand_shape(rng: &mut Rng) -> Vec<usize> {
        let rank = 1 + rng.below(3) as usize;
        (0..rank).map(|_| 1 + rng.below(3) as usize).collect()
    }
    // a shape with the same element count as `n`, split into 1..3 dims (always valid for reshape).
    fn factor(rng: &mut Rng, n: usize) -> Vec<usize> {
        let rank = 1 + rng.below(3);
        let mut dims = Vec::new();
        let mut rem = n;
        for _ in 0..rank - 1 {
            let divs: Vec<usize> = (1..=rem).filter(|d| rem.is_multiple_of(*d)).collect();
            let d = divs[rng.below(divs.len() as u64) as usize];
            dims.push(d);
            rem /= d;
        }
        dims.push(rem);
        dims
    }
    fn rand_perm(rng: &mut Rng, r: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..r).collect();
        for i in (1..r).rev() {
            let j = rng.below((i + 1) as u64) as usize;
            p.swap(i, j);
        }
        p
    }

    let mut fuzzed = 0usize;
    let mut had_eqns = 0usize;
    for seed in 0..200u64 {
        let mut rng = Rng(seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x1234_5678)
            | 1);
        let b = Builder::new();
        let mut pool: Vec<(Traced, Vec<usize>)> = Vec::new();
        let mut inputs = HashMap::new();
        for k in 0..2 + rng.below(2) as usize {
            let shape = rand_shape(&mut rng);
            let t = b.constant(&format!("x{k}"), TensorType::f32(shape.clone()));
            inputs.insert(
                t.id,
                Value::from(HostTensor::f32(
                    shape.clone(),
                    fill(numel(&shape), seed * 7 + k as u64),
                )),
            );
            pool.push((t, shape));
        }
        for _ in 0..5 + rng.below(8) as usize {
            let (v, vs) = pool[rng.below(pool.len() as u64) as usize].clone();
            let (nv, ns): (Traced, Vec<usize>) = match rng.below(7) {
                0 => (b.unary(UnOp::Neg, v), vs.clone()),
                1 => {
                    // pointwise binary with a same-shape pool partner (else with itself) - fusion fodder.
                    let partner = pool
                        .iter()
                        .find(|(_, s)| *s == vs)
                        .map(|(t, _)| *t)
                        .unwrap_or(v);
                    let op = [BinOp::Add, BinOp::Sub, BinOp::Mul][rng.below(3) as usize];
                    (b.binary(op, v, partner), vs.clone())
                }
                2 => {
                    let ns = factor(&mut rng, numel(&vs));
                    (b.reshape(v, ns.clone()), ns)
                }
                3 if vs.len() >= 2 => {
                    let perm = rand_perm(&mut rng, vs.len());
                    let ns = perm.iter().map(|&p| vs[p]).collect::<Vec<_>>();
                    (b.transpose(v, perm), ns)
                }
                4 => {
                    // broadcast a size-1 axis if there is one, else a no-op-ish neg.
                    if let Some(ax) = vs.iter().position(|&d| d == 1) {
                        let mut ns = vs.clone();
                        ns[ax] = 2 + rng.below(2) as usize;
                        (b.broadcast(v, ns.clone()), ns)
                    } else {
                        (b.unary(UnOp::Neg, v), vs.clone())
                    }
                }
                5 => {
                    let ax = rng.below(vs.len() as u64) as usize;
                    let len = vs[ax];
                    let start = rng.below(len as u64) as usize;
                    let end = start + 1 + rng.below((len - start) as u64) as usize;
                    let mut ns = vs.clone();
                    ns[ax] = end - start;
                    (b.slice(v, ax, start, end), ns)
                }
                _ => {
                    // concat the value with itself along an axis (doubles that axis).
                    let ax = rng.below(vs.len() as u64) as usize;
                    let mut ns = vs.clone();
                    ns[ax] = vs[ax] * 2;
                    (b.concat(ax, &[v, v]), ns)
                }
            };
            if numel(&ns) <= 4096 {
                pool.push((nv, ns));
            }
        }
        let (out, _) = *pool.last().unwrap();
        let g = b.finish(out);
        g.validate()
            .unwrap_or_else(|e| panic!("seed {seed}: invalid graph: {e}"));
        if !g.eqns.is_empty() {
            had_eqns += 1;
        }
        let base = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        let bits = |t: &HostTensor| {
            t.as_f32()
                .unwrap()
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>()
        };
        let want = bits(&base);
        for (name, gg) in [
            ("cse", cse(&g)),
            ("elide_noop_transposes", elide_noop_transposes(&g)),
            ("collapse_reshape_chains", collapse_reshape_chains(&g)),
            ("fuse(cse)", fuse(&cse(&g))),
        ] {
            let got = eval(&gg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap();
            assert_eq!(
                bits(&got),
                want,
                "seed {seed}: {name} changed values (shape {:?})",
                base.shape()
            );
        }
        fuzzed += 1;
    }
    assert_eq!(fuzzed, 200);
    assert!(
        had_eqns > 180,
        "most graphs should contain ops ({had_eqns}/200)"
    );
}

#[test]
fn fused_multi_axis_broadcast_leaf_matches_unfused() {
    // Fusion coverage gap 7: the other fuzzers broadcast one axis at a time. This exercises a leaf broadcast on multiple
    // axes at once ([1,N,1] -> [M,N,K]) feeding a pointwise chain that `fuse` merges into one kernel; the fused kernel's
    // `broadcast_eff_strides` must zero both axes' strides. Pure CPU; broadcast+add is exact, so fused must be
    // bit-identical to unfused.
    use poot_graph_plan::cse;

    for &(m, n, k) in &[(3usize, 4usize, 5usize), (2, 6, 3), (5, 2, 7)] {
        let b = Builder::new();
        // leaf broadcasts from [1,N,1] to [M,N,K]: axes 0 and 2 both expand.
        let leaf = b.constant("leaf", TensorType::f32(vec![1, n, 1]));
        let bc = b.broadcast(leaf, vec![m, n, k]);
        let other = b.constant("other", TensorType::f32(vec![m, n, k]));
        // a short pointwise chain on top so `fuse` has something to merge the broadcast read into.
        let sum = b.binary(BinOp::Add, bc, other);
        let out = b.unary(UnOp::Neg, sum);
        let out = b.binary(BinOp::Mul, out, other);
        let g = b.finish(out);

        let mut inputs = HashMap::new();
        inputs.insert(
            leaf.id,
            Value::from(HostTensor::f32(
                vec![1, n, 1],
                fill(n, m as u64 * 97 + k as u64),
            )),
        );
        inputs.insert(
            other.id,
            Value::from(HostTensor::f32(
                vec![m, n, k],
                fill(m * n * k, m as u64 * 13 + n as u64 * 7 + k as u64),
            )),
        );

        let fg = fuse(&cse(&g));
        fg.validate().expect("fused graph valid");
        assert!(
            fg.eqns.len() < g.eqns.len(),
            "m={m} n={n} k={k}: expected fusion to merge the chain ({} -> {} eqns)",
            g.eqns.len(),
            fg.eqns.len()
        );

        let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("eval unfused")
            .output
            .into_host()
            .unwrap();
        let fused = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("eval fused")
            .output
            .into_host()
            .unwrap();
        assert_eq!(unfused.shape(), vec![m, n, k]);
        assert_eq!(fused.shape(), unfused.shape());
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
            ub, fb,
            "m={m} n={n} k={k}: multi-axis-broadcast fused eval not bit-identical to unfused"
        );
    }
}

#[test]
fn fused_pointwise_over_empty_tensor_edges() {
    // Fusion coverage gap 7 (empty-tensor edge): a graph with a 0-size dim through a movement + pointwise chain, fused.
    // The CPU oracle's movement/pointwise eval loops all bound on `out_shape.product()`, so a 0-size dim iterates zero
    // times and returns a correctly-shaped empty Tensor. Confirms `fuse`/`cse` also accept a 0-numel value.
    use poot_graph_plan::cse;

    for shape in [vec![2usize, 0, 3], vec![0usize, 4], vec![3usize, 0]] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(shape.clone()));
        let y = b.constant("y", TensorType::f32(shape.clone()));
        let sum = b.binary(BinOp::Add, x, y);
        let neg = b.unary(UnOp::Neg, sum);
        // a movement op (reshape to the same numel, still 0) on top, so the fused region carries a zero-size value through a
        // shape change.
        let reshaped = b.reshape(neg, shape.clone());
        let g = b.finish(reshaped);

        let mut inputs = HashMap::new();
        inputs.insert(x.id, Value::from(HostTensor::f32(shape.clone(), vec![])));
        inputs.insert(y.id, Value::from(HostTensor::f32(shape.clone(), vec![])));

        let unfused = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap_or_else(|e| {
                panic!("shape {shape:?}: eval (unfused) failed on an empty tensor: {e}")
            })
            .output
            .into_host()
            .unwrap();
        assert_eq!(
            unfused.shape(),
            shape,
            "shape {shape:?}: empty tensor shape preserved"
        );
        assert!(
            unfused.as_f32().unwrap().is_empty(),
            "shape {shape:?}: 0-size dim must yield no data"
        );

        let fg = fuse(&cse(&g));
        fg.validate()
            .unwrap_or_else(|e| panic!("shape {shape:?}: fused graph invalid: {e}"));
        let fused = eval(&fg, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap_or_else(|e| {
                panic!("shape {shape:?}: eval (fused) failed on an empty tensor: {e}")
            })
            .output
            .into_host()
            .unwrap();
        assert_eq!(fused.shape(), unfused.shape(), "shape {shape:?}");
        assert_eq!(
            fused.as_f32().unwrap(),
            unfused.as_f32().unwrap(),
            "shape {shape:?}: fused empty-tensor eval must match unfused (both empty)"
        );
    }
}

#[test]
fn compile_chooses_the_generated_tiled_gemm_for_m_over_one_only() {
    // card 099b (spec 060), Card 557: the planner's contraction choice takes the synthesized
    // element-by-element tiled GEMM (`kg::tiled_region`) for a bare M>1 / rank-2-[K,N]-weight F32
    // matmul, and records it as the plan's kernel choice (the graph keeps a plain `MatMul`). A decode
    // GEMV (M==1) does not take it.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{
        CompileOptions, FusionPolicy, KernelChoice, Submission, Target, compile,
    };
    use poot_kernelgen::{ContractionSpec, KernelRequest};
    use poot_target::{AmdArch, Backend};

    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let target = Target {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    };
    let choice_of = |m: usize| {
        let (k, n) = (16usize, 8usize);
        let b = Builder::new();
        let a = b.constant("a", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let y = b.matmul(a, w);
        let program = compile(&b.finish(y), &target, &options).expect("the matmul compiles");
        let (eqn, _) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::MatMul))
            .expect("the graph keeps a plain MatMul");
        program.kernel_choice(eqn).clone()
    };
    let tiled = |choice: &KernelChoice| {
        matches!(
            choice,
            KernelChoice::Generated(KernelRequest::Contraction(
                ContractionSpec::TiledRegion { .. }
            ))
        )
    };
    let prefill = choice_of(6);
    assert!(
        tiled(&prefill),
        "the M>1 matmul should take the generated tiled GEMM, got {prefill:?}"
    );
    let decode = choice_of(1);
    assert!(
        !tiled(&decode),
        "M==1 decode GEMV must not take the tiled GEMM, got {decode:?}"
    );
}

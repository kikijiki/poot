//! Contraction epilogue rows on wgpu: a bias, a pointwise activation and a residual add fuse into the
//! contraction that feeds them (`OpKind::ContractionEpilogue`), and the fused device program meets the CPU
//! oracle of the unfused graph.

use super::*;

use poot_executor::StepInputs;
use poot_graph_ir::analysis::{Tier2Class, dispatch_count};
use poot_graph_ir::op::{BinOp, ContractionHead, UnOp};
use poot_graph_ir::types::Scalar;
use poot_graph_plan::{CompileOptions, compile};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_test_util::assert_close_scaled;

/// `tanh(head + bias) * 0.5 + residual`: a bias, a representative pointwise activation, a literal scale and a
/// residual add, each value read once, so the whole chain is one epilogue.
fn epilogue_chain(
    b: &Builder,
    head: poot_graph_ir::Traced,
    bias: poot_graph_ir::Traced,
    residual: poot_graph_ir::Traced,
) -> poot_graph_ir::Traced {
    let biased = b.binary(BinOp::Add, head, bias).unwrap();
    let activated = b.unary(UnOp::Tanh, biased).unwrap();
    let scaled = b
        .binary_scalar(BinOp::Mul, activated, Scalar::F32(0.5))
        .unwrap();
    b.binary(BinOp::Add, scaled, residual).unwrap()
}

/// A deterministic fill in `[-0.5, 0.5)` with no zero element, so a dropped input changes every output.
fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.9 - 0.45 + 0.05
        })
        .collect()
}

/// Round an f32 to its BF16 bits, nearest even.
fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// `g` run once on `gpu`, every input a named const, with the numerics class the compiled program publishes.
fn run_under(
    gpu: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
) -> (HostTensor, Tier2Class) {
    let mut builder = WeightStore::builder();
    for (&id, tensor) in inputs {
        let meta = g.meta(id);
        assert!(matches!(meta.storage, Storage::Const), "{id:?} is a const");
        let dense = DenseWeight::try_new(
            tensor.dtype(),
            tensor.shape().to_vec(),
            StoredBytes::from(tensor.view().bytes().to_vec()),
        )
        .unwrap();
        builder
            .insert(meta.name.clone().unwrap(), WeightEntry::Dense(dense))
            .unwrap();
    }
    let store = builder.build();
    let (bound, map) = ConstWeights::bind(g, &store);
    let exe = gpu.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let staged = staged_replay(&bound, target);
    let entry = gpu.add_entry(exe, &staged).unwrap();
    let bytes = gpu
        .step(exe, entry, &StepInputs::new(), &mut poot_executor::NoSync)
        .unwrap()
        .read()
        .unwrap();
    gpu.remove_entry(exe, entry).unwrap();
    gpu.unload(exe).unwrap();
    let out = g.aval(g.result());
    assert_eq!(out.dtype, DType::F32, "the wgpu rows read an f32 output");
    let out = HostTensor::f32(
        out.shape.clone(),
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    );
    (out, staged.numerics_class())
}

fn compiled(g: &poot_graph_ir::Graph, target: &Target) -> poot_graph_plan::Program {
    compile(
        g,
        target,
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("the graph compiles")
}

/// The head of the one `ContractionEpilogue` in the compiled program; the chain, its bias and residual leaves
/// and the head are that one dispatch.
fn epilogue_head(g: &poot_graph_ir::Graph, target: &Target) -> ContractionHead {
    let fused = compiled(g, target);
    let heads: Vec<ContractionHead> = fused
        .graph()
        .eqns()
        .iter()
        .filter_map(|eqn| match &eqn.op {
            OpKind::ContractionEpilogue { head, .. } => Some(*head),
            _ => None,
        })
        .collect();
    assert_eq!(
        heads.len(),
        1,
        "one fused contraction: {:?}",
        fused.graph().eqns()
    );
    assert_eq!(
        dispatch_count(fused.graph()),
        1,
        "the head and its whole chain are one dispatch"
    );
    heads[0]
}

/// Run the fused program on `gpu` and the CPU oracle of the graph; the device result meets the oracle's tier-2
/// tolerance.
fn assert_fused_matches_oracle(
    gpu: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
    label: &str,
) {
    let (fused, class) = run_under(gpu, target, g, inputs);
    let oracle_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let want = eval(g, &oracle_inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(fused.shape(), want.shape(), "{label}: shape");
    assert_close_scaled(
        fused.as_f32().unwrap(),
        want.as_f32().unwrap(),
        class.tolerance(),
        label,
    );
}

/// SC-001 (wgpu, the decode Gemv): an F32 `[N, K]` projection at `M == 1` with a bias, an activation, a scale
/// and a residual fuses into one `ContractionEpilogue { head: Dense }` dispatch whose device result meets the
/// oracle. Mutation: omit the residual or the bias leaf in
/// `emit_epilogue` (`read_leaf` returns zero); the fixture's nonzero inputs then differ and this row goes red.
#[test]
fn a_dense_gemv_epilogue_matches_the_oracle_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![n, k])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let residual = b
        .constant("residual", TensorType::f32(vec![1, 1, n]))
        .unwrap();
    let head = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let out = epilogue_chain(&b, head, bias, residual);
    let inputs = HashMap::from([
        (x.id(), HostTensor::f32(vec![1, 1, k], fill(1, k))),
        (w.id(), HostTensor::f32(vec![n, k], fill(2, n * k))),
        (bias.id(), HostTensor::f32(vec![n], fill(3, n))),
        (residual.id(), HostTensor::f32(vec![1, 1, n], fill(4, n))),
    ]);
    let g = b.finish(out).unwrap();
    let head = epilogue_head(&g, &target);
    assert!(matches!(
        head,
        ContractionHead::Dense { weight: DType::F32 }
    ));
    assert_fused_matches_oracle(&mut gpu, target, &g, &inputs, "dense gemv epilogue");
}

/// SC-001 (wgpu, the one-thread-per-output body): a batched `MatMul` the planner lowers serially, with the
/// same chain, fuses into `ContractionEpilogue { head: MatMul }` and meets the oracle.
/// The weight's size-1 batch axis makes the shapes ineligible for the tiled and decode routes, so the serial
/// body is the head's schedule. Mutation as above.
#[test]
fn a_serial_matmul_epilogue_matches_the_oracle_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![1, 2, 3, 4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![1, 1, 4, 5])).unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![5])).unwrap();
    let residual = b
        .constant("residual", TensorType::f32(vec![1, 2, 3, 5]))
        .unwrap();
    let head = b.matmul(a, w).unwrap();
    let out = epilogue_chain(&b, head, bias, residual);
    let inputs = HashMap::from([
        (a.id(), HostTensor::f32(vec![1, 2, 3, 4], fill(11, 24))),
        (w.id(), HostTensor::f32(vec![1, 1, 4, 5], fill(12, 20))),
        (bias.id(), HostTensor::f32(vec![5], fill(13, 5))),
        (
            residual.id(),
            HostTensor::f32(vec![1, 2, 3, 5], fill(14, 30)),
        ),
    ]);
    let g = b.finish(out).unwrap();
    let head = epilogue_head(&g, &spirv_fixture_target());
    assert!(matches!(head, ContractionHead::MatMul));
    assert_fused_matches_oracle(&mut gpu, target, &g, &inputs, "serial matmul epilogue");
}

/// POOT-1036 (wgpu, the generated tiled GEMM over an f32 `[K, N]` weight): a plain `MatMul` head at `M > 1`, the
/// non-checkpoint route (`is_tileable_matmul_in`), with the same four-step chain fuses into one
/// `ContractionEpilogue { head: MatMul }` dispatch, at a single partial row-tile (`M == 5`) and at many
/// (`M == 160`); the device result meets the oracle.
/// Mutation: drop the `is_tileable_matmul_in` arm of `epilogue_head_lowers` (`Tileable` no longer lowers); the head
/// and its chain stay apart and this row goes red.
#[test]
fn a_tiled_matmul_epilogue_matches_the_oracle_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    for m in [5usize, 160] {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
        let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
        let residual = b.constant("residual", TensorType::f32(vec![m, n])).unwrap();
        let head = b.matmul(x, w).unwrap();
        let out = epilogue_chain(&b, head, bias, residual);
        let inputs = HashMap::from([
            (x.id(), HostTensor::f32(vec![m, k], fill(1, m * k))),
            (w.id(), HostTensor::f32(vec![k, n], fill(2, k * n))),
            (bias.id(), HostTensor::f32(vec![n], fill(3, n))),
            (residual.id(), HostTensor::f32(vec![m, n], fill(4, m * n))),
        ]);
        let g = b.finish(out).unwrap();
        let head = epilogue_head(&g, &target);
        assert!(matches!(head, ContractionHead::MatMul), "M={m}");
        assert_fused_matches_oracle(
            &mut gpu,
            target,
            &g,
            &inputs,
            &format!("tiled matmul epilogue M={m}"),
        );
    }
}

/// SC-001 (wgpu, a packed-BF16 weight): the dense head over a checkpoint BF16 `[N, K]` weight read as packed
/// `u32` lanes at `M == 1`, the decode Gemv, with the same chain (`M > 1` is
/// `a_dense_tiled_epilogue_matches_the_oracle_on_wgpu`). The activation, bias and
/// residual are f32; the head result is f32 (the weight is the only narrow operand), so no head rounding applies
/// and the fused result meets the oracle.
#[test]
fn a_dense_bf16_weight_epilogue_matches_the_oracle_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (1usize, 18usize, 9usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n, k], DType::BF16))
        .unwrap();
    let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
    let residual = b.constant("residual", TensorType::f32(vec![m, n])).unwrap();
    let head = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    let out = epilogue_chain(&b, head, bias, residual);
    let weight: Vec<u16> = fill(22, n * k).into_iter().map(bf16_bits).collect();
    let inputs = HashMap::from([
        (x.id(), HostTensor::f32(vec![m, k], fill(21, m * k))),
        (w.id(), HostTensor::bf16(vec![n, k], weight)),
        (bias.id(), HostTensor::f32(vec![n], fill(23, n))),
        (residual.id(), HostTensor::f32(vec![m, n], fill(24, m * n))),
    ]);
    let g = b.finish(out).unwrap();
    let head = epilogue_head(&g, &target);
    assert!(matches!(
        head,
        ContractionHead::Dense {
            weight: DType::BF16
        }
    ));
    assert_fused_matches_oracle(&mut gpu, target, &g, &inputs, "dense bf16 epilogue");
}

/// POOT-1032 (wgpu, the generated tiled GEMM): the dense head at `M > 1` over an F32 and a packed-BF16 `[N, K]`
/// weight, with the same four-step chain, fuses into one `ContractionEpilogue { head: Dense }` dispatch, at a
/// single partial row-tile (`M == 5`) and at many (`M == 160`); the device result meets the oracle. Mutation: omit the residual or the bias leaf in `emit_epilogue`,
/// or drop the epilogue from the tiled store in `tiled_region`; the nonzero inputs then differ and this row goes red.
#[test]
fn a_dense_tiled_epilogue_matches_the_oracle_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    for weight_dtype in [DType::F32, DType::BF16] {
        for m in [5usize, 160] {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![m, k])).unwrap();
            let w = b
                .constant("w", TensorType::new(vec![n, k], weight_dtype))
                .unwrap();
            let bias = b.constant("bias", TensorType::f32(vec![n])).unwrap();
            let residual = b.constant("residual", TensorType::f32(vec![m, n])).unwrap();
            let head = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
            let out = epilogue_chain(&b, head, bias, residual);
            let weight_values = fill(2, n * k);
            let weight = match weight_dtype {
                DType::BF16 => HostTensor::bf16(
                    vec![n, k],
                    weight_values.into_iter().map(bf16_bits).collect(),
                ),
                _ => HostTensor::f32(vec![n, k], weight_values),
            };
            let inputs = HashMap::from([
                (x.id(), HostTensor::f32(vec![m, k], fill(1, m * k))),
                (w.id(), weight),
                (bias.id(), HostTensor::f32(vec![n], fill(3, n))),
                (residual.id(), HostTensor::f32(vec![m, n], fill(4, m * n))),
            ]);
            let g = b.finish(out).unwrap();
            let head = epilogue_head(&g, &target);
            assert_eq!(
                head,
                ContractionHead::Dense {
                    weight: weight_dtype
                }
            );
            assert_fused_matches_oracle(
                &mut gpu,
                target,
                &g,
                &inputs,
                &format!("dense tiled epilogue {weight_dtype:?} M={m}"),
            );
        }
    }
}

/// SC-006 (wgpu): SPIR-V has no scalar bf16, so no narrow-head contraction schedule is admitted on wgpu. The
/// narrow rows (BF16 and F16 heads, multistage narrow chains) therefore run on ROCm (and PTX in the pod batch);
/// here the planner refuses the narrow head by name, so a narrow epilogue can never reach a wgpu kernel with its
/// rounding dropped: `DtypeLowering` on the `bf16` result.
#[test]
fn a_narrow_head_is_refused_by_name_on_wgpu() {
    use poot_graph_plan::{Capability, CompileError, PlanError};

    let b = Builder::for_tests();
    let a = b
        .constant("a", TensorType::new(vec![1, 2], DType::BF16))
        .unwrap();
    let w = b
        .constant("w", TensorType::new(vec![2, 1], DType::BF16))
        .unwrap();
    let residual = b
        .constant("residual", TensorType::new(vec![1, 1], DType::BF16))
        .unwrap();
    let out = b
        .binary(BinOp::Add, b.matmul(a, w).unwrap(), residual)
        .unwrap();
    let g = b.finish(out).unwrap();
    let Err(error) = poot_graph_plan::compile(
        &g,
        &spirv_fixture_target(),
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    ) else {
        panic!("a bf16 head must not plan on wgpu");
    };
    let CompileError::Plan(plan) = error else {
        panic!("a planner refusal, got {error}");
    };
    let PlanError::Refused(refusal) = *plan else {
        panic!("a typed refusal, got {plan}");
    };
    assert_eq!(refusal.missing, Capability::DtypeLowering);
    assert_eq!(refusal.dtypes.output, DType::BF16);
}

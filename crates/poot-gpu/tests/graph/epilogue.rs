//! Contraction epilogue rows on wgpu: a bias, a pointwise activation and a residual add fuse into the
//! contraction that feeds them (`OpKind::ContractionEpilogue`), and the fused device program equals the same
//! graph compiled with no fusion bit for bit, with target, schedule, accumulation order and contraction
//! policy held fixed. The routed-expert hang guard gives the unfused program: it plans one kernel per
//! equation, a contraction on the same schedule the fused plan picks for its head.

use super::*;

use poot_executor::StepInputs;
use poot_executor_parity::weight_map::{TIER2, assert_close as assert_tier2};
use poot_graph_ir::analysis::dispatch_count;
use poot_graph_ir::op::{BinOp, ContractionHead, UnOp};
use poot_graph_ir::types::Scalar;
use poot_graph_plan::{CompileOptions, Submission, compile};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};

/// `tanh(head + bias) * 0.5 + residual`: a bias, a representative pointwise activation, a literal scale and a
/// residual add, each value read once, so the whole chain is one epilogue.
fn epilogue_chain(
    b: &Builder,
    head: poot_graph_ir::Traced,
    bias: poot_graph_ir::Traced,
    residual: poot_graph_ir::Traced,
) -> poot_graph_ir::Traced {
    let biased = b.binary(BinOp::Add, head, bias);
    let activated = b.unary(UnOp::Tanh, biased);
    let scaled = b.binary_scalar(BinOp::Mul, activated, Scalar::F32(0.5));
    b.binary(BinOp::Add, scaled, residual)
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

/// `g` run once on `gpu` (under the routed-expert hang guard when `guarded`), every input a named const.
fn run_under(
    gpu: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
    guarded: bool,
) -> HostTensor {
    let mut builder = WeightStore::builder();
    for (&id, tensor) in inputs {
        let meta = g.meta(id);
        assert!(matches!(meta.storage, Storage::Const), "{id:?} is a const");
        let dense = DenseWeight::try_new(
            tensor.dtype(),
            tensor.shape().to_vec(),
            Arc::from(tensor.view().bytes()),
        )
        .unwrap();
        builder
            .insert(meta.name.clone().unwrap(), WeightEntry::Dense(dense))
            .unwrap();
    }
    let exe = gpu
        .load_weights(
            Arc::new(builder.build()),
            poot_executor::WeightSource::ConstNames,
        )
        .unwrap();
    let staged = staged_replay_with(g, target, guarded);
    let entry = gpu.add_entry(exe, &staged).unwrap();
    let bytes = gpu
        .step(exe, entry, &StepInputs::new(), &mut poot_executor::NoSync)
        .unwrap()
        .read()
        .unwrap();
    gpu.remove_entry(exe, entry).unwrap();
    gpu.unload(exe).unwrap();
    let out = g.aval(g.output);
    assert_eq!(out.dtype, DType::F32, "the wgpu rows read an f32 output");
    HostTensor::f32(
        out.shape.clone(),
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

fn compiled(g: &poot_graph_ir::Graph, target: &Target, guarded: bool) -> poot_graph_plan::Program {
    let (g, caps) = poot_test_util::graph_fixtures::pipeline_input(g, target.caps, guarded);
    compile(
        &g,
        &Target { caps, ..*target },
        &CompileOptions {
            execution: Submission::Replay,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("the graph compiles")
}

/// The one `ContractionEpilogue` of the fully fused program and its head, with the dispatches fusion removed.
fn epilogue_head(g: &poot_graph_ir::Graph, target: &Target) -> (ContractionHead, usize) {
    let fused = compiled(g, target, false);
    let unfused = compiled(g, target, true);
    let heads: Vec<ContractionHead> = fused
        .graph()
        .eqns
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
        fused.graph().eqns
    );
    assert!(
        unfused
            .graph()
            .eqns
            .iter()
            .all(|eqn| !matches!(eqn.op, OpKind::ContractionEpilogue { .. })),
        "the guard keeps one kernel per equation"
    );
    (
        heads[0],
        dispatch_count(unfused.graph()) - dispatch_count(fused.graph()),
    )
}

/// Run the fused program on `gpu` and the unfused one on `hung` (an engine whose device reports the routed-expert
/// hang, which the guarded program is compiled for) and the CPU oracle; the two device results are equal bit
/// for bit and both meet the oracle's tier-2 tolerance.
fn assert_fused_equals_unfused(
    gpu: &mut dyn Executor,
    hung: &mut dyn Executor,
    target: Target,
    g: &poot_graph_ir::Graph,
    inputs: &HashMap<poot_graph_ir::ValueId, HostTensor>,
    label: &str,
) {
    let fused = run_under(gpu, target, g, inputs, false);
    let unfused = run_under(hung, target, g, inputs, true);
    let fused_bits: Vec<u32> = fused
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let unfused_bits: Vec<u32> = unfused
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        fused_bits, unfused_bits,
        "{label}: the fused device program must equal the unfused one bit for bit"
    );
    let oracle_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let want = eval(g, &oracle_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    assert_eq!(fused.shape(), want.shape(), "{label}: shape");
    assert_tier2(
        fused.as_f32().unwrap(),
        want.as_f32().unwrap(),
        TIER2,
        label,
    );
}

/// SC-001 (wgpu, the decode Gemv): an F32 `[N, K]` projection at `M == 1` with a bias, an activation, a scale
/// and a residual fuses into one `ContractionEpilogue { head: Dense }` that removes four dispatches, and the
/// fused and unfused device results are bit-identical. Mutation: omit the residual or the bias leaf in
/// `emit_epilogue` (`read_leaf` returns zero); the fixture's nonzero inputs then differ and this row goes red.
#[test]
fn a_dense_gemv_epilogue_equals_the_unfused_program_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let Some(mut hung) = open_hung_engine_or_skip() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1, k]));
    let w = b.constant("w", TensorType::f32(vec![n, k]));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let residual = b.constant("residual", TensorType::f32(vec![1, 1, n]));
    let head = b.matmul(x, b.transpose(w, vec![1, 0]));
    let out = epilogue_chain(&b, head, bias, residual);
    let inputs = HashMap::from([
        (x.id, HostTensor::f32(vec![1, 1, k], fill(1, k))),
        (w.id, HostTensor::f32(vec![n, k], fill(2, n * k))),
        (bias.id, HostTensor::f32(vec![n], fill(3, n))),
        (residual.id, HostTensor::f32(vec![1, 1, n], fill(4, n))),
    ]);
    let g = b.finish(out);
    let (head, removed) = epilogue_head(&g, &target);
    assert!(matches!(
        head,
        ContractionHead::Dense { weight: DType::F32 }
    ));
    assert_eq!(
        removed, 4,
        "the bias, tanh, scale and residual steps leave the plan"
    );
    assert_fused_equals_unfused(
        &mut gpu,
        &mut hung,
        target,
        &g,
        &inputs,
        "dense gemv epilogue",
    );
}

/// SC-001 (wgpu, the one-thread-per-output body): a batched `MatMul` the planner lowers serially, with the
/// same chain, fuses into `ContractionEpilogue { head: MatMul }` and equals the unfused program bit for bit.
/// The weight's size-1 batch axis makes the shapes ineligible for the tiled and decode routes, so the serial
/// body is the unfused plan's schedule too. Mutation as above.
#[test]
fn a_serial_matmul_epilogue_equals_the_unfused_program_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let Some(mut hung) = open_hung_engine_or_skip() else {
        return;
    };
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, 2, 3, 4]));
    let w = b.constant("w", TensorType::f32(vec![1, 1, 4, 5]));
    let bias = b.constant("bias", TensorType::f32(vec![5]));
    let residual = b.constant("residual", TensorType::f32(vec![1, 2, 3, 5]));
    let head = b.matmul(a, w);
    let out = epilogue_chain(&b, head, bias, residual);
    let inputs = HashMap::from([
        (a.id, HostTensor::f32(vec![1, 2, 3, 4], fill(11, 24))),
        (w.id, HostTensor::f32(vec![1, 1, 4, 5], fill(12, 20))),
        (bias.id, HostTensor::f32(vec![5], fill(13, 5))),
        (residual.id, HostTensor::f32(vec![1, 2, 3, 5], fill(14, 30))),
    ]);
    let g = b.finish(out);
    let (head, removed) = epilogue_head(&g, &spirv_fixture_target());
    assert!(matches!(head, ContractionHead::MatMul));
    assert_eq!(removed, 4);
    assert_fused_equals_unfused(
        &mut gpu,
        &mut hung,
        target,
        &g,
        &inputs,
        "serial matmul epilogue",
    );
}

/// POOT-1036 (wgpu, the generated tiled GEMM over an f32 `[K, N]` weight): a plain `MatMul` head at `M > 1`, the
/// non-checkpoint route (`is_tileable_matmul_in`), with the same four-step chain fuses into one
/// `ContractionEpilogue { head: MatMul }` that removes four dispatches, at a single partial row-tile (`M == 5`)
/// and at many (`M == 160`); the fused and unfused device results are bit-identical and meet the oracle.
/// Mutation: drop the `is_tileable_matmul_in` arm of `epilogue_head_lowers` (`Tileable` no longer lowers); the head
/// and its chain stay apart and this row goes red.
#[test]
fn a_tiled_matmul_epilogue_equals_the_unfused_program_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let Some(mut hung) = open_hung_engine_or_skip() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    for m in [5usize, 160] {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![m, k]));
        let w = b.constant("w", TensorType::f32(vec![k, n]));
        let bias = b.constant("bias", TensorType::f32(vec![n]));
        let residual = b.constant("residual", TensorType::f32(vec![m, n]));
        let head = b.matmul(x, w);
        let out = epilogue_chain(&b, head, bias, residual);
        let inputs = HashMap::from([
            (x.id, HostTensor::f32(vec![m, k], fill(1, m * k))),
            (w.id, HostTensor::f32(vec![k, n], fill(2, k * n))),
            (bias.id, HostTensor::f32(vec![n], fill(3, n))),
            (residual.id, HostTensor::f32(vec![m, n], fill(4, m * n))),
        ]);
        let g = b.finish(out);
        let (head, removed) = epilogue_head(&g, &target);
        assert!(matches!(head, ContractionHead::MatMul), "M={m}");
        assert_eq!(removed, 4, "M={m}: the chain's four steps leave the plan");
        assert_fused_equals_unfused(
            &mut gpu,
            &mut hung,
            target,
            &g,
            &inputs,
            &format!("tiled matmul epilogue M={m}"),
        );
    }
}

/// SC-001 (wgpu, a packed-BF16 weight): the dense head over a checkpoint BF16 `[N, K]` weight read as packed
/// `u32` lanes at `M == 1`, the decode Gemv, with the same chain (`M > 1` is
/// `a_dense_tiled_epilogue_equals_the_unfused_program_on_wgpu`). The activation, bias and
/// residual are f32; the head result is f32 (the weight is the only narrow operand), so no head rounding applies
/// and the fused result equals the unfused one bit for bit.
#[test]
fn a_dense_bf16_weight_epilogue_equals_the_unfused_program_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let Some(mut hung) = open_hung_engine_or_skip() else {
        return;
    };
    let (m, k, n) = (1usize, 18usize, 9usize);
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, k]));
    let w = b.constant("w", TensorType::new(vec![n, k], DType::BF16));
    let bias = b.constant("bias", TensorType::f32(vec![n]));
    let residual = b.constant("residual", TensorType::f32(vec![m, n]));
    let head = b.matmul(x, b.transpose(w, vec![1, 0]));
    let out = epilogue_chain(&b, head, bias, residual);
    let weight: Vec<u16> = fill(22, n * k).into_iter().map(bf16_bits).collect();
    let inputs = HashMap::from([
        (x.id, HostTensor::f32(vec![m, k], fill(21, m * k))),
        (w.id, HostTensor::bf16(vec![n, k], weight)),
        (bias.id, HostTensor::f32(vec![n], fill(23, n))),
        (residual.id, HostTensor::f32(vec![m, n], fill(24, m * n))),
    ]);
    let g = b.finish(out);
    let (head, removed) = epilogue_head(&g, &target);
    assert!(matches!(
        head,
        ContractionHead::Dense {
            weight: DType::BF16
        }
    ));
    assert_eq!(removed, 4);
    assert_fused_equals_unfused(
        &mut gpu,
        &mut hung,
        target,
        &g,
        &inputs,
        "dense bf16 epilogue",
    );
}

/// POOT-1032 (wgpu, the generated tiled GEMM): the dense head at `M > 1` over an F32 and a packed-BF16 `[N, K]`
/// weight, with the same four-step chain, fuses into one `ContractionEpilogue { head: Dense }` that removes four
/// dispatches, at a single partial row-tile (`M == 5`) and at many (`M == 160`); the fused and unfused device
/// results are bit-identical and meet the oracle. Mutation: omit the residual or the bias leaf in `emit_epilogue`,
/// or drop the epilogue from the tiled store in `tiled_region`; the nonzero inputs then differ and this row goes red.
#[test]
fn a_dense_tiled_epilogue_equals_the_unfused_program_on_wgpu() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };
    let Some(mut hung) = open_hung_engine_or_skip() else {
        return;
    };
    let (k, n) = (37usize, 53usize);
    for weight_dtype in [DType::F32, DType::BF16] {
        for m in [5usize, 160] {
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![m, k]));
            let w = b.constant("w", TensorType::new(vec![n, k], weight_dtype));
            let bias = b.constant("bias", TensorType::f32(vec![n]));
            let residual = b.constant("residual", TensorType::f32(vec![m, n]));
            let head = b.matmul(x, b.transpose(w, vec![1, 0]));
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
                (x.id, HostTensor::f32(vec![m, k], fill(1, m * k))),
                (w.id, weight),
                (bias.id, HostTensor::f32(vec![n], fill(3, n))),
                (residual.id, HostTensor::f32(vec![m, n], fill(4, m * n))),
            ]);
            let g = b.finish(out);
            let (head, removed) = epilogue_head(&g, &target);
            assert_eq!(
                head,
                ContractionHead::Dense {
                    weight: weight_dtype
                }
            );
            assert_eq!(
                removed, 4,
                "{weight_dtype:?} M={m}: the bias, tanh, scale and residual steps leave the plan"
            );
            assert_fused_equals_unfused(
                &mut gpu,
                &mut hung,
                target,
                &g,
                &inputs,
                &format!("dense tiled epilogue {weight_dtype:?} M={m}"),
            );
        }
    }
}

/// SC-006 (wgpu): SPIR-V has no scalar bf16, so no narrow-head contraction schedule is admitted on wgpu, fused or
/// not. The narrow rows (BF16 and F16 heads, multistage narrow chains) therefore run on ROCm (and PTX in the pod
/// batch); here the planner refuses the narrow head by name under both policies, so a narrow epilogue can never
/// reach a wgpu kernel with its rounding dropped. The refusal is the same equation-and-capability value either way:
/// `DtypeLowering` on the `bf16` result.
#[test]
fn a_narrow_head_is_refused_by_name_on_wgpu_fused_or_not() {
    use poot_graph_plan::{Capability, CompileError, PlanError};

    let b = Builder::new();
    let a = b.constant("a", TensorType::new(vec![1, 2], DType::BF16));
    let w = b.constant("w", TensorType::new(vec![2, 1], DType::BF16));
    let residual = b.constant("residual", TensorType::new(vec![1, 1], DType::BF16));
    let out = b.binary(BinOp::Add, b.matmul(a, w), residual);
    let g = b.finish(out);
    for guarded in [false, true] {
        let target = spirv_fixture_target();
        let (g, caps) = poot_test_util::graph_fixtures::pipeline_input(&g, target.caps, guarded);
        let error = poot_graph_plan::compile(
            &g,
            &Target { caps, ..target },
            &CompileOptions {
                execution: Submission::Replay,
                limits: poot_graph_plan::CompileLimits::STANDARD,
            },
        )
        .err()
        .unwrap_or_else(|| panic!("{guarded}: a bf16 head must not plan on wgpu"));
        let CompileError::Plan(plan) = error else {
            panic!("{guarded}: a planner refusal, got {error}");
        };
        let PlanError::Refused(refusal) = *plan else {
            panic!("{guarded}: a typed refusal, got {plan}");
        };
        assert_eq!(refusal.missing, Capability::DtypeLowering, "{guarded}");
        assert_eq!(refusal.dtypes.output, DType::BF16, "{guarded}");
    }
}

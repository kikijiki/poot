use super::*;

#[test]
fn concat2_plan_selection() {
    // The two-input concat is the one generated movement kernel on every backend and dtype: the request
    // names the inputs' shapes and the axis, and the dispatch binds no metadata buffer. Pure planner test
    // (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::KernelChoice;
    use poot_kernelgen::{MovementOp, MovementSpec};
    use poot_target::Backend;
    use poot_tensor::DType;
    // a [2,3] concat b [2,5] along axis 1 -> out [2,8].
    let plan_for = |dt: DType, backend| {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2, 3], dt));
        let y = b.constant("y", TensorType::new(vec![2, 5], dt));
        let out = b.concat(1, &[x, y]);
        let g = b.finish(out);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::Concat { .. }))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    for (dt, backend) in [
        (DType::F32, Backend::SpirvVulkan),
        (DType::F32, Backend::Nvptx),
        (DType::I32, Backend::SpirvVulkan),
    ] {
        let (plan_, choice_) = plan_for(dt, backend);
        assert!(
            matches!(single_meta(&plan_), Some([])),
            "{dt:?} {backend:?}: one dispatch, no metadata buffer"
        );
        let KernelChoice::Generated(poot_kernelgen::KernelRequest::Movement(MovementSpec {
            op: MovementOp::Concat { axis, in_shapes },
            out_shape,
            ..
        })) = choice_
        else {
            panic!("{dt:?} {backend:?}: expected the generated concat");
        };
        assert_eq!(axis, 1);
        assert_eq!(in_shapes, [vec![2, 3], vec![2, 5]]);
        assert_eq!(out_shape, [2, 8]);
    }
}

#[test]
fn dyn_update_slice_plan_selection() {
    // A dynamic update-slice is the one generated movement kernel on every backend and dtype: a runtime
    // index names its own element type in the request (the contiguous KV-cache write stays valid at every
    // position of a capture), a literal index is baked, and no form binds a metadata buffer. Pure planner
    // test (no GPU).
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::KernelChoice;
    use poot_kernelgen::{MovementOp, MovementSpec, UpdateIndex};
    use poot_target::Backend;
    // operand [4,5], update [4,2] along axis 1, runtime index -> the dynamic form.
    let plan_dyn = |dt: TensorType, backend| {
        let b = Builder::new();
        let operand = b.constant("operand", dt.clone());
        let mut ushape = dt.shape.clone();
        ushape[1] = 2;
        let update = b.constant("update", TensorType::new(ushape, dt.dtype));
        let index = b.constant("index", TensorType::scalar(poot_tensor::DType::F32));
        let out = b.dynamic_update_slice_dyn(operand, update, index, 1);
        let g = b.finish(out);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::DynamicUpdateSlice { .. }))
            .unwrap();
        plan_eqn_with_choice(
            &g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    let update_index = |plan_: &poot_graph_plan::Plan, choice_: KernelChoice| {
        assert!(
            matches!(single_meta(plan_), Some([])),
            "one dispatch, no metadata buffer"
        );
        match choice_ {
            KernelChoice::Generated(poot_kernelgen::KernelRequest::Movement(MovementSpec {
                op:
                    MovementOp::UpdateSlice {
                        update_shape,
                        axis,
                        index,
                    },
                ..
            })) => {
                assert_eq!((update_shape.as_slice(), axis), (&[4, 2][..], 1));
                index
            }
            other => panic!("expected the generated update-slice: {other:?}"),
        }
    };
    for (table, backend) in [
        (TensorType::f32(vec![4, 5]), Backend::SpirvVulkan),
        (TensorType::f32(vec![4, 5]), Backend::Nvptx),
        (
            TensorType::new(vec![4, 5], poot_tensor::DType::I32),
            Backend::SpirvVulkan,
        ),
    ] {
        let (plan_, choice_) = plan_dyn(table, backend);
        assert_eq!(
            update_index(&plan_, choice_),
            UpdateIndex::Runtime {
                dtype: poot_kernel_ir::Ty::F32
            },
            "{backend:?}"
        );
    }
    // A LITERAL index is baked into the body.
    {
        let b = Builder::new();
        let operand = b.constant("operand", TensorType::f32(vec![4, 5]));
        let update = b.constant("update", TensorType::f32(vec![4, 2]));
        let out = b.dynamic_update_slice(operand, update, 1, 1);
        let g = b.finish(out);
        let eqn = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::DynamicUpdateSlice { .. }))
            .unwrap();
        let (plan_, choice_) = plan_eqn_with_choice(
            &g,
            eqn,
            Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
        )
        .unwrap();
        assert_eq!(update_index(&plan_, choice_), UpdateIndex::Static(1));
    }
}

#[test]
fn dyn_update_slice_dynamic_gpu_matches_static() {
    // G3d-2 on the Arc: a DynamicUpdateSlice with a RUNTIME index (read from a buffer) matches both the
    // CPU eval and the static-index (baked) result. operand[2,4,3], update[2,1,3] written at axis=1 slot 2.
    let _gpu_guard = gpu_lock();
    let Some((mut exec, target)) = open_engine_or_skip() else {
        return;
    };
    let (rows, cap, cols, idx) = (2usize, 4usize, 3usize, 2usize);
    let operand_d: Vec<f32> = (0..rows * cap * cols).map(|i| i as f32).collect();
    let update_d: Vec<f32> = (0..rows * cols).map(|i| 100.0 + i as f32).collect();

    // dynamic-index graph: index is a scalar value buffer (the runtime slot index, like the Pos slot).
    let b = Builder::new();
    let operand = b.constant("operand", TensorType::f32(vec![rows, cap, cols]));
    let update = b.constant("update", TensorType::f32(vec![rows, 1, cols]));
    let index = b.constant("index", TensorType::scalar(poot_tensor::DType::F32));
    let out = b.dynamic_update_slice_dyn(operand, update, index, 1);
    let (oi, ui, ii) = (operand.id(), update.id(), index.id());
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        oi,
        HostTensor::f32(vec![rows, cap, cols], operand_d.clone()),
    );
    inputs.insert(ui, HostTensor::f32(vec![rows, 1, cols], update_d.clone()));
    inputs.insert(ii, HostTensor::scalar(idx as f32));

    let eval_inputs: HashMap<_, Value> = inputs
        .iter()
        .map(|(&id, t)| (id, Value::from(t.clone())))
        .collect();
    let cpu = eval(&g, &eval_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let got = run_via_contract(&mut exec, target, &g, &inputs);

    // static-index reference (baked idx).
    let bs = Builder::new();
    let so = bs.constant("operand", TensorType::f32(vec![rows, cap, cols]));
    let su = bs.constant("update", TensorType::f32(vec![rows, 1, cols]));
    let sout = bs.dynamic_update_slice(so, su, idx, 1);
    let (soi, sui) = (so.id(), su.id());
    let gs = bs.finish(sout);
    let mut sinputs = HashMap::new();
    sinputs.insert(
        soi,
        Value::from(HostTensor::f32(vec![rows, cap, cols], operand_d)),
    );
    sinputs.insert(
        sui,
        Value::from(HostTensor::f32(vec![rows, 1, cols], update_d)),
    );
    let stat = eval(&gs, &sinputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    assert_eq!(
        got.as_f32().unwrap(),
        cpu.as_f32().unwrap(),
        "gpu dynamic-dus != cpu"
    );
    assert_eq!(
        cpu.as_f32().unwrap(),
        stat.as_f32().unwrap(),
        "dynamic-dus != static-dus"
    );
}

/// Card 636 SC-001: a compiled program records which kernel each equation chose. One traced graph on
/// AMDGCN holds a softcapped prefill attention, which `compile` fuses into a flash prefill (the shipped
/// `flash_prefill` template: the synthesized region kernel
/// has no softcap), a reshape of its output (an alias, no launch) and a matmul over it (M > 1 on AMDGCN
/// has no tensor-core, tiled or GEMV arm and a rank-4 weight never takes the tiled GEMM, so it is the generated
/// serial `matmul_batched_dt_grid`). The
/// choices read `Imported`, `NoDispatch(Alias)` and `Generated`, through `Program::kernel_choice`.
/// Mutation: plan the flash prefill as the generated region kernel in `planner/attention.rs`; the
/// attention row reads `Generated` and goes red. Pure planner test (no GPU).
#[test]
fn the_kernel_choice_names_generated_imported_and_no_dispatch_equations() {
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{CompileOptions, KernelChoice, NoDispatch, Submission, Target, compile};
    use poot_kernelgen::{ContractionSpec, KernelRequest, ScoreTransform};
    use poot_target::{AmdArch, Backend};

    let (hq, l, d) = (2usize, 4usize, 8usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let attention = poot_graph_ir::ops::attention_prefill_softcap(
        &b,
        q,
        k,
        v,
        1,
        1.0 / (d as f32).sqrt(),
        mask,
        Some(30.0),
    );
    let regrouped = b.reshape(attention, vec![1, hq, d, l]);
    let w = b.constant("w", TensorType::f32(vec![1, hq, l, 6]));
    let out = b.matmul(regrouped, w);
    let g = b.finish(out);

    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    let target = Target {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    };
    let options = CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD);
    let program = compile(&g, &target, &options).expect("the fixture compiles for AMDGCN");
    let choice_of = |want: fn(&OpKind) -> bool| {
        let (eqn, _) = program
            .planned()
            .find(|(eqn, _)| want(&eqn.op))
            .unwrap_or_else(|| {
                panic!(
                    "the compiled program keeps the equation: {:?}",
                    program
                        .planned()
                        .map(|(e, _)| e.op.name())
                        .collect::<Vec<_>>()
                )
            });
        program.kernel_choice(eqn)
    };

    let attention = choice_of(|op| matches!(op, OpKind::Attention(_)));
    assert!(
        matches!(
            attention,
            KernelChoice::Generated(KernelRequest::Attention(spec))
                if matches!(spec.score, ScoreTransform::ScaleSoftcap { .. })
        ),
        "softcapped flash prefill is the generated attention: {attention:?}"
    );
    let reshape = choice_of(|op| matches!(op, OpKind::Reshape { .. }));
    assert!(
        matches!(reshape, KernelChoice::NoDispatch(NoDispatch::Alias)),
        "a reshape launches nothing: {reshape:?}"
    );
    let matmul = choice_of(|op| matches!(op, OpKind::MatMul));
    assert!(
        matches!(
            matmul,
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Serial { .. }))
        ),
        "the M > 1 matmul is the generated serial kernel: {matmul:?}"
    );
}

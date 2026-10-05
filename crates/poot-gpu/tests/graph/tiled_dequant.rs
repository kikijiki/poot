use super::*;
use poot_graph_plan::{ImportedKernel, KernelChoice};

/// Whether a plan's choice is the shipped kernel `kernel`.
fn is_imported(choice: &KernelChoice, kernel: ImportedKernel) -> bool {
    matches!(choice, KernelChoice::Imported { kernel: k, .. } if *k == kernel)
}

#[test]
fn pack_unpack_i8_gpu_matches_cpu() {
    // GPU pack/unpack-i8 kernels (quantized KV stored as i32 words, 4 codes per word, since wgpu has no 8-bit buffer).
    // `unpack_i8(pack_i8(x))` on the GPU must reproduce CPU eval's lossless round-trip (round(x) clamped to [-127,127]) and
    // exercises the i32 intermediate buffer. cols is not a multiple of 4 (tail word zero-pads) and x spans past +-127 (clamp).
    let _gpu_guard = gpu_lock();
    let Some((mut exec, target)) = open_engine_or_skip() else {
        return;
    };
    let (rows, cols) = (4usize, 10usize); // W = ceil(10/4) = 3, last word holds only 2 codes
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    let packed = b.pack_i8(x);
    let out = b.unpack_i8(packed, cols);
    let xi = x.id;
    let g = b.finish(out);

    // span past +-127 so the clamp is exercised; mix signs.
    let xd: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.37).sin() * 200.0 - 30.0)
        .collect();
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![rows, cols], xd));
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
    assert_eq!(got.shape(), cpu.shape(), "round-trip shape");
    assert_eq!(
        got.shape(),
        vec![rows, cols],
        "round-trip restores [rows, cols]"
    );
    for (i, (a, c)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu.as_f32().unwrap().iter())
        .enumerate()
    {
        assert_eq!(*a, *c, "pack/unpack roundtrip elem {i}: gpu {a} vs cpu {c}");
        assert!(
            *a >= -127.0 && *a <= 127.0,
            "code {i} out of int8 range: {a}"
        );
        assert_eq!(*a, a.round(), "code {i} not integral: {a}");
    }
}

#[test]
fn pack_unpack_i8_plan_selection() {
    // The INT8 KV-cache pack/unpack kernels lower to the imported ComputeMeta kernels (keys "pack_i8:imported" /
    // "unpack_i8:imported", meta=[L]) on both backends. No GPU.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::Plan;
    use poot_target::Backend;
    let plan_of = |g: &poot_graph_ir::Graph, want: fn(&OpKind) -> bool, backend| {
        let eqn = g.eqns.iter().find(|e| want(&e.op)).unwrap();
        plan_eqn_with_choice(
            g,
            eqn,
            backend,
            &poot_test_util::device_caps::default_caps_for(backend),
        )
        .unwrap()
    };
    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        // one graph carrying both ops (unpack consumes pack); L=10 -> W=ceil(10/4)=3.
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 10]));
        let packed = b.pack_i8(x);
        let out = b.unpack_i8(packed, 10);
        let g = b.finish(out);
        match plan_of(&g, |op| matches!(op, OpKind::PackI8), backend) {
            (Plan::ComputeMeta { meta, .. }, choice) => {
                assert!(
                    is_imported(&choice, ImportedKernel::PackI8),
                    "{backend:?}: {choice:?}"
                );
                assert_eq!(meta, vec![10u32], "pack meta=[L] {backend:?}");
            }
            _ => panic!("pack_i8 {backend:?} should be the imported ComputeMeta kernel"),
        }
        match plan_of(&g, |op| matches!(op, OpKind::UnpackI8 { .. }), backend) {
            (Plan::ComputeMeta { meta, .. }, choice) => {
                assert!(
                    is_imported(&choice, ImportedKernel::UnpackI8),
                    "{backend:?}: {choice:?}"
                );
                assert_eq!(meta, vec![10u32], "unpack meta=[len] {backend:?}");
            }
            _ => panic!("unpack_i8 {backend:?} should be the imported ComputeMeta kernel"),
        }
    }
}

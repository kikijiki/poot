//! Packed-weight production receipts, driven through the executor contract (Card 546b): a packed
//! `format` contraction at real projection dims, the packed-KV read composition, and the Q6_K GEMV
//! determinism probe all load a `WeightStore` and run `Engine<WgpuDevice>` through
//! `load_weights`/`add_entry`/`step` (or `poot_executor_parity::run_once` for the one-shot graph),
//! comparing against an independent CPU reference.

use poot_graph_ir::test_support::const_weights::ConstWeights;
use std::sync::Arc;

use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, OpKind, Slot};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::format::WeightFormat;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_tensor::{DType, HostView};

use crate::device::WgpuDevice;

fn wgpu() -> Option<WgpuDevice> {
    poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compile `g` for `target` through the contract's one staged program: single stage, single device.
fn staged(g: &Graph, target: Target) -> StagedProgram {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .unwrap()
}

/// One named packed linear at `[n, k]`: load it into `engine`, run one `[m, k]` activation step, and
/// compare against `owner.decode_row`'s dense reference (the 541-oracle decoder, independent of the
/// GPU kernel's own in-kernel unpack).
fn packed_linear_case(
    engine: &mut Engine<WgpuDevice>,
    target: Target,
    k: usize,
    n: usize,
    format: WeightFormat,
    m: usize,
    seed: u64,
) {
    use poot_graph_ir::test_support::packed_graphs::packed_linear;

    let owner = Arc::new(poot_quant::test_support::packed::random_payload(
        format,
        [n, k],
        seed,
    ));
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]))
        .unwrap();
    let y = packed_linear(&b, x, "layer", owner.weight(), None, None)
        .unwrap_or_else(|e| panic!("{format:?} K={k} N={n} M={m}: packed_linear: {e}"));
    let compiled = b.finish(y).unwrap();
    let store = {
        let mut builder = WeightStore::builder();
        builder
            .insert("layer", WeightEntry::Packed(Arc::clone(&owner)))
            .unwrap();
        builder.build()
    };
    let (named, map) = ConstWeights::bind(&compiled, &store);
    let staged = staged(&named, target);
    assert!(
        staged
            .stages()
            .next()
            .unwrap()
            .2
            .planned()
            .any(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. })),
        "{format:?} K={k} N={n} M={m}: compile should claim the packed linear"
    );

    let exe = engine.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = engine.add_entry(exe, &staged).unwrap();

    let x_id = compiled
        .slots()
        .iter()
        .find(|&&(_, kind)| kind == Slot::Activation)
        .expect("packed_linear declares an activation slot")
        .0;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
    let shape = [m, k];
    let mut inputs = StepInputs::new();
    inputs.push(
        compiled.meta(x_id).slot_key().unwrap().clone(),
        &shape,
        HostView::new(DType::F32, m * k, bytemuck::cast_slice(&a)).unwrap(),
    );
    let got = bytemuck_f32(
        &engine
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|e| panic!("{format:?} K={k} N={n} M={m}: dispatch: {e}"))
            .read()
            .unwrap(),
    );
    assert_eq!(got.len(), m * n, "{format:?} K={k} N={n} M={m} shape");
    let mut row = vec![0.0f32; k];
    let mut cref = vec![0.0f32; m * n];
    for o in 0..n {
        owner
            .decode_row(o, &mut row)
            .unwrap_or_else(|e| panic!("{format:?} K={k} N={n}: decode_row({o}): {e}"));
        for mi in 0..m {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[mi * k + kk] * row[kk];
            }
            cref[mi * n + o] = acc;
        }
    }
    // The GPU kernel's tiled/tree accumulation order differs from the CPU reference's naive
    // sequential sum, so summing K terms accumulates rounding error that grows with K (observed
    // ~sqrt(K) at these real dims: max_abs 9.9e-5 at K=896 vs 2.7e-4 at K=4864, Q8_0 M=4). The
    // relative term alone catches a real correctness bug (which produces errors orders of
    // magnitude larger); this K-scaled absolute floor only absorbs that legitimate accumulation
    // noise at the real K=4864 down-proj width.
    for (i, (x, c)) in got.iter().zip(&cref).enumerate() {
        let tol = 1e-3 * c.abs().max(1e-2) + 3e-6 * (k as f32).sqrt();
        assert!(
            (x - c).abs() <= tol,
            "{format:?} K={k} N={n} M={m} elem {i}: gpu {x} vs dense {c}"
        );
    }
    engine.remove_entry(exe, entry).unwrap();
    engine.unload(exe).unwrap();
}

/// Card 097 hardening, retargeted onto `packed_linear` (card 545b: the legacy `MatMulDequant` family is
/// deleted). The packed contraction kernel was only ever verified at toy K; real models use K up to 4864
/// (qwen2.5-0.5b intermediate), where a large-K indexing or f32-accumulation bug would be invisible.
///
/// Q8_0 at M=1 (decode, `Schedule::Gemv`) at these real dims is already covered end to end against a real
/// GGUF checkpoint (`poot-llm`'s `probe_rocm_q8_0_decode_matches_cpu`,
/// `q8_0_0p5b_gguf_ptx_decode_matches_cpu`), so this covers what those leave: Q8_0 at M=4 (ragged prefill,
/// `Schedule::RegisterTiled`) and Q4_0 at both M=1 and M=4 (no real-GGUF checkpoint in this repo stores a Q4_0
/// tensor at these widths, so Q4_0's real-dims decode has no other coverage at all).
///
/// Binds the weight as a `WeightEntry::Packed` source (never a dense f32 copy) and compares against a
/// host dense reference built from [`poot_quant::PackedPayload::decode_row`].
#[test]
fn packed_linear_real_projection_dims_gpu_matches_dense() {
    let Some(device) = wgpu() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);
    // (K=in, N=out): q/o-proj (896,896), k/v-proj (896,128), gate/up-proj (896,4864), down-proj (4864,896).
    let shapes = [(896usize, 896usize), (896, 128), (896, 4864), (4864, 896)];
    let mut seed = 1u64;
    for &(k, n) in &shapes {
        packed_linear_case(&mut engine, target, k, n, WeightFormat::Q8_0, 4, seed); // the M>1/ragged-prefill gap
        seed += 1;
        packed_linear_case(&mut engine, target, k, n, WeightFormat::Q4_0, 1, seed); // no real-GGUF Q4_0 coverage at all
        seed += 1;
        packed_linear_case(&mut engine, target, k, n, WeightFormat::Q4_0, 4, seed);
        seed += 1;
    }
}

/// Q6_K GEMV determinism regression probe, the same class of bug fixed for ROCm in card 174: two
/// back-to-back replay steps of the same entry with identical inputs must match bit for bit.
#[test]
fn wgpu_q6k_gemv_determinism_probe() {
    use poot_graph_ir::test_support::packed_graphs::packed_linear;
    use poot_test_util::max_abs_error;

    let Some(device) = wgpu() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let (m, k, n) = (1usize, 5376usize, 4096usize);
    let owner = Arc::new(poot_quant::test_support::packed::random_payload(
        WeightFormat::Q6_K,
        [n, k],
        0x9E3779B9,
    ));
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]))
        .unwrap();
    let y = packed_linear(&b, x, "layer", owner.weight(), None, None)
        .expect("Q6_K packed_linear at the GEMV shape");
    let compiled = b.finish(y).unwrap();
    let store = {
        let mut builder = WeightStore::builder();
        builder
            .insert("layer", WeightEntry::Packed(Arc::clone(&owner)))
            .unwrap();
        builder.build()
    };
    let (named, map) = ConstWeights::bind(&compiled, &store);
    let staged = staged(&named, target);

    let exe = engine.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = engine.add_entry(exe, &staged).unwrap();

    let x_id = compiled
        .slots()
        .iter()
        .find(|&&(_, kind)| kind == Slot::Activation)
        .expect("packed_linear declares an activation slot")
        .0;
    // Deterministic PRNG-free fill (xorshift), same generator as the ROCm probe.
    let mut state: u32 = 0x1234_5678;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let a_data: Vec<f32> = (0..m * k)
        .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let shape = [m, k];
    let mut inputs = StepInputs::new();
    inputs.push(
        compiled.meta(x_id).slot_key().unwrap().clone(),
        &shape,
        HostView::new(DType::F32, m * k, bytemuck::cast_slice(&a_data)).unwrap(),
    );

    let run1 = bytemuck_f32(
        &engine
            .step(exe, entry, &inputs, &mut NoSync)
            .expect("wgpu q6k gemv determinism probe run 1")
            .read()
            .unwrap(),
    );
    let run2 = bytemuck_f32(
        &engine
            .step(exe, entry, &inputs, &mut NoSync)
            .expect("wgpu q6k gemv determinism probe run 2")
            .read()
            .unwrap(),
    );

    assert_eq!(run1.len(), m * n, "Q6K gemv output shape");
    let bits1: Vec<u32> = run1.iter().map(|f| f.to_bits()).collect();
    let bits2: Vec<u32> = run2.iter().map(|f| f.to_bits()).collect();
    let same = bits1 == bits2;
    let max_abs = max_abs_error(&run1, &run2);
    let n_diff = bits1
        .iter()
        .zip(bits2.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "wgpu_q6k_gemv_determinism_probe: k={k} n={n} run1==run2? {same} \
         max_abs={max_abs:.6e} n_diff_elems={n_diff}/{n}"
    );
    assert!(
        same,
        "REGRESSION (or latent bug): wgpu/SPIRV Q6_K GEMV kernel is nondeterministic across two \
         back-to-back steps with IDENTICAL inputs on the SAME device: \
         max_abs={max_abs:.6e} n_diff_elems={n_diff}/{n} - cross-wave LDS at GEMV_WIDTH=128 (4 \
         wave32s) is racing on the SPIRV/wgpu backend, the same class of bug fixed for ROCm in card 174."
    );
}

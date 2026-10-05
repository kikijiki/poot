//! Packed-weight production receipts, driven through the executor contract (Card 546b): a packed
//! `format` contraction at real projection dims, the packed-KV read composition, and the Q6_K GEMV
//! determinism probe all load a `WeightStore` and run `Engine<WgpuDevice>` through
//! `load_weights`/`add_entry`/`step` (or `poot_executor_parity::run_once` for the one-shot graph),
//! comparing against an independent CPU reference.

use std::sync::Arc;

use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_executor_parity::{ConstFixture, run_once};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::{Scalar, TensorType};
use poot_graph_ir::{Graph, OpKind, Slot, ValidationOutputs};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::format::WeightFormat;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_tensor::{DType, HostTensor, HostView};

use crate::device::WgpuDevice;

fn wgpu() -> Option<WgpuDevice> {
    poot_test_util::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Compile `g` for `target` through the contract's one staged program: single stage, single device,
/// `Submission::Replay` (the only submission the engine admits, Card 546a).
fn staged(g: &Graph<ValidationOutputs>, target: Target) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
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
    use poot_graph_ir::ops::packed_linear;

    let owner = Arc::new(poot_test_util::packed::random_payload(format, [n, k], seed));
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]));
    let y = packed_linear(&b, x, "layer", owner.weight(), None, None)
        .unwrap_or_else(|e| panic!("{format:?} K={k} N={n} M={m}: packed_linear: {e}"));
    let compiled = b.finish(y).with_validations(Vec::new());
    let staged = staged(&compiled, target);
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

    let store = {
        let mut builder = WeightStore::builder();
        builder
            .insert("layer", WeightEntry::Packed(Arc::clone(&owner)))
            .unwrap();
        builder.build()
    };
    let exe = engine
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged).unwrap();

    let x_id = compiled
        .slots
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
/// `Schedule::Tiled`) and Q4_0 at both M=1 and M=4 (no real-GGUF checkpoint in this repo stores a Q4_0
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
    use poot_graph_ir::ops::packed_linear;
    use poot_test_util::max_abs_error;

    let Some(device) = wgpu() else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let (m, k, n) = (1usize, 5376usize, 4096usize);
    let owner = Arc::new(poot_test_util::packed::random_payload(
        WeightFormat::Q6_K,
        [n, k],
        0x9E3779B9,
    ));
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![m, k]));
    let y = packed_linear(&b, x, "layer", owner.weight(), None, None)
        .expect("Q6_K packed_linear at the GEMV shape");
    let compiled = b.finish(y).with_validations(Vec::new());
    let staged = staged(&compiled, target);

    let store = {
        let mut builder = WeightStore::builder();
        builder
            .insert("layer", WeightEntry::Packed(Arc::clone(&owner)))
            .unwrap();
        builder.build()
    };
    let exe = engine
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = engine.add_entry(exe, &staged).unwrap();

    let x_id = compiled
        .slots
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

#[test]
fn packed_kv_read_path_gpu_matches_cpu() {
    // Quantized-KV read-side composition as the paged decode graph uses it: quantize (per-row absmax/127 + round + clamp), pack
    // to i32 words, gather the packed words and per-row scales by an index (paged slot-map fetch), unpack, dequantize. Checks
    // the packed i32 tensor survives `gather` on the GPU: gather is a 4-byte-word copy and poot-graph-plan's `fty` maps
    // I32 -> Ty::F32, so it runs its f32 kernel on the i32 bytes; the result must be byte-identical to CPU eval.
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};

    let Some(device) = wgpu() else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let (rows, cols) = (4usize, 8usize); // 4 tokens x head_dim 8 -> 2 i32 words/row
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![rows, cols]));
    // per-token scale = absmax/127 (the 4c-i quantize composition).
    let neg = b.unary(UnOp::Neg, x);
    let absx = b.binary(BinOp::Max, x, neg);
    let absmax = b.reduce(RedOp::Max, absx, 1, true);
    let scale = b.binary_scalar(BinOp::Mul, absmax, Scalar::F32(1.0 / 127.0));
    let q = b.binary(BinOp::Div, x, scale);
    let r = b.unary(UnOp::Round, q);
    let lo = b.binary_scalar(BinOp::Max, r, Scalar::F32(-127.0));
    let neg_lo = b.unary(UnOp::Neg, lo);
    let m = b.binary_scalar(BinOp::Max, neg_lo, Scalar::F32(-127.0));
    let codes = b.unary(UnOp::Neg, m); // = min(lo, 127), the clamped int8 codes
    // pack + the slot-map gather (reverse permutation) on BOTH the i32 words and the f32 scales, then
    // unpack + dequant - the read path of one cached layer.
    let packed = b.pack_i8(codes); // [rows, 2] i32 - the storage word tensor
    let idx = b.constant("idx", TensorType::f32(vec![rows]));
    let pg = b.gather(packed, 0, idx); // i32 gather (the byte-copy under test)
    let sg = b.gather(scale, 0, idx); // f32 scale gather, same permutation
    let un = b.unpack_i8(pg, cols); // [rows, cols] f32
    let deq = b.binary(BinOp::Mul, un, sg); // dequant: codes * per-row scale (broadcasts over cols)
    let g = b.finish(deq);

    let xd: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.41).cos() * 6.0 - 1.0)
        .collect();
    let idxd: Vec<f32> = (0..rows).rev().map(|i| i as f32).collect(); // [3,2,1,0]
    let consts = [
        ConstFixture {
            name: "x",
            tensor: HostTensor::f32(vec![rows, cols], xd.clone()),
        },
        ConstFixture {
            name: "idx",
            tensor: HostTensor::f32(vec![rows], idxd.clone()),
        },
    ];
    let got = run_once(exec.as_mut(), target, &g, &consts, &[]).unwrap();
    let got = got.as_f32().expect("the dequant output is F32");

    // Independent hand reference (not eval): per-row quantize-dequantize of the gathered rows. eval cannot be the oracle: its
    // copy ops drop the i32 payload and its f32 mirror is lossy for arbitrary 32-bit words.
    assert_eq!(got.len(), rows * cols, "dequant restores [rows, cols]");
    for (i, &idx) in idxd.iter().enumerate() {
        let src = idx as usize;
        let row = &xd[src * cols..src * cols + cols];
        let absmax = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = absmax / 127.0;
        for (j, &rv) in row.iter().enumerate() {
            let code = if scale == 0.0 {
                0.0
            } else {
                (rv / scale).round().clamp(-127.0, 127.0)
            };
            let want = code * scale;
            let g = got[i * cols + j];
            assert!(
                (g - want).abs() < 1e-4,
                "packed-kv read [{i},{j}]: gpu {g} vs ref {want} (src row {src}; i32 gather must survive)"
            );
        }
    }
}

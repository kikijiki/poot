//! Card 547b acceptance on real wgpu hardware: SC-001 (disjoint-lifetime arena reuse - peak bytes
//! below the naive sum, output matches the CPU oracle) and the arena-reuse-is-transparent half of
//! SC-002 (a short value reusing a larger freed arena slot still matches the oracle through a real
//! dispatch). `pootc`'s STABLE-MIR-compiled `Add`/`Reduce` kernels below have a statically known
//! shape per equation and bake their own bound into the generated SPIR-V, so reverting `poot-runtime`'s
//! length-buffer wiring to `DeviceBuffer::elem_count()` left this fixture's own tests green (recorded,
//! then reverted) - neither kernel class here ever reads the per-dispatch length buffer at runtime.
//! This file is still real evidence: arena-slot reuse does not, by itself, corrupt a real GPU
//! dispatch. The decoupled-length contract itself (review F1: a shape-generic kernel that actually
//! reads its own `Arg::elems` at runtime, not a value baked into the compiled code) is proven by
//! [`len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity`] below, driving
//! `WgpuDevice` directly through the same begin/dispatch/finish/replay sequence `Engine::step` uses
//! internally (the `executor_contract.rs` SC-015 precedent) - and, model-free and backend-neutral,
//! in `poot_executor`'s own `tests/buffer_plan_arena.rs`.
//!
//! The fixture below is unfused (`FusionPolicy::MoeHangGuard`) so its five equations compile to five
//! separate dispatches, in a known order, with a known buffer-plan lifetime for each:
//!
//! ```text
//! a, b: const [K]
//! v1  = a + b                 // eqn 0: [K], born here
//! s1  = reduce_sum(v1, axis0) // eqn 1: [1]; v1 last read here (dies)
//! c:    const [1], negative
//! v2  = c + c                 // eqn 2: [1]; v1's freed [K] slot is reused (SC-002's "short value")
//! s2  = reduce_max(v2, axis0) // eqn 3: [1]
//! out = s1 + s2                // eqn 4
//! ```
//!
//! `c` is negative and `s2` uses `Max` (not `Sum`, like `s1`) so that if this kernel class ever did
//! start reading the length buffer and an unguarded extra thread folded in a zero, the result would
//! visibly differ (`max(c + c, 0) != c + c`) rather than coincidentally agreeing (`sum` would not:
//! zero is `Add`'s identity) - the sharpest oracle comparison this file's fixture can offer, even
//! though today's kernels do not exercise that path.

use std::sync::Arc;

use poot_executor::{Arg, BufferRole, Device, Dispatch, Engine, Executor, NoSync, StepInputs};
use poot_gpu::device::WgpuDevice;
use poot_graph_ir::op::RedOp;
use poot_graph_ir::{Builder, TensorType, ValidationOutputs};
use poot_graph_plan::{
    CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
    StagedProgram, Submission, Target, TargetSet, compile_staged,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_target::BufferStorage;
use poot_tensor::DType;
use poot_test_util::assert_close_rel;

type Graph = poot_graph_ir::Graph<ValidationOutputs>;

const K: usize = 64;

/// `a`, `b` distinct per-lane values (so `reduce_add(a + b)` is not accidentally invariant under a
/// wrong element count); `c` a single *negative* scalar, and `s2` a `Max` reduce (not `Sum`): a
/// robustness-clamped out-of-bounds read of a storage buffer commonly reads back zero, so an extra,
/// wrongly-admitted thread folding a zero into a `Sum` is invisible (zero is `Add`'s identity) but
/// folding a zero into a `Max` of an all-negative true element is not - `max(c + c, 0) != c + c`.
fn fixture() -> (Graph, Vec<f32>, Vec<f32>, f32) {
    let a: Vec<f32> = (0..K).map(|i| 1.0 + i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..K).map(|i| -0.5 + i as f32 * 0.02).collect();
    let c = -5.0f32;

    let builder = Builder::new();
    let ta = builder.constant("a", TensorType::f32(vec![K]));
    let tb = builder.constant("b", TensorType::f32(vec![K]));
    let v1 = builder.binary(poot_graph_ir::op::BinOp::Add, ta, tb);
    let s1 = builder.reduce(RedOp::Sum, v1, 0, true);
    let tc = builder.constant("c", TensorType::f32(vec![1]));
    let v2 = builder.binary(poot_graph_ir::op::BinOp::Add, tc, tc);
    let s2 = builder.reduce(RedOp::Max, v2, 0, true);
    let out = builder.binary(poot_graph_ir::op::BinOp::Add, s1, s2);
    (builder.finish(out).with_validations(Vec::new()), a, b, c)
}

fn store(a: &[f32], b: &[f32], c: f32) -> Arc<WeightStore> {
    let mut wb = WeightStore::builder();
    for (name, bytes) in [("a", a.to_vec()), ("b", b.to_vec()), ("c", vec![c])] {
        let dense = DenseWeight::try_new(
            DType::F32,
            vec![bytes.len()],
            Arc::from(bytemuck_bytes(&bytes)),
        )
        .unwrap();
        wb.insert(name, WeightEntry::Dense(dense)).unwrap();
    }
    Arc::new(wb.build())
}

fn bytemuck_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `compile`'s own `MoeHangGuard` test pattern (`compile.rs`'s `compile_moe_hang_guard_skips_fuse`):
/// skip `fuse` (and the tiled-GEMM choice) so the fixture's five equations stay five separate
/// dispatches, each with the exact lifetime the module doc names.
const UNFUSED_REPLAY: CompileOptions = CompileOptions {
    execution: Submission::Replay,
    fusion: FusionPolicy::MoeHangGuard,
    limits: poot_graph_plan::CompileLimits::STANDARD,
};

fn staged(g: &Graph, target: Target) -> StagedProgram<ValidationOutputs> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &UNFUSED_REPLAY,
    )
    .expect("the fixture compiles unfused")
}

fn oracle(a: &[f32], b: &[f32], c: f32) -> f32 {
    let s1: f32 = a.iter().zip(b).map(|(&x, &y)| x + y).sum();
    let s2 = c + c;
    s1 + s2
}

fn require_wgpu() -> Option<WgpuDevice> {
    WgpuDevice::new().ok()
}

/// SC-001: peak arena bytes below the naive per-value sum, output equal to the CPU oracle.
///
/// Mutation (card 547b, applied by hand, reverted, never left in the tree): give every
/// locally-computed value its own arena slot (skip `BufferPlan`'s free-list reuse entirely, i.e.
/// `out_loc`'s slot lookup returning a fresh `ArenaSlotId` every time instead of consulting
/// `BufferPlan::slot`). `Program::buffer_plan().arena_bytes()` then equals the naive sum exactly
/// (every value gets its own slot), turning the `<` assertion below red.
#[test]
fn disjoint_lifetimes_reuse_one_slot_and_match_the_oracle() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = device.target();
    let (g, a, b, c) = fixture();
    let program = poot_graph_plan::compile(&g, &target, &UNFUSED_REPLAY)
        .expect("the fixture compiles for wgpu");

    // The naive "one buffer per locally-computed value" sum this card's arena coloring replaces:
    // v1 [K], s1 [1], v2 [1], s2 [1], out [1] (4 bytes per f32 element). `out` always gets its own
    // dedicated slot (an export, module doc), so the only reuse this fixture can show is v2 folding
    // into v1's freed [K] slot - one fewer element-sized slot than the naive count.
    let naive_sum_bytes = (K + 1 + 1 + 1 + 1) * 4;
    let arena_bytes = program.buffer_plan().arena_bytes();
    assert!(
        arena_bytes < naive_sum_bytes,
        "v1's freed [K] slot must be reused by v2 (arena_bytes={arena_bytes}, naive sum={naive_sum_bytes})"
    );

    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let exe = exec
        .load_weights(store(&a, &b, c), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &staged(&g, target)).unwrap();
    let mut out = exec
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let got = bytemuck_f32(&out.read().unwrap());
    assert_eq!(got.len(), 1);
    assert_close_rel(&got, &[oracle(&a, &b, c)], 1e-5);
}

/// SC-002 (wgpu, arena-reuse-is-transparent half): `s2`'s reduce still returns exactly `v2`'s one
/// logical element, correct to the oracle, even though `v2` is read through an arena slot sized for
/// `v1`'s earlier, larger [K] value - the reuse itself never corrupts a real dispatch.
///
/// This is not the decoupled-length proof (that one is `poot_executor`'s own
/// `tests/buffer_plan_arena.rs`, model-free and backend-neutral): `pootc`'s STABLE-MIR-compiled
/// `Add`/`Reduce` kernels here have a statically known shape per equation and bake their own bound
/// into the generated SPIR-V, so they never read the per-dispatch length buffer at
/// all - reverting `build_dispatch_objects` to `DeviceBuffer::elem_count()` (recorded once, by hand,
/// in this card's own history) left this test green, because nothing in these two kernels' bodies
/// ever consults that buffer's contents. The length buffer's runtime value governs a shape-generic
/// kernel body (an imported kernel such as batched decode GEMV, `Len(param)` read at kernel-build
/// time per `poot-kernelgen`'s own doc), which this fixture does not exercise -
/// [`len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity`] below is that hardware
/// test, via a small generated probe body instead of an imported GEMV (review F1).
#[test]
fn a_short_value_reusing_a_larger_slot_publishes_its_own_length_not_the_slots() {
    let Some(device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let target = device.target();
    let (g, a, b, c) = fixture();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let exe = exec
        .load_weights(store(&a, &b, c), poot_executor::WeightSource::ConstNames)
        .unwrap();
    let entry = exec.add_entry(exe, &staged(&g, target)).unwrap();
    let mut out = exec
        .step(exe, entry, &StepInputs::new(), &mut NoSync)
        .unwrap();
    let got = bytemuck_f32(&out.read().unwrap());
    assert_close_rel(&got, &[oracle(&a, &b, c)], 1e-5);
}

/// SC-002 (review F1, decoupled-length proof on real wgpu hardware): `poot_test_util::kernel_fixtures::
/// len_probe_kernel`'s body reads `Len(len_src)` at kernel-build time, so its compiled SPIR-V has no
/// shape baked in - unlike the `Add`/`Reduce` kernels above. `len_src` is allocated with
/// `SLOT_CAPACITY=64` device elements (standing in for an arena slot reused from a larger occupant,
/// exactly the situation [`a_short_value_reusing_a_larger_slot_publishes_its_own_length_not_the_slots`]
/// exercises end to end), but the dispatch publishes `Arg::elems = REAL_LEN = 4` for it - the value's
/// own logical length, never the slot's capacity. Drives `WgpuDevice` directly through the identical
/// begin/dispatch/finish/replay/synchronize sequence `Engine::step` uses internally (same precedent as
/// `executor_contract.rs`'s SC-015 fault probe), since no existing graph op's kernelgen body reads a
/// meaningful dynamic `Len()` at this fixture's tiny, fixed shapes (the module doc above).
///
/// Mutation (card 547b, applied by hand, reverted, never left in the tree): in
/// `crates/poot-runtime/src/context/dispatch.rs`'s `build_dispatch_objects`, build `lengths` from
/// `ins.iter().map(|b| b.elem_count() as u32)` instead of `ins_lens` (the pre-547b placeholder
/// `WgpuDevice::dispatch` itself fed to `build_dispatch_objects` before this card, S46-8). Observed:
/// the probe reports `64.0` (the buffer's full capacity) instead of `4.0`, and the row goes red;
/// reverting to `ins_lens` restored `4.0`.
#[test]
fn len_probe_reads_the_operand_own_length_never_a_larger_slots_capacity() {
    let Some(mut device) = require_wgpu() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    const REAL_LEN: usize = 4;
    const SLOT_CAPACITY: usize = 64;

    let body = poot_test_util::kernel_fixtures::len_probe_kernel();
    let dir = std::env::temp_dir().join("poot-gpu-sc002-len-probe");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path =
        poot_codegen::artifact_path(&dir, "sc002_len_probe", poot_codegen::Target::SpirvVulkan);
    poot_codegen::compile(&body, poot_codegen::Target::SpirvVulkan, &out_path)
        .expect("len_probe must compile to SpirV");
    let spv = std::fs::read(&out_path).unwrap();
    let compiled = poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, spv);
    let kernel = device
        .load_kernel("sc002_len_probe", compiled)
        .expect("load the len_probe kernel");

    let len_src = device
        .allocate(BufferRole::Input, BufferStorage::f32(), SLOT_CAPACITY)
        .unwrap();
    device
        .write(&len_src, &bytemuck_bytes(&vec![0.0f32; SLOT_CAPACITY]))
        .unwrap();
    let out_buf = device
        .allocate(BufferRole::Output, BufferStorage::f32(), 1)
        .unwrap();

    device.begin(Submission::Replay).unwrap();
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[Arg {
                buffer: &len_src,
                elems: REAL_LEN as u32,
            }],
            output: Arg {
                buffer: &out_buf,
                elems: 1,
            },
            threads: [1, 1, 1],
            workgroup: [1, 1, 1],
            work: 1,
        })
        .unwrap();
    let recording = device
        .finish()
        .unwrap()
        .expect("Submission::Replay always returns a recording");
    device
        .replay(&recording)
        .expect("replay only re-encodes metadata; it never runs the dispatch itself");
    device.synchronize().unwrap();

    let mut out_bytes = [0u8; 4];
    device.read(&out_buf, &mut out_bytes).unwrap();
    let probed_len = f32::from_le_bytes(out_bytes);
    assert_eq!(
        probed_len, REAL_LEN as f32,
        "Len(len_src) inside the kernel must equal the dispatch's own Arg::elems ({REAL_LEN}), never \
         len_src's SLOT_CAPACITY={SLOT_CAPACITY} device buffer capacity"
    );
}

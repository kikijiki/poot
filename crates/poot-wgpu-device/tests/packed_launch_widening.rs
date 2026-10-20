//! Card 656: one plan key, two kernels. The planner's launch postamble widens a
//! one-thread-per-element body's workgroup from 64 to 256 once `ceil(out_numel / 64)` exceeds the
//! 65535 grid cap, and keeps the plan key; the packed keys (`packed_materialize:{format}`,
//! `packed_row_gather:{format}`) are shape-free by design. The executor's kernel tier and the
//! runtime's pipeline tier used to hit on that key alone, so after a small shape compiled the narrow
//! module, a large shape dispatched it over a grid sized for 256 lanes: exactly a quarter of the
//! output was written and the rest read back as zero. That was the "real-size Materialize garbage"
//! the eager 1.5B Q4_K decode produced (8c36446de): its small K/V projections compiled first. The
//! indexed packed contraction rides the same rows: its shapes ride in metadata, so one body must serve
//! every table size.
//!
//! Each row runs the small shape, then the large one, in one executor, and compares every element of
//! the large output with the oracle decoder. The order matters: the large shape alone was correct.

use poot_graph_ir::test_support::const_weights::ConstWeights;
use std::sync::Arc;

use poot_executor::{Device, Executor, NoSync, StepInputs};
use poot_graph_ir::{Builder, Graph, Slot, TensorType};
use poot_graph_plan::{
    CompileOptions, Target, multi_device::topology::DeviceId, staged::DevicePlacement,
    staged::ExpertPlacement, staged::Partition, staged::StagedProgram, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::PackedPayload;
use poot_quant::format::WeightFormat;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_tensor::{DType, HostView};
use poot_wgpu_device::device::WgpuDevice;

/// Compile `g` for `target`, single-stage, single-device.
fn staged(g: &Graph, target: Target) -> StagedProgram {
    let g = g.clone();
    compile_staged(
        &g,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the packed graph")
}

fn f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The first element of `got` that differs from `want` bit for bit (zero sign canonicalized), with
/// the count of differing elements.
fn mismatches(got: &[f32], want: &[f32]) -> Option<(usize, usize, f32, f32)> {
    assert_eq!(got.len(), want.len());
    let bad: Vec<usize> = (0..got.len())
        .filter(|&i| (got[i] + 0.0).to_bits() != (want[i] + 0.0).to_bits())
        .collect();
    bad.first().map(|&i| (bad.len(), i, got[i], want[i]))
}

/// The indexed packed contraction (`packed_indexed_linear`: one `PackedContraction` over a stacked table, no
/// `Materialize`) at `[out, k]` Q4_K experts, run through the
/// executor contract (`exec` is reused across calls so its kernel/pipeline cache persists exactly as
/// the deleted `GpuExecutor`'s own `self.compiled` cache did); returns the output and the oracle's. Row `m` reads
/// expert `m % 2` at column `(m * 37 + 11) % k` through a one-hot activation, so every output is one
/// decoded weight value.
fn run_moe(exec: &mut dyn Executor, target: Target, out: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    const EXPERTS: usize = 2;
    const ROWS: usize = 4;
    let owners: Vec<Arc<PackedPayload>> = (0..EXPERTS)
        .map(|e| {
            Arc::new(poot_quant::test_support::packed::random_payload(
                WeightFormat::Q4_K,
                [out, k],
                e as u64 + 656,
            ))
        })
        .collect();
    let table = poot_graph_ir::test_support::packed_graphs::PackedExpertStack {
        stack: poot_graph_ir::test_support::packed_graphs::TEST_STACK,
        descriptor: owners[0].weight(),
        experts: EXPERTS,
    };
    let column = |m: usize| (m * 37 + 11) % k;
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![ROWS, k]))
        .unwrap();
    let ids = b
        .slot_named_ranged(
            Slot::Activation,
            "moe-ids",
            TensorType::new(vec![ROWS], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let y = poot_graph_ir::test_support::packed_graphs::packed_indexed_linear(&b, x, ids, &table)
        .unwrap();
    let g = b.finish(y).unwrap();

    let mut store = WeightStore::builder();
    for (ordinal, owner) in owners.iter().enumerate() {
        store
            .insert(
                format!(
                    "{}.{ordinal}",
                    poot_graph_ir::test_support::packed_graphs::TEST_STACK.const_name()
                ),
                WeightEntry::Packed(Arc::clone(owner)),
            )
            .unwrap();
    }
    let store = store.build();
    let (bound, map) = ConstWeights::bind(&g, &store);
    let program = staged(&bound, target);
    let compiled_graph = program.stages().next().expect("single stage").2.graph();
    assert!(
        matches!(
            compiled_graph.eqns(),
            [poot_graph_ir::Eqn {
                op: poot_graph_ir::OpKind::PackedContraction {
                    rows: poot_graph_ir::ContractionRows::Indexed { .. },
                    ..
                },
                ..
            }]
        ),
        "[{out}, {k}]: one indexed packed contraction, got {:?}",
        compiled_graph.eqns()
    );

    let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
    let entry = exec.add_entry(exe, &program).unwrap();

    let mut one_hot = vec![0.0f32; ROWS * k];
    for m in 0..ROWS {
        one_hot[m * k + column(m)] = 1.0;
    }
    let selected: Vec<i32> = (0..ROWS).map(|m| (m % EXPERTS) as i32).collect();
    let x_shape = [ROWS, k];
    let ids_shape = [ROWS];
    let mut inputs = StepInputs::new();
    inputs.push(
        g.meta(x.id()).slot_key().unwrap().clone(),
        &x_shape,
        HostView::new(DType::F32, ROWS * k, bytemuck::cast_slice(&one_hot)).unwrap(),
    );
    inputs.push(
        g.meta(ids.id()).slot_key().unwrap().clone(),
        &ids_shape,
        HostView::new(DType::I32, ROWS, bytemuck::cast_slice(&selected)).unwrap(),
    );
    let got = f32_bytes(
        &exec
            .step(exe, entry, &inputs, &mut NoSync)
            .unwrap_or_else(|error| panic!("[{out}, {k}]: {error}"))
            .read()
            .unwrap_or_else(|error| panic!("[{out}, {k}]: {error}")),
    );
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();

    let mut want = Vec::with_capacity(ROWS * out);
    let mut row = vec![0.0f32; k];
    for m in 0..ROWS {
        for o in 0..out {
            owners[m % EXPERTS].decode_row(o, &mut row).unwrap();
            want.push(row[column(m)]);
        }
    }
    (got, want)
}

/// SC-004 (indexed contraction): the stacked-expert contraction at `[8, 256]` experts, then at
/// `[2816, 1536]` (a different Gemv launch over a larger table), is bit-exact at both sizes in one executor.
#[test]
fn a_wide_indexed_contraction_after_a_narrow_one_is_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    for [out, k] in [[8, 256], [2816, 1536]] {
        let (got, want) = run_moe(&mut *exec, target, out, k);
        if let Some((bad, index, device, oracle)) = mismatches(&got, &want) {
            panic!(
                "[{out}, {k}] experts: {bad}/{} outputs differ; first at {index}: device {device} vs \
                 oracle {oracle}",
                want.len()
            );
        }
    }
}

/// SC-004 (row gather): a compiled Q4_K token embedding (`PackedRowGather`) over a `[64, 1536]`
/// table for 8 ids, then 4096 ids (6.3M elements, past the 64-lane grid cap), is bit-exact at both
/// lengths in one executor: a long prompt after a short one (the production order).
#[test]
fn a_widened_row_gather_after_a_narrow_one_is_bit_exact() {
    let Some(device) =
        poot_runtime_common::device_skip::open_or_skip(DeviceBackend::Wgpu, WgpuDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut exec: Box<dyn Executor> = Box::new(poot_executor::Engine::new(device));
    const VOCAB: usize = 64;
    const K: usize = 1536;
    let owner = Arc::new(poot_quant::test_support::packed::random_payload(
        WeightFormat::Q4_K,
        [VOCAB, K],
        656,
    ));
    for n in [8, 4096] {
        let b = Builder::for_tests();
        let token = b
            .slot(Slot::Token, TensorType::new(vec![n], DType::I32))
            .unwrap();
        let rows = poot_graph_ir::test_support::packed_graphs::packed_embedding(
            &b,
            token,
            "model.embed_tokens",
            owner.weight(),
        )
        .unwrap();
        let g = b.finish(rows).unwrap();

        let mut store = WeightStore::builder();
        store
            .insert(
                "model.embed_tokens",
                WeightEntry::Packed(Arc::clone(&owner)),
            )
            .unwrap();
        let store = store.build();
        let (bound, map) = ConstWeights::bind(&g, &store);
        let program = staged(&bound, target);
        let compiled_graph = program.stages().next().expect("single stage").2.graph();
        assert!(
            compiled_graph
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedRowGather { .. })),
            "{n} ids: compile claims the embedding"
        );

        let exe = exec.load_weights(Arc::new(store), Arc::new(map)).unwrap();
        let entry = exec.add_entry(exe, &program).unwrap();

        let ids: Vec<usize> = (0..n).map(|r| (r * 13 + 5) % VOCAB).collect();
        let id_values: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let token_shape = [n];
        let mut inputs = StepInputs::new();
        inputs.push(
            g.meta(token.id()).slot_key().unwrap().clone(),
            &token_shape,
            HostView::new(DType::I32, n, bytemuck::cast_slice(&id_values)).unwrap(),
        );
        let got = f32_bytes(
            &exec
                .step(exe, entry, &inputs, &mut NoSync)
                .unwrap_or_else(|error| panic!("{n} ids: {error}"))
                .read()
                .unwrap_or_else(|error| panic!("{n} ids: {error}")),
        );
        exec.remove_entry(exe, entry).unwrap();
        exec.unload(exe).unwrap();

        let mut want = vec![0.0f32; n * K];
        for (r, &id) in ids.iter().enumerate() {
            owner.decode_row(id, &mut want[r * K..(r + 1) * K]).unwrap();
        }
        if let Some((bad, index, device, oracle)) = mismatches(&got, &want) {
            panic!(
                "{n} ids: {bad}/{} elements differ; first at row {} col {}: device {device} vs \
                 oracle {oracle}",
                want.len(),
                index / K,
                index % K
            );
        }
    }
}

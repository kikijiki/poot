//! Card 656, the PTX twin of `poot-gpu/tests/packed_launch_widening.rs`: a
//! regression guard for the body-identity kernel tier. The planner's launch postamble widens a
//! one-thread-per-element body's workgroup from 64 to 256 once `ceil(out_numel / 64)` exceeds 65535
//! and keeps the plan key, so one shape-free key (`packed_materialize:{format}`,
//! `packed_row_gather:{format}`) names two bodies. On wgpu a key-only hit dispatched the narrow
//! module over the wide grid and wrote a quarter of the output. PTX is not exposed to that failure:
//! an NVPTX module declares no `.reqntid`, each launch takes its block size from the requested
//! body, and the kernel indexes by the runtime block size, so a key-only hit still covers the grid
//! (PTX batch 10 row 4: the key-only mutation of `cached_kernel` stays green here by design).
//!
//! Each row runs the small shape, then the large one, in one executor, and compares every element of
//! the large output with the oracle decoder, so a PTX tier or launch change that loses coverage
//! across shapes turns it red.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_executor::Device;
use poot_graph_ir::{Builder, Graph, PackedSourceName, Slot, TensorType, ValueId};
use poot_graph_plan::{CompileOptions, Submission, Target, compile};
use poot_ptx_gpu::PtxDevice;
use poot_quant::format::WeightFormat;
use poot_quant::{PackedComponentRef, PackedPayload, PackedStack, PackedStackComponentRef};
use poot_runtime_common::DeviceBackend;
use poot_target::Backend;
use poot_tensor::{DType, HostTensor};

mod common;

/// Not stepped through the executor contract: used only to inspect the plan shape the assertions
/// below check (`compile`'s own `Program`, not `common::run_resident`'s internal staging, which never
/// hands its compiled program back to the caller).
fn compiled(gpu: &PtxDevice, graph: &Graph) -> Graph {
    compile(
        graph,
        &Target {
            backend: Backend::Nvptx,
            caps: Device::target(gpu).caps,
        },
        &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("compile the packed graph")
    .graph()
    .clone()
}

/// `graph`'s packed carriers, each bound as its role's component of the owner `owner_of` names for
/// the carrier's linear id.
fn carriers(
    graph: &Graph,
    owner_of: impl Fn(&str) -> Arc<PackedPayload>,
) -> HashMap<ValueId, Value> {
    graph
        .inputs
        .iter()
        .filter_map(|&id| {
            let name = PackedSourceName::parse(graph.meta(id).name.as_deref()?)?;
            let owner = owner_of(name.linear_id());
            Some((id, PackedComponentRef::new(owner, name.role()).into()))
        })
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

/// The indexed packed contraction (`packed_indexed_linear`: one `PackedContraction` over a stacked
/// table, no `Materialize`) at `[out, k]` Q4_K experts, run through
/// `run_resident`; returns the output and the oracle's. Row `m` reads expert `m % 2` at column
/// `(m * 37 + 11) % k` through a one-hot activation, so every output is one decoded weight value.
fn run_moe(gpu: &mut PtxDevice, out: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    const EXPERTS: usize = 2;
    const ROWS: usize = 4;
    let owners: Vec<Arc<PackedPayload>> = (0..EXPERTS)
        .map(|e| {
            Arc::new(poot_test_util::packed::random_payload(
                WeightFormat::Q4_K,
                [out, k],
                e as u64 + 656,
            ))
        })
        .collect();
    let table = poot_graph_ir::ops::PackedExpertStack {
        stack_id: "experts".to_string(),
        descriptor: owners[0].weight(),
        experts: EXPERTS,
    };
    let stack = Arc::new(PackedStack::try_new(owners.clone()).unwrap());
    let column = |m: usize| (m * 37 + 11) % k;
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "moe-x", TensorType::f32(vec![ROWS, k]));
    let ids = b.slot_named(
        Slot::Activation,
        "moe-ids",
        TensorType::new(vec![ROWS], DType::I32),
    );
    let y = poot_graph_ir::ops::packed_indexed_linear(&b, x, ids, &table).unwrap();
    let graph = compiled(gpu, &b.finish(y));
    assert!(
        matches!(
            graph.eqns.as_slice(),
            [poot_graph_ir::Eqn {
                op: poot_graph_ir::OpKind::PackedContraction {
                    rows: poot_graph_ir::ContractionRows::Indexed { .. },
                    ..
                },
                ..
            }]
        ),
        "[{out}, {k}]: one indexed packed contraction, got {:?}",
        graph.eqns
    );
    let mut inputs: HashMap<ValueId, Value> = graph
        .inputs
        .iter()
        .filter_map(|&id| {
            let name = PackedSourceName::parse(graph.meta(id).name.as_deref()?)?;
            Some((
                id,
                PackedStackComponentRef::new(Arc::clone(&stack), name.role()).into(),
            ))
        })
        .collect();
    let mut one_hot = vec![0.0f32; ROWS * k];
    for m in 0..ROWS {
        one_hot[m * k + column(m)] = 1.0;
    }
    inputs.insert(x.id(), HostTensor::f32(vec![ROWS, k], one_hot).into());
    let selected: Vec<i32> = (0..ROWS).map(|m| (m % EXPERTS) as i32).collect();
    inputs.insert(ids.id(), HostTensor::i32(vec![ROWS], selected).into());
    let got = common::run_resident(gpu, &graph, &inputs)
        .as_f32()
        .unwrap()
        .to_vec();
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
    let Some(mut gpu) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    for [out, k] in [[8, 256], [2816, 1536]] {
        let (got, want) = run_moe(&mut gpu, out, k);
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
    let Some(mut gpu) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };
    const VOCAB: usize = 64;
    const K: usize = 1536;
    let owner = Arc::new(poot_test_util::packed::random_payload(
        WeightFormat::Q4_K,
        [VOCAB, K],
        656,
    ));
    for n in [8, 4096] {
        let b = Builder::new();
        let token = b.slot(Slot::Token, TensorType::new(vec![n], DType::I32));
        let rows =
            poot_graph_ir::ops::packed_embedding(&b, token, "model.embed_tokens", owner.weight())
                .unwrap();
        let graph = compiled(&gpu, &b.finish(rows));
        assert!(
            graph
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::PackedRowGather { .. })),
            "{n} ids: compile claims the embedding"
        );
        let mut inputs = carriers(&graph, |_| Arc::clone(&owner));
        let ids: Vec<usize> = (0..n).map(|r| (r * 13 + 5) % VOCAB).collect();
        inputs.insert(
            token.id(),
            HostTensor::i32(vec![n], ids.iter().map(|&id| id as i32).collect()).into(),
        );
        let got = common::run_resident(&mut gpu, &graph, &inputs)
            .as_f32()
            .unwrap()
            .to_vec();
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

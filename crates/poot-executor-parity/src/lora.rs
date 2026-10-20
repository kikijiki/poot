//! The batched-LoRA device row, shared by the wgpu, ROCm and PTX twins (POOT-579).
//!
//! Each of the seven targetable projections is a hand-built graph - `x @ W^T + bias` over the stored
//! `[out, in]` weight - compiled with a [`LoraPlan`], so the LoRA compile pass inserts the correction and
//! the device runs the result through the executor contract (the `IndexedMatMul`/`Gather`/`MatMul`
//! kernels of the target). Two adapters sit on rows 0 and 1 and row 2 runs no adapter; every row must
//! equal the host's `x @ (W + scale * A B)^T + bias` computed without the op under test, at one token
//! and at two tokens per row (the per-token index expansion).

use std::sync::Arc;

use poot_executor::{Executor, NoSync, StepInputs};
use poot_graph_ir::{Builder, Slot, TensorType, WeightRef};
use poot_graph_plan::{
    CompileLimits, CompileOptions, LoraPlan, Target, multi_device::topology::DeviceId,
    staged::DevicePlacement, staged::ExpertPlacement, staged::Partition, staged::TargetSet,
    staged::compile_staged,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_tensor::{DType, HostTensor};
use poot_weights::{AttnRole, LoraRole, LoraTarget, WeightId, WeightMap, WeightRole, WeightView};

const BATCH: usize = 3;
const RANK: usize = 2;
const ADAPTERS: usize = 2;

/// Deterministic fill in [-1, 1) from `seed`.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The `(projection, in, out)` of each targetable projection of the fixture: `o_proj` reads the
/// attention output and `down_proj` the intermediate, so both are asymmetric.
fn projections() -> [(LoraTarget, usize, usize); 7] {
    let (hidden, inter, q_dim, kv_dim) = (8, 16, 8, 4);
    [
        (LoraTarget::AttnQ, hidden, q_dim),
        (LoraTarget::AttnK, hidden, kv_dim),
        (LoraTarget::AttnV, hidden, kv_dim),
        (LoraTarget::AttnO, q_dim, hidden),
        (LoraTarget::FfnGate, hidden, inter),
        (LoraTarget::FfnUp, hidden, inter),
        (LoraTarget::FfnDown, inter, hidden),
    ]
}

fn dense(shape: Vec<usize>, values: &[f32]) -> WeightEntry {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    WeightEntry::Dense(DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap())
}

/// Every row of `tokens` tokens per row against the adapter it names, on `exec`: panics with the
/// projection, row and element on the first logit outside `rel_tol` of the host reference, or when the
/// three rows do not differ.
pub fn assert_batched_lora_matches_merged_weights(
    exec: &mut dyn Executor,
    target: Target,
    rel_tol: f32,
) {
    // Row 0 runs adapter 1, row 1 adapter 2, row 2 no adapter.
    let rows_adapter = [1usize, 2, 0];
    let scales = [0.0f32, 2.0, 3.0];
    for tokens in [1usize, 2] {
        let mut all_rows: Vec<Vec<f32>> = vec![Vec::new(); BATCH];
        for (lora_target, in_dim, out_dim) in projections() {
            let rows = run_projection(
                exec,
                target,
                lora_target,
                (in_dim, out_dim),
                tokens,
                &rows_adapter,
                &scales,
                rel_tol,
            );
            for (all, row) in all_rows.iter_mut().zip(rows) {
                all.extend(row);
            }
        }
        let (d01, d02, d12) = (
            poot_test_util::max_abs_error(&all_rows[0], &all_rows[1]),
            poot_test_util::max_abs_error(&all_rows[0], &all_rows[2]),
            poot_test_util::max_abs_error(&all_rows[1], &all_rows[2]),
        );
        assert!(
            d01 > 1e-3 && d02 > 1e-3 && d12 > 1e-3,
            "tokens {tokens}: the rows (adapter 1, adapter 2, no adapter) must differ: d01={d01} d02={d02} d12={d12}"
        );
    }
}

#[expect(clippy::too_many_arguments)]
fn run_projection(
    exec: &mut dyn Executor,
    target: Target,
    lora_target: LoraTarget,
    (in_dim, out_dim): (usize, usize),
    tokens: usize,
    rows_adapter: &[usize; BATCH],
    scales: &[f32; ADAPTERS + 1],
    rel_tol: f32,
) -> Vec<Vec<f32>> {
    let seed = |tag: &str| -> u64 {
        format!("{lora_target:?}.{tag}")
            .bytes()
            .fold(1469598103934665603u64, |acc, c| {
                (acc ^ u64::from(c)).wrapping_mul(1099511628211)
            })
    };
    // Stored `[out, in]` base weight, bias, and the stacked factors (entry 0 all zero).
    let base = fill(out_dim * in_dim, seed("w"));
    let bias = fill(out_dim, seed("bias"));
    let mut a = vec![0.0f32; (ADAPTERS + 1) * in_dim * RANK];
    let mut b = vec![0.0f32; (ADAPTERS + 1) * RANK * out_dim];
    for adapter in 1..=ADAPTERS {
        let s = seed(&format!("adapter{adapter}"));
        a[adapter * in_dim * RANK..(adapter + 1) * in_dim * RANK]
            .copy_from_slice(&fill(in_dim * RANK, s));
        b[adapter * RANK * out_dim..(adapter + 1) * RANK * out_dim]
            .copy_from_slice(&fill(RANK * out_dim, s + 1));
    }
    let x = fill(BATCH * tokens * in_dim, seed("x"));

    // The host reference: row r's tokens through `x @ (W + scale A B)^T + bias` for its adapter.
    let mut reference: Vec<Vec<f32>> = vec![Vec::new(); BATCH];
    for (row, reference_row) in reference.iter_mut().enumerate() {
        let adapter = rows_adapter[row];
        let mut merged = base.clone();
        for i in 0..in_dim {
            for o in 0..out_dim {
                let delta: f32 = (0..RANK)
                    .map(|k| {
                        a[adapter * in_dim * RANK + i * RANK + k]
                            * b[adapter * RANK * out_dim + k * out_dim + o]
                    })
                    .sum();
                merged[o * in_dim + i] += scales[adapter] * delta;
            }
        }
        for token in 0..tokens {
            let xs = &x[(row * tokens + token) * in_dim..(row * tokens + token + 1) * in_dim];
            for o in 0..out_dim {
                let dot: f32 = (0..in_dim).map(|i| xs[i] * merged[o * in_dim + i]).sum();
                reference_row.push(dot + bias[o]);
            }
        }
    }

    // The graph: `x [BATCH, tokens, in] @ W^T + bias`, the LoRA pass adds the correction.
    let weight_id = WeightId::layer(0, lora_target.weight_role());
    let bias_id = WeightId::layer(0, WeightRole::Attn(AttnRole::QBias));
    let graph = {
        let builder = Builder::for_tests();
        let xs = builder
            .slot_named(
                Slot::Activation,
                "x",
                TensorType::f32(vec![BATCH, tokens, in_dim]),
            )
            .unwrap();
        let w = builder
            .weight(
                WeightRef::whole(weight_id),
                TensorType::f32(vec![out_dim, in_dim]),
            )
            .unwrap();
        let bias_const = builder
            .weight(WeightRef::whole(bias_id), TensorType::f32(vec![out_dim]))
            .unwrap();
        let wt = builder.transpose(w, vec![1, 0]).unwrap();
        let y = builder.matmul(xs, wt).unwrap();
        let shape = builder.aval(y).unwrap().shape;
        let y = builder
            .binary(
                poot_graph_ir::BinOp::Add,
                y,
                builder.broadcast(bias_const, shape).unwrap(),
            )
            .unwrap();
        builder.finish(y).unwrap()
    };

    let ids = [
        (weight_id, vec![out_dim, in_dim], &base[..]),
        (bias_id, vec![out_dim], &bias[..]),
        (
            WeightId::layer(0, WeightRole::Lora(LoraRole::A(lora_target))),
            vec![ADAPTERS + 1, in_dim, RANK],
            &a[..],
        ),
        (
            WeightId::layer(0, WeightRole::Lora(LoraRole::B(lora_target))),
            vec![ADAPTERS + 1, RANK, out_dim],
            &b[..],
        ),
        (
            WeightId::model(WeightRole::Lora(LoraRole::Scale)),
            vec![ADAPTERS + 1],
            &scales[..],
        ),
    ];
    let mut builder = WeightStore::builder();
    for (id, shape, values) in &ids {
        builder
            .insert(id.const_name(), dense(shape.clone(), values))
            .unwrap();
    }
    let store = builder.build();
    let mut map = WeightMap::builder(&store);
    for (id, _, _) in &ids {
        map.map(*id, WeightView::Stored(WeightKey::from(id.const_name())))
            .unwrap();
    }
    let map = map.build();

    let plan = LoraPlan {
        rows: std::num::NonZeroUsize::new(BATCH).unwrap(),
        adapters: std::num::NonZeroUsize::new(ADAPTERS).unwrap(),
        rank: std::num::NonZeroUsize::new(RANK).unwrap(),
        factor_dtype: poot_tensor::DType::F32,
        projections: vec![lora_target],
    };
    let program = compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(CompileLimits::STANDARD).with_lora(plan),
    )
    .unwrap_or_else(|e| panic!("{lora_target:?}: compile_staged: {e}"));
    let exe = exec
        .load_weights(Arc::new(store), Arc::new(map))
        .unwrap_or_else(|e| panic!("{lora_target:?}: load_weights: {e}"));
    let entry = exec
        .add_entry(exe, &program)
        .unwrap_or_else(|e| panic!("{lora_target:?}: add_entry: {e}"));

    let activation = HostTensor::f32(vec![BATCH, tokens, in_dim], x);
    let lora_idx = HostTensor::i32(
        vec![BATCH],
        rows_adapter.iter().map(|&i| i as i32).collect(),
    );
    let x_key = poot_graph_ir::SlotKey::new(Slot::Activation, Some("x"));
    let idx_key = poot_graph_ir::SlotKey::new(Slot::LoraIdx, None);
    let mut inputs = StepInputs::new();
    inputs.push(x_key, activation.shape(), activation.view());
    inputs.push(idx_key, lora_idx.shape(), lora_idx.view());
    let output = exec
        .step(exe, entry, &inputs, &mut NoSync)
        .unwrap_or_else(|e| panic!("{lora_target:?}: step: {e}"))
        .to_host()
        .unwrap_or_else(|e| panic!("{lora_target:?}: readback: {e}"));
    exec.remove_entry(exe, entry).unwrap();
    exec.unload(exe).unwrap();

    let got = output.as_f32().expect("f32 output");
    let per_row = tokens * out_dim;
    assert_eq!(got.len(), BATCH * per_row, "{lora_target:?}: output length");
    let mut rows = Vec::with_capacity(BATCH);
    for (row, want) in reference.iter().enumerate() {
        let got_row = &got[row * per_row..(row + 1) * per_row];
        for (i, (g, w)) in got_row.iter().zip(want).enumerate() {
            let tol = rel_tol * w.abs().max(1e-3);
            assert!(
                (g - w).abs() <= tol,
                "{lora_target:?} tokens {tokens} row {row} element {i}: device {g} vs merged-weight reference {w}"
            );
        }
        rows.push(got_row.to_vec());
    }
    rows
}

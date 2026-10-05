//! Batched-LoRA on ROCm/AMD (spec 248 phase 2, epic 129 B8), covering the shape of all seven LoRA
//! projections (`q/k/v/o_proj`, `gate/up/down_proj`). Same test as the wgpu and PTX twins
//! (`crates/poot-gpu/tests/graph/lora.rs`, `crates/poot-ptx-gpu/tests/batched_lora_ptx.rs`): two real
//! adapters registered into a `LoraAdapterPool`, stacked via `stack_for_module`, dispatched through
//! `ops::lora_linear_batched` with a 3-row batch (`adapter_a`, `adapter_b`, `NO_ADAPTER`) as one
//! projection graph per module shape (real `IndexedMatMul`/`Gather`/`MatMul` kernels through the
//! executor contract). Each row must match its own per-row merged-weight reference (computed on the CPU
//! without `lora_linear`/`lora_linear_batched`) within `1e-3` relative tolerance (the PTX bar; wgpu
//! uses `1e-2`), and the three rows must differ from each other. LoRA is not part of `Model::trace`:
//! the op is exercised on a single projection, the wiring into a model is a graph transform's concern.
//!
//! Skips cleanly with no AMD GPU/HSA runtime.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;
use std::sync::Arc;

use poot_executor::{Device, Engine, Executor, NoSync, StepInputs};
use poot_graph_ir::ops::lora_linear_batched;
use poot_graph_ir::{Builder, Graph, Slot, SlotKey, Storage, TensorType};
use poot_load::lora::{
    LoraAdapter, LoraAdapterConfig, LoraAdapterPool, LoraStackedModule, LoraWeight,
};
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_rocm_gpu::device::RocmDevice;
use poot_tensor::DType;

/// Deterministic pseudo-random fill in [-1, 1) with no rng dependency, a local copy as in the
/// wgpu/PTX/CPU-oracle twins (`poot-eval`'s helper is `pub(super)`).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// `a [m,k] @ b [k,n] -> [m,n]`, row-major; hand-rolled reference that never calls the op under test.
fn matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn add_scaled(base: &[f32], correction: &[f32], scale: f32) -> Vec<f32> {
    base.iter()
        .zip(correction)
        .map(|(b, c)| b + scale * c)
        .collect()
}

use poot_test_util::max_abs_error;

fn raw(shape: Vec<usize>, data: Vec<f32>) -> poot_tensor::HostTensor {
    poot_tensor::HostTensor::f32(shape, data)
}

/// A synthetic adapter targeting all seven projections at layer 0 (same as the wgpu/PTX/CPU-oracle
/// twins). `proj_specs` carries each projection's `(module_suffix, in_dim, out_dim)`: `o_proj` projects
/// from the concatenated attention output (`q_dim`), and `down_proj` from `inter` to `hidden` (the
/// reverse of `gate_proj`/`up_proj`).
fn adapter(r: usize, proj_specs: &[(&str, usize, usize)], alpha: f32, seed: u64) -> LoraAdapter {
    let mut weights = HashMap::new();
    for (i, &(name, in_dim, out_dim)) in proj_specs.iter().enumerate() {
        let s = seed + (i as u64) * 10; // distinct seed per projection, mirrors the old hand-picked offsets
        weights.insert(
            format!("model.layers.0.{name}"),
            LoraWeight {
                a: raw(vec![in_dim, r], fill(in_dim * r, s)),
                b: raw(vec![r, out_dim], fill(r * out_dim, s + 1)),
            },
        );
    }
    LoraAdapter {
        config: LoraAdapterConfig {
            r,
            lora_alpha: alpha,
            target_modules: proj_specs
                .iter()
                .map(|&(name, _, _)| name.rsplit('.').next().unwrap().to_string())
                .collect(),
            use_rslora: false,
        },
        weights,
    }
}

type SlotValue = (SlotKey, Vec<usize>, DType, Vec<u8>);

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn step_inputs_from(values: &[SlotValue]) -> StepInputs<'_> {
    let mut inputs = StepInputs::new();
    for (key, shape, dtype, bytes) in values {
        let elems = shape.iter().product();
        inputs.push(
            key.clone(),
            shape,
            poot_executor::HostView::new(*dtype, elems, bytes).unwrap(),
        );
    }
    inputs
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// One projection's graph: `lora_linear_batched` over `x [batch, in]`, the shared base weight `w
/// [in, out]`, the pool's stacked factors and the per-row adapter index slot.
fn projection_graph(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    n_adapters: usize,
    r: usize,
) -> Graph {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![batch, in_dim]));
    let idx = b.slot(Slot::LoraIdx, TensorType::f32(vec![batch]));
    let w = b.constant("w", TensorType::f32(vec![in_dim, out_dim]));
    let a = b.constant(
        "lora_a_stacked",
        TensorType::f32(vec![n_adapters, in_dim, r]),
    );
    let bs = b.constant(
        "lora_b_stacked",
        TensorType::f32(vec![n_adapters, r, out_dim]),
    );
    let scaling = b.constant("lora_scaling_vec", TensorType::f32(vec![n_adapters]));
    let out = lora_linear_batched(&b, x, w, None, a, bs, scaling, idx);
    b.finish(out)
}

fn staged(
    target: poot_graph_plan::Target,
    g: &Graph,
) -> poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs> {
    let g = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            fusion: poot_graph_plan::FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
    .expect("compile_staged")
}

#[test]
fn batched_lora_decode_rocm_matches_per_row_merged_weight_reference() {
    let Some(device) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Rocm, RocmDevice::new())
    else {
        return;
    };
    let target = device.target();
    let mut engine = Engine::new(device);

    let (hidden, inter, q_dim, kv_dim) = (8usize, 16usize, 8usize, 4usize);
    let r = 2usize;

    // The seven targeted projections (four attention, three MLP), each with its own
    // `(module_suffix, in_dim, out_dim)`.
    let proj_specs: [(&str, usize, usize); 7] = [
        ("self_attn.q_proj", hidden, q_dim),
        ("self_attn.k_proj", hidden, kv_dim),
        ("self_attn.v_proj", hidden, kv_dim),
        ("self_attn.o_proj", q_dim, hidden),
        ("mlp.gate_proj", hidden, inter),
        ("mlp.up_proj", hidden, inter),
        ("mlp.down_proj", inter, hidden),
    ];

    // Two real adapters registered into a real `LoraAdapterPool` and stacked via its own
    // `stack_for_module`, not a hand-built stacked tensor.
    let adapter_a = adapter(r, &proj_specs, 4.0, 100); // scaling = 4/2 = 2.0
    let adapter_b = adapter(r, &proj_specs, 6.0, 200); // scaling = 6/2 = 3.0
    let mut pool = LoraAdapterPool::new();
    let idx_a = pool.register("adapter_a", adapter_a.clone());
    let idx_b = pool.register("adapter_b", adapter_b.clone());

    // batch=3: row 0 -> adapter_a, row 1 -> adapter_b, row 2 -> NO_ADAPTER.
    let batch = 3usize;
    let lora_idx: Vec<f32> = vec![
        idx_a as f32,
        idx_b as f32,
        LoraAdapterPool::NO_ADAPTER as f32,
    ];

    let mut all_rows: Vec<Vec<f32>> = vec![Vec::new(); batch];
    for &(name, in_dim, out_dim) in &proj_specs {
        let module = format!("model.layers.0.{name}");
        let sm: LoraStackedModule = pool
            .stack_for_module(&module)
            .unwrap()
            .unwrap_or_else(|| panic!("both adapters target {name}"));
        assert_eq!(sm.r, r, "equal-rank adapters need no padding in this test");
        let n_adapters = sm.a.shape()[0];

        let seed = |tag: &str| -> u64 {
            format!("{module}.{tag}")
                .bytes()
                .fold(1469598103934665603u64, |acc, c| {
                    (acc ^ c as u64).wrapping_mul(1099511628211)
                })
        };
        let base_weight = fill(in_dim * out_dim, seed("weight"));
        let x = fill(batch * in_dim, seed("x"));

        // Independent per-row merged-weight reference (CPU, no lora_linear/lora_linear_batched): row
        // m's weight is `W + scaling_m*(A_m@B_m)` for its adapter, or plain `W` for NO_ADAPTER.
        let slot_a = |slot: usize| -> Vec<f32> {
            let per_slot = in_dim * sm.r;
            sm.a.as_f32().unwrap()[slot * per_slot..(slot + 1) * per_slot].to_vec()
        };
        let slot_b = |slot: usize| -> Vec<f32> {
            let per_slot = sm.r * out_dim;
            sm.b.as_f32().unwrap()[slot * per_slot..(slot + 1) * per_slot].to_vec()
        };
        let ref_rows: Vec<Vec<f32>> = (0..batch)
            .map(|row| {
                let merged = match row {
                    0 => add_scaled(
                        &base_weight,
                        &matmul_ref(&slot_a(idx_a), &slot_b(idx_a), in_dim, r, out_dim),
                        adapter_a.config.scaling(),
                    ),
                    1 => add_scaled(
                        &base_weight,
                        &matmul_ref(&slot_a(idx_b), &slot_b(idx_b), in_dim, r, out_dim),
                        adapter_b.config.scaling(),
                    ),
                    _ => base_weight.clone(), // NO_ADAPTER: plain base weight
                };
                matmul_ref(
                    &x[row * in_dim..(row + 1) * in_dim],
                    &merged,
                    1,
                    in_dim,
                    out_dim,
                )
            })
            .collect();

        // Device under test: one batched dispatch of all 3 rows with the pool's stacked constants and
        // `Slot::LoraIdx` carrying each row's adapter id, on ROCm.
        let g = projection_graph(batch, in_dim, out_dim, n_adapters, r);
        let mut builder = WeightStore::builder();
        for (const_name, shape, data) in [
            ("w", vec![in_dim, out_dim], base_weight.clone()),
            (
                "lora_a_stacked",
                vec![n_adapters, in_dim, r],
                sm.a.as_f32().unwrap().to_vec(),
            ),
            (
                "lora_b_stacked",
                vec![n_adapters, r, out_dim],
                sm.b.as_f32().unwrap().to_vec(),
            ),
            (
                "lora_scaling_vec",
                vec![n_adapters],
                sm.scaling.as_f32().unwrap().to_vec(),
            ),
        ] {
            let dense =
                DenseWeight::try_new(DType::F32, shape, Arc::from(f32_bytes(&data))).unwrap();
            builder
                .insert(const_name, WeightEntry::Dense(dense))
                .unwrap();
        }
        let mut values: Vec<SlotValue> = Vec::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            let Storage::Slot(slot) = m.storage else {
                continue;
            };
            let key = m.slot_key().unwrap().clone();
            let shape = m.aval.shape.clone();
            match slot {
                Slot::Activation => values.push((key, shape, DType::F32, f32_bytes(&x))),
                Slot::LoraIdx => values.push((key, shape, DType::F32, f32_bytes(&lora_idx))),
                other => unreachable!("unexpected slot {other:?} in a batched lora projection"),
            }
        }
        let device_inputs = step_inputs_from(&values);
        let exe = engine
            .load_weights(
                Arc::new(builder.build()),
                poot_executor::WeightSource::ConstNames,
            )
            .unwrap();
        let entry = engine.add_entry(exe, &staged(target, &g)).unwrap();
        let out_bytes = engine
            .step(exe, entry, &device_inputs, &mut NoSync)
            .expect("rocm batched lora step")
            .read()
            .expect("rocm batched lora readback");
        engine.unload(exe).unwrap();
        let out_data = bytemuck_f32(&out_bytes);
        assert_eq!(out_data.len(), batch * out_dim, "{name}: output length");

        for (row, ref_row) in ref_rows.iter().enumerate() {
            let got_row = &out_data[row * out_dim..(row + 1) * out_dim];
            for (i, (g, r)) in got_row.iter().zip(ref_row).enumerate() {
                // 1e-3 relative tolerance.
                let tol = 1e-3 * r.abs().max(1e-3);
                assert!(
                    (g - r).abs() <= tol,
                    "{name} row {row} elem {i}: rocm {g} vs cpu merged-weight reference {r}"
                );
            }
            all_rows[row].extend_from_slice(got_row);
        }
    }

    // The three rows must differ (two adapters plus a no-adapter row must not degenerate to one
    // output), across all projections.
    let d01 = max_abs_error(&all_rows[0], &all_rows[1]);
    let d02 = max_abs_error(&all_rows[0], &all_rows[2]);
    let d12 = max_abs_error(&all_rows[1], &all_rows[2]);
    assert!(
        d01 > 1e-3 && d02 > 1e-3 && d12 > 1e-3,
        "the three rows (adapter_a, adapter_b, NO_ADAPTER) must be visibly different: d01={d01} \
         d02={d02} d12={d12}"
    );
}

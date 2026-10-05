//! Spec 248 Phase 2 engine wiring, PTX port (epic 129 B8, update 0594): first real-hardware run of the
//! batched-LoRA op on PTX/NVIDIA, covering the shape of all seven projections (`q/k/v/o_proj`,
//! `gate/up/down_proj`). Same test as the wgpu twin (`crates/poot-gpu/tests/graph/lora.rs`) and the
//! CPU-oracle twin in `poot-eval`: two real adapters registered into a `LoraAdapterPool`, stacked via
//! `stack_for_module`, dispatched through `ops::lora_linear_batched` with a 3-row batch (`adapter_a`,
//! `adapter_b`, `NO_ADAPTER`), one projection graph per module shape, run through the executor
//! contract's resident path (CUDA `IndexedMatMul`/`Gather`/`MatMul` kernels). Each row must match its
//! own CPU merged-weight reference (never touching `lora_linear`/`lora_linear_batched`) within `1e-3`
//! relative tolerance (the wgpu test uses `1e-2`), and the three rows must differ from each other.
//! LoRA is not part of `Model::trace`: the op is exercised on a single projection, the wiring into a
//! model is a graph transform's concern.
//!
//! Skips cleanly with no NVIDIA GPU. Real-hardware runs go through RunPod as a
//! compiled test binary (`cargo test -p poot-ptx-gpu --test batched_lora_ptx --no-run`, then run the binary on
//! the pod), as in `cross_backend.rs`.

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::ops::lora_linear_batched;
use poot_graph_ir::{Builder, Graph, Slot, Storage, TensorType};
use poot_load::lora::{
    LoraAdapter, LoraAdapterConfig, LoraAdapterPool, LoraStackedModule, LoraWeight,
};
use poot_ptx_gpu::PtxDevice;
use poot_tensor::HostTensor;

mod common;

/// Deterministic pseudo-random fill in [-1, 1), no rng dependency. Local copy of the wgpu/CPU-oracle twins'
/// `fill` (integration-test crates cannot reach `poot-eval`'s `pub(super)` helper).
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

/// `a [m,k] @ b [k,n] -> [m,n]`, row-major: the hand-rolled reference for the merged-weight tests.
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

/// A synthetic adapter targeting all seven projections at layer 0 (same as the wgpu/CPU-oracle `adapter` helper).
/// `proj_specs` carries each `(module_suffix, in_dim, out_dim)`: `o_proj` projects from the concatenated
/// attention output (`q_dim`) rather than `hidden`, and `down_proj` projects `inter` to `hidden` (the reverse
/// of `gate_proj`/`up_proj`).
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

#[test]
fn batched_lora_decode_ptx_matches_per_row_merged_weight_reference() {
    let Some(mut ptx) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };

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
        // `Slot::LoraIdx` carrying each row's adapter id, on PTX/NVIDIA.
        let g = projection_graph(batch, in_dim, out_dim, n_adapters, r);
        let mut inputs: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            let tensor = match m.storage {
                Storage::Slot(Slot::Activation) => HostTensor::f32(m.aval.shape.clone(), x.clone()),
                Storage::Slot(Slot::LoraIdx) => HostTensor::f32(vec![batch], lora_idx.clone()),
                Storage::Slot(other) => {
                    unreachable!("unexpected slot {other:?} in a batched lora projection")
                }
                Storage::Const => {
                    let data = match m.name.as_deref().expect("const without a name") {
                        "w" => base_weight.clone(),
                        "lora_a_stacked" => sm.a.as_f32().unwrap().to_vec(),
                        "lora_b_stacked" => sm.b.as_f32().unwrap().to_vec(),
                        "lora_scaling_vec" => sm.scaling.as_f32().unwrap().to_vec(),
                        other => unreachable!("unexpected const {other}"),
                    };
                    assert_eq!(data.len(), m.aval.numel(), "data/shape length mismatch");
                    HostTensor::f32(m.aval.shape.clone(), data)
                }
                other => unreachable!("unexpected input storage {other:?}"),
            };
            inputs.insert(id, tensor.into());
        }
        let out = common::run_resident(&mut ptx, &g, &inputs);
        let out_data = out.as_f32().unwrap();
        assert_eq!(out_data.len(), batch * out_dim, "{name}: output length");

        for (row, ref_row) in ref_rows.iter().enumerate() {
            let got_row = &out_data[row * out_dim..(row + 1) * out_dim];
            for (i, (g, r)) in got_row.iter().zip(ref_row).enumerate() {
                // 1e-3 relative tolerance (the wgpu test uses 1e-2; PTX f32 accumulation is at least as tight).
                let tol = 1e-3 * r.abs().max(1e-3);
                assert!(
                    (g - r).abs() <= tol,
                    "{name} row {row} elem {i}: ptx {g} vs cpu merged-weight reference {r}"
                );
            }
            all_rows[row].extend_from_slice(got_row);
        }
    }

    // The three rows must differ (two different adapters plus a no-adapter row), across all projections.
    let d01 = max_abs_error(&all_rows[0], &all_rows[1]);
    let d02 = max_abs_error(&all_rows[0], &all_rows[2]);
    let d12 = max_abs_error(&all_rows[1], &all_rows[2]);
    assert!(
        d01 > 1e-3 && d02 > 1e-3 && d12 > 1e-3,
        "the three rows (adapter_a, adapter_b, NO_ADAPTER) must be visibly different: d01={d01} \
         d02={d02} d12={d12}"
    );
}

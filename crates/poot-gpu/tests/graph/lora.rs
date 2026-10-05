//! Spec 248 Phase 2 engine wiring (epic 129 B8): the first real-hardware run of the batched-LoRA linear. The CPU-oracle
//! twin is `poot-eval`'s per-row merged-weight test. This module is the same test (two real adapters registered into a
//! real `LoraAdapterPool`, the same independently computed per-row merged-weight reference that never touches
//! `lora_linear`/`lora_linear_batched`) with one difference: the batched-LoRA linear dispatches through the executor
//! contract (`poot_executor::Engine<WgpuDevice>`, via `run_via_contract`; wgpu/RADV `IndexedMatMul`/`Gather`/`MatMul`
//! kernels) instead of `poot_eval`'s CPU executor. `IndexedMatMul` itself is unchanged by spec 248 (MoE's
//! `moe_sparse`/`moe_grouped` already exercise it on this backend); this is the first run with LoRA's tiny `r` (rank 2)
//! and a per-row `Slot::LoraIdx` selecting between two real adapters plus a `NO_ADAPTER` row, not MoE's routing shape.
//!
//! Each of the seven targetable projections is a hand-built graph around one `lora_linear_batched`: the family tracers
//! no longer wire LoRA (the adapter transform is the graph pass the driver's LoRA registry prepares), so the op's device
//! coverage is its own graph, with each projection's own `(in, out)` shape (`o_proj` and `down_proj` are asymmetric).

use std::collections::HashMap;

use poot_graph_ir::ops::lora_linear_batched;
use poot_graph_ir::{Builder, Slot, Storage, TensorType};
use poot_load::lora::{
    LoraAdapter, LoraAdapterConfig, LoraAdapterPool, LoraStackedModule, LoraWeight,
};
use poot_tensor::HostTensor;

/// Deterministic pseudo-random fill in [-1, 1), no rng dependency; mirrors `poot-eval`'s test helper of the same shape
/// (`crates/poot-eval/src/tests/helpers.rs::fill`), reimplemented since this crate cannot reach its `pub(super)` helpers
/// (as in `crates/poot-llm/tests/lora_wiring.rs`).
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

/// `a [m,k] @ b [k,n] -> [m,n]`, row-major: the hand-rolled reference every merged-weight test in this spec uses as ground
/// truth that never calls the op under test.
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

/// A synthetic adapter targeting all seven targetable projections at layer 0 (mirrors the PTX/ROCm/CPU-oracle twins'
/// `adapter` helper). `proj_specs` carries each projection's `(module_suffix, in_dim, out_dim)`: `o_proj` projects from the
/// concatenated attention output (`q_dim`), not from `hidden` like the other three attention projections, and `down_proj`
/// projects from `inter` to `hidden` (the reverse of `gate_proj`/`up_proj`).
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

#[test]
fn batched_lora_decode_gpu_matches_per_row_merged_weight_reference() {
    let _gpu_guard = super::gpu_lock();
    let Some((mut exec, target)) = super::open_engine_or_skip() else {
        return;
    };

    let (hidden, inter, q_dim, kv_dim) = (8usize, 16usize, 8usize, 4usize);
    let r = 2usize;

    // All seven targeted projections (four attention, three MLP), each with its own `(module_suffix, in_dim, out_dim)`.
    let proj_specs: [(&str, usize, usize); 7] = [
        ("self_attn.q_proj", hidden, q_dim),
        ("self_attn.k_proj", hidden, kv_dim),
        ("self_attn.v_proj", hidden, kv_dim),
        ("self_attn.o_proj", q_dim, hidden),
        ("mlp.gate_proj", hidden, inter),
        ("mlp.up_proj", hidden, inter),
        ("mlp.down_proj", inter, hidden),
    ];

    // Two real adapters registered into a real LoraAdapterPool and stacked via its own `stack_for_module`, not a hand-built stacked tensor (same setup as the CPU-oracle test).
    let adapter_a = adapter(r, &proj_specs, 4.0, 100); // scaling = 4/2 = 2.0
    let adapter_b = adapter(r, &proj_specs, 6.0, 200); // scaling = 6/2 = 3.0
    let mut pool = LoraAdapterPool::new();
    let idx_a = pool.register("adapter_a", adapter_a.clone());
    let idx_b = pool.register("adapter_b", adapter_b.clone());

    let stacked: HashMap<&str, LoraStackedModule> = proj_specs
        .iter()
        .map(|&(name, _, _)| {
            let module = format!("model.layers.0.{name}");
            let sm = pool
                .stack_for_module(&module)
                .unwrap()
                .unwrap_or_else(|| panic!("both adapters target {name}"));
            (name, sm)
        })
        .collect();
    assert_eq!(
        stacked["self_attn.q_proj"].r, r,
        "equal-rank adapters need no padding in this test"
    );

    // batch=3: row 0 -> adapter_a, row 1 -> adapter_b, row 2 -> NO_ADAPTER. A different input row each so a row-index bug is caught, not just a row-adapter-id bug.
    let batch = 3usize;
    let lora_idx: Vec<f32> = vec![
        idx_a as f32,
        idx_b as f32,
        LoraAdapterPool::NO_ADAPTER as f32,
    ];
    let scalings = [adapter_a.config.scaling(), adapter_b.config.scaling(), 0.0];
    let slots_of_row = [idx_a, idx_b, LoraAdapterPool::NO_ADAPTER];

    let seeded = |name: &str, shape: &[usize]| -> Vec<f32> {
        let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
            (acc ^ c as u64).wrapping_mul(1099511628211)
        });
        fill(shape.iter().product::<usize>().max(1), seed)
    };

    for &(name, in_dim, out_dim) in &proj_specs {
        let sm = &stacked[name];
        let (a_all, b_all) = (sm.a.as_f32().unwrap(), sm.b.as_f32().unwrap());
        let base_w = seeded(&format!("model.layers.0.{name}.weight"), &[in_dim, out_dim]);
        let bias = seeded(&format!("model.layers.0.{name}.bias"), &[out_dim]);
        // The same input row in every batch row: the rows differ only by their adapter.
        let x = seeded(&format!("{name}.x"), &[1, in_dim]).repeat(batch);

        // Independent per-row merged-weight reference (host, never calls lora_linear/lora_linear_batched): row m is
        // `x_m @ (W + scaling_m * A_m @ B_m) + bias` for its own adapter, or `x_m @ W + bias` for NO_ADAPTER.
        let mut reference = Vec::with_capacity(batch * out_dim);
        for row in 0..batch {
            let merged = if row == 2 {
                base_w.clone()
            } else {
                let slot = slots_of_row[row];
                let a = &a_all[slot * in_dim * r..(slot + 1) * in_dim * r];
                let b = &b_all[slot * r * out_dim..(slot + 1) * r * out_dim];
                add_scaled(
                    &base_w,
                    &matmul_ref(a, b, in_dim, r, out_dim),
                    scalings[row],
                )
            };
            let y = matmul_ref(
                &x[row * in_dim..(row + 1) * in_dim],
                &merged,
                1,
                in_dim,
                out_dim,
            );
            reference.extend(y.iter().zip(&bias).map(|(y, b)| y + b));
        }

        // The device under test: one `lora_linear_batched` over all 3 rows with the pool's real stacked constants and
        // `Slot::LoraIdx` carrying each row's adapter id (wgpu/RADV `IndexedMatMul`/`Gather`/`MatMul` kernels).
        let b = Builder::new();
        let x_in = b.slot_named(Slot::Activation, "x", TensorType::f32(vec![batch, in_dim]));
        let w = b.constant("w", TensorType::f32(vec![in_dim, out_dim]));
        let bias_in = b.constant("bias", TensorType::f32(vec![out_dim]));
        let a_in = b.constant("lora_a_stacked", TensorType::f32(sm.a.shape().to_vec()));
        let b_in = b.constant("lora_b_stacked", TensorType::f32(sm.b.shape().to_vec()));
        let s_in = b.constant(
            "lora_scaling_vec",
            TensorType::f32(sm.scaling.shape().to_vec()),
        );
        let idx = b.slot(Slot::LoraIdx, TensorType::f32(vec![batch]));
        let out = lora_linear_batched(&b, x_in, w, Some(bias_in), a_in, b_in, s_in, idx);
        let g = b.finish(out);

        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            let data = match m.name.as_deref().unwrap() {
                "w" => base_w.clone(),
                "bias" => bias.clone(),
                "lora_a_stacked" => a_all.to_vec(),
                "lora_b_stacked" => b_all.to_vec(),
                "lora_scaling_vec" => sm.scaling.as_f32().unwrap().to_vec(),
                _ if m.storage == Storage::Slot(Slot::LoraIdx) => lora_idx.clone(),
                _ => x.clone(),
            };
            assert_eq!(data.len(), m.aval.numel(), "{name}: data/shape length");
            inputs.insert(id, HostTensor::f32(m.aval.shape.clone(), data));
        }
        let got = super::run_via_contract(&mut exec, target, &g, &inputs);
        let got = got.as_f32().unwrap();
        assert_eq!(got.len(), batch * out_dim);
        for (i, (g, r)) in got.iter().zip(&reference).enumerate() {
            let tol = 1e-3 * r.abs().max(1e-3);
            assert!(
                (g - r).abs() <= tol,
                "{name} row {} elem {}: gpu {g} vs host merged-weight reference {r}",
                i / out_dim,
                i % out_dim
            );
        }

        // The three rows must be visibly different on real hardware too (two different real adapters plus a no-adapter row, not degenerating to the same output).
        let row = |m: usize| &got[m * out_dim..(m + 1) * out_dim];
        let (d01, d02, d12) = (
            max_abs_error(row(0), row(1)),
            max_abs_error(row(0), row(2)),
            max_abs_error(row(1), row(2)),
        );
        assert!(
            d01 > 1e-3 && d02 > 1e-3 && d12 > 1e-3,
            "{name}: the three rows (adapter_a, adapter_b, NO_ADAPTER) must be visibly different: \
             d01={d01} d02={d02} d12={d12}"
        );
    }
}

//! PTX/NVIDIA receipt for updates 0749/0751's Mixtral shared-pool decode/prefill tracers
//! (`poot_models::mixtral::trace_mixtral_decode_kv_masked_batched_shared_pool` /
//! `trace_mixtral_prefill_kv_shared_pool`), previously verified on the CPU oracle, wgpu/RADV (updates 0750/0751)
//! and ROCm (update 0753, `crates/poot-rocm-gpu/tests/mixtral_moe_batched.rs`). This is the PTX twin of that
//! file: same graphs, schedules and CPU-oracle comparison, run through the shared
//! `poot_executor::Device` contract (card 549) - `PtxDevice` via `tests/common`'s `run_resident`
//! (prefill) and `DecodeEntry::capture` (the staggered batched decode schedule), not the deleted
//! `PtxGraphExecutor`.
//!
//! Skips cleanly with no NVIDIA GPU; run on hardware via RunPod (the
//! `poot-orchestrator exec` route).

use poot_runtime_common::DeviceBackend;
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_models::mixtral::{
    MixtralParams, trace_mixtral_decode_kv_masked_batched_shared_pool,
    trace_mixtral_prefill_kv_shared_pool,
};
use poot_models::qwen2::Qwen2Config;
use poot_ptx_gpu::PtxDevice;
use poot_tensor::DType;
use poot_tensor::HostTensor;

mod common;

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.1
        })
        .collect()
}

/// Deterministic per-name weight fill, same as the wgpu/ROCm sibling tests and `poot-eval`'s
/// `tests::moe_paged_decode::bind_const`.
fn bind_const(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
        (acc ^ c as u64).wrapping_mul(1099511628211)
    });
    HostTensor::f32(
        shape.to_vec(),
        fill(shape.iter().product::<usize>().max(1), seed)
            .iter()
            .map(|v| v * 0.1)
            .collect(),
    )
}

/// One staggered-admission request, same shape as the wgpu/ROCm sibling tests.
struct MixtralReq {
    slot: usize,
    admit_step: usize,
    tokens: Vec<u32>,
}

fn mixtral_staggered_schedule() -> Vec<MixtralReq> {
    vec![
        MixtralReq {
            slot: 0,
            admit_step: 0,
            tokens: vec![0, 3, 1, 5, 2, 4],
        },
        MixtralReq {
            slot: 1,
            admit_step: 2,
            tokens: vec![4, 1, 0, 2],
        },
        MixtralReq {
            slot: 2,
            admit_step: 3,
            tokens: vec![5, 2, 4, 1, 3],
        },
        MixtralReq {
            slot: 3,
            admit_step: 5,
            tokens: vec![1, 0, 5],
        },
        // Reuses slot 0 after the first request there finishes at engine step 6.
        MixtralReq {
            slot: 0,
            admit_step: 7,
            tokens: vec![2, 4, 0, 3],
        },
    ]
}

/// Bind one batched shared-pool decode step where physical rows sit at independent positions (as in the
/// wgpu/ROCm sibling tests).
fn bind_batched_step_staggered(
    g: &Graph,
    tokens: &[u32],
    positions: &[usize],
    cap: usize,
    batch: usize,
    global_slotmap: &[i32],
) -> HashMap<ValueId, Value> {
    let mut inputs = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            // An I32-dtype slot needs the authoritative `Tensor::ints` words (`poot-eval`'s exact-I32
            // check, card 449); a plain f32-backed `Tensor` gets refused for whichever of
            // these three is I32 on this graph.
            Storage::Slot(Slot::Token) => {
                let data = tokens.iter().map(|&t| t as i32).collect();
                let tensor = if m.aval.dtype == DType::I32 {
                    HostTensor::i32(vec![batch], data)
                } else {
                    HostTensor::f32(vec![batch], data.into_iter().map(|t| t as f32).collect())
                };
                inputs.insert(id, tensor.into());
            }
            Storage::Slot(Slot::Pos) => {
                let data: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
                let tensor = if m.aval.dtype == DType::I32 {
                    HostTensor::i32(vec![batch], data)
                } else {
                    HostTensor::f32(vec![batch], data.into_iter().map(|p| p as f32).collect())
                };
                inputs.insert(id, tensor.into());
            }
            Storage::Slot(Slot::SeqLen) => {
                let seq_len = (positions.iter().max().copied().unwrap_or(0) + 1) as i32;
                let tensor = if m.aval.dtype == DType::I32 {
                    HostTensor::i32(m.aval.shape.clone(), vec![seq_len])
                } else {
                    HostTensor::scalar(seq_len as f32)
                };
                inputs.insert(id, tensor.into());
            }
            Storage::Slot(Slot::Mask) => {
                let mask: Vec<f32> = positions
                    .iter()
                    .flat_map(|&pos| (0..cap).map(move |t| if t <= pos { 0.0 } else { -1.0e9 }))
                    .collect();
                inputs.insert(id, HostTensor::f32(vec![batch, cap], mask).into());
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![batch, cap], global_slotmap.to_vec()).into(),
                );
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in batched shared-pool MoE decode")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape).into());
            }
            Storage::Computed(computed) => {
                inputs.insert(
                    id,
                    HostTensor::f32(computed.shape(), computed.values_f32()).into(),
                );
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

/// Receipt for spec 266-batched Phase 1's Mixtral decode tracer on PTX: drives the staggered admit/evict
/// schedule of the CPU-oracle/wgpu/ROCm tests through `common::DecodeEntry::capture` and compares each
/// step's logits against `eval_with_state` on the same graph and inputs. Skips if no NVIDIA/CUDA device
/// is present.
#[test]
fn mixtral_batched_shared_pool_decode_ptx_matches_cpu_staggered() {
    let Some(mut ptx) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };

    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let mp = MixtralParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
    };
    let n_slots = 4usize;
    let cap = 8usize;
    // Private per-physical-row pool region, as in the CPU-oracle/wgpu/ROCm tests.
    let pool_slots = n_slots * cap;
    let reqs = mixtral_staggered_schedule();
    let max_step = reqs
        .iter()
        .map(|r| r.admit_step + r.tokens.len())
        .max()
        .unwrap();

    let g = trace_mixtral_decode_kv_masked_batched_shared_pool(cfg, mp, cap, n_slots, pool_slots);
    g.validate()
        .expect("mixtral batched shared-pool decode graph should validate");

    let mut cpu_caches: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();

    let global_slotmap: Vec<i32> = (0..n_slots)
        .flat_map(|row| (0..cap).map(move |t| (row * cap + t) as i32))
        .collect();

    // One decode entry for the whole staggered schedule: its carried state starts zero (Z5, matching
    // `cpu_caches` above) and evolves in place across `step` calls, replacing the pre-549 manual
    // cpu_caches/ptx_caches buffer threading on the PTX side.
    let first_tokens = vec![0u32; n_slots];
    let first_positions = vec![0usize; n_slots];
    let first_inputs = bind_batched_step_staggered(
        &g,
        &first_tokens,
        &first_positions,
        cap,
        n_slots,
        &global_slotmap,
    );
    let mut decode = common::DecodeEntry::capture(
        &mut ptx,
        &g,
        &first_inputs,
        poot_graph_plan::FusionPolicy::Full,
    );

    let mut max_rel_seen = 0.0f32;
    for t in 0..max_step {
        let mut tokens = vec![0u32; n_slots];
        let mut positions = vec![0usize; n_slots];
        for r in &reqs {
            if t < r.admit_step || t >= r.admit_step + r.tokens.len() {
                continue;
            }
            let local_pos = t - r.admit_step;
            tokens[r.slot] = r.tokens[local_pos];
            positions[r.slot] = local_pos;
        }
        let inputs =
            bind_batched_step_staggered(&g, &tokens, &positions, cap, n_slots, &global_slotmap);

        let mut cpu_inputs = inputs.clone();
        for (ci, &(si, _)) in g.state.iter().enumerate() {
            cpu_inputs.insert(si, cpu_caches[ci].clone().into());
        }
        let cpu_eval = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("mixtral batched decode CPU eval");
        let cpu_logits = cpu_eval.output.into_host().expect("dense output");
        let cpu_new: Vec<HostTensor> = cpu_eval
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .expect("dense state");
        cpu_caches = cpu_new;

        let ptx_logits = decode.step(&g, &inputs);

        assert_eq!(
            ptx_logits.shape(),
            cpu_logits.shape(),
            "logits shape at step {t}"
        );
        for row in 0..n_slots {
            let got = &ptx_logits.as_f32().unwrap()[row * cfg.vocab..(row + 1) * cfg.vocab];
            let want = &cpu_logits.as_f32().unwrap()[row * cfg.vocab..(row + 1) * cfg.vocab];
            for (a, b) in got.iter().zip(want.iter()) {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "non-finite logit at step {t} row {row}: PTX {a} vs CPU {b}"
                );
                let rel = (a - b).abs() / b.abs().max(1e-3);
                max_rel_seen = max_rel_seen.max(rel);
            }
        }
        assert!(
            max_rel_seen <= 5e-3,
            "mixtral batched shared-pool PTX-vs-CPU diff {max_rel_seen} too large at step {t}"
        );
    }
    eprintln!(
        "mixtral_batched_shared_pool_decode_ptx_matches_cpu_staggered: max rel diff vs CPU = \
         {max_rel_seen:.2e} over {max_step} staggered engine steps, {n_slots} slots"
    );
}

/// Bind one shared-pool prefill call over a non-contiguous physical slot layout (as in the wgpu/ROCm sibling
/// tests).
fn bind_shared_pool_prefill(g: &Graph, tokens: &[u32], inv: &[i32]) -> HashMap<ValueId, Value> {
    let n = tokens.len();
    let mut inputs = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            // See `bind_batched_step_staggered` above: an I32-dtype Token slot needs the
            // authoritative `Tensor::ints` words, not a plain f32-backed `Tensor`.
            Storage::Slot(Slot::Token) => {
                let data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                let tensor = if m.aval.dtype == DType::I32 {
                    HostTensor::i32(vec![n], data)
                } else {
                    HostTensor::f32(vec![n], data.into_iter().map(|t| t as f32).collect())
                };
                inputs.insert(id, tensor.into());
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(
                    id,
                    HostTensor::i32(m.aval.shape.clone(), inv.to_vec()).into(),
                );
            }
            Storage::Slot(Slot::Mask) => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                let l = m.aval.shape[2];
                let mut mask = vec![0.0f32; l * l];
                for i in 0..l {
                    for j in 0..l {
                        mask[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                    }
                }
                inputs.insert(id, HostTensor::f32(m.aval.shape.clone(), mask).into());
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in shared-pool MoE prefill")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape).into());
            }
            Storage::Computed(computed) => {
                inputs.insert(
                    id,
                    HostTensor::f32(computed.shape(), computed.values_f32()).into(),
                );
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

/// Receipt for spec 266-batched Phase 1's Mixtral prefill tracer on PTX: dispatches
/// `poot_models::mixtral::trace_mixtral_prefill_kv_shared_pool` through `common::run_resident` over the
/// same non-contiguous physical slot layout as the CPU-oracle/wgpu/ROCm tests and compares against
/// `eval_with_state`. Skips if no NVIDIA/CUDA device is present.
#[test]
fn mixtral_shared_pool_prefill_ptx_matches_cpu_noncontiguous() {
    let Some(mut ptx) =
        poot_test_util::device_skip::open_or_skip(DeviceBackend::Ptx, PtxDevice::new())
    else {
        return;
    };

    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    };
    let mp = MixtralParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
    };
    // Copy of `poot_eval::tests::moe_paged_prefill`'s `TOKENS`/`GLOBAL_SLOTS`/`POOL_SLOTS` (as in the wgpu/ROCm
    // sibling tests).
    const TOKENS: [u32; 7] = [5, 11, 3, 20, 8, 1, 27];
    const GLOBAL_SLOTS: [usize; 7] = [3, 17, 1, 9, 14, 2, 11];
    const POOL_SLOTS: usize = 20;

    let n = TOKENS.len();
    let g = trace_mixtral_prefill_kv_shared_pool(cfg, mp, n, POOL_SLOTS);
    g.validate()
        .expect("mixtral shared-pool prefill graph should validate");

    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let inputs = bind_shared_pool_prefill(&g, &TOKENS, &inv);

    let mut cpu_inputs = inputs.clone();
    for &(si, _) in &g.state {
        cpu_inputs.insert(si, HostTensor::zeros(g.aval(si).shape.clone()).into());
    }
    let cpu_logits = eval(&g, &cpu_inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("mixtral shared-pool prefill CPU eval")
        .output
        .into_host()
        .expect("dense output");

    // Zero-seeded carried state, matching `cpu_inputs`'s zeros above and the executor contract's own
    // zero-at-first-declaration state (Z5): a one-shot prefill entry, stepped once.
    let ptx_logits = common::run_resident(&mut ptx, &g, &inputs);

    assert_eq!(ptx_logits.shape(), cpu_logits.shape(), "logits shape");
    let mut max_rel = 0.0f32;
    for (a, b) in ptx_logits
        .as_f32()
        .unwrap()
        .iter()
        .zip(cpu_logits.as_f32().unwrap().iter())
    {
        assert!(
            a.is_finite() && b.is_finite(),
            "non-finite logit: PTX {a} vs CPU {b}"
        );
        let rel = (a - b).abs() / b.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    assert!(
        max_rel <= 5e-3,
        "mixtral shared-pool prefill PTX-vs-CPU diff {max_rel} too large"
    );
    eprintln!(
        "mixtral_shared_pool_prefill_ptx_matches_cpu_noncontiguous: max rel diff vs CPU = {max_rel:.2e}"
    );
}

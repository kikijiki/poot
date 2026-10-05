//! Differential CPU-oracle test for spec 249 (MoE shared-pool prefill): `poot_models::moe_prefill::trace_moe_prefill_kv_shared_pool`
//! and its wrappers (`granite::trace_granite_prefill_kv_shared_pool`, `qwen3moe::trace_qwen3_moe_prefill_kv_shared_pool`)
//! must produce the same logits as the contiguous prefill tracers (`trace_granite_prefill_kv`/
//! `trace_qwen3_moe_prefill_kv`) for identical weights/tokens. The shared-pool tracer's attention reads the fresh k/v
//! directly, so the cache-write layout (contiguous `DynamicUpdateSlice` vs shared-pool `ScatterUpdate` at arbitrary
//! slots) never feeds the output (a prefill attends over its fresh k/v, never over the pool it just
//! wrote). A non-contiguous physical slot layout proves the equivalence does
//! not depend on slot order.
//!
//! Style follows `qwen2_pipeline::chunked_prefill_matches_one_shot_shared_pool`, but both tracers share the op sequence
//! up to the cache write, so the tolerance is strict (the chunked-vs-one-shot test reduces in a different order).

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_models::granite::{
    GraniteParams, MoeShape, trace_granite_prefill_kv, trace_granite_prefill_kv_shared_pool,
};
use poot_models::mixtral::{
    MixtralParams, trace_mixtral_prefill, trace_mixtral_prefill_kv_shared_pool,
};
use poot_models::qwen2::Qwen2Config;
use poot_models::qwen3moe::{
    Qwen3MoeParams, trace_qwen3_moe_prefill_kv, trace_qwen3_moe_prefill_kv_shared_pool,
};
use poot_tensor::HostTensor;
use std::collections::HashMap as Map;

use super::helpers::*;
use poot_test_util::max_abs_error;

/// Deterministic per-name weight fill (same scheme as `qwen2_pipeline::chunked_prefill_matches_one_shot_shared_pool`'s
/// `weight`).
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

/// The `mask.prefill` step input's content (card 550a): `[1,1,l,l]`, 0 on/below the diagonal, `-1e30`
/// above (the additive prefill mask both contiguous and shared-pool graphs bind identically).
fn prefill_causal_tensor(shape: &[usize]) -> HostTensor {
    let l = shape[2];
    let mut data = vec![-1.0e30f32; l * l];
    for i in 0..l {
        for j in 0..=i {
            data[i * l + j] = 0.0;
        }
    }
    HostTensor::f32(shape.to_vec(), data)
}

fn bind_contiguous(g: &Graph, tokens: &[u32]) -> Map<ValueId, HostTensor> {
    let n = tokens.len();
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![n], tokens.iter().map(|&t| t as i32).collect()),
                );
            }
            Storage::Slot(Slot::Mask) => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                inputs.insert(id, prefill_causal_tensor(&m.aval.shape));
            }
            Storage::Slot(Slot::Pos) => {
                // Card 550: dense tracers (e.g. `trace_granite_prefill_kv`) left the host-built `mask.prefill`
                // step input behind in favor of an in-graph mask over this `[1,n]` absolute-position input.
                inputs.insert(id, HostTensor::i32(vec![1, n], (0..n as i32).collect()));
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in contiguous MoE prefill")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape));
            }
            Storage::Computed(computed) => {
                inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

fn bind_shared_pool(g: &Graph, tokens: &[u32], inv: &[i32]) -> Map<ValueId, HostTensor> {
    let n = tokens.len();
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                inputs.insert(
                    id,
                    HostTensor::i32(vec![n], tokens.iter().map(|&t| t as i32).collect()),
                );
            }
            Storage::Slot(Slot::SlotMap) => {
                inputs.insert(id, HostTensor::i32(m.aval.shape.clone(), inv.to_vec()));
            }
            Storage::Slot(Slot::Mask) => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                inputs.insert(id, prefill_causal_tensor(&m.aval.shape));
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in shared-pool MoE prefill")
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, bind_const(name, &m.aval.shape));
            }
            Storage::Computed(computed) => {
                inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    inputs
}

fn zero_seed_state(g: &Graph, inputs: &mut Map<ValueId, HostTensor>) {
    for &(si, _) in &g.state {
        inputs.insert(si, HostTensor::zeros(g.aval(si).shape.clone()));
    }
}

/// The non-contiguous physical slot layout shared by both tests below (from
/// `qwen2_pipeline::chunked_prefill_matches_one_shot_shared_pool`'s `global_slots`).
const GLOBAL_SLOTS: [usize; 7] = [3, 17, 1, 9, 14, 2, 11];
const TOKENS: [u32; 7] = [5, 11, 3, 20, 8, 1, 27];
const POOL_SLOTS: usize = 20;

#[test]
fn granitemoe_shared_pool_prefill_matches_contiguous() {
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
    let gp = GraniteParams {
        moe: Some(MoeShape {
            n_experts: 4,
            top_k: 2,
            inter: 12,
        }),
        embed_mult: 1.5,
        attn_mult: 0.0625,
        residual_mult: 0.7,
        logits_scale: 3.0,
    };
    let n = TOKENS.len();
    let tokens: Vec<u32> = TOKENS.to_vec();

    let g_contig = trace_granite_prefill_kv(cfg, gp, n, n);
    let mut ci = bind_contiguous(&g_contig, &tokens);
    zero_seed_state(&g_contig, &mut ci);
    let (logits_contig, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = ci
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_contig, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let g_pool = trace_granite_prefill_kv_shared_pool(cfg, gp, n, POOL_SLOTS);
    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let mut pi = bind_shared_pool(&g_pool, &tokens, &inv);
    zero_seed_state(&g_pool, &mut pi);
    let (logits_pool, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = pi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_pool, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(
        logits_contig.shape(),
        logits_pool.shape(),
        "granitemoe shared-pool vs contiguous prefill: logits shape mismatch"
    );
    let err = max_abs_error(
        logits_contig.as_f32().unwrap(),
        logits_pool.as_f32().unwrap(),
    );
    assert!(
        err < 1e-5,
        "granitemoe shared-pool vs contiguous prefill logits differ: max abs err {err}"
    );
}

#[test]
fn granite_dense_shared_pool_prefill_matches_contiguous() {
    // dense (non-MoE) granite through the same generic shared-pool tracer (the `gp.moe: None` arm of
    // `trace_granite_prefill_kv_shared_pool`'s FFN closure), proving the generic core is not MoE-only.
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
    let gp = GraniteParams {
        moe: None,
        embed_mult: 12.0,
        attn_mult: 0.015625,
        residual_mult: 0.22,
        logits_scale: 6.0,
    };
    let n = TOKENS.len();
    let tokens: Vec<u32> = TOKENS.to_vec();

    let g_contig = trace_granite_prefill_kv(cfg, gp, n, n);
    let mut ci = bind_contiguous(&g_contig, &tokens);
    zero_seed_state(&g_contig, &mut ci);
    let (logits_contig, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = ci
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_contig, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let g_pool = trace_granite_prefill_kv_shared_pool(cfg, gp, n, POOL_SLOTS);
    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let mut pi = bind_shared_pool(&g_pool, &tokens, &inv);
    zero_seed_state(&g_pool, &mut pi);
    let (logits_pool, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = pi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_pool, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(logits_contig.shape(), logits_pool.shape());
    let err = max_abs_error(
        logits_contig.as_f32().unwrap(),
        logits_pool.as_f32().unwrap(),
    );
    assert!(
        err < 1e-5,
        "dense granite shared-pool vs contiguous prefill logits differ: max abs err {err}"
    );
}

#[test]
fn qwen3_moe_shared_pool_prefill_matches_contiguous() {
    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24,
        layers: 3,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: true,
        ..Default::default()
    };
    // Mixed dense/MoE layers (layer 0 dense, layer 1 routed, layer 2 dense): the per-layer switch must survive the generic
    // shared-pool tracer, not just the all-sparse case.
    let mp = Qwen3MoeParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
        sparse_layer: vec![false, true, false],
    };
    let n = TOKENS.len();
    let tokens: Vec<u32> = TOKENS.to_vec();

    let g_contig = trace_qwen3_moe_prefill_kv(cfg, mp.clone(), n, n);
    let mut ci = bind_contiguous(&g_contig, &tokens);
    zero_seed_state(&g_contig, &mut ci);
    let (logits_contig, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = ci
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_contig, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let g_pool = trace_qwen3_moe_prefill_kv_shared_pool(cfg, mp, n, POOL_SLOTS);
    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let mut pi = bind_shared_pool(&g_pool, &tokens, &inv);
    zero_seed_state(&g_pool, &mut pi);
    let (logits_pool, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = pi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_pool, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(
        logits_contig.shape(),
        logits_pool.shape(),
        "qwen3-moe shared-pool vs contiguous prefill: logits shape mismatch"
    );
    let err = max_abs_error(
        logits_contig.as_f32().unwrap(),
        logits_pool.as_f32().unwrap(),
    );
    assert!(
        err < 1e-5,
        "qwen3-moe shared-pool vs contiguous prefill logits differ: max abs err {err}"
    );
}

/// Spec 266-batched Phase 1: `poot_models::mixtral::trace_mixtral_prefill_kv_shared_pool` must be bit-exact against
/// Mixtral's contiguous prefill ([`trace_mixtral_prefill`], the CPU-oracle path with no KV-writing state) for identical
/// weights/tokens, over the same non-contiguous slot layout as the granitemoe/qwen3-moe tests above.
#[test]
fn mixtral_shared_pool_prefill_matches_contiguous() {
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
    let n = TOKENS.len();
    let tokens: Vec<u32> = TOKENS.to_vec();

    let g_contig = trace_mixtral_prefill(cfg, mp, n);
    let ci = bind_contiguous(&g_contig, &tokens);
    let (logits_contig, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = ci
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_contig, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let g_pool = trace_mixtral_prefill_kv_shared_pool(cfg, mp, n, POOL_SLOTS);
    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let mut pi = bind_shared_pool(&g_pool, &tokens, &inv);
    zero_seed_state(&g_pool, &mut pi);
    let (logits_pool, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = pi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_pool, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(
        logits_contig.shape(),
        logits_pool.shape(),
        "mixtral shared-pool vs contiguous prefill: logits shape mismatch"
    );
    let err = max_abs_error(
        logits_contig.as_f32().unwrap(),
        logits_pool.as_f32().unwrap(),
    );
    assert!(
        err < 1e-5,
        "mixtral shared-pool vs contiguous prefill logits differ: max abs err {err}"
    );
}

/// Spec 269 stage 5 (card 274): `poot_models::deepseek3::trace_deepseek3_prefill_kv_shared_pool`, the single-sequence
/// `Slot::SlotMap`-inverse-map-addressed MLA shared-pool prefill tracer, must be bit-exact against DeepSeek-V3's
/// contiguous prefill ([`poot_models::deepseek3::trace_deepseek3_prefill`], no KV-writing state) for identical
/// weights/tokens, over the same non-contiguous slot layout. Reuses [`bind_contiguous`]/[`bind_shared_pool`]: the dense
/// routed-MoE FFN branch ([`poot_models::deepseek3::DeepseekV3MoeParams`] with `first_k_dense_replace: 1`) binds every
/// weight name through the generic `Storage::Const` fallback.
#[test]
fn deepseek3_shared_pool_prefill_matches_contiguous() {
    use poot_models::deepseek2::DeepseekV2Config;
    use poot_models::deepseek3::{
        DeepseekV3MoeParams, trace_deepseek3_prefill, trace_deepseek3_prefill_kv_shared_pool,
    };

    let cfg = DeepseekV2Config {
        vocab: 32,
        hidden: 16,
        layers: 2,
        n_heads: 4,
        q_lora_rank: Some(8),
        kv_lora_rank: 8,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        rope_theta: 10_000.0,
        yarn: None,
    };
    // `first_k_dense_replace: 1` -> layer 0 dense, layer 1 routed: both FFN branches under one fixture (as
    // `poot_llm::deepseek3_mla_batched_kv::tests::moe_params`).
    let mp = DeepseekV3MoeParams {
        n_routed_experts: 4,
        top_k: 2,
        moe_inter: 8,
        n_shared_experts: 1,
        dense_inter: 12,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    };
    let n = TOKENS.len();
    let tokens: Vec<u32> = TOKENS.to_vec();

    let g_contig = trace_deepseek3_prefill(cfg, mp, n);
    let ci = bind_contiguous(&g_contig, &tokens);
    let (logits_contig, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = ci
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_contig, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let g_pool = trace_deepseek3_prefill_kv_shared_pool(cfg, mp, n, POOL_SLOTS);
    let mut inv = vec![-1i32; POOL_SLOTS];
    for (logical, &physical) in GLOBAL_SLOTS.iter().enumerate() {
        inv[physical] = logical as i32;
    }
    let mut pi = bind_shared_pool(&g_pool, &tokens, &inv);
    zero_seed_state(&g_pool, &mut pi);
    let (logits_pool, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = pi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_pool, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(
        logits_contig.shape(),
        logits_pool.shape(),
        "deepseek3 shared-pool vs contiguous prefill: logits shape mismatch"
    );
    let err = max_abs_error(
        logits_contig.as_f32().unwrap(),
        logits_pool.as_f32().unwrap(),
    );
    assert!(
        err < 1e-5,
        "deepseek3 shared-pool vs contiguous prefill logits differ: max abs err {err}"
    );
}

/// Spec 269 stage 5: the admission seam between the MLA shared-pool prefill tracer and the batched shared-pool decode
/// tracer (`trace_deepseek3_decode_kv_masked_batched_shared_pool`). Unlike
/// `gemma4_shared_pool_prefill_matches_sequential_decode_without_disturbing_other_rows` (prefill's last-position logits
/// vs sequential decode, a different computation path needing tolerance), this isolates the seam: admit `l` prompt
/// positions via [`poot_models::deepseek3::trace_deepseek3_prefill_kv_shared_pool`], feed the resulting state directly
/// as [`poot_models::deepseek3::trace_deepseek3_decode_kv_masked_batched_shared_pool`]'s (`batch=1`) carried state, run
/// `m` further decode steps, and compare their logits against a reference that decodes all `l+m` tokens through the
/// same batched (`batch=1`) decode tracer from a zero-seeded pool. Identity (contiguous) slot layout on both sides;
/// non-contiguous layouts are covered by [`deepseek3_shared_pool_prefill_matches_contiguous`] and spec 269 stage 1's
/// SC-001 test.
///
/// Bit-exact (`max_abs_err = 0.0` at both post-seed positions): the compressed latent/RoPE-key state per prompt position
/// has no cross-position reduction (`linear`/`rmsnorm`/interleaved RoPE are per-row independent), so a batched
/// `l`-position prefill and `l` sequential decode calls compute identical per-position state.
#[test]
fn deepseek3_shared_pool_prefill_seeds_batched_decode_matches_full_decode_reference() {
    use poot_models::deepseek2::DeepseekV2Config;
    use poot_models::deepseek3::{
        DeepseekV3MoeParams, trace_deepseek3_decode_kv_masked_batched_shared_pool,
        trace_deepseek3_prefill_kv_shared_pool,
    };

    let cfg = DeepseekV2Config {
        vocab: 32,
        hidden: 16,
        layers: 2,
        n_heads: 4,
        q_lora_rank: Some(8),
        kv_lora_rank: 8,
        qk_nope_head_dim: 4,
        qk_rope_head_dim: 4,
        v_head_dim: 4,
        eps: 1e-6,
        max_pos: 32,
        rope_theta: 10_000.0,
        yarn: None,
    };
    let mp = DeepseekV3MoeParams {
        n_routed_experts: 4,
        top_k: 2,
        moe_inter: 8,
        n_shared_experts: 1,
        dense_inter: 12,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    };

    let l = 4usize; // admitted prompt length
    let m = 2usize; // further decode steps through the batched tracer
    let cap = 8usize; // decode capacity (>= l+m); also this test's prefill pool size (pool == kv_pool_slots)
    let all_tokens: [u32; 6] = [5, 11, 3, 20, 8, 1];
    assert_eq!(all_tokens.len(), l + m);
    let identity_map: Vec<i32> = (0..cap as i32).collect();

    // One weight map, built from the decode graph's Consts; both tracers declare the same per-layer names
    // (`mla_query_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/FFN compositions).
    let g_dec = trace_deepseek3_decode_kv_masked_batched_shared_pool(cfg, mp, cap, 1, cap);
    let mut weights: Map<String, HostTensor> = Map::new();
    for &id in &g_dec.inputs {
        let meta = g_dec.meta(id);
        if let Storage::Const = meta.storage {
            let name = meta.name.clone().unwrap();
            weights
                .entry(name.clone())
                .or_insert_with(|| bind_const(&name, &meta.aval.shape));
        }
    }

    let bind_decode_step = |tok: u32, pos: usize| -> Map<ValueId, HostTensor> {
        let mut inputs: Map<ValueId, HostTensor> = Map::new();
        for &id in &g_dec.inputs {
            let meta = g_dec.meta(id);
            match meta.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, HostTensor::i32(vec![1], vec![tok as i32]));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, HostTensor::i32(vec![1], vec![pos as i32]));
                }
                Storage::Slot(Slot::SeqLen) => {
                    inputs.insert(id, HostTensor::i32(vec![], vec![(pos + 1) as i32]));
                }
                Storage::Slot(Slot::Mask) => {
                    let row: Vec<f32> = (0..cap)
                        .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                        .collect();
                    inputs.insert(id, HostTensor::f32(vec![1, cap], row));
                }
                Storage::Slot(Slot::SlotMap) => {
                    inputs.insert(id, HostTensor::i32(vec![1, cap], identity_map.clone()));
                }
                Storage::Slot(other) => {
                    unreachable!("unexpected slot {other:?} in deepseek3 batched (batch=1) decode")
                }
                Storage::Const => {
                    let name = meta.name.as_deref().unwrap();
                    inputs.insert(id, weights.get(name).cloned().unwrap());
                }
                Storage::Computed(computed) => {
                    inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
                }
                Storage::State => {}
                Storage::Device => unreachable!(),
            }
        }
        inputs
    };

    // --- Reference: l+m sequential steps through the same batched (batch=1) decode tracer, zero-seeded. ---
    let mut ref_state: Vec<HostTensor> = g_dec
        .state
        .iter()
        .map(|&(sid, _)| HostTensor::zeros(g_dec.aval(sid).shape.clone()))
        .collect();
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(l + m);
    for (pos, &tok) in all_tokens.iter().enumerate() {
        let mut inputs = bind_decode_step(tok, pos);
        for (ci, &(sid, _)) in g_dec.state.iter().enumerate() {
            inputs.insert(sid, ref_state[ci].clone());
        }
        let (logits, new_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: std::collections::HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g_dec, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("deepseek3 decode eval (reference)");
        ref_logits.push(logits.as_f32().unwrap().to_vec());
        ref_state = new_state;
    }

    // --- Prefill-then-decode: admit the first l tokens in one shared-pool prefill call (identity slot layout), then
    // continue with m batched-decode steps seeded from the prefill's state. ---
    let g_pre = trace_deepseek3_prefill_kv_shared_pool(cfg, mp, l, cap);
    let mask_full: Vec<f32> = {
        let mut d = vec![-1.0e30f32; l * l];
        for i in 0..l {
            for j in 0..=i {
                d[i * l + j] = 0.0;
            }
        }
        d
    };
    let mut inv = vec![-1i32; cap];
    for (logical, slot) in inv.iter_mut().take(l).enumerate() {
        *slot = logical as i32;
    }
    let mut pre_inputs: Map<ValueId, HostTensor> = Map::new();
    for &id in &g_pre.inputs {
        let meta = g_pre.meta(id);
        match meta.storage {
            Storage::Slot(Slot::Token) => {
                pre_inputs.insert(
                    id,
                    HostTensor::i32(vec![l], all_tokens[..l].iter().map(|&t| t as i32).collect()),
                );
            }
            Storage::Slot(Slot::SlotMap) => {
                pre_inputs.insert(id, HostTensor::i32(vec![cap], inv.clone()));
            }
            Storage::Slot(Slot::Mask) => {
                let name = meta.name.as_deref().unwrap();
                assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                pre_inputs.insert(id, HostTensor::f32(vec![1, 1, l, l], mask_full.clone()));
            }
            Storage::Slot(other) => {
                unreachable!("unexpected slot {other:?} in deepseek3 shared-pool prefill")
            }
            Storage::Const => {
                let name = meta.name.as_deref().unwrap();
                let t = weights
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| panic!("no weight for {name}"));
                pre_inputs.insert(id, t);
            }
            Storage::Computed(computed) => {
                pre_inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {}
            Storage::Device => unreachable!(),
        }
    }
    for &(sid, _) in &g_pre.state {
        pre_inputs.insert(sid, HostTensor::zeros(g_pre.aval(sid).shape.clone()));
    }
    let (_pre_logits, pre_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: std::collections::HashMap<ValueId, Value> = pre_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g_pre, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .expect("deepseek3 shared-pool prefill eval");
    assert_eq!(
        pre_state.len(),
        g_dec.state.len(),
        "prefill and decode must carry the same number of per-layer pool tensors"
    );

    let mut chain_state = pre_state;
    let mut chain_logits: Vec<Vec<f32>> = Vec::with_capacity(m);
    for step in 0..m {
        let pos = l + step;
        let tok = all_tokens[pos];
        let mut inputs = bind_decode_step(tok, pos);
        for (ci, &(sid, _)) in g_dec.state.iter().enumerate() {
            inputs.insert(sid, chain_state[ci].clone());
        }
        let (logits, new_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: std::collections::HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g_dec, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("deepseek3 decode eval (post-prefill)");
        chain_logits.push(logits.as_f32().unwrap().to_vec());
        chain_state = new_state;
    }

    for (step, got) in chain_logits.iter().enumerate().take(m) {
        let pos = l + step;
        let want = &ref_logits[pos];
        let err = max_abs_error(got, want);
        eprintln!(
            "deepseek3_shared_pool_prefill_seeds_batched_decode_matches_full_decode_reference: \
             pos={pos} max_abs_err={err:.3e}"
        );
        assert!(
            err == 0.0,
            "prefill-then-decode vs full-decode-reference diverge at position {pos}: max abs err {err}"
        );
    }
}

//! The granite analog of `olmo2_batched_prefill.rs` (spec 023 sub-step 1 / card 190).
//!
//! Builds the per-layer K/V cache two ways on a tiny synthetic granite-shaped config and asserts
//! near-identity (both are f32 CPU eval of the same primitive ops in a different composition order, so
//! a tight but non-zero tolerance is used, as in `olmo2_batched_prefill.rs`):
//!   (a) batched prefill via `trace_granite_prefill_kv` (one multi-token forward, fills slots [0,N)),
//!   (b) token-by-token via `trace_granite_decode_kv_masked` replayed for pos 0..N.
//!
//! This is the independently derived reference check card 190 requires, aimed at granite's risk
//! surface: the four scalar multipliers (`embed_mult`, `attn_mult` in place of `1/sqrt(head_dim)`,
//! `residual_mult` on each sublayer output before the residual add, `logits_scale` dividing the final
//! logits), and both MLP shapes: dense swiglu (`GraniteParams::moe: None`) and the top-k expert
//! mixture (`Some(MoeShape)`, granitemoe). The MoE case also exercises both `moe` dispatch arms:
//! prefill (`L=n>1`) takes the grouped/`indexed_matmul` path and decode (`L=1`) the sparse path, two
//! graph shapes for the same per-token routing math.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_models::granite::{
    GraniteParams, MoeShape, trace_granite_decode_kv_masked, trace_granite_prefill_kv,
};
use poot_models::qwen2::Qwen2Config;
use poot_tensor::HostTensor;

/// GQA tiny granite config (n_heads != n_kv_heads), following the `olmo2_batched_prefill.rs` /
/// `gemma3_batched_prefill.rs` convention of stressing the n_rep repeat path.
fn tiny() -> Qwen2Config {
    Qwen2Config {
        vocab: 40,
        hidden: 32,
        inter: 48,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        rotary_dim: 8,
        eps: 1e-6,
        max_pos: 64,
        qkv_bias: false, // granite has no qkv/attention bias
        qk_norm: false,  // granite has no qk-norm
        ..Default::default()
    }
}

fn tiny_granite_params(moe: Option<MoeShape>) -> GraniteParams {
    GraniteParams {
        moe,
        embed_mult: 12.0,
        attn_mult: 0.015625,
        residual_mult: 0.22,
        logits_scale: 6.0,
    }
}

/// A small deterministic value for a named weight at flat index `i`, roughly in [-0.5, 0.5); the name
/// is hashed to decorrelate tables so attention/MoE routing is not degenerate.
fn synth(name: &str, i: usize) -> f32 {
    let mut h = poot_test_util::seed_of(name);
    h ^= i as u64;
    h = h.wrapping_mul(1099511628211);
    ((h >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 1.0
}

/// RoPE cos/sin tables [max_pos, head_dim] matching the runner's layout (two halves duplicated).
fn rope_tables(cfg: &Qwen2Config) -> (HostTensor, HostTensor) {
    let d = cfg.head_dim;
    let half = d / 2;
    let p = cfg.max_pos;
    let theta = 10_000.0f32;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| theta.powf(-((2 * j) as f32) / d as f32))
        .collect();
    let mut cos = vec![0.0f32; p * d];
    let mut sin = vec![0.0f32; p * d];
    for pos in 0..p {
        for j in 0..half {
            let ang = pos as f32 * inv_freq[j];
            let (c, s) = (ang.cos(), ang.sin());
            cos[pos * d + j] = c;
            cos[pos * d + half + j] = c;
            sin[pos * d + j] = s;
            sin[pos * d + half + j] = s;
        }
    }
    (
        HostTensor::f32(vec![p, d], cos),
        HostTensor::f32(vec![p, d], sin),
    )
}

/// additive causal mask [1,1,L,L]: 0 where key j <= query i, large-negative above.
fn causal_mask(l: usize) -> HostTensor {
    let mut data = vec![0.0f32; l * l];
    for i in 0..l {
        for j in (i + 1)..l {
            data[i * l + j] = -1e30;
        }
    }
    HostTensor::f32(vec![1, 1, l, l], data)
}

/// The additive decode mask over the full cache: 0 for valid slots t <= pos, -1e9 past pos.
fn decode_mask(cap: usize, pos: usize) -> HostTensor {
    let data = (0..cap)
        .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
        .collect();
    HostTensor::f32(vec![cap], data)
}

struct Env {
    cos: HostTensor,
    sin: HostTensor,
}

fn const_tensor(name: &str, shape: &[usize], env: &Env) -> HostTensor {
    match name {
        "rope.cos" => env.cos.clone(),
        "rope.sin" => env.sin.clone(),
        _ => {
            let n: usize = shape.iter().product::<usize>().max(1);
            let data = (0..n).map(|i| synth(name, i)).collect();
            HostTensor::f32(shape.to_vec(), data)
        }
    }
}

/// Bind every input of a graph. Consts by name (synthetic/computed), the Token/Pos/SeqLen/Mask slots from
/// the args, State buffers from `caches` (in g.state order). `tokens` is the [N] prompt for prefill; for
/// decode it is a single id in tokens[0].
fn bind(
    env: &Env,
    g: &Graph,
    tokens: &[u32],
    pos: usize,
    cap: usize,
    caches: &[HostTensor],
) -> HashMap<ValueId, Value> {
    let mut inputs = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let shape = &meta.aval.shape;
        let t = match meta.storage {
            Storage::Slot(Slot::Token) => {
                if shape.is_empty() {
                    HostTensor::i32(vec![], vec![tokens[0] as i32])
                } else {
                    HostTensor::i32(shape.clone(), tokens.iter().map(|&t| t as i32).collect())
                }
            }
            Storage::Slot(Slot::Activation) => {
                unreachable!("standalone Activation slots are not model prefill inputs")
            }
            // Card 550: see `batched_prefill.rs`'s identical arm.
            Storage::Slot(Slot::Pos) => {
                let tokens_axis = shape.last().copied().unwrap_or(1);
                let row: Vec<i32> = (0..tokens_axis as i32).map(|i| pos as i32 + i).collect();
                HostTensor::i32(shape.clone(), row)
            }
            Storage::Slot(Slot::MropePosition) => {
                unreachable!("granite prefill has no mRoPE position slot")
            }
            Storage::Slot(Slot::SeqLen) => HostTensor::i32(vec![], vec![(pos + 1) as i32]),
            Storage::Slot(Slot::SlotMap) => {
                unreachable!("paged SlotMap binds only on the paged decode path (spec 045)")
            }
            Storage::Slot(Slot::GdnSlotMap) => {
                unreachable!("granite has no GDN state; GdnSlotMap binds only on qwen3next")
            }
            Storage::Slot(Slot::Mask) => {
                let name = meta.name.as_deref().unwrap_or("");
                if name == "mask.prefill" {
                    causal_mask(shape[2])
                } else {
                    decode_mask(cap, pos)
                }
            }
            Storage::Slot(Slot::TokenEmbed) => {
                unreachable!("Slot::TokenEmbed binds only on the wgpu gemma4 decode path")
            }
            Storage::Slot(Slot::LoraIdx) => {
                unreachable!(
                    "LoraIdx binds only on the batched shared-pool LoRA decode path (spec 248 Phase 2)"
                )
            }
            Storage::Slot(Slot::ExpertPoolMap) => {
                unreachable!(
                    "ExpertPoolMap binds only on the pooled-MoE decode path (spec 266 phase 1), and is \
                     resolved per value NAME (Builder::slot_named), never per slot kind"
                )
            }
            Storage::Computed(computed) => HostTensor::f32(computed.shape(), computed.values_f32()),
            Storage::State => continue, // bound below in state order
            Storage::Const => {
                let name = meta.name.as_deref().expect("const has a name");
                const_tensor(name, shape, env)
            }
            Storage::Slot(Slot::Sampler) => {
                unreachable!("Sampler binds only on a graph with a card-551b-appended suffix")
            }
            Storage::Device => panic!("device value in input set"),
        };
        inputs.insert(id, t.into());
    }
    for (ci, &(si, _)) in g.state.iter().enumerate() {
        inputs.insert(si, caches[ci].clone().into());
    }
    inputs
}

/// Shared body: fill the KV cache both ways for the given config/params and assert near-identity of
/// the caches and the final-position logits. Returns (cache_max_abs, logits_max_abs) for the caller
/// to report.
fn run_case(cfg: Qwen2Config, gp: GraniteParams) -> (f32, f32) {
    let n = 5usize; // prompt length
    let cap = n + 3; // a few free slots, as a real generation would leave
    let (cos, sin) = rope_tables(&cfg);
    let env = Env { cos, sin };
    let tokens: Vec<u32> = vec![3, 7, 1, 9, 2];
    assert_eq!(tokens.len(), n);

    // (b) token-by-token via the granite constant-shape masked decode, replayed for pos 0..N: an
    // independently derived reference (single-token graph, one cache write per step, and in the MoE
    // case the `moe_sparse` dispatch arm).
    let gd = trace_granite_decode_kv_masked(cfg, gp, cap);
    let mut tbt_caches: Vec<HostTensor> = gd
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(gd.aval(si).shape.clone()))
        .collect();
    let mut tbt_last_logits = HostTensor::scalar(0.0);
    for pos in 0..n {
        let inputs = bind(&env, &gd, &tokens[pos..pos + 1], pos, cap, &tbt_caches);
        let step = eval(&gd, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("granite decode eval");
        tbt_caches = step
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();
        tbt_last_logits = step.output.into_host().expect("dense output");
    }

    // (a) batched prefill: one forward fills slots [0,N) (the MoE case takes the `moe_grouped` dispatch
    // arm here, since L=n>1).
    let gp_graph = trace_granite_prefill_kv(cfg, gp, n, cap);
    let zero_caches: Vec<HostTensor> = gp_graph
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(gp_graph.aval(si).shape.clone()))
        .collect();
    let inputs = bind(&env, &gp_graph, &tokens, 0, cap, &zero_caches);
    let prefill_step = eval(&gp_graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .expect("granite prefill eval");
    let pf_logits = prefill_step.output.into_host().expect("dense output");
    let pf_caches: Vec<HostTensor> = prefill_step
        .state
        .into_iter()
        .map(|v| v.into_host().expect("dense state"))
        .collect();

    // both graphs carry 2*layers cache buffers in the same (K,V per layer) order.
    assert_eq!(pf_caches.len(), 2 * cfg.layers);
    assert_eq!(pf_caches.len(), tbt_caches.len());

    // the caches must match (every slot, including the unwritten tail which stays zero).
    let mut cache_max_abs = 0.0f32;
    for (ci, (a, b)) in pf_caches.iter().zip(&tbt_caches).enumerate() {
        assert_eq!(a.shape(), b.shape(), "cache {ci} shape");
        let max_abs = poot_test_util::max_abs_error(a.as_f32().unwrap(), b.as_f32().unwrap());
        cache_max_abs = cache_max_abs.max(max_abs);
        assert!(
            max_abs <= 1e-4,
            "cache {ci} differs (max abs {max_abs}) between granite batched prefill and token-by-token fill"
        );
    }

    // the final-position logits must match too (so the next greedy token is unchanged).
    assert_eq!(pf_logits.shape(), tbt_last_logits.shape(), "logits shape");
    let logit_max_abs = poot_test_util::max_abs_error(
        pf_logits.as_f32().unwrap(),
        tbt_last_logits.as_f32().unwrap(),
    );
    assert!(
        logit_max_abs <= 1e-4,
        "final-position logits differ (max abs {logit_max_abs})"
    );

    (cache_max_abs, logit_max_abs)
}

#[test]
fn granite_dense_batched_prefill_fills_cache_identically_to_token_by_token() {
    let cfg = tiny();
    let gp = tiny_granite_params(None);
    run_case(cfg, gp);
}

#[test]
fn granite_moe_batched_prefill_fills_cache_identically_to_token_by_token() {
    let cfg = tiny();
    let moe = MoeShape {
        n_experts: 4,
        top_k: 2,
        inter: 16,
    };
    let gp = tiny_granite_params(Some(moe));
    run_case(cfg, gp);
}

//! Card 135d: `Runner::load`'s DeepSeek-V2 (MLA) safetensors path: loading, `DeepseekV2Params`, and the
//! expert-fusion step in `build_weights` (`fuse_qwen3_moe_experts`, reused a third time). Internal
//! (`#[cfg(test)] mod`, like `mixtral_load_tests`/`olmoe_load_tests`/`gptoss_load_tests` above) to reach
//! the private `deepseek2`/`cfg` fields and the `pub(crate) bind` binder.

use super::super::*;

use poot_eval::Value;
use poot_models::deepseek2::{DeepseekV2Config, DeepseekV2MoeParams, trace_deepseek2_prefill};

use poot_test_util::rmsnorm_ref;

use poot_test_util::silu_ref;

use poot_test_util::linear_ref;

/// DeepSeek's interleaved-pair rotation (see `poot_models::deepseek2`'s module doc, item 6): pairs
/// `(x[2i], x[2i+1])`, not the half-split pairing every other arch's test in this file uses.
fn rope_interleaved_ref(
    row: &[f32],
    cos: &[f32],
    sin: &[f32],
    pos: usize,
    half: usize,
) -> Vec<f32> {
    let c = &cos[pos * half..(pos + 1) * half];
    let s = &sin[pos * half..(pos + 1) * half];
    let mut out = vec![0.0f32; 2 * half];
    for i in 0..half {
        let (a, b) = (row[2 * i], row[2 * i + 1]);
        out[2 * i] = a * c[i] - b * s[i];
        out[2 * i + 1] = a * s[i] + b * c[i];
    }
    out
}

/// DeepSeek-V2 router: `softmax(logits)` over all experts, then `topk_method: "group_limited_greedy"`
/// group-limiting (card 296: `group_scores = scores.view(.., n_group, -1).max(-1)`, keep the `topk_group`
/// highest-scoring groups, mask other experts) ahead of `topk(scores, top_k)` (raw softmax values, not
/// renormalized), then `* routed_scaling_factor`; mirrors `poot_models::deepseek2::deepseek2_router_gate`.
/// The `deepseek2-tiny` fixture (under POOT_MODELS_DIR) sets `n_group: 8`/`topk_group: 3` (a group-limited case,
/// not the `n_group: 1` no-op), so this checks the checkpoint's routing, not just the weight-layout
/// crosswalk.
fn top_k_router_ref(
    logits: &[f32],
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
) -> Vec<(usize, f32)> {
    let e = logits.len();
    assert!(
        e.is_multiple_of(n_group),
        "n_routed_experts must divide n_group"
    );
    let group_size = e / n_group;
    let mut group_scores = vec![f32::MIN; n_group];
    for (g, gs) in group_scores.iter_mut().enumerate() {
        *gs = logits[g * group_size..(g + 1) * group_size]
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max);
    }
    let mut group_idx: Vec<usize> = (0..n_group).collect();
    group_idx.sort_by(|&a, &b| group_scores[b].partial_cmp(&group_scores[a]).unwrap());
    let selected_groups: std::collections::HashSet<usize> =
        group_idx[..topk_group].iter().copied().collect();

    let mut idx: Vec<usize> = (0..e)
        .filter(|&i| selected_groups.contains(&(i / group_size)))
        .collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let sel = &idx[..top_k];
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exp_sel: Vec<f32> = sel.iter().map(|&i| (logits[i] - m).exp()).collect();
    let denom_full: f32 = logits.iter().map(|&v| (v - m).exp()).sum();
    sel.iter()
        .zip(exp_sel.iter())
        .map(|(&i, &ex)| (i, (ex / denom_full) * routed_scaling_factor))
        .collect()
}

/// Independent, from-scratch Rust reference forward pass reading an already-loaded `Runner::weights` map
/// (the production `build_weights`/`fuse_qwen3_moe_experts` output). Like `poot_models::deepseek2`'s
/// `deepseek2_prefill_ref` (a direct loop, not a copy of the traced decomposition), but it exercises the
/// production loader path: MLA compress/decompress with the decoupled interleaved-RoPE split, and the
/// routed+shared MoE MLP (real fixture: every layer from first_k_dense_replace=1 routes; layer 0 stays
/// dense).
fn deepseek2_prefill_ref_from_runner(
    dcfg: &DeepseekV2Config,
    mp: &DeepseekV2MoeParams,
    tokens: &[u32],
    w: &HashMap<String, Value>,
) -> Vec<f32> {
    let g = |name: &str| -> &[f32] { w[name].as_host().expect("dense weight").as_f32().unwrap() };
    let h = dcfg.hidden;
    let hq = dcfg.n_heads;
    let (nope, rope_d, vd) = (
        dcfg.qk_nope_head_dim,
        dcfg.qk_rope_head_dim,
        dcfg.v_head_dim,
    );
    let qk_head_dim = dcfg.qk_head_dim();
    let kv_rank = dcfg.kv_lora_rank;
    let scale = dcfg.attn_scale();
    let l = tokens.len();
    let embed = g("model.embed_tokens.weight");
    let cos = g("rope.cos");
    let sin = g("rope.sin");
    let half = rope_d / 2;

    let mut x: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| embed[t as usize * h..(t as usize + 1) * h].to_vec())
        .collect();

    for li in 0..dcfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1w = g(&p("input_layernorm.weight"));
        let normed: Vec<Vec<f32>> = x
            .iter()
            .map(|row| rmsnorm_ref(row, ln1w, h, dcfg.eps))
            .collect();

        let q_flat: Vec<Vec<f32>> = match dcfg.q_lora_rank {
            Some(r) => {
                let wqa = g(&p("self_attn.q_a_proj.weight"));
                let wqaln = g(&p("self_attn.q_a_layernorm.weight"));
                let wqb = g(&p("self_attn.q_b_proj.weight"));
                normed
                    .iter()
                    .map(|row| {
                        let qa = linear_ref(row, wqa, h, r);
                        let qa_n = rmsnorm_ref(&qa, wqaln, r, dcfg.eps);
                        linear_ref(&qa_n, wqb, r, hq * qk_head_dim)
                    })
                    .collect()
            }
            None => {
                let wq = g(&p("self_attn.q_proj.weight"));
                normed
                    .iter()
                    .map(|row| linear_ref(row, wq, h, hq * qk_head_dim))
                    .collect()
            }
        };
        let mut q_full: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0f32; qk_head_dim]; hq]; l];
        for (pos, qrow) in q_flat.iter().enumerate() {
            for (hh, q_slot) in q_full[pos].iter_mut().enumerate() {
                let base = hh * qk_head_dim;
                let q_nope = &qrow[base..base + nope];
                let q_pe = &qrow[base + nope..base + qk_head_dim];
                let q_pe_rot = rope_interleaved_ref(q_pe, cos, sin, pos, half);
                q_slot[..nope].copy_from_slice(q_nope);
                q_slot[nope..].copy_from_slice(&q_pe_rot);
            }
        }

        let wkva = g(&p("self_attn.kv_a_proj_with_mqa.weight"));
        let wkvaln = g(&p("self_attn.kv_a_layernorm.weight"));
        let wkvb = g(&p("self_attn.kv_b_proj.weight"));
        let mut c_kv_cache: Vec<Vec<f32>> = Vec::with_capacity(l);
        let mut k_pe_cache: Vec<Vec<f32>> = Vec::with_capacity(l);
        for (pos, row) in normed.iter().enumerate() {
            let kva = linear_ref(row, wkva, h, kv_rank + rope_d);
            let kv_nope_raw = &kva[..kv_rank];
            let k_pe_raw = &kva[kv_rank..];
            let c_kv = rmsnorm_ref(kv_nope_raw, wkvaln, kv_rank, dcfg.eps);
            let k_pe_rot = rope_interleaved_ref(k_pe_raw, cos, sin, pos, half);
            c_kv_cache.push(c_kv);
            k_pe_cache.push(k_pe_rot);
        }

        let mut attn_out = vec![vec![0.0f32; hq * vd]; l];
        for qpos in 0..l {
            let s = qpos + 1;
            let mut k_nope = vec![vec![vec![0.0f32; nope]; s]; hq];
            let mut v = vec![vec![vec![0.0f32; vd]; s]; hq];
            for (t, c) in c_kv_cache[..s].iter().enumerate() {
                let expanded = linear_ref(c, wkvb, kv_rank, hq * (nope + vd));
                for hh in 0..hq {
                    let base = hh * (nope + vd);
                    k_nope[hh][t].copy_from_slice(&expanded[base..base + nope]);
                    v[hh][t].copy_from_slice(&expanded[base + nope..base + nope + vd]);
                }
            }
            for hh in 0..hq {
                let mut scores = vec![0.0f32; s];
                for (t, sc) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for i in 0..nope {
                        dot += q_full[qpos][hh][i] * k_nope[hh][t][i];
                    }
                    for i in 0..rope_d {
                        dot += q_full[qpos][hh][nope + i] * k_pe_cache[t][i];
                    }
                    *sc = dot * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                let mut e = vec![0.0f32; s];
                for (t, sc) in scores.iter().enumerate() {
                    e[t] = (sc - m).exp();
                    denom += e[t];
                }
                for i in 0..vd {
                    let mut acc = 0.0f32;
                    for (t, ei) in e.iter().enumerate() {
                        acc += (ei / denom) * v[hh][t][i];
                    }
                    attn_out[qpos][hh * vd + i] = acc;
                }
            }
        }

        let wo = g(&p("self_attn.o_proj.weight"));
        for pos in 0..l {
            let proj = linear_ref(&attn_out[pos], wo, hq * vd, h);
            for c in 0..h {
                x[pos][c] += proj[c];
            }
        }

        let ln2w = g(&p("post_attention_layernorm.weight"));
        for row in x.iter_mut().take(l) {
            let normed2 = rmsnorm_ref(row, ln2w, h, dcfg.eps);
            let ffn_out = if mp.is_moe_layer(li) {
                let router_w = g(&p("mlp.gate.weight"));
                let gate_up = g(&p("mlp.experts.gate_up_proj.weight"));
                let down = g(&p("mlp.experts.down_proj.weight"));
                let logits = linear_ref(&normed2, router_w, h, mp.n_routed_experts);
                let selected = top_k_router_ref(
                    &logits,
                    mp.top_k,
                    mp.n_group,
                    mp.topk_group,
                    mp.routed_scaling_factor,
                );
                let inter = mp.moe_inter;
                let mut moe_out = vec![0.0f32; h];
                for (e, gate_weight) in selected {
                    let e_gate_up = &gate_up[e * h * 2 * inter..(e + 1) * h * 2 * inter];
                    let mut gate_w = vec![0.0f32; h * inter];
                    let mut up_w = vec![0.0f32; h * inter];
                    for row_i in 0..h {
                        let src = row_i * 2 * inter;
                        gate_w[row_i * inter..(row_i + 1) * inter]
                            .copy_from_slice(&e_gate_up[src..src + inter]);
                        up_w[row_i * inter..(row_i + 1) * inter]
                            .copy_from_slice(&e_gate_up[src + inter..src + 2 * inter]);
                    }
                    let e_down = &down[e * inter * h..(e + 1) * inter * h];
                    let gate = linear_ref(&normed2, &gate_w, h, inter);
                    let up = linear_ref(&normed2, &up_w, h, inter);
                    let act: Vec<f32> = gate
                        .iter()
                        .zip(up.iter())
                        .map(|(&gg, &uu)| silu_ref(gg) * uu)
                        .collect();
                    let expert_out = linear_ref(&act, e_down, inter, h);
                    for c in 0..h {
                        moe_out[c] += gate_weight * expert_out[c];
                    }
                }
                if mp.n_shared_experts > 0 {
                    let shared_inter = mp.moe_inter * mp.n_shared_experts;
                    let wsg = g(&p("mlp.shared_experts.gate_proj.weight"));
                    let wsu = g(&p("mlp.shared_experts.up_proj.weight"));
                    let wsd = g(&p("mlp.shared_experts.down_proj.weight"));
                    let sg = linear_ref(&normed2, wsg, h, shared_inter);
                    let su = linear_ref(&normed2, wsu, h, shared_inter);
                    let sact: Vec<f32> = sg
                        .iter()
                        .zip(su.iter())
                        .map(|(&gg, &uu)| silu_ref(gg) * uu)
                        .collect();
                    let shared_out = linear_ref(&sact, wsd, shared_inter, h);
                    for c in 0..h {
                        moe_out[c] += shared_out[c];
                    }
                }
                moe_out
            } else {
                let wg = g(&p("mlp.gate_proj.weight"));
                let wu = g(&p("mlp.up_proj.weight"));
                let wd = g(&p("mlp.down_proj.weight"));
                let gg = linear_ref(&normed2, wg, h, mp.dense_inter);
                let uu = linear_ref(&normed2, wu, h, mp.dense_inter);
                let act: Vec<f32> = gg
                    .iter()
                    .zip(uu.iter())
                    .map(|(&a, &b)| silu_ref(a) * b)
                    .collect();
                linear_ref(&act, wd, mp.dense_inter, h)
            };
            for c in 0..h {
                row[c] += ffn_out[c];
            }
        }
    }

    let ln_f_w = g("model.norm.weight");
    let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, dcfg.eps);
    let lm_head = g("lm_head.weight");
    linear_ref(&last, lm_head, h, dcfg.vocab)
}

/// Proves `Runner::load`'s MLA weight-layout crosswalk (attention: q_a/q_b/kv_a/kv_b/o are transposed
/// generically since they end in "proj.weight", so no special loader code; MoE: router transpose +
/// `fuse_qwen3_moe_experts`) on a real checkpoint: the traced-and-bound `trace_deepseek2_prefill` graph,
/// evaluated through the production `Runner::load` + `build_weights` + `bind` path, must match an
/// independent from-scratch Rust reference (`deepseek2_prefill_ref_from_runner`) reading the same weight
/// map. A transpose/fuse/MLA-decompress bug would give a materially different output. Card 296: the
/// fixture sets `n_group: 8`/`topk_group: 3` (group-limited routing, unlike the `n_group: 1`
/// no-op of DeepSeek-V2-Lite), so this also checks `deepseek2_router_gate`'s group-limited routing.
/// Needs `deepseek2-tiny` under POOT_MODELS_DIR (`hf download yujiepan/deepseek-v2-tiny-random
/// --local-dir <POOT_MODELS_DIR>/deepseek2-tiny` to populate it).
#[test]
fn deepseek2_tiny_checkpoint_runner_load_matches_hand_rolled_reference() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-tiny"))
    else {
        return;
    };
    let runner = Runner::load(&dir).expect("load deepseek2-tiny via Runner");
    assert_eq!(runner.arch, "deepseek_v2");
    let dp = runner
        .deepseek2
        .expect("Runner::load must set deepseek2 params for a deepseek_v2 checkpoint");
    assert_eq!(
        dp.cfg.q_lora_rank,
        Some(2),
        "the tiny fixture DOES set q_lora_rank (unlike real V2-Lite)"
    );
    assert_eq!(dp.cfg.kv_lora_rank, 2);
    assert_eq!(dp.cfg.qk_nope_head_dim, 2);
    assert_eq!(dp.cfg.qk_rope_head_dim, 2);
    assert_eq!(dp.cfg.v_head_dim, 2);
    assert_eq!(dp.moe.n_routed_experts, 160);
    assert_eq!(dp.moe.top_k, 6);
    assert_eq!(dp.moe.n_shared_experts, 2);
    assert_eq!(dp.moe.first_k_dense_replace, 1);
    assert_eq!(dp.moe.routed_scaling_factor, 16.0);
    assert_eq!(
        dp.moe.n_group, 8,
        "real group-limited routing, not the n_group=1 no-op case"
    );
    assert_eq!(dp.moe.topk_group, 3);

    let tokens = [1u32, 2, 3, 4];
    let g = trace_deepseek2_prefill(dp.cfg, dp.moe, tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind deepseek2 prefill graph");
    let out = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");
    assert_eq!(out.shape(), vec![1, 1, dp.cfg.vocab]);
    assert!(out.as_f32().unwrap().iter().all(|v| v.is_finite()));

    let want = deepseek2_prefill_ref_from_runner(&dp.cfg, &dp.moe, &tokens, &runner.weights);
    assert_eq!(out.as_f32().unwrap().len(), want.len());
    // Runner-loaded output (actual) vs the hand-rolled reference (expected).
    poot_test_util::assert_close_rel(out.as_f32().unwrap(), &want, 1e-4);
}

/// `Runner::load_gguf`'s deepseek2 MLA support (card 135d, spec 264), checked against the verified
/// safetensors path on the same real checkpoint (executor equivalence, per AGENTS.md's verification
/// stack), like `mixtral_load_tests::mixtral_tiny_gguf_matches_safetensors`/
/// `olmoe_load_tests::olmoe_tiny_gguf_matches_safetensors`. Uses explicit tokens through `bind`+`eval`,
/// not `generate()`, to avoid the BOS-handling asymmetry between the loaders (safetensors hardcodes
/// `bos: None`; the GGUF path honors `add_bos_token`).
///
/// The GGUF is a real llama.cpp conversion (`convert_hf_to_gguf.py --outtype f32`, in a throwaway `uv`
/// venv) of the same `deepseek2-tiny` checkpoint, saved beside it as `deepseek2-tiny-f32.gguf`
/// (not committed). This is the only test that exercises `reconstruct_deepseek2_kv_b` (`gguf.rs`) against
/// real converted weights; a transpose/reconstruction bug would give a materially different forward pass
/// (both paths are f32 with bit-identical weights). It also exercises the YaRN-scaling GGUF metadata
/// crosswalk (`gguf_deepseek2_yarn_params`), since `deepseek2-tiny`'s config sets
/// `rope_scaling.type: "yarn"`.
#[test]
fn deepseek2_tiny_gguf_matches_safetensors() {
    let Some(st_dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek2-tiny"))
    else {
        return;
    };
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek2-tiny/deepseek2-tiny-f32.gguf"
    )) else {
        return;
    };
    let st_runner = Runner::load(&st_dir).expect("load deepseek2-tiny safetensors");
    let gguf_runner = Runner::load_gguf(&gguf_path).expect("load deepseek2-tiny gguf");
    assert_eq!(st_runner.arch, "deepseek_v2");
    assert_eq!(gguf_runner.arch, "deepseek_v2");

    let st_dp = st_runner
        .deepseek2
        .expect("safetensors Runner must set deepseek2 params");
    let gguf_dp = gguf_runner
        .deepseek2
        .expect("gguf Runner must set deepseek2 params (card 135d GGUF follow-on)");
    assert_eq!(st_dp.cfg.q_lora_rank, gguf_dp.cfg.q_lora_rank);
    assert_eq!(st_dp.cfg.kv_lora_rank, gguf_dp.cfg.kv_lora_rank);
    assert_eq!(st_dp.cfg.qk_nope_head_dim, gguf_dp.cfg.qk_nope_head_dim);
    assert_eq!(st_dp.cfg.qk_rope_head_dim, gguf_dp.cfg.qk_rope_head_dim);
    assert_eq!(st_dp.cfg.v_head_dim, gguf_dp.cfg.v_head_dim);
    assert_eq!(st_dp.moe.n_routed_experts, gguf_dp.moe.n_routed_experts);
    assert_eq!(st_dp.moe.top_k, gguf_dp.moe.top_k);
    assert_eq!(st_dp.moe.n_shared_experts, gguf_dp.moe.n_shared_experts);
    assert_eq!(
        st_dp.moe.first_k_dense_replace,
        gguf_dp.moe.first_k_dense_replace
    );
    assert_eq!(
        st_dp.moe.routed_scaling_factor,
        gguf_dp.moe.routed_scaling_factor
    );
    assert_eq!(st_dp.moe.n_group, gguf_dp.moe.n_group);
    assert_eq!(st_dp.moe.topk_group, gguf_dp.moe.topk_group);
    // YaRN scaling round-trips through the GGUF path too (see `gguf_deepseek2_yarn_params`): GGUF's
    // `yarn_log_multiplier` recovers `mscale_all_dim` exactly and `mscale` is passed as the same value
    // (matching the real checkpoint's mscale==mscale_all_dim case), so the resolved scalars should match
    // the safetensors path closely, not exactly: `attn_factor` is absent from the GGUF, a documented
    // approximation.
    let st_yarn = st_dp
        .cfg
        .yarn
        .expect("safetensors deepseek2-tiny sets real YaRN scaling");
    let gguf_yarn = gguf_dp
        .cfg
        .yarn
        .expect("gguf deepseek2-tiny sets real YaRN scaling");
    assert!((st_yarn.factor - gguf_yarn.factor).abs() < 1e-6);
    assert!((st_yarn.softmax_mscale_sq - gguf_yarn.softmax_mscale_sq).abs() < 1e-3);
    assert!((st_yarn.attention_factor - gguf_yarn.attention_factor).abs() < 1e-3);

    let tokens = [1u32, 2, 3, 4];
    let st_g = trace_deepseek2_prefill(st_dp.cfg, st_dp.moe, tokens.len());
    let st_inputs = st_runner
        .bind(&st_g, &tokens)
        .expect("bind safetensors prefill graph");
    let st_out =
        crate::core::cpu_oracle::cpu_eval(&st_g, &st_inputs).expect("cpu eval (safetensors)");

    let gguf_g = trace_deepseek2_prefill(gguf_dp.cfg, gguf_dp.moe, tokens.len());
    let gguf_inputs = gguf_runner
        .bind(&gguf_g, &tokens)
        .expect("bind gguf prefill graph");
    let gguf_out =
        crate::core::cpu_oracle::cpu_eval(&gguf_g, &gguf_inputs).expect("cpu eval (gguf)");

    assert_eq!(st_out.shape(), gguf_out.shape());
    assert!(st_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert!(gguf_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // Safetensors vs gguf last-position logits: a divergence here is a real transpose/reconstruct bug in
    // the new deepseek2 GGUF loader (gguf.rs), not float noise (both paths are f32, no quantization).
    poot_test_util::assert_close(st_out.as_f32().unwrap(), gguf_out.as_f32().unwrap(), 1e-3);
}

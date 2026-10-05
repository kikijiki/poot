//! Card 135d (spec 261): `Runner::load`'s mixtral safetensors path: loading, `MixtralParams`, and the
//! expert-fusion step in `build_weights` (`fuse_mixtral_experts`). Internal (`#[cfg(test)] mod`, like
//! `qwen3_moe_load_tests` above) to reach the private `mixtral`/`cfg` fields and the `pub(crate) bind`
//! binder.

use super::super::*;

use poot_eval::Value;
use poot_models::mixtral::trace_mixtral_prefill;

use poot_test_util::rmsnorm_ref;

use poot_test_util::silu_ref;

use poot_test_util::linear_ref;

use poot_test_util::rope_ref;

fn top_k_router_ref(logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let sel = &idx[..top_k];
    let m = sel.iter().map(|&i| logits[i]).fold(f32::MIN, f32::max);
    let exp: Vec<f32> = sel.iter().map(|&i| (logits[i] - m).exp()).collect();
    let denom: f32 = exp.iter().sum();
    sel.iter()
        .zip(exp.iter())
        .map(|(&i, &e)| (i, e / denom))
        .collect()
}

/// Independent, from-scratch Rust reference forward pass reading an already-loaded `Runner::weights`
/// map (the production `build_weights`/`fuse_mixtral_experts` output). Like `poot_models::mixtral`'s
/// `mixtral_prefill_ref` (a direct loop, not a copy of the traced graph's `moe` op decomposition), but
/// it exercises the production loader path rather than a test-only fuse routine.
fn mixtral_prefill_ref_from_runner(
    cfg: &Qwen2Config,
    mp: &MixtralParams,
    tokens: &[u32],
    w: &HashMap<String, Value>,
) -> Vec<f32> {
    let g = |name: &str| -> &[f32] { w[name].as_host().expect("dense weight").as_f32().unwrap() };
    let (h, d, hq, hkv, l) = (
        cfg.hidden,
        cfg.head_dim,
        cfg.n_heads,
        cfg.n_kv_heads,
        tokens.len(),
    );
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let embed = g("model.embed_tokens.weight");
    let cos = g("rope.cos");
    let sin = g("rope.sin");

    let mut x: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| embed[t as usize * h..(t as usize + 1) * h].to_vec())
        .collect();

    for li in 0..cfg.layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let ln1w = g(&p("input_layernorm.weight"));
        let normed: Vec<Vec<f32>> = x
            .iter()
            .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
            .collect();

        let qw = g(&p("self_attn.q_proj.weight"));
        let kw = g(&p("self_attn.k_proj.weight"));
        let vw = g(&p("self_attn.v_proj.weight"));
        let mut q: Vec<Vec<f32>> = normed
            .iter()
            .map(|row| linear_ref(row, qw, h, hq * d))
            .collect();
        let mut k: Vec<Vec<f32>> = normed
            .iter()
            .map(|row| linear_ref(row, kw, h, hkv * d))
            .collect();
        let v: Vec<Vec<f32>> = normed
            .iter()
            .map(|row| linear_ref(row, vw, h, hkv * d))
            .collect();

        for (pos, qrow) in q.iter_mut().enumerate() {
            for hh in 0..hq {
                let rotated = rope_ref(&qrow[hh * d..(hh + 1) * d], cos, sin, pos, d);
                qrow[hh * d..(hh + 1) * d].copy_from_slice(&rotated);
            }
        }
        for (pos, krow) in k.iter_mut().enumerate() {
            for hh in 0..hkv {
                let rotated = rope_ref(&krow[hh * d..(hh + 1) * d], cos, sin, pos, d);
                krow[hh * d..(hh + 1) * d].copy_from_slice(&rotated);
            }
        }

        let mut attn_out = vec![vec![0.0f32; h]; l];
        for qh in 0..hq {
            let kh = qh / n_rep;
            for i in 0..l {
                let mut scores = vec![0.0f32; i + 1];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q[i][qh * d + dd] * k[j][kh * d + dd];
                    }
                    *sc = s * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0f32;
                let mut e = vec![0.0f32; scores.len()];
                for (j, sc) in scores.iter().enumerate() {
                    e[j] = (sc - m).exp();
                    denom += e[j];
                }
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for (j, ej) in e.iter().enumerate() {
                        acc += (ej / denom) * v[j][kh * d + dd];
                    }
                    attn_out[i][qh * d + dd] = acc;
                }
            }
        }

        let ow = g(&p("self_attn.o_proj.weight"));
        for i in 0..l {
            let proj = linear_ref(&attn_out[i], ow, hq * d, h);
            for c in 0..h {
                x[i][c] += proj[c];
            }
        }

        let ln2w = g(&p("post_attention_layernorm.weight"));
        let router_w = g(&p("block_sparse_moe.gate.weight"));
        let gate_up = g(&p("block_sparse_moe.experts.gate_up_proj.weight"));
        let down = g(&p("block_sparse_moe.experts.down_proj.weight"));
        for row in x.iter_mut().take(l) {
            let normed = rmsnorm_ref(row, ln2w, h, cfg.eps);
            let logits = linear_ref(&normed, router_w, h, mp.n_experts);
            let selected = top_k_router_ref(&logits, mp.top_k);
            let mut moe_out = vec![0.0f32; h];
            for (e, gate_weight) in selected {
                let e_gate_up = &gate_up[e * h * 2 * mp.inter..(e + 1) * h * 2 * mp.inter];
                let mut gate_w = vec![0.0f32; h * mp.inter];
                let mut up_w = vec![0.0f32; h * mp.inter];
                for row_i in 0..h {
                    let src = row_i * 2 * mp.inter;
                    gate_w[row_i * mp.inter..(row_i + 1) * mp.inter]
                        .copy_from_slice(&e_gate_up[src..src + mp.inter]);
                    up_w[row_i * mp.inter..(row_i + 1) * mp.inter]
                        .copy_from_slice(&e_gate_up[src + mp.inter..src + 2 * mp.inter]);
                }
                let e_down = &down[e * mp.inter * h..(e + 1) * mp.inter * h];
                let gate = linear_ref(&normed, &gate_w, h, mp.inter);
                let up = linear_ref(&normed, &up_w, h, mp.inter);
                let act: Vec<f32> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(&gv, &uv)| silu_ref(gv) * uv)
                    .collect();
                let expert_out = linear_ref(&act, e_down, mp.inter, h);
                for c in 0..h {
                    moe_out[c] += gate_weight * expert_out[c];
                }
            }
            for (c, mv) in moe_out.iter().enumerate() {
                row[c] += mv;
            }
        }
    }

    let ln_f_w = g("model.norm.weight");
    let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
    let lm_head = g("lm_head.weight");
    linear_ref(&last, lm_head, h, cfg.vocab)
}

/// Proves `Runner::load`'s weight-layout crosswalk (transpose + per-expert fuse,
/// `fuse_mixtral_experts`) on a real checkpoint: the traced-and-bound `trace_mixtral_prefill` graph,
/// evaluated through the production `Runner::load` + `build_weights` + `bind` path, must match an
/// independent from-scratch Rust reference (`mixtral_prefill_ref_from_runner`) reading the same
/// production-loaded weight map. A transpose/fuse bug would give a materially different output (both
/// paths read the same `runner.weights`). Needs `mixtral-tiny` under POOT_MODELS_DIR (`hf download
/// optimum-intel-internal-testing/tiny-mixtral --local-dir <POOT_MODELS_DIR>/mixtral-tiny` to populate it).
#[test]
fn mixtral_tiny_checkpoint_runner_load_matches_hand_rolled_reference() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny")) else {
        return;
    };
    let runner = Runner::load(&dir).expect("load mixtral-tiny via Runner");
    assert_eq!(runner.arch, "mixtral");
    let mp = runner
        .mixtral
        .expect("Runner::load must set mixtral params for a mixtral checkpoint");
    assert_eq!(mp.n_experts, 8);
    assert_eq!(mp.top_k, 2);
    assert_eq!(mp.inter, 3584);

    let tokens = [1u32, 2, 3, 4];
    let g = trace_mixtral_prefill(runner.cfg, mp, tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind mixtral prefill graph");
    let out = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");
    assert_eq!(out.shape(), vec![1, 1, runner.cfg.vocab]);
    assert!(out.as_f32().unwrap().iter().all(|v| v.is_finite()));

    let want = mixtral_prefill_ref_from_runner(&runner.cfg, &mp, &tokens, &runner.weights);
    assert_eq!(out.as_f32().unwrap().len(), want.len());
    // Runner-loaded output (actual) vs the hand-rolled reference (expected).
    poot_test_util::assert_close_rel(out.as_f32().unwrap(), &want, 1e-4);
}

/// `Runner::load_gguf`'s Mixtral detection (gguf.rs/runner.rs: a "llama"-arch GGUF whose layer 0 has
/// `ffn_gate_inp.weight`), cross-checked against `Runner::load`'s verified safetensors path on the same
/// real checkpoint (executor equivalence, per AGENTS.md's verification stack), like
/// `qwen3_moe_gguf_matches_safetensors_on_self_authored_sparse_checkpoint` above. Uses explicit tokens
/// through `bind`+`eval`, not `generate()`: `generate()`/`encode()` would hit the BOS-handling asymmetry
/// between the loaders (safetensors hardcodes `bos: None`; the GGUF path honors the file's
/// `add_bos_token`, which `tiny-mixtral`'s conversion sets to true), and the extra leading BOS would
/// change the attention context independent of any weight-loading bug.
///
/// The GGUF is a real llama.cpp conversion (`convert_hf_to_gguf.py`) of the same
/// `optimum-intel-internal-testing/tiny-mixtral` checkpoint, saved beside it as `mixtral-tiny-f32.gguf`
/// (not committed; a local model-cache artifact). A transpose/fuse bug in the GGUF Mixtral arm
/// (`gguf_weights` in `gguf.rs`'s `arch == "llama"` MoE branch) would give a materially different
/// forward pass, since both paths are f32 with bit-identical weights.
#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_tiny_gguf_matches_safetensors() {
    let Some(st_dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny"))
    else {
        return;
    };
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-tiny/mixtral-tiny-f32.gguf"
    )) else {
        return;
    };
    let st_runner = Runner::load(&st_dir).expect("load mixtral-tiny safetensors");
    let gguf_runner = Runner::load_gguf(&gguf_path).expect("load mixtral-tiny gguf");
    assert_eq!(st_runner.arch, "mixtral");
    assert_eq!(gguf_runner.arch, "mixtral");

    let st_mp = st_runner
        .mixtral
        .expect("safetensors Runner must set mixtral params");
    let gguf_mp = gguf_runner
        .mixtral
        .expect("gguf Runner must set mixtral params (card 135d GGUF follow-on)");
    assert_eq!(st_mp.n_experts, gguf_mp.n_experts);
    assert_eq!(st_mp.top_k, gguf_mp.top_k);
    assert_eq!(st_mp.inter, gguf_mp.inter);

    let tokens = [3u32, 7, 11, 2];
    let st_g = trace_mixtral_prefill(st_runner.cfg, st_mp, tokens.len());
    let st_inputs = st_runner
        .bind(&st_g, &tokens)
        .expect("bind safetensors prefill graph");
    let st_out =
        crate::core::cpu_oracle::cpu_eval(&st_g, &st_inputs).expect("cpu eval (safetensors)");

    let gguf_g = trace_mixtral_prefill(gguf_runner.cfg, gguf_mp, tokens.len());
    let gguf_inputs = gguf_runner
        .bind(&gguf_g, &tokens)
        .expect("bind gguf prefill graph");
    let gguf_out =
        crate::core::cpu_oracle::cpu_eval(&gguf_g, &gguf_inputs).expect("cpu eval (gguf)");

    assert_eq!(st_out.shape(), gguf_out.shape());
    assert!(st_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert!(gguf_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // Safetensors vs gguf last-position logits: a divergence here is a real transpose/fuse bug in the new
    // Mixtral GGUF arm, not float noise (both paths are f32, no quantization involved).
    poot_test_util::assert_close(st_out.as_f32().unwrap(), gguf_out.as_f32().unwrap(), 1e-3);
    let argmax = |data: &[f32]| -> usize {
        data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap()
    };
    assert_eq!(
        argmax(st_out.as_f32().unwrap()),
        argmax(gguf_out.as_f32().unwrap()),
        "safetensors and gguf loaders must agree on the argmax token"
    );
}

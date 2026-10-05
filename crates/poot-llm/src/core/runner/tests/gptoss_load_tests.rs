use super::super::*;

use poot_eval::Value;
use poot_models::gpt_oss::trace_gptoss_prefill;

use poot_test_util::rmsnorm_ref;

use poot_graph_plan::passes_without_target as optimize;

fn sigmoid_ref(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

fn clamp_ref(v: f32, lo: Option<f32>, hi: Option<f32>) -> f32 {
    let mut x = v;
    if let Some(lo) = lo {
        x = x.max(lo);
    }
    if let Some(hi) = hi {
        x = x.min(hi);
    }
    x
}

fn linear_ref(
    x: &[f32],
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_dim {
            acc += x[i] * w[i * out_dim + o];
        }
        y[o] = acc;
    }
    y
}

use poot_test_util::rope_ref;

/// gpt-oss's router, mathematically identical to `top_k_gate`/Mixtral's (see `poot_models::gpt_oss`):
/// topk on raw (biased) logits, then softmax over just the selected values.
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
/// map (the production `build_weights` output, including the YaRN-scaled `rope.cos`/`rope.sin` table,
/// computed generically for every arch via `hf.rope_scaling`; so this exercises YaRN end to end, unlike
/// `poot_models::gpt_oss`'s real-checkpoint test, which uses a plain-theta table). Like
/// `poot_models::gpt_oss::gptoss_prefill_ref` (a direct loop, not a copy of the traced
/// `gptoss_ffn`/attention-sink decomposition), but it exercises the production loader path: biased
/// Q/K/V/O, attention sinks, the per-layer alternating sliding-window/full schedule, a biased router,
/// and the biased clamped-GLU expert MLP.
fn gptoss_prefill_ref_from_runner(
    cfg: &Qwen2Config,
    mp: &GptOssParams,
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
        let is_local = mp.layer_is_sliding[li];
        let ln1w = g(&p("input_layernorm.weight"));
        let normed: Vec<Vec<f32>> = x
            .iter()
            .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
            .collect();

        let qw = g(&p("self_attn.q_proj.weight"));
        let qb = g(&p("self_attn.q_proj.bias"));
        let kw = g(&p("self_attn.k_proj.weight"));
        let kb = g(&p("self_attn.k_proj.bias"));
        let vw = g(&p("self_attn.v_proj.weight"));
        let vb = g(&p("self_attn.v_proj.bias"));
        let mut q: Vec<Vec<f32>> = normed
            .iter()
            .map(|row| linear_ref(row, qw, h, hq * d, Some(qb)))
            .collect();
        let mut k: Vec<Vec<f32>> = normed
            .iter()
            .map(|row| linear_ref(row, kw, h, hkv * d, Some(kb)))
            .collect();
        let v: Vec<Vec<f32>> = normed
            .iter()
            .map(|row| linear_ref(row, vw, h, hkv * d, Some(vb)))
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

        let sinks = g(&p("self_attn.sinks"));
        let mut attn_out = vec![vec![0.0f32; hq * d]; l];
        for qh in 0..hq {
            let kh = qh / n_rep;
            for i in 0..l {
                let start = if is_local {
                    i.saturating_sub(mp.sliding_window - 1)
                } else {
                    0
                };
                let mut scores = vec![0.0f32; i - start + 1];
                for (jj, sc) in scores.iter_mut().enumerate() {
                    let j = start + jj;
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q[i][qh * d + dd] * k[j][kh * d + dd];
                    }
                    *sc = s * scale;
                }
                let sink = sinks[qh];
                let m = scores.iter().cloned().fold(sink, f32::max);
                let mut denom = (sink - m).exp();
                let mut e = vec![0.0f32; scores.len()];
                for (jj, sc) in scores.iter().enumerate() {
                    e[jj] = (sc - m).exp();
                    denom += e[jj];
                }
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for (jj, ej) in e.iter().enumerate() {
                        let j = start + jj;
                        acc += (ej / denom) * v[j][kh * d + dd];
                    }
                    attn_out[i][qh * d + dd] = acc;
                }
            }
        }

        let ow = g(&p("self_attn.o_proj.weight"));
        let ob = g(&p("self_attn.o_proj.bias"));
        for i in 0..l {
            let proj = linear_ref(&attn_out[i], ow, hq * d, h, Some(ob));
            for c in 0..h {
                x[i][c] += proj[c];
            }
        }

        let ln2w = g(&p("post_attention_layernorm.weight"));
        let router_w = g(&p("mlp.router.weight"));
        let router_b = g(&p("mlp.router.bias"));
        let gate_up = g(&p("mlp.experts.gate_up_proj"));
        let gu_bias = g(&p("mlp.experts.gate_up_proj_bias"));
        let down = g(&p("mlp.experts.down_proj"));
        let down_bias = g(&p("mlp.experts.down_proj_bias"));
        for row in x.iter_mut().take(l) {
            let normed = rmsnorm_ref(row, ln2w, h, cfg.eps);
            let logits = linear_ref(&normed, router_w, h, mp.n_experts, Some(router_b));
            let selected = top_k_router_ref(&logits, mp.top_k);
            let mut moe_out = vec![0.0f32; h];
            for (e, gate_weight) in selected {
                let e_gate_up = &gate_up[e * h * 2 * mp.inter..(e + 1) * h * 2 * mp.inter];
                let e_gu_bias = &gu_bias[e * 2 * mp.inter..(e + 1) * 2 * mp.inter];
                let e_down = &down[e * mp.inter * h..(e + 1) * mp.inter * h];
                let e_down_bias = &down_bias[e * h..(e + 1) * h];

                let mut gate_w = vec![0.0f32; h * mp.inter];
                let mut up_w = vec![0.0f32; h * mp.inter];
                for row_i in 0..h {
                    for i in 0..mp.inter {
                        gate_w[row_i * mp.inter + i] = e_gate_up[row_i * 2 * mp.inter + 2 * i];
                        up_w[row_i * mp.inter + i] = e_gate_up[row_i * 2 * mp.inter + 2 * i + 1];
                    }
                }
                let mut gate_b = vec![0.0f32; mp.inter];
                let mut up_b = vec![0.0f32; mp.inter];
                for i in 0..mp.inter {
                    gate_b[i] = e_gu_bias[2 * i];
                    up_b[i] = e_gu_bias[2 * i + 1];
                }

                let mut gate = linear_ref(&normed, &gate_w, h, mp.inter, Some(&gate_b));
                let mut up = linear_ref(&normed, &up_w, h, mp.inter, Some(&up_b));
                for gv in gate.iter_mut() {
                    *gv = clamp_ref(*gv, None, Some(mp.swiglu_limit));
                }
                for uv in up.iter_mut() {
                    *uv = clamp_ref(*uv, Some(-mp.swiglu_limit), Some(mp.swiglu_limit));
                }
                let act: Vec<f32> = gate
                    .iter()
                    .zip(up.iter())
                    .map(|(&g_, &u)| {
                        let glu = g_ * sigmoid_ref(g_ * poot_models::gpt_oss::GPTOSS_SWIGLU_ALPHA);
                        (u + 1.0) * glu
                    })
                    .collect();
                let expert_out = linear_ref(&act, e_down, mp.inter, h, Some(e_down_bias));
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
    linear_ref(&last, lm_head, h, cfg.vocab, None)
}

/// Proves `Runner::load`'s weight-layout crosswalk (router transpose, tied-embedding lm_head,
/// direct-bind already-fused `[in,out]` expert tensors, the bias tensors, and the YaRN-scaled rope table)
/// on a real checkpoint: the traced-and-bound `trace_gptoss_prefill` graph, evaluated through the
/// production `Runner::load` + `build_weights` + `bind` path, must match an independent from-scratch Rust
/// reference (`gptoss_prefill_ref_from_runner`) reading the same weight map. Needs `gptoss-tiny` under POOT_MODELS_DIR;
/// `hf download tiny-random/gpt-oss --local-dir <POOT_MODELS_DIR>/gptoss-tiny` populates it.
#[test]
fn gptoss_tiny_checkpoint_runner_load_matches_hand_rolled_reference() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny")) else {
        return;
    };
    let runner = Runner::load(&dir).expect("load gptoss-tiny via Runner");
    assert_eq!(runner.arch, "gpt_oss");
    let mp = runner
        .gpt_oss
        .clone()
        .expect("Runner::load must set gpt_oss params for a gpt_oss checkpoint");
    assert_eq!(mp.n_experts, 32);
    assert_eq!(mp.top_k, 4);
    assert_eq!(mp.inter, 64);
    assert_eq!(mp.layer_is_sliding, vec![true, false]);
    assert!(
        runner.sliding_window.is_none(),
        "the uniform Runner::sliding_window must stay inert - gpt-oss's per-layer schedule lives in GptOssParams::layer_is_sliding instead (mirrors Gemma3's own scoping)"
    );

    let tokens = [1u32, 2, 3, 4];
    let g = trace_gptoss_prefill(runner.cfg, mp.clone(), tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind gpt-oss prefill graph");
    let out = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");
    assert_eq!(out.shape(), vec![1, 1, runner.cfg.vocab]);
    assert!(out.as_f32().unwrap().iter().all(|v| v.is_finite()));

    let want = gptoss_prefill_ref_from_runner(&runner.cfg, &mp, &tokens, &runner.weights);
    assert_eq!(out.as_f32().unwrap().len(), want.len());
    for (i, (&a, &b)) in out.as_f32().unwrap().iter().zip(want.iter()).enumerate() {
        // See `poot_models::gpt_oss`'s real-checkpoint test: this huge (201088) tied-embedding vocab has a
        // small minority of near-zero-crossing logits where plain relative tolerance amplifies f32
        // summation-order noise; a slightly larger absolute floor (1e-3, far above the measured noise) absorbs it.
        let denom = a.abs().max(b.abs()).max(1e-3);
        let rel = (a - b).abs() / denom;
        assert!(
            rel <= 1e-4,
            "index {i}: Runner-loaded={a} hand-rolled-ref={b} (rel {rel} > 1e-4)"
        );
    }
}

/// `Runner::load_gguf`'s gpt-oss detection (gguf.rs/runner.rs: a plain `general.architecture ==
/// "gpt-oss"` arch-string match, unlike Mixtral's router-tensor detection), cross-checked against
/// `Runner::load`'s verified safetensors path on the same real checkpoint (executor equivalence, per
/// AGENTS.md's verification stack), like `mixtral_load_tests::mixtral_tiny_gguf_matches_safetensors` and
/// `olmoe_load_tests::olmoe_tiny_gguf_matches_safetensors`. Uses explicit tokens through `bind`+`eval`,
/// not `generate()`, to avoid the BOS-handling asymmetry between the loaders.
///
/// Also checks a non-obvious finding: gpt-oss's native checkpoint interleaves expert gate/up
/// (even/odd), but llama.cpp's GGUF conversion re-packs it into the standard split-half
/// `ffn_gate_exps`/`ffn_up_exps` convention (see `gguf.rs`'s `interleave_experts_last` and the
/// `arch == "gpt-oss"` arm). A bug in that re-interleave (e.g. `concat_experts_last`) would give a
/// shape-valid but numerically wrong tensor, which a shape/graph-validation check would not catch.
///
/// The GGUF is a real llama.cpp conversion (`convert_hf_to_gguf.py --outtype f32`, in a throwaway `uv`
/// venv) of the same `gptoss-tiny` checkpoint, saved beside it as `gptoss-tiny-f32.gguf` (not
/// committed). A transpose/fuse/interleave bug in the GGUF gpt-oss arm (`gguf_weights` in `gguf.rs`'s
/// `arch == "gpt-oss"` MoE branch) would give a materially different forward pass, since both paths
/// are f32 with bit-identical weights.
#[test]
fn gptoss_tiny_gguf_matches_safetensors() {
    let Some(st_dir) = poot_test_util::model_path(poot_test_util::checkpoint!("gptoss-tiny"))
    else {
        return;
    };
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gptoss-tiny/gptoss-tiny-f32.gguf"
    )) else {
        return;
    };
    let st_runner = Runner::load(&st_dir).expect("load gptoss-tiny safetensors");
    let gguf_runner = Runner::load_gguf(&gguf_path).expect("load gptoss-tiny gguf");
    assert_eq!(st_runner.arch, "gpt_oss");
    assert_eq!(gguf_runner.arch, "gpt_oss");

    let st_mp = st_runner
        .gpt_oss
        .clone()
        .expect("safetensors Runner must set gpt_oss params");
    let gguf_mp = gguf_runner
        .gpt_oss
        .clone()
        .expect("gguf Runner must set gpt_oss params (card 135d GGUF follow-on)");
    assert_eq!(st_mp.n_experts, gguf_mp.n_experts);
    assert_eq!(st_mp.top_k, gguf_mp.top_k);
    assert_eq!(st_mp.inter, gguf_mp.inter);
    assert_eq!(st_mp.swiglu_limit, gguf_mp.swiglu_limit);
    assert_eq!(st_mp.sliding_window, gguf_mp.sliding_window);
    assert_eq!(
        st_mp.layer_is_sliding, gguf_mp.layer_is_sliding,
        "GGUF has no layer_types metadata array - the hardcoded even-sliding/odd-full parity in \
             runner.rs must reproduce the real config's own layer_types exactly"
    );

    let tokens = [3u32, 7, 11, 2];
    let st_g = trace_gptoss_prefill(st_runner.cfg, st_mp, tokens.len());
    let st_inputs = st_runner
        .bind(&st_g, &tokens)
        .expect("bind safetensors prefill graph");
    let st_out =
        crate::core::cpu_oracle::cpu_eval(&st_g, &st_inputs).expect("cpu eval (safetensors)");

    let gguf_g = trace_gptoss_prefill(gguf_runner.cfg, gguf_mp, tokens.len());
    let gguf_inputs = gguf_runner
        .bind(&gguf_g, &tokens)
        .expect("bind gguf prefill graph");
    let gguf_out =
        crate::core::cpu_oracle::cpu_eval(&gguf_g, &gguf_inputs).expect("cpu eval (gguf)");

    assert_eq!(st_out.shape(), gguf_out.shape());
    assert!(st_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert!(gguf_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // Safetensors vs gguf last-position logits: a divergence here is a real transpose/fuse/interleave bug
    // in the new gpt-oss GGUF arm, not float noise (both paths are f32, no quantization involved).
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

/// A GGUF-loaded gpt-oss masked decode must bind its consts and carry the alternating sliding-window/full
/// DECODE floor's range (`gemma4_local_window_floor`, reused by `trace_gptoss_decode_kv_masked`) as a
/// computed iota.
///
/// The floor used to read a synthesized `causal.iota` const: the safetensors path inserted it
/// (`build_weights`'s `is_gpt_oss()` arm) but `gguf_weights` inserted it only under `arch == "gemma3"`,
/// so every GGUF-loaded gpt-oss failed the masked KV decode with `no weight bound for causal.iota`, in
/// `Runner::const_inputs`, before any executor touched the GPU (the real 20B run died there after a
/// 78.8s MXFP4 load). Card 550a removed the synthesized constant entirely: the range is an in-graph
/// `iota` folded to a computed input, so this cell now pins that the fold and the bind both hold.
///
/// The gpt-oss GGUF tests only exercise prefill and the masked-decode tests only load safetensors;
/// this is the GGUF x masked-decode cell. CPU-only.
#[test]
fn gptoss_gguf_binds_causal_iota_for_masked_decode() {
    use poot_graph_ir::Storage;
    use poot_models::gpt_oss::trace_gptoss_decode_kv_masked;
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gptoss-tiny/gptoss-tiny-f32.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load gptoss-tiny gguf");
    let mp = runner
        .gpt_oss
        .clone()
        .expect("gguf Runner must set gpt_oss params");
    // Same construction as the real masked-decode paths (see `generate_kv_gpu_masked_ptx`): trace at a
    // fixed KV capacity, optimize, then bind the Const inputs.
    let cap = 8usize;
    let g = trace_gptoss_decode_kv_masked(runner.cfg, mp, cap);
    let g = optimize(&g);
    let consts = runner
        .const_inputs(&g)
        .expect("gguf gpt-oss masked decode must bind every Const (regression: causal.iota)");

    // Present and correct: the folded iota range [0, max_pos), the identity the window floor relies on
    // (`pos_f` is read as `gather(iota, pos)`, so a wrong table silently mis-masks).
    let iota = g
        .inputs
        .iter()
        .copied()
        .find(|&id| matches!(g.meta(id).storage, Storage::Computed(_)))
        .expect("the gpt-oss masked decode graph must carry its computed causal range");
    let iota = consts
        .get(&iota)
        .expect("computed causal range bound by const_inputs");
    assert_eq!(
        iota.as_host().expect("dense range").shape(),
        vec![runner.cfg.max_pos]
    );
    assert!(
        iota.as_host()
            .expect("dense range")
            .as_f32()
            .unwrap()
            .iter()
            .enumerate()
            .all(|(i, v)| *v == i as f32),
        "the computed causal range must be exactly [0,1,..,max_pos)"
    );
}

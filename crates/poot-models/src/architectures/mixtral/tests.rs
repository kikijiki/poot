use super::*;

mod trace;

use poot_graph_ir::Storage;

fn tiny_cfg() -> Qwen2Config {
    // Small but non-degenerate dims: GQA (n_kv_heads < n_heads), no qk_norm, no qkv_bias (the Mixtral flags).
    Qwen2Config {
        vocab: 24,
        hidden: 16,
        inter: 16, // unused directly by mixtral (MoE inter comes from MixtralParams), kept non-zero
        layers: 3,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-5,
        max_pos: 32,
        qkv_bias: false,
        qk_norm: false,
        ..Default::default()
    }
}

fn tiny_mp() -> MixtralParams {
    MixtralParams {
        n_experts: 4,
        top_k: 2,
        inter: 12,
    }
}

// ---- CPU-oracle numerics: checked against an independent from-scratch Rust reference forward pass. ----

mod cpu_oracle {
    use super::*;
    use std::collections::HashMap;

    use poot_test_util::fill;

    use poot_test_util::seed_of;

    fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
        let raw = fill(n, seed_of(name));
        if ln_gamma {
            raw.iter().map(|v| 1.0 + v * 0.05).collect()
        } else {
            raw.iter().map(|v| v * 0.1).collect()
        }
    }

    fn rope_tables(max_pos: usize, d: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
        let mut cos = vec![0.0f32; max_pos * d];
        let mut sin = vec![0.0f32; max_pos * d];
        for pos in 0..max_pos {
            for i in 0..d / 2 {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / d as f32);
                let ang = pos as f32 * freq;
                let (s, c) = ang.sin_cos();
                cos[pos * d + i] = c;
                cos[pos * d + i + d / 2] = c;
                sin[pos * d + i] = s;
                sin[pos * d + i + d / 2] = s;
            }
        }
        (cos, sin)
    }

    /// All named constants [`trace_mixtral_prefill`] declares, keyed by exact graph const name.
    fn all_weights(cfg: &Qwen2Config, mp: &MixtralParams) -> HashMap<String, Vec<f32>> {
        let (h, d, hq, hkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let (q_dim, kv_dim) = (hq * d, hkv * d);
        let mut w = HashMap::new();
        w.insert(
            "model.embed_tokens.weight".to_string(),
            weight("embed", cfg.vocab * h, false),
        );
        let (cos, sin) = rope_tables(cfg.max_pos, d, 1_000_000.0); // real rope_theta
        w.insert("rope.cos".to_string(), cos);
        w.insert("rope.sin".to_string(), sin);
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            w.insert(p("input_layernorm.weight"), weight(&p("ln1"), h, true));
            w.insert(
                p("self_attn.q_proj.weight"),
                weight(&p("qw"), h * q_dim, false),
            );
            w.insert(
                p("self_attn.k_proj.weight"),
                weight(&p("kw"), h * kv_dim, false),
            );
            w.insert(
                p("self_attn.v_proj.weight"),
                weight(&p("vw"), h * kv_dim, false),
            );
            w.insert(
                p("self_attn.o_proj.weight"),
                weight(&p("ow"), q_dim * h, false),
            );
            w.insert(
                p("post_attention_layernorm.weight"),
                weight(&p("ln2"), h, true),
            );
            w.insert(
                p("block_sparse_moe.gate.weight"),
                weight(&p("router"), h * mp.n_experts, false),
            );
            // gate_up_proj: [E, H, 2I], gate||up per expert, [in,out] convention.
            let mut gate_up = vec![0.0f32; mp.n_experts * h * 2 * mp.inter];
            let mut down = vec![0.0f32; mp.n_experts * mp.inter * h];
            for e in 0..mp.n_experts {
                let g = weight(&p(&format!("e{e}.gate")), h * mp.inter, false);
                let u = weight(&p(&format!("e{e}.up")), h * mp.inter, false);
                let d_ = weight(&p(&format!("e{e}.down")), mp.inter * h, false);
                for row in 0..h {
                    let dst = (e * h + row) * 2 * mp.inter;
                    gate_up[dst..dst + mp.inter]
                        .copy_from_slice(&g[row * mp.inter..(row + 1) * mp.inter]);
                    gate_up[dst + mp.inter..dst + 2 * mp.inter]
                        .copy_from_slice(&u[row * mp.inter..(row + 1) * mp.inter]);
                }
                let dst = e * mp.inter * h;
                down[dst..dst + mp.inter * h].copy_from_slice(&d_);
            }
            w.insert(p("block_sparse_moe.experts.gate_up_proj.weight"), gate_up);
            w.insert(p("block_sparse_moe.experts.down_proj.weight"), down);
        }
        w.insert("model.norm.weight".to_string(), weight("ln_f", h, true));
        w.insert(
            "lm_head.weight".to_string(),
            weight("lm_head", h * cfg.vocab, false),
        );
        w
    }

    use poot_test_util::rmsnorm_ref;

    use poot_test_util::silu_ref;

    use poot_test_util::linear_ref;

    use poot_test_util::rope_ref;

    /// Independent top-`top_k` renormalized-softmax router (softmax, topk, renormalize; equal to softmax over the
    /// selected raw logits, see the module doc), re-derived directly rather than via `moe`'s decomposition.
    /// Returns `(expert_id, weight)` pairs for the `top_k` selected experts, weights summing to 1.
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

    /// Independent reference forward pass for [`trace_mixtral_prefill`] (last token only), written as direct
    /// loops rather than a copy of `moe`'s decomposition.
    fn mixtral_prefill_ref(
        cfg: &Qwen2Config,
        mp: &MixtralParams,
        tokens: &[usize],
        w: &HashMap<String, Vec<f32>>,
    ) -> Vec<f32> {
        let (h, d, hq, hkv, l) = (
            cfg.hidden,
            cfg.head_dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            tokens.len(),
        );
        let n_rep = hq / hkv;
        let scale = 1.0 / (d as f32).sqrt();
        let embed = &w["model.embed_tokens.weight"];
        let cos = &w["rope.cos"];
        let sin = &w["rope.sin"];

        let mut x: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| embed[t * h..(t + 1) * h].to_vec())
            .collect();

        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            let ln1w = &w[&p("input_layernorm.weight")];
            let normed: Vec<Vec<f32>> = x
                .iter()
                .map(|row| rmsnorm_ref(row, ln1w, h, cfg.eps))
                .collect();

            let qw = &w[&p("self_attn.q_proj.weight")];
            let kw = &w[&p("self_attn.k_proj.weight")];
            let vw = &w[&p("self_attn.v_proj.weight")];
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

            let ow = &w[&p("self_attn.o_proj.weight")];
            for i in 0..l {
                let proj = linear_ref(&attn_out[i], ow, hq * d, h);
                for c in 0..h {
                    x[i][c] += proj[c];
                }
            }

            let ln2w = &w[&p("post_attention_layernorm.weight")];
            let router_w = &w[&p("block_sparse_moe.gate.weight")];
            let gate_up = &w[&p("block_sparse_moe.experts.gate_up_proj.weight")];
            let down = &w[&p("block_sparse_moe.experts.down_proj.weight")];
            for row in x.iter_mut().take(l) {
                let normed = rmsnorm_ref(row, ln2w, h, cfg.eps);
                let logits = linear_ref(&normed, router_w, h, mp.n_experts);
                let selected = top_k_router_ref(&logits, mp.top_k);
                let mut moe_out = vec![0.0f32; h];
                for (e, gate_weight) in selected {
                    let e_gate_up = &gate_up[e * h * 2 * mp.inter..(e + 1) * h * 2 * mp.inter];
                    // slice out the gate half / up half from the fused [H, 2I] layout.
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
                        .map(|(&g, &u)| silu_ref(g) * u)
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

        let ln_f_w = &w["model.norm.weight"];
        let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
        let lm_head = &w["lm_head.weight"];
        linear_ref(&last, lm_head, h, cfg.vocab)
    }

    fn eval_mixtral_prefill(
        g: &Graph,
        tokens: &[usize],
        weights: &HashMap<String, Vec<f32>>,
    ) -> poot_tensor::HostTensor {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match &meta.storage {
                Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                    vec![tokens.len()],
                    tokens.iter().map(|&t| t as i32).collect(),
                ),
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    let l = meta.aval.shape[2];
                    let mut m = vec![0.0f32; l * l];
                    for i in 0..l {
                        for j in 0..l {
                            m[i * l + j] = if j <= i { 0.0 } else { -1.0e30 };
                        }
                    }
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), m)
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data = weights
                        .get(name)
                        .unwrap_or_else(|| panic!("no weight bound for {name}"));
                    poot_tensor::HostTensor::f32(meta.aval.shape.clone(), data.clone())
                }
                other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
            };
            inputs.insert(id, t.into());
        }
        poot_eval::eval(
            g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu eval")
        .output
        .into_host()
        .expect("dense output")
    }

    /// `trace_mixtral_prefill` matches an independent hand-loop reference within f32 tolerance.
    #[test]
    fn mixtral_prefill_matches_hand_rolled_reference() {
        let cfg = tiny_cfg();
        let mp = tiny_mp();
        let tokens = [3usize, 7, 1, 9, 15];
        let weights = all_weights(&cfg, &mp);

        let g = trace_mixtral_prefill(cfg, mp, tokens.len());
        let got = eval_mixtral_prefill(&g, &tokens, &weights);

        let want = mixtral_prefill_ref(&cfg, &mp, &tokens, &weights);
        assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
        poot_test_util::assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
        assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
        assert!(
            got.as_f32()
                .unwrap()
                .iter()
                .any(|&v| v != got.as_f32().unwrap()[0])
        );
    }

    /// Perturbing a deep expert weight (one entry of layer 1 expert 2's down-projection) changes the output: the
    /// routing/gather chain is live.
    #[test]
    fn mixtral_prefill_output_is_sensitive_to_a_perturbed_expert_weight() {
        let cfg = tiny_cfg();
        let mp = tiny_mp();
        let tokens = [2usize, 5, 8];
        let mut weights = all_weights(&cfg, &mp);
        let g = trace_mixtral_prefill(cfg, mp, tokens.len());
        let base = eval_mixtral_prefill(&g, &tokens, &weights);

        let key = "model.layers.1.block_sparse_moe.experts.down_proj.weight".to_string();
        weights.get_mut(&key).unwrap()[0] += 5.0;
        let perturbed = eval_mixtral_prefill(&g, &tokens, &weights);

        let max_diff =
            poot_test_util::max_abs_error(base.as_f32().unwrap(), perturbed.as_f32().unwrap());
        assert!(
            max_diff > 1e-4,
            "perturbing {key} should change the output; max_diff={max_diff:.2e}"
        );
    }

    // ---- real-checkpoint weight-layout crosswalk: optimum-intel-internal-testing/tiny-mixtral. ----
    //
    // A randomly initialized `MixtralForCausalLM` HF export (2 layers, hidden 1024, 8 experts top-2, ~945MB f32).
    // It proves the loader's transpose/fuse crosswalk (w1/w3/w2 -> gate/up/down, per-expert fuse) against real
    // HF-shaped tensors, as `yujiepan/qwen3-moe-tiny-random` does for `crate::qwen3moe`. Loads via `poot_load`
    // (poot-llm sits above this crate) and compares the traced graph's CPU eval to `mixtral_prefill_ref` fed the
    // loaded weights.
    //
    // The config has `sliding_window: 4096` (the real 8x7B has `null`); every sequence here is far shorter, so
    // this says nothing about SWA handling (out of scope, spec 261).
    #[test]
    fn mixtral_tiny_random_checkpoint_matches_hand_rolled_reference() {
        // Populate with `hf download optimum-intel-internal-testing/tiny-mixtral --local-dir
        // $POOT_MODELS_DIR/mixtral-tiny`.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("mixtral-tiny"))
        else {
            return;
        };

        let hf = poot_load::Qwen2HfConfig::load(dir.join("config.json")).expect("load config.json");
        assert!(hf.is_mixtral());
        let st = poot_load::safetensors::load_weight_store(&dir).expect("load safetensors");

        let cfg = Qwen2Config {
            vocab: hf.vocab_size,
            hidden: hf.hidden_size,
            inter: hf.intermediate_size,
            layers: hf.num_hidden_layers,
            n_heads: hf.num_attention_heads,
            n_kv_heads: hf.num_key_value_heads,
            head_dim: hf.head_dim(),
            rotary_dim: hf.rotary_dim(),
            eps: hf.rms_norm_eps,
            max_pos: 8, // only positions 0..4 are ever read; keep the rope table cheap.
            qkv_bias: hf.qkv_bias(),
            qk_norm: hf.qk_norm(),
            ..Default::default()
        };
        let mp = MixtralParams {
            n_experts: hf
                .num_local_experts
                .expect("mixtral config carries num_local_experts"),
            top_k: hf
                .num_experts_per_tok
                .expect("mixtral config carries num_experts_per_tok"),
            inter: hf.intermediate_size,
        };
        assert_eq!(mp.n_experts, 8);
        assert_eq!(mp.top_k, 2);

        fn transpose2d(data: &[f32], r: usize, c: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; r * c];
            for i in 0..r {
                for j in 0..c {
                    out[j * r + i] = data[i * c + j];
                }
            }
            out
        }

        /// Fuse a layer's separate per-expert `{prefix}.{e}.{w1,w2,w3}.weight` tensors into the fused `[E,H,2I]`
        /// gate||up / `[E,I,H]` down layout `mixtral_ffn` binds. `w1` = gate, `w3` = up, `w2` = down.
        fn fuse_experts(
            st: &poot_quant::weights::WeightStore,
            prefix: &str,
            n_experts: usize,
            hidden: usize,
            inter: usize,
        ) -> (Vec<f32>, Vec<f32>) {
            let mut gate_up = vec![0.0f32; n_experts * hidden * 2 * inter];
            let mut down = vec![0.0f32; n_experts * inter * hidden];
            for e in 0..n_experts {
                let w1 = poot_eval::materialize_dense(st, &format!("{prefix}.{e}.w1.weight"))
                    .unwrap_or_else(|err| panic!("{prefix}.{e}.w1.weight: {err}"));
                let w3 = poot_eval::materialize_dense(st, &format!("{prefix}.{e}.w3.weight"))
                    .unwrap_or_else(|err| panic!("{prefix}.{e}.w3.weight: {err}"));
                let w2 = poot_eval::materialize_dense(st, &format!("{prefix}.{e}.w2.weight"))
                    .unwrap_or_else(|err| panic!("{prefix}.{e}.w2.weight: {err}"));
                // w1/w3: HF [inter, hidden] (out,in) -> transposed [hidden, inter] (in,out).
                let gt = transpose2d(w1.as_f32().unwrap(), inter, hidden);
                let ut = transpose2d(w3.as_f32().unwrap(), inter, hidden);
                for row in 0..hidden {
                    let dst = (e * hidden + row) * 2 * inter;
                    gate_up[dst..dst + inter].copy_from_slice(&gt[row * inter..(row + 1) * inter]);
                    gate_up[dst + inter..dst + 2 * inter]
                        .copy_from_slice(&ut[row * inter..(row + 1) * inter]);
                }
                // w2: HF [hidden, inter] (out,in) -> transposed [inter, hidden] (in,out).
                let dt = transpose2d(w2.as_f32().unwrap(), hidden, inter);
                let dst = e * inter * hidden;
                down[dst..dst + inter * hidden].copy_from_slice(&dt);
            }
            (gate_up, down)
        }

        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        let get2d = |name: &str| -> Vec<f32> {
            let rt =
                poot_eval::materialize_dense(&st, name).unwrap_or_else(|e| panic!("{name}: {e}"));
            transpose2d(rt.as_f32().unwrap(), rt.shape()[0], rt.shape()[1])
        };
        let get1d = |name: &str| -> Vec<f32> {
            poot_eval::materialize_dense(&st, name)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .as_f32()
                .unwrap()
                .to_vec()
        };

        weights.insert(
            "model.embed_tokens.weight".to_string(),
            get1d("model.embed_tokens.weight"),
        );
        // Untied here (tie_word_embeddings: false): a separate lm_head.weight is present, transposed like any
        // other projection.
        weights.insert("lm_head.weight".to_string(), get2d("lm_head.weight"));
        weights.insert("model.norm.weight".to_string(), get1d("model.norm.weight"));
        let (cos, sin) = rope_tables(
            cfg.max_pos,
            cfg.head_dim,
            hf.effective_rope_theta()
                .expect("config carries rope_theta"),
        );
        weights.insert("rope.cos".to_string(), cos);
        weights.insert("rope.sin".to_string(), sin);

        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            weights.insert(
                p("input_layernorm.weight"),
                get1d(&p("input_layernorm.weight")),
            );
            weights.insert(
                p("post_attention_layernorm.weight"),
                get1d(&p("post_attention_layernorm.weight")),
            );
            weights.insert(
                p("self_attn.q_proj.weight"),
                get2d(&p("self_attn.q_proj.weight")),
            );
            weights.insert(
                p("self_attn.k_proj.weight"),
                get2d(&p("self_attn.k_proj.weight")),
            );
            weights.insert(
                p("self_attn.v_proj.weight"),
                get2d(&p("self_attn.v_proj.weight")),
            );
            weights.insert(
                p("self_attn.o_proj.weight"),
                get2d(&p("self_attn.o_proj.weight")),
            );
            weights.insert(
                p("block_sparse_moe.gate.weight"),
                get2d(&p("block_sparse_moe.gate.weight")),
            );
            let (gate_up, down) = fuse_experts(
                &st,
                &p("block_sparse_moe.experts"),
                mp.n_experts,
                cfg.hidden,
                mp.inter,
            );
            weights.insert(p("block_sparse_moe.experts.gate_up_proj.weight"), gate_up);
            weights.insert(p("block_sparse_moe.experts.down_proj.weight"), down);
        }

        let tokens = [1usize, 2, 3, 4];
        let g = trace_mixtral_prefill(cfg, mp, tokens.len());
        let got = eval_mixtral_prefill(&g, &tokens, &weights);
        let want = mixtral_prefill_ref(&cfg, &mp, &tokens, &weights);

        assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
        assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
        poot_test_util::assert_close_rel(got.as_f32().unwrap(), &want, 1e-4);
    }
}

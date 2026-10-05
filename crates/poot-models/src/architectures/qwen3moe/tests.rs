use super::*;

mod trace;

use poot_graph_ir::Storage;

fn qwen3_moe_cfg() -> Qwen2Config {
    // small but non-degenerate dims (Qwen3-30B-A3B ratios scaled down): qkv_bias=false, qk_norm=true,
    // head_dim independent of hidden/n_heads.
    Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 24, // dense-layer intermediate (distinct from the MoE inter below)
        layers: 4,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        rotary_dim: 8,
        eps: 1e-6,
        max_pos: 64,
        qkv_bias: false,
        qk_norm: true,
        ..Default::default()
    }
}

fn moe_params(cfg: &Qwen2Config, sparse_layer: Vec<bool>) -> Qwen3MoeParams {
    assert_eq!(sparse_layer.len(), cfg.layers);
    Qwen3MoeParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        sparse_layer,
    }
}

// ---- CPU-oracle numerics (poot_eval) ----

mod cpu_oracle {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic non-degenerate synthetic weights (`sin`-based); never all zero/one, which would hide
    /// indexing/transpose bugs.
    fn synth(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.0137 + seed * 0.911).sin() * 0.6)
            .collect()
    }

    fn rope_cos(max_pos: usize, rotary_dim: usize, theta: f32) -> poot_tensor::HostTensor {
        let half = rotary_dim / 2;
        let mut cos = vec![0.0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let c = (pos as f32 * freq).cos();
                cos[pos * rotary_dim + i] = c;
                cos[pos * rotary_dim + half + i] = c;
            }
        }
        poot_tensor::HostTensor::f32(vec![max_pos, rotary_dim], cos)
    }

    /// Causal prefill mask `[1,1,l,l]`: `0.0` where key `j <= query i`, else a large negative value. Matches
    /// `poot_llm::generate`'s `prefill_causal_mask`.
    fn causal_mask(l: usize) -> poot_tensor::HostTensor {
        let mut m = vec![0.0f32; l * l];
        for i in 0..l {
            for j in (i + 1)..l {
                m[i * l + j] = -1.0e30;
            }
        }
        poot_tensor::HostTensor::f32(vec![1, 1, l, l], m)
    }

    /// Attention/norm/embed/lm_head weights of a qwen3-moe graph over `cfg` (`qkv_bias=false,
    /// qk_norm=true`); the dense-layer logits recorded in the decomposition oracle came from these.
    fn shared_weights(cfg: &Qwen2Config) -> HashMap<String, poot_tensor::HostTensor> {
        let (h, d, hq, hkv, inter, vocab) = (
            cfg.hidden,
            cfg.head_dim,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.inter,
            cfg.vocab,
        );
        let q_dim = hq * d;
        let kv_dim = hkv * d;
        let mut w = HashMap::new();
        let mut seed = 1.0f32;
        let mut next = |n: usize| {
            seed += 1.0;
            synth(n, seed)
        };
        w.insert(
            "model.embed_tokens.weight".to_string(),
            poot_tensor::HostTensor::f32(vec![vocab, h], next(vocab * h)),
        );
        w.insert(
            "model.norm.weight".to_string(),
            poot_tensor::HostTensor::f32(vec![h], next(h)),
        );
        w.insert(
            "lm_head.weight".to_string(),
            poot_tensor::HostTensor::f32(vec![h, vocab], next(h * vocab)),
        );
        w.insert(
            "rope.cos".to_string(),
            rope_cos(cfg.max_pos, cfg.rotary_dim, 1.0e4),
        );
        w.insert(
            "rope.sin".to_string(),
            rope_sin(cfg.max_pos, cfg.rotary_dim, 1.0e4),
        );
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            w.insert(
                p("input_layernorm.weight"),
                poot_tensor::HostTensor::f32(vec![h], next(h)),
            );
            w.insert(
                p("post_attention_layernorm.weight"),
                poot_tensor::HostTensor::f32(vec![h], next(h)),
            );
            w.insert(
                p("self_attn.q_proj.weight"),
                poot_tensor::HostTensor::f32(vec![h, q_dim], next(h * q_dim)),
            );
            w.insert(
                p("self_attn.k_proj.weight"),
                poot_tensor::HostTensor::f32(vec![h, kv_dim], next(h * kv_dim)),
            );
            w.insert(
                p("self_attn.v_proj.weight"),
                poot_tensor::HostTensor::f32(vec![h, kv_dim], next(h * kv_dim)),
            );
            w.insert(
                p("self_attn.o_proj.weight"),
                poot_tensor::HostTensor::f32(vec![q_dim, h], next(q_dim * h)),
            );
            w.insert(
                p("self_attn.q_norm.weight"),
                poot_tensor::HostTensor::f32(vec![d], next(d)),
            );
            w.insert(
                p("self_attn.k_norm.weight"),
                poot_tensor::HostTensor::f32(vec![d], next(d)),
            );
            w.insert(
                p("mlp.gate_proj.weight"),
                poot_tensor::HostTensor::f32(vec![h, inter], next(h * inter)),
            );
            w.insert(
                p("mlp.up_proj.weight"),
                poot_tensor::HostTensor::f32(vec![h, inter], next(h * inter)),
            );
            w.insert(
                p("mlp.down_proj.weight"),
                poot_tensor::HostTensor::f32(vec![inter, h], next(inter * h)),
            );
        }
        w
    }

    fn rope_sin(max_pos: usize, rotary_dim: usize, theta: f32) -> poot_tensor::HostTensor {
        let half = rotary_dim / 2;
        let mut sin = vec![0.0f32; max_pos * rotary_dim];
        for pos in 0..max_pos {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
                let angle = pos as f32 * freq;
                let s = angle.sin();
                sin[pos * rotary_dim + i] = s;
                sin[pos * rotary_dim + half + i] = s;
            }
        }
        poot_tensor::HostTensor::f32(vec![max_pos, rotary_dim], sin)
    }

    /// Transpose a row-major `[r,c]` HF weight (`[out,in]`) to `[c,r]` (`[in,out]`).
    fn transpose2d(data: &[f32], r: usize, c: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = data[i * c + j];
            }
        }
        out
    }

    /// Fuse a layer's separate per-expert `{prefix}.{e}.gate_proj/up_proj/down_proj.weight` tensors into the two
    /// constants [`ffn`] binds: `gate_up[E,H,2I]` (gate||up on the last axis, `[in,out]` per expert) and
    /// `down[E,I,H]`. The safetensors analog of `poot_llm::gguf`'s `transpose_experts`/`concat_experts_last`.
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
            let g = poot_eval::materialize_dense(st, &format!("{prefix}.{e}.gate_proj.weight"))
                .unwrap_or_else(|err| panic!("{prefix}.{e}.gate_proj.weight: {err}"));
            let u = poot_eval::materialize_dense(st, &format!("{prefix}.{e}.up_proj.weight"))
                .unwrap_or_else(|err| panic!("{prefix}.{e}.up_proj.weight: {err}"));
            let d = poot_eval::materialize_dense(st, &format!("{prefix}.{e}.down_proj.weight"))
                .unwrap_or_else(|err| panic!("{prefix}.{e}.down_proj.weight: {err}"));
            // gate_proj/up_proj: HF [inter, hidden] -> transposed [hidden, inter].
            let gt = transpose2d(g.as_f32().unwrap(), inter, hidden);
            let ut = transpose2d(u.as_f32().unwrap(), inter, hidden);
            for row in 0..hidden {
                let dst = (e * hidden + row) * 2 * inter;
                gate_up[dst..dst + inter].copy_from_slice(&gt[row * inter..(row + 1) * inter]);
                gate_up[dst + inter..dst + 2 * inter]
                    .copy_from_slice(&ut[row * inter..(row + 1) * inter]);
            }
            // down_proj: HF [hidden, inter] -> transposed [inter, hidden].
            let dt = transpose2d(d.as_f32().unwrap(), hidden, inter);
            let dst = e * inter * hidden;
            down[dst..dst + inter * hidden].copy_from_slice(&dt);
        }
        (gate_up, down)
    }

    /// Add the routed-expert constants for `n_experts`/`inter` at every layer index in `sparse_layers`.
    fn add_expert_weights(
        w: &mut HashMap<String, poot_tensor::HostTensor>,
        cfg: &Qwen2Config,
        n_experts: usize,
        inter: usize,
        sparse_layers: &[usize],
    ) {
        let h = cfg.hidden;
        let mut seed = 500.0f32;
        let mut next = |n: usize| {
            seed += 1.0;
            synth(n, seed)
        };
        for &li in sparse_layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            w.insert(
                p("mlp.gate.weight"),
                poot_tensor::HostTensor::f32(vec![h, n_experts], next(h * n_experts)),
            );
            w.insert(
                p("mlp.experts.gate_up_proj.weight"),
                poot_tensor::HostTensor::f32(
                    vec![n_experts, h, 2 * inter],
                    next(n_experts * h * 2 * inter),
                ),
            );
            w.insert(
                p("mlp.experts.down_proj.weight"),
                poot_tensor::HostTensor::f32(
                    vec![n_experts, inter, h],
                    next(n_experts * inter * h),
                ),
            );
        }
    }

    /// Bind and evaluate a stateless prefill graph (no KV state) against a name-keyed weight map.
    fn eval_prefill(
        g: &Graph,
        tokens: &[u32],
        weights: &HashMap<String, poot_tensor::HostTensor>,
    ) -> poot_tensor::HostTensor {
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match meta.storage {
                Storage::Slot(Slot::Token) => poot_tensor::HostTensor::i32(
                    vec![tokens.len()],
                    tokens.iter().map(|&t| t as i32).collect(),
                ),
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    causal_mask(meta.aval.shape[2])
                }
                Storage::Slot(Slot::Pos) => {
                    // card 550's dense qwen2/qwen3 prefill always starts at position 0; the sparse
                    // MoE prefill traced here keeps the pre-card host `Slot::Mask` (out of scope).
                    let l = meta.aval.shape[1];
                    poot_tensor::HostTensor::i32(meta.aval.shape.clone(), (0..l as i32).collect())
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    weights
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| panic!("no weight bound for {name}"))
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

    /// Decomposition oracle: with every layer dense, [`trace_qwen3_moe_prefill`] must compute what
    /// plain qwen3 dense computes for the same `cfg` + weights, since it degenerates to plain qwen3
    /// dense when no layer routes. The reference is the last-token logits recorded from the legacy
    /// `qwen2::trace_prefill` graph over the same weights at the base commit `bbd5c3232`.
    #[test]
    fn qwen3_moe_all_dense_layers_matches_qwen3_dense_trace() {
        const DENSE_LOGITS: [f32; 32] = [
            -0.26599258,
            -0.26598275,
            -0.265923,
            -0.26581344,
            -0.265654,
            -0.26544467,
            -0.26518548,
            -0.2648765,
            -0.26451784,
            -0.2641095,
            -0.2636518,
            -0.2631443,
            -0.2625876,
            -0.2619816,
            -0.2613263,
            -0.2606222,
            -0.25986904,
            -0.2590671,
            -0.25821638,
            -0.25731754,
            -0.25637016,
            -0.25537473,
            -0.2543313,
            -0.25324023,
            -0.25210172,
            -0.25091568,
            -0.24968268,
            -0.2484028,
            -0.24707626,
            -0.24570331,
            -0.2442843,
            -0.24281952,
        ];
        let cfg = qwen3_moe_cfg();
        let weights = shared_weights(&cfg);
        let tokens = [3u32, 7, 1, 22, 5];

        let mp = Qwen3MoeParams {
            n_experts: 6,
            top_k: 2,
            inter: 12,
            sparse_layer: vec![false; cfg.layers],
        };
        let g_moe = trace_qwen3_moe_prefill(cfg, mp, tokens.len());
        let out_moe = eval_prefill(&g_moe, &tokens, &weights);

        assert_eq!(out_moe.shape(), &[1, 1, 32]);
        for (i, (got, want)) in out_moe
            .as_f32()
            .unwrap()
            .iter()
            .zip(DENSE_LOGITS)
            .enumerate()
        {
            assert!(
                (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                "an all-dense-layer qwen3-moe trace must equal qwen3 dense's recorded logits: element {i}: got {got}, want {want}"
            );
        }
        // sanity: the shared weights are non-degenerate, so the output is not trivially all-zero.
        assert!(out_moe.as_f32().unwrap().iter().any(|&v| v != 0.0));
        assert!(out_moe.as_f32().unwrap().iter().all(|v| v.is_finite()));
    }

    /// The MoE branch is wired into the forward pass: routing layer 0 through the expert mixture instead of the
    /// dense MLP, with the same attention/embedding weights, must change the output, and the result must be finite.
    #[test]
    fn qwen3_moe_sparse_layer_flag_actually_changes_output() {
        let cfg = qwen3_moe_cfg();
        let mut weights = shared_weights(&cfg);
        let (n_experts, inter) = (6, 12);
        add_expert_weights(&mut weights, &cfg, n_experts, inter, &[0]);
        let tokens = [3u32, 7, 1, 22, 5];

        let all_dense = Qwen3MoeParams {
            n_experts,
            top_k: 2,
            inter,
            sparse_layer: vec![false; cfg.layers],
        };
        let g_dense = trace_qwen3_moe_prefill(cfg, all_dense, tokens.len());
        let out_dense = eval_prefill(&g_dense, &tokens, &weights);

        let mut sparse_layer = vec![false; cfg.layers];
        sparse_layer[0] = true;
        let one_sparse = Qwen3MoeParams {
            n_experts,
            top_k: 2,
            inter,
            sparse_layer,
        };
        let g_sparse = trace_qwen3_moe_prefill(cfg, one_sparse, tokens.len());
        let out_sparse = eval_prefill(&g_sparse, &tokens, &weights);

        assert_eq!(out_dense.shape(), out_sparse.shape());
        assert!(out_sparse.as_f32().unwrap().iter().all(|v| v.is_finite()));
        assert_ne!(
            out_dense.as_f32().unwrap(),
            out_sparse.as_f32().unwrap(),
            "routing layer 0 through the expert mixture must change the output vs. the dense MLP"
        );
    }

    /// Mixed dense/MoE layers with a batched prompt (`L>1`, so `moe()` takes the
    /// [`poot_graph_ir::ops::moe_grouped`] path) must give a finite, non-degenerate output.
    #[test]
    fn qwen3_moe_mixed_layers_batched_prefill_is_finite() {
        let cfg = qwen3_moe_cfg();
        let mut weights = shared_weights(&cfg);
        let (n_experts, inter) = (6, 12);
        add_expert_weights(&mut weights, &cfg, n_experts, inter, &[1, 3]);
        let tokens = [3u32, 7, 1, 22, 5];

        let mp = Qwen3MoeParams {
            n_experts,
            top_k: 2,
            inter,
            sparse_layer: vec![false, true, false, true],
        };
        let g = trace_qwen3_moe_prefill(cfg, mp, tokens.len());
        let out = eval_prefill(&g, &tokens, &weights);
        assert_eq!(out.shape(), vec![1, 1, cfg.vocab]);
        assert!(out.as_f32().unwrap().iter().all(|v| v.is_finite()));
        assert!(out.as_f32().unwrap().iter().any(|&v| v != 0.0));
    }

    // ---- end-to-end coherence: a real checkpoint ----
    //
    // `yujiepan/qwen3-moe-tiny-random` (HF Hub) is a randomly initialized export of HF's own Qwen3-MoE code. It
    // exercises the weight-name/layout crosswalk: separate per-expert `mlp.experts.{e}.{gate,up,down}_proj`
    // tensors (fused here), a tied lm_head (`tie_word_embeddings: true`), and `decoder_sparse_step: 2` (layer 0
    // dense, layer 1 MoE).
    //
    // The golden top-8-by-|logit| values were computed in pure numpy (float64) by reimplementing HF's
    // `modeling_qwen3_moe.py` forward against the same weights. Not bit-exact against poot's f32 CPU oracle
    // (different summation order in `moe_grouped`); the 0.01 tolerance leaves margin, and a transpose/fuse bug
    // would be off by orders of magnitude.
    #[test]
    fn qwen3_moe_tiny_random_checkpoint_matches_numpy_reference() {
        // Populate with `hf download yujiepan/qwen3-moe-tiny-random --local-dir
        // $POOT_MODELS_DIR/qwen3-moe-tiny`.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny"))
        else {
            return;
        };

        let hf = poot_load::Qwen2HfConfig::load(dir.join("config.json")).expect("load config.json");
        assert!(hf.is_qwen3_moe());
        hf.qwen3_moe_norm_topk_prob()
            .expect("checkpoint's norm_topk_prob must be true");
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
        let n_experts = hf
            .num_experts
            .expect("qwen3_moe config carries num_experts");
        let moe_inter = hf
            .moe_intermediate_size
            .expect("qwen3_moe config carries moe_intermediate_size");
        let top_k = hf
            .num_experts_per_tok
            .expect("qwen3_moe config carries num_experts_per_tok");
        let sparse_layer: Vec<bool> = (0..cfg.layers)
            .map(|li| hf.qwen3_moe_layer_is_sparse(li))
            .collect();
        let sparse_indices: Vec<usize> = (0..cfg.layers).filter(|&li| sparse_layer[li]).collect();
        // sanity: decoder_sparse_step=2 must give a mixed pattern, otherwise the per-layer switch is not exercised.
        assert!(!sparse_indices.is_empty() && sparse_indices.len() < cfg.layers);

        let mp = Qwen3MoeParams {
            n_experts,
            top_k,
            inter: moe_inter,
            sparse_layer,
        };

        let mut weights: HashMap<String, poot_tensor::HostTensor> = HashMap::new();
        let get2d = |name: &str| -> poot_tensor::HostTensor {
            let rt =
                poot_eval::materialize_dense(&st, name).unwrap_or_else(|e| panic!("{name}: {e}"));
            let (r, c) = (rt.shape()[0], rt.shape()[1]);
            poot_tensor::HostTensor::f32(vec![c, r], transpose2d(rt.as_f32().unwrap(), r, c))
        };
        let get1d = |name: &str| -> poot_tensor::HostTensor {
            poot_eval::materialize_dense(&st, name).unwrap_or_else(|e| panic!("{name}: {e}"))
        };

        let embed =
            poot_eval::materialize_dense(&st, "model.embed_tokens.weight").expect("embed_tokens");
        weights.insert("model.embed_tokens.weight".to_string(), embed.clone());
        // tied embeddings (this checkpoint has no separate lm_head.weight tensor): lm_head = embed^T.
        weights.insert(
            "lm_head.weight".to_string(),
            poot_tensor::HostTensor::f32(
                vec![cfg.hidden, cfg.vocab],
                transpose2d(embed.as_f32().unwrap(), cfg.vocab, cfg.hidden),
            ),
        );
        weights.insert("model.norm.weight".to_string(), get1d("model.norm.weight"));
        weights.insert(
            "rope.cos".to_string(),
            rope_cos(
                cfg.max_pos,
                cfg.rotary_dim,
                hf.effective_rope_theta()
                    .expect("config carries rope_theta"),
            ),
        );
        weights.insert(
            "rope.sin".to_string(),
            rope_sin(
                cfg.max_pos,
                cfg.rotary_dim,
                hf.effective_rope_theta()
                    .expect("config carries rope_theta"),
            ),
        );

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
                p("self_attn.q_norm.weight"),
                get1d(&p("self_attn.q_norm.weight")),
            );
            weights.insert(
                p("self_attn.k_norm.weight"),
                get1d(&p("self_attn.k_norm.weight")),
            );

            if mp.sparse_layer[li] {
                weights.insert(p("mlp.gate.weight"), get2d(&p("mlp.gate.weight")));
                let (gate_up, down) =
                    fuse_experts(&st, &p("mlp.experts"), n_experts, cfg.hidden, moe_inter);
                weights.insert(
                    p("mlp.experts.gate_up_proj.weight"),
                    poot_tensor::HostTensor::f32(
                        vec![n_experts, cfg.hidden, 2 * moe_inter],
                        gate_up,
                    ),
                );
                weights.insert(
                    p("mlp.experts.down_proj.weight"),
                    poot_tensor::HostTensor::f32(vec![n_experts, moe_inter, cfg.hidden], down),
                );
            } else {
                weights.insert(p("mlp.gate_proj.weight"), get2d(&p("mlp.gate_proj.weight")));
                weights.insert(p("mlp.up_proj.weight"), get2d(&p("mlp.up_proj.weight")));
                weights.insert(p("mlp.down_proj.weight"), get2d(&p("mlp.down_proj.weight")));
            }
        }

        let tokens = [1u32, 2, 3, 4];
        let g = trace_qwen3_moe_prefill(cfg, mp, tokens.len());
        let out = eval_prefill(&g, &tokens, &weights);
        assert_eq!(out.shape(), vec![1, 1, cfg.vocab]);
        assert!(out.as_f32().unwrap().iter().all(|v| v.is_finite()));

        // golden values from the independent numpy reference (see the comment above), truncated from float64 to f32.
        let golden: [(usize, f32); 8] = [
            (27959, -8.709_933),
            (68261, -8.360_13),
            (182, 7.893_251),
            (53765, -7.761_153),
            (34591, -7.591_176_5),
            (586, -7.377_918_7),
            (120483, -7.355_760_6),
            (131148, 7.283_245_6),
        ];
        let mut argmax_id = 0usize;
        let mut argmax_val = f32::NEG_INFINITY;
        for (i, &v) in out.as_f32().unwrap().iter().enumerate() {
            if v > argmax_val {
                argmax_val = v;
                argmax_id = i;
            }
        }
        // the largest-magnitude golden entry (182) is also the golden argmax; poot should agree unless two logits tie.
        assert_eq!(
            argmax_id, 182,
            "argmax token id should match the numpy reference (got value {argmax_val})"
        );
        for (id, expect) in golden {
            // token `id`: poot (actual) vs the numpy golden (expected)
            poot_test_util::assert_close(&[out.as_f32().unwrap()[id]], &[expect], 0.01);
        }
    }
}

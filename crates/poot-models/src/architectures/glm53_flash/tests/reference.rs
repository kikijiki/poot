//! The independent reference of GLM-5.3-Flash: the model written token by token from the layer
//! definitions in f64 over the checkpoint's own tensors, calling no component and no tracer.
//! Position `p` reads the histories positions `0..p` wrote, so a prefill chunk, a decode step and a
//! continuation are all compared with the same function of the prefix.
//!
//! The whole-text composition of three block-level references: the KDA recurrence (a gated delta rule
//! with a decay per key channel behind a causal conv), the DSA layer (latent attention without
//! rotary embedding over the positions a pooled indexer keeps) and the mHC streams around both
//! blocks, with the dense and the sigmoid-routed feed-forwards of the checkpoint.

use std::collections::HashMap;

use poot_quant::weights::WeightStore;

use crate::components::testing::values;

const TOWER: &str = "model.language_model.";

/// The dimensions the reference reads from the fixture's `config.json`.
#[derive(Clone, Debug)]
pub(super) struct Dims {
    pub vocab: usize,
    pub width: usize,
    pub layers: usize,
    /// Per layer: KDA (`true`) or DSA.
    pub kda: Vec<bool>,
    /// Per layer: dense (`true`) or routed feed-forward.
    pub dense: Vec<bool>,
    pub hc: usize,
    pub sinkhorn: usize,
    pub hc_eps: f64,
    pub eps: f64,
    pub limit: f64,
    pub kda_heads: usize,
    pub kda_dim: usize,
    pub kernel: usize,
    pub lower_bound: f64,
    pub heads: usize,
    pub q_rank: usize,
    pub kv_rank: usize,
    pub qk: usize,
    pub v: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub pool: usize,
    pub pool_topk: usize,
    pub inter: usize,
    pub moe_inter: usize,
    pub experts: usize,
    pub top_k: usize,
    pub shared: usize,
    pub route_scale: f64,
}

impl Dims {
    pub(super) fn of(config: &serde_json::Value) -> Self {
        let text = &config["text_config"];
        let n = |k: &str| crate::registry::Fixture::count(text, k);
        let f = |k: &str| text[k].as_f64().unwrap();
        let linear = &text["linear_attn_config"];
        let layers = n("num_hidden_layers");
        let has = |kind: &str| {
            text["layer_types"]
                .as_array()
                .unwrap()
                .iter()
                .any(|t| t == kind)
        };
        // A dimension of a layer kind no layer has is not in the config.
        let (dsa, kda) = (has("deepseek_sparse_attention"), has("linear_attention"));
        let opt = |present: bool, k: &str| if present { n(k) } else { 0 };
        let kn = |k: &str| {
            if kda {
                linear[k].as_u64().unwrap() as usize
            } else {
                0
            }
        };
        let pool = opt(dsa, "index_kpool");
        Self {
            vocab: n("vocab_size"),
            width: n("hidden_size"),
            layers,
            kda: (0..layers)
                .map(|l| text["layer_types"][l] == "linear_attention")
                .collect(),
            dense: (0..layers)
                .map(|l| text["mlp_layer_types"][l] == "dense")
                .collect(),
            hc: n("hc_mult"),
            sinkhorn: n("hc_sinkhorn_iters"),
            hc_eps: f("hc_eps"),
            eps: f("rms_norm_eps"),
            limit: f("swiglu_limit"),
            kda_heads: kn("num_heads"),
            kda_dim: kn("head_dim"),
            kernel: kn("short_conv_kernel_size"),
            lower_bound: if kda {
                linear["gate_lower_bound"].as_f64().unwrap()
            } else {
                0.0
            },
            heads: opt(dsa, "num_attention_heads"),
            q_rank: opt(dsa, "q_lora_rank"),
            kv_rank: opt(dsa, "kv_lora_rank"),
            qk: opt(dsa, "qk_nope_head_dim"),
            v: opt(dsa, "v_head_dim"),
            index_heads: opt(dsa, "index_n_heads"),
            index_dim: opt(dsa, "index_head_dim"),
            pool,
            pool_topk: if dsa { n("index_topk") / pool } else { 0 },
            inter: n("intermediate_size"),
            moe_inter: n("moe_intermediate_size"),
            experts: n("n_routed_experts"),
            top_k: n("num_experts_per_tok"),
            shared: n("n_shared_experts"),
            route_scale: f("routed_scaling_factor"),
        }
    }
}

/// What a run returns: the logits of every position and the carried state at the end, by the name
/// the traced graph gives it, laid out as the graph carries it with zeros past the last position.
pub(super) struct Run {
    pub logits: Vec<Vec<f64>>,
    /// The closing norm's output at every position: the mean of the streams, normalized, before the
    /// head reads it (what the hidden-state carry holds).
    pub hidden: Vec<Vec<f64>>,
    pub state: Vec<(String, Vec<f64>)>,
}

/// A draft chain of the multi-token-prediction layer: the logits of every draft step and the greedy
/// token each drew.
pub(super) struct MtpChain {
    pub logits: Vec<Vec<f64>>,
    pub drafts: Vec<usize>,
}

pub(super) struct Reference {
    d: Dims,
    w: HashMap<String, Vec<f64>>,
}

fn matvec(w: &[f64], out: usize, k: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(w.len(), out * k);
    assert_eq!(x.len(), k);
    (0..out)
        .map(|o| (0..k).map(|i| w[o * k + i] * x[i]).sum())
        .collect()
}

fn rms(x: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * inv).collect()
}

fn rms_scaled(x: &[f64], scale: &[f64], eps: f64) -> Vec<f64> {
    rms(x, eps).iter().zip(scale).map(|(v, s)| v * s).collect()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

/// The histories one layer carries, indexed by absolute position.
#[derive(Default, Clone)]
struct LayerHistory {
    /// KDA: the raw conv inputs of q, k and v (the `kernel - 1` zero rows before the first token,
    /// then one per token) and the matrices `[head][key channel][value channel]`.
    raw: [Vec<Vec<f64>>; 3],
    matrices: Vec<f64>,
    /// DSA: the normalized latents, the indexer keys and the indexer gates.
    latents: Vec<Vec<f64>>,
    keys: Vec<Vec<f64>>,
    gates: Vec<Vec<f64>>,
}

impl Reference {
    pub(super) fn new(config: &serde_json::Value, store: &WeightStore) -> Self {
        let d = Dims::of(config);
        let w = store
            .iter()
            .map(|(key, entry)| {
                (
                    key.as_str().to_string(),
                    values(entry).into_iter().map(f64::from).collect(),
                )
            })
            .collect();
        Self { d, w }
    }

    fn t(&self, name: &str) -> &[f64] {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("the fixture has no tensor {name}"))
    }

    /// One mHC site: `(post, comb, collapsed)` of the streams `[hc][width]`.
    fn mhc(&self, streams: &[Vec<f64>], prefix: &str) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
        let d = &self.d;
        let hc = d.hc;
        let flat: Vec<f64> = streams.iter().flatten().copied().collect();
        let mix = matvec(
            self.t(&format!("{prefix}_fn")),
            (2 + hc) * hc,
            hc * d.width,
            &rms(&flat, d.eps),
        );
        let (base, scale) = (
            self.t(&format!("{prefix}_base")),
            self.t(&format!("{prefix}_scale")),
        );
        let pre: Vec<f64> = (0..hc)
            .map(|i| sigmoid(mix[i] * scale[0] + base[i]) + d.hc_eps)
            .collect();
        let post: Vec<f64> = (0..hc)
            .map(|i| 2.0 * sigmoid(mix[hc + i] * scale[1] + base[hc + i]))
            .collect();
        let mut comb = vec![vec![0.0; hc]; hc];
        for (j, row) in comb.iter_mut().enumerate() {
            let logits: Vec<f64> = (0..hc)
                .map(|k| mix[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k])
                .collect();
            let max = logits.iter().cloned().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for k in 0..hc {
                row[k] = exps[k] / sum + d.hc_eps;
            }
        }
        let columns = |m: &mut Vec<Vec<f64>>| {
            for k in 0..hc {
                let s: f64 = (0..hc).map(|j| m[j][k]).sum::<f64>() + d.hc_eps;
                for row in m.iter_mut() {
                    row[k] /= s;
                }
            }
        };
        let rows = |m: &mut Vec<Vec<f64>>| {
            for row in m.iter_mut() {
                let s: f64 = row.iter().sum::<f64>() + d.hc_eps;
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
        };
        columns(&mut comb);
        for _ in 1..d.sinkhorn {
            rows(&mut comb);
            columns(&mut comb);
        }
        let mut collapsed = vec![0.0; d.width];
        for (s, stream) in streams.iter().enumerate() {
            for (c, v) in collapsed.iter_mut().zip(stream) {
                *c += pre[s] * v;
            }
        }
        (post, comb, collapsed)
    }

    fn place(
        &self,
        post: &[f64],
        comb: &[Vec<f64>],
        out: &[f64],
        streams: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        (0..self.d.hc)
            .map(|k| {
                (0..self.d.width)
                    .map(|c| {
                        post[k] * out[c]
                            + (0..self.d.hc)
                                .map(|j| comb[j][k] * streams[j][c])
                                .sum::<f64>()
                    })
                    .collect()
            })
            .collect()
    }

    /// The KDA block of `layer` for the token with normalized input `x`.
    fn kda(&self, layer: usize, x: &[f64], hist: &mut LayerHistory) -> Vec<f64> {
        let d = &self.d;
        let (heads, dim, kernel) = (d.kda_heads, d.kda_dim, d.kernel);
        let qkv = heads * dim;
        let at = |s: &str| format!("{TOWER}layers.{layer}.self_attn.{s}");
        if hist.raw[0].is_empty() {
            hist.raw = std::array::from_fn(|_| vec![vec![0.0; qkv]; kernel - 1]);
            hist.matrices = vec![0.0; heads * dim * dim];
        }
        let mut conv: [Vec<f64>; 3] = Default::default();
        for (i, (proj, kern)) in [
            ("q_proj", "q_conv1d"),
            ("k_proj", "k_conv1d"),
            ("v_proj", "v_conv1d"),
        ]
        .into_iter()
        .enumerate()
        {
            hist.raw[i].push(matvec(
                self.t(&at(&format!("{proj}.weight"))),
                qkv,
                d.width,
                x,
            ));
            let n = hist.raw[i].len();
            let kern = self.t(&at(&format!("{kern}.weight")));
            conv[i] = (0..qkv)
                .map(|ch| {
                    silu(
                        (0..kernel)
                            .map(|j| kern[ch * kernel + j] * hist.raw[i][n - kernel + j][ch])
                            .sum(),
                    )
                })
                .collect();
        }
        let low_rank = |a: &str, b: &str| {
            matvec(
                self.t(&at(&format!("{b}.weight"))),
                qkv,
                dim,
                &matvec(self.t(&at(&format!("{a}.weight"))), dim, d.width, x),
            )
        };
        let forget = low_rank("f_a_proj", "f_b_proj");
        let output_gate = low_rank("g_a_proj", "g_b_proj");
        let (dt, a_log) = (self.t(&at("dt_bias")), self.t(&at("A_log")));
        let beta = matvec(self.t(&at("b_proj.weight")), heads, d.width, x);
        let o_norm = self.t(&at("o_norm.weight"));
        let l2 = |head: &[f64]| -> Vec<f64> {
            let norm = (head.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
            head.iter().map(|v| v / norm).collect()
        };
        let mut flat = vec![0.0f64; qkv];
        for h in 0..heads {
            let range = h * dim..(h + 1) * dim;
            let q = l2(&conv[0][range.clone()]);
            let k = l2(&conv[1][range.clone()]);
            let value = &conv[2][range.clone()];
            let s = &mut hist.matrices[h * dim * dim..(h + 1) * dim * dim];
            for i in 0..dim {
                let ch = h * dim + i;
                let g = d.lower_bound * sigmoid(a_log[h].exp() * (forget[ch] + dt[ch]));
                s[i * dim..(i + 1) * dim]
                    .iter_mut()
                    .for_each(|e| *e *= g.exp());
            }
            let b = sigmoid(beta[h]);
            let delta: Vec<f64> = (0..dim)
                .map(|j| {
                    let read: f64 = (0..dim).map(|i| s[i * dim + j] * k[i]).sum();
                    b * (value[j] - read)
                })
                .collect();
            for i in 0..dim {
                for j in 0..dim {
                    s[i * dim + j] += k[i] * delta[j];
                }
            }
            let o: Vec<f64> = (0..dim)
                .map(|j| (0..dim).map(|i| s[i * dim + j] * q[i]).sum::<f64>() / (dim as f64).sqrt())
                .collect();
            let inv = 1.0 / (o.iter().map(|v| v * v).sum::<f64>() / dim as f64 + d.eps).sqrt();
            for e in 0..dim {
                flat[h * dim + e] = o[e] * inv * o_norm[e] * sigmoid(output_gate[h * dim + e]);
            }
        }
        matvec(self.t(&at("o_proj.weight")), d.width, qkv, &flat)
    }

    /// The DSA block of `layer` for the token at position `p` with normalized input `x`.
    fn dsa(&self, layer: usize, x: &[f64], p: usize, hist: &mut LayerHistory) -> Vec<f64> {
        let d = &self.d;
        let at = |s: &str| format!("{TOWER}layers.{layer}.self_attn.{s}");
        let (heads, qk, vd, pool) = (d.heads, d.qk, d.v, d.pool);
        let (ih, id) = (d.index_heads, d.index_dim);

        let q_res = rms_scaled(
            &matvec(self.t(&at("q_a_proj.weight")), d.q_rank, d.width, x),
            self.t(&at("q_a_layernorm.weight")),
            d.eps,
        );
        let q = matvec(self.t(&at("q_b_proj.weight")), heads * qk, d.q_rank, &q_res);
        hist.latents.push(rms_scaled(
            &matvec(
                self.t(&at("kv_a_proj_with_mqa.weight")),
                d.kv_rank,
                d.width,
                x,
            ),
            self.t(&at("kv_a_layernorm.weight")),
            d.eps,
        ));
        let raw = matvec(self.t(&at("indexer.wk.weight")), id, d.width, x);
        let mean = raw.iter().sum::<f64>() / id as f64;
        let var = raw.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / id as f64;
        let (scale, bias) = (
            self.t(&at("indexer.k_norm.weight")),
            self.t(&at("indexer.k_norm.bias")),
        );
        hist.keys.push(
            (0..id)
                .map(|j| (raw[j] - mean) / (var + 1e-6).sqrt() * scale[j] + bias[j])
                .collect(),
        );
        hist.gates.push(matvec(
            self.t(&at("indexer.index_kpool_compress_gate")),
            id,
            d.width,
            x,
        ));

        // The pools complete at this position, scored from their pooled keys.
        let complete = (p + 1) / pool;
        let ape = self.t(&at("indexer.index_kpool_compress_ape"));
        let q_idx = matvec(
            self.t(&at("indexer.wq_b.weight")),
            ih * id,
            d.q_rank,
            &q_res,
        );
        let head_w = matvec(self.t(&at("indexer.weights_proj.weight")), ih, d.width, x);
        let scores: Vec<f64> = (0..complete)
            .map(|j| {
                let pooled: Vec<f64> = (0..id)
                    .map(|f| {
                        let logits: Vec<f64> = (0..pool)
                            .map(|l| hist.gates[j * pool + l][f] + ape[l * id + f])
                            .collect();
                        let max = logits.iter().cloned().fold(f64::MIN, f64::max);
                        let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                        let sum: f64 = exps.iter().sum();
                        (0..pool)
                            .map(|l| exps[l] / sum * hist.keys[j * pool + l][f])
                            .sum()
                    })
                    .collect();
                (0..ih)
                    .map(|h| {
                        let dot: f64 = (0..id).map(|f| q_idx[h * id + f] * pooled[f]).sum::<f64>()
                            / (id as f64).sqrt();
                        dot.max(0.0) * head_w[h] / (ih as f64).sqrt()
                    })
                    .sum()
            })
            .collect();
        // The best pools, equal scores keeping the lower pool, and every position after the last
        // complete pool.
        let mut order: Vec<usize> = (0..complete).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let mut selected: Vec<usize> = order[..d.pool_topk.min(complete)]
            .iter()
            .flat_map(|&j| (0..pool).map(move |l| j * pool + l))
            .collect();
        selected.extend(complete * pool..=p);

        let kv_b = self.t(&at("kv_b_proj.weight"));
        let up: Vec<Vec<f64>> = selected
            .iter()
            .map(|&s| matvec(kv_b, heads * (qk + vd), d.kv_rank, &hist.latents[s]))
            .collect();
        let mut attended = vec![0.0; heads * vd];
        for h in 0..heads {
            let span = |u: &[f64]| u[h * (qk + vd)..(h + 1) * (qk + vd)].to_vec();
            let logits: Vec<f64> = up
                .iter()
                .map(|u| {
                    let u = span(u);
                    (0..qk).map(|f| q[h * qk + f] * u[f]).sum::<f64>() / (qk as f64).sqrt()
                })
                .collect();
            let max = logits.iter().cloned().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for (u, e) in up.iter().zip(&exps) {
                let u = span(u);
                for f in 0..vd {
                    attended[h * vd + f] += e / sum * u[qk + f];
                }
            }
        }
        matvec(self.t(&at("o_proj.weight")), d.width, heads * vd, &attended)
    }

    /// One gated feed-forward block: `down(silu(min(gate, limit)) * clamp(up, -limit, limit))`.
    fn gated(&self, prefix: &str, inter: usize, x: &[f64]) -> Vec<f64> {
        let d = &self.d;
        let w = |n: &str| self.t(&format!("{prefix}.{n}_proj.weight"));
        let gate = matvec(w("gate"), inter, d.width, x);
        let up = matvec(w("up"), inter, d.width, x);
        let act: Vec<f64> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| silu(g.min(d.limit)) * u.clamp(-d.limit, d.limit))
            .collect();
        matvec(w("down"), d.width, inter, &act)
    }

    /// The routed feed-forward of `layer`: the shared expert plus the `top_k` experts of the
    /// sigmoid scores (selected with the correction bias, gated by the unbiased scores normalized
    /// over the selection and scaled).
    fn moe(&self, layer: usize, x: &[f64]) -> Vec<f64> {
        let d = &self.d;
        let at = |s: &str| format!("{TOWER}layers.{layer}.mlp.{s}");
        let logits = matvec(self.t(&at("gate.weight")), d.experts, d.width, x);
        let score: Vec<f64> = logits.iter().map(|&l| sigmoid(l)).collect();
        let bias = self.t(&at("gate.e_score_correction_bias"));
        let mut order: Vec<usize> = (0..d.experts).collect();
        order.sort_by(|&a, &b| {
            (score[b] + bias[b])
                .partial_cmp(&(score[a] + bias[a]))
                .unwrap()
                .then(a.cmp(&b))
        });
        let sum: f64 = order[..d.top_k].iter().map(|&e| score[e]).sum();
        let mut out = self.gated(&at("shared_experts"), d.moe_inter * d.shared, x);
        for &e in &order[..d.top_k] {
            let weight = score[e] / sum * d.route_scale;
            let expert = self.gated(&at(&format!("experts.{e}")), d.moe_inter, x);
            for (o, v) in out.iter_mut().zip(expert) {
                *o += v * weight;
            }
        }
        out
    }

    /// The multi-token-prediction layer for the token `token` that follows position `p`, whose
    /// target hidden state is `hidden` (the closing norm's output): `(post-norm state, logits)`. The
    /// block is a DSA layer at index `layers` over the routed or dense feed-forward of the last text
    /// layer, with no hyper-connections; the embedding is zero at position 0.
    fn mtp_step(
        &self,
        hidden: &[f64],
        token: usize,
        p: usize,
        hist: &mut LayerHistory,
    ) -> (Vec<f64>, Vec<f64>) {
        let d = &self.d;
        let layer = d.layers;
        let at = |s: &str| format!("{TOWER}layers.{layer}.{s}");
        let embed = if p == 0 {
            vec![0.0; d.width]
        } else {
            self.t(&format!("{TOWER}embed_tokens.weight"))[token * d.width..(token + 1) * d.width]
                .to_vec()
        };
        let mut fused_input = rms_scaled(&embed, self.t(&at("enorm.weight")), d.eps);
        fused_input.extend(rms_scaled(hidden, self.t(&at("hnorm.weight")), d.eps));
        let x = matvec(
            self.t(&at("eh_proj.weight")),
            d.width,
            2 * d.width,
            &fused_input,
        );
        let normed = rms_scaled(&x, self.t(&at("input_layernorm.weight")), d.eps);
        let mixed = self.dsa(layer, &normed, p, hist);
        let r: Vec<f64> = x.iter().zip(&mixed).map(|(a, b)| a + b).collect();
        let normed = rms_scaled(&r, self.t(&at("post_attention_layernorm.weight")), d.eps);
        let ffn = if *d.dense.last().unwrap() {
            self.gated(&at("mlp"), d.inter, &normed)
        } else {
            self.moe(layer, &normed)
        };
        let out: Vec<f64> = r.iter().zip(&ffn).map(|(a, b)| a + b).collect();
        let normed = rms_scaled(&out, self.t(&at("shared_head.norm.weight")), d.eps);
        let logits = matvec(self.t("lm_head.weight"), d.vocab, d.width, &normed);
        (normed, logits)
    }

    /// The draft chain from scratch: the target over `tokens` (positions `0..n`) in f64, then the
    /// layer over every position `p < n` with the token that follows it (the last with `bonus`, the
    /// token the target chose after the prefix), one position at a time with no cache but its own
    /// history; then `k - 1` more steps, each fed the previous step's post-norm state and greedy
    /// token at the next position. Returns the logits of the `k` draft steps and the tokens drawn.
    pub(super) fn mtp_chain(&self, tokens: &[usize], bonus: usize, k: usize) -> MtpChain {
        let target = self.run(tokens, tokens.len());
        let n = tokens.len();
        let mut hist = LayerHistory::default();
        let mut last = (Vec::new(), Vec::new());
        for (p, hidden) in target.hidden.iter().enumerate() {
            let next = tokens.get(p + 1).copied().unwrap_or(bonus);
            last = self.mtp_step(hidden, next, p, &mut hist);
        }
        let argmax = |v: &[f64]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        let mut chain = MtpChain {
            logits: vec![last.1.clone()],
            drafts: vec![argmax(&last.1)],
        };
        for i in 1..k {
            last = self.mtp_step(&last.0, chain.drafts[i - 1], n - 1 + i, &mut hist);
            chain.drafts.push(argmax(&last.1));
            chain.logits.push(last.1.clone());
        }
        chain
    }

    /// The model over `tokens` from position 0 with a cache of `capacity` positions.
    pub(super) fn run(&self, tokens: &[usize], capacity: usize) -> Run {
        let d = &self.d;
        let mut hist = vec![LayerHistory::default(); d.layers];
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        for (p, &token) in tokens.iter().enumerate() {
            let embed = &self.t(&format!("{TOWER}embed_tokens.weight"))
                [token * d.width..(token + 1) * d.width];
            let mut streams: Vec<Vec<f64>> = vec![embed.to_vec(); d.hc];
            for (layer, h) in hist.iter_mut().enumerate() {
                let prefix = format!("{TOWER}layers.{layer}");
                let (post, comb, collapsed) = self.mhc(&streams, &format!("{prefix}.hc_attn"));
                let normed = rms_scaled(
                    &collapsed,
                    self.t(&format!("{prefix}.input_layernorm.weight")),
                    d.eps,
                );
                let mixed = if d.kda[layer] {
                    self.kda(layer, &normed, h)
                } else {
                    self.dsa(layer, &normed, p, h)
                };
                streams = self.place(&post, &comb, &mixed, &streams);

                let (post, comb, collapsed) = self.mhc(&streams, &format!("{prefix}.hc_ffn"));
                let normed = rms_scaled(
                    &collapsed,
                    self.t(&format!("{prefix}.post_attention_layernorm.weight")),
                    d.eps,
                );
                let out = if d.dense[layer] {
                    self.gated(&format!("{prefix}.mlp"), d.inter, &normed)
                } else {
                    self.moe(layer, &normed)
                };
                streams = self.place(&post, &comb, &out, &streams);
            }
            let mean: Vec<f64> = (0..d.width)
                .map(|c| streams.iter().map(|s| s[c]).sum::<f64>() / d.hc as f64)
                .collect();
            let normed = rms_scaled(&mean, self.t(&format!("{TOWER}norm.weight")), d.eps);
            logits.push(matvec(self.t("lm_head.weight"), d.vocab, d.width, &normed));
            hidden.push(normed);
        }

        // The carried state, in the order the graph carries it: per layer, the conv windows and
        // the matrices of a KDA layer, the latent and indexer caches of a DSA layer.
        let pad = |rows: &[Vec<f64>], width: usize| {
            let mut out = vec![0.0; capacity * width];
            for (p, row) in rows.iter().enumerate() {
                out[p * width..(p + 1) * width].copy_from_slice(row);
            }
            out
        };
        let mut state = Vec::new();
        for (layer, h) in hist.iter().enumerate() {
            if d.kda[layer] {
                for (i, name) in ["conv_q", "conv_k", "conv_v"].into_iter().enumerate() {
                    let window = &h.raw[i][h.raw[i].len() - (d.kernel - 1)..];
                    state.push((format!("kda.l{layer}.{name}"), window.concat()));
                }
                state.push((format!("kda.l{layer}.state"), h.matrices.clone()));
            } else {
                let rows: Vec<Vec<f64>> = h
                    .keys
                    .iter()
                    .zip(&h.gates)
                    .map(|(k, g)| [k.clone(), g.clone()].concat())
                    .collect();
                state.push((format!("kv.l{layer}.c"), pad(&h.latents, d.kv_rank)));
                state.push((format!("kv.l{layer}.index"), pad(&rows, 2 * d.index_dim)));
            }
        }
        Run {
            logits,
            hidden,
            state,
        }
    }
}

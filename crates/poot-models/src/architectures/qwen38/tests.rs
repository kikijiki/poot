//! Qwen4Exp on the CPU oracle: the model against an independent plain-loop reference of the
//! reference model, built from the same checkpoint bytes and sharing no code with the tracer.
//!
//! The reference ([`reference`]) is written in f64, over a whole sequence at once (no carried
//! state): the Hyper-Connections gates, the PLE hash and layer (the component's own plain-loop
//! oracle), the Gated DeltaNet recurrence one token at a time, QSA by a plain sort and a softmax over
//! the selected keys only, and the routed experts by sorting the router's probabilities. What it
//! reports are the logits of every position and every element the step carries (the KV and indexer
//! caches, the conv windows, the delta-rule matrices, the n-gram history), so a model that gets a
//! prompt right but carries the wrong state fails the continuation rows.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use super::*;
use crate::components::ple::oracle as ple_oracle;
use crate::components::standard::oracle::run_step;
use crate::components::testing::values;
use poot_quant::weights::WeightStore;

use crate::model::{KvLayout, LogitRows};
use crate::registry::Registry;

/// The cache capacity the rows trace at: a multiple of the fixture's block of two, and exactly
/// the prompt's length, so every position of a continuation is a position of the reference.
const CAP: usize = 8;

/// A prompt with the end-of-sequence id inside it (so the n-gram segment rule matters), tokens
/// that route to different experts and more than the budget's two blocks complete at the end (so the
/// selection drops one).
const TOKENS: [i32; 8] = [3, 17, 47, 5, 22, 9, 31, 12];

/// A plain-loop f64 reference of the family, over the fixture's checkpoint bytes.
mod reference {
    use std::collections::HashMap;

    use super::*;

    type Row = Vec<f64>;

    const ROOT: &str = "model.language_model";

    fn matvec(w: &[f64], out: usize, k: usize, x: &[f64]) -> Row {
        assert_eq!(w.len(), out * k, "weight is [out, in]");
        (0..out)
            .map(|o| (0..k).map(|j| w[o * k + j] * x[j]).sum())
            .collect()
    }

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    fn silu(v: f64) -> f64 {
        v * sigmoid(v)
    }

    fn softplus(v: f64) -> f64 {
        (1.0 + v.exp()).ln()
    }

    /// `x * rsqrt(mean(x^2) + eps) * (1 + w)`: the reference model's `Qwen4ExpTextRMSNorm`.
    fn plus_one_rms(x: &[f64], w: &[f64], eps: f64) -> Row {
        let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v * inv * (1.0 + w)).collect()
    }

    /// The same norm over each of `groups` runs of `x`, with the one full-width scale.
    fn grouped(x: &[f64], w: &[f64], groups: usize, eps: f64) -> Row {
        let width = x.len() / groups;
        (0..groups)
            .flat_map(|g| {
                let part = &x[g * width..(g + 1) * width];
                let ms = part.iter().map(|v| v * v).sum::<f64>() / width as f64;
                let inv = 1.0 / (ms + eps).sqrt();
                (0..width).map(move |i| part[i] * inv * (1.0 + w[g * width + i]))
            })
            .collect()
    }

    /// Half-split RoPE of the leading `rot` dims of `x` at `pos`.
    fn rope(x: &[f64], pos: usize, rot: usize, theta: f64) -> Row {
        let half = rot / 2;
        let mut out = x.to_vec();
        for i in 0..half {
            let angle = pos as f64 * theta.powf(-2.0 * i as f64 / rot as f64);
            let (s, c) = angle.sin_cos();
            out[i] = x[i] * c - x[i + half] * s;
            out[i + half] = x[i + half] * c + x[i] * s;
        }
        out
    }

    /// The model config the reference reads, straight from the fixture's JSON.
    pub(super) struct Cfg {
        width: usize,
        vocab: usize,
        kinds: Vec<bool>,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        rotary: usize,
        eps: f64,
        theta: f64,
        index_heads: usize,
        index_dim: usize,
        index_budget: usize,
        ratio: usize,
        key_heads: usize,
        value_heads: usize,
        gdn_dim: usize,
        conv_kernel: usize,
        experts: usize,
        top_k: usize,
        inter: usize,
        shared: usize,
        streams: usize,
        ple: Vec<Option<crate::components::ple::PleSpec>>,
    }

    impl Cfg {
        fn new(config: &serde_json::Value) -> Self {
            let t = &config["text_config"];
            let n = |f: &str| crate::registry::Fixture::count(t, f);
            let head_dim = n("head_dim");
            let params = Qwen4ExpParams::from_raw(&RawConfig::HfJson {
                config,
                generation: None,
            })
            .unwrap();
            Self {
                width: n("hidden_size"),
                vocab: n("vocab_size"),
                kinds: t["layer_types"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|k| k == "full_attention")
                    .collect(),
                heads: n("num_attention_heads"),
                kv_heads: n("num_key_value_heads"),
                head_dim,
                rotary: (head_dim as f64 * t["partial_rotary_factor"].as_f64().unwrap()).round()
                    as usize,
                eps: t["rms_norm_eps"].as_f64().unwrap(),
                theta: t["rope_parameters"]["rope_theta"].as_f64().unwrap(),
                index_heads: n("indexer_n_heads"),
                index_dim: n("indexer_head_dim"),
                index_budget: n("indexer_budget"),
                ratio: n("indexer_compress_ratio"),
                key_heads: n("linear_num_key_heads"),
                value_heads: n("linear_num_value_heads"),
                gdn_dim: n("linear_key_head_dim"),
                conv_kernel: n("linear_conv_kernel_dim"),
                experts: n("num_experts"),
                top_k: n("num_experts_per_tok"),
                inter: n("moe_intermediate_size"),
                shared: n("shared_expert_intermediate_size"),
                streams: n("hc_count"),
                ple: params
                    .ple
                    .iter()
                    .map(|p| p.as_ref().map(|p| *p.spec()))
                    .collect(),
            }
        }
    }

    /// What a whole-sequence run reports: every position's logits and every carried element, by
    /// the state's name, shaped at a cache of `cap` positions.
    pub(super) struct Run {
        pub(super) logits: Vec<Row>,
        pub(super) state: BTreeMap<String, Row>,
    }

    pub(super) struct Reference {
        cfg: Cfg,
        tensors: HashMap<String, Row>,
    }

    impl Reference {
        pub(super) fn new(fixture: &Fixture) -> Self {
            Self {
                cfg: Cfg::new(&fixture.config),
                tensors: fixture
                    .store
                    .iter()
                    .map(|(key, entry)| {
                        (
                            key.as_str().to_string(),
                            values(entry).into_iter().map(f64::from).collect(),
                        )
                    })
                    .collect(),
            }
        }

        fn t(&self, name: &str) -> &[f64] {
            self.tensors
                .get(name)
                .unwrap_or_else(|| panic!("the fixture has no tensor {name}"))
        }

        fn layer(&self, l: usize, suffix: &str) -> &[f64] {
            self.t(&format!("{ROOT}.layers.{l}.{suffix}"))
        }

        /// A Hyper-Connections gate at `stem`: the block input and, for a layer's gate, the
        /// per-stream injection weights.
        fn gate(&self, x: &[f64], stem: &str, inject: bool) -> (Row, Option<Row>) {
            let c = &self.cfg;
            let (h, hc) = (c.width, c.streams);
            let stream = h * hc;
            let rank = self
                .t(&format!("{stem}.input_mix_weight_down.weight"))
                .len()
                / stream;
            let normed = grouped(x, self.t(&format!("{stem}.hc_norm.weight")), hc, c.eps);
            let down = matvec(
                self.t(&format!("{stem}.input_mix_weight_down.weight")),
                rank,
                stream,
                &normed,
            );
            let act: Row = down.iter().map(|&v| silu(v / hc as f64)).collect();
            let up = matvec(
                self.t(&format!("{stem}.input_mix_weight_up.weight")),
                stream,
                rank,
                &act,
            );
            let mixed = (0..h)
                .map(|i| {
                    (0..hc)
                        .map(|g| sigmoid(up[g * h + i]) * normed[g * h + i])
                        .sum::<f64>()
                        / hc as f64
                })
                .collect();
            let weights = inject.then(|| {
                matvec(
                    self.t(&format!("{stem}.block_inject_weight.weight")),
                    hc,
                    stream,
                    &normed,
                )
                .into_iter()
                .map(|v| 2.0 * sigmoid(v / hc as f64))
                .collect()
            });
            (mixed, weights)
        }

        fn add_into_streams(&self, x: &mut Row, block: &[f64], weights: &[f64]) {
            let h = self.cfg.width;
            for (g, w) in weights.iter().enumerate() {
                for i in 0..h {
                    x[g * h + i] += block[i] * w;
                }
            }
        }

        /// One expert's `(gate, up, down)` matrices, `[inter, width]`, `[inter, width]`, `[width,
        /// inter]`, from the fused stacks or the per-expert tensors, whichever the store holds.
        fn expert(&self, l: usize, e: usize) -> (Row, Row, Row) {
            let c = &self.cfg;
            let (w, i) = (c.width, c.inter);
            let key = format!("{ROOT}.layers.{l}.mlp.experts");
            match self.tensors.get(&format!("{key}.gate_up_proj")) {
                Some(gate_up) => {
                    let slab = &gate_up[e * 2 * i * w..(e + 1) * 2 * i * w];
                    let down = self.t(&format!("{key}.down_proj"));
                    (
                        slab[..i * w].to_vec(),
                        slab[i * w..].to_vec(),
                        down[e * w * i..(e + 1) * w * i].to_vec(),
                    )
                }
                None => (
                    self.t(&format!("{key}.{e}.gate_proj")).to_vec(),
                    self.t(&format!("{key}.{e}.up_proj")).to_vec(),
                    self.t(&format!("{key}.{e}.down_proj")).to_vec(),
                ),
            }
        }

        /// The routed block of one position: softmax over every expert, the best `top_k` by a plain
        /// sort (ties to the lower expert), renormalized, plus the sigmoid-gated shared expert.
        fn moe(&self, l: usize, x: &[f64]) -> Row {
            let c = &self.cfg;
            let (w, i) = (c.width, c.inter);
            let logits = matvec(self.layer(l, "mlp.gate.weight"), c.experts, w, x);
            let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp: Row = logits.iter().map(|v| (v - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let probs: Row = exp.iter().map(|v| v / sum).collect();
            let mut order: Vec<usize> = (0..c.experts).collect();
            order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
            let chosen = &order[..c.top_k];
            let kept: f64 = chosen.iter().map(|&e| probs[e]).sum();
            let mut out = vec![0.0; w];
            for &e in chosen {
                let (gate, up, down) = self.expert(l, e);
                let (g, u) = (matvec(&gate, i, w, x), matvec(&up, i, w, x));
                let act: Row = g.iter().zip(&u).map(|(&g, &u)| silu(g) * u).collect();
                for (o, v) in out.iter_mut().zip(matvec(&down, w, i, &act)) {
                    *o += probs[e] / kept * v;
                }
            }
            let s = c.shared;
            let sg = matvec(self.layer(l, "mlp.shared_expert.gate_proj.weight"), s, w, x);
            let su = matvec(self.layer(l, "mlp.shared_expert.up_proj.weight"), s, w, x);
            let act: Row = sg.iter().zip(&su).map(|(&g, &u)| silu(g) * u).collect();
            let shared = matvec(
                self.layer(l, "mlp.shared_expert.down_proj.weight"),
                w,
                s,
                &act,
            );
            let gate = sigmoid(matvec(self.layer(l, "mlp.shared_expert_gate.weight"), 1, w, x)[0]);
            out.iter().zip(&shared).map(|(o, s)| o + gate * s).collect()
        }

        /// The Gated DeltaNet layer over a whole sequence, one token at a time; reports the conv
        /// window and the state matrices after the last token.
        fn gdn(&self, l: usize, h: &[Row], state: &mut BTreeMap<String, Row>) -> Vec<Row> {
            let c = &self.cfg;
            let (hk, hv, d, k) = (c.key_heads, c.value_heads, c.gdn_dim, c.conv_kernel);
            let (key_dim, value_dim) = (hk * d, hv * d);
            let conv = 2 * key_dim + value_dim;
            let n = h.len();
            let project = |name: &str, out: usize| -> Vec<Row> {
                h.iter()
                    .map(|x| matvec(self.layer(l, name), out, c.width, x))
                    .collect()
            };
            let (qkv, z) = (
                project("linear_attn.in_proj_qkv.weight", conv),
                project("linear_attn.in_proj_z.weight", value_dim),
            );
            let (b, a) = (
                project("linear_attn.in_proj_b.weight", hv),
                project("linear_attn.in_proj_a.weight", hv),
            );
            let taps = self.layer(l, "linear_attn.conv1d.weight"); // [conv, 1, K]
            let conv_out: Vec<Row> = (0..n)
                .map(|t| {
                    (0..conv)
                        .map(|ch| {
                            let mut acc = 0.0;
                            for j in 0..k {
                                if let Some(src) = t.checked_sub(k - 1 - j) {
                                    acc += taps[ch * k + j] * qkv[src][ch];
                                }
                            }
                            silu(acc)
                        })
                        .collect()
                })
                .collect();
            let l2 = |x: &[f64]| -> Row {
                let norm = (x.iter().map(|v| v * v).sum::<f64>() + c.eps).sqrt();
                x.iter().map(|v| v / norm).collect()
            };
            let (dt_bias, a_log) = (
                self.layer(l, "linear_attn.dt_bias"),
                self.layer(l, "linear_attn.A_log"),
            );
            let norm_scale = self.layer(l, "linear_attn.norm.weight");
            let rep = hv / hk;
            let mut s = vec![vec![vec![0.0; d]; d]; hv];
            let mut out = Vec::new();
            for t in 0..n {
                let mut row = vec![0.0; value_dim];
                for vh in 0..hv {
                    // V heads group by key head: value head `vh` reads key head `vh / rep`.
                    let kh = vh / rep;
                    let q = l2(&conv_out[t][kh * d..(kh + 1) * d]);
                    let key = l2(&conv_out[t][key_dim + kh * d..key_dim + (kh + 1) * d]);
                    let v = &conv_out[t][2 * key_dim + vh * d..2 * key_dim + (vh + 1) * d];
                    let g = softplus(a[t][vh] + dt_bias[vh]) * -a_log[vh].exp();
                    let beta = sigmoid(b[t][vh]);
                    for r in s[vh].iter_mut() {
                        for cell in r.iter_mut() {
                            *cell *= g.exp();
                        }
                    }
                    let delta: Row = (0..d)
                        .map(|j| beta * (v[j] - (0..d).map(|i| s[vh][i][j] * key[i]).sum::<f64>()))
                        .collect();
                    for i in 0..d {
                        for j in 0..d {
                            s[vh][i][j] += key[i] * delta[j];
                        }
                    }
                    let o: Row = (0..d)
                        .map(|j| {
                            (0..d).map(|i| s[vh][i][j] * q[i]).sum::<f64>() / (d as f64).sqrt()
                        })
                        .collect();
                    let ms = o.iter().map(|v| v * v).sum::<f64>() / d as f64;
                    let inv = 1.0 / (ms + c.eps).sqrt();
                    for j in 0..d {
                        row[vh * d + j] = o[j] * inv * norm_scale[j] * silu(z[t][vh * d + j]);
                    }
                }
                out.push(matvec(
                    self.layer(l, "linear_attn.out_proj.weight"),
                    c.width,
                    value_dim,
                    &row,
                ));
            }
            // The conv window: the last K - 1 inputs, oldest first, zeros before the sequence.
            let window: Row = (n as i64 - (k as i64 - 1)..n as i64)
                .flat_map(|t| {
                    if t < 0 {
                        vec![0.0; conv]
                    } else {
                        qkv[t as usize].clone()
                    }
                })
                .collect();
            state.insert(format!("gdn.l{l}.conv"), window);
            state.insert(
                format!("gdn.l{l}.ssm"),
                s.into_iter().flatten().flatten().collect(),
            );
            out
        }

        /// The QSA layer: the indexer's blocks by a plain sort, attention over the selected keys
        /// only. Reports the keys, values, block keys and unfinished-block raw keys the layer caches.
        fn qsa(
            &self,
            l: usize,
            h: &[Row],
            cap: usize,
            state: &mut BTreeMap<String, Row>,
        ) -> Vec<Row> {
            let c = &self.cfg;
            let n = h.len();
            let (w, rot, theta, eps) = (c.width, c.rotary, c.theta, c.eps);
            let (hi, di, ratio) = (c.index_heads, c.index_dim, c.ratio);
            let (nh, nkv, hd) = (c.heads, c.kv_heads, c.head_dim);
            let fused = self.layer(l, "self_attn.indexer.index_qk_proj.weight");
            let (wq, wk) = fused.split_at(hi * di * w);
            let (q_ln, k_ln) = (
                self.layer(l, "self_attn.indexer.q_layernorm.weight"),
                self.layer(l, "self_attn.indexer.k_layernorm.weight"),
            );
            let index_q: Vec<Vec<Row>> = (0..n)
                .map(|t| {
                    let raw = matvec(wq, hi * di, w, &h[t]);
                    (0..hi)
                        .map(|head| {
                            let normed = plus_one_rms(&raw[head * di..(head + 1) * di], q_ln, eps);
                            rope(&normed, t, rot, theta)
                        })
                        .collect()
                })
                .collect();
            let raw_k: Vec<Row> = h.iter().map(|x| matvec(wk, di, w, x)).collect();
            let block_keys: Vec<Row> = (0..n / ratio)
                .map(|b| {
                    let pooled: Row = (0..di)
                        .map(|j| {
                            raw_k[b * ratio..(b + 1) * ratio]
                                .iter()
                                .map(|r| r[j])
                                .sum::<f64>()
                                / ratio as f64
                        })
                        .collect();
                    rope(&plus_one_rms(&pooled, k_ln, eps), b * ratio, rot, theta)
                })
                .collect();

            let (q_norm, k_norm) = (
                self.layer(l, "self_attn.q_norm.weight"),
                self.layer(l, "self_attn.k_norm.weight"),
            );
            let mut q = vec![vec![Vec::new(); nh]; n];
            let mut gate = vec![vec![Vec::new(); nh]; n];
            let mut k = vec![vec![Vec::new(); nkv]; n];
            let mut v = vec![vec![Vec::new(); nkv]; n];
            for t in 0..n {
                let qg = matvec(
                    self.layer(l, "self_attn.q_proj.weight"),
                    nh * 2 * hd,
                    w,
                    &h[t],
                );
                for head in 0..nh {
                    let base = head * 2 * hd;
                    q[t][head] = rope(
                        &plus_one_rms(&qg[base..base + hd], q_norm, eps),
                        t,
                        rot,
                        theta,
                    );
                    gate[t][head] = qg[base + hd..base + 2 * hd]
                        .iter()
                        .map(|&g| sigmoid(g))
                        .collect();
                }
                let kr = matvec(self.layer(l, "self_attn.k_proj.weight"), nkv * hd, w, &h[t]);
                let vr = matvec(self.layer(l, "self_attn.v_proj.weight"), nkv * hd, w, &h[t]);
                for kv in 0..nkv {
                    k[t][kv] = rope(
                        &plus_one_rms(&kr[kv * hd..(kv + 1) * hd], k_norm, eps),
                        t,
                        rot,
                        theta,
                    );
                    v[t][kv] = vr[kv * hd..(kv + 1) * hd].to_vec();
                }
            }

            let topk = c.index_budget / ratio;
            let out = (0..n)
                .map(|t| {
                    let complete = (t + 1) / ratio;
                    let mut scored: Vec<(usize, f64)> = (0..complete)
                        .map(|b| {
                            let sum: f64 = index_q[t]
                                .iter()
                                .map(|head| {
                                    head.iter()
                                        .zip(&block_keys[b])
                                        .map(|(a, k)| a * k)
                                        .sum::<f64>()
                                        .max(0.0)
                                })
                                .sum();
                            (b, sum / (di as f64).sqrt())
                        })
                        .collect();
                    // `sort_by` is stable: equal scores keep the lower block first.
                    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                    let mut selected: Vec<usize> = scored[..topk.min(complete)]
                        .iter()
                        .flat_map(|&(b, _)| b * ratio..(b + 1) * ratio)
                        .collect();
                    // The query's own unfinished block is always visible.
                    if complete * ratio <= t {
                        selected.extend(complete * ratio..=t);
                    }
                    selected.sort_unstable();
                    let mut row = vec![0.0; nh * hd];
                    for head in 0..nh {
                        let kv = head / (nh / nkv);
                        let scores: Row = selected
                            .iter()
                            .map(|&j| {
                                q[t][head]
                                    .iter()
                                    .zip(&k[j][kv])
                                    .map(|(a, b)| a * b)
                                    .sum::<f64>()
                                    / (hd as f64).sqrt()
                            })
                            .collect();
                        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exp: Row = scores.iter().map(|s| (s - max).exp()).collect();
                        let sum: f64 = exp.iter().sum();
                        for d in 0..hd {
                            let acc: f64 = selected
                                .iter()
                                .enumerate()
                                .map(|(i, &j)| exp[i] / sum * v[j][kv][d])
                                .sum();
                            row[head * hd + d] = acc * gate[t][head][d];
                        }
                    }
                    matvec(self.layer(l, "self_attn.o_proj.weight"), w, nh * hd, &row)
                })
                .collect();

            let cache = |rows: &dyn Fn(usize, usize) -> Row, heads: usize, dim: usize| -> Row {
                (0..heads)
                    .flat_map(|head| {
                        (0..cap).flat_map(
                            move |p| {
                                if p < n { rows(p, head) } else { vec![0.0; dim] }
                            },
                        )
                    })
                    .collect()
            };
            state.insert(
                format!("kv.l{l}.k"),
                cache(&|p, kv| k[p][kv].clone(), nkv, hd),
            );
            state.insert(
                format!("kv.l{l}.v"),
                cache(&|p, kv| v[p][kv].clone(), nkv, hd),
            );
            // The block keys of the complete blocks, zeros after them; the unfinished block's raw keys
            // by residue: slot `j` holds the latest position `= j mod ratio`.
            let blocks = cap / ratio;
            state.insert(
                format!("kv.l{l}.index_blocks"),
                (0..blocks)
                    .flat_map(|b| block_keys.get(b).cloned().unwrap_or_else(|| vec![0.0; di]))
                    .collect(),
            );
            state.insert(
                format!("kv.l{l}.index_tail"),
                (0..ratio)
                    .flat_map(|j| {
                        (0..n)
                            .rev()
                            .find(|p| p % ratio == j)
                            .map_or_else(|| vec![0.0; di], |p| raw_k[p].clone())
                    })
                    .collect(),
            );
            out
        }

        /// The whole model over `tokens`, as one sequence from position 0.
        pub(super) fn forward(&self, tokens: &[i32], cap: usize) -> Run {
            let c = &self.cfg;
            let (h, hc) = (c.width, c.streams);
            let embed = self.t(&format!("{ROOT}.embed_tokens.weight"));
            let mut x: Vec<Row> = tokens
                .iter()
                .map(|&t| {
                    let row = &embed[t as usize * h..(t as usize + 1) * h];
                    (0..hc).flat_map(|_| row.to_vec()).collect()
                })
                .collect();
            let mut state = BTreeMap::new();
            for l in 0..c.kinds.len() {
                if let Some(spec) = &c.ple[l] {
                    let ctx = spec.ngram_size - 1;
                    let ids = ple_oracle::ngram_ids(
                        spec,
                        &ple_oracle::tables(spec),
                        &vec![spec.eos as i32; ctx],
                        tokens,
                    );
                    let layer = ple_oracle::Layer {
                        table: self
                            .layer(l, "ple.ple_embedding.ngram_embedding.weight")
                            .to_vec(),
                        key: self.layer(l, "ple.key_proj.weight").to_vec(),
                        value: self.layer(l, "ple.value_proj.weight").to_vec(),
                        key_norm: self.layer(l, "ple.norm_key.weight").to_vec(),
                        query_norm: self.layer(l, "ple.norm_query.weight").to_vec(),
                        conv_norm: self.layer(l, "ple.norm_conv.weight").to_vec(),
                        conv: self.layer(l, "ple.conv1d.weight").to_vec(),
                    };
                    let contribution = ple_oracle::contribution(spec, &layer, &x, &ids);
                    for (row, add) in x.iter_mut().zip(&contribution.out) {
                        for (v, a) in row.iter_mut().zip(add) {
                            *v += a;
                        }
                    }
                    // The history is the last `ctx` tokens, XOR-encoded; the conv window the last
                    // `(K - 1) * dilation` normed inputs, zeros before the sequence.
                    let mut history = vec![spec.eos as i32; ctx];
                    history.extend_from_slice(tokens);
                    state.insert(
                        format!("ple.l{l}.history"),
                        history[history.len() - ctx..]
                            .iter()
                            .map(|&t| f64::from(t ^ spec.eos as i32))
                            .collect(),
                    );
                    let window = (spec.conv_kernel - 1) * spec.ngram_size;
                    let stream = spec.streams * spec.width;
                    state.insert(
                        format!("ple.l{l}.conv"),
                        (tokens.len() as i64 - window as i64..tokens.len() as i64)
                            .flat_map(|t| {
                                if t < 0 {
                                    vec![0.0; stream]
                                } else {
                                    contribution.normed[t as usize].clone()
                                }
                            })
                            .collect(),
                    );
                }
                let layer_stem = |site: &str| format!("{ROOT}.layers.{l}.{site}");
                let gates: Vec<(Row, Row)> = x
                    .iter()
                    .map(|x| {
                        let (m, i) = self.gate(x, &layer_stem("attn_hyper_connection"), true);
                        (m, i.unwrap())
                    })
                    .collect();
                let inputs: Vec<Row> = gates.iter().map(|(m, _)| m.clone()).collect();
                let mixed = if c.kinds[l] {
                    self.qsa(l, &inputs, cap, &mut state)
                } else {
                    self.gdn(l, &inputs, &mut state)
                };
                for ((x, m), (_, inj)) in x.iter_mut().zip(&mixed).zip(&gates) {
                    self.add_into_streams(x, m, inj);
                }
                for x in x.iter_mut() {
                    let (m, inj) = self.gate(x, &layer_stem("mlp_hyper_connection"), true);
                    let routed = self.moe(l, &m);
                    self.add_into_streams(x, &routed, &inj.unwrap());
                }
            }
            let head = self.t("lm_head.weight");
            let logits = x
                .iter()
                .map(|x| {
                    let (m, _) = self.gate(x, &format!("{ROOT}.hyper_connection_mixer"), false);
                    matvec(head, c.vocab, h, &m)
                })
                .collect();
            Run { logits, state }
        }
    }
}

use reference::Reference;

/// A fixture, the prompt run through it and the cache it is traced at.
struct Case {
    fixture: Fixture,
    tokens: Vec<i32>,
    cap: usize,
}

impl Case {
    fn new(fixture: Fixture) -> Self {
        Self {
            fixture,
            tokens: TOKENS.to_vec(),
            cap: CAP,
        }
    }

    fn model(&self) -> Box<dyn Model> {
        built(&self.fixture)
    }

    fn vocab(&self) -> usize {
        self.fixture.config["text_config"]["vocab_size"]
            .as_u64()
            .unwrap() as usize
    }

    /// One step of `tokens` from `start`: a decode for one token, a prefill for more.
    fn step(
        &self,
        model: &dyn Model,
        tokens: &[i32],
        start: usize,
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let phase = if tokens.len() == 1 {
            Phase::Decode
        } else {
            Phase::Prefill
        };
        run_step(
            model,
            &self.fixture.store,
            phase,
            self.cap,
            tokens,
            start as i32,
            state,
        )
    }

    /// `plan`'s chunk lengths run in order over the prompt; every position's logits and the final
    /// carried state.
    fn run(&self, model: &dyn Model, plan: &[usize]) -> (Vec<f32>, Vec<Vec<f32>>) {
        assert_eq!(plan.iter().sum::<usize>(), self.tokens.len());
        let (mut logits, mut state, mut at) = (Vec::new(), Vec::new(), 0);
        for &len in plan {
            let (out, next) = self.step(model, &self.tokens[at..at + len], at, &state);
            logits.extend(out);
            state = next;
            at += len;
        }
        (logits, state)
    }

    /// The names of the state a whole-prompt prefill carries, in the graph's order.
    fn state_names(&self, model: &dyn Model) -> Vec<String> {
        let shape = StepShape {
            rows: NonZeroUsize::MIN,
            tokens: NonZeroUsize::new(self.tokens.len()).unwrap(),
            capacity: NonZeroUsize::new(self.cap).unwrap(),
            kv: KvLayout::Contiguous,
            logits: LogitRows::All,
            hidden: None,
        };
        let g = model
            .trace(
                Phase::Prefill,
                shape,
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        g.state()
            .map(|(input, _)| g.meta(input).name.clone().expect("state is named"))
            .collect()
    }
}

fn built(fixture: &Fixture) -> Box<dyn Model> {
    let registry = Registry::builtin().expect("the builtin registry");
    registry
        .build(&fixture.raw(), &fixture.store)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Tier 2 (ADR-0101): `got` within `1e-4 * (1 + |want|)` of `want` at every element, NaN failing; the
/// message names the element.
fn assert_close(label: &str, got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: element count");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let g = f64::from(g);
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "{label}: element {i}: got {g}, reference {w}"
        );
    }
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x.into()).collect()
}

/// Every logit against the reference's rows from `from`, naming the position and the token.
fn assert_logits(label: &str, vocab: usize, got: &[f32], want: &[Vec<f64>], from: usize) {
    let want: Vec<f64> = want[from..].concat();
    assert_eq!(got.len(), want.len(), "{label}: logits count");
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        let g = f64::from(g);
        assert!(
            (g - w).abs() <= 1e-4 * (1.0 + w.abs()),
            "{label}: position {} logit {}: got {g}, reference {w}",
            from + i / vocab,
            i % vocab
        );
    }
}

/// Every carried element against the reference's, by state name.
fn assert_state(
    label: &str,
    names: &[String],
    got: &[Vec<f32>],
    want: &BTreeMap<String, Vec<f64>>,
) {
    assert_eq!(
        names.iter().collect::<std::collections::BTreeSet<_>>(),
        want.keys().collect(),
        "{label}: the state the step carries"
    );
    assert_eq!(names.len(), got.len(), "{label}: state count");
    for (name, got) in names.iter().zip(got) {
        assert_close(&format!("{label}: state {name}"), got, &want[name]);
    }
}

/// SC-001 and SC-010: the fixture (non-identical experts, so a routing error is observable) compiles
/// and matches the independent reference at every position and every carried element: one prefill
/// of the whole prompt, and the prompt as a prefill followed by decode steps, whichever way its
/// experts are stored. Mutation: swap two experts' routing indices in the tracer (the router's
/// selected ids read through `[1, 0, ..]`); the first failing element is a logit of position 0.
fn matches_the_reference(case: &Case, label: &str) {
    let model = case.model();
    let reference = Reference::new(&case.fixture);
    let want = reference.forward(&case.tokens, case.cap);
    let vocab = case.vocab();
    let names = case.state_names(&*model);

    let (logits, state) = case.run(&*model, &[case.tokens.len()]);
    assert_logits(&format!("{label} prefill"), vocab, &logits, &want.logits, 0);
    assert_state(&format!("{label} prefill"), &names, &state, &want.state);

    // A prefill of half the prompt, then a decode step per remaining token.
    let half = case.tokens.len() / 2;
    let plan: Vec<usize> = std::iter::once(half)
        .chain(std::iter::repeat_n(1, case.tokens.len() - half))
        .collect();
    let (logits, state) = case.run(&*model, &plan);
    assert_logits(&format!("{label} chain"), vocab, &logits, &want.logits, 0);
    assert_state(&format!("{label} chain"), &names, &state, &want.state);
}

#[test]
fn bf16_experts_match_the_independent_reference() {
    matches_the_reference(&Case::new(super::fixture()), "bf16");
}

#[test]
fn e4m3_packed_experts_match_the_independent_reference() {
    matches_the_reference(&Case::new(super::packed_fixture()), "e4m3");
}

/// SC-002: a config the old pinned counts rejected (two layers, three experts, one expert per
/// token, three streams, blocks of four) and no manifest loads through the registry, traces, and
/// matches its reference at every position and element. Mutation: reinstate a pinned-count check
/// in `Qwen4ExpParams::from_raw` (refuse a layer count other than three); the build is refused
/// and the row goes red at `built`.
#[test]
fn a_finetune_like_config_loads_traces_and_matches_its_reference() {
    matches_the_reference(&Case::new(super::finetune_fixture()), "finetune");
}

/// SC-003: a store missing any one tensor of the checkpoint fails the build with the typed
/// missing-weight error naming that tensor, before anything compiles. Run for every tensor of the
/// fixture with fused experts and of the one with packed experts. Mutation: make `put` skip a
/// tensor the store lacks; the build then fails later with an unmapped-weight error that names a
/// weight id, not the tensor, and the row goes red.
#[test]
fn a_missing_tensor_is_named_before_any_trace() {
    use crate::model::ModelError;
    use crate::registry::LoadModelError;
    use poot_quant::weights::WeightMapError;

    let registry = Registry::builtin().unwrap();
    for (label, fixture) in [
        ("fused", super::fixture()),
        ("packed", super::packed_fixture()),
    ] {
        let keys: Vec<String> = fixture
            .store
            .keys()
            .map(|k| k.as_str().to_string())
            .collect();
        assert!(
            keys.len() > 60,
            "{label}: a real checkpoint's worth of tensors"
        );
        for missing in &keys {
            let mut store = poot_quant::weights::WeightStore::builder();
            for (key, entry) in fixture.store.iter() {
                if key.as_str() != missing {
                    store.insert(key.clone(), entry.clone()).unwrap();
                }
            }
            let error = registry
                .build(&fixture.raw(), &store.build())
                .expect_err(&format!("{label}: {missing} is required"));
            let LoadModelError::Model(ModelError::Weight {
                source: WeightMapError::Missing { key, .. },
                ..
            }) = &error
            else {
                panic!("{label}: without {missing}: {error:?}");
            };
            // A packed tensor is stored under its bare linear id and a dense one under `.weight`.
            let named = |k: &str| k.strip_suffix(".weight").unwrap_or(k).to_string();
            assert_eq!(
                named(key.as_str()),
                named(missing),
                "{label}: the error names the missing tensor"
            );
        }
    }
}

/// SC-005: the prompt prefilled whole, prefilled in chunks, and prefilled then decoded token by
/// token carry the same state and produce the same logits at every position (tier 2). The chunks
/// are multiples of the fixture's block of two, the model's prefill granule. Mutation: ignore the
/// carried state matrices in a chunk that continues a prompt (`components::gdn` reads zeros for
/// its state input); every plan with a second chunk goes red at the first logit of that chunk.
#[test]
fn chunked_prefill_and_decode_continue_to_the_same_logits_and_state() {
    let case = Case::new(super::fixture());
    let model = case.model();
    let (whole, whole_state) = case.run(&*model, &[8]);
    for plan in [
        vec![4, 4],
        vec![2, 2, 4],
        vec![6, 2],
        vec![4, 1, 1, 1, 1],
        vec![2, 1, 1, 1, 1, 1, 1],
    ] {
        let (logits, state) = case.run(&*model, &plan);
        assert_close(&format!("{plan:?} logits"), &logits, &widen(&whole));
        for (i, (got, want)) in state.iter().zip(&whole_state).enumerate() {
            assert_close(&format!("{plan:?} state {i}"), got, &widen(want));
        }
    }
}

/// SC-008 (the model's half): a prefill off the granule, and a cache that is not a whole number of
/// blocks, are typed `ShapeUnsupported { Granule }` refusals from `Model::trace`, which builds no
/// graph (so nothing can dispatch); a decode is never off the granule. Mutation: drop the granule
/// check from `Qwen4Exp::trace`; the odd prefill traces and the refusal assertions go red.
#[test]
fn a_shape_off_the_granule_is_a_typed_refusal() {
    use crate::model::{ShapeReason, TraceError};

    let model = built(&super::fixture());
    assert_eq!(model.config().prefill_granule.get(), 2);
    let shape = |tokens: usize, capacity: usize| StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(tokens).unwrap(),
        capacity: NonZeroUsize::new(capacity).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::All,
        hidden: None,
    };
    for (phase, tokens, capacity) in [
        (Phase::Prefill, 3, 8),
        (Phase::Prefill, 1, 8),
        (Phase::Prefill, 4, 7),
    ] {
        let refusal = model
            .trace(
                phase,
                shape(tokens, capacity),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .expect_err("an off-granule shape is refused");
        assert!(
            matches!(
                refusal,
                TraceError::ShapeUnsupported {
                    reason: ShapeReason::Granule { granule: 2 },
                    ..
                }
            ),
            "{tokens} tokens over {capacity}: {refusal:?}"
        );
    }
    assert!(
        model
            .trace(
                Phase::Prefill,
                shape(4, 8),
                poot_graph_ir::TraceLimits::for_tests()
            )
            .is_ok()
    );
    assert!(
        model
            .trace(
                Phase::Decode,
                shape(1, 8),
                poot_graph_ir::TraceLimits::for_tests()
            )
            .is_ok()
    );
}

/// SC-1143-5: a paged-KV step is a typed refusal naming the indexer, and builds no graph: the block-key
/// cache is per row and its paging granule is POOT-1148's. Mutation: drop the paged check from
/// `Qwen4Exp::trace_into`; the paged shape traces and the refusal assertion goes red.
#[test]
fn a_paged_step_is_a_typed_indexer_refusal() {
    use crate::model::{ShapeReason, TraceError};

    let model = built(&super::fixture());
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(2).unwrap(),
        capacity: NonZeroUsize::new(8).unwrap(),
        kv: KvLayout::Paged {
            pool_slots: NonZeroUsize::new(16).unwrap(),
            seqs: NonZeroUsize::MIN,
        },
        logits: LogitRows::All,
        hidden: None,
    };
    let refusal = model
        .trace(
            Phase::Prefill,
            shape,
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect_err("a paged step is refused");
    assert!(
        matches!(
            refusal,
            TraceError::ShapeUnsupported {
                reason: ShapeReason::PagedIndexer,
                ..
            }
        ),
        "{refusal:?}"
    );
}

/// SC-1143-6: the indexer's block keys are declared [`StateRole::Blocked`] at the block ratio and its
/// unfinished-block raw keys `Recurrent`; the K/V caches stay `Positional`. Mutation: declare the block
/// keys `Positional { axis: 2 }`; the traffic accounting would read each entry as one position.
#[test]
fn the_block_key_cache_is_declared_blocked() {
    use poot_graph_ir::StateRole;

    let case = Case::new(super::fixture());
    let model = case.model();
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(2).unwrap(),
        capacity: NonZeroUsize::new(case.cap).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::All,
        hidden: None,
    };
    let g = model
        .trace(
            Phase::Prefill,
            shape,
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let role = |name: &str| {
        g.state_pairs()
            .find(|pair| g.meta(pair.input).name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no state {name}"))
            .role
    };
    assert_eq!(
        role("kv.l2.index_blocks"),
        StateRole::Blocked {
            axis: 2,
            ratio: std::num::NonZeroU32::new(2).unwrap()
        }
    );
    assert_eq!(role("kv.l2.index_tail"), StateRole::Recurrent);
    assert_eq!(role("kv.l2.k"), StateRole::Positional { axis: 2 });
}

/// SC-006 (the model's half): the PLE reads the token history through the carried state. The same
/// continuation chunk after three different prefixes (a fresh-looking one, a nonzero one, one with
/// the end-of-sequence id in the window) matches the reference after each, and the three differ;
/// each PLE weight matters. Mutation: index the table with the previous step's ids (the history
/// state read as this step's ids); the first prefix's continuation goes red at its first logit.
#[test]
fn the_ple_layer_follows_the_token_history() {
    let fixture = super::fixture();
    let reference = Reference::new(&fixture);
    let model = built(&fixture);
    let continuation = [9, 31];
    let mut produced = Vec::new();
    for prefix in [[3, 17], [8, 40], [22, 47]] {
        let tokens: Vec<i32> = prefix.iter().chain(&continuation).copied().collect();
        let case = Case {
            fixture: fixture.clone(),
            tokens: tokens.clone(),
            cap: CAP,
        };
        let (logits, _) = case.run(&*model, &[2, 2]);
        let want = reference.forward(&tokens, CAP);
        assert_logits(&format!("prefix {prefix:?}"), 48, &logits, &want.logits, 0);
        produced.push(logits[2 * 48..].to_vec());
    }
    assert_ne!(produced[0], produced[1]);
    assert_ne!(produced[1], produced[2]);
    assert_ne!(produced[0], produced[2]);
    // Every PLE tensor changes the logits (an unread weight would be a mapped-but-dead one).
    let case = Case::new(fixture.clone());
    let base = case.run(&*model, &[8]).0;
    for suffix in [
        "ple.ple_embedding.ngram_embedding.weight",
        "ple.key_proj.weight",
        "ple.value_proj.weight",
        "ple.norm_key.weight",
        "ple.norm_query.weight",
        "ple.norm_conv.weight",
        "ple.conv1d.weight",
    ] {
        let key = format!("model.language_model.layers.1.{suffix}");
        let mut changed = fixture.clone();
        changed.store = replaced(&fixture.store, &key, 0.75);
        let model = built(&changed);
        let case = Case {
            fixture: changed,
            tokens: TOKENS.to_vec(),
            cap: CAP,
        };
        let logits = case.run(&*model, &[8]).0;
        assert!(
            logits.iter().zip(&base).any(|(a, b)| (a - b).abs() > 1e-6),
            "the logits do not depend on {key}"
        );
    }
}

/// `store` with every element of `key` (BF16) shifted by `by`.
fn replaced(store: &WeightStore, key: &str, by: f32) -> WeightStore {
    use poot_quant::weights::{DenseWeight, WeightEntry};
    let mut out = WeightStore::builder();
    for (k, entry) in store.iter() {
        let entry = match entry {
            WeightEntry::Dense(dense) if k.as_str() == key => {
                let bytes: Vec<u8> = dense
                    .bytes()
                    .as_slice()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|&pair| {
                        let v = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16) + by;
                        ((v.to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect();
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into())
                        .unwrap(),
                )
            }
            other => other.clone(),
        };
        out.insert(k.clone(), entry).unwrap();
    }
    out.build()
}

/// SC-009 (the model's half): the step's graph declares the PLE's token-range witness on the PLE's
/// layer, so a token outside the vocabulary reaches the guard as a typed validation fault, not as
/// garbage logits (the witness's value is the component's row, and the driver's device row reads the
/// fault; the CPU oracle's own embedding gather refuses such a token first). Mutation: discard the
/// guard witness in `components::ple::ngram_ids`; the graph declares no validation and the row goes
/// red.
#[test]
fn an_out_of_vocabulary_token_is_a_typed_validation_fault() {
    use crate::components::ple::PleWitness;

    let fixture = super::fixture();
    let model = built(&fixture);
    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(2).unwrap(),
        capacity: NonZeroUsize::new(CAP).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::All,
        hidden: None,
    };
    let g = model
        .trace(
            Phase::Prefill,
            shape,
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let witnesses: Vec<usize> = g
        .witnesses()
        .filter_map(|v| PleWitness::decode(v.id))
        .collect();
    assert_eq!(witnesses, vec![1], "one PLE witness, on layer 1");
}

/// SC-010 (the graph half): the BF16 fixture's experts trace as the stored BF16 stacks (no
/// packed dequantization, no F32 stack), and the packed one's as packed weights: the traced graph
/// holds `PackedDequant` and declares no F32 expert const (an expert is only reachable through its
/// packed sources). Mutation: make `components::moe` ignore the weight handle's descriptor (read
/// every packed expert as a dense F32 weight); the packed graph declares F32 expert consts and holds
/// no `PackedDequant`, and the row goes red.
#[test]
fn packed_experts_trace_as_packed_weights_and_dense_ones_as_stored_stacks() {
    use poot_graph_ir::{OpKind, Storage};
    use poot_tensor::DType;

    let shape = StepShape {
        rows: NonZeroUsize::MIN,
        tokens: NonZeroUsize::new(4).unwrap(),
        capacity: NonZeroUsize::new(CAP).unwrap(),
        kv: KvLayout::Contiguous,
        logits: LogitRows::All,
        hidden: None,
    };
    let analyze = |fixture: &Fixture| {
        let model = built(fixture);
        let g = model
            .trace(
                Phase::Prefill,
                shape,
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let formats = poot_graph_plan::WeightFormats::from_weight_map(model.weights());
        let bound = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
        let dequants = bound
            .eqns()
            .iter()
            .filter(|e| {
                matches!(
                    e.op,
                    OpKind::PackedDequant { .. } | OpKind::PackedContraction { .. }
                )
            })
            .count();
        let expert_consts: Vec<(String, DType)> = bound
            .inputs()
            .iter()
            .filter_map(|&id| {
                let meta = bound.meta(id);
                let name = meta.name.clone()?;
                let expert = name.contains(".moe.expert") || name.contains(".moe.stack");
                (meta.storage == Storage::Const && expert).then_some((name, meta.aval.dtype))
            })
            .collect();
        (dequants, expert_consts)
    };
    let (dequants, consts) = analyze(&super::fixture());
    assert_eq!(dequants, 0, "no packed weight in the BF16 fixture");
    assert!(!consts.is_empty());
    assert!(
        consts.iter().all(|(_, dtype)| *dtype == DType::BF16),
        "the dense experts stay BF16 as stored: {consts:?}"
    );
    let (dequants, consts) = analyze(&super::packed_fixture());
    assert!(dequants > 0, "the packed experts decode in the graph");
    assert!(
        consts.iter().all(|(_, dtype)| *dtype != DType::F32),
        "no packed expert is declared as an F32 weight: {consts:?}"
    );
    assert!(!consts.is_empty());
}

/// A tie keeps exactly the budget: with the indexer's projection zeroed every block scores 0, so the
/// budget (two blocks) is filled by the lowest two eligible blocks, not by every tied block. The
/// reference sorts stably; the model ranks by the pairwise construction. Mutation: rank without the
/// stable tie-break (a shared rank admits every block tied at the cut); a query with three
/// complete blocks sees all of them, and the logits of positions 5 and after go red.
#[test]
fn tied_indexer_scores_keep_exactly_the_budget() {
    let mut fixture = super::fixture();
    let key = "model.language_model.layers.2.self_attn.indexer.index_qk_proj.weight";
    let mut store = poot_quant::weights::WeightStore::builder();
    for (k, entry) in fixture.store.iter() {
        let entry = match entry {
            poot_quant::weights::WeightEntry::Dense(dense) if k.as_str() == key => {
                poot_quant::weights::WeightEntry::Dense(
                    poot_quant::weights::DenseWeight::try_new(
                        dense.dtype(),
                        dense.shape().to_vec(),
                        vec![0u8; dense.bytes().len()].into(),
                    )
                    .unwrap(),
                )
            }
            other => other.clone(),
        };
        store.insert(k.clone(), entry).unwrap();
    }
    fixture.store = store.build();
    matches_the_reference(&Case::new(fixture), "tied indexer");
}

/// A config the family cannot configure is a typed error naming the field, never a panic.
#[test]
fn from_raw_refuses_what_the_family_cannot_configure() {
    let refuse = |edit: &dyn Fn(&mut serde_json::Value)| -> ModelError {
        let mut config = super::fixture_config();
        edit(&mut config);
        Qwen4ExpParams::from_raw(&RawConfig::HfJson {
            config: &config,
            generation: None,
        })
        .expect_err("the config is refused")
    };
    let field = |e: ModelError| match e {
        ModelError::Config { field, reason, .. } => (field, reason),
        other => panic!("{other:?}"),
    };
    use crate::model::ConfigReason;
    assert_eq!(
        field(refuse(&|c| {
            c.as_object_mut().unwrap().remove("text_config");
        })),
        ("text_config", ConfigReason::Missing)
    );
    assert_eq!(
        field(refuse(&|c| c["text_config"]["indexer_kv_heads"] = 2.into())),
        ("indexer_kv_heads", ConfigReason::Unsupported)
    );
    // The PLE sits on a linear-attention layer: id 3 names the full-attention one.
    assert_eq!(
        field(refuse(
            &|c| c["text_config"]["ple_layer_ids"] = serde_json::json!([3])
        )),
        ("ple_layer_ids", ConfigReason::Unsupported)
    );
    assert_eq!(
        field(refuse(
            &|c| c["text_config"]["ple_layer_ids"] = serde_json::json!([4])
        )),
        ("ple_layer_ids", ConfigReason::Exceeds { max: 3 })
    );
    assert_eq!(
        field(refuse(
            &|c| c["text_config"]["layer_types"] = serde_json::json!(["linear_attention"])
        )),
        ("layer_types", ConfigReason::Length { expected: 3 })
    );
    assert_eq!(
        field(refuse(&|c| c["text_config"]["indexer_budget"] = 1.into())),
        ("indexer_budget", ConfigReason::Unsupported)
    );
    // The indexer finds a position's block with a shift and a mask.
    assert_eq!(
        field(refuse(
            &|c| c["text_config"]["indexer_compress_ratio"] = 3.into()
        )),
        ("indexer_compress_ratio", ConfigReason::Unsupported)
    );
    assert_eq!(
        field(refuse(&|c| c["text_config"]["hc_count"] = 0.into())),
        ("hc_count", ConfigReason::Zero)
    );
    assert_eq!(
        field(refuse(
            &|c| c["text_config"]["linear_value_head_dim"] = 4.into()
        )),
        ("linear_value_head_dim", ConfigReason::Unsupported)
    );
}

/// The prompt every driver row feeds the families, and the greedy tokens the independent reference
/// generates after it (re-running the whole sequence for every token: no carried state at all). The
/// CPU oracle's token-by-token run (every token a decode step: no prefill, no chunk, no granule)
/// equals them, and the device rows of `poot-llm` generate the same tokens through the driver, the
/// prompt as whole-granule prefill plus decode steps: SC-008's row on the real family.
const PROMPT: [i32; 7] = [3, 17, 40, 8, 25, 11, 2];
const GREEDY: [u32; 24] = [
    38, 45, 2, 44, 16, 27, 35, 25, 34, 10, 28, 16, 14, 10, 15, 35, 36, 3, 38, 30, 4, 39, 18, 21,
];
const GREEDY_PACKED: [u32; 24] = [
    22, 22, 46, 11, 6, 18, 7, 22, 26, 28, 45, 9, 11, 13, 30, 33, 28, 30, 12, 31, 11, 5, 11, 11,
];
const GREEDY_FINETUNE: [u32; 24] = [
    12, 6, 34, 37, 30, 7, 38, 40, 18, 38, 7, 40, 7, 6, 34, 21, 38, 46, 27, 7, 43, 31, 28, 28,
];

/// The greedy argmax of `logits` (one position).
fn argmax<T: Copy + PartialOrd + std::fmt::Debug>(logits: &[T]) -> u32 {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        assert!(v.partial_cmp(&v).is_some(), "logit {i} is {v:?}");
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// The CPU oracle's greedy tokens, the prompt and every new token one decode step each.
fn model_greedy(fixture: &Fixture) -> Vec<u32> {
    let case = Case {
        fixture: fixture.clone(),
        tokens: Vec::new(),
        cap: 32,
    };
    let model = case.model();
    let vocab = case.vocab();
    let mut state = Vec::new();
    let mut next = 0;
    for (position, &token) in PROMPT.iter().enumerate() {
        let (logits, carried) = case.step(&*model, &[token], position, &state);
        state = carried;
        next = argmax(&logits[..vocab]);
    }
    let mut tokens = vec![next];
    for position in PROMPT.len()..PROMPT.len() + 23 {
        let (logits, carried) = case.step(&*model, &[next as i32], position, &state);
        state = carried;
        next = argmax(&logits[..vocab]);
        tokens.push(next);
    }
    tokens
}

/// The independent reference's greedy tokens: each token is the argmax of the last position of the
/// whole sequence so far.
fn reference_greedy(fixture: &Fixture) -> Vec<u32> {
    let reference = Reference::new(fixture);
    let mut sequence = PROMPT.to_vec();
    let mut tokens = Vec::new();
    for _ in 0..24 {
        let run = reference.forward(&sequence, 32);
        let next = argmax(run.logits.last().unwrap());
        tokens.push(next);
        sequence.push(next as i32);
    }
    tokens
}

#[test]
fn the_fixtures_generate_the_recorded_tokens() {
    for (label, fixture, recorded) in [
        ("dense", super::fixture(), GREEDY),
        ("packed", super::packed_fixture(), GREEDY_PACKED),
        ("finetune", super::finetune_fixture(), GREEDY_FINETUNE),
    ] {
        assert_eq!(
            reference_greedy(&fixture),
            recorded,
            "{label}: the reference"
        );
        assert_eq!(
            model_greedy(&fixture),
            recorded,
            "{label}: the model, token by token"
        );
        assert!(
            recorded.iter().any(|&t| t != recorded[0]),
            "{label}: the recorded run is not constant"
        );
    }
}

/// Three prompts of six tokens with different histories for the PLE: a plain one, one with an
/// end-of-sequence id inside the window, and one that opens with two. The driver rows run each of
/// them through the one binder and expect the reference's first-step top five (a token and its
/// log-probability), which these literals are (the tracer cannot move them: they come from the
/// reference).
const HISTORIES: [[i32; 6]; 3] = [
    [3, 17, 40, 8, 25, 11],
    [8, 40, 5, 22, 47, 9],
    [47, 47, 3, 9, 22, 31],
];
const HISTORIES_TOP5: [[(u32, f64); 5]; 3] = [
    [
        (25, -1.903_203_6),
        (33, -2.256_853_4),
        (15, -3.129_722),
        (21, -3.237_417),
        (1, -3.352_075_7),
    ],
    [
        (35, -2.689_749_4),
        (25, -2.868_450_9),
        (30, -2.914_160_6),
        (26, -2.922_117_3),
        (2, -3.010_46),
    ],
    [
        (32, -2.643_691_5),
        (21, -2.647_688_7),
        (38, -2.836_676_7),
        (23, -2.846_178_8),
        (29, -2.876_545),
    ],
];

/// The five most likely tokens of a position's logits and their log-probabilities.
fn top5(logits: &[f64]) -> [(u32, f64); 5] {
    let max = logits.iter().cloned().fold(f64::MIN, f64::max);
    let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln() + max;
    let mut ranked: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(t, l)| (t as u32, l - log_sum))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked[..5].try_into().unwrap()
}

#[test]
fn the_ple_history_literals_are_the_references_first_step() {
    let fixture = super::fixture();
    let reference = Reference::new(&fixture);
    let model = built(&fixture);
    for (prompt, want) in HISTORIES.iter().zip(HISTORIES_TOP5) {
        let from_reference = top5(reference.forward(prompt, 32).logits.last().unwrap());
        let case = Case {
            fixture: fixture.clone(),
            tokens: prompt.to_vec(),
            cap: 32,
        };
        let (logits, _) = case.run(&*model, &[6]);
        let from_model = top5(&widen(&logits[5 * 48..]));
        for (label, got) in [("reference", from_reference), ("model", from_model)] {
            for ((token, logprob), (want_token, want_logprob)) in got.iter().zip(want) {
                assert_eq!(
                    *token, want_token,
                    "{label}: {prompt:?}: the five most likely tokens"
                );
                assert!(
                    (logprob - want_logprob).abs() <= 1e-3,
                    "{label}: {prompt:?}: token {token} has log-probability {logprob}, the literal is {want_logprob}"
                );
            }
        }
    }
    // Non-degenerate: the three histories rank different tokens first.
    let firsts: std::collections::BTreeSet<u32> = HISTORIES_TOP5.iter().map(|t| t[0].0).collect();
    assert!(firsts.len() > 1, "{HISTORIES_TOP5:?}");
}

/// A checkpoint that stores the n-gram table as numbered shards under a scale (the official
/// release's layout) builds a model whose logits equal the single-table checkpoint's, bit for bit.
/// Mutation: map the shards in reverse ordinal order in `weights_of`; the shard rows swap and the
/// logits differ.
#[test]
fn a_checkpoint_with_sharded_ple_tables_matches_the_single_table() {
    use poot_quant::weights::{DenseWeight, WeightEntry};

    let fixture = super::fixture();
    let table = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding";
    let single = format!("{table}.weight");
    let mut store = WeightStore::builder();
    for (key, entry) in fixture.store.iter() {
        if key.as_str() != single {
            store.insert(key.clone(), entry.clone()).unwrap();
            continue;
        }
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture's table is dense");
        };
        // Three shards of different sizes, the scale one.
        let (rows, width) = (dense.shape()[0], dense.shape()[1]);
        let bytes = dense.bytes().as_slice();
        let row_bytes = bytes.len() / rows;
        let mut at = 0;
        for (ordinal, count) in [rows / 4, rows / 2, rows - rows / 4 - rows / 2]
            .into_iter()
            .enumerate()
        {
            let cut = bytes[at * row_bytes..(at + count) * row_bytes].to_vec();
            let shard =
                DenseWeight::try_new(dense.dtype(), vec![count, width], cut.into()).unwrap();
            store
                .insert(
                    format!("{table}.shard_{ordinal}.weight"),
                    WeightEntry::Dense(shard),
                )
                .unwrap();
            at += count;
        }
        let one = DenseWeight::try_new(
            poot_tensor::DType::F32,
            vec![],
            1.0f32.to_le_bytes().to_vec().into(),
        )
        .unwrap();
        store
            .insert(format!("{table}.weight_scale"), WeightEntry::Dense(one))
            .unwrap();
    }
    let store = store.build();
    let registry = Registry::builtin().unwrap();
    let sharded = registry.build(&fixture.raw(), &store).unwrap();
    let single = built(&fixture);
    let step = |model: &dyn Model, store: &WeightStore| {
        run_step(model, store, Phase::Prefill, CAP, &TOKENS, 0, &[])
    };
    let (want, _) = step(single.as_ref(), &fixture.store);
    let (got, _) = step(sharded.as_ref(), &store);
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

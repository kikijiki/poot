//! The independent reference of DeepSeek-V4: the model written token by token from the HF
//! `deepseek_v4` formulas in f64 over the checkpoint's own tensors, calling no component and no
//! tracer. Position `p` reads the histories positions `0..p` wrote, so a prefill chunk, a decode
//! step and a continuation are all compared with the same function of the prefix.
//!
//! Written from `transformers` `models/deepseek_v4/modeling_deepseek_v4.py` as the card's spec 281
//! records it: low-rank query, one shared key-value head, interleaved partial RoPE on the trailing
//! `qk_rope_head_dim` dims (inverse on the output), attention sinks, a grouped output projection,
//! the three attention kinds, mHC around both blocks, and the routed feed-forward with the
//! `sqrt(softplus)` score.

use std::collections::HashMap;

use poot_quant::weights::{WeightEntry, WeightStore};

use crate::components::testing::values;

/// The dimensions the reference reads from the fixture's `config.json`.
#[derive(Clone, Debug)]
pub(super) struct Dims {
    pub vocab: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub groups: usize,
    pub o_lora: usize,
    pub window: usize,
    pub hc: usize,
    pub sinkhorn: usize,
    pub hc_eps: f64,
    pub eps: f64,
    pub theta: f64,
    pub compress_theta: f64,
    pub csa_rate: usize,
    pub hca_rate: usize,
    pub ratios: Vec<usize>,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub experts: usize,
    pub top_k: usize,
    pub inter: usize,
    pub route_scale: f64,
    pub limit: f64,
}

impl Dims {
    pub(super) fn of(config: &serde_json::Value) -> Self {
        let n = |k: &str| crate::registry::Fixture::count(config, k);
        let f = |k: &str| config[k].as_f64().unwrap();
        let rate = |k: &str, default: usize| {
            config["compress_rates"][k]
                .as_u64()
                .map_or(default, |r| r as usize)
        };
        let ratios: Vec<usize> = config["compress_ratios"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_u64().unwrap() as usize)
            .collect();
        let layers = n("num_hidden_layers");
        let has_csa = ratios[..layers]
            .iter()
            .any(|&r| r == rate("compressed_sparse_attention", 4));
        Self {
            vocab: n("vocab_size"),
            width: n("hidden_size"),
            layers,
            heads: n("num_attention_heads"),
            head_dim: n("head_dim"),
            rope_dim: n("qk_rope_head_dim"),
            q_lora: n("q_lora_rank"),
            groups: n("o_groups"),
            o_lora: n("o_lora_rank"),
            window: n("sliding_window"),
            hc: n("hc_mult"),
            sinkhorn: n("hc_sinkhorn_iters"),
            hc_eps: f("hc_eps"),
            eps: f("rms_norm_eps"),
            theta: f("rope_theta"),
            compress_theta: f("compress_rope_theta"),
            csa_rate: rate("compressed_sparse_attention", 4),
            hca_rate: rate("heavily_compressed_attention", 128),
            ratios,
            index_heads: if has_csa { n("index_n_heads") } else { 0 },
            index_dim: if has_csa { n("index_head_dim") } else { 0 },
            index_topk: if has_csa { n("index_topk") } else { 0 },
            experts: n("n_routed_experts"),
            top_k: n("num_experts_per_tok"),
            inter: n("moe_intermediate_size"),
            route_scale: f("routed_scaling_factor"),
            limit: f("swiglu_limit"),
        }
    }

    fn kind(&self, layer: usize) -> Kind {
        match self.ratios[layer] {
            0 => Kind::Sliding,
            r if r == self.csa_rate => Kind::Csa,
            _ => Kind::Hca,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Sliding,
    Csa,
    Hca,
}

/// What a run returns: the logits of every position and the carried state at the end, by the name
/// the traced graph gives it, laid out `[capacity, width]` with zeros past the last position.
pub(super) struct Run {
    pub logits: Vec<Vec<f64>>,
    pub state: Vec<(String, Vec<f64>)>,
}

pub(super) struct Reference {
    d: Dims,
    w: HashMap<String, Vec<f64>>,
    /// Per hash layer, the `[vocab][top_k]` table.
    tables: HashMap<usize, Vec<Vec<usize>>>,
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

/// Rotate (or with `inverse`, un-rotate) the trailing `rot` dims of `v` as adjacent pairs at
/// position `pos`: pair `i` turns by `pos * theta^(-2i/rot)`.
fn rotate_tail(v: &mut [f64], rot: usize, pos: usize, theta: f64, inverse: bool) {
    let start = v.len() - rot;
    for i in 0..rot / 2 {
        let angle = pos as f64 * theta.powf(-((2 * i) as f64) / rot as f64);
        let (s, c) = angle.sin_cos();
        let s = if inverse { -s } else { s };
        let (a, b) = (v[start + 2 * i], v[start + 2 * i + 1]);
        v[start + 2 * i] = a * c - b * s;
        v[start + 2 * i + 1] = a * s + b * c;
    }
}

/// The histories one layer carries, indexed by absolute position.
#[derive(Default, Clone)]
struct LayerHistory {
    kv: Vec<Vec<f64>>,
    compress_kv: Vec<Vec<f64>>,
    compress_gate: Vec<Vec<f64>>,
    indexer_kv: Vec<Vec<f64>>,
    indexer_gate: Vec<Vec<f64>>,
}

impl Reference {
    pub(super) fn new(config: &serde_json::Value, store: &WeightStore) -> Self {
        let d = Dims::of(config);
        let mut w = HashMap::new();
        let mut tables = HashMap::new();
        for (key, entry) in store.iter() {
            let WeightEntry::Dense(dense) = entry else {
                w.insert(
                    key.as_str().to_string(),
                    values(entry).into_iter().map(f64::from).collect(),
                );
                continue;
            };
            if key.as_str().ends_with("tid2eid") {
                let layer: usize = key.as_str().split('.').nth(1).unwrap().parse().unwrap();
                let k = dense.shape()[1];
                let words: Vec<usize> = dense
                    .bytes()
                    .as_slice()
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c) as usize)
                    .collect();
                tables.insert(layer, words.chunks(k).map(<[usize]>::to_vec).collect());
                continue;
            }
            w.insert(
                key.as_str().to_string(),
                values(entry).into_iter().map(f64::from).collect(),
            );
        }
        Self { d, w, tables }
    }

    fn t(&self, name: &str) -> &[f64] {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("the fixture has no tensor {name}"))
    }

    fn theta(&self, kind: Kind) -> f64 {
        match kind {
            Kind::Sliding => self.d.theta,
            _ => self.d.compress_theta,
        }
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

    /// The compressed entries of window `w` of a compressor: `rate` positions of `kv`/`gate`
    /// (raw histories), pooled disjointly or, for `overlapping`, with the previous window's first
    /// half; normed and rotated at the window's first position.
    #[allow(clippy::too_many_arguments)]
    fn window_entry(
        &self,
        prefix: &str,
        kv: &[Vec<f64>],
        gate: &[Vec<f64>],
        w: usize,
        rate: usize,
        entry: usize,
        overlapping: bool,
        theta: f64,
    ) -> Vec<f64> {
        let d = &self.d;
        let bias = self.t(&format!("{prefix}.ape"));
        let raw = if overlapping { 2 * entry } else { entry };
        // The (key, gate) rows pooled per channel: previous window's first half, then this window's
        // second half (or the whole window for disjoint pooling).
        let mut pooled = vec![0.0; entry];
        for (c, out) in pooled.iter_mut().enumerate() {
            let mut rows: Vec<(f64, f64)> = Vec::new();
            if overlapping {
                for j in 0..rate {
                    rows.push(if w == 0 {
                        (0.0, -1.0e30)
                    } else {
                        let tok = (w - 1) * rate + j;
                        (kv[tok][c], gate[tok][c] + bias[j * raw + c])
                    });
                }
                for j in 0..rate {
                    let tok = w * rate + j;
                    rows.push((
                        kv[tok][entry + c],
                        gate[tok][entry + c] + bias[j * raw + entry + c],
                    ));
                }
            } else {
                for j in 0..rate {
                    let tok = w * rate + j;
                    rows.push((kv[tok][c], gate[tok][c] + bias[j * raw + c]));
                }
            }
            let max = rows.iter().map(|r| r.1).fold(f64::MIN, f64::max);
            let exps: Vec<f64> = rows.iter().map(|r| (r.1 - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            *out = rows.iter().zip(&exps).map(|(r, e)| r.0 * e / sum).sum();
        }
        let mut v = rms_scaled(&pooled, self.t(&format!("{prefix}.norm.weight")), d.eps);
        rotate_tail(&mut v, d.rope_dim, w * rate, theta, false);
        v
    }

    /// The attention block of `layer` for the token at position `p`.
    fn attention(
        &self,
        layer: usize,
        normed: &[f64],
        p: usize,
        hist: &mut LayerHistory,
    ) -> Vec<f64> {
        let d = &self.d;
        let kind = d.kind(layer);
        let theta = self.theta(kind);
        let at = |s: &str| format!("layers.{layer}.{s}");
        let (hq, hd, rd) = (d.heads, d.head_dim, d.rope_dim);

        let q_a = matvec(self.t(&at("attn.wq_a.weight")), d.q_lora, d.width, normed);
        let q_res = rms_scaled(&q_a, self.t(&at("attn.q_norm.weight")), d.eps);
        let q_all = matvec(self.t(&at("attn.wq_b.weight")), hq * hd, d.q_lora, &q_res);
        let queries: Vec<Vec<f64>> = (0..hq)
            .map(|h| {
                let mut q = rms(&q_all[h * hd..(h + 1) * hd], d.eps);
                rotate_tail(&mut q, rd, p, theta, false);
                q
            })
            .collect();

        let kv = matvec(self.t(&at("attn.wkv.weight")), hd, d.width, normed);
        let mut kv = rms_scaled(&kv, self.t(&at("attn.kv_norm.weight")), d.eps);
        rotate_tail(&mut kv, rd, p, theta, false);
        hist.kv.push(kv);

        // The raw histories of the compressors, written before the windows are read.
        if kind != Kind::Sliding {
            let overlapping = kind == Kind::Csa;
            let raw = if overlapping { 2 * hd } else { hd };
            hist.compress_kv.push(matvec(
                self.t(&at("attn.compressor.wkv.weight")),
                raw,
                d.width,
                normed,
            ));
            hist.compress_gate.push(matvec(
                self.t(&at("attn.compressor.wgate.weight")),
                raw,
                d.width,
                normed,
            ));
        }
        if kind == Kind::Csa {
            let raw = 2 * d.index_dim;
            hist.indexer_kv.push(matvec(
                self.t(&at("attn.indexer.compressor.wkv.weight")),
                raw,
                d.width,
                normed,
            ));
            hist.indexer_gate.push(matvec(
                self.t(&at("attn.indexer.compressor.wgate.weight")),
                raw,
                d.width,
                normed,
            ));
        }

        // The keys (and values) the token reads: the window, then the compressed entries.
        let mut keys: Vec<Vec<f64>> = hist.kv[(p + 1).saturating_sub(d.window)..=p].to_vec();
        if kind != Kind::Sliding {
            let rate = if kind == Kind::Csa {
                d.csa_rate
            } else {
                d.hca_rate
            };
            let complete = (p + 1) / rate;
            let selected: Vec<usize> = if kind == Kind::Hca {
                (0..complete).collect()
            } else {
                let ix = d.index_dim;
                let q_idx = matvec(
                    self.t(&at("attn.indexer.wq_b.weight")),
                    d.index_heads * ix,
                    d.q_lora,
                    &q_res,
                );
                let weights = matvec(
                    self.t(&at("attn.indexer.weights_proj.weight")),
                    d.index_heads,
                    d.width,
                    normed,
                );
                let scores: Vec<f64> = (0..complete)
                    .map(|w| {
                        let k = self.window_entry(
                            &at("attn.indexer.compressor"),
                            &hist.indexer_kv,
                            &hist.indexer_gate,
                            w,
                            rate,
                            ix,
                            true,
                            theta,
                        );
                        (0..d.index_heads)
                            .map(|h| {
                                let mut q = q_idx[h * ix..(h + 1) * ix].to_vec();
                                rotate_tail(&mut q, rd, p, theta, false);
                                let dot: f64 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
                                weights[h]
                                    * (d.index_heads as f64).powf(-0.5)
                                    * dot.max(0.0)
                                    * (ix as f64).powf(-0.5)
                            })
                            .sum()
                    })
                    .collect();
                // The selector's contract (`sparse_top_k`): a window is kept when fewer than
                // `index_topk` windows score strictly higher, so windows that tie at the cut (a
                // ReLU score of exactly zero is common) are all kept.
                (0..complete)
                    .filter(|&w| {
                        scores.iter().filter(|&&other| other > scores[w]).count() < d.index_topk
                    })
                    .collect()
            };
            for w in selected {
                keys.push(self.window_entry(
                    &at("attn.compressor"),
                    &hist.compress_kv,
                    &hist.compress_gate,
                    w,
                    rate,
                    hd,
                    kind == Kind::Csa,
                    theta,
                ));
            }
        }

        // Attention with a per-head sink in the softmax denominator, then the inverse rotation.
        let sinks = self.t(&at("attn.attn_sink"));
        let scale = (hd as f64).powf(-0.5);
        let mut heads = Vec::with_capacity(hq * hd);
        for (h, q) in queries.iter().enumerate() {
            let scores: Vec<f64> = keys
                .iter()
                .map(|k| scale * q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            let max = scores.iter().cloned().fold(sinks[h], f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom = (sinks[h] - max).exp() + exps.iter().sum::<f64>();
            let mut out = vec![0.0; hd];
            for (k, e) in keys.iter().zip(&exps) {
                for (o, v) in out.iter_mut().zip(k) {
                    *o += e / denom * v;
                }
            }
            rotate_tail(&mut out, rd, p, theta, true);
            heads.extend(out);
        }

        // The grouped output projection: `groups` contiguous blocks of heads, each its own rows.
        let ipg = hq * hd / d.groups;
        let wo_a = self.t(&at("attn.wo_a.weight"));
        let mut grouped = Vec::with_capacity(d.groups * d.o_lora);
        for g in 0..d.groups {
            let block = &wo_a[g * d.o_lora * ipg..(g + 1) * d.o_lora * ipg];
            grouped.extend(matvec(block, d.o_lora, ipg, &heads[g * ipg..(g + 1) * ipg]));
        }
        matvec(
            self.t(&at("attn.wo_b.weight")),
            d.width,
            d.groups * d.o_lora,
            &grouped,
        )
    }

    /// One layer's routed feed-forward of `x` for token id `token`.
    fn moe(&self, layer: usize, token: usize, x: &[f64]) -> Vec<f64> {
        let d = &self.d;
        let at = |s: &str| format!("layers.{layer}.{s}");
        let logits = matvec(self.t(&at("ffn.gate.weight")), d.experts, d.width, x);
        let score: Vec<f64> = logits.iter().map(|l| (1.0 + l.exp()).ln().sqrt()).collect();
        let ids: Vec<usize> = match self.tables.get(&layer) {
            Some(table) => table[token].clone(),
            None => {
                let bias = self.t(&at("ffn.gate.bias"));
                let mut order: Vec<usize> = (0..d.experts).collect();
                order.sort_by(|&a, &b| {
                    (score[b] + bias[b])
                        .partial_cmp(&(score[a] + bias[a]))
                        .unwrap()
                        .then(a.cmp(&b))
                });
                order[..d.top_k].to_vec()
            }
        };
        let sum: f64 = ids.iter().map(|&e| score[e]).sum();
        let expert = |prefix: &str, scale: f64| -> Vec<f64> {
            let w = |n: &str| self.t(&format!("{prefix}.{n}.weight"));
            let gate = matvec(w("w1"), d.inter, d.width, x);
            let up = matvec(w("w3"), d.inter, d.width, x);
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| silu(g.min(d.limit)) * u.clamp(-d.limit, d.limit))
                .collect();
            matvec(w("w2"), d.width, d.inter, &act)
                .into_iter()
                .map(|v| v * scale)
                .collect()
        };
        let mut out = expert(&at("ffn.shared_experts"), 1.0);
        for &e in &ids {
            let weight = score[e] / sum * d.route_scale;
            for (o, v) in out
                .iter_mut()
                .zip(expert(&at(&format!("ffn.experts.{e}")), weight))
            {
                *o += v;
            }
        }
        out
    }

    /// The model over `tokens` from position 0 with a cache of `capacity` positions.
    pub(super) fn run(&self, tokens: &[usize], capacity: usize) -> Run {
        let d = &self.d;
        let mut hist = vec![LayerHistory::default(); d.layers];
        let mut logits = Vec::new();
        for (p, &token) in tokens.iter().enumerate() {
            let embed = &self.t("embed.weight")[token * d.width..(token + 1) * d.width];
            let mut streams: Vec<Vec<f64>> = vec![embed.to_vec(); d.hc];
            for (layer, h) in hist.iter_mut().enumerate() {
                let (post, comb, collapsed) =
                    self.mhc(&streams, &format!("layers.{layer}.hc_attn"));
                let normed = rms_scaled(
                    &collapsed,
                    self.t(&format!("layers.{layer}.attn_norm.weight")),
                    d.eps,
                );
                let attended = self.attention(layer, &normed, p, h);
                streams = self.place(&post, &comb, &attended, &streams);

                let (post, comb, collapsed) = self.mhc(&streams, &format!("layers.{layer}.hc_ffn"));
                let normed = rms_scaled(
                    &collapsed,
                    self.t(&format!("layers.{layer}.ffn_norm.weight")),
                    d.eps,
                );
                let routed = self.moe(layer, token, &normed);
                streams = self.place(&post, &comb, &routed, &streams);
            }
            let flat: Vec<f64> = streams.iter().flatten().copied().collect();
            let mixes = matvec(
                self.t("hc_head_fn"),
                d.hc,
                d.hc * d.width,
                &rms(&flat, d.eps),
            );
            let (base, scale) = (self.t("hc_head_base"), self.t("hc_head_scale")[0]);
            let mut collapsed = vec![0.0; d.width];
            for (s, stream) in streams.iter().enumerate() {
                let pre = sigmoid(mixes[s] * scale + base[s]) + d.hc_eps;
                for (c, v) in collapsed.iter_mut().zip(stream) {
                    *c += pre * v;
                }
            }
            let normed = rms_scaled(&collapsed, self.t("norm.weight"), d.eps);
            logits.push(matvec(self.t("head.weight"), d.vocab, d.width, &normed));
        }

        // The carried state, `[capacity, width]` per history, in the order the graph carries them.
        let pad = |rows: &[Vec<f64>], width: usize| {
            let mut out = vec![0.0; capacity * width];
            for (p, row) in rows.iter().enumerate() {
                out[p * width..(p + 1) * width].copy_from_slice(row);
            }
            out
        };
        let mut state = Vec::new();
        for (layer, h) in hist.iter().enumerate() {
            let kind = d.kind(layer);
            let name = |s: &str| format!("kv.l{layer}.{s}");
            state.push((name("kv"), pad(&h.kv, d.head_dim)));
            if kind != Kind::Sliding {
                let raw = if kind == Kind::Csa {
                    2 * d.head_dim
                } else {
                    d.head_dim
                };
                state.push((name("compress.kv"), pad(&h.compress_kv, raw)));
                state.push((name("compress.gate"), pad(&h.compress_gate, raw)));
            }
            if kind == Kind::Csa {
                let raw = 2 * d.index_dim;
                state.push((name("indexer.kv"), pad(&h.indexer_kv, raw)));
                state.push((name("indexer.gate"), pad(&h.indexer_gate, raw)));
            }
        }
        Run { logits, state }
    }
}

//! Multi-head latent attention (DeepSeek-V2 and later), one body for every phase.
//!
//! The keys and values of every head are rebuilt from one shared low-rank latent per position. A
//! step carries only the normalized latent `c` (`kv_lora_rank` wide) and the shared rotated key
//! slice (`qk_rope_head_dim` wide), each as a one-head cache in the layout the step chose (a
//! private `[rows, 1, capacity, d]` cache, or one `[pool_slots, 1, d]` pool), written by
//! [`attention::carried_cache`](super::attention::carried_cache) exactly as dense attention's K and
//! V are. Every step then expands the whole cached latent through the up projection into per-head
//! `k_nope` and `v` (no weight absorption), joins the shared rotated slice onto each head's key and
//! attends under the causal-over-cache mask.
//!
//! Q is a plain projection or the low-rank `q_a -> norm -> q_b`; either way it splits per head into
//! an unrotated and a rotated slice. The rotary pairing is DeepSeek's interleaved one
//! (`view_as_complex` of adjacent pairs).
//!
//! The up projection is stored either as HF's one `kv_b_proj` (`[heads * (nope + v), rank]`, read
//! through one dense contraction and split per head in the graph) or as llama.cpp's two per-head
//! tensors, `k_b` `[heads, rank, nope]` and `v_b` `[heads, v, rank]`, which the graph contracts in
//! their stored orientation (ADR-0103: nothing is rebuilt into a derived buffer).
//!
//! A family may narrow which cached positions a query attends to ([`Selection`]): DeepSeek sparse
//! attention hands the causal mask and the low-rank query to a selector and attends under the mask
//! it returns.

use poot_graph_ir::{Builder, Traced, ops};
use poot_quant::weights::{AttnRole, MlaRole, WeightId, WeightMap, WeightRole};

use super::attention::{carried_cache, first_writes};
use super::linear::{Weight, WeightError, linear};
use super::norm::{NormParams, norm};
use super::rope::{RopeParams, apply_rope, rope_rows};
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;
use crate::model::{TraceError, WiringError};

/// How the per-head keys and values are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvForm {
    /// HF's one `kv_b_proj`, `[heads * (nope + v), kv_lora_rank]`.
    Fused,
    /// llama.cpp's `k_b` `[heads, kv_lora_rank, nope]` and `v_b` `[heads, v, kv_lora_rank]`.
    Split,
}

/// Checked MLA parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaParams {
    heads: usize,
    kv_rank: usize,
    q_rank: Option<usize>,
    nope: usize,
    rope_dim: usize,
    v_dim: usize,
    norm: NormParams,
    rope: RopeParams,
    scale: f32,
    kv: KvForm,
}

impl MlaParams {
    /// `heads` heads of `nope + rope_dim` query/key width and `v_dim` value width over a
    /// `kv_rank`-wide latent; the query is low-rank when `q_rank` is set. `rope` is the rotation of
    /// the rotated slice (its width is `rope_dim`) and `scale` the score scale (the family's: it
    /// carries YaRN's softmax correction).
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        heads: usize,
        kv_rank: usize,
        q_rank: Option<usize>,
        nope: usize,
        rope_dim: usize,
        v_dim: usize,
        norm: NormParams,
        rope: RopeParams,
        scale: f32,
        kv: KvForm,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("num_attention_heads", heads),
            ("kv_lora_rank", kv_rank),
            ("qk_nope_head_dim", nope),
            ("qk_rope_head_dim", rope_dim),
            ("v_head_dim", v_dim),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if q_rank == Some(0) {
            return Err(ParamError::new("q_lora_rank", ConfigReason::Zero));
        }
        if rope.rotary() != rope_dim {
            return Err(ParamError::new(
                "qk_rope_head_dim",
                ConfigReason::Unsupported,
            ));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(ParamError::new("scale", ConfigReason::NotFinitePositive));
        }
        Ok(Self {
            heads,
            kv_rank,
            q_rank,
            nope,
            rope_dim,
            v_dim,
            norm,
            rope,
            scale,
            kv,
        })
    }

    /// The attention head count.
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// The per-head query/key width the scores contract over.
    pub fn qk_dim(&self) -> usize {
        self.nope + self.rope_dim
    }

    /// The rank of the low-rank query bottleneck, or `None` when Q is one plain projection.
    pub fn q_rank(&self) -> Option<usize> {
        self.q_rank
    }

    /// The rotation of the rotated slice.
    pub fn rope(&self) -> RopeParams {
        self.rope
    }

    /// The score scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Query {
    Plain(Weight),
    LowRank {
        a: Weight,
        a_norm: Weight,
        b: Weight,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum KvUp {
    Fused(Weight),
    Split { k: Weight, v: Weight },
}

/// A layer's MLA projections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaWeights {
    q: Query,
    kv_a: Weight,
    kv_a_norm: Weight,
    kv_up: KvUp,
    o: Weight,
}

impl MlaWeights {
    /// Resolves layer `layer`'s query, latent KV, up-projection and output tensors from `map` and
    /// checks each shape against `p` and the model `width`; a missing or mis-shaped tensor is a typed
    /// [`WeightError`].
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &MlaParams,
    ) -> Result<Self, WeightError> {
        let mla = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Mla(role)), shape)
        };
        let attn = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Attn(role)), shape)
        };
        let (h, qk) = (p.heads, p.qk_dim());
        let q = match p.q_rank {
            Some(r) => Query::LowRank {
                a: mla(MlaRole::QA, &[r, width])?,
                a_norm: mla(MlaRole::QANorm, &[r])?,
                b: mla(MlaRole::QB, &[h * qk, r])?,
            },
            None => Query::Plain(attn(AttnRole::Q, &[h * qk, width])?),
        };
        let kv_up = match p.kv {
            KvForm::Fused => KvUp::Fused(mla(MlaRole::KvB, &[h * (p.nope + p.v_dim), p.kv_rank])?),
            KvForm::Split => KvUp::Split {
                k: mla(MlaRole::KB, &[h, p.kv_rank, p.nope])?,
                v: mla(MlaRole::VB, &[h, p.v_dim, p.kv_rank])?,
            },
        };
        Ok(Self {
            q,
            kv_a: mla(MlaRole::KvA, &[p.kv_rank + p.rope_dim, width])?,
            kv_a_norm: mla(MlaRole::KvANorm, &[p.kv_rank])?,
            kv_up,
            o: attn(AttnRole::O, &[width, h * p.v_dim])?,
        })
    }
}

/// Replaces the causal mask of the attention with the one a selector builds from [`Selection`].
pub type Selector<'a> = &'a dyn Fn(&Builder, Selection) -> Result<Traced, TraceError>;

/// What a selector sees of the attention it narrows.
#[derive(Clone, Copy, Debug)]
pub struct Selection {
    /// The normalized low-rank query `[rows, tokens, q_rank]`.
    pub qr: Traced,
    /// The causal-over-cache additive mask `[rows, 1, tokens, capacity]`.
    pub mask: Traced,
}

/// MLA of `x` (`[rows, tokens, width]`) at layer `layer`: `[rows, tokens, width]`. The layer's two
/// caches (`kv.l{layer}.c`, `kv.l{layer}.rope`) are carried by `step`. With a `select`or the
/// attention runs under the mask it returns in place of the causal one; a selector needs the
/// low-rank query, which the family guarantees by refusing a plain query beside one.
pub fn mla(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &MlaWeights,
    p: &MlaParams,
    select: Option<Selector<'_>>,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (h, qk, nope, rd, vd, rank) = (p.heads, p.qk_dim(), p.nope, p.rope_dim, p.v_dim, p.kv_rank);
    let rope = rope_rows(b, p.rope, step.pos(), cap)?;

    // Q: the whole per-head query, its trailing slice rotated.
    let (q, qr) = match &w.q {
        Query::Plain(q) => (linear(b, x, q, None)?, None),
        Query::LowRank { a, a_norm, b: up } => {
            let qr = norm(b, linear(b, x, a, None)?, a_norm, p.norm)?;
            (linear(b, qr, up, None)?, Some(qr))
        }
    };
    let q = b.transpose(b.reshape(q, vec![rows, t, h, qk])?, vec![0, 2, 1, 3])?;
    let q_pe = apply_rope(b, b.slice(q, 3, nope, qk)?, &rope)?;
    let q = b.concat(3, &[b.slice(q, 3, 0, nope)?, q_pe])?;

    // The shared latent and rotated key slice of the new tokens.
    let kv_a = linear(b, x, &w.kv_a, None)?;
    let c_new = norm(b, b.slice(kv_a, 2, 0, rank)?, &w.kv_a_norm, p.norm)?;
    let c_new = b.reshape(c_new, vec![rows, 1, t, rank])?;
    let k_pe_new = b.reshape(b.slice(kv_a, 2, rank, rank + rd)?, vec![rows, 1, t, rd])?;
    let k_pe_new = apply_rope(b, k_pe_new, &rope)?;

    let write_at = first_writes(b, step)?;
    let c = carried_cache(b, step, &write_at, layer, "c", c_new, 1, rank)?;
    let k_pe = carried_cache(b, step, &write_at, layer, "rope", k_pe_new, 1, rd)?;

    // Every head's keys and values from the whole cached latent.
    let (k_nope, v) = match &w.kv_up {
        KvUp::Fused(up) => {
            let c = b.reshape(c, vec![rows, cap, rank])?;
            let kv = linear(b, c, up, None)?;
            let kv = b.transpose(
                b.reshape(kv, vec![rows, cap, h, nope + vd])?,
                vec![0, 2, 1, 3],
            )?;
            (b.slice(kv, 3, 0, nope)?, b.slice(kv, 3, nope, nope + vd)?)
        }
        KvUp::Split { k, v } => {
            // `c [rows, 1, cap, rank]` against each head's `k_b [rank, nope]` as stored and
            // `v_b [v, rank]` transposed: `[rows, heads, cap, ..]`.
            let k = b.reshape(k.declare_f32(b)?, vec![1, h, rank, nope])?;
            let v = b.transpose(
                b.reshape(v.declare_f32(b)?, vec![1, h, vd, rank])?,
                vec![0, 1, 3, 2],
            )?;
            (b.matmul(c, k)?, b.matmul(c, v)?)
        }
    };
    let k = b.concat(3, &[k_nope, ops::repeat_kv(b, k_pe, h)?])?;

    let mask = ops::causal_mask_from_pos(b, step.pos(), cap, None)?;
    let mask = match (select, qr) {
        (Some(select), Some(qr)) => select(b, Selection { qr, mask })?,
        (Some(_), None) => {
            return Err(WiringError::SelectorWithoutLowRankQuery.into());
        }
        (None, _) => mask,
    };
    let out = ops::attention_masked(b, q, k, v, 1, p.scale, mask)?;
    let out = b.reshape(b.transpose(out, vec![0, 2, 1, 3])?, vec![rows, t, h * vd])?;
    linear(b, out, &w.o, None)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use poot_graph_ir::rope_table::RopeFlavor;
    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const WIDTH: usize = 6;
    const HEADS: usize = 2;
    const RANK: usize = 4;
    const Q_RANK: usize = 3;
    const NOPE: usize = 3;
    const ROPE: usize = 2;
    const V: usize = 5;
    const QK: usize = NOPE + ROPE;
    const THETA: f32 = 10_000.0;
    const EPS: f32 = 1e-6;
    const SCALE: f32 = 0.37;

    fn params(q_rank: Option<usize>, kv: KvForm) -> MlaParams {
        let rope = RopeParams::new(ROPE, THETA, &RopeFlavor::Plain)
            .unwrap()
            .interleaved();
        MlaParams::new(
            HEADS,
            RANK,
            q_rank,
            NOPE,
            ROPE,
            V,
            NormParams::new(EPS).unwrap(),
            rope,
            SCALE,
            kv,
        )
        .unwrap()
    }

    fn mla_id(role: MlaRole) -> WeightId {
        WeightId::layer(0, WeightRole::Mla(role))
    }

    fn attn_id(role: AttnRole) -> WeightId {
        WeightId::layer(0, WeightRole::Attn(role))
    }

    /// Every weight of a layer, the up projection in the form `kv` names; a split fixture is the
    /// same values re-laid per head from `fused`.
    fn fixture(q_rank: Option<usize>, kv: KvForm) -> Fx {
        let mut fx = Fx::new(31);
        match q_rank {
            Some(r) => {
                fx.f32(mla_id(MlaRole::QA), &[r, WIDTH]);
                fx.set(
                    mla_id(MlaRole::QANorm),
                    &[r],
                    vec![1.2, 0.8, 1.1][..r].to_vec(),
                );
                fx.bf16(mla_id(MlaRole::QB), &[HEADS * QK, r]);
            }
            None => fx.f32(attn_id(AttnRole::Q), &[HEADS * QK, WIDTH]),
        }
        fx.f32(mla_id(MlaRole::KvA), &[RANK + ROPE, WIDTH]);
        fx.set(mla_id(MlaRole::KvANorm), &[RANK], vec![0.9, 1.3, 1.0, 0.7]);
        fx.f32(attn_id(AttnRole::O), &[WIDTH, HEADS * V]);
        let fused = fused_up();
        match kv {
            KvForm::Fused => fx.set(mla_id(MlaRole::KvB), &[HEADS * (NOPE + V), RANK], fused),
            KvForm::Split => {
                // `k_b [h, rank, nope]` and `v_b [h, v, rank]` from `kv_b [h * (nope + v), rank]`.
                let (mut k, mut v) = (Vec::new(), Vec::new());
                for h in 0..HEADS {
                    let row = |c: usize, r: usize| fused[(h * (NOPE + V) + c) * RANK + r];
                    for r in 0..RANK {
                        k.extend((0..NOPE).map(|c| row(c, r)));
                    }
                    for c in 0..V {
                        v.extend((0..RANK).map(|r| row(NOPE + c, r)));
                    }
                }
                fx.set(mla_id(MlaRole::KB), &[HEADS, RANK, NOPE], k);
                fx.set(mla_id(MlaRole::VB), &[HEADS, V, RANK], v);
            }
        }
        fx
    }

    fn fused_up() -> Vec<f32> {
        oracle::random(HEADS * (NOPE + V) * RANK, 77)
    }

    /// The stored tensors of one layer in the form `kv`, by role: every graph const.
    fn weights(fx: &Fx, p: &MlaParams) -> MlaWeights {
        MlaWeights::new(&fx.map(), 0, WIDTH, p).unwrap()
    }

    struct Run {
        out: Vec<f32>,
        state: Vec<Vec<f32>>,
    }

    /// One chunk of `x` (`[tokens, WIDTH]`) from position `start` over `state` (zeros when empty).
    fn run(fx: &Fx, p: &MlaParams, x: &[f32], start: i32, state: &[Vec<f32>]) -> Run {
        let w = weights(fx, p);
        let tokens = x.len() / WIDTH;
        let b = Builder::for_tests();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All)).unwrap();
        let xs = b
            .constant("x", TensorType::f32(vec![1, tokens, WIDTH]))
            .unwrap();
        let y = mla(&b, &step, 0, xs, &w, p, None).unwrap();
        let g = step.finish(b, y).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
            .iter()
            .map(Vec::as_slice)
            .collect();
        let ids: Vec<i32> = vec![0; tokens];
        let pos: Vec<i32> = (start..).take(tokens).collect();
        let (out, state) = fx.eval(
            &g,
            &[("x", x)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &state,
        );
        Run { out, state }
    }

    fn matvec(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        (0..rows)
            .map(|r| (0..k).map(|j| f64::from(w[r * k + j]) * x[j]).sum())
            .collect()
    }

    fn rms(x: &[f64], scale: &[f32]) -> Vec<f64> {
        let inv =
            1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(EPS)).sqrt();
        x.iter()
            .zip(scale)
            .map(|(v, s)| v * inv * f64::from(*s))
            .collect()
    }

    /// DeepSeek's interleaved rotation in f64: each adjacent pair `(a, b)` of `row` turns by
    /// `pos * theta^(-2i / d)`.
    fn rotate(row: &[f64], pos: usize) -> Vec<f64> {
        let mut out = row.to_vec();
        for i in 0..row.len() / 2 {
            let angle = pos as f64 * f64::from(THETA).powf(-2.0 * i as f64 / row.len() as f64);
            let (s, c) = angle.sin_cos();
            out[2 * i] = row[2 * i] * c - row[2 * i + 1] * s;
            out[2 * i + 1] = row[2 * i] * s + row[2 * i + 1] * c;
        }
        out
    }

    /// MLA over `x` from position 0, written straight from the definition in f64: the output of
    /// every token and the carried latent and rotated-key caches (`[cap, rank]`, `[cap, rope]`).
    fn reference(fx: &Fx, q_rank: Option<usize>, x: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let tokens = x.len() / WIDTH;
        let value = |id| fx.values(id);
        let fused = fused_up();
        let (mut latent, mut kpe) = (Vec::new(), Vec::new());
        let (mut q_all, mut out) = (Vec::new(), Vec::new());
        for t in 0..tokens {
            let row: Vec<f64> = x[t * WIDTH..(t + 1) * WIDTH]
                .iter()
                .map(|&v| v.into())
                .collect();
            let q = match q_rank {
                Some(r) => {
                    let qa = matvec(&value(mla_id(MlaRole::QA)), r, WIDTH, &row);
                    let qr = rms(&qa, &value(mla_id(MlaRole::QANorm)));
                    matvec(&value(mla_id(MlaRole::QB)), HEADS * QK, r, &qr)
                }
                None => matvec(&value(attn_id(AttnRole::Q)), HEADS * QK, WIDTH, &row),
            };
            let mut q_heads = Vec::new();
            for h in 0..HEADS {
                let head = &q[h * QK..(h + 1) * QK];
                let mut full = head[..NOPE].to_vec();
                full.extend(rotate(&head[NOPE..], t));
                q_heads.push(full);
            }
            q_all.push(q_heads);
            let kv_a = matvec(&value(mla_id(MlaRole::KvA)), RANK + ROPE, WIDTH, &row);
            latent.push(rms(&kv_a[..RANK], &value(mla_id(MlaRole::KvANorm))));
            kpe.push(rotate(&kv_a[RANK..], t));
        }
        for (t, q_heads) in q_all.iter().enumerate() {
            let mut heads_out = Vec::new();
            for h in 0..HEADS {
                // The keys and values of every visible position from its latent.
                let expand = |j: usize| matvec(&fused, HEADS * (NOPE + V), RANK, &latent[j]);
                let scores: Vec<f64> = (0..=t)
                    .map(|j| {
                        let kv = expand(j);
                        let mut key = kv[h * (NOPE + V)..h * (NOPE + V) + NOPE].to_vec();
                        key.extend(&kpe[j]);
                        key.iter().zip(&q_heads[h]).map(|(a, b)| a * b).sum::<f64>()
                            * f64::from(SCALE)
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::MIN, f64::max);
                let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = exp.iter().sum();
                let mut acc = vec![0.0; V];
                for (j, e) in exp.iter().enumerate() {
                    let kv = expand(j);
                    for c in 0..V {
                        acc[c] += e / total * kv[h * (NOPE + V) + NOPE + c];
                    }
                }
                heads_out.extend(acc);
            }
            out.extend(matvec(
                &value(attn_id(AttnRole::O)),
                WIDTH,
                HEADS * V,
                &heads_out,
            ));
        }
        let pad = |rows: &[Vec<f64>], d: usize| {
            let mut flat: Vec<f64> = rows.concat();
            flat.resize(CAP * d, 0.0);
            flat
        };
        (out, pad(&latent, RANK), pad(&kpe, ROPE))
    }

    fn inputs(tokens: usize) -> Vec<f32> {
        oracle::random(tokens * WIDTH, 5)
            .into_iter()
            .map(|v| 2.0 * v)
            .collect()
    }

    /// SC-002: MLA equals the f64 reference (outputs and both carried caches) for a plain and a
    /// low-rank query, with the up projection as HF's one tensor and as llama.cpp's split per-head
    /// pair; a chunk continued from the carried caches equals the whole prompt. Mutation: skip the
    /// RoPE rotation of the cached key slice (`k_pe_new`); every row goes red.
    #[test]
    fn mla_matches_an_f64_reference_in_every_form_and_chunks_continue() {
        let x = inputs(5);
        for q_rank in [None, Some(Q_RANK)] {
            let (want, want_latent, want_kpe) =
                reference(&fixture(q_rank, KvForm::Fused), q_rank, &x);
            for kv in [KvForm::Fused, KvForm::Split] {
                let fx = fixture(q_rank, kv);
                let p = params(q_rank, kv);
                let whole = run(&fx, &p, &x, 0, &[]);
                oracle::assert_matches_f64(&whole.out, &want, 1e-5);
                oracle::assert_matches_f64(&whole.state[0], &want_latent, 1e-5);
                oracle::assert_matches_f64(&whole.state[1], &want_kpe, 1e-5);

                let first = run(&fx, &p, &x[..3 * WIDTH], 0, &[]);
                let second = run(&fx, &p, &x[3 * WIDTH..], 3, &first.state);
                let continued = [first.out, second.out].concat();
                oracle::assert_matches_f64(&continued, &want, 1e-5);
                oracle::assert_matches_f64(&second.state[0], &want_latent, 1e-5);
            }
        }
    }

    /// Each weight matters: perturbing it changes the output, in every form.
    #[test]
    fn every_mla_weight_changes_the_output() {
        let x = inputs(4);
        for (q_rank, kv) in [
            (Some(Q_RANK), KvForm::Fused),
            (Some(Q_RANK), KvForm::Split),
            (None, KvForm::Fused),
        ] {
            let fx = fixture(q_rank, kv);
            let p = params(q_rank, kv);
            let base = run(&fx, &p, &x, 0, &[]).out;
            let ids: BTreeMap<_, _> = fx.map().iter().map(|(id, ..)| (id, ())).collect();
            for id in ids.into_keys() {
                let mut perturbed = fx.clone();
                perturbed.perturb(id);
                oracle::assert_differs(&run(&perturbed, &p, &x, 0, &[]).out, &base, id);
            }
        }
    }

    #[test]
    fn params_refuse_an_impossible_attention() {
        let rope = RopeParams::new(ROPE, THETA, &RopeFlavor::Plain).unwrap();
        let norm = NormParams::new(EPS).unwrap();
        let new = |heads, rope_dim, scale| {
            MlaParams::new(
                heads,
                RANK,
                None,
                NOPE,
                rope_dim,
                V,
                norm,
                rope,
                scale,
                KvForm::Fused,
            )
        };
        assert_eq!(
            new(0, ROPE, 1.0),
            Err(ParamError::new("num_attention_heads", ConfigReason::Zero))
        );
        assert_eq!(
            new(2, ROPE + 2, 1.0),
            Err(ParamError::new(
                "qk_rope_head_dim",
                ConfigReason::Unsupported
            ))
        );
        assert_eq!(
            new(2, ROPE, f32::NAN),
            Err(ParamError::new("scale", ConfigReason::NotFinitePositive))
        );
    }
}

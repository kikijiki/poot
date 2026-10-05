//! `Runner::load_gguf`'s DeepSeek-V3 detection/config/weights (`gguf.rs`'s
//! `gguf_deepseek2_is_v3_style`/`gguf_config_deepseek3_moe`/`gguf_deepseek3_weights`), checked
//! loader-level bit-exact against an independent hand-rolled Rust reference (as `deepseek2_load_tests`
//! above: a from-scratch forward pass, not a copy of the traced graph), on a synthetic in-memory GGUF
//! (`poot_load::gguf::write_gguf`) built from known ground-truth values, since no real DeepSeek-V3
//! checkpoint was available (as `gguf_fixture_tests` in `gguf.rs`). Ground truth is generated in poot's
//! `[in,out]` matmul convention, then transformed by hand into GGUF's wire layout (the inverse of
//! `gguf_deepseek3_weights`'s `transpose2d`/`reconstruct_deepseek2_kv_b`/`transpose_experts`/
//! `concat_experts_last`), so the real loader transforms run end to end. A transpose/reconstruct bug in
//! the V3-only tensor reads (`exp_probs_b.bias`) or the shared V2 transforms would give a materially
//! different output (both sides are f32). The fixture uses a real group-limiting case (`n_group: 3`,
//! `topk_group: 1` on 6 experts: one group of 2 selected out of 3), not the `n_group: 1` no-op that
//! `poot_models::deepseek3`'s tests cover, since that is the one new numerical surface a GGUF-loaded V3
//! checkpoint adds over V2.

use super::super::*;

use poot_eval::Value;
use poot_load::gguf::{GgufValue, write_gguf};
use poot_models::deepseek2::DeepseekV2Config;
use poot_models::deepseek3::DeepseekV3MoeParams;
use poot_models::deepseek3::trace_deepseek3_prefill;
use poot_tensor::HostTensor;

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
fn wgt(seed: u64, n: usize) -> Vec<f32> {
    fill(seed, n).iter().map(|v| v * 0.1).collect()
}
fn gamma(seed: u64, n: usize) -> Vec<f32> {
    fill(seed, n).iter().map(|v| 1.0 + v * 0.05).collect()
}
/// The correction bias is a real-scale additive shift (as `poot_models::deepseek3`'s test fixture), large
/// enough to change which experts a group-limited top-k selects on this tiny fixture.
fn bias_vec(seed: u64, n: usize) -> Vec<f32> {
    fill(seed, n).iter().map(|v| v * 0.5).collect()
}

use poot_test_util::f32_bytes;

/// Row-major `[rows,cols]` -> row-major `[cols,rows]`: turns a poot-convention `[in,out]` ground-truth
/// matrix into GGUF's `[out,in]` wire layout, the inverse of `gguf.rs`'s `transpose2d`.
fn transpose_flat(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = m[r * cols + c];
        }
    }
    out
}

use poot_test_util::rmsnorm_ref;
use poot_test_util::silu_ref;
fn sigmoid_ref(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}
use poot_test_util::linear_ref;
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

/// V3 router: sigmoid + selection-only correction bias + group-limited top-k + renormalize (see
/// `poot_models::deepseek3::deepseek3_router_gate`, whose `MoEGate.forward` algorithm this mirrors). This
/// fixture's random weights never produce an exact float tie, so plain descending-sort top-k is
/// bit-equivalent to the graph's stable pairwise-rank construction.
fn deepseek3_router_ref(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
) -> Vec<(usize, f32)> {
    let e = logits.len();
    let group_size = e / n_group;
    let p: Vec<f32> = logits.iter().map(|&v| sigmoid_ref(v)).collect();
    let sc: Vec<f32> = p.iter().zip(bias.iter()).map(|(&a, &b)| a + b).collect();
    let mut group_scores = vec![0.0f32; n_group];
    for (g, gs) in group_scores.iter_mut().enumerate() {
        let mut vals: Vec<f32> = sc[g * group_size..(g + 1) * group_size].to_vec();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        *gs = vals[0] + vals[1]; // top-2 within the group (real V3 always has group_size >= 2)
    }
    let mut group_idx: Vec<usize> = (0..n_group).collect();
    group_idx.sort_by(|&a, &b| group_scores[b].partial_cmp(&group_scores[a]).unwrap());
    let selected_groups: std::collections::HashSet<usize> =
        group_idx[..topk_group].iter().copied().collect();
    let mut cand: Vec<usize> = (0..e)
        .filter(|&i| selected_groups.contains(&(i / group_size)))
        .collect();
    cand.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap());
    let sel = &cand[..top_k];
    let denom: f32 = sel.iter().map(|&i| p[i]).sum::<f32>() + 1e-20;
    sel.iter()
        .map(|&i| (i, (p[i] / denom) * routed_scaling_factor))
        .collect()
}

/// Independent from-scratch Rust reference forward pass reading the same ground-truth weight values
/// (`ground`, poot's `[in,out]` convention) that are hand-transposed into the synthetic GGUF's wire
/// layout below, not `runner.weights`. Unlike `deepseek2_prefill_ref_from_runner` (which validates the
/// tracer against production-loaded weights), this validates the loader's transforms, since there is no
/// safetensors V3 loader to cross-check as `deepseek2_tiny_gguf_matches_safetensors` does for V2.
#[allow(clippy::too_many_arguments)]
fn deepseek3_prefill_ref(
    cfg: &DeepseekV2Config,
    mp: &DeepseekV3MoeParams,
    tokens: &[u32],
    ground: &HashMap<String, Vec<f32>>,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let g = |name: &str| -> &[f32] { &ground[name] };
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();
    let l = tokens.len();
    let embed = g("model.embed_tokens.weight");
    let half = rope_d / 2;

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

        let q_flat: Vec<Vec<f32>> = match cfg.q_lora_rank {
            Some(r) => {
                let wqa = g(&p("self_attn.q_a_proj.weight"));
                let wqaln = g(&p("self_attn.q_a_layernorm.weight"));
                let wqb = g(&p("self_attn.q_b_proj.weight"));
                normed
                    .iter()
                    .map(|row| {
                        let qa = linear_ref(row, wqa, h, r);
                        let qa_n = rmsnorm_ref(&qa, wqaln, r, cfg.eps);
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
            let c_kv = rmsnorm_ref(kv_nope_raw, wkvaln, kv_rank, cfg.eps);
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
            let normed2 = rmsnorm_ref(row, ln2w, h, cfg.eps);
            let ffn_out = if mp.is_moe_layer(li) {
                let router_w = g(&p("mlp.gate.weight"));
                let bias = g(&p("mlp.gate.e_score_correction_bias"));
                let gate_up = g(&p("mlp.experts.gate_up_proj.weight"));
                let down = g(&p("mlp.experts.down_proj.weight"));
                let logits = linear_ref(&normed2, router_w, h, mp.n_routed_experts);
                let selected = deepseek3_router_ref(
                    &logits,
                    bias,
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
    let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
    let lm_head = g("lm_head.weight");
    linear_ref(&last, lm_head, h, cfg.vocab)
}

/// Builds a synthetic V3-style "deepseek2" GGUF fixture (3 tiny layers: layer 0 dense, layers 1-2
/// routed; `n_group: 3`/`topk_group: 1` on 6 experts, a real group-limiting case), loads it through
/// `Runner::load_gguf`, and checks the traced-and-bound `trace_deepseek3_prefill` graph against
/// [`deepseek3_prefill_ref`] reading the same pre-GGUF-encoding ground-truth values.
#[test]
fn deepseek3_synthetic_gguf_matches_hand_rolled_reference() {
    let (h, hq, r, kv_rank, nope, rope_d, vd) =
        (8usize, 2usize, 4usize, 4usize, 2usize, 2usize, 2usize);
    let qk_head_dim = nope + rope_d;
    let layers = 3usize;
    let vocab = 6usize;
    let max_pos = 8usize;
    let eps = 1e-5f32;
    let rope_theta = 10_000.0f32;
    let first_k_dense_replace = 1usize;
    let dense_inter = 6usize;
    let (n_routed_experts, top_k, n_group, topk_group, moe_inter, n_shared_experts) =
        (6usize, 2usize, 3usize, 1usize, 4usize, 1usize);
    let routed_scaling_factor = 1.7f32;
    let shared_inter = moe_inter * n_shared_experts;

    let cfg = DeepseekV2Config {
        vocab,
        hidden: h,
        layers,
        n_heads: hq,
        q_lora_rank: Some(r),
        kv_lora_rank: kv_rank,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: rope_d,
        v_head_dim: vd,
        eps,
        max_pos,
        rope_theta,
        yarn: None,
    };
    let mp = DeepseekV3MoeParams {
        n_routed_experts,
        top_k,
        moe_inter,
        n_shared_experts,
        dense_inter,
        first_k_dense_replace,
        n_group,
        topk_group,
        routed_scaling_factor,
    };

    // Ground truth (poot's `[in,out]` matmul convention), keyed by the HF-style names
    // `trace_deepseek3_prefill` binds; the shape of `all_weights()` in `poot_models::deepseek3`'s CPU-oracle
    // tests, reimplemented since that helper is private to the other crate.
    let mut ground: HashMap<String, Vec<f32>> = HashMap::new();
    let mut seed = 1u64;
    let mut next_seed = || {
        seed = seed.wrapping_add(0x9E37_79B9);
        seed
    };
    ground.insert(
        "model.embed_tokens.weight".to_string(),
        wgt(next_seed(), vocab * h),
    );
    for li in 0..layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        ground.insert(p("input_layernorm.weight"), gamma(next_seed(), h));
        ground.insert(p("post_attention_layernorm.weight"), gamma(next_seed(), h));
        ground.insert(p("self_attn.q_a_proj.weight"), wgt(next_seed(), h * r));
        ground.insert(p("self_attn.q_a_layernorm.weight"), gamma(next_seed(), r));
        ground.insert(
            p("self_attn.q_b_proj.weight"),
            wgt(next_seed(), r * hq * qk_head_dim),
        );
        ground.insert(
            p("self_attn.kv_a_proj_with_mqa.weight"),
            wgt(next_seed(), h * (kv_rank + rope_d)),
        );
        ground.insert(
            p("self_attn.kv_a_layernorm.weight"),
            gamma(next_seed(), kv_rank),
        );
        ground.insert(
            p("self_attn.kv_b_proj.weight"),
            wgt(next_seed(), kv_rank * hq * (nope + vd)),
        );
        ground.insert(p("self_attn.o_proj.weight"), wgt(next_seed(), hq * vd * h));
        if li < first_k_dense_replace {
            ground.insert(p("mlp.gate_proj.weight"), wgt(next_seed(), h * dense_inter));
            ground.insert(p("mlp.up_proj.weight"), wgt(next_seed(), h * dense_inter));
            ground.insert(p("mlp.down_proj.weight"), wgt(next_seed(), dense_inter * h));
        } else {
            ground.insert(p("mlp.gate.weight"), wgt(next_seed(), h * n_routed_experts));
            ground.insert(
                p("mlp.gate.e_score_correction_bias"),
                bias_vec(next_seed(), n_routed_experts),
            );
            let mut gate_up = vec![0.0f32; n_routed_experts * h * 2 * moe_inter];
            let mut down = vec![0.0f32; n_routed_experts * moe_inter * h];
            for e in 0..n_routed_experts {
                let ge = wgt(next_seed(), h * moe_inter);
                let ue = wgt(next_seed(), h * moe_inter);
                let de = wgt(next_seed(), moe_inter * h);
                for row in 0..h {
                    let dst = (e * h + row) * 2 * moe_inter;
                    gate_up[dst..dst + moe_inter]
                        .copy_from_slice(&ge[row * moe_inter..(row + 1) * moe_inter]);
                    gate_up[dst + moe_inter..dst + 2 * moe_inter]
                        .copy_from_slice(&ue[row * moe_inter..(row + 1) * moe_inter]);
                }
                let dst = e * moe_inter * h;
                down[dst..dst + moe_inter * h].copy_from_slice(&de);
            }
            ground.insert(p("mlp.experts.gate_up_proj.weight"), gate_up);
            ground.insert(p("mlp.experts.down_proj.weight"), down);
            ground.insert(
                p("mlp.shared_experts.gate_proj.weight"),
                wgt(next_seed(), h * shared_inter),
            );
            ground.insert(
                p("mlp.shared_experts.up_proj.weight"),
                wgt(next_seed(), h * shared_inter),
            );
            ground.insert(
                p("mlp.shared_experts.down_proj.weight"),
                wgt(next_seed(), shared_inter * h),
            );
        }
    }
    ground.insert("model.norm.weight".to_string(), gamma(next_seed(), h));
    ground.insert("lm_head.weight".to_string(), wgt(next_seed(), h * vocab));

    let (cos, sin) = poot_models::deepseek2::deepseek2_rope_tables_interleaved(
        max_pos, rope_d, rope_theta, None,
    );

    // Hand-transform the ground truth into GGUF's wire layout (the inverse of `gguf_deepseek3_weights`'s
    // transforms) and build the tensor/metadata list.
    const F32: u32 = 0;
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    tensors.push((
        "token_embd.weight".to_string(),
        vec![h as u64, vocab as u64],
        F32,
        f32_bytes(&ground["model.embed_tokens.weight"]),
    ));
    for li in 0..layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        let blk = |s: &str| format!("blk.{li}.{s}");
        tensors.push((
            blk("attn_norm.weight"),
            vec![h as u64],
            F32,
            f32_bytes(&ground[&p("input_layernorm.weight")]),
        ));
        tensors.push((
            blk("ffn_norm.weight"),
            vec![h as u64],
            F32,
            f32_bytes(&ground[&p("post_attention_layernorm.weight")]),
        ));
        tensors.push((
            blk("attn_q_a.weight"),
            vec![h as u64, r as u64],
            F32,
            f32_bytes(&transpose_flat(
                &ground[&p("self_attn.q_a_proj.weight")],
                h,
                r,
            )),
        ));
        tensors.push((
            blk("attn_q_a_norm.weight"),
            vec![r as u64],
            F32,
            f32_bytes(&ground[&p("self_attn.q_a_layernorm.weight")]),
        ));
        tensors.push((
            blk("attn_q_b.weight"),
            vec![r as u64, (hq * qk_head_dim) as u64],
            F32,
            f32_bytes(&transpose_flat(
                &ground[&p("self_attn.q_b_proj.weight")],
                r,
                hq * qk_head_dim,
            )),
        ));
        tensors.push((
            blk("attn_kv_a_mqa.weight"),
            vec![h as u64, (kv_rank + rope_d) as u64],
            F32,
            f32_bytes(&transpose_flat(
                &ground[&p("self_attn.kv_a_proj_with_mqa.weight")],
                h,
                kv_rank + rope_d,
            )),
        ));
        tensors.push((
            blk("attn_kv_a_norm.weight"),
            vec![kv_rank as u64],
            F32,
            f32_bytes(&ground[&p("self_attn.kv_a_layernorm.weight")]),
        ));
        // Split the combined `kv_b_proj` ground truth into llama.cpp's `attn_k_b`/`attn_v_b`, the inverse of
        // `gguf.rs`'s `reconstruct_deepseek2_kv_b`.
        let kvb = &ground[&p("self_attn.kv_b_proj.weight")];
        let out_per_head = nope + vd;
        let mut k_b = vec![0.0f32; hq * kv_rank * nope];
        let mut v_b = vec![0.0f32; hq * vd * kv_rank];
        for hh in 0..hq {
            let col0 = hh * out_per_head;
            for row in 0..kv_rank {
                for c in 0..nope {
                    k_b[hh * kv_rank * nope + row * nope + c] =
                        kvb[row * (hq * out_per_head) + col0 + c];
                }
                for c in 0..vd {
                    v_b[hh * vd * kv_rank + c * kv_rank + row] =
                        kvb[row * (hq * out_per_head) + col0 + nope + c];
                }
            }
        }
        tensors.push((
            blk("attn_k_b.weight"),
            vec![nope as u64, kv_rank as u64, hq as u64],
            F32,
            f32_bytes(&k_b),
        ));
        tensors.push((
            blk("attn_v_b.weight"),
            vec![kv_rank as u64, vd as u64, hq as u64],
            F32,
            f32_bytes(&v_b),
        ));
        tensors.push((
            blk("attn_output.weight"),
            vec![(hq * vd) as u64, h as u64],
            F32,
            f32_bytes(&transpose_flat(
                &ground[&p("self_attn.o_proj.weight")],
                hq * vd,
                h,
            )),
        ));
        if li < first_k_dense_replace {
            tensors.push((
                blk("ffn_gate.weight"),
                vec![h as u64, dense_inter as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.gate_proj.weight")],
                    h,
                    dense_inter,
                )),
            ));
            tensors.push((
                blk("ffn_up.weight"),
                vec![h as u64, dense_inter as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.up_proj.weight")],
                    h,
                    dense_inter,
                )),
            ));
            tensors.push((
                blk("ffn_down.weight"),
                vec![dense_inter as u64, h as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.down_proj.weight")],
                    dense_inter,
                    h,
                )),
            ));
        } else {
            tensors.push((
                blk("ffn_gate_inp.weight"),
                vec![h as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.gate.weight")],
                    h,
                    n_routed_experts,
                )),
            ));
            tensors.push((
                blk("exp_probs_b.bias"),
                vec![n_routed_experts as u64],
                F32,
                f32_bytes(&ground[&p("mlp.gate.e_score_correction_bias")]),
            ));
            let gate_up = &ground[&p("mlp.experts.gate_up_proj.weight")];
            let down = &ground[&p("mlp.experts.down_proj.weight")];
            let mut gate_wire = vec![0.0f32; n_routed_experts * moe_inter * h];
            let mut up_wire = vec![0.0f32; n_routed_experts * moe_inter * h];
            let mut down_wire = vec![0.0f32; n_routed_experts * h * moe_inter];
            for e in 0..n_routed_experts {
                let mut ge = vec![0.0f32; h * moe_inter];
                let mut ue = vec![0.0f32; h * moe_inter];
                for row in 0..h {
                    let src = (e * h + row) * 2 * moe_inter;
                    ge[row * moe_inter..(row + 1) * moe_inter]
                        .copy_from_slice(&gate_up[src..src + moe_inter]);
                    ue[row * moe_inter..(row + 1) * moe_inter]
                        .copy_from_slice(&gate_up[src + moe_inter..src + 2 * moe_inter]);
                }
                let ge_t = transpose_flat(&ge, h, moe_inter); // [I,H]
                let ue_t = transpose_flat(&ue, h, moe_inter);
                gate_wire[e * moe_inter * h..(e + 1) * moe_inter * h].copy_from_slice(&ge_t);
                up_wire[e * moe_inter * h..(e + 1) * moe_inter * h].copy_from_slice(&ue_t);
                let de = &down[e * moe_inter * h..(e + 1) * moe_inter * h];
                let de_t = transpose_flat(de, moe_inter, h); // [H,I]
                down_wire[e * h * moe_inter..(e + 1) * h * moe_inter].copy_from_slice(&de_t);
            }
            tensors.push((
                blk("ffn_gate_exps.weight"),
                vec![h as u64, moe_inter as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&gate_wire),
            ));
            tensors.push((
                blk("ffn_up_exps.weight"),
                vec![h as u64, moe_inter as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&up_wire),
            ));
            tensors.push((
                blk("ffn_down_exps.weight"),
                vec![moe_inter as u64, h as u64, n_routed_experts as u64],
                F32,
                f32_bytes(&down_wire),
            ));
            tensors.push((
                blk("ffn_gate_shexp.weight"),
                vec![h as u64, shared_inter as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.shared_experts.gate_proj.weight")],
                    h,
                    shared_inter,
                )),
            ));
            tensors.push((
                blk("ffn_up_shexp.weight"),
                vec![h as u64, shared_inter as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.shared_experts.up_proj.weight")],
                    h,
                    shared_inter,
                )),
            ));
            tensors.push((
                blk("ffn_down_shexp.weight"),
                vec![shared_inter as u64, h as u64],
                F32,
                f32_bytes(&transpose_flat(
                    &ground[&p("mlp.shared_experts.down_proj.weight")],
                    shared_inter,
                    h,
                )),
            ));
        }
    }
    tensors.push((
        "output_norm.weight".to_string(),
        vec![h as u64],
        F32,
        f32_bytes(&ground["model.norm.weight"]),
    ));
    tensors.push((
        "output.weight".to_string(),
        vec![h as u64, vocab as u64],
        F32,
        f32_bytes(&transpose_flat(&ground["lm_head.weight"], h, vocab)),
    ));

    let tokenizer_tokens: Vec<GgufValue> = ["a", "b", "c", "d", "e", "f"]
        .iter()
        .map(|s| GgufValue::Str(s.to_string()))
        .collect();
    let kvs: Vec<(&str, GgufValue)> = vec![
        (
            "general.architecture",
            GgufValue::Str("deepseek2".to_string()),
        ),
        ("deepseek2.embedding_length", GgufValue::U32(h as u32)),
        ("deepseek2.block_count", GgufValue::U32(layers as u32)),
        ("deepseek2.attention.head_count", GgufValue::U32(hq as u32)),
        ("deepseek2.attention.q_lora_rank", GgufValue::U32(r as u32)),
        (
            "deepseek2.attention.kv_lora_rank",
            GgufValue::U32(kv_rank as u32),
        ),
        (
            "deepseek2.attention.key_length_mla",
            GgufValue::U32(qk_head_dim as u32),
        ),
        (
            "deepseek2.rope.dimension_count",
            GgufValue::U32(rope_d as u32),
        ),
        (
            "deepseek2.attention.value_length_mla",
            GgufValue::U32(vd as u32),
        ),
        (
            "deepseek2.attention.layer_norm_rms_epsilon",
            GgufValue::F32(eps),
        ),
        ("deepseek2.context_length", GgufValue::U32(max_pos as u32)),
        ("deepseek2.rope.freq_base", GgufValue::F32(rope_theta)),
        (
            "deepseek2.feed_forward_length",
            GgufValue::U32(dense_inter as u32),
        ),
        (
            "deepseek2.expert_count",
            GgufValue::U32(n_routed_experts as u32),
        ),
        ("deepseek2.expert_used_count", GgufValue::U32(top_k as u32)),
        (
            "deepseek2.expert_feed_forward_length",
            GgufValue::U32(moe_inter as u32),
        ),
        (
            "deepseek2.expert_shared_count",
            GgufValue::U32(n_shared_experts as u32),
        ),
        (
            "deepseek2.leading_dense_block_count",
            GgufValue::U32(first_k_dense_replace as u32),
        ),
        (
            "deepseek2.expert_weights_scale",
            GgufValue::F32(routed_scaling_factor),
        ),
        (
            "deepseek2.expert_group_count",
            GgufValue::U32(n_group as u32),
        ),
        (
            "deepseek2.expert_group_used_count",
            GgufValue::U32(topk_group as u32),
        ),
        ("tokenizer.ggml.tokens", GgufValue::Array(tokenizer_tokens)),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let tensors_ref: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
        .iter()
        .map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone()))
        .collect();
    let bytes = write_gguf(&kvs, &tensors_ref);
    let path = poot_test_util::unique_temp_path("poot_deepseek3_synthetic_fixture.gguf");
    std::fs::write(&path, &bytes).expect("write synthetic deepseek3 gguf fixture");

    let runner = Runner::load_gguf(&path).expect("load synthetic deepseek3 gguf");
    assert_eq!(runner.arch, "deepseek_v3");
    assert!(
        runner.deepseek2.is_none(),
        "a V3-style checkpoint must not also set deepseek2"
    );
    let dp = runner
        .deepseek3
        .expect("Runner::load_gguf must set deepseek3 params for a V3-style deepseek2 gguf");
    assert_eq!(dp.moe.n_routed_experts, n_routed_experts);
    assert_eq!(dp.moe.top_k, top_k);
    assert_eq!(dp.moe.n_group, n_group);
    assert_eq!(dp.moe.topk_group, topk_group);
    assert_eq!(dp.moe.first_k_dense_replace, first_k_dense_replace);
    assert_eq!(dp.moe.n_shared_experts, n_shared_experts);
    assert!((dp.moe.routed_scaling_factor - routed_scaling_factor).abs() < 1e-6);
    assert_eq!(dp.cfg.q_lora_rank, Some(r));

    let tokens = [0u32, 2, 4, 1];
    let want = deepseek3_prefill_ref(&cfg, &mp, &tokens, &ground, &cos, &sin);

    let g = trace_deepseek3_prefill(dp.cfg, dp.moe, tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind deepseek3 prefill graph");
    let got = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");

    assert_eq!(got.shape(), vec![1, 1, vocab]);
    assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // GGUF-loaded-and-traced logits (actual) vs the hand-rolled reference (expected): a divergence here is
    // a real transpose/reconstruct bug in the new deepseek3 GGUF loader (gguf.rs), not float noise (both
    // paths are f32, no quantization involved).
    poot_test_util::assert_close(got.as_f32().unwrap(), &want, 1e-3);
}

/// Writes a minimal safetensors file (same format as `mpt_load.rs`'s/`bloom_load.rs`'s
/// `write_safetensors`; no shared fixture crate).
fn write_safetensors(path: &std::path::Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut data = Vec::new();
    let mut header = serde_json::Map::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}),
        );
    }
    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write safetensors fixture");
}

/// Safetensors-side counterpart of `deepseek3_synthetic_gguf_matches_hand_rolled_reference` above
/// (docs/updates/0791, "Not done" item 3): covers `Runner::load`'s `is_deepseek3()` detection, the
/// `DeepseekV3Params` config (`n_group`/`topk_group` read from `config.json`; field names checked
/// against a real `deepseek-ai/DeepSeek-V3` `config.json`, see `Qwen2HfConfig::is_deepseek3`), and
/// `build_weights`' `is_deepseek3()` block (`kv_a_proj_with_mqa.weight`/router transpose +
/// `fuse_qwen3_moe_experts`, plus the `mlp.gate.e_score_correction_bias` pass-through). Same dims and
/// ground-truth construction as the GGUF test above (duplicated on purpose, per this module's
/// per-test-fixture precedent) and the same group-limiting case (`n_group: 3`, `topk_group: 1` on 6
/// experts). Unlike the GGUF wire format (which splits `kv_b_proj` into `attn_k_b`/`attn_v_b`), a real HF
/// checkpoint stores the combined `kv_b_proj.weight`, so this fixture writes it as one `[out,in]` HF
/// tensor, transposed by the generic "proj.weight" rule like every other MLA projection.
#[test]
fn deepseek3_synthetic_safetensors_matches_hand_rolled_reference() {
    let (h, hq, r, kv_rank, nope, rope_d, vd) =
        (8usize, 2usize, 4usize, 4usize, 2usize, 2usize, 2usize);
    let qk_head_dim = nope + rope_d;
    let layers = 3usize;
    let vocab = 6usize;
    let max_pos = 8usize;
    let eps = 1e-5f32;
    let rope_theta = 10_000.0f32;
    let first_k_dense_replace = 1usize;
    let dense_inter = 6usize;
    let (n_routed_experts, top_k, n_group, topk_group, moe_inter, n_shared_experts) =
        (6usize, 2usize, 3usize, 1usize, 4usize, 1usize);
    let routed_scaling_factor = 1.7f32;
    let shared_inter = moe_inter * n_shared_experts;

    let cfg = DeepseekV2Config {
        vocab,
        hidden: h,
        layers,
        n_heads: hq,
        q_lora_rank: Some(r),
        kv_lora_rank: kv_rank,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: rope_d,
        v_head_dim: vd,
        eps,
        max_pos,
        rope_theta,
        yarn: None,
    };
    let mp = DeepseekV3MoeParams {
        n_routed_experts,
        top_k,
        moe_inter,
        n_shared_experts,
        dense_inter,
        first_k_dense_replace,
        n_group,
        topk_group,
        routed_scaling_factor,
    };

    // Ground truth (poot's `[in,out]` matmul convention), identical to the GGUF test above (same seeds and
    // shapes).
    let mut ground: HashMap<String, Vec<f32>> = HashMap::new();
    let mut seed = 1u64;
    let mut next_seed = || {
        seed = seed.wrapping_add(0x9E37_79B9);
        seed
    };
    ground.insert(
        "model.embed_tokens.weight".to_string(),
        wgt(next_seed(), vocab * h),
    );
    for li in 0..layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        ground.insert(p("input_layernorm.weight"), gamma(next_seed(), h));
        ground.insert(p("post_attention_layernorm.weight"), gamma(next_seed(), h));
        ground.insert(p("self_attn.q_a_proj.weight"), wgt(next_seed(), h * r));
        ground.insert(p("self_attn.q_a_layernorm.weight"), gamma(next_seed(), r));
        ground.insert(
            p("self_attn.q_b_proj.weight"),
            wgt(next_seed(), r * hq * qk_head_dim),
        );
        ground.insert(
            p("self_attn.kv_a_proj_with_mqa.weight"),
            wgt(next_seed(), h * (kv_rank + rope_d)),
        );
        ground.insert(
            p("self_attn.kv_a_layernorm.weight"),
            gamma(next_seed(), kv_rank),
        );
        ground.insert(
            p("self_attn.kv_b_proj.weight"),
            wgt(next_seed(), kv_rank * hq * (nope + vd)),
        );
        ground.insert(p("self_attn.o_proj.weight"), wgt(next_seed(), hq * vd * h));
        if li < first_k_dense_replace {
            ground.insert(p("mlp.gate_proj.weight"), wgt(next_seed(), h * dense_inter));
            ground.insert(p("mlp.up_proj.weight"), wgt(next_seed(), h * dense_inter));
            ground.insert(p("mlp.down_proj.weight"), wgt(next_seed(), dense_inter * h));
        } else {
            ground.insert(p("mlp.gate.weight"), wgt(next_seed(), h * n_routed_experts));
            ground.insert(
                p("mlp.gate.e_score_correction_bias"),
                bias_vec(next_seed(), n_routed_experts),
            );
            let mut gate_up = vec![0.0f32; n_routed_experts * h * 2 * moe_inter];
            let mut down = vec![0.0f32; n_routed_experts * moe_inter * h];
            for e in 0..n_routed_experts {
                let ge = wgt(next_seed(), h * moe_inter);
                let ue = wgt(next_seed(), h * moe_inter);
                let de = wgt(next_seed(), moe_inter * h);
                for row in 0..h {
                    let dst = (e * h + row) * 2 * moe_inter;
                    gate_up[dst..dst + moe_inter]
                        .copy_from_slice(&ge[row * moe_inter..(row + 1) * moe_inter]);
                    gate_up[dst + moe_inter..dst + 2 * moe_inter]
                        .copy_from_slice(&ue[row * moe_inter..(row + 1) * moe_inter]);
                }
                let dst = e * moe_inter * h;
                down[dst..dst + moe_inter * h].copy_from_slice(&de);
            }
            ground.insert(p("mlp.experts.gate_up_proj.weight"), gate_up);
            ground.insert(p("mlp.experts.down_proj.weight"), down);
            ground.insert(
                p("mlp.shared_experts.gate_proj.weight"),
                wgt(next_seed(), h * shared_inter),
            );
            ground.insert(
                p("mlp.shared_experts.up_proj.weight"),
                wgt(next_seed(), h * shared_inter),
            );
            ground.insert(
                p("mlp.shared_experts.down_proj.weight"),
                wgt(next_seed(), shared_inter * h),
            );
        }
    }
    ground.insert("model.norm.weight".to_string(), gamma(next_seed(), h));
    ground.insert("lm_head.weight".to_string(), wgt(next_seed(), h * vocab));

    let (cos, sin) = poot_models::deepseek2::deepseek2_rope_tables_interleaved(
        max_pos, rope_d, rope_theta, None,
    );

    // Hand-transform the ground truth into HF safetensors wire layout (row-major `[out,in]` for every
    // matmul weight; the inverse of `transpose2d`/`fuse_qwen3_moe_experts`).
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    tensors.push((
        "model.embed_tokens.weight".to_string(),
        vec![vocab, h],
        ground["model.embed_tokens.weight"].clone(),
    ));
    for li in 0..layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        tensors.push((
            p("input_layernorm.weight"),
            vec![h],
            ground[&p("input_layernorm.weight")].clone(),
        ));
        tensors.push((
            p("post_attention_layernorm.weight"),
            vec![h],
            ground[&p("post_attention_layernorm.weight")].clone(),
        ));
        tensors.push((
            p("self_attn.q_a_proj.weight"),
            vec![r, h],
            transpose_flat(&ground[&p("self_attn.q_a_proj.weight")], h, r),
        ));
        tensors.push((
            p("self_attn.q_a_layernorm.weight"),
            vec![r],
            ground[&p("self_attn.q_a_layernorm.weight")].clone(),
        ));
        tensors.push((
            p("self_attn.q_b_proj.weight"),
            vec![hq * qk_head_dim, r],
            transpose_flat(
                &ground[&p("self_attn.q_b_proj.weight")],
                r,
                hq * qk_head_dim,
            ),
        ));
        tensors.push((
            p("self_attn.kv_a_proj_with_mqa.weight"),
            vec![kv_rank + rope_d, h],
            transpose_flat(
                &ground[&p("self_attn.kv_a_proj_with_mqa.weight")],
                h,
                kv_rank + rope_d,
            ),
        ));
        tensors.push((
            p("self_attn.kv_a_layernorm.weight"),
            vec![kv_rank],
            ground[&p("self_attn.kv_a_layernorm.weight")].clone(),
        ));
        // HF safetensors stores the combined kv_b_proj directly (GGUF splits it into attn_k_b/attn_v_b for
        // llama.cpp's weight-absorption decode), so this is the generic "proj.weight" transpose with no
        // reconstruct step.
        tensors.push((
            p("self_attn.kv_b_proj.weight"),
            vec![hq * (nope + vd), kv_rank],
            transpose_flat(
                &ground[&p("self_attn.kv_b_proj.weight")],
                kv_rank,
                hq * (nope + vd),
            ),
        ));
        tensors.push((
            p("self_attn.o_proj.weight"),
            vec![h, hq * vd],
            transpose_flat(&ground[&p("self_attn.o_proj.weight")], hq * vd, h),
        ));
        if li < first_k_dense_replace {
            tensors.push((
                p("mlp.gate_proj.weight"),
                vec![dense_inter, h],
                transpose_flat(&ground[&p("mlp.gate_proj.weight")], h, dense_inter),
            ));
            tensors.push((
                p("mlp.up_proj.weight"),
                vec![dense_inter, h],
                transpose_flat(&ground[&p("mlp.up_proj.weight")], h, dense_inter),
            ));
            tensors.push((
                p("mlp.down_proj.weight"),
                vec![h, dense_inter],
                transpose_flat(&ground[&p("mlp.down_proj.weight")], dense_inter, h),
            ));
        } else {
            tensors.push((
                p("mlp.gate.weight"),
                vec![n_routed_experts, h],
                transpose_flat(&ground[&p("mlp.gate.weight")], h, n_routed_experts),
            ));
            tensors.push((
                p("mlp.gate.e_score_correction_bias"),
                vec![n_routed_experts],
                ground[&p("mlp.gate.e_score_correction_bias")].clone(),
            ));
            let gate_up = &ground[&p("mlp.experts.gate_up_proj.weight")];
            let down = &ground[&p("mlp.experts.down_proj.weight")];
            for e in 0..n_routed_experts {
                let mut ge = vec![0.0f32; h * moe_inter];
                let mut ue = vec![0.0f32; h * moe_inter];
                for row in 0..h {
                    let src = (e * h + row) * 2 * moe_inter;
                    ge[row * moe_inter..(row + 1) * moe_inter]
                        .copy_from_slice(&gate_up[src..src + moe_inter]);
                    ue[row * moe_inter..(row + 1) * moe_inter]
                        .copy_from_slice(&gate_up[src + moe_inter..src + 2 * moe_inter]);
                }
                let de = &down[e * moe_inter * h..(e + 1) * moe_inter * h];
                let ep = |s: &str| format!("model.layers.{li}.mlp.experts.{e}.{s}");
                tensors.push((
                    ep("gate_proj.weight"),
                    vec![moe_inter, h],
                    transpose_flat(&ge, h, moe_inter),
                ));
                tensors.push((
                    ep("up_proj.weight"),
                    vec![moe_inter, h],
                    transpose_flat(&ue, h, moe_inter),
                ));
                tensors.push((
                    ep("down_proj.weight"),
                    vec![h, moe_inter],
                    transpose_flat(de, moe_inter, h),
                ));
            }
            tensors.push((
                p("mlp.shared_experts.gate_proj.weight"),
                vec![shared_inter, h],
                transpose_flat(
                    &ground[&p("mlp.shared_experts.gate_proj.weight")],
                    h,
                    shared_inter,
                ),
            ));
            tensors.push((
                p("mlp.shared_experts.up_proj.weight"),
                vec![shared_inter, h],
                transpose_flat(
                    &ground[&p("mlp.shared_experts.up_proj.weight")],
                    h,
                    shared_inter,
                ),
            ));
            tensors.push((
                p("mlp.shared_experts.down_proj.weight"),
                vec![h, shared_inter],
                transpose_flat(
                    &ground[&p("mlp.shared_experts.down_proj.weight")],
                    shared_inter,
                    h,
                ),
            ));
        }
    }
    tensors.push((
        "model.norm.weight".to_string(),
        vec![h],
        ground["model.norm.weight"].clone(),
    ));
    // The real V3 config sets `tie_word_embeddings: false` (checked against a real config.json): a
    // separate lm_head.weight, untied from the embedding.
    tensors.push((
        "lm_head.weight".to_string(),
        vec![vocab, h],
        transpose_flat(&ground["lm_head.weight"], h, vocab),
    ));

    let dir = poot_test_util::unique_temp_path("poot_deepseek3_safetensors_loader_fixture");
    std::fs::create_dir_all(&dir).expect("create fixture dir");

    let config = serde_json::json!({
        "model_type": "deepseek_v3",
        "vocab_size": vocab,
        "hidden_size": h,
        "intermediate_size": dense_inter,
        "num_hidden_layers": layers,
        "num_attention_heads": hq,
        "num_key_value_heads": hq,
        "rms_norm_eps": eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_pos,
        "tie_word_embeddings": false,
        "q_lora_rank": r,
        "kv_lora_rank": kv_rank,
        "qk_nope_head_dim": nope,
        "qk_rope_head_dim": rope_d,
        "v_head_dim": vd,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": moe_inter,
        "first_k_dense_replace": first_k_dense_replace,
        "routed_scaling_factor": routed_scaling_factor,
        "n_group": n_group,
        "topk_group": topk_group,
        "bos_token_id": 0,
        "eos_token_id": 0,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .expect("write config.json");
    write_safetensors(&dir.join("model.safetensors"), &tensors);

    // A minimal `tokenizers::Tokenizer` (WordLevel + whitespace splitting), as in bloom_load.rs's/
    // mpt_load.rs's synthetic fixtures.
    let vocab_map: std::collections::HashMap<String, u32> =
        (0..vocab).map(|i| (format!("t{i}"), i as u32)).collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab_map)
        .unk_token("t0".to_string())
        .build()
        .expect("build wordlevel model");
    let mut tok = tokenizers::Tokenizer::new(model);
    tok.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tok.save(dir.join("tokenizer.json"), false)
        .expect("save tokenizer.json");

    let runner = Runner::load(&dir).expect("load synthetic deepseek3 safetensors checkpoint");
    assert_eq!(runner.arch, "deepseek_v3");
    assert!(
        runner.deepseek2.is_none(),
        "a V3 checkpoint must not also set deepseek2"
    );
    let dp = runner
        .deepseek3
        .expect("Runner::load must set deepseek3 params for a deepseek_v3 checkpoint");
    assert_eq!(dp.moe.n_routed_experts, n_routed_experts);
    assert_eq!(dp.moe.top_k, top_k);
    assert_eq!(dp.moe.n_group, n_group);
    assert_eq!(dp.moe.topk_group, topk_group);
    assert_eq!(dp.moe.first_k_dense_replace, first_k_dense_replace);
    assert_eq!(dp.moe.n_shared_experts, n_shared_experts);
    assert!((dp.moe.routed_scaling_factor - routed_scaling_factor).abs() < 1e-6);
    assert_eq!(dp.cfg.q_lora_rank, Some(r));

    let tokens = [0u32, 2, 4, 1];
    let want = deepseek3_prefill_ref(&cfg, &mp, &tokens, &ground, &cos, &sin);

    let g = trace_deepseek3_prefill(dp.cfg, dp.moe, tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind deepseek3 prefill graph");
    let got = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");

    assert_eq!(got.shape(), vec![1, 1, vocab]);
    assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // Safetensors-loaded-and-traced logits (actual) vs the hand-rolled reference (expected): a divergence
    // here is a real transpose/fuse bug in the new deepseek3 safetensors loader (runner.rs's
    // is_deepseek3() blocks), not float noise (both paths are f32, no quantization involved).
    poot_test_util::assert_close(got.as_f32().unwrap(), &want, 1e-3);
}

/// The V3 counterpart of [`deepseek2_prefill_ref_from_runner`]: an independent from-scratch reference
/// forward pass reading the production-loaded `runner.weights` map (`Runner::load` -> `build_weights` ->
/// `bind`), not a hand-transposed synthetic `ground` map like [`deepseek3_prefill_ref`]. Only the router
/// differs from the V2 helper (`deepseek3_router_ref`: sigmoid + selection-only correction bias +
/// group-limited top-k, reading `mlp.gate.e_score_correction_bias`, vs V2's softmax `top_k_router_ref`);
/// MLA attention and the dense/shared-expert MLP paths are the same logic.
fn deepseek3_prefill_ref_from_runner(
    cfg: &DeepseekV2Config,
    mp: &DeepseekV3MoeParams,
    tokens: &[u32],
    w: &HashMap<String, Value>,
) -> Vec<f32> {
    let g = |name: &str| -> &[f32] { w[name].as_host().expect("dense weight").as_f32().unwrap() };
    let h = cfg.hidden;
    let hq = cfg.n_heads;
    let (nope, rope_d, vd) = (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim);
    let qk_head_dim = cfg.qk_head_dim();
    let kv_rank = cfg.kv_lora_rank;
    let scale = cfg.attn_scale();
    let l = tokens.len();
    let embed = g("model.embed_tokens.weight");
    let cos = g("rope.cos");
    let sin = g("rope.sin");
    let half = rope_d / 2;

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

        let q_flat: Vec<Vec<f32>> = match cfg.q_lora_rank {
            Some(r) => {
                let wqa = g(&p("self_attn.q_a_proj.weight"));
                let wqaln = g(&p("self_attn.q_a_layernorm.weight"));
                let wqb = g(&p("self_attn.q_b_proj.weight"));
                normed
                    .iter()
                    .map(|row| {
                        let qa = linear_ref(row, wqa, h, r);
                        let qa_n = rmsnorm_ref(&qa, wqaln, r, cfg.eps);
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
            let c_kv = rmsnorm_ref(kv_nope_raw, wkvaln, kv_rank, cfg.eps);
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
            let normed2 = rmsnorm_ref(row, ln2w, h, cfg.eps);
            let ffn_out = if mp.is_moe_layer(li) {
                let router_w = g(&p("mlp.gate.weight"));
                let bias = g(&p("mlp.gate.e_score_correction_bias"));
                let gate_up = g(&p("mlp.experts.gate_up_proj.weight"));
                let down = g(&p("mlp.experts.down_proj.weight"));
                let logits = linear_ref(&normed2, router_w, h, mp.n_routed_experts);
                let selected = deepseek3_router_ref(
                    &logits,
                    bias,
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
    let last = rmsnorm_ref(&x[l - 1], ln_f_w, h, cfg.eps);
    let lm_head = g("lm_head.weight");
    linear_ref(&last, lm_head, h, cfg.vocab)
}

/// Proves `Runner::load`'s DeepSeek-V3 support (MLA weight-layout crosswalk shared with V2, plus the
/// V3 router transpose + `fuse_qwen3_moe_experts` + `mlp.gate.e_score_correction_bias` pass-through) on
/// a real checkpoint, not just a hand-transposed synthetic fixture (unlike
/// [`deepseek3_synthetic_safetensors_matches_hand_rolled_reference`] above, whose doc still says there
/// is no safetensors V3 loader to cross-check against): the traced-and-bound `trace_deepseek3_prefill`
/// graph, evaluated through the production `Runner::load` + `build_weights` + `bind` path, must match an
/// independent from-scratch Rust reference (`deepseek3_prefill_ref_from_runner`) reading the same weight
/// map. A transpose/fuse/MLA-decompress/router bug would give a materially different output.
///
/// The fixture (`yujiepan/deepseek-v3-tiny-random`) is a real `model_type: "deepseek_v3"` HF config with
/// `n_group: 8` / `topk_group: 4` over 256 routed experts (group_size 32): group-limited routing,
/// not the `n_group: 1` no-op. Needs `deepseek3-tiny` under POOT_MODELS_DIR (`hf download
/// yujiepan/deepseek-v3-tiny-random --local-dir <POOT_MODELS_DIR>/deepseek3-tiny` to populate it).
#[test]
fn deepseek3_tiny_checkpoint_runner_load_matches_hand_rolled_reference() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek3-tiny"))
    else {
        return;
    };
    let mut runner = Runner::load(&dir).expect("load deepseek3-tiny via Runner");
    assert_eq!(runner.arch, "deepseek_v3");
    assert!(
        runner.deepseek2.is_none(),
        "a V3-style checkpoint must not also set deepseek2"
    );
    // `yujiepan/deepseek-v3-tiny-random`'s published `mlp.gate.e_score_correction_bias` is all-NaN
    // (checked in the raw bf16 bytes): a quirk of this tiny-random conversion, not a poot bug. Real training
    // populates it via aux-loss-free load-balancing statistics, which a random-init fixture lacks. Zero it
    // in place ("no selection bias adjustment") so the production loader can be compared against a
    // plain-IEEE-754 reference: `deepseek3_router_ref`'s `partial_cmp().unwrap()` panics on NaN, and poot's
    // graph-side topk silently selects differently with a NaN bias, making the comparison meaningless.
    {
        let bias_key = "model.layers.1.mlp.gate.e_score_correction_bias";
        let bias = runner
            .weights
            .get(bias_key)
            .unwrap_or_else(|| panic!("missing {bias_key}"));
        assert!(
            bias.as_host()
                .expect("dense weight")
                .as_f32()
                .unwrap()
                .iter()
                .all(|v| v.is_nan()),
            "checkpoint's e_score_correction_bias is no longer all-NaN - re-check whether the \
                 upstream fixture changed and this workaround is still needed"
        );
        let zeroed = HostTensor::f32(
            bias.as_host().expect("dense weight").shape().to_vec(),
            vec![
                0.0f32;
                bias.as_host()
                    .expect("dense weight")
                    .as_f32()
                    .unwrap()
                    .len()
            ],
        );
        runner
            .weights
            .insert(bias_key.to_string(), poot_eval::Value::from(zeroed));
    }
    let dp = runner
        .deepseek3
        .expect("Runner::load must set deepseek3 params for a deepseek_v3 checkpoint");
    assert_eq!(dp.cfg.q_lora_rank, Some(16));
    assert_eq!(dp.cfg.kv_lora_rank, 16);
    assert_eq!(dp.cfg.qk_nope_head_dim, 16);
    assert_eq!(dp.cfg.qk_rope_head_dim, 16);
    assert_eq!(dp.cfg.v_head_dim, 16);
    assert_eq!(dp.moe.n_routed_experts, 256);
    assert_eq!(dp.moe.top_k, 8);
    assert_eq!(dp.moe.n_shared_experts, 1);
    assert_eq!(dp.moe.first_k_dense_replace, 1);
    assert_eq!(
        dp.moe.n_group, 8,
        "real group-limited routing, not the n_group=1 no-op case"
    );
    assert_eq!(dp.moe.topk_group, 4);
    assert!((dp.moe.routed_scaling_factor - 2.5).abs() < 1e-6);

    let tokens = [1u32, 2, 3, 4];
    let g = trace_deepseek3_prefill(dp.cfg, dp.moe, tokens.len());
    let inputs = runner
        .bind(&g, &tokens)
        .expect("bind deepseek3 prefill graph");
    let out = crate::core::cpu_oracle::cpu_eval(&g, &inputs).expect("cpu eval");
    assert_eq!(out.shape(), vec![1, 1, dp.cfg.vocab]);
    assert!(out.as_f32().unwrap().iter().all(|v| v.is_finite()));

    let want = deepseek3_prefill_ref_from_runner(&dp.cfg, &dp.moe, &tokens, &runner.weights);
    assert_eq!(out.as_f32().unwrap().len(), want.len());
    // Runner-loaded output (actual) vs the hand-rolled reference (expected).
    poot_test_util::assert_close_rel(out.as_f32().unwrap(), &want, 1e-4);
}

/// The V3 counterpart of [`deepseek2_tiny_gguf_matches_safetensors`]: executor equivalence between
/// `Runner::load` (safetensors) and `Runner::load_gguf` on the same real `yujiepan/deepseek-v3-tiny-random`
/// checkpoint, converted with a real `llama.cpp convert_hf_to_gguf.py --outtype f32` run (confirming
/// `conversion/deepseek.py`'s `DeepseekV2Model.register("DeepseekV2ForCausalLM",
/// "DeepseekV3ForCausalLM")`: a `model_type: "deepseek_v3"` config takes the V2 conversion path and the
/// shared "deepseek2" GGUF arch string, as docs/updates/0791 found from the source). Both paths are f32
/// with bit-identical weights; a transpose/reconstruction/router bug in either loader would give a
/// materially different forward pass.
#[test]
fn deepseek3_tiny_gguf_matches_safetensors() {
    let Some(st_dir) = poot_test_util::model_path(poot_test_util::checkpoint!("deepseek3-tiny"))
    else {
        return;
    };
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek3-tiny/deepseek3-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut st_runner = Runner::load(&st_dir).expect("load deepseek3-tiny safetensors");
    let mut gguf_runner = Runner::load_gguf(&gguf_path).expect("load deepseek3-tiny gguf");
    assert_eq!(st_runner.arch, "deepseek_v3");
    assert_eq!(gguf_runner.arch, "deepseek_v3");

    let st_dp = st_runner
        .deepseek3
        .expect("safetensors Runner must set deepseek3 params");
    let gguf_dp = gguf_runner
        .deepseek3
        .expect("gguf Runner must set deepseek3 params");
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
    assert_eq!(st_dp.moe.n_group, gguf_dp.moe.n_group);
    assert_eq!(st_dp.moe.topk_group, gguf_dp.moe.topk_group);
    assert_eq!(
        st_dp.moe.routed_scaling_factor,
        gguf_dp.moe.routed_scaling_factor
    );

    // Same NaN e_score_correction_bias workaround as
    // `deepseek3_tiny_checkpoint_runner_load_matches_hand_rolled_reference` above; the NaN is in the
    // original safetensors bf16 bytes, so the f32 GGUF inherits it. Zero it on both runners so the test
    // compares the two loaders' MLA/MoE weight handling, not NaN-propagation quirks.
    for runner in [&mut st_runner, &mut gguf_runner] {
        let key = "model.layers.1.mlp.gate.e_score_correction_bias";
        let bias = runner
            .weights
            .get(key)
            .unwrap_or_else(|| panic!("missing {key}"));
        let zeroed = HostTensor::f32(
            bias.as_host().expect("dense weight").shape().to_vec(),
            vec![
                0.0f32;
                bias.as_host()
                    .expect("dense weight")
                    .as_f32()
                    .unwrap()
                    .len()
            ],
        );
        runner
            .weights
            .insert(key.to_string(), poot_eval::Value::from(zeroed));
    }

    let tokens = [1u32, 2, 3, 4];
    let st_g = trace_deepseek3_prefill(st_dp.cfg, st_dp.moe, tokens.len());
    let st_inputs = st_runner
        .bind(&st_g, &tokens)
        .expect("bind safetensors prefill graph");
    let st_out =
        crate::core::cpu_oracle::cpu_eval(&st_g, &st_inputs).expect("cpu eval (safetensors)");

    let gguf_g = trace_deepseek3_prefill(gguf_dp.cfg, gguf_dp.moe, tokens.len());
    let gguf_inputs = gguf_runner
        .bind(&gguf_g, &tokens)
        .expect("bind gguf prefill graph");
    let gguf_out =
        crate::core::cpu_oracle::cpu_eval(&gguf_g, &gguf_inputs).expect("cpu eval (gguf)");

    assert_eq!(st_out.shape(), gguf_out.shape());
    assert!(st_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert!(gguf_out.as_f32().unwrap().iter().all(|v| v.is_finite()));
    // Safetensors-loaded vs gguf-loaded logits on the real deepseek3-tiny checkpoint: both paths are f32
    // with no quantization, so this should be tight; a divergence is a transpose/reconstruction bug in one
    // of the two loaders.
    poot_test_util::assert_close(st_out.as_f32().unwrap(), gguf_out.as_f32().unwrap(), 1e-4);
}

/// GPU dispatch receipt for DeepSeek-V3: updates 0793/0797 verified `generate_ptx_reprefill`'s
/// `self.deepseek3` arm on a synthesized fixture only. This checks it on the real
/// `yujiepan/deepseek-v3-tiny-random` checkpoint (its GGUF conversion is loaded by
/// `deepseek3_tiny_gguf_matches_safetensors` above) on a real NVIDIA GPU. It runs the CPU side (load,
/// tokenizer encode, `generate()`) before creating the PTX executor, unlike the synthetic PTX test: this
/// checkpoint's real tokenizer/BPE vocab and `generate()` text path were never exercised before (earlier
/// real-checkpoint tests bound raw token ids), so a local run without a GPU still validates that path.
///
/// Same NaN `e_score_correction_bias` workaround as
/// `deepseek3_tiny_checkpoint_runner_load_matches_hand_rolled_reference` above (a quirk of the published
/// fixture): without it the two backends' topk kernels could pick different but individually valid
/// experts from an undefined NaN ordering, turning a token mismatch into noise.
///
/// Skips if the checkpoint's GGUF conversion is absent or no PTX/NVIDIA GPU is available.
#[test]
#[ignore = "needs the real deepseek3-tiny checkpoint's GGUF conversion at deepseek3-tiny/\
                deepseek3-tiny-f32.gguf under POOT_MODELS_DIR (see docs/updates/0800) and a real rented NVIDIA GPU; run with \
                --ignored --release"]
fn deepseek3_tiny_checkpoint_ptx_reprefill_matches_cpu() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek3-tiny/deepseek3-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut runner = Runner::load_gguf(&gguf_path).expect("load deepseek3-tiny gguf");
    assert_eq!(runner.arch, "deepseek_v3");
    assert!(
        runner.deepseek3.is_some(),
        "Runner::load_gguf must set deepseek3 params for a deepseek_v3 GGUF"
    );
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV3
    );

    // Same NaN e_score_correction_bias workaround as
    // `deepseek3_tiny_checkpoint_runner_load_matches_hand_rolled_reference` above.
    {
        let key = "model.layers.1.mlp.gate.e_score_correction_bias";
        let bias = runner
            .weights
            .get(key)
            .unwrap_or_else(|| panic!("missing {key}"));
        assert!(
            bias.as_host()
                .expect("dense weight")
                .as_f32()
                .unwrap()
                .iter()
                .all(|v| v.is_nan()),
            "checkpoint's e_score_correction_bias is no longer all-NaN - re-check whether the \
                 upstream fixture changed and this workaround is still needed"
        );
        let zeroed = HostTensor::f32(
            bias.as_host().expect("dense weight").shape().to_vec(),
            vec![
                0.0f32;
                bias.as_host()
                    .expect("dense weight")
                    .as_f32()
                    .unwrap()
                    .len()
            ],
        );
        runner
            .weights
            .insert(key.to_string(), poot_eval::Value::from(zeroed));
    }

    let prompt = "Hello";
    let max_new = 10;
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek3-tiny CPU generate");

    let mut ptx = match poot_ptx_gpu::PtxDevice::new() {
        Ok(g) => poot_executor::Engine::new(g),
        Err(e) => {
            eprintln!(
                "no PTX GPU ({e}); skipping (CPU-side load/tokenize/generate already ran clean: \
                     {cpu_toks:?})"
            );
            return;
        }
    };
    let ptx_toks = runner
        .generate_ptx_reprefill(prompt, max_new, &mut ptx, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek3-tiny PTX re-prefill generate");
    eprintln!(
        "deepseek3-tiny-real ptx={:?} cpu={:?}",
        runner.decode(&ptx_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        ptx_toks, cpu_toks,
        "GGUF-loaded PTX re-prefill decode for the REAL deepseek3-tiny checkpoint (group-limited MoE \
             routing over 256 experts, n_group=8/topk_group=4, plus MLA attention) must match the \
             GGUF-loaded CPU eager reference"
    );
}

/// ROCm/AMD counterpart of [`deepseek3_tiny_checkpoint_ptx_reprefill_matches_cpu`] above: same real
/// checkpoint, same ordering (CPU load/tokenize/generate before the GPU executor, so a no-GPU run still
/// validates the real-tokenizer `generate()` path), same NaN `e_score_correction_bias` workaround.
/// Verifies `generate_rocm_reprefill`'s `self.deepseek3` arm (`crates/poot-llm/src/rocm_vulkan.rs`, earlier
/// checked only against the synthetic fixture in `deepseek3_synthetic_rocm_reprefill.rs`) against a real
/// MI300X.
#[cfg(feature = "rocm")]
#[test]
#[ignore = "needs the real deepseek3-tiny checkpoint's GGUF conversion at deepseek3-tiny/\
                deepseek3-tiny-f32.gguf under POOT_MODELS_DIR (see docs/updates/0800) and a real rented ROCm/AMD GPU; run with \
                --features rocm --ignored --test-threads=1 --release"]
fn deepseek3_tiny_checkpoint_rocm_reprefill_matches_cpu() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek3-tiny/deepseek3-tiny-f32.gguf"
    )) else {
        return;
    };
    let mut runner = Runner::load_gguf(&gguf_path).expect("load deepseek3-tiny gguf");
    assert_eq!(runner.arch, "deepseek_v3");
    assert!(
        runner.deepseek3.is_some(),
        "Runner::load_gguf must set deepseek3 params for a deepseek_v3 GGUF"
    );
    assert_eq!(
        runner.decode_arch().unwrap(),
        crate::core::decode_arch::DecodeArch::DeepseekV3
    );

    {
        let key = "model.layers.1.mlp.gate.e_score_correction_bias";
        let bias = runner
            .weights
            .get(key)
            .unwrap_or_else(|| panic!("missing {key}"));
        assert!(
            bias.as_host()
                .expect("dense weight")
                .as_f32()
                .unwrap()
                .iter()
                .all(|v| v.is_nan()),
            "checkpoint's e_score_correction_bias is no longer all-NaN - re-check whether the \
                 upstream fixture changed and this workaround is still needed"
        );
        let zeroed = HostTensor::f32(
            bias.as_host().expect("dense weight").shape().to_vec(),
            vec![
                0.0f32;
                bias.as_host()
                    .expect("dense weight")
                    .as_f32()
                    .unwrap()
                    .len()
            ],
        );
        runner
            .weights
            .insert(key.to_string(), poot_eval::Value::from(zeroed));
    }

    let prompt = "Hello";
    let max_new = 10;
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek3-tiny CPU generate");

    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "no ROCm GPU ({e}); skipping (CPU-side load/tokenize/generate already ran clean: \
                     {cpu_toks:?})"
            );
            return;
        }
    };
    let mut rocm = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut rocm).expect("load_on");
    let rocm_toks = runner
        .generate_rocm_reprefill(prompt, max_new, &mut rocm, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("deepseek3-tiny ROCm re-prefill generate");
    eprintln!(
        "deepseek3-tiny-real rocm={:?} cpu={:?}",
        runner.decode(&rocm_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        rocm_toks, cpu_toks,
        "GGUF-loaded ROCm re-prefill decode for the REAL deepseek3-tiny checkpoint (group-limited MoE \
             routing over 256 experts, n_group=8/topk_group=4, plus MLA attention) must match the \
             GGUF-loaded CPU eager reference"
    );
}

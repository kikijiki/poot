//! Gemma2 and llama-shaped (Mistral) decode/prefill bit-exact reference checks with hand-rolled reference math.

use poot_graph_ir::rope_table::{RopeFlavor, rope_tables};
use poot_tensor::HostTensor;

use super::helpers::*;
use super::swa_gemma3::{named_weights, run_step};
use poot_test_util::{assert_close, max_abs_error};
// --- Shared reference helpers for bit-exact model tests ---

/// Deterministic weight tensor by name (same xorshift seed as the bench `weight()` helper).
/// Scaled by 0.1 to keep activations bounded during small-weight forward passes.
fn named_weight(name: &str, n: usize) -> Vec<f32> {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    fill(n, seed).iter().map(|v| v * 0.1).collect()
}

/// RMSNorm over a 1-D slice `x` with scale `w` and `eps`.
/// Matches poot-eval's rmsnorm: `(x / sqrt(mean(x^2) + eps)) * w`.
fn rmsnorm_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let den = (ms + eps).sqrt();
    x.iter().zip(w).map(|(&xi, &wi)| xi / den * wi).collect()
}

/// RMSNorm with a Gemma checkpoint's scale, stored as an offset from one: `x / rms * (1 + w)`.
fn gemma_rmsnorm_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let w: Vec<f32> = w.iter().map(|v| 1.0 + v).collect();
    rmsnorm_ref(x, &w, eps)
}

/// `x @ w^T` for a checkpoint weight `w` stored `[out, in]` row-major (`k` = in, `n` = out).
fn linear_ref(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|j| (0..k).map(|i| x[i] * w[j * k + i]).sum())
        .collect()
}

/// Softcap: c * tanh(x / c).
fn softcap_ref(v: f32, c: f32) -> f32 {
    c * (v / c).tanh()
}

/// RoPE for a `[heads, d]` flat buffer using pre-selected cos/sin `[d]` at one position.
/// rotate_half: split D -> [0..D/2, D/2..D], rh = concat([-x[D/2..], x[..D/2]]).
/// Matches poot-eval's rope_partial: `out = x * cos + rh * sin`.
fn rope_ref(x: &[f32], cos: &[f32], sin: &[f32], heads: usize, d: usize) -> Vec<f32> {
    let half = d / 2;
    let mut out = vec![0.0f32; heads * d];
    for hi in 0..heads {
        for i in 0..d {
            let xi = x[hi * d + i];
            let rhi = if i < half {
                -x[hi * d + i + half]
            } else {
                x[hi * d + i - half]
            };
            out[hi * d + i] = xi * cos[i] + rhi * sin[i];
        }
    }
    out
}

/// Hand-rolled Gemma2 decode forward pass. Matches the Gemma 2 family's decode step for step; uses the same numeric primitives as poot-eval's eager executor.
///
/// `kv_caches`: one entry per layer, each a `(k_flat, v_flat)` pair with shape `[hkv, cap, d]`
/// (flat row-major). Position `pos` will be WRITTEN by this call (simulating the DUS), so the
/// caller should pass the initial cache state (same as the eval state inputs).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn gemma2_decode_ref(
    vocab_size: usize,
    hidden: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    rms_norm_eps: f32,
    max_pos: usize,
    sliding_window: usize,
    attn_logit_softcap: Option<f32>,
    final_logit_softcap: Option<f32>,
    query_pre_attn_scalar: Option<f32>,
    pos: usize,
    token: usize,
    kv_caches: &mut [(Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let h = hidden;
    let d = head_dim;
    let hq = num_heads;
    let hkv = num_kv_heads;
    let n_rep = hq / hkv;
    let inter = intermediate_size;
    let cap = kv_caches[0].0.len() / (hkv * d);
    let scale = match query_pre_attn_scalar {
        Some(s) => 1.0 / s.sqrt(),
        None => 1.0 / (d as f32).sqrt(),
    };

    // Embedding lookup + scale by sqrt(hidden).
    let embed = named_weight("w.embed", vocab_size * h);
    let embed_scale = (h as f32).sqrt();
    let mut x: Vec<f32> = embed[token * h..(token + 1) * h]
        .iter()
        .map(|v| v * embed_scale)
        .collect();

    // RoPE cos/sin row at position `pos`.
    let tables = rope_tables(d, max_pos, 10_000.0, &RopeFlavor::Plain, None);
    let cos = tables.cos[pos * d..(pos + 1) * d].to_vec();
    let sin = tables.sin[pos * d..(pos + 1) * d].to_vec();

    for li in 0..num_layers {
        let p = |s: &str| format!("w.l{li}.{s}");
        // HF slides the even layers (layer 0 here) and keeps the odd ones global.
        let is_swa = li % 2 == 0;

        // Pre-attention RMSNorm.
        let ln1 = named_weight(&p("norm.attn"), h);
        let normed = gemma_rmsnorm_ref(&x, &ln1, rms_norm_eps);

        // QKV projections (no bias).
        let q_dim = hq * d;
        let kv_dim = hkv * d;
        let wq = named_weight(&p("attn.q"), h * q_dim);
        let wk = named_weight(&p("attn.k"), h * kv_dim);
        let wv = named_weight(&p("attn.v"), h * kv_dim);
        let wo = named_weight(&p("attn.o"), q_dim * h);

        // normed is [h]; treat as [1, h] and matmul -> [1, q_dim] = [q_dim].
        let q = linear_ref(&normed, &wq, h, q_dim);
        let k = linear_ref(&normed, &wk, h, kv_dim);
        let v = linear_ref(&normed, &wv, h, kv_dim);

        // RoPE: q is [hq, d] (flat), k is [hkv, d] (flat).
        let q_rot = rope_ref(&q, &cos, &sin, hq, d);
        let k_rot = rope_ref(&k, &cos, &sin, hkv, d);

        // Write k_rot and v into the KV cache at position `pos`.
        // Cache layout: [hkv, cap, d] flat.
        let (k_cache, v_cache) = &mut kv_caches[li];
        for hi in 0..hkv {
            let start = hi * cap * d + pos * d;
            k_cache[start..start + d].copy_from_slice(&k_rot[hi * d..(hi + 1) * d]);
            v_cache[start..start + d].copy_from_slice(&v[hi * d..(hi + 1) * d]);
        }

        // Determine the attention window (after writing, so the new token is included).
        let seq = pos + 1;
        let (attn_start, _attn_len) = if is_swa && seq > sliding_window {
            (seq - sliding_window, sliding_window)
        } else {
            (0, seq)
        };

        // Attention for each query head.
        let (k_cache, v_cache) = (&kv_caches[li].0, &kv_caches[li].1);
        let mut attn_out = vec![0.0f32; q_dim];
        for qh in 0..hq {
            let kvh = qh / n_rep;
            // Dot q[qh] against each key in the window.
            let mut scores: Vec<f32> = (attn_start..seq)
                .map(|s| {
                    let mut dot = 0.0f32;
                    for di in 0..d {
                        dot += q_rot[qh * d + di] * k_cache[kvh * cap * d + s * d + di];
                    }
                    dot * scale
                })
                .collect();
            // Optional attention-logit softcap.
            if let Some(c) = attn_logit_softcap {
                for s in &mut scores {
                    *s = softcap_ref(*s, c);
                }
            }
            let weights = softmax_ref(&scores);
            for (wi, s) in (attn_start..seq).enumerate() {
                for di in 0..d {
                    attn_out[qh * d + di] += weights[wi] * v_cache[kvh * cap * d + s * d + di];
                }
            }
        }

        // o_proj: [q_dim] -> [h].
        let attn_proj = linear_ref(&attn_out, &wo, q_dim, h);

        // Post-attention RMSNorm (applied to attn output BEFORE the residual add).
        let pan = named_weight(&p("norm.post_attn"), h);
        let attn_normed = gemma_rmsnorm_ref(&attn_proj, &pan, rms_norm_eps);

        for i in 0..h {
            x[i] += attn_normed[i];
        }

        // Pre-feedforward RMSNorm.
        let pre_ff = named_weight(&p("norm.ffn"), h);
        let normed_ff = gemma_rmsnorm_ref(&x, &pre_ff, rms_norm_eps);

        // GeGLU MLP.
        let wg = named_weight(&p("ffn.gate"), h * inter);
        let wu = named_weight(&p("ffn.up"), h * inter);
        let wd = named_weight(&p("ffn.down"), inter * h);
        let gate = linear_ref(&normed_ff, &wg, h, inter);
        let up = linear_ref(&normed_ff, &wu, h, inter);
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| gelu_ref(g) * u)
            .collect();
        let mlp_out = linear_ref(&act, &wd, inter, h);

        // Post-feedforward RMSNorm (applied to MLP output BEFORE the residual add).
        let post_ff = named_weight(&p("norm.post_ffn"), h);
        let mlp_normed = gemma_rmsnorm_ref(&mlp_out, &post_ff, rms_norm_eps);

        for i in 0..h {
            x[i] += mlp_normed[i];
        }
    }

    // Final RMSNorm.
    let norm = named_weight("w.final_norm", h);
    let x = gemma_rmsnorm_ref(&x, &norm, rms_norm_eps);

    // LM head.
    let lm_head = named_weight("w.head", h * vocab_size);
    let logits = linear_ref(&x, &lm_head, h, vocab_size);

    // Final logit softcap.
    match final_logit_softcap {
        Some(c) => logits.iter().map(|&v| softcap_ref(v, c)).collect(),
        None => logits,
    }
}

// --- Gemma2 numeric eval tests ---

use poot_executor_parity::dense::{Dense, Family, step};
use poot_executor_parity::weight_map::MappedModel;
use poot_graph_ir::{Graph, ValidationOutputs};
use poot_models::model::{LogitRows, Phase};

/// The tiny Gemma 2 of these rows: `vocab` 8, hidden 8, two layers (layer 0 slides, layer 1 is global), two
/// full-width heads, head dim 4 (query scalar 4, scale 0.5), softcaps 50 and 30, `window` keys on the sliding layer.
fn gemma2(window: usize, attn: Option<f32>, last: Option<f32>, vocab: usize) -> MappedModel {
    Dense::new(Family::Gemma2)
        .vocab(vocab)
        .dims(8, 16, 2)
        .heads(2, 2)
        .head_dim(4)
        .max_positions(16)
        .with("query_pre_attn_scalar", 4.0)
        .with("sliding_window", window)
        .with("attn_logit_softcapping", attn)
        .with("final_logit_softcapping", last)
        .f32_model()
}

fn decode_graph(m: &MappedModel, cap: usize) -> Graph<ValidationOutputs> {
    m.model
        .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
        .unwrap()
}

fn prefill_graph(m: &MappedModel, n: usize, cap: usize) -> Graph<ValidationOutputs> {
    m.model
        .trace(Phase::Prefill, step(1, n, cap, LogitRows::Last))
        .unwrap()
}

/// Per-state-input values: `fill(name)` of the state's index, scaled like the weights.
fn filled_state(g: &Graph<ValidationOutputs>, tag: &str) -> Vec<HostTensor> {
    g.state
        .iter()
        .enumerate()
        .map(|(i, &(input, _))| {
            let shape = g.aval(input).shape.clone();
            HostTensor::f32(shape.clone(), named_weights(&format!("{tag}.{i}"), &shape))
        })
        .collect()
}

/// Gemma2 decode output is finite (no NaN/Inf) for deterministic small-weight inputs.
#[test]
fn gemma2_decode_output_is_finite() {
    let m = gemma2(4, Some(5.0), Some(3.0), 16);
    let g = decode_graph(&m, 8);
    let (logits, _) = run_step(&g, &[5], 3, &named_weights, &[]);
    assert_eq!(logits.shape(), vec![1, 1, 16]);
    for &v in logits.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "logit {v} is not finite");
    }
}

/// Softcap changes output: a binding final_logit_softcap (0.005, below the ~0.05 typical logit
/// magnitude with these weights) produces clearly different logits than None.
#[test]
fn gemma2_softcap_changes_output() {
    let weights = |name: &str, shape: &[usize]| -> Vec<f32> {
        named_weights(name, shape).iter().map(|v| v * 2.0).collect()
    };
    // A cap of 0.005 is well below the typical ~0.05 logit magnitude, so it clips visibly.
    let logits = |last: Option<f32>| {
        let g = decode_graph(&gemma2(4, None, last, 16), 8);
        run_step(&g, &[3], 1, &weights, &[]).0
    };
    let (out_base, out_capped) = (logits(None), logits(Some(0.005)));

    // The capped logits must be bounded by the cap value (|logit| <= cap + eps).
    for &v in out_capped.as_f32().unwrap().iter() {
        assert!(
            v.abs() <= 0.005 + 1e-5,
            "capped logit {v} exceeds cap 0.005"
        );
    }
    // The base logits must differ from the capped logits (cap was binding).
    let max_diff = max_abs_error(out_base.as_f32().unwrap(), out_capped.as_f32().unwrap());
    assert!(
        max_diff > 1e-4,
        "softcap should change logits; max_diff={max_diff:.2e}"
    );
}

/// SWA layers constrain the key window: with a small window and pos beyond the window,
/// the SWA config (window=2) produces different output than a wide-window config (window=32).
#[test]
fn gemma2_swa_affects_output_when_beyond_window() {
    // window=2: the sliding layer only sees the last 2 keys at pos=5. window=32: it sees all keys.
    // pos=5 puts us well beyond the narrow window.
    let (pos, cap) = (5, 16);
    let logits = |window: usize| {
        let g = decode_graph(&gemma2(window, None, None, 16), cap);
        g.validate().expect("graph valid");
        // Non-trivial cached keys, so keys beyond the window differ.
        let state = filled_state(&g, "state");
        run_step(&g, &[7], pos, &named_weights, &state).0
    };
    let (out_narrow, out_wide) = (logits(2), logits(32));

    for &v in out_narrow.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "narrow-window logit {v} not finite");
    }
    for &v in out_wide.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "wide-window logit {v} not finite");
    }

    // With non-trivial cached keys, narrow and wide windows must produce different logits.
    let max_diff = max_abs_error(out_narrow.as_f32().unwrap(), out_wide.as_f32().unwrap());
    assert!(
        max_diff > 1e-6,
        "narrow vs wide window should produce different logits; max_diff={max_diff:.2e}"
    );
}

/// Gemma2 decode: comparison against an independent hand-rolled scalar reference.
/// Config exercises all Gemma2-specific features: embedding scale, 4-norm sandwich, GeGLU MLP,
/// alternating global/SWA attention, attention-logit softcap, and final-logit softcap.
/// pos=4 with window=3 pushes the SWA layer into windowed attention (seq=5 > window=3).
#[test]
fn gemma2_decode_bitexact_vs_reference() {
    let (vocab, hidden, layers, heads, head_dim, inter, max_pos) = (8, 8, 2, 2, 4, 16, 16);
    let (window, attn_cap, final_cap) = (3, Some(50.0), Some(30.0));
    let (pos, cap) = (4usize, 8usize);
    let token = 5usize;
    let hkv = heads;
    let d = head_dim;

    // Pre-populate KV caches with deterministic values (simulating prior decode steps).
    // The decode step will overwrite position `pos`; positions 0..pos-1 hold meaningful data.
    let make_initial_caches = || -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..layers)
            .map(|li| {
                // Large cached keys make the attention peaked, so a key the window drops (or keeps) moves the logits
                // well above the tolerance below.
                let k = named_weight(&format!("init.k.{li}"), hkv * cap * d)
                    .iter()
                    .map(|v| v * 100.0)
                    .collect();
                let v = named_weight(&format!("init.v.{li}"), hkv * cap * d);
                (k, v)
            })
            .collect()
    };

    // Run the hand-rolled reference (modifies caches in place).
    let mut kv_caches_ref = make_initial_caches();
    let ref_logits = gemma2_decode_ref(
        vocab,
        hidden,
        layers,
        heads,
        hkv,
        head_dim,
        inter,
        1e-6,
        max_pos,
        window,
        attn_cap,
        final_cap,
        None, // scale = 1/sqrt(head_dim) = 0.5, the family's query_pre_attn_scalar 4
        pos,
        token,
        &mut kv_caches_ref,
    );

    // Run the traced graph using the SAME initial cache data, bound in state order (k0, v0, k1, v1, ...).
    let g = decode_graph(&gemma2(window, attn_cap, final_cap, vocab), cap);
    let kv_init = make_initial_caches();
    let state: Vec<HostTensor> = g
        .state
        .iter()
        .enumerate()
        .map(|(ci, &(si, _))| {
            let (k_flat, v_flat) = &kv_init[ci / 2];
            let data = if ci % 2 == 0 { k_flat } else { v_flat };
            HostTensor::f32(g.aval(si).shape.clone(), data.clone())
        })
        .collect();
    let (eval_logits, _) = run_step(&g, &[token as i32], pos, &named_weights, &state);
    assert_eq!(eval_logits.shape(), vec![1, 1, vocab]);
    assert_eq!(ref_logits.len(), vocab);

    let max_abs = max_abs_error(eval_logits.as_f32().unwrap(), &ref_logits);

    eprintln!("gemma2_decode_bitexact max_abs = {max_abs:.2e}");
    assert!(
        max_abs <= 1e-4,
        "Gemma2 decode reference mismatch: max_abs={max_abs:.2e} (expected <= 1e-4)"
    );
}

/// Gemma2 prefill eval: output is finite and the graph executes through the oracle.
#[test]
fn gemma2_prefill_eval_is_finite() {
    let (n, cap) = (5, 8);
    let g = prefill_graph(&gemma2(4, Some(5.0), Some(3.0), 16), n, cap);
    let tokens = [5, 9, 2, 14, 7];
    let (logits, _) = run_step(&g, &tokens, 0, &named_weights, &[]);
    assert_eq!(logits.shape(), vec![1, 1, 16]);
    for &v in logits.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "prefill logit {v} is not finite");
    }
}

/// Gemma2 prefill/decode consistency: prefill on n tokens returns the same last-position logits as
/// stepping decode through pos=n-1 with the same weights and tokens.
#[test]
fn gemma2_prefill_matches_decode_last_position() {
    let tokens = [3, 7, 1, 5]; // n=4 tokens
    let n = tokens.len();
    let cap = n;
    let m = gemma2(3, Some(50.0), Some(30.0), 8);

    // --- Decode path: run pos=0..n-1, carry caches, take last logits ---
    let g = decode_graph(&m, cap);
    let mut decode_caches: Vec<HostTensor> = Vec::new();
    let mut decode_logits: Option<HostTensor> = None;
    for (pos, &tok) in tokens.iter().enumerate() {
        let (logits, caches) = run_step(&g, &[tok], pos, &named_weights, &decode_caches);
        decode_caches = caches;
        decode_logits = Some(logits);
    }
    let decode_logits = decode_logits.expect("at least one decode step");

    // --- Prefill path: prefill n tokens, get last-position logits ---
    let g = prefill_graph(&m, n, cap);
    let (prefill_logits, prefill_caches) = run_step(&g, &tokens, 0, &named_weights, &[]);
    assert_eq!(prefill_logits.shape(), vec![1, 1, 8]);

    // The prefill last-position logits must match the decode logits at pos=n-1.
    let max_abs = max_abs_error(
        decode_logits.as_f32().unwrap(),
        prefill_logits.as_f32().unwrap(),
    );

    eprintln!("gemma2_prefill_matches_decode max_abs = {max_abs:.2e}");
    assert!(
        max_abs <= 5e-3,
        "Gemma2 prefill/decode consistency mismatch: max_abs={max_abs:.2e}"
    );

    // The carried state matters as much as the logits: the K/V caches prefill leaves must be the ones n decode
    // steps leave (R486-007), or later decode steps would attend to different keys and values.
    assert_eq!(prefill_caches.len(), decode_caches.len());
    for (i, (prefill_cache, decode_cache)) in prefill_caches.iter().zip(&decode_caches).enumerate()
    {
        assert_eq!(
            prefill_cache.shape(),
            decode_cache.shape(),
            "cache {i} shape"
        );
        assert_close(
            prefill_cache.as_f32().unwrap(),
            decode_cache.as_f32().unwrap(),
            5e-3,
        );
    }
}

/// Hand-rolled llama-shaped (Mistral/Qwen2 without bias) decode reference of
/// the llama family's decode step, which:
/// - Attends the keys `0..=pos` (full causal, no sliding window configured).
/// - Uses 2 norms per layer: `input_layernorm` before attention, `post_attention_layernorm`
///   applied to `x + attn_proj` (the residual sum) before the MLP.
/// - No QKV bias, no QK-norm, SwiGLU MLP.
/// - No embedding scale (unlike Gemma2).
///
/// This mirrors the graph exactly so the comparison is bit-exact.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn llama_decode_ref(
    vocab: usize,
    hidden: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    max_pos: usize,
    pos: usize,
    token: usize,
    kv_caches: &mut [(Vec<f32>, Vec<f32>)],
) -> Vec<f32> {
    let h = hidden;
    let d = head_dim;
    let hq = num_heads;
    let hkv = num_kv_heads;
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let cap = kv_caches[0].0.len() / (hkv * d);

    // Embedding lookup (no embedding scale in Qwen2/Mistral).
    let embed = named_weight("w.embed", vocab * h);
    let mut x: Vec<f32> = embed[token * h..(token + 1) * h].to_vec();

    // RoPE tables: one cos/sin row per position.
    let tables = rope_tables(d, max_pos, 10_000.0, &RopeFlavor::Plain, None);
    let cos = tables.cos[pos * d..(pos + 1) * d].to_vec();
    let sin = tables.sin[pos * d..(pos + 1) * d].to_vec();

    for li in 0..num_layers {
        let p = |s: &str| format!("w.l{li}.{s}");

        // Pre-attention RMSNorm (input_layernorm).
        let ln1 = named_weight(&p("norm.attn"), h);
        let normed = rmsnorm_ref(&x, &ln1, eps);

        // QKV projections (no bias).
        let q_dim = hq * d;
        let kv_dim = hkv * d;
        let wq = named_weight(&p("attn.q"), h * q_dim);
        let wk = named_weight(&p("attn.k"), h * kv_dim);
        let wv = named_weight(&p("attn.v"), h * kv_dim);
        let wo = named_weight(&p("attn.o"), q_dim * h);

        let q = linear_ref(&normed, &wq, h, q_dim);
        let k = linear_ref(&normed, &wk, h, kv_dim);
        let v = linear_ref(&normed, &wv, h, kv_dim);

        // RoPE.
        let q_rot = rope_ref(&q, &cos, &sin, hq, d);
        let k_rot = rope_ref(&k, &cos, &sin, hkv, d);

        // Write KV cache at position `pos`.
        let (k_cache, v_cache) = &mut kv_caches[li];
        for hi in 0..hkv {
            let start = hi * cap * d + pos * d;
            k_cache[start..start + d].copy_from_slice(&k_rot[hi * d..(hi + 1) * d]);
            v_cache[start..start + d].copy_from_slice(&v[hi * d..(hi + 1) * d]);
        }

        // Attend all valid keys 0..=pos (sliced prefix; NO sliding window in this path).
        let seq = pos + 1;
        let (k_cache, v_cache) = (&kv_caches[li].0, &kv_caches[li].1);
        let mut attn_out = vec![0.0f32; q_dim];
        for qh in 0..hq {
            let kvh = qh / n_rep;
            let scores: Vec<f32> = (0..seq)
                .map(|s| {
                    let mut dot = 0.0f32;
                    for di in 0..d {
                        dot += q_rot[qh * d + di] * k_cache[kvh * cap * d + s * d + di];
                    }
                    dot * scale
                })
                .collect();
            let weights = softmax_ref(&scores);
            for (wi, s) in (0..seq).enumerate() {
                for di in 0..d {
                    attn_out[qh * d + di] += weights[wi] * v_cache[kvh * cap * d + s * d + di];
                }
            }
        }

        // o_proj: [q_dim] -> [h].
        let attn_proj = linear_ref(&attn_out, &wo, q_dim, h);

        // Residual add: x = x + attn_proj (no norm on the attn output itself).
        for i in 0..h {
            x[i] += attn_proj[i];
        }

        // Pre-MLP RMSNorm (post_attention_layernorm applied to the updated x).
        let ln2 = named_weight(&p("norm.ffn"), h);
        let normed_ff = rmsnorm_ref(&x, &ln2, eps);

        // SwiGLU MLP.
        let wg = named_weight(&p("ffn.gate"), h * inter);
        let wu = named_weight(&p("ffn.up"), h * inter);
        let wd = named_weight(&p("ffn.down"), inter * h);
        let gate = linear_ref(&normed_ff, &wg, h, inter);
        let up = linear_ref(&normed_ff, &wu, h, inter);
        // SwiGLU: silu(gate) * up. silu(v) = v / (1 + exp(-v)).
        let act: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();
        let mlp_out = linear_ref(&act, &wd, inter, h);

        // Second residual add: x = x + mlp_out (no norm on the MLP output).
        for i in 0..h {
            x[i] += mlp_out[i];
        }
    }

    // Final RMSNorm + LM head.
    let norm = named_weight("w.final_norm", h);
    let x = rmsnorm_ref(&x, &norm, eps);
    let lm_head = named_weight("w.head", h * vocab);
    linear_ref(&x, &lm_head, h, vocab)
}

/// llama-shaped decode: comparison of the llama family's decode step (no sliding window) against the hand-rolled
/// reference above. Tests: no QKV bias, correct 2-norm pre-norm structure, SwiGLU MLP, no embedding scale.
#[test]
fn llama_decode_bitexact_vs_reference() {
    let (vocab, hidden, layers, heads, kv_heads, head_dim, inter, max_pos) =
        (8, 8, 2, 4, 2, 4, 16, 16);
    let eps = 1e-5;
    let (pos, cap) = (4usize, 8usize);
    let token = 3usize;
    let (hkv, d) = (kv_heads, head_dim);

    let make_initial_caches = || -> Vec<(Vec<f32>, Vec<f32>)> {
        (0..layers)
            .map(|li| {
                let k = named_weight(&format!("llama.init.k.{li}"), hkv * cap * d);
                let v = named_weight(&format!("llama.init.v.{li}"), hkv * cap * d);
                (k, v)
            })
            .collect()
    };

    // --- Hand-rolled reference ---
    let mut kv_ref = make_initial_caches();
    let ref_logits = llama_decode_ref(
        vocab,
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        inter,
        eps,
        max_pos,
        pos,
        token,
        &mut kv_ref,
    );

    // --- The family's decode step over the same initial caches ---
    // GQA (n_rep = 2) exercises the grouped head path (R486-013).
    let m = Dense::new(Family::Llama)
        .vocab(vocab)
        .dims(hidden, inter, layers)
        .heads(heads, kv_heads)
        .head_dim(head_dim)
        .max_positions(max_pos)
        .with("rms_norm_eps", eps)
        .f32_model();
    let g = decode_graph(&m, cap);
    let kv_init = make_initial_caches();
    let state: Vec<HostTensor> = g
        .state
        .iter()
        .enumerate()
        .map(|(ci, &(si, _))| {
            let (k_flat, v_flat) = &kv_init[ci / 2];
            let data = if ci % 2 == 0 { k_flat } else { v_flat };
            HostTensor::f32(g.aval(si).shape.clone(), data.clone())
        })
        .collect();
    let (eval_logits, _) = run_step(&g, &[token as i32], pos, &named_weights, &state);
    assert_eq!(eval_logits.shape(), vec![1, 1, vocab]);
    assert_eq!(ref_logits.len(), vocab);

    for &v in eval_logits.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "eval logit {v} is not finite");
    }
    for &v in ref_logits.iter() {
        assert!(v.is_finite(), "ref logit {v} is not finite");
    }

    let max_abs = max_abs_error(eval_logits.as_f32().unwrap(), &ref_logits);

    eprintln!("llama_decode_bitexact max_abs = {max_abs:.2e}");
    assert!(
        max_abs <= 1e-4,
        "llama decode reference mismatch: max_abs={max_abs:.2e} (expected <= 1e-4)"
    );
}

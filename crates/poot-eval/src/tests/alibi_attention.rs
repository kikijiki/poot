//! ALiBi (Attention with Linear Biases, spec 254) CPU-oracle tests.
//!
//! Two levels: (1) the `attention_masked` op fed a hand-built per-head ALiBi bias matches an
//! independently hand-rolled reference forward pass bit-for-bit (decomposition correctness); (2) the
//! BLOOM family's `Model::trace` decode (ALiBi attention) produces the right graph structure (no RoPE, no
//! `Slot::Mask`) and a finite output.
//!
//! Card 557: tracers no longer emit the flash ops, so the tracer-level "flash-fused trace equals the
//! decomposed trace" checks live with the pass that forms the op (`poot-graph-plan`'s
//! `alibi_decode_flash_fusion.rs`); the flash-op tests here stage the op directly ([`stage_flash`]).

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::{Builder, Traced};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::ops::attention_masked;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{ComputedConst, Operand, Slot, Storage};
use poot_models::model::{LogitRows, Model, Phase};
use poot_models::registry::Registry;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap;
use std::sync::Arc;

use super::helpers::*;
use super::qwen2_pipeline::{eval_step, trace_step};
use poot_test_util::{assert_close_rel, max_abs_error, seed_of};

/// Stage one flash-attention `op` over `[q, k, v, mask]`. Card 557: `Builder` has no method for a
/// composite (`compile`'s `flash_attention_capped` forms it from the traced chain), so a test of the
/// op's own eval stages it through an append plan, as `e4m3_per_channel.rs` stages a
/// `PackedContraction`.
fn stage_flash(b: &Builder, op: OpKind, [q, k, v, mask]: [Traced; 4]) -> Traced {
    let mut plan = b.append_plan(0);
    let out = plan
        .equation(
            op,
            [q, k, v, mask]
                .iter()
                .map(|t| Operand::Value(t.id))
                .collect(),
        )
        .expect("stage the flash op");
    plan.declare_result(out).expect("declare result");
    let mut prepared = b.preflight_append(plan).expect("preflight append");
    Traced {
        id: b.commit_append(&mut prepared).expect("commit append"),
    }
}

fn decode(n_rep: usize, scale: f32) -> OpKind {
    OpKind::FlashAttentionDecode { n_rep, scale }
}

fn prefill(n_rep: usize, scale: f32) -> OpKind {
    OpKind::FlashAttentionPrefill {
        n_rep,
        scale,
        softcap: None,
    }
}

/// Hand-computed ALiBi decode mask row, independent of `poot_llm::graphs::alibi_decode_mask_row`
/// (poot-eval does not depend on poot-llm): visibility (0 / -1e9) plus `-slope[h] * (pos - t)` on visible slots.
fn alibi_row_ref(slopes: &[f32], cap: usize, pos: usize) -> Vec<f32> {
    let n_heads = slopes.len();
    let mut out = vec![0.0f32; n_heads * cap];
    for h in 0..n_heads {
        for t in 0..cap {
            out[h * cap + t] = if t <= pos {
                -slopes[h] * (pos as f32 - t as f32)
            } else {
                -1.0e9
            };
        }
    }
    out
}

/// Hand-rolled reference: per-head softmax(scale*q.k^T + bias) @ v, GQA via n_rep, as a direct Rust loop
/// independent of `attention_masked`'s decomposition (same style as `gemma2_decode_ref`).
#[allow(clippy::too_many_arguments)]
fn attention_alibi_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: &[f32], // [hq, cap]
    hq: usize,
    hkv: usize,
    cap: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let n_rep = hq / hkv;
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kv = h / n_rep;
        let mut scores = vec![0.0f32; cap];
        for (t, sc) in scores.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[(kv * cap + t) * d + dd];
            }
            *sc = s * scale + bias[h * cap + t];
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - m).exp();
            denom += *sc;
        }
        for dd in 0..d {
            let mut acc = 0.0f32;
            for (t, &sc) in scores.iter().enumerate() {
                acc += sc / denom * v[(kv * cap + t) * d + dd];
            }
            out[h * d + dd] = acc;
        }
    }
    out
}

/// `attention_masked` fed a per-head ALiBi bias (`[1,Hq,1,cap]` instead of `[1,1,1,cap]`) matches the
/// hand-rolled reference bit-for-bit: elementwise-add broadcast already carries a per-head bias, so no
/// new op is needed (see `attention_masked` in poot-graph-ir/src/ops.rs).
#[test]
fn attention_masked_with_alibi_bias_matches_hand_rolled_reference() {
    let (hq, hkv, cap, d, pos) = (4usize, 2usize, 5usize, 4usize, 2usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();

    // n_heads=4: slope_i = 2^(-2*(i+1)), as hand-typed literals independent of poot-llm's
    // alibi_slopes (unit-tested in crates/poot-llm/src/graphs.rs::mask_tests).
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];

    let qd = fill(hq * d, 11);
    let kd = fill(hkv * cap * d, 12);
    let vd = fill(hkv * cap * d, 13);
    let mask_data = alibi_row_ref(&slopes, cap, pos);

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, hq, 1, cap]));
    let out = attention_masked(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);

    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, 1, d], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, cap, d], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, cap, d], vd.clone())),
    );
    inputs.insert(
        mi,
        Value::from(HostTensor::f32(vec![1, hq, 1, cap], mask_data.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            r.output
                .into_host()
                .expect("alibi_attention tests evaluate dense graphs")
        })
        .unwrap();

    let want = attention_alibi_ref(&qd, &kd, &vd, &mask_data, hq, hkv, cap, d, scale);
    assert_eq!(got.shape(), vec![1, hq, 1, d]);
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);

    // Sanity: not NaN, and differs from plain causal attention on the same q/k/v.
    for &v in got.as_f32().unwrap().iter() {
        assert!(v.is_finite());
    }
    let plain_mask: Vec<f32> = alibi_row_ref(&[0.0, 0.0, 0.0, 0.0], cap, pos);
    let plain = attention_alibi_ref(&qd, &kd, &vd, &plain_mask, hq, hkv, cap, d, scale);
    let max_diff = max_abs_error(got.as_f32().unwrap(), &plain);
    assert!(
        max_diff > 1e-4,
        "ALiBi bias should change attention output vs. zero-slope baseline; max_diff={max_diff:.2e}"
    );
}

/// The BLOOM family's decode (ALiBi attention, no RoPE): it declares no RoPE table and no `Slot::Mask` input (card 550:
/// the per-head bias is a graph computation, `alibi_mask_from_pos`, over `Slot::Pos` and the family's `AlibiSlopes`
/// computed constant), and it evaluates to finite logits of the vocabulary's shape.
#[test]
fn bloom_alibi_decode_skips_rope_and_has_no_mask_slot() {
    let (model, cap) = (bloom_f32_model(), 8usize);
    let vocab = model.config().vocab;
    let g = trace_step(&*model, Phase::Decode, (1, 1, cap), None, LogitRows::Last);

    // FR-002: no rope tables declared on the ALiBi path.
    for id in &g.inputs {
        let m = &g.values[*id];
        if let Storage::Const = m.storage {
            let name = m.name.as_deref().unwrap_or("");
            assert!(
                !name.contains("rope"),
                "an ALiBi decode must not declare rope tables, found {name}"
            );
        }
    }

    // Card 550: no Slot::Mask input at all; the widened per-head bias is a graph computation over
    // Slot::Pos ([1,1]) and the slopes.
    assert!(
        !g.inputs
            .iter()
            .any(|id| matches!(g.values[*id].storage, Storage::Slot(Slot::Mask))),
        "an ALiBi decode graph must declare no Slot::Mask input"
    );
    assert!(
        g.inputs.iter().any(|id| matches!(
            g.values[*id].storage,
            Storage::Computed(ComputedConst::AlibiSlopes { .. })
        )),
        "an ALiBi decode graph must read the per-head slopes"
    );

    let out = eval_step(&g, &[2], &[3], None, None).logits;
    assert_eq!(out.shape(), vec![1, 1, vocab]);
    for &v in out.as_f32().unwrap().iter() {
        assert!(v.is_finite(), "alibi logit {v} is not finite");
    }
}

/// The BLOOM fixture checkpoint with every tensor held as F32 (deterministic name-seeded values), built through the
/// registry: a graph whose consts the F32 name-seeded fill of [`eval_step`] binds.
fn bloom_f32_model() -> Box<dyn Model> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|e| e.family.as_str() == "bloom")
        .expect("the bloom family is registered");
    let fixture = (entry.fixture)();
    let mut store = WeightStore::builder();
    for (key, tensor) in fixture.store.iter() {
        let shape = tensor.shape();
        let n: usize = shape.iter().product();
        let values: Vec<f32> = fill(n, seed_of(key.as_str()))
            .iter()
            .map(|v| v * 0.1)
            .collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, shape, Arc::from(bytes)).unwrap();
        store
            .insert(key.clone(), WeightEntry::Dense(dense))
            .unwrap();
    }
    registry
        .build(&fixture.raw(), &store.build())
        .unwrap_or_else(|e| panic!("{e}"))
}

// Card 259: flash attention with a per-head mask (the shape ALiBi needs).

/// Hand-computed ALiBi prefill mask `[hq, l, l]`, row-major: the prefill counterpart of [`alibi_row_ref`]
/// (0 / -1e30 for future keys, `-slope[h] * (i - j)` on visible pairs), independent of
/// `poot_llm::graphs::alibi_prefill_mask`.
fn alibi_prefill_ref(slopes: &[f32], l: usize) -> Vec<f32> {
    let n_heads = slopes.len();
    let mut out = vec![0.0f32; n_heads * l * l];
    for h in 0..n_heads {
        for i in 0..l {
            for j in 0..l {
                out[(h * l + i) * l + j] = if j <= i {
                    -slopes[h] * (i as f32 - j as f32)
                } else {
                    -1.0e30
                };
            }
        }
    }
    out
}

/// Hand-rolled prefill reference: per-head, per-row `softmax(scale*q.k^T + bias[h,row,:]) @ v`, GQA via
/// n_rep; the decode twin is [`attention_alibi_ref`].
#[allow(clippy::too_many_arguments)]
fn prefill_alibi_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: &[f32], // [hq, l, l]
    hq: usize,
    hkv: usize,
    l: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let n_rep = hq / hkv;
    let mut out = vec![0.0f32; hq * l * d];
    for h in 0..hq {
        let kv = h / n_rep;
        for row in 0..l {
            let mut scores = vec![0.0f32; l];
            for (t, sc) in scores.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for dd in 0..d {
                    s += q[(h * l + row) * d + dd] * k[(kv * l + t) * d + dd];
                }
                *sc = s * scale + bias[(h * l + row) * l + t];
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for sc in scores.iter_mut() {
                *sc = (*sc - m).exp();
                denom += *sc;
            }
            for dd in 0..d {
                let mut acc = 0.0f32;
                for (t, &sc) in scores.iter().enumerate() {
                    acc += sc / denom * v[(kv * l + t) * d + dd];
                }
                out[(h * l + row) * d + dd] = acc;
            }
        }
    }
    out
}

/// Card 259 decode: `OpKind::FlashAttentionDecode` fed a per-head mask `[1,Hq,1,cap]` scores head `h`
/// with its own bias row, matching the hand-rolled reference and the materialized `attention_masked`
/// decomposition.
///
/// Mutation pinned: with the head stride dropped, every head reads head 0's bias row. The final
/// assertion requires the per-head result to differ from the head-0-broadcast result beyond tolerance.
#[test]
fn flash_decode_per_head_alibi_mask_matches_reference_and_decomposition() {
    let (hq, hkv, cap, d, pos) = (4usize, 2usize, 6usize, 4usize, 3usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];

    let qd = fill(hq * d, 21);
    let kd = fill(hkv * cap * d, 22);
    let vd = fill(hkv * cap * d, 23);
    let mask_data = alibi_row_ref(&slopes, cap, pos);

    let run_flash = |mask_shape: Vec<usize>, mask_data: Vec<f32>| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let mask = b.constant("mask", TensorType::f32(mask_shape.clone()));
        let out = stage_flash(&b, decode(n_rep, scale), [q, k, v, mask]);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(
            qi,
            Value::from(HostTensor::f32(vec![1, hq, 1, d], qd.clone())),
        );
        inputs.insert(
            ki,
            Value::from(HostTensor::f32(vec![1, hkv, cap, d], kd.clone())),
        );
        inputs.insert(
            vi,
            Value::from(HostTensor::f32(vec![1, hkv, cap, d], vd.clone())),
        );
        inputs.insert(mi, Value::from(HostTensor::f32(mask_shape, mask_data)));
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("alibi_attention tests evaluate dense graphs")
            })
            .unwrap()
    };

    let got = run_flash(vec![1, hq, 1, cap], mask_data.clone());
    assert_eq!(got.shape(), vec![1, hq, 1, d]);

    // (1) vs the independent hand-rolled forward pass.
    let want = attention_alibi_ref(&qd, &kd, &vd, &mask_data, hq, hkv, cap, d, scale);
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);

    // (2) vs the materialized `attention_masked` decomposition with the same per-head mask.
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, hq, 1, cap]));
    let out = attention_masked(&b, q, k, v, n_rep, scale, mask);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, 1, d], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, cap, d], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, cap, d], vd.clone())),
    );
    inputs.insert(
        mi,
        Value::from(HostTensor::f32(vec![1, hq, 1, cap], mask_data.clone())),
    );
    let decomposed = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            r.output
                .into_host()
                .expect("alibi_attention tests evaluate dense graphs")
        })
        .unwrap();
    // Card 556: the oracle evaluates the flash op from this very decomposition (ADR-0101 tier 1), so
    // the two agree bit for bit, not within a tolerance.
    let (got_bits, decomposed_bits): (Vec<u32>, Vec<u32>) = got
        .as_f32()
        .unwrap()
        .iter()
        .zip(decomposed.as_f32().unwrap())
        .map(|(got, decomposed)| (got.to_bits(), decomposed.to_bits()))
        .unzip();
    assert!(got.as_f32().unwrap().iter().all(|v| v.is_finite()));
    assert_eq!(
        got_bits, decomposed_bits,
        "flash decode vs its decomposition"
    );

    // (3) Mutation: head 0's row broadcast over all heads (a dropped head stride) gives a materially
    // different answer.
    let head0: Vec<f32> = mask_data[..cap].to_vec();
    let broadcast = run_flash(vec![1, 1, 1, cap], head0);
    let max_diff = max_abs_error(got.as_f32().unwrap(), broadcast.as_f32().unwrap());
    assert!(
        max_diff > 1e-4,
        "a per-head ALiBi mask must differ from head 0's row broadcast over every head (a dropped \
         mask head stride would silently produce the latter); max_diff={max_diff:.2e}"
    );
}

/// Backward compatibility (card 259): a broadcast mask `[B,1,1,cap]` (head stride 0) scores as before.
/// Checked batched (B>1, card 038) so the batch stride is exercised too.
#[test]
fn flash_decode_broadcast_mask_unchanged_batched() {
    let (bsz, hq, hkv, cap, d) = (3usize, 4usize, 2usize, 5usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();

    let qd = fill(bsz * hq * d, 31);
    let kd = fill(bsz * hkv * cap * d, 32);
    let vd = fill(bsz * hkv * cap * d, 33);
    // One visibility row per batch row at different positions, so a wrong batch stride shows.
    let mask_data: Vec<f32> = (0..bsz)
        .flat_map(|bi| (0..cap).map(move |t| if t <= bi + 1 { 0.0 } else { -1.0e9 }))
        .collect();

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![bsz, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![bsz, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![bsz, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![bsz, 1, 1, cap]));
    let out = stage_flash(&b, decode(n_rep, scale), [q, k, v, mask]);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![bsz, hq, 1, d], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![bsz, hkv, cap, d], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![bsz, hkv, cap, d], vd.clone())),
    );
    inputs.insert(
        mi,
        Value::from(HostTensor::f32(vec![bsz, 1, 1, cap], mask_data.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            r.output
                .into_host()
                .expect("alibi_attention tests evaluate dense graphs")
        })
        .unwrap();

    // Reference: `mask[bi*cap + t]` written out per batch row.
    let mut want = vec![0.0f32; bsz * hq * d];
    for bi in 0..bsz {
        let bias: Vec<f32> = (0..hq)
            .flat_map(|_| mask_data[bi * cap..(bi + 1) * cap].to_vec())
            .collect();
        let row = attention_alibi_ref(
            &qd[bi * hq * d..(bi + 1) * hq * d],
            &kd[bi * hkv * cap * d..(bi + 1) * hkv * cap * d],
            &vd[bi * hkv * cap * d..(bi + 1) * hkv * cap * d],
            &bias,
            hq,
            hkv,
            cap,
            d,
            scale,
        );
        want[bi * hq * d..(bi + 1) * hq * d].copy_from_slice(&row);
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

/// Card 259 prefill, the `[1,Hq,L,L]` twin of
/// [`flash_decode_per_head_alibi_mask_matches_reference_and_decomposition`]: one `[L,L]` bias plane per
/// head, matching the reference and differing from head 0's plane broadcast (dropped head stride).
#[test]
fn flash_prefill_per_head_alibi_mask_matches_reference() {
    let (hq, hkv, l, d) = (4usize, 2usize, 5usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];

    let qd = fill(hq * l * d, 41);
    let kd = fill(hkv * l * d, 42);
    let vd = fill(hkv * l * d, 43);
    let mask_data = alibi_prefill_ref(&slopes, l);

    let run_flash = |mask_shape: Vec<usize>, mask_data: Vec<f32>| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
        let mask = b.constant("mask", TensorType::f32(mask_shape.clone()));
        let out = stage_flash(&b, prefill(n_rep, scale), [q, k, v, mask]);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(
            qi,
            Value::from(HostTensor::f32(vec![1, hq, l, d], qd.clone())),
        );
        inputs.insert(
            ki,
            Value::from(HostTensor::f32(vec![1, hkv, l, d], kd.clone())),
        );
        inputs.insert(
            vi,
            Value::from(HostTensor::f32(vec![1, hkv, l, d], vd.clone())),
        );
        inputs.insert(mi, Value::from(HostTensor::f32(mask_shape, mask_data)));
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .map(|r| {
                r.output
                    .into_host()
                    .expect("alibi_attention tests evaluate dense graphs")
            })
            .unwrap()
    };

    let got = run_flash(vec![1, hq, l, l], mask_data.clone());
    assert_eq!(got.shape(), vec![1, hq, l, d]);
    let want = prefill_alibi_ref(&qd, &kd, &vd, &mask_data, hq, hkv, l, d, scale);
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);

    // Mutation: head 0's [L,L] plane broadcast over every head (a dropped head stride) differs.
    let head0: Vec<f32> = mask_data[..l * l].to_vec();
    let broadcast = run_flash(vec![1, 1, l, l], head0);
    let max_diff = max_abs_error(got.as_f32().unwrap(), broadcast.as_f32().unwrap());
    assert!(
        max_diff > 1e-4,
        "a per-head ALiBi prefill mask must differ from head 0's plane broadcast over every head; \
         max_diff={max_diff:.2e}"
    );
}

/// Backward compatibility: the broadcast `[1,1,L,L]` prefill mask still scores as `mask[row*L + t]`.
#[test]
fn flash_prefill_broadcast_mask_unchanged() {
    let (hq, hkv, l, d) = (4usize, 2usize, 5usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();

    let qd = fill(hq * l * d, 51);
    let kd = fill(hkv * l * d, 52);
    let vd = fill(hkv * l * d, 53);
    let causal = alibi_prefill_ref(&[0.0], l); // zero slope -> plain causal [1,L,L]

    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let out = stage_flash(&b, prefill(n_rep, scale), [q, k, v, mask]);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, l, d], qd.clone())),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(vec![1, hkv, l, d], kd.clone())),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(vec![1, hkv, l, d], vd.clone())),
    );
    inputs.insert(
        mi,
        Value::from(HostTensor::f32(vec![1, 1, l, l], causal.clone())),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            r.output
                .into_host()
                .expect("alibi_attention tests evaluate dense graphs")
        })
        .unwrap();

    let bias: Vec<f32> = (0..hq).flat_map(|_| causal.clone()).collect();
    let want = prefill_alibi_ref(&qd, &kd, &vd, &bias, hq, hkv, l, d, scale);
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

/// A mask whose head axis is neither 1 nor Hq is rejected clearly, not misread. The
/// op's typing rule refuses it (card 496), so the tracer cannot build it; a graph mutated after tracing
/// fails `Graph::validate` with the typed operand error, and eval's own stride guard still refuses it.
#[test]
fn flash_decode_rejects_bad_mask_head_axis() {
    let (hq, hkv, cap, d) = (4usize, 2usize, 5usize, 4usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let out = stage_flash(&b, decode(hq / hkv, 1.0), [q, k, v, mask]);
    let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
    let mut g = b.finish(out);
    g.values[mi].aval.shape = vec![1, 2, 1, cap]; // 2 is neither 1 nor Hq=4
    let mut inputs = HashMap::new();
    inputs.insert(
        qi,
        Value::from(HostTensor::f32(vec![1, hq, 1, d], fill(hq * d, 61))),
    );
    inputs.insert(
        ki,
        Value::from(HostTensor::f32(
            vec![1, hkv, cap, d],
            fill(hkv * cap * d, 62),
        )),
    );
    inputs.insert(
        vi,
        Value::from(HostTensor::f32(
            vec![1, hkv, cap, d],
            fill(hkv * cap * d, 63),
        )),
    );
    inputs.insert(
        mi,
        Value::from(HostTensor::f32(vec![1, 2, 1, cap], fill(2 * cap, 64))),
    );
    let invalid = g.validate().unwrap_err().to_string();
    assert!(
        invalid.contains("flash attention mask must be [1, 1, 1, 5], got [1, 2, 1, 5]"),
        "expected the typed mask operand error, got: {invalid}"
    );
    // Card 554d: `eval` now runs `g.validate()` as its own first step (SC-010's fail-closed
    // binding sits behind the same gate), so the walk's error for this malformed graph is the same
    // typed `g.validate()` mismatch just asserted above, not a distinct "mask head axis" message.
    let err = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .map(|r| {
            r.output
                .into_host()
                .expect("alibi_attention tests evaluate dense graphs")
        })
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("flash attention mask must be [1, 1, 1, 5], got [1, 2, 1, 5]"),
        "expected the same typed mask error from eval, got: {err}"
    );
}

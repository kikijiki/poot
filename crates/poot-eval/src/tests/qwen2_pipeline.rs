//! Masked attention, softcap, matmul, cse/fuse, qwen2 end-to-end, fixed KV, legalize's host-embed
//! rewrite, verify-quant.

use crate::{EvalBudget, EvalOptions, Value};
use poot_executor_parity::dense::{Dense, Family, paged_step, step};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{attention, attention_masked, attention_softcap, softcap};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, Slot, StateRole, Storage, ValidationOutputs, ValueId};
use poot_models::model::{LogitRows, Model, Phase};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error};

#[test]
fn masked_attention_equals_sliced() {
    // G3d FR-001: masked attention over the full fixed-capacity cache == sliced attention over the valid prefix [0..=pos].
    // cap=4, pos=1 -> slots 0,1 valid, slots 2,3 masked (arbitrary stale data).
    let (hq, hkv, cap, d, pos) = (2usize, 1usize, 4usize, 4usize, 1usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();

    let qd = fill(hq * d, 8);
    let kd = fill(hkv * cap * d, 9); // slots past pos hold arbitrary (stale) values
    let vd = fill(hkv * cap * d, 10);
    let mut maskd = vec![0.0f32; cap];
    for m in maskd.iter_mut().skip(pos + 1) {
        *m = -1.0e9; // additive -inf for slots t > pos
    }

    // masked path: full cache + additive mask.
    let masked = {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked(&b, q, k, v, n_rep, scale, mask);
        let (qi, ki, vi, mi) = (q.id, k.id, v.id, mask.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, cap, d], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, cap, d], vd.clone()));
        inputs.insert(mi, HostTensor::f32(vec![1, 1, 1, cap], maskd.clone()));
        {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        }
    };

    // sliced path (G2): attend only the valid prefix [0..=pos].
    let sliced = {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
        let kvalid = b.slice(k, 2, 0, pos + 1);
        let vvalid = b.slice(v, 2, 0, pos + 1);
        let out = attention(&b, q, kvalid, vvalid, n_rep, scale);
        let (qi, ki, vi) = (q.id, k.id, v.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, cap, d], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, cap, d], vd.clone()));
        {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        }
    };

    assert_close_rel(masked.as_f32().unwrap(), sliced.as_f32().unwrap(), 1e-5);
}

#[test]
fn softcap_evaluates_to_c_tanh_and_approaches_identity() {
    // B9 (Gemma2/Grok logit softcapping): softcap(x, c) = c * tanh(x / c), a composition of scalar mul, tanh, scalar mul.
    // Must evaluate elementwise to c*tanh(x/c), including large |x| where the internal tanh must not overflow.
    let n = 8usize;
    let c = 5.0f32;
    // include big magnitudes so |x/c| saturates tanh (stable-tanh no-overflow path).
    let xd: Vec<f32> = fill(n, 123).iter().map(|v| v * 40.0).collect();

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![n]));
    let y = softcap(&b, x, c);
    let xi = x.id;
    let g = b.finish(y);
    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![n], xd.clone()));
    let out = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let want: Vec<f32> = xd.iter().map(|&v| c * (v / c).tanh()).collect();
    assert_close_rel(out.as_f32().unwrap(), &want, 1e-5);

    // As c grows, softcap approaches the identity on modest logits (tanh(x/c) ~ x/c). The internal tanh is an exp-based
    // composition (no `tanh`/`expm1` primitive), so for x/c below ~1e-4 the `1 - exp(-2 x/c)` numerator cancels to 0 in f32;
    // c ~ 1e9 is beyond what it can represent. A cap of 1e4 (~200x the real Gemma2/Grok caps of 30-50) shows the identity
    // limit in the accurate range.
    let big = 1.0e4f32;
    let modest: Vec<f32> = fill(n, 7).iter().map(|v| v * 10.0).collect();
    let b2 = Builder::new();
    let x2 = b2.constant("x", TensorType::f32(vec![n]));
    let y2 = softcap(&b2, x2, big);
    let xi2 = x2.id;
    let g2 = b2.finish(y2);
    let mut in2 = HashMap::new();
    in2.insert(xi2, HostTensor::f32(vec![n], modest.clone()));
    let out2 = {
        let values: HashMap<ValueId, Value> = in2
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g2, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    assert_close_rel(out2.as_f32().unwrap(), &modest, 1e-3);
}

#[test]
fn attention_softcap_matches_reference_and_differs_from_uncapped() {
    // B9: decode attention with attn_logit_softcap = Some(c) must equal a hand-rolled reference that applies c*tanh(scores/c)
    // to the scaled QK^T scores before softmax, and must differ from un-capped attention. Tiny deterministic GQA shapes.
    let (hq, hkv, s, d) = (2usize, 1usize, 3usize, 4usize);
    let n_rep = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let cap = 0.2f32; // small cap -> meaningful saturation vs the un-capped scores
    // scale the operands up so the raw scores are large relative to `cap` (guarantees a visible delta).
    let qd: Vec<f32> = fill(hq * d, 11).iter().map(|v| v * 2.0).collect();
    let kd: Vec<f32> = fill(hkv * s * d, 12).iter().map(|v| v * 2.0).collect();
    let vd = fill(hkv * s * d, 13);

    let run = |softcap_c: Option<f32>| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hkv, s, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hkv, s, d]));
        let out = attention_softcap(&b, q, k, v, n_rep, scale, softcap_c);
        let (qi, ki, vi) = (q.id, k.id, v.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(qi, HostTensor::f32(vec![1, hq, 1, d], qd.clone()));
        inputs.insert(ki, HostTensor::f32(vec![1, hkv, s, d], kd.clone()));
        inputs.insert(vi, HostTensor::f32(vec![1, hkv, s, d], vd.clone()));
        {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        }
    };
    let capped = run(Some(cap));
    let uncapped = run(None);

    // independent Rust reference: capped attention per query head (GQA maps head h -> kv head h/n_rep).
    let mut want = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kvh = h / n_rep;
        let mut scores = vec![0.0f32; s];
        for (si, sc_out) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += qd[h * d + di] * kd[(kvh * s + si) * d + di];
            }
            let sc = dot * scale;
            *sc_out = cap * (sc / cap).tanh(); // softcap before softmax
        }
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&z| (z - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        for (di, w) in want[h * d..h * d + d].iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for si in 0..s {
                acc += (exps[si] / denom) * vd[(kvh * s + si) * d + di];
            }
            *w = acc;
        }
    }
    assert_close_rel(capped.as_f32().unwrap(), &want, 1e-5);

    // softcapping changed the attention output (capped != uncapped somewhere).
    let maxdiff = max_abs_error(capped.as_f32().unwrap(), uncapped.as_f32().unwrap());
    assert!(
        maxdiff > 1e-3,
        "attn_logit_softcap should change the attention output (max diff {maxdiff})"
    );
}

#[test]
fn matmul_matches_naive() {
    // [1,1,3]x[3,2] -> [1,1,2]
    let b = Builder::new();
    let a = b.constant("a", TensorType::f32(vec![1, 1, 3]));
    let w = b.constant("w", TensorType::f32(vec![3, 2]));
    let out = b.matmul(a, w);
    let (ai, wi) = (a.id, w.id);
    let g = b.finish(out);

    let ad = vec![1.0, 2.0, 3.0];
    let wd = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // row-major [3,2]
    let mut inputs = HashMap::new();
    inputs.insert(ai, HostTensor::f32(vec![1, 1, 3], ad.clone()));
    inputs.insert(wi, HostTensor::f32(vec![3, 2], wd.clone()));
    let got = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    // [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
    assert_close_rel(got.as_f32().unwrap(), &[14.0, 32.0], 1e-6);
}

/// A qwen2 of 2 layers, 4 query heads over 2 key-value heads of width 4, built through the registry:
/// the graph source every test below traces (`Model::trace`), not a tracer of its own.
pub(super) fn tiny_qwen2(
    vocab: usize,
    hidden: usize,
    inter: usize,
    max_pos: usize,
) -> Box<dyn Model> {
    Dense::new(Family::Qwen2)
        .vocab(vocab)
        .dims(hidden, inter, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(max_pos)
        .f32_model()
        .model
}

/// `rows` rows of `tokens` new tokens over `capacity` positions, one cache per row (`pool: None`) or
/// one `pool`-slot pool shared by every row.
pub(super) fn trace_step(
    model: &dyn Model,
    phase: Phase,
    (rows, tokens, capacity): (usize, usize, usize),
    pool: Option<usize>,
    logits: LogitRows,
) -> Graph<ValidationOutputs> {
    let shape = match pool {
        None => step(rows, tokens, capacity, logits),
        Some(pool) => paged_step(rows, tokens, capacity, pool, logits),
    };
    model.trace(phase, shape).expect("the shape traces")
}

/// A deterministic f32 value per named const, small enough to keep activations bounded.
pub(super) fn named_weight(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    HostTensor::f32(
        shape.to_vec(),
        fill(shape.iter().product::<usize>().max(1), seed)
            .iter()
            .map(|v| v * 0.1)
            .collect(),
    )
}

/// One row of a paged step: the pool slot of each logical position and the position its first new
/// token takes.
pub(super) struct PagedRow<'a> {
    pub slots: &'a [usize],
    pub start: usize,
}

/// The `(read, write)` slot maps of a paged step of `tokens` new tokens per row: `read` is each
/// row's `capacity` slots; `write` sends the pool slot of row `r`'s token `i` (position
/// `start + i`) to the flat token index `r * tokens + i`, and every other slot to -1.
pub(super) fn paged_maps(
    pool: usize,
    rows: &[PagedRow<'_>],
    tokens: usize,
) -> (Vec<i32>, Vec<i32>) {
    let read = rows
        .iter()
        .flat_map(|row| row.slots.iter().map(|&s| s as i32))
        .collect();
    let mut write = vec![-1i32; pool];
    for (r, row) in rows.iter().enumerate() {
        for i in 0..tokens {
            write[row.slots[row.start + i]] = (r * tokens + i) as i32;
        }
    }
    (read, write)
}

/// One evaluated step: its logits and the carried state, in `Graph::state` order.
pub(super) struct Stepped {
    pub logits: HostTensor,
    pub state: Vec<HostTensor>,
}

/// Evaluate `g` on the CPU oracle: weights by name ([`named_weight`]), the step's `tokens` and `pos`
/// (`[rows * tokens]`, row-major), a paged step's `(read, write)` maps, and `state` (zeros when
/// `None`).
pub(super) fn eval_step(
    g: &Graph<ValidationOutputs>,
    tokens: &[i32],
    pos: &[i32],
    maps: Option<(&[i32], &[i32])>,
    state: Option<&[HostTensor]>,
) -> Stepped {
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let t = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(m.aval.shape.clone(), tokens.to_vec()),
            Storage::Slot(Slot::Pos) => HostTensor::i32(m.aval.shape.clone(), pos.to_vec()),
            Storage::Slot(Slot::SlotMap) => {
                let (read, write) = maps.expect("a paged step binds its slot maps");
                let values = match m.slot_key().unwrap().to_string().as_str() {
                    "slotmap.read" => read,
                    "slotmap.write" => write,
                    other => panic!("unexpected slot map {other}"),
                };
                HostTensor::i32(m.aval.shape.clone(), values.to_vec())
            }
            Storage::Slot(other) => panic!("unexpected slot {other:?}"),
            Storage::Const => named_weight(m.name.as_deref().unwrap(), &m.aval.shape),
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()),
            Storage::State => continue,
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, Value::from(t));
    }
    for (i, &(si, _)) in g.state.iter().enumerate() {
        let carried = match state {
            None => HostTensor::zeros(g.aval(si).shape.clone()),
            Some(prev) => prev[i].clone(),
        };
        inputs.insert(si, Value::from(carried));
    }
    let evaluation = crate::eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    Stepped {
        logits: evaluation.output.into_host().unwrap(),
        state: evaluation
            .state
            .into_iter()
            .map(|v| v.into_host().unwrap())
            .collect(),
    }
}

/// Sanity: the full qwen2 trace evaluates end to end on random weights and yields finite logits of the right shape
/// (confirms the executor runs the whole primitive graph without shape/index bugs; coherence vs a real reference needs a
/// weight loader and tokenizer).
#[test]
fn qwen2_tiny_runs_end_to_end() {
    let model = tiny_qwen2(32, 16, 32, 16);
    let g = trace_step(&*model, Phase::Decode, (1, 1, 8), None, LogitRows::Last);
    g.validate().expect("validate");
    let logits = eval_step(&g, &[5], &[3], None, None).logits;
    assert_eq!(logits.shape(), vec![1, 1, 32]);
    assert!(
        logits.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "logits must be finite"
    );
}

#[test]
// `st[0] * 0` / `st[0] * 1` below are the row indices in the stride formula (kept parallel across the two row
// assertions); the explicit form triggers erasing_op/identity_op.
#[allow(clippy::erasing_op, clippy::identity_op)]
fn dynamic_update_slice_writes_one_slot() {
    // a [2,4,3] cache; write a [2,1,3] update at slot index 2 on axis 1, the rest unchanged.
    let b = Builder::new();
    let cache = b.state_input(
        "kcache",
        TensorType::f32(vec![2, 4, 3]),
        StateRole::Recurrent,
    );
    let upd = b.constant("upd", TensorType::f32(vec![2, 1, 3]));
    let written = b.dynamic_update_slice(cache, upd, 2, 1);
    // a trivial logits output so the graph is well-formed.
    let logits = b.constant("logits", TensorType::f32(vec![1, 1, 4]));
    let g = b.finish_with_state(logits, &[(cache, written)]);
    g.validate().expect("valid");

    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    inputs.insert(cache.id, HostTensor::zeros(vec![2, 4, 3]));
    inputs.insert(
        upd.id,
        HostTensor::f32(vec![2, 1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    );
    inputs.insert(logits.id, HostTensor::zeros(vec![1, 1, 4]));

    let (_out, states) = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (evaluation.output.into_host().unwrap(), state)
    };
    assert_eq!(states.len(), 1);
    let kc = &states[0];
    assert_eq!(kc.shape(), vec![2, 4, 3]);
    // row 0, slot 2 == [1,2,3]; row 1, slot 2 == [4,5,6]; everything else 0.
    let st = crate::strides(kc.shape());
    assert_eq!(
        &kc.as_f32().unwrap()[st[0] * 0 + st[1] * 2..][..3],
        &[1.0, 2.0, 3.0]
    );
    assert_eq!(
        &kc.as_f32().unwrap()[st[0] * 1 + st[1] * 2..][..3],
        &[4.0, 5.0, 6.0]
    );
    assert_eq!(
        kc.as_f32().unwrap().iter().filter(|&&v| v != 0.0).count(),
        6
    );
}

/// The G2 KV-cache invariant (oracle stage 1): decoding token by token through the fixed-KV path with a carried cache
/// reproduces the full causal-attention prefill, step for step. Weights are arbitrary (bound identically by name in
/// both graphs); the prefill's causal mask is real.
#[test]
fn fixed_kv_decode_matches_prefill() {
    let model = tiny_qwen2(32, 16, 32, 16);
    let tokens: Vec<i32> = vec![5, 9, 2, 14, 7, 1];
    let cap = tokens.len();

    // reference: prefill the first pos+1 tokens, take the last-position logits.
    let prefill_logits = |upto: usize| -> HostTensor {
        let g = trace_step(
            &*model,
            Phase::Prefill,
            (1, upto, cap),
            None,
            LogitRows::Last,
        );
        let pos: Vec<i32> = (0..upto as i32).collect();
        eval_step(&g, &tokens[..upto], &pos, None, None).logits
    };

    // fixed-KV decode: carry the per-layer caches across steps.
    let g = trace_step(&*model, Phase::Decode, (1, 1, cap), None, LogitRows::Last);
    let mut caches: Option<Vec<HostTensor>> = None;
    for (pos, &token) in tokens.iter().enumerate() {
        let out = eval_step(&g, &[token], &[pos as i32], None, caches.as_deref());
        caches = Some(out.state);
        let want = prefill_logits(pos + 1);
        assert_close_rel(out.logits.as_f32().unwrap(), want.as_f32().unwrap(), 5e-3);
    }
}

#[test]
fn prefill_from_embeds_matches_token_prefill() {
    // spec 049 (VLM): trace_prefill_kv_embeds (input embeddings) reproduces trace_prefill_kv (token ids) when fed
    // gather(embed_tokens, tokens); only the embedding source differs, so logits + KV match.
    use poot_graph_ir::{Slot, Storage};
    use poot_models::qwen2::{Qwen2Config, trace_prefill_kv, trace_prefill_kv_embeds};
    use std::collections::HashMap as Map;

    let cfg = Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 32,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 4,
        rotary_dim: 4,
        eps: 1e-6,
        max_pos: 16,
        qkv_bias: true,
        qk_norm: false,
        ..Default::default()
    };
    let tokens: Vec<u32> = vec![5, 9, 2, 14];
    let n = tokens.len();
    let cap = n;
    let h = cfg.hidden;
    let weight = |name: &str, shape: &[usize]| -> HostTensor {
        let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
            (acc ^ c as u64).wrapping_mul(1099511628211)
        });
        HostTensor::f32(
            shape.to_vec(),
            fill(shape.iter().product::<usize>().max(1), seed)
                .iter()
                .map(|v| v * 0.1)
                .collect(),
        )
    };

    // token-id prefill.
    let gt = trace_prefill_kv(cfg, n, cap);
    let mut ti: Map<ValueId, HostTensor> = Map::new();
    for &id in &gt.inputs {
        let m = gt.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                ti.insert(
                    id,
                    HostTensor::i32(vec![n], tokens.iter().map(|&t| t as i32).collect()),
                );
            }
            Storage::Slot(Slot::Pos) => {
                ti.insert(id, HostTensor::i32(vec![1, n], (0..n as i32).collect()));
            }
            Storage::Const => {
                ti.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape));
            }
            Storage::State => {}
            _ => panic!("unexpected input {:?}", m.storage),
        }
    }
    for &(si, _) in &gt.state {
        ti.insert(si, HostTensor::zeros(gt.aval(si).shape.clone()));
    }
    let (logits_tok, _) = {
        let values: HashMap<ValueId, Value> = ti
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation =
            crate::eval(&gt, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (evaluation.output.into_host().unwrap(), state)
    };

    // embeddings prefill: input_embeds[i] = embed_weight[tokens[i]].
    let embed_w = weight("model.embed_tokens.weight", &[cfg.vocab, h]);
    let mut input_embeds = vec![0.0f32; n * h];
    for (i, &t) in tokens.iter().enumerate() {
        input_embeds[i * h..(i + 1) * h]
            .copy_from_slice(&embed_w.as_f32().unwrap()[t as usize * h..(t as usize + 1) * h]);
    }
    let ge = trace_prefill_kv_embeds(cfg, n, cap);
    let mut ei: Map<ValueId, HostTensor> = Map::new();
    for &id in &ge.inputs {
        let m = ge.meta(id);
        match m.storage {
            Storage::Slot(Slot::Pos) => {
                ei.insert(id, HostTensor::i32(vec![1, n], (0..n as i32).collect()));
            }
            Storage::Slot(Slot::Activation) => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(
                    name, "activation.vlm.input_embeds",
                    "unexpected activation slot {name}"
                );
                ei.insert(id, HostTensor::f32(vec![n, h], input_embeds.clone()));
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                ei.insert(id, weight(name, &m.aval.shape));
            }
            Storage::State => {}
            _ => panic!("unexpected embeds input {:?}", m.storage),
        }
    }
    for &(si, _) in &ge.state {
        ei.insert(si, HostTensor::zeros(ge.aval(si).shape.clone()));
    }
    let (logits_emb, _) = {
        let values: HashMap<ValueId, Value> = ei
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation =
            crate::eval(&ge, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (evaluation.output.into_host().unwrap(), state)
    };

    assert_close_rel(
        logits_emb.as_f32().unwrap(),
        logits_tok.as_f32().unwrap(),
        1e-6,
    );
}

/// Spec 233 Phase 2: splitting one prompt's prefill into several paged steps (each writing `[pos, pos+l)` into the shared
/// pool, attending over the full resident prefix through the read map) must produce the same resident KV pool and
/// first-token logits as one paged step over the whole prompt. Both write the same per-position K/V into the same physical
/// slots (`global_slots`); only the number of forwards differs.
#[test]
fn chunked_prefill_matches_one_shot_shared_pool() {
    let model = tiny_qwen2(32, 16, 24, 32);
    let n = 7; // prompt length
    let pool_slots = 20; // shared physical pool (bigger than the prompt - other sequences' slots exist)
    let cap = n; // this sequence's logical capacity == the prompt length (no generation headroom needed)
    // a non-contiguous physical layout (real block-table allocation is never guaranteed contiguous), used by both the
    // one-shot step and the chunk steps.
    let global_slots: Vec<usize> = vec![3, 17, 1, 9, 14, 2, 11];
    assert_eq!(global_slots.len(), n);
    let tokens: Vec<i32> = vec![5, 11, 3, 20, 8, 1, 27];

    // --- one-shot paged prefill: the whole prompt in one forward. ---
    let g_one = trace_step(
        &*model,
        Phase::Prefill,
        (1, n, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let (read, write) = paged_maps(
        pool_slots,
        &[PagedRow {
            slots: &global_slots,
            start: 0,
        }],
        n,
    );
    let pos: Vec<i32> = (0..n as i32).collect();
    let one_shot = eval_step(&g_one, &tokens, &pos, Some((&read, &write)), None);

    // --- chunked paged prefill: the same prompt in 3 chunks (l = 3, 3, 1), the pool threaded across chunks. ---
    let chunk_size = 3;
    let mut chunk_state: Option<Vec<HostTensor>> = None;
    let mut logits_chunked = HostTensor::zeros(vec![1, 1, 32]);
    let mut start = 0;
    while start < n {
        let l = chunk_size.min(n - start);
        let g = trace_step(
            &*model,
            Phase::Prefill,
            (1, l, cap),
            Some(pool_slots),
            LogitRows::Last,
        );
        assert_eq!(
            g.state.len(),
            g_one.state.len(),
            "chunk and one-shot graphs must carry the same per-layer state pairs"
        );
        let (read, write) = paged_maps(
            pool_slots,
            &[PagedRow {
                slots: &global_slots,
                start,
            }],
            l,
        );
        let pos: Vec<i32> = (start as i32..(start + l) as i32).collect();
        let out = eval_step(
            &g,
            &tokens[start..start + l],
            &pos,
            Some((&read, &write)),
            chunk_state.as_deref(),
        );
        chunk_state = Some(out.state);
        logits_chunked = out.logits;
        start += l;
    }

    // Tolerance: the chunks reduce over the same gathered `cap`-wide pool with a per-row additive mask, so only
    // floating-point reassociation of the residual stream separates the paths; both checks compare max absolute error
    // (some logits sit near 0, where a relative tolerance overreads a tiny absolute error).
    assert_eq!(
        one_shot.logits.shape(),
        logits_chunked.shape(),
        "chunked_prefill_matches_one_shot_shared_pool: logits shape mismatch"
    );
    let logit_err = max_abs_error(
        logits_chunked.as_f32().unwrap(),
        one_shot.logits.as_f32().unwrap(),
    );
    assert!(
        logit_err < 5e-3,
        "chunked vs one-shot logits differ by more than FP-reassociation noise: max abs err {logit_err}"
    );

    // cross-check the resident pool itself, not just the final logits: every physical slot's K/V must match between the
    // one-shot and chunked pools, including slots never written by this admission (both leave them at zero).
    let final_chunk_state = chunk_state.expect("at least one chunk ran");
    assert_eq!(final_chunk_state.len(), one_shot.state.len());
    for (i, (chunk_pool, one_shot_pool)) in final_chunk_state
        .iter()
        .zip(one_shot.state.iter())
        .enumerate()
    {
        assert_eq!(
            chunk_pool.shape(),
            one_shot_pool.shape(),
            "state pair {i}: pool shape mismatch"
        );
        let pool_err = max_abs_error(
            chunk_pool.as_f32().unwrap(),
            one_shot_pool.as_f32().unwrap(),
        );
        assert!(
            pool_err < 5e-3,
            "state pair {i}: chunked vs one-shot pool differ by more than FP-reassociation noise: \
             max abs err {pool_err}"
        );
    }
}

/// Decomposition-correctness layer: `batch_engine_loop`'s fast paged prefill step
/// (`crates/poot-serve/src/batch.rs`) advances a mid-chunking slot's `Slot::pos` to the last position the chunk wrote, and
/// then the same iteration's unconditional wide-decode dispatch re-derives and overwrites that slot's KV with the
/// single-token decode kernel (a different reduction order than the chunk kernel's multi-row attention).
/// `chunked_prefill_matches_one_shot_shared_pool` only measures the chunk path's own noise (no redundant overwrite,
/// N=4 chunks). This test simulates the mechanism: a paged chunk write followed by one single-token paged decode forward at
/// batch=1 overwriting the position just written, repeated across 35 chunk boundaries (an 8K-token prompt at
/// chunk_size=256 hits this 32 times), checkpointed every few chunks against a one-shot forward over the same-length
/// prefix, to see whether the error grows boundedly.
#[test]
fn chunked_prefill_redundant_decode_overwrite_error_growth_bounded() {
    let chunk_size = 8;
    let num_chunks = 35; // realistic scale (the update's own GPU tests only covered N=4)
    let n = chunk_size * num_chunks; // 280-token prompt
    let pool_slots = n + 20; // headroom, like the N=4 test's pool_slots > prompt convention

    // max_pos covers every real position used below.
    let model = tiny_qwen2(32, 16, 24, n + 4);
    // this sequence's logical capacity == the prompt length (no generation headroom; as the N=4 test's `cap = n`).
    let cap = n;

    // non-contiguous physical layout (a fixed xorshift permutation of a pool bigger than the prompt); real block-table
    // allocation is never guaranteed contiguous.
    let mut global_slots: Vec<usize> = (0..n).collect();
    let mut seed: u64 = 88172645463325252;
    for i in (1..global_slots.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let j = (seed as usize) % (i + 1);
        global_slots.swap(i, j);
    }
    let global_slots: Vec<usize> = global_slots.iter().map(|&s| s + 5).collect(); // offset into pool_slots
    assert!(global_slots.iter().all(|&s| s < pool_slots));
    let tokens: Vec<i32> = (0..n).map(|i| (i % 32) as i32).collect();

    // independent one-shot reference over the first `m` tokens (what the pool would hold after processing exactly this many
    // tokens in a single forward, no chunking, no redundant overwrite).
    let one_shot_pool = |m: usize| -> Vec<HostTensor> {
        let g = trace_step(
            &*model,
            Phase::Prefill,
            (1, m, cap),
            Some(pool_slots),
            LogitRows::Last,
        );
        let (read, write) = paged_maps(
            pool_slots,
            &[PagedRow {
                slots: &global_slots,
                start: 0,
            }],
            m,
        );
        let pos: Vec<i32> = (0..m as i32).collect();
        eval_step(&g, &tokens[..m], &pos, Some((&read, &write)), None).state
    };

    // chunked prefill replaying `batch_engine_loop`'s two steps per iteration: a chunk write, then (unconditionally, as the
    // wide decode dispatch does) a single-token decode-shaped overwrite of the position the chunk just wrote.
    let gd = trace_step(
        &*model,
        Phase::Decode,
        (1, 1, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let mut pool: Option<Vec<HostTensor>> = None;
    let mut pos = 0;
    let mut chunk_idx = 0;
    let mut checkpoints: Vec<(usize, f32)> = Vec::new(); // (chunks done, max abs pool error vs one-shot)
    while pos < n {
        let l = chunk_size.min(n - pos);
        let g = trace_step(
            &*model,
            Phase::Prefill,
            (1, l, cap),
            Some(pool_slots),
            LogitRows::Last,
        );
        let (read, write) = paged_maps(
            pool_slots,
            &[PagedRow {
                slots: &global_slots,
                start: pos,
            }],
            l,
        );
        let chunk_pos: Vec<i32> = (pos as i32..(pos + l) as i32).collect();
        let out = eval_step(
            &g,
            &tokens[pos..pos + l],
            &chunk_pos,
            Some((&read, &write)),
            pool.as_deref(),
        );
        pos += l;
        chunk_idx += 1;

        // the redundant same-iteration wide-decode overwrite: batch=1, position = the last position this chunk wrote.
        let last_pos = pos - 1;
        let (read, write) = paged_maps(
            pool_slots,
            &[PagedRow {
                slots: &global_slots,
                start: last_pos,
            }],
            1,
        );
        let overwritten = eval_step(
            &gd,
            &[tokens[last_pos]],
            &[last_pos as i32],
            Some((&read, &write)),
            Some(&out.state),
        );
        pool = Some(overwritten.state);

        if chunk_idx % 4 == 0 || pos == n {
            let reference = one_shot_pool(pos);
            let err = pool
                .as_ref()
                .unwrap()
                .iter()
                .zip(reference.iter())
                .map(|(a, r)| max_abs_error(a.as_f32().unwrap(), r.as_f32().unwrap()))
                .fold(0.0f32, f32::max);
            checkpoints.push((chunk_idx, err));
        }
    }

    eprintln!(
        "chunked_prefill_redundant_decode_overwrite_error_growth_bounded: (chunks_done, max_abs_err) = \
         {checkpoints:?}"
    );

    // Absolute ceiling: no checkpoint may exceed a small multiple of the N=4 baseline (5e-3,
    // `chunked_prefill_matches_one_shot_shared_pool`), even at 35 compounded redundant overwrites.
    for &(k, err) in &checkpoints {
        assert!(
            err < 0.05,
            "chunk {k}: max abs pool error {err} exceeds the bounded-growth ceiling (10x the N=4 \
             baseline of 5e-3) - the redundant wide-decode overwrite noise looks unbounded, not \
             FP-reassociation-sized"
        );
    }
    // Shape-of-growth check: catches divergent (exponential-in-N) growth even if every checkpoint stays under the absolute
    // ceiling; the last checkpoint's error should not dwarf the first's by more than a generous linear-ish factor.
    let (first_k, first_err) = checkpoints.first().copied().unwrap();
    let (last_k, last_err) = checkpoints.last().copied().unwrap();
    assert!(
        last_err < first_err * 20.0 + 1.0e-3,
        "error grew disproportionately from {first_err} (chunk {first_k}) to {last_err} (chunk \
         {last_k}): looks like divergent growth, not bounded FP-reassociation noise"
    );
}

/// Spec 233 Phase 3: a paged prefill step of `batch` rows' `l`-token draft windows in one forward against the shared KV
/// pool (the verify shape) must produce identical per-position logits and resident pool state to running the same `l`
/// query tokens through `l` sequential batched single-token paged decode steps (batch=2, one step per query position)
/// over the same starting pool. Same rigor as `chunked_prefill_matches_one_shot_shared_pool`, for the verify shape's
/// multi-row, multi-position form (two rows share one pool; each row's first query token rewrites its own resident
/// frontier position, the same harmless idempotent "redundant first position" convention verification relies on).
#[test]
fn speculative_verify_batched_matches_sequential_decode() {
    let model = tiny_qwen2(32, 16, 24, 32);
    let vocab = 32;
    let n = 5; // each row's prompt length
    let l = 4; // draft window width (1 requery + 3 drafts)
    let cap = n + 4; // each row's own logical capacity (room for the l-token draft window)
    let batch = 2;
    let pool_slots = 2 * cap + 4; // shared physical pool, bigger than 2 rows' reserved capacity

    // two rows' prompts, disjoint (non-contiguous) physical slot ranges within the same shared pool (two independently
    // admitted sequences sharing one engine pool).
    let tokens_row: [Vec<i32>; 2] = [vec![5, 11, 3, 20, 8], vec![2, 9, 30, 4, 17]];
    let global_slots_row: [Vec<usize>; 2] = [
        (0..cap).rev().collect(), // row 0: reverse order within its own disjoint block [0, cap)
        (0..cap).map(|i| cap + (cap - 1 - i)).collect(), // row 1: reverse order within [cap, 2*cap)
    ];
    // drafts for the l-1 speculative positions (arbitrary token ids: this test checks the verify step's numerics match
    // sequential decode of the same tokens, not any accept/reject outcome).
    let drafts_row: [Vec<i32>; 2] = [vec![1, 6, 19], vec![13, 0, 25]];
    assert_eq!(drafts_row[0].len(), l - 1);

    // --- prefill both rows' prompts into one shared pool, threading state across both admissions. ---
    let g_prefill = trace_step(
        &*model,
        Phase::Prefill,
        (1, n, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let mut pool_state: Option<Vec<HostTensor>> = None;
    for row in 0..batch {
        let (read, write) = paged_maps(
            pool_slots,
            &[PagedRow {
                slots: &global_slots_row[row],
                start: 0,
            }],
            n,
        );
        let pos: Vec<i32> = (0..n as i32).collect();
        let out = eval_step(
            &g_prefill,
            &tokens_row[row],
            &pos,
            Some((&read, &write)),
            pool_state.as_deref(),
        );
        pool_state = Some(out.state);
    }
    let prefilled_pool = pool_state.expect("at least one row prefilled");

    // --- (a) batched verify: both rows' l-token draft windows in one forward. ---
    let pos = n - 1; // both rows at their own frontier (the last resident prompt position)
    let query_row: [Vec<i32>; 2] = [0, 1].map(|row| {
        std::iter::once(tokens_row[row][n - 1])
            .chain(drafts_row[row].iter().copied())
            .collect()
    });
    let rows: Vec<PagedRow<'_>> = global_slots_row
        .iter()
        .map(|slots| PagedRow { slots, start: pos })
        .collect();
    let g_verify = trace_step(
        &*model,
        Phase::Prefill,
        (batch, l, cap),
        Some(pool_slots),
        LogitRows::All,
    );
    let (read, write) = paged_maps(pool_slots, &rows, l);
    let verify_tokens: Vec<i32> = query_row.iter().flatten().copied().collect();
    let verify_pos: Vec<i32> = (0..batch)
        .flat_map(|_| (0..l).map(|i| (pos + i) as i32))
        .collect();
    let verify = eval_step(
        &g_verify,
        &verify_tokens,
        &verify_pos,
        Some((&read, &write)),
        Some(&prefilled_pool),
    );
    assert_eq!(verify.logits.shape(), vec![batch, l, vocab]);

    // --- (b) sequential reference: the same l query tokens, one ordinary batched decode step at a time. ---
    let g_decode = trace_step(
        &*model,
        Phase::Decode,
        (batch, 1, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let mut seq_state = prefilled_pool.clone();
    let mut logits_seq = vec![0.0f32; batch * l * vocab];
    for i in 0..l {
        let step_pos = pos + i;
        let rows: Vec<PagedRow<'_>> = global_slots_row
            .iter()
            .map(|slots| PagedRow {
                slots,
                start: step_pos,
            })
            .collect();
        let (read, write) = paged_maps(pool_slots, &rows, 1);
        let toks: Vec<i32> = query_row.iter().map(|row| row[i]).collect();
        let out = eval_step(
            &g_decode,
            &toks,
            &vec![step_pos as i32; batch],
            Some((&read, &write)),
            Some(&seq_state),
        );
        assert_eq!(out.logits.shape(), vec![batch, 1, vocab]);
        for row in 0..batch {
            let dst = (row * l + i) * vocab;
            let src = row * vocab;
            logits_seq[dst..dst + vocab]
                .copy_from_slice(&out.logits.as_f32().unwrap()[src..src + vocab]);
        }
        seq_state = out.state;
    }

    // Tolerance: FP-reassociation only (a different but equivalent reduction order over the gathered pool window vs the
    // fresh k/v); compare max absolute error, not relative (some logits sit near 0).
    let logit_err = max_abs_error(verify.logits.as_f32().unwrap(), &logits_seq);
    assert!(
        logit_err < 5e-3,
        "batched verify vs sequential decode logits differ by more than FP-reassociation noise: \
         max abs err {logit_err}"
    );

    assert_eq!(verify.state.len(), seq_state.len());
    for (i, (verify_pool, seq_pool)) in verify.state.iter().zip(seq_state.iter()).enumerate() {
        assert_eq!(
            verify_pool.shape(),
            seq_pool.shape(),
            "state pair {i}: pool shape mismatch"
        );
        let pool_err = max_abs_error(verify_pool.as_f32().unwrap(), seq_pool.as_f32().unwrap());
        assert!(
            pool_err < 5e-3,
            "state pair {i}: batched-verify vs sequential-decode pool differ by more than \
             FP-reassociation noise: max abs err {pool_err}"
        );
    }
}

/// Spec 233 Phase 3b: `draft_propose_batch`'s (`crates/poot-serve/src/batch.rs`) round schedule drives the draft model's
/// dense per-slot batched decode graph through interleaved catch-up rounds (feed a row's committed tokens), propose rounds
/// (feed a row's own argmax back), and hold rounds (re-feed an unchanged token/position for an inactive row), all through
/// one `n_slots`-wide graph. Run on poot-eval's CPU interpreter (no GPU needed). For two rows with different catch-up
/// depths, the interleaving must give the same per-row logits (and proposed draft tokens) as running each row's decode
/// independently on its own batch=1 graph with no holds: a row's result never depends on what another row does in that
/// call. Mirrors `speculative_verify_batched_matches_sequential_decode`, for the propose side.
#[test]
fn draft_propose_round_schedule_matches_sequential_draft_decode() {
    let model = tiny_qwen2(32, 16, 24, 32);
    let vocab = 32;
    let cap = 12;
    let batch = 2;
    let max_k = 2;
    // row 0: a short catch-up (4 committed tokens, `draft_pos == 0`); row 1: a longer one (7), so the schedule must hold row 0
    // while row 1 catches up, then hold row 0 again once it finishes proposing before row 1 does (see `draft_propose_batch`'s
    // doc for the hold/catch-up/propose contract).
    let tokens_row: [Vec<i32>; 2] = [vec![5, 11, 3, 20], vec![2, 9, 30, 4, 17, 12, 7]];
    let catchup: [usize; 2] = [tokens_row[0].len(), tokens_row[1].len()];
    let total_rounds = catchup.iter().copied().max().unwrap() + max_k;

    // lowest-index tie-break, as `poot-serve`'s `argmax_row`.
    let argmax = |row: &[f32]| -> i32 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        best as i32
    };

    let g = trace_step(
        &*model,
        Phase::Decode,
        (batch, 1, cap),
        None,
        LogitRows::Last,
    );

    // --- BATCHED: the round schedule `draft_propose_batch` drives, one call per round, both rows sharing one state. ---
    let mut state: Option<Vec<HostTensor>> = None;
    let mut last_logits: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    let mut finished_logits: [Option<Vec<f32>>; 2] = [None, None];
    let mut drafts: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
    for round in 0..total_rounds {
        let mut toks = vec![0i32; batch];
        let mut poss = vec![0i32; batch];
        for row in 0..batch {
            let cu = catchup[row];
            if round < cu {
                toks[row] = tokens_row[row][round];
                poss[row] = round as i32;
            } else if round < cu + max_k {
                let am = argmax(&last_logits[row]);
                drafts[row].push(am);
                toks[row] = am;
                poss[row] = round as i32;
            } else {
                // hold: re-feed position 0's token (both rows start catch-up from `draft_pos == 0`); a harmless idempotent recompute,
                // never read back for drafting.
                toks[row] = tokens_row[row][0];
                poss[row] = 0;
            }
        }
        let out = eval_step(&g, &toks, &poss, None, state.as_deref());
        assert_eq!(out.logits.shape(), vec![batch, 1, vocab]);
        for row in 0..batch {
            let off = row * vocab;
            last_logits[row] = out.logits.as_f32().unwrap()[off..off + vocab].to_vec();
            if round == catchup[row] + max_k - 1 {
                // the row's last active (propose) round: snapshot now, before a later round's hold-phase recompute (a different query
                // position, not comparable) overwrites `last_logits[row]`.
                finished_logits[row] = Some(last_logits[row].clone());
            }
        }
        state = Some(out.state);
    }

    // --- SEQUENTIAL reference: each row decoded on its own batch=1 graph, independently (no interleaving, holds, shared
    // call or shared state). ---
    let g1 = trace_step(&*model, Phase::Decode, (1, 1, cap), None, LogitRows::Last);
    for row in 0..batch {
        let mut state1: Option<Vec<HostTensor>> = None;
        let mut dl: Vec<f32> = Vec::new();
        let mut ref_drafts: Vec<i32> = Vec::new();
        // `pos` is the absolute decode position (fed to the graph's Slot::Pos input), and only incidentally an index into
        // `tokens_row[row]` during catch-up; an iterator rewrite would not simplify this.
        #[allow(clippy::needless_range_loop)]
        for pos in 0..(catchup[row] + max_k) {
            let tok = if pos < catchup[row] {
                tokens_row[row][pos]
            } else {
                let am = argmax(&dl);
                ref_drafts.push(am);
                am
            };
            let out = eval_step(&g1, &[tok], &[pos as i32], None, state1.as_deref());
            assert_eq!(out.logits.shape(), vec![1, 1, vocab]);
            dl = out.logits.as_f32().unwrap().to_vec();
            state1 = Some(out.state);
        }

        assert_eq!(
            ref_drafts, drafts[row],
            "row {row}: batched round-schedule drafts diverge from independent sequential decode"
        );
        let finished = finished_logits[row]
            .as_ref()
            .expect("every row's own final propose round ran within total_rounds");
        let logit_err = max_abs_error(&dl, finished);
        assert!(
            logit_err < 5e-3,
            "row {row}: batched vs sequential final logits differ by more than FP-reassociation noise: \
             max abs err {logit_err}"
        );
    }
}

/// Card 550 SC-006: a chunked prefill starting at a NONZERO position (`Slot::Pos = [s, s+1, s+2]`) must
/// equal the same tokens run one decode step at a time from the same starting KV pool state - both the
/// per-position logits and the final pool - on the CPU oracle. The chunk and the decode steps share the
/// contiguous-equivalent shared-pool layout (`global_slots` is a non-contiguous permutation of `[0,cap)`,
/// so this also proves the chunk's write/read addressing, not just its mask, matches decode's).
#[test]
fn chunked_prefill_at_nonzero_start_matches_stepped_decode() {
    let model = tiny_qwen2(32, 16, 24, 16);
    let s = 3usize; // nonzero chunk start
    let l = 3usize; // chunk width
    let cap = s + l; // this sequence's logical capacity (room for s prior + l new positions)
    let pool_slots = cap + 4; // headroom, non-contiguous layout below
    let global_slots: Vec<usize> = (0..cap).map(|i| cap + 3 - i).collect(); // non-contiguous permutation
    let tokens: Vec<i32> = vec![5, 11, 3, 20, 8, 1];
    assert_eq!(tokens.len(), cap);

    let paged = |start: usize, tokens: usize| {
        paged_maps(
            pool_slots,
            &[PagedRow {
                slots: &global_slots,
                start,
            }],
            tokens,
        )
    };

    // --- build the first `s` positions' pool state via a paged step (pos=0..s). ---
    let g0 = trace_step(
        &*model,
        Phase::Prefill,
        (1, s, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let (read, write) = paged(0, s);
    let pos0: Vec<i32> = (0..s as i32).collect();
    let pool_after_s = eval_step(&g0, &tokens[..s], &pos0, Some((&read, &write)), None).state;

    // --- (a) chunk-prefill the next l tokens at positions [s, s+l) in one forward. ---
    let gc = trace_step(
        &*model,
        Phase::Prefill,
        (1, l, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let (read, write) = paged(s, l);
    let posc: Vec<i32> = (s as i32..(s + l) as i32).collect();
    let chunk = eval_step(
        &gc,
        &tokens[s..s + l],
        &posc,
        Some((&read, &write)),
        Some(&pool_after_s),
    );

    // --- (b) step decode through the same l positions, one token at a time, from the same pool_after_s. ---
    let gd = trace_step(
        &*model,
        Phase::Decode,
        (1, 1, cap),
        Some(pool_slots),
        LogitRows::Last,
    );
    let mut pool = pool_after_s;
    let mut logits_decode = HostTensor::zeros(vec![1, 1, 32]);
    for i in 0..l {
        let (read, write) = paged(s + i, 1);
        let out = eval_step(
            &gd,
            &[tokens[s + i]],
            &[(s + i) as i32],
            Some((&read, &write)),
            Some(&pool),
        );
        pool = out.state;
        logits_decode = out.logits;
    }

    // Both paths run the same per-row attention/projections over the same gathered keys in the same
    // order (the chunk's mask row i and decode step i are the identical computation), so this comparison
    // is exact: bit-identical in practice.
    let logit_err = max_abs_error(
        chunk.logits.as_f32().unwrap(),
        logits_decode.as_f32().unwrap(),
    );
    assert!(
        logit_err < 1e-5,
        "chunk prefill at nonzero start vs stepped decode: last-position logits differ: max abs err {logit_err}"
    );
    assert_eq!(chunk.state.len(), pool.len());
    for (i, (chunk_pool, decode_pool)) in chunk.state.iter().zip(pool.iter()).enumerate() {
        assert_eq!(
            chunk_pool.shape(),
            decode_pool.shape(),
            "state pair {i}: shape"
        );
        let pool_err = max_abs_error(chunk_pool.as_f32().unwrap(), decode_pool.as_f32().unwrap());
        assert!(
            pool_err < 1e-5,
            "state pair {i}: chunk vs stepped-decode pool differ: max abs err {pool_err}"
        );
    }
}

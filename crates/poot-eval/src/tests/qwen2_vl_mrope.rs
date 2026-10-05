//! Card 152 / spec 267: wiring `rope_sectioned` into a Qwen2-VL-class text-tower tracer
//! (`poot_models::qwen2::trace_prefill_mrope`/`trace_decode_mrope`, gated by `Qwen2Config::mrope_section`). The primitive
//! is covered by `crate::tests::mrope`; these tests target the tracer wiring (shapes, per-layer application, position
//! plumbing). An all-sections-equal mRoPE trace must be bit-exact to the plain-RoPE tracer on identical weights (any
//! wiring bug breaks this), plus a distinctness check (sectioning changes the result) and a multimodal fixture (finite,
//! correctly-shaped logits) for the 3-distinct-position case. CPU-oracle only; GPU parity and a real checkpoint are deferred.

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::{Slot, Storage};
use poot_models::mrope::{
    MropeGrid, MropePosition, MropePositionIds, MropeSegment, build_mrope_position_ids,
};
use poot_models::qwen2::{
    Qwen2Config, trace_decode, trace_decode_kv_masked_batched, trace_decode_mrope, trace_prefill,
    trace_prefill_kv_embeds, trace_prefill_mrope, trace_qwen2_5_vl_prefill_kv_embeds,
};
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap as Map;

/// A tiny Qwen2-VL-text config: `rotary_dim=8` (half=4) with sections `[1,1,2]` (temporal/height/width, summing to half;
/// the real Qwen2-VL-7B uses [16,24,24]), `qkv_bias=true`/`qk_norm=false` as in the real text tower (see
/// `Qwen2Config::mrope_section`).
fn tiny_cfg() -> Qwen2Config {
    Qwen2Config {
        vocab: 32,
        hidden: 16,
        inter: 32,
        layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        rotary_dim: 8,
        eps: 1e-6,
        max_pos: 16,
        qkv_bias: true,
        qk_norm: false,
        mrope_section: Some([1, 1, 2]),
        ..Default::default()
    }
}

/// Deterministic data for a named const, keyed by the name's hash (as `qwen2_pipeline::fixed_kv_decode_matches_prefill`'s
/// `weight` closure), so identical names bind identical data across two graphs.
fn weight(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    // No `.max(1)` clamp here: the trace_decode_mrope backward-compat test binds a zero-length `kv.cached_k`/`kv.cached_v` at
    // `kv_len=0`, and a clamped fill would exceed the declared shape and trip `HostTensor::new`'s length check.
    HostTensor::f32(
        shape.to_vec(),
        super::helpers::fill(shape.iter().product::<usize>(), seed)
            .iter()
            .map(|v| v * 0.1)
            .collect(),
    )
}

/// Bind a `trace_prefill_mrope` graph's inputs: the shared `weight()` consts, `Slot::Token`, and the axis-major `[3,L]`
/// `Slot::MropePosition` input from the host layout algorithm.
fn bind_prefill_mrope(
    g: &poot_graph_ir::Graph,
    tokens: &[u32],
    positions: &MropePositionIds,
) -> Map<poot_graph_ir::ValueId, HostTensor> {
    let l = tokens.len();
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = &g.values[id];
        match m.storage {
            Storage::Slot(Slot::Token) => {
                let toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                inputs.insert(id, HostTensor::i32(vec![l], toks));
            }
            Storage::Slot(Slot::MropePosition) => {
                inputs.insert(
                    id,
                    HostTensor::i32(
                        vec![3, l],
                        positions.packed_for_len(l).expect("matching prompt layout"),
                    ),
                );
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(id, HostTensor::i32(vec![1, l], (0..l as i32).collect()));
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, weight(name, &m.aval.shape));
            }
            _ => panic!(
                "unexpected trace_prefill_mrope input storage: {:?}",
                m.storage
            ),
        }
    }
    inputs
}

fn bind_prefill_kv_embeds(
    g: &poot_graph_ir::Graph,
    input_embeds: &[f32],
    positions: Option<&MropePositionIds>,
) -> Map<poot_graph_ir::ValueId, HostTensor> {
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        match m.storage {
            Storage::Slot(Slot::Pos) => {
                let l = m.aval.shape[1];
                inputs.insert(id, HostTensor::i32(vec![1, l], (0..l as i32).collect()));
            }
            Storage::Slot(Slot::Activation) => {
                let name = m.name.as_deref().unwrap();
                assert_eq!(
                    name, "activation.vlm.input_embeds",
                    "unexpected activation slot {name}"
                );
                inputs.insert(
                    id,
                    HostTensor::f32(m.aval.shape.clone(), input_embeds.to_vec()),
                );
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                inputs.insert(id, weight(name, &m.aval.shape));
            }
            Storage::Slot(Slot::MropePosition) => {
                let positions = positions.expect("mRoPE KV prefill graph needs positions");
                inputs.insert(
                    id,
                    HostTensor::i32(
                        m.aval.shape.clone(),
                        positions
                            .packed_for_len(m.aval.shape[1])
                            .expect("matching prompt positions"),
                    ),
                );
            }
            Storage::State => {
                inputs.insert(id, HostTensor::zeros(m.aval.shape.clone()));
            }
            _ => panic!(
                "unexpected KV embeds prefill input storage: {:?}",
                m.storage
            ),
        }
    }
    inputs
}

fn embed_rows(n: usize, h: usize, seed: u64) -> Vec<f32> {
    super::helpers::fill(n * h, seed)
        .iter()
        .map(|v| v * 0.1)
        .collect()
}

/// Structural backward compatibility (spec 267 FR-006) through the real tracer: an all-text prompt assigns every mRoPE axis
/// the same running counter `0..L` (HF's `get_rope_index`), which `rope_sectioned`'s structural collapse (mrope.rs) makes
/// bit-exact to plain `rope_prefill`. Holding through the full tracer checks the wiring: a shape/axis/position bug would
/// break this equality even if both graphs look finite and correctly shaped.
#[test]
fn trace_prefill_mrope_matches_trace_prefill_when_positions_collapse_to_running_counter() {
    let cfg = tiny_cfg();
    let tokens: Vec<u32> = vec![5, 9, 2, 14, 7, 1];
    let l = tokens.len();
    let positions = build_mrope_position_ids(&[MropeSegment::Text(l)], 1).unwrap();

    let g_mrope = trace_prefill_mrope(cfg, l);
    g_mrope.validate().expect("trace_prefill_mrope valid");
    let inputs = bind_prefill_mrope(&g_mrope, &tokens, &positions);
    let got = (|| -> Result<HostTensor, EvalError> {
        let values: Map<_, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g_mrope, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();

    let g_plain = trace_prefill(cfg, l);
    g_plain.validate().expect("trace_prefill valid");
    let mut plain_inputs: Map<poot_graph_ir::ValueId, HostTensor> = Map::new();
    for &id in &g_plain.inputs {
        let m = &g_plain.values[id];
        match m.storage {
            Storage::Slot(Slot::Token) => {
                let toks: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                plain_inputs.insert(id, HostTensor::i32(vec![l], toks));
            }
            Storage::Slot(Slot::Pos) => {
                plain_inputs.insert(id, HostTensor::i32(vec![1, l], (0..l as i32).collect()));
            }
            Storage::Const => {
                plain_inputs.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape));
            }
            _ => panic!("unexpected trace_prefill input storage: {:?}", m.storage),
        }
    }
    let want = (|| -> Result<HostTensor, EvalError> {
        let values: Map<_, Value> = plain_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g_plain, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();

    assert_eq!(got.shape(), want.shape());
    let gb: Vec<u32> = got.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    let wb: Vec<u32> = want.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        gb, wb,
        "an all-text prompt (mRoPE axes collapsed to the same running counter) must be bit-exact to \
         plain rope_prefill through the real tracer, not just the bare primitive"
    );
}

#[test]
fn trace_qwen25_vl_prefill_kv_embeds_collapsed_axes_match_plain_embeds_kv() {
    let cfg = tiny_cfg();
    let n = 5usize;
    let cap = 7usize;
    let input_embeds = embed_rows(n, cfg.hidden, 2468);
    let positions = build_mrope_position_ids(&[MropeSegment::Text(n)], 1).unwrap();

    let mrope_graph = trace_qwen2_5_vl_prefill_kv_embeds(cfg, n, cap);
    mrope_graph
        .validate()
        .expect("Qwen2.5-VL embeds KV prefill graph validates");
    let mrope_slot = mrope_graph
        .slots
        .iter()
        .filter_map(|&(id, slot)| (slot == Slot::MropePosition).then_some(id))
        .collect::<Vec<_>>();
    assert_eq!(mrope_slot.len(), 1);
    assert_eq!(mrope_graph.aval(mrope_slot[0]).shape, vec![3, n]);
    assert_eq!(mrope_graph.aval(mrope_slot[0]).dtype, DType::I32);

    let mrope_inputs = bind_prefill_kv_embeds(&mrope_graph, &input_embeds, Some(&positions));
    let (mrope_logits, mrope_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: Map<_, Value> = mrope_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(
            &mrope_graph,
            &values,
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let mut plain_cfg = cfg;
    plain_cfg.mrope_section = None;
    let plain_graph = trace_prefill_kv_embeds(plain_cfg, n, cap);
    plain_graph
        .validate()
        .expect("ordinary embeds KV prefill graph validates");
    assert!(
        plain_graph
            .slots
            .iter()
            .all(|(_, slot)| *slot != Slot::MropePosition)
    );
    let plain_inputs = bind_prefill_kv_embeds(&plain_graph, &input_embeds, None);
    let (plain_logits, plain_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: Map<_, Value> = plain_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(
            &plain_graph,
            &values,
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(
        mrope_logits.as_f32().unwrap(),
        plain_logits.as_f32().unwrap()
    );
    assert_eq!(mrope_state, plain_state);
}

#[test]
fn trace_qwen25_vl_prefill_kv_embeds_distinct_axes_reach_logits_and_kv_state() {
    let cfg = tiny_cfg();
    let positions = build_mrope_position_ids(
        &[
            MropeSegment::Text(1),
            MropeSegment::Image(MropeGrid::new(1, 4, 4)),
            MropeSegment::Text(2),
        ],
        2,
    )
    .unwrap();
    let n = positions.len();
    let cap = n + 2;
    let input_embeds = embed_rows(n, cfg.hidden, 97531);
    let graph = trace_qwen2_5_vl_prefill_kv_embeds(cfg, n, cap);
    graph.validate().expect("Qwen2.5-VL KV prefill validates");

    let distinct_inputs = bind_prefill_kv_embeds(&graph, &input_embeds, Some(&positions));
    let (distinct_logits, distinct_state) =
        (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: Map<_, Value> = distinct_inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
    let collapsed_positions = build_mrope_position_ids(&[MropeSegment::Text(n)], 1).unwrap();
    let collapsed_inputs =
        bind_prefill_kv_embeds(&graph, &input_embeds, Some(&collapsed_positions));
    let (collapsed_logits, collapsed_state) =
        (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: Map<_, Value> = collapsed_inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();

    assert_eq!(distinct_logits.shape(), vec![1, 1, cfg.vocab]);
    assert!(
        distinct_logits
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        distinct_logits.as_f32().unwrap() != collapsed_logits.as_f32().unwrap(),
        "distinct multimodal mRoPE axes must affect last-token logits"
    );
    assert!(
        distinct_state != collapsed_state,
        "distinct multimodal mRoPE axes must affect the filled KV state"
    );
}

#[test]
fn trace_prefill_kv_embeds_ignores_mrope_section_unless_qwen25_wrapper_is_used() {
    let mut mrope_cfg = tiny_cfg();
    let mut plain_cfg = mrope_cfg;
    plain_cfg.mrope_section = None;

    let mrope_field_graph = trace_prefill_kv_embeds(mrope_cfg, 3, 5);
    let plain_graph = trace_prefill_kv_embeds(plain_cfg, 3, 5);

    assert_eq!(
        format!("{mrope_field_graph:?}"),
        format!("{plain_graph:?}"),
        "ordinary trace_prefill_kv_embeds must remain structurally unchanged when cfg.mrope_section is set"
    );

    mrope_cfg.mrope_section = Some([1, 1, 2]);
    let qwen25_graph = trace_qwen2_5_vl_prefill_kv_embeds(mrope_cfg, 3, 5);
    assert!(
        qwen25_graph
            .slots
            .iter()
            .any(|(_, slot)| *slot == Slot::MropePosition)
    );
}

/// The general multimodal case: one host layout containing text, an image grid, more text, a video grid, and trailing text.
/// The positions flow through the packed slot and the Qwen2-VL text-tower tracer. Finite, correctly-shaped logits prove
/// graph coherence; inequality with the all-text layout proves sectioning is not inert.
#[test]
fn trace_prefill_mrope_distinct_positions_are_finite_and_differ_from_collapsed() {
    let cfg = tiny_cfg();
    let positions = build_mrope_position_ids(
        &[
            MropeSegment::Text(2),
            MropeSegment::Image(MropeGrid::new(1, 4, 6)),
            MropeSegment::Text(1),
            MropeSegment::Video(MropeGrid::new(2, 2, 4)),
            MropeSegment::Text(2),
        ],
        2,
    )
    .unwrap();
    let tokens: Vec<u32> = (0..positions.len())
        .map(|i| (i % cfg.vocab) as u32)
        .collect();
    let l = tokens.len();

    let g = trace_prefill_mrope(cfg, l);
    g.validate().expect("trace_prefill_mrope valid");
    let inputs = bind_prefill_mrope(&g, &tokens, &positions);
    let got = (|| -> Result<HostTensor, EvalError> {
        let values: Map<_, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();

    assert_eq!(got.shape(), vec![1, 1, cfg.vocab]);
    assert!(
        got.as_f32().unwrap().iter().all(|v| v.is_finite()),
        "mrope prefill logits must be finite"
    );

    let collapsed_positions = build_mrope_position_ids(&[MropeSegment::Text(l)], 1).unwrap();
    let collapsed = (|| -> Result<HostTensor, EvalError> {
        let values: Map<_, Value> = bind_prefill_mrope(&g, &tokens, &collapsed_positions)
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();
    assert_ne!(
        got.as_f32().unwrap(),
        collapsed.as_f32().unwrap(),
        "distinct multimodal position ids must change the logits vs the collapsed (all-equal) case - \
         otherwise the sectioning is silently inert"
    );
}

/// Same backward-compatibility property (spec 267 FR-006) at the decode/scalar-position shape, `trace_decode_mrope` vs
/// `trace_decode`. `kv_len=0` (first decode token, empty cache) keeps the fixture minimal; the property concerns the RoPE
/// wiring, not KV-cache length.
#[test]
fn trace_decode_mrope_matches_trace_decode_when_positions_collapse() {
    let cfg = tiny_cfg();
    let kv_len = 0usize;

    let g_mrope = trace_decode_mrope(cfg, kv_len);
    g_mrope.validate().expect("trace_decode_mrope valid");
    let mut mrope_inputs: Map<poot_graph_ir::ValueId, HostTensor> = Map::new();
    for &id in &g_mrope.inputs {
        let m = &g_mrope.values[id];
        match m.storage {
            Storage::Slot(Slot::Token) => {
                mrope_inputs.insert(id, HostTensor::i32(vec![], vec![5]));
            }
            Storage::Slot(Slot::MropePosition) => {
                mrope_inputs.insert(id, HostTensor::i32(vec![3], vec![0, 0, 0]));
            }
            Storage::Const => {
                let name = m.name.as_deref().unwrap();
                mrope_inputs.insert(id, weight(name, &m.aval.shape));
            }
            _ => panic!(
                "unexpected trace_decode_mrope input storage: {:?}",
                m.storage
            ),
        }
    }
    let got = (|| -> Result<HostTensor, EvalError> {
        let values: Map<_, Value> = mrope_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g_mrope, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();

    let g_plain = trace_decode(cfg, kv_len);
    g_plain.validate().expect("trace_decode valid");
    let mut plain_inputs: Map<poot_graph_ir::ValueId, HostTensor> = Map::new();
    for &id in &g_plain.inputs {
        let m = &g_plain.values[id];
        match m.storage {
            Storage::Slot(Slot::Token) => plain_inputs.insert(id, HostTensor::i32(vec![], vec![5])),
            Storage::Slot(Slot::Pos) => {
                plain_inputs.insert(id, HostTensor::i32(vec![], vec![kv_len as i32]))
            }
            Storage::Slot(Slot::SeqLen) => {
                plain_inputs.insert(id, HostTensor::i32(vec![], vec![(kv_len + 1) as i32]))
            }
            Storage::Const => {
                plain_inputs.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape))
            }
            _ => panic!("unexpected trace_decode input storage: {:?}", m.storage),
        };
    }
    let want = (|| -> Result<HostTensor, EvalError> {
        let values: Map<_, Value> = plain_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g_plain, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();

    assert_eq!(got.shape(), want.shape());
    let gb: Vec<u32> = got.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    let wb: Vec<u32> = want.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        gb, wb,
        "decode mRoPE with all three positions equal must be bit-exact to plain trace_decode"
    );
}

fn bind_batched_decode_mrope_oracle(
    g: &poot_graph_ir::Graph,
    tokens: &[u32],
    positions: &[usize],
    mrope_positions: Option<&[MropePosition]>,
) -> Map<poot_graph_ir::ValueId, HostTensor> {
    let batch = tokens.len();
    let mut inputs = Map::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let tensor = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(
                vec![batch],
                tokens.iter().map(|&token| token as i32).collect(),
            ),
            Storage::Slot(Slot::Pos) => HostTensor::i32(
                vec![batch, 1],
                positions.iter().map(|&position| position as i32).collect(),
            ),
            Storage::Slot(Slot::MropePosition) => {
                let positions = mrope_positions.expect("mRoPE graph needs typed batched positions");
                let mut packed = Vec::with_capacity(batch * 3);
                packed.extend(positions.iter().map(|position| position.temporal));
                packed.extend(positions.iter().map(|position| position.height));
                packed.extend(positions.iter().map(|position| position.width));
                HostTensor::i32(vec![3, batch], packed)
            }
            Storage::Const | Storage::State => weight(m.name.as_deref().unwrap(), &m.aval.shape),
            Storage::Computed(computed) => HostTensor::f32(computed.shape(), computed.values_f32()),
            Storage::Slot(other) => panic!("unexpected batched decode slot: {other:?}"),
            Storage::Device => panic!("unexpected device input in CPU oracle"),
        };
        inputs.insert(id, tensor);
    }
    inputs
}

#[test]
fn trace_batched_decode_mrope_is_cpu_coherent_and_collapses_bit_exact() {
    let cfg = tiny_cfg();
    let batch = 2usize;
    let cap = 4usize;
    let tokens = [5u32, 9u32];
    let absolute_positions = [1usize, 2usize];
    let collapsed = [MropePosition::collapsed(1), MropePosition::collapsed(2)];

    let graph = trace_decode_kv_masked_batched(cfg, cap, batch);
    graph
        .validate()
        .expect("batched mRoPE decode graph validates");
    let mrope_slots: Vec<_> = graph
        .slots
        .iter()
        .filter_map(|&(id, slot)| (slot == Slot::MropePosition).then_some(id))
        .collect();
    assert_eq!(mrope_slots.len(), 1);
    assert_eq!(graph.aval(mrope_slots[0]).shape, vec![3, batch]);
    assert_eq!(graph.aval(mrope_slots[0]).dtype, DType::I32);

    let collapsed_inputs =
        bind_batched_decode_mrope_oracle(&graph, &tokens, &absolute_positions, Some(&collapsed));
    let (got_logits, got_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: Map<_, Value> = collapsed_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    let mut plain_cfg = cfg;
    plain_cfg.mrope_section = None;
    let plain_graph = trace_decode_kv_masked_batched(plain_cfg, cap, batch);
    plain_graph
        .validate()
        .expect("ordinary batched decode graph validates");
    let plain_inputs =
        bind_batched_decode_mrope_oracle(&plain_graph, &tokens, &absolute_positions, None);
    let (want_logits, want_state) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: Map<_, Value> = plain_inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(
            &plain_graph,
            &values,
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();

    assert_eq!(got_logits.shape(), vec![batch, 1, cfg.vocab]);
    assert_eq!(got_logits.as_f32().unwrap(), want_logits.as_f32().unwrap());
    assert_eq!(got_state, want_state);

    let distinct = [
        MropePosition {
            temporal: 1,
            height: 3,
            width: 4,
        },
        MropePosition {
            temporal: 2,
            height: 5,
            width: 1,
        },
    ];
    let distinct_inputs =
        bind_batched_decode_mrope_oracle(&graph, &tokens, &absolute_positions, Some(&distinct));
    let (distinct_logits, distinct_state) =
        (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: Map<_, Value> = distinct_inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&graph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
    assert!(
        distinct_logits
            .as_f32()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(
        distinct_logits.as_f32().unwrap() != got_logits.as_f32().unwrap()
            || distinct_state != got_state,
        "distinct per-axis batched positions must reach Q/K rotation"
    );
}

//! Card 259 fusion pass: the flash-attention matcher's mask-shape gate admits the per-head
//! `[B,Hq,1,cap]` / `[1,Hq,L,L]` shapes as well as the broadcast ones, so `flash_attention_capped`
//! fuses the decomposed ALiBi decode chain to one `FlashAttentionDecode` per layer without changing
//! the logits.
//!
//! Card 626: moved here from `poot-eval/src/tests/alibi_attention.rs` with the pass itself -
//! poot-eval must never depend on poot-graph-plan (its own architecture test); this crate already
//! dev-depends on poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor_parity::dense::step;
use poot_graph_ir::op::OpKind;
use poot_graph_ir::{Graph, Slot, Storage, ValidationOutputs};
use poot_graph_plan::{cse, dce, flash_attention_capped};
use poot_models::model::{LogitRows, Phase};
use poot_models::registry::Registry;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Heads and layers of the MPT registry fixture (ALiBi, no RoPE, no GQA).
const HEADS: usize = 4;
const LAYERS: usize = 2;

/// ADR-0101 tier 1 (Card 556): the oracle evaluates the flash op from its decomposition, so the fused
/// graph's logits are the decomposed chain's, bit for bit. A non-finite element fails.
fn assert_bits_equal(got: &HostTensor, want: &HostTensor, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (got, want) = (got.as_f32().unwrap(), want.as_f32().unwrap());
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{what}: element {index} is not finite: {g} vs {w}"
        );
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {index}: fused {g} vs decomposed {w}"
        );
    }
}

/// Deterministic pseudo-random fill in [-1, 1), no rng dependency (poot-eval's own test helper of the
/// same name, duplicated: this is an external integration test, so it cannot reach poot-eval's
/// `pub(super)` test helpers).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The MPT registry fixture (ALiBi lives in the MPT and BLOOM families) over an F32 copy of its
/// checkpoint, traced for `phase` with `tokens` new tokens over `cap` positions.
fn mpt_graph(phase: Phase, tokens: usize, cap: usize) -> Graph<ValidationOutputs> {
    let registry = Registry::builtin().unwrap();
    let entry = registry
        .entries()
        .iter()
        .find(|e| e.family.as_str() == "mpt")
        .expect("the registry ships mpt");
    let fixture = (entry.fixture)();
    let mut f32_store = WeightStore::builder();
    for (key, stored) in fixture.store.iter() {
        let WeightEntry::Dense(_) = stored else {
            panic!("the fixture is dense")
        };
        let values = poot_eval::materialize_dense(&fixture.store, key.as_str())
            .unwrap()
            .to_f32()
            .unwrap()
            .into_owned();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = DenseWeight::try_new(DType::F32, stored.shape(), Arc::from(bytes)).unwrap();
        f32_store
            .insert(key.clone(), WeightEntry::Dense(dense))
            .unwrap();
    }
    let model = registry
        .build(&fixture.raw(), &f32_store.build())
        .expect("the mpt fixture builds");
    model
        .trace(phase, step(1, tokens, cap, LogitRows::Last))
        .unwrap()
}

/// Bind every input of `g`: tokens `i % 13`, positions from `first_pos`, constants filled by shape,
/// the ALiBi slopes (the graph's computed constant) as `slopes` when given, zeroed state.
fn bind(
    g: &Graph<ValidationOutputs>,
    first_pos: i32,
    slopes: Option<&[f32]>,
) -> HashMap<poot_graph_ir::ValueId, Value> {
    let mut inputs: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
    for id in g.inputs.iter().copied() {
        let m = &g.values[id];
        let numel = m.aval.shape.iter().product::<usize>().max(1);
        let t = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(
                m.aval.shape.clone(),
                (0..numel).map(|i| (i % 13) as i32).collect(),
            ),
            Storage::Slot(Slot::Pos) => HostTensor::i32(
                m.aval.shape.clone(),
                (0..numel as i32).map(|i| first_pos + i).collect(),
            ),
            Storage::Slot(other) => panic!("unexpected slot {other:?}"),
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()),
            Storage::Const if m.name.as_deref() == Some(SLOPES) => {
                HostTensor::f32(m.aval.shape.clone(), slopes.expect("bound slopes").to_vec())
            }
            Storage::Const => {
                // Small Q and K keep the scores near one, where the softmax still reads the ALiBi
                // distance term; at full scale it saturates and no slope shows.
                let gain = if m
                    .name
                    .as_deref()
                    .is_some_and(|n| n.ends_with(".attn.q") || n.ends_with(".attn.k"))
                {
                    0.05
                } else {
                    1.0
                };
                let values = fill(numel, 91).into_iter().map(|v| v * gain).collect();
                HostTensor::f32(m.aval.shape.clone(), values)
            }
            Storage::State => HostTensor::zeros(m.aval.shape.clone()),
            Storage::Device => continue,
        };
        inputs.insert(id, Value::from(t));
    }
    inputs
}

/// The name the slopes take once [`slopes_as_input`] makes them a bound constant.
const SLOPES: &str = "alibi.slopes";

/// `g` with its computed ALiBi slopes turned into a constant input, so a test can bind other values.
fn slopes_as_input(g: &Graph<ValidationOutputs>) -> Graph<ValidationOutputs> {
    let mut g = g.clone();
    for &id in &g.inputs {
        if matches!(g.values[id].storage, Storage::Computed(_)) {
            g.values[id].storage = Storage::Const;
            g.values[id].name = Some(SLOPES.to_string());
        }
    }
    g
}

fn logits(g: &Graph<ValidationOutputs>, first_pos: i32, slopes: Option<&[f32]>) -> HostTensor {
    let g = &if slopes.is_some() {
        slopes_as_input(g)
    } else {
        g.clone()
    };
    eval(
        g,
        &bind(g, first_pos, slopes),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("the graph evaluates")
    .output
    .into_host()
    .expect("a dense output")
}

fn count(g: &Graph<ValidationOutputs>, is: fn(&OpKind) -> bool) -> usize {
    g.eqns.iter().filter(|e| is(&e.op)).count()
}

#[test]
fn alibi_decode_chain_fuses_to_flash_without_changing_logits() {
    let (pos, cap) = (3usize, 8usize);
    let g = mpt_graph(Phase::Decode, 1, cap);
    let fused = dce(&flash_attention_capped(&cse(&g), None));
    let n_flash = count(&fused, |op| {
        matches!(op, OpKind::FlashAttentionDecode { .. })
    });
    assert_eq!(
        n_flash, LAYERS,
        "the ALiBi decode chain should fuse to one FlashAttentionDecode per layer"
    );
    let out_fused = logits(&fused, pos as i32, None);
    assert_bits_equal(
        &out_fused,
        &logits(&g, pos as i32, None),
        "ALiBi decode logits",
    );

    // The slopes must reach the fused op: zero slopes change the logits.
    let out_zero = logits(&fused, pos as i32, Some(&[0.0; HEADS]));
    let max_diff = out_fused
        .as_f32()
        .unwrap()
        .iter()
        .zip(out_zero.as_f32().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "the ALiBi slopes must change the flash-fused decode logits; max_diff={max_diff:.2e}"
    );
}

/// Prefill twin of [`alibi_decode_chain_fuses_to_flash_without_changing_logits`] (Card 557: it replaces
/// the poot-eval test that compared the deleted `trace_prefill_flash` tracer with `trace_prefill`):
/// the matcher fuses the decomposed prefill chain, fed the `[1,Hq,L,L]` per-head mask, to one
/// `FlashAttentionPrefill` per layer without changing the logits, and the slopes still reach the fused
/// op. The prefill is a whole-prompt step (capacity equals the token count), the only shape whose
/// mask is square.
#[test]
fn alibi_prefill_chain_fuses_to_flash_without_changing_logits() {
    let l = 5usize;
    let g = mpt_graph(Phase::Prefill, l, l);
    let fused = dce(&flash_attention_capped(&cse(&g), None));
    let n_flash = count(&fused, |op| {
        matches!(op, OpKind::FlashAttentionPrefill { .. })
    });
    assert_eq!(
        n_flash, LAYERS,
        "the ALiBi prefill chain should fuse to one FlashAttentionPrefill per layer"
    );

    let out_fused = logits(&fused, 0, None);
    assert_bits_equal(&out_fused, &logits(&g, 0, None), "ALiBi prefill logits");

    // The slopes must reach the fused op: zero slopes change the logits.
    let out_zero = logits(&fused, 0, Some(&[0.0; HEADS]));
    let max_diff = out_fused
        .as_f32()
        .unwrap()
        .iter()
        .zip(out_zero.as_f32().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-4,
        "the ALiBi slopes must change the flash-fused logits; max_diff={max_diff:.2e}"
    );
}

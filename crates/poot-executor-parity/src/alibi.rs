//! ALiBi attention over [`ComputedConst::AlibiSlopes`]: the graph BLOOM and MPT trace (the in-graph
//! `alibi_mask_from_pos` over a computed slope constant, then masked attention), run on a device
//! for a head count that is not a power of two (so the slope series' second half is exercised).
//!
//! Each fixture carries an f64 reference that does not read the slope constant: the published
//! slopes are written out as literals, so a wrong slope formula or zero slopes disagree with the
//! device output even though a device and the oracle both read the same constant.

use poot_graph_ir::ops::{alibi_mask_from_pos, attention_masked};
use poot_graph_ir::{Builder, ComputedConst, Graph, Slot, TensorType};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_tensor::{DType, HostTensor};

use poot_test_util::StepFixture;

use crate::{Fixture, step_fixtures, store_for};
use poot_graph_plan::FusionPolicy;

const HEADS: usize = 6;
const D: usize = 4;
const CAP: usize = 8;

/// The slopes of 6 heads by the published rule (BLOOM's `get_slopes`): the 4-head series
/// `2^-2, 2^-4, 2^-6, 2^-8`, then every other slope of the 8-head series, `2^-1` and `2^-3`.
const SLOPES: [f64; HEADS] = [0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125];

fn graph(tokens: usize) -> Graph {
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, HEADS, tokens, D]));
    let k = b.constant("k", TensorType::f32(vec![1, HEADS, CAP, D]));
    let v = b.constant("v", TensorType::f32(vec![1, HEADS, CAP, D]));
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, tokens], DType::I32));
    let slopes = b.computed(ComputedConst::AlibiSlopes { heads: HEADS });
    let mask = alibi_mask_from_pos(&b, pos, CAP, None, slopes, HEADS);
    let y = attention_masked(&b, q, k, v, 1, 1.0 / (D as f32).sqrt(), mask);
    b.finish(y)
}

fn fixture(name: &'static str, tokens: usize, starts: &[usize]) -> Fixture {
    let graph = graph(tokens);
    let store = store_for(&graph);
    let steps = starts
        .iter()
        .map(|&start| step_fixtures(&graph, &vec![0; tokens], start))
        .collect();
    Fixture {
        name,
        graph,
        store,
        steps,
        fusion: FusionPolicy::Full,
    }
}

/// ALiBi decode (one query per step at positions 2, 5 and 7 of an 8-slot cache) and a 4-token
/// prefill from position 0.
pub fn alibi_slopes_fixtures() -> Vec<Fixture> {
    vec![
        fixture("alibi_slopes_decode", 1, &[2, 5, 7]),
        fixture("alibi_slopes_prefill", 4, &[0]),
    ]
}

fn stored_f32(store: &WeightStore, name: &str) -> Vec<f64> {
    let Some(WeightEntry::Dense(dense)) = store.get(name) else {
        panic!("fixture store has no dense `{name}`");
    };
    dense
        .bytes()
        .as_slice()
        .chunks_exact(4)
        .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}

fn positions(step: &[StepFixture]) -> Vec<usize> {
    // The graph's one slot is `Slot::Pos`.
    let [pos] = step else {
        panic!("an ALiBi step carries only its positions");
    };
    pos.tensor
        .view()
        .bytes()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect()
}

/// The expected output of every step of `fixture`, `[1, heads, tokens, head_dim]` flattened, from
/// an f64 softmax over the causal keys with the published slopes (`score - slope * distance`).
pub fn alibi_slopes_reference(fixture: &Fixture) -> Vec<Vec<f64>> {
    let (q, k, v) = (
        stored_f32(&fixture.store, "q"),
        stored_f32(&fixture.store, "k"),
        stored_f32(&fixture.store, "v"),
    );
    let scale = 1.0 / (D as f64).sqrt();
    fixture
        .steps
        .iter()
        .map(|step| {
            let pos = positions(step);
            let tokens = pos.len();
            let mut out = vec![0.0; HEADS * tokens * D];
            for h in 0..HEADS {
                for (i, &p) in pos.iter().enumerate() {
                    let qi = &q[(h * tokens + i) * D..][..D];
                    let scores: Vec<f64> = (0..=p)
                        .map(|j| {
                            let kj = &k[(h * CAP + j) * D..][..D];
                            let dot: f64 = qi.iter().zip(kj).map(|(a, b)| a * b).sum();
                            dot * scale - SLOPES[h] * (p - j) as f64
                        })
                        .collect();
                    let max = scores.iter().cloned().fold(f64::MIN, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    for (j, e) in exps.iter().enumerate() {
                        for d in 0..D {
                            out[(h * tokens + i) * D + d] += e / sum * v[(h * CAP + j) * D + d];
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// Assert each device output of `fixture` equals [`alibi_slopes_reference`] within `1e-4` relative
/// tolerance (ADR-0101 tier 2).
pub fn assert_matches_alibi_reference(fixture: &Fixture, outputs: &[HostTensor]) {
    let reference = alibi_slopes_reference(fixture);
    assert_eq!(
        outputs.len(),
        reference.len(),
        "{}: step count",
        fixture.name
    );
    for (i, (got, want)) in outputs.iter().zip(&reference).enumerate() {
        let got = got.to_f32().expect("f32 output");
        assert_eq!(got.len(), want.len(), "{}: step {i} length", fixture.name);
        for (e, (&g, &w)) in got.iter().zip(want).enumerate() {
            assert!(
                (f64::from(g) - w).abs() <= 1e-4 * (1.0 + w.abs()),
                "{}: step {i} element {e}: device {g}, reference {w}",
                fixture.name
            );
        }
    }
}

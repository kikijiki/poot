//! POOT-1268: the sampler suffix that adjusts the logits (penalties, bias, a guided mask), draws, records
//! top-n logprobs and keeps the per-sequence history on the device, planned and run by every executor.
//!
//! The reference is the CPU oracle evaluating the same composed graph step after step (it carries the
//! history pool between steps like the device does). What the suffix computes is anchored to the exact host
//! sampler where the suffix is built (`poot-llm`'s sampler tests); this module holds each lowering to the
//! oracle: the greedy head exactly (the token, the non-finite flag and the alternatives' ids), its F32
//! logprobs within the tier-2 transcendental tolerance, and the stochastic head's draw exactly. A discarded
//! write (`-1` in the write map) must leave its pool row as it was, which the next step's penalties show.

use std::num::NonZeroUsize;

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::Executor;
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::{
    Adjust, HISTORY_WRITE_TAG, LogprobRecord, MASKED_LOGIT, SampleSpec, SamplerInput, sample_head,
};
use poot_graph_ir::{Builder, Outputs, Slot, SlotKey, TensorType};
use poot_graph_plan::Target;
use poot_quant::weights::WeightStore;
use poot_tensor::{DType, HostTensor};

use crate::{Fixture, run_outputs};

const ROWS: usize = 3;
const VOCAB: usize = 1200;
const POOL_ROWS: usize = 4;
const STEPS: usize = 3;
/// The alternatives a logprob record carries.
const TOP_N: usize = 4;

/// A scrambled, tie-free logit row.
fn logits(row: usize, step: usize) -> Vec<f32> {
    let salt = (row * 7919 + step * 104_729) as u32;
    (0..VOCAB as u32)
        .map(|i| {
            ((i.wrapping_mul(2_654_435_761).wrapping_add(salt) >> 9) % 8192) as f32 * 0.0013 - 5.0
        })
        .collect()
}

/// One step's slot values. Row 0 carries penalties, row 1 a mask that excludes every third token, row 2 is
/// neutral. Pool row `r` is read by step row `r`; the last step drops row 2's write.
fn step(spec: SampleSpec, graph: &poot_graph_ir::Graph, index: usize) -> Vec<StepFixture> {
    let key = |kind: Slot, tag: &str| SlotKey::new(kind, Some(tag));
    let put = |key: SlotKey, tensor: HostTensor| {
        assert!(
            graph
                .slots()
                .iter()
                .any(|&(id, _)| graph.meta(id).slot_key() == Some(&key)),
            "the head declares no slot {key}"
        );
        StepFixture { key, tensor }
    };
    let mut fixtures = vec![put(
        key(Slot::Activation, "logits"),
        HostTensor::f32(
            vec![ROWS, VOCAB],
            (0..ROWS).flat_map(|r| logits(r, index)).collect(),
        ),
    )];
    if spec.rule != SampleRule::Greedy {
        let cols = spec.rule.params_cols();
        let row = [1.0 / 0.8, f32::NEG_INFINITY, 1.0, 0.9];
        fixtures.push(put(
            SamplerInput::Seed.key(),
            HostTensor::i32(
                vec![ROWS],
                (0..ROWS as i32).map(|r| 31 * index as i32 + r).collect(),
            ),
        ));
        fixtures.push(put(
            SamplerInput::Params.key(),
            HostTensor::f32(vec![ROWS, cols], vec![row[..cols].to_vec(); ROWS].concat()),
        ));
        if spec.rule.has_top_k() {
            fixtures.push(put(
                SamplerInput::TopK.key(),
                HostTensor::i32(vec![ROWS], vec![0; ROWS]),
            ));
        }
    }
    if spec.adjust != Adjust::None {
        let penalty = [1.3, 0.2, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mask: Vec<f32> = (0..ROWS)
            .flat_map(|r| {
                (0..VOCAB).map(move |t| {
                    if r == 1 && t % 3 != 0 {
                        MASKED_LOGIT
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let write: Vec<i32> = (0..ROWS as i32)
            .map(|r| if index + 1 == STEPS && r == 2 { -1 } else { r })
            .collect();
        fixtures.push(put(
            SamplerInput::Penalty.key(),
            HostTensor::f32(vec![ROWS, 3], penalty.to_vec()),
        ));
        if spec.adjust == Adjust::Guided {
            fixtures.push(put(
                SamplerInput::Mask.key(),
                HostTensor::f32(vec![ROWS, VOCAB], mask),
            ));
        }
        fixtures.push(put(
            SamplerInput::HistoryRead.key(),
            HostTensor::i32(vec![ROWS], (0..ROWS as i32).collect()),
        ));
        fixtures.push(put(
            key(Slot::WriteMap, HISTORY_WRITE_TAG),
            HostTensor::i32(vec![ROWS], write),
        ));
    }
    fixtures
}

fn fixture(name: &'static str, spec: SampleSpec) -> Fixture {
    let b = Builder::for_tests();
    let raw = b
        .slot_named(
            Slot::Activation,
            "logits",
            TensorType::new(vec![ROWS, VOCAB], DType::F32),
        )
        .unwrap();
    let pool = spec
        .adjust
        .ne(&Adjust::None)
        .then(|| NonZeroUsize::new(POOL_ROWS).expect("a pool has rows"));
    let head = sample_head(&b, raw, spec, pool).unwrap();
    let mut outputs = Outputs::new(head.out);
    if let Some(pair) = head.history {
        outputs = outputs.with_state(&[pair]);
    }
    let graph = b.finish(outputs).unwrap();
    let steps = (0..STEPS).map(|i| step(spec, &graph, i)).collect();
    Fixture {
        name,
        graph,
        store: WeightStore::builder().build(),
        steps,
    }
}

/// The cases: the greedy head with penalties, mask and top-n logprobs, and the stochastic top-p head with
/// penalties and a mask.
pub fn cases() -> Vec<Fixture> {
    vec![
        fixture(
            "adjusted_greedy_with_logprobs",
            SampleSpec::draw(SampleRule::Greedy)
                .guided()
                .with_logprobs(TOP_N),
        ),
        fixture(
            "adjusted_top_p_draw",
            SampleSpec::draw(SampleRule::GumbelTopKTopP).adjusted(),
        ),
    ]
}

/// Run every case on `exec` and hold each step's record to the oracle's. Index columns and the I32
/// `(token, flag)` pairs are equal; logprobs agree to `1e-3` absolute (a device `exp` and `log` against the
/// oracle's, tier 2).
pub fn assert_matches_the_oracle(exec: &mut dyn Executor, target: Target) {
    for fixture in cases() {
        let name = fixture.name;
        let outputs = run_outputs(exec, target, &fixture).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        for (index, (step, got)) in fixture.steps.iter().zip(&outputs).enumerate() {
            let want = oracle(step);
            match (got.as_i32(), want.as_i32()) {
                (Some(got), Some(want)) => {
                    assert_eq!(
                        got, want,
                        "{name} step {index}: (token, non_finite) per row"
                    );
                }
                _ => {
                    let width = LogprobRecord { n: TOP_N }.width();
                    let (got, want) = (got.as_f32().unwrap(), want.as_f32().unwrap());
                    for (row, (g, w)) in got.chunks(width).zip(want.chunks(width)).enumerate() {
                        for (col, (g, w)) in g.iter().zip(w).enumerate() {
                            let is_index = col < 2
                                || (col >= LogprobRecord::HEADER
                                    && (col - LogprobRecord::HEADER).is_multiple_of(2));
                            if is_index {
                                assert_eq!(g, w, "{name} step {index} row {row} column {col}");
                            } else {
                                assert!(
                                    (g - w).abs() <= 1e-3,
                                    "{name} step {index} row {row} column {col}: {g} vs oracle {w}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_plan::Target;
    use poot_target::{AmdArch, Backend};
    use poot_test_util::device_caps::default_caps_for;

    use super::*;

    /// Support is decided by planning: every case compiles for each GPU backend, no device needed. A
    /// primitive the composition used without a lowering (a float-to-integer cast was one) is a typed
    /// refusal here, not a failure on the first machine that has the backend.
    ///
    /// Mutation: in `ops/sampling.rs`'s `blend` build the flag through `b.cast(flag, DType::I32)` and a
    /// `select`; planning refuses with `DtypeLowering` (a float-to-integer cast has no lowering).
    #[test]
    fn every_case_plans_on_every_gpu_backend() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::Nvptx,
        ] {
            let target = Target::Gpu {
                backend,
                caps: default_caps_for(backend),
            };
            for case in cases() {
                crate::staged(&case, target);
            }
        }
    }

    /// The cases mean something on the oracle: the history the first step wrote changes the penalized row's
    /// pick on the second (the same logits, a seen token now penalized), the masked row only ever picks a
    /// token the mask allows, and the unmasked neutral row is unaffected by its neighbours.
    ///
    /// Mutation: in `Adjusted::carry` return the old pool as the updated one; the second record of the
    /// penalized row equals the first and the case fails.
    #[test]
    fn the_history_a_step_writes_changes_the_next_picks() {
        let case = &cases()[0];
        let mut oracle = oracle_for(&case.graph, &case.store);
        let width = LogprobRecord { n: TOP_N }.width();
        let tokens: Vec<Vec<usize>> = case
            .steps
            .iter()
            .map(|step| {
                let record = oracle(step);
                record
                    .as_f32()
                    .unwrap()
                    .chunks(width)
                    .map(|row| row[0] as usize)
                    .collect()
            })
            .collect();
        for step in &tokens {
            assert_eq!(step[1] % 3, 0, "row 1 is masked to every third token");
        }
        // Row 0 sees fresh logits every step; replay step 0's logits after its pick entered the history.
        let mut second = oracle_for(&case.graph, &case.store);
        let first = second(&case.steps[0]);
        let again = second(&case.steps[0]);
        assert_ne!(
            first.as_f32().unwrap()[0..width],
            again.as_f32().unwrap()[0..width],
            "the penalized row's record is the same after its pick entered the history"
        );
    }
}

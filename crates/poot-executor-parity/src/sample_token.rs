//! POOT-1146: the greedy `SampleToken` pick is exact on every backend: the lowest index among equal
//! maxima, whatever chunk or workgroup the equal values fall in, and the non-finite policy of the oracle
//! (the token is 0 and the flag names the lowest non-finite index whenever the row holds a NaN or an
//! infinity).
//!
//! Each case carries the `(token, non_finite_index)` pair a row must produce, derived here by
//! [`greedy_pair`] (a one-pass scan written from the contract, never the oracle or a lowering), so a row
//! asserts the selection itself and not only agreement with `eval`. The cases put the fault classes of a
//! split reduction at the seams a chunked scan has: a tie across a 1024-element chunk boundary, a tie
//! between the first and the last element, signed zeros, a non-finite value at a chunk boundary and in the
//! last (partial) chunk, and the whole production vocabulary.

use poot_executor::Executor;
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::Target;
use poot_tensor::HostTensor;

use crate::{Fixture, const_fixture, run_outputs};

/// The vocabulary of the Qwen2.5 family, the width the decode step's greedy pick runs over.
const PRODUCTION_VOCAB: usize = 151_936;

/// One `[rows, vocab]` logits fixture and the `[rows, 2]` pairs its greedy pick must produce.
pub struct GreedyCase {
    pub fixture: Fixture,
    /// `(token, non_finite_index)` per row, as [`greedy_pair`] defines them.
    pub expected: Vec<(i32, i32)>,
}

/// The contract of `SampleToken { Greedy }` over one row: the token is the lowest index holding the
/// largest finite value (0 when there is none), the flag is the lowest non-finite index (`-1` when the
/// row is finite), and any non-finite value forces the token to 0.
pub fn greedy_pair(row: &[f32]) -> (i32, i32) {
    let mut best: Option<(usize, f32)> = None;
    let mut non_finite = None;
    for (i, &v) in row.iter().enumerate() {
        if v.is_finite() {
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((i, v));
            }
        } else if non_finite.is_none() {
            non_finite = Some(i);
        }
    }
    match non_finite {
        Some(i) => (0, i as i32),
        None => (best.map_or(0, |(i, _)| i as i32), -1),
    }
}

fn ramp(vocab: usize) -> Vec<f32> {
    // Strictly decreasing, so index 0 is the unique maximum until a case plants a larger value.
    (0..vocab).map(|i| -(i as f32) * 1e-3).collect()
}

fn with(mut row: Vec<f32>, edits: &[(usize, f32)]) -> Vec<f32> {
    for &(i, v) in edits {
        row[i] = v;
    }
    row
}

fn case(name: &'static str, vocab: usize, rows: Vec<Vec<f32>>) -> GreedyCase {
    assert!(rows.iter().all(|row| row.len() == vocab), "{name}: ragged");
    let expected: Vec<(i32, i32)> = rows.iter().map(|row| greedy_pair(row)).collect();
    let logits: Vec<f32> = rows.into_iter().flatten().collect();
    let n = expected.len();
    let b = Builder::for_tests();
    let x = b
        .constant("logits", TensorType::f32(vec![n, vocab]))
        .unwrap();
    let out = b
        .sample_token(SampleRule::Greedy, x, None, None, None)
        .unwrap();
    GreedyCase {
        fixture: const_fixture(
            name,
            b.finish(out).unwrap(),
            vec![("logits", HostTensor::f32(vec![n, vocab], logits))],
            2,
        ),
        expected,
    }
}

/// The cases. `5000` leaves a partial last chunk of a 1024-wide chunking; `1000` is a single chunk;
/// [`PRODUCTION_VOCAB`] is the decode step's own width.
pub fn greedy_cases() -> Vec<GreedyCase> {
    let ramp_5000 = ramp(5000);
    let flat_5000 = vec![0.25f32; 5000];
    vec![
        case(
            "greedy_unique_maxima",
            5000,
            vec![
                with(ramp_5000.clone(), &[(0, 1.0)]),
                with(ramp_5000.clone(), &[(2500, 1.0)]),
                with(ramp_5000.clone(), &[(4999, 1.0)]),
                with(ramp_5000.clone(), &[(1024, 1.0)]),
            ],
        ),
        case(
            "greedy_tie_across_a_chunk_boundary",
            5000,
            vec![
                with(ramp_5000.clone(), &[(1023, 2.0), (1024, 2.0)]),
                with(ramp_5000.clone(), &[(1024, 2.0), (1023, 2.0), (3000, 2.0)]),
                with(ramp_5000.clone(), &[(2047, 2.0), (2048, 2.0), (4999, 2.0)]),
                with(ramp_5000.clone(), &[(0, 2.0), (4999, 2.0)]),
                with(ramp_5000.clone(), &[(4998, 2.0), (4999, 2.0)]),
            ],
        ),
        case(
            "greedy_all_equal_and_signed_zero",
            5000,
            vec![
                flat_5000.clone(),
                vec![f32::MIN; 5000],
                with(vec![-0.0; 5000], &[(3000, 0.0), (2000, 0.0)]),
                with(vec![0.0; 5000], &[(4000, -0.0), (10, -0.0)]),
                with(
                    vec![-1.0; 5000],
                    &[(1500, f32::MIN), (4500, -0.5), (900, -0.5)],
                ),
            ],
        ),
        case(
            "greedy_non_finite_row_forces_token_zero",
            5000,
            vec![
                with(
                    ramp_5000.clone(),
                    &[(4000, f32::NAN), (10, f32::INFINITY), (2500, 9.0)],
                ),
                with(ramp_5000.clone(), &[(4999, f32::NAN), (2500, 9.0)]),
                with(ramp_5000.clone(), &[(1024, f32::NEG_INFINITY), (2500, 9.0)]),
                with(
                    ramp_5000.clone(),
                    &[(1023, f32::NAN), (1024, f32::NAN), (2500, 9.0)],
                ),
                vec![f32::NAN; 5000],
                with(
                    ramp_5000.clone(),
                    &[(3000, f32::INFINITY), (3001, f32::NAN)],
                ),
            ],
        ),
        case(
            "greedy_single_chunk_ties",
            1000,
            vec![
                with(ramp(1000), &[(10, 3.0), (999, 3.0), (500, 3.0)]),
                with(ramp(1000), &[(999, 3.0)]),
                with(ramp(1000), &[(63, f32::NAN)]),
            ],
        ),
        case(
            "greedy_production_vocab",
            PRODUCTION_VOCAB,
            vec![
                with(
                    ramp(PRODUCTION_VOCAB),
                    &[(100_000, 5.0), (151_935, 5.0), (70_000, 5.0)],
                ),
                with(ramp(PRODUCTION_VOCAB), &[(151_935, 5.0)]),
                with(ramp(PRODUCTION_VOCAB), &[(65_536, 5.0), (65_535, 5.0)]),
                with(
                    ramp(PRODUCTION_VOCAB),
                    &[(151_900, f32::NAN), (70_000, 5.0)],
                ),
            ],
        ),
    ]
}

/// Run every case through `exec` for `target`: each step's `[rows, 2]` I32 output equals the pair the
/// contract defines (the CPU oracle is pinned to the same pair by this module's own test).
pub fn assert_greedy_pick_is_exact(exec: &mut dyn Executor, target: Target) {
    for GreedyCase { fixture, expected } in greedy_cases() {
        let name = fixture.name;
        let want: Vec<i32> = expected.iter().flat_map(|&(t, b)| [t, b]).collect();
        let outputs = run_outputs(exec, target, &fixture).unwrap_or_else(|e| panic!("{name}: {e}"));
        for (step, got) in outputs.iter().enumerate() {
            let got = got.as_i32().expect("SampleToken publishes I32 pairs");
            assert_eq!(
                got,
                want.as_slice(),
                "{name} step {step}: (token, non_finite_index) per row"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use poot_eval::test_support::weight_map_oracle::oracle_for;

    use super::*;

    /// The CPU oracle publishes, for every case, the pair [`greedy_pair`] defines: lowest index among
    /// equal maxima across chunk seams and signed zeros, token 0 and the lowest non-finite index on a
    /// non-finite row. This pins the semantics a lowering must reproduce without a device, so a device
    /// row's expectation and the oracle cannot drift apart unseen.
    ///
    /// Mutation: break ties to the later index in [`greedy_pair`] (`v > b` to `v >= b`); the tie cases
    /// fail naming the row.
    #[test]
    fn the_oracle_publishes_the_pair_the_contract_defines() {
        for GreedyCase { fixture, expected } in greedy_cases() {
            let want: Vec<i32> = expected.iter().flat_map(|&(t, b)| [t, b]).collect();
            let got = oracle_for(&fixture.graph, &fixture.store)(&fixture.steps[0]);
            assert_eq!(
                got.as_i32().expect("I32 pairs"),
                want.as_slice(),
                "{}",
                fixture.name
            );
        }
    }
}

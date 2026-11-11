//! Stable rank fixtures (POOT-1543): the router's rank over scores with ties, signed zeros, NaN, infinities and
//! extents on both sides of every padding boundary, plus masked ranks, each built as the `StableRank` op itself
//! so the device runs the sort kernel, compared on every executor with the plain sort of
//! `poot_graph_ir::test_support::stable_rank` and with the oracle's primitive definition.

use poot_graph_ir::test_support::composites;
use poot_graph_ir::test_support::stable_rank::{eligibility_rows, reference_rank, score_rows};
use poot_graph_ir::{Builder, TensorType};
use poot_graph_plan::{KernelChoice, Target};
use poot_tensor::HostTensor;

use crate::{Fixture, const_fixture};

/// One case: the rank program over constant scores (and eligibility), and the ranks it must produce.
pub struct StableRankCase {
    /// The `StableRank` program and its constants.
    pub fixture: Fixture,
    /// The `[rows, e]` ranks `fixture` must produce, row-major.
    pub expected: Vec<f32>,
}

/// The cases:
///
/// - `pair`: two experts, the extent with one compare-exchange stage (one expert ranks 0 on every input, which the
///   oracle comparison refuses as a fixture that cannot show a wrong device; the kernel's own rows cover it).
/// - `ties`: five experts, where every row of the router's tie cases ties across the cut.
/// - `experts_512`: the Flash-Next router width, one full workgroup of lanes.
/// - `experts_513`: one past it, padded to 1024 so a lane owns two pairs of a stage.
/// - `masked_64`, `masked_512`: an eligibility that is `1.0`, `2.0`, `0.0`, `0.5` and NaN.
pub fn cases() -> Vec<StableRankCase> {
    vec![
        case("stable_rank_pair", 2, false),
        case("stable_rank_ties", 5, false),
        case("stable_rank_experts_512", 512, false),
        case("stable_rank_experts_513", 513, false),
        case("stable_rank_masked_64", 64, true),
        case("stable_rank_masked_512", 512, true),
    ]
}

fn case(name: &'static str, experts: usize, masked: bool) -> StableRankCase {
    let scores = score_rows(experts);
    let rows = scores.len();
    let eligible = masked.then(|| eligibility_rows(rows, experts));
    let expected = scores
        .iter()
        .enumerate()
        .flat_map(|(r, row)| reference_rank(row, eligible.as_ref().map(|mask| mask[r].as_slice())))
        .collect();
    let b = Builder::for_tests();
    let ty = TensorType::f32(vec![rows, experts]);
    let scores_in = b.constant("scores", ty.clone()).unwrap();
    let eligible_in = masked.then(|| b.constant("eligible", ty.clone()).unwrap());
    let rank = composites::stable_rank(&b, scores_in, eligible_in).unwrap();
    let graph = b.finish(rank).unwrap();
    let mut consts = vec![(
        "scores",
        HostTensor::f32(vec![rows, experts], scores.concat()),
    )];
    if let Some(eligible) = &eligible {
        consts.push((
            "eligible",
            HostTensor::f32(vec![rows, experts], eligible.concat()),
        ));
    }
    StableRankCase {
        fixture: const_fixture(name, graph, consts, 2),
        expected,
    }
}

/// `fixture`'s program, compiled for `target`, runs its one rank through the generated stable rank kernel, so a
/// row shows the kernel it ran is the sort and not only that the words matched.
pub fn assert_runs_the_kernel(fixture: &Fixture, target: Target) {
    let staged = crate::staged(fixture, target);
    let kernels = staged
        .stages()
        .flat_map(|(_, _, program)| {
            program.planned().filter(move |(eqn, _)| {
                matches!(
                    program.kernel_choice(eqn),
                    KernelChoice::Generated(request) if request.family() == "stable_rank"
                )
            })
        })
        .count();
    assert_eq!(kernels, 1, "{}: the rank runs the kernel", fixture.name);
}

/// Run `case` on `exec`: the program ran the kernel, and the device's ranks equal the plain sort's, bit for bit,
/// at every step. The caller then compares the device with the oracle's definition through `run_parity`.
pub fn assert_device_ranks(
    exec: &mut dyn poot_executor::Executor,
    target: Target,
    case: &StableRankCase,
) {
    let name = case.fixture.name;
    assert_runs_the_kernel(&case.fixture, target);
    let outputs = crate::run_outputs(exec, target, &case.fixture).unwrap_or_else(|e| panic!("{e}"));
    for (step, got) in outputs.iter().enumerate() {
        let got = got.as_f32().expect("a rank is F32");
        assert_eq!(
            got.len(),
            case.expected.len(),
            "{name} step {step}: result length"
        );
        if let Some(at) = got
            .iter()
            .zip(&case.expected)
            .position(|(g, w)| g.to_bits() != w.to_bits())
        {
            panic!(
                "{name} step {step}: element {at} of the ranks: the device has {}, the plain sort {}",
                got[at], case.expected[at]
            );
        }
    }
}

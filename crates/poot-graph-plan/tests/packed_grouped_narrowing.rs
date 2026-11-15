//! POOT-1547: a routed projection over a packed expert stack narrows where the grouped matrix-unit schedule, with the
//! grouping of its rows by expert, costs less than the per-row Gemv the exact twin runs, and stays exact where it does
//! not. The bank's `expert-prefill-packed-iq2xs-up` pair measured the narrow twin 7.3 % slower than the exact one on
//! ROCm (gfx1151) at 128 tokens, because the grouping dispatch (one workgroup scanning the row ids, about 0.27 us a
//! row) and the grouped fragment round were priced by no term of the cost model: the model chose the grouped schedule
//! for a contraction whose Gemv is faster. Every row drives the production `compile` on the gfx1151 caps and reads
//! the program it holds.

use poot_graph_ir::test_support::packed_graphs::{
    PackedExpertStack, TEST_STACK, packed_indexed_linear,
};
use poot_graph_ir::{Builder, Graph, IndexRange, MatmulPrecision, OpKind, Slot, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest, RegisterTiling, Schedule};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::calibration::Calibrated;
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried};
use poot_tensor::DType;

fn rocm() -> Target {
    let backend = Backend::AmdGcn(AmdArch::gfx1151());
    Target::Gpu {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    }
}

/// The up projection of the bank's wide mixture (512 experts, K 2560, N 640, top 10).
const UP: (usize, usize) = (640, 2560);
/// The down projection of the same mixture (K 640, N 2560).
const DOWN: (usize, usize) = (2560, 640);

/// A projection `(out, k)` of the bank's wide mixture (512 experts, top 10) over `tokens` tokens: one per-row indexed
/// packed contraction of `tokens * 10` routed rows.
fn routed_projection(format: WeightFormat, (out, k): (usize, usize), tokens: usize) -> Graph {
    let (rows, experts) = (tokens * 10, 512);
    let table = PackedExpertStack {
        stack: TEST_STACK,
        descriptor: PackedWeight::try_new(format, [out, k]).unwrap(),
        experts,
    };
    let builder = Builder::for_tests();
    let x = builder
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![rows, k]))
        .unwrap();
    let ids = builder
        .constant_ranged(
            "ids",
            TensorType::new(vec![rows], DType::I32),
            IndexRange::below(experts).unwrap(),
        )
        .unwrap();
    let y = packed_indexed_linear(&builder, x, ids, &table).unwrap();
    builder.finish(y).unwrap()
}

fn compiled(tokens: usize) -> Program {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Narrow);
    compile(
        &routed_projection(WeightFormat::Iq2_Xs, UP, tokens),
        &rocm(),
        &options,
    )
    .expect("the projection compiles")
}

/// An RTX 4090 (sm_89, 128 SMs) as the PTX runtime reports it, with the calibration `calibrated`.
fn rtx_4090(calibrated: Calibrated) -> Target {
    let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
    Target::Gpu {
        backend: Backend::Nvptx,
        caps: DeviceCaps {
            matrix_units: Queried::Known(MatrixUnits::nvidia((8, 9))),
            compute_units: Queried::Known(128),
            perf_calibration: calibrated,
            ..caps
        },
    }
}

/// The program the planner chooses for `format` over `projection` at `tokens` on `target`, its precision its own (the
/// default plan of `feedback_end_to_end`).
fn planned(
    format: WeightFormat,
    projection: (usize, usize),
    tokens: usize,
    target: &Target,
) -> Program {
    compile(
        &routed_projection(format, projection, tokens),
        target,
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect("the projection compiles")
}

/// The schedule of the program's packed contraction.
fn packed_schedule(program: &Program) -> Schedule {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
        .expect("the contraction is planned");
    let KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) =
        program.kernel_choice(eqn)
    else {
        panic!("expected a generated packed contraction");
    };
    spec.schedule
}

/// The tiling the program's packed contraction runs and whether it groups its rows by expert: `None` for the
/// per-row Gemv.
fn grouped_tiling(program: &Program) -> Option<RegisterTiling> {
    let grouped = program
        .planned()
        .any(|(eqn, _)| matches!(eqn.op, OpKind::ExpertGroup { .. }));
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::PackedContraction { .. }))
        .expect("the contraction is planned");
    let KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Packed(spec))) =
        program.kernel_choice(eqn)
    else {
        panic!("expected a generated packed contraction");
    };
    let tiling = RegisterTiling::of(spec.schedule);
    assert_eq!(
        grouped,
        tiling.is_some(),
        "rows are grouped exactly when a tiling runs them"
    );
    tiling
}

/// The 128-token prompt of the bank's `expert-prefill-packed-iq2xs-up-narrow` pair (1280 routed rows over 512
/// experts, about 2.5 rows an expert) narrows to the 16x32 fragment tiling at depth 64, the fastest of the 31 the
/// device runs (4.16 ms of contraction), not the 16x64 tiling at depth 32 (5.05 ms) the model priced the same: the
/// grouped fragment tilings tie in the model (twice the columns at half the depth is the same lane step over the same
/// lanes), and the tie goes to the deeper round, which runs fewer barriers. With the grouping dispatch (0.38 ms) the
/// narrow twin takes 4.5 ms against the exact twin's per-row Gemv at 5.3 ms. Mutation: break the tie to the larger
/// output tile first (swap the two `then` keys of `RegisterTiling::cheapest`); the plan is the depth-32 tiling.
#[test]
fn a_tie_between_grouped_fragment_tilings_goes_to_the_deeper_round() {
    let tiling = grouped_tiling(&compiled(128)).expect("128 tokens narrow and group");
    assert_eq!(
        (tiling.output_tile(), tiling.depth()),
        ((16, 32), 64),
        "{tiling:?}"
    );
}

/// The decision between the per-row Gemv and the grouped matrix-unit schedule is the cost model's, grouping included,
/// not "always narrow": 96 tokens (960 routed rows) stay the Gemv, 128 and up narrow with their grouping (the device
/// measures the grouped schedule 14 % ahead at 128 and by a wide margin above: 5.8 ms against the Gemv's 20.6 at 512
/// tokens, 15.0 against 84.8 at 2048). Without the grouping dispatch priced the boundary falls to 96. Mutation: drop
/// the `group_scan` term of `CostTerms::modelled_time`; 96 tokens group their rows.
#[test]
fn the_decision_between_the_gemv_and_the_grouped_tiling_prices_the_grouping() {
    assert!(
        grouped_tiling(&compiled(96)).is_none(),
        "96 tokens: the Gemv"
    );
    for tokens in [128, 512, 2048] {
        assert!(
            grouped_tiling(&compiled(tokens)).is_some(),
            "{tokens} tokens: the grouped matrix-unit schedule wins"
        );
    }
}

/// POOT-1602: a Gemv's dot products cost what does not scale with `K` (`CostRatios::gemv_dot`), so a short-`K` Gemv
/// loses to a tile its long-`K` twin beats. On an RTX 4090 the bank's two packed projections at 128 tokens do the same
/// work (1280 routed rows, K x N = 2560 x 640 against 640 x 2560) and the device ranks them oppositely
/// (`benchmarks/calibration/ptx-sm89.tsv`): the IQ2_XS up projection's Gemv takes 1360 us against 1584 us for the best
/// grouped narrowed tiling, the IQ4_NL down projection's Gemv 1667 us against 1144 us. The sm_89 entry plans the Gemv
/// for the first and groups the second. Mutation: drop the `dots_per_k * gemv_dot` term of
/// `CostTerms::modelled_time`; the down projection plans the Gemv. Resolve sm_89 to the portable prior; the up
/// projection groups its rows.
#[test]
fn a_short_contraction_gives_its_gemv_up_to_the_grouped_tiling_on_sm_89() {
    let sm_89 = rtx_4090(Calibrated::of_cuda_compute_capability(8, 9));
    assert!(
        grouped_tiling(&planned(WeightFormat::Iq2_Xs, UP, 128, &sm_89)).is_none(),
        "IQ2_XS up, K 2560: the Gemv"
    );
    let down = planned(WeightFormat::Iq4_Nl, DOWN, 128, &sm_89);
    assert!(
        grouped_tiling(&down).is_some(),
        "IQ4_NL down, K 640: the rows are grouped"
    );
    let schedule = packed_schedule(&down);
    assert!(
        matches!(schedule, Schedule::FragmentTiled { .. }),
        "IQ4_NL down, K 640: the grouped tiling narrows: {schedule:?}"
    );
}

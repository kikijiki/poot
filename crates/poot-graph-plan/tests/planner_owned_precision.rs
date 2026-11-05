//! POOT-1476: the planner owns the precision of an op (POOT-1471
//! section 5.2 and rows 7.3). `select_operand_precision` narrows an op exactly where the planner picks a matrix-unit
//! schedule; a decode GEMV, a scalar tiling and a streaming attention stay exact, and the planner holds to it: a
//! narrowed op plans only on a matrix unit and is otherwise refused by name, never run exactly in silence.
//!
//! Every row drives the production `compile` over real dimensions (a 2560-wide block with a 9728-wide feed-forward,
//! a Q4_0 projection) at a prefill row count and at one decode row.

use poot_graph_ir::BinOp;
use poot_graph_ir::op::ContractionHead;
use poot_graph_ir::{
    Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, OperandPrecision, TensorType,
};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, KernelChoice, PlanError, Program,
    Target, compile,
};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::DType;

const WIDTH: usize = 2560;
const INTER: usize = 9728;
const PREFILL: usize = 512;

const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

/// gfx1151: the RDNA3 matrix unit with a known register file, as the ROCm runtime reports it.
fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.register_file = AmdArch::gfx1151().register_file();
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

/// gfx1151 with no matrix unit reported.
fn rocm_without_units() -> Target {
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps: DeviceCaps::rocm_default(),
    }
}

/// wgpu over RADV: an opaque f16 cooperative matrix, one wave per workgroup.
fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

fn options(policy: MatmulPrecision) -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy)
}

fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision) -> Program {
    compile(g, target, &options(policy)).expect("the graph compiles")
}

/// The capability `g` is refused for, or `None` when it compiles.
fn refusal(g: &Graph, target: &Target, policy: MatmulPrecision) -> Option<Capability> {
    match compile(g, target, &options(policy)) {
        Err(CompileError::Plan(error)) => match *error {
            PlanError::Refused(refusal) => Some(refusal.missing),
            other => panic!("expected a typed refusal or a program, got {other:?}"),
        },
        Err(other) => panic!("expected a typed refusal or a program, got {other:?}"),
        Ok(_) => None,
    }
}

/// A feed-forward block over `rows` rows: `down(silu(x Wg^T) * (x Wu^T))` with BF16 checkpoint-order weights, and
/// a prefill attention over the same rows when `rows` is more than one.
fn block(rows: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, WIDTH])).unwrap();
    let weight = |name: &str, n: usize, k: usize| {
        b.constant(name, TensorType::new(vec![n, k], DType::BF16))
            .unwrap()
    };
    let project = |input, w| {
        b.matmul(input, b.transpose(w, vec![1, 0]).unwrap())
            .unwrap()
    };
    let gate = project(x, weight("wg", INTER, WIDTH));
    let up = project(x, weight("wu", INTER, WIDTH));
    let act = b
        .binary(BinOp::Mul, poot_graph_ir::ops::silu(&b, gate).unwrap(), up)
        .unwrap();
    let out = project(act, weight("wd", WIDTH, INTER));
    b.finish(out).unwrap()
}

/// A Q4_0 packed projection over `rows` rows.
fn packed(rows: usize) -> Graph {
    let descriptor = PackedWeight::try_new(WeightFormat::Q4_0, [INTER, WIDTH]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, WIDTH])).unwrap();
    let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
    b.finish(out).unwrap()
}

/// What a planned contraction or attention runs: its declared precision and its schedule.
#[derive(Debug, PartialEq)]
enum Runs {
    Gemv,
    Scalar,
    Matrix,
}

/// The `(precision, runs)` of every contraction and attention the program plans.
fn schedules(program: &Program) -> Vec<(OperandPrecision, Runs)> {
    program
        .planned()
        .filter_map(|(eqn, _)| {
            let precision = eqn.op.operand_precision()?;
            let runs = match program.kernel_choice(eqn) {
                KernelChoice::Generated(KernelRequest::Contraction(spec)) => {
                    let schedule = match spec {
                        ContractionSpec::DenseTiled(spec) => spec.schedule,
                        ContractionSpec::Packed(spec) => spec.schedule,
                        ContractionSpec::Indexed(spec) => spec.schedule,
                        _ => return Some((precision, Runs::Gemv)),
                    };
                    match schedule {
                        Schedule::FragmentTiled { .. } => Runs::Matrix,
                        Schedule::RegisterTiled { .. } => Runs::Scalar,
                        Schedule::Gemv { .. } => Runs::Gemv,
                    }
                }
                _ => return None,
            };
            Some((precision, runs))
        })
        .collect()
}

fn is_contraction(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::DenseContraction { .. }
            | OpKind::PackedContraction { .. }
            | OpKind::MatMul { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Dense { .. }
                    | ContractionHead::Packed { .. }
                    | ContractionHead::MatMul { .. },
                ..
            }
    )
}

/// The invariant itself (design 7.3, "narrow means matrix unit"): in a compiled program an op reads narrowed operands
/// if and only if it runs a matrix-unit schedule, and a decode GEMV is exact. Over a prefill and a decode program of a
/// dense and a packed model on gfx1151. Mutation: in `select_operand_precision` skip the planner's answer (narrow
/// every op the dtype rule admits); the decode programs narrow (`dense decode: [Narrow(F16), ..]`) and go red.
/// Mutation: drop the Gemv alternative from `RegisterTiling::matrix_unit_wins`; the same rows go red.
#[test]
fn a_narrowed_op_runs_on_a_matrix_unit_and_a_decode_gemv_stays_exact() {
    for (name, graph, narrows) in [
        ("dense prefill", block(PREFILL), true),
        ("dense decode", block(1), false),
        ("packed prefill", packed(PREFILL), true),
        ("packed decode", packed(1), false),
    ] {
        let program = compiled(&graph, &rocm_wmma(), MatmulPrecision::Narrow);
        let planned = schedules(&program);
        assert!(!planned.is_empty(), "{name}: contractions are planned");
        for (precision, runs) in &planned {
            assert_eq!(
                !precision.is_exact(),
                *runs == Runs::Matrix,
                "{name}: {precision} runs {runs:?}"
            );
        }
        let contractions: Vec<_> = program
            .planned()
            .filter(|(eqn, _)| is_contraction(&eqn.op))
            .filter_map(|(eqn, _)| eqn.op.operand_precision())
            .collect();
        assert!(!contractions.is_empty(), "{name}");
        assert_eq!(
            contractions.iter().all(|p| *p == F16),
            narrows,
            "{name}: {contractions:?}"
        );
        assert_eq!(
            contractions.iter().all(|p| p.is_exact()),
            !narrows,
            "{name}: {contractions:?}"
        );
    }
}

/// The wgpu fragment workgroup is one wave, which the cost model finds dearer than the exact tiling at the prefill
/// rows too: nothing narrows there, with no refusal. Mutation: drop the `narrowing` term from `ContractionCost::tiled`;
/// the prefill narrows on wgpu and the row goes red.
#[test]
fn nothing_narrows_on_a_one_wave_device() {
    for graph in [block(PREFILL), block(1), packed(PREFILL)] {
        let program = compiled(&graph, &wgpu(), MatmulPrecision::Narrow);
        assert!(
            schedules(&program).iter().all(|(p, _)| p.is_exact()),
            "{:?}",
            schedules(&program)
        );
    }
}

/// The planner holds to the pass's decision: an op that DECLARES the narrowing at a row count where the matrix unit
/// does not win is planned on a matrix unit anyway (one fragment row), never on the GEMV, and where the device has no
/// matrix unit for it the plan is the typed refusal. Mutation: remove `.filter(|_| precision.is_exact())` from the dense
/// GEMV route in `planner::packed::plan`; the declared decode row plans the narrowed GEMV and goes red.
#[test]
fn a_declared_narrowing_is_held_to_a_matrix_unit_or_refused() {
    let declared = |rows: usize| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![rows, WIDTH])).unwrap();
        let w = b
            .constant("w", TensorType::f32(vec![WIDTH, INTER]))
            .unwrap();
        let out = b.matmul_as(x, w, F16).unwrap();
        b.finish(out).unwrap()
    };
    for rows in [PREFILL, 16] {
        let program = compiled(&declared(rows), &rocm_wmma(), MatmulPrecision::Exact);
        assert_eq!(
            schedules(&program),
            vec![(F16, Runs::Matrix)],
            "{rows} rows"
        );
    }
    assert_eq!(
        refusal(
            &declared(PREFILL),
            &rocm_without_units(),
            MatmulPrecision::Exact
        ),
        Some(Capability::NoMatrixUnitSchedule)
    );
    // The dense decode projection, declared at one row (`declare_precision`: the pass skips the planner's question): a
    // narrowed read never takes the GEMV, so every contraction plans one fragment row.
    let options = CompileOptions {
        declare_precision: true,
        ..options(MatmulPrecision::Narrow)
    };
    let program = compile(&block(1), &rocm_wmma(), &options).expect("the declared decode compiles");
    let planned = schedules(&program);
    assert!(!planned.is_empty());
    assert!(
        planned.iter().all(|entry| *entry == (F16, Runs::Matrix)),
        "{planned:?}"
    );
}

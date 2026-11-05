//! POOT-1451: `compile` groups the rows of every indexed contraction it tiles by expert, block-aligned
//! (`group_expert_rows`), so a routed MoE prefill reaches the fragment contraction: one `ExpertGroup` per routing
//! site, every contraction of the site reading it, and every tile one expert's rows.
//!
//! Every row drives the production `compile` and reads what `Program::planned()` holds.

use poot_graph_ir::op::{BinOp, ContractionHead, ExpertRows};
use poot_graph_ir::{Builder, Graph, MatmulPrecision, OpKind, TensorType};
use poot_graph_plan::{
    Capability, CompileError, CompileLimits, CompileOptions, KernelChoice, PlanError, Program,
    Target, compile,
};
use poot_kernelgen::{ContractionSpec, IndexedContraction, KernelRequest, Schedule};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::DType;

const EXPERTS: usize = 8;
const WIDTH: usize = 256;
const INTER: usize = 128;

/// gfx1151 with its WMMA units and wave32 subgroups declared, as a live agent query reports them.
fn rocm() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

fn compiled(g: &Graph, policy: MatmulPrecision) -> Program {
    compile(
        g,
        &rocm(),
        &CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy),
    )
    .unwrap_or_else(|error| panic!("the routed block compiles: {error}"))
}

/// A routed gated FFN over `rows` routed rows, as the MoE block traces it: gate and up `[E, INTER, WIDTH]` and down
/// `[E, WIDTH, INTER]` BF16 stacks in checkpoint order, each read through its transposed view by an indexed
/// contraction over the same ids, `down(silu(gate) * up)` weighted per row.
fn routed_ffn(rows: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, WIDTH])).unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![rows], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let stack = |name: &str, out: usize, k: usize| {
        let w = b
            .constant(name, TensorType::new(vec![EXPERTS, out, k], DType::BF16))
            .unwrap();
        b.transpose(w, vec![0, 2, 1]).unwrap()
    };
    let gate = b
        .indexed_matmul(x, stack("gate", INTER, WIDTH), ids)
        .unwrap();
    let up = b.indexed_matmul(x, stack("up", INTER, WIDTH), ids).unwrap();
    let hidden = b
        .binary(BinOp::Mul, poot_graph_ir::ops::silu(&b, gate).unwrap(), up)
        .unwrap();
    let down = b
        .indexed_matmul(hidden, stack("down", WIDTH, INTER), ids)
        .unwrap();
    let weights = b.constant("weights", TensorType::f32(vec![rows])).unwrap();
    let out = b
        .binary(BinOp::Mul, down, b.reshape(weights, vec![rows, 1]).unwrap())
        .unwrap();
    b.finish(out).unwrap()
}

/// How each planned indexed contraction names its experts, with its request.
fn indexed(program: &Program) -> Vec<(ExpertRows, &IndexedContraction)> {
    program
        .planned()
        .filter_map(|(eqn, _)| {
            let rows = match &eqn.op {
                OpKind::IndexedMatMul { rows, .. }
                | OpKind::ContractionEpilogue {
                    head: ContractionHead::Indexed { rows, .. },
                    ..
                } => *rows,
                _ => return None,
            };
            match program.kernel_choice(eqn) {
                KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Indexed(
                    request,
                ))) => Some((rows, request)),
                other => panic!("expected a generated indexed contraction, got {other:?}"),
            }
        })
        .collect()
}

fn groupings(program: &Program) -> Vec<usize> {
    program
        .planned()
        .filter_map(|(eqn, _)| match eqn.op {
            OpKind::ExpertGroup { experts, block } => {
                assert_eq!(experts, EXPERTS);
                Some(block)
            }
            _ => None,
        })
        .collect()
}

/// SC-1451-1: a routed prefill (1024 rows over 8 experts) under the narrow policy on ROCm groups its rows once for
/// the gate, up and down projections, every projection reads that grouping (`Grouped` at its block) and runs the
/// fragment contraction whose output tile is the block. The exact policy groups too, on the scalar tile.
///
/// Mutation (observed red): drop the `group_expert_rows` pass from `compile`; the per-row prefill contractions are
/// refused (`UngroupedExpertRows`) and the compile panics.
#[test]
fn a_routed_prefill_is_grouped_once_and_runs_the_fragment_tile() {
    for (policy, fragment) in [
        (MatmulPrecision::Narrow, true),
        (MatmulPrecision::Exact, false),
    ] {
        let program = compiled(&routed_ffn(1024), policy);
        let [block] = groupings(&program)[..] else {
            panic!("{policy:?}: one grouping, got {:?}", groupings(&program));
        };
        let contractions = indexed(&program);
        assert_eq!(contractions.len(), 3, "{policy:?}: gate, up and down");
        for (rows, request) in contractions {
            assert_eq!(rows, ExpertRows::Grouped { block }, "{policy:?}");
            assert_eq!(
                request.schedule.output_tile().map(|(rows, _)| rows),
                Some(block),
                "{policy:?}: {:?}",
                request.schedule
            );
            assert_eq!(
                matches!(request.schedule, Schedule::FragmentTiled { .. }),
                fragment,
                "{policy:?}: {:?}",
                request.schedule
            );
        }
    }
}

/// SC-1451-2: a decode (one token at its experts: a Gemv's rows) is not grouped: its contractions read the per-row
/// ids on the Gemv, and no `ExpertGroup` is planned.
///
/// Mutation (observed red): group every per-row indexed contraction in `group_expert_rows` (`candidate` ignoring
/// the Gemv); the decode plans a grouping and this row goes red.
#[test]
fn a_decode_keeps_its_per_row_ids_on_the_gemv() {
    let program = compiled(&routed_ffn(4), MatmulPrecision::Narrow);
    assert!(groupings(&program).is_empty(), "{:?}", groupings(&program));
    for (rows, request) in indexed(&program) {
        assert_eq!(rows, ExpertRows::PerRow);
        assert!(
            matches!(request.schedule, Schedule::Gemv { .. }),
            "{:?}",
            request.schedule
        );
    }
}

/// SC-1451-3: a grouped contraction is planned only on a tiling whose output tile is the grouping's block: a block
/// no tiling has (3 rows) is refused by name, never run with tiles that span two buckets.
///
/// Mutation (observed red): drop the `admits_tile_rows` check in `register_tiled_launch`; the block-3 contraction
/// plans on a 16-row tiling and this row goes red.
#[test]
fn a_grouping_whose_block_no_tiling_has_is_refused_by_name() {
    let rows = 64;
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, WIDTH])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, WIDTH, INTER]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![rows], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let groups = b.expert_group(ids, EXPERTS, 3).unwrap();
    let out = b.grouped_indexed_matmul(x, w, groups, 3).unwrap();
    let error = compile(
        &b.finish(out).unwrap(),
        &rocm(),
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect_err("no tiling has 3-row tiles");
    let CompileError::Plan(plan) = &error else {
        panic!("a planner refusal, got {error}");
    };
    let PlanError::Refused(refusal) = plan.as_ref() else {
        panic!("a typed refusal, got {error}");
    };
    assert!(
        matches!(
            refusal.missing,
            Capability::GroupedTileRows { block: 3, .. }
        ),
        "{error}"
    );
}

/// SC-1451-4: the planner runs a per-row indexed contraction past the Gemv rows on no schedule: the tiled
/// contraction reads grouped rows, so planning the per-row form directly (as no `compile` does, its pass groups it
/// first) is a typed refusal, never the per-row ids read as a grouping.
///
/// Mutation (observed red): plan `(ExpertRows::PerRow, None)` as the tiled schedule in `plan_indexed`
/// (`planner/moe.rs`); the equation plans and this row goes red.
#[test]
fn a_per_row_contraction_past_the_gemv_rows_is_refused_by_name() {
    let rows = 1024;
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, WIDTH])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, WIDTH, INTER]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![rows], DType::I32),
            poot_graph_ir::IndexRange::below(EXPERTS).unwrap(),
        )
        .unwrap();
    let out = b.indexed_matmul(x, w, ids).unwrap();
    let g = b.finish(out).unwrap();
    let eqn = g
        .eqns()
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::IndexedMatMul { .. }))
        .expect("the gate projection");
    let Target::Gpu { backend, caps } = rocm() else {
        unreachable!("a GPU target")
    };
    match poot_graph_plan::test_support::plan_eqn(&g, eqn, backend, &caps) {
        Err(PlanError::Refused(refusal)) => {
            assert_eq!(refusal.missing, Capability::UngroupedExpertRows)
        }
        other => panic!("expected the typed refusal, got {other:?}"),
    }
}

//! POOT-1436: the planner takes a fragment tiling for a narrowed contraction where the device has a matrix unit for
//! the operands, and the scalar tiling everywhere else.
//!
//! Every row drives the production `compile` and reads the schedule of the kernel request the plan holds. The
//! choice reads the policy (through the operand precision `select_operand_precision` writes), the target's matrix
//! units and the shapes, and nothing else.

use poot_graph_ir::op::ContractionHead;
use poot_graph_ir::{
    Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, OperandPrecision, TensorType,
};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{ContractionSpec, KernelRequest, Schedule};
use poot_quant::PackedWeight;
use poot_quant::format::WeightFormat;
use poot_target::{
    AmdArch, Backend, DeviceCaps, FragmentLayout, LaneMap, MatrixUnits, Queried, SubgroupSupport,
};
use poot_tensor::DType;

/// The long prefill projection of the bank (`prefill-dense-bf16-nk`): 2048 rows by a 3072 x 1024 weight.
const M: usize = 2048;
const K: usize = 1024;
const N: usize = 3072;

/// ROCm caps with the RDNA3 WMMA lowering (bf16 and f16 fragments, declared lane map), as the ROCm runtime reports
/// them on gfx11.
fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    // The wavefront is the device's: a live agent query reports it, the fixture states it.
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

/// wgpu over RADV: an opaque f16 cooperative matrix, one subgroup per workgroup.
fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// Raw Vulkan over RADV: the same unit, with several subgroups per workgroup.
fn vulkan() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::vulkan_rdna3_igpu(),
    }
}

fn rocm_without_units() -> Target {
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps: DeviceCaps::rocm_default(),
    }
}

fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision) -> Program {
    let options = CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy);
    compile(g, target, &options).expect("the graph compiles")
}

/// `matmul(x, transpose(w))` with a bf16 `[N, K]` checkpoint weight: the dense fold turns it into one
/// `DenseContraction` over the weight as stored.
fn dense_bf16() -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![N, K], DType::BF16))
        .unwrap();
    let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
    b.finish(out).unwrap()
}

/// `matmul(x, w)` over an f32 `[K, N]` weight: the plain matmul's tiled route.
fn matmul_f32() -> Graph {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![K, N])).unwrap();
    let out = b.matmul(x, w).unwrap();
    b.finish(out).unwrap()
}

/// A packed (Q8_0) linear over `M` rows.
fn packed_q8() -> Graph {
    let descriptor = PackedWeight::try_new(WeightFormat::Q8_0, [N, K]).unwrap();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
        &b, x, "layer", descriptor, None, None,
    )
    .unwrap();
    b.finish(out).unwrap()
}

/// `IndexedMatMul(x [M, K], w [8, K, N], ids [M])`: each row reads its own expert's slab.
fn indexed() -> Graph {
    let experts = 8;
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![experts, K, 512]))
        .unwrap();
    let ids = b
        .constant_ranged(
            "ids",
            TensorType::new(vec![M], DType::I32),
            poot_graph_ir::IndexRange::below(experts).unwrap(),
        )
        .unwrap();
    let out = b.indexed_matmul(x, w, ids).unwrap();
    b.finish(out).unwrap()
}

/// The schedule of the kernel request behind the first equation `want` selects.
fn schedule_of(program: &Program, want: impl Fn(&OpKind) -> bool) -> Schedule {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| want(&eqn.op))
        .expect("the equation is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Contraction(spec)) => match spec {
            ContractionSpec::DenseTiled(spec) => spec.schedule,
            ContractionSpec::Indexed(spec) => spec.schedule,
            ContractionSpec::Packed(spec) => spec.schedule,
            other => panic!("not a tiled contraction: {other:?}"),
        },
        other => panic!("expected a generated contraction, got {other:?}"),
    }
}

/// Each family is matched bare or as an epilogue's head.
fn is_dense(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::DenseContraction { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Dense { .. },
                ..
            }
    )
}

fn is_matmul(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::MatMul { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::MatMul { .. },
                ..
            }
    )
}

fn is_packed(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::PackedContraction { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Packed { .. },
                ..
            }
    )
}

fn is_indexed(op: &OpKind) -> bool {
    matches!(
        op,
        OpKind::IndexedMatMul { .. }
            | OpKind::ContractionEpilogue {
                head: ContractionHead::Indexed { .. },
                ..
            }
    )
}

/// The fragment dtype, layout and waves of a fragment schedule.
fn fragment(schedule: Schedule) -> (FragmentDtype, FragmentLayout, u32) {
    match schedule {
        Schedule::FragmentTiled {
            waves_m,
            waves_n,
            tile,
            ..
        } => (tile.dtype, tile.layout, waves_m.get() * waves_n.get()),
        other => panic!("expected a fragment tiling, got {other:?}"),
    }
}

/// A `Narrow` policy on ROCm plans every prefill-shaped dense and packed contraction family on the
/// f16 fragment (POOT-1458: f16 first where the unit has it) with the declared lane map: a dense contraction over
/// a bf16 weight, a plain matmul over an f32 `[K, N]` weight, and a packed (quantized) contraction. Mutation: in `RegisterTiling::candidates`, drop the
/// fragment chain; every row plans the scalar tiling and goes red with `expected a fragment tiling`.
#[test]
fn a_narrow_policy_plans_the_fragment_tiling_on_rocm() {
    let target = rocm_wmma();
    let declared = FragmentLayout::Declared(LaneMap::Rdna3);
    for (name, graph, want) in [
        (
            "dense bf16 nk",
            dense_bf16(),
            is_dense as fn(&OpKind) -> bool,
        ),
        ("matmul f32 kn", matmul_f32(), is_matmul),
        ("packed q8_0", packed_q8(), is_packed),
    ] {
        let program = compiled(&graph, &target, MatmulPrecision::Narrow);
        let schedule = schedule_of(&program, want);
        let (dtype, layout, waves) = fragment(schedule);
        assert_eq!(
            (dtype, layout),
            (FragmentDtype::F16, declared),
            "{name}: {schedule:?}"
        );
        assert!(
            waves > 1,
            "{name}: a long prefill takes several waves: {schedule:?}"
        );
        let (rows, cols) = schedule.output_tile().unwrap();
        assert!(
            rows >= 64 && cols >= 64,
            "{name}: a long prefill takes a large output tile, got {rows} x {cols}"
        );
    }
}

/// Without a `Narrow` policy, or without a matrix unit for the operands, the scalar tiling plans: the policy is a
/// permission and the device's units are the other half of it. Mutation: let `candidates` offer fragments to a
/// contraction whose `StagedTerms::fragment` is `None`; the exact row goes red.
#[test]
fn an_exact_policy_or_a_device_without_units_plans_the_scalar_tiling() {
    for (name, target, policy) in [
        ("exact policy", rocm_wmma(), MatmulPrecision::Exact),
        ("no units", rocm_without_units(), MatmulPrecision::Narrow),
        (
            "no units, split",
            rocm_without_units(),
            MatmulPrecision::Split,
        ),
    ] {
        let program = compiled(&dense_bf16(), &target, policy);
        let schedule = schedule_of(&program, is_dense);
        assert!(
            matches!(schedule, Schedule::RegisterTiled { .. }),
            "{name}: {schedule:?}"
        );
    }
}

/// A `Split` policy plans the same fragments: both products of a split operand run on the matrix unit.
#[test]
fn a_split_policy_plans_the_fragment_tiling_too() {
    let program = compiled(&dense_bf16(), &rocm_wmma(), MatmulPrecision::Split);
    let (dtype, _, _) = fragment(schedule_of(&program, is_dense));
    assert_eq!(dtype, FragmentDtype::F16);
}

/// RADV has an opaque f16 unit, and a prefill projection does not narrow on it under the `narrow` policy: its staged
/// values are rounded in integer arithmetic (SPIR-V keeps the integer form, POOT-1458), and the cost
/// model finds the exact register tiling cheaper (bank `prefill-dense-bf16-nk-narrow`: the fragment tiling measured
/// 0.61x of f32 on device time through wgpu, whose workgroup is one wave). An op that DECLARES the narrowing is still
/// held to a matrix unit: raw Vulkan, whose workgroup is several waves, plans opaque f16 fragments for it and never
/// the narrowed register tiling. Mutation: let `register_tiled_launch` fall back to `RegisterTiling::choose` for a
/// narrowed shape; the declared row plans a narrowed register tiling and goes red.
#[test]
fn radv_keeps_the_exact_tiling_and_plans_a_declared_narrowing_on_fragments() {
    for target in [wgpu(), vulkan()] {
        let program = compiled(&dense_bf16(), &target, MatmulPrecision::Narrow);
        let schedule = schedule_of(&program, is_dense);
        assert!(
            matches!(schedule, Schedule::RegisterTiled { .. }),
            "the narrow policy leaves RADV exact: {schedule:?}"
        );
    }
    let declared = {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![M, K])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![K, N])).unwrap();
        let out = b
            .matmul_as(x, w, OperandPrecision::Narrow(FragmentDtype::F16))
            .unwrap();
        b.finish(out).unwrap()
    };
    let program = compiled(&declared, &vulkan(), MatmulPrecision::Exact);
    let (dtype, layout, waves) = fragment(schedule_of(&program, is_matmul));
    assert_eq!(
        (dtype, layout),
        (FragmentDtype::F16, FragmentLayout::Opaque)
    );
    assert!(waves > 1, "raw Vulkan runs several subgroups: {waves}");
}

/// An indexed prefill (each row reads its own expert's slab) runs the fragment tiling under a narrow policy: compile
/// groups its rows by expert (POOT-1451), so every fragment's rows share one expert's weight tile, and the tiling's
/// output tile is the grouping's block. Mutation: drop the fragment dtype from the indexed planner's terms
/// (`fragment: None` in `indexed_tiled_shape`); the request keeps the scalar tiling and this row goes red.
#[test]
fn an_indexed_prefill_runs_the_fragment_tiling_over_its_grouped_rows_under_a_narrow_policy() {
    let program = compiled(&indexed(), &rocm_wmma(), MatmulPrecision::Narrow);
    let schedule = schedule_of(&program, is_indexed);
    assert!(
        matches!(schedule, Schedule::FragmentTiled { .. }),
        "{schedule:?}"
    );
    let block = program
        .planned()
        .find_map(|(eqn, _)| match eqn.op {
            OpKind::ExpertGroup { block, .. } => Some(block),
            _ => None,
        })
        .expect("the rows are grouped");
    assert_eq!(schedule.output_tile().map(|(rows, _)| rows), Some(block));
}

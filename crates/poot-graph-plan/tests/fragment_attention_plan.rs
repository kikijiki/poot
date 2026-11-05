//! POOT-1437: the planner takes a fragment tiling for a narrowed prefill attention where the device has a matrix unit
//! for the operands, and the scalar tiling everywhere else.
//!
//! Every row drives the production `compile` and reads the schedule of the attention kernel request the plan holds.
//! The choice reads the equation's operand precision, the target's matrix units and the shapes, and nothing else.

use poot_graph_ir::{
    AttentionAttrs, Builder, FragmentDtype, Graph, OpKind, Operand, OperandPrecision, TensorType,
};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{
    AttentionMicro, AttentionSchedule, AttentionSpec, KernelRequest, KvLane, WaveTile,
};
use poot_target::{
    AmdArch, Backend, DeviceCaps, FragmentLayout, LaneMap, MatrixUnits, Queried, SubgroupSupport,
};
use poot_tensor::DType;

/// A causal prefill of the bank's `prefill-attention-2048` shape: 32 query heads over 8 key/value heads of width
/// 128, 2048 positions, an f16 KV cache.
const HQ: usize = 32;
const HKV: usize = 8;
const L: usize = 2048;

/// ROCm caps with the RDNA3 WMMA lowering (bf16 and f16 fragments, declared lane map), as the ROCm runtime reports
/// them on gfx11.
fn rocm_wmma() -> Target {
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

/// wgpu over RADV: an opaque f16 cooperative matrix, one subgroup per workgroup.
fn wgpu() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
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

/// One attention equation declaring `precision` over `[1, HQ, L, d]` queries and an f16 KV cache.
fn attention(precision: OperandPrecision, d: usize) -> Graph {
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, HQ, L, d])).unwrap();
    let kv = TensorType::new(vec![1, HKV, L, d], DType::F16);
    let k = b.constant("k", kv.clone()).unwrap();
    let v = b.constant("v", kv).unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, L, L]))
        .unwrap();
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs {
                precision,
                ..AttentionAttrs::new(HQ / HKV, 0.25, None)
            }),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    b.finish(out).unwrap()
}

fn compiled(g: &Graph, target: &Target) -> Program {
    compile(g, target, &CompileOptions::new(CompileLimits::STANDARD)).expect("the graph compiles")
}

/// The attention kernel request behind the plan's one attention equation.
fn spec_of(program: &Program) -> AttentionSpec {
    let (eqn, _) = program
        .planned()
        .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .expect("the attention equation is planned");
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Attention(spec)) => spec.clone(),
        KernelChoice::Generated(KernelRequest::AttentionRows(rows)) => rows.spec.clone(),
        other => panic!("expected a generated attention, got {other:?}"),
    }
}

/// The fragment micro-tile of the tiled schedule `spec` holds: `(lanes, dtype, layout, score, out)`.
fn fragment_of(
    spec: &AttentionSpec,
) -> Option<(usize, FragmentDtype, FragmentLayout, WaveTile, WaveTile)> {
    match spec.schedule {
        AttentionSchedule::Tiled {
            lanes,
            micro:
                AttentionMicro::Fragment {
                    dtype,
                    layout,
                    score,
                    out,
                },
            ..
        } => Some((lanes, dtype, layout, score, out)),
        _ => None,
    }
}

/// On the gfx11 matrix unit an f16-narrowed attention plans fragments of f16 with the declared lane map: a
/// workgroup of whole waves, both products' wave grids covering exactly those waves. The rows pass `select`
/// through the equation alone, with no driver option.
#[test]
fn a_narrowed_prefill_attention_plans_declared_f16_fragments_on_the_rocm_matrix_unit() {
    let program = compiled(
        &attention(OperandPrecision::Narrow(FragmentDtype::F16), 128),
        &rocm_wmma(),
    );
    let spec = spec_of(&program);
    assert_eq!(spec.kv, KvLane::F16);
    let (lanes, dtype, layout, score, out) =
        fragment_of(&spec).unwrap_or_else(|| panic!("a fragment tiling, got {:?}", spec.schedule));
    assert_eq!(dtype, FragmentDtype::F16);
    assert_eq!(layout, FragmentLayout::Declared(LaneMap::Rdna3));
    assert_eq!(lanes % 32, 0, "whole waves: {lanes} lanes");
    assert_eq!(score.waves_m * score.waves_n * 32, lanes);
    assert_eq!(out.waves_m * out.waves_n * 32, lanes);
}

/// bf16 operands on the same unit plan bf16 fragments (the K and V an f16 cache holds are converted as they are
/// staged).
#[test]
fn a_bf16_narrowed_prefill_attention_plans_bf16_fragments_on_the_rocm_matrix_unit() {
    let program = compiled(
        &attention(OperandPrecision::Narrow(FragmentDtype::Bf16), 128),
        &rocm_wmma(),
    );
    let (_, dtype, ..) = fragment_of(&spec_of(&program)).expect("a fragment tiling");
    assert_eq!(dtype, FragmentDtype::Bf16);
}

/// Raw Vulkan over RADV reports an f16 cooperative matrix with an opaque layout and runs several subgroups per
/// workgroup: it plans opaque fragments over a workgroup as wide as the register tiles' (a head of width 64 here).
#[test]
fn a_narrowed_prefill_attention_plans_opaque_f16_fragments_on_raw_vulkan() {
    let program = compiled(
        &attention(OperandPrecision::Narrow(FragmentDtype::F16), 64),
        &vulkan(),
    );
    let spec = spec_of(&program);
    let (lanes, dtype, layout, ..) =
        fragment_of(&spec).unwrap_or_else(|| panic!("a fragment tiling, got {:?}", spec.schedule));
    assert_eq!(
        (dtype, layout),
        (FragmentDtype::F16, FragmentLayout::Opaque)
    );
    assert!(lanes > 32, "{lanes} lanes: several waves");
}

/// wgpu cannot widen its subgroups, so its fragment workgroup is one wave: measured on the bank's prefill attention
/// cells that is 11.6x to 13.8x slower than the register tiles, so the planner keeps the register tiles there.
#[test]
fn a_one_wave_fragment_workgroup_is_not_planned_where_register_tiles_run_wider() {
    let program = compiled(
        &attention(OperandPrecision::Narrow(FragmentDtype::F16), 64),
        &wgpu(),
    );
    let spec = spec_of(&program);
    assert!(
        matches!(
            spec.schedule,
            AttentionSchedule::Tiled {
                micro: AttentionMicro::Scalar { .. },
                ..
            }
        ),
        "{:?}",
        spec.schedule
    );
}

/// RADV has no bf16 fragment, so a bf16 declaration keeps the register tiles there; a device without a matrix unit
/// keeps them for any narrowed dtype, and an exact attention keeps them on every device.
#[test]
fn attention_keeps_register_tiles_without_a_matrix_unit_for_its_dtype_or_when_exact() {
    for (label, target, precision) in [
        (
            "radv bf16",
            wgpu(),
            OperandPrecision::Narrow(FragmentDtype::Bf16),
        ),
        (
            "no unit",
            rocm_without_units(),
            OperandPrecision::Narrow(FragmentDtype::F16),
        ),
        ("exact rocm", rocm_wmma(), OperandPrecision::Exact),
        ("exact wgpu", wgpu(), OperandPrecision::Exact),
    ] {
        let spec = spec_of(&compiled(&attention(precision, 128), &target));
        assert!(
            matches!(
                spec.schedule,
                AttentionSchedule::Tiled {
                    micro: AttentionMicro::Scalar { .. },
                    ..
                }
            ),
            "{label}: {:?}",
            spec.schedule
        );
    }
}

/// A head width that is not a whole number of fragment depths cannot run the score product on fragments: the
/// narrowed attention keeps the register tiles.
#[test]
fn a_head_width_off_the_fragment_depth_keeps_register_tiles() {
    let spec = spec_of(&compiled(
        &attention(OperandPrecision::Narrow(FragmentDtype::F16), 72),
        &rocm_wmma(),
    ));
    assert!(
        fragment_of(&spec).is_none(),
        "dk 72 is not a multiple of 16: {:?}",
        spec.schedule
    );
}

//! POOT-1437: the planner takes a fragment tiling for a narrowed prefill attention where the device has a matrix unit
//! for the operands, and the scalar tiling everywhere else.
//!
//! Every row drives the production `compile` and reads the schedule of the attention kernel request the plan holds.
//! The choice reads the equation's operand precision, the target's matrix units and the shapes, and nothing else.

use poot_graph_ir::{
    AttentionAttrs, Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, Operand,
    OperandPrecision, TensorType,
};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Program, Target, compile};
use poot_kernelgen::{
    AttentionMicro, AttentionSchedule, AttentionSpec, KernelRequest, KvLane, QOperand, WaveTile,
};
use poot_target::{
    AmdArch, Backend, DeviceCaps, FragmentLayout, LaneMap, MatrixUnits, Queried, RegisterFile,
    SubgroupSupport,
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

/// One attention equation declaring `precision` over `[1, HQ, L, d]` queries and an f16 KV cache.
fn attention(precision: OperandPrecision, d: usize) -> Graph {
    attention_of(precision, HQ, d)
}

/// [`attention`] over `hq` query heads.
fn attention_of(precision: OperandPrecision, hq: usize, d: usize) -> Graph {
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, hq, L, d])).unwrap();
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
                ..AttentionAttrs::new(hq / HKV, 0.25, None)
            }),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    b.finish(out).unwrap()
}

/// `g` compiled under the exact policy: each graph declares its own per-equation precision, which the narrow
/// policy would otherwise override.
fn compiled(g: &Graph, target: &Target) -> Program {
    let options =
        CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(MatmulPrecision::Exact);
    compile(g, target, &options).expect("the graph compiles")
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
                    ..
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

/// A fragment tiling spends the workgroup memory on tile rows before workgroup width, and at equal rows takes the
/// narrowest workgroup: every key block is restaged once per `rows`, and more waves over the same rows measured
/// slower even with a wider key block (the bank's `prefill-attention-8192` with an f16 cache on gfx1151: POOT-1440,
/// 64 rows over 512 lanes 25.2 ms, 128 rows over 256 lanes 18.8 ms; POOT-1447, 128 rows with Q held over 512 lanes
/// 20.5 ms at a 64-key block, against 18.2 ms over 256 lanes at 32 keys with Q staged). The bank's 16-head,
/// 8-key/value-head, 128-wide prefill therefore plans the most rows the planner holds (128: `br` 64 of 2 heads) over
/// 256 lanes (the register tiles' width), a key block of 32 (the widest that fits beside them in the 64 KiB of
/// workgroup memory) and Q staged (held, its 64 registers do not fit beside the 80 of the accumulators).
///
/// Mutation: rank the fragment candidates `(lanes, rows, bc)` as the register tiles are; the plan is 32 rows over
/// 512 lanes and the rows assertion reads `left: 64, right: 128`. Mutation: give the softmax's maxima and sums
/// separate partial arrays; the 128-row, 32-key tiling needs 73 KiB, the plan falls back to a 16-key block and the
/// key-block assertion goes red. Mutation: rank the widest workgroup first at equal rows (the order before POOT-1447);
/// the plan is 512 lanes with a 64-key block and Q held, and the key-block assertion reads `left: 64, right: 32`.
#[test]
fn a_fragment_attention_spends_its_workgroup_memory_on_rows_before_lanes() {
    let target = rocm_wmma();
    let Target::Gpu { caps, .. } = &target else {
        unreachable!("a GPU target")
    };
    for dtype in [FragmentDtype::F16, FragmentDtype::Bf16] {
        let program = compiled(
            &attention_of(OperandPrecision::Narrow(dtype), 16, 128),
            &target,
        );
        let spec = spec_of(&program);
        let AttentionSchedule::Tiled { br, bc, lanes, .. } = spec.schedule else {
            panic!("{dtype:?}: a tiled schedule, got {:?}", spec.schedule);
        };
        assert!(fragment_of(&spec).is_some(), "{dtype:?}: fragments");
        assert_eq!(br * spec.n_rep, 128, "{dtype:?}: tile rows");
        assert_eq!(bc, 32, "{dtype:?}: key block");
        assert_eq!(lanes, 256, "{dtype:?}: lanes");
        assert!(
            matches!(
                spec.schedule,
                AttentionSchedule::Tiled {
                    micro: AttentionMicro::Fragment {
                        q: QOperand::Staged,
                        ..
                    },
                    ..
                }
            ),
            "{dtype:?}: Q staged"
        );
        let lds = spec.tiled_lds_bytes().expect("tiled").expect("fits");
        assert!(
            lds <= caps.lds_bytes as usize,
            "{dtype:?}: {lds} bytes of LDS"
        );
    }
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
    assert!(
        matches!(
            spec.schedule,
            AttentionSchedule::Tiled {
                micro: AttentionMicro::Fragment {
                    q: QOperand::Held,
                    ..
                },
                ..
            }
        ),
        "an opaque fragment holds Q in registers too: {:?}",
        spec.schedule
    );
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

/// The tiled schedule of the bank's 16-head, 8-key/value-head, 128-wide prefill on `target`: `(rows, bc, lanes, q)`
/// of its fragment micro-tile, `None` for register tiles.
fn bank_fragment_tiling(target: &Target) -> Option<(usize, usize, usize, QOperand)> {
    let spec = spec_of(&compiled(
        &attention_of(OperandPrecision::Narrow(FragmentDtype::F16), 16, 128),
        target,
    ));
    match spec.schedule {
        AttentionSchedule::Tiled {
            br,
            bc,
            lanes,
            micro: AttentionMicro::Fragment { q, .. },
        } => Some((br * spec.n_rep, bc, lanes, q)),
        _ => None,
    }
}

/// gfx1151's matrix unit with a register file of `per_workgroup` registers summed over a workgroup's lanes.
fn rocm_wmma_with_file(file: Queried<RegisterFile>) -> Target {
    let Target::Gpu { backend, mut caps } = rocm_wmma() else {
        unreachable!("a GPU target")
    };
    caps.register_file = file;
    Target::Gpu { backend, caps }
}

/// A workgroup keeps only its share of the register file resident: with 32 768 registers a workgroup, a lane of a
/// 256-lane workgroup holds 128, half of them for held fragments, so the 128-row tiling (80 held registers) does
/// not fit and the plan keeps 64 rows (40). Mutation: make `RegisterFile::per_lane_of` return `per_lane` alone; the
/// plan reads 128 rows.
#[test]
fn a_workgroup_too_wide_for_its_share_of_the_register_file_is_not_planned() {
    let tiling = bank_fragment_tiling(&rocm_wmma_with_file(Queried::Known(RegisterFile {
        per_lane: 256,
        per_workgroup: 32_768,
    })))
    .expect("a fragment tiling");
    assert_eq!(
        (tiling.0, tiling.2),
        (64, 256),
        "{tiling:?}: rows and lanes"
    );
}

/// Where a held Q fits beside the accumulators of the narrowest workgroup it is held: a 64-wide head costs 32 Q
/// registers per fragment row, so the 64-wide prefill on gfx1151 plans 128 rows over 256 lanes with Q held (a held Q
/// measured 17.5 against 18.2 ms with Q staged at the same tiling of the 128-wide bank cell, POOT-1447).
/// Mutation: rank a staged Q ahead of a held one; the plan stages Q.
#[test]
fn a_fragment_attention_holds_q_where_it_fits_the_register_file() {
    let spec = spec_of(&compiled(
        &attention_of(OperandPrecision::Narrow(FragmentDtype::F16), 16, 64),
        &rocm_wmma(),
    ));
    let AttentionSchedule::Tiled {
        br,
        lanes,
        micro: AttentionMicro::Fragment { q, .. },
        ..
    } = spec.schedule
    else {
        panic!("a fragment tiling, got {:?}", spec.schedule);
    };
    assert_eq!(
        (br * spec.n_rep, lanes, q),
        (128, 256, QOperand::Held),
        "{:?}",
        spec.schedule
    );
}

/// A fragment tiling's register budget is the device's: where the runtime cannot tell its register file the
/// planner holds no fragments and keeps the register tiles. Mutation: assume a 256-register lane when the file is
/// `Unknown` in `attention_tiling_of`; the plan is a fragment tiling.
#[test]
fn an_unknown_register_file_plans_no_fragments() {
    assert_eq!(
        bank_fragment_tiling(&rocm_wmma_with_file(Queried::Unknown)),
        None
    );
}

//! POOT-1520: the planner invariant, model-free half. The schedule [`RegisterTiling::select`] chooses never loses to
//! the best exact schedule, over a sweep of each shape class's free dimension on every backend that narrows.
//!
//! Three statements make the invariant, each checked against a brute-force walk of the candidates and not against the
//! selection's own answer:
//!
//! 1. The choice is the argmin of the one cost model over the exact tilings, the exact Gemv and the narrowed tilings,
//!    and a tie goes to the exact schedule (a narrowing rounds, so it must win).
//! 2. A narrowed choice holds its registers: the tiling's accumulators and the registers it keeps across the `K`
//!    loop fit the lane's budget (`caps.gemm_launch().register_budget`). The cost model has no spill term; a
//!    schedule that spills loses to exact by an order of magnitude (POOT-1505: 720 VGPRs spilled, POOT-1509: 13 ms
//!    against 0.9 ms) while its modelled cost still wins, so the register bound is part of the invariant.
//! 3. A one-row dense decode contraction is never narrowed: the Gemv and a fragment tiling read the same weight bytes
//!    once, and the tiling pads one row to a whole fragment, so a matrix unit can only add an LDS stage. The model
//!    puts the two within a few percent of each other, so a ratio fitted on indexed rows alone moved the vocabulary
//!    head onto a matrix unit and decode went 8 % slower (POOT-1503).
//!
//! A violation is a failing row that names the shape, the device and the schedule.

use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_quant::format::WeightFormat;
use poot_target::{Backend, DeviceCaps};

use super::cost::ContractionCost;
use super::{
    Feasibility, GemvShape, RegisterTiling, Selection, SpillCheck, StagedWeight, TiledRows,
    TiledShape, probe_body,
};
use crate::helpers::WeightLayout;
use crate::precision::StagedTerms;

/// A device the invariant is checked on: the planner's caps for one backend.
struct Device {
    name: &'static str,
    caps: DeviceCaps,
}

/// Every backend that has a narrow schedule, with the calibration its planner column runs: gfx1151 through HSA,
/// RADV through wgpu and raw Vulkan, and the two NVIDIA columns (sm_80's portable prior and the measured sm_120).
fn devices() -> Vec<Device> {
    let sm_120 = DeviceCaps {
        matrix_units: poot_target::Queried::Known(poot_target::MatrixUnits::nvidia((12, 0))),
        register_file: poot_target::RegisterFile::nvidia(12)
            .map_or(poot_target::Queried::Unknown, poot_target::Queried::Known),
        perf_calibration: poot_target::calibration::Calibrated::SM_120,
        ..DeviceCaps::ptx_default()
    };
    vec![
        Device {
            name: "rocm gfx1151",
            caps: poot_test_util::device_caps::default_caps_for(Backend::AmdGcn(
                poot_target::AmdArch::gfx1151(),
            )),
        },
        Device {
            name: "wgpu rdna3",
            caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
        },
        Device {
            name: "vulkan rdna3",
            caps: poot_test_util::device_caps::vulkan_rdna3_igpu(),
        },
        Device {
            name: "ptx sm_80",
            caps: DeviceCaps::ptx_default(),
        },
        Device {
            name: "ptx sm_120",
            caps: sm_120,
        },
    ]
}

/// The fragment dtype the planner narrows to on `caps`: f16 where the unit takes it, else bf16.
fn fragment_dtype(caps: &DeviceCaps) -> Option<FragmentDtype> {
    let units = caps.matrix_units.known()?;
    [FragmentDtype::F16, FragmentDtype::Bf16]
        .into_iter()
        .find(|&dtype| units.config(dtype).is_some())
}

/// What a shape class is: the shape of one contraction, the narrowing the policy permits and the Gemv the planner
/// would also cost. `one_row_decode` is the class of statement 3.
struct Case {
    name: String,
    shape: TiledShape,
    gemv: Option<GemvShape>,
    one_row_decode: bool,
}

impl Case {
    fn new(name: String, shape: TiledShape) -> Self {
        Self {
            name,
            shape,
            gemv: None,
            one_row_decode: false,
        }
    }
}

fn dense(rows: usize, k: usize, n: usize, weight_bits: u32) -> TiledShape {
    TiledShape {
        blocks: 1,
        m_per_block: rows,
        out_per_block: n,
        k,
        layout: WeightLayout::Nk,
        rows: TiledRows::Dense,
        terms: StagedTerms::ONE,
        weight: StagedWeight::Dense { bits: weight_bits },
        activation_bits: 32,
    }
}

fn packed(rows: usize, k: usize, n: usize, format: WeightFormat) -> TiledShape {
    TiledShape {
        weight: StagedWeight::Packed(format),
        ..dense(rows, k, n, 32)
    }
}

fn routed(shape: TiledShape, experts: usize) -> TiledShape {
    TiledShape {
        rows: TiledRows::Grouped {
            groups: experts + 1,
            block: 0,
        },
        ..shape
    }
}

/// The shape classes of the card, swept over the free dimension of each: dense decode (the vocabulary head
/// included) over output counts, dense and packed prefill over rows, packed grouped experts at short and long
/// prompts over rows and experts.
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    // Dense decode: one row; the Gemv is costed beside the tilings. The outputs run from a projection to the 262 144
    // columns of the largest vocabulary head.
    for k in [1024, 2560, 4096] {
        for n in [1024, 2560, 9728, 16_384, 65_536, 128_256, 151_936, 262_144] {
            for bits in [16, 32] {
                cases.push(Case {
                    name: format!("dense decode [1, {k}] x [{n}, {k}]^T w{bits}"),
                    gemv: Some(GemvShape {
                        outputs: n,
                        width: 128,
                        cols: 4,
                    }),
                    one_row_decode: true,
                    ..Case::new(String::new(), dense(1, k, n, bits))
                });
            }
        }
    }
    // Batched decode: a few rows, no Gemv.
    for rows in [2, 4, 8, 16, 32, 64] {
        for (k, n) in [(1024, 3072), (2560, 9728), (9728, 2560)] {
            cases.push(Case::new(
                format!("dense batched decode [{rows}, {k}] x [{n}, {k}]^T"),
                dense(rows, k, n, 16),
            ));
        }
    }
    // Dense prefill: short to long prompts over the bank's projections.
    for rows in [16, 32, 64, 128, 256, 512, 1024, 2048, 8192] {
        for (k, n) in [(1024, 3072), (2048, 2048), (2560, 9728), (9728, 2560)] {
            for bits in [16, 32] {
                cases.push(Case::new(
                    format!("dense prefill [{rows}, {k}] x [{n}, {k}]^T w{bits}"),
                    dense(rows, k, n, bits),
                ));
            }
        }
    }
    // Packed dense: decode (with the Gemv) and prefill, over the formats the checkpoints carry.
    for format in [
        WeightFormat::Q8_0,
        WeightFormat::Q4_K,
        WeightFormat::Q6_K,
        WeightFormat::Iq4_Nl,
        WeightFormat::Iq2_Xs,
    ] {
        for rows in [1, 4, 16, 64, 128, 512, 2048] {
            for (k, n) in [(2048, 2048), (2560, 9728)] {
                let gemv = (rows == 1).then_some(GemvShape {
                    outputs: n,
                    width: 128,
                    cols: 4,
                });
                cases.push(Case {
                    gemv,
                    one_row_decode: rows == 1,
                    ..Case::new(
                        format!("packed {format:?} [{rows}, {k}] x [{n}, {k}]^T"),
                        packed(rows, k, n, format),
                    )
                });
            }
        }
    }
    // Packed grouped experts: the wide-mixture up projection (512 experts, K 2560, N 640, top 10) and the down
    // projection, at prompts of 1 to 8192 tokens. 128 tokens is POOT-1505/1509's short prompt.
    for tokens in [1, 16, 64, 128, 256, 512, 2048, 8192] {
        for (format, k, n) in [
            (WeightFormat::Iq2_Xs, 2560, 640),
            (WeightFormat::Iq4_Nl, 640, 2560),
        ] {
            cases.push(Case::new(
                format!("packed grouped {format:?} {tokens} tokens x top 10 over 512 experts, K {k} N {n}"),
                routed(packed(tokens * 10, k, n, format), 512),
            ));
        }
    }
    // Dense grouped experts (bf16 stacks): the 128-expert mixture's up projection at 1 to 8192 tokens, and a small
    // one with a few experts, where many rows share each.
    for tokens in [1, 16, 128, 512, 2048, 8192] {
        for experts in [8, 128] {
            cases.push(Case::new(
                format!(
                    "dense grouped {tokens} tokens x top 8 over {experts} experts, K 2048 N 768"
                ),
                routed(dense(tokens * 8, 2048, 768, 16), experts),
            ));
        }
    }
    cases
}

/// The cost of `tiling` over `shape` as the plan runs it: a grouped shape at the block the tiling has.
fn tiling_cost(tiling: RegisterTiling, shape: TiledShape, caps: &DeviceCaps) -> f64 {
    let shape = match shape.rows {
        TiledRows::Dense => shape,
        TiledRows::Grouped { groups, .. } => TiledShape {
            rows: TiledRows::Grouped {
                groups,
                block: tiling.output_tile().0,
            },
            ..shape
        },
    };
    ContractionCost::tiled(tiling, shape, caps)
}

/// Whether `feasibility` lets `tiling` run over `shape`: always under the register model (its bound is the
/// candidate set's), and under compile feedback when the device's compiler compiles the probe without a spill.
fn compiles(
    shape: TiledShape,
    tiling: RegisterTiling,
    caps: &DeviceCaps,
    feasibility: Feasibility<'_>,
) -> bool {
    match feasibility {
        Feasibility::Model => true,
        Feasibility::Compiled(check) => probe_body(shape, tiling, caps.rounding_conversions)
            .is_ok_and(|body| check.holds(&body)),
    }
}

/// The cheapest cost of any tiling of `shape` the device runs under `feasibility`, found by walking every candidate
/// at every block a grouping can take: no use of the selection under test.
fn cheapest_tiling(
    shape: TiledShape,
    caps: &DeviceCaps,
    feasibility: Feasibility<'_>,
) -> Option<f64> {
    let blocks: Vec<usize> = match shape.rows {
        TiledRows::Dense => vec![0],
        TiledRows::Grouped { .. } => RegisterTiling::blocks_of(&[shape], caps),
    };
    blocks
        .into_iter()
        .flat_map(|block| {
            let at = match shape.rows {
                TiledRows::Dense => shape,
                TiledRows::Grouped { groups, .. } => TiledShape {
                    rows: TiledRows::Grouped { groups, block },
                    ..shape
                },
            };
            RegisterTiling::prefiltered(at, caps, feasibility)
                .into_iter()
                .filter(move |&tiling| compiles(at, tiling, caps, feasibility))
                .map(move |tiling| ContractionCost::tiled(tiling, at, caps))
        })
        .reduce(f64::min)
}

/// The narrowable shape of `case` on `caps` and the schedule the planner chooses for it: `None` where the device has
/// no matrix unit to narrow onto, or runs no schedule at all (which the planner refuses by name).
fn selection(
    case: &Case,
    caps: &DeviceCaps,
    feasibility: Feasibility<'_>,
) -> Option<(TiledShape, Selection)> {
    let dtype = fragment_dtype(caps)?;
    let narrowed = TiledShape {
        terms: StagedTerms::of(OperandPrecision::Narrow(dtype), None, None),
        ..case.shape
    };
    let chosen = RegisterTiling::select(narrowed, case.gemv, caps, feasibility)?;
    Some((narrowed, chosen))
}

/// Why the schedule chosen for `case` on `caps` loses to the best exact schedule, or `None` when it does not (or the
/// device runs no schedule at all, which the planner refuses by name).
fn violation(case: &Case, caps: &DeviceCaps, feasibility: Feasibility<'_>) -> Option<String> {
    let (narrowed, chosen) = selection(case, caps, feasibility)?;

    let exact = TiledShape {
        terms: StagedTerms::ONE,
        ..narrowed
    };
    let gemv = case.gemv.map(|gemv| ContractionCost::gemv(gemv, caps));
    let exact_floor = [cheapest_tiling(exact, caps, feasibility), gemv]
        .into_iter()
        .flatten()
        .reduce(f64::min);
    let narrow_floor = cheapest_tiling(narrowed, caps, feasibility);

    // 1. The argmin, a tie going to exact.
    let floor = [exact_floor, narrow_floor]
        .into_iter()
        .flatten()
        .reduce(f64::min)?;
    let chosen_cost = match chosen {
        Selection::Exact(tiling) => tiling_cost(tiling, exact, caps),
        Selection::Gemv => gemv.expect("the Gemv was costed"),
        Selection::Narrowed(tiling) => tiling_cost(tiling, narrowed, caps),
    };
    if chosen_cost > floor {
        return Some(format!(
            "{chosen:?} costs {chosen_cost:.3}, the cheapest schedule {floor:.3}"
        ));
    }
    if let Selection::Narrowed(tiling) = chosen {
        if exact_floor.is_some_and(|exact| chosen_cost >= exact) {
            return Some(format!(
                "narrowed {tiling:?} costs {chosen_cost:.3} and the exact floor is {:.3}: a tie or a loss goes to exact",
                exact_floor.unwrap_or(f64::NAN)
            ));
        }
        // 2. The registers it holds.
        if let Feasibility::Compiled(_) = feasibility {
            let block = tiling.output_tile().0;
            let at = match narrowed.rows {
                TiledRows::Dense => narrowed,
                TiledRows::Grouped { groups, .. } => TiledShape {
                    rows: TiledRows::Grouped { groups, block },
                    ..narrowed
                },
            };
            if !compiles(at, tiling, caps, feasibility) {
                return Some(format!(
                    "narrowed {tiling:?} spills when the device's compiler compiles it"
                ));
            }
        }
        let budget = caps.gemm_launch().register_budget;
        let lanes = u32::try_from(tiling.workgroup_lanes()).expect("a workgroup fits u32 lanes");
        if matches!(feasibility, Feasibility::Model) {
            let held = tiling
                .loop_registers_over(narrowed)
                .expect("a narrowed tiling states its registers");
            if tiling.accumulator_registers() > budget.accumulators(lanes)
                || budget.held(lanes).is_none_or(|bound| held > bound)
            {
                return Some(format!(
                    "narrowed {tiling:?} holds {held} registers a lane across its loop ({} accumulators) against the budget {:?}/{:?}: it spills",
                    tiling.accumulator_registers(),
                    budget.held(lanes),
                    budget.accumulators(lanes),
                ));
            }
        }
        // 3. A one-row decode.
        if case.one_row_decode {
            return Some(format!(
                "a one-row decode moved onto a matrix unit: {tiling:?}"
            ));
        }
    }
    None
}

/// Every case on every device, the violations named.
fn violations<'a>(
    devices: &[Device],
    cases: &[Case],
    feasibility: impl Fn() -> Feasibility<'a>,
) -> Vec<String> {
    devices
        .iter()
        .flat_map(|device| {
            cases.iter().filter_map(|case| {
                violation(case, &device.caps, feasibility())
                    .map(|why| format!("{} on {}: {why}", case.name, device.name))
            })
        })
        .collect()
}

/// The chosen schedule never loses to the best exact schedule: over every shape class on every backend that narrows,
/// the choice is the argmin of the cost model with a tie to exact, a narrowed choice holds its registers, and a
/// one-row decode stays exact. Mutations (each red, then green): `RegisterTiling::first_cheapest` taking the later of
/// two equal schedules (`next.0 <= best.0`), the tie rule; the `loop_registers` clause of `RegisterTiling::runs_on`
/// dropped, the register budget; `STRIX_HALO_HSA_COST_RATIOS.direct_value` back to 16.8, the 1503 revert.
#[test]
fn the_chosen_schedule_never_loses_to_exact() {
    let violations = violations(&devices(), &cases(), || Feasibility::Model);
    assert!(
        violations.is_empty(),
        "{} shapes lose to exact:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

/// The invariant is not met by never narrowing: on each device that narrows at all, the sweep holds contractions the
/// planner narrows and contractions it keeps exact, so a row that can fail is among them. Mutation: make
/// `RegisterTiling::select` drop the narrowed candidates; every device reads zero narrowed choices.
#[test]
fn the_sweep_reaches_both_sides_of_the_choice() {
    let cases = cases();
    for device in devices() {
        let (mut narrowed, mut exact) = (0, 0);
        for case in &cases {
            match selection(case, &device.caps, Feasibility::Model) {
                Some((_, chosen)) if chosen.narrows() => narrowed += 1,
                Some(_) => exact += 1,
                None => {}
            }
        }
        eprintln!(
            "{}: {narrowed} narrowed, {exact} exact of {}",
            device.name,
            cases.len()
        );
        assert!(exact > 0, "{}: the sweep keeps nothing exact", device.name);
        if device.name.starts_with("rocm") || device.name.starts_with("ptx") {
            assert!(narrowed > 0, "{}: the sweep narrows nothing", device.name);
        }
    }
}

/// A stand-in for a device compiler that spills every fragment tiling of more than one wave and holds the rest: the
/// shape of POOT-1509's measurement (the two-wave tilings of the Flash-Next expert projection at 1280 rows spilled
/// 680 to 708 bytes of local memory on sm_120 while the one-wave tiling compiled clean).
struct SpillsMultiWave;

impl SpillCheck for SpillsMultiWave {
    fn holds(&self, probe: &poot_kernel_ir::Body) -> bool {
        !probe.name.contains("FragmentTiled") || probe.name.contains("waves_m__1__waves_n__1_")
    }
}

/// Under compile feedback the chosen schedule is still the argmin over what the compiler holds, and a narrowed choice
/// is a tiling the compiler compiles without a spill: a device whose compiler refuses the model's cheapest fragment
/// tiling takes the next that it holds or stays exact, and never runs the spilling one (POOT-1509: PTX
/// `packed_contraction` at 128 tokens, 13 ms against 0.9 ms). Mutation: take the first candidate in cost order
/// without asking the compiler (`RegisterTiling::cheapest`, the `Feasibility::Compiled` arm); a narrowed choice
/// spills.
#[test]
fn a_compiled_choice_never_runs_a_tiling_its_compiler_spills() {
    let compiler = SpillsMultiWave;
    // The packed grouped experts (the shapes POOT-1509 measured) on the NVIDIA columns; each candidate's probe is a
    // generated kernel body, so the sweep is not every class.
    let devices: Vec<Device> = devices()
        .into_iter()
        .filter(|device| device.name.starts_with("ptx"))
        .collect();
    let cases: Vec<Case> = cases()
        .into_iter()
        .filter(|case| case.name.starts_with("packed grouped") && case.name.contains("Iq2_Xs"))
        .collect();
    let violations = violations(&devices, &cases, || Feasibility::Compiled(&compiler));
    assert!(
        violations.is_empty(),
        "{} shapes lose to exact:\n{}",
        violations.len(),
        violations.join("\n")
    );
    // The compiler's refusal is what the row exercises: the register model alone picks a multi-wave tiling for some
    // shape, or there is nothing here to hold back.
    let model_multi_wave = devices.iter().any(|device| {
        cases.iter().any(|case| {
            matches!(
                selection(case, &device.caps, Feasibility::Model),
                Some((_, Selection::Narrowed(tiling))) if tiling.workgroup_lanes() > super::FRAGMENT_LANES
            )
        })
    });
    assert!(
        model_multi_wave,
        "the model never picks a multi-wave tiling: the stand-in compiler refuses nothing"
    );
}

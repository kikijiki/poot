//! The registers an AMDGCN kernel the planner's tilings produce takes: compiled offline for gfx1151, its code
//! object's metadata must show no spilled VGPR. A spilled register-tiled body moves its accumulators through
//! scratch memory inside the key and depth loops; the tiled attention at its planned tiling did, 60-120 VGPRs
//! of it, until the AMDGCN compile stopped LLVM re-unrolling the generators' own unrolled loops (POOT-1428).

use std::path::Path;

use poot_codegen::{Target, artifact_path};
use poot_target::{AmdArch, Backend, BufferStorage, DeviceCaps, Queried, RegisterFile};

use crate::attention::{
    AttentionMicro, AttentionSchedule, AttentionSink, AttentionSpec, KeyBound, KeyLayout, LaneTile,
    MaskKind, QOperand, ScoreTransform, WaveTile, attention_body,
};
use crate::contraction::{
    DenseTiledWeight, RegisterTiling, TiledExtents, TiledShape, dense_tiled_contraction,
};
use crate::helpers::WeightLayout;

/// How a gfx1151 body rounds to a fragment dtype: the ROCm caps' kept conversions (POOT-1458).
const ROCM_ROUNDING: poot_target::RoundingConversions = poot_target::RoundingConversions::F16;

/// The caps of a gfx1151 ROCm agent: its matrix units and its register file, so the planner's register budget is the
/// one it derives from the device's own file.
fn gfx1151_caps() -> DeviceCaps {
    poot_test_util::device_caps::default_caps_for(Backend::AmdGcn(AmdArch::gfx1151()))
}

/// `(vgprs, spilled vgprs)` of the kernel in code object `hsaco`, from `llvm-readelf --notes` of the toolchain
/// that compiled it (next to `$POOT_AMD_CLANG`), or `None` when that tool is absent.
fn vgprs(hsaco: &Path) -> Option<(u32, u32)> {
    let clang = std::env::var_os("POOT_AMD_CLANG")?;
    let readelf = Path::new(&clang).parent()?.join("llvm-readelf");
    let notes = std::process::Command::new(readelf)
        .arg("--notes")
        .arg(hsaco)
        .output()
        .ok()?;
    let notes = String::from_utf8_lossy(&notes.stdout);
    let field = |name: &str| -> u32 {
        notes
            .lines()
            .find_map(|line| line.trim().strip_prefix(name))
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or_else(|| panic!("no {name} in the code object notes:\n{notes}"))
    };
    Some((field(".vgpr_count:"), field(".vgpr_spill_count:")))
}

/// Compile `body` for gfx1151 into a fresh directory and return the code object; `None` (a reported skip) without
/// the AMD toolchain.
fn compiled_code_object(body: &poot_kernel_ir::Body) -> Option<std::path::PathBuf> {
    let arch = AmdArch::gfx1151();
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-unit-test")
        .join(format!("{}_registers", body.name));
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, &body.name, Target::AmdGcn(arch));
    crate::unit_helpers::compile_amdgcn_or_skip(&body.name, body, arch, &out).then_some(out)
}

/// The assembly of code object `hsaco`, from the `llvm-objdump` next to `$POOT_AMD_CLANG`, or `None` when that tool
/// is absent.
fn disassembly(hsaco: &Path) -> Option<String> {
    let clang = std::env::var_os("POOT_AMD_CLANG")?;
    let objdump = Path::new(&clang).parent()?.join("llvm-objdump");
    let listing = std::process::Command::new(objdump)
        .args(["-d", "--mcpu=gfx1151"])
        .arg(hsaco)
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&listing.stdout).into_owned())
}

/// Compile `body` for gfx1151 into a fresh directory and return its registers; `None` (a reported skip) without
/// the AMD toolchain.
fn compiled_registers(body: &poot_kernel_ir::Body) -> Option<(u32, u32)> {
    let out = compiled_code_object(body)?;
    let registers = vgprs(&out);
    if registers.is_none() {
        eprintln!(
            "POOT-SKIP poot-kernelgen: {} (no llvm-readelf next to POOT_AMD_CLANG)",
            body.name
        );
    }
    registers
}

/// The 16-head, 8-key/value-head, 128-wide causal prefill attention of 8192 positions under the tiled schedule
/// `br` positions by `bc` keys per block, 256 lanes, with the given score and output lane tiles.
fn tiled_spec(br: usize, bc: usize, score: (usize, usize), out: (usize, usize)) -> AttentionSpec {
    AttentionSpec {
        kv: crate::KvLane::F16,
        batch: 1,
        hq: 16,
        n_rep: 2,
        lq: 8192,
        lk: 8192,
        dk: 128,
        dv: 128,
        key: KeyLayout::Single,
        score: ScoreTransform::Scale {
            scale_bits: 0.088f32.to_bits(),
        },
        mask: MaskKind::Causal { window: None },
        bound: KeyBound::Static,
        sink: AttentionSink::Absent,
        schedule: AttentionSchedule::Tiled {
            br,
            bc,
            lanes: 256,
            micro: AttentionMicro::Scalar {
                score: LaneTile {
                    rows: score.0,
                    cols: score.1,
                },
                out: LaneTile {
                    rows: out.0,
                    cols: out.1,
                },
            },
        },
        precision: poot_graph_ir::OperandPrecision::Exact,
    }
}

/// The tiled attention at every tiling the planner can choose for a 16-head, 8-key/value-head, 128-wide
/// prefill (its rows, key blocks and lane tiles: `attention_tiling`'s candidates) spills no register. Before the
/// AMDGCN compile disabled LLVM's unroller, `br 32, bc 16` (the planner's pick on a 64 KiB LDS) took 256 VGPRs with
/// 68 spilled, `br 16, bc 32` 117 spilled. Mutation: drop `-fno-unroll-loops` from `Target::llc_args`; the
/// `br 32, bc 16` row reads `spilled 68 VGPRs`.
#[test]
fn the_tiled_attention_at_its_planned_tilings_spills_no_register() {
    let tilings = [
        (32usize, 16usize, (2usize, 2usize), (8usize, 4usize)),
        (32, 32, (2, 4), (4, 8)),
        (32, 16, (4, 1), (8, 4)),
        (16, 32, (2, 2), (4, 4)),
        (16, 16, (1, 2), (2, 8)),
    ];
    for (br, bc, score, out) in tilings {
        let spec = tiled_spec(br, bc, score, out);
        let name = format!("budget_attention_br{br}_bc{bc}_s{}x{}", score.0, score.1);
        let body = attention_body(&name, &spec, 0..spec.workgroups(), 1, ROCM_ROUNDING).unwrap();
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(
            spilled, 0,
            "{name}: spilled {spilled} VGPRs of {vgprs} (the planner's tiling must fit the register file)"
        );
    }
}

/// A wave grid and the fragments each wave holds: `(waves_m, waves_n, rows, cols)`.
type WaveShape = (usize, usize, u32, u32);

/// A fragment tiling: `(br, bc, lanes, score waves, output waves, Q)` of `n_rep = 2`, each wave
/// `(waves_m, waves_n, rows, cols)`.
type FragmentTiling = (usize, usize, usize, WaveShape, WaveShape, QOperand);

/// The fragment tiling of the 16-head, 8-key/value-head, 128-wide prefill that the planner chooses on gfx1151 (128
/// rows over 8 waves, 32 keys a block, Q staged: a held Q does not fit beside these accumulators).
const PLANNED_FRAGMENT_TILING: FragmentTiling =
    (64, 32, 256, (8, 1, 1, 2), (4, 2, 2, 4), QOperand::Staged);

/// A tiling of the same prefill that holds Q in registers and fits the file (128 rows over 16 waves, 64 keys a
/// block): what the planner chooses where a held Q fits its narrowest workgroup.
const HELD_FRAGMENT_TILING: FragmentTiling =
    (64, 64, 512, (8, 2, 1, 2), (4, 4, 2, 2), QOperand::Held);

/// The f16 fragment attention (a declared lane map) of that shape under the given tiling.
fn fragment_spec((br, bc, lanes, score, out, q): FragmentTiling) -> AttentionSpec {
    let wave = |(waves_m, waves_n, rows, cols)| WaveTile {
        waves_m,
        waves_n,
        rows,
        cols,
    };
    let mut spec = tiled_spec(br, bc, (1, 1), (1, 1));
    spec.precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::F16);
    spec.schedule = AttentionSchedule::Tiled {
        br,
        bc,
        lanes,
        micro: AttentionMicro::Fragment {
            dtype: poot_target::FragmentDtype::F16,
            layout: poot_target::FragmentLayout::Declared(poot_target::LaneMap::Rdna3),
            score: wave(score),
            out: wave(out),
            q,
        },
    };
    spec
}

/// `spec` with the per-row state and loop bounds that cost the most registers beside the products: a sink logit per
/// head, a sliding window and a live length bound. The planner chooses a tiling without looking at them, so a tiling
/// it holds must fit the file with them too.
fn with_row_state(mut spec: AttentionSpec) -> AttentionSpec {
    spec.sink = AttentionSink::PerHead;
    spec.mask = MaskKind::Causal { window: Some(4096) };
    spec.bound = KeyBound::Live;
    spec
}

/// The fragment tiled attention (f16 operands on the gfx11 matrix unit, a declared lane map) at the tilings the
/// planner chooses for a 16-head, 8-key/value-head, 128-wide prefill spills no register, bare and with a sink, a
/// sliding window and a live bound ([`with_row_state`]): the score and output accumulator fragments, a held Q, the
/// operand fragments of a step and the softmax's per-row state fit the file beside each other (174 to 241 VGPRs on
/// gfx1151). Mutation: raise `trip_steps` for a fragment to 8; the hoisted operand loads of eight steps read
/// `spilled`.
#[test]
fn the_fragment_attention_at_its_planned_tilings_spills_no_register() {
    let tilings = [
        (
            32usize,
            32usize,
            256usize,
            (4, 2, 1, 1),
            (4, 2, 1, 4),
            QOperand::Staged,
        ),
        (32, 16, 128, (4, 1, 1, 1), (2, 2, 2, 4), QOperand::Staged),
        (16, 32, 128, (2, 2, 1, 1), (2, 2, 1, 4), QOperand::Staged),
        (64, 16, 256, (8, 1, 1, 1), (4, 2, 2, 4), QOperand::Staged),
        (64, 32, 512, (8, 2, 1, 1), (4, 4, 2, 2), QOperand::Held),
        HELD_FRAGMENT_TILING,
        PLANNED_FRAGMENT_TILING,
    ];
    for tiling in tilings {
        let (br, bc, lanes, .., q) = tiling;
        for (variant, spec) in [
            ("bare", fragment_spec(tiling)),
            ("row_state", with_row_state(fragment_spec(tiling))),
        ] {
            let name = format!("budget_fragment_attention_br{br}_bc{bc}_l{lanes}_{q:?}_{variant}");
            let body =
                attention_body(&name, &spec, 0..spec.workgroups(), 1, ROCM_ROUNDING).unwrap();
            let Some((vgprs, spilled)) = compiled_registers(&body) else {
                return;
            };
            eprintln!("{name}: {vgprs} VGPRs, {spilled} spilled");
            assert_eq!(
                spilled, 0,
                "{name}: spilled {spilled} VGPRs of {vgprs} (the planner's tiling must fit the register file)"
            );
        }
    }
}

/// The planner's register rule refuses a held Q where the lane's held fragments pass half of its registers, and the
/// compiler shows the refusal is needed: 128 rows over 8 waves with Q held (64 Q registers beside 80 accumulator
/// registers, over the 128 of a 256-lane workgroup on gfx1151) takes 255 of the 256 VGPRs bare and spills once the
/// body carries a sink, a sliding window and a live bound (15 VGPRs on gfx1151). The planner's tiling does not
/// depend on those, so a rule that admitted this tiling would spill some attention it plans. The registers outside
/// the held ones measured 111 to 133 VGPRs over the planned tilings, the half the rule keeps. Mutation: stage Q
/// whatever the schedule says (`TiledLayout::of` binds `QOperand::Staged` for a fragment micro-tile); the tiling
/// compiles to 228 VGPRs and this row goes red.
#[test]
fn a_held_query_the_planner_refuses_spills() {
    let spec = with_row_state(fragment_spec((
        64,
        32,
        256,
        (8, 1, 1, 2),
        (4, 2, 2, 4),
        QOperand::Held,
    )));
    let body = attention_body(
        "budget_fragment_attention_refused_held_q",
        &spec,
        0..spec.workgroups(),
        1,
        ROCM_ROUNDING,
    )
    .unwrap();
    let Some((vgprs, spilled)) = compiled_registers(&body) else {
        return;
    };
    eprintln!("{}: {vgprs} VGPRs, {spilled} spilled", body.name);
    assert!(
        spilled > 0,
        "the refused tiling fits ({vgprs} VGPRs): the held-register rule refuses what the file holds"
    );
}

/// V is staged `[channel][key]` from `[key][channel]` rows: each lane stages a run of 16 keys of one channel as two
/// 16-byte stores (POOT-1447), where one 2-byte store per key put adjacent lanes `(bc + 8) / 2` words apart.
/// Mutation: stage V `StageOrder::AlongColumns` in `tiled_body`; the listing has no 16-byte store.
#[test]
fn the_fragment_attention_stages_v_in_wide_stores() {
    for tiling in [PLANNED_FRAGMENT_TILING, HELD_FRAGMENT_TILING] {
        let spec = fragment_spec(tiling);
        let body = attention_body(
            "budget_fragment_attention_v_stores",
            &spec,
            0..spec.workgroups(),
            1,
            ROCM_ROUNDING,
        )
        .unwrap();
        let Some(hsaco) = compiled_code_object(&body) else {
            return;
        };
        let Some(listing) = disassembly(&hsaco) else {
            eprintln!(
                "POOT-SKIP poot-kernelgen: {} (no llvm-objdump next to POOT_AMD_CLANG)",
                body.name
            );
            return;
        };
        let wide = listing
            .lines()
            .filter(|line| line.trim_start().starts_with("ds_store_b128"))
            .count();
        assert_eq!(wide, 2, "{tiling:?}: 16-byte V stores per lane");
    }
}

/// Every operand fragment of the planned fragment attention comes from workgroup memory in 16-byte loads: Q, K, P
/// and V are each staged with the 16 depth values of a fragment lane's row adjacent. A tile staged the other way
/// reads a fragment lane's 16 values one 2-byte load at a time (`ds_load_u16`: 192 a key block for the planned tiling,
/// when P and V were staged `K`-major), which measured 5% of the bank's 8192-position cell. Mutation: stage `P` and
/// `V` `K`-major in `TiledLayout::tiles` (`Staged::k_major(P_ARRAY, rows + FRAGMENT_PAD)`, the same for `V` with
/// `dv + FRAGMENT_PAD`) and the disassembly shows `ds_load_u16`.
#[test]
fn the_fragment_attention_loads_every_operand_fragment_wide() {
    let spec = fragment_spec(PLANNED_FRAGMENT_TILING);
    let body = attention_body(
        "budget_fragment_attention_wide",
        &spec,
        0..spec.workgroups(),
        1,
        ROCM_ROUNDING,
    )
    .unwrap();
    let Some(hsaco) = compiled_code_object(&body) else {
        return;
    };
    let Some(listing) = disassembly(&hsaco) else {
        eprintln!(
            "POOT-SKIP poot-kernelgen: {} (no llvm-objdump next to POOT_AMD_CLANG)",
            body.name
        );
        return;
    };
    let count = |mnemonic: &str| {
        listing
            .lines()
            .filter(|line| line.trim_start().starts_with(mnemonic))
            .count()
    };
    assert_eq!(
        count("ds_load_u16"),
        0,
        "an operand fragment is loaded 2 bytes at a time"
    );
    assert!(
        count("ds_load_b128") > 0,
        "no 16-byte workgroup-memory load in the fragment attention"
    );
}

/// The register-tiled dense contraction at the tiling the planner chooses for each projection of a prefill
/// spills no register and keeps its occupancy-friendly register count. Planned on the gfx1151 caps, so the tile is
/// bounded by the budget derived from the device's register file.
#[test]
fn the_dense_register_tiled_contraction_spills_no_register() {
    let caps = gfx1151_caps();
    for (label, m, n, k) in [
        ("qkv", 8192usize, 4096usize, 2560usize),
        ("down", 8192, 2560, 9728),
    ] {
        let shape = TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: WeightLayout::Nk,
            rows: crate::TiledRows::Dense,
            terms: crate::precision::StagedTerms::ONE,
            weight: crate::StagedWeight::Dense { bits: 16 },
            activation_bits: 32,
        };
        let tiling = RegisterTiling::choose(shape, &caps);
        let name = format!("budget_dense_{label}");
        let body = dense_tiled_contraction(
            &name,
            DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
            tiling.schedule(),
            TiledExtents {
                k,
                out_per_block: n,
            },
            None,
            crate::contraction::TiledRead {
                precision: poot_graph_ir::OperandPrecision::Exact,
                activation_stored: None,
                conversions: caps.rounding_conversions,
            },
        )
        .unwrap();
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(spilled, 0, "{name}: spilled {spilled} VGPRs of {vgprs}");
    }
}

/// The fragment dense contraction at the tiling the planner chooses for each narrowed prefill projection (bf16
/// operands on the gfx11 matrix unit) spills no register: its accumulator fragments, the operand fragments of a step,
/// the round in flight and the staging addresses fit the file. Over a bf16 `[N, K]` weight (64-deep rounds, whole or
/// with a `K` tail) and an f32 `[K, N]` one (32-deep rounds of transposed four-column blocks). Mutation: raise
/// `ACCUMULATOR_SHARE` to 2 (`poot-target`'s `register_file.rs`); the planner's pick for the long projections holds
/// 128 accumulator registers beside its operands and the row reads `spilled`.
#[test]
fn the_fragment_dense_contraction_spills_no_register() {
    let nk = DenseTiledWeight::Nk(BufferStorage::bf16_packed());
    for (label, weight, m, n, k) in [
        ("qkv", nk, 8192usize, 4096usize, 2560usize),
        ("down", nk, 8192, 2560, 9728),
        ("short", nk, 128, 2048, 2048),
        ("tail", nk, 2048, 3072, 1000),
        ("long_kn", DenseTiledWeight::KnF32, 1024, 2048, 2048),
    ] {
        let name = format!("budget_fragment_{label}");
        let body = planned_fragment_contraction(&name, weight, (m, n, k));
        let Some((vgprs, spilled)) = compiled_registers(&body) else {
            return;
        };
        assert_eq!(spilled, 0, "{name}: spilled {spilled} VGPRs of {vgprs}");
    }
}

/// The narrowed (bf16) fragment contraction of an `m x k` activation over an `n`-column `weight` at the tiling the
/// planner chooses on the gfx11 matrix unit.
fn planned_fragment_contraction(
    name: &str,
    weight: DenseTiledWeight,
    (m, n, k): (usize, usize, usize),
) -> poot_kernel_ir::Body {
    let caps = gfx1151_caps();
    let precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::Bf16);
    let (layout, stored) = match weight {
        DenseTiledWeight::Nk(storage) => (WeightLayout::Nk, crate::stored_fragment_dtype(storage)),
        DenseTiledWeight::KnF32 => (WeightLayout::Kn, None),
    };
    let shape = TiledShape {
        blocks: 1,
        m_per_block: m,
        out_per_block: n,
        k,
        layout,
        rows: crate::TiledRows::Dense,
        terms: crate::precision::StagedTerms::of(precision, None, stored),
        weight: crate::StagedWeight::Dense {
            bits: weight.bits(),
        },
        activation_bits: 32,
    };
    let tiling = RegisterTiling::choose(shape, &caps);
    assert!(
        matches!(tiling.schedule(), crate::Schedule::FragmentTiled { .. }),
        "{name}: {:?}",
        tiling.schedule()
    );
    dense_tiled_contraction(
        name,
        weight,
        tiling.schedule(),
        TiledExtents {
            k,
            out_per_block: n,
        },
        None,
        crate::contraction::TiledRead {
            precision,
            activation_stored: None,
            conversions: caps.rounding_conversions,
        },
    )
    .unwrap()
}

/// Every operand fragment of the planned fragment contraction over an f32 `[K, N]` weight comes from workgroup
/// memory in 16-byte loads: the weight is staged transposed (its 16 depth values of a column adjacent), so its
/// fragments load like the activation's. Staged `K`-major, a weight fragment was 16 two-byte loads (`ds_load_u16`,
/// 128 a round) and the VALU that packs them, the long-gemm cell's 35 non-WMMA VALU per `v_wmma`
/// (POOT-1443). Mutation: make `MicroTile::staged_major` return `StagedMajor::K` for a fragment;
/// the disassembly shows `ds_load_u16`.
#[test]
fn the_fragment_contraction_over_a_kn_weight_loads_every_operand_fragment_wide() {
    let body = planned_fragment_contraction(
        "budget_fragment_kn_wide",
        DenseTiledWeight::KnF32,
        (1024, 2048, 2048),
    );
    let Some(hsaco) = compiled_code_object(&body) else {
        return;
    };
    let Some(listing) = disassembly(&hsaco) else {
        eprintln!(
            "POOT-SKIP poot-kernelgen: {} (no llvm-objdump next to POOT_AMD_CLANG)",
            body.name
        );
        return;
    };
    let count = |mnemonic: &str| {
        listing
            .lines()
            .filter(|line| line.trim_start().starts_with(mnemonic))
            .count()
    };
    assert_eq!(
        count("ds_load_u16"),
        0,
        "an operand fragment is loaded 2 bytes at a time"
    );
    assert!(
        count("ds_load_b128") > 0,
        "no 16-byte workgroup-memory load in the fragment contraction"
    );
}

/// A body whose generator unrolled it carries the no-unroll loop metadata on AMDGCN; the same body on NVPTX and a
/// body the backend may unroll (the streaming attention a decode runs, which LLVM unrolls to good effect: the
/// low-head decode attention took 2.3x longer without it) carry none. Mutation: make `emit_terminator` skip
/// the `Unrolling::Generator` test; the streaming body gains the metadata and its row goes red.
#[test]
fn only_a_body_the_generator_unrolled_carries_the_no_unroll_metadata() {
    let arch = AmdArch::gfx1151();
    let tiled = tiled_spec(32, 16, (2, 2), (8, 4));
    let tiled_body =
        attention_body("tiled", &tiled, 0..tiled.workgroups(), 1, ROCM_ROUNDING).unwrap();
    let ir = poot_codegen::emit_llvm_ir(&tiled_body, Target::AmdGcn(arch)).unwrap();
    assert!(
        ir.contains(", !llvm.loop !0"),
        "no loop carries the metadata"
    );
    assert!(ir.contains("!1 = !{!\"llvm.loop.unroll.disable\"}"));
    let nvptx = poot_codegen::emit_llvm_ir(&tiled_body, Target::Nvptx).unwrap();
    assert!(!nvptx.contains("llvm.loop"), "NVPTX takes no loop metadata");

    let streaming = AttentionSpec {
        lq: 1,
        schedule: AttentionSchedule::Streaming { lanes: 64 },
        ..tiled
    };
    let streaming_body = attention_body(
        "streaming",
        &streaming,
        0..streaming.workgroups(),
        1,
        ROCM_ROUNDING,
    )
    .unwrap();
    let ir = poot_codegen::emit_llvm_ir(&streaming_body, Target::AmdGcn(arch)).unwrap();
    assert!(
        !ir.contains("llvm.loop"),
        "the backend unrolls a streaming body"
    );
}

/// The instructions of the `K` loop of the AMDGCN kernel `body` compiles to: the last backward branch whose body
/// multiplies on the matrix unit, from its target to the branch, disassembled with the `llvm-objdump` next to
/// `$POOT_AMD_CLANG`. `None` (a reported skip) without the toolchain.
fn compiled_k_loop(body: &poot_kernel_ir::Body) -> Option<Vec<String>> {
    let arch = AmdArch::gfx1151();
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-unit-test")
        .join(format!("{}_loop", body.name));
    std::fs::create_dir_all(&dir).unwrap();
    let out = artifact_path(&dir, &body.name, Target::AmdGcn(arch));
    if !crate::unit_helpers::compile_amdgcn_or_skip(&body.name, body, arch, &out) {
        return None;
    }
    let clang = std::env::var_os("POOT_AMD_CLANG")?;
    let objdump = Path::new(&clang).parent()?.join("llvm-objdump");
    let Ok(listing) = std::process::Command::new(objdump)
        .args(["-d", "--mcpu=gfx1151"])
        .arg(&out)
        .output()
    else {
        eprintln!(
            "POOT-SKIP poot-kernelgen: {} (no llvm-objdump next to POOT_AMD_CLANG)",
            body.name
        );
        return None;
    };
    // `<mnemonic> <operands> // <address>: <encoding>`; a branch's operand is its signed word offset from the next
    // instruction.
    let instructions: Vec<(u64, String)> = String::from_utf8_lossy(&listing.stdout)
        .lines()
        .filter_map(|line| {
            let (text, comment) = line.split_once("//")?;
            let address = u64::from_str_radix(comment.trim().split(':').next()?.trim(), 16).ok()?;
            Some((address, text.trim().to_string()))
        })
        .collect();
    let target_of = |at: usize| -> Option<u64> {
        let (address, text) = &instructions[at];
        let offset: u16 = text
            .strip_prefix("s_cbranch_")?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()?;
        Some((*address as i64 + 4 + 4 * i64::from(offset as i16)) as u64)
    };
    (0..instructions.len()).rev().find_map(|at| {
        let target = target_of(at).filter(|&target| target < instructions[at].0)?;
        let start = instructions
            .iter()
            .position(|(address, _)| *address == target)?;
        let body: Vec<String> = instructions[start..=at]
            .iter()
            .map(|(_, text)| text.clone())
            .collect();
        body.iter()
            .any(|text| text.starts_with("v_wmma"))
            .then_some(body)
    })
}

/// The pipelined fragment contraction issues the next round's loads before the current round's multiplies and
/// waits for them only after the last multiply, on gfx1151: in the compiled `K` loop every `global_load` comes
/// before the first `v_wmma` and no `s_waitcnt vmcnt` sits between the first load and the last multiply. Checked at
/// the planner's tiling of a narrowed `[N, K]` bf16 projection and of a `[K, N]` f32 one. Mutation: drop the
/// `SchedulingBoundary` ahead of the multiplies in `tiled_body`; the machine scheduler sinks loads below the first
/// multiply and this row goes red. Mutation: emit the uses of the loaded values at the end of the latch instead of
/// at the top of the loop header; their wait lands before the multiplies and this row goes red.
#[test]
fn the_pipelined_fragment_contraction_loads_ahead_of_its_multiplies() {
    let caps = gfx1151_caps();
    let precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::Bf16);
    for (label, weight, layout, stored) in [
        (
            "nk_bf16",
            DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
            WeightLayout::Nk,
            Some(poot_graph_ir::FragmentDtype::Bf16),
        ),
        ("kn_f32", DenseTiledWeight::KnF32, WeightLayout::Kn, None),
    ] {
        let (m, n, k) = (2048usize, 3072usize, 1024usize);
        let shape = TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout,
            rows: crate::TiledRows::Dense,
            terms: crate::precision::StagedTerms::of(precision, None, stored),
            weight: crate::StagedWeight::Dense {
                bits: weight.bits(),
            },
            activation_bits: 32,
        };
        let tiling = RegisterTiling::choose(shape, &caps);
        let name = format!("pipelined_{label}");
        let body = dense_tiled_contraction(
            &name,
            weight,
            tiling.schedule(),
            TiledExtents {
                k,
                out_per_block: n,
            },
            None,
            crate::contraction::TiledRead {
                precision,
                activation_stored: None,
                conversions: caps.rounding_conversions,
            },
        )
        .unwrap();
        let Some(k_loop) = compiled_k_loop(&body) else {
            return;
        };
        let first = |prefix: &str| k_loop.iter().position(|text| text.starts_with(prefix));
        let last = |prefix: &str| k_loop.iter().rposition(|text| text.starts_with(prefix));
        let (Some(first_load), Some(last_load), Some(first_wmma), Some(last_wmma)) = (
            first("global_load"),
            last("global_load"),
            first("v_wmma"),
            last("v_wmma"),
        ) else {
            panic!(
                "{name}: no loads or multiplies in the K loop:\n{}",
                k_loop.join("\n")
            );
        };
        assert!(
            last_load < first_wmma,
            "{name}: a load issues after a multiply:\n{}",
            k_loop.join("\n")
        );
        let early_wait = k_loop[first_load..last_wmma]
            .iter()
            .find(|text| text.starts_with("s_waitcnt") && text.contains("vmcnt"));
        assert!(
            early_wait.is_none(),
            "{name}: {early_wait:?} waits for the loads before the multiplies:\n{}",
            k_loop.join("\n")
        );
    }
}

/// The narrowed `[N, K]` bf16-weight dense shape of an `m x n x k` problem.
fn bf16_nk_shape(
    precision: poot_graph_ir::OperandPrecision,
    (m, n, k): (usize, usize, usize),
) -> TiledShape {
    TiledShape {
        blocks: 1,
        m_per_block: m,
        out_per_block: n,
        k,
        layout: WeightLayout::Nk,
        rows: crate::TiledRows::Dense,
        terms: crate::precision::StagedTerms::of(
            precision,
            None,
            Some(poot_graph_ir::FragmentDtype::Bf16),
        ),
        weight: crate::StagedWeight::Dense { bits: 16 },
        activation_bits: 32,
    }
}

/// The dense `[N, K]` bf16 contraction `tiling` runs over an `m x n x k` problem at `precision`.
fn bf16_nk_body(
    name: &str,
    tiling: RegisterTiling,
    precision: poot_graph_ir::OperandPrecision,
    (n, k): (usize, usize),
    caps: &DeviceCaps,
) -> poot_kernel_ir::Body {
    dense_tiled_contraction(
        name,
        DenseTiledWeight::Nk(BufferStorage::bf16_packed()),
        tiling.schedule(),
        TiledExtents {
            k,
            out_per_block: n,
        },
        None,
        crate::contraction::TiledRead {
            precision,
            activation_stored: None,
            conversions: caps.rounding_conversions,
        },
    )
    .unwrap()
}

/// Whether `tiling` is a fragment tiling.
fn is_fragment(tiling: RegisterTiling) -> bool {
    matches!(tiling.schedule(), crate::Schedule::FragmentTiled { .. })
}

/// The register budget derived from the device's file is the boundary of spilling on gfx1151: every tiling the
/// planner weighs holds at most the 64 accumulators a quarter of the 256-register lane allows, including the
/// 8-fragment tile, and the tiling a device with a twice as large lane would add (the cheapest of the long
/// projections at 128 accumulators, 16 fragments) spills on gfx1151. Mutation: make `ACCUMULATOR_SHARE` 2
/// (`poot-target`'s `register_file.rs`); the planner's candidates reach 128 accumulators.
#[test]
fn the_derived_register_budget_is_the_boundary_of_spilling() {
    let caps = gfx1151_caps();
    let precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::Bf16);
    let (m, n, k) = (8192usize, 4096usize, 2560usize);
    let shape = bf16_nk_shape(precision, (m, n, k));
    let planned: Vec<RegisterTiling> = RegisterTiling::candidates(shape, &caps).collect();
    let most = planned
        .iter()
        .map(|tiling| tiling.accumulator_registers())
        .max()
        .unwrap();
    assert_eq!(
        most, 64,
        "the derived budget admits 64 accumulators and no more"
    );
    assert!(
        planned
            .iter()
            .any(|&tiling| is_fragment(tiling) && tiling.accumulator_registers() == 64),
        "the 8-fragment tile is within the budget"
    );
    // The same file with a lane twice as wide: a quarter of it is 128 accumulators.
    let wide = DeviceCaps {
        register_file: Queried::Known(RegisterFile {
            per_lane: 512,
            per_workgroup: 2 * caps.register_file.known().unwrap().per_workgroup,
        }),
        ..caps
    };
    let over = RegisterTiling::choose(shape, &wide);
    assert_eq!(
        over.accumulator_registers(),
        128,
        "{:?}: the unbounded pick holds 16 fragments",
        over.schedule()
    );
    let body = bf16_nk_body(
        "budget_over_budget_fragment",
        over,
        precision,
        (n, k),
        &caps,
    );
    let Some((vgprs, spilled)) = compiled_registers(&body) else {
        return;
    };
    eprintln!("{}: {vgprs} VGPRs, {spilled} spilled", body.name);
    assert!(
        spilled > 0,
        "the 128-accumulator tiling fits ({vgprs} VGPRs): the budget refuses what the file holds"
    );
}

/// The registers `ptxas -v` reports for one PTX kernel.
struct PtxRegisters {
    /// Registers per thread.
    used: u32,
    /// Bytes the kernel spills to local memory (stores and loads).
    spilled_bytes: u32,
}

/// The CUDA toolkit image `ptxas` is taken from: the ptxas of the image is the NVIDIA compiler the driver's JIT
/// shares its register allocator with, and no pod or GPU is needed to run it.
const PTXAS_IMAGE: &str = "docker.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04";

/// What `ptxas -v -arch=sm_80` reports for `body` compiled to PTX, or `None` (a reported skip) where `llc`, `podman`
/// or the CUDA image is absent. The PTX is the kernel `compile` writes for `Target::Nvptx`, the same artifact a pod
/// loads.
fn ptxas_registers(body: &poot_kernel_ir::Body) -> Option<PtxRegisters> {
    let skip = |why: &str| {
        poot_test_util::skip_report(
            "poot-kernelgen",
            &format_args!("{} ptxas -v: {why}", body.name),
        );
        None
    };
    if std::process::Command::new("llc")
        .arg("--version")
        .output()
        .is_err()
    {
        return skip("llc not on PATH");
    }
    let image_present = std::process::Command::new("podman")
        .args(["image", "exists", PTXAS_IMAGE])
        .status()
        .is_ok_and(|status| status.success());
    if !image_present {
        return skip("no local podman image nvidia/cuda");
    }
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-unit-test")
        .join(format!("{}_ptxas", body.name));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = dir.join(format!("{}.ptx", body.name));
    poot_codegen::compile(body, Target::Nvptx, &ptx)
        .unwrap_or_else(|error| panic!("{}: NVPTX compile: {error}", body.name));
    let mount = format!("{}:/work:ro,z", dir.display());
    let output = std::process::Command::new("podman")
        .args(["run", "--rm", "-v", &mount, PTXAS_IMAGE])
        .args(["ptxas", "-v", "-arch=sm_80", "-o", "/dev/null"])
        .arg(format!("/work/{}.ptx", body.name))
        .output()
        .expect("podman runs");
    let report = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "ptxas failed:\n{report}");
    // `ptxas info : Used 64 registers, ...` and `ptxas info : 0 bytes stack frame, 0 bytes spill stores, 0 bytes
    // spill loads`.
    let number_before = |unit: &str| -> u32 {
        report
            .split_once(unit)
            .and_then(|(before, _)| before.trim_end().rsplit(' ').next()?.parse().ok())
            .unwrap_or_else(|| panic!("no `{unit}` in the ptxas report:\n{report}"))
    };
    Some(PtxRegisters {
        used: number_before(" registers"),
        spilled_bytes: number_before(" bytes spill stores") + number_before(" bytes spill loads"),
    })
}

/// On sm_80 the derived budget keeps the 8 x 8 scalar tile and the 8-fragment tile, and `ptxas` spills neither: the
/// `ceil` of a quarter of an NVIDIA thread's 255 registers is 64, and a floor would drop the scalar tile (63).
/// Compiled offline (`ptxas -v -arch=sm_80` from the local CUDA image), so no pod is needed. Two sides: the planner's
/// candidates on the sm_80 caps include an 8 x 8 scalar tile and reach no more than 64 accumulators, and the tiles
/// at 64 accumulators spill nothing. Mutation: make `div_ceil` a floor in `RegisterBudget::accumulators`
/// (`poot-target`'s `register_file.rs`); the candidates lose the 8 x 8 tile and the first assertion names it.
#[test]
fn the_budget_keeps_sm_80_at_its_8_by_8_tile_without_a_spill() {
    let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
    let (m, n, k) = (8192usize, 4096usize, 2560usize);
    for (label, precision) in [
        ("scalar", poot_graph_ir::OperandPrecision::Exact),
        (
            "fragment",
            poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::Bf16),
        ),
    ] {
        let shape = bf16_nk_shape(precision, (m, n, k));
        let planned: Vec<RegisterTiling> = RegisterTiling::candidates(shape, &caps).collect();
        assert_eq!(
            planned
                .iter()
                .map(|tiling| tiling.accumulator_registers())
                .max(),
            Some(64),
            "{label}: the sm_80 budget is 64 accumulators"
        );
        let widest = planned
            .iter()
            .copied()
            .filter(|&tiling| is_fragment(tiling) == (label == "fragment"))
            .max_by_key(|tiling| tiling.accumulator_registers())
            .unwrap_or_else(|| panic!("no {label} tiling on sm_80"));
        assert_eq!(widest.accumulator_registers(), 64, "{label}: {widest:?}");
        let name = format!("budget_sm80_{label}");
        let body = bf16_nk_body(&name, widest, precision, (n, k), &caps);
        let Some(registers) = ptxas_registers(&body) else {
            return;
        };
        eprintln!(
            "{name}: {} registers, {} spilled bytes",
            registers.used, registers.spilled_bytes
        );
        assert_eq!(
            registers.spilled_bytes, 0,
            "{name}: ptxas spills {} bytes of {} registers (the planner's tiling must fit the register file)",
            registers.spilled_bytes, registers.used
        );
    }
    // The other side: the 16-fragment tile a lane of twice the registers would be offered spills on sm_80.
    let precision = poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::Bf16);
    let wide = DeviceCaps {
        register_file: Queried::Known(RegisterFile {
            per_lane: 510,
            per_workgroup: 2 * caps.register_file.known().unwrap().per_workgroup,
        }),
        ..caps
    };
    let over = RegisterTiling::choose(bf16_nk_shape(precision, (m, n, k)), &wide);
    assert_eq!(over.accumulator_registers(), 128, "{:?}", over.schedule());
    let body = bf16_nk_body("budget_sm80_over", over, precision, (n, k), &caps);
    let Some(registers) = ptxas_registers(&body) else {
        return;
    };
    eprintln!(
        "budget_sm80_over: {} registers, {} spilled bytes",
        registers.used, registers.spilled_bytes
    );
    assert!(
        registers.spilled_bytes > 0,
        "the 128-accumulator tiling fits in {} registers: the budget refuses what the file holds",
        registers.used
    );
}

/// The grouped shape of a routed projection of `m` rows over a packed `[n, k]` expert stack of `format` (512 experts),
/// narrowed to f16 as the planner narrows it on gfx1151, its tiles `block` rows.
fn packed_expert_shape(
    format: poot_quant::format::WeightFormat,
    (m, n, k): (usize, usize, usize),
    block: usize,
) -> TiledShape {
    TiledShape {
        blocks: 1,
        m_per_block: m,
        out_per_block: n,
        k,
        layout: WeightLayout::Nk,
        rows: crate::TiledRows::Grouped { groups: 513, block },
        terms: crate::precision::StagedTerms::of(
            poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::F16),
            None,
            None,
        ),
        weight: crate::StagedWeight::Packed(format),
        activation_bits: 32,
    }
}

/// The narrowed indexed packed contraction body of `tiling` over `format` with `n` columns of `k`.
fn packed_expert_body(
    name: &str,
    format: poot_quant::format::WeightFormat,
    tiling: RegisterTiling,
    (n, k): (usize, usize),
) -> poot_kernel_ir::Body {
    let meta = [k as u32, n as u32, 8208, 0, 0, 512, 1024];
    crate::packed::packed_contraction(
        name,
        format,
        tiling.schedule(),
        &meta,
        None,
        crate::PackedRows::Indexed,
        poot_graph_ir::OperandPrecision::Narrow(poot_graph_ir::FragmentDtype::F16),
        ROCM_ROUNDING,
    )
    .unwrap()
}

/// The expert projections of Flash-Next: the IQ2_XS up projection (N 640, K 2560) and the IQ4_NL down projection
/// (N 2560, K 640).
const PACKED_EXPERTS: [(poot_quant::format::WeightFormat, usize, usize); 2] = [
    (poot_quant::format::WeightFormat::Iq2_Xs, 640, 2560),
    (poot_quant::format::WeightFormat::Iq4_Nl, 2560, 640),
];

/// POOT-1505: a tiling the planner counts within the held share of the lane's registers across its `K` loop
/// (`RegisterTiling::loop_registers`: its accumulators, the next round's reads, the values they decode to, against
/// `RegisterBudget::loop_registers`, 128) compiles on gfx1151 with no register spilled: every fragment geometry of a
/// 16-row expert tile and the 64- and 128-row tiles, at both round depths, over both Flash-Next expert formats. The
/// bound is safe, not tight: a few tilings it refuses compile just under the file. Mutation: count a packed run's
/// reads alone (drop the decoded `run` values from `StagedWeight::run_registers`); an admitted tiling reads `spilled`.
#[test]
fn a_packed_expert_tiling_within_the_held_registers_spills_none() {
    let caps = gfx1151_caps();
    let layout = caps
        .matrix_units
        .known()
        .and_then(|units| Some(units.config(poot_graph_ir::FragmentDtype::F16)?.layout))
        .expect("gfx1151 has an f16 matrix unit");
    let geometries = [
        (1u32, 1u32, 1u32, 1u32),
        (1, 1, 1, 2),
        (1, 1, 1, 4),
        (1, 2, 1, 1),
        (1, 2, 1, 2),
        (1, 2, 1, 4),
        (2, 2, 2, 4),
        (4, 2, 2, 4),
    ];
    let mut admitted = 0;
    for (format, n, k) in PACKED_EXPERTS {
        for (waves_m, waves_n, rows, cols) in geometries {
            for depth in [32u32, 64] {
                let tiling = RegisterTiling::of(crate::Schedule::FragmentTiled {
                    waves_m: crate::contraction::TileSize::new(waves_m).unwrap(),
                    waves_n: crate::contraction::TileSize::new(waves_n).unwrap(),
                    tile: crate::FragmentTile {
                        dtype: poot_graph_ir::FragmentDtype::F16,
                        layout,
                        rows,
                        cols,
                    },
                    run: crate::contraction::RunLength::dividing(k),
                    depth,
                })
                .unwrap();
                let shape = packed_expert_shape(format, (1280, n, k), tiling.output_tile().0);
                let counted = tiling.loop_registers_over(shape).unwrap();
                let budget = caps
                    .gemm_launch()
                    .register_budget
                    .loop_registers(tiling.workgroup_lanes() as u32);
                let (tile_rows, tile_cols) = tiling.output_tile();
                let name = format!(
                    "budget_packed_{format:?}_{tile_rows}x{tile_cols}_w{waves_m}x{waves_n}_d{depth}"
                );
                let body = packed_expert_body(&name, format, tiling, (n, k));
                let Some((vgprs, spilled)) = compiled_registers(&body) else {
                    return;
                };
                eprintln!(
                    "{name}: counted {counted} of {budget}, {vgprs} VGPRs, {spilled} spilled"
                );
                if counted <= budget {
                    admitted += 1;
                    assert_eq!(
                        spilled, 0,
                        "{name}: counted {counted} of {budget} registers across the loop, compiled to {vgprs} VGPRs, {spilled} spilled"
                    );
                }
            }
        }
    }
    assert!(admitted >= 8, "the bound admits {admitted} of 32 tilings");
}

/// POOT-1505: the narrowed tiling the planner chooses for each Flash-Next expert projection, from a short prompt
/// (1280 routed rows, 128 tokens) to a long one (20480), grouped at the block it chooses, spills no register on
/// gfx1151. Before the loop-register term the 128-token up projection took the 16 x 128 tile at a 64-deep round,
/// 720 VGPRs spilled (f16). Mutation: drop the `loop_registers` clause of `RegisterTiling::runs_on`; the 128-token row
/// reads `spilled 720 VGPRs of 256`.
#[test]
fn the_planned_packed_expert_tilings_spill_no_register() {
    let caps = gfx1151_caps();
    for (format, n, k) in PACKED_EXPERTS {
        for m in [1280usize, 20480] {
            let free = packed_expert_shape(format, (m, n, k), 0);
            let block = RegisterTiling::grouped_block(&[free], &caps).expect("a grouping block");
            let shape = packed_expert_shape(format, (m, n, k), block);
            let tiling = RegisterTiling::tiling_of(shape, &caps).expect("a fragment tiling");
            let name = format!("budget_planned_packed_{format:?}_{m}");
            let body = packed_expert_body(&name, format, tiling, (n, k));
            let Some((vgprs, spilled)) = compiled_registers(&body) else {
                return;
            };
            assert_eq!(
                spilled,
                0,
                "{name} {:?}: spilled {spilled} VGPRs of {vgprs}",
                tiling.schedule()
            );
        }
    }
}

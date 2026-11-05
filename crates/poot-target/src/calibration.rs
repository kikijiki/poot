//! Calibration: the figures no driver API reports, measured once on a named device and keyed by that device's
//! identity. A device nobody measured is [`Queried::Unknown`], never given a neighbour's figures: the
//! display-watchdog budgets and the tiled-GEMM ceiling below were observed on one part (Strix Halo behind
//! RADV), and copying them to every AMD GPU chunked work a discrete card runs unsplit and fenced it with a
//! ceiling nobody observed on it. A performance figure ([`PerfCalibration`]) only ranks schedules that are all
//! correct, so a device nobody measured plans with one named [`PerfCalibration::PORTABLE`] entry and the plan
//! records that it did ([`Calibrated`]).

use crate::{
    AMD_RADV_SAFE_MAX_BUFFER_BYTES, ATTENTION_CHUNK_WORK, DECODE_GEMV_CHUNK_TRIGGER,
    KnownMiscompiles, PACKED_TILED_CHUNK_WORK, Queried, WatchdogBudget,
};

/// The lane-grid edge of the narrow register-tiled schedule: 64 lanes, one wave64, the widest workgroup every
/// device is known to run.
pub const REGISTER_LANE_EDGE: u32 = 8;

/// The lane-grid edge of the wide register-tiled schedule: 256 lanes, offered to the cost model on a device whose
/// workgroup and LDS hold it.
pub const REGISTER_LANE_EDGE_WIDE: u32 = 16;

/// The packed decode Gemv's longest lane run on a device nobody measured: the run every device launched before
/// POOT-1398. [`PerfCalibration::PORTABLE`]'s figure.
pub const PORTABLE_PACKED_GEMV_RUN: u32 = 8;

/// The packed decode Gemv's longest lane run on Strix Halo (gfx1151, through HSA or RADV), measured (POOT-1390):
/// a 32-element run (one Q4_K/Q5_K sub-block) amortises the block addressing and the factor decode over
/// four times the weights of the portable run and keeps the body within the register file on both
/// compilers.
pub const STRIX_HALO_PACKED_GEMV_RUN: u32 = 32;

/// The packed decode Gemv's longest lane run on the RTX PRO 6000 Blackwell Server (sm_120), the one NVIDIA part it
/// was measured on (POOT-1405). With the activation read in 16-byte groups and 32-bit index math, a 16-element run
/// amortises a K-quant's block decode over twice the weights of the 8-element one, while a 32-element run spreads
/// each warp-wide activation load over 16 cache lines and loses (Q4_K 980 at run 16 against 665 at 32 and 721 at
/// 8 GB/s). Another NVIDIA part is unmeasured and plans with [`PerfCalibration::PORTABLE`] until a pod sweep says
/// otherwise.
pub const SM_120_PACKED_GEMV_RUN: u32 = 16;

/// The resident lanes per compute unit the register-tiled schedule's occupancy model treats as saturating it:
/// eight 32-lane waves, four per SIMD of an RDNA compute unit (two SIMD32s) or two per partition of an NVIDIA SM
/// (four), the residency at which a unit hides its LDS and memory latency behind other waves' FMAs. Structural,
/// not measured per device.
pub const SATURATING_LANES: u32 = 256;

/// The performance figures one device's launches were measured with. A figure here only ranks schedules that are
/// all correct, so a device nobody measured plans with [`PerfCalibration::PORTABLE`] and the plan says so
/// ([`Calibrated::Portable`]); a safety figure (a watchdog budget, a miscompile ceiling) is [`Calibration`]'s and
/// stays absent for such a device.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PerfCalibration {
    /// The longest run of contiguous K elements a packed decode Gemv lane decodes per trip: the register and issue
    /// budget of the device's compiler. The planner runs the format's factor run up to this cap.
    pub packed_gemv_max_run: u32,
}

impl PerfCalibration {
    /// The prior an unmeasured device plans with. Every field is named for the device it was measured on or the
    /// launch every device ran before it was measured, never a value nobody observed.
    pub const PORTABLE: Self = Self {
        packed_gemv_max_run: PORTABLE_PACKED_GEMV_RUN,
    };

    /// Strix Halo (gfx1151), through HSA or behind RADV: the device the Gemv figures were measured on.
    pub const STRIX_HALO: Self = Self {
        packed_gemv_max_run: STRIX_HALO_PACKED_GEMV_RUN,
    };

    /// The RTX PRO 6000 Blackwell Server (compute capability 12.0). Measured for the packed Gemv run only; a field
    /// it did not measure is spelled out as [`Self::PORTABLE`]'s, so a reader sees which figures are borrowed.
    pub const SM_120: Self = Self {
        packed_gemv_max_run: SM_120_PACKED_GEMV_RUN,
    };
}

/// A device whose performance figures were measured: the identity [`Calibrated::Measured`] names, one per entry of
/// the table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeasuredDevice {
    /// Strix Halo (gfx1151), through HSA or behind RADV.
    StrixHalo,
    /// The RTX PRO 6000 Blackwell Server (compute capability 12.0).
    Sm120,
}

impl MeasuredDevice {
    /// What was measured on this device.
    pub const fn figures(self) -> PerfCalibration {
        match self {
            Self::StrixHalo => PerfCalibration::STRIX_HALO,
            Self::Sm120 => PerfCalibration::SM_120,
        }
    }
}

/// Whether a device's performance figures were measured on it or are the portable prior. It is part of
/// [`GemmLaunch`], so of every plan made for the device: two devices that differ only in this plan differently
/// and the plan names the one it used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Calibrated {
    /// Measured on this device.
    Measured(MeasuredDevice),
    /// Nobody measured this device: it plans with [`PerfCalibration::PORTABLE`].
    Portable,
}

impl Calibrated {
    /// Strix Halo's measured entry.
    pub const STRIX_HALO: Self = Self::Measured(MeasuredDevice::StrixHalo);

    /// The RTX PRO 6000 Blackwell's measured entry.
    pub const SM_120: Self = Self::Measured(MeasuredDevice::Sm120);

    /// The figures the planner reads.
    pub const fn figures(self) -> PerfCalibration {
        match self {
            Self::Measured(device) => device.figures(),
            Self::Portable => PerfCalibration::PORTABLE,
        }
    }

    /// The entry of an AMD device the HSA runtime drives, by its gfx code.
    pub fn of_hsa_arch(arch: crate::AmdArch) -> Self {
        if is_strix_halo(arch.mcpu()) {
            Self::STRIX_HALO
        } else {
            Self::Portable
        }
    }

    /// The entry of the Vulkan device a `vendor_id` and device name identify. Keyed by identity, not by the
    /// runtime family: another RADV part, or an Intel or NVIDIA device behind Vulkan, is [`Self::Portable`].
    pub fn of_vulkan_device(vendor_id: u32, device_name: &str) -> Self {
        if vendor_id == VENDOR_AMD && is_strix_halo(device_name) {
            Self::STRIX_HALO
        } else {
            Self::Portable
        }
    }

    /// The entry of the CUDA device of this compute capability.
    pub fn of_cuda_compute_capability(major: u32, minor: u32) -> Self {
        if (major, minor) == (12, 0) {
            Self::SM_120
        } else {
            Self::Portable
        }
    }
}

/// The launch figures of the contraction schedules on one device: the narrow and wide lane grids of the
/// register-tiled one and the residency its occupancy model saturates a compute unit at, and the measured or
/// portable [`PerfCalibration`]. Resolved once from the device's identity; the planner reads the fields and holds
/// no tile or run literal of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmLaunch {
    /// The narrow register-tiled lane-grid edge.
    pub register_lane_edge: u32,
    /// The wide register-tiled lane-grid edge.
    pub register_lane_edge_wide: u32,
    /// Resident lanes that saturate one compute unit, in the register-tiled schedule's occupancy model.
    pub saturating_lanes: u32,
    /// The performance figures, and whether the device measured them.
    pub calibrated: Calibrated,
}

impl GemmLaunch {
    /// The launch of a device whose performance figures are `calibrated`. No device has a confirmed defect that
    /// narrows the register-tiled lane grids (the RADV wave64 LDS defect of update 0169 does not reproduce, see
    /// `REGISTER_LANE_EDGE_WIDE`'s provenance).
    pub const fn new(calibrated: Calibrated) -> Self {
        Self {
            register_lane_edge: REGISTER_LANE_EDGE,
            register_lane_edge_wide: REGISTER_LANE_EDGE_WIDE,
            saturating_lanes: SATURATING_LANES,
            calibrated,
        }
    }
}

/// PCI vendor id of AMD, as Vulkan reports it.
pub const VENDOR_AMD: u32 = 0x1002;

/// Whether a device identity string names the Strix Halo part (gfx1151): an HSA ISA name
/// (`"amdgcn-amd-amdhsa--gfx1151"`) or a Mesa RADV device name (`"AMD Radeon 8060S Graphics (RADV STRIX_HALO)"`).
pub(crate) fn is_strix_halo(device_name: &str) -> bool {
    let name = device_name.to_ascii_lowercase();
    name.contains("gfx1151") || name.contains("strix_halo")
}

/// What was measured on one Vulkan device, for the planner's chunking and routing around observed faults.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Calibration {
    /// The per-dispatch work budgets that keep the display watchdog from killing a dispatch; `None` on a
    /// device measured to need none.
    pub watchdog_budget: Option<WatchdogBudget>,
    /// The defects observed on this device.
    pub known_miscompiles: KnownMiscompiles,
    /// A single-buffer ceiling below the device's advertised limit, where a larger buffer wedged the ring.
    pub safe_max_buffer_bytes: Option<u64>,
}

impl Calibration {
    /// Strix Halo behind RADV, as cards 095, 163, 258 and 159 measured it and the 2026-09-25 lm_head wedge
    /// bounded the buffer size.
    const STRIX_HALO_MISCOMPILES: KnownMiscompiles = KnownMiscompiles {
        tiled_gemm_max_workgroups: Some(1 << 15),
    };

    /// Strix Halo (gfx1151) behind RADV, the one device these figures were measured on.
    pub const STRIX_HALO_RADV: Self = Self {
        watchdog_budget: Some(WatchdogBudget {
            decode_gemv_out_elems: DECODE_GEMV_CHUNK_TRIGGER,
            serial_dequant_work: 400_000_000,
            scatter_update_work: 1_000_000,
            indexed_dequant_work: 800_000_000,
            packed_tiled_work: PACKED_TILED_CHUNK_WORK,
            // Measured on Strix Halo behind RADV only (POOT-1313); another RADV or display-attached device has
            // no calibration, so no `watchdog_budget` and an unsplit 8192-token prefill.
            attention_work: ATTENTION_CHUNK_WORK,
        }),
        known_miscompiles: Self::STRIX_HALO_MISCOMPILES,
        safe_max_buffer_bytes: Some(AMD_RADV_SAFE_MAX_BUFFER_BYTES),
    };

    /// The calibration of the Vulkan device a `vendor_id` and device name identify, or `Unknown` for a
    /// device nobody measured.
    pub fn of_vulkan_device(vendor_id: u32, device_name: &str) -> Queried<Self> {
        if vendor_id == VENDOR_AMD && is_strix_halo(device_name) {
            Queried::Known(Self::STRIX_HALO_RADV)
        } else {
            Queried::Unknown
        }
    }

    /// The calibration a device has when nobody measured it: no budget, no known defect, no buffer ceiling.
    /// It claims nothing; a consumer that needs a figure sees it absent.
    pub const UNMEASURED: Self = Self {
        watchdog_budget: None,
        known_miscompiles: KnownMiscompiles::NONE,
        safe_max_buffer_bytes: None,
    };
}

/// The output-column count above which a decode GEMV is slower than the batched matmul on an NVIDIA part:
/// the 2026-07-03 benchmark measured the GEMV's serial lane-0 LDS reduction and stride-N weight reads at about
/// 2x slower on Ampere (compute capability 8.x) for projections in the thousands of columns. `Unknown` for a
/// generation nobody measured.
pub fn nvidia_decode_gemv_max_out_cols(compute_capability_major: u32) -> Queried<u32> {
    if compute_capability_major == 8 {
        Queried::Known(256)
    } else {
        Queried::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the part that was measured gets its figures: the Strix Halo RADV device is calibrated and another
    /// AMD Vulkan device is not. Mutation: key on the vendor id alone; the discrete card reads `Known`.
    #[test]
    fn calibration_is_keyed_by_device_identity_not_vendor() {
        let strix = Calibration::of_vulkan_device(VENDOR_AMD, "AMD Radeon 8060S (RADV STRIX_HALO)");
        assert!(matches!(
            strix,
            Queried::Known(c) if c.watchdog_budget.is_some()
        ));
        assert_eq!(
            Calibration::of_vulkan_device(VENDOR_AMD, "AMD Radeon RX 7900 XTX (RADV NAVI31)"),
            Queried::Unknown
        );
        assert_eq!(
            Calibration::of_vulkan_device(0x10de, "NVIDIA GeForce RTX 4090"),
            Queried::Unknown
        );
    }

    /// Where a launch figure came from: what it is called in the source, its value, and the evidence for it. A
    /// literal that shapes a launch has an entry in a table of these, so a number is never free-floating.
    struct Provenance {
        /// The constant's name in the source.
        name: &'static str,
        /// Its value.
        value: u64,
        /// The measurement, card or structural reason it rests on; never empty.
        source: &'static str,
    }

    /// The provenance of each figure a [`GemmLaunch`] and a [`PerfCalibration`] entry carry.
    const LAUNCH_PROVENANCE: &[Provenance] = &[
        Provenance {
            name: "REGISTER_LANE_EDGE",
            value: 8,
            source: "card 542b: the 8x8 lane grid, one wave64",
        },
        Provenance {
            name: "REGISTER_LANE_EDGE_WIDE",
            value: 16,
            source: "POOT-1193: the 16x16 lane grid, 256 lanes. POOT-1403: runs on RADV too; the wave64 LDS defect (update 0169: a 256-lane LDS-tiled GEMM read stale upper-wave LDS) does not reproduce on Strix Halo RADV Mesa 26.1.3, on the register-tiled body at 16x16 (16.5 KiB LDS; poot-kernelgen the_register_tiled_body_runs_256_lane_workgroups_on_radv, mutation-proven) nor on the deleted one-output-per-lane Tiled body at 16x16 (bit-exact); master already ran the packed register-tiled 16x16 there",
        },
        Provenance {
            name: "SATURATING_LANES",
            value: 256,
            source: "structural (POOT-1403): eight wave32s per compute unit, four per RDNA SIMD32 or two per NVIDIA SM partition; the register-tiled cost model's residency, not a per-device measurement",
        },
        Provenance {
            name: "PORTABLE_PACKED_GEMV_RUN",
            value: 8,
            source: "card 653: the 8-element run every class launched before POOT-1398; the run of a decode that addresses each element on its own (gemv_run's Elementwise arm)",
        },
        Provenance {
            name: "STRIX_HALO_PACKED_GEMV_RUN",
            value: 32,
            source: "measured: gfx1151 ROCm and RADV at 17408 x 5120, run 32 vs 8: Q4_K 120 -> 195 GB/s, Q5_K 119 -> 190 (ROCm), Q4_K 143 -> 171-184 (Vulkan); POOT-1390 section 5 and the POOT-1398 receipt",
        },
        Provenance {
            name: "SM_120_PACKED_GEMV_RUN",
            value: 16,
            source: "measured on one part only, RTX PRO 6000 Blackwell Server (sm_120), at 17408 x 5120 with the vector activation read and 32-bit index math, run 8 / 16 / 32 GB/s: Q4_K 721/980/665, Q5_K 747/955/741, Q6_K 861/1254/1253 (factor run 16), Q3_K 567/710/711, Q2_K 539/695/695, Q4_0 1026/1094/751, Q8_0 1383/1402/1260, IQ4_XS 894/856/601, IQ4_NL 998/911/641; POOT-1405 receipts. Other NVIDIA parts unmeasured",
        },
    ];

    /// The provenance entry named `name`, whose value is `value` and whose source is stated.
    fn assert_provenance(name: &str, value: u32) {
        let entry = LAUNCH_PROVENANCE
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("{name} has no provenance entry"));
        assert_eq!(entry.value, u64::from(value), "{name}");
        assert!(!entry.source.is_empty(), "{name} states no source");
    }

    /// Every provenance entry is claimed by a figure below, so a row for a deleted figure does not linger.
    fn claimed(names: &[&str]) {
        let table: Vec<&str> = LAUNCH_PROVENANCE.iter().map(|entry| entry.name).collect();
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        let mut table_sorted = table.clone();
        table_sorted.sort_unstable();
        assert_eq!(table_sorted, sorted, "provenance rows and figures differ");
    }

    /// Every figure a `GemmLaunch` and every `PerfCalibration` entry carries has a provenance row whose value is
    /// the figure's and whose source is stated. Both structs are destructured without `..`, so a new field fails
    /// to compile here until it is listed. Each row pins the value as a literal, so a figure changed without its row fails. Mutation: change
    /// `STRIX_HALO_PACKED_GEMV_RUN` without its row; the value check fails.
    #[test]
    fn every_launch_figure_has_provenance() {
        let GemmLaunch {
            register_lane_edge,
            register_lane_edge_wide,
            saturating_lanes,
            calibrated: _,
        } = GemmLaunch::new(Calibrated::Portable);
        assert_provenance("REGISTER_LANE_EDGE", register_lane_edge);
        assert_provenance("REGISTER_LANE_EDGE_WIDE", register_lane_edge_wide);
        assert_provenance("SATURATING_LANES", saturating_lanes);
        for (entry, row) in [
            (PerfCalibration::PORTABLE, "PORTABLE_PACKED_GEMV_RUN"),
            (PerfCalibration::STRIX_HALO, "STRIX_HALO_PACKED_GEMV_RUN"),
            (PerfCalibration::SM_120, "SM_120_PACKED_GEMV_RUN"),
        ] {
            let PerfCalibration {
                packed_gemv_max_run,
            } = entry;
            assert_provenance(row, packed_gemv_max_run);
        }
        claimed(&[
            "REGISTER_LANE_EDGE",
            "REGISTER_LANE_EDGE_WIDE",
            "SATURATING_LANES",
            "PORTABLE_PACKED_GEMV_RUN",
            "STRIX_HALO_PACKED_GEMV_RUN",
            "SM_120_PACKED_GEMV_RUN",
        ]);
    }

    /// A device's figures follow its identity, not its runtime family: Strix Halo is measured through HSA and
    /// through Vulkan, another AMD, an Intel or an NVIDIA Vulkan device is portable, and CUDA is measured on
    /// sm_120 only. Mutation: key the Vulkan arm on the vendor alone; the discrete AMD card reads `Measured`.
    #[test]
    fn performance_figures_follow_device_identity() {
        assert_eq!(
            Calibrated::of_hsa_arch(crate::AmdArch::gfx1151()),
            Calibrated::STRIX_HALO
        );
        assert_eq!(
            Calibrated::of_vulkan_device(VENDOR_AMD, "AMD Radeon 8060S (RADV STRIX_HALO)"),
            Calibrated::STRIX_HALO
        );
        for (vendor, name) in [
            (VENDOR_AMD, "AMD Radeon RX 7900 XTX (RADV NAVI31)"),
            (0x8086, "Intel(R) Arc(tm) A770 Graphics (DG2)"),
            (0x10de, "NVIDIA GeForce RTX 4090"),
        ] {
            assert_eq!(
                Calibrated::of_vulkan_device(vendor, name),
                Calibrated::Portable,
                "{name}"
            );
        }
        assert_eq!(
            Calibrated::of_cuda_compute_capability(12, 0),
            Calibrated::SM_120
        );
        for (major, minor) in [(8, 0), (9, 0), (10, 0), (12, 1)] {
            assert_eq!(
                Calibrated::of_cuda_compute_capability(major, minor),
                Calibrated::Portable,
                "sm_{major}{minor}"
            );
        }
    }

    /// The packed Gemv run an unmeasured device plans with is the portable 8, whichever runtime drives it, and a
    /// measured one carries its own: the launch a plan is made with names which. Mutation: make `figures` return
    /// the measured run for `Portable`.
    #[test]
    fn a_launch_carries_the_figures_of_its_calibration() {
        let portable = GemmLaunch::new(Calibrated::Portable);
        let strix = GemmLaunch::new(Calibrated::STRIX_HALO);
        let sm_120 = GemmLaunch::new(Calibrated::SM_120);
        assert_eq!(portable.calibrated.figures().packed_gemv_max_run, 8);
        assert_eq!(strix.calibrated.figures().packed_gemv_max_run, 32);
        assert_eq!(sm_120.calibrated.figures().packed_gemv_max_run, 16);
        assert_ne!(
            portable, strix,
            "the launch must record which calibration it used"
        );
    }

    /// The NVIDIA GEMV ceiling is a measured Ampere figure. Mutation: return `Known` for every major.
    #[test]
    fn the_gemv_ceiling_is_known_only_for_the_measured_generation() {
        assert_eq!(nvidia_decode_gemv_max_out_cols(8), Queried::Known(256));
        assert_eq!(nvidia_decode_gemv_max_out_cols(9), Queried::Unknown);
        assert_eq!(nvidia_decode_gemv_max_out_cols(7), Queried::Unknown);
    }
}

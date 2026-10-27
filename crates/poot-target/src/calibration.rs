//! Calibration: the figures no driver API reports, measured once on a named device and keyed by that device's
//! identity. A device nobody measured is [`Queried::Unknown`], never given a neighbour's figures: the
//! display-watchdog budgets and the tiled-GEMM ceiling below were observed on one part (Strix Halo behind
//! RADV), and copying them to every AMD GPU chunked work a discrete card runs unsplit and fenced it with a
//! ceiling nobody observed on it.

use crate::{
    AMD_RADV_SAFE_MAX_BUFFER_BYTES, ATTENTION_CHUNK_WORK, DECODE_GEMV_CHUNK_TRIGGER,
    KnownMiscompiles, PACKED_TILED_CHUNK_WORK, Queried, WatchdogBudget,
};

/// The kind of device a launch figure is calibrated for: the runtime family that drives it, since the same
/// silicon runs different compilers and schedulers behind each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaunchClass {
    /// A device driven through Vulkan (wgpu or raw), whatever it is behind the driver.
    Vulkan,
    /// An AMD device driven through HSA (the ROCm runtime).
    Hsa,
    /// An NVIDIA device driven through CUDA (the PTX runtime). Its launch figures are unmeasured and take the
    /// portable ones until a pod sweep measures them.
    Cuda,
}

/// The lane-grid edge of the narrow register-tiled schedule: 64 lanes, one wave64, the widest workgroup every
/// device is known to run.
pub const REGISTER_LANE_EDGE: u32 = 8;

/// The lane-grid edge of the wide register-tiled schedule: 256 lanes, offered to the cost model on a device whose
/// workgroup and LDS hold it.
pub const REGISTER_LANE_EDGE_WIDE: u32 = 16;

/// The longest contiguous run of K elements one lane of the packed decode Gemv decodes per trip on a class
/// nobody measured: the run every class launched before POOT-1398.
pub const PORTABLE_PACKED_GEMV_RUN: u32 = 8;

/// The packed decode Gemv's longest lane run on the HSA and Vulkan classes, measured on gfx1151 (POOT-1390):
/// a 32-element run (one Q4_K/Q5_K sub-block) amortises the block addressing and the factor decode over
/// four times the weights of the portable run and keeps the body within the register file on both
/// compilers.
pub const MEASURED_PACKED_GEMV_RUN: u32 = 32;

/// The packed decode Gemv's longest lane run on the CUDA class: a calibration measured on one part, the RTX PRO
/// 6000 Blackwell Server (sm_120), not a rule derived for every NVIDIA GPU (POOT-1405). With the activation read
/// in 16-byte groups and 32-bit index math, a 16-element run amortises a K-quant's block decode over twice the
/// weights of the 8-element one, while a 32-element run spreads each warp-wide activation load over 16 cache lines
/// and loses (Q4_K 980 at run 16 against 665 at 32 and 721 at 8 GB/s). Other NVIDIA parts are unmeasured and take
/// it as the class figure until a pod sweep says otherwise.
pub const CUDA_PACKED_GEMV_RUN: u32 = 16;

/// The resident lanes per compute unit the register-tiled schedule's occupancy model treats as saturating it:
/// eight 32-lane waves, four per SIMD of an RDNA compute unit (two SIMD32s) or two per partition of an NVIDIA SM
/// (four), the residency at which a unit hides its LDS and memory latency behind other waves' FMAs. Structural,
/// not measured per device.
pub const SATURATING_LANES: u32 = 256;

/// The launch figures of the contraction schedules on one device: the narrow and wide lane grids of the
/// register-tiled one and the residency its occupancy model saturates a compute unit at, and the longest lane run
/// of the packed decode Gemv. Resolved once from the device class and its [`KnownMiscompiles`]; the planner reads
/// the fields and holds no tile or run literal of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmLaunch {
    /// The narrow register-tiled lane-grid edge.
    pub register_lane_edge: u32,
    /// The wide register-tiled lane-grid edge.
    pub register_lane_edge_wide: u32,
    /// The longest run of contiguous K elements a packed decode Gemv lane decodes per trip: the register and
    /// issue budget of the class's compiler. The planner runs the format's factor run up to this cap.
    pub packed_gemv_max_run: u32,
    /// Resident lanes that saturate one compute unit, in the register-tiled schedule's occupancy model.
    pub saturating_lanes: u32,
}

impl GemmLaunch {
    /// The launch of a device of `class`. No device has a confirmed defect that narrows the register-tiled lane
    /// grids (the RADV wave64 LDS defect of update 0169 does not reproduce, see `REGISTER_LANE_EDGE_WIDE`'s
    /// provenance); the packed decode Gemv's lane run is the class's.
    pub const fn resolve(class: LaunchClass) -> Self {
        Self {
            register_lane_edge: REGISTER_LANE_EDGE,
            register_lane_edge_wide: REGISTER_LANE_EDGE_WIDE,
            packed_gemv_max_run: match class {
                LaunchClass::Vulkan | LaunchClass::Hsa => MEASURED_PACKED_GEMV_RUN,
                LaunchClass::Cuda => CUDA_PACKED_GEMV_RUN,
            },
            saturating_lanes: SATURATING_LANES,
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

    pub(crate) const STRIX_HALO_RADV: Self = Self {
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

    /// The provenance of each figure [`GemmLaunch::resolve`] resolves from.
    const LAUNCH_PROVENANCE: &[Provenance] = &[
        Provenance {
            name: "REGISTER_LANE_EDGE",
            value: REGISTER_LANE_EDGE as u64,
            source: "card 542b: the 8x8 lane grid, one wave64",
        },
        Provenance {
            name: "REGISTER_LANE_EDGE_WIDE",
            value: REGISTER_LANE_EDGE_WIDE as u64,
            source: "POOT-1193: the 16x16 lane grid, 256 lanes. POOT-1403: runs on RADV too; the wave64 LDS defect (update 0169: a 256-lane LDS-tiled GEMM read stale upper-wave LDS) does not reproduce on Strix Halo RADV Mesa 26.1.3, on the register-tiled body at 16x16 (16.5 KiB LDS; poot-kernelgen the_register_tiled_body_runs_256_lane_workgroups_on_radv, mutation-proven) nor on the deleted one-output-per-lane Tiled body at 16x16 (bit-exact); master already ran the packed register-tiled 16x16 there",
        },
        Provenance {
            name: "SATURATING_LANES",
            value: SATURATING_LANES as u64,
            source: "structural (POOT-1403): eight wave32s per compute unit, four per RDNA SIMD32 or two per NVIDIA SM partition; the register-tiled cost model's residency, not a per-device measurement",
        },
        Provenance {
            name: "PORTABLE_PACKED_GEMV_RUN",
            value: PORTABLE_PACKED_GEMV_RUN as u64,
            source: "card 653: the 8-element run every class launched before POOT-1398; the run of a decode that addresses each element on its own (gemv_run's Elementwise arm)",
        },
        Provenance {
            name: "MEASURED_PACKED_GEMV_RUN",
            value: MEASURED_PACKED_GEMV_RUN as u64,
            source: "measured: gfx1151 ROCm and RADV at 17408 x 5120, run 32 vs 8: Q4_K 120 -> 195 GB/s, Q5_K 119 -> 190 (ROCm), Q4_K 143 -> 171-184 (Vulkan); backstage designs/quant-matmul-1390.md section 5 and the POOT-1398 receipt",
        },
        Provenance {
            name: "CUDA_PACKED_GEMV_RUN",
            value: CUDA_PACKED_GEMV_RUN as u64,
            source: "measured on one part only, RTX PRO 6000 Blackwell Server (sm_120), at 17408 x 5120 with the vector activation read and 32-bit index math, run 8 / 16 / 32 GB/s: Q4_K 721/980/665, Q5_K 747/955/741, Q6_K 861/1254/1253 (factor run 16), Q3_K 567/710/711, Q2_K 539/695/695, Q4_0 1026/1094/751, Q8_0 1383/1402/1260, IQ4_XS 894/856/601, IQ4_NL 998/911/641; POOT-1405 receipts. Other NVIDIA parts unmeasured",
        },
    ];

    /// Every launch figure `GemmLaunch` resolves from has a provenance entry whose value is the constant's and
    /// whose source is stated. Mutation: change a constant without its entry; the value check fails.
    #[test]
    fn every_gemm_launch_figure_has_provenance() {
        let figures = [
            ("REGISTER_LANE_EDGE", REGISTER_LANE_EDGE),
            ("REGISTER_LANE_EDGE_WIDE", REGISTER_LANE_EDGE_WIDE),
            ("PORTABLE_PACKED_GEMV_RUN", PORTABLE_PACKED_GEMV_RUN),
            ("MEASURED_PACKED_GEMV_RUN", MEASURED_PACKED_GEMV_RUN),
            ("CUDA_PACKED_GEMV_RUN", CUDA_PACKED_GEMV_RUN),
            ("SATURATING_LANES", SATURATING_LANES),
        ];
        assert_eq!(LAUNCH_PROVENANCE.len(), figures.len());
        for (name, value) in figures {
            let entry = LAUNCH_PROVENANCE
                .iter()
                .find(|entry| entry.name == name)
                .unwrap_or_else(|| panic!("{name} has no provenance entry"));
            assert_eq!(entry.value, u64::from(value), "{name}");
            assert!(!entry.source.is_empty(), "{name} states no source");
        }
    }

    /// The NVIDIA GEMV ceiling is a measured Ampere figure. Mutation: return `Known` for every major.
    #[test]
    fn the_gemv_ceiling_is_known_only_for_the_measured_generation() {
        assert_eq!(nvidia_decode_gemv_max_out_cols(8), Queried::Known(256));
        assert_eq!(nvidia_decode_gemv_max_out_cols(9), Queried::Unknown);
        assert_eq!(nvidia_decode_gemv_max_out_cols(7), Queried::Unknown);
    }
}

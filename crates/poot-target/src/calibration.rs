//! Calibration: the figures no driver API reports, measured once on a named device and keyed by that device's
//! identity. A device nobody measured is [`Queried::Unknown`], never given a neighbour's figures: the
//! display-watchdog budgets and the tiled-GEMM ceiling below were observed on one part (Strix Halo behind
//! RADV), and copying them to every AMD GPU chunked work a discrete card runs unsplit and fenced it with a
//! ceiling nobody observed on it.

use crate::{
    AMD_RADV_SAFE_MAX_BUFFER_BYTES, DECODE_GEMV_CHUNK_TRIGGER, KnownMiscompiles,
    PACKED_TILED_CHUNK_WORK, Queried, WatchdogBudget,
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

/// The tile edge of the LDS-staged M>1 contraction (an indexed `Tiled` tile, or the generated dense one): a
/// workgroup of `edge * edge` lanes. 8 is 64 lanes, one wave64, the edge every Vulkan device is known to run;
/// a Vulkan device nobody measured keeps it, and only a measured part gets a wider one.
pub const PORTABLE_TILE_EDGE: u32 = 8;

/// The tile edge the HSA runtime launches: 16x16 is a 256-lane workgroup, within the 1024 every AMDGPU part
/// allows, measured on gfx1151 (see the provenance table in the tests). The RADV defect lives in the Vulkan
/// driver stack ([`KnownMiscompiles::lds_stale_above_wave64`]).
pub const HSA_TILE_EDGE: u32 = 16;

/// The lane-grid edge of the narrow register-tiled schedule: 64 lanes, one wave64, the grid a row count below
/// the wide one's threshold runs.
pub const REGISTER_LANE_EDGE: u32 = 8;

/// The lane-grid edge of the wide register-tiled schedule: 256 lanes, chosen once there are 128 rows and the
/// device fits its two LDS tiles.
pub const REGISTER_LANE_EDGE_WIDE: u32 = 16;

/// The launch edges of the M>1 contraction schedules on one device: the tile edge of the LDS-staged `Tiled`
/// schedule and the narrow and wide lane grids of the register-tiled one. Resolved once from the device
/// class and its [`KnownMiscompiles`]; the planner reads the fields and holds no tile literal of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmLaunch {
    /// The edge of the square tile one `Tiled` workgroup computes, in lanes per side.
    pub tile_edge: u32,
    /// The narrow register-tiled lane-grid edge.
    pub register_lane_edge: u32,
    /// The wide register-tiled lane-grid edge, used once the rows fill its tile.
    pub register_lane_edge_wide: u32,
}

impl GemmLaunch {
    /// The launch of a device of `class` whose confirmed defects are `known`. A device with
    /// [`KnownMiscompiles::lds_stale_above_wave64`] runs the portable 64-lane tile, whatever its class would
    /// otherwise launch.
    pub const fn resolve(class: LaunchClass, known: &KnownMiscompiles) -> Self {
        let class_edge = match class {
            LaunchClass::Vulkan => PORTABLE_TILE_EDGE,
            LaunchClass::Hsa => HSA_TILE_EDGE,
            // Unmeasured on CUDA: the portable edge until a pod sweep says otherwise.
            LaunchClass::Cuda => PORTABLE_TILE_EDGE,
        };
        Self {
            tile_edge: if known.lds_stale_above_wave64 {
                PORTABLE_TILE_EDGE
            } else {
                class_edge
            },
            register_lane_edge: REGISTER_LANE_EDGE,
            register_lane_edge_wide: REGISTER_LANE_EDGE_WIDE,
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
        lds_stale_above_wave64: true,
    };

    pub(crate) const STRIX_HALO_RADV: Self = Self {
        watchdog_budget: Some(WatchdogBudget {
            decode_gemv_out_elems: DECODE_GEMV_CHUNK_TRIGGER,
            serial_dequant_work: 400_000_000,
            scatter_update_work: 1_000_000,
            indexed_dequant_work: 800_000_000,
            packed_tiled_work: PACKED_TILED_CHUNK_WORK,
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

    /// The provenance of each figure [`GemmLaunch`] resolves from.
    const LAUNCH_PROVENANCE: &[Provenance] = &[
        Provenance {
            name: "PORTABLE_TILE_EDGE",
            value: PORTABLE_TILE_EDGE as u64,
            source: "card 095: a 256-lane workgroup miscomputes on wgpu/RADV (Strix Halo); 8x8 = one wave64 is the edge every Vulkan device is known to run. Also the CUDA edge: unmeasured on CUDA, sweep owed on a pod",
        },
        Provenance {
            name: "HSA_TILE_EDGE",
            value: HSA_TILE_EDGE as u64,
            source: "measured: gfx1151 ROCm, tile 16 bit-identical to 8 and 1.3-3.6x faster on the routed contraction, backstage updates/1343-poot-1233-rocm-gemm-tile-edge-sweep-receipt.md",
        },
        Provenance {
            name: "REGISTER_LANE_EDGE",
            value: REGISTER_LANE_EDGE as u64,
            source: "card 542b: the 8x8 lane grid, one wave64",
        },
        Provenance {
            name: "REGISTER_LANE_EDGE_WIDE",
            value: REGISTER_LANE_EDGE_WIDE as u64,
            source: "POOT-1193: the 16x16 lane grid for 128-row tiles",
        },
    ];

    /// The tile edge is the class's calibrated one unless the device has the LDS-visibility defect: the HSA and
    /// class launches 16; a Vulkan device, the unmeasured CUDA class and any defective one 8. Mutation: drop the
    /// `lds_stale_above_wave64` gate in [`GemmLaunch::resolve`]; the RADV row reads 16.
    #[test]
    fn the_tile_edge_is_the_class_figure_gated_by_the_lds_defect() {
        let defect = KnownMiscompiles {
            lds_stale_above_wave64: true,
            ..KnownMiscompiles::NONE
        };
        let edge = |class, known: &KnownMiscompiles| GemmLaunch::resolve(class, known).tile_edge;
        assert_eq!(
            edge(LaunchClass::Hsa, &KnownMiscompiles::NONE),
            HSA_TILE_EDGE
        );
        assert_eq!(
            edge(LaunchClass::Cuda, &KnownMiscompiles::NONE),
            PORTABLE_TILE_EDGE
        );
        assert_eq!(
            edge(LaunchClass::Vulkan, &KnownMiscompiles::NONE),
            PORTABLE_TILE_EDGE
        );
        assert_eq!(edge(LaunchClass::Hsa, &defect), PORTABLE_TILE_EDGE);
        assert_eq!(edge(LaunchClass::Cuda, &defect), PORTABLE_TILE_EDGE);
    }

    /// Every launch figure `GemmLaunch` resolves from has a provenance entry whose value is the constant's and
    /// whose source is stated. Mutation: change a constant without its entry; the value check fails.
    #[test]
    fn every_gemm_launch_figure_has_provenance() {
        let figures = [
            ("PORTABLE_TILE_EDGE", PORTABLE_TILE_EDGE),
            ("HSA_TILE_EDGE", HSA_TILE_EDGE),
            ("REGISTER_LANE_EDGE", REGISTER_LANE_EDGE),
            ("REGISTER_LANE_EDGE_WIDE", REGISTER_LANE_EDGE_WIDE),
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

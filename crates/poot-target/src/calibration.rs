//! Calibration: the figures no driver API reports, measured once on a named device and keyed by that device's
//! identity. A device nobody measured is [`Queried::Unknown`], never given a neighbour's figures: the
//! display-watchdog budgets and the tiled-GEMM ceiling below were observed on one part (Strix Halo behind
//! RADV), and copying them to every AMD GPU chunked work a discrete card runs unsplit and fenced it with a
//! ceiling nobody observed on it.

use crate::{
    AMD_RADV_SAFE_MAX_BUFFER_BYTES, DECODE_GEMV_CHUNK_TRIGGER, KnownMiscompiles,
    PACKED_TILED_CHUNK_WORK, Queried, WatchdogBudget,
};

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
    pub(crate) const STRIX_HALO_RADV: Self = Self {
        watchdog_budget: Some(WatchdogBudget {
            decode_gemv_out_elems: DECODE_GEMV_CHUNK_TRIGGER,
            serial_dequant_work: 400_000_000,
            scatter_update_work: 1_000_000,
            indexed_dequant_work: 800_000_000,
            packed_tiled_work: PACKED_TILED_CHUNK_WORK,
        }),
        known_miscompiles: KnownMiscompiles {
            tiled_gemm_max_workgroups: Some(1 << 15),
        },
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
        known_miscompiles: KnownMiscompiles {
            tiled_gemm_max_workgroups: None,
        },
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

    /// The NVIDIA GEMV ceiling is a measured Ampere figure. Mutation: return `Known` for every major.
    #[test]
    fn the_gemv_ceiling_is_known_only_for_the_measured_generation() {
        assert_eq!(nvidia_decode_gemv_max_out_cols(8), Queried::Known(256));
        assert_eq!(nvidia_decode_gemv_max_out_cols(9), Queried::Unknown);
        assert_eq!(nvidia_decode_gemv_max_out_cols(7), Queried::Unknown);
    }
}

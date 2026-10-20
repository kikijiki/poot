//! The DRAM channel model of POOT-1139 (research note 2026-10-12, section 2): the Strix Halo APU selects a
//! channel by address bits [11:8], a 256 B granule over 16 channels, so a read whose row stride is a multiple
//! of several granules camps on a subset of the channels. [`DramChannels`] is the per-device record of that
//! map, carried by [`DeviceCaps::dram_channels`](crate::DeviceCaps::dram_channels): a device with no
//! measured channel data has none, and no layout rule applies to it. One definition for the planner's stored
//! KV extent (`poot-graph-plan`'s `legalize`) and its camping check (`storage_analysis`), and for the device
//! crates' strided-read probe (`poot-executor-parity`).

/// DRAM channels the address map interleaves over.
pub const CHANNELS: usize = 16;
/// The interleave granule: consecutive 256 B of address space sit on one channel.
pub const GRANULE_BYTES: usize = 256;

/// How many times fewer channels a read at `row_stride_bytes` per row reaches than the full set:
/// `gcd(row_stride_bytes / 256, 16)`, so `1` is a stride that reaches every channel and `16` one that camps
/// on a single channel. A stride that is not a whole number of granules drifts across them and counts as `1`.
pub const fn alias_factor(row_stride_bytes: usize) -> usize {
    const PERIOD: usize = CHANNELS * GRANULE_BYTES;
    let (mut a, mut b) = (row_stride_bytes, PERIOD);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let factor = a / GRANULE_BYTES;
    if factor == 0 { 1 } else { factor }
}

/// The largest alias factor a served stride may have: a row stride above it leaves more than four of the
/// sixteen channels idle, which POOT-1139's probe measured at 2.6-4.6x on both backends.
pub const MAX_ALIAS: usize = 4;

/// A device's DRAM channel map: what the planner needs to tell a camping layout from one that reaches enough
/// channels. Only a device whose map was measured has one ([`DramChannels::STRIX_HALO`]); a layout rule
/// that reads it never applies to another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DramChannels {
    max_alias: usize,
}

impl DramChannels {
    /// Strix Halo (gfx1151): sixteen channels over 256 B granules, a stride camping above [`MAX_ALIAS`].
    pub const STRIX_HALO: Self = Self {
        max_alias: MAX_ALIAS,
    };

    /// The channel map of the part a device identity string names: an HSA ISA name
    /// (`"amdgcn-amd-amdhsa--gfx1151"`) or a Mesa RADV device name (`"AMD Radeon 8060S Graphics (RADV STRIX_HALO)"`).
    /// `Some` only for a part whose map was measured; any other AMD device has none.
    pub fn of_device_name(name: &str) -> Option<Self> {
        crate::calibration::is_strix_halo(name).then_some(Self::STRIX_HALO)
    }

    /// Whether a read at `row_stride_bytes` per row camps on too few channels.
    pub const fn camps(self, row_stride_bytes: usize) -> bool {
        alias_factor(row_stride_bytes) > self.max_alias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alias factor is `gcd(stride / 256, 16)` for the strides the probe measures, and a stride off the
    /// granule does not alias. Mutation: use a period of eight channels; the 4096 B row goes red.
    #[test]
    fn alias_factor_is_gcd_of_stride_granules_and_channels() {
        assert_eq!(alias_factor(4096), 16);
        assert_eq!(alias_factor(8192), 16);
        assert_eq!(alias_factor(10240), 8);
        assert_eq!(alias_factor(5120), 4);
        assert_eq!(alias_factor(10752), 2);
        assert_eq!(alias_factor(10496), 1);
        assert_eq!(alias_factor(256), 1);
        assert_eq!(alias_factor(384), 1);
        assert_eq!(alias_factor(4160), 1);
    }

    /// `camps` is the alias factor above [`MAX_ALIAS`]: 2048 B (eight granules) is the smallest camping stride.
    /// Mutation: compare with `>=`; the 1024 B row goes red.
    #[test]
    fn a_stride_camps_above_four_channels_idle() {
        let map = DramChannels::STRIX_HALO;
        assert!(map.camps(4096));
        assert!(map.camps(2048));
        assert!(!map.camps(1024));
        assert!(!map.camps(4160));
        assert!(!map.camps(10240 + 512));
    }

    /// Only a measured part has a channel map: Strix Halo's ISA name does, another gfx code does not.
    /// Mutation: return `Some` for every name; the gfx1100 row goes red.
    #[test]
    fn only_a_measured_amd_part_has_a_channel_map() {
        assert_eq!(
            DramChannels::of_device_name("amdgcn-amd-amdhsa--gfx1151"),
            Some(DramChannels::STRIX_HALO)
        );
        assert_eq!(
            DramChannels::of_device_name("AMD Radeon 8060S Graphics (RADV STRIX_HALO)"),
            Some(DramChannels::STRIX_HALO)
        );
        assert_eq!(
            DramChannels::of_device_name("amdgcn-amd-amdhsa--gfx1100"),
            None
        );
        assert_eq!(
            DramChannels::of_device_name("AMD Radeon RX 7900 XTX (RADV NAVI31)"),
            None
        );
    }
}

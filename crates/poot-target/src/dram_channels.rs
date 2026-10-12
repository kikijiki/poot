//! The DRAM channel model of POOT-1139 (research note 2026-10-12, section 2): the Strix Halo APU selects a
//! channel by address bits [11:8], a 256 B granule over 16 channels, so a read whose row stride is a multiple
//! of several granules camps on a subset of the channels. One definition for the KV layout rule
//! (`poot-models`' `HeadSpan`), the planner's stride audit (`poot-models`' `stride_audit`) and the device
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

/// Whether a read at `row_stride_bytes` per row camps on too few channels ([`MAX_ALIAS`]).
pub const fn camps(row_stride_bytes: usize) -> bool {
    alias_factor(row_stride_bytes) > MAX_ALIAS
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
        assert!(camps(4096));
        assert!(camps(2048));
        assert!(!camps(1024));
        assert!(!camps(4160));
        assert!(!camps(10240 + 512));
    }
}

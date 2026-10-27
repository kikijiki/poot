//! The host's memory on a unified-memory device. An APU's device allocations come out of the host's RAM,
//! so the HSA pool's free figure alone overstates what a process may take: the device is not the only
//! tenant of that memory.

/// What the host keeps for itself when the device is an APU: the operating system, the desktop and the
/// kernel's own buffers. A device plan that leaves the host less than this starves the box, whatever the
/// HSA pool still reports free.
pub(crate) const UNIFIED_HOST_FLOOR_BYTES: u64 = 8 << 30;

/// The memory a unified-memory device may still allocate: the HSA pool's free bytes, bounded by what the
/// host reports available (`MemAvailable`, which counts reclaimable page cache) less
/// [`UNIFIED_HOST_FLOOR_BYTES`].
pub(crate) fn unified_free_bytes(pool_free: u64, host_available: u64) -> u64 {
    pool_free.min(host_available.saturating_sub(UNIFIED_HOST_FLOOR_BYTES))
}

/// `MemAvailable` of a `/proc/meminfo` text, in bytes.
fn parse_mem_available(meminfo: &str) -> Option<u64> {
    let line = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?;
    let kib: u64 = line.trim().strip_suffix("kB")?.trim().parse().ok()?;
    kib.checked_mul(1024)
}

/// The host's available memory now, in bytes; `None` where `/proc/meminfo` cannot be read or parsed.
pub(crate) fn host_available_bytes() -> Option<u64> {
    parse_mem_available(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    #[test]
    fn mem_available_is_read_in_bytes() {
        let text = "MemTotal:       130000000 kB\nMemFree:         1000 kB\nMemAvailable:   117440512 kB\nBuffers: 1 kB\n";
        assert_eq!(parse_mem_available(text), Some(117_440_512 * 1024));
    }

    #[test]
    fn a_meminfo_without_a_readable_mem_available_is_none() {
        assert_eq!(parse_mem_available("MemTotal: 1 kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: lots kB\n"), None);
        assert_eq!(parse_mem_available("MemAvailable: 12\n"), None);
    }

    /// The pool says 112 GiB free while the host has 115 GiB available: the host's floor, not the pool,
    /// bounds the device.
    #[test]
    fn the_host_floor_bounds_a_pool_that_reports_more() {
        assert_eq!(unified_free_bytes(112 * GIB, 115 * GIB), 107 * GIB);
    }

    #[test]
    fn a_pool_below_the_host_bound_stands() {
        assert_eq!(unified_free_bytes(20 * GIB, 115 * GIB), 20 * GIB);
    }

    #[test]
    fn a_host_below_its_floor_leaves_the_device_nothing() {
        assert_eq!(unified_free_bytes(112 * GIB, 6 * GIB), 0);
    }
}

//! The host process's own memory, as the kernel accounts it (`/proc/self/status`): what a checkpoint load
//! costs the host beside the device's memory counters ([`Device::memory`](crate::Device::memory)).
//! `RssAnon` is the heap and the pinned staging, `RssFile` the file-backed pages the process has mapped in
//! (a checkpoint read through a mapping shows up here, one read by `pread` does not), `VmHWM` the process's
//! peak resident set since it started or since [`HostMemory::reset_peak`].

/// One reading of the process's resident memory, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostMemory {
    /// `RssAnon`: resident anonymous pages.
    pub rss_anon: u64,
    /// `RssFile`: resident file-backed pages the process maps.
    pub rss_file: u64,
    /// `VmHWM`: the peak of `RssAnon + RssFile + RssShmem` since the process started or the last
    /// [`Self::reset_peak`].
    pub peak_rss: u64,
}

impl HostMemory {
    /// The process's memory now; `None` where `/proc/self/status` cannot be read or parsed.
    pub fn now() -> Option<Self> {
        Self::parse(&std::fs::read_to_string("/proc/self/status").ok()?)
    }

    /// Restart the peak (`VmHWM`) from the current resident set, so the next reading's peak covers only
    /// what runs after this call (`clear_refs` mode 5). Best effort: a kernel that refuses leaves the
    /// peak the process-lifetime one.
    pub fn reset_peak() {
        let _ = std::fs::write("/proc/self/clear_refs", "5");
    }

    fn parse(status: &str) -> Option<Self> {
        let field = |name: &str| -> Option<u64> {
            let line = status.lines().find_map(|line| line.strip_prefix(name))?;
            let kib: u64 = line
                .trim()
                .strip_prefix(':')?
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse()
                .ok()?;
            kib.checked_mul(1024)
        };
        Some(Self {
            rss_anon: field("RssAnon")?,
            rss_file: field("RssFile")?,
            peak_rss: field("VmHWM")?,
        })
    }
}

/// The host memory a load passed through: when the engine opened, and after its last weight upload.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadMemory {
    /// When the engine was created.
    pub opened: Option<HostMemory>,
    /// After the latest [`Engine`](crate::Engine) `add_part` finished uploading: `peak_rss` is the peak
    /// since the last [`HostMemory::reset_peak`], so a process that resets it before it loads reads the
    /// load's own peak here.
    pub uploaded: Option<HostMemory>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_status_text_is_read_in_bytes() {
        let status = "Name:\tx\nVmHWM:\t  2048 kB\nVmRSS:\t 1000 kB\nRssAnon:\t     512 kB\nRssFile:\t     256 kB\nRssShmem:\t 0 kB\n";
        assert_eq!(
            HostMemory::parse(status),
            Some(HostMemory {
                rss_anon: 512 * 1024,
                rss_file: 256 * 1024,
                peak_rss: 2048 * 1024
            })
        );
    }

    #[test]
    fn a_status_without_a_field_is_none() {
        assert_eq!(HostMemory::parse("RssAnon: 1 kB\n"), None);
    }

    /// The live reading is the process's: anonymous memory it just touched is in `RssAnon`.
    #[test]
    fn the_live_reading_sees_anonymous_memory_the_process_touches() {
        let before = HostMemory::now().expect("/proc/self/status is readable on Linux");
        let held = vec![1u8; 64 << 20];
        let after = HostMemory::now().unwrap();
        assert!(
            after.rss_anon >= before.rss_anon + (48 << 20),
            "{before:?} -> {after:?}"
        );
        assert!(after.peak_rss >= after.rss_anon);
        drop(held);
    }
}

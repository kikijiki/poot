//! A device's memory topology (POOT-1273, ADR-0118 decision 2): which memory it allocates from, how that
//! memory can be shared with another device, and what moving bytes across costs. The runtime states it from
//! what its API reports about the memory it allocates from, never from a backend or product name; the
//! residency planner reads it as data to decide which devices share bytes and which need a staging copy.
//!
//! Share mechanisms and measured link costs are not implemented yet (POOT-1279, POOT-1280): every device
//! declares [`ShareMechanisms::NONE`] and a [`Queried::Unknown`] staging cost until they are.

use crate::Queried;

/// Where a device's memory lives relative to the host's RAM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryDomain {
    /// The device allocates from memory the host and every other `Host` device address too (an APU's RAM, an
    /// NPU on the same package): a byte one `Host` device wrote is in memory the next one reads.
    Host,
    /// The device allocates from memory only it addresses directly (a discrete GPU's VRAM): a byte reaches
    /// another device only through a staging copy.
    Local,
}

/// A way one device's allocation is handed to another without a copy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShareMechanism {
    /// A dma-buf file descriptor.
    DmaBuf,
    /// An allocation made in host memory both devices map.
    HostAllocation,
}

/// The mechanisms a device offers for sharing its memory: those it can hand out (`export`) and those it can
/// accept (`import`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ShareMechanisms {
    /// Mechanisms the device can export an allocation through.
    pub export: &'static [ShareMechanism],
    /// Mechanisms the device can import an allocation through.
    pub import: &'static [ShareMechanism],
}

impl ShareMechanisms {
    /// No mechanism: nothing implements one yet (POOT-1279).
    pub const NONE: Self = Self {
        export: &[],
        import: &[],
    };
}

/// What moving bytes between two memories costs: a fixed latency per transfer and a sustained rate.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LinkCost {
    /// Time to start one transfer, in nanoseconds.
    pub latency_ns: u64,
    /// Sustained transfer rate, in bytes per second.
    pub bytes_per_second: u64,
}

/// A device's memory topology: the domain it allocates from, how it shares memory and what staging costs.
/// No `Default`: a runtime states each field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryTopology {
    /// Where the device's memory lives.
    pub domain: MemoryDomain,
    /// How the device shares its memory without a copy.
    pub share: ShareMechanisms,
    /// What staging a copy through host memory costs; [`Queried::Unknown`] until measured (POOT-1280).
    pub staging: Queried<LinkCost>,
}

impl MemoryTopology {
    /// A device in `domain` that shares no memory and whose staging cost nobody measured: what every runtime
    /// reports until POOT-1279 and POOT-1280 land.
    pub const fn unmeasured(domain: MemoryDomain) -> Self {
        Self {
            domain,
            share: ShareMechanisms::NONE,
            staging: Queried::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_topology_shares_nothing_and_knows_no_staging_cost() {
        let topology = MemoryTopology::unmeasured(MemoryDomain::Local);
        assert_eq!(topology.domain, MemoryDomain::Local);
        assert_eq!(topology.share, ShareMechanisms::NONE);
        assert_eq!(topology.staging, Queried::Unknown);
    }
}

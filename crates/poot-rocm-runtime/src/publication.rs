//! Typed AQL queue-ring publication ordering.

use std::fmt;

/// Memory that backs the AQL packet ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueRingMemory {
    /// System memory with host-coherent packet publication.
    HostCoherent,
    /// The GPU agent's local device memory.
    DeviceMemory,
}

/// Host-to-agent link used for CPU stores to a device-memory packet ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueHostLink {
    /// A coherent CPU-to-agent link with proven device-store ordering.
    Coherent,
    /// A PCIe link that requires an architecture-specific device-store fence.
    Pcie,
}

/// Host architecture relevant to device-store publication ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueHostTarget {
    /// x86_64, where `_mm_sfence` supplies the required ordering. (32-bit x86 cannot build this crate:
    /// the 64-byte AQL packet layout needs 64-bit pointers.)
    X86,
    /// A target without a proven device-store fence in this runtime.
    Other(&'static str),
}

impl QueueHostTarget {
    pub(crate) const fn current() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self::X86
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::Other(std::env::consts::ARCH)
        }
    }
}

impl fmt::Display for QueueHostTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86 => f.write_str("x86_64"),
            Self::Other(target) => f.write_str(target),
        }
    }
}

/// Architecture-specific fence selected for a validated device-memory ring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceStoreFence {
    /// x86_64 `_mm_sfence` for write-combined stores to a PCIe device ring.
    #[cfg(target_arch = "x86_64")]
    X86StoreFence,
}

impl DeviceStoreFence {
    /// Execute the architecture store fence. Callers must invoke this only through the single
    /// publication seam (`AqlChunkSink::device_store_fence`) so production and tests share one
    /// effect position between body writes and header publication.
    #[inline]
    pub(crate) fn issue(self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `sfence` is part of SSE, baseline on x86_64; it only orders stores.
        unsafe {
            core::arch::x86_64::_mm_sfence();
        }
        #[cfg(not(target_arch = "x86_64"))]
        match self {}

        ISSUED_DEVICE_STORE_FENCES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Cumulative architecture store fences issued through [`DeviceStoreFence::issue`].
#[cfg(test)]
pub(crate) fn issued_device_store_fence_count() -> usize {
    ISSUED_DEVICE_STORE_FENCES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Swap the fence counter to zero and return the previous count (receipt / tests).
pub fn take_issued_device_store_fence_count() -> usize {
    ISSUED_DEVICE_STORE_FENCES.swap(0, std::sync::atomic::Ordering::Relaxed)
}

static ISSUED_DEVICE_STORE_FENCES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Closed publication contract retained by a live HSA queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueuePublicationContract {
    /// The ring is host-coherent and needs no device-store fence.
    HostCoherent,
    /// The device ring is reached through a link with proven coherent ordering.
    CoherentDeviceMemory,
    /// The device ring is reached over PCIe and uses the selected host fence.
    PcieDeviceMemory { fence: DeviceStoreFence },
}

/// Typed failure to establish safe AQL publication ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QueuePublicationError {
    /// The queue-ring allocation or required link fact was not authoritative.
    #[error(
        "unknown AQL queue publication facts: ring={ring_memory:?}, link={host_link:?}, host={host_target}"
    )]
    Unknown {
        ring_memory: Option<QueueRingMemory>,
        host_link: Option<QueueHostLink>,
        host_target: QueueHostTarget,
    },
    /// The host target has no proven fence for this device-ring/link combination.
    #[error(
        "unsupported AQL queue publication ordering: ring={ring_memory:?}, link={host_link:?}, host={host_target}"
    )]
    UnsupportedOrdering {
        ring_memory: QueueRingMemory,
        host_link: QueueHostLink,
        host_target: QueueHostTarget,
    },
}

/// Validated queue publication state. Construction is the only raw-fact boundary; publication
/// consumes this wrapper so ring properties cannot be dropped between context creation and replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueuePublication {
    contract: QueuePublicationContract,
}

impl QueuePublication {
    pub(crate) fn validate(
        ring_memory: Option<QueueRingMemory>,
        host_link: Option<QueueHostLink>,
        host_target: QueueHostTarget,
    ) -> Result<Self, QueuePublicationError> {
        let contract = match (ring_memory, host_link, host_target) {
            (Some(QueueRingMemory::HostCoherent), _, _) => QueuePublicationContract::HostCoherent,
            (Some(QueueRingMemory::DeviceMemory), Some(QueueHostLink::Coherent), _) => {
                QueuePublicationContract::CoherentDeviceMemory
            }
            (
                Some(QueueRingMemory::DeviceMemory),
                Some(QueueHostLink::Pcie),
                QueueHostTarget::X86,
            ) => {
                #[cfg(target_arch = "x86_64")]
                {
                    QueuePublicationContract::PcieDeviceMemory {
                        fence: DeviceStoreFence::X86StoreFence,
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    return Err(QueuePublicationError::UnsupportedOrdering {
                        ring_memory: QueueRingMemory::DeviceMemory,
                        host_link: QueueHostLink::Pcie,
                        host_target,
                    });
                }
            }
            (
                Some(ring_memory @ QueueRingMemory::DeviceMemory),
                Some(host_link @ QueueHostLink::Pcie),
                host_target @ QueueHostTarget::Other(_),
            ) => {
                return Err(QueuePublicationError::UnsupportedOrdering {
                    ring_memory,
                    host_link,
                    host_target,
                });
            }
            (ring_memory, host_link, host_target) => {
                return Err(QueuePublicationError::Unknown {
                    ring_memory,
                    host_link,
                    host_target,
                });
            }
        };
        Ok(Self { contract })
    }

    pub(crate) const fn contract(self) -> QueuePublicationContract {
        self.contract
    }

    pub(crate) fn publish_chunks<S: AqlChunkSink>(
        self,
        packet_count: usize,
        queue_size: u64,
        sink: &mut S,
    ) -> Result<usize, S::Error> {
        let mask = queue_size - 1;
        let chunk_size = (queue_size / 2).max(1) as usize;
        let mut physical_submissions = 0;

        for chunk_start in (0..packet_count).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(packet_count);
            let chunk_n = chunk_end - chunk_start;
            let write_index = sink.reserve(chunk_n)?;
            sink.wait_for_space(write_index, chunk_n, queue_size)?;

            for offset in 0..chunk_n {
                let packet_index = chunk_start + offset;
                let slot_index = (write_index + offset as u64) & mask;
                sink.write_body(packet_index, slot_index, packet_index == packet_count - 1)?;
            }

            // Single injected fence effect: the sink issues the real architecture fence (or the
            // test recorder observes that exact call). Do not also call fence.issue() here.
            if let QueuePublicationContract::PcieDeviceMemory { fence } = self.contract {
                sink.device_store_fence(fence);
            }

            for offset in (0..chunk_n).rev() {
                let packet_index = chunk_start + offset;
                let slot_index = (write_index + offset as u64) & mask;
                sink.publish_header(packet_index, slot_index)?;
            }

            sink.ring_doorbell(write_index + chunk_n as u64 - 1)?;
            physical_submissions += 1;
        }

        Ok(physical_submissions)
    }
}

pub(crate) trait AqlChunkSink {
    type Error;

    fn reserve(&mut self, packet_count: usize) -> Result<u64, Self::Error>;
    fn wait_for_space(
        &mut self,
        write_index: u64,
        packet_count: usize,
        queue_size: u64,
    ) -> Result<(), Self::Error>;
    fn write_body(
        &mut self,
        packet_index: usize,
        slot_index: u64,
        is_final_packet: bool,
    ) -> Result<(), Self::Error>;
    /// Issue the device-store fence for a PCIe device-memory chunk. Production must execute the
    /// real architecture fence here; tests must observe this same call position.
    fn device_store_fence(&mut self, fence: DeviceStoreFence);
    fn publish_header(&mut self, packet_index: usize, slot_index: u64) -> Result<(), Self::Error>;
    fn ring_doorbell(&mut self, packet_index: u64) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::convert::Infallible;
    use std::sync::Mutex;

    use super::*;

    static DEVICE_FENCE_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn issued_device_store_fences() -> usize {
        issued_device_store_fence_count()
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Event {
        Reserve(usize),
        Body {
            packet: usize,
            slot: u64,
            final_packet: bool,
        },
        Fence(DeviceStoreFence),
        Header {
            packet: usize,
            slot: u64,
        },
        Doorbell(u64),
    }

    struct RecordingSink {
        reservations: VecDeque<u64>,
        events: Vec<Event>,
    }

    impl RecordingSink {
        fn new(reservations: impl IntoIterator<Item = u64>) -> Self {
            Self {
                reservations: reservations.into_iter().collect(),
                events: Vec::new(),
            }
        }
    }

    impl AqlChunkSink for RecordingSink {
        type Error = Infallible;

        fn reserve(&mut self, packet_count: usize) -> Result<u64, Self::Error> {
            self.events.push(Event::Reserve(packet_count));
            Ok(self
                .reservations
                .pop_front()
                .expect("missing test reservation"))
        }

        fn wait_for_space(
            &mut self,
            _write_index: u64,
            _packet_count: usize,
            _queue_size: u64,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn write_body(
            &mut self,
            packet_index: usize,
            slot_index: u64,
            is_final_packet: bool,
        ) -> Result<(), Self::Error> {
            self.events.push(Event::Body {
                packet: packet_index,
                slot: slot_index,
                final_packet: is_final_packet,
            });
            Ok(())
        }

        fn device_store_fence(&mut self, fence: DeviceStoreFence) {
            // Same injected effect production uses: issue the real fence, then record its position.
            fence.issue();
            self.events.push(Event::Fence(fence));
        }

        fn publish_header(
            &mut self,
            packet_index: usize,
            slot_index: u64,
        ) -> Result<(), Self::Error> {
            self.events.push(Event::Header {
                packet: packet_index,
                slot: slot_index,
            });
            Ok(())
        }

        fn ring_doorbell(&mut self, packet_index: u64) -> Result<(), Self::Error> {
            self.events.push(Event::Doorbell(packet_index));
            Ok(())
        }
    }

    fn validated(
        ring_memory: QueueRingMemory,
        host_link: Option<QueueHostLink>,
        host_target: QueueHostTarget,
    ) -> QueuePublication {
        QueuePublication::validate(Some(ring_memory), host_link, host_target)
            .expect("test publication facts should validate")
    }

    #[test]
    fn device_store_fence_wrapper_is_the_direct_x86_intrinsic() {
        let source = include_str!("publication.rs");
        let wrapper_start = source
            .find("impl DeviceStoreFence {")
            .expect("DeviceStoreFence implementation missing");
        let wrapper_end = source[wrapper_start..]
            .find("/// Closed publication contract")
            .map(|offset| wrapper_start + offset)
            .expect("DeviceStoreFence implementation boundary missing");
        let wrapper = &source[wrapper_start..wrapper_end];

        assert_eq!(
            wrapper.matches("core::arch::x86_64::_mm_sfence();").count(),
            1,
            "the x86_64 intrinsic must remain in the fence wrapper"
        );
        assert!(
            !wrapper.contains("atomic::fence") && !wrapper.contains("compiler_fence"),
            "a Rust memory-model fence does not replace the PCIe device-store fence"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pcie_device_ring_fences_each_chunk_between_bodies_and_reverse_headers() {
        let _fence_test = DEVICE_FENCE_TEST_LOCK.lock().unwrap();
        let publication = validated(
            QueueRingMemory::DeviceMemory,
            Some(QueueHostLink::Pcie),
            QueueHostTarget::X86,
        );
        let fences_before = issued_device_store_fences();
        let mut sink = RecordingSink::new([3, 5]);

        let submissions = publication.publish_chunks(3, 4, &mut sink).unwrap();

        assert_eq!(submissions, 2);
        assert_eq!(issued_device_store_fences() - fences_before, 2);
        assert_eq!(
            sink.events,
            vec![
                Event::Reserve(2),
                Event::Body {
                    packet: 0,
                    slot: 3,
                    final_packet: false,
                },
                Event::Body {
                    packet: 1,
                    slot: 0,
                    final_packet: false,
                },
                Event::Fence(DeviceStoreFence::X86StoreFence),
                Event::Header { packet: 1, slot: 0 },
                Event::Header { packet: 0, slot: 3 },
                Event::Doorbell(4),
                Event::Reserve(1),
                Event::Body {
                    packet: 2,
                    slot: 1,
                    final_packet: true,
                },
                Event::Fence(DeviceStoreFence::X86StoreFence),
                Event::Header { packet: 2, slot: 1 },
                Event::Doorbell(5),
            ]
        );
    }

    /// A single dispatch is a one-packet batch through this publisher (`RocmContext::dispatch`), so
    /// on a PCIe device ring its header is stored only after the body write and the device-store
    /// fence, and the doorbell rings last.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn single_packet_dispatch_orders_body_then_fence_then_header_then_doorbell() {
        let _fence_test = DEVICE_FENCE_TEST_LOCK.lock().unwrap();
        let publication = validated(
            QueueRingMemory::DeviceMemory,
            Some(QueueHostLink::Pcie),
            QueueHostTarget::X86,
        );
        let mut sink = RecordingSink::new([6]);

        let submissions = publication.publish_chunks(1, 8, &mut sink).unwrap();

        assert_eq!(submissions, 1);
        assert_eq!(
            sink.events,
            vec![
                Event::Reserve(1),
                Event::Body {
                    packet: 0,
                    slot: 6,
                    final_packet: true,
                },
                Event::Fence(DeviceStoreFence::X86StoreFence),
                Event::Header { packet: 0, slot: 6 },
                Event::Doorbell(6),
            ]
        );
    }

    #[test]
    fn host_coherent_ring_has_no_device_fence_and_preserves_publication_order() {
        let _fence_test = DEVICE_FENCE_TEST_LOCK.lock().unwrap();
        let publication = validated(
            QueueRingMemory::HostCoherent,
            Some(QueueHostLink::Pcie),
            QueueHostTarget::Other("test-host"),
        );
        let fences_before = issued_device_store_fences();
        let mut sink = RecordingSink::new([7]);

        let submissions = publication.publish_chunks(2, 8, &mut sink).unwrap();

        assert_eq!(submissions, 1);
        assert_eq!(issued_device_store_fences(), fences_before);
        assert_eq!(
            sink.events,
            vec![
                Event::Reserve(2),
                Event::Body {
                    packet: 0,
                    slot: 7,
                    final_packet: false,
                },
                Event::Body {
                    packet: 1,
                    slot: 0,
                    final_packet: true,
                },
                Event::Header { packet: 1, slot: 0 },
                Event::Header { packet: 0, slot: 7 },
                Event::Doorbell(8),
            ]
        );
    }

    #[test]
    fn coherent_device_ring_has_no_pcie_fence() {
        let _fence_test = DEVICE_FENCE_TEST_LOCK.lock().unwrap();
        let publication = validated(
            QueueRingMemory::DeviceMemory,
            Some(QueueHostLink::Coherent),
            QueueHostTarget::Other("coherent-test-host"),
        );
        let fences_before = issued_device_store_fences();
        let mut sink = RecordingSink::new([0]);

        publication.publish_chunks(1, 2, &mut sink).unwrap();

        assert_eq!(issued_device_store_fences(), fences_before);
        assert_eq!(
            sink.events,
            vec![
                Event::Reserve(1),
                Event::Body {
                    packet: 0,
                    slot: 0,
                    final_packet: true,
                },
                Event::Header { packet: 0, slot: 0 },
                Event::Doorbell(0),
            ]
        );
    }

    #[test]
    fn unknown_ring_or_device_link_rejects_before_reservation() {
        let cases = [
            (None, Some(QueueHostLink::Pcie)),
            (Some(QueueRingMemory::DeviceMemory), None),
        ];

        for (ring_memory, host_link) in cases {
            let mut sink = RecordingSink::new([0]);
            let result = QueuePublication::validate(
                ring_memory,
                host_link,
                QueueHostTarget::Other("unknown-test-host"),
            )
            .and_then(|publication| {
                publication
                    .publish_chunks(1, 2, &mut sink)
                    .map_err(|never| match never {})
            });

            assert!(matches!(result, Err(QueuePublicationError::Unknown { .. })));
            assert!(sink.events.is_empty());
        }
    }

    #[test]
    fn unsupported_pcie_host_rejects_before_reservation() {
        let mut sink = RecordingSink::new([0]);
        let result = QueuePublication::validate(
            Some(QueueRingMemory::DeviceMemory),
            Some(QueueHostLink::Pcie),
            QueueHostTarget::Other("aarch64"),
        )
        .and_then(|publication| {
            publication
                .publish_chunks(1, 2, &mut sink)
                .map_err(|never| match never {})
        });

        assert_eq!(
            result,
            Err(QueuePublicationError::UnsupportedOrdering {
                ring_memory: QueueRingMemory::DeviceMemory,
                host_link: QueueHostLink::Pcie,
                host_target: QueueHostTarget::Other("aarch64"),
            })
        );
        assert!(sink.events.is_empty());
    }

    #[test]
    fn empty_replay_has_no_reservation_or_publication() {
        let publication = validated(
            QueueRingMemory::HostCoherent,
            None,
            QueueHostTarget::current(),
        );
        let mut sink = RecordingSink::new([]);

        let submissions = publication.publish_chunks(0, 8, &mut sink).unwrap();

        assert_eq!(submissions, 0);
        assert!(sink.events.is_empty());
    }

    #[test]
    fn admitted_359_packet_graph_reports_one_successful_production_submission() {
        let publication = validated(
            QueueRingMemory::HostCoherent,
            None,
            QueueHostTarget::current(),
        );
        let mut sink = RecordingSink::new([17]);

        let submissions = publication.publish_chunks(359, 1_024, &mut sink).unwrap();

        assert_eq!(submissions, 1);
        assert_eq!(
            sink.events
                .iter()
                .filter(|event| matches!(event, Event::Doorbell(_)))
                .count(),
            1
        );
        assert_eq!(sink.events.last(), Some(&Event::Doorbell(375)));
    }
}

use crate::*;

/// Provenance retained for a Card 333 discrete PCIe device-ring receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRingReceiptProvenance {
    pub allocate_queue_dev_mem: bool,
    pub queue_api: &'static str,
    pub queue_type_requested: u32,
    pub queue_type_returned: u32,
    pub queue_features: u32,
    pub queue_size: u32,
    pub agent_name: String,
    pub product_name: String,
    pub isa_name: String,
    pub bdfid: Option<u32>,
    pub pci_domain: Option<u32>,
    pub num_link_hops: u32,
    pub link_types: Vec<u32>,
    pub contract: QueuePublicationContract,
}

/// HSA packet type `INVALID` (header bits 0..7). Body writes must leave this value until the
/// reverse header-publication step release-stores a valid KERNEL_DISPATCH header.
pub(crate) const AQL_PACKET_HEADER_INVALID: u16 = 0;

/// Write one kernel-dispatch packet body into `ring[slot_index]` while leaving the header
/// `INVALID`. Shared by the live publisher so model-free tests can exercise the real writer.
///
/// # Safety
///
/// `ring` must point to at least `slot_index + 1` writable packet slots that no other thread writes
/// concurrently, and the slot's current packet must not be one the packet processor may still read.
pub(crate) unsafe fn write_kernel_dispatch_packet_body(
    ring: *mut bindings::hsa_kernel_dispatch_packet_t,
    slot_index: u64,
    kernel: &KernelHandle,
    kernarg_address: *mut c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    completion_signal: bindings::hsa_signal_t,
) {
    let setup = if block.1 == 1 && block.2 == 1 && grid.1 == 1 && grid.2 == 1 {
        1u16
    } else if block.2 == 1 && grid.2 == 1 {
        2u16
    } else {
        3u16
    };
    let packet = bindings::hsa_kernel_dispatch_packet_t {
        header: AQL_PACKET_HEADER_INVALID,
        setup,
        workgroup_size_x: block.0 as u16,
        workgroup_size_y: block.1 as u16,
        workgroup_size_z: block.2 as u16,
        reserved0: 0,
        grid_size_x: grid.0,
        grid_size_y: grid.1,
        grid_size_z: grid.2,
        private_segment_size: kernel.private_segment_size().max(256),
        group_segment_size: kernel.group_segment_size(),
        kernel_object: kernel.kernel_object(),
        kernarg_address,
        reserved2: 0,
        completion_signal,
    };
    // SAFETY: the caller guarantees `ring[slot_index]` is an in-bounds, writable slot the packet
    // processor is not reading.
    unsafe {
        std::ptr::write(ring.add(slot_index as usize), packet);
    }
}

/// The kernarg address a dispatch entry contributes to its AQL packet. Production dispatches carry a
/// [`RocmBuffer`], whose allocation only a live [`RocmContext`] can create (its `Drop` frees HSA
/// memory through the `ResourceOwner` that made it), while the ring writer needs nothing but the
/// address - so a model-free test supplies a synthetic one instead of a production-shaped buffer
/// constructor (plan 558-648).
pub(crate) trait KernargAddress {
    /// The value the packet's `kernarg_address` field takes for this dispatch.
    fn kernarg_address(&self) -> *mut c_void;
}

impl KernargAddress for RocmBuffer {
    fn kernarg_address(&self) -> *mut c_void {
        self.alloc.as_ptr()
    }
}

/// The context-free half of [`LiveAqlChunkSink`]: everything `write_body`, `publish_header` and
/// `device_store_fence` touch - the ring pointer, the dispatches, the completion signal and the
/// header value - with no `RocmContext` field, so a model-free test can construct it over a synthetic
/// ring and assert publication behavior instead of reading source (card 617). `K` is the dispatches'
/// kernarg source: [`RocmBuffer`] in production, a test newtype in `tests.rs`.
///
/// `ring` is the live queue's `base_address`, so every slot index passed here (masked by the
/// validated power-of-two queue size, after the queue half's `wait_for_space` proved the packet
/// processor has consumed it) addresses a free slot the packet processor is not reading. The writer
/// shares that ring with the queue half and never touches the queue, the doorbell or the context.
#[allow(clippy::type_complexity)]
pub(crate) struct AqlRingWriter<'a, K: KernargAddress> {
    pub(crate) dispatches: &'a [(KernelHandle, K, (u32, u32, u32), (u32, u32, u32))],
    pub(crate) completion: bindings::hsa_signal_t,
    pub(crate) ring: *mut bindings::hsa_kernel_dispatch_packet_t,
    pub(crate) header: u16,
}

impl<K: KernargAddress> AqlRingWriter<'_, K> {
    /// Write `dispatches[packet_index]`'s packet body into `ring[slot_index]` through the shared
    /// [`write_kernel_dispatch_packet_body`], leaving the header `INVALID`. Only the final packet
    /// carries the completion signal; earlier ones get the null signal.
    pub(crate) fn write_body(
        &mut self,
        packet_index: usize,
        slot_index: u64,
        is_final_packet: bool,
    ) {
        let (kernel, kernarg, grid, block) = &self.dispatches[packet_index];
        // SAFETY: `slot_index` addresses a free slot of the ring (see `AqlRingWriter`), and only this
        // thread writes it.
        unsafe {
            write_kernel_dispatch_packet_body(
                self.ring,
                slot_index,
                kernel,
                kernarg.kernarg_address(),
                *grid,
                *block,
                if is_final_packet {
                    self.completion
                } else {
                    bindings::hsa_signal_t { handle: 0 }
                },
            );
        }
    }

    /// The single injected fence effect between a chunk's body writes and its header publication:
    /// the real architecture fence, issued without a live queue in the way.
    pub(crate) fn device_store_fence(&mut self, fence: DeviceStoreFence) {
        fence.issue();
    }

    /// Release-store the header of the slot whose body `write_body` just wrote, so the packet
    /// processor reads the completed packet exactly once.
    pub(crate) fn publish_header(&mut self, slot_index: u64) {
        // SAFETY: `slot_index` addresses the slot whose body `write_body` just wrote; the header is the
        // packet's first, 2-byte-aligned `u16`, which the packet processor reads atomically.
        unsafe {
            let slot = self.ring.add(slot_index as usize);
            let header_ptr = slot as *mut std::sync::atomic::AtomicU16;
            (*header_ptr).store(self.header, std::sync::atomic::Ordering::Release);
        }
    }
}

/// Publishes packets into the context's live AQL ring. `doorbell` is the context queue's doorbell
/// signal, and this half owns the only [`publication::AqlChunkSink`] methods that call HSA
/// (`reserve`, `wait_for_space`, `ring_doorbell`); the three ring methods forward to the
/// [`AqlRingWriter`] field, which needs no queue. `writer.ring` is the queue's `base_address`, so
/// every slot index either half passes (masked by the validated power-of-two queue size, after
/// `wait_for_space` proved the packet processor has consumed it) addresses a free slot of the ring.
pub(crate) struct LiveAqlChunkSink<'a> {
    pub(crate) context: &'a RocmContext,
    pub(crate) doorbell: bindings::hsa_signal_t,
    pub(crate) writer: AqlRingWriter<'a, RocmBuffer>,
}

impl publication::AqlChunkSink for LiveAqlChunkSink<'_> {
    type Error = RocmError;

    fn reserve(&mut self, packet_count: usize) -> Result<u64, Self::Error> {
        // SAFETY: `context.queue` is the context's live queue (a `RocmContext` invariant).
        Ok(unsafe {
            (self.context.hsa.funcs.hsa_queue_add_write_index_acq_rel)(
                self.context.queue,
                packet_count as u64,
            )
        })
    }

    fn wait_for_space(
        &mut self,
        write_index: u64,
        packet_count: usize,
        queue_size: u64,
    ) -> Result<(), Self::Error> {
        let timeout = self.context.wait_timeout.as_secs();
        let start = std::time::Instant::now();
        let mut spins: u64 = 0;
        loop {
            // SAFETY: `context.queue` is the context's live queue (a `RocmContext` invariant).
            let read_index = unsafe {
                (self.context.hsa.funcs.hsa_queue_load_read_index_relaxed)(self.context.queue)
            };
            if write_index + packet_count as u64 - read_index <= queue_size {
                return Ok(());
            }
            spins = spins.wrapping_add(1);
            if spins & 0xFFFF == 0 && start.elapsed().as_secs() >= timeout {
                return Err(RocmError::QueueWaitTimeout(timeout, "upload back-pressure"));
            }
            std::thread::yield_now();
        }
    }

    fn write_body(
        &mut self,
        packet_index: usize,
        slot_index: u64,
        is_final_packet: bool,
    ) -> Result<(), Self::Error> {
        self.writer
            .write_body(packet_index, slot_index, is_final_packet);
        Ok(())
    }

    fn device_store_fence(&mut self, fence: DeviceStoreFence) {
        // Production half of the single injected fence effect from publish_chunks, on the writer that
        // owns the body and header stores it orders.
        self.writer.device_store_fence(fence);
    }

    fn publish_header(&mut self, _packet_index: usize, slot_index: u64) -> Result<(), Self::Error> {
        self.writer.publish_header(slot_index);
        Ok(())
    }

    fn ring_doorbell(&mut self, packet_index: u64) -> Result<(), Self::Error> {
        // SAFETY: `doorbell` is the live queue's doorbell signal.
        unsafe {
            (self.context.hsa.funcs.hsa_signal_store_release)(self.doorbell, packet_index as i64);
        }
        Ok(())
    }
}

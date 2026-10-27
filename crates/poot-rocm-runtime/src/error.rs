use crate::QueuePublicationError;
use crate::aql::DispatchGeometryViolation;
use crate::bindings::hsa_status_t;
use crate::queue::QueueRole;
use poot_target::BufferStorage;

/// Typed HSA error. `Driver(HsaStatus)` carries the raw status; its `Display` maps common ones to
/// their names and falls back to "status N". [`RocmError::LibraryNotFound`] and
/// [`RocmError::NoGpuAgent`] are the typed skip-dispatch signals (spec 063 FR-004, FR-005).
#[derive(Debug, thiserror::Error)]
pub enum RocmError {
    /// `libhsa-runtime64.so.1` (or one of its direct dependencies) is not on the loader path. `0`
    /// carries the OS dlopen error string. Reported at runtime, not build time (FR-004).
    #[error("libhsa-runtime64.so.1 not found: {0}")]
    LibraryNotFound(String),
    /// A required HSA symbol was missing from the loaded library.
    #[error("hsa symbol missing: {0}")]
    MissingSymbol(&'static str),
    /// `hsa_iterate_agents` reported a callback error (e.g. the GPU picker failed).
    #[error("hsa agent enumeration: {0}")]
    AgentEnum(&'static str),
    /// No GPU agent was visible to HSA (only CPU agents, or none). Dispatch tests treat this as a clean
    /// skip (FR-005).
    #[error("no GPU agent found (only CPU agents visible)")]
    NoGpuAgent,
    /// A handle passed to an HSA call was invalid for this runtime instance.
    #[error("invalid agent handle")]
    InvalidAgent,
    /// Generic driver / HSA failure.
    #[error("HSA driver error: {0}")]
    Driver(HsaStatus),
    /// Generic dispatch-side error with a free-form message: kernel lookup failures, kernel-arg packing
    /// errors, and the oracle-mismatch fallback (`Hsa("N elements ... disagreed with CPU oracle")`).
    #[error("HSA error: {0}")]
    Hsa(String),
    /// The kernel name passed to `lookup_kernel` contained an embedded NUL byte.
    #[error("invalid kernel name (NUL byte)")]
    InvalidKernelName,
    /// [`RocmContext::load_hsaco`] was given a [`poot_codegen::CompiledKernel`] compiled for a
    /// different backend (card 608): this runtime only ever loads an `AmdGcn`-compiled kernel.
    #[error("ROCm load_hsaco requires an AmdGcn-compiled kernel, got one compiled for {actual:?}")]
    WrongKernelTarget {
        /// The target the kernel was actually compiled for.
        actual: poot_target::TargetId,
    },
    /// The buffers a ROCm dispatch packed into its kernarg segment did not meet the compiled kernel's
    /// argument schema (card 608, SC-002): a typed release-build error before
    /// any submission, checked against the schema `load_hsaco`/`lookup_kernel` carried from the
    /// `CompiledKernel`.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_codegen::KernelArgError),
    /// A GPU queue wait (`synchronize` / upload back-pressure) spun past the wall-clock guard without
    /// the dispatch completing. Fails with a clean error instead of an infinite 99%-CPU busy-spin. The
    /// bound is generous (see [`crate::RocmContextOptions::wait_timeout`]). The context is poisoned
    /// by this error.
    #[error("GPU queue wait timed out after {0}s ({1}) - dispatch hung or stalled")]
    QueueWaitTimeout(u64, &'static str),
    /// An earlier bounded wait timed out, so its packet may still be in flight and its resources were
    /// leaked instead of freed. The context refuses every later submission before touching the queue.
    #[error("{op} refused: an earlier GPU wait timed out, so this ROCm context is poisoned")]
    ContextPoisoned {
        /// The runtime entry point that refused.
        op: &'static str,
    },
    /// The device reported a fault and is gone: a queue error, a GPU memory fault or a reset. The
    /// context is poisoned by this error; every later call is [`RocmError::ContextPoisoned`].
    #[error("the GPU device was lost: {0}")]
    DeviceLost(DeviceFault),
    /// A dispatch whose grid or workgroup size no well-formed AQL packet can carry, refused before any
    /// ring slot was reserved.
    #[error("{op} refused dispatch {dispatch} (grid {grid:?}, workgroup {block:?}): {violation}")]
    InvalidDispatchGeometry {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Index of the offending dispatch within the submitted list.
        dispatch: usize,
        /// Requested grid size in work-items per dimension.
        grid: (u32, u32, u32),
        /// Requested workgroup size per dimension.
        block: (u32, u32, u32),
        /// Which packet-format rule the geometry broke.
        violation: DispatchGeometryViolation,
    },
    /// The queue returned by HSA does not satisfy the requested type/features/size contract.
    #[error("HSA queue contract violation: {0}")]
    QueueContract(String),
    /// The selected ordinary-queue creator conflicts with the process-wide override snapshot
    /// taken before this crate first loaded HSA.
    #[error(
        "queue role {role:?} conflicts with HSA_ALLOCATE_QUEUE_DEV_MEM provenance (at first HSA load: {override_at_load}, now: {override_now})"
    )]
    QueueRingProvenanceConflict {
        /// The queue role the context was opened for.
        role: QueueRole,
        /// Whether `HSA_ALLOCATE_QUEUE_DEV_MEM=1` was set when this crate first loaded HSA.
        override_at_load: bool,
        /// Whether it is set now.
        override_now: bool,
    },
    /// Restricted replay unexpectedly produced a number of physical submissions other than one.
    #[error("restricted replay required one physical AQL submission, got {0}")]
    RestrictedReplaySubmissions(usize),
    /// Element-to-byte conversion overflowed before allocation or transfer.
    #[error("{op} byte size overflow: {elements} elements * {element_bytes} bytes")]
    ByteSizeOverflow {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Element count of the request.
        elements: usize,
        /// Bytes per element.
        element_bytes: usize,
    },
    /// A typed transfer was requested against a differently represented allocation (card 527: checked
    /// against the buffer's full storage contract - element kind, logical dtype and layout - not just
    /// element kind).
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        /// The runtime entry point that refused.
        op: &'static str,
        /// The storage the operation requires.
        expected: BufferStorage,
        /// The storage the buffer holds.
        actual: BufferStorage,
    },
    /// A whole-buffer transfer had the wrong logical element count.
    #[error("{op} size mismatch: buffer has {have} elements, got {got}")]
    SizeMismatch {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Elements the buffer holds.
        have: usize,
        /// Elements the caller supplied.
        got: usize,
    },
    /// A checked byte range exceeded the immutable allocation capacity.
    #[error(
        "{op} range [{byte_offset}, {byte_end}) exceeds the buffer's {byte_capacity}-byte capacity"
    )]
    RangeOutOfBounds {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Inclusive start of the requested range, in bytes.
        byte_offset: usize,
        /// Exclusive end of the requested range, in bytes.
        byte_end: usize,
        /// The allocation's capacity, in bytes.
        byte_capacity: usize,
    },
    /// Offset or length arithmetic overflowed before pointer arithmetic.
    #[error("{op} byte range overflow at element offset {element_offset} with {elements} elements")]
    RangeOverflow {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Requested start, in elements.
        element_offset: usize,
        /// Requested length, in elements.
        elements: usize,
    },
    /// A native narrow payload ended in a partial element.
    #[error("{op} got {bytes} bytes, which is not a whole number of {element_bytes}-byte elements")]
    InvalidByteLength {
        /// The runtime entry point that refused.
        op: &'static str,
        /// Length of the payload, in bytes.
        bytes: usize,
        /// Bytes per element of the buffer.
        element_bytes: usize,
    },
    /// More buffers were bound than the kernel's kernarg segment has room for (16 bytes each).
    #[error(
        "{op}: {slots} pointer/length pairs do not fit the kernel's {segment}-byte kernarg segment"
    )]
    KernargSlotsExceedSegment {
        /// The runtime entry point that refused.
        op: &'static str,
        /// How many buffers the caller tried to bind.
        slots: usize,
        /// The kernel's kernarg segment size, in bytes.
        segment: usize,
    },
    /// The queue's packet-ring memory and host-link facts could not be validated into a safe
    /// publication contract.
    #[error(transparent)]
    QueuePublication(#[from] QueuePublicationError),
    /// Card 333 PCIe receipt path: the selected GPU agent reports the APU memory property.
    #[error("GPU agent is an APU; discrete PCIe device-ring receipt requires a non-APU agent")]
    ApuNotDiscrete,
    /// Card 333 PCIe receipt path: no CPU agent was visible for CPU-to-GPU link classification.
    #[error("no CPU agent found for PCIe host-link classification")]
    NoCpuAgent,
    /// Card 333 PCIe receipt path: the CPU-to-GPU pool path is not a PCIe link.
    #[error(
        "CPU-to-GPU memory-pool link is not PCIe (num_hops={num_hops}, link_types={link_types:?})"
    )]
    HostLinkNotPcie {
        /// Number of hops on the CPU-to-GPU pool path.
        num_hops: u32,
        /// HSA link-info type of each hop; none was PCIe.
        link_types: Vec<u32>,
    },
    /// Card 333 PCIe receipt path: ROCr rejected device-memory queue creation (often Large BAR).
    #[error("device-memory AQL ring queue creation failed: {0}")]
    DeviceRingQueueCreate(HsaStatus),
}

/// What the device reported when it was lost, as ROCr delivered it to the queue error callback or the
/// system event handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DeviceFault {
    /// The HSA queue was inactivated by an asynchronous error (for example an invalid packet).
    #[error("queue error {0}")]
    QueueError(HsaStatus),
    /// A kernel accessed memory the GPU cannot reach.
    #[error("memory fault at {address:#x} (reason mask {reason_mask:#x})")]
    MemoryFault {
        /// The virtual address accessed.
        address: u64,
        /// `hsa_amd_memory_fault_reason_t` bits.
        reason_mask: u32,
    },
    /// The GPU hit a hardware exception and was reset.
    #[error("hardware exception (reset cause {reset_cause:#x})")]
    HardwareException {
        /// `hsa_amd_hw_exception_reset_cause_t` bits.
        reset_cause: u32,
    },
    /// The GPU reported a memory error.
    #[error("memory error at {address:#x} (reason mask {reason_mask:#x})")]
    MemoryError {
        /// The virtual address involved.
        address: u64,
        /// `hsa_amd_memory_error_reason_t` bits.
        reason_mask: u32,
    },
}

/// Human-readable HSA status. `Display` prints the textual name (static table) and the raw hex code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HsaStatus {
    /// The raw `hsa_status_t` returned by the failing HSA call.
    pub code: hsa_status_t,
}

impl std::fmt::Display for HsaStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (status 0x{:x})", describe(self.code), self.code)
    }
}

pub(crate) fn describe(code: hsa_status_t) -> &'static str {
    match code {
        0x0 => "HSA_STATUS_SUCCESS",
        0x1000 => "HSA_STATUS_ERROR",
        0x1001 => "HSA_STATUS_ERROR_INVALID_ARGUMENT",
        0x1002 => "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION",
        0x1003 => "HSA_STATUS_ERROR_INVALID_ALLOCATION",
        0x1004 => "HSA_STATUS_ERROR_INVALID_AGENT",
        0x1005 => "HSA_STATUS_ERROR_INVALID_REGION",
        0x1006 => "HSA_STATUS_ERROR_INVALID_SIGNAL",
        0x1007 => "HSA_STATUS_ERROR_INVALID_QUEUE",
        0x1008 => "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
        0x1009 => "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT",
        0x100A => "HSA_STATUS_ERROR_RESOURCE_FREE",
        0x100B => "HSA_STATUS_ERROR_NOT_INITIALIZED",
        0x100C => "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW",
        0x100D => "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS",
        0x100E => "HSA_STATUS_ERROR_INVALID_INDEX",
        0x100F => "HSA_STATUS_ERROR_INVALID_ISA",
        0x1010 => "HSA_STATUS_ERROR_INVALID_CODE_OBJECT",
        0x1011 => "HSA_STATUS_ERROR_OUT_REGISTRATIONS",
        0x1017 => "HSA_STATUS_ERROR_INVALID_ISA_NAME",
        _ => "HSA_STATUS_ERROR_OTHER",
    }
}

pub(crate) fn check(code: hsa_status_t) -> Result<(), RocmError> {
    if code == 0 {
        Ok(())
    } else {
        Err(RocmError::Driver(HsaStatus { code }))
    }
}

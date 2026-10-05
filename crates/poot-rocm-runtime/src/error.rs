use crate::*;

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
    MissingSymbol(#[from] ffi::MissingSymbol),
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
    /// [`RocmContext::load_hsaco`] was given a [`poot_runtime_common::CompiledKernel`] compiled for a
    /// different backend (card 608): this runtime only ever loads an `AmdGcn`-compiled kernel.
    #[error("ROCm load_hsaco requires an AmdGcn-compiled kernel, got one compiled for {actual:?}")]
    WrongKernelTarget { actual: poot_target::Backend },
    /// The buffers a ROCm dispatch packed into its kernarg segment did not meet the compiled kernel's
    /// argument schema (card 608, SC-002): a typed release-build error before
    /// any submission, checked against the schema `load_hsaco`/`lookup_kernel` carried from the
    /// `CompiledKernel`.
    #[error("kernel argument schema violation: {0}")]
    KernelArgs(#[from] poot_runtime_common::KernelArgError),
    /// A GPU queue wait (`synchronize` / upload back-pressure) spun past the wall-clock guard without
    /// the dispatch completing. Fails with a clean error instead of an infinite 99%-CPU busy-spin. The
    /// bound is generous (see [`crate::RocmContextOptions::wait_timeout`]). The context is poisoned
    /// by this error.
    #[error("GPU queue wait timed out after {0}s ({1}) - dispatch hung or stalled")]
    QueueWaitTimeout(u64, &'static str),
    /// An earlier bounded wait timed out, so its packet may still be in flight and its resources were
    /// leaked instead of freed. The context refuses every later submission before touching the queue.
    #[error("{op} refused: an earlier GPU wait timed out, so this ROCm context is poisoned")]
    ContextPoisoned { op: &'static str },
    /// The queue returned by HSA does not satisfy the requested type/features/size contract.
    #[error("HSA queue contract violation: {0}")]
    QueueContract(String),
    /// The selected ordinary-queue creator conflicts with the process-wide override snapshot
    /// taken before this crate first loaded HSA.
    #[error(
        "queue role {role:?} conflicts with HSA_ALLOCATE_QUEUE_DEV_MEM provenance (at first HSA load: {override_at_load}, now: {override_now})"
    )]
    QueueRingProvenanceConflict {
        role: QueueRole,
        override_at_load: bool,
        override_now: bool,
    },
    /// Restricted replay unexpectedly produced a number of physical submissions other than one.
    #[error("restricted replay required one physical AQL submission, got {0}")]
    RestrictedReplaySubmissions(usize),
    /// A logical element count cannot be represented by the native u32 kernel-length ABI.
    #[error("{op} element count {elements} exceeds the native u32 kernel-length ABI")]
    ElementCountTooLarge { op: &'static str, elements: usize },
    /// Element-to-byte conversion overflowed before allocation or transfer.
    #[error("{op} byte size overflow: {elements} elements * {element_bytes} bytes")]
    ByteSizeOverflow {
        op: &'static str,
        elements: usize,
        element_bytes: usize,
    },
    /// A typed transfer was requested against a differently represented allocation (card 527: checked
    /// against the buffer's full storage contract - element kind, logical dtype and layout - not just
    /// element kind).
    #[error("{op} requires {expected} storage, but the buffer stores {actual}")]
    RepresentationMismatch {
        op: &'static str,
        expected: BufferStorage,
        actual: BufferStorage,
    },
    /// A whole-buffer transfer had the wrong logical element count.
    #[error("{op} size mismatch: buffer has {have} elements, got {got}")]
    SizeMismatch {
        op: &'static str,
        have: usize,
        got: usize,
    },
    /// A checked byte range exceeded the immutable allocation capacity.
    #[error(
        "{op} range [{byte_offset}, {byte_end}) exceeds the buffer's {byte_capacity}-byte capacity"
    )]
    RangeOutOfBounds {
        op: &'static str,
        byte_offset: usize,
        byte_end: usize,
        byte_capacity: usize,
    },
    /// Offset or length arithmetic overflowed before pointer arithmetic.
    #[error("{op} byte range overflow at element offset {element_offset} with {elements} elements")]
    RangeOverflow {
        op: &'static str,
        element_offset: usize,
        elements: usize,
    },
    /// A native narrow payload ended in a partial element.
    #[error("{op} got {bytes} bytes, which is not a whole number of {element_bytes}-byte elements")]
    InvalidByteLength {
        op: &'static str,
        bytes: usize,
        element_bytes: usize,
    },
    /// A dispatch received a kernarg allocation whose capacity does not match the kernel ABI.
    #[error(
        "{op} kernarg size mismatch: kernel requires {expected} bytes, buffer has {actual} bytes"
    )]
    KernargSizeMismatch {
        op: &'static str,
        expected: usize,
        actual: usize,
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
    HostLinkNotPcie { num_hops: u32, link_types: Vec<u32> },
    /// Card 333 PCIe receipt path: ROCr rejected device-memory queue creation (often Large BAR).
    #[error("device-memory AQL ring queue creation failed: {0}")]
    DeviceRingQueueCreate(HsaStatus),
}

/// Human-readable HSA status. `Display` prints the textual name (static table) and the raw hex code.
#[derive(Debug, Clone, Copy)]
pub struct HsaStatus {
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

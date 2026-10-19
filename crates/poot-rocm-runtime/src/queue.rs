use crate::bindings::{
    HSA_AMD_GPU_HW_EXCEPTION_EVENT, HSA_AMD_GPU_MEMORY_ERROR_EVENT, HSA_AMD_GPU_MEMORY_FAULT_EVENT,
    HSA_QUEUE_FEATURE_KERNEL_DISPATCH, HSA_QUEUE_TYPE_MULTI, HSA_STATUS_SUCCESS, hsa_amd_event_t,
    hsa_queue_t, hsa_status_t,
};
use crate::context::{DevicePoison, PoisonState};
use crate::{DeviceFault, HsaStatus, QueueHostLink, QueueRingMemory, RocmError, bindings, ffi};
use std::ffi::{OsString, c_void};
use std::sync::{Mutex, OnceLock, Weak};

/// Immutable purpose of the context's sole HSA queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueRole {
    /// The production queue: an ordinary MULTI queue whose packet ring lives in system memory.
    SystemMemoryRing,
    /// Receipt-only ordinary queue with a forced PCIe device-memory ring.
    PcieDeviceRingReceipt,
}

/// Authoritative creation route for the packet ring, independent of the HSA queue type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueRingProvenance {
    /// ROCr's ordinary queue creator with `HSA_ALLOCATE_QUEUE_DEV_MEM` unset: a system-memory ring.
    OrdinarySystemMemory,
    /// Card 333's receipt-only ordinary queue with the device-memory-ring override enabled.
    PcieDeviceMemoryReceipt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueueRequest {
    pub(crate) queue_type: u32,
    pub(crate) size: u32,
}

/// Extends `hsa_queue_t`'s opaque struct (a 0-sized struct in bindings.rs) with the fields read at
/// dispatch time; the HSA spec pins the layout (Programmer's Reference Manual v1.2):
///   offset  0: type           (u32)
///   offset  4: features       (u32)
///   offset  8: base_address   (*mut c_void)
///   offset 16: doorbell_signal (hsa_signal_t = {u64})
///   offset 24: size           (u32)
///   offset 28: reserved1       (u32)
///   offset 32: id             (u64)
#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct QueueLayout {
    pub r#type: u32,
    pub features: u32,
    pub base_address: *mut c_void,
    pub doorbell_signal: bindings::hsa_signal_t,
    pub size: u32,
    pub reserved1: u32,
    pub id: u64,
}

/// Authoritative packet-ring facts for the production queue creator. ROCr's ordinary queue creator
/// allocates the ring in system memory unless `HSA_ALLOCATE_QUEUE_DEV_MEM` is set, which
/// [`validate_queue_plan_provenance`] refuses for this plan; do not substitute device-memory or
/// coherent-device facts without changing that creator.
fn system_memory_ring_publication_facts() -> (Option<QueueRingMemory>, Option<QueueHostLink>) {
    (Some(QueueRingMemory::HostCoherent), None)
}

pub(crate) fn device_ring_override_present(get: impl Fn(&str) -> Option<OsString>) -> bool {
    get("HSA_ALLOCATE_QUEUE_DEV_MEM").is_some()
}

pub(crate) static DEVICE_RING_OVERRIDE_AT_FIRST_HSA_LOAD: OnceLock<bool> = OnceLock::new();

pub(crate) fn validate_queue_plan_provenance(
    plan: QueueCreationPlan,
    override_at_load: bool,
    override_now: bool,
) -> Result<(), RocmError> {
    let valid = match plan {
        QueueCreationPlan::SystemMemoryRing => !override_at_load && !override_now,
        QueueCreationPlan::PcieDeviceMemoryRingReceipt => override_at_load && override_now,
    };
    if valid {
        Ok(())
    } else {
        Err(RocmError::QueueRingProvenanceConflict {
            role: plan.role(),
            override_at_load,
            override_now,
        })
    }
}

/// Queue creation plan for [`RocmContext`] construction.
///
/// Production creates an ordinary MULTI queue with a system-memory ring. The PCIe device-ring variant
/// is receipt-only: it requests ROCr's `HSA_ALLOCATE_QUEUE_DEV_MEM` path and creates the same kind of
/// queue with the ring in device memory, without Card 276's MULTI admission or pooled-MoE policy.
/// Both pass an error callback, so a queue fault is recorded instead of aborting the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QueueCreationPlan {
    SystemMemoryRing,
    PcieDeviceMemoryRingReceipt,
}

impl QueueCreationPlan {
    pub(crate) const fn role(self) -> QueueRole {
        match self {
            Self::SystemMemoryRing => QueueRole::SystemMemoryRing,
            Self::PcieDeviceMemoryRingReceipt => QueueRole::PcieDeviceRingReceipt,
        }
    }

    /// The ring memory and host link this plan's queue is created with: the facts
    /// `QueuePublication::validate` turns into the publication contract. One owner for both plans, so a
    /// creation arm cannot spell its own.
    pub(crate) fn publication_facts(self) -> (Option<QueueRingMemory>, Option<QueueHostLink>) {
        match self {
            Self::SystemMemoryRing => system_memory_ring_publication_facts(),
            Self::PcieDeviceMemoryRingReceipt => (
                Some(QueueRingMemory::DeviceMemory),
                Some(QueueHostLink::Pcie),
            ),
        }
    }

    pub(crate) const fn provenance(self) -> QueueRingProvenance {
        match self {
            Self::SystemMemoryRing => QueueRingProvenance::OrdinarySystemMemory,
            Self::PcieDeviceMemoryRingReceipt => QueueRingProvenance::PcieDeviceMemoryReceipt,
        }
    }
}

pub(crate) fn queue_request(agent_max_queue_size: u32) -> QueueRequest {
    QueueRequest {
        queue_type: HSA_QUEUE_TYPE_MULTI,
        size: agent_max_queue_size,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueueProperties {
    pub(crate) queue_type: u32,
    pub(crate) features: u32,
    pub(crate) size: u32,
}

impl From<&QueueLayout> for QueueProperties {
    fn from(queue: &QueueLayout) -> Self {
        Self {
            queue_type: queue.r#type,
            features: queue.features,
            size: queue.size,
        }
    }
}

pub(crate) fn validate_created_queue(
    request: QueueRequest,
    actual: QueueProperties,
) -> Result<(), RocmError> {
    if actual.queue_type != request.queue_type {
        return Err(RocmError::QueueContract(format!(
            "requested type {}, HSA returned type {}",
            request.queue_type, actual.queue_type
        )));
    }
    if actual.features & HSA_QUEUE_FEATURE_KERNEL_DISPATCH == 0 {
        return Err(RocmError::QueueContract(format!(
            "returned features {:#x} omit kernel dispatch",
            actual.features
        )));
    }
    if actual.size == 0 || !actual.size.is_power_of_two() {
        return Err(RocmError::QueueContract(format!(
            "returned queue size {} is not a nonzero power of two",
            actual.size
        )));
    }
    Ok(())
}

pub(crate) fn take_queue_reference(queue: &mut *mut hsa_queue_t) -> Option<*mut hsa_queue_t> {
    if queue.is_null() {
        None
    } else {
        Some(std::mem::replace(queue, std::ptr::null_mut()))
    }
}

pub(crate) fn destroy_queue_reference<R>(
    queue: &mut *mut hsa_queue_t,
    destroy: impl FnOnce(*mut hsa_queue_t) -> R,
) -> Option<R> {
    take_queue_reference(queue).map(destroy)
}

/// The `hsa_queue_create` error callback: ROCr inactivates the queue after an asynchronous error and
/// calls this instead of its default handler, which prints and aborts the process. `data` is the
/// queue's [`DevicePoison::queue_callback_data`].
pub(crate) unsafe extern "C" fn queue_error_callback(
    status: hsa_status_t,
    _queue: *mut hsa_queue_t,
    data: *mut c_void,
) {
    // SAFETY: `data` is the pointer `queue_callback_data` produced for this queue; the context keeps
    // it alive for as long as the queue can still call back (see `TimeoutLeaks`).
    let state = unsafe { PoisonState::from_callback_data(data) };
    state.lose(DeviceFault::QueueError(HsaStatus { code: status }));
}

/// The contexts a system event can concern: each live context's GPU agent and its poison state.
static LIVE_CONTEXTS: Mutex<Vec<(u64, Weak<PoisonState>)>> = Mutex::new(Vec::new());

/// Make `poison` receive the faults of the GPU `agent` and ensure the process's one HSA system event
/// handler is installed. Without a handler ROCr aborts the process on a GPU memory fault.
pub(crate) fn watch_agent_faults(
    funcs: &ffi::Funcs,
    agent: bindings::hsa_agent_t,
    poison: &DevicePoison,
) -> Result<(), RocmError> {
    {
        let mut live = LIVE_CONTEXTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        live.retain(|(_, state)| state.strong_count() > 0);
        live.push((agent.handle, poison.downgrade()));
    }
    // SAFETY: `system_event_handler` matches the callback type and stays valid as long as the library
    // is loaded; the data pointer is unused.
    let status = unsafe {
        (funcs.hsa_amd_register_system_event_handler)(system_event_handler, std::ptr::null_mut())
    };
    // HSA reports a second registration as the generic `HSA_STATUS_ERROR`. This crate is the only
    // registrant and its handler serves every context, so the handler is already installed.
    if status == HSA_STATUS_SUCCESS || status == bindings::HSA_STATUS_ERROR {
        Ok(())
    } else {
        Err(RocmError::Driver(HsaStatus { code: status }))
    }
}

/// The system event handler: a GPU fault or reset loses every context on the agent it names.
unsafe extern "C" fn system_event_handler(
    event: *const hsa_amd_event_t,
    _data: *mut c_void,
) -> hsa_status_t {
    // SAFETY: HSA passes a valid event that lives for the duration of the call.
    let event = unsafe { &*event };
    // SAFETY: the union member read is the one `event_type` selects.
    let (agent, fault) = unsafe {
        match event.event_type {
            HSA_AMD_GPU_MEMORY_FAULT_EVENT => {
                let info = event.payload.memory_fault;
                (
                    info.agent.handle,
                    DeviceFault::MemoryFault {
                        address: info.virtual_address,
                        reason_mask: info.fault_reason_mask,
                    },
                )
            }
            HSA_AMD_GPU_HW_EXCEPTION_EVENT => {
                let info = event.payload.hw_exception;
                (
                    info.agent.handle,
                    DeviceFault::HardwareException {
                        reset_cause: info.reset_cause,
                    },
                )
            }
            HSA_AMD_GPU_MEMORY_ERROR_EVENT => {
                let info = event.payload.memory_error;
                (
                    info.agent.handle,
                    DeviceFault::MemoryError {
                        address: info.virtual_address,
                        reason_mask: info.error_reason_mask,
                    },
                )
            }
            _ => return HSA_STATUS_SUCCESS,
        }
    };
    let live = LIVE_CONTEXTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for (watched, state) in live.iter() {
        if *watched == agent
            && let Some(state) = state.upgrade()
        {
            state.lose(fault);
        }
    }
    HSA_STATUS_SUCCESS
}

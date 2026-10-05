use crate::*;

/// Immutable purpose of the context's sole HSA queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueRole {
    /// Runtime-owned GWS queue required by persistent cooperative kernels.
    Cooperative,
    /// Ordinary user queue reserved for retained AQL graph replay.
    RecordedReplay,
    /// Receipt-only ordinary queue with a forced PCIe device-memory ring.
    PcieDeviceRingReceipt,
}

/// Authoritative creation route for the packet ring, independent of the HSA queue type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueRingProvenance {
    /// ROCr's cooperative GWS creator with its explicit system-memory-ring flag.
    CooperativeGwsSystemMemory,
    /// Ordinary `hsa_queue_create` with the device-memory-ring override absent.
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

/// Authoritative packet-ring facts for the current cooperative GWS queue creator. ROCr's
/// cooperative path allocates the ring with `HSA_AMD_QUEUE_CREATE_SYSTEM_MEM`; do not substitute
/// device-memory or coherent-device facts without changing that creator.
pub(crate) fn cooperative_gws_publication_facts() -> (Option<QueueRingMemory>, Option<QueueHostLink>)
{
    (Some(QueueRingMemory::HostCoherent), None)
}

/// ROCr's ordinary queue creator defaults to a system-memory ring when the process-wide
/// `HSA_ALLOCATE_QUEUE_DEV_MEM` override was absent before the runtime was loaded.
#[cfg(test)]
pub(crate) fn ordinary_system_publication_facts() -> (Option<QueueRingMemory>, Option<QueueHostLink>)
{
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
        QueueCreationPlan::CooperativeGws => true,
        #[cfg(test)]
        QueueCreationPlan::RecordedReplay => !override_at_load && !override_now,
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
/// Production keeps the cooperative GWS / system-memory ring. The PCIe device-ring variant is
/// receipt-only: it requests ROCr's `HSA_ALLOCATE_QUEUE_DEV_MEM` path and creates an ordinary MULTI
/// queue, without Card 276's MULTI admission or pooled-MoE policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QueueCreationPlan {
    CooperativeGws,
    #[cfg(test)]
    RecordedReplay,
    PcieDeviceMemoryRingReceipt,
}

impl QueueCreationPlan {
    pub(crate) const fn role(self) -> QueueRole {
        match self {
            Self::CooperativeGws => QueueRole::Cooperative,
            #[cfg(test)]
            Self::RecordedReplay => QueueRole::RecordedReplay,
            Self::PcieDeviceMemoryRingReceipt => QueueRole::PcieDeviceRingReceipt,
        }
    }

    pub(crate) const fn provenance(self) -> QueueRingProvenance {
        match self {
            Self::CooperativeGws => QueueRingProvenance::CooperativeGwsSystemMemory,
            #[cfg(test)]
            Self::RecordedReplay => QueueRingProvenance::OrdinarySystemMemory,
            Self::PcieDeviceMemoryRingReceipt => QueueRingProvenance::PcieDeviceMemoryReceipt,
        }
    }
}

pub(crate) fn queue_request(plan: QueueCreationPlan, agent_max_queue_size: u32) -> QueueRequest {
    QueueRequest {
        queue_type: match plan {
            QueueCreationPlan::CooperativeGws => HSA_QUEUE_TYPE_COOPERATIVE,
            #[cfg(test)]
            QueueCreationPlan::RecordedReplay => HSA_QUEUE_TYPE_MULTI,
            QueueCreationPlan::PcieDeviceMemoryRingReceipt => HSA_QUEUE_TYPE_MULTI,
        },
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

//! Accounting: checked per-device byte categories and independent capacity admission (FR-007/FR-008/
//! FR-009).
//!
//! Every device passes its own checked total independently. Aggregate capacity across the topology is
//! never consulted: a deficient device in any position, including a non-endpoint one, rejects on its own.

use std::collections::HashMap;

use crate::multi_device::topology::{DeviceId, DeviceSpec};

/// Checked byte categories for one device (FR-007). Each category is a distinct accounting bucket; a
/// later exact-model child fills these in without Card 360 knowing what a "constant" or "workspace" byte
/// is for, only that the categories add up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByteCategories {
    /// Bytes the caller accounts to the source (or the carrier it is held in) on this device.
    pub source_or_carrier: u64,
    /// Bytes the caller accounts to padding that only the transport needs.
    pub transport_padding: u64,
    /// Bytes the caller accounts to dense constants resident on this device.
    pub dense_constants: u64,
    /// Bytes the caller accounts to persistent state resident on this device.
    pub state: u64,
    /// Bytes the caller accounts to activations on this device.
    pub activations: u64,
    /// Bytes the caller accounts to kernel workspace on this device.
    pub workspace: u64,
    /// Bytes the caller accounts to communication buffers on this device.
    pub communication: u64,
    /// Bytes the caller accounts to program outputs held on this device.
    pub outputs: u64,
    /// Bytes the caller holds back on purpose, outside every other category.
    pub reserve: u64,
}

impl ByteCategories {
    /// The checked sum of every category, or `None` on overflow.
    pub fn checked_total(&self) -> Option<u64> {
        [
            self.source_or_carrier,
            self.transport_padding,
            self.dense_constants,
            self.state,
            self.activations,
            self.workspace,
            self.communication,
            self.outputs,
            self.reserve,
        ]
        .into_iter()
        .try_fold(0u64, |acc, category| acc.checked_add(category))
    }
}

/// Why a device's byte categories were not admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AccountingError {
    /// A device's categories sum past `u64::MAX`.
    #[error("device {device:?} byte categories overflowed u64")]
    Overflow {
        /// The device whose categories overflowed.
        device: DeviceId,
    },
    /// A device's checked total is more than its usable capacity.
    #[error("device {device:?} needs {required} bytes but has {usable} usable")]
    CapacityExceeded {
        /// The device that is over capacity.
        device: DeviceId,
        /// The device's checked total across all categories, in bytes.
        required: u64,
        /// The device's `usable_bytes`.
        usable: u64,
    },
}

/// Admit every device independently against its own checked total (FR-008/FR-009). A device with no
/// entry in `usage` accounts zero bytes. This loop never sums across devices: each check compares one
/// device's total to that device's `usable_bytes`, so a deficient device in a non-endpoint position
/// rejects on its own.
pub fn admit_per_device_capacity(
    devices: &[DeviceSpec],
    usage: &HashMap<DeviceId, ByteCategories>,
) -> Result<(), AccountingError> {
    for device in devices {
        let categories = usage.get(&device.id).copied().unwrap_or_default();
        let total = categories
            .checked_total()
            .ok_or(AccountingError::Overflow { device: device.id })?;
        if total > device.usable_bytes {
            return Err(AccountingError::CapacityExceeded {
                device: device.id,
                required: total,
                usable: device.usable_bytes,
            });
        }
    }
    Ok(())
}

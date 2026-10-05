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
    pub source_or_carrier: u64,
    pub transport_padding: u64,
    pub dense_constants: u64,
    pub state: u64,
    pub activations: u64,
    pub workspace: u64,
    pub communication: u64,
    pub outputs: u64,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AccountingError {
    #[error("device {device:?} byte categories overflowed u64")]
    Overflow { device: DeviceId },
    #[error("device {device:?} needs {required} bytes but has {usable} usable")]
    CapacityExceeded {
        device: DeviceId,
        required: u64,
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

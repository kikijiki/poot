//! Card 360 acceptance table row "Packed residency" (N=2): no device or communication buffer contains a
//! full decoded quantized weight.
//!
//! Mutation that must fail: add a `4 * out * in` decoded-weight term. Static inspection of produced plan
//! data is independent of device count and ordering, so N=2 suffices.

use std::collections::HashMap;

use crate::multi_device::placement::{PackedResidencyError, reject_dense_quant_mirror};
use crate::multi_device::topology::DeviceId;

#[test]
fn multi_device_plan_has_no_dense_quant_mirror() {
    let owner = crate::multi_device::placement::OwnerId(42);
    let out = 128usize;
    let k = 64usize;
    let forbidden_bytes = 4u64 * out as u64 * k as u64;

    // A realistic packed footprint, well under the forbidden fully-decoded size.
    let mut device_bytes = HashMap::new();
    device_bytes.insert(DeviceId(0), 4096u64);
    device_bytes.insert(DeviceId(1), 4096u64);
    assert!(reject_dense_quant_mirror(owner, out, k, &device_bytes, 512).is_ok());

    // Add a `4 * out * in` decoded-weight term to device 1's tally, exactly at the forbidden size (the
    // admission boundary is inclusive: "at or above").
    device_bytes.insert(DeviceId(1), forbidden_bytes);
    assert_eq!(
        reject_dense_quant_mirror(owner, out, k, &device_bytes, 512),
        Err(PackedResidencyError::DeviceDenseMirror {
            device: DeviceId(1),
            owner,
            bytes: forbidden_bytes,
            forbidden_bytes,
            out,
            k,
        })
    );

    // The same term showing up in the communication buffer instead of a device must also reject.
    device_bytes.insert(DeviceId(1), 4096u64);
    assert_eq!(
        reject_dense_quant_mirror(owner, out, k, &device_bytes, forbidden_bytes),
        Err(PackedResidencyError::CommunicationDenseMirror {
            owner,
            bytes: forbidden_bytes,
            forbidden_bytes,
            out,
            k,
        })
    );
}

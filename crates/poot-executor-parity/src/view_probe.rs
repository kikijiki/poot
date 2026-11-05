//! POOT-1450: an untyped arena slot reaches a kernel only through a typed view its bytes admit, the same probe
//! on every backend. The engine binds every arena value through `Device::view`; this drives that production
//! method and `Device::dispatch` on a real device with the two sides of the contract differing:
//!
//! - the f32 view of a 256-byte slot holds 64 elements and runs the length probe (`len_probe_kernel`, one f32
//!   input, one f32 output);
//! - the slot itself, or a bf16 view of it, bound to the probe's f32 parameter is refused by the runtime's
//!   element-schema check, before the kernel runs;
//! - a typed f32 buffer viewed as bf16 is refused by `Device::view` (R484-001 for typed buffers);
//! - a 6-byte slot viewed as f32 is refused by `Device::view` (not whole elements), on a device that holds the six
//!   bytes it was asked for; one that allocates whole words (raw Vulkan) holds eight, and admits two f32s.

use poot_executor::{Arg, Device, Dispatch, LaunchKind};
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;

/// Run the probe on `device`. `len_probe` is `poot_kernel_ir::test_support::len_probe_kernel()` loaded on it.
///
/// # Panics
///
/// When any refusal above is missing or any admitted view misbehaves.
pub fn an_arena_slot_is_bound_only_through_a_view_it_admits<D: Device>(
    device: &mut D,
    len_probe: &D::Kernel,
) {
    let out = device
        .allocate(BufferRole::Output, BufferStorage::f32(), 1)
        .unwrap();
    let slot = device
        .allocate(BufferRole::Activation, BufferStorage::untyped(), 256)
        .unwrap();
    let f32_view = device.view(&slot, BufferStorage::f32()).unwrap();
    assert_eq!(device.buffer_elems(&f32_view), 64, "256 bytes are 64 f32");
    let bf16_view = device.view(&slot, BufferStorage::bf16()).unwrap();
    assert_eq!(
        device.buffer_elems(&bf16_view),
        128,
        "256 bytes are 128 bf16"
    );

    let probe = |device: &mut D, input: &D::Buffer| -> Result<(), String> {
        let ran = (|| {
            device.begin()?;
            device.dispatch(Dispatch {
                kernel: len_probe,
                inputs: &[Arg {
                    buffer: input,
                    elems: 4,
                }],
                output: Arg {
                    buffer: &out,
                    elems: 1,
                },
                launch: LaunchKind::Grid {
                    threads: [1, 1, 1],
                    workgroup: [1, 1, 1],
                },
                work: 1,
            })?;
            let recording = device.finish()?;
            device.replay(&recording)?;
            device.synchronize()
        })();
        ran.map_err(|error| {
            device.abort().expect("a refused dispatch aborts cleanly");
            error.to_string()
        })
    };
    probe(device, &f32_view).expect("the f32 view runs");
    let mut bytes = [0u8; 4];
    device.read(&out, &mut bytes).unwrap();
    assert_eq!(
        f32::from_le_bytes(bytes),
        4.0,
        "the probe reads its own length"
    );
    for (what, input) in [("the slot itself", &slot), ("its bf16 view", &bf16_view)] {
        let refused = probe(device, input).expect_err(what);
        assert!(
            refused.contains("element-kind mismatch"),
            "{what} is refused by the schema check: {refused}"
        );
    }

    let typed = device
        .allocate(BufferRole::Activation, BufferStorage::f32(), 64)
        .unwrap();
    let refused = device
        .view(&typed, BufferStorage::bf16())
        .err()
        .expect("a typed f32 buffer has no bf16 view")
        .to_string();
    assert!(refused.contains("cannot be viewed as"), "{refused}");
    // Six bytes asked; a device that allocates whole words (raw Vulkan) holds eight, which are whole f32s.
    let short = device
        .allocate(BufferRole::Activation, BufferStorage::untyped(), 6)
        .unwrap();
    let held = device.buffer_elems(&short);
    let viewed = device.view(&short, BufferStorage::f32());
    if held.is_multiple_of(4) {
        let view = viewed.unwrap_or_else(|e| panic!("{held} bytes are whole f32s: {e}"));
        assert_eq!(device.buffer_elems(&view), held / 4);
    } else {
        let refused = viewed
            .err()
            .unwrap_or_else(|| panic!("{held} bytes are not whole f32 elements"))
            .to_string();
        assert!(refused.contains("is not whole"), "{refused}");
    }
}

//! POOT-1566 W1: a dispatch binds its argument at [`Arg::offset`] bytes into its buffer, and a device refuses an
//! offset off its binding alignment before it records anything, on whichever real `Device` the caller opens.
//!
//! The probe dispatches `poot_kernel_ir::test_support::float_ne_kernel` (`out[i] = (a[i] != b[i]) as f32`): `a` is a
//! buffer of distinct words bound 256 bytes in, `b` holds those same words from the start, so every lane is 0.0
//! exactly when the device read `a` from its offset (read from the base, it compares word `i` with word `i + 64`).
//! The caller compiles and loads the kernel for its own backend.

use poot_executor::{Arg, Device, Dispatch, LaunchKind};
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;

/// Lanes compared; the window starts this many words (256 bytes) into `a`.
const LANES: usize = 64;

fn words(range: std::ops::Range<usize>) -> Vec<u8> {
    range
        .flat_map(|word| (word as f32 + 1.0).to_le_bytes())
        .collect()
}

fn upload<D: Device>(device: &mut D, bytes: &[u8]) -> D::Buffer {
    let buffer = device
        .allocate(BufferRole::Input, BufferStorage::f32(), bytes.len() / 4)
        .expect("allocate input");
    poot_executor::write_slice(device, &buffer, 0, bytes).expect("write input");
    buffer
}

fn dispatch<D: Device>(
    device: &mut D,
    kernel: &D::Kernel,
    a_offset: u64,
    a: &D::Buffer,
    b: &D::Buffer,
    out: &D::Buffer,
) -> Result<(), D::Error> {
    device.begin()?;
    device.dispatch(Dispatch {
        kernel,
        inputs: &[
            Arg {
                buffer: a,
                offset: a_offset,
                elems: LANES,
            },
            Arg {
                buffer: b,
                offset: 0,
                elems: LANES,
            },
        ],
        output: Arg {
            buffer: out,
            offset: 0,
            elems: LANES,
        },
        launch: LaunchKind::Grid {
            threads: [LANES as u32, 1, 1],
            workgroup: [LANES as u32, 1, 1],
        },
        work: LANES as u64,
    })
}

/// SC-e: argument `a` is bound 256 bytes into a buffer of `2 * LANES` distinct words; every lane compares equal to
/// `b`, which holds the words from 256 bytes on.
pub fn an_offset_binding_reads_from_its_offset<D: Device>(
    backend: &str,
    device: &mut D,
    kernel: &D::Kernel,
) {
    let a = upload(device, &words(0..2 * LANES));
    let b = upload(device, &words(LANES..2 * LANES));
    let out = device
        .allocate(BufferRole::Output, BufferStorage::f32(), LANES)
        .expect("allocate output");
    dispatch(device, kernel, 256, &a, &b, &out).expect("dispatch at an aligned offset");
    let recording = device.finish().expect("finish");
    device.replay(&recording).expect("replay");
    device.synchronize().expect("synchronize");
    let mut raw = vec![0u8; LANES * 4];
    device.read(&out, &mut raw).expect("read");
    let got: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    assert_eq!(
        got,
        vec![0.0; LANES],
        "{backend}: an argument bound 256 bytes in must read the words from 256 on"
    );
}

/// SC-f: the same dispatch with `a` bound 2 bytes in (off every device's binding alignment, and off the element
/// width) is refused by the device, returned for the caller to match against its typed variant, before anything is
/// recorded.
pub fn a_misaligned_offset_is_refused<D: Device>(device: &mut D, kernel: &D::Kernel) -> D::Error {
    let a = upload(device, &words(0..2 * LANES));
    let b = upload(device, &words(LANES..2 * LANES));
    let out = device
        .allocate(BufferRole::Output, BufferStorage::f32(), LANES)
        .expect("allocate output");
    dispatch(device, kernel, 2, &a, &b, &out)
        .expect_err("a binding off the device's alignment must be refused")
}

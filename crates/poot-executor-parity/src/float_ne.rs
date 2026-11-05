//! POOT-1460: float `Ne` is IEEE on every backend. The kernel interpreter says `NaN != x` is true for every
//! `x`, `NaN` included; codegen once lowered it to an ordered compare (`fcmp one`, `FOrdNotEqual`), false for a
//! `NaN`. The probe dispatches `poot_kernel_ir::test_support::float_ne_kernel` (`out[i] = (a[i] != b[i]) as
//! f32`) over lanes that pair a `NaN` with a number, with a `NaN` and with itself, and reads the answer back
//! on the device the caller opened. The caller compiles and loads the kernel for its own backend; the lanes
//! and the expected answer live here so every backend asserts the same table.

use poot_executor::{Arg, Device, Dispatch, LaunchKind};
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;

/// A quiet `NaN` with a payload the other `NaN` lacks: the compare is by value, not by bit pattern.
const OTHER_NAN: f32 = f32::from_bits(0x7fc0_1234);

/// `(a, b, a != b)` per lane.
const LANES: [(f32, f32, bool); 8] = [
    (f32::NAN, 1.0, true),
    (1.0, f32::NAN, true),
    (f32::NAN, f32::NAN, true),
    (f32::NAN, OTHER_NAN, true),
    (1.0, 1.0, false),
    (1.0, 2.0, true),
    (0.0, -0.0, false),
    (f32::INFINITY, f32::INFINITY, false),
];

fn bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

/// Run the probe on `device` with `kernel` (the float-`Ne` kernel compiled for this backend) and assert every
/// lane against IEEE.
pub fn check<D: Device>(backend: &str, device: &mut D, kernel: &D::Kernel) {
    let n = LANES.len();
    let upload = |device: &mut D, values: Vec<u8>| {
        let buffer = device
            .allocate(BufferRole::Input, BufferStorage::f32(), n)
            .expect("allocate input");
        poot_executor::write_slice(device, &buffer, 0, &values).expect("write input");
        buffer
    };
    let a = upload(device, bytes(LANES.iter().map(|lane| lane.0)));
    let b = upload(device, bytes(LANES.iter().map(|lane| lane.1)));
    let out = device
        .allocate(BufferRole::Output, BufferStorage::f32(), n)
        .expect("allocate output");

    device.begin().expect("begin");
    device
        .dispatch(Dispatch {
            kernel,
            inputs: &[
                Arg {
                    buffer: &a,
                    elems: n,
                },
                Arg {
                    buffer: &b,
                    elems: n,
                },
            ],
            output: Arg {
                buffer: &out,
                elems: n,
            },
            launch: LaunchKind::Grid {
                threads: [n as u32, 1, 1],
                workgroup: [n as u32, 1, 1],
            },
            work: n as u64,
        })
        .expect("dispatch");
    let recording = device.finish().expect("finish");
    device.replay(&recording).expect("replay");
    device.synchronize().expect("synchronize");

    let mut raw = vec![0u8; n * 4];
    device.read(&out, &mut raw).expect("read");
    let got: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    let want: Vec<f32> = LANES
        .iter()
        .map(|lane| if lane.2 { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(
        got, want,
        "{backend}: float `Ne` per lane {LANES:?} must be IEEE (NaN != anything is true)"
    );
}

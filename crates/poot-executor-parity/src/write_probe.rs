//! POOT-1100 LP-3: the chunk contract of `Device::write_with`, the same probe on every backend. A buffer
//! larger than two staging chunks with an odd tail is written twice through the production
//! `Device::write_with` (once whole, once from an offset that crosses every chunk boundary
//! unaligned), each byte produced from its destination address by a function the probe computes
//! independently, then read back and compared byte for byte. The `fill` callback also checks the
//! contract it is handed: consecutive, ascending, non-overlapping ranges covering exactly
//! `offset..offset + len`, `out.len() == range.len()`.

use std::ops::Range;

use poot_executor::Device;
use poot_runtime_common::BufferRole;
use poot_target::BufferStorage;

/// The byte every correct write puts at device byte `address` for write `seed`.
fn expected_byte(seed: u64, address: usize) -> u8 {
    let x = (address as u64 ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_mul(0xD6E8_FEB8_6659_FD93);
    (x >> 29) as u8
}

/// Write `len` bytes at `offset` of `buffer` through `write_with` and return the ranges `fill` saw.
fn write_pattern<D: Device>(
    device: &mut D,
    buffer: &D::Buffer,
    offset: usize,
    len: usize,
    seed: u64,
) -> Vec<Range<usize>> {
    let mut seen: Vec<Range<usize>> = Vec::new();
    device
        .write_with(buffer, offset, len, &mut |range, out| {
            let expected_start = seen.last().map_or(offset, |last| last.end);
            assert_eq!(
                range.start, expected_start,
                "fill ranges are consecutive and ascending"
            );
            assert_eq!(out.len(), range.len(), "out is exactly the range");
            for (i, byte) in out.iter_mut().enumerate() {
                *byte = expected_byte(seed, range.start + i);
            }
            seen.push(range);
        })
        .expect("write_with");
    assert_eq!(seen.first().map(|r| r.start), Some(offset));
    assert_eq!(seen.last().map(|r| r.end), Some(offset + len));
    seen
}

/// Run the probe on `device`, whose staging chunk is `chunk` bytes (the backend's own constant, read by
/// the caller from the production crate). `min_ranges` is how many `fill` calls the whole write must make:
/// 3 for a backend that stages in chunks, 1 for one that writes in place.
pub fn chunked_write_is_exact<D: Device>(device: &mut D, chunk: usize, min_ranges: usize) {
    // Two chunks plus an odd tail, in whole words: the last chunk is neither empty nor full.
    let words = 2 * (chunk / 4) + 1027;
    let bytes = words * 4;
    let buffer = device
        .allocate(BufferRole::Weight, BufferStorage::i32(), words)
        .expect("allocate");
    let mut expected = vec![0u8; bytes];

    let first = write_pattern(device, &buffer, 0, bytes, 1);
    assert!(
        first.len() >= min_ranges,
        "{} ranges for {bytes} bytes",
        first.len()
    );
    for (address, byte) in expected.iter_mut().enumerate() {
        *byte = expected_byte(1, address);
    }

    // From byte 8, ending four bytes short: every chunk boundary falls off the buffer's own alignment.
    let (offset, len) = (8, bytes - 12);
    write_pattern(device, &buffer, offset, len, 2);
    for (address, byte) in expected.iter_mut().enumerate().skip(offset).take(len) {
        *byte = expected_byte(2, address);
    }

    device.synchronize().expect("synchronize");
    let mut read = vec![0u8; bytes];
    device.read(&buffer, &mut read).expect("read");
    let first_diff = read.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(
        first_diff, None,
        "the buffer holds exactly the bytes written, first difference at byte {first_diff:?}"
    );
}

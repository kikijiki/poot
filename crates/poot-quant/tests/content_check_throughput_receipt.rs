//! Throughput receipt of the packed-weight content check (POOT-1097): `PackedPayload::try_new`
//! over a 4.4 GB synthetic Q4_K buffer, reported in GB/s. Ignored: a measurement, never a CI gate
//! (no wall-clock assertion). Run it in release:
//! `cargo test --release -p poot-quant --test content_check_throughput_receipt -- --ignored --nocapture`

use std::time::Instant;

use poot_quant::format::WeightFormat;
use poot_quant::{PackedPayload, PackedWeight, SourceRole, StoredBytes};

const ROWS: usize = 4096;
/// 7458 Q4_K super-blocks of 256 values per row: 4096 x 7458 x 144 B = 4.40 GB.
const K: usize = 256 * 7458;
const BLOCK_BYTES: usize = 144;

#[test]
#[ignore = "throughput receipt, run by hand in release"]
fn content_check_throughput_receipt() {
    let weight = PackedWeight::try_new(WeightFormat::Q4_K, [ROWS, K]).unwrap();
    let len = weight.source_bytes(SourceRole::Blocks);
    let mut bytes = vec![0x11u8; len];
    for block in bytes.as_chunks_mut::<BLOCK_BYTES>().0.iter_mut() {
        // d = 1.0, dmin = 0.5 as f16: finite, so the whole buffer is scanned.
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
    }
    let bytes = StoredBytes::from(bytes);
    let start = Instant::now();
    let payload = PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes)]).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "RECEIPT {len} bytes validated in {seconds:.3} s = {:.3} GB/s",
        len as f64 / seconds / 1e9
    );
    drop(payload);
}

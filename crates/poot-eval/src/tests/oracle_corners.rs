//! Literal-expectation pins for the CPU oracle's own corners (R486-003, EV-1/EV-2): the places a
//! matching executor must reproduce bit for bit, written as raw bit patterns and hand-computed values
//! rather than recomputed with the production function under test.

use poot_quant::scalar::f32_to_bf16;

/// EV-1: `f32_to_bf16` rounds a tie to even. `0x3F81_8000` is exactly halfway between the bf16 values
/// `0x3F81_0000` and `0x3F82_0000`; the retained low bit is 1 (odd), so round-to-nearest-even carries
/// up to `0x3F82_0000`. Dropping the tie-break term (`0x7FFF + lsb` -> `0x7FFF`) rounds it down and the
/// assertion fails.
#[test]
fn round_bf16_tie_rounds_to_even() {
    let odd_tie = f32::from_bits(0x3F81_8000);
    assert_eq!(
        f32_to_bf16(odd_tie),
        0x3F82,
        "a tie with an odd retained bit must round up to even, got {:#06x}",
        f32_to_bf16(odd_tie)
    );

    // The matching even tie stays put under both rules; the pair together pins `ties to even`, not
    // `ties away`.
    let even_tie = f32::from_bits(0x3F80_8000);
    assert_eq!(
        f32_to_bf16(even_tie),
        0x3F80,
        "a tie with an even retained bit must round down, got {:#06x}",
        f32_to_bf16(even_tie)
    );
}

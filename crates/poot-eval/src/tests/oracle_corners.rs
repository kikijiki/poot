//! Literal-expectation pins for the CPU oracle's own corners (R486-003, EV-1/EV-2/EV-8): the places a
//! matching executor must reproduce bit for bit, written as raw bit patterns and hand-computed values
//! rather than recomputed with the production function under test.

use crate::ops::elementwise::unary_f32;
use poot_graph_ir::op::UnOp;
use poot_runtime_common::f32_to_bf16;
use poot_tensor::HostTensor;

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

/// EV-8: `UnOp::Round` rounds half away from zero (`f32::round`), not half to even. `round_ties_even`
/// would give `0, 2, -0`; the oracle pins `1, 3, -1`.
#[test]
fn unary_round_ties_away_from_zero() {
    let x = HostTensor::f32(vec![4], vec![0.5, 2.5, -0.5, -2.5]);
    let out = unary_f32(UnOp::Round, &x).unwrap();
    let expected = [1.0f32, 3.0, -1.0, -3.0];
    for (index, (&got, &want)) in out
        .as_f32()
        .unwrap()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert_eq!(
            got, want,
            "Round[{index}] must be half-away-from-zero: got {got}, want {want}"
        );
    }
}

//! SC-003 (Card 721): a BF16 `HostTensor` bound to the oracle and to wgpu gives equal words, and
//! binding never widens it to f32: the only `to_f32` read in the whole run is the one the graph's own
//! `Cast(BF16 -> F32)` equation makes in the oracle.

use super::*;

/// Literal BF16 words: `-0.0`, +inf, the smallest normal, a negative and a positive fraction. (A NaN
/// payload or a subnormal would test the device's NaN/denormal policy, not the carrier.)
const WORDS: [u16; 6] = [0x3f80, 0x8000, 0x7f80, 0xc02e, 0x0080, 0x4049];

/// The read counter is process-wide: this row is red-capable only under nextest (one process per test).
/// Mutation (recorded): in `walk::bind_inputs`, widen a bound `Value::Host` BF16 tensor through
/// `to_f32` before storing it. The read counter then moves by two instead of one and the first
/// assertion below goes red.
#[test]
fn bf16_host_tensor_binds_to_the_oracle_and_wgpu_without_widening() {
    let _gpu_guard = gpu_lock();
    let Some((mut gpu, target)) = open_engine_or_skip() else {
        return;
    };

    let b = Builder::new();
    let x = b.constant("x", TensorType::new(vec![3, 2], DType::BF16));
    let widened = b.cast(x, DType::F32);
    let g = b.finish(widened);
    let input = HostTensor::bf16(vec![3, 2], WORDS.to_vec());

    let reads_before = poot_tensor::read_counter::reads();
    let oracle = eval(
        &g,
        &HashMap::from([(x.id, Value::from(input.clone()))]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .unwrap();
    let got = run_via_contract(&mut gpu, target, &g, &HashMap::from([(x.id, input)]));

    assert_eq!(
        poot_tensor::read_counter::reads() - reads_before,
        1,
        "the oracle's Cast equation widens once; binding to the oracle and to wgpu must not widen"
    );
    let bits = |t: &HostTensor| -> Vec<u32> {
        t.as_f32()
            .expect("a widened BF16 is F32")
            .iter()
            .map(|v| v.to_bits())
            .collect()
    };
    let expected: Vec<u32> = WORDS.iter().map(|&w| u32::from(w) << 16).collect();
    assert_eq!(bits(&oracle), expected, "oracle words");
    assert_eq!(bits(&got), expected, "wgpu words");
}

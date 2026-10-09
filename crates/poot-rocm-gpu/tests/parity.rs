//! SC-002/SC-003: the parity table (`poot-executor-parity`, Card 546a) runs on ROCm through the
//! executor contract, and every replayed step (the record-then-replay step and every later
//! pure-replay step, including a repeat of the first step's inputs) matches the CPU oracle within
//! tolerance (ADR-0101 tier 1). Mutation (SC-002): drop one plan-kind arm from `RocmDevice`
//! (e.g. make `copy` a no-op); the fixture needing it goes red. Mutation (SC-003): skip recording in
//! `Engine::step` (always treat the entry as needing a fresh recording); the repeated first step's
//! replay stops matching the oracle it already matched once, since the two recordings would not be
//! identical.

#[path = "parity/dense_gemv.rs"]
mod dense_gemv;

use poot_executor::{Device, Engine, Executor};
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::weight_map_oracle::oracle_for;

#[test]
fn parity_table_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let fixtures = [
        poot_executor_parity::qwen2_decode_fixture(8),
        poot_executor_parity::qwen2_prefill_fixture(3, 8),
    ];
    for fixture in &fixtures {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Card 557 SC-004 (dkernel K1): a qwen2-class decode and prefill traced with
/// `attn_logit_softcap = Some(50.0)` and compiled for ROCm match `eval(g)` of that same softcapped
/// trace within the parity tolerance, at every replayed step. The fixtures' scores reach the softcap
/// regime: the softcapped oracle differs from the uncapped trace's by far more than that tolerance, so
/// the device output cannot match it without applying the softcap. Reproduction: the pre-card flash
/// tracer arms (`trace_*_flash`) dropped the softcap. Mutation: make the qwen2 tracers call attention
/// without `cfg.attn_logit_softcap`; the softcapped and uncapped traces coincide and this row goes red.
#[test]
fn softcapped_qwen2_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let softcap = Some(poot_executor_parity::ATTN_LOGIT_SOFTCAP);
    let pairs = [
        (
            poot_executor_parity::qwen2_softcap_decode_fixture(8, softcap),
            poot_executor_parity::qwen2_softcap_decode_fixture(8, None),
        ),
        (
            poot_executor_parity::qwen2_softcap_prefill_fixture(3, 8, softcap),
            poot_executor_parity::qwen2_softcap_prefill_fixture(3, 8, None),
        ),
    ];
    for (softcapped, uncapped) in &pairs {
        let mut oracle = oracle_for(&softcapped.graph, &softcapped.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, softcapped, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
        // The softcap must be visible: the softcapped trace's oracle differs from the uncapped
        // trace's by more than ten times the parity tolerance at some step.
        let (mut capped_oracle, mut plain_oracle) = (
            oracle_for(&softcapped.graph, &softcapped.store),
            oracle_for(&uncapped.graph, &uncapped.store),
        );
        let largest = softcapped
            .steps
            .iter()
            .zip(&uncapped.steps)
            .map(|(capped_step, plain_step)| {
                let capped = capped_oracle(capped_step);
                let plain = plain_oracle(plain_step);
                capped
                    .as_f32()
                    .expect("F32 oracle output")
                    .iter()
                    .zip(plain.as_f32().expect("F32 oracle output"))
                    .map(|(c, p)| (c - p).abs() / p.abs().max(1.0))
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        assert!(
            largest > 5e-2,
            "{}: the softcap barely changes the oracle (max relative difference {largest:.2e})",
            softcapped.name
        );
    }
}

/// Card 645 SC-001: `matmul(x, transpose(w))` over an F32 `[N, K]` weight, at `M = 1` (the decode GEMV), `4` and `33`
/// (the tiled GEMM), compiled for ROCm matches `eval(g)` of the traced graph, within the parity tolerance
/// (ADR-0101 tier 2), at every replayed step. The oracle evaluates the `Transpose` + `MatMul` the tracer
/// emitted; the device runs the `DenseContraction` the fold made of it, reading the weight as stored.
/// Mutation: in the F32 GEMV body read B as `[K, N]` (`gemv_lds`, `WeightLayout::Nk` arm); the `M = 1` row goes red.
#[test]
fn checkpoint_orientation_projection_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::checkpoint_projection_fixtures() {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Card 736: the ALiBi attention BLOOM and MPT trace (`alibi_mask_from_pos` over a
/// `ComputedConst::AlibiSlopes` of 6 heads, decode at three positions and a 4-token prefill), compiled for
/// ROCm, matches `eval(g)` at every replayed step (ADR-0101 tier 1) AND an f64 reference that writes the
/// published slopes out as literals (tier 2), so a slope formula both the oracle and the device read from
/// the same constant cannot hide. Mutation: in `ComputedConst::tensor` return zero slopes (or drop the
/// odd-index series of the second half); the reference comparison goes red.
#[test]
fn alibi_slopes_attention_matches_the_oracle_and_the_reference_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::alibi::alibi_slopes_fixtures() {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
        let outputs = poot_executor_parity::run_outputs(exec.as_mut(), t, fixture)
            .unwrap_or_else(|e| panic!("{e}"));
        poot_executor_parity::alibi::assert_matches_alibi_reference(fixture, &outputs);
    }
}

/// Card 1007 acceptance row 1: `matmul(x, transpose(w))` over an F16 `[N, K]` const, at `M = 1` (the decode GEMV),
/// `4` and `33` (the tiled GEMM), compiled for ROCm: the program
/// holds one `DenseContraction` and no `Transpose`, and every replayed step matches `eval(g)` of the traced graph
/// within the parity tolerance (ADR-0101 tier 2). The weight lives on the device as packed binary16 words.
/// Mutation: remove `DType::F16` from `DENSE_CONTRACTION_WEIGHT_DTYPES`; a `Transpose` is planned and this row goes
/// red.
#[test]
fn f16_checkpoint_orientation_projection_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::checkpoint_projection_f16_fixtures() {
        poot_executor_parity::assert_one_dense_contraction(fixture, t);
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Card 1024: `matmul(x, transpose(w))` over a BF16 `[N, K]` const, at `M = 1` (the decode GEMV), `4` and `33`
/// (the generated tiled GEMM), compiled for rocm: the program
/// holds one `DenseContraction` and no `Transpose`, and every replayed step matches `eval(g)` of the traced graph
/// within the parity tolerance (ADR-0101 tier 2). The weight lives on the device as packed bfloat16 words.
/// Mutation: in `tiled_region`'s `WeightLayout::Nk` arm index the weight as `[K, N]`; the `M = 4` and `M = 33`
/// rows go red.
#[test]
fn bf16_checkpoint_orientation_projection_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::checkpoint_projection_bf16_fixtures() {
        poot_executor_parity::assert_one_dense_contraction(fixture, t);
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// POOT-1048: a grouped projection `matmul(x, transpose(reshape(w, [G, N, K]), [0, 2, 1]))` over a BF16 `[G * N, K]`
/// const (DeepSeek V4's `wo_a`), at `M = 1`, `5` and `33` with three groups and at one group, compiled for rocm: the
/// program holds one `GroupedDenseContraction` and neither a `Transpose` nor a `Cast`, and every replayed step
/// matches `eval(g)` of the traced graph within the parity tolerance (ADR-0101 tier 2). The weight lives on the
/// device as packed bfloat16 words.
/// Mutation: in `matmul_batched_impl` give the weight no per-group batch offset (zero its `b_beff`); the three-group
/// rows go red.
#[test]
fn bf16_grouped_projection_matches_the_oracle_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::grouped_projection_bf16_fixtures() {
        poot_executor_parity::assert_one_grouped_dense_contraction(fixture, t);
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// POOT-1033: `Select` is a bitwise select. An F32 select (standalone, and inside a fused float region) over
/// poisoned branches and an I32 select over nonzero condition words each equal the hand-written select words
/// and the oracle's, bit for bit: a NaN or an infinity in the unselected branch stays out, and a NaN payload,
/// an infinity or `-0.0` in the selected one passes through.
/// Mutation: emit the arithmetic form `if_false + cond * (if_true - if_false)` in `emit_select`
/// (`poot-kernelgen`'s `fused.rs`), with a fresh `POOT_KERNEL_CACHE_DIR`; every row goes red.
#[test]
fn select_is_bitwise_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::select::assert_select_is_bitwise(exec.as_mut(), t);
}

/// POOT-1041: a BF16 and an F16 `Select` equal the hand-written words and the oracle's, bit for bit: a signaling
/// NaN, NaN payloads, infinities and `-0.0` in the selected branch pass through unquieted, the same values in the
/// unselected one stay out.
/// Mutation (each with a fresh `POOT_KERNEL_CACHE_DIR`): make the `Select` always take `if_true` in `emit_select`
/// (`poot-kernelgen`'s `fused.rs`), or widen each branch through F32 and narrow back; the rows go red.
#[test]
fn two_byte_select_is_bitwise_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::select::assert_two_byte_select_is_bitwise(exec.as_mut(), t);
}

/// Card 1011: a BF16 const cast to F32 is read from its packed `u32` lanes by the imported
/// `packed_bf16_to_f32` body. Four cases (five elements ending in a half-used lane, and three single
/// elements) carry a quiet NaN payload, a negative NaN payload and a subnormal; every device word equals the
/// oracle's and the BF16 widening's, bit for bit.
/// Mutation: delete the `i / 2 < words.len()` tail guard in `packed_bf16_to_f32.rs` (and regenerate the
/// asset with a fresh `POOT_KERNEL_CACHE_DIR`), or change its shift; this row goes red.
#[test]
fn bf16_const_cast_is_bit_exact_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::bf16_cast::bf16_const_cast_fixtures() {
        poot_executor_parity::bf16_cast::assert_one_packed_cast(fixture, t);
        let oracle = oracle_for(&fixture.graph, &fixture.store)(&fixture.steps[0]);
        let got = poot_executor_parity::run_outputs(exec.as_mut(), t, fixture)
            .unwrap_or_else(|e| panic!("{e}"));
        poot_executor_parity::bf16_cast::assert_bit_exact(fixture, &got[0], &oracle);
    }
}

/// POOT-1074: a recurrent state pool addressed by slot (rows in any order, a masked row committing
/// nothing) and `reset_state(Row(n))` zeroing exactly pool row `n`, on ROCm; a row outside the pool is a
/// typed `StateRow` refusal that zeroes nothing.
/// The pool runs over F32 and over I32 (the PLE token history), and a `Positional` counter beside it must
/// survive every `Row`; the I32 run puts values above 2^24 in the pool, so it also proves the pool stays on
/// the exact lane. Mutations: write the row at offset 0 instead of `row * bytes` in `Engine::reset_state`
/// (the rows after `Row(2)` continue from slot 2's old state); let `Row` zero every state (the counter
/// resets); drop the fixture's own `b.scatter_update` for the I32 pool (`state_row.rs`; the pool never
/// changes); pin the I32 values to the F32 mirror lane (`required` always false in
/// `ExactI32StorageAnalysis::new`; `2^24 + 1` reads back as `2^24`). `Step::carry_recurrent`'s I32 path is
/// guarded by `served_qwen4exp_*`, not by this row.
#[test]
fn state_pool_rows_reset_independently_on_rocm() {
    let Some(device) = open_or_skip(DeviceBackend::Rocm, RocmDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::state_row::run_state_pool_row_reset(exec.as_mut(), t)
        .unwrap_or_else(|e| panic!("{e}"));
}

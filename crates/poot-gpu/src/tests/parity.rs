//! SC-002: the parity table runs on wgpu, and for each fixture every replayed step (the
//! record-then-replay step and every later pure-replay step, including a repeat of the first
//! step's inputs) matches the CPU oracle within tolerance (ADR-0101 tier 1).

use crate::device::WgpuDevice;
use poot_executor::Executor;
use poot_executor::{Device, Engine};
use poot_executor_parity::Fixture;
use poot_tensor::HostTensor;
use poot_test_util::weight_map_oracle::oracle_for;

fn target(device: &WgpuDevice) -> poot_graph_plan::Target {
    device.target()
}

#[test]
fn parity_table_matches_the_oracle_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
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

/// Card 645 SC-001: `matmul(x, transpose(w))` over an F32 `[N, K]` weight, at `M = 1` (the decode GEMV), `4` and `33`
/// (the tiled GEMM), compiled for wgpu matches `eval(g)` of the traced graph, within the parity tolerance
/// (ADR-0101 tier 2), at every replayed step. The oracle evaluates the `Transpose` + `MatMul` the tracer
/// emitted; the device runs the `DenseContraction` the fold made of it, reading the weight as stored.
/// Mutation: in the F32 GEMV body read B as `[K, N]` (`gemv_lds`, `WeightLayout::Nk` arm); the `M = 1` row goes red.
#[test]
fn checkpoint_orientation_projection_matches_the_oracle_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::checkpoint_projection_fixtures() {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Card 736: the ALiBi attention BLOOM and MPT trace (`alibi_mask_from_pos` over a
/// `ComputedConst::AlibiSlopes` of 6 heads, decode at three positions and a 4-token prefill), compiled for
/// wgpu, matches `eval(g)` at every replayed step (ADR-0101 tier 1) AND an f64 reference that writes the
/// published slopes out as literals (tier 2), so a slope formula both the oracle and the device read from
/// the same constant cannot hide. Mutation: in `ComputedConst::values_f32` return zero slopes (or drop the
/// odd-index series of the second half); the reference comparison goes red.
#[test]
fn alibi_slopes_attention_matches_the_oracle_and_the_reference_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
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
/// `4` and `33` (the tiled GEMM) and under the MoE hang guard (the serial kernel), compiled for wgpu: the program
/// holds one `DenseContraction` and no `Transpose`, and every replayed step matches `eval(g)` of the traced graph
/// within the parity tolerance (ADR-0101 tier 2). The weight lives on the device as packed binary16 words.
/// Mutation: remove `DType::F16` from `DENSE_CONTRACTION_WEIGHT_DTYPES`; a `Transpose` is planned and this row goes
/// red.
#[test]
fn f16_checkpoint_orientation_projection_matches_the_oracle_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::checkpoint_projection_f16_fixtures() {
        poot_executor_parity::assert_one_dense_contraction(fixture, t);
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// Card 557 SC-004 (dkernel K1): a qwen2-class decode and prefill traced with
/// `attn_logit_softcap = Some(50.0)` and compiled for wgpu match `eval(g)` of that same softcapped
/// trace within the parity tolerance, at every replayed step. The fixtures' scores reach the softcap
/// regime: the softcapped oracle differs from the uncapped trace's by far more than that tolerance, so
/// the device output cannot match it without applying the softcap. Reproduction: the pre-card flash
/// tracer arms (`trace_*_flash`) dropped the softcap. Mutation: make the Gemma 2 family skip
/// `AttentionParams::with_softcap`; the softcapped and uncapped traces coincide and this row goes red.
#[test]
fn softcapped_qwen2_matches_the_oracle_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
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
        assert_softcap_is_visible(softcapped, uncapped);
    }
}

/// The softcapped trace's oracle differs from the uncapped trace's at some step, by more than ten
/// times the parity tolerance (relative to `max(|uncapped|, 1)`).
fn assert_softcap_is_visible(softcapped: &Fixture, uncapped: &Fixture) {
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
        "{}: the softcap barely changes the oracle (max relative difference {largest:.2e}), so this \
         fixture cannot show a device applying it",
        softcapped.name
    );
}

/// Card 546b: `run_once` is the executor-contract replacement for the deleted pre-contract `run`
/// (see its own doc comment for the before/after shape a migrated test call takes). This proves it
/// on a hand-built graph with one `Storage::Const` row and one `Storage::Slot` row - the same shape
/// a poot-gpu test that built its own graph via `Builder` and called `gpu.run(&g, &inputs)` has -
/// and checks the elementwise-add output against hand-computed expected bytes.
#[test]
fn run_once_matches_a_hand_computed_elementwise_add() {
    use poot_executor_parity::ConstFixture;
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::op::BinOp;
    use poot_graph_ir::{Slot, TensorType};
    use poot_test_util::StepFixture;

    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));

    let b = Builder::new();
    let w = b.constant("w", TensorType::f32(vec![2, 2]));
    let x = b.slot(Slot::Activation, TensorType::f32(vec![2, 2]));
    let out = b.binary(BinOp::Add, w, x);
    let g = b.finish(out);
    let slot_key = g
        .meta(x.id)
        .slot_key()
        .expect("declared slot carries a SlotKey")
        .clone();

    let consts = [ConstFixture {
        name: "w",
        tensor: HostTensor::f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
    }];
    let slots = [StepFixture {
        key: slot_key,
        tensor: HostTensor::f32(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]),
    }];

    let got = poot_executor_parity::run_once(exec.as_mut(), t, &g, &consts, &slots)
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(got.as_f32().unwrap(), [11.0, 22.0, 33.0, 44.0]);
}

/// Card 1011: a BF16 const cast to F32 is read from its packed `u32` lanes by the imported
/// `packed_bf16_to_f32` body. Four cases (five elements ending in a half-used lane, and three single
/// elements) carry a quiet NaN payload, a negative NaN payload and a subnormal; every device word equals the
/// oracle's and the BF16 widening's, bit for bit.
/// Mutation: delete the `i / 2 < words.len()` tail guard in `packed_bf16_to_f32.rs` (and regenerate the
/// asset with a fresh `POOT_KERNEL_CACHE_DIR`), or change its shift; this row goes red.
#[test]
fn bf16_const_cast_is_bit_exact_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for fixture in &poot_executor_parity::bf16_cast::bf16_const_cast_fixtures() {
        poot_executor_parity::bf16_cast::assert_one_packed_cast(fixture, t);
        let oracle = oracle_for(&fixture.graph, &fixture.store)(&fixture.steps[0]);
        let got = poot_executor_parity::run_outputs(exec.as_mut(), t, fixture)
            .unwrap_or_else(|e| panic!("{e}"));
        poot_executor_parity::bf16_cast::assert_bit_exact(fixture, &got[0], &oracle);
    }
}

/// The packed linear per scheme (Q4_0, Q8_0, AWQ, GPTQ) at the decode GEMV and at a few rows, and the canonical
/// indexed and grouped expert chain over each, compiled for wgpu, match the shared CPU oracle at every replayed
/// step. The oracle binds each packed const as its component of the named role (`oracle_for`), so this row is the
/// wgpu witness that the shared oracle's packed arm is live: with the arm gone a packed const binds as dense and
/// the oracle panics in `materialize_dense`.
/// Mutation: drop the packed-components arm of `poot_test_util::weight_map_oracle::oracle_for`; every row goes red.
#[test]
fn packed_linears_and_the_canonical_expert_chain_match_the_oracle_on_wgpu() {
    let Ok(device) = WgpuDevice::new() else {
        eprintln!("skip: no wgpu adapter");
        return;
    };
    let t = target(&device);
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixtures = poot_executor_parity::packed::packed_linear_fixtures()
        .into_iter()
        .chain(poot_executor_parity::packed::packed_chain_fixtures());
    for fixture in fixtures {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_parity(exec.as_mut(), t, &fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

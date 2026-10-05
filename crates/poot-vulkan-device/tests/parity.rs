//! SC-001: the parity table (`poot-executor-parity`, Card 546a) and the backend-neutral executor-coverage
//! rows run on raw Vulkan through the executor contract, and every replayed step (the record-then-replay
//! step and every later pure-replay step, including a repeat of the first step's inputs) matches the CPU
//! oracle within tolerance (ADR-0101 tier 1). Mutation (SC-001): the caps the device reports must be the
//! physical device's; `large_grid.rs` is the row that turns red when they are not.
//!
//! The rows are the ROCm lane's (`poot-rocm-gpu/tests/parity.rs`, `executor_coverage.rs`) minus the ones
//! that name AMD matrix hardware (WMMA bf16) or load a real checkpoint (the Qwen2.5-0.5B row is the
//! `generate.rs` ignored row on this backend). The kernels and planner paths these rows guard are
//! backend-neutral; each row's own mutation was recorded by the card that added it and is not repeated here.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::Fixture;
use poot_graph_plan::{Plan, Submission, Target};
use poot_kernel_ir::Body;
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::weight_map_oracle::oracle_for;
use poot_vulkan_device::VulkanDevice;

/// The raw-Vulkan executor and the target it compiles for, or `None` (a skip) without a device.
fn vulkan() -> Option<(Box<dyn Executor>, Target)> {
    let device = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new())?;
    let target = device.target();
    Some((Box::new(Engine::new(device)), target))
}

/// Run `fixture` through the parity runner against the oracle's per-step expectation.
fn assert_matches_oracle(exec: &mut dyn Executor, target: Target, fixture: &Fixture) {
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    poot_executor_parity::run_parity(exec, target, fixture, &mut oracle)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// The kernel bodies `fixture`'s program dispatches for `target`, each with its total thread grid.
fn dispatched_bodies(fixture: &Fixture, target: Target) -> Vec<(Body, [u32; 3])> {
    poot_executor_parity::staged(fixture, target, Submission::Replay)
        .stages()
        .flat_map(|(_, _, program)| {
            program
                .planned()
                .filter_map(|(_, plan)| match plan {
                    Plan::Compute { body, grid, .. } | Plan::ComputeMeta { body, grid, .. } => {
                        Some((body.clone(), *grid))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The largest `|a - b| / max(|b|, 1)` between two same-shaped F32 tensors.
fn max_relative_difference(a: &HostTensor, b: &HostTensor) -> f32 {
    a.as_f32()
        .expect("F32 output")
        .iter()
        .zip(b.as_f32().expect("F32 output"))
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0, f32::max)
}

#[test]
fn parity_table_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
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
/// `attn_logit_softcap = Some(50.0)` and compiled for raw Vulkan match `eval(g)` of that same softcapped
/// trace within the parity tolerance, at every replayed step. The fixtures' scores reach the softcap
/// regime: the softcapped oracle differs from the uncapped trace's by far more than that tolerance, so
/// the device output cannot match it without applying the softcap. Reproduction: the pre-card flash
/// tracer arms (`trace_*_flash`) dropped the softcap. Mutation: make the qwen2 tracers call attention
/// without `cfg.attn_logit_softcap`; the softcapped and uncapped traces coincide and this row goes red.
#[test]
fn softcapped_qwen2_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
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
/// (the tiled GEMM), compiled for raw Vulkan matches `eval(g)` of the traced graph, within the parity tolerance
/// (ADR-0101 tier 2), at every replayed step. The oracle evaluates the `Transpose` + `MatMul` the tracer
/// emitted; the device runs the `DenseContraction` the fold made of it, reading the weight as stored.
/// Mutation: in the F32 GEMV body read B as `[K, N]` (`gemv_lds`, `WeightLayout::Nk` arm); the `M = 1` row goes red.
#[test]
fn checkpoint_orientation_projection_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
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
/// raw Vulkan, matches `eval(g)` at every replayed step (ADR-0101 tier 1) AND an f64 reference that writes the
/// published slopes out as literals (tier 2), so a slope formula both the oracle and the device read from
/// the same constant cannot hide. Mutation: in `ComputedConst::values_f32` return zero slopes (or drop the
/// odd-index series of the second half); the reference comparison goes red.
#[test]
fn alibi_slopes_attention_matches_the_oracle_and_the_reference_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
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
/// `4` and `33` (the tiled GEMM) and under the MoE hang guard (the serial kernel), compiled for raw Vulkan: the program
/// holds one `DenseContraction` and no `Transpose`, and every replayed step matches `eval(g)` of the traced graph
/// within the parity tolerance (ADR-0101 tier 2). The weight lives on the device as packed binary16 words.
/// Mutation: remove `DType::F16` from `DENSE_CONTRACTION_WEIGHT_DTYPES`; a `Transpose` is planned and this row goes
/// red.
#[test]
fn f16_checkpoint_orientation_projection_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
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

/// Card 1011: a BF16 const cast to F32 is read from its packed `u32` lanes by the imported
/// `packed_bf16_to_f32` body. Four cases (five elements ending in a half-used lane, and three single
/// elements) carry a quiet NaN payload, a negative NaN payload and a subnormal; every device word equals the
/// oracle's and the BF16 widening's, bit for bit.
/// Mutation: delete the `i / 2 < words.len()` tail guard in `packed_bf16_to_f32.rs` (and regenerate the
/// asset with a fresh `POOT_KERNEL_CACHE_DIR`), or change its shift; this row goes red.
#[test]
fn bf16_const_cast_is_bit_exact_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
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

/// `ArgTopK` over a partial last workgroup: `l = 3, e = 16, k = 4` (12 output lanes) and `17 x 16, k = 4` (a
/// full 64-lane workgroup and a 4-lane tail whose lanes are the last row's outputs, that row ranking its last
/// expert first). Each case equals the oracle at every step and the ids derived by inverting the rank
/// permutation directly, so a lost tail winner cannot hide behind an oracle that shares the fault.
#[test]
fn arg_top_k_over_a_partial_last_workgroup_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for case in poot_executor_parity::routing::arg_top_k_tail_cases() {
        let name = case.fixture.name;
        let outputs = poot_executor_parity::run_outputs(exec.as_mut(), target, &case.fixture)
            .unwrap_or_else(|e| panic!("{e}"));
        for (step, got) in outputs.iter().enumerate() {
            assert_eq!(
                got.as_f32().expect("ids are F32"),
                case.expected.as_slice(),
                "{name} step {step}: ids differ from the rank permutation's inverse"
            );
        }
        assert_matches_oracle(exec.as_mut(), target, &case.fixture);
    }
}

/// Router scores with ties (all equal, a tie at the top-k cut, split maxima, signed zeros, `k = 1`, `k = experts`,
/// and four rows with different ties) go through the stable descending rank, `ArgTopK`, the keep mask and the
/// gate on the device. The device's selected experts equal a stable descending sort's (tied experts lowest
/// index first, in that order), computed here without the oracle, and the whole `[ids, mask, gate]` row equals
/// the oracle's.
#[test]
fn tied_moe_routing_picks_the_lowest_index_first_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for case in poot_executor_parity::routing::tie_cases() {
        let name = case.fixture.name;
        let outputs = poot_executor_parity::run_outputs(exec.as_mut(), target, &case.fixture)
            .unwrap_or_else(|e| panic!("{e}"));
        let k = case.k;
        for (step, got) in outputs.iter().enumerate() {
            let shape = got.shape().to_vec();
            let width = shape[1];
            let rows = got.as_f32().expect("routed output is F32").chunks(width);
            let expected_rows = case.expected_ids.chunks(k);
            for (row, (got_row, want_ids)) in rows.zip(expected_rows).enumerate() {
                assert_eq!(
                    &got_row[..k],
                    want_ids,
                    "{name} step {step} row {row}: selected experts"
                );
            }
        }
        assert_matches_oracle(exec.as_mut(), target, &case.fixture);
    }
}

/// Flash decode and flash prefill (the generated region kernels, and the imported flash prefill that a
/// softcapped prefill selects) with a real per-head ALiBi mask match the oracle's materialized attention.
/// Every head has its own slope, so a lowering that read the mask at head stride 0 would score every head
/// with head 0's row; the row also requires the oracle itself to tell that apart (the per-head output differs
/// materially from the head-0 broadcast output), so it cannot pass vacuously.
#[test]
fn flash_attention_with_a_per_head_alibi_mask_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for case in poot_executor_parity::attention::alibi_flash_cases() {
        let name = case.per_head.name;
        let ops = poot_executor_parity::planned_ops(&case.per_head, target);
        assert!(
            ops.iter().any(|op| op.starts_with(case.flash_op)),
            "{name}: the program must hold {}: {ops:?}",
            case.flash_op
        );
        let per_head = oracle_for(&case.per_head.graph, &case.per_head.store)(&[]);
        let head0 = oracle_for(&case.head0_broadcast.graph, &case.head0_broadcast.store)(&[]);
        let spread = max_relative_difference(&per_head, &head0);
        assert!(
            spread > 2e-2,
            "{name}: the per-head mask barely changes the oracle (max relative difference {spread:.2e})"
        );
        assert_matches_oracle(exec.as_mut(), target, &case.per_head);
    }
}

/// A qwen2 decode over the multi-head (`2 x 2`) and multi-query (`2 x 1`) query/key-value head layouts matches
/// the oracle at every step. (The grouped-query layout, `4 x 2`, is the parity table's `qwen2_decode` row.)
#[test]
fn decode_over_multi_head_and_multi_query_layouts_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for fixture in &poot_executor_parity::attention::head_layout_decode_fixtures(8) {
        assert_matches_oracle(exec.as_mut(), target, fixture);
    }
}

/// Masked attention with RoPE and a KV cache, a Gated-DeltaNet layer and an `ArgTopK`-routed MoE layer (the bias
/// toy with biased Q/K/V, so the matmul-plus-bias contraction runs inside the recorded program) decoded over
/// six steps through one entry. The entry records on its first step and replays after; each step matches the
/// oracle, and after a state reset the same six steps replayed from the recording equal the first pass bit for
/// bit, the first step against the recording step's own execution.
///
/// The KV caches are updated in place; the conv window and the recurrent matrix carry through the engine's
/// explicit two-phase commit copies (`Device::copy`, `vkCmdCopyBuffer` here). The toy's weights are scaled so
/// those two states move the logits by far more than the tolerance.
#[test]
fn captured_replay_of_the_toy_graphs_equals_the_recording_step_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for fixture in [
        poot_executor_parity::capture::gdn_moe_toy_fixture(),
        poot_executor_parity::capture::bias_toy_fixture(),
    ] {
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_capture_replay(exec.as_mut(), target, &fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// The generated tiled GEMM at the gemma4-MoE router shape `M = 32, K = 2816, N = 128`, and the same shape with a
/// remainder tile on both output axes (`M = 33`, `N = 130`). Each matches the oracle at every step.
#[test]
fn tiled_region_with_a_remainder_tile_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for fixture in &poot_executor_parity::launch::tiled_region_fixtures() {
        assert_matches_oracle(exec.as_mut(), target, fixture);
    }
}

/// One elementwise region over 70001 elements is one dispatch whose grid needs more than 256 workgroups at any
/// workgroup size, so `workgroup_id.x > 255` and the last workgroup is partial. It matches the oracle at every
/// step.
#[test]
fn a_one_dispatch_grid_past_255_workgroups_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    let fixture = poot_executor_parity::launch::wide_grid_fixture();
    let widest = dispatched_bodies(&fixture, target)
        .iter()
        .map(|(body, grid)| grid[0].div_ceil(body.workgroup_size[0].max(1)))
        .max()
        .expect("the program dispatches a kernel");
    assert!(
        widest > 256,
        "the fixture's widest dispatch is {widest} workgroups: it must need workgroup_id.x > 255"
    );
    assert_matches_oracle(exec.as_mut(), target, &fixture);
}

/// A packed linear per scheme (Q4_0, Q8_0, AWQ, GPTQ), at the decode GEMV and at a few rows, and the canonical
/// indexed and grouped expert chain (`PackedDequant -> Transpose -> Reshape -> Concat ->
/// IndexedMatMul`, the traced graph, so the row survives the later contraction claim) over each, match the
/// oracle at every step. The weights are real-magnitude (`|w| <= 0.25`, checked when the payload is built) and
/// the activations of unit order, so a role or group fault moves the output far past the tolerance; the row
/// also requires each oracle output to be of order one.
#[test]
fn packed_linears_and_the_canonical_expert_chain_match_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    let fixtures = poot_executor_parity::packed::packed_linear_fixtures()
        .into_iter()
        .chain(poot_executor_parity::packed::packed_chain_fixtures());
    for fixture in fixtures {
        let expected = oracle_for(&fixture.graph, &fixture.store)(&fixture.steps[0]);
        let largest = expected
            .as_f32()
            .expect("F32 oracle output")
            .iter()
            .fold(0.0f32, |largest, v| largest.max(v.abs()));
        assert!(
            largest > 0.5,
            "{}: the oracle output is not of order one (largest {largest:.2e})",
            fixture.name
        );
        assert_matches_oracle(exec.as_mut(), target, &fixture);
    }
}

/// The chunked Gated-DeltaNet prefill (many small chained dependent dispatches in one recorded program: a Hillis-Steele chunked decay scan, a tiled
/// key-head repeat `hv -> hv % h_k`) over two calls that carry one recurrent state, so the second call
/// starts from a nonzero state and both state-dependent cross terms are exercised. Each call equals
/// the oracle of the chunked graph, and equals the sequential `gated_delta_net_decode` recurrence run
/// token by token on the CPU oracle (state threaded across the two calls), which the chunked algebra
/// must reproduce.
#[test]
fn chunked_gdn_prefill_matches_the_sequential_recurrence_on_vulkan() {
    use poot_executor_parity::gdn::{D, GdnInputs, H_K, H_V, LEN};
    use poot_graph_ir::ops::gated_delta_net_decode;
    use poot_graph_ir::{Builder, StateRole, TensorType};

    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    let case = poot_executor_parity::gdn::gdn_prefill_chunked_case();
    assert_matches_oracle(exec.as_mut(), target, &case.fixture);
    let outputs = poot_executor_parity::run_outputs(exec.as_mut(), target, &case.fixture)
        .unwrap_or_else(|e| panic!("{e}"));

    // The sequential reference: one decode step per token, key heads repeated tiled.
    let b = Builder::new();
    let per_head = |tag: &str, last: usize| b.constant(tag, TensorType::f32(vec![1, H_V, 1, last]));
    let (q, k, v) = (per_head("q", D), per_head("k", D), per_head("v", D));
    let (g, beta) = (per_head("g", 1), per_head("beta", 1));
    let state = b.state_input(
        "s",
        TensorType::f32(vec![1, H_V, D, D]),
        StateRole::Recurrent,
    );
    let (out, state_out) = gated_delta_net_decode(&b, q, k, v, g, beta, state);
    let graph = b.finish_with_state(out, &[(state, state_out)]);
    let mut carried = HostTensor::zeros(vec![1, H_V, D, D]);
    for (call, (inputs, got)) in case.inputs.iter().zip(&outputs).enumerate() {
        let GdnInputs {
            q: qd,
            k: kd,
            v: vd,
            g: gd,
            beta: bd,
        } = inputs;
        let mut want = vec![0.0f32; H_V * LEN * D];
        for t in 0..LEN {
            let heads = |data: &[f32], heads_in: usize, last: usize| -> HostTensor {
                let values = (0..H_V)
                    .flat_map(|hv| {
                        let source = hv % heads_in;
                        (0..last).map(move |d| data[(source * LEN + t) * last + d])
                    })
                    .collect();
                HostTensor::f32(vec![1, H_V, 1, last], values)
            };
            let bound = HashMap::from([
                (q.id, Value::from(heads(qd, H_K, D))),
                (k.id, Value::from(heads(kd, H_K, D))),
                (v.id, Value::from(heads(vd, H_V, D))),
                (g.id, Value::from(heads(gd, H_V, 1))),
                (beta.id, Value::from(heads(bd, H_V, 1))),
                (state.id, Value::from(carried.clone())),
            ]);
            let step = eval(&graph, &bound, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
            carried = step.state.into_iter().next().unwrap().into_host().unwrap();
            let token = step.output.into_host().unwrap();
            for hv in 0..H_V {
                for d in 0..D {
                    want[(hv * LEN + t) * D + d] = token.as_f32().unwrap()[hv * D + d];
                }
            }
        }
        let got = got.as_f32().unwrap();
        assert_eq!(got.len(), want.len(), "call {call}: output length");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "call {call}: the chunked prefill differs from the sequential recurrence by {worst:.3e}"
        );
    }
}

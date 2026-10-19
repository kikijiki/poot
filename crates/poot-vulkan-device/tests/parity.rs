//! SC-001: the parity table (`poot-executor-parity`, Card 546a) and the backend-neutral executor-coverage
//! rows run on raw Vulkan through the executor contract, and every replayed step (the record-then-replay
//! step and every later pure-replay step, including a repeat of the first step's inputs) matches the CPU
//! oracle within tolerance (ADR-0101 tier 1). Mutation (SC-001): the caps the device reports must be the
//! physical device's; `large_grid.rs` is the row that turns red when they are not.
//!
//! The rows are the ROCm lane's (`poot-rocm-device/tests/parity.rs`, `executor_coverage.rs`) minus the ones
//! that name AMD matrix hardware (WMMA bf16) or load a real checkpoint (the Qwen2.5-0.5B row is the
//! poot-llm `vulkan_generate.rs` ignored row on this backend). The kernels and planner paths these rows guard are
//! backend-neutral; each row's own mutation was recorded by the card that added it and is not repeated here.

use std::collections::HashMap;

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_eval::{EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::Fixture;
use poot_graph_plan::Target;
use poot_graph_plan::test_support::gpu_caps;
use poot_kernel_ir::Body;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::HostTensor;
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
    poot_executor_parity::staged(fixture, target)
        .stages()
        .flat_map(|(_, _, program)| {
            program
                .planned()
                .filter_map(|(_, plan)| {
                    plan.single()
                        .map(|dispatch| (dispatch.body().clone(), dispatch.launch().grid))
                })
                .collect::<Vec<_>>()
        })
        .collect()
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
/// the same constant cannot hide. Mutation: in `ComputedConst::tensor` return zero slopes (or drop the
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
/// `4` and `33` (the tiled GEMM), compiled for raw Vulkan: the program
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

/// POOT-1033: `Select` is a bitwise select. An F32 select (standalone, and inside a fused float region) over
/// poisoned branches and an I32 select over nonzero condition words each equal the hand-written select words
/// and the oracle's, bit for bit: a NaN or an infinity in the unselected branch stays out, and a NaN payload,
/// an infinity or `-0.0` in the selected one passes through.
/// Mutation: emit the arithmetic form `if_false + cond * (if_true - if_false)` in `emit_select`
/// (`poot-kernelgen`'s `fused.rs`), with a fresh `POOT_KERNEL_CACHE_DIR`; every row goes red.
#[test]
fn select_is_bitwise_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::select::assert_select_is_bitwise(exec.as_mut(), t);
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
                got.as_i32().expect("ids are I32"),
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

/// Attention decode and prefill (the one generated attention kernel, with and without a softcap) with a real per-head ALiBi mask match the oracle's materialized attention.
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
        let spread = poot_test_util::max_unit_floor_error(
            per_head.as_f32().expect("F32 output"),
            head0.as_f32().expect("F32 output"),
        );
        assert!(
            spread > 2e-2,
            "{name}: the per-head mask barely changes the oracle (max relative difference {spread:.2e})"
        );
        assert_matches_oracle(exec.as_mut(), target, &case.per_head);
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

/// Card 663 SC-001, raw Vulkan. The batched, multi-block cooperative-matrix matmul: `[2, 32, 48] x [2, 48, 32]`
/// f16 operands (`poot_executor_parity::launch::wmma_f16_batched_fixture`), 2 x 2 x 2 output tiles each carrying its
/// accumulator across three 16-wide K blocks, every output element of both steps compared with the oracle. The
/// fixture values give every K block a distinct nonzero contribution to every output (at least 16 per block), so a
/// dropped, doubled or wrong-batch block moves all outputs far past the parity tolerance. Skips on a device with no
/// cooperative-matrix hardware.
///
/// Mutation: drop the last K block in `poot-kernelgen`'s `tensor_core_tiles` (the loop bound `k0 < k` becomes
/// `k0 + 16 < k`), with a fresh `POOT_KERNEL_CACHE_DIR`; every output is short by its last block's contribution and
/// the row goes red. A second mutation, restoring the single K tile (`K == 16` only) in
/// `matmul_spirv_coopmat_eligible`, plans the serial kernel, and the dispatched-body check goes red.
#[test]
fn batched_multi_block_coopmat_matmul_matches_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    if gpu_caps(&target).tensor_core == poot_target::TensorCoreSupport::None {
        eprintln!("SKIP: this device reports no cooperative-matrix hardware");
        return;
    }
    let fixture = poot_executor_parity::launch::wmma_f16_batched_fixture();
    assert!(
        dispatched_bodies(&fixture, target)
            .iter()
            .any(|(body, _)| body
                .blocks
                .iter()
                .flat_map(|block| &block.statements)
                .any(|statement| matches!(statement, poot_kernel_ir::Statement::WmmaMma { .. }))),
        "{}: no dispatched kernel multiplies on the cooperative matrix",
        fixture.name
    );
    assert_matches_oracle(exec.as_mut(), target, &fixture);
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

/// A packed linear per scheme (Q4_0, Q8_0, AWQ, GPTQ, IQ3_S), at the decode GEMV and at a few rows, and the canonical
/// indexed and grouped expert chain (`PackedDequant -> Transpose -> Reshape -> Concat ->
/// IndexedMatMul`, the traced graph, so the row survives the later contraction claim) over each, match the
/// oracle at every step. The weights are real-magnitude (`|w| <= 0.25`, checked when the payload is built) and
/// the activations of unit order, so a role or group fault moves the output far past the tolerance; the row
/// also requires each oracle output to be of order one.
#[test]
fn packed_linears_and_the_indexed_expert_contraction_match_the_oracle_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    let fixtures = poot_executor_parity::packed::packed_linear_fixtures()
        .into_iter()
        .chain(poot_executor_parity::packed::packed_expert_fixtures());
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
    let b = Builder::for_tests();
    let per_head = |tag: &str, last: usize| {
        b.constant(tag, TensorType::f32(vec![1, H_V, 1, last]))
            .unwrap()
    };
    let (q, k, v) = (per_head("q", D), per_head("k", D), per_head("v", D));
    let (g, beta) = (per_head("g", 1), per_head("beta", 1));
    let state = b
        .state_input(
            "s",
            TensorType::f32(vec![1, H_V, D, D]),
            StateRole::Recurrent,
        )
        .unwrap();
    let (out, state_out) = gated_delta_net_decode(&b, q, k, v, g, beta, state).unwrap();
    let graph = b
        .finish(poot_graph_ir::Outputs::new(out).with_state(&[(state, state_out)]))
        .unwrap();
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
                (q.id(), Value::from(heads(qd, H_K, D))),
                (k.id(), Value::from(heads(kd, H_K, D))),
                (v.id(), Value::from(heads(vd, H_V, D))),
                (g.id(), Value::from(heads(gd, H_V, 1))),
                (beta.id(), Value::from(heads(bd, H_V, 1))),
                (state.id(), Value::from(carried.clone())),
            ]);
            let step = eval(&graph, &bound, EvalOptions::default()).unwrap();
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
        let worst = poot_test_util::max_abs_error(got, &want);
        assert!(
            worst < 1e-4,
            "call {call}: the chunked prefill differs from the sequential recurrence by {worst:.3e}"
        );
    }
}

/// POOT-1113: a low-head live decode (2 query rows over a 1024-key cache, `Slot::Pos` bounding the keys)
/// plans the split-KV schedule (two launches through one scratch) and, stepped at live lengths 1024, 5, 400,
/// 1, 1024 over one executable, matches the CPU oracle's evaluation of each length within
/// `SPLIT_KV_DEVICE_TOLERANCE`. The scratch slot is stale from the
/// previous step at every step but the first (length 5 leaves seven of the eight splits with no key), so a
/// split that did not publish its own neutral partial reads the first step's.
/// Mutation: in `poot-kernelgen`'s `attention_body`, skip the partial writes of a split with no key; the
/// length-5 and length-1 steps go red. Dropping one split's partial in the combine (`count` starting at 1)
/// goes red on every step.
#[test]
fn split_kv_decode_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::attention::assert_split_kv_decode_matches_the_oracle(exec.as_mut(), t)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// A paged decode attention reads the K and V pools in place through the `[B, Lk]` read map: the plan holds one
/// attention over the pools (no gather, no transpose), and two steps with different scattered maps over the
/// same pools match the CPU oracle's evaluation of the gathered views, the second as a pure replay.
/// Mutation: in `poot-kernelgen`'s `attention_body`, address a paged key by the position instead of the slot it
/// names (`read[b * lk + t]` becomes `t` in `KeyRows::Paged`); both steps go red against the oracle.
#[test]
fn paged_attention_reads_the_pools_in_place_and_matches_the_oracle_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::attention::assert_paged_attention_matches_the_oracle(exec.as_mut(), t)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// POOT-1113: the split-KV decode stepped through its live lengths twice over one entry: the second pass
/// replays the captured two-launch plan, and every step's output equals the first pass's bit for bit.
#[test]
fn split_kv_decode_replay_equals_the_recording_step_on_vulkan() {
    let Some(device) = open_or_skip(DeviceBackend::Vulkan, VulkanDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::attention::assert_split_kv_decode_replays_bit_identically(
        exec.as_mut(),
        t,
    )
    .unwrap_or_else(|e| panic!("{e}"));
}

/// POOT-1146: the greedy `SampleToken` pick is exact on raw Vulkan: over `[rows, vocab]` logits at a vocabulary that
/// leaves a partial last chunk, a single chunk and the production width, the pair `(token, non_finite_index)`
/// of every row equals the one the contract defines (lowest index among equal maxima across chunk and
/// workgroup seams, signed zeros equal, any non-finite value forces token 0 and names its lowest index), at
/// the recording step and the replay.
/// Mutations (each with a fresh `POOT_KERNEL_CACHE_DIR`; a kernel edit needs
/// `cargo test -p pootc --test import_run regen_kernel_assets -- --ignored` to regenerate
/// `argmax_chunk.kir.json` / `argmax_combine.kir.json`, which the planner embeds): in
/// `pootc/kernels/sampling/argmax_combine.rs` break ties to the larger index (`ci < best_idx` to
/// `ci > best_idx` in the lane loop and `ci < bi` to `ci > bi` in the lane-0 fold); the tie rows go red.
/// In `argmax_chunk.rs` drop the `bad_idx` update of the non-finite branch; the non-finite rows go red.
/// In `planner/sampling.rs` plan `groups = vocab / ARGMAX_CHUNK` (dropping the partial last chunk); the
/// rows whose maximum sits in the last chunk go red.
#[test]
fn greedy_sample_token_is_exact_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    poot_executor_parity::sample_token::assert_greedy_pick_is_exact(exec.as_mut(), target);
}

/// POOT-1143 stage 2: the indexer's block selection (ties straddling the cut, fewer complete blocks than `k`,
/// queries as rows by tokens, a wide row, all scores equal) runs through the generated radix-select kernel
/// on the device, and its words equal both the plain stable-sort selection (computed here without the oracle)
/// and the oracle's primitive definition at every replayed step.
/// Mutation: in `block_top_k.rs` take the last equal blocks at the threshold instead of the first (ties
/// ascending by block); the ties case goes red.
#[test]
fn block_top_k_selects_what_the_plain_selection_selects_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for case in poot_executor_parity::block_top_k::cases() {
        poot_executor_parity::block_top_k::assert_device_selects(exec.as_mut(), target, &case);
        if case.oracle {
            assert_matches_oracle(exec.as_mut(), target, &case.fixture);
        }
    }
}

/// POOT-1143 stage 3: attention over the blocks a selection names (a GQA decode, a prefill with empty and short
/// selections, every block selected, and the design's deep decode: 24 query heads over 2 of 256, 32768 positions,
/// 512 blocks selected) runs through the generated kernel on the device and equals the independent f64 attention
/// over the selected keys within the tier-2 tolerance, and the oracle's dense definition where it is cheap.
/// Mutation: in `block_candidate` read the unfinished block's start one block early; the queries take keys of a
/// block that is not their own and the first case goes red.
#[test]
fn selected_attention_reads_exactly_the_selected_blocks_on_vulkan() {
    let Some((mut exec, target)) = vulkan() else {
        return;
    };
    for case in poot_executor_parity::selected_attention::all() {
        poot_executor_parity::selected_attention::assert_device_attends(
            exec.as_mut(),
            target,
            &case,
        );
        if case.oracle {
            assert_matches_oracle(exec.as_mut(), target, &case.fixture);
        }
    }
}

//! Card 607: the ROCm device coverage the deleted probes carried (Card 511 removed the probe methods,
//! `rocm-graph-check` and their tests), rebuilt as device rows over the public executor contract
//! (`load_weights`, `add_entry`, `step`) on `Engine<RocmDevice>`. No probe entry returns to the
//! executor: each row builds a fixture program, runs it through the one contract path, and compares
//! every element of every step's output with the CPU oracle (NaN never agrees, ADR-0101) and, where
//! the row has a value an oracle could share a fault with, with a value derived here without it.
//!
//! The fixture programs that are backend neutral live in `poot-executor-parity`; the rows that name
//! a ROCm-only shape (a wavefront-sized partial workgroup, the gfx1151 tiled region, WMMA, a grid
//! past 255 workgroups) and the real-weight row live here.
//!
//! Each row names the deleted probe it replaces and the mutation that turns it red; the mutations
//! guard code outside this file, so each was applied once, observed red, and restored.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor::{Device, Engine, Executor};
use poot_executor_parity::Fixture;
use poot_graph_ir::Storage;
use poot_graph_plan::{Plan, Submission, Target};
use poot_kernel_ir::{Body, Statement};
use poot_rocm_gpu::device::RocmDevice;
use poot_runtime_common::DeviceBackend;
use poot_tensor::HostTensor;
use poot_test_util::device_skip::open_or_skip;
use poot_test_util::weight_map_oracle::oracle_for;

/// The ROCm executor and the target it compiles for, or `None` (a skip) without a device.
fn rocm() -> Option<(Box<dyn Executor>, Target)> {
    let device = open_or_skip(DeviceBackend::Rocm, RocmDevice::new())?;
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

/// SC-001. Replaces `probe_arg_top_k_rocm_test` (`l = 3, e = 16, k = 4`: 12 output lanes, the partial
/// workgroup shape that crashed ROCm with an HSA aperture violation through the idle-lane out-of-bounds
/// read of card 165), and adds `17 x 16, k = 4`: a full 64-lane workgroup and a 4-lane tail whose lanes
/// are the last row's outputs, that row ranking its last expert first. Each case equals the oracle at
/// every step and the ids derived by inverting the rank permutation directly, so a lost tail winner
/// cannot hide behind an oracle that shares the fault.
///
/// Mutation: bound the launch by full workgroups only (`padded_launch` in `poot-rocm-gpu/src/device.rs`
/// floors the workgroup count instead of rounding it up); the tail lanes never run, the last row's ids
/// stay unwritten, and the `17 x 16` case goes red.
#[test]
fn arg_top_k_over_a_partial_last_workgroup_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
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

/// SC-002. Replaces `probe_stable_moe_ties_rocm_test`. Router scores with ties (all equal, a tie at the
/// top-k cut, split maxima, signed zeros, `k = 1`, `k = experts`, and four rows with different ties) go
/// through the stable descending rank, `ArgTopK`, the keep mask and the gate on the device. The device's
/// selected experts equal a stable descending sort's (tied experts lowest index first, in that order),
/// computed here without the oracle, and the whole `[ids, mask, gate]` row equals the oracle's.
///
/// Mutation: break the comparison the ties turn on in the device's code (`poot-codegen`'s `emit.rs`
/// compiles a float `>=` as `>`, with a fresh `POOT_KERNEL_CACHE_DIR`); tied experts no longer order by
/// index and the first tie row goes red. (A change to the comparison in the traced rank would change the
/// graph the oracle evaluates too, so only the selected ids derived here would catch it.)
#[test]
fn tied_moe_routing_picks_the_lowest_index_first_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
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

/// SC-004. Replaces `probe_rocm_flash_decode_test`, `probe_rocm_flash_prefill_test` and
/// `probe_rocm_flash_per_head_alibi_mask_test`. Flash decode and flash prefill (the generated region
/// kernels, and the imported flash prefill that a softcapped prefill selects) with a real per-head
/// ALiBi mask match the oracle's materialized attention. Every head has its own slope, so a lowering
/// that read the mask at head stride 0 would score every head with head 0's row; the row also requires
/// the oracle itself to tell that apart (the per-head output differs materially from the head-0
/// broadcast output), so it cannot pass vacuously.
///
/// Mutation: read the mask with head stride 0 in the flash lowering (the planner's flash arms in
/// `poot-graph-plan`'s `planner/attention.rs` plan every mask as the broadcast layout); the decode case
/// goes red.
#[test]
fn flash_attention_with_a_per_head_alibi_mask_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
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

/// Replaces `probe_gqa_decode_rocm` and `probe_multihead_decode_rocm`: a qwen2 decode over the
/// multi-head (`2 x 2`) and multi-query (`2 x 1`) query/key-value head layouts matches the oracle at
/// every step. (The grouped-query layout, `4 x 2`, is the parity table's `qwen2_decode` row in
/// `tests/parity.rs`.)
#[test]
fn decode_over_multi_head_and_multi_query_layouts_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
        return;
    };
    for fixture in &poot_executor_parity::attention::head_layout_decode_fixtures(8) {
        assert_matches_oracle(exec.as_mut(), target, fixture);
    }
}

/// SC-005. Replaces `probe_capture_replay_gdn_moe_toy_rocm_test` and
/// `probe_capture_replay_bias_toy_rocm_test`: masked attention with RoPE and a KV cache, a
/// Gated-DeltaNet layer and an `ArgTopK`-routed MoE layer (the bias toy with biased Q/K/V, so the
/// matmul-plus-bias contraction runs inside the recorded program) decoded over six steps through one
/// entry. The entry records on its first step and replays after; each step matches the oracle, and
/// after a state reset the same six steps replayed from the recording equal the first pass bit for
/// bit, the first step against the recording step's own execution. (`Submission::Eager` no longer
/// reaches an executor: the contract refuses it, Card 713, so there is no eager path to compare.)
///
/// The KV caches are updated in place; the conv window and the recurrent matrix carry through the
/// engine's explicit two-phase commit copies. The toy's weights are scaled so those two states move the
/// logits by far more than the tolerance.
///
/// Mutation: skip the state-commit copies between replays (`Engine`'s step walk in `poot-executor`
/// records no `entry.commits`); step 1 reads a stale conv window and recurrent matrix, diverges from the
/// oracle, and the toy goes red. A second mutation, `Engine::reset_state` leaving the carried state, makes
/// the second pass diverge from the first and turns the bit-for-bit comparison red.
#[test]
fn captured_replay_of_the_toy_graphs_equals_the_recording_step_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
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

/// SC-006. Replaces `probe_tiled_region_rocm_test` (card 186: does the generated tiled GEMM hang
/// gfx1151 at the gemma4-MoE router shape `M = 32, K = 2816, N = 128`), and adds the same shape with a
/// remainder tile on both output axes (`M = 33`, `N = 130`). Each matches the oracle at every step.
///
/// Mutation: in `poot-kernelgen`'s `tiled_region` kernel, drop the last tile's remainder (the column
/// tile count is `n / tile` in place of `n.div_ceil(tile)`), with a fresh `POOT_KERNEL_CACHE_DIR`; the
/// remainder-tile row goes red and the exact router shape stays green.
#[test]
fn tiled_region_with_a_remainder_tile_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
        return;
    };
    for fixture in &poot_executor_parity::launch::tiled_region_fixtures() {
        assert_matches_oracle(exec.as_mut(), target, fixture);
    }
}

/// SC-008, WMMA half. Replaces the deleted WMMA bf16 `16x16x16` and `64x64x64` probes: a mixed-precision
/// matmul (bf16 operands, f32 accumulate and result, the shape the planner runs on the RDNA3 tensor
/// cores) at one tile and at a `4 x 4` grid of tiles with a four-step K loop. The operands are small
/// integers, so the exact result is an integer and any wrong product, lost K step or unseeded
/// accumulator moves an element by at least one, three orders of magnitude past the parity tolerance
/// (the tensor cores' own accumulation is not IEEE-exact: the device lands within about `1e-6` of the
/// integer, which is why the comparison is a tolerance and not a bit match).
///
/// Mutation: zero the accumulator fragment after each K step (a `WmmaZero` after the `WmmaMma` in
/// `poot-kernelgen`'s `matmul_tensorcore`), with a fresh `POOT_KERNEL_CACHE_DIR`; the stored result is
/// all zeros and both sizes go red.
#[test]
fn wmma_bf16_matmuls_match_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
        return;
    };
    for fixture in &poot_executor_parity::launch::wmma_bf16_fixtures() {
        assert!(
            dispatched_bodies(fixture, target)
                .iter()
                .any(|(body, _)| body
                    .blocks
                    .iter()
                    .flat_map(|block| &block.statements)
                    .any(|statement| matches!(statement, Statement::WmmaMma { .. }))),
            "{}: no dispatched kernel multiplies on the WMMA tensor cores",
            fixture.name
        );
        assert_matches_oracle(exec.as_mut(), target, fixture);
    }
}

/// SC-008, grid half. Replaces the deleted single-AQL 256-workgroup probe: one elementwise region over
/// 70001 elements is one dispatch whose grid needs more than 256 workgroups at any workgroup size, so
/// `workgroup_id.x > 255` and the last workgroup is partial. It matches the oracle at every step.
///
/// Mutation: clamp `workgroup_id.x` to 255 in the ROCm launch (`padded_launch` in
/// `poot-rocm-gpu/src/device.rs` caps the x workgroup count at 256); every element past the 256th
/// workgroup is unwritten and the row goes red.
#[test]
fn a_one_dispatch_grid_past_255_workgroups_matches_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
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

/// SC-003. Replaces the deleted dequant probes (`dequant_probes.rs` GPTQ, AWQ, Q8_0 and Q4_0, and
/// `probe_indexed_matmul_dequant_rocm_test` and `probe_grouped_dequant_gemm_rocm_test`, whose dequant
/// matmul ops Card 712 deleted). A packed linear per scheme (Q4_0, Q8_0, AWQ, GPTQ), at the decode GEMV
/// and at a few rows, and the canonical indexed and grouped expert chain (`PackedDequant ->
/// Transpose -> Reshape -> Concat -> IndexedMatMul`, the traced graph, so the row survives the later
/// contraction claim) over each, match the oracle at every step. The weights are real-magnitude
/// (`|w| <= 0.25`, checked when the payload is built) and the activations of unit order, so a role or
/// group fault moves the output far past the tolerance; the row also requires each oracle output to be
/// of order one.
///
/// Mutation: swap two `SourceRole`s in the engine's packed bind (the W9 role swap: `weight_buffer` in
/// `poot-executor`'s `engine.rs` reads the zero-point component for a scale source and the scale for a
/// zero-point source); the AWQ rows go red (the swapped component has the wrong length, so `add_entry`
/// refuses it).
#[test]
fn packed_linears_and_the_canonical_expert_chain_match_the_oracle_on_rocm() {
    let Some((mut exec, target)) = rocm() else {
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

/// Replaces `probe_gdn_prefill_chunked_rocm_test`. The chunked Gated-DeltaNet prefill (many small
/// chained dependent dispatches in one recorded program: a Hillis-Steele chunked decay scan, a tiled
/// key-head repeat `hv -> hv % h_k`) over two calls that carry one recurrent state, so the second call
/// starts from a nonzero state and both state-dependent cross terms are exercised. Each call equals
/// the oracle of the chunked graph, and equals the sequential `gated_delta_net_decode` recurrence run
/// token by token on the CPU oracle (state threaded across the two calls), which the chunked algebra
/// must reproduce.
#[test]
fn chunked_gdn_prefill_matches_the_sequential_recurrence_on_rocm() {
    use poot_executor_parity::gdn::{D, GdnInputs, H_K, H_V, LEN};
    use poot_graph_ir::ops::gated_delta_net_decode;
    use poot_graph_ir::{Builder, StateRole, TensorType};

    let Some((mut exec, target)) = rocm() else {
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

/// Qwen2.5-0.5B as a decode fixture over its real weights: the registry's qwen2 model over the
/// checkpoint's own config and tensors (24 layers, grouped-query attention with `q/k/v` bias, tied
/// embedding), traced for one decode token, its weights bound by const name as stored (bf16 words
/// for the projections), and one decode step per prompt token.
fn qwen05b_fixture(checkpoint: &std::path::Path, prompt: &[u32]) -> Fixture {
    use poot_executor_parity::dense::{const_named_store, plain, step};
    use poot_executor_parity::weight_map::MappedModel;
    use poot_models::model::{LogitRows, Phase};
    use poot_models::registry::{RawConfig, Registry};

    const CAP: usize = 8;
    const MAX_POS: usize = 64;
    assert!(prompt.len() <= CAP, "the KV cache holds {CAP} positions");
    let stored =
        poot_load::safetensors::load_weight_store(checkpoint).expect("load the checkpoint");
    let mut config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(checkpoint.join("config.json")).expect("read config.json"),
    )
    .expect("parse config.json");
    config["max_position_embeddings"] = MAX_POS.into();
    let model = Registry::builtin()
        .unwrap()
        .build(
            &RawConfig::HfJson {
                config: &config,
                generation: None,
            },
            &stored,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let map = Arc::new(model.weights().clone());
    let mapped = MappedModel {
        model,
        store: Arc::new(stored),
        map,
    };
    let graph = plain(
        mapped
            .model
            .trace(Phase::Decode, step(1, 1, CAP, LogitRows::Last))
            .unwrap(),
    );
    let store = const_named_store(&mapped, &graph);

    let slot = |wanted: poot_graph_ir::Slot| {
        graph
            .inputs
            .iter()
            .map(|&id| graph.meta(id))
            .find(|meta| meta.storage == Storage::Slot(wanted))
            .unwrap_or_else(|| panic!("the decode graph has no {wanted:?} slot"))
    };
    let (token, pos) = (
        slot(poot_graph_ir::Slot::Token),
        slot(poot_graph_ir::Slot::Pos),
    );
    let steps = prompt
        .iter()
        .enumerate()
        .map(|(position, &id)| {
            let bound = |meta: &poot_graph_ir::ValueMeta, value: i32| poot_test_util::StepFixture {
                key: meta.slot_key().unwrap().clone(),
                tensor: HostTensor::i32(meta.aval.shape.clone(), vec![value]),
            };
            vec![bound(token, id as i32), bound(pos, position as i32)]
        })
        .collect();
    Fixture {
        name: "qwen2_5_0_5b",
        graph,
        store,
        steps,
        fusion: poot_graph_plan::FusionPolicy::Full,
    }
}

/// SC-007. Replaces `probe_qwen05b_real_weights_passes_on_strix_halo` and
/// `probe_qwen05b_1layer_decode_passes_on_strix_halo` (layer 0, head 0 and layer 0 only): the whole
/// Qwen2.5-0.5B (24 layers) over its real checkpoint decodes a six-token prompt through the ROCm
/// executor, and the logits of every position (not only the last) match the CPU oracle's evaluation of
/// the same graph, element by element within the parity tolerance, with the same top-1 token at every
/// position.
///
/// Mutation: make `RocmDevice::replay` skip one recorded dispatch (the recording's last item, the
/// unembedding); the logits are never written and the row goes red.
///
/// `#[ignore]`: it loads a real checkpoint (about 1 GB of bf16, 2.4 GB resident beside the oracle's
/// f32 copies), which `just test` hides by design. Run it by exact name with `POOT_MODELS_DIR` set and
/// at least 40Gi available:
/// `flock /tmp/poot-gpu.lock cargo nextest run -p poot-rocm-gpu --test executor_coverage --run-ignored
/// ignored-only -E 'test(=qwen2_5_0_5b_real_weights_match_the_oracle_logits_at_every_prompt_position_on_rocm)'`.
#[test]
#[ignore = "loads the real Qwen2.5-0.5B checkpoint; run by exact name with POOT_MODELS_DIR set and 40Gi available"]
fn qwen2_5_0_5b_real_weights_match_the_oracle_logits_at_every_prompt_position_on_rocm() {
    let Some(checkpoint) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b"))
    else {
        return;
    };
    let Some((mut exec, target)) = rocm() else {
        return;
    };
    let prompt = [785, 6722, 315, 9625, 374, 12095];
    let fixture = qwen05b_fixture(&checkpoint, &prompt);
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let expected: Vec<HostTensor> = fixture.steps.iter().map(|step| oracle(step)).collect();
    let got = poot_executor_parity::run_outputs(exec.as_mut(), target, &fixture)
        .unwrap_or_else(|e| panic!("{e}"));

    let top1 = |logits: &HostTensor| {
        let values = logits.as_f32().expect("F32 logits");
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap()
    };
    assert_eq!(got.len(), prompt.len());
    for (position, (got, want)) in got.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.shape(),
            want.shape(),
            "position {position}: logits shape"
        );
        let (got_values, want_values) = (got.as_f32().unwrap(), want.as_f32().unwrap());
        for (index, (a, b)) in got_values.iter().zip(want_values).enumerate() {
            assert!(
                (a - b).abs() <= 5e-3 * b.abs().max(1.0),
                "position {position} logit {index}: device {a} vs oracle {b}"
            );
        }
        assert_eq!(top1(got), top1(want), "position {position}: top-1 token");
        let spread = want_values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            spread > 1.0,
            "position {position}: the oracle logits are degenerate (max {spread})"
        );
    }
}

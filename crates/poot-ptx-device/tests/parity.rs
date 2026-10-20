//! Card 557 SC-004 on PTX, beside the parity table: the table (`poot-executor-parity`, Card 546a) runs
//! on PTX through the executor contract, mirroring `poot-rocm-device/tests/parity.rs`, and every replayed step (the record-then-replay step and every later
//! pure-replay step, including a repeat of the first step's inputs) matches the CPU oracle within
//! tolerance (ADR-0101 tier 1). Mutation (SC-002): drop one plan-kind arm from `PtxDevice`
//! (e.g. make `copy` a no-op); the fixture needing it goes red. Mutation (SC-003): skip recording in
//! `Engine::step` (always treat the entry as needing a fresh recording); the repeated first step's
//! replay stops matching the oracle it already matched once, since the two recordings would not be
//! identical.

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, Engine, Executor};
use poot_ptx_device::PtxDevice;
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;

/// Card 645 SC-001: `matmul(x, transpose(w))` over an F32 `[N, K]` weight, at `M = 1` (the decode GEMV), `4` and `33`
/// (the tiled GEMM), compiled for PTX matches `eval(g)` of the traced graph, within the parity tolerance
/// (ADR-0101 tier 2), at every replayed step. The oracle evaluates the `Transpose` + `MatMul` the tracer
/// emitted; the device runs the `DenseContraction` the fold made of it, reading the weight as stored.
/// Mutation: in the F32 GEMV body read B as `[K, N]` (`gemv_lds`, `WeightLayout::Nk` arm); the `M = 1` row goes red.
#[test]
fn checkpoint_orientation_projection_matches_the_oracle_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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

/// POOT-1048: a grouped projection `matmul(x, transpose(reshape(w, [G, N, K]), [0, 2, 1]))` over a BF16 `[G * N, K]`
/// const (DeepSeek V4's `wo_a`), at `M = 1`, `5` and `33` with three groups and at one group, compiled for PTX: the
/// program holds one `GroupedDenseContraction` and neither a `Transpose` nor a `Cast`, and every replayed step
/// matches `eval(g)` of the traced graph within the parity tolerance (ADR-0101 tier 2). The weight lives on the
/// device as packed bfloat16 words.
/// Mutation: in `matmul_batched_impl` give the weight no per-group batch offset (zero its `b_beff`); the three-group
/// rows go red.
#[test]
fn bf16_grouped_projection_matches_the_oracle_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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

/// Card 736: the ALiBi attention BLOOM and MPT trace (`alibi_mask_from_pos` over a
/// `ComputedConst::AlibiSlopes` of 6 heads, decode at three positions and a 4-token prefill), compiled for
/// PTX, matches `eval(g)` at every replayed step (ADR-0101 tier 1) AND an f64 reference that writes the
/// published slopes out as literals (tier 2), so a slope formula both the oracle and the device read from
/// the same constant cannot hide. Mutation: in `ComputedConst::tensor` return zero slopes (or drop the
/// odd-index series of the second half); the reference comparison goes red.
#[test]
fn alibi_slopes_attention_matches_the_oracle_and_the_reference_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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
/// `4` and `33` (the tiled GEMM), compiled for PTX: the program
/// holds one `DenseContraction` and no `Transpose`, and every replayed step matches `eval(g)` of the traced graph
/// within the parity tolerance (ADR-0101 tier 2). The weight lives on the device as packed binary16 words, the
/// same mechanism as wgpu and ROCm (no native two-byte F16 body).
/// Mutation: remove `DType::F16` from `DENSE_CONTRACTION_WEIGHT_DTYPES`; a `Transpose` is planned and this row goes
/// red.
#[test]
fn f16_checkpoint_orientation_projection_matches_the_oracle_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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
fn select_is_bitwise_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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
fn two_byte_select_is_bitwise_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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
fn bf16_const_cast_is_bit_exact_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
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

/// Card 663 SC-001, PTX. The batched, multi-block tensor-core matmul: `[2, 32, 48] x [2, 48, 32]` bf16 operands
/// (`poot_executor_parity::launch::wmma_bf16_batched_fixture`), 2 x 2 x 2 output tiles each accumulating three
/// 16-wide K blocks, every output element of both steps compared with the oracle. The fixture values give every
/// K block a distinct nonzero contribution to every output (at least 16 per block), so a dropped, doubled or
/// wrong-batch block moves all outputs far past the parity tolerance.
///
/// Mutation: drop the last K block in `poot-kernelgen`'s `tensor_core_tiles` (the loop bound `k0 < k` becomes
/// `k0 + 16 < k`), with a fresh `POOT_KERNEL_CACHE_DIR`; every output is short by its last block's contribution
/// and the row goes red.
#[test]
fn batched_multi_block_wmma_matmul_matches_the_oracle_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    let fixture = poot_executor_parity::launch::wmma_bf16_batched_fixture();
    let multiplies_on_tensor_cores = poot_executor_parity::staged(&fixture, t)
        .stages()
        .flat_map(|(_, _, program)| {
            program
                .planned()
                .filter_map(|(_, plan)| plan.single().map(|dispatch| dispatch.body().clone()))
                .collect::<Vec<_>>()
        })
        .any(|body| {
            body.blocks
                .iter()
                .flat_map(|block| &block.statements)
                .any(|statement| matches!(statement, poot_kernel_ir::Statement::WmmaMma { .. }))
        });
    assert!(
        multiplies_on_tensor_cores,
        "{}: no dispatched kernel multiplies on the WMMA tensor cores",
        fixture.name
    );
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    poot_executor_parity::run_parity(exec.as_mut(), t, &fixture, &mut oracle)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// POOT-1074: a recurrent state pool addressed by slot (rows in any order, a masked row committing
/// nothing) and `reset_state(Row(n))` zeroing exactly pool row `n`, on PTX; a row outside the pool is a
/// typed `StateRow` refusal that zeroes nothing.
/// The pool runs over F32 and over I32 (the PLE token history), and a `Positional` counter beside it must
/// survive every `Row`; the I32 run puts values above 2^24 in the pool, so it also proves the pool stays on
/// the exact lane. Mutations: write the row at offset 0 instead of `row * bytes` in `Engine::reset_state`
/// (the rows after `Row(2)` continue from slot 2's old state); let `Row` zero every state (the counter
/// resets); drop the fixture's own `b.scatter_update` for the I32 pool (`state_row.rs`; the pool never
/// changes); type the I32 pool through `Ty::F32` in `fty_ty` (`2^24 + 1` reads back as
/// `2^24`). `Step::carry_recurrent`'s I32 path is
/// guarded by `served_qwen4exp_*`, not by this row.
#[test]
fn state_pool_rows_reset_independently_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::state_row::run_state_pool_row_reset(exec.as_mut(), t)
        .unwrap_or_else(|e| panic!("{e}"));
}

/// POOT-1146: the greedy `SampleToken` pick is exact on PTX: over `[rows, vocab]` logits at a vocabulary that
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
fn greedy_sample_token_is_exact_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::sample_token::assert_greedy_pick_is_exact(exec.as_mut(), t);
}

/// POOT-1268: the sampler suffix that adjusts the logits (penalties, a guided mask), draws, records top-n
/// logprobs and carries the per-sequence history runs on PTX: every step's record equals the CPU oracle's
/// (token and flag columns exactly, logprobs within the tier-2 tolerance), with the history pool the first
/// steps wrote changing the next steps' picks.
/// Mutation (fresh `POOT_KERNEL_CACHE_DIR`): in `poot-graph-plan`'s `planner/sampling.rs` plan the greedy
/// pick with `groups = vocab / ARGMAX_CHUNK` (the partial last chunk dropped); a winner or an alternative
/// in the last 176 tokens of a row goes missing and the record differs from the oracle's.
#[test]
fn adjusted_sampling_head_matches_the_oracle_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    poot_executor_parity::device_sampling::assert_matches_the_oracle(exec.as_mut(), t);
}

/// POOT-1143 stage 2: the indexer's block selection (ties straddling the cut, fewer complete blocks than `k`,
/// queries as rows by tokens, a wide row, all scores equal) runs through the generated radix-select kernel
/// on the device, and its words equal both the plain stable-sort selection (computed here without the oracle)
/// and the oracle's primitive definition at every replayed step.
/// Mutation: in `block_top_k.rs` take the last equal blocks at the threshold instead of the first (ties
/// ascending by block); the ties case goes red.
#[test]
fn block_top_k_selects_what_the_plain_selection_selects_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for case in poot_executor_parity::block_top_k::cases() {
        poot_executor_parity::block_top_k::assert_device_selects(exec.as_mut(), t, &case);
        if case.oracle {
            let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
            poot_executor_parity::run_parity(exec.as_mut(), t, &case.fixture, &mut oracle)
                .unwrap_or_else(|e| panic!("{e}"));
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
fn selected_attention_reads_exactly_the_selected_blocks_on_ptx() {
    let Some(device) = open_or_skip(DeviceBackend::Ptx, PtxDevice::new()) else {
        return;
    };
    let t = device.target();
    let mut exec: Box<dyn Executor> = Box::new(Engine::new(device));
    for case in poot_executor_parity::selected_attention::all() {
        poot_executor_parity::selected_attention::assert_device_attends(exec.as_mut(), t, &case);
        if case.oracle {
            let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
            poot_executor_parity::run_parity(exec.as_mut(), t, &case.fixture, &mut oracle)
                .unwrap_or_else(|e| panic!("{e}"));
        }
    }
}

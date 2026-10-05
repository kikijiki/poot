use super::*;

/// The fake `admit` used by the `drain_ready_jobs` tests: fills the first free slot with the job id,
/// like the real `admit` closures (one fill per call on the success path).
fn admit_into_first_free(slots: &mut [Option<u32>], job: u32) {
    let free = slots.iter().position(Option::is_none).expect("a free slot");
    slots[free] = Some(job);
}

/// `drain_ready_jobs` runs the engine loops' non-blocking intake: with 5 ready jobs and 3 free
/// slots it must admit exactly 3, in arrival order, and leave 2 queued behind an open input.
#[test]
fn drain_ready_jobs_never_admits_more_than_free_slots() {
    let (tx, rx) = mpsc::channel();
    for job in 0..5u32 {
        tx.send(job).expect("queue job");
    }
    let mut slots: Vec<Option<u32>> = vec![None; 3];
    let mut input_closed = false;

    drain_ready_jobs(&rx, &mut slots, &mut input_closed, admit_into_first_free);

    assert_eq!(slots, vec![Some(0), Some(1), Some(2)]);
    assert!(!input_closed, "a live sender is never reported closed");
    assert_eq!(rx.try_recv().expect("job 3 is still queued"), 3);
    assert_eq!(rx.try_recv().expect("job 4 is still queued"), 4);
    assert!(matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)));
}

/// On disconnect the helper must drain what was already queued first, then report the closed input:
/// with a spare free slot left after the drain, all 3 jobs land and `input_closed` flips in the same
/// call. (A drain that fills the last free slot exits on "full" without probing the disconnect; the
/// flag is set on a later call once a free slot exists again - the engine loops' idle check.)
#[test]
fn drain_ready_jobs_reports_closed_on_disconnect_after_draining() {
    let (tx, rx) = mpsc::channel();
    for job in 0..3u32 {
        tx.send(job).expect("queue job");
    }
    drop(tx);
    let mut slots: Vec<Option<u32>> = vec![None; 4];
    let mut input_closed = false;

    drain_ready_jobs(&rx, &mut slots, &mut input_closed, admit_into_first_free);

    assert_eq!(
        slots,
        vec![Some(0), Some(1), Some(2), None],
        "every queued job must be admitted before the close is reported"
    );
    assert!(input_closed, "a disconnected input must be reported closed");
    assert!(
        matches!(rx.try_recv(), Err(mpsc::TryRecvError::Disconnected)),
        "the queue must be fully drained"
    );
}

/// A drain that fills the last free slot exits on "full" without probing the input, so the
/// disconnect is not reported yet; it is reported on the next call, once a slot is free again and
/// `try_recv` can observe the empty disconnected queue.
#[test]
fn drain_ready_jobs_defers_disconnect_until_a_slot_is_free() {
    let (tx, rx) = mpsc::channel();
    for job in 0..3u32 {
        tx.send(job).expect("queue job");
    }
    drop(tx);
    let mut slots: Vec<Option<u32>> = vec![None; 3];
    let mut input_closed = false;

    drain_ready_jobs(&rx, &mut slots, &mut input_closed, admit_into_first_free);
    assert_eq!(slots, vec![Some(0), Some(1), Some(2)]);
    assert!(
        !input_closed,
        "the loop exited on a full slot set, before it could observe the disconnect"
    );

    slots[0] = None;
    drain_ready_jobs(&rx, &mut slots, &mut input_closed, admit_into_first_free);
    assert!(
        input_closed,
        "with a free slot again, the empty disconnected queue must be reported"
    );
}

/// An `Empty` read leaves the input open: the first call on a live empty queue admits nothing, and a
/// later partial drain (2 jobs into 3 free slots) ends on `Empty` with the flag still clear.
#[test]
fn drain_ready_jobs_keeps_a_live_empty_queue_open() {
    let (tx, rx) = mpsc::channel();
    let mut slots: Vec<Option<u32>> = vec![None; 3];
    let mut input_closed = false;

    drain_ready_jobs(&rx, &mut slots, &mut input_closed, admit_into_first_free);
    assert_eq!(slots, vec![None, None, None]);
    assert!(!input_closed);

    for job in 0..2u32 {
        tx.send(job).expect("queue job");
    }
    drain_ready_jobs(&rx, &mut slots, &mut input_closed, admit_into_first_free);
    assert_eq!(slots, vec![Some(0), Some(1), None]);
    assert!(
        !input_closed,
        "a live partial drain must not close the input"
    );
}

/// The first shortfall ends the step: nothing installed, and the queue keeps its exact front-to-back
/// order (`[10, 20, 30]`, so a drop or a requeue-behind is visible). The attempt budget inside the
/// closure makes a helper that keeps spinning after the shortfall fail with a named assertion
/// instead of hanging the test process.
#[test]
fn resume_into_free_slots_stops_at_the_first_shortfall_and_requeues_at_the_front() {
    let mut slots: Vec<Option<u32>> = vec![None; 3];
    let mut preempted: VecDeque<u32> = VecDeque::from([10, 20, 30]);
    let mut attempts = 0;

    resume_into_free_slots(&mut slots, &mut preempted, |_free_j, seq| {
        attempts += 1;
        assert!(
            attempts <= 1,
            "resume attempted {attempts} times after the first shortfall; the step must break"
        );
        Err(seq)
    });

    assert_eq!(
        attempts, 1,
        "exactly one attempt runs before the shortfall stops the step"
    );
    assert_eq!(
        slots,
        vec![None, None, None],
        "a shortfall installs nothing"
    );
    assert_eq!(
        preempted.into_iter().collect::<Vec<_>>(),
        vec![10, 20, 30],
        "the shortfalling sequence returns to the FRONT with the queue order preserved"
    );
}

/// Successes fill the free indices in queue order (`10` then `20`, never `30` first), an occupied
/// slot is untouched, and the backlog (`30`) survives behind the now-full slot set.
#[test]
fn resume_into_free_slots_fills_free_slots_in_queue_order_and_leaves_the_backlog() {
    let mut slots: Vec<Option<u32>> = vec![None, None, Some(99)];
    let mut preempted: VecDeque<u32> = VecDeque::from([10, 20, 30]);
    let mut attempts = 0;

    resume_into_free_slots(&mut slots, &mut preempted, |_free_j, seq| {
        attempts += 1;
        assert!(
            attempts <= 2,
            "resume attempted {attempts} times with only 2 free slots; the step must stop when full"
        );
        Ok(seq)
    });

    assert_eq!(slots, vec![Some(10), Some(20), Some(99)]);
    assert_eq!(
        preempted.into_iter().collect::<Vec<_>>(),
        vec![30],
        "the unresumed backlog stays queued in order"
    );
}

#[test]
fn kv_fit_reduces_only_when_over_budget() {
    // per-(slot,token) KV cost = 2*layers*kv_heads*head_dim*dtype. Use a round 1000 B/slot-token, 4 slots
    // -> 4000 B per +1 cap.
    let bpt = 1000u64;
    // fits: requested 1024 cap needs 4000*1024 = 4.096 MB; budget 8 MB -> unchanged.
    assert_eq!(fit_kv_to_budget(4, 1024, bpt, 8_000_000, 64), (4, 1024));
    // over budget: 2 MB budget / 4000 = 500 -> reduce cap, keep slots.
    assert_eq!(fit_kv_to_budget(4, 1024, bpt, 2_000_000, 64), (4, 500));
    // tight: even min_cap (64) across 4 slots = 256 KB; with 100 KB budget that overflows, so drop slots.
    // 100 KB / (1000*64) = 1 slot at cap 64.
    assert_eq!(fit_kv_to_budget(4, 1024, bpt, 100_000, 64), (1, 64));
    // mid-tight: 200 KB / (1000*64)=3 slots fit at the cap floor.
    assert_eq!(fit_kv_to_budget(4, 1024, bpt, 200_000, 64), (3, 64));
    // unknown budget / cost -> never touch the requested config.
    assert_eq!(fit_kv_to_budget(4, 1024, bpt, 0, 64), (4, 1024));
    assert_eq!(fit_kv_to_budget(4, 1024, 0, 2_000_000, 64), (4, 1024));
    // never INCREASES a small requested cap / slot count even with a huge budget.
    assert_eq!(fit_kv_to_budget(4, 256, bpt, 8_000_000_000, 64), (4, 256));
}

/// `schedule_prefill_admissions` is the pure `max_num_batched_tokens` decision behind `batch_engine_loop`'s fast paged prefill step (whole-admission granularity; deferred slots retry next iteration).
#[test]
fn schedule_prefill_admissions_admits_everything_that_fits() {
    // budget covers both pending admissions with room to spare -> both admitted, in order.
    let pending = [(0usize, 10usize), (1, 20)];
    assert_eq!(schedule_prefill_admissions(&pending, 0, 100), vec![0, 1]);
}

#[test]
fn schedule_prefill_admissions_defers_what_does_not_fit() {
    // budget only covers the first admission; the second is deferred (not admitted this iteration).
    let pending = [(0usize, 10usize), (1, 20)];
    assert_eq!(schedule_prefill_admissions(&pending, 0, 15), vec![0]);
}

#[test]
fn schedule_prefill_admissions_reserves_budget_for_active_decode_first() {
    // 3 active decode slots reserve 3 tokens off a budget of 10, leaving 7 - a prompt needing 8 tokens no
    // longer fits even though the raw budget looks big enough.
    let pending = [(0usize, 8usize)];
    assert_eq!(
        schedule_prefill_admissions(&pending, 3, 10),
        Vec::<usize>::new()
    );
    // one token less and it fits exactly.
    let pending = [(0usize, 7usize)];
    assert_eq!(schedule_prefill_admissions(&pending, 3, 10), vec![0]);
}

#[test]
fn schedule_prefill_admissions_active_decode_can_exhaust_the_whole_budget() {
    // decode reservation alone can consume (or exceed, via saturating_sub) the entire budget - no pending
    // admission is ever selected in that case, but the function must not panic/underflow.
    let pending = [(0usize, 1usize)];
    assert_eq!(
        schedule_prefill_admissions(&pending, 50, 10),
        Vec::<usize>::new()
    );
}

#[test]
fn schedule_prefill_admissions_is_best_effort_bin_packing_not_a_fifo_cutoff() {
    // A big admission at the front that does not fit must not block a smaller one behind it (strict FIFO would starve short requests; vLLM chunked prefill packs best-effort).
    let pending = [(0usize, 90usize), (1, 5), (2, 3)];
    assert_eq!(schedule_prefill_admissions(&pending, 0, 10), vec![1, 2]);
}

#[test]
fn schedule_prefill_admissions_empty_pending_is_a_no_op() {
    assert_eq!(
        schedule_prefill_admissions(&[], 0, 100),
        Vec::<usize>::new()
    );
}

/// `fit_kv_to_budget` is fed the per-slot-token byte cost `2*layers*kv_heads*head_dim*dtype_bytes * KV_POOL_VRAM_MULTIPLIER` (3) by `batch_engine_loop` (card 159).
/// This repeats that arithmetic (the constant is private) against gemma4-31B-like dims, so a dropped multiplier or an undercounting derivation fails here rather than at GPU allocation.
///
/// Dims: 60 layers, 5:1 local/global schedule (`sliding_window_pattern` of the 31B GGUF), local head_dim=256/n_kv_heads=16, global head_dim=512/n_kv_heads=4 (the dims of the deleted Gemma4 runtime's `kv_cost_dims`, card 595).
/// `sum(n_kv_heads*head_dim)` = 50*16*256 + 10*4*512 = 225_280; raw per-slot-token = `2 * 225_280 * 4B(f32)` = 1_802_240 B (~1.72 MiB).
/// `weight_bytes` ~19 GiB and `vram_budget_bytes` ~37.5 GiB mirror the diagnosed box: `kv_budget = 0.85*budget - weight` ~= 12.9 GiB.
#[test]
fn kv_fit_prices_the_honest_3x_shared_pool_footprint_gemma4_31b() {
    let raw_bpt = 2 * 225_280 * 4; // 2 (k+v) * sum(n_kv_heads*head_dim) * f32 dtype bytes
    assert_eq!(raw_bpt, 1_802_240);
    let honest_bpt = raw_bpt as u64 * KV_POOL_VRAM_MULTIPLIER;

    let weight = 19 * 1024 * 1024 * 1024u64;
    let budget = 37_500 * 1024 * 1024u64; // ~37.5 GiB, the diagnosed free-VRAM figure
    let kv_budget = ((budget as f64 * 0.85) as u64).saturating_sub(weight);

    // n_slots=1 (known-good): cap=1024 stays affordable at the 3x price; the single-slot path must not be over-shrunk.
    assert_eq!(
        fit_kv_to_budget(1, 1024, honest_bpt, kv_budget, 64),
        (1, 1024)
    );
    // n_slots=2: still fits at cap=1024 (the diagnosed borderline was ~30 GiB against ~37.5 GiB free); hardware verification of this bracket is deferred (card 159).
    assert_eq!(
        fit_kv_to_budget(2, 1024, honest_bpt, kv_budget, 64),
        (2, 1024)
    );
    // n_slots=4 (the reported OOM config): cap=1024 does not fit, so cap is reduced and the batch width kept.
    let (fit_slots, fit_cap) = fit_kv_to_budget(4, 1024, honest_bpt, kv_budget, 64);
    assert_eq!(
        fit_slots, 4,
        "batch width should be preserved when only cap needs to shrink"
    );
    assert!(
        fit_cap < 1024,
        "n_slots=4 must be reduced from the OOMing cap=1024, got cap={fit_cap}"
    );
    // the fitted config's total footprint (weights + 3x KV pool) must clear the raw `vram_budget_bytes`, not just the 0.85-derived `kv_budget`.
    let fitted_kv_bytes = honest_bpt * 4 * fit_cap as u64;
    assert!(
        weight + fitted_kv_bytes < budget,
        "fitted config ({fit_slots},{fit_cap}) honest footprint {} B must fit under budget {budget} B",
        weight + fitted_kv_bytes
    );
    // at the same fitted cap the old 1x pricing under-prices enough that the original (4,1024) looked affordable: the raw formula says "fits", the 3x price says it does not.
    let (bug_slots, bug_cap) = fit_kv_to_budget(4, 1024, raw_bpt as u64, kv_budget, 64);
    assert_eq!(
        (bug_slots, bug_cap),
        (4, 1024),
        "sanity: the pre-fix 1x pricing must reproduce the diagnosed under-count (returns the request \
             unchanged, which is what let the real allocation OOM)"
    );
}

/// Card 030 follow-on: `batch_engine_loop_ptx`, `batch_engine_loop_rocm` and `batch_engine_loop_rocm_runner` price `fit_kv_to_budget`'s per-slot-token cost at `KV_POOL_VRAM_MULTIPLIER_INPLACE` (2x), not wgpu's `KV_POOL_VRAM_MULTIPLIER` (3x): PTX/ROCm bind `state_out` in place onto `state_in`, so there is no ping-pong twin buffer (see the constant in `batch.rs`).
/// This repeats the arithmetic of those call sites (`2 * layers * kv_heads * head_dim * dtype_bytes * KV_POOL_VRAM_MULTIPLIER_INPLACE`, then `fit_kv_to_budget`), so dropping to 1x (undercount, OOM risk) or reusing wgpu's 3x (over-shrink) fails here.
#[test]
fn kv_fit_ptx_rocm_prices_the_honest_2x_inplace_footprint() {
    assert_eq!(KV_POOL_VRAM_MULTIPLIER_INPLACE, 2);

    // qwen2.5-0.5b-like dims: 24 layers, 2 kv_heads, 64 head_dim (GQA), f32 KV.
    let raw_bpt = 2 * 24 * 2 * 64 * 4; // 2(k+v) * layers * kv_heads * head_dim * f32 dtype bytes
    assert_eq!(raw_bpt, 24_576);
    let inplace_bpt = raw_bpt as u64 * KV_POOL_VRAM_MULTIPLIER_INPLACE;
    assert_eq!(inplace_bpt, 49_152);

    let weight = 1024u64 * 1024 * 1024; // ~1 GiB weights (small model)
    let budget = 4096u64 * 1024 * 1024; // ~4 GiB budget (a small/contended device)
    let kv_budget = ((budget as f64 * 0.85) as u64).saturating_sub(weight);

    // n_slots=64, cap=2048: too large for this budget at either pricing - cap must shrink (batch
    // width preserved), same reduce-cap-first contract `fit_kv_to_budget` documents.
    let (fit_slots, fit_cap) = fit_kv_to_budget(64, 2048, inplace_bpt, kv_budget, 64);
    assert_eq!(
        fit_slots, 64,
        "batch width should be preserved when only cap needs to shrink"
    );
    assert_eq!(
        fit_cap, 819,
        "regression guard on the exact fitted cap at this budget/dims"
    );

    // the fitted config's total footprint (weights + 2x KV pool) clears the raw budget, not just the 0.85-derived `kv_budget`.
    let fitted_kv_bytes = inplace_bpt * fit_slots as u64 * fit_cap as u64;
    assert!(
        weight + fitted_kv_bytes < budget,
        "fitted config ({fit_slots},{fit_cap}) honest footprint {} B must fit under budget {budget} B",
        weight + fitted_kv_bytes
    );

    // wgpu's 3x pricing at the same budget/dims must fit a smaller or equal cap: the 2x price covers strictly less VRAM per token, so it can only relax the fit.
    let wgpu_bpt = raw_bpt as u64 * KV_POOL_VRAM_MULTIPLIER;
    let (_, fit_cap_wgpu) = fit_kv_to_budget(64, 2048, wgpu_bpt, kv_budget, 64);
    assert_eq!(
        fit_cap_wgpu, 546,
        "regression guard on wgpu's own 3x fitted cap for comparison"
    );
    assert!(
        fit_cap >= fit_cap_wgpu,
        "PTX/ROCm's 2x in-place pricing (cap={fit_cap}) must allow AT LEAST as large a cap as wgpu's \
         3x pricing (cap={fit_cap_wgpu}) at the identical budget/dims"
    );

    // a request that already fits is left unchanged (the fit is reduce-only).
    assert_eq!(
        fit_kv_to_budget(2, 64, inplace_bpt, kv_budget, 64),
        (2, 64),
        "a request that already fits must be returned unchanged (reduce-only, never increases)"
    );

    // unknown budget (0, i.e. `vram_budget_bytes()` returning `None` so the call site skips the fit) is covered by the `kv_fit_to_budget_*` tests above.
}

/// Card 267: `batch_engine_loop_ptx`'s fit must also price the attention broadcast buffers, not only the persistent KV pool (`kv_fit_ptx_rocm_prices_the_honest_2x_inplace_footprint`).
/// `attention_masked_softcap`'s GQA `repeat_kv(k)`/`repeat_kv(v)`/`transpose(k)` chain (`crates/poot-graph-ir/src/ops.rs`) always materializes a `[n_slots, n_heads, cap, head_dim]` f32 buffer (`MatMul` is never `is_strided_capable`, so `compute_views` never makes it a free view), and `capture_decode_paged`'s warmup+capture never frees a dead buffer (CUDA-graph capture cannot `cuMemFree`), so all `layers` copies of all 3 buffers stay resident.
///
/// This repeats the additive arithmetic of the call site (`3 * layers * n_heads * head_dim * dtype_bytes` on top of the KV-pool price) with the numbers a rented RTX 4090 reported: qwen2.5-0.5b-instruct dims (24 layers, 14 query heads, 2 kv heads, head_dim 64, GQA n_rep=7), `POOT_SLOTS=8 POOT_CAP=222712`, `vram_budget_mib=24081`, `weight_mib=2419`, which reproduced `CUDA_ERROR_OUT_OF_MEMORY` in `capture_decode_paged`.
#[test]
fn kv_fit_ptx_prices_the_honest_attn_broadcast_footprint_card267() {
    // the KV-pool-only price; this is what OOM'd.
    let kv_pool_bpt = 2 * 24 * 2 * 64 * 4 * KV_POOL_VRAM_MULTIPLIER_INPLACE as usize;
    assert_eq!(kv_pool_bpt, 49_152);

    // n_rep = n_heads / n_kv_heads = 14/2 = 7 > 1, so repeat_kv(k)/repeat_kv(v)/transpose(k) all materialize (3 * layers * n_heads * head_dim * dtype_bytes).
    let n_heads = 14;
    let n_kv_heads = 2;
    let n_rep = n_heads / n_kv_heads;
    assert_eq!(n_rep, 7);
    let attn_broadcast_bpt = 3 * 24 * n_heads * 64 * 4;
    assert_eq!(attn_broadcast_bpt, 258_048);
    let honest_bpt = (kv_pool_bpt + attn_broadcast_bpt) as u64;
    assert_eq!(honest_bpt, 307_200);

    // the pod's reported numbers: vram_budget_mib=24081, weight_mib=2419.
    let budget = 24_081u64 * 1024 * 1024;
    let weight = 2_419u64 * 1024 * 1024;
    let kv_budget = ((budget as f64 * 0.85) as u64).saturating_sub(weight);

    // the KV-pool-only price shrinks cap but not enough; pins the fitted cap that OOM'd on hardware.
    let (bug_slots, bug_cap) = fit_kv_to_budget(8, 222_712, kv_pool_bpt as u64, kv_budget, 64);
    assert_eq!(
        (bug_slots, bug_cap),
        (8, 48_132),
        "sanity: the pre-fix KV-pool-only price must reproduce the exact real-hardware fitted cap that \
         actually OOM'd (docs/updates/0718) - this is the bug this test guards against reintroducing"
    );

    // the full price shrinks cap much further: the fitted config's capture-resident footprint (weights + KV pool + broadcast) must clear the raw budget.
    let (fit_slots, fit_cap) = fit_kv_to_budget(8, 222_712, honest_bpt, kv_budget, 64);
    assert_eq!(fit_slots, 8, "batch width should be preserved");
    assert!(
        fit_cap < bug_cap,
        "the honest price must shrink cap further than the pre-fix bug did: fit_cap={fit_cap} bug_cap={bug_cap}"
    );
    let fitted_bytes = honest_bpt * 8 * fit_cap as u64;
    assert!(
        weight + fitted_bytes < budget,
        "fitted config (8,{fit_cap}) honest footprint {} B must fit under budget {budget} B",
        weight + fitted_bytes
    );
    // the 3 materialized attention buffers at the KV-pool-only fitted cap would have needed more than the whole card.
    let attn_bytes_at_bug_cap = attn_broadcast_bpt as u64 * 8 * bug_cap as u64;
    assert!(
        attn_bytes_at_bug_cap > budget,
        "regression guard: the unpriced attention-broadcast footprint at the pre-fix fitted cap ({bug_cap}) \
         must exceed the WHOLE card's VRAM budget ({budget} B), confirming that config really did OOM \
         (got {attn_bytes_at_bug_cap} B)"
    );

    // Card 267 calibration: the unmargined price above still OOM'd on the RTX 4090 (see `PTX_CAPTURE_MARGIN_NUM`/`_DEN` in `batch_ptx.rs`); real usage ran ~20-40% over the closed-form price.
    // The call site applies a flat 3/2 margin; this checks the margined price against the same numbers and the two caps (8x5134, 8x5924) hand-verified to complete a real request on that card.
    let margined_bpt = honest_bpt * 3 / 2;
    assert_eq!(margined_bpt, 460_800);
    let (margined_slots, margined_cap) = fit_kv_to_budget(8, 222_712, margined_bpt, kv_budget, 64);
    assert_eq!(margined_slots, 8);
    assert!(
        margined_cap < fit_cap,
        "the margin must shrink cap further than the honest (unmargined) price alone: \
         margined_cap={margined_cap} honest_cap={fit_cap}"
    );
    assert!(
        (5_134..=5_924).contains(&margined_cap),
        "regression guard: the margined fit at these real pod numbers should land in the exact range \
         (8x5134..8x5924) this round hand-verified completes a real request on a rented RTX 4090 with no \
         OOM - got cap={margined_cap} (if this moves, re-verify on real hardware before trusting it)"
    );
}

/// Card 267-class fix on ROCm: `batch_engine_loop_rocm`/`batch_engine_loop_rocm_runner` had the same unpriced attention-broadcast gap that `kv_fit_ptx_prices_the_honest_attn_broadcast_footprint_card267` guards on PTX.
/// `compute_views` runs for `Backend::AmdGcn(_)`, but `MatMul` is absent from `is_strided_capable` on every backend, so the `repeat_kv`/`transpose` chain never becomes a free view, and `RocmGraphExecutor::capture_decode_batched` retains every buffer `exec_eqns` allocates for the captured graph's lifetime (no `free_dead`).
///
/// Same dims as `kv_fit_ptx_rocm_prices_the_honest_2x_inplace_footprint` (24 layers, 14 query heads, 2 kv heads, head_dim 64, n_rep=7); repeats the additive arithmetic of the ROCm call sites.
/// Unlike PTX, no calibrated margin is applied (no real ROCm measurement of the fitted boundary yet), so this guards only the structural unmargined term.
#[test]
fn kv_fit_rocm_prices_the_honest_attn_broadcast_footprint_card267() {
    // the KV-pool-only price, same shape as PTX.
    let kv_pool_bpt = 2 * 24 * 2 * 64 * 4 * KV_POOL_VRAM_MULTIPLIER_INPLACE as usize;
    assert_eq!(kv_pool_bpt, 49_152);

    // n_rep = n_heads / n_kv_heads = 14/2 = 7 > 1, so all 3 buffers materialize.
    let n_heads = 14;
    let n_kv_heads = 2;
    let n_rep = n_heads / n_kv_heads;
    assert_eq!(n_rep, 7);
    let attn_broadcast_bpt = 3 * 24 * n_heads * 64 * 4;
    assert_eq!(attn_broadcast_bpt, 258_048);
    let honest_bpt = (kv_pool_bpt + attn_broadcast_bpt) as u64;
    assert_eq!(honest_bpt, 307_200);

    // a small/contended-device budget (not this box's `vram_budget_bytes()`, which reports the whole ~124 GiB unified pool); only the arithmetic shape is checked.
    let budget = 8192u64 * 1024 * 1024; // 8 GiB
    let weight = 512u64 * 1024 * 1024; // ~512 MiB weights (qwen2.5-0.5b-like)
    let kv_budget = ((budget as f64 * 0.85) as u64).saturating_sub(weight);

    // KV-pool-only price: shrinks cap but not enough, since it omits the attention-broadcast buffers.
    let (bug_slots, bug_cap) = fit_kv_to_budget(16, 32_768, kv_pool_bpt as u64, kv_budget, 64);
    assert_eq!(bug_slots, 16, "batch width preserved when only cap shrinks");
    assert!(
        bug_cap < 32_768,
        "the pre-fix price must still shrink SOME cap at this budget"
    );

    // the full price shrinks cap further; the fitted config's capture-resident footprint (weights + KV pool + broadcast) clears the raw budget.
    let (fit_slots, fit_cap) = fit_kv_to_budget(16, 32_768, honest_bpt, kv_budget, 64);
    assert_eq!(fit_slots, 16, "batch width preserved when only cap shrinks");
    assert!(
        fit_cap < bug_cap,
        "the honest price must shrink cap further than the pre-fix bug did: fit_cap={fit_cap} bug_cap={bug_cap}"
    );
    let fitted_bytes = honest_bpt * 16 * fit_cap as u64;
    assert!(
        weight + fitted_bytes < budget,
        "fitted config (16,{fit_cap}) honest footprint {} B must fit under budget {budget} B",
        weight + fitted_bytes
    );

    // the 3 materialized attention buffers at the KV-pool-only fitted cap would have needed more than the device budget, the failure card 267 found on PTX (structural only; no ROCm hardware receipt for this number).
    let attn_bytes_at_bug_cap = attn_broadcast_bpt as u64 * 16 * bug_cap as u64;
    assert!(
        attn_bytes_at_bug_cap > budget,
        "regression guard: the unpriced attention-broadcast footprint at the pre-fix fitted cap ({bug_cap}) \
         must exceed the whole device's VRAM budget ({budget} B) - the same class of real failure card 267 \
         found on PTX (got {attn_bytes_at_bug_cap} B)"
    );
}

#[test]
fn kv_fit_rocm_prices_the_honest_shared_pool_unroll_footprint_card269() {
    // Card 269 guard, with the numbers from the MI300X receipt: `POOT_SLOTS=64 POOT_CAP=32768`, `vram_budget_mib=196592`, `weight_mib=2937`.
    // The card 267 pricing (KV pool + attn broadcast) picked `fitted_cap=8755`, which hit `HSA_STATUS_ERROR_OUT_OF_RESOURCES` in capture (unpriced per-row-unrolled intermediates of `scatter_shared_pool`/`gather_shared_pool`; see `ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN`).
    // This pins that the old price gives `fitted_cap=8755` and the new one gives `fitted_cap=688`, which completed capture cleanly on the same MI300X (`n_slots=64 cap=688`).
    let budget = 196_592u64 * 1024 * 1024;
    let weight = 2_937u64 * 1024 * 1024;
    let kv_budget = ((budget as f64 * 0.85) as u64).saturating_sub(weight);
    let n_slots = 64;
    let requested_cap = 32_768;

    let kv_pool_bpt = 2 * 24 * 2 * 64 * 4 * KV_POOL_VRAM_MULTIPLIER_INPLACE;
    let attn_broadcast_bpt = 3 * 24 * 14 * 64 * 4; // n_rep = 14/2 = 7 > 1, all 3 buffers materialize.
    let pre269_bpt = kv_pool_bpt + attn_broadcast_bpt;
    let (pre269_slots, pre269_cap) =
        fit_kv_to_budget(n_slots, requested_cap, pre269_bpt, kv_budget, 64);
    assert_eq!(
        pre269_slots, n_slots,
        "batch width preserved when only cap shrinks"
    );
    assert_eq!(
        pre269_cap, 8_755,
        "pre-fix price must reproduce the exact diagnosed fitted_cap that OOM'd on real MI300X hardware"
    );

    let shared_pool_unroll_bpt =
        24 * ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN;
    let post269_bpt = pre269_bpt + shared_pool_unroll_bpt;
    let (post269_slots, post269_cap) =
        fit_kv_to_budget(n_slots, requested_cap, post269_bpt, kv_budget, 64);
    assert_eq!(
        post269_slots, n_slots,
        "batch width preserved when only cap shrinks"
    );
    assert_eq!(
        post269_cap, 688,
        "post-fix price must reproduce the exact fitted_cap this round hand-verified completes capture \
         cleanly on real MI300X hardware (n_slots=64 cap=688, no HSA_STATUS_ERROR_OUT_OF_RESOURCES)"
    );
    assert!(
        post269_cap < pre269_cap,
        "the honest price must shrink cap further than the pre-269 price did"
    );
}

/// wgpu's `batch_engine_loop` traces the same per-row-unrolled `scatter_shared_pool`/`gather_shared_pool` ops as ROCm card 269, and `GpuExecutor::build_decode_cache`/`materialize` (`crates/poot-gpu/src/lib.rs`) keep a persistent `DeviceBuffer` per intermediate in `decode_cache` (see `WGPU_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN`).
/// No wgpu hardware calibration exists, so ROCm's constant is reused by shape analogy: an estimate, not a measurement.
///
/// qwen2.5-0.5b-like dims (24 layers, 2 kv_heads, 64 head_dim, f32 KV) as in `kv_fit_ptx_rocm_prices_the_honest_2x_inplace_footprint`, with a small/contended-device budget; only the arithmetic shape is checked.
#[test]
fn kv_fit_wgpu_prices_the_honest_shared_pool_unroll_footprint_card269() {
    assert_eq!(
        WGPU_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN,
        ROCM_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN,
        "wgpu's constant is a documented direct port of ROCm's - a divergence here is either an intentional \
         wgpu-specific calibration (update the doc comments to explain why) or an accidental edit"
    );

    let kv_layers = 24usize;
    let kv_n_kv_heads = 2usize;
    let kv_head_dim = 64usize;
    let kv_dtype_bytes = 4usize; // f32

    // pre-fix price: wgpu's 3x formula (base pool + ping-pong twin + final gather readback).
    let pre_bpt = (2 * kv_layers * kv_n_kv_heads * kv_head_dim * kv_dtype_bytes) as u64
        * KV_POOL_VRAM_MULTIPLIER;
    assert_eq!(pre_bpt, 73_728);

    let budget = 8192u64 * 1024 * 1024; // 8 GiB, a small/contended device
    let weight = 512u64 * 1024 * 1024; // ~512 MiB weights (qwen2.5-0.5b-like)
    let kv_budget = ((budget as f64 * 0.85) as u64).saturating_sub(weight);

    let (bug_slots, bug_cap) = fit_kv_to_budget(16, 32_768, pre_bpt, kv_budget, 64);
    assert_eq!(bug_slots, 16, "batch width preserved when only cap shrinks");
    assert!(
        bug_cap < 32_768,
        "the pre-fix price must still shrink SOME cap at this budget"
    );

    // post-fix price: adds the per-row-unroll term, as in `batch_engine_loop`'s fit block.
    let shared_pool_unroll_bpt =
        kv_layers as u64 * WGPU_SHARED_POOL_UNROLL_OVERHEAD_BYTES_PER_LAYER_PER_SLOT_TOKEN;
    let post_bpt = pre_bpt + shared_pool_unroll_bpt;
    let (fit_slots, fit_cap) = fit_kv_to_budget(16, 32_768, post_bpt, kv_budget, 64);
    assert_eq!(fit_slots, 16, "batch width preserved when only cap shrinks");
    assert!(
        fit_cap < bug_cap,
        "the honest price must shrink cap further than the pre-fix price did: fit_cap={fit_cap} bug_cap={bug_cap}"
    );
    let fitted_bytes = post_bpt * 16 * fit_cap as u64;
    assert!(
        weight + fitted_bytes < budget,
        "fitted config (16,{fit_cap}) honest footprint {} B must fit under budget {budget} B",
        weight + fitted_bytes
    );

    // the unpriced per-row-unroll footprint at the pre-fix fitted cap alone exceeds the whole device budget, as on ROCm (card 269).
    let unroll_bytes_at_bug_cap = shared_pool_unroll_bpt * 16 * bug_cap as u64;
    assert!(
        unroll_bytes_at_bug_cap > budget,
        "regression guard: the unpriced shared-pool-unroll footprint at the pre-fix fitted cap ({bug_cap}) \
         must exceed the whole device's VRAM budget ({budget} B) - the same class of real failure card 269 \
         found on ROCm (got {unroll_bytes_at_bug_cap} B)"
    );
}

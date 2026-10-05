// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! Update 0593 follow-on (epic 129 B7): isolates whether the ROCm MoE shared-pool batched decode
//! divergence found in update 0593 lives in
//! `RocmGraphExecutor::capture_decode_batched`/`decode_step_batched` (the dispatch primitives) or in
//! `batch_engine_loop_rocm_runner`'s (`poot-serve`) per-step bookkeeping around them (job scheduling,
//! token/position/mask construction, the prefill-vs-decode transition, spec/chunk pre-passes).
//!
//! Structurally identical to `card234_rocm_batched_decode_gpu.rs`'s
//! `card234_rocm_batched_decode_two_slots_match_single_seq` (the qwen2/llama test that verified these
//! primitives at `n_slots > 1` on ROCm with no `poot-serve` dependency: no admission queue, no
//! scheduler), but for granitemoe: the `granitemoe-tiny` fixture and two prompts, both admitted at
//! startup, for its first two slots. `POOT_PREFILL_CHUNK_SIZE` is unset in both, so each fills its
//! prompt token-by-token through the batched-decode replay (no separate prefill graph).
//!
//! The divergence was root-caused and fixed by card 252 round 14 (554cada40):
//! `RocmGraphExecutor::bind_decode_slots` uploaded any Const tensor carrying an exact i32 mirror as
//! raw i32 bytes, unconditional on whether the consuming kernel actually wanted a `Slice<i32>` param.
//! `moe_grouped_prep`'s `moe.iota_{E}` constant is declared F32 but synthesized via `Tensor::int`, so
//! its raw-i32 upload corrupted `gate_sorted`'s non-first top-k slot to a hard `0.0`. The fix gates the
//! raw-i32 path on `i32_param_value_ids(g, backend).contains(&id) && t.ints.is_some()`, mirroring the
//! sibling `RocmGraphExecutor::bind`'s own gate (card 188 Inc 6c-scatter-fix). Both tests below are
//! now stable, token-identical passes on real gfx1151/RADV hardware; their `#[ignore]` attributes are
//! removed.

use super::*;
use poot_executor::{Device, Executor};
use poot_graph_ir::{Slot, SlotKey, Storage};
use poot_tensor::DType;

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best = i;
        }
    }
    best
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytemuck_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

type SlotValue = (SlotKey, Vec<usize>, DType, Vec<u8>);

fn step_inputs_from(values: &[SlotValue]) -> poot_executor::StepInputs<'_> {
    let mut inputs = poot_executor::StepInputs::new();
    for (key, shape, dtype, bytes) in values {
        let elems = shape.iter().product();
        inputs.push(
            key.clone(),
            shape,
            poot_executor::HostView::new(*dtype, elems, bytes).unwrap(),
        );
    }
    inputs
}

/// `trace_batched_shared_pool_decode`'s Slot entries for one engine step.
fn batched_decode_step_values(
    g: &poot_graph_ir::Graph,
    tokens: &[u32],
    positions: &[usize],
    cap: usize,
    slot_map: &[u32],
) -> Vec<SlotValue> {
    let mut values = Vec::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let Storage::Slot(slot) = m.storage else {
            continue;
        };
        let key = m.slot_key().unwrap().clone();
        let shape = m.aval.shape.clone();
        match slot {
            Slot::Token => values.push((
                key,
                shape,
                DType::I32,
                i32_bytes(&tokens.iter().map(|&t| t as i32).collect::<Vec<_>>()),
            )),
            Slot::Pos => values.push((
                key,
                shape,
                DType::I32,
                i32_bytes(&positions.iter().map(|&p| p as i32).collect::<Vec<_>>()),
            )),
            Slot::SeqLen => {
                let max_pos = positions.iter().max().copied().unwrap_or(0) as i32;
                values.push((key, shape, DType::I32, i32_bytes(&[max_pos + 1])));
            }
            Slot::Mask => {
                let mut mask = Vec::new();
                for &p in positions {
                    mask.extend((0..cap).map(|t| if t <= p { 0.0f32 } else { -1.0e9 }));
                }
                values.push((key, shape, DType::F32, f32_bytes(&mask)));
            }
            Slot::SlotMap => values.push((
                key,
                shape,
                DType::I32,
                i32_bytes(&slot_map.iter().map(|&s| s as i32).collect::<Vec<_>>()),
            )),
            other => unreachable!("unexpected slot {other:?} in batched shared-pool decode"),
        }
    }
    values
}

/// Add one `trace_batched_shared_pool_decode` entry on `exec`/`exe` (the contract's
/// capture-on-first-step default stands in for the deleted `RocmGraphExecutor::
/// capture_decode_batched`, strict-dead-pub per Card 548's Scope).
fn add_batched_decode_entry(
    exec: &mut dyn poot_executor::Executor,
    exe: poot_executor::ExecutableId,
    target: poot_graph_plan::Target,
    g: &poot_graph_ir::Graph,
) -> poot_executor::EntryId {
    let staged = {
        let g = g.clone().with_validations(Vec::new());
        poot_graph_plan::compile_staged(
            &g,
            &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
            &poot_graph_plan::Partition {
                experts: poot_graph_plan::ExpertPlacement::AllResident,
                devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
            },
            &poot_graph_plan::CompileOptions {
                execution: poot_graph_plan::Submission::Replay,
                fusion: poot_graph_plan::FusionPolicy::Full,
                limits: poot_graph_plan::CompileLimits::STANDARD,
            },
        )
        .expect("compile_staged batched shared-pool decode")
    };
    exec.add_entry(exe, &staged).expect("add_entry")
}

#[test]
fn moe_rocm_batched_decode_two_slots_match_single_seq_granitemoe() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granitemoe-tiny"))
    else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "SKIP moe_rocm_batched_decode_two_slots_match_single_seq_granitemoe (ROCm \
                 unavailable: {e})"
            );
            return;
        }
    };
    let target = device.target();
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load granitemoe-tiny");
    // One executable for the batched entry; the baseline loop below opens its own per prompt
    // (state is shared by name/aval/storage *within* one executable, and these two
    // prompts have different token lengths, so each single-seq decode's cap-dependent KV shape
    // differs too - sharing one executable across any of these is a real `BindError::StateSchema`
    // mismatch, not an aval the call sites happen to agree on).
    let exe = runner.load_on(&mut rocm).expect("load_on (batched)");
    assert!(
        runner.is_moe(),
        "fixture must be a granitemoe (MoE special-tracer) model"
    );

    // Update 0593's two "both admitted at startup" prompts/lengths (still diverging with a stable,
    // never-recaptured 2-slot run).
    let prompts = [
        "The quick brown fox jumps over",
        "In a village of La Mancha, the name of",
    ];
    let max_news = [6usize, 10];

    // Baseline: independent single-sequence contiguous decode (`Runner::generate_kv_rocm`), the
    // reference update 0593 used; separate machinery from the shared-pool paged path under test, so
    // agreement is not circular.
    let mut baseline: Vec<Vec<u32>> = Vec::new();
    for (p, &mn) in prompts.iter().zip(&max_news) {
        let exe_baseline = runner.load_on(&mut rocm).expect("load_on (baseline)");
        let toks = runner
            .generate_kv_gpu_cached(p, mn, &mut rocm, exe_baseline, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect("single-seq baseline decode");
        baseline.push(toks);
    }

    // Batched: both prompts decode concurrently through one n_slots=2 captured shared-pool decode graph,
    // the primitives `batch_engine_loop_rocm_runner` drives, with no server loop, admission queue or
    // scheduler: this test owns the slot/step bookkeeping.
    let n_slots = 2usize;
    const CAP: usize = 32; // matches update 0593's own regression test
    let prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| runner.encode(p).expect("encode"))
        .collect();
    for t in &prompt_tokens {
        assert!(
            t.len() + max_news[0].max(max_news[1]) <= CAP,
            "prompt+max_new must fit CAP"
        );
    }

    let pool_slots = n_slots * CAP; // tight private-per-row pool (no paged block-table indirection)
    let g = runner
        .trace_batched_shared_pool_decode(CAP, n_slots, pool_slots, false)
        .expect("trace batched shared-pool decode");

    // Simple non-paged interleaved slot map: row r's logical position t -> pool slot t*n_slots+r
    // (mirrors `poot-eval`'s `moe_paged_decode` CPU-oracle test; no PagedKvCache block-table indirection
    // is needed for a fixed, non-evicting 2-slot run).
    let mut slot_map: Vec<u32> = Vec::with_capacity(n_slots * CAP);
    for r in 0..n_slots {
        for t in 0..CAP {
            slot_map.push((t * n_slots + r) as u32);
        }
    }

    let entry = add_batched_decode_entry(&mut rocm, exe, target, &g);

    let vocab = runner.cfg.vocab;
    let eos = runner.eos();
    let mut seqs: Vec<Vec<u32>> = prompt_tokens;
    let mut pos = vec![0usize; n_slots];
    let mut generated = vec![0usize; n_slots];
    let mut done = vec![false; n_slots];
    let mut steps = 0usize;
    loop {
        if done.iter().all(|&d| d) {
            break;
        }
        steps += 1;
        assert!(
            steps < CAP * 2,
            "runaway decode loop (no slot ever finished)"
        );
        let tokens: Vec<u32> = (0..n_slots).map(|j| seqs[j][pos[j]]).collect();
        let positions: Vec<usize> = pos.clone();
        let values = batched_decode_step_values(&g, &tokens, &positions, CAP, &slot_map);
        let inputs = step_inputs_from(&values);
        let bytes = rocm
            .step(exe, entry, &inputs, &mut poot_executor::NoSync)
            .expect("batched decode step")
            .read()
            .expect("batched decode readback");
        let logits_data = bytemuck_f32(&bytes);
        for j in 0..n_slots {
            if done[j] {
                continue;
            }
            let at_frontier = pos[j] + 1 == seqs[j].len();
            if at_frontier {
                let row = &logits_data[j * vocab..(j + 1) * vocab];
                let next = argmax(row) as u32;
                if next == eos {
                    done[j] = true;
                    continue;
                }
                seqs[j].push(next);
                generated[j] += 1;
                if generated[j] >= max_news[j] {
                    done[j] = true;
                    continue;
                }
            }
            pos[j] += 1;
        }
    }

    let mut all_match = true;
    for j in 0..n_slots {
        eprintln!(
            "moe_rocm_batched_decode_two_slots_match_single_seq_granitemoe: slot {j} \
             batched={:?} baseline={:?}",
            seqs[j], baseline[j]
        );
        if seqs[j] != baseline[j] {
            all_match = false;
        }
    }
    assert!(
        all_match,
        "granitemoe batched (via raw capture_decode_batched/decode_step_batched, NO server loop) \
         diverges from the single-seq reference - the bug reproduces at the dispatch-primitive \
         level with no batch_engine_loop_rocm_runner scheduling involved, narrowing it to \
         RocmGraphExecutor / the graph's ROCm kernel lowering for a batch=2 MoE decode graph"
    );
    eprintln!(
        "moe_rocm_batched_decode_two_slots_match_single_seq_granitemoe: OK ({n_slots} slots, both \
         token-identical to the single-sequence baseline)"
    );
}

/// The prompts of the test above have different token lengths, so once the shorter finishes echoing
/// it starts generating while the longer is still echoing, and the rows' `positions` are
/// heterogeneous for part of the run. A synthetic test in `poot-rocm-gpu`
/// (`decode_step_batched_moe_granitemoe_real_dims_matches_cpu_oracle_rocm`) with granitemoe-tiny's shape/
/// scalar config (vocab=49156, hidden=32, 8 experts top_k=2, no-GQA head_dim=8, attn_mult=1.0, etc.),
/// synthetic weights and always-synchronized positions (`vec![s; bsz]`) passed at bsz=2, ruling out
/// shape/scalar config alone. This test isolates the remaining variable: the same prompt and `max_new`
/// in both slots keep `positions` identical for the whole run, with the real granitemoe-tiny weights.
///
/// If it passes, heterogeneous positions (rows at different positions in one batched step) are the
/// trigger. If it fails, they are not required, implicating the real weight values (vs synthetic
/// xorshift ones) or the real token ids. A same-prompt pair also gives a free check: slot 0 and slot 1
/// must match each other (not only the single-seq baseline); any slot-vs-slot difference is direct
/// proof of batch-row crosstalk.
#[test]
fn moe_rocm_batched_decode_two_slots_same_prompt_granitemoe() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granitemoe-tiny"))
    else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "SKIP moe_rocm_batched_decode_two_slots_same_prompt_granitemoe (ROCm unavailable: \
                 {e})"
            );
            return;
        }
    };
    let target = device.target();
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load granitemoe-tiny");
    // Two separate executables: the single-seq baseline's KV state shape differs
    // from the batched entry's under the same state name, so they must not share one executable.
    let exe_baseline = runner.load_on(&mut rocm).expect("load_on (baseline)");
    let exe = runner.load_on(&mut rocm).expect("load_on (batched)");
    assert!(
        runner.is_moe(),
        "fixture must be a granitemoe (MoE special-tracer) model"
    );

    // Same prompt and max_new in both slots: positions stay synchronized across both rows for the whole
    // run, unlike the sibling test above.
    let prompt = "The quick brown fox jumps over";
    let max_new = 10usize;
    let prompts = [prompt, prompt];
    let max_news = [max_new, max_new];

    let mut baseline: Vec<Vec<u32>> = Vec::new();
    for (p, &mn) in prompts.iter().zip(&max_news) {
        let toks = runner
            .generate_kv_gpu_cached(p, mn, &mut rocm, exe_baseline, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect("single-seq baseline decode");
        baseline.push(toks);
    }

    let n_slots = 2usize;
    const CAP: usize = 32;
    let prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| runner.encode(p).expect("encode"))
        .collect();
    for t in &prompt_tokens {
        assert!(t.len() + max_new <= CAP, "prompt+max_new must fit CAP");
    }

    let pool_slots = n_slots * CAP;
    let g = runner
        .trace_batched_shared_pool_decode(CAP, n_slots, pool_slots, false)
        .expect("trace batched shared-pool decode");

    let mut slot_map: Vec<u32> = Vec::with_capacity(n_slots * CAP);
    for r in 0..n_slots {
        for t in 0..CAP {
            slot_map.push((t * n_slots + r) as u32);
        }
    }

    let entry = add_batched_decode_entry(&mut rocm, exe, target, &g);

    let vocab = runner.cfg.vocab;
    let eos = runner.eos();
    let mut seqs: Vec<Vec<u32>> = prompt_tokens;
    let mut pos = vec![0usize; n_slots];
    let mut generated = vec![0usize; n_slots];
    let mut done = vec![false; n_slots];
    let mut steps = 0usize;
    loop {
        if done.iter().all(|&d| d) {
            break;
        }
        steps += 1;
        assert!(
            steps < CAP * 2,
            "runaway decode loop (no slot ever finished)"
        );
        let tokens: Vec<u32> = (0..n_slots).map(|j| seqs[j][pos[j]]).collect();
        let positions: Vec<usize> = pos.clone();
        assert_eq!(
            positions[0], positions[1],
            "same-prompt/same-max_new rows must stay position-synchronized by construction"
        );
        let values = batched_decode_step_values(&g, &tokens, &positions, CAP, &slot_map);
        let inputs = step_inputs_from(&values);
        let bytes = rocm
            .step(exe, entry, &inputs, &mut poot_executor::NoSync)
            .expect("batched decode step")
            .read()
            .expect("batched decode readback");
        let logits_data = bytemuck_f32(&bytes);
        for j in 0..n_slots {
            if done[j] {
                continue;
            }
            let at_frontier = pos[j] + 1 == seqs[j].len();
            if at_frontier {
                let row = &logits_data[j * vocab..(j + 1) * vocab];
                let next = argmax(row) as u32;
                if next == eos {
                    done[j] = true;
                    continue;
                }
                seqs[j].push(next);
                generated[j] += 1;
                if generated[j] >= max_news[j] {
                    done[j] = true;
                    continue;
                }
            }
            pos[j] += 1;
        }
    }

    let mut all_match = true;
    for j in 0..n_slots {
        eprintln!(
            "moe_rocm_batched_decode_two_slots_same_prompt_granitemoe: slot {j} batched={:?} \
             baseline={:?}",
            seqs[j], baseline[j]
        );
        if seqs[j] != baseline[j] {
            all_match = false;
        }
    }
    assert_eq!(
        seqs[0], seqs[1],
        "same prompt in both slots must give IDENTICAL outputs - a difference here is direct \
         proof of batch-row crosstalk"
    );
    assert!(
        all_match,
        "granitemoe batched (same prompt in both slots, positions always synchronized) still \
         diverges from the single-seq reference - heterogeneous positions are NOT required to \
         trigger this bug"
    );
    eprintln!(
        "moe_rocm_batched_decode_two_slots_same_prompt_granitemoe: OK ({n_slots} slots, both \
         token-identical to the single-sequence baseline AND to each other)"
    );
}

/// `..._same_prompt_granitemoe` (above) passed, but a same-prompt pair is a weak control for
/// heterogeneous positions: both rows compute identical embeddings, router logits and expert
/// selection at every step, so row-to-row MoE crosstalk (row 0's selected-expert weights or gate
/// value leaking into row 1's FFN output, or vice versa) is undetectable by construction. This test
/// separates "heterogeneous positions" from "distinct-row MoE routing crosstalk" by using different
/// token content in the two rows with synchronized positions (equal length, `pos[0] == pos[1]` at
/// every step): row 1's context is row 0's prompt tokens reversed (equal length, different
/// embeddings/routing at nearly every position, no re-tokenization; `generate_kv_rocm_prefilled_tokens`
/// accepts a raw token context).
///
/// - If it passes: heterogeneous positions were the missing ingredient in the original divergence,
///   which refocuses the search on position/mask/at-frontier bookkeeping.
/// - If it fails: synchronized positions are not enough once the rows carry different content
///   (hence different per-row expert routing), implicating MoE row crosstalk in the batched
///   routing/expert-gather computation (`moe_grouped`'s `indexed_matmul` gather-scatter over the
///   flattened `M = batch*top_k` rows) independent of position heterogeneity.
///
/// `ignore_eos = true` throughout (single-seq baselines and the batched loop run a fixed `max_new`
/// with no early stop), so both rows' positions stay synchronized whatever either row generates.
///
/// Originally (update 0593 round 2), this test failed even with positions fully synchronized,
/// implicating a real-weight-value-dependent numerical difference in ROCm's `moe_grouped` lowering
/// rather than a structural indexing bug. That was the same `bind_decode_slots` raw-i32 upload bug
/// (see the module preamble) fixed by card 252 round 14 (554cada40); diagnosis re-ran this test on
/// real gfx1151/RADV hardware after the fix and it is now a stable, token-identical pass.
#[test]
fn moe_rocm_batched_decode_two_slots_equal_length_diff_content_granitemoe() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granitemoe-tiny"))
    else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "SKIP moe_rocm_batched_decode_two_slots_equal_length_diff_content_granitemoe \
                 (ROCm unavailable: {e})"
            );
            return;
        }
    };
    let target = device.target();
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load(&dir).expect("load granitemoe-tiny");
    // Two separate executables: the single-seq baseline's KV state shape differs
    // from the batched entry's under the same state name, so they must not share one executable.
    let exe_baseline = runner.load_on(&mut rocm).expect("load_on (baseline)");
    let exe = runner.load_on(&mut rocm).expect("load_on (batched)");
    assert!(
        runner.is_moe(),
        "fixture must be a granitemoe (MoE special-tracer) model"
    );

    let t0 = runner
        .encode("The quick brown fox jumps over")
        .expect("encode");
    let mut t1 = t0.clone();
    t1.reverse();
    assert_ne!(t0, t1, "reversed context must differ (not a palindrome)");
    let prompt_tokens: Vec<Vec<u32>> = vec![t0, t1];
    let max_new = 10usize;

    let mut baseline: Vec<Vec<u32>> = Vec::new();
    for t in &prompt_tokens {
        let toks = runner
            .generate_kv_rocm_prefilled_tokens(t, max_new, &mut rocm, exe_baseline, true, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect("single-seq baseline decode");
        baseline.push(toks);
    }

    let n_slots = 2usize;
    const CAP: usize = 32;
    for t in &prompt_tokens {
        assert!(t.len() + max_new <= CAP, "prompt+max_new must fit CAP");
    }
    assert_eq!(
        prompt_tokens[0].len(),
        prompt_tokens[1].len(),
        "both rows must start at the SAME length so positions stay synchronized"
    );

    let pool_slots = n_slots * CAP;
    let g = runner
        .trace_batched_shared_pool_decode(CAP, n_slots, pool_slots, false)
        .expect("trace batched shared-pool decode");

    let mut slot_map: Vec<u32> = Vec::with_capacity(n_slots * CAP);
    for r in 0..n_slots {
        for t in 0..CAP {
            slot_map.push((t * n_slots + r) as u32);
        }
    }

    let entry = add_batched_decode_entry(&mut rocm, exe, target, &g);

    let vocab = runner.cfg.vocab;
    let mut seqs: Vec<Vec<u32>> = prompt_tokens;
    let start_len = seqs[0].len();
    let mut pos = vec![0usize; n_slots];
    let mut generated = vec![0usize; n_slots];
    let mut done = vec![false; n_slots];
    let mut steps = 0usize;
    loop {
        if done.iter().all(|&d| d) {
            break;
        }
        steps += 1;
        assert!(
            steps < CAP * 2,
            "runaway decode loop (no slot ever finished)"
        );
        let tokens: Vec<u32> = (0..n_slots).map(|j| seqs[j][pos[j]]).collect();
        let positions: Vec<usize> = pos.clone();
        assert_eq!(
            positions[0], positions[1],
            "equal-length rows must stay position-synchronized by construction"
        );
        let values = batched_decode_step_values(&g, &tokens, &positions, CAP, &slot_map);
        let inputs = step_inputs_from(&values);
        let bytes = rocm
            .step(exe, entry, &inputs, &mut poot_executor::NoSync)
            .expect("batched decode step")
            .read()
            .expect("batched decode readback");
        let logits_data = bytemuck_f32(&bytes);
        for j in 0..n_slots {
            if done[j] {
                continue;
            }
            let at_frontier = pos[j] + 1 == seqs[j].len();
            if at_frontier {
                let row = &logits_data[j * vocab..(j + 1) * vocab];
                let next = argmax(row) as u32;
                // ignore_eos: never stop early on EOS, matching the baseline's `ignore_eos=true`.
                seqs[j].push(next);
                generated[j] += 1;
                if generated[j] >= max_new {
                    done[j] = true;
                    continue;
                }
            }
            pos[j] += 1;
        }
    }
    assert_eq!(seqs[0].len(), start_len + max_new);
    assert_eq!(seqs[1].len(), start_len + max_new);

    let mut all_match = true;
    for j in 0..n_slots {
        eprintln!(
            "moe_rocm_batched_decode_two_slots_equal_length_diff_content_granitemoe: slot {j} \
             batched={:?} baseline={:?}",
            seqs[j], baseline[j]
        );
        if seqs[j] != baseline[j] {
            all_match = false;
        }
    }
    assert!(
        all_match,
        "granitemoe batched (two DIFFERENT-content, EQUAL-length, position-synchronized rows) \
         diverges from the single-seq reference - heterogeneous positions are NOT required; \
         distinct-row content/routing alone triggers this"
    );
    eprintln!(
        "moe_rocm_batched_decode_two_slots_equal_length_diff_content_granitemoe: OK ({n_slots} \
         slots, both token-identical to the single-sequence baseline)"
    );
}

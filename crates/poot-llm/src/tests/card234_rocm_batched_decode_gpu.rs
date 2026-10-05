// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! Real-hardware check (card 129 Track A5, spec 234): N sequences decoding concurrently through the
//! generic-`Runner` batched-decode graph (`Runner::trace_batched_shared_pool_decode`) match the
//! single-sequence ROCm path (`Runner::generate_kv_gpu_cached`) token for token. Card 548: both
//! halves now run on `Engine<RocmDevice>` - the batched half is one entry stepped once per engine
//! tick with each tick's tokens/positions/slot map as `StepInputs` (the contract's capture-once
//! default stands in for the deleted `RocmGraphExecutor::capture_decode_batched`/
//! `decode_step_batched`, strict-dead-pub per this card's Scope). These are the primitives
//! `batch_engine_loop_rocm_runner` (`crates/poot-serve/src/batch.rs`) drives, called directly here so
//! no `poot-serve` dependency is needed.
//!
//! Uses the qwen2.5-0.5b Q4_K_M GGUF, not the bf16 safetensors checkpoint: native bf16 projections hit
//! an unrelated planner gap (plain
//! `MatMul` decode of a natively-bf16 checkpoint has no non-tensor-core kernel at M=1 on any backend).
//! GGUF packed-quant decode uses `PackedDequant` and avoids it.

use super::*;
use crate::driver::block_table::{BLOCK_SIZE, PagedKvCache};
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

/// `trace_batched_shared_pool_decode`'s Slot entries for one engine step: `tokens`/`positions` one
/// per physical row, `slot_map` the full `[n_slots, cap]` block table.
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

#[test]
fn card234_rocm_batched_decode_two_slots_match_single_seq() {
    let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )) else {
        return;
    };
    let device = match poot_rocm_gpu::device::RocmDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "SKIP card234_rocm_batched_decode_two_slots_match_single_seq (ROCm unavailable: {e})"
            );
            return;
        }
    };
    let target = device.target();
    let mut rocm = poot_executor::Engine::new(device);
    let runner = Runner::load_gguf(&path).expect("load qwen2.5-0.5b gguf");
    // A separate executable per baseline prompt, plus one more for the batched entry (state is shared by name/aval/storage *within* one executable). The two prompts have different
    // token lengths, so `generate_kv_gpu_cached`'s decode graph (cap = tokens.len() + max_new) gives
    // each a differently-shaped `model.layers.0.kv.k_cache`; the batched entry's shape differs again
    // (`[1, n_slots, cap, head_dim]`). Sharing one executable across any of these is a real
    // `BindError::StateSchema` mismatch, not an aval the call sites happen to agree on.
    let exe = runner.load_on(&mut rocm).expect("load_on (batched)");

    let prompts = ["Hello, how are you?", "The capital of France is"];
    let max_new = 6usize;

    // Baseline: independent single-sequence decode per prompt (the `rocm_generator` path in
    // poot-serve, POOT_SLOTS==1), through the same backend-neutral contract entry point every
    // backend's single-sequence decode uses.
    let mut baseline: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        let exe_baseline = runner.load_on(&mut rocm).expect("load_on (baseline)");
        let toks = runner
            .generate_kv_gpu_cached(p, max_new, &mut rocm, exe_baseline, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect("single-seq baseline decode");
        baseline.push(toks);
    }

    // Batched: both prompts decode concurrently through one n_slots=2 entry, stepped once per
    // engine tick with that tick's tokens/positions/slot map (the contract's capture-on-first-step
    // default stands in for `batch_engine_loop_rocm_runner`'s per-step batched replay).
    let n_slots = 2usize;
    let prompt_tokens: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| runner.encode(p).expect("encode"))
        .collect();
    let cap = prompt_tokens.iter().map(|t| t.len()).max().unwrap() + max_new;

    let num_blocks = (n_slots * cap).div_ceil(BLOCK_SIZE);
    let pool_slots = num_blocks * BLOCK_SIZE + 1;
    let g = runner
        .trace_batched_shared_pool_decode(cap, n_slots, pool_slots, false)
        .expect("trace batched shared-pool decode");
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
    let entry = rocm.add_entry(exe, &staged).expect("add_entry");

    let mut paged = PagedKvCache::new(n_slots, num_blocks);
    for (j, pt) in prompt_tokens.iter().enumerate() {
        paged
            .append(j, pt.len() + max_new)
            .expect("reserve KV blocks");
    }
    let mut slot_map: Vec<u32> = Vec::with_capacity(n_slots * cap);
    for j in 0..n_slots {
        slot_map.extend(paged.slot_mapping(j, cap));
    }

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
            steps < cap * 2,
            "runaway decode loop (no slot ever finished)"
        );
        let tokens: Vec<u32> = (0..n_slots).map(|j| seqs[j][pos[j]]).collect();
        let positions: Vec<usize> = pos.clone();
        let values = batched_decode_step_values(&g, &tokens, &positions, cap, &slot_map);
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
                if generated[j] >= max_new {
                    done[j] = true;
                    continue;
                }
            }
            pos[j] += 1;
        }
    }

    for j in 0..n_slots {
        eprintln!(
            "card234_rocm_batched_decode: slot {j} batched={:?} baseline={:?}",
            seqs[j], baseline[j]
        );
        assert_eq!(
            seqs[j], baseline[j],
            "slot {j}: batched ROCm decode diverges from the single-sequence baseline"
        );
    }
    eprintln!(
        "card234_rocm_batched_decode_two_slots_match_single_seq: OK ({n_slots} slots, both token-\
         identical to the single-sequence baseline)"
    );
}

//! Follow-on to `moe_rocm_routing_diag.rs` (update 0593, epic 129 B7): its
//! `moe_rocm_full_logits_diag_granitemoe` found that the CPU oracle and ROCm (both eager/uncaptured)
//! agree bit for bit (argmax-identical, logits within ~1.5e-7) on the batched shared-pool decode
//! graph's output at the divergent step, including on the "wrong" token (32046) that the captured-decode
//! test found. So the divergence is not a ROCm bug.
//!
//! This pure-CPU test asks whether the CPU oracle's single-sequence contiguous decode graph
//! (`poot_models::granite::trace_granite_decode_kv_masked`, always `L=1`, so `moe()` takes the
//! `moe_sparse` path) agrees with its batched shared-pool decode graph
//! (`trace_granite_decode_kv_masked_batched_shared_pool`, `L=batch=2`, so `moe()` takes the different
//! `moe_grouped` path) for identical checkpoint weights, tokens and position at the step that diverges.
//!
//! `poot_graph_ir::ops::moe_grouped`'s doc comment states it is "Identical output to `moe_dense` up to
//! floating-point summation order... matches within `~1e-5`, not bit-exact", a documented,
//! backend-independent difference between the two MoE paths that `moe()` picks by `L==1`. Every
//! single-sequence reference in this investigation is `L=1` (`moe_sparse`); every batched
//! (`n_slots>1`) decode is `L=batch>1` (`moe_grouped`). If the two CPU graphs disagree at this step,
//! the ~1e-5 per-layer difference, compounded over granitemoe-tiny's 2 layers, is enough to flip a
//! close (not tied) top-1/top-2 final-logit margin. That is not a backend bug but an inherent property
//! of `moe()` switching algorithms by batch size, so the single-sequence reference is not valid ground
//! truth for a bit-exact `moe_grouped` decode.

use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_llm::Runner;
use poot_models::granite::trace_granite_decode_kv_masked_batched_shared_pool;
use poot_models::granite::{GraniteParams, MoeShape, trace_granite_decode_kv_masked};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use poot_graph_plan::passes_without_target as optimize;

/// The Const (weight) inputs of `g`, bound by name from `runner`'s loaded weights: the same binding
/// `Runner::const_inputs` used before card 549 made it `pub(crate)` (its last production caller, the
/// pre-contract PTX executor, is gone). This integration test is a separate crate target and cannot
/// see a `pub(crate)` item, so it binds the same way directly through the still-`pub`
/// `Runner::weight_value`.
fn bind_consts(runner: &poot_llm::Runner, g: &Graph) -> HashMap<ValueId, poot_eval::Value> {
    let mut consts = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if let Storage::Computed(computed) = meta.storage {
            consts.insert(
                id,
                poot_eval::Value::from(HostTensor::f32(computed.shape(), computed.values_f32())),
            );
            continue;
        }
        if meta.storage == Storage::Const {
            let name = meta.name.as_deref().expect("const without a name");
            consts.insert(
                id,
                runner.weight_value(name, &meta.aval).expect("weight_value"),
            );
        }
    }
    consts
}

fn argmax(v: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    (best, best_v)
}

fn top2_margin(v: &[f32]) -> f32 {
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[0] - sorted[1]
}

#[test]
fn moe_cpu_batched_vs_single_seq_diverge_at_real_step_granitemoe() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granitemoe-tiny"))
    else {
        return;
    };
    let runner = Runner::load(&dir).expect("load granitemoe-tiny");
    assert!(runner.is_moe(), "fixture must be a granitemoe (MoE) model");

    // granitemoe-tiny's own config.json: all scalar multipliers 1.0, 8 experts, top_k=2, inter=40.
    let gp = GraniteParams {
        moe: Some(MoeShape {
            n_experts: 8,
            top_k: 2,
            inter: 40,
        }),
        embed_mult: 1.0,
        attn_mult: 1.0,
        residual_mult: 1.0,
        logits_scale: 1.0,
    };

    const CAP: usize = 32;
    let n_slots = 2usize;
    let pool_slots = n_slots * CAP;

    // Row 0's continuation, generated via the same single-seq CPU-oracle graph (pure CPU, no GPU).
    let prompt0 = "The quick brown fox jumps over";
    let prompt1 = "In a village of La Mancha, the name of";
    let max_new0 = 6usize;
    let row0_prompt = runner.encode(prompt0).expect("encode prompt0");
    let row1_prompt = runner.encode(prompt1).expect("encode prompt1");

    let seq_g = trace_granite_decode_kv_masked(*runner.config(), gp, CAP);
    let seq_consts = bind_consts(&runner, &seq_g);

    let bind_seq_step = |g: &Graph, tok: u32, pos: usize| -> HashMap<ValueId, poot_eval::Value> {
        let mut inputs: HashMap<ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, poot_eval::Value::from(HostTensor::scalar(tok as f32)));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, poot_eval::Value::from(HostTensor::scalar(pos as f32)));
                }
                Storage::Slot(Slot::SeqLen) => {
                    inputs.insert(
                        id,
                        poot_eval::Value::from(HostTensor::scalar((pos + 1) as f32)),
                    );
                }
                Storage::Slot(Slot::Mask) => {
                    let mask: Vec<f32> = (0..CAP)
                        .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
                        .collect();
                    inputs.insert(id, poot_eval::Value::from(HostTensor::f32(vec![CAP], mask)));
                }
                Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
                Storage::Const | Storage::Computed(_) | Storage::State | Storage::Device => {}
            }
        }
        inputs
    };

    // Generate row0's continuation greedily via the single-seq CPU-oracle graph.
    let mut row0_full = row0_prompt.clone();
    {
        let mut state: Vec<HostTensor> = seq_g
            .state
            .iter()
            .map(|&(sid, _)| HostTensor::zeros(seq_g.aval(sid).shape.clone()))
            .collect();
        for pos in 0..(row0_prompt.len() + max_new0 - 1) {
            let tok = row0_full[pos];
            let mut inputs = bind_seq_step(&seq_g, tok, pos);
            for (&id, t) in &seq_consts {
                inputs.insert(id, t.clone());
            }
            for (i, &(sid, _)) in seq_g.state.iter().enumerate() {
                inputs.insert(sid, poot_eval::Value::from(state[i].clone()));
            }
            let cpu_eval = poot_eval::eval(
                &seq_g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu single-seq eval");
            let logits = cpu_eval.output.into_host().expect("dense output");
            let new_state: Vec<HostTensor> = cpu_eval
                .state
                .into_iter()
                .map(poot_eval::Value::into_host)
                .collect::<Result<Vec<_>, _>>()
                .expect("dense state");
            state = new_state;
            if pos + 1 == row0_full.len() {
                let (next, _) = argmax(logits.as_f32().unwrap());
                row0_full.push(next as u32);
            }
        }
    }
    assert_eq!(row0_full.len(), row0_prompt.len() + max_new0);
    eprintln!("row0 CPU-oracle single-seq continuation: {row0_full:?}");

    let target_step = row1_prompt.len();

    // Replay the SAME bookkeeping as moe_rocm_routing_diag.rs to build the per-step tokens/positions.
    #[derive(Clone, Copy)]
    struct Step {
        tokens: [u32; 2],
        positions: [usize; 2],
    }
    let mut steps: Vec<Step> = Vec::with_capacity(target_step);
    let mut seqs0: Vec<u32> = row0_prompt.clone();
    let mut pos0 = 0usize;
    let mut done0 = false;
    let mut generated0 = 0usize;
    for (pos1, step) in (1..=target_step).enumerate() {
        let tok0 = seqs0[pos0];
        let tok1 = row1_prompt[pos1];
        steps.push(Step {
            tokens: [tok0, tok1],
            positions: [pos0, pos1],
        });
        if step == target_step {
            break;
        }
        if !done0 {
            let at_frontier0 = pos0 + 1 == seqs0.len();
            if at_frontier0 {
                let next = row0_full[seqs0.len()];
                seqs0.push(next);
                generated0 += 1;
                if generated0 >= max_new0 {
                    done0 = true;
                }
            }
            if !done0 {
                pos0 += 1;
            }
        }
    }
    let last = steps[target_step - 1];
    eprintln!(
        "moe_cpu_batched_vs_single_seq_diverge_at_real_step_granitemoe: divergent step {target_step}: \
         tokens={:?} positions={:?}",
        last.tokens, last.positions
    );

    // --- Single-seq CPU oracle: row1's OWN independent decode, moe_sparse (L=1) at every step. ---
    let mut seq_logits_row1: Vec<f32> = Vec::new();
    {
        let mut state: Vec<HostTensor> = seq_g
            .state
            .iter()
            .map(|&(sid, _)| HostTensor::zeros(seq_g.aval(sid).shape.clone()))
            .collect();
        for (pos, &tok) in row1_prompt.iter().enumerate() {
            let mut inputs = bind_seq_step(&seq_g, tok, pos);
            for (&id, t) in &seq_consts {
                inputs.insert(id, t.clone());
            }
            for (i, &(sid, _)) in seq_g.state.iter().enumerate() {
                inputs.insert(sid, poot_eval::Value::from(state[i].clone()));
            }
            let cpu_eval = poot_eval::eval(
                &seq_g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu single-seq eval");
            let logits = cpu_eval.output.into_host().expect("dense output");
            let new_state: Vec<HostTensor> = cpu_eval
                .state
                .into_iter()
                .map(poot_eval::Value::into_host)
                .collect::<Result<Vec<_>, _>>()
                .expect("dense state");
            state = new_state;
            if pos + 1 == row1_prompt.len() {
                seq_logits_row1 = logits.as_f32().unwrap().to_vec();
            }
        }
    }
    let (seq_argmax, seq_argmax_v) = argmax(&seq_logits_row1);
    let seq_margin = top2_margin(&seq_logits_row1);
    eprintln!(
        "single-seq CPU oracle (moe_sparse, L=1, RAW unoptimized graph): row1 argmax={seq_argmax} \
         ({seq_argmax_v:.6}) top1-top2_margin={seq_margin:.6e}"
    );

    // Does the pre-`compile` pass chain (`optimize`) change the answer? `Runner::generate_kv_rocm`'s
    // single-seq reference path calls `optimize(&dg)` before running, while this test's `seq_g` above did
    // not (raw graph, direct `eval_with_state`). If they differ, the reference harness ran a numerically
    // different (fused/optimized) graph than the batched path it was compared against, rather than a
    // moe_sparse-vs-moe_grouped difference.
    let seq_g_opt = optimize(&seq_g);
    let seq_opt_consts = bind_consts(&runner, &seq_g_opt);
    let mut seq_opt_logits_row1: Vec<f32> = Vec::new();
    {
        let mut state: Vec<HostTensor> = seq_g_opt
            .state
            .iter()
            .map(|&(sid, _)| HostTensor::zeros(seq_g_opt.aval(sid).shape.clone()))
            .collect();
        for (pos, &tok) in row1_prompt.iter().enumerate() {
            let mut inputs = bind_seq_step(&seq_g_opt, tok, pos);
            for (&id, t) in &seq_opt_consts {
                inputs.insert(id, t.clone());
            }
            for (i, &(sid, _)) in seq_g_opt.state.iter().enumerate() {
                inputs.insert(sid, poot_eval::Value::from(state[i].clone()));
            }
            let cpu_eval = poot_eval::eval(
                &seq_g_opt,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("cpu single-seq (optimized) eval");
            let logits = cpu_eval.output.into_host().expect("dense output");
            let new_state: Vec<HostTensor> = cpu_eval
                .state
                .into_iter()
                .map(poot_eval::Value::into_host)
                .collect::<Result<Vec<_>, _>>()
                .expect("dense state");
            state = new_state;
            if pos + 1 == row1_prompt.len() {
                seq_opt_logits_row1 = logits.as_f32().unwrap().to_vec();
            }
        }
    }
    let (seq_opt_argmax, seq_opt_argmax_v) = argmax(&seq_opt_logits_row1);
    let seq_opt_margin = top2_margin(&seq_opt_logits_row1);
    let opt_vs_raw_max_abs = poot_test_util::max_abs_error(&seq_opt_logits_row1, &seq_logits_row1);
    eprintln!(
        "single-seq CPU oracle (moe_sparse, L=1, OPTIMIZED graph - matches Runner::generate_kv_rocm's \
         own actual call path): row1 argmax={seq_opt_argmax} ({seq_opt_argmax_v:.6}) \
         top1-top2_margin={seq_opt_margin:.6e} max|logit diff vs RAW graph|={opt_vs_raw_max_abs:.6e} \
         argmax_differs_from_raw={}",
        seq_opt_argmax != seq_argmax
    );

    // --- Batched CPU oracle: BOTH rows together, moe_grouped (L=batch=2) at every step. ---
    let batched_g = trace_granite_decode_kv_masked_batched_shared_pool(
        *runner.config(),
        gp,
        CAP,
        n_slots,
        pool_slots,
    );
    batched_g.validate().expect("batched graph must validate");
    let batched_consts = bind_consts(&runner, &batched_g);
    let global_slotmap: Vec<i32> = (0..n_slots)
        .flat_map(|r| (0..CAP).map(move |t| (t * n_slots + r) as i32))
        .collect();
    let bind_batched_step = |g: &Graph, s: &Step| -> HashMap<ValueId, poot_eval::Value> {
        let mut inputs: HashMap<ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(
                        id,
                        poot_eval::Value::from(HostTensor::f32(
                            vec![n_slots],
                            s.tokens.iter().map(|&t| t as f32).collect(),
                        )),
                    );
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(
                        id,
                        poot_eval::Value::from(HostTensor::f32(
                            vec![n_slots],
                            s.positions.iter().map(|&p| p as f32).collect(),
                        )),
                    );
                }
                Storage::Slot(Slot::SeqLen) => {
                    let sl = s.positions.iter().max().copied().unwrap_or(0) + 1;
                    inputs.insert(id, poot_eval::Value::from(HostTensor::scalar(sl as f32)));
                }
                Storage::Slot(Slot::Mask) => {
                    let mut mask = Vec::with_capacity(n_slots * CAP);
                    for &p in &s.positions {
                        for t in 0..CAP {
                            mask.push(if t <= p { 0.0f32 } else { -1.0e9 });
                        }
                    }
                    inputs.insert(
                        id,
                        poot_eval::Value::from(HostTensor::f32(vec![n_slots, CAP], mask)),
                    );
                }
                Storage::Slot(Slot::SlotMap) => {
                    inputs.insert(
                        id,
                        poot_eval::Value::from(HostTensor::i32(
                            vec![n_slots, CAP],
                            global_slotmap.clone(),
                        )),
                    );
                }
                Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
                Storage::Const | Storage::Computed(_) | Storage::State | Storage::Device => {}
            }
        }
        inputs
    };
    let mut batched_state: Vec<HostTensor> = batched_g
        .state
        .iter()
        .map(|&(sid, _)| HostTensor::zeros(batched_g.aval(sid).shape.clone()))
        .collect();
    let mut batched_logits = HostTensor::zeros(vec![n_slots, 1, runner.config().vocab]);
    for s in &steps {
        let mut inputs = bind_batched_step(&batched_g, s);
        for (&id, t) in &batched_consts {
            inputs.insert(id, t.clone());
        }
        for (i, &(sid, _)) in batched_g.state.iter().enumerate() {
            inputs.insert(sid, poot_eval::Value::from(batched_state[i].clone()));
        }
        let cpu_eval = poot_eval::eval(
            &batched_g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("cpu batched eval");
        let out = cpu_eval.output.into_host().expect("dense output");
        let new_state: Vec<HostTensor> = cpu_eval
            .state
            .into_iter()
            .map(poot_eval::Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .expect("dense state");
        batched_logits = out;
        batched_state = new_state;
    }
    let vocab = runner.config().vocab;
    let batched_row1 = &batched_logits.as_f32().unwrap()[vocab..2 * vocab];
    let (batched_argmax, batched_argmax_v) = argmax(batched_row1);
    let batched_margin = top2_margin(batched_row1);
    eprintln!(
        "batched CPU oracle (moe_grouped, L=batch={n_slots}): row1 argmax={batched_argmax} \
         ({batched_argmax_v:.6}) top1-top2_margin={batched_margin:.6e}"
    );

    let max_abs = poot_test_util::max_abs_error(batched_row1, &seq_logits_row1);
    eprintln!(
        "moe_cpu_batched_vs_single_seq_diverge_at_real_step_granitemoe: single_seq_argmax={seq_argmax} \
         batched_argmax={batched_argmax} max|logit diff| (moe_sparse vs moe_grouped path)={max_abs:.6e} \
         argmax_differs={}",
        seq_argmax != batched_argmax
    );

    // Report only: documents whether the CPU-oracle single-seq (moe_sparse) and batched (moe_grouped)
    // graphs already disagree on this prompt/step, independent of any GPU backend (round 3 addendum).
}

//! Card 251 follow-on: after fixing `moe_sparse`'s Scatter-based routing (the
//! tie-under-capture-replay-staleness bug), `batch_engine_loop_rocm_runner_granitemoe_matches_single_seq_reference`
//! matches the single-seq reference at the first generated token (`32046`, both sides) but still
//! diverges at the second generated token for row 1 (`batch_engine_loop_rocm_runner` gives `14410`;
//! `Runner::generate_kv_rocm` gives `14644`).
//!
//! Pure-CPU check, mirroring `moe_cpu_batched_vs_single_seq_diag.rs`: does the CPU oracle's
//! `moe_grouped` (batched, L=2) computation at this second step agree with the single-seq reference's
//! `14644`, with the batched engine's `14410`, or neither? That settles whether the remaining
//! divergence is (a) a `moe_sparse` vs `moe_grouped` floating-point-summation-order difference at this
//! step (not a bug; `moe_grouped`'s doc comment says "not bit-exact", `~1e-5`), or (b) a ROCm-specific
//! batched-dispatch bug (the CPU batched oracle would then side with `14644`, not `14410`).

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

#[test]
fn moe_cpu_batched_vs_single_seq_step2_granitemoe() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granitemoe-tiny"))
    else {
        return;
    };
    let runner = Runner::load(&dir).expect("load granitemoe-tiny");
    assert!(runner.is_moe());

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

    // Job 0 / job 1 prompts and max_new match the first two jobs of
    // batch_engine_loop_rocm_runner_granitemoe_matches_single_seq_reference. Job 2 admits into whichever
    // slot frees first, which happens only after row 0 or row 1 finishes; row 1's 2nd generated token
    // comes well before either finishes at max_new={6,10}, so a 2-slot schedule with no third job
    // mirrors the real engine's admission order this early.
    let prompt0 = "The quick brown fox jumps over";
    let prompt1 = "In a village of La Mancha, the name of";
    let max_new0 = 6usize;
    let row0_prompt = runner.encode(prompt0).expect("encode prompt0");
    let row1_prompt = runner.encode(prompt1).expect("encode prompt1");

    let seq_g = trace_granite_decode_kv_masked(*runner.config(), gp, CAP);
    let seq_g = optimize(&seq_g); // matches generate_kv_rocm's own call path
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

    // Row 0's greedy continuation via the single-seq CPU oracle (needed to build the interleaved batched
    // step schedule below).
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

    // Row 1's greedy continuation via the single-seq CPU oracle, two generated tokens (to reach the
    // step-2 divergence).
    let mut row1_full = row1_prompt.clone();
    {
        let mut state: Vec<HostTensor> = seq_g
            .state
            .iter()
            .map(|&(sid, _)| HostTensor::zeros(seq_g.aval(sid).shape.clone()))
            .collect();
        for pos in 0..(row1_prompt.len() + 1) {
            let tok = row1_full[pos];
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
            if pos + 1 == row1_full.len() {
                let (next, _) = argmax(logits.as_f32().unwrap());
                eprintln!(
                    "single-seq CPU oracle row1 token at pos {}: {next} (logit={:.6})",
                    pos + 1,
                    logits.as_f32().unwrap()[next]
                );
                row1_full.push(next as u32);
            }
        }
    }
    eprintln!("row0 CPU-oracle single-seq continuation: {row0_full:?}");
    eprintln!("row1 CPU-oracle single-seq continuation (2 generated): {row1_full:?}");
    assert_eq!(
        row1_full[row1_prompt.len()],
        32046,
        "sanity: row1's first generated token via CPU oracle must be the already-confirmed 32046"
    );

    // Build the interleaved per-step (token,pos) schedule for both rows up to row 1's second generated
    // token (one step past `moe_cpu_batched_vs_single_seq_diag.rs`).
    let target_step = row1_prompt.len() + 1; // one step PAST the first divergence point
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
    let mut seqs1: Vec<u32> = row1_prompt.clone();
    for (pos1, step) in (1..=target_step).enumerate() {
        let tok0 = seqs0[pos0];
        let tok1 = seqs1[pos1];
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
        let at_frontier1 = pos1 + 1 == seqs1.len();
        if at_frontier1 {
            let next = row1_full[seqs1.len()];
            seqs1.push(next);
        }
    }
    let last = steps[target_step - 1];
    eprintln!(
        "moe_cpu_batched_vs_single_seq_step2_granitemoe: divergent step2 {target_step}: tokens={:?} \
         positions={:?}",
        last.tokens, last.positions
    );

    // --- Batched CPU oracle: BOTH rows together, moe_grouped (L=batch=2), replayed through ALL
    // steps up to and including the step-2 divergence. ---
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
    eprintln!(
        "moe_cpu_batched_vs_single_seq_step2_granitemoe: CPU batched (moe_grouped) oracle argmax={batched_argmax} \
         ({batched_argmax_v:.6}) -- ROCm batched engine gave 14410, ROCm single-seq reference gave 14644"
    );
}

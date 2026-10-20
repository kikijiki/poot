//! Serving-surface rows on the counting fake executor (Card 735): entry counting, typed refusals,
//! adapter identity, state residency. The device is a fake, so these rows read what reached the
//! executor and what the driver published, never a token's value; token parity is
//! `serving_device_tests`.

use super::*;
use crate::driver::RowSpeculation;
use crate::driver::error::{InvalidRequest, PreparedSetRefusal};
use crate::driver::lora::{AdapterSet, AdapterSource, AdapterSpec, LoraError, LoraIdx};
use crate::driver::speculate::NotSpeculated;
use crate::driver::{
    AdapterRef, Admission, Feature, Incompatible, PoolShape, PrefixOutcome, PrefixSkip, Release,
    RowWork, SeqId, SeqRequest, StepRow,
};
use poot_models::components::embed::EMBEDS;
use poot_models::components::rope::MROPE;
use poot_models::io::{Fill, InputError, InputValue, RequestInputs, TokenValues};
use poot_tensor::{DType, HostTensor};

/// The fake device's answer to any step: every row picks token 1, or favors it in the logits.
fn picks_one() -> Script {
    Box::new(|record, len| {
        // A head with logprobs publishes F32 records `[token, flag, logprob]`.
        if let Some((DType::F32, shape)) = &record.output
            && shape.len() == 2
            && shape[1] != VOCAB
        {
            // (token, flag, logprob), then zeros for the alternatives.
            return (0..shape[0])
                .flat_map(|_| {
                    let mut row = vec![0.0f32; shape[1]];
                    (row[0], row[1]) = (1.0, -1.0);
                    row
                })
                .flat_map(f32::to_le_bytes)
                .collect();
        }
        if len.is_multiple_of(VOCAB * 4) {
            (0..len / (VOCAB * 4))
                .flat_map(|_| favoring(1))
                .flat_map(f32::to_le_bytes)
                .collect()
        } else {
            (0..len / 8)
                .flat_map(|_| {
                    let mut row = 1i32.to_le_bytes().to_vec();
                    row.extend((-1i32).to_le_bytes());
                    row
                })
                .collect()
        }
    })
}

fn pool(blocks: usize, max_seqs: usize) -> PoolShape {
    PoolShape {
        blocks: nz(blocks),
        max_seqs: nz(max_seqs),
    }
}

/// A prepared admission over the decode step only (`Warm::Prompts(&[])`): one-token prompts prefill
/// through the same width-1 entry.
fn shapes<'a>(
    pool: PoolShape,
    rows: &'a [NonZeroUsize],
    heads: &'a [Head],
    adapters: &'a [crate::driver::GenerationId],
) -> ServingShapes<'a> {
    ServingShapes {
        layout: Layout::Paged(pool),
        rows,
        heads,
        warm: Warm::Prompts(&[]),
        windows: &[],
        adapters,
        chain: None,
    }
}

fn open_one(rig: &mut Rig<Serving>, prompt: &[u32], adapter: AdapterRef) -> SeqId {
    let admission = rig
        .driver
        .open(SeqRequest::new(prompt, 16, adapter))
        .unwrap();
    let Admission::Opened { seq, .. } = admission else {
        panic!("{admission:?}")
    };
    seq
}

/// One step of `works` over `seqs`, each with its own sampler; commits every pick.
fn step_rows(
    rig: &mut Rig<Serving>,
    rows: &mut [(SeqId, RowWork<'static>, Sampler)],
) -> Result<Vec<Vec<u32>>, DriverError> {
    let mut step_rows: Vec<StepRow<'_>> = rows
        .iter_mut()
        .map(|(seq, work, sampler)| StepRow {
            seq: *seq,
            work: *work,
            sampler,
        })
        .collect();
    let results = rig.driver.step(&mut step_rows)?;
    drop(step_rows);
    let mut picked = Vec::new();
    for result in results {
        let sampler = &mut rows
            .iter_mut()
            .find(|(seq, ..)| *seq == result.seq)
            .expect("a result names a row of the step")
            .2;
        rig.driver
            .commit(result.seq, result.tokens.len(), sampler)?;
        picked.push(result.tokens);
    }
    Ok(picked)
}

const PREFILL1: RowWork<'static> = RowWork::Prefill { upto: 1 };
/// A plain decode step.
const DECODE: RowWork<'static> = RowWork::Decode { ahead: &[] };

/// A greedy row, and a row that records logprobs (the head with logprobs serves it).
fn plain() -> Sampler {
    Sampler::greedy()
}

fn hosted() -> Sampler {
    Sampler::greedy().with_logprobs(0)
}

fn paged_rig(options: RigOptions) -> Rig {
    build_rig(options, picks_one()).unwrap()
}

/// SC-008: with `max_entries = 4`, rows {1, 2} x heads {argmax, argmax with logprobs} for one fixed schema are four
/// entries; exercising all four repeatedly with changing row assignments compiles and adds nothing; a
/// fifth static schema is refused (and a step that needs it is refused typed) while every old entry
/// still runs; a driver with a larger scope admits it. Mutations: drop `rows` from the entry identity
/// (two entries, not four), drop the head (two), skip the entry-count check (the fifth enters).
#[test]
fn rows_and_heads_are_four_entries_that_serve_every_composition() {
    let rig = paged_rig(RigOptions {
        max_entries: 4,
        ..RigOptions::default()
    });
    let rows = [nz(1), nz(2)];
    let greedy =
        poot_graph_ir::ops::sampling::SampleSpec::draw(poot_graph_ir::op::SampleRule::Greedy);
    let heads = [Head::GREEDY, Head::Sample(greedy.with_logprobs(0))];
    let mut rig = rig
        .prepare(&shapes(pool(8, 3), &rows, &heads, &[]))
        .unwrap();
    let caps = rig.driver.caps().unwrap().clone();
    assert_eq!((caps.rows, caps.kv_blocks), (nz(2), 8));
    assert_eq!(caps.heads.suffix, [greedy, greedy.with_logprobs(0)]);
    assert!(!caps.heads.with_logits);
    assert_eq!(caps.row_counts, [nz(1), nz(2)]);
    assert_eq!(caps.widths, [nz(1)]);
    assert_eq!(rig.driver.compile_count(), 4);
    assert_eq!(rig.world.borrow().add_entry_calls, 4);

    let seqs: Vec<SeqId> = (0..3)
        .map(|i| open_one(&mut rig, &[5 + i], AdapterRef::Base))
        .collect();
    // Prefill each prompt's one token (one row, two rows), then decode in every composition.
    let first = step_rows(&mut rig, &mut [(seqs[0], PREFILL1, plain())]).unwrap();
    assert_eq!(first, [[1]]);
    step_rows(
        &mut rig,
        &mut [(seqs[1], PREFILL1, plain()), (seqs[2], PREFILL1, hosted())],
    )
    .unwrap();
    for round in 0..3 {
        step_rows(&mut rig, &mut [(seqs[0], DECODE, plain())]).unwrap();
        step_rows(
            &mut rig,
            &mut [(seqs[1], DECODE, plain()), (seqs[2], DECODE, plain())],
        )
        .unwrap();
        step_rows(&mut rig, &mut [(seqs[round % 3], DECODE, hosted())]).unwrap();
        step_rows(
            &mut rig,
            &mut [(seqs[0], DECODE, plain()), (seqs[1], DECODE, hosted())],
        )
        .unwrap();
    }
    assert_eq!(rig.driver.compile_count(), 4, "no step compiled");
    assert_eq!(
        rig.world.borrow().add_entry_calls,
        4,
        "no step added an entry"
    );
    assert_eq!(rig.driver.prepared_entries(), 4);

    // A third live row needs a fifth schema: the step is refused, nothing is compiled.
    let err = step_rows(
        &mut rig,
        &mut [
            (seqs[0], DECODE, plain()),
            (seqs[1], DECODE, plain()),
            (seqs[2], DECODE, plain()),
        ],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidRequest(InvalidRequest::TooManyRows { rows: 3, max: 2 })
        ),
        "{err:?}"
    );
    // Admitting it explicitly is refused at the limit; the old entries still run.
    let err = rig
        .driver
        .extend(&shapes(pool(8, 3), &[nz(3)], &heads, &[]))
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::Entries { held: 4, limit: 4 })
        ),
        "{err:?}"
    );
    step_rows(
        &mut rig,
        &mut [(seqs[0], DECODE, plain()), (seqs[1], DECODE, hosted())],
    )
    .unwrap();
    assert_eq!(rig.driver.compile_count(), 4);

    // A driver with a larger prepared scope admits rows {1, 2, 3}.
    let wide = paged_rig(RigOptions {
        max_entries: 8,
        ..RigOptions::default()
    });
    let mut wide = wide
        .prepare(&shapes(pool(8, 3), &[nz(1), nz(2), nz(3)], &heads, &[]))
        .unwrap();
    assert_eq!(wide.driver.compile_count(), 6);
    let seqs: Vec<SeqId> = (0..3)
        .map(|i| open_one(&mut wide, &[5 + i], AdapterRef::Base))
        .collect();
    let mut all: Vec<_> = seqs.iter().map(|&s| (s, PREFILL1, plain())).collect();
    step_rows(&mut wide, &mut all).unwrap();
    assert_eq!(wide.driver.compile_count(), 6);
}

/// A step the prepared rows cannot hold, with the rows admitted but not the shape (a token width no
/// prepare admitted), is a typed refusal that compiles nothing.
#[test]
fn a_shape_prepare_never_admitted_is_refused_without_compiling() {
    let rig = paged_rig(RigOptions::default());
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(2)], &[Head::GREEDY], &[]))
        .unwrap();
    let before = rig.driver.compile_count();
    let seq = open_one(&mut rig, &[1, 2, 3], AdapterRef::Base);
    // A three-token prefill plans a 2-token piece, which `Warm::Prompts(&[])` never admitted.
    let err = step_rows(
        &mut rig,
        &mut [(seq, RowWork::Prefill { upto: 3 }, plain())],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::NotAdmitted { .. })
        ),
        "{err:?}"
    );
    assert_eq!(rig.driver.compile_count(), before);
}

fn set(source: u64) -> AdapterSet {
    AdapterSet {
        adapters: ["a", "b"]
            .map(|name| AdapterSpec {
                name: name.to_string(),
                shape: vec![4, 2],
            })
            .to_vec(),
        source: AdapterSource(std::num::NonZeroU64::new(source).unwrap()),
    }
}

/// Two adapters ("a", "b") of rank 2 over the fake family's one projection (`in = 4`, `out = VOCAB`),
/// deterministic from `seed`.
fn adapters(seed: u32) -> Vec<poot_load::lora::LoraAdapter> {
    ["a", "b"]
        .into_iter()
        .enumerate()
        .map(|(i, _)| {
            let fill = |n: usize, salt: u32| -> Vec<f32> {
                (0..n)
                    .map(|k| {
                        (((k as u32 + 1) * (seed * 31 + i as u32 * 7 + salt)) % 13) as f32 / 13.0
                            - 0.5
                    })
                    .collect()
            };
            poot_load::lora::LoraAdapter {
                config: poot_load::lora::LoraAdapterConfig {
                    r: 2,
                    lora_alpha: 4.0,
                    target_modules: vec!["q_proj".to_string()],
                    use_rslora: false,
                },
                weights: std::collections::HashMap::from([(
                    "model.layers.0.self_attn.q_proj".to_string(),
                    poot_load::lora::LoraWeight {
                        a: crate::driver::lora::test_only::f32_factor(vec![2, 4], &fill(8, 1)),
                        b: crate::driver::lora::test_only::f32_factor(vec![12, 2], &fill(24, 2)),
                    },
                )]),
            }
        })
        .collect()
}

fn lora_rig(options: RigOptions) -> Rig {
    paged_rig(RigOptions {
        variant: Variant::Lora,
        ..options
    })
}

fn lora_idx(index: u32) -> LoraIdx {
    LoraIdx::new(index).unwrap()
}

/// Whether an input buffer of the last replayed step holds exactly `expected` as i32 words: the
/// `Slot::LoraIdx` row values. The step's token buffer is `[3]`/`[4]`/`[5]` and its position buffer `[0]`,
/// so a one- or two-valued adapter index is told apart by its value.
fn last_step_bound(rig: &Rig<Serving>, expected: &[i32]) -> bool {
    let world = rig.world.borrow();
    let last = world.steps.last().expect("a step ran");
    last.inputs.iter().any(|bytes| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .eq(expected.iter().copied())
    })
}

/// SC-003: two captured sets with identical names and shapes but distinct source identity are two
/// generations with their own entries; a set captured again is the same generation; requests keep
/// their generation; changing only the `LoraIdx` inside one set reuses the entry and binds the new
/// index; unloading a generation a live request holds is a typed refusal; one step cannot mix
/// generations. Mutation: capture on the logical names only; the second set is the first generation.
#[test]
fn adapter_generations_have_their_own_entries_and_indices_do_not() {
    let mut rig = lora_rig(RigOptions::default());
    let g1 = rig
        .driver
        .capture_adapter_set(set(1), &adapters(1))
        .unwrap();
    assert_eq!(
        rig.driver
            .capture_adapter_set(set(1), &adapters(1))
            .unwrap(),
        g1,
        "the same set"
    );
    let g2 = rig
        .driver
        .capture_adapter_set(set(2), &adapters(2))
        .unwrap();
    assert_ne!(g1, g2, "a different source is a different generation");

    let heads = [Head::GREEDY];
    let mut rig = rig
        .prepare(&shapes(pool(8, 4), &[nz(1), nz(2)], &heads, &[g1, g2]))
        .unwrap();
    let caps = rig.driver.caps().unwrap().clone();
    assert_eq!(caps.lora.map(|lora| lora.generations), Some(2));
    assert_eq!(
        rig.driver.compile_count(),
        6,
        "base, g1 and g2 each compile once per row count"
    );

    let on = |generation, index| AdapterRef::Adapter {
        generation,
        index: lora_idx(index),
    };
    let first = open_one(&mut rig, &[3], on(g1, 1));
    let second = open_one(&mut rig, &[4], on(g1, 2));
    let other = open_one(&mut rig, &[5], on(g2, 1));
    step_rows(&mut rig, &mut [(first, PREFILL1, plain())]).unwrap();
    assert!(last_step_bound(&rig, &[1]));
    step_rows(&mut rig, &mut [(second, PREFILL1, plain())]).unwrap();
    assert!(last_step_bound(&rig, &[2]), "the index is a bound input");
    step_rows(&mut rig, &mut [(other, PREFILL1, plain())]).unwrap();
    assert_eq!(
        rig.driver.compile_count(),
        6,
        "indices and generations switch without compiling"
    );

    let err = step_rows(
        &mut rig,
        &mut [(first, DECODE, plain()), (other, DECODE, plain())],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidRequest(InvalidRequest::MixedAdapters)
        ),
        "{err:?}"
    );

    // The generation a live request holds cannot be unloaded; once released it can, and the other
    // generation's entry is untouched.
    let err = rig.driver.unload_adapter_set(g1).unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidRequest(InvalidRequest::Adapter(LoraError::InUse { live: 2, .. }))
        ),
        "{err:?}"
    );
    rig.driver.release(first, Release::Finished).unwrap();
    rig.driver.release(second, Release::Finished).unwrap();
    rig.driver.unload_adapter_set(g1).unwrap();
    assert_eq!(
        rig.driver.prepared_entries(),
        4,
        "g1's entries went with it"
    );
    step_rows(&mut rig, &mut [(other, DECODE, plain())]).unwrap();
    let err = rig
        .driver
        .open(SeqRequest::new(&[1], 1, on(g1, 1)))
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidRequest(InvalidRequest::Adapter(LoraError::UnknownGeneration(_)))
        ),
        "{err:?}"
    );
}

/// SC-008 (bytes): with room for exactly the base entry and one generation's, a new adapter
/// generation is refused on bytes and the retained entries keep running. Mutation: skip the byte
/// reservation; the second generation enters.
#[test]
fn a_new_adapter_generation_over_the_byte_budget_leaves_the_old_ones_running() {
    let mut rig = lora_rig(RigOptions {
        max_retained_bytes: 64,
        charge: |_| crate::driver::Retention::private(32),
        ..RigOptions::default()
    });
    let g1 = rig
        .driver
        .capture_adapter_set(set(1), &adapters(1))
        .unwrap();
    let g2 = rig
        .driver
        .capture_adapter_set(set(2), &adapters(2))
        .unwrap();
    let heads = [Head::GREEDY];
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(1)], &heads, &[g1]))
        .unwrap();
    assert_eq!(rig.driver.retained_bytes(), 64);
    let err = rig
        .driver
        .extend(&shapes(pool(4, 2), &[nz(1)], &heads, &[g2]))
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::PreparedSet(PreparedSetRefusal::Bytes {
                charge: 32,
                held: 64,
                would_retain: 96,
                limit: 64
            })
        ),
        "{err:?}"
    );
    assert_eq!(rig.driver.retained_bytes(), 64);
    assert_eq!(rig.driver.prepared_entries(), 2);
    let seq = open_one(
        &mut rig,
        &[2],
        AdapterRef::Adapter {
            generation: g1,
            index: lora_idx(1),
        },
    );
    let before = rig.world.borrow().add_entry_calls;
    step_rows(&mut rig, &mut [(seq, PREFILL1, plain())]).unwrap();
    assert_eq!(rig.world.borrow().add_entry_calls, before);
}

/// SC-007: a family whose tracer declares a state value with axis 1 as the row axis fails `prepare`
/// with a typed `ShapeUnsupported`, naming the leading axis it found; a state with the row axis first
/// prepares. Mutation: skip the row-axis check; the malformed fixture prepares.
#[test]
fn a_state_value_without_a_leading_row_axis_fails_prepare() {
    let bad = |rows| {
        let rig = paged_rig(RigOptions {
            variant: Variant::BadRowAxis,
            ..RigOptions::default()
        });
        let heads = [Head::GREEDY];
        rig.prepare(&shapes(pool(4, rows), &[nz(rows)], &heads, &[]))
            .map(|_| ())
    };
    for rows in [1, 2] {
        let err = bad(rows).unwrap_err();
        let DriverError::Unsupported(Unsupported::Trace(TraceError::ShapeUnsupported {
            reason,
            shape,
            ..
        })) = err
        else {
            panic!("{err:?}")
        };
        assert_eq!(reason, ShapeReason::StateRowAxis { leading: 4, rows });
        assert_eq!(shape.rows.get(), rows);
    }
    let ok = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    ok.prepare(&shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
}

fn prompt_of(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 1 + (i * 7 + salt) % 10).collect()
}

/// Drive `prompt` to its first committed token and release it as finished, returning `release`'s
/// outcome; the prefix cache then holds (or does not) what the outcome says.
fn finish_prompt(rig: &mut Rig<Serving>, prompt: &[u32]) -> (usize, PrefixOutcome) {
    finish_prompt_as(rig, prompt, AdapterRef::Base)
}

fn finish_prompt_as(
    rig: &mut Rig<Serving>,
    prompt: &[u32],
    adapter: AdapterRef,
) -> (usize, PrefixOutcome) {
    let admission = rig
        .driver
        .open(SeqRequest::new(prompt, 2, adapter))
        .unwrap();
    let Admission::Opened { seq, reused } = admission else {
        panic!("{admission:?}")
    };
    let mut done = reused;
    while done < prompt.len() {
        let mut sampler = plain();
        let mut rows = [StepRow {
            seq,
            work: RowWork::Prefill { upto: prompt.len() },
            sampler: &mut sampler,
        }];
        let result = rig.driver.step(&mut rows).unwrap().remove(0);
        done += result.consumed;
        rig.driver
            .commit(seq, result.tokens.len(), &mut sampler)
            .unwrap();
    }
    (reused, rig.driver.release(seq, Release::Finished).unwrap())
}

/// SC-005: the hybrid-state guard lives in `release`. On a fake family with one recurrent state,
/// `release(.., Finished)` is `Skipped(RecurrentState)`, registers no prefix, and a second request over
/// the same prompt reuses nothing; the dense fixture is `Registered` and the second request reuses its
/// two blocks. Mutation: drop the residency check in `release`; the recurrent fixture registers and the
/// second request reuses blocks that hold none of its state.
#[test]
fn release_skips_recurrent_state_and_registers_dense_prefixes() {
    let prompt = prompt_of(35, 3);
    let admit = |variant| {
        let rig = paged_rig(RigOptions {
            variant,
            capacity: 64,
            ..RigOptions::default()
        });
        let rows = [nz(1)];
        let heads = [Head::GREEDY];
        rig.prepare(&ServingShapes {
            layout: Layout::Paged(pool(6, 1)),
            rows: &rows,
            heads: &heads,
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap()
    };
    let mut dense = admit(Variant::Plain);
    assert_eq!(
        finish_prompt(&mut dense, &prompt),
        (0, PrefixOutcome::Registered { tokens: 32 })
    );
    assert_eq!(
        finish_prompt(&mut dense, &prompt).0,
        32,
        "the dense prefix is reused"
    );

    let mut hybrid = admit(Variant::Recurrent);
    assert_eq!(
        hybrid.driver.caps().unwrap().state,
        crate::driver::StateResidency::Recurrent
    );
    assert_eq!(
        finish_prompt(&mut hybrid, &prompt),
        (0, PrefixOutcome::Skipped(PrefixSkip::RecurrentState))
    );
    assert_eq!(
        finish_prompt(&mut hybrid, &prompt).0,
        0,
        "a recurrent family reuses nothing"
    );
}

/// POOT-1148: the KV write map is forward: one entry per flat token (`rows * width`), the pool slot the
/// token's position reads through the same step's read map, and -1 for a masked row, so the append costs the
/// tokens and not the pool. Two live rows step in a three-row entry, at positions 0 and then 1, and in
/// swapped order.
/// Mutation: write token `t` at the slot of the NEXT position (`map[position + 1]` in
/// `Driver::run_plans`): the entries differ from `read[row * CAP + position]` and the first assert goes red.
#[test]
fn the_kv_write_map_names_each_tokens_slot_and_pads_masked_rows() {
    const CAP: usize = 64;
    let rig = paged_rig(RigOptions::default());
    let rows = [nz(1), nz(3)];
    let mut rig = rig
        .prepare(&shapes(pool(8, 3), &rows, &[Head::GREEDY], &[]))
        .unwrap();
    let a = open_one(&mut rig, &[1], AdapterRef::Base);
    let b = open_one(&mut rig, &[2], AdapterRef::Base);

    // The read map of the last step: row `r`'s `CAP` logical positions to pool slots, three rows.
    let read_of = |rig: &Rig<Serving>| -> Vec<i32> {
        last_step_inputs(rig)
            .into_iter()
            .find(|words| words.len() == 3 * CAP)
            .expect("the step binds its read map")
    };
    let check = |rig: &Rig<Serving>, positions: [usize; 2]| {
        let read = read_of(rig);
        let want = vec![read[positions[0]], read[CAP + positions[1]], -1];
        let inputs = last_step_inputs(rig);
        assert!(
            inputs.contains(&want),
            "write map {want:?} (token -> slot of its position, masked row -1) not in {inputs:?}"
        );
    };

    step_rows(
        &mut rig,
        &mut [(a, PREFILL1, plain()), (b, PREFILL1, plain())],
    )
    .unwrap();
    check(&rig, [0, 0]);
    step_rows(&mut rig, &mut [(a, DECODE, plain()), (b, DECODE, plain())]).unwrap();
    check(&rig, [1, 1]);
    // Swapped rows: the map follows the step's row order, not the sequences' slots.
    step_rows(&mut rig, &mut [(b, DECODE, plain()), (a, DECODE, plain())]).unwrap();
    let read = read_of(&rig);
    let want = vec![read[2], read[CAP + 2], -1];
    assert!(
        last_step_inputs(&rig).contains(&want),
        "swapped rows: {want:?}"
    );
}

/// POOT-1074 SC-001: admitting a sequence into a recurrent model zeroes that sequence's state row,
/// through `reset_state(Row(slot))`, and writes nothing else into the state. The fake's `toy.state` is
/// `[1, 4]` f32 per slot: sixteen bytes at the slot's offset.
/// Mutation: drop the reset in `Driver::open` (the `private || residency == Recurrent` branch); the
/// admission writes no state and the row goes red.
#[test]
fn opening_a_recurrent_sequence_zeroes_its_state_row() {
    let rig = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    let mut rig = rig
        .prepare(&shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    let before = rig.world.borrow().state_writes.len();
    open_one(&mut rig, &[1], AdapterRef::Base);
    let writes = rig.world.borrow().state_writes[before..].to_vec();
    assert_eq!(writes, [(0, 16)]);
}

/// Every input the last step bound, read as the i32 words an index slot is stored as.
fn last_step_inputs(rig: &Rig<Serving>) -> Vec<Vec<i32>> {
    rig.world
        .borrow()
        .last
        .inputs
        .iter()
        .map(|bytes| {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| i32::from_le_bytes(*word))
                .collect()
        })
        .collect()
}

/// POOT-1074 SC-002: a recurrent model serves several sequences in one step. A sequence keeps the pool
/// row of its slot whichever step row it occupies: `state_read` names each row's slot and
/// `state_write` the pool row each step row's result is written to (-1: a masked row writes nothing).
/// Releasing a sequence and admitting another reuses its slot and zeroes exactly that row.
/// Mutation: bind `state_read[i] = i` (the row index, not the sequence's slot); the swapped step reads
/// slots `[0, 1]` instead of `[2, 0]`.
#[test]
fn recurrent_rows_address_their_slots_and_a_readmitted_slot_zeroes_only_its_row() {
    let rig = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    let rows = [nz(1), nz(3)];
    let mut rig = rig
        .prepare(&shapes(pool(8, 4), &rows, &[Head::GREEDY], &[]))
        .unwrap();
    let before = rig.world.borrow().state_writes.len();
    let a = open_one(&mut rig, &[1], AdapterRef::Base);
    let b = open_one(&mut rig, &[2], AdapterRef::Base);
    let c = open_one(&mut rig, &[3], AdapterRef::Base);
    // Row bytes: the fake's `toy.state` row is four f32.
    let resets = rig.world.borrow().state_writes[before..].to_vec();
    assert_eq!(
        resets,
        [(0, 16), (16, 16), (32, 16)],
        "one row per admission"
    );

    // Rows in the order C, A: C is slot 2 and A slot 0, in step rows 0 and 1. The third row is masked.
    step_rows(
        &mut rig,
        &mut [(c, PREFILL1, plain()), (a, PREFILL1, plain())],
    )
    .unwrap();
    let inputs = last_step_inputs(&rig);
    assert!(
        inputs.contains(&vec![2, 0, 0]),
        "state_read names slots [2, 0] and the masked row reads 0: {inputs:?}"
    );
    assert!(
        inputs.contains(&vec![2, 0, -1]),
        "state_write: row 0 writes slot 2, row 1 writes slot 0, the masked row writes nothing: {inputs:?}"
    );

    // B's slot is freed and re-used: only row 1 is zeroed.
    rig.driver.release(b, Release::Finished).unwrap();
    let before = rig.world.borrow().state_writes.len();
    open_one(&mut rig, &[4], AdapterRef::Base);
    let resets = rig.world.borrow().state_writes[before..].to_vec();
    assert_eq!(
        resets,
        [(16, 16)],
        "the re-admitted sequence zeroes its own slot's row"
    );
}

/// POOT-1074 SC-002: rows feeding different token counts cannot share a recurrent step (the shorter
/// row's padding would fold into its state), so the driver runs one entry per feed width and returns
/// the picks in row order. A dense model runs the same rows as one entry.
/// Mutation: group by nothing (`same_width` always true); the recurrent model runs one entry and the
/// row goes red.
#[test]
fn recurrent_rows_of_different_widths_run_as_one_entry_per_width() {
    let step_count = |variant| {
        let rig = paged_rig(RigOptions {
            variant,
            granule: 1,
            chunk: 4,
            ..RigOptions::default()
        });
        let mut rig = rig
            .prepare(&ServingShapes {
                warm: Warm::Prompts(&[4]),
                ..shapes(pool(8, 2), &[nz(1), nz(2)], &[Head::GREEDY], &[])
            })
            .unwrap();
        let long = open_one(&mut rig, &[1, 2, 3, 4], AdapterRef::Base);
        let short = open_one(&mut rig, &[4], AdapterRef::Base);
        // `short` prefills its one token, then decodes beside `long`'s three-token piece.
        step_rows(&mut rig, &mut [(short, PREFILL1, plain())]).unwrap();
        let before = rig.world.borrow().steps.len();
        let picked = step_rows(
            &mut rig,
            &mut [
                (long, RowWork::Prefill { upto: 4 }, plain()),
                (short, DECODE, plain()),
            ],
        )
        .unwrap();
        assert_eq!(picked, [vec![1], vec![1]], "picks come back in row order");
        rig.world.borrow().steps.len() - before
    };
    assert_eq!(step_count(Variant::Recurrent), 2, "one entry per width");
    assert_eq!(
        step_count(Variant::Plain),
        1,
        "a dense model pads to one entry"
    );
}

/// A sequence released as preempted frees its blocks and registers nothing, even on a dense family.
#[test]
fn a_preempted_sequence_registers_no_prefix() {
    let rig = paged_rig(RigOptions {
        capacity: 64,
        ..RigOptions::default()
    });
    let mut rig = rig
        .prepare(&ServingShapes {
            layout: Layout::Paged(pool(6, 1)),
            rows: &[nz(1)],
            heads: &[Head::GREEDY],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
            chain: None,
        })
        .unwrap();
    let prompt = prompt_of(35, 5);
    let Admission::Opened { seq, .. } = rig
        .driver
        .open(SeqRequest::new(&prompt, 2, AdapterRef::Base))
        .unwrap()
    else {
        panic!("room")
    };
    assert_eq!(
        rig.driver.release(seq, Release::Preempted).unwrap(),
        PrefixOutcome::Skipped(PrefixSkip::Preempted)
    );
    assert_eq!(finish_prompt(&mut rig, &prompt).0, 0);
}

/// Misuse of the serving surface is a typed refusal, never a panic: an unknown or repeated sequence,
/// a step over an uncommitted result, a pool that cannot hold the request, a `generate` with every
/// sequence slot taken. (Serving before `prepare` no longer compiles: only a prepared driver serves.)
#[test]
fn misuse_is_a_typed_refusal() {
    let rig = paged_rig(RigOptions::default());
    let invalid = |err: DriverError| match err {
        DriverError::InvalidRequest(request) => request,
        other => panic!("{other:?}"),
    };
    let mut rig = rig
        .prepare(&shapes(pool(2, 2), &[nz(2)], &[Head::GREEDY], &[]))
        .unwrap();
    // Two blocks hold 32 positions; a 20-token prompt with 20 new tokens needs three.
    let Admission::NoRoom { needed_blocks } = rig
        .driver
        .open(SeqRequest::new(&[1; 20], 20, AdapterRef::Base))
        .unwrap()
    else {
        panic!("expected no room")
    };
    assert_eq!(needed_blocks, 1);
    let seq = open_one(&mut rig, &[1], AdapterRef::Base);
    step_rows(&mut rig, &mut [(seq, PREFILL1, plain())]).unwrap();
    let mut a = plain();
    let mut b = plain();
    let mut rows = [
        StepRow {
            seq,
            work: DECODE,
            sampler: &mut a,
        },
        StepRow {
            seq,
            work: DECODE,
            sampler: &mut b,
        },
    ];
    assert_eq!(
        invalid(rig.driver.step(&mut rows).unwrap_err()),
        InvalidRequest::DuplicateRow(seq)
    );
    // The first token was committed by the helper; stepping twice over one uncommitted result:
    let mut sampler = plain();
    let mut rows = [StepRow {
        seq,
        work: DECODE,
        sampler: &mut sampler,
    }];
    rig.driver.step(&mut rows).unwrap();
    let mut sampler = plain();
    let mut rows = [StepRow {
        seq,
        work: DECODE,
        sampler: &mut sampler,
    }];
    assert_eq!(
        invalid(rig.driver.step(&mut rows).unwrap_err()),
        InvalidRequest::UncommittedResult(seq)
    );
    // `generate` is a sequence on this same surface: with the pool full it is the typed refusal `open`
    // gives, and it leaves the open sequence alone.
    let err = run(&mut rig, request_generate());
    assert!(
        matches!(
            err,
            Err(DriverError::InvalidRequest(InvalidRequest::NoRoom {
                needed_blocks: 1
            }))
        ),
        "{err:?}"
    );
    rig.driver.release(seq, Release::Finished).unwrap();
    assert_eq!(
        invalid(rig.driver.release(seq, Release::Finished).unwrap_err()),
        InvalidRequest::UnknownSeq(seq)
    );
}

/// `generate` while the contiguous layout's one slot holds an open sequence is `SlotsBusy`, not a pool
/// refusal: that layout has no pool. Mutation: map the busy slot back to `NoRoom { needed_blocks: 0 }`;
/// the match fails.
#[test]
fn generate_with_the_contiguous_slot_taken_is_slots_busy() {
    let mut rig = rig_default(logits_script(vec![favoring(1)]));
    rig.driver.extend(&contiguous(&[Head::GREEDY])).unwrap();
    let seq = open_one(&mut rig, &[1], AdapterRef::Base);
    let err = run(&mut rig, request_generate());
    assert!(
        matches!(
            err,
            Err(DriverError::InvalidRequest(InvalidRequest::SlotsBusy))
        ),
        "{err:?}"
    );
    rig.driver.release(seq, Release::Finished).unwrap();
    run(&mut rig, request_generate()).unwrap();
}

fn request_generate() -> GenerateRequest {
    request(&[1, 2], 1, Sampler::greedy())
}

fn admit_paged(
    rig: Rig,
    pool: PoolShape,
    rows: &[NonZeroUsize],
    windows: &[NonZeroUsize],
) -> Rig<Serving> {
    rig.prepare(&ServingShapes {
        layout: Layout::Paged(pool),
        rows,
        heads: &[Head::GREEDY, Head::Logits],
        warm: Warm::Prompts(&[]),
        windows,
        adapters: &[],
        chain: None,
    })
    .unwrap()
}

/// A verify window is one greedy suffix entry for the whole step, so it is refused, typed, beside any
/// row the greedy suffix cannot reproduce (a penalised, guided or sampled row): that row would get an
/// unmasked argmax. Beside a greedy row it runs. Mutation: restore the unconditional greedy head for a
/// window step; the row beside the window is no longer refused.
#[test]
fn a_window_beside_a_row_the_greedy_suffix_cannot_pick_is_refused() {
    let mut rig = admit_paged(
        paged_rig(RigOptions::default()),
        pool(4, 2),
        &[nz(2)],
        &[nz(2)],
    );
    let a = open_one(&mut rig, &[3], AdapterRef::Base);
    let b = open_one(&mut rig, &[4], AdapterRef::Base);
    step_rows(
        &mut rig,
        &mut [(a, PREFILL1, plain()), (b, PREFILL1, plain())],
    )
    .unwrap();
    let window = RowWork::Decode { ahead: &[7] };
    let sampled = || Sampler::new(1.0, 0, 1.0, 5);
    for other in [hosted as fn() -> Sampler, sampled] {
        let err =
            step_rows(&mut rig, &mut [(a, window, plain()), (b, DECODE, other())]).unwrap_err();
        assert!(
            matches!(
                err,
                DriverError::InvalidRequest(InvalidRequest::WindowStepNeedsGreedy(seq)) if seq == b
            ),
            "{err:?}"
        );
    }
    let picks = step_rows(&mut rig, &mut [(a, window, plain()), (b, DECODE, plain())]).unwrap();
    assert_eq!((picks[0].len(), picks[1].len()), (2, 1));
}

/// A verify window rejects a width below two at `prepare`, before anything is admitted.
#[test]
fn a_window_narrower_than_two_is_a_typed_refusal() {
    let rig = paged_rig(RigOptions::default());
    let refused = rig
        .driver
        .prepare(&ServingShapes {
            windows: &[nz(1)],
            ..shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[])
        })
        .unwrap_err();
    let err = refused.error;
    assert!(
        matches!(
            err,
            DriverError::InvalidOptions(InvalidOptions::WindowBelowTwo { width: 1 })
        ),
        "{err:?}"
    );
    assert!(refused.driver.caps().is_none());
}

/// `VerifyWindows` on a recurrent model: refused by `prepare`, which gives the unprepared driver back.
fn refused_recurrent_window() -> (Box<crate::driver::caps::PrepareRefused>, Shared) {
    let rig = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    let refused = rig
        .driver
        .prepare(&ServingShapes {
            windows: &[nz(2)],
            ..shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[])
        })
        .unwrap_err();
    (refused, rig.world)
}

/// `VerifyWindows` on a recurrent model.
pub(super) fn witness_recurrent_verify_windows() -> DriverError {
    refused_recurrent_window().0.error
}

/// `Resume` of a sequence that carries request inputs, on the contiguous layout: the new prompt does not
/// carry the values the sequence's prompt does.
pub(super) fn witness_resume_with_inputs() -> DriverError {
    let mut rig = build_rig(
        RigOptions {
            variant: Variant::Sectioned,
            ..RigOptions::default()
        },
        logits_script(vec![favoring(1)]),
    )
    .unwrap()
    .serve();
    let Admission::Opened { seq, .. } = rig
        .driver
        .open(SeqRequest::new(&[1, 2], 2, AdapterRef::Base).with_inputs(text_positions(2)))
        .unwrap()
    else {
        panic!("the contiguous slot is free");
    };
    rig.driver
        .resume(seq, &[1, 2, 3], 1)
        .expect_err("a sequence that carries inputs is not resumed")
}

/// A refused `prepare` leaves the driver as it was: a recurrent model asked for a verify window is
/// refused (a rejected draft has already advanced its state), and the driver it gives back is still
/// unprepared (no caps, nothing to `open` on) instead of admitting up to `max_seqs`; a valid call
/// afterwards succeeds. Mutation: publish the pool before the residency check; the returned driver
/// then has caps after the refusal.
#[test]
fn a_refused_prepare_publishes_nothing() {
    let (refused, world) = refused_recurrent_window();
    let err = refused.error;
    assert!(
        matches!(
            err,
            DriverError::Unsupported(Unsupported::Incompatible(Incompatible {
                first: Feature::RecurrentState,
                second: Feature::VerifyWindows,
                ..
            }))
        ),
        "{err:?}"
    );
    assert!(refused.driver.caps().is_none());
    // The refused call compiled its entries, and state avals pin the pool's size: the retry asks for it.
    let mut rig = Rig {
        driver: refused
            .driver
            .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
            .unwrap(),
        world,
    };
    open_one(&mut rig, &[1], AdapterRef::Base);
}

/// The pool returns to its size when a sequence is released: a preempted one registers nothing and
/// frees every block; a finished one leaves its prompt blocks pinned in the prefix cache until another
/// request evicts them. Mutation: skip `cache.reset` in the preempted branch; the blocks leak.
#[test]
fn release_returns_every_block_to_the_pool() {
    let rig = paged_rig(RigOptions {
        capacity: 64,
        ..RigOptions::default()
    });
    let mut rig = rig
        .prepare(&ServingShapes {
            warm: Warm::Admitted,
            ..shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[])
        })
        .unwrap();
    assert_eq!(rig.driver.free_blocks(), Some(4));
    // Short prompt: nothing to register.
    finish_prompt(&mut rig, &prompt_of(5, 1));
    assert_eq!(rig.driver.free_blocks(), Some(4));
    // 35 tokens: two full blocks stay cached, one block frees.
    let prompt = prompt_of(35, 2);
    assert_eq!(
        finish_prompt(&mut rig, &prompt).1,
        PrefixOutcome::Registered { tokens: 32 }
    );
    assert_eq!(rig.driver.free_blocks(), Some(2));
    // A request for the whole pool evicts them; preempted, it gives all four back.
    let Admission::Opened { seq, reused } = rig
        .driver
        .open(SeqRequest::new(&prompt_of(40, 9), 24, AdapterRef::Base))
        .unwrap()
    else {
        panic!("the cached prefix is evicted for it")
    };
    assert_eq!((reused, rig.driver.free_blocks()), (0, Some(0)));
    rig.driver.release(seq, Release::Preempted).unwrap();
    assert_eq!(rig.driver.free_blocks(), Some(4));
}

/// The sampler's context is the sequence's: the kept tokens enter it at `commit`, a result committed
/// with `kept = 0` leaves it unchanged, and a verify window's drafts and discarded picks never do.
/// Mutation: observe at step time; the dropped result's token is in the history.
#[test]
fn only_committed_tokens_enter_the_sampler() {
    let mut rig = admit_paged(
        paged_rig(RigOptions::default()),
        pool(4, 1),
        &[nz(1)],
        &[nz(2)],
    );
    let seq = open_one(&mut rig, &[3], AdapterRef::Base);
    let mut sampler = plain();
    let step = |rig: &mut Rig<Serving>, work, sampler: &mut Sampler| {
        let mut rows = [StepRow { seq, work, sampler }];
        rig.driver.step(&mut rows).unwrap().remove(0).tokens
    };
    // The fake picks token 1 every time.
    assert_eq!(step(&mut rig, PREFILL1, &mut sampler), [1]);
    assert_eq!(sampler.seen(3), 1, "the prompt seeds the context");
    assert_eq!(sampler.seen(1), 0, "a pick is not history until committed");
    rig.driver.commit(seq, 1, &mut sampler).unwrap();
    assert_eq!(sampler.seen(1), 1, "the kept token is");
    // A window of two picks, one kept: one observation, and the draft (7) never.
    let picks = step(&mut rig, RowWork::Decode { ahead: &[7] }, &mut sampler);
    assert_eq!(picks, [1, 1]);
    assert_eq!(
        sampler.seen(1),
        1,
        "an uncommitted window pick is not history"
    );
    rig.driver.commit(seq, 1, &mut sampler).unwrap();
    assert_eq!((sampler.seen(1), sampler.seen(7)), (2, 0));
    // A result dropped with `kept = 0` leaves the sampler as the sequence is.
    assert_eq!(step(&mut rig, DECODE, &mut sampler), [1]);
    rig.driver.commit(seq, 0, &mut sampler).unwrap();
    assert_eq!(sampler.seen(1), 2);
}

/// Unloading an adapter generation is retryable and leaves nothing behind: a failing `remove_entry`
/// leaves the generation loaded with its entries (the driver still serves it); the retry drops the
/// entries and evicts the prefix blocks cached under that generation. Mutation: unload the registry
/// first; the failed call leaves the generation gone but its entries held.
#[test]
fn a_failed_unload_leaves_the_generation_and_a_retry_frees_its_blocks() {
    let mut rig = lora_rig(RigOptions {
        capacity: 64,
        ..RigOptions::default()
    });
    let g = rig
        .driver
        .capture_adapter_set(set(1), &adapters(1))
        .unwrap();
    let mut rig = rig
        .prepare(&ServingShapes {
            warm: Warm::Admitted,
            ..shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[g])
        })
        .unwrap();
    let adapter = AdapterRef::Adapter {
        generation: g,
        index: lora_idx(1),
    };
    let held = rig.driver.prepared_entries();
    assert_eq!(
        finish_prompt_as(&mut rig, &prompt_of(35, 4), adapter).1,
        PrefixOutcome::Registered { tokens: 32 }
    );
    assert_eq!(rig.driver.free_blocks(), Some(2));
    rig.world.borrow_mut().fail_remove = true;
    let err = rig.driver.unload_adapter_set(g).unwrap_err();
    assert!(matches!(err, DriverError::Device(_)), "{err:?}");
    assert_eq!(rig.driver.prepared_entries(), held, "no entry was removed");
    assert_eq!(
        rig.driver.caps().unwrap().lora.map(|l| l.generations),
        Some(1)
    );
    let probe = open_one(&mut rig, &[1], adapter);
    rig.driver.release(probe, Release::Finished).unwrap();
    rig.world.borrow_mut().fail_remove = false;
    rig.driver.unload_adapter_set(g).unwrap();
    assert!(rig.driver.prepared_entries() < held);
    assert!(rig.driver.caps().unwrap().lora.is_none());
    assert_eq!(
        rig.driver.free_blocks(),
        Some(4),
        "the unloaded generation's cached prefix blocks are freed"
    );
}

/// An executable that cannot unload leaves the generation registered, mapped and retryable: the entries
/// are gone, the generation still lists, capturing the same set returns it, and the retry frees it.
/// Mutation: drop the map entry and the registry generation before `executor.unload`; the failed attempt
/// strands the executable's weights and the retry finds no generation (`UnknownGeneration`).
#[test]
fn a_failed_executable_unload_leaves_the_generation_for_a_retry() {
    let mut rig = lora_rig(RigOptions {
        ..RigOptions::default()
    });
    let g = rig
        .driver
        .capture_adapter_set(set(1), &adapters(1))
        .unwrap();
    let mut rig = rig
        .prepare(&ServingShapes {
            warm: Warm::Admitted,
            ..shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[g])
        })
        .unwrap();
    rig.world.borrow_mut().fail_unload = true;
    let err = rig.driver.unload_adapter_set(g).unwrap_err();
    assert!(matches!(err, DriverError::Device(_)), "{err:?}");
    assert_eq!(
        rig.driver
            .capture_adapter_set(set(1), &adapters(1))
            .unwrap(),
        g,
        "the generation is still registered: the same set is the same generation"
    );
    rig.world.borrow_mut().fail_unload = false;
    rig.driver.unload_adapter_set(g).unwrap();
    assert!(rig.driver.caps().unwrap().lora.is_none());
}

/// Two module paths that name one (layer, projection) are refused, not overwritten.
#[test]
fn an_adapter_naming_one_projection_twice_is_a_typed_refusal() {
    let mut rig = lora_rig(RigOptions::default());
    let mut twice = adapters(1);
    let weight = twice[0].weights["model.layers.0.self_attn.q_proj"].clone();
    // Leading zeros parse to the same layer index.
    twice[0]
        .weights
        .insert("model.layers.00.self_attn.q_proj".to_string(), weight);
    let err = rig.driver.capture_adapter_set(set(1), &twice).unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::Unsupported(crate::driver::error::Unsupported::Adapter(
                crate::driver::lora::AdapterError::DuplicateModule {
                    adapter: 0,
                    layer: 0,
                    ..
                }
            ))
        ),
        "{err:?}"
    );
}

/// One decode loop (Card 1016): `generate` is the serving surface driven for one row. The same script
/// through `generate` and through hand-driven `open`/`step`/`commit` reaches the executor with the same
/// steps (kind, tokens, positions) and yields the same tokens, so `generate` holds no pick or feed
/// logic of its own. Mutation: have `generate` decode with a sampler of its own instead of the
/// request's; the tokens or the bound sampler inputs diverge from the surface loop.
#[test]
fn generate_is_the_serving_surface_driven_for_one_row() {
    let script = || logits_script(vec![favoring(1), favoring(2), favoring(3), favoring(4)]);
    let prompt = [5u32, 6, 7, 8, 9];
    let max_new = 3;
    let mut generated = rig_default(script());
    let out = run(&mut generated, request(&prompt, max_new, Sampler::greedy())).unwrap();

    let mut manual = rig_default(script());
    manual.driver.extend(&contiguous(&[Head::GREEDY])).unwrap();
    let Admission::Opened { seq, .. } = manual
        .driver
        .open(SeqRequest::new(&prompt, max_new, AdapterRef::Base))
        .unwrap()
    else {
        panic!("the contiguous layout always has room")
    };
    let mut sampler = Sampler::greedy();
    let mut work = RowWork::Prefill { upto: prompt.len() };
    let mut tokens = Vec::new();
    while tokens.len() < max_new {
        let mut rows = [StepRow {
            seq,
            work,
            sampler: &mut sampler,
        }];
        let result = manual.driver.step(&mut rows).unwrap().remove(0);
        let Some(&token) = result.tokens.first() else {
            continue;
        };
        manual.driver.commit(seq, 1, &mut sampler).unwrap();
        tokens.push(token);
        work = DECODE;
    }
    let shapes = |rig: &Rig<Serving>| -> Vec<(Option<StepKind>, Vec<i32>, Vec<i32>)> {
        rig.world
            .borrow()
            .steps
            .iter()
            .map(|s| (s.kind, s.tokens.clone(), s.positions.clone()))
            .collect()
    };
    assert_eq!(out.tokens, tokens);
    assert_eq!(shapes(&generated), shapes(&manual));
    assert_eq!(generated.driver.stats().steps, manual.driver.stats().steps);
}

/// Embedding rows over some of a prompt's positions: zeros of the fake family's width.
fn embed_rows(positions: Vec<usize>) -> RequestInputs {
    let rows = vec![0.0; positions.len() * 4];
    poot_models::components::embed::embeds_inputs(4, positions, rows).unwrap()
}

/// Sectioned positions (`[len, 3]` I32) over `len` tokens, text positions on every axis.
fn text_positions(len: usize) -> RequestInputs {
    let at: Vec<i32> = (0..len as i32).flat_map(|p| [p; 3]).collect();
    let values = TokenValues::dense(HostTensor::i32(vec![len, 3], at))
        .unwrap()
        .with_beyond(Fill::Advance {
            from: vec![len as i32; 3],
            step: 1,
        });
    RequestInputs::none().with(MROPE, InputValue::Tokens(values))
}

/// POOT-1076: a request that carries embedding rows is served by the entries `prepare_carried`
/// admitted, and only the base weights have them (adapter entries are never prepared with carried
/// inputs), so the same request under an adapter is refused `InputNotPrepared` before it holds a slot
/// or blocks, while under the base weights it opens. Mutation: drop the `key.adapters == generation`
/// filter of `Driver::check_inputs`; the base entry answers for the adapter, the open succeeds and the
/// row goes red.
#[test]
fn embedding_rows_under_an_adapter_are_refused_before_admission() {
    let mut rig = lora_rig(RigOptions::default());
    let g = rig
        .driver
        .capture_adapter_set(set(1), &adapters(1))
        .unwrap();
    let (rows, heads, generations) = ([nz(1)], [Head::GREEDY], [g]);
    let prepared = ServingShapes {
        warm: Warm::Prompts(&[3]),
        ..shapes(pool(4, 2), &rows, &heads, &generations)
    };
    let mut rig = rig.prepare(&prepared).unwrap();
    rig.driver.prepare_carried(&prepared, &[EMBEDS]).unwrap();
    let free = rig.driver.free_blocks();
    let adapter = AdapterRef::Adapter {
        generation: g,
        index: lora_idx(1),
    };
    let request =
        |adapter| SeqRequest::new(&[1, 2, 3], 2, adapter).with_inputs(embed_rows(vec![1]));
    let refusal = rig.driver.open(request(adapter)).unwrap_err();
    assert!(
        matches!(
            refusal,
            DriverError::InvalidRequest(InvalidRequest::InputNotPrepared { input })
                if input == EMBEDS
        ),
        "{refusal:?}"
    );
    assert_eq!(rig.driver.free_blocks(), free, "the refusal held no blocks");
    assert!(
        matches!(
            rig.driver.open(request(AdapterRef::Base)).unwrap(),
            Admission::Opened { .. }
        ),
        "control: the base weights serve the same rows"
    );
}

/// A request that carries a per-row tensor is served by the entry that traced its extent: a value that fits
/// the extent opens and steps (its slot binds the value padded to the extent), and a longer one is refused
/// at admission with the typed support defect, before it holds a slot or a block. Mutation: drop the
/// extent comparison in `Driver::check_inputs`; the long value is admitted and the row goes red.
#[test]
fn a_per_row_value_longer_than_the_traced_extent_is_refused_at_admission() {
    use super::{ROWS, ROWS_EXTENT};
    let rig = paged_rig(RigOptions {
        variant: Variant::Rows,
        ..RigOptions::default()
    });
    let (rows, heads) = ([nz(1)], [Head::GREEDY]);
    let prepared = ServingShapes {
        warm: Warm::Prompts(&[2]),
        ..shapes(pool(4, 2), &rows, &heads, &[])
    };
    let mut rig = rig.prepare(&prepared).unwrap();
    rig.driver.prepare_carried(&prepared, &[ROWS]).unwrap();
    let free = rig.driver.free_blocks();
    let carrying = |len: usize| {
        RequestInputs::none().with(
            ROWS,
            InputValue::Row(HostTensor::i32(vec![len], vec![1; len])),
        )
    };
    let refusal = rig
        .driver
        .open(SeqRequest::new(&[1, 2], 2, AdapterRef::Base).with_inputs(carrying(ROWS_EXTENT + 1)))
        .unwrap_err();
    assert!(
        matches!(
            refusal,
            DriverError::InvalidRequest(InvalidRequest::Input(InputError::Support { id, .. }))
                if id == ROWS
        ),
        "{refusal:?}"
    );
    assert_eq!(rig.driver.free_blocks(), free, "the refusal held no blocks");
    let Admission::Opened { seq, .. } = rig
        .driver
        .open(SeqRequest::new(&[1, 2], 2, AdapterRef::Base).with_inputs(carrying(ROWS_EXTENT)))
        .unwrap()
    else {
        panic!("the pool admits a value that fits")
    };
    let mut sampler = plain();
    let mut rows = [StepRow {
        seq,
        work: RowWork::Prefill { upto: 2 },
        sampler: &mut sampler,
    }];
    rig.driver
        .step(&mut rows)
        .expect("the entry that carries the row binds the padded value");
}

/// An encode row whose per-row tensor has no values has no positions and no entry shape: it is refused
/// `EmptyRow` instead of reaching the shape constructor. Mutation: drop the zero-position check in
/// `Driver::encode`; the row panics on the shape constructor and this row is red.
#[test]
fn an_encode_row_with_an_empty_per_row_tensor_is_refused() {
    use super::ROWS;
    use crate::driver::EncodeRequest;
    use crate::driver::EncodeRow;
    use crate::driver::error::EncodeError;
    let mut rig = paged_rig(RigOptions {
        variant: Variant::Rows,
        ..RigOptions::default()
    });
    let empty = RequestInputs::none().with(ROWS, InputValue::Row(HostTensor::i32(vec![0], vec![])));
    let refusal = rig
        .driver
        .encode(&EncodeRequest {
            rows: &[EncodeRow {
                tokens: None,
                inputs: &empty,
            }],
        })
        .unwrap_err();
    assert!(
        matches!(
            refusal,
            DriverError::InvalidRequest(InvalidRequest::Encode(EncodeError::EmptyRow { row: 0 }))
        ),
        "{refusal:?}"
    );
}

/// POOT-1076: a draft model prefills over the prompt's tokens and cannot read embedding rows, so its
/// proposals for such a prompt would come from placeholders: a sequence that carries rows is admitted
/// and decodes, but is never speculated (`NotSpeculated::Inputs`, counted as `skipped_inputs`), while
/// the same prompt without rows is (its row is not `NotSpeculated(Inputs)`). Mutation: drop the
/// `drafts_for` arm of the decode row's speculation choice in `Driver::step`; the row with rows
/// reports another outcome and the first assertion is red.
#[test]
fn embedding_rows_under_a_draft_model_are_admitted_but_not_speculated() {
    let decode_after_prefill = |inputs: RequestInputs| {
        let rig = paged_rig(RigOptions {
            draft_model_k: Some(nz(2)),
            ..RigOptions::default()
        });
        let (rows, heads) = ([nz(1)], [Head::GREEDY]);
        let prepared = ServingShapes {
            warm: Warm::Prompts(&[1]),
            ..shapes(pool(4, 2), &rows, &heads, &[])
        };
        let mut rig = rig.prepare(&prepared).unwrap();
        rig.driver.prepare_carried(&prepared, &[EMBEDS]).unwrap();
        let Admission::Opened { seq, .. } = rig
            .driver
            .open(SeqRequest::new(&[3], 4, AdapterRef::Base).with_inputs(inputs))
            .expect("the rows are admitted under a draft-model policy")
        else {
            panic!("the pool admits the sequence")
        };
        let mut sampler = plain();
        let step = |rig: &mut Rig<Serving>, sampler: &mut Sampler, work| {
            let mut rows = [StepRow { seq, work, sampler }];
            rig.driver.step(&mut rows).unwrap().remove(0)
        };
        let prefill = step(&mut rig, &mut sampler, PREFILL1);
        assert_eq!(prefill.tokens, [1], "the fake picks token 1");
        rig.driver.commit(seq, 1, &mut sampler).unwrap();
        let decode = step(&mut rig, &mut sampler, DECODE);
        (decode, rig.driver.stats().speculation)
    };
    let (with_rows, stats) = decode_after_prefill(embed_rows(vec![0]));
    assert_eq!(
        with_rows.speculation,
        RowSpeculation::NotSpeculated(NotSpeculated::Inputs)
    );
    assert_eq!(with_rows.tokens, [1], "the row decodes one token");
    assert_eq!((stats.skipped_inputs, stats.rounds), (1, 0));

    let (plain_row, stats) = decode_after_prefill(RequestInputs::none());
    assert_ne!(
        plain_row.speculation,
        RowSpeculation::NotSpeculated(NotSpeculated::Inputs),
        "control: the same prompt without rows is not skipped for its inputs"
    );
    assert_eq!(stats.skipped_inputs, 0);
}

/// POOT-650: sectioned positions that do not cover the prompt are refused at admission, before the
/// sequence holds a slot or blocks: a dense value over 2 positions for a 3-token prompt is
/// `InputError::Support`. Mutation: drop the dense-length arm of `check_value` (`ModelSchema::
/// check_request`); the open reaches the entry lookup and the row goes red.
#[test]
fn mrope_positions_that_do_not_cover_the_prompt_are_refused_before_admission() {
    let rig = paged_rig(RigOptions {
        variant: Variant::Sectioned,
        ..RigOptions::default()
    });
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    let free = rig.driver.free_blocks();
    let request = |positions: usize| {
        SeqRequest::new(&[1, 2, 3], 2, AdapterRef::Base).with_inputs(text_positions(positions))
    };
    let refusal = rig.driver.open(request(2)).unwrap_err();
    assert!(
        matches!(
            refusal,
            DriverError::InvalidRequest(InvalidRequest::Input(InputError::Support { id, .. }))
                if id == MROPE
        ),
        "{refusal:?}"
    );
    assert_eq!(rig.driver.free_blocks(), free, "the refusal held no blocks");
    assert!(
        matches!(
            rig.driver.open(request(3)).unwrap(),
            Admission::Opened { .. }
        ),
        "control: positions over the whole prompt are admitted"
    );
}

/// POOT-650: positions for a model that declares no `MROPE` input are refused at admission
/// (`InputError::Unknown`), before the sequence holds a slot or blocks; the same prompt then opens as
/// plain tokens. Mutation: drop the declared-input lookup of `ModelSchema::check_request`; the open
/// succeeds (or reaches a later check) and the row goes red.
#[test]
fn mrope_positions_on_a_model_without_sectioned_rope_are_refused_before_admission() {
    let rig = paged_rig(RigOptions::default());
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    let free = rig.driver.free_blocks();
    let request = || SeqRequest::new(&[1, 2, 3], 2, AdapterRef::Base);
    let refusal = rig
        .driver
        .open(request().with_inputs(text_positions(3)))
        .unwrap_err();
    assert!(
        matches!(
            refusal,
            DriverError::InvalidRequest(InvalidRequest::Input(InputError::Unknown { id }))
                if id == MROPE
        ),
        "{refusal:?}"
    );
    assert_eq!(rig.driver.free_blocks(), free, "the refusal held no blocks");
    assert!(matches!(
        rig.driver.open(request()).unwrap(),
        Admission::Opened { .. }
    ));
}

fn encode_tokens<S: State>(driver: &mut Driver<S>) -> Result<crate::driver::Encoded, DriverError> {
    let ids = [1u32, 2];
    driver.prepare_encode(&crate::driver::EncodeShapes {
        rows: NonZeroUsize::MIN,
        max_len: NonZeroUsize::new(ids.len()).unwrap(),
    })?;
    let none = RequestInputs::none();
    driver.encode(&crate::driver::EncodeRequest {
        rows: &[crate::driver::EncodeRow {
            tokens: Some(&ids),
            inputs: &none,
        }],
    })
}

fn encode_request<S: State>(
    driver: &mut Driver<S>,
    rows: &[&[u32]],
) -> Result<crate::driver::Encoded, DriverError> {
    let none = RequestInputs::none();
    let rows: Vec<crate::driver::EncodeRow<'_>> = rows
        .iter()
        .map(|ids| crate::driver::EncodeRow {
            tokens: Some(ids),
            inputs: &none,
        })
        .collect();
    driver.encode(&crate::driver::EncodeRequest { rows: &rows })
}

/// POOT-1204: a token step runs only on an entry `prepare_encode` admitted. Mutation: prepare the
/// entry on a miss in `encode` (restore the lazy path): the first request compiles and succeeds.
#[test]
fn token_rows_are_refused_until_encode_shapes_are_prepared() {
    let mut rig = rig_default(picks_one());
    let error = encode_request(&mut rig.driver, &[&[1, 2]]).expect_err("nothing is prepared");
    assert!(
        matches!(
            error,
            DriverError::InvalidRequest(InvalidRequest::Encode(
                crate::driver::error::EncodeError::NotPrepared
            ))
        ),
        "{error:?}"
    );
    assert_eq!(
        rig.driver.stats().compiles,
        0,
        "the refusal compiled nothing"
    );
}

/// POOT-1204: a request of more rows or a longer row than the prepared shapes is a typed refusal that
/// compiles nothing, and a shorter or narrower one runs on the prepared entries. Mutation: drop the
/// `TooManyRows` or `RowTooLong` check: the request pads past the entry and the row goes red.
#[test]
fn token_rows_over_the_prepared_shapes_are_refused_and_smaller_ones_run() {
    use crate::driver::error::EncodeError;
    let mut rig = rig_default(picks_one());
    rig.driver
        .prepare_encode(&crate::driver::EncodeShapes {
            rows: NonZeroUsize::new(2).unwrap(),
            max_len: NonZeroUsize::new(4).unwrap(),
        })
        .expect("the shapes prepare");
    let compiled = rig.driver.stats().compiles;

    let error = encode_request(&mut rig.driver, &[&[1], &[1], &[1]]).expect_err("three rows");
    assert!(
        matches!(
            error,
            DriverError::InvalidRequest(InvalidRequest::Encode(EncodeError::TooManyRows {
                rows: 3,
                max: 2
            }))
        ),
        "{error:?}"
    );
    let error = encode_request(&mut rig.driver, &[&[1, 2, 3, 4, 5]]).expect_err("five tokens");
    assert!(
        matches!(
            error,
            DriverError::InvalidRequest(InvalidRequest::Encode(EncodeError::RowTooLong {
                row: 0,
                tokens: 5,
                max: 4
            }))
        ),
        "{error:?}"
    );

    let one = encode_request(&mut rig.driver, &[&[1, 2, 3]]).expect("one short row runs");
    assert_eq!(one.rows.len(), 1, "padded rows are dropped from the answer");
    assert_eq!(
        rig.driver.stats().compiles,
        compiled,
        "serving never compiles"
    );
}

/// POOT-1084 SC-001: a device error from one `Encode` step is that step's failure, not the driver's.
/// The step's executable is reset (the graph's cache is scratch), the error is `EncodeStep` in class
/// `Step`, and the next encode on the same driver runs. Mutation: return the raw `Device` error from
/// `encode` (skip `recover_encode`): the class is `Fault` and the row goes red.
#[test]
fn a_device_error_in_one_encode_step_fails_that_step_and_the_driver_encodes_again() {
    let mut rig = rig_default(picks_one());
    encode_tokens(&mut rig.driver).expect("the first encode runs");

    rig.world.borrow_mut().fail_synchronize = true;
    let error = encode_tokens(&mut rig.driver).expect_err("the device fails the step");
    assert!(matches!(error, DriverError::EncodeStep(_)), "{error:?}");
    assert_eq!(error.class(), crate::driver::error::FailureClass::Step);

    rig.world.borrow_mut().fail_synchronize = false;
    encode_tokens(&mut rig.driver).expect("the executable was reset, so the next encode runs");
}

/// POOT-1084 SC-002: a device that cannot even abort the failed step is lost (the engine poisons), and
/// that stays a `Fault`: the driver cannot prove its state intact. Mutation: classify every device
/// error as `EncodeStep` (drop the reset in `recover_encode`): the poisoned device reads `Step`.
#[test]
fn a_device_lost_during_an_encode_step_stays_a_fault() {
    let mut rig = rig_default(picks_one());
    encode_tokens(&mut rig.driver).expect("the first encode runs");

    {
        let mut world = rig.world.borrow_mut();
        world.fail_synchronize = true;
        world.fail_abort = true;
    }
    for _ in 0..2 {
        let error = encode_tokens(&mut rig.driver).expect_err("the device is lost");
        assert!(
            matches!(&error, DriverError::Device(ExecError::Poisoned)),
            "the reset finds the poisoned device and the step reports that: {error:?}"
        );
        assert_eq!(error.class(), crate::driver::error::FailureClass::Fault);
    }
}

/// An id equal to the vocabulary size names no embedding row: `open` (with and without request
/// inputs) and `resume` each refuse it typed before the sequence holds anything. Mutation: drop the
/// vocabulary check from `check_admission`'s caller in one entry point; that row opens or resumes the
/// prompt instead.
#[test]
fn a_token_id_at_the_vocabulary_size_is_refused_at_every_open_entry() {
    let bad = [1, 2, VOCAB as u32];
    let refused = |error: DriverError| {
        assert!(
            matches!(
                error,
                DriverError::InvalidRequest(InvalidRequest::TokenOutOfVocab {
                    token,
                    at: 2,
                    vocab: VOCAB,
                }) if token == VOCAB as u32
            ),
            "{error:?}"
        );
    };
    let request = |prompt| SeqRequest::new(prompt, 2, AdapterRef::Base);

    let rig = paged_rig(RigOptions::default());
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    let free = rig.driver.free_blocks();
    refused(rig.driver.open(request(&bad)).unwrap_err());
    refused(
        rig.driver
            .open(request(&bad).with_inputs(embed_rows(vec![1])))
            .unwrap_err(),
    );
    assert_eq!(rig.driver.free_blocks(), free, "a refusal held no blocks");

    let mut rig = rig_default(picks_one());
    rig.driver.extend(&contiguous(&[Head::GREEDY])).unwrap();
    let seq = open_one(&mut rig, &[1, 2, 3], AdapterRef::Base);
    refused(rig.driver.resume(seq, &bad, 2).unwrap_err());
}

/// `max_new = usize::MAX` overflows the prompt-plus-reserve sum: it is the typed `Capacity` refusal at
/// `open` and `generate`, not an arithmetic panic. Mutation: `+` for `checked_add` in `check_admission`;
/// the debug build panics on overflow.
#[test]
fn an_overflowing_max_new_is_a_capacity_refusal() {
    let rig = paged_rig(RigOptions::default());
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    let expected = DriverError::InvalidRequest(InvalidRequest::Capacity {
        prompt: 3,
        max_new: usize::MAX,
        max: 64,
    });
    let error = rig
        .driver
        .open(SeqRequest::new(&[1, 2, 3], usize::MAX, AdapterRef::Base))
        .unwrap_err();
    assert_eq!(format!("{error:?}"), format!("{expected:?}"));
}

/// A row reset that fails inside `open` leaves the driver as it was: the slot is still free, no
/// sequence is recorded and the pool holds every block the admission took. Mutation: move the reset
/// after the slot pop and sequence insert; the slot, the sequence and the blocks leak.
#[test]
fn a_failed_row_reset_leaves_no_slot_sequence_or_blocks_behind() {
    let rig = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    let mut rig = rig
        .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    let before = (
        rig.driver.state.free_slots.clone(),
        rig.driver.state.seqs.len(),
        rig.driver.free_blocks(),
    );
    rig.world.borrow_mut().fail_reset = true;
    let error = rig
        .driver
        .open(SeqRequest::new(&[1, 2, 3], 2, AdapterRef::Base))
        .unwrap_err();
    assert!(matches!(error, DriverError::Device(_)), "{error:?}");
    let after = (
        rig.driver.state.free_slots.clone(),
        rig.driver.state.seqs.len(),
        rig.driver.free_blocks(),
    );
    assert_eq!(after, before);
    rig.world.borrow_mut().fail_reset = false;
    open_one(&mut rig, &[1, 2, 3], AdapterRef::Base);
}

/// The adapter lease an `open` took is given back when its row reset fails: the generation can still be
/// unloaded. Mutation: drop the `lora.release` in `open_seq`'s reset-failure path; the unload is refused.
#[test]
fn a_failed_row_reset_gives_the_adapter_lease_back() {
    let mut rig = lora_rig(RigOptions::default());
    let g = rig
        .driver
        .capture_adapter_set(set(1), &adapters(1))
        .unwrap();
    let mut rig = rig
        .prepare(&ServingShapes {
            layout: Layout::Contiguous,
            rows: &[NonZeroUsize::MIN],
            heads: &[Head::GREEDY],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[g],
            chain: None,
        })
        .unwrap();
    let adapter = AdapterRef::Adapter {
        generation: g,
        index: lora_idx(1),
    };
    rig.world.borrow_mut().fail_reset = true;
    let error = rig
        .driver
        .open(SeqRequest::new(&[1, 2, 3], 2, adapter))
        .unwrap_err();
    assert!(matches!(error, DriverError::Device(_)), "{error:?}");
    assert_eq!(rig.driver.state.seqs.len(), 0);
    rig.driver
        .unload_adapter_set(g)
        .expect("a failed open holds no lease on the generation");
}

/// A `resume` whose row reset fails leaves the sequence as it was (tokens, positions done), so the
/// caller can retry or release it. Mutation: reset after the sequence is rewritten (the old order);
/// the sequence holds the new prompt while its row is dirty.
#[test]
fn a_failed_row_reset_in_resume_leaves_the_sequence_untouched() {
    let mut rig = rig_default(picks_one());
    rig.driver.extend(&contiguous(&[Head::GREEDY])).unwrap();
    let seq = open_one(&mut rig, &[1, 2, 3], AdapterRef::Base);
    let held = |rig: &Rig<Serving>| format!("{:?}", rig.driver.state.seqs[&seq]);
    let before = held(&rig);
    rig.world.borrow_mut().fail_reset = true;
    // A prompt sharing nothing with the held one restarts the row from zero.
    let error = rig.driver.resume(seq, &[4, 5], 2).unwrap_err();
    assert!(matches!(error, DriverError::Device(_)), "{error:?}");
    assert_eq!(held(&rig), before);
}

/// `beam` takes the same admission check as `open`: an overflowing `max_new` and an id at the
/// vocabulary size are typed refusals. Mutation: the old inline `+` (overflow panic) / no vocabulary
/// check in `beam`.
#[test]
fn beam_refuses_an_overflowing_max_new_and_an_out_of_vocab_id() {
    let mut rig = rig_default(picks_one());
    let request = |prompt: Vec<u32>, max_new| crate::driver::BeamRequest {
        prompt,
        max_new,
        width: nz(2),
        length_penalty: 1.0,
        ignore_eos: false,
    };
    let error = rig
        .driver
        .beam(request(vec![1, 2, 3], usize::MAX))
        .unwrap_err();
    assert_eq!(
        format!("{error:?}"),
        format!(
            "{:?}",
            crate::driver::BeamError::from(InvalidRequest::Capacity {
                prompt: 3,
                max_new: usize::MAX,
                max: 64
            })
        )
    );
    let error = rig
        .driver
        .beam(request(vec![1, VOCAB as u32], 2))
        .unwrap_err();
    assert!(
        matches!(
            error,
            crate::driver::BeamError::Driver(DriverError::InvalidRequest(
                InvalidRequest::TokenOutOfVocab { at: 1, .. }
            ))
        ),
        "{error:?}"
    );
}

/// A draw the caller drops is in the device history but not in the sampler: the next adjusted step
/// seeds the row again from the sampler, which holds only what was committed.
/// Mutation: keep the row synced when a commit keeps fewer tokens than the step drew; the second seed
/// never happens.
#[test]
fn a_dropped_draw_makes_the_next_adjusted_step_seed_the_history_row_again() {
    let adjusted = Head::Sample(
        poot_graph_ir::ops::sampling::SampleSpec::draw(poot_graph_ir::op::SampleRule::Greedy)
            .adjusted(),
    );
    let mut rig = paged_rig(RigOptions::default())
        .prepare(&shapes(pool(8, 2), &[nz(1)], &[adjusted], &[]))
        .unwrap();
    let mut sampler = Sampler::greedy().with_penalties(1.0, 1.0, 0.0);
    let seq = open_one(&mut rig, &[3], AdapterRef::Base);
    let seeds = |rig: &Rig<Serving>| {
        rig.world
            .borrow()
            .state_bytes
            .iter()
            .filter(|bytes| bytes.len() == 2 * VOCAB * 4)
            .count()
    };
    let step = |rig: &mut Rig<Serving>, sampler: &mut Sampler, work| {
        let mut rows = [StepRow { seq, work, sampler }];
        rig.driver.step(&mut rows).unwrap()
    };
    step(&mut rig, &mut sampler, RowWork::Prefill { upto: 1 });
    assert_eq!(seeds(&rig), 1);
    rig.driver.commit(seq, 1, &mut sampler).unwrap();
    step(&mut rig, &mut sampler, RowWork::Decode { ahead: &[] });
    assert_eq!(seeds(&rig), 1, "a committed draw keeps the row synced");
    // The caller drops this draw: the device row holds a token the sampler never saw.
    rig.driver.commit(seq, 0, &mut sampler).unwrap();
    step(&mut rig, &mut sampler, RowWork::Decode { ahead: &[] });
    assert_eq!(seeds(&rig), 2, "the row is seeded again from the sampler");
}

/// The head specs the sampling rows below use: the argmax with the penalties, bias and mask applied on
/// the device, and the same with two alternatives per token.
fn adjusted_heads() -> (poot_graph_ir::ops::sampling::SampleSpec, [Head; 2]) {
    let adjusted =
        poot_graph_ir::ops::sampling::SampleSpec::draw(poot_graph_ir::op::SampleRule::Greedy)
            .adjusted();
    (
        adjusted,
        [
            Head::Sample(adjusted),
            Head::Sample(adjusted.with_logprobs(2)),
        ],
    )
}

/// SC-001: a batched decode step whose rows carry penalties and a logit bias reads back the tokens and,
/// when a row asks for them, its top-n logprobs - the byte counter equals that, not vocabulary x 4 x
/// rows. The step is one launch of the adjusted head; no row's pick takes the host path.
///
/// Mutation: route a row with a penalty to the logits head (`step_head` returns `Head::Logits` for an
/// adjusted need and the pick runs on the host); the counter reads vocabulary x 4 x rows and goes red.
#[test]
fn a_batched_step_with_penalty_and_bias_rows_reads_back_only_tokens_and_logprobs() {
    // The logits head is prepared too, as it is for a driver that serves a sampled verify window: the
    // adjusted rows must still not take it.
    let (_, [adjusted, logprobs]) = adjusted_heads();
    let heads = [adjusted, logprobs, Head::Logits];
    let mut rig = paged_rig(RigOptions::default())
        .prepare(&shapes(pool(8, 3), &[nz(2)], &heads, &[]))
        .unwrap();
    let seqs: Vec<SeqId> = (0..2)
        .map(|i| open_one(&mut rig, &[5 + i], AdapterRef::Base))
        .collect();
    let penalty = || Sampler::greedy().with_penalties(1.3, 0.2, 0.1);
    let biased =
        || Sampler::greedy().with_logit_bias(std::collections::HashMap::from([(4u32, 2.0f32)]));
    let read = |rig: &Rig<Serving>| rig.driver.executor_stats().output_read_bytes;

    let before = read(&rig);
    step_rows(
        &mut rig,
        &mut [
            (seqs[0], PREFILL1, penalty()),
            (seqs[1], PREFILL1, biased()),
        ],
    )
    .unwrap();
    assert_eq!(
        read(&rig) - before,
        2 * 8,
        "two (token, flag) pairs, not two rows of {VOCAB} logits"
    );
    assert_eq!(rig.driver.stats().host_picks, 0);

    // A row that asks for two alternatives widens the record to its F32 form, still tokens and logprobs
    // only: (token, flag, logprob) and two (token, logprob) pairs per row.
    let before = read(&rig);
    let mut rows = [
        (seqs[0], DECODE, penalty().with_logprobs(2)),
        (seqs[1], DECODE, biased()),
    ];
    let picked = step_rows(&mut rig, &mut rows).unwrap();
    assert_eq!(picked, [[1], [1]]);
    assert_eq!(read(&rig) - before, 2 * (3 + 2 * 2) * 4);
    assert_eq!(rig.driver.stats().host_picks, 0);
}

/// SC-003: a request without logprobs or full logits reads no full-logits buffer. The buffer a step
/// reads back is the entry's output: a sampling head's is a token pair or a logprob record per row, and
/// only the logits head's is a vocabulary of logits per row. A driver that prepared no logits head has
/// no entry whose output is that big.
///
/// Mutation: make `with_head` return the raw logits graph for a sampling head (always write the full
/// logits out); the sampling entries' output holds a vocabulary per row and this row goes red.
#[test]
fn a_request_without_logprobs_or_full_logits_reads_no_full_logits_buffer() {
    let (_, heads) = adjusted_heads();
    let rig = paged_rig(RigOptions::default())
        .prepare(&shapes(pool(8, 3), &[nz(2)], &heads, &[]))
        .unwrap();
    let outputs = |rig: &Rig<Serving>| -> Vec<usize> {
        rig.driver
            .ledger
            .entries
            .values()
            .map(|entry| entry.output.iter().product())
            .collect()
    };
    let sampling = outputs(&rig);
    assert!(!sampling.is_empty(), "the entries have outputs");
    assert!(
        sampling.iter().all(|&elems| elems < 2 * VOCAB),
        "a sampling head's readback holds records, not logits: {sampling:?}"
    );

    // The control: the logits head's output is a vocabulary per row, so the measure can fail.
    let logits = paged_rig(RigOptions::default())
        .prepare(&shapes(pool(8, 3), &[nz(2)], &[Head::Logits], &[]))
        .unwrap();
    assert!(
        outputs(&logits).iter().all(|&elems| elems >= 2 * VOCAB),
        "{:?}",
        outputs(&logits)
    );
}

/// SC-004: a guided row's mask is built on the host while the device runs the forward pass, and applied by
/// the sampling stage on the device. The device events of the step, in order: the step's ordinary inputs
/// are written, the forward stage is replayed, only then is the mask written (the host builds it while that
/// stage runs), the sampling stage is replayed, the host waits once and reads back one (token, flag) pair.
/// The mask never reaches the device before the forward stage is on it, and nothing but the pair comes back.
///
/// Mutation: apply the mask on the host - `step_head` returns the logits head for a guided need and the
/// pick runs on the host; there is no mask write, the readback is a vocabulary of logits and this row
/// goes red.
#[test]
fn a_guided_row_builds_its_mask_while_the_forward_pass_runs() {
    let guided = Head::Sample(
        poot_graph_ir::ops::sampling::SampleSpec::draw(poot_graph_ir::op::SampleRule::Greedy)
            .guided(),
    );
    // The logits head is prepared too: the mutation needs somewhere to send the row.
    let mut rig = paged_rig(RigOptions::default())
        .prepare(&shapes(pool(8, 3), &[nz(1)], &[guided, Head::Logits], &[]))
        .unwrap();
    let seq = open_one(&mut rig, &[5], AdapterRef::Base);
    let text = tiny_text();
    let constraint = text.build_choice_constraint(&["c".to_string()]).unwrap();
    let sampler = Sampler::greedy().with_constraint(constraint);

    let before = rig.world.borrow().events.len();
    step_rows(&mut rig, &mut [(seq, PREFILL1, sampler)]).unwrap();
    let world = rig.world.borrow();
    let events = &world.events[before..];
    let (keys, _) = world
        .entry_slots
        .iter()
        .find(|(keys, _)| keys.iter().any(|key| key == "sampler.mask"))
        .expect("the guided head declares the mask slot");
    let entry = world
        .entry_slots
        .iter()
        .position(|(other, _)| other == keys)
        .unwrap();
    let ordinal = keys.iter().position(|key| key == "sampler.mask").unwrap();
    let mask = world.entry_inputs[entry][ordinal];

    let at = |wanted: DeviceEvent| -> Vec<usize> {
        events
            .iter()
            .enumerate()
            .filter(|(_, event)| **event == wanted)
            .map(|(i, _)| i)
            .collect()
    };
    let (replays, mask_writes) = (at(DeviceEvent::Replay), at(DeviceEvent::Write(mask)));
    assert_eq!(
        replays.len(),
        2,
        "forward stage, then sampling stage: {events:?}"
    );
    assert_eq!(mask_writes.len(), 1, "the mask is written once: {events:?}");
    assert!(
        replays[0] < mask_writes[0] && mask_writes[0] < replays[1],
        "the mask is built between the two stages: {events:?}"
    );
    assert!(
        events[..replays[0]]
            .iter()
            .all(|event| matches!(event, DeviceEvent::Write(_))),
        "only ordinary inputs are written before the forward stage runs: {events:?}"
    );
    assert_eq!(
        &events[replays[1] + 1..],
        [DeviceEvent::Wait, DeviceEvent::Read],
        "one wait after the sampling stage, then one readback: {events:?}"
    );
    drop(world);
    assert_eq!(
        rig.driver.executor_stats().output_read_bytes,
        8,
        "only the (token, flag) pair came back"
    );
}

/// SC-005: with no guided rows, k decode steps run with one host wait. A chain of four steps is four
/// replays and three device copies (each step's pick into the next step's token slot) behind one
/// completion wait and returns four tokens; the same four tokens as four ordinary steps wait four times.
///
/// Mutation: force a wait after every step of the chain (`step_chain` completes each entry); the chain
/// waits four times and this row goes red.
#[test]
fn four_chained_decode_steps_run_with_one_host_wait() {
    let depth = nz(4);
    let (heads, rows) = ([Head::GREEDY], [nz(1)]);
    let shapes = ServingShapes {
        chain: Some(depth),
        ..shapes(pool(8, 2), &rows, &heads, &[])
    };
    let mut rig = paged_rig(RigOptions::default()).prepare(&shapes).unwrap();
    let seq = open_one(&mut rig, &[5], AdapterRef::Base);
    step_rows(&mut rig, &mut [(seq, PREFILL1, plain())]).unwrap();

    let waits = |rig: &Rig<Serving>| rig.driver.executor_stats().waits;
    let before = (waits(&rig), rig.driver.executor_stats().replays);
    let mut sampler = plain();
    let mut rows = [StepRow {
        seq,
        work: RowWork::Chain { steps: depth },
        sampler: &mut sampler,
    }];
    let results = rig.driver.step(&mut rows).unwrap();
    assert_eq!(results[0].tokens, [1, 1, 1, 1]);
    let after = (waits(&rig), rig.driver.executor_stats().replays);
    assert_eq!(after.0 - before.0, 1, "four steps, one host wait");
    assert_eq!(after.1 - before.1, 4, "four entries replayed");
    assert_eq!(rig.driver.stats().chains, 1);
    rig.driver.commit(seq, 4, &mut sampler).unwrap();

    // The same four tokens one step at a time wait once each.
    let before = waits(&rig);
    for _ in 0..4 {
        step_rows(&mut rig, &mut [(seq, DECODE, plain())]).unwrap();
    }
    assert_eq!(waits(&rig) - before, 4, "four ordinary steps, four waits");
}

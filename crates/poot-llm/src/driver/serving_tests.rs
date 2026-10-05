//! Serving-surface rows on the counting fake executor (Card 735): entry counting, typed refusals,
//! adapter identity, state residency. The device is a fake, so these rows read what reached the
//! executor and what the driver published, never a token's value; token parity is
//! `serving_device_tests`.

use super::*;
use crate::driver::error::{InvalidRequest, PreparedSetRefusal};
use crate::driver::lora::{AdapterSet, AdapterSource, AdapterSpec, LoraError, LoraIdx};
use crate::driver::step::PickPolicy;
use crate::driver::{
    AdapterRef, Admission, PoolShape, PrefixOutcome, PrefixSkip, Release, RowWork, SeqId,
    SeqRequest, StepRow,
};

/// The fake device's answer to any step: every row picks token 1, or favors it in the logits.
fn picks_one() -> Script {
    Box::new(|_, len| {
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
    }
}

fn open_one(rig: &mut Rig, prompt: &[u32], adapter: AdapterRef) -> SeqId {
    let admission = rig
        .driver
        .open(SeqRequest {
            prompt,
            max_new: 16,
            adapter,
        })
        .unwrap();
    let Admission::Opened { seq, .. } = admission else {
        panic!("{admission:?}")
    };
    seq
}

/// One step of `works` over `seqs`, each with its own sampler; commits every pick.
fn step_rows(
    rig: &mut Rig,
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
const DECODE: RowWork<'static> = RowWork::Decode {
    ahead: &[],
    pick: PickPolicy::Greedy,
};

/// A greedy row, and a row the suffix cannot express (logprobs force the host path).
fn plain() -> Sampler {
    Sampler::greedy()
}

fn hosted() -> Sampler {
    Sampler::greedy().with_logprobs(0)
}

fn paged_rig(options: RigOptions) -> Rig {
    build_rig(options, picks_one()).unwrap()
}

/// SC-008: with `max_entries = 4`, rows {1, 2} x heads {suffix, logits} for one fixed schema are four
/// entries; exercising all four repeatedly with changing row assignments compiles and adds nothing; a
/// fifth static schema is refused (and a step that needs it is refused typed) while every old entry
/// still runs; a driver with a larger scope admits it. Mutations: drop `rows` from the entry identity
/// (two entries, not four), drop the head (two), skip the entry-count check (the fifth enters).
#[test]
fn rows_and_heads_are_four_entries_that_serve_every_composition() {
    let mut rig = paged_rig(RigOptions {
        max_entries: 4,
        ..RigOptions::default()
    });
    let rows = [nz(1), nz(2)];
    let heads = [Head::GREEDY, Head::Logits];
    let caps = rig
        .driver
        .prepare(&shapes(pool(8, 3), &rows, &heads, &[]))
        .unwrap()
        .clone();
    assert_eq!((caps.rows, caps.kv_blocks), (nz(2), 8));
    assert_eq!(caps.heads.suffix, [poot_graph_ir::op::SampleRule::Greedy]);
    assert!(caps.heads.with_logits);
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
        .prepare(&shapes(pool(8, 3), &[nz(3)], &heads, &[]))
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
    let mut wide = paged_rig(RigOptions {
        max_entries: 8,
        ..RigOptions::default()
    });
    wide.driver
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
    let mut rig = paged_rig(RigOptions::default());
    rig.driver
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

fn lora_rig(options: RigOptions) -> Rig {
    paged_rig(RigOptions {
        variant: Variant::Lora,
        ..options
    })
}

fn lora_idx(index: u32) -> LoraIdx {
    LoraIdx::new(index).unwrap()
}

/// Whether an input buffer of the last replayed step holds exactly `expected` as f32 words: the
/// `Slot::LoraIdx` row values (the token, position and map buffers hold i32 words, whose bit patterns
/// read as f32 are never a small whole number).
fn last_step_bound(rig: &Rig, expected: &[f32]) -> bool {
    let world = rig.world.borrow();
    let last = world.steps.last().expect("a step ran");
    last.inputs.iter().any(|bytes| floats(bytes) == expected)
}

/// SC-003: two captured sets with identical names and shapes but distinct source identity are two
/// generations with their own entries; a set captured again is the same generation; requests keep
/// their generation; changing only the `LoraIdx` inside one set reuses the entry and binds the new
/// index; unloading a generation a live request holds is a typed refusal; one step cannot mix
/// generations. Mutation: capture on the logical names only; the second set is the first generation.
#[test]
fn adapter_generations_have_their_own_entries_and_indices_do_not() {
    let mut rig = lora_rig(RigOptions::default());
    let g1 = rig.driver.capture_adapter_set(set(1)).unwrap();
    assert_eq!(
        rig.driver.capture_adapter_set(set(1)).unwrap(),
        g1,
        "the same set"
    );
    let g2 = rig.driver.capture_adapter_set(set(2)).unwrap();
    assert_ne!(g1, g2, "a different source is a different generation");

    let heads = [Head::GREEDY];
    let caps = rig
        .driver
        .prepare(&shapes(pool(8, 4), &[nz(1), nz(2)], &heads, &[g1, g2]))
        .unwrap()
        .clone();
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
    assert!(last_step_bound(&rig, &[1.0]));
    step_rows(&mut rig, &mut [(second, PREFILL1, plain())]).unwrap();
    assert!(last_step_bound(&rig, &[2.0]), "the index is a bound input");
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
        .open(SeqRequest {
            prompt: &[1],
            max_new: 1,
            adapter: on(g1, 1),
        })
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
    let g1 = rig.driver.capture_adapter_set(set(1)).unwrap();
    let g2 = rig.driver.capture_adapter_set(set(2)).unwrap();
    let heads = [Head::GREEDY];
    rig.driver
        .prepare(&shapes(pool(4, 2), &[nz(1)], &heads, &[g1]))
        .unwrap();
    assert_eq!(rig.driver.retained_bytes(), 64);
    let err = rig
        .driver
        .prepare(&shapes(pool(4, 2), &[nz(1)], &heads, &[g2]))
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
        let mut rig = paged_rig(RigOptions {
            variant: Variant::BadRowAxis,
            ..RigOptions::default()
        });
        let heads = [Head::GREEDY];
        rig.driver
            .prepare(&shapes(pool(4, 1), &[nz(rows)], &heads, &[]))
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
    let mut ok = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    ok.driver
        .prepare(&shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
}

fn prompt_of(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32).map(|i| 1 + (i * 7 + salt) % 10).collect()
}

/// Drive `prompt` to its first committed token and release it as finished, returning `release`'s
/// outcome; the prefix cache then holds (or does not) what the outcome says.
fn finish_prompt(rig: &mut Rig, prompt: &[u32]) -> (usize, PrefixOutcome) {
    finish_prompt_as(rig, prompt, AdapterRef::Base)
}

fn finish_prompt_as(rig: &mut Rig, prompt: &[u32], adapter: AdapterRef) -> (usize, PrefixOutcome) {
    let admission = rig
        .driver
        .open(SeqRequest {
            prompt,
            max_new: 2,
            adapter,
        })
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
        let mut rig = paged_rig(RigOptions {
            variant,
            capacity: 64,
            ..RigOptions::default()
        });
        let rows = [nz(1)];
        let heads = [Head::GREEDY];
        rig.driver
            .prepare(&ServingShapes {
                layout: Layout::Paged(pool(6, 1)),
                rows: &rows,
                heads: &heads,
                warm: Warm::Admitted,
                windows: &[],
                adapters: &[],
            })
            .unwrap();
        rig
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

/// A sequence released as preempted frees its blocks and registers nothing, even on a dense family.
#[test]
fn a_preempted_sequence_registers_no_prefix() {
    let mut rig = paged_rig(RigOptions {
        capacity: 64,
        ..RigOptions::default()
    });
    rig.driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(pool(6, 1)),
            rows: &[nz(1)],
            heads: &[Head::GREEDY],
            warm: Warm::Admitted,
            windows: &[],
            adapters: &[],
        })
        .unwrap();
    let prompt = prompt_of(35, 5);
    let Admission::Opened { seq, .. } = rig
        .driver
        .open(SeqRequest {
            prompt: &prompt,
            max_new: 2,
            adapter: AdapterRef::Base,
        })
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

/// Misuse of the serving surface is a typed refusal, never a panic: nothing prepared, an unknown or
/// repeated sequence, a step over an uncommitted result, a pool that cannot hold the request, a
/// `generate` with every sequence slot taken.
#[test]
fn misuse_is_a_typed_refusal() {
    let mut rig = paged_rig(RigOptions::default());
    let request = |prompt| SeqRequest {
        prompt,
        max_new: 2,
        adapter: AdapterRef::Base,
    };
    let invalid = |err: DriverError| match err {
        DriverError::InvalidRequest(request) => request,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        invalid(rig.driver.open(request(&[1])).unwrap_err()),
        InvalidRequest::NotPrepared
    );
    rig.driver
        .prepare(&shapes(pool(2, 2), &[nz(2)], &[Head::GREEDY], &[]))
        .unwrap();
    // Two blocks hold 32 positions; a 20-token prompt with 20 new tokens needs three.
    let Admission::NoRoom { needed_blocks } = rig
        .driver
        .open(SeqRequest {
            prompt: &[1; 20],
            max_new: 20,
            adapter: AdapterRef::Base,
        })
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
    rig.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
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

fn admit_paged(rig: &mut Rig, pool: PoolShape, rows: &[NonZeroUsize], windows: &[NonZeroUsize]) {
    rig.driver
        .prepare(&ServingShapes {
            layout: Layout::Paged(pool),
            rows,
            heads: &[Head::GREEDY, Head::Logits],
            warm: Warm::Prompts(&[]),
            windows,
            adapters: &[],
        })
        .unwrap();
}

/// A verify window is one greedy suffix entry for the whole step, so it is refused, typed, beside any
/// row the greedy suffix cannot reproduce (a penalised, guided or sampled row): that row would get an
/// unmasked argmax. Beside a greedy row it runs. Mutation: restore the unconditional greedy head for a
/// window step; the row beside the window is no longer refused.
#[test]
fn a_window_beside_a_row_the_greedy_suffix_cannot_pick_is_refused() {
    let mut rig = paged_rig(RigOptions::default());
    admit_paged(&mut rig, pool(4, 2), &[nz(2)], &[nz(2)]);
    let a = open_one(&mut rig, &[3], AdapterRef::Base);
    let b = open_one(&mut rig, &[4], AdapterRef::Base);
    step_rows(
        &mut rig,
        &mut [(a, PREFILL1, plain()), (b, PREFILL1, plain())],
    )
    .unwrap();
    let window = RowWork::Decode {
        ahead: &[7],
        pick: PickPolicy::Greedy,
    };
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
    let mut rig = paged_rig(RigOptions::default());
    let err = rig
        .driver
        .prepare(&ServingShapes {
            windows: &[nz(1)],
            ..shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[])
        })
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidOptions(InvalidOptions::WindowBelowTwo { width: 1 })
        ),
        "{err:?}"
    );
    assert!(rig.driver.caps().is_none());
}

/// A refused `prepare` leaves the driver as it was: a recurrent model asked for two sequences is
/// refused, and `open` still reports nothing prepared instead of admitting up to `max_seqs`; a valid
/// call afterwards succeeds. Mutation: publish the pool before the residency check; `open` is then
/// accepted after the refusal.
#[test]
fn a_refused_prepare_publishes_nothing() {
    let mut rig = paged_rig(RigOptions {
        variant: Variant::Recurrent,
        ..RigOptions::default()
    });
    let err = rig
        .driver
        .prepare(&shapes(pool(4, 2), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidOptions(InvalidOptions::RecurrentServesOneSequence)
        ),
        "{err:?}"
    );
    assert!(rig.driver.caps().is_none());
    let err = rig
        .driver
        .open(SeqRequest {
            prompt: &[1],
            max_new: 1,
            adapter: AdapterRef::Base,
        })
        .unwrap_err();
    assert!(
        matches!(
            err,
            DriverError::InvalidRequest(InvalidRequest::NotPrepared)
        ),
        "{err:?}"
    );
    rig.driver
        .prepare(&shapes(pool(4, 1), &[nz(1)], &[Head::GREEDY], &[]))
        .unwrap();
    open_one(&mut rig, &[1], AdapterRef::Base);
}

/// The pool returns to its size when a sequence is released: a preempted one registers nothing and
/// frees every block; a finished one leaves its prompt blocks pinned in the prefix cache until another
/// request evicts them. Mutation: skip `cache.reset` in the preempted branch; the blocks leak.
#[test]
fn release_returns_every_block_to_the_pool() {
    let mut rig = paged_rig(RigOptions {
        capacity: 64,
        ..RigOptions::default()
    });
    rig.driver
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
        .open(SeqRequest {
            prompt: &prompt_of(40, 9),
            max_new: 24,
            adapter: AdapterRef::Base,
        })
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
    let mut rig = paged_rig(RigOptions::default());
    admit_paged(&mut rig, pool(4, 1), &[nz(1)], &[nz(2)]);
    let seq = open_one(&mut rig, &[3], AdapterRef::Base);
    let mut sampler = plain();
    let step = |rig: &mut Rig, work, sampler: &mut Sampler| {
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
    let picks = step(
        &mut rig,
        RowWork::Decode {
            ahead: &[7],
            pick: PickPolicy::Greedy,
        },
        &mut sampler,
    );
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
    let g = rig.driver.capture_adapter_set(set(1)).unwrap();
    rig.driver
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

/// One decode loop (Card 1016): `generate` is the serving surface driven for one row. The same script
/// through `generate` and through hand-driven `open`/`step`/`commit` reaches the executor with the same
/// steps (phase, tokens, positions) and yields the same tokens, so `generate` holds no pick or feed
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
    manual.driver.prepare(&contiguous(&[Head::GREEDY])).unwrap();
    let Admission::Opened { seq, .. } = manual
        .driver
        .open(SeqRequest {
            prompt: &prompt,
            max_new,
            adapter: AdapterRef::Base,
        })
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
    let shapes = |rig: &Rig| -> Vec<(Option<Phase>, Vec<i32>, Vec<i32>)> {
        rig.world
            .borrow()
            .steps
            .iter()
            .map(|s| (s.phase, s.tokens.clone(), s.positions.clone()))
            .collect()
    };
    assert_eq!(out.tokens, tokens);
    assert_eq!(shapes(&generated), shapes(&manual));
    assert_eq!(generated.driver.stats().steps, manual.driver.stats().steps);
}

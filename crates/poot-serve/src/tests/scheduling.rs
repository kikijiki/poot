use super::*;

use super::recording_engine::peak_open;
use crate::batch::engine::Exit;
use crate::types::{AdminError, AdminJob, AdminReply, AdminRequest};

/// SC-001: the loop clamps one step's prefill work to the token budget. Three rows of four prompt tokens
/// are open at once on a four-wide engine (budget: the widest piece plus one token per row), so a step
/// takes two of them and the third waits for the next. Mutation: build the step's rows from every
/// prefilling row, without `schedule_prefill_admissions`; the first step carries all three and the row
/// goes red.
#[test]
fn the_loop_clamps_prefill_rows_to_the_step_budget() {
    let mut engine = RecordingEngine::new(4).picking(|_, _| vec![1]);
    engine.widest = 4;
    let specs = vec![
        JobSpec::new("aaaa", 1),
        JobSpec::new("bbbb", 1),
        JobSpec::new("cccc", 1),
    ];
    let (exit, _metrics, submitted) = serve(&mut engine, specs);
    assert_eq!(exit, Exit::Shutdown);
    let steps: Vec<usize> = engine
        .events()
        .into_iter()
        .filter_map(|event| match event {
            Event::Step(rows) => Some(rows.len()),
            _ => None,
        })
        .collect();
    assert_eq!(
        steps,
        vec![2, 1],
        "two prefill rows fit the budget of 8 tokens, the third waits"
    );
    for request in &submitted {
        assert!(matches!(outcome(request).1, GenEvent::Done(..)));
    }
}

/// The loop never opens more sequences than the engine has rows, and serves every queued job once the
/// queue has closed (the tests of the deleted loops' drain and admit-into-free-slot behavior).
#[test]
fn the_loop_opens_no_more_sequences_than_rows_and_serves_the_whole_queue() {
    let mut engine = RecordingEngine::new(2).picking(|_, committed| vec![committed as u32 + 1]);
    let specs = (0..5).map(|i| JobSpec::new(&format!("p{i}"), 3)).collect();
    let (exit, metrics, submitted) = serve(&mut engine, specs);
    assert_eq!(
        exit,
        Exit::Shutdown,
        "the closed queue ends the loop once it is served"
    );
    assert_eq!(peak_open(&engine.events()), 2);
    for request in &submitted {
        let (_, terminal) = outcome(request);
        let GenEvent::Done(completion) = terminal else {
            panic!("every queued job is served");
        };
        assert_eq!(completion.prompt_tokens, 2, "the prompt");
        assert_eq!(completion.tokens.len(), 3, "three generated tokens");
        assert_eq!(
            completion.finish,
            FinishReason::Length,
            "max_new ends the request: finish_reason length"
        );
    }
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 5);
}

/// A request that finds no KV room while another runs waits for it: the second `open` comes after the
/// first release, and both complete. (The driver reserves a request's blocks for its whole `max_new` at
/// open, so a running request is never starved and nothing is preempted.)
#[test]
fn a_request_with_no_room_waits_for_a_running_one_to_finish() {
    let mut engine = RecordingEngine::new(2);
    engine.slots = 1;
    let (exit, _metrics, submitted) = serve(
        &mut engine,
        vec![JobSpec::new("first", 2), JobSpec::new("second", 2)],
    );
    assert_eq!(exit, Exit::Shutdown);
    let events = engine.events();
    let second_open = events
        .iter()
        .rposition(|event| matches!(event, Event::Open { .. }))
        .unwrap();
    let first_release = events
        .iter()
        .position(|event| matches!(event, Event::Release { .. }))
        .unwrap();
    assert!(first_release < second_open, "{events:?}");
    assert_eq!(peak_open(&events), 1);
    for request in &submitted {
        assert!(matches!(outcome(request).1, GenEvent::Done(..)));
    }
}

/// A request whose prompt does not fit the engine's window fails alone; the next one is served.
#[test]
fn a_prompt_longer_than_the_window_fails_alone() {
    let mut engine = RecordingEngine::new(2);
    engine.max_seq = 8;
    let (_exit, metrics, submitted) = serve(
        &mut engine,
        vec![
            JobSpec::new("much too long a prompt", 2),
            JobSpec::new("ok", 2),
        ],
    );
    let GenEvent::Failed(failure) = outcome(&submitted[0]).1 else {
        panic!("the long prompt fails");
    };
    assert!(failure.to_string().contains("does not fit"), "{failure}");
    assert!(matches!(outcome(&submitted[1]).1, GenEvent::Done(..)));
    assert_eq!(metrics.requests_errored.load(Ordering::Relaxed), 1);
}

/// `max_new` past the window is cut to it, and the request ends "length", as `max_tokens` past the KV cap
/// always has.
#[test]
fn max_new_is_cut_to_the_positions_the_window_leaves() {
    let mut engine = RecordingEngine::new(1);
    engine.max_seq = 10;
    let (_exit, _metrics, submitted) = serve(&mut engine, vec![JobSpec::new("abcdef", 100)]);
    let GenEvent::Done(completion) = outcome(&submitted[0]).1 else {
        panic!("served");
    };
    assert_eq!(completion.prompt_tokens, 6);
    assert_eq!(
        completion.tokens.len(),
        4,
        "the four positions the window leaves after six prompt tokens"
    );
    assert_eq!(completion.finish, FinishReason::Length);
    assert!(matches!(engine.events()[0], Event::Open { max_new: 4, .. }));
}

/// An idle loop with a live queue stays open for a late job, and ends once the queue closes.
#[test]
fn an_idle_loop_serves_a_late_job_and_ends_when_the_queue_closes() {
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    let metrics = Arc::new(test_metrics("idle"));
    let loop_metrics = Arc::clone(&metrics);
    let thread = std::thread::spawn(move || {
        let mut engine = RecordingEngine::new(1);
        run_engine(&mut engine, &rx, &loop_metrics)
    });
    std::thread::sleep(Duration::from_millis(60));
    assert!(
        !thread.is_finished(),
        "a live empty queue keeps the loop open"
    );
    let (job, submitted) = JobSpec::new("late", 1).build();
    tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    let terminal = submitted
        .replies
        .recv_timeout(Duration::from_secs(5))
        .expect("the late job is served");
    assert!(matches!(terminal, GenEvent::Done(..)));
    drop(tx);
    assert_eq!(thread.join().unwrap(), Exit::Shutdown);
}

/// `schedule_prefill_admissions` is the pure `max_num_batched_tokens` decision behind the loop's prefill rows (deferred rows retry on a later step).
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

/// SC-003 (loop half): the loop counts the draft policy's report for each row as the driver gave it. A
/// verified round that returned three tokens (two accepted drafts and one more) counts two accepted, not
/// three, and a row that decoded without speculating counts as a skip. Mutation: record the row's token
/// count where its accepted count belongs; the accepted counter reads 6 for 4 and the row goes red.
#[test]
fn the_loop_counts_the_drivers_accepted_drafts_not_the_tokens_a_round_returned() {
    use poot_llm::driver::{NotSpeculated, RowSpeculation};

    let mut engine = RecordingEngine::new(1).picking(|_, committed| {
        if committed == 0 {
            vec![1]
        } else {
            vec![5, 6, 7]
        }
    });
    engine.speculation = Some(RowSpeculation::Verified {
        proposed: 4,
        accepted: 2,
    });
    let (exit, metrics, submitted) = serve(&mut engine, vec![JobSpec::new("ab", 7)]);
    assert_eq!(exit, Exit::Shutdown);
    let GenEvent::Done(completion) = outcome(&submitted[0]).1 else {
        panic!("the request completes");
    };
    assert_eq!(completion.tokens.len(), 7, "seven generated tokens");
    let speculative = &metrics.speculative;
    let count = |counter: &std::sync::atomic::AtomicU64| counter.load(Ordering::Relaxed);
    assert_eq!(count(&speculative.rounds), 2);
    assert_eq!(count(&speculative.proposed), 8);
    assert_eq!(count(&speculative.accepted), 4);
    assert_eq!(count(&speculative.skipped), 0);

    let mut engine = RecordingEngine::new(1);
    engine.speculation = Some(RowSpeculation::NotSpeculated(NotSpeculated::NoProposal));
    let (_, metrics, _) = serve(&mut engine, vec![JobSpec::new("ab", 3)]);
    assert_eq!(count(&metrics.speculative.skipped), 2);
    assert_eq!(count(&metrics.speculative.rounds), 0);
}

/// Queue `first`, then `admins` administration jobs, then `second`, close the queue and run the loop on
/// this thread. Returns the replies of the administration jobs.
fn serve_with_admin_jobs(
    engine: &mut RecordingEngine,
    first: JobSpec,
    admins: usize,
    second: Option<JobSpec>,
) -> Vec<mpsc::Receiver<Result<AdminReply, AdminError>>> {
    let (tx, rx) = mpsc::channel();
    let (job, _first) = first.build();
    tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    let mut replies = Vec::new();
    for index in 0..admins {
        let (reply, answer) = mpsc::channel();
        tx.send(EngineMsg::Admin(AdminJob {
            request: AdminRequest::Load {
                name: format!("adapter{index}"),
                path: format!("/adapters/{index}"),
            },
            reply,
        }))
        .unwrap();
        replies.push(answer);
    }
    if let Some(second) = second {
        let (job, _second) = second.build();
        tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    }
    drop(tx);
    let metrics = test_metrics("admin-bound");
    assert_eq!(run_engine(engine, &rx, &metrics), Exit::Shutdown);
    replies
}

/// POOT-1204 SC-005: queued administration jobs do not stall generation. Three jobs queue ahead of a
/// second request; the loop runs at most `EXCLUSIVE_JOBS_PER_ITERATION` of them before the step, so the
/// request's first step is the loop's first step and fewer than all three jobs ran before it; every job
/// still runs. Mutation: run every queued exclusive job before admitting (strict FIFO, the loop of
/// `for _ in 0..usize::MAX`): three jobs run before the request's step and the row goes red.
#[test]
fn queued_admin_jobs_delay_a_generation_request_by_no_more_than_the_iteration_bound() {
    let mut engine = RecordingEngine::new(2);
    let replies = serve_with_admin_jobs(
        &mut engine,
        JobSpec::new("aa", 6),
        3,
        Some(JobSpec::new("bb", 1)),
    );
    let events = engine.events();
    let first_seq = events
        .iter()
        .find_map(|event| match event {
            Event::Step(rows) => Some(rows[0].0),
            _ => None,
        })
        .expect("a step ran");
    let second_step = events
        .iter()
        .position(|event| matches!(event, Event::Step(rows) if rows.iter().any(|(seq, _)| *seq != first_seq)))
        .expect("the second request ran a step");
    let before = &events[..second_step];
    let admins_before = before
        .iter()
        .filter(|event| matches!(event, Event::Admin(_)))
        .count();
    let steps_before = before
        .iter()
        .filter(|event| matches!(event, Event::Step(_)))
        .count();
    assert_eq!(
        steps_before, 0,
        "the request joined the loop's first step: {events:?}"
    );
    assert!(
        admins_before <= 1,
        "{admins_before} administration jobs ran before the request's first step: {events:?}"
    );
    let ran = events
        .iter()
        .filter(|event| matches!(event, Event::Admin(_)))
        .count();
    assert_eq!(ran, 3, "every queued job still runs: {events:?}");
    for reply in replies {
        assert!(reply.try_recv().expect("answered").is_ok());
    }
}

/// POOT-1204 SC-005: an administration job runs between steps while a sequence is open, not after it
/// ends, and no more than the bound run between two steps. Mutation: wait for an empty engine before
/// running them (the old rule): every job runs before the first step and the row goes red.
#[test]
fn admin_jobs_run_between_steps_of_an_open_sequence() {
    let mut engine = RecordingEngine::new(1);
    serve_with_admin_jobs(&mut engine, JobSpec::new("aa", 6), 3, None);
    let events = engine.events();
    let mut open = false;
    let mut run_open = 0;
    let mut since_step = 0;
    for event in &events {
        match event {
            Event::Open { .. } => open = true,
            Event::Release { .. } => open = false,
            Event::Step(_) => since_step = 0,
            Event::Admin(_) => {
                since_step += 1;
                assert!(
                    since_step <= 1,
                    "{since_step} jobs ran between two steps: {events:?}"
                );
                run_open += usize::from(open);
            }
            _ => {}
        }
    }
    assert!(
        run_open >= 2,
        "jobs ran while the sequence was open: {events:?}"
    );
}

/// POOT-1204 review: encode and administration jobs alternate when both wait, so a steady encode stream
/// cannot starve an administration job. Three of each queue together; the log never runs two of one
/// class back to back while the other class waits. Mutation: prefer encodes (take an encode whenever one
/// is ready): the three encodes run before the first administration job and the row goes red.
#[test]
fn encode_and_admin_jobs_alternate_when_both_wait() {
    let mut engine = RecordingEngine::new(2);
    let (tx, rx) = mpsc::channel();
    let mut answers = Vec::new();
    // The guards cancel their jobs on drop: they live until the loop has run.
    let mut guards = Vec::new();
    for index in 0..3 {
        let (reply, answer) = mpsc::channel();
        let (guard, cancel) = crate::lifecycle::JobGuard::new();
        tx.send(EngineMsg::Encode(crate::types::EncodeJob {
            rows: vec![crate::types::EncodeRow {
                ids: vec![index + 1],
                types: Vec::new(),
            }],
            reply,
            cancel,
        }))
        .unwrap();
        answers.push(answer);
        guards.push(guard);
    }
    let mut admins = Vec::new();
    for index in 0..3 {
        let (reply, answer) = mpsc::channel();
        tx.send(EngineMsg::Admin(AdminJob {
            request: AdminRequest::Load {
                name: format!("adapter{index}"),
                path: format!("/adapters/{index}"),
            },
            reply,
        }))
        .unwrap();
        admins.push(answer);
    }
    drop(tx);
    let metrics = test_metrics("alternate");
    assert_eq!(run_engine(&mut engine, &rx, &metrics), Exit::Shutdown);
    let classes: Vec<bool> = engine
        .events()
        .iter()
        .filter_map(|event| match event {
            Event::Encode { .. } => Some(true),
            Event::Admin(_) => Some(false),
            _ => None,
        })
        .collect();
    assert_eq!(classes.len(), 6, "every job ran: {classes:?}");
    assert!(
        classes.windows(2).all(|pair| pair[0] != pair[1]),
        "the classes alternate while both wait: {classes:?}"
    );
    for answer in admins {
        assert!(answer.try_recv().expect("answered").is_ok());
    }
}

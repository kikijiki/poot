//! The loop's release, failure and lifecycle rows over the recording fake: cancellation, the prefix a
//! release registers, the engine thread's death, and the faults a step can end in.

use std::net::IpAddr;

use poot_llm::driver::{PrefixOutcome, Release};

use super::*;
use crate::engine_plan::{Death, EngineState};
use crate::lifecycle::JobGuard;
use crate::types::EncodeRow;

/// SC-005 (R477-011): a non-streaming request whose client disconnects is cancelled and its slot released
/// within one step. The request never ends on its own and streams nothing, so only the hang-up probe can
/// tell the engine; the `JobGuard` that drops as `submit` returns sets the job's cancel flag, and the
/// release follows the next step boundary. Mutation: ignore the hang-up for non-streaming requests (drop
/// the `client_gone` poll in `submit`); the request keeps running, `submit` never returns and the row goes
/// red on its deadline.
#[test]
fn a_non_streaming_request_whose_client_hangs_up_is_released_within_one_step() {
    let mut engine = RecordingEngine::new(1).picking(|_, _| {
        std::thread::sleep(Duration::from_millis(1));
        vec![7]
    });
    engine.max_seq = usize::MAX / 2;
    let log = Arc::clone(&engine.log);
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    let metrics = Arc::new(test_metrics("submit-cancel"));
    let health = EngineHealth::serving_for_test("fake", 1, 64);
    let thread = {
        let (metrics, health) = (Arc::clone(&metrics), health.clone());
        std::thread::spawn(move || run_engine_thread(engine, rx, &metrics, &health))
    };
    let started = Instant::now();
    let (result_tx, result_rx) = mpsc::channel();
    let submit_tx = tx.clone();
    let submit_metrics = Arc::clone(&metrics);
    std::thread::spawn(move || {
        let outcome = submit_one(
            &submit_tx,
            &submit_metrics,
            generation_request("ab", 1_000_000, false),
            &|| started.elapsed() > Duration::from_millis(150),
            |_, _| GenerationControl::Continue(()),
        )
        .map(drop);
        let _ = result_tx.send(outcome);
    });
    let result = result_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("submit never returned: the hang-up was ignored and the request kept running");
    assert_eq!(
        result.expect_err("a hung-up client cancels"),
        GenerationError::ClientCancelled
    );
    let deadline = Instant::now() + Duration::from_secs(5);
    let release = loop {
        let events = log.lock().unwrap().clone();
        if let Some(index) = events
            .iter()
            .position(|event| matches!(event, Event::Release { .. }))
        {
            break (index, events);
        }
        assert!(
            Instant::now() < deadline,
            "the slot was never released after the client left"
        );
        std::thread::sleep(Duration::from_millis(2));
    };
    let (index, events) = release;
    assert!(
        matches!(
            events[index],
            Event::Release {
                mode: Release::Finished,
                ..
            }
        ),
        "{:?}",
        events[index]
    );
    drop(tx);
    thread.join().expect("the loop ends once the queue closes");
    let events = log.lock().unwrap().clone();
    let steps_after = events[index..]
        .iter()
        .filter(|event| matches!(event, Event::Step(_)))
        .count();
    assert_eq!(steps_after, 0, "no step runs after the release");
}

/// The cancel flag in isolation: set during step 3, the request is released before step 4, and nothing
/// runs for it after. Also the ordering rule for a cancel during a step: the release comes after that
/// step's commit, never in the middle of it, so the engine still owns what the step used.
#[test]
fn a_cancel_during_a_step_releases_after_that_step_commits_and_before_the_next() {
    let (job, submitted) = JobSpec::new("ab", 1000).build();
    let cancel = Arc::clone(&submitted.cancel);
    let mut engine = RecordingEngine::new(1)
        .picking(|_, _| vec![7])
        .on_step(move |step, _| {
            if step == 3 {
                cancel.store(true, Ordering::SeqCst);
            }
            Ok(())
        });
    let (tx, rx) = mpsc::channel();
    tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    drop(tx);
    let metrics = test_metrics("cancel-order");
    assert_eq!(run_engine(&mut engine, &rx, &metrics), Exit::Shutdown);
    let events = engine.events();
    let third_step = events
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, Event::Step(_)))
        .nth(2)
        .map(|(index, _)| index)
        .unwrap();
    assert!(
        matches!(events[third_step + 1], Event::Commit { .. }),
        "the cancelled step's picks are committed first: {events:?}"
    );
    assert!(
        matches!(events[third_step + 2], Event::Release { .. }),
        "then the sequence is released: {events:?}"
    );
    assert_eq!(
        events.len(),
        third_step + 3,
        "nothing runs after: {events:?}"
    );
    assert_eq!(metrics.requests_cancelled.load(Ordering::Relaxed), 1);
    drop(submitted);
}

/// A `JobGuard` sets the cancel flag however its owner stops waiting, and not before.
#[test]
fn a_job_guard_sets_the_cancel_flag_when_dropped() {
    let (guard, cancel) = JobGuard::new();
    assert!(!cancel.load(Ordering::SeqCst));
    drop(guard);
    assert!(cancel.load(Ordering::SeqCst));
}

fn two_requests_over_one_prompt(recurrent: bool) -> (RecordingEngine, Arc<Metrics>) {
    let mut engine = RecordingEngine::new(1);
    engine.recurrent = recurrent;
    let (_exit, metrics, submitted) = serve(
        &mut engine,
        vec![JobSpec::new("abcdefghi", 1), JobSpec::new("abcdefghi", 1)],
    );
    let tokens: Vec<Vec<u32>> = submitted
        .iter()
        .map(|request| match outcome(request).1 {
            GenEvent::Done(completion) => completion.tokens,
            GenEvent::Failed(failure) => panic!("the request failed: {failure}"),
            GenEvent::Token(..) => unreachable!("outcome stops at the terminal"),
        })
        .collect();
    assert_eq!(
        tokens[0], tokens[1],
        "the second request returns the cold run's tokens"
    );
    (engine, metrics)
}

/// SC-007: the loop releases a finished sequence through the engine and routes the outcome. On a
/// recurrent-state fixture the release is `Skipped(RecurrentState)`, nothing registers, and the second
/// request over the same prompt opens with nothing reused and runs a full prefill; on a dense fixture the
/// release registers the prompt's whole blocks and the second request reuses them. Mutation: skip the
/// `ServeEngine::release` call on finish; the sequence stays open on a one-slot engine, the second
/// request finds no room, fails, and the row goes red. (The guard itself is `Driver::release`'s, proven
/// in `poot-llm`; this row is the loop calling it, so the hybrid evidence is fake-level: no served hybrid
/// fixture exists.)
#[test]
fn a_finished_request_is_released_and_a_recurrent_model_registers_no_prefix() {
    let (dense, metrics) = two_requests_over_one_prompt(false);
    let reused: Vec<usize> = dense
        .events()
        .iter()
        .filter_map(|event| match event {
            Event::Open { reused, .. } => Some(*reused),
            _ => None,
        })
        .collect();
    assert_eq!(
        reused,
        vec![0, 8],
        "the dense prefix is registered and reused"
    );
    assert_eq!(
        metrics
            .cache
            .prefix_registered_tokens
            .load(Ordering::Relaxed),
        16
    );
    assert_eq!(metrics.cache.prefix_hit_tokens.load(Ordering::Relaxed), 8);

    let (hybrid, metrics) = two_requests_over_one_prompt(true);
    let reused: Vec<usize> = hybrid
        .events()
        .iter()
        .filter_map(|event| match event {
            Event::Open { reused, .. } => Some(*reused),
            _ => None,
        })
        .collect();
    assert_eq!(reused, vec![0, 0], "a recurrent model reuses nothing");
    assert_eq!(
        metrics
            .cache
            .prefix_registered_tokens
            .load(Ordering::Relaxed),
        0
    );
    assert_eq!(
        metrics
            .cache
            .prefix_skipped_releases
            .load(Ordering::Relaxed),
        2
    );
}

/// A real `GET /health` through the production `handle`: status line and JSON body.
pub(super) fn health_over_http(health: &EngineHealth) -> (String, serde_json::Value) {
    let model_path = write_tiny_gguf();
    let model = load_handle(&model_path);
    let _ = std::fs::remove_file(&model_path);
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(20)))
        .expect("bound the read");
    let (server, _) = listener.accept().expect("accept");
    client
        .write_all(b"GET /health HTTP/1.1\r\n\r\n")
        .expect("write request");
    let (served, _queue) = served_over(model, 1);
    let metrics = test_metrics("health");
    let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
        .expect("loopback policy");
    handle(
        server,
        Instant::now(),
        &served,
        health,
        &metrics,
        &test_limits(),
        &policy,
    )
    .expect("handled");
    let mut response = String::new();
    client.read_to_string(&mut response).expect("read response");
    let status = response.lines().next().unwrap_or_default().to_string();
    let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
    (status, serde_json::from_str(body).expect("json body"))
}

/// Wait until `health` leaves `Serving`, which the engine thread's last act does.
fn wait_for_death(health: &EngineHealth) -> EngineState {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let state = health.state();
        if !matches!(state, EngineState::Serving { .. }) || Instant::now() > deadline {
            return state;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// SC-009: the fake's `step` panics mid-request. The engine thread's guard publishes `Dead(Panicked)`
/// with the panic's message, so `/health` is 503 `dead` with that reason within one poll, and the
/// in-flight request gets a typed error (its reply channel closes) instead of hanging. Mutation: drop the
/// thread's drop guard (forget it); `/health` stays `serving` and the row goes red.
#[test]
fn a_panicking_step_makes_health_dead_with_the_reason_and_fails_the_request() {
    let engine = RecordingEngine::new(1).on_step(|step, _| {
        if step == 2 {
            panic!("fake step blew up");
        }
        Ok(())
    });
    let health = EngineHealth::serving_for_test("fake", 1, 64);
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    let metrics = Arc::new(test_metrics("panic"));
    let thread = {
        let (metrics, health) = (Arc::clone(&metrics), health.clone());
        std::thread::spawn(move || run_engine_thread(engine, rx, &metrics, &health))
    };
    let error = submit_one(
        &tx,
        &metrics,
        generation_request("ab", 10, false),
        &|| false,
        |_, _| GenerationControl::Continue(()),
    )
    .expect_err("the request cannot finish");
    assert_eq!(error, GenerationError::ReplyChannelClosed);
    let state = wait_for_death(&health);
    let EngineState::Dead(Death::Panicked(reason)) = &state else {
        panic!("the engine must be dead after a panic: {state:?}");
    };
    assert!(reason.contains("fake step blew up"), "{reason}");
    let (status, body) = health_over_http(&health);
    assert_eq!(status, "HTTP/1.1 503 Service Unavailable", "{body}");
    assert_eq!(body["status"], "dead", "{body}");
    assert_eq!(body["cause"], "panicked", "{body}");
    assert!(
        body["reason"]
            .as_str()
            .unwrap()
            .contains("fake step blew up"),
        "{body}"
    );
    assert_eq!(body["engine_loaded"], false, "{body}");
    drop(tx);
    thread.join().expect("the thread returns its engine");
}

/// A fault ends the loop and the engine is dead with the fault's reason; every request, active and
/// queued, gets the typed failure with that reason (a 500 the producer counted once).
#[test]
fn a_faulting_step_fails_every_request_and_marks_the_engine_dead_with_the_fault() {
    let engine = RecordingEngine::new(1).on_step(|step, _| {
        if step == 2 {
            return Err(EngineError::Fault("device lost".into()));
        }
        Ok(())
    });
    let health = EngineHealth::serving_for_test("fake", 1, 64);
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    let metrics = Arc::new(test_metrics("fault"));
    let mut requests = Vec::new();
    for prompt in ["ab", "cd"] {
        let (job, submitted) = JobSpec::new(prompt, 10).build();
        tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
        requests.push(submitted);
    }
    drop(tx);
    let engine = run_engine_thread(engine, rx, &metrics, &health);
    for request in &requests {
        let GenEvent::Failed(failure) = outcome(request).1 else {
            panic!("every request fails");
        };
        assert!(failure.to_string().contains("device lost"), "{failure}");
    }
    assert_eq!(metrics.requests_errored.load(Ordering::Relaxed), 2);
    assert_eq!(
        health.state(),
        EngineState::Dead(Death::Fault("step: device lost".into()))
    );
    let (_, body) = health_over_http(&health);
    assert_eq!(body["status"], "dead", "{body}");
    assert_eq!(body["cause"], "fault", "{body}");
    drop(engine);
}

/// SC-010, preflight refusal: a step the engine refuses before running fails the requests of that step
/// alone, releases them as failed (no prefix registered) and keeps serving: a request that arrives next
/// completes.
#[test]
fn a_refused_step_fails_its_requests_and_the_engine_keeps_serving() {
    let mut engine = RecordingEngine::new(2).on_step(|step, _| {
        if step == 1 {
            return Err(refused("no prepared entry for 2 rows"));
        }
        Ok(())
    });
    engine.slots = 2;
    // A second wave: the refused step's rows fail, then a fresh pair is served.
    let (tx, rx) = mpsc::channel();
    let mut first_wave = Vec::new();
    for prompt in ["ab", "cd"] {
        let (job, submitted) = JobSpec::new(prompt, 2).build();
        tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
        first_wave.push(submitted);
    }
    let metrics = test_metrics("refused");
    let thread_rx = rx;
    let tx2 = tx.clone();
    let sender = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(80));
        let (job, submitted) = JobSpec::new("ef", 2).build();
        tx2.send(EngineMsg::Generate(Box::new(job))).unwrap();
        submitted
    });
    drop(tx);
    let exit = run_engine(&mut engine, &thread_rx, &metrics);
    let late = sender.join().unwrap();
    assert_eq!(exit, Exit::Shutdown);
    for request in &first_wave {
        let GenEvent::Failed(failure) = outcome(request).1 else {
            panic!("the refused step's requests fail");
        };
        assert!(
            failure.to_string().contains("no prepared entry"),
            "{failure}"
        );
    }
    assert!(
        matches!(outcome(&late).1, GenEvent::Done(..)),
        "the engine kept serving"
    );
    let released_failed = engine
        .events()
        .iter()
        .filter(|event| {
            matches!(
                event,
                Event::Release {
                    mode: Release::Preempted,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        released_failed, 2,
        "failed requests release without registering a prefix"
    );
}

/// SC-010, partial state mutation and poisoned-device completion: a faulting step leaves the device
/// poisoned. The loop commits nothing from it and releases nothing afterwards (a release would hand
/// blocks back that the device may still own, and in this fake clears the poison), and the engine is
/// returned to its owner still poisoned. Mutation: release the open sequences on a fault (clear
/// poison on release); the log shows a `Release` and the poison is gone, so the row goes red.
#[test]
fn a_fault_commits_nothing_releases_nothing_and_leaves_the_poison_in_place() {
    let engine = RecordingEngine::new(2).on_step(|step, _| {
        if step == 2 {
            return Err(EngineError::Fault("the device is poisoned".into()));
        }
        Ok(())
    });
    let health = EngineHealth::serving_for_test("fake", 2, 64);
    let (tx, rx) = mpsc::channel();
    let mut requests = Vec::new();
    for prompt in ["ab", "cd"] {
        let (job, submitted) = JobSpec::new(prompt, 10).build();
        tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
        requests.push(submitted);
    }
    drop(tx);
    let metrics = test_metrics("poisoned");
    let engine = run_engine_thread(engine, rx, &Arc::new(metrics), &health);
    let events = engine.events();
    let fault_at = events
        .iter()
        .enumerate()
        .filter(|(_, event)| matches!(event, Event::Step(_)))
        .nth(1)
        .map(|(index, _)| index)
        .unwrap();
    assert_eq!(
        events.len(),
        fault_at + 1,
        "nothing is committed or released after the faulting step: {events:?}"
    );
    assert!(engine.poisoned, "the poison stays until the process ends");
    for request in &requests {
        assert!(matches!(outcome(request).1, GenEvent::Failed(_)));
    }
}

/// SC-010, an invalid completion: a step whose results do not match its rows is a fault, and none of its
/// outputs is committed. Mutation: commit the mismatched results; the log shows commits and the row goes
/// red.
#[test]
fn a_step_whose_results_do_not_match_its_rows_is_a_fault_and_commits_nothing() {
    struct Mismatch(RecordingEngine);
    impl ServeEngine for Mismatch {
        type Seq = FakeSeq;
        fn rows(&self) -> usize {
            self.0.rows()
        }
        fn max_seq(&self) -> usize {
            self.0.max_seq()
        }
        fn widest_piece(&self) -> usize {
            self.0.widest_piece()
        }
        fn eos(&self) -> &std::collections::BTreeSet<u32> {
            self.0.eos()
        }
        fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
            self.0.decode(tokens)
        }
        fn piece(&self, tokens: &[u32], gen_start: usize) -> anyhow::Result<String> {
            self.0.piece(tokens, gen_start)
        }
        fn open(&mut self, request: OpenRequest<'_>) -> Result<Opening<FakeSeq>, EngineError> {
            self.0.open(request)
        }
        fn step(
            &mut self,
            rows: &mut [Row<'_, FakeSeq>],
        ) -> Result<Vec<RowResult<FakeSeq>>, EngineError> {
            let mut results = self.0.step(rows)?;
            // The device answers for a sequence the step never ran.
            results[0].seq = FakeSeq(results[0].seq.0 + 100);
            Ok(results)
        }
        fn commit(
            &mut self,
            seq: FakeSeq,
            kept: usize,
            sampler: &mut Sampler,
        ) -> Result<(), EngineError> {
            self.0.commit(seq, kept, sampler)
        }
        fn release(&mut self, seq: FakeSeq, mode: Release) -> Result<PrefixOutcome, EngineError> {
            self.0.release(seq, mode)
        }
        fn encode(&mut self, rows: &[EncodeRow]) -> Result<poot_llm::driver::Encoded, EngineError> {
            self.0.encode(rows)
        }
        fn admin(
            &mut self,
            request: crate::types::AdminRequest,
        ) -> Result<crate::types::AdminReply, crate::types::AdminError> {
            self.0.admin(request)
        }
        fn kv_blocks(&self) -> Option<(usize, usize)> {
            self.0.kv_blocks()
        }
        fn resident_bytes(&self) -> u64 {
            self.0.resident_bytes()
        }
        fn compiles(&self) -> u64 {
            self.0.compiles()
        }
    }
    let mut engine = Mismatch(RecordingEngine::new(1));
    let (job, submitted) = JobSpec::new("ab", 3).build();
    let (tx, rx) = mpsc::channel();
    tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    drop(tx);
    let metrics = test_metrics("mismatch");
    let exit = run_engine(&mut engine, &rx, &metrics);
    assert!(
        matches!(&exit, Exit::Fault(reason) if reason.contains("do not match its rows")),
        "{exit:?}"
    );
    assert!(
        !engine
            .0
            .events()
            .iter()
            .any(|event| matches!(event, Event::Commit { .. })),
        "a mismatched step commits nothing: {:?}",
        engine.0.events()
    );
    assert!(matches!(outcome(&submitted).1, GenEvent::Failed(_)));
}

/// SC-003 on the fake: a streamed and a non-streamed request both complete, each with a terminal event.
/// Mutation: drop the terminal event of a non-streamed request (skip `deliver_done` unless the request
/// streams); the non-streamed request gets no terminal and the row goes red.
#[test]
fn streamed_and_non_streamed_requests_both_end_with_a_terminal_event() {
    let mut engine = RecordingEngine::new(2).picking(|_, committed| vec![committed as u32 + 1]);
    let (_exit, metrics, submitted) = serve(
        &mut engine,
        vec![JobSpec::new("ab", 2).streaming(), JobSpec::new("cd", 2)],
    );
    let (pieces, terminal) = outcome(&submitted[0]);
    assert_eq!(pieces.len(), 2);
    assert!(matches!(terminal, GenEvent::Done(..)));
    let (pieces, terminal) = outcome(&submitted[1]);
    assert!(pieces.is_empty(), "a non-streamed request streams nothing");
    assert!(matches!(terminal, GenEvent::Done(..)));
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 2);
}

/// A release that fails after the request finished: the client still hears the fault (a typed failure
/// naming the release), not a closed reply channel, and the loop stops with that fault. Mutation: return
/// the release error before sending the failure; the reply channel closes with no event and the row goes
/// red.
#[test]
fn a_failed_release_tells_the_client_the_fault_before_the_loop_stops() {
    let mut engine = RecordingEngine::new(1);
    engine.release_error = Some("device lost in release".into());
    let (exit, _metrics, submitted) = serve(&mut engine, vec![JobSpec::new("ab", 1)]);
    assert!(
        matches!(&exit, Exit::Fault(reason) if reason.contains("device lost in release")),
        "{exit:?}"
    );
    let GenEvent::Failed(failure) = outcome(&submitted[0]).1 else {
        panic!("the client must receive the failure");
    };
    assert!(
        failure.to_string().contains("device lost in release"),
        "{failure}"
    );
}

/// POOT-1084: an `Encode` step whose device error left the engine intact (`FailureClass::Step`) fails that
/// request alone. A generation queued before it and one queued after both complete, the encode's caller gets
/// the typed failure in its class and `/health` stays `serving`. Mutation: treat every encode error as an
/// engine fault (`Err(Fault)` for `Refused` in `Run::encode`): the later generation fails, the engine is
/// `dead`, and the row goes red.
#[test]
fn an_encode_step_failing_with_the_engine_intact_fails_only_its_request() {
    let engine = RecordingEngine::new(1);
    let mut engine = engine;
    engine.on_encode = Box::new(|_| {
        Err(EngineError::Refused {
            class: poot_llm::driver::error::FailureClass::Step,
            reason: "encode step failed, executable reset".into(),
        })
    });
    let health = EngineHealth::serving_for_test("fake", 1, 64);
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    let metrics = Arc::new(test_metrics("encode-step"));
    let (before, before_events) = JobSpec::new("ab", 2).build();
    tx.send(EngineMsg::Generate(Box::new(before))).unwrap();
    let (reply, encode_reply) = mpsc::channel();
    tx.send(EngineMsg::Encode(crate::types::EncodeJob {
        rows: vec![EncodeRow {
            ids: vec![1, 2],
            types: Vec::new(),
        }],
        reply,
        cancel: Arc::default(),
    }))
    .unwrap();
    let (after, after_events) = JobSpec::new("cd", 2).build();
    tx.send(EngineMsg::Generate(Box::new(after))).unwrap();
    drop(tx);
    let engine = run_engine_thread(engine, rx, &metrics, &health);

    let failure = encode_reply
        .recv()
        .expect("the encode caller is answered")
        .expect_err("the encode step failed");
    assert_eq!(
        failure.class(),
        poot_llm::driver::error::FailureClass::Step,
        "{failure}"
    );
    for request in [&before_events, &after_events] {
        assert!(
            matches!(outcome(request).1, GenEvent::Done(..)),
            "a generation beside the failed encode completes"
        );
    }
    assert!(
        matches!(health.state(), EngineState::Serving { .. }),
        "{:?}",
        health.state()
    );
    let (status, body) = health_over_http(&health);
    assert!(status.contains("200"), "{status} {body}");
    assert_eq!(body["status"], "serving", "{body}");
    drop(engine);
}

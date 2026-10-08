use super::*;

use super::recording_engine::peak_open;
use crate::batch::engine::Exit;

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
        let GenEvent::Done(tokens, _, natural) = terminal else {
            panic!("every queued job is served");
        };
        assert_eq!(
            tokens.len(),
            2 + 3,
            "the prompt, then three generated tokens"
        );
        assert!(!natural, "max_new ends the request: finish_reason length");
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
    let GenEvent::Done(tokens, _, natural) = outcome(&submitted[0]).1 else {
        panic!("served");
    };
    assert_eq!(
        tokens.len(),
        10,
        "six prompt tokens and the four the window leaves"
    );
    assert!(!natural);
    assert!(matches!(engine.events()[0], Event::Open { max_new: 4, .. }));
}

/// An idle loop with a live queue stays open for a late job, and ends once the queue closes.
#[test]
fn an_idle_loop_serves_a_late_job_and_ends_when_the_queue_closes() {
    let (tx, rx) = mpsc::channel::<Job>();
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
    tx.send(job).unwrap();
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

//! Publish-before-terminal ordering at the terminal call sites. Each case drives production code to a
//! terminal outcome and reads `terminal_delivery_probe`'s delivery-time snapshot - the counters a
//! consumer would see if it scraped `/metrics` the instant the event was sent. A path that bypasses
//! `deliver_terminal` records no snapshot, and the reset before each case stops the previous case's
//! snapshot from standing in.

use super::*;
use crate::types::deliver_terminal;

/// Which terminal branch of a commit function to drive: EOS ends generation before the token is
/// pushed, `max_new` ends it after.
#[derive(Clone, Copy, Debug)]
enum TerminalBranch {
    Eos,
    MaxNew,
}

/// An admitted slot for a `stream: false` job, plus the reply receiver the test keeps.
fn terminal_slot(max_new: usize) -> (Slot, Receiver<GenEvent>) {
    let (reply, reply_rx) = mpsc::channel();
    let job = Job {
        prompt: "terminal-done".into(),
        max_new,
        sampler: Sampler::greedy(),
        stop: vec![],
        stream: false,
        reply,
        submitted: Instant::now(),
        lora_adapter: LoraAdapterLease::none(),
        mrope_positions: None,
    };
    (
        admit_job_to_slot(job, vec![1], 0, max_new, true, None),
        reply_rx,
    )
}

/// Drive one production terminal branch once and require the delivery-time snapshot to show the
/// outcome already published. `commit` runs that path's own commit function, which must evict the
/// slot (a terminal commit returns `false`); `eos` is the token that branch treats as EOS.
fn assert_publishes_before_delivery(
    label: &str,
    branch: TerminalBranch,
    receiver_alive: bool,
    eos: u32,
    commit: impl FnOnce(&Metrics, &mut PagedKvCache, &mut [Option<Slot>], u32) -> bool,
) {
    let me = std::thread::current().id();
    terminal_delivery_probe::reset(me);
    let metrics = test_metrics(label);
    let max_new = match branch {
        // Only the EOS token ends the slot; `max_new` stays out of the way.
        TerminalBranch::Eos => 4,
        // The first committed token hits the cap.
        TerminalBranch::MaxNew => 1,
    };
    let (slot, reply_rx) = terminal_slot(max_new);
    let reply_rx = receiver_alive.then_some(reply_rx);
    let mut slots = vec![Some(slot)];
    let mut paged = PagedKvCache::new(1, 1);
    paged
        .append(0, 1)
        .expect("fixture block for the admitted slot");
    let next = match branch {
        TerminalBranch::Eos => eos,
        TerminalBranch::MaxNew => 1,
    };
    assert!(
        !commit(&metrics, &mut paged, &mut slots, next),
        "{label}: a terminal commit must evict the slot"
    );
    assert!(
        slots[0].is_none(),
        "{label}: a terminal commit must take the slot"
    );
    let at_delivery = terminal_delivery_probe::at_delivery(me).unwrap_or_else(|| {
        panic!(
            "{label}: the terminal must be delivered through GenEvent::deliver_done, which is \
             what publishes the outcome count before the event exists"
        )
    });
    let expected = if receiver_alive { (1, 0) } else { (0, 1) };
    assert_eq!(
        (at_delivery.completed, at_delivery.cancelled),
        expected,
        "{label}: /metrics at the instant Done is sent must already carry the outcome"
    );
    if let Some(rx) = reply_rx {
        assert!(
            matches!(rx.recv().expect(label), GenEvent::Done(..)),
            "{label}: the receiver must get the event the count was published for"
        );
    }
    assert_eq!(
        metrics.requests_completed.load(Ordering::Relaxed),
        expected.0,
        "{label}"
    );
    assert_eq!(
        metrics.requests_cancelled.load(Ordering::Relaxed),
        expected.1,
        "{label}"
    );
}

/// The commit path: both of `commit_token`'s terminal sends (EOS and stop/`max_new`), each delivered to
/// a live receiver and undelivered to a dropped one.
#[test]
fn commit_token_publishes_the_terminal_count_before_done_is_delivered() {
    let model_path = write_tiny_granitemoe_gguf();
    let runner = Runner::load_gguf(&model_path).expect("load terminal fixture runner");
    let _ = std::fs::remove_file(&model_path);
    for branch in [TerminalBranch::Eos, TerminalBranch::MaxNew] {
        for receiver_alive in [true, false] {
            let label = format!("commit_token {branch:?} receiver_alive={receiver_alive}");
            assert_publishes_before_delivery(
                &label,
                branch,
                receiver_alive,
                runner.eos(),
                |metrics, paged, slots, next| {
                    commit_token(&runner, metrics, paged, 1, slots, 0, next)
                },
            );
        }
    }
}

/// Drive both outcomes of [`deliver_terminal`] - the one delivery every terminal `Done` in the server
/// goes through - and require the delivery-time snapshot to show the count already published: a live
/// receiver is a completion, a dropped one a cancellation.
#[test]
fn deliver_terminal_publishes_the_outcome_before_the_event_is_delivered() {
    for (label, receiver_alive, expected) in [
        ("Done delivered", true, (1u64, 0u64, 0u64)),
        ("Done undelivered", false, (0, 1, 0)),
    ] {
        let me = std::thread::current().id();
        terminal_delivery_probe::reset(me);
        let metrics = test_metrics(label);
        let (reply, rx) = mpsc::channel::<&'static str>();
        let rx = receiver_alive.then_some(rx);
        let delivered = deliver_terminal(&metrics, &reply, "terminal");
        let at_delivery = terminal_delivery_probe::at_delivery(me).unwrap_or_else(|| {
            panic!(
                "{label}: the terminal must go through deliver_terminal, which publishes the count \
                 before the event exists"
            )
        });
        assert_eq!(
            (
                at_delivery.completed,
                at_delivery.cancelled,
                at_delivery.errored
            ),
            expected,
            "{label}: /metrics at the instant the event is sent must already carry the outcome"
        );
        assert_eq!(delivered, receiver_alive, "{label}: delivery outcome");
        if let Some(rx) = rx {
            assert_eq!(
                rx.try_iter().count(),
                1,
                "{label}: exactly one terminal event"
            );
        }
        assert_eq!(
            (
                metrics.requests_completed.load(Ordering::Relaxed),
                metrics.requests_cancelled.load(Ordering::Relaxed),
                metrics.requests_errored.load(Ordering::Relaxed),
            ),
            expected,
            "{label}: final counters"
        );
    }
}

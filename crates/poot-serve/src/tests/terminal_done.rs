//! Publish-before-terminal ordering at the terminal call sites. Each case drives production code to a
//! terminal outcome and reads `terminal_delivery_probe`'s delivery-time snapshot - the counters a
//! consumer would see if it scraped `/metrics` the instant the event was sent. A path that bypasses
//! `deliver_terminal` records no snapshot, and the reset before each case stops the previous case's
//! snapshot from standing in.

use super::*;
use crate::types::deliver_terminal;

/// Which terminal branch of the loop to drive: EOS ends generation before the token is pushed, `max_new`
/// ends it after.
#[derive(Clone, Copy, Debug)]
enum TerminalBranch {
    Eos,
    MaxNew,
}

/// The loop's terminal `Done`, through its release and `GenEvent::deliver_done`, for both branches and for a
/// live and a dropped receiver. A dropped receiver is a cancellation, a live one a completion, and the
/// count is published before the event exists. Mutation: send the `Done` with a bare `reply.send`; no
/// snapshot is recorded and the row goes red.
#[test]
fn the_loops_terminal_done_publishes_its_count_before_the_event_is_delivered() {
    for branch in [TerminalBranch::Eos, TerminalBranch::MaxNew] {
        for receiver_alive in [true, false] {
            let label = format!("loop {branch:?} receiver_alive={receiver_alive}");
            let me = std::thread::current().id();
            terminal_delivery_probe::reset(me);
            let mut engine = RecordingEngine::new(1).picking(move |_, _| match branch {
                TerminalBranch::Eos => vec![99],
                TerminalBranch::MaxNew => vec![1],
            });
            let (job, submitted) = JobSpec::new("a", 1).build();
            let Submitted { replies, .. } = submitted;
            let replies = receiver_alive.then_some(replies);
            let (tx, rx) = mpsc::channel();
            tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
            drop(tx);
            let metrics = test_metrics(&label);
            run_engine(&mut engine, &rx, &metrics);
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
            if let Some(replies) = replies {
                assert!(
                    matches!(replies.try_recv().expect(&label), GenEvent::Done(..)),
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

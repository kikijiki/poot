//! vLLM's `ignore_eos` (POOT-1334): a request that generates to its token cap whatever the model samples.
//! The rows drive the production handlers over the production loop with a scripted engine whose model
//! samples its end-of-sequence token (99) as its third token, so a request that does not ignore it ends
//! after two kept tokens.

use super::*;
use crate::api::RefusedRequest;
use crate::handlers::{handle_chat, handle_completion};

/// The scripted model: token 7 twice, then its end-of-sequence token 99, then 7 again.
fn eos_on_the_third() -> RecordingEngine {
    RecordingEngine::new(1).picking(|_, committed| vec![if committed == 2 { 99 } else { 7 }])
}

fn completion(body: serde_json::Value) -> CompletionReq {
    serde_json::from_value(body).expect("a completion request")
}

fn chat(body: serde_json::Value) -> ChatReq {
    serde_json::from_value(body).expect("a chat request")
}

/// What a non-streaming reply reports: its `usage.completion_tokens` and the first choice's finish reason.
fn reported(reply: &str) -> (u64, String) {
    let reply: serde_json::Value = serde_json::from_str(reply).expect("a JSON reply");
    let finish = reply["choices"][0]["finish_reason"]
        .as_str()
        .expect("a finish reason")
        .to_string();
    (
        reply["usage"]["completion_tokens"].as_u64().expect("usage"),
        finish,
    )
}

fn served_by(engine: RecordingEngine) -> (Arc<Served>, EngineThread, Arc<Metrics>) {
    let metrics = Arc::new(test_metrics("ignore-eos"));
    let (served, thread) = serve_with_fake(engine, tiny_model(), &metrics);
    (served, thread, metrics)
}

/// A request without `ignore_eos`, and one with it `false`, end at the model's EOS exactly as before the
/// field existed: two tokens, finish `stop`. Mutation: treat an absent field as `true` (give `ignore_eos` a
/// `default = "..."` returning true); the request runs to its cap and the row goes red.
#[test]
fn a_request_that_does_not_ask_stops_at_the_models_eos() {
    for body in [
        serde_json::json!({ "prompt": "a", "max_tokens": 8 }),
        serde_json::json!({ "prompt": "a", "max_tokens": 8, "ignore_eos": false }),
    ] {
        let (served, thread, metrics) = served_by(eos_on_the_third());
        let reply = handle_completion(&served, &metrics, &completion(body), &|| false)
            .expect("the completion succeeds");
        thread.finish(served);
        assert_eq!(reported(&reply), (2, "stop".to_string()));
    }
}

/// `ignore_eos: true` on the same scripted model generates all eight tokens, the EOS among them, and ends at
/// `length`. Mutation: drop the `!slot.ignore_eos` condition in `commit_tokens`; the request ends at the
/// EOS after two tokens and the row goes red.
#[test]
fn a_completion_that_ignores_eos_runs_to_its_cap() {
    let (served, thread, metrics) = served_by(eos_on_the_third());
    let body = serde_json::json!({ "prompt": "a", "max_tokens": 8, "ignore_eos": true });
    let reply = handle_completion(&served, &metrics, &completion(body), &|| false)
        .expect("the completion succeeds");
    thread.finish(served);
    assert_eq!(reported(&reply), (8, "length".to_string()));
}

/// The same on a chat route. Mutation: as above.
#[test]
fn a_chat_that_ignores_eos_runs_to_its_cap() {
    let (served, thread, metrics) = served_by(eos_on_the_third());
    let body = serde_json::json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "max_tokens": 8,
        "ignore_eos": true
    });
    let reply = handle_chat(&served, &metrics, &chat(body), &|| false).expect("the chat succeeds");
    thread.finish(served);
    assert_eq!(reported(&reply), (8, "length".to_string()));
}

/// A chat that ignores EOS carries none of its template's turn-end markers as stop strings, and one that
/// does not still carries them. Mutation: always extend with the turn stops in `shape_chat`; the first
/// assertion finds the marker and the row goes red.
#[test]
fn a_chat_that_ignores_eos_has_no_turn_stops() {
    let (served, queue) = served_over(
        tiny_model_with_chat_template(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}",
        ),
        1,
    );
    let submitted = |ignore_eos: bool| {
        let body = serde_json::json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "max_tokens": 4,
            "ignore_eos": ignore_eos
        });
        // Nothing reads the queue: the handler waits, and the client gives up at once.
        let _ = handle_chat(&served, &test_metrics("stops"), &chat(body), &|| true);
        let EngineMsg::Generate(job) = queue
            .recv_timeout(Duration::from_secs(5))
            .expect("the job is queued")
        else {
            panic!("a generation job");
        };
        (job.stop, job.ignore_eos)
    };
    let (stops, ignoring) = submitted(true);
    assert!(ignoring);
    assert!(stops.is_empty(), "no stop string: {stops:?}");
    let (stops, ignoring) = submitted(false);
    assert!(!ignoring);
    assert!(
        !stops.is_empty(),
        "the template's turn end is a stop without the field"
    );
}

/// A `client_gone` that fires after a while: a request wrongly admitted waits on a queue no engine reads, and
/// this turns that wait into a failed assertion instead of a hang.
fn gives_up_after_half_a_minute() -> impl Fn() -> bool {
    let started = Instant::now();
    move || started.elapsed() > Duration::from_secs(30)
}

/// Each field `ignore_eos` contradicts is refused by name as a 400, on both routes, and nothing reaches the
/// engine. Mutation: return `Ok(true)` from `check_ignore_eos` before the conflict scan; the request is
/// admitted and the row goes red.
#[test]
fn ignore_eos_with_a_field_it_contradicts_is_a_typed_400() {
    let completions = [
        ("stop", serde_json::json!({ "stop": ["x"] })),
        (
            "guided_choice",
            serde_json::json!({ "guided_choice": ["a"] }),
        ),
        ("guided_regex", serde_json::json!({ "guided_regex": "a+" })),
        (
            "guided_grammar",
            serde_json::json!({ "guided_grammar": "root ::= \"a\"" }),
        ),
        (
            "guided_json",
            serde_json::json!({ "guided_json": { "type": "string" } }),
        ),
        (
            "response_format",
            serde_json::json!({ "response_format": { "type": "json_object" } }),
        ),
    ];
    for (field, extra) in completions {
        let (served, queue) = served_over(tiny_model(), 1);
        let mut body = serde_json::json!({ "prompt": "a", "ignore_eos": true });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let error = handle_completion(
            &served,
            &test_metrics("refuse"),
            &completion(body),
            &gives_up_after_half_a_minute(),
        )
        .expect_err("the combination is refused");
        let refused = error.downcast_ref::<RefusedRequest>().expect("a refusal");
        assert!(refused.0.contains(field), "{field}: {}", refused.0);
        assert!(refused.0.contains("ignore_eos"), "{}", refused.0);
        assert_eq!(classify_handler_error(&error).0, 400, "{field}");
        assert!(queue.try_recv().is_err(), "{field} reached the engine");
    }
    for (field, extra) in [
        ("tools", serde_json::json!({ "tools": [] })),
        ("tool_choice", serde_json::json!({ "tool_choice": "auto" })),
        ("stop", serde_json::json!({ "stop": "x" })),
    ] {
        let (served, queue) = served_over(tiny_model(), 1);
        let mut body = serde_json::json!({
            "messages": [{ "role": "user", "content": "hi" }],
            "ignore_eos": true
        });
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let error = handle_chat(
            &served,
            &test_metrics("refuse"),
            &chat(body),
            &gives_up_after_half_a_minute(),
        )
        .expect_err("the combination is refused");
        let refused = error.downcast_ref::<RefusedRequest>().expect("a refusal");
        assert!(refused.0.contains(field), "{field}: {}", refused.0);
        assert_eq!(classify_handler_error(&error).0, 400, "{field}");
        assert!(queue.try_recv().is_err(), "{field} reached the engine");
    }
}

/// `response_format: {"type": "text"}` constrains nothing, so it is not a conflict.
#[test]
fn ignore_eos_with_a_plain_text_response_format_is_admitted() {
    let (served, thread, metrics) = served_by(eos_on_the_third());
    let body = serde_json::json!({
        "prompt": "a", "max_tokens": 4, "ignore_eos": true, "response_format": { "type": "text" }
    });
    let reply = handle_completion(&served, &metrics, &completion(body), &|| false)
        .expect("the completion succeeds");
    thread.finish(served);
    assert_eq!(reported(&reply), (4, "length".to_string()));
}

/// At the commit: a multi-token result with the EOS in the middle keeps all three tokens when the request
/// ignores EOS (the engine is told `kept = 3`, the EOS is in the sequence), where
/// `commit_tokens_keeps_the_tokens_before_an_eos_inside_a_multi_token_result` keeps one without it.
/// Mutation: drop the `!slot.ignore_eos` condition in `commit_tokens`; the request ends at the EOS and the
/// row goes red.
#[test]
fn commit_tokens_keeps_an_eos_when_the_request_ignores_it() {
    let mut engine = RecordingEngine::new(1);
    let (job, _submitted) = JobSpec::new("ab", 10).ignoring_eos().build();
    let mut slot = Slot::new(FakeSeq(1), job, 0, 10);
    let result = RowResult {
        seq: FakeSeq(1),
        consumed: 0,
        tokens: vec![5, 99, 7],
        logprobs: Vec::new(),
        speculation: RowSpeculation::NotApplicable,
    };
    let metrics = test_metrics("commit-ignore-eos");
    let fate = commit_tokens(&mut engine, &metrics, &mut slot, &result).unwrap();
    assert_eq!(fate, SlotFate::Continue);
    assert_eq!(engine.events(), vec![Event::Commit { seq: 1, kept: 3 }]);
    assert_eq!(slot.tokens, vec![97, 98, 5, 99, 7]);
    assert_eq!(metrics.completion_tokens.load(Ordering::Relaxed), 3);
}

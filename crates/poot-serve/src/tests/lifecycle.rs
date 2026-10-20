//! The one request lifecycle, every route and model kind over the recording fake running the production
//! loop on its own engine thread (POOT-589). Each row drives the production `http::handle` over a
//! loopback socket, or the loop itself for a job no route builds yet (a multimodal prompt). The fake
//! stands where the driver stands: a [`Served`] holds only the queue, so what the fake records is what
//! reached the engine thread.

use std::net::IpAddr;

use poot_llm::driver::Encoded;
use poot_llm::driver::error::FailureClass;

use super::*;
use crate::batch::engine::Exit;
use crate::types::{AdminRequest, EncodeRow};

/// A request through the production `handle` and the full response text.
pub(super) fn roundtrip(
    served: &Served,
    metrics: &Metrics,
    method: &str,
    path: &str,
    body: &str,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind lifecycle socket");
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect client");
    client
        .set_read_timeout(Some(Duration::from_secs(20)))
        .expect("bound the response read");
    let (server, _) = listener.accept().expect("accept client");
    let health = EngineHealth::serving_for_test("fake", 4, 64);
    let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
        .expect("loopback lora policy");
    write!(
        client,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .expect("write request");
    client.flush().expect("flush request");
    handle(
        server,
        Instant::now(),
        served,
        &health,
        metrics,
        &test_limits(),
        &policy,
    )
    .expect("the handler answers without a transport error");
    let mut response = String::new();
    client
        .read_to_string(&mut response)
        .expect("read the response to EOF");
    response
}

pub(super) fn status_line(response: &str) -> &str {
    response.lines().next().unwrap_or_default()
}

pub(super) fn json_body(response: &str) -> serde_json::Value {
    let (_, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no header/body split in {response:?}"));
    serde_json::from_str(body)
        .unwrap_or_else(|error| panic!("body is not JSON ({error}): {body:?}"))
}

/// A fake that picks a distinct token per step and can hold a whole chat prompt in one prefill step.
fn counting_engine(rows: usize) -> RecordingEngine {
    let mut engine = RecordingEngine::new(rows).picking(|_, committed| vec![committed as u32 + 1]);
    engine.widest = 256;
    engine.max_seq = 4096;
    engine
}

/// The failing step the rows below use: the third step (the first two stream a piece each).
fn faulting_third_step(engine: RecordingEngine) -> RecordingEngine {
    engine.on_step(|step, _| {
        if step == 3 {
            return Err(EngineError::Fault("device lost".into()));
        }
        Ok(())
    })
}

fn opens(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, Event::Open { .. }))
        .count()
}

fn encodes(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, Event::Encode { .. }))
        .count()
}

/// SC-001 (R477-002): an engine failure after the 200 headers is an error event and no success
/// metadata, on every generation route that streams: a finish reason, the usage receipt and `[DONE]`
/// all signal success, and a request nothing finished must not carry them. Mutation: report the engine
/// failure as a length-finished completion (the fault path delivers a `Done` instead of the failure);
/// the stream ends with a finish chunk, a usage receipt and `[DONE]`, and the row goes red.
#[test]
fn an_engine_failure_after_headers_is_an_error_event_and_no_success_metadata() {
    let model = tiny_model();
    for (path, body, piece) in [
        (
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":10,"stream":true}"#,
            "\"text\":\"<1>\"",
        ),
        (
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":10,"stream":true}"#,
            "\"content\":\"<1>\"",
        ),
    ] {
        let metrics = Arc::new(test_metrics("failure-after-headers"));
        let (served, engine) = serve_with_fake(
            faulting_third_step(counting_engine(1)),
            Arc::clone(&model),
            &metrics,
        );
        let response = roundtrip(&served, &metrics, "POST", path, body);
        engine.finish(served);
        assert!(
            response.starts_with("HTTP/1.1 200 OK\r\n"),
            "{path}: {response}"
        );
        assert!(
            response.contains(piece),
            "{path}: the pieces before the failure were streamed: {response}"
        );
        assert!(
            response.contains("\"type\":\"server_error\"") && response.contains("\"code\":500"),
            "{path}: the failure is an error event: {response}"
        );
        assert!(
            !response.contains("\"finish_reason\":\""),
            "{path}: no finish reason after a failure: {response}"
        );
        assert!(
            !response.contains("\"usage\"") && !response.contains("data: [DONE]"),
            "{path}: no success metadata after a failure: {response}"
        );
        assert_eq!(
            metrics.requests_errored.load(Ordering::Relaxed),
            1,
            "{path}"
        );
        assert_eq!(
            metrics.requests_completed.load(Ordering::Relaxed),
            0,
            "{path}"
        );
    }
}

/// SC-001, the encode kinds: a failed `Encode` step is an error status and no embeddings or ranking.
#[test]
fn a_failed_encode_step_is_an_error_response_for_embeddings_and_rerank() {
    let model = tiny_model();
    for (path, body, success_key) in [
        ("/v1/embeddings", r#"{"input":"a"}"#, "\"data\""),
        (
            "/v1/rerank",
            r#"{"query":"a","documents":["b","c"]}"#,
            "\"results\"",
        ),
    ] {
        let metrics = Arc::new(test_metrics("encode-failure"));
        let mut fake = counting_engine(8);
        fake.on_encode = Box::new(|_| Err(EngineError::Fault("device lost".into())));
        let (served, engine) = serve_with_fake(fake, Arc::clone(&model), &metrics);
        let response = roundtrip(&served, &metrics, "POST", path, body);
        engine.finish(served);
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 500 Internal Server Error",
            "{path}: {response}"
        );
        assert!(!response.contains(success_key), "{path}: {response}");
        assert_eq!(
            metrics.requests_errored.load(Ordering::Relaxed),
            1,
            "{path}"
        );
        assert_eq!(
            metrics.requests_completed.load(Ordering::Relaxed),
            0,
            "{path}"
        );
        assert_eq!(
            metrics.e2e.count(),
            0,
            "{path}: a failed request is not a finished one"
        );
    }
}

/// SC-001, the multimodal kind: a multimodal job whose step faults ends in a failure and never a `Done`,
/// and was opened with its embedding rows.
#[test]
fn a_failing_multimodal_job_ends_in_a_failure_and_no_completion() {
    let mut engine = faulting_third_step(counting_engine(1));
    let (_exit, metrics, submitted) = serve(
        &mut engine,
        vec![JobSpec::new("ab", 10).with_embeds().streaming()],
    );
    let (pieces, terminal) = outcome(&submitted[0]);
    assert!(
        matches!(&terminal, GenEvent::Failed(failure) if failure.class() == FailureClass::Fault),
        "a fault ends the request as a fault"
    );
    assert_eq!(pieces, vec!["<1>", "<2>"], "the pieces before the fault");
    assert!(matches!(
        engine.events()[0],
        Event::Open { embeds: true, .. }
    ));
    assert_eq!(metrics.requests_errored.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
}

/// SC-002 (R477-004): a guided, logprobs, `n` > 1 request is served with every field: each of its `n`
/// jobs reaches the engine with the request's logprobs and guided constraint on its sampler. The fake
/// picks through the sampler it is handed, as the driver does: a constrained sampler picks the choice's
/// token whatever the logits say, and a logprobs sampler records the pick. Mutation: drop the `n` field
/// in the request-to-job conversion (`n: 1` in `shape_completion`); one job opens, one choice returns,
/// and the row goes red.
#[test]
fn a_guided_logprobs_request_with_n_choices_runs_n_jobs_with_every_field() {
    let model = tiny_model();
    let forced = model.text().encode("c").expect("encode the choice")[0];
    let unconstrained = 1u32; // the argmax of the logits below
    assert_ne!(forced, unconstrained, "the constraint must change the pick");
    let seen: Arc<Mutex<Vec<(u64, u32, usize)>>> = Arc::default();
    let probe_seen = Arc::clone(&seen);
    let fake = counting_engine(4).probing(move |seq, sampler| {
        let mut seen = probe_seen.lock().unwrap();
        if seen.iter().any(|(known, ..)| *known == seq) {
            return;
        }
        let pick = sampler.pick(&[0.0, 3.0, 1.0, 2.0]).expect("a finite pick") as u32;
        seen.push((seq, pick, sampler.take_logprobs().len()));
    });
    let metrics = Arc::new(test_metrics("n-choices"));
    let (served, engine) = serve_with_fake(fake, Arc::clone(&model), &metrics);
    let body = r#"{"prompt":"a","max_tokens":2,"n":3,"logprobs":2,"guided_choice":["c"]}"#;
    let response = roundtrip(&served, &metrics, "POST", "/v1/completions", body);
    let events = engine.finish(served).events();
    assert_eq!(status_line(&response), "HTTP/1.1 200 OK", "{response}");
    let json = json_body(&response);
    let choices = json["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 3, "n = 3 choices: {json}");
    for (index, choice) in choices.iter().enumerate() {
        assert_eq!(choice["index"], index);
        assert!(
            choice["logprobs"].is_object(),
            "logprobs reach the response: {choice}"
        );
    }
    assert_eq!(
        opens(&events),
        3,
        "n = 3 jobs reached the engine: {events:?}"
    );
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3, "each job's sampler was probed");
    for &(seq, pick, logprobs) in seen.iter() {
        assert_eq!(
            pick, forced,
            "job {seq}: the guided constraint reached its sampler"
        );
        assert_eq!(
            logprobs, 1,
            "job {seq}: the logprobs request reached its sampler"
        );
    }
}

/// SC-002: what the lifecycle cannot serve as asked it refuses with a typed 400 before the engine sees
/// anything: more choices than the engine has rows, logprobs on a stream (which has nowhere to put
/// them), a named adapter no engine loaded, and an image part the model has no front end for.
#[test]
fn a_request_the_engine_cannot_serve_as_asked_is_a_typed_400_and_reaches_no_engine() {
    let model = tiny_model();
    let metrics = Arc::new(test_metrics("typed-400"));
    let (served, engine) = serve_with_fake(counting_engine(2), Arc::clone(&model), &metrics);
    let image = r#"{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}"#;
    for (case, path, body, needle) in [
        (
            "n above the rows",
            "/v1/completions",
            r#"{"prompt":"a","n":3}"#.to_string(),
            "exceed the engine's 2 rows",
        ),
        (
            "n above the rows, streamed",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"a"}],"n":3,"stream":true}"#.to_string(),
            "exceed the engine's 2 rows",
        ),
        (
            "streamed logprobs",
            "/v1/completions",
            r#"{"prompt":"a","logprobs":1,"stream":true}"#.to_string(),
            "logprobs are not streamed",
        ),
        (
            "streamed chat logprobs",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"a"}],"logprobs":true,"stream":true}"#
                .to_string(),
            "logprobs are not streamed",
        ),
        (
            "unknown adapter",
            "/v1/completions",
            r#"{"prompt":"a","lora_adapter":"style"}"#.to_string(),
            "unknown lora_adapter",
        ),
        (
            "image part",
            "/v1/chat/completions",
            format!(r#"{{"messages":[{image}]}}"#),
            "no image front end",
        ),
    ] {
        let response = roundtrip(&served, &metrics, "POST", path, &body);
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 400 Bad Request",
            "{case}: {response}"
        );
        let message = json_body(&response)["error"]["message"].to_string();
        assert!(message.contains(needle), "{case}: {message}");
    }
    let events = engine.finish(served).events();
    assert!(
        events.is_empty(),
        "a refused request never reaches the engine: {events:?}"
    );
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inflight.load(Ordering::Relaxed), 0);
}

/// A chat template that raises on the conversation refuses the request: a 400 that carries the template's
/// own message, on the streaming route before its 200 headers too, with no other format answering and
/// nothing reaching the engine.
#[test]
fn a_chat_template_that_raises_is_a_400_naming_its_error_and_reaches_no_engine() {
    let model = tiny_model_with_chat_template(
        "{{ raise_exception('conversation roles must alternate user and assistant') }}",
    );
    let metrics = Arc::new(test_metrics("template-raises"));
    let (served, engine) = serve_with_fake(counting_engine(1), model, &metrics);
    for body in [
        r#"{"messages":[{"role":"user","content":"a"}]}"#,
        r#"{"messages":[{"role":"user","content":"a"}],"stream":true}"#,
    ] {
        let response = roundtrip(&served, &metrics, "POST", "/v1/chat/completions", body);
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 400 Bad Request",
            "{response}"
        );
        let message = json_body(&response)["error"]["message"].to_string();
        assert!(
            message.contains("conversation roles must alternate"),
            "{message}"
        );
    }
    let events = engine.finish(served).events();
    assert!(
        events.is_empty(),
        "a refused request never reaches the engine: {events:?}"
    );
    assert_eq!(metrics.inflight.load(Ordering::Relaxed), 0);
}

/// The JSON of every `data:` frame of a streamed response, `[DONE]` excluded.
fn sse_frames(response: &str) -> Vec<serde_json::Value> {
    let body = response.split_once("\r\n\r\n").map_or("", |(_, body)| body);
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap_or_else(|e| panic!("{e}: {data}")))
        .collect()
}

/// The two routes a request can stream or not, with the body that asks for `n` choices.
const N_ROUTES: [(&str, &str); 2] = [
    ("/v1/completions", r#""prompt":"a""#),
    (
        "/v1/chat/completions",
        r#""messages":[{"role":"user","content":"a"}]"#,
    ),
];

/// SC-002, streamed: `n` choices of a streamed request are `n` concurrent jobs on both routes. The fake
/// engine sees two opens, the chunks carry both choice indices, each choice ends in its own finish chunk,
/// and the usage receipt counts both choices' tokens. Mutation: `shaped.n` replaced by 1 in
/// `stream_completion` or `stream_chat`; one job opens, one index appears, and the row goes red.
#[test]
fn a_streamed_request_with_n_choices_runs_n_jobs_and_reports_every_choice() {
    let model = tiny_model();
    for (path, fields) in N_ROUTES {
        let metrics = Arc::new(test_metrics("streamed-n"));
        let (served, engine) = serve_with_fake(counting_engine(2), Arc::clone(&model), &metrics);
        let body = format!("{{{fields},\"max_tokens\":2,\"n\":2,\"stream\":true}}");
        let response = roundtrip(&served, &metrics, "POST", path, &body);
        let events = engine.finish(served).events();
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "{path}: {response}"
        );
        assert_eq!(
            opens(&events),
            2,
            "{path}: two jobs reached the engine: {events:?}"
        );
        let frames = sse_frames(&response);
        let index_of = |frame: &serde_json::Value| frame["choices"][0]["index"].as_u64();
        let mut indices: Vec<u64> = frames.iter().filter_map(index_of).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(
            indices,
            vec![0, 1],
            "{path}: chunks carry both indices: {response}"
        );
        let mut finished: Vec<u64> = frames
            .iter()
            .filter(|frame| frame["choices"][0]["finish_reason"].is_string())
            .filter_map(index_of)
            .collect();
        finished.sort_unstable();
        assert_eq!(
            finished,
            vec![0, 1],
            "{path}: one finish chunk per choice: {response}"
        );
        let receipt = frames
            .iter()
            .find(|frame| frame["usage"].is_object())
            .unwrap_or_else(|| panic!("{path}: no usage receipt: {response}"));
        assert_eq!(
            receipt["usage"]["completion_tokens"], 4,
            "{path}: two choices of two tokens: {receipt}"
        );
    }
}

/// One request reports the same usage streamed and not: both paths count the completion tokens with
/// `completion_tokens` (the generated ids, or the cut text's tokens after a stop string), with `n` choices
/// summed. Mutation: re-tokenize the text on the streamed path only; the counts disagree and the row goes
/// red.
#[test]
fn a_request_reports_the_same_usage_streamed_and_not() {
    let model = tiny_model();
    for (path, fields) in N_ROUTES {
        let usage_of = |stream: bool, n: usize| {
            let metrics = Arc::new(test_metrics("usage-equality"));
            let (served, engine) =
                serve_with_fake(counting_engine(2), Arc::clone(&model), &metrics);
            let body = format!("{{{fields},\"max_tokens\":3,\"n\":{n},\"stream\":{stream}}}");
            let response = roundtrip(&served, &metrics, "POST", path, &body);
            engine.finish(served);
            assert_eq!(
                status_line(&response),
                "HTTP/1.1 200 OK",
                "{path}: {response}"
            );
            if stream {
                let frames = sse_frames(&response);
                let receipt = frames
                    .iter()
                    .find(|f| f["usage"].is_object())
                    .expect("receipt");
                receipt["usage"].clone()
            } else {
                json_body(&response)["usage"].clone()
            }
        };
        for n in [1, 2] {
            let (plain, streamed) = (usage_of(false, n), usage_of(true, n));
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
                assert_eq!(
                    plain[key], streamed[key],
                    "{path} n={n} {key}: {plain} vs {streamed}"
                );
            }
            assert_eq!(plain["completion_tokens"], 3 * n, "{path} n={n}: {plain}");
        }
    }
}

/// SC-003 (R477-005): every route, streamed or not, ends in the request metrics: the request count of
/// the latency histogram and the prompt tokens it prefilled, counted once per job where `submit` sees
/// the request finish. Mutation: skip the metrics accounting for the route (`record_request` in
/// `submit`); the route's request leaves no trace and the row goes red.
#[test]
fn every_generation_route_appears_in_the_request_metrics() {
    let model = tiny_model();
    let completion_prompt = model.text().encode("a").unwrap().len();
    let chat_prompt = {
        let msgs = serde_json::json!([{"role": "user", "content": "a"}]);
        let rendered = model.text().render_chat_value(&msgs, None).unwrap();
        model.text().encode_chat(&rendered).unwrap().len()
    };
    for (case, path, body) in [
        (
            "completion",
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":2}"#,
        ),
        (
            "streamed completion",
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":2,"stream":true}"#,
        ),
        (
            "chat",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":2}"#,
        ),
        (
            "streamed chat",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":2,"stream":true}"#,
        ),
    ] {
        let metrics = Arc::new(test_metrics("request-metrics"));
        let (served, engine) = serve_with_fake(counting_engine(1), Arc::clone(&model), &metrics);
        let response = roundtrip(&served, &metrics, "POST", path, body);
        engine.finish(served);
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "{case}: {response}"
        );
        assert_eq!(
            metrics.e2e.count(),
            1,
            "{case}: the request is in the latency histogram"
        );
        let prompt_tokens = if path == "/v1/completions" {
            completion_prompt
        } else {
            chat_prompt
        };
        assert_eq!(
            metrics.prompt_tokens.load(Ordering::Relaxed),
            prompt_tokens as u64,
            "{case}: its prompt tokens are counted once, by the loop's prefill"
        );
        let streamed = case.contains("streamed");
        assert_eq!(
            metrics.ttft.count(),
            u64::from(streamed),
            "{case}: only a stream observes a first token"
        );
    }
}

/// SC-003, the multimodal kind: a job with embedding rows is a request like any other. It goes through
/// `submit`, so it is in the request metrics, and the engine saw its rows.
#[test]
fn a_multimodal_job_appears_in_the_request_metrics() {
    let metrics = Arc::new(test_metrics("multimodal-metrics"));
    let (served, engine) = serve_with_fake(counting_engine(1), tiny_model(), &metrics);
    let mut request = generation_request("ab", 2, false);
    request.prompt = PromptInput {
        tokens: tokens_of("ab"),
        inputs: embed_inputs(1, vec![0], vec![1.0]),
    };
    let completion = submit_one(&served.engine, &metrics, request, &|| false, |_, _| {
        GenerationControl::Continue(())
    })
    .expect("the multimodal job completes");
    let events = engine.finish(served).events();
    assert_eq!(completion.prompt_tokens, 2);
    assert!(
        matches!(events[0], Event::Open { embeds: true, .. }),
        "{events:?}"
    );
    assert_eq!(
        metrics.e2e.count(),
        1,
        "the job is in the latency histogram"
    );
    assert_eq!(metrics.prompt_tokens.load(Ordering::Relaxed), 2);
}

/// SC-004 (R484-007): a typed engine error is answered by its class, not by a blanket 500: a request the
/// engine refuses is the client's 400, one this engine cannot serve is a 503, a step failure the engine survived and a fault are 500s. Streamed
/// the class is the status the error event carries. Mutation: map every engine error to 500
/// (`class_status`); the refusal rows read 500 and the row goes red.
#[test]
fn typed_engine_errors_are_answered_by_class() {
    let model = tiny_model();
    for (class, status, line, type_) in [
        (
            FailureClass::Request,
            400u16,
            "HTTP/1.1 400 Bad Request",
            "invalid_request_error",
        ),
        (
            FailureClass::Unsupported,
            503,
            "HTTP/1.1 503 Service Unavailable",
            "server_error",
        ),
        (
            FailureClass::Step,
            500,
            "HTTP/1.1 500 Internal Server Error",
            "server_error",
        ),
        (
            FailureClass::Fault,
            500,
            "HTTP/1.1 500 Internal Server Error",
            "server_error",
        ),
    ] {
        let failing_step = |engine: RecordingEngine| {
            engine.on_step(move |_, _| {
                Err(match class {
                    FailureClass::Fault => EngineError::Fault("device lost".into()),
                    class => EngineError::Refused {
                        class,
                        reason: format!("{class:?} refusal"),
                    },
                })
            })
        };
        let metrics = Arc::new(test_metrics("class-status"));
        let (served, engine) = serve_with_fake(
            failing_step(counting_engine(1)),
            Arc::clone(&model),
            &metrics,
        );
        let response = roundtrip(
            &served,
            &metrics,
            "POST",
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":3}"#,
        );
        engine.finish(served);
        assert_eq!(status_line(&response), line, "{class:?}: {response}");
        let error = &json_body(&response)["error"];
        assert_eq!(error["code"], status, "{class:?}");
        assert_eq!(error["type"], type_, "{class:?}");

        // Streamed, the 200 is already sent: the event carries the class's status.
        let metrics = Arc::new(test_metrics("class-status-stream"));
        let (served, engine) = serve_with_fake(
            failing_step(counting_engine(1)),
            Arc::clone(&model),
            &metrics,
        );
        let response = roundtrip(
            &served,
            &metrics,
            "POST",
            "/v1/completions",
            r#"{"prompt":"a","max_tokens":3,"stream":true}"#,
        );
        engine.finish(served);
        assert!(
            response.contains(&format!("\"code\":{status}")),
            "{class:?} streamed: {response}"
        );
    }
}

/// SC-005 (ruling V5/S8): embeddings and rerank requests are `EngineMsg::Encode` jobs, and they are in
/// the request metrics like any other request: its latency, and the tokens it encoded as prompt tokens.
/// Mutation: skip `record_request` for `Encode` (in `submit_encode`); neither request leaves a trace
/// and the row goes red.
#[test]
fn embeddings_and_rerank_go_through_encode_and_appear_in_the_request_metrics() {
    let model = tiny_model();
    let tokens = |text: &str| {
        model
            .text()
            .encode_for_encoder(text, None)
            .unwrap()
            .ids
            .len()
    };
    let metrics = Arc::new(test_metrics("encode-metrics"));
    let (served, engine) = serve_with_fake(counting_engine(8), Arc::clone(&model), &metrics);
    let embeddings = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":["a","ab"]}"#,
    );
    assert_eq!(status_line(&embeddings), "HTTP/1.1 200 OK", "{embeddings}");
    assert_eq!(
        metrics.e2e.count(),
        1,
        "the embeddings request is in the latency histogram"
    );
    let embedded = tokens("a") + tokens("ab");
    assert_eq!(
        metrics.prompt_tokens.load(Ordering::Relaxed),
        embedded as u64
    );
    let rerank = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/rerank",
        r#"{"query":"a","documents":["b","c"]}"#,
    );
    assert_eq!(status_line(&rerank), "HTTP/1.1 200 OK", "{rerank}");
    assert_eq!(
        metrics.e2e.count(),
        2,
        "the rerank request is in the latency histogram"
    );
    assert_eq!(
        metrics.prompt_tokens.load(Ordering::Relaxed),
        (embedded + tokens("a") + tokens("b") + tokens("c")) as u64
    );
    let events = engine.finish(served).events();
    assert_eq!(
        events,
        vec![Event::Encode { rows: 2 }, Event::Encode { rows: 3 }],
        "one Encode step per request: the two inputs, then the query and its two documents"
    );
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 2);
}

/// SC-006 (R-589-3): a served `/v1/embeddings` request on the fixture decoder is answered by the engine
/// thread: the fake records one `Encode`, on a thread that is not the connection thread, with the rows of
/// the request's tokens. Mutation: route embeddings on the connection thread (fabricate the encoded rows
/// in `handle_embeddings` instead of `submit_encode`); the fake records nothing and the row goes red.
#[test]
fn a_served_embeddings_request_is_answered_by_the_engine_thread() {
    let model = tiny_model();
    let connection_thread = std::thread::current().id();
    type Ran = Vec<(std::thread::ThreadId, Vec<EncodeRow>)>;
    let ran_on: Arc<Mutex<Ran>> = Arc::default();
    let hook_ran_on = Arc::clone(&ran_on);
    let mut fake = counting_engine(8);
    fake.on_encode = Box::new(move |rows| {
        hook_ran_on
            .lock()
            .unwrap()
            .push((std::thread::current().id(), rows.to_vec()));
        Ok(Encoded {
            width: 2,
            rows: rows
                .iter()
                .map(|row| row.ids.iter().flat_map(|&id| [id as f32, 1.0]).collect())
                .collect(),
        })
    });
    let metrics = Arc::new(test_metrics("engine-thread-embeddings"));
    let (served, engine) = serve_with_fake(fake, Arc::clone(&model), &metrics);
    let response = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":"ab"}"#,
    );
    let events = engine.finish(served).events();
    assert_eq!(status_line(&response), "HTTP/1.1 200 OK", "{response}");
    assert_eq!(
        events,
        vec![Event::Encode { rows: 1 }],
        "one Encode, answered by the fake"
    );
    let ran_on = ran_on.lock().unwrap();
    assert_eq!(ran_on.len(), 1);
    assert_ne!(
        ran_on[0].0, connection_thread,
        "the step ran on the engine thread, not the connection thread"
    );
    let expected = model.text().encode_for_encoder("ab", None).unwrap();
    assert_eq!(
        ran_on[0].1,
        vec![EncodeRow {
            ids: expected.ids,
            types: expected.types
        }],
        "the engine got the request's tokens"
    );
}

/// An encode job waits for the sequences in flight to finish, and nothing new opens meanwhile: the
/// driver refuses an encode step beside an open sequence, so the loop runs it with the engine empty. The
/// fake asserts that no sequence is open when it is asked to encode. The jobs arrive while the two
/// sequences run: the first step queues an encode job and a generation job behind them.
#[test]
fn an_encode_job_runs_with_no_sequence_open_and_after_the_running_ones() {
    let (tx, rx) = mpsc::channel();
    let mut requests = Vec::new();
    for prompt in ["ab", "cd"] {
        let (job, submitted) = JobSpec::new(prompt, 3).build();
        tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
        requests.push(submitted);
    }
    let (reply, encoded) = mpsc::channel();
    let (_guard, cancel) = crate::lifecycle::JobGuard::new();
    let (late, late_submitted) = JobSpec::new("ef", 1).build();
    let mut arriving = Some((tx.clone(), reply, cancel, late));
    drop(tx);
    let mut engine = counting_engine(2).on_step(move |step, _| {
        if step == 1 {
            let (tx, reply, cancel, late) = arriving.take().expect("the first step runs once");
            tx.send(EngineMsg::Encode(crate::types::EncodeJob {
                rows: vec![EncodeRow {
                    ids: vec![1, 2],
                    types: Vec::new(),
                }],
                reply,
                cancel,
            }))
            .unwrap();
            // A request that arrives behind the encode waits for it.
            tx.send(EngineMsg::Generate(Box::new(late))).unwrap();
        }
        Ok(())
    });
    engine.slots = 2;
    let metrics = test_metrics("encode-order");
    assert_eq!(run_engine(&mut engine, &rx, &metrics), Exit::Shutdown);
    assert!(
        encoded
            .try_recv()
            .expect("the encode job was answered")
            .is_ok()
    );
    let events = engine.events();
    let encode_at = events
        .iter()
        .position(|event| matches!(event, Event::Encode { .. }))
        .expect("the encode step ran");
    let opened_before = opens(&events[..encode_at]);
    let released_before = events[..encode_at]
        .iter()
        .filter(|event| matches!(event, Event::Release { .. }))
        .count();
    assert_eq!(
        opened_before, 2,
        "the running requests were not stopped: {events:?}"
    );
    assert_eq!(
        released_before, 2,
        "they finished before the encode step: {events:?}"
    );
    assert!(
        events[encode_at..]
            .iter()
            .any(|event| matches!(event, Event::Open { .. })),
        "the request behind the encode opened after it: {events:?}"
    );
    for request in requests.iter().chain([&late_submitted]) {
        assert!(matches!(outcome(request).1, GenEvent::Done(..)));
    }
}

/// SC-007: a multimodal prompt opens with its embedding rows and skips prefix matching: the second of
/// two requests over the same tokens finds the first's prompt registered when it has no rows (it reuses
/// it) and reuses nothing when it carries rows, and its release registers no prefix. Mutation: open the
/// sequence without the rows (`embeds: None` in `admit`); the engine records an open with no rows and
/// the row goes red. (The device row compares the loop's tokens with `Driver::generate`.)
#[test]
fn a_multimodal_job_opens_with_its_rows_and_skips_prefix_matching() {
    let mut engine = counting_engine(1);
    let prompt = "abcdefghi";
    let (_exit, metrics, submitted) = serve(
        &mut engine,
        vec![
            JobSpec::new(prompt, 1),
            JobSpec::new(prompt, 1),
            JobSpec::new(prompt, 1).with_embeds(),
        ],
    );
    let events = engine.events();
    let opened: Vec<(usize, bool)> = events
        .iter()
        .filter_map(|event| match event {
            Event::Open { reused, embeds, .. } => Some((*reused, *embeds)),
            _ => None,
        })
        .collect();
    assert_eq!(
        opened,
        vec![(0, false), (8, false), (0, true)],
        "the plain repeat reuses the registered prefix; the multimodal one opens with its rows and \
         reuses none: {events:?}"
    );
    let registered_before_multimodal = 8 + 8; // the first two requests' filled blocks
    assert_eq!(
        metrics
            .cache
            .prefix_registered_tokens
            .load(Ordering::Relaxed),
        registered_before_multimodal,
        "the multimodal request's release registers no prefix"
    );
    assert_eq!(
        metrics
            .cache
            .prefix_skipped_releases
            .load(Ordering::Relaxed),
        1,
        "its release is the one that skipped"
    );
    for request in &submitted {
        assert!(matches!(outcome(request).1, GenEvent::Done(..)));
    }
}

/// SC-008 (ruling P7-2): each route this card restores answers 200 on the fixture instead of the typed
/// 503 it returned since POOT-739: embeddings, rerank, the three LoRA administration operations, and the
/// request-side text services (tokenization, token counts and chat rendering, which the completion and
/// chat routes run on the connection thread through the loaded model). Mutation: leave one route on the
/// 503 path; that route's row goes red.
#[test]
fn every_restored_route_answers_on_the_fixture() {
    let model = tiny_model();
    let metrics = Arc::new(test_metrics("restored-routes"));
    let (served, engine) = serve_with_fake(counting_engine(8), Arc::clone(&model), &metrics);
    let rendered_tokens = |content: &str| {
        let msgs = serde_json::json!([{"role": "user", "content": content}]);
        let rendered = model.text().render_chat_value(&msgs, None).unwrap();
        model.text().encode_chat(&rendered).unwrap().len()
    };
    for (route, method, path, body) in [
        ("embeddings", "POST", "/v1/embeddings", r#"{"input":"a"}"#),
        (
            "rerank",
            "POST",
            "/v1/rerank",
            r#"{"query":"a","documents":["b","c"]}"#,
        ),
        ("lora list", "GET", "/v1/lora_adapters", ""),
        (
            "lora load",
            "POST",
            "/v1/lora_adapters",
            r#"{"name":"style","path":"/adapters/style"}"#,
        ),
        ("lora unload", "DELETE", "/v1/lora_adapters/style", ""),
        (
            "tokenization and token counts",
            "POST",
            "/v1/completions",
            r#"{"prompt":"ab","max_tokens":2}"#,
        ),
        (
            "chat rendering",
            "POST",
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"ab"}],"max_tokens":2}"#,
        ),
    ] {
        let response = roundtrip(&served, &metrics, method, path, body);
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 200 OK",
            "{route}: the route answers, not the typed 503: {response}"
        );
        let json = json_body(&response);
        match route {
            "tokenization and token counts" => {
                let tokens = model.text().encode("ab").unwrap().len();
                assert_eq!(json["usage"]["prompt_tokens"], tokens, "{route}: {json}");
                assert_eq!(
                    json["usage"]["completion_tokens"], 2,
                    "{route}: counted from the generated tokens: {json}"
                );
            }
            "chat rendering" => assert_eq!(
                json["usage"]["prompt_tokens"],
                rendered_tokens("ab"),
                "{route}: the prompt is the model's rendering of the conversation: {json}"
            ),
            "lora load" => assert_eq!(json, serde_json::json!({"name": "style", "index": 1})),
            "lora unload" => {
                assert_eq!(json, serde_json::json!({"name": "style", "freed_index": 1}));
            }
            "lora list" => assert_eq!(json["object"], "list", "{route}: {json}"),
            "embeddings" => assert_eq!(json["data"].as_array().map(Vec::len), Some(1), "{json}"),
            _ => assert_eq!(json["results"].as_array().map(Vec::len), Some(2), "{json}"),
        }
    }
    let events = engine.finish(served).events();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, Event::Admin(_)))
            .count(),
        3,
        "the three administration operations ran on the engine thread: {events:?}"
    );
    assert!(events.contains(&Event::Admin(AdminRequest::Load {
        name: "style".into(),
        path: "/adapters/style".into()
    })));
}

/// The administration errors keep their statuses: a refusal by class, an unknown name 404, an adapter
/// running requests hold 409.
#[test]
fn administration_errors_are_answered_by_their_kind() {
    use crate::types::AdminError;
    let model = tiny_model();
    for (error, line) in [
        (
            AdminError::NotFound("style".into()),
            "HTTP/1.1 404 Not Found",
        ),
        (
            AdminError::Busy("held by 2 live sequences".into()),
            "HTTP/1.1 409 Conflict",
        ),
        (
            AdminError::Refused {
                class: FailureClass::Request,
                reason: "no adapter at that path".into(),
            },
            "HTTP/1.1 400 Bad Request",
        ),
        (
            AdminError::Refused {
                class: FailureClass::Unsupported,
                reason: "the retention limit".into(),
            },
            "HTTP/1.1 503 Service Unavailable",
        ),
    ] {
        let metrics = Arc::new(test_metrics("admin-errors"));
        let mut fake = counting_engine(1);
        fake.admin_error = Some(error.clone());
        let (served, engine) = serve_with_fake(fake, Arc::clone(&model), &metrics);
        let response = roundtrip(&served, &metrics, "DELETE", "/v1/lora_adapters/style", "");
        engine.finish(served);
        assert_eq!(status_line(&response), line, "{error:?}: {response}");
        assert!(
            json_body(&response)["error"]["message"]
                .as_str()
                .is_some_and(|message| message == error.to_string()),
            "{error:?}: {response}"
        );
    }
}

/// The embeddings a request returns are the model's hidden states pooled as the checkpoint says and
/// normalized: the fake's hidden state for a token is `[id, 1]`, so a text's mean-pooled vector is
/// `[mean id, 1]` over its norm. `encoding_format: base64` carries the same floats, and an unknown
/// pooling or format is refused rather than ignored.
#[test]
fn embeddings_are_pooled_and_normalized_and_every_option_is_honored_or_refused() {
    let model = tiny_model();
    let ids = |text: &str| model.text().encode_for_encoder(text, None).unwrap().ids;
    let mean_vector = |text: &str| {
        let ids = ids(text);
        let mean = ids.iter().map(|&id| id as f32).sum::<f32>() / ids.len() as f32;
        let norm = (mean * mean + 1.0).sqrt();
        vec![mean / norm, 1.0 / norm]
    };
    let metrics = Arc::new(test_metrics("embedding-values"));
    let (served, engine) = serve_with_fake(counting_engine(8), Arc::clone(&model), &metrics);
    let response = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":["ab","c"]}"#,
    );
    let json = json_body(&response);
    for (index, text) in ["ab", "c"].into_iter().enumerate() {
        let got: Vec<f32> = json["data"][index]["embedding"]
            .as_array()
            .expect("float embedding")
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let want = mean_vector(text);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{text}: {got:?} vs {want:?}");
        }
        assert_eq!(json["data"][index]["index"], index);
    }
    assert_eq!(
        json["usage"]["prompt_tokens"],
        ids("ab").len() + ids("c").len()
    );

    // base64: the same floats, little-endian.
    let b64 = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":"ab","encoding_format":"base64"}"#,
    );
    let encoded = json_body(&b64)["data"][0]["embedding"]
        .as_str()
        .expect("a base64 string")
        .to_string();
    {
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let floats: Vec<f32> = bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let want = mean_vector("ab");
        assert!(floats.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-5));
    }

    // CLS pooling takes the first token's state.
    let cls = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":"ab","pooling":"cls"}"#,
    );
    let first = ids("ab")[0] as f32;
    let norm = (first * first + 1.0).sqrt();
    let got = json_body(&cls)["data"][0]["embedding"][0].as_f64().unwrap() as f32;
    assert!((got - first / norm).abs() < 1e-5, "cls: {got}");

    for (case, body) in [
        ("unknown pooling", r#"{"input":"a","pooling":"max"}"#),
        (
            "unknown format",
            r#"{"input":"a","encoding_format":"int8"}"#,
        ),
        ("no inputs", r#"{"input":[]}"#),
        ("malformed", r#"{"input":3}"#),
    ] {
        let response = roundtrip(&served, &metrics, "POST", "/v1/embeddings", body);
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 400 Bad Request",
            "{case}: {response}"
        );
    }
    let events = engine.finish(served).events();
    assert_eq!(
        encodes(&events),
        3,
        "only the three valid requests reached the engine"
    );
}

/// Rerank orders the documents by cosine of the pooled vectors, most relevant first, and `top_n` cuts it.
#[test]
fn rerank_orders_documents_by_cosine_and_top_n_cuts_the_list() {
    let model = tiny_model();
    let ids = |text: &str| model.text().encode_for_encoder(text, None).unwrap().ids;
    let unit = |text: &str| {
        let ids = ids(text);
        let mean = ids.iter().map(|&id| id as f32).sum::<f32>() / ids.len() as f32;
        let norm = (mean * mean + 1.0).sqrt();
        [mean / norm, 1.0 / norm]
    };
    let cosine = |a: &str, b: &str| {
        let (a, b) = (unit(a), unit(b));
        a[0] * b[0] + a[1] * b[1]
    };
    let documents = ["b", "c", "ab"];
    let mut want: Vec<(usize, f32)> = documents
        .iter()
        .enumerate()
        .map(|(index, doc)| (index, cosine("a", doc)))
        .collect();
    want.sort_by(|x, y| y.1.total_cmp(&x.1));
    let metrics = Arc::new(test_metrics("rerank-order"));
    let (served, engine) = serve_with_fake(counting_engine(8), Arc::clone(&model), &metrics);
    let response = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/rerank",
        r#"{"query":"a","documents":["b","c","ab"]}"#,
    );
    let results = json_body(&response)["results"].as_array().unwrap().clone();
    assert_eq!(results.len(), 3);
    for (result, (index, score)) in results.iter().zip(&want) {
        assert_eq!(result["index"], *index, "{results:?} vs {want:?}");
        assert!((result["relevance_score"].as_f64().unwrap() as f32 - score).abs() < 1e-5);
    }
    let cut = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/rerank",
        r#"{"query":"a","documents":["b","c","ab"],"top_n":1}"#,
    );
    assert_eq!(json_body(&cut)["results"].as_array().unwrap().len(), 1);
    let none = roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/rerank",
        r#"{"query":"a","documents":[]}"#,
    );
    assert_eq!(json_body(&none)["results"], serde_json::json!([]));
    let events = engine.finish(served).events();
    assert_eq!(
        encodes(&events),
        2,
        "an empty document list needs no Encode step"
    );
}

/// A request invalid on its face is the 400 it always was, before it touches the engine: the JSON
/// parse, `max_tokens` typing, an unknown LoRA adapter, and an unsatisfiable guided-decoding regex. The
/// queue has no engine behind it, so a request that got past validation would fail at the send.
#[test]
fn a_request_invalid_on_its_face_is_a_400_before_the_engine() {
    let model = tiny_model();
    let metrics = Arc::new(test_metrics("invalid-on-its-face"));
    let (served, queue) = served_over(model, 1);
    drop(queue);
    for (path, fields, stream) in GENERATION_ROUTES {
        let label = format!("{path} stream={stream}");
        let cases = [
            ("malformed JSON", "{not json".to_string()),
            (
                "max_tokens is not an integer",
                generation_body(fields, Some("-1"), stream),
            ),
            (
                "unknown lora adapter",
                format!("{{{fields},\"lora_adapter\":\"no-such-adapter\",\"stream\":{stream}}}"),
            ),
            (
                "invalid guided regex",
                format!("{{{fields},\"guided_regex\":\"(\",\"stream\":{stream}}}"),
            ),
            (
                "over-long guided regex",
                format!(
                    "{{{fields},\"guided_regex\":\"{}\",\"stream\":{stream}}}",
                    "a".repeat(5000)
                ),
            ),
        ];
        for (case, body) in cases {
            let response = roundtrip(&served, &metrics, "POST", path, &body);
            assert_eq!(
                status_line(&response),
                "HTTP/1.1 400 Bad Request",
                "{label}: {case}: validation must precede the engine: {response}"
            );
        }
    }
    assert_eq!(metrics.requests_errored.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inflight.load(Ordering::Relaxed), 0);
}

/// A `guided_regex` whose automaton would have 2^18 states is refused as a 400 within a second on every
/// generation route, and the server keeps answering `/health`. Mutation: remove the DFA size limits in
/// `TextCodec::guided_dfa`; the request builds the automaton and runs far past the deadline.
#[test]
fn a_state_explosion_guided_regex_is_a_prompt_400_and_the_server_stays_up() {
    let model = tiny_model();
    let metrics = Arc::new(test_metrics("state-explosion"));
    let (served, queue) = served_over(model, 1);
    drop(queue);
    for (path, fields, stream) in GENERATION_ROUTES {
        let body =
            format!("{{{fields},\"guided_regex\":\"(a|b)*a(a|b){{17}}\",\"stream\":{stream}}}");
        let started = Instant::now();
        let response = roundtrip(&served, &metrics, "POST", path, &body);
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 400 Bad Request",
            "{path} stream={stream}: {response}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "{path} stream={stream}: the refusal took {:?}",
            started.elapsed()
        );
    }
    let health = roundtrip(&served, &metrics, "GET", "/health", "");
    assert_eq!(status_line(&health), "HTTP/1.1 200 OK", "{health}");
}

/// The four generation requests: both routes, streaming and not. `max_tokens` is spliced in as raw JSON
/// so a row can send a value no Rust integer type holds.
const GENERATION_ROUTES: [(&str, &str, bool); 4] = [
    ("/v1/completions", r#""prompt":"a""#, false),
    ("/v1/completions", r#""prompt":"a""#, true),
    (
        "/v1/chat/completions",
        r#""messages":[{"role":"user","content":"a"}]"#,
        false,
    ),
    (
        "/v1/chat/completions",
        r#""messages":[{"role":"user","content":"a"}]"#,
        true,
    ),
];

fn generation_body(fields: &str, max_tokens: Option<&str>, stream: bool) -> String {
    let max_tokens = max_tokens
        .map(|value| format!(r#","max_tokens":{value}"#))
        .unwrap_or_default();
    format!("{{{fields}{max_tokens},\"stream\":{stream}}}")
}

/// `max_tokens` is bounded once, at parse time, before any engine sizes an allocation from it. A job
/// that reaches an engine carries the clamped value on every generation route, streaming and not: a
/// value far above the cap and `usize::MAX` both arrive as `MAX_NEW_TOKENS_CAP`, a small value arrives
/// untouched, and the chat route's `max_completion_tokens` wins over `max_tokens`.
#[test]
fn an_oversized_max_tokens_reaches_the_engine_clamped_on_every_generation_route() {
    let model = tiny_model();
    let cases = [
        (usize::MAX.to_string(), MAX_NEW_TOKENS_CAP),
        ("999999999999".to_string(), MAX_NEW_TOKENS_CAP),
        ("5".to_string(), 5),
    ];
    let metrics = Arc::new(test_metrics("max-tokens-clamp"));
    for (path, fields, stream) in GENERATION_ROUTES {
        for (max_tokens, want) in &cases {
            let label = format!("{path} stream={stream} max_tokens={max_tokens}");
            let (served, queue) = served_over(Arc::clone(&model), 1);
            let body = generation_body(fields, Some(max_tokens), stream);
            let submitted = std::thread::scope(|scope| {
                let engine = scope.spawn(move || {
                    let EngineMsg::Generate(job) = queue
                        .recv_timeout(Duration::from_secs(10))
                        .expect("the job must reach the engine queue")
                    else {
                        panic!("a generation route sends a generation job");
                    };
                    let max_new = job.max_new;
                    job.reply
                        .send(GenEvent::Done(Completion {
                            prompt_tokens: 1,
                            tokens: vec![1, 2, 3],
                            text: "abc".into(),
                            logprobs: vec![],
                            finish: FinishReason::Eos,
                        }))
                        .expect("answer the job");
                    max_new
                });
                let response = roundtrip(&served, &metrics, "POST", path, &body);
                assert_eq!(
                    status_line(&response),
                    "HTTP/1.1 200 OK",
                    "{label}: {response}"
                );
                engine.join().unwrap()
            });
            assert_eq!(submitted, *want, "{label}");
        }
    }
    // `max_completion_tokens` supersedes `max_tokens` on chat and is bounded the same way.
    let (served, queue) = served_over(model, 1);
    let body = r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":7,"max_completion_tokens":999999999999}"#;
    let submitted = std::thread::scope(|scope| {
        let engine = scope.spawn(move || {
            let EngineMsg::Generate(job) = queue
                .recv_timeout(Duration::from_secs(10))
                .expect("the job must reach the engine queue")
            else {
                panic!("a chat route sends a generation job");
            };
            let max_new = job.max_new;
            let _ = job.reply.send(GenEvent::Done(Completion {
                prompt_tokens: 1,
                tokens: vec![1],
                text: "a".into(),
                logprobs: vec![],
                finish: FinishReason::Eos,
            }));
            max_new
        });
        roundtrip(&served, &metrics, "POST", "/v1/chat/completions", body);
        engine.join().unwrap()
    });
    assert_eq!(submitted, MAX_NEW_TOKENS_CAP);
}

/// `/health` derives `engine_loaded` from the engine's state, not from whether a job queue exists:
/// `Starting` and `Dead` have no engine thread, `Serving` and `Draining` do.
#[test]
fn health_reports_whether_an_engine_is_loaded() {
    use crate::engine_plan::Death;
    let draining = EngineHealth::serving_for_test("spirv-vulkan", 4, 1024);
    draining.begin_draining();
    let dead = EngineHealth::serving_for_test("spirv-vulkan", 4, 1024);
    dead.die(Death::Fault("device lost".into()));
    for (health, status_line, status, loaded) in [
        (
            EngineHealth::starting(),
            "HTTP/1.1 503 Service Unavailable",
            "starting",
            false,
        ),
        (
            EngineHealth::serving_for_test("spirv-vulkan", 4, 1024),
            "HTTP/1.1 200 OK",
            "serving",
            true,
        ),
        (
            draining,
            "HTTP/1.1 503 Service Unavailable",
            "draining",
            true,
        ),
        (dead, "HTTP/1.1 503 Service Unavailable", "dead", false),
    ] {
        let (line, body) = super::engine_loop::health_over_http(&health);
        assert_eq!(line, status_line, "{status}: {body}");
        assert_eq!(body["status"], status, "{body}");
        assert_eq!(body["engine_loaded"], loaded, "{status}: {body}");
    }
}

/// A job submitted to an engine queue whose receiver is gone is `GenerationError::EngineUnavailable`
/// (a 503 the response side counts), not a hang and not a completion. An encode job and an
/// administration job are the same.
#[test]
fn a_job_for_a_gone_engine_is_engine_unavailable() {
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    drop(rx);
    let metrics = test_metrics("gone-engine");
    let error = submit_one(
        &tx,
        &metrics,
        generation_request("a", 1, false),
        &|| false,
        |_, _| std::ops::ControlFlow::Continue(()),
    )
    .expect_err("a closed engine queue accepts no job");
    assert_eq!(error, GenerationError::EngineUnavailable);
    assert_eq!(error.stream_status(), 503);
    assert!(error.needs_response_error_metric());
    let error = crate::lifecycle::submit_encode(
        &tx,
        &metrics,
        vec![EncodeRow {
            ids: vec![1],
            types: Vec::new(),
        }],
        &|| false,
    )
    .expect_err("a closed engine queue accepts no encode job");
    assert_eq!(error, GenerationError::EngineUnavailable);
    let error = crate::lifecycle::submit_admin(&tx, AdminRequest::List)
        .expect_err("a closed engine queue accepts no administration job");
    assert_eq!(error, crate::types::AdminError::EngineUnavailable);
    assert_eq!(error.status(), 503);
}

/// The loop over a real `Driver` on a device (wgpu and ROCm lanes): the rows the recording fake cannot
/// answer, because what they check is what the driver does with the engine's calls. Each row starts the
/// engine thread the way the server does, builds its own executor there (an executor is not `Send`), and
/// talks to it only through the queue, `submit`, `submit_encode` and `submit_admin`, as a connection
/// thread does. The checkpoint is the registered qwen2 fixture written as an HF directory.
mod device {
    use std::num::{NonZeroU64, NonZeroUsize};

    use poot_executor::Executor;
    use poot_graph_plan::{CompileLimits, CompileOptions};
    use poot_llm::driver::{
        AdapterRef, BackendChoice, DraftPolicy, Driver, DriverOptions, EncodeRequest,
        EncodeRow as DriverRow, GenerateRequest, PreparedSetLimits, open_executor,
        program_retention,
    };
    use poot_runtime_common::DeviceBackend;
    use poot_test_util::per_backend;

    use super::*;
    use crate::batch::engine::{DriverEngine, run_engine_thread};
    use crate::engine_plan::{
        EngineArgs, EnginePlan, Opened, Started, StartupError, device_budget,
    };
    use crate::lifecycle::{submit_admin, submit_encode};
    use crate::types::{AdapterInfo, AdminError, AdminReply};

    /// A prompt inside the fixture's 48-token vocabulary and 64 positions.
    const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];
    const MAX_NEW: usize = 8;

    /// The engine thread and the queue that feeds it. Dropping it closes the queue and joins the thread,
    /// so a row that fails an assertion does not leave the device in use at process exit.
    struct Engine {
        tx: Option<mpsc::Sender<EngineMsg>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl Engine {
        fn new(tx: mpsc::Sender<EngineMsg>, thread: std::thread::JoinHandle<()>) -> Self {
            Self {
                tx: Some(tx),
                thread: Some(thread),
            }
        }

        fn queue(&self) -> &mpsc::Sender<EngineMsg> {
            self.tx
                .as_ref()
                .expect("the queue is open until the engine is dropped")
        }
    }

    impl Drop for Engine {
        fn drop(&mut self) {
            drop(self.tx.take());
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero")
    }

    fn backend_of(choice: BackendChoice) -> DeviceBackend {
        match choice {
            BackendChoice::Wgpu => DeviceBackend::Wgpu,
            BackendChoice::Rocm => DeviceBackend::Rocm,
            BackendChoice::Ptx => DeviceBackend::Ptx,
            other => panic!("poot-serve has no device row on {other:?}"),
        }
    }

    /// The device, or `None` after printing the skip (a required device panics instead).
    fn open(choice: BackendChoice) -> Option<Box<dyn Executor>> {
        poot_runtime_common::device_skip::open_or_skip(backend_of(choice), open_executor(choice))
    }

    fn options(chunk: usize, capacity: usize) -> DriverOptions {
        let compile = CompileOptions::new(CompileLimits::STANDARD);
        DriverOptions {
            prefill: compile.clone(),
            decode: compile,
            capacity: nz(capacity),
            prefill_chunk: nz(chunk),
            trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(64)),
            prepared: PreparedSetLimits {
                max_entries: nz(64),
                max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
            },
            charge: program_retention,
            draft: DraftPolicy::Off,
            hidden_window: None,
        }
    }

    /// The registered qwen2 fixture as an HF checkpoint directory (config, a word-level tokenizer over its
    /// 48-token vocabulary, safetensors), loaded the way the server loads one.
    fn qwen2_dir() -> (std::path::PathBuf, poot_models::registry::Fixture) {
        let registry = Registry::builtin().expect("shipped families register disjoint keys");
        let entry = registry
            .entries()
            .iter()
            .find(|entry| entry.family.as_str() == "qwen2")
            .expect("the qwen2 family is registered");
        let fixture = poot_models::test_support::family_fixtures::base(entry.family);
        let vocab: String = (0..48)
            .map(|i| format!("\"w{i}\":{i}"))
            .collect::<Vec<_>>()
            .join(",");
        let tokenizer = format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"WhitespaceSplit"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"w0"}}}}"#
        );
        let dir =
            std::env::temp_dir().join(format!("poot_serve_qwen2_{}", unique_fixture_suffix()));
        std::fs::create_dir_all(&dir).expect("create the checkpoint directory");
        std::fs::write(dir.join("config.json"), fixture.config.to_string()).expect("write config");
        std::fs::write(dir.join("tokenizer.json"), tokenizer).expect("write tokenizer");
        poot_quant::test_support::safetensors::write(
            &dir.join("model.safetensors"),
            &fixture.store,
        );
        (dir, fixture)
    }

    fn qwen2() -> (Arc<ModelHandle>, poot_models::registry::Fixture) {
        let (dir, fixture) = qwen2_dir();
        let handle = load_handle(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        (handle, fixture)
    }

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn greedy_request(tokens: Vec<u32>, adapter: AdapterRef) -> GenerationRequest {
        GenerationRequest {
            prompt: PromptInput::text(tokens),
            max_new: MAX_NEW,
            sampler: Sampler::greedy(),
            stop: Vec::new(),
            stream: false,
            lora_adapter: adapter,
        }
    }

    fn run(
        tx: &mpsc::Sender<EngineMsg>,
        metrics: &Metrics,
        request: GenerationRequest,
    ) -> Completion {
        submit_one(tx, metrics, request, &|| false, |_, _| {
            GenerationControl::Continue(())
        })
        .expect("the request completes")
    }

    /// SC-007: a multimodal prompt served through the loop returns the tokens `Driver::generate` gives the
    /// same prompt with the same embedding-row request input, streamed piece by piece, and a plain prompt after it
    /// returns the plain tokens: the rows belong to their job. The embedding rows replace the placeholder
    /// tokens' own, so the run differs from the tokens-only run (the row's precondition). Mutation: open the
    /// sequence without the rows (`embeds: None` in `admit`); the tokens are the placeholders' and the row
    /// goes red. The layout is the contiguous one, the only one that takes embedding rows.
    fn a_multimodal_job_through_the_loop_matches_generate_with_inputs(choice: BackendChoice) {
        let (handle, _) = qwen2();
        let placeholders = vec![3, 0, 40, 0, 25, 11, 2];
        let embeds = embed_inputs(
            64,
            vec![1, 3],
            fill(2 * 64, 7).iter().map(|v| v * 0.6).collect(),
        );
        let (tx, rx) = mpsc::channel::<EngineMsg>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let metrics = Arc::new(test_metrics("vlm-device"));
        let thread = {
            let (handle, metrics, embeds) =
                (Arc::clone(&handle), Arc::clone(&metrics), embeds.clone());
            let placeholders = placeholders.clone();
            std::thread::spawn(move || {
                let Some(executor) = open(choice) else {
                    let _ = ready_tx.send(None);
                    return;
                };
                let mut driver = Driver::new(Arc::clone(&handle), executor, options(8, 64))
                    .expect("driver")
                    .serve(Layout::Contiguous);
                let mut sink = |_: u32, _: &str| GenerationControl::Continue(());
                let request = |prompt: &[u32]| GenerateRequest {
                    prompt: prompt.to_vec(),
                    max_new: MAX_NEW,
                    sampler: Sampler::greedy(),
                    stops: Vec::new(),
                    ignore_eos: false,
                };
                let with = driver
                    .generate_with_inputs(request(&placeholders), embeds, &mut sink)
                    .expect("generate with inputs")
                    .tokens;
                let without = driver
                    .generate(request(&placeholders), &mut sink)
                    .expect("generate without embeds")
                    .tokens;
                let view = EngineView::new(driver.caps().expect("prepared by generate").clone());
                ready_tx
                    .send(Some((with, without)))
                    .expect("report the oracle");
                let health = EngineHealth::serving_for_test("device", 1, 64);
                run_engine_thread(
                    DriverEngine::new(driver, handle, view),
                    rx,
                    &metrics,
                    &health,
                );
            })
        };
        let engine = Engine::new(tx, thread);
        let tx = engine.queue();
        let Some((with, without)) = ready_rx.recv().expect("the engine thread starts") else {
            return;
        };
        assert_ne!(
            with, without,
            "the rows must change the run, or the row proves nothing"
        );

        let pieces = Mutex::new(String::new());
        let mut request = greedy_request(Vec::new(), AdapterRef::Base);
        request.prompt = PromptInput {
            tokens: placeholders.clone(),
            inputs: embeds,
        };
        request.stream = true;
        let completion = submit_one(tx, &metrics, request, &|| false, |piece, _| {
            pieces.lock().unwrap().push_str(piece);
            GenerationControl::Continue(())
        })
        .expect("the multimodal job completes");
        assert_eq!(
            completion.tokens, with,
            "the loop's tokens are generate_with_inputs'"
        );
        assert_eq!(
            *pieces.lock().unwrap(),
            completion.text,
            "the streamed pieces are the text"
        );
        let plain = run(tx, &metrics, greedy_request(placeholders, AdapterRef::Base));
        assert_eq!(
            plain.tokens, without,
            "the rows belonged to the first job only"
        );
        drop(engine);
    }

    /// A BERT-class encoder checkpoint starts through `EnginePlan::build` (its encode shapes fitted and
    /// prepared before `Serving`) and serves on the device (POOT-1204 SC-003). `Encode` jobs on that
    /// engine come back as a fresh driver's `Driver::encode` returns them, element for element, for
    /// batches that differ in rows and padded length; the caps report the encode shapes; a
    /// `/v1/embeddings` request through the production `http::handle` returns the pooled, unit-length
    /// vector of the same hidden states; and nothing compiles after `Serving`. Mutation: restore the
    /// startup refusal of encoders in `EnginePlan::build`; the engine does not start.
    fn an_encoder_checkpoint_serves_embeddings_on_the_device(choice: BackendChoice) {
        let handle = crate::tests::encoder_routes::encoder_handle(None);
        encode_jobs_match_a_fresh_driver(choice, handle, 16, Some("w3 w17 w40"));
    }

    /// `Encode` jobs on the engine the server builds for a decoder (the planned paged pool plus its
    /// fitted encode shapes, started through `EnginePlan::build`), after the engine served a request:
    /// each batch comes back as a fresh driver's `Driver::encode` returns it, element for element, the
    /// published caps report the encode shapes before any request, the step runs with no sequence open
    /// on a driver that has served, nothing compiles after `Serving`, and the engine serves again
    /// afterwards. Mutation: prepare encode entries lazily (skip them in the plan): the caps assertion
    /// and the compile count go red.
    fn an_encode_job_on_the_serving_driver_matches_a_fresh_drivers_encode(choice: BackendChoice) {
        let (handle, _) = qwen2();
        encode_jobs_match_a_fresh_driver(choice, handle, 64, None);
    }

    /// The shared body: `positions` is the model's context. With `text` (an encoder), also a
    /// `/v1/embeddings` request; without it (a decoder), generation requests around the encodes.
    fn encode_jobs_match_a_fresh_driver(
        choice: BackendChoice,
        handle: Arc<ModelHandle>,
        positions: usize,
        text: Option<&str>,
    ) {
        let batch = |rows: &[&[u32]]| -> Vec<EncodeRow> {
            rows.iter()
                .map(|ids| EncodeRow {
                    ids: ids.to_vec(),
                    types: Vec::new(),
                })
                .collect()
        };
        let batches = vec![
            batch(&[&[3, 17, 40], &[8, 25, 11, 2, 9]]),
            batch(&[&[3], &[8, 25, 11, 2, 9, 4, 6, 1, 5, 7], &[40, 12]]),
            batch(&[&[3, 17, 40], &[8, 25, 11, 2, 9]]),
        ];
        let rows = if text.is_some() {
            crate::config::DEFAULT_ENCODE_ROWS
        } else {
            crate::config::DEFAULT_ROWS
        };
        let text = text.map(str::to_owned);
        let (tx, rx) = mpsc::channel::<EngineMsg>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (compiles_tx, compiles_rx) = mpsc::channel();
        let metrics = Arc::new(test_metrics("encode-device"));
        let thread = {
            let (handle, metrics, batches, text) = (
                Arc::clone(&handle),
                Arc::clone(&metrics),
                batches.clone(),
                text.clone(),
            );
            std::thread::spawn(move || {
                // Every executor opens before any runs (the ROCm runtime does not survive a second open
                // after a drop).
                let (Some(oracle_executor), Some(served_executor)) = (open(choice), open(choice))
                else {
                    let _ = ready_tx.send(None);
                    return;
                };
                let args = EngineArgs::new(choice, DraftPolicy::Off);
                let mut fresh =
                    Driver::new(Arc::clone(&handle), oracle_executor, options(8, positions))
                        .expect("driver");
                fresh
                    .prepare_encode(&poot_llm::driver::EncodeShapes {
                        rows,
                        max_len: NonZeroUsize::new(positions).unwrap(),
                    })
                    .expect("the fresh driver prepares the same shapes");
                let encode = |fresh: &mut Driver, rows: &[EncodeRow]| {
                    let inputs: Vec<_> = rows
                        .iter()
                        .map(|row| poot_llm::encoders::segment_inputs(&row.types))
                        .collect();
                    let driver_rows: Vec<DriverRow<'_>> = rows
                        .iter()
                        .zip(&inputs)
                        .map(|(row, inputs)| DriverRow {
                            tokens: Some(&row.ids),
                            inputs,
                        })
                        .collect();
                    fresh
                        .encode(&EncodeRequest { rows: &driver_rows })
                        .expect("the fresh driver encodes")
                };
                let want: Vec<_> = batches
                    .iter()
                    .map(|rows| encode(&mut fresh, rows))
                    .collect();
                let sentence = text.as_deref().map(|text| {
                    let tokens = handle
                        .text()
                        .encode_for_encoder(text, None)
                        .expect("the fixture tokenizes the text");
                    encode(
                        &mut fresh,
                        &[EncodeRow {
                            ids: tokens.ids,
                            types: tokens.types,
                        }],
                    )
                });
                let health = EngineHealth::starting();
                let mut executor = Some(served_executor);
                let started = EnginePlan::build(
                    &args,
                    &handle,
                    |_| Ok(Opened::One(executor.take().expect("opened once"))),
                    device_budget,
                    &health,
                );
                let Started { plan, driver } = match started {
                    Ok(started) => started,
                    Err(StartupError::Device { .. }) => {
                        let _ = ready_tx.send(None);
                        return;
                    }
                    Err(error) => panic!("the engine starts: {error}"),
                };
                let compiles = driver.stats().compiles;
                let view = EngineView::new(plan.caps.clone());
                ready_tx
                    .send(Some((want, sentence, view.clone(), compiles)))
                    .expect("report the oracle");
                let engine = run_engine_thread(
                    DriverEngine::new(driver, handle, view),
                    rx,
                    &metrics,
                    &health,
                );
                compiles_tx
                    .send((compiles, engine.compiles()))
                    .expect("the compile counts reach the test");
            })
        };
        let engine = Engine::new(tx, thread);
        let tx = engine.queue();
        let Some((want, sentence, view, _)) = ready_rx.recv().expect("the engine thread starts")
        else {
            return;
        };
        let fitted = view
            .caps()
            .encode
            .expect("the caps carry the encode shapes");
        assert_eq!(
            (fitted.rows.get(), fitted.lengths.last().map(|l| l.get())),
            (rows.get(), Some(positions)),
            "the caps carry the shapes before any request"
        );
        assert!(view.caps().inputs.encode, "encode entries are prepared");
        let generates = |tx: &mpsc::Sender<EngineMsg>| {
            let done = run(
                tx,
                &metrics,
                greedy_request(PROMPT.to_vec(), AdapterRef::Base),
            );
            assert!(!done.tokens.is_empty());
        };
        if text.is_none() {
            generates(tx);
        }
        for (index, (rows, want)) in batches.into_iter().zip(&want).enumerate() {
            let got =
                submit_encode(tx, &metrics, rows, &|| false).expect("the encode job completes");
            assert_eq!(got.width, want.width, "batch {index}");
            assert_eq!(got.rows.len(), want.rows.len(), "batch {index}");
            for (row, (got, want)) in got.rows.iter().zip(&want.rows).enumerate() {
                assert_eq!(got.len(), want.len(), "batch {index} row {row}");
                for (position, (g, w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        g.is_finite(),
                        "batch {index} row {row} value {position}: {g}"
                    );
                    assert_eq!(g, w, "batch {index} row {row} value {position}");
                }
            }
        }
        if let (Some(text), Some(sentence)) = (&text, sentence) {
            // The route: /v1/embeddings through the production handler returns the pooled sentence vector.
            let served = Served {
                model: Arc::clone(&handle),
                engine: tx.clone(),
                view,
                pooling: poot_llm::encoders::Pooling::Mean,
                images: None,
                max_queue_wait: crate::tests::TEST_QUEUE_WAIT,
            };
            let response = roundtrip(
                &served,
                &metrics,
                "POST",
                "/v1/embeddings",
                &format!(r#"{{"input":"{text}"}}"#),
            );
            assert_eq!(status_line(&response), "HTTP/1.1 200 OK", "{response}");
            let embedding: Vec<f32> = json_body(&response)["data"][0]["embedding"]
                .as_array()
                .expect("an embedding")
                .iter()
                .map(|v| v.as_f64().expect("a number") as f32)
                .collect();
            let expected = poot_llm::encoders::pool(
                &sentence.rows[0],
                sentence.width,
                poot_llm::encoders::Pooling::Mean,
            );
            assert_eq!(
                embedding, expected,
                "the route's vector is the pooled hidden states"
            );
            drop(served);
        } else {
            // The engine keeps serving after the encodes.
            generates(tx);
        }
        drop(engine);
        let (at_serving, at_end) = compiles_rx.recv().expect("the engine thread ended cleanly");
        assert_eq!(at_end, at_serving, "nothing compiles after Serving");
    }

    /// A PEFT adapter directory over every projection of every layer of the fixture, rank 2, `alpha` scaled:
    /// `adapter_config.json` and `adapter_model.safetensors` with `lora_A` `[r, in]` and `lora_B` `[out, r]`.
    fn write_adapter(
        fixture: &poot_models::registry::Fixture,
        seed: u64,
        alpha: f32,
    ) -> std::path::PathBuf {
        use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
        use poot_tensor::DType;
        const RANK: usize = 2;
        let mut adapter = WeightStore::builder();
        let mut targets = std::collections::BTreeSet::new();
        for (index, (key, entry)) in fixture.store.iter().enumerate() {
            let name = key.as_str();
            let Some(module) = name.strip_suffix(".weight") else {
                continue;
            };
            if !module.starts_with("model.layers.") || !module.ends_with("_proj") {
                continue;
            }
            let WeightEntry::Dense(dense) = entry else {
                continue;
            };
            let &[out_dim, in_dim] = dense.shape() else {
                continue;
            };
            targets.insert(
                module
                    .rsplit('.')
                    .next()
                    .expect("a module name")
                    .to_string(),
            );
            let f32_entry = |shape: Vec<usize>, values: Vec<f32>| {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::F32, shape, bytes.into()).expect("dense"),
                )
            };
            let s = seed + index as u64;
            adapter
                .insert(
                    format!("base_model.model.{module}.lora_A.weight"),
                    f32_entry(vec![RANK, in_dim], fill(RANK * in_dim, s)),
                )
                .expect("unique key");
            adapter
                .insert(
                    format!("base_model.model.{module}.lora_B.weight"),
                    f32_entry(vec![out_dim, RANK], fill(out_dim * RANK, s + 1000)),
                )
                .expect("unique key");
        }
        let dir =
            std::env::temp_dir().join(format!("poot_serve_adapter_{}", unique_fixture_suffix()));
        std::fs::create_dir_all(&dir).expect("create the adapter directory");
        let config = serde_json::json!({
            "r": RANK, "lora_alpha": alpha, "target_modules": targets, "use_rslora": false,
        });
        std::fs::write(dir.join("adapter_config.json"), config.to_string()).expect("write config");
        poot_quant::test_support::safetensors::write(
            &dir.join("adapter_model.safetensors"),
            &adapter.build(),
        );
        dir
    }

    /// LoRA administration on the engine the server builds: a hot-loaded adapter serves requests that name
    /// it and changes the run, the base requests around it are unaffected, a second adapter loads beside the
    /// first and the first keeps its run, unloading shifts the indices and frees the generation, and a
    /// request holding an adapter that was unloaded is a typed refusal. The view the connection threads read
    /// is what each step leaves.
    fn a_hot_loaded_adapter_serves_requests_and_unload_frees_it(choice: BackendChoice) {
        let (handle, fixture) = qwen2();
        let style = write_adapter(&fixture, 11, 4.0);
        let other = write_adapter(&fixture, 77, 6.0);
        let (tx, rx) = mpsc::channel::<EngineMsg>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let metrics = Arc::new(test_metrics("lora-device"));
        let thread = {
            let (handle, metrics) = (Arc::clone(&handle), Arc::clone(&metrics));
            std::thread::spawn(move || {
                let Some(executor) = open(choice) else {
                    let _ = ready_tx.send(None);
                    return;
                };
                let health = EngineHealth::starting();
                let mut executor = Some(executor);
                let started = EnginePlan::build(
                    &EngineArgs::new(choice, DraftPolicy::Off),
                    &handle,
                    |_| Ok(Opened::One(executor.take().expect("opened once"))),
                    device_budget,
                    &health,
                );
                let Started { plan, driver } = match started {
                    Ok(started) => started,
                    Err(StartupError::Device { .. }) => {
                        let _ = ready_tx.send(None);
                        return;
                    }
                    Err(error) => panic!("the engine starts: {error}"),
                };
                let view = EngineView::new(plan.caps.clone());
                ready_tx.send(Some(view.clone())).expect("report the view");
                run_engine_thread(
                    DriverEngine::new(driver, handle, view),
                    rx,
                    &metrics,
                    &health,
                );
            })
        };
        let engine = Engine::new(tx, thread);
        let tx = engine.queue();
        let Some(view) = ready_rx.recv().expect("the engine thread starts") else {
            return;
        };
        let path = |dir: &std::path::Path| dir.to_string_lossy().into_owned();
        let load = |name: &str, dir: &std::path::Path| {
            submit_admin(
                tx,
                AdminRequest::Load {
                    name: name.into(),
                    path: path(dir),
                },
            )
        };
        let tokens_on = |name: &str| {
            let adapter = view
                .adapter(name)
                .unwrap_or_else(|| panic!("{name} is loaded"));
            run(tx, &metrics, greedy_request(PROMPT.to_vec(), adapter)).tokens
        };

        let base = run(
            tx,
            &metrics,
            greedy_request(PROMPT.to_vec(), AdapterRef::Base),
        )
        .tokens;
        assert!(
            view.caps().lora.is_none(),
            "no adapter before the first load"
        );
        assert_eq!(
            submit_admin(tx, AdminRequest::List),
            Ok(AdminReply::Adapters(Vec::new()))
        );

        let loaded = load("style", &style).expect("the adapter loads");
        assert_eq!(
            loaded,
            AdminReply::Loaded(AdapterInfo {
                name: "style".into(),
                index: 1
            })
        );
        assert!(
            view.caps().lora.is_some(),
            "the driver now serves an adapter generation"
        );
        let stale = view.adapter("style").expect("the view names the adapter");
        let styled = tokens_on("style");
        assert_ne!(
            styled, base,
            "the adapter must change the run, or the row proves nothing"
        );
        assert_eq!(
            run(
                tx,
                &metrics,
                greedy_request(PROMPT.to_vec(), AdapterRef::Base)
            )
            .tokens,
            base,
            "a base request is unaffected by a loaded adapter"
        );
        assert!(
            matches!(
                load("style", &style),
                Err(AdminError::Refused {
                    class: FailureClass::Request,
                    ..
                })
            ),
            "a name loads once"
        );

        // A second adapter loads beside the first; the set is rebuilt, the first keeps index 1 and its run.
        let loaded = load("other", &other).expect("the second adapter loads");
        assert_eq!(
            loaded,
            AdminReply::Loaded(AdapterInfo {
                name: "other".into(),
                index: 2
            })
        );
        assert_eq!(
            tokens_on("style"),
            styled,
            "the first adapter's run is unchanged by the second"
        );
        let others = tokens_on("other");
        assert_ne!(others, styled);
        assert_ne!(others, base);

        // Unloading the first shifts the second to index 1 and it keeps its run.
        assert_eq!(
            submit_admin(
                tx,
                AdminRequest::Unload {
                    name: "style".into()
                }
            ),
            Ok(AdminReply::Unloaded(AdapterInfo {
                name: "style".into(),
                index: 1
            }))
        );
        assert!(view.adapter("style").is_none());
        assert!(matches!(
            view.adapter("other"),
            Some(AdapterRef::Adapter { index, .. }) if index.get() == 1
        ));
        assert_eq!(
            tokens_on("other"),
            others,
            "the survivor keeps its run at its new index"
        );
        assert_eq!(
            submit_admin(
                tx,
                AdminRequest::Unload {
                    name: "style".into()
                }
            ),
            Err(AdminError::NotFound("style".into()))
        );

        // A request that holds the unloaded generation is refused, not served the base weights.
        let refused = submit_one(
            tx,
            &metrics,
            greedy_request(PROMPT.to_vec(), stale),
            &|| false,
            |_, _| GenerationControl::Continue(()),
        )
        .expect_err("the unloaded adapter's generation is gone");
        let GenerationError::EngineFailed(failure) = refused else {
            panic!("expected the engine's refusal, got {refused:?}");
        };
        assert_eq!(failure.class(), FailureClass::Request, "{failure}");

        assert!(
            submit_admin(
                tx,
                AdminRequest::Unload {
                    name: "other".into()
                }
            )
            .is_ok()
        );
        assert!(view.caps().lora.is_none(), "every generation is freed");
        assert_eq!(
            run(
                tx,
                &metrics,
                greedy_request(PROMPT.to_vec(), AdapterRef::Base)
            )
            .tokens,
            base,
            "the base still serves its own run"
        );
        drop(engine);
        let _ = std::fs::remove_dir_all(style);
        let _ = std::fs::remove_dir_all(other);
    }

    per_backend!(
        BackendChoice,
        a_multimodal_job_through_the_loop_matches_generate_with_inputs,
        a_multimodal_job_through_the_loop_matches_generate_with_inputs_wgpu,
        a_multimodal_job_through_the_loop_matches_generate_with_inputs_rocm,
        a_multimodal_job_through_the_loop_matches_generate_with_inputs_ptx
    );
    per_backend!(
        BackendChoice,
        an_encoder_checkpoint_serves_embeddings_on_the_device,
        an_encoder_checkpoint_serves_embeddings_on_the_device_wgpu,
        an_encoder_checkpoint_serves_embeddings_on_the_device_rocm,
        an_encoder_checkpoint_serves_embeddings_on_the_device_ptx
    );
    per_backend!(
        BackendChoice,
        an_encode_job_on_the_serving_driver_matches_a_fresh_drivers_encode,
        an_encode_job_on_the_serving_driver_matches_a_fresh_drivers_encode_wgpu,
        an_encode_job_on_the_serving_driver_matches_a_fresh_drivers_encode_rocm,
        an_encode_job_on_the_serving_driver_matches_a_fresh_drivers_encode_ptx
    );
    per_backend!(
        BackendChoice,
        a_hot_loaded_adapter_serves_requests_and_unload_frees_it,
        a_hot_loaded_adapter_serves_requests_and_unload_frees_it_wgpu,
        a_hot_loaded_adapter_serves_requests_and_unload_frees_it_rocm,
        a_hot_loaded_adapter_serves_requests_and_unload_frees_it_ptx
    );
}

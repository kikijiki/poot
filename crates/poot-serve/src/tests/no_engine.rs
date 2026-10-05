//! The server without a generation engine (card 597). Every row drives the production `http::handle`
//! over a loopback socket. `engine: None` is the production state; a bare job channel stands in for the
//! serving loop where a row needs a job to reach an engine.

use std::net::IpAddr;

use super::*;

/// The four generation requests: both routes, streaming and not. `max_tokens` is spliced in as raw
/// JSON so a row can send a value no Rust integer type holds.
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

fn tiny_decoder_backend() -> Backend {
    let model_path = write_tiny_granitemoe_gguf();
    let runner = Runner::load_gguf(&model_path).expect("load no-engine fixture runner");
    let _ = std::fs::remove_file(&model_path);
    Backend::Decoder(Arc::new(runner))
}

fn generation_body(fields: &str, max_tokens: Option<&str>, stream: bool) -> String {
    let max_tokens = max_tokens
        .map(|value| format!(r#","max_tokens":{value}"#))
        .unwrap_or_default();
    format!("{{{fields}{max_tokens},\"stream\":{stream}}}")
}

/// Send one request through `handle` and return the full response text. `engine_side` runs on its own
/// thread beside the handler and owns the receiving end of the job channel, when a row gave one.
fn roundtrip(
    backend: &Backend,
    engine: Option<&mpsc::Sender<Job>>,
    method: &str,
    path: &str,
    body: &str,
    engine_side: impl FnOnce() + Send,
) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind no-engine socket");
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect client");
    client
        .set_read_timeout(Some(Duration::from_secs(20)))
        .expect("bound the response read");
    let (server, _) = listener.accept().expect("accept client");
    let metrics = test_metrics("no-engine");
    let limiter = RateLimiter {
        rpm: 0,
        windows: Mutex::new(HashMap::new()),
    };
    let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
        .expect("loopback lora policy");
    write!(
        client,
        "{method} {path} HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .expect("write request");
    client.flush().expect("flush request");
    std::thread::scope(|scope| {
        let handler = scope.spawn(|| handle(server, engine, backend, &metrics, &limiter, &policy));
        scope.spawn(engine_side);
        handler
            .join()
            .expect("join handler")
            .expect("handler answers without a transport error");
    });
    let mut response = String::new();
    client
        .read_to_string(&mut response)
        .expect("read the response to EOF");
    response
}

fn status_line(response: &str) -> &str {
    response.lines().next().unwrap_or_default()
}

fn json_body(response: &str) -> serde_json::Value {
    let (_, body) = response
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no header/body split in {response:?}"));
    serde_json::from_str(body)
        .unwrap_or_else(|error| panic!("body is not JSON ({error}): {body:?}"))
}

/// SC-001: a valid generation request with no engine is a typed 503 that names the missing engine,
/// for both routes, streaming and not. It is never a 200, never an event stream, and never carries a
/// `finish_reason` (the failure class of R477-002: a completion fabricated for a request nothing ran).
#[test]
fn generation_without_an_engine_is_a_typed_503_naming_the_missing_engine() {
    let backend = tiny_decoder_backend();
    for (path, fields, stream) in GENERATION_ROUTES {
        let label = format!("{path} stream={stream}");
        let body = generation_body(fields, Some("3"), stream);
        let response = roundtrip(&backend, None, "POST", path, &body, || {});
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 503 Service Unavailable",
            "{label}: {response}"
        );
        assert!(
            !response.contains("finish_reason") && !response.contains("text/event-stream"),
            "{label}: a refusal is not a completion and not a stream: {response}"
        );
        let error = &json_body(&response)["error"];
        assert_eq!(error["code"], 503, "{label}");
        assert_eq!(error["type"], "server_error", "{label}");
        assert!(
            error["message"]
                .as_str()
                .is_some_and(|message| message.contains("no generation engine is loaded")),
            "{label}: the message must name the missing engine: {error}"
        );
    }
}

/// SC-001, the counter side: a refusal is not a request that ran, so it starts no generation, counts
/// no completion and leaves no request in flight.
#[test]
fn a_refused_generation_request_counts_no_completion_and_leaves_nothing_in_flight() {
    let backend = tiny_decoder_backend();
    let metrics = test_metrics("no-engine-counters");
    for (path, fields, stream) in GENERATION_ROUTES {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (server, _) = listener.accept().unwrap();
        let limiter = RateLimiter {
            rpm: 0,
            windows: Mutex::new(HashMap::new()),
        };
        let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
            .expect("loopback lora policy");
        let body = generation_body(fields, Some("3"), stream);
        write!(
            client,
            "POST {path} HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        handle(server, None, &backend, &metrics, &limiter, &policy).expect("handled");
        let mut sink = String::new();
        client.read_to_string(&mut sink).unwrap();
    }
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.requests.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.inflight.load(Ordering::Relaxed), 0);
}

/// SC-002: the request is validated before the engine lookup, so a request that is wrong on its face
/// is the 400 it always was, not the 503. Each case names a different validation stage: the JSON
/// parse, `max_tokens` typing, an unknown LoRA adapter, and an unsatisfiable guided-decoding regex.
#[test]
fn a_request_invalid_on_its_face_is_a_400_before_the_engine_lookup() {
    let backend = tiny_decoder_backend();
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
        ];
        for (case, body) in cases {
            let response = roundtrip(&backend, None, "POST", path, &body, || {});
            assert_eq!(
                status_line(&response),
                "HTTP/1.1 400 Bad Request",
                "{label}: {case}: validation must precede the engine lookup: {response}"
            );
        }
    }
}

/// SC-002: `max_tokens` is bounded once, at parse time, before any engine sizes an allocation from it.
/// A job that reaches an engine carries the clamped value on every generation route, streaming and
/// not: a value far above the cap and `usize::MAX` both arrive as `MAX_NEW_TOKENS_CAP`, a small value
/// arrives untouched, and the chat route's `max_completion_tokens` wins over `max_tokens`.
#[test]
fn an_oversized_max_tokens_reaches_the_engine_clamped_on_every_generation_route() {
    let backend = tiny_decoder_backend();
    let cases = [
        (usize::MAX.to_string(), MAX_NEW_TOKENS_CAP),
        ("999999999999".to_string(), MAX_NEW_TOKENS_CAP),
        ("5".to_string(), 5),
    ];
    for (path, fields, stream) in GENERATION_ROUTES {
        for (max_tokens, want) in &cases {
            let label = format!("{path} stream={stream} max_tokens={max_tokens}");
            let (tx, engine_rx) = mpsc::channel::<Job>();
            let submitted = Mutex::new(None);
            let submitted_ref = &submitted;
            let body = generation_body(fields, Some(max_tokens), stream);
            let response = roundtrip(&backend, Some(&tx), "POST", path, &body, move || {
                let job = engine_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("the job must reach the engine queue");
                *submitted_ref.lock().unwrap() = Some(job.max_new);
                job.reply
                    .send(GenEvent::Done(vec![1, 2, 3], vec![], true))
                    .expect("answer the job");
            });
            assert_eq!(
                status_line(&response),
                "HTTP/1.1 200 OK",
                "{label}: {response}"
            );
            assert_eq!(*submitted.lock().unwrap(), Some(*want), "{label}");
        }
    }
    // `max_completion_tokens` supersedes `max_tokens` on chat and is bounded the same way.
    let (tx, engine_rx) = mpsc::channel::<Job>();
    let submitted = Mutex::new(None);
    let submitted_ref = &submitted;
    let body = r#"{"messages":[{"role":"user","content":"a"}],"max_tokens":7,"max_completion_tokens":999999999999}"#;
    roundtrip(
        &backend,
        Some(&tx),
        "POST",
        "/v1/chat/completions",
        body,
        move || {
            let job = engine_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("the job must reach the engine queue");
            *submitted_ref.lock().unwrap() = Some(job.max_new);
            job.reply
                .send(GenEvent::Done(vec![1], vec![], true))
                .expect("answer the job");
        },
    );
    assert_eq!(*submitted.lock().unwrap(), Some(MAX_NEW_TOKENS_CAP));
}

/// SC-002, no engine: an oversized `max_tokens` is bounded, not fatal. The request that used to size a
/// multi-TB allocation is the same typed 503 as any other valid request.
#[test]
fn an_oversized_max_tokens_without_an_engine_is_the_same_typed_503() {
    let backend = tiny_decoder_backend();
    for (path, fields, stream) in GENERATION_ROUTES {
        let body = generation_body(fields, Some(&usize::MAX.to_string()), stream);
        let response = roundtrip(&backend, None, "POST", path, &body, || {});
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 503 Service Unavailable",
            "{path} stream={stream}: {response}"
        );
    }
}

/// `/health` reports whether an engine is loaded: `false` in production today, `true` once a job
/// queue exists.
#[test]
fn health_reports_whether_an_engine_is_loaded() {
    let backend = tiny_decoder_backend();
    let (tx, _engine_rx) = mpsc::channel::<Job>();
    for (engine, loaded) in [(None, false), (Some(&tx), true)] {
        let response = roundtrip(&backend, engine, "GET", "/health", "", || {});
        assert_eq!(status_line(&response), "HTTP/1.1 200 OK", "{response}");
        let body = json_body(&response);
        assert_eq!(body["status"], "ok");
        assert_eq!(body["engine_loaded"], loaded, "{response}");
    }
}

/// A causal decoder serves no embeddings and no reranking: `/v1/embeddings` and `/v1/rerank` on a
/// decoder backend are a typed 400 naming the encoder models that serve them, never an answer. Mutation:
/// skip the `/v1/rerank` branch so a decoder falls through to the generation routes (a 404), or change
/// the embeddings refusal text; the row goes red on that path.
#[test]
fn a_decoder_refuses_embeddings_and_rerank_with_a_typed_400() {
    let backend = tiny_decoder_backend();
    for (path, body, names) in [
        (
            "/v1/embeddings",
            r#"{"input":"a"}"#,
            "embeddings come from an encoder model",
        ),
        (
            "/v1/rerank",
            r#"{"query":"a","documents":["b"]}"#,
            "reranking comes from an encoder or cross-encoder model",
        ),
    ] {
        let response = roundtrip(&backend, None, "POST", path, body, || {});
        assert_eq!(
            status_line(&response),
            "HTTP/1.1 400 Bad Request",
            "{path}: {response}"
        );
        let error = &json_body(&response)["error"];
        assert_eq!(error["type"], "invalid_request_error", "{path}");
        assert!(
            error["message"].as_str().is_some_and(
                |message| message.contains("causal decoder") && message.contains(names)
            ),
            "{path}: the refusal must name the decoder and the models that serve it: {error}"
        );
    }
}

/// A job submitted to an engine queue whose receiver is gone is `GenerationError::EngineUnavailable`
/// (a 503 the response side counts), not a hang and not a completion. Mutation: map the failed send to
/// `ReplyChannelClosed`; the row goes red.
#[test]
fn a_job_for_a_gone_engine_is_engine_unavailable() {
    let (tx, rx) = mpsc::channel::<Job>();
    drop(rx);
    let error = submit(
        &tx,
        "a",
        1,
        Sampler::greedy(),
        vec![],
        false,
        LoraAdapterLease::none(),
        |_, _| std::ops::ControlFlow::Continue(()),
    )
    .expect_err("a closed engine queue accepts no job");
    assert_eq!(error, GenerationError::EngineUnavailable);
    assert_eq!(error.stream_status(), 503);
    assert!(error.needs_response_error_metric());
}

use super::*;

#[test]
fn histogram_buckets_and_sum() {
    // observe a known set: 3ms (le=5), 40ms (le=50), 600ms (le=1000). count=3, sum=643.
    let h = Histogram::new();
    for ms in [3.0, 40.0, 600.0] {
        h.observe(ms);
    }
    assert_eq!(h.count(), 3);
    assert!((h.sum_ms() - 643.0).abs() < 0.5);
    assert!((h.avg_ms() - 643.0 / 3.0).abs() < 0.5);
    let out = h.prometheus("test_ms", "help");
    // cumulative buckets are monotonic and the +Inf bucket equals the count (SC-005).
    assert!(out.contains("test_ms_bucket{le=\"5\"} 1"));
    assert!(out.contains("test_ms_bucket{le=\"50\"} 2"));
    assert!(out.contains("test_ms_bucket{le=\"1000\"} 3"));
    assert!(out.contains("test_ms_bucket{le=\"+Inf\"} 3"));
    assert!(out.contains("test_ms_count 3"));
}

#[test]
fn histogram_overflow_bucket() {
    // a value above the last bound (10000ms) lands only in +Inf, not in any finite bucket.
    let h = Histogram::new();
    h.observe(20_000.0);
    let out = h.prometheus("t_ms", "h");
    assert!(out.contains("t_ms_bucket{le=\"10000\"} 0"));
    assert!(out.contains("t_ms_bucket{le=\"+Inf\"} 1"));
    assert_eq!(h.count(), 1);
}

#[test]
fn model_object_is_openai_shaped() {
    let v = model_object("poot-model");
    assert_eq!(v["id"], "poot-model");
    assert_eq!(v["object"], "model");
    assert!(v.get("created").is_some() && v.get("owned_by").is_some());
}

#[test]
fn error_body_is_openai_shaped() {
    // OpenAI clients read error.message (a nested object); a flat {"error":"..."} hides it. The code is
    // the HTTP status, the type the OpenAI category.
    let v: serde_json::Value =
        serde_json::from_str(&error_body(400, "bad input", "invalid_request_error")).unwrap();
    assert_eq!(v["error"]["message"], "bad input");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], 400);
}

#[test]
fn chat_max_completion_tokens_supersedes_max_tokens() {
    let eff = |body: &str| {
        serde_json::from_str::<ChatReq>(body)
            .unwrap()
            .effective_max_tokens()
    };
    // OpenAI precedence: max_completion_tokens wins when both are sent.
    assert_eq!(
        eff(r#"{"messages":[],"max_tokens":10,"max_completion_tokens":99}"#),
        99
    );
    // either field alone is honored.
    assert_eq!(eff(r#"{"messages":[],"max_completion_tokens":77}"#), 77);
    assert_eq!(eff(r#"{"messages":[],"max_tokens":33}"#), 33);
    // neither -> the default.
    assert_eq!(eff(r#"{"messages":[]}"#), default_max());
}

#[test]
fn effective_max_tokens_is_clamped() {
    let eff = |body: &str| {
        serde_json::from_str::<ChatReq>(body)
            .unwrap()
            .effective_max_tokens()
    };
    // A normal small value passes through untouched.
    assert_eq!(eff(r#"{"messages":[],"max_tokens":128}"#), 128);
    // An absurd value is clamped to the cap, not passed through to size a KV allocation.
    assert_eq!(
        eff(r#"{"messages":[],"max_tokens":999999999999}"#),
        MAX_NEW_TOKENS_CAP
    );
    // usize::MAX via max_completion_tokens is clamped too (would otherwise overflow cap arithmetic).
    assert_eq!(
        eff(&format!(
            r#"{{"messages":[],"max_completion_tokens":{}}}"#,
            usize::MAX
        )),
        MAX_NEW_TOKENS_CAP
    );
    // At-cap value is preserved.
    assert_eq!(
        eff(&format!(
            r#"{{"messages":[],"max_tokens":{MAX_NEW_TOKENS_CAP}}}"#
        )),
        MAX_NEW_TOKENS_CAP
    );
}

/// Card 498 SC-001: `/v1/completions` bounds `max_tokens` exactly as chat does (R477-001).
#[test]
fn completions_effective_max_tokens_is_clamped() {
    let eff = |max_tokens: &str| {
        serde_json::from_str::<CompletionReq>(&format!(
            r#"{{"prompt":"a","max_tokens":{max_tokens}}}"#
        ))
        .unwrap()
        .effective_max_tokens()
    };
    // A normal small value passes through untouched.
    assert_eq!(eff("128"), 128);
    // An absurd value is clamped to the cap, not passed on to size a KV allocation.
    assert_eq!(eff("999999999999"), MAX_NEW_TOKENS_CAP);
    // usize::MAX would overflow the engine's `prompt_len + max_new + 1` arithmetic.
    assert_eq!(eff(&usize::MAX.to_string()), MAX_NEW_TOKENS_CAP);
    // The default (no `max_tokens`) is below the cap and survives.
    let default: CompletionReq = serde_json::from_str(r#"{"prompt":"a"}"#).unwrap();
    assert_eq!(default.effective_max_tokens(), default_max());
}

/// Card 498 SC-002: the completions handlers hand `submit` the bounded value. The job channel has no
/// engine behind it; the test plays the engine, reads the queued `Job`, and answers `Done`. The tiny
/// dense fixture is only the tokenizer/encode surface the handlers need before they submit.
#[test]
fn completions_handlers_submit_bounded_max_new() {
    use crate::handlers::handle_completion;
    let runner = tiny_model();
    let cases = [
        (usize::MAX.to_string(), MAX_NEW_TOKENS_CAP),
        ("999999999999".to_string(), MAX_NEW_TOKENS_CAP),
        ("5".to_string(), 5),
    ];
    for streaming in [true, false] {
        for (max_tokens, want) in &cases {
            let body =
                format!(r#"{{"prompt":"a","max_tokens":{max_tokens},"stream":{streaming}}}"#);
            let req: CompletionReq = serde_json::from_str(&body).unwrap();
            let (served, engine_rx) = served_over(Arc::clone(&runner), 1);
            let metrics = test_metrics("completions-bound");
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
            let (mut server, _) = listener.accept().unwrap();

            let engine_job = std::thread::scope(|scope| {
                let handler = scope.spawn(|| {
                    if streaming {
                        stream_completion(&mut server, &served, &metrics, &req).map(drop)
                    } else {
                        handle_completion(&served, &metrics, &req, &|| false).map(drop)
                    }
                });
                let EngineMsg::Generate(job) = engine_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("job must reach the bare channel")
                else {
                    panic!("a completions request sends a generation job");
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
                    .unwrap();
                handler
                    .join()
                    .unwrap()
                    .expect("completions handler must succeed");
                max_new
            });
            assert_eq!(
                engine_job, *want,
                "streaming={streaming} max_tokens={max_tokens}: submitted max_new"
            );
            drop(server);
            let mut sink = String::new();
            client.read_to_string(&mut sink).unwrap();
        }
    }
}

#[test]
fn cache_metric_ratios_handle_the_empty_case() {
    // before any paged pool exists / any request arrives, the ratios are a well-defined 0 (no div-by-0).
    let m = Metrics {
        started: Instant::now(),
        requests: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        prompt_tokens: AtomicU64::new(0),
        requests_completed: AtomicU64::new(0),
        requests_cancelled: AtomicU64::new(0),
        requests_errored: AtomicU64::new(0),
        inflight: AtomicU64::new(0),
        ttft: Histogram::new(),
        itl: Histogram::new(),
        e2e: Histogram::new(),
        queue_wait: Histogram::new(),
        batch_size: AtomicU64::new(0),
        slot_capacity: 1,
        backend: "spirv-vulkan".to_string(),
        model: "test-model".into(),
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics::default(),
        speculative: SpeculativeMetrics::default(),
        timing: TimingMetrics::new(TelemetryConfig::CountersOnly),
    };
    assert_eq!(m.kv_cache_usage(), 0.0);
    assert_eq!(m.prefix_cache_hit_rate(), 0.0);
    assert_eq!(m.gpu_active_ratio(), 0.0); // no decode steps timed yet
    // a populated pool: 6/8 blocks, 3/4 prompt tokens cached.
    m.cache.kv_total_blocks.store(8, Ordering::Relaxed);
    m.cache.kv_used_blocks.store(6, Ordering::Relaxed);
    m.cache.prefix_query_tokens.store(4, Ordering::Relaxed);
    m.cache.prefix_hit_tokens.store(3, Ordering::Relaxed);
    assert_eq!(m.kv_cache_usage(), 0.75);
    assert_eq!(m.prefix_cache_hit_rate(), 0.75);
    // GPU-active ratio: 750ns GPU of 1000ns step = 0.75; saturates at 1.0 if timing skews high.
    m.cache.decode_gpu_ns.store(750, Ordering::Relaxed);
    m.cache.decode_step_ns.store(1000, Ordering::Relaxed);
    assert_eq!(m.gpu_active_ratio(), 0.75);
    m.cache.decode_gpu_ns.store(1200, Ordering::Relaxed);
    assert_eq!(m.gpu_active_ratio(), 1.0);
}

#[test]
fn resolve_forced_tools_handles_choice_modes() {
    let tools = serde_json::json!([
        {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
        {"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{}}}},
    ]);
    // "auto"/"none"/absent -> not forced.
    assert!(
        resolve_forced_tools(&Some(tools.clone()), &None)
            .unwrap()
            .is_none()
    );
    assert!(
        resolve_forced_tools(&Some(tools.clone()), &Some(serde_json::json!("auto")))
            .unwrap()
            .is_none()
    );
    assert!(
        resolve_forced_tools(&Some(tools.clone()), &Some(serde_json::json!("none")))
            .unwrap()
            .is_none()
    );
    // "required" -> all tools.
    let all = resolve_forced_tools(&Some(tools.clone()), &Some(serde_json::json!("required")))
        .unwrap()
        .unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].0, "get_weather");
    // a named function -> just that one, with its parameter schema.
    let named = resolve_forced_tools(
        &Some(tools.clone()),
        &Some(serde_json::json!({"type":"function","function":{"name":"get_time"}})),
    )
    .unwrap()
    .unwrap();
    assert_eq!(named.len(), 1);
    assert_eq!(named[0].0, "get_time");
    // a named function not in tools -> client error.
    assert!(
        resolve_forced_tools(
            &Some(tools.clone()),
            &Some(serde_json::json!({"type":"function","function":{"name":"nope"}})),
        )
        .is_err()
    );
    // forcing with no tools -> client error.
    assert!(resolve_forced_tools(&None, &Some(serde_json::json!("required"))).is_err());
}

#[test]
fn deeply_nested_guided_json_is_rejected_at_the_door() {
    // `guided_json`/`response_format` schemas are compiled by a recursive walk. serde_json caps nesting depth (~128) when the body is deserialized, so a deeply nested schema is a clean 400 before reaching the compiler (unlike the `guided_grammar` string parser, which has its own depth guard in poot_text::grammar).
    let deep = format!("{}{}", "[".repeat(400), "]".repeat(400));
    let body = format!(r#"{{"prompt":"x","guided_json":{deep}}}"#);
    assert!(
        serde_json::from_str::<CompletionReq>(&body).is_err(),
        "a 400-deep guided_json must be rejected during request deserialization"
    );
}

// ---- card 224: request-validation hardening ----

#[test]
fn conn_guard_decrements_on_drop_including_a_panicking_thread() {
    // Regression shape: the live-connection counter must come back down even if the connection-handler thread panics mid-`handle`: ConnGuard's decrement lives in Drop, so unwinding runs it. A counter that only goes up would eventually wedge the `--max-connections` cap shut with zero connections live.
    let counter = Arc::new(AtomicUsize::new(1));
    {
        let _guard = ConnGuard(Arc::clone(&counter));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "normal drop must decrement"
    );

    let counter = Arc::new(AtomicUsize::new(1));
    let counter_for_thread = Arc::clone(&counter);
    let result = std::thread::spawn(move || {
        let _guard = ConnGuard(counter_for_thread);
        panic!("simulated handle() panic");
    })
    .join();
    assert!(result.is_err(), "the panic must propagate to join()");
    assert_eq!(
        counter.load(Ordering::SeqCst),
        0,
        "ConnGuard must still decrement on unwind, or the cap wedges shut over time"
    );
}

#[test]
fn clamp_n_boundaries() {
    assert_eq!(clamp_n(0), 1, "0 floors to 1, matching the old .max(1)");
    assert_eq!(clamp_n(1), 1);
    assert_eq!(clamp_n(MAX_CHOICES), MAX_CHOICES);
    assert_eq!(clamp_n(MAX_CHOICES + 1), MAX_CHOICES);
    assert_eq!(clamp_n(usize::MAX), MAX_CHOICES);
}

#[test]
fn clamp_logprobs_boundaries() {
    assert_eq!(clamp_logprobs(0), 0);
    assert_eq!(clamp_logprobs(MAX_LOGPROBS), MAX_LOGPROBS);
    assert_eq!(clamp_logprobs(MAX_LOGPROBS + 1), MAX_LOGPROBS);
    assert_eq!(clamp_logprobs(usize::MAX), MAX_LOGPROBS);
}

/// card 224: `handle_completion`/`handle_chat` wrap a guided-decode constraint-compile failure in `GuidedDecodeError` so the shared error path can tell it from an internal failure.
/// `classify_handler_error` reads that back; this is a revert-catch for both the wrapping (map_err(GuidedDecodeError) at the apply_guided call sites) and the classification.
#[test]
fn guided_decode_error_classifies_400_other_errors_stay_500() {
    let guided: anyhow::Error = GuidedDecodeError(anyhow::anyhow!("bad guided_regex")).into();
    let (status, ty) = classify_handler_error(&guided);
    assert_eq!(status, 400);
    assert_eq!(ty, "invalid_request_error");

    let other = anyhow::anyhow!("some unrelated internal failure");
    let (status, ty) = classify_handler_error(&other);
    assert_eq!(status, 500);
    assert_eq!(ty, "server_error");
}

/// A SentencePiece vocabulary with `add_space_prefix`: the word-initial piece `\u{2581}the` decodes to
/// `the` alone (the isolated decode strips its space), but its logprob `bytes` are the vocabulary's.
/// Mutation: decode the token id alone in `logprobs_json`; the first byte is 0x74, not 0x20.
#[test]
fn logprob_bytes_of_a_word_initial_spm_token_start_with_a_space() {
    let tokens: Vec<String> = ["<unk>", "<s>", "</s>", "\u{2581}the", "a"]
        .map(String::from)
        .to_vec();
    let codec = poot_text::TextCodec::from_gguf(
        &poot_text::GgufVocab {
            model: Some("llama".into()),
            scores: Some(vec![0.0; tokens.len()]),
            tokens: Some(tokens),
            unknown_token_id: Some(0),
            add_space_prefix: Some(true),
            ..poot_text::GgufVocab::default()
        },
        2,
        poot_text::ChatFormat::ChatML,
    )
    .expect("build the SPM codec");
    let records = [poot_llm::TokenLogprob {
        token: 3,
        logprob: -0.5,
        top: vec![(4, -1.0)],
    }];
    let json = crate::api::logprobs_json(&codec, &records, 1);
    let entry = &json["content"][0];
    assert_eq!(
        entry["bytes"],
        serde_json::json!([32, 116, 104, 101]),
        "{entry}"
    );
    assert_eq!(entry["token"], " the");
    assert_eq!(entry["top_logprobs"][0]["bytes"], serde_json::json!([97]));
}

fn chat_req_with_part(part: &str) -> ChatReq {
    serde_json::from_str(&format!(
        r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"describe"}},{part}]}}]}}"#
    ))
    .expect("chat request with a content part parses")
}

/// Card 573: a request part whose modality the driver has no input for is refused with a typed error
/// naming the modality, before any job reaches the engine. Without the check the part is dropped and the
/// model answers a different request than the one sent. The decision reads what the driver prepared
/// (`DriverCaps::inputs`), so the same tokens-only caps refuse each of these.
#[test]
fn a_driver_with_only_token_inputs_refuses_an_image_audio_video_or_unknown_part_with_a_typed_error()
{
    let runner = tiny_model();
    for (part, modality) in [
        (
            r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#,
            "image",
        ),
        (
            r#"{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}"#,
            "audio",
        ),
        (r#"{"type":"video_url","video_url":{"url":"x"}}"#, "video"),
        (r#"{"type":"hologram"}"#, "content"),
    ] {
        let req = chat_req_with_part(part);
        let metrics = Arc::new(test_metrics("modality-refusal"));
        // No engine listens: a request that got past the refusal fails at the send instead of hanging.
        let (served, job_rx) = served_over(Arc::clone(&runner), 1);
        drop(job_rx);
        let error = handle_chat(&served, &metrics, &req, &|| false)
            .expect_err("a part the driver cannot take must be refused, not ignored");
        let refusal = error
            .downcast_ref::<UnsupportedCapabilityError>()
            .unwrap_or_else(|| panic!("{modality}: expected the typed refusal, got {error:#}"));
        assert_eq!(refusal.modality, modality);
        assert_eq!(
            classify_handler_error(&error),
            (400, "unsupported_capability")
        );
        assert_eq!(
            metrics.requests.load(Ordering::SeqCst),
            0,
            "{modality}: the refusal happens before any job is enqueued"
        );
    }
}

#[test]
fn a_text_only_part_array_is_accepted() {
    let req: ChatReq = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"describe"}]}]}"#,
    )
    .expect("text part array parses");
    reject_unsupported_modality(false, &req).expect("text parts are accepted");
}

/// The image decision is whether the served checkpoint has an image front end: without one an image is
/// the typed refusal naming that, and with one the image part is taken while audio, video and unknown
/// parts are still refused. Never a dropped part.
#[test]
fn an_image_part_is_taken_only_by_a_model_with_an_image_front_end() {
    let image = chat_req_with_part(
        r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#,
    );
    let refusal = reject_unsupported_modality(false, &image).expect_err("no image front end");
    assert_eq!(
        classify_handler_error(&refusal),
        (400, "unsupported_capability")
    );
    assert!(refusal.to_string().contains("image front end"), "{refusal}");
    reject_unsupported_modality(true, &image).expect("an image front end takes the image part");
    let audio = chat_req_with_part(r#"{"type":"input_audio","input_audio":{"data":"AAAA"}}"#);
    let refusal = reject_unsupported_modality(true, &audio).expect_err("no audio front end");
    assert!(refusal.to_string().contains("audio"), "{refusal}");
}

/// The streaming route refuses before it sends the 200 headers.
#[test]
fn streaming_chat_refuses_an_unsupported_part_before_the_200_headers() {
    let runner = tiny_model();
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect");
    client
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("bound the read");
    let (mut server, _) = listener.accept().expect("accept");
    let metrics = Arc::new(test_metrics("modality-refusal-stream"));
    // No engine listens: a request that got past the refusal fails at the send instead of hanging.
    let (served, job_rx) = served_over(runner, 1);
    drop(job_rx);
    let mut req = chat_req_with_part(
        r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#,
    );
    req.stream = true;
    stream_chat(&mut server, &served, &metrics, &req)
        .expect("the refusal is a written error response, not a handler failure");
    drop(server);
    let mut response = String::new();
    client.read_to_string(&mut response).expect("read response");
    assert!(
        response.starts_with("HTTP/1.1 400"),
        "a 400, not the streaming 200: {response}"
    );
    assert!(response.contains("unsupported_capability"), "{response}");
    assert!(response.contains("image"), "{response}");
    assert_eq!(metrics.requests.load(Ordering::SeqCst), 0);
}

use super::*;

#[test]
fn submit_propagates_natural_stop() {
    // The engine's natural-stop flag (EOS / stop sequence vs running to length) must reach the response handler through submit. A mock engine reports each value; submit surfaces it.
    use std::thread;
    for natural in [true, false] {
        let (tx, rx) = mpsc::channel::<Job>();
        let m = Arc::new(test_metrics("test"));
        let h = thread::spawn(move || {
            test_engine_loop(rx, m, move |_p, max_new, _s, _st, on| {
                for _ in 0..max_new {
                    if on("x").is_break() {
                        break;
                    }
                }
                Ok((vec![1, 2, 3], natural))
            });
        });
        let (_t, _lp, got) = submit(
            &tx,
            "p",
            3,
            Sampler::greedy(),
            vec![],
            false,
            LoraAdapterLease::none(),
            |_, _| std::ops::ControlFlow::Continue(()),
        )
        .unwrap();
        assert_eq!(
            got, natural,
            "submit must surface the engine's natural-stop flag"
        );
        drop(tx);
        h.join().unwrap();
    }
}

#[test]
fn apply_stop_truncates_at_earliest_match() {
    // earliest of multiple stops wins; the stop string itself is removed.
    let mut t = "Paris. It is the largest city".to_string();
    let hit = apply_stop(&mut t, &["XYZ".into(), " It".into(), " city".into()]);
    assert!(hit);
    assert_eq!(t, "Paris.");
}

#[test]
fn apply_stop_no_match_leaves_text() {
    let mut t = "Paris.".to_string();
    let hit = apply_stop(&mut t, &["ZZ".into()]);
    assert!(!hit);
    assert_eq!(t, "Paris.");
}

#[test]
fn apply_stop_ignores_empty_and_handles_no_stops() {
    let mut t = "hello".to_string();
    assert!(!apply_stop(&mut t, &[]));
    assert!(!apply_stop(&mut t, &["".into()]));
    assert_eq!(t, "hello");
}

#[test]
fn stream_receipt_reports_usage_and_throughput() {
    // 20 tokens in 500ms total, first at 100ms -> 40 tok/s, OpenAI-shaped usage with empty choices.
    let json = stream_receipt("text_completion", 7, 20, 100.0, 500.0);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"], serde_json::json!([]));
    assert_eq!(v["usage"]["prompt_tokens"], 7);
    assert_eq!(v["usage"]["completion_tokens"], 20);
    assert_eq!(v["usage"]["total_tokens"], 27);
    assert_eq!(v["timing"]["ttft_ms"], 100.0);
    assert_eq!(v["timing"]["total_ms"], 500.0);
    assert_eq!(v["timing"]["tokens_per_second"], 40.0);
}

#[test]
fn stream_receipt_handles_zero_generation() {
    // No tokens generated -> tok/s is 0, not NaN/inf (div-by-zero guard).
    let json = stream_receipt("chat.completion.chunk", 5, 0, 0.0, 0.0);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["object"], "chat.completion.chunk");
    assert_eq!(v["usage"]["completion_tokens"], 0);
    assert_eq!(v["timing"]["tokens_per_second"], 0.0);
}

#[test]
fn suppress_stream_usage_only_on_explicit_opt_out() {
    // None (stream_options absent) -> emit (current default; backward compat).
    assert!(!suppress_stream_usage(&None));
    // include_usage: true -> emit.
    assert!(!suppress_stream_usage(&Some(StreamOptions {
        include_usage: Some(true)
    })));
    // include_usage omitted inside stream_options ({}) -> emit (not an explicit opt-out).
    assert!(!suppress_stream_usage(&Some(StreamOptions {
        include_usage: None
    })));
    // include_usage: false -> the only case that suppresses (a helper that never suppresses fails here).
    assert!(suppress_stream_usage(&Some(StreamOptions {
        include_usage: Some(false)
    })));
}

#[test]
fn stream_stops_suppresses_marker_in_one_piece() {
    // SC-001: a stop fully inside one piece -> emit only the text before it, report a hit.
    let (out, hit) = run_stream_stops(&["<|im_end|>"], &["Hello<|im_end|>"]);
    assert_eq!(out, "Hello");
    assert!(hit);
}

#[test]
fn stream_stops_suppresses_marker_split_across_pieces() {
    // SC-001 / FR-004: the marker is split across token boundaries; the buffer holds back the partial
    // suffix until it resolves, emitting exactly the content before the stop and nothing of the marker.
    let (out, hit) = run_stream_stops(&["<|im_end|>"], &["Hel", "lo<|im_", "end|>", " ignored"]);
    assert_eq!(out, "Hello");
    assert!(hit);
}

#[test]
fn stream_stops_lossless_when_no_stop() {
    // SC-002: with no stop present, every character is emitted exactly once (nothing lost or held).
    let (out, hit) = run_stream_stops(&["<|im_end|>"], &["The ", "capital ", "of <| France"]);
    assert_eq!(out, "The capital of <| France");
    assert!(!hit);
}

#[test]
fn stream_stops_holds_back_dangling_partial_prefix() {
    // A trailing partial that is a stop prefix but never completes is flushed by finish() (not dropped).
    let (out, hit) = run_stream_stops(&["<|im_end|>"], &["done<|im_"]);
    assert_eq!(out, "done<|im_");
    assert!(!hit);
}

#[test]
fn submit_stops_forwarding_when_client_disconnects() {
    // card 058 cancellation: when `on_token` returns Break (failed socket write, client gone), `submit` stops forwarding and drops the reply receiver.
    // Model-free: a generator that would emit 100 tokens, but the caller bails after 2.
    use std::thread;
    let (tx, rx) = mpsc::channel::<Job>();
    let m = Arc::new(test_metrics("test"));
    let engine = thread::spawn(move || {
        test_engine_loop(rx, m, |_p, max_new, _s, _stops, on_token| {
            for _ in 0..max_new {
                if on_token("x").is_break() {
                    break;
                }
            }
            Ok((vec![1, 2, 3], false))
        });
    });
    let mut forwarded = 0;
    let result = submit(
        &tx,
        "p",
        100,
        Sampler::greedy(),
        vec![],
        true,
        LoraAdapterLease::none(),
        |_, _| {
            forwarded += 1;
            if forwarded < 2 {
                std::ops::ControlFlow::Continue(())
            } else {
                std::ops::ControlFlow::Break(())
            }
        },
    );
    drop(tx);
    engine.join().unwrap();
    assert_eq!(
        forwarded, 2,
        "submit stops forwarding once the client is gone"
    );
    assert_eq!(
        result.expect_err("client cancellation cannot be terminal success"),
        GenerationError::ClientCancelled
    );
}

#[test]
fn stop_list_accepts_string_or_array() {
    assert_eq!(
        Stop::list(&Some(Stop::One("a".into()))),
        vec!["a".to_string()]
    );
    assert_eq!(
        Stop::list(&Some(Stop::Many(vec!["a".into(), "b".into()]))),
        vec!["a".to_string(), "b".to_string()]
    );
    assert!(Stop::list(&None).is_empty());
}

#[test]
fn stop_list_truncates_unbounded_stop_array_to_cap() {
    // An untrusted client can send an arbitrarily long `stop` array (bounded only by the ~64MB body
    // limit -> millions of short strings); each drives an O(stops * text) per-token scan. `Stop::list`
    // must clamp the count at the boundary so the per-token cost stays a trivial constant.
    let many: Vec<String> = (0..10_000).map(|i| format!("s{i}")).collect();
    let out = Stop::list(&Some(Stop::Many(many.clone())));
    assert_eq!(out.len(), MAX_STOP_SEQUENCES, "stop count must be capped");
    // Truncation keeps the FIRST N (a prefix of the client's list), not a reordering.
    assert_eq!(out, many[..MAX_STOP_SEQUENCES]);
    // At or under the cap, the list is untouched.
    let few: Vec<String> = (0..MAX_STOP_SEQUENCES).map(|i| format!("s{i}")).collect();
    assert_eq!(Stop::list(&Some(Stop::Many(few.clone()))), few);
}

#[test]
fn sampler_of_applies_adjusters_on_both_greedy_and_temperature_paths() {
    // The API->engine mapping (`sampler_of`) must apply penalties / logit_bias on both the greedy (temperature<=0) and temperature bases. A greedy request with a penalty/bias must be non-greedy so it takes the host readback path (where the adjuster reshapes the argmax) instead of the on-device argmax that ignores it.
    let greedy_penalty: CompletionReq =
        serde_json::from_str(r#"{"prompt":"x","temperature":0.0,"repetition_penalty":1.2}"#)
            .unwrap();
    assert!(
        !sampler_of(&greedy_penalty.sampling).is_greedy(),
        "greedy request + penalty must take the host path so the penalty is applied"
    );
    let greedy_bias: CompletionReq =
        serde_json::from_str(r#"{"prompt":"x","temperature":0.0,"logit_bias":{"5":-10.0}}"#)
            .unwrap();
    assert!(
        !sampler_of(&greedy_bias.sampling).is_greedy(),
        "greedy request + logit_bias must take the host path"
    );

    // temperature>0 with only truncation knobs (top_k/top_p/min_p) is the on-device gumbel fast path.
    let temp_trunc: CompletionReq = serde_json::from_str(
        r#"{"prompt":"x","temperature":0.8,"top_k":40,"top_p":0.9,"min_p":0.1}"#,
    )
    .unwrap();
    assert!(
        sampler_of(&temp_trunc.sampling).is_simple_temperature(),
        "temperature + top_k/top_p/min_p is the on-device gumbel path"
    );
    // temperature>0 WITH a penalty is neither fast path - the gumbel kernel ignores penalties, so it must
    // fall back to the host readback path.
    let temp_penalty: CompletionReq =
        serde_json::from_str(r#"{"prompt":"x","temperature":0.8,"presence_penalty":0.5}"#).unwrap();
    let s = sampler_of(&temp_penalty.sampling);
    assert!(
        !s.is_greedy() && !s.is_simple_temperature(),
        "temperature + a penalty must take the host readback path (neither fast path)"
    );
}

#[test]
#[ignore = "loads qwen2.5-0.5b under POOT_MODELS_DIR on CPU (no GPU) to check the guided-error 400 path; run with --ignored"]
fn streaming_rejects_bad_guided_with_400_before_headers() {
    // A malformed/unsupported guided_json on the streaming path must return a clean 400 before the 200 + SSE headers, not a 200 followed by a broken stream. Validation (encode + apply_guided) is CPU-only and fails before `submit`.
    use std::io::Read;
    use std::sync::atomic::AtomicU64;
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b")) else {
        return;
    };
    let runner = Runner::load(&dir).expect("load model");
    let metrics = Metrics {
        started: Instant::now(),
        requests: AtomicU64::new(0),
        completion_tokens: AtomicU64::new(0),
        prompt_tokens: AtomicU64::new(0),
        requests_completed: AtomicU64::new(0),
        requests_cancelled: AtomicU64::new(0),
        requests_errored: AtomicU64::new(0),
        requests_preempted: AtomicU64::new(0),
        inflight: AtomicU64::new(0),
        ttft: Histogram::new(),
        itl: Histogram::new(),
        e2e: Histogram::new(),
        queue_wait: Histogram::new(),
        batch_size: AtomicU64::new(0),
        slot_capacity: 1,
        on_gpu: false,
        profile: false,
        model: "qwen2.5-0.5b".into(),
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics::default(),
        speculative: SpeculativeMetrics::default(),
    };
    let (tx, _rx) = mpsc::channel::<Job>(); // never used: validation fails before submit

    // loopback: the handler writes its response to `server`; read it back from `client`.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();

    // `{"type":"null"}` is an unsupported schema -> json_schema_to_regex errors.
    let body = r#"{"prompt":"hi","max_tokens":4,"stream":true,"guided_json":{"type":"null"}}"#;
    let req: CompletionReq = serde_json::from_str(body).unwrap();
    stream_completion(&mut server, Some(&tx), &runner, &metrics, &req)
        .expect("handler returns Ok after writing the 400");
    drop(server); // close so the client read finishes (EOF)

    let mut resp = String::new();
    client.read_to_string(&mut resp).unwrap();
    let status = resp.lines().next().unwrap_or("");
    assert!(
        status.starts_with("HTTP/1.1 400"),
        "bad guided_json must 400, got: {status}"
    );
    assert!(resp.contains("invalid_request_error"), "OpenAI error shape");
    assert!(
        !resp.contains("text/event-stream"),
        "must NOT have begun an SSE stream before validating"
    );
}

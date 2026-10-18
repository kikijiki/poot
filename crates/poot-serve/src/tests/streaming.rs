use super::*;

#[test]
fn submit_propagates_the_finish_reason() {
    // The engine's finish reason (EOS, a stop string, or running to length) must reach the response
    // handler through submit. A mock engine reports each value; submit surfaces it.
    use std::thread;
    for finish in [FinishReason::Eos, FinishReason::Stop, FinishReason::Length] {
        let (tx, rx) = mpsc::channel::<EngineMsg>();
        let m = Arc::new(test_metrics("test"));
        let engine_metrics = Arc::clone(&m);
        let h = thread::spawn(move || {
            test_engine_loop(rx, engine_metrics, move |_p, max_new, _s, _st, on| {
                for _ in 0..max_new {
                    if on("x").is_break() {
                        break;
                    }
                }
                Ok((vec![1, 2, 3], "xxx".to_string(), finish))
            });
        });
        let completion = submit_one(
            &tx,
            &m,
            generation_request("p", 3, false),
            &|| false,
            |_, _| std::ops::ControlFlow::Continue(()),
        )
        .unwrap();
        assert_eq!(
            completion.finish, finish,
            "submit must surface the engine's finish reason"
        );
        drop(tx);
        h.join().unwrap();
    }
}

/// Run `pieces` through a [`StopScanner`] over `stops`: everything released, then what `finish` flushes,
/// and whether a stop string completed.
fn scan_pieces(stops: &[&str], pieces: &[&str]) -> (String, bool) {
    let mut scanner = StopScanner::new(stops.iter().map(|s| s.to_string()).collect());
    let mut out = String::new();
    for piece in pieces {
        out.push_str(&scanner.push(piece));
    }
    out.push_str(&scanner.finish());
    (out, scanner.hit())
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
fn stop_scanner_suppresses_marker_in_one_piece() {
    // SC-001: a stop fully inside one piece -> release only the text before it, report a hit.
    let (out, hit) = scan_pieces(&["<|im_end|>"], &["Hello<|im_end|>"]);
    assert_eq!(out, "Hello");
    assert!(hit);
}

#[test]
fn stop_scanner_suppresses_marker_split_across_pieces() {
    // SC-001 / FR-004: the marker is split across token boundaries; the scanner holds back the partial
    // suffix until it resolves, releasing exactly the content before the stop and nothing of the marker.
    let (out, hit) = scan_pieces(&["<|im_end|>"], &["Hel", "lo<|im_", "end|>", " ignored"]);
    assert_eq!(out, "Hello");
    assert!(hit);
}

#[test]
fn stop_scanner_is_lossless_when_no_stop() {
    // SC-002: with no stop present, every character is released exactly once (nothing lost or held).
    let (out, hit) = scan_pieces(&["<|im_end|>"], &["The ", "capital ", "of <| France"]);
    assert_eq!(out, "The capital of <| France");
    assert!(!hit);
}

#[test]
fn stop_scanner_flushes_a_dangling_partial_prefix_on_finish() {
    // A trailing partial that is a stop prefix but never completes is flushed by finish() (not dropped).
    let (out, hit) = scan_pieces(&["<|im_end|>"], &["done<|im_"]);
    assert_eq!(out, "done<|im_");
    assert!(!hit);
}

#[test]
fn stop_scanner_cuts_at_the_earliest_of_several_stops_and_ignores_empty_ones() {
    // The earliest of multiple stops wins and the stop string itself is not released.
    let (out, hit) = scan_pieces(
        &["XYZ", " It", " city", ""],
        &["Paris. It is the largest city"],
    );
    assert_eq!(out, "Paris.");
    assert!(hit);
    // No stops, or only an empty one: nothing is scanned and nothing is held.
    let (out, hit) = scan_pieces(&[""], &["hello"]);
    assert_eq!(out, "hello");
    assert!(!hit);
    assert!(!StopScanner::new(vec![String::new()]).active());
}

#[test]
fn submit_stops_forwarding_when_client_disconnects() {
    // card 058 cancellation: when `on_token` returns Break (failed socket write, client gone), `submit` stops forwarding and drops the reply receiver.
    // Model-free: a generator that would emit 100 tokens, but the caller bails after 2.
    use std::thread;
    let (tx, rx) = mpsc::channel::<EngineMsg>();
    let m = Arc::new(test_metrics("test"));
    let engine_metrics = Arc::clone(&m);
    let engine = thread::spawn(move || {
        test_engine_loop(rx, engine_metrics, |_p, max_new, _s, _stops, on_token| {
            for _ in 0..max_new {
                if on_token("x").is_break() {
                    break;
                }
            }
            Ok((vec![1, 2, 3], "xxx".to_string(), FinishReason::Length))
        });
    });
    let mut forwarded = 0;
    let result = submit_one(
        &tx,
        &m,
        generation_request("p", 100, true),
        &|| false,
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
    let runner = load_handle(&dir);
    let metrics = Metrics {
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
        model: "qwen2.5-0.5b".into(),
        sys: Mutex::new(sysinfo::System::new()),
        cache: CacheMetrics::default(),
        speculative: SpeculativeMetrics::default(),
        timing: TimingMetrics::new(TelemetryConfig::CountersOnly),
    };
    let (served, _rx) = served_over(Arc::clone(&runner), 1); // never used: validation fails before submit

    // loopback: the handler writes its response to `server`; read it back from `client`.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = std::net::TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();

    // `{"type":"null"}` is an unsupported schema -> json_schema_to_regex errors.
    let body = r#"{"prompt":"hi","max_tokens":4,"stream":true,"guided_json":{"type":"null"}}"#;
    let req: CompletionReq = serde_json::from_str(body).unwrap();
    stream_completion(&mut server, &served, &metrics, &req)
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

/// SC-008: `commit_tokens` over the fake engine. A multi-token result whose second token is EOS keeps
/// one token and ends the request at EOS: the engine is told `kept = 1`, the sequence holds the prompt and
/// that token, and the token after EOS never enters either. Mutation: commit `result.tokens.len()`; the
/// engine is told 3 and the row goes red.
#[test]
fn commit_tokens_keeps_the_tokens_before_an_eos_inside_a_multi_token_result() {
    let mut engine = RecordingEngine::new(1);
    let (job, _submitted) = JobSpec::new("ab", 10).build();
    let mut slot = Slot::new(FakeSeq(1), job, 0, 10);
    let result = RowResult {
        seq: FakeSeq(1),
        consumed: 0,
        tokens: vec![5, 99, 7],
        logprobs: Vec::new(),
        speculation: RowSpeculation::NotApplicable,
    };
    let metrics = test_metrics("commit-eos");
    let fate = commit_tokens(&mut engine, &metrics, &mut slot, &result).unwrap();
    assert_eq!(
        fate,
        SlotFate::Finished {
            reason: FinishReason::Eos
        }
    );
    assert_eq!(engine.events(), vec![Event::Commit { seq: 1, kept: 1 }]);
    assert_eq!(slot.tokens, vec![97, 98, 5]);
    assert_eq!(metrics.completion_tokens.load(Ordering::Relaxed), 1);
}

/// SC-008: a stop string that lands on the middle token of a three-token result keeps the tokens up to
/// and including it: the engine is told `kept = 2`, the sequence holds the prompt and those two, and the
/// token after the stop never enters it. Mutation: skip the stop scan in `commit_tokens`; all three are
/// kept and committed and the row goes red.
#[test]
fn commit_tokens_keeps_the_tokens_up_to_a_stop_string_inside_a_multi_token_result() {
    let mut engine = RecordingEngine::new(1);
    let (job, _submitted) = JobSpec::new("ab", 10).stop("<2>").build();
    let mut slot = Slot::new(FakeSeq(1), job, 0, 10);
    let result = RowResult {
        seq: FakeSeq(1),
        consumed: 0,
        tokens: vec![1, 2, 3],
        logprobs: Vec::new(),
        speculation: RowSpeculation::NotApplicable,
    };
    let metrics = test_metrics("commit-stop");
    let fate = commit_tokens(&mut engine, &metrics, &mut slot, &result).unwrap();
    assert_eq!(
        fate,
        SlotFate::Finished {
            reason: FinishReason::Stop
        }
    );
    assert_eq!(engine.events(), vec![Event::Commit { seq: 1, kept: 2 }]);
    assert_eq!(slot.tokens, vec![97, 98, 1, 2]);
    assert_eq!(metrics.completion_tokens.load(Ordering::Relaxed), 2);
}

/// SC-006 (R477-014): a request with a stop string ends at the stop. The first result carries four
/// tokens and the stop string is the second one's piece, so two are kept and committed, the stream ends
/// there, and the loop runs no further step. The text before the stop is released and the stop's own
/// piece is not. Mutation: skip the stop check in the commit step (ignore `StopScanner::push`'s answer);
/// the request runs on to `max_new` and the row goes red.
#[test]
fn a_stop_string_ends_the_request_at_the_stop_and_no_token_is_decoded_past_it() {
    let mut engine = RecordingEngine::new(1).picking(|_, committed| {
        if committed == 0 {
            vec![1, 2, 3, 4]
        } else {
            vec![5]
        }
    });
    let (_exit, _metrics, submitted) = serve(
        &mut engine,
        vec![JobSpec::new("ab", 50).stop("<2>").streaming()],
    );
    let (pieces, terminal) = outcome(&submitted[0]);
    let GenEvent::Done(completion) = terminal else {
        panic!("the request ends with a Done");
    };
    assert_eq!(
        completion.tokens,
        vec![1, 2],
        "the output ends at the stop token"
    );
    assert_eq!(completion.finish, FinishReason::Stop);
    assert_eq!(completion.text, "<1>", "the text ends before the stop");
    assert_eq!(completion.prompt_tokens, 2);
    assert_eq!(
        pieces,
        vec!["<1>"],
        "no piece is streamed past the stop, and the stop's own piece is not streamed"
    );
    let events = engine.events();
    assert!(
        events.contains(&Event::Commit { seq: 1, kept: 2 }),
        "{events:?}"
    );
    let steps = events
        .iter()
        .filter(|event| matches!(event, Event::Step(_)))
        .count();
    assert_eq!(steps, 1, "no step runs after the stop: {events:?}");
}

/// A stop string that straddles two decode steps ends the request when its last piece lands, and not
/// before.
#[test]
fn a_stop_string_straddling_two_steps_ends_the_request_on_the_second() {
    let mut engine = RecordingEngine::new(1).picking(|_, committed| vec![committed as u32 + 1]);
    let (_exit, _metrics, submitted) =
        serve(&mut engine, vec![JobSpec::new("ab", 50).stop("<2><3>")]);
    let GenEvent::Done(completion) = outcome(&submitted[0]).1 else {
        panic!("the request ends with a Done");
    };
    assert_eq!(completion.tokens, vec![1, 2, 3]);
    assert_eq!(completion.finish, FinishReason::Stop);
    assert_eq!(completion.text, "<1>");
}

/// A request that reaches `max_new` flushes the text the stop scanner still holds as its last piece:
/// no stop string completed on it, so it is output.
#[test]
fn a_held_stop_prefix_is_flushed_when_the_request_ends_without_a_stop() {
    let mut engine = RecordingEngine::new(1).picking(|_, committed| vec![committed as u32 + 1]);
    let (_exit, _metrics, submitted) = serve(
        &mut engine,
        // `<2>` is held as the start of `<2><9>`; the request ends after two tokens.
        vec![JobSpec::new("ab", 2).stop("<2><9>").streaming()],
    );
    let (pieces, terminal) = outcome(&submitted[0]);
    let GenEvent::Done(completion) = terminal else {
        panic!("the request ends with a Done");
    };
    assert_eq!(completion.finish, FinishReason::Length);
    assert_eq!(completion.text, "<1><2>");
    assert_eq!(pieces, vec!["<1>", "<2>"], "the held tail is streamed last");
}

/// A request that streams nothing and scans no stop string has its text decoded once at the end.
#[test]
fn a_plain_request_has_its_text_decoded_once_when_it_ends() {
    let mut engine = RecordingEngine::new(1).picking(|_, committed| vec![committed as u32 + 1]);
    let (_exit, _metrics, submitted) = serve(&mut engine, vec![JobSpec::new("ab", 3)]);
    let GenEvent::Done(completion) = outcome(&submitted[0]).1 else {
        panic!("the request ends with a Done");
    };
    assert_eq!(completion.tokens, vec![1, 2, 3]);
    assert_eq!(completion.text, "<1><2><3>");
    assert_eq!(completion.finish, FinishReason::Length);
}

/// Streamed tokens carry the engine-produced position and pool usage, one per kept token.
#[test]
fn streamed_tokens_carry_their_index_and_the_pool_usage() {
    let mut engine = RecordingEngine::new(1).picking(|_, committed| vec![committed as u32 + 1]);
    let (_exit, _metrics, submitted) = serve(&mut engine, vec![JobSpec::new("ab", 3).streaming()]);
    let mut indices = Vec::new();
    for event in submitted[0].replies.try_iter() {
        if let GenEvent::Token(_, info) = event {
            assert_eq!(info.kv_used_ratio, 1.0 / 8.0, "one of eight blocks held");
            indices.push(info.token_idx);
        }
    }
    assert_eq!(indices, vec![0, 1, 2]);
}

/// A streamed piece with no receiver is a cancellation: the request is released (registering the prompt
/// it filled), counted as cancelled, and gets no terminal event.
#[test]
fn a_streamed_piece_with_no_receiver_cancels_and_releases_the_request() {
    let mut engine = RecordingEngine::new(1);
    let (job, submitted) = JobSpec::new("ab", 10).streaming().build();
    drop(submitted);
    let (tx, rx) = mpsc::channel();
    tx.send(EngineMsg::Generate(Box::new(job))).unwrap();
    drop(tx);
    let metrics = test_metrics("stream-cancel");
    run_engine(&mut engine, &rx, &metrics);
    assert!(matches!(
        engine.events().last(),
        Some(Event::Release {
            mode: poot_llm::driver::Release::Finished,
            ..
        })
    ));
    assert_eq!(metrics.requests_cancelled.load(Ordering::Relaxed), 1);
    assert_eq!(metrics.requests_completed.load(Ordering::Relaxed), 0);
}

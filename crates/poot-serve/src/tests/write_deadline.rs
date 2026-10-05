use super::*;

// ---- card 316: bounded response writes ----

#[test]
fn response_write_timeout_configuration_matrix_rejects_every_explicit_invalid_value() {
    let cases = [
        (None, Ok(Duration::from_secs(30))),
        (Some("5"), Ok(Duration::from_secs(5))),
        (Some(" 7 "), Ok(Duration::from_secs(7))),
        (Some("0"), Err(ResponseWriteTimeoutError::Zero)),
        (
            Some("-1"),
            Err(ResponseWriteTimeoutError::Negative { value: "-1".into() }),
        ),
        (
            Some("garbage"),
            Err(ResponseWriteTimeoutError::Malformed {
                value: "garbage".into(),
            }),
        ),
        (
            Some(""),
            Err(ResponseWriteTimeoutError::Malformed {
                value: String::new(),
            }),
        ),
        (
            Some("18446744073709551616"),
            Err(ResponseWriteTimeoutError::Overflow {
                value: "18446744073709551616".into(),
            }),
        ),
    ];
    for (raw, expected) in cases {
        assert_eq!(response_write_timeout_from(raw), expected, "raw={raw:?}");
    }
}

/// The table is data-driven: another socket lifecycle case is one new row, not new handler/engine branches and counters.
#[cfg(unix)]
#[test]
fn write_deadline_stream_socket_matrix_releases_connections_and_cancels_engine() {
    for case in [
        WriteDeadlineClientCase::NeverReads,
        WriteDeadlineClientCase::HealthyStream,
        WriteDeadlineClientCase::DisconnectedBeforeEvent,
    ] {
        run_write_deadline_stream_case(case);
    }
}

#[cfg(unix)]
#[test]
fn write_deadline_bounds_normal_http_response_to_never_reading_client() {
    let write_timeout = Duration::from_millis(150);
    let completion_budget = Duration::from_secs(3);
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind normal-response fixture");
    let addr = listener.local_addr().expect("normal-response address");
    let client = TcpStream::connect(addr).expect("connect normal-response client");
    set_write_deadline_socket_buffer(&client, libc::SO_RCVBUF);
    let (mut server, _) = listener.accept().expect("accept normal-response client");
    set_write_deadline_socket_buffer(&server, libc::SO_SNDBUF);
    configure_response_writes(&server, write_timeout)
        .expect("configure production response timeout");

    let (done_tx, done_rx) = mpsc::channel();
    let handler = std::thread::spawn(move || {
        let started = Instant::now();
        let body = "x".repeat(8 * 1024 * 1024);
        done_tx
            .send((respond(&mut server, 200, &body), started.elapsed()))
            .expect("report normal-response completion");
    });
    let (result, elapsed) = done_rx
        .recv_timeout(completion_budget)
        .expect("ordinary response write must terminate within its configured budget");
    assert!(result.is_err(), "the saturated ordinary response must fail");
    assert!(
        elapsed <= completion_budget,
        "ordinary response elapsed {elapsed:?} exceeds {completion_budget:?}"
    );
    handler.join().expect("join ordinary response handler");
    drop(client);
}

/// The claim is one write deadline, not a wall-clock budget: after the held-tail write saturates, the
/// handler gives up on the FIRST timeout instead of retrying finalization frames, each of which would
/// cost another `write_timeout`. So the assertion counts the frame writes `write_sse_data` recorded on
/// the handler thread ([`sse_write_probe`]) and checks that the single attempt actually blocked on the
/// deadline. Elapsed time from handler start also carries work the deadline does not govern (encoding,
/// serializing the held tail), which is what made a start-to-exit budget fail on a loaded box.
#[cfg(unix)]
#[test]
fn write_deadline_bounds_production_stream_finalization_after_held_tail_saturates() {
    // The deadline must stay long enough that "the attempt blocked" separates a real timeout from an
    // instant transport error, and short enough to keep one saturated attempt cheap in the suite.
    let write_timeout = Duration::from_millis(300);
    let completion_budget = Duration::from_secs(3);
    let model_path = write_tiny_granitemoe_gguf();
    let runner =
        Arc::new(Runner::load_gguf(&model_path).expect("load finalization fixture runner"));
    let _ = std::fs::remove_file(&model_path);

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind finalization fixture");
    let addr = listener.local_addr().expect("finalization fixture address");
    let client = TcpStream::connect(addr).expect("connect finalization fixture");
    set_write_deadline_socket_buffer(&client, libc::SO_RCVBUF);
    let (server, _) = listener.accept().expect("accept finalization fixture");
    set_write_deadline_socket_buffer(&server, libc::SO_SNDBUF);
    configure_response_writes(&server, write_timeout).expect("configure finalization timeout");

    let held_tail = "x".repeat(8 * 1024 * 1024);
    let stop = format!("{held_tail}y");
    let req: CompletionReq = serde_json::from_value(serde_json::json!({
        "prompt": "a",
        "max_tokens": 1,
        "stream": true,
        "stop": stop,
    }))
    .expect("finalization request");
    let metrics = Arc::new(test_metrics("write-finalization"));
    let live_connections = Arc::new(AtomicUsize::new(1));
    let handler_connections = Arc::clone(&live_connections);
    let handler_runner = Arc::clone(&runner);
    let handler_metrics = Arc::clone(&metrics);
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let (done_tx, done_rx) = mpsc::channel();
    let handler = std::thread::spawn(move || {
        let result = {
            let _guard = ConnGuard(handler_connections);
            let mut server = server;
            stream_completion(
                &mut server,
                Some(&job_tx),
                &handler_runner,
                &handler_metrics,
                &req,
            )
        };
        // The probe is thread-local, so it is read here: this thread recorded the writes.
        let (attempts, blocked) = sse_write_probe::summary();
        done_tx
            .send((result, attempts, blocked))
            .expect("report finalization result");
    });

    let job = job_rx
        .recv_timeout(completion_budget)
        .expect("production handler submits finalization job");
    job.reply
        .send(GenEvent::Token(held_tail, terminal_token_info()))
        .expect("send held tail");
    job.reply
        .send(GenEvent::Done(vec![1], vec![], false))
        .expect("send generation Done");

    let (result, attempts, blocked) = done_rx
        .recv_timeout(completion_budget)
        .expect("production handler terminates after first finalization timeout");
    assert!(
        result.is_err(),
        "the held-tail write must saturate and fail"
    );
    assert_eq!(
        attempts, 1,
        "finalization must attempt exactly one frame write after the held tail saturates \
         (attempts={attempts}, blocked={blocked:?}): every retry costs another {write_timeout:?} \
         write timeout"
    );
    assert!(
        blocked >= write_timeout / 2,
        "the single attempt must have blocked on the {write_timeout:?} write deadline before \
         failing, got {blocked:?}"
    );
    assert_eq!(live_connections.load(Ordering::SeqCst), 0);
    handler.join().expect("join finalization handler");
    drop(client);
}

/// card 224: a `Content-Length` over `MAX_BODY_BYTES` must produce a 413 on the wire (not a hang or crash).
/// This drives `handle`'s early-return path via `respond_error` semantics, as `handle` does with a `None` from `read_request`; `read_request_rejects_oversized_content_length_without_reading_body` covers the allocation-avoidance half.
#[test]
fn oversized_content_length_413_response_shape() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();

    respond_error(
        &mut server,
        413,
        "request body exceeds the maximum allowed size",
        "invalid_request_error",
    )
    .unwrap();
    drop(server);

    let mut resp = String::new();
    client.read_to_string(&mut resp).unwrap();
    let status = resp.lines().next().unwrap_or("");
    assert!(
        status.starts_with("HTTP/1.1 413 Payload Too Large"),
        "got: {status}"
    );
    assert!(resp.contains("invalid_request_error"));
}

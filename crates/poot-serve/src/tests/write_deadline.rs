use super::*;

// ---- card 316: bounded response writes ----

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
    // A hang guard only, as in the stream row: the bound under test is the write deadline, observed as the
    // time the write blocked, not a wall-clock budget a loaded box can overrun.
    let hang_guard = Duration::from_secs(120);
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
        .recv_timeout(hang_guard)
        .expect("ordinary response write must terminate on its write deadline");
    assert!(result.is_err(), "the saturated ordinary response must fail");
    assert!(
        elapsed >= write_timeout / 2,
        "the write must have blocked on the {write_timeout:?} deadline before failing, got {elapsed:?}"
    );
    handler.join().expect("join ordinary response handler");
    drop(client);
}

/// The claim is one write deadline, not a wall-clock budget: after a streamed piece's write saturates,
/// the handler gives up on the FIRST timeout instead of retrying later frames (an error event, finish
/// chunks, the receipt), each of which would cost another `write_timeout`. So the assertion counts the
/// frame writes `write_sse_data` recorded on the handler thread ([`sse_write_probe`]) and checks that
/// the single attempt actually blocked on the deadline. Elapsed time from handler start also carries
/// work the deadline does not govern (encoding, serializing the piece), which is what made a
/// start-to-exit budget fail on a loaded box. A client that cannot be written to is cancelled, which
/// writes nothing more.
#[cfg(unix)]
#[test]
fn write_deadline_bounds_production_stream_after_a_piece_saturates() {
    // The deadline must stay long enough that "the attempt blocked" separates a real timeout from an
    // instant transport error, and short enough to keep one saturated attempt cheap in the suite.
    let write_timeout = Duration::from_millis(300);
    // A hang guard only: the handler's own bound is the write deadline above, and the assertions below
    // count attempts and measure how long the attempt blocked. The guard is far above anything a loaded
    // box adds, so it fails a handler that never returns and no other run (a 3 s guard raced the kernel's
    // `SO_SNDTIMEO` write of the held tail under concurrent suite load).
    let hang_guard = Duration::from_secs(120);
    let runner = tiny_model();

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind finalization fixture");
    let addr = listener.local_addr().expect("finalization fixture address");
    let client = TcpStream::connect(addr).expect("connect finalization fixture");
    set_write_deadline_socket_buffer(&client, libc::SO_RCVBUF);
    let (server, _) = listener.accept().expect("accept finalization fixture");
    set_write_deadline_socket_buffer(&server, libc::SO_SNDBUF);
    configure_response_writes(&server, write_timeout).expect("configure finalization timeout");

    let piece = "x".repeat(8 * 1024 * 1024);
    let req: CompletionReq = serde_json::from_value(serde_json::json!({
        "prompt": "a",
        "max_tokens": 1,
        "stream": true,
    }))
    .expect("finalization request");
    let metrics = Arc::new(test_metrics("write-finalization"));
    let live_connections = Arc::new(AtomicUsize::new(1));
    let handler_connections = Arc::clone(&live_connections);
    let handler_metrics = Arc::clone(&metrics);
    let (served, job_rx) = served_over(runner, 1);
    let (done_tx, done_rx) = mpsc::channel();
    let handler = std::thread::spawn(move || {
        sse_write_probe::install();
        let result = {
            let _guard = ConnGuard(handler_connections);
            let mut server = server;
            stream_completion(&mut server, &served, &handler_metrics, &req)
        };
        // The probe is thread-local, so it is read here: this thread recorded the writes.
        let (attempts, blocked) = sse_write_probe::summary();
        done_tx
            .send((result, attempts, blocked))
            .expect("report finalization result");
    });

    let EngineMsg::Generate(job) = job_rx
        .recv_timeout(hang_guard)
        .expect("production handler submits finalization job")
    else {
        panic!("a stream handler sends a generation job");
    };
    job.reply
        .send(GenEvent::Token(piece, terminal_token_info()))
        .expect("send the piece");
    // The handler gives up on the piece and drops its subscription: later sends may find no receiver.
    let _ = job.reply.send(GenEvent::Done(Completion {
        prompt_tokens: 1,
        tokens: vec![1],
        text: String::new(),
        logprobs: vec![],
        finish: FinishReason::Length,
    }));

    let (result, attempts, blocked) = done_rx
        .recv_timeout(hang_guard)
        .expect("production handler terminates after the first write timeout");
    result.expect("a client that cannot be written to is cancelled, which writes nothing more");
    assert_eq!(
        attempts, 1,
        "the handler must attempt exactly one frame write after the piece saturates \
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

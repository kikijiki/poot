//! HTTP admission (POOT-1201): the bounded request queue, per-route body caps, the whole-request deadline,
//! strict framing and the origin checks. Every row drives the production `http::handle` over a loopback
//! socket (or the production request readers over a counting reader), against a queue the test reads in
//! place of an engine, so a refusal is told from an admission by what reached that queue.

use std::io::Cursor;
use std::net::{IpAddr, Shutdown};
use std::thread::JoinHandle;

use super::*;
use crate::http::{ModelRoute, write_all_within};
use crate::types::{Completion, class_status};

/// How long a row waits for something that must happen at once before it calls it a failure.
const PROMPT: Duration = Duration::from_secs(5);

/// One connection to `http::handle`: the client socket and the thread serving the other end.
struct Connection {
    client: TcpStream,
    handler: JoinHandle<()>,
}

impl Connection {
    /// Accept `limits` and `served` on a fresh loopback socket, `handle` running on its own thread from
    /// the moment of accept.
    fn open(served: &Arc<Served>, limits: &Arc<Limits>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let client = TcpStream::connect(listener.local_addr().unwrap()).expect("connect");
        client
            .set_read_timeout(Some(Duration::from_secs(20)))
            .expect("bound the response read");
        let (server, _) = listener.accept().expect("accept");
        let accepted = Instant::now();
        let (served, limits) = (Arc::clone(served), Arc::clone(limits));
        let handler = std::thread::spawn(move || {
            let policy = LoraAdminPolicy::from_configured_key(IpAddr::from([127, 0, 0, 1]), None)
                .expect("loopback policy");
            handle(
                server,
                accepted,
                &served,
                &EngineHealth::serving_for_test("fake", 4, 64),
                &test_metrics("admission"),
                &limits,
                &policy,
            )
            .expect("the handler answers without a transport error");
        });
        Self { client, handler }
    }

    fn send(&mut self, bytes: &[u8]) {
        self.client.write_all(bytes).expect("write request bytes");
        self.client.flush().expect("flush request");
    }

    /// A POST of `body` to `path` with the headers a well-formed client sends, plus `extra`.
    fn post(&mut self, path: &str, extra: &str, body: &str) {
        self.send(post_head(path, extra, body.len()).as_bytes());
        self.send(body.as_bytes());
    }

    /// The response to EOF, once the handler has ended.
    fn response(self) -> String {
        let Self {
            mut client,
            handler,
        } = self;
        let mut bytes = Vec::new();
        client
            .read_to_end(&mut bytes)
            .expect("read the response to EOF");
        // The handler waits for the client to close after it has answered.
        drop(client);
        handler.join().expect("the handler ends cleanly");
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// The request line and headers of a well-formed POST of `length` body bytes.
fn post_head(path: &str, extra: &str, length: usize) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{extra}Content-Length: {length}\r\n\r\n"
    )
}

fn status(response: &str) -> &str {
    response.lines().next().unwrap_or_default()
}

/// A served model over a queue the row reads, and the limits `limits` names.
fn setup(rows: usize, limits: Limits) -> (Arc<Served>, Receiver<EngineMsg>, Arc<Limits>) {
    let (served, queue) = served_over(tiny_model(), rows);
    (served, queue, Arc::new(limits))
}

const COMPLETION: &str = r#"{"prompt":"a","max_tokens":1}"#;

/// The next generation job on the queue.
fn next_job(queue: &Receiver<EngineMsg>) -> Box<Job> {
    match queue
        .recv_timeout(PROMPT)
        .expect("a job reaches the engine")
    {
        EngineMsg::Generate(job) => job,
        _ => panic!("a completions request sends a generation job"),
    }
}

fn assert_nothing_queued(queue: &Receiver<EngineMsg>, why: &str) {
    assert!(
        matches!(
            queue.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ),
        "{why}: a refused request must not reach the engine"
    );
}

/// SC-001. With the queue cap at 4, the fifth concurrent model request is answered 429 at once; the four
/// before it are admitted and held at the engine. Baseline: the fifth queued without bound.
/// Mutation: make `RequestQueue::admit` always admit; the fifth reaches the engine and its read times out.
#[test]
fn the_request_past_the_queue_cap_is_answered_429_at_once() {
    let (served, queue, limits) = setup(8, test_limits_with(4, Duration::from_secs(30)));
    let mut held = Vec::new();
    for _ in 0..4 {
        let mut connection = Connection::open(&served, &limits);
        connection.post("/v1/completions", "", COMPLETION);
        held.push((connection, next_job(&queue)));
    }

    let mut fifth = Connection::open(&served, &limits);
    fifth.post("/v1/completions", "", COMPLETION);
    fifth
        .client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let started = Instant::now();
    let response = fifth.response();
    assert!(
        status(&response).starts_with("HTTP/1.1 429"),
        "the fifth request must be refused, got {response:?}"
    );
    assert!(response.contains("Retry-After:"), "{response}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the refusal comes at once, not after a wait: {:?}",
        started.elapsed()
    );
    assert_nothing_queued(&queue, "the fifth request");

    // A finished request frees its place: the next one is admitted again.
    let (first, job) = held.remove(0);
    drop(job);
    drop(first.response());
    let mut sixth = Connection::open(&served, &limits);
    sixth.post("/v1/completions", "", COMPLETION);
    let _job = next_job(&queue);
    drop(held);
    drop(queue);
}

/// SC-002. A client that sends one byte every 0.5 s is cut off at the 2 s total deadline from accept; a
/// per-read timeout would let it go on for as long as it likes. Baseline: held until the client was done.
/// Mutation: make `DeadlineStream::read` wait a fixed 2 s per read instead of what is left of the total;
/// the trickle never stalls for 2 s and the row goes red.
#[test]
fn a_client_that_trickles_its_request_is_cut_off_at_the_total_deadline() {
    let (served, _queue, limits) = setup(1, test_limits_with(8, Duration::from_secs(2)));
    let mut connection = Connection::open(&served, &limits);
    connection.send(b"POST /v1/completions HTTP/1.1\r\nX-Slow: ");
    let mut trickle = connection.client.try_clone().unwrap();
    let writer = std::thread::spawn(move || {
        for _ in 0..16 {
            std::thread::sleep(Duration::from_millis(500));
            if trickle.write_all(b"a").is_err() {
                return;
            }
        }
    });
    let started = Instant::now();
    let response = connection.response();
    let elapsed = started.elapsed();
    writer.join().unwrap();
    assert!(
        status(&response).starts_with("HTTP/1.1 408"),
        "the trickling client is told its time is up: {response:?}"
    );
    assert!(
        elapsed >= Duration::from_millis(1900) && elapsed < Duration::from_millis(3500),
        "cut off at the 2 s deadline, not before and not when the client stops: {elapsed:?}"
    );
}

/// A reader over `inner` that counts the bytes taken from it.
struct Counting<R> {
    inner: R,
    taken: Arc<AtomicUsize>,
}

impl<R: Read> Read for Counting<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.taken.fetch_add(n, Ordering::SeqCst);
        Ok(n)
    }
}

/// SC-003. A 10 MiB body on a route capped at a few MiB is refused from its `Content-Length` with no body
/// byte read: the counting reader has taken exactly the head. Mutation: read the body before checking the
/// cap in `read_request_body_from`; the counter passes the head length and the row goes red.
#[test]
fn a_ten_mib_body_is_refused_from_content_length_before_any_of_it_is_read() {
    let head = post_head("/v1/completions", "", 10 * 1024 * 1024);
    let taken = Arc::new(AtomicUsize::new(0));
    let source = Counting {
        inner: Cursor::new(head.clone().into_bytes()).chain(std::io::repeat(b'x')),
        taken: Arc::clone(&taken),
    };
    // One byte of read-ahead, so the count is what the parser asked for and not what a buffer fetched.
    let mut reader = std::io::BufReader::with_capacity(1, source);
    let refused = read_request_from(&mut reader, BODY_CAP);
    assert!(
        matches!(refused, Err(ReadFailure::Refused(Refusal::BodyTooLarge))),
        "a body over the cap is refused, got {:?}",
        refused.map(|request| request.body.len())
    );
    assert_eq!(
        taken.load(Ordering::SeqCst),
        head.len(),
        "only the head was read: no byte of the 10 MiB body"
    );
}

/// The same refusal on the wire: the client declares 10 MiB and sends none of it, and is answered 413
/// without the server waiting for a body.
#[test]
fn an_oversized_declaration_is_answered_413_without_waiting_for_the_body() {
    let (served, queue, limits) = setup(1, test_limits_with(8, Duration::from_secs(30)));
    let mut connection = Connection::open(&served, &limits);
    connection.send(post_head("/v1/completions", "", 10 * 1024 * 1024).as_bytes());
    let response = connection.response();
    assert!(status(&response).starts_with("HTTP/1.1 413"), "{response}");
    assert_nothing_queued(&queue, "an oversized body");
}

/// The image cap belongs to a chat on a server that takes images and to no other route: every route of a
/// text server, and every route but chat on an image server, takes the text cap.
#[test]
fn the_image_cap_belongs_to_chat_on_a_server_that_takes_images() {
    let caps = BodyCaps {
        text: 4,
        images: 64,
    };
    for (path, takes_images, cap) in [
        ("/v1/chat/completions", true, 64),
        ("/v1/chat/completions", false, 4),
        ("/v1/completions", true, 4),
        ("/v1/embeddings", true, 4),
        ("/v1/rerank", true, 4),
        ("/v1/completions", false, 4),
    ] {
        let route = ModelRoute::of(path).expect("a model route");
        assert_eq!(
            caps.of(route, takes_images),
            cap,
            "{path} images={takes_images}"
        );
    }
}

/// On the wire, a text server refuses a body over the text cap on every model route, chat included.
#[test]
fn a_text_server_refuses_a_body_over_the_text_cap_on_every_model_route() {
    for path in [
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/rerank",
    ] {
        let (served, queue, limits) = setup(1, test_limits());
        let mut connection = Connection::open(&served, &limits);
        connection.send(post_head(path, "", BODY_CAP + 1).as_bytes());
        let response = connection.response();
        assert!(
            status(&response).starts_with("HTTP/1.1 413"),
            "{path}: {response}"
        );
        assert_nothing_queued(&queue, path);
    }
}

/// The rate-limit decision comes before the body is read: the second request of a client allowed one
/// per minute is answered 429 on its head alone, while its declared body is withheld.
/// Mutation: check the limiter after `read_request_body_from` in `serve_model`; the answer waits for a
/// body that never comes and the request deadline answers 408 instead.
#[test]
fn the_rate_limit_is_decided_before_the_body_is_read() {
    let mut limits = test_limits_with(8, Duration::from_secs(3));
    limits.limiter = RateLimiter::new(1);
    let (served, queue, limits) = setup(2, limits);
    let mut first = Connection::open(&served, &limits);
    first.post("/v1/completions", "", COMPLETION);
    let _job = next_job(&queue);

    let mut second = Connection::open(&served, &limits);
    second.send(post_head("/v1/completions", "", 64).as_bytes());
    let started = Instant::now();
    let response = second.response();
    assert!(status(&response).starts_with("HTTP/1.1 429"), "{response}");
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "answered on the head, not after the deadline: {:?}",
        started.elapsed()
    );
}

/// SC-004. `n = 128` runs as 128 engine jobs from the one handler thread: while all 128 are in the
/// engine the process holds no thread more than before the request, and the response carries 128
/// choices. Baseline: a scoped thread per choice. Mutation: restore the per-choice thread fan-out in
/// `handle_completion`; the thread count rises by 127 and the row goes red. It runs in a child process,
/// so the other tests of the binary cannot change the count.
#[test]
fn n_128_runs_as_128_jobs_from_one_handler_thread() {
    // The thread count is a property of the whole process, and the other tests of this binary start
    // threads of their own while they run in parallel. The row therefore runs alone, in a child process
    // that executes only this test.
    const CHILD: &str = "POOT_SERVE_N128_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().expect("the test binary"))
            .args([
                "--exact",
                "tests::admission::n_128_runs_as_128_jobs_from_one_handler_thread",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .output()
            .expect("run the test alone in a child process");
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    fn threads() -> usize {
        std::fs::read_dir("/proc/self/task").unwrap().count()
    }
    let (served, queue, limits) = setup(128, test_limits());
    let mut connection = Connection::open(&served, &limits);
    // The handler thread exists and waits for its request; the count to beat is taken with it running.
    let before = threads();
    connection.post(
        "/v1/completions",
        "",
        r#"{"prompt":"a","max_tokens":1,"n":128}"#,
    );
    let jobs: Vec<_> = (0..128).map(|_| next_job(&queue)).collect();
    let during = threads();
    assert!(
        during <= before,
        "128 jobs are in the engine and the handler spent {} more threads on them",
        during.saturating_sub(before)
    );
    for job in &jobs {
        job.reply
            .send(GenEvent::Done(Completion {
                prompt_tokens: 1,
                tokens: vec![1],
                text: "x".into(),
                logprobs: Vec::new(),
                finish: FinishReason::Length,
            }))
            .expect("the handler still waits for the job");
    }
    let response = connection.response();
    assert!(status(&response).starts_with("HTTP/1.1 200"), "{response}");
    let json = lifecycle::json_body(&response);
    assert_eq!(json["choices"].as_array().map(Vec::len), Some(128));
}

/// SC-005. Each framing and origin violation is refused with its own 4xx and never reaches the engine.
/// Mutation: remove one check at a time (the `Transfer-Encoding` arm, the duplicate `Content-Length`
/// arm, the UTF-8 conversion, `require_json`, `HostPolicy::check`); exactly that row goes red.
#[test]
fn each_framing_and_origin_violation_gets_its_typed_4xx() {
    struct Case {
        name: &'static str,
        head: &'static str,
        body: &'static [u8],
        status: &'static str,
        /// What the refusal says, so a row is told from another that happens to share its status.
        reason: &'static str,
    }
    let cases = [
        Case {
            name: "chunked transfer encoding",
            head: "Host: localhost\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\nContent-Length: 12\r\n",
            body: b"2\r\n{}\r\n0\r\n\r\n",
            status: "HTTP/1.1 411",
            reason: "Transfer-Encoding is not supported",
        },
        Case {
            name: "duplicate Content-Length",
            head: "Host: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\nContent-Length: 2\r\n",
            body: b"{}",
            status: "HTTP/1.1 400",
            reason: "Content-Length appears more than once",
        },
        Case {
            name: "signed Content-Length",
            head: "Host: localhost\r\nContent-Type: application/json\r\nContent-Length: +2\r\n",
            body: b"{}",
            status: "HTTP/1.1 400",
            reason: "whole number of bytes",
        },
        Case {
            name: "no Content-Length",
            head: "Host: localhost\r\nContent-Type: application/json\r\n",
            body: b"{}",
            status: "HTTP/1.1 411",
            reason: "needs a Content-Length",
        },
        Case {
            name: "non-UTF-8 body",
            head: "Host: localhost\r\nContent-Type: application/json\r\nContent-Length: 4\r\n",
            body: &[b'{', 0xff, 0xfe, b'}'],
            status: "HTTP/1.1 400",
            reason: "not valid UTF-8",
        },
        Case {
            name: "Content-Type text/plain",
            head: "Host: localhost\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n",
            body: b"{}",
            status: "HTTP/1.1 415",
            reason: "application/json",
        },
        Case {
            name: "no Content-Type",
            head: "Host: localhost\r\nContent-Length: 2\r\n",
            body: b"{}",
            status: "HTTP/1.1 415",
            reason: "application/json",
        },
        Case {
            name: "foreign Host",
            head: "Host: evil.example\r\nContent-Type: application/json\r\nContent-Length: 2\r\n",
            body: b"{}",
            status: "HTTP/1.1 403",
            reason: "does not name this server",
        },
        Case {
            name: "rebound Host on a foreign port",
            head: "Host: localhost:1\r\nContent-Type: application/json\r\nContent-Length: 2\r\n",
            body: b"{}",
            status: "HTTP/1.1 403",
            reason: "does not name this server",
        },
        Case {
            name: "no Host",
            head: "Content-Type: application/json\r\nContent-Length: 2\r\n",
            body: b"{}",
            status: "HTTP/1.1 400",
            reason: "needs a Host header",
        },
    ];
    for case in cases {
        let (served, queue, limits) = setup(1, test_limits());
        let mut connection = Connection::open(&served, &limits);
        connection.send(format!("POST /v1/completions HTTP/1.1\r\n{}\r\n", case.head).as_bytes());
        connection.send(case.body);
        let response = connection.response();
        assert!(
            status(&response).starts_with(case.status),
            "{}: wanted {}, got {response:?}",
            case.name,
            case.status
        );
        assert!(
            response.contains(case.reason),
            "{}: wanted a refusal saying {:?}, got {response:?}",
            case.name,
            case.reason
        );
        assert_nothing_queued(&queue, case.name);
    }
}

/// The checks do not refuse what a client may rightly send: a charset parameter on the media type and the
/// loopback addresses of the listener are admitted and reach the engine.
#[test]
fn a_well_formed_request_is_admitted() {
    for extra in ["", "X-Other: 1\r\n"] {
        for (host, content_type) in [
            ("localhost", "application/json"),
            ("127.0.0.1:0", "Application/JSON; charset=utf-8"),
            ("[::1]", "application/json"),
        ] {
            let (served, queue, limits) = setup(1, test_limits());
            let mut connection = Connection::open(&served, &limits);
            connection.send(
                format!(
                    "POST /v1/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: {content_type}\r\n{extra}Content-Length: {}\r\n\r\n{COMPLETION}",
                    COMPLETION.len()
                )
                .as_bytes(),
            );
            let job = next_job(&queue);
            drop(job);
            let response = connection.response();
            assert!(
                !status(&response).starts_with("HTTP/1.1 4"),
                "{host} {content_type}: {response}"
            );
        }
    }
}

/// The administration routes are held to the same origin check as a POST: a request that carries a
/// foreign Host is refused after authorization and before any body is read.
#[test]
fn a_lora_admin_request_from_a_foreign_host_is_refused() {
    let (served, _queue, limits) = setup(1, test_limits());
    let mut connection = Connection::open(&served, &limits);
    connection.send(b"GET /v1/lora_adapters HTTP/1.1\r\nHost: evil.example\r\n\r\n");
    let response = connection.response();
    assert!(status(&response).starts_with("HTTP/1.1 403"), "{response}");
    assert!(response.contains("Host"), "{response}");
}

/// A write to a reader that drains a few bytes inside every window is cut off at the window as a whole,
/// not once per `write` call. Mutation: make `write_all_within` a plain `write_all`; the writer outlives
/// the row's bound.
#[cfg(unix)]
#[test]
fn a_slow_reader_cannot_stretch_a_write_past_its_deadline() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    set_write_deadline_socket_buffer(&client, libc::SO_RCVBUF);
    let (mut server, _) = listener.accept().unwrap();
    set_write_deadline_socket_buffer(&server, libc::SO_SNDBUF);
    server
        .set_write_timeout(Some(Duration::from_secs(1)))
        .unwrap();
    // Drains 512 bytes every 200 ms: well inside the 1 s window each time, and far too slow to finish.
    let stop = Arc::new(AtomicBool::new(false));
    let reader_stop = Arc::clone(&stop);
    let reader = std::thread::spawn(move || {
        let mut chunk = [0u8; 512];
        while !reader_stop.load(Ordering::SeqCst) && client.read(&mut chunk).is_ok_and(|n| n > 0) {
            std::thread::sleep(Duration::from_millis(200));
        }
    });
    let (done_tx, done_rx) = mpsc::channel();
    let writer = std::thread::spawn(move || {
        let started = Instant::now();
        let result = write_all_within(&mut server, &vec![0u8; 1 << 20]);
        done_tx.send((result, started.elapsed())).unwrap();
    });
    let (result, elapsed) = done_rx
        .recv_timeout(Duration::from_secs(15))
        .expect("the write ends within the bound");
    assert_eq!(
        result
            .expect_err("the slow reader costs the connection")
            .kind(),
        std::io::ErrorKind::TimedOut
    );
    assert!(elapsed < Duration::from_secs(3), "{elapsed:?}");
    writer.join().unwrap();
    stop.store(true, Ordering::SeqCst);
    reader.join().unwrap();
}

/// A request the engine has not opened by its `start_by` is failed as unavailable (503) instead of
/// queueing without bound; one still within its wait is served.
/// Mutation: skip `Scheduler::fail_overdue`; the overdue job is served and the row goes red.
#[test]
fn a_request_that_waited_past_its_maximum_queue_wait_is_failed_as_unavailable() {
    let mut engine = RecordingEngine::new(1);
    let (_exit, _metrics, submitted) = serve(
        &mut engine,
        vec![JobSpec::new("ab", 2).overdue(), JobSpec::new("cd", 2)],
    );
    let GenEvent::Failed(failure) = outcome(&submitted[0]).1 else {
        panic!("the overdue request must be failed, not served");
    };
    assert_eq!(class_status(failure.class()), 503, "{failure}");
    assert!(failure.to_string().contains("for the engine"), "{failure}");
    assert!(
        matches!(outcome(&submitted[1]).1, GenEvent::Done(..)),
        "a request within its wait is served"
    );
}

/// Only so many admitted requests may be reading their body at once: a request that declares a body and
/// then sends none holds its place, and the next one is answered 429 without waiting for it.
/// Mutation: skip the `limits.receiving.admit()` check; the second request is admitted and reaches the
/// engine.
#[test]
fn requests_still_sending_their_body_are_bounded_apart_from_the_queue() {
    let limits = Limits {
        receiving: RequestQueue::new(std::num::NonZeroUsize::new(1).unwrap()),
        ..test_limits()
    };
    let (served, queue, limits) = setup(2, limits);
    let mut stalled = Connection::open(&served, &limits);
    stalled.send(post_head("/v1/completions", "", COMPLETION.len()).as_bytes());
    // Let the stalled request take its place before the second arrives.
    std::thread::sleep(Duration::from_millis(300));

    let mut second = Connection::open(&served, &limits);
    second.post("/v1/completions", "", COMPLETION);
    let response = second.response();
    assert!(
        status(&response).starts_with("HTTP/1.1 429"),
        "the second request is turned away: {response:?}"
    );
    assert_nothing_queued(&queue, "the second request");

    // The stalled request completes: its place was held, not lost.
    stalled.send(COMPLETION.as_bytes());
    let _job = next_job(&queue);
    drop(stalled);
}

/// A refusal sent before the body arrives still reaches a client that is mid-way through sending it: the
/// connection is closed after its input is taken in, so the kernel does not reset it and drop the answer.
/// The client sends its body in pieces after the head is refused, as a client that has not yet seen the
/// answer does; a server that has closed with input unread resets the connection under those writes.
/// Whatever the timing, a correct server never fails the client: the pause only gives a broken one the
/// time to have closed.
/// Mutation: drop the drain in `close_with_body_unread` (leave the shutdown); the client's writes or its
/// read fail with a reset.
#[test]
fn a_refusal_reaches_a_client_that_is_still_sending_its_body() {
    let (served, _queue, limits) = setup(1, test_limits());
    for _ in 0..10 {
        let Connection {
            mut client,
            handler,
        } = Connection::open(&served, &limits);
        client
            .write_all(post_head("/v1/completions", "", BODY_CAP + 1).as_bytes())
            .expect("send the head");
        for _ in 0..50 {
            if handler.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        for piece in 0..16 {
            client
                .write_all(&[b'x'; 8 * 1024])
                .unwrap_or_else(|e| panic!("body piece {piece} of a refused request: {e}"));
        }
        client.shutdown(Shutdown::Write).expect("finish sending");
        let mut bytes = Vec::new();
        client
            .read_to_end(&mut bytes)
            .expect("the refusal is read, not reset");
        handler.join().expect("the handler ends cleanly");
        let response = String::from_utf8_lossy(&bytes);
        assert!(
            status(&response).starts_with("HTTP/1.1 413"),
            "the refusal arrives intact: {response:?}"
        );
    }
}

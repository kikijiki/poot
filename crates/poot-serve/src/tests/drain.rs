use super::*;
use crate::startup::sigint_handler;

pub static DRAIN_FIXTURE_RELEASE: AtomicBool = AtomicBool::new(false);

pub extern "C" fn drain_fixture_release_handler(_: libc::c_int) {
    DRAIN_FIXTURE_RELEASE.store(true, Ordering::SeqCst);
}

#[test]
fn bounded_drain_timeout_parser_keeps_positive_default() {
    assert_eq!(
        server_drain_timeout_from(None).duration(),
        Duration::from_secs(30)
    );
    assert_eq!(
        server_drain_timeout_from(Some(" 7 ")).duration(),
        Duration::from_secs(7)
    );
    assert_eq!(
        server_drain_timeout_from(Some("0")).duration(),
        Duration::from_secs(30)
    );
    assert_eq!(
        server_drain_timeout_from(Some("invalid")).duration(),
        Duration::from_secs(30)
    );
    assert_eq!(
        server_drain_timeout_from(Some(&u64::MAX.to_string())).duration(),
        Duration::from_secs(MAX_DRAIN_TIMEOUT_SECS)
    );

    let start = Instant::now();
    assert_eq!(deadline_after(start, Duration::MAX), start);
    assert!(server_drain_timeout_from(Some("1")).deadline_from(start) > start);
}

static SLOW_THREAD_EXIT_ENTERED: AtomicBool = AtomicBool::new(false);

struct SlowThreadExit(u8);

impl Drop for SlowThreadExit {
    fn drop(&mut self) {
        SLOW_THREAD_EXIT_ENTERED.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(300));
    }
}

std::thread_local! {
    static SLOW_THREAD_EXIT: SlowThreadExit = const { SlowThreadExit(1) };
}

#[test]
fn bounded_drain_completion_notice_does_not_authorize_live_thread_join() {
    SLOW_THREAD_EXIT_ENTERED.store(false, Ordering::SeqCst);
    let mut workers = WorkerRegistry::new();
    workers
        .spawn("slow-thread-exit", move || {
            SLOW_THREAD_EXIT.with(|value| std::hint::black_box(value.0));
        })
        .unwrap();

    // The completion guard sends while the worker closure returns, before Rust runs this thread-local
    // destructor. Even though `is_finished` may already be true, the controller must not perform the join
    // itself and turn the bounded wait into a blocking wait. The injected deadline is `now`: an expired
    // deadline must return `false`, with the live handle still owned and no join dispatched.
    let entered_deadline = deadline_after(Instant::now(), Duration::from_secs(1));
    while !SLOW_THREAD_EXIT_ENTERED.load(Ordering::SeqCst) {
        assert!(
            Instant::now() < entered_deadline,
            "worker did not enter its thread-local destructor"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        !workers.wait_for_all_until(Instant::now()),
        "an expired deadline must not report the live worker drained"
    );
    assert_eq!(workers.worker_count(), 1);

    assert!(workers.wait_for_all_until(deadline_after(Instant::now(), Duration::from_secs(1))));
}

#[test]
fn bounded_drain_false_completion_notice_keeps_live_handle() {
    let worker_started = Arc::new(AtomicBool::new(false));
    let worker_started_in_thread = Arc::clone(&worker_started);
    let mut workers = WorkerRegistry::new();
    workers
        .spawn("early-completion-notice", move || {
            worker_started_in_thread.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(200));
        })
        .unwrap();
    while !worker_started.load(Ordering::SeqCst) {
        std::thread::yield_now();
    }

    workers.notify_completion_for_test();
    workers.reap_ready();
    assert_eq!(workers.worker_count(), 1);
    assert_eq!(
        workers.joins_in_flight_for_test(),
        0,
        "a completion notice dispatched a join before is_finished"
    );
    assert!(workers.wait_for_all_until(deadline_after(Instant::now(), Duration::from_secs(1))));
}

/// SC-005: with an injected (already-expired) deadline the bounded drain reports "not done" while the
/// in-flight job is unfinished, and a future deadline completes only after that job ran to completion.
/// A `wait_until` that returns `true` before the job finishes fails the expired-deadline assertion.
#[test]
fn drain_wait_returns_only_after_the_in_flight_job_finishes() {
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let finished = Arc::new(AtomicBool::new(false));
    let finished_worker = Arc::clone(&finished);
    let mut workers = WorkerRegistry::new();
    workers
        .spawn("in-flight", move || {
            let _ = release_rx.recv();
            finished_worker.store(true, Ordering::SeqCst);
        })
        .unwrap();

    // Injected clock: the deadline is `now`. The job has not finished, so the drain must report the
    // bounded wait as not done, not return "drained".
    assert!(
        !workers.wait_for_all_until(Instant::now()),
        "drain reported done while the in-flight job was unfinished"
    );
    assert!(
        !finished.load(Ordering::SeqCst),
        "the fixture job finished before the expired-deadline wait ended"
    );
    assert_eq!(workers.worker_count(), 1);

    // Release the job; a future deadline completes only after it ran.
    release_tx.send(()).unwrap();
    assert!(workers.wait_for_all_until(deadline_after(Instant::now(), Duration::from_secs(5))));
    assert!(
        finished.load(Ordering::SeqCst),
        "drain returned before the in-flight job finished"
    );
}

/// Child half of Card 327's subprocess contract. The parent starts this exact test in a fresh process.
/// `barrier` waits for SIGUSR1; `nonreading` fills a small response socket. Both use the production worker
/// registry, response-write timeout configurator, signal flag, graceful deadline, and cancellation phase.
#[test]
fn bounded_drain_subprocess_child() {
    let Ok(case) = std::env::var("POOT_DRAIN_FIXTURE_CASE") else {
        return;
    };

    use std::io::{Read as _, Write as _};
    use std::os::fd::AsRawFd as _;

    SHUTDOWN.store(false, Ordering::SeqCst);
    DRAIN_FIXTURE_RELEASE.store(false, Ordering::SeqCst);
    reset_drain_cancellation();
    unsafe {
        libc::signal(
            libc::SIGTERM,
            sigint_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGINT,
            sigint_handler as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGUSR1,
            drain_fixture_release_handler as *const () as libc::sighandler_t,
        );
    }

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind bounded-drain fixture");
    listener
        .set_nonblocking(true)
        .expect("make bounded-drain fixture nonblocking");
    println!("DRAIN_FIXTURE_READY {}", listener.local_addr().unwrap());
    std::io::stdout().flush().unwrap();

    let mut workers = WorkerRegistry::new();
    while !SHUTDOWN.load(Ordering::SeqCst) {
        workers.reap_ready();
        match listener.accept() {
            Ok((stream, _)) => {
                if SHUTDOWN.load(Ordering::SeqCst) {
                    drop(stream);
                    break;
                }
                configure_response_writes(&stream, Duration::from_secs(1)).unwrap();
                let send_buffer: libc::c_int = 4096;
                let configured = unsafe {
                    libc::setsockopt(
                        stream.as_raw_fd(),
                        libc::SOL_SOCKET,
                        libc::SO_SNDBUF,
                        (&send_buffer as *const libc::c_int).cast(),
                        std::mem::size_of_val(&send_buffer) as libc::socklen_t,
                    )
                };
                assert_eq!(configured, 0, "configure fixture response send buffer");
                workers
                    .spawn_connection("poot-drain-fixture", stream, move |mut stream| {
                        let mut command = [0u8; 1];
                        stream.read_exact(&mut command).expect("read fixture command");
                        println!("DRAIN_FIXTURE_ACCEPTED {}", command[0] as char);
                        std::io::stdout().flush().unwrap();
                        match command[0] {
                            b'B' => {
                                while !DRAIN_FIXTURE_RELEASE.load(Ordering::SeqCst)
                                    && !drain_cancellation_requested()
                                {
                                    std::thread::sleep(Duration::from_millis(10));
                                }
                                if drain_cancellation_requested() {
                                    println!("DRAIN_FIXTURE_CANCELLED");
                                    std::io::stdout().flush().unwrap();
                                    return;
                                }
                                stream
                                    .write_all(
                                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK",
                                    )
                                    .expect("write released fixture response");
                                println!("DRAIN_FIXTURE_COMPLETED");
                                std::io::stdout().flush().unwrap();
                            }
                            b'N' => {
                                println!("DRAIN_FIXTURE_WRITE_STARTED");
                                std::io::stdout().flush().unwrap();
                                let body = vec![b'x'; 32 * 1024 * 1024];
                                let result = stream.write_all(&body);
                                assert!(result.is_err(), "non-reading control unexpectedly read 32 MiB");
                                println!("DRAIN_FIXTURE_WRITE_FAILED");
                                std::io::stdout().flush().unwrap();
                            }
                            other => panic!("unknown drain fixture command {other}"),
                        }
                    })
                    .expect("spawn bounded-drain fixture connection");
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("bounded-drain fixture accept failed: {error}"),
        }
    }

    drop(listener);
    println!("DRAIN_FIXTURE_DRAINING {case}");
    std::io::stdout().flush().unwrap();
    // The graceful window is the injected clock: the parent sets `POOT_DRAIN_TIMEOUT_SECS`, so the
    // bounded wait below uses the configured deadline rather than a hard-coded duration.
    let deadline = crate::lifecycle::server_drain_timeout().deadline_from(Instant::now());
    let graceful = workers.wait_for_connections_until(deadline);
    if !graceful {
        workers.cancel_connections();
        println!("DRAIN_FIXTURE_CANCEL_REQUESTED");
        std::io::stdout().flush().unwrap();
        let _ =
            workers.wait_for_all_until(deadline_after(Instant::now(), CANCELLATION_SETTLE_TIMEOUT));
    }
    println!("DRAIN_FIXTURE_EXIT graceful={graceful}");
    std::io::stdout().flush().unwrap();
}

struct BoundedDrainChild {
    child: std::process::Child,
    lines: Receiver<String>,
    readers: Vec<std::thread::JoinHandle<()>>,
}

impl BoundedDrainChild {
    fn spawn(case: &str) -> Self {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::drain::bounded_drain_subprocess_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("POOT_DRAIN_FIXTURE_CASE", case)
            .env("POOT_DRAIN_TIMEOUT_SECS", "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn bounded-drain child");
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        let (line_tx, lines) = mpsc::channel();
        let stdout_tx = line_tx.clone();
        let stdout_reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout)
                .lines()
                .map_while(Result::ok)
            {
                let _ = stdout_tx.send(line);
            }
        });
        let stderr_reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stderr)
                .lines()
                .map_while(Result::ok)
            {
                let _ = line_tx.send(format!("stderr: {line}"));
            }
        });
        Self {
            child,
            lines,
            readers: vec![stdout_reader, stderr_reader],
        }
    }

    fn marker(&self, prefix: &str, timeout: Duration) -> String {
        let deadline = deadline_after(Instant::now(), timeout);
        let mut seen = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "missing {prefix:?}; saw {seen:?}");
            let line = self
                .lines
                .recv_timeout(remaining)
                .unwrap_or_else(|error| panic!("missing {prefix:?}: {error}; saw {seen:?}"));
            if let Some(index) = line.find(prefix) {
                return line[index..].to_string();
            }
            seen.push(line);
        }
    }

    fn signal(&self, signal: libc::c_int) {
        let result = unsafe { libc::kill(self.child.id() as libc::pid_t, signal) };
        assert_eq!(result, 0, "signal bounded-drain child");
    }

    fn assert_alive(&mut self) {
        assert!(
            self.child.try_wait().unwrap().is_none(),
            "bounded-drain child exited while accepted work was still draining"
        );
    }

    fn wait_success(&mut self, timeout: Duration) {
        let deadline = deadline_after(Instant::now(), timeout);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "bounded-drain child failed: {status}");
                return;
            }
            assert!(
                Instant::now() < deadline,
                "bounded-drain child exceeded its process-exit budget"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for BoundedDrainChild {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
    }
}

fn bounded_drain_fixture_client(child: &BoundedDrainChild, command: u8) -> TcpStream {
    let ready = child.marker("DRAIN_FIXTURE_READY ", Duration::from_secs(3));
    let addr = ready
        .strip_prefix("DRAIN_FIXTURE_READY ")
        .unwrap()
        .parse::<std::net::SocketAddr>()
        .unwrap();
    let mut client = TcpStream::connect(addr).expect("connect bounded-drain fixture");
    client
        .set_read_timeout(Some(Duration::from_secs(4)))
        .unwrap();
    client.write_all(&[command]).unwrap();
    client
}

#[test]
fn bounded_drain_subprocess_completes_accepted_request_before_exit() {
    let mut child = BoundedDrainChild::spawn("graceful");
    let mut client = bounded_drain_fixture_client(&child, b'B');
    child.marker("DRAIN_FIXTURE_ACCEPTED B", Duration::from_secs(3));
    child.signal(libc::SIGTERM);
    child.marker("DRAIN_FIXTURE_DRAINING", Duration::from_secs(3));
    child.assert_alive();

    // Admission is closed before drain starts. A fresh connect must fail while the already accepted
    // barrier request keeps the process itself alive.
    let ready_addr = client.local_addr().unwrap();
    let server_addr = client.peer_addr().unwrap();
    assert_ne!(ready_addr, server_addr);
    assert!(
        TcpStream::connect_timeout(&server_addr, Duration::from_millis(200)).is_err(),
        "a post-signal connection reached the closed admission listener"
    );

    child.signal(libc::SIGUSR1);
    child.marker("DRAIN_FIXTURE_COMPLETED", Duration::from_secs(3));
    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    assert!(
        response.ends_with("\r\n\r\nOK"),
        "response was {response:?}"
    );
    child.marker("DRAIN_FIXTURE_EXIT graceful=true", Duration::from_secs(3));
    child.wait_success(Duration::from_secs(3));
}

#[test]
fn bounded_drain_subprocess_cancels_barrier_on_timeout() {
    let mut child = BoundedDrainChild::spawn("timeout");
    let mut client = bounded_drain_fixture_client(&child, b'B');
    child.marker("DRAIN_FIXTURE_ACCEPTED B", Duration::from_secs(3));
    child.signal(libc::SIGTERM);
    child.marker("DRAIN_FIXTURE_DRAINING", Duration::from_secs(3));
    // Ordering under the injected graceful window (`POOT_DRAIN_TIMEOUT_SECS=1`): the accepted work is
    // still alive when the drain starts, and it is cancelled only when that window expires.
    child.assert_alive();
    child.marker("DRAIN_FIXTURE_CANCELLED", Duration::from_secs(3));
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    assert!(
        response.is_empty(),
        "cancelled barrier wrote response bytes"
    );
    child.marker("DRAIN_FIXTURE_EXIT graceful=false", Duration::from_secs(3));
    child.wait_success(Duration::from_secs(3));
}

#[test]
fn bounded_drain_subprocess_closes_nonreading_response() {
    let mut child = BoundedDrainChild::spawn("nonreading");
    let mut client = bounded_drain_fixture_client(&child, b'N');
    child.marker("DRAIN_FIXTURE_ACCEPTED N", Duration::from_secs(3));
    child.marker("DRAIN_FIXTURE_WRITE_STARTED", Duration::from_secs(3));
    child.signal(libc::SIGTERM);
    child.marker("DRAIN_FIXTURE_DRAINING", Duration::from_secs(3));
    child.marker("DRAIN_FIXTURE_WRITE_FAILED", Duration::from_secs(4));
    child.wait_success(Duration::from_secs(3));

    // Buffered bytes may remain readable, but the peer must reach EOF after the bounded child exit.
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    assert!(response.len() < 32 * 1024 * 1024);
}

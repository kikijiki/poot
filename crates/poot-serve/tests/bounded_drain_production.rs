//! Shutdown coverage through the production `poot-serve` binary, on every build. The tiny GGUF is the
//! model input; the accepted-request barrier is a deliberately incomplete HTTP head handled by the
//! production parser, held open through the real listener. Nothing of the accept loop, deadline, or
//! cancellation path is reimplemented here, and the server has no test hook: the drain bound is observed
//! from outside, over its sockets.

use poot_test_util::contained_child::ContainChild;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

#[path = "support/tiny_gguf.rs"]
mod tiny_gguf;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(20);
const PROCESS_EXIT_TIMEOUT: Duration = Duration::from_secs(5);

fn poot_serve_bin() -> &'static str {
    // `std::env::var`, not `env!`: the latter bakes the path into this test binary's compiled object code
    // at build time, and a shared compile cache (kache) that reuses that object across worktrees by
    // source-content hash would then serve whichever worktree's path happened to compile it first
    // (card 530's build.rs fix; card 543 review). `std::env::var` reads the environment cargo sets fresh
    // for every test-binary invocation, so it is correct regardless of which worktree compiled the binary.
    // Leaked to `&'static str` so every call site keeps this function's prior return type.
    static BIN: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        std::env::var("CARGO_BIN_EXE_poot-serve")
            .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
    })
    .as_str()
}

fn unused_loopback_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve loopback port");
    listener.local_addr().unwrap()
}

fn unique_suffix() -> String {
    format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

struct TinyModel(PathBuf);

impl TinyModel {
    fn create() -> Self {
        let path = std::env::temp_dir().join(format!(
            "poot-card327-production-drain-{}.gguf",
            unique_suffix()
        ));
        // A deepseek2 GGUF: a registered family the production binary loads as a `ModelHandle`.
        tiny_gguf::write_tiny_deepseek2_gguf(&path, 65_536);
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TinyModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Strip ANSI SGR sequences (`\x1b[...m`) from captured log text before string matching (card 417).
/// The server already avoids emitting them on a non-terminal sink (`main.rs`'s
/// `.with_ansi(is_terminal)`); this guards against a future default change.
fn strip_ansi(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b && bytes.get(i + 1) == Some(&b'[') {
            i += 2;
            while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                i += 1;
            }
            i += 1; // skip the final letter (e.g. 'm') that ends the SGR sequence.
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn spawn_reader<R: Read + Send + 'static>(
    mut reader: R,
    log: Arc<Mutex<Vec<u8>>>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut bytes = [0u8; 4096];
        loop {
            match reader.read(&mut bytes) {
                Ok(0) | Err(_) => return,
                Ok(len) => log.lock().unwrap().extend_from_slice(&bytes[..len]),
            }
        }
    })
}

struct ProductionServer {
    child: Child,
    addr: SocketAddr,
    log: Arc<Mutex<Vec<u8>>>,
    readers: Vec<JoinHandle<()>>,
    _model: TinyModel,
}

impl ProductionServer {
    fn spawn(drain_secs: u64) -> Self {
        Self::spawn_inner(drain_secs, None)
    }

    /// As [`Self::spawn`], with the removed `POOT_CARD327_TEST_ACCEPT_BARRIER_ADDR` pointing at `hook`.
    fn spawn_with_removed_hook(drain_secs: u64, hook: SocketAddr) -> Self {
        Self::spawn_inner(drain_secs, Some(hook))
    }

    fn spawn_inner(drain_secs: u64, removed_hook: Option<SocketAddr>) -> Self {
        let model = TinyModel::create();
        let addr = unused_loopback_addr();
        let mut command = Command::new(poot_serve_bin());
        // The server opens a device before it listens (the engine plan prepares the model on it), so the
        // production binary runs on the wgpu device lane, named explicitly.
        command
            .contained()
            .arg(model.path())
            .arg(addr.to_string())
            .args(["--backend", "wgpu"])
            .args(["--drain-timeout-secs", &drain_secs.to_string()])
            .args(["--read-timeout-secs", "30"])
            .args(["--write-timeout-secs", "30"]);
        if let Some(hook) = removed_hook {
            command.env("POOT_CARD327_TEST_ACCEPT_BARRIER_ADDR", hook.to_string());
        }
        let mut child = command
            .env("RUST_LOG", "info")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn production poot-serve");
        let log = Arc::new(Mutex::new(Vec::new()));
        let readers = vec![
            spawn_reader(child.stdout.take().unwrap(), Arc::clone(&log)),
            spawn_reader(child.stderr.take().unwrap(), Arc::clone(&log)),
        ];
        let mut server = Self {
            child,
            addr,
            log,
            readers,
            _model: model,
        };
        server.wait_ready();
        server
    }

    fn log_text(&self) -> String {
        strip_ansi(&String::from_utf8_lossy(&self.log.lock().unwrap()))
    }

    fn wait_log(&mut self, needle: &str, timeout: Duration) {
        let deadline = Instant::now().checked_add(timeout).unwrap();
        loop {
            let log = self.log_text();
            if log.contains(needle) {
                return;
            }
            assert!(
                self.child.try_wait().unwrap().is_none(),
                "server exited before log {needle:?}; log:\n{log}"
            );
            assert!(
                Instant::now() < deadline,
                "timed out waiting for log {needle:?}; log:\n{log}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn wait_ready(&mut self) {
        let deadline = Instant::now().checked_add(STARTUP_TIMEOUT).unwrap();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                panic!(
                    "production server exited during startup: {status}; log:\n{}",
                    self.log_text()
                );
            }
            if http_get(self.addr, "/health", Duration::from_millis(250))
                .is_ok_and(|response| response.contains("200 OK"))
            {
                self.wait_log("poot-serve listening", Duration::from_secs(1));
                return;
            }
            assert!(
                Instant::now() < deadline,
                "production server did not become ready; log:\n{}",
                self.log_text()
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn signal(&self, signal: libc::c_int) {
        let result = unsafe { libc::kill(self.child.id() as libc::pid_t, signal) };
        assert_eq!(result, 0, "signal production server");
    }

    fn assert_alive(&mut self) {
        assert!(
            self.child.try_wait().unwrap().is_none(),
            "production server exited before accepted work completed; log:\n{}",
            self.log_text()
        );
    }

    fn wait_success(&mut self, timeout: Duration) {
        let deadline = Instant::now().checked_add(timeout).unwrap();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "production server failed: {status}; log:\n{}",
                    self.log_text()
                );
                return;
            }
            assert!(
                Instant::now() < deadline,
                "production server exceeded exit bound; log:\n{}",
                self.log_text()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for ProductionServer {
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

fn http_get(addr: SocketAddr, path: &str, timeout: Duration) -> std::io::Result<String> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn accepted_request_barrier(server: &ProductionServer) -> TcpStream {
    let mut barrier = TcpStream::connect(server.addr).expect("connect barrier request");
    barrier
        .set_read_timeout(Some(Duration::from_secs(4)))
        .unwrap();
    barrier
        .set_write_timeout(Some(Duration::from_secs(4)))
        .unwrap();
    barrier
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n")
        .unwrap();

    // A later health request completes only after the single accept loop admitted and spawned the
    // earlier partial request, which proves it was accepted before the signal.
    let probe = http_get(server.addr, "/health", Duration::from_secs(2)).unwrap();
    assert!(probe.contains("200 OK"));
    barrier
}

/// A signal closes admission and lets an accepted request finish: the incomplete head is completed after
/// SIGINT and gets its response, then the process exits cleanly inside the drain bound. The old
/// accept-barrier environment variable is set to a listener that expects no connection: it has no effect
/// on any build, so the server never connects to it. Mutation: restore the `debug_assertions`-gated read
/// in `lifecycle.rs`; on a debug build the server connects to the listener and this row goes red.
#[test]
fn bounded_drain_production_completes_accepted_barrier_before_exit() {
    let removed_hook = TcpListener::bind("127.0.0.1:0").expect("bind the removed-hook listener");
    removed_hook
        .set_nonblocking(true)
        .expect("make the removed-hook listener nonblocking");
    let mut server =
        ProductionServer::spawn_with_removed_hook(3, removed_hook.local_addr().unwrap());
    let mut barrier = accepted_request_barrier(&server);
    server.signal(libc::SIGINT);
    server.wait_log("shutting down (signal received", Duration::from_secs(2));
    server.assert_alive();

    barrier.write_all(b"\r\n").unwrap();
    let mut response = String::new();
    barrier.read_to_string(&mut response).unwrap();
    assert!(
        response.contains("503 Service Unavailable")
            && response.contains(r#""engine_loaded":true"#)
            && response.contains(r#""status":"draining""#),
        "barrier response did not complete: {response:?}"
    );
    server.wait_success(PROCESS_EXIT_TIMEOUT);
    assert!(server.log_text().contains("server stopped"));
    match removed_hook.accept() {
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
        Ok((_, peer)) => panic!("the server connected to the removed test hook from {peer}"),
        Err(error) => panic!("removed-hook listener failed: {error}"),
    }
}

#[test]
fn bounded_drain_production_timeout_closes_accepted_barrier() {
    let mut server = ProductionServer::spawn(1);
    let mut barrier = accepted_request_barrier(&server);
    let started = Instant::now();
    server.signal(libc::SIGTERM);
    server.wait_log("graceful drain expired", Duration::from_secs(3));

    let mut response = Vec::new();
    match barrier.read_to_end(&mut response) {
        Ok(_) => {}
        Err(error) => assert!(
            matches!(
                error.kind(),
                std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::UnexpectedEof
            ),
            "unexpected barrier cancellation error: {error}"
        ),
    }
    assert!(
        response.is_empty(),
        "timed-out barrier wrote response bytes"
    );
    server.wait_success(PROCESS_EXIT_TIMEOUT);
    assert!(
        started.elapsed() >= Duration::from_millis(800),
        "drain timeout fired before its configured window"
    );
    assert!(
        started.elapsed() < Duration::from_secs(4),
        "timeout cancellation exceeded the drain and settle bound"
    );
    assert!(
        server.log_text().contains("closed_response_sockets=1"),
        "timeout did not cancel the accepted response socket; log:\n{}",
        server.log_text()
    );
}

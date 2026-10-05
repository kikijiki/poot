//! Card 327 shutdown coverage through the production `poot-serve` binary. The tiny GGUF is the model
//! input; the accepted-request barrier is a deliberately incomplete HTTP head handled by the production
//! parser. Nothing of the accept loop, deadline, or cancellation path is reimplemented here.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use poot_load::gguf::{GgufValue, write_gguf};

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
        let (hidden, query, key_value, intermediate, vocab) =
            (32usize, 32usize, 16usize, 64usize, 4usize);
        let f32_bytes = |len: usize| -> Vec<u8> {
            (0..len)
                .flat_map(|index| (((index % 7) as f32) * 0.1 - 0.3).to_le_bytes())
                .collect()
        };
        const F32: u32 = 0;
        // granitemoe: an unregistered family, so the production binary loads it on the Runner.
        const EXPERTS: usize = 2;
        let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
            (
                "token_embd.weight",
                vec![hidden as u64, vocab as u64],
                F32,
                f32_bytes(hidden * vocab),
            ),
            (
                "output.weight",
                vec![hidden as u64, vocab as u64],
                F32,
                f32_bytes(hidden * vocab),
            ),
            (
                "output_norm.weight",
                vec![hidden as u64],
                F32,
                f32_bytes(hidden),
            ),
            (
                "blk.0.attn_q.weight",
                vec![hidden as u64, query as u64],
                F32,
                f32_bytes(hidden * query),
            ),
            (
                "blk.0.attn_k.weight",
                vec![hidden as u64, key_value as u64],
                F32,
                f32_bytes(hidden * key_value),
            ),
            (
                "blk.0.attn_v.weight",
                vec![hidden as u64, key_value as u64],
                F32,
                f32_bytes(hidden * key_value),
            ),
            (
                "blk.0.attn_output.weight",
                vec![query as u64, hidden as u64],
                F32,
                f32_bytes(query * hidden),
            ),
            (
                "blk.0.attn_norm.weight",
                vec![hidden as u64],
                F32,
                f32_bytes(hidden),
            ),
            (
                "blk.0.ffn_norm.weight",
                vec![hidden as u64],
                F32,
                f32_bytes(hidden),
            ),
            (
                "blk.0.ffn_gate_inp.weight",
                vec![hidden as u64, EXPERTS as u64],
                F32,
                f32_bytes(hidden * EXPERTS),
            ),
            (
                "blk.0.ffn_gate_exps.weight",
                vec![hidden as u64, intermediate as u64, EXPERTS as u64],
                F32,
                f32_bytes(hidden * intermediate * EXPERTS),
            ),
            (
                "blk.0.ffn_up_exps.weight",
                vec![hidden as u64, intermediate as u64, EXPERTS as u64],
                F32,
                f32_bytes(hidden * intermediate * EXPERTS),
            ),
            (
                "blk.0.ffn_down_exps.weight",
                vec![intermediate as u64, hidden as u64, EXPERTS as u64],
                F32,
                f32_bytes(intermediate * hidden * EXPERTS),
            ),
        ];
        let metadata = vec![
            ("general.architecture", GgufValue::Str("granitemoe".into())),
            ("granitemoe.embedding_length", GgufValue::U32(hidden as u32)),
            ("granitemoe.block_count", GgufValue::U32(1)),
            ("granitemoe.attention.head_count", GgufValue::U32(2)),
            ("granitemoe.attention.head_count_kv", GgufValue::U32(1)),
            (
                "granitemoe.feed_forward_length",
                GgufValue::U32(intermediate as u32),
            ),
            ("granitemoe.context_length", GgufValue::U32(65_536)),
            ("granitemoe.expert_count", GgufValue::U32(EXPERTS as u32)),
            ("granitemoe.expert_used_count", GgufValue::U32(1)),
            ("granitemoe.embedding_scale", GgufValue::F32(1.0)),
            ("granitemoe.attention.scale", GgufValue::F32(0.25)),
            ("granitemoe.residual_scale", GgufValue::F32(1.0)),
            ("granitemoe.logit_scale", GgufValue::F32(1.0)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(99)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(
                    ["a", "b", "ab", "c"]
                        .iter()
                        .map(|token| GgufValue::Str((*token).to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array(vec![GgufValue::Str("a b".into())]),
            ),
        ];
        let path = std::env::temp_dir().join(format!(
            "poot-card327-production-drain-{}.gguf",
            unique_suffix()
        ));
        std::fs::write(&path, write_gguf(&metadata, &tensors)).expect("write tiny GGUF");
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
    accept_barrier: Option<TcpStream>,
    log: Arc<Mutex<Vec<u8>>>,
    readers: Vec<JoinHandle<()>>,
    _model: TinyModel,
}

impl ProductionServer {
    fn spawn(drain_secs: u64) -> Self {
        Self::spawn_inner(drain_secs, false)
    }

    #[cfg(debug_assertions)]
    fn spawn_with_accept_barrier(drain_secs: u64) -> Self {
        Self::spawn_inner(drain_secs, true)
    }

    fn spawn_inner(drain_secs: u64, with_accept_barrier: bool) -> Self {
        let model = TinyModel::create();
        let addr = unused_loopback_addr();
        let accept_barrier_listener = with_accept_barrier.then(|| {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind Card 327 accept barrier");
            listener
                .set_nonblocking(true)
                .expect("make Card 327 accept barrier nonblocking");
            listener
        });
        let mut command = Command::new(poot_serve_bin());
        command
            .arg(model.path())
            .arg(addr.to_string())
            .env("POOT_DRAIN_TIMEOUT_SECS", drain_secs.to_string())
            .env("POOT_READ_TIMEOUT_SECS", "30")
            .env("POOT_WRITE_TIMEOUT_SECS", "30")
            .env("POOT_SLOTS", "1")
            .env("POOT_CAP", "65536");
        if let Some(listener) = &accept_barrier_listener {
            command.env(
                "POOT_CARD327_TEST_ACCEPT_BARRIER_ADDR",
                listener.local_addr().unwrap().to_string(),
            );
        }
        // Keep this always-run test hardware-free: wgpu honors this backend selector, and an
        // unavailable Vulkan ICD selects the server's CPU fallback.
        let mut child = command
            .env("WGPU_BACKEND", "vulkan")
            .env("VK_DRIVER_FILES", "/nonexistent/poot-card327-vulkan.json")
            .env("VK_ICD_FILENAMES", "/nonexistent/poot-card327-vulkan.json")
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
        let accept_barrier = accept_barrier_listener.map(|listener| {
            let accept_deadline = Instant::now().checked_add(STARTUP_TIMEOUT).unwrap();
            let stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        if let Some(status) = child.try_wait().unwrap() {
                            panic!(
                                "production server exited before accept-barrier setup: {status}"
                            );
                        }
                        assert!(
                            Instant::now() < accept_deadline,
                            "production server did not connect its Card 327 accept barrier"
                        );
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("accept Card 327 control connection: {error}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
        });
        let mut server = Self {
            child,
            addr,
            accept_barrier,
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
                self.wait_log(
                    "decoder loaded (no generation engine",
                    Duration::from_secs(1),
                );
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

    fn arm_accept_barrier(&mut self) {
        self.accept_barrier
            .as_mut()
            .expect("Card 327 accept barrier enabled")
            .write_all(b"A")
            .unwrap();
    }

    fn wait_for_guarded_accept(&mut self) {
        let mut marker = [0u8; 1];
        self.accept_barrier
            .as_mut()
            .expect("Card 327 accept barrier enabled")
            .read_exact(&mut marker)
            .unwrap();
        assert_eq!(marker, [b'B']);
    }

    fn release_guarded_accept(&mut self) {
        self.accept_barrier
            .as_mut()
            .expect("Card 327 accept barrier enabled")
            .write_all(b"R")
            .unwrap();
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

#[cfg(debug_assertions)]
#[test]
fn bounded_drain_production_completes_accepted_barrier_before_exit() {
    let mut server = ProductionServer::spawn_with_accept_barrier(3);
    let mut barrier = accepted_request_barrier(&server);
    server.arm_accept_barrier();
    let mut rejected =
        TcpStream::connect(server.addr).expect("connect guarded post-signal request");
    rejected
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    rejected
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .unwrap();
    server.wait_for_guarded_accept();
    server.signal(libc::SIGINT);
    server.release_guarded_accept();
    server.wait_log("shutting down (signal received", Duration::from_secs(2));
    server.assert_alive();
    let mut rejected_response = String::new();
    if let Err(error) = rejected.read_to_string(&mut rejected_response) {
        assert_eq!(error.kind(), std::io::ErrorKind::ConnectionReset);
    }
    assert!(
        rejected_response.is_empty(),
        "post-signal accepted socket reached a handler: {rejected_response:?}"
    );

    barrier.write_all(b"\r\n").unwrap();
    let mut response = String::new();
    barrier.read_to_string(&mut response).unwrap();
    assert!(
        response.contains("200 OK")
            && response.ends_with(r#"{"engine_loaded":false,"status":"ok"}"#),
        "barrier response did not complete: {response:?}"
    );
    server.wait_success(PROCESS_EXIT_TIMEOUT);
    assert!(server.log_text().contains("server stopped"));
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

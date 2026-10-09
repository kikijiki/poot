//! The one scheduling loop served end to end on a device: the production `poot-serve` binary on a tiny
//! GGUF, driven over HTTP. Each row runs on wgpu and on ROCm (`_wgpu` / `_rocm`), and a lane selects it
//! by that suffix. Nothing of the loop is reimplemented here; the tests read what the server answers and
//! what its `/metrics` reports.

use poot_test_util::contained_child::ContainChild;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[path = "support/tiny_gguf.rs"]
mod tiny_gguf;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(120);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Read at runtime, not baked in with `env!`: a shared compile cache would serve another worktree's path.
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

struct Server {
    child: Child,
    addr: SocketAddr,
    model: PathBuf,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.model);
    }
}

impl Server {
    fn spawn(backend: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let model = std::env::temp_dir().join(format!(
            "poot-served-loop-{backend}-{}-{nanos}.gguf",
            std::process::id()
        ));
        tiny_gguf::write_tiny_deepseek2_gguf(&model, 64);
        let addr = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let child = Command::new(poot_serve_bin())
            .contained()
            .arg(&model)
            .arg(addr.to_string())
            .args(["--backend", backend])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn poot-serve");
        let mut server = Self { child, addr, model };
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Some(status) = server.child.try_wait().unwrap() {
                panic!("poot-serve exited during startup: {status}");
            }
            if let Ok((200, _)) = server.request("GET", "/health", "") {
                return server;
            }
            assert!(Instant::now() < deadline, "poot-serve did not become ready");
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// One request on its own connection: the status and the body to end of stream.
    fn request(&self, method: &str, path: &str, body: &str) -> std::io::Result<(u16, String)> {
        let mut stream = TcpStream::connect_timeout(&self.addr, Duration::from_secs(2))?;
        stream.set_read_timeout(Some(REQUEST_TIMEOUT))?;
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Connection: close\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )?;
        let mut response = String::new();
        stream.read_to_string(&mut response)?;
        let status = response
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let body = response
            .split_once("\r\n\r\n")
            .map_or_else(String::new, |(_, body)| body.to_string());
        Ok((status, body))
    }

    fn complete(&self, body: &serde_json::Value) -> serde_json::Value {
        let (status, text) = self
            .request("POST", "/v1/completions", &body.to_string())
            .expect("completions request");
        assert_eq!(status, 200, "{text}");
        serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}"))
    }

    fn metrics(&self) -> serde_json::Value {
        let (status, text) = self.request("GET", "/metrics", "").expect("metrics");
        assert_eq!(status, 200);
        serde_json::from_str(&text).expect("metrics json")
    }
}

fn text_of(response: &serde_json::Value) -> String {
    response["choices"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("no completion text: {response}"))
        .to_string()
}

fn greedy(prompt: &str, max_tokens: usize) -> serde_json::Value {
    serde_json::json!({ "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0 })
}

/// SC-003: a streamed and a non-streamed request both complete. The non-streamed one answers with its
/// text and a finish reason; the streamed one sends its chunks, the usage receipt and `[DONE]`.
/// Mutation: drop the terminal event of a non-streamed request; it never completes and the row goes red.
fn streamed_and_non_streamed_requests_complete(backend: &str) {
    let server = Server::spawn(backend);
    let plain = server.complete(&greedy("a", 6));
    assert!(
        plain["choices"][0]["finish_reason"].is_string(),
        "no finish reason: {plain}"
    );
    assert_eq!(plain["usage"]["completion_tokens"], 6, "{plain}");

    let mut body = greedy("a", 6);
    body["stream"] = true.into();
    let (status, stream) = server
        .request("POST", "/v1/completions", &body.to_string())
        .expect("streamed request");
    assert_eq!(status, 200, "{stream}");
    assert!(
        stream.contains("data: [DONE]"),
        "the stream never ended: {stream}"
    );
    let chunks = stream.matches("\"object\":\"text_completion\"").count();
    assert!(chunks >= 6, "one chunk per token, got {chunks}: {stream}");
}

/// SC-002 (R476-001): a sampled request (temperature 1, fixed seed) returns the sampler's tokens, not the
/// greedy argmax, and a guided request is held to its choice. The sampled text is reproducible for one
/// seed and differs from the greedy text for some seed; the guided text is exactly the choice, which the
/// unconstrained greedy text is not. Mutation: take the greedy argmax in the loop's sampler head
/// (replace the job's sampler with `Sampler::greedy()` when the slot opens); every seed gives the greedy
/// text, the guided request loses its constraint, and the row goes red.
fn sampled_and_guided_requests_return_the_samplers_tokens(backend: &str) {
    let server = Server::spawn(backend);
    let greedy_text = text_of(&server.complete(&greedy("a", 8)));

    let sampled = |seed: u64| {
        text_of(&server.complete(&serde_json::json!({
            "prompt": "a", "max_tokens": 8, "temperature": 1.0, "seed": seed
        })))
    };
    assert_eq!(sampled(3), sampled(3), "one seed, one sampled text");
    let differs = (0..16).any(|seed| sampled(seed) != greedy_text);
    assert!(
        differs,
        "no seed in 0..16 left the greedy text {greedy_text:?}: the sampler is not used"
    );

    // A one-token choice the greedy run does not start with: the constraint, not luck, makes the text equal
    // it. One token, because the fixture's vocabulary holds no end-of-sequence token for the constraint to
    // finish on.
    let first_greedy = text_of(&server.complete(&greedy("a", 1)));
    let choice = ["ab", "c", "b"]
        .into_iter()
        .find(|choice| *choice != first_greedy)
        .unwrap();
    let guided = text_of(&server.complete(&serde_json::json!({
        "prompt": "a", "max_tokens": 1, "temperature": 0.0, "guided_choice": [choice]
    })));
    assert_eq!(guided, choice, "greedy started with {first_greedy:?}");
}

/// SC-004 (R473-007): batch composition changes mid-run, a request admitted while others decode and
/// others retiring in between, and the driver's compile count stays where the warm-up left it. Mutation:
/// key the step entries on the active-row count instead of the capacity rows; the compile counter moves
/// (or the step is refused) and the row goes red.
fn composition_changes_keep_the_compile_count_constant(backend: &str) {
    let server = Server::spawn(backend);
    server.complete(&greedy("a", 4)); // warm-up
    let warm = server.metrics()["engine_compiles"].as_u64().unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = [(3usize, 0u64), (12, 0), (7, 1), (20, 2), (2, 3)]
            .into_iter()
            .enumerate()
            .map(|(index, (tokens, seed))| {
                let server = &server;
                scope.spawn(move || {
                    // Staggered starts: later requests join a batch that is already decoding.
                    std::thread::sleep(Duration::from_millis(15 * index as u64));
                    let mut body = greedy("ab", tokens);
                    if seed > 0 {
                        body["temperature"] = 1.0.into();
                        body["seed"] = seed.into();
                    }
                    server.complete(&body);
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("concurrent request");
        }
    });
    server.complete(&greedy("a", 4));
    let after = server.metrics()["engine_compiles"].as_u64().unwrap();
    assert_eq!(
        after, warm,
        "admitting and retiring requests compiled a program"
    );
}

/// SC-007 (dense half, served): a finished request's prompt prefix is registered through
/// `Driver::release`, and a second request over the same prompt reuses it and returns the same text as the
/// cold run. (The recurrent-state half, `Skipped(RecurrentState)`, is the driver's own row in `poot-llm`
/// and the fake-engine row `a_finished_request_is_released_and_a_recurrent_model_registers_no_prefix`.)
/// Mutation: skip the `ServeEngine::release` call on finish; nothing registers and the row goes red.
fn a_dense_prefix_is_registered_on_release_and_reused(backend: &str) {
    let server = Server::spawn(backend);
    let prompt = "a".repeat(poot_llm::driver::block_table::BLOCK_SIZE * 2 + 1);
    let cold = text_of(&server.complete(&greedy(&prompt, 4)));
    let metrics = server.metrics();
    assert_eq!(
        metrics["prefix_registered_tokens"].as_u64().unwrap(),
        (poot_llm::driver::block_table::BLOCK_SIZE * 2) as u64,
        "the finished prompt's whole blocks are registered: {metrics}"
    );
    let warm = text_of(&server.complete(&greedy(&prompt, 4)));
    assert_eq!(warm, cold, "the reused prefix returns the cold run's text");
    let metrics = server.metrics();
    assert!(
        metrics["prefix_cache_hit_rate"].as_f64().unwrap() > 0.0,
        "the second request found its prefix resident: {metrics}"
    );
}

#[test]
fn served_streamed_and_non_streamed_requests_complete_wgpu() {
    streamed_and_non_streamed_requests_complete("wgpu");
}

#[test]
fn served_streamed_and_non_streamed_requests_complete_rocm() {
    streamed_and_non_streamed_requests_complete("rocm");
}

#[test]
fn served_sampled_and_guided_requests_return_the_samplers_tokens_wgpu() {
    sampled_and_guided_requests_return_the_samplers_tokens("wgpu");
}

#[test]
fn served_sampled_and_guided_requests_return_the_samplers_tokens_rocm() {
    sampled_and_guided_requests_return_the_samplers_tokens("rocm");
}

#[test]
fn served_composition_changes_keep_the_compile_count_constant_wgpu() {
    composition_changes_keep_the_compile_count_constant("wgpu");
}

#[test]
fn served_composition_changes_keep_the_compile_count_constant_rocm() {
    composition_changes_keep_the_compile_count_constant("rocm");
}

#[test]
fn served_dense_prefix_is_registered_on_release_and_reused_wgpu() {
    a_dense_prefix_is_registered_on_release_and_reused("wgpu");
}

#[test]
fn served_dense_prefix_is_registered_on_release_and_reused_rocm() {
    a_dense_prefix_is_registered_on_release_and_reused("rocm");
}

//! Served multi-device requests through the one scheduling loop (POOT-750), on two NVIDIA GPUs.
//!
//! The production `poot-serve` binary opens a two-GPU placement (`--placement pipeline:0,1` or
//! `tensor:0,1`) over a tiny qwen2 GGUF and is driven over HTTP. Nothing of the loop is reimplemented
//! here: the rows read what the server answers, what it logged as its engine plan, and compare it with
//! the one-device driver's tokens and log-probabilities. A row skips with fewer than two NVIDIA GPUs
//! and fails under `POOT_REQUIRE_PTX_2GPU=1`.
//!
//! Admission of the same placements (exact per-device budgets, per-rank refusals) is proven in-crate over
//! the engine plan (`engine_plan/tests.rs`), where the production `EnginePlan::build` runs over fake
//! devices whose capacity can be set to an exact figure.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::num::{NonZeroU64, NonZeroUsize};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use poot_executor::{Collectives, Coordinator};
use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
use poot_llm::Sampler;
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Generation, LoadOptions,
    ModelHandle, PreparedSetLimits, open_executor, open_group, program_retention,
};
use poot_load::gguf::{GgufValue, write_gguf};
use poot_models::registry::Registry;
use poot_test_util::contained_child::ContainChild;

/// A pod JIT-compiles every kernel on first use, so a cold start is slow.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// The most a served log-probability may differ from the one-device driver's: the same math summed in
/// another order across devices.
const LOGPROB_TOLERANCE: f32 = 1e-3;

const PROMPTS: [&str; 3] = ["a", "abc", "gfedcba"];
const MAX_NEW: usize = 8;

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const LAYERS: usize = 2;
const INTER: usize = 16;
const VOCAB: usize = 8;
const CONTEXT: usize = 64;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("nonzero")
}

// ---- the fixture -------------------------------------------------------------------------------

/// A tiny dense qwen2 GGUF every placement can cut: two layers, two heads (each a whole share per
/// tensor-parallel rank), an even intermediate size and F32 weights on a grid of distinct logits.
fn write_qwen2_gguf(path: &Path) {
    // A xorshift fill: non-degenerate magnitudes.
    fn fill(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = Vec::new();
    let mut seed = 7u64;
    let mut push = |name: String, dims: &[usize], kind: char| {
        seed = seed.wrapping_add(0x9E37_79B9);
        let n: usize = dims.iter().product();
        let bytes: Vec<u8> = fill(seed, n)
            .into_iter()
            .map(|v| match kind {
                'g' => 1.0 + v * 0.05,
                'b' => v * 0.1,
                _ => v * 2.0,
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        tensors.push((name, dims.iter().map(|&d| d as u64).collect(), 0, bytes));
    };
    push("token_embd.weight".into(), &[HIDDEN, VOCAB], 'w');
    for layer in 0..LAYERS {
        let blk = |name: &str| format!("blk.{layer}.{name}");
        push(blk("attn_norm.weight"), &[HIDDEN], 'g');
        push(blk("ffn_norm.weight"), &[HIDDEN], 'g');
        for projection in ["attn_q", "attn_k", "attn_v"] {
            push(blk(&format!("{projection}.weight")), &[HIDDEN, HIDDEN], 'w');
            push(blk(&format!("{projection}.bias")), &[HIDDEN], 'b');
        }
        push(blk("attn_output.weight"), &[HIDDEN, HIDDEN], 'w');
        push(blk("ffn_gate.weight"), &[HIDDEN, INTER], 'w');
        push(blk("ffn_up.weight"), &[HIDDEN, INTER], 'w');
        push(blk("ffn_down.weight"), &[INTER, HIDDEN], 'w');
    }
    push("output_norm.weight".into(), &[HIDDEN], 'g');
    push("output.weight".into(), &[HIDDEN, VOCAB], 'w');

    let u = |value: usize| GgufValue::U32(value as u32);
    let kvs: Vec<(&str, GgufValue)> = vec![
        ("general.architecture", GgufValue::Str("qwen2".into())),
        ("qwen2.embedding_length", u(HIDDEN)),
        ("qwen2.block_count", u(LAYERS)),
        ("qwen2.attention.head_count", u(HEADS)),
        ("qwen2.attention.head_count_kv", u(HEADS)),
        ("qwen2.feed_forward_length", u(INTER)),
        ("qwen2.context_length", u(CONTEXT)),
        // No generated token is the end of sequence: every request runs to its length.
        ("tokenizer.ggml.eos_token_id", GgufValue::U32(99)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                ["a", "b", "c", "d", "e", "f", "g", "ab"]
                    .into_iter()
                    .map(|piece| GgufValue::Str(piece.into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("a b".into())]),
        ),
    ];
    let refs: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dims, ty, bytes)| (name.as_str(), dims.clone(), *ty, bytes.clone()))
        .collect();
    std::fs::write(path, write_gguf(&kvs, &refs)).expect("write the tiny qwen2 GGUF");
}

/// The fixture on disk for one test, removed on drop.
struct Checkpoint(PathBuf);

impl Checkpoint {
    fn write(label: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "poot-served-multi-{label}-{}-{nanos}.gguf",
            std::process::id()
        ));
        write_qwen2_gguf(&path);
        Self(path)
    }

    fn handle(&self) -> Arc<ModelHandle> {
        let registry = Registry::builtin().expect("the builtin registry");
        Arc::new(
            ModelHandle::load(&self.0, &registry, &LoadOptions::default())
                .expect("the fixture loads"),
        )
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

// ---- the devices -------------------------------------------------------------------------------

/// A coordinator over NVIDIA GPUs 0 and 1, or a skip: `None` without two devices, and a panic when the
/// second device is required (`POOT_REQUIRE_PTX_2GPU`).
fn open_pair() -> Option<Coordinator<impl Collectives + 'static>> {
    match open_group(BackendChoice::Ptx, &[0, 1]) {
        Ok(coordinator) => Some(coordinator),
        Err(error) => {
            assert!(
                std::env::var_os("POOT_REQUIRE_PTX_2GPU").is_none(),
                "two NVIDIA devices are required (POOT_REQUIRE_PTX_2GPU) and unavailable: {error}"
            );
            eprintln!("SKIP: two NVIDIA devices are unavailable: {error}");
            None
        }
    }
}

/// Whether two NVIDIA GPUs open, probed in this process and released before a server opens them.
fn two_gpus_available() -> bool {
    open_pair().is_some()
}

/// `handle`'s tokens on one NVIDIA GPU through the driver: the reference every placement must match.
fn driver_generation(
    handle: &Arc<ModelHandle>,
    prompt: &str,
    sampler: Sampler,
) -> (Generation, String) {
    let tokens = handle.text().encode(prompt).expect("the prompt encodes");
    let executor = open_executor(BackendChoice::Ptx).expect("GPU 0 opens");
    let mut driver = Driver::new(
        Arc::clone(handle),
        executor,
        driver_options(handle, nz(tokens.len() + MAX_NEW)),
    )
    .expect("the driver builds");
    let mut text = String::new();
    let generation = driver
        .generate(
            GenerateRequest {
                prompt: tokens,
                max_new: MAX_NEW,
                sampler,
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, piece: &str| {
                text.push_str(piece);
                ControlFlow::Continue(())
            },
        )
        .expect("the driver generates");
    (generation, text)
}

fn driver_options(handle: &ModelHandle, capacity: NonZeroUsize) -> DriverOptions {
    let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
    let granule = handle.config().prefill_granule;
    // A multiple of the granule within the window, never below one granule.
    let chunk = nz((capacity.get().min(128) / granule).max(1) * granule.get());
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity,
        prefill_chunk: chunk,
        max_trace_tokens: chunk,
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

// ---- the served process ------------------------------------------------------------------------

/// Read at runtime, not baked in with `env!`: a shared compile cache would serve another worktree's path.
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

struct Server {
    child: Child,
    addr: SocketAddr,
    log: PathBuf,
    _model: Checkpoint,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_file(&self.log);
    }
}

impl Server {
    /// `poot-serve` on the ptx backend with `placement` (`single`, `pipeline:0,1`, `tensor:0,1`), once
    /// it answers `/health`. Its log (stdout) is kept for the engine-plan line.
    fn spawn(placement: &str) -> Self {
        let model = Checkpoint::write(&placement.replace([':', ','], "-"));
        let addr = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let log = model.0.with_extension("log");
        let child = Command::new(poot_serve_bin())
            .contained()
            .arg(&model.0)
            .arg(addr.to_string())
            .args(["--backend", "ptx", "--placement", placement])
            .stdout(std::fs::File::create(&log).expect("the server log"))
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn poot-serve");
        let mut server = Self {
            child,
            addr,
            log,
            _model: model,
        };
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        loop {
            if let Some(status) = server.child.try_wait().unwrap() {
                panic!(
                    "poot-serve exited during startup: {status}\n{}",
                    server.log_text()
                );
            }
            if let Ok((200, _)) = server.request("GET", "/health", "") {
                return server;
            }
            assert!(
                Instant::now() < deadline,
                "poot-serve did not become ready:\n{}",
                server.log_text()
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn log_text(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
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

    /// The engine plan line the server logged once it was prepared: the devices and the placement it
    /// actually runs, as the process itself states them.
    fn engine_plan(&self) -> String {
        self.log_text()
            .lines()
            .find(|line| line.contains("engine plan"))
            .unwrap_or_else(|| panic!("no engine plan line in the log:\n{}", self.log_text()))
            .to_string()
    }
}

fn text_of(response: &serde_json::Value) -> String {
    response["choices"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("no completion text: {response}"))
        .to_string()
}

fn greedy_body(prompt: &str) -> serde_json::Value {
    serde_json::json!({
        "prompt": prompt, "max_tokens": MAX_NEW, "temperature": 0.0, "logprobs": 1
    })
}

fn sampled_body(prompt: &str, seed: u64) -> serde_json::Value {
    serde_json::json!({
        "prompt": prompt, "max_tokens": MAX_NEW, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
        "seed": seed
    })
}

/// A greedy request of every prompt, served through `placement`, equals the one-device driver's text,
/// token count and per-token log-probabilities.
fn served_greedy_requests_match_the_driver(placement: &str, expect_in_plan: &[&str]) {
    if !two_gpus_available() {
        return;
    }
    let server = Server::spawn(placement);
    let plan = server.engine_plan();
    for expected in expect_in_plan {
        assert!(
            plan.contains(expected),
            "the server did not run {expected:?}: {plan}"
        );
    }
    let handle = Checkpoint::write("reference").handle();
    for prompt in PROMPTS {
        let (reference, reference_text) =
            driver_generation(&handle, prompt, Sampler::greedy().with_logprobs(1));
        assert_eq!(
            reference.tokens.len(),
            MAX_NEW,
            "{prompt:?}: the reference runs to its length"
        );
        let served = server.complete(&greedy_body(prompt));
        assert_eq!(
            text_of(&served),
            reference_text,
            "{prompt:?}: served text differs from the driver's tokens {:?}",
            reference.tokens
        );
        assert_eq!(
            served["usage"]["completion_tokens"], MAX_NEW,
            "{prompt:?}: {served}"
        );
        let content = served["choices"][0]["logprobs"]["content"]
            .as_array()
            .unwrap_or_else(|| panic!("{prompt:?}: no logprobs: {served}"));
        assert_eq!(content.len(), reference.logprobs.len(), "{prompt:?}");
        for (position, (got, want)) in content.iter().zip(&reference.logprobs).enumerate() {
            let got = got["logprob"].as_f64().expect("a logprob") as f32;
            assert!(
                got.is_finite() && (got - want.logprob).abs() <= LOGPROB_TOLERANCE,
                "{prompt:?} token {position}: served logprob {got}, driver {}",
                want.logprob
            );
        }
    }
}

/// POOT-451 SC-005 / POOT-750 SC-001: a served two-device pipeline request completes through the one
/// loop and matches the driver's tokens and log-probabilities. The engine plan the server logs names
/// both devices and a pipeline cut after layer 0, so a silent fallback to one device is red.
/// Mutation: skip the boundary transfer of one stage (the coordinator's peer copy of a stage's exported
/// value); the second stage reads zeros and the served text or a log-probability diverges.
#[test]
fn a_served_two_device_pipeline_request_matches_the_drivers_tokens_ptx() {
    served_greedy_requests_match_the_driver(
        "pipeline:0,1",
        &["DeviceId(0), DeviceId(1)", "Pipeline { after_layers: [0] }"],
    );
}

/// POOT-750 SC-001: a served tensor-parallel request completes through the one loop over the paged KV
/// sharded by KV heads and matches the driver's tokens and log-probabilities. The engine plan names
/// both devices and the Megatron rule.
/// Mutation: skip the boundary transfer of one rank's partial sum; the combined activation is wrong
/// and the served text or a log-probability diverges.
#[test]
fn a_served_tensor_parallel_request_matches_the_drivers_tokens_ptx() {
    served_greedy_requests_match_the_driver(
        "tensor:0,1",
        &["DeviceId(0), DeviceId(1)", "Tensor { rule: Megatron }"],
    );
}

/// POOT-750 SC-002 (R477-006): a sampled tensor-parallel request returns the sampler's tokens, not the
/// greedy argmax. The sampler's tokens are the one-device server's for the same seed (the same host
/// sampling over the same logits); they are reproducible for one seed and differ from the greedy text
/// for some seed. Mutation: take the greedy argmax in the tensor-parallel request path; every seed gives
/// the greedy text and the row goes red.
#[test]
fn a_sampled_tensor_parallel_request_returns_the_samplers_tokens_ptx() {
    if !two_gpus_available() {
        return;
    }
    const SEEDS: std::ops::Range<u64> = 0..12;
    let prompt = PROMPTS[0];
    let single = Server::spawn("single");
    let reference: Vec<String> = SEEDS
        .map(|seed| text_of(&single.complete(&sampled_body(prompt, seed))))
        .collect();
    let greedy_reference = text_of(&single.complete(&greedy_body(prompt)));
    drop(single);
    assert!(
        reference.iter().any(|text| *text != greedy_reference),
        "the fixture's sampled text never leaves the greedy text {greedy_reference:?}: {reference:?}"
    );

    let tensor = Server::spawn("tensor:0,1");
    assert!(
        tensor.engine_plan().contains("Tensor { rule: Megatron }"),
        "{}",
        tensor.engine_plan()
    );
    let greedy_tensor = text_of(&tensor.complete(&greedy_body(prompt)));
    assert_eq!(greedy_tensor, greedy_reference, "the greedy control");
    let sampled: Vec<String> = SEEDS
        .map(|seed| text_of(&tensor.complete(&sampled_body(prompt, seed))))
        .collect();
    assert_eq!(
        sampled, reference,
        "a sampled tensor-parallel request must return the one-device sampler's tokens per seed"
    );
    assert_eq!(
        text_of(&tensor.complete(&sampled_body(prompt, 3))),
        sampled[3],
        "one seed, one sampled text"
    );
    assert!(
        sampled.iter().any(|text| *text != greedy_tensor),
        "no seed left the greedy text {greedy_tensor:?}: the sampler is not used"
    );
}

//! A served run per family group, through the real startup path: the production `poot-serve` binary on a
//! checkpoint written from a family's registry fixture, driven over HTTP, against the tokens
//! `Driver::generate` produces for the same request on the same backend in this process.
//!
//! The groups (POOT-754): the five exact families, Gemma4, Qwen3-Next, the dense group and the MoE group
//! on wgpu and ROCm; PTX (a pod batch) and raw Vulkan serve the dense fixture, greedy (PTX also sampled).
//! Raw Vulkan reports its budget from `VK_EXT_memory_budget`, so startup admits it. A row is named
//! `served_<family>_<backend>` so a lane selects it by the suffix.
//! [`every_generating_family_is_in_a_served_group`] closes the lists over `Registry::builtin()`: a family
//! added to the registry must join a group before this binary passes.

use poot_test_util::contained_child::ContainChild;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_llm::driver::{
    BackendChoice, DraftPolicy, Driver, DriverOptions, GenerateRequest, Layout, LoadOptions,
    ModelHandle, PreparedSetLimits, Serving, open_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::model::ModelOutput;
use poot_models::registry::{Fixture, Registry};
use poot_quant::weights::WeightEntry;
use poot_runtime_common::DeviceBackend;
use poot_test_util::per_backend;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// No end-of-sequence token is reachable, so every request runs to its length.
const NO_EOS: u32 = 999;
const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];
/// A prompt that repeats itself: prompt lookup finds an earlier match of its last two tokens.
const REPEATING_PROMPT: [u32; 8] = [3, 17, 3, 17, 3, 17, 3, 17];
const NEW_TOKENS: usize = 24;
/// The positions of the in-process reference driver.
const CAPACITY: usize = 64;

/// Read at runtime, not baked in with `env!`: a shared compile cache would serve another worktree's path.
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

// ---------------------------------------------------------------------------------------------
// The groups.
// ---------------------------------------------------------------------------------------------

/// Registry fixtures served on wgpu and ROCm, by group. `family` or `family/variant` as
/// `device_tests::RECORDED` names them.
const EXACT: &[&str] = &["deepseek4", "glm53_flash", "minimax_m2", "qwen35"];
/// Families the server cannot serve yet: their paged KV layout is undecided, so startup is the typed
/// refusal (`a_paged_pool_refuses_the_qwen4exp_indexer_startup_*`). POOT-1148 owns the paging
/// granule of the indexer's block-key cache; its landing moves `qwen4exp` back into [`EXACT`].
const PAGED_REFUSED: &[&str] = &["qwen4exp"];
const GEMMA4: &[&str] = &["gemma4"];
const QWEN3_NEXT: &[&str] = &["qwen3next"];
const DENSE: &[&str] = &[
    "qwen2",
    "llama",
    "qwen3",
    "gemma3",
    "gemma2",
    "granite",
    "olmo2",
    "bloom",
    "mpt",
    "smollm3",
    "phi3",
    "qwen2_5_vl",
    "idefics3",
];
/// Fixture stores: real packed MoE serving waits for POOT-762 (R-632-8).
const MOE: &[&str] = &[
    "qwen3moe",
    "olmoe",
    "llama/mixtral",
    "granite/granitemoe",
    "gpt_oss",
];
/// MLA attention and the hybrid state-space family.
const MLA_AND_HYBRID: &[&str] = &["deepseek2", "deepseek3", "deepseek32", "nemotron_h"];

/// The fixture a name selects; `family/variant` is one of the family's option fixtures.
fn fixture_of(name: &str) -> Fixture {
    let registry = Registry::builtin().unwrap();
    let (family, variant) = name.split_once('/').unwrap_or((name, ""));
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == family)
        .unwrap_or_else(|| panic!("no builtin family {family}"));
    match variant {
        "" => poot_models::test_support::family_fixtures::base(entry.family),
        variant => poot_models::test_support::family_fixtures::variant(entry.family, variant),
    }
}

/// Every family of the registry whose output is logits belongs to a served group, so a new family
/// cannot join without a served row. Mutation: drop `phi3` from [`DENSE`]; the registry's phi3 entry is
/// in no group and the row goes red.
#[test]
fn every_generating_family_is_in_a_served_group() {
    let groups = [
        EXACT,
        PAGED_REFUSED,
        GEMMA4,
        QWEN3_NEXT,
        DENSE,
        MOE,
        MLA_AND_HYBRID,
    ];
    let registry = Registry::builtin().unwrap();
    let mut missing = Vec::new();
    for entry in registry.entries() {
        let fixture = poot_models::test_support::family_fixtures::base(entry.family);
        let model = registry
            .build(&fixture.raw(), &fixture.store)
            .unwrap_or_else(|error| panic!("{}: {error}", entry.family))
            .model;
        if !matches!(
            model.schema().step().map(|program| program.output()),
            Some(ModelOutput::Logits { .. })
        ) {
            continue;
        }
        let family = entry.family.as_str();
        let grouped = groups
            .iter()
            .flat_map(|group| group.iter())
            .any(|name| name.split('/').next() == Some(family));
        if !grouped {
            missing.push(family);
        }
    }
    assert!(
        missing.is_empty(),
        "generating families in no served group: {missing:?}"
    );
}

// ---------------------------------------------------------------------------------------------
// The checkpoint a fixture is served from.
// ---------------------------------------------------------------------------------------------

fn vocab_of(config: &serde_json::Value) -> usize {
    config
        .get("vocab_size")
        .or_else(|| config.pointer("/text_config/vocab_size"))
        .and_then(serde_json::Value::as_u64)
        .expect("a fixture config states its vocabulary") as usize
}

/// A word-level tokenizer whose word `t<i>` is token `i`, so a prompt of ids is the text of its words.
fn tokenizer_json(vocab: usize) -> serde_json::Value {
    let words: serde_json::Map<String, serde_json::Value> = (0..vocab)
        .map(|id| (format!("t{id}"), serde_json::json!(id)))
        .collect();
    serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": words, "unk_token": "t0" }
    })
}

fn prompt_text(prompt: &[u32]) -> String {
    prompt
        .iter()
        .map(|id| format!("t{id}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The fixture written the way a downloaded checkpoint is laid out, with no reachable EOS.
fn write_checkpoint(name: &str) -> poot_test_util::UniqueTempPath {
    write_fixture(fixture_of(name))
}

fn write_fixture(fixture: Fixture) -> poot_test_util::UniqueTempPath {
    let dir = poot_test_util::unique_temp_path("served_groups_checkpoint");
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    // A vision-language checkpoint keeps its text model's config in `text_config`.
    if let Some(text) = config.get_mut("text_config") {
        text["eos_token_id"] = serde_json::json!(NO_EOS);
    }
    let tokenizer = tokenizer_json(vocab_of(&config));
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    if let Some(preprocessor) = &fixture.preprocessor {
        std::fs::write(
            dir.join("preprocessor_config.json"),
            preprocessor.to_string(),
        )
        .unwrap();
    }
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
    dir
}

/// The ids the server's encoder makes of the words of `prompt`: the request the driver is asked for is the
/// one the server builds from the same text (a family may add a beginning-of-sequence token).
fn encoded(dir: &Path, prompt: &[u32]) -> Vec<u32> {
    load(dir)
        .text()
        .encode(&prompt_text(prompt))
        .expect("encode the prompt")
}

fn load(dir: &Path) -> Arc<ModelHandle> {
    Arc::new(
        ModelHandle::load(dir, &Registry::builtin().unwrap(), &LoadOptions::default())
            .expect("load the fixture checkpoint"),
    )
}

// ---------------------------------------------------------------------------------------------
// The reference: `Driver::generate` in this process.
// ---------------------------------------------------------------------------------------------

/// What `Driver::generate` produced: the tokens and the pieces its sink saw.
struct Reference {
    tokens: Vec<u32>,
    text: String,
}

fn driver_options() -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: nz(CAPACITY),
        prefill_chunk: nz(8),
        trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(CAPACITY)),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

/// The requirement variable's backend of a row's choice.
fn backend_of(choice: BackendChoice) -> DeviceBackend {
    match choice {
        BackendChoice::Wgpu => DeviceBackend::Wgpu,
        BackendChoice::Rocm => DeviceBackend::Rocm,
        BackendChoice::Ptx => DeviceBackend::Ptx,
        BackendChoice::Vulkan => DeviceBackend::Vulkan,
        BackendChoice::Auto => unreachable!("rows name their backend"),
    }
}

/// The server's `--backend` flag value of a row's choice.
fn flag_of(choice: BackendChoice) -> &'static str {
    match choice {
        BackendChoice::Wgpu => "wgpu",
        BackendChoice::Rocm => "rocm",
        BackendChoice::Ptx => "ptx",
        BackendChoice::Vulkan => "vulkan",
        BackendChoice::Auto => unreachable!("rows name their backend"),
    }
}

/// Generate `prompt` once with each of `samplers`, on one driver over `choice`'s device (a ROCm executor
/// cannot be reopened in a process), or `None` where the device is absent (a skip).
fn references(
    choice: BackendChoice,
    backend: DeviceBackend,
    dir: &Path,
    prompt: &[u32],
    samplers: Vec<Sampler>,
) -> Option<Vec<Reference>> {
    let executor = poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice))?;
    let mut driver = Driver::new(load(dir), executor, driver_options())
        .expect("a driver")
        .serve(Layout::Contiguous);
    Some(
        samplers
            .into_iter()
            .map(|sampler| {
                let mut text = String::new();
                let generation = driver
                    .generate(
                        GenerateRequest {
                            prompt: prompt.to_vec(),
                            max_new: NEW_TOKENS,
                            sampler,
                            stops: Vec::new(),
                            ignore_eos: true,
                        },
                        &mut |_: u32, piece: &str| {
                            text.push_str(piece);
                            GenerationControl::Continue(())
                        },
                    )
                    .unwrap_or_else(|error| panic!("{error}"));
                Reference {
                    tokens: generation.tokens,
                    text,
                }
            })
            .collect(),
    )
}

/// [`references`] for one sampler.
fn reference(
    choice: BackendChoice,
    backend: DeviceBackend,
    dir: &Path,
    prompt: &[u32],
    sampler: Sampler,
) -> Option<Reference> {
    references(choice, backend, dir, prompt, vec![sampler])?.pop()
}

// ---------------------------------------------------------------------------------------------
// The server.
// ---------------------------------------------------------------------------------------------

struct Server {
    child: Child,
    addr: SocketAddr,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// `poot-serve` on a free port with the model, backend and extra flags, its stderr as `stderr`.
fn command(dir: &Path, backend: &str, extra: &[String], stderr: Stdio) -> (Child, SocketAddr) {
    let addr = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let child = Command::new(poot_serve_bin())
        .contained()
        .arg(dir)
        .arg(addr.to_string())
        .args(["--backend", backend])
        .args(extra)
        .stdout(Stdio::null())
        .stderr(stderr)
        .spawn()
        .expect("spawn poot-serve");
    (child, addr)
}

/// A startup `poot-serve` refuses: it exits unsuccessfully without ever answering `/health`, and this is
/// what it printed. The listener binds only after the engine plan is built, so an exit here is a
/// refusal before the server listened.
fn refused(dir: &Path, backend: &str, extra: &[String]) -> String {
    let (mut child, addr) = command(dir, backend, extra, Stdio::piped());
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    let status = loop {
        assert!(
            TcpStream::connect_timeout(&addr, Duration::from_millis(20)).is_err(),
            "poot-serve listened instead of refusing"
        );
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("poot-serve neither served nor refused");
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(!status.success(), "poot-serve exited cleanly: {stderr}");
    stderr
}

impl Server {
    fn spawn(dir: &Path, backend: &str, extra: &[String]) -> Self {
        let (child, addr) = command(dir, backend, extra, Stdio::inherit());
        let mut server = Self { child, addr };
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

    /// A streamed completion: the text of its chunks in order, after the stream ended with `[DONE]`.
    fn stream(&self, mut body: serde_json::Value) -> String {
        body["stream"] = true.into();
        let (status, stream) = self
            .request("POST", "/v1/completions", &body.to_string())
            .expect("streamed request");
        assert_eq!(status, 200, "{stream}");
        assert!(
            stream.contains("data: [DONE]"),
            "the stream never ended: {stream}"
        );
        stream
            .split("data: ")
            .skip(1)
            .filter_map(|event| {
                let json = event.lines().next()?.trim();
                let chunk: serde_json::Value = serde_json::from_str(json).ok()?;
                Some(chunk["choices"][0]["text"].as_str()?.to_string())
            })
            .collect()
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

fn greedy(prompt: &[u32]) -> serde_json::Value {
    serde_json::json!({
        "prompt": prompt_text(prompt), "max_tokens": NEW_TOKENS, "temperature": 0.0
    })
}

// ---------------------------------------------------------------------------------------------
// SC-001: a served run per family group.
// ---------------------------------------------------------------------------------------------

/// SC-001: `name`'s fixture, served, completes a non-streamed and a streamed request and returns the
/// tokens `Driver::generate` returns for the same request (the generated text decodes the same ids, and
/// the stream's chunks add up to it). Mutation: skip `engine.commit` in `commit_tokens` for the group;
/// the sequence never advances, the served text diverges from (or never completes against) the driver's
/// and the row goes red.
fn served_family(name: &str, choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    let dir = write_checkpoint(name);
    let Some(expected) = reference(
        choice,
        backend,
        &dir,
        &encoded(&dir, &PROMPT),
        Sampler::greedy(),
    ) else {
        return;
    };
    assert_eq!(
        expected.tokens.len(),
        NEW_TOKENS,
        "{name}: reference length"
    );
    let text = load(&dir)
        .text()
        .decode(&expected.tokens)
        .expect("decode the reference tokens");
    let server = Server::spawn(&dir, flag, &[]);
    let plain = server.complete(&greedy(&PROMPT));
    assert_eq!(
        plain["usage"]["completion_tokens"], NEW_TOKENS,
        "{name}: {plain}"
    );
    assert_eq!(
        text_of(&plain),
        text,
        "{name} on {flag}: the served tokens differ from Driver::generate's {:?}",
        expected.tokens
    );
    assert_eq!(
        server.stream(greedy(&PROMPT)),
        expected.text,
        "{name} on {flag}: the streamed chunks differ from Driver::generate's pieces"
    );
}

per_backend!(
    BackendChoice,
    |choice| served_family("deepseek4", choice),
    served_deepseek4_wgpu,
    served_deepseek4_rocm,
    served_deepseek4_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("glm53_flash", choice),
    served_glm53_flash_wgpu,
    served_glm53_flash_rocm,
    served_glm53_flash_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("minimax_m2", choice),
    served_minimax_m2_wgpu,
    served_minimax_m2_rocm,
    served_minimax_m2_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("qwen35", choice),
    served_qwen35_wgpu,
    served_qwen35_rocm,
    served_qwen35_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("gemma4", choice),
    served_gemma4_wgpu,
    served_gemma4_rocm,
    served_gemma4_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("qwen3next", choice),
    served_qwen3next_wgpu,
    served_qwen3next_rocm,
    served_qwen3next_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("qwen2", choice),
    served_qwen2_wgpu,
    served_qwen2_rocm,
    served_qwen2_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("llama", choice),
    served_llama_wgpu,
    served_llama_rocm,
    served_llama_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("qwen3", choice),
    served_qwen3_wgpu,
    served_qwen3_rocm,
    served_qwen3_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("gemma3", choice),
    served_gemma3_wgpu,
    served_gemma3_rocm,
    served_gemma3_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("gemma2", choice),
    served_gemma2_wgpu,
    served_gemma2_rocm,
    served_gemma2_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("granite", choice),
    served_granite_wgpu,
    served_granite_rocm,
    served_granite_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("olmo2", choice),
    served_olmo2_wgpu,
    served_olmo2_rocm,
    served_olmo2_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("bloom", choice),
    served_bloom_wgpu,
    served_bloom_rocm,
    served_bloom_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("mpt", choice),
    served_mpt_wgpu,
    served_mpt_rocm,
    served_mpt_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("smollm3", choice),
    served_smollm3_wgpu,
    served_smollm3_rocm,
    served_smollm3_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("phi3", choice),
    served_phi3_wgpu,
    served_phi3_rocm,
    served_phi3_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("qwen2_5_vl", choice),
    served_qwen2_5_vl_wgpu,
    served_qwen2_5_vl_rocm,
    served_qwen2_5_vl_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("idefics3", choice),
    served_idefics3_wgpu,
    served_idefics3_rocm,
    served_idefics3_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("qwen3moe", choice),
    served_qwen3moe_wgpu,
    served_qwen3moe_rocm,
    served_qwen3moe_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("olmoe", choice),
    served_olmoe_wgpu,
    served_olmoe_rocm,
    served_olmoe_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("llama/mixtral", choice),
    served_mixtral_wgpu,
    served_mixtral_rocm,
    served_mixtral_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("granite/granitemoe", choice),
    served_granitemoe_wgpu,
    served_granitemoe_rocm,
    served_granitemoe_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("gpt_oss", choice),
    served_gpt_oss_wgpu,
    served_gpt_oss_rocm,
    served_gpt_oss_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("deepseek2", choice),
    served_deepseek2_wgpu,
    served_deepseek2_rocm,
    served_deepseek2_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("deepseek3", choice),
    served_deepseek3_wgpu,
    served_deepseek3_rocm,
    served_deepseek3_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("deepseek32", choice),
    served_deepseek32_wgpu,
    served_deepseek32_rocm,
    served_deepseek32_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_family("nemotron_h", choice),
    served_nemotron_h_wgpu,
    served_nemotron_h_rocm,
    served_nemotron_h_ptx
);

/// The dense fixture served on raw Vulkan, greedy (POOT-1081): the executor reports a live memory budget
/// from `VK_EXT_memory_budget`, so startup admits it and the served tokens equal `Driver::generate`'s.
/// Mutation: `VulkanDevice::free_memory_bytes` returns `None` and startup refuses (`UnknownBudget`), so
/// the server never listens.
#[test]
fn served_qwen2_vulkan() {
    served_family("qwen2", BackendChoice::Vulkan);
}

// ---------------------------------------------------------------------------------------------
// SC-002 (R477-006): a sampled request returns the sampler's tokens.
// ---------------------------------------------------------------------------------------------

/// A temperature request with a fixed seed returns the tokens `Driver::generate` draws with the same
/// sampler, and they are not the greedy tokens. Mutation: take the greedy argmax in the served path
/// (replace the job's sampler with `Sampler::greedy()` when the slot opens); the served text is the
/// greedy text and the row goes red.
fn served_sampled(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    const SEED: u64 = 7;
    let dir = write_checkpoint("qwen2");
    let sampler = || Sampler::new(1.0, 0, 1.0, SEED);
    let Some(mut both) = references(
        choice,
        backend,
        &dir,
        &encoded(&dir, &PROMPT),
        vec![sampler(), Sampler::greedy()],
    ) else {
        return;
    };
    let argmax = both.pop().unwrap();
    let drawn = both.pop().unwrap();
    assert_ne!(
        drawn.tokens, argmax.tokens,
        "the seed must draw something other than the argmax, or the row proves nothing"
    );
    let server = Server::spawn(&dir, flag, &[]);
    let mut body = greedy(&PROMPT);
    body["temperature"] = 1.0.into();
    body["seed"] = SEED.into();
    let served = server.complete(&body);
    let handle = load(&dir);
    assert_eq!(
        text_of(&served),
        handle.text().decode(&drawn.tokens).unwrap(),
        "{flag}: the served sampled tokens differ from the sampler's {:?} (argmax {:?})",
        drawn.tokens,
        argmax.tokens
    );
}

per_backend!(
    BackendChoice,
    served_sampled,
    served_sampled_request_returns_the_samplers_tokens_wgpu,
    served_sampled_request_returns_the_samplers_tokens_rocm,
    served_sampled_request_returns_the_samplers_tokens_ptx
);

// ---------------------------------------------------------------------------------------------
// SC-003: served speculative decoding.
// ---------------------------------------------------------------------------------------------

/// A served speculative request returns the `Off` tokens, and the metrics count what the driver reported.
/// A verified round returns its accepted drafts and one more token, a row that decoded without
/// speculating returns one, and the prompt's last prefill piece returns the first: so the tokens served
/// are `1 + rounds + accepted + skipped`. Recording a round's token count where its accepted count
/// belongs breaks that sum by one per round.
///
/// Mutations: accept every draft in the driver's verification (the tokens diverge from `Off`); record
/// `tokens.len()` instead of the driver's accepted count in the loop (the sum goes red).
fn served_speculative(choice: BackendChoice, draft: Draft) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    // The MTP policy needs a family with an MTP program: the GLM-5.3-Flash fixture, which is also
    // recurrent, so its rejecting rounds restore and replay under the server's own loop.
    let family = match draft {
        Draft::Mtp => "glm53_flash",
        Draft::Lookup | Draft::Model => "qwen2",
    };
    let dir = write_checkpoint(family);
    let Some(off) = reference(
        choice,
        backend,
        &dir,
        &encoded(&dir, &REPEATING_PROMPT),
        Sampler::greedy(),
    ) else {
        return;
    };
    let mut args = Vec::new();
    // The draft model is a different family over the same vocabulary, so it proposes tokens the target
    // sometimes rejects.
    let draft_dir;
    match draft {
        Draft::Lookup => args.extend(["--draft-lookup".to_string(), "2:4".to_string()]),
        Draft::Mtp => args.extend(["--draft-mtp".to_string(), "3".to_string()]),
        Draft::Model => {
            draft_dir = write_checkpoint("qwen3");
            args.extend([
                "--draft-model".to_string(),
                format!("{}:4", draft_dir.display()),
            ]);
        }
    }
    let server = Server::spawn(&dir, flag, &args);
    let served = server.complete(&greedy(&REPEATING_PROMPT));
    assert_eq!(
        text_of(&served),
        load(&dir).text().decode(&off.tokens).unwrap(),
        "{flag} {draft:?}: the speculative tokens differ from Off's {:?}",
        off.tokens
    );
    let metrics = server.metrics();
    let count = |key: &str| {
        metrics[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{key}: {metrics}"))
    };
    let (rounds, proposed, accepted, skipped) = (
        count("speculative_rounds"),
        count("speculative_proposed_tokens"),
        count("speculative_accepted_tokens"),
        count("speculative_skipped_rows"),
    );
    eprintln!(
        "{flag} {draft:?}: rounds {rounds}, proposed {proposed}, accepted {accepted}, skipped {skipped}"
    );
    assert!(rounds > 0, "{draft:?}: no round was verified: {metrics}");
    match draft {
        // The repeating prompt gives the lookup drafts the target keeps.
        Draft::Lookup => {
            assert!(accepted > 0, "{draft:?}: no draft was kept: {metrics}");
            assert!(accepted <= proposed, "{draft:?}: {metrics}");
        }
        // A qwen3 draft for a qwen2 target is a wrong draft: the target rejects some of what it proposes,
        // so accepting every draft shows in the counts and not only in the tokens.
        Draft::Model => assert!(
            accepted < proposed,
            "{draft:?}: a wrong draft was never rejected: {metrics}"
        ),
        // The fixture's random MTP layer mostly misses, so the target rejects and the recurrent
        // rollback runs inside the served loop.
        Draft::Mtp => assert!(
            proposed > 0 && accepted < proposed,
            "{draft:?}: no draft was proposed, or none was rejected: {metrics}"
        ),
    }
    assert_eq!(
        1 + rounds + accepted + skipped,
        NEW_TOKENS as u64,
        "{flag} {draft:?}: the counted rounds do not add up to the served tokens: {metrics}"
    );
}

#[derive(Clone, Copy, Debug)]
enum Draft {
    Lookup,
    Model,
    Mtp,
}

per_backend!(
    BackendChoice,
    |choice| served_speculative(choice, Draft::Lookup),
    served_lookup_speculation_returns_the_off_tokens_wgpu,
    served_lookup_speculation_returns_the_off_tokens_rocm,
    served_lookup_speculation_returns_the_off_tokens_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_speculative(choice, Draft::Model),
    served_draft_model_speculation_returns_the_off_tokens_wgpu,
    served_draft_model_speculation_returns_the_off_tokens_rocm,
    served_draft_model_speculation_returns_the_off_tokens_ptx
);
per_backend!(
    BackendChoice,
    |choice| served_speculative(choice, Draft::Mtp),
    served_mtp_speculation_returns_the_off_tokens_wgpu,
    served_mtp_speculation_returns_the_off_tokens_rocm,
    served_mtp_speculation_returns_the_off_tokens_ptx
);

/// The server pages its KV pool, and the qwen4exp indexer's per-row block-key cache has no paged
/// layout yet (POOT-1148): startup stops with the typed `PagedBlockedState` refusal, before the listener
/// binds. Mutation: drop the paged check from `Qwen4Exp::trace_into`; the engine plans a paged
/// indexer and the server starts, so the row goes red.
fn a_paged_pool_refuses_the_qwen4exp_indexer_startup(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    if poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)).is_none() {
        return;
    }
    let dir = write_checkpoint("qwen4exp");
    let stderr = refused(&dir, flag, &[]);
    assert!(
        stderr.contains("PagedBlockedState"),
        "the refusal does not name the paged indexer: {stderr}"
    );
}

per_backend!(
    BackendChoice,
    a_paged_pool_refuses_the_qwen4exp_indexer_startup,
    a_paged_pool_refuses_the_qwen4exp_indexer_startup_wgpu,
    a_paged_pool_refuses_the_qwen4exp_indexer_startup_rocm,
    a_paged_pool_refuses_the_qwen4exp_indexer_startup_ptx
);

/// `--draft-mtp` on a family with no `draft` program stops startup with the family's typed program
/// refusal (`TraceError::ProgramUnsupported`), before the listener binds. Mutation: build `DraftPolicy::Off` for the flag in `draft_policy`; the
/// server starts and serves, and the row goes red.
fn a_model_without_an_mtp_layer_refuses_draft_mtp_startup(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    if poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)).is_none() {
        return;
    }
    let dir = write_checkpoint("qwen2");
    let stderr = refused(&dir, flag, &["--draft-mtp".to_string(), "2".to_string()]);
    assert!(
        stderr.contains("has no program draft"),
        "the refusal does not name the missing draft program: {stderr}"
    );
}

per_backend!(
    BackendChoice,
    a_model_without_an_mtp_layer_refuses_draft_mtp_startup,
    a_model_without_an_mtp_layer_refuses_draft_mtp_startup_wgpu,
    a_model_without_an_mtp_layer_refuses_draft_mtp_startup_rocm,
    a_model_without_an_mtp_layer_refuses_draft_mtp_startup_ptx
);

/// A `--draft-model` path that holds no checkpoint stops startup with the load error, before the engine
/// plan is built. The server never binds. Mutation: ignore the draft path in `draft_policy` (return
/// `DraftPolicy::Off`); the server serves and the row goes red.
#[test]
fn a_missing_draft_model_refuses_startup() {
    let dir = write_checkpoint("qwen2");
    let missing = dir.join("no-such-draft");
    let stderr = refused(
        &dir,
        "wgpu",
        &[
            "--draft-model".to_string(),
            format!("{}:4", missing.display()),
        ],
    );
    assert!(
        stderr.contains("load the draft model"),
        "the refusal does not name the draft model: {stderr}"
    );
}

/// The `name` fixture with `vocab` rows in every tensor that has one row per token.
fn wider_vocabulary(name: &str, vocab: usize) -> Fixture {
    let fixture = fixture_of(name);
    let narrow = vocab_of(&fixture.config);
    let mut config = fixture.config.clone();
    config["vocab_size"] = vocab.into();
    let tensors = fixture.store.iter().map(|(key, entry)| {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture store is dense: {key} is packed")
        };
        let mut shape = dense.shape().to_vec();
        if shape[0] == narrow {
            shape[0] = vocab;
        }
        (key.as_str().to_string(), shape)
    });
    Fixture::bf16(config, tensors.collect::<Vec<_>>())
}

/// A draft model whose vocabulary is larger than the target's is refused by the driver as a typed
/// `DraftMismatch` while the engine plan is built, so startup fails before the listener binds. Mutation:
/// skip `check_draft` in the driver; the server serves a draft it cannot verify against and the row
/// goes red.
fn draft_with_a_larger_vocabulary_refuses_startup(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    if poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)).is_none() {
        return;
    }
    let target_dir = write_checkpoint("qwen2");
    let draft_dir = write_fixture(wider_vocabulary("qwen2", 64));
    let stderr = refused(
        &target_dir,
        flag,
        &[
            "--draft-model".to_string(),
            format!("{}:4", draft_dir.display()),
        ],
    );
    assert!(
        stderr.contains("the draft's vocabulary of"),
        "a draft of 64 tokens for a target of 48: the refusal is not the typed vocabulary mismatch: {stderr}"
    );
}

per_backend!(
    BackendChoice,
    draft_with_a_larger_vocabulary_refuses_startup,
    draft_with_a_larger_vocabulary_refuses_startup_wgpu,
    draft_with_a_larger_vocabulary_refuses_startup_rocm,
    draft_with_a_larger_vocabulary_refuses_startup_ptx
);

/// Rows the verify window cannot check (a penalty, logprobs) decode one token under a draft policy and
/// the driver reports why: `speculative_skipped_rows` moves for each such request, and the tokens are
/// the ones `Driver::generate` returns for that sampler without a policy. `n > 1` of such a request
/// skips on every choice. Mutation: have `Driver::step` speculate these rows without the sampler check;
/// the counter does not move or the tokens diverge and the row goes red.
fn served_unspeculated_rows_decode_as_off(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    let dir = write_checkpoint("qwen2");
    let penalised = || Sampler::greedy().with_penalties(1.5, 0.0, 0.0);
    let Some(mut both) = references(
        choice,
        backend,
        &dir,
        &encoded(&dir, &REPEATING_PROMPT),
        vec![penalised(), Sampler::greedy()],
    ) else {
        return;
    };
    let plain = both.pop().unwrap();
    let penalty = both.pop().unwrap();
    assert_ne!(
        penalty.tokens, plain.tokens,
        "the penalty must change the tokens, or the row proves nothing"
    );
    let handle = load(&dir);
    let server = Server::spawn(
        &dir,
        flag,
        &["--draft-lookup".to_string(), "2:4".to_string()],
    );
    let skipped = || {
        server.metrics()["speculative_skipped_rows"]
            .as_u64()
            .unwrap()
    };

    let mut body = greedy(&REPEATING_PROMPT);
    body["repetition_penalty"] = 1.5.into();
    let before = skipped();
    let served = server.complete(&body);
    assert_eq!(
        text_of(&served),
        handle.text().decode(&penalty.tokens).unwrap(),
        "{flag}: a penalised request differs from Off's {:?}",
        penalty.tokens
    );
    let after_penalty = skipped();
    assert!(
        after_penalty > before,
        "{flag}: the penalised row was not counted as skipped"
    );

    let mut body = greedy(&REPEATING_PROMPT);
    body["logprobs"] = 1.into();
    let served = server.complete(&body);
    assert_eq!(
        text_of(&served),
        handle.text().decode(&plain.tokens).unwrap(),
        "{flag}: a logprobs request differs from Off's {:?}",
        plain.tokens
    );
    let after_logprobs = skipped();
    assert!(
        after_logprobs > after_penalty,
        "{flag}: the logprobs row was not counted as skipped"
    );

    let mut body = greedy(&REPEATING_PROMPT);
    body["repetition_penalty"] = 1.5.into();
    body["n"] = 2.into();
    let served = server.complete(&body);
    let choices = served["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 2);
    for choice in choices {
        assert_eq!(
            choice["text"].as_str().unwrap(),
            handle.text().decode(&penalty.tokens).unwrap(),
            "{flag}: a choice of an n = 2 request differs from Off's {:?}",
            penalty.tokens
        );
    }
    assert!(
        skipped() - after_logprobs >= 2 * (NEW_TOKENS as u64 - 1),
        "{flag}: both choices must be counted as skipped on every decode step"
    );
}

per_backend!(
    BackendChoice,
    served_unspeculated_rows_decode_as_off,
    served_unspeculated_rows_decode_as_off_wgpu,
    served_unspeculated_rows_decode_as_off_rocm,
    served_unspeculated_rows_decode_as_off_ptx
);

// ---------------------------------------------------------------------------------------------
// A vision-language checkpoint: a served image request returns the driver's caption.
// ---------------------------------------------------------------------------------------------

/// `store` with every bf16 weight whose name `keep` does not select multiplied by `factor`.
fn scaled_except(
    store: &poot_quant::weights::WeightStore,
    factor: f32,
    keep: impl Fn(&str) -> bool,
) -> poot_quant::weights::WeightStore {
    use poot_quant::weights::{DenseWeight, WeightStore};
    let mut out = WeightStore::builder();
    for (key, entry) in store.iter() {
        let WeightEntry::Dense(dense) = entry else {
            panic!("the fixture is dense")
        };
        let bytes: Vec<u8> = if keep(key.as_str()) {
            dense.bytes().as_slice().to_vec()
        } else {
            dense
                .bytes()
                .as_slice()
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&pair| {
                    let value = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                    (((value * factor).to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect()
        };
        let dense =
            DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into()).unwrap();
        out.insert(key.clone(), WeightEntry::Dense(dense)).unwrap();
    }
    out.build()
}

/// The in-process reference of a vision-language checkpoint: its front end on a vision driver and its
/// text model on a driver of its own, the single-request contiguous run a served request must equal.
struct VlmReference {
    handle: Arc<ModelHandle>,
    session: poot_llm::FrontEndSession,
    text: Driver<Serving>,
}

impl VlmReference {
    fn open(
        handle: &Arc<ModelHandle>,
        vision: Box<dyn poot_executor::Executor>,
        text: Box<dyn poot_executor::Executor>,
        text_options: DriverOptions,
    ) -> Self {
        let mut vision = Some(vision);
        let session = poot_llm::FrontEndSession::open(Arc::clone(handle), |_, positions| {
            Ok::<_, poot_llm::driver::error::DriverError>((
                vision.take().expect("one tower"),
                poot_llm::driver::DriverOptions::tower(positions),
            ))
        })
        .expect("open the vision towers")
        .expect("a vision-language checkpoint has a front end");
        Self {
            handle: Arc::clone(handle),
            session,
            text: Driver::new(Arc::clone(handle), text, text_options)
                .expect("a text driver")
                .serve(Layout::Contiguous),
        }
    }

    /// The greedy `max_new` tokens of `turns` over the encoded `images`, as text.
    fn generate(
        &mut self,
        images: &[&Vec<u8>],
        turns: &[poot_models::multimodal::front_end::ChatTurn<'_>],
        max_new: usize,
    ) -> String {
        let images: Vec<&[u8]> = images.iter().map(|png| png.as_slice()).collect();
        let prepared = self
            .session
            .prepare_images(&images, turns)
            .expect("the front end lowers the request");
        let generation = self
            .text
            .generate_with_inputs(
                GenerateRequest {
                    prompt: prepared.prompt,
                    max_new,
                    sampler: Sampler::greedy(),
                    stops: Vec::new(),
                    ignore_eos: false,
                },
                prepared.inputs,
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .expect("generate");
        self.handle
            .text()
            .decode(&generation.tokens)
            .expect("decode the generation")
    }
}

/// The SmolVLM fixture: ids of the tokenizer's words, then the idefics3 markers the front end's prompt
/// names, then a filler up to the vocabulary. `b c d` is the question.
mod smolvlm {
    use poot_models::registry::Registry;
    use poot_quant::weights::WeightStore;

    pub(super) const IMAGE: u32 = 42;
    pub(super) const VOCAB: usize = 48;
    const MARKERS: [&str; 11] = [
        "User:",
        "Assistant:",
        "<|im_start|>",
        "<end_of_utterance>",
        "<fake_token_around_image>",
        "<global-img>",
        "<image>",
        "<row_1_col_1>",
        "<row_1_col_2>",
        "<row_2_col_1>",
        "<row_2_col_2>",
    ];

    /// A word-level tokenizer whose markers (from `<|im_start|>` on) are special tokens, so the prompt's
    /// `<image>` runs split out of the surrounding text.
    fn tokenizer_json() -> serde_json::Value {
        let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
        words.extend(('0'..='9').map(String::from));
        words.extend(MARKERS.map(String::from));
        words.push("filler".to_string());
        assert_eq!(words.len(), VOCAB);
        assert_eq!(words[IMAGE as usize], "<image>");
        let vocab: serde_json::Map<String, serde_json::Value> = words
            .iter()
            .enumerate()
            .map(|(id, word)| (word.clone(), serde_json::json!(id)))
            .collect();
        let added: Vec<serde_json::Value> = words
            .iter()
            .enumerate()
            .skip(36 + 2)
            .take(MARKERS.len() - 2)
            .map(|(id, word)| {
                serde_json::json!({
                    "id": id, "content": word, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect();
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": { "type": "WhitespaceSplit" },
            "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "a" }
        })
    }

    /// `store` with every non-norm bf16 weight multiplied by `factor`: the fixture's small weights would
    /// otherwise leave the image rows too faint to reach the greedy token.
    fn scaled(store: &WeightStore, factor: f32) -> WeightStore {
        super::scaled_except(store, factor, |name| name.contains("norm"))
    }

    /// The registry's `idefics3` fixture (8-pixel frames, 16 patches, 4 visual tokens of the 24-wide llama
    /// text tower) as one HF directory with the checkpoint's `config.json`, `preprocessor_config.json` and
    /// tokenizer.
    pub(super) fn write_checkpoint(no_eos: u32) -> poot_test_util::UniqueTempPath {
        let registry = Registry::builtin().unwrap();
        let fixture = poot_models::test_support::family_fixtures::base(
            registry
                .entries()
                .iter()
                .find(|entry| entry.family.as_str() == "idefics3")
                .unwrap()
                .family,
        );
        // The text tower's weights are scaled up so the image rows reach the greedy token.
        let mut text = WeightStore::builder();
        for (key, entry) in fixture.store.iter().filter(|(key, _)| {
            key.as_str().starts_with("model.text_model.") || key.as_str() == "lm_head.weight"
        }) {
            text.insert(key.clone(), entry.clone()).unwrap();
        }
        let text = scaled(&text.build(), 8.0);
        let mut store = WeightStore::builder();
        for (key, entry) in fixture.store.iter() {
            let entry = text.get(key.as_str()).unwrap_or(entry);
            store.insert(key.clone(), entry.clone()).unwrap();
        }
        let dir = poot_test_util::unique_temp_path("served_smolvlm_checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = fixture.config.clone();
        config["text_config"]["eos_token_id"] = no_eos.into();
        assert_eq!(config["image_token_id"], IMAGE);
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({ "chat_template": poot_test_util::smolvlm::CHAT_TEMPLATE })
                .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("preprocessor_config.json"),
            fixture.preprocessor.unwrap().to_string(),
        )
        .unwrap();
        poot_quant::test_support::safetensors::write(
            &dir.join("model.safetensors"),
            &store.build(),
        );
        dir
    }
}

/// A 16x16 PNG: `fill` gives each pixel's grey. A 16-pixel image splits into a 2x2 tile grid and a
/// global frame, five frames of four visual tokens.
fn png_of(fill: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let image = image::RgbImage::from_fn(16, 16, |x, y| image::Rgb([fill(x, y); 3]));
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    png
}

/// An OpenAI `image_url` part holding `png` as a data URI.
fn image_part(png: &[u8]) -> serde_json::Value {
    use base64::Engine as _;
    let url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    );
    serde_json::json!({ "type": "image_url", "image_url": { "url": url } })
}

/// A chat request over `messages`, greedy.
fn chat_of(messages: serde_json::Value) -> serde_json::Value {
    serde_json::json!({ "messages": messages, "max_tokens": 8, "temperature": 0.0 })
}

fn chat_with_image(png: &[u8]) -> serde_json::Value {
    chat_of(serde_json::json!([{ "role": "user", "content": [
        { "type": "text", "text": "b c d" },
        image_part(png),
    ]}]))
}

impl Server {
    fn chat(&self, body: &serde_json::Value) -> (u16, serde_json::Value) {
        let (status, text) = self
            .request("POST", "/v1/chat/completions", &body.to_string())
            .expect("chat request");
        let json = serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}"));
        (status, json)
    }
}

/// A served image request returns the caption the driver gives the same image: the image front end in
/// the server (preprocess, vision tower, prompt) and the engine's paged text driver together equal
/// the in-process caption of the same front end and text driver (the single-request, contiguous reference). Two different images
/// caption differently (the rows reach the logits), and a request that cannot be lowered is a typed
/// 400, not a dropped image. Mutation: zero the rows in `vision::ImageClient::prepare`; the served
/// caption is the zero-row continuation, equal for both images, and the row goes red. Mutation (the
/// registry binds each front end to its checkpoint): swap the `keys` of the `idefics3` and `qwen2_5_vl`
/// entries, so each checkpoint is built by the other's preprocessor; observed on wgpu, this row,
/// `served_qwen_vl_caption`, `served_idefics3_wgpu` and `served_qwen2_5_vl_wgpu` all fail to load their
/// fixture (`config field scale_factor: Missing`, `config field model_type: Unsupported`) and go red.
fn served_smolvlm_caption(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    use poot_models::multimodal::front_end::ChatTurn;

    let dir = smolvlm::write_checkpoint(NO_EOS);
    let black = png_of(|_, _| 0);
    let white = png_of(|_, _| 255);
    let quadrant = png_of(|x, y| if x < 8 && y < 8 { 0 } else { 255 });
    let options = |capacity: usize| {
        let mut options = driver_options();
        options.capacity = nz(capacity);
        options.prefill_chunk = nz(capacity);
        options.trace.max_tokens = nz(capacity);
        options
    };
    let (Some(vision), Some(text)) = (
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)),
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)),
    ) else {
        return;
    };
    let handle = load(&dir);
    let mut vlm = VlmReference::open(&handle, vision, text, options(CAPACITY * 2));
    // The driver's caption of a conversation: the reference the served one must equal.
    let mut caption = |pngs: &[&Vec<u8>], turns: &[ChatTurn<'_>]| vlm.generate(pngs, turns, 8);
    let expected: Vec<String> = [&black, &white, &quadrant]
        .into_iter()
        .map(|png| caption(&[png], &[ChatTurn::user(1, "b c d")]))
        .collect();
    // A conversation: a system prompt, an image question, the assistant's answer, a second image.
    let conversation = [
        ChatTurn {
            role: "system",
            media: 0,
            text: "b",
        },
        ChatTurn::user(1, "b c d"),
        ChatTurn {
            role: "assistant",
            media: 0,
            text: "e f",
        },
        ChatTurn::user(1, "c d"),
    ];
    let expected_conversation = caption(&[&black, &white], &conversation);
    let expected_last_turn_alone = caption(&[&white], &[ChatTurn::user(1, "c d")]);
    // A question that spells the image mark is the client's text: it takes no image's rows (the front
    // end's `Placeholders` check would refuse a prompt that did), so the caption is the driver's.
    let expected_spelled = caption(&[&black], &[ChatTurn::user(1, "b <image> c")]);
    drop(caption);
    drop(vlm);
    assert_ne!(
        expected[0], expected[1],
        "the image rows must reach the logits, or the row proves nothing"
    );
    assert_ne!(
        expected[1], expected[2],
        "the tile order is part of the caption"
    );
    assert_ne!(
        expected_conversation, expected_last_turn_alone,
        "the earlier turns must reach the logits, or the multi-turn row proves nothing"
    );

    let server = Server::spawn(&dir, flag, &[]);
    for (png, want) in [&black, &white, &quadrant].into_iter().zip(&expected) {
        let (status, served) = server.chat(&chat_with_image(png));
        assert_eq!(status, 200, "{served}");
        assert_eq!(
            served["choices"][0]["message"]["content"].as_str().unwrap(),
            want.trim_end(),
            "{flag}: the served caption differs from the in-process caption"
        );
    }

    // An image request the front end cannot lower is the client's.
    let refused = |body: &serde_json::Value, needle: &str| {
        let (status, error) = server.chat(body);
        assert_eq!(status, 400, "{error}");
        assert!(error.to_string().contains(needle), "{error}");
    };
    let mut remote = chat_with_image(&black);
    remote["messages"][0]["content"][1]["image_url"]["url"] = "https://example.com/a.png".into();
    refused(&remote, "data: URIs");
    let mut garbage = chat_with_image(&black);
    garbage["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64,AAAA".into();
    refused(&garbage, "does not decode");
    // A text that spells the image mark is plain text, not a second image: it is served, and the caption
    // is the driver's. Mutation: drop the `contains_id` check in `TextCodec::encode_without_added_tokens`;
    // the word-level fixture maps the spelled mark to the image token and the request is refused with
    // `Placeholders` (21 tokens for 20 rows).
    let mut spelled = chat_with_image(&black);
    spelled["messages"][0]["content"][0]["text"] = "b <image> c".into();
    let (status, served) = server.chat(&spelled);
    assert_eq!(status, 200, "{served}");
    assert_eq!(
        served["choices"][0]["message"]["content"].as_str().unwrap(),
        expected_spelled.trim_end(),
        "{flag}: the served caption of a spelled mark differs from the in-process caption"
    );
    // A tool result has no place in the image chat template.
    let mut tool_result = chat_with_image(&black);
    tool_result["messages"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "role": "tool", "content": "b", "tool_call_id": "t" }));
    refused(&tool_result, "tool");

    // A conversation with images in two turns, a system prompt and an assistant answer is rendered
    // through the model's chat template: it equals the driver's caption of the same turns. Mutation:
    // keep only the last message in `handlers::image_prompt`; the earlier turns and the first image
    // are dropped, the served caption becomes the last turn alone, and the row goes red.
    let served_conversation = chat_of(serde_json::json!([
        { "role": "system", "content": "b" },
        { "role": "user", "content": [
            { "type": "text", "text": "b c d" },
            image_part(&black),
        ]},
        { "role": "assistant", "content": "e f" },
        { "role": "user", "content": [
            { "type": "text", "text": "c d" },
            image_part(&white),
        ]},
    ]));
    let (status, served) = server.chat(&served_conversation);
    assert_eq!(status, 200, "{served}");
    assert_eq!(
        served["choices"][0]["message"]["content"].as_str().unwrap(),
        expected_conversation.trim_end(),
        "{flag}: the served conversation differs from the in-process caption"
    );

    // Tools would be dropped by the image prompt, so they are refused rather than ignored. Mutation:
    // delete the tools check in `handlers::image_prompt`; the request is served and the row goes red.
    let mut tools = chat_with_image(&black);
    tools["tools"] =
        serde_json::json!([{ "type": "function", "function": { "name": "f", "parameters": {} } }]);
    refused(&tools, "tools");
}

per_backend!(
    BackendChoice,
    served_smolvlm_caption,
    served_smolvlm_caption_wgpu,
    served_smolvlm_caption_rocm,
    served_smolvlm_caption_ptx
);

// ---------------------------------------------------------------------------------------------
// Qwen2.5-VL: the second image front end, with the prompt's sectioned positions.
// ---------------------------------------------------------------------------------------------

/// The Qwen2.5-VL fixture: the registry's text and vision towers over one set of weights, a 48-word
/// tokenizer with the vision markers, a one-line chat template and a processor of 2-pixel patches that
/// merge 2 x 2.
mod qwen_vl {
    use std::collections::HashMap;

    use poot_models::registry::Registry;
    use poot_quant::weights::WeightStore;

    pub(super) const VOCAB: usize = 48;
    const VISION_START: (&str, u32) = ("<|vision_start|>", 40);
    const IMAGE_PAD: (&str, u32) = ("<|image_pad|>", 41);
    const VIDEO_PAD: (&str, u32) = ("<|video_pad|>", 42);
    const VISION_END: (&str, u32) = ("<|vision_end|>", 43);
    const IM_END: (&str, u32) = ("<|im_end|>", 47);

    /// A word-level tokenizer: word `t<i>` is token `i` but for `user`, `hello` and the markers.
    fn tokenizer_json() -> serde_json::Value {
        let mut vocab: HashMap<String, u32> =
            (0..VOCAB as u32).map(|id| (format!("t{id}"), id)).collect();
        let markers = [VISION_START, IMAGE_PAD, VIDEO_PAD, VISION_END, IM_END];
        for (word, id) in [("user", 2), ("hello", 4)].into_iter().chain(markers) {
            vocab.retain(|_, v| *v != id);
            vocab.insert(word.to_string(), id);
        }
        let added: Vec<serde_json::Value> = markers
            .iter()
            .map(|&(word, id)| {
                serde_json::json!({
                    "id": id, "content": word, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect();
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": { "type": "WhitespaceSplit" },
            "post_processor": null,
            "decoder": null,
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "t0" }
        })
    }

    fn fixture_of(registry: &Registry, family: &str) -> poot_models::registry::Fixture {
        let entry = registry
            .entries()
            .iter()
            .find(|entry| entry.family.as_str() == family)
            .unwrap_or_else(|| panic!("{family} is a shipped family"));
        poot_models::test_support::family_fixtures::base(entry.family)
    }

    /// The registry's Qwen2.5-VL fixture (the text and vision towers over one set of weights, multiplied by
    /// `factor`, norm scales and the merger's `ln_q` kept) as one checkpoint directory: `config.json` with
    /// the text model on top and `vision_config` nested.
    pub(super) fn write_checkpoint(factor: f32, no_eos: u32) -> poot_test_util::UniqueTempPath {
        let registry = Registry::builtin().unwrap();
        let fixture = fixture_of(&registry, "qwen2_5_vl");
        let dir = poot_test_util::unique_temp_path("served_qwen_vl_checkpoint");
        std::fs::create_dir_all(&dir).unwrap();
        let keep = |name: &str| name.contains("norm") || name.contains("ln_q");
        let store: WeightStore = super::scaled_except(&fixture.store, factor, keep);
        poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &store);
        let mut config = fixture.config.clone();
        config["eos_token_id"] = no_eos.into();
        assert_eq!(config["image_token_id"], IMAGE_PAD.1);
        assert_eq!(config["vision_start_token_id"], VISION_START.1);
        assert_eq!(config["video_token_id"], VIDEO_PAD.1);
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(
            dir.join("preprocessor_config.json"),
            fixture.preprocessor.unwrap().to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::json!({ "chat_template": poot_test_util::qwen_vl::CHAT_TEMPLATE })
                .to_string(),
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer_json().to_string()).unwrap();
        dir
    }
}

/// A 12-row, 8-column PNG (a 6 x 4 patch grid of 2-pixel patches, 6 merged tokens): `fill` gives each
/// pixel's grey.
fn small_png_of(fill: impl Fn(u32, u32) -> u8) -> Vec<u8> {
    let image = image::RgbImage::from_fn(8, 12, |x, y| image::Rgb([fill(x, y); 3]));
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    png
}

/// A served image request on a Qwen2.5-VL checkpoint returns the generation the same front end and text
/// driver give the same image and question in this process: the vision tower's rows and the prompt's sectioned
/// positions (`Slot::MropePosition`, the image's tokens off their index) reach the engine's paged text
/// driver, which equals the single-request contiguous reference. Different images generate differently
/// (the rows reach the logits). A request the front end cannot lower is a typed 400. Mutation: replace the
/// positions with zeros where `DriverEngine::open` hands them to `Driver::open` as request inputs; the served generation
/// leaves the reference's and the row goes red. The registry mutation of `served_smolvlm_caption` (swap
/// the two entries' `keys`) also turns this row red.
fn served_qwen_vl_caption(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    use poot_models::multimodal::front_end::ChatTurn;

    const QUESTION: &str = "hello";
    let dir = qwen_vl::write_checkpoint(2.0, NO_EOS);
    let pngs = [
        small_png_of(|_, _| 0),
        small_png_of(|_, _| 255),
        small_png_of(|x, y| if x < 4 && y < 6 { 0 } else { 255 }),
    ];
    let options = |capacity: usize| {
        let mut options = driver_options();
        options.capacity = nz(capacity);
        options.prefill_chunk = nz(capacity);
        options.trace.max_tokens = nz(capacity);
        options
    };
    let (Some(vision), Some(text)) = (
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)),
        poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)),
    ) else {
        return;
    };
    let handle = load(&dir);
    let mut vlm = VlmReference::open(&handle, vision, text, options(32));
    let expected: Vec<String> = pngs
        .iter()
        .map(|png| vlm.generate(&[png], &[ChatTurn::user(1, QUESTION)], 8))
        .collect();
    drop(vlm);
    assert_ne!(
        expected[0], expected[1],
        "the image rows must reach the logits, or the row proves nothing"
    );
    assert_ne!(
        expected[1], expected[2],
        "the patch order is part of the generation"
    );

    let server = Server::spawn(&dir, flag, &[]);
    let chat = |png: &[u8]| {
        let mut body = chat_with_image(png);
        body["messages"][0]["content"][0]["text"] = QUESTION.into();
        body
    };
    for (png, want) in pngs.iter().zip(&expected) {
        let (status, served) = server.chat(&chat(png));
        assert_eq!(status, 200, "{served}");
        assert_eq!(
            served["choices"][0]["message"]["content"].as_str().unwrap(),
            want.trim_end(),
            "{flag}: the served generation differs from the in-process generation"
        );
    }

    // An image request the front end cannot lower is the client's.
    let refused = |body: &serde_json::Value, needle: &str| {
        let (status, error) = server.chat(body);
        assert_eq!(status, 400, "{error}");
        assert!(error.to_string().contains(needle), "{error}");
    };
    let mut remote = chat(&pngs[0]);
    remote["messages"][0]["content"][1]["image_url"]["url"] = "https://example.com/a.png".into();
    refused(&remote, "data: URIs");
    let mut garbage = chat(&pngs[0]);
    garbage["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64,AAAA".into();
    refused(&garbage, "image");
}

per_backend!(
    BackendChoice,
    served_qwen_vl_caption,
    served_qwen_vl_caption_wgpu,
    served_qwen_vl_caption_rocm,
    served_qwen_vl_caption_ptx
);

/// A Qwen2.5-VL checkpoint refuses what it cannot take, as the client's error: an image of more patches
/// than the vision tower takes in one step (4100 against 4096) is a 400 from the tower's own bound, not a
/// dropped image or a server fault; and the same checkpoint without a `vision_config` serves as the text
/// model it is, whose image part is the typed `unsupported_capability` refusal. Mutation: stop mapping the
/// tower's patch cap into a `Refused` in `Qwen25VlFrontEnd::prepare`; the oversize image is a 500 and
/// the row goes red.
fn served_qwen_vl_refusals(choice: BackendChoice) {
    let (backend, flag) = (backend_of(choice), flag_of(choice));
    // The refusals are the server's, not the device's: skip when the backend is absent.
    if poot_runtime_common::device_skip::open_or_skip(backend, open_executor(choice)).is_none() {
        return;
    }
    let dir = qwen_vl::write_checkpoint(2.0, NO_EOS);
    std::fs::write(
        dir.join("preprocessor_config.json"),
        r#"{"do_resize":false,"min_pixels":1,"max_pixels":1048576,"patch_size":2,
           "temporal_patch_size":2,"merge_size":2}"#,
    )
    .unwrap();
    // 164 rows by 100 columns of 2-pixel patches: an 82 x 50 grid, 4100 patches.
    let oversize = {
        let image = image::RgbImage::from_fn(100, 164, |x, y| image::Rgb([(x + y) as u8; 3]));
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    };
    {
        let server = Server::spawn(&dir, flag, &[]);
        let (status, error) = server.chat(&chat_with_image(&oversize));
        assert_eq!(status, 400, "{error}");
        assert!(error.to_string().contains("4096"), "{error}");
    }

    let config_path = dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("vision_config");
    std::fs::write(&config_path, config.to_string()).unwrap();
    let server = Server::spawn(&dir, flag, &[]);
    let (status, error) = server.chat(&chat_with_image(&small_png_of(|_, _| 0)));
    assert_eq!(status, 400, "{error}");
    assert!(error.to_string().contains("no image front end"), "{error}");
    assert_eq!(error["error"]["type"], "unsupported_capability", "{error}");
}

per_backend!(
    BackendChoice,
    served_qwen_vl_refusals,
    served_qwen_vl_refusals_wgpu,
    served_qwen_vl_refusals_rocm,
    served_qwen_vl_refusals_ptx
);

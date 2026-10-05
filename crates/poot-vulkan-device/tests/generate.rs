//! SC-002: a dense greedy decode through `Driver::generate` on `open_executor(Vulkan)` equals the wgpu run
//! of the same model token for token, and the tokens recorded for the fixture when the driver landed.
//!
//! The model is the qwen2 fixture the registry ships, written out as an HF checkpoint directory and
//! loaded through `ModelHandle::load` (the same path a downloaded checkpoint takes). Both drivers open
//! before either runs. A row skips when its device does not open (`POOT_REQUIRE_VULKAN=1` /
//! `POOT_REQUIRE_WGPU=1` turn that into a failure). Device rows run serially under `/tmp/poot-gpu.lock`.
//!
//! Mutation (SC-002): make `VulkanDevice::write` a no-op after the first step (the `Device::write` arm
//! in `poot-vulkan-device/src/device.rs`); the second token is computed from a stale input and the
//! tiny-fixture row goes red.

use std::collections::HashMap;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;

use poot_graph_plan::{CompileLimits, CompileOptions, FusionPolicy, Submission};
use poot_llm::driver::{
    BackendChoice, Driver, DriverOptions, GenerateRequest, ModelHandle, PreparedSetLimits,
    open_executor, program_retention,
};
use poot_llm::{GenerationControl, Sampler};
use poot_models::registry::Registry;
use poot_quant::weights::WeightEntry;
use poot_runtime_common::DeviceBackend;
use poot_test_util::device_skip::open_or_skip;
use tokenizers::Tokenizer;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

const VOCAB: usize = 48;
/// An end-of-sequence id outside the fixture's vocabulary, so no run ends early on a sampled token.
const NO_EOS: u32 = 999;
const PROMPT: [u32; 7] = [3, 17, 40, 8, 25, 11, 2];
/// The fixture's greedy tokens after `PROMPT`, recorded on the wgpu executor when the driver landed
/// (`poot-llm`'s `FIXTURE_GREEDY`).
const FIXTURE_GREEDY: [u32; 24] = [
    33, 37, 41, 3, 41, 46, 41, 3, 8, 37, 41, 32, 41, 32, 44, 3, 8, 37, 3, 41, 41, 41, 41, 17,
];
/// Qwen2.5-0.5B's 24 greedy tokens after "The capital of Germany is Berlin. The capital of France is".
const QWEN2_5_0_5B: [u32; 24] = [
    12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 6323, 374, 26194, 13, 576,
    6722, 315, 279, 3639, 4180, 374, 6515,
];
const QWEN2_5_0_5B_PROMPT: [u32; 12] = [
    785, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315, 9625, 374,
];

/// The fixture's tokenizer: 48 single-character words, so ids are the positions in this list.
fn fixture_tokenizer() -> Tokenizer {
    let mut words: Vec<String> = ('a'..='z').map(String::from).collect();
    words.extend(('0'..='9').map(String::from));
    words.extend((0..VOCAB - words.len()).map(|i| format!("p{i}")));
    let vocab: HashMap<String, u32> = words
        .into_iter()
        .enumerate()
        .map(|(i, w)| (w, i as u32))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("a".to_string())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer
}

/// The registry's qwen2 fixture as an HF checkpoint directory: its config (with the unreachable
/// end-of-sequence id), one safetensors archive of its bf16 tensors and the fixture tokenizer.
fn fixture_checkpoint() -> poot_test_util::UniqueTempPath {
    let registry = Registry::builtin().unwrap();
    let fixture = (registry.entries()[0].fixture)();
    let mut config = fixture.config.clone();
    config["eos_token_id"] = serde_json::json!(NO_EOS);
    let dir = poot_test_util::unique_temp_path("poot_card553_qwen2_fixture");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (key, weight) in fixture.store.iter() {
        let WeightEntry::Dense(dense) = weight else {
            panic!("the fixture is dense")
        };
        let start = data.len();
        data.extend_from_slice(dense.bytes().as_slice());
        header.insert(
            key.as_str().to_string(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": dense.shape(),
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut archive = (header.len() as u64).to_le_bytes().to_vec();
    archive.extend_from_slice(&header);
    archive.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), archive).unwrap();
    fixture_tokenizer()
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    dir
}

fn options(chunk: usize, capacity: usize) -> DriverOptions {
    let compile = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: CompileLimits::STANDARD,
    };
    let nz = |n| NonZeroUsize::new(n).unwrap();
    DriverOptions {
        prefill: compile,
        decode: compile,
        capacity: nz(capacity),
        prefill_chunk: nz(chunk),
        max_trace_tokens: nz(64),
        prepared: PreparedSetLimits {
            max_entries: nz(64),
            max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
        },
        charge: program_retention,
    }
}

fn driver(
    backend: DeviceBackend,
    choice: BackendChoice,
    handle: &Arc<ModelHandle>,
    chunk: usize,
) -> Option<Driver> {
    let executor = open_or_skip(backend, open_executor(choice))?;
    Some(Driver::new(Arc::clone(handle), executor, options(chunk, 64)).unwrap())
}

fn greedy(driver: &mut Driver, prompt: &[u32], max_new: usize) -> Vec<u32> {
    driver
        .generate(
            GenerateRequest {
                prompt: prompt.to_vec(),
                max_new,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .unwrap()
        .tokens
}

fn load(path: &Path) -> Arc<ModelHandle> {
    Arc::new(ModelHandle::load(path, &Registry::builtin().unwrap()).unwrap())
}

#[test]
fn tiny_fixture_greedy_decode_equals_wgpu_on_vulkan() {
    let dir = fixture_checkpoint();
    let handle = load(&dir);
    // Both executors open before either runs.
    let Some(mut vulkan) = driver(DeviceBackend::Vulkan, BackendChoice::Vulkan, &handle, 8) else {
        return;
    };
    let Some(mut wgpu) = driver(DeviceBackend::Wgpu, BackendChoice::Wgpu, &handle, 8) else {
        return;
    };
    let on_wgpu = greedy(&mut wgpu, &PROMPT, 24);
    let on_vulkan = greedy(&mut vulkan, &PROMPT, 24);
    assert_eq!(
        on_vulkan, on_wgpu,
        "raw Vulkan and wgpu generate the same tokens"
    );
    assert_eq!(
        on_vulkan, FIXTURE_GREEDY,
        "and the tokens recorded for the fixture"
    );
}

/// The prompt in chunks of 5 (two prefill entries and a decode entry share one executable) generates
/// the same tokens as the whole prompt at once.
#[test]
fn chunked_prefill_equals_the_whole_prompt_on_vulkan() {
    let dir = fixture_checkpoint();
    let handle = load(&dir);
    let Some(mut whole) = driver(DeviceBackend::Vulkan, BackendChoice::Vulkan, &handle, 8) else {
        return;
    };
    let Some(mut chunked) = driver(DeviceBackend::Vulkan, BackendChoice::Vulkan, &handle, 5) else {
        return;
    };
    assert_eq!(greedy(&mut whole, &PROMPT, 24), FIXTURE_GREEDY);
    assert_eq!(greedy(&mut chunked, &PROMPT, 24), FIXTURE_GREEDY);
}

/// Qwen2.5-0.5B (bf16 safetensors) generates the 24 recorded greedy tokens at the whole prompt and in
/// chunks of 5. Checkpoint-device row: needs `POOT_MODELS_DIR` with the model and at least 40Gi available.
#[test]
#[ignore = "loads the real qwen2.5-0.5b bf16 checkpoint onto the GPU; needs >= 40Gi available"]
fn qwen2_5_0_5b_greedy_tokens_equal_the_recorded_literals_on_vulkan() {
    let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b")) else {
        return;
    };
    let handle = load(&path);
    let Some(mut whole) = driver(
        DeviceBackend::Vulkan,
        BackendChoice::Vulkan,
        &handle,
        QWEN2_5_0_5B_PROMPT.len(),
    ) else {
        return;
    };
    let Some(mut chunked) = driver(DeviceBackend::Vulkan, BackendChoice::Vulkan, &handle, 5) else {
        return;
    };
    assert_eq!(greedy(&mut whole, &QWEN2_5_0_5B_PROMPT, 24), QWEN2_5_0_5B);
    assert_eq!(greedy(&mut chunked, &QWEN2_5_0_5B_PROMPT, 24), QWEN2_5_0_5B);
}

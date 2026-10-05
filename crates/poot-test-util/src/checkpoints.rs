//! The checkpoint directory names a test may load: the table [`checkpoint!`](crate::checkpoint) checks at
//! compile time.
//!
//! Every entry is a row of the model inventory (Backstage `projects/poot/docs/model-inventory.md`, section
//! "Referenced checkpoints and local directory names"), and this table mirrors the part of that list that tests
//! load. A name that is not here does not compile, because `model_path` skips a model that is missing from the
//! models directory even under `POOT_REQUIRE_MODELS=1`: a wrong name would skip forever and no lane would
//! ever fail. Backstage `projects/poot/docs/check_model_inventory.py` fails any entry that is not an inventory
//! row, so the table cannot drift from the inventory either.
//!
//! One entry per line, as a plain string literal, so the check can read this file.

/// The checkpoint directory names tests use, sorted.
pub const CHECKPOINTS: &[&str] = &[
    "all-minilm-l6-v2",
    "bge-small-en",
    "bloom-560m",
    "bloom-560m-gguf",
    "deepseek2-lite",
    "deepseek2-tiny",
    "deepseek3-tiny",
    "gemma-3-1b-gguf",
    "gemma-3-1b-it",
    "gemma-4-31B-it-GGUF",
    "gptoss-20b",
    "gpt-oss-20b-gguf",
    "gptoss-tiny",
    "granite-3.1-2b",
    "granite-moe-1b",
    "granite-moe-1b-gguf",
    "granitemoe-tiny",
    "lite-mistral-150m",
    "llama-3.2-1b-gguf",
    "llama-3.2-1b-instruct",
    "mixtral-8x7b-instruct-gguf",
    "mixtral-tiny",
    "mptk-1b",
    "ms-marco-minilm-l6",
    "olmo2-1b",
    "olmo2-1b-gguf",
    "olmoe-1b-7b",
    "olmoe-tiny",
    "phi-3.5-mini-gguf",
    "phi-3-mini-gguf",
    "phi-4-mini",
    "phi-4-mini-gguf",
    "qwen2.5-0.5b",
    "qwen2.5-0.5b-awq",
    "qwen2.5-0.5b-fp8",
    "qwen2.5-0.5b-gguf",
    "qwen2.5-0.5b-gptq",
    "qwen2.5-0.5b-instruct",
    "qwen2.5-0.5b-q4km",
    "qwen2.5-1.5b",
    "qwen2.5-1.5b-q4km",
    "qwen2.5-1.5b-q4ks",
    "qwen2.5-7b-q4km",
    "qwen3-0.6b",
    "qwen3-0.6b-gguf",
    "qwen3-moe-tiny",
    "qwen3-moe-tiny-sparse",
    "qwen3-moe-tiny-sparse-gguf",
    "smollm2-135m-gguf",
    "smollm2-360m",
    "smollm3-3b",
    "smollm3-tiny",
    "smolvlm-256m",
    "tinyllama-1.1b-chat",
];

/// Whether `name` is a table entry. A `const fn`, so [`Checkpoint::new`](crate::Checkpoint::new) can refuse an
/// unlisted name during compilation.
pub(crate) const fn is_listed(name: &[u8]) -> bool {
    let mut entry = 0;
    while entry < CHECKPOINTS.len() {
        if bytes_equal(CHECKPOINTS[entry].as_bytes(), name) {
            return true;
        }
        entry += 1;
    }
    false
}

/// The bytes of `path` before its first `/`: the checkpoint directory a path inside it belongs to.
pub(crate) const fn first_component(path: &str) -> &[u8] {
    let bytes = path.as_bytes();
    let mut end = 0;
    while end < bytes.len() && bytes[end] != b'/' {
        end += 1;
    }
    bytes.split_at(end).0
}

const fn bytes_equal(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

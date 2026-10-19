//! Header-only whole-file plans of every Qwen3.8 GGUF under the models directory.
//!
//! The file set is found by scanning `POOT_MODELS_DIR`: every `.gguf` (a split set once, by its first
//! shard) in a directory whose name contains `qwen3.8`, models and `mmproj` projectors alike, so a file
//! added to the directory is planned without a test edit. Each file is mapped, its header parsed, and
//! **every** tensor planned through `gguf_plan` with `IdentityNames` (what `poot-llm`'s driver loads a
//! GGUF with). No tensor is decoded and no name map narrows the file, so a ggml type with no
//! `WeightFormat` refuses the plan naming its tensor. A first-blocks or single-tensor name map cannot see
//! such a tensor: the 27B UD-Q4_K_XL file's one `IQ3_S` tensor (`blk.14.ffn_down.weight`, ggml type 21)
//! hid behind one until POOT-1192 mapped the type.
//!
//! These rows are regression guards: type 21 already maps to `Iq3_S` on the base. Mutation (applied by
//! hand, shown red, restored; recorded in the POOT-1197 report): map ggml type 21 to `None` in
//! `weight_format_of`; both rows go red naming `blk.14.ffn_down.weight`.
//!
//! Without `POOT_MODELS_DIR` the rows skip with a printed SKIP line, so they hold only where a lane sets
//! `POOT_REQUIRE_MODELS=1`, which turns the skip into a failure (as does a models directory that holds
//! no Qwen3.8 GGUF). Both rows are in the `checkpoint-large` lane of `scripts/required-checkpoint-cases.txt`
//! (`just test-checkpoint-large`), which sets the variable.

use std::path::{Path, PathBuf};

use poot_load::LoadError;
use poot_load::gguf::{GgufIndex, IdentityNames, TensorNameMap, gguf_plan, read_gguf, shard_paths};
use poot_load::plan::default_workers;
use poot_quant::StoredBytes;
use poot_quant::format::WeightFormat;
use poot_quant::weights::{WeightEntry, WeightKey};
use poot_test_util::{MODELS_DIR_VARIABLE, REQUIRE_MODELS_VARIABLE, checkpoint, model_path};

/// The first shard of every Qwen3.8 GGUF set under the models directory, or `None` after reporting a
/// skip when the directory is unset.
fn local_qwen38_files() -> Option<Vec<PathBuf>> {
    let required = std::env::var(REQUIRE_MODELS_VARIABLE).as_deref() == Ok("1");
    let Some(models) = std::env::var(MODELS_DIR_VARIABLE)
        .ok()
        .filter(|dir| !dir.is_empty())
    else {
        assert!(
            !required,
            "{MODELS_DIR_VARIABLE} is unset but {REQUIRE_MODELS_VARIABLE}=1"
        );
        eprintln!("SKIP: the Qwen3.8 GGUF files need {MODELS_DIR_VARIABLE}, which is unset");
        return None;
    };
    let mut files = Vec::new();
    for directory in std::fs::read_dir(&models).unwrap() {
        let directory = directory.unwrap().path();
        let named = directory
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.to_lowercase().contains("qwen3.8"));
        if !named || !directory.is_dir() {
            continue;
        }
        for file in std::fs::read_dir(&directory).unwrap() {
            let file = file.unwrap().path();
            if file.extension().is_some_and(|ext| ext == "gguf") {
                // A split set is planned once, from its first shard.
                files.push(shard_paths(&file).swap_remove(0));
            }
        }
    }
    files.sort();
    files.dedup();
    assert!(
        !(required && files.is_empty()),
        "{REQUIRE_MODELS_VARIABLE}=1 but {models} holds no Qwen3.8 GGUF"
    );
    if files.is_empty() {
        eprintln!("SKIP: no Qwen3.8 GGUF under {models}");
        return None;
    }
    Some(files)
}

/// The header index of the file or split set starting at `first`.
fn open(first: &Path) -> GgufIndex {
    let shards: Vec<StoredBytes> = shard_paths(first)
        .iter()
        .map(|path| StoredBytes::map(std::fs::File::open(path).unwrap()).unwrap())
        .collect();
    GgufIndex::parse_shards(&shards).unwrap()
}

#[test]
fn every_tensor_of_every_local_qwen38_gguf_maps_to_a_format() {
    let Some(files) = local_qwen38_files() else {
        return;
    };
    for file in &files {
        let index = open(file);
        let plan = gguf_plan(&index, &IdentityNames).unwrap_or_else(|e| match e {
            LoadError::UnsupportedGgufType { tensor, ggml_type } => panic!(
                "{}: tensor {tensor} has ggml type {ggml_type}, which has no format",
                file.display()
            ),
            other => panic!("{}: {other}", file.display()),
        });
        // Every tensor of the file is one owner of the plan (a quantized rank-2 tensor is one owner).
        assert_eq!(plan.len(), index.tensors.len(), "{}", file.display());
    }
}

/// A name map that keeps one tensor.
struct Only(&'static str);

impl TensorNameMap for Only {
    fn weight_key(&self, gguf_name: &str) -> Option<WeightKey> {
        (gguf_name == self.0).then(|| WeightKey::from(gguf_name))
    }
}

/// The 27B file's `IQ3_S` tensor, the one the review baseline refused, plans as exactly one packed
/// `Iq3_S` owner of its shape `[5120, 17408]`, viewing its own bytes of the mapped file.
#[test]
fn the_27b_iq3_s_tensor_plans_as_one_packed_iq3_s_owner() {
    let Some(path) = model_path(checkpoint!("Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf")) else {
        return;
    };
    let index = open(&path);
    assert_eq!(index.tensors["blk.14.ffn_down.weight"].ggml_type, 21);
    let store = read_gguf(&index, &Only("blk.14.ffn_down.weight"), default_workers())
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(store.len(), 1);
    let WeightEntry::Packed(payload) = store.get("blk.14.ffn_down.weight").expect("one owner")
    else {
        panic!("blk.14.ffn_down.weight is not a packed owner");
    };
    assert_eq!(payload.weight().format(), WeightFormat::Iq3_S);
    assert_eq!(payload.weight().shape(), [5120, 17408]);
}

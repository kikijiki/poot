//! POOT-1224 SC-002: the driver names no family, no family slot tag and no model-specific input.
//!
//! Adding a model touches only the model (ADR-0104 rule 3), so the driver's production source may not
//! spell a family (`bert`, `idefics`, `qwen`) or a family's request input (`mrope`, `pixels`,
//! `token_type`): such a name in the driver is a per-family arm waiting to be written. The driver binds
//! every input by kind from the model's schema ([`bind`]); this row reads the source and fails on the
//! first name that says otherwise. The driver's own test modules run real checkpoints and fixture
//! families on purpose, so only production files are read.

use std::path::{Path, PathBuf};

/// The names the driver's production source must not contain, matched case-insensitively.
const FAMILY_NAMES: [&str; 5] = ["bert", "idefics", "qwen", "mrope", "pixels"];

/// The driver's source directory.
fn driver_dir() -> PathBuf {
    poot_test_util::manifest_dir().join("src/driver")
}

/// Whether `path` is a test module of the driver: its fixtures and device rows name real families.
fn is_test_source(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    name == "tests.rs"
        || name.ends_with("_tests.rs")
        || path
            .components()
            .any(|part| part.as_os_str() == "device_tests")
}

fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, found);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// The first violation in `line`, if any. `token_type` is a family input (BERT's segment ids) except as
/// part of the GGUF tokenizer's own metadata key `tokenizer.ggml.token_type`.
fn violation(line: &str) -> Option<&'static str> {
    let lower = line.to_ascii_lowercase();
    if let Some(name) = FAMILY_NAMES.into_iter().find(|name| lower.contains(name)) {
        return Some(name);
    }
    let bare = lower.replace("tokenizer.ggml.token_type", "");
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    bare.match_indices("token_type")
        .any(|(at, hit)| !bare[..at].ends_with(word) && !bare[at + hit.len()..].starts_with(word))
        .then_some("token_type")
}

/// SC-002: no family name or family input in `crates/poot-llm/src/driver`. Mutation: re-add a family
/// slot tag to the driver (`const BERT_SLOT: &str = "token_type";` in `bind.rs`); this row names the
/// file, line and word and is red.
#[test]
fn the_driver_source_names_no_family_and_no_family_input() {
    let mut files = Vec::new();
    rust_files(&driver_dir(), &mut files);
    files.retain(|path| !is_test_source(path));
    assert!(
        files.len() > 10,
        "the driver's production files were not found under {}",
        driver_dir().display()
    );
    let mut found = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        for (number, line) in source.lines().enumerate() {
            if let Some(name) = violation(line) {
                found.push(format!(
                    "{}:{}: `{name}`: {}",
                    path.display(),
                    number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        found.is_empty(),
        "the driver names a family or a family input:\n{}",
        found.join("\n")
    );
}

/// The matcher can fail: it flags each family name and the bare `token_type`, and spares the GGUF
/// tokenizer's own key and the plural `token_types` of its vocabulary reader.
#[test]
fn the_matcher_flags_family_names_and_spares_the_gguf_key() {
    for line in [
        "const SLOT: &str = \"BERT_slot\";",
        "// the idefics3 tower",
        "let qwen = 1;",
        "Slot::MropePosition",
        "let pixels = vec![];",
        "activation_slot(\"token_type\")",
    ] {
        assert!(violation(line).is_some(), "{line}");
    }
    for line in [
        "index.get(\"tokenizer.ggml.token_type\")",
        "token_types: index",
        "let kind = 1;",
    ] {
        assert!(violation(line).is_none(), "{line}");
    }
}

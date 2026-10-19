//! POOT-1226 SC-001: `poot-llm` and `poot-load` never read a config's `model_type` to choose what to do.
//!
//! The registry is the one place a checkpoint's `model_type` becomes a family (ADR-0104): a ladder of
//! `model_type` sniffs in the driver crate or the loader is a second support list (the deleted
//! `read_image_checkpoint` was one, and so was the `model_type == "qwen2_5_vl"` branch of the caption
//! binary). This row reads both crates' sources, binaries and test modules included, and fails on the
//! first line that reads the key: a quoted `"model_type"` that is not the key of a JSON literal
//! (`"model_type": ...`, which a fixture writes) is a lookup (`config["model_type"]`,
//! `.get("model_type")`) or a comparison.

use std::path::{Path, PathBuf};

/// The source directories that may not read `model_type`.
fn source_dirs() -> [PathBuf; 2] {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    [manifest.join("src"), manifest.join("../poot-load/src")]
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

/// Whether `line` ends a string at `model_type` (a key, a path or a bare name) other than as the key of a
/// JSON literal.
fn reads_model_type(line: &str) -> bool {
    const KEY: &str = "model_type\"";
    line.match_indices(KEY)
        .any(|(at, _)| !line[at + KEY.len()..].trim_start().starts_with(':'))
}

/// The `model_type` values of the shipped vision-language families: a quoted one in either crate is a
/// family the driver crate or the loader names.
const FAMILY_LITERALS: [&str; 2] = ["\"qwen2_5_vl\"", "\"idefics3\""];

/// Whether `line` quotes the `model_type` of a shipped vision-language family.
fn names_a_family(line: &str) -> bool {
    FAMILY_LITERALS.iter().any(|literal| line.contains(literal))
}

/// SC-001: no read of `"model_type"` in `poot-llm/src` or `poot-load/src`. Mutation: re-add the ladder,
/// `config.get("model_type").and_then(serde_json::Value::as_str) == Some("idefics3")`, anywhere under
/// either directory (it was in `multimodal/front_end.rs`), or quote `"idefics3"` or `"qwen2_5_vl"` in
/// either (`entry.family.as_str() == "idefics3"`); this row names the file, line and text and is red.
#[test]
fn the_driver_crate_and_the_loader_never_read_model_type() {
    let mut files = Vec::new();
    for dir in source_dirs() {
        rust_files(&dir, &mut files);
    }
    assert!(
        files.len() > 50,
        "the sources were not found under {:?}",
        source_dirs()
    );
    let mut found = Vec::new();
    for path in &files {
        let source = std::fs::read_to_string(path).unwrap();
        for (number, line) in source.lines().enumerate() {
            if reads_model_type(line) || names_a_family(line) {
                found.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a config's model_type is read, or a vision-language family is named, outside the registry:\n{}",
        found.join("\n")
    );
}

/// The matcher can fail: it flags the lookups and comparisons the deleted ladders spelled, and spares
/// the key of a JSON literal and the identifier of a field.
#[test]
fn the_matcher_flags_reads_and_spares_json_keys() {
    for line in [
        r#"config.get("model_type").and_then(serde_json::Value::as_str) == Some("idefics3")"#,
        r#"if config["model_type"] == "qwen2_5_vl" {"#,
        r#"let kind = raw.pointer("/model_type");"#,
        r#"matches!(config.get("model_type"), Some(_))"#,
    ] {
        assert!(reads_model_type(line), "{line}");
    }
    for line in [
        r#"serde_json::json!({"model_type": "tiny", "granule": 4})"#,
        r#"        "model_type": "poot_fixture_carrier","#,
        r#"model_type: String,"#,
        r#"#[error("config has no rope_theta and model_type {model_type:?}")]"#,
    ] {
        assert!(!reads_model_type(line), "{line}");
    }
    assert!(names_a_family(r#"entry.family.as_str() == "idefics3""#));
    assert!(names_a_family(r#"{"model_type": "qwen2_5_vl"}"#));
    assert!(!names_a_family(
        r#"let idefics3_rows = 4; // the idefics3 tests"#
    ));
}

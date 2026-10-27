//! No Rust source bakes a checkout path into a binary (POOT-1354). The `env!` macro form of `CARGO_MANIFEST_DIR` is fixed at
//! compile time, and the shared target directories reuse a binary across worktrees by content, so a test that
//! reads files through it can read a dead or foreign checkout and fail, or pass on stale data. Tests read
//! `poot_test_util::manifest_dir()` or `repo_root()`, which look at the environment cargo sets for the run.
//!
//! The scan covers every `.rs` file under `crates/` and `benchmarks/` of the checkout the run was invoked from.

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name != "target") {
                rust_sources(&path, found);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// The 1-based lines of `text` that capture `variable` at compile time with `env!` or `option_env!`.
fn baked_lines(text: &str, variables: &[String]) -> Vec<usize> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| {
            variables.iter().any(|variable| {
                let capture = format!("env!(\"{variable}\")");
                line.contains(&capture)
            })
        })
        .map(|(index, _)| index + 1)
        .collect()
}

fn path_variables() -> Vec<String> {
    // Built from parts so this file does not match itself.
    ["MANIFEST", "WORKSPACE"]
        .iter()
        .map(|kind| format!("CARGO_{kind}_DIR"))
        .collect()
}

#[test]
fn no_source_bakes_a_checkout_path_at_compile_time() {
    let root = poot_test_util::repo_root();
    let mut files = Vec::new();
    rust_sources(&root.join("crates"), &mut files);
    rust_sources(&root.join("benchmarks"), &mut files);
    assert!(
        files.len() > 100,
        "the scan found only {} files",
        files.len()
    );
    let variables = path_variables();
    let mut offenders = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for line in baked_lines(&text, &variables) {
            offenders.push(format!("{}:{line}", file.display()));
        }
    }
    assert!(
        offenders.is_empty(),
        "a compile-time checkout path (use poot_test_util::manifest_dir() or repo_root()): {offenders:#?}"
    );
}

#[test]
fn the_scan_recognises_a_baked_path() {
    let variables = path_variables();
    let baked = format!("let a = Path::new(env!(\"{}\"));\n", variables[0]);
    assert_eq!(baked_lines(&baked, &variables), [1]);
    let runtime = format!("let a = std::env::var(\"{}\");\n", variables[0]);
    assert!(baked_lines(&runtime, &variables).is_empty());
}

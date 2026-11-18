//! The process edges report how the process was launched (POOT-1252): library code never reads the environment,
//! so a binary that opens an executor passes `HSA_ALLOCATE_QUEUE_DEV_MEM` to the driver with
//! `poot_llm::driver::report_launch_environment`, and the ROCm production queue then refuses a process that
//! exports it (its ring would be in device memory, not the system memory the queue's publication assumes).
//!
//! Two rows: every named edge calls the report, and no other binary or example that opens an executor (or a
//! device group) lacks the call. Mutation: delete the call from `crates/poot-serve/src/main.rs`; the first row
//! names the file.

use std::path::{Path, PathBuf};

const REPORT: &str = "report_launch_environment(";

/// The process edges that open ROCm through the driver without naming `open_executor` in their own file.
const NAMED_EDGES: &[&str] = &[
    "crates/poot-serve/src/main.rs",
    "crates/poot-llm/src/bin/poot/main.rs",
    "benchmarks/runners/poot/src/main.rs",
    "benchmarks/runners/poot/src/bin/score.rs",
];

const OPENERS: &[&str] = &["open_executor(", "open_timed_executor(", "open_group("];

fn root() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR for test runs"),
    )
    .join("../..")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// Code of `source` without comment lines.
fn code(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The binary and example files of the workspace: `src/main.rs`, `src/bin/**`, `examples/**`.
fn edge_files(root: &Path) -> Vec<PathBuf> {
    let mut crate_dirs: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("a crates directory")
        .flatten()
        .map(|entry| entry.path())
        .collect();
    crate_dirs.push(root.join("benchmarks/runners/poot"));
    let mut out = Vec::new();
    for dir in crate_dirs {
        let main = dir.join("src/main.rs");
        if main.is_file() {
            out.push(main);
        }
        rust_files(&dir.join("src/bin"), &mut out);
        rust_files(&dir.join("examples"), &mut out);
    }
    out.sort();
    out
}

#[test]
fn every_named_process_edge_reports_its_launch_environment() {
    let root = root();
    let missing: Vec<&str> = NAMED_EDGES
        .iter()
        .copied()
        .filter(|edge| {
            !code(&std::fs::read_to_string(root.join(edge)).unwrap_or_default()).contains(REPORT)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "these process edges no longer call `poot_llm::driver::report_launch_environment`: {missing:?}"
    );
}

#[test]
fn no_binary_that_opens_an_executor_skips_the_report() {
    let root = root();
    let mut missing = Vec::new();
    for file in edge_files(&root) {
        let source = code(&std::fs::read_to_string(&file).unwrap_or_default());
        // A binary of a crate whose `src/bin` helpers are modules of one `main` is covered by that main.
        let opens = OPENERS.iter().any(|opener| source.contains(opener));
        if opens && !source.contains(REPORT) {
            let relative = file
                .strip_prefix(&root)
                .unwrap_or(&file)
                .display()
                .to_string();
            // Modules of the `poot` binary (`src/bin/poot/*.rs` other than `main.rs`) are reached from its main.
            if relative.starts_with("crates/poot-llm/src/bin/poot/")
                && !relative.ends_with("/main.rs")
            {
                continue;
            }
            missing.push(relative);
        }
    }
    assert!(
        missing.is_empty(),
        "binaries and examples that open an executor without reporting the launch environment: {missing:?}"
    );
}

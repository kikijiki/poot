//! A test that cannot run on this machine says so with the reserved `POOT-SKIP` token (POOT-1467). The required-case
//! gate (`scripts/check-required-cases.py`) reads that token and nothing else as a skip, so a test that prints
//! `no GPU; skipping` and returns is a green row that ran nothing, in every lane. A test that skips reports it through
//! `poot_runtime_common::device_skip` (a device) or fails (`poot_test_util::toolchain`, a tool); it never prints its own
//! free text.
//!
//! The scan covers the `src/` and `tests/` of every crate under `crates/` of the checkout the run was invoked from.
//! Examples and benches are programs, not tests, and are out of scope.

use std::path::{Path, PathBuf};

fn rust_sources(dir: &Path, found: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_sources(&path, found);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// Every `.rs` file under `src/` and `tests/` of every crate.
fn test_sources() -> Vec<(PathBuf, String)> {
    let crates = poot_test_util::repo_root().join("crates");
    let mut paths = Vec::new();
    for krate in std::fs::read_dir(&crates).unwrap().flatten() {
        for dir in ["src", "tests"] {
            let dir = krate.path().join(dir);
            if dir.is_dir() {
                rust_sources(&dir, &mut paths);
            }
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let text = std::fs::read_to_string(&path).unwrap();
            (path, text)
        })
        .collect()
}

/// Prints of a skipped-item counter, which report a statistic and skip nothing: (file name, the counter's text).
const STATISTICS: [(&str, &str); 2] = [
    ("speculate_recurrent.rs", "spec.skipped_no_proposal"),
    ("served_groups.rs", "skipped {skipped}"),
];

/// The `eprintln!(...)` and `println!(...)` calls in `text` whose message says something is being skipped, with the
/// 1-based line each starts on.
fn skip_prints(text: &str) -> Vec<(usize, &str)> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("println!(") {
        let start = from + at;
        let end = text[start..].find(");").map_or(text.len(), |e| start + e);
        let call = &text[start..end];
        let lower = call.to_lowercase();
        if lower.contains("skip") || lower.contains("no gpu") || lower.contains("no vulkan device")
        {
            found.push((text[..start].matches('\n').count() + 1, call));
        }
        from = end;
    }
    found
}

#[test]
fn a_skipping_test_reports_it_with_the_reserved_token() {
    let mut silent = Vec::new();
    for (path, text) in test_sources() {
        // This file quotes the phrases it hunts for.
        if path
            .file_name()
            .is_some_and(|name| name == "skip_reports.rs")
        {
            continue;
        }
        let file = path.file_name().unwrap().to_string_lossy();
        for (line, call) in skip_prints(&text) {
            let statistic = STATISTICS
                .iter()
                .any(|(name, counter)| *name == file && call.contains(counter));
            if !call.contains("POOT-SKIP") && !statistic {
                silent.push(format!("{}:{line}: {call}", path.display()));
            }
        }
    }
    assert!(
        silent.is_empty(),
        "these print a skip without the POOT-SKIP token; open the device through \
         poot_runtime_common::device_skip or require the tool through poot_test_util::toolchain:\n{}",
        silent.join("\n")
    );
}

#[test]
fn the_scan_reads_a_skip_that_prints_no_token() {
    let silent = r#"
        eprintln!("no GPU ({e}); skipping");
        eprintln!(
            "llc not on PATH; skipping"
        );
        println!("no Vulkan device");
    "#;
    assert_eq!(skip_prints(silent).len(), 3);
    let reported = r#"eprintln!("POOT-SKIP poot-x: case (no GPU)");"#;
    assert!(
        skip_prints(reported)
            .iter()
            .all(|(_, call)| call.contains("POOT-SKIP"))
    );
    assert!(skip_prints(r#"eprintln!("device time: {:?}", t);"#).is_empty());
}

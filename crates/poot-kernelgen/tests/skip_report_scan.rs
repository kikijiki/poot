//! Source scans over this crate's own tests (POOT-1341): a test that cannot run on this machine reports it with the
//! reserved `POOT-SKIP` token (`poot_test_util::skip_report`), because the required-case gate reads nothing else as
//! a skip, and every fixture `ptx-check` lists as compile-only has a staging test here.

use std::path::{Path, PathBuf};

fn rust_sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = poot_test_util::manifest_dir();
    let mut paths = Vec::new();
    walk(&root.join("src"), &mut paths);
    walk(&root.join("tests"), &mut paths);
    paths
        .into_iter()
        .filter(|path| path.file_name().is_some_and(|n| n != "skip_report_scan.rs"))
        .map(|path| {
            let text = std::fs::read_to_string(&path).unwrap();
            (path, text)
        })
        .collect()
}

/// The `eprintln!(...)` calls in `text` whose message says the test is skipping something, with the 1-based line each
/// starts on.
fn skip_eprintlns(text: &str) -> Vec<(usize, &str)> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("eprintln!(") {
        let start = from + at;
        let end = text[start..].find(");").map_or(text.len(), |e| start + e);
        let call = &text[start..end];
        let lower = call.to_lowercase();
        if lower.contains("skip") || lower.contains("no gpu") || lower.contains("no device") {
            found.push((text[..start].matches('\n').count() + 1, call));
        }
        from = end;
    }
    found
}

#[test]
fn a_skipping_test_reports_it_with_the_reserved_token() {
    let mut silent = Vec::new();
    for (path, text) in rust_sources() {
        for (line, call) in skip_eprintlns(&text) {
            if !call.contains("POOT-SKIP") {
                silent.push(format!("{}:{line}: {call}", path.display()));
            }
        }
    }
    assert!(
        silent.is_empty(),
        "these skips print no POOT-SKIP token; call poot_test_util::skip_report instead:\n{}",
        silent.join("\n")
    );
}

#[test]
fn every_compile_only_fixture_has_a_staging_test() {
    let sources = rust_sources();
    for label in poot_codegen::NVPTX_COMPILE_ONLY_FIXTURES {
        let quoted = format!("\"{label}\"");
        let staged = sources.iter().any(|(_, text)| {
            text.contains(&quoted)
                && (text.contains("stage_nvptx") || text.contains("ptx_stage_dir"))
        });
        assert!(
            staged,
            "ptx-check lists {label} as compile-only but no poot-kernelgen test that stages NVPTX names it"
        );
    }
}

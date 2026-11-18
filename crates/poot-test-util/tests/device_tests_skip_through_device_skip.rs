//! A required backend never turns into a silent skip (POOT-1252): a test that opens a GPU backend either fails on
//! the error (`unwrap`, `expect`, `?`) or goes through `poot_runtime_common::device_skip` (`open_or_skip`), which
//! panics when the lane requires the backend (`POOT_REQUIRE_<BACKEND>=1`) and prints the `POOT-SKIP` line the lane
//! gates read otherwise. Device constructors never read the switch themselves, so a hand-written
//! `match Context::new() { Err(_) => return }` is a skip that passes under a lane that requires the device.
//!
//! The scan reads the test code of every crate (files under `tests/`, `tests.rs` and `*_tests.rs`, modules
//! declared `#[cfg(test)]`, and everything after a column-0 `#[cfg(test)] mod`) and reports each call of a GPU
//! backend constructor that is neither followed by `.unwrap(`, `.expect(`, `?` or `.map_err` nor within reach of
//! an `open_or_skip(`. A line comment is not code. [`EXEMPT`] lists the sites that open a device on purpose
//! without it, each with its reason. The check is textual: a call split across helper functions that rethrow the
//! error is not seen, and an `open_or_skip` within reach of an unrelated constructor hides it.

use std::path::{Path, PathBuf};

/// Substrings that open a GPU backend. An unqualified `Context::new(` is wgpu's or raw Vulkan's.
const CONSTRUCTORS: &[&str] = &[
    "Context::new(",
    "Context::new_with_options(",
    "Context::new_with_device_timing(",
    "Context::open(",
    "WgpuDevice::new",
    "PtxDevice::new",
    "PtxDevice::open(",
    "RocmDevice::new",
    "RocmDevice::open(",
    "VulkanDevice::new",
    "open_executor(",
    "open_timed_executor(",
    "open_group(",
];

/// `(path suffix, reason)`: test code that opens a device without `open_or_skip`.
const EXEMPT: &[(&str, &str)] = &[
    (
        "crates/poot-serve/tests/served_multi_device.rs",
        "two-GPU rows with their own required switch, POOT_REQUIRE_PTX_2GPU, which asserts instead of skipping",
    ),
    (
        "crates/pootc/tests/macro_kernel.rs",
        "the probe is a separate program (a string compiled by rustc without poot-runtime-common) that prints POOT-SKIP; \
         the test that runs it calls `skip_unsupported`, which fails when the backend is required",
    ),
    (
        "crates/poot-llm/src/driver/open.rs",
        "the zero-timing-bound row expects the open to be refused before any device opens",
    ),
    (
        "crates/poot-llm/tests/host/",
        "host rows that name constructors in error-size fixtures and open nothing",
    ),
];

fn is_comment(line: &str) -> bool {
    line.trim_start().starts_with("//")
}

/// `source` with comment lines blanked (keeping line numbers) and string literals left alone.
fn code(source: &str) -> String {
    source
        .lines()
        .map(|line| if is_comment(line) { "" } else { line })
        .collect::<Vec<_>>()
        .join("\n")
}

fn window(text: &str, from: usize, to: usize) -> &str {
    let from = text.floor_char_boundary(from.min(text.len()));
    let to = text.floor_char_boundary(to.min(text.len()));
    &text[from..to]
}

/// Offsets of `needle` in `text` that are calls: a lowercase function name is not the tail of a longer identifier.
fn call_sites(text: &str, needle: &str) -> Vec<usize> {
    text.match_indices(needle)
        .map(|(at, _)| at)
        .filter(|&at| {
            let before = text[..at].chars().next_back();
            !needle.starts_with(|c: char| c.is_lowercase())
                || !before.is_some_and(|c| c.is_alphanumeric() || c == '_')
        })
        .collect()
}

/// Every constructor call in `text` that is neither hard-failing nor guarded by `open_or_skip`: `(line, snippet)`.
fn unguarded(text: &str) -> Vec<(usize, String)> {
    let text = code(text);
    let mut found = Vec::new();
    for needle in CONSTRUCTORS {
        for at in call_sites(&text, needle) {
            // Find the end of the call's argument list.
            let Some(paren) = text[at..].find('(') else {
                continue;
            };
            let open = at + paren + 1;
            let mut depth = 1usize;
            let mut end = open;
            for (offset, c) in text[open..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + offset + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let rest = window(&text, end, end + 40);
            let rest_trim = rest.trim_start();
            if rest_trim.starts_with(".expect(")
                || rest_trim.starts_with(".unwrap(")
                || rest_trim.starts_with('?')
                || rest_trim.starts_with(".map_err")
                || rest_trim.starts_with(".unwrap_or_else(")
            {
                continue;
            }
            let guarded = window(&text, at.saturating_sub(160), at).contains("open_or_skip")
                || window(&text, end, end + 1200).contains("open_or_skip")
                || window(&text, at.saturating_sub(500), at).contains("open_or_skip");
            if guarded {
                continue;
            }
            let line = text[..at].matches('\n').count() + 1;
            found.push((line, window(&text, at, at + 60).replace('\n', " ")));
        }
    }
    found.sort();
    found
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
    out.sort();
}

/// The text of `source` that is test code, given whether the whole file is.
fn test_text(source: &str, whole_file: bool) -> String {
    if whole_file {
        return source.to_string();
    }
    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if line.starts_with("#[cfg(test)]") || line.starts_with("#[cfg(any(test") {
            let item = lines[index + 1..]
                .iter()
                .find(|l| !l.trim_start().starts_with("#["));
            if item.is_some_and(|l| {
                let l = l.trim_start();
                (l.starts_with("mod ") || l.starts_with("pub mod ")) && l.trim_end().ends_with('{')
            }) {
                let keep = lines[..index].iter().map(|_| "").collect::<Vec<_>>();
                return keep
                    .into_iter()
                    .chain(lines[index..].iter().copied())
                    .collect::<Vec<_>>()
                    .join("\n");
            }
        }
    }
    String::new()
}

fn gated_modules(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut gated = false;
    for line in source.lines() {
        let t = line.trim();
        if t.starts_with("#[cfg(test)]") || t.starts_with("#[cfg(any(test") {
            gated = true;
            continue;
        }
        if t.starts_with("#[") || t.is_empty() {
            continue;
        }
        if gated
            && let Some(name) = t
                .strip_prefix("pub(crate) ")
                .or_else(|| t.strip_prefix("pub "))
                .unwrap_or(t)
                .strip_prefix("mod ")
                .and_then(|r| r.strip_suffix(';'))
        {
            found.push(name.trim().to_string());
        }
        gated = false;
    }
    found
}

/// `(relative path, line, snippet)` of every unguarded constructor call in the workspace's test code.
fn findings(root: &Path) -> Vec<(String, usize, String)> {
    let mut out = Vec::new();
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("a crates directory")
        .flatten()
        .map(|e| e.path())
        .collect();
    crates.sort();
    for dir in crates {
        if dir.file_name().is_some_and(|n| n == "poot-test-util") {
            continue;
        }
        let mut files = Vec::new();
        rust_files(&dir, &mut files);
        let mut gated = std::collections::BTreeSet::new();
        for file in &files {
            let src = std::fs::read_to_string(file).unwrap_or_default();
            let stem = file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            let parent = file.parent().expect("a parent");
            let base = if matches!(stem, "lib" | "main" | "mod") {
                parent.to_path_buf()
            } else {
                parent.join(stem)
            };
            for module in gated_modules(&src) {
                gated.insert(base.join(format!("{module}.rs")));
                let nested = base.join(&module);
                for inner in files.iter().filter(|f| f.starts_with(&nested)) {
                    gated.insert(inner.clone());
                }
            }
        }
        for file in &files {
            let relative = file.strip_prefix(root).expect("under the root");
            let shown = relative.display().to_string();
            let in_src = relative.components().any(|c| c.as_os_str() == "src");
            if relative.components().any(|c| c.as_os_str() == "benches")
                || relative.components().any(|c| c.as_os_str() == "examples")
                || (in_src
                    && (relative.starts_with(dir.strip_prefix(root).unwrap().join("src/bin"))
                        || shown.ends_with("/main.rs")))
            {
                continue;
            }
            let name = file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            let whole = relative.components().any(|c| c.as_os_str() == "tests")
                || name == "tests.rs"
                || name.ends_with("_tests.rs")
                || gated.contains(file);
            let src = std::fs::read_to_string(file).unwrap_or_default();
            let text = test_text(&src, whole);
            for (line, snippet) in unguarded(&text) {
                if !EXEMPT.iter().any(|(suffix, _)| shown.contains(suffix)) {
                    out.push((shown.clone(), line, snippet));
                }
            }
        }
    }
    out
}

fn workspace_root() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR for test runs"),
    )
    .join("../..")
}

/// No test opens a GPU backend in a way that skips silently under a lane that requires it.
/// Mutation: put `match Context::new() { Ok(c) => c, Err(_) => return }` back into any device test.
#[test]
fn no_test_skips_a_required_backend_silently() {
    let found = findings(&workspace_root());
    assert!(
        found.is_empty(),
        "{} test site(s) open a GPU backend without `device_skip::open_or_skip` or a hard failure:\n  {}",
        found.len(),
        found
            .iter()
            .map(|(file, line, snippet)| format!("{file}:{line}: {snippet}"))
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// The scan flags a manual skip and accepts the guarded and hard-failing spellings.
#[test]
fn the_scan_tells_a_manual_skip_from_a_guarded_open() {
    let manual = "let ctx = match Context::new() {\n    Ok(c) => c,\n    Err(_) => return,\n};\n";
    assert_eq!(unguarded(manual).len(), 1, "{:?}", unguarded(manual));
    let let_else = "let Ok(ctx) = PtxContext::new() else { return };\n";
    assert_eq!(unguarded(let_else).len(), 1);
    let ok = "let ctx = Context::new().expect(\"a device\");\nlet d = RocmDevice::new()?;\n";
    assert!(unguarded(ok).is_empty());
    let guarded = "let Some(ctx) = open_or_skip(\n    DeviceBackend::Wgpu,\n    Context::new(),\n) else { return };\n";
    assert!(unguarded(guarded).is_empty());
    let comment = "// match Context::new() { Err(_) => return }\n";
    assert!(unguarded(comment).is_empty());
    let executor = "let exec = open_executor(BackendChoice::Rocm).ok();\n";
    assert_eq!(unguarded(executor).len(), 1);
}

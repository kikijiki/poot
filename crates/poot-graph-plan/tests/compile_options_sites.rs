//! POOT-1505: every production compile plans with its devices' facts (their NPU lowering, their compiler's
//! register feedback). A device fact reaches a compile only through `CompileOptions::with_device`, so every place
//! production code builds `CompileOptions` either attaches them in the same statement or is listed below with the
//! reason it need not.
//!
//! Production code is every `.rs` file under `crates/*/src`, `crates/*/examples` and `benchmarks/`, less its inline
//! `#[cfg(test)] mod … { … }` blocks and the files whose name says test. Test harness crates and test-support
//! modules build options for fixtures and are not production.
#![allow(missing_docs)]

use std::path::{Path, PathBuf};

/// The sites that build `CompileOptions` without `with_device`, each with why it need not attach the facts. An entry
/// is `(file, the statement's first line, reason)`; an entry that matches no site fails, so the list cannot go stale.
const DEVICE_FREE: &[(&str, &str, &str)] = &[
    (
        "crates/poot-graph-plan/src/staged.rs",
        "let stage_options = CompileOptions {",
        "copies the caller's options, the devices' facts included; `for_device` gives each stage its own",
    ),
    (
        "crates/poot-llm/src/driver/mod.rs",
        "let compile = CompileOptions::new(CompileLimits::STANDARD);",
        "`DriverOptions::generation`, the caller's policy only: `Driver::compile` attaches the executor's facts to \
         every program it compiles",
    ),
];

/// Directories and files that are not production code.
fn is_test_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some("tests" | "benches" | "target" | ".patched" | "test_support")
        )
    }) || text.contains("poot-executor-parity")
        || path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("test"))
}

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_test_path(&path) {
            continue;
        }
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `text` without its inline `#[cfg(test)] mod name { … }` blocks (a module declared `mod name;` lives in its own
/// file, which the file filter judges).
fn without_test_modules(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("#[cfg(test)]") {
        let after = &rest[at + "#[cfg(test)]".len()..];
        let trimmed = after.trim_start();
        let is_inline_mod = ["mod ", "pub mod ", "pub(crate) mod "]
            .iter()
            .any(|p| trimmed.starts_with(p))
            && trimmed
                .find(['{', ';'])
                .is_some_and(|i| trimmed.as_bytes()[i] == b'{');
        if !is_inline_mod {
            out.push_str(&rest[..at + "#[cfg(test)]".len()]);
            rest = after;
            continue;
        }
        out.push_str(&rest[..at]);
        let open = after.find('{').expect("an inline module opens");
        let mut depth = 0usize;
        let mut end = after.len();
        for (i, ch) in after[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = open + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

/// Every construction of `CompileOptions` in `text`: its line, the statement's first line, and whether the statement
/// attaches the devices' facts.
fn sites(text: &str) -> Vec<(usize, String, bool)> {
    let mut found = Vec::new();
    for pattern in ["CompileOptions::new(", "CompileOptions {"] {
        let mut from = 0;
        while let Some(i) = text[from..].find(pattern) {
            let at = from + i;
            from = at + pattern.len();
            let before = text[..at].trim_end();
            // A type position (`-> CompileOptions {`, `impl CompileOptions {`, `struct CompileOptions {`) builds
            // nothing.
            if pattern == "CompileOptions {"
                && ["->", "impl", "struct", "for"]
                    .iter()
                    .any(|word| before.ends_with(word))
            {
                continue;
            }
            let line_start = text[..at].rfind('\n').map_or(0, |i| i + 1);
            let statement_end = text[at..].find(';').map_or(text.len(), |i| at + i);
            let first_line = text[line_start..]
                .lines()
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            let attaches = text[at..statement_end].contains(".with_device(");
            found.push((text[..at].matches('\n').count() + 1, first_line, attaches));
        }
    }
    found
}

#[test]
fn every_compile_options_site_takes_the_device_options() {
    let root = poot_test_util::repo_root();
    let mut files = Vec::new();
    let crates = std::fs::read_dir(root.join("crates")).expect("the crates directory");
    for krate in crates.flatten() {
        for sub in ["src", "examples"] {
            sources(&krate.path().join(sub), &mut files);
        }
    }
    sources(&root.join("benchmarks"), &mut files);
    files.sort();
    let mut unlisted = Vec::new();
    let mut used = vec![false; DEVICE_FREE.len()];
    let mut checked = 0;
    for path in &files {
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let relative = relative.trim_start_matches("./").to_string();
        let text = without_test_modules(&std::fs::read_to_string(path).expect("a readable source"));
        for (line, first, attaches) in sites(&text) {
            checked += 1;
            if attaches {
                continue;
            }
            match DEVICE_FREE
                .iter()
                .position(|(file, start, _)| relative.ends_with(file) && first == *start)
            {
                Some(index) => used[index] = true,
                None => unlisted.push(format!("{relative}:{line}: {first}")),
            }
        }
    }
    assert!(
        checked >= 3,
        "the scan found {checked} sites: it is not reading the sources"
    );
    assert!(
        unlisted.is_empty(),
        "CompileOptions built without `.with_device(..)` and not listed as device-free:\n{}",
        unlisted.join("\n")
    );
    let stale: Vec<_> = DEVICE_FREE
        .iter()
        .zip(&used)
        .filter(|(_, used)| !**used)
        .map(|((file, start, _), _)| format!("{file}: {start}"))
        .collect();
    assert!(
        stale.is_empty(),
        "device-free entries that match no site: {stale:?}"
    );
}

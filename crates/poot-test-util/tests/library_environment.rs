//! Library crates read no environment (R2-I2-013, POOT-1252 SC-002): execution choices are typed options a
//! caller passes in, never `env::var` in library code (architecture rule R2, ADR-0104 decision 5).
//!
//! The check walks every `crates/*/src` tree of a crate that has a library target and reports each
//! `env::var`, `env::var_os`, `env::vars`, `env::vars_os` and `option_env!` in code that is not test
//! harness code. These are not library code: a binary (`src/main.rs`, `src/bin/**`), a build script
//! (`build.rs`, outside `src`), an integration test, a module declared `#[cfg(test)]` (or
//! `#[cfg(any(test, ..))]`) and everything after a column-0 `#[cfg(test)]` module in a file. A line comment is
//! not code. The allowed exceptions are the [`ALLOWED_FILES`] and [`EXCLUDED_CRATES`] tables, each with its
//! reason; a row that no longer reads the environment fails, so the tables shrink.
//!
//! `cargo` is not needed: the workspace root is `CARGO_MANIFEST_DIR/../..`, read at run time (`CARGO_MANIFEST_DIR`
//! through `std::env::var`, never `env!`: the compile cache reuses compile-time values across worktrees).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Library files that read the process environment on purpose: `(path under the workspace root, reason)`.
const ALLOWED_FILES: &[(&str, &str)] = &[
    (
        "crates/poot-codegen/src/environment.rs",
        "the kernel compiler's process boundary: the tools the devshell exports and the machine-wide cache \
         location; every other file takes them as values (`Toolchain`, `CacheEnv`) and tests inject them",
    ),
    (
        "crates/poot-vulkan-runtime/src/validation.rs",
        "`POOT_VULKAN_VALIDATION` is the raw Vulkan lane's switch for the Khronos validation layer; moving it \
         into `ContextOptions` needs the option plumbed through the driver so the llm and serve rows of that \
         lane keep running under the layer (owed, POOT-1252 report)",
    ),
];

/// Crates whose job is the process environment, not a library the engine links.
const EXCLUDED_CRATES: &[(&str, &str)] = &[
    (
        "poot-orchestrator",
        "a tool that provisions pods and runs benchmarks: its inputs are the host's home, state directory, \
         hostname and API keys",
    ),
    (
        "poot-test-util",
        "test harness: model directories, required-backend switches and toolchain paths for tests",
    ),
    (
        "poot-build-identity",
        "build-script support: reads the variables cargo sets for a build script",
    ),
];

/// One environment read found in code.
#[derive(Debug, PartialEq, Eq)]
struct Read {
    line: usize,
    text: String,
}

const PATTERNS: &[&str] = &[
    "env::var(",
    "env::var_os(",
    "env::vars(",
    "env::vars_os(",
    "option_env!(",
];

/// Whether the item after the attributes at the head of `rest` is an inline module (`mod name {`).
fn opens_inline_module(rest: &[&str]) -> bool {
    let item = rest
        .iter()
        .find(|line| !line.trim_start().starts_with("#["));
    item.is_some_and(|line| {
        let line = line.trim_start();
        let line = line
            .strip_prefix("pub(crate) ")
            .or_else(|| line.strip_prefix("pub "))
            .unwrap_or(line);
        line.starts_with("mod ") && line.trim_end().ends_with('{')
    })
}

/// The code of `source` with line comments removed: a doc comment or `//` note naming `env::var` is not a read.
fn without_line_comments(line: &str) -> &str {
    line.find("//").map_or(line, |at| &line[..at])
}

/// Every environment read in `source` before its first column-0 `#[cfg(test)]` / `#[cfg(any(test, ..))]`
/// inline module, outside line comments.
fn reads(source: &str) -> Vec<Read> {
    let lines: Vec<&str> = source.lines().collect();
    let mut found = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        if (line.starts_with("#[cfg(test)]") || line.starts_with("#[cfg(any(test"))
            && opens_inline_module(&lines[index + 1..])
        {
            break;
        }
        let code = without_line_comments(line);
        if PATTERNS.iter().any(|pattern| code.contains(pattern)) {
            found.push(Read {
                line: index + 1,
                text: line.trim().to_string(),
            });
        }
    }
    found
}

/// The out-of-line modules `source` declares under a `#[cfg(test)]` / `#[cfg(any(test, ..))]` attribute.
fn test_only_modules(source: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut gated = false;
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#[cfg(test)]") || trimmed.starts_with("#[cfg(any(test") {
            gated = true;
            continue;
        }
        if trimmed.starts_with("#[") || trimmed.is_empty() {
            continue;
        }
        if gated {
            let rest = trimmed
                .strip_prefix("pub(crate) ")
                .or_else(|| trimmed.strip_prefix("pub "))
                .unwrap_or(trimmed);
            if let Some(name) = rest.strip_prefix("mod ").and_then(|r| r.strip_suffix(';')) {
                found.push(name.trim().to_string());
            }
        }
        gated = false;
    }
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

/// The files under `src` that are test-only because a parent declares their module `#[cfg(test)]`.
fn test_only_files(src: &Path, files: &[PathBuf]) -> BTreeSet<PathBuf> {
    let mut gated = BTreeSet::new();
    for file in files {
        let source = std::fs::read_to_string(file).unwrap_or_default();
        let modules = test_only_modules(&source);
        if modules.is_empty() {
            continue;
        }
        let stem = file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        let parent = file.parent().expect("a file has a parent");
        let dir = if matches!(stem, "lib" | "main" | "mod") {
            parent.to_path_buf()
        } else {
            parent.join(stem)
        };
        for module in modules {
            gated.insert(dir.join(format!("{module}.rs")));
            let nested = dir.join(&module);
            gated.insert(nested.join("mod.rs"));
            for inner in files.iter().filter(|f| f.starts_with(&nested)) {
                gated.insert(inner.clone());
            }
        }
    }
    gated.retain(|path| path.starts_with(src));
    gated
}

/// Whether `file` (under a library crate's `src`) is test harness or binary code by where it lives.
fn is_harness_path(src: &Path, file: &Path) -> bool {
    let relative = file.strip_prefix(src).expect("a file under src");
    let name = relative
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    relative.starts_with("bin")
        || name == "main.rs"
        || name == "tests.rs"
        || name.ends_with("_tests.rs")
        || relative.components().any(|c| c.as_os_str() == "tests")
        || relative.components().any(|c| {
            c.as_os_str()
                .to_str()
                .is_some_and(|s| s.starts_with("test_support"))
        })
}

/// `(path relative to the root, reads)` for every library file with at least one read.
fn library_reads(root: &Path) -> Vec<(String, Vec<Read>)> {
    let mut crates: Vec<PathBuf> = std::fs::read_dir(root.join("crates"))
        .expect("the workspace has a crates directory")
        .flatten()
        .map(|entry| entry.path())
        .collect();
    crates.sort();
    let mut out = Vec::new();
    for dir in crates {
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let src = dir.join("src");
        if !src.join("lib.rs").is_file() || EXCLUDED_CRATES.iter().any(|(krate, _)| *krate == name)
        {
            continue;
        }
        let mut files = Vec::new();
        rust_files(&src, &mut files);
        let gated = test_only_files(&src, &files);
        for file in &files {
            if is_harness_path(&src, file) || gated.contains(file) {
                continue;
            }
            let source = std::fs::read_to_string(file).expect("a readable source file");
            let found = reads(&source);
            if !found.is_empty() {
                let relative = file.strip_prefix(root).expect("a file under the root");
                out.push((relative.display().to_string(), found));
            }
        }
    }
    out
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR for test runs"),
    );
    manifest_dir.join("../..")
}

/// SC-002: no library file outside the allowed table reads the environment, and every allowed row still does.
/// Mutation: add `let _ = std::env::var("X");` to any library source (not a test module); this row names it.
#[test]
fn library_crates_read_no_environment() {
    let found = library_reads(&workspace_root());
    let mut problems = Vec::new();
    for (file, reads) in &found {
        if ALLOWED_FILES.iter().any(|(allowed, _)| allowed == file) {
            continue;
        }
        for read in reads {
            problems.push(format!(
                "{file}:{}: `{}` reads the process environment; take the value as a typed option from the caller",
                read.line, read.text
            ));
        }
    }
    for (allowed, _) in ALLOWED_FILES {
        if !found.iter().any(|(file, _)| file == allowed) {
            problems.push(format!(
                "ALLOWED_FILES row `{allowed}` no longer reads the environment (or no longer exists); delete the row"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "{} environment read(s) in library crates:\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}

/// The scan finds every spelling of a read, ignores comments, and stops at a `#[cfg(test)]` module.
#[test]
fn the_scan_finds_every_spelling_and_ignores_what_is_not_code() {
    let source = "\
//! Doc naming std::env::var(\"X\") is not a read.
fn a() { let _ = std::env::var(\"A\"); }
fn b() { let _ = std::env::var_os(\"B\"); } // env::vars( in a trailing note
fn c() { let _ = std::env::vars(); }
const D: Option<&str> = option_env!(\"D\");
fn e() { let _ = env::vars_os(); }
// std::env::var(\"F\")
#[cfg(test)]
mod tests {
    fn g() { let _ = std::env::var(\"G\"); }
}
";
    let lines: Vec<usize> = reads(source).iter().map(|read| read.line).collect();
    assert_eq!(lines, [2, 3, 4, 5, 6]);
}

/// A `#[cfg(test)]` out-of-line module and everything under it is test-only; an ungated module is not.
#[test]
fn modules_declared_cfg_test_are_test_only() {
    let root = std::env::temp_dir().join(format!("poot-library-env-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let src = root.join("crates/demo/src");
    std::fs::create_dir_all(src.join("probe")).unwrap();
    std::fs::create_dir_all(src.join("kept")).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        "#[cfg(test)]\nmod probe;\n#[cfg(any(test, feature = \"test-support\"))]\npub mod fixtures;\nmod kept;\n",
    )
    .unwrap();
    for (file, body) in [
        ("probe/mod.rs", "fn p() { let _ = std::env::var(\"P\"); }\n"),
        ("fixtures.rs", "fn f() { let _ = std::env::var(\"F\"); }\n"),
        ("kept/mod.rs", "fn k() { let _ = std::env::var(\"K\"); }\n"),
    ] {
        std::fs::write(src.join(file), body).unwrap();
    }
    let found = library_reads(&root);
    let _ = std::fs::remove_dir_all(&root);
    let files: Vec<&str> = found.iter().map(|(file, _)| file.as_str()).collect();
    assert_eq!(files, ["crates/demo/src/kept/mod.rs"]);
}

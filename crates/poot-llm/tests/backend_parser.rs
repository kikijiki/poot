//! POOT-1230 SC-001: a backend name is parsed in one place. `BackendChoice` has exactly one `FromStr`, and
//! no binary, example or server source matches a backend name itself. The scan reads source text, so a
//! parser written next to a flag fails here without running anything.

use std::fs;
use std::path::{Path, PathBuf};

/// The workspace root: this package is `crates/poot-llm`.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the workspace root")
}

/// Every `.rs` file under `dir`, depth first.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
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

fn sources(under: &[&str]) -> Vec<(PathBuf, String)> {
    let root = root();
    let mut files = Vec::new();
    for dir in under {
        rust_files(&root.join(dir), &mut files);
    }
    files
        .into_iter()
        .map(|path| {
            let text = fs::read_to_string(&path).expect("a readable source file");
            (path, text)
        })
        .collect()
}

/// Exactly one `impl FromStr for BackendChoice` in the workspace. Mutation: add a second one (a copy under
/// another path); the count is two and the row goes red.
#[test]
fn backend_choice_has_exactly_one_from_str() {
    let mut impls = Vec::new();
    for (path, text) in sources(&["crates", "benchmarks"]) {
        for (number, line) in text.lines().enumerate() {
            let line = line.trim_start();
            if line.starts_with("impl")
                && line.contains("FromStr for")
                && line.contains("BackendChoice")
            {
                impls.push(format!("{}:{}", path.display(), number + 1));
            }
        }
    }
    assert_eq!(
        impls.len(),
        1,
        "expected one parser of BackendChoice: {impls:?}"
    );
}

/// No binary, example, server source or benchmark runner matches a backend name: they take `--backend`
/// through `BackendChoice`'s parser. Mutation: add a local `match name { "wgpu" => ... }` to a binary; the
/// scan finds it and the row goes red.
#[test]
fn no_binary_matches_a_backend_name() {
    const NAMES: [&str; 5] = ["wgpu", "rocm", "ptx", "vulkan", "auto"];
    let mut matches = Vec::new();
    let scanned = sources(&[
        "crates/poot-llm/src/bin",
        "crates/poot-llm/examples",
        "crates/poot-serve/src",
        "benchmarks/runners",
    ]);
    assert!(!scanned.is_empty(), "the scan found no sources");
    for (path, text) in scanned {
        // The server's unit tests list backend names to exercise the parser; they are not a parser.
        if path.components().any(|part| part.as_os_str() == "tests") {
            continue;
        }
        for (number, line) in text.lines().enumerate() {
            for name in NAMES {
                if line.contains(&format!("\"{name}\" =>"))
                    || line.contains(&format!("Some(\"{name}\")"))
                {
                    matches.push(format!(
                        "{}:{}: {}",
                        path.display(),
                        number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(
        matches.is_empty(),
        "a backend name is matched outside the parser:\n{}",
        matches.join("\n")
    );
}

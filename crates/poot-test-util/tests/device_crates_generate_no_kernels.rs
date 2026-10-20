//! POOT-1235: a device crate is mechanism, not semantic work (POOT-ADR-0102 decision 2): it loads and launches
//! the kernels the engine compiles and generates none of its own. The state-commit copy is the engine's
//! kernel (`CopyMechanism::Kernel`) or the runtime's native buffer copy, never a body a device builds.
//!
//! The check reads the library sources of every `crates/poot-*-device` crate and refuses a kernel-generation
//! marker in them: building a body (`poot_kernelgen`, `movement_body`, an imported asset), resolving or
//! compiling one (`KernelCache`, `load_or_compile`, `compile_kernel`), or wrapping a compiled artifact into
//! a loadable handle (`kernel_handle`, `CompiledKernel::new`). Test modules (`src/tests/`, any item gated by
//! `#[cfg(test)]`; only that item is skipped, code after it is scanned) and the diagnostic binaries under
//! `src/bin/`, which are probes that compile their own
//! fixtures, are not the device and are skipped.
//!
//! `CARGO_MANIFEST_DIR` is read at run time, never with `env!`: the compile cache reuses compile-time
//! values across worktrees.

use std::fs;
use std::path::{Path, PathBuf};

/// What generating a kernel looks like in source.
const GENERATION_MARKERS: &[&str] = &[
    "poot_kernelgen",
    "movement_body",
    "ImportedKernel",
    "KernelCache",
    "load_or_compile",
    "compile_kernel",
    "kernel_handle",
    "CompiledKernel::new",
];

fn workspace_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    Path::new(&manifest)
        .ancestors()
        .nth(2)
        .expect("the manifest dir is crates/<name>")
        .to_path_buf()
}

/// `crates/poot-*-device/src` of every device crate.
fn device_sources(root: &Path) -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = fs::read_dir(root.join("crates"))
        .expect("read crates/")
        .map(|entry| entry.expect("crate entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("poot-") && name.ends_with("-device"))
        })
        .map(|path| path.join("src"))
        .filter(|src| src.is_dir())
        .collect();
    sources.sort();
    sources
}

/// Every library `.rs` file under `dir`, leaving out `bin/` and `tests/`.
fn library_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read a source dir") {
        let path = entry.expect("source entry").path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if name != "bin" && name != "tests" {
                library_files(&path, out);
            }
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The `(line number, text)` of every line outside a `#[cfg(test)]` item: the attribute, any attributes
/// stacked under it, and the item it gates (a `mod name;` line, or a braced block through its closing brace).
/// Only the gated item is dropped; production code after it is still production code.
fn production_lines(source: &str) -> Vec<(usize, &str)> {
    let mut kept = Vec::new();
    let mut gated = false;
    let mut depth = 0usize;
    for (index, line) in source.lines().enumerate() {
        let code = line.trim_start();
        if depth > 0 {
            depth = (depth + line.matches('{').count()).saturating_sub(line.matches('}').count());
        } else if code.starts_with("#[cfg(test)]") {
            gated = true;
        } else if gated && code.starts_with("#[") {
            // another attribute on the gated item
        } else if gated {
            gated = false;
            if line.contains('{') {
                depth = line
                    .matches('{')
                    .count()
                    .saturating_sub(line.matches('}').count());
            }
        } else {
            kept.push((index + 1, line));
        }
    }
    kept
}

/// Each `file:line: marker` in `source` that generates a kernel. A doc or line comment is prose, not code.
fn generation_in(source: &str) -> Vec<(usize, &'static str)> {
    production_lines(source)
        .into_iter()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .flat_map(|(index, line)| {
            GENERATION_MARKERS
                .iter()
                .filter(move |marker| line.contains(**marker))
                .map(move |marker| (index, *marker))
        })
        .collect()
}

#[test]
fn device_crates_generate_no_kernels() {
    let root = workspace_root();
    let sources = device_sources(&root);
    assert!(
        sources.len() >= 4,
        "the ptx, rocm, vulkan and wgpu device crates are all found, got {sources:?}"
    );
    let mut found = Vec::new();
    for src in &sources {
        let mut files = Vec::new();
        library_files(src, &mut files);
        for file in files {
            let text = fs::read_to_string(&file).expect("read a device source");
            for (line, marker) in generation_in(&text) {
                found.push(format!(
                    "{}:{line}: {marker}",
                    file.strip_prefix(&root).unwrap_or(&file).display()
                ));
            }
        }
    }
    assert!(
        found.is_empty(),
        "a device crate generates a kernel (the engine owns every kernel, POOT-1235):\n{}",
        found.join("\n")
    );
}

#[test]
fn the_scan_sees_a_generator_in_production_code_and_not_in_prose_or_tests() {
    let production = "fn copy() {\n    let body = poot_kernelgen::movement_body();\n}\n";
    assert_eq!(
        generation_in(production),
        [(2, "poot_kernelgen"), (2, "movement_body")]
    );
    let prose = "//! The engine's KernelCache compiles it.\n/// kernel_handle is the engine's.\nfn copy() {}\n";
    assert!(generation_in(prose).is_empty());
    let tested =
        "fn copy() {}\n#[cfg(test)]\nmod tests {\n    fn f() { let _ = KernelCache::open(); }\n}\n";
    assert!(generation_in(tested).is_empty());
    let declared = "#[cfg(test)]\nmod tests;\n#[cfg(test)]\n#[allow(dead_code)]\nmod more;\nconst _M: &str = \"kernel_handle\";\n";
    assert_eq!(generation_in(declared), [(6, "kernel_handle")]);
    let after_block = "#[cfg(test)]\nmod tests {\n    fn f() {\n        g();\n    }\n}\nfn copy() { KernelCache::open(); }\n";
    assert_eq!(generation_in(after_block), [(7, "KernelCache")]);
}

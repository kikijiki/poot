//! Source scan over this crate's own tests (POOT-1341, POOT-1467): every fixture `ptx-check` lists as compile-only has a staging
//! test here. (That a skipping test reports it with the reserved `POOT-SKIP` token is scanned workspace-wide by
//! `poot-test-util`'s `skip_reports` test.)

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
        .filter(|path| {
            path.file_name()
                .is_some_and(|n| n != "compile_only_fixtures.rs")
        })
        .map(|path| {
            let text = std::fs::read_to_string(&path).unwrap();
            (path, text)
        })
        .collect()
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

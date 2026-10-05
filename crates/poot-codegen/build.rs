//! Hash `poot-codegen`'s `src/**/*.rs` tree and export it as `POOT_CODEGEN_SRC_HASH` (card 211). The
//! persistent kernel-artifact cache folds this into its toolchain epoch so an emitter edit (e.g.
//! `src/emit.rs`) invalidates the cache even though it does not change the `Body` the content-addressed
//! half of the key covers. A comment-only edit costs one recompile; a hand-bumped constant would risk
//! serving stale kernels.
use std::path::{Path, PathBuf};

fn fnv1a(data: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn main() {
    // `std::env::var`, not `env!`: the latter is a compile-time macro baked into this build script's own
    // compiled binary. A shared compile cache (kache) that reuses that binary across worktrees (build.rs's
    // source text, and so its content hash, is identical everywhere) would then bake in whichever
    // worktree's path happened to compile it first, silently hashing another worktree's `src` (or a path
    // that does not exist yet in a fresh worktree) instead of this run's own manifest dir (card 530). `std::env::var` reads the environment cargo sets for *this* build-script invocation,
    // which is correct regardless of which worktree compiled the binary.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo when running a build script");
    let src_dir = Path::new(&manifest_dir).join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    // Sort: directory iteration order is not guaranteed.
    files.sort();
    // A zero-file hash is indistinguishable from "nothing changed": it is the FNV offset basis, and a
    // build-script bug (a wrong or missing src_dir) must not silently serve that as a valid epoch key -
    // it would let the persistent kernel cache serve pre-mutation artifacts forever.
    assert!(
        !files.is_empty(),
        "POOT_CODEGEN_SRC_HASH: found zero .rs files under {}; refusing to publish an empty-tree hash",
        src_dir.display()
    );

    let mut hash: u64 = 0xcbf29ce484222325;
    for file in &files {
        let rel = file.strip_prefix(&src_dir).unwrap_or(file);
        hash = fnv1a(rel.to_string_lossy().as_bytes(), hash);
        if let Ok(contents) = std::fs::read(file) {
            hash = fnv1a(&contents, hash);
        }
    }

    println!("cargo:rustc-env=POOT_CODEGEN_SRC_HASH={hash:016x}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=build.rs");
}

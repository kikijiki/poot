//! Embeds the git sha this binary is built from as `POOT_BUILD_SHA`, so every benchmark result can name the
//! commit of the binary that produced it (not the checkout the harness happens to run in).
//!
//! Inside a git checkout the sha is the checkout's HEAD, with a `-dirty` suffix when a tracked build input
//! differs from HEAD; git is the only source of truth there. A build from a source export with no `.git`
//! supplies the sha in the `POOT_BUILD_SHA` environment variable instead. In a checkout that variable is only
//! cross-checked: a value that is not HEAD fails the build, so a stale exported variable cannot mislabel a
//! binary. With neither source the build fails: a binary with no known origin must not be benchmarked.
//! A sha must be 40 lowercase hex digits, optionally followed by `-dirty` (the harness enforces the same in
//! `result_contract.py`).
//! `harness/tests/test_build_sha.py` compiles this file and runs it against temporary checkouts.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Paths whose contents decide the binary. Also the paths cargo watches, so the `-dirty` suffix is
/// recomputed exactly when one of them changes.
const BUILD_INPUTS: [&str; 4] = ["crates", "benchmarks/runners", "Cargo.toml", "Cargo.lock"];

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The checkout's state: its HEAD and whether a build input differs from it.
struct Checkout {
    head: String,
    dirty: bool,
}

fn main() {
    println!("cargo:rerun-if-env-changed=POOT_BUILD_SHA");
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let exported = std::env::var("POOT_BUILD_SHA")
        .ok()
        .filter(|sha| !sha.trim().is_empty());

    let sha = resolve_sha(checkout(&manifest_dir), exported.as_deref())
        .unwrap_or_else(|reason| panic!("cannot determine the build sha: {reason}"));
    println!("cargo:rustc-env=POOT_BUILD_SHA={sha}");
}

fn is_sha(value: &str) -> bool {
    let hex = value.strip_suffix("-dirty").unwrap_or(value);
    hex.len() == 40 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The sha to embed, from the checkout when there is one and from the exported variable otherwise.
fn resolve_sha(checkout: Option<Checkout>, exported: Option<&str>) -> Result<String, String> {
    let exported = exported.map(str::trim);
    if let Some(value) = exported {
        if !is_sha(value) {
            return Err(format!(
                "POOT_BUILD_SHA={value:?} is not 40 lowercase hex digits (optionally followed by -dirty)"
            ));
        }
    }
    match (checkout, exported) {
        (Some(Checkout { head, dirty }), exported) => {
            let sha = if dirty {
                format!("{head}-dirty")
            } else {
                head.clone()
            };
            match exported {
                Some(value) if value != sha && value != head => Err(format!(
                    "POOT_BUILD_SHA={value} disagrees with the checkout ({sha}); unset it, git is the source in a checkout"
                )),
                _ => Ok(sha),
            }
        }
        (None, Some(value)) => Ok(value.to_string()),
        (None, None) => {
            Err("the source is not in a git checkout and POOT_BUILD_SHA is not set".to_string())
        }
    }
}

fn checkout(manifest_dir: &Path) -> Option<Checkout> {
    let head = git(manifest_dir, &["rev-parse", "HEAD"])?;
    let root = PathBuf::from(
        git(manifest_dir, &["rev-parse", "--show-toplevel"])
            .expect("git cannot name the toplevel of a git checkout"),
    );

    // HEAD moves by checkout (the HEAD file) or by commit (the branch ref it names).
    let mut moving = vec!["HEAD".to_string()];
    moving.extend(git(&root, &["symbolic-ref", "-q", "HEAD"]));
    for name in moving {
        if let Some(file) = git(&root, &["rev-parse", "--git-path", &name]) {
            println!("cargo:rerun-if-changed={}", root.join(file).display());
        }
    }
    for input in BUILD_INPUTS {
        println!("cargo:rerun-if-changed={}", root.join(input).display());
    }

    let mut status = vec!["status", "--porcelain", "--untracked-files=no", "--"];
    status.extend(BUILD_INPUTS);
    // Past this point it is a checkout: failing to read its state is an error, never a fall back to the variable.
    let dirty = !git(&root, &status)
        .expect("git status failed in a git checkout")
        .is_empty();
    Some(Checkout { head, dirty })
}

//! Bake the git commit into the binary so `--pcie-publication-receipt` can report it on pods
//! without a checkout. Runtime `POOT_GIT_COMMIT` still wins when set.
use std::process::Command;

fn git_stdout(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn main() {
    println!("cargo:rerun-if-env-changed=POOT_GIT_COMMIT");

    // `--git-path`/`--git-common-dir` resolve correctly whether `.git` is a directory or a worktree pointer
    // file. A hardcoded `../../.git/HEAD` cannot be stat'd through a pointer file, which made cargo treat this
    // build script as stale on every invocation in a linked worktree.
    if let Some(head) = git_stdout(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    // refs/heads lives in the common git dir (shared across worktrees), unlike HEAD. Watching it catches a commit
    // on the current branch (git updates a ref via lock-file-then-rename, touching the directory mtime). Weak on
    // packed refs (`git gc`), where updates land in `packed-refs`.
    if let Some(common_dir) = git_stdout(&["rev-parse", "--git-common-dir"]) {
        println!("cargo:rerun-if-changed={common_dir}/refs/heads");
    }

    if let Ok(sha) = std::env::var("POOT_GIT_COMMIT") {
        let trimmed = sha.trim();
        if !trimmed.is_empty() {
            println!("cargo:rustc-env=POOT_GIT_COMMIT={trimmed}");
            return;
        }
    }

    if let Some(sha) = git_stdout(&["rev-parse", "HEAD"]) {
        println!("cargo:rustc-env=POOT_GIT_COMMIT={sha}");
    }
}

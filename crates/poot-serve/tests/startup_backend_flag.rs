//! `--backend` through the production `poot-serve` binary: the flag reaches the engine plan and a
//! backend the build or the host cannot satisfy is refused at the process boundary, before the listener
//! binds. A parse that dropped the flag and resolved `Auto` would name another device in the error.

use std::process::Command;

#[path = "support/tiny_gguf.rs"]
mod tiny_gguf;

/// `CARGO_BIN_EXE_poot-serve`, read at runtime (see `startup_write_timeout.rs`).
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

fn run_with(args: &[&str]) -> (Option<i32>, String) {
    use std::io::Read;
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    let path = std::env::temp_dir().join(format!(
        "poot_serve_backend_flag_{}.gguf",
        std::process::id()
    ));
    tiny_gguf::write_tiny_deepseek2_gguf(&path, 64);
    let mut child = Command::new(poot_serve_bin())
        .arg(&path)
        .arg("127.0.0.1:0")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn poot-serve");
    // A refused flag exits at once; a server that started anyway never does, so bound the wait.
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the child") {
            break Some(status);
        }
        if Instant::now() > deadline {
            child
                .kill()
                .expect("kill the server that should not have started");
            child.wait().expect("reap");
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let _ = std::fs::remove_file(&path);
    let mut output = String::new();
    let _ = child
        .stdout
        .take()
        .expect("stdout")
        .read_to_string(&mut output);
    let mut stderr = String::new();
    let _ = child
        .stderr
        .take()
        .expect("stderr")
        .read_to_string(&mut stderr);
    output.push_str(&stderr);
    match status {
        Some(status) => (status.code(), output),
        None => panic!("the server kept running: the flag did not stop startup\n{output}"),
    }
}

/// `--backend rocm` is accepted by the parser and reaches `open_executor` as ROCm: this build has no
/// ROCm executor, so startup fails naming the ROCm device, not `Auto`'s or another backend's. Mutation:
/// pass `Auto` regardless of the flag; the error names `Auto` and the row goes red.
#[test]
fn a_named_backend_the_build_cannot_open_is_refused_by_name() {
    let (code, output) = run_with(&["--backend", "rocm"]);
    assert_ne!(code, Some(0), "{output}");
    assert!(output.contains("open the Rocm device"), "{output}");
    assert!(
        !output.contains("listening"),
        "the listener must not bind: {output}"
    );
}

/// A backend name the parser does not know is refused by the parser, naming the accepted ones.
#[test]
fn an_unknown_backend_name_is_rejected_by_the_parser() {
    let (code, output) = run_with(&["--backend", "cuda"]);
    assert_eq!(code, Some(2), "{output}");
    assert!(
        output.contains("expected wgpu, rocm, ptx, vulkan or auto"),
        "{output}"
    );
}

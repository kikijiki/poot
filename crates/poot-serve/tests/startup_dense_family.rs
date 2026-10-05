//! A dense family runs only on the driver, which the server does not use yet: startup refuses its
//! checkpoint with the Runner's typed `RunnerError::RegisteredFamily` and the server never binds.

/// `CARGO_BIN_EXE_poot-serve`, read at runtime rather than baked in with `env!` (see
/// `startup_write_timeout.rs`: a shared compile cache would serve another worktree's path).
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

/// One HF config per dense family key, nothing else in the directory: the refusal comes from the
/// config alone, before any weight or tokenizer is read. Mutation: drop `refuse_registered` from
/// `Runner::load_impl`; the Runner reads on, startup fails on something else and the row goes red.
#[test]
fn startup_refuses_a_dense_family_with_the_typed_registered_family_error() {
    use std::process::Command;

    let root = std::env::temp_dir().join(format!("poot-serve-dense-family-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (model_type, family) in [
        ("qwen2", "qwen2"),
        ("mistral", "llama"),
        ("gemma3_text", "gemma3"),
    ] {
        let dir = root.join(model_type);
        std::fs::create_dir_all(&dir).expect("create model fixture dir");
        std::fs::write(
            dir.join("config.json"),
            format!(r#"{{"model_type":"{model_type}"}}"#),
        )
        .expect("write model fixture config");
        // Port 0: a server that wrongly started would bind and wait; the refusal exits first.
        let output = Command::new(poot_serve_bin())
            .arg(&dir)
            .arg("127.0.0.1:0")
            .output()
            .expect("run poot-serve on a dense family");
        let process_output = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !output.status.success(),
            "{model_type}: startup must fail: {process_output}"
        );
        assert!(
            process_output.contains(&format!(
                "{family} is a registered family: load it with driver::ModelHandle"
            )),
            "{model_type}: the typed RegisteredFamily refusal is missing: {process_output}"
        );
        assert!(
            !process_output.contains("listening"),
            "{model_type}: the server must not bind: {process_output}"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

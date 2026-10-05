/// `CARGO_BIN_EXE_poot-serve`, read at runtime (not baked via `env!`): the compile-time macro's value is
/// embedded into this test binary's compiled object code at build time, and a shared compile cache (kache)
/// that reuses that object across worktrees by source-content hash would then serve whichever worktree's
/// path happened to compile it first (card 530's build.rs fix; card 543 review). `std::env::var` reads the
/// environment cargo sets fresh for every test-binary invocation, so it is correct regardless of which
/// worktree compiled the binary.
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

#[cfg(unix)]
#[test]
fn non_unicode_write_timeout_fails_before_model_loading() {
    use std::os::unix::ffi::OsStringExt;
    use std::process::Command;

    let output = Command::new(poot_serve_bin())
        .arg("/card316/model-must-not-load.gguf")
        .env(
            "POOT_WRITE_TIMEOUT_SECS",
            std::ffi::OsString::from_vec(vec![0xff]),
        )
        .output()
        .expect("run poot-serve with non-Unicode timeout configuration");

    assert!(!output.status.success(), "invalid startup must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("POOT_WRITE_TIMEOUT_SECS must be valid Unicode"),
        "typed configuration error missing from stderr: {stderr}"
    );
    let process_output = format!("{}{}", String::from_utf8_lossy(&output.stdout), stderr);
    assert!(
        !process_output.contains("loading model") && !process_output.contains("load gguf model"),
        "model-loading progress must not appear before configuration rejection: {process_output}"
    );
}

/// The response write timeout is validated before any model loader runs, whichever loader the
/// model path selects. One model path per loader arm of startup (cross-encoder, encoder, decoder
/// directory, GGUF file), each holding nothing a loader could load: with the timeout invalid the
/// process must fail on the timeout, and none of the loaders' progress lines or errors may appear.
/// Mutation observed red: the timeout is validated after the backend `match` instead of before it.
#[test]
fn invalid_write_timeout_fails_before_every_model_loader() {
    use std::process::Command;

    let root =
        std::env::temp_dir().join(format!("poot-serve-startup-order-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create model fixture root");

    let model_dir = |name: &str, config: &str| {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("create model fixture dir");
        std::fs::write(dir.join("config.json"), config).expect("write model fixture config");
        dir.to_string_lossy().into_owned()
    };
    let cases = [
        (
            "cross-encoder",
            model_dir(
                "cross-encoder",
                r#"{"architectures":["BertForSequenceClassification"]}"#,
            ),
        ),
        (
            "encoder",
            model_dir("encoder", r#"{"architectures":["BertModel"]}"#),
        ),
        (
            "decoder directory",
            model_dir("decoder", r#"{"model_type":"qwen2"}"#),
        ),
        (
            "gguf file",
            root.join("model.gguf").to_string_lossy().into_owned(),
        ),
    ];

    for (arm, model) in cases {
        let output = Command::new(poot_serve_bin())
            .arg(&model)
            .env("POOT_WRITE_TIMEOUT_SECS", "0")
            .output()
            .expect("run poot-serve with a zero write timeout");
        let process_output = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.status.success(), "{arm}: invalid startup must fail");
        assert!(
            process_output.contains("POOT_WRITE_TIMEOUT_SECS must be greater than zero"),
            "{arm}: the typed timeout error is missing: {process_output}"
        );
        for loader_trace in [
            "loading cross-encoder",
            "loading encoder",
            "loading model",
            "load cross-encoder model",
            "load encoder model",
            "load gguf model",
            "load model",
        ] {
            assert!(
                !process_output.contains(loader_trace),
                "{arm}: a model loader ran before the timeout was rejected ({loader_trace}): {process_output}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&root);
}

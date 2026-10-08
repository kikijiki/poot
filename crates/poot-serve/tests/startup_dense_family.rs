//! Startup loads a checkpoint through the family registry as a `ModelHandle`. A config naming no
//! registered family is the registry's typed refusal and the server never binds; a registered family
//! (a dense one the pre-driver engine refused) is resolved, so the load goes on to read its weights.

/// `CARGO_BIN_EXE_poot-serve`, read at runtime rather than baked in with `env!` (see
/// `startup_write_timeout.rs`: a shared compile cache would serve another worktree's path).
fn poot_serve_bin() -> String {
    std::env::var("CARGO_BIN_EXE_poot-serve")
        .expect("CARGO_BIN_EXE_poot-serve must be set by cargo for test binaries")
}

/// Run the production binary on a directory holding only `config.json` (`model_type`), on port 0: a
/// server that wrongly started would bind and wait, so the process must exit by itself. Returns whether
/// it succeeded and its combined output.
fn start_on_config_only(label: &str, model_type: &str) -> (bool, String) {
    use std::process::Command;

    let dir =
        std::env::temp_dir().join(format!("poot-serve-startup-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create model fixture dir");
    std::fs::write(
        dir.join("config.json"),
        format!(r#"{{"model_type":"{model_type}"}}"#),
    )
    .expect("write model fixture config");
    let output = Command::new(poot_serve_bin())
        .arg(&dir)
        .arg("127.0.0.1:0")
        .output()
        .expect("run poot-serve");
    let _ = std::fs::remove_dir_all(&dir);
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// A `model_type` no family registers is the registry's typed error, before any weight is read.
/// Mutation: resolve an unregistered key to a default family; the load goes on and fails on something
/// else, and the row goes red.
#[test]
fn startup_refuses_an_unregistered_family_with_the_registry_error() {
    let (ok, output) = start_on_config_only("unregistered", "not_a_registered_family");
    assert!(!ok, "startup must fail: {output}");
    assert!(
        output.contains("no family is registered for HF model_type \"not_a_registered_family\""),
        "the typed registry refusal is missing: {output}"
    );
    assert!(
        !output.contains("listening"),
        "the server must not bind: {output}"
    );
}

/// A dense family the pre-driver engine used to refuse is registered: the config resolves its family and the load
/// goes on to the checkpoint's weights (absent here), which is a load error and not a registry one.
/// Mutation: drop the family from the registry; the row reads the registry refusal and goes red.
#[test]
fn startup_resolves_a_dense_family_and_goes_on_to_its_weights() {
    for model_type in ["qwen2", "mistral", "gemma3_text"] {
        let (ok, output) = start_on_config_only(model_type, model_type);
        assert!(!ok, "{model_type}: no weights, startup must fail: {output}");
        assert!(
            !output.contains("no family is registered"),
            "{model_type}: the family must resolve: {output}"
        );
        assert!(
            !output.contains("registered family: load it with driver::ModelHandle"),
            "{model_type}: the old dense-family refusal is gone: {output}"
        );
        assert!(
            !output.contains("listening"),
            "{model_type}: the server must not bind: {output}"
        );
    }
}

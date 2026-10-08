//! The encoder backends over the production `http::handle` (POOT-739): a BERT embedder or
//! cross-encoder is loaded as a `ModelHandle`, and the routes that would run it on the driver return
//! the typed 503 until the serving loop restores them (POOT-589).

use super::no_engine::{json_body, roundtrip, status_line};
use super::*;

/// A BERT-class checkpoint directory: `family`'s registered fixture (or its named variant) written
/// as safetensors, with a word-level tokenizer over the fixture's 48-token vocabulary.
fn write_encoder_checkpoint(variant: Option<&str>) -> std::path::PathBuf {
    let registry = Registry::builtin().expect("shipped families register disjoint keys");
    let entry = registry
        .entries()
        .iter()
        .find(|entry| entry.family.as_str() == "bert")
        .expect("the bert family is registered");
    let fixture = match variant {
        None => (entry.fixture)(),
        Some(name) => entry
            .variants
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, build)| build())
            .expect("the bert family registers the variant"),
    };
    let vocab: String = (0..48)
        .map(|i| format!("\"w{i}\":{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let tokenizer = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{{"type":"WhitespaceSplit"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocab}}},"unk_token":"w0"}}}}"#
    );
    let dir = std::env::temp_dir().join(format!("poot_serve_bert_{}", unique_fixture_suffix()));
    std::fs::create_dir_all(&dir).expect("create the checkpoint directory");
    std::fs::write(dir.join("config.json"), fixture.config.to_string()).expect("write config");
    std::fs::write(dir.join("tokenizer.json"), tokenizer).expect("write tokenizer");
    poot_test_util::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
    dir
}

pub(crate) fn encoder_handle(variant: Option<&str>) -> Arc<ModelHandle> {
    let dir = write_encoder_checkpoint(variant);
    let handle = load_handle(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    handle
}

/// A route the driver has no serving path for answers the typed `NotServed` refusal: a 503 whose
/// body is that variant's message for that route, never a 200.
fn assert_not_served(response: &str, route: &'static str, label: &str) {
    assert_eq!(
        status_line(response),
        "HTTP/1.1 503 Service Unavailable",
        "{label}: {response}"
    );
    let error = &json_body(response)["error"];
    let refusal = GenerationError::NotServed { route };
    assert_eq!(error["code"], refusal.stream_status(), "{label}");
    assert_eq!(error["type"], "server_error", "{label}");
    assert_eq!(error["message"], refusal.to_string(), "{label}: {error}");
}

/// SC-006 (the serve row): an embedder answers both of its routes with the typed 503, and a
/// cross-encoder answers rerank with it, while the route it has no output for stays a 400.
#[test]
fn encoder_backends_answer_the_routes_they_cannot_run_yet_with_the_typed_503() {
    let embedder = Backend::Encoder(encoder_handle(None));
    let body = r#"{"input":"w1 w2"}"#;
    let rerank = r#"{"query":"w1","documents":["w2","w3"]}"#;
    assert_not_served(
        &roundtrip(&embedder, None, "POST", "/v1/embeddings", body, || {}),
        "/v1/embeddings",
        "encoder embeddings",
    );
    assert_not_served(
        &roundtrip(&embedder, None, "POST", "/v1/rerank", rerank, || {}),
        "/v1/rerank",
        "encoder rerank",
    );

    let cross = Backend::CrossEncoder(encoder_handle(Some("cross_encoder")));
    assert_not_served(
        &roundtrip(&cross, None, "POST", "/v1/rerank", rerank, || {}),
        "/v1/rerank",
        "cross-encoder rerank",
    );
    let response = roundtrip(&cross, None, "POST", "/v1/embeddings", body, || {});
    assert_eq!(
        status_line(&response),
        "HTTP/1.1 400 Bad Request",
        "a cross-encoder has no embedding output: {response}"
    );
}

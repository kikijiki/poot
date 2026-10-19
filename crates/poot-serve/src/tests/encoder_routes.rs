//! The encoder models over the production `http::handle` (POOT-589): a BERT embedder or cross-encoder is
//! loaded as a `ModelHandle`, and `/v1/embeddings` and `/v1/rerank` run it as `Encode` jobs on the
//! engine thread, scored by what the model's output is (hidden states pooled on the host, or a
//! classifier's score per pair). The engine is the recording fake, so these rows read the rows the
//! handlers send it and the answers they build from what it returns.

use super::lifecycle::{json_body, status_line};
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
    poot_quant::test_support::safetensors::write(&dir.join("model.safetensors"), &fixture.store);
    dir
}

pub(crate) fn encoder_handle(variant: Option<&str>) -> Arc<ModelHandle> {
    let dir = write_encoder_checkpoint(variant);
    let handle = load_handle(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    handle
}

/// The request `body` to `path` through the production `handle`, with `fake` on the engine thread behind
/// it: the response and the `Encode` rows the fake was asked to run.
fn serve_encode(
    model: Arc<ModelHandle>,
    path: &str,
    body: &str,
    answer: impl FnMut(&[crate::types::EncodeRow]) -> poot_llm::driver::Encoded + Send + 'static,
) -> (String, Vec<Vec<crate::types::EncodeRow>>) {
    let asked: Arc<Mutex<Vec<Vec<crate::types::EncodeRow>>>> = Arc::default();
    let hook_asked = Arc::clone(&asked);
    let mut answer = answer;
    let mut fake = RecordingEngine::new(8);
    fake.on_encode = Box::new(move |rows| {
        hook_asked.lock().unwrap().push(rows.to_vec());
        Ok(answer(rows))
    });
    let metrics = Arc::new(test_metrics("encoder-routes"));
    let (served, engine) = serve_with_fake(fake, model, &metrics);
    let response = super::lifecycle::roundtrip(&served, &metrics, "POST", path, body);
    engine.finish(served);
    let asked = asked.lock().unwrap().clone();
    (response, asked)
}

/// A cross-encoder scores each `(query, document)` pair jointly in one `Encode` step: the rows are
/// the pairs, their second sentence carries segment id 1, and `relevance_score` is the raw logit the
/// model returned for the pair, ordered descending. The fake's score for a pair is its token count, so
/// the longer pair ranks first and the order cannot come from the input order.
#[test]
fn a_cross_encoder_scores_each_pair_jointly_and_ranks_by_the_raw_logit() {
    let model = encoder_handle(Some("cross_encoder"));
    let (response, asked) = serve_encode(
        model,
        "/v1/rerank",
        r#"{"query":"w1","documents":["w2","w3 w4 w5","w6 w7"]}"#,
        |rows| poot_llm::driver::Encoded {
            width: 1,
            rows: rows.iter().map(|row| vec![row.ids.len() as f32]).collect(),
        },
    );
    assert_eq!(status_line(&response), "HTTP/1.1 200 OK", "{response}");
    assert_eq!(asked.len(), 1, "one Encode step for all the pairs");
    let rows = &asked[0];
    assert_eq!(rows.len(), 3, "one row per pair");
    for row in rows {
        assert!(
            row.types.contains(&1) && row.types.contains(&0),
            "a pair row carries both segments: {row:?}"
        );
    }
    let results = json_body(&response)["results"].as_array().unwrap().clone();
    let order: Vec<u64> = results
        .iter()
        .map(|r| r["index"].as_u64().unwrap())
        .collect();
    assert_eq!(order, vec![1, 2, 0], "the longest pair first: {results:?}");
    for (result, index) in results.iter().zip(&order) {
        let score = rows[*index as usize].ids.len() as f64;
        assert_eq!(
            result["relevance_score"].as_f64().unwrap(),
            score,
            "{result}"
        );
    }
}

/// A BERT embedder answers embeddings through the engine: the hidden states it returns, mean-pooled and
/// unit length, one vector per input.
#[test]
fn an_encoder_embedder_answers_embeddings_through_the_engine() {
    let model = encoder_handle(None);
    let (response, asked) = serve_encode(
        model,
        "/v1/embeddings",
        r#"{"input":["w1 w2","w3"]}"#,
        |rows| poot_llm::driver::Encoded {
            width: 2,
            rows: rows
                .iter()
                .map(|row| row.ids.iter().flat_map(|_| [3.0, 4.0]).collect())
                .collect(),
        },
    );
    assert_eq!(status_line(&response), "HTTP/1.1 200 OK", "{response}");
    assert_eq!(asked[0].len(), 2, "both inputs in one step");
    let json = json_body(&response);
    for entry in json["data"].as_array().unwrap() {
        let vector: Vec<f64> = entry["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert!(
            (vector[0] - 0.6).abs() < 1e-6 && (vector[1] - 0.8).abs() < 1e-6,
            "{vector:?}"
        );
    }
}

/// POOT-1204 SC-002: a request of more texts than the fitted rows, or a text longer than the fitted
/// length, is a 400 that names the cap and its flag, and the engine is never asked. A request inside
/// the caps runs. Mutation: remove `check_encode_fit` from `handlers::encode`; the request reaches the
/// fake, answers 200, and the row goes red.
#[test]
fn an_embeddings_request_over_the_fitted_shapes_is_a_400_naming_the_cap() {
    let model = encoder_handle(None);
    let caps = crate::tests::test_encoder_caps(2, 8);
    let metrics = Arc::new(test_metrics("encoder-caps"));
    let (served, engine) = crate::tests::serve_with_fake_caps(
        RecordingEngine::new(2),
        Arc::clone(&model),
        caps,
        &metrics,
    );
    let too_many = super::lifecycle::roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":["w1","w2","w3"]}"#,
    );
    // Eight tokens fit; nine plus the encoder's special tokens do not.
    let long = ["w1"; 9].join(" ");
    let too_long = super::lifecycle::roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        &format!(r#"{{"input":"{long}"}}"#),
    );
    let fits = super::lifecycle::roundtrip(
        &served,
        &metrics,
        "POST",
        "/v1/embeddings",
        r#"{"input":["w1","w2"]}"#,
    );
    let asked = engine.finish(served).events();
    assert_eq!(
        status_line(&too_many),
        "HTTP/1.1 400 Bad Request",
        "{too_many}"
    );
    assert!(
        too_many.contains("3 texts exceed the 2 rows") && too_many.contains("--rows"),
        "{too_many}"
    );
    assert_eq!(
        status_line(&too_long),
        "HTTP/1.1 400 Bad Request",
        "{too_long}"
    );
    assert!(
        too_long.contains("over the 8") && too_long.contains("--max-seq"),
        "{too_long}"
    );
    assert_eq!(status_line(&fits), "HTTP/1.1 200 OK", "{fits}");
    assert_eq!(
        asked,
        vec![super::recording_engine::Event::Encode { rows: 2 }],
        "only the request inside the caps reached the engine"
    );
}

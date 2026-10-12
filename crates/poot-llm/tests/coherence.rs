//! Coherence checks through the public API for the encoders (embeddings and reranking on the driver's
//! `Encode` step) and the text services. Every real-checkpoint test needs POOT_MODELS_DIR and skips
//! (passes) when it is unset or the model is absent. The decoder families' coherence rows run through
//! the driver (`driver/device_tests.rs`).

use poot_llm::driver::{BackendChoice, open_executor};
use poot_llm::encoders::{CrossEncoder, Pooling, SentenceEncoder};

fn sentence_encoder(dir: &std::path::Path) -> SentenceEncoder {
    SentenceEncoder::load(dir, open_executor(BackendChoice::Wgpu).expect("open wgpu"))
        .expect("load the encoder")
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The all-MiniLM-L6-v2 BERT encoder produces sentence embeddings that separate related from unrelated
/// sentences (higher quality than decoder pooling). wgpu; ignored.
#[test]
#[ignore = "loads all-minilm-l6-v2 on wgpu; BERT encoder embedding check. run with --ignored"]
fn minilm_encoder_separates_related() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("all-minilm-l6-v2"))
    else {
        return;
    };
    let mut enc = sentence_encoder(&dir);
    assert_eq!(enc.pooling(), Pooling::Mean, "MiniLM mean-pools");
    let vectors = enc
        .embed(&[
            "a small domestic cat",
            "a young playful kitten",
            "a fast red sports car",
        ])
        .expect("embed");
    let (rel, unrel) = (
        cosine(&vectors[0], &vectors[1]),
        cosine(&vectors[0], &vectors[2]),
    );
    eprintln!("minilm cosine(cat,kitten)={rel:.3} cosine(cat,car)={unrel:.3}");
    assert_eq!(vectors[0].len(), 384, "embedding dim = hidden (384)");
    assert!(
        (cosine(&vectors[0], &vectors[0]) - 1.0).abs() < 1e-4,
        "self-cosine ~1 (L2-normalized)"
    );
    // a real encoder separates much more cleanly than decoder pooling.
    assert!(
        rel > unrel + 0.1,
        "related must clearly beat unrelated: {rel:.3} vs {unrel:.3}"
    );
}

/// The encoder's bi-encoder `rerank` orders documents by relevance: cat-related docs rank above the
/// unrelated one, top-1 is the kitten. wgpu; ignored.
#[test]
#[ignore = "loads all-minilm-l6-v2 on wgpu; encoder rerank check. run with --ignored"]
fn minilm_encoder_reranks() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("all-minilm-l6-v2"))
    else {
        return;
    };
    let mut enc = sentence_encoder(&dir);
    let docs = [
        "a fast red sports car",  // 0: unrelated
        "a young playful kitten", // 1: most related
        "the weather forecast",   // 2: unrelated
        "a small domestic cat",   // 3: related
    ];
    let ranked = enc.rerank("a cute kitten", &docs).expect("rerank");
    eprintln!("minilm rerank: {ranked:?}");
    assert_eq!(ranked.len(), docs.len());
    // The order the CPU encoder recorded at the base commit (scores 0.744, 0.602, 0.228, 0.101).
    let order: Vec<usize> = ranked.iter().map(|r| r.0).collect();
    assert_eq!(order, [1, 3, 0, 2]);
    assert!(ranked[0].1 >= ranked[1].1, "scores are sorted descending");
}

/// BGE-small (`pooling_mode_cls_token`) loads with CLS pooling and separates related from unrelated
/// sentences; poot reads the `1_Pooling` config and pools the `[CLS]` token (not mean) for BGE/GTE-class
/// models. wgpu; ignored.
#[test]
#[ignore = "loads bge-small-en on wgpu; CLS-pooled embedding check. run with --ignored"]
fn bge_cls_pooling_separates_related() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("bge-small-en")) else {
        return;
    };
    let mut enc = sentence_encoder(&dir);
    assert_eq!(enc.pooling(), Pooling::Cls, "BGE pools the CLS token");
    let vectors = enc
        .embed(&[
            "a small domestic cat",
            "a young playful kitten",
            "a fast red sports car",
        ])
        .expect("embed");
    assert_eq!(vectors[0].len(), 384);
    let (rel, unrel) = (
        cosine(&vectors[0], &vectors[1]),
        cosine(&vectors[0], &vectors[2]),
    );
    eprintln!("bge cosine(cat,kitten)={rel:.3} cosine(cat,car)={unrel:.3}");
    assert!(
        (cosine(&vectors[0], &vectors[0]) - 1.0).abs() < 1e-4,
        "self-cosine ~1 (L2-normalized)"
    );
    assert!(
        rel > unrel + 0.1,
        "related must clearly beat unrelated: {rel:.3} vs {unrel:.3}"
    );
}

/// `CrossEncoder` scores a query+doc pair jointly through `BertForSequenceClassification`.
/// Model-card example: the Berlin-population passage scores far above the museum passage for "How many
/// people live in Berlin?". wgpu; ignored. The scores equal the CPU encoder's at the base commit
/// (relevant 8.846, irrelevant -4.320).
#[test]
#[ignore = "loads ms-marco-minilm-l6 cross-encoder on wgpu. run with --ignored"]
fn cross_encoder_reranks() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("ms-marco-minilm-l6"))
    else {
        return;
    };
    let mut ce = CrossEncoder::load(&dir, open_executor(BackendChoice::Wgpu).expect("open wgpu"))
        .expect("load the cross-encoder");
    let query = "How many people live in Berlin?";
    let relevant = "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 \
                    square kilometers.";
    let irrelevant = "Berlin is well known for its museums.";
    let scores = ce.score(query, &[relevant, irrelevant]).expect("score");
    let (s_rel, s_irrel) = (scores[0], scores[1]);
    eprintln!("cross-encoder: relevant={s_rel:.3} irrelevant={s_irrel:.3}");
    assert!(
        (s_rel - 8.846).abs() < 0.05 && (s_irrel + 4.320).abs() < 0.05,
        "the recorded scores: {s_rel:.3} and {s_irrel:.3}"
    );
    // rerank returns them sorted, most-relevant first.
    let ranked = ce.rerank(query, &[irrelevant, relevant]).expect("rerank");
    assert_eq!(ranked[0].0, 1, "the relevant doc (index 1) ranks first");
}

/// Cheap (header-only, no GPU, no weight load) precondition for
/// the gpt-oss-20b driver coherence row (`driver/device_tests.rs`): the real `ggml-org/gpt-oss-20b-GGUF` file ships the
/// full OpenAI "harmony" jinja chat template in `tokenizer.chat_template`, and poot's
/// `render_jinja_value` must render it rather than silently falling back to the hardcoded
/// `ChatFormat`.
///
/// gpt-oss is an instruct model whose prompt format is not ChatML: a raw or ChatML-shaped prompt
/// degenerates on poot and llama.cpp alike, so a silent fallback would be misread as an incoherence
/// verdict. The harmony template is heavy jinja (nested macros, `namespace`, `strftime_now`, python
/// `dict`/`str` methods), the class `render_jinja_core`'s `pycompat` + `strftime_now` wiring exists
/// for.
///
/// Uses `GgufIndex::parse` over a mapping (only the header pages are touched), not the checkpoint's 12.1 GB. Run it
/// before renting anything.
#[test]
#[ignore = "reads the GGUF HEADER ONLY of a real ggml-org/gpt-oss-20b-GGUF MXFP4 file (no tensor data, no \
            GPU); run with --ignored --nocapture"]
fn gptoss_20b_gguf_harmony_chat_template_renders() {
    // The GGUF is e.g. `hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gpt-oss-20b-gguf/gpt-oss-20b-MXFP4.gguf"
    )) else {
        return;
    };
    let g = poot_load::gguf::GgufIndex::parse(
        &poot_quant::StoredBytes::map(std::fs::File::open(&gguf_path).unwrap()).unwrap(),
    )
    .expect("read gpt-oss-20b gguf header");
    assert_eq!(
        g.architecture(),
        Some("gpt-oss"),
        "this file must be a real gpt-oss GGUF"
    );
    let tmpl = g
        .get("tokenizer.chat_template")
        .and_then(|v| v.as_str())
        .expect("a real gpt-oss GGUF ships tokenizer.chat_template (the harmony template)");
    // The GGUF's bos/eos strings, looked up in the token table at the ids the metadata names (the
    // crosswalk the GGUF text codec does), so this test needs no weights.
    let toks = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .expect("token table");
    let tok_at = |key: &str| -> Option<String> {
        let id = g.get(key)?.as_u64()? as usize;
        Some(toks.get(id)?.as_str()?.to_string())
    };
    let bos = tok_at("tokenizer.ggml.bos_token_id");
    let eos = tok_at("tokenizer.ggml.eos_token_id");
    let rendered = poot_text::render_jinja_value(
        tmpl,
        &serde_json::json!([{"role": "user", "content": "What is the capital of France?"}]),
        bos.as_deref(),
        eos.as_deref(),
        None,
    )
    .expect(
        "the harmony chat template must render through poot's own minijinja path - a failure here is \
         exactly the silent ChatFormat fallback that would make a coherence run produce garbage",
    );
    eprintln!("gpt-oss-20b harmony rendered prompt:\n{rendered}");
    // Harmony's own structural markers (see the openai/gpt-oss model card): every turn is
    // `<|start|>{role}<|message|>{content}<|end|>`, and the generation prompt opens an assistant turn.
    for marker in ["<|start|>", "<|message|>", "<|end|>", "assistant"] {
        assert!(
            rendered.contains(marker),
            "rendered harmony prompt is missing {marker:?} - poot probably fell back to ChatFormat: \
             {rendered:?}"
        );
    }
    assert!(
        rendered.contains("What is the capital of France?"),
        "the user turn must survive rendering: {rendered:?}"
    );
    assert!(
        rendered.trim_end().ends_with("<|start|>assistant"),
        "the generation prompt must end with an OPEN assistant turn (harmony's add_generation_prompt \
         shape) so the model continues rather than starting a fresh user turn: {rendered:?}"
    );
}

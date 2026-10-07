//! Coherence checks through the public API for the families still on the Runner (MLA and hybrid) and
//! the encoders. Every real-checkpoint test needs POOT_MODELS_DIR and skips (passes) when it is unset or
//! the model is absent. The dense and MoE families' coherence rows run through the driver
//! (`driver/device_tests.rs`).

/// The all-MiniLM-L6-v2 BERT encoder produces sentence embeddings that separate related from unrelated
/// sentences (higher quality than decoder pooling). CPU; ignored.
#[test]
#[ignore = "loads all-minilm-l6-v2; BERT encoder embedding check. run with --ignored"]
fn minilm_encoder_separates_related() {
    use poot_llm::encoder::EncoderRunner;
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("all-minilm-l6-v2"))
    else {
        return;
    };
    let enc = EncoderRunner::load(&dir).expect("load encoder");
    let cosine = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let cat = enc.embed("a small domestic cat").expect("embed");
    let kitten = enc.embed("a young playful kitten").expect("embed");
    let car = enc.embed("a fast red sports car").expect("embed");
    let (rel, unrel) = (cosine(&cat, &kitten), cosine(&cat, &car));
    eprintln!("minilm cosine(cat,kitten)={rel:.3} cosine(cat,car)={unrel:.3}");
    assert_eq!(cat.len(), enc.dim(), "embedding dim = hidden (384)");
    assert_eq!(enc.dim(), 384);
    assert!(
        (cosine(&cat, &cat) - 1.0).abs() < 1e-4,
        "self-cosine ~1 (L2-normalized)"
    );
    // a real encoder separates much more cleanly than decoder pooling.
    assert!(
        rel > unrel + 0.1,
        "related must clearly beat unrelated: {rel:.3} vs {unrel:.3}"
    );
}

/// The encoder's bi-encoder `rerank` orders documents by relevance: cat-related docs rank above the
/// unrelated one, top-1 is the kitten. CPU; ignored.
#[test]
#[ignore = "loads all-minilm-l6-v2; encoder rerank check. run with --ignored"]
fn minilm_encoder_reranks() {
    use poot_llm::encoder::EncoderRunner;
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("all-minilm-l6-v2"))
    else {
        return;
    };
    let enc = EncoderRunner::load(&dir).expect("load encoder");
    let docs = [
        "a fast red sports car",  // 0: unrelated
        "a young playful kitten", // 1: most related
        "the weather forecast",   // 2: unrelated
        "a small domestic cat",   // 3: related
    ];
    let ranked = enc.rerank("a cute kitten", &docs).expect("rerank");
    eprintln!("minilm rerank: {ranked:?}");
    assert_eq!(ranked.len(), docs.len());
    // sorted most-relevant first; the unrelated car/weather sink below the cat/kitten docs.
    assert!(
        ranked[0].0 == 1 || ranked[0].0 == 3,
        "a cat/kitten doc ranks first"
    );
    assert!(ranked[0].1 >= ranked[1].1, "scores are sorted descending");
    let last = ranked.last().unwrap().0;
    assert!(last == 0 || last == 2, "an unrelated doc ranks last");
}

/// BGE-small (`pooling_mode_cls_token`) loads with CLS pooling and separates related from unrelated
/// sentences; poot reads the `1_Pooling` config and pools the `[CLS]` token (not mean) for BGE/GTE-class
/// models. CPU; ignored.
#[test]
#[ignore = "loads bge-small-en; CLS-pooled embedding check. run with --ignored"]
fn bge_cls_pooling_separates_related() {
    use poot_llm::encoder::{EncoderPooling, EncoderRunner};
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("bge-small-en")) else {
        return;
    };
    let enc = EncoderRunner::load(&dir).expect("load bge");
    assert_eq!(
        enc.pooling(),
        EncoderPooling::Cls,
        "BGE pools the CLS token"
    );
    assert_eq!(enc.dim(), 384);
    let cosine = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let cat = enc.embed("a small domestic cat").expect("embed");
    let kitten = enc.embed("a young playful kitten").expect("embed");
    let car = enc.embed("a fast red sports car").expect("embed");
    let (rel, unrel) = (cosine(&cat, &kitten), cosine(&cat, &car));
    eprintln!("bge cosine(cat,kitten)={rel:.3} cosine(cat,car)={unrel:.3}");
    assert!(
        (cosine(&cat, &cat) - 1.0).abs() < 1e-4,
        "self-cosine ~1 (L2-normalized)"
    );
    assert!(
        rel > unrel + 0.1,
        "related must clearly beat unrelated: {rel:.3} vs {unrel:.3}"
    );
}

/// `bert_kind` recognizes the BERT-class MiniLM checkpoint as an encoder and a Qwen decoder checkpoint as
/// not. CPU; ignored (needs the model dirs).
#[test]
#[ignore = "reads config.json of all-minilm-l6-v2 + qwen2.5-0.5b. run with --ignored"]
fn is_encoder_dir_detects_bert() {
    use poot_llm::encoder::bert_kind;
    let Some(bert) = poot_test_util::model_path(poot_test_util::checkpoint!("all-minilm-l6-v2"))
    else {
        return;
    };
    let Some(dec) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b")) else {
        return;
    };
    assert!(
        bert_kind(&bert).is_some(),
        "MiniLM (model_type=bert) is an encoder"
    );
    assert!(
        bert_kind(&dec).is_none(),
        "a Qwen decoder is not an encoder"
    );
}

/// `CrossEncoderRunner` scores a query+doc pair jointly through `BertForSequenceClassification`.
/// Model-card example: the Berlin-population passage scores far above the museum passage for "How many
/// people live in Berlin?". CPU; ignored. Also checks `bert_kind` detects the cross-encoder vs the
/// embedding encoder.
#[test]
#[ignore = "loads ms-marco-minilm-l6 cross-encoder. run with --ignored"]
fn cross_encoder_reranks() {
    use poot_llm::encoder::{BertKind, CrossEncoderRunner, bert_kind};
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("ms-marco-minilm-l6"))
    else {
        return;
    };
    let Some(embedding_dir) =
        poot_test_util::model_path(poot_test_util::checkpoint!("all-minilm-l6-v2"))
    else {
        return;
    };
    assert_eq!(
        bert_kind(&dir),
        Some(BertKind::CrossEncoder),
        "detected as cross-encoder"
    );
    assert_eq!(
        bert_kind(&embedding_dir),
        Some(BertKind::Embedding),
        "the embedding encoder is NOT a cross-encoder"
    );
    let ce = CrossEncoderRunner::load(&dir).expect("load cross-encoder");
    let query = "How many people live in Berlin?";
    let relevant = "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 \
                    square kilometers.";
    let irrelevant = "Berlin is well known for its museums.";
    let (s_rel, s_irrel) = (
        ce.score(query, relevant).expect("score"),
        ce.score(query, irrelevant).expect("score"),
    );
    eprintln!("cross-encoder: relevant={s_rel:.3} irrelevant={s_irrel:.3}");
    // the cross-encoder logit is a strong, well-separated relevance signal (the ms-marco reference puts the
    // population passage around +8 and the museum passage well below it).
    assert!(
        s_rel > s_irrel + 4.0,
        "relevant must clearly beat irrelevant: {s_rel:.3} vs {s_irrel:.3}"
    );
    assert!(
        s_rel > 0.0,
        "the relevant passage scores positive: {s_rel:.3}"
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
/// Uses `GgufIndex::open` (reads at most the first 256 MB), not the checkpoint's 12.1 GB. Run it
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
    let g = poot_load::gguf::GgufIndex::open(&gguf_path).expect("read gpt-oss-20b gguf header");
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
    // crosswalk `Runner::load_gguf` does in runner.rs), so this test needs no weights.
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
    let rendered = poot_llm::render_jinja_value(
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

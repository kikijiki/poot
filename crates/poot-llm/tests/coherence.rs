//! Coherence checks through the public API for the families still on the Runner (MoE and hybrid) and
//! the encoders. Every real-checkpoint test needs POOT_MODELS_DIR and skips (passes) when it is unset or
//! the model is absent. The dense families' coherence rows run through the driver
//! (`driver/device_tests.rs`).

use poot_llm::Runner;

#[test]
#[ignore = "loads a granitemoe-arch GGUF (granite-3.0-1b-a400m, ~1.4GB Q8_0); run with --ignored"]
fn gguf_granitemoe_arch_greedy_is_coherent() {
    // load_gguf handles granitemoe (top-k expert mixture): GGUF carries the experts as separate 3D
    // ffn_{gate,up,down}_exps + an ffn_gate_inp router; the loader fuses gate||up into the moe op's
    // input_linear and transposes each expert. The four Granite scalar multipliers come from GGUF
    // metadata.
    let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "granite-moe-1b-gguf/granite-3.0-1b-a400m-instruct-Q8_0.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&path).expect("load granitemoe-arch gguf");
    let toks = runner
        .generate("The capital of France is", 12, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let text = runner.decode(&toks).unwrap();
    eprintln!("gguf granitemoe-arch (granite-3.0-1b-a400m) greedy: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "granitemoe gguf continuation was: {text:?}"
    );
}

#[test]
#[ignore = "loads granite-moe-1b + the efficient cached MoE decode on the Arc; run with --ignored"]
fn granite_moe_decodes_on_gpu_cached() {
    // Fast O(n) cached MoE decode on the GPU: `generate_kv_gpu_cached` traces the granite decode graph
    // (MoE branch), which lowers fully on-device (the gate is the `Ge`-based `top_k_gate` primitive
    // composition, not the host-fallback `TopKGate` op). Coherence is the end-to-end check.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granite-moe-1b"))
    else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load granite-moe-1b");
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let toks = runner
        .generate_kv_gpu_cached("The capital of France is", 12, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let text = runner.decode(&toks).unwrap();
    eprintln!("granite-moe cached-decode on GPU: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "granite-moe cached-decode continuation was: {text:?}"
    );
}

#[test]
#[ignore = "loads granite-moe-1b + runs cached GPU decode AND CPU eager (slow); run with --ignored"]
fn granite_moe_cached_gpu_matches_cpu() {
    // Executor equivalence for the MoE decode (stronger than the coherence check): the cached GPU decode
    // (on-device gate + experts) must produce the same greedy tokens as the CPU eager reference, i.e. the
    // on-device stable gate is numerically faithful to the CPU `top_k_gate`.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granite-moe-1b"))
    else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load granite-moe-1b");
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 8;
    let gpu_toks = runner
        .generate_kv_gpu_cached(prompt, max_new, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .unwrap();
    eprintln!(
        "granite-moe gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "cached-GPU MoE decode tokens must match the CPU eager reference"
    );
}

#[test]
#[ignore = "loads granite-moe-1b f32; run with --ignored"]
fn granite_moe_greedy_is_coherent() {
    // GraniteMoE tracer (router + top-k expert mixture + the four Granite scalar multipliers) produces a
    // coherent English greedy continuation on the CPU eager path.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granite-moe-1b"))
    else {
        return;
    };
    let runner = Runner::load(&dir).expect("load granite-moe-1b");
    let prompt = "The capital of France is";
    let toks = runner
        .generate(prompt, 12, |_| std::ops::ControlFlow::Continue(()))
        .unwrap();
    let text = runner.decode(&toks).unwrap();
    eprintln!("granite-moe greedy: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "granite-moe continuation was: {text:?}"
    );
}

#[test]
fn qwen3_moe_tiny_checkpoint_generates_without_crashing() {
    // `Runner::load`'s qwen3_moe safetensors path through the arch-dispatched CPU re-prefill `generate()`.
    // `yujiepan/qwen3-moe-tiny-random` is a real qwen3_moe checkpoint but randomly initialized, so there is
    // no expected-English oracle; the bar is "loads, decodes N tokens, stays numerically sane, no crash".
    // This proves `Runner::load` + `generate` reach the tracer end to end through the public API
    // (`runner::qwen3_moe_load_tests` in src/runner.rs checks `bind`+`eval` internally). Small (~20 MB)
    // fixture; not `#[ignore]`; skips if the checkpoint directory is absent.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny"))
    else {
        return;
    };
    let runner = Runner::load(&dir).expect("load qwen3-moe-tiny");
    let prompt = "The capital of France is";
    let toks = runner
        .generate(prompt, 8, |_| std::ops::ControlFlow::Continue(()))
        .expect("qwen3-moe generate should not error");
    let prompt_len = runner.encode(prompt).unwrap().len();
    assert!(
        toks.len() > prompt_len,
        "expected at least one generated token beyond the {prompt_len}-token prompt, got {} total",
        toks.len()
    );
    for &t in &toks {
        assert!(
            (t as usize) < runner.config().vocab,
            "generated token {t} is out of the {}-entry vocab range",
            runner.config().vocab
        );
    }
    // Decode must not panic on the (nonsense, randomly initialized) tokens greedy decode picked; proves
    // the round trip end to end, not just that argmax stayed in range.
    let text = runner.decode(&toks).expect("decode generated tokens");
    eprintln!("qwen3-moe-tiny (random weights) greedy continuation: {text:?}");
}

#[test]
fn qwen3_moe_gguf_tiny_checkpoint_generates_without_crashing() {
    // GGUF analog of `qwen3_moe_tiny_checkpoint_generates_without_crashing`, through
    // `Runner::load_gguf`. Uses a small self-authored fully-sparse `qwen3_moe` checkpoint converted with
    // `convert_hf_to_gguf.py` (llama.cpp's GGUF tooling only supports qwen3moe with every layer routed, so
    // `yujiepan/qwen3-moe-tiny-random`'s mixed layout cannot be used; see
    // `runner::qwen3_moe_load_tests::qwen3_moe_gguf_matches_safetensors_on_self_authored_sparse_checkpoint`
    // in `src/runner.rs`). Randomly initialized, so the bar is "loads, decodes N tokens, stays numerically
    // sane, no crash".
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "qwen3-moe-tiny-sparse-gguf/model.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load qwen3-moe-tiny-sparse gguf");
    let prompt = "The capital of France is";
    let toks = runner
        .generate(prompt, 8, |_| std::ops::ControlFlow::Continue(()))
        .expect("qwen3-moe gguf generate should not error");
    let prompt_len = runner.encode(prompt).unwrap().len();
    assert!(
        toks.len() > prompt_len,
        "expected at least one generated token beyond the {prompt_len}-token prompt, got {} total",
        toks.len()
    );
    for &t in &toks {
        assert!(
            (t as usize) < runner.config().vocab,
            "generated token {t} is out of the {}-entry vocab range",
            runner.config().vocab
        );
    }
    let text = runner.decode(&toks).expect("decode generated tokens");
    eprintln!("qwen3-moe-tiny-sparse gguf (random weights) greedy continuation: {text:?}");
}

/// Real-wgpu end-to-end generation for qwen3-MoE. Compares `generate_kv_gpu_prefilled` (one batched
/// `bind_prefill_kv` prefill through `prefill_kv_graph`, covering GPU-resident MoE prefill, then
/// carried-KV single-token decode through `decode_masked_graph`) against `generate` (CPU, full re-prefill
/// every step via the plain prefill tracer): two different code paths that must agree exactly on
/// greedy token ids. Not `#[ignore]`: tiny fixture.
#[test]
fn qwen3_moe_gpu_prefill_then_decode_matches_cpu_safetensors() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny"))
    else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load(&dir).expect("load qwen3-moe-tiny");
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 6;
    let gpu_toks = runner
        .generate_kv_gpu_prefilled(prompt, max_new, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("qwen3-moe GPU prefill+decode generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("qwen3-moe CPU re-prefill generate");
    eprintln!(
        "qwen3-moe (safetensors) gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "qwen3-moe GPU prefill+decode tokens must match the CPU re-prefill reference"
    );
}

/// GGUF analog of [`qwen3_moe_gpu_prefill_then_decode_matches_cpu_safetensors`]: the self-authored,
/// fully-sparse checkpoint through `Runner::load_gguf`.
#[test]
fn qwen3_moe_gpu_prefill_then_decode_matches_cpu_gguf_sparse() {
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "qwen3-moe-tiny-sparse-gguf/model.gguf"
    )) else {
        return;
    };
    let device = match poot_gpu::device::WgpuDevice::new() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping");
            return;
        }
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load qwen3-moe-tiny-sparse gguf");
    let mut engine = poot_executor::Engine::new(device);
    let exe = runner.load_on(&mut engine).unwrap();
    let prompt = "The capital of France is";
    let max_new = 6;
    let gpu_toks = runner
        .generate_kv_gpu_prefilled(prompt, max_new, &mut engine, exe, |_| {
            std::ops::ControlFlow::Continue(())
        })
        .expect("qwen3-moe gguf GPU prefill+decode generate");
    let cpu_toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("qwen3-moe gguf CPU re-prefill generate");
    eprintln!(
        "qwen3-moe (gguf) gpu={:?} cpu={:?}",
        runner.decode(&gpu_toks).unwrap(),
        runner.decode(&cpu_toks).unwrap()
    );
    assert_eq!(
        gpu_toks, cpu_toks,
        "qwen3-moe gguf GPU prefill+decode tokens must match the CPU re-prefill reference"
    );
}

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

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of olmoe-tiny under POOT_MODELS_DIR (converted this round via \
            convert_hf_to_gguf.py, not hand-built) through the full production Runner::load_gguf -> \
            generate -> decode pipeline; run with --ignored --release"]
fn olmoe_tiny_gguf_generates_finite_output() {
    // GGUF follow-on for OlmoE, mirroring `mixtral_tiny_gguf_generates_finite_output`. Unlike Mixtral,
    // OlmoE has its own GGUF arch string ("olmoe": llama.cpp `conversion/olmo.py`'s `OlmoeModel`,
    // `gguf.MODEL_ARCH.OLMOE`), confirmed against a `convert_hf_to_gguf.py --outtype f32` conversion of
    // `olmoe-tiny` (under POOT_MODELS_DIR); see `gguf.rs`'s `gguf_weights` and `runner.rs`'s `olmoe` detection arm. This
    // test runs the pipeline (GGUF -> Runner::load_gguf -> olmoe detection -> trace_olmoe_prefill ->
    // greedy decode -> detokenize) and checks finite, non-degenerate output; untrained weights give
    // gibberish, so coherent English is not asserted. The numerically precise check (GGUF weight
    // crosswalk vs the safetensors loader's `fuse_qwen3_moe_experts` on identical weights, via explicit
    // tokens through `bind`+`eval` to bypass the loaders' BOS-handling asymmetry) is `runner.rs`'s
    // `olmoe_load_tests::olmoe_tiny_gguf_matches_safetensors`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "olmoe-tiny/olmoe-tiny-f32.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load olmoe-tiny gguf");
    let prompt = "The capital of France is";
    let max_new = 12;
    let toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-tiny gguf generate");
    let text = runner.decode(&toks).unwrap();
    eprintln!("olmoe-tiny gguf greedy: {text:?}");
    assert!(!text.is_empty(), "decoded text must be non-empty");
    let prompt_len = runner.encode(prompt).unwrap().len();
    let generated = &toks[prompt_len..];
    assert_eq!(
        generated.len(),
        max_new,
        "must have generated max_new tokens (no early EOS on an untrained checkpoint)"
    );
    assert!(
        generated.iter().any(|&t| t != generated[0]),
        "generated tokens must not all be identical (a degenerate-output signature): {generated:?}"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of deepseek2-tiny under POOT_MODELS_DIR (converted this round via \
            convert_hf_to_gguf.py, not hand-built) through the full production Runner::load_gguf -> \
            generate -> decode pipeline; run with --ignored --release"]
fn deepseek2_tiny_gguf_generates_finite_output() {
    // GGUF follow-on for DeepSeek-V2, mirroring `mixtral_tiny_gguf_generates_finite_output`/
    // `olmoe_tiny_gguf_generates_finite_output`. DeepSeek-V2's headline mechanism is Multi-head Latent
    // Attention (MLA): a compressed low-rank KV latent shared across heads, decompressed through a
    // reconstructed `kv_b_proj` (`gguf.rs`'s `reconstruct_deepseek2_kv_b`; llama.cpp's conversion splits
    // this weight into `attn_k_b`/`attn_v_b` for its own weight-absorption decode, which poot does not
    // use), plus a decoupled-RoPE slice with DeepSeek's interleaved-pair convention. llama.cpp's GGUF
    // arch string is "deepseek2" (`src/llama-arch.cpp` `LLM_ARCH_NAMES`), unlike the safetensors
    // `model_type` "deepseek_v2"; see `gguf.rs`'s `gguf_config_deepseek2`/`gguf_deepseek2_weights` and
    // the `arch == "deepseek2"` early-return arm in `Runner::load_gguf`. `deepseek2-tiny` (under POOT_MODELS_DIR)
    // (`yujiepan/deepseek-v2-tiny-random`) is untrained, so only finite, non-degenerate output through
    // the full pipeline (GGUF -> deepseek2 detection -> trace_deepseek2_prefill -> greedy decode ->
    // detokenize) is asserted. The precise check (including the `kv_b_proj` reconstruction and YaRN
    // metadata crosswalk) is `runner.rs`'s `deepseek2_load_tests::deepseek2_tiny_gguf_matches_safetensors`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "deepseek2-tiny/deepseek2-tiny-f32.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load deepseek2-tiny gguf");
    let prompt = "The capital of France is";
    let max_new = 12;
    let toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("deepseek2-tiny gguf generate");
    let text = runner.decode(&toks).unwrap();
    eprintln!("deepseek2-tiny gguf greedy: {text:?}");
    assert!(!text.is_empty(), "decoded text must be non-empty");
    let prompt_len = runner.encode(prompt).unwrap().len();
    let generated = &toks[prompt_len..];
    assert_eq!(
        generated.len(),
        max_new,
        "must have generated max_new tokens (no early EOS on an untrained checkpoint)"
    );
    assert!(
        generated.iter().any(|&t| t != generated[0]),
        "generated tokens must not all be identical (a degenerate-output signature): {generated:?}"
    );
    assert!(
        toks.iter().all(|&t| (t as usize) < runner.config().vocab),
        "every generated token id must be within vocab bounds (a finite/non-degenerate output signature)"
    );
}

#[test]
#[ignore = "loads a REAL llama.cpp-converted GGUF of gptoss-tiny under POOT_MODELS_DIR (converted this round via \
            convert_hf_to_gguf.py, not hand-built) through the full production Runner::load_gguf -> \
            generate -> decode pipeline; run with --ignored --release"]
fn gptoss_tiny_gguf_generates_finite_output() {
    // GGUF follow-on for gpt-oss, mirroring `mixtral_tiny_gguf_generates_finite_output`/
    // `olmoe_tiny_gguf_generates_finite_output`. gpt-oss has its own GGUF arch string "gpt-oss" (hyphen,
    // unlike the HF `model_type` "gpt_oss"; llama.cpp `conversion/gpt_oss.py`'s `GptOssModel`,
    // `gguf.MODEL_ARCH.GPT_OSS`), confirmed against a `convert_hf_to_gguf.py --outtype f32` conversion of
    // `gptoss-tiny` (under POOT_MODELS_DIR). The converter did not force MXFP4 for this bf16 fixture: it only repacks to
    // MXFP4 when the source tensors are already MXFP4 `_blocks`/`_scales` pairs, and otherwise falls back
    // to plain f32 with "is not in MXFP4" warnings. See `gguf.rs`'s `gguf_weights` and `runner.rs`'s
    // `gpt_oss` detection arm, including the gate/up-interleaving re-pack (`interleave_experts_last`).
    //
    // The pipeline (GGUF -> Runner::load_gguf -> gpt-oss detection -> trace_gptoss_prefill -> greedy
    // decode -> detokenize) must produce finite, non-degenerate output; untrained weights give gibberish,
    // so coherent English is not asserted. The precise check (weight crosswalk including the interleave
    // re-pack, router bias, o_proj bias and attention-sink tensor, vs the safetensors loader via explicit
    // tokens through `bind`+`eval`) is `runner.rs`'s `gptoss_load_tests::gptoss_tiny_gguf_matches_safetensors`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "gptoss-tiny/gptoss-tiny-f32.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load gptoss-tiny gguf");
    let prompt = "The capital of France is";
    let max_new = 12;
    let toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("gptoss-tiny gguf generate");
    let text = runner.decode(&toks).unwrap();
    eprintln!("gptoss-tiny gguf greedy: {text:?}");
    assert!(!text.is_empty(), "decoded text must be non-empty");
    let prompt_len = runner.encode(prompt).unwrap().len();
    let generated = &toks[prompt_len..];
    assert_eq!(
        generated.len(),
        max_new,
        "must have generated max_new tokens (no early EOS on an untrained checkpoint)"
    );
    assert!(
        generated.iter().any(|&t| t != generated[0]),
        "generated tokens must not all be identical (a degenerate-output signature): {generated:?}"
    );
}

#[test]
#[ignore = "loads a real ~14GB bf16 safetensors checkpoint (allenai/OLMoE-1B-7B-0924-Instruct) + generates \
            on CPU; run with --ignored --release --nocapture"]
fn olmoe_1b_7b_greedy_is_coherent() {
    // OlmoE real-trained-checkpoint coherence check (card 135d, spec 262). Earlier rounds verified
    // OlmoE's tracer/loader and PTX-vs-CPU parity only against
    // `hf-tiny-v2/tiny-random-OlmoeForCausalLM` (2 layers, hidden 32, randomly initialized), which shows
    // self-consistency but not that the tracer's semantics are right on real trained weights (the gap
    // the bloom/mptk/smollm3/mixtral real-checkpoint tests close for their archs). `norm_topk_prob: false`
    // is this checkpoint's real setting, so this also shows OlmoE's non-renormalized
    // `olmoe_ffn`/`olmoe_top_k_gate` composition (built from `poot_graph_ir::ops::moe_dense` primitives,
    // not the shared `poot_graph_ir::ops::moe` op used by Mixtral/granitemoe/qwen3-moe) produces
    // coherent, factually correct text end to end.
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("olmoe-1b-7b")) else {
        return;
    };
    let runner = Runner::load(&dir).expect("load olmoe-1b-7b");
    let prompt = "The capital of France is";
    let toks = runner
        .generate(prompt, 20, |_| std::ops::ControlFlow::Continue(()))
        .expect("olmoe-1b-7b generate");
    let text = runner.decode(&toks).expect("decode");
    eprintln!("olmoe-1b-7b greedy: {text:?}");
    assert!(
        text.to_lowercase().contains("paris"),
        "olmoe-1b-7b continuation was: {text:?}"
    );
}

/// Cheap (header-only, no GPU, no weight load) precondition for
/// [`gptoss_20b_ptx_kv_masked_is_coherent`] below: the real `ggml-org/gpt-oss-20b-GGUF` file ships the
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

#[test]
#[ignore = "held for POOT-738: a Mixtral GGUF names `general.architecture = llama`, a registered family, so the Runner refuses it and the driver's llama family does not carry experts yet"]
fn mixtral_tiny_gguf_generates_finite_output() {
    // GGUF follow-on for Mixtral. Mixtral converts under llama.cpp's plain "llama" GGUF architecture
    // (`conversion/llama.py`'s `LlamaModel`, on which `MixtralForCausalLM` registers), so there is no
    // distinct arch string; see `gguf.rs`'s `gguf_weights` and `runner.rs`'s `mixtral` GGUF-detection arm
    // in `Runner::load_gguf`. This test exercises that detection against a real conversion: llama.cpp's
    // `convert_hf_to_gguf.py` on the `optimum-intel-internal-testing/tiny-mixtral` checkpoint (2 layers,
    // hidden 1024, 8 experts top-2, untrained), giving an F32 GGUF
    // (`mixtral-tiny/mixtral-tiny-f32.gguf` under POOT_MODELS_DIR, ~942MB, a local model-cache artifact, not
    // committed).
    //
    // It proves the pipeline (GGUF -> Runner::load_gguf -> mixtral detection -> trace_mixtral_prefill ->
    // greedy decode -> detokenize) runs end to end and produces finite, non-degenerate output; untrained
    // weights give gibberish, so coherent English is not asserted. The numerically precise check, that the
    // GGUF weight crosswalk (`transpose_experts`/`concat_experts_last` in `gguf.rs`) matches the
    // safetensors loader's `fuse_mixtral_experts` on identical weights via explicit tokens through
    // `bind`+`eval` (bypassing `generate()`'s tokenization, which would confound it with the loaders'
    // BOS-handling asymmetry), lives in `runner.rs`'s `mixtral_load_tests::mixtral_tiny_gguf_matches_safetensors`.
    let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
        "mixtral-tiny/mixtral-tiny-f32.gguf"
    )) else {
        return;
    };
    let runner = Runner::load_gguf(&gguf_path).expect("load mixtral-tiny gguf");
    let prompt = "The capital of France is";
    let max_new = 12;
    let toks = runner
        .generate(prompt, max_new, |_| std::ops::ControlFlow::Continue(()))
        .expect("mixtral-tiny gguf generate");
    let text = runner.decode(&toks).unwrap();
    eprintln!("mixtral-tiny gguf greedy: {text:?}");
    assert!(!text.is_empty(), "decoded text must be non-empty");
    let prompt_len = runner.encode(prompt).unwrap().len();
    let generated = &toks[prompt_len..];
    assert_eq!(
        generated.len(),
        max_new,
        "must have generated max_new tokens (no early EOS on an untrained checkpoint)"
    );
    assert!(
        generated.iter().any(|&t| t != generated[0]),
        "generated tokens must not all be identical (a degenerate-output signature): {generated:?}"
    );
}

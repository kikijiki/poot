//! Embeddings and reranking with poot's encoder models. Loads a BERT-class sentence encoder and shows:
//!   1. sentence embeddings: an L2-normalized vector per text, where cosine (== dot product) measures
//!      semantic similarity (related sentences score higher than unrelated ones);
//!   2. bi-encoder reranking: ordering candidate documents by similarity to a query.
//!
//! Pass a BERT embedding model directory (BertModel: all-MiniLM / E5 / BGE; pooling mean or CLS is read from
//! its `1_Pooling` config). Optionally pass a cross-encoder directory (BertForSequenceClassification:
//! ms-marco-MiniLM) as a second arg to also show cross-encoder reranking, which scores a query+document pair
//! jointly in one pass. With no directory argument it reads `all-minilm-l6-v2` under `POOT_MODELS_DIR`, and
//! fails naming that variable when it is unset. A directory without a model prints the usage and exits cleanly.
//!
//!   cargo run --release --example embeddings -- /path/to/all-minilm-l6-v2 [ /path/to/ms-marco-minilm ]

use poot_llm::encoder::{CrossEncoderRunner, EncoderRunner};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    // both vectors are L2-normalized by `embed`, so the dot product is the cosine similarity.
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> anyhow::Result<()> {
    let dir = match std::env::args().nth(1) {
        Some(dir) => dir,
        None => {
            let models = std::env::var("POOT_MODELS_DIR")
                .ok()
                .filter(|dir| !dir.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no encoder directory given and POOT_MODELS_DIR is not set: pass a directory or set POOT_MODELS_DIR to the directory that holds `all-minilm-l6-v2`"
                    )
                })?;
            std::path::Path::new(&models)
                .join("all-minilm-l6-v2")
                .to_string_lossy()
                .into_owned()
        }
    };
    if !std::path::Path::new(&dir)
        .join("model.safetensors")
        .exists()
    {
        eprintln!("encoder model not found at {dir}");
        eprintln!("usage: embeddings <bert-encoder-dir> [cross-encoder-dir]");
        return Ok(());
    }

    let enc = EncoderRunner::load(&dir)?;
    println!(
        "loaded encoder: dim {}, pooling {:?}",
        enc.dim(),
        enc.pooling()
    );

    // 1. sentence embeddings: related sentences should be more similar than unrelated ones.
    let query = "a small domestic cat";
    let docs = [
        "a young playful kitten",  // related
        "a fast red sports car",   // unrelated
        "the weather forecast",    // unrelated
        "a fluffy tabby sleeping", // related
    ];
    let q = enc.embed(query)?;
    println!("\nembeddings - cosine similarity to {query:?}:");
    for d in docs {
        println!("  {:+.3}  {d}", cosine(&q, &enc.embed(d)?));
    }

    // 2. bi-encoder rerank: order the documents by similarity to the query.
    println!("\nbi-encoder rerank (most relevant first):");
    let refs: Vec<&str> = docs.to_vec();
    for (rank, (i, score)) in enc.rerank(query, &refs)?.iter().enumerate() {
        println!("  {}. {:+.3}  {}", rank + 1, score, docs[*i]);
    }

    // 3. optional: cross-encoder rerank (scores the pair jointly - higher quality).
    if let Some(ce_dir) = std::env::args().nth(2) {
        if std::path::Path::new(&ce_dir)
            .join("model.safetensors")
            .exists()
        {
            let ce = CrossEncoderRunner::load(&ce_dir)?;
            println!(
                "\ncross-encoder rerank of {query:?} (raw relevance logit, most relevant first):"
            );
            for (rank, (i, score)) in ce.rerank(query, &refs)?.iter().enumerate() {
                println!("  {}. {:+.3}  {}", rank + 1, score, docs[*i]);
            }
        } else {
            eprintln!("\ncross-encoder model not found at {ce_dir}; skipping cross-encoder demo");
        }
    }
    Ok(())
}

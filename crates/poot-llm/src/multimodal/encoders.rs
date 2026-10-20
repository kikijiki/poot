//! The encoder front ends over [`Driver::encode`]: sentence embeddings, bi-encoder reranking and
//! cross-encoder reranking (POOT-739).
//!
//! The model side is the registered `bert` family (or a decoder family's `encode` program): it returns
//! every position's final hidden state, or a cross-encoder's classifier score. What a sentence vector
//! is, the pooling of those hidden states and their normalization, is the host's: [`Pooling`] and
//! [`pool`] are plain arithmetic on the rows the driver returns, so they are exact and need no
//! device.

use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use poot_executor::Executor;
use poot_models::bert::TOKEN_TYPE;
use poot_models::io::{InputValue, RequestInputs, TokenValues};
use poot_models::registry::Registry;
use poot_tensor::HostTensor;

use crate::driver::error::DriverError;
use crate::driver::{
    Driver, DriverOptions, EncodeRequest, EncodeRow, EncodeShapes, Encoded, LoadOptions,
    ModelHandle,
};
use poot_text::TextError;

/// Why an encoder request is refused.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    /// The driver refused or failed the `Encode` step.
    #[error(transparent)]
    Driver(#[from] DriverError),
    /// Tokenizing the text for the encoder failed.
    #[error(transparent)]
    Text(#[from] TextError),
    /// The text tokenizes to no token at all, so there is no row to pool or score.
    #[error("cannot embed empty text")]
    EmptyText,
}

/// The encoder checkpoint in `dir` as a handle and a driver on `executor`. An encode step has no
/// cache a later step reads, so the driver's capacity and chunk are the model's own positions.
fn open(
    dir: &Path,
    executor: Box<dyn Executor>,
    rows: NonZeroUsize,
) -> Result<(Driver, Arc<ModelHandle>), EncoderError> {
    let handle = Arc::new(ModelHandle::load(
        dir,
        &Registry::builtin().expect("shipped families register disjoint keys"),
        &LoadOptions::default(),
    )?);
    let positions = NonZeroUsize::new(handle.config().max_positions)
        .expect("a model has at least one position");
    let options = DriverOptions::tower(positions);
    let mut driver = Driver::new(Arc::clone(&handle), executor, options)?;
    driver.prepare_encode(&EncodeShapes {
        rows,
        max_len: positions,
    })?;
    Ok((driver, handle))
}

/// How a sentence-transformers model pools the encoder's token hidden states into one sentence
/// vector. MiniLM and E5 mean-pool; BGE and GTE take the `[CLS]` token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// The average over every token position.
    Mean,
    /// The `[CLS]` token's (position 0) hidden state.
    Cls,
}

impl Pooling {
    /// The pooling the checkpoint in `dir` declares: `1_Pooling/config.json`'s
    /// `pooling_mode_cls_token`. A checkpoint with no pooling module (a bare BERT) mean-pools, the
    /// MiniLM convention.
    pub fn read(dir: &Path) -> Self {
        let config = std::fs::read(dir.join("1_Pooling").join("config.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
        match config
            .as_ref()
            .and_then(|c| c.get("pooling_mode_cls_token"))
            .and_then(|v| v.as_bool())
        {
            Some(true) => Pooling::Cls,
            _ => Pooling::Mean,
        }
    }
}

/// The sentence vector of `hidden` (`tokens * width` values, row-major): pooled by `pooling`, then
/// L2-normalized (so the dot product of two vectors is their cosine). A zero vector stays zero.
pub fn pool(hidden: &[f32], width: usize, pooling: Pooling) -> Vec<f32> {
    let tokens = hidden.len() / width;
    let mut vector = match pooling {
        Pooling::Cls => hidden[..width].to_vec(),
        Pooling::Mean => {
            let mut sum = vec![0.0f32; width];
            for row in hidden.chunks(width) {
                for (acc, v) in sum.iter_mut().zip(row) {
                    *acc += v;
                }
            }
            sum.iter_mut().for_each(|v| *v /= tokens as f32);
            sum
        }
    };
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
    vector.iter_mut().for_each(|v| *v /= norm);
    vector
}

/// A relevance ranking: `(index, score)` pairs, most relevant first.
pub type Ranking = Vec<(usize, f32)>;

/// `scores` ordered most relevant first, each with its index in the input.
pub fn rank(scores: impl IntoIterator<Item = f32>) -> Ranking {
    let mut ranked: Ranking = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked
}

/// A sentence encoder: embeddings and bi-encoder reranking of any model whose `Encode` step returns
/// hidden states (a BERT embedder, or a decoder's embedding pass).
pub struct SentenceEncoder {
    driver: Driver,
    handle: Arc<ModelHandle>,
    pooling: Pooling,
}

impl SentenceEncoder {
    /// The sentence-transformers checkpoint in `dir` on `executor`, pooled as its `1_Pooling` says, ready
    /// to encode up to `rows` texts at once.
    pub fn load(
        dir: &Path,
        executor: Box<dyn Executor>,
        rows: NonZeroUsize,
    ) -> Result<Self, EncoderError> {
        let (driver, handle) = open(dir, executor, rows)?;
        Ok(Self {
            driver,
            handle,
            pooling: Pooling::read(dir),
        })
    }

    /// The pooling this encoder applies, read from the checkpoint's `1_Pooling` module at load.
    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// The driver the encoder runs its `Encode` steps on.
    pub fn driver(&self) -> &Driver {
        &self.driver
    }

    /// The hidden states of each text in one `Encode` step: every real token's `width` values.
    fn hidden_states(&mut self, texts: &[&str]) -> Result<Encoded, EncoderError> {
        let tokens = texts
            .iter()
            .map(|text| self.handle.text().encode_for_encoder(text, None))
            .collect::<Result<Vec<_>, _>>()?;
        if tokens.iter().any(|row| row.ids.is_empty()) {
            return Err(EncoderError::EmptyText);
        }
        let inputs: Vec<RequestInputs> = tokens
            .iter()
            .map(|row| segment_inputs(&row.types))
            .collect();
        let rows: Vec<EncodeRow<'_>> = tokens
            .iter()
            .zip(&inputs)
            .map(|(row, inputs)| EncodeRow {
                tokens: Some(&row.ids),
                inputs,
            })
            .collect();
        Ok(self.driver.encode(&EncodeRequest { rows: &rows })?)
    }

    /// The L2-normalized sentence vector of each text, pooled as the model's `1_Pooling` says.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EncoderError> {
        let encoded = self.hidden_states(texts)?;
        Ok(encoded
            .rows
            .iter()
            .map(|row| pool(row, encoded.width, self.pooling))
            .collect())
    }

    /// Bi-encoder rerank: each document scored by its cosine to the query, most relevant first.
    pub fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Ranking, EncoderError> {
        let mut texts = vec![query];
        texts.extend_from_slice(docs);
        let vectors = self.embed(&texts)?;
        let (query, docs) = vectors
            .split_first()
            .expect("the query is one of the texts");
        Ok(rank(docs.iter().map(|doc| {
            query.iter().zip(doc).map(|(a, b)| a * b).sum::<f32>()
        })))
    }
}

/// The request inputs of an encoder row whose tokens carry segment ids `types`: the dense segment
/// column when some token is in a later segment, nothing (the model's default, all zeros) otherwise.
pub fn segment_inputs(types: &[u32]) -> RequestInputs {
    if types.iter().all(|&kind| kind == 0) {
        return RequestInputs::none();
    }
    let words: Vec<i32> = types.iter().map(|&kind| kind as i32).collect();
    let values = TokenValues::dense(HostTensor::i32(vec![words.len(), 1], words))
        .expect("a column of segment ids is [tokens, 1]");
    RequestInputs::none().with(TOKEN_TYPE, InputValue::Tokens(values))
}

/// A cross-encoder reranker (`BertForSequenceClassification`): scores a query and a document
/// jointly in one `Encode` step per batch of pairs.
pub struct CrossEncoder {
    driver: Driver,
    handle: Arc<ModelHandle>,
}

impl CrossEncoder {
    /// The `BertForSequenceClassification` checkpoint in `dir` on `executor`, ready to score up to `rows`
    /// pairs at once.
    pub fn load(
        dir: &Path,
        executor: Box<dyn Executor>,
        rows: NonZeroUsize,
    ) -> Result<Self, EncoderError> {
        let (driver, handle) = open(dir, executor, rows)?;
        Ok(Self { driver, handle })
    }

    /// The driver the cross-encoder runs its `Encode` steps on.
    pub fn driver(&self) -> &Driver {
        &self.driver
    }

    /// The relevance logit of `query` against each of `docs` (higher is more relevant; the model's
    /// output activation is the identity, so the raw logit is the score), in one step: each
    /// `[CLS] query [SEP] doc [SEP]` pair right-padded to the longest and masked.
    pub fn score(&mut self, query: &str, docs: &[&str]) -> Result<Vec<f32>, EncoderError> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        let pairs = docs
            .iter()
            .map(|doc| self.handle.text().encode_for_encoder(query, Some(doc)))
            .collect::<Result<Vec<_>, _>>()?;
        let inputs: Vec<RequestInputs> = pairs
            .iter()
            .map(|pair| segment_inputs(&pair.types))
            .collect();
        let rows: Vec<EncodeRow<'_>> = pairs
            .iter()
            .zip(&inputs)
            .map(|(pair, inputs)| EncodeRow {
                tokens: Some(&pair.ids),
                inputs,
            })
            .collect();
        let encoded = self.driver.encode(&EncodeRequest { rows: &rows })?;
        Ok(encoded.rows.iter().map(|row| row[0]).collect())
    }

    /// Rerank `docs` against `query`, most relevant first.
    pub fn rerank(&mut self, query: &str, docs: &[&str]) -> Result<Ranking, EncoderError> {
        Ok(rank(self.score(query, docs)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean pooling averages every position and CLS takes the first, each then unit length.
    #[test]
    fn pooling_averages_or_takes_the_first_token_and_normalizes() {
        let hidden = [3.0, 0.0, 1.0, 4.0, 5.0, 4.0];
        let mean = pool(&hidden, 2, Pooling::Mean);
        // mean = [3, 2.666..]; unit length.
        let norm = (3.0f32 * 3.0 + (8.0f32 / 3.0).powi(2)).sqrt();
        assert!((mean[0] - 3.0 / norm).abs() < 1e-6 && (mean[1] - (8.0 / 3.0) / norm).abs() < 1e-6);
        assert_eq!(pool(&hidden, 2, Pooling::Cls), vec![1.0, 0.0]);
        assert_eq!(pool(&[0.0, 0.0], 2, Pooling::Cls), vec![0.0, 0.0]);
    }

    #[test]
    fn a_ranking_sorts_descending_and_keeps_indices() {
        assert_eq!(
            rank([0.5, 2.0, -1.0, 1.0]),
            vec![(1, 2.0), (3, 1.0), (0, 0.5), (2, -1.0)]
        );
    }

    /// `1_Pooling/config.json` picks the pooling; no file or a false flag is mean.
    #[test]
    fn the_pooling_mode_is_read_from_the_checkpoints_pooling_module() {
        let base = std::env::temp_dir().join(format!("poot-pooling-{}", std::process::id()));
        let none = base.join("none");
        std::fs::create_dir_all(&none).unwrap();
        assert_eq!(Pooling::read(&none), Pooling::Mean);
        for (name, config, want) in [
            ("cls", r#"{"pooling_mode_cls_token": true}"#, Pooling::Cls),
            (
                "mean",
                r#"{"pooling_mode_cls_token": false}"#,
                Pooling::Mean,
            ),
        ] {
            let dir = base.join(name);
            std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
            std::fs::write(dir.join("1_Pooling").join("config.json"), config).unwrap();
            assert_eq!(Pooling::read(&dir), want, "{name}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}

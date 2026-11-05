//! The encoder front ends over [`Driver::encode`]: sentence embeddings, bi-encoder reranking and
//! cross-encoder reranking (POOT-739).
//!
//! The model side is the registered `bert` family (or a decoder family's `encode` program): it returns
//! every position's final hidden state, or a cross-encoder's classifier score. What a sentence vector
//! is, the pooling of those hidden states and their normalization, is the host's: [`Pooling`] and
//! [`pool`] are plain arithmetic on the rows the driver returns, so they are exact and need no
//! device.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
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
    /// The checkpoint's pooling module is unreadable or selects a pooling poot does not implement.
    #[error(transparent)]
    Pooling(#[from] PoolingError),
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

/// Why a checkpoint's `1_Pooling/config.json` is refused.
#[derive(Debug, thiserror::Error)]
pub enum PoolingError {
    /// The file exists but cannot be read.
    #[error("pooling config {path}: {source}")]
    Unreadable {
        /// The `1_Pooling/config.json` that could not be read.
        path: PathBuf,
        /// The underlying I/O failure.
        source: std::io::Error,
    },
    /// The file is not a JSON object.
    #[error("pooling config {path}: {source}")]
    Malformed {
        /// The `1_Pooling/config.json` that did not parse.
        path: PathBuf,
        /// The parse failure.
        source: serde_json::Error,
    },
    /// A `pooling_mode_*` field is present and not a boolean.
    #[error("pooling config {path}: `{field}` must be a boolean")]
    NotBoolean {
        /// The `1_Pooling/config.json` carrying the field.
        path: PathBuf,
        /// The offending field.
        field: &'static str,
    },
    /// The checkpoint selects a pooling poot does not implement.
    #[error(
        "pooling config {path}: `{field}` is set, and poot implements only CLS and mean pooling"
    )]
    Unsupported {
        /// The `1_Pooling/config.json` carrying the field.
        path: PathBuf,
        /// The `pooling_mode_*` field that is true.
        field: &'static str,
    },
    /// Not exactly one of `pooling_mode_cls_token` and `pooling_mode_mean_tokens` is set.
    #[error(
        "pooling config {path}: exactly one of `pooling_mode_cls_token` and `pooling_mode_mean_tokens` must be true"
    )]
    NoSingleMode {
        /// The `1_Pooling/config.json` that selects no single supported mode.
        path: PathBuf,
    },
}

impl Pooling {
    /// The `pooling_mode_*` flags sentence-transformers defines besides CLS and mean; a checkpoint that
    /// sets one pools in a way [`pool`] does not implement.
    const UNSUPPORTED_MODES: [&'static str; 4] = [
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
        "pooling_mode_lasttoken",
    ];

    /// The pooling the checkpoint in `dir` declares in `1_Pooling/config.json`. A checkpoint with no
    /// pooling module (a bare BERT) mean-pools, the MiniLM convention; a module that is present must be
    /// readable and select exactly CLS or mean (an absent flag is false, as in sentence-transformers).
    pub fn read(dir: &Path) -> Result<Self, PoolingError> {
        let path = dir.join("1_Pooling").join("config.json");
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Pooling::Mean),
            Err(source) => return Err(PoolingError::Unreadable { path, source }),
        };
        let config: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|source| PoolingError::Malformed {
                path: path.clone(),
                source,
            })?;
        let flag = |field: &'static str| match config.get(field) {
            None => Ok(false),
            Some(value) => value.as_bool().ok_or_else(|| PoolingError::NotBoolean {
                path: path.clone(),
                field,
            }),
        };
        for field in Self::UNSUPPORTED_MODES {
            if flag(field)? {
                return Err(PoolingError::Unsupported { path, field });
            }
        }
        match (
            flag("pooling_mode_cls_token")?,
            flag("pooling_mode_mean_tokens")?,
        ) {
            (true, false) => Ok(Pooling::Cls),
            (false, true) => Ok(Pooling::Mean),
            _ => Err(PoolingError::NoSingleMode { path }),
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
        let pooling = Pooling::read(dir)?;
        let (driver, handle) = open(dir, executor, rows)?;
        Ok(Self {
            driver,
            handle,
            pooling,
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

    fn pooling_dir(name: &str, config: Option<&str>) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("poot-pooling-{}-{name}", std::process::id()));
        std::fs::create_dir_all(dir.join("1_Pooling")).unwrap();
        if let Some(config) = config {
            std::fs::write(dir.join("1_Pooling").join("config.json"), config).unwrap();
        }
        dir
    }

    /// `1_Pooling/config.json` picks the pooling; no pooling module is mean.
    #[test]
    fn the_pooling_mode_is_read_from_the_checkpoints_pooling_module() {
        let none = std::env::temp_dir().join(format!("poot-pooling-{}-none", std::process::id()));
        std::fs::create_dir_all(&none).unwrap();
        assert_eq!(Pooling::read(&none).unwrap(), Pooling::Mean);
        for (name, config, want) in [
            (
                "cls",
                r#"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}"#,
                Pooling::Cls,
            ),
            (
                "mean",
                r#"{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}"#,
                Pooling::Mean,
            ),
            (
                "mean-absent-cls",
                r#"{"pooling_mode_mean_tokens": true}"#,
                Pooling::Mean,
            ),
        ] {
            let dir = pooling_dir(name, Some(config));
            assert_eq!(Pooling::read(&dir).unwrap(), want, "{name}");
        }
    }

    /// A pooling module that is present but corrupt, or that selects a pooling poot does not implement,
    /// is a typed refusal naming the file and field, never a silent fall back to mean pooling.
    #[test]
    fn a_corrupt_or_unsupported_pooling_module_is_refused() {
        let corrupt = pooling_dir("corrupt", Some("{not json"));
        assert!(matches!(
            Pooling::read(&corrupt),
            Err(PoolingError::Malformed { .. })
        ));
        let unreadable = pooling_dir("unreadable", None);
        std::fs::create_dir(unreadable.join("1_Pooling").join("config.json")).unwrap();
        assert!(matches!(
            Pooling::read(&unreadable),
            Err(PoolingError::Unreadable { .. })
        ));
        let not_bool = pooling_dir("not-bool", Some(r#"{"pooling_mode_cls_token": "yes"}"#));
        assert!(matches!(
            Pooling::read(&not_bool),
            Err(PoolingError::NotBoolean {
                field: "pooling_mode_cls_token",
                ..
            })
        ));
        let lasttoken = pooling_dir(
            "lasttoken",
            Some(r#"{"pooling_mode_mean_tokens": false, "pooling_mode_lasttoken": true}"#),
        );
        assert!(matches!(
            Pooling::read(&lasttoken),
            Err(PoolingError::Unsupported {
                field: "pooling_mode_lasttoken",
                ..
            })
        ));
        for (name, config) in [
            ("neither", r#"{"pooling_mode_cls_token": false}"#),
            (
                "both",
                r#"{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": true}"#,
            ),
        ] {
            let dir = pooling_dir(name, Some(config));
            assert!(
                matches!(Pooling::read(&dir), Err(PoolingError::NoSingleMode { .. })),
                "{name}"
            );
        }
    }
}

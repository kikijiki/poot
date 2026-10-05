//! [`ModelHandle`]: a loaded checkpoint, immutable and shared by every driver (Card 734). It holds the
//! family's [`Model`] built through a [`Registry`], the checkpoint's weights exactly as stored, and the
//! [`TextCodec`] of its tokenizer and chat template. It names no family and no backend.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf};
use poot_models::model::{Model, ModelConfig};
use poot_models::registry::{FamilyEntry, RawConfig, Registry};
use poot_quant::weights::WeightStore;

use crate::driver::error::{DriverError, LoadError, Unsupported};
use crate::text::tokenize::TextCodec;

/// A model, its weights and its text services.
#[derive(Debug)]
pub struct ModelHandle {
    model: Box<dyn Model>,
    store: Arc<WeightStore>,
    text: TextCodec,
}

/// The id a guided constraint forces once its pattern is complete: the lowest declared end-of-sequence
/// token, or one no vocabulary holds when the checkpoint declares none.
fn constraint_eos(config: &ModelConfig) -> u32 {
    config.eos.first().copied().unwrap_or(u32::MAX)
}

fn io_error(path: &Path) -> impl FnOnce(std::io::Error) -> LoadError + '_ {
    move |source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, LoadError> {
    let bytes = std::fs::read(path).map_err(io_error(path))?;
    serde_json::from_slice(&bytes).map_err(|source| LoadError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// The registered family `raw` names, or [`Unsupported::Registry`].
fn resolve<'r>(
    raw: &RawConfig<'_>,
    registry: &'r Registry,
) -> Result<&'r FamilyEntry, DriverError> {
    registry
        .resolve(raw)
        .map_err(|e| DriverError::Unsupported(Unsupported::Registry(e)))
}

impl ModelHandle {
    /// Load `path`: a directory is an HF checkpoint (`config.json`, `generation_config.json` when it
    /// ships one, safetensors and `tokenizer.json`), a file is a GGUF. The family is resolved from the
    /// config before any weight is read, so an unregistered family is refused without reading the
    /// checkpoint's tensors; the one resolve then builds the model.
    pub fn load(path: &Path, registry: &Registry) -> Result<Self, DriverError> {
        if path.is_dir() {
            Self::load_hf(path, registry)
        } else {
            Self::load_gguf(path, registry)
        }
    }

    fn load_hf(dir: &Path, registry: &Registry) -> Result<Self, DriverError> {
        let config = read_json(&dir.join("config.json"))?;
        let generation_path = dir.join("generation_config.json");
        let generation = generation_path
            .exists()
            .then(|| read_json(&generation_path))
            .transpose()?;
        let raw = RawConfig::HfJson {
            config: &config,
            generation: generation.as_ref(),
        };
        let family = resolve(&raw, registry)?;
        // A GPTQ, AWQ or FP8 checkpoint's quantized linears are packed straight from their stored
        // bytes, never decoded: the family maps the packed entries like any other weight.
        let scheme = poot_load::config::quant_scheme(&config).map_err(LoadError::from)?;
        let store = poot_load::safetensors::load_weight_store(dir).map_err(LoadError::from)?;
        let store = match scheme {
            Some(scheme) => poot_load::safetensors::pack_quantized_linears(&store, scheme)
                .map_err(LoadError::from)?,
            None => store,
        };
        Self::build(family, &raw, store, |config| {
            TextCodec::from_hf_dir(dir, constraint_eos(config), config.prompt_bos, config.chat)
                .map_err(|e| LoadError::Text(Box::new(e)))
        })
    }

    fn load_gguf(path: &Path, registry: &Registry) -> Result<Self, DriverError> {
        let index = GgufIndex::open(path).map_err(LoadError::from)?;
        let raw = RawConfig::Gguf(&index);
        let family = resolve(&raw, registry)?;
        let file = File::open(path).map_err(io_error(path))?;
        let store = read_gguf(&index, &file, &IdentityNames).map_err(LoadError::from)?;
        Self::build(family, &raw, store, |config| {
            TextCodec::from_gguf(&index, constraint_eos(config), config.chat)
                .map_err(|e| LoadError::Text(Box::new(e)))
        })
    }

    /// Build the family `raw` names over an in-memory `store` (a test fixture), then its text services
    /// from the model's own config. An unregistered family is [`Unsupported::Registry`], a checkpoint
    /// the family cannot take a [`DriverError::Model`].
    #[cfg(test)]
    pub(crate) fn from_checkpoint(
        raw: &RawConfig<'_>,
        store: WeightStore,
        registry: &Registry,
        text: impl FnOnce(&ModelConfig) -> Result<TextCodec, LoadError>,
    ) -> Result<Self, DriverError> {
        Self::build(resolve(raw, registry)?, raw, store, text)
    }

    /// Build `family`'s model over `store`, then its text services from the model's own config.
    fn build(
        family: &FamilyEntry,
        raw: &RawConfig<'_>,
        store: WeightStore,
        text: impl FnOnce(&ModelConfig) -> Result<TextCodec, LoadError>,
    ) -> Result<Self, DriverError> {
        let model = (family.build)(raw, &store).map_err(DriverError::Model)?;
        let text = text(model.config())?;
        Ok(Self {
            model,
            store: Arc::new(store),
            text,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    pub fn text(&self) -> &TextCodec {
        &self.text
    }

    pub(crate) fn model(&self) -> &dyn Model {
        &*self.model
    }

    /// The checkpoint's weights exactly as stored.
    pub fn store(&self) -> &WeightStore {
        &self.store
    }

    pub(crate) fn shared_store(&self) -> &Arc<WeightStore> {
        &self.store
    }
}

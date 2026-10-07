//! [`ModelHandle`]: a loaded checkpoint, immutable and shared by every driver (Card 734). It holds the
//! family's [`Model`] built through a [`Registry`], the checkpoint's weights exactly as stored, and the
//! [`TextCodec`] of its tokenizer and chat template. It names no family and no backend.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf};
use poot_load::packed_safetensors::{FilePin, Sha256Digest, check_pin_reader};
use poot_models::model::{Model, ModelConfig};
use poot_models::registry::{FamilyEntry, RawConfig, Registry};
use poot_quant::weights::WeightStore;

use crate::driver::error::{DriverError, LoadError, Unsupported};
use crate::text::tokenize::TextCodec;

/// How [`ModelHandle::load`] treats the checkpoint's files.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// A JSON manifest pinning checkpoint files by name:
    /// `{"files": {"model.safetensors": {"length": 123, "sha256": "<64 hex digits>"}}}`.
    /// Every listed file is checked against its pin before any tensor is read, and every weight
    /// file the load will parse (the GGUF itself, each `*.safetensors` shard of an HF directory)
    /// must be listed: an unlisted one is refused. `None` checks nothing.
    pub integrity: Option<PathBuf>,
}

#[derive(serde::Deserialize)]
struct Manifest {
    files: BTreeMap<String, ManifestPin>,
}

#[derive(serde::Deserialize)]
struct ManifestPin {
    length: usize,
    #[serde(deserialize_with = "digest_from_hex")]
    sha256: [u8; 32],
}

fn digest_from_hex<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
    use serde::de::Error;
    let hex = <String as serde::Deserialize>::deserialize(d)?;
    if hex.len() != 64 || !hex.is_ascii() {
        return Err(D::Error::custom("sha256 must be 64 hex digits"));
    }
    let mut out = [0u8; 32];
    for (byte, pair) in out.iter_mut().zip(hex.as_bytes().chunks(2)) {
        let pair = std::str::from_utf8(pair).map_err(D::Error::custom)?;
        *byte = u8::from_str_radix(pair, 16).map_err(D::Error::custom)?;
    }
    Ok(out)
}

/// Check the checkpoint's files against `options.integrity`, each streamed through the hash once,
/// before any tensor is read. The weight files the load will parse must all be pinned: a GGUF's own
/// file name (`only`), or for an HF directory every `*.safetensors` shard in it. An unpinned weight
/// file is [`LoadError::Unpinned`], never admitted unchecked. Files that are not tensors (config,
/// tokenizer, the shard index) need no pin, but an HF manifest entry for one is still checked.
fn check_integrity(options: &LoadOptions, dir: &Path, only: Option<&str>) -> Result<(), LoadError> {
    let Some(manifest_path) = &options.integrity else {
        return Ok(());
    };
    let text = std::fs::read(manifest_path).map_err(io_error(manifest_path))?;
    let manifest: Manifest = serde_json::from_slice(&text).map_err(|source| LoadError::Json {
        path: manifest_path.clone(),
        source,
    })?;
    let mut required: Vec<String> = match only {
        Some(name) => vec![name.to_string()],
        None => std::fs::read_dir(dir)
            .map_err(io_error(dir))?
            .map(|entry| entry.map_err(io_error(dir)))
            .filter_map(|entry| match entry {
                Ok(entry) => {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    name.ends_with(".safetensors").then_some(Ok(name))
                }
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<_, _>>()?,
    };
    required.sort();
    if let Some(file) = required
        .iter()
        .find(|name| !manifest.files.contains_key(*name))
    {
        return Err(LoadError::Unpinned { file: file.clone() });
    }
    for (name, pin) in &manifest.files {
        if only.is_some_and(|only| only != name) {
            continue;
        }
        let path = dir.join(name);
        let file = File::open(&path).map_err(io_error(&path))?;
        let pin = FilePin {
            length: pin.length,
            sha256: Sha256Digest::new(pin.sha256),
        };
        check_pin_reader(file, pin, name)?;
    }
    Ok(())
}

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
    pub fn load(
        path: &Path,
        registry: &Registry,
        options: &LoadOptions,
    ) -> Result<Self, DriverError> {
        if path.is_dir() {
            Self::load_hf(path, registry, options)
        } else {
            Self::load_gguf(path, registry, options)
        }
    }

    fn load_hf(
        dir: &Path,
        registry: &Registry,
        options: &LoadOptions,
    ) -> Result<Self, DriverError> {
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
        check_integrity(options, dir, None)?;
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

    fn load_gguf(
        path: &Path,
        registry: &Registry,
        options: &LoadOptions,
    ) -> Result<Self, DriverError> {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            check_integrity(options, path.parent().unwrap_or(Path::new("")), Some(name))?;
        }
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

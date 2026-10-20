//! [`ModelHandle`]: a loaded checkpoint, immutable and shared by every driver (Card 734). It holds the
//! family's model container built through a [`Registry`] ([`ModelParts`]: the served [`Model`] and, for a
//! vision-language family, its towers and front end), the checkpoint's weights exactly as stored, and the
//! [`TextCodec`] of its tokenizer and chat template. The checkpoint's config files are read here, once.
//! It names no family and no backend.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf, shard_paths};
use poot_load::packed_safetensors::{FilePin, Sha256Digest, check_pin_reader};
use poot_load::plan::default_workers;
use poot_models::model::{Model, ModelConfig};
use poot_models::multimodal::front_end::{FrontEnd, TowerId};
use poot_models::registry::{FamilyEntry, ModelParts, RawConfig, Registry};
use poot_quant::StoredBytes;
use poot_quant::weights::WeightStore;

use crate::driver::error::{DriverError, LoadError, Unsupported};
use crate::driver::gguf_vocab::gguf_vocab;
use poot_text::{ChatTemplateOverride, CheckedChatTemplate, HfTextFiles, TextCodec, TextError};

/// How [`ModelHandle::load`] treats the checkpoint's files.
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    /// A JSON manifest pinning checkpoint files by name:
    /// `{"files": {"model.safetensors": {"length": 123, "sha256": "<64 hex digits>"}}}`.
    /// Every file the load reads must be listed and match its pin: the weight files (the GGUF
    /// itself, each `*.safetensors` shard of an HF directory) before any tensor is read, and the
    /// config, tokenizer and chat-template files as the exact bytes the load parses. An unlisted
    /// one is refused. `None` checks nothing.
    pub integrity: Option<PathBuf>,
    /// A chat template to render with in place of the checkpoint's own (or in place of none, for a base
    /// model). `None` keeps the checkpoint's.
    pub chat_template: Option<ChatTemplateOverride>,
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

/// The checkpoint's integrity manifest, parsed once ([`LoadOptions::integrity`]); no manifest pins nothing.
///
/// Every file the load reads is pinned when a manifest is given: the weight files are checked over the
/// mappings the store views ([`Self::check_weights`]), and every other file is read once into held bytes
/// that are hashed and then parsed ([`Self::read`]), so the digest and the parse come from the same
/// bytes. A file the manifest does not list is [`LoadError::Unpinned`], never admitted unchecked.
pub(super) struct Pins {
    files: Option<BTreeMap<String, FilePin>>,
}

impl Pins {
    pub(super) fn load(options: &LoadOptions) -> Result<Self, LoadError> {
        let Some(manifest_path) = &options.integrity else {
            return Ok(Self { files: None });
        };
        let text = std::fs::read(manifest_path).map_err(io_error(manifest_path))?;
        let manifest: Manifest =
            serde_json::from_slice(&text).map_err(|source| LoadError::Json {
                path: manifest_path.clone(),
                source,
            })?;
        let files = manifest
            .files
            .into_iter()
            .map(|(name, pin)| {
                let pin = FilePin {
                    length: pin.length,
                    sha256: Sha256Digest::new(pin.sha256),
                };
                (name, pin)
            })
            .collect();
        Ok(Self { files: Some(files) })
    }

    /// The bytes of `dir/name`, read once and checked against the file's pin when there is a manifest.
    pub(super) fn read(&self, dir: &Path, name: &str) -> Result<Vec<u8>, LoadError> {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).map_err(io_error(&path))?;
        if let Some(files) = &self.files {
            let pin = files.get(name).ok_or_else(|| LoadError::Unpinned {
                file: name.to_string(),
            })?;
            check_pin_reader(bytes.as_slice(), *pin, name)?;
        }
        Ok(bytes)
    }

    /// [`Self::read`] of a file the checkpoint may not ship: `None` when `dir/name` does not exist.
    pub(super) fn read_optional(
        &self,
        dir: &Path,
        name: &str,
    ) -> Result<Option<Vec<u8>>, LoadError> {
        dir.join(name)
            .exists()
            .then(|| self.read(dir, name))
            .transpose()
    }

    /// Check the weight files of the load against the manifest, each hashed once, before any tensor is
    /// read. The weight files the load will parse must all be pinned: a GGUF's own file name (`only`),
    /// or for an HF directory every `*.safetensors` shard in it.
    ///
    /// `mapped` holds the weight files the load will read, already mapped: a pin on one of them is checked
    /// against that mapping, so the digest, the header and the tensors all come from one inode (ADR-0103
    /// amendment). A pinned shard that is not one of them (one the index does not name) is streamed from
    /// its path.
    pub(super) fn check_weights(
        &self,
        dir: &Path,
        only: Option<&str>,
        mapped: &[(String, StoredBytes)],
    ) -> Result<(), LoadError> {
        let Some(files) = &self.files else {
            return Ok(());
        };
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
        for name in &required {
            let pin = *files
                .get(name)
                .ok_or_else(|| LoadError::Unpinned { file: name.clone() })?;
            match mapped.iter().find(|(file, _)| file == name) {
                Some((_, bytes)) => check_pin_reader(bytes.as_slice(), pin, name)?,
                None => {
                    let path = dir.join(name);
                    let file = File::open(&path).map_err(io_error(&path))?;
                    check_pin_reader(file, pin, name)?;
                }
            }
        }
        Ok(())
    }
}

/// A model, its weights and its text services, and for a vision-language model its towers and the
/// front end that runs them.
#[derive(Debug)]
pub struct ModelHandle {
    model: Box<dyn Model>,
    store: Arc<WeightStore>,
    /// The checkpoint's text services; a tower's handle shares its model's, and never tokenizes.
    text: Arc<TextCodec>,
    towers: Vec<(TowerId, Arc<ModelHandle>)>,
    front_end: Option<Arc<dyn FrontEnd>>,
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

/// `bytes` of the file `path` as JSON with unique keys.
fn parse_json(path: &Path, bytes: &[u8]) -> Result<serde_json::Value, LoadError> {
    poot_load::packed_safetensors::parse_json_unique_keys(bytes).map_err(|source| LoadError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// The JSON file `dir/name`, read once through `pins`; `None` when the checkpoint does not ship it.
fn read_optional_json(
    pins: &Pins,
    dir: &Path,
    name: &str,
) -> Result<Option<serde_json::Value>, LoadError> {
    pins.read_optional(dir, name)?
        .map(|bytes| parse_json(&dir.join(name), &bytes))
        .transpose()
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
    /// ships one, `preprocessor_config.json` when it ships one, safetensors and `tokenizer.json`), a file is a GGUF. The family is resolved from the
    /// config before any weight is read, so an unregistered family is refused without reading the
    /// checkpoint's tensors; the one resolve then builds the model.
    pub fn load(
        path: &Path,
        registry: &Registry,
        options: &LoadOptions,
    ) -> Result<Self, DriverError> {
        // The override is checked before any checkpoint file is read: a template poot refuses costs no load.
        let template = options
            .chat_template
            .as_ref()
            .map(ChatTemplateOverride::check)
            .transpose()
            .map_err(LoadError::from)?;
        if path.is_dir() {
            Self::load_hf(path, registry, options, template)
        } else {
            Self::load_gguf(path, registry, options, template)
        }
    }

    fn load_hf(
        dir: &Path,
        registry: &Registry,
        options: &LoadOptions,
        template: Option<CheckedChatTemplate>,
    ) -> Result<Self, DriverError> {
        let pins = Pins::load(options)?;
        let config = parse_json(&dir.join("config.json"), &pins.read(dir, "config.json")?)?;
        let generation = read_optional_json(&pins, dir, "generation_config.json")?;
        let preprocessor = read_optional_json(&pins, dir, "preprocessor_config.json")?;
        let raw = RawConfig::HfJson {
            config: &config,
            generation: generation.as_ref(),
            preprocessor: preprocessor.as_ref(),
        };
        let family = resolve(&raw, registry)?;
        let files = poot_load::safetensors::map_weight_files(dir, |name| pins.read(dir, name))?;
        pins.check_weights(dir, None, &files)?;
        // A GPTQ, AWQ or FP8 checkpoint's quantized linears are packed straight from their stored
        // bytes, never decoded: the family maps the packed entries like any other weight.
        let scheme = poot_load::config::quant_scheme(&config).map_err(LoadError::from)?;
        let store = poot_load::safetensors::load_mapped_weight_store(&files, default_workers())
            .map_err(LoadError::from)?;
        let store = match scheme {
            Some(scheme) => poot_load::safetensors::pack_quantized_linears(&store, scheme)
                .map_err(LoadError::from)?,
            None => store,
        };
        let text_files = HfTextFiles {
            tokenizer: pins.read(dir, "tokenizer.json")?,
            tokenizer_config: pins.read_optional(dir, "tokenizer_config.json")?,
            chat_template_jinja: pins
                .read_optional(dir, "chat_template.jinja")?
                .map(String::from_utf8)
                .transpose()
                .map_err(|e| {
                    LoadError::Text(Box::new(TextError::Msg(format!(
                        "chat_template.jinja: {e}"
                    ))))
                })?,
        };
        Self::build(family, &raw, Arc::new(store), |config| {
            let text = TextCodec::from_hf_files(
                &text_files,
                constraint_eos(config),
                config.prompt_bos,
                config.chat,
            )
            .map_err(|e| LoadError::Text(Box::new(e)))?
            .with_tool_codec(config.tool_codec.clone());
            Ok(match template {
                Some(template) => text.with_chat_template(template),
                None => text,
            })
        })
    }

    fn load_gguf(
        path: &Path,
        registry: &Registry,
        options: &LoadOptions,
        template: Option<CheckedChatTemplate>,
    ) -> Result<Self, DriverError> {
        // Naming any shard of a split checkpoint names the set; each shard is mapped once and pinned
        // on its own.
        let pins = Pins::load(options)?;
        let mut files = Vec::new();
        for shard in shard_paths(path) {
            let file = File::open(&shard).map_err(io_error(&shard))?;
            let file = StoredBytes::map(file)
                .map_err(poot_load::LoadError::from)
                .map_err(LoadError::from)?;
            if let Some(name) = shard.file_name().and_then(|n| n.to_str()) {
                pins.check_weights(
                    shard.parent().unwrap_or(Path::new("")),
                    Some(name),
                    &[(name.to_string(), file.clone())],
                )?;
            }
            files.push(file);
        }
        let index = GgufIndex::parse_shards(&files).map_err(LoadError::from)?;
        let raw = RawConfig::Gguf(&index);
        let family = resolve(&raw, registry)?;
        let store =
            read_gguf(&index, &IdentityNames, default_workers()).map_err(LoadError::from)?;
        Self::build(family, &raw, Arc::new(store), |config| {
            let text =
                TextCodec::from_gguf(&gguf_vocab(&index), constraint_eos(config), config.chat)
                    .map_err(|e| LoadError::Text(Box::new(e)))?
                    .with_tool_codec(config.tool_codec.clone());
            Ok(match template {
                Some(template) => text.with_chat_template(template),
                None => text,
            })
        })
    }

    /// Build `family`'s model container over `store`, then its text services from the served model's own
    /// config. The served model reads the container's section of `store` when it names one; every tower
    /// reads the checkpoint's own names.
    fn build(
        family: &FamilyEntry,
        raw: &RawConfig<'_>,
        store: Arc<WeightStore>,
        text: impl FnOnce(&ModelConfig) -> Result<TextCodec, LoadError>,
    ) -> Result<Self, DriverError> {
        let ModelParts {
            model,
            store: served,
            towers,
            front_end,
        } = (family.build)(raw, &store).map_err(DriverError::Model)?;
        let served = served.map_or_else(|| Arc::clone(&store), Arc::new);
        let report = model.weights().report(&served);
        tracing::info!(
            family = %model.config().family,
            loaded = report.loaded.len(),
            known = report.known.len(),
            unknown = report.unknown.len(),
            "checkpoint tensors"
        );
        let text = Arc::new(text(model.config())?);
        let towers = towers
            .into_iter()
            .map(|tower| {
                let handle = Self {
                    model: tower.model,
                    store: Arc::clone(&store),
                    text: Arc::clone(&text),
                    towers: Vec::new(),
                    front_end: None,
                };
                (tower.id, Arc::new(handle))
            })
            .collect();
        Ok(Self {
            model,
            store: served,
            text,
            towers,
            front_end: front_end.map(Arc::from),
        })
    }

    /// What the model's programs read, carry and require.
    pub fn schema(&self) -> &poot_models::io::ModelSchema {
        self.model.schema()
    }

    /// The request inputs the model's `step` program binds only on the steps that carry them (a sparse
    /// per-token or per-row input): what a deployment that serves them passes to
    /// [`Driver::prepare_carried`](crate::driver::Driver::prepare_carried).
    pub fn carriable_inputs(&self) -> Vec<poot_models::io::InputId> {
        let schema = self.model.schema();
        let Some(program) = schema.step() else {
            return Vec::new();
        };
        let carriable = schema.carriable(program);
        let mut ids = schema.ids_of(carriable);
        ids.sort();
        ids
    }

    /// The model's configuration.
    pub fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    /// The tokenizer and chat template built from the checkpoint.
    pub fn text(&self) -> &TextCodec {
        &self.text
    }

    /// The model's front end, when it takes media (a vision-language family): the thing that turns a
    /// conversation with images into a prompt and its request inputs.
    pub fn front_end(&self) -> Option<&Arc<dyn FrontEnd>> {
        self.front_end.as_ref()
    }

    /// The towers the front end runs, each as a handle a [`Driver`](crate::driver::Driver) is built over.
    pub fn towers(&self) -> &[(TowerId, Arc<ModelHandle>)] {
        &self.towers
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

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    impl ModelHandle {
        /// Build the family `raw` names over an in-memory `store` (a test fixture), then its text services
        /// from the model's own config. An unregistered family is [`Unsupported::Registry`], a checkpoint
        /// the family cannot take a [`DriverError::Model`].
        pub(crate) fn from_checkpoint(
            raw: &RawConfig<'_>,
            store: WeightStore,
            registry: &Registry,
            text: impl FnOnce(&ModelConfig) -> Result<TextCodec, LoadError>,
        ) -> Result<Self, DriverError> {
            Self::build(resolve(raw, registry)?, raw, Arc::new(store), text)
        }
    }
}

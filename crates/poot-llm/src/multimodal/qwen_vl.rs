//! The Qwen2.5-VL front end on the driver: a chat or rendered request as driver step inputs.
//!
//! A Qwen2.5-VL checkpoint is two registered families over one set of weights. The vision tower
//! (`qwen2_5_vl_vision`: the 3-D patch embedding, the windowed blocks and the patch merger) is an
//! `Encode`-only model run through its own [`Driver::encode`]; the text tower is the `qwen2_5_vl`
//! family, qwen2 with sectioned RoPE. [`Qwen25Vl`] assembles the request (the chat template, the
//! processor's patches and grids, the prompt's tokens and sectioned positions: the held front end of
//! [`qwen_vl_chat`](super::qwen_vl_chat) and [`qwen_vl_image`](super::qwen_vl_image)), encodes every
//! image and video through the vision driver into the merged rows the processor's placeholders stand
//! for, and generates on the text driver with those rows as step inputs
//! (`StepInput::TokensWithEmbeds`) and the prompt's positions bound to `Slot::MropePosition`
//! ([`Driver::generate_with_mrope`]): the pixels, the rows and the positions are bound by value on
//! every step, never baked into a program.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use poot_executor::Executor;
use poot_models::qwen_vl_vision::{GridError, MAX_PATCHES, Qwen25VlVisionParams};
use poot_models::registry::{RawConfig, Registry};
use poot_tensor::HostTensor;

use super::qwen_vl_chat::{
    Qwen25VlChatContent, Qwen25VlChatMessage, Qwen25VlChatMessageContent, Qwen25VlChatRequestError,
    Qwen25VlChatRequestMropeModelInputRequest, assemble_qwen2_5_vl_chat_request_mrope_model_input,
};
use super::qwen_vl_image::{
    Qwen25VlApngFrameSource, Qwen25VlDecodedFrameSource, Qwen25VlRawRequestAssemblyError,
    Qwen25VlRawRequestMropeModelInput, decode_qwen2_5_vl_image,
};
use super::qwen_vl_mrope::Qwen25VlMropeModelInputView;
use super::{
    ChatTurn, ImageCheckpoint, ImageCheckpointError, ImageFrontEnd, ImageRequestError,
    MultimodalPrompt,
};
use crate::core::sampler::Sampler;
use crate::driver::error::{DriverError, InvalidRequest, LoadError};
use crate::driver::{
    Driver, DriverOptions, EncodeRequest, EncodeRow, GenerateRequest, Generation, LoadOptions,
    ModelHandle, PromptEmbeds, TokenSink,
};
use poot_models::mrope::{MropePositionIds, MropeProcessorFpsInput, MropeVisualKind};
use poot_text::TextCodec;

/// Why a Qwen2.5-VL checkpoint cannot load or a request cannot run.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlError {
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// `config.json` has no `vision_config` object.
    #[error("{path}: vision_config is missing or not an object")]
    NoVisionConfig { path: PathBuf },
    #[error("image or video grid {grid:?}: {source}")]
    Grid {
        grid: [usize; 3],
        #[source]
        source: GridError,
    },
    /// The processor's patches are not `patch_dim` values wide, or do not cover the grids.
    #[error(
        "the processed {kind} patches do not match their grids and the tower's {patch_dim}-value patch"
    )]
    Patches {
        kind: MropeVisualKind,
        patch_dim: usize,
    },
    /// The tower returned a different number of rows than the prompt has placeholders.
    #[error(
        "the prompt holds {placeholders} image and video placeholders but the media give {rows} rows"
    )]
    Placeholders { placeholders: usize, rows: usize },
    #[error(transparent)]
    Driver(#[from] DriverError),
}

/// Why a chat request cannot run: it does not assemble, or what it assembled does not.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Assemble(Qwen25VlChatRequestError<E>),
    #[error(transparent)]
    Run(#[from] Qwen25VlError),
}

/// What to generate from an assembled request.
pub struct Qwen25VlGenerate {
    pub max_new: usize,
    pub sampler: Sampler,
}

/// A Qwen2.5-VL checkpoint read from its directory: the two towers' handles over the one set of
/// weights. Nothing is on a device yet; the image side ([`Qwen25VlFrontEnd`]) and the text side (a
/// [`Driver`] over [`Qwen25VlCheckpoint::text`]) are opened from it.
pub struct Qwen25VlCheckpoint {
    dir: PathBuf,
    vision: Arc<ModelHandle>,
    text: Arc<ModelHandle>,
    params: Qwen25VlVisionParams,
}

/// Whether `dir`'s `config.json` declares a Qwen2.5-VL checkpoint with a vision tower (a
/// `vision_config`). Any other directory, a file, and a text-only `qwen2_5_vl` config are not one; a
/// config that cannot be read is the loader's error to report, not this question's.
pub fn is_qwen2_5_vl_dir(dir: &Path) -> bool {
    read_config(&dir.join("config.json")).is_ok_and(|config| {
        config.get("model_type").and_then(serde_json::Value::as_str) == Some("qwen2_5_vl")
            && config.get("vision_config").is_some_and(|v| v.is_object())
    })
}

/// The image side of a Qwen2.5-VL: the vision tower on its own driver, and the request assembly
/// (chat template, processor patches and grids, sectioned positions) whose placeholders the tower's
/// rows replace. It produces what the text driver takes ([`MultimodalPrompt`]); it never generates.
pub struct Qwen25VlFrontEnd {
    dir: PathBuf,
    vision: Driver,
    params: Qwen25VlVisionParams,
}

/// A Qwen2.5-VL request lowered to what the text driver takes: unlike [`MultimodalPrompt`], the
/// sectioned positions are not optional.
#[derive(Clone, Debug)]
pub struct Qwen25VlPrompt {
    pub prompt: Vec<u32>,
    pub embeds: PromptEmbeds,
    /// Where every prompt token rotates (`Slot::MropePosition`).
    pub mrope: MropePositionIds,
}

impl From<Qwen25VlPrompt> for MultimodalPrompt {
    fn from(prompt: Qwen25VlPrompt) -> Self {
        Self {
            prompt: prompt.prompt,
            embeds: prompt.embeds,
            mrope: Some(prompt.mrope),
        }
    }
}

/// A loaded Qwen2.5-VL: the image side and the text tower, each on its own driver.
pub struct Qwen25Vl {
    images: Qwen25VlFrontEnd,
    text: Driver,
}

fn read_config(path: &Path) -> Result<serde_json::Value, Qwen25VlError> {
    let bytes = std::fs::read(path).map_err(|source| Qwen25VlError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    poot_load::packed_safetensors::parse_json_unique_keys(&bytes).map_err(|source| {
        Qwen25VlError::Json {
            path: path.to_path_buf(),
            source,
        }
    })
}

/// The processor's flattened patches of one kind (none when the request has none of that kind),
/// checked to be `patch_dim` values wide.
fn patch_values(
    kind: MropeVisualKind,
    tensor: Option<&HostTensor>,
    patch_dim: usize,
) -> Result<&[f32], Qwen25VlError> {
    let Some(tensor) = tensor else {
        return Ok(&[]);
    };
    tensor
        .as_f32()
        .filter(|_| tensor.shape().get(1) == Some(&patch_dim))
        .ok_or(Qwen25VlError::Patches { kind, patch_dim })
}

impl Qwen25VlCheckpoint {
    /// Read the Qwen2.5-VL checkpoint directory `dir`: the text tower from its top-level config and
    /// the vision tower from its `vision_config`, both through `registry` over the one set of
    /// weights.
    pub fn read(dir: &Path, registry: &Registry) -> Result<Self, Qwen25VlError> {
        let config_path = dir.join("config.json");
        let config = read_config(&config_path)?;
        let mut vision_config = config
            .get("vision_config")
            .filter(|v| v.is_object())
            .ok_or_else(|| Qwen25VlError::NoVisionConfig {
                path: config_path.clone(),
            })?
            .clone();
        vision_config["model_type"] = serde_json::json!("qwen2_5_vl_vision");
        let text = Arc::new(ModelHandle::load(dir, registry, &LoadOptions::default())?);
        let vision_raw = RawConfig::HfJson {
            config: &vision_config,
            generation: None,
        };
        let params = Qwen25VlVisionParams::from_raw(&vision_raw)
            .map_err(|e| Qwen25VlError::Driver(DriverError::Model(e)))?;
        let vision = Arc::new(ModelHandle::from_shared(
            &vision_raw,
            Arc::clone(text.shared_store()),
            registry,
            |config| {
                TextCodec::from_hf_dir(
                    dir,
                    config.eos.first().copied().unwrap_or(u32::MAX),
                    None,
                    config.chat,
                )
                .map_err(|e| LoadError::Text(Box::new(e)))
            },
        )?);
        Ok(Self {
            dir: dir.to_path_buf(),
            vision,
            text,
            params,
        })
    }

    /// Open the image side on `vision`'s executor and driver options (the capacity bounds the
    /// patches of one image). Inherent, not only the trait's, because [`Qwen25Vl`] holds the concrete type.
    pub fn front_end(
        &self,
        vision: (Box<dyn Executor>, DriverOptions),
    ) -> Result<Qwen25VlFrontEnd, Qwen25VlError> {
        Ok(Qwen25VlFrontEnd {
            dir: self.dir.clone(),
            vision: Driver::new(Arc::clone(&self.vision), vision.0, vision.1)?,
            params: self.params.clone(),
        })
    }
}

impl Qwen25VlFrontEnd {
    /// The vision tower's driver, for its stats and caps.
    pub fn vision(&self) -> &Driver {
        &self.vision
    }

    /// Assemble the chat `messages` (the checkpoint's chat template, the media decoded and
    /// processed) into a request.
    pub fn assemble_chat<S: Qwen25VlDecodedFrameSource>(
        &self,
        messages: &[Qwen25VlChatMessage<'_>],
        fps: Option<MropeProcessorFpsInput<'_>>,
        video_sources: &mut [S],
    ) -> Result<Qwen25VlRawRequestMropeModelInput, Qwen25VlChatRequestError<S::Error>> {
        assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &self.dir,
                messages,
                fps,
                video_sources,
            },
        )
    }

    /// Encode every image and video of `input` into the rows its placeholders stand for, in prompt
    /// order: `out_width` values per merged cell of every item.
    pub fn encode_media(
        &mut self,
        input: &Qwen25VlRawRequestMropeModelInput,
    ) -> Result<Vec<f32>, Qwen25VlError> {
        let media = input.processor_media();
        let patch_dim = self.params.patch_dim();
        let images = patch_values(MropeVisualKind::Image, media.pixel_values, patch_dim)?;
        let videos = patch_values(MropeVisualKind::Video, media.pixel_values_videos, patch_dim)?;
        let (mut next_image, mut next_video) = (0, 0);
        let (mut image_at, mut video_at) = (0, 0);
        let mut rows = Vec::new();
        for &kind in input.media_order() {
            let (grids, next, at, pixels) = match kind {
                MropeVisualKind::Image => {
                    (media.image_grids, &mut next_image, &mut image_at, images)
                }
                MropeVisualKind::Video => {
                    (media.video_grids, &mut next_video, &mut video_at, videos)
                }
            };
            let grid = grids[*next];
            *next += 1;
            let grid = [grid.temporal, grid.height, grid.width];
            let geometry = self
                .params
                .geometry(grid)
                .map_err(|source| Qwen25VlError::Grid { grid, source })?;
            let end = *at + geometry.patches() * patch_dim;
            let pixels = pixels
                .get(*at..end)
                .ok_or(Qwen25VlError::Patches { kind, patch_dim })?;
            *at = end;
            let ordered = geometry.to_window_order(pixels, patch_dim);
            let encoded = self.vision.encode(&EncodeRequest {
                rows: &[EncodeRow::Patches {
                    pixels: &ordered,
                    geometry: &geometry,
                }],
            })?;
            let item = encoded
                .rows
                .into_iter()
                .next()
                .expect("one item in, one row out");
            rows.extend(geometry.from_window_order(&item, self.params.out_width()));
        }
        Ok(rows)
    }

    /// The prompt tokens, embedding rows and sectioned positions of an assembled request: encode its
    /// media and check that the rows cover the prompt's image and video placeholders.
    pub fn prepare_input(
        &mut self,
        input: &Qwen25VlRawRequestMropeModelInput,
    ) -> Result<Qwen25VlPrompt, Qwen25VlError> {
        let rows = self.encode_media(input)?;
        let ids = input.metadata().special_tokens;
        let prompt = input.prompt_token_ids().to_vec();
        let positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter(|&(_, &token)| token == ids.image || token == ids.video)
            .map(|(position, _)| position)
            .collect();
        let width = self.params.out_width();
        if positions.len() * width != rows.len() {
            return Err(Qwen25VlError::Placeholders {
                placeholders: positions.len(),
                rows: rows.len() / width,
            });
        }
        let embeds = PromptEmbeds::new(width, positions, rows)
            .map_err(|e| DriverError::from(InvalidRequest::Embeds(e)))?;
        Ok(Qwen25VlPrompt {
            prompt,
            embeds,
            mrope: input.mrope_positions().clone(),
        })
    }
}

impl ImageFrontEnd for Qwen25VlFrontEnd {
    /// The conversation `turns` rendered through the checkpoint's chat template, each turn's images
    /// first in its content.
    fn prepare(
        &mut self,
        images: &[&[u8]],
        turns: &[ChatTurn<'_>],
    ) -> Result<MultimodalPrompt, ImageRequestError> {
        if images.is_empty() {
            return Err(ImageRequestError::Refused(
                "a request needs at least one image".into(),
            ));
        }
        let in_turns: usize = turns.iter().map(|turn| turn.images).sum();
        if in_turns != images.len() {
            return Err(ImageRequestError::Refused(format!(
                "the turns hold {in_turns} images but the request has {}",
                images.len()
            )));
        }
        let decoded = images
            .iter()
            .map(|bytes| decode_qwen2_5_vl_image(bytes))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ImageRequestError::Refused(e.to_string()))?;
        let mut next = decoded.iter();
        let messages: Vec<_> = turns
            .iter()
            .map(|turn| {
                let mut parts: Vec<_> = next
                    .by_ref()
                    .take(turn.images)
                    .map(Qwen25VlChatContent::Image)
                    .collect();
                parts.push(Qwen25VlChatContent::Text(turn.text));
                Qwen25VlChatMessage {
                    role: turn.role,
                    content: Qwen25VlChatMessageContent::Parts(parts),
                }
            })
            .collect();
        let mut no_videos: [Qwen25VlApngFrameSource<'_>; 0] = [];
        let input = self
            .assemble_chat(&messages, None, &mut no_videos)
            .map_err(|error| {
                if request_shaped(&error) {
                    ImageRequestError::Refused(error.to_string())
                } else {
                    ImageRequestError::Failed(Box::new(error))
                }
            })?;
        let prompt = self.prepare_input(&input).map_err(|error| match error {
            // An image of more patches than the tower takes is refused by the tower's own bound or by the
            // driver's trace limit set to it.
            Qwen25VlError::Grid { .. } | Qwen25VlError::Driver(DriverError::Unsupported(_)) => {
                ImageRequestError::Refused(error.to_string())
            }
            other => ImageRequestError::Failed(Box::new(other)),
        })?;
        Ok(prompt.into())
    }
}

/// Whether a chat assembly failed on the request (its messages or its images) and not on the checkpoint
/// (its metadata, processor config or chat template).
fn request_shaped<E: std::error::Error + 'static>(error: &Qwen25VlChatRequestError<E>) -> bool {
    use Qwen25VlChatRequestError as Chat;
    use Qwen25VlRawRequestAssemblyError as Raw;
    match error {
        Chat::EmptyMessages
        | Chat::MissingRole { .. }
        | Chat::EmptyRole { .. }
        | Chat::EmptyImagePath { .. }
        | Chat::ImageFile { .. }
        | Chat::TooManyMediaItems { .. } => true,
        Chat::RawRequest(raw) => matches!(
            raw,
            Raw::TooManyMediaItems { .. }
                | Raw::ImagePreprocess { .. }
                | Raw::ImageBatchPreprocess(_)
                | Raw::MediaOrderMismatch { .. }
                | Raw::MediaCountMismatch { .. }
                | Raw::PadExpansion(_)
        ),
        Chat::Metadata(_)
        | Chat::UnsupportedModelType { .. }
        | Chat::MissingChatTemplate
        | Chat::ChatTemplate(_) => false,
    }
}

impl ImageCheckpoint for Qwen25VlCheckpoint {
    fn text(&self) -> &Arc<ModelHandle> {
        &self.text
    }

    /// The tower's own bound: an image of more patches is refused when it is traced.
    fn patches(&self) -> usize {
        MAX_PATCHES
    }

    fn front_end(
        &self,
        vision: (Box<dyn Executor>, DriverOptions),
    ) -> Result<Box<dyn ImageFrontEnd>, ImageCheckpointError> {
        Ok(Box::new(Qwen25VlCheckpoint::front_end(self, vision)?))
    }
}

impl Qwen25Vl {
    /// Load the Qwen2.5-VL checkpoint directory `dir` and open a driver for each tower. `vision` and
    /// `text` are the towers' executors and driver options (the vision tower's capacity bounds the
    /// patches of one image; the text tower's, the prompt and the generation).
    pub fn load(
        dir: &Path,
        registry: &Registry,
        vision: (Box<dyn Executor>, DriverOptions),
        text: (Box<dyn Executor>, DriverOptions),
    ) -> Result<Self, Qwen25VlError> {
        let checkpoint = Qwen25VlCheckpoint::read(dir, registry)?;
        Ok(Self {
            images: checkpoint.front_end(vision)?,
            text: Driver::new(Arc::clone(&checkpoint.text), text.0, text.1)?,
        })
    }

    /// The vision tower's driver, for its stats and caps.
    pub fn vision(&self) -> &Driver {
        self.images.vision()
    }

    /// The text tower's driver.
    pub fn text(&self) -> &Driver {
        &self.text
    }

    /// Assemble the chat `messages` and generate from them.
    pub fn generate_chat<S: Qwen25VlDecodedFrameSource>(
        &mut self,
        messages: &[Qwen25VlChatMessage<'_>],
        fps: Option<MropeProcessorFpsInput<'_>>,
        video_sources: &mut [S],
        generate: Qwen25VlGenerate,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, Qwen25VlChatError<S::Error>> {
        let input = self
            .images
            .assemble_chat(messages, fps, video_sources)
            .map_err(Qwen25VlChatError::Assemble)?;
        Ok(self.generate(&input, generate, sink)?)
    }

    /// Generate from an assembled request: prepare its prompt (the media encoded and scattered over
    /// the placeholders) and generate on the text driver at the prompt's sectioned positions.
    pub fn generate(
        &mut self,
        input: &Qwen25VlRawRequestMropeModelInput,
        generate: Qwen25VlGenerate,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, Qwen25VlError> {
        let Qwen25VlPrompt {
            prompt,
            embeds,
            mrope,
        } = self.images.prepare_input(input)?;
        Ok(self.text.generate_with_mrope(
            GenerateRequest {
                prompt,
                max_new: generate.max_new,
                sampler: generate.sampler,
                stops: Vec::new(),
                ignore_eos: false,
            },
            embeds,
            mrope,
            sink,
        )?)
    }
}

/// A written Qwen2.5-VL checkpoint of the shipped fixtures, and the CPU oracle of its two towers: the
/// rows the device rows (`driver::device_tests`) and the host rows compare against.
#[cfg(test)]
pub(crate) mod fixture {
    use std::collections::HashMap;
    use std::num::NonZeroUsize;

    use poot_eval::test_support::weight_map_oracle::eval_mapped;
    use poot_graph_ir::{Slot, SlotKey};
    use poot_models::model::{KvLayout, LogitRows, Model, Phase, StepShape};
    use poot_models::mrope::MropeVisualKind;
    use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
    use poot_tensor::HostTensor;
    use tokenizers::AddedToken;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    use super::*;
    use crate::multimodal::qwen_vl_chat::{Qwen25VlChatContent, Qwen25VlChatMessageContent};
    use crate::multimodal::qwen_vl_image::{Qwen25VlApngFrameSource, Qwen25VlDecodedImage};

    pub(crate) const VOCAB: usize = 48;
    pub(crate) const VISION_START: &str = "<|vision_start|>";
    pub(crate) const IMAGE_PAD: &str = "<|image_pad|>";
    pub(crate) const VIDEO_PAD: &str = "<|video_pad|>";
    pub(crate) const VISION_END: &str = "<|vision_end|>";
    pub(crate) const IM_END: &str = "<|im_end|>";
    pub(crate) const IMAGE_ID: u32 = 41;
    pub(crate) const VISION_START_ID: u32 = 40;
    pub(crate) const VIDEO_ID: u32 = 42;
    const CHAT_TEMPLATE: &str =
        "{% for message in messages %}{{ message.role }} {{ message.content }}\n{% endfor %}";

    /// A checkpoint directory removed on drop.
    pub(crate) struct Checkpoint {
        dir: poot_test_util::UniqueTempPath,
    }

    impl std::ops::Deref for Checkpoint {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.dir
        }
    }

    fn fixture_of(registry: &Registry, family: &str) -> poot_models::registry::Fixture {
        let entry = registry
            .entries()
            .iter()
            .find(|entry| entry.family.as_str() == family)
            .unwrap_or_else(|| panic!("{family} is a shipped family"));
        (entry.fixture)()
    }

    /// `store` with every tensor but the norm scales multiplied by `factor`: the fixtures' random
    /// weights are small enough that a tiny tower's logits barely depend on its inputs.
    fn scaled(store: &WeightStore, factor: f32) -> WeightStore {
        let mut out = WeightStore::builder();
        for (key, entry) in store.iter() {
            let WeightEntry::Dense(dense) = entry else {
                panic!("the fixture is dense")
            };
            let keep = key.as_str().contains("norm") || key.as_str().contains("ln_q");
            let bytes: Vec<u8> = if keep {
                dense.bytes().as_slice().to_vec()
            } else {
                dense
                    .bytes()
                    .as_slice()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|&pair| {
                        let value = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                        (((value * factor).to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect()
            };
            let dense =
                DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into()).unwrap();
            out.insert(key.clone(), WeightEntry::Dense(dense)).unwrap();
        }
        out.build()
    }

    /// The text and the vision fixtures as one checkpoint directory: `config.json` with the text
    /// tower on top and `vision_config` nested, one `model.safetensors` holding both towers'
    /// tensors (the weights multiplied by `factor`), the 48-word tokenizer, its chat template and the
    /// processor config of 2-pixel patches that merge 2 x 2.
    pub(crate) fn checkpoint(name: &str, factor: f32) -> Checkpoint {
        let registry = Registry::builtin().unwrap();
        let text = fixture_of(&registry, "qwen2_5_vl");
        let vision = fixture_of(&registry, "qwen2_5_vl_vision");
        let dir = poot_test_util::unique_temp_path(format!("poot_qwen25_vl_{name}"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut store = WeightStore::builder();
        for (key, entry) in scaled(&text.store, factor)
            .iter()
            .chain(scaled(&vision.store, factor).iter())
        {
            store.insert(key.clone(), entry.clone()).unwrap();
        }
        poot_quant::test_support::safetensors::write(
            &dir.join("model.safetensors"),
            &store.build(),
        );

        let mut config = text.config.clone();
        let mut vision_config = vision.config.clone();
        vision_config.as_object_mut().unwrap().remove("model_type");
        vision_config["tokens_per_second"] = 2.into();
        config["vision_config"] = vision_config;
        config["vision_start_token_id"] = VISION_START_ID.into();
        config["image_token_id"] = IMAGE_ID.into();
        config["video_token_id"] = VIDEO_ID.into();
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&config).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("preprocessor_config.json"),
            r#"{"do_resize":false,"min_pixels":1,"max_pixels":4096,"patch_size":2,
               "temporal_patch_size":2,"merge_size":2}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "vision_start_token": VISION_START,
                "image_token": IMAGE_PAD,
                "video_token": VIDEO_PAD,
                "chat_template": CHAT_TEMPLATE,
            }))
            .unwrap(),
        )
        .unwrap();
        let mut vocab: HashMap<String, u32> =
            (0..VOCAB as u32).map(|id| (format!("t{id}"), id)).collect();
        for (word, id) in [
            ("user", 2),
            ("hello", 4),
            (VISION_START, VISION_START_ID),
            (IMAGE_PAD, IMAGE_ID),
            (VIDEO_PAD, VIDEO_ID),
            (VISION_END, 43),
            (IM_END, 47),
        ] {
            vocab.retain(|_, v| *v != id);
            vocab.insert(word.to_string(), id);
        }
        let mut tokenizer = tokenizers::Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("t0".to_string())
                .build()
                .unwrap(),
        );
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.add_special_tokens(&[
            AddedToken::from(VISION_START, true),
            AddedToken::from(IMAGE_PAD, true),
            AddedToken::from(VIDEO_PAD, true),
            AddedToken::from(VISION_END, true),
            AddedToken::from(IM_END, true),
        ]);
        tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
        Checkpoint { dir }
    }

    /// A `height x width` RGB image whose pixels differ with `seed`.
    pub(crate) fn image(height: usize, width: usize, seed: u8) -> Qwen25VlDecodedImage {
        let rgb = (0..height * width * 3)
            .map(|i| {
                (i as u8)
                    .wrapping_mul(37)
                    .wrapping_add(seed.wrapping_mul(91))
            })
            .collect();
        Qwen25VlDecodedImage::from_rgb8(height, width, rgb).unwrap()
    }

    /// The chat of one user turn: `hello`, the image, `t1`.
    pub(crate) fn messages<'a>(image: &'a Qwen25VlDecodedImage) -> [Qwen25VlChatMessage<'a>; 1] {
        [Qwen25VlChatMessage {
            role: "user",
            content: Qwen25VlChatMessageContent::Parts(vec![
                Qwen25VlChatContent::Text("hello"),
                Qwen25VlChatContent::Image(image),
                Qwen25VlChatContent::Text("t1"),
            ]),
        }]
    }

    /// The request the front end assembles for `messages`.
    pub(crate) fn assemble(
        dir: &Path,
        messages: &[Qwen25VlChatMessage<'_>],
    ) -> Qwen25VlRawRequestMropeModelInput {
        let mut no_videos: [Qwen25VlApngFrameSource<'_>; 0] = [];
        assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: dir,
                messages,
                fps: None,
                video_sources: &mut no_videos,
            },
        )
        .unwrap()
    }

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    fn build(
        dir: &Path,
    ) -> (
        Box<dyn Model>,
        Box<dyn Model>,
        WeightStore,
        Qwen25VlVisionParams,
    ) {
        let registry = Registry::builtin().unwrap();
        let store = poot_load::safetensors::load_weight_store(dir).unwrap();
        let config = read_config(&dir.join("config.json")).unwrap();
        let mut vision_config = config["vision_config"].clone();
        vision_config["model_type"] = "qwen2_5_vl_vision".into();
        let vision_raw = RawConfig::HfJson {
            config: &vision_config,
            generation: None,
        };
        let params = Qwen25VlVisionParams::from_raw(&vision_raw).unwrap();
        let vision = registry.build(&vision_raw, &store).unwrap();
        let text = registry
            .build(
                &RawConfig::HfJson {
                    config: &config,
                    generation: None,
                },
                &store,
            )
            .unwrap();
        (vision, text, store, params)
    }

    /// The merged rows of every image of `input`, in prompt order, on the CPU oracle: each image
    /// through the vision tower's Encode graph with its grid geometry bound.
    pub(crate) fn oracle_rows(dir: &Path, input: &Qwen25VlRawRequestMropeModelInput) -> Vec<f32> {
        let (vision, _, store, params) = build(dir);
        let media = input.processor_media();
        let pixels = media.pixel_values.expect("images").as_f32().unwrap();
        let patch_dim = params.patch_dim();
        let mut rows = Vec::new();
        let mut at = 0;
        for (&kind, grid) in input.media_order().iter().zip(media.image_grids) {
            assert_eq!(kind, MropeVisualKind::Image, "the oracle rows cover images");
            let geometry = params
                .geometry([grid.temporal, grid.height, grid.width])
                .unwrap();
            let n = geometry.patches();
            let graph = vision
                .trace(
                    Phase::Encode,
                    StepShape {
                        rows: nz(1),
                        tokens: nz(n),
                        capacity: nz(n),
                        kv: KvLayout::Contiguous,
                        logits: LogitRows::All,
                        hidden: None,
                    },
                    poot_graph_ir::TraceLimits::for_tests(),
                )
                .unwrap();
            let ordered =
                geometry.to_window_order(&pixels[at * patch_dim..(at + n) * patch_dim], patch_dim);
            at += n;
            let activation = |tag| SlotKey::new(Slot::Activation, Some(tag));
            let slots = [
                (
                    activation("pixels"),
                    HostTensor::f32(vec![1, n, patch_dim], ordered),
                ),
                (
                    activation("pos_h"),
                    HostTensor::i32(vec![1, n], geometry.pos_h().to_vec()),
                ),
                (
                    activation("pos_w"),
                    HostTensor::i32(vec![1, n], geometry.pos_w().to_vec()),
                ),
                (
                    activation("window"),
                    HostTensor::f32(vec![1, n], geometry.window().to_vec()),
                ),
                (
                    activation("frame"),
                    HostTensor::f32(vec![1, n], geometry.frame().to_vec()),
                ),
            ];
            let (out, _) = eval_mapped(&graph, &store, vision.weights(), &slots, &[]);
            rows.extend(geometry.from_window_order(out.as_f32().unwrap(), params.out_width()));
        }
        rows
    }

    /// The log-probabilities (natural log softmax) of every vocabulary token after `input`'s prompt and
    /// the `generated` tokens that follow it, on the CPU oracle: the text tower's `PrefillEmbeds`
    /// graph over the whole sequence, with `rows` scattered over the placeholders, the prompt at its
    /// sectioned positions and each generated token one further than the last on every axis.
    pub(crate) fn oracle_logprobs(
        dir: &Path,
        input: &Qwen25VlRawRequestMropeModelInput,
        rows: &[f32],
        generated: &[u32],
    ) -> Vec<f32> {
        let (_, text, store, params) = build(dir);
        let prompt: Vec<u32> = input
            .prompt_token_ids()
            .iter()
            .chain(generated)
            .copied()
            .collect();
        let n = prompt.len();
        let ids = input.metadata().special_tokens;
        let width = params.out_width();
        let continuation = input.mrope_positions.next_text_position().temporal;
        let positions: Vec<i32> = input
            .mrope_positions
            .axes()
            .iter()
            .flat_map(|axis| {
                axis.iter()
                    .copied()
                    .chain((0..generated.len() as i32).map(|k| continuation + k))
            })
            .collect();
        let mut embeds = vec![0.0f32; n * width];
        let mut at = vec![0i32; n];
        let mut next = 0;
        for (i, &token) in prompt.iter().enumerate() {
            if token == ids.image || token == ids.video {
                embeds[i * width..(i + 1) * width]
                    .copy_from_slice(&rows[next * width..(next + 1) * width]);
                at[i] = 1;
                next += 1;
            }
        }
        assert_eq!(next * width, rows.len(), "one row per placeholder");
        let graph = text
            .trace(
                Phase::PrefillEmbeds,
                StepShape {
                    rows: nz(1),
                    tokens: nz(n),
                    capacity: nz(text.config().max_positions),
                    kv: KvLayout::Contiguous,
                    logits: LogitRows::Last,
                    hidden: None,
                },
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let state: Vec<HostTensor> = graph
            .state()
            .map(|(input, _)| HostTensor::zeros(graph.aval(input).shape.clone()))
            .collect();
        let activation = |tag| SlotKey::new(Slot::Activation, Some(tag));
        let slots = [
            (
                SlotKey::new(Slot::Token, None),
                HostTensor::i32(vec![1, n], prompt.iter().map(|&t| t as i32).collect()),
            ),
            (
                SlotKey::new(Slot::Pos, None),
                HostTensor::i32(vec![1, n], (0..n as i32).collect()),
            ),
            (
                SlotKey::new(Slot::MropePosition, None),
                HostTensor::i32(vec![3, n], positions),
            ),
            (
                activation("embeds"),
                HostTensor::f32(vec![1, n, width], embeds),
            ),
            (activation("embeds_at"), HostTensor::i32(vec![1, n], at)),
        ];
        let (logits, _) = eval_mapped(&graph, &store, text.weights(), &slots, &state);
        let logits = logits.as_f32().unwrap();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
        logits.iter().map(|&l| l - max - sum.ln()).collect()
    }
}

#[cfg(test)]
mod tests {
    use poot_models::mrope::MropeGrid;

    use super::fixture::*;
    use super::{
        Qwen25VlApngFrameSource, Qwen25VlChatMessage, Qwen25VlChatRequestError,
        Qwen25VlChatRequestMropeModelInputRequest,
        assemble_qwen2_5_vl_chat_request_mrope_model_input, request_shaped,
    };

    /// SC-002: the sectioned positions the front end produces for `hello <image> t1` equal the
    /// reference's, written out by hand from its rule: text sits at its index on all three axes; an
    /// image of merged grid `3 x 2` sits at one temporal position and spreads over rows and columns from
    /// the position after the text before it; the text after it continues from the largest position of
    /// the image plus one. Mutation: give the image's tokens collapsed positions in
    /// `append_grid`; the height and width axes of the image go red.
    #[test]
    fn an_image_prompts_positions_equal_the_independent_reference() {
        let dir = checkpoint("positions", 1.0);
        let image = image(12, 8, 1);
        let input = assemble(&dir, &messages(&image));
        assert_eq!(input.media.image_grids, [MropeGrid::new(1, 6, 4)]);
        // `user hello <start> <pad> x 6 <end> t1`: 3 text tokens, 6 image tokens, 2 text tokens.
        let mut want = vec![user_id(), 4, VISION_START_ID];
        want.extend([IMAGE_ID; 6]);
        want.extend([43, 1]);
        assert_eq!(input.prompt_token_ids, want);
        let [temporal, height, width] = input.mrope_positions.axes();
        assert_eq!(temporal, [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7]);
        assert_eq!(height, [0, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7]);
        assert_eq!(width, [0, 1, 2, 3, 4, 3, 4, 3, 4, 6, 7]);
        assert_eq!(input.mrope_positions.next_text_position().temporal, 8);
    }

    fn user_id() -> u32 {
        2
    }

    /// A chat assembly fails as the client's only when the request is at fault: no messages is the
    /// request's, a checkpoint without a chat template is the checkpoint's. Mutation: make
    /// `request_shaped` return true for `MissingChatTemplate`; the second assertion goes red.
    #[test]
    fn only_request_faults_of_a_chat_assembly_are_the_clients() {
        let dir = checkpoint("shaped", 1.0);
        let assemble = |messages: &[Qwen25VlChatMessage<'_>]| {
            let mut no_videos: [Qwen25VlApngFrameSource<'_>; 0] = [];
            assemble_qwen2_5_vl_chat_request_mrope_model_input(
                Qwen25VlChatRequestMropeModelInputRequest {
                    model_dir: &dir,
                    messages,
                    fps: None,
                    video_sources: &mut no_videos,
                },
            )
            .unwrap_err()
        };
        assert!(request_shaped(&assemble(&[])));
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json")).unwrap())
                .unwrap();
        config.as_object_mut().unwrap().remove("chat_template");
        std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
        let image = image(12, 8, 1);
        let error = assemble(&messages(&image));
        assert!(
            matches!(error, Qwen25VlChatRequestError::MissingChatTemplate),
            "{error:?}"
        );
        assert!(!request_shaped(&error));
    }

    /// A serving front end exists for a checkpoint with a vision tower and for no other: a text-only
    /// `qwen2_5_vl` config (no `vision_config`) is served as the text model it is. Mutation: drop the
    /// `vision_config` clause of `is_qwen2_5_vl_dir`; the text-only directory is claimed and the row goes
    /// red.
    #[test]
    fn only_a_checkpoint_with_a_vision_tower_has_an_image_front_end() {
        let dir = checkpoint("detect", 1.0);
        let registry = poot_models::registry::Registry::builtin().unwrap();
        assert!(
            crate::read_image_checkpoint(&dir, &registry)
                .unwrap()
                .is_some()
        );
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        config.as_object_mut().unwrap().remove("vision_config");
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        assert!(
            crate::read_image_checkpoint(&dir, &registry)
                .unwrap()
                .is_none()
        );
    }
}

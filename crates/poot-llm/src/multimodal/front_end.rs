//! The idefics3 / SmolVLM front end: the multimodal prompt as driver step inputs (POOT-739).
//!
//! A SmolVLM checkpoint is two registered families over one set of weights. The vision tower
//! (`idefics3_vision`: SigLIP and the connector) is an `Encode`-only model run through its own
//! [`Driver::encode`]; the text tower is the `llama` family parsed from the checkpoint's `text_config`.
//! [`Idefics3`] preprocesses the images into frames, encodes each frame into the embedding rows the
//! connector produces, builds the prompt around the `<image>` placeholders and generates on the text
//! driver with those rows as step inputs (`StepInput::TokensWithEmbeds`): the frames' pixels and the
//! rows are bound by value on every step, never baked into a program, so two images of the same size
//! share every entry and nothing is evicted between them.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use poot_models::registry::{RawConfig, Registry};
use poot_models::vision::{PreprocessConfig, idefics3_preprocess};
use poot_quant::weights::{WeightKey, WeightStore};

use crate::core::sampler::Sampler;
use crate::driver::error::{DriverError, LoadError};
use crate::driver::{
    Driver, DriverOptions, EncodeRequest, EncodeRow, GenerateRequest, Generation, ModelHandle,
    PromptEmbeds, TokenSink,
};
use crate::error::TextError;
use crate::text::tokenize::TextCodec;

/// Upper bound on a decoded image's pixel count, guarding against a decompression bomb (a tiny file
/// that decodes to huge dimensions, blowing up the `w*h*3` RGB buffer and its f32 expansion). About 32
/// megapixels is far above any real SmolVLM input. The `image` crate's default limits are not honored
/// by every decoder, so the header dimensions are checked directly.
const MAX_IMAGE_PIXELS: u64 = 32 * 1024 * 1024;

/// Upper bound on the idefics3 frames of one request. Each image tiles into 17 to 34 frames (about
/// 3 MB of pixels each), all held before the tower runs, and the image count is client-controlled, so
/// the per-image pixel cap does not bound it.
const MAX_FRAMES: usize = 512;

/// Why a multimodal request or checkpoint is refused.
#[derive(Debug, thiserror::Error)]
pub enum FrontEndError {
    #[error("decode image: {0}")]
    Image(#[from] image::ImageError),
    #[error("image {width}x{height} ({pixels} pixels) exceeds the {MAX_IMAGE_PIXELS}-pixel cap")]
    ImageTooLarge {
        width: u32,
        height: u32,
        pixels: u64,
    },
    #[error("a caption needs at least one image")]
    NoImage,
    #[error(
        "too many image frames ({frames}+); at most {MAX_FRAMES} (idefics3 tiles each image into 17 to 34 frames)"
    )]
    TooManyFrames { frames: usize },
    #[error("the prompt holds {placeholders} <image> tokens but the frames give {rows} rows")]
    Placeholders { placeholders: usize, rows: usize },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// A field of the checkpoint's config the front end reads is missing or malformed.
    #[error("{path}: {field} is missing or not {expected}")]
    Config {
        path: PathBuf,
        field: &'static str,
        expected: &'static str,
    },
    #[error(transparent)]
    Load(#[from] LoadError),
    #[error(transparent)]
    Driver(#[from] DriverError),
    #[error(transparent)]
    Text(#[from] TextError),
}

/// An image as interleaved RGB in `[0, 255]`, `height * width * 3` values.
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    pub rgb: &'a [f32],
    pub height: usize,
    pub width: usize,
}

/// Decode PNG or JPEG bytes: interleaved RGB `[0, 255]`, alpha dropped. The header's dimensions are
/// checked against the pixel cap before any pixel is decoded.
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage, FrontEndError> {
    let (width, height) = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(image::ImageError::IoError)?
        .into_dimensions()?;
    check_dimensions(width, height)?;
    let rgb = image::load_from_memory(bytes)?.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    Ok(DecodedImage {
        rgb: rgb.into_raw().into_iter().map(f32::from).collect(),
        height,
        width,
    })
}

/// A decoded image that owns its pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedImage {
    pub rgb: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

impl DecodedImage {
    pub fn as_image(&self) -> Image<'_> {
        Image {
            rgb: &self.rgb,
            height: self.height,
            width: self.width,
        }
    }
}

fn check_dimensions(width: u32, height: u32) -> Result<(), FrontEndError> {
    let pixels = u64::from(width) * u64::from(height);
    if pixels > MAX_IMAGE_PIXELS {
        return Err(FrontEndError::ImageTooLarge {
            width,
            height,
            pixels,
        });
    }
    Ok(())
}

/// The idefics3 image-placeholder block for one split image (the reference
/// `Idefics3Processor.replace_image_token`). `rows` and `cols` are the tile grid (0, 0 for an unsplit
/// image, a single global frame); `tokens` is the per-frame visual-token count. A split image is, per
/// tile in row-major order, `<fake_token_around_image><row_R_col_C>` and `tokens` `<image>`
/// placeholders, a newline after each row, then the global frame
/// `\n<fake_token_around_image><global-img>` with its placeholders and a closing fake token.
fn image_block(rows: usize, cols: usize, tokens: usize) -> String {
    // The idefics3 tokenizer only registers `<row_R_col_C>` markers up to 6x6, so a larger grid would
    // emit markers that shred to byte pieces; SmolVLM's geometry (2048 longest edge, 512 tiles) maxes
    // at 4x4.
    debug_assert!(
        rows <= 6 && cols <= 6,
        "tile grid {rows}x{cols} exceeds the 6x6 <row_R_col_C> markers the tokenizer registers"
    );
    let images = "<image>".repeat(tokens);
    if rows == 0 && cols == 0 {
        return format!("<fake_token_around_image><global-img>{images}<fake_token_around_image>");
    }
    let mut block = String::new();
    for row in 0..rows {
        for col in 0..cols {
            block.push_str(&format!(
                "<fake_token_around_image><row_{}_col_{}>{images}",
                row + 1,
                col + 1
            ));
        }
        block.push('\n');
    }
    block.push_str(&format!(
        "\n<fake_token_around_image><global-img>{images}<fake_token_around_image>"
    ));
    block
}

/// The chat turn of one or more images and a question: each image's block, in order, before the
/// question. `grids` is one `(rows, cols)` tile grid per image.
fn caption_prompt(question: &str, grids: &[(usize, usize)], tokens: usize) -> String {
    let blocks: String = grids
        .iter()
        .map(|&(rows, cols)| image_block(rows, cols, tokens))
        .collect();
    format!("<|im_start|>User:{blocks}{question}<end_of_utterance>\nAssistant:")
}

/// What the front end reads of a checkpoint's `config.json`.
struct CheckpointConfig {
    /// The vision tower's config: `vision_config` with the two top-level facts it reads.
    vision: serde_json::Value,
    /// The text tower's config: `text_config`, a `llama`.
    text: serde_json::Value,
    image_token: u32,
    /// Patches per frame and the connector's visual tokens per frame.
    patches: usize,
    visual_tokens: usize,
    image_size: usize,
}

fn read_json(path: &Path) -> Result<serde_json::Value, FrontEndError> {
    let bytes = std::fs::read(path).map_err(|source| FrontEndError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| FrontEndError::Json {
        path: path.to_path_buf(),
        source,
    })
}

fn config_field<'a>(
    config: &'a serde_json::Value,
    path: &Path,
    field: &'static str,
    expected: &'static str,
) -> Result<&'a serde_json::Value, FrontEndError> {
    config
        .get(field)
        .filter(|value| !value.is_null())
        .ok_or_else(|| FrontEndError::Config {
            path: path.to_path_buf(),
            field,
            expected,
        })
}

fn config_count(
    config: &serde_json::Value,
    path: &Path,
    field: &'static str,
) -> Result<usize, FrontEndError> {
    config_field(config, path, field, "a count")?
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| FrontEndError::Config {
            path: path.to_path_buf(),
            field,
            expected: "a positive count",
        })
}

impl CheckpointConfig {
    fn read(dir: &Path) -> Result<Self, FrontEndError> {
        let path = dir.join("config.json");
        let config = read_json(&path)?;
        let mut vision = config_field(&config, &path, "vision_config", "an object")?.clone();
        let text = config_field(&config, &path, "text_config", "an object")?.clone();
        let scale = config_count(&config, &path, "scale_factor")?;
        let image_token =
            u32::try_from(config_count(&config, &path, "image_token_id")?).map_err(|_| {
                FrontEndError::Config {
                    path: path.clone(),
                    field: "image_token_id",
                    expected: "a token id",
                }
            })?;
        let text_width = config_count(&text, &path, "hidden_size")?;
        let image_size = config_count(&vision, &path, "image_size")?;
        let patch = config_count(&vision, &path, "patch_size")?;
        let grid = image_size / patch;
        // The family reads these two facts from its own config object; they live at the top level.
        vision["model_type"] = serde_json::json!("idefics3_vision");
        vision["scale_factor"] = serde_json::json!(scale);
        vision["text_hidden_size"] = serde_json::json!(text_width);
        Ok(Self {
            vision,
            text,
            image_token,
            patches: grid * grid,
            visual_tokens: grid * grid / (scale * scale),
            image_size,
        })
    }
}

/// `preprocessor_config.json`'s parameters of the idefics3 splitting pipeline.
fn read_preprocess(dir: &Path, image_size: usize) -> Result<PreprocessConfig, FrontEndError> {
    let path = dir.join("preprocessor_config.json");
    let config = read_json(&path)?;
    let size = config_field(&config, &path, "size", "an object")?;
    let longest_edge = config_count(size, &path, "longest_edge")?;
    let floats = |field: &'static str| -> Result<[f32; 3], FrontEndError> {
        let wrong = || FrontEndError::Config {
            path: path.clone(),
            field,
            expected: "three numbers",
        };
        let values = config_field(&config, &path, field, "three numbers")?
            .as_array()
            .ok_or_else(wrong)?;
        let mut out = [0.0; 3];
        if values.len() != 3 {
            return Err(wrong());
        }
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = value.as_f64().ok_or_else(wrong)? as f32;
        }
        Ok(out)
    };
    let rescale_factor = config_field(&config, &path, "rescale_factor", "a number")?
        .as_f64()
        .ok_or_else(|| FrontEndError::Config {
            path: path.clone(),
            field: "rescale_factor",
            expected: "a number",
        })? as f32;
    Ok(PreprocessConfig {
        image_size,
        longest_edge,
        rescale_factor,
        mean: floats("image_mean")?,
        std: floats("image_std")?,
    })
}

/// The entries of `store` under `prefix`, re-keyed under `to`, and the entries named in `keep`
/// unchanged: one section of a multimodal checkpoint as a store of its own. The entries share the
/// stored bytes; nothing is copied.
fn section(store: &WeightStore, prefix: &str, to: &str, keep: &[&str]) -> WeightStore {
    let mut out = WeightStore::builder();
    for (key, entry) in store.iter() {
        let name = key.as_str();
        let renamed = name
            .strip_prefix(prefix)
            .map(|rest| format!("{to}{rest}"))
            .or_else(|| keep.contains(&name).then(|| name.to_string()));
        if let Some(renamed) = renamed {
            out.insert(WeightKey::new(renamed), entry.clone())
                .expect("a section's keys are unique: its prefix is stripped uniformly");
        }
    }
    out.build()
}

/// One image to caption alongside the question that asks about it.
#[derive(Clone, Copy, Debug)]
pub struct CaptionRequest<'a> {
    pub images: &'a [Image<'a>],
    pub question: &'a str,
    pub max_new: usize,
}

/// A SmolVLM checkpoint read from its directory: the two towers' handles over the one set of weights,
/// and the facts of its image pipeline. Nothing is on a device yet; the image side
/// ([`ImageFrontEnd`]) and the text side (a [`Driver`] over [`Idefics3Checkpoint::text`]) are opened from it.
pub struct Idefics3Checkpoint {
    vision: Arc<ModelHandle>,
    text: Arc<ModelHandle>,
    preprocess: PreprocessConfig,
    image_token: u32,
    patches: usize,
    visual_tokens: usize,
}

/// Whether `dir`'s `config.json` declares an idefics3 (SmolVLM) checkpoint. Any other directory, and
/// a file, is not one; a config that cannot be read is the loader's error to report, not this question's.
pub fn is_idefics3_dir(dir: &Path) -> bool {
    read_json(&dir.join("config.json")).is_ok_and(|config| {
        config.get("model_type").and_then(serde_json::Value::as_str) == Some("idefics3")
    })
}

impl Idefics3Checkpoint {
    /// Read the SmolVLM / idefics3 checkpoint directory `dir`: parse its two towers through `registry`
    /// over its one set of weights.
    pub fn read(dir: &Path, registry: &Registry) -> Result<Self, FrontEndError> {
        let config = CheckpointConfig::read(dir)?;
        let preprocess = read_preprocess(dir, config.image_size)?;
        let store =
            Arc::new(poot_load::safetensors::load_weight_store(dir).map_err(LoadError::from)?);
        let codec = |config: &poot_models::model::ModelConfig| {
            TextCodec::from_hf_dir(
                dir,
                config.eos.first().copied().unwrap_or(u32::MAX),
                None,
                config.chat,
            )
            .map_err(|e| LoadError::Text(Box::new(e)))
        };
        let vision_raw = RawConfig::HfJson {
            config: &config.vision,
            generation: None,
        };
        let vision = Arc::new(ModelHandle::from_shared(
            &vision_raw,
            Arc::clone(&store),
            registry,
            codec,
        )?);
        let text_store = Arc::new(section(
            &store,
            "model.text_model.",
            "model.",
            &["lm_head.weight"],
        ));
        let text_raw = RawConfig::HfJson {
            config: &config.text,
            generation: None,
        };
        let text = Arc::new(ModelHandle::from_shared(
            &text_raw, text_store, registry, codec,
        )?);
        Ok(Self {
            vision,
            text,
            preprocess,
            image_token: config.image_token,
            patches: config.patches,
            visual_tokens: config.visual_tokens,
        })
    }

    /// The text tower: the model a server generates with, and the codec that tokenizes prompts.
    pub fn text(&self) -> &Arc<ModelHandle> {
        &self.text
    }

    /// The positions the vision tower's driver needs: the patches of one frame.
    pub fn patches(&self) -> usize {
        self.patches
    }

    /// Open the image side on `vision`'s executor: the vision tower's driver beside whatever drives
    /// the text tower.
    pub fn front_end(
        &self,
        vision: (Box<dyn poot_executor::Executor>, DriverOptions),
    ) -> Result<ImageFrontEnd, FrontEndError> {
        Ok(ImageFrontEnd {
            vision: Driver::new(Arc::clone(&self.vision), vision.0, vision.1)?,
            text: Arc::clone(&self.text),
            preprocess: self.preprocess,
            image_token: self.image_token,
            patches: self.patches,
            visual_tokens: self.visual_tokens,
        })
    }
}

/// The image side of a SmolVLM: preprocessing, the vision tower on its own driver, and the prompt
/// whose `<image>` placeholders the tower's rows replace. It produces what any text driver takes
/// (token ids and [`PromptEmbeds`]); it never generates.
pub struct ImageFrontEnd {
    vision: Driver,
    text: Arc<ModelHandle>,
    preprocess: PreprocessConfig,
    image_token: u32,
    patches: usize,
    visual_tokens: usize,
}

impl ImageFrontEnd {
    /// The vision tower's driver, for its stats and caps.
    pub fn vision(&self) -> &Driver {
        &self.vision
    }

    /// The embedding rows each image frame gives the prompt.
    pub fn visual_tokens(&self) -> usize {
        self.visual_tokens
    }

    /// Encode every frame of `frames` into its embedding rows (`visual_tokens * width` values per
    /// frame), one `Encode` step per frame: the tower's attention over a frame's patches is the
    /// step's memory, so frames do not batch.
    pub fn encode_frames(&mut self, frames: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, FrontEndError> {
        let mut rows = Vec::with_capacity(frames.len());
        for pixels in frames {
            let encoded = self.vision.encode(&EncodeRequest {
                rows: &[EncodeRow::Frame {
                    pixels,
                    positions: self.patches,
                }],
            })?;
            rows.push(
                encoded
                    .rows
                    .into_iter()
                    .next()
                    .expect("one frame in, one row out"),
            );
        }
        Ok(rows)
    }

    /// The prompt tokens and embedding rows for `images` and `question`: preprocess each image into
    /// idefics3 frames, encode the frames, and build the prompt whose `<image>` placeholders the
    /// frames' rows replace. The text driver serves it alone ([`Idefics3::caption`]) or beside other
    /// requests' prompts (`Driver::open_with_embeds` on a paged pool).
    pub fn prepare(
        &mut self,
        images: &[Image<'_>],
        question: &str,
    ) -> Result<CaptionPrompt, FrontEndError> {
        if images.is_empty() {
            return Err(FrontEndError::NoImage);
        }
        // Every image splits into its own frames (aspect-preserving tiles and a global thumbnail);
        // the prompt has one block per image, so its `<image>` count is the frames times the
        // visual tokens per frame.
        let mut frames: Vec<Vec<f32>> = Vec::new();
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let split = idefics3_preprocess(image.rgb, image.height, image.width, &self.preprocess);
            grids.push((split.rows, split.cols));
            frames.extend(split.frames);
            // Checked as frames accumulate, so the oversized batch is never materialized whole.
            if frames.len() > MAX_FRAMES {
                return Err(FrontEndError::TooManyFrames {
                    frames: frames.len(),
                });
            }
        }
        let prompt =
            self.text
                .text()
                .encode(&caption_prompt(question, &grids, self.visual_tokens))?;
        let positions: Vec<usize> = prompt
            .iter()
            .enumerate()
            .filter(|&(_, &token)| token == self.image_token)
            .map(|(position, _)| position)
            .collect();
        let rows = frames.len() * self.visual_tokens;
        if positions.len() != rows {
            return Err(FrontEndError::Placeholders {
                placeholders: positions.len(),
                rows,
            });
        }
        let encoded = self.encode_frames(&frames)?;
        let width = encoded[0].len() / self.visual_tokens;
        let embeds = PromptEmbeds::new(width, positions, encoded.concat())
            .map_err(|e| DriverError::from(crate::driver::error::InvalidRequest::Embeds(e)))?;
        Ok(CaptionPrompt { prompt, embeds })
    }
}

/// A loaded SmolVLM: the image side and the text tower, each on its own driver.
pub struct Idefics3 {
    pub images: ImageFrontEnd,
    pub text: Driver,
}

impl Idefics3 {
    /// Load the SmolVLM / idefics3 checkpoint directory `dir` and open a driver for each tower.
    /// `vision` and `text` are the towers' executors and driver options (the vision tower's capacity
    /// is its patch count).
    pub fn load(
        dir: &Path,
        registry: &Registry,
        vision: (Box<dyn poot_executor::Executor>, DriverOptions),
        text: (Box<dyn poot_executor::Executor>, DriverOptions),
    ) -> Result<Self, FrontEndError> {
        let checkpoint = Idefics3Checkpoint::read(dir, registry)?;
        Ok(Self {
            images: checkpoint.front_end(vision)?,
            text: Driver::new(Arc::clone(&checkpoint.text), text.0, text.1)?,
        })
    }

    /// Caption `request.images` (greedy): [`ImageFrontEnd::prepare`], then generate on the text
    /// driver with the frames' rows scattered over the prompt's `<image>` placeholders.
    pub fn caption(
        &mut self,
        request: &CaptionRequest<'_>,
        sink: &mut dyn TokenSink,
    ) -> Result<Generation, FrontEndError> {
        let CaptionPrompt { prompt, embeds } =
            self.images.prepare(request.images, request.question)?;
        Ok(self.text.generate_with_embeds(
            GenerateRequest {
                prompt,
                max_new: request.max_new,
                sampler: Sampler::greedy(),
                stops: Vec::new(),
                ignore_eos: false,
            },
            embeds,
            sink,
        )?)
    }
}

/// A caption request lowered to what the text driver takes: the prompt's token ids (placeholders
/// included) and the embedding rows that replace the placeholders.
#[derive(Clone, Debug)]
pub struct CaptionPrompt {
    pub prompt: Vec<u32>,
    pub embeds: PromptEmbeds,
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_models::vision::idefics3_preprocess;

    /// The split-image prompt emits `(rows * cols + 1) * tokens` `<image>` placeholders (one per
    /// visual token across all frames and the global one) and one `<row_R_col_C>` marker per tile, in
    /// row-major order; an unsplit image is one global frame without tile markers.
    #[test]
    fn a_tiled_prompt_has_the_right_placeholders_and_markers() {
        let tokens = 64usize;
        let (rows, cols) = (2usize, 4usize);
        let prompt = caption_prompt("Describe.", &[(rows, cols)], tokens);
        assert_eq!(
            prompt.matches("<image>").count(),
            (rows * cols + 1) * tokens
        );
        for row in 1..=rows {
            for col in 1..=cols {
                let marker = format!("<row_{row}_col_{col}>");
                assert_eq!(prompt.matches(&marker).count(), 1, "{marker}");
            }
        }
        assert_eq!(prompt.matches("<global-img>").count(), 1);
        assert!(prompt.ends_with("<end_of_utterance>\nAssistant:"));
        let single = caption_prompt("Describe.", &[(0, 0)], tokens);
        assert_eq!(single.matches("<image>").count(), tokens);
        assert_eq!(single.matches("<row_").count(), 0);
    }

    /// The splitting geometry and the prompt agree on the placeholder count for realistic image
    /// sizes, and a tile grid stays within the tokenizer's 6x6 markers: a disagreement would give the
    /// scatter the wrong number of rows.
    #[test]
    fn preprocessing_and_the_prompt_agree_on_the_placeholder_count() {
        let config = PreprocessConfig {
            image_size: 512,
            longest_edge: 2048,
            rescale_factor: 1.0 / 255.0,
            mean: [0.5; 3],
            std: [0.5; 3],
        };
        let tokens = 64usize;
        for (h, w) in [
            (64usize, 64usize),
            (512, 512),
            (96, 96),
            (800, 600),
            (1920, 1080),
            (100, 2000),
            (1, 1),
            (3, 4000),
        ] {
            let split = idefics3_preprocess(&vec![100.0; h * w * 3], h, w, &config);
            let (rows, cols, frames) = (split.rows, split.cols, split.frames.len());
            assert!(rows <= 6 && cols <= 6, "{h}x{w} -> {rows}x{cols}");
            let prompt = caption_prompt("Describe.", &[(rows, cols)], tokens);
            assert_eq!(
                prompt.matches("<image>").count(),
                frames * tokens,
                "{h}x{w}"
            );
        }
    }

    /// Several images each contribute their own block, so the placeholders sum over their frames.
    #[test]
    fn several_images_sum_their_placeholders() {
        let tokens = 64usize;
        let grids = [(0, 0), (2, 4), (3, 2)];
        let prompt = caption_prompt("Compare.", &grids, tokens);
        let frames: usize = grids
            .iter()
            .map(|&(r, c)| if r == 0 { 1 } else { r * c + 1 })
            .sum();
        assert_eq!(prompt.matches("<image>").count(), frames * tokens);
        assert_eq!(prompt.matches("<global-img>").count(), grids.len());
    }

    /// An image whose header declares more pixels than the cap is refused before it decodes.
    #[test]
    fn an_oversized_image_is_refused_by_its_header() {
        let mut png = Vec::new();
        image::RgbImage::new(1, 1)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        // Rewrite the IHDR width and height (8192 x 8192 is 64 megapixels) and the chunk's CRC-32, which
        // covers its type and data: the decoder checks it before it reports the header.
        png[16..20].copy_from_slice(&8192u32.to_be_bytes());
        png[20..24].copy_from_slice(&8192u32.to_be_bytes());
        let mut crc = u32::MAX;
        for &byte in &png[12..29] {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        png[29..33].copy_from_slice(&(!crc).to_be_bytes());
        assert!(matches!(
            decode_image(&png),
            Err(FrontEndError::ImageTooLarge {
                width: 8192,
                height: 8192,
                ..
            })
        ));
    }

    /// PNG bytes decode to their pixels, alpha dropped.
    #[test]
    fn a_png_decodes_to_its_pixels() {
        let (w, h) = (2u32, 3u32);
        let mut img = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb([(x * 100) as u8, (y * 50) as u8, 7]));
            }
        }
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let decoded = decode_image(&png).unwrap();
        assert_eq!((decoded.height, decoded.width), (3, 2));
        assert_eq!(&decoded.rgb[..3], &[0.0, 0.0, 7.0]);
        assert_eq!(&decoded.rgb[3..6], &[100.0, 0.0, 7.0]);
    }

    /// A section re-keys the entries under its prefix and keeps the named ones: the text tower's
    /// store reads `model.layers.*` out of `model.text_model.layers.*`, and holds nothing of the
    /// vision tower.
    #[test]
    fn a_section_rekeys_its_prefix_and_drops_the_rest() {
        let fx = poot_models::registry::Fixture::bf16(
            serde_json::json!({}),
            [
                ("model.text_model.norm.weight".to_string(), vec![2]),
                (
                    "model.vision_model.post_layernorm.weight".to_string(),
                    vec![2],
                ),
                ("lm_head.weight".to_string(), vec![2, 2]),
            ],
        );
        let text = section(
            &fx.store,
            "model.text_model.",
            "model.",
            &["lm_head.weight"],
        );
        let mut keys: Vec<&str> = text.keys().map(WeightKey::as_str).collect();
        keys.sort();
        assert_eq!(keys, ["lm_head.weight", "model.norm.weight"]);
    }
}

/// The real `smolvlm-256m` checkpoint through the front end on a device: rows of the card's
/// acceptance (SC-001, SC-002). They need the local checkpoint and a device, so they are ignored
/// (`just test` hides every checkpoint test) and run by exact name under `/tmp/poot-gpu.lock`.
#[cfg(test)]
mod smolvlm {
    use std::num::{NonZeroU64, NonZeroUsize};

    use poot_graph_plan::{CompileLimits, CompileOptions, Submission};
    use poot_runtime_common::DeviceBackend;

    use super::*;
    use crate::GenerationControl;
    use crate::driver::{BackendChoice, DraftPolicy, PreparedSetLimits, open_executor};

    fn options(capacity: usize, chunk: usize) -> DriverOptions {
        let compile = CompileOptions::new(Submission::Replay, CompileLimits::STANDARD);
        let nz = |n: usize| NonZeroUsize::new(n).unwrap();
        DriverOptions {
            prefill: compile.clone(),
            decode: compile,
            capacity: nz(capacity),
            prefill_chunk: nz(chunk),
            max_trace_tokens: nz(chunk),
            prepared: PreparedSetLimits {
                max_entries: nz(64),
                max_retained_bytes: NonZeroU64::new(1 << 32).unwrap(),
            },
            charge: crate::driver::program_retention,
            draft: DraftPolicy::Off,
        }
    }

    /// The checkpoint on `choice`'s device, or `None` when the checkpoint or the device is absent.
    fn load(choice: BackendChoice) -> Option<Idefics3> {
        let dir = poot_test_util::model_path(poot_test_util::checkpoint!("smolvlm-256m"))?;
        let backend = match choice {
            BackendChoice::Wgpu => DeviceBackend::Wgpu,
            BackendChoice::Rocm => DeviceBackend::Rocm,
            BackendChoice::Ptx => DeviceBackend::Ptx,
            other => unreachable!("{other:?} has no smolvlm row"),
        };
        let open = || poot_test_util::device_skip::open_or_skip(backend, open_executor(choice));
        let (vision, text) = (open()?, open()?);
        Some(
            Idefics3::load(
                &dir,
                &Registry::builtin().unwrap(),
                (vision, options(1024, 1024)),
                (text, options(4096, 256)),
            )
            .expect("load smolvlm-256m"),
        )
    }

    fn pattern(h: usize, w: usize) -> Vec<f32> {
        (0..h * w * 3).map(|i| ((i * 7) % 256) as f32).collect()
    }

    fn solid(h: usize, w: usize) -> Vec<f32> {
        vec![128.0; h * w * 3]
    }

    fn stripes(h: usize, w: usize) -> Vec<f32> {
        (0..h * w)
            .flat_map(|i| {
                let v = if (i / w / 8).is_multiple_of(2) {
                    0.0
                } else {
                    255.0
                };
                [v, v, v]
            })
            .collect()
    }

    fn caption(vlm: &mut Idefics3, rgb: &[f32], question: &str, max_new: usize) -> Vec<u32> {
        let image = Image {
            rgb,
            height: 96,
            width: 96,
        };
        vlm.caption(
            &CaptionRequest {
                images: &[image],
                question,
                max_new,
            },
            &mut |_: u32, _: &str| GenerationControl::Continue(()),
        )
        .expect("caption")
        .tokens
    }

    /// The tokens the pre-driver engine (base commit 01aa1d40f) generated for the 96x96 pattern image and this
    /// question on wgpu (12 greedy tokens: " The image displays a series of parallel, horizontal
    /// stripes that appear").
    const PATTERN_TOKENS: [u32; 12] = [
        378, 2443, 10637, 253, 3086, 282, 7623, 28, 10691, 23158, 338, 1972,
    ];

    /// SC-001: the SmolVLM fixture (an image and a prompt) generates through `Driver` with the front
    /// end producing the step inputs; the tokens equal the literals recorded from the pre-driver engine at the
    /// base commit (tier 3). Mutation: feed zeroed image embeddings; the tokens change and the row
    /// goes red.
    fn caption_matches_the_recorded_vlm_runner_tokens(choice: BackendChoice) {
        let Some(mut vlm) = load(choice) else {
            return;
        };
        assert_eq!(vlm.images.visual_tokens(), 64);
        let tokens = caption(
            &mut vlm,
            &pattern(96, 96),
            "Can you describe this image?",
            12,
        );
        assert_eq!(tokens, PATTERN_TOKENS);
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + wgpu (17 vision frames, ~1200-token prefill)"]
    fn caption_matches_the_recorded_vlm_runner_tokens_wgpu() {
        caption_matches_the_recorded_vlm_runner_tokens(BackendChoice::Wgpu);
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + ROCm (17 vision frames, ~1200-token prefill)"]
    fn caption_matches_the_recorded_vlm_runner_tokens_rocm() {
        caption_matches_the_recorded_vlm_runner_tokens(BackendChoice::Rocm);
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + a PTX device (17 vision frames, ~1200-token prefill)"]
    fn caption_matches_the_recorded_vlm_runner_tokens_ptx() {
        caption_matches_the_recorded_vlm_runner_tokens(BackendChoice::Ptx);
    }

    /// SC-002: two frames of equal numel and different content in one front end give different
    /// embedding rows and different captions, each the recorded pre-driver tokens, with no eviction
    /// between them: the frames are bound by value. Mutation: bind `pixels` as a const again; the
    /// second frame cannot load (or reuses the first) and the row goes red.
    fn two_captions_on_one_front_end_are_not_stale(choice: BackendChoice) {
        let Some(mut vlm) = load(choice) else {
            return;
        };
        let question = "Can you describe this image?";
        let first = caption(&mut vlm, &solid(96, 96), question, 12);
        let second = caption(&mut vlm, &stripes(96, 96), question, 12);
        assert_eq!(
            first,
            [
                378, 2443, 21559, 253, 2232, 28, 2408, 18029, 28, 15932, 11955, 282
            ]
        );
        assert_eq!(
            second,
            [
                378, 2443, 21559, 253, 2632, 284, 2537, 2443, 282, 253, 2244, 28
            ]
        );
        assert_ne!(first, second, "the second caption reused the first image");
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + wgpu (two captions)"]
    fn two_captions_on_one_front_end_are_not_stale_wgpu() {
        two_captions_on_one_front_end_are_not_stale(BackendChoice::Wgpu);
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + ROCm (two captions)"]
    fn two_captions_on_one_front_end_are_not_stale_rocm() {
        two_captions_on_one_front_end_are_not_stale(BackendChoice::Rocm);
    }

    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + a PTX device (two captions)"]
    fn two_captions_on_one_front_end_are_not_stale_ptx() {
        two_captions_on_one_front_end_are_not_stale(BackendChoice::Ptx);
    }

    /// Two images in one turn caption together: each splits into 17 frames, so 34 vision steps and a
    /// ~2200-token prompt (2 x 1088 image tokens) go through the one scatter.
    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + wgpu (34 vision frames, ~2200-token prefill)"]
    fn two_images_in_one_turn_caption_together() {
        let Some(mut vlm) = load(BackendChoice::Wgpu) else {
            return;
        };
        let (a, b) = (solid(96, 96), stripes(96, 96));
        let images = [
            Image {
                rgb: &a,
                height: 96,
                width: 96,
            },
            Image {
                rgb: &b,
                height: 96,
                width: 96,
            },
        ];
        let generation = vlm
            .caption(
                &CaptionRequest {
                    images: &images,
                    question: "Describe both images.",
                    max_new: 16,
                },
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .expect("caption");
        assert!(!generation.tokens.is_empty(), "produced an empty caption");
    }

    /// DoS guard: each image tiles into about 17 in-memory frames and the image count is
    /// client-controlled, so the total is capped and refused in preprocessing, before the tower runs.
    #[test]
    #[ignore = "needs the local smolvlm-256m checkpoint + wgpu (bails before any dispatch)"]
    fn too_many_images_are_refused_before_the_tower_runs() {
        let Some(mut vlm) = load(BackendChoice::Wgpu) else {
            return;
        };
        // 40 tiny images tile to about 17 frames each: over the 512-frame cap.
        let one: Vec<f32> = (0..4 * 4 * 3).map(|i| (i % 256) as f32).collect();
        let images = vec![
            Image {
                rgb: &one,
                height: 4,
                width: 4
            };
            40
        ];
        let error = vlm
            .caption(
                &CaptionRequest {
                    images: &images,
                    question: "describe",
                    max_new: 8,
                },
                &mut |_: u32, _: &str| GenerationControl::Continue(()),
            )
            .expect_err("an oversized batch is refused");
        assert!(
            matches!(error, FrontEndError::TooManyFrames { .. }),
            "{error}"
        );
        assert_eq!(
            vlm.images.vision().stats().steps,
            0,
            "no frame reached the tower"
        );
    }
}

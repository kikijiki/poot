//! Chat construction of the Qwen2.5-VL raw rendered request.
//!
//! Content-part order is the media order. Vision parts are flattened with existing processor markers, then
//! the checkpoint chat template is rendered through the existing minijinja path. Local PNG/JPEG paths
//! are decoded here. After render, the raw assembler expands each pad from merged-grid counts.
//! [`Qwen25Vl`](super::qwen_vl::Qwen25Vl) runs the request on the driver; codec/container video
//! decoding stays with the caller's frame source.

use std::fs;
use std::path::{Path, PathBuf};

use poot_load::qwen_vl::QwenVlModelType;
use poot_models::mrope::{
    MropeProcessorFpsInput, MropeProcessorSpecialToken, MropeSpecialTokenKind,
};

use super::qwen_vl_image::{
    QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS, Qwen25VlDecodedFrameSource, Qwen25VlDecodedImage,
    Qwen25VlImageDecodeError, Qwen25VlRawMediaInput, Qwen25VlRawRequestAssemblyError,
    Qwen25VlRawRequestMropeModelInput, Qwen25VlRawRequestMropeModelInputRequest,
    Qwen25VlVideoMetadata, assemble_qwen2_5_vl_raw_request_mrope_model_input,
};
use super::qwen_vl_mrope::{QwenVlMropeMetadataLoadError, load_qwen_vl_mrope_source_metadata};
use poot_text::{ChatFormat, TextCodec};

const VISION_END_DELIMITER: &str = "<|vision_end|>";
/// One typed Qwen2.5-VL chat message.
pub struct Qwen25VlChatMessage<'a> {
    /// The role the chat template renders (`"user"`, `"assistant"`, `"system"`); must not be empty.
    pub role: &'a str,
    /// The message's content, text or ordered typed parts.
    pub content: Qwen25VlChatMessageContent<'a>,
}

/// Message content: a plain string or ordered typed parts.
pub enum Qwen25VlChatMessageContent<'a> {
    /// The whole content as one string.
    Text(&'a str),
    /// Ordered parts; the order of the media parts is the media order of the request.
    Parts(Vec<Qwen25VlChatContent<'a>>),
}

/// One content part. Images are already decoded or a local PNG/JPEG path; videos are already probed.
#[derive(Clone, Copy)]
pub enum Qwen25VlChatContent<'a> {
    /// Literal text.
    Text(&'a str),
    /// An image already decoded.
    Image(&'a Qwen25VlDecodedImage),
    /// A local PNG or JPEG file, decoded when the request is assembled.
    ImagePath(&'a Path),
    /// A video already probed; its frames come from the caller's frame source.
    Video(Qwen25VlVideoMetadata<'a>),
}

/// Inputs for chat-template construction of the existing raw rendered request.
pub(crate) struct Qwen25VlChatRequestMropeModelInputRequest<'dir, 'msg, 'data, 'sources, S> {
    pub model_dir: &'dir Path,
    pub messages: &'msg [Qwen25VlChatMessage<'data>],
    pub fps: Option<MropeProcessorFpsInput<'data>>,
    pub video_sources: &'sources mut [S],
}

/// Why a chat request does not assemble into a raw rendered request.
/// `E` is the caller's video frame source error.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatRequestError<E: std::error::Error + 'static> {
    /// The request carried no message.
    #[error("Qwen2.5-VL chat request has no messages")]
    EmptyMessages,
    /// A message has no role.
    #[error("Qwen2.5-VL chat message {message_index} is missing a role")]
    MissingRole {
        /// Index of the message in the request.
        message_index: usize,
    },
    /// A message has an empty role string.
    #[error("Qwen2.5-VL chat message {message_index} has an empty role")]
    EmptyRole {
        /// Index of the message in the request.
        message_index: usize,
    },
    /// An image part names an empty path.
    #[error(
        "Qwen2.5-VL chat message {message_index} image part {part_index} has an empty image path"
    )]
    EmptyImagePath {
        /// Index of the message in the request.
        message_index: usize,
        /// Index of the part in that message's content.
        part_index: usize,
    },
    /// A local image file could not be read or decoded.
    #[error(
        "load Qwen2.5-VL chat image file {path:?} at message {message_index} part {part_index}"
    )]
    ImageFile {
        /// Index of the message in the request.
        message_index: usize,
        /// Index of the part in that message's content.
        part_index: usize,
        /// The path that failed.
        path: PathBuf,
        /// Whether the read or the decode failed.
        #[source]
        source: Qwen25VlChatImageFileError,
    },
    /// The request carries more media items than the raw assembler's cap.
    #[error("Qwen2.5-VL chat request has {actual} media items, exceeding the {limit}-item cap")]
    TooManyMediaItems {
        /// Media items (images and videos) in the request.
        actual: usize,
        /// The cap, `QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS`.
        limit: usize,
    },
    /// The checkpoint's processor and mrope metadata could not be loaded.
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    /// The checkpoint is a Qwen vision-language model of another type than Qwen2.5-VL.
    #[error("Qwen2.5-VL chat request does not support model type {model_type:?}")]
    UnsupportedModelType {
        /// The model type the checkpoint declares.
        model_type: QwenVlModelType,
    },
    /// The checkpoint's tokenizer config has no chat template.
    #[error("Qwen2.5-VL chat request is missing a chat template")]
    MissingChatTemplate,
    /// Rendering the chat template failed; carries the renderer's message.
    #[error("render Qwen2.5-VL chat template: {0}")]
    ChatTemplate(String),
    /// The rendered request did not assemble (media, markers, frames or grids).
    #[error(transparent)]
    RawRequest(Qwen25VlRawRequestAssemblyError<E>),
}

/// Why a local image file of a chat part could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatImageFileError {
    /// The file could not be read.
    #[error("read Qwen2.5-VL image file: {0}")]
    Read(#[from] std::io::Error),
    /// The file's bytes did not decode to an image.
    #[error(transparent)]
    Decode(#[from] Qwen25VlImageDecodeError),
}

enum ValidatedContent<'a> {
    Text(&'a str),
    Parts(Vec<Qwen25VlChatContent<'a>>),
}

struct ValidatedMessage<'a> {
    role: &'a str,
    content: ValidatedContent<'a>,
}

/// Construct the existing raw rendered request from typed chat messages.
pub(crate) fn assemble_qwen2_5_vl_chat_request_mrope_model_input<S>(
    input: Qwen25VlChatRequestMropeModelInputRequest<'_, '_, '_, '_, S>,
) -> Result<Qwen25VlRawRequestMropeModelInput, Qwen25VlChatRequestError<S::Error>>
where
    S: Qwen25VlDecodedFrameSource,
{
    let Qwen25VlChatRequestMropeModelInputRequest {
        model_dir,
        messages,
        fps,
        video_sources,
    } = input;
    let mut loaded_images = Vec::new();
    collect_typed_images(messages, &mut loaded_images)?;
    let validated = bind_typed_messages(messages, &loaded_images);
    assemble_validated_chat(model_dir, &validated, fps, video_sources)
}

fn assemble_validated_chat<'a, S>(
    model_dir: &Path,
    messages: &[ValidatedMessage<'a>],
    fps: Option<MropeProcessorFpsInput<'a>>,
    video_sources: &mut [S],
) -> Result<Qwen25VlRawRequestMropeModelInput, Qwen25VlChatRequestError<S::Error>>
where
    S: Qwen25VlDecodedFrameSource,
{
    let media_count = count_media(messages);
    if media_count > QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS {
        return Err(Qwen25VlChatRequestError::TooManyMediaItems {
            actual: media_count,
            limit: QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS,
        });
    }
    // Marker strings and model kind only. The public assembler still loads its own retained snapshot.
    let sources = load_qwen_vl_mrope_source_metadata(model_dir)?;
    if sources.model_type != QwenVlModelType::Qwen25Vl {
        return Err(Qwen25VlChatRequestError::UnsupportedModelType {
            model_type: sources.model_type,
        });
    }
    let tokens = sources.processor_special_tokens();
    let text = TextCodec::from_hf_dir(model_dir, 0, None, ChatFormat::ChatML)
        .map_err(|error| Qwen25VlChatRequestError::ChatTemplate(error.to_string()))?;
    if !text.has_chat_template() {
        return Err(Qwen25VlChatRequestError::MissingChatTemplate);
    }
    let (rendered_messages, media) = flatten_validated_messages(messages, &tokens);
    // The raw-request assembler below tokenizes the prompt with its own tokenizer and parses control tokens
    // in all of it: the chat's untrusted spans stop here, so a message that spells one still forges it.
    let rendered_prompt = text
        .render_template_chat(&rendered_messages)
        .map_err(|error| Qwen25VlChatRequestError::ChatTemplate(error.to_string()))?
        .prompt;
    assemble_qwen2_5_vl_raw_request_mrope_model_input(Qwen25VlRawRequestMropeModelInputRequest {
        model_dir,
        rendered_prompt: &rendered_prompt,
        media,
        fps,
        video_sources,
    })
    .map_err(Qwen25VlChatRequestError::RawRequest)
}

fn collect_typed_images<'a, E: std::error::Error + 'static>(
    messages: &[Qwen25VlChatMessage<'a>],
    loaded: &mut Vec<Qwen25VlDecodedImage>,
) -> Result<(), Qwen25VlChatRequestError<E>> {
    if messages.is_empty() {
        return Err(Qwen25VlChatRequestError::EmptyMessages);
    }
    for (message_index, message) in messages.iter().enumerate() {
        validate_role(message.role, message_index)?;
        if let Qwen25VlChatMessageContent::Parts(parts) = &message.content {
            for (part_index, part) in parts.iter().enumerate() {
                if let Qwen25VlChatContent::ImagePath(path) = part {
                    loaded.push(load_image_file(message_index, part_index, path)?);
                }
            }
        }
    }
    Ok(())
}

fn bind_typed_messages<'a>(
    messages: &'a [Qwen25VlChatMessage<'_>],
    loaded: &'a [Qwen25VlDecodedImage],
) -> Vec<ValidatedMessage<'a>> {
    let mut image_index = 0;
    messages
        .iter()
        .map(|message| ValidatedMessage {
            role: message.role,
            content: match &message.content {
                Qwen25VlChatMessageContent::Text(text) => ValidatedContent::Text(text),
                Qwen25VlChatMessageContent::Parts(parts) => ValidatedContent::Parts(
                    parts
                        .iter()
                        .map(|part| match part {
                            Qwen25VlChatContent::ImagePath(_) => {
                                let image = &loaded[image_index];
                                image_index += 1;
                                Qwen25VlChatContent::Image(image)
                            }
                            other => *other,
                        })
                        .collect(),
                ),
            },
        })
        .collect()
}

fn count_media(messages: &[ValidatedMessage<'_>]) -> usize {
    messages
        .iter()
        .map(|message| match &message.content {
            ValidatedContent::Text(_) => 0,
            ValidatedContent::Parts(parts) => parts
                .iter()
                .filter(|part| !matches!(part, Qwen25VlChatContent::Text(_)))
                .count(),
        })
        .sum()
}

fn flatten_validated_messages<'a>(
    messages: &[ValidatedMessage<'a>],
    tokens: &[MropeProcessorSpecialToken<'_>; 3],
) -> (serde_json::Value, Vec<Qwen25VlRawMediaInput<'a>>) {
    let vision_start = token_string(tokens, MropeSpecialTokenKind::VisionStart);
    let image_pad = token_string(tokens, MropeSpecialTokenKind::Image);
    let video_pad = token_string(tokens, MropeSpecialTokenKind::Video);
    let mut rendered = Vec::with_capacity(messages.len());
    let mut media = Vec::new();
    for message in messages {
        let content = match &message.content {
            ValidatedContent::Text(text) => (*text).to_string(),
            ValidatedContent::Parts(parts) => {
                let mut content = String::new();
                for part in parts {
                    match part {
                        Qwen25VlChatContent::Text(text) => content.push_str(text),
                        Qwen25VlChatContent::Image(image) => {
                            media.push(Qwen25VlRawMediaInput::Image((*image).clone()));
                            push_vision_run(&mut content, vision_start, image_pad);
                        }
                        Qwen25VlChatContent::ImagePath(_) => {
                            unreachable!(
                                "typed filesystem image paths are loaded before flattening"
                            );
                        }
                        Qwen25VlChatContent::Video(video) => {
                            media.push(Qwen25VlRawMediaInput::Video(*video));
                            push_vision_run(&mut content, vision_start, video_pad);
                        }
                    }
                }
                content
            }
        };
        rendered.push(serde_json::json!({
            "role": message.role,
            "content": content,
        }));
    }
    (serde_json::Value::Array(rendered), media)
}

fn validate_role<E: std::error::Error + 'static>(
    role: &str,
    message_index: usize,
) -> Result<(), Qwen25VlChatRequestError<E>> {
    if role.is_empty() {
        return Err(Qwen25VlChatRequestError::EmptyRole { message_index });
    }
    Ok(())
}

fn load_image_file<E: std::error::Error + 'static>(
    message_index: usize,
    part_index: usize,
    path: &Path,
) -> Result<Qwen25VlDecodedImage, Qwen25VlChatRequestError<E>> {
    if path.as_os_str().is_empty() {
        return Err(Qwen25VlChatRequestError::EmptyImagePath {
            message_index,
            part_index,
        });
    }
    let bytes = fs::read(path).map_err(|source| Qwen25VlChatRequestError::ImageFile {
        message_index,
        part_index,
        path: path.to_path_buf(),
        source: Qwen25VlChatImageFileError::Read(source),
    })?;
    super::qwen_vl_image::decode_qwen2_5_vl_image(&bytes).map_err(|source| {
        Qwen25VlChatRequestError::ImageFile {
            message_index,
            part_index,
            path: path.to_path_buf(),
            source: Qwen25VlChatImageFileError::Decode(source),
        }
    })
}

fn token_string<'a>(
    tokens: &'a [MropeProcessorSpecialToken<'a>; 3],
    role: MropeSpecialTokenKind,
) -> &'a str {
    tokens
        .iter()
        .find(|token| token.role == role)
        .map(|token| token.token)
        .expect("processor special tokens include vision_start, image, and video")
}

fn push_vision_run(content: &mut String, vision_start: &str, pad: &str) {
    content.push_str(vision_start);
    content.push_str(pad);
    content.push_str(VISION_END_DELIMITER);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::qwen_vl_image::{
        Qwen25VlDecodedFrameSource, Qwen25VlPadExpansionError, Qwen25VlRawRequestAssemblyError,
        assemble_qwen2_5_vl_raw_request_mrope_model_input,
    };
    use poot_load::qwen_vl::QwenVlModelType;
    use poot_models::mrope::{MropeProcessorFpsInput, MropeVisualKind};
    use std::{fs, path::PathBuf};
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    const VISION_START: &str = "<|vision_start|>";
    const IMAGE_PAD: &str = "<|image_pad|>";
    const VIDEO_PAD: &str = "<|video_pad|>";
    const IM_START: &str = "<|im_start|>";
    const IM_END: &str = "<|im_end|>";
    const CHAT_TEMPLATE: &str = "{% for message in messages %}{{ message.role }} {{ message.content }}\n{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}";
    // Frozen Qwen2.5-VL Instruct chat-template shape: default system prompt, string-content branch,
    // list-content vision markers, and add_generation_prompt. Flattening must take the string branch so
    // the list branch cannot emit a second marker run.
    const QWEN25_VL_INSTRUCT_CHAT_TEMPLATE: &str = r"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are a helpful assistant.' }}
    {%- endif %}
    {{- '\n<|im_end|>\n' }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == 'user') or (message.role == 'system' and not loop.first) or (message.role == 'assistant' and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {%- if message.content is string %}
            {{- message.content }}
        {%- else %}
            {%- for item in message.content %}
                {%- if item.image or item.image_url or item.type == 'image' or item.type == 'image_url' %}
                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                {%- elif item.video or item.video_url or item.type == 'video' or item.type == 'video_url' %}
                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                {%- elif item.text %}
                    {{- item.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}";
    fn encoded_image_bytes(format: image::ImageFormat, value: u8) -> Vec<u8> {
        use image::{Rgb, RgbImage};
        use std::io::Cursor;
        let mut bytes = Cursor::new(Vec::new());
        RgbImage::from_pixel(1, 1, Rgb([value, value, value]))
            .write_to(&mut bytes, format)
            .unwrap();
        bytes.into_inner()
    }

    fn write_image_file(
        dir: &std::path::Path,
        name: &str,
        format: image::ImageFormat,
        value: u8,
    ) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, encoded_image_bytes(format, value)).unwrap();
        path
    }

    fn decode_image_file(path: &std::path::Path) -> Qwen25VlDecodedImage {
        crate::multimodal::qwen_vl_image::decode_qwen2_5_vl_image(&fs::read(path).unwrap()).unwrap()
    }

    struct CwdImage {
        name: String,
    }

    impl Drop for CwdImage {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.name);
        }
    }

    struct FixtureDir {
        path: PathBuf,
    }

    impl std::ops::Deref for FixtureDir {
        type Target = std::path::Path;
        fn deref(&self) -> &Self::Target {
            &self.path
        }
    }

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[derive(Debug, thiserror::Error)]
    #[error("fixture video decode {0}")]
    struct FixtureVideoDecodeError(usize);

    struct RecordingFrameSource {
        frames: Vec<Qwen25VlDecodedImage>,
        calls: Vec<usize>,
    }

    impl Qwen25VlDecodedFrameSource for RecordingFrameSource {
        type Error = FixtureVideoDecodeError;

        fn decode_frame(
            &mut self,
            frame_index: usize,
        ) -> Result<Qwen25VlDecodedImage, Self::Error> {
            self.calls.push(frame_index);
            Ok(self.frames[frame_index].clone())
        }
    }

    fn chat_fixture(name: &str, model_type: &str, template: Option<&str>) -> FixtureDir {
        use std::collections::HashMap;

        let path =
            std::env::temp_dir().join(format!("poot_qwen25_vl_chat_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        fs::write(
            path.join("preprocessor_config.json"),
            r#"{
                "do_resize":false,"do_rescale":false,"do_normalize":false,
                "min_pixels":1,"max_pixels":1024,"patch_size":1,
                "temporal_patch_size":2,"merge_size":1
            }"#,
        )
        .unwrap();
        fs::write(
            path.join("config.json"),
            format!(
                r#"{{
                    "model_type":"{model_type}","vocab_size":100,"hidden_size":8,
                    "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":1,
                    "num_key_value_heads":1,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
                    "max_position_embeddings":128,"vision_start_token_id":10,
                    "image_token_id":11,"video_token_id":12,
                    "rope_scaling":{{"mrope_section":[1,1,2]}},
                    "vision_config":{{"tokens_per_second":4}}
                }}"#
            ),
        )
        .unwrap();
        let mut tokenizer_config = serde_json::json!({
            "vision_start_token": VISION_START,
            "image_token": IMAGE_PAD,
            "video_token": VIDEO_PAD,
        });
        if let Some(template) = template {
            tokenizer_config["chat_template"] = serde_json::Value::String(template.to_string());
        }
        fs::write(
            path.join("tokenizer_config.json"),
            serde_json::to_vec_pretty(&tokenizer_config).unwrap(),
        )
        .unwrap();
        let mut vocab = HashMap::from([
            ("t0".to_string(), 0),
            ("t1".to_string(), 1),
            ("user".to_string(), 2),
            ("assistant".to_string(), 3),
            ("hello".to_string(), 4),
            ("system".to_string(), 5),
            (VISION_START.to_string(), 10),
            (IMAGE_PAD.to_string(), 11),
            (VIDEO_PAD.to_string(), 12),
            (VISION_END_DELIMITER.to_string(), 13),
            (IM_START.to_string(), 14),
            (IM_END.to_string(), 15),
        ]);
        for id in 0..=15u32 {
            if !vocab.values().any(|&existing| existing == id) {
                vocab.insert(format!("fill{id}"), id);
            }
        }
        use tokenizers::AddedToken;
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("t0".to_string())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.add_special_tokens(&[
            AddedToken::from(VISION_START, true),
            AddedToken::from(IMAGE_PAD, true),
            AddedToken::from(VIDEO_PAD, true),
            AddedToken::from(VISION_END_DELIMITER, true),
            AddedToken::from(IM_START, true),
            AddedToken::from(IM_END, true),
        ]);
        tokenizer.save(path.join("tokenizer.json"), false).unwrap();
        FixtureDir { path }
    }

    fn raw_image(value: u8) -> Qwen25VlDecodedImage {
        Qwen25VlDecodedImage::from_rgb8(1, 1, vec![value; 3]).unwrap()
    }

    fn raw_image_hw(height: usize, width: usize, value: u8) -> Qwen25VlDecodedImage {
        Qwen25VlDecodedImage::from_rgb8(
            height,
            width,
            vec![value; height.checked_mul(width).unwrap().checked_mul(3).unwrap()],
        )
        .unwrap()
    }

    fn video_source(values: &[u8]) -> RecordingFrameSource {
        RecordingFrameSource {
            frames: values.iter().copied().map(raw_image).collect(),
            calls: Vec::new(),
        }
    }

    fn video_metadata<'a>(timestamps: &'a [f64]) -> Qwen25VlVideoMetadata<'a> {
        Qwen25VlVideoMetadata {
            duration_seconds: timestamps.len() as f64 / 2.0,
            source_frames_per_second: 2.0,
            frame_count: timestamps.len(),
            frame_timestamps_seconds: timestamps,
        }
    }

    fn expected_prompt(role: &str, content: &str) -> String {
        format!("{role} {content}\nassistant\n")
    }

    fn vision_run(pad: &str) -> String {
        format!("{VISION_START}{pad}{VISION_END_DELIMITER}")
    }

    fn assert_matches_raw(
        dir: &std::path::Path,
        actual: &Qwen25VlRawRequestMropeModelInput,
        rendered: &str,
        media: Vec<Qwen25VlRawMediaInput<'_>>,
        fps: Option<MropeProcessorFpsInput<'_>>,
        video_sources: &mut [RecordingFrameSource],
    ) {
        let direct = assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: dir,
                rendered_prompt: rendered,
                media,
                fps,
                video_sources,
            },
        )
        .unwrap();
        assert_eq!(actual.prompt_token_ids, direct.prompt_token_ids);
        assert_eq!(actual.media, direct.media);
        assert_eq!(actual.mrope_positions, direct.mrope_positions);
        assert_eq!(actual.media_order(), direct.media_order());
    }

    #[test]
    fn qwen25_vl_chat_request_assembles_text_image_and_matches_raw_assembler() {
        let dir = chat_fixture("text_image", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let image = raw_image(7);
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let actual = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Text("hello"),
                        Qwen25VlChatContent::Image(&image),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(actual.media_order(), [MropeVisualKind::Image]);
        let rendered = expected_prompt(
            "user",
            &format!("hello{VISION_START}{IMAGE_PAD}{VISION_END_DELIMITER}"),
        );
        let direct = assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &rendered,
                media: vec![Qwen25VlRawMediaInput::Image(image.clone())],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(actual.prompt_token_ids, direct.prompt_token_ids);
        assert_eq!(actual.media, direct.media);
        assert_eq!(actual.mrope_positions, direct.mrope_positions);

        let text_only = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Text("hello"),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(text_only.media_order(), []);
        let text_direct = assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &expected_prompt("user", "hello"),
                media: vec![],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(text_only.prompt_token_ids, text_direct.prompt_token_ids);

        let image_only = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![Qwen25VlChatContent::Image(
                        &image,
                    )]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(image_only.media_order(), [MropeVisualKind::Image]);
        let image_only_direct = assemble_qwen2_5_vl_raw_request_mrope_model_input(
            Qwen25VlRawRequestMropeModelInputRequest {
                model_dir: &dir,
                rendered_prompt: &expected_prompt(
                    "user",
                    &format!("{VISION_START}{IMAGE_PAD}{VISION_END_DELIMITER}"),
                ),
                media: vec![Qwen25VlRawMediaInput::Image(image.clone())],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(
            image_only.prompt_token_ids,
            image_only_direct.prompt_token_ids
        );
    }

    #[test]
    fn qwen25_vl_chat_request_preserves_interleaved_image_video_order() {
        let dir = chat_fixture("interleaved", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let first = raw_image(1);
        let second = raw_image(2);
        let timestamps = [0.0, 0.5];
        let mut sources = [video_source(&[10, 20])];
        let actual = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Image(&first),
                        Qwen25VlChatContent::Text("t0"),
                        Qwen25VlChatContent::Video(video_metadata(&timestamps)),
                        Qwen25VlChatContent::Image(&second),
                    ]),
                }],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut sources,
            },
        )
        .unwrap();
        assert_eq!(
            actual.media_order(),
            [
                MropeVisualKind::Image,
                MropeVisualKind::Video,
                MropeVisualKind::Image
            ]
        );
        assert_eq!(sources[0].calls, [0, 1]);
        let interleaved_content = format!(
            "{}t0{}{}",
            vision_run(IMAGE_PAD),
            vision_run(VIDEO_PAD),
            vision_run(IMAGE_PAD)
        );
        let mut interleaved_direct_sources = [video_source(&[10, 20])];
        assert_matches_raw(
            &dir,
            &actual,
            &expected_prompt("user", &interleaved_content),
            vec![
                Qwen25VlRawMediaInput::Image(first.clone()),
                Qwen25VlRawMediaInput::Video(video_metadata(&timestamps)),
                Qwen25VlRawMediaInput::Image(second.clone()),
            ],
            Some(MropeProcessorFpsInput::Scalar(2.0)),
            &mut interleaved_direct_sources,
        );

        let mut video_only_sources = [video_source(&[3, 4])];
        let video_only = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![Qwen25VlChatContent::Video(
                        video_metadata(&timestamps),
                    )]),
                }],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut video_only_sources,
            },
        )
        .unwrap();
        assert_eq!(video_only.media_order(), [MropeVisualKind::Video]);
        assert_eq!(video_only_sources[0].calls, [0, 1]);
        let mut video_only_direct_sources = [video_source(&[3, 4])];
        assert_matches_raw(
            &dir,
            &video_only,
            &expected_prompt("user", &vision_run(VIDEO_PAD)),
            vec![Qwen25VlRawMediaInput::Video(video_metadata(&timestamps))],
            Some(MropeProcessorFpsInput::Scalar(2.0)),
            &mut video_only_direct_sources,
        );
    }

    #[test]
    fn qwen25_vl_chat_request_reports_empty_missing_template_kind_cap_and_render_errors() {
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let empty = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: std::path::Path::new("/missing-qwen25-vl-chat-empty"),
                messages: &[],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(empty, Qwen25VlChatRequestError::EmptyMessages));

        let dir = chat_fixture("missing_template", "qwen2_5_vl", None);
        let missing = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Text("hello"),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            missing,
            Qwen25VlChatRequestError::MissingChatTemplate
        ));

        let qwen2 = chat_fixture("qwen2_vl", "qwen2_vl", Some("{% for x in %}"));
        let kind = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &qwen2,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Text("hello"),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            kind,
            Qwen25VlChatRequestError::UnsupportedModelType {
                model_type: QwenVlModelType::Qwen2Vl
            }
        ));

        let broken = chat_fixture("broken_template", "qwen2_5_vl", Some("{% for x in %}"));
        let render = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &broken,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Text("hello"),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(render, Qwen25VlChatRequestError::ChatTemplate(_)));

        let images = (0..=QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS)
            .map(|_| raw_image(1))
            .collect::<Vec<_>>();
        let parts = images
            .iter()
            .map(Qwen25VlChatContent::Image)
            .collect::<Vec<_>>();
        let cap = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: std::path::Path::new("/missing-qwen25-vl-chat-cap"),
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(parts),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            cap,
            Qwen25VlChatRequestError::TooManyMediaItems {
                actual: 65,
                limit: 64
            }
        ));

        let qwen2_no_template = chat_fixture("qwen2_vl_no_template", "qwen2_vl", None);
        let kind_before_template = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &qwen2_no_template,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Text("hello"),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            kind_before_template,
            Qwen25VlChatRequestError::UnsupportedModelType {
                model_type: QwenVlModelType::Qwen2Vl
            }
        ));
    }

    #[test]
    fn qwen25_vl_chat_request_renders_instruct_template_string_branch_once() {
        let dir = chat_fixture(
            "instruct_template",
            "qwen2_5_vl",
            Some(QWEN25_VL_INSTRUCT_CHAT_TEMPLATE),
        );
        let image = raw_image(7);
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let actual = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Text("hello"),
                        Qwen25VlChatContent::Image(&image),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        let rendered = format!(
            "{IM_START}system\nYou are a helpful assistant.{IM_END}\n{IM_START}user\nhello{}{IM_END}\n{IM_START}assistant\n",
            vision_run(IMAGE_PAD)
        );
        assert_matches_raw(
            &dir,
            &actual,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(image.clone())],
            None,
            &mut no_video_sources,
        );
        assert_eq!(
            rendered.matches(&vision_run(IMAGE_PAD)).count(),
            1,
            "flattened string content must not take the list-content marker branch"
        );
    }

    #[test]
    fn qwen25_vl_chat_loads_local_filesystem_images() {
        let dir = chat_fixture("filesystem", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let media_dir = dir.join("media");
        fs::create_dir_all(&media_dir).unwrap();
        let png_path = write_image_file(&media_dir, "pixel.png", image::ImageFormat::Png, 7);
        let png_image = decode_image_file(&png_path);
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let rendered = expected_prompt("user", &format!("hello{}", vision_run(IMAGE_PAD)));

        let typed = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Text("hello"),
                        Qwen25VlChatContent::ImagePath(&png_path),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &typed,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image.clone())],
            None,
            &mut no_video_sources,
        );

        let mixed_decoded = raw_image(3);
        let mixed = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Image(&mixed_decoded),
                        Qwen25VlChatContent::ImagePath(&png_path),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(
            mixed.media_order(),
            [MropeVisualKind::Image, MropeVisualKind::Image]
        );
        assert_matches_raw(
            &dir,
            &mixed,
            &expected_prompt(
                "user",
                &format!("{}{}", vision_run(IMAGE_PAD), vision_run(IMAGE_PAD)),
            ),
            vec![
                Qwen25VlRawMediaInput::Image(mixed_decoded.clone()),
                Qwen25VlRawMediaInput::Image(png_image.clone()),
            ],
            None,
            &mut no_video_sources,
        );

        let relative_name = format!(
            "poot_qwen25_vl_chat_rel_{}_{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        fs::write(
            &relative_name,
            encoded_image_bytes(image::ImageFormat::Png, 13),
        )
        .unwrap();
        let relative = CwdImage {
            name: relative_name.clone(),
        };
        let relative_image = decode_image_file(Path::new(&relative.name));
        let relative_path = Path::new(&relative.name);
        let relative_actual = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Text("hello"),
                        Qwen25VlChatContent::ImagePath(relative_path),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &relative_actual,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(relative_image)],
            None,
            &mut no_video_sources,
        );
        drop(relative);
    }

    #[test]
    fn qwen25_vl_chat_rejects_filesystem_image_errors_before_checkpoint_io() {
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let missing_dir = std::path::Path::new("/missing-qwen25-vl-chat-fs");
        let missing_path = PathBuf::from("/missing-qwen25-vl-chat-fs-image.png");
        let missing = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: missing_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::ImagePath(&missing_path),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            missing,
            Qwen25VlChatRequestError::ImageFile {
                message_index: 0,
                part_index: 0,
                path,
                source: Qwen25VlChatImageFileError::Read(_)
            } if path == missing_path
        ));

        let empty = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: missing_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::ImagePath(Path::new("")),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            empty,
            Qwen25VlChatRequestError::EmptyImagePath {
                message_index: 0,
                part_index: 0
            }
        ));
    }

    #[test]
    fn qwen25_vl_chat_expands_merged_pads_and_rejects_extra_template_pads() {
        let dir = chat_fixture("pad_expansion", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let image = raw_image_hw(2, 2, 7);
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let actual = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Text("hello"),
                        Qwen25VlChatContent::Image(&image),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(
            actual
                .prompt_token_ids
                .iter()
                .filter(|&&id| id == 11)
                .count(),
            4
        );
        let rendered = expected_prompt(
            "user",
            &format!("hello{VISION_START}{IMAGE_PAD}{VISION_END_DELIMITER}"),
        );
        assert_matches_raw(
            &dir,
            &actual,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(image.clone())],
            None,
            &mut no_video_sources,
        );

        let mut video_sources = [video_source(&[10, 20])];
        video_sources[0].frames = vec![raw_image_hw(2, 2, 10), raw_image_hw(2, 2, 20)];
        let timestamps = [0.0, 0.5];
        let interleaved = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Image(&image),
                        Qwen25VlChatContent::Text("t0"),
                        Qwen25VlChatContent::Video(video_metadata(&timestamps)),
                    ]),
                }],
                fps: Some(MropeProcessorFpsInput::Scalar(2.0)),
                video_sources: &mut video_sources,
            },
        )
        .unwrap();
        assert_eq!(
            interleaved
                .prompt_token_ids
                .iter()
                .filter(|&&id| id == 11)
                .count(),
            4
        );
        assert_eq!(
            interleaved
                .prompt_token_ids
                .iter()
                .filter(|&&id| id == 12)
                .count(),
            4
        );

        let merge_dir = chat_fixture("pad_expansion_merge", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        fs::write(
            merge_dir.join("preprocessor_config.json"),
            r#"{
                "do_resize":false,"do_rescale":false,"do_normalize":false,
                "min_pixels":1,"max_pixels":1024,"patch_size":1,
                "temporal_patch_size":2,"merge_size":2
            }"#,
        )
        .unwrap();
        let merge_image = raw_image_hw(4, 4, 3);
        let merged = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &merge_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![Qwen25VlChatContent::Image(
                        &merge_image,
                    )]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(
            merged
                .prompt_token_ids
                .iter()
                .filter(|&&id| id == 11)
                .count(),
            4
        );

        let extra_dir = chat_fixture(
            "pad_expansion_extra",
            "qwen2_5_vl",
            Some(
                "{% for message in messages %}{{ message.role }} {{ message.content }} <|image_pad|>\n{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}",
            ),
        );
        let extra = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &extra_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![Qwen25VlChatContent::Image(
                        &image,
                    )]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            extra,
            Qwen25VlChatRequestError::RawRequest(Qwen25VlRawRequestAssemblyError::PadExpansion(
                Qwen25VlPadExpansionError::PadCountMismatch {
                    kind: MropeVisualKind::Image,
                    expected: 1,
                    actual: 2
                }
            ))
        ));

        let identity = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![Qwen25VlChatContent::Image(
                        &raw_image(1),
                    )]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_eq!(
            identity
                .prompt_token_ids
                .iter()
                .filter(|&&id| id == 11)
                .count(),
            1
        );

        let missing_dir = chat_fixture(
            "pad_expansion_missing",
            "qwen2_5_vl",
            Some(
                "{% for message in messages %}{{ message.role }}\n{% endfor %}{% if add_generation_prompt %}assistant\n{% endif %}",
            ),
        );
        let missing = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &missing_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![Qwen25VlChatContent::Image(
                        &image,
                    )]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            missing,
            Qwen25VlChatRequestError::RawRequest(Qwen25VlRawRequestAssemblyError::PadExpansion(
                Qwen25VlPadExpansionError::PadCountMismatch {
                    kind: MropeVisualKind::Image,
                    expected: 1,
                    actual: 0
                }
            ))
        ));
    }
}

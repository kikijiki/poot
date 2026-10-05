//! Chat and OpenAI construction of the Qwen2.5-VL raw rendered request.
//!
//! Content-part order is the media order. Vision parts are flattened with existing processor markers, then
//! the checkpoint chat template is rendered through the existing minijinja path. Local PNG/JPEG paths and
//! `file:` URLs and bounded remote HTTP(S) URLs are decoded here. After render, the public raw assembler expands
//! each pad from merged-grid counts. Codec/container video work, vision, Runner/HTTP, and GPU stay out of this
//! boundary.
//!
//! Held for Card 566b (the VLM front end on the driver): the `pub(crate)` chat-request entry points below have no
//! production caller yet, and the front end consumes them. Where rustc reports one as dead it carries
//! `#[expect(dead_code, reason = "held for POOT-739 (formerly 566b)")]`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine as _;
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
use crate::text::chat::{read_tokenizer_config_chat, render_jinja_value};

const VISION_END_DELIMITER: &str = "<|vision_end|>";
const REMOTE_IMAGE_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const REMOTE_IMAGE_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const REMOTE_IMAGE_MAX_REDIRECTS: usize = 5;
pub const QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES: usize = 64 * 1024 * 1024;

/// One typed Qwen2.5-VL chat message.
pub struct Qwen25VlChatMessage<'a> {
    pub role: &'a str,
    pub content: Qwen25VlChatMessageContent<'a>,
}

/// Message content: a plain string or ordered typed parts.
pub enum Qwen25VlChatMessageContent<'a> {
    Text(&'a str),
    Parts(Vec<Qwen25VlChatContent<'a>>),
}

/// One content part. Images are already decoded or a local PNG/JPEG path; videos are already probed.
#[derive(Clone, Copy)]
pub enum Qwen25VlChatContent<'a> {
    Text(&'a str),
    Image(&'a Qwen25VlDecodedImage),
    ImagePath(&'a Path),
    Video(Qwen25VlVideoMetadata<'a>),
}

/// Inputs for chat-template construction of the existing raw rendered request.
pub(crate) struct Qwen25VlChatRequestMropeModelInputRequest<'dir, 'msg, 'data, 'sources, S> {
    pub model_dir: &'dir Path,
    pub messages: &'msg [Qwen25VlChatMessage<'data>],
    pub fps: Option<MropeProcessorFpsInput<'data>>,
    pub video_sources: &'sources mut [S],
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatRequestError<E: std::error::Error + 'static> {
    #[error("parse Qwen2.5-VL OpenAI chat JSON: {0}")]
    Json(#[source] serde_json::Error),
    #[error("Qwen2.5-VL chat request is missing a messages array")]
    MissingMessages,
    #[error("Qwen2.5-VL chat request has no messages")]
    EmptyMessages,
    #[error("Qwen2.5-VL chat message {message_index} is missing a role")]
    MissingRole { message_index: usize },
    #[error("Qwen2.5-VL chat message {message_index} has an empty role")]
    EmptyRole { message_index: usize },
    #[error("Qwen2.5-VL chat message {message_index} is missing content")]
    MissingContent { message_index: usize },
    #[error(
        "Qwen2.5-VL chat message {message_index} part {part_index} has unsupported content type {content_type:?}"
    )]
    UnsupportedContentType {
        message_index: usize,
        part_index: usize,
        content_type: String,
    },
    #[error("Qwen2.5-VL chat message {message_index} text part {part_index} is missing text")]
    MissingText {
        message_index: usize,
        part_index: usize,
    },
    #[error(
        "Qwen2.5-VL chat message {message_index} image part {part_index} is missing an image_url.url"
    )]
    MissingImageUrl {
        message_index: usize,
        part_index: usize,
    },
    #[error(
        "Qwen2.5-VL chat message {message_index} image part {part_index} rejects URL scheme {scheme:?}"
    )]
    UnsupportedMediaUrl {
        message_index: usize,
        part_index: usize,
        scheme: String,
    },
    #[error(
        "Qwen2.5-VL chat message {message_index} image part {part_index} rejects non-local file URL host {host:?}"
    )]
    UnsupportedFileUrlHost {
        message_index: usize,
        part_index: usize,
        host: String,
    },
    #[error(
        "Qwen2.5-VL chat message {message_index} image part {part_index} has an invalid file URL"
    )]
    InvalidFileUrl {
        message_index: usize,
        part_index: usize,
        #[source]
        source: Qwen25VlChatFileUrlError,
    },
    #[error(
        "Qwen2.5-VL chat message {message_index} image part {part_index} has an empty image path"
    )]
    EmptyImagePath {
        message_index: usize,
        part_index: usize,
    },
    #[error(
        "load Qwen2.5-VL chat image file {path:?} at message {message_index} part {part_index}"
    )]
    ImageFile {
        message_index: usize,
        part_index: usize,
        path: PathBuf,
        #[source]
        source: Qwen25VlChatImageFileError,
    },
    #[error("load Qwen2.5-VL remote image {url:?} at message {message_index} part {part_index}")]
    RemoteImage {
        message_index: usize,
        part_index: usize,
        url: String,
        #[source]
        source: Qwen25VlChatRemoteImageError,
    },
    #[error("Qwen2.5-VL chat message {message_index} part {part_index} does not accept video JSON")]
    UnsupportedVideoJson {
        message_index: usize,
        part_index: usize,
    },
    #[error("decode Qwen2.5-VL OpenAI image data URL at message {message_index} part {part_index}")]
    ImageDataUrl {
        message_index: usize,
        part_index: usize,
        #[source]
        source: Qwen25VlChatImageDataUrlError,
    },
    #[error("Qwen2.5-VL chat request has {actual} media items, exceeding the {limit}-item cap")]
    TooManyMediaItems { actual: usize, limit: usize },
    #[error(transparent)]
    Metadata(#[from] QwenVlMropeMetadataLoadError),
    #[error("Qwen2.5-VL chat request does not support model type {model_type:?}")]
    UnsupportedModelType { model_type: QwenVlModelType },
    #[error("Qwen2.5-VL chat request is missing a chat template")]
    MissingChatTemplate,
    #[error("render Qwen2.5-VL chat template: {0}")]
    ChatTemplate(String),
    #[error(transparent)]
    RawRequest(Qwen25VlRawRequestAssemblyError<E>),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatImageDataUrlError {
    #[error("decode Qwen2.5-VL image data URL base64: {0}")]
    Base64(#[from] base64::DecodeError),
    #[error(transparent)]
    Decode(#[from] Qwen25VlImageDecodeError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen25VlChatFileUrlError {
    #[error("Qwen2.5-VL file URL path is empty")]
    EmptyPath,
    #[error("Qwen2.5-VL file URL path is not absolute")]
    RelativePath,
    #[error("Qwen2.5-VL file URL must not include a query string")]
    QueryString,
    #[error("Qwen2.5-VL file URL must not include a fragment")]
    Fragment,
    #[error("Qwen2.5-VL file URL percent-encoding is truncated")]
    TruncatedPercent,
    #[error("Qwen2.5-VL file URL percent-encoding is not hexadecimal")]
    InvalidPercent,
    #[error("Qwen2.5-VL file URL decodes a NUL byte")]
    NulByte,
    #[error("Qwen2.5-VL file URL is not valid UTF-8 after percent-decoding")]
    InvalidUtf8,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatImageFileError {
    #[error("read Qwen2.5-VL image file: {0}")]
    Read(#[from] std::io::Error),
    #[error(transparent)]
    Decode(#[from] Qwen25VlImageDecodeError),
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen25VlChatRemoteImageError {
    #[error("build Qwen2.5-VL remote image HTTP client: {0}")]
    Client(#[source] reqwest::Error),
    #[error("request Qwen2.5-VL remote image: {0}")]
    Request(#[source] reqwest::Error),
    #[error("Qwen2.5-VL remote image returned HTTP status {status}")]
    Status { status: u16 },
    #[error("Qwen2.5-VL remote image declares {declared} bytes, exceeding the {limit}-byte cap")]
    DeclaredTooLarge { declared: u64, limit: usize },
    #[error("read Qwen2.5-VL remote image response body: {0}")]
    Body(#[source] std::io::Error),
    #[error("Qwen2.5-VL remote image exceeds the {limit}-byte cap")]
    StreamTooLarge { limit: usize },
    #[error("allocate Qwen2.5-VL remote image response buffer: {0}")]
    Allocate(#[source] std::collections::TryReserveError),
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

struct OwnedOpenAiMessage {
    role: String,
    content: OwnedOpenAiContent,
}

enum OwnedOpenAiContent {
    Text(String),
    Parts(Vec<OwnedOpenAiPart>),
}

enum OwnedOpenAiPart {
    Text(String),
    Image(usize),
}

/// Construct the existing raw rendered request from typed chat messages.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
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

/// Construct the existing raw rendered request from OpenAI chat JSON.
///
/// Video parts are rejected. Images are accepted as PNG/JPEG data URLs, local filesystem paths, local `file:`
/// URLs, and bounded remote HTTP(S) URLs.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "held for POOT-739 (formerly 566b)")
)]
pub(crate) fn assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json<S>(
    model_dir: &Path,
    body: &str,
    fps: Option<MropeProcessorFpsInput<'_>>,
    video_sources: &mut [S],
) -> Result<Qwen25VlRawRequestMropeModelInput, Qwen25VlChatRequestError<S::Error>>
where
    S: Qwen25VlDecodedFrameSource,
{
    let mut client = None;
    let mut remote_fetch = |url: &str| {
        if client.is_none() {
            client = Some(build_remote_image_client()?);
        }
        fetch_remote_image_bytes(
            client
                .as_ref()
                .expect("remote image client was initialized above"),
            url,
        )
    };
    assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
        model_dir,
        body,
        fps,
        video_sources,
        &mut remote_fetch,
    )
}

fn assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote<S, F>(
    model_dir: &Path,
    body: &str,
    fps: Option<MropeProcessorFpsInput<'_>>,
    video_sources: &mut [S],
    remote_fetch: &mut F,
) -> Result<Qwen25VlRawRequestMropeModelInput, Qwen25VlChatRequestError<S::Error>>
where
    S: Qwen25VlDecodedFrameSource,
    F: FnMut(&str) -> Result<Vec<u8>, Qwen25VlChatRemoteImageError>,
{
    let (messages, images) = parse_openai_chat_messages_with_remote(body, remote_fetch)?;
    let validated = validate_owned_messages(&messages, &images);
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
    let (template, bos, eos) = read_tokenizer_config_chat(model_dir);
    let Some(template) = template.as_deref() else {
        return Err(Qwen25VlChatRequestError::MissingChatTemplate);
    };
    let (rendered_messages, media) = flatten_validated_messages(messages, &tokens);
    let rendered_prompt = render_jinja_value(
        template,
        &rendered_messages,
        bos.as_deref(),
        eos.as_deref(),
        None,
    )
    .map_err(|error| Qwen25VlChatRequestError::ChatTemplate(error.to_string()))?;
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

fn validate_owned_messages<'a>(
    messages: &'a [OwnedOpenAiMessage],
    images: &'a [Qwen25VlDecodedImage],
) -> Vec<ValidatedMessage<'a>> {
    messages
        .iter()
        .map(|message| ValidatedMessage {
            role: message.role.as_str(),
            content: match &message.content {
                OwnedOpenAiContent::Text(text) => ValidatedContent::Text(text),
                OwnedOpenAiContent::Parts(parts) => ValidatedContent::Parts(
                    parts
                        .iter()
                        .map(|part| match part {
                            OwnedOpenAiPart::Text(text) => Qwen25VlChatContent::Text(text),
                            OwnedOpenAiPart::Image(index) => {
                                Qwen25VlChatContent::Image(&images[*index])
                            }
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

fn parse_openai_chat_messages_with_remote<E, F>(
    body: &str,
    remote_fetch: &mut F,
) -> Result<(Vec<OwnedOpenAiMessage>, Vec<Qwen25VlDecodedImage>), Qwen25VlChatRequestError<E>>
where
    E: std::error::Error + 'static,
    F: FnMut(&str) -> Result<Vec<u8>, Qwen25VlChatRemoteImageError>,
{
    let root: serde_json::Value =
        serde_json::from_str(body).map_err(Qwen25VlChatRequestError::Json)?;
    let Some(messages) = root.get("messages") else {
        return Err(Qwen25VlChatRequestError::MissingMessages);
    };
    let Some(messages) = messages.as_array() else {
        return Err(Qwen25VlChatRequestError::MissingMessages);
    };
    if messages.is_empty() {
        return Err(Qwen25VlChatRequestError::EmptyMessages);
    }
    let mut owned = Vec::with_capacity(messages.len());
    let mut images = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(|value| value.as_str())
            .ok_or(Qwen25VlChatRequestError::MissingRole { message_index })?;
        validate_role(role, message_index)?;
        let Some(content) = message.get("content") else {
            return Err(Qwen25VlChatRequestError::MissingContent { message_index });
        };
        let content = match content {
            serde_json::Value::String(text) => OwnedOpenAiContent::Text(text.clone()),
            serde_json::Value::Array(parts) => {
                let mut owned_parts = Vec::with_capacity(parts.len());
                for (part_index, part) in parts.iter().enumerate() {
                    owned_parts.push(parse_openai_part(
                        message_index,
                        part_index,
                        part,
                        &mut images,
                        remote_fetch,
                    )?);
                }
                OwnedOpenAiContent::Parts(owned_parts)
            }
            _ => return Err(Qwen25VlChatRequestError::MissingContent { message_index }),
        };
        owned.push(OwnedOpenAiMessage {
            role: role.to_string(),
            content,
        });
    }
    Ok((owned, images))
}

fn parse_openai_part<E, F>(
    message_index: usize,
    part_index: usize,
    part: &serde_json::Value,
    images: &mut Vec<Qwen25VlDecodedImage>,
    remote_fetch: &mut F,
) -> Result<OwnedOpenAiPart, Qwen25VlChatRequestError<E>>
where
    E: std::error::Error + 'static,
    F: FnMut(&str) -> Result<Vec<u8>, Qwen25VlChatRemoteImageError>,
{
    match part.get("type").and_then(|value| value.as_str()) {
        Some("text") => {
            let text = part.get("text").and_then(|value| value.as_str()).ok_or(
                Qwen25VlChatRequestError::MissingText {
                    message_index,
                    part_index,
                },
            )?;
            Ok(OwnedOpenAiPart::Text(text.to_string()))
        }
        Some("image_url") => {
            let url = part
                .get("image_url")
                .and_then(|value| value.get("url"))
                .and_then(|value| value.as_str())
                .ok_or(Qwen25VlChatRequestError::MissingImageUrl {
                    message_index,
                    part_index,
                })?;
            let image = decode_openai_image_url(message_index, part_index, url, remote_fetch)?;
            let index = images.len();
            images.push(image);
            Ok(OwnedOpenAiPart::Image(index))
        }
        Some("video") | Some("video_url") => Err(Qwen25VlChatRequestError::UnsupportedVideoJson {
            message_index,
            part_index,
        }),
        Some(other) => Err(Qwen25VlChatRequestError::UnsupportedContentType {
            message_index,
            part_index,
            content_type: other.to_string(),
        }),
        None => Err(Qwen25VlChatRequestError::UnsupportedContentType {
            message_index,
            part_index,
            content_type: String::new(),
        }),
    }
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

fn decode_openai_image_url<E, F>(
    message_index: usize,
    part_index: usize,
    url: &str,
    remote_fetch: &mut F,
) -> Result<Qwen25VlDecodedImage, Qwen25VlChatRequestError<E>>
where
    E: std::error::Error + 'static,
    F: FnMut(&str) -> Result<Vec<u8>, Qwen25VlChatRemoteImageError>,
{
    if url.starts_with("data:image/png;base64,")
        || url.starts_with("data:image/jpeg;base64,")
        || url.starts_with("data:image/jpg;base64,")
    {
        return decode_image_data_url(message_index, part_index, url);
    }
    if let Some(scheme) = uri_scheme(url) {
        if scheme.eq_ignore_ascii_case("file") {
            let path = parse_local_file_url(url).map_err(|source| match source {
                FileUrlParseError::RemoteHost(host) => {
                    Qwen25VlChatRequestError::UnsupportedFileUrlHost {
                        message_index,
                        part_index,
                        host,
                    }
                }
                FileUrlParseError::Invalid(source) => Qwen25VlChatRequestError::InvalidFileUrl {
                    message_index,
                    part_index,
                    source,
                },
            })?;
            return load_image_file(message_index, part_index, &path);
        }
        if scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https") {
            let bytes =
                remote_fetch(url).map_err(|source| Qwen25VlChatRequestError::RemoteImage {
                    message_index,
                    part_index,
                    url: url.to_string(),
                    source,
                })?;
            return super::qwen_vl_image::decode_qwen2_5_vl_image(&bytes).map_err(|source| {
                Qwen25VlChatRequestError::RemoteImage {
                    message_index,
                    part_index,
                    url: url.to_string(),
                    source: Qwen25VlChatRemoteImageError::Decode(source),
                }
            });
        }
        return Err(Qwen25VlChatRequestError::UnsupportedMediaUrl {
            message_index,
            part_index,
            scheme: scheme.to_string(),
        });
    }
    load_image_file(message_index, part_index, Path::new(url))
}

fn build_remote_image_client() -> Result<reqwest::blocking::Client, Qwen25VlChatRemoteImageError> {
    reqwest::blocking::Client::builder()
        .connect_timeout(REMOTE_IMAGE_CONNECT_TIMEOUT)
        .timeout(REMOTE_IMAGE_REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            // `previous` includes the initial URL, so reqwest's own limited policy uses `> max`.
            if attempt.previous().len() > REMOTE_IMAGE_MAX_REDIRECTS {
                return attempt.error("Qwen2.5-VL remote image exceeded the redirect cap");
            }
            match attempt.url().scheme() {
                "http" | "https" => attempt.follow(),
                _ => attempt.error("Qwen2.5-VL remote image redirect used an unsupported scheme"),
            }
        }))
        .build()
        .map_err(Qwen25VlChatRemoteImageError::Client)
}

fn fetch_remote_image_bytes(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<Vec<u8>, Qwen25VlChatRemoteImageError> {
    let response = client
        .get(url)
        .send()
        .map_err(Qwen25VlChatRemoteImageError::Request)?;
    let status = response.status();
    if !status.is_success() {
        return Err(Qwen25VlChatRemoteImageError::Status {
            status: status.as_u16(),
        });
    }
    let declared = response.content_length();
    read_remote_image_body(response, declared, QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES)
}

fn read_remote_image_body<R: Read>(
    mut reader: R,
    declared: Option<u64>,
    limit: usize,
) -> Result<Vec<u8>, Qwen25VlChatRemoteImageError> {
    if let Some(declared) = declared
        && declared > limit as u64
    {
        return Err(Qwen25VlChatRemoteImageError::DeclaredTooLarge { declared, limit });
    }
    let mut bytes = Vec::new();
    if let Some(declared) = declared {
        bytes
            .try_reserve_exact(declared as usize)
            .map_err(Qwen25VlChatRemoteImageError::Allocate)?;
    }
    let mut chunk = [0u8; 8192];
    while bytes.len() < limit {
        let remaining = limit - bytes.len();
        let chunk_limit = remaining.min(chunk.len());
        let read = reader
            .read(&mut chunk[..chunk_limit])
            .map_err(Qwen25VlChatRemoteImageError::Body)?;
        if read == 0 {
            break;
        }
        reserve_remote_image_capacity(&mut bytes, read, limit)?;
        bytes.extend_from_slice(&chunk[..read]);
    }
    if bytes.len() == limit {
        let mut overflow = [0u8; 1];
        if reader
            .read(&mut overflow)
            .map_err(Qwen25VlChatRemoteImageError::Body)?
            != 0
        {
            return Err(Qwen25VlChatRemoteImageError::StreamTooLarge { limit });
        }
    }
    Ok(bytes)
}

fn reserve_remote_image_capacity(
    bytes: &mut Vec<u8>,
    additional: usize,
    limit: usize,
) -> Result<(), Qwen25VlChatRemoteImageError> {
    let required = bytes
        .len()
        .checked_add(additional)
        .expect("bounded remote image read length does not overflow usize");
    debug_assert!(required <= limit);
    if required <= bytes.capacity() {
        return Ok(());
    }

    // Keep amortized growth without Vec's unbounded doubling at the cap. The overflow byte is probed separately,
    // so the response Vec never requests capacity beyond the byte limit.
    let target = bytes.capacity().saturating_mul(2).max(required).min(limit);
    bytes
        .try_reserve_exact(target - bytes.len())
        .map_err(Qwen25VlChatRemoteImageError::Allocate)
}

fn decode_image_data_url<E: std::error::Error + 'static>(
    message_index: usize,
    part_index: usize,
    url: &str,
) -> Result<Qwen25VlDecodedImage, Qwen25VlChatRequestError<E>> {
    let encoded = if let Some(rest) = url.strip_prefix("data:image/png;base64,") {
        rest
    } else if let Some(rest) = url
        .strip_prefix("data:image/jpeg;base64,")
        .or_else(|| url.strip_prefix("data:image/jpg;base64,"))
    {
        rest
    } else {
        return Err(Qwen25VlChatRequestError::UnsupportedMediaUrl {
            message_index,
            part_index,
            scheme: media_url_scheme(url),
        });
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|source| Qwen25VlChatRequestError::ImageDataUrl {
            message_index,
            part_index,
            source: Qwen25VlChatImageDataUrlError::Base64(source),
        })?;
    super::qwen_vl_image::decode_qwen2_5_vl_image(&bytes).map_err(|source| {
        Qwen25VlChatRequestError::ImageDataUrl {
            message_index,
            part_index,
            source: Qwen25VlChatImageDataUrlError::Decode(source),
        }
    })
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

enum FileUrlParseError {
    RemoteHost(String),
    Invalid(Qwen25VlChatFileUrlError),
}

fn parse_local_file_url(url: &str) -> Result<PathBuf, FileUrlParseError> {
    let rest = strip_file_scheme(url).ok_or(FileUrlParseError::Invalid(
        Qwen25VlChatFileUrlError::EmptyPath,
    ))?;
    if let Some(delimiter) = first_query_or_fragment(rest) {
        return Err(FileUrlParseError::Invalid(delimiter));
    }
    let (host, path) = split_file_url_host_path(rest);
    let host = percent_decode(host).map_err(FileUrlParseError::Invalid)?;
    if !host.is_empty() && !host.eq_ignore_ascii_case("localhost") {
        return Err(FileUrlParseError::RemoteHost(host));
    }
    let path = percent_decode(path).map_err(FileUrlParseError::Invalid)?;
    if path.is_empty() {
        return Err(FileUrlParseError::Invalid(
            Qwen25VlChatFileUrlError::EmptyPath,
        ));
    }
    if !path.starts_with('/') {
        return Err(FileUrlParseError::Invalid(
            Qwen25VlChatFileUrlError::RelativePath,
        ));
    }
    Ok(PathBuf::from(path))
}

fn strip_file_scheme(url: &str) -> Option<&str> {
    let scheme = uri_scheme(url)?;
    if !scheme.eq_ignore_ascii_case("file") {
        return None;
    }
    Some(&url[scheme.len() + 1..])
}

fn split_file_url_host_path(rest: &str) -> (&str, &str) {
    let Some(after_slashes) = rest.strip_prefix("//") else {
        return ("", rest);
    };
    if after_slashes.starts_with('/') {
        return ("", after_slashes);
    }
    match after_slashes.find('/') {
        Some(index) => (&after_slashes[..index], &after_slashes[index..]),
        None => (after_slashes, ""),
    }
}

fn first_query_or_fragment(rest: &str) -> Option<Qwen25VlChatFileUrlError> {
    let query = rest.find('?');
    let fragment = rest.find('#');
    match (query, fragment) {
        (None, None) => None,
        (Some(query), Some(fragment)) if query < fragment => {
            Some(Qwen25VlChatFileUrlError::QueryString)
        }
        (Some(_), None) => Some(Qwen25VlChatFileUrlError::QueryString),
        (Some(_), Some(_)) | (None, Some(_)) => Some(Qwen25VlChatFileUrlError::Fragment),
    }
}

fn percent_decode(input: &str) -> Result<String, Qwen25VlChatFileUrlError> {
    let bytes = input.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        if index + 2 >= bytes.len() {
            return Err(Qwen25VlChatFileUrlError::TruncatedPercent);
        }
        let high = hex_nibble(bytes[index + 1])?;
        let low = hex_nibble(bytes[index + 2])?;
        let byte = (high << 4) | low;
        if byte == 0 {
            return Err(Qwen25VlChatFileUrlError::NulByte);
        }
        decoded.push(byte);
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| Qwen25VlChatFileUrlError::InvalidUtf8)
}

fn hex_nibble(byte: u8) -> Result<u8, Qwen25VlChatFileUrlError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(Qwen25VlChatFileUrlError::InvalidPercent),
    }
}

fn uri_scheme(url: &str) -> Option<&str> {
    let bytes = url.as_bytes();
    let first = *bytes.first()?;
    if !first.is_ascii_alphabetic() {
        return None;
    }
    let mut end = 1;
    while end < bytes.len() {
        match bytes[end] {
            b':' => return Some(&url[..end]),
            byte if byte.is_ascii_alphanumeric()
                || byte == b'+'
                || byte == b'-'
                || byte == b'.' =>
            {
                end += 1;
            }
            _ => return None,
        }
    }
    None
}

fn media_url_scheme(url: &str) -> String {
    url.split_once(':')
        .map(|(scheme, _)| scheme.to_string())
        .unwrap_or_else(|| url.to_string())
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
    use std::io::Write;
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
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
    fn image_data_url(format: image::ImageFormat, mime: &str) -> String {
        use base64::Engine as _;
        use image::{Rgb, RgbImage};
        use std::io::Cursor;
        let mut bytes = Cursor::new(Vec::new());
        RgbImage::from_pixel(1, 1, Rgb([7, 8, 9]))
            .write_to(&mut bytes, format)
            .unwrap();
        format!(
            "data:image/{mime};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes.get_ref())
        )
    }

    fn png_data_url() -> String {
        image_data_url(image::ImageFormat::Png, "png")
    }

    fn jpeg_data_url() -> String {
        image_data_url(image::ImageFormat::Jpeg, "jpeg")
    }

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

    fn file_url(path: &std::path::Path) -> String {
        format!("file://{}", path.display())
    }

    fn openai_image_url_body(url: &str) -> String {
        let url = serde_json::to_string(url).unwrap();
        format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"hello"}},{{"type":"image_url","image_url":{{"url":{url}}}}}]}}]}}"#
        )
    }

    fn openai_image_only_body(url: &str) -> String {
        let url = serde_json::to_string(url).unwrap();
        format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":{url}}}}}]}}]}}"#
        )
    }

    struct LoopbackServer {
        base_url: String,
        stop: mpsc::Sender<()>,
        thread: Option<thread::JoinHandle<()>>,
    }

    impl LoopbackServer {
        fn start<H>(handler: H) -> Self
        where
            H: Fn(&str) -> Vec<u8> + Send + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let address = listener.local_addr().unwrap();
            let (stop, stopped) = mpsc::channel();
            let thread = thread::spawn(move || {
                loop {
                    if stopped.try_recv().is_ok() {
                        break;
                    }
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            stream
                                .set_read_timeout(Some(Duration::from_secs(2)))
                                .unwrap();
                            let path = read_http_path(&mut stream);
                            stream.write_all(&handler(&path)).unwrap();
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(1));
                        }
                        Err(error) => panic!("loopback HTTP accept failed: {error}"),
                    }
                }
            });
            Self {
                base_url: format!("http://{address}"),
                stop,
                thread: Some(thread),
            }
        }

        fn url(&self, path: &str) -> String {
            format!("{}{path}", self.base_url)
        }
    }

    impl Drop for LoopbackServer {
        fn drop(&mut self) {
            let _ = self.stop.send(());
            if let Some(thread) = self.thread.take() {
                thread.join().unwrap();
            }
        }
    }

    fn read_http_path(stream: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut chunk = [0u8; 1024];
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut chunk).unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
        }
        let request = std::str::from_utf8(&request).unwrap();
        request
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .unwrap()
            .to_string()
    }

    fn http_response(status: &str, headers: &str, body: &[u8]) -> Vec<u8> {
        let head = format!(
            "HTTP/1.1 {status}\r\nConnection: close\r\n{headers}Content-Length: {}\r\n\r\n",
            body.len()
        );
        let mut response = head.into_bytes();
        response.extend_from_slice(body);
        response
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
    fn qwen25_vl_chat_openai_json_decodes_png_data_url_and_rejects_video() {
        let dir = chat_fixture("openai", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"hello"}},{{"type":"image_url","image_url":{{"url":"{}"}}}}]}}]}}"#,
            png_data_url()
        );
        let actual = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &body,
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_eq!(actual.media_order(), [MropeVisualKind::Image]);
        let png_image = crate::multimodal::qwen_vl_image::decode_qwen2_5_vl_image(
            &base64::engine::general_purpose::STANDARD
                .decode(png_data_url().rsplit_once(',').unwrap().1)
                .unwrap(),
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &actual,
            &expected_prompt("user", &format!("hello{}", vision_run(IMAGE_PAD))),
            vec![Qwen25VlRawMediaInput::Image(png_image)],
            None,
            &mut no_video_sources,
        );

        let text_only = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            r#"{"messages":[{"role":"user","content":"hello"}]}"#,
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &text_only,
            &expected_prompt("user", "hello"),
            vec![],
            None,
            &mut no_video_sources,
        );

        let jpeg_body = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":"{}"}}}}]}}]}}"#,
            jpeg_data_url()
        );
        let jpeg = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &jpeg_body,
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_eq!(jpeg.media_order(), [MropeVisualKind::Image]);

        let video = r#"{"messages":[{"role":"user","content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,AA"}}]}]}"#;
        let err = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-video"),
            video,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlChatRequestError::UnsupportedVideoJson {
                message_index: 0,
                part_index: 0
            }
        ));

        let video_type =
            r#"{"messages":[{"role":"user","content":[{"type":"video","video":"x"}]}]}"#;
        let err = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-video-type"),
            video_type,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlChatRequestError::UnsupportedVideoJson {
                message_index: 0,
                part_index: 0
            }
        ));

        let hf_image = r#"{"messages":[{"role":"user","content":[{"type":"image"}]}]}"#;
        let err = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-hf-image"),
            hf_image,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Qwen25VlChatRequestError::UnsupportedContentType {
                message_index: 0,
                part_index: 0,
                content_type
            } if content_type == "image"
        ));

        let missing_messages = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-messages"),
            "{}",
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            missing_messages,
            Qwen25VlChatRequestError::MissingMessages
        ));
        let json = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-json"),
            "{",
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(json, Qwen25VlChatRequestError::Json(_)));
        let missing_role = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-role"),
            r#"{"messages":[{"content":"hello"}]}"#,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            missing_role,
            Qwen25VlChatRequestError::MissingRole { message_index: 0 }
        ));
        let empty_role = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-empty-role"),
            r#"{"messages":[{"role":"","content":"hello"}]}"#,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            empty_role,
            Qwen25VlChatRequestError::EmptyRole { message_index: 0 }
        ));
        let missing_content = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-content"),
            r#"{"messages":[{"role":"user"}]}"#,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            missing_content,
            Qwen25VlChatRequestError::MissingContent { message_index: 0 }
        ));
        let missing_text = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-text"),
            r#"{"messages":[{"role":"user","content":[{"type":"text"}]}]}"#,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            missing_text,
            Qwen25VlChatRequestError::MissingText {
                message_index: 0,
                part_index: 0
            }
        ));
        let missing_url = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-image-url"),
            r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{}}]}]}"#,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            missing_url,
            Qwen25VlChatRequestError::MissingImageUrl {
                message_index: 0,
                part_index: 0
            }
        ));
    }

    #[test]
    fn qwen25_vl_chat_loads_loopback_http_png_jpeg_and_five_redirects() {
        let png_bytes = encoded_image_bytes(image::ImageFormat::Png, 21);
        let jpeg_bytes = encoded_image_bytes(image::ImageFormat::Jpeg, 31);
        let served_png = png_bytes.clone();
        let served_jpeg = jpeg_bytes.clone();
        let server = LoopbackServer::start(move |path| {
            if path == "/image.txt" || path == "/redirect/0" {
                return http_response("200 OK", "Content-Type: text/plain\r\n", &served_png);
            }
            if path == "/photo" {
                return http_response(
                    "200 OK",
                    "Content-Type: application/octet-stream\r\n",
                    &served_jpeg,
                );
            }
            if let Some(remaining) = path.strip_prefix("/redirect/") {
                let remaining: usize = remaining.parse().unwrap();
                let location = format!("Location: /redirect/{}\r\n", remaining - 1);
                return http_response("302 Found", &location, &[]);
            }
            http_response("404 Not Found", "", &[])
        });
        let dir = chat_fixture("remote_http", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let mut no_video_sources: [RecordingFrameSource; 0] = [];

        for (path, expected_bytes) in [
            ("/image.txt", png_bytes.as_slice()),
            ("/photo", jpeg_bytes.as_slice()),
            ("/redirect/5", png_bytes.as_slice()),
        ] {
            let actual = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
                &dir,
                &openai_image_url_body(&server.url(path)),
                None,
                &mut no_video_sources,
            )
            .unwrap();
            let expected =
                crate::multimodal::qwen_vl_image::decode_qwen2_5_vl_image(expected_bytes).unwrap();
            assert_matches_raw(
                &dir,
                &actual,
                &expected_prompt("user", &format!("hello{}", vision_run(IMAGE_PAD))),
                vec![Qwen25VlRawMediaInput::Image(expected)],
                None,
                &mut no_video_sources,
            );
        }
    }

    #[test]
    fn qwen25_vl_chat_routes_https_through_injected_bounded_decode_path() {
        let dir = chat_fixture("remote_https", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let bytes = encoded_image_bytes(image::ImageFormat::Png, 41);
        let expected = crate::multimodal::qwen_vl_image::decode_qwen2_5_vl_image(&bytes).unwrap();
        let url = "HTTPS://example.invalid/no-image-suffix";
        let mut fetched = Vec::new();
        let mut fetch = |actual_url: &str| {
            fetched.push(actual_url.to_string());
            Ok(bytes.clone())
        };
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let actual =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
                &dir,
                &openai_image_url_body(url),
                None,
                &mut no_video_sources,
                &mut fetch,
            )
            .unwrap();
        assert_eq!(fetched, [url]);
        assert_matches_raw(
            &dir,
            &actual,
            &expected_prompt("user", &format!("hello{}", vision_run(IMAGE_PAD))),
            vec![Qwen25VlRawMediaInput::Image(expected)],
            None,
            &mut no_video_sources,
        );
    }

    #[test]
    fn qwen25_vl_chat_reports_bounded_remote_failures_with_context() {
        let server = LoopbackServer::start(|path| match path {
            "/status" => http_response("503 Service Unavailable", "", b"not an image"),
            "/declared" => format!(
                "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
                QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES as u64 + 1
            )
            .into_bytes(),
            "/loop" => http_response("302 Found", "Location: /loop\r\n", &[]),
            "/bad-redirect" => {
                http_response("302 Found", "Location: file:///tmp/image.png\r\n", &[])
            }
            _ => http_response("404 Not Found", "", &[]),
        });
        let mut no_video_sources: [RecordingFrameSource; 0] = [];
        let missing_dir = Path::new("/missing-qwen25-vl-chat-remote-errors");

        let status_url = server.url("/status");
        let status = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_url_body(&status_url),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            status,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 1,
                url,
                source: Qwen25VlChatRemoteImageError::Status { status: 503 }
            } if url == status_url
        ));

        let declared_url = server.url("/declared");
        let declared = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body(&declared_url),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            declared,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 0,
                url,
                source: Qwen25VlChatRemoteImageError::DeclaredTooLarge {
                    declared,
                    limit: QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES
                }
            } if url == declared_url && declared == QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES as u64 + 1
        ));

        let loop_url = server.url("/loop");
        let redirect_overflow =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
                missing_dir,
                &openai_image_only_body(&loop_url),
                None,
                &mut no_video_sources,
            )
            .unwrap_err();
        assert!(matches!(
            redirect_overflow,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 0,
                url,
                source: Qwen25VlChatRemoteImageError::Request(_)
            } if url == loop_url
        ));

        let bad_redirect_url = server.url("/bad-redirect");
        let bad_redirect = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body(&bad_redirect_url),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            bad_redirect,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 0,
                url,
                source: Qwen25VlChatRemoteImageError::Status { status: 302 }
            } if url == bad_redirect_url
        ));

        let malformed_url = "http://[";
        let request = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body(malformed_url),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            request,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 0,
                url,
                source: Qwen25VlChatRemoteImageError::Request(_)
            } if url == malformed_url
        ));

        let invalid_url = "https://example.invalid/not-an-image";
        let mut invalid_fetch = |_url: &str| Ok(b"not an image".to_vec());
        let decode =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
                missing_dir,
                &openai_image_url_body(invalid_url),
                None,
                &mut no_video_sources,
                &mut invalid_fetch,
            )
            .unwrap_err();
        assert!(matches!(
            decode,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 1,
                url,
                source: Qwen25VlChatRemoteImageError::Decode(_)
            } if url == invalid_url
        ));

        let streamed_url = "https://example.invalid/streamed-too-large";
        let mut streamed_fetch = |_url: &str| {
            Err(Qwen25VlChatRemoteImageError::StreamTooLarge {
                limit: QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES,
            })
        };
        let streamed =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
                missing_dir,
                &openai_image_url_body(streamed_url),
                None,
                &mut no_video_sources,
                &mut streamed_fetch,
            )
            .unwrap_err();
        assert!(matches!(
            streamed,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 1,
                url,
                source: Qwen25VlChatRemoteImageError::StreamTooLarge {
                    limit: QWEN2_5_VL_MAX_REMOTE_IMAGE_BYTES
                }
            } if url == streamed_url
        ));

        let body_url = "https://example.invalid/body-read";
        let mut body_fetch = |_url: &str| {
            Err(Qwen25VlChatRemoteImageError::Body(std::io::Error::other(
                "fixture read failure",
            )))
        };
        let body = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
            missing_dir,
            &openai_image_only_body(body_url),
            None,
            &mut no_video_sources,
            &mut body_fetch,
        )
        .unwrap_err();
        assert!(matches!(
            body,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 0,
                url,
                source: Qwen25VlChatRemoteImageError::Body(_)
            } if url == body_url
        ));

        let two_remote = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":{}}}}},{{"type":"image_url","image_url":{{"url":{}}}}}]}}]}}"#,
            serde_json::to_string("https://example.invalid/first").unwrap(),
            serde_json::to_string("https://example.invalid/second").unwrap()
        );
        let mut fetch_calls = 0;
        let mut ordered_fetch = |_url: &str| {
            fetch_calls += 1;
            Err(Qwen25VlChatRemoteImageError::Status { status: 502 })
        };
        let first =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
                missing_dir,
                &two_remote,
                None,
                &mut no_video_sources,
                &mut ordered_fetch,
            )
            .unwrap_err();
        assert!(matches!(
            first,
            Qwen25VlChatRequestError::RemoteImage {
                message_index: 0,
                part_index: 0,
                url,
                source: Qwen25VlChatRemoteImageError::Status { status: 502 }
            } if url == "https://example.invalid/first"
        ));
        assert_eq!(fetch_calls, 1);
    }

    #[test]
    fn qwen25_vl_chat_remote_body_reader_enforces_stream_cap_and_read_errors() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("fixture read failure"))
            }
        }

        assert!(matches!(
            read_remote_image_body(std::io::Cursor::new([]), Some(5), 4),
            Err(Qwen25VlChatRemoteImageError::DeclaredTooLarge {
                declared: 5,
                limit: 4
            })
        ));
        assert!(matches!(
            read_remote_image_body(std::io::Cursor::new([1, 2, 3, 4, 5]), None, 4),
            Err(Qwen25VlChatRemoteImageError::StreamTooLarge { limit: 4 })
        ));
        assert!(matches!(
            read_remote_image_body(std::io::Cursor::new([1, 2, 3, 4, 5]), Some(2), 4),
            Err(Qwen25VlChatRemoteImageError::StreamTooLarge { limit: 4 })
        ));
        assert!(matches!(
            read_remote_image_body(FailingReader, None, 4),
            Err(Qwen25VlChatRemoteImageError::Body(_))
        ));
        assert_eq!(
            read_remote_image_body(std::io::Cursor::new([1, 2, 3, 4]), None, 4).unwrap(),
            [1, 2, 3, 4]
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

        let url = png_data_url();
        let many_parts = (0..=QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS)
            .map(|_| format!(r#"{{"type":"image_url","image_url":{{"url":"{url}"}}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let openai_cap = format!(r#"{{"messages":[{{"role":"user","content":[{many_parts}]}}]}}"#);
        let openai_cap_err = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            std::path::Path::new("/missing-qwen25-vl-chat-openai-cap"),
            &openai_cap,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            openai_cap_err,
            Qwen25VlChatRequestError::TooManyMediaItems {
                actual: 65,
                limit: 64
            }
        ));
        let role_before_cap = format!(
            r#"{{"messages":[{{"content":"x"}},{{"role":"user","content":[{many_parts}]}}]}}"#
        );
        let role_before_cap_err =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
                std::path::Path::new("/missing-qwen25-vl-chat-role-before-cap"),
                &role_before_cap,
                None,
                &mut no_video_sources,
            )
            .unwrap_err();
        assert!(matches!(
            role_before_cap_err,
            Qwen25VlChatRequestError::MissingRole { message_index: 0 }
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
    fn qwen25_vl_chat_loads_local_filesystem_and_file_url_images() {
        let dir = chat_fixture("filesystem", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        let media_dir = dir.join("media");
        fs::create_dir_all(&media_dir).unwrap();
        let png_path = write_image_file(&media_dir, "pixel.png", image::ImageFormat::Png, 7);
        let jpeg_path = write_image_file(&media_dir, "pixel.jpeg", image::ImageFormat::Jpeg, 9);
        let spaced = write_image_file(&media_dir, "hello world.png", image::ImageFormat::Png, 11);
        let png_image = decode_image_file(&png_path);
        let jpeg_image = decode_image_file(&jpeg_path);
        let spaced_image = decode_image_file(&spaced);
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

        let openai_path = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(png_path.to_str().unwrap()),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &openai_path,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image.clone())],
            None,
            &mut no_video_sources,
        );

        let openai_file = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&file_url(&png_path)),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &openai_file,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image.clone())],
            None,
            &mut no_video_sources,
        );

        let uppercase = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&format!("FILE://{}", png_path.display())),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &uppercase,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image.clone())],
            None,
            &mut no_video_sources,
        );

        let localhost = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&format!("file://localhost{}", png_path.display())),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &localhost,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image.clone())],
            None,
            &mut no_video_sources,
        );

        let encoded = file_url(&spaced).replace(' ', "%20");
        let percent = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&encoded),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &percent,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(spaced_image)],
            None,
            &mut no_video_sources,
        );

        let jpeg = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&file_url(&jpeg_path)),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &jpeg,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(jpeg_image.clone())],
            None,
            &mut no_video_sources,
        );

        let jpeg_typed = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: &dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::Text("hello"),
                        Qwen25VlChatContent::ImagePath(&jpeg_path),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &jpeg_typed,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(jpeg_image.clone())],
            None,
            &mut no_video_sources,
        );

        let jpeg_openai_path = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(jpeg_path.to_str().unwrap()),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &jpeg_openai_path,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(jpeg_image)],
            None,
            &mut no_video_sources,
        );

        let no_authority = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&format!("file:{}", png_path.display())),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &no_authority,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image.clone())],
            None,
            &mut no_video_sources,
        );

        let localhost_host = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            &dir,
            &openai_image_url_body(&format!("file://LOCALHOST{}", png_path.display())),
            None,
            &mut no_video_sources,
        )
        .unwrap();
        assert_matches_raw(
            &dir,
            &localhost_host,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(png_image)],
            None,
            &mut no_video_sources,
        );

        let relative_openai_name = format!(
            "poot_qwen25_vl_chat_rel_openai_{}_{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        fs::write(
            &relative_openai_name,
            encoded_image_bytes(image::ImageFormat::Png, 17),
        )
        .unwrap();
        let relative_openai = CwdImage {
            name: relative_openai_name.clone(),
        };
        let relative_openai_image = decode_image_file(Path::new(&relative_openai.name));
        let relative_openai_actual =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
                &dir,
                &openai_image_url_body(&relative_openai.name),
                None,
                &mut no_video_sources,
            )
            .unwrap();
        assert_matches_raw(
            &dir,
            &relative_openai_actual,
            &rendered,
            vec![Qwen25VlRawMediaInput::Image(relative_openai_image)],
            None,
            &mut no_video_sources,
        );
        drop(relative_openai);
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

        let openai_empty = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body(""),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            openai_empty,
            Qwen25VlChatRequestError::EmptyImagePath {
                message_index: 0,
                part_index: 0
            }
        ));

        let host = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file://example.com/a.png"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            host,
            Qwen25VlChatRequestError::UnsupportedFileUrlHost {
                message_index: 0,
                part_index: 0,
                host
            } if host == "example.com"
        ));

        let relative_file = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:relative.png"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            relative_file,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::RelativePath
            }
        ));

        let query = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:///tmp/a.png?x=1"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            query,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::QueryString
            }
        ));

        let fragment = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:///tmp/a.png#x"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            fragment,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::Fragment
            }
        ));

        let percent = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:///tmp/%zz.png"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            percent,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::InvalidPercent
            }
        ));

        let truncated = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:///tmp/%2"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            truncated,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::TruncatedPercent
            }
        ));

        let nul = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:///tmp/%00.png"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            nul,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::NulByte
            }
        ));

        let utf8 = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:///tmp/%FF.png"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            utf8,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::InvalidUtf8
            }
        ));

        let empty_file = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file:"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            empty_file,
            Qwen25VlChatRequestError::InvalidFileUrl {
                message_index: 0,
                part_index: 0,
                source: Qwen25VlChatFileUrlError::EmptyPath
            }
        ));

        let loopback = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &openai_image_only_body("file://127.0.0.1/a.png"),
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            loopback,
            Qwen25VlChatRequestError::UnsupportedFileUrlHost {
                message_index: 0,
                part_index: 0,
                host
            } if host == "127.0.0.1"
        ));

        let dir = chat_fixture("filesystem_errors", "qwen2_5_vl", Some(CHAT_TEMPLATE));
        fs::write(dir.join("not-an-image.png"), b"not-png").unwrap();
        let decode_path = dir.join("not-an-image.png");
        let decode = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: missing_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::ImagePath(&decode_path),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            decode,
            Qwen25VlChatRequestError::ImageFile {
                message_index: 0,
                part_index: 0,
                path,
                source: Qwen25VlChatImageFileError::Decode(_)
            } if path == decode_path
        ));

        let directory = assemble_qwen2_5_vl_chat_request_mrope_model_input(
            Qwen25VlChatRequestMropeModelInputRequest {
                model_dir: missing_dir,
                messages: &[Qwen25VlChatMessage {
                    role: "user",
                    content: Qwen25VlChatMessageContent::Parts(vec![
                        Qwen25VlChatContent::ImagePath(&dir),
                    ]),
                }],
                fps: None,
                video_sources: &mut no_video_sources,
            },
        )
        .unwrap_err();
        assert!(matches!(
            directory,
            Qwen25VlChatRequestError::ImageFile {
                message_index: 0,
                part_index: 0,
                path,
                source: Qwen25VlChatImageFileError::Read(_)
            } if path == *dir
        ));

        let missing_before_https = format!(
            r#"{{"messages":[{{"role":"user","content":[{{"type":"image_url","image_url":{{"url":{}}}}},{{"type":"image_url","image_url":{{"url":"https://example.com/a.png"}}}}]}}]}}"#,
            serde_json::to_string(missing_path.to_str().unwrap()).unwrap()
        );
        let mut remote_calls = 0;
        let mut remote_fetch = |_url: &str| {
            remote_calls += 1;
            Ok(encoded_image_bytes(image::ImageFormat::Png, 1))
        };
        let precedence =
            assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json_with_remote(
                missing_dir,
                &missing_before_https,
                None,
                &mut no_video_sources,
                &mut remote_fetch,
            )
            .unwrap_err();
        assert!(matches!(
            precedence,
            Qwen25VlChatRequestError::ImageFile {
                message_index: 0,
                part_index: 0,
                path,
                source: Qwen25VlChatImageFileError::Read(_)
            } if path == missing_path
        ));
        assert_eq!(remote_calls, 0);

        let png_path = write_image_file(&dir, "cap.png", image::ImageFormat::Png, 1);
        let png_url = serde_json::to_string(&file_url(&png_path)).unwrap();
        let many_parts = (0..=QWEN2_5_VL_MAX_RAW_MEDIA_ITEMS)
            .map(|_| format!(r#"{{"type":"image_url","image_url":{{"url":{png_url}}}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let cap_body = format!(r#"{{"messages":[{{"role":"user","content":[{many_parts}]}}]}}"#);
        let cap = assemble_qwen2_5_vl_chat_request_mrope_model_input_from_openai_json(
            missing_dir,
            &cap_body,
            None,
            &mut no_video_sources,
        )
        .unwrap_err();
        assert!(matches!(
            cap,
            Qwen25VlChatRequestError::TooManyMediaItems {
                actual: 65,
                limit: 64
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

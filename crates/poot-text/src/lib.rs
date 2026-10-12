//! Text services of a checkpoint: tokenization, chat rendering and constrained decoding.
//!
//! A leaf crate (rank 0): it holds the tokenizer, the chat templates and the guided constraint builders, and
//! reads no model weights and opens no device. A GGUF's vocabulary reaches it as plain data
//! ([`GgufVocab`]); the checkpoint reader that fills it lives in `poot-llm`.

/// `TextError::Msg(format!(...))`. Same argument shape as `anyhow::anyhow!(...)`. Defined at the
/// crate root so it is textually in scope for the child modules declared below.
macro_rules! err {
    ($($arg:tt)*) => { $crate::TextError::Msg(format!($($arg)*)) };
}

/// `return Err(TextError::Msg(format!(...)))`. Same argument shape as `bail!(...)`.
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::TextError::Msg(format!($($arg)*))) };
}

mod chat;
mod chat_format;
mod error;
pub mod grammar;
mod guided;
mod tokenize;

pub use chat::{
    ChatTemplateOverride, CheckedChatTemplate, RenderedChat, Speaker, read_tokenizer_config_chat,
    render_jinja_value,
};
pub use chat_format::ChatFormat;
pub use error::{ChatTemplateError, Result, TextError};
pub use guided::{Constraint, JsonAcceptor};
pub use tokenize::{ChatTemplate, EncoderTokens, GgufVocab, TextCodec};

//! The error of the text services ([`crate::TextCodec`]: tokenization, chat rendering and the guided
//! constraint builders), which need no model and no device. Everything that runs a model reports
//! [`crate::driver::error::DriverError`].

/// The text services' result type.
pub type Result<T, E = TextError> = std::result::Result<T, E>;

/// Why a tokenizer, a chat template or a guided constraint could not be built or used.
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// A guided-decoding schema or a chat-template value did not parse as JSON.
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    /// A tokenizer, template or constraint the services refuse, with the reason: a validation
    /// failure, or an external error that is not a `std::error::Error` (`tokenizers`,
    /// `regex-automata`) formatted through [`err!`].
    #[error("{0}")]
    Msg(String),
}

/// Same `.context()` on an `Option` as anyhow's, turning `None` into a [`TextError::Msg`].
pub trait OptionExt<T> {
    fn context<C: std::fmt::Display>(self, ctx: C) -> Result<T>;
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn context<C: std::fmt::Display>(self, ctx: C) -> Result<T> {
        self.ok_or_else(|| TextError::Msg(ctx.to_string()))
    }
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| TextError::Msg(f().to_string()))
    }
}

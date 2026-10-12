//! The error of the text services ([`crate::TextCodec`]: tokenization, chat rendering and the guided
//! constraint builders), which need no model and no device. `poot-llm` wraps it in its own driver error.

/// The text services' result type.
pub type Result<T, E = TextError> = std::result::Result<T, E>;

/// Why a tokenizer, a chat template or a guided constraint could not be built or used.
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// A guided-decoding schema or a chat-template value did not parse as JSON.
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    /// A chat template given in place of the checkpoint's own is unusable.
    #[error(transparent)]
    ChatTemplate(#[from] ChatTemplateError),
    /// A tokenizer, template or constraint the services refuse, with the reason: a validation
    /// failure, or an external error that is not a `std::error::Error` (`tokenizers`,
    /// `regex-automata`) formatted through [`err!`].
    #[error("{0}")]
    Msg(String),
}

/// Why a chat template the caller supplied in place of the checkpoint's own is refused.
#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    /// The value is not jinja (it has no `{{`, `{%` or `{#`), so it names a built-in template; poot has
    /// none, only templates given as text or as a file.
    #[error(
        "`{name}` is not a jinja template, and poot has no built-in chat templates by name: pass the template text, or a template file"
    )]
    UnknownBuiltin {
        /// The value that was given as a template name.
        name: String,
    },
    /// The template file could not be read.
    #[error("chat template file {path}: {source}")]
    Unreadable {
        /// The template file that could not be read.
        path: std::path::PathBuf,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The template does not compile.
    #[error("the chat template does not compile: {0}")]
    Invalid(String),
    /// The template compiled but failed to render a conversation (`raise_exception`, an undefined
    /// access). An override never falls back to the family's hardcoded format.
    #[error("the chat template failed to render the conversation: {0}")]
    Render(String),
}

/// Same `.context()` on an `Option` as anyhow's, turning `None` into a [`TextError::Msg`].
pub(crate) trait OptionExt<T> {
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

//! Typed errors for the poot-llm library (thiserror in libraries, anyhow only in binaries).
//! Every public Runner / loader / encoder fallible API returns [`Result`] (alias for
//! `Result<T, RunnerError>`). The binaries (`qwen2-generate`, `poot-caption`) and poot-serve keep anyhow at
//! the top level and consume `RunnerError` through its `std::error::Error` impl.
//!
//! Variants split load (safetensors / tokenizer / config), eval (shape / executor), gpu, a contextualized
//! wrapper and a plain message. The `#[from]` variants let a bare `?` carry the typed error;
//! [`ResultExt::context`] adds a human string like anyhow's `.context()` without erasing the type.

/// The library result type. Defaults its error to [`RunnerError`].
pub type Result<T, E = RunnerError> = std::result::Result<T, E>;

/// Spec 384: an error enum never carries another poot error enum by value. Every cause below is boxed, so
/// this type's size does not depend on the cause chain's depth; that matters because `Result<T>` defaults its
/// error to this type. Foreign errors already one pointer wide (`std::io::Error`, `serde_json::Error`) and the
/// two `Box<dyn Error>` sources are not boxed. `thiserror` cannot box a `#[from]` field, so each boxed cause
/// has a hand-written `From` below and `?` keeps working; `#[error("{0}")]` prints the cause verbatim:
///
/// ```
/// let wrapped: poot_llm::RunnerError = poot_llm::MropeBindError::MissingSlot.into();
/// assert_eq!(wrapped.to_string(), "mRoPE graph has no Slot::MropePosition input");
/// ```
///
/// Naming a boxed variant directly needs the box; deleting one of those impls stops the conversion above
/// compiling:
///
/// ```compile_fail,E0308
/// let wrong = poot_llm::RunnerError::Mrope(poot_llm::MropeBindError::MissingSlot);
/// ```
///
/// All the ways a poot-llm operation can fail.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    /// Loading weights / config / a tokenizer off disk (poot-load).
    #[error("{0}")]
    Load(#[source] Box<poot_load::LoadError>),
    /// Evaluating a graph on the CPU eager executor (shape, dtype, unsupported op).
    #[error("{0}")]
    Eval(#[source] Box<poot_eval::EvalError>),
    /// Precompiling / running a graph on the PTX (NVIDIA) backend.
    #[error("{0}")]
    Ptx(#[source] Box<poot_ptx_gpu::PtxGpuError>),
    /// Running a graph on the ROCm/HSA (AMD) backend.
    #[cfg(feature = "rocm")]
    #[error("{0}")]
    Rocm(#[source] Box<poot_rocm_gpu::RocmGpuError>),
    /// A traced graph that cannot take the checkpoint's packed storage (card 545a).
    #[error("{0}")]
    PackedBind(#[source] Box<poot_graph_plan::PackedBindError>),
    /// Typed Card 152 mRoPE position-slot validation.
    #[error("{0}")]
    Mrope(#[source] Box<crate::multimodal::mrope::MropeBindError>),
    /// Non-finite logits reached the sampler (ADR-0101 decision 4).
    #[error("{0}")]
    Sampler(#[source] Box<crate::core::sampler::SamplerFault>),
    /// A model-owned validation adapter error, retained as a typed source without coupling the runner
    /// to any model family's validation-id table.
    #[error("model execution validation failed: {source}")]
    ModelValidation {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Filesystem / IO.
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// JSON (config or guided-decoding schema) parsing.
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    /// An underlying error annotated with a human context string (the `.context()` path).
    #[error("{context}: {source}")]
    Context {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// A checkpoint this Runner deliberately does not serve. Card 364a: the generic Runner binds `Tensor` weights
    /// and consumes a `Graph` with no validation channel, so it cannot execute a model whose graph is packed and
    /// validation-bearing; it fails closed rather than trace and produce wrong output.
    #[error("{model_type} is not supported by the generic Runner: {reason}")]
    UnsupportedModel {
        model_type: String,
        reason: &'static str,
    },
    /// A checkpoint of a registered family: it loads through `driver::ModelHandle` and runs on the
    /// driver, never on the Runner.
    #[error(
        "{family} is a registered family: load it with driver::ModelHandle and run it on the driver"
    )]
    RegisteredFamily {
        family: poot_models::model::FamilyKey,
    },
    /// A plain message: validation failures (`bail!`) and external errors that are not
    /// `std::error::Error` (tokenizers, regex-automata) formatted via [`err!`].
    #[error("{0}")]
    Msg(String),
}

impl From<poot_graph_plan::PackedBindError> for RunnerError {
    fn from(error: poot_graph_plan::PackedBindError) -> Self {
        RunnerError::PackedBind(Box::new(error))
    }
}

impl From<poot_load::LoadError> for RunnerError {
    fn from(error: poot_load::LoadError) -> Self {
        RunnerError::Load(Box::new(error))
    }
}

impl From<poot_eval::EvalError> for RunnerError {
    fn from(error: poot_eval::EvalError) -> Self {
        RunnerError::Eval(Box::new(error))
    }
}

impl From<poot_ptx_gpu::PtxGpuError> for RunnerError {
    fn from(error: poot_ptx_gpu::PtxGpuError) -> Self {
        RunnerError::Ptx(Box::new(error))
    }
}

#[cfg(feature = "rocm")]
impl From<poot_rocm_gpu::RocmGpuError> for RunnerError {
    fn from(error: poot_rocm_gpu::RocmGpuError) -> Self {
        RunnerError::Rocm(Box::new(error))
    }
}

impl From<crate::multimodal::mrope::MropeBindError> for RunnerError {
    fn from(error: crate::multimodal::mrope::MropeBindError) -> Self {
        RunnerError::Mrope(Box::new(error))
    }
}

impl From<crate::core::sampler::SamplerFault> for RunnerError {
    fn from(error: crate::core::sampler::SamplerFault) -> Self {
        RunnerError::Sampler(Box::new(error))
    }
}

impl RunnerError {
    #[cfg(test)]
    pub(crate) fn model_validation(source: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::ModelValidation {
            source: Box::new(source),
        }
    }
}

/// `.context()` / `.with_context()` on a `Result` with an `Error` payload, mirroring anyhow's
/// ergonomics but producing a typed [`RunnerError`] rather than an erased `anyhow::Error`.
pub trait ResultExt<T> {
    fn context<C: std::fmt::Display>(self, ctx: C) -> Result<T>;
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C: std::fmt::Display>(self, ctx: C) -> Result<T> {
        self.map_err(|e| RunnerError::Context {
            context: ctx.to_string(),
            source: Box::new(e),
        })
    }
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.map_err(|e| RunnerError::Context {
            context: f().to_string(),
            source: Box::new(e),
        })
    }
}

/// Same `.context()` on an `Option`, turning `None` into a [`RunnerError::Msg`].
pub trait OptionExt<T> {
    fn context<C: std::fmt::Display>(self, ctx: C) -> Result<T>;
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T>;
}

impl<T> OptionExt<T> for Option<T> {
    fn context<C: std::fmt::Display>(self, ctx: C) -> Result<T> {
        self.ok_or_else(|| RunnerError::Msg(ctx.to_string()))
    }
    fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, f: F) -> Result<T> {
        self.ok_or_else(|| RunnerError::Msg(f().to_string()))
    }
}

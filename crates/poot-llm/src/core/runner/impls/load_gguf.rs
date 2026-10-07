use std::path::Path;

use poot_load::gguf::GgufIndex;

use super::super::{Runner, refuse_registered};
use crate::error::{Result, ResultExt, RunnerError};

impl Runner {
    /// Reads a GGUF's index and refuses the checkpoint: a registered family loads through
    /// `driver::ModelHandle` ([`RunnerError::RegisteredFamily`]) and no other architecture loads on
    /// the Runner ([`RunnerError::UnsupportedModel`]).
    pub fn load_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let g = GgufIndex::open(path.as_ref()).context("open gguf index")?;
        refuse_registered(&poot_models::registry::RawConfig::Gguf(&g))?;
        Err(RunnerError::UnsupportedModel {
            model_type: g.architecture().unwrap_or("").to_string(),
            reason: "no family loads on the Runner: every decoder family is a registered Model",
        })
    }
}

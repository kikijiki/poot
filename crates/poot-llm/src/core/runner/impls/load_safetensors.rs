use std::io::Read;
use std::path::Path;

use super::super::{Runner, refuse_registered, unsupported_safetensors_arch};
use crate::error::{Result, ResultExt, RunnerError};

impl Runner {
    /// Reads config.json from a model directory and refuses the checkpoint: a registered family loads
    /// through `driver::ModelHandle` ([`RunnerError::RegisteredFamily`]) and no other family loads on
    /// the Runner ([`RunnerError::UnsupportedModel`]).
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_impl(dir.as_ref())
    }

    pub(crate) fn load_impl(dir: &Path) -> Result<Self> {
        const MAX_CONFIG_BYTES: usize = 16 * 1024 * 1024;

        let mut raw_config_bytes = Vec::new();
        std::fs::File::open(dir.join("config.json"))
            .and_then(|file| {
                file.take(MAX_CONFIG_BYTES as u64 + 1)
                    .read_to_end(&mut raw_config_bytes)
            })
            .map_err(|error| err!("read config.json: {error}"))?;
        if raw_config_bytes.len() > MAX_CONFIG_BYTES {
            bail!("read config.json: larger than {MAX_CONFIG_BYTES} bytes");
        }
        let raw_config: serde_json::Value =
            serde_json::from_slice(&raw_config_bytes).context("parse config.json")?;
        refuse_registered(&poot_models::registry::RawConfig::HfJson {
            config: &raw_config,
            generation: None,
        })?;
        // A config with no string `model_type` is refused too: it must never fall through to another
        // family's tracer (card 190).
        let model_type = raw_config
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| RunnerError::UnsupportedModel {
                model_type: "<none>".to_string(),
                reason: "config.json has no string model_type",
            })?;
        Err(unsupported_safetensors_arch(model_type))
    }
}

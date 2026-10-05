//! Model-local metadata for the pinned `zai-org/GLM-5.3-Flash` FP8 checkpoint.
//!
//! This module owns config and checkpoint metadata plus the model-local text disposition policy and owner-load
//! handoff used inside Card 359's authenticated transaction. It does not define model execution.

use std::collections::BTreeSet;

#[cfg(test)]
use poot_quant::format::{ScaleEncoding, WeightFormat};

use ring::digest::{SHA256, digest};

use serde::Deserialize;

use crate::packed_safetensors::MixedLoadResult;
#[cfg(test)]
use crate::packed_safetensors::{
    AuthenticatedArtifactIdentity, AuthenticatedInventory, AuthenticatedSafetensorsHandleSet,
    ExactSourceOwnerCache, InventoryDecision, MixedLoadError, PackedOwnerCache, Sha256Digest,
    TensorDisposition,
};

mod classifier;
mod config;
mod text;

pub use classifier::*;
pub use config::*;
pub use text::*;

pub const GLM53_FLASH_MODEL_TYPE: &str = "glm5_next";
pub const GLM53_FLASH_TEXT_MODEL_TYPE: &str = "glm5_next_text";
pub const GLM53_FLASH_VISION_MODEL_TYPE: &str = "glm5_next_vision";
pub const GLM53_FLASH_ARCHITECTURE: &str = "Glm5NextForConditionalGeneration";
pub const GLM53_FLASH_TENSOR_COUNT: usize = 76_108;
pub const GLM53_FLASH_QUANT_EXCLUSION_COUNT: usize = 1_509;
pub const GLM53_FLASH_EXCLUSION_SHA256: &str =
    "4905d54193c2419800fb2ccc767695c4f0f64726da93f6a8ea972eac916874cb";
/// Revision of the pinned snapshot.
pub const GLM53_FLASH_REVISION: &str = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a";

#[derive(Debug, thiserror::Error)]
pub enum Glm53FlashMetadataError {
    #[error("GLM-5.3-Flash JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid GLM-5.3-Flash config: {0}")]
    InvalidConfig(String),
    #[error("GLM-5.3-Flash checked byte accounting overflowed at {field}")]
    AccountingOverflow { field: &'static str },
    #[error("GLM-5.3-Flash text inventory needs revision {expected}, got {actual}")]
    TextRevision { expected: String, actual: String },
    #[error("GLM-5.3-Flash text artifact has {field} {actual}, expected {expected}")]
    TextArtifact {
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("GLM-5.3-Flash text inventory has {actual} rows, expected {expected}")]
    TextInventoryCount { expected: usize, actual: usize },
    #[error("GLM-5.3-Flash text inventory row {name} has {field} {actual}, expected {expected}")]
    TextInventoryRow {
        name: String,
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("invalid GLM-5.3-Flash text inventory contract: {0}")]
    InvalidTextInventory(String),
}

fn sha256_hex(bytes: &[u8]) -> String {
    digest(&SHA256, bytes)
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
fn sha256_digest_hex(digest: Sha256Digest) -> String {
    digest
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &[u8] = include_bytes!("glm53_flash_data/config.json");

    fn config() -> Glm53FlashHfConfig {
        Glm53FlashHfConfig::from_slice(CONFIG).unwrap()
    }

    mod classifier;
    mod config;
}

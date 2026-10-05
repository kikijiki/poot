//! Dependency-light loading of safetensors weights and an HF `config.json`. Floating-point tensors are
//! decoded to f32 for the eager reference executor; host dequant remains the compatibility oracle. A
//! decoded tensor is a `poot_tensor::HostTensor` of its own dtype.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub mod gguf;
pub mod glm53_flash;
pub mod lora;
pub mod minimax_m2;
pub mod packed_safetensors;
pub mod qwen_vl;

// New split modules
pub mod config;
pub mod dtype;
pub mod error;
pub mod safetensors;

// Re-export main types from each module for convenience
pub use config::{QuantConfig, QuantKind, QuantScheme, Qwen2HfConfig, RopeParameters, RopeScaling};
pub use error::LoadError;

// Re-export helper functions from dtype for internal use
pub(crate) use dtype::{checked_header_end, decode, miss, parse_dtype};

#[cfg(test)]
mod tests;

//! Dependency-light loading of safetensors and GGUF weights into a `WeightStore` (every tensor's bytes as
//! stored, quantized tensors packed) and of an HF `config.json`. [`safetensors::decode_dense`] widens one dense
//! entry to f32 for the CPU oracle.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub mod gguf;
pub mod lora;
pub mod packed_safetensors;
pub mod qwen_vl;

// New split modules
pub mod config;
pub mod dtype;
pub mod error;
pub mod safetensors;

// Re-export main types from each module for convenience
pub use config::{QuantConfig, QuantKind, QuantScheme};
pub use error::LoadError;

// Re-export helper functions from dtype for internal use
pub(crate) use dtype::{checked_header_end, decode, miss, parse_dtype};

#[cfg(test)]
mod tests;

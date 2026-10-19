//! poot LLM driver: load a checkpoint and generate on the wgpu, ROCm and PTX device executors.
//!
//! A checkpoint loads as a [`driver::ModelHandle`] (its family from the registry, its weights as
//! stored, its text services) and generates through [`driver::Driver`], which also runs `Encode`
//! steps (encoder embeddings and reranking, a vision tower) and prompts with embedding rows in place
//! of tokens' own. A multimodal checkpoint's front end belongs to its registry entry; [`FrontEndSession`]
//! runs it on the towers' drivers.
//!
//! # The graph passes are sealed behind `compile` (Card 626, SC-001)
//!
//! A consumer crate cannot name a pass or sequence its own pipeline: `poot_graph_plan::passes` is
//! private (E0603), and the pre-card-626 home `poot_graph_ir::transform` no longer exists (E0433).
//!
//! ```compile_fail,E0603
//! let g = poot_graph_ir::Graph::default();
//! poot_graph_plan::passes::cse(&g);
//! ```
//!
//! ```compile_fail,E0433
//! let g = poot_graph_ir::Graph::default();
//! poot_graph_ir::transform::cse(&g);
//! ```
//!
//! # There is no `Runner`
//!
//! The per-family `Runner` that owned weights and tracer selection is gone: every family is a
//! registered `Model` on the driver.
//!
//! ```compile_fail,E0432
//! use poot_llm::Runner; // deleted
//! ```

/// Shared control result for every single-sequence text-piece callback.
///
/// [`std::ops::ControlFlow::Continue`] continues generation. [`std::ops::ControlFlow::Break`], returned
/// after consuming the current piece, stops before the next model forward; generators then return the
/// tokens produced so far rather than an error.
pub type GenerationControl = std::ops::ControlFlow<()>;

mod core;
pub mod driver;
mod multimodal;

pub use multimodal::{
    Apng, ApngError, ApngFrames, FrontEndSession, ImageDecodeError, TowerDrivers, decode_image,
    encoders, tower_options,
};

pub use core::sampler::{Sampler, SamplerFault, TokenLogprob};

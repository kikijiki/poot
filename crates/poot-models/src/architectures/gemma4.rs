//! Gemma4 remnant (cards 162, 166, 159).
//!
//! Gemma4's packed-quant tracers and config types (`trace_gemma4_prefill_kv`,
//! `gemma4_decode_trace_quant`, `Gemma4Config`/`Gemma4AttnShape`/`Gemma4MoeConfig`, and their MoE/shared-pool
//! siblings) were deleted (card 545a): Gemma4 was never wired dense and had no Runner
//! consumer (its runtime dispatch was already removed, card 595), so nothing here reaches a checkpoint
//! anymore. What remains is [`gemma4_local_window_floor`], which `crate::gpt_oss`'s decode tracer reuses
//! for its own local/global window schedule (the two architectures share the same in-graph sliding-window
//! floor arithmetic).

use poot_graph_ir::{BinOp, Builder, Scalar, Traced};

mod config;

pub use config::*;

//! A Qwen2.5-class dense decoder authored against [`poot_graph_ir::ops`]; tracing one decode token
//! produces a flat primitive graph.

use poot_graph_ir::ops::{
    alibi_mask_from_pos, attention_masked_softcap, attention_prefill_softcap, attention_softcap,
    causal_mask_from_pos, linear, rmsnorm, rope, rope_batched, rope_prefill, rope_sectioned,
    rope_sectioned_batched, softcap, swiglu,
};
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};
use poot_tensor::DType;

pub mod config;
mod decode;
pub mod model;
mod prefill;

pub use config::Qwen2Config;

// The legacy tracers held for POOT-739 (the VLM caption path and its mRoPE rows).
pub use decode::{
    LoraBatchedSpec, trace_decode, trace_decode_kv_masked, trace_decode_kv_masked_batched,
    trace_decode_mrope,
};
pub use prefill::{
    trace_prefill, trace_prefill_kv, trace_prefill_kv_embeds, trace_prefill_mrope,
    trace_qwen2_5_vl_prefill_kv_embeds,
};

#[cfg(test)]
mod tests;

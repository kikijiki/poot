//! Qwen3-MoE tracer (spec 244): the mixture-of-experts sibling of Qwen3 dense.
//!
//! Reuses [`Qwen2Config`] and the qwen3 attention block (per-head QK-norm, no qkv bias, explicit head_dim,
//! GQA + RoPE, as in [`crate::qwen2`]'s qwen3 tracers) and the shared MoE op [`poot_graph_ir::ops::moe`]
//! (also used by [`crate::granite`]). The new piece is the per-layer dense/MoE MLP switch: the HF config
//! carries `decoder_sparse_step` + `mlp_only_layers`, so some layers keep a plain dense swiglu MLP
//! (`Qwen3MoeParams::sparse_layer`, precomputed by the loader from `Qwen2HfConfig::qwen3_moe_layer_is_sparse`).
//!
//! Weight naming for routed layers matches the HF/safetensors checkpoint (checked against
//! `yujiepan/qwen3-moe-tiny-random`): `mlp.gate.weight` is the router (`[E,H]` in HF's `[out,in]`
//! convention, transposed to `[H,E]` at load). The per-expert `mlp.experts.{e}.{gate,up,down}_proj.weight`
//! tensors are separate in the checkpoint, so the loader must fuse them into the two constants this tracer
//! binds: `mlp.experts.gate_up_proj.weight` (`[E,H,2I]`, gate||up concatenated on the last axis, transposed
//! to `[in,out]` per expert; mirrors `poot_llm::gguf`'s `concat_experts_last`/`transpose_experts`) and
//! `mlp.experts.down_proj.weight` (`[E,I,H]`, transposed to `[in,out]` per expert). Dense layers use
//! `mlp.{gate,up,down}_proj.weight` at `cfg.inter` (the dense `intermediate_size`, distinct from
//! `Qwen3MoeParams::inter`, the `moe_intermediate_size`).
//!
//! The real-checkpoint test below loads and fuses the weights directly, bypassing `poot_llm::Runner`.

use poot_graph_ir::ops::{
    attention_masked, attention_prefill, linear, moe, rmsnorm, rope, rope_prefill, swiglu,
};
use poot_tensor::DType;

use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};

use crate::qwen2::Qwen2Config;

mod trace;

pub use trace::*;

#[cfg(test)]
mod tests;

//! Mixtral tracer (card 135d, `specs/261-mixtral-tracer/spec.md`) for `mistralai/Mixtral-8x7B-v0.1`. Checked
//! against the published `config.json`, `modeling_mixtral.py` (`transformers` `main` and `v4.53.3`), the 8x7B's
//! `model.safetensors.index.json`, and the `optimum-intel-internal-testing/tiny-mixtral` checkpoint.
//!
//! Mixtral is a Mistral-shaped GQA+RoPE decoder (RMSNorm, no bias) whose every layer's MLP is a top-k mixture of
//! experts; unlike `crate::qwen3moe` there is no dense/MoE switch, and `intermediate_size` is the per-expert FFN
//! width. Config (`Mixtral-8x7B-v0.1`): `num_local_experts: 8`, `num_experts_per_tok: 2`, `hidden_size: 4096`,
//! `intermediate_size: 14336`, `num_attention_heads: 32`, `num_key_value_heads: 8` (`n_rep = 4`),
//! `rms_norm_eps: 1e-5`, `rope_theta: 1000000.0`, `sliding_window: null` (not used, out of scope),
//! `tie_word_embeddings: false`.
//!
//! **Router** (`MixtralTopKRouter.forward`/`MixtralSparseMoeBlock.forward`): softmax over all experts, then
//! top-2, then renormalize the two probabilities. That equals softmax over just the 2 selected raw logits (the
//! shared normalizer cancels), which is what [`poot_graph_ir::ops::moe`] (shared with `crate::granite`/
//! `crate::qwen3moe`) computes: rank-select the top-k raw logits, mask the rest to `-1e30`, softmax over the full
//! `E` axis. There is no shared/always-on expert.
//!
//! **Expert MLP** (`MixtralBlockSparseTop2MLP`): `down(silu(w1(x)) * w3(x))`, plain SwiGLU with the older
//! `w1`/`w3`/`w2` names. Checkpoint tensors are `model.layers.{li}.block_sparse_moe.gate.weight` (router,
//! `[E,H]`) and `model.layers.{li}.block_sparse_moe.experts.{e}.{w1,w2,w3}.weight`, separate per-expert 2D
//! tensors (like `crate::qwen3moe`, unlike `crate::granite`'s fused format), so the safetensors loader
//! (`crates/poot-llm/src/runner.rs::build_weights`'s `hf.is_mixtral()` block, `fuse_mixtral_experts`) fuses them
//! into the two constants this tracer binds, as `fuse_qwen3_moe_experts` does.
//!
//! Rides on [`crate::qwen2::Qwen2Config`] for the shared attention/RoPE/norm dims plus a small
//! [`MixtralParams`] for the MoE shape. Primitive composition only (AGENTS.md): every op is an existing
//! `poot_graph_ir::ops` composition.

use poot_graph_ir::ops::{
    attention_masked, attention_prefill, linear, moe, rmsnorm, rope, rope_prefill,
};
use poot_tensor::DType;

use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, TensorType, Traced};

use crate::qwen2::Qwen2Config;

mod trace;

pub use trace::*;

#[cfg(test)]
mod tests;

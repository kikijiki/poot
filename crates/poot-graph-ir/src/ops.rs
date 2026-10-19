//! High-level op decompositions: the readable surface a model is authored against. Each has one
//! definition shared by the tracer, the oracle and every lowering (the `jax.nn` -> `lax` analog).

use crate::builder::{Builder, Traced};

use crate::error::{BuilderError, Requirement};

use crate::op::{BinOp, RedOp, UnOp};

use crate::types::{DType, Scalar, TensorType};

/// The leading axes of a packed little-endian u64 `ty`: I32 words whose last axis is `[low, high]`.
pub(crate) fn word_pair_prefix(
    operation: &'static str,
    ty: &TensorType,
) -> Result<Vec<usize>, BuilderError> {
    if ty.dtype != DType::I32 {
        return Err(BuilderError::Requirement {
            operation,
            requirement: Requirement::Dtype {
                expected: DType::I32,
                found: ty.dtype,
            },
        });
    }
    match ty.shape.split_last() {
        Some((&2, prefix)) => Ok(prefix.to_vec()),
        _ => Err(BuilderError::Requirement {
            operation,
            requirement: Requirement::TrailingWordPair {
                found: ty.shape.clone(),
            },
        }),
    }
}

/// The last axis of `shape`, or the typed refusal that `operation` needs an axis to read.
pub(crate) fn last_axis(operation: &'static str, shape: &[usize]) -> Result<usize, BuilderError> {
    shape.last().copied().ok_or(BuilderError::Requirement {
        operation,
        requirement: Requirement::Scalar,
    })
}

mod activation;
mod attention;
#[cfg(any(test, feature = "test-support"))]
mod block_top_k;
mod counter;
mod linear;
mod moe;
mod norm;
mod prefill;
mod rope;
pub mod sampling;
mod selected_attention;
mod ssm;
mod u64_hash;

#[cfg(any(test, feature = "test-support"))]
pub use activation::decay_mask_from_cumlog;
#[cfg(any(test, feature = "test-support"))]
pub use activation::decay_mask_from_gates;
pub use activation::{geglu, gelu, gelu_erf, relu, sigmoid, silu, softcap, softplus, swiglu, tanh};
pub(crate) use attention::attention_over_keys;
pub use attention::{
    AttentionKeys, LiveAttention, attention, attention_live, attention_masked,
    attention_masked_softcap, attention_prefill_with_sink, attention_softcap, paged_view,
    repeat_kv, repeat_kv_tiled,
};
#[cfg(any(test, feature = "test-support"))]
pub use block_top_k::block_top_k_packed;
pub use linear::linear;
pub use moe::{
    canonical_router_scores, moe, moe_grouped, moe_grouped_prep, moe_sparse,
    stable_descending_rank, stable_descending_rank_masked, top_k_keep_mask,
};
#[cfg(any(test, feature = "test-support"))]
pub use moe::{moe_dense, top_k_gate};
pub use norm::{layernorm, layernorm_no_bias, rmsnorm, rmsnorm_no_weight};
#[cfg(any(test, feature = "test-support"))]
pub use prefill::linear_attention_prefill;
pub use prefill::{alibi_mask_from_pos, causal_mask_from_pos, softmax};
#[cfg(any(test, feature = "test-support"))]
pub use prefill::{attention_prefill, attention_prefill_softcap, rope_prefill};
#[cfg(any(test, feature = "test-support"))]
pub use prefill::{
    cumulative_log_decay_scan, gdn_prefill_chunked, lower_triangle, scan_delta_rule_packed,
    unit_lower_triangular_inverse,
};
pub use rope::{rope, rope_partial, rope_sectioned, rope_sectioned_partial};
#[cfg(any(test, feature = "test-support"))]
pub use rope::{rope_batched, rope_sectioned_batched};
pub use selected_attention::selected_attention;
#[cfg(any(test, feature = "test-support"))]
pub use ssm::{
    causal_conv1d_decode, causal_conv1d_prefill, causal_conv1d_prefill_dilated,
    linear_attention_decode_gated, mamba2_ssd_decode,
};
pub use ssm::{causal_conv1d_windowed, gated_delta_net_decode};
#[cfg(any(test, feature = "test-support"))]
pub use ssm::{mamba2_ssd_prefill, scan_ssd_packed};
pub use u64_hash::{U64HashError, u64_mul_i32_const, u64_rem_u32, u64_xor};

#[cfg(test)]
mod tests;

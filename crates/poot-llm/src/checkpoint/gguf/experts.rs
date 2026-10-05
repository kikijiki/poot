use poot_tensor::HostTensor;

use super::permute::gather_from;

/// Interleave two same-shaped tensors along the last axis: `out[..., 2*i] = left[..., i]`,
/// `out[..., 2*i+1] = right[..., i]`; the mirror of the other MoE arms' gate||up row concatenation. Used for gpt-oss's GGUF gate/up
/// crosswalk: llama.cpp de-interleaves gpt-oss's native `gate_up_proj` (even=gate, odd=up; see
/// `poot_models::gpt_oss`) into separate `ffn_gate_exps`/`ffn_up_exps` tensors, so it must be re-interleaved
/// to match `poot_models::gpt_oss::gptoss_ffn`'s deinterleave-by-reshape. Generic over rank: works on the 3D
/// weights `[E,H,I]` and the 2D biases `[E,I]`; only the last axis is interleaved and leading axes are a
/// flattened row index.
pub fn interleave_experts_last(left: &HostTensor, right: &HostTensor) -> HostTensor {
    debug_assert_eq!(
        left.shape(),
        right.shape(),
        "interleave_experts_last needs matching shapes"
    );
    let c = *left.shape().last().expect("non-empty shape");
    let mut shape = left.shape().to_vec();
    *shape.last_mut().expect("non-empty shape") = 2 * c;
    let count = left.numel() * 2;
    gather_from(
        &[left, right],
        shape,
        (0..count).map(move |d| {
            let (row, col) = (d / (2 * c), d % (2 * c));
            (col % 2, row * c + col / 2)
        }),
    )
}

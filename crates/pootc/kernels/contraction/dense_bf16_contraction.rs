//! Card 380: activation matmul against a dense BF16 weight held in checkpoint `[N, K]` order.
//!
//! One thread owns one output element. The weight arrives as packed `u32` lanes holding the checkpoint
//! bytes verbatim: two little-endian BF16 elements per lane, element `i` in lane `i / 2` at shift
//! `(i % 2) * 16`. This is the only portable BF16 representation on wgpu (the SPIR-V emitter rejects
//! `Ty::BF16` because RADV has no `VK_KHR_shader_bfloat16`). Widening happens in-register, so no F32 copy
//! of the weight is materialized.
//!
//! Decode is `f32::from_bits(word << 16)` (card 370), exact. The activation is read as F32 and the product
//! accumulates in F32 with `k` ascending, the order `poot_eval::matmul_accumulate` uses, so a small
//! fixture matches the CPU oracle bit for bit.
//!
//! Metadata (3 u32 words): 0 rows (every leading output dim folded into M), 1 K, 2 N.
//!
//! The planner proves every index from the graph's shapes; the guards stay so an out-of-contract binding
//! reads nothing outside its buffers.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_dense_bf16_contraction(
    activation: &[f32],
    weight_words: &[u32],
    metadata: &[u32],
    output: &mut [f32],
) {
    let output_flat = thread_index();
    let rows = metadata[0] as usize;
    let input_width = metadata[1] as usize;
    let output_width = metadata[2] as usize;
    if output_flat < output.len() && output_flat < rows * output_width {
        let row = output_flat / output_width;
        let output_index = output_flat % output_width;
        let weight_end = output_index * input_width + input_width;
        let activation_end = row * input_width + input_width;
        if weight_end <= 2 * weight_words.len() && activation_end <= activation.len() {
            let mut accumulator = 0.0f32;
            let mut input_index = 0usize;
            while input_index < input_width {
                let element = output_index * input_width + input_index;
                let shift = ((element % 2) * 16) as u32;
                let word = (weight_words[element / 2] >> shift) & 0xffff;
                accumulator = accumulator
                    + activation[row * input_width + input_index] * f32::from_bits(word << 16);
                input_index = input_index + 1;
            }
            output[output_flat] = accumulator;
        }
    }
}

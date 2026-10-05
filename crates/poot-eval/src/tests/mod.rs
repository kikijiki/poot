//! Decomposition-correctness tests (oracle stage 1): each high-level op's primitive composition, run on
//! the eager executor, equals an independent direct computation within tight f32 tolerance.

mod admission_corpus;
mod alibi_attention;
mod allocation;
mod cast_quant_norm_rope;
mod composites;
mod e4m3_per_channel;
mod exact_bf16;
mod exact_dense;
mod exact_i32;
mod flash_optimize;
mod fuzz_primitives;
mod gemma2_mistral;
mod helpers;
mod i32_semantics;
mod linear_attention;
mod linear_attention_hybrid;
mod lora;
mod moe_gating;
mod moe_paged_decode;
mod moe_paged_prefill;
mod mrope;
mod oracle_corners;
mod packed_oracle_guards;
mod qwen2_pipeline;
mod qwen2_vl_mrope;
mod qwen3next_block_refs;
mod qwen3next_gdn_prefill;
mod qwen3next_ladder;
mod swa_gemma3;
mod tensor_residency;
mod validation;
mod weight_store;
mod widen;

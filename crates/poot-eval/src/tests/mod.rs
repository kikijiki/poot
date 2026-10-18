//! Decomposition-correctness tests (oracle stage 1): each high-level op's primitive composition, run on
//! the eager executor, equals an independent direct computation within tight f32 tolerance.

mod admission_corpus;
mod alibi_attention;
mod allocation;
mod attention_cache_ops;
mod block_top_k;
mod cast_quant_norm_rope;
mod composites;
mod contraction_composites;
mod e4m3_per_channel;
mod exact_bf16;
mod exact_i32;
mod fuzz_primitives;
mod helpers;
mod i32_semantics;
mod linear_attention;
mod linear_attention_hybrid;
mod moe_gating;
mod mrope;
mod oracle_corners;
mod packed_oracle_guards;
mod qwen3next_block_refs;
mod qwen3next_gdn_prefill;
mod qwen3next_ladder;
mod scan_contract;
mod select;
mod selected_attention;
mod swa_gemma3;
mod tensor_residency;
mod validation;
mod value_release;
mod weight_store;
mod widen;

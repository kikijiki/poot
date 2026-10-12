//! Model traces evaluated on the CPU oracle (`poot-eval`): decomposition-correctness rows that build their
//! graph through `Model::trace`. They live here, not in poot-eval, because a test that traces a model sits
//! above both the models and the oracle (POOT-1130); the oracle's own hand-built-graph rows stay in
//! `poot-eval/src/tests`. A pass-soundness row (a graph-plan pass over these traces) lives in the sibling
//! `*_pass_soundness.rs` binaries.

mod alibi_attention;
mod flash_optimize;
mod gemma2_mistral;
mod qwen2_pipeline;
mod swa_gemma3;

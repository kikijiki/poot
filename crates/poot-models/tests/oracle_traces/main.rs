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

/// Asserts two CPU-oracle evaluations of equivalent graphs of one model (a decode against a prefill, a chunked
/// against a one-shot prefill, a batched verify against sequential decode) agree within the tolerance of the
/// reassociating class: both run the same f32 arithmetic and differ only in how a reduction is ordered, which is
/// what that class of the tier-2 table prices (ADR-0101 decision 2). No row here picks its own number.
#[track_caller]
fn assert_reassociation_close(what: &str, got: &[f32], want: &[f32]) {
    let class = poot_graph_ir::analysis::Tier2Class::Reassociating;
    let tolerance = class.tolerance();
    if let Err(worst) = poot_test_util::check_close_unit_floor(got, want, tolerance) {
        panic!("{what}: {worst} outside the {class:?} tolerance {tolerance}");
    }
}

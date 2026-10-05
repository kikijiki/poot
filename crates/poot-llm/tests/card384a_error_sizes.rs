//! Card 384a acceptance: the six slice-1 error types stay under clippy's `result_large_err`
//! threshold, and boxing their causes changed neither what an error prints nor what
//! `Error::source` reports.
//!
//! Lives in `poot-llm` because it transitively returns all six (`RunnerError` wraps the other
//! five). The size table diagnoses which type grew; the guarantee is the clippy gate
//! (`cargo clippy -p <crate> --all-targets -- -D warnings`).

use std::error::Error;
use std::mem::size_of;

/// clippy's `large-error-threshold` default (no `clippy.toml` in this repo). The row check is `>=`:
/// clippy fires the lint at exactly 128 bytes (`RocmGpuError` measured 128 and failed `-D warnings`).
const LARGE_ERROR_THRESHOLD: usize = 128;

fn error_sizes() -> Vec<(&'static str, usize)> {
    let mut rows = vec![
        ("poot_llm::RunnerError", size_of::<poot_llm::RunnerError>()),
        (
            "poot_ptx_gpu::PtxGpuError",
            size_of::<poot_ptx_gpu::PtxGpuError>(),
        ),
    ];
    rows.extend(rocm_error_sizes());
    rows
}

/// Separate function rather than a `#[cfg]` on the `extend`, which would leave `rows` unmutated
/// without the feature and trip `unused_mut` under `-D warnings`.
#[cfg(feature = "rocm")]
fn rocm_error_sizes() -> Vec<(&'static str, usize)> {
    vec![(
        "poot_rocm_gpu::RocmGpuError",
        size_of::<poot_rocm_gpu::RocmGpuError>(),
    )]
}

#[cfg(not(feature = "rocm"))]
fn rocm_error_sizes() -> Vec<(&'static str, usize)> {
    Vec::new()
}

/// A1. Real sizes stay well clear of the threshold (max measured: 72 bytes, `PtxGpuError`); every
/// leaf type (`EvalError` 120, `PlanError` 112, `GraphValidationError` 112, ...) is under 128 alone.
/// A type crosses it only when unboxing inlines a ~120-byte cause and removes enough pointer-shaped
/// variants that the discriminant can no longer be niche-filled. Re-measure rather than estimate
/// (the size model overestimated `PlanError`/`EvalError` by 20-30 bytes).
#[test]
fn slice_one_error_types_stay_under_the_clippy_threshold() {
    let over: Vec<String> = error_sizes()
        .into_iter()
        .filter(|(_, size)| *size >= LARGE_ERROR_THRESHOLD)
        .map(|(name, size)| format!("{name} is {size} bytes"))
        .collect();
    assert!(
        over.is_empty(),
        "over clippy's {LARGE_ERROR_THRESHOLD}-byte result_large_err threshold: {over:?}"
    );
}

fn plan_cause() -> poot_graph_plan::PlanError {
    poot_graph_plan::PlanError::BadShape("test cause".to_string())
}

const PLAN_CAUSE_TEXT: &str = "bad shape for planning: test cause";

fn stage_cause() -> poot_graph_plan::CompileError {
    poot_graph_plan::CompileError::from(plan_cause())
}

/// A4. Boxing must not change output: retitle any `#[error(..)]` touched by this card and the
/// matching row fails on the literal.
#[test]
fn boxing_a_cause_does_not_change_what_an_error_prints() {
    assert_eq!(
        poot_ptx_gpu::PtxGpuError::from(plan_cause()).to_string(),
        format!("plan: {PLAN_CAUSE_TEXT}")
    );
    #[cfg(feature = "rocm")]
    assert_eq!(
        poot_rocm_gpu::RocmGpuError::from(poot_rocm_runtime::RocmError::LibraryNotFound(
            "t".to_string()
        ))
        .to_string(),
        "rocm runtime: libhsa-runtime64.so.1 not found: t"
    );
}

/// A4, source chain: `#[from]` implies `#[source]`, so dropping it while boxing a cause would cut
/// the chain with no compile error and no `Display` change. Delete one `#[source]` and the
/// corresponding link goes `None`.
#[test]
fn boxing_a_cause_does_not_cut_the_source_chain() {
    let runner = poot_llm::RunnerError::from(poot_ptx_gpu::PtxGpuError::from(plan_cause()));

    // RunnerError::Ptx prints its cause verbatim, so the outermost Display is the innermost text.
    assert_eq!(runner.to_string(), format!("plan: {PLAN_CAUSE_TEXT}"));

    let ptx = runner.source().expect("RunnerError::Ptx keeps its cause");
    assert_eq!(ptx.to_string(), format!("plan: {PLAN_CAUSE_TEXT}"));

    let plan = ptx.source().expect("PtxGpuError::Plan keeps its cause");
    assert_eq!(plan.to_string(), PLAN_CAUSE_TEXT);

    assert!(
        plan.source().is_none(),
        "PlanError::BadShape is the root of this chain"
    );
}

/// The hand-written `From` impls across this error chain keep every `?` in the workspace
/// converting in one step. Delete any one impl and this stops compiling.
#[test]
fn every_boxed_cause_still_converts_through_a_single_question_mark() {
    fn planner_fails() -> Result<(), poot_graph_plan::PlanError> {
        Err(plan_cause())
    }
    fn ptx() -> Result<(), poot_ptx_gpu::PtxGpuError> {
        planner_fails()?;
        Ok(())
    }
    fn runner() -> poot_llm::Result<()> {
        ptx()?;
        Ok(())
    }

    assert!(ptx().is_err());
    assert!(matches!(runner(), Err(poot_llm::RunnerError::Ptx(_))));
}

/// Covers every one of the boxed-cause links (`boxing_a_cause_does_not_cut_the_source_chain`
/// covers only `RunnerError -> PtxGpuError -> PlanError`): each row's parent has an explicit
/// `#[source]` field (or a field named `source`, as in `PtxGpuError::EqnFault`), so `source()`
/// returns the cause one hop away, compared by `Display` so a substituted or truncated cause is
/// also caught. Most links go through the crate's `From` impl; `PtxGpuError::EqnFault` constructs
/// the variant directly, since its field carries extra data (op/shape/grid/wg) alongside the
/// cause.
fn source_row<C, P>(
    link: &'static str,
    cause: C,
    wrap: impl FnOnce(C) -> P,
) -> (&'static str, Box<dyn Error>, Option<String>)
where
    C: Error,
    P: Error + 'static,
{
    let expected = Some(cause.to_string());
    (link, Box::new(wrap(cause)) as Box<dyn Error>, expected)
}

fn non_rocm_chain_links() -> Vec<(&'static str, Box<dyn Error>, Option<String>)> {
    vec![
        // RunnerError (poot-llm/src/error.rs): 5 links, 4 here + Rocm gated below. All
        // explicit `#[source]`.
        source_row(
            "RunnerError::Load -> poot_load::LoadError",
            poot_load::LoadError::UnknownCheckpointDtype {
                name: "t".to_string(),
            },
            poot_llm::RunnerError::from,
        ),
        source_row(
            "RunnerError::Eval -> poot_eval::EvalError",
            poot_eval::EvalError::MissingInput(0),
            poot_llm::RunnerError::from,
        ),
        source_row(
            "RunnerError::Ptx -> poot_ptx_gpu::PtxGpuError",
            poot_ptx_gpu::PtxGpuError::UseBeforeDef(0),
            poot_llm::RunnerError::from,
        ),
        source_row(
            "RunnerError::Mrope -> poot_llm::MropeBindError",
            poot_llm::MropeBindError::MissingSlot,
            poot_llm::RunnerError::from,
        ),
        // poot_ptx_gpu::PtxGpuError (crates/poot-ptx-gpu/src/lib.rs): 4 links, all explicit `#[source]`.
        source_row(
            "poot_ptx_gpu::PtxGpuError::Ptx -> poot_ptx_runtime::PtxError",
            poot_ptx_runtime::PtxError::CudaUnavailable("t".to_string()),
            poot_ptx_gpu::PtxGpuError::from,
        ),
        source_row(
            "poot_ptx_gpu::PtxGpuError::Codegen -> poot_codegen::CompileError",
            poot_codegen::CompileError::Io(std::io::Error::other("t")),
            poot_ptx_gpu::PtxGpuError::from,
        ),
        source_row(
            "poot_ptx_gpu::PtxGpuError::Plan -> poot_graph_plan::PlanError",
            plan_cause(),
            poot_ptx_gpu::PtxGpuError::from,
        ),
        source_row(
            "poot_ptx_gpu::PtxGpuError::Stage -> poot_graph_plan::CompileError",
            stage_cause(),
            poot_ptx_gpu::PtxGpuError::from,
        ),
    ]
}

/// The ROCm-gated links: `RunnerError::Rocm` (1), `RocmGpuError` (2). Card 548: `RocmGpuError`
/// shrank to a thin wrapper over `RocmError`/`CompileError` (the graph-walking executor's own
/// `Plan`/`Eval`/`UseBeforeDef` variants it used to carry moved to the generic
/// `poot_executor::LoadError`/`ExecError`, which this file does not track - `RocmDevice` never
/// calls `poot_graph_plan::compile`/`poot_eval::eval` itself).
/// A separate function for the same `unused_mut` reason as `rocm_error_sizes`.
#[cfg(feature = "rocm")]
fn rocm_chain_links() -> Vec<(&'static str, Box<dyn Error>, Option<String>)> {
    vec![
        source_row(
            "RunnerError::Rocm -> poot_rocm_gpu::RocmGpuError",
            poot_rocm_gpu::RocmGpuError::Graph("test cause".to_string()),
            poot_llm::RunnerError::from,
        ),
        // poot_rocm_gpu::RocmGpuError (crates/poot-rocm-gpu/src/error.rs): 2 links, both
        // explicit `#[source]`.
        source_row(
            "poot_rocm_gpu::RocmGpuError::Rocm -> poot_rocm_runtime::RocmError",
            poot_rocm_runtime::RocmError::LibraryNotFound("t".to_string()),
            poot_rocm_gpu::RocmGpuError::from,
        ),
        source_row(
            "poot_rocm_gpu::RocmGpuError::Codegen -> poot_codegen::CompileError",
            poot_codegen::CompileError::Io(std::io::Error::other("t")),
            poot_rocm_gpu::RocmGpuError::from,
        ),
    ]
}

#[cfg(not(feature = "rocm"))]
fn rocm_chain_links() -> Vec<(&'static str, Box<dyn Error>, Option<String>)> {
    Vec::new()
}

/// Checks every boxed-cause link in one hop (the parent's immediate `source()`; the full three-level
/// chain is `boxing_a_cause_does_not_cut_the_source_chain`). Dropping `#[source]` from any one
/// `source_row` field, e.g. `RocmGpuError::Eval`, sends only that row's `source()` to `None`.
#[test]
fn every_boxed_cause_across_all_seven_enums_preserves_its_source() {
    let mut links = non_rocm_chain_links();
    links.extend(rocm_chain_links());

    assert_eq!(
        links.len(),
        if cfg!(feature = "rocm") { 11 } else { 8 },
        "a link was added or removed without updating this count"
    );

    let mut mismatched = Vec::new();
    for (link, parent, expected) in &links {
        let actual = parent.source().map(|s| s.to_string());
        if actual != *expected {
            mismatched.push(format!(
                "{link}: source() was {actual:?}, expected {expected:?}"
            ));
        }
    }
    assert!(
        mismatched.is_empty(),
        "these links do not preserve source() the way boxing was supposed to: {mismatched:?}"
    );
}

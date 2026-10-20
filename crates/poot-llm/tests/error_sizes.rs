//! Card 384a acceptance: the six slice-1 error types stay under clippy's `result_large_err`
//! threshold, and boxing their causes changed neither what an error prints nor what
//! `Error::source` reports.
//!
//! Lives in `poot-llm` because `DriverError` wraps the device errors. The size table diagnoses which type grew; the guarantee is the clippy gate
//! (`cargo clippy -p <crate> --all-targets -- -D warnings`).

use std::error::Error;
use std::mem::size_of;

/// clippy's `large-error-threshold` default (no `clippy.toml` in this repo). The row check is `>=`:
/// clippy fires the lint at exactly 128 bytes (`RocmGpuError` measured 128 and failed `-D warnings`).
const LARGE_ERROR_THRESHOLD: usize = 128;

fn error_sizes() -> Vec<(&'static str, usize)> {
    vec![
        (
            "poot_llm::driver::error::DriverError",
            size_of::<poot_llm::driver::error::DriverError>(),
        ),
        ("poot_text::TextError", size_of::<poot_text::TextError>()),
        (
            "poot_ptx_device::PtxGpuError",
            size_of::<poot_ptx_device::PtxGpuError>(),
        ),
        (
            "poot_rocm_device::RocmGpuError",
            size_of::<poot_rocm_device::RocmGpuError>(),
        ),
    ]
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
        poot_ptx_device::PtxGpuError::from(plan_cause()).to_string(),
        format!("plan: {PLAN_CAUSE_TEXT}")
    );
    assert_eq!(
        poot_rocm_device::RocmGpuError::from(poot_rocm_runtime::RocmError::LibraryNotFound(
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
    let ptx = poot_ptx_device::PtxGpuError::from(plan_cause());
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
    fn ptx() -> Result<(), poot_ptx_device::PtxGpuError> {
        planner_fails()?;
        Ok(())
    }

    assert!(matches!(ptx(), Err(poot_ptx_device::PtxGpuError::Plan(_))));
}

/// Covers every one of the boxed-cause links (`boxing_a_cause_does_not_cut_the_source_chain`
/// covers only `PtxGpuError -> PlanError`): each row's parent has an explicit
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

fn ptx_chain_links() -> Vec<(&'static str, Box<dyn Error>, Option<String>)> {
    vec![
        // poot_ptx_device::PtxGpuError (crates/poot-ptx-device/src/lib.rs): 4 links, all explicit `#[source]`.
        source_row(
            "poot_ptx_device::PtxGpuError::Ptx -> poot_ptx_runtime::PtxError",
            poot_ptx_runtime::PtxError::CudaUnavailable("t".to_string()),
            poot_ptx_device::PtxGpuError::from,
        ),
        source_row(
            "poot_ptx_device::PtxGpuError::Codegen -> poot_codegen::CompileError",
            poot_codegen::CompileError::Io(std::io::Error::other("t")),
            poot_ptx_device::PtxGpuError::from,
        ),
        source_row(
            "poot_ptx_device::PtxGpuError::Plan -> poot_graph_plan::PlanError",
            plan_cause(),
            poot_ptx_device::PtxGpuError::from,
        ),
        source_row(
            "poot_ptx_device::PtxGpuError::Stage -> poot_graph_plan::CompileError",
            stage_cause(),
            poot_ptx_device::PtxGpuError::from,
        ),
    ]
}

/// The ROCm links: `RocmGpuError` (2). Card 548: `RocmGpuError`
/// shrank to a thin wrapper over `RocmError`/`ArchError` (the graph-walking executor's own
/// `Plan`/`Eval`/`UseBeforeDef` variants it used to carry moved to the generic
/// `poot_executor::LoadError`/`ExecError`, which this file does not track - `RocmDevice` never
/// calls `poot_graph_plan::compile`/`poot_eval::eval` itself).
fn rocm_chain_links() -> Vec<(&'static str, Box<dyn Error>, Option<String>)> {
    vec![
        // poot_rocm_device::RocmGpuError (crates/poot-rocm-device/src/error.rs): 2 links, both
        // explicit `#[source]`.
        source_row(
            "poot_rocm_device::RocmGpuError::Rocm -> poot_rocm_runtime::RocmError",
            poot_rocm_runtime::RocmError::LibraryNotFound("t".to_string()),
            poot_rocm_device::RocmGpuError::from,
        ),
        source_row(
            "poot_rocm_device::RocmGpuError::Arch -> poot_target::ArchError",
            poot_target::ArchError::NoGfxCode("t".to_string()),
            poot_rocm_device::RocmGpuError::from,
        ),
    ]
}

/// Checks every boxed-cause link in one hop (the parent's immediate `source()`; the full three-level
/// chain is `boxing_a_cause_does_not_cut_the_source_chain`). Dropping `#[source]` from any one
/// `source_row` field, e.g. `RocmGpuError::Eval`, sends only that row's `source()` to `None`.
#[test]
fn every_boxed_cause_across_the_device_error_enums_preserves_its_source() {
    let mut links = ptx_chain_links();
    links.extend(rocm_chain_links());

    assert_eq!(
        links.len(),
        6,
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

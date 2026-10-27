//! Generate shape-specialized [`poot_kernel_ir::Body`] kernels for primitive tensor ops: the un-fused GPU
//! path (one kernel per graph op), plus the fused and synthesized generators. These are runtime-generated
//! equivalents of the hand-written fixtures.
//!
//! Conventions: a 1-D grid covers the output; the output length comes from `Len(out)` (so a kernel works for
//! any length, not a baked-in size); the naive reductions loop serially over the reduced dim (no LDS). The
//! standard skeleton is `i = tid.x; if i < out.len() { ... }`.

mod aie_row;
mod attention;
mod block_top_k;
mod contraction;
mod data_movement;
mod elementwise;
mod emit;
mod error;
mod fp8;
mod fused;
mod gemv;
mod helpers;
mod matmul;
mod microtile;
mod packed;
mod reduce;
mod request;
mod scan;
#[cfg(test)]
mod synchronization;
#[cfg(test)]
mod vec4_probe;
#[cfg(test)]
mod wmma;

pub use attention::{
    AttentionCombineSpec, AttentionMicro, AttentionRows, AttentionSchedule, AttentionSink,
    AttentionSpec, KeyBound, KeyLayout, KvLane, LaneTile, MaskKind, ScoreTransform,
};

// The one generation seam. `poot-graph-plan` plans every generated kernel through `generate` and the request
// types; nothing else it needs from this crate is a generator.
pub use block_top_k::{BlockTopKSpec, LANES as BLOCK_TOP_K_LANES};
pub use error::{BodyResource, BodyStage, KernelGenError};
pub use request::{
    AieOp, BodyUse, CastSpec, ContractionSpec, DenseTiledContraction, ErrorBound, ExpertGroupSpec,
    Fold, FragmentUse, Generated, IndexedContraction, KernelRequest, KernelRequirements, Launch,
    MatmulShapes, NumericalContract, PackedContraction, PackedRequest, PackedRows, PointwiseForm,
    PointwiseSpec, RopeSpec, RowSpec, SerializedAccess, SerializedBinding, SerializedGenerated,
    TopKSpec, UnmeasurableLds, ViewOperand, fold_groups, generate, index_bound,
    serialized_schedule,
};
pub use scan::{ScanKernel, ScanSpec};

// The types a request carries (the planner builds them), and `view_eff_strides`, the planner's strided-view
// legality predicate shares with the generators.
pub use contraction::{
    DenseTiledWeight, Epilogue, EpilogueInput, EpilogueLeaf, EpilogueStep, GemvActivation,
    RegisterTiling, RunLength, Schedule, TileSize, TiledShape,
};
pub use fused::{
    FusedInput, FusedKernel, FusedScalarOp, FusedStep, LaneId, LeafId, LeafTy, RowKernel, RowOp,
    RowReduce, RowValue, StepId,
};
pub use helpers::{BodyBudget, BodyLimits, BodySize, Layout, WeightLayout, view_eff_strides};
pub use microtile::{LaneRows, MicroCols};
pub use packed::{PackedKernelOp, PackedKernelSpec};

// Generators still called by name outside `generate`, each for a recorded reason (card 636 census, at
// dispatch the planner calls none of them):
//
// - `poot-codegen`'s emitter, probe and parity tests; `pootc`'s import
//   equivalence tests; `test_support`; `poot-wgpu-device`, `poot-rocm-device` and `poot-graph-ir`
//   tests; this crate's own `tests/`: they build one generator's body directly to compile, run or compare
//   it, which is the generator under test rather than planning. None is a dispatch path.
pub use data_movement::{
    MovementOp, MovementSpec, Storage, UnitWindow, UpdateIndex, arg_top_k_dt, movement_body, rope,
    rope_dt,
};
pub use elementwise::{cast_bf16_to_f32, cast_f16_to_f32, cast_f32_to_bf16, cast_f32_to_f16};
pub use fp8::{e4m3fn_packed_to_f32, f32_to_e4m3fn_packed};
pub use fused::{
    fused, fused_i32_views_grid, fused_i32_views_grid_pack, fused_row_parallel_views, fused_views,
};
pub use gemv::{attn_scores_v_gemv_lds, gemv_lds};
pub use matmul::{matmul_batched_dt_grid, matmul_batched_epilogue_dt_grid};
pub use packed::{PackedRunSharing, packed_kernel, packed_run_sharing};
pub use reduce::reduce_last_dt;

/// Shared compile-and-check helpers for this crate's own unit tests (card 622): the crate's `pub(crate)`
/// kernel builders have no cross-crate caller, so their coverage lives here rather than in `tests/`, which
/// compiles as a separate crate and cannot see `pub(crate)` items.
#[cfg(test)]
pub(crate) mod unit_helpers {
    use std::path::PathBuf;

    use poot_codegen::{Target, compile, compile_kernel};
    use poot_kernel_ir::Body;
    use poot_target::Backend;

    pub(crate) fn spv(body: &Body, name: &str) -> poot_codegen::CompiledKernel {
        let dir: PathBuf = std::env::temp_dir()
            .join("poot-kernelgen-unit-test")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
        let artifact = compile_kernel(body, Backend::SpirvVulkan, &out).expect("compile");
        poot_codegen::kernel_handle(body, &artifact).unwrap()
    }

    /// The compiled SPIR-V word count of a handle [`spv`] built, for callers that just want to assert a
    /// non-empty compile (card 608: `CompiledKernel` has no `len`/`is_empty` of its own).
    pub(crate) fn spirv_word_count(kernel: &poot_codegen::CompiledKernel) -> usize {
        let poot_codegen::KernelCode::SpirvWords(words) = kernel.code() else {
            panic!("expected a SpirvVulkan-compiled kernel");
        };
        words.len()
    }

    pub(crate) fn ptx(body: &Body, name: &str) -> String {
        let dir: PathBuf = std::env::temp_dir()
            .join("poot-kernelgen-unit-test")
            .join(format!("{name}_ptx"));
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, Target::Nvptx);
        compile(body, Target::Nvptx, &out).expect("compile to NVPTX");
        std::fs::read_to_string(&out).unwrap()
    }

    pub(crate) fn ctx() -> Option<poot_wgpu_runtime::Context> {
        match poot_wgpu_runtime::Context::new() {
            Ok(c) => Some(c),
            Err(e) => {
                poot_test_util::skip_report(
                    "poot-kernelgen",
                    &format_args!("no wgpu context: {e}"),
                );
                None
            }
        }
    }

    /// Assert every fixture in `names` is staged in the drop point: a staging test that stops calling
    /// [`stage_nvptx`] for one fails here instead of leaving `ptx-check` to find it missing on a pod. Without
    /// `llc` nothing can be staged, and [`stage_nvptx`] has reported the skip.
    pub(crate) fn assert_staged(names: &[&str]) {
        if !which("llc") {
            return;
        }
        let dir = poot_test_util::ptx_stage_dir();
        for name in names {
            assert!(
                poot_codegen::NVPTX_COMPILE_ONLY_FIXTURES.contains(name),
                "{name} is staged by a unit test but is not in poot_codegen::NVPTX_COMPILE_ONLY_FIXTURES"
            );
            let path = dir.join(format!("{name}.ptx"));
            assert!(path.is_file(), "{} was not staged", path.display());
        }
    }

    pub(crate) fn which(tool: &str) -> bool {
        std::process::Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Compile `body` for `arch` to `out`. `false` when the AMD toolchain is absent (`POOT_AMD_CLANG` unset: the
    /// typed [`poot_codegen::CompileError::ToolchainMissing`]), after reporting the skip with its reason: a row
    /// that needs the toolchain skips on a machine without it (a pod), never fails there. Any other compile
    /// error is the row's failure.
    pub(crate) fn compile_amdgcn_or_skip(
        test: &str,
        body: &Body,
        arch: poot_target::AmdArch,
        out: &std::path::Path,
    ) -> bool {
        match compile(body, Target::AmdGcn(arch), out) {
            Ok(_) => true,
            Err(poot_codegen::CompileError::ToolchainMissing { variable, .. }) => {
                eprintln!("POOT-SKIP poot-kernelgen: {test} (no AMD toolchain: {variable} unset)");
                false
            }
            Err(error) => panic!("{test}: AMDGCN compile: {error}"),
        }
    }

    /// Stage `body`'s NVPTX to `<poot_test_util::ptx_stage_dir()>/<name>.ptx`, the drop point `tests/emit_nvptx.rs`
    /// and `ptx-check` (the PTX lane's `POOT_PTX_STAGE_DIR`) share. Skips (not fails) if `llc` is
    /// absent, matching `emit_nvptx.rs`'s own skip.
    pub(crate) fn stage_nvptx(body: &Body) {
        if std::process::Command::new("llc")
            .arg("--version")
            .output()
            .is_err()
        {
            poot_test_util::skip_report(
                "poot-kernelgen",
                &format_args!("llc not on PATH, {} not staged", body.name),
            );
            return;
        }
        let dir = poot_test_util::ptx_stage_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join(format!("{}.ptx", body.name));
        compile(body, Target::Nvptx, &out).unwrap_or_else(|e| panic!("{}: {e}", body.name));
    }
}

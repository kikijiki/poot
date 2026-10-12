//! Generate shape-specialized [`poot_kernel_ir::Body`] kernels for primitive tensor ops: the un-fused GPU
//! path (one kernel per graph op), plus the fused and synthesized generators. These are runtime-generated
//! equivalents of the hand-written fixtures.
//!
//! Conventions: a 1-D grid covers the output; the output length comes from `Len(out)` (so a kernel works for
//! any length, not a baked-in size); the naive reductions loop serially over the reduced dim (no LDS). The
//! standard skeleton is `i = tid.x; if i < out.len() { ... }`.

mod attention;
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
mod packed;
mod reduce;
mod request;
mod scan;
#[cfg(test)]
mod synchronization;
mod tiled_region;
mod wmma;

pub use attention::{
    AttentionCombineSpec, AttentionSchedule, AttentionSpec, KeyBound, KeyLayout, MaskKind,
    ScoreTransform,
};

// The one generation seam. `poot-graph-plan` plans every generated kernel through `generate` and the request
// types; nothing else it needs from this crate is a generator.
pub use error::{BodyResource, BodyStage, KernelGenError};
pub use request::{
    BodyUse, CastSpec, ContractionSpec, ElementwiseSpec, Fold, FragmentUse, Generated, I32Binary,
    IndexedContraction, KernelRequest, KernelRequirements, Launch, MatmulShapes, PackedContraction,
    PackedRequest, PackedRows, PointwiseForm, PointwiseSpec, RopeSpec, RowSpec, SequentialBinding,
    SequentialGenerated, SerialWork, TopKSpec, UnaryOp, UnmeasurableLds, ValueBinary, ViewOperand,
    fold_groups, generate, sequential_schedule,
};
pub use scan::{ScanKernel, ScanPhase, ScanSpec};

// The types a request carries (the planner builds them), and `view_eff_strides`, the planner's strided-view
// legality predicate shares with the generators.
pub use contraction::{Epilogue, EpilogueInput, EpilogueLeaf, EpilogueStep, Schedule, TileSize};
pub use fused::{
    FusedInput, FusedKernel, FusedScalarOp, FusedStep, LeafTy, RowKernel, RowOp, RowReduce,
};
pub use helpers::{BodyBudget, BodyLimits, BodySize, Layout, WeightLayout, view_eff_strides};
pub use packed::{PackedKernelOp, PackedKernelSpec};

// Generators still called by name outside `generate`, each for a recorded reason (card 636 census, at
// dispatch the planner calls none of them):
//
// - `poot-codegen`'s cache key fixture (`unary`) and its emitter, probe and parity tests; `pootc`'s import
//   equivalence tests; `test_support`; `poot-wgpu-device`, `poot-rocm-device` and `poot-graph-ir`
//   tests; this crate's own `tests/`: they build one generator's body directly to compile, run or compare
//   it, which is the generator under test rather than planning. None is a dispatch path.
pub use data_movement::{
    MovementOp, MovementSpec, Storage, UnitWindow, UpdateIndex, arg_top_k_dt, movement_body, rope,
    rope_dt,
};
pub use elementwise::{
    binary, binary_broadcast_dt_views_grid, binary_broadcast_i32_geu_views_grid, binary_scalar,
    binary_scalar_i32_geu_grid, binary_scalar_i32_grid, binary_scalar_i32_remu_grid,
    cast_bf16_to_f32, cast_f16_to_f32, cast_f32_to_bf16, cast_f32_to_f16, unary, unary_dt_grid,
    unary_i32_clz_grid,
};
pub use fp8::{e4m3fn_packed_to_f32, f32_to_e4m3fn_packed};
pub use fused::{
    fused, fused_i32_views_grid, fused_i32_views_grid_pack, fused_row_parallel_views, fused_views,
};
pub use gemv::{attn_scores_v_gemv_lds, gemv_lds};
pub use matmul::{matmul_batched_dt_grid, matmul_batched_epilogue_dt_grid};
pub use packed::packed_kernel;
pub use reduce::reduce_last_dt;
pub use tiled_region::tiled_region;
pub use wmma::{matmul_tensorcore, matmul_tensorcore_coopmat};

/// Shared compile-and-check helpers for this crate's own unit tests (card 622): the crate's `pub(crate)`
/// kernel builders have no cross-crate caller, so their coverage lives here rather than in `tests/`, which
/// compiles as a separate crate and cannot see `pub(crate)` items.
#[cfg(test)]
pub(crate) mod unit_helpers {
    use std::path::PathBuf;

    use poot_codegen::{Target, compile};
    use poot_kernel_ir::Body;

    pub(crate) fn spv(body: &Body, name: &str) -> poot_runtime_common::CompiledKernel {
        let dir: PathBuf = std::env::temp_dir()
            .join("poot-kernelgen-unit-test")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
        compile(body, Target::SpirvVulkan, &out).expect("compile");
        let bytes = std::fs::read(&out).unwrap();
        poot_codegen::kernel_handle(body, poot_codegen::GpuTarget::SpirvVulkan, bytes)
    }

    /// The compiled SPIR-V word count of a handle [`spv`] built, for callers that just want to assert a
    /// non-empty compile (card 608: `CompiledKernel` has no `len`/`is_empty` of its own).
    pub(crate) fn spirv_word_count(kernel: &poot_runtime_common::CompiledKernel) -> usize {
        let poot_runtime_common::KernelCode::SpirvWords(words) = kernel.code() else {
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
                eprintln!("no GPU ({e}); skipping");
                None
            }
        }
    }

    pub(crate) fn which(tool: &str) -> bool {
        std::process::Command::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Stage `body`'s NVPTX to `/tmp/poot-ptx-stage2/<name>.ptx`, the well-known drop point `tests/emit_nvptx.rs`
    /// (and, downstream, `ptx-check` on a real NVIDIA pod) reads from. Skips (not fails) if `llc` is
    /// absent, matching `emit_nvptx.rs`'s own skip.
    pub(crate) fn stage_nvptx(body: &Body) {
        if std::process::Command::new("llc")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("llc not on PATH; skipping NVPTX stage for {}", body.name);
            return;
        }
        let dir = std::path::PathBuf::from("/tmp/poot-ptx-stage2");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join(format!("{}.ptx", body.name));
        compile(body, Target::Nvptx, &out).unwrap_or_else(|e| panic!("{}: {e}", body.name));
    }
}

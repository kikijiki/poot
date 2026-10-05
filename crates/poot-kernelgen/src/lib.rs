//! Generate shape-specialized [`poot_kernel_ir::Body`] kernels for primitive tensor ops: the un-fused GPU
//! path (one kernel per graph op), plus the fused and synthesized generators. These are runtime-generated
//! equivalents of the hand-written fixtures.
//!
//! Conventions: a 1-D grid covers the output; the output length comes from `Len(out)` (so a kernel works for
//! any length, not a baked-in size); the naive reductions loop serially over the reduced dim (no LDS). The
//! standard skeleton is `i = tid.x; if i < out.len() { ... }`.

mod contraction;
mod data_movement;
mod elementwise;
mod emit;
mod error;
mod flash;
mod fp8;
mod fused;
mod gemv;
mod helpers;
mod matmul;
mod packed;
mod reduce;
mod request;
mod synchronization;
mod tiled_region;
mod wmma;

// The one generation seam. `poot-graph-plan` plans every generated kernel through `generate` and the request
// types; nothing else it needs from this crate is a generator.
pub use error::{BodyResource, BodyStage, KernelGenError};
pub use request::{
    AttentionSpec, BodyUse, CastSpec, ContractionSpec, E4m3Movement, ElementwiseSpec, Fold,
    FragmentUse, Generated, I32Binary, KernelRequest, KernelRequirements, Launch, MatmulShapes,
    MovementSpec, PackedRequest, PointwiseForm, PointwiseSpec, RopeSpec, RowSpec, SerialWork,
    TopKSpec, UnaryOp, UnmeasurableLds, ValueBinary, ViewOperand, fold_groups, generate,
};

// The types a request carries (the planner builds them), and `view_eff_strides`, the planner's strided-view
// legality predicate shares with the generators.
pub use contraction::{Schedule, TileSize};
pub use fused::{FusedInput, FusedKernel, FusedScalarOp, FusedStep, RowKernel, RowOp, RowReduce};
pub use helpers::{BodyBudget, BodyLimits, BodySize, Layout, WeightLayout, view_eff_strides};
pub use packed::{PackedKernelOp, PackedKernelSpec, RowSelect};

// Generators still called by name outside `generate`, each for a recorded reason (card 636 census, at
// dispatch the planner calls none of them):
//
// - PTX collective and segment kernels (`poot-ptx-gpu` `p2p.rs`, `multi_device/segment.rs`): `binary`. They
//   build a body for a peer-to-peer add outside any graph equation, so there is no request to plan; Card
//   748's rewrite owns them.
// - `poot-codegen`'s cache key fixture (`unary`) and its emitter, probe and parity tests; `pootc`'s import
//   equivalence tests; `poot-test-util`'s `kernel-fixtures`; `poot-gpu`, `poot-rocm-gpu` and `poot-graph-ir`
//   tests; this crate's own `tests/`: they build one generator's body directly to compile, run or compare
//   it, which is the generator under test rather than planning. None is a dispatch path.
pub use data_movement::{
    arg_top_k_dt, broadcast, broadcast_dt, concat_n_dt, concat2_dt, dyn_update_slice_dt,
    dyn_update_slice_dynamic_dt, gather_axis0_index_dt, rope, rope_dt, scatter_axis0_dt,
    scatter_update_dt, slice_dt, transpose_dt,
};
pub use elementwise::{
    binary, binary_broadcast_dt_views_grid, binary_broadcast_i32_geu_views_grid, binary_scalar,
    binary_scalar_i32_geu_grid, binary_scalar_i32_grid, binary_scalar_i32_remu_grid,
    cast_bf16_to_f32, cast_f16_to_f32, cast_f32_to_bf16, cast_f32_to_f16, unary, unary_dt_grid,
    unary_i32_clz_grid,
};
pub use flash::{flash_attention_decode, flash_region_decode, flash_region_prefill};
pub use fp8::{
    e4m3fn_broadcast_packed, e4m3fn_concat_packed, e4m3fn_dynamic_update_slice_dynamic_packed,
    e4m3fn_dynamic_update_slice_packed, e4m3fn_gather_packed, e4m3fn_packed_to_f32,
    e4m3fn_repack_reshape, e4m3fn_scatter_update_packed, e4m3fn_slice_packed,
    e4m3fn_transpose_packed, f32_to_e4m3fn_packed,
};
pub use fused::{
    fused, fused_i32_views_grid, fused_i32_views_grid_pack, fused_row_parallel_views, fused_views,
};
pub use gemv::{attn_scores_v_gemv_lds, gemv_lds};
pub use matmul::{indexed_matmul_dt, matmul_batched_bias_dt_grid, matmul_batched_dt_grid};
pub use packed::packed_kernel;
pub use reduce::reduce_last_dt;
pub use synchronization::cross_block_fence_spin_wait_coop;
pub use tiled_region::tiled_region;
pub use wmma::{matmul_tensorcore, matmul_tensorcore_coopmat};

/// Shared compile-and-check helpers for this crate's own unit tests (card 622): the crate's `pub(crate)`
/// kernel builders have no cross-crate caller, so their coverage lives here rather than in `tests/`, which
/// compiles as a separate crate and cannot see `pub(crate)` items.
#[cfg(test)]
pub(crate) mod test_support {
    use std::path::PathBuf;

    use poot_codegen::{Target, compile};
    use poot_kernel_ir::Body;

    pub(crate) fn spv(body: &Body, name: &str) -> poot_runtime::CompiledKernel {
        let dir: PathBuf = std::env::temp_dir()
            .join("poot-kernelgen-unit-test")
            .join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let out = poot_codegen::artifact_path(&dir, name, Target::SpirvVulkan);
        compile(body, Target::SpirvVulkan, &out).expect("compile");
        let bytes = std::fs::read(&out).unwrap();
        poot_codegen::kernel_handle(body, Target::SpirvVulkan, bytes)
    }

    /// The compiled SPIR-V word count of a handle [`spv`] built, for callers that just want to assert a
    /// non-empty compile (card 608: `CompiledKernel` has no `len`/`is_empty` of its own).
    pub(crate) fn spirv_word_count(kernel: &poot_runtime::CompiledKernel) -> usize {
        let poot_runtime::KernelCode::SpirvWords(words) = kernel.code() else {
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

    pub(crate) fn ctx() -> Option<poot_runtime::Context> {
        match poot_runtime::Context::new() {
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
    /// (and, downstream, `poot-ptx-check` on a real NVIDIA pod) reads from. Skips (not fails) if `llc` is
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

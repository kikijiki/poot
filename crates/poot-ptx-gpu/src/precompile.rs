//! Pre-compile a graph's kernels to the on-disk PTX cache with no CUDA context (Card 549, R-546-13 as
//! amended by R-592a-2): used by `poot-llm`'s `bin/ptx-graph-check --precompile`, so there is one
//! definition of "every kernel `g` needs", over the shared [`poot_graph_plan::compile`] (never a
//! second, bespoke per-equation plan walk - R472-001).

use poot_graph_ir::Graph;
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission, Target, compile};
use poot_target::{Backend, DeviceCaps};

use crate::PtxGpuError;

/// Pre-compile every `Plan::Compute`/`ComputeMeta`/`ComputeChunks` kernel body `g` needs through
/// `compile`, into the on-disk PTX [`poot_codegen::KernelCache`] (shared across calls by its own
/// epoch-scoped directory). Returns the number of kernels newly built this call (a cache hit built
/// none). `caps` lets a caller with no measured device (a precompile/warm-up host may have no GPU at
/// all, only `llc`) pass [`DeviceCaps::ptx_default`].
pub fn precompile_graph(g: &Graph, caps: &DeviceCaps) -> Result<usize, PtxGpuError> {
    let target = Target {
        backend: Backend::Nvptx,
        caps: *caps,
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let program = compile(g, &target, &options)?;
    let cache = poot_codegen::KernelCache::open(poot_codegen::Target::Nvptx);
    let mut built = 0usize;
    for (_, plan) in program.planned() {
        match plan {
            poot_graph_plan::Plan::Compute { body, .. }
            | poot_graph_plan::Plan::ComputeMeta { body, .. } => {
                built += usize::from(
                    cache
                        .load_or_compile(body, options.limits.max_artifact_bytes)?
                        .built,
                );
            }
            poot_graph_plan::Plan::ComputeChunks(chunks) => {
                for chunk in chunks {
                    built += usize::from(
                        cache
                            .load_or_compile(&chunk.body, options.limits.max_artifact_bytes)?
                            .built,
                    );
                }
            }
            poot_graph_plan::Plan::Alias(_) | poot_graph_plan::Plan::View { .. } => {}
            poot_graph_plan::Plan::Collective { .. } => {
                return Err(PtxGpuError::Graph(format!(
                    "precompile_graph: plan is {:?}, which never lowers to a PTX kernel",
                    plan.kind()
                )));
            }
        }
    }
    Ok(built)
}

//! A bounds-guard branch inside a loop that also holds a workgroup barrier
//! ([[barrier-in-loop-crashes-spirv]]); this shape used to make llc emit invalid IR and SIGSEGV/hang RADV.
//! Two fixes cover it: the structurization pass gives the in-loop selection a dedicated merge block, and
//! the SPIR-V barrier-semantics post-process rewrites OpControlBarrier from SequentiallyConsistent
//! (Vulkan-illegal) to AcquireRelease|WorkgroupMemory. Regression guard: must emit valid Vulkan SPIR-V.
#![crate_type = "lib"]
use poot_kernel_intrinsics::{local_index, wg_read, wg_write, workgroup_barrier};
const W: usize = 64;

pub fn __poot_kernel_barrier_branch_loop(a: &[f32], dims: &[u32], out: &mut [f32]) {
    let len = dims[0] as usize;
    let chunks = dims[1] as usize;
    let lane = local_index();
    let mut acc = 0.0f32;
    let mut k = 0usize;
    while k < chunks {
        let idx = k * W + lane;
        if idx < len {
            wg_write(0, lane, a[idx]);
        } else {
            wg_write(0, lane, 0.0);
        }
        workgroup_barrier();
        acc = acc + wg_read(0, lane);
        k = k + 1;
    }
    out[lane] = acc;
}

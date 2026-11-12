//! SC-4 (wgpu): the `Rvalue::SubgroupShuffleXor` fixtures, compiled to SPIR-V and dispatched through wgpu, store what
//! the interpreter does: lane `LocalX` receives the value of lane `LocalX ^ mask`. wgpu cannot require full
//! subgroups, so each fixture runs at 128 lanes and at two of the adapter's `max_size` subgroups, which is what
//! checks the contiguity contract of `SubgroupSupport::Present` on this adapter. A missing device is a
//! `POOT-SKIP`, and a failure when `POOT_REQUIRE_WGPU` requires it.
//!
//! Mutation (kept out of the tree): in `emit_subgroup_shuffle_xor`'s SPIR-V arm, take the source lane
//! `lane ^ (mask << 1)`; every received value is the wrong lane's and this row goes red.

use poot_codegen::{Target, compile_kernel, kernel_handle};
use poot_kernel_ir::Ty;
use poot_kernel_ir::test_support::{
    SHUFFLE_MASKS, subgroup_shuffle_xor_expected, subgroup_shuffle_xor_kernel,
};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::{open_or_skip, skip_unsupported};
use poot_target::{Queried, SubgroupSupport};
use poot_wgpu_runtime::{Context, KernelBuffer};

const THREADS: u32 = 512;

#[test]
fn subgroup_shuffle_xor_matches_the_interpreter_on_wgpu() {
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let Queried::Known(SubgroupSupport::Present { max_size, .. }) = ctx.device_caps().subgroup
    else {
        skip_unsupported(DeviceBackend::Wgpu, "subgroup shuffles");
        return;
    };
    let dir = std::env::temp_dir().join("poot-wgpu-runtime-subgroup-shuffle");
    std::fs::create_dir_all(&dir).unwrap();
    let mut workgroups = vec![128, 2 * max_size.max(32)];
    workgroups.dedup();
    for workgroup in workgroups {
        for ty in [Ty::U32, Ty::F32] {
            let body = subgroup_shuffle_xor_kernel(ty.clone(), workgroup);
            let name = format!("{}_{workgroup}", body.name);
            let out = poot_codegen::artifact_path(&dir, &name, Target::SpirvVulkan);
            let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let kernel = kernel_handle(&body, &artifact).unwrap();
            let words = THREADS as usize * SHUFFLE_MASKS.len();
            let mut bufs = [match ty {
                Ty::U32 => KernelBuffer::write_i32(words),
                _ => KernelBuffer::write_f32(words),
            }];
            ctx.dispatch(
                &name,
                &kernel,
                body.workgroup_size,
                [THREADS, 1, 1],
                &mut bufs,
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let got: Vec<u32> = bufs[0]
                .as_bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| u32::from_le_bytes(*word))
                .collect();
            assert_eq!(got, subgroup_shuffle_xor_expected(&ty, THREADS), "{name}");
        }
    }
}

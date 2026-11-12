//! SC-4 (raw Vulkan): the `Rvalue::SubgroupShuffleXor` fixtures, compiled to SPIR-V and dispatched through `ash`
//! under the validation layer, store what the interpreter does: lane `LocalX` receives the value of lane
//! `LocalX ^ mask`. A missing Vulkan device is a `POOT-SKIP`, and a failure when the required lane selected it.
//!
//! Mutation (kept out of the tree): in `emit_subgroup_shuffle_xor`'s SPIR-V arm, take the source lane
//! `lane ^ (mask << 1)`; every received value is the wrong lane's and this row goes red.

use poot_codegen::{Target, compile_kernel, kernel_handle};
use poot_kernel_ir::Ty;
use poot_kernel_ir::test_support::{
    SHUFFLE_MASKS, subgroup_shuffle_xor_expected, subgroup_shuffle_xor_kernel,
};
use poot_runtime_common::device_skip::{open_or_skip, skip_unsupported};
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::{BufferStorage, Queried, SubgroupSupport};
use poot_vulkan_runtime::Context;

const THREADS: u32 = 512;
const WORKGROUP: u32 = 128;

#[test]
fn subgroup_shuffle_xor_matches_the_interpreter_on_vulkan() {
    let Some(ctx) = open_or_skip(DeviceBackend::Vulkan, Context::new()) else {
        return;
    };
    if !matches!(
        ctx.device_caps().subgroup,
        Queried::Known(SubgroupSupport::Present { .. })
    ) {
        skip_unsupported(DeviceBackend::Vulkan, "subgroup shuffles");
        return;
    }
    let dir = std::env::temp_dir().join("poot-vulkan-runtime-subgroup-shuffle");
    std::fs::create_dir_all(&dir).unwrap();
    for ty in [Ty::U32, Ty::F32] {
        let body = subgroup_shuffle_xor_kernel(ty.clone(), WORKGROUP);
        let name = body.name.clone();
        let out = poot_codegen::artifact_path(&dir, &name, Target::SpirvVulkan);
        let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = kernel_handle(&body, &artifact).unwrap();
        let words = THREADS as usize * SHUFFLE_MASKS.len();
        let storage = match ty {
            Ty::U32 => BufferStorage::i32(),
            _ => BufferStorage::f32(),
        };
        let output = ctx
            .alloc_storage(BufferRole::Output, storage, words)
            .expect("alloc output");
        ctx.dispatch(&kernel, body.workgroup_size, [THREADS, 1, 1], &[&output])
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut raw = vec![0u8; words * 4];
        output.read_bytes(&mut raw).expect("read output");
        let got: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(got, subgroup_shuffle_xor_expected(&ty, THREADS), "{name}");
    }
}

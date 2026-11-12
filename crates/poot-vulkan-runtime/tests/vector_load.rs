//! SC-d (raw Vulkan): the integer `Rvalue::VectorLoad` fixtures, compiled to SPIR-V and dispatched through `ash`
//! under the validation layer, read what scalar loads of the same words read. See the wgpu row for the two
//! fixtures. A missing Vulkan device is a `POOT-SKIP`, and a failure when the required lane selected it.
//!
//! Mutation (kept out of the tree): in `emit_vector_load`'s SPIR-V vector-resource arm, take the group index
//! `idx >> log2(lanes) + 1` (a wrong offset); the first fixture's lanes come from the next group and this row goes red.

use poot_codegen::{Target, compile_kernel, kernel_handle};
use poot_kernel_ir::test_support::{VECTOR_LOAD_THREADS, vector_load_cases};
use poot_runtime_common::device_skip::open_or_skip;
use poot_runtime_common::{BufferRole, DeviceBackend};
use poot_target::BufferStorage;
use poot_vulkan_runtime::Context;

#[test]
fn u32x4_vector_load_matches_scalar_loads_on_vulkan() {
    let Some(ctx) = open_or_skip(DeviceBackend::Vulkan, Context::new()) else {
        return;
    };
    let dir = std::env::temp_dir().join("poot-vulkan-runtime-vector-load");
    std::fs::create_dir_all(&dir).unwrap();
    for case in vector_load_cases() {
        let name = case.body.name.clone();
        let out = poot_codegen::artifact_path(&dir, &name, Target::SpirvVulkan);
        let artifact = compile_kernel(&case.body, poot_target::Backend::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = kernel_handle(&case.body, &artifact).unwrap();

        let input = ctx
            .alloc_storage(BufferRole::Input, BufferStorage::i32(), case.input.len())
            .expect("alloc input");
        let bytes: Vec<u8> = case.input.iter().flat_map(|w| w.to_le_bytes()).collect();
        input
            .write_with(0, bytes.len(), &mut |_, out| out.copy_from_slice(&bytes))
            .expect("write input");
        let output = ctx
            .alloc_storage(BufferRole::Output, BufferStorage::i32(), case.want.len())
            .expect("alloc output");

        ctx.dispatch(
            &kernel,
            case.body.workgroup_size,
            [VECTOR_LOAD_THREADS, 1, 1],
            &[&input, &output],
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));

        let mut raw = vec![0u8; case.want.len() * 4];
        output.read_bytes(&mut raw).expect("read output");
        let got: Vec<u32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(got, case.want, "{name}");
    }
}

//! SC-d (wgpu): the integer `Rvalue::VectorLoad` fixtures, compiled to SPIR-V and dispatched through wgpu, read
//! what scalar loads of the same words read. The `u32x4` fixture binds its input as a vector resource; the mixed
//! one reads the same param as `u32x4`, `u32x2` and scalar, so its lanes are scalar loads. A missing device is a
//! `POOT-SKIP`, and a failure when `POOT_REQUIRE_WGPU` requires it.
//!
//! Mutation (kept out of the tree): in `emit_vector_load`'s SPIR-V vector-resource arm, take the group index
//! `idx >> log2(lanes) + 1` (a wrong offset); the first fixture's lanes come from the next group and this row goes red.

use poot_codegen::{Target, compile_kernel, kernel_handle};
use poot_kernel_ir::test_support::{VECTOR_LOAD_THREADS, vector_load_cases};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_wgpu_runtime::{Context, KernelBuffer};

#[test]
fn u32x4_vector_load_matches_scalar_loads_on_wgpu() {
    let Some(ctx) = open_or_skip(DeviceBackend::Wgpu, Context::new()) else {
        return;
    };
    let dir = std::env::temp_dir().join("poot-wgpu-runtime-vector-load");
    std::fs::create_dir_all(&dir).unwrap();
    for case in vector_load_cases() {
        let name = case.body.name.clone();
        let out = poot_codegen::artifact_path(&dir, &name, Target::SpirvVulkan);
        let artifact = compile_kernel(&case.body, poot_target::Backend::SpirvVulkan, &out)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = kernel_handle(&case.body, &artifact).unwrap();
        let mut bufs = [
            KernelBuffer::read_only_u32(&case.input),
            KernelBuffer::write_i32(case.want.len()),
        ];
        ctx.dispatch(
            &name,
            &kernel,
            case.body.workgroup_size,
            [VECTOR_LOAD_THREADS, 1, 1],
            &mut bufs,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let got: Vec<u32> = bufs[1]
            .as_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(got, case.want, "{name}");
    }
}

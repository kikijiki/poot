use poot_codegen::{Target, compile};
use poot_kernel_ir::Ty;

/// Compile the single WMMA tile kernel (card 530: one target-neutral F16 fragment body) to NVPTX and confirm
/// the emitter produces the tensor-core instructions (load.a/b.f16, mma.sync...f32.f32, store.d.f32).
/// Compile-only (no GPU); runtime correctness is the pod probe.
#[test]
#[ignore = "probe: compiles wmma_tile, prints NVPTX"]
fn wmma_tile_nvptx() {
    let dir = std::env::temp_dir().join("poot-wmma");
    std::fs::create_dir_all(&dir).unwrap();
    let body = poot_test_util::kernel_fixtures::wmma_tile("k");
    let out = dir.join("wmma_tile.ptx");
    match compile(&body, Target::Nvptx, &out) {
        Ok(_ir) => {
            // check the emitted PTX (not the IR, which has the llvm.nvvm.wmma.* calls) for tensor-core ops.
            let ptx = std::fs::read_to_string(&out).unwrap();
            let n_mma = ptx.matches("wmma.mma.sync").count();
            let n_load = ptx.matches("wmma.load").count();
            let n_store = ptx.matches("wmma.store").count();
            eprintln!("wmma_tile PTX: load={n_load} mma={n_mma} store={n_store}");
            assert!(
                n_load >= 2 && n_mma >= 1 && n_store >= 1,
                "missing wmma ops in PTX:\n{ptx}"
            );
        }
        Err(e) => panic!("wmma_tile compile failed: {e}"),
    }
    // The same body must also compile on SpirvVulkan (coopmat) - the target-neutral claim (AmdGcn is covered
    // by `poot-codegen`'s `wmma_tile_is_one_body_lowered_on_every_target` unit test and `llc.rs`'s AMD
    // WMMA tests).
    assert!(compile(&body, Target::SpirvVulkan, &dir.join("x.spv")).is_ok());
}

/// The tiled general gemm `matmul_tensorcore` compiles to NVPTX with the WMMA ops inside a K-loop (load A/B +
/// one mma in the loop body, one store after). Compile-only; runtime correctness is the pod probe
/// `--probe-tc-matmul-tiled`.
#[test]
#[ignore = "probe: compiles matmul_tensorcore, prints NVPTX"]
fn matmul_tensorcore_nvptx() {
    let dir = std::env::temp_dir().join("poot-wmma-tiled");
    std::fs::create_dir_all(&dir).unwrap();
    // 32x32x32: 4 output tiles, K-loop of 2 steps. Single (no-batch) 2D case, f32 output.
    let body = poot_kernelgen::matmul_tensorcore("k", Ty::F32, &[32, 32], &[32, 32], &[32, 32])
        .expect("matmul_tensorcore precondition");
    let out = dir.join("matmul_tc.ptx");
    match compile(&body, Target::Nvptx, &out) {
        Ok(_ir) => {
            let ptx = std::fs::read_to_string(&out).unwrap();
            let n_mma = ptx.matches("wmma.mma.sync").count();
            let n_load = ptx.matches("wmma.load").count();
            let n_store = ptx.matches("wmma.store").count();
            eprintln!("matmul_tensorcore PTX: load={n_load} mma={n_mma} store={n_store}");
            assert!(
                n_load >= 2 && n_mma >= 1 && n_store >= 1,
                "missing wmma ops in PTX:\n{ptx}"
            );
        }
        Err(e) => panic!("matmul_tensorcore compile failed: {e}"),
    }
    // a batched 4D shape (attention-style) must also compile.
    let bat = poot_kernelgen::matmul_tensorcore(
        "kb",
        Ty::F32,
        &[1, 4, 32, 48],
        &[1, 4, 32, 16],
        &[1, 4, 16, 48],
    )
    .expect("matmul_tensorcore precondition");
    assert!(compile(&bat, Target::Nvptx, &dir.join("matmul_tc_b.ptx")).is_ok());
    // the bf16-output variant (LDS-stage + barrier + narrow epilogue) must also compile to PTX.
    let bf = poot_kernelgen::matmul_tensorcore("kc", Ty::BF16, &[32, 32], &[32, 32], &[32, 32])
        .expect("matmul_tensorcore precondition");
    let bf_out = dir.join("matmul_tc_bf16.ptx");
    let bf_ptx = match compile(&bf, Target::Nvptx, &bf_out) {
        Ok(_) => std::fs::read_to_string(&bf_out).unwrap(),
        Err(e) => panic!("bf16 matmul_tensorcore compile failed: {e}"),
    };
    assert!(
        bf_ptx.contains("wmma.store") && bf_ptx.contains("bar.sync"),
        "bf16 variant must store to LDS + barrier:\n{bf_ptx}"
    );
    // SPIR-V must reject.
    assert!(compile(&body, Target::SpirvVulkan, &dir.join("x.spv")).is_err());
}

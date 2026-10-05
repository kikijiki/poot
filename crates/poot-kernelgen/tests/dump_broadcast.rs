//! Diagnostic: emit the NVPTX for the V bias-add broadcast that diverges at n=864 (`[1,864,128]+[128]`) and a
//! small reference (`[1,5,128]+[128]`), to inspect for a large-shape codegen bug.
//! Run: `cargo test -p poot-kernelgen --test dump_broadcast -- --ignored --nocapture`.

use poot_codegen::{Target, compile};
use poot_kernel_ir::{BinOp, Ty};
use poot_test_util::kernel_fixtures::{binary_broadcast, gather_axis0_dt};

/// Emit the NVPTX IR of the embedding gather (rest=896, the embedding row size), which reads garbage (maxabs
/// 1e9) at n=864 on PTX while wgpu is correct, to inspect the f32->i64 token-id cast and the index/data loads.
/// NVPTX (usize=i64) vs SPIR-V (usize=i32) is the difference.
#[test]
#[ignore = "diagnostic: prints emitted gather IR"]
fn dump_gather_nvptx() {
    let dir = std::env::temp_dir().join("poot-gather-dump");
    std::fs::create_dir_all(&dir).unwrap();
    let body = gather_axis0_dt("k", Ty::F32, 896);
    let out = dir.join("gather.ptx");
    match compile(&body, Target::Nvptx, &out) {
        Ok(ir) => eprintln!("\n========== gather_axis0(rest=896) NVPTX IR ==========\n{ir}"),
        Err(e) => eprintln!("compile failed: {e}"),
    }
    // and the SPIR-V (wgpu, correct) for comparison
    let out2 = dir.join("gather.spv");
    if let Ok(ir) = compile(&body, Target::SpirvVulkan, &out2) {
        eprintln!("\n========== gather_axis0(rest=896) SPIR-V IR ==========\n{ir}");
    }
}

#[test]
#[ignore = "diagnostic: prints emitted IR/PTX"]
fn dump_v_bias_broadcast_nvptx() {
    let dir = std::env::temp_dir().join("poot-bcast-dump");
    std::fs::create_dir_all(&dir).unwrap();
    for (tag, n) in [("big", 864usize), ("small", 5usize)] {
        let body = binary_broadcast("k", BinOp::Add, &[1, n, 128], &[1, n, 128], &[128]);
        let out = dir.join(format!("k_{tag}.ptx"));
        match compile(&body, Target::Nvptx, &out) {
            Ok(ir) => {
                eprintln!("\n========== {tag} (n={n}) LLVM IR ==========\n{ir}");
            }
            Err(e) => eprintln!("{tag}: compile failed: {e}"),
        }
    }
}

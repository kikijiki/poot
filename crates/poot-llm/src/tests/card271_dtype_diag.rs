// Moved in-crate (card 622): references a pub(crate)-only item (poot has no users, tests are not consumers).

//! CPU-only diagnostic: print the operand dtypes of the projection MatMul eqns in the sequential
//! decode graph (`trace_decode_kv_masked`, captured by `Runner::generate_kv_ptx_tokens`) and the
//! batched shared-pool decode graph (`Runner::trace_batched_shared_pool_decode`) for a dense
//! bf16-native qwen2.5 checkpoint, and check what `widen_mismatched_matmul_dtypes` does to the
//! sequential graph.

use super::*;
use poot_tensor::DType;

use poot_graph_ir::op::OpKind;

use poot_graph_plan::passes_without_target as optimize;

#[test]
fn dense_qwen2_projection_matmul_dtypes() {
    let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b")) else {
        return;
    };
    let runner = Runner::load(&dir).expect("load dense qwen2.5 safetensors checkpoint");
    eprintln!("cfg.proj_dtype = {:?}", runner.cfg.proj_dtype);

    // --- Sequential decode graph (what generate_kv_ptx_tokens captures) ---
    let cap = 64;
    let g_seq = poot_models::qwen2::trace_decode_kv_masked(runner.cfg, cap);
    let g_seq = optimize(&g_seq);
    eprintln!("\n=== sequential decode graph (post-optimize, pre-widen) MatMul eqns ===");
    for eqn in &g_seq.eqns {
        if matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias) {
            let ids: Vec<_> = eqn
                .inputs
                .iter()
                .filter_map(|op| match op {
                    poot_graph_ir::graph::Operand::Value(v) => Some(*v),
                    _ => None,
                })
                .collect();
            let dts: Vec<DType> = ids.iter().map(|&id| g_seq.aval(id).dtype).collect();
            let odt = g_seq.aval(eqn.out).dtype;
            eprintln!("  {:?} operands={dts:?} out={odt:?}", eqn.op);
        }
    }

    let caps = poot_test_util::device_caps::default_caps_for(poot_target::Backend::Nvptx);
    let g_widened =
        poot_graph_plan::widen_mismatched_matmul_dtypes(&g_seq, poot_target::Backend::Nvptx, &caps);
    eprintln!("\n=== sequential decode graph AFTER widen_mismatched_matmul_dtypes ===");
    eprintln!(
        "eqn count: pre-widen={} post-widen={}",
        g_seq.eqns.len(),
        g_widened.eqns.len()
    );
    for eqn in &g_widened.eqns {
        if matches!(eqn.op, OpKind::Cast { .. }) {
            let ids: Vec<_> = eqn
                .inputs
                .iter()
                .filter_map(|op| match op {
                    poot_graph_ir::graph::Operand::Value(v) => Some(*v),
                    _ => None,
                })
                .collect();
            let in_dt: Vec<DType> = ids.iter().map(|&id| g_widened.aval(id).dtype).collect();
            let out_dt = g_widened.aval(eqn.out).dtype;
            eprintln!("  Cast eqn: in={in_dt:?} out={out_dt:?}");
        }
    }
    for eqn in &g_widened.eqns {
        if matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias) {
            let ids: Vec<_> = eqn
                .inputs
                .iter()
                .filter_map(|op| match op {
                    poot_graph_ir::graph::Operand::Value(v) => Some(*v),
                    _ => None,
                })
                .collect();
            let dts: Vec<DType> = ids.iter().map(|&id| g_widened.aval(id).dtype).collect();
            let odt = g_widened.aval(eqn.out).dtype;
            eprintln!("  post-widen {:?} operands={dts:?} out={odt:?}", eqn.op);
        }
    }

    // --- Batched shared-pool decode graph (what batch_engine_loop_ptx / capture_decode_paged captures) ---
    let g_batch = runner
        .trace_batched_shared_pool_decode(cap, 2, cap * 2, false)
        .expect("trace batched shared-pool decode");
    eprintln!("\n=== batched shared-pool decode graph (as traced) MatMul eqns ===");
    for eqn in &g_batch.eqns {
        if matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias) {
            let ids: Vec<_> = eqn
                .inputs
                .iter()
                .filter_map(|op| match op {
                    poot_graph_ir::graph::Operand::Value(v) => Some(*v),
                    _ => None,
                })
                .collect();
            let dts: Vec<DType> = ids.iter().map(|&id| g_batch.aval(id).dtype).collect();
            let odt = g_batch.aval(eqn.out).dtype;
            eprintln!("  {:?} operands={dts:?} out={odt:?}", eqn.op);
        }
    }
}

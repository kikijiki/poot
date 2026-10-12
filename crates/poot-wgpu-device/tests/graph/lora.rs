//! The batched-LoRA compile pass on wgpu (POOT-579, spec 248 phase 2 / epic 129 B8): the first
//! real-hardware run of the correction the LoRA pass inserts. Each of the seven targetable projections is
//! a hand-built graph compiled with a `LoraPlan` (`poot_executor_parity::lora`): two adapters on rows 0
//! and 1, no adapter on row 2, at one and at two tokens per row, every row against the host's
//! merged-weight reference. Twins: `poot-rocm-device/tests/batched_lora_rocm.rs`,
//! `poot-ptx-device/tests/batched_lora_ptx.rs`.

#[test]
fn batched_lora_decode_gpu_matches_per_row_merged_weight_reference() {
    let _gpu_guard = super::gpu_lock();
    let Some((mut exec, target)) = super::open_engine_or_skip() else {
        return;
    };
    // wgpu's `IndexedMatMul`/`MatMul` kernels reassociate: 1e-2 relative.
    poot_executor_parity::lora::assert_batched_lora_matches_merged_weights(&mut exec, target, 1e-2);
}

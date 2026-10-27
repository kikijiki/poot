//! The engine's state-copy kernel: `copy_words`, the raw word-for-word copy a device with no native buffer
//! copy runs for `Device::copy` (`poot_executor::CopyKernel`, `CopyMechanism::Kernel`). It has no planner call
//! site; the executor engine compiles it for the device's target. Authored in Rust
//! (`pootc/kernels/state/*.rs`).

kernel_assets! {
    kind = ImportedKernel;
    family = body_state;

    CopyWords {
        name: "copy_words",
        source: "copy_words",
        entry: "__poot_kernel_copy_words",
        params: 2,
        dest: GraphPlan,
    },
}

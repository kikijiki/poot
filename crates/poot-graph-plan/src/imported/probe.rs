//! Cross-crate standalone kernels (card 559, R468-016): not part of the planner's per-equation kernel
//! choice (none has a `planner/*.rs` call site), each loaded directly by the crate that needs it -
//! `poot-rocm-gpu`'s own executor-contract fixtures and `Device::copy` kernel, and `poot-gpu`'s
//! importer-capability probes (multi-array LDS + barrier, a runtime-value loop bound). Authored in Rust
//! (`pootc/kernels/probe/*.rs`). Grouped here by destination, not by kernel family, since the "one
//! manifest" invariant (every committed asset has exactly one entry) applies regardless of who reads it.

kernel_assets! {
    family = body_probe;

    AssertTrap {
        name: "assert_trap",
        source: "assert_trap",
        entry: "__poot_kernel_assert_trap",
        params: 3,
        dest: Crate("poot-rocm-gpu/assets"),
    },
    AtomicAddSumSmoke {
        name: "atomic_add_sum_smoke",
        source: "atomic_add_sum_smoke",
        entry: "__poot_kernel_atomic_add_sum_smoke",
        params: 1,
        dest: Crate("poot-rocm-gpu/assets"),
    },
    CopyWords {
        name: "copy_words",
        source: "copy_words",
        entry: "__poot_kernel_copy_words",
        params: 2,
        dest: Crate("poot-rocm-gpu/assets"),
    },
    SpinBusy {
        name: "spin_busy",
        source: "spin_busy",
        entry: "__poot_kernel_spin_busy",
        params: 2,
        dest: Crate("poot-rocm-gpu/assets"),
    },
    RowsumFor {
        name: "rowsum_for",
        source: "rowsum_for",
        entry: "__poot_kernel_rowsum_for",
        params: 2,
        dest: Crate("poot-gpu/src/tests/assets"),
    },
    RmsnormDyn {
        name: "rmsnorm_dyn",
        source: "rmsnorm_dyn",
        entry: "__poot_kernel_rmsnorm_dyn",
        params: 4,
        dest: Crate("poot-gpu/src/tests/assets"),
    },
}

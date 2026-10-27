//! Test-support kernels (card 559, R468-016; POOT-1237): the executor-contract fixtures and importer-capability
//! probes, none with a planner call site. Each is loaded directly by the crate whose test needs it -
//! `poot-rocm-device`'s executor-contract fixtures (a device trap, an atomic counter, a busy-spin) and
//! `poot-wgpu-device`'s importer-capability probes (multi-array LDS + barrier, a runtime-value loop bound).
//! Authored in Rust (`pootc/kernels/probe/*.rs`). Grouped here by destination, not by kernel family, since the
//! "one manifest" invariant (every committed asset has exactly one entry) applies regardless of who reads it.
//! Compiled only for tests and the `test-support` feature: the shipped planner carries none of them.

kernel_assets! {
    kind = ProbeKernel;
    family = body_probe;

    AssertTrap {
        name: "assert_trap",
        source: "assert_trap",
        entry: "__poot_kernel_assert_trap",
        params: 3,
        dest: Crate("poot-rocm-device/assets"),
    },
    AtomicAddSumSmoke {
        name: "atomic_add_sum_smoke",
        source: "atomic_add_sum_smoke",
        entry: "__poot_kernel_atomic_add_sum_smoke",
        params: 1,
        dest: Crate("poot-rocm-device/assets"),
    },
    SpinBusy {
        name: "spin_busy",
        source: "spin_busy",
        entry: "__poot_kernel_spin_busy",
        params: 2,
        dest: Crate("poot-rocm-device/assets"),
    },
    RowsumFor {
        name: "rowsum_for",
        source: "rowsum_for",
        entry: "__poot_kernel_rowsum_for",
        params: 2,
        dest: Crate("poot-wgpu-device/src/tests/assets"),
    },
    RmsnormDyn {
        name: "rmsnorm_dyn",
        source: "rmsnorm_dyn",
        entry: "__poot_kernel_rmsnorm_dyn",
        params: 4,
        dest: Crate("poot-wgpu-device/src/tests/assets"),
    },
}

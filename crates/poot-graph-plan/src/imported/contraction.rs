//! Matmul-family imported kernels (card 044): the coalesced decode GEMV (single and batched, f32 `[K, N]` weight)
//! and the thread-coarsened tiled GEMM (single and batched-weight), authored in Rust
//! (`pootc/kernels/contraction/*.rs`). The BF16-weight column-tile GEMVs are retired: a BF16 weight decodes
//! through kernelgen's dense Gemv over `[N, K]` (card 727), and a `[K, N]` BF16 decode projection is refused.
//! 560a retires this family into kernelgen's `ContractionSpec`.

kernel_assets! {
    family = body_contraction;

    GemvCoalesced {
        name: "gemv_coalesced",
        source: "gemv_coalesced",
        entry: "__poot_kernel_gemv_coalesced",
        params: 3,
        dest: GraphPlan,
    },
    GemvBatchedCoalesced {
        name: "gemv_batched_coalesced",
        source: "gemv_batched_coalesced",
        entry: "__poot_kernel_gemv_batched_coalesced",
        params: 4,
        dest: GraphPlan,
    },
    TiledGemm {
        name: "tiled_gemm",
        source: "tiled_gemm_coarsened",
        entry: "__poot_kernel_tiled_gemm_coarsened",
        params: 4,
        dest: GraphPlan,
    },
    TiledGemmBatched {
        name: "tiled_gemm_batched",
        source: "tiled_gemm_batched",
        entry: "__poot_kernel_tiled_gemm_batched",
        params: 4,
        dest: GraphPlan,
    },
}

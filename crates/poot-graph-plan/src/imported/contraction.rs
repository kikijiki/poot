//! Matmul-family imported kernels (card 044): the coalesced decode GEMV (single and batched, f32 and
//! BF16-weight, with and without the fused bias epilogue), the thread-coarsened tiled GEMM (single and
//! batched-weight), and card 380's dense BF16 contraction at M > 1 - authored in Rust
//! (`pootc/kernels/contraction/*.rs`). Its M == 1 decode GEMV is kernelgen's dense Gemv (card 727). 560a
//! retires this family into kernelgen's `ContractionSpec`.

kernel_assets! {
    family = body_contraction;

    GemvCoalesced {
        name: "gemv_coalesced",
        source: "gemv_coalesced",
        entry: "__poot_kernel_gemv_coalesced",
        params: 3,
        dest: GraphPlan,
    },
    GemvCoalescedBias {
        name: "gemv_coalesced_bias",
        source: "gemv_coalesced_bias",
        entry: "__poot_kernel_gemv_coalesced_bias",
        params: 4,
        dest: GraphPlan,
    },
    GemvBatchedCoalesced {
        name: "gemv_batched_coalesced",
        source: "gemv_batched_coalesced",
        entry: "__poot_kernel_gemv_batched_coalesced",
        params: 4,
        dest: GraphPlan,
    },
    GemvBatchedCoalescedBias {
        name: "gemv_batched_coalesced_bias",
        source: "gemv_batched_coalesced_bias",
        entry: "__poot_kernel_gemv_batched_coalesced_bias",
        params: 5,
        dest: GraphPlan,
    },
    GemvCoalescedBf16 {
        name: "gemv_coalesced_bf16",
        source: "gemv_coalesced_bf16",
        entry: "__poot_kernel_gemv_coalesced_bf16",
        params: 3,
        dest: GraphPlan,
    },
    GemvCoalescedBiasBf16 {
        name: "gemv_coalesced_bias_bf16",
        source: "gemv_coalesced_bias_bf16",
        entry: "__poot_kernel_gemv_coalesced_bias_bf16",
        params: 4,
        dest: GraphPlan,
    },
    GemvBatchedCoalescedBf16 {
        name: "gemv_batched_coalesced_bf16",
        source: "gemv_batched_coalesced_bf16",
        entry: "__poot_kernel_gemv_batched_coalesced_bf16",
        params: 4,
        dest: GraphPlan,
    },
    GemvBatchedCoalescedBiasBf16 {
        name: "gemv_batched_coalesced_bias_bf16",
        source: "gemv_batched_coalesced_bias_bf16",
        entry: "__poot_kernel_gemv_batched_coalesced_bias_bf16",
        params: 5,
        dest: GraphPlan,
    },
    TiledGemm {
        name: "tiled_gemm",
        source: "tiled_gemm_coarsened",
        entry: "__poot_kernel_tiled_gemm_coarsened",
        params: 4,
        dest: GraphPlan,
    },
    TiledGemmBias {
        name: "tiled_gemm_bias",
        source: "tiled_gemm_coarsened_bias",
        entry: "__poot_kernel_tiled_gemm_coarsened_bias",
        params: 5,
        dest: GraphPlan,
    },
    TiledGemmBatched {
        name: "tiled_gemm_batched",
        source: "tiled_gemm_batched",
        entry: "__poot_kernel_tiled_gemm_batched",
        params: 4,
        dest: GraphPlan,
    },
    DenseBf16Contraction {
        name: "dense_bf16_contraction",
        source: "dense_bf16_contraction",
        entry: "__poot_kernel_dense_bf16_contraction",
        params: 4,
        dest: GraphPlan,
    },
}

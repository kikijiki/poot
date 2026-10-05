//! On-device sampler imported kernels (card 551a, R472-007; card 447's original consolidation, deleted
//! with the old wgpu cached decode by card 546b and recreated on the contract path): one workgroup of 64
//! lanes per row for the four `SampleToken` bodies (shape-generic over `[rows, vocab]`, `dims =
//! [vocab]`), one thread per output element for `RandomUniform` (shape-generic over `[rows, cols]`,
//! `dims = [cols]`). Authored in Rust (`pootc/kernels/sampling/*.rs`); `poot-gpu` lowers each to SPIR-V,
//! `poot-ptx-gpu` to NVPTX and `poot-rocm-gpu` to AMDGCN, so there is one kernel definition and one copy
//! of each asset. See `planner::sampling` for their `Plan::ComputeMeta` lowering.

kernel_assets! {
    family = body_sampling;

    ArgmaxBatched {
        name: "argmax_batched",
        source: "argmax_batched",
        entry: "__poot_kernel_argmax_batched",
        params: 3,
        dest: GraphPlan,
    },
    SampleGumbelArgmaxBatched {
        name: "sample_gumbel_argmax_batched",
        source: "sample_gumbel_argmax_batched",
        entry: "__poot_kernel_sample_gumbel_argmax_batched",
        params: 5,
        dest: GraphPlan,
    },
    SampleTruncatedGumbelArgmaxBatched {
        name: "sample_truncated_gumbel_argmax_batched",
        source: "sample_truncated_gumbel_argmax_batched",
        entry: "__poot_kernel_sample_truncated_gumbel_argmax_batched",
        params: 6,
        dest: GraphPlan,
    },
    SampleToppGumbelArgmaxBatched {
        name: "sample_topp_gumbel_argmax_batched",
        source: "sample_topp_gumbel_argmax_batched",
        entry: "__poot_kernel_sample_topp_gumbel_argmax_batched",
        params: 6,
        dest: GraphPlan,
    },
    RandomUniform {
        name: "random_uniform",
        source: "random_uniform",
        entry: "__poot_kernel_random_uniform",
        params: 3,
        dest: GraphPlan,
    },
}

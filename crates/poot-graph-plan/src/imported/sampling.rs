//! On-device sampler imported kernels (card 551a, R472-007; card 447's original consolidation, deleted
//! with the old wgpu cached decode by card 546b and recreated on the contract path): one workgroup of 64
//! lanes per row for the three stochastic `SampleToken` bodies (shape-generic over `[rows, vocab]`, `dims =
//! [vocab]`), the greedy pick as two stages (`ArgmaxChunk` reduces each chunk of a row in its own
//! workgroup into a partial, `ArgmaxCombine` reduces a row's partials; POOT-1146), one thread per output
//! element for `RandomUniform` (shape-generic over `[rows, cols]`, `dims = [cols]`). Authored in Rust (`pootc/kernels/sampling/*.rs`); `poot-wgpu-device` lowers each to SPIR-V,
//! `poot-ptx-device` to NVPTX and `poot-rocm-device` to AMDGCN, so there is one kernel definition and one copy
//! of each asset. See `planner::sampling` for their metadata-launch lowering.

kernel_assets! {
    family = body_sampling;

    ArgmaxChunk {
        name: "argmax_chunk",
        source: "argmax_chunk",
        entry: "__poot_kernel_argmax_chunk",
        params: 3,
        dest: GraphPlan,
    },
    ArgmaxCombine {
        name: "argmax_combine",
        source: "argmax_combine",
        entry: "__poot_kernel_argmax_combine",
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

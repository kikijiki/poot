//! Flash-attention imported kernels (card 044, spec 055): decode and prefill, GQA + masked online
//! softmax, each holding its running output `o[D]` in an LDS array (the kernelgen kernel uses a private
//! array, unimportable and SPIR-V-crashing, hence NVPTX-only there; LDS makes these importable on every
//! backend). Authored in Rust (`pootc/kernels/attention/*.rs`). 561a retires this family into kernelgen.

kernel_assets! {
    family = body_attention;

    FlashDecode {
        name: "flash_decode",
        source: "flash_decode",
        entry: "__poot_kernel_flash_decode",
        params: 6,
        dest: GraphPlan,
    },
    FlashPrefill {
        name: "flash_prefill",
        source: "flash_prefill",
        entry: "__poot_kernel_flash_prefill",
        params: 6,
        dest: GraphPlan,
    },
}

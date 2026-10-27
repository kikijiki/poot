//! Packed-reader imported kernels (card 044): the packed BF16/E4M3FN embedding row gathers (cards 381,
//! 449 D1) and the BF16 const widening cast, which read checkpoint bytes as packed `u32` lanes. Authored in
//! Rust (`pootc/kernels/packed/*.rs`).

kernel_assets! {
    kind = ImportedKernel;
    family = body_packed;

    PackedBf16RowGather {
        name: "packed_bf16_row_gather",
        source: "packed_bf16_row_gather",
        entry: "__poot_kernel_packed_bf16_row_gather",
        params: 3,
        dest: GraphPlan,
    },
    PackedBf16ToF32 {
        name: "packed_bf16_to_f32",
        source: "packed_bf16_to_f32",
        entry: "__poot_kernel_packed_bf16_to_f32",
        params: 2,
        dest: GraphPlan,
    },
    PackedE4m3RowGather {
        name: "packed_e4m3_row_gather",
        source: "packed_e4m3_row_gather",
        entry: "__poot_kernel_packed_e4m3_row_gather",
        params: 3,
        dest: GraphPlan,
    },
}

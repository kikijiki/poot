//! Packed-reader imported kernels (card 044): the I8 KV-cache pack/unpack pair (spec 048) and the
//! packed BF16/E4M3FN embedding row gathers (cards 381, 449 D1) - the dense BF16 contraction bodies
//! live in [`crate::imported::contraction`] (card 380 shares that family's "read checkpoint bytes as
//! packed u32 lanes" shape with the GEMV/GEMM bodies there). Authored in Rust
//! (`pootc/kernels/packed/*.rs`).

kernel_assets! {
    family = body_packed;

    PackI8 {
        name: "pack_i8",
        source: "pack_i8",
        entry: "__poot_kernel_pack_i8",
        params: 3,
        dest: GraphPlan,
    },
    UnpackI8 {
        name: "unpack_i8",
        source: "unpack_i8",
        entry: "__poot_kernel_unpack_i8",
        params: 3,
        dest: GraphPlan,
    },
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

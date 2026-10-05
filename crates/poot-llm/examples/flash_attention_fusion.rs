//! poot fuses a model's materialized softmax attention into a single flash-attention op automatically, as a
//! compiler pass over the primitive graph: `transform::flash_attention` pattern-matches the
//! `softmax(scale*QKt + mask) @ V` decomposition and rewrites it. This example traces a tiny decode step, runs
//! the default optimization pipeline, and prints the dispatch-count and peak-activation-memory it saves (poot's
//! own cost model, no GPU or vendor API).
//!
//! Run: `cargo run -p poot-llm --example flash_attention_fusion` (CPU only, no GPU or weights needed).

use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::OpKind;
use poot_models::model::{LogitRows, Phase};

use poot_graph_plan::passes_without_target as optimize;

const LAYERS: usize = 4;

fn main() {
    // Tiny dims, same structure as Qwen2.5.
    let model = Dense::new(Family::Qwen2)
        .vocab(256)
        .dims(64, 128, LAYERS)
        .heads(8, 4)
        .head_dim(8)
        .max_positions(64)
        .model();
    let cap = 16;
    let g = plain(
        model
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .expect("trace the decode step"),
    );

    // Default pipeline without flash (cse + pointwise/row fusion) vs with it (`optimize` = cse ->
    // flash_attention -> dce -> fuse). flash_attention collapses each layer's materialized attention chain
    // (transpose K, QKt matmul, scale, mask add, softmax, PV matmul) into one FlashAttentionDecode op.
    let no_flash = poot_graph_plan::fuse(&poot_graph_plan::cse(&g));
    let with_flash = optimize(&g);

    let n_flash = with_flash
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionDecode { .. }))
        .count();

    let (d0, m0) = (
        poot_graph_ir::analysis::dispatch_count(&no_flash),
        poot_graph_ir::analysis::peak_transient_bytes(&no_flash),
    );
    let (d1, m1) = (
        poot_graph_ir::analysis::dispatch_count(&with_flash),
        poot_graph_ir::analysis::peak_transient_bytes(&with_flash),
    );

    println!(
        "traced a {}-layer decode step (cap={cap}) into the primitive graph.",
        LAYERS
    );
    println!(
        "flash_attention rewrote {n_flash} attention chains -> {n_flash} FlashAttentionDecode ops (one per layer).\n"
    );
    println!("                       no flash    with flash");
    println!(
        "  GPU dispatches/token   {d0:>7}    {d1:>8}   ({} fewer, {} per layer)",
        d0 - d1,
        (d0 - d1) / LAYERS
    );
    println!("  peak activation bytes  {m0:>7}    {m1:>8}");
    println!(
        "\nflash is the engine DEFAULT (transform::optimize): every attention tracer flashes for free, on \nboth the wgpu and NVPTX backends, with the same logits (invariant 6)."
    );
}

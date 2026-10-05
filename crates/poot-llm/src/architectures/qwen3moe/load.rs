//! Qwen3-MoE GGUF loading helpers.

use poot_load::gguf::GgufIndex;
use poot_models::qwen2::Qwen2Config;

/// Every layer index whose `blk.{li}.ffn_gate_inp.weight` is present: the qwen3-moe per-layer dense/MoE
/// switch, which GGUF metadata has no key for (see `Qwen3MoeParams` in `runner.rs`). `pub(crate)`:
/// `Runner::load_gguf` computes it once and reuses it for the `Qwen3MoeParams` it builds.
pub(crate) fn qwen3moe_sparse_layers(g: &GgufIndex, cfg: &Qwen2Config) -> Vec<bool> {
    (0..cfg.layers)
        .map(|li| {
            g.tensors
                .contains_key(&format!("blk.{li}.ffn_gate_inp.weight"))
        })
        .collect()
}

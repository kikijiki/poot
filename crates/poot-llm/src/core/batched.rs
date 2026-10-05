//! Batched and continuous-batching decode plumbing (shared-pool and paged batched steps).

#[cfg(test)]
use poot_graph_ir::Graph;

use crate::core::runner::Runner;
use crate::error::{OptionExt, Result};

impl Runner {
    /// Trace the shared-pool paged batched decode graph: `n_slots` decode rows sharing one
    /// `[pool_slots, Hkv, D]` K/V pool per layer (vs a private `[n_slots, Hkv, cap, D]` cache each), addressed by a
    /// `[n_slots, cap]` global slot map. Companion to [`Self::batched_decode_step_paged`]; sized once for a server
    /// engine to replay with a `driver::block_table::PagedKvCache` driving the per-step slot map. `pool_slots` should be
    /// `num_blocks * block_table::BLOCK_SIZE`. granitemoe/qwen3_moe via
    /// [`poot_models::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool`]; the families without a
    /// shared-pool tracer are rejected below, and none has `quant_kv` support (rejected explicitly).
    ///
    /// Mixtral routes to its own dense (non-pooled) [`poot_models::mixtral::trace_mixtral_decode_kv_masked_batched_shared_pool`],
    /// checked as `self.mixtral.is_some()` directly rather than folded into `is_moe()` (whose other callers,
    /// poot-serve's engine-selection gate and `BatchDecodable` reporting, assume "is_moe" implies a shared-pool
    /// prefill tracer via `Self::prefill_kv_graph_shared_pool`, which Mixtral lacks; see
    /// `Self::decode_kv_graph_shared_pool`'s doc).
    ///
    /// Dense (non-MoE-pooled) DeepSeek-V3 routes to
    /// [`poot_models::deepseek3::trace_deepseek3_decode_kv_masked_batched_shared_pool`], checked as
    /// `self.deepseek3.is_some()` directly for the same reason as Mixtral. F32 only, decode only; `quant_kv` is
    /// rejected below like MoE/Mixtral.
    #[cfg(test)]
    pub(crate) fn trace_batched_shared_pool_decode(
        &self,
        cap: usize,
        n_slots: usize,
        pool_slots: usize,
        quant_kv: bool,
    ) -> Result<Graph> {
        self.shared_pool_arch()?;
        let is_moe = self.is_moe();
        let is_mixtral = self.mixtral.is_some();
        let is_deepseek3 = self.deepseek3.is_some();
        if (is_moe || is_mixtral || is_deepseek3) && quant_kv {
            bail!(
                "shared-pool batched decode KV quantization (POOT_KV_QUANT) is not supported for \
                 granitemoe/qwen3_moe/mixtral/deepseek3 (spec 249 \"Decode counterpart\"/spec \
                 266-batched/spec 269: none of the MoE/Mixtral/MLA decode tracers have a quant_kv knob \
                 this round)"
            );
        }
        // The traced graph is returned as traced: `compile` is the one pass pipeline (Card 626), and a graph
        // already run through the pre-compile passes reaches it with its bias adds fused ahead of the packed
        // claims, which match the unfused `PackedDequant -> Transpose -> MatMul` chain (Card 557).
        let g = self.decode_kv_graph_shared_pool(cap, n_slots.max(1), pool_slots, quant_kv)?;
        Ok(g)
    }

    /// Host-compute a `Slot::TokenEmbed` input: one already-embedded row per token id in `tokens`,
    /// gathered from the host-resident dense embedding table `poot_graph_plan::legalize`
    /// (card 523a) hosted, in order. The caller wraps the flat `tokens.len() * hidden` payload in the
    /// graph's own slot shape (`[hidden]` for one row, `[batch,hidden]` / `[l,hidden]` for several).
    /// `name` is the hosted `Slot::TokenEmbed` value's own name: legalize names it after the constant
    /// it replaced, so it is whatever checkpoint name that tracer's dense embed gather declared
    /// (`"model.embed_tokens.weight"` for every current qwen2-shaped tracer), not a name this function
    /// assumes.
    pub(crate) fn gather_token_embed_rows(&self, name: &str, tokens: &[u32]) -> Result<Vec<f32>> {
        let h = self.cfg.hidden;
        let embed = self
            .dense_weight(name)
            .map_err(|error| err!("host-embed needs {name} resident: {error}"))?;
        let mut rows = Vec::with_capacity(tokens.len() * h);
        for &tok in tokens {
            let start = tok as usize * h;
            let row = embed
                .as_f32()
                .unwrap()
                .get(start..start + h)
                .with_context(|| format!("token {tok} out of embedding range"))?;
            rows.extend_from_slice(row);
        }
        Ok(rows)
    }
}

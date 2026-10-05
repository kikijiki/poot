//! CPU-oracle generation loops and the shared decode/prefill binders.

use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_models::deepseek2::trace_deepseek2_prefill;
use poot_models::deepseek3::trace_deepseek3_prefill;
use poot_models::deepseek32::trace_deepseek32_dsa_prefill;
use poot_models::gpt_oss::trace_gptoss_prefill;
use poot_models::granite::trace_granite_prefill;
use poot_models::mixtral::trace_mixtral_prefill;
use poot_models::nemotron_h::trace_nemotron_h_prefill;
use poot_models::olmoe::trace_olmoe_prefill;
use poot_models::qwen3moe::trace_qwen3_moe_prefill;
use poot_tensor::HostTensor;

use crate::GenerationControl;

use crate::core::decode_arch::DecodeArch;
use crate::core::graphs::{
    computed_const_tensor, decode_mask_row, prefill_causal_mask, prefill_pos_rows,
};
use crate::core::runner::Runner;
use crate::core::sampler::{Sampler, SamplerFault, check_pickable};
use crate::error::{OptionExt, Result, ResultExt};
use crate::multimodal::mrope::{bind_mrope_decode_position, bind_mrope_prefill_positions};

impl Runner {
    /// One greedy next-token id for the given context (a full prefill forward) on the CPU executor.
    #[cfg(test)]
    pub(crate) fn next_token(&self, tokens: &[u32]) -> Result<u32> {
        let g = self.stateless_prefill_graph(tokens.len())?;
        let inputs = self.bind(&g, tokens)?;
        let logits = crate::core::cpu_oracle::cpu_eval(&g, &inputs).context("eval")?;
        Ok(argmax(logits.as_f32().unwrap())? as u32)
    }

    /// Greedily generate up to `max_new` tokens, calling `on_token` with each new token's text piece.
    /// Returns the full token id sequence (prompt + generated).
    pub fn generate(
        &self,
        prompt: &str,
        max_new: usize,
        on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        self.generate_sampled(prompt, max_new, &mut Sampler::greedy(), &[], on_token)
    }

    /// The arch's stateless full-sequence prefill graph over `n` tokens: no KV state, `[vocab]` logits
    /// for the last position. Traced by every re-prefill CPU generate path. The exhaustive [`DecodeArch`]
    /// match keeps an architecture from silently inheriting Qwen2 stateless prefill.
    ///
    /// The traced graph leaves through [`Runner::bind_storage`] (card 545a).
    pub(crate) fn stateless_prefill_graph(&self, n: usize) -> Result<poot_graph_ir::Graph> {
        let g = match self.decode_arch()? {
            DecodeArch::GraniteMoe => trace_granite_prefill(
                self.cfg,
                self.granite_moe
                    .expect("DecodeArch::GraniteMoe implies granite params"),
                n,
            ),
            DecodeArch::Qwen3Moe => trace_qwen3_moe_prefill(
                self.cfg,
                self.qwen3_moe
                    .as_ref()
                    .expect("DecodeArch::Qwen3Moe implies qwen3_moe params")
                    .clone(),
                n,
            ),
            // Every layer routes through MoE.
            DecodeArch::Mixtral => trace_mixtral_prefill(
                self.cfg,
                self.mixtral
                    .expect("DecodeArch::Mixtral implies mixtral params"),
                n,
            ),
            // Every layer routes through MoE.
            DecodeArch::Olmoe => trace_olmoe_prefill(
                self.cfg,
                self.olmoe.expect("DecodeArch::Olmoe implies olmoe params"),
                n,
            ),
            // Every layer routes through MoE.
            DecodeArch::GptOss => trace_gptoss_prefill(
                self.cfg,
                self.gpt_oss
                    .as_ref()
                    .expect("DecodeArch::GptOss implies gpt_oss params")
                    .clone(),
                n,
            ),
            // Uses its own DeepseekV2Config, not `self.cfg` (MLA does not map onto Qwen2Config).
            DecodeArch::DeepseekV2 => {
                let dp = self
                    .deepseek2
                    .expect("DecodeArch::DeepseekV2 implies deepseek2 params");
                trace_deepseek2_prefill(dp.cfg, dp.moe, n)
            }
            DecodeArch::DeepseekV3 => {
                let dp = self
                    .deepseek3
                    .expect("DecodeArch::DeepseekV3 implies deepseek3 params");
                trace_deepseek3_prefill(dp.cfg, dp.moe, n)
            }
            // Same `DeepseekV2Config`/`DeepseekV3MoeParams` as `deepseek3`, plus `DsaConfig` for the
            // Lightning Indexer.
            DecodeArch::DeepseekV32 => {
                let dp = self
                    .deepseek32
                    .expect("DecodeArch::DeepseekV32 implies deepseek32 params");
                trace_deepseek32_dsa_prefill(dp.cfg, dp.dsa, dp.moe, n)
            }
            // Own config/tracer pair.
            DecodeArch::NemotronH => trace_nemotron_h_prefill(
                self.nemotron_h
                    .as_ref()
                    .expect("DecodeArch::NemotronH implies nemotron_h config"),
                n,
            ),
        };
        self.bind_storage(g)
    }

    /// Like [`Self::generate`] but with a [`Sampler`] (greedy when temperature is 0) and `stops`.
    /// Re-prefills each step on the CPU executor. Stops on EOS, `max_new`, or when the generated text
    /// contains one of `stops` (empty disables the check).
    pub fn generate_sampled(
        &self,
        prompt: &str,
        max_new: usize,
        sampler: &mut Sampler,
        stops: &[String],
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = self.encode(prompt)?;
        let gen_start = tokens.len();
        // The penalties see the prompt context plus each generated token.
        sampler.seed_context(&tokens);
        for _ in 0..max_new {
            let g = self.stateless_prefill_graph(tokens.len())?;
            let inputs = self.bind(&g, &tokens)?;
            let logits = crate::core::cpu_oracle::cpu_eval(&g, &inputs).context("eval")?;
            let next = sampler.pick(logits.as_f32().unwrap())? as u32;
            if next == self.eos {
                break;
            }
            tokens.push(next);
            sampler.observe(next);
            if on_token(&self.stream_piece(&tokens, gen_start)?).is_break() {
                break;
            }
            if self.hit_stop(&tokens[gen_start..], stops)? {
                break;
            }
        }
        Ok(tokens)
    }

    /// Whether the decoded `generated` token slice contains any non-empty stop string. Returns false
    /// immediately (no decode) when `stops` is empty, keeping the no-stop hot path allocation-free.
    pub(crate) fn hit_stop(&self, generated: &[u32], stops: &[String]) -> Result<bool> {
        if stops.is_empty() {
            return Ok(false);
        }
        let text = self.decode(generated)?;
        Ok(stops
            .iter()
            .any(|s| !s.is_empty() && text.contains(s.as_str())))
    }

    /// Greedily generate by re-prefilling each step on the executor contract. Unlike
    /// [`Self::generate_kv_gpu_cached`] it re-traces the full prefill per step, so it works for any arch
    /// the Runner traces, at O(n^2) cost. Each
    /// step's shape differs (the growing prompt), so each step stages, steps and removes its own entry
    /// (the contract admits no caller-side pipeline cache).
    #[cfg(test)]
    pub(crate) fn generate_gpu_reprefill(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        // DSA is CPU-oracle only (see the `deepseek32` field): no GPU kernelgen exists for the
        // Lightning Indexer's top-k. Bail rather than fall through to the plain-qwen2 tracer, since
        // `self.cfg` is not populated for a DSA Runner.
        if self.deepseek32.is_some() {
            bail!(
                "generate_gpu_reprefill: DeepSeek-V3.2 DSA has no GPU dispatch yet (CPU-oracle only, \
                 spec 277) - use Runner::generate/generate_sampled/generate_kv_masked (CPU) for this \
                 checkpoint"
            );
        }
        let mut tokens = self.encode(prompt)?;
        let mut sampler = Sampler::greedy();
        for _ in 0..max_new {
            let g = self.stateless_prefill_graph(tokens.len())?;
            let staged = crate::backends::gpu_generate::staged_program(
                &g,
                crate::backends::gpu_generate::executor_target(exec),
            )
            .context("compiling the reprefill graph for the executor contract")?;
            let entry = exec
                .add_entry(exe, &staged)
                .context("add_entry (reprefill)")?;
            let bound = self.bind(&g, &tokens)?;
            let inputs = crate::backends::gpu_generate::slot_step_inputs(&g, &bound)?;
            let result = exec
                .step(exe, entry, &inputs, &mut poot_executor::NoSync)
                .context("gpu reprefill (executor contract)")
                .and_then(|mut outputs| outputs.read().context("gpu reprefill readback"));
            match exec.remove_entry(exe, entry) {
                Ok(()) => {}
                Err(e) if result.is_ok() => return Err(e).context("remove_entry (reprefill)"),
                Err(_) => {}
            }
            let bytes = result?;
            let logits: &[f32] = bytemuck::cast_slice(&bytes);
            let next = sampler.pick(logits)? as u32;
            if next == self.eos {
                break;
            }
            tokens.push(next);
            if on_token(&self.decode(&[next])?).is_break() {
                break;
            }
        }
        Ok(tokens)
    }

    /// Greedily generate via the constant-shape masked fixed-KV decode: the graph is built once
    /// (the arch's masked decode tracer, identical for every position, as the PTX capture path needs),
    /// attends over the full `cap` cache with a per-token additive mask, and writes each token's K/V
    /// at the runtime `pos` slot. CPU eager executor; the host-verifiable counterpart of the
    /// captured PTX path.
    #[cfg(test)]
    pub(crate) fn generate_kv_masked(
        &self,
        prompt: &str,
        max_new: usize,
        on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let tokens = self.encode(prompt)?;
        self.generate_kv_masked_tokens(tokens, max_new, on_token)
    }

    /// [`Self::generate_kv_masked`] on already-tokenized input. For callers that hold token ids or a
    /// synthetic-weight `Runner` with no tokenizer (e.g. the greedy reference for `width = 1` beam
    /// search).
    #[cfg(test)]
    pub(crate) fn generate_kv_masked_tokens(
        &self,
        prompt_tokens: Vec<u32>,
        max_new: usize,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = prompt_tokens;
        let cap = tokens.len() + max_new;
        let g = self.decode_masked_graph(cap)?; // built ONCE
        let mut caches: Vec<HostTensor> = g
            .state
            .iter()
            .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
            .collect();
        let mut generated = 0;
        let mut pos = 0;
        loop {
            let inputs = self.bind_decode(&g, tokens[pos], pos, &caches, None)?;
            let (logits, new_caches) = crate::core::cpu_oracle::cpu_eval_with_state(&g, &inputs)
                .context("masked kv decode eval")?;
            caches = new_caches;
            if pos + 1 == tokens.len() {
                let next = argmax(logits.as_f32().unwrap())? as u32;
                if next == self.eos {
                    break;
                }
                tokens.push(next);
                if on_token(&self.decode(&[next])?).is_break() {
                    break;
                }
                generated += 1;
                if generated >= max_new {
                    break;
                }
            }
            pos += 1;
        }
        Ok(tokens)
    }

    /// Bind a decode graph: scalar token / pos / seq_len slots, weights by name, and the
    /// carried cache buffers (in `g.state` order).
    pub(crate) fn bind_decode(
        &self,
        g: &Graph,
        token: u32,
        pos: usize,
        caches: &[HostTensor],
        slot_map: Option<&[u32]>,
    ) -> Result<HashMap<ValueId, Value>> {
        self.bind_decode_with_mrope(g, token, pos, caches, slot_map, None)
    }

    /// Bind a Qwen2-VL-class decode graph with one packed temporal/height/width position tuple.
    /// Other decode callers use [`Self::bind_decode`]; multimodal positions are an explicit argument
    /// rather than hidden Runner state. No production caller wires the mRoPE decode path yet, so this
    /// is test-only until the VLM route lands.
    #[cfg(test)]
    pub(crate) fn bind_decode_mrope_inputs(
        &self,
        g: &Graph,
        token: u32,
        pos: usize,
        caches: &[HostTensor],
        position: poot_models::mrope::MropePosition,
    ) -> Result<HashMap<ValueId, Value>> {
        self.bind_decode_with_mrope(g, token, pos, caches, None, Some(position))
    }

    fn bind_decode_with_mrope(
        &self,
        g: &Graph,
        token: u32,
        pos: usize,
        caches: &[HostTensor],
        slot_map: Option<&[u32]>,
        mrope_position: Option<poot_models::mrope::MropePosition>,
    ) -> Result<HashMap<ValueId, Value>> {
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match meta.storage {
                Storage::Slot(Slot::Token) => crate::core::graphs::token_slot(&meta.aval, &[token]),
                Storage::Slot(Slot::Activation) => {
                    bail!("standalone Activation slots are not model-engine decode inputs")
                }
                // Card 550: the dense families' decode `Slot::Pos` is now declared `[1,1]` (still one
                // row, one token), not a bare scalar `[]`; fill by element count so either declared
                // shape (the pre-card scalar of an unconverted family, or the new `[1,1]`) binds the
                // same single value.
                Storage::Slot(Slot::Pos) => {
                    let n = meta.aval.numel().max(1);
                    HostTensor::i32(meta.aval.shape.clone(), vec![pos as i32; n])
                }
                Storage::Slot(Slot::MropePosition) => continue,
                Storage::Slot(Slot::SeqLen) => HostTensor::i32(vec![], vec![(pos + 1) as i32]),
                // Paged decode: the per-step logical->physical slot map from a `block_table::BlockTable`.
                // Only present on the paged graph.
                Storage::Slot(Slot::SlotMap) => {
                    let sm = slot_map.context("paged graph needs a slot_map (none supplied)")?;
                    HostTensor::i32(
                        meta.aval.shape.clone(),
                        sm.iter().map(|&x| x as i32).collect(),
                    )
                }
                // The single-sequence qwen2 decode graph never declares a GDN pool row map; return a
                // typed error rather than panic in a library binder.
                Storage::Slot(Slot::GdnSlotMap) => {
                    bail!(
                        "GdnSlotMap binds only on the batched qwen3next decode path (card 188); \
                         the single-sequence qwen2 decode graph never declares one"
                    )
                }
                // The per-row adapter selection belongs to a batched LoRA graph; this binder's
                // single-sequence graph never declares one.
                Storage::Slot(Slot::LoraIdx) => {
                    bail!(
                        "LoraIdx binds only on a batched LoRA graph, never a single-sequence decode"
                    )
                }
                Storage::Slot(Slot::ExpertPoolMap) => {
                    unreachable!(
                        "ExpertPoolMap binds only on the pooled-MoE decode path (spec 266 phase 1), and is \
                         resolved per value NAME (Builder::slot_named), never per slot kind"
                    )
                }
                // The constant-shape masked decode attends over the full cache with this additive
                // mask: 0 for visible slots, -1e9 for masked slots.
                // sliding_window=None: visible iff t <= pos. Some(w): visible iff t <= pos && pos-t < w.
                Storage::Slot(Slot::Mask) => {
                    let cap = meta.aval.shape.iter().product::<usize>().max(1);
                    HostTensor::f32(
                        meta.aval.shape.clone(),
                        decode_mask_row(cap, pos, self.sliding_window),
                    )
                }
                // Single-row case of `bind_decode_batched`'s `Slot::TokenEmbed` arm.
                Storage::Slot(Slot::TokenEmbed) => HostTensor::f32(
                    meta.aval.shape.clone(),
                    self.gather_token_embed_rows(
                        meta.name
                            .as_deref()
                            .context("TokenEmbed slot without a name")?,
                        &[token],
                    )?,
                ),
                Storage::State => continue, // bound below in state order
                Storage::Computed(computed) => computed_const_tensor(computed),
                Storage::Const => {
                    let name = meta.name.as_deref().context("const without a name")?;
                    inputs.insert(id, self.weight_value(name, &meta.aval)?);
                    continue;
                }
                Storage::Slot(Slot::Sampler) => {
                    bail!(
                        "Sampler binds only on a graph with a card-551b-appended sampling suffix; \
                         this binder's graph never declares one"
                    )
                }
                Storage::Device => bail!("device value in input set"),
            };
            inputs.insert(id, t.into());
        }
        match mrope_position {
            Some(position) => bind_mrope_decode_position(g, &mut inputs, position)?,
            None if g.slots.iter().any(|(_, slot)| *slot == Slot::MropePosition) => {
                bail!(
                    "mRoPE decode graph needs a typed position; no production route wires the mRoPE \
                     binder yet"
                )
            }
            None => {}
        }
        // The CPU path binds the carried cache tensors here; the GPU path passes `&[]` and binds the
        // resident device buffers via `run_resident_kv`.
        if !caches.is_empty() {
            for (ci, &(si, _)) in g.state.iter().enumerate() {
                inputs.insert(si, caches[ci].clone().into());
            }
        }
        Ok(inputs)
    }

    pub(crate) fn bind(&self, g: &Graph, tokens: &[u32]) -> Result<HashMap<ValueId, Value>> {
        self.bind_with_mrope(g, tokens, None)
    }

    /// Bind a Qwen2-VL-class prefill graph with host-constructed packed mRoPE positions. No production
    /// caller wires the mRoPE prefill path yet, so this is test-only until the VLM route lands.
    #[cfg(test)]
    pub(crate) fn bind_mrope_prefill_inputs(
        &self,
        g: &Graph,
        tokens: &[u32],
        positions: &poot_models::mrope::MropePositionIds,
    ) -> Result<HashMap<ValueId, Value>> {
        self.bind_with_mrope(g, tokens, Some(positions))
    }

    fn bind_with_mrope(
        &self,
        g: &Graph,
        tokens: &[u32],
        mrope_positions: Option<&poot_models::mrope::MropePositionIds>,
    ) -> Result<HashMap<ValueId, Value>> {
        let l = tokens.len();
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match meta.storage {
                Storage::Slot(Slot::Token) => {
                    let declared = meta.aval.shape[0];
                    if declared != l {
                        bail!("token slot declares {declared} tokens but {l} were supplied");
                    }
                    crate::core::graphs::token_slot(&meta.aval, tokens)
                }
                Storage::Slot(Slot::MropePosition) => continue,
                // Same host gather as `bind_prefill_kv`'s `Slot::TokenEmbed` arm: one row per prompt token.
                Storage::Slot(Slot::TokenEmbed) => HostTensor::f32(
                    meta.aval.shape.clone(),
                    self.gather_token_embed_rows(
                        meta.name
                            .as_deref()
                            .context("TokenEmbed slot without a name")?,
                        tokens,
                    )?,
                ),
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().context("mask slot without a name")?;
                    match name {
                        "mask.prefill" => {
                            prefill_causal_mask(meta.aval.shape[2], self.sliding_window)
                        }
                        // gpt-oss's alternating sliding/full schedule: the sliding layers' mask.
                        "mask.prefill.local" => prefill_causal_mask(
                            l,
                            self.gpt_oss.as_ref().map(|gp| gp.sliding_window),
                        ),
                        other => bail!("unexpected mask slot {other} in prefill graph"),
                    }
                }
                // Card 550: the mask is a graph computation over `Slot::Pos` and `iota`; this one-shot
                // prefill always starts at position 0.
                Storage::Slot(Slot::Pos) => prefill_pos_rows(&meta.aval, l),
                Storage::Slot(other) => bail!("unexpected slot {other:?} in prefill graph"),
                // A from-scratch re-prefill carries no prior state, so `Storage::State` is zeros.
                Storage::State => HostTensor::zeros(meta.aval.shape.clone()),
                // A compiled graph carries each folded `Iota` (compile's `fold_iota`) as a self-describing input.
                Storage::Computed(computed) => computed_const_tensor(computed),
                Storage::Const => {
                    let name = meta.name.as_deref().context("const without a name")?;
                    inputs.insert(id, self.weight_value(name, &meta.aval)?);
                    continue;
                }
                Storage::Device => bail!("device value in input set"),
            };
            inputs.insert(id, t.into());
        }
        match mrope_positions {
            Some(positions) => bind_mrope_prefill_positions(g, &mut inputs, positions)?,
            None if g.slots.iter().any(|(_, slot)| *slot == Slot::MropePosition) => {
                bail!(
                    "mRoPE prefill graph needs typed positions; no production route wires the mRoPE \
                     binder yet"
                )
            }
            None => {}
        }
        Ok(inputs)
    }
}

/// Upper bound on a single-sequence generation's KV span. The
/// single-sequence drivers size a KV allocation from it, so an unbounded `prompt_len` or `max_new`
/// would allocate multiple TB and abort, and a value near `usize::MAX` would overflow (release builds
/// skip the check) into an undersized buffer and an out-of-bounds GPU write. A sanity ceiling
/// (256K), not a per-model context bound: a near-ceiling span may still not fit the device.
pub(crate) const MAX_PREFILL_CAP: usize = 262_144;

/// The index of the largest logit (the first on a tie). A [`SamplerFault`] when `v` holds a NaN or `+inf`, or no
/// finite element: an unchecked maximum turns an all-NaN row into token 0.
pub(crate) fn argmax(v: &[f32]) -> Result<usize, SamplerFault> {
    check_pickable(v)?;
    let mut best = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    Ok(best)
}

/// Tests for the mRoPE arms of `Runner::bind_decode` and `Runner::bind` through the production binder
/// with a tiny synthetic-weight `Runner` (the VLM binders the Runner keeps until POOT-739).
#[cfg(test)]
pub(crate) mod mrope_binder_tests {
    use super::*;
    use crate::core::runner::LoraHotState;

    use poot_models::mrope::{MropeGrid, MropeSegment, build_mrope_position_ids};
    use poot_models::qwen2::{Qwen2Config, trace_decode_mrope, trace_prefill_mrope};
    use poot_tensor::DType;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    /// A tiny GQA qwen2-shaped config.
    fn tiny_cfg() -> Qwen2Config {
        Qwen2Config {
            vocab: 32,
            hidden: 16,
            inter: 24,
            layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            rotary_dim: 4,
            eps: 1e-6,
            max_pos: 32,
            qkv_bias: true,
            qk_norm: false,
            ..Default::default()
        }
    }

    fn tiny_mrope_cfg() -> Qwen2Config {
        Qwen2Config {
            head_dim: 8,
            rotary_dim: 8,
            mrope_section: Some([1, 1, 2]),
            ..tiny_cfg()
        }
    }

    /// Deterministic pseudo-random fill in `[-1, 1)`, xorshift-seeded per name; the same algorithm as
    /// `poot-eval`'s `tests::helpers::fill`, which is `pub(super)` and cannot be imported. A weak
    /// hash-based mix gives near-zero output differences (<1e-7) regardless of the mask, so it
    /// cannot exercise the wiring.
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    use poot_test_util::seed_of;

    /// `[-0.1, 0.1)`-scaled synthetic data for a named weight (the scale `poot-eval`'s
    /// `qwen2_alibi_tracer_skips_rope_and_widens_mask` is proven sensitive at). Norm gamma weights get
    /// values near 1.0, since a near-zero gamma would shrink activations every layer.
    fn synth_data(name: &str, n: usize) -> Vec<f32> {
        let raw = fill(n, seed_of(name));
        if name.ends_with("norm.weight") {
            raw.iter().map(|v| 1.0 + v * 0.2).collect()
        } else {
            raw.iter().map(|v| v * 0.1).collect()
        }
    }

    /// RoPE cos/sin tables `[max_pos, head_dim]` (unused by the alibi path; `bind_decode`/`bind` only
    /// look up what the traced graph declares).
    fn rope_tables(cfg: &Qwen2Config) -> (HostTensor, HostTensor) {
        let d = cfg.head_dim;
        let half = d / 2;
        let p = cfg.max_pos;
        let theta = 1_000_000.0f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| theta.powf(-((2 * j) as f32) / d as f32))
            .collect();
        let mut cos = vec![0.0f32; p * d];
        let mut sin = vec![0.0f32; p * d];
        for pos in 0..p {
            for j in 0..half {
                let ang = pos as f32 * inv_freq[j];
                let (c, s) = (ang.cos(), ang.sin());
                cos[pos * d + j] = c;
                cos[pos * d + half + j] = c;
                sin[pos * d + j] = s;
                sin[pos * d + half + j] = s;
            }
        }
        (
            HostTensor::f32(vec![p, d], cos),
            HostTensor::f32(vec![p, d], sin),
        )
    }

    /// A fully-weighted `Runner` for `cfg`: every const the qwen2-shaped tracers declare, RoPE tables
    /// included. No checkpoint on disk.
    pub(crate) fn weighted_runner(cfg: Qwen2Config) -> Runner {
        let (q_dim, kv_dim) = (cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
        let mut weights = HashMap::new();
        let mut put = |name: String, shape: Vec<usize>| {
            let n: usize = shape.iter().product::<usize>().max(1);
            let data = synth_data(&name, n);
            weights.insert(name, HostTensor::f32(shape, data));
        };
        put(
            "model.embed_tokens.weight".to_string(),
            vec![cfg.vocab, cfg.hidden],
        );
        for li in 0..cfg.layers {
            let p = |s: &str| format!("model.layers.{li}.{s}");
            put(p("input_layernorm.weight"), vec![cfg.hidden]);
            put(p("self_attn.q_proj.weight"), vec![cfg.hidden, q_dim]);
            put(p("self_attn.k_proj.weight"), vec![cfg.hidden, kv_dim]);
            put(p("self_attn.v_proj.weight"), vec![cfg.hidden, kv_dim]);
            put(p("self_attn.o_proj.weight"), vec![q_dim, cfg.hidden]);
            if cfg.qkv_bias {
                put(p("self_attn.q_proj.bias"), vec![q_dim]);
                put(p("self_attn.k_proj.bias"), vec![kv_dim]);
                put(p("self_attn.v_proj.bias"), vec![kv_dim]);
            }
            put(p("post_attention_layernorm.weight"), vec![cfg.hidden]);
            put(p("mlp.gate_proj.weight"), vec![cfg.hidden, cfg.inter]);
            put(p("mlp.up_proj.weight"), vec![cfg.hidden, cfg.inter]);
            put(p("mlp.down_proj.weight"), vec![cfg.inter, cfg.hidden]);
        }
        put("model.norm.weight".to_string(), vec![cfg.hidden]);
        put("lm_head.weight".to_string(), vec![cfg.hidden, cfg.vocab]);
        let (cos, sin) = rope_tables(&cfg);
        weights.insert("rope.cos".to_string(), cos);
        weights.insert("rope.sin".to_string(), sin);

        let model = WordLevel::builder()
            .vocab(HashMap::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build empty wordlevel model");
        Runner {
            cfg,
            weights: crate::core::runner::dense_weight_map(weights),
            text: crate::text::tokenize::TextCodec::new(
                Tokenizer::new(model),
                None,
                None,
                u32::MAX,
                poot_models::chat::ChatFormat::ChatML,
                Default::default(),
            ),
            eos: u32::MAX,
            arch: "test".to_string(),
            granite_moe: None,
            qwen3_moe: None,
            mixtral: None,
            olmoe: None,
            gpt_oss: None,
            deepseek2: None,
            deepseek3: None,
            deepseek32: None,
            nemotron_h: None,
            formats: Default::default(),
            sliding_window: None,
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        }
    }

    /// `trace_decode_mrope` is the older growing-KV tracer and uses its G1 constant names. Alias
    /// the equivalent tiny fixture weights and add the two baked cache constants per layer so the
    /// decode wrapper can be exercised through CPU evaluation.
    fn add_mrope_decode_weights(runner: &mut Runner, kv_len: usize) {
        fn alias(weights: &mut HashMap<String, Value>, from: &str, to: String) {
            let tensor = weights
                .get(from)
                .unwrap_or_else(|| panic!("missing tiny fixture source weight {from}"))
                .clone();
            weights.insert(to, tensor);
        }

        alias(
            &mut runner.weights,
            "model.embed_tokens.weight",
            "embed_tokens".to_string(),
        );
        for li in 0..runner.cfg.layers {
            let standard = |suffix: &str| format!("model.layers.{li}.{suffix}");
            let legacy = |suffix: &str| format!("layers.{li}.{suffix}");
            for (from, to) in [
                ("input_layernorm.weight", "input_layernorm"),
                ("self_attn.q_proj.weight", "q_proj.w"),
                ("self_attn.q_proj.bias", "q_proj.b"),
                ("self_attn.k_proj.weight", "k_proj.w"),
                ("self_attn.k_proj.bias", "k_proj.b"),
                ("self_attn.v_proj.weight", "v_proj.w"),
                ("self_attn.v_proj.bias", "v_proj.b"),
                ("self_attn.o_proj.weight", "o_proj.w"),
                (
                    "post_attention_layernorm.weight",
                    "post_attention_layernorm",
                ),
                ("mlp.gate_proj.weight", "gate_proj.w"),
                ("mlp.up_proj.weight", "up_proj.w"),
                ("mlp.down_proj.weight", "down_proj.w"),
            ] {
                alias(&mut runner.weights, &standard(from), legacy(to));
            }
            for suffix in ["kv.cached_k", "kv.cached_v"] {
                let name = legacy(suffix);
                let shape = vec![1, runner.cfg.n_kv_heads, kv_len, runner.cfg.head_dim];
                runner.weights.insert(
                    name.clone(),
                    poot_eval::Value::from(HostTensor::f32(
                        shape,
                        synth_data(&name, runner.cfg.n_kv_heads * kv_len * runner.cfg.head_dim),
                    )),
                );
            }
        }
        alias(
            &mut runner.weights,
            "model.norm.weight",
            "model.norm".to_string(),
        );
        alias(
            &mut runner.weights,
            "lm_head.weight",
            "lm_head (tied)".to_string(),
        );
    }

    fn one_mrope_slot(graph: &Graph) -> ValueId {
        let ids: Vec<_> = graph
            .slots
            .iter()
            .filter_map(|&(id, slot)| (slot == Slot::MropePosition).then_some(id))
            .collect();
        assert_eq!(
            ids.len(),
            1,
            "graph must declare one authoritative mRoPE slot"
        );
        ids[0]
    }

    #[test]
    fn runner_mrope_prefill_wrapper_binds_packed_i32_slot_and_evals() {
        let cfg = tiny_mrope_cfg();
        let runner = weighted_runner(cfg);
        let positions = build_mrope_position_ids(
            &[
                MropeSegment::Text(1),
                MropeSegment::Image(MropeGrid::new(1, 2, 4)),
                MropeSegment::Text(1),
            ],
            2,
        )
        .expect("build mixed text/image positions");
        let tokens = vec![1, 2, 3, 4];
        let graph = trace_prefill_mrope(cfg, tokens.len());

        let inputs = runner
            .bind_mrope_prefill_inputs(&graph, &tokens, &positions)
            .expect("production mRoPE prefill wrapper must bind the real tracer");
        let position_id = one_mrope_slot(&graph);
        let packed = &inputs[&position_id];
        assert_eq!(graph.aval(position_id).dtype, DType::I32);
        assert_eq!(packed.as_host().expect("dense weight").shape(), vec![3, 4]);
        assert_eq!(
            packed.as_host().expect("dense weight").as_i32(),
            Some(&[0, 1, 1, 3, 0, 1, 1, 3, 0, 1, 2, 3][..])
        );
        assert_eq!(
            inputs.len(),
            graph.inputs.len(),
            "every graph input is bound"
        );

        let logits = crate::core::cpu_oracle::cpu_eval(&graph, &inputs)
            .expect("CPU eval through production prefill wrapper");
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );

        let err = runner
            .bind(&graph, &tokens)
            .expect_err("ordinary prefill binder must reject an mRoPE graph");
        assert_eq!(
            err.to_string(),
            "mRoPE prefill graph needs typed positions; no production route wires the mRoPE binder yet"
        );
    }

    #[test]
    fn runner_mrope_decode_wrapper_binds_packed_i32_slot_and_evals() {
        let cfg = tiny_mrope_cfg();
        let positions = build_mrope_position_ids(
            &[
                MropeSegment::Text(1),
                MropeSegment::Video(MropeGrid::new(1, 2, 4)),
                MropeSegment::Text(1),
            ],
            2,
        )
        .expect("build mixed text/video positions");
        let kv_len = positions.len();
        let mut runner = weighted_runner(cfg);
        add_mrope_decode_weights(&mut runner, kv_len);
        let graph = trace_decode_mrope(cfg, kv_len);

        let inputs = runner
            .bind_decode_mrope_inputs(&graph, 5, kv_len, &[], positions.next_text_position())
            .expect("production mRoPE decode wrapper must bind the real tracer");
        let position_id = one_mrope_slot(&graph);
        let packed = &inputs[&position_id];
        assert_eq!(graph.aval(position_id).dtype, DType::I32);
        assert_eq!(packed.as_host().expect("dense weight").shape(), vec![3]);
        assert_eq!(
            packed.as_host().expect("dense weight").as_i32(),
            Some(&[4, 4, 4][..])
        );
        assert_eq!(
            inputs.len(),
            graph.inputs.len(),
            "every graph input is bound"
        );

        let logits = crate::core::cpu_oracle::cpu_eval(&graph, &inputs)
            .expect("CPU eval through production decode wrapper");
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );

        let err = runner
            .bind_decode(&graph, 5, kv_len, &[], None)
            .expect_err("ordinary decode binder must reject an mRoPE graph");
        assert_eq!(
            err.to_string(),
            "mRoPE decode graph needs a typed position; no production route wires the mRoPE binder yet"
        );
    }
}

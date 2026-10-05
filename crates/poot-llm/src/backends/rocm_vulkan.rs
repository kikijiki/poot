//! ROCm/HSA backend generation loops, on the executor contract (`Engine<RocmDevice>`, Card 548).
//!
//! The pre-contract `RocmGraphExecutor`'s eager/recorded/capture-replay API split is gone: every
//! call here goes through `&mut dyn poot_executor::Executor` + `ExecutableId` (`Runner::load_on`
//! builds the executable once, from the loaded weights). The ordinary single-sequence decode loop
//! (the pre-contract `generate_kv_rocm`, no prefill) is now exactly
//! [`crate::Runner::generate_kv_gpu_cached`] - that function is backend-neutral already, so this
//! file no longer carries its own copy of it; callers that used to name `generate_kv_rocm` call
//! `generate_kv_gpu_cached` directly. What stays ROCm-local is what still has no generic,
//! contract-based counterpart anywhere in the tree yet: batched prefill-then-decode (prefill is one
//! captured-and-replayed-once entry; Card 546b has not moved wgpu's own prefill loop onto the
//! contract either), the masked special-arch decode family, and the stateless re-prefill parity
//! probe.

#[cfg(feature = "rocm")]
use crate::GenerationControl;
#[cfg(feature = "rocm")]
use crate::core::generate::argmax;
use crate::core::runner::Runner;
#[cfg(all(test, feature = "rocm"))]
use crate::core::sampler::Sampler;
#[cfg(feature = "rocm")]
use crate::error::{OptionExt, Result, ResultExt};

/// The one [`poot_graph_plan::Target`] `exec` drives (Card 546a's M3 contract: `Engine<D>` drives
/// exactly one device).
#[cfg(feature = "rocm")]
pub(crate) fn executor_target(exec: &dyn poot_executor::Executor) -> poot_graph_plan::Target {
    exec.target_set()
        .devices()
        .first()
        .expect("Card 546a's M3 Engine<D> always names exactly one device")
        .1
}

/// `g`, staged for the executor contract (always `Submission::Replay`: the contract admits no other
/// mode). `fusion` is [`poot_graph_plan::FusionPolicy::Full`] for the decode family;
/// [`Runner::rocm_prefill_fusion_policy`] decides it for prefill (cards 186/192's MoE-router-hang
/// guard).
#[cfg(feature = "rocm")]
pub(crate) fn staged_program(
    g: &poot_graph_ir::Graph,
    target: poot_graph_plan::Target,
    fusion: poot_graph_plan::FusionPolicy,
) -> std::result::Result<
    poot_graph_plan::StagedProgram<poot_graph_ir::ValidationOutputs>,
    poot_graph_plan::StagedCompileError,
> {
    let g = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &poot_graph_plan::CompileOptions {
            execution: poot_graph_plan::Submission::Replay,
            fusion,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
}

/// `bound`'s `Storage::Slot` entries only (Card 546a's contract binds every `Storage::Const`/
/// `Storage::State` value from the executable's `WeightStore`/carried state instead, through
/// `Runner::load_on` and the engine's own state buffers - never a per-step value), as the
/// `SlotKey`-keyed `StepInputs` the contract's `step` takes. Borrows `bound`'s tensors,
/// so the caller's `bind_*` result must outlive the returned `StepInputs`.
#[cfg(feature = "rocm")]
pub(crate) fn slot_step_inputs<'a>(
    g: &poot_graph_ir::Graph,
    bound: &'a std::collections::HashMap<poot_graph_ir::ValueId, poot_eval::Value>,
) -> Result<poot_executor::StepInputs<'a>> {
    let mut inputs = poot_executor::StepInputs::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if !matches!(meta.storage, poot_graph_ir::Storage::Slot(_)) {
            continue;
        }
        let key = meta
            .slot_key()
            .context("decode/prefill slot without a structured SlotKey")?
            .clone();
        let value = bound
            .get(&id)
            .context("bind_decode/bind_prefill_kv did not bind a declared slot")?;
        let tensor = value
            .as_host()
            .context("a slot value must be a dense tensor")?;
        inputs.push(key, tensor.shape(), tensor.view());
    }
    Ok(inputs)
}

impl Runner {
    // ---- ROCm/HSA backend ----

    /// Card 535a: whether the prefill graph needs cards 186/192's MoE-router-hang guard
    /// ([`poot_graph_plan::FusionPolicy::MoeHangGuard`]). One place, so the production calls and
    /// their acceptance test cannot drift apart.
    #[cfg(feature = "rocm")]
    fn rocm_prefill_fusion_policy(&self) -> poot_graph_plan::FusionPolicy {
        if self.is_moe() {
            poot_graph_plan::FusionPolicy::MoeHangGuard
        } else {
            poot_graph_plan::FusionPolicy::Full
        }
    }

    /// One batched Q=N prefill step: builds, stages, adds and steps the prefill entry exactly once
    /// (the contract's capture-on-first-step default), removes it, and returns the
    /// last position's logits. The KV state this writes is the executable's own, carried by (name,
    /// aval, storage) - a decode entry added on the same `exe` afterward sees it already filled
    /// (`prefill_kv_graph`/`decode_masked_graph` carry matching state pairs by construction).
    #[cfg(feature = "rocm")]
    fn run_rocm_prefill(
        &self,
        tokens: &[u32],
        cap: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        let pg = self.prefill_kv_graph(n, cap)?;
        let staged = staged_program(
            &pg,
            executor_target(exec),
            self.rocm_prefill_fusion_policy(),
        )
        .context("compiling the ROCm prefill graph for the executor contract")?;
        let entry = exec
            .add_entry(exe, &staged)
            .context("add_entry (rocm prefill)")?;
        let bound = self.bind_prefill_kv(&pg, tokens)?;
        let inputs = slot_step_inputs(&pg, &bound)?;
        let result = exec
            .step(exe, entry, &inputs, &mut poot_executor::NoSync)
            .context("rocm prefill step")
            .and_then(|mut out| out.read().context("rocm prefill readback"));
        exec.remove_entry(exe, entry)
            .context("remove_entry (rocm prefill)")?;
        let bytes = result?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Greedy decode from `tokens[n..]` on `exec`/`exe`'s already-filled KV state, through the
    /// standard single-token decode graph, continuing the running token/piece state `on_token` was
    /// already called with up to `tokens.len()`. Through `Runner::pick_step` (card 551b): the device
    /// suffix's typed non-finite fault applies here too, not just to sampled requests.
    #[cfg(feature = "rocm")]
    #[allow(clippy::too_many_arguments)]
    fn continue_decode_rocm(
        &self,
        tokens: &mut Vec<u32>,
        n: usize,
        cap: usize,
        max_new: usize,
        mut generated: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        ignore_eos: bool,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<()> {
        if generated >= max_new {
            return Ok(());
        }
        let dg = self.decode_masked_graph(cap)?;
        let mut entries = crate::core::decode_step::DecodeEntries::new(dg);
        let mut sampler = crate::core::sampler::Sampler::greedy();
        let result: Result<()> = (|| {
            let mut pos = n;
            loop {
                let next = self
                    .pick_step(
                        exec,
                        exe,
                        &mut entries,
                        &mut sampler,
                        tokens[pos],
                        pos,
                        true,
                    )?
                    .expect("committed pick_step always returns a token");
                if next == self.eos && !ignore_eos {
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
                pos += 1;
            }
            Ok(())
        })();
        match entries.remove_all(exec, exe) {
            Ok(()) => result,
            Err(e) if result.is_ok() => Err(e).context("remove_entry (decode after ROCm prefill)"),
            Err(_) => result,
        }
    }

    /// ROCm batched prefill (one Q=N forward fills the KV cache), then capture/replay decode. ROCm
    /// counterpart of [`Self::generate_kv_ptx_prefilled`].
    #[cfg(feature = "rocm")]
    pub fn generate_kv_rocm_prefilled(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = self.encode(prompt)?;
        let n = tokens.len();
        let cap = n + max_new;
        let logits = self.run_rocm_prefill(&tokens, cap, exec, exe)?;
        let mut generated = 0;
        let next = argmax(&logits)? as u32;
        if next == self.eos || max_new == 0 {
            return Ok(tokens);
        }
        tokens.push(next);
        if on_token(&self.decode(&[next])?).is_break() {
            return Ok(tokens);
        }
        generated += 1;
        self.continue_decode_rocm(
            &mut tokens,
            n,
            cap,
            max_new,
            generated,
            exec,
            exe,
            false,
            on_token,
        )?;
        Ok(tokens)
    }

    /// [`Self::generate_kv_rocm_prefilled`] from pre-encoded tokens; ROCm counterpart of
    /// [`Self::generate_kv_ptx_prefilled_tokens`]. Used by the decode-curve benchmark and shares its
    /// MoE guard.
    #[cfg(feature = "rocm")]
    pub fn generate_kv_rocm_prefilled_tokens(
        &self,
        context: &[u32],
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        ignore_eos: bool,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = context.to_vec();
        let n = tokens.len();
        let cap = n + max_new;
        let logits = self.run_rocm_prefill(&tokens, cap, exec, exe)?;
        let mut generated = 0;
        let next = argmax(&logits)? as u32;
        if (next == self.eos && !ignore_eos) || max_new == 0 {
            return Ok(tokens);
        }
        tokens.push(next);
        if on_token(&self.decode(&[next])?).is_break() {
            return Ok(tokens);
        }
        generated += 1;
        self.continue_decode_rocm(
            &mut tokens,
            n,
            cap,
            max_new,
            generated,
            exec,
            exe,
            ignore_eos,
            on_token,
        )?;
        Ok(tokens)
    }

    /// Masked fixed-KV decode on ROCm through [`Self::decode_masked_graph`], for the families
    /// without a contiguous prefill tracer (Mixtral, OlmoE, gpt-oss, DeepSeek-V2/MLA).
    #[cfg(feature = "rocm")]
    #[cfg(test)]
    pub(crate) fn generate_kv_gpu_masked_rocm(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = self.encode(prompt)?;
        let gen_start = tokens.len();
        let cap = tokens.len() + max_new;
        let g = self.decode_masked_graph(cap)?;
        let mut entries = crate::core::decode_step::DecodeEntries::new(g);
        let mut sampler = crate::core::sampler::Sampler::greedy();
        let result: Result<()> = (|| {
            let mut generated = 0;
            let mut pos = 0;
            loop {
                let at_gen = pos + 1 == tokens.len();
                let next = self.pick_step(
                    exec,
                    exe,
                    &mut entries,
                    &mut sampler,
                    tokens[pos],
                    pos,
                    at_gen,
                )?;
                if let Some(next) = next {
                    if next == self.eos {
                        break;
                    }
                    tokens.push(next);
                    if on_token(&self.stream_piece(&tokens, gen_start)?).is_break() {
                        break;
                    }
                    generated += 1;
                    if generated >= max_new {
                        break;
                    }
                }
                pos += 1;
            }
            Ok(())
        })();
        match entries.remove_all(exec, exe) {
            Ok(()) => result.map(|()| tokens),
            Err(e) if result.is_ok() => Err(e).context("remove_entry (masked special-arch decode)"),
            Err(_) => result.map(|()| tokens),
        }
    }

    /// Greedily generate by re-prefilling each step on the ROCm/HSA executor. ROCm counterpart of
    /// [`Self::generate_gpu_reprefill`] (wgpu) / [`Self::generate_ptx_reprefill`] (PTX). Re-traces
    /// and re-binds the growing prompt every step: a parity check, not a perf path. Each step is its
    /// own fresh entry (the contract's capture-once model has no benefit here, since the graph shape
    /// changes every step anyway).
    #[cfg(feature = "rocm")]
    #[cfg(test)]
    pub(crate) fn generate_rocm_reprefill(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        if self.deepseek32.is_some() {
            bail!(
                "generate_rocm_reprefill: DeepSeek-V3.2 DSA has no GPU dispatch yet (CPU-oracle only, \
                 spec 277) - use Runner::generate/generate_sampled/generate_kv_masked (CPU) for this \
                 checkpoint"
            );
        }
        if self.nemotron_h.is_some() {
            bail!(
                "generate_rocm_reprefill: Nemotron-H has no ROCm receipt yet (spec 279 / card 297: the \
                 wgpu path is verified, AMDGCN is not) - use Runner::generate_gpu_reprefill (wgpu) or \
                 Runner::generate/generate_sampled (CPU) for this checkpoint"
            );
        }
        let mut tokens = self.encode(prompt)?;
        let mut sampler = Sampler::greedy();
        for _ in 0..max_new {
            let g = self.stateless_prefill_graph(tokens.len())?;
            let staged =
                staged_program(&g, executor_target(exec), self.rocm_prefill_fusion_policy())
                    .context(
                        "compiling the ROCm stateless re-prefill graph for the executor contract",
                    )?;
            let entry = exec
                .add_entry(exe, &staged)
                .context("add_entry (generate_rocm_reprefill)")?;
            let bound = self.bind(&g, &tokens)?;
            let inputs = slot_step_inputs(&g, &bound)?;
            let step_result = exec
                .step(exe, entry, &inputs, &mut poot_executor::NoSync)
                .context("rocm reprefill step")
                .and_then(|mut out| out.read().context("rocm reprefill readback"));
            exec.remove_entry(exe, entry)
                .context("remove_entry (generate_rocm_reprefill)")?;
            let bytes = step_result?;
            let logits: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let next = sampler.pick(&logits)? as u32;
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
}

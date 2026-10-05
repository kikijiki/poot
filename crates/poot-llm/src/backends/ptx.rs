//! PTX/NVIDIA backend: the executor-contract consumer (Card 549). Decode reuses the shared,
//! backend-neutral [`Self::generate_kv_gpu_cached`]/[`Self::generate_kv_gpu_cached_sampled`]
//! (`backends/gpu_generate.rs`, landed with Card 546a) directly over an `Engine<PtxDevice>` - there is
//! no PTX-specific decode loop left to own, since those functions only ever touch `&mut dyn
//! poot_executor::Executor`. What is genuinely PTX-shaped here: the batched (Q=N) prefill + decode-from-
//! prefill loop (Card 546b's wgpu twin, `generate_kv_gpu_prefilled_tokens`, has not landed, so this is
//! written independently against the same contract) and the one-shot eager-equivalent `run_resident`
//! free function [`Self::generate_ptx_reprefill`] still needs.

use std::collections::HashMap;

use poot_eval::Value;
use poot_executor::{EntryId, ExecutableId, Executor, NoSync};
use poot_graph_ir::{Graph, ValueId};
use poot_graph_plan::{
    CompileOptions, FusionPolicy, StagedCompileError, StagedProgram, Submission, Target,
};
use poot_tensor::HostTensor;

use crate::GenerationControl;
use crate::backends::gpu_generate::{executor_target, slot_step_inputs};
use crate::core::generate::argmax;
use crate::core::runner::Runner;
use crate::error::{Result, ResultExt};

#[cfg(test)]
use crate::error::OptionExt;
#[cfg(test)]
use {
    poot_graph_ir::Storage,
    poot_quant::weights::{DenseWeight, WeightEntry, WeightStore},
    std::sync::Arc,
};

/// `g` staged for the executor contract against `target` (Card 546a: the contract admits
/// `Submission::Replay` only - it records once and replays once for a one-shot entry just as much as
/// for a many-step decode entry). Ordinary models load a zero-lane validation packet (dmodel.md R2); no
/// production caller wires real decode-graph witnesses yet.
pub(crate) fn staged_program(
    g: &Graph,
    target: Target,
    fusion: FusionPolicy,
) -> std::result::Result<StagedProgram<poot_graph_ir::ValidationOutputs>, StagedCompileError> {
    let g = g.clone().with_validations(Vec::new());
    poot_graph_plan::compile_staged(
        &g,
        &poot_graph_plan::TargetSet::single(poot_graph_plan::DeviceId(0), target),
        &poot_graph_plan::Partition {
            experts: poot_graph_plan::ExpertPlacement::AllResident,
            devices: poot_graph_plan::DevicePlacement::Single(poot_graph_plan::DeviceId(0)),
        },
        &CompileOptions {
            execution: Submission::Replay,
            fusion,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
}

/// Run `exe`'s `entry` once on `g`'s slot inputs (bound by `bind`) and read back the primary output as
/// an owned [`Tensor`], removing the entry afterward regardless of the step's outcome (a one-shot entry's
/// recording/locals are not left resident).
pub(crate) fn step_once_and_remove(
    exec: &mut dyn Executor,
    exe: ExecutableId,
    entry: EntryId,
    g: &Graph,
    bound: &HashMap<ValueId, Value>,
) -> Result<HostTensor> {
    let result: Result<HostTensor> = (|| {
        let inputs = slot_step_inputs(g, bound)?;
        let bytes = exec
            .step(exe, entry, &inputs, &mut NoSync)
            .context("ptx step")?
            .read()
            .context("ptx step readback")?;
        Ok(HostTensor::f32(
            g.aval(g.output).shape.clone(),
            bytemuck::cast_slice(&bytes).to_vec(),
        ))
    })();
    exec.remove_entry(exe, entry).context("ptx remove_entry")?;
    result
}

/// A [`WeightStore`] holding every `Storage::Const` value `g` declares, read from `inputs` by name
/// (Card 546a's name-equality binding, Z8): the same shape [`Runner::load_on`] bakes from `self.weights`,
/// but over an arbitrary caller-supplied value map instead of the Runner's own checkpoint. Dense-only:
/// `run_resident`'s remaining callers (the ALiBi/unverified-arch CPU-parity fallback, ad hoc test
/// graphs) never hand it a packed-quant const.
#[cfg(test)]
fn const_store_from_inputs(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Result<WeightStore> {
    let mut builder = WeightStore::builder();
    for &id in &g.inputs {
        let meta = g.meta(id);
        if meta.storage != Storage::Const {
            continue;
        }
        let name = meta
            .name
            .clone()
            .context("run_resident: an unnamed Const cannot bind by name")?;
        let value = inputs
            .get(&id)
            .with_context(|| format!("run_resident: no value bound for const {name}"))?;
        let dense = value
            .as_host()
            .with_context(|| format!("run_resident: const {name} is not a dense tensor"))?;
        let entry = DenseWeight::try_new(
            dense.dtype(),
            dense.shape().to_vec(),
            Arc::from(dense.view().bytes()),
        )
        .with_context(|| format!("{name}: building run_resident's weight store"))?;
        builder
            .insert(name.clone(), WeightEntry::Dense(entry))
            .with_context(|| format!("{name}: duplicate const in run_resident's weight store"))?;
    }
    Ok(builder.build())
}

/// Device-resident, stateless run of `g` on the executor contract: a fresh one-shot executable loaded
/// from `inputs`'s own consts, stepped once, unloaded. The PTX twin of
/// `poot_gpu::GpuExecutor::run_resident` before Card 546b's migration; `g` must declare no carried
/// state (every current caller, [`Runner::generate_ptx_reprefill`]'s per-step re-prefill and ad hoc
/// graphs, is already stateless by construction). Slower than a cached decode entry by design (every
/// call re-uploads every const): this is the CPU-oracle-parity fallback path, not the production decode
/// loop.
#[cfg(test)]
pub(crate) fn run_resident(
    exec: &mut dyn Executor,
    g: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Result<HostTensor> {
    let store = const_store_from_inputs(g, inputs)?;
    let exe = exec
        .load_weights(Arc::new(store), poot_executor::WeightSource::ConstNames)
        .context("run_resident: load_weights")?;
    let result = (|| {
        let staged = staged_program(g, executor_target(exec), FusionPolicy::Full)
            .context("run_resident: stage")?;
        let entry = exec
            .add_entry(exe, &staged)
            .context("run_resident: add_entry")?;
        step_once_and_remove(exec, exe, entry, g, inputs)
    })();
    exec.unload(exe).context("run_resident: unload")?;
    result
}

impl Runner {
    /// PTX (NVIDIA) greedy generation using a batched prefill (spec 023 sub-step 3) to fill the prompt's
    /// KV cache in one multi-token (Q=N) forward through the executor contract, then a captured
    /// single-token decode entry continues from `pos = n`. `exec`/`exe` are shared across the whole
    /// call: the prefill entry and the decode entry bind the same carried KV state by (name, aval,
    /// storage), so the prompt's K/V written by prefill carries into decode with no buffer
    /// handoff - the contract's own state sharing replaces the pre-549 buffer-identity handoff
    /// (`capture_decode_from`). Each entry is removed after its own use: the prefill
    /// entry after its one step, the decode entry once generation stops.
    pub fn generate_kv_ptx_prefilled_tokens(
        &self,
        context: &[u32],
        max_new: usize,
        exec: &mut dyn Executor,
        exe: ExecutableId,
        ignore_eos: bool,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = context.to_vec();
        let n = tokens.len();
        let cap = n + max_new;

        // 1) one batched (Q=N) prefill forward fills the cache for positions [0,N) and yields the
        //    last-position logits (the first next-token distribution): one Replay entry, stepped once,
        //    removed (Card 546b's wgpu twin does the same once it lands).
        let pg = self.prefill_kv_graph(n, cap)?;
        // Every contiguous prefill family left on the Runner is MoE: cards 186/192's MoE-router-hang
        // guard applies.
        let staged = staged_program(&pg, executor_target(exec), FusionPolicy::MoeHangGuard)
            .context("ptx stage (prefill)")?;
        let prefill_entry = exec
            .add_entry(exe, &staged)
            .context("ptx add_entry (prefill)")?;
        let bound = self.bind_prefill_kv(&pg, &tokens)?;
        let logits = step_once_and_remove(exec, exe, prefill_entry, &pg, &bound)
            .context("ptx batched prefill")?;

        let mut generated = 0;
        let next = argmax(logits.as_f32().unwrap())? as u32;
        if (next == self.eos && !ignore_eos) || max_new == 0 {
            return Ok(tokens);
        }
        tokens.push(next);
        if on_token(&self.decode(&[next])?).is_break() {
            return Ok(tokens);
        }
        generated += 1;

        // 2) a single-token decode entry continues from pos = n, over the same `exe`: its own carried
        //    state is the SAME (name, aval, storage) the prefill entry just wrote, so no
        //    buffer handoff is needed. Through `Runner::pick_step` (card 551b): greedy, through the
        //    same device-suffix contract path every decode loop uses.
        if generated < max_new {
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
            entries
                .remove_all(exec, exe)
                .context("ptx remove_entry (decode after prefill)")?;
            result?;
        }
        Ok(tokens)
    }

    /// Like [`Self::generate_kv_ptx_prefilled_tokens`] but from a prompt string.
    pub fn generate_kv_ptx_prefilled(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn Executor,
        exe: ExecutableId,
        on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let context = self.encode(prompt)?;
        self.generate_kv_ptx_prefilled_tokens(&context, max_new, exec, exe, false, on_token)
    }

    /// PTX counterpart of [`Self::generate_gpu_reprefill`]: re-prefills each step through
    /// [`run_resident`] (stateless single-shot, now over the executor contract). A correctness-parity
    /// path (e.g. ALiBi), slower than cached PTX decode by design.
    #[cfg(test)]
    pub(crate) fn generate_ptx_reprefill(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn Executor,
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        // DSA is CPU-oracle only, as in `generate_gpu_reprefill`.
        if self.deepseek32.is_some() {
            bail!(
                "generate_ptx_reprefill: DeepSeek-V3.2 DSA has no GPU dispatch yet (CPU-oracle only, \
                 spec 277) - use Runner::generate/generate_sampled/generate_kv_masked (CPU) for this \
                 checkpoint"
            );
        }
        // Nemotron-H runs on wgpu (verified against the CPU oracle on RADV,
        // `crates/poot-gpu/tests/nemotron_h.rs`) but has no NVIDIA verification
        //, so it bails here.
        if self.nemotron_h.is_some() {
            bail!(
                "generate_ptx_reprefill: Nemotron-H has no PTX receipt yet (spec 279 / card 297: the \
                 wgpu path is verified, NVIDIA is not) - use Runner::generate_gpu_reprefill (wgpu) or \
                 Runner::generate/generate_sampled (CPU) for this checkpoint"
            );
        }
        let mut tokens = self.encode(prompt)?;
        let mut sampler = crate::core::sampler::Sampler::greedy();
        for _ in 0..max_new {
            let g = self.stateless_prefill_graph(tokens.len())?;
            let inputs = self.bind(&g, &tokens)?;
            let logits = run_resident(exec, &g, &inputs)?;
            let next = sampler.pick(logits.as_f32().unwrap())? as u32;
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

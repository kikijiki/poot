//! Card 551b: [`Runner::pick_step`], the one step-and-pick every Runner decode
//! loop calls, and [`DecodeEntries`], the small per-call cache that compiles a decode entry for the
//! head (`Logits`/`Sample(rule)`, [`crate::driver::suffix::Head`]) a loop's sampler
//! needs, lazily, and removes it with the sequence (R-546-3). Staging/binding reuse
//! `backends::gpu_generate`'s `staged_program`/`executor_target`/`slot_step_inputs` (the same owner
//! `core::speculative` already calls across this `core`/`backends` boundary) rather than a fourth
//! copy.

use std::collections::HashMap;

use poot_graph_ir::{Graph, ValueId};

use crate::backends::gpu_generate::{executor_target, slot_step_inputs, staged_program};
use crate::core::runner::Runner;
use crate::core::sampler::Sampler;
use crate::driver::suffix::{self, Head, SuffixRows};
use crate::error::{Result, ResultExt};

/// One compiled decode entry: its `EntryId`, the exact (possibly suffix-appended) graph it was
/// staged from (`slot_step_inputs` needs this one, not the base graph, to see the suffix's own
/// slots), and the suffix's own input ids (`None` for `Logits` and for a tag `Sample(Greedy)` never
/// declares).
struct CompiledHead {
    graph: Graph,
    entry: poot_executor::EntryId,
    seed: Option<ValueId>,
    params: Option<ValueId>,
    top_k: Option<ValueId>,
}

/// The small per-generate-call cache [`Runner::pick_step`] shares across one decode loop's steps
/// (card 551b: "entries are compiled lazily per `(Head, cap)` and removed with the sequence",
/// R-546-3). `cap` is implicit: one `DecodeEntries` is built for one `base` graph (already traced at
/// its call's `cap`), so the cache key is just the head. In practice one loop ever asks for exactly
/// one head (a loop's `Sampler` config is fixed for its whole call), but the cache is correct for
/// more - the same discipline [`DecodeEntries::remove_all`] cleans up regardless.
pub(crate) struct DecodeEntries {
    base: Graph,
    cache: HashMap<Head, CompiledHead>,
}

impl DecodeEntries {
    /// `base` is the Runner's already-traced-and-bound-storage dense decode graph (e.g.
    /// `Runner::decode_masked_graph`'s result), never itself carrying a sampling suffix.
    pub(crate) fn new(base: Graph) -> Self {
        Self {
            base,
            cache: HashMap::new(),
        }
    }

    fn ensure(
        &mut self,
        head: Head,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
    ) -> Result<()> {
        if self.cache.contains_key(&head) {
            return Ok(());
        }
        let appended = suffix::with_head(self.base.clone(), head);
        let target = executor_target(exec);
        let staged = staged_program(&appended.graph, target)
            .context("compiling a decode_step entry (card 551b sampling suffix)")?;
        let entry = exec
            .add_entry(exe, &staged)
            .context("add_entry (decode_step entry, card 551b)")?;
        self.cache.insert(
            head,
            CompiledHead {
                graph: appended.graph,
                entry,
                seed: appended.seed,
                params: appended.params,
                top_k: appended.top_k,
            },
        );
        Ok(())
    }

    /// Remove every entry this cache compiled (R-546-3): the decode loop calls this once, on every
    /// exit path (including an early `on_token` stop), the same discipline the pre-551b loops
    /// applied to their own single add_entry/remove_entry pair.
    pub(crate) fn remove_all(
        self,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
    ) -> Result<()> {
        for compiled in self.cache.into_values() {
            exec.remove_entry(exe, compiled.entry)
                .context("remove_entry (decode_step entry, card 551b)")?;
        }
        Ok(())
    }
}

impl Runner {
    /// The one step-and-pick every Runner decode loop calls (card 551b):
    /// compiles (lazily, via `entries`) and steps the `Sample(rule)` entry when
    /// [`suffix::rule_of`] admits `sampler`'s request, or the raw `Logits` entry plus the
    /// host pick ([`Sampler::pick`]) otherwise - the twelve decode-loop call sites' own step-and-pick
    /// code, in one place. `token`/`pos` bind the usual per-step slots exactly as before
    /// (`Runner::bind_decode`); a `Sample(rule)` entry's extra sampler-slot inputs are filled from
    /// [`SuffixRows::push`] and merged in before the step.
    ///
    /// `commit` is `false` only for a throwaway step whose output nothing will use
    /// (`generate_kv_gpu_cached_sampled`'s cached re-encode loop single-steps every prompt position
    /// through the same entry before it reaches the one it must actually pick at): the step still
    /// runs (advancing the entry's carried KV state) and still binds every input the entry declares
    /// ([`SuffixRows::placeholder`] in place of `push`, so a discarded step never advances
    /// `sampler`'s rng stream), but no pick runs and the result is `None` - skipping `sampler.pick`
    /// here, not just discarding its result, matters whenever logprob recording is on (it would
    /// otherwise record a bogus entry for a token nobody asked for).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pick_step(
        &self,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        entries: &mut DecodeEntries,
        sampler: &mut Sampler,
        token: u32,
        pos: usize,
        commit: bool,
    ) -> Result<Option<u32>> {
        let rule = suffix::rule_of(sampler);
        let head = match rule {
            Some(r) => Head::Sample(r),
            None => Head::Logits,
        };
        entries.ensure(head, exec, exe)?;
        let compiled = entries
            .cache
            .get(&head)
            .expect("DecodeEntries::ensure just inserted this head");

        let mut bound = self.bind_decode(&entries.base, token, pos, &[], None)?;
        if let Some(rule) = rule
            && rule != poot_graph_ir::op::SampleRule::Greedy
        {
            let row = if commit {
                SuffixRows::push(sampler, rule)
            } else {
                SuffixRows::placeholder(sampler, rule)
            };
            // The suffix's `seed`/`params`/`top_k` inputs carry the decode graph's own leading
            // shape (e.g. `[1,1]` for the Runner's single-sequence single-token decode, card 550),
            // not a bare scalar - read each declared aval rather than assuming one.
            let seed_id = compiled
                .seed
                .expect("a non-Greedy Sample(rule) entry declares a seed slot");
            let seed_shape = compiled.graph.meta(seed_id).aval.shape.clone();
            bound.insert(
                seed_id,
                poot_tensor::HostTensor::i32(seed_shape, vec![row.seed]).into(),
            );
            let params_id = compiled
                .params
                .expect("a non-Greedy Sample(rule) entry declares a params slot");
            let params_shape = compiled.graph.meta(params_id).aval.shape.clone();
            bound.insert(
                params_id,
                poot_tensor::HostTensor::f32(params_shape, row.params).into(),
            );
            if let Some(k) = row.top_k {
                let top_k_id = compiled
                    .top_k
                    .expect("GumbelTopK/GumbelTopKTopP declares a top_k slot");
                let top_k_shape = compiled.graph.meta(top_k_id).aval.shape.clone();
                bound.insert(
                    top_k_id,
                    poot_tensor::HostTensor::i32(top_k_shape, vec![k]).into(),
                );
            }
        }

        let inputs = slot_step_inputs(&compiled.graph, &bound)?;
        let bytes = exec
            .step(exe, compiled.entry, &inputs, &mut poot_executor::NoSync)
            .context("decode step (card 551b sampling suffix)")?
            .read()
            .context("decode step readback (card 551b sampling suffix)")?;

        if !commit {
            return Ok(None);
        }
        match head {
            Head::Logits => {
                let logits: &[f32] = bytemuck::cast_slice(&bytes);
                sampler
                    .pick(logits)
                    .map(|t| Some(t as u32))
                    .map_err(Into::into)
            }
            Head::Sample(_) => suffix::read_tokens(&bytes).map(Some).map_err(Into::into),
        }
    }
}

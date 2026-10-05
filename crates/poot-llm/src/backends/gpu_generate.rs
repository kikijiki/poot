//! wgpu GPU generation: resident/cached/paged/chunked KV loops and KV-quant helpers.

use crate::GenerationControl;
use crate::core::generate::argmax;
use crate::core::runner::Runner;
use crate::core::sampler::Sampler;
use crate::error::{OptionExt, Result, ResultExt};

/// The one [`poot_graph_plan::Target`] `exec` drives (Card 546a's M3 contract: `Engine<D>` drives
/// exactly one device, so `target_set()` always names exactly one). `pub(crate)`: every backend's
/// consumer needs this same generic read, not just wgpu's (Card 549).
pub(crate) fn executor_target(exec: &dyn poot_executor::Executor) -> poot_graph_plan::Target {
    exec.target_set()
        .devices()
        .first()
        .expect("Card 546a's M3 Engine<D> always names exactly one device")
        .1
}

/// `g`, staged for the executor contract (always `Submission::Replay`: the contract admits no other
/// mode). Ordinary models load a zero-lane validation packet; no
/// production caller wires real decode-graph witnesses yet. Shared by every production entry program
/// (prefill and decode alike): the contract's `add_entry` shares carried state across entries by
/// (name, aval, storage), so a prefill entry's written K/V is the decode entry's input.
pub(crate) fn staged_program(
    g: &poot_graph_ir::Graph,
    target: poot_graph_plan::Target,
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
            fusion: poot_graph_plan::FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        },
    )
}

/// `bound`'s `Storage::Slot` entries only (Card 546a's contract binds every `Storage::Const`/
/// `Storage::State` value from the executable's `WeightStore`/carried state instead, through
/// `Runner::load_on` and the engine's own state buffers - never a per-step value), as the
/// `SlotKey`-keyed `StepInputs` the contract's `step` takes. Borrows `bound`'s tensors,
/// so the caller's `bind_decode` result must outlive the returned `StepInputs`. `pub(crate)`: every
/// backend's consumer needs this same generic bind, not just wgpu's (Card 549).
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
            .context("decode slot without a structured SlotKey")?
            .clone();
        let value = bound
            .get(&id)
            .context("bind_decode did not bind a declared slot")?;
        let tensor = value
            .as_host()
            .context("a slot value must be a dense tensor")?;
        inputs.push(key, tensor.shape(), tensor.view());
    }
    Ok(inputs)
}

impl Runner {
    /// Greedily generate via the cached re-encode path on the GPU: the constant-shape masked decode graph
    /// (`decode_masked_graph`, built once) runs through the executor contract's recorded-replay entry
    /// (`exec.add_entry` once, then `exec.step` per token), which encodes every per-token dispatch into one
    /// submit with cached pipelines. Same carried-KV semantics and greedy text as `generate_kv_gpu`, fewer
    /// host submits. Returns the full token id sequence.
    pub fn generate_kv_gpu_cached(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        self.generate_kv_gpu_cached_sampled(
            prompt,
            max_new,
            exec,
            exe,
            &mut Sampler::greedy(),
            &[],
            on_token,
        )
    }

    /// Like [`Self::generate_kv_gpu_cached`] but with a [`Sampler`] for the next-token choice (greedy when the
    /// sampler's temperature is 0) and `stops` for early termination (EOS / `max_new` / any non-empty stop
    /// string in the decoded generated text; empty `stops` disables the check).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_kv_gpu_cached_sampled(
        &self,
        prompt: &str,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        sampler: &mut Sampler,
        stops: &[String],
        mut on_token: impl FnMut(&str) -> GenerationControl,
    ) -> Result<Vec<u32>> {
        let mut tokens = self.encode(prompt)?;
        let gen_start = tokens.len();
        let cap = tokens.len() + max_new;
        // Each family runs its own fixed-KV decode tracer (selected by `decode_masked_graph`).
        // `DecodeEntries` stages it (lazily, per `Head`)
        // for the executor contract (Card 546a); `add_entry` binds every const from the executable's
        // `WeightStore` (`Runner::load_on`) and every state pair by (name, aval, storage) - never a
        // per-call host-side cache buffer.
        let g = self.decode_masked_graph(cap)?;
        let mut entries = crate::core::decode_step::DecodeEntries::new(g);
        // this call's own entries, never shared with another `generate` call on
        // the same `exec`/`exe` (a card 563 prepared catalog later keeps one resident per (graph
        // fingerprint, cap) across calls) - removed on every exit path below so repeated calls (one
        // per bench request, one per `card297_cached_families` prompt, ...) do not grow
        // `stats().recordings` or device memory without bound.
        let result: Result<()> = (|| {
            // Card 551b: every backend samples through the same `Runner::pick_step` contract path,
            // greedy and sampled alike - the device suffix carries Card 601's typed
            // non-finite fault on both.
            // penalties (rep/presence/frequency) see the prompt context plus each generated token.
            sampler.seed_context(&tokens);
            let mut generated = 0;
            let mut pos = 0;
            loop {
                let at_gen = pos + 1 == tokens.len();
                let next =
                    self.pick_step(exec, exe, &mut entries, sampler, tokens[pos], pos, at_gen)?;
                if let Some(next) = next {
                    if next == self.eos {
                        break;
                    }
                    tokens.push(next);
                    sampler.observe(next);
                    if on_token(&self.stream_piece(&tokens, gen_start)?).is_break() {
                        break;
                    }
                    generated += 1;
                    if generated >= max_new || self.hit_stop(&tokens[gen_start..], stops)? {
                        break;
                    }
                }
                pos += 1;
            }
            Ok(())
        })();
        match entries.remove_all(exec, exe) {
            Ok(()) => result.map(|()| tokens),
            Err(e) if result.is_ok() => {
                Err(e).context("remove_entry (cached-sampled decode graph)")
            }
            Err(_) => result.map(|()| tokens),
        }
    }
}

impl Runner {
    /// Greedily generate on the GPU using a batched prefill to fill the prompt's KV cache in one multi-token
    /// forward, then the single-token cached decode from pos=N. Same carried-KV semantics and bit-identical
    /// greedy text as [`Self::generate_kv_gpu_cached`], without N single-token forwards for the prompt. The
    /// prefill graph (`prefill_kv_graph`) writes K/V for positions [0,N) into the same
    /// `[1,n_kv_heads,cap,head_dim]` cache buffers the decode loop carries.
    pub fn generate_kv_gpu_prefilled(
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
        let next = self.run_prefill_entry(&tokens, cap, exec, exe)?;
        if next == self.eos || max_new == 0 {
            return Ok(tokens);
        }
        tokens.push(next);
        let decode = (max_new > 1)
            .then(|| self.prepare_post_prefill_decode(cap))
            .transpose()?;
        self.deliver_prefill_token_and_decode(
            decode,
            n,
            max_new,
            exec,
            exe,
            false,
            &mut tokens,
            &mut on_token,
        )?;
        Ok(tokens)
    }

    /// One batched prefill forward: fills the cache for positions `[0, tokens.len())` and returns the
    /// argmax of the last-position logits (the first next-token distribution). `prefill_kv_graph` traces with full fusion so flash-attention fusion
    /// fires (otherwise the materialized `[1,Hq,N,N]` score matrix stays in the graph, the O(N^2) path
    /// that hits the wgpu grid cap). The contract admits only `Submission::Replay`, so
    /// this still pays one full recording for a forward that never replays again; the entry is removed
    /// right after its one step (R-546-3).
    fn run_prefill_entry(
        &self,
        tokens: &[u32],
        cap: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
    ) -> Result<u32> {
        let pg = self.prefill_kv_graph(tokens.len(), cap)?;
        let staged = staged_program(&pg, executor_target(exec))
            .context("compiling the batched prefill graph for the executor contract")?;
        let entry = exec
            .add_entry(exe, &staged)
            .context("add_entry (batched prefill)")?;
        let result: Result<usize> = (|| {
            let bound = self.bind_prefill_kv(&pg, tokens)?;
            let inputs = slot_step_inputs(&pg, &bound)?;
            let bytes = exec
                .step(exe, entry, &inputs, &mut poot_executor::NoSync)
                .context("gpu batched prefill (executor contract)")?
                .read()
                .context("gpu batched prefill readback")?;
            Ok(argmax(bytemuck::cast_slice(&bytes))?)
        })();
        match exec.remove_entry(exe, entry) {
            Ok(()) => result.map(|t| t as u32),
            Err(e) if result.is_ok() => Err(e).context("remove_entry (batched prefill)"),
            Err(_) => result.map(|t| t as u32),
        }
    }

    /// The post-prefill decode graph: its first step shares the carried K/V state with the
    /// just-removed prefill entry by (name, aval, storage), so the decode loop continues
    /// from the state the prefill step wrote. `DecodeEntries` compiles and adds the actual entry
    /// lazily, on the first step [`Self::deliver_prefill_token_and_decode`] takes.
    fn prepare_post_prefill_decode(&self, cap: usize) -> Result<poot_graph_ir::Graph> {
        self.decode_masked_graph(cap)
    }

    /// Delivers the prefill-derived first token (already the last element of `tokens`) to `on_token`,
    /// then - unless the callback stops or there is nothing left to decode - continues the constant-
    /// shape single-token cached decode from `pos = tokens.len() - 1`, carrying the state the prefill
    /// entry wrote (shared by (name, aval, storage), never a per-call host cache buffer).
    /// Greedy, through the same `Runner::pick_step` contract path every decode loop uses (card 551b):
    /// the device suffix's typed non-finite fault applies here too, not just to sampled requests.
    /// `decode`'s entries are always removed before this returns, on every exit path including an
    /// early `on_token` stop (R-546-3's cleanup discipline extended to the decode entry it pairs
    /// with). `ignore_eos` lets the decode-curve benchmark run a fixed token count past a real EOS.
    #[allow(clippy::too_many_arguments)]
    fn deliver_prefill_token_and_decode(
        &self,
        decode: Option<poot_graph_ir::Graph>,
        n: usize,
        max_new: usize,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        ignore_eos: bool,
        tokens: &mut Vec<u32>,
        on_token: &mut impl FnMut(&str) -> GenerationControl,
    ) -> Result<()> {
        let first = *tokens.last().expect("the prefill token is already pushed");
        let Some(dg) = decode else {
            let _ = on_token(&self.decode(&[first])?);
            return Ok(());
        };
        let mut entries = crate::core::decode_step::DecodeEntries::new(dg);
        let mut sampler = Sampler::greedy();
        let result: Result<()> = (|| {
            if on_token(&self.decode(&[first])?).is_break() {
                return Ok(());
            }
            let mut generated = 1;
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
            Err(e) if result.is_ok() => Err(e).context("remove_entry (post-prefill decode)"),
            Err(_) => result,
        }
    }

    /// wgpu batched prefill + cached decode from pre-encoded tokens; the wgpu/Vulkan counterpart of
    /// [`Self::generate_kv_ptx_prefilled_tokens`], used by the decode-curve benchmark.
    pub fn generate_kv_gpu_prefilled_tokens(
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
        let next = self.run_prefill_entry(&tokens, cap, exec, exe)?;
        if (next == self.eos && !ignore_eos) || max_new == 0 {
            return Ok(tokens);
        }
        tokens.push(next);
        let decode = (max_new > 1)
            .then(|| self.prepare_post_prefill_decode(cap))
            .transpose()?;
        self.deliver_prefill_token_and_decode(
            decode,
            n,
            max_new,
            exec,
            exe,
            ignore_eos,
            &mut tokens,
            &mut on_token,
        )?;
        Ok(tokens)
    }
}

/// The four synthetic architecture cases share one acceptance driver. The table is the only family
/// inventory here: adding a family is one row, not another copied decode loop. The device test selects one
/// row per invocation so the required-wgpu lane stays bounded and attributable.
#[cfg(test)]
mod card297_cached_families {
    use super::{Runner, argmax, executor_target, slot_step_inputs, staged_program};
    use crate::core::decode_arch::DecodeArch;
    use poot_graph_ir::Graph;
    use poot_tensor::HostTensor;
    use poot_test_util::max_abs_error;

    const GENERATED_STEPS: usize = 3;
    const REL_TOL: f32 = 1.0e-4;

    use poot_graph_plan::passes_without_target as optimize;

    struct CachedFamilyCase {
        family: &'static str,
        fixture_dir: &'static str,
        write_fixture: fn() -> poot_test_util::UniqueTempPath,
        arch: DecodeArch,
        prompts: &'static [&'static str],
        required_state_names: &'static [&'static str],
    }

    const CASES: &[CachedFamilyCase] = &[
        CachedFamilyCase {
            family: "nemotron-h",
            fixture_dir: "poot_nemotron_h_loader_fixture",
            write_fixture:
                crate::architectures::nemotron_h_load::tests::write_tiny_nemotron_h_checkpoint,
            arch: DecodeArch::NemotronH,
            prompts: &["t1 t3 t5 t7", "t2 t4 t6 t8"],
            required_state_names: &["conv_cache", "ssm_state", "k_cache", "v_cache"],
        },
        CachedFamilyCase {
            family: "dsa",
            fixture_dir: "poot_deepseek32_loader_fixture",
            write_fixture:
                crate::architectures::deepseek32_load::tests::write_tiny_deepseek32_checkpoint,
            arch: DecodeArch::DeepseekV32,
            prompts: &["t1 t3 t5 t7", "t2 t4 t6 t8"],
            required_state_names: &["mla.c_cache", "mla.rope_cache", "self_attn.indexer.k_cache"],
        },
    ];

    struct SequenceResult {
        generated: Vec<u32>,
        final_state: Vec<HostTensor>,
    }

    fn zero_state(g: &Graph) -> Vec<HostTensor> {
        g.state
            .iter()
            .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
            .collect()
    }

    fn assert_close(family: &str, what: &str, actual: &HostTensor, expected: &HostTensor) -> f32 {
        assert_eq!(actual.shape(), expected.shape(), "{family} {what} shape");
        let mut max_relative_error = 0.0f32;
        for (index, (&got, &want)) in actual
            .as_f32()
            .unwrap()
            .iter()
            .zip(expected.as_f32().unwrap().iter())
            .enumerate()
        {
            assert!(
                got.is_finite() && want.is_finite(),
                "{family} {what}[{index}] is not finite: gpu={got} cpu={want}"
            );
            let error = (got - want).abs();
            let tolerance = REL_TOL * want.abs().max(1.0);
            max_relative_error = max_relative_error.max(error / want.abs().max(1.0));
            assert!(
                error <= tolerance,
                "{family} {what}[{index}] differs: gpu={got} cpu={want} error={error} tolerance={tolerance}"
            );
        }
        max_relative_error
    }

    fn state_label(g: &Graph, index: usize) -> String {
        let state_in = g.state[index].0;
        g.meta(state_in)
            .name
            .clone()
            .unwrap_or_else(|| format!("v{state_in}"))
    }

    fn assert_reused_state(
        case: &CachedFamilyCase,
        runner: &Runner,
        g: &Graph,
        token: u32,
        pos: usize,
        prior: &[HostTensor],
        expected: &[HostTensor],
    ) {
        for state_index in 0..prior.len() {
            let mut omitted = prior.to_vec();
            omitted[state_index] = HostTensor::zeros(omitted[state_index].shape().to_vec());
            let inputs = runner
                .bind_decode(g, token, pos, &omitted, None)
                .unwrap_or_else(|error| panic!("{} bind omitted state: {error}", case.family));
            let (_, omitted_state) = crate::core::cpu_oracle::cpu_eval_with_state(g, &inputs)
                .unwrap_or_else(|error| panic!("{} eval omitted state: {error}", case.family));
            let difference = max_abs_error(
                omitted_state[state_index].as_f32().unwrap(),
                expected[state_index].as_f32().unwrap(),
            );
            assert!(
                difference > 1.0e-7,
                "{} state {} was not reused at position {pos}",
                case.family,
                state_label(g, state_index)
            );
        }
    }

    fn cpu_sequence(
        case: &CachedFamilyCase,
        runner: &Runner,
        g: &Graph,
        prompt: &str,
    ) -> SequenceResult {
        let mut tokens = runner.encode(prompt).expect("encode synthetic prompt");
        assert!(
            tokens.len() > 1,
            "{} prompt must be multi-token",
            case.family
        );
        assert_eq!(
            g.aval(g.output).shape.last(),
            Some(&runner.cfg.vocab),
            "{} wrong-family graph output",
            case.family
        );
        assert!(
            !g.state.is_empty(),
            "{} graph must carry state",
            case.family
        );

        let mut state = zero_state(g);
        let mut state_changed = vec![false; state.len()];
        let mut generated = Vec::with_capacity(GENERATED_STEPS);
        let mut pos = 0;
        let mut reuse_probe = None;
        while generated.len() < GENERATED_STEPS {
            let prior = state.clone();
            let token = tokens[pos];
            let inputs = runner
                .bind_decode(g, token, pos, &prior, None)
                .unwrap_or_else(|error| panic!("{} bind position {pos}: {error}", case.family));
            let (logits, next_state) = crate::core::cpu_oracle::cpu_eval_with_state(g, &inputs)
                .unwrap_or_else(|error| panic!("{} eval position {pos}: {error}", case.family));
            for (state_index, (before, after)) in prior.iter().zip(&next_state).enumerate() {
                state_changed[state_index] |=
                    max_abs_error(before.as_f32().unwrap(), after.as_f32().unwrap()) > 1.0e-7;
            }
            reuse_probe = Some((token, pos, prior, next_state.clone()));
            state = next_state;

            if pos + 1 == tokens.len() {
                let next = argmax(logits.as_f32().unwrap()).unwrap() as u32;
                tokens.push(next);
                generated.push(next);
            }
            pos += 1;
        }

        for (state_index, changed) in state_changed.into_iter().enumerate() {
            assert!(
                changed,
                "{} state {} never changed",
                case.family,
                state_label(g, state_index)
            );
        }
        let (token, pos, prior, expected) =
            reuse_probe.expect("sequence executes at least one step");
        assert_reused_state(case, runner, g, token, pos, &prior, &expected);

        SequenceResult {
            generated,
            final_state: state,
        }
    }

    fn assert_prompts_do_not_share_stale_state(
        case: &CachedFamilyCase,
        left: &SequenceResult,
        right: &SequenceResult,
    ) {
        let state_differs = left
            .final_state
            .iter()
            .zip(&right.final_state)
            .any(|(a, b)| max_abs_error(a.as_f32().unwrap(), b.as_f32().unwrap()) > 1.0e-7);
        assert!(
            state_differs,
            "{} changed prompt produced identical final state; stale-cache guard is ineffective",
            case.family
        );
    }

    fn load_case(case: &CachedFamilyCase) -> (Runner, Graph) {
        let dir = (case.write_fixture)();
        assert_eq!(
            dir.file_name().and_then(|name| name.to_str()),
            Some(case.fixture_dir),
            "{} fixture path drifted from the shared inventory",
            case.family
        );
        let runner = Runner::load(&dir)
            .unwrap_or_else(|error| panic!("{} load {}: {error}", case.family, dir.display()));
        assert_eq!(
            runner.decode_arch().unwrap(),
            case.arch,
            "{} dispatch",
            case.family
        );
        assert_eq!(
            case.prompts.len(),
            2,
            "{} stale-cache prompt count",
            case.family
        );
        let prompt_len = runner.encode(case.prompts[0]).expect("encode prompt").len();
        assert_eq!(
            prompt_len,
            runner.encode(case.prompts[1]).expect("encode prompt").len(),
            "{} prompts must share one fixed-capacity graph",
            case.family
        );
        let g = runner
            .decode_masked_graph(prompt_len + GENERATED_STEPS)
            .unwrap();
        let g = optimize(&g);
        let state_names: Vec<_> = g
            .state
            .iter()
            .map(|&(state_in, _)| g.meta(state_in).name.as_deref().unwrap_or(""))
            .collect();
        for required in case.required_state_names {
            assert!(
                state_names.iter().any(|name| name.ends_with(required)),
                "{} dispatch omitted required state {required}; got {state_names:?}",
                case.family
            );
        }
        (runner, g)
    }

    #[test]
    fn card297_cached_family_cpu_contracts() {
        for case in CASES {
            eprintln!("card297 CPU contract: {}", case.family);
            let (runner, g) = load_case(case);
            let results: Vec<_> = case
                .prompts
                .iter()
                .map(|prompt| cpu_sequence(case, &runner, &g, prompt))
                .collect();
            eprintln!(
                "card297 CPU tokens: {} {:?} {:?}",
                case.family, results[0].generated, results[1].generated
            );
            assert_prompts_do_not_share_stale_state(case, &results[0], &results[1]);
        }
    }

    /// Card 546b: the contract exposes no per-tensor state readback (`ReplayCache`/`DecodeCache`/
    /// `GraphIdentity` and the raw `GpuExecutor` state buffers this used to download are gone, R-546-8),
    /// so this drives the SAME graph through the executor contract (one entry per prompt, carried KV/SSM
    /// state kept internal to the entry) and checks only what the contract exposes: each step's logits
    /// and the resulting greedy token. The per-tensor state-vs-CPU-oracle comparison and the
    /// pipeline-rebuild-reuse check this replaced are not expressible against the contract; production
    /// correctness is still proven end to end by [`assert_production_generation`] below, which drives the
    /// real `generate_kv_gpu_cached` entry point and checks its tokens equal this oracle-verified
    /// sequence.
    fn device_sequence(
        case: &CachedFamilyCase,
        runner: &Runner,
        g: &Graph,
        prompt: &str,
        engine: &mut poot_executor::Engine<poot_gpu::device::WgpuDevice>,
        exe: poot_executor::ExecutableId,
    ) -> Vec<u32> {
        use poot_executor::Executor as _;
        let mut tokens = runner.encode(prompt).expect("encode synthetic prompt");
        let mut cpu_state = zero_state(g);
        let staged = staged_program(g, executor_target(engine))
            .unwrap_or_else(|error| panic!("{} stage decode graph: {error}", case.family));
        let entry = engine
            .add_entry(exe, &staged)
            .unwrap_or_else(|error| panic!("{} add_entry: {error}", case.family));
        let mut generated = Vec::with_capacity(GENERATED_STEPS);
        let mut max_relative_error = 0.0f32;
        let mut pos = 0;
        while generated.len() < GENERATED_STEPS {
            let token = tokens[pos];
            let cpu_inputs = runner
                .bind_decode(g, token, pos, &cpu_state, None)
                .unwrap_or_else(|error| panic!("{} CPU bind position {pos}: {error}", case.family));
            let (cpu_logits, next_cpu_state) =
                crate::core::cpu_oracle::cpu_eval_with_state(g, &cpu_inputs).unwrap_or_else(
                    |error| panic!("{} CPU eval position {pos}: {error}", case.family),
                );
            let bound = runner
                .bind_decode(g, token, pos, &[], None)
                .unwrap_or_else(|error| {
                    panic!("{} wgpu bind position {pos}: {error}", case.family)
                });
            let inputs = slot_step_inputs(g, &bound)
                .unwrap_or_else(|error| panic!("{} slot step inputs: {error}", case.family));
            let bytes = engine
                .step(exe, entry, &inputs, &mut poot_executor::NoSync)
                .unwrap_or_else(|error| {
                    panic!("{} cached wgpu decode position {pos}: {error}", case.family)
                })
                .read()
                .unwrap_or_else(|error| {
                    panic!(
                        "{} cached wgpu decode readback position {pos}: {error}",
                        case.family
                    )
                });
            let gpu_logits = HostTensor::f32(
                g.aval(g.output).shape.clone(),
                bytemuck::cast_slice::<u8, f32>(&bytes).to_vec(),
            );

            max_relative_error = max_relative_error.max(assert_close(
                case.family,
                &format!("logits at position {pos}"),
                &gpu_logits,
                &cpu_logits,
            ));
            let cpu_token = argmax(cpu_logits.as_f32().unwrap()).unwrap() as u32;
            let gpu_token = argmax(gpu_logits.as_f32().unwrap()).unwrap() as u32;
            assert_eq!(
                gpu_token, cpu_token,
                "{} greedy token at position {pos}",
                case.family
            );

            cpu_state = next_cpu_state;
            if pos + 1 == tokens.len() {
                tokens.push(cpu_token);
                generated.push(cpu_token);
            }
            pos += 1;
        }
        engine
            .remove_entry(exe, entry)
            .unwrap_or_else(|error| panic!("{} remove_entry: {error}", case.family));

        eprintln!(
            "card297 cached wgpu: {} prompt={prompt:?}, generated={}, states={}, max relative error={max_relative_error:.2e}",
            case.family,
            generated.len(),
            g.state.len()
        );
        generated
    }

    fn assert_production_generation(
        case: &CachedFamilyCase,
        runner: &Runner,
        exec: &mut dyn poot_executor::Executor,
        exe: poot_executor::ExecutableId,
        prompt: &str,
        expected: &[u32],
    ) {
        let prompt_len = runner
            .encode(prompt)
            .expect("encode synthetic prompt")
            .len();
        let tokens = runner
            .generate_kv_gpu_cached(prompt, GENERATED_STEPS, exec, exe, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .unwrap_or_else(|error| {
                panic!("{} production cached generation: {error}", case.family)
            });
        assert_eq!(
            &tokens[prompt_len..],
            expected,
            "{} production cached generation did not carry the verified state",
            case.family
        );
    }

    #[test]
    fn card297_cached_family_wgpu_matches_cpu() {
        let Ok(selected) = std::env::var("POOT_CARD297_FAMILY") else {
            eprintln!(
                "SKIP card297 required-wgpu case: set POOT_CARD297_FAMILY to one of nemotron-h,dsa"
            );
            return;
        };
        let case = CASES
            .iter()
            .find(|case| case.family == selected)
            .unwrap_or_else(|| panic!("unknown POOT_CARD297_FAMILY={selected:?}"));
        let (mut runner, g) = load_case(case);
        // Some synthetic weights greedily choose the fixture EOS before three steps. The acceptance covers a
        // bounded three-step cache continuation, so keep production generation running to max_new.
        runner.eos = u32::MAX;
        use poot_executor::Executor as _;
        let mut engine =
            poot_executor::Engine::new(poot_gpu::device::WgpuDevice::new().unwrap_or_else(
                |error| panic!("{} required wgpu unavailable: {error}", case.family),
            ));
        let exe = runner
            .load_on(&mut engine)
            .unwrap_or_else(|error| panic!("{} load_on: {error}", case.family));
        let results: Vec<Vec<u32>> = case
            .prompts
            .iter()
            .map(|prompt| device_sequence(case, &runner, &g, prompt, &mut engine, exe))
            .collect();
        // generate_kv_gpu_cached_sampled must not grow the executor's resident
        // entries across calls - two production generate calls on the same (exec, exe) (one per
        // prompt, exactly this loop) leave stats().recordings unchanged.
        let recordings_before_any_call = engine.stats().recordings;
        for (prompt, expected) in case.prompts.iter().zip(&results) {
            assert_production_generation(case, &runner, &mut engine, exe, prompt, expected);
            assert_eq!(
                engine.stats().recordings,
                recordings_before_any_call,
                "{} generate_kv_gpu_cached must not grow stats().recordings across calls",
                case.family
            );
        }
        assert_ne!(
            results[0], results[1],
            "{} two different prompts produced identical generated tokens; the per-request cache guard \
             may be ineffective",
            case.family
        );
        assert_eq!(results[0].len(), GENERATED_STEPS);
        assert_eq!(results[1].len(), GENERATED_STEPS);
    }
}

#[cfg(test)]
mod granite_moe_gpu {
    use super::{Runner, executor_target, slot_step_inputs, staged_program};

    use poot_executor::Executor as _;
    use poot_models::granite::trace_granite_prefill;
    use poot_test_util::max_abs_error;

    // MoE on the GPU: the gate is an on-device primitive composition (Ge-based top-k), so the whole MoE
    // block lowers on the GPU. Trace granite-MoE prefill, run through CPU eval and the executor contract,
    // compare the next-token argmax.
    #[test]
    #[ignore = "loads granite-moe-1b + runs prefill on the Arc GPU vs CPU"]
    fn granite_moe_prefill_gpu_matches_cpu() {
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("granite-moe-1b"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let runner = Runner::load(&dir).expect("load granite-moe");
        let grp = runner.granite_moe.expect("granite params");
        let tokens = runner.encode("The capital of France is").unwrap();
        let g = trace_granite_prefill(runner.cfg, grp, tokens.len());
        let bound = runner.bind(&g, &tokens).unwrap();
        let cpu = crate::core::cpu_oracle::cpu_eval(&g, &bound).unwrap();

        let mut engine = poot_executor::Engine::new(device);
        let exe = runner.load_on(&mut engine).unwrap();
        let staged = staged_program(&g, executor_target(&engine))
            .expect("stage granite-moe prefill for the executor contract");
        let entry = engine
            .add_entry(exe, &staged)
            .expect("add_entry (granite-moe prefill)");
        let inputs = slot_step_inputs(&g, &bound).expect("slot step inputs (granite-moe prefill)");
        let bytes = engine
            .step(exe, entry, &inputs, &mut poot_executor::NoSync)
            .expect("granite-moe graph on GPU (on-device primitive gate)")
            .read()
            .expect("granite-moe prefill readback");
        engine
            .remove_entry(exe, entry)
            .expect("remove_entry (granite-moe prefill)");
        let gpu_out: &[f32] = bytemuck::cast_slice(&bytes);

        let argmax = |t: &[f32]| {
            t.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let (ca, ga) = (argmax(cpu.as_f32().unwrap()), argmax(gpu_out));
        let maxdiff = max_abs_error(cpu.as_f32().unwrap(), gpu_out);
        eprintln!("granite-moe prefill: cpu_argmax={ca} gpu_argmax={ga} maxdiff={maxdiff:.3e}");
        // Observed maxdiff on this device is 3.1e-5 (f32 GPU vs f32 CPU over the full logit vector, 2026-09-27);
        // 1e-3 is ~30x that, loose enough for run-to-run accumulation order, tight enough to fail a wrong
        // expert or a corrupted logit tail that leaves the argmax intact.
        assert!(
            maxdiff < 1e-3,
            "granite-moe GPU prefill max_abs too large: {maxdiff:.3e}"
        );
        assert_eq!(ca, ga, "granite-moe GPU prefill argmax must match CPU");
    }
}

/// Real-wgpu verification for qwen3-MoE: exercises `bind_prefill_kv`/`bind_prefill_kv_paged`
/// (`graphs.rs`) with a prompt length > 1 through `Runner::prefill_kv_graph` (the arch-dispatched
/// KV-writing prefill). Not `#[ignore]`: both fixtures are tiny (2 layers, hidden<=64) and already used
/// without `#[ignore]` in `tests/coherence.rs`'s `qwen3_moe_tiny_checkpoint_generates_without_crashing`.
#[cfg(test)]
mod qwen3_moe_gpu {
    use super::{Runner, executor_target, slot_step_inputs, staged_program};

    use poot_executor::Executor as _;
    use poot_test_util::max_abs_error;

    fn zero_seed_state(
        g: &poot_graph_ir::Graph,
        inputs: &mut std::collections::HashMap<poot_graph_ir::ValueId, poot_eval::Value>,
    ) {
        for &(si, _) in &g.state {
            inputs.insert(
                si,
                poot_tensor::HostTensor::zeros(g.aval(si).shape.clone()).into(),
            );
        }
    }

    fn argmax(t: &[f32]) -> usize {
        t.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    /// Drives `g` through the executor contract (one entry, one step) against `bound`'s `Storage::Slot`
    /// entries, and returns the readback logits. State is implicit (a fresh entry starts zero-seeded, as
    /// `Engine::add_entry` does for every production prefill), matching the CPU oracle's explicit
    /// `zero_seed_state`.
    fn gpu_prefill(
        runner: &Runner,
        device: poot_gpu::device::WgpuDevice,
        g: &poot_graph_ir::Graph,
        bound: &std::collections::HashMap<poot_graph_ir::ValueId, poot_eval::Value>,
    ) -> Vec<u8> {
        let mut engine = poot_executor::Engine::new(device);
        let exe = runner.load_on(&mut engine).expect("load_on");
        let staged = staged_program(g, executor_target(&engine))
            .expect("stage qwen3-moe prefill for the executor contract");
        let entry = engine
            .add_entry(exe, &staged)
            .expect("add_entry (qwen3-moe prefill)");
        let inputs = slot_step_inputs(g, bound).expect("slot step inputs (qwen3-moe prefill)");
        let bytes = engine
            .step(exe, entry, &inputs, &mut poot_executor::NoSync)
            .expect("qwen3-moe prefill-kv graph on real GPU")
            .read()
            .expect("qwen3-moe prefill-kv readback");
        engine
            .remove_entry(exe, entry)
            .expect("remove_entry (qwen3-moe prefill)");
        bytes
    }

    /// `Runner::prefill_kv_graph` + `bind_prefill_kv` on a multi-token (`n=6`) prompt, CPU (`eval_with_state`,
    /// zero-seeded state) vs the executor contract (a fresh entry zero-seeds state itself).
    /// `yujiepan/qwen3-moe-tiny-random` has mixed dense/MoE layers (`decoder_sparse_step: 2`: layer 0 dense,
    /// layer 1 routed), so this also checks the per-layer dense/MoE switch on a real device.
    #[test]
    fn qwen3_moe_prefill_kv_gpu_matches_cpu_safetensors() {
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let runner = Runner::load(&dir).expect("load qwen3-moe-tiny");
        let tokens = runner.encode("The capital of France is").unwrap();
        let n = tokens.len();
        let cap = n + 4;
        let pg = runner
            .prefill_kv_graph(n, cap)
            .expect("Qwen3-MoE supports cached prefill");

        let mut cpu_inputs = runner.bind_prefill_kv(&pg, &tokens).unwrap();
        zero_seed_state(&pg, &mut cpu_inputs);
        let (cpu_logits, _) = crate::core::cpu_oracle::cpu_eval_with_state(&pg, &cpu_inputs)
            .expect("cpu prefill-kv oracle");

        let gpu_inputs = runner.bind_prefill_kv(&pg, &tokens).unwrap();
        let bytes = gpu_prefill(&runner, device, &pg, &gpu_inputs);
        let gpu_logits: &[f32] = bytemuck::cast_slice(&bytes);

        let (ca, ga) = (argmax(cpu_logits.as_f32().unwrap()), argmax(gpu_logits));
        let maxdiff = max_abs_error(cpu_logits.as_f32().unwrap(), gpu_logits);
        eprintln!(
            "qwen3-moe (safetensors, mixed layers) prefill-kv: cpu_argmax={ca} gpu_argmax={ga} maxdiff={maxdiff:.3e}"
        );
        assert_eq!(ca, ga, "qwen3-moe GPU prefill-kv argmax must match CPU");
        assert!(
            maxdiff < 1e-2,
            "qwen3-moe GPU prefill-kv max_abs too large: {maxdiff:.3e}"
        );
    }

    /// GGUF analog of [`qwen3_moe_prefill_kv_gpu_matches_cpu_safetensors`]: the self-authored, fully-sparse
    /// checkpoint (every layer routed) converted with `convert_hf_to_gguf.py`. Exercises the same
    /// `bind_prefill_kv` binder through `Runner::load_gguf` instead of `Runner::load`.
    #[test]
    fn qwen3_moe_prefill_kv_gpu_matches_cpu_gguf_sparse() {
        let Some(gguf_path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen3-moe-tiny-sparse-gguf/model.gguf"
        )) else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let runner = Runner::load_gguf(&gguf_path).expect("load qwen3-moe-tiny-sparse gguf");
        let tokens = runner.encode("The capital of France is").unwrap();
        let n = tokens.len();
        let cap = n + 4;
        let pg = runner
            .prefill_kv_graph(n, cap)
            .expect("Qwen3-MoE supports cached prefill");

        let mut cpu_inputs = runner.bind_prefill_kv(&pg, &tokens).unwrap();
        zero_seed_state(&pg, &mut cpu_inputs);
        let (cpu_logits, _) = crate::core::cpu_oracle::cpu_eval_with_state(&pg, &cpu_inputs)
            .expect("cpu prefill-kv oracle");

        let gpu_inputs = runner.bind_prefill_kv(&pg, &tokens).unwrap();
        let bytes = gpu_prefill(&runner, device, &pg, &gpu_inputs);
        let gpu_logits: &[f32] = bytemuck::cast_slice(&bytes);

        let (ca, ga) = (argmax(cpu_logits.as_f32().unwrap()), argmax(gpu_logits));
        let maxdiff = max_abs_error(cpu_logits.as_f32().unwrap(), gpu_logits);
        eprintln!(
            "qwen3-moe (gguf, all-sparse) prefill-kv: cpu_argmax={ca} gpu_argmax={ga} maxdiff={maxdiff:.3e}"
        );
        assert_eq!(
            ca, ga,
            "qwen3-moe gguf GPU prefill-kv argmax must match CPU"
        );
        assert!(
            maxdiff < 1e-2,
            "qwen3-moe gguf GPU prefill-kv max_abs too large: {maxdiff:.3e}"
        );
    }

    /// Real-hardware companion to the host-side `bind_prefill_kv`/`bind_prefill_kv_paged` coverage:
    /// binds the qwen3-moe prefill-kv graph through `bind_prefill_kv_paged` (identity slot map,
    /// since `prefill_kv_graph` declares no `Slot::SlotMap` input and no qwen3-moe paged tracer exists) and
    /// drives it through the executor contract.
    #[test]
    fn qwen3_moe_prefill_kv_paged_binder_resolves_on_real_gpu() {
        let Some(dir) =
            poot_test_util::model_path(poot_test_util::checkpoint!("qwen3-moe-tiny-sparse"))
        else {
            return;
        };
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no GPU ({e}); skipping");
                return;
            }
        };
        let runner = Runner::load(&dir).expect("load qwen3-moe-tiny-sparse");
        let tokens = runner.encode("Hello there").unwrap();
        let n = tokens.len();
        let cap = n + 4;
        let pg = runner
            .prefill_kv_graph(n, cap)
            .expect("Qwen3-MoE supports cached prefill");
        let inv: Vec<i32> = (0..cap as i32)
            .map(|i| if i < n as i32 { i } else { -1 })
            .collect();

        let mut cpu_inputs = runner.bind_prefill_kv_paged(&pg, &tokens, &inv).unwrap();
        zero_seed_state(&pg, &mut cpu_inputs);
        let (cpu_logits, _) = crate::core::cpu_oracle::cpu_eval_with_state(&pg, &cpu_inputs)
            .expect("cpu prefill-kv-paged oracle");

        let gpu_inputs = runner.bind_prefill_kv_paged(&pg, &tokens, &inv).unwrap();
        let bytes = gpu_prefill(&runner, device, &pg, &gpu_inputs);
        let gpu_logits: &[f32] = bytemuck::cast_slice(&bytes);

        let (ca, ga) = (argmax(cpu_logits.as_f32().unwrap()), argmax(gpu_logits));
        let maxdiff = max_abs_error(cpu_logits.as_f32().unwrap(), gpu_logits);
        eprintln!(
            "qwen3-moe prefill-kv (paged binder) on real GPU: cpu_argmax={ca} gpu_argmax={ga} maxdiff={maxdiff:.3e}"
        );
        assert_eq!(
            ca, ga,
            "qwen3-moe GPU prefill-kv-paged-binder argmax must match CPU"
        );
        assert!(
            maxdiff < 1e-2,
            "qwen3-moe GPU prefill-kv-paged-binder max_abs too large: {maxdiff:.3e}"
        );
    }
}

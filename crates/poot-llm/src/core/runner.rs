//! Runner core: the struct definition, safetensors/GGUF load, and configuration accessors.

use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_plan::WeightFormats;
use poot_load::lora::LoraAdapterPool;
use poot_models::deepseek2::DeepseekV2Params;
use poot_models::deepseek3::DeepseekV3Params;
use poot_models::deepseek32::Deepseek32Params;
use poot_models::gpt_oss::GptOssParams;
use poot_models::granite::GraniteParams;
use poot_models::mixtral::MixtralParams;
use poot_models::nemotron_h::NemotronHConfig;
use poot_models::olmoe::OlmoeParams;
use poot_models::qwen2::Qwen2Config;
use poot_models::qwen3moe::Qwen3MoeParams;

use crate::error::{Result, ResultExt, RunnerError};
use crate::text::tokenize::TextCodec;

mod impls;
mod weights;
mod yarn;

pub(crate) use weights::{dense_weight_map, weight_store};
pub(crate) use yarn::{deepseek2_yarn_params, deepseek2_yarn_resolve};

/// A loaded checkpoint of a family that is not yet a registered `Model`: the MoE and hybrid families
/// (POOT-738 moves them onto the driver). A registered family never loads here; it runs on
/// `driver::Driver` through `driver::ModelHandle`. The dense entry points are gone:
///
/// ```compile_fail,E0599
/// fn dense_prompt_lookup(runner: &poot_llm::Runner) {
///     let _ = runner.supports_prompt_lookup();
/// }
/// ```
///
/// There is no pooled MoE mode: a `Runner` holds every routed expert resident, and no entry point opts it into an
/// expert pool or loads a pooled expert store (Card 594 deleted them; expert pooling returns as layer
/// segmentation plus a runtime residency manager). Each former entry point is an unknown method:
///
/// ```compile_fail,E0599
/// fn pooled_attach(runner: &mut poot_llm::Runner) {
///     let _ = runner.attach_qwen3_moe_pool(4, 0);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn pooled_quant_attach(runner: &mut poot_llm::Runner) {
///     let _ = runner.attach_qwen3_moe_pool_quant(0, 4, 0);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn pooled_deepseek3_attach(runner: &mut poot_llm::Runner) {
///     let _ = runner.attach_deepseek3_pool(4, 0);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn pooled_mixtral_load() {
///     let _ = poot_llm::Runner::load_gguf_mixtral_pooled("mixtral.gguf");
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn pooled_mixtral_batched_load() {
///     let _ = poot_llm::Runner::load_gguf_mixtral_pooled_batched("mixtral.gguf", 4, 0, false);
/// }
/// ```
///
/// ```compile_fail,E0599
/// fn pooled_qwen3_moe_load() {
///     let _ = poot_llm::Runner::load_gguf_qwen3_moe_pooled_quant("qwen3-moe.gguf", 4, 0);
/// }
/// ```
pub struct Runner {
    /// The model shape every Runner tracer reads; read through [`Runner::config`].
    pub(crate) cfg: Qwen2Config,
    /// The baked store, by constant name: a dense weight is `Value::Host`, a packed weight's sources
    /// are `Value::Packed`s keyed by their `PackedSourceName`s, so every binder hands the executors
    /// (and the CPU oracle) the weight as stored. Card 739 deletes it with the Runner.
    pub(crate) weights: HashMap<String, Value>,
    /// Which of `weights`' constants the checkpoint stores packed (card 545a): derived
    /// once by the loader from the stored payload descriptors, and applied to every graph this Runner
    /// traces by `poot_graph_plan::bind_packed_weights` ([`Runner::bind_storage`]) before the
    /// graph reaches an executor or the CPU oracle. Empty for a dense checkpoint.
    pub(crate) formats: WeightFormats,
    /// Tokenization, chat rendering and guided constraint builders. The Runner reaches
    /// them through `Deref` until Card 739 deletes it.
    pub(crate) text: TextCodec,
    pub(crate) eos: u32,
    /// The model architecture string (`model_type` from config.json, or `general.architecture` from a GGUF):
    /// "granitemoe"/"qwen3_moe"/"mixtral"/... The tracer is selected by the family fields below.
    pub(crate) arch: String,
    /// `Some` for GraniteMoE (mixture-of-experts block + Granite scalars; `moe` is always `Some`). Selects
    /// the granite tracer. Dense granite is a registered family and loads through `ModelHandle`; this
    /// field goes with the MoE families (POOT-738).
    pub(crate) granite_moe: Option<GraniteParams>,
    /// `Some` for Qwen3-MoE (per-layer dense/routed-expert FFN switch, card 246). Selects the qwen3moe
    /// tracer (safetensors only so far - `Runner::load_gguf` never sets this). Not `Copy` (carries a
    /// `Vec<bool>` per-layer switch), unlike `granite_moe`, so dispatch sites clone or borrow it.
    pub(crate) qwen3_moe: Option<Qwen3MoeParams>,
    /// `Some` for Mixtral (card 135d, `crates/poot-models/src/mixtral.rs`,
    /// `specs/261-mixtral-tracer/spec.md`): selects `poot_models::mixtral::trace_mixtral_prefill`/
    /// `trace_mixtral_decode_kv_masked` (a GQA+RoPE decoder whose every layer routes through a top-k MoE
    /// MLP; no dense/MoE switch, no Granite scalars) on the CPU re-prefill generate paths. Safetensors only;
    /// `Runner::load_gguf` never sets this.
    pub(crate) mixtral: Option<MixtralParams>,
    /// `Some` for OlmoE (card 135d, `crates/poot-models/src/olmoe.rs`, `specs/262-olmoe-tracer/spec.md`):
    /// selects `poot_models::olmoe::trace_olmoe_prefill`/`trace_olmoe_decode_kv_masked` (olmo2's
    /// full-dimension QK-norm on a standard pre-norm block, every layer routed through a top-k MoE MLP with a
    /// non-renormalized-capable router, see `OlmoeParams::norm_topk_prob`) on the CPU re-prefill generate
    /// paths. Safetensors only; `Runner::load_gguf` never sets this.
    pub(crate) olmoe: Option<OlmoeParams>,
    /// `Some` for gpt-oss (card 135d, `crates/poot-models/src/gpt_oss.rs`, `specs/263-gptoss-tracer/spec.md`):
    /// selects `poot_models::gpt_oss::trace_gptoss_prefill`/`trace_gptoss_decode_kv_masked` (attention
    /// sinks, alternating sliding-window/full layers, biases everywhere, a clamped sigmoid-gated expert
    /// activation) on the CPU re-prefill generate paths. Safetensors only; the real 20b/120b checkpoints
    /// ship MXFP4 expert weights poot cannot dequantize on this path. `Runner::load_gguf` never sets this.
    pub(crate) gpt_oss: Option<GptOssParams>,
    /// `Some` for DeepSeek-V2 (card 135d, MLA, `poot_models::deepseek2`): selects
    /// `poot_models::deepseek2::trace_deepseek2_prefill`/`trace_deepseek2_decode_kv_masked` on the CPU
    /// re-prefill generate paths. Unlike `mixtral`/`olmoe`/`gpt_oss`, MLA has no correspondence to
    /// `Qwen2Config`'s `n_heads`/`n_kv_heads`/`head_dim`/`rotary_dim` (compressed shared KV latent,
    /// decoupled interleaved-pair RoPE on an independent slice), like `bloom`/`mpt`'s separate config type.
    /// `self.cfg` is still populated (vocab/hidden/layers/eps/max_pos) for generic bookkeeping; its
    /// attention-shape fields are unused by the deepseek2 tracer. See `poot_models::deepseek2`.
    pub(crate) deepseek2: Option<DeepseekV2Params>,
    /// `Some` for DeepSeek-V3: selects `poot_models::deepseek3::trace_deepseek3_prefill`/
    /// `trace_deepseek3_decode_kv_masked`. Uses the same `poot_models::deepseek2::DeepseekV2Config` as
    /// `deepseek2` for attention/MLA/YaRN (the attention is identical to V2's; only the router differs,
    /// see `poot_models::deepseek3`); `self.cfg` is populated as for `deepseek2`. A real V2 and V3 GGUF share
    /// `general.architecture == "deepseek2"` (see `gguf_deepseek2_is_v3_style` in `gguf.rs`), so `Runner`
    /// tells them apart by the `blk.N.exp_probs_b.bias` tensor. `deepseek2` and `deepseek3` are mutually
    /// exclusive.
    pub(crate) deepseek3: Option<DeepseekV3Params>,
    /// `Some` for DeepSeek-V3.2 DSA (spec 277, `poot_models::deepseek32`): selects
    /// `poot_models::deepseek32::trace_deepseek32_dsa_prefill`/`trace_deepseek32_dsa_decode` on the CPU
    /// re-prefill/masked-KV generate paths. Same `DeepseekV2Config`/`DeepseekV3MoeParams` shape as `deepseek3`
    /// (the only architectural change is DSA), plus [`DsaConfig`] for the Lightning Indexer. `deepseek_v32`
    /// is a distinct `model_type` from `deepseek_v3` (see [`Qwen2HfConfig::is_deepseek32`]), so `deepseek3`
    /// and `deepseek32` stay mutually exclusive. Unlike `deepseek2`/`deepseek3`, a `deepseek_v32` checkpoint
    /// is detected and loaded through a dedicated early-return path ([`Runner::load_deepseek32_impl`]) that
    /// calls `deepseek32_load::deepseek32_config_from_hf`/`build_deepseek32_weights`, which already build a
    /// complete weight map keyed by the tracer's HF constant names (block-FP8 linears kept packed, Card 654).
    /// CPU-oracle only: no GPU dispatch, no batched/pooled engine; see `deepseek32_load`.
    pub(crate) deepseek32: Option<Deepseek32Params>,
    /// `Some` for Nemotron-H (spec 279, `poot_models::nemotron_h`): selects
    /// `poot_models::nemotron_h::trace_nemotron_h_prefill` on the CPU re-prefill generate path. A real
    /// Nemotron-H `config.json` (`hybrid_override_pattern`, `mamba_num_heads`, no `rope_theta`; NoPE) lacks
    /// `Qwen2HfConfig`'s required fields (`rope_theta`/`max_position_embeddings`/`tie_word_embeddings`) and
    /// would fail that parse, so detection is a raw-JSON peek before the parse, like `bloom`/`mpt`. See
    /// [`Runner::load_nemotron_h_impl`] and `nemotron_h_load`. `self.cfg` holds only
    /// vocab/hidden/layers/eps (a hybrid Mamba/Attention/MLP stack has no single attention shape).
    /// CPU-oracle only: no GPU dispatch, no batched/pooled engine, and no fixed-KV masked decode tracer in
    /// `poot_models::nemotron_h` (only per-layer decode-step compositions and the prefill trace), so
    /// [`Runner::generate_kv_masked`] bails for a Nemotron-H `Runner`; use [`Runner::generate`]/
    /// [`Runner::generate_sampled`] (re-prefill).
    pub(crate) nemotron_h: Option<NemotronHConfig>,
    /// Sliding-window attention size. `None` = full causal. `Some(w)`: query at position `i` attends only
    /// to keys `j` with `i - j < w`. Used in prefill (the `mask.prefill` step input) and decode (`Slot::Mask`)
    /// binders; VLM paths are unaffected (always `None`).
    pub(crate) sliding_window: Option<usize>,
    /// Spec 248 Phase 2 (epic 129 B8): the multi-adapter pool a batched-serving caller registers named
    /// adapters into via [`Self::register_lora_adapter`]/[`Self::register_lora_adapter_dir`], plus (hot-load/
    /// unload) any post-construction override of the pool's stacked consts, its traced rank ceiling, and
    /// per-adapter lease counters; see [`LoraHotState`] for why they share one `RwLock`. Only
    /// [`Self::rebind_lora_pool`] binds the pool's stacked constants into `Self::weights` at
    /// startup (an explicit step, not automatic on each `register_*`). An empty `LoraHotState` behaves as
    /// the old `None`.
    pub(crate) lora_hot: std::sync::RwLock<LoraHotState>,
}

/// Spec 248 hot-load/unload: everything a hot-load/hot-unload admin op mutates atomically, behind one
/// lock. `Arc<Runner>`-shared serving threads (the HTTP admin handlers, `crates/poot-serve/src/admin.rs`)
/// mutate the pool after the `Runner` is shared, which a bare field never allowed; this replaces the old
/// `Option<LoraAdapterPool>`. `register_lora_adapter`/`rebind_lora_pool` (still `&mut self`, the
/// startup-only path, before the `Runner` is `Arc`-wrapped) reach the same state via `RwLock::get_mut`
/// (no lock cost; `&mut Runner` proves exclusive access).
#[derive(Default)]
pub(crate) struct LoraHotState {
    /// The registered-adapter pool (`None` until the first `register_lora_adapter*`/`hot_load_lora_adapter*`
    /// call).
    pub(crate) pool: Option<LoraAdapterPool>,
    /// Per-`{module}.lora_{a,b}_stacked`/`.lora_scaling_vec` name overrides of `Self::weights`' startup
    /// binding, written only by a hot-load/hot-unload once the `Runner` is `Arc`-shared (`Self::weights` has
    /// no interior mutability and is not touched post-load). [`Runner::bind_decode_batched`]'s
    /// `Storage::Const` arm checks this map first, then `Self::weights`. Empty forever on a server that
    /// never hot-reloads.
    pub(crate) overrides: HashMap<String, Value>,
    /// The batched-pool graph's traced rank dimension (the last [`Runner::rebind_lora_pool`]'s
    /// `LoraBatchedSpec::rank`, the GPU-resident graph's fixed shape). A hot-loaded adapter whose `r` exceeds
    /// this is refused (see [`Runner::hot_load_lora_adapter`]); changing it needs a full re-trace (spec 248
    /// "adapter hot-load/unload"). `0` until `rebind_lora_pool` runs.
    pub(crate) rank_ceiling: usize,
    /// Live request-lease count per pool index (hot-unload safety). A named request increments its adapter's
    /// counter atomically with name resolution under this state's write lock, before enqueueing. The
    /// request-owned [`LoraAdapterLease`](crate::LoraAdapterLease) decrements it when its last clone is
    /// dropped, so queueing, admission, preemption, completion, cancellation, failure, and shutdown share
    /// one lifetime contract. [`Runner::hot_unload_lora_adapter`] refuses (HTTP 409 from the admin handler)
    /// while an index's count is nonzero.
    pub(crate) in_flight: HashMap<usize, std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    /// Bumped by every hot-load/hot-unload (spec 248); `0` if neither has run.
    pub(crate) generation: u64,
}

/// `model_type` values the safetensors `Runner::load` path traces: the MoE and hybrid families that are
/// not yet registered families (POOT-738 moves them onto `Model`). A registered family never reaches
/// this list: `Runner::load` refuses it first ([`RunnerError::RegisteredFamily`]).
const SAFETENSORS_SUPPORTED_ARCHS: &[&str] = &[
    "granitemoe",
    "qwen3_moe",
    "mixtral",
    "olmoe",
    "gpt_oss",
    "deepseek_v2",
    "deepseek_v3",
    "deepseek_v32",
];

/// Rejects a safetensors `model_type` this Runner cannot trace (see `SAFETENSORS_SUPPORTED_ARCHS`).
/// Mirrors the allowlist `match` in `Runner::load_gguf`. Pure, so it is unit-testable without loading a
/// file.
fn validate_safetensors_arch(model_type: &str) -> Result<()> {
    if SAFETENSORS_SUPPORTED_ARCHS.contains(&model_type) {
        return Ok(());
    }
    match model_type {
        // qwen3_next and gemma4 have no Runner route: their private runtimes are deleted (card 595) and they
        // return as ordinary driver models (cards 574 and 575). Until then they take the typed refusal below
        // like any unknown architecture.
        "qwen2_vl" | "qwen2_5_vl" => bail!(
            "safetensors arch {model_type:?} not supported by Runner yet (Qwen-VL still needs vision \
             weights, visual embedding splice, and complete mRoPE model-input admission); use \
             load_qwen2_5_vl_text_tower_config_from_files for the typed text-tower config handoff"
        ),
        other => Err(RunnerError::UnsupportedModel {
            model_type: other.to_string(),
            reason: "not a supported safetensors architecture (supported: granitemoe, qwen3_moe, \
                     mixtral, olmoe, gpt_oss, deepseek_v2, deepseek_v3, deepseek_v32)",
        }),
    }
}

/// The fallback chat format of a family still on the Runner: Granite's for granitemoe, ChatML for the
/// other MoE and hybrid families. Each registered family names its own in its `ModelConfig`; this goes
/// with the families when POOT-738 moves them onto `Model`.
pub(crate) fn runner_chat_format(arch: &str) -> poot_models::chat::ChatFormat {
    match arch {
        "granitemoe" => poot_models::chat::ChatFormat::Granite,
        _ => poot_models::chat::ChatFormat::ChatML,
    }
}

/// Refuse a checkpoint whose config names a registered family: it loads through `ModelHandle` and
/// the driver, and the Runner keeps no second path for it. One registry resolve decides; a config the
/// registry cannot read (no family key) or does not know continues on the Runner's own path.
pub(crate) fn refuse_registered(raw: &poot_models::registry::RawConfig<'_>) -> Result<()> {
    let registry = poot_models::registry::Registry::builtin()
        .map_err(|e| err!("the builtin registry: {e}"))?;
    match registry.resolve(raw) {
        Ok(entry) => Err(RunnerError::RegisteredFamily {
            family: entry.family,
        }),
        Err(_) => Ok(()),
    }
}

impl std::ops::Deref for Runner {
    type Target = TextCodec;

    /// The text services until Card 739 deletes the Runner.
    fn deref(&self) -> &TextCodec {
        &self.text
    }
}

impl Runner {
    /// Put this checkpoint's packed storage onto a graph the Runner traced (card 545a):
    /// every constant `formats` records packed becomes the decode of its stored owners, so the graph
    /// binds the packed carriers in `weights` by name. The one way storage reaches a Runner graph; a
    /// graph that declares a packed weight in a shape its storage cannot serve is a typed refusal. The
    /// claims are `compile`'s (the CPU oracle claims explicitly in `cpu_oracle`). A dense
    /// checkpoint's graph passes through untouched.
    pub(crate) fn bind_storage(&self, g: poot_graph_ir::Graph) -> Result<poot_graph_ir::Graph> {
        let g = weights::bind_stored_dtypes(&g, &self.weights, &self.formats)
            .context("binding the checkpoint's stored dtypes")?;
        if self.formats.is_empty() {
            return Ok(g);
        }
        Ok(poot_graph_plan::bind_packed_weights(&g, &self.formats)?)
    }

    /// The model shape the Runner's tracers read.
    pub fn config(&self) -> &Qwen2Config {
        &self.cfg
    }

    /// The end-of-sequence token id (config.json `eos_token_id`); a generation loop stops on it.
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Card 546a's one Runner consumer: bakes this Runner's `weights` into the
    /// [`poot_quant::weights::WeightStore`] `exec`'s backend-neutral contract binds every decode/
    /// prefill graph's consts from (`weights::weight_store`), and loads it. The returned
    /// `ExecutableId` is reusable across `generate_kv_gpu_cached`/`_sampled` calls (and any future
    /// prefill entry sharing it), since `load_weights` records the store once and uploads nothing
    /// until an entry actually binds a const.
    pub fn load_on(
        &self,
        exec: &mut dyn poot_executor::Executor,
    ) -> Result<poot_executor::ExecutableId> {
        let store = weights::weight_store(&self.weights)?;
        exec.load_weights(
            std::sync::Arc::new(store),
            poot_executor::WeightSource::ConstNames,
        )
        .map_err(|error| err!("load_on: {error}"))
    }

    /// The stored weight `name` as the CPU oracle binds it to a const declared `declared` (a dense
    /// tensor or a packed source component), or a typed error naming the missing weight.
    ///
    /// A dense weight binds as stored: a tracer declares the dtype its loader stores, so a stored
    /// and declared dtype that differ are a [`weights::StoredDtypeError::Mismatch`], never a conversion.
    ///
    /// `pub`, not `pub(crate)` (card 549 fallout): the pure-CPU diagnostics in `poot-llm/tests/`
    /// bind each weight through this directly.
    pub fn weight_value(
        &self,
        name: &str,
        declared: &poot_graph_ir::TensorType,
    ) -> Result<poot_eval::Value> {
        let value = self
            .weights
            .get(name)
            .ok_or_else(|| err!("no weight bound for {name}"))?;
        if let poot_eval::Value::Host(tensor) = value
            && tensor.dtype() != declared.dtype
        {
            return Err(weights::StoredDtypeError::Mismatch {
                name: name.to_string(),
                stored: tensor.dtype(),
                declared: declared.dtype,
            })
            .context("binding a weight for the CPU oracle");
        }
        Ok(value.clone())
    }

    /// The dense tensor weight `name`, for a host-side reader (LoRA stacking, host embedding
    /// gathers): a missing weight or a packed one is a typed error, never a reinterpretation of
    /// packed source bytes.
    pub(crate) fn dense_weight(&self, name: &str) -> Result<&poot_tensor::HostTensor> {
        match self.weights.get(name) {
            Some(value) => value.as_host().ok_or_else(|| {
                err!("{name} is stored packed; this host-side reader needs a dense weight")
            }),
            None => Err(err!("no weight bound for {name}")),
        }
    }

    /// Total bytes the model weights occupy in GPU memory once resident: a packed weight uploads its
    /// source components as stored (card 545a: the native bytes, never a widened copy) and a dense
    /// weight its stored payload (a bf16/f16 weight is 2 bytes/elem, exactly what the executor's
    /// const upload sends; R475-011 was a gauge that counted a host f32 mirror instead). The
    /// dominant fixed VRAM cost on the GPU decode path (weights stay warm in the executor's const
    /// cache); paired with the engine's KV-pool bytes as the resident-memory gauge (card 030/051).
    ///
    /// Also the source of `poot-serve`'s `poot_gpu_weight_bytes` gauge, read once at load
    /// (`startup.rs`) since the resident weight set is fixed for the model's lifetime.
    pub fn weight_bytes(&self) -> usize {
        self.weights
            .values()
            .map(poot_eval::Value::physical_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests;

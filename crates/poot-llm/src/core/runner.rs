//! Runner core: the struct definition, safetensors/GGUF load, and configuration accessors.

use std::collections::HashMap;

use poot_eval::Value;
use poot_load::lora::LoraAdapterPool;
use poot_models::qwen2::Qwen2Config;

use crate::error::{Result, RunnerError};
use crate::text::tokenize::TextCodec;

mod impls;
mod weights;

pub(crate) use weights::weight_store;

/// The text services, weights and LoRA pool of a loaded checkpoint, held until the Runner is deleted
/// (POOT-739). Every decoder family is a registered `Model` and runs on `driver::Driver` through
/// `driver::ModelHandle`; no family loads here. The dense entry points are gone:
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
    /// Tokenization, chat rendering and guided constraint builders. The Runner reaches
    /// them through `Deref` until Card 739 deletes it.
    pub(crate) text: TextCodec,
    pub(crate) eos: u32,
    /// The model architecture string (`model_type` from config.json, or `general.architecture` from a GGUF).
    pub(crate) arch: String,
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

/// The refusal for a safetensors `model_type` the Runner cannot load. Every decoder family is a
/// registered `Model` ([`refuse_registered`] names it); nothing else loads here.
fn unsupported_safetensors_arch(model_type: &str) -> RunnerError {
    match model_type {
        // qwen3_next has no Runner route: its private runtime is deleted (card 595) and it returns as
        // an ordinary driver model (card 575). Until then it takes the typed refusal below like any
        // unknown architecture.
        "qwen2_vl" | "qwen2_5_vl" => err!(
            "safetensors arch {model_type:?} not supported by Runner yet (Qwen-VL still needs vision \
             weights, visual embedding splice, and complete mRoPE model-input admission); use \
             load_qwen2_5_vl_text_tower_config_from_files for the typed text-tower config handoff"
        ),
        other => RunnerError::UnsupportedModel {
            model_type: other.to_string(),
            reason: "no family loads on the Runner: every decoder family is a registered Model",
        },
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

/// A `Runner` with no weights and an empty vocabulary, built by direct struct construction (the
/// fields are visible in the crate): the plumbing tests (weight bytes, weight binding, the LoRA pool)
/// need no checkpoint, and no family loads on the Runner any more.
#[cfg(test)]
pub(crate) mod fixtures {
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    use super::{LoraHotState, Runner};

    pub(crate) fn bare_runner() -> Runner {
        let model = WordLevel::builder()
            .vocab(std::collections::HashMap::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build empty wordlevel model");
        Runner {
            cfg: poot_models::qwen2::Qwen2Config::default(),
            weights: std::collections::HashMap::new(),
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
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        }
    }
}

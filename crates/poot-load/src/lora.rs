//! LoRA adapter loading (spec 248, epic 129 B8): the PEFT/HF adapter format,
//! `adapter_config.json` + `adapter_model.safetensors` next to (not merged into) a base
//! checkpoint. Format per PEFT's docs and source (citations in spec 248 "Adapter weight loading"):
//!
//! - State dict keys are `base_model.model.{module_path}.lora_A.weight` / `.lora_B.weight`, e.g.
//!   `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` for a causal LM. A
//!   single-adapter save strips the adapter name segment; a multi-adapter save keeps it
//!   (`lora_A.default.weight`, ...). This loader accepts either by locating the `lora_A`/`lora_B`
//!   segment and ignoring anything between it and the trailing `.weight`.
//! - `lora_A.weight` is `[r, in]` and `lora_B.weight` is `[out, r]` (bias-free `nn.Linear`, so the
//!   `[out_features, in_features]` convention of the base model's `.weight` tensors).
//! - `adapter_config.json` carries `r`, `lora_alpha`, `target_modules` (module name suffixes, e.g.
//!   `["q_proj","v_proj"]` or all seven attention+MLP projections), and `use_rslora`
//!   (`alpha/sqrt(r)` instead of `alpha/r`).
//!
//! Embedding-layer LoRA (`lora_embedding_A`/`lora_embedding_B`, a bare `nn.Parameter` pair with a
//! different shape convention) and DoRA's magnitude vector are out of scope (spec 248) and are
//! rejected with an error rather than ignored.

use crate::LoadError;
use crate::safetensors::{decode_dense, dense_bytes};
use poot_quant::weights::WeightStore;
use poot_tensor::HostTensor;
use std::collections::HashMap;
use std::path::Path;

/// The `adapter_config.json` fields this loader needs. PEFT's file has many training and
/// bookkeeping fields; `#[serde(deny_unknown_fields)]` is not set so they are ignored.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LoraAdapterConfig {
    /// Adapter rank `r` (LoRA's `A`/`B` inner dimension). Typical range 8-64.
    pub r: usize,
    /// PEFT's `lora_alpha`, the scale numerator (see `scaling`).
    pub lora_alpha: f32,
    /// Module name suffixes the adapter targets, e.g. `["q_proj","v_proj"]`, matched against a
    /// graph's per-projection weight prefix by suffix (`LoraAdapter::weight_for`), as PEFT matches
    /// `nn.Module` names. PEFT's single-regex-string form is unsupported (rejected by serde's type
    /// mismatch, spec 248).
    pub target_modules: Vec<String>,
    /// Rank-stabilized LoRA (Kalajdzievski 2023): scales by `alpha/sqrt(r)` instead of `alpha/r`.
    #[serde(default)]
    pub use_rslora: bool,
}

impl LoraAdapterConfig {
    /// PEFT's correction scale: `lora_alpha / r` (standard), or `lora_alpha / sqrt(r)` under rslora.
    pub fn scaling(&self) -> f32 {
        if self.use_rslora {
            self.lora_alpha / (self.r as f32).sqrt()
        } else {
            self.lora_alpha / self.r as f32
        }
    }
}

/// One target module's adapter weights, transposed to poot's `x @ W` layout (as
/// `SafeTensors::dequant_fp8` transposes the base weight): `a` is `[in, r]` (PEFT stores
/// `[r, in]`), `b` is `[r, out]` (PEFT stores `[out, r]`). Feed both to
/// `poot_graph_ir::ops::lora_linear`.
#[derive(Debug, Clone)]
pub struct LoraWeight {
    /// `[in, r]`. A LoRA adapter is always plain F32.
    pub a: HostTensor,
    /// `[r, out]`.
    pub b: HostTensor,
}

/// A loaded LoRA adapter: its config plus each target module's transposed `A`/`B` pair, keyed by
/// checkpoint module path with the `base_model.model.` prefix and
/// `.lora_A/.lora_B[.<adapter>].weight` suffix stripped (e.g. `model.layers.0.self_attn.q_proj`).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub config: LoraAdapterConfig,
    pub weights: HashMap<String, LoraWeight>,
}

impl LoraAdapter {
    /// Load `adapter_config.json` + `adapter_model.safetensors` from a PEFT adapter directory
    /// (`PeftModel.save_pretrained` output). The older pickle `adapter_model.bin` is not supported.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config: LoraAdapterConfig =
            serde_json::from_slice(&std::fs::read(dir.join("adapter_config.json"))?)?;
        let store =
            crate::safetensors::load_weight_store_file(dir.join("adapter_model.safetensors"))?;
        Self::from_store(config, &store)
    }

    /// Parse an already-loaded [`WeightStore`] (the non-file-IO half of `load`, and the test
    /// entry point).
    pub fn from_store(config: LoraAdapterConfig, store: &WeightStore) -> Result<Self, LoadError> {
        let mut a_by_module: HashMap<String, String> = HashMap::new();
        let mut b_by_module: HashMap<String, String> = HashMap::new();
        for key in store.keys() {
            let key = key.as_str();
            // DoRA checkpoints carry full lora_A/lora_B tensors plus a magnitude vector (PEFT's
            // `layer.py::update_layer`), so the empty check below would not catch them. Reject
            // eagerly rather than load with the magnitude renormalization dropped.
            if is_dora_magnitude_key(key) {
                return Err(LoadError::SafeTensors(format!(
                    "lora adapter: {key} is a DoRA magnitude-vector tensor - DoRA adapters are not \
                     supported (the magnitude renormalization would be silently dropped) - see \
                     poot_load::lora module docs"
                )));
            }
            match parse_lora_key(key) {
                Some((module, LoraSide::A)) => {
                    a_by_module.insert(module, key.to_string());
                }
                Some((module, LoraSide::B)) => {
                    b_by_module.insert(module, key.to_string());
                }
                None => {} // not a lora_A/lora_B tensor (or an out-of-scope embedding key); skip
            }
        }
        if a_by_module.is_empty() {
            return Err(LoadError::SafeTensors(
                "lora adapter: no lora_A.*.weight tensors found (embedding-only or DoRA adapters are \
                 not supported - see poot_load::lora module docs)"
                    .into(),
            ));
        }
        let mut weights = HashMap::with_capacity(a_by_module.len());
        for (module, a_key) in a_by_module {
            let b_key = b_by_module.get(&module).ok_or_else(|| {
                LoadError::SafeTensors(format!("lora adapter: {module} has lora_A but no lora_B"))
            })?;
            let a = decode_dense(dense_bytes(store, &a_key)?)?;
            let b = decode_dense(dense_bytes(store, b_key)?)?;
            if a.0.len() != 2 || b.0.len() != 2 {
                return Err(LoadError::SafeTensors(format!(
                    "lora adapter: {module} lora_A/lora_B must be 2-D linear weights, got {:?}/{:?}",
                    a.0, b.0
                )));
            }
            if a.0[0] != config.r || b.0[1] != config.r {
                return Err(LoadError::SafeTensors(format!(
                    "lora adapter: {module} lora_A shape {:?} / lora_B shape {:?} don't match config r={}",
                    a.0, b.0, config.r
                )));
            }
            weights.insert(
                module,
                LoraWeight {
                    a: transpose_2d(&a),
                    b: transpose_2d(&b),
                },
            );
        }
        Ok(LoraAdapter { config, weights })
    }

    /// Look up the adapter weight for a graph weight prefix like `model.layers.3.self_attn.q_proj`
    /// by exact module-path match. `None` if the module was not LoRA-adapted (untargeted, or
    /// pruned by rank/alpha_pattern overrides this loader does not parse, spec 248).
    pub fn weight_for(&self, module_path: &str) -> Option<&LoraWeight> {
        self.weights.get(module_path)
    }
}

/// One targeted module's stacked multi-adapter weights (spec 248 Phase 2), in the layout
/// `poot_graph_ir::ops::lora_linear_batched` expects: `a [n+1, in, r]`, `b [n+1, r, out]`,
/// `scaling [n+1]`. Slot 0 is the reserved "no adapter" sentinel (`LoraAdapterPool::NO_ADAPTER`);
/// slots `1..=n` are the registered adapters in registration order.
#[derive(Debug, Clone)]
pub struct LoraStackedModule {
    /// `[n_adapters+1, in, r]`, matmul-ready (poot's `[in,out]` layout per adapter slot).
    pub a: HostTensor,
    /// `[n_adapters+1, r, out]`.
    pub b: HostTensor,
    /// `[n_adapters+1]`: each slot's `lora_alpha/r` (rslora: `lora_alpha/sqrt(r)`); slot 0 is 0.0
    /// (its `a`/`b` are all-zero, so the correction is inert regardless).
    pub scaling: HostTensor,
    /// The padded rank of every slot: the max `r` among adapters targeting this module. Smaller
    /// adapters are zero-padded, which is exact since padded ranks contribute nothing to `B@A`.
    pub r: usize,
}

/// A pool of loaded LoRA adapters registered under stable names and stacked into the
/// `[n_adapters+1, ...]` layout `poot_graph_ir::ops::lora_linear_batched` expects (spec 248
/// Phase 2). Stacking is per targeted module (each layer's `q_proj`/`v_proj` has its own stacked
/// tensor); only the name -> index assignment is shared, so a request's adapter id is picked once
/// per batch.
///
/// Index 0 is reserved as [`LoraAdapterPool::NO_ADAPTER`]: every module's slot 0 is all-zero
/// `A`/`B` with `scaling=0.0`, so a row with `idx` 0 gets an inert correction (`x@0=0`, `0@B=0`)
/// with no graph branch, the per-row analog of the per-module zero-fill in
/// `poot_llm::Runner::attach_lora_adapter` (FR-008, update 0577).
///
/// This is a stacking/bookkeeping helper with no Runner or `poot-serve` wiring (spec 248 Phase 2):
/// a caller `register`s adapters, builds an `idx` vector via `index_of`, and calls
/// `stack_for_module` per targeted module to get the constants `lora_linear_batched` needs.
#[derive(Debug, Clone, Default)]
pub struct LoraAdapterPool {
    /// Name -> pool index (1-based; see `NO_ADAPTER`).
    index: HashMap<String, usize>,
    /// Pool index `i` (1-based) lives at `adapters[i-1]`.
    adapters: Vec<LoraAdapter>,
    /// Pool index `i` (1-based) lives at `identities[i-1]`: a unique value identifying the effective
    /// weights in slot `i` for paged prefix-cache safety (spec 248, ADR-0030). Assigned afresh on every
    /// `register`/`hot_load`, so replacing or hot-reloading an adapter (even into the same index) never
    /// aliases K/V computed under the previous weights. `0` marks a free slot and is never a live
    /// adapter's identity.
    identities: Vec<u64>,
    /// Next identity to hand out. Starts at 0 so the first assigned value is 1; `0` stays reserved for
    /// the no-adapter/base identity.
    next_identity: u64,
}

impl LoraAdapterPool {
    /// The reserved "no adapter" row index.
    pub const NO_ADAPTER: usize = 0;

    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh, never-reused identity for the effective weights in a pool slot.
    fn fresh_identity(&mut self) -> u64 {
        self.next_identity = self
            .next_identity
            .checked_add(1)
            .expect("LoRA adapter identity counter exhausted");
        self.next_identity
    }

    /// Register an adapter, or replace it if `name` is already registered, returning its pool
    /// index. Replacing keeps the index and never renumbers other adapters, so an index in an
    /// in-flight batch's `idx` vector stays valid across a hot-swap of a different name.
    pub fn register(&mut self, name: impl Into<String>, adapter: LoraAdapter) -> usize {
        let name = name.into();
        if let Some(&existing) = self.index.get(&name) {
            self.adapters[existing - 1] = adapter;
            let identity = self.fresh_identity();
            self.identities[existing - 1] = identity;
            return existing;
        }
        let identity = self.fresh_identity();
        self.adapters.push(adapter);
        self.identities.push(identity);
        let idx = self.adapters.len(); // 1-based; 0 stays reserved for NO_ADAPTER
        self.index.insert(name, idx);
        idx
    }

    /// The pool index for `name`, or `None` if never registered (callers typically fall back to
    /// `NO_ADAPTER`).
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// The effective-weights identity for pool index `idx`, or `None` if the index is out of range or a
    /// free slot. A live adapter's identity is stable across requests and changes on re-registration or
    /// hot-load, so prefix-cache entries never cross adapter identities (spec 248, ADR-0030).
    pub fn identity_of(&self, idx: usize) -> Option<u64> {
        let pos = idx.checked_sub(1)?;
        self.identities
            .get(pos)
            .copied()
            .filter(|&identity| identity != 0)
    }

    /// Spec 248 hot-load: bring a new name online by taking the first free slot (empty
    /// `LoraAdapter::weights`, either a startup placeholder from `poot-serve`'s
    /// `--lora-pool-capacity` or a slot freed by [`Self::hot_unload`]) without growing the pool. A
    /// resident/captured graph bakes the pool's slot count and shared rank into const tensor shapes
    /// at trace time, so growing would need a re-trace. Reusing provisioned capacity changes only
    /// slot data, which `poot_gpu::GpuExecutor::evict_const` forces to re-upload.
    ///
    /// Errors if `name` is already registered (use [`Self::register`] to replace) or every slot is
    /// occupied by a real adapter (`hot_unload` one first, or restart with a larger pool).
    pub fn hot_load(
        &mut self,
        name: impl Into<String>,
        adapter: LoraAdapter,
    ) -> Result<usize, LoadError> {
        let name = name.into();
        if self.index.contains_key(&name) {
            return Err(LoadError::SafeTensors(format!(
                "lora pool: {name:?} is already registered - hot_load is only for bringing a NEW name \
                 online (use register()/replace-in-place semantics for an already-registered name)"
            )));
        }
        let Some(pos) = self.adapters.iter().position(|a| a.weights.is_empty()) else {
            return Err(LoadError::SafeTensors(format!(
                "lora pool: no free slot - the pool is at its provisioned capacity of {} adapter(s); \
                 hot_unload an adapter first, or restart with a larger --lora-pool-capacity",
                self.adapters.len()
            )));
        };
        self.adapters[pos] = adapter;
        let identity = self.fresh_identity();
        self.identities[pos] = identity;
        let idx = pos + 1; // 1-based; 0 stays reserved for NO_ADAPTER
        self.index.insert(name, idx);
        Ok(idx)
    }

    /// Spec 248 hot-unload: remove `name`'s mapping and revert its slot to the empty placeholder
    /// [`Self::hot_load`] looks for, without shrinking the pool or changing any other adapter's
    /// index (in-flight requests reference adapters by pool index). Returns the freed index and
    /// the removed adapter's `target_modules` (the caller recomputes those stacked consts to zero,
    /// see `poot_llm::Runner::hot_unload_lora_adapter`), or `None` if `name` was never registered.
    pub fn hot_unload(&mut self, name: &str) -> Option<(usize, Vec<String>)> {
        let idx = self.index.remove(name)?;
        let old_targets = std::mem::take(&mut self.adapters[idx - 1].config.target_modules);
        self.adapters[idx - 1] = LoraAdapter {
            config: LoraAdapterConfig {
                r: 0,
                lora_alpha: 0.0,
                target_modules: Vec::new(),
                use_rslora: false,
            },
            weights: HashMap::new(),
        };
        self.identities[idx - 1] = 0; // the slot is free; a later hot_load assigns a fresh identity
        Some((idx, old_targets))
    }

    /// Number of registered adapters (excluding the reserved slot 0).
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Every registered name and its pool index (`poot-serve`'s `GET /v1/lora_adapters`), excluding
    /// unused placeholder slots. Order is unspecified (`HashMap`).
    pub fn names(&self) -> Vec<(String, usize)> {
        self.index.iter().map(|(n, &i)| (n.clone(), i)).collect()
    }

    /// Stack every registered adapter's `A`/`B` for one module path (e.g.
    /// `model.layers.3.self_attn.q_proj`) into `LoraStackedModule`'s `[n_adapters+1, ...]` layout.
    ///
    /// An adapter that does not target `module_path` (`weight_for` returns `None`) gets an
    /// all-zero slot, the same zero-correction identity as the Runner's partial-adapter handling
    /// (FR-008, update 0577).
    ///
    /// Returns `Ok(None)` if no registered adapter targets `module_path` (the caller should use
    /// plain `linear`). Returns `Err` if two adapters that target it disagree on `in`/`out`
    /// dimensions (e.g. trained against different base checkpoints).
    pub fn stack_for_module(
        &self,
        module_path: &str,
    ) -> Result<Option<LoraStackedModule>, LoadError> {
        let mut in_out: Option<(usize, usize)> = None;
        let mut r_max = 0usize;
        for adapter in &self.adapters {
            if let Some(w) = adapter.weight_for(module_path) {
                let (in_dim, r) = (w.a.shape()[0], w.a.shape()[1]);
                let out_dim = w.b.shape()[1];
                match in_out {
                    None => in_out = Some((in_dim, out_dim)),
                    Some((ei, eo)) if ei != in_dim || eo != out_dim => {
                        return Err(LoadError::SafeTensors(format!(
                            "lora pool: {module_path} adapters disagree on shape: \
                             expected in={ei} out={eo}, got in={in_dim} out={out_dim}"
                        )));
                    }
                    _ => {}
                }
                r_max = r_max.max(r);
            }
        }
        let Some((in_dim, out_dim)) = in_out else {
            return Ok(None);
        };
        let n = self.adapters.len();
        let slots = n + 1; // +1 for the reserved NO_ADAPTER sentinel at index 0
        let mut a_data = vec![0.0f32; slots * in_dim * r_max];
        let mut b_data = vec![0.0f32; slots * r_max * out_dim];
        let mut scaling = vec![0.0f32; slots];
        for (i, adapter) in self.adapters.iter().enumerate() {
            let slot = i + 1; // slot 0 stays all-zero (NO_ADAPTER)
            let Some(w) = adapter.weight_for(module_path) else {
                continue; // untargeted by this adapter: leave this slot's zero-init in place
            };
            let r = w.a.shape()[1];
            let (a_values, b_values) = (
                w.a.as_f32().expect("a LoRA adapter weight is F32"),
                w.b.as_f32().expect("a LoRA adapter weight is F32"),
            );
            // w.a is [in_dim, r] row-major; pad into a_data's [in_dim, r_max] slot (extra columns 0).
            let a_base = slot * in_dim * r_max;
            for row in 0..in_dim {
                for c in 0..r {
                    a_data[a_base + row * r_max + c] = a_values[row * r + c];
                }
            }
            // w.b is [r, out_dim] row-major; pad into b_data's [r_max, out_dim] slot (extra rows stay 0).
            let b_base = slot * r_max * out_dim;
            for row in 0..r {
                for col in 0..out_dim {
                    b_data[b_base + row * out_dim + col] = b_values[row * out_dim + col];
                }
            }
            scaling[slot] = adapter.config.scaling();
        }
        Ok(Some(LoraStackedModule {
            a: HostTensor::f32(vec![slots, in_dim, r_max], a_data),
            b: HostTensor::f32(vec![slots, r_max, out_dim], b_data),
            scaling: HostTensor::f32(vec![slots], scaling),
            r: r_max,
        }))
    }
}

enum LoraSide {
    A,
    B,
}

/// Recognize a `lora_A`/`lora_B` state-dict key and return `(module_path, side)`, or `None` for
/// anything else (`lora_embedding_A/B`, a DoRA magnitude key, an unrelated tensor). `module_path`
/// has the `base_model.model.` prefix and any adapter-name segment stripped, so
/// `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight` and
/// `...q_proj.lora_A.default.weight` both yield `model.layers.0.self_attn.q_proj`.
fn parse_lora_key(key: &str) -> Option<(String, LoraSide)> {
    let stripped = key.strip_prefix("base_model.model.")?;
    let parts: Vec<&str> = stripped.split('.').collect();
    if parts.last() != Some(&"weight") {
        return None;
    }
    let idx = parts.iter().position(|&p| p == "lora_A" || p == "lora_B")?;
    let side = if parts[idx] == "lora_A" {
        LoraSide::A
    } else {
        LoraSide::B
    };
    Some((parts[..idx].join("."), side))
}

/// True if `key` is a DoRA magnitude-vector tensor, `base_model.model.{module}.lora_magnitude_vector[.<adapter>]`
/// (a `nn.Parameter` with no trailing `.weight`, one per DoRA-adapted module). Kept separate from
/// `parse_lora_key` so `from_safetensors` can tell "not a lora tensor, skip" from "DoRA present,
/// refuse the adapter".
fn is_dora_magnitude_key(key: &str) -> bool {
    match key.strip_prefix("base_model.model.") {
        Some(stripped) => stripped.split('.').any(|p| p == "lora_magnitude_vector"),
        None => false,
    }
}

/// Transpose a 2-D `(shape, data)` tensor `[r0, r1] -> [r1, r0]` (HF `[out,in]` -> poot's
/// `[in,out]`, as `SafeTensors::dequant_fp8` does for a base weight).
fn transpose_2d(t: &(Vec<usize>, Vec<f32>)) -> HostTensor {
    let (rows, cols) = (t.0[0], t.0[1]);
    let mut data = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = t.1[r * cols + c];
        }
    }
    HostTensor::f32(vec![cols, rows], data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal `adapter_model.safetensors` blob with one target module's `lora_A`/`lora_B` (r=2,
    /// in=4, out=3), keyed with the PEFT causal-LM prefix (`base_model.model.model.layers...`) and
    /// a `.default.` adapter-name segment, so the stripped and named forms are shown to parse alike.
    fn synthetic_adapter_safetensors() -> Vec<u8> {
        let (r, in_dim, out_dim) = (2usize, 4usize, 3usize);
        // lora_A [r,in] row-major, lora_B [out,r] row-major.
        let a_data: Vec<f32> = (0..r * in_dim).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..out_dim * r).map(|i| i as f32 * 0.1 + 1.0).collect();
        let mut data = Vec::new();
        let a_off = data.len();
        for v in &a_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let b_off = data.len();
        for v in &b_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight";
        let key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight";
        let header = serde_json::json!({
            key_a: {"dtype": "F32", "shape": [r, in_dim], "data_offsets": [a_off, b_off]},
            key_b: {"dtype": "F32", "shape": [out_dim, r], "data_offsets": [b_off, end]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&data);
        blob
    }

    #[test]
    fn parses_real_peft_causal_lm_key_shape() {
        let key = "base_model.model.model.layers.5.self_attn.v_proj.lora_A.weight";
        let (module, side) = parse_lora_key(key).expect("should parse");
        assert_eq!(module, "model.layers.5.self_attn.v_proj");
        assert!(matches!(side, LoraSide::A));
    }

    #[test]
    fn parses_named_adapter_segment() {
        let key = "base_model.model.model.layers.5.self_attn.v_proj.lora_B.default.weight";
        let (module, side) = parse_lora_key(key).expect("should parse");
        assert_eq!(module, "model.layers.5.self_attn.v_proj");
        assert!(matches!(side, LoraSide::B));
    }

    #[test]
    fn rejects_embedding_and_unrelated_keys() {
        assert!(parse_lora_key("base_model.model.model.embed_tokens.lora_embedding_A").is_none());
        assert!(parse_lora_key("model.layers.0.self_attn.q_proj.weight").is_none());
    }

    #[test]
    fn from_safetensors_transposes_and_indexes_by_module_path() {
        let (r, in_dim, out_dim) = (2usize, 4usize, 3usize);
        let store =
            crate::safetensors::load_weight_store_bytes(&synthetic_adapter_safetensors()).unwrap();
        let config = LoraAdapterConfig {
            r,
            lora_alpha: 8.0,
            target_modules: vec!["q_proj".into()],
            use_rslora: false,
        };
        let adapter = LoraAdapter::from_store(config, &store).unwrap();
        assert_eq!(adapter.config.scaling(), 4.0); // alpha/r = 8/2

        let w = adapter
            .weight_for("model.layers.0.self_attn.q_proj")
            .expect("q_proj adapter present");
        assert_eq!(w.a.shape(), vec![in_dim, r]);
        assert_eq!(w.b.shape(), vec![r, out_dim]);

        // spot-check the transpose: lora_A[r,in] row-major 0.0,0.1,0.2,0.3 / 0.4,0.5,0.6,0.7 ->
        // a[in,r] column-major read: a[0,0]=A[0,0]=0.0, a[0,1]=A[1,0]=0.4, a[1,0]=A[0,1]=0.1.
        assert!((w.a.as_f32().unwrap()[0] - 0.0).abs() < 1e-6);
        assert!((w.a.as_f32().unwrap()[1] - 0.4).abs() < 1e-6);
        assert!((w.a.as_f32().unwrap()[r] - 0.1).abs() < 1e-6);

        assert!(
            adapter
                .weight_for("model.layers.0.self_attn.k_proj")
                .is_none()
        );
    }

    #[test]
    fn rslora_scaling_divides_by_sqrt_r() {
        let config = LoraAdapterConfig {
            r: 16,
            lora_alpha: 16.0,
            target_modules: vec![],
            use_rslora: true,
        };
        assert!((config.scaling() - 4.0).abs() < 1e-6); // 16/sqrt(16) = 4
    }

    #[test]
    fn dora_magnitude_vector_key_is_rejected() {
        // A DoRA checkpoint has full lora_A/lora_B tensors plus a lora_magnitude_vector per adapted
        // module (PEFT's `layer.py::update_layer`). The loader must refuse it rather than drop the
        // magnitude renormalization.
        let (r, in_dim, out_dim, out_features) = (2usize, 4usize, 3usize, 3usize);
        let a_data: Vec<f32> = (0..r * in_dim).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..out_dim * r).map(|i| i as f32 * 0.1 + 1.0).collect();
        let mag_data: Vec<f32> = (0..out_features).map(|i| i as f32 * 0.01 + 1.0).collect();

        let mut data = Vec::new();
        let a_off = data.len();
        for v in &a_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let b_off = data.len();
        for v in &b_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let mag_off = data.len();
        for v in &mag_data {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();

        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let key_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";
        let key_mag =
            "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.default";
        let header = serde_json::json!({
            key_a: {"dtype": "F32", "shape": [r, in_dim], "data_offsets": [a_off, b_off]},
            key_b: {"dtype": "F32", "shape": [out_dim, r], "data_offsets": [b_off, mag_off]},
            key_mag: {"dtype": "F32", "shape": [out_features], "data_offsets": [mag_off, end]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&data);

        let store = crate::safetensors::load_weight_store_bytes(&blob).unwrap();
        let config = LoraAdapterConfig {
            r,
            lora_alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            use_rslora: false,
        };
        let err = LoraAdapter::from_store(config, &store).expect_err(
            "a DoRA checkpoint (lora_A/lora_B present PLUS lora_magnitude_vector) must error, \
                         not silently load without the magnitude renormalization",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("DoRA"),
            "error should name DoRA as the reason, got: {msg}"
        );
    }

    #[test]
    fn missing_lora_b_is_an_error() {
        let key_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
        let header = serde_json::json!({
            key_a: {"dtype": "F32", "shape": [2, 4], "data_offsets": [0, 32]},
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(&header_bytes);
        blob.extend_from_slice(&[0u8; 32]);
        let store = crate::safetensors::load_weight_store_bytes(&blob).unwrap();
        let config = LoraAdapterConfig {
            r: 2,
            lora_alpha: 4.0,
            target_modules: vec!["q_proj".into()],
            use_rslora: false,
        };
        assert!(LoraAdapter::from_store(config, &store).is_err());
    }

    /// Build a `LoraAdapter` directly, skipping the safetensors round-trip (the pool tests only
    /// need the transposed `[in,r]`/`[r,out]` shapes, covered by
    /// `from_safetensors_transposes_and_indexes_by_module_path`).
    fn make_adapter(
        r: usize,
        in_dim: usize,
        out_dim: usize,
        module: &str,
        seed: f32,
    ) -> LoraAdapter {
        let a: Vec<f32> = (0..in_dim * r).map(|i| seed + i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..r * out_dim)
            .map(|i| seed + 10.0 + i as f32 * 0.01)
            .collect();
        let mut weights = HashMap::new();
        weights.insert(
            module.to_string(),
            LoraWeight {
                a: HostTensor::f32(vec![in_dim, r], a),
                b: HostTensor::f32(vec![r, out_dim], b),
            },
        );
        LoraAdapter {
            config: LoraAdapterConfig {
                r,
                lora_alpha: (2 * r) as f32, // scaling() = alpha/r = 2.0, easy to check
                target_modules: vec![module.rsplit('.').next().unwrap().to_string()],
                use_rslora: false,
            },
            weights,
        }
    }

    #[test]
    fn pool_stacks_two_adapters_with_rank_padding_and_zero_sentinel() {
        let module = "model.layers.0.self_attn.q_proj";
        let (in_dim, out_dim) = (4usize, 3usize);
        let mut pool = LoraAdapterPool::new();
        let idx_small = pool.register("small_r", make_adapter(2, in_dim, out_dim, module, 1.0));
        let idx_big = pool.register("big_r", make_adapter(3, in_dim, out_dim, module, 100.0));
        assert_eq!(idx_small, 1);
        assert_eq!(idx_big, 2);
        assert_eq!(pool.index_of("small_r"), Some(1));
        assert_eq!(pool.len(), 2);

        let stacked = pool
            .stack_for_module(module)
            .unwrap()
            .expect("both adapters target this module");
        assert_eq!(stacked.r, 3); // padded up to the larger adapter's rank
        assert_eq!(stacked.a.shape(), vec![3, in_dim, 3]); // n_adapters(2) + 1 sentinel slot
        assert_eq!(stacked.b.shape(), vec![3, 3, out_dim]);
        assert_eq!(stacked.scaling.shape(), vec![3]);

        // slot 0 (NO_ADAPTER): all-zero A/B, scaling 0.
        assert_eq!(LoraAdapterPool::NO_ADAPTER, 0);
        let a_slot0 = &stacked.a.as_f32().unwrap()[0..in_dim * 3];
        assert!(a_slot0.iter().all(|&v| v == 0.0));
        let b_slot0 = &stacked.b.as_f32().unwrap()[0..3 * out_dim];
        assert!(b_slot0.iter().all(|&v| v == 0.0));
        assert_eq!(stacked.scaling.as_f32().unwrap()[0], 0.0);

        // slot 1 (small_r, r=2): padded rank column 2 (0-indexed) of every row must be exactly zero.
        let a_slot1_base = in_dim * 3; // slot 1 (small_r): 1 * in_dim * r_max
        for row in 0..in_dim {
            assert_eq!(
                stacked.a.as_f32().unwrap()[a_slot1_base + row * 3 + 2],
                0.0,
                "row {row} pad column"
            );
            // The unpadded columns must match the unstacked adapter's data exactly.
            let orig = make_adapter(2, in_dim, out_dim, module, 1.0);
            let w = orig.weight_for(module).unwrap();
            assert_eq!(
                stacked.a.as_f32().unwrap()[a_slot1_base + row * 3],
                w.a.as_f32().unwrap()[row * 2]
            );
            assert_eq!(
                stacked.a.as_f32().unwrap()[a_slot1_base + row * 3 + 1],
                w.a.as_f32().unwrap()[row * 2 + 1]
            );
        }
        // small_r's B padded row (rank index 2) must be all-zero.
        let b_slot1_base = 3 * out_dim; // slot 1 (small_r): 1 * r_max * out_dim
        for col in 0..out_dim {
            assert_eq!(
                stacked.b.as_f32().unwrap()[b_slot1_base + 2 * out_dim + col],
                0.0
            );
        }
        assert_eq!(stacked.scaling.as_f32().unwrap()[1], 2.0); // small_r: alpha=2*r=4, r=2 -> 4/2=2.0

        // slot 2 (big_r, r=3): fully populated, no padding.
        assert_eq!(stacked.scaling.as_f32().unwrap()[2], 2.0); // big_r: alpha=2*r=6, r=3 -> 6/3=2.0
    }

    #[test]
    fn stack_for_module_returns_none_when_no_adapter_targets_it() {
        let mut pool = LoraAdapterPool::new();
        pool.register(
            "a",
            make_adapter(2, 4, 3, "model.layers.0.self_attn.q_proj", 1.0),
        );
        assert!(
            pool.stack_for_module("model.layers.0.self_attn.k_proj")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn stack_for_module_errors_on_shape_mismatch() {
        let module = "model.layers.0.self_attn.q_proj";
        let mut pool = LoraAdapterPool::new();
        pool.register("a", make_adapter(2, 4, 3, module, 1.0));
        pool.register("b", make_adapter(2, 8, 3, module, 1.0)); // different in_dim
        let err = pool
            .stack_for_module(module)
            .expect_err("mismatched in/out dims across adapters should error");
        assert!(err.to_string().contains("disagree on shape"));
    }

    #[test]
    fn register_replacing_a_name_keeps_its_index() {
        let module = "model.layers.0.self_attn.q_proj";
        let mut pool = LoraAdapterPool::new();
        let first = pool.register("a", make_adapter(2, 4, 3, module, 1.0));
        let second = pool.register("a", make_adapter(2, 4, 3, module, 999.0));
        assert_eq!(
            first, second,
            "re-registering the same name keeps its pool index"
        );
        assert_eq!(pool.len(), 1, "replacing must not grow the pool");
    }

    #[test]
    fn untargeted_module_gets_a_zero_slot_not_an_error() {
        // Adapter "a" targets q_proj only, "b" targets v_proj only. Stacking for q_proj must
        // succeed with "b"'s slot all-zero (per-row analog of FR-008's zero-fill), not error.
        let q = "model.layers.0.self_attn.q_proj";
        let v = "model.layers.0.self_attn.v_proj";
        let mut pool = LoraAdapterPool::new();
        pool.register("a", make_adapter(2, 4, 3, q, 1.0));
        pool.register("b", make_adapter(2, 4, 3, v, 1.0));
        let stacked = pool.stack_for_module(q).unwrap().expect("a targets q_proj");
        assert_eq!(stacked.a.shape(), vec![3, 4, 2]);
        // slot 2 ("b", which does not target q_proj) must be all-zero.
        let b_slot2_a = &stacked.a.as_f32().unwrap()[2 * 4 * 2..3 * 4 * 2];
        assert!(b_slot2_a.iter().all(|&v| v == 0.0));
        assert_eq!(stacked.scaling.as_f32().unwrap()[2], 0.0);
    }

    /// Spec 248 hot-load: a placeholder (empty-weights) slot under a reserved name, as
    /// `poot-serve`'s `--lora-pool-capacity` provisions, is what [`LoraAdapterPool::hot_load`]
    /// takes over without growing the pool.
    #[test]
    fn hot_load_takes_over_a_free_placeholder_slot_without_growing_the_pool() {
        let module = "model.layers.0.self_attn.q_proj";
        let mut pool = LoraAdapterPool::new();
        pool.register("real", make_adapter(2, 4, 3, module, 1.0));
        // Reserved placeholder: empty weights, as poot-serve's startup provisioning.
        pool.register(
            "__reserved_0",
            LoraAdapter {
                config: LoraAdapterConfig {
                    r: 0,
                    lora_alpha: 0.0,
                    target_modules: Vec::new(),
                    use_rslora: false,
                },
                weights: HashMap::new(),
            },
        );
        assert_eq!(pool.len(), 2);

        let idx = pool
            .hot_load("new_style", make_adapter(2, 4, 3, module, 5.0))
            .expect("a free placeholder slot exists");
        assert_eq!(idx, 2, "took over the placeholder's own slot");
        assert_eq!(pool.len(), 2, "hot_load must never grow the pool");
        assert_eq!(pool.index_of("new_style"), Some(2));
        assert_eq!(
            pool.index_of("real"),
            Some(1),
            "an unrelated adapter's index must be completely unaffected"
        );
    }

    #[test]
    fn hot_load_errors_when_no_free_slot_or_name_already_registered() {
        let module = "model.layers.0.self_attn.q_proj";
        let mut pool = LoraAdapterPool::new();
        pool.register("real", make_adapter(2, 4, 3, module, 1.0));

        let err = pool
            .hot_load("brand_new", make_adapter(2, 4, 3, module, 2.0))
            .expect_err("no free (placeholder) slot exists - the pool is at capacity");
        assert!(err.to_string().contains("no free slot"));

        let err = pool
            .hot_load("real", make_adapter(2, 4, 3, module, 2.0))
            .expect_err("hot_load must refuse an already-registered name");
        assert!(err.to_string().contains("already registered"));
    }

    #[test]
    fn hot_unload_frees_the_name_and_reverts_the_slot_to_a_placeholder_index_stable() {
        let module = "model.layers.0.self_attn.q_proj";
        let mut pool = LoraAdapterPool::new();
        let idx_a = pool.register("a", make_adapter(2, 4, 3, module, 1.0));
        let idx_b = pool.register("b", make_adapter(2, 4, 3, module, 2.0));

        let (freed_idx, old_targets) = pool.hot_unload("a").expect("a is registered");
        assert_eq!(freed_idx, idx_a);
        assert_eq!(old_targets, vec!["q_proj".to_string()]);
        assert_eq!(pool.index_of("a"), None, "the name is no longer resolvable");
        assert_eq!(
            pool.index_of("b"),
            Some(idx_b),
            "the OTHER adapter's index must be completely unaffected by an unrelated unload"
        );
        assert_eq!(pool.len(), 2, "hot_unload must never shrink the pool");

        // The freed slot is a placeholder hot_load can take over without growing the pool.
        let idx_new = pool
            .hot_load("c", make_adapter(3, 4, 3, module, 9.0))
            .expect("the freed slot is available");
        assert_eq!(idx_new, freed_idx, "hot_load reuses the freed slot exactly");
        assert_eq!(pool.len(), 2);

        assert!(
            pool.hot_unload("never_registered").is_none(),
            "unloading a name that was never registered is a clean None, not a panic"
        );
    }

    #[test]
    fn pool_identity_is_unique_per_registration_and_changes_on_reload() {
        // Spec 248 paged prefix reuse: every live registration gets a distinct, never-reused effective-
        // weights identity, so cached K/V can never cross adapters or survive a hot reload.
        let module = "model.layers.0.self_attn.q_proj";
        let mut pool = LoraAdapterPool::new();
        assert_eq!(
            pool.identity_of(LoraAdapterPool::NO_ADAPTER),
            None,
            "the no-adapter sentinel has no adapter identity"
        );
        let idx_a = pool.register("a", make_adapter(2, 4, 3, module, 1.0));
        let id_a = pool
            .identity_of(idx_a)
            .expect("a registered adapter has an identity");
        assert_ne!(id_a, 0);
        let idx_b = pool.register("b", make_adapter(2, 4, 3, module, 2.0));
        let id_b = pool.identity_of(idx_b).unwrap();
        assert_ne!(id_a, id_b, "different adapters get different identities");

        // Replacing a name keeps its index but changes its identity (the weights changed).
        let replaced = pool.register("a", make_adapter(2, 4, 3, module, 3.0));
        assert_eq!(replaced, idx_a);
        let id_a2 = pool.identity_of(idx_a).unwrap();
        assert_ne!(
            id_a2, id_a,
            "replacement must change the effective-weights identity"
        );

        // hot_unload frees the slot's identity; hot_load into the freed index gets a fresh one.
        let (freed, _) = pool.hot_unload("b").unwrap();
        assert_eq!(freed, idx_b);
        assert_eq!(
            pool.identity_of(idx_b),
            None,
            "a freed slot has no adapter identity"
        );
        let idx_c = pool
            .hot_load("c", make_adapter(2, 4, 3, module, 9.0))
            .unwrap();
        assert_eq!(idx_c, idx_b, "hot_load reuses the freed pool index");
        let id_c = pool.identity_of(idx_c).unwrap();
        assert_ne!(
            id_c, id_b,
            "an adapter reloaded into the same index must not reuse the prior identity"
        );
        assert_ne!(id_c, 0);
    }
}

//! Graph builders/binders shared across backends: KV graphs, masks, PTX graph variants.

use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::{ComputedConst, Graph, Slot, Storage, ValueId};
#[cfg(test)]
use poot_models::deepseek3::{
    trace_deepseek3_decode_kv_masked_batched_shared_pool, trace_deepseek3_prefill_kv_shared_pool,
};
use poot_models::granite::trace_granite_prefill_kv;
#[cfg(test)]
use poot_models::granite::{
    trace_granite_decode_kv_masked_batched_shared_pool, trace_granite_prefill_kv_shared_pool,
};
#[cfg(test)]
use poot_models::mixtral::{
    trace_mixtral_decode_kv_masked_batched_shared_pool, trace_mixtral_prefill_kv_shared_pool,
};
use poot_models::qwen3moe::trace_qwen3_moe_prefill_kv;
#[cfg(test)]
use poot_models::qwen3moe::{
    trace_qwen3_moe_decode_kv_masked_batched_shared_pool, trace_qwen3_moe_prefill_kv_shared_pool,
};
use poot_tensor::HostTensor;

use crate::core::decode_arch::ContiguousPrefillArch;
#[cfg(test)]
use crate::core::decode_arch::SharedPoolArch;
use crate::core::runner::Runner;
use crate::error::{OptionExt, Result};

/// Materialize a `Storage::Computed` value: a constant the compiler computed from
/// its own definition (`fold_iota`'s output today), not a checkpoint or binder supply. Shared by every
/// Runner binder that walks a graph's `Storage` so the arm is written once.
pub(crate) fn computed_const_tensor(computed: ComputedConst) -> HostTensor {
    HostTensor::f32(computed.shape(), computed.values_f32())
}

impl Runner {
    /// Bind a prefill graph for the device-resident GPU path: the `[N]` token-id vector, the
    /// `mask.prefill` step input, and weights by name. K/V cache state is not bound here; the GPU seeds it with
    /// zero device buffers in `run_resident_prefill`.
    pub(crate) fn bind_prefill_kv(
        &self,
        g: &Graph,
        tokens: &[u32],
    ) -> Result<HashMap<ValueId, Value>> {
        let l = tokens.len();
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match meta.storage {
                Storage::Slot(Slot::Token) => token_slot(&meta.aval, tokens),
                // legalize (card 523a) hosts the embed gather off-device as this named `Slot::TokenEmbed`
                // input when the target's buffer limit needs it.
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
                        "mask.prefill" => prefill_causal_mask(l, self.sliding_window),
                        other => bail!("unexpected mask slot {other} in prefill-kv graph"),
                    }
                }
                // Card 550: the mask is a graph computation over `Slot::Pos` and `iota`; this one-shot
                // prefill always starts at position 0.
                Storage::Slot(Slot::Pos) => prefill_pos_rows(&meta.aval, l),
                Storage::Slot(other) => bail!("unexpected slot {other:?} in prefill-kv graph"),
                Storage::State => continue, // seeded as zero device buffers on the GPU
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
        Ok(inputs)
    }

    /// Bind a paged prefill graph: like [`Self::bind_prefill_kv`] but also binds
    /// the `Slot::SlotMap` input to the `[cap]` inverse map `inv` (i32, `inv[physical] = logical`, or -1).
    #[cfg(test)]
    pub(crate) fn bind_prefill_kv_paged(
        &self,
        g: &Graph,
        tokens: &[u32],
        inv: &[i32],
    ) -> Result<HashMap<ValueId, Value>> {
        let l = tokens.len();
        let mut inputs = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let t = match meta.storage {
                Storage::Slot(Slot::Token) => token_slot(&meta.aval, tokens),
                Storage::Slot(Slot::SlotMap) => {
                    HostTensor::i32(meta.aval.shape.clone(), inv.to_vec())
                }
                // Same host gather as `bind_prefill_kv`'s `Slot::TokenEmbed` arm: one row per prompt token in order.
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
                        "mask.prefill" => prefill_causal_mask(l, self.sliding_window),
                        other => bail!("unexpected mask slot {other} in paged prefill-kv graph"),
                    }
                }
                // Card 550: see `bind_prefill_kv`.
                Storage::Slot(Slot::Pos) => prefill_pos_rows(&meta.aval, l),
                Storage::Slot(other) => {
                    bail!("unexpected slot {other:?} in paged prefill-kv graph")
                }
                Storage::State => continue, // seeded as zero device buffers on the GPU
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
        Ok(inputs)
    }

    /// The Const (weight) inputs of a graph, bound by name from the loaded weights.
    ///
    /// `pub(crate)` (card 549 fallout): its last production caller was the pre-contract
    /// `PtxGraphExecutor::capture_decode`'s `consts` map; PTX (card 549) and ROCm (card 548) both
    /// moved onto `Engine::load_weights`, which binds consts a different way, so this is now a
    /// same-crate unit-test-only helper (`core::runner::tests::gptoss_load_tests`). The two
    /// `poot-llm/tests/` integration diagnostics that used to call this through `Runner` (a
    /// separate crate target, so `pub(crate)` is invisible to them) now bind the same way inline,
    /// through [`Runner::weight_value`].
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "unit-test-only since cards 548/549 (see doc above)"
        )
    )]
    pub(crate) fn const_inputs(&self, g: &Graph) -> Result<HashMap<ValueId, Value>> {
        let mut consts = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            if let Storage::Computed(computed) = meta.storage {
                // A folded `iota`: materialize it from its definition (no name, no weights map).
                consts.insert(id, computed_const_tensor(computed).into());
                continue;
            }
            if meta.storage == Storage::Const {
                let name = meta.name.as_deref().context("const without a name")?;
                consts.insert(id, self.weight_value(name, &meta.aval)?);
            }
        }
        Ok(consts)
    }

    /// The arch's batched (Q=N) KV-writing prefill graph (raw, unfused). Supported architectures carry state
    /// pairs matching the decode graph, so the filled `[1,Hkv,cap,D]` caches drop straight into the decode loop.
    /// Architectures without a contiguous prefill tracer return a capability error.
    pub(crate) fn prefill_kv_graph(&self, n: usize, cap: usize) -> Result<Graph> {
        let graph = match self.contiguous_prefill_arch()? {
            ContiguousPrefillArch::GraniteMoe => trace_granite_prefill_kv(
                self.cfg,
                self.granite_moe
                    .expect("ContiguousPrefillArch::GraniteMoe implies granite params"),
                n,
                cap,
            ),
            ContiguousPrefillArch::Qwen3Moe => trace_qwen3_moe_prefill_kv(
                self.cfg,
                self.qwen3_moe
                    .as_ref()
                    .expect("ContiguousPrefillArch::Qwen3Moe implies qwen3_moe params")
                    .clone(),
                n,
                cap,
            ),
        };
        self.bind_storage(graph)
    }

    /// The arch's shared-pool (paged, engine-serving) KV-writing prefill graph: the shared-pool analogue of
    /// [`Self::prefill_kv_graph`], for the `[pool,Hkv,D]` cache layout. granitemoe and qwen3_moe route to
    /// [`poot_models::moe_prefill::trace_moe_prefill_kv_shared_pool`]'s per-arch wrappers; the families
    /// without one are rejected by the shared classifier before tracing.
    ///
    /// Mixtral routes to [`poot_models::mixtral::trace_mixtral_prefill_kv_shared_pool`] (dense `[E,...]`-stacked
    /// weights, as its batched decode tracer). It is checked as `self.mixtral.is_some()` directly rather than
    /// folded into `is_moe()`; see [`Self::decode_kv_graph_shared_pool`].
    ///
    /// Dense (non-MoE-pooled) DeepSeek-V3 routes to
    /// [`poot_models::deepseek3::trace_deepseek3_prefill_kv_shared_pool`] (single-sequence MLA shared-KV-pool
    /// prefill addressed by the inverse `Slot::SlotMap`), checked as `self.deepseek3.is_some()` for the same
    /// reason. The support matrix lives in [`Runner::shared_pool_arch`]; unsupported families return an error
    /// before tracing.
    #[cfg(test)]
    pub(crate) fn prefill_kv_graph_shared_pool(&self, n: usize, pool: usize) -> Result<Graph> {
        let graph = match self.shared_pool_arch()? {
            SharedPoolArch::GraniteMoe => trace_granite_prefill_kv_shared_pool(
                self.cfg,
                self.granite_moe
                    .expect("SharedPoolArch::GraniteMoe implies granite params"),
                n,
                pool,
            ),
            SharedPoolArch::Qwen3Moe => trace_qwen3_moe_prefill_kv_shared_pool(
                self.cfg,
                self.qwen3_moe
                    .as_ref()
                    .expect("SharedPoolArch::Qwen3Moe implies qwen3_moe params")
                    .clone(),
                n,
                pool,
            ),
            SharedPoolArch::Mixtral => trace_mixtral_prefill_kv_shared_pool(
                self.cfg,
                self.mixtral
                    .expect("SharedPoolArch::Mixtral implies mixtral params"),
                n,
                pool,
            ),
            SharedPoolArch::DeepseekV3 => {
                let params = self
                    .deepseek3
                    .expect("SharedPoolArch::DeepseekV3 implies deepseek3 params");
                trace_deepseek3_prefill_kv_shared_pool(params.cfg, params.moe, n, pool)
            }
        };
        self.bind_storage(graph)
    }

    /// The arch's batched shared-pool decode graph: the decode-side sibling of
    /// [`Self::prefill_kv_graph_shared_pool`], for the per-iteration batched decode step of
    /// [`Self::trace_batched_shared_pool_decode`]. granitemoe and qwen3_moe route to
    /// [`poot_models::moe_decode::trace_moe_decode_kv_masked_batched_shared_pool`]'s per-arch wrappers. No
    /// tracer here has a `quant_kv` knob; the caller rejects `quant_kv` before reaching here.
    ///
    /// Mixtral routes to [`poot_models::mixtral::trace_mixtral_decode_kv_masked_batched_shared_pool`], the dense
    /// `[E,...]`-stacked-weights path (not the expert-pool one). Mixtral is not part of `Self::is_moe()`, whose
    /// other callers (poot-serve's engine-selection gate and `BatchDecodable` capability reporting) assume
    /// "is_moe" implies a shared-pool prefill tracer, so the caller checks `self.mixtral.is_some()` directly.
    ///
    /// Dense (non-MoE-pooled) DeepSeek-V3 routes to
    /// [`poot_models::deepseek3::trace_deepseek3_decode_kv_masked_batched_shared_pool`] (batched,
    /// `Slot::SlotMap`-addressed MLA shared-KV-pool decode), checked directly for the same reason. The support
    /// matrix lives in [`Runner::shared_pool_arch`], so unsupported families are refused before tracing.
    #[cfg(test)]
    pub(crate) fn decode_kv_graph_shared_pool(
        &self,
        cap: usize,
        batch: usize,
        pool_slots: usize,
        _quant_kv: bool,
    ) -> Result<Graph> {
        let graph = match self.shared_pool_arch()? {
            SharedPoolArch::GraniteMoe => trace_granite_decode_kv_masked_batched_shared_pool(
                self.cfg,
                self.granite_moe
                    .expect("SharedPoolArch::GraniteMoe implies granite params"),
                cap,
                batch,
                pool_slots,
            ),
            SharedPoolArch::Qwen3Moe => trace_qwen3_moe_decode_kv_masked_batched_shared_pool(
                self.cfg,
                self.qwen3_moe
                    .as_ref()
                    .expect("SharedPoolArch::Qwen3Moe implies qwen3_moe params")
                    .clone(),
                cap,
                batch,
                pool_slots,
            ),
            SharedPoolArch::Mixtral => trace_mixtral_decode_kv_masked_batched_shared_pool(
                self.cfg,
                self.mixtral
                    .expect("SharedPoolArch::Mixtral implies mixtral params"),
                cap,
                batch,
                pool_slots,
            ),
            SharedPoolArch::DeepseekV3 => {
                let params = self
                    .deepseek3
                    .expect("SharedPoolArch::DeepseekV3 implies deepseek3 params");
                trace_deepseek3_decode_kv_masked_batched_shared_pool(
                    params.cfg, params.moe, cap, batch, pool_slots,
                )
            }
        };
        self.bind_storage(graph)
    }

    /// Is this `Runner`'s loaded arch a routed-expert (MoE) model with a shared-pool prefill tracer:
    /// granitemoe (every layer routed) or qwen3_moe (a per-layer dense/routed switch, so `Some` means at
    /// least one layer routes). The ROCm/PTX batched-prefill entry points use it to keep MoE off the
    /// flash/tiled-GEMM path (see [`Self::generate_kv_rocm_prefilled_tokens`]).
    pub fn is_moe(&self) -> bool {
        self.granite_moe.is_some() || self.qwen3_moe.is_some()
    }

    /// Is this `Runner`'s loaded arch Mixtral? Separate from [`Self::is_moe`] (see
    /// [`Self::prefill_kv_graph_shared_pool`]), but `pub` for the same reason: `batch.rs`'s
    /// `impl BatchDecodable for Runner` needs a cross-crate check of the private `self.mixtral` to keep
    /// `supports_chunked_prefill()`/`supports_speculative_batched()` correct, since
    /// [`Self::prefill_kv_graph_shared_pool`]/[`Self::decode_kv_graph_shared_pool`] admit it directly.
    pub fn is_mixtral(&self) -> bool {
        self.mixtral.is_some()
    }

    /// Is this `Runner`'s loaded arch dense (non-MoE-pooled) DeepSeek-V3? Separate from [`Self::is_moe`], as
    /// with [`Self::is_mixtral`]: DeepSeek-V3 has no shared-pool prefill tracer, so folding it into `is_moe()`
    /// would wrongly imply one. `pub` because `batch.rs`'s `impl BatchDecodable for Runner` and `poot-serve`'s
    /// `main.rs` spawn gate check it. `false` for `self.deepseek2` (non-V3-style DeepSeek-V2), which has no
    /// batched tracer.
    pub fn is_deepseek3(&self) -> bool {
        self.deepseek3.is_some()
    }
}

/// Additive causal mask `[1,1,L,L]`: 0 where key j is visible to query i, -1e30 elsewhere.
/// `window = None`: full causal, visible iff `j <= i`.
/// `window = Some(w)`: sliding window, visible iff `j <= i && i - j < w`.
/// `window = None` and `window = Some(w)` with `w >= L` produce byte-identical outputs.
/// The host value of a `Slot::Token` input: the ids as the slot declares them - authoritative I32
/// words for an I32 slot (the storage-aware CPU oracle refuses an f32-only I32 input), f32 otherwise.
pub(crate) fn token_slot(aval: &poot_graph_ir::TensorType, ids: &[u32]) -> HostTensor {
    match aval.dtype {
        poot_tensor::DType::I32 => {
            HostTensor::i32(aval.shape.clone(), ids.iter().map(|&t| t as i32).collect())
        }
        _ => HostTensor::f32(aval.shape.clone(), ids.iter().map(|&t| t as f32).collect()),
    }
}

/// `Slot::Pos` for a one-shot prefill that always starts at position 0 (card 550): the
/// `[rows, tokens]` I32 absolute positions `[0, 1, .., tokens-1]`, repeated per row (`rows` is always 1
/// for every dense-family one-shot prefill tracer today). `l` is the caller's prompt/chunk length,
/// asserted to match the slot's declared `tokens` axis.
pub(crate) fn prefill_pos_rows(aval: &poot_graph_ir::TensorType, l: usize) -> HostTensor {
    let (rows, tokens) = (aval.shape[0], aval.shape[1]);
    debug_assert_eq!(
        tokens, l,
        "Slot::Pos token axis must match the prompt length"
    );
    let row: Vec<i32> = (0..tokens as i32).collect();
    let mut data = Vec::with_capacity(rows * tokens);
    for _ in 0..rows {
        data.extend_from_slice(&row);
    }
    HostTensor::i32(aval.shape.clone(), data)
}

pub(crate) fn prefill_causal_mask(l: usize, window: Option<usize>) -> HostTensor {
    let mut data = vec![-1e30f32; l * l];
    for i in 0..l {
        let lo = window.map_or(0, |w| i.saturating_sub(w - 1));
        for j in lo..=i {
            data[i * l + j] = 0.0;
        }
    }
    HostTensor::f32(vec![1, 1, l, l], data)
}

/// Additive decode mask over a KV cache of `cap` slots for a single query at `pos`. Returns a
/// `cap`-element row: 0 for visible slots, -1e9 for masked slots.
/// `window = None`: full causal, visible iff `t <= pos`.
/// `window = Some(w)`: sliding window, visible iff `t <= pos && pos - t < w`.
pub(crate) fn decode_mask_row(cap: usize, pos: usize, window: Option<usize>) -> Vec<f32> {
    (0..cap)
        .map(|t| {
            // short-circuit: `pos - t` is only computed when `t <= pos` (no underflow).
            let visible = t <= pos && window.is_none_or(|w| pos - t < w);
            if visible { 0.0 } else { -1.0e9 }
        })
        .collect()
}

/// Additive causal mask `[1,1,L,L]`: 0 where key j <= query i, large-negative above. Legacy alias used by
/// VLM paths; new code should use `prefill_causal_mask(l, None)`.
pub(crate) fn causal_mask(l: usize) -> HostTensor {
    prefill_causal_mask(l, None)
}

#[cfg(test)]
mod mask_tests {
    use super::{decode_mask_row, prefill_causal_mask};

    /// window=None must equal the classic full-causal mask (0 below diagonal, -1e30 above).
    #[test]
    fn prefill_window_none_equals_full_causal() {
        let l = 4;
        let got = prefill_causal_mask(l, None);
        // reference: hand-built lower-triangular additive mask
        let mut expected = vec![-1e30f32; l * l];
        for i in 0..l {
            for j in 0..=i {
                expected[i * l + j] = 0.0;
            }
        }
        assert_eq!(got.shape(), vec![1, 1, l, l]);
        assert_eq!(
            got.as_f32().unwrap(),
            &expected[..],
            "window=None must match full-causal"
        );
    }

    /// window=Some(w) zeros exactly the in-window band; the rest is -1e30.
    #[test]
    fn prefill_window_some_zeroes_correct_band() {
        let l = 5;
        let w = 2;
        let got = prefill_causal_mask(l, Some(w));
        // for each query row i, visible keys are max(0, i-w+1)..=i
        let mut expected = vec![-1e30f32; l * l];
        for i in 0..l {
            let lo = i.saturating_sub(w - 1);
            for j in lo..=i {
                expected[i * l + j] = 0.0;
            }
        }
        assert_eq!(
            got.as_f32().unwrap(),
            &expected[..],
            "window=Some({w}) must match band mask"
        );
    }

    /// w=1 means only self-attention is visible (diagonal only).
    #[test]
    fn prefill_window_1_is_diagonal_only() {
        let l = 4;
        let got = prefill_causal_mask(l, Some(1));
        let mut expected = vec![-1e30f32; l * l];
        for i in 0..l {
            expected[i * l + i] = 0.0; // only self
        }
        assert_eq!(
            got.as_f32().unwrap(),
            &expected[..],
            "window=1 must be diagonal only"
        );
    }

    /// w >= L must be byte-identical to window=None (full causal).
    #[test]
    fn prefill_window_ge_l_equals_full_causal() {
        let l = 4;
        let full = prefill_causal_mask(l, None);
        let wide = prefill_causal_mask(l, Some(l));
        let wider = prefill_causal_mask(l, Some(l + 10));
        assert_eq!(
            full.as_f32().unwrap(),
            wide.as_f32().unwrap(),
            "window=L must equal full-causal"
        );
        assert_eq!(
            full.as_f32().unwrap(),
            wider.as_f32().unwrap(),
            "window>L must equal full-causal"
        );
    }

    /// decode_mask_row with window=None: 0 for t<=pos, -1e9 for t>pos.
    #[test]
    fn decode_row_window_none_equals_full_causal() {
        let cap = 6;
        let pos = 3;
        let got = decode_mask_row(cap, pos, None);
        let expected: Vec<f32> = (0..cap)
            .map(|t| if t <= pos { 0.0 } else { -1.0e9 })
            .collect();
        assert_eq!(got, expected, "decode window=None must match full-causal");
    }

    /// decode_mask_row with window=Some(w): 0 for pos-w < t <= pos, -1e9 elsewhere.
    #[test]
    fn decode_row_window_some_zeroes_correct_range() {
        let cap = 8;
        let pos = 5;
        let w = 3;
        let got = decode_mask_row(cap, pos, Some(w));
        // visible: t in [pos-w+1, pos] = [3, 5]
        let expected: Vec<f32> = (0..cap)
            .map(|t| if t <= pos && pos - t < w { 0.0 } else { -1.0e9 })
            .collect();
        assert_eq!(
            got, expected,
            "decode window=Some({w}) must match windowed range"
        );
        // slots 3, 4, 5 visible; 0, 1, 2 masked (too old); 6, 7 masked (future)
        assert_eq!(
            &got[..3],
            &[-1.0e9f32, -1.0e9, -1.0e9],
            "too-old keys masked"
        );
        assert_eq!(&got[3..6], &[0.0f32, 0.0, 0.0], "in-window keys visible");
        assert_eq!(&got[6..], &[-1.0e9f32, -1.0e9], "future keys masked");
    }

    /// decode_mask_row w=1: only self-slot visible.
    #[test]
    fn decode_row_window_1_only_self_visible() {
        let cap = 5;
        let pos = 3;
        let got = decode_mask_row(cap, pos, Some(1));
        let mut expected = vec![-1.0e9f32; cap];
        expected[pos] = 0.0;
        assert_eq!(got, expected, "decode window=1 must be self-only");
    }

    /// decode_mask_row w >= cap: same as window=None.
    #[test]
    fn decode_row_window_ge_cap_equals_full_causal() {
        let cap = 6;
        let pos = 4;
        let full = decode_mask_row(cap, pos, None);
        let wide = decode_mask_row(cap, pos, Some(cap));
        let wider = decode_mask_row(cap, pos, Some(cap + 10));
        assert_eq!(full, wide, "decode window=cap must equal full-causal");
        assert_eq!(full, wider, "decode window>cap must equal full-causal");
    }
}

/// `Self::prefill_kv_graph_shared_pool` must route granitemoe/qwen3_moe to the MoE shared-pool tracer
/// instead of another family's. Host-side structural checks only
/// (graph shape, no binding or device), using a minimal `Runner` built by direct struct construction as in
/// `decode_arch::fixtures::runner_for`, so no checkpoint is needed.
#[cfg(test)]
mod shared_pool_prefill_routing_tests {
    use super::*;
    use crate::core::runner::LoraHotState;
    use poot_models::deepseek3::DeepseekV3Params;
    use poot_models::granite::{GraniteParams, MoeShape};
    use poot_models::mixtral::MixtralParams;
    use poot_models::qwen3moe::Qwen3MoeParams;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    fn runner_with(
        granite: Option<GraniteParams>,
        qwen3_moe: Option<Qwen3MoeParams>,
        mixtral: Option<MixtralParams>,
        deepseek3: Option<DeepseekV3Params>,
    ) -> Runner {
        let model = WordLevel::builder()
            .vocab(std::collections::HashMap::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build empty wordlevel model");
        Runner {
            cfg: poot_models::qwen2::Qwen2Config {
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
                qkv_bias: false,
                qk_norm: true,
                ..Default::default()
            },
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
            granite_moe: granite,
            qwen3_moe,
            mixtral,
            olmoe: None,
            gpt_oss: None,
            deepseek2: None,
            deepseek3,
            deepseek32: None,
            nemotron_h: None,
            formats: Default::default(),
            sliding_window: None,
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        }
    }

    #[test]
    fn routes_granitemoe_to_the_moe_shared_pool_tracer() {
        let gp = GraniteParams {
            moe: Some(MoeShape {
                n_experts: 4,
                top_k: 2,
                inter: 12,
            }),
            embed_mult: 1.0,
            attn_mult: 1.0,
            residual_mult: 1.0,
            logits_scale: 1.0,
        };
        let runner = runner_with(Some(gp), None, None, None);
        let (n, pool) = (5, 20);
        let g = runner
            .prefill_kv_graph_shared_pool(n, pool)
            .expect("GraniteMoE supports shared-pool prefill");
        g.validate()
            .expect("granitemoe shared-pool prefill graph should validate");
        assert_eq!(g.state.len(), 2 * runner.cfg.layers);
        for &(si, _) in &g.state {
            assert_eq!(
                g.aval(si).shape,
                vec![pool, runner.cfg.n_kv_heads, runner.cfg.head_dim],
                "granitemoe shared-pool cache state must use the [pool,Hkv,D] layout"
            );
        }
    }

    #[test]
    fn routes_qwen3_moe_to_the_moe_shared_pool_tracer() {
        let mp = Qwen3MoeParams {
            n_experts: 4,
            top_k: 2,
            inter: 12,
            sparse_layer: vec![true, false],
        };
        let runner = runner_with(None, Some(mp), None, None);
        let (n, pool) = (5, 20);
        let g = runner
            .prefill_kv_graph_shared_pool(n, pool)
            .expect("Qwen3-MoE supports shared-pool prefill");
        g.validate()
            .expect("qwen3-moe shared-pool prefill graph should validate");
        assert_eq!(g.state.len(), 2 * runner.cfg.layers);
        for &(si, _) in &g.state {
            assert_eq!(
                g.aval(si).shape,
                vec![pool, runner.cfg.n_kv_heads, runner.cfg.head_dim],
                "qwen3-moe shared-pool cache state must use the [pool,Hkv,D] layout"
            );
        }
    }

    /// Mixtral routes to its own dense (non-pooled) batched shared-pool prefill tracer, checked as
    /// `self.mixtral.is_some()` directly. Prefill-side twin of
    /// `shared_pool_decode_routing_tests::routes_mixtral_to_its_own_dense_shared_pool_decode_tracer`.
    #[test]
    fn routes_mixtral_to_its_own_dense_shared_pool_prefill_tracer() {
        let mp = MixtralParams {
            n_experts: 4,
            top_k: 2,
            inter: 12,
        };
        let runner = runner_with(None, None, Some(mp), None);
        let (n, pool) = (5, 20);
        let g = runner
            .prefill_kv_graph_shared_pool(n, pool)
            .expect("Mixtral supports shared-pool prefill");
        g.validate()
            .expect("mixtral shared-pool prefill graph should validate");
        assert_eq!(g.state.len(), 2 * runner.cfg.layers);
        for &(si, _) in &g.state {
            assert_eq!(
                g.aval(si).shape,
                vec![pool, runner.cfg.n_kv_heads, runner.cfg.head_dim],
                "mixtral shared-pool cache state must use the [pool,Hkv,D] layout"
            );
        }
    }

    /// Dense (non-MoE-pooled) DeepSeek-V3 routes to its own MLA shared-pool prefill tracer, checked as
    /// `self.deepseek3.is_some()` directly. Prefill-side twin of `Self::decode_kv_graph_shared_pool`'s
    /// `self.deepseek3` arm.
    #[test]
    fn routes_deepseek3_to_its_own_dense_shared_pool_prefill_tracer() {
        use poot_models::deepseek2::DeepseekV2Config;
        use poot_models::deepseek3::DeepseekV3MoeParams;

        let dcfg = DeepseekV2Config {
            vocab: 32,
            hidden: 16,
            layers: 2,
            n_heads: 4,
            q_lora_rank: Some(8),
            kv_lora_rank: 8,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 4,
            eps: 1e-6,
            max_pos: 32,
            rope_theta: 10_000.0,
            yarn: None,
        };
        let dmp = DeepseekV3MoeParams {
            n_routed_experts: 4,
            top_k: 2,
            moe_inter: 8,
            n_shared_experts: 1,
            dense_inter: 12,
            first_k_dense_replace: 1,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
        };
        let runner = runner_with(
            None,
            None,
            None,
            Some(DeepseekV3Params {
                cfg: dcfg,
                moe: dmp,
            }),
        );
        let (n, pool) = (5, 20);
        let g = runner
            .prefill_kv_graph_shared_pool(n, pool)
            .expect("DeepSeek-V3 supports shared-pool prefill");
        g.validate()
            .expect("deepseek3 shared-pool prefill graph should validate");
        assert_eq!(g.state.len(), 2 * dcfg.layers);
        // one `mla.c_cache` [pool,1,kv_lora_rank] and one `mla.rope_cache` [pool,1,qk_rope_head_dim] per layer;
        // both share the [pool,1,D] leading shape, only D differs.
        for &(si, _) in &g.state {
            let shape = g.aval(si).shape.clone();
            assert_eq!(
                shape.len(),
                3,
                "deepseek3 shared-pool cache state must be rank 3"
            );
            assert_eq!(
                shape[0], pool,
                "deepseek3 shared-pool cache state must use the [pool,1,D] layout"
            );
            assert_eq!(
                shape[1], 1,
                "deepseek3 shared-pool cache state must use the [pool,1,D] layout"
            );
        }
    }
}

/// `Self::decode_kv_graph_shared_pool` must route granitemoe/qwen3_moe to the MoE batched shared-pool
/// decode tracer instead of another family's. Mirrors
/// `shared_pool_prefill_routing_tests` above (same `runner_with` construction, host-side structural checks).
#[cfg(test)]
mod shared_pool_decode_routing_tests {
    use super::*;
    use crate::core::runner::LoraHotState;
    use poot_models::granite::{GraniteParams, MoeShape};
    use poot_models::mixtral::MixtralParams;
    use poot_models::qwen3moe::Qwen3MoeParams;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    fn runner_with(
        granite: Option<GraniteParams>,
        qwen3_moe: Option<Qwen3MoeParams>,
        mixtral: Option<MixtralParams>,
    ) -> Runner {
        let model = WordLevel::builder()
            .vocab(std::collections::HashMap::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build empty wordlevel model");
        Runner {
            cfg: poot_models::qwen2::Qwen2Config {
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
                qkv_bias: false,
                qk_norm: true,
                ..Default::default()
            },
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
            granite_moe: granite,
            qwen3_moe,
            mixtral,
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

    #[test]
    fn routes_granitemoe_to_the_moe_shared_pool_decode_tracer() {
        let gp = GraniteParams {
            moe: Some(MoeShape {
                n_experts: 4,
                top_k: 2,
                inter: 12,
            }),
            embed_mult: 1.0,
            attn_mult: 1.0,
            residual_mult: 1.0,
            logits_scale: 1.0,
        };
        let runner = runner_with(Some(gp), None, None);
        let (cap, batch, pool) = (6, 3, 12);
        let g = runner
            .decode_kv_graph_shared_pool(cap, batch, pool, false)
            .expect("GraniteMoE supports shared-pool decode");
        g.validate()
            .expect("granitemoe shared-pool decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![batch, 1, runner.cfg.vocab]);
        assert_eq!(g.state.len(), 2 * runner.cfg.layers);
        for &(si, _) in &g.state {
            assert_eq!(
                g.aval(si).shape,
                vec![pool, runner.cfg.n_kv_heads, runner.cfg.head_dim],
                "granitemoe shared-pool decode cache state must use the [pool,Hkv,D] layout"
            );
        }
    }

    #[test]
    fn routes_qwen3_moe_to_the_moe_shared_pool_decode_tracer() {
        let mp = Qwen3MoeParams {
            n_experts: 4,
            top_k: 2,
            inter: 12,
            sparse_layer: vec![true, false],
        };
        let runner = runner_with(None, Some(mp), None);
        let (cap, batch, pool) = (6, 3, 12);
        let g = runner
            .decode_kv_graph_shared_pool(cap, batch, pool, false)
            .expect("Qwen3-MoE supports shared-pool decode");
        g.validate()
            .expect("qwen3-moe shared-pool decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![batch, 1, runner.cfg.vocab]);
        assert_eq!(g.state.len(), 2 * runner.cfg.layers);
        for &(si, _) in &g.state {
            assert_eq!(
                g.aval(si).shape,
                vec![pool, runner.cfg.n_kv_heads, runner.cfg.head_dim],
                "qwen3-moe shared-pool decode cache state must use the [pool,Hkv,D] layout"
            );
        }
    }

    /// Mixtral routes to its own dense (non-pooled) batched shared-pool decode tracer, checked as
    /// `self.mixtral.is_some()` directly (Mixtral is not part of `Runner::is_moe()`; see
    /// `Runner::decode_kv_graph_shared_pool`).
    #[test]
    fn routes_mixtral_to_its_own_dense_shared_pool_decode_tracer() {
        let mp = poot_models::mixtral::MixtralParams {
            n_experts: 4,
            top_k: 2,
            inter: 12,
        };
        let runner = runner_with(None, None, Some(mp));
        let (cap, batch, pool) = (6, 3, 12);
        let g = runner
            .decode_kv_graph_shared_pool(cap, batch, pool, false)
            .expect("Mixtral supports shared-pool decode");
        g.validate()
            .expect("mixtral shared-pool decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![batch, 1, runner.cfg.vocab]);
        assert_eq!(g.state.len(), 2 * runner.cfg.layers);
        for &(si, _) in &g.state {
            assert_eq!(
                g.aval(si).shape,
                vec![pool, runner.cfg.n_kv_heads, runner.cfg.head_dim],
                "mixtral shared-pool decode cache state must use the [pool,Hkv,D] layout"
            );
        }
    }
}

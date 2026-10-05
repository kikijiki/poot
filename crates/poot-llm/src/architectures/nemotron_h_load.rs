//! Nemotron-H checkpoint loader (spec 279): a real-name-aware per-layer safetensors weight loader for
//! `poot_models::nemotron_h`, following `bloom_load.rs`/`mpt_load.rs` (a dedicated config struct + an
//! `is_*` detection predicate checked before the shared `Qwen2HfConfig` parse). Nemotron-H's
//! `config.json` (`hybrid_override_pattern`, `mamba_num_heads`, no `rope_theta`, NoPE) has no
//! correspondence to `Qwen2HfConfig`'s required fields, hence the dedicated struct.
//!
//! Tensor names/shapes were verified against `nvidia/Nemotron-H-8B-Base-8K`: `config.json` and
//! `model.safetensors.index.json` in full, plus a byte-range fetch of shard 1's safetensors header (8-byte
//! length prefix + ~9.8KB JSON) to confirm per-tensor shapes without downloading weights.
//! [`build_nemotron_h_weights`] reads those names and produces the tracer-facing constant names below,
//! including the three top-level tensors `poot_models::nemotron_h::trace_nemotron_h_prefill` needs
//! (`backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight`).
//!
//! `Runner::load_impl`/`Runner::load_nemotron_h_impl` call [`is_nemotron_h`]/
//! [`nemotron_h_config_from_json`]/[`build_nemotron_h_weights`], and `trace_nemotron_h_prefill` is wired
//! into `Runner::stateless_prefill_graph` (see the `nemotron_h` field doc on `Runner`), so a Nemotron-H
//! safetensors checkpoint loads and runs through `Runner::load` +
//! `Runner::generate`/`generate_sampled` (CPU re-prefill, spec 279's scope). `Runner::generate_kv_masked`
//! uses `poot_models::nemotron_h::trace_nemotron_h_decode` (whole-model, fixed-KV-capacity, traced once
//! and replayed per token; O(1) per step vs re-prefill's O(n)). Not wired: GPU dispatch beyond
//! re-prefill, and batched/pooled decode (see `specs/279-nemotron-h/spec.md`, Out-of-scope).
//!
//! **Real tensor names** (`nvidia/Nemotron-H-8B-Base-8K`): `backbone.embeddings.weight`
//! `[vocab,hidden]`; per layer `backbone.layers.{i}.norm.weight` `[hidden]` (the pre-mixer norm, on
//! every layer); and, under one shared `mixer` namespace for all layer kinds (the kind comes from
//! `hybrid_override_pattern`, not from which mixer tensors are present):
//! - Mamba: `mixer.A_log` `[Hq]`, `mixer.D` `[Hq]`, `mixer.conv1d.weight` `[ConvC,1,K]` (an
//!   `nn.Conv1d(groups=ConvC)` depthwise weight, `[10240,1,4]` on the 8B layer 0; not the tracer's
//!   `[K,ConvC]` convention), `mixer.conv1d.bias` `[ConvC]`, `mixer.dt_bias` `[Hq]`,
//!   `mixer.in_proj.weight` `[Inner+ConvC+Hq,hidden]` (one fused HF `[out,in]` tensor, `[18560,4096]` on
//!   the 8B; z/xBC/dt are contiguous row ranges: `z=[0,Inner)`, `xBC=[Inner,Inner+ConvC)`,
//!   `dt=[Inner+ConvC,Inner+ConvC+Hq)`), `mixer.norm.weight` `[Inner]` (the gated-RMSNorm weight, a
//!   second norm distinct from the per-layer `norm.weight`; `[8192]`), `mixer.out_proj.weight`
//!   `[hidden,Inner]` (`[4096,8192]`).
//!   - `A_log` needs a load-time transform: `mamba_ssm` uses `self.A = -torch.exp(self.A_log.float())`,
//!     and `nemotron_h_mamba_layer`'s `a_param` is used directly as `exp(Delta*A)`'s `A`
//!     (`poot_graph_ir::ops::mamba2_ssd_decode`), so this loader computes `a_param = -exp(A_log)` on the
//!     host, like `poot_load::SafeTensors::dequant_fp8`/`dequant_gptq`.
//! - Attention: `mixer.q_proj.weight` `[Aq*D,hidden]` (`[4096,4096]`), `mixer.k_proj.weight`/
//!   `mixer.v_proj.weight` `[Akv*D,hidden]` (`[1024,4096]`), `mixer.o_proj.weight` `[hidden,Aq*D]`
//!   (`[4096,4096]`).
//! - MLP: `mixer.up_proj.weight` `[Inter,hidden]` (`[21504,4096]`), `mixer.down_proj.weight`
//!   `[hidden,Inter]` (`[4096,21504]`).
//! - Top level (consumed by `trace_nemotron_h_prefill`): `backbone.norm_f.weight` `[hidden]`,
//!   `lm_head.weight` `[vocab,hidden]` (`tie_word_embeddings: false` on the real 8B/56B configs, so it is
//!   a separate tensor), transposed to `[hidden,vocab]` at load time, following `crate::bloom_load`'s
//!   card-258 convention of never re-transposing in-graph.
//!
//! All linear projections are bias-free (`use_bias`/`mamba_proj_bias`/`attention_bias`/`mlp_bias` are
//! `false` in the 8B config), matching `nemotron_h_mamba_layer`'s `linear(.., None)` calls except `dt`
//! (has `dt_bias`) and the conv (has `conv_bias`, FR-007 in `poot_models::nemotron_h`).
//!
//! **Tracer-facing constant names** are a separate convention from the on-disk names, as in
//! `build_bloom_weights` (its `p(...)`/`src_name(...)` pair): `layers.{i}.norm.weight`; Mamba:
//! `layers.{i}.mixer.z.weight`, `.xbc.weight`, `.dt.weight`, `.dt_bias`, `.conv1d.weight` (reshaped/
//! transposed to the tracer's `[K,ConvC]`), `.conv1d.bias`, `.a` (`-exp(A_log)`, `[1,Hq,1,1]`), `.d`
//! (`[1,Hq,1,1]`), `.gate_norm.weight`, `.out_proj.weight`; Attention: `.mixer.q_proj.weight`/
//! `.k_proj.weight`/`.v_proj.weight`/`.o_proj.weight`; MLP: `.mixer.up_proj.weight`/`.down_proj.weight`.
//! `trace_nemotron_h_prefill` declares its `b.constant(...)` calls under these names plus the on-disk
//! top-level names unchanged (`backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight`;
//! singular tensors that need no per-kind convention).
//!

use std::collections::HashMap;

use poot_eval::{Value, materialize_dense};
use poot_models::nemotron_h::{
    NemotronHAttnConfig, NemotronHConfig, NemotronHLayerKind, NemotronHMambaConfig,
    parse_hybrid_pattern,
};
use poot_tensor::HostTensor;

use crate::checkpoint::gguf::{row_slice, transpose2d};
use crate::error::{Result, ResultExt};
use poot_quant::weights::WeightStore;

/// The subset of a real Nemotron-H `config.json` this loader reads (verified against
/// `nvidia/Nemotron-H-8B-Base-8K`'s published `config.json`; see the module doc).
#[derive(Debug, Clone, serde::Deserialize)]
struct NemotronHHfConfig {
    model_type: String,
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    hybrid_override_pattern: String,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    ssm_state_size: usize,
    n_groups: usize,
    conv_kernel: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    attention_head_dim: usize,
    intermediate_size: usize,
    /// Both `rms_norm_eps` and `layer_norm_epsilon` are `1e-05` on every real config seen (spec 279's
    /// open epsilon question, see its Open-questions section); this loader prefers `rms_norm_eps`,
    /// falling back to `layer_norm_epsilon`, then `1e-5`.
    #[serde(default)]
    rms_norm_eps: Option<f32>,
    #[serde(default)]
    layer_norm_epsilon: Option<f32>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<u32>,
}

/// `true` iff `raw`'s `model_type` is `"nemotron_h"`. Checked before `Qwen2HfConfig::load` in
/// `Runner::load_impl`, like `is_bloom`/`is_mpt`.
pub(crate) fn is_nemotron_h(raw: &serde_json::Value) -> bool {
    raw.get("model_type").and_then(|v| v.as_str()) == Some("nemotron_h")
}

/// Parse `raw` (an already-loaded `config.json`) into a [`NemotronHConfig`] plus its bos/eos token ids.
/// Defaults (`eps=1e-5`, `eos=2`, `bos=1`) match `nvidia/Nemotron-H-8B-Base-8K`'s published values and
/// apply only if a config omits them (none seen to).
pub(crate) fn nemotron_h_config_from_json(
    raw: &serde_json::Value,
) -> Result<(NemotronHConfig, u32, u32)> {
    let hf: NemotronHHfConfig = serde_json::from_value(raw.clone())?;
    let pattern = parse_hybrid_pattern(&hf.hybrid_override_pattern);
    if pattern.len() != hf.num_hidden_layers {
        bail!(
            "nemotron_h: hybrid_override_pattern length {} != num_hidden_layers {}",
            pattern.len(),
            hf.num_hidden_layers
        );
    }
    let eps = hf.rms_norm_eps.or(hf.layer_norm_epsilon).unwrap_or(1e-5);
    let cfg = NemotronHConfig {
        vocab_size: hf.vocab_size,
        hidden: hf.hidden_size,
        mlp_inter: hf.intermediate_size,
        eps,
        pattern,
        mamba: NemotronHMambaConfig {
            hidden: hf.hidden_size,
            mamba_num_heads: hf.mamba_num_heads,
            mamba_head_dim: hf.mamba_head_dim,
            n_groups: hf.n_groups,
            ssm_state: hf.ssm_state_size,
            conv_kernel: hf.conv_kernel,
        },
        attn: NemotronHAttnConfig {
            hidden: hf.hidden_size,
            num_heads: hf.num_attention_heads,
            num_kv_heads: hf.num_key_value_heads,
            head_dim: hf.attention_head_dim,
        },
    };
    let eos = hf.eos_token_id.unwrap_or(2);
    let bos = hf.bos_token_id.unwrap_or(1);
    debug_assert_eq!(hf.model_type, "nemotron_h");
    Ok((cfg, eos, bos))
}

/// Undo the `[ConvC,1,K]` depthwise `nn.Conv1d` weight layout into the tracer's `[K,ConvC]` convention
/// (`nemotron_h_mamba_layer`'s `w_conv`). The middle size-1 axis carries no data (`groups=ConvC`), so
/// `[ConvC,1,K]`'s row-major bytes are `[ConvC,K]`'s; relabeling the shape is a no-op and only the
/// trailing transpose moves data.
fn conv1d_weight_kc(rt: &HostTensor, conv_c: usize, k: usize) -> Result<HostTensor> {
    let squeezed = rt.reshaped(vec![conv_c, k]).map_err(|error| {
        err!("nemotron_h: conv1d.weight must hold {conv_c}*{k} elements: {error}")
    })?;
    Ok(transpose2d(&squeezed))
}

/// `a_param = -exp(A_log)` (the `mamba_ssm` convention, see the module doc), reshaped to
/// `nemotron_h_mamba_layer`'s `[1,Hq,1,1]`. The one place `A_log` is widened: the exponential is f32 math.
fn a_param_from_a_log(rt: &HostTensor, hq: usize) -> Result<HostTensor> {
    let a_log = rt
        .to_f32()
        .map_err(|error| err!("nemotron_h: A_log is not a float tensor: {error}"))?;
    let data: Vec<f32> = a_log.iter().map(|v| -v.exp()).collect();
    Ok(HostTensor::f32(vec![1, hq, 1, 1], data))
}

/// Reshape a flat `[Hq]` per-head scalar (`D`) to `nemotron_h_mamba_layer`'s `[1,Hq,1,1]`; no value
/// transform, just the shape the tracer's broadcast rules expect.
fn head_scalar_4d(rt: &HostTensor, hq: usize) -> Result<HostTensor> {
    rt.reshaped(vec![1, hq, 1, 1])
        .map_err(|error| err!("nemotron_h: per-head scalar must hold {hq} elements: {error}"))
}

fn as_tensor(rt: &HostTensor) -> HostTensor {
    rt.clone()
}

/// Load and transform every Nemotron-H weight from `st` (on-disk `backbone.layers.{i}.*` names and the
/// three top-level tensors, see the module doc) into a flat map keyed by this loader's tracer-facing
/// constant names.
pub(crate) fn build_nemotron_h_weights(
    store: &WeightStore,
    cfg: &NemotronHConfig,
) -> Result<HashMap<String, Value>> {
    let mut w = HashMap::new();
    let hq = cfg.mamba.mamba_num_heads;
    let inner = cfg.mamba.inner();
    let conv_c = cfg.mamba.conv_channels();
    let k = cfg.mamba.conv_kernel;

    // Top-level constants `poot_models::nemotron_h::trace_nemotron_h_prefill` expects: the embedding
    // gather source (untransposed, on-disk name unchanged), the final norm, and the lm_head,
    // pre-transposed at load time (`tie_word_embeddings: false` on every real config, so it is a
    // separate on-disk tensor, unlike `crate::bloom_load`'s tied lm_head). Card 258: transpose once
    // host-side, not as a per-step graph op.
    w.insert(
        "backbone.embeddings.weight".to_string(),
        as_tensor(
            &materialize_dense(store, "backbone.embeddings.weight")
                .with_context(|| "nemotron_h: missing backbone.embeddings.weight".to_string())?,
        ),
    );
    w.insert(
        "backbone.norm_f.weight".to_string(),
        as_tensor(
            &materialize_dense(store, "backbone.norm_f.weight")
                .with_context(|| "nemotron_h: missing backbone.norm_f.weight".to_string())?,
        ),
    );
    w.insert(
        "lm_head.weight".to_string(),
        transpose2d(
            &materialize_dense(store, "lm_head.weight")
                .with_context(|| "nemotron_h: missing lm_head.weight".to_string())?,
        ),
    );

    for (li, kind) in cfg.pattern.iter().enumerate() {
        let norm = materialize_dense(store, &format!("backbone.layers.{li}.norm.weight"))
            .with_context(|| format!("nemotron_h: missing backbone.layers.{li}.norm.weight"))?;
        w.insert(format!("layers.{li}.norm.weight"), as_tensor(&norm));

        let mprefix = format!("backbone.layers.{li}.mixer");
        let get = |suf: &str| -> Result<HostTensor> {
            materialize_dense(store, &format!("{mprefix}.{suf}"))
                .with_context(|| format!("nemotron_h: missing {mprefix}.{suf}"))
        };

        match kind {
            NemotronHLayerKind::Mamba => {
                let in_proj = get("in_proj.weight")?;
                let z_raw = row_slice(&in_proj, 0, inner);
                let xbc_raw = row_slice(&in_proj, inner, inner + conv_c);
                let dt_raw = row_slice(&in_proj, inner + conv_c, inner + conv_c + hq);
                w.insert(format!("layers.{li}.mixer.z.weight"), transpose2d(&z_raw));
                w.insert(
                    format!("layers.{li}.mixer.xbc.weight"),
                    transpose2d(&xbc_raw),
                );
                w.insert(format!("layers.{li}.mixer.dt.weight"), transpose2d(&dt_raw));
                w.insert(
                    format!("layers.{li}.mixer.dt_bias"),
                    as_tensor(&get("dt_bias")?),
                );
                w.insert(
                    format!("layers.{li}.mixer.conv1d.weight"),
                    conv1d_weight_kc(&get("conv1d.weight")?, conv_c, k)?,
                );
                w.insert(
                    format!("layers.{li}.mixer.conv1d.bias"),
                    as_tensor(&get("conv1d.bias")?),
                );
                w.insert(
                    format!("layers.{li}.mixer.a"),
                    a_param_from_a_log(&get("A_log")?, hq)?,
                );
                w.insert(
                    format!("layers.{li}.mixer.d"),
                    head_scalar_4d(&get("D")?, hq)?,
                );
                w.insert(
                    format!("layers.{li}.mixer.gate_norm.weight"),
                    as_tensor(&get("norm.weight")?),
                );
                w.insert(
                    format!("layers.{li}.mixer.out_proj.weight"),
                    transpose2d(&get("out_proj.weight")?),
                );
            }
            NemotronHLayerKind::Attention => {
                for suf in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    let rt = get(&format!("{suf}.weight"))?;
                    w.insert(format!("layers.{li}.mixer.{suf}.weight"), transpose2d(&rt));
                }
            }
            NemotronHLayerKind::Mlp => {
                for suf in ["up_proj", "down_proj"] {
                    let rt = get(&format!("{suf}.weight"))?;
                    w.insert(format!("layers.{li}.mixer.{suf}.weight"), transpose2d(&rt));
                }
            }
        }
    }
    Ok(crate::core::runner::dense_weight_map(w))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::runner::Runner;

    use poot_graph_ir::builder::{Builder, Traced};
    use poot_graph_ir::graph::Storage;
    use poot_models::nemotron_h::{
        nemotron_h_attention_layer, nemotron_h_mamba_layer, nemotron_h_mlp_layer,
        trace_nemotron_h_prefill,
    };

    /// Deterministic xorshift PRNG stream in `[-1, 1)` scaled small, as in
    /// `crates/poot-models/src/nemotron_h.rs`'s `cpu_oracle::fill` (real-magnitude, non-degenerate
    /// synthetic weights).
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * 0.3
            })
            .collect()
    }

    use poot_test_util::seed_of;

    fn eps_silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    fn eps_softplus(x: f32) -> f32 {
        (1.0 + x.exp()).ln()
    }

    /// Write a minimal in-memory safetensors buffer (the format `poot_load::SafeTensors::load_bytes`
    /// parses), like `bloom_load.rs`'s `write_safetensors` but returning bytes, since this test needs no
    /// directory/config.json/tokenizer.
    fn write_safetensors_bytes(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let end = data.len();
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    /// `nemotron_h_config_from_json` on a real-shaped synthetic config.json (verbatim 8B
    /// `hybrid_override_pattern` prefix) produces the per-layer-kind counts `parse_hybrid_pattern`'s unit
    /// tests establish: a loader-level check that config parsing and pattern decoding agree end to end.
    #[test]
    fn config_from_json_parses_real_field_names_and_pattern() {
        let raw = serde_json::json!({
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "num_hidden_layers": 9,
            "vocab_size": 131072,
            "hybrid_override_pattern": "M-M-M-M*-",
            "mamba_num_heads": 128,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "attention_head_dim": 128,
            "intermediate_size": 21504,
            "rms_norm_eps": 1e-5,
            "layer_norm_epsilon": 1e-5,
            "bos_token_id": 1,
            "eos_token_id": 2,
        });
        assert!(is_nemotron_h(&raw));
        let (cfg, eos, bos) = nemotron_h_config_from_json(&raw).expect("parse config");
        assert_eq!(cfg.pattern.len(), 9);
        assert_eq!(
            cfg.pattern
                .iter()
                .filter(|k| **k == NemotronHLayerKind::Attention)
                .count(),
            1
        );
        assert_eq!(cfg.mamba.mamba_num_heads, 128);
        assert_eq!(cfg.attn.num_kv_heads, 8);
        assert_eq!(eos, 2);
        assert_eq!(bos, 1);
    }

    /// The core wiring proof: a synthetic, real-named and real-layout safetensors buffer for a 3-layer toy
    /// hybrid stack (`"M*-"`: one Mamba, one Attention, one MLP layer) is loaded through
    /// `build_nemotron_h_weights`, bound as `Traced` constants under the loader's tracer-facing names, and
    /// run through the CPU-oracle-verified (updates 0853/0858) `nemotron_h_mamba_layer`/
    /// `nemotron_h_attention_layer`/`nemotron_h_mlp_layer` composition for one decode step. The expected
    /// output is computed in plain Rust from the original real-shaped arrays (HF `[out,in]` convention,
    /// the fused `in_proj` row-sliced like `torch.split`, the `[ConvC,1,K]` conv1d layout, `A_log` before
    /// any `-exp`), never via this module's transform helpers, so a bug in `build_nemotron_h_weights`
    /// (wrong row range, wrong transpose axis, missing `-exp`, wrong conv1d reshape) shows as a numeric
    /// mismatch, not just a shape-compatible wrong load.
    #[test]
    fn synthetic_real_named_checkpoint_loads_and_matches_independent_reference() {
        let (hidden, mlp_inter) = (6usize, 10usize);
        let mcfg = NemotronHMambaConfig {
            hidden,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            ssm_state: 3,
            conv_kernel: 3,
        };
        let acfg = NemotronHAttnConfig {
            hidden,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
        };
        let eps = 1e-6f32;
        let pattern = parse_hybrid_pattern("M*-");
        let cfg = NemotronHConfig {
            vocab_size: 16,
            hidden,
            mlp_inter,
            eps,
            pattern: pattern.clone(),
            mamba: mcfg,
            attn: acfg,
        };

        let (hq, p, n, g) = (
            mcfg.mamba_num_heads,
            mcfg.mamba_head_dim,
            mcfg.ssm_state,
            mcfg.n_groups,
        );
        let inner = mcfg.inner();
        let conv_c = mcfg.conv_channels();
        let k = mcfg.conv_kernel;
        let (aq, akv, d) = (acfg.num_heads, acfg.num_kv_heads, acfg.head_dim);
        let n_rep_m = mcfg.heads_per_group();
        let n_rep_a = aq / akv;
        let scale = 1.0 / (d as f32).sqrt();

        // --- real-shaped, real-named synthetic tensors (layer 0 = Mamba, layer 1 = Attention,
        // layer 2 = MLP) ---
        let norm0 = fill(hidden, seed_of("l0.norm"));
        let norm1 = fill(hidden, seed_of("l1.norm"));
        let norm2 = fill(hidden, seed_of("l2.norm"));

        // Fused in_proj.weight [inner+conv_c+hq, hidden] (HF [out,in]), built by row-concatenating
        // independently seeded z/xBC/dt blocks, the `torch.split` layout.
        let z_block = fill(inner * hidden, seed_of("l0.z"));
        let xbc_block = fill(conv_c * hidden, seed_of("l0.xbc"));
        let dt_block = fill(hq * hidden, seed_of("l0.dt"));
        let mut in_proj_raw = Vec::with_capacity((inner + conv_c + hq) * hidden);
        in_proj_raw.extend_from_slice(&z_block);
        in_proj_raw.extend_from_slice(&xbc_block);
        in_proj_raw.extend_from_slice(&dt_block);

        // Real conv1d.weight [conv_c, 1, k] == flat [conv_c, k] row-major (channel-major).
        let conv1d_raw = fill(conv_c * k, seed_of("l0.conv1d"));
        let conv1d_bias = fill(conv_c, seed_of("l0.conv1d.bias"));
        let dt_bias = fill(hq, seed_of("l0.dt_bias"));
        // A_log: values before the -exp transform, kept small so a_par=-exp(A_log) stays a reasonable
        // decay (mamba_ssm init: A_log ~ log(uniform(1,16))).
        let a_log = fill(hq, seed_of("l0.a_log"))
            .iter()
            .map(|v| v.abs() + 0.5)
            .collect::<Vec<f32>>();
        let d_param = fill(hq, seed_of("l0.d"));
        let gate_norm = fill(inner, seed_of("l0.gate_norm"))
            .iter()
            .map(|v| 1.0 + v * 0.2)
            .collect::<Vec<f32>>();
        let out_proj_raw = fill(hidden * inner, seed_of("l0.out_proj")); // real [hidden, inner]

        let qd = aq * d;
        let kvd = akv * d;
        let q_raw = fill(qd * hidden, seed_of("l1.q")); // real [qd, hidden]
        let k_raw = fill(kvd * hidden, seed_of("l1.k"));
        let v_raw = fill(kvd * hidden, seed_of("l1.v"));
        let o_raw = fill(hidden * qd, seed_of("l1.o")); // real [hidden, qd]

        let up_raw = fill(mlp_inter * hidden, seed_of("l2.up")); // real [inter, hidden]
        let down_raw = fill(hidden * mlp_inter, seed_of("l2.down")); // real [hidden, inter]

        let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
            // Top-level tensors `build_nemotron_h_weights` also loads, present on every real checkpoint
            // though this test only runs one decode step through the per-layer composition (see
            // `synthetic_checkpoint_prefill_end_to_end_via_trace_nemotron_h_prefill` below for the
            // top-level trace).
            (
                "backbone.embeddings.weight".into(),
                vec![cfg.vocab_size, hidden],
                fill(cfg.vocab_size * hidden, seed_of("embed")),
            ),
            (
                "backbone.norm_f.weight".into(),
                vec![hidden],
                fill(hidden, seed_of("norm_f")),
            ),
            (
                "lm_head.weight".into(),
                vec![cfg.vocab_size, hidden],
                fill(cfg.vocab_size * hidden, seed_of("lm_head")),
            ),
            (
                "backbone.layers.0.norm.weight".into(),
                vec![hidden],
                norm0.clone(),
            ),
            (
                "backbone.layers.0.mixer.in_proj.weight".into(),
                vec![inner + conv_c + hq, hidden],
                in_proj_raw.clone(),
            ),
            (
                "backbone.layers.0.mixer.conv1d.weight".into(),
                vec![conv_c, 1, k],
                conv1d_raw.clone(),
            ),
            (
                "backbone.layers.0.mixer.conv1d.bias".into(),
                vec![conv_c],
                conv1d_bias.clone(),
            ),
            (
                "backbone.layers.0.mixer.dt_bias".into(),
                vec![hq],
                dt_bias.clone(),
            ),
            (
                "backbone.layers.0.mixer.A_log".into(),
                vec![hq],
                a_log.clone(),
            ),
            (
                "backbone.layers.0.mixer.D".into(),
                vec![hq],
                d_param.clone(),
            ),
            (
                "backbone.layers.0.mixer.norm.weight".into(),
                vec![inner],
                gate_norm.clone(),
            ),
            (
                "backbone.layers.0.mixer.out_proj.weight".into(),
                vec![hidden, inner],
                out_proj_raw.clone(),
            ),
            (
                "backbone.layers.1.norm.weight".into(),
                vec![hidden],
                norm1.clone(),
            ),
            (
                "backbone.layers.1.mixer.q_proj.weight".into(),
                vec![qd, hidden],
                q_raw.clone(),
            ),
            (
                "backbone.layers.1.mixer.k_proj.weight".into(),
                vec![kvd, hidden],
                k_raw.clone(),
            ),
            (
                "backbone.layers.1.mixer.v_proj.weight".into(),
                vec![kvd, hidden],
                v_raw.clone(),
            ),
            (
                "backbone.layers.1.mixer.o_proj.weight".into(),
                vec![hidden, qd],
                o_raw.clone(),
            ),
            (
                "backbone.layers.2.norm.weight".into(),
                vec![hidden],
                norm2.clone(),
            ),
            (
                "backbone.layers.2.mixer.up_proj.weight".into(),
                vec![mlp_inter, hidden],
                up_raw.clone(),
            ),
            (
                "backbone.layers.2.mixer.down_proj.weight".into(),
                vec![hidden, mlp_inter],
                down_raw.clone(),
            ),
        ];
        let bytes = write_safetensors_bytes(&tensors);
        let store = poot_load::safetensors::load_weight_store_bytes(&bytes)
            .expect("parse synthetic safetensors");

        let loaded = build_nemotron_h_weights(&store, &cfg).expect("build_nemotron_h_weights");
        // Every name this loader is documented to produce for a 3-layer M*- stack must be present.
        for name in [
            "layers.0.norm.weight",
            "layers.0.mixer.z.weight",
            "layers.0.mixer.xbc.weight",
            "layers.0.mixer.dt.weight",
            "layers.0.mixer.dt_bias",
            "layers.0.mixer.conv1d.weight",
            "layers.0.mixer.conv1d.bias",
            "layers.0.mixer.a",
            "layers.0.mixer.d",
            "layers.0.mixer.gate_norm.weight",
            "layers.0.mixer.out_proj.weight",
            "layers.1.norm.weight",
            "layers.1.mixer.q_proj.weight",
            "layers.1.mixer.k_proj.weight",
            "layers.1.mixer.v_proj.weight",
            "layers.1.mixer.o_proj.weight",
            "layers.2.norm.weight",
            "layers.2.mixer.up_proj.weight",
            "layers.2.mixer.down_proj.weight",
        ] {
            assert!(loaded.contains_key(name), "loader did not produce {name}");
        }

        // --- build the traced graph: one decode step through the 3-layer stack, using the loaded
        // weights as constants under the loader's own naming convention. ---
        let b = Builder::new();
        let x_val = fill(hidden, seed_of("input.x"));
        let x0 = b.constant(
            "x",
            poot_graph_ir::types::TensorType::f32(vec![1, 1, hidden]),
        );
        let mut inp: HashMap<poot_graph_ir::ValueId, HostTensor> = HashMap::new();
        inp.insert(x0.id, HostTensor::f32(vec![1, 1, hidden], x_val.clone()));

        fn bind_const(
            b: &Builder,
            loaded: &HashMap<String, Value>,
            inp: &mut HashMap<poot_graph_ir::ValueId, HostTensor>,
            name: &str,
            shape: Vec<usize>,
        ) -> Traced {
            let t = b.constant(name, poot_graph_ir::types::TensorType::f32(shape));
            inp.insert(
                t.id,
                loaded
                    .get(name)
                    .unwrap()
                    .clone()
                    .as_host()
                    .expect("dense weight")
                    .clone(),
            );
            t
        }
        macro_rules! c {
            ($name:expr, $shape:expr) => {
                bind_const(&b, &loaded, &mut inp, $name, $shape)
            };
        }

        let norm_w0 = c!("layers.0.norm.weight", vec![hidden]);
        let wz = c!("layers.0.mixer.z.weight", vec![hidden, inner]);
        let wxbc = c!("layers.0.mixer.xbc.weight", vec![hidden, conv_c]);
        let wdt = c!("layers.0.mixer.dt.weight", vec![hidden, hq]);
        let dtb = c!("layers.0.mixer.dt_bias", vec![hq]);
        let wconv = c!("layers.0.mixer.conv1d.weight", vec![k, conv_c]);
        let convb = c!("layers.0.mixer.conv1d.bias", vec![conv_c]);
        let ap = c!("layers.0.mixer.a", vec![1, hq, 1, 1]);
        let dp = c!("layers.0.mixer.d", vec![1, hq, 1, 1]);
        let gn = c!("layers.0.mixer.gate_norm.weight", vec![inner]);
        let wout = c!("layers.0.mixer.out_proj.weight", vec![inner, hidden]);
        // Nonzero initial conv cache / SSM state (a second decode step, not step 0): with a zero initial
        // SSM state `a_par` (and the A_log -> -exp(A_log) transform) only multiplies zero and would be
        // invisible to the comparison. Nonzero cache/state makes both matter.
        let conv_cache_data = fill((k - 1) * conv_c, seed_of("conv_cache0"));
        let conv_cache = b.constant(
            "conv_cache0",
            poot_graph_ir::types::TensorType::f32(vec![1, k - 1, conv_c]),
        );
        inp.insert(
            conv_cache.id,
            HostTensor::f32(vec![1, k - 1, conv_c], conv_cache_data.clone()),
        );
        let ssm_in_data = fill(hq * n * p, seed_of("ssm_in0"));
        let ssm_in = b.constant(
            "ssm_in0",
            poot_graph_ir::types::TensorType::f32(vec![1, hq, n, p]),
        );
        inp.insert(
            ssm_in.id,
            HostTensor::f32(vec![1, hq, n, p], ssm_in_data.clone()),
        );

        let (out0, _conv_out, _ssm_out) = nemotron_h_mamba_layer(
            &b, x0, &mcfg, norm_w0, wz, wxbc, wdt, dtb, wconv, convb, ap, dp, gn, wout, conv_cache,
            ssm_in, eps,
        );

        let norm_w1 = c!("layers.1.norm.weight", vec![hidden]);
        let wq = c!("layers.1.mixer.q_proj.weight", vec![hidden, qd]);
        let wk = c!("layers.1.mixer.k_proj.weight", vec![hidden, kvd]);
        let wv = c!("layers.1.mixer.v_proj.weight", vec![hidden, kvd]);
        let wo = c!("layers.1.mixer.o_proj.weight", vec![qd, hidden]);
        let kc_in = b.constant(
            "kc_in1",
            poot_graph_ir::types::TensorType::f32(vec![1, akv, 1, d]),
        );
        inp.insert(
            kc_in.id,
            HostTensor::f32(vec![1, akv, 1, d], vec![0.0; akv * d]),
        );
        let vc_in = b.constant(
            "vc_in1",
            poot_graph_ir::types::TensorType::f32(vec![1, akv, 1, d]),
        );
        inp.insert(
            vc_in.id,
            HostTensor::f32(vec![1, akv, 1, d], vec![0.0; akv * d]),
        );

        let (out1, _kc_out, _vc_out) = nemotron_h_attention_layer(
            &b, out0, &acfg, norm_w1, wq, wk, wv, wo, kc_in, vc_in, 0, eps,
        );

        let norm_w2 = c!("layers.2.norm.weight", vec![hidden]);
        let wup = c!("layers.2.mixer.up_proj.weight", vec![hidden, mlp_inter]);
        let wdown = c!("layers.2.mixer.down_proj.weight", vec![mlp_inter, hidden]);
        let out2 = nemotron_h_mlp_layer(&b, out1, norm_w2, wup, wdown, eps);

        let graph = b.finish(out2);
        let inp: HashMap<poot_graph_ir::ValueId, poot_eval::Value> =
            inp.into_iter().map(|(k, v)| (k, v.into())).collect();
        let got = poot_eval::eval(
            &graph,
            &inp,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval loaded-weight graph")
        .output
        .into_host()
        .expect("dense output");

        // --- independent hand reference, computed from the original real-shaped arrays (never via
        // build_nemotron_h_weights' transform helpers). ---
        let mut x = x_val.clone();

        // Layer 0: Mamba.
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let denom = (ms + eps).sqrt();
        let xn: Vec<f32> = x
            .iter()
            .zip(&norm0)
            .map(|(v, gm)| (v / denom) * gm)
            .collect();

        let hf_dot = |raw: &[f32], out_dim: usize, in_dim: usize, xn: &[f32]| -> Vec<f32> {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|i| xn[i] * raw[o * in_dim + i]).sum())
                .collect()
        };
        let z = hf_dot(&z_block, inner, hidden, &xn);
        let xbc_raw_vec = hf_dot(&xbc_block, conv_c, hidden, &xn);
        let mut dt_raw_vec = hf_dot(&dt_block, hq, hidden, &xn);
        for (h, dv) in dt_raw_vec.iter_mut().enumerate() {
            *dv += dt_bias[h];
        }

        // Causal conv1d against the nonzero synthetic conv_cache_data (the last K-1 raw xBC history rows,
        // oldest first, matching `causal_conv1d_decode`'s `window = concat(cache, x)`), [conv_c,k]
        // channel-major layout (`conv1d_raw[c*k+kk]`), plus the per-channel bias.
        let mut xbc_act = vec![0.0f32; conv_c];
        for (c_idx, slot) in xbc_act.iter_mut().enumerate() {
            let mut acc = conv1d_bias[c_idx];
            for kk in 0..k - 1 {
                acc += conv1d_raw[c_idx * k + kk] * conv_cache_data[kk * conv_c + c_idx];
            }
            acc += conv1d_raw[c_idx * k + (k - 1)] * xbc_raw_vec[c_idx];
            *slot = eps_silu(acc);
        }
        let x_part = &xbc_act[0..inner];
        let b_part = &xbc_act[inner..inner + g * n];
        let c_part = &xbc_act[inner + g * n..inner + 2 * g * n];

        let mut y = vec![0.0f32; inner];
        for h in 0..hq {
            let grp = h / n_rep_m;
            let a_par_h = -a_log[h].exp(); // the real mamba_ssm A = -exp(A_log) transform
            let delta_h = eps_softplus(dt_raw_vec[h]);
            let a_bar = (delta_h * a_par_h).exp();
            for nn in 0..n {
                let b_val = b_part[grp * n + nn];
                let b_bar = delta_h * b_val;
                for pp in 0..p {
                    let xv = x_part[h * p + pp];
                    let h_in_val = ssm_in_data[(h * n + nn) * p + pp];
                    let ssm_val = a_bar * h_in_val + b_bar * xv;
                    y[h * p + pp] += c_part[grp * n + nn] * ssm_val;
                }
            }
            for pp in 0..p {
                y[h * p + pp] += d_param[h] * x_part[h * p + pp];
            }
        }
        let gated: Vec<f32> = y
            .iter()
            .zip(&z)
            .map(|(yv, zv)| yv * eps_silu(*zv))
            .collect();
        let gms: f32 = gated.iter().map(|v| v * v).sum::<f32>() / inner as f32;
        let gdenom = (gms + eps).sqrt();
        let normed: Vec<f32> = gated
            .iter()
            .zip(&gate_norm)
            .map(|(v, gg)| (v / gdenom) * gg)
            .collect();
        let mamba_out = hf_dot(&out_proj_raw, hidden, inner, &normed);
        for j in 0..hidden {
            x[j] += mamba_out[j];
        }

        // Layer 1: Attention (single position, NoPE).
        let ms1: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let denom1 = (ms1 + eps).sqrt();
        let xn1: Vec<f32> = x
            .iter()
            .zip(&norm1)
            .map(|(v, gm)| (v / denom1) * gm)
            .collect();
        let q = hf_dot(&q_raw, qd, hidden, &xn1);
        let kk = hf_dot(&k_raw, kvd, hidden, &xn1);
        let vv = hf_dot(&v_raw, kvd, hidden, &xn1);
        let mut o = vec![0.0f32; qd];
        for h in 0..aq {
            let kvh = h / n_rep_a;
            let mut dot = 0.0f32;
            for e in 0..d {
                dot += q[h * d + e] * kk[kvh * d + e];
            }
            let score = dot * scale;
            // single position -> softmax over one element is always weight 1.
            let _ = score;
            for e in 0..d {
                o[h * d + e] = vv[kvh * d + e];
            }
        }
        let attn_out = hf_dot(&o_raw, hidden, qd, &o);
        for j in 0..hidden {
            x[j] += attn_out[j];
        }

        // Layer 2: MLP (relu^2, no gate).
        let ms2: f32 = x.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let denom2 = (ms2 + eps).sqrt();
        let xn2: Vec<f32> = x
            .iter()
            .zip(&norm2)
            .map(|(v, gm)| (v / denom2) * gm)
            .collect();
        let up = hf_dot(&up_raw, mlp_inter, hidden, &xn2);
        let sq: Vec<f32> = up.iter().map(|v| v.max(0.0).powi(2)).collect();
        let down = hf_dot(&down_raw, hidden, mlp_inter, &sq);
        for j in 0..hidden {
            x[j] += down[j];
        }

        for (i, (a, w)) in got.as_f32().unwrap().iter().zip(x.iter()).enumerate() {
            let diff = (a - w).abs();
            let tol = 1e-4 * w.abs().max(1.0);
            assert!(diff <= tol, "element {i}: got {a} want {w}");
        }
    }

    /// Build a real-named, real-layout synthetic safetensors buffer for `cfg.pattern`'s full hybrid stack,
    /// including the three top-level tensors `trace_nemotron_h_prefill` needs
    /// (`backbone.embeddings.weight`, `backbone.norm_f.weight`, `lm_head.weight`), which
    /// `synthetic_real_named_checkpoint_loads_and_matches_independent_reference` does not exercise (it
    /// covers one decode step through the per-layer composition only).
    fn write_synthetic_nemotron_h_checkpoint_bytes(cfg: &NemotronHConfig) -> Vec<u8> {
        let hidden = cfg.hidden;
        let inner = cfg.mamba.inner();
        let conv_c = cfg.mamba.conv_channels();
        let hq = cfg.mamba.mamba_num_heads;
        let k = cfg.mamba.conv_kernel;
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        tensors.push((
            "backbone.embeddings.weight".into(),
            vec![cfg.vocab_size, hidden],
            fill(cfg.vocab_size * hidden, seed_of("embed")),
        ));
        tensors.push((
            "backbone.norm_f.weight".into(),
            vec![hidden],
            fill(hidden, seed_of("norm_f"))
                .iter()
                .map(|v| 1.0 + v * 0.2)
                .collect(),
        ));
        tensors.push((
            "lm_head.weight".into(),
            vec![cfg.vocab_size, hidden],
            fill(cfg.vocab_size * hidden, seed_of("lm_head")),
        ));

        for (li, kind) in cfg.pattern.iter().enumerate() {
            tensors.push((
                format!("backbone.layers.{li}.norm.weight"),
                vec![hidden],
                fill(hidden, seed_of(&format!("l{li}.norm")))
                    .iter()
                    .map(|v| 1.0 + v * 0.2)
                    .collect(),
            ));
            match kind {
                NemotronHLayerKind::Mamba => {
                    let z_block = fill(inner * hidden, seed_of(&format!("l{li}.z")));
                    let xbc_block = fill(conv_c * hidden, seed_of(&format!("l{li}.xbc")));
                    let dt_block = fill(hq * hidden, seed_of(&format!("l{li}.dt")));
                    let mut in_proj = Vec::with_capacity((inner + conv_c + hq) * hidden);
                    in_proj.extend_from_slice(&z_block);
                    in_proj.extend_from_slice(&xbc_block);
                    in_proj.extend_from_slice(&dt_block);
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.in_proj.weight"),
                        vec![inner + conv_c + hq, hidden],
                        in_proj,
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.conv1d.weight"),
                        vec![conv_c, 1, k],
                        fill(conv_c * k, seed_of(&format!("l{li}.conv1d"))),
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.conv1d.bias"),
                        vec![conv_c],
                        fill(conv_c, seed_of(&format!("l{li}.conv1d.bias"))),
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.dt_bias"),
                        vec![hq],
                        fill(hq, seed_of(&format!("l{li}.dt_bias"))),
                    ));
                    // A_log: values before the -exp transform, kept small so a_par = -exp(A_log) stays a
                    // reasonable decay (as in mamba_ssm's init).
                    let a_log: Vec<f32> = fill(hq, seed_of(&format!("l{li}.a_log")))
                        .iter()
                        .map(|v| v.abs() + 0.5)
                        .collect();
                    tensors.push((format!("backbone.layers.{li}.mixer.A_log"), vec![hq], a_log));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.D"),
                        vec![hq],
                        fill(hq, seed_of(&format!("l{li}.d"))),
                    ));
                    let gate_norm: Vec<f32> = fill(inner, seed_of(&format!("l{li}.gate_norm")))
                        .iter()
                        .map(|v| 1.0 + v * 0.2)
                        .collect();
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.norm.weight"),
                        vec![inner],
                        gate_norm,
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.out_proj.weight"),
                        vec![hidden, inner],
                        fill(hidden * inner, seed_of(&format!("l{li}.out_proj"))),
                    ));
                }
                NemotronHLayerKind::Attention => {
                    let qd = cfg.attn.num_heads * cfg.attn.head_dim;
                    let kvd = cfg.attn.num_kv_heads * cfg.attn.head_dim;
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.q_proj.weight"),
                        vec![qd, hidden],
                        fill(qd * hidden, seed_of(&format!("l{li}.q"))),
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.k_proj.weight"),
                        vec![kvd, hidden],
                        fill(kvd * hidden, seed_of(&format!("l{li}.k"))),
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.v_proj.weight"),
                        vec![kvd, hidden],
                        fill(kvd * hidden, seed_of(&format!("l{li}.v"))),
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.o_proj.weight"),
                        vec![hidden, qd],
                        fill(hidden * qd, seed_of(&format!("l{li}.o"))),
                    ));
                }
                NemotronHLayerKind::Mlp => {
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.up_proj.weight"),
                        vec![cfg.mlp_inter, hidden],
                        fill(cfg.mlp_inter * hidden, seed_of(&format!("l{li}.up"))),
                    ));
                    tensors.push((
                        format!("backbone.layers.{li}.mixer.down_proj.weight"),
                        vec![hidden, cfg.mlp_inter],
                        fill(hidden * cfg.mlp_inter, seed_of(&format!("l{li}.down"))),
                    ));
                }
            }
        }

        write_safetensors_bytes(&tensors)
    }

    /// Loader-to-tracer integration (synthetic checkpoint; no real Nemotron-H checkpoint is available
    /// locally): a synthetic real-shaped safetensors buffer for the first 9 characters of the 8B
    /// `hybrid_override_pattern` (`"M-M-M-M*-"`, the toy stack of `crates/poot-models/src/nemotron_h.rs`'s
    /// SC-002/SC-003 and prefill-capstone tests: all three layer kinds, grouped B/C, NoPE attention) is
    /// loaded through `build_nemotron_h_weights` and the whole model is traced via
    /// `trace_nemotron_h_prefill`, not just one layer's composition. This is a coherence check (finite,
    /// non-NaN, non-degenerate logits of the expected shape), not a numerical match: the per-layer math
    /// was verified in updates 0853/0858/0861. It shows that loading real-named weights and tracing the
    /// full embed -> hybrid-stack -> norm -> lm_head graph connects without a wiring bug
    /// (a shape mismatch, a missing constant, a NaN from a mismatched transform).
    #[test]
    fn synthetic_checkpoint_prefill_end_to_end_via_trace_nemotron_h_prefill() {
        let hidden = 6usize;
        let mcfg = NemotronHMambaConfig {
            hidden,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            ssm_state: 3,
            conv_kernel: 3,
        };
        let acfg = NemotronHAttnConfig {
            hidden,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
        };
        let pattern = parse_hybrid_pattern("M-M-M-M*-");
        let cfg = NemotronHConfig {
            vocab_size: 12,
            hidden,
            mlp_inter: 10,
            eps: 1e-6,
            pattern,
            mamba: mcfg,
            attn: acfg,
        };

        let bytes = write_synthetic_nemotron_h_checkpoint_bytes(&cfg);
        let store = poot_load::safetensors::load_weight_store_bytes(&bytes)
            .expect("parse synthetic nemotron_h checkpoint");
        let loaded = build_nemotron_h_weights(&store, &cfg).expect("build_nemotron_h_weights");
        for name in [
            "backbone.embeddings.weight",
            "backbone.norm_f.weight",
            "lm_head.weight",
        ] {
            assert!(loaded.contains_key(name), "loader did not produce {name}");
        }

        let seq_len = 4usize;
        let graph = trace_nemotron_h_prefill(&cfg, seq_len);

        // causal.mask (now the mask.prefill step input, card 550a): [1,1,L,L] additive causal mask
        // (0 on/below diagonal, large-negative above).
        let mut cmask = vec![0.0f32; seq_len * seq_len];
        for t in 0..seq_len {
            for j in 0..seq_len {
                if j > t {
                    cmask[t * seq_len + j] = -1.0e9;
                }
            }
        }

        let tokens: Vec<u32> = (0..seq_len as u32).collect(); // all < vocab_size=12

        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &graph.inputs {
            let meta = graph.meta(id);
            let t = match meta.storage {
                Storage::Slot(poot_graph_ir::Slot::Token) => HostTensor::i32(
                    vec![seq_len],
                    tokens.iter().map(|&tok| tok as i32).collect(),
                ),
                Storage::Slot(poot_graph_ir::Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    HostTensor::f32(vec![1, 1, seq_len, seq_len], cmask.clone())
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    loaded
                        .get(name)
                        .unwrap_or_else(|| panic!("missing loaded weight {name}"))
                        .as_host()
                        .expect("dense weight")
                        .clone()
                }
                other => panic!("unexpected storage {other:?} in nemotron_h prefill graph"),
            };
            inputs.insert(id, t.into());
        }

        let logits = poot_eval::eval(
            &graph,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval nemotron_h prefill graph")
        .output
        .into_host()
        .expect("dense output");
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab_size]);
        assert!(
            logits.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "logits must be finite: {:?}",
            logits.as_f32().unwrap()
        );
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .any(|&v| v != logits.as_f32().unwrap()[0]),
            "logits must not be degenerate (all-equal)"
        );
    }

    /// The `cfg`/real-shaped `config.json` twin of `synthetic_checkpoint_prefill_end_to_end_via_
    /// trace_nemotron_h_prefill`'s `cfg`: the same shape (hidden=6, mlp_inter=10, vocab_size=12,
    /// `"M-M-M-M*-"` hybrid pattern), also producing the `config.json` field names that
    /// `Runner::load_impl`'s `is_nemotron_h`/`nemotron_h_config_from_json` parse, for the `Runner`-level
    /// wiring test below.
    fn tiny_stack() -> NemotronHConfig {
        let hidden = 6usize;
        let mamba = NemotronHMambaConfig {
            hidden,
            mamba_num_heads: 4,
            mamba_head_dim: 2,
            n_groups: 2,
            ssm_state: 3,
            conv_kernel: 3,
        };
        let attn = NemotronHAttnConfig {
            hidden,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 2,
        };
        let pattern = parse_hybrid_pattern("M-M-M-M*-");
        NemotronHConfig {
            vocab_size: 12,
            hidden,
            mlp_inter: 10,
            eps: 1e-6,
            pattern,
            mamba,
            attn,
        }
    }

    /// Real Nemotron-H `config.json` field names (see the module doc) for [`tiny_stack`]'s shape, which
    /// `nemotron_h_config_from_json` (via `Runner::load_impl`'s `is_nemotron_h` peek) parses; like
    /// `deepseek32_load.rs`'s `real_shaped_config_json` for its `Runner`-level wiring test.
    fn real_shaped_config_json(cfg: &NemotronHConfig) -> serde_json::Value {
        serde_json::json!({
            "model_type": "nemotron_h",
            "hidden_size": cfg.hidden,
            "num_hidden_layers": cfg.pattern.len(),
            "vocab_size": cfg.vocab_size,
            "hybrid_override_pattern": "M-M-M-M*-",
            "mamba_num_heads": cfg.mamba.mamba_num_heads,
            "mamba_head_dim": cfg.mamba.mamba_head_dim,
            "ssm_state_size": cfg.mamba.ssm_state,
            "n_groups": cfg.mamba.n_groups,
            "conv_kernel": cfg.mamba.conv_kernel,
            "num_attention_heads": cfg.attn.num_heads,
            "num_key_value_heads": cfg.attn.num_kv_heads,
            "attention_head_dim": cfg.attn.head_dim,
            "intermediate_size": cfg.mlp_inter,
            "rms_norm_eps": cfg.eps,
            "layer_norm_epsilon": cfg.eps,
            "bos_token_id": 1,
            "eos_token_id": 2,
        })
    }

    /// Write `bytes` (from [`write_synthetic_nemotron_h_checkpoint_bytes`]) to `path`, the on-disk
    /// counterpart of the in-memory tests above (like `deepseek32_load.rs`'s `write_safetensors_file`).
    /// `Runner::load` need a real `model.safetensors` file, unlike this
    /// module's other tests, which parse bytes via `SafeTensors::load_bytes`.
    fn write_safetensors_file(path: &std::path::Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write model.safetensors fixture");
    }

    /// Build a tiny real-shaped Nemotron-H checkpoint directory (config.json + model.safetensors +
    /// tokenizer.json) under `std::env::temp_dir()`: the on-disk counterpart of [`tiny_stack`]/
    /// [`write_synthetic_nemotron_h_checkpoint_bytes`], like `deepseek32_load.rs`'s
    /// `write_tiny_deepseek32_checkpoint`. Returns the directory path.
    pub(crate) fn write_tiny_nemotron_h_checkpoint() -> poot_test_util::UniqueTempPath {
        let cfg = tiny_stack();
        let dir = poot_test_util::unique_temp_path("poot_nemotron_h_loader_fixture");
        std::fs::create_dir_all(&dir).expect("create fixture dir");

        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&real_shaped_config_json(&cfg)).unwrap(),
        )
        .expect("write config.json");

        let bytes = write_synthetic_nemotron_h_checkpoint_bytes(&cfg);
        write_safetensors_file(&dir.join("model.safetensors"), &bytes);

        // A minimal tokenizers::Tokenizer (WordLevel, "t0".."t{vocab-1}"), as in
        // deepseek32_load.rs's/bloom_load.rs's fixtures, saved to disk so Runner::load's
        // Tokenizer::from_file path is exercised, not an in-test encoder.
        let vocab_map: std::collections::HashMap<String, u32> = (0..cfg.vocab_size)
            .map(|i| (format!("t{i}"), i as u32))
            .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab_map)
            .unk_token("t0".to_string())
            .build()
            .expect("build wordlevel model");
        let mut tok = tokenizers::Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        tok.save(dir.join("tokenizer.json"), false)
            .expect("save tokenizer.json");

        dir
    }

    /// End to end through the full production `Runner::load` path (update 0870, extended for decode),
    /// unlike the tests above that call `nemotron_h_config_from_json`/`build_nemotron_h_weights`
    /// directly: it shows `Runner::load_impl`'s `is_nemotron_h` raw-JSON-peek detection,
    /// `Runner::load_nemotron_h_impl`'s config/weight-map construction, and `Runner::stateless_prefill_graph`'s
    /// `nemotron_h` dispatch arm (plus `Runner::bind`'s mask slot arm) connect, like
    /// `deepseek32_load.rs`'s `runner_load_deepseek32_and_generates_finite_output`.
    ///
    /// `generate_kv_masked` is called for real (spec 279 decode-step follow-up), exercising the `trace_nemotron_h_decode`
    /// dispatch arm on a production-loaded checkpoint.
    #[test]
    fn runner_load_nemotron_h_and_generates_finite_output() {
        let dir = write_tiny_nemotron_h_checkpoint();
        let runner = Runner::load(&dir).expect("load synthetic nemotron_h checkpoint");

        assert_eq!(runner.arch, "nemotron_h");
        assert_eq!(
            runner.decode_arch().unwrap(),
            crate::core::decode_arch::DecodeArch::NemotronH
        );
        assert!(
            runner.deepseek32.is_none(),
            "a nemotron_h checkpoint must not also set bloom/mpt/deepseek32"
        );
        let np = runner
            .nemotron_h
            .as_ref()
            .expect("Runner::load must set nemotron_h for a nemotron_h checkpoint");
        assert_eq!(np.vocab_size, 12);
        assert_eq!(np.hidden, 6);
        assert_eq!(np.pattern.len(), 9);
        assert_eq!(np.mamba.mamba_num_heads, 4);
        assert_eq!(np.attn.num_kv_heads, 2);

        // The production entry point (Runner::generate -> generate_sampled's nemotron_h arm ->
        // stateless_prefill_graph, re-prefilling the growing context each step).
        let tokens = runner
            .generate("t1 t2 t3", 2, |_| std::ops::ControlFlow::Continue(()))
            .expect("generate (re-prefill) on synthetic nemotron_h checkpoint");
        assert!(
            tokens.len() >= 3,
            "at least the 3 prompt tokens should come back"
        );

        // The fixed-KV masked decode entry point has a Nemotron-H arm: `trace_nemotron_h_decode` built
        // once for cap=5 (3 prompt tokens + 2 generated) and replayed via `Runner::bind_decode` for every
        // position, the G3d convention of every fixed-KV decode tracer here. Must return `Ok`.
        let kv_tokens = runner
            .generate_kv_masked("t1 t2 t3", 2, |_| std::ops::ControlFlow::Continue(()))
            .expect(
                "generate_kv_masked must now work for a nemotron_h Runner (spec 279 decode-step)",
            );
        assert!(
            kv_tokens.len() >= 3,
            "at least the 3 prompt tokens should come back"
        );
        assert!(
            kv_tokens.iter().all(|&t| (t as usize) < np.vocab_size),
            "every generated id must be a valid vocab index: {kv_tokens:?}"
        );

        // A direct finite/non-degenerate numeric check on the production-loaded weights via the same
        // dispatch generate/generate_sampled use (stateless_prefill_graph + bind), like
        // deepseek32_load.rs's fixture.
        let tokens3 = runner.encode("t1 t2 t3").expect("encode prompt");
        assert_eq!(tokens3.len(), 3);
        let g = runner.stateless_prefill_graph(tokens3.len()).unwrap();
        let inputs = runner
            .bind(&g, &tokens3)
            .expect("bind nemotron_h prefill graph via Runner::bind");
        let logits = crate::core::cpu_oracle::cpu_eval(&g, &inputs)
            .expect("eval nemotron_h prefill graph via Runner::bind");
        assert_eq!(logits.shape(), vec![1, 1, np.vocab_size]);
        assert!(
            logits.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "logits must be finite: {:?}",
            logits.as_f32().unwrap()
        );
        assert!(
            logits
                .as_f32()
                .unwrap()
                .iter()
                .any(|&v| v != logits.as_f32().unwrap()[0]),
            "logits must not be degenerate (all-equal)"
        );
    }

    /// Card 297 reachability receipt: a production-loaded Nemotron-H `Runner` generates on the wgpu GPU
    /// executor and produces the same tokens as the CPU oracle.
    ///
    /// `Runner::generate_gpu_reprefill` used to bail ("Nemotron-H has no GPU dispatch yet"), leaving
    /// `poot-serve`'s non-batched CPU spawn arm as the only way to run this arch (card 297's audit called
    /// it the family's biggest limitation). The bail was conservative, not structural: the Mamba2 mixer is
    /// a pure primitive composition, and `crates/poot-gpu/tests/nemotron_h.rs` shows every eqn of both
    /// Nemotron-H graphs plans to a SPIR-V kernel and matches the CPU oracle on RADV hardware. This test
    /// is the end-to-end half: the full production path (`Runner::load` ->
    /// `generate_gpu_reprefill` -> `stateless_prefill_graph`'s nemotron_h arm -> `Runner::bind`'s
    /// mask slot arm -> `GpuExecutor::run`), not a hand-built graph.
    ///
    /// The assertion is token identity against `Runner::generate`, not finiteness: both paths use a
    /// greedy sampler over the same weights, so a GPU/CPU divergence large enough to flip an argmax fails,
    /// while a finiteness check would pass on almost any wrong answer. Skips with no Vulkan adapter (a
    /// skip, never hardware coverage).
    #[test]
    fn runner_nemotron_h_gpu_reprefill_matches_cpu_tokens() {
        let dir = write_tiny_nemotron_h_checkpoint();
        let runner = Runner::load(&dir).expect("load synthetic nemotron_h checkpoint");
        let device = match poot_gpu::device::WgpuDevice::new() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP runner_nemotron_h_gpu_reprefill_matches_cpu_tokens (no GPU: {e})");
                return;
            }
        };
        let mut engine = poot_executor::Engine::new(device);
        let exe = runner.load_on(&mut engine).expect("load_on");

        let cpu_tokens = runner
            .generate("t1 t2 t3", 3, |_| std::ops::ControlFlow::Continue(()))
            .expect("CPU re-prefill generate on synthetic nemotron_h checkpoint");
        let gpu_tokens = runner
            .generate_gpu_reprefill("t1 t2 t3", 3, &mut engine, exe, |_| {
                std::ops::ControlFlow::Continue(())
            })
            .expect("generate_gpu_reprefill must now work for a nemotron_h Runner (card 297)");

        assert_eq!(
            gpu_tokens, cpu_tokens,
            "GPU re-prefill must produce the same greedy tokens as the CPU oracle"
        );
        assert!(
            gpu_tokens.len() > 3,
            "at least one token must have been generated past the 3-token prompt, \
             otherwise this compares two empty generations: {gpu_tokens:?}"
        );
    }
}

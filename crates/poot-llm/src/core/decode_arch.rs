//! The single per-architecture fixed-KV decode dispatch for the families still on the Runner.
//!
//! One registry maps each architecture family to its trace: [`DecodeArch`] classifies a [`Runner`]
//! ([`Runner::decode_arch`]) and [`Runner::decode_masked_graph`] returns that arch's fixed-KV masked
//! decode graph. Both are exhaustive over the enum, so adding an architecture is one enum variant and
//! the arms the compiler demands; there is no fall-through arm. The dense families are registered
//! families and run on the driver (POOT-737); the MoE and hybrid families here move onto `Model` with
//! POOT-738, which deletes this file.

use poot_graph_ir::Graph;
use poot_models::deepseek2::trace_deepseek2_decode_kv_masked;
use poot_models::deepseek3::trace_deepseek3_decode_kv_masked;
use poot_models::deepseek32::trace_deepseek32_dsa_decode;
use poot_models::gpt_oss::trace_gptoss_decode_kv_masked;
use poot_models::granite::trace_granite_decode_kv_masked;
use poot_models::mixtral::trace_mixtral_decode_kv_masked;
use poot_models::nemotron_h::trace_nemotron_h_decode;
use poot_models::olmoe::trace_olmoe_decode_kv_masked;
use poot_models::qwen3moe::trace_qwen3_moe_decode_kv_masked;

use crate::core::runner::Runner;
use crate::error::Result;

/// Every architecture family still on the Runner, one variant per fixed-KV masked decode tracer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodeArch {
    GraniteMoe,
    Qwen3Moe,
    Mixtral,
    Olmoe,
    GptOss,
    DeepseekV2,
    DeepseekV3,
    DeepseekV32,
    NemotronH,
}

/// Architectures with a contiguous fixed-capacity KV prefill tracer. Deliberately narrower than [`DecodeArch`]:
/// having a decode graph does not imply a one-shot prefill graph can produce its carried state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ContiguousPrefillArch {
    GraniteMoe,
    Qwen3Moe,
}

/// Architectures with a matched shared-pool prefill and batched-decode pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum SharedPoolArch {
    GraniteMoe,
    Qwen3Moe,
    Mixtral,
    DeepseekV3,
}

impl DecodeArch {
    /// Every variant, in [`Self::index`] order. Kept complete by
    /// `decode_arch_tests::decode_arch_all_lists_every_variant_once`, which cross-checks it against the exhaustive
    /// [`Self::index`] match: a new variant fails to compile there first, then fails that test until listed here.
    ///
    /// `cfg(test)`: the enumeration exists for the drift guard; production dispatch uses the exhaustive matches.
    #[cfg(test)]
    pub(crate) const ALL: &'static [DecodeArch] = &[
        DecodeArch::GraniteMoe,
        DecodeArch::Qwen3Moe,
        DecodeArch::Mixtral,
        DecodeArch::Olmoe,
        DecodeArch::GptOss,
        DecodeArch::DeepseekV2,
        DecodeArch::DeepseekV3,
        DecodeArch::DeepseekV32,
        DecodeArch::NemotronH,
    ];

    /// Dense 0-based index into [`Self::ALL`]. Exhaustive on purpose (no `_` arm): a new variant needs an index here to build.
    #[cfg(test)]
    pub(crate) fn index(self) -> usize {
        match self {
            DecodeArch::GraniteMoe => 0,
            DecodeArch::Qwen3Moe => 1,
            DecodeArch::Mixtral => 2,
            DecodeArch::Olmoe => 3,
            DecodeArch::GptOss => 4,
            DecodeArch::DeepseekV2 => 5,
            DecodeArch::DeepseekV3 => 6,
            DecodeArch::DeepseekV32 => 7,
            DecodeArch::NemotronH => 8,
        }
    }
}

impl Runner {
    /// Which decode tracer family this `Runner` needs. The single classifier: every dispatch site asks this
    /// instead of re-deriving it from the `Option` fields. The loaders set exactly one family field; a
    /// Runner with none has no tracer and is refused, never given a default family.
    pub(crate) fn decode_arch(&self) -> Result<DecodeArch> {
        Ok(if self.granite_moe.is_some() {
            DecodeArch::GraniteMoe
        } else if self.qwen3_moe.is_some() {
            DecodeArch::Qwen3Moe
        } else if self.mixtral.is_some() {
            DecodeArch::Mixtral
        } else if self.olmoe.is_some() {
            DecodeArch::Olmoe
        } else if self.gpt_oss.is_some() {
            DecodeArch::GptOss
        } else if self.deepseek2.is_some() {
            DecodeArch::DeepseekV2
        } else if self.deepseek3.is_some() {
            DecodeArch::DeepseekV3
        } else if self.deepseek32.is_some() {
            DecodeArch::DeepseekV32
        } else if self.nemotron_h.is_some() {
            DecodeArch::NemotronH
        } else {
            bail!(
                "this Runner carries no architecture family ({}): it has no tracer",
                self.arch
            )
        })
    }

    /// Classify the Runner for contiguous fixed-KV prefill. Every [`DecodeArch`] arm is explicit so a new decode
    /// family cannot silently inherit another family's prefill.
    pub(crate) fn contiguous_prefill_arch(&self) -> Result<ContiguousPrefillArch> {
        let arch = self.decode_arch()?;
        let supported = match arch {
            DecodeArch::GraniteMoe => ContiguousPrefillArch::GraniteMoe,
            DecodeArch::Qwen3Moe => ContiguousPrefillArch::Qwen3Moe,
            DecodeArch::Mixtral
            | DecodeArch::Olmoe
            | DecodeArch::GptOss
            | DecodeArch::DeepseekV2
            | DecodeArch::DeepseekV3
            | DecodeArch::DeepseekV32
            | DecodeArch::NemotronH => {
                bail!("contiguous fixed-KV prefill is unsupported for architecture {arch:?}")
            }
        };
        Ok(supported)
    }

    /// Classify the Runner for the matched shared-pool prefill/decode contract. Both builders consume this result,
    /// so capability admission and rejection cannot drift between the two halves.
    #[cfg(test)]
    pub(crate) fn shared_pool_arch(&self) -> Result<SharedPoolArch> {
        let arch = self.decode_arch()?;
        let supported = match arch {
            DecodeArch::GraniteMoe => SharedPoolArch::GraniteMoe,
            DecodeArch::Qwen3Moe => SharedPoolArch::Qwen3Moe,
            DecodeArch::Mixtral => SharedPoolArch::Mixtral,
            DecodeArch::DeepseekV3 => SharedPoolArch::DeepseekV3,
            DecodeArch::Olmoe
            | DecodeArch::GptOss
            | DecodeArch::DeepseekV2
            | DecodeArch::DeepseekV32
            | DecodeArch::NemotronH => {
                bail!("shared-pool prefill/decode is unsupported for architecture {arch:?}")
            }
        };
        Ok(supported)
    }

    /// The per-architecture fixed-KV masked decode graph at capacity `cap`. Every decode dispatch in the crate
    /// routes here.
    ///
    /// The match is exhaustive over [`DecodeArch`] (no fall-through arm), so an unregistered architecture is a
    /// compile error rather than a silently wrong graph.
    ///
    /// The traced graph leaves through [`Runner::bind_storage`], so a packed checkpoint's weights reach it as
    /// their claimed decodes (card 545a).
    pub(crate) fn decode_masked_graph(&self, cap: usize) -> Result<Graph> {
        let g = match self.decode_arch()? {
            DecodeArch::GraniteMoe => {
                let grp = self
                    .granite_moe
                    .expect("DecodeArch::GraniteMoe implies granite params");
                trace_granite_decode_kv_masked(self.cfg, grp, cap)
            }
            DecodeArch::Qwen3Moe => {
                let mp = self
                    .qwen3_moe
                    .as_ref()
                    .expect("DecodeArch::Qwen3Moe implies qwen3_moe params");
                trace_qwen3_moe_decode_kv_masked(self.cfg, mp.clone(), cap)
            }
            DecodeArch::Mixtral => {
                let mp = self
                    .mixtral
                    .expect("DecodeArch::Mixtral implies mixtral params");
                trace_mixtral_decode_kv_masked(self.cfg, mp, cap)
            }
            DecodeArch::Olmoe => {
                let op = self.olmoe.expect("DecodeArch::Olmoe implies olmoe params");
                trace_olmoe_decode_kv_masked(self.cfg, op, cap)
            }
            DecodeArch::GptOss => {
                let gp = self
                    .gpt_oss
                    .as_ref()
                    .expect("DecodeArch::GptOss implies gpt_oss params");
                trace_gptoss_decode_kv_masked(self.cfg, gp.clone(), cap)
            }
            DecodeArch::DeepseekV2 => {
                let dp = self
                    .deepseek2
                    .expect("DecodeArch::DeepseekV2 implies deepseek2 params");
                trace_deepseek2_decode_kv_masked(dp.cfg, dp.moe, cap)
            }
            DecodeArch::DeepseekV3 => {
                let dp = self
                    .deepseek3
                    .expect("DecodeArch::DeepseekV3 implies deepseek3 params");
                trace_deepseek3_decode_kv_masked(dp.cfg, dp.moe, cap)
            }
            // DSA: DeepSeek-V3.2's fixed-KV decode tracer. Adds a third per-layer carried state tensor beyond MLA's pair
            // (the indexer's decoupled key cache); callers' cache plumbing is generic over `g.state`'s length.
            DecodeArch::DeepseekV32 => {
                let dp = self
                    .deepseek32
                    .expect("DecodeArch::DeepseekV32 implies deepseek32 params");
                trace_deepseek32_dsa_decode(dp.cfg, dp.dsa, dp.moe, cap)
            }
            // Nemotron-H's fixed-KV decode tracer.
            DecodeArch::NemotronH => {
                let ncfg = self
                    .nemotron_h
                    .as_ref()
                    .expect("DecodeArch::NemotronH implies nemotron_h config");
                trace_nemotron_h_decode(ncfg, cap)
            }
        };
        self.bind_storage(g)
    }
}

/// Synthetic `Runner`s, one per [`DecodeArch`], for the dispatch tests. Built by direct struct construction:
/// private fields are visible in the same module tree, so no GGUF,
/// checkpoint or GPU is needed.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::DecodeArch;
    use poot_models::deepseek2::{DeepseekV2Config, DeepseekV2MoeParams, DeepseekV2Params};
    use poot_models::deepseek3::{DeepseekV3MoeParams, DeepseekV3Params};
    use poot_models::deepseek32::{Deepseek32Params, DsaConfig};
    use poot_models::gpt_oss::GptOssParams;
    use poot_models::granite::{GraniteParams, MoeShape};
    use poot_models::mixtral::MixtralParams;
    use poot_models::nemotron_h::{NemotronHAttnConfig, NemotronHConfig, NemotronHMambaConfig};
    use poot_models::olmoe::OlmoeParams;
    use poot_models::qwen3moe::Qwen3MoeParams;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    use crate::core::runner::{LoraHotState, Runner};

    /// Shared tiny traceable qwen2-shaped config for the fixtures that read `Runner::cfg` (granitemoe,
    /// qwen3_moe, mixtral, olmoe, gpt-oss). Copied from `poot_models::gpt_oss`'s `tiny_cfg`: non-degenerate,
    /// GQA, head_dim decoupled from hidden/n_heads, and a real `rotary_dim`/`max_pos` (`Qwen2Config::default()`
    /// leaves both at 0, so every RoPE tracer would slice a zero-length axis).
    const LAYERS: usize = 3;

    fn base_cfg() -> poot_models::qwen2::Qwen2Config {
        poot_models::qwen2::Qwen2Config {
            vocab: 24,
            hidden: 16,
            inter: 16,
            layers: LAYERS,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 6,
            rotary_dim: 6,
            eps: 1e-5,
            max_pos: 32,
            qkv_bias: false,
            qk_norm: false,
            ..poot_models::qwen2::Qwen2Config::default()
        }
    }

    /// DeepSeek-family MLA dims copied from `poot_models::deepseek2`'s `tiny_cfg`: non-degenerate (v_head_dim
    /// differs from qk_nope+qk_rope, qk_rope_head_dim even).
    fn deepseek_cfg(q_lora_rank: Option<usize>) -> DeepseekV2Config {
        DeepseekV2Config {
            vocab: 24,
            hidden: 16,
            layers: 3,
            n_heads: 4,
            q_lora_rank,
            kv_lora_rank: 6,
            qk_nope_head_dim: 5,
            qk_rope_head_dim: 4,
            v_head_dim: 7,
            eps: 1e-5,
            max_pos: 32,
            rope_theta: 10_000.0,
            yarn: None,
        }
    }

    /// V3-family MoE shape, from `poot_models::deepseek3`'s own `tiny_mp`.
    fn deepseek_moe() -> DeepseekV3MoeParams {
        DeepseekV3MoeParams {
            n_routed_experts: 8,
            top_k: 3,
            moe_inter: 6,
            n_shared_experts: 1,
            dense_inter: 10,
            first_k_dense_replace: 2,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
        }
    }

    /// A minimal `Runner` whose only non-inert state is the one architecture selector `arch` names.
    pub(crate) fn runner_for(arch: DecodeArch) -> Runner {
        let model = WordLevel::builder()
            .vocab(std::collections::HashMap::new())
            .unk_token("<unk>".to_string())
            .build()
            .expect("build empty wordlevel model");
        Runner {
            cfg: base_cfg(),
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
            granite_moe: (arch == DecodeArch::GraniteMoe).then_some(GraniteParams {
                moe: Some(MoeShape {
                    n_experts: 4,
                    top_k: 2,
                    inter: 12,
                }),
                embed_mult: 1.0,
                attn_mult: 1.0,
                residual_mult: 1.0,
                logits_scale: 1.0,
            }),
            qwen3_moe: (arch == DecodeArch::Qwen3Moe).then_some(Qwen3MoeParams {
                n_experts: 6,
                top_k: 2,
                inter: 12,
                sparse_layer: vec![false, true, true],
            }),
            mixtral: (arch == DecodeArch::Mixtral).then_some(MixtralParams {
                n_experts: 6,
                top_k: 2,
                inter: 12,
            }),
            olmoe: (arch == DecodeArch::Olmoe).then_some(OlmoeParams {
                n_experts: 6,
                top_k: 2,
                inter: 12,
                norm_topk_prob: false,
            }),
            gpt_oss: (arch == DecodeArch::GptOss).then_some(GptOssParams {
                n_experts: 6,
                top_k: 2,
                inter: 12,
                swiglu_limit: 2.5,
                sliding_window: 2,
                layer_is_sliding: vec![true, false, true],
            }),
            deepseek2: (arch == DecodeArch::DeepseekV2).then_some(DeepseekV2Params {
                cfg: deepseek_cfg(Some(3)),
                // `poot_models::deepseek2`'s own `tiny_mp` shape.
                moe: DeepseekV2MoeParams {
                    n_routed_experts: 6,
                    top_k: 2,
                    moe_inter: 8,
                    n_shared_experts: 2,
                    dense_inter: 10,
                    first_k_dense_replace: 1,
                    n_group: 1,
                    topk_group: 1,
                    routed_scaling_factor: 1.0,
                },
            }),
            deepseek3: (arch == DecodeArch::DeepseekV3).then_some(DeepseekV3Params {
                cfg: deepseek_cfg(Some(3)),
                moe: deepseek_moe(),
            }),
            deepseek32: (arch == DecodeArch::DeepseekV32).then_some(Deepseek32Params {
                // real DeepSeek-V3.2 always sets q_lora_rank - DSA's indexer wq_b consumes MLA's qr,
                // which only exists on that branch (`poot_models::deepseek32`'s own note).
                cfg: deepseek_cfg(Some(5)),
                moe: deepseek_moe(),
                dsa: DsaConfig {
                    index_n_heads: 2,
                    index_head_dim: 4,
                    index_topk: 3,
                },
            }),
            nemotron_h: (arch == DecodeArch::NemotronH).then_some(NemotronHConfig {
                // `poot_models::nemotron_h`'s own toy dims, with all three layer kinds present.
                vocab_size: 12,
                hidden: 6,
                mlp_inter: 10,
                eps: 1e-6,
                pattern: vec![
                    poot_models::nemotron_h::NemotronHLayerKind::Mamba,
                    poot_models::nemotron_h::NemotronHLayerKind::Attention,
                    poot_models::nemotron_h::NemotronHLayerKind::Mlp,
                ],
                mamba: NemotronHMambaConfig {
                    hidden: 6,
                    mamba_num_heads: 4,
                    mamba_head_dim: 2,
                    n_groups: 2,
                    ssm_state: 3,
                    conv_kernel: 3,
                },
                attn: NemotronHAttnConfig {
                    hidden: 6,
                    num_heads: 4,
                    num_kv_heads: 2,
                    head_dim: 2,
                },
            }),
            formats: Default::default(),
            sliding_window: None,
            lora_hot: std::sync::RwLock::new(LoraHotState::default()),
        }
    }
}

/// The drift guard for the dispatch: an architecture that has its own decode tracer but whose arm resolves to
/// the plain qwen2 `trace_decode_kv_masked` fall-through would build a graph for a different model and fail
/// later in weight binding. Enumerating [`DecodeArch::ALL`] rather than checking one arch means a future arm
/// cannot be added to one path and forgotten in another.
#[cfg(test)]
mod decode_arch_tests {
    use super::fixtures::runner_for;
    use super::*;
    use poot_models::deepseek2::trace_deepseek2_prefill;
    use poot_models::deepseek3::{
        trace_deepseek3_decode_kv_masked_batched_shared_pool, trace_deepseek3_prefill,
        trace_deepseek3_prefill_kv_shared_pool,
    };
    use poot_models::deepseek32::trace_deepseek32_dsa_prefill;
    use poot_models::gpt_oss::trace_gptoss_prefill;
    use poot_models::granite::{
        trace_granite_decode_kv_masked_batched_shared_pool, trace_granite_prefill,
        trace_granite_prefill_kv, trace_granite_prefill_kv_shared_pool,
    };
    use poot_models::mixtral::{
        trace_mixtral_decode_kv_masked_batched_shared_pool, trace_mixtral_prefill,
        trace_mixtral_prefill_kv_shared_pool,
    };
    use poot_models::nemotron_h::trace_nemotron_h_prefill;
    use poot_models::olmoe::trace_olmoe_prefill;
    use poot_models::qwen3moe::{
        trace_qwen3_moe_decode_kv_masked_batched_shared_pool, trace_qwen3_moe_prefill,
        trace_qwen3_moe_prefill_kv, trace_qwen3_moe_prefill_kv_shared_pool,
    };

    const N: usize = 8;
    const CAP: usize = 12;
    const POOL: usize = 16;

    #[derive(Clone, Copy)]
    struct PrefillCase {
        arch: DecodeArch,
        contiguous: bool,
        shared_pool: bool,
    }

    /// One row per architecture (the acceptance table for card 318): adding a family means adding one
    /// row, not another copied dispatch test.
    const PREFILL_CASES: &[PrefillCase] = &[
        PrefillCase {
            arch: DecodeArch::GraniteMoe,
            contiguous: true,
            shared_pool: true,
        },
        PrefillCase {
            arch: DecodeArch::Qwen3Moe,
            contiguous: true,
            shared_pool: true,
        },
        PrefillCase {
            arch: DecodeArch::Mixtral,
            contiguous: false,
            shared_pool: true,
        },
        PrefillCase {
            arch: DecodeArch::Olmoe,
            contiguous: false,
            shared_pool: false,
        },
        PrefillCase {
            arch: DecodeArch::GptOss,
            contiguous: false,
            shared_pool: false,
        },
        PrefillCase {
            arch: DecodeArch::DeepseekV2,
            contiguous: false,
            shared_pool: false,
        },
        PrefillCase {
            arch: DecodeArch::DeepseekV3,
            contiguous: false,
            shared_pool: true,
        },
        PrefillCase {
            arch: DecodeArch::DeepseekV32,
            contiguous: false,
            shared_pool: false,
        },
        PrefillCase {
            arch: DecodeArch::NemotronH,
            contiguous: false,
            shared_pool: false,
        },
    ];

    fn expected_stateless(runner: &Runner, arch: DecodeArch) -> Graph {
        match arch {
            DecodeArch::GraniteMoe => {
                trace_granite_prefill(runner.cfg, runner.granite_moe.unwrap(), N)
            }
            DecodeArch::Qwen3Moe => {
                trace_qwen3_moe_prefill(runner.cfg, runner.qwen3_moe.clone().unwrap(), N)
            }
            DecodeArch::Mixtral => trace_mixtral_prefill(runner.cfg, runner.mixtral.unwrap(), N),
            DecodeArch::Olmoe => trace_olmoe_prefill(runner.cfg, runner.olmoe.unwrap(), N),
            DecodeArch::GptOss => {
                trace_gptoss_prefill(runner.cfg, runner.gpt_oss.clone().unwrap(), N)
            }
            DecodeArch::DeepseekV2 => {
                let params = runner.deepseek2.unwrap();
                trace_deepseek2_prefill(params.cfg, params.moe, N)
            }
            DecodeArch::DeepseekV3 => {
                let params = runner.deepseek3.unwrap();
                trace_deepseek3_prefill(params.cfg, params.moe, N)
            }
            DecodeArch::DeepseekV32 => {
                let params = runner.deepseek32.unwrap();
                trace_deepseek32_dsa_prefill(params.cfg, params.dsa, params.moe, N)
            }
            DecodeArch::NemotronH => {
                trace_nemotron_h_prefill(runner.nemotron_h.as_ref().unwrap(), N)
            }
        }
    }

    fn state_signature(g: &Graph) -> Vec<(String, String)> {
        g.state
            .iter()
            .map(|&(state_in, state_out)| {
                let input = g.meta(state_in);
                let output = g.meta(state_out);
                (
                    format!(
                        "{}:{:?}",
                        input.name.as_deref().unwrap_or("<unnamed>"),
                        input.aval
                    ),
                    format!(
                        "{}:{:?}",
                        output.name.as_deref().unwrap_or("<unnamed>"),
                        output.aval
                    ),
                )
            })
            .collect()
    }

    fn assert_state_handoff(prefill: &Graph, decode: &Graph, label: &str) {
        let prefill_state = state_signature(prefill);
        let decode_state = state_signature(decode);
        assert_eq!(
            prefill_state, decode_state,
            "{label} ordered state input/output name, dtype, and shape contract"
        );
        for (index, (&(_, prefill_output), &(decode_input, _))) in
            prefill.state.iter().zip(&decode.state).enumerate()
        {
            let prefill_output_aval = prefill.aval(prefill_output);
            let decode_input_aval = decode.aval(decode_input);
            assert_eq!(
                prefill_output_aval, decode_input_aval,
                "{label} state {index} output-to-input layout"
            );
        }
    }

    /// A stable structural fingerprint of a graph. `Graph` has no `PartialEq`, and its `Debug` (the full value/eqn
    /// listing) identifies which tracer built it. Hashed to a `u64` so a failing assertion prints a short line.
    fn fingerprint(g: &poot_graph_ir::Graph) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{g:?}").hash(&mut h);
        h.finish()
    }

    /// `ALL` is every variant. `index()` is an exhaustive match, so a new variant fails to compile there first;
    /// this then fails until it is listed in `ALL` too.
    #[test]
    fn decode_arch_all_lists_every_variant_once() {
        let mut seen = vec![false; DecodeArch::ALL.len()];
        for &a in DecodeArch::ALL {
            let i = a.index();
            assert!(
                i < seen.len(),
                "{a:?} has index {i} but DecodeArch::ALL has only {} entries - add it to ALL",
                seen.len()
            );
            assert!(!seen[i], "{a:?} appears twice in DecodeArch::ALL");
            seen[i] = true;
            assert_eq!(
                DecodeArch::ALL[i],
                a,
                "DecodeArch::ALL is not in index order"
            );
        }
        assert!(
            seen.iter().all(|&s| s),
            "DecodeArch::ALL is missing a variant (index() assigns an index no ALL entry uses)"
        );
    }

    /// Every architecture classifies as itself: the fixture that sets only one selector field must be recognised
    /// as that arch, or a family silently routes to another family's tracer.
    #[test]
    fn every_arch_classifies_as_itself() {
        for &a in DecodeArch::ALL {
            assert_eq!(
                runner_for(a).decode_arch().unwrap(),
                a,
                "a Runner built for {a:?} does not classify as {a:?}"
            );
        }
    }

    /// A Runner that carries no family field has no tracer: every dispatch refuses it instead of
    /// tracing a default family's graph. Mutation: give `decode_arch` a default arm; the graph builds
    /// and the row fails.
    #[test]
    fn a_runner_without_a_family_is_refused() {
        let mut runner = runner_for(DecodeArch::Mixtral);
        runner.mixtral = None;
        let error = runner.decode_masked_graph(CAP).unwrap_err().to_string();
        assert!(error.contains("no architecture family"), "{error}");
        assert!(runner.stateless_prefill_graph(N).is_err());
    }

    /// Each architecture gets a distinct graph: no two arms may resolve to the same tracer, which is how a
    /// copy-paste arm ("added DeepSeek-V4, pasted DSA's tracer") slips past a test that only checks "not the
    /// fallback".
    #[test]
    fn every_arch_gets_a_distinct_decode_graph() {
        let mut seen: Vec<(DecodeArch, u64)> = Vec::new();
        for &a in DecodeArch::ALL {
            let g = fingerprint(&runner_for(a).decode_masked_graph(CAP).unwrap());
            if let Some((other, _)) = seen.iter().find(|(_, s)| *s == g) {
                panic!("{a:?} and {other:?} dispatch to the SAME decode graph");
            }
            seen.push((a, g));
        }
    }

    #[test]
    fn card318_prefill_table_covers_every_arch_and_stateless_source() {
        for &arch in DecodeArch::ALL {
            let count = PREFILL_CASES
                .iter()
                .filter(|case| case.arch == arch)
                .count();
            assert_eq!(count, 1, "{arch:?} must have exactly one prefill table row");
        }

        for &case in PREFILL_CASES {
            let runner = runner_for(case.arch);
            let actual = runner.stateless_prefill_graph(N).unwrap();
            let expected = expected_stateless(&runner, case.arch);
            assert_eq!(
                fingerprint(&actual),
                fingerprint(&expected),
                "{:?} stateless prefill selected the wrong tracer source",
                case.arch
            );
        }
    }

    #[test]
    fn card318_contiguous_prefill_table_has_explicit_outcomes_and_state_handoff() {
        for &case in PREFILL_CASES {
            let runner = runner_for(case.arch);
            let result = runner.prefill_kv_graph(N, CAP);
            if !case.contiguous {
                let error =
                    result.expect_err("unsupported contiguous architecture returned a graph");
                let message = error.to_string();
                assert!(message.contains("contiguous fixed-KV prefill"), "{message}");
                assert!(message.contains(&format!("{:?}", case.arch)), "{message}");
                continue;
            }

            let actual = result.expect("supported contiguous architecture returned an error");
            let expected = match case.arch {
                DecodeArch::GraniteMoe => {
                    trace_granite_prefill_kv(runner.cfg, runner.granite_moe.unwrap(), N, CAP)
                }
                DecodeArch::Qwen3Moe => trace_qwen3_moe_prefill_kv(
                    runner.cfg,
                    runner.qwen3_moe.clone().unwrap(),
                    N,
                    CAP,
                ),
                other => panic!("table marks unsupported {other:?} as contiguous"),
            };
            actual
                .validate()
                .expect("contiguous prefill graph validates");
            assert_eq!(
                fingerprint(&actual),
                fingerprint(&expected),
                "{:?} contiguous prefill selected the wrong tracer source",
                case.arch
            );
            assert_state_handoff(
                &actual,
                &runner.decode_masked_graph(CAP).unwrap(),
                &format!("{:?} contiguous", case.arch),
            );
        }
    }

    #[test]
    fn card318_shared_pool_table_has_explicit_paired_outcomes_and_state_handoff() {
        for &case in PREFILL_CASES {
            let runner = runner_for(case.arch);
            let prefill = runner.prefill_kv_graph_shared_pool(N, POOL);
            let decode = runner.decode_kv_graph_shared_pool(CAP, 1, POOL, false);
            if !case.shared_pool {
                for (mode, result) in [("prefill", prefill), ("decode", decode)] {
                    let error =
                        result.expect_err("unsupported shared-pool architecture returned a graph");
                    let message = error.to_string();
                    assert!(message.contains("shared-pool prefill/decode"), "{message}");
                    assert!(
                        message.contains(&format!("{:?}", case.arch)),
                        "{mode} error did not name {:?}: {message}",
                        case.arch
                    );
                }
                continue;
            }

            let prefill = prefill.expect("supported shared-pool prefill returned an error");
            let decode = decode.expect("supported shared-pool decode returned an error");
            let (expected_prefill, expected_decode) = match case.arch {
                DecodeArch::GraniteMoe => {
                    let params = runner.granite_moe.unwrap();
                    (
                        trace_granite_prefill_kv_shared_pool(runner.cfg, params, N, POOL),
                        trace_granite_decode_kv_masked_batched_shared_pool(
                            runner.cfg, params, CAP, 1, POOL,
                        ),
                    )
                }
                DecodeArch::Qwen3Moe => {
                    let params = runner.qwen3_moe.clone().unwrap();
                    (
                        trace_qwen3_moe_prefill_kv_shared_pool(runner.cfg, params.clone(), N, POOL),
                        trace_qwen3_moe_decode_kv_masked_batched_shared_pool(
                            runner.cfg, params, CAP, 1, POOL,
                        ),
                    )
                }
                DecodeArch::Mixtral => {
                    let params = runner.mixtral.unwrap();
                    (
                        trace_mixtral_prefill_kv_shared_pool(runner.cfg, params, N, POOL),
                        trace_mixtral_decode_kv_masked_batched_shared_pool(
                            runner.cfg, params, CAP, 1, POOL,
                        ),
                    )
                }
                DecodeArch::DeepseekV3 => {
                    let params = runner.deepseek3.unwrap();
                    (
                        trace_deepseek3_prefill_kv_shared_pool(params.cfg, params.moe, N, POOL),
                        trace_deepseek3_decode_kv_masked_batched_shared_pool(
                            params.cfg, params.moe, CAP, 1, POOL,
                        ),
                    )
                }
                other => panic!("table marks unsupported {other:?} as shared-pool"),
            };
            prefill.validate().expect("shared-pool prefill validates");
            decode.validate().expect("shared-pool decode validates");
            assert_eq!(
                fingerprint(&prefill),
                fingerprint(&expected_prefill),
                "{:?} shared-pool prefill selected the wrong tracer source",
                case.arch
            );
            assert_eq!(
                fingerprint(&decode),
                fingerprint(&expected_decode),
                "{:?} shared-pool decode selected the wrong tracer source",
                case.arch
            );
            assert_state_handoff(&prefill, &decode, &format!("{:?} shared-pool", case.arch));
        }
    }
}

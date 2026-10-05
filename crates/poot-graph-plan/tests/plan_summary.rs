//! Card 599: a committed plan-summary dump over a named fixture corpus.
//!
//! Cards that change plans on purpose (or by accident) gate on "every fixture graph plans with unchanged
//! dispatch counts". This test is that gate. For each graph in [`corpus`] and each target it compiles
//! through [`compile`], the one graph-to-program entry point every executor runs (ADR-0100 decision 1):
//! no hand-rolled sub-pipeline here duplicates what `compile` already does (card 523b review: that
//! duplication is the exact "per-executor pre-plan pipelines drift" this epic removes). A
//! graph `compile` accepts gets one summary line per equation (graph, equation index, op, plan kind,
//! kernel choice, a digest of its request, kernel launches, launch grid, detail); a graph `compile` refuses gets exactly one summary
//! line for the whole graph, carrying the typed [`CompileError`] as `compile` reports it - never a
//! partial per-equation dump of an intermediate pipeline stage `compile` itself would still rewrite.
//! The lines are compared with the checked-in `tests/plan_summary/<target>.tsv`.
//!
//! A raw hash of the plans would change whenever a plan type changes; a line per equation names what
//! moved. A card that changes plans by design regenerates the files with `just test-plan-summary-regen`
//! and lists every changed line in its landing note.
//!
//! The corpus needs no checkpoint: every graph is a tracer from `poot-models` at tiny fixed dims.
//! [`corpus`] names what it holds and what it leaves out.
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_executor_parity::weight_map::Projections;
use poot_graph_ir::Graph;
use poot_graph_plan::{
    CompileError, CompileOptions, FusionPolicy, KernelChoice, Plan, PlanError, Submission, Target,
    compile,
};
use poot_models::bloom::{BloomConfig, trace_bloom_decode_kv_masked, trace_bloom_prefill};
use poot_models::deepseek2::{
    DeepseekV2Config, DeepseekV2MoeParams, trace_deepseek2_decode_kv_masked,
    trace_deepseek2_prefill,
};
use poot_models::deepseek3::{
    DeepseekV3MoeParams, trace_deepseek3_decode_kv_masked, trace_deepseek3_prefill,
};
use poot_models::deepseek32::{
    DsaConfig, trace_deepseek32_dsa_decode, trace_deepseek32_dsa_prefill,
};
use poot_models::gpt_oss::{GptOssParams, trace_gptoss_decode_kv_masked, trace_gptoss_prefill};
use poot_models::granite::{
    GraniteParams, MoeShape, trace_granite_decode_kv_masked, trace_granite_prefill,
    trace_granite_prefill_kv,
};
use poot_models::mixtral::{MixtralParams, trace_mixtral_decode_kv_masked, trace_mixtral_prefill};
use poot_models::model::{LogitRows, Phase};
use poot_models::mpt::{MptConfig, trace_mpt_decode_kv_masked, trace_mpt_prefill};
use poot_models::nemotron_h::{
    NemotronHAttnConfig, NemotronHConfig, NemotronHLayerKind, NemotronHMambaConfig,
    trace_nemotron_h_decode, trace_nemotron_h_prefill,
};
use poot_models::olmoe::{OlmoeParams, trace_olmoe_decode_kv_masked, trace_olmoe_prefill};
use poot_models::qwen2::Qwen2Config;
use poot_models::qwen3moe::{
    Qwen3MoeParams, trace_qwen3_moe_decode_kv_masked, trace_qwen3_moe_prefill,
    trace_qwen3_moe_prefill_kv,
};
use poot_models::qwen38::{
    QsaConfig, Qwen4ExpConfig, Qwen4ExpGdnConfig, Qwen4ExpModelConfig, trace_qwen38_decode,
    trace_qwen38_prefill,
};
use poot_models::smollm3::{Smollm3Config, trace_smollm3_decode_kv_masked, trace_smollm3_prefill};
use poot_target::AmdArch;
use poot_target::Backend;

/// Layers in every fixture: two is the least that has a first and a later layer (dense then sparse,
/// local then global) while keeping the expected files small.
const LAYERS: usize = 2;
/// Decode KV capacity, and the prefill prompt length and cache capacity.
const CAP: usize = 8;
const PROMPT: usize = 4;
const BATCH: usize = 2;

/// The environment variable that turns the comparison into a rewrite of the expected files. The
/// `test-plan-summary-regen` recipe sets it.
const REGEN_ENV: &str = "POOT_PLAN_SUMMARY_REGEN";

/// The three planner targets, each with the file its summary is checked in as.
fn targets() -> [(&'static str, Backend); 3] {
    [
        ("spirv_vulkan", Backend::SpirvVulkan),
        ("amd_gcn", Backend::AmdGcn(AmdArch::gfx1151())),
        ("nvptx", Backend::Nvptx),
    ]
}

/// One row of the corpus: the name the summary lines carry, and a builder (graphs are rebuilt per run so a
/// planner pass cannot leak state from one target to the next).
struct Row {
    name: &'static str,
    build: fn() -> Graph,
}

// ---- tiny dims -------------------------------------------------------------------------------

/// The shared tiny qwen2-shaped config for the fixtures that read a `Qwen2Config`: GQA, head_dim
/// decoupled from hidden/n_heads, and a real `rotary_dim` and `max_pos`.
fn base_cfg() -> Qwen2Config {
    Qwen2Config {
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
        ..Qwen2Config::default()
    }
}

/// The tiny dense checkpoint of `family` the registry rows trace: [`base_cfg`]'s dimensions through
/// the family's own config (the head dimension is explicit, as the family fixtures give it).
fn dense(family: Family) -> Dense {
    Dense::new(family)
        .vocab(24)
        .dims(16, 16, LAYERS)
        .heads(4, 2)
        .head_dim(6)
        .max_positions(32)
        .with("rms_norm_eps", 1e-5)
}

/// Gemma 3 as the old gemma rows configured it: a query scalar of one (attention scale 1), a
/// window of three on the local layers, every second layer global.
fn gemma() -> Dense {
    dense(Family::Gemma3)
        .with("query_pre_attn_scalar", 1.0)
        .with("sliding_window", 3)
        .with("sliding_window_pattern", 2)
}

/// `d`'s step: `rows` sequences of `tokens` new tokens over `cap` positions, last-token logits.
fn traced(d: Dense, phase: Phase, rows: usize, tokens: usize, cap: usize) -> Graph {
    plain(
        d.f32_model()
            .model
            .trace(phase, step(rows, tokens, cap, LogitRows::Last))
            .unwrap(),
    )
}

/// [`traced`] over a checkpoint whose projections and head are stored BF16, the embedding F32: the
/// mixed-width upload path.
fn traced_bf16(d: Dense, phase: Phase, tokens: usize, cap: usize) -> Graph {
    plain(
        d.build(Projections::Bf16, poot_tensor::DType::F32, None)
            .model
            .trace(phase, step(1, tokens, cap, LogitRows::Last))
            .unwrap(),
    )
}

fn granite_params(moe: Option<MoeShape>) -> GraniteParams {
    GraniteParams {
        moe,
        embed_mult: 1.0,
        attn_mult: 1.0,
        residual_mult: 1.0,
        logits_scale: 1.0,
    }
}

fn granite_moe() -> MoeShape {
    MoeShape {
        n_experts: 6,
        top_k: 2,
        inter: 12,
    }
}

fn qwen3_moe_params() -> Qwen3MoeParams {
    Qwen3MoeParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        sparse_layer: vec![false, true],
    }
}

fn mixtral_params() -> MixtralParams {
    MixtralParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
    }
}

fn olmoe_params() -> OlmoeParams {
    OlmoeParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        norm_topk_prob: false,
    }
}

fn gpt_oss_params() -> GptOssParams {
    GptOssParams {
        n_experts: 6,
        top_k: 2,
        inter: 12,
        swiglu_limit: 2.5,
        sliding_window: 2,
        layer_is_sliding: vec![true, false],
    }
}

fn bloom_cfg() -> BloomConfig {
    BloomConfig {
        vocab: 24,
        hidden: 16,
        n_heads: 4,
        layers: LAYERS,
        ffn_inter: 16,
        eps: 1e-5,
    }
}

fn mpt_cfg() -> MptConfig {
    MptConfig {
        vocab: 24,
        hidden: 16,
        n_heads: 4,
        layers: LAYERS,
        ffn_inter: 16,
        eps: 1e-5,
    }
}

fn smollm3_cfg() -> Smollm3Config {
    Smollm3Config {
        vocab: 24,
        hidden: 16,
        inter: 16,
        layers: LAYERS,
        n_heads: 4,
        n_kv_heads: 2,
        eps: 1e-6,
        max_pos: 32,
        use_rope: vec![true, false],
    }
}

/// DeepSeek-family MLA dims: v_head_dim differs from qk_nope+qk_rope, qk_rope_head_dim is even.
fn deepseek_cfg(q_lora_rank: Option<usize>) -> DeepseekV2Config {
    DeepseekV2Config {
        vocab: 24,
        hidden: 16,
        layers: LAYERS,
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

/// One dense layer, then one routed layer.
fn deepseek2_moe() -> DeepseekV2MoeParams {
    DeepseekV2MoeParams {
        n_routed_experts: 6,
        top_k: 2,
        moe_inter: 8,
        n_shared_experts: 2,
        dense_inter: 10,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    }
}

fn deepseek3_moe() -> DeepseekV3MoeParams {
    DeepseekV3MoeParams {
        n_routed_experts: 8,
        top_k: 3,
        moe_inter: 6,
        n_shared_experts: 1,
        dense_inter: 10,
        first_k_dense_replace: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
    }
}

fn dsa_cfg() -> DsaConfig {
    DsaConfig {
        index_n_heads: 2,
        index_head_dim: 4,
        index_topk: 3,
    }
}

/// All three layer kinds, so the mixer, attention and MLP planners each see a graph.
fn nemotron_h_cfg() -> NemotronHConfig {
    NemotronHConfig {
        vocab_size: 12,
        hidden: 6,
        mlp_inter: 10,
        eps: 1e-6,
        pattern: vec![
            NemotronHLayerKind::Mamba,
            NemotronHLayerKind::Attention,
            NemotronHLayerKind::Mlp,
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
    }
}

/// A mixed GDN then QSA schedule; `chunk` is a multiple of both `index_compress_ratio` and `conv_k`.
fn qwen38_cfg() -> Qwen4ExpModelConfig {
    Qwen4ExpModelConfig {
        cfg: Qwen4ExpConfig {
            hidden: 8,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            rotary_dim: 2,
            eps: 1e-5,
        },
        qcfg: QsaConfig {
            index_n_heads: 2,
            index_kv_heads: 1,
            index_head_dim: 3,
            index_budget: 4,
            index_compress_ratio: 2,
        },
        gdn: Qwen4ExpGdnConfig {
            num_k_heads: 1,
            num_v_heads: 2,
            head_dim: 2,
            conv_k: 4,
        },
        layer_is_full: vec![false, true],
        vocab: 6,
        max_pos: 16,
        ffn_inter: 8,
        moe_n_experts: 3,
        moe_top_k: 2,
        moe_inter: 4,
        eps: 1e-5,
        chunk: 4,
        hc_count: 2,
        hc_lowrank: 3,
        ple: None,
    }
}

// ---- the corpus ------------------------------------------------------------------------------

/// The named corpus. A new row is a new graph in the summary; renaming a row renames its lines.
///
/// - Every `DecodeArch` family of poot-llm: fixed-capacity decode, and the stateless prefill the CPU
///   oracle traces. Qwen2 appears as qwen2 (bias), qwen3 (QK-norm) and qwen2 with bf16 projections.
/// - The contiguous fixed-KV prefill of the five architectures that have one, and a batched decode.
/// - The MoE families are the routed rows above: qwen3-moe, mixtral, olmoe, gpt-oss, granitemoe and the
///   DeepSeek family.
///
/// Deliberately not in the corpus: the exact families (DeepSeek-V4, MiniMax-M2, Qwen3.5 exact, Qwen4Exp,
/// GLM-5.3). Their tracers served the exact route that Card 573 deleted, and the test-support features
/// their fixtures needed are gone. Each returns as an ordinary model in Cards 568-572, and each of those
/// cards adds its family's rows here when it lands.
fn corpus() -> Vec<Row> {
    vec![
        Row {
            name: "qwen2.decode",
            build: || traced(dense(Family::Qwen2), Phase::Decode, 1, 1, CAP),
        },
        Row {
            name: "qwen2.prefill",
            build: || traced(dense(Family::Qwen2), Phase::Prefill, 1, PROMPT, PROMPT),
        },
        Row {
            name: "qwen2.prefill_kv",
            build: || traced(dense(Family::Qwen2), Phase::Prefill, 1, PROMPT, CAP),
        },
        Row {
            name: "qwen2.decode_batched",
            build: || traced(dense(Family::Qwen2), Phase::Decode, BATCH, 1, CAP),
        },
        Row {
            name: "qwen3.decode",
            build: || traced(dense(Family::Qwen3), Phase::Decode, 1, 1, CAP),
        },
        Row {
            name: "qwen3.prefill",
            build: || traced(dense(Family::Qwen3), Phase::Prefill, 1, PROMPT, PROMPT),
        },
        Row {
            name: "qwen2_bf16.decode",
            build: || traced_bf16(dense(Family::Qwen2), Phase::Decode, 1, CAP),
        },
        Row {
            name: "qwen2_bf16.prefill",
            build: || traced_bf16(dense(Family::Qwen2), Phase::Prefill, PROMPT, PROMPT),
        },
        Row {
            name: "gemma.decode",
            build: || traced(gemma(), Phase::Decode, 1, 1, CAP),
        },
        Row {
            name: "gemma.prefill",
            build: || traced(gemma(), Phase::Prefill, 1, PROMPT, PROMPT),
        },
        Row {
            name: "gemma.prefill_kv",
            build: || traced(gemma(), Phase::Prefill, 1, PROMPT, CAP),
        },
        Row {
            name: "granite.decode",
            build: || trace_granite_decode_kv_masked(base_cfg(), granite_params(None), CAP),
        },
        Row {
            name: "granite.prefill",
            build: || trace_granite_prefill(base_cfg(), granite_params(None), PROMPT),
        },
        Row {
            name: "granite.prefill_kv",
            build: || trace_granite_prefill_kv(base_cfg(), granite_params(None), PROMPT, CAP),
        },
        Row {
            name: "granitemoe.decode",
            build: || {
                trace_granite_decode_kv_masked(base_cfg(), granite_params(Some(granite_moe())), CAP)
            },
        },
        Row {
            name: "granitemoe.prefill",
            build: || {
                trace_granite_prefill(base_cfg(), granite_params(Some(granite_moe())), PROMPT)
            },
        },
        Row {
            name: "qwen3_moe.decode",
            build: || trace_qwen3_moe_decode_kv_masked(base_cfg(), qwen3_moe_params(), CAP),
        },
        Row {
            name: "qwen3_moe.prefill",
            build: || trace_qwen3_moe_prefill(base_cfg(), qwen3_moe_params(), PROMPT),
        },
        Row {
            name: "qwen3_moe.prefill_kv",
            build: || trace_qwen3_moe_prefill_kv(base_cfg(), qwen3_moe_params(), PROMPT, CAP),
        },
        Row {
            name: "olmo2.decode",
            build: || traced(dense(Family::Olmo2), Phase::Decode, 1, 1, CAP),
        },
        Row {
            name: "olmo2.prefill",
            build: || traced(dense(Family::Olmo2), Phase::Prefill, 1, PROMPT, PROMPT),
        },
        Row {
            name: "olmo2.prefill_kv",
            build: || traced(dense(Family::Olmo2), Phase::Prefill, 1, PROMPT, CAP),
        },
        Row {
            name: "bloom.decode",
            build: || trace_bloom_decode_kv_masked(&bloom_cfg(), CAP),
        },
        Row {
            name: "bloom.prefill",
            build: || trace_bloom_prefill(&bloom_cfg(), PROMPT),
        },
        Row {
            name: "mpt.decode",
            build: || trace_mpt_decode_kv_masked(&mpt_cfg(), CAP),
        },
        Row {
            name: "mpt.prefill",
            build: || trace_mpt_prefill(&mpt_cfg(), PROMPT),
        },
        Row {
            name: "smollm3.decode",
            build: || trace_smollm3_decode_kv_masked(&smollm3_cfg(), CAP),
        },
        Row {
            name: "smollm3.prefill",
            build: || trace_smollm3_prefill(&smollm3_cfg(), PROMPT),
        },
        Row {
            name: "mixtral.decode",
            build: || trace_mixtral_decode_kv_masked(base_cfg(), mixtral_params(), CAP),
        },
        Row {
            name: "mixtral.prefill",
            build: || trace_mixtral_prefill(base_cfg(), mixtral_params(), PROMPT),
        },
        Row {
            name: "olmoe.decode",
            build: || trace_olmoe_decode_kv_masked(base_cfg(), olmoe_params(), CAP),
        },
        Row {
            name: "olmoe.prefill",
            build: || trace_olmoe_prefill(base_cfg(), olmoe_params(), PROMPT),
        },
        Row {
            name: "gpt_oss.decode",
            build: || trace_gptoss_decode_kv_masked(base_cfg(), gpt_oss_params(), CAP),
        },
        Row {
            name: "gpt_oss.prefill",
            build: || trace_gptoss_prefill(base_cfg(), gpt_oss_params(), PROMPT),
        },
        Row {
            name: "deepseek2.decode",
            build: || trace_deepseek2_decode_kv_masked(deepseek_cfg(Some(3)), deepseek2_moe(), CAP),
        },
        Row {
            name: "deepseek2.prefill",
            build: || trace_deepseek2_prefill(deepseek_cfg(Some(3)), deepseek2_moe(), PROMPT),
        },
        Row {
            name: "deepseek3.decode",
            build: || trace_deepseek3_decode_kv_masked(deepseek_cfg(Some(3)), deepseek3_moe(), CAP),
        },
        Row {
            name: "deepseek3.prefill",
            build: || trace_deepseek3_prefill(deepseek_cfg(Some(3)), deepseek3_moe(), PROMPT),
        },
        Row {
            name: "deepseek32.decode",
            build: || {
                trace_deepseek32_dsa_decode(deepseek_cfg(Some(5)), dsa_cfg(), deepseek3_moe(), CAP)
            },
        },
        Row {
            name: "deepseek32.prefill",
            build: || {
                trace_deepseek32_dsa_prefill(
                    deepseek_cfg(Some(5)),
                    dsa_cfg(),
                    deepseek3_moe(),
                    PROMPT,
                )
            },
        },
        Row {
            name: "qwen38.decode",
            build: || trace_qwen38_decode(&qwen38_cfg(), CAP),
        },
        Row {
            name: "qwen38.prefill",
            build: || trace_qwen38_prefill(&qwen38_cfg(), PROMPT),
        },
        Row {
            name: "nemotron_h.decode",
            build: || trace_nemotron_h_decode(&nemotron_h_cfg(), CAP),
        },
        Row {
            name: "nemotron_h.prefill",
            build: || trace_nemotron_h_prefill(&nemotron_h_cfg(), PROMPT),
        },
    ]
}

// ---- the summary -----------------------------------------------------------------------------

const HEADER: &str = "graph\teqn\top\tplan\tchoice\trequest\tlaunches\tgrid\tdetail\n";

/// Tabs and newlines in a free-text field would break the line format.
fn cell(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// A 64-bit FNV-1a digest of a text, so a short cell still changes whenever the text does.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The `request` cell of a kernel choice: a digest of its whole `Debug` encoding (every parameter that
/// selects a different kernel, shapes and layouts included), so any change to a request changes the
/// line even though the `choice` column only names its kind. The plan's own `key` is not written: it
/// folds in the codegen epoch, so it changes with the toolchain and every `poot-codegen` edit, which
/// would make this committed file churn for reasons that have nothing to do with planning.
fn request_cell(choice: &KernelChoice) -> String {
    format!("{:016x}", fnv1a(&format!("{choice:?}")))
}

/// The variant name of a debug-printed value: `Unsupported("x")` is `Unsupported`.
fn variant_name(debug: &str) -> &str {
    debug
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .next()
        .unwrap_or(debug)
}

/// One summary line body (everything after `graph`, `eqn` and `op`) for a plan result.
struct PlanCells {
    plan: &'static str,
    /// [`KernelChoice::label`], or `-` for a refused graph.
    choice: String,
    request: String,
    launches: usize,
    grid: String,
    detail: String,
}

impl PlanCells {
    fn no_launch(plan: &'static str, choice: &KernelChoice, detail: String) -> Self {
        Self {
            plan,
            choice: choice.label(),
            request: request_cell(choice),
            launches: 0,
            grid: "-".to_string(),
            detail,
        }
    }
}

/// `plan`'s summary cells, for one equation of a successfully compiled [`poot_graph_plan::Program`],
/// with the kernel choice the program recorded for it.
fn describe_plan(plan: &Plan, choice: &KernelChoice) -> PlanCells {
    let grid_str = |[x, y, z]: [u32; 3]| format!("{x},{y},{z}");
    let cells = |plan, grid, detail| PlanCells {
        plan,
        choice: choice.label(),
        request: request_cell(choice),
        launches: 1,
        grid,
        detail,
    };
    match plan {
        Plan::Compute { grid, .. } => cells("compute", grid_str(*grid), "-".to_string()),
        Plan::ComputeMeta { meta, grid, .. } => {
            cells("compute_meta", grid_str(*grid), format!("meta={meta:?}"))
        }
        Plan::ComputeChunks(chunks) => PlanCells {
            plan: "chunks",
            choice: choice.label(),
            request: request_cell(choice),
            launches: chunks.len(),
            grid: format!(
                "groups={:?}",
                chunks.iter().map(|c| c.groups).collect::<Vec<_>>()
            ),
            detail: format!(
                "meta={:?}",
                chunks.iter().map(|c| c.meta.as_slice()).collect::<Vec<_>>()
            ),
        },
        Plan::Alias(src) => PlanCells::no_launch("alias", choice, format!("src=v{src}")),
        Plan::View {
            src,
            strides,
            offset,
        } => PlanCells::no_launch(
            "view",
            choice,
            format!("src=v{src} strides={strides:?} offset={offset}"),
        ),
        Plan::Collective { kind, op, axis } => {
            PlanCells::no_launch("collective", choice, format!("{kind:?} {op:?} axis={axis}"))
        }
    }
}

/// The `eqn`, `op` and cells of the one summary line a whole-graph [`CompileError`] produces: `compile`
/// refuses (or otherwise fails) the graph as a unit, not one equation at a time, so a refused corpus
/// graph gets exactly one line, naming the refused equation and op when the error is a typed
/// [`PlanError::Refused`] (the only refusal kind the corpus is expected to hit) and `-` for any other
/// [`CompileError`] variant (numerics, resident-E4M3, legalize, or a non-refusal [`PlanError`]).
fn describe_compile_error(error: &CompileError) -> (String, String, PlanCells) {
    let (eqn, op, key) = match error {
        CompileError::Plan(plan_error) => match plan_error.as_ref() {
            PlanError::Refused(refusal) => (
                refusal.eqn.to_string(),
                refusal.op.name(),
                variant_name(&format!("{:?}", refusal.missing)).to_string(),
            ),
            other => (
                "-".to_string(),
                "-".to_string(),
                variant_name(&format!("{other:?}")).to_string(),
            ),
        },
        other => (
            "-".to_string(),
            "-".to_string(),
            variant_name(&format!("{other:?}")).to_string(),
        ),
    };
    (
        eqn,
        op,
        PlanCells {
            plan: "refused",
            choice: "-".to_string(),
            request: key,
            launches: 0,
            grid: "-".to_string(),
            detail: error.to_string(),
        },
    )
}

/// One summary line.
fn write_line(out: &mut String, name: &str, eqn: &str, op: &str, c: &PlanCells) {
    writeln!(
        out,
        "{name}\t{eqn}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        cell(op),
        c.plan,
        cell(&c.choice),
        cell(&c.request),
        c.launches,
        c.grid,
        cell(&c.detail),
    )
    .expect("write to a String");
}

/// The whole summary for one target: every corpus graph, in corpus order, compiled through the real
/// [`compile`] entry point (card 523b: a hand-rolled `optimize`/`widen_mismatched_matmul_dtypes`/
/// `plan_eqn_views_analyzed` sub-pipeline here was the same "second entry point" anti-pattern R469-021
/// found in the wgpu/ROCm/PTX executors before card 534a unified them onto `compile`).
fn render(backend: Backend) -> String {
    let mut out = String::from(HEADER);
    let target = Target {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    for row in corpus() {
        let g: Graph = (row.build)();
        match compile(&g, &target, &options) {
            Ok(program) => {
                for (i, (eqn, plan)) in program.planned().enumerate() {
                    let c = describe_plan(plan, program.kernel_choice(eqn));
                    write_line(&mut out, row.name, &i.to_string(), &eqn.op.name(), &c);
                }
            }
            Err(error) => {
                let (eqn, op, c) = describe_compile_error(&error);
                write_line(&mut out, row.name, &eqn, &op, &c);
            }
        }
    }
    out
}

fn expected_path(target: &str) -> PathBuf {
    // `std::env::var`, not `env!`: the latter bakes the path into this test binary's compiled object code
    // at build time, and a shared compile cache (kache) that reuses that object across worktrees by
    // source-content hash would then serve whichever worktree's path happened to compile it first
    // (card 530's build.rs fix; card 543 review). `std::env::var` reads the environment cargo sets fresh
    // for every test-binary invocation, so it is correct regardless of which worktree compiled the binary.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries");
    PathBuf::from(manifest_dir)
        .join("tests")
        .join("plan_summary")
        .join(format!("{target}.tsv"))
}

// ---- the comparison --------------------------------------------------------------------------

/// The `graph` and `eqn` columns of a summary line: the identity a changed line is paired by.
fn line_key(line: &str) -> (&str, &str) {
    let mut cols = line.splitn(3, '\t');
    (cols.next().unwrap_or(""), cols.next().unwrap_or(""))
}

/// Lines shown in a failure before the rest are counted.
const DIFF_LIMIT: usize = 40;

/// The columns of two lines for the same equation that differ, one `column: old -> new` line each.
fn changed_columns(old: &str, new: &str) -> String {
    let names = HEADER.trim_end().split('\t');
    let mut old_cols = old.split('\t');
    let mut new_cols = new.split('\t');
    let mut lines = Vec::new();
    for name in names {
        let (o, n) = (old_cols.next().unwrap_or(""), new_cols.next().unwrap_or(""));
        if o != n {
            lines.push(format!("    {name}: {o} -> {n}"));
        }
    }
    lines.join("\n")
}

/// A readable diff of two summaries: each changed equation once, with the columns that moved (old to
/// new), paired by (graph, equation). Empty when they are equal.
fn diff(expected: &str, actual: &str) -> String {
    fn by_key(text: &str) -> BTreeMap<(&str, usize), &str> {
        text.lines()
            .skip(1)
            .map(|l| {
                let (g, e) = line_key(l);
                ((g, e.parse::<usize>().unwrap_or(usize::MAX)), l)
            })
            .collect()
    }
    let (old, new) = (by_key(expected), by_key(actual));
    let mut changes: Vec<(&str, String)> = Vec::new();
    for (key, old_line) in &old {
        match new.get(key) {
            Some(new_line) if new_line == old_line => {}
            Some(new_line) => changes.push((
                key.0,
                format!(
                    "changed {}#{} ({})\n{}",
                    key.0,
                    key.1,
                    old_line.split('\t').nth(2).unwrap_or("?"),
                    changed_columns(old_line, new_line)
                ),
            )),
            None => changes.push((
                key.0,
                format!("removed {}#{}\n  - {old_line}", key.0, key.1),
            )),
        }
    }
    for (key, new_line) in &new {
        if !old.contains_key(key) {
            changes.push((key.0, format!("added {}#{}\n  + {new_line}", key.0, key.1)));
        }
    }
    if changes.is_empty() {
        // Pairing by (graph, equation) cannot see order: the same lines in another order are a difference.
        return if expected == actual {
            String::new()
        } else {
            "every equation line is unchanged but the lines are in a different order".to_string()
        };
    }
    let mut per_graph: BTreeMap<&str, usize> = BTreeMap::new();
    for (graph, _) in &changes {
        *per_graph.entry(graph).or_default() += 1;
    }
    let total = changes.len();
    let mut out = format!(
        "{total} changed equations: {}\n",
        per_graph
            .iter()
            .map(|(graph, n)| format!("{graph} ({n})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let shown: Vec<String> = changes
        .into_iter()
        .take(DIFF_LIMIT)
        .map(|(_, text)| text)
        .collect();
    out.push_str(&shown.join("\n"));
    if total > DIFF_LIMIT {
        write!(out, "\n... and {} more", total - DIFF_LIMIT).expect("write to a String");
    }
    out
}

/// Compare (or, under [`REGEN_ENV`], rewrite) one target's checked-in summary.
fn check_target(target: &str, backend: Backend) {
    let actual = render(backend);
    let path = expected_path(target);
    if std::env::var_os(REGEN_ENV).is_some() {
        std::fs::create_dir_all(path.parent().expect("expected file has a parent"))
            .expect("create tests/plan_summary");
        std::fs::write(&path, &actual).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\nregenerate with `just test-plan-summary-regen`",
            path.display()
        )
    });
    let changes = diff(&expected, &actual);
    assert!(
        changes.is_empty(),
        "{target}: the plan summary differs from {}\n{changes}\n\
         If the plan change is intended, run `just test-plan-summary-regen` and list every changed \
         line in the card's landing note.",
        path.display()
    );
}

#[test]
fn spirv_vulkan_plan_summary_matches_the_checked_in_file() {
    let (target, backend) = targets()[0];
    check_target(target, backend);
}

#[test]
fn amd_gcn_plan_summary_matches_the_checked_in_file() {
    let (target, backend) = targets()[1];
    check_target(target, backend);
}

#[test]
fn nvptx_plan_summary_matches_the_checked_in_file() {
    let (target, backend) = targets()[2];
    check_target(target, backend);
}

/// The plan kinds `poot_executor::Engine::load` accepts: every kind a single-device `compile` may emit.
/// An exhaustive match (no wildcard), so a new `Plan` variant must be classified here before this
/// file compiles; `Collective` is the multi-rank kind a single-device executor refuses with
/// `LoadError::UnloadablePlan`.
fn is_loadable(plan: &Plan) -> bool {
    match plan {
        Plan::Compute { .. }
        | Plan::ComputeMeta { .. }
        | Plan::ComputeChunks(_)
        | Plan::Alias(_)
        | Plan::View { .. } => true,
        Plan::Collective { .. } => false,
    }
}

/// Card 626 SC-003: `compile` of every corpus graph, on every target, never yields a plan kind an
/// executor cannot load (the deleted `Plan::Host` was one). Mutation: re-add a `Plan::Host` variant,
/// classify it `false` above, and make the planner emit it for a corpus op; this row goes red naming
/// the graph, target and equation.
#[test]
fn compile_emits_only_loadable_plan_kinds_over_the_whole_corpus() {
    let mut planned = 0usize;
    for (target_name, backend) in targets() {
        let target = Target {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        let options = CompileOptions {
            execution: Submission::Replay,
            fusion: FusionPolicy::Full,
            limits: poot_graph_plan::CompileLimits::STANDARD,
        };
        for row in corpus() {
            let g: Graph = (row.build)();
            let Ok(program) = compile(&g, &target, &options) else {
                continue;
            };
            for (i, (eqn, plan)) in program.planned().enumerate() {
                assert!(
                    is_loadable(plan),
                    "{} on {target_name}: eqn {i} ({}) compiled to the unloadable kind {:?}",
                    row.name,
                    eqn.op.name(),
                    plan.kind()
                );
                planned += 1;
            }
        }
    }
    assert!(
        planned > 1000,
        "the corpus planned only {planned} equations"
    );
}

/// Rendering the same corpus twice, from freshly traced graphs, gives the same text: nothing in the
/// summary depends on hash-map order, an address or a timestamp. Without this a checked-in file could
/// pass once and fail on the next run.
#[test]
fn plan_summary_rendering_is_deterministic() {
    for (target, backend) in targets() {
        let first = render(backend);
        let second = render(backend);
        assert!(
            first == second,
            "{target}: two renders of the same corpus differ\n{}",
            diff(&first, &second)
        );
    }
}

/// Card 557 SC-003: a tracer states primitives and `ops::` compositions only. Every corpus graph, as
/// traced and before `compile`, holds no compiler-fusion or schedule-choice composite (`OpClass::
/// Composite`: the flash ops, `Rope`, `MatMulBias` and the fused regions; the packed and dense claims
/// are `Primitive` until Card 629 gives them a decomposition); no
/// bounded semantic source carrier is authorized yet. Then `compile`'s passes do form composites, so
/// the check is not vacuous. Mutation: emit `MatMulBias` from `ops::linear`, or mark one of those ops
/// `OpClass::Primitive` and trace it; this row names the graph and equation.
#[test]
fn every_corpus_trace_is_primitive_before_compile() {
    use poot_graph_ir::OpClass;

    let mut composites_after_compile = 0usize;
    let target = Target {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    for row in corpus() {
        let g: Graph = (row.build)();
        for (i, eqn) in g.eqns.iter().enumerate() {
            assert_eq!(
                eqn.op.class(),
                OpClass::Primitive,
                "{}: traced eqn {i} is the composite {}",
                row.name,
                eqn.op.name()
            );
        }
        if let Ok(program) = compile(&g, &target, &options) {
            composites_after_compile += program
                .graph()
                .eqns
                .iter()
                .filter(|eqn| eqn.op.class() == OpClass::Composite)
                .count();
        }
    }
    assert!(
        composites_after_compile > 0,
        "compile formed no composite over the corpus, so the traced check proves nothing"
    );
}

use super::*;
use crate::deepseek4::parse_compress_ratios;

/// The real `deepseek-ai/DeepSeek-V4-Flash-0731` config at the Card 348 revision. Every value is the one
/// `crate::deepseek4`'s module doc records.
/// `intermediate` is the only field with no real counterpart: no V4 layer has a dense FFN, so it is set to
/// the MoE width and never read.
fn exact_config() -> DeepseekV4Config {
    DeepseekV4Config {
        vocab: 129_280,
        hidden: 4096,
        layers: 43,
        num_heads: 64,
        head_dim: 512,
        qk_rope_head_dim: 64,
        q_lora_rank: 1024,
        o_groups: 8,
        o_lora_rank: 1024,
        intermediate: 2048,
        swiglu_limit: 10.0,
        sliding_window: 128,
        eps: 1e-6,
        max_pos: 1_048_576,
        rope_theta: 10_000.0,
        hc_mult: 4,
        hc_sinkhorn_iters: 20,
        hc_eps: 1e-6,
        hca_compress_rate: 128,
        compress_rope_theta: 160_000.0,
        csa_compress_rate: 4,
        index_n_heads: 64,
        index_head_dim: 128,
        index_topk: 512,
        routed_experts: 256,
        experts_per_tok: 6,
        moe_intermediate: 2048,
        hash_router_layers: 3,
        route_scale: 1.5,
    }
}

/// The real 46-entry `compress_ratios`, truncated to `num_hidden_layers`. The trailing three zeros belong to
/// the `mtp.*` modules, which this plan does not cover.
fn exact_schedule(cfg: &DeepseekV4Config) -> Vec<V4LayerKind> {
    #[rustfmt::skip]
    let ratios: [usize; 46] = [
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
        128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
    ];
    parse_compress_ratios(cfg, &ratios[..cfg.layers])
}

fn exact_plan() -> DeepseekV4SourcePlan {
    let cfg = exact_config();
    let schedule = exact_schedule(&cfg);
    DeepseekV4SourcePlan::new(&cfg, &schedule).expect("exact V4 source plan")
}

/// Small but non-degenerate: one layer of each kind, two experts, and every width distinct so a swapped
/// shape cannot pass by coincidence.
fn tiny_config() -> DeepseekV4Config {
    DeepseekV4Config {
        vocab: 10,
        hidden: 8,
        layers: 3,
        num_heads: 2,
        head_dim: 6,
        qk_rope_head_dim: 2,
        q_lora_rank: 5,
        o_groups: 2,
        o_lora_rank: 3,
        intermediate: 7,
        swiglu_limit: 0.5,
        sliding_window: 3,
        eps: 1e-5,
        max_pos: 16,
        rope_theta: 10_000.0,
        hc_mult: 2,
        hc_sinkhorn_iters: 4,
        hc_eps: 1e-6,
        hca_compress_rate: 2,
        compress_rope_theta: 20_000.0,
        csa_compress_rate: 4,
        index_n_heads: 2,
        index_head_dim: 4,
        index_topk: 2,
        routed_experts: 2,
        experts_per_tok: 2,
        moe_intermediate: 4,
        hash_router_layers: 1,
        route_scale: 1.5,
    }
}

const TINY_SCHEDULE: [V4LayerKind; 3] = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];

fn tiny_plan() -> DeepseekV4SourcePlan {
    DeepseekV4SourcePlan::new(&tiny_config(), &TINY_SCHEDULE).expect("tiny V4 source plan")
}

/// The pinned main-decoder inventory. The E4M3 count is `43 * (5 + 3) + 21`: it is reached only when the
/// five attention linears, the three shared-expert linears, and the indexer query projection on exactly
/// the 21 CSA layers all carry the packed role.
#[test]
fn deepseek4_source_plan_row_counts_are_the_audited_inventory() {
    let plan = exact_plan();
    plan.check_exact_counts().expect("audited row counts");

    let e4m3 = plan
        .packed_sources()
        .filter(|row| {
            row.descriptor.format()
                == WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::E8m0,
                }
        })
        .count();
    let fp4 = plan
        .packed_sources()
        .filter(|row| row.descriptor.format() == WeightFormat::E2m1Row32)
        .count();
    let exact_i32 = plan
        .dense_sources()
        .filter(|row| row.is_exact_i32())
        .count();
    assert_eq!(e4m3, 365, "Card 348's audited main E4M3 pair count");
    assert_eq!(fp4, 33_024, "Card 348's audited main FP4 pair count");
    assert_eq!(plan.dense_sources().len() - exact_i32, 831);
    assert_eq!(exact_i32, 3, "one tid2eid per hash-routed layer");

    let csa = plan
        .schedule()
        .iter()
        .filter(|kind| **kind == V4LayerKind::Csa)
        .count();
    let indexer = plan
        .packed_sources()
        .filter(|row| row.role == V4PackedRole::IndexerWqB)
        .count();
    assert_eq!((csa, indexer), (21, 21), "the indexer weight is CSA-only");

    // Card 348's audited I64 payload total, reached by the three hash tables and nothing else.
    let i64_bytes: usize = plan
        .dense_sources()
        .filter(|row| row.is_exact_i32())
        .map(|row| row.shape.iter().product::<usize>() * 8)
        .sum();
    assert_eq!(i64_bytes, 18_616_320);
}

/// The one place a V4 source name is written as a literal. Every other consumer reads the plan, so a
/// rename shows up here and nowhere else.
#[test]
fn deepseek4_source_plan_names_match_the_checkpoint() {
    let plan = exact_plan();

    for (row, name, shape) in [
        (V4TopDense::Embedding, "embed.weight", vec![129_280, 4096]),
        (V4TopDense::FinalNorm, "norm.weight", vec![4096]),
        (V4TopDense::Head, "head.weight", vec![129_280, 4096]),
        (
            V4TopDense::HcHeadProjection,
            "hc_head_fn",
            vec![4, 4 * 4096],
        ),
        (V4TopDense::HcHeadBase, "hc_head_base", vec![4]),
        (V4TopDense::HcHeadScale, "hc_head_scale", vec![1]),
    ] {
        assert_eq!(row.name(), name);
        let spec = plan.top_dense(row).expect(name);
        assert_eq!(spec.name, name);
        assert_eq!(spec.shape, shape, "{name}");
    }

    // Layer 0 is sliding: the five attention linears, the three shared experts, no indexer.
    for (role, linear_id, logical) in [
        (V4PackedRole::WqA, "layers.0.attn.wq_a", [1024, 4096]),
        (V4PackedRole::WqB, "layers.0.attn.wq_b", [64 * 512, 1024]),
        (V4PackedRole::Wkv, "layers.0.attn.wkv", [512, 4096]),
        (V4PackedRole::WoA, "layers.0.attn.wo_a", [8 * 1024, 4096]),
        (V4PackedRole::WoB, "layers.0.attn.wo_b", [4096, 8 * 1024]),
        (
            V4PackedRole::SharedExpert(V4ExpertProjection::W1),
            "layers.0.ffn.shared_experts.w1",
            [2048, 4096],
        ),
        (
            V4PackedRole::SharedExpert(V4ExpertProjection::W2),
            "layers.0.ffn.shared_experts.w2",
            [4096, 2048],
        ),
        (
            V4PackedRole::SharedExpert(V4ExpertProjection::W3),
            "layers.0.ffn.shared_experts.w3",
            [2048, 4096],
        ),
    ] {
        let spec = plan.packed(0, role).expect(linear_id);
        assert_eq!(spec.linear_id, linear_id);
        assert_eq!(spec.descriptor.shape(), logical, "{linear_id}");
        assert_eq!(
            spec.descriptor.format(),
            WeightFormat::E4m3Block128 {
                scale: ScaleEncoding::E8m0
            },
            "{linear_id}"
        );
    }

    for (name, kind, shape) in [
        (
            "layers.0.attn_norm.weight",
            ExactSourceKind::Bf16,
            vec![4096],
        ),
        (
            "layers.0.ffn_norm.weight",
            ExactSourceKind::Bf16,
            vec![4096],
        ),
        (
            "layers.0.attn.q_norm.weight",
            ExactSourceKind::Bf16,
            vec![1024],
        ),
        (
            "layers.0.attn.kv_norm.weight",
            ExactSourceKind::Bf16,
            vec![512],
        ),
        ("layers.0.attn.attn_sink", ExactSourceKind::Bf16, vec![64]),
        (
            "layers.0.hc_attn_fn",
            ExactSourceKind::Bf16,
            vec![24, 4 * 4096],
        ),
        ("layers.0.hc_attn_base", ExactSourceKind::F32, vec![24]),
        ("layers.0.hc_attn_scale", ExactSourceKind::F32, vec![3]),
        (
            "layers.0.hc_ffn_fn",
            ExactSourceKind::Bf16,
            vec![24, 4 * 4096],
        ),
        ("layers.0.hc_ffn_base", ExactSourceKind::F32, vec![24]),
        ("layers.0.hc_ffn_scale", ExactSourceKind::F32, vec![3]),
        (
            "layers.0.ffn.gate.weight",
            ExactSourceKind::Bf16,
            vec![256, 4096],
        ),
        (
            "layers.0.ffn.gate.tid2eid",
            ExactSourceKind::I64,
            vec![129_280, 6],
        ),
        ("layers.3.ffn.gate.bias", ExactSourceKind::F32, vec![256]),
        // Layer 2 is CSA (pooled width 2 * head_dim), layer 3 is HCA (pooled width head_dim); both read
        // the same real prefix.
        (
            "layers.2.attn.compressor.wkv.weight",
            ExactSourceKind::Bf16,
            vec![1024, 4096],
        ),
        (
            "layers.3.attn.compressor.wkv.weight",
            ExactSourceKind::Bf16,
            vec![512, 4096],
        ),
        (
            "layers.2.attn.compressor.ape",
            ExactSourceKind::Bf16,
            vec![4, 1024],
        ),
        (
            "layers.3.attn.compressor.ape",
            ExactSourceKind::Bf16,
            vec![128, 512],
        ),
        (
            "layers.2.attn.compressor.norm.weight",
            ExactSourceKind::Bf16,
            vec![512],
        ),
        (
            "layers.2.attn.indexer.compressor.wkv.weight",
            ExactSourceKind::Bf16,
            vec![256, 4096],
        ),
        (
            "layers.2.attn.indexer.compressor.ape",
            ExactSourceKind::Bf16,
            vec![4, 256],
        ),
        (
            "layers.2.attn.indexer.compressor.norm.weight",
            ExactSourceKind::Bf16,
            vec![128],
        ),
        (
            "layers.2.attn.indexer.weights_proj.weight",
            ExactSourceKind::Bf16,
            vec![64, 4096],
        ),
    ] {
        let spec = plan.dense(name).expect(name);
        assert_eq!(spec.name, name);
        assert_eq!(spec.kind, kind, "{name}");
        assert_eq!(spec.shape, shape, "{name}");
    }

    let indexer = plan
        .packed(2, V4PackedRole::IndexerWqB)
        .expect("CSA indexer query projection");
    assert_eq!(indexer.linear_id, "layers.2.attn.indexer.wq_b");
    assert_eq!(indexer.descriptor.shape(), [64 * 128, 1024]);

    let routed = plan
        .packed(
            7,
            V4PackedRole::RoutedExpert {
                expert: 255,
                projection: V4ExpertProjection::W2,
            },
        )
        .expect("last routed expert down projection");
    assert_eq!(routed.linear_id, "layers.7.ffn.experts.255.w2");
    assert_eq!(routed.descriptor.format(), WeightFormat::E2m1Row32);
    assert_eq!(routed.descriptor.shape(), [4096, 2048]);

    // No placeholder name survives anywhere in the plan.
    for placeholder in [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.hc_head.fn",
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_a_proj.weight",
        "layers.2.self_attn.csa.indexer.q_b_proj.weight",
    ] {
        assert!(
            plan.classify(placeholder).is_none(),
            "{placeholder} is the tracer's old placeholder and must not be a plan row"
        );
    }
}

/// Every packed row stages exactly the authoritative source bytes: one `I8` constant per descriptor
/// source (weight and scale for these E4M3/E2M1 cells), each spanning that source's own buffer.
/// Nothing in the plan is a decoded logical weight.
#[test]
fn deepseek4_packed_plan_has_no_f32_mirror() {
    let plan = tiny_plan();
    for row in plan.packed_sources() {
        let constants = row.source_constants();
        assert_eq!(
            constants.len(),
            row.descriptor.sources().len(),
            "{}",
            row.linear_id
        );
        for ((name, tensor_type), role) in constants.iter().zip(row.descriptor.sources()) {
            assert_eq!(name.linear_id(), row.linear_id);
            assert_eq!(name.role(), role, "{}", row.linear_id);
            assert_eq!(tensor_type.dtype, DType::I8, "{}", row.linear_id);
            assert_eq!(
                tensor_type.shape.iter().product::<usize>(),
                row.descriptor.source_bytes(role),
                "{} {role:?}",
                row.linear_id
            );
        }
    }

    // The mirror guard itself: a differently named F32 matrix with an admitted logical weight's shape, or
    // its transpose, is rejected rather than accepted as an ordinary dense row.
    let cfg = tiny_config();
    let logical = V4ExpertProjection::W2.logical_shape(cfg.hidden, cfg.moe_intermediate);
    for shape in [logical.to_vec(), vec![logical[1], logical[0]]] {
        let mut rows = V4SourceRows::default();
        rows.packed(&cfg, 0, V4PackedRole::WqA, "layers.0")
            .expect("packed row");
        rows.packed(
            &cfg,
            0,
            V4PackedRole::SharedExpert(V4ExpertProjection::W2),
            "layers.0",
        )
        .expect("packed row");
        rows.dense("layers.0.decoded_w2", ExactSourceKind::F32, shape.clone())
            .expect("the row is inserted; the mirror rule runs over the collected rows");
        let error = rows
            .reject_decoded_weight_mirrors()
            .expect_err("an F32 mirror of a packed logical weight must be rejected");
        assert_eq!(
            error,
            V4SourcePlanError::DecodedWeightMirror {
                name: "layers.0.decoded_w2".to_string(),
                shape,
            }
        );
    }

    // A BF16 row may legitimately share a packed logical shape: on an HCA layer
    // `attn.compressor.wkv.weight` is `[head_dim, hidden]`, which is exactly `attn.wkv`'s logical shape.
    // The real checkpoint has both, so the rule is about F32 rows only.
    let plan = exact_plan();
    assert_eq!(
        plan.dense("layers.3.attn.compressor.wkv.weight")
            .expect("HCA compressor")
            .shape,
        plan.packed(3, V4PackedRole::Wkv)
            .expect("kv projection")
            .descriptor
            .shape()
            .to_vec()
    );
}

/// The host tables are named, not reached by falling through the checkpoint roles, so a mistyped
/// checkpoint name cannot be silently accepted as a structural table.
#[test]
fn deepseek4_derived_constants_are_named_not_inferred() {
    let plan = tiny_plan();
    for name in V4_DERIVED_CONSTANTS {
        assert_eq!(plan.classify(name), Some(V4SourceClass::Derived), "{name}");
        assert!(deepseek4_is_derived_constant(name), "{name}");
    }
    for name in [
        "expert.iota_",
        "expert.iota_x",
        "expert.iota_2x",
        "rope.cosine",
        "causal",
        "layers.0.attn.wq_a.weight",
    ] {
        assert!(!deepseek4_is_derived_constant(name), "{name}");
    }
}

/// A layer's role set is a function of its kind, so a schedule change cannot leave a stale row behind.
#[test]
fn deepseek4_source_plan_roles_follow_the_layer_kind() {
    let plan = tiny_plan();
    let has = |name: &str| plan.classify(name).is_some();
    for (layer, kind) in TINY_SCHEDULE.iter().copied().enumerate() {
        let prefix = v4_layer_prefix(layer);
        for row in V4LayerDense::ALL {
            assert!(
                has(&format!("{prefix}.{}", row.suffix())),
                "{prefix}.{} is carried by every layer kind",
                row.suffix()
            );
        }
        for role in V4PackedRole::ATTENTION {
            plan.packed(layer, role).expect("every layer kind");
        }

        let compressor = has(&format!("{prefix}.attn.compressor.wkv.weight"));
        let indexer = has(&format!("{prefix}.attn.indexer.weights_proj.weight"));
        let indexer_packed = plan.packed(layer, V4PackedRole::IndexerWqB).is_ok();
        let expected = match kind {
            V4LayerKind::Sliding => (false, false, false),
            V4LayerKind::Hca => (true, false, false),
            V4LayerKind::Csa => (true, true, true),
        };
        assert_eq!(
            (compressor, indexer, indexer_packed),
            expected,
            "layer {layer} is {kind:?}"
        );
    }
}

/// Hash and score rows never share a layer: which one a layer carries is how Card 364c re-derives the
/// router partition from the manifest.
#[test]
fn deepseek4_router_selection_rows_partition_the_layers() {
    let plan = exact_plan();
    let cfg = *plan.config();
    for layer in 0..cfg.layers {
        let prefix = v4_layer_prefix(layer);
        let table = plan
            .dense(&format!("{prefix}.{V4_HASH_TABLE_SUFFIX}"))
            .is_ok();
        let bias = plan
            .dense(&format!("{prefix}.{V4_SCORE_BIAS_SUFFIX}"))
            .is_ok();
        assert_eq!(
            (table, bias),
            (
                layer < cfg.hash_router_layers,
                layer >= cfg.hash_router_layers
            ),
            "layer {layer} carries exactly one selection row"
        );
        let selection = plan.router_selection(layer).expect("one selection row");
        assert_eq!(selection.is_exact_i32(), table);
    }
}

/// The ordered expert table is numeric expert order, never map iteration order over names - where
/// `experts.10` sorts before `experts.2`.
#[test]
fn deepseek4_routed_rows_are_numeric_expert_order() {
    let cfg = DeepseekV4Config {
        routed_experts: 12,
        ..tiny_config()
    };
    let plan = DeepseekV4SourcePlan::new(&cfg, &TINY_SCHEDULE).expect("plan");
    let rows = plan
        .routed_rows(1, V4ExpertProjection::W3)
        .expect("routed table");
    assert_eq!(rows.len(), 12);
    for (expert, row) in rows.iter().enumerate() {
        assert_eq!(row.ordinal, expert);
        assert_eq!(row.linear_id, format!("layers.1.ffn.experts.{expert}.w3"));
    }
}

/// `attn.wo_a`'s grouped view is the split-then-transpose one Card 385's
/// `ops::packed_block_diagonal_linear` contracts against, not a direct reshape of the stored weight into
/// the same three extents.
#[test]
fn deepseek4_wo_a_keeps_the_grouped_weight_view() {
    let cfg = exact_config();
    let [out, in_per_group] = exact_plan()
        .packed(0, V4PackedRole::WoA)
        .expect("wo_a")
        .descriptor
        .shape();
    assert_eq!(
        [out, in_per_group],
        [8 * 1024, 4096],
        "the checkpoint's own [out, in]"
    );
    assert!(cfg.o_groups > 0 && out.is_multiple_of(cfg.o_groups));
    let o_lora_rank = out / cfg.o_groups;
    let grouped = [cfg.o_groups, in_per_group, o_lora_rank];
    assert_eq!(grouped, [8, 4096, 1024], "[o_groups, in_per_group, o_lora]");
    assert_ne!(
        grouped,
        [cfg.o_groups, o_lora_rank, in_per_group],
        "the direct reshape swaps the last two axes"
    );
}

/// A schedule that does not describe the config is rejected before any row is built.
#[test]
fn deepseek4_source_plan_rejects_a_mismatched_schedule() {
    let cfg = tiny_config();
    let error = DeepseekV4SourcePlan::new(&cfg, &TINY_SCHEDULE[..2])
        .expect_err("a two-entry schedule cannot describe three layers");
    assert_eq!(
        error,
        V4SourcePlanError::Source(V4SourceError::Config {
            field: "schedule",
            requirement: "must have one entry per layer",
        })
    );
}

/// A plan that is not the exact one fails the count check.
#[test]
fn deepseek4_exact_count_check_rejects_a_non_exact_plan() {
    let error = tiny_plan()
        .check_exact_counts()
        .expect_err("the tiny plan is not the exact inventory");
    assert!(matches!(error, V4SourceError::InventoryCount { .. }));
}

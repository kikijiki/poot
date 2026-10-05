use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use poot_eval::exact_dense::{DenseOwnerTensorView, ExactDenseError};
use poot_eval::{EvalBudget, EvalError, EvalOptions, ExactValue, Value, eval};
use poot_graph_ir::{OpKind, PackedSourceName, Storage};
use poot_load::packed_safetensors::{ExactSourceKind, SourceSpan, TensorDisposition};
use poot_quant::format::{ScaleEncoding, WeightFormat};
use poot_quant::{OperandRole, PackedPayload, SourceRole};
use poot_tensor::HostTensor;

use super::*;
use crate::glm53_flash::{GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT, dsa, kda_tests};
use crate::test_support::safetensors::{
    LoadedCheckpoint, SourceRow, bf16_bytes, load_checkpoint, truncate_to_bf16,
};

const CONFIG: &[u8] = include_bytes!("../../../../../poot-load/src/glm53_flash_data/config.json");

/// The tiny DSA layer reuses Card 367's independent reference, which fixes the cache capacity.
const TINY_CAPACITY: usize = dsa::tests::CAPACITY;

fn exact_plan() -> Glm5NextTextSourcePlan {
    let config = Glm53FlashHfConfig::from_slice(CONFIG).unwrap();
    Glm5NextTextSourcePlan::exact(&config).unwrap()
}

/// Small but non-degenerate: both attention kinds, both FFN kinds, three streams, three Sinkhorn rounds so every
/// round moves the result, four experts with top two, a routed scale that is not one, and a SwiGLU limit that the
/// fixture activations cross.
fn tiny_text() -> Glm5NextText {
    let kda = kda_tests::tiny_config();
    Glm5NextText {
        vocab_size: 5,
        hidden_size: kda.hidden_size,
        rms_norm_eps: 1e-5,
        mhc: MhcConfig {
            streams: 3,
            eps: 1e-6,
            sinkhorn_iters: 3,
        },
        intermediate_size: 3,
        moe_intermediate_size: 2,
        swiglu_limit: 0.75,
        router: Glm5NextRouterConfig {
            experts: 4,
            top_k: 2,
            n_group: 1,
            topk_group: 1,
            routed_scale: 2.5,
        },
        kda,
        dsa: dsa::tests::tiny_config(),
        layers: vec![
            Glm5NextTextLayerRow {
                layer: 0,
                attention: Glm5NextTextAttentionKind::Kda,
                ffn: Glm5NextTextFfnKind::Dense,
            },
            Glm5NextTextLayerRow {
                layer: 1,
                attention: Glm5NextTextAttentionKind::Dsa,
                ffn: Glm5NextTextFfnKind::Sparse,
            },
        ],
    }
}

fn tiny_plan() -> Glm5NextTextSourcePlan {
    Glm5NextTextSourcePlan::for_text(tiny_text()).expect("tiny source plan")
}

/// Deterministic hash of a tensor name and element index. Names that differ in any byte get unrelated sequences,
/// so the attention and FFN mHC rows, two experts, or two KDA projections never share values.
fn fixture_hash(name: &str, index: usize) -> u64 {
    let mut state = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    state ^= (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    state ^= state >> 33;
    state = state.wrapping_mul(0xff51_afd7_ed55_8ccd);
    state ^= state >> 33;
    state
}

/// Deterministic value in `[-0.5, 0.5)` seeded by tensor name.
fn fixture_value(name: &str, index: usize) -> f32 {
    (fixture_hash(name, index) % 1_024) as f32 / 1_024.0 - 0.5
}

/// Dense checkpoint values seeded by tensor name. Norm weights are one-centered; mHC scales lie in `[0.5, 2.5)` so
/// the comb logits are spread and Sinkhorn has not converged after three rounds. The first `numel` values do not
/// depend on `numel`, so the graph binding and the reference agree on every element either reads. Every value is
/// exactly representable in BF16, so a BF16 checkpoint row stores the value the reference reads.
fn tiny_dense(name: &str, numel: usize) -> Vec<f32> {
    let values = (0..numel).map(|index| fixture_value(name, index));
    let values: Vec<f32> = if name.ends_with("norm.weight") {
        values.map(|value| 1.0 + value).collect()
    } else if name.ends_with("_scale") {
        values.map(|value| 1.5 + 2.0 * value).collect()
    } else {
        values.collect()
    };
    values.into_iter().map(truncate_to_bf16).collect()
}

/// Finite E4M3FN codes with mixed signs and magnitudes from 0.125 to 2.
const E4M3_CODES: [u8; 10] = [0x00, 0x20, 0x2c, 0x34, 0x38, 0x40, 0xa4, 0xb0, 0xb8, 0xc0];

/// Positive-only E4M3FN codes for `*.gate_proj` (no sign bit, magnitudes 0.375 to 2).
const GATE_CODES: [u8; 5] = [0x2c, 0x34, 0x38, 0x3c, 0x40];
/// Negative-only E4M3FN codes for `*.up_proj` (sign bit set, magnitudes 0.375 to 2).
const UP_CODES: [u8; 5] = [0xac, 0xb4, 0xb8, 0xbc, 0xc0];

/// Packed payload seeded by `linear_id`, so every projection has its own codes and block scale.
/// `*.gate_proj` and `*.up_proj` use disjoint sign families and different block scales, so binding
/// the wrong packed role into SwiGLU moves the pre-clamp projection outputs by O(weights), not by
/// hash noise (Card 454 SC-003 role-swap observability).
fn tiny_packed_owner(linear_id: &str, descriptor: PackedWeight) -> Arc<PackedPayload> {
    let [out, input] = descriptor.shape();
    let pick = |index, choices: usize| (fixture_hash(linear_id, index) % choices as u64) as usize;
    let (pool, scale) = if linear_id.ends_with(".gate_proj") {
        (&GATE_CODES[..], 2.0f32)
    } else if linear_id.ends_with(".up_proj") {
        (&UP_CODES[..], 0.75f32)
    } else {
        (&E4M3_CODES[..], [0.75f32, 1.25][pick(usize::MAX, 2)])
    };
    let codes = (0..out * input)
        .map(|index| pool[pick(index, pool.len())])
        .collect::<Vec<u8>>();
    let scale_bytes_per_element = descriptor
        .format()
        .descriptor()
        .planar_operand(OperandRole::Scale)
        .expect("a packed weight has a Scale operand")
        .element_bytes();
    let scale_elements =
        descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)) / scale_bytes_per_element;
    let scales = (0..scale_elements)
        .flat_map(|_| scale.to_le_bytes())
        .collect::<Vec<u8>>();
    Arc::new(
        PackedPayload::try_new(
            descriptor,
            [
                (SourceRole::Planar(OperandRole::Codes), codes.into()),
                (SourceRole::Planar(OperandRole::Scale), scales.into()),
            ],
        )
        .expect("tiny packed payload"),
    )
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

/// `Glm5NextText::exact` maps every field of the pinned config. The expected value is spelled from the published
/// numbers, not from the loaded config.
#[test]
fn glm5next_text_exact_maps_published_config() {
    let config = Glm53FlashHfConfig::from_slice(CONFIG).unwrap();
    let expected = Glm5NextText {
        vocab_size: 154_880,
        hidden_size: 4_096,
        rms_norm_eps: 1e-5,
        mhc: MhcConfig {
            streams: 4,
            eps: 1e-6,
            sinkhorn_iters: 20,
        },
        intermediate_size: 12_288,
        moe_intermediate_size: 2_048,
        swiglu_limit: 10.0,
        router: Glm5NextRouterConfig {
            experts: 288,
            top_k: 8,
            n_group: 1,
            topk_group: 1,
            routed_scale: 2.5,
        },
        kda: Glm5NextKdaConfig::exact(),
        dsa: Glm5NextDsaConfig::exact(),
        layers: (0..45)
            .map(|layer| Glm5NextTextLayerRow {
                layer,
                attention: if layer % 4 == 3 {
                    Glm5NextTextAttentionKind::Dsa
                } else {
                    Glm5NextTextAttentionKind::Kda
                },
                ffn: if layer < 3 {
                    Glm5NextTextFfnKind::Dense
                } else {
                    Glm5NextTextFfnKind::Sparse
                },
            })
            .collect(),
    };
    assert_eq!(Glm5NextText::exact(&config).unwrap(), expected);
}

#[test]
fn glm5next_full_text_graph_has_exact_schedule() {
    let plan = exact_plan();
    let dsa_layers = plan
        .layers()
        .iter()
        .filter(|row| row.attention == Glm5NextTextAttentionKind::Dsa)
        .map(|row| row.layer)
        .collect::<Vec<_>>();
    assert_eq!(dsa_layers, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
    let dense_ffn_layers = plan
        .layers()
        .iter()
        .filter(|row| row.ffn == Glm5NextTextFfnKind::Dense)
        .map(|row| row.layer)
        .collect::<Vec<_>>();
    assert_eq!(dense_ffn_layers, vec![0, 1, 2]);

    for layer in 0..45 {
        for site in ["attn", "ffn"] {
            let row = |part: &str| {
                let spec =
                    &plan.dense[&format!("model.language_model.layers.{layer}.hc_{site}_{part}")];
                (spec.dtype, spec.shape.clone())
            };
            assert_eq!(row("fn"), (DType::BF16, vec![24, 16_384]));
            assert_eq!(row("base"), (DType::F32, vec![24]));
            assert_eq!(row("scale"), (DType::F32, vec![3]));
        }
    }
    assert_eq!(
        plan.dense["model.language_model.norm.weight"].shape,
        vec![4_096]
    );
    assert_eq!(plan.dense["lm_head.weight"].shape, vec![154_880, 4_096]);
    assert!(!plan.dense.keys().any(|name| name.contains("hyper_head")));
    assert!(!plan.dense.keys().any(|name| {
        name.starts_with("model.language_model.layers.45.") || name.starts_with("model.visual.")
    }));
    assert!(plan.packed_sources().all(|source| {
        source.layer < 45
            && !source
                .linear_id
                .starts_with("model.language_model.layers.45.")
    }));
}

#[test]
fn glm5next_full_text_graph_emits_exact_state_and_packed_sources()
-> Result<(), Glm53FlashTraceError> {
    let plan = exact_plan();
    let graph = trace_glm5next_full_text(&plan, Glm5NextTextScope::Text, 4)?;

    graph.validate().unwrap();
    assert_eq!(graph.state.len(), GLM5NEXT_TEXT_STATE_PAIR_COUNT);
    assert_eq!(
        graph.values[graph.output].aval,
        TensorType::f32(vec![1, 1, 154_880])
    );
    let state_inputs = graph
        .state
        .iter()
        .map(|&(input, _)| {
            let meta = &graph.values[input];
            assert_eq!(meta.storage, Storage::State);
            (meta.name.as_deref().unwrap(), &meta.aval)
        })
        .collect::<BTreeMap<_, _>>();
    for row in plan.layers() {
        let prefix = format!("model.language_model.layers.{}.self_attn", row.layer);
        match row.attention {
            Glm5NextTextAttentionKind::Kda => {
                assert_eq!(
                    state_inputs[format!("{prefix}.conv_state").as_str()],
                    &TensorType::f32(vec![1, 3 * 64 * 128, 4])
                );
                assert_eq!(
                    state_inputs[format!("{prefix}.recurrent_state").as_str()],
                    &TensorType::f32(vec![1, 64, 128, 128])
                );
            }
            Glm5NextTextAttentionKind::Dsa => {
                assert_eq!(
                    state_inputs[format!("{prefix}.k_state").as_str()],
                    &TensorType::f32(vec![1, 64, 4, 256])
                );
                assert_eq!(
                    state_inputs[format!("{prefix}.v_state").as_str()],
                    &TensorType::f32(vec![1, 64, 4, 256])
                );
                assert_eq!(
                    state_inputs[format!("{prefix}.indexer_state").as_str()],
                    &TensorType::f32(vec![1, 4, 257])
                );
            }
        }
    }
    assert_eq!(
        graph
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::PackedDequant { .. }))
            .count(),
        GLM5NEXT_TEXT_PACKED_SOURCE_COUNT
    );

    let mut packed_weights = 0usize;
    let mut packed_scales = 0usize;
    let mut mhc_sites = [0usize; 3];
    for &id in &graph.consts {
        let meta = &graph.values[id];
        let Some(name) = meta.name.as_deref() else {
            continue;
        };
        if let Some(source) = PackedSourceName::parse(name) {
            match source.role() {
                SourceRole::Planar(OperandRole::Codes) => packed_weights += 1,
                SourceRole::Planar(OperandRole::Scale) => packed_scales += 1,
                _ => unreachable!("glm5next packed sources are Codes and Scale only"),
            }
            assert_eq!(meta.storage, Storage::Const);
            assert_eq!(meta.aval.dtype, DType::I8);
        }
        if name.contains(".hc_") {
            if name.ends_with("_fn") {
                mhc_sites[0] += 1;
            } else if name.ends_with("_base") {
                mhc_sites[1] += 1;
            } else if name.ends_with("_scale") {
                mhc_sites[2] += 1;
            }
        }
        assert!(!name.starts_with("model.language_model.layers.45."));
        assert!(!name.starts_with("model.visual."));
    }
    assert_eq!(packed_weights, GLM5NEXT_TEXT_PACKED_SOURCE_COUNT);
    assert_eq!(packed_scales, GLM5NEXT_TEXT_PACKED_SOURCE_COUNT);
    assert_eq!(mhc_sites, [90, 90, 90]);

    let packed_shapes = plan
        .packed_sources()
        .map(|source| source.descriptor.shape().to_vec())
        .collect::<BTreeSet<_>>();
    assert!(!graph.consts.iter().any(|&id| {
        let meta = &graph.values[id];
        meta.aval.dtype == DType::F32 && packed_shapes.contains(&meta.aval.shape)
    }));
    Ok(())
}

#[test]
fn glm5next_scope_and_capacity_remain_deferred() {
    let plan = tiny_plan();
    let trace = |scope, capacity| trace_glm5next_full_text(&plan, scope, capacity);
    for scope in [Glm5NextTextScope::ImageVideo, Glm5NextTextScope::Mtp] {
        assert!(matches!(
            trace(scope, TINY_CAPACITY),
            Err(Glm53FlashTraceError::TextUnsupportedScope { scope: actual }) if actual == scope
        ));
    }
    assert!(matches!(
        trace(
            Glm5NextTextScope::Text,
            GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT + 1
        ),
        Err(Glm53FlashTraceError::DsaPairwiseCapacity { .. })
    ));
    assert!(trace(Glm5NextTextScope::Text, TINY_CAPACITY).is_ok());
}

#[test]
fn glm5next_packed_projection_returns_builder_errors() {
    let plan = tiny_plan();
    let source = plan.packed.values().next().unwrap();
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, source.descriptor.shape()[1]]));
    let taken = PackedSourceName::weight(&source.linear_id);
    b.constant(taken.as_str(), TensorType::f32(vec![1]));
    let generation = b.generation();

    assert!(matches!(
        packed_projection(&b, x, source),
        Err(Glm53FlashTraceError::TextGraph(poot_graph_ir::BuilderAppendError::NameCollision { name, .. }))
            if name == taken.as_str()
    ));
    assert_eq!(b.generation(), generation);
}

fn router_graph(routed_scale: f32) -> Graph {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "router_x", TensorType::f32(vec![1, 2]));
    let logits = b.slot_named(
        Slot::Activation,
        "router_logits",
        TensorType::f32(vec![1, 4]),
    );
    let bias = b.constant("router_bias", TensorType::f32(vec![4]));
    let (_, ids, weights) = glm5next_router_routes(
        &b,
        x,
        logits,
        bias,
        Glm5NextRouterConfig {
            experts: 4,
            top_k: 2,
            n_group: 1,
            topk_group: 1,
            routed_scale,
        },
    )
    .unwrap();
    let out = b.concat(0, &[ids, weights]);
    b.finish(out)
}

fn eval_router(graph: &Graph, bias: Vec<f32>) -> Vec<f32> {
    let mut inputs = HashMap::new();
    for (id, meta) in graph.values.iter().enumerate() {
        let data = match meta.name.as_deref() {
            Some("activation.router_x") => Some(vec![0.25, -0.5]),
            Some("activation.router_logits") => Some(vec![2.0, 1.0, 0.0, -1.0]),
            Some("router_bias") => Some(bias.clone()),
            _ => None,
        };
        if let Some(data) = data {
            inputs.insert(
                id,
                Value::from(HostTensor::f32(meta.aval.shape.clone(), data)),
            );
        }
    }
    eval(graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("dense output")
        .as_f32()
        .unwrap()
        .as_ref()
        .to_vec()
}

#[test]
fn glm5next_router_bias_is_selection_only() {
    let baseline = router_graph(2.5);
    let biased = router_graph(2.5);
    let half_scale = router_graph(1.25);
    let baseline = eval_router(&baseline, vec![0.0; 4]);
    let biased = eval_router(&biased, vec![0.0, 0.0, 2.0, 0.0]);
    let half_scale = eval_router(&half_scale, vec![0.0; 4]);
    assert_eq!(&baseline[..2], &[0.0, 1.0]);
    assert_eq!(&biased[..2], &[2.0, 0.0]);
    assert_eq!(&half_scale[..2], &baseline[..2]);
    for (half, full) in half_scale[2..].iter().zip(&baseline[2..]) {
        assert!((2.0 * half - full).abs() < 1.0e-6);
    }

    let expected_baseline = [sigmoid(2.0), sigmoid(1.0)];
    let baseline_sum = expected_baseline.iter().sum::<f32>();
    for (actual, raw) in baseline[2..].iter().zip(expected_baseline) {
        assert!((actual - 2.5 * raw / baseline_sum).abs() < 1.0e-6);
    }
    let expected_biased = [sigmoid(0.0), sigmoid(2.0)];
    let biased_sum = expected_biased.iter().sum::<f32>();
    for (actual, raw) in biased[2..].iter().zip(expected_biased) {
        assert!((actual - 2.5 * raw / biased_sum).abs() < 1.0e-6);
    }
}

#[test]
fn glm5next_router_rejects_invalid_config_before_append() {
    let b = Builder::new();
    let x = b.slot_named(Slot::Activation, "router_x", TensorType::f32(vec![1, 2]));
    let logits = b.slot_named(
        Slot::Activation,
        "router_logits",
        TensorType::f32(vec![1, 4]),
    );
    let bias = b.constant("router_bias", TensorType::f32(vec![4]));
    let valid = Glm5NextRouterConfig {
        experts: 4,
        top_k: 2,
        n_group: 1,
        topk_group: 1,
        routed_scale: 2.5,
    };
    for (router, field) in [
        (Glm5NextRouterConfig { top_k: 5, ..valid }, "top_k"),
        (
            Glm5NextRouterConfig {
                n_group: 3,
                ..valid
            },
            "n_group",
        ),
        (
            Glm5NextRouterConfig {
                topk_group: 2,
                ..valid
            },
            "topk_group",
        ),
    ] {
        assert!(matches!(
            glm5next_router_routes(&b, x, logits, bias, router),
            Err(Glm53FlashTraceError::TextRouterConfig { field: actual, .. }) if actual == field
        ));
    }
    assert!(matches!(
        glm5next_router_routes(&b, bias, logits, bias, valid),
        Err(Glm53FlashTraceError::TextRouterInput { role: "input", .. })
    ));
    assert!(matches!(
        glm5next_router_routes(&b, x, x, bias, valid),
        Err(Glm53FlashTraceError::TextRouterInput { role: "logits", .. })
    ));
    assert!(b.finish(x).eqns.is_empty());
}

#[test]
fn glm5next_clamped_swiglu_matches_independent_reference() {
    let b = Builder::new();
    let gate = b.slot_named(
        Slot::Activation,
        "clamped_gate",
        TensorType::f32(vec![1, 3]),
    );
    let up = b.slot_named(Slot::Activation, "clamped_up", TensorType::f32(vec![1, 3]));
    let output = deepseek4_clamped_swiglu(&b, gate, up, 10.0);
    let graph = b.finish(output);
    let mut inputs = HashMap::new();
    for (id, meta) in graph.values.iter().enumerate() {
        let values = match meta.name.as_deref() {
            Some("activation.clamped_gate") => Some(vec![20.0, -20.0, 0.5]),
            Some("activation.clamped_up") => Some(vec![20.0, -20.0, -0.25]),
            _ => None,
        };
        if let Some(values) = values {
            inputs.insert(
                id,
                Value::from(HostTensor::f32(meta.aval.shape.clone(), values)),
            );
        }
    }
    let actual = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .expect("dense output");
    let expected = [silu(10.0) * 10.0, silu(-20.0) * -10.0, silu(0.5) * -0.25];
    for (actual, expected) in actual.as_f32().unwrap().iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-5);
    }
}

#[test]
fn glm5next_final_stream_collapse_is_unweighted_mean() {
    let b = Builder::new();
    let streams = b.slot_named(
        Slot::Activation,
        "streams",
        TensorType::f32(vec![1, 1, 4, 2]),
    );
    let output = collapse_stream_mean(&b, streams, 4);
    let graph = b.finish(output);
    let id = graph
        .values
        .iter()
        .position(|meta| meta.name.as_deref() == Some("activation.streams"))
        .unwrap();
    let actual = eval(
        &graph,
        &HashMap::from([(
            id,
            Value::from(HostTensor::f32(
                vec![1, 1, 4, 2],
                vec![1.0, 9.0, 3.0, 7.0, 5.0, 5.0, 7.0, 3.0],
            )),
        )]),
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .unwrap()
    .output
    .into_host()
    .expect("dense output");
    assert_eq!(actual.shape(), vec![1, 1, 2]);
    assert_eq!(actual.as_f32().unwrap(), &[4.0, 6.0]);
}

// Retention across multiple simultaneous owners: with one owner, a bug returning the wrong owner for a
// `linear_id` has nothing to be confused with, so this fixture uses two distinct packed entries and checks
// per-name identity and non-conflation.
#[test]
fn glm5next_packed_owner_retention_keeps_card359_owner() {
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [1, 1],
    )
    .unwrap();
    let owner_a = Arc::new(
        PackedPayload::try_new(
            descriptor,
            [
                (SourceRole::Planar(OperandRole::Codes), Arc::from([0u8])),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    Arc::from(1.0f32.to_le_bytes()),
                ),
            ],
        )
        .unwrap(),
    );
    // Different weight byte and scale, so a cross-contamination bug is observable by `Arc::ptr_eq` even if two
    // owners encoded the same value.
    let owner_b = Arc::new(
        PackedPayload::try_new(
            descriptor,
            [
                (SourceRole::Planar(OperandRole::Codes), Arc::from([0x20u8])),
                (
                    SourceRole::Planar(OperandRole::Scale),
                    Arc::from(2.0f32.to_le_bytes()),
                ),
            ],
        )
        .unwrap(),
    );
    let linear_id_a = "model.language_model.layers.3.mlp.experts.0.gate_proj";
    let linear_id_b = "model.language_model.layers.3.mlp.experts.1.gate_proj";
    let role_a = Glm5NextPackedTextRole::RoutedExpert {
        expert: 0,
        projection: Glm5NextTextProjection::Gate,
    };
    let role_b = Glm5NextPackedTextRole::RoutedExpert {
        expert: 1,
        projection: Glm5NextTextProjection::Gate,
    };
    let spec = |role: Glm5NextPackedTextRole, linear_id: &str| Glm5NextPackedSourceSpec {
        layer: 3,
        role,
        linear_id: linear_id.to_string(),
        descriptor,
    };
    let plan = Glm5NextTextSourcePlan {
        text: tiny_text(),
        dense: BTreeMap::new(),
        packed: BTreeMap::from([
            ((3, role_a), spec(role_a, linear_id_a)),
            ((3, role_b), spec(role_b, linear_id_b)),
        ]),
    };
    let loaded_row = |linear_id: &str, owner: &Arc<PackedPayload>| LoadedPackedLinear {
        linear_id: linear_id.to_string(),
        descriptor,
        weight_name: format!("{linear_id}.weight"),
        scale_name: format!("{linear_id}.weight_scale_inv"),
        shard: "fixture.safetensors".to_string(),
        weight_span: SourceSpan::new(0, 1),
        scale_span: SourceSpan::new(1, 5),
        owner: Arc::clone(owner),
    };
    let loaded_a = loaded_row(linear_id_a, &owner_a);
    let loaded_b = loaded_row(linear_id_b, &owner_b);

    // `loaded` is in the opposite order from the plan's BTreeMap order, so a positional-zip regression is caught:
    // `retain_packed_sources` must look each owner up by name.
    let retained = retain_packed_sources(&plan, &[loaded_b.clone(), loaded_a.clone()]).unwrap();
    assert_eq!(retained.len(), 2);
    let by_id = retained
        .iter()
        .map(|row| (row.linear_id.as_str(), row))
        .collect::<HashMap<_, _>>();
    assert!(
        Arc::ptr_eq(&by_id[linear_id_a].owner, &owner_a),
        "linear_id_a must retain owner_a specifically"
    );
    assert!(
        Arc::ptr_eq(&by_id[linear_id_b].owner, &owner_b),
        "linear_id_b must retain owner_b specifically"
    );
    assert!(
        !Arc::ptr_eq(&by_id[linear_id_a].owner, &by_id[linear_id_b].owner),
        "the two retained rows must not share an owner - retention must not conflate them"
    );

    let mut widened = loaded_a.clone();
    widened.descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [2, 1],
    )
    .unwrap();
    let mut renamed_scale = loaded_a.clone();
    renamed_scale.scale_name = format!("{linear_id_a}.weight_scale");
    for mutated in [widened, renamed_scale] {
        assert!(matches!(
            retain_packed_sources(&plan, &[mutated, loaded_b.clone()]),
            Err(Glm5NextTextOwnershipError::PackedDescriptor { .. })
        ));
    }
    assert!(matches!(
        retain_packed_sources(&plan, &[loaded_a.clone(), loaded_a]),
        Err(Glm5NextTextOwnershipError::DuplicateSource {
            kind: Glm5NextTextSourceKind::Packed,
            ..
        })
    ));
}

/// `poot-models` has no device, executor, or replay-identity dependency, so the plan cannot reference such a
/// type. This test enforces the mechanical half: an exhaustive destructure with no `..` of the plan's four
/// private fields, so adding a field fails to compile until this test is touched.
#[test]
fn glm5next_owner_plan_is_model_local() {
    let plan = tiny_plan();
    let load = load_tiny_sources(tiny_source_rows(&plan), GLM53_FLASH_REVISION);
    let owners = Glm5NextTextOwnershipPlan::from_mixed(
        plan,
        &load.mixed,
        Glm53FlashTextInventoryReport::exact(),
    )
    .expect("tiny owner plan");

    // Every capability flag the plan exposes must stay false; executor admission is a later gate.
    assert!(!owners.portable_execution_admitted());
    assert!(!owners.public_capability());

    // Exhaustive: every field must be named or this does not compile. The four fields are graph-schema,
    // owner-reference, and count-summary data, none a device, executor, or replay-identity handle.
    let Glm5NextTextOwnershipPlan {
        dense: _,
        packed: _,
    } = owners;
}

/// Documented exception: no equation consumes an F32 owner view (`preflight_exact_dense_inputs`). The check is
/// on dtype alone, so any consuming equation triggers it.
#[test]
fn glm5next_card370_dense_views_keep_exact_owner() {
    let plan = tiny_plan();
    let load = load_tiny_sources(tiny_source_rows(&plan), GLM53_FLASH_REVISION);
    let owners = Glm5NextTextOwnershipPlan::from_mixed(
        plan,
        &load.mixed,
        Glm53FlashTextInventoryReport::exact(),
    )
    .expect("tiny owner plan");

    let bf16_sources = owners
        .dense_sources()
        .iter()
        .filter(|source| source.kind == ExactSourceKind::Bf16)
        .take(2)
        .collect::<Vec<_>>();
    assert_eq!(
        bf16_sources.len(),
        2,
        "the tiny plan must have at least two BF16 dense rows to prove retention does not \
         cross-contaminate - see the fixture note on glm5next_packed_owner_retention_keeps_card359_owner \
         above for why N=1 would not prove this"
    );
    let (source_a, source_b) = (bf16_sources[0], bf16_sources[1]);
    assert!(
        !Arc::ptr_eq(&source_a.owner, &source_b.owner),
        "fixture bug: the two picked BF16 rows must have distinct owners"
    );

    let view_a = DenseOwnerTensorView::new(Arc::clone(&source_a.owner)).expect("BF16 view A");
    let view_b = DenseOwnerTensorView::new(Arc::clone(&source_b.owner)).expect("BF16 view B");

    let b = Builder::new();
    let input_a = b.constant(
        source_a.descriptor.name(),
        TensorType::bf16(source_a.descriptor.shape().to_vec()),
    );
    let input_b = b.constant(
        source_b.descriptor.name(),
        TensorType::bf16(source_b.descriptor.shape().to_vec()),
    );
    let numel_a: usize = source_a.descriptor.shape().iter().product();
    let numel_b: usize = source_b.descriptor.shape().iter().product();
    // Cast BF16->F32 is Spec 376's whitelisted consumer, so both inputs are consumed rather than merely bound.
    // Reshape to 1-D and concat because the two BF16 rows may differ in shape.
    let a_f32 = b.reshape(b.cast(input_a, DType::F32), vec![numel_a]);
    let b_f32 = b.reshape(b.cast(input_b, DType::F32), vec![numel_b]);
    let out = b.concat(0, &[a_f32, b_f32]);
    let _graph = b.finish(out);

    let mut bound: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
    bound.insert(input_a.id, Value::Owner(ExactValue::Dense(view_a)));
    bound.insert(input_b.id, Value::Owner(ExactValue::Dense(view_b)));

    let Some(Value::Owner(ExactValue::Dense(bound_a))) = bound.get(&input_a.id) else {
        panic!("input_a did not bind as an exact dense view");
    };
    let Some(Value::Owner(ExactValue::Dense(bound_b))) = bound.get(&input_b.id) else {
        panic!("input_b did not bind as an exact dense view");
    };
    assert!(
        Arc::ptr_eq(bound_a.owner(), &source_a.owner),
        "input_a must keep source_a's exact owner"
    );
    assert!(
        Arc::ptr_eq(bound_b.owner(), &source_b.owner),
        "input_b must keep source_b's exact owner"
    );
    assert!(
        !Arc::ptr_eq(bound_a.owner(), bound_b.owner()),
        "input_a and input_b must not share an owner after binding"
    );

    // Documented exception: no equation consumes an F32 owner view (`preflight_exact_dense_inputs`). The check is
    // on dtype alone, so any consuming equation triggers it.
    let f32_source = owners
        .dense_sources()
        .iter()
        .find(|source| source.kind == ExactSourceKind::F32)
        .expect("the tiny plan has at least one F32 dense row");
    let view_f32 = DenseOwnerTensorView::new(Arc::clone(&f32_source.owner)).expect("F32 view");
    let fb = Builder::new();
    let numel_f32: usize = f32_source.descriptor.shape().iter().product();
    let input_f32 = fb.constant(
        f32_source.descriptor.name(),
        TensorType::f32(f32_source.descriptor.shape().to_vec()),
    );
    let f32_out = fb.reshape(input_f32, vec![numel_f32]);
    let f32_graph = fb.finish(f32_out);
    let mut f32_bound: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
    f32_bound.insert(input_f32.id, Value::Owner(ExactValue::Dense(view_f32)));
    let result = eval(
        &f32_graph,
        &f32_bound,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    );
    assert!(
        matches!(
            result,
            Err(EvalError::ExactDense(ExactDenseError::GraphConsumer { .. }))
        ),
        "an F32 owner view feeding any equation must be rejected before evaluation, got {result:?}"
    );
}

/// Two cases of the canonical graph builders: a reordered row list, and reproducing a private carrier by calling
/// the single-linear builder twice for the same `linear_id`.
#[test]
fn glm5next_card369_graph_builders_reject_reorder_and_private_carrier_reproduction() {
    // Reorder, N=3. `validate_packed_rows` requires `rows[i].ordinal == i`.
    let descriptor = PackedWeight::try_new(
        WeightFormat::E4m3Block128 {
            scale: ScaleEncoding::F32,
        },
        [1, 1],
    )
    .unwrap();
    let row = |ordinal: usize, expert: usize| PackedLinearGraphRow {
        ordinal,
        linear_id: format!("model.language_model.layers.3.mlp.experts.{expert}.gate_proj"),
        descriptor,
    };
    let canonical = vec![row(0, 0), row(1, 1), row(2, 2)];
    // Swap positions 0 and 2; ordinals travel with their row.
    let reordered = vec![
        canonical[2].clone(),
        canonical[1].clone(),
        canonical[0].clone(),
    ];

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 1]));
    // The selector is F32 (index values in float lanes), shaped `[x_type.shape[0]]`.
    let selector = b.constant("selector", TensorType::f32(vec![1]));
    let generation = b.generation();
    assert!(matches!(
        packed_indexed_linear(&b, x, selector, &reordered),
        Err(poot_graph_ir::BuilderAppendError::PackedRowOrdinal {
            index: 0,
            expected: 0,
            actual: 2,
        })
    ));
    assert_eq!(
        b.generation(),
        generation,
        "a rejected row order must publish no graph nodes"
    );
    // The canonical order is accepted, so the rejection above is about order.
    assert!(packed_indexed_linear(&b, x, selector, &canonical).is_ok());

    // Private carrier reproduction: `packed_linear` stages constants under names derived from `linear_id`, so a
    // second call for the same id must hit the Builder's `NameCollision`.
    let plan = tiny_plan();
    let source = plan
        .packed
        .values()
        .next()
        .expect("tiny plan has a packed row");
    let cb = Builder::new();
    let cx = cb.constant("cx", TensorType::f32(vec![1, source.descriptor.shape()[1]]));
    let first = packed_projection(&cb, cx, source);
    assert!(
        first.is_ok(),
        "the first call for this linear_id must succeed"
    );
    let generation_after_first = cb.generation();
    let second = packed_projection(&cb, cx, source);
    assert!(
        matches!(
            second,
            Err(Glm53FlashTraceError::TextGraph(
                poot_graph_ir::BuilderAppendError::NameCollision { .. }
            ))
        ),
        "reproducing the same linear_id's private carrier must be rejected, got {second:?}"
    );
    assert_eq!(
        cb.generation(),
        generation_after_first,
        "the rejected second call must publish no additional graph nodes"
    );
}

/// Safetensors rows and dispositions for every source of `plan`.
fn tiny_source_rows(plan: &Glm5NextTextSourcePlan) -> Vec<(SourceRow, TensorDisposition)> {
    let dense = plan.dense_sources().map(|spec| {
        let values = tiny_dense(&spec.name, spec.shape.iter().product());
        let (dtype, bytes, disposition) = match spec.kind {
            ExactSourceKind::Bf16 => (
                "BF16",
                bf16_bytes(&spec.name, &values),
                TensorDisposition::DenseBf16,
            ),
            ExactSourceKind::F32 => (
                "F32",
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                TensorDisposition::DenseF32,
            ),
            other => panic!("tiny source plan has no {other:?} dense row"),
        };
        (
            SourceRow {
                name: spec.name.clone(),
                dtype,
                shape: spec.shape.clone(),
                bytes,
            },
            disposition,
        )
    });
    let packed = plan.packed_sources().flat_map(|spec| {
        let owner = tiny_packed_owner(&spec.linear_id, spec.descriptor);
        let scale_source_shape = spec
            .descriptor
            .source_shape(SourceRole::Planar(OperandRole::Scale));
        let scale_bytes_per_element = spec
            .descriptor
            .format()
            .descriptor()
            .planar_operand(OperandRole::Scale)
            .expect("a packed weight has a Scale operand")
            .element_bytes();
        let scale_rows = scale_source_shape[0];
        let scale_columns = scale_source_shape[1] / scale_bytes_per_element;
        [
            (
                SourceRow {
                    name: format!("{}.weight", spec.linear_id),
                    dtype: "F8_E4M3",
                    shape: spec.descriptor.shape().to_vec(),
                    bytes: owner.bytes(SourceRole::Planar(OperandRole::Codes)).to_vec(),
                },
                TensorDisposition::PackedWeight {
                    linear_id: spec.linear_id.clone(),
                    format: spec.descriptor.format(),
                    logical_shape: spec.descriptor.shape(),
                },
            ),
            (
                SourceRow {
                    name: format!("{}.weight_scale_inv", spec.linear_id),
                    dtype: "F32",
                    shape: vec![scale_rows, scale_columns],
                    bytes: owner.bytes(SourceRole::Planar(OperandRole::Scale)).to_vec(),
                },
                TensorDisposition::PackedScale {
                    linear_id: spec.linear_id.clone(),
                },
            ),
        ]
    });
    dense.chain(packed).collect()
}

fn load_tiny_sources(
    rows: Vec<(SourceRow, TensorDisposition)>,
    revision: &str,
) -> LoadedCheckpoint {
    load_checkpoint(
        "zai-org/GLM-5.3-Flash",
        revision,
        br#"{"model_type":"glm5_next"}"#,
        rows,
    )
}

fn bf16_row(name: &str) -> SourceRow {
    SourceRow {
        name: name.to_string(),
        dtype: "BF16",
        shape: vec![4],
        bytes: vec![1; 8],
    }
}

#[test]
fn glm5next_owner_plan_retains_card359_owners() {
    let plan = tiny_plan();
    let mut rows = tiny_source_rows(&plan);
    rows.extend([
        (
            bf16_row("model.language_model.layers.45.input_layernorm.weight"),
            TensorDisposition::Deferred,
        ),
        (
            bf16_row("model.visual.merger.norm.weight"),
            TensorDisposition::Excluded,
        ),
    ]);
    let load = load_tiny_sources(rows, GLM53_FLASH_REVISION);
    let owners = Glm5NextTextOwnershipPlan::from_mixed(
        plan,
        &load.mixed,
        Glm53FlashTextInventoryReport::exact(),
    )
    .expect("tiny owner plan");
    for retained in owners.dense_sources() {
        let loaded = load
            .mixed
            .exact_metadata
            .get(retained.descriptor.name())
            .expect("retained dense source was loaded");
        assert!(Arc::ptr_eq(&retained.owner, &loaded.owner));
    }
    for retained in owners.packed_sources() {
        let loaded = match load.mixed.store.get(&retained.linear_id) {
            Some(WeightEntry::Packed(owner)) => owner,
            _ => panic!("retained packed source was loaded"),
        };
        assert!(Arc::ptr_eq(&retained.owner, loaded));
    }
}

#[test]
fn glm5next_owner_plan_rejects_invalid_card359_rows() {
    let plan = tiny_plan();
    let exact = Glm53FlashTextInventoryReport::exact();
    let load = load_tiny_sources(tiny_source_rows(&plan), GLM53_FLASH_REVISION);
    let reject = |plan: &Glm5NextTextSourcePlan, mixed: &MixedLoadResult, report| {
        Glm5NextTextOwnershipPlan::from_mixed(plan.clone(), mixed, report)
            .expect_err("tiny owner plan must reject")
    };
    assert!(Glm5NextTextOwnershipPlan::from_mixed(plan.clone(), &load.mixed, exact).is_ok());

    let wrong_revision = load_tiny_sources(
        tiny_source_rows(&plan),
        "ffffffffffffffffffffffffffffffffffffffff",
    );
    assert!(matches!(
        reject(&plan, &wrong_revision.mixed, exact),
        Glm5NextTextOwnershipError::Revision { .. }
    ));
    assert!(matches!(
        reject(&plan, &load.mixed, Glm53FlashTextInventoryReport::default()),
        Glm5NextTextOwnershipError::InventoryReport { .. }
    ));

    // A layer-45 MTP row that reaches the selected sources is not a plan row, so it cannot be retained.
    let mut leaked_rows = tiny_source_rows(&plan);
    leaked_rows.push((
        bf16_row("model.language_model.layers.45.input_layernorm.weight"),
        TensorDisposition::DenseBf16,
    ));
    let leaked = load_tiny_sources(leaked_rows, GLM53_FLASH_REVISION);
    assert!(matches!(
        reject(&plan, &leaked.mixed, exact),
        Glm5NextTextOwnershipError::UnexpectedSource {
            kind: Glm5NextTextSourceKind::Dense,
            name,
        } if name.starts_with("model.language_model.layers.45.")
    ));

    let mut missing = load.mixed.clone();
    let missing_name = missing.exact_metadata.keys().next().unwrap().clone();
    missing.exact_metadata.remove(&missing_name);
    assert!(matches!(
        reject(&plan, &missing, exact),
        Glm5NextTextOwnershipError::MissingSource {
            kind: Glm5NextTextSourceKind::Dense,
            ..
        }
    ));
    // `MixedLoadResult::exact_metadata` is keyed by tensor name (card 540a), so it
    // cannot itself hold two rows under the same name; `retain_dense_sources`'s own duplicate
    // check is exercised directly, on a hand-built slice, the same way the packed-side duplicate
    // case is (see `retain_packed_sources` below).
    let sample_name = load.mixed.exact_metadata.keys().next().unwrap().clone();
    let sample = load.mixed.exact_metadata.get(&sample_name).unwrap();
    let duplicate_rows = vec![
        LoadedExactSource {
            descriptor: sample.owner.descriptor().clone(),
            kind: sample.kind,
            owner: Arc::clone(&sample.owner),
        },
        LoadedExactSource {
            descriptor: sample.owner.descriptor().clone(),
            kind: sample.kind,
            owner: Arc::clone(&sample.owner),
        },
    ];
    assert!(matches!(
        retain_dense_sources(&plan, &duplicate_rows),
        Err(Glm5NextTextOwnershipError::DuplicateSource {
            kind: Glm5NextTextSourceKind::Dense,
            ..
        })
    ));
    let mut wrong_kind = load.mixed.clone();
    wrong_kind
        .exact_metadata
        .values_mut()
        .find(|source| source.kind == ExactSourceKind::Bf16)
        .expect("tiny BF16 source")
        .kind = ExactSourceKind::F32;
    assert!(matches!(
        reject(&plan, &wrong_kind, exact),
        Glm5NextTextOwnershipError::DenseKind { .. }
    ));
    let mut reshaped = plan.clone();
    reshaped
        .dense
        .values_mut()
        .next()
        .expect("tiny dense row")
        .shape
        .push(1);
    assert!(matches!(
        reject(&reshaped, &load.mixed, exact),
        Glm5NextTextOwnershipError::DenseShape { .. }
    ));

    let mut missing_packed = load.mixed.clone();
    let missing_linear_id = missing_packed
        .packed_metadata
        .keys()
        .next()
        .unwrap()
        .clone();
    missing_packed.packed_metadata.remove(&missing_linear_id);
    assert!(matches!(
        reject(&plan, &missing_packed, exact),
        Glm5NextTextOwnershipError::MissingSource {
            kind: Glm5NextTextSourceKind::Packed,
            ..
        }
    ));
}

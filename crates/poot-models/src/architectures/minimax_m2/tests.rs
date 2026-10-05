//! Card 365a acceptance. Every numeric expectation comes from [`MiniMaxReference`], a plain-f32
//! transcription of the MiniMax-M2 semantics that shares no code, no helper and no name table with the
//! tracers: it addresses packed weights by literal checkpoint module paths and decodes the fixture codes
//! itself.

use std::collections::HashMap;
use std::sync::Arc;

use poot_eval::Value;
use poot_graph_ir::{Graph, OpKind, PackedSourceName, Slot, Storage, ValidationChannel, ValueId};
use poot_load::minimax_m2::MiniMaxM2Manifest;
use poot_load::packed_safetensors::{ExactSourceOwner, TensorDisposition};
use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, SourceRole};
use poot_tensor::HostTensor;
use poot_test_util::seed_of;

use super::*;
use crate::test_support::safetensors::{
    LoadedCheckpoint, SourceRow, bf16_bytes, load_checkpoint, truncate_to_bf16,
};

const CAPACITY: usize = 6;
const PROMPT: [i32; 4] = [3, 7, 1, 5];

fn small_config(layers: usize, experts: usize) -> MiniMaxM2Config {
    let json = serde_json::json!({
        "architectures": ["MiniMaxM2ForCausalLM"],
        "attn_type_list": vec![1; layers],
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": 8,
        "intermediate_size": 6,
        "max_position_embeddings": 16,
        "model_type": "minimax_m2",
        "mtp_transformer_layers": 1,
        "num_attention_heads": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": layers,
        "num_key_value_heads": 2,
        "num_local_experts": experts,
        "num_mtp_modules": 3,
        "qk_norm_type": "per_layer",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "float8_e4m3fn",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["gate", "e_score_correction_bias", "lm_head"]
        },
        "rms_norm_eps": 1e-6,
        "rope_theta": 5_000_000,
        "rotary_dim": 2,
        "scoring_func": "sigmoid",
        "shared_intermediate_size": 0,
        "tie_word_embeddings": false,
        "use_cache": true,
        "use_mtp": true,
        "use_qk_norm": true,
        "use_routing_bias": true,
        "vocab_size": 12
    });
    MiniMaxM2Config::from_slice(&serde_json::to_vec(&json).expect("serialize config"))
        .expect("parse tiny MiniMax config")
}

fn plan(layers: usize, experts: usize) -> MiniMaxM2TextSourcePlan {
    MiniMaxM2TextSourcePlan::new(&small_config(layers, experts)).expect("tiny source plan")
}

// ---- fixture values -------------------------------------------------------------------------

/// SplitMix64 finalizer: neighbouring seeds give unrelated outputs.
fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// A BF16-exact fixture value that depends on the whole name, so a constant bound under the wrong name changes
/// the result.
///
/// Correction biases get a wide `(-2, 2)` spread instead of `(-0.5, 0.5)`. Sigmoid scores live in `(0, 1)`, so a
/// narrow bias barely reorders anything, and the oracle stayed green when the router gathered the biased scores
/// as its weights. A bias that can exceed the score range makes selection bias-driven.
fn dense_value(name: &str, index: usize) -> f32 {
    let pick = mix64(seed_of(name) ^ index as u64) % 29;
    let unit = pick as f32 / 29.0 - 0.5;
    let scale = if name.ends_with("e_score_correction_bias") {
        4.0
    } else {
        1.0
    };
    truncate_to_bf16(unit * scale)
}

fn dense_values(name: &str, numel: usize) -> Vec<f32> {
    (0..numel).map(|index| dense_value(name, index)).collect()
}

/// Finite E4M3FN codes with mixed signs and magnitudes.
const E4M3_CODES: [u8; 9] = [0x00, 0x20, 0x28, 0x2c, 0x30, 0x34, 0xa4, 0xa8, 0xb0];

/// Decode one E4M3FN code from its sign, exponent and mantissa fields.
fn e4m3_value(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((code >> 3) & 0x0f);
    let mantissa = f32::from(code & 0x07) / 8.0;
    let magnitude = if exponent == 0 {
        mantissa * 2f32.powi(-6)
    } else {
        (1.0 + mantissa) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// Weight codes and the block scale of the packed linear named `linear_id`. Seeding by the checkpoint
/// module path gives same-shaped projections such as `k_proj`/`v_proj`, or two experts' `w1`, different
/// weights, so a projection bound under the wrong name changes the result.
fn packed_codes(linear_id: &str, [out, input]: [usize; 2]) -> (Vec<u8>, f32) {
    let seed = seed_of(linear_id);
    let codes = (0..out * input)
        .map(|element| E4M3_CODES[(mix64(seed ^ element as u64) % 9) as usize])
        .collect();
    (codes, [0.75, 1.25][(mix64(seed) % 2) as usize])
}

/// The dequantized `[out, in]` weight of one packed linear, as the reference reads it.
fn packed_weight(linear_id: &str, shape: [usize; 2]) -> Vec<f32> {
    let (codes, scale) = packed_codes(linear_id, shape);
    codes
        .into_iter()
        .map(|code| e4m3_value(code) * scale)
        .collect()
}

fn rope_table(name: &str, positions: usize, rotary: usize) -> Vec<f32> {
    let cosine = name.ends_with("cos");
    (0..positions)
        .flat_map(|position| {
            (0..rotary).map(move |dim| {
                // Row 0 is not the identity, so a skipped rotation cannot match at any position.
                let angle = 0.7 * (position + 1) as f32 / (1 + dim % (rotary / 2).max(1)) as f32;
                if cosine { angle.cos() } else { angle.sin() }
            })
        })
        .collect()
}

/// Every F32 constant: the ones the graph names itself, then the checkpoint's own F32 rows.
fn constant_values(name: &str, shape: &[usize]) -> Vec<f32> {
    let numel = shape.iter().product::<usize>();
    match name {
        MINIMAX_M2_ROPE_COS | MINIMAX_M2_ROPE_SIN => rope_table(name, shape[0], shape[1]),
        MINIMAX_M2_EXPERT_IOTA => (0..numel).map(|index| index as f32).collect(),
        // The router gate and its correction bias: dense F32 checkpoint rows.
        _ => dense_values(name, numel),
    }
}

/// Additive visibility for `tokens` queries whose first is at `first_position`.
fn mask(tokens: usize, first_position: usize, capacity: usize) -> Vec<f32> {
    (0..tokens)
        .flat_map(|query| {
            (0..capacity).map(move |key| {
                if key <= first_position + query {
                    0.0
                } else {
                    -1.0e9
                }
            })
        })
        .collect()
}

// ---- owners ---------------------------------------------------------------------------------

struct Owners {
    packed: HashMap<String, Arc<PackedPayload>>,
    dense: HashMap<String, Arc<ExactSourceOwner>>,
    _checkpoint: LoadedCheckpoint,
}

fn packed_scale_elements(descriptor: poot_quant::PackedWeight) -> usize {
    let bytes_per_element = descriptor
        .format()
        .descriptor()
        .planar_operand(OperandRole::Scale)
        .expect("a packed weight has a Scale operand")
        .element_bytes();
    descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)) / bytes_per_element
}

fn owners(plan: &MiniMaxM2TextSourcePlan) -> Owners {
    let table = plan.table();
    let packed = table
        .packed()
        .map(|source| {
            let descriptor = source.descriptor();
            let (codes, scale) = packed_codes(source.linear_id(), descriptor.shape());
            let scales = (0..packed_scale_elements(descriptor))
                .flat_map(|_| scale.to_le_bytes())
                .collect::<Vec<_>>();
            let payload = PackedPayload::try_new(
                descriptor,
                [
                    (SourceRole::Planar(OperandRole::Codes), codes.into()),
                    (SourceRole::Planar(OperandRole::Scale), scales.into()),
                ],
            )
            .expect("tiny packed payload");
            (source.linear_id().to_string(), Arc::new(payload))
        })
        .collect();

    // Only the BF16 rows need Card 359 owners: owner-backed binding of every role is card 365c's evidence.
    let rows = table
        .dense()
        .filter(|source| source.role() == MiniMaxM2DenseRole::Bf16)
        .map(|source| {
            let name = source.name().to_string();
            let numel = source.shape().iter().product::<usize>();
            let row = SourceRow {
                bytes: bf16_bytes(&name, &dense_values(&name, numel)),
                name,
                dtype: "BF16",
                shape: source.shape().to_vec(),
            };
            (row, TensorDisposition::DenseBf16)
        })
        .collect();
    let checkpoint = load_checkpoint("MiniMaxAI/MiniMax-M2.5", "tiny-fixture", b"{}", rows);
    Owners {
        packed,
        dense: checkpoint
            .mixed
            .exact_metadata
            .iter()
            .map(|(name, source)| (name.clone(), Arc::clone(&source.owner)))
            .collect(),
        _checkpoint: checkpoint,
    }
}

/// Bind every graph input. BF16 checkpoint rows become card 370 owner views directly rather than through
/// `poot_eval::exact_dense::bind_dense_owners`, which takes a `Graph<NoValidations>` and cannot see a
/// validation-bearing graph (card 365c's seam).
fn bindings(
    graph: &Graph<ValidationOutputs>,
    owners: &Owners,
    tokens: &[i32],
    position: Option<usize>,
    carried: Option<&[Value]>,
) -> HashMap<ValueId, Value> {
    let state_index = graph
        .state
        .iter()
        .enumerate()
        .map(|(index, &(state_in, _))| (state_in, index))
        .collect::<HashMap<_, _>>();
    let first_position = position.unwrap_or(0);
    graph
        .inputs
        .iter()
        .map(|&id| {
            let meta = graph.meta(id);
            let name = meta.name.as_deref().unwrap_or_default();
            let shape = meta.aval.shape.clone();
            if let Some(source) = PackedSourceName::parse(name) {
                let owner = owners
                    .packed
                    .get(source.linear_id())
                    .unwrap_or_else(|| panic!("no packed owner {}", source.linear_id()));
                return (
                    id,
                    Value::Packed(PackedComponentRef::new(Arc::clone(owner), source.role())),
                );
            }
            if let Some(owner) = owners.dense.get(name) {
                // A card 370 owner view, not a plain `Value::Host` f32 mirror: the embedding `Gather` and the
                // LM head's `Transpose` read their BF16 source directly (card 365a's "reads the BF16 source
                // directly" doc note), and `operand::bf16_view` - the walk's one reader for a raw BF16
                // `Transpose`/`Gather`/`Reshape` operand - has no `Value::Host` arm at all (Card 554d, the
                // BF16 sibling of the I32-slot binding regression).
                return (
                    id,
                    Value::from(
                        poot_eval::exact_dense::DenseOwnerTensorView::new(Arc::clone(owner))
                            .unwrap_or_else(|error| panic!("{name}: bf16 owner view: {error}")),
                    ),
                );
            }
            let value = match meta.storage {
                Storage::Slot(Slot::Token) => Value::Host(HostTensor::i32(shape, tokens.to_vec())),
                Storage::Slot(Slot::Pos) => Value::Host(HostTensor::i32(
                    shape,
                    vec![i32::try_from(first_position).expect("tiny position")],
                )),
                Storage::Slot(Slot::Mask) => Value::Host(HostTensor::f32(
                    shape.clone(),
                    mask(shape[0], first_position, shape[1]),
                )),
                Storage::State => carried.map_or_else(
                    || Value::Host(HostTensor::f32(shape.clone(), vec![0.0; meta.aval.numel()])),
                    |values| values[state_index[&id]].clone(),
                ),
                Storage::Const => Value::Host(HostTensor::f32(
                    shape.clone(),
                    constant_values(name, &shape),
                )),
                other => panic!("unexpected tiny graph input storage {other:?}"),
            };
            (id, value)
        })
        .collect()
}

fn dense_rows(value: &Value) -> &HostTensor {
    match value {
        Value::Host(tensor) => tensor,
        other => panic!("expected a dense value, got {other:?}"),
    }
}

// ---- the independent reference --------------------------------------------------------------

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inverse = (mean_square + eps).sqrt().recip();
    x.iter().zip(weight).map(|(v, w)| v * inverse * w).collect()
}

fn matvec(weight: &[f32], x: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect()
}

fn projection(linear_id: &str, out: usize, x: &[f32]) -> Vec<f32> {
    matvec(&packed_weight(linear_id, [out, x.len()]), x)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Half-split rotation over the leading `rotary` values; the tail passes through.
fn partial_rope(head: &mut [f32], position: usize, rotary: usize) {
    let cos = rope_table(MINIMAX_M2_ROPE_COS, position + 1, rotary);
    let sin = rope_table(MINIMAX_M2_ROPE_SIN, position + 1, rotary);
    let (cos, sin) = (&cos[position * rotary..], &sin[position * rotary..]);
    let half = rotary / 2;
    let source = head[..rotary].to_vec();
    for dim in 0..rotary {
        let rotated = if dim < half {
            -source[dim + half]
        } else {
            source[dim - half]
        };
        head[dim] = source[dim] * cos[dim] + rotated * sin[dim];
    }
}

struct LayerCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

/// A token-by-token MiniMax-M2 text model in plain f32 loops.
///
/// It shares no math and no source table with the tracers. Packed weights are addressed by literal
/// checkpoint module paths and decoded from the fixture codes here.
struct MiniMaxReference {
    config: MiniMaxM2Config,
    layers: Vec<LayerCache>,
    /// The post-attention normed activation each block handed its router, newest step only.
    router_inputs: Vec<Vec<f32>>,
}

impl MiniMaxReference {
    fn new(config: MiniMaxM2Config) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| LayerCache {
                keys: vec![Vec::new(); CAPACITY],
                values: vec![Vec::new(); CAPACITY],
            })
            .collect();
        Self {
            config,
            layers,
            router_inputs: Vec::new(),
        }
    }

    fn step(&mut self, token: i32, position: usize) -> Vec<f32> {
        let config = self.config.clone();
        self.router_inputs.clear();
        let hidden = config.hidden_size;
        let token = usize::try_from(token).expect("token id");
        let embedding = dense_values("model.embed_tokens.weight", config.vocab_size * hidden);
        let mut h = embedding[token * hidden..(token + 1) * hidden].to_vec();

        for layer in 0..config.num_hidden_layers {
            let root = format!("model.layers.{layer}");
            let normed = rms_norm(
                &h,
                &dense_values(&format!("{root}.input_layernorm.weight"), hidden),
                config.rms_norm_eps,
            );
            let attention = self.attention(layer, &root, &normed, position);
            for (value, update) in h.iter_mut().zip(&attention) {
                *value += update;
            }
            let normed = rms_norm(
                &h,
                &dense_values(&format!("{root}.post_attention_layernorm.weight"), hidden),
                config.rms_norm_eps,
            );
            self.router_inputs.push(normed.clone());
            let routed = self.routed_moe(&root, &normed);
            for (value, update) in h.iter_mut().zip(&routed) {
                *value += update;
            }
        }

        let normed = rms_norm(
            &h,
            &dense_values("model.norm.weight", hidden),
            config.rms_norm_eps,
        );
        matvec(
            &dense_values("lm_head.weight", config.vocab_size * hidden),
            &normed,
        )
    }

    fn attention(&mut self, layer: usize, root: &str, x: &[f32], position: usize) -> Vec<f32> {
        let config = self.config.clone();
        let (head_dim, heads, kv_heads) = (
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        let mut q = projection(&format!("{root}.self_attn.q_proj"), config.q_dim(), x);
        let mut k = projection(&format!("{root}.self_attn.k_proj"), config.kv_dim(), x);
        let v = projection(&format!("{root}.self_attn.v_proj"), config.kv_dim(), x);

        // Projection-wide, over the whole flattened projection, before the head split.
        q = rms_norm(
            &q,
            &dense_values(&format!("{root}.self_attn.q_norm.weight"), config.q_dim()),
            config.rms_norm_eps,
        );
        k = rms_norm(
            &k,
            &dense_values(&format!("{root}.self_attn.k_norm.weight"), config.kv_dim()),
            config.rms_norm_eps,
        );
        for head in 0..heads {
            partial_rope(
                &mut q[head * head_dim..(head + 1) * head_dim],
                position,
                config.rotary_dim,
            );
        }
        for head in 0..kv_heads {
            partial_rope(
                &mut k[head * head_dim..(head + 1) * head_dim],
                position,
                config.rotary_dim,
            );
        }
        // Post-RoPE keys and direct values enter the cache.
        self.layers[layer].keys[position] = k;
        self.layers[layer].values[position] = v;

        let scale = config.attention_scale();
        let groups = config.kv_groups();
        let mut context = vec![0.0; heads * head_dim];
        for head in 0..heads {
            let kv_head = head / groups; // blocked GQA, matching `ops::repeat_kv`
            let query = &q[head * head_dim..(head + 1) * head_dim];
            let scores = (0..=position)
                .map(|slot| {
                    let key = &self.layers[layer].keys[slot][kv_head * head_dim..][..head_dim];
                    query.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
                })
                .collect::<Vec<_>>();
            let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores.iter().map(|s| (s - peak).exp()).collect::<Vec<_>>();
            let total = weights.iter().sum::<f32>();
            for (slot, weight) in weights.iter().enumerate() {
                let value = &self.layers[layer].values[slot][kv_head * head_dim..][..head_dim];
                for (dim, v) in value.iter().enumerate() {
                    context[head * head_dim + dim] += weight / total * v;
                }
            }
        }
        projection(
            &format!("{root}.self_attn.o_proj"),
            config.hidden_size,
            &context,
        )
    }

    /// Sigmoid routing, selection-only correction bias, stable top-k, unbiased renormalized weights, and
    /// `w2(silu(w1 x) * w3 x)` accumulated over the selected experts. No shared branch, no route scale.
    fn routed_moe(&self, root: &str, x: &[f32]) -> Vec<f32> {
        let config = &self.config;
        let (hidden, experts, top_k) = (
            config.hidden_size,
            config.num_local_experts,
            config.num_experts_per_tok,
        );
        let gate = dense_values(
            &format!("{root}.block_sparse_moe.gate.weight"),
            experts * hidden,
        );
        let bias = dense_values(
            &format!("{root}.block_sparse_moe.e_score_correction_bias"),
            experts,
        );
        let raw = matvec(&gate, x)
            .into_iter()
            .map(sigmoid)
            .collect::<Vec<_>>();

        let mut order = (0..experts).collect::<Vec<_>>();
        // Descending by the bias-adjusted score, lower expert id winning a tie.
        order.sort_by(|&a, &b| {
            (raw[b] + bias[b])
                .partial_cmp(&(raw[a] + bias[a]))
                .expect("finite fixture scores")
                .then(a.cmp(&b))
        });
        let selected = &order[..top_k];
        let total = selected.iter().map(|&id| raw[id]).sum::<f32>();

        let mut output = vec![0.0; hidden];
        for &id in selected {
            let expert = format!("{root}.block_sparse_moe.experts.{id}");
            let gate_out = projection(&format!("{expert}.w1"), config.intermediate_size, x);
            let up = projection(&format!("{expert}.w3"), config.intermediate_size, x);
            let activated = gate_out
                .iter()
                .zip(&up)
                .map(|(g, u)| silu(*g) * u)
                .collect::<Vec<_>>();
            let down = projection(&format!("{expert}.w2"), hidden, &activated);
            let weight = raw[id] / total;
            for (value, update) in output.iter_mut().zip(&down) {
                *value += weight * update;
            }
        }
        output
    }
}

// ---- acceptance -----------------------------------------------------------------------------

/// Evaluate a MiniMax text graph on the CPU.
///
/// `eval_value_with_state` cannot run these graphs: its storage-aware loop evaluates only F32-output equations,
/// so the I32 arithmetic from card 372c's `guard_index_bounds` (`GeU`, `Sub`, `Select`) has no arm.
/// `eval_exact_i32_with_state` (card 382) can, and needs a cast-role callback. A model test cannot call
/// `poot-graph-plan`, so the role is read off the graph: a MiniMax graph has exactly one I32-to-F32 cast, the
/// token guard's witness (asserted by `minimax_m2_has_exactly_one_i32_cast`). Only the logits are read; the
/// prefill row writes each key into the cache and reads it back within one evaluation.
fn eval_graph(
    graph: &Graph<ValidationOutputs>,
    inputs: &HashMap<ValueId, Value>,
) -> Result<Value, poot_eval::EvalError> {
    let ceiling = graph.values.len();
    let mut roles = |_: ValueId| {
        Some(poot_eval::cast_authority::ExactI32CastRole::Witness {
            selected_element_ceiling: ceiling,
        })
    };
    let evaluation = poot_eval::eval(
        graph,
        inputs,
        poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED).cast_authority(&mut roles),
    )?;
    Ok(evaluation.output)
}

/// The only I32-to-F32 cast in a MiniMax graph is the token guard's witness. Red if a tracer adds a second: a
/// selector cast would depend on card 371 and make `eval_graph`'s single role wrong.
#[test]
fn minimax_m2_has_exactly_one_i32_cast() {
    let plan = plan(2, 4);
    for graph in [
        trace_minimax_m2_text_decode(&plan, CAPACITY).expect("decode graph"),
        trace_minimax_m2_text_prefill(&plan, PROMPT.len(), CAPACITY).expect("prefill graph"),
    ] {
        let casts = graph
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::Cast { .. }))
            .filter(|eqn| match eqn.inputs.first() {
                Some(poot_graph_ir::Operand::Value(id)) => {
                    graph.aval(*id).dtype == poot_tensor::DType::I32
                }
                _ => false,
            })
            .map(|eqn| eqn.out)
            .collect::<Vec<_>>();
        assert_eq!(casts.len(), 1, "exactly one I32-to-F32 cast");
        let witness = graph
            .validation_outputs()
            .iter()
            .find(|output| output.name == MINIMAX_M2_TOKEN_INDEX_VALIDATION)
            .expect("the token guard declares a witness");
        assert!(
            graph.eqns.iter().any(|eqn| {
                eqn.out == witness.value
                    && eqn.inputs.iter().any(|operand| {
                        matches!(operand, poot_graph_ir::Operand::Value(id) if *id == casts[0])
                    })
            }),
            "the single cast is the token guard's witness"
        );
    }
}

fn producers<V: ValidationChannel>(graph: &Graph<V>) -> HashMap<ValueId, &OpKind> {
    graph.eqns.iter().map(|eqn| (eqn.out, &eqn.op)).collect()
}

fn named<V: ValidationChannel>(graph: &Graph<V>, name: &str) -> ValueId {
    graph
        .values
        .iter()
        .position(|value| value.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing graph value {name}"))
}

/// One tiny block - four experts, top two, nonuniform logits, bias, payloads and weights - must match
/// the independent reference, on both phases.
///
/// Red under: gathering the bias-adjusted scores as the weights, adding a shared-expert branch, or
/// applying a routed scaling factor.
#[test]
fn minimax_m2_full_block_matches_independent_oracle() {
    let plan = plan(1, 4);
    let owners = owners(&plan);
    let mut reference = MiniMaxReference::new(plan.config().clone());
    let expected = reference.step(PROMPT[0], 0);

    let graph = trace_minimax_m2_text_prefill(&plan, 1, CAPACITY).expect("tiny one-token graph");
    let inputs = bindings(&graph, &owners, &PROMPT[..1], None, None);
    let logits = eval_graph(&graph, &inputs).expect("tiny one-token graph evaluates");
    // one-block logits: graph (actual) vs the reference (expected)
    poot_test_util::assert_close(dense_rows(&logits).as_f32().unwrap(), &expected, 2e-4);
}

/// The whole tower is present: an embedding, the configured number of blocks with two residual adds
/// each, a final norm, a head, and one non-identity K/V state pair per layer.
///
/// Red under: removing a layer, a residual, the embedding, the final norm, the head, or a state pair.
#[test]
fn minimax_m2_full_text_graph_is_complete() {
    let layers = 3;
    let plan = plan(layers, 4);
    for (tokens, graph) in [
        (
            1,
            trace_minimax_m2_text_decode(&plan, CAPACITY).expect("decode graph"),
        ),
        (
            PROMPT.len(),
            trace_minimax_m2_text_prefill(&plan, PROMPT.len(), CAPACITY).expect("prefill graph"),
        ),
    ] {
        graph.validate().expect("tiny graph validates");
        let table = plan.table();
        for source in [table.embedding(), table.final_norm(), table.lm_head()] {
            named(&graph, source.name());
        }
        assert_eq!(graph.state.len(), 2 * layers, "one K/V pair per layer");
        for (index, &(state_in, state_out)) in graph.state.iter().enumerate() {
            assert_ne!(state_in, state_out, "state pair {index} is an identity");
        }
        for layer in 0..layers {
            named(&graph, &cache_name(layer, "k_cache"));
            named(&graph, &cache_name(layer, "v_cache"));
            let sources = &table.layers()[layer];
            named(&graph, sources.input_norm().name());
            named(&graph, sources.post_attention_norm().name());
            named(&graph, sources.router_gate().name());
        }
        // The head reads the final norm, which reads the last block: the whole chain is live.
        assert_eq!(
            graph.aval(graph.output).shape.len(),
            2,
            "logits are [tokens, vocab]"
        );
        assert_eq!(
            graph.aval(graph.output).shape[1],
            plan.config().vocab_size,
            "logits span the vocabulary"
        );
        // Two residual adds per block: an `Add` of two whole `[T, hidden]` activations. Other `Add`s have a
        // literal or broadcast operand or a different shape (including the `[T * top_k, hidden]` expert rows).
        let hidden = plan.config().hidden_size;
        let activation = vec![tokens, hidden];
        let residuals = graph
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::Binary(BinOp::Add)))
            .filter(|eqn| {
                graph.aval(eqn.out).shape == activation
                    && eqn.inputs.len() == 2
                    && eqn.inputs.iter().all(|operand| match operand {
                        poot_graph_ir::Operand::Value(id) => graph.aval(*id).shape == activation,
                        poot_graph_ir::Operand::Lit(_) => false,
                    })
            })
            .count();
        assert_eq!(residuals, 2 * layers, "two residual adds per block");
    }
}

/// Whole-prompt prefill reproduces the independent reference at every position.
///
/// Red under: moving the Q/K norm after the head reshape, rotating all `head_dim` values instead of the leading
/// `rotary_dim`, or caching K before RoPE (prefill reads its own cached keys back, so pre-RoPE K diverges from
/// position one).
///
/// Not covered: the decode graph. No CPU evaluator can run one: `eval_value_with_state` has no arm for the
/// token guard's I32 arithmetic, and `eval_exact_i32` rejects decode's `DynamicUpdateSlice` and RoPE-table
/// `Gather` on the runtime `Slot::Pos` (card 382). Binding decode's caches from the reference would make the
/// pre-RoPE-K mutation green, so decode is held to the structural rows instead.
#[test]
fn minimax_m2_prefill_matches_the_reference_at_every_position() {
    let plan = plan(2, 4);
    let owners = owners(&plan);

    let mut reference = MiniMaxReference::new(plan.config().clone());
    let expected = PROMPT
        .iter()
        .enumerate()
        .map(|(position, &token)| reference.step(token, position))
        .collect::<Vec<_>>();

    let prefill = trace_minimax_m2_text_prefill(&plan, PROMPT.len(), CAPACITY).expect("prefill");
    let inputs = bindings(&prefill, &owners, &PROMPT, None, None);
    let logits = eval_graph(&prefill, &inputs).expect("prefill evaluates");
    let vocab = plan.config().vocab_size;
    let rows = dense_rows(&logits);
    for (position, expected) in expected.iter().enumerate() {
        // prefill logits at `position`: graph (actual) vs the reference (expected)
        poot_test_util::assert_close(
            &rows.as_f32().unwrap()[position * vocab..(position + 1) * vocab],
            expected,
            2e-4,
        );
    }
}

/// Changing one byte of one selected expert's payload changes the block output.
///
/// Red under: an expert path that does not read its packed source - a dropped `w3`, a routed
/// weight applied to the wrong slot, or an accumulation that never reaches the residual.
#[test]
fn minimax_m2_expert_payload_is_load_bearing() {
    let plan = plan(1, 4);
    let owners = owners(&plan);
    let graph = trace_minimax_m2_text_prefill(&plan, 1, CAPACITY).expect("one-token graph");
    let inputs = bindings(&graph, &owners, &PROMPT[..1], None, None);
    let baseline = eval_graph(&graph, &inputs).expect("baseline evaluates");
    let baseline = dense_rows(&baseline).as_f32().unwrap().to_vec();

    // Which expert ran? Perturb each in turn; at least `top_k` of them must move the output,
    // and an unselected one must not.
    let experts = plan.config().num_local_experts;
    let moved = (0..experts)
        .filter(|expert| {
            let linear_id = format!("model.layers.0.block_sparse_moe.experts.{expert}.w2");
            let descriptor = owners.packed[&linear_id].weight();
            let (mut codes, scale) = packed_codes(&linear_id, descriptor.shape());
            codes[0] = if codes[0] == 0x30 { 0x34 } else { 0x30 };
            let scales = (0..packed_scale_elements(descriptor))
                .flat_map(|_| scale.to_le_bytes())
                .collect::<Vec<_>>();
            let mutated = Arc::new(
                PackedPayload::try_new(
                    descriptor,
                    [
                        (SourceRole::Planar(OperandRole::Codes), codes.into()),
                        (SourceRole::Planar(OperandRole::Scale), scales.into()),
                    ],
                )
                .expect("mutated payload"),
            );
            let mut inputs = inputs.clone();
            let weight = named(&graph, PackedSourceName::weight(&linear_id).as_str());
            let scale_id = named(&graph, PackedSourceName::scale(&linear_id).as_str());
            inputs.insert(
                weight,
                Value::Packed(PackedComponentRef::new(
                    Arc::clone(&mutated),
                    SourceRole::Planar(OperandRole::Codes),
                )),
            );
            inputs.insert(
                scale_id,
                Value::Packed(PackedComponentRef::new(
                    mutated,
                    SourceRole::Planar(OperandRole::Scale),
                )),
            );
            let logits = eval_graph(&graph, &inputs).expect("mutated evaluates");
            poot_test_util::max_abs_error(dense_rows(&logits).as_f32().unwrap(), &baseline) > 1e-6
        })
        .count();
    assert_eq!(
        moved,
        plan.config().num_experts_per_tok,
        "exactly the selected experts' payloads may change the output"
    );
}

/// Decode takes card 369's indexed composition and prefill its grouped one, and neither phase contains a
/// model-local packed expert builder.
///
/// Red under: calling the indexed helper from prefill, or hand-building either composition here. The
/// grouped form is the only one that emits the stable-sort `Scatter` pair.
#[test]
fn minimax_m2_decode_is_indexed_and_prefill_is_grouped() {
    let plan = plan(1, 4);
    let decode = trace_minimax_m2_text_decode(&plan, CAPACITY).expect("decode graph");
    let prefill = trace_minimax_m2_text_prefill(&plan, PROMPT.len(), CAPACITY).expect("prefill");

    let scatters = |graph: &Graph<ValidationOutputs>| {
        graph
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::Scatter { .. }))
            .count()
    };
    assert_eq!(
        scatters(&decode),
        0,
        "the indexed form emits no sort scatter"
    );
    assert!(
        scatters(&prefill) >= 2,
        "the grouped form scatters both the activations and the ids"
    );

    // Both phases carry one `IndexedMatMul` carrier per expert projection, from card 369's helpers.
    for (label, graph) in [("decode", &decode), ("prefill", &prefill)] {
        let carriers = graph
            .eqns
            .iter()
            .filter(|eqn| matches!(eqn.op, OpKind::IndexedMatMul))
            .count();
        assert_eq!(
            carriers,
            MiniMaxM2ExpertProjection::ALL.len(),
            "{label} has one indexed carrier per expert projection"
        );
    }
}

/// The runtime token index is bounded in graph, and its witness is a declared validation output.
///
/// Red under: gathering the raw token id, or dropping the guard's witness declaration.
#[test]
fn minimax_m2_token_index_is_guarded() {
    let plan = plan(1, 4);
    let graph = trace_minimax_m2_text_decode(&plan, CAPACITY).expect("decode graph");
    let witness = graph
        .validation_outputs()
        .iter()
        .find(|output| output.name == MINIMAX_M2_TOKEN_INDEX_VALIDATION)
        .expect("the token index guard declares a witness");

    let token = graph
        .inputs
        .iter()
        .copied()
        .find(|&id| graph.meta(id).storage == Storage::Slot(Slot::Token))
        .expect("a token slot");
    let embedding = named(&graph, plan.table().embedding().name());
    let producers = producers(&graph);
    let gather = graph
        .eqns
        .iter()
        .find(|eqn| {
            matches!(eqn.op, OpKind::Gather { axis: 0 })
                && matches!(
                    eqn.inputs.first(),
                    Some(poot_graph_ir::Operand::Value(source)) if *source == embedding
                )
        })
        .expect("the embedding is gathered");
    let index = match gather.inputs.get(1) {
        Some(poot_graph_ir::Operand::Value(index)) => *index,
        other => panic!("embedding gather index {other:?}"),
    };
    assert_ne!(index, token, "the embedding must not gather the raw token");
    assert!(
        matches!(producers.get(&index), Some(OpKind::Select)),
        "the embedding gathers the guarded index"
    );
    poot_graph_ir::recognize_index_bounds_guard(&graph, index, witness.value)
        .expect("the guard and its witness are the canonical pair");
}

/// A nonfinite router logit fails deterministically through card 375a's CPU path, publishing neither a
/// result nor state, and maps back to the typed error naming its layer.
///
/// Red under: publishing a result or state on failure, or varying the typed failure for one input.
#[test]
fn minimax_m2_router_cpu_errors_are_transactional() {
    let plan = plan(2, 4);
    let owners = owners(&plan);
    let graph = trace_minimax_m2_text_prefill(&plan, 1, CAPACITY).expect("one-token graph");
    let experts = plan.config().num_local_experts;
    let hidden = plan.config().hidden_size;

    for layer in 0..plan.config().num_hidden_layers {
        let gate = named(&graph, plan.table().layers()[layer].router_gate().name());
        let mut inputs = bindings(&graph, &owners, &PROMPT[..1], None, None);
        inputs.insert(
            gate,
            Value::Host(HostTensor::f32(
                vec![experts, hidden],
                vec![f32::NAN; experts * hidden],
            )),
        );

        let first = eval_graph(&graph, &inputs)
            .expect_err("a nonfinite router logit must fail before publication");
        let poot_eval::EvalError::Validation(failure) = &first else {
            panic!("layer {layer} must fail through the validation packet, got {first:?}");
        };
        let error = minimax_m2_router_validation_error(failure)
            .expect("the failure maps back to a typed router error");
        match error {
            MiniMaxM2TraceError::Router {
                layer: named_layer,
                condition,
            } => {
                assert_eq!(named_layer, layer, "the error names the failing layer");
                assert_eq!(condition, "router_logits_finite");
            }
            other => panic!("layer {layer} produced {other}"),
        }
        // The same input fails the same way, and an `Err` publishes neither result nor state.
        let second = eval_graph(&graph, &inputs).expect_err("still fails");
        assert_eq!(
            format!("{first:?}"),
            format!("{second:?}"),
            "layer {layer} varies its typed failure"
        );
    }
}

/// Every layer declares all three witness families plus the graph-wide token guard, and each id maps back to
/// exactly its own layer and condition.
///
/// Two of the three fire in real executions: nonfinite logits, and
/// `minimax_m2_underflowing_router_fails_instead_of_publishing_nan`. The selector range check cannot be provoked
/// through a finite router input (`ArgTopK` indexes the rank it inverts), so FR-016's third condition is proven
/// at the declaration and mapping level.
///
/// Red under: dropping a witness declaration, reusing one id across layers, or mapping a foreign id.
#[test]
fn minimax_m2_router_witnesses_are_declared_and_map_back() {
    let layers = 3;
    let plan = plan(layers, 4);
    let graph = trace_minimax_m2_text_decode(&plan, CAPACITY).expect("decode graph");
    let declared = graph.validation_outputs();
    assert_eq!(
        declared.len(),
        1 + layers * MiniMaxM2Validation::ALL.len(),
        "one token guard plus three families per layer"
    );
    let unique = declared
        .iter()
        .map(|output| output.id)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique.len(), declared.len(), "validation ids collide");

    for layer in 0..layers {
        for family in MiniMaxM2Validation::ALL {
            let name = family.name(layer);
            let output = declared
                .iter()
                .find(|output| output.name == name)
                .unwrap_or_else(|| panic!("layer {layer} does not declare {name}"));
            let failure = poot_graph_ir::ExecutionValidationFailure {
                id: output.id,
                name: name.clone(),
                lane: 0,
                observed_bits: 1,
            };
            match minimax_m2_router_validation_error(&failure) {
                Some(MiniMaxM2TraceError::Router {
                    layer: named_layer,
                    condition,
                }) => {
                    assert_eq!(named_layer, layer);
                    assert_eq!(condition, family.label());
                }
                other => panic!("{name} mapped to {other:?}"),
            }
        }
    }

    // A name that does not match the id it arrives with is not guessed at.
    let mismatched = poot_graph_ir::ExecutionValidationFailure {
        id: MiniMaxM2Validation::RouterLogitsFinite
            .id(0)
            .expect("layer 0 id"),
        name: "minimax_m2.layer2.selected_sum_usable".to_string(),
        lane: 0,
        observed_bits: 1,
    };
    assert!(minimax_m2_router_validation_error(&mismatched).is_none());

    // The token-guard sentinel is not in the per-layer id space: `u32::MAX % 3 == 0`, so arithmetic alone would
    // read it as layer 1431655765's logit family.
    let sentinel = declared
        .iter()
        .find(|output| output.name == MINIMAX_M2_TOKEN_INDEX_VALIDATION)
        .expect("the token guard declares a witness")
        .id;
    assert!(
        MiniMaxM2Validation::decode(sentinel).is_none(),
        "the token-guard id must not decode as a per-layer family"
    );
    assert!(matches!(
        minimax_m2_router_validation_error(&poot_graph_ir::ExecutionValidationFailure {
            id: sentinel,
            name: MINIMAX_M2_TOKEN_INDEX_VALIDATION.to_string(),
            lane: 0,
            observed_bits: 1,
        }),
        Some(MiniMaxM2TraceError::TokenIndexOutOfRange)
    ));
}

/// A finite router logit far below zero saturates its sigmoid score to exactly zero (`exp(-x) = +inf` near
/// `-1e4`). When every selected expert's does, the selected sum is zero and normalization would publish
/// `0 / 0 = NaN`; the sum witness must fail the step instead. Unlike card 364a's `softplus` overflow (an
/// unusably large sum), MiniMax's bounded sigmoid gives an unusably small one.
///
/// Red under: dropping the sum witness or its positivity half (either returns `Ok` with NaN logits).
#[test]
fn minimax_m2_zero_selected_scores_fail_instead_of_publishing_nan() {
    let plan = plan(1, 4);
    let owners = owners(&plan);
    let graph = trace_minimax_m2_text_prefill(&plan, 1, CAPACITY).expect("one-token graph");

    // `logits[e] = -scale * ||x||^2` for every expert: all negative, and `scale` sets how far past overflow.
    let mut reference = MiniMaxReference::new(plan.config().clone());
    reference.step(PROMPT[0], 0);
    let x = reference.router_inputs[0].clone();
    let norm_squared = x.iter().map(|v| v * v).sum::<f32>();
    assert!(norm_squared > 0.0, "the fixture router input is nonzero");
    let scale = 1.0e4 / norm_squared;
    let experts = plan.config().num_local_experts;
    let row = (0..experts)
        .flat_map(|_| x.iter().map(|v| -scale * v))
        .collect::<Vec<_>>();

    let gate = named(&graph, plan.table().layers()[0].router_gate().name());
    let mut inputs = bindings(&graph, &owners, &PROMPT[..1], None, None);
    inputs.insert(
        gate,
        Value::Host(HostTensor::f32(
            vec![experts, plan.config().hidden_size],
            row,
        )),
    );

    let error = eval_graph(&graph, &inputs)
        .expect_err("an all-underflowing router must fail, not publish NaN");
    let poot_eval::EvalError::Validation(failure) = &error else {
        panic!("must fail through the validation packet, got {error:?}");
    };
    match minimax_m2_router_validation_error(failure) {
        Some(MiniMaxM2TraceError::Router {
            layer: 0,
            condition,
        }) => {
            assert_eq!(condition, "selected_sum_usable");
        }
        other => panic!("underflowing router mapped to {other:?}"),
    }
}

/// The two route-slot selections agree, and both match an independent expectation.
///
/// Decode's `Gather` form is exact only because a single leading row shares one index set; prefill's one-hot form
/// is the general case. This builds the selection alone, so it reaches decode-only composition a CPU oracle
/// otherwise cannot.
///
/// Red under: dropping the `eq` product to a single `Ge`, reading `rank` where the slot ordinal belongs, or
/// letting either form return the biased scores.
#[test]
fn minimax_m2_route_slot_weights_agree_across_phases() {
    const EXPERTS: usize = 5;
    const TOP_K: usize = 3;
    // Descending rank of a made-up score row, lower id winning ties: expert 3 is best, then 0, 4, 1, 2.
    let rank = [1.0f32, 3.0, 4.0, 0.0, 2.0];
    let weights = [0.25f32, 0.05, 0.02, 0.5, 0.18];
    // Slot r holds the expert whose rank is r.
    let ids = [3.0f32, 0.0, 4.0];
    let expected = [weights[3], weights[0], weights[4]];

    for batched in [false, true] {
        let b = Builder::new();
        let weights_value = b.constant("weights", TensorType::f32(vec![1, EXPERTS]));
        let rank_value = b.constant("rank", TensorType::f32(vec![1, EXPERTS]));
        let ids_value = b.constant("ids", TensorType::f32(vec![1, TOP_K]));
        let slots = if batched {
            MiniMaxM2RouteSlots::Batched {
                expert_iota: b.constant("expert_iota", TensorType::f32(vec![EXPERTS + 1])),
            }
        } else {
            MiniMaxM2RouteSlots::Single
        };
        let out = route_slot_weights(&b, slots, weights_value, rank_value, ids_value);
        let graph = b.finish(out);

        let inputs = graph
            .inputs
            .iter()
            .map(|&id| {
                let meta = graph.meta(id);
                let shape = meta.aval.shape.clone();
                let data = match meta.name.as_deref().expect("named input") {
                    "weights" => weights.to_vec(),
                    "rank" => rank.to_vec(),
                    "ids" => ids.to_vec(),
                    "expert_iota" => (0..=EXPERTS).map(|index| index as f32).collect(),
                    other => panic!("unexpected input {other}"),
                };
                (id, Value::from(HostTensor::f32(shape, data)))
            })
            .collect::<HashMap<_, _>>();
        let got = poot_eval::eval(
            &graph,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("slot weights evaluate")
        .output
        .into_host()
        .expect("slot weights output is dense");
        // the batched and the single-slot graph (actual) vs the reference (expected)
        poot_test_util::assert_close(got.as_f32().unwrap(), &expected, 2e-4);
    }
}

/// The pinned 62-layer graph's validation packet fits, and every witness is one lane.
///
/// Other tests build one to three layers, far below `MAX_VALIDATION_PACKET_BYTES`; a `sum_to_scalar` that kept
/// an axis would stay green there while the real graph fails to build.
///
/// Red under: a witness that keeps an axis, or a fourth per-layer family (either changes the lane count).
#[test]
fn minimax_m2_pinned_layer_count_fits_the_validation_packet() {
    const PINNED_LAYERS: usize = 62;
    // Two experts rather than the pinned 256: the packet depends on layer count, not expert count.
    let plan = plan(PINNED_LAYERS, 2);
    let graph = trace_minimax_m2_text_decode(&plan, CAPACITY).expect("pinned-depth decode graph");

    let declared = graph.validation_outputs();
    let expected_outputs = 1 + PINNED_LAYERS * MiniMaxM2Validation::ALL.len();
    assert_eq!(
        declared.len(),
        expected_outputs,
        "one guard plus three per layer"
    );

    let layout = poot_graph_ir::ValidationPacketLayout::for_graph(&graph)
        .expect("the pinned-depth packet is within its cap");
    assert_eq!(
        layout.lane_count, expected_outputs,
        "every witness must reduce to exactly one lane"
    );
    assert_eq!(layout.byte_len, expected_outputs * 4);
    assert!(
        layout.byte_len <= poot_graph_ir::MAX_VALIDATION_PACKET_BYTES,
        "{} bytes exceeds the {}-byte packet cap",
        layout.byte_len,
        poot_graph_ir::MAX_VALIDATION_PACKET_BYTES
    );
}

/// The checkpoint namespace and the graph namespace agree, proven where both are visible.
///
/// `poot-load` owns the checkpoint spelling (`...weight` / `...weight_scale_inv`) and card 379 the graph spelling
/// (`...packed_weight_source` / `...packed_scale_source`); `poot-load` depends on `poot-quant` alone, so the
/// bridge is the linear id plus the shared `SourceRole` of a card 365b manifest row. This checks it
/// against the tracer's own staged constants.
///
/// Red under: changing either suffix, or deriving a manifest row's graph name from anything but card 379's
/// constructor (including one that ignores `role`; staged-set membership alone cannot catch that, so the
/// uniqueness check below does).
#[test]
fn minimax_m2_checkpoint_and_graph_namespaces_agree() {
    let config = small_config(2, 3);
    let manifest = MiniMaxM2Manifest::new(&config).expect("manifest");
    let graph = trace_minimax_m2_text_decode(&plan(2, 3), CAPACITY).expect("decode graph");
    let staged = graph
        .values
        .iter()
        .filter_map(|value| PackedSourceName::parse(value.name.as_deref()?))
        .map(String::from)
        .collect::<std::collections::BTreeSet<_>>();
    assert!(!staged.is_empty(), "the decode graph stages packed sources");

    let mut bridged = 0usize;
    let mut bridged_names = std::collections::BTreeSet::new();
    for (checkpoint_name, entry) in manifest.entries() {
        let Some((linear_id, component)) = entry.packed_component() else {
            continue;
        };
        // The checkpoint half is pinned against literals in `poot-load`; here the graph half must land on a staged constant.
        let graph_name = PackedSourceName::new(linear_id, component.source_role());
        assert!(
            staged.contains(graph_name.as_str()),
            "{graph_name} is not staged by the graph that reads {checkpoint_name}"
        );
        // Every row's graph name must be distinct: a weight row and its scale row share a linear id but not a name.
        assert!(
            bridged_names.insert(String::from(graph_name)),
            "{checkpoint_name} collapsed onto a graph name another packed row already claimed"
        );
        bridged += 1;
    }
    assert_eq!(
        bridged,
        2 * manifest
            .report()
            .packed_pairs()
            .expect("packed pairs fit in usize"),
        "every packed row of both classes crosses the bridge"
    );
}

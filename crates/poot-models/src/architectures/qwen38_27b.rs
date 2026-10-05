//! Source-agnostic Qwen3.8-27B (`qwen3_5`) text graphs.
//!
//! Separate from [`crate::qwen38`], which implements the `qwen4_exp` Qwen3.8-Flash-Next
//! architecture. The graph boundary consumes token ids, caller-bound checkpoint sources, and explicit
//! recurrent state. It does not load a checkpoint or register a runner.
//!
//! All 400 quantized text projections are fallible packed-linear graph appends. Dense checkpoint
//! tensors remain in PyTorch source order and source dtype; casts, transposes, zero-centered
//! RMSNorm weights, and `A_log -> -exp(A_log)` are ordinary graph operations. The semantics follow HF
//! `modeling_qwen3_5.py`: only `Qwen3_5RMSNorm` weights scale by `1 + weight`, the GDN gated norm
//! scales by its stored weight, and GDN q/k heads expand in the grouped order of the HF V heads.
//!
//! The CPU-oracle tests bind BF16 checkpoint owners to the production graphs and compare them with an
//! independent plain-loop reference of the HF model.

use poot_graph_ir::op::{PackedWeight, ScaleEncoding, WeightFormat};
use poot_graph_ir::ops::{linear, packed_linear, rmsnorm, sigmoid, softplus, swiglu};
use poot_graph_ir::{
    BinOp, Builder, BuilderAppendError, Graph, Scalar, Slot, StateRole, TensorType, Traced, UnOp,
};
use poot_tensor::DType;

use crate::qwen3next::{
    GdnHeadOrder, qwen3next_gated_attention_projected, qwen3next_gdn_projected,
    qwen3next_split_query_gate,
};
#[cfg(test)]
use crate::qwen3next::{
    qwen3next_gated_attention_prefill_projected, qwen3next_gdn_prefill_projected,
};

const PACKED_FORMAT: WeightFormat = WeightFormat::E4m3Block128 {
    scale: ScaleEncoding::Bf16,
};
/// Namespace prefix of every text-tower tensor. `lm_head.weight` sits outside it in the checkpoint.
const TEXT_ROOT: &str = "model.language_model";
/// Storage dtype of every dense tensor in the pinned `Qwen/Qwen3.8-27B-FP8` artifact.
const CHECKPOINT_DENSE_DTYPE: DType = DType::BF16;
/// V-head order of the HF safetensors: `Qwen3_5GatedDeltaNet.forward` expands q/k with
/// `repeat_interleave`, and this graph binds the GDN tensors without reordering V heads.
const GDN_HEAD_ORDER: GdnHeadOrder = GdnHeadOrder::Grouped;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen35LayerKind {
    LinearAttention,
    FullAttention,
}

/// Text-tower fields projected from the pinned `Qwen/Qwen3.8-27B-FP8` config.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35TextConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layer_types: Vec<Qwen35LayerKind>,
    pub eps: f32,
    pub max_pos: usize,
    pub rotary_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub gdn_num_k_heads: usize,
    pub gdn_num_v_heads: usize,
    pub gdn_key_head_dim: usize,
    pub gdn_value_head_dim: usize,
    pub conv_k: usize,
    pub fp8_block_out: usize,
    pub fp8_block_in: usize,
}

impl Qwen35TextConfig {
    pub fn pinned() -> Self {
        const LAYERS: usize = 64;
        const FULL_ATTENTION_INTERVAL: usize = 4;
        Self {
            vocab: 248_320,
            hidden: 5_120,
            intermediate: 17_408,
            layer_types: (0..LAYERS)
                .map(|layer| {
                    if (layer + 1).is_multiple_of(FULL_ATTENTION_INTERVAL) {
                        Qwen35LayerKind::FullAttention
                    } else {
                        Qwen35LayerKind::LinearAttention
                    }
                })
                .collect(),
            eps: 1e-6,
            max_pos: 262_144,
            rotary_dim: 64,
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            gdn_num_k_heads: 16,
            gdn_num_v_heads: 48,
            gdn_key_head_dim: 128,
            gdn_value_head_dim: 128,
            conv_k: 4,
            fp8_block_out: 128,
            fp8_block_in: 128,
        }
    }

    /// Reject any drift from [`Qwen35TextConfig::pinned`] before graph construction begins.
    pub fn validate_exact(&self) -> Result<(), Qwen35GraphError> {
        let pinned = Self::pinned();
        let exact_fields = [
            ("vocab", self.vocab, pinned.vocab),
            ("hidden", self.hidden, pinned.hidden),
            ("intermediate", self.intermediate, pinned.intermediate),
            ("max_pos", self.max_pos, pinned.max_pos),
            ("rotary_dim", self.rotary_dim, pinned.rotary_dim),
            ("n_heads", self.n_heads, pinned.n_heads),
            ("n_kv_heads", self.n_kv_heads, pinned.n_kv_heads),
            ("head_dim", self.head_dim, pinned.head_dim),
            (
                "gdn_num_k_heads",
                self.gdn_num_k_heads,
                pinned.gdn_num_k_heads,
            ),
            (
                "gdn_num_v_heads",
                self.gdn_num_v_heads,
                pinned.gdn_num_v_heads,
            ),
            (
                "gdn_key_head_dim",
                self.gdn_key_head_dim,
                pinned.gdn_key_head_dim,
            ),
            (
                "gdn_value_head_dim",
                self.gdn_value_head_dim,
                pinned.gdn_value_head_dim,
            ),
            ("conv_k", self.conv_k, pinned.conv_k),
            ("fp8_block_out", self.fp8_block_out, pinned.fp8_block_out),
            ("fp8_block_in", self.fp8_block_in, pinned.fp8_block_in),
        ];
        for (field, actual, expected) in exact_fields {
            if actual != expected {
                return Err(Qwen35GraphError::ConfigField {
                    field,
                    actual: actual.to_string(),
                    expected: expected.to_string(),
                });
            }
        }
        if self.eps != pinned.eps {
            return Err(Qwen35GraphError::ConfigField {
                field: "eps",
                actual: self.eps.to_string(),
                expected: pinned.eps.to_string(),
            });
        }
        if self.layer_types.len() != pinned.layer_types.len() {
            return Err(Qwen35GraphError::LayerCount {
                actual: self.layer_types.len(),
                expected: pinned.layer_types.len(),
            });
        }
        for (layer, (&actual, &expected)) in
            self.layer_types.iter().zip(&pinned.layer_types).enumerate()
        {
            if actual != expected {
                return Err(Qwen35GraphError::LayerKind {
                    layer,
                    actual,
                    expected,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Qwen35Layout {
    vocab: usize,
    hidden: usize,
    intermediate: usize,
    layer_types: Vec<Qwen35LayerKind>,
    eps: f32,
    max_pos: usize,
    rotary_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    gdn_num_k_heads: usize,
    gdn_num_v_heads: usize,
    /// Key and value head dim of the GDN layers. The shared GDN core uses one head dim for both.
    gdn_head_dim: usize,
    conv_k: usize,
}

impl Qwen35Layout {
    /// Project a config that passes [`Qwen35TextConfig::validate_exact`]. The exact pins give
    /// `gdn_key_head_dim` and `gdn_value_head_dim` the same value, which is what lets the layout carry
    /// the single GDN head dim.
    fn from_config(config: &Qwen35TextConfig) -> Result<Self, Qwen35GraphError> {
        config.validate_exact()?;
        Ok(Self {
            vocab: config.vocab,
            hidden: config.hidden,
            intermediate: config.intermediate,
            layer_types: config.layer_types.clone(),
            eps: config.eps,
            max_pos: config.max_pos,
            rotary_dim: config.rotary_dim,
            n_heads: config.n_heads,
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            gdn_num_k_heads: config.gdn_num_k_heads,
            gdn_num_v_heads: config.gdn_num_v_heads,
            gdn_head_dim: config.gdn_value_head_dim,
            conv_k: config.conv_k,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen35PackedRole {
    GdnQkv,
    GdnZ,
    GdnOut,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionOut,
    MlpGate,
    MlpUp,
    MlpDown,
}

const GDN_ROLES: [Qwen35PackedRole; 3] = [
    Qwen35PackedRole::GdnQkv,
    Qwen35PackedRole::GdnZ,
    Qwen35PackedRole::GdnOut,
];
const ATTENTION_ROLES: [Qwen35PackedRole; 4] = [
    Qwen35PackedRole::AttentionQ,
    Qwen35PackedRole::AttentionK,
    Qwen35PackedRole::AttentionV,
    Qwen35PackedRole::AttentionOut,
];
const MLP_ROLES: [Qwen35PackedRole; 3] = [
    Qwen35PackedRole::MlpGate,
    Qwen35PackedRole::MlpUp,
    Qwen35PackedRole::MlpDown,
];

impl Qwen35PackedRole {
    fn suffix(self) -> &'static str {
        match self {
            Self::GdnQkv => "linear_attn.in_proj_qkv",
            Self::GdnZ => "linear_attn.in_proj_z",
            Self::GdnOut => "linear_attn.out_proj",
            Self::AttentionQ => "self_attn.q_proj",
            Self::AttentionK => "self_attn.k_proj",
            Self::AttentionV => "self_attn.v_proj",
            Self::AttentionOut => "self_attn.o_proj",
            Self::MlpGate => "mlp.gate_proj",
            Self::MlpUp => "mlp.up_proj",
            Self::MlpDown => "mlp.down_proj",
        }
    }

    fn logical_shape(self, layout: &Qwen35Layout) -> [usize; 2] {
        let key_dim = layout.gdn_num_k_heads * layout.gdn_head_dim;
        let value_dim = layout.gdn_num_v_heads * layout.gdn_head_dim;
        let query_dim = layout.n_heads * layout.head_dim;
        let kv_dim = layout.n_kv_heads * layout.head_dim;
        match self {
            Self::GdnQkv => [2 * key_dim + value_dim, layout.hidden],
            Self::GdnZ => [value_dim, layout.hidden],
            Self::GdnOut => [layout.hidden, value_dim],
            Self::AttentionQ => [2 * query_dim, layout.hidden],
            Self::AttentionK | Self::AttentionV => [kv_dim, layout.hidden],
            Self::AttentionOut => [layout.hidden, query_dim],
            Self::MlpGate | Self::MlpUp => [layout.intermediate, layout.hidden],
            Self::MlpDown => [layout.hidden, layout.intermediate],
        }
    }
}

/// One deterministic source-agnostic row for a canonical packed linear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35PackedLinearSpec {
    pub layer: usize,
    pub role: Qwen35PackedRole,
    pub linear_id: String,
    pub descriptor: PackedWeight,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35GraphError {
    #[error("Qwen3.8-27B config field {field} is {actual}, expected {expected}")]
    ConfigField {
        field: &'static str,
        actual: String,
        expected: String,
    },
    #[error("Qwen3.8-27B layer table has {actual} rows, expected {expected}")]
    LayerCount { actual: usize, expected: usize },
    #[error("Qwen3.8-27B layer {layer} is {actual:?}, expected pinned schedule row {expected:?}")]
    LayerKind {
        layer: usize,
        actual: Qwen35LayerKind,
        expected: Qwen35LayerKind,
    },
    #[error("Qwen3.8-27B packed descriptor for {linear_id:?} is invalid: {message}")]
    PackedDescriptor { linear_id: String, message: String },
    #[error("Qwen3.8-27B packed graph ended before row {index}: {message}")]
    PackedCursor { index: usize, message: String },
    #[error(transparent)]
    PackedAppend(#[from] BuilderAppendError),
    #[error("Qwen3.8-27B prefill length must be nonzero")]
    EmptyPrefill,
    #[error("Qwen3.8-27B cache capacity {capacity} is smaller than prefill length {sequence}")]
    PrefillCapacity { sequence: usize, capacity: usize },
    #[error("Qwen3.8-27B GDN prefill chunk must be nonzero")]
    EmptyChunk,
    #[error("Qwen3.8-27B cache capacity {capacity} must exceed decode position {position}")]
    DecodeCapacity { position: usize, capacity: usize },
}

fn linear_id(layer: usize, role: Qwen35PackedRole) -> String {
    format!("{TEXT_ROOT}.layers.{layer}.{}", role.suffix())
}

fn packed_linear_specs_for_layout(
    layout: &Qwen35Layout,
) -> Result<Vec<Qwen35PackedLinearSpec>, Qwen35GraphError> {
    let mut rows = Vec::new();
    for (layer, &kind) in layout.layer_types.iter().enumerate() {
        let mixer_roles = match kind {
            Qwen35LayerKind::LinearAttention => GDN_ROLES.as_slice(),
            Qwen35LayerKind::FullAttention => ATTENTION_ROLES.as_slice(),
        };
        for &role in mixer_roles.iter().chain(MLP_ROLES.iter()) {
            let id = linear_id(layer, role);
            let descriptor = PackedWeight::try_new(PACKED_FORMAT, role.logical_shape(layout))
                .map_err(|error| Qwen35GraphError::PackedDescriptor {
                    linear_id: id.clone(),
                    message: error.to_string(),
                })?;
            rows.push(Qwen35PackedLinearSpec {
                layer,
                role,
                linear_id: id,
                descriptor,
            });
        }
    }
    Ok(rows)
}

/// Enumerate the 400 main-text packed linears in forward-use order. The config must pass
/// [`Qwen35TextConfig::validate_exact`], and its pinned schedule of 48 GDN and 16 attention
/// layers fixes the row count.
#[cfg(test)]
pub(crate) fn qwen38_packed_linear_specs(
    config: &Qwen35TextConfig,
) -> Result<Vec<Qwen35PackedLinearSpec>, Qwen35GraphError> {
    packed_linear_specs_for_layout(&Qwen35Layout::from_config(config)?)
}

/// One dense text tensor. The graph binds dense sources by checkpoint name, so the row needs no
/// role or layer label of its own.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Qwen35DenseSourceSpec {
    /// Checkpoint tensor name, which is also the graph constant name.
    pub name: String,
    pub shape: Vec<usize>,
}

/// Enumerate the dense text tensors in checkpoint source order, with shapes derived from `config`.
///
/// These are the BF16 constants the traced graph binds by name; the loader builds its dense manifest
/// from this table. Unlike [`qwen38_packed_linear_specs`] there is no pinned-schedule requirement, so a
/// reduced config yields a reduced table.
#[cfg(test)]
pub(crate) fn qwen38_dense_source_specs(config: &Qwen35TextConfig) -> Vec<Qwen35DenseSourceSpec> {
    let hidden = config.hidden;
    let value_heads = config.gdn_num_v_heads;
    let conv_channels = 2 * config.gdn_num_k_heads * config.gdn_key_head_dim
        + value_heads * config.gdn_value_head_dim;
    let mut rows = Vec::new();
    let mut push = |name: String, shape: Vec<usize>| {
        rows.push(Qwen35DenseSourceSpec { name, shape });
    };
    push(
        format!("{TEXT_ROOT}.embed_tokens.weight"),
        vec![config.vocab, hidden],
    );
    for (layer, &kind) in config.layer_types.iter().enumerate() {
        let root = format!("{TEXT_ROOT}.layers.{layer}");
        push(format!("{root}.input_layernorm.weight"), vec![hidden]);
        let mixer = match kind {
            Qwen35LayerKind::LinearAttention => vec![
                ("linear_attn.in_proj_a.weight", vec![value_heads, hidden]),
                ("linear_attn.in_proj_b.weight", vec![value_heads, hidden]),
                (
                    "linear_attn.conv1d.weight",
                    vec![conv_channels, 1, config.conv_k],
                ),
                ("linear_attn.A_log", vec![value_heads]),
                ("linear_attn.dt_bias", vec![value_heads]),
                ("linear_attn.norm.weight", vec![config.gdn_value_head_dim]),
            ],
            Qwen35LayerKind::FullAttention => vec![
                ("self_attn.q_norm.weight", vec![config.head_dim]),
                ("self_attn.k_norm.weight", vec![config.head_dim]),
            ],
        };
        for (suffix, shape) in mixer {
            push(format!("{root}.{suffix}"), shape);
        }
        push(
            format!("{root}.post_attention_layernorm.weight"),
            vec![hidden],
        );
    }
    push(format!("{TEXT_ROOT}.norm.weight"), vec![hidden]);
    push("lm_head.weight".to_string(), vec![config.vocab, hidden]);
    rows
}

struct PackedCursor<'a> {
    layout: &'a Qwen35Layout,
    rows: &'a [Qwen35PackedLinearSpec],
    next: usize,
}

impl<'a> PackedCursor<'a> {
    fn new(layout: &'a Qwen35Layout, rows: &'a [Qwen35PackedLinearSpec]) -> Self {
        Self {
            layout,
            rows,
            next: 0,
        }
    }

    fn project(
        &mut self,
        b: &Builder,
        x: Traced,
        layer: usize,
        role: Qwen35PackedRole,
    ) -> Result<Traced, Qwen35GraphError> {
        let index = self.next;
        let row = self
            .rows
            .get(index)
            .ok_or_else(|| Qwen35GraphError::PackedCursor {
                index,
                message: format!("missing layer {layer} {role:?}"),
            })?;
        if row.layer != layer
            || row.role != role
            || row.descriptor.shape() != role.logical_shape(self.layout)
        {
            return Err(Qwen35GraphError::PackedCursor {
                index,
                message: format!(
                    "found layer {} {:?} {:?}, expected layer {layer} {role:?} {:?}",
                    row.layer,
                    row.role,
                    row.descriptor.shape(),
                    role.logical_shape(self.layout)
                ),
            });
        }
        self.next += 1;
        packed_linear(b, x, &row.linear_id, row.descriptor, None, None)
            .map_err(Qwen35GraphError::PackedAppend)
    }

    fn finish(self) -> Result<(), Qwen35GraphError> {
        if self.next != self.rows.len() {
            return Err(Qwen35GraphError::PackedCursor {
                index: self.next,
                message: format!("{} rows were not consumed", self.rows.len() - self.next),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum Qwen35TraceMode {
    #[cfg(test)]
    Prefill {
        sequence: usize,
        capacity: usize,
        chunk: usize,
    },
    Decode {
        position: usize,
        capacity: usize,
    },
}

impl Qwen35TraceMode {
    fn capacity(self) -> usize {
        match self {
            Self::Decode { capacity, .. } => capacity,
            #[cfg(test)]
            Self::Prefill { capacity, .. } => capacity,
        }
    }
}

/// Trace the prompt graph for a fresh sequence that starts at position 0.
///
/// The graph publishes the same state names and shapes as the decode graph, so a following decode step
/// reads its state outputs directly. Every state input must be bound to zero. K/V caches are written
/// only at positions `0..sequence`, the recurrent state seeds the chunked delta rule, and the conv cache
/// input only names its state pair: the prefill conv zero-pads the prompt instead of reading it.
/// Continuing a sequence from carried state is the decode graph's job.
///
/// Tracers always emit the dense in-graph embed gather and an unsplit lm_head;
/// `poot_graph_plan::legalize` (card 523a) decides, from the compile target's measured
/// `DeviceCaps`, whether either needs to move off device.
#[cfg(test)]
pub(crate) fn trace_qwen38_text_prefill(
    config: &Qwen35TextConfig,
    sequence: usize,
    capacity: usize,
    chunk: usize,
) -> Result<Graph, Qwen35GraphError> {
    if sequence == 0 {
        return Err(Qwen35GraphError::EmptyPrefill);
    }
    if capacity < sequence {
        return Err(Qwen35GraphError::PrefillCapacity { sequence, capacity });
    }
    if chunk == 0 {
        return Err(Qwen35GraphError::EmptyChunk);
    }
    trace_qwen38_layout(
        &Qwen35Layout::from_config(config)?,
        Qwen35TraceMode::Prefill {
            sequence,
            capacity,
            chunk,
        },
    )
}

/// Trace one position-specialized decode step with fixed-capacity K/V and GDN state.
pub fn trace_qwen38_text_decode(
    config: &Qwen35TextConfig,
    position: usize,
    capacity: usize,
) -> Result<Graph, Qwen35GraphError> {
    if capacity <= position {
        return Err(Qwen35GraphError::DecodeCapacity { position, capacity });
    }
    trace_qwen38_layout(
        &Qwen35Layout::from_config(config)?,
        Qwen35TraceMode::Decode { position, capacity },
    )
}

/// Build one text graph whose dense checkpoint sources keep the checkpoint's BF16 dtype.
fn trace_qwen38_layout(
    layout: &Qwen35Layout,
    mode: Qwen35TraceMode,
) -> Result<Graph, Qwen35GraphError> {
    let rows = packed_linear_specs_for_layout(layout)?;
    let b = Builder::new();
    let (token, step) = match mode {
        #[cfg(test)]
        Qwen35TraceMode::Prefill {
            sequence, chunk, ..
        } => {
            let token = b.slot(Slot::Token, TensorType::new(vec![sequence], DType::I32));
            (
                token,
                StepInputs::Prefill {
                    sequence,
                    chunk,
                    causal_mask: b.slot_named(
                        Slot::Mask,
                        "prefill",
                        TensorType::f32(vec![1, 1, sequence, sequence]),
                    ),
                    tril_incl: {
                        let iota = b.iota(chunk);
                        let rows = b.broadcast(b.reshape(iota, vec![chunk, 1]), vec![chunk, chunk]);
                        let cols = b.broadcast(b.reshape(iota, vec![1, chunk]), vec![chunk, chunk]);
                        let ge = b.binary(BinOp::Ge, rows, cols);
                        b.reshape(ge, vec![1, 1, chunk, chunk])
                    },
                    tril_strict: {
                        let iota = b.iota(chunk);
                        let rows = b.broadcast(b.reshape(iota, vec![chunk, 1]), vec![chunk, chunk]);
                        let cols = b.broadcast(b.reshape(iota, vec![1, chunk]), vec![chunk, chunk]);
                        let cols1 = b.binary_scalar(BinOp::Add, cols, Scalar::F32(1.0));
                        let ge = b.binary(BinOp::Ge, rows, cols1);
                        b.reshape(ge, vec![1, 1, chunk, chunk])
                    },
                },
            )
        }
        Qwen35TraceMode::Decode { position, .. } => {
            let token = b.slot(Slot::Token, TensorType::scalar(DType::I32));
            (
                token,
                StepInputs::Decode {
                    position,
                    position_slot: b.slot(Slot::Pos, TensorType::scalar(DType::I32)),
                },
            )
        }
    };
    let rope_table = TensorType::f32(vec![layout.max_pos, layout.rotary_dim]);
    let rope = (
        b.constant("qwen38_27b.rope.cos", rope_table.clone()),
        b.constant("qwen38_27b.rope.sin", rope_table),
    );
    Qwen35Trace {
        b,
        layout,
        capacity: mode.capacity(),
        rope,
        step,
        packed: PackedCursor::new(layout, &rows),
        state: Vec::with_capacity(layout.layer_types.len() * 2),
    }
    .trace_text_model(token)
}

/// Graph inputs that differ between prefill and decode, created once per trace.
#[derive(Clone, Copy)]
enum StepInputs {
    #[cfg(test)]
    Prefill {
        sequence: usize,
        chunk: usize,
        causal_mask: Traced,
        tril_incl: Traced,
        tril_strict: Traced,
    },
    Decode {
        position: usize,
        position_slot: Traced,
    },
}

impl StepInputs {
    fn sequence(self) -> usize {
        match self {
            Self::Decode { .. } => 1,
            #[cfg(test)]
            Self::Prefill { sequence, .. } => sequence,
        }
    }
}

/// One text graph under construction.
struct Qwen35Trace<'a> {
    b: Builder,
    layout: &'a Qwen35Layout,
    capacity: usize,
    /// Caller-bound `[max_pos, rotary_dim]` cos and sin tables.
    rope: (Traced, Traced),
    step: StepInputs,
    packed: PackedCursor<'a>,
    state: Vec<(Traced, Traced)>,
}

impl Qwen35Trace<'_> {
    fn trace_text_model(mut self, token: Traced) -> Result<Graph, Qwen35GraphError> {
        let layout = self.layout;
        // legalize (card 523a) hosts this gather off-device when the target's buffer limit needs it.
        let embedding = self.dense_source(
            "model.language_model.embed_tokens.weight",
            vec![layout.vocab, layout.hidden],
        );
        let embedded = match self.step {
            #[cfg(test)]
            StepInputs::Prefill { .. } => self.b.gather(embedding, 0, token),
            StepInputs::Decode { .. } => self.b.gather_scalar(embedding, 0, token),
        };
        let embedded = self
            .b
            .reshape(embedded, vec![1, self.step.sequence(), layout.hidden]);
        let mut x = self.b.cast(embedded, DType::F32);
        for (layer, &kind) in layout.layer_types.iter().enumerate() {
            let _layer_scope = self.b.layer_scope(layer);
            x = self.trace_decoder_layer(x, layer, kind)?;
        }
        let normalized = self.zero_centered_norm(x, "model.language_model.norm.weight");
        let last = match self.step {
            #[cfg(test)]
            StepInputs::Prefill { sequence, .. } => {
                self.b.slice(normalized, 1, sequence - 1, sequence)
            }
            StepInputs::Decode { .. } => normalized,
        };
        let logits = self.lm_head(last)?;
        let Self {
            b, packed, state, ..
        } = self;
        packed.finish()?;
        Ok(b.finish_with_state(logits, &state))
    }

    fn trace_decoder_layer(
        &mut self,
        x: Traced,
        layer: usize,
        kind: Qwen35LayerKind,
    ) -> Result<Traced, Qwen35GraphError> {
        let root = format!("model.language_model.layers.{layer}");
        let normed = self.zero_centered_norm(x, &format!("{root}.input_layernorm.weight"));
        let mixer = match kind {
            Qwen35LayerKind::FullAttention => self.trace_full_attention(normed, layer, &root)?,
            Qwen35LayerKind::LinearAttention => self.trace_gated_delta_net(normed, layer, &root)?,
        };
        let x = self.b.binary(BinOp::Add, x, mixer);

        let mlp_input =
            self.zero_centered_norm(x, &format!("{root}.post_attention_layernorm.weight"));
        let gate = self
            .packed
            .project(&self.b, mlp_input, layer, Qwen35PackedRole::MlpGate)?;
        let up = self
            .packed
            .project(&self.b, mlp_input, layer, Qwen35PackedRole::MlpUp)?;
        let activated = swiglu(&self.b, gate, up);
        let mlp = self
            .packed
            .project(&self.b, activated, layer, Qwen35PackedRole::MlpDown)?;
        Ok(self.b.binary(BinOp::Add, x, mlp))
    }

    fn trace_full_attention(
        &mut self,
        x: Traced,
        layer: usize,
        root: &str,
    ) -> Result<Traced, Qwen35GraphError> {
        let layout = self.layout;
        let query_gate = self
            .packed
            .project(&self.b, x, layer, Qwen35PackedRole::AttentionQ)?;
        let (query, gate) =
            qwen3next_split_query_gate(&self.b, query_gate, layout.n_heads, layout.head_dim);
        let key = self
            .packed
            .project(&self.b, x, layer, Qwen35PackedRole::AttentionK)?;
        let value = self
            .packed
            .project(&self.b, x, layer, Qwen35PackedRole::AttentionV)?;
        let q_norm =
            self.zero_centered_weight(&format!("{root}.self_attn.q_norm.weight"), layout.head_dim);
        let k_norm =
            self.zero_centered_weight(&format!("{root}.self_attn.k_norm.weight"), layout.head_dim);

        let b = &self.b;
        let cache = TensorType::f32(vec![1, layout.n_kv_heads, self.capacity, layout.head_dim]);
        let k_cache = b.state_input(
            &format!("{root}.self_attn.k_cache"),
            cache.clone(),
            StateRole::Recurrent,
        );
        let v_cache = b.state_input(
            &format!("{root}.self_attn.v_cache"),
            cache,
            StateRole::Recurrent,
        );
        let (cos, sin) = self.rope;
        let (attention, k_out, v_out) = match self.step {
            #[cfg(test)]
            StepInputs::Prefill { causal_mask, .. } => qwen3next_gated_attention_prefill_projected(
                b,
                query,
                gate,
                key,
                value,
                q_norm,
                k_norm,
                cos,
                sin,
                causal_mask,
                k_cache,
                v_cache,
                layout.n_heads,
                layout.n_kv_heads,
                layout.head_dim,
                layout.eps,
            ),
            StepInputs::Decode {
                position,
                position_slot,
            } => qwen3next_gated_attention_projected(
                b,
                query,
                gate,
                key,
                value,
                q_norm,
                k_norm,
                cos,
                sin,
                position_slot,
                k_cache,
                v_cache,
                layout.n_heads,
                layout.n_kv_heads,
                layout.head_dim,
                position,
                layout.eps,
            ),
        };
        self.state.push((k_cache, k_out));
        self.state.push((v_cache, v_out));
        self.packed
            .project(&self.b, attention, layer, Qwen35PackedRole::AttentionOut)
    }

    fn trace_gated_delta_net(
        &mut self,
        x: Traced,
        layer: usize,
        root: &str,
    ) -> Result<Traced, Qwen35GraphError> {
        let layout = self.layout;
        let key_heads = layout.gdn_num_k_heads;
        let value_heads = layout.gdn_num_v_heads;
        let head_dim = layout.gdn_head_dim;
        let conv_dim = (2 * key_heads + value_heads) * head_dim;
        let qkv = self
            .packed
            .project(&self.b, x, layer, Qwen35PackedRole::GdnQkv)?;
        let z = self
            .packed
            .project(&self.b, x, layer, Qwen35PackedRole::GdnZ)?;
        let alpha_raw = self.dense_linear(
            x,
            &format!("{root}.linear_attn.in_proj_a.weight"),
            value_heads,
            layout.hidden,
        );
        let beta_raw = self.dense_linear(
            x,
            &format!("{root}.linear_attn.in_proj_b.weight"),
            value_heads,
            layout.hidden,
        );
        let dt_bias =
            self.dense_source_f32(&format!("{root}.linear_attn.dt_bias"), vec![value_heads]);
        let a_log = self.dense_source_f32(&format!("{root}.linear_attn.A_log"), vec![value_heads]);
        let conv = self.dense_source_f32(
            &format!("{root}.linear_attn.conv1d.weight"),
            vec![conv_dim, 1, layout.conv_k],
        );
        // `Qwen3_5RMSNormGated` scales by its stored weight. Only `Qwen3_5RMSNorm` adds one.
        let norm_weight =
            self.dense_source_f32(&format!("{root}.linear_attn.norm.weight"), vec![head_dim]);

        let b = &self.b;
        let beta = sigmoid(b, beta_raw);
        // HF: g = -exp(A_log) * softplus(a + dt_bias).
        let alpha = softplus(b, b.binary(BinOp::Add, alpha_raw, dt_bias));
        let decay = b.unary(UnOp::Neg, b.unary(UnOp::Exp, a_log));
        let g = b.binary(BinOp::Mul, alpha, decay);
        // PyTorch depthwise `[conv_dim, 1, K]` storage to the shared conv's `[K, conv_dim]`.
        let conv = b.transpose(b.reshape(conv, vec![conv_dim, layout.conv_k]), vec![1, 0]);
        let conv_in = b.state_input(
            &format!("{root}.linear_attn.conv_cache"),
            TensorType::f32(vec![1, layout.conv_k - 1, conv_dim]),
            StateRole::Recurrent,
        );
        let ssm_in = b.state_input(
            &format!("{root}.linear_attn.recurrent_state"),
            TensorType::f32(vec![1, value_heads, head_dim, head_dim]),
            StateRole::Recurrent,
        );
        let (mixed, conv_out, ssm_out) = match self.step {
            #[cfg(test)]
            StepInputs::Prefill {
                chunk,
                tril_incl,
                tril_strict,
                ..
            } => qwen3next_gdn_prefill_projected(
                b,
                qkv,
                z,
                beta,
                g,
                conv,
                norm_weight,
                ssm_in,
                tril_incl,
                tril_strict,
                key_heads,
                value_heads,
                head_dim,
                layout.conv_k,
                chunk,
                layout.eps,
                GDN_HEAD_ORDER,
            ),
            StepInputs::Decode { .. } => qwen3next_gdn_projected(
                b,
                qkv,
                z,
                beta,
                g,
                conv,
                norm_weight,
                conv_in,
                ssm_in,
                key_heads,
                value_heads,
                head_dim,
                layout.conv_k,
                layout.eps,
                GDN_HEAD_ORDER,
            ),
        };
        self.state.push((conv_in, conv_out));
        self.state.push((ssm_in, ssm_out));
        self.packed
            .project(&self.b, mixed, layer, Qwen35PackedRole::GdnOut)
    }

    /// The LM head multiplies the F32 activation by the BF16 weight in one mixed-dtype `MatMul` with an F32
    /// product. The activation is not narrowed, and a production graph never holds an F32 copy of the
    /// `[vocab, hidden]` weight. `poot_graph_plan::legalize` (card 523a) is the one place a
    /// device limit is read; this tracer always emits the single unsplit matmul.
    fn lm_head(&self, x: Traced) -> Result<Traced, Qwen35GraphError> {
        let b = &self.b;
        let vocab = self.layout.vocab;
        let hidden = self.layout.hidden;
        let weight = self.dense_source("lm_head.weight", vec![vocab, hidden]);
        Ok(linear(b, x, b.transpose(weight, vec![1, 0]), None))
    }

    fn dense_source(&self, name: &str, shape: Vec<usize>) -> Traced {
        self.b
            .constant(name, TensorType::new(shape, CHECKPOINT_DENSE_DTYPE))
    }

    fn dense_source_f32(&self, name: &str, shape: Vec<usize>) -> Traced {
        self.b.cast(self.dense_source(name, shape), DType::F32)
    }

    /// Apply a PyTorch `[out, in]` dense linear weight to `x`.
    fn dense_linear(&self, x: Traced, name: &str, out_dim: usize, in_dim: usize) -> Traced {
        let weight = self.dense_source_f32(name, vec![out_dim, in_dim]);
        linear(&self.b, x, self.b.transpose(weight, vec![1, 0]), None)
    }

    /// `Qwen3_5RMSNorm` stores its weight zero-centered and scales by `1 + weight`.
    fn zero_centered_weight(&self, name: &str, width: usize) -> Traced {
        let stored = self.dense_source_f32(name, vec![width]);
        self.b.binary_scalar(BinOp::Add, stored, Scalar::F32(1.0))
    }

    fn zero_centered_norm(&self, x: Traced, name: &str) -> Traced {
        let weight = self.zero_centered_weight(name, self.layout.hidden);
        rmsnorm(&self.b, x, weight, self.layout.eps)
    }
}

#[cfg(test)]
pub mod card416_test_support {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use poot_eval::exact_dense::DenseOwnerTensorView;
    use poot_eval::{EvalBudget, EvalOptions, ExactValue, Value, eval};
    use poot_graph_ir::{Eqn, OpKind, Operand, PackedSourceName, Storage, ValueId};
    use poot_load::packed_safetensors::{ExactSourceOwner, TensorDisposition};
    use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, SourceRole};
    use poot_tensor::HostTensor;

    use super::*;
    use crate::test_support::safetensors::{
        LoadedCheckpoint, SourceRow, bf16_bytes, load_checkpoint, truncate_to_bf16,
    };

    const TINY_CAPACITY: usize = 3;
    const TINY_PROMPT: [i32; 3] = [3, 6, 1];

    /// Small but non-degenerate: GQA on both mixers (`H_v / H_k = 3` for GDN, 2 for attention), partial
    /// rotary, a hidden width distinct from the MLP width, and attention between GDN layers.
    fn tiny_layout() -> Qwen35Layout {
        Qwen35Layout {
            vocab: 7,
            hidden: 6,
            intermediate: 5,
            layer_types: vec![
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::LinearAttention,
                Qwen35LayerKind::FullAttention,
                Qwen35LayerKind::LinearAttention,
            ],
            eps: 1e-6,
            max_pos: 4,
            rotary_dim: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            gdn_num_k_heads: 2,
            gdn_num_v_heads: 6,
            gdn_head_dim: 2,
            conv_k: 3,
        }
    }

    fn tiny_graph(mode: Qwen35TraceMode) -> Result<Graph, Qwen35GraphError> {
        trace_qwen38_layout(&tiny_layout(), mode)
    }

    #[test]
    fn qwen35_packed_projection_returns_builder_errors() -> Result<(), Qwen35GraphError> {
        let layout = tiny_layout();
        let rows = packed_linear_specs_for_layout(&layout)?;
        let row = &rows[0];
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, row.descriptor.shape()[1]]));
        let taken = PackedSourceName::weight(&row.linear_id);
        b.constant(taken.as_str(), TensorType::f32(vec![1]));
        let generation = b.generation();

        let mut cursor = PackedCursor::new(&layout, &rows);
        assert!(matches!(
            cursor.project(&b, x, row.layer, row.role),
            Err(Qwen35GraphError::PackedAppend(BuilderAppendError::NameCollision { name, .. }))
                if name == taken.as_str()
        ));
        assert_eq!(b.generation(), generation);
        Ok(())
    }

    /// Finite E4M3FN codes with mixed signs and magnitudes.
    const E4M3_CODES: [u8; 9] = [0x00, 0x20, 0x28, 0x2c, 0x30, 0x34, 0xa4, 0xa8, 0xb0];

    /// SplitMix64 finalizer: spreads a seed so neighboring inputs give unrelated outputs.
    fn mix64(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Weight codes and block scale for the packed projection named `module`. Seeding by the checkpoint
    /// module path, not by table position, gives same-shaped projections such as `k_proj`/`v_proj` or
    /// `gate_proj`/`up_proj` different weights, so a projection bound under the wrong name changes the
    /// logits.
    fn tiny_packed_codes(module: &str, [out, input]: [usize; 2]) -> (Vec<u8>, f32) {
        let seed = module
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            });
        let codes = (0..out * input)
            .map(|element| {
                let pick = mix64(seed ^ element as u64) % E4M3_CODES.len() as u64;
                E4M3_CODES[pick as usize]
            })
            .collect();
        (codes, [0.75, 1.25][(mix64(seed) % 2) as usize])
    }

    /// The scale operand's element count: its byte length divided by one encoded scale's width.
    fn tiny_packed_scale_elements(descriptor: PackedWeight) -> usize {
        let bytes_per_element = descriptor
            .format()
            .descriptor()
            .planar_operand(OperandRole::Scale)
            .expect("a packed weight has a Scale operand")
            .element_bytes();
        descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)) / bytes_per_element
    }

    /// Owners keyed by linear id. Codes come from the id the graph asks for, so the graph and the
    /// reference agree only when the graph's role table names each projection correctly.
    fn tiny_packed_owners(layout: &Qwen35Layout) -> HashMap<String, Arc<PackedPayload>> {
        packed_linear_specs_for_layout(layout)
            .expect("tiny packed table")
            .into_iter()
            .map(|row| {
                let (codes, scale) = tiny_packed_codes(&row.linear_id, row.descriptor.shape());
                let bf16_scale = u16::try_from(scale.to_bits() >> 16)
                    .expect("bf16 is the top half of an f32")
                    .to_le_bytes();
                let scales = (0..tiny_packed_scale_elements(row.descriptor))
                    .flat_map(|_| bf16_scale)
                    .collect::<Vec<_>>();
                let owner = PackedPayload::try_new(
                    row.descriptor,
                    [
                        (SourceRole::Planar(OperandRole::Codes), codes.into()),
                        (SourceRole::Planar(OperandRole::Scale), scales.into()),
                    ],
                )
                .expect("tiny packed payload");
                (row.linear_id, Arc::new(owner))
            })
            .collect()
    }

    /// Deterministic value in `(-0.5, 0.5)`, never zero. Names that differ in one character get
    /// different sequences, so paired tensors such as `q_norm`/`k_norm` or `in_proj_a`/`in_proj_b` differ.
    fn fixture_value(name: &str, index: usize) -> f32 {
        let seed = name.bytes().fold(0usize, |seed, byte| {
            seed.wrapping_mul(31).wrapping_add(usize::from(byte))
        });
        ((seed % 23 + index * 17) % 23) as f32 / 23.0 - 0.5
    }

    fn tiny_dense(name: &str, shape: &[usize]) -> Vec<f32> {
        let numel = shape.iter().product::<usize>();
        match name {
            "qwen38_27b.rope.cos" | "qwen38_27b.rope.sin" => {
                let rotary = shape[1];
                let cosine = name.ends_with("cos");
                // HF repeats each inverse frequency over both rotary halves. Row 0 is not the identity,
                // so a RoPE that is skipped cannot match the reference at any position.
                (0..shape[0])
                    .flat_map(|position| {
                        (0..rotary).map(move |dim| {
                            let angle =
                                0.9 * (position + 1) as f32 / (1 + dim % (rotary / 2)) as f32;
                            if cosine { angle.cos() } else { angle.sin() }
                        })
                    })
                    .collect()
            }
            // Checkpoint rows are BF16, so every value is exactly representable in BF16.
            // One-centered like the real gated norm, and never exactly one.
            _ if name.ends_with("linear_attn.norm.weight") => (0..numel)
                .map(|index| truncate_to_bf16(1.0 + fixture_value(name, index)))
                .collect(),
            _ => (0..numel)
                .map(|index| truncate_to_bf16(fixture_value(name, index)))
                .collect(),
        }
    }

    /// Checkpoint owners for the tiny model: packed FP8 payloads and the authenticated BF16 dense rows.
    struct TinyOwners {
        packed: HashMap<String, Arc<PackedPayload>>,
        dense: HashMap<String, Arc<ExactSourceOwner>>,
        _checkpoint: LoadedCheckpoint,
    }

    fn tiny_owners(layout: &Qwen35Layout) -> Result<TinyOwners, Qwen35GraphError> {
        // Prefill and decode graphs read the same dense checkpoint rows.
        let graph = trace_qwen38_layout(
            layout,
            Qwen35TraceMode::Decode {
                position: 0,
                capacity: TINY_CAPACITY,
            },
        )?;
        let rows = graph
            .consts
            .iter()
            .map(|&id| graph.meta(id))
            .filter(|meta| meta.aval.dtype == DType::BF16)
            .map(|meta| {
                let name = meta.name.clone().expect("named dense source");
                let bytes = bf16_bytes(&name, &tiny_dense(&name, &meta.aval.shape));
                let row = SourceRow {
                    name,
                    dtype: "BF16",
                    shape: meta.aval.shape.clone(),
                    bytes,
                };
                (row, TensorDisposition::DenseBf16)
            })
            .collect();
        let checkpoint = load_checkpoint("Qwen/Qwen3.8-27B-FP8", "tiny", b"{}", rows);
        Ok(TinyOwners {
            packed: tiny_packed_owners(layout),
            dense: checkpoint
                .mixed
                .exact_metadata
                .iter()
                .map(|(name, source)| (name.clone(), Arc::clone(&source.owner)))
                .collect(),
            _checkpoint: checkpoint,
        })
    }

    /// Bind every graph input. BF16 checkpoint rows bind as exact owner views through the Card 370
    /// binder. A legalized graph's `Slot::TokenEmbed` (card 523a) is gathered from the embed owner
    /// for `tokens`.
    fn tiny_bindings(
        graph: &Graph,
        owners: &TinyOwners,
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
        let dense_views = graph
            .inputs
            .iter()
            .filter(|&&id| graph.aval(id).dtype == DType::BF16)
            .map(|&id| {
                let name = graph.meta(id).name.as_deref().unwrap_or_default();
                let owner = owners
                    .dense
                    .get(name)
                    .unwrap_or_else(|| panic!("no tiny BF16 owner {name}"));
                (
                    id,
                    DenseOwnerTensorView::new(Arc::clone(owner)).expect("BF16 owner view"),
                )
            })
            .collect::<Vec<_>>();
        let dense_owner_map = &owners.dense;
        let packed_owner_map = &owners.packed;
        let mut inputs = graph
            .inputs
            .iter()
            .filter(|&&id| graph.aval(id).dtype != DType::BF16)
            .map(|&id| {
                let meta = graph.meta(id);
                let name = meta.name.as_deref().unwrap_or_default();
                let shape = meta.aval.shape.clone();
                let value = if let Some(source) = PackedSourceName::parse(name) {
                    Value::Packed(PackedComponentRef::new(
                        Arc::clone(&packed_owner_map[source.linear_id()]),
                        source.role(),
                    ))
                } else {
                    match meta.storage {
                        Storage::Slot(Slot::Token) => {
                            Value::Host(HostTensor::i32(shape, tokens.to_vec()))
                        }
                        Storage::Slot(Slot::Pos) => {
                            let position = position.expect("decode position");
                            let position = i32::try_from(position).expect("tiny position");
                            Value::Host(HostTensor::i32(shape, vec![position]))
                        }
                        // Card 453 D1 host-embed: gather the already-embedded F32 rows from the
                        // dense embed owner (the same rows an in-graph gather would read).
                        Storage::Slot(Slot::TokenEmbed) => {
                            Value::Host(gather_token_embed_f32(dense_owner_map, tokens, &shape))
                        }
                        Storage::Slot(Slot::Mask) => {
                            let name = meta.name.as_deref().expect("mask slot without a name");
                            assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                            let sequence = shape[2];
                            let data = (0..sequence)
                                .flat_map(|query| {
                                    (0..sequence)
                                        .map(move |key| if key <= query { 0.0 } else { -1.0e9 })
                                })
                                .collect();
                            Value::Host(HostTensor::f32(shape, data))
                        }
                        Storage::State => carried.map_or_else(
                            || Value::Host(HostTensor::f32(shape, vec![0.0; meta.aval.numel()])),
                            |values| values[state_index[&id]].clone(),
                        ),
                        Storage::Const => {
                            Value::Host(HostTensor::f32(shape.clone(), tiny_dense(name, &shape)))
                        }
                        other => panic!("unexpected tiny graph input storage {other:?}"),
                    }
                };
                (id, value)
            })
            .collect::<HashMap<_, _>>();
        for (id, view) in dense_views {
            inputs.insert(id, Value::Owner(ExactValue::Dense(view)));
        }
        inputs
    }

    /// Gather host-embed F32 rows for `Slot::TokenEmbed` from the tiny embed owner.
    fn gather_token_embed_f32(
        dense_owners: &HashMap<String, Arc<ExactSourceOwner>>,
        tokens: &[i32],
        shape: &[usize],
    ) -> HostTensor {
        let hidden = match shape {
            [h] => *h,
            [_, h] => *h,
            _ => panic!("TokenEmbed shape {shape:?}"),
        };
        let owner = dense_owners
            .get("model.language_model.embed_tokens.weight")
            .expect("tiny embed owner for host-embed gather");
        let bytes = owner.bytes();
        let mut data = Vec::with_capacity(tokens.len() * hidden);
        for &token in tokens {
            let row = usize::try_from(token).expect("token id");
            let start = row * hidden * 2;
            for element in 0..hidden {
                let word = u16::from_le_bytes([
                    bytes[start + 2 * element],
                    bytes[start + 2 * element + 1],
                ]);
                data.push(f32::from_bits(u32::from(word) << 16));
            }
        }
        HostTensor::f32(shape.to_vec(), data)
    }

    fn assert_matches_reference(actual: &Value, expected: &[f32], label: &str) {
        let Value::Host(actual) = actual else {
            panic!("{label} must be a dense value")
        };
        assert_eq!(
            actual.as_f32().unwrap().len(),
            expected.len(),
            "{label} length"
        );
        for (index, (&actual, &expected)) in
            actual.as_f32().unwrap().iter().zip(expected).enumerate()
        {
            assert!(
                (actual - expected).abs() <= 1e-4,
                "{label}[{index}]: graph {actual} != reference {expected}"
            );
        }
    }

    /// Decode one E4M3FN code from its sign, exponent, and mantissa fields.
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

    fn matvec(weight: &[f32], x: &[f32]) -> Vec<f32> {
        weight
            .chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
            .collect()
    }

    fn rms_normalize(x: &[f32], eps: f32) -> Vec<f32> {
        let mean_square = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inverse = (mean_square + eps).sqrt().recip();
        x.iter().map(|v| v * inverse).collect()
    }

    fn zero_centered_rms_norm(x: &[f32], stored: &[f32], eps: f32) -> Vec<f32> {
        rms_normalize(x, eps)
            .iter()
            .zip(stored)
            .map(|(v, w)| v * (1.0 + w))
            .collect()
    }

    fn l2_normalize(x: &[f32], eps: f32) -> Vec<f32> {
        let inverse = (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt().recip();
        x.iter().map(|v| v * inverse).collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu(x: f32) -> f32 {
        x * sigmoid(x)
    }

    fn add_assign(target: &mut [f32], update: &[f32]) {
        target.iter_mut().zip(update).for_each(|(t, u)| *t += u);
    }

    enum ReferenceLayerState {
        /// `conv`: the last `K - 1` raw QKV inputs, oldest first. `recurrent`: `[H_v, D_k, D_v]`.
        Gdn { conv: Vec<f32>, recurrent: Vec<f32> },
        /// `[n_kv_heads, capacity, head_dim]` post-RoPE keys and values.
        Attention { keys: Vec<f32>, values: Vec<f32> },
    }

    /// Token-by-token Qwen3.5 text model written from HF `modeling_qwen3_5.py`
    /// (`Qwen3_5GatedDeltaNet`, `torch_recurrent_gated_delta_rule`, `Qwen3_5RMSNormGated`,
    /// `Qwen3_5Attention`, `apply_rotary_pos_emb`, `Qwen3_5MLP`, `Qwen3_5RMSNorm`). It shares no math with
    /// the graph builders and none of its role table: packed weights are addressed by literal HF module
    /// paths, decoded from the fixture codes here, and every operation is a plain f32 loop.
    #[derive(Clone, Copy)]
    enum ReferenceRope {
        Card416Fixture,
        Card453Production { theta: f32 },
    }

    struct HfReference {
        layout: Qwen35Layout,
        capacity: usize,
        layers: Vec<ReferenceLayerState>,
        rope: ReferenceRope,
    }

    /// Apply the packed projection named `module` (an HF module path) with `out` output features.
    fn reference_projection(module: &str, out: usize, x: &[f32]) -> Vec<f32> {
        let (codes, scale) = tiny_packed_codes(module, [out, x.len()]);
        let weight = codes
            .into_iter()
            .map(|code| e4m3_value(code) * scale)
            .collect::<Vec<_>>();
        matvec(&weight, x)
    }

    impl HfReference {
        fn new(layout: Qwen35Layout, capacity: usize) -> Self {
            Self::with_rope(layout, capacity, ReferenceRope::Card416Fixture)
        }

        fn card453(layout: Qwen35Layout, capacity: usize) -> Self {
            Self::with_rope(
                layout,
                capacity,
                ReferenceRope::Card453Production {
                    theta: 10_000_000.0,
                },
            )
        }

        fn with_rope(layout: Qwen35Layout, capacity: usize, rope: ReferenceRope) -> Self {
            let conv_dim =
                (2 * layout.gdn_num_k_heads + layout.gdn_num_v_heads) * layout.gdn_head_dim;
            let layers = layout
                .layer_types
                .iter()
                .map(|kind| match kind {
                    Qwen35LayerKind::LinearAttention => ReferenceLayerState::Gdn {
                        conv: vec![0.0; (layout.conv_k - 1) * conv_dim],
                        recurrent: vec![
                            0.0;
                            layout.gdn_num_v_heads
                                * layout.gdn_head_dim
                                * layout.gdn_head_dim
                        ],
                    },
                    Qwen35LayerKind::FullAttention => {
                        let cache = vec![0.0; layout.n_kv_heads * capacity * layout.head_dim];
                        ReferenceLayerState::Attention {
                            keys: cache.clone(),
                            values: cache,
                        }
                    }
                })
                .collect();
            Self {
                layout,
                capacity,
                layers,
                rope,
            }
        }

        /// Run one token and return its logits.
        fn step(&mut self, token: i32, position: usize) -> Vec<f32> {
            let layout = self.layout.clone();
            let hidden = layout.hidden;
            let token = usize::try_from(token).expect("token id");
            let embedding = tiny_dense(
                "model.language_model.embed_tokens.weight",
                &[layout.vocab, hidden],
            );
            let mut h = embedding[token * hidden..(token + 1) * hidden].to_vec();
            for (layer, kind) in layout.layer_types.iter().enumerate() {
                let root = format!("model.language_model.layers.{layer}");
                let input_norm = tiny_dense(&format!("{root}.input_layernorm.weight"), &[hidden]);
                let normed = zero_centered_rms_norm(&h, &input_norm, layout.eps);
                let mixer = match kind {
                    Qwen35LayerKind::LinearAttention => self.gated_delta_net(layer, &root, &normed),
                    Qwen35LayerKind::FullAttention => {
                        self.attention(layer, &root, &normed, position)
                    }
                };
                add_assign(&mut h, &mixer);

                let post_norm = tiny_dense(
                    &format!("{root}.post_attention_layernorm.weight"),
                    &[hidden],
                );
                let mlp_input = zero_centered_rms_norm(&h, &post_norm, layout.eps);
                let gate = reference_projection(
                    &format!("{root}.mlp.gate_proj"),
                    layout.intermediate,
                    &mlp_input,
                );
                let up = reference_projection(
                    &format!("{root}.mlp.up_proj"),
                    layout.intermediate,
                    &mlp_input,
                );
                let activated = gate
                    .iter()
                    .zip(&up)
                    .map(|(&gate, &up)| silu(gate) * up)
                    .collect::<Vec<_>>();
                add_assign(
                    &mut h,
                    &reference_projection(&format!("{root}.mlp.down_proj"), hidden, &activated),
                );
            }
            let final_norm = tiny_dense("model.language_model.norm.weight", &[hidden]);
            let normed = zero_centered_rms_norm(&h, &final_norm, layout.eps);
            matvec(
                &tiny_dense("lm_head.weight", &[layout.vocab, hidden]),
                &normed,
            )
        }

        fn gated_delta_net(&mut self, layer: usize, root: &str, x: &[f32]) -> Vec<f32> {
            let layout = &self.layout;
            let (key_heads, value_heads) = (layout.gdn_num_k_heads, layout.gdn_num_v_heads);
            let (d, kernel, eps) = (layout.gdn_head_dim, layout.conv_k, layout.eps);
            let key_dim = key_heads * d;
            let conv_dim = 2 * key_dim + value_heads * d;
            let dense = |suffix: &str, shape: &[usize]| {
                tiny_dense(&format!("{root}.linear_attn.{suffix}"), shape)
            };
            let b = matvec(&dense("in_proj_b.weight", &[value_heads, layout.hidden]), x);
            let a = matvec(&dense("in_proj_a.weight", &[value_heads, layout.hidden]), x);
            let dt_bias = dense("dt_bias", &[value_heads]);
            let a_log = dense("A_log", &[value_heads]);
            let conv_weight = dense("conv1d.weight", &[conv_dim, 1, kernel]);
            let norm_weight = dense("norm.weight", &[d]);
            let qkv = reference_projection(&format!("{root}.linear_attn.in_proj_qkv"), conv_dim, x);
            let z =
                reference_projection(&format!("{root}.linear_attn.in_proj_z"), value_heads * d, x);
            let hidden = layout.hidden;

            let ReferenceLayerState::Gdn { conv, recurrent } = &mut self.layers[layer] else {
                panic!("layer {layer} is not a GDN layer");
            };
            // Depthwise causal conv over the last `kernel` inputs, oldest first, then SiLU.
            let mut window = conv.clone();
            window.extend_from_slice(&qkv);
            let mixed = (0..conv_dim)
                .map(|channel| {
                    silu(
                        (0..kernel)
                            .map(|tap| {
                                conv_weight[channel * kernel + tap]
                                    * window[tap * conv_dim + channel]
                            })
                            .sum(),
                    )
                })
                .collect::<Vec<_>>();
            *conv = window[conv_dim..].to_vec();
            let (query, rest) = mixed.split_at(key_dim);
            let (key, value) = rest.split_at(key_dim);

            let n_rep = value_heads / key_heads;
            let scale = (d as f32).sqrt().recip();
            let mut core = Vec::with_capacity(value_heads * d);
            for head in 0..value_heads {
                // `repeat_interleave(n_rep, dim=2)`: value head `head` reads key head `head / n_rep`.
                let source = head / n_rep;
                let q = l2_normalize(&query[source * d..(source + 1) * d], eps);
                let k = l2_normalize(&key[source * d..(source + 1) * d], eps);
                let v = &value[head * d..(head + 1) * d];
                let beta = sigmoid(b[head]);
                let g = -a_log[head].exp() * (a[head] + dt_bias[head]).exp().ln_1p();

                let state = &mut recurrent[head * d * d..(head + 1) * d * d];
                let decay = g.exp();
                state.iter_mut().for_each(|s| *s *= decay);
                let delta = (0..d)
                    .map(|j| (v[j] - (0..d).map(|i| state[i * d + j] * k[i]).sum::<f32>()) * beta)
                    .collect::<Vec<_>>();
                for (row, &k_i) in state.chunks_exact_mut(d).zip(&k) {
                    row.iter_mut()
                        .zip(&delta)
                        .for_each(|(s, delta)| *s += k_i * delta);
                }
                let out = (0..d)
                    .map(|j| (0..d).map(|i| state[i * d + j] * q[i] * scale).sum())
                    .collect::<Vec<f32>>();

                // `Qwen3_5RMSNormGated`: stored weight, then the SiLU(z) gate.
                let normed = rms_normalize(&out, eps);
                core.extend((0..d).map(|j| normed[j] * norm_weight[j] * silu(z[head * d + j])));
            }
            reference_projection(&format!("{root}.linear_attn.out_proj"), hidden, &core)
        }

        fn attention(&mut self, layer: usize, root: &str, x: &[f32], position: usize) -> Vec<f32> {
            let layout = &self.layout;
            let (n_heads, n_kv_heads, head_dim) =
                (layout.n_heads, layout.n_kv_heads, layout.head_dim);
            let (rotary, eps, capacity) = (layout.rotary_dim, layout.eps, self.capacity);
            let q_norm = tiny_dense(&format!("{root}.self_attn.q_norm.weight"), &[head_dim]);
            let k_norm = tiny_dense(&format!("{root}.self_attn.k_norm.weight"), &[head_dim]);
            let (cos, sin) = match self.rope {
                ReferenceRope::Card416Fixture => (
                    tiny_dense("qwen38_27b.rope.cos", &[layout.max_pos, rotary]),
                    tiny_dense("qwen38_27b.rope.sin", &[layout.max_pos, rotary]),
                ),
                ReferenceRope::Card453Production { theta } => {
                    let half = rotary / 2;
                    let angles = (0..layout.max_pos)
                        .flat_map(|position| {
                            (0..rotary).map(move |column| {
                                let frequency = column % half;
                                let exponent = -2.0 * frequency as f32 / rotary as f32;
                                position as f32 * theta.powf(exponent)
                            })
                        })
                        .collect::<Vec<_>>();
                    (
                        angles.iter().map(|angle| angle.cos()).collect(),
                        angles.iter().map(|angle| angle.sin()).collect(),
                    )
                }
            };
            let cos = &cos[position * rotary..(position + 1) * rotary];
            let sin = &sin[position * rotary..(position + 1) * rotary];
            // `apply_rotary_pos_emb`: rotate the leading `rotary` dims, pass the rest through.
            let rope = |x: &mut [f32]| {
                let half = rotary / 2;
                let rotated = (0..rotary)
                    .map(|i| {
                        let rotate_half = if i < half { -x[i + half] } else { x[i - half] };
                        x[i] * cos[i] + rotate_half * sin[i]
                    })
                    .collect::<Vec<_>>();
                x[..rotary].copy_from_slice(&rotated);
            };
            let hidden = layout.hidden;
            let query_gate = reference_projection(
                &format!("{root}.self_attn.q_proj"),
                2 * n_heads * head_dim,
                x,
            );
            let key = reference_projection(
                &format!("{root}.self_attn.k_proj"),
                n_kv_heads * head_dim,
                x,
            );
            let value = reference_projection(
                &format!("{root}.self_attn.v_proj"),
                n_kv_heads * head_dim,
                x,
            );

            let ReferenceLayerState::Attention { keys, values } = &mut self.layers[layer] else {
                panic!("layer {layer} is not an attention layer");
            };
            for kv_head in 0..n_kv_heads {
                let range = kv_head * head_dim..(kv_head + 1) * head_dim;
                let mut k = zero_centered_rms_norm(&key[range.clone()], &k_norm, eps);
                rope(&mut k);
                let slot = (kv_head * capacity + position) * head_dim;
                keys[slot..slot + head_dim].copy_from_slice(&k);
                values[slot..slot + head_dim].copy_from_slice(&value[range]);
            }
            let n_rep = n_heads / n_kv_heads;
            let mut gated = Vec::with_capacity(n_heads * head_dim);
            for head in 0..n_heads {
                // Each head's projection is `[query, gate]`.
                let offset = head * 2 * head_dim;
                let mut q =
                    zero_centered_rms_norm(&query_gate[offset..offset + head_dim], &q_norm, eps);
                rope(&mut q);
                let gate_logits = &query_gate[offset + head_dim..offset + 2 * head_dim];
                // `repeat_kv`: query head `head` reads KV head `head / n_rep`.
                let kv_head = head / n_rep;
                let cached = |cache: &[f32], p: usize| {
                    let slot = (kv_head * capacity + p) * head_dim;
                    cache[slot..slot + head_dim].to_vec()
                };
                let scores = (0..=position)
                    .map(|p| {
                        let k = cached(keys.as_slice(), p);
                        q.iter().zip(&k).map(|(q, k)| q * k).sum::<f32>() / (head_dim as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                let total = exp.iter().sum::<f32>();
                for (i, &gate) in gate_logits.iter().enumerate() {
                    let mixed = (0..=position)
                        .map(|p| exp[p] / total * cached(values.as_slice(), p)[i])
                        .sum::<f32>();
                    gated.push(mixed * sigmoid(gate));
                }
            }
            reference_projection(&format!("{root}.self_attn.o_proj"), hidden, &gated)
        }

        /// State tensors in `Graph::state` order.
        fn state(&self) -> Vec<Vec<f32>> {
            self.layers
                .iter()
                .flat_map(|layer| match layer {
                    ReferenceLayerState::Gdn { conv, recurrent } => {
                        [conv.clone(), recurrent.clone()]
                    }
                    ReferenceLayerState::Attention { keys, values } => {
                        [keys.clone(), values.clone()]
                    }
                })
                .collect()
        }
    }

    /// Opaque owner-authored Card 416 fixture, available to this crate's tests only.
    #[derive(Clone, Debug)]
    pub struct Card416Fixture {
        layout: Qwen35Layout,
        sources: Card416SourceSet,
    }

    #[derive(Clone, Debug)]
    pub struct Card416SourceSet {
        packed: Vec<Card416PackedSource>,
        dense: Vec<Card416DenseSource>,
    }

    #[derive(Clone, Debug)]
    pub struct Card416PackedSource {
        linear_id: String,
        descriptor: PackedWeight,
        weight_bytes: Vec<u8>,
        scale_bytes: Vec<u8>,
    }

    #[derive(Clone, Debug)]
    pub struct Card416DenseSource {
        name: String,
        shape: Vec<usize>,
        dtype: DType,
        bytes: Vec<u8>,
    }

    #[derive(Clone, Debug)]
    pub struct Card416ExpectedStep {
        pub logits: Vec<f32>,
        pub state: Vec<Vec<f32>>,
    }

    pub fn fixture() -> Card416Fixture {
        let layout = tiny_layout();
        let packed = packed_linear_specs_for_layout(&layout)
            .expect("Card 416 packed fixture table")
            .into_iter()
            .map(|row| {
                let (weight_bytes, scale) =
                    tiny_packed_codes(&row.linear_id, row.descriptor.shape());
                let scale = u16::try_from(scale.to_bits() >> 16)
                    .expect("BF16 scale is the upper half of f32")
                    .to_le_bytes();
                let scale_bytes = (0..tiny_packed_scale_elements(row.descriptor))
                    .flat_map(|_| scale)
                    .collect();
                Card416PackedSource {
                    linear_id: row.linear_id,
                    descriptor: row.descriptor,
                    weight_bytes,
                    scale_bytes,
                }
            })
            .collect();
        let graph = trace_qwen38_layout(
            &layout,
            Qwen35TraceMode::Decode {
                position: 0,
                capacity: 4,
            },
        )
        .expect("Card 416 dense fixture graph");
        let dense = graph
            .consts
            .iter()
            .map(|&id| graph.meta(id))
            .filter(|meta| meta.aval.dtype == DType::BF16)
            .map(|meta| {
                let name = meta.name.clone().expect("named Card 416 dense source");
                Card416DenseSource {
                    bytes: bf16_bytes(&name, &tiny_dense(&name, &meta.aval.shape)),
                    name,
                    shape: meta.aval.shape.clone(),
                    dtype: meta.aval.dtype,
                }
            })
            .collect();
        Card416Fixture {
            layout,
            sources: Card416SourceSet { packed, dense },
        }
    }

    impl Card416Fixture {
        pub const fn sources(&self) -> &Card416SourceSet {
            &self.sources
        }

        pub fn trace_prefill(
            &self,
            sequence: usize,
            capacity: usize,
            chunk: usize,
        ) -> Result<Graph, Qwen35GraphError> {
            if sequence == 0 {
                return Err(Qwen35GraphError::EmptyPrefill);
            }
            if capacity < sequence {
                return Err(Qwen35GraphError::PrefillCapacity { sequence, capacity });
            }
            if chunk == 0 {
                return Err(Qwen35GraphError::EmptyChunk);
            }
            trace_qwen38_layout(
                &self.layout,
                Qwen35TraceMode::Prefill {
                    sequence,
                    capacity,
                    chunk,
                },
            )
        }

        pub fn trace_decode(
            &self,
            position: usize,
            capacity: usize,
        ) -> Result<Graph, Qwen35GraphError> {
            if capacity <= position {
                return Err(Qwen35GraphError::DecodeCapacity { position, capacity });
            }
            trace_qwen38_layout(&self.layout, Qwen35TraceMode::Decode { position, capacity })
        }

        pub fn reference_prefill(
            &self,
            tokens: &[u32],
            capacity: usize,
            chunk: usize,
        ) -> Card416ExpectedStep {
            assert!(
                !tokens.is_empty(),
                "Card 416 prefill tokens must be nonempty"
            );
            assert!(
                tokens.len() <= capacity,
                "Card 416 prefill exceeds capacity"
            );
            assert!(chunk > 0, "Card 416 prefill chunk must be nonzero");
            let mut reference = HfReference::card453(self.layout.clone(), capacity);
            let mut logits = Vec::new();
            for (position, &token) in tokens.iter().enumerate() {
                logits = reference.step(
                    i32::try_from(token).expect("Card 416 token fits I32"),
                    position,
                );
            }
            Card416ExpectedStep {
                logits,
                state: reference.state(),
            }
        }

        pub fn reference_cold_decode(
            &self,
            tokens: &[u32],
            capacity: usize,
        ) -> Vec<Card416ExpectedStep> {
            assert!(tokens.len() <= capacity, "Card 416 decode exceeds capacity");
            let mut reference = HfReference::card453(self.layout.clone(), capacity);
            tokens
                .iter()
                .enumerate()
                .map(|(position, &token)| Card416ExpectedStep {
                    logits: reference.step(
                        i32::try_from(token).expect("Card 416 token fits I32"),
                        position,
                    ),
                    state: reference.state(),
                })
                .collect()
        }
    }

    impl Card416SourceSet {
        pub fn packed(&self) -> &[Card416PackedSource] {
            &self.packed
        }

        pub fn dense(&self) -> &[Card416DenseSource] {
            &self.dense
        }
    }

    impl Card416PackedSource {
        pub fn linear_id(&self) -> &str {
            &self.linear_id
        }

        pub const fn descriptor(&self) -> &PackedWeight {
            &self.descriptor
        }

        pub fn weight_bytes(&self) -> &[u8] {
            &self.weight_bytes
        }

        pub fn scale_bytes(&self) -> &[u8] {
            &self.scale_bytes
        }
    }

    impl Card416DenseSource {
        pub fn name(&self) -> &str {
            &self.name
        }

        pub fn shape(&self) -> &[usize] {
            &self.shape
        }

        pub const fn dtype(&self) -> DType {
            self.dtype
        }

        pub fn bytes(&self) -> &[u8] {
            &self.bytes
        }
    }

    fn assert_state_matches_reference(state: &[Value], reference: &HfReference, label: &str) {
        let expected = reference.state();
        assert_eq!(state.len(), expected.len(), "{label} state count");
        for (index, (actual, expected)) in state.iter().zip(&expected).enumerate() {
            assert_matches_reference(actual, expected, &format!("{label} state {index}"));
        }
    }

    /// The tiny decode graphs, and a chunked prefill with a padded tail chunk, must reproduce the HF
    /// reference logits and every carried state tensor.
    #[test]
    fn qwen35_tiny_graphs_match_independent_hf_reference() -> Result<(), Qwen35GraphError> {
        let layout = tiny_layout();
        let owners = tiny_owners(&layout)?;
        let mut reference = HfReference::new(layout, TINY_CAPACITY);

        let mut carried: Option<Vec<Value>> = None;
        let mut reference_logits = Vec::new();
        for (position, &token) in TINY_PROMPT.iter().enumerate() {
            reference_logits = reference.step(token, position);
            let graph = tiny_graph(Qwen35TraceMode::Decode {
                position,
                capacity: TINY_CAPACITY,
            })?;
            let inputs = tiny_bindings(
                &graph,
                &owners,
                &[token],
                Some(position),
                carried.as_deref(),
            );
            let step = eval(&graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("tiny decode graph evaluates");
            let (logits, state) = (step.output, step.state);
            let label = format!("decode position {position}");
            assert_matches_reference(&logits, &reference_logits, &format!("{label} logits"));
            assert_state_matches_reference(&state, &reference, &label);
            carried = Some(state);
        }

        let prefill = tiny_graph(Qwen35TraceMode::Prefill {
            sequence: TINY_PROMPT.len(),
            capacity: TINY_CAPACITY,
            chunk: 2,
        })?;
        let inputs = tiny_bindings(&prefill, &owners, &TINY_PROMPT, None, None);
        let step = eval(&prefill, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("tiny prefill graph evaluates");
        let (logits, state) = (step.output, step.state);
        assert_matches_reference(&logits, &reference_logits, "prefill logits");
        assert_state_matches_reference(&state, &reference, "prefill");
        Ok(())
    }

    /// One decode step on the CPU evaluator: the primary output as a dense tensor and the carried state.
    fn step_with_state(
        graph: &Graph,
        inputs: &HashMap<ValueId, Value>,
    ) -> (HostTensor, Vec<Value>) {
        let step = eval(graph, inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .expect("the tiny decode graph must evaluate on the CPU evaluator");
        let (primary, state) = (step.output, step.state);
        let Value::Host(tensor) = primary else {
            panic!("tiny decode graph's primary output must be dense")
        };
        (tensor, state)
    }

    /// Card 430: state from position 0 is threaded into position 1 and checked against the independent
    /// HF reference, plus a fresh-state control proving position 1 depends on the carried state.
    #[cfg(test)]
    #[test]
    fn card430_decode_state_threads_across_steps() -> Result<(), Qwen35GraphError> {
        let layout = tiny_layout();
        let owners = tiny_owners(&layout)?;
        let mut reference = HfReference::new(layout, TINY_CAPACITY);

        let position0 = 0usize;
        let token0 = TINY_PROMPT[0];
        let graph0 = tiny_graph(Qwen35TraceMode::Decode {
            position: position0,
            capacity: TINY_CAPACITY,
        })?;
        let inputs0 = tiny_bindings(&graph0, &owners, &[token0], Some(position0), None);
        let (_tensor0, state0) = step_with_state(&graph0, &inputs0);
        let _ = reference.step(token0, position0);

        let position1 = 1usize;
        let token1 = TINY_PROMPT[position1];
        let reference_logits1 = reference.step(token1, position1);
        let graph1 = tiny_graph(Qwen35TraceMode::Decode {
            position: position1,
            capacity: TINY_CAPACITY,
        })?;
        let threaded_inputs1 = tiny_bindings(
            &graph1,
            &owners,
            &[token1],
            Some(position1),
            Some(state0.as_slice()),
        );
        let (threaded_tensor1, _threaded_state1) = step_with_state(&graph1, &threaded_inputs1);
        assert_matches_reference(
            &Value::Host(threaded_tensor1.clone()),
            &reference_logits1,
            "position 1 threaded through the carried state",
        );

        // Control: a fresh (zero) state at position 1 must diverge from the threaded result, otherwise
        // dropping the carried state would pass the reference match unnoticed.
        let fresh_inputs1 = tiny_bindings(&graph1, &owners, &[token1], Some(position1), None);
        let (fresh_tensor1, _fresh_state1) = step_with_state(&graph1, &fresh_inputs1);
        assert_ne!(
            threaded_tensor1, fresh_tensor1,
            "position 1's output must depend on position 0's carried state, or this row cannot fail"
        );

        Ok(())
    }

    fn named_value(graph: &Graph, name: &str) -> ValueId {
        graph
            .values
            .iter()
            .position(|value| value.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing graph value {name}"))
    }

    fn producer(graph: &Graph, value: ValueId) -> &Eqn {
        graph
            .eqns
            .iter()
            .find(|eqn| eqn.out == value)
            .unwrap_or_else(|| panic!("missing producer for v{value}"))
    }

    fn value_inputs(eqn: &Eqn) -> Vec<ValueId> {
        eqn.inputs
            .iter()
            .filter_map(|operand| match operand {
                Operand::Value(value) => Some(*value),
                Operand::Lit(_) => None,
            })
            .collect()
    }

    /// Output of one packed projection: `PackedDequant -> Transpose -> MatMul(x, weight)`.
    fn packed_projection_output(graph: &Graph, linear_id: &str) -> ValueId {
        let source = named_value(graph, PackedSourceName::weight(linear_id).as_str());
        let find = |predicate: &dyn Fn(&Eqn) -> bool| {
            graph
                .eqns
                .iter()
                .find(|eqn| predicate(eqn))
                .unwrap_or_else(|| panic!("missing packed projection equation for {linear_id}"))
                .out
        };
        let dequant = find(&|eqn: &Eqn| {
            matches!(eqn.op, OpKind::PackedDequant { .. })
                && value_inputs(eqn).first() == Some(&source)
        });
        let weight = find(&|eqn: &Eqn| {
            matches!(eqn.op, OpKind::Transpose { .. }) && value_inputs(eqn) == [dequant]
        });
        find(&|eqn: &Eqn| eqn.op == OpKind::MatMul && value_inputs(eqn).get(1) == Some(&weight))
    }

    /// Tie every state output to the equation its own layer core emits for it, by operand position.
    fn assert_state_outputs_bind_own_layer(
        graph: &Graph,
        layout: &Qwen35Layout,
        mode: Qwen35TraceMode,
    ) {
        assert_eq!(graph.state.len(), 2 * layout.layer_types.len());
        for (layer, (&kind, pairs)) in layout
            .layer_types
            .iter()
            .zip(graph.state.chunks_exact(2))
            .enumerate()
        {
            let root = format!("model.language_model.layers.{layer}");
            let names = match kind {
                Qwen35LayerKind::FullAttention => ["self_attn.k_cache", "self_attn.v_cache"],
                Qwen35LayerKind::LinearAttention => {
                    ["linear_attn.conv_cache", "linear_attn.recurrent_state"]
                }
            };
            for (&(state_in, _), suffix) in pairs.iter().zip(names) {
                let name = format!("{root}.{suffix}");
                assert_eq!(graph.meta(state_in).name.as_deref(), Some(name.as_str()));
            }
            let [(first_in, first_out), (second_in, second_out)] = [pairs[0], pairs[1]];
            match kind {
                Qwen35LayerKind::FullAttention => {
                    for (state_in, state_out) in [(first_in, first_out), (second_in, second_out)] {
                        let update = producer(graph, state_out);
                        assert_eq!(update.op, OpKind::DynamicUpdateSlice { axis: 2 });
                        assert_eq!(
                            value_inputs(update).first(),
                            Some(&state_in),
                            "layer {layer} cache output must update its own cache input"
                        );
                    }
                }
                Qwen35LayerKind::LinearAttention => {
                    let qkv =
                        packed_projection_output(graph, &format!("{root}.linear_attn.in_proj_qkv"));
                    let window = producer(graph, first_out);
                    let concat = producer(graph, value_inputs(window)[0]);
                    assert_eq!(concat.op, OpKind::Concat { axis: 1 });
                    match mode {
                        Qwen35TraceMode::Decode { .. } => {
                            let expected = OpKind::Slice {
                                axis: 1,
                                start: 1,
                                end: layout.conv_k,
                            };
                            assert_eq!(window.op, expected);
                            assert_eq!(
                                value_inputs(concat),
                                [first_in, qkv],
                                "layer {layer} conv window must be its own cache plus its own QKV"
                            );
                        }
                        Qwen35TraceMode::Prefill { sequence, .. } => {
                            let expected = OpKind::Slice {
                                axis: 1,
                                start: sequence,
                                end: sequence + layout.conv_k - 1,
                            };
                            assert_eq!(window.op, expected);
                            assert_eq!(
                                value_inputs(concat)[1],
                                qkv,
                                "layer {layer} conv cache must hold its own QKV"
                            );
                        }
                    }
                    // Follow `state = state * decay + update` back to the recurrent state input.
                    let mut carried = second_out;
                    while graph.meta(carried).storage != Storage::State {
                        let update = producer(graph, carried);
                        assert_eq!(update.op, OpKind::Binary(BinOp::Add), "layer {layer}");
                        let decayed = producer(graph, value_inputs(update)[0]);
                        assert_eq!(decayed.op, OpKind::Binary(BinOp::Mul), "layer {layer}");
                        carried = value_inputs(decayed)[0];
                    }
                    assert_eq!(
                        carried, second_in,
                        "layer {layer} recurrent state output must carry its own input"
                    );
                }
            }
        }
    }

    #[test]
    fn qwen35_state_outputs_bind_their_own_layer_core() -> Result<(), Qwen35GraphError> {
        let layout = tiny_layout();
        for mode in [
            Qwen35TraceMode::Decode {
                position: 1,
                capacity: TINY_CAPACITY,
            },
            Qwen35TraceMode::Prefill {
                sequence: 3,
                capacity: TINY_CAPACITY,
                chunk: 2,
            },
        ] {
            assert_state_outputs_bind_own_layer(&tiny_graph(mode)?, &layout, mode);
        }
        let config = Qwen35TextConfig::pinned();
        let production = trace_qwen38_text_decode(&config, 1, 2)?;
        assert_state_outputs_bind_own_layer(
            &production,
            &Qwen35Layout::from_config(&config)?,
            Qwen35TraceMode::Decode {
                position: 1,
                capacity: 2,
            },
        );
        Ok(())
    }

    fn official_dense_sources() -> HashMap<String, Vec<usize>> {
        let mut expected = HashMap::new();
        expected.insert(
            "model.language_model.embed_tokens.weight".into(),
            vec![248_320, 5_120],
        );
        expected.insert("model.language_model.norm.weight".into(), vec![5_120]);
        expected.insert("lm_head.weight".into(), vec![248_320, 5_120]);
        const GDN: [(&str, &[usize]); 6] = [
            ("linear_attn.A_log", &[48]),
            ("linear_attn.conv1d.weight", &[10_240, 1, 4]),
            ("linear_attn.dt_bias", &[48]),
            ("linear_attn.in_proj_a.weight", &[48, 5_120]),
            ("linear_attn.in_proj_b.weight", &[48, 5_120]),
            ("linear_attn.norm.weight", &[128]),
        ];
        const ATTENTION: [(&str, &[usize]); 2] = [
            ("self_attn.q_norm.weight", &[256]),
            ("self_attn.k_norm.weight", &[256]),
        ];
        for layer in 0..64usize {
            let root = format!("model.language_model.layers.{layer}");
            expected.insert(format!("{root}.input_layernorm.weight"), vec![5_120]);
            expected.insert(
                format!("{root}.post_attention_layernorm.weight"),
                vec![5_120],
            );
            let local = if (layer + 1).is_multiple_of(4) {
                ATTENTION.as_slice()
            } else {
                GDN.as_slice()
            };
            for &(suffix, shape) in local {
                expected.insert(format!("{root}.{suffix}"), shape.to_vec());
            }
        }
        expected
    }

    #[test]
    fn qwen35_packed_inventory_has_exact_400_rows() {
        let config = Qwen35TextConfig::pinned();
        let rows = qwen38_packed_linear_specs(&config).expect("packed rows");
        assert_eq!(rows.len(), 400);
        assert_eq!(
            rows.iter()
                .filter(|row| {
                    matches!(
                        row.role,
                        Qwen35PackedRole::GdnQkv
                            | Qwen35PackedRole::GdnZ
                            | Qwen35PackedRole::GdnOut
                    )
                })
                .count(),
            144
        );
        assert_eq!(
            rows.iter()
                .filter(|row| {
                    matches!(
                        row.role,
                        Qwen35PackedRole::AttentionQ
                            | Qwen35PackedRole::AttentionK
                            | Qwen35PackedRole::AttentionV
                            | Qwen35PackedRole::AttentionOut
                    )
                })
                .count(),
            64
        );
        assert_eq!(
            rows.iter()
                .filter(|row| {
                    matches!(
                        row.role,
                        Qwen35PackedRole::MlpGate
                            | Qwen35PackedRole::MlpUp
                            | Qwen35PackedRole::MlpDown
                    )
                })
                .count(),
            192
        );
        assert!(
            rows.iter()
                .all(|row| row.descriptor.format() == PACKED_FORMAT)
        );
    }

    #[test]
    fn qwen35_packed_inventory_names_match_pinned_index_and_exclude_mtp() {
        let config = Qwen35TextConfig::pinned();
        let rows = qwen38_packed_linear_specs(&config).expect("packed rows");
        let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../poot-llm/tests/fixtures/qwen38_27b_model_safetensors_index.json"
        ))
        .expect("pinned index fixture parses");
        let weight_map = fixture["weight_map"]
            .as_object()
            .expect("weight_map object");
        for row in &rows {
            assert!(
                weight_map.contains_key(&format!("{}.weight", row.linear_id)),
                "missing indexed weight for {}",
                row.linear_id
            );
            assert!(
                weight_map.contains_key(&format!("{}.weight_scale_inv", row.linear_id)),
                "missing indexed scale for {}",
                row.linear_id
            );
        }
        let indexed_scales = weight_map
            .keys()
            .filter(|name| name.ends_with(".weight_scale_inv"))
            .count();
        let mtp_scales = weight_map
            .keys()
            .filter(|name| name.starts_with("mtp.") && name.ends_with(".weight_scale_inv"))
            .count();
        assert_eq!(indexed_scales, 407);
        assert_eq!(mtp_scales, 7);
        assert_eq!(indexed_scales - mtp_scales, rows.len());
        assert!(rows.iter().all(|row| !row.linear_id.starts_with("mtp.")));

        let unique = rows
            .iter()
            .map(|row| row.linear_id.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(unique.len(), 400);
    }

    fn bf16_dense_rows(graph: &Graph) -> Vec<(String, Vec<usize>)> {
        graph
            .consts
            .iter()
            .filter_map(|&id| {
                let meta = graph.meta(id);
                let name = meta.name.as_deref()?;
                (meta.storage == Storage::Const && meta.aval.dtype == DType::BF16)
                    .then(|| (name.to_string(), meta.aval.shape.clone()))
            })
            .collect()
    }

    /// The table the loader builds its dense manifest from must name exactly the BF16 constants the
    /// traced graph declares. `official_dense_sources` checks the graph independently; this ties the
    /// published table to it.
    #[test]
    fn qwen35_dense_source_specs_match_the_traced_graph() -> Result<(), Qwen35GraphError> {
        let config = Qwen35TextConfig::pinned();
        let graph = trace_qwen38_text_decode(&config, 0, 2)?;
        let specs = qwen38_dense_source_specs(&config);
        let graph_rows = bf16_dense_rows(&graph);
        assert_eq!(specs.len(), graph_rows.len());
        let unique = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(unique.len(), specs.len(), "dense source names repeat");
        assert_eq!(
            specs
                .into_iter()
                .map(|spec| (spec.name, spec.shape))
                .collect::<HashMap<_, _>>(),
            graph_rows.into_iter().collect::<HashMap<_, _>>()
        );
        Ok(())
    }

    #[test]
    fn qwen35_dense_inventory_matches_independent_official_table() -> Result<(), Qwen35GraphError> {
        let graph = trace_qwen38_text_decode(&Qwen35TextConfig::pinned(), 0, 2)?;
        let expected = official_dense_sources();
        assert_eq!(expected.len(), 451);
        let raw_rows = bf16_dense_rows(&graph);
        assert_eq!(
            raw_rows.len(),
            expected.len(),
            "raw BF16 dense source rows must match the official count before map construction"
        );
        let raw_names = raw_rows
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(
            raw_names.len(),
            expected.len(),
            "raw BF16 dense source names must be unique before map construction"
        );
        let actual = raw_rows.into_iter().collect::<HashMap<_, _>>();
        assert_eq!(actual, expected);
        assert!(actual.keys().all(|name| !name.starts_with("mtp.")));

        let fixture: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../poot-llm/tests/fixtures/qwen38_27b_model_safetensors_index.json"
        ))
        .expect("pinned index fixture parses");
        let weight_map = fixture["weight_map"]
            .as_object()
            .expect("weight_map object");
        for name in expected.keys() {
            assert!(
                weight_map.contains_key(name),
                "missing official dense row {name}"
            );
        }

        let f32_sources = graph
            .consts
            .iter()
            .filter_map(|&id| {
                let meta = graph.meta(id);
                (meta.storage == Storage::Const && meta.aval.dtype == DType::F32).then(|| {
                    (
                        meta.name.as_deref().unwrap_or_default(),
                        meta.aval.shape.as_slice(),
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            f32_sources,
            [
                ("qwen38_27b.rope.cos", [262_144, 64].as_slice()),
                ("qwen38_27b.rope.sin", [262_144, 64].as_slice()),
            ]
        );

        let prefill = trace_qwen38_text_prefill(&Qwen35TextConfig::pinned(), 2, 4, 2)?;
        let prefill_f32 = prefill
            .consts
            .iter()
            .filter_map(|&id| {
                let meta = prefill.meta(id);
                (meta.storage == Storage::Const && meta.aval.dtype == DType::F32).then(|| {
                    (
                        meta.name.as_deref().unwrap_or_default().to_string(),
                        meta.aval.shape.clone(),
                    )
                })
            })
            .collect::<HashMap<_, _>>();
        assert_eq!(
            prefill_f32,
            HashMap::from([
                ("qwen38_27b.rope.cos".into(), vec![262_144, 64]),
                ("qwen38_27b.rope.sin".into(), vec![262_144, 64]),
            ])
        );

        for layer in 0..64usize {
            let pair = layer * 2;
            let root = format!("model.language_model.layers.{layer}");
            let expected = if (layer + 1).is_multiple_of(4) {
                [
                    (format!("{root}.self_attn.k_cache"), vec![1, 4, 2, 256]),
                    (format!("{root}.self_attn.v_cache"), vec![1, 4, 2, 256]),
                ]
            } else {
                [
                    (format!("{root}.linear_attn.conv_cache"), vec![1, 3, 10_240]),
                    (
                        format!("{root}.linear_attn.recurrent_state"),
                        vec![1, 48, 128, 128],
                    ),
                ]
            };
            for (offset, (name, shape)) in expected.iter().enumerate() {
                let state_in = graph.state[pair + offset].0;
                assert_eq!(graph.meta(state_in).name.as_deref(), Some(name.as_str()));
                assert_eq!(graph.aval(state_in).shape, *shape);
            }
        }
        Ok(())
    }

    #[test]
    fn qwen35_schedule_drives_distinct_layer_graphs() -> Result<(), Qwen35GraphError> {
        let config = Qwen35TextConfig::pinned();
        let prefill = trace_qwen38_text_prefill(&config, 2, 4, 2)?;
        let decode = trace_qwen38_text_decode(&config, 1, 4)?;
        prefill.validate().expect("prefill graph validates");
        decode.validate().expect("decode graph validates");
        assert_eq!(
            prefill.aval(prefill.output),
            &TensorType::f32(vec![1, 1, config.vocab])
        );
        assert_eq!(
            decode.aval(decode.output),
            &TensorType::f32(vec![1, 1, config.vocab])
        );
        assert_eq!(prefill.state.len(), 128);
        assert_eq!(decode.state.len(), 128);
        assert_eq!(
            prefill
                .eqns
                .iter()
                .filter(|eqn| matches!(eqn.op, OpKind::PackedDequant { .. }))
                .count(),
            400
        );
        assert_eq!(
            decode
                .eqns
                .iter()
                .filter(|eqn| matches!(eqn.op, OpKind::PackedDequant { .. }))
                .count(),
            400
        );
        assert!(prefill.values.iter().any(|value| {
            value.name.as_deref() == Some("model.language_model.layers.0.linear_attn.A_log")
        }));
        assert!(!prefill.values.iter().any(|value| {
            value.name.as_deref() == Some("model.language_model.layers.0.self_attn.q_norm.weight")
        }));
        assert!(prefill.values.iter().any(|value| {
            value.name.as_deref() == Some("model.language_model.layers.3.self_attn.q_norm.weight")
        }));
        assert!(!prefill.values.iter().any(|value| {
            value.name.as_deref() == Some("model.language_model.layers.3.linear_attn.A_log")
        }));
        assert!(prefill.inputs.iter().all(|&id| {
            let name = prefill.meta(id).name.as_deref().unwrap_or_default();
            !name.contains("model.visual") && !name.contains(".mtp.")
        }));

        let mut changed = config;
        changed.layer_types[3] = Qwen35LayerKind::LinearAttention;
        assert!(matches!(
            changed.validate_exact(),
            Err(Qwen35GraphError::LayerKind { layer: 3, .. })
        ));
        Ok(())
    }

    #[test]
    fn qwen35_graph_sources_stay_canonical_and_source_typed() -> Result<(), Qwen35GraphError> {
        let config = Qwen35TextConfig::pinned();
        let graph = trace_qwen38_text_decode(&config, 0, 2)?;
        for &id in &graph.consts {
            let meta = graph.meta(id);
            let name = meta.name.as_deref().expect("named source");
            if PackedSourceName::parse(name).is_some() {
                assert_eq!(meta.aval.dtype, DType::I8, "{name}");
            } else if (name.starts_with("model.language_model") || name == "lm_head.weight")
                && meta.storage == Storage::Const
            {
                assert_eq!(meta.aval.dtype, DType::BF16, "{name}");
            }
        }
        for role in [
            SourceRole::Planar(OperandRole::Codes),
            SourceRole::Planar(OperandRole::Scale),
        ] {
            assert_eq!(
                graph
                    .values
                    .iter()
                    .filter(|value| value
                        .name
                        .as_deref()
                        .and_then(PackedSourceName::parse)
                        .is_some_and(|source| source.role() == role))
                    .count(),
                400,
                "{role:?}"
            );
        }
        // The embedding gathers BF16 rows and the LM head multiplies the BF16 weight directly, so no graph
        // value is an F32 copy of either `[vocab, hidden]` matrix.
        let vocab_matrices = [[config.vocab, config.hidden], [config.hidden, config.vocab]];
        for value in &graph.values {
            assert!(
                value.aval.dtype != DType::F32
                    || !vocab_matrices
                        .iter()
                        .any(|shape| value.aval.shape == *shape),
                "F32 vocabulary matrix {:?}",
                value.name
            );
        }
        Ok(())
    }
}

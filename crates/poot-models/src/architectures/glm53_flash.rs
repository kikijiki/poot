//! Device-free graph slices for `zai-org/GLM-5.3-Flash`.
//!
//! Cards 366 and 367 provide the two one-token attention compositions. Card 368's text module assembles their
//! exact 45-layer schedule with model-local source ownership. None of these graph surfaces claims a runner,
//! portable backend binding, vision tower, MTP head, or public model capability.

#[cfg(test)]
use poot_tensor::DType;
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use poot_graph_ir::ops::{rmsnorm, sigmoid, silu};
#[cfg(test)]
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, StateRole, TensorType, Traced, UnOp};
#[cfg(test)]
use poot_graph_ir::{Graph, Slot};
#[cfg(test)]
use poot_load::packed_safetensors::{ExactSourceKind, ExactSourceOwner, MixedLoadResult};

// Cards 366-368's attention/text-tracing/source-ownership modules (see the module doc above: no
// runner or binding claim yet). Every re-exported item has a real caller only from this crate's own
// unit tests today; Card 569 (GLM's per-family Runner transaction) is what wires a production caller
// in. Gated as a whole rather than item-by-item (card 540b review): an unrelated rename elsewhere
// broke an accidental identifier collision that had been hiding `text`'s dead pub surface from
// `dead-pub`'s name-based scan (`from_loaded`, deleted; see git history), and once `text` lost that
// alibi, `dsa`'s own config/role types turned out to have no reader left outside it either.
#[cfg(test)]
mod dsa;
#[cfg(test)]
pub use dsa::*;
#[cfg(test)]
mod text;
#[cfg(test)]
pub use text::*;

// Only this crate's own unit tests (across glm53_flash.rs, dsa.rs and the now-gated text.rs) read
// this today; the real revision check moves to `poot_load`'s admission path when Card 569 wires a
// production caller in (card 540b review: this and the other items below lost their sole non-test
// reader when text.rs was gated).
#[cfg(test)]
pub const GLM53_FLASH_REVISION: &str = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a";

#[cfg(test)]
#[derive(Debug, thiserror::Error)]
pub(crate) enum Glm53FlashTraceError {
    #[error("GLM-5.3-Flash KDA needs revision {expected}, got {actual}")]
    KdaRevision {
        expected: &'static str,
        actual: String,
    },
    #[error("GLM-5.3-Flash KDA source prefix is invalid: {prefix}")]
    KdaSourcePrefix { prefix: String },
    #[error("GLM-5.3-Flash KDA dimension {field} must be {requirement}, got {actual}")]
    KdaDimension {
        field: &'static str,
        requirement: String,
        actual: usize,
    },
    #[error("GLM-5.3-Flash KDA shape arithmetic overflowed while computing {field}")]
    KdaShapeOverflow { field: &'static str },
    #[error("GLM-5.3-Flash KDA {role} has dtype {actual}, expected {expected}")]
    KdaDtype {
        role: &'static str,
        expected: DType,
        actual: DType,
    },
    #[error("GLM-5.3-Flash KDA {role} has shape {actual:?}, expected {expected:?}")]
    KdaShape {
        role: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("GLM-5.3-Flash KDA does not implement {scope:?} scope")]
    KdaUnsupportedScope { scope: Glm5NextKdaScope },
    #[error("GLM-5.3-Flash KDA owner table repeats role {role:?}")]
    KdaDuplicateRole { role: Glm5NextKdaRole },
    #[error("GLM-5.3-Flash KDA owner table is missing {name}")]
    KdaMissingSource { name: String },
    #[error("GLM-5.3-Flash KDA source {name} has kind {actual:?}, expected {expected:?}")]
    KdaSourceKind {
        name: String,
        expected: ExactSourceKind,
        actual: ExactSourceKind,
    },
    #[error("GLM-5.3-Flash KDA source {name} has dtype {actual}, expected {expected}")]
    KdaSourceDtype {
        name: String,
        expected: &'static str,
        actual: String,
    },
    #[error("GLM-5.3-Flash KDA source {name} has shape {actual:?}, expected {expected:?}")]
    KdaSourceShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("GLM-5.3-Flash DSA needs revision {expected}, got {actual}")]
    DsaRevision {
        expected: &'static str,
        actual: String,
    },
    #[error("GLM-5.3-Flash DSA source prefix is invalid: {prefix}")]
    DsaSourcePrefix { prefix: String },
    #[error("GLM-5.3-Flash DSA dimension {field} must be {requirement}, got {actual}")]
    DsaDimension {
        field: &'static str,
        requirement: String,
        actual: usize,
    },
    #[error("GLM-5.3-Flash DSA shape arithmetic overflowed while computing {field}")]
    DsaShapeOverflow { field: &'static str },
    #[error("GLM-5.3-Flash DSA {role} has dtype {actual}, expected {expected}")]
    DsaDtype {
        role: &'static str,
        expected: DType,
        actual: DType,
    },
    #[error("GLM-5.3-Flash DSA {role} has shape {actual:?}, expected {expected:?}")]
    DsaShape {
        role: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("GLM-5.3-Flash DSA does not implement {scope:?} scope")]
    DsaUnsupportedScope { scope: Glm5NextDsaScope },
    #[error(
        "GLM-5.3-Flash DSA requires NoPE, got rope width {rope_width} and mla_use_nope={mla_use_nope}"
    )]
    DsaRopeContract {
        rope_width: usize,
        mla_use_nope: bool,
    },
    #[error(
        "GLM-5.3-Flash DSA needs a full compressed tail indexer, got {kind:?}, compress={compress}, always_select_tail={always_select_tail}"
    )]
    DsaIndexerContract {
        kind: Glm5NextIndexerKind,
        compress: bool,
        always_select_tail: bool,
    },
    #[error(
        "GLM-5.3-Flash DSA pairwise semantic capacity {capacity} exceeds bounded limit {limit}"
    )]
    DsaPairwiseCapacity { capacity: usize, limit: usize },
    #[error("GLM-5.3-Flash DSA owner table repeats role {role:?}")]
    DsaDuplicateRole { role: Glm5NextDsaRole },
    #[error("GLM-5.3-Flash DSA packed owner for {role:?} is deferred to Card 368")]
    DsaPackedOwnerDeferred { role: Glm5NextDsaRole },
    #[error("GLM-5.3-Flash DSA owner table is missing {name}")]
    DsaMissingSource { name: String },
    #[error("GLM-5.3-Flash DSA source {name} has kind {actual:?}, expected {expected:?}")]
    DsaSourceKind {
        name: String,
        expected: ExactSourceKind,
        actual: ExactSourceKind,
    },
    #[error("GLM-5.3-Flash DSA source {name} has dtype {actual}, expected {expected}")]
    DsaSourceDtype {
        name: String,
        expected: &'static str,
        actual: String,
    },
    #[error("GLM-5.3-Flash DSA source {name} has shape {actual:?}, expected {expected:?}")]
    DsaSourceShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error(transparent)]
    TextConfig(#[from] poot_load::glm53_flash::Glm53FlashMetadataError),
    #[error("GLM-5.3-Flash text graph does not implement {scope:?} input")]
    TextUnsupportedScope { scope: Glm5NextTextScope },
    #[error(
        "GLM-5.3-Flash KDA config {derived:?} from the text config differs from the pinned block config {:?}",
        Glm5NextKdaConfig::exact()
    )]
    TextKdaConfig { derived: Glm5NextKdaConfig },
    #[error(
        "GLM-5.3-Flash DSA config {derived:?} from the text config differs from the pinned block config {:?}",
        Glm5NextDsaConfig::exact()
    )]
    TextDsaConfig { derived: Glm5NextDsaConfig },
    #[error("GLM-5.3-Flash dense source {name} has unsupported dtype {dtype}")]
    TextDenseSourceDtype { name: String, dtype: DType },
    #[error("GLM-5.3-Flash text source plan repeats {kind} source {name}")]
    TextDuplicateSource {
        kind: Glm5NextTextSourceKind,
        name: String,
    },
    #[error("GLM-5.3-Flash text source plan has no dense source {name}")]
    TextMissingDenseSource { name: String },
    #[error("GLM-5.3-Flash text source plan has no layer {layer} packed source {role:?}")]
    TextMissingPackedSource {
        layer: usize,
        role: Glm5NextPackedTextRole,
    },
    #[error("GLM-5.3-Flash DSA packed source {name} has no .weight suffix")]
    TextPackedSourceName { name: String },
    #[error("GLM-5.3-Flash DSA packed source {name} has shape {shape:?}, expected rank two")]
    TextPackedSourceRank { name: String, shape: Vec<usize> },
    #[error("GLM-5.3-Flash packed source {linear_id} has invalid logical shape {logical_shape:?}")]
    TextPackedDescriptor {
        linear_id: String,
        logical_shape: [usize; 2],
    },
    #[error("GLM-5.3-Flash text inventory has {actual} {field}, expected {expected}")]
    TextInventoryCount {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("GLM-5.3-Flash text shape arithmetic overflowed while computing {field}")]
    TextShapeOverflow { field: &'static str },
    #[error("GLM-5.3-Flash router {field} must be {requirement}, got {actual}")]
    TextRouterConfig {
        field: &'static str,
        requirement: &'static str,
        actual: usize,
    },
    #[error("GLM-5.3-Flash router {role} has unexpected type {actual}")]
    TextRouterInput {
        role: &'static str,
        actual: TensorType,
    },
    #[error(transparent)]
    TextGraph(#[from] poot_graph_ir::BuilderAppendError),
}

/// One fixed decode-state tensor that an attention block reads and writes.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Glm5NextStateSpec {
    /// Name suffix below the block's `self_attn` source prefix.
    pub suffix: &'static str,
    pub ty: TensorType,
}

/// Apply a PyTorch `[out, in]` dense checkpoint weight to `x` in F32.
#[cfg(test)]
fn dense_source_linear(b: &Builder, x: Traced, weight: Traced) -> Traced {
    b.matmul(x, b.transpose(b.cast(weight, DType::F32), vec![1, 0]))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum Glm5NextKdaScope {
    Kda,
    ImageVideo,
    Mtp,
    Dsa,
    Mhc,
    Ffn,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm5NextKdaConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub conv_width: usize,
    pub gate_rank: usize,
    /// Lower bound of the log forget gate: `forget = gate_lower_bound * sigmoid(A * f)`.
    pub gate_lower_bound: f32,
}

#[cfg(test)]
impl Glm5NextKdaConfig {
    pub const fn exact() -> Self {
        Self {
            hidden_size: 4_096,
            num_heads: 64,
            head_dim: 128,
            conv_width: 4,
            gate_rank: 128,
            gate_lower_bound: -5.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg(test)]
#[repr(usize)]
pub(crate) enum Glm5NextKdaRole {
    QProj,
    KProj,
    VProj,
    QConv1d,
    KConv1d,
    VConv1d,
    FAProj,
    FBProj,
    BProj,
    GAProj,
    GBProj,
    ALog,
    DtBias,
    ONorm,
    OProj,
}

#[derive(Clone, Copy)]
#[cfg(test)]
enum KdaRoleShape {
    QkvProjection,
    Convolution,
    RankIn,
    RankOut,
    HeadProjection,
    Heads,
    Qkv,
    HeadDim,
    OutputProjection,
}

#[cfg(test)]
const KDA_ROLE_ROWS: [(Glm5NextKdaRole, &str, DType, KdaRoleShape); 15] = [
    (
        Glm5NextKdaRole::QProj,
        "q_proj.weight",
        DType::BF16,
        KdaRoleShape::QkvProjection,
    ),
    (
        Glm5NextKdaRole::KProj,
        "k_proj.weight",
        DType::BF16,
        KdaRoleShape::QkvProjection,
    ),
    (
        Glm5NextKdaRole::VProj,
        "v_proj.weight",
        DType::BF16,
        KdaRoleShape::QkvProjection,
    ),
    (
        Glm5NextKdaRole::QConv1d,
        "q_conv1d.weight",
        DType::BF16,
        KdaRoleShape::Convolution,
    ),
    (
        Glm5NextKdaRole::KConv1d,
        "k_conv1d.weight",
        DType::BF16,
        KdaRoleShape::Convolution,
    ),
    (
        Glm5NextKdaRole::VConv1d,
        "v_conv1d.weight",
        DType::BF16,
        KdaRoleShape::Convolution,
    ),
    (
        Glm5NextKdaRole::FAProj,
        "f_a_proj.weight",
        DType::BF16,
        KdaRoleShape::RankIn,
    ),
    (
        Glm5NextKdaRole::FBProj,
        "f_b_proj.weight",
        DType::BF16,
        KdaRoleShape::RankOut,
    ),
    (
        Glm5NextKdaRole::BProj,
        "b_proj.weight",
        DType::BF16,
        KdaRoleShape::HeadProjection,
    ),
    (
        Glm5NextKdaRole::GAProj,
        "g_a_proj.weight",
        DType::BF16,
        KdaRoleShape::RankIn,
    ),
    (
        Glm5NextKdaRole::GBProj,
        "g_b_proj.weight",
        DType::BF16,
        KdaRoleShape::RankOut,
    ),
    (
        Glm5NextKdaRole::ALog,
        "A_log",
        DType::F32,
        KdaRoleShape::Heads,
    ),
    (
        Glm5NextKdaRole::DtBias,
        "dt_bias",
        DType::F32,
        KdaRoleShape::Qkv,
    ),
    (
        Glm5NextKdaRole::ONorm,
        "o_norm.weight",
        DType::BF16,
        KdaRoleShape::HeadDim,
    ),
    (
        Glm5NextKdaRole::OProj,
        "o_proj.weight",
        DType::BF16,
        KdaRoleShape::OutputProjection,
    ),
];

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct Glm5NextKdaRoleSpec {
    pub role: Glm5NextKdaRole,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
#[cfg(test)]
pub(crate) struct Glm5NextKdaInputs {
    pub x: Traced,
    pub validity: Option<Traced>,
    pub conv_state: Traced,
    pub recurrent_state: Traced,
}

#[derive(Clone, Copy, Debug)]
#[cfg(test)]
pub(crate) struct Glm5NextKdaOutput {
    pub y: Traced,
    pub conv_state: Traced,
    pub recurrent_state: Traced,
}

#[derive(Clone, Debug)]
#[cfg(test)]
pub(crate) struct Glm5NextKdaOwnerBinding {
    role: Glm5NextKdaRole,
    name: String,
    owner: Arc<ExactSourceOwner>,
}

#[cfg(test)]
impl Glm5NextKdaOwnerBinding {
    pub const fn role(&self) -> Glm5NextKdaRole {
        self.role
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn owner(&self) -> &Arc<ExactSourceOwner> {
        &self.owner
    }
}

#[derive(Clone, Copy)]
#[cfg(test)]
struct KdaDimensions {
    qkv: usize,
    conv_channels: usize,
}

#[cfg(test)]
fn preflight_kda_config(
    revision: &str,
    prefix: &str,
    scope: Glm5NextKdaScope,
    cfg: Glm5NextKdaConfig,
) -> Result<KdaDimensions, Glm53FlashTraceError> {
    if revision != GLM53_FLASH_REVISION {
        return Err(Glm53FlashTraceError::KdaRevision {
            expected: GLM53_FLASH_REVISION,
            actual: revision.to_string(),
        });
    }
    if scope != Glm5NextKdaScope::Kda {
        return Err(Glm53FlashTraceError::KdaUnsupportedScope { scope });
    }
    let valid_prefix = prefix
        .strip_prefix("model.language_model.layers.")
        .and_then(|suffix| suffix.strip_suffix(".self_attn"))
        .is_some_and(|layer| {
            layer == "{layer}"
                || (!layer.is_empty() && layer.bytes().all(|byte| byte.is_ascii_digit()))
        });
    if !valid_prefix {
        return Err(Glm53FlashTraceError::KdaSourcePrefix {
            prefix: prefix.to_string(),
        });
    }
    kda_dimensions(cfg)
}

#[cfg(test)]
fn kda_dimensions(cfg: Glm5NextKdaConfig) -> Result<KdaDimensions, Glm53FlashTraceError> {
    for (field, value) in [
        ("hidden_size", cfg.hidden_size),
        ("num_heads", cfg.num_heads),
        ("head_dim", cfg.head_dim),
        ("gate_rank", cfg.gate_rank),
    ] {
        if value == 0 {
            return Err(Glm53FlashTraceError::KdaDimension {
                field,
                requirement: "nonzero".to_string(),
                actual: value,
            });
        }
    }
    if cfg.conv_width != 4 {
        return Err(Glm53FlashTraceError::KdaDimension {
            field: "conv_width",
            requirement: "exactly 4".to_string(),
            actual: cfg.conv_width,
        });
    }
    let qkv = cfg
        .num_heads
        .checked_mul(cfg.head_dim)
        .ok_or(Glm53FlashTraceError::KdaShapeOverflow { field: "qkv" })?;
    let conv_channels = qkv
        .checked_mul(3)
        .ok_or(Glm53FlashTraceError::KdaShapeOverflow {
            field: "convolution channels",
        })?;
    Ok(KdaDimensions { qkv, conv_channels })
}

/// Number of fixed decode-state tensors one KDA block carries.
#[cfg(test)]
pub const GLM5NEXT_KDA_STATE_PAIR_COUNT: usize = 2;

/// Fixed KDA decode state for `batch` rows, in [`Glm5NextKdaOutput`] order: convolution, then recurrent.
#[cfg(test)]
pub(crate) fn glm5next_kda_state_specs(
    cfg: Glm5NextKdaConfig,
    batch: usize,
) -> Result<[Glm5NextStateSpec; GLM5NEXT_KDA_STATE_PAIR_COUNT], Glm53FlashTraceError> {
    let dims = kda_dimensions(cfg)?;
    Ok([
        Glm5NextStateSpec {
            suffix: "conv_state",
            ty: TensorType::f32(vec![batch, dims.conv_channels, cfg.conv_width]),
        },
        Glm5NextStateSpec {
            suffix: "recurrent_state",
            ty: TensorType::f32(vec![batch, cfg.num_heads, cfg.head_dim, cfg.head_dim]),
        },
    ])
}

#[cfg(test)]
fn preflight_kda_value(
    b: &Builder,
    value: Traced,
    role: &'static str,
    shape: Vec<usize>,
) -> Result<(), Glm53FlashTraceError> {
    let actual = b.aval(value);
    if actual.dtype != DType::F32 {
        return Err(Glm53FlashTraceError::KdaDtype {
            role,
            expected: DType::F32,
            actual: actual.dtype,
        });
    }
    if actual.shape != shape {
        return Err(Glm53FlashTraceError::KdaShape {
            role,
            expected: shape,
            actual: actual.shape,
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn glm5next_kda_role_specs(
    revision: &str,
    prefix: &str,
    cfg: Glm5NextKdaConfig,
) -> Result<Vec<Glm5NextKdaRoleSpec>, Glm53FlashTraceError> {
    let dims = preflight_kda_config(revision, prefix, Glm5NextKdaScope::Kda, cfg)?;
    Ok(KDA_ROLE_ROWS
        .iter()
        .map(|&(role, suffix, dtype, shape)| {
            let shape = match shape {
                KdaRoleShape::QkvProjection => vec![dims.qkv, cfg.hidden_size],
                KdaRoleShape::Convolution => vec![dims.qkv, 1, cfg.conv_width],
                KdaRoleShape::RankIn => vec![cfg.gate_rank, cfg.hidden_size],
                KdaRoleShape::RankOut => vec![dims.qkv, cfg.gate_rank],
                KdaRoleShape::HeadProjection => vec![cfg.num_heads, cfg.hidden_size],
                KdaRoleShape::Heads => vec![cfg.num_heads],
                KdaRoleShape::Qkv => vec![dims.qkv],
                KdaRoleShape::HeadDim => vec![cfg.head_dim],
                KdaRoleShape::OutputProjection => vec![cfg.hidden_size, dims.qkv],
            };
            Glm5NextKdaRoleSpec {
                role,
                name: format!("{prefix}.{suffix}"),
                dtype,
                shape,
            }
        })
        .collect())
}

/// Retain selected Card 359 exact owners in model-role order without copying their bytes.
#[cfg(test)]
pub(crate) fn glm5next_kda_owner_table(
    loaded: &MixedLoadResult,
    revision: &str,
    prefix: &str,
    cfg: Glm5NextKdaConfig,
    roles: &[Glm5NextKdaRole],
) -> Result<Vec<Glm5NextKdaOwnerBinding>, Glm53FlashTraceError> {
    let specs = glm5next_kda_role_specs(revision, prefix, cfg)?;
    if loaded.artifact.revision() != revision {
        return Err(Glm53FlashTraceError::KdaRevision {
            expected: GLM53_FLASH_REVISION,
            actual: loaded.artifact.revision().to_string(),
        });
    }
    let mut table = Vec::with_capacity(roles.len());
    for (ordinal, &role) in roles.iter().enumerate() {
        if roles[..ordinal].contains(&role) {
            return Err(Glm53FlashTraceError::KdaDuplicateRole { role });
        }
        let spec = specs
            .iter()
            .find(|spec| spec.role == role)
            .expect("KDA role table covers every Glm5NextKdaRole");
        let source = loaded
            .exact_metadata
            .get(spec.name.as_str())
            .ok_or_else(|| Glm53FlashTraceError::KdaMissingSource {
                name: spec.name.clone(),
            })?;
        let expected_kind = match spec.dtype {
            DType::BF16 => ExactSourceKind::Bf16,
            DType::F32 => ExactSourceKind::F32,
            _ => unreachable!("KDA role table contains only BF16 and F32"),
        };
        if source.kind != expected_kind {
            return Err(Glm53FlashTraceError::KdaSourceKind {
                name: spec.name.clone(),
                expected: expected_kind,
                actual: source.kind,
            });
        }
        let expected_dtype = match spec.dtype {
            DType::BF16 => "BF16",
            DType::F32 => "F32",
            _ => unreachable!("KDA role table contains only BF16 and F32"),
        };
        if source.owner.descriptor().dtype() != expected_dtype {
            return Err(Glm53FlashTraceError::KdaSourceDtype {
                name: spec.name.clone(),
                expected: expected_dtype,
                actual: source.owner.descriptor().dtype().to_string(),
            });
        }
        if source.owner.descriptor().shape() != spec.shape.as_slice() {
            return Err(Glm53FlashTraceError::KdaSourceShape {
                name: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: source.owner.descriptor().shape().to_vec(),
            });
        }
        table.push(Glm5NextKdaOwnerBinding {
            role,
            name: spec.name.clone(),
            owner: Arc::clone(&source.owner),
        });
    }
    Ok(table)
}

#[cfg(test)]
fn kda_l2_norm(b: &Builder, x: Traced, eps: f32) -> Traced {
    let axis = b.aval(x).shape.len() - 1;
    let squares = b.binary(BinOp::Mul, x, x);
    let sum = b.reduce(RedOp::Sum, squares, axis, true);
    let denominator = b.unary(
        UnOp::Sqrt,
        b.binary_scalar(BinOp::Add, sum, Scalar::F32(eps)),
    );
    b.binary(BinOp::Div, x, denominator)
}

/// Compose one GLM5Next one-token KDA block from ordinary tensor graph primitives.
#[cfg(test)]
pub(crate) fn glm5next_kda_block(
    b: &Builder,
    revision: &str,
    prefix: &str,
    scope: Glm5NextKdaScope,
    cfg: Glm5NextKdaConfig,
    inputs: Glm5NextKdaInputs,
) -> Result<Glm5NextKdaOutput, Glm53FlashTraceError> {
    let dims = preflight_kda_config(revision, prefix, scope, cfg)?;
    let x_type = b.aval(inputs.x);
    let batch = *x_type.shape.first().unwrap_or(&0);
    preflight_kda_value(b, inputs.x, "input", vec![batch, 1, cfg.hidden_size])?;
    if batch == 0 {
        return Err(Glm53FlashTraceError::KdaDimension {
            field: "batch",
            requirement: "nonzero".to_string(),
            actual: batch,
        });
    }
    if let Some(validity) = inputs.validity {
        preflight_kda_value(b, validity, "validity", vec![batch, 1])?;
    }
    let [conv_spec, recurrent_spec] = glm5next_kda_state_specs(cfg, batch)?;
    preflight_kda_value(
        b,
        inputs.conv_state,
        "convolution state",
        conv_spec.ty.shape,
    )?;
    preflight_kda_value(
        b,
        inputs.recurrent_state,
        "recurrent state",
        recurrent_spec.ty.shape,
    )?;

    let specs = glm5next_kda_role_specs(revision, prefix, cfg)?;
    let sources = specs
        .iter()
        .map(|spec| {
            (
                spec.role,
                b.constant(&spec.name, TensorType::new(spec.shape.clone(), spec.dtype)),
            )
        })
        .collect::<Vec<_>>();
    let source = |role| {
        sources
            .iter()
            .find_map(|&(candidate, value)| (candidate == role).then_some(value))
            .expect("KDA source table covers every Glm5NextKdaRole")
    };

    let x = match inputs.validity {
        Some(validity) => {
            let validity = b.reshape(validity, vec![batch, 1, 1]);
            b.binary(BinOp::Mul, inputs.x, validity)
        }
        None => inputs.x,
    };

    let raw_qkv = [
        dense_source_linear(b, x, source(Glm5NextKdaRole::QProj)),
        dense_source_linear(b, x, source(Glm5NextKdaRole::KProj)),
        dense_source_linear(b, x, source(Glm5NextKdaRole::VProj)),
    ];
    let raw_qkv = b.concat(2, &raw_qkv);
    let current = b.transpose(raw_qkv, vec![0, 2, 1]);
    let history = b.slice(inputs.conv_state, 2, 1, cfg.conv_width);
    let conv_state = b.concat(2, &[history, current]);

    let conv_weights = [
        Glm5NextKdaRole::QConv1d,
        Glm5NextKdaRole::KConv1d,
        Glm5NextKdaRole::VConv1d,
    ]
    .map(|role| {
        b.reshape(
            b.cast(source(role), DType::F32),
            vec![dims.qkv, cfg.conv_width],
        )
    });
    let conv_weights = b.reshape(
        b.concat(0, &conv_weights),
        vec![1, dims.conv_channels, cfg.conv_width],
    );
    let convolved = b.reduce(
        RedOp::Sum,
        b.binary(BinOp::Mul, conv_state, conv_weights),
        2,
        true,
    );
    let convolved = silu(b, b.transpose(convolved, vec![0, 2, 1]));
    let q = b.reshape(
        b.slice(convolved, 2, 0, dims.qkv),
        vec![batch, cfg.num_heads, 1, cfg.head_dim],
    );
    let k = b.reshape(
        b.slice(convolved, 2, dims.qkv, 2 * dims.qkv),
        vec![batch, cfg.num_heads, 1, cfg.head_dim],
    );
    let v = b.reshape(
        b.slice(convolved, 2, 2 * dims.qkv, dims.conv_channels),
        vec![batch, cfg.num_heads, 1, cfg.head_dim],
    );
    let q = kda_l2_norm(b, q, 1e-6);
    let k = kda_l2_norm(b, k, 1e-6);

    let f = dense_source_linear(b, x, source(Glm5NextKdaRole::FAProj));
    let f = dense_source_linear(b, f, source(Glm5NextKdaRole::FBProj));
    let dt_bias = source(Glm5NextKdaRole::DtBias);
    let f = b.reshape(
        b.binary(BinOp::Add, f, dt_bias),
        vec![batch, cfg.num_heads, cfg.head_dim, 1],
    );
    let a = b.reshape(
        b.unary(UnOp::Exp, source(Glm5NextKdaRole::ALog)),
        vec![1, cfg.num_heads, 1, 1],
    );
    let forget = b.binary_scalar(
        BinOp::Mul,
        sigmoid(b, b.binary(BinOp::Mul, a, f)),
        Scalar::F32(cfg.gate_lower_bound),
    );
    let beta = sigmoid(b, dense_source_linear(b, x, source(Glm5NextKdaRole::BProj)));
    let beta = b.reshape(beta, vec![batch, cfg.num_heads, 1, 1]);

    let decay = b.unary(UnOp::Exp, forget);
    let decayed_state = b.binary(BinOp::Mul, inputs.recurrent_state, decay);
    let old_read = b.matmul(k, decayed_state);
    let delta = b.binary(BinOp::Mul, beta, b.binary(BinOp::Sub, v, old_read));
    let outer = b.matmul(b.transpose(k, vec![0, 1, 3, 2]), delta);
    let recurrent_state = b.binary(BinOp::Add, decayed_state, outer);
    let q = b.binary_scalar(
        BinOp::Mul,
        q,
        Scalar::F32(1.0 / (cfg.head_dim as f32).sqrt()),
    );
    let recurrent_output = b.matmul(q, recurrent_state);

    let norm_weight = b.cast(source(Glm5NextKdaRole::ONorm), DType::F32);
    let recurrent_output = rmsnorm(b, recurrent_output, norm_weight, 1e-5);
    let output_gate = dense_source_linear(b, x, source(Glm5NextKdaRole::GAProj));
    let output_gate = dense_source_linear(b, output_gate, source(Glm5NextKdaRole::GBProj));
    let output_gate = sigmoid(
        b,
        b.reshape(output_gate, vec![batch, cfg.num_heads, 1, cfg.head_dim]),
    );
    let output = b.binary(BinOp::Mul, recurrent_output, output_gate);
    let output = b.reshape(output, vec![batch, 1, dims.qkv]);
    let y = dense_source_linear(b, output, source(Glm5NextKdaRole::OProj));

    Ok(Glm5NextKdaOutput {
        y,
        conv_state,
        recurrent_state,
    })
}

#[cfg(test)]
mod kda_tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::fs;

    use poot_eval::exact_dense::DenseOwnerTensorView;
    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_graph_ir::{OpKind, Storage, ValueId};
    use poot_load::packed_safetensors::{
        AuthenticatedInventory, AuthenticatedSafetensorsHandleSet, ExactSourceOwnerCache,
        InventoryDecision, PackedArtifactManifest, PackedOwnerCache, PackedSafetensorsLimits,
        SourceSpan, TensorDisposition, sha256_digest,
    };
    use poot_tensor::HostTensor;
    use poot_test_util::{assert_close, max_abs_error};
    use serde_json::json;

    use crate::test_support::safetensors::{SourceRow, TempDir, write_shard};

    const PREFIX: &str = "model.language_model.layers.0.self_attn";

    pub(super) fn tiny_config() -> Glm5NextKdaConfig {
        Glm5NextKdaConfig {
            hidden_size: 4,
            num_heads: 2,
            head_dim: 2,
            conv_width: 4,
            gate_rank: 3,
            // Not the pinned -5.0, so a block that ignores the config value fails the oracle.
            gate_lower_bound: -4.0,
        }
    }

    /// Every KDA checkpoint tensor below `self_attn`, spelled as the HF checkpoint names it, with its element
    /// count. This is independent of `KDA_ROLE_ROWS`, so a consistent role/name swap there fails the oracle.
    pub(super) fn checkpoint_tensors(cfg: Glm5NextKdaConfig) -> [(&'static str, usize); 15] {
        let qkv = cfg.num_heads * cfg.head_dim;
        [
            ("q_proj.weight", qkv * cfg.hidden_size),
            ("k_proj.weight", qkv * cfg.hidden_size),
            ("v_proj.weight", qkv * cfg.hidden_size),
            ("q_conv1d.weight", qkv * cfg.conv_width),
            ("k_conv1d.weight", qkv * cfg.conv_width),
            ("v_conv1d.weight", qkv * cfg.conv_width),
            ("f_a_proj.weight", cfg.gate_rank * cfg.hidden_size),
            ("f_b_proj.weight", qkv * cfg.gate_rank),
            ("b_proj.weight", cfg.num_heads * cfg.hidden_size),
            ("g_a_proj.weight", cfg.gate_rank * cfg.hidden_size),
            ("g_b_proj.weight", qkv * cfg.gate_rank),
            ("A_log", cfg.num_heads),
            ("dt_bias", qkv),
            ("o_norm.weight", cfg.head_dim),
            ("o_proj.weight", cfg.hidden_size * qkv),
        ]
    }

    fn role_specs(cfg: Glm5NextKdaConfig) -> Vec<Glm5NextKdaRoleSpec> {
        glm5next_kda_role_specs(GLM53_FLASH_REVISION, PREFIX, cfg).unwrap()
    }

    fn named_const(graph: &Graph, name: &str) -> ValueId {
        graph
            .consts
            .iter()
            .copied()
            .find(|&value| graph.meta(value).name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing constant {name}"))
    }

    fn trace_kda(cfg: Glm5NextKdaConfig, batch: usize) -> Graph {
        let dims =
            preflight_kda_config(GLM53_FLASH_REVISION, PREFIX, Glm5NextKdaScope::Kda, cfg).unwrap();
        let b = Builder::new();
        let x = b.slot(
            Slot::Activation,
            TensorType::f32(vec![batch, 1, cfg.hidden_size]),
        );
        let validity = b.slot_named(
            Slot::Mask,
            "glm5next_kda_validity",
            TensorType::f32(vec![batch, 1]),
        );
        let conv_state = b.state_input(
            &format!("{PREFIX}.kda.conv_state"),
            TensorType::f32(vec![batch, dims.conv_channels, cfg.conv_width]),
            StateRole::Recurrent,
        );
        let recurrent_state = b.state_input(
            &format!("{PREFIX}.kda.recurrent_state"),
            TensorType::f32(vec![batch, cfg.num_heads, cfg.head_dim, cfg.head_dim]),
            StateRole::Recurrent,
        );
        let output = glm5next_kda_block(
            &b,
            GLM53_FLASH_REVISION,
            PREFIX,
            Glm5NextKdaScope::Kda,
            cfg,
            Glm5NextKdaInputs {
                x,
                validity: Some(validity),
                conv_state,
                recurrent_state,
            },
        )
        .unwrap();
        b.finish_with_state(
            output.y,
            &[
                (conv_state, output.conv_state),
                (recurrent_state, output.recurrent_state),
            ],
        )
    }

    /// KDA weights keyed by checkpoint tensor name below `self_attn`. The Card 368 text oracle builds one per
    /// layer and reuses [`reference_step`].
    pub(super) struct Fixture {
        pub(super) cfg: Glm5NextKdaConfig,
        pub(super) weights: BTreeMap<&'static str, Vec<f32>>,
    }

    impl Fixture {
        fn new() -> Self {
            let cfg = tiny_config();
            let weights = checkpoint_tensors(cfg)
                .into_iter()
                .enumerate()
                .map(|(ordinal, (name, len))| {
                    let values = match name {
                        "A_log" => vec![-0.5, 0.25],
                        "dt_bias" => vec![-0.25, 0.5, 0.125, -0.375],
                        "o_norm.weight" => vec![0.75, 1.25],
                        // `3 * ordinal` differs modulo 17 for every tensor, so no two tensors share values.
                        _ => (0..len)
                            .map(|index| {
                                let lane = (index * 5 + ordinal * 3) % 17;
                                (lane as f32 - 8.0) * 0.0625
                            })
                            .collect(),
                    };
                    assert_eq!(values.len(), len);
                    (name, values)
                })
                .collect();
            Self { cfg, weights }
        }

        fn weight(&self, name: &str) -> &[f32] {
            self.weights
                .get(name)
                .unwrap_or_else(|| panic!("no KDA fixture tensor {name}"))
        }

        fn inputs(
            &self,
            graph: &Graph,
            x: &[f32],
            validity: f32,
            state: &ReferenceState,
        ) -> HashMap<ValueId, Value> {
            graph
                .inputs
                .iter()
                .copied()
                .map(|id| {
                    let meta = graph.meta(id);
                    let name = meta.name.as_deref().unwrap_or_default();
                    let tensor = match meta.storage {
                        Storage::Slot(Slot::Activation) => {
                            HostTensor::f32(meta.aval.shape.clone(), x.to_vec())
                        }
                        Storage::Slot(Slot::Mask) => {
                            HostTensor::f32(meta.aval.shape.clone(), vec![validity])
                        }
                        Storage::State if name.ends_with("conv_state") => {
                            HostTensor::f32(meta.aval.shape.clone(), state.conv.clone())
                        }
                        Storage::State if name.ends_with("recurrent_state") => {
                            HostTensor::f32(meta.aval.shape.clone(), state.recurrent.clone())
                        }
                        Storage::Const => {
                            let tensor = name
                                .strip_prefix(PREFIX)
                                .and_then(|suffix| suffix.strip_prefix('.'))
                                .unwrap_or_else(|| panic!("unexpected KDA constant {name}"));
                            let data = self.weight(tensor).to_vec();
                            // A BF16-declared KDA weight binds as BF16 words (the typed carrier holds no
                            // f32 mirror), narrowed from the loaded f32 values.
                            if meta.aval.dtype == DType::BF16 {
                                let words: Vec<u16> = data
                                    .iter()
                                    .map(|&f| poot_runtime_common::f32_to_bf16(f))
                                    .collect();
                                HostTensor::bf16(meta.aval.shape.clone(), words)
                            } else {
                                HostTensor::f32(meta.aval.shape.clone(), data)
                            }
                        }
                        _ => panic!("unexpected KDA input {name}"),
                    };
                    (id, tensor.into())
                })
                .collect()
        }
    }

    #[derive(Clone)]
    pub(super) struct ReferenceState {
        pub(super) conv: Vec<f32>,
        pub(super) recurrent: Vec<f32>,
    }

    impl ReferenceState {
        fn zeros(cfg: Glm5NextKdaConfig) -> Self {
            let qkv = cfg.num_heads * cfg.head_dim;
            Self {
                conv: vec![0.0; 3 * qkv * cfg.conv_width],
                recurrent: vec![0.0; cfg.num_heads * cfg.head_dim * cfg.head_dim],
            }
        }
    }

    pub(super) struct ReferenceStep {
        pub(super) output: Vec<f32>,
        pub(super) state: ReferenceState,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(super) enum OracleMutation {
        None,
        QwenForget,
        BetaBeforeResidual,
        NoQueryScale,
        NoOutputGate,
        NormEpsilonOneEminusSix,
        FlattenedOutputNorm,
    }

    fn matvec(weight: &[f32], out: usize, input: &[f32]) -> Vec<f32> {
        let in_dim = input.len();
        (0..out)
            .map(|row| {
                (0..in_dim)
                    .map(|column| weight[row * in_dim + column] * input[column])
                    .sum()
            })
            .collect()
    }

    fn sigmoid_ref(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn silu_ref(x: f32) -> f32 {
        x * sigmoid_ref(x)
    }

    pub(super) fn reference_step(
        fixture: &Fixture,
        input: &[f32],
        validity: f32,
        state: &ReferenceState,
        mutation: OracleMutation,
    ) -> ReferenceStep {
        let cfg = fixture.cfg;
        let h = cfg.num_heads;
        let d = cfg.head_dim;
        let qkv = h * d;
        let x = input
            .iter()
            .map(|value| value * validity)
            .collect::<Vec<_>>();
        let projected = ["q_proj.weight", "k_proj.weight", "v_proj.weight"]
            .map(|name| matvec(fixture.weight(name), qkv, &x));
        let raw = projected.concat();

        let mut conv = state.conv.clone();
        let mut convolved = vec![0.0; 3 * qkv];
        for channel in 0..3 * qkv {
            let base = channel * cfg.conv_width;
            conv.copy_within(base + 1..base + cfg.conv_width, base);
            conv[base + cfg.conv_width - 1] = raw[channel];
            let name = ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"][channel / qkv];
            let local_channel = channel % qkv;
            let weight = fixture.weight(name);
            convolved[channel] = (0..cfg.conv_width)
                .map(|tap| conv[base + tap] * weight[local_channel * cfg.conv_width + tap])
                .sum::<f32>();
            convolved[channel] = silu_ref(convolved[channel]);
        }

        let mut q = convolved[..qkv].to_vec();
        let mut k = convolved[qkv..2 * qkv].to_vec();
        let v = convolved[2 * qkv..].to_vec();
        for head in 0..h {
            for values in [&mut q, &mut k] {
                let start = head * d;
                let denominator = (values[start..start + d]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    + 1e-6)
                    .sqrt();
                for value in &mut values[start..start + d] {
                    *value /= denominator;
                }
            }
        }

        let f_rank = matvec(fixture.weight("f_a_proj.weight"), cfg.gate_rank, &x);
        let mut f = matvec(fixture.weight("f_b_proj.weight"), qkv, &f_rank);
        for (value, bias) in f.iter_mut().zip(fixture.weight("dt_bias")) {
            *value += bias;
        }
        let beta = matvec(fixture.weight("b_proj.weight"), h, &x)
            .into_iter()
            .map(sigmoid_ref)
            .collect::<Vec<_>>();
        let a_log = fixture.weight("A_log");
        let mut recurrent = state.recurrent.clone();
        let mut recurrent_output = vec![0.0; qkv];
        for head in 0..h {
            let mut old_read = vec![0.0; d];
            for key in 0..d {
                let gate_input = a_log[head].exp() * f[head * d + key];
                let forget = if mutation == OracleMutation::QwenForget {
                    -a_log[head].exp() * (1.0 + gate_input.exp()).ln()
                } else {
                    cfg.gate_lower_bound * sigmoid_ref(gate_input)
                };
                let decay = forget.exp();
                for (value, old_read_value) in old_read.iter_mut().enumerate() {
                    let index = (head * d + key) * d + value;
                    recurrent[index] *= decay;
                    *old_read_value += k[head * d + key] * recurrent[index];
                }
            }
            let mut delta = vec![0.0; d];
            for value in 0..d {
                delta[value] = if mutation == OracleMutation::BetaBeforeResidual {
                    v[head * d + value] - beta[head] * old_read[value]
                } else {
                    beta[head] * (v[head * d + value] - old_read[value])
                };
            }
            for key in 0..d {
                for value in 0..d {
                    recurrent[(head * d + key) * d + value] += k[head * d + key] * delta[value];
                }
            }
            let query_scale = if mutation == OracleMutation::NoQueryScale {
                1.0
            } else {
                1.0 / (d as f32).sqrt()
            };
            for value in 0..d {
                recurrent_output[head * d + value] = (0..d)
                    .map(|key| {
                        q[head * d + key] * query_scale * recurrent[(head * d + key) * d + value]
                    })
                    .sum();
            }
        }

        let norm_eps = if mutation == OracleMutation::NormEpsilonOneEminusSix {
            1e-6
        } else {
            1e-5
        };
        let norm_weight = fixture.weight("o_norm.weight");
        if mutation == OracleMutation::FlattenedOutputNorm {
            let denominator = (recurrent_output
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / qkv as f32
                + norm_eps)
                .sqrt();
            for (index, value) in recurrent_output.iter_mut().enumerate() {
                *value = *value / denominator * norm_weight[index % d];
            }
        } else {
            for head in 0..h {
                let start = head * d;
                let denominator = (recurrent_output[start..start + d]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / d as f32
                    + norm_eps)
                    .sqrt();
                for value in 0..d {
                    recurrent_output[start + value] =
                        recurrent_output[start + value] / denominator * norm_weight[value];
                }
            }
        }

        let gate_rank = matvec(fixture.weight("g_a_proj.weight"), cfg.gate_rank, &x);
        let gate = matvec(fixture.weight("g_b_proj.weight"), qkv, &gate_rank);
        if mutation != OracleMutation::NoOutputGate {
            for (value, gate) in recurrent_output.iter_mut().zip(gate) {
                *value *= sigmoid_ref(gate);
            }
        }
        let output = matvec(
            fixture.weight("o_proj.weight"),
            cfg.hidden_size,
            &recurrent_output,
        );
        ReferenceStep {
            output,
            state: ReferenceState { conv, recurrent },
        }
    }

    fn graph_step(
        graph: &Graph,
        fixture: &Fixture,
        input: &[f32],
        validity: f32,
        state: &ReferenceState,
    ) -> ReferenceStep {
        let result = eval(
            graph,
            &fixture.inputs(graph, input, validity, state),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .unwrap();
        let output = result.output.into_host().expect("dense output");
        let states: Vec<HostTensor> = result
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();
        assert_eq!(states.len(), 2);
        ReferenceStep {
            output: output.as_f32().unwrap().to_vec(),
            state: ReferenceState {
                conv: states[0].as_f32().unwrap().to_vec(),
                recurrent: states[1].as_f32().unwrap().to_vec(),
            },
        }
    }

    #[test]
    fn glm5next_kda_matches_independent_recurrence() {
        let fixture = Fixture::new();
        let graph = trace_kda(fixture.cfg, 1);
        graph.validate().unwrap();
        let mut graph_state = ReferenceState::zeros(fixture.cfg);
        let mut reference_state = graph_state.clone();
        for input in [
            vec![0.5, -0.25, 0.75, 0.125],
            vec![-0.375, 0.625, 0.25, -0.5],
            vec![0.125, 0.25, -0.625, 0.875],
            vec![0.75, -0.125, -0.25, 0.5],
        ] {
            let actual = graph_step(&graph, &fixture, &input, 1.0, &graph_state);
            let expected = reference_step(
                &fixture,
                &input,
                1.0,
                &reference_state,
                OracleMutation::None,
            );
            assert_close(&actual.output, &expected.output, 3e-5);
            assert_close(&actual.state.conv, &expected.state.conv, 3e-5);
            assert_close(&actual.state.recurrent, &expected.state.recurrent, 3e-5);
            for mutation in [
                OracleMutation::BetaBeforeResidual,
                OracleMutation::NoQueryScale,
            ] {
                let changed = reference_step(&fixture, &input, 1.0, &reference_state, mutation);
                assert!(
                    max_abs_error(&actual.output, &changed.output) > 1e-6
                        || max_abs_error(&actual.state.recurrent, &changed.state.recurrent,) > 1e-6,
                    "recurrence mutation was not observable"
                );
            }
            graph_state = actual.state;
            reference_state = expected.state;
        }
    }

    #[test]
    fn glm5next_kda_forget_gate_uses_safe_lower_bound() {
        let fixture = Fixture::new();
        let graph = trace_kda(fixture.cfg, 1);
        let mut state = ReferenceState::zeros(fixture.cfg);
        for (index, value) in state.recurrent.iter_mut().enumerate() {
            *value = (index as f32 - 3.0) * 0.125;
        }
        let input = [0.625, -0.25, 0.5, 0.125];
        let actual = graph_step(&graph, &fixture, &input, 1.0, &state);
        let expected = reference_step(&fixture, &input, 1.0, &state, OracleMutation::None);
        let qwen = reference_step(&fixture, &input, 1.0, &state, OracleMutation::QwenForget);
        assert_close(&actual.state.recurrent, &expected.state.recurrent, 3e-5);
        assert!(max_abs_error(&actual.state.recurrent, &qwen.state.recurrent) > 1e-3);
    }

    #[test]
    fn glm5next_kda_conv_state_keeps_width_four() {
        let fixture = Fixture::new();
        let graph = trace_kda(fixture.cfg, 1);
        assert_eq!(graph.state.len(), 2);
        assert_eq!(
            graph.aval(graph.state[0].0),
            &TensorType::f32(vec![1, 12, 4])
        );
        assert_eq!(graph.aval(graph.state[0].0), graph.aval(graph.state[0].1));
        assert!(
            !graph
                .values
                .iter()
                .any(|meta| meta.aval.shape == [1, 3, 12])
        );

        let exact = trace_kda(Glm5NextKdaConfig::exact(), 2);
        exact.validate().unwrap();
        assert_eq!(exact.state.len(), 2);
        assert_eq!(
            exact.aval(exact.state[0].0),
            &TensorType::f32(vec![2, 24_576, 4])
        );
        assert_eq!(
            exact.aval(exact.state[1].0),
            &TensorType::f32(vec![2, 64, 128, 128])
        );
        assert_eq!(exact.aval(exact.state[0].0), exact.aval(exact.state[0].1));
        assert_eq!(exact.aval(exact.state[1].0), exact.aval(exact.state[1].1));
        assert_eq!(
            exact.aval(exact.output),
            &TensorType::f32(vec![2, 1, 4_096])
        );
        assert!(
            !exact
                .values
                .iter()
                .any(|meta| meta.aval.shape == [2, 3, 24_576])
        );
        let exact_specs = glm5next_kda_role_specs(
            GLM53_FLASH_REVISION,
            "model.language_model.layers.0.self_attn",
            Glm5NextKdaConfig::exact(),
        )
        .unwrap();
        for spec in exact_specs {
            let value = named_const(&exact, &spec.name);
            assert_eq!(exact.aval(value), &TensorType::new(spec.shape, spec.dtype));
        }
        assert!(!exact.eqns.iter().any(|eqn| {
            matches!(
                eqn.op,
                OpKind::PackedDequant { .. } | OpKind::PackedContraction { .. }
            )
        }));

        let mut state = ReferenceState::zeros(fixture.cfg);
        for input in [
            vec![0.25, 0.5, -0.25, 0.75],
            vec![0.5, -0.5, 0.125, 0.25],
            vec![-0.25, 0.75, 0.5, -0.125],
            vec![0.875, 0.25, -0.5, 0.375],
            vec![-0.625, 0.5, 0.25, 0.125],
        ] {
            let actual = graph_step(&graph, &fixture, &input, 1.0, &state);
            let expected = reference_step(&fixture, &input, 1.0, &state, OracleMutation::None);
            assert_close(&actual.state.conv, &expected.state.conv, 2e-6);
            state = actual.state;
        }
    }

    #[test]
    fn glm5next_kda_output_gate_and_norm_are_load_bearing() {
        let fixture = Fixture::new();
        let graph = trace_kda(fixture.cfg, 1);
        let mut state = ReferenceState::zeros(fixture.cfg);
        for (index, value) in state.recurrent.iter_mut().enumerate() {
            *value = (index as f32 + 1.0) * 0.0625;
        }
        let input = [0.75, -0.375, 0.625, 0.25];
        let actual = graph_step(&graph, &fixture, &input, 1.0, &state);
        let expected = reference_step(&fixture, &input, 1.0, &state, OracleMutation::None);
        assert_close(&actual.output, &expected.output, 3e-5);
        for mutation in [
            OracleMutation::NoOutputGate,
            OracleMutation::NormEpsilonOneEminusSix,
            OracleMutation::FlattenedOutputNorm,
        ] {
            let changed = reference_step(&fixture, &input, 1.0, &state, mutation);
            assert!(
                max_abs_error(&actual.output, &changed.output) > 1e-6,
                "output mutation was not observable"
            );
        }
    }

    #[test]
    fn glm5next_kda_state_is_explicit_and_sensitive() {
        let fixture = Fixture::new();
        let graph = trace_kda(fixture.cfg, 1);
        assert_eq!(graph.state.len(), 2);
        assert_eq!(
            graph.aval(graph.state[1].0),
            &TensorType::f32(vec![1, 2, 2, 2])
        );
        let zeros = ReferenceState::zeros(fixture.cfg);
        let mut nonzero = zeros.clone();
        for (index, value) in nonzero.conv.iter_mut().enumerate() {
            *value = (index as f32 - 8.0) * 0.03125;
        }
        for (index, value) in nonzero.recurrent.iter_mut().enumerate() {
            *value = (index as f32 + 1.0) * 0.0625;
        }
        let zero_input = [0.0; 4];
        let nonzero_input = [0.5, -0.25, 0.75, 0.125];
        let from_state = graph_step(&graph, &fixture, &zero_input, 1.0, &nonzero);
        let expected_from_state =
            reference_step(&fixture, &zero_input, 1.0, &nonzero, OracleMutation::None);
        assert_close(&from_state.output, &expected_from_state.output, 3e-5);
        assert_close(
            &from_state.state.conv,
            &expected_from_state.state.conv,
            3e-5,
        );
        assert_close(
            &from_state.state.recurrent,
            &expected_from_state.state.recurrent,
            3e-5,
        );
        let without_state = graph_step(&graph, &fixture, &zero_input, 1.0, &zeros);
        let expected_without_state =
            reference_step(&fixture, &zero_input, 1.0, &zeros, OracleMutation::None);
        assert_close(&without_state.output, &expected_without_state.output, 3e-5);
        assert_close(
            &without_state.state.conv,
            &expected_without_state.state.conv,
            3e-5,
        );
        assert_close(
            &without_state.state.recurrent,
            &expected_without_state.state.recurrent,
            3e-5,
        );
        assert!(max_abs_error(&from_state.state.conv, &without_state.state.conv) > 1e-4);
        assert!(max_abs_error(&from_state.state.recurrent, &without_state.state.recurrent) > 1e-4);
        let from_input = graph_step(&graph, &fixture, &nonzero_input, 1.0, &zeros);
        let expected_from_input =
            reference_step(&fixture, &nonzero_input, 1.0, &zeros, OracleMutation::None);
        assert_close(&from_input.output, &expected_from_input.output, 3e-5);
        assert_close(
            &from_input.state.conv,
            &expected_from_input.state.conv,
            3e-5,
        );
        assert_close(
            &from_input.state.recurrent,
            &expected_from_input.state.recurrent,
            3e-5,
        );
        assert!(max_abs_error(&from_input.output, &without_state.output) > 1e-5);
        let masked = graph_step(&graph, &fixture, &nonzero_input, 0.0, &nonzero);
        let expected_masked = reference_step(
            &fixture,
            &nonzero_input,
            0.0,
            &nonzero,
            OracleMutation::None,
        );
        assert_close(&masked.output, &expected_masked.output, 3e-5);
        assert_close(&masked.state.conv, &expected_masked.state.conv, 3e-5);
        assert_close(
            &masked.state.recurrent,
            &expected_masked.state.recurrent,
            3e-5,
        );
        assert_close(&masked.output, &from_state.output, 2e-6);
        assert_close(&masked.state.conv, &from_state.state.conv, 2e-6);
        assert_close(&masked.state.recurrent, &from_state.state.recurrent, 2e-6);

        let b = Builder::new();
        let x = b.slot(Slot::Activation, TensorType::f32(vec![1, 1, 4]));
        let conv = b.state_input(
            "bad.conv",
            TensorType::f32(vec![1, 12, 3]),
            StateRole::Recurrent,
        );
        let recurrent = b.state_input(
            "bad.recurrent",
            TensorType::f32(vec![1, 2, 2, 2]),
            StateRole::Recurrent,
        );
        assert!(matches!(
            glm5next_kda_block(
                &b,
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextKdaScope::Kda,
                fixture.cfg,
                Glm5NextKdaInputs {
                    x,
                    validity: None,
                    conv_state: conv,
                    recurrent_state: recurrent,
                },
            ),
            Err(Glm53FlashTraceError::KdaShape {
                role: "convolution state",
                ..
            })
        ));
        let rejected = b.finish(x);
        assert!(rejected.eqns.is_empty());
        assert!(rejected.consts.iter().all(|&id| {
            !rejected
                .meta(id)
                .name
                .as_deref()
                .is_some_and(|name| name.starts_with(PREFIX))
        }));
    }

    #[test]
    fn glm5next_kda_manifest_rows_are_exact() {
        let manifest: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../poot-load/src/glm53_flash_data/audit-manifest.json"
        ))
        .unwrap();
        let raw_index: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../poot-load/src/glm53_flash_data/raw-index.json"
        ))
        .unwrap();
        let specs = glm5next_kda_role_specs(
            GLM53_FLASH_REVISION,
            "model.language_model.layers.{layer}.self_attn",
            Glm5NextKdaConfig::exact(),
        )
        .unwrap();
        let names = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(names.len(), 15);
        let roles = specs.iter().map(|spec| spec.role).collect::<BTreeSet<_>>();
        assert_eq!(roles.len(), 15);
        let produced = specs
            .iter()
            .map(|spec| (spec.name.clone(), spec.shape.iter().product::<usize>()))
            .collect::<BTreeSet<_>>();
        let literal = checkpoint_tensors(Glm5NextKdaConfig::exact()).map(|(name, len)| {
            (
                format!("model.language_model.layers.{{layer}}.self_attn.{name}"),
                len,
            )
        });
        assert_eq!(produced, BTreeSet::from(literal));
        let families = manifest["tensor_families"].as_array().unwrap();
        for spec in &specs {
            let family = families
                .iter()
                .find(|family| family["pattern"].as_str() == Some(spec.name.as_str()))
                .unwrap_or_else(|| panic!("missing committed KDA family {}", spec.name));
            let expected_dtype = match spec.dtype {
                DType::BF16 => "BF16",
                DType::F32 => "F32",
                _ => unreachable!(),
            };
            let variant = family["variants"]
                .as_array()
                .unwrap()
                .iter()
                .find(|variant| {
                    variant["dtype"].as_str() == Some(expected_dtype)
                        && variant["quant_role"].as_str() == Some("unquantized")
                        && variant["count"].as_u64() == Some(34)
                })
                .unwrap_or_else(|| panic!("missing unquantized KDA variant {}", spec.name));
            let shape = variant["physical_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(shape, spec.shape, "committed shape for {}", spec.name);
            assert!(variant["pair_pattern"].is_null());
            assert!(variant["block_shape"].is_null());
        }

        let weight_map = raw_index["weight_map"].as_object().unwrap();
        let exact_layer_zero =
            glm5next_kda_role_specs(GLM53_FLASH_REVISION, PREFIX, Glm5NextKdaConfig::exact())
                .unwrap();
        for spec in exact_layer_zero {
            assert!(weight_map.contains_key(&spec.name), "missing {}", spec.name);
            if let Some(prefix) = spec.name.strip_suffix(".weight") {
                let scale = format!("{prefix}.weight_scale_inv");
                assert!(
                    !weight_map.contains_key(&scale),
                    "KDA source unexpectedly has a scale sibling: {scale}"
                );
            }
        }
    }

    fn classify_kda_fixture(
        inventory: AuthenticatedInventory<'_>,
    ) -> Result<Vec<InventoryDecision>, &'static str> {
        inventory
            .rows()
            .map(|row| {
                let disposition = if row.name().ends_with("q_proj.weight")
                    || row.name().ends_with("k_proj.weight")
                {
                    TensorDisposition::DenseBf16
                } else if row.name().ends_with("A_log") {
                    TensorDisposition::DenseF32
                } else {
                    return Err("unexpected KDA fixture row");
                };
                Ok(InventoryDecision::new(row.key(), disposition))
            })
            .collect()
    }

    #[test]
    fn glm5next_kda_synthetic_owner_handoff_is_zero_copy() {
        let cfg = tiny_config();
        let specs = role_specs(cfg);
        let selected = [
            Glm5NextKdaRole::QProj,
            Glm5NextKdaRole::KProj,
            Glm5NextKdaRole::ALog,
        ]
        .map(|role| specs.iter().find(|spec| spec.role == role).unwrap());
        let rows = [
            SourceRow {
                name: selected[0].name.clone(),
                dtype: "BF16",
                shape: selected[0].shape.clone(),
                bytes: (0..selected[0].shape.iter().product::<usize>() * 2)
                    .map(|byte| byte as u8)
                    .collect(),
            },
            SourceRow {
                name: selected[1].name.clone(),
                dtype: "BF16",
                shape: selected[1].shape.clone(),
                bytes: (0..selected[1].shape.iter().product::<usize>() * 2)
                    .map(|byte| 0x80_u8.wrapping_add(byte as u8))
                    .collect(),
            },
            SourceRow {
                name: selected[2].name.clone(),
                dtype: "F32",
                shape: selected[2].shape.clone(),
                bytes: vec![0, 0, 0, 0, 0, 0, 0x80, 0x3f],
            },
        ];
        let root = TempDir::new();
        let config = br#"{"model_type":"glm5_next"}"#;
        fs::write(root.path.join("config.json"), config).unwrap();
        let filename = "model-00001-of-00001.safetensors";
        let shard = write_shard(&root.path, filename, &rows);
        let weight_map = rows
            .iter()
            .map(|row| (row.name.clone(), json!(filename)))
            .collect::<serde_json::Map<_, _>>();
        let index = serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap();
        fs::write(root.path.join("model.safetensors.index.json"), &index).unwrap();
        let manifest = PackedArtifactManifest {
            repository: "zai-org/GLM-5.3-Flash".to_string(),
            revision: GLM53_FLASH_REVISION.to_string(),
            config_length: config.len(),
            config_sha256: sha256_digest(config),
            index_sha256: sha256_digest(&index),
            shards: vec![shard],
        };
        let mut authenticated = AuthenticatedSafetensorsHandleSet::authenticate(
            &root.path,
            manifest,
            PackedSafetensorsLimits {
                config_bytes: 4_096,
                index_bytes: 4_096,
                header_bytes_per_shard: 4_096,
                shard_count: 1,
                tensor_entries: rows.len(),
                selected_source_bytes: 4_096,
                packed_source_bytes: 1,
            },
        )
        .unwrap();
        let mut packed_cache = PackedOwnerCache::new();
        let mut exact_cache = ExactSourceOwnerCache::new();
        let loaded = authenticated
            .load_mixed(&mut packed_cache, &mut exact_cache, classify_kda_fixture)
            .unwrap();
        let table = glm5next_kda_owner_table(
            &loaded,
            GLM53_FLASH_REVISION,
            PREFIX,
            cfg,
            &[
                Glm5NextKdaRole::QProj,
                Glm5NextKdaRole::KProj,
                Glm5NextKdaRole::ALog,
            ],
        )
        .unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(loaded.exact_metadata.len(), 3);
        assert_eq!(exact_cache.len(), 3);
        assert!(matches!(
            glm5next_kda_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextKdaRole::QProj, Glm5NextKdaRole::QProj],
            ),
            Err(Glm53FlashTraceError::KdaDuplicateRole {
                role: Glm5NextKdaRole::QProj
            })
        ));
        assert!(matches!(
            glm5next_kda_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextKdaRole::VProj],
            ),
            Err(Glm53FlashTraceError::KdaMissingSource { .. })
        ));
        let mut wrong_kind = loaded.clone();
        wrong_kind
            .exact_metadata
            .get_mut(selected[2].name.as_str())
            .unwrap()
            .kind = ExactSourceKind::Bf16;
        assert!(matches!(
            glm5next_kda_owner_table(
                &wrong_kind,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextKdaRole::ALog],
            ),
            Err(Glm53FlashTraceError::KdaSourceKind { .. })
        ));
        let wrong_shape = Glm5NextKdaConfig {
            hidden_size: cfg.hidden_size + 1,
            ..cfg
        };
        assert!(matches!(
            glm5next_kda_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                wrong_shape,
                &[Glm5NextKdaRole::QProj],
            ),
            Err(Glm53FlashTraceError::KdaSourceShape { .. })
        ));
        let views = table
            .iter()
            .map(|binding| DenseOwnerTensorView::new(Arc::clone(binding.owner())).unwrap())
            .collect::<Vec<_>>();
        for (binding, view) in table.iter().zip(&views) {
            let spec = selected
                .iter()
                .find(|spec| spec.role == binding.role())
                .unwrap();
            let source = rows.iter().find(|row| row.name == binding.name()).unwrap();
            let loaded_source = loaded.exact_metadata.get(binding.name()).unwrap();
            assert!(Arc::ptr_eq(binding.owner(), &loaded_source.owner));
            assert!(Arc::ptr_eq(binding.owner(), view.owner()));
            assert_eq!(binding.owner().artifact(), &loaded.artifact);
            assert_eq!(binding.name(), view.descriptor().name());
            assert_eq!(view.dtype(), spec.dtype);
            assert_eq!(view.shape(), spec.shape.as_slice());
            assert_eq!(binding.owner().bytes(), source.bytes.as_slice());
            assert_eq!(
                binding.owner().bytes().as_ptr(),
                view.owner().bytes().as_ptr()
            );
            assert_eq!(binding.owner().bytes().len(), view.source_byte_len());
            assert_eq!(
                binding.owner().descriptor().shard(),
                "model-00001-of-00001.safetensors"
            );
        }
        assert_eq!(table[0].owner().descriptor().span(), SourceSpan::new(0, 32));
        assert_eq!(
            table[1].owner().descriptor().span(),
            SourceSpan::new(32, 64)
        );
        assert_eq!(
            table[2].owner().descriptor().span(),
            SourceSpan::new(64, 72)
        );
        let graph = trace_kda(cfg, 1);
        let bound: HashMap<ValueId, Value> = table
            .iter()
            .zip(&views)
            .map(|(binding, view)| {
                (
                    named_const(&graph, binding.name()),
                    Value::from(view.clone()),
                )
            })
            .collect();
        for (binding, view) in table.iter().zip(&views) {
            assert_eq!(
                &bound[&named_const(&graph, binding.name())],
                &Value::from(view.clone())
            );
        }
        assert!(packed_cache.is_empty());
    }

    #[test]
    fn glm5next_kda_typed_preflight_rejects_revision_scope_and_dtype() {
        let cfg = tiny_config();
        for scope in [
            Glm5NextKdaScope::ImageVideo,
            Glm5NextKdaScope::Mtp,
            Glm5NextKdaScope::Dsa,
            Glm5NextKdaScope::Mhc,
            Glm5NextKdaScope::Ffn,
        ] {
            assert!(matches!(
                preflight_kda_config(GLM53_FLASH_REVISION, PREFIX, scope, cfg),
                Err(Glm53FlashTraceError::KdaUnsupportedScope { .. })
            ));
        }
        assert!(matches!(
            preflight_kda_config("wrong", PREFIX, Glm5NextKdaScope::Kda, cfg),
            Err(Glm53FlashTraceError::KdaRevision { .. })
        ));
        assert!(matches!(
            preflight_kda_config(
                GLM53_FLASH_REVISION,
                "model.language_model.layers..self_attn",
                Glm5NextKdaScope::Kda,
                cfg,
            ),
            Err(Glm53FlashTraceError::KdaSourcePrefix { .. })
        ));
        assert!(matches!(
            preflight_kda_config(
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextKdaScope::Kda,
                Glm5NextKdaConfig {
                    conv_width: 3,
                    ..cfg
                },
            ),
            Err(Glm53FlashTraceError::KdaDimension {
                field: "conv_width",
                ..
            })
        ));
        let b = Builder::new();
        let x = b.constant("bad.x", TensorType::new(vec![1, 1, 4], DType::BF16));
        let conv = b.state_input(
            "bad.conv",
            TensorType::f32(vec![1, 12, 4]),
            StateRole::Recurrent,
        );
        let recurrent = b.state_input(
            "bad.recurrent",
            TensorType::f32(vec![1, 2, 2, 2]),
            StateRole::Recurrent,
        );
        assert!(matches!(
            glm5next_kda_block(
                &b,
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextKdaScope::Kda,
                cfg,
                Glm5NextKdaInputs {
                    x,
                    validity: None,
                    conv_state: conv,
                    recurrent_state: recurrent,
                },
            ),
            Err(Glm53FlashTraceError::KdaDtype { role: "input", .. })
        ));
        let graph = b.finish(x);
        assert!(graph.eqns.is_empty());
    }
}

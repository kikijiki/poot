use poot_tensor::DType;
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use super::{GLM53_FLASH_REVISION, Glm5NextStateSpec, Glm53FlashTraceError};
#[cfg(test)]
use poot_graph_ir::ops::{
    attention_masked, layernorm, relu, rmsnorm, softmax, stable_descending_rank,
    stable_descending_rank_masked,
};
#[cfg(test)]
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Slot, StateRole, TensorType, Traced};
#[cfg(test)]
use poot_load::packed_safetensors::{ExactSourceKind, ExactSourceOwner, MixedLoadResult};

#[cfg(test)]
use super::dense_source_linear;

/// Largest cache accepted by Card 367's deliberately quadratic pairwise-rank composition.
///
/// A semantic-oracle bound, not a production capacity: a larger cache needs an exact top-k/pool lowering in
/// place of the pairwise rank.
#[cfg(test)]
pub(crate) const GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT: usize = 64;

#[cfg(test)]
const MASK_NEG: f32 = -1.0e30;

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Glm5NextDsaScope {
    Dsa,
    ImageVideo,
    Mtp,
    Kda,
    Mhc,
    Ffn,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextIndexerKind {
    Full,
    Shared,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm5NextDsaConfig {
    pub hidden_size: usize,
    pub query_rank: usize,
    pub kv_rank: usize,
    pub num_heads: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    /// Selected raw-token budget. It must be divisible by `index_kpool`.
    pub index_topk: usize,
    pub index_kpool: usize,
    pub qk_rope_head_dim: usize,
    pub mla_use_nope: bool,
    pub index_kpool_always_select_tail: bool,
    pub index_kpool_compress: bool,
    pub indexer_kind: Glm5NextIndexerKind,
}

#[cfg(test)]
impl Glm5NextDsaConfig {
    pub const fn exact() -> Self {
        Self {
            hidden_size: 4_096,
            query_rank: 1_536,
            kv_rank: 512,
            num_heads: 64,
            qk_head_dim: 256,
            v_head_dim: 256,
            index_n_heads: 32,
            index_head_dim: 128,
            index_topk: 2_048,
            index_kpool: 4,
            qk_rope_head_dim: 0,
            mla_use_nope: true,
            index_kpool_always_select_tail: true,
            index_kpool_compress: true,
            indexer_kind: Glm5NextIndexerKind::Full,
        }
    }

    #[cfg(test)]
    pub(crate) const fn checked_selection_width(self) -> Option<usize> {
        match self.index_kpool.checked_sub(1) {
            Some(tail) => self.index_topk.checked_add(tail),
            None => None,
        }
    }

    #[cfg(test)]
    pub(crate) const fn pool_topk(self) -> usize {
        self.index_topk / self.index_kpool
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Glm5NextDsaRole {
    QAProj,
    QALayerNorm,
    QBProj,
    KVAProjWithMqa,
    KVALayerNorm,
    KVBProj,
    OProj,
    IndexerWQB,
    IndexerWK,
    IndexerWeightsProj,
    IndexerKNormWeight,
    IndexerKNormBias,
    IndexerCompressGate,
    IndexerCompressApe,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Glm5NextDsaSourceClass {
    PackedProjection,
    ExactDense,
}

#[derive(Clone, Copy)]
#[cfg(test)]
enum DsaRoleShape {
    QueryA,
    QueryRank,
    QueryB,
    KvA,
    KvRank,
    KvB,
    Output,
    IndexQuery,
    IndexKey,
    IndexWeights,
    IndexWidth,
    IndexGate,
    IndexApe,
}

#[cfg(test)]
const DSA_ROLE_ROWS: [(
    Glm5NextDsaRole,
    &str,
    DType,
    Glm5NextDsaSourceClass,
    DsaRoleShape,
); 14] = [
    (
        Glm5NextDsaRole::QAProj,
        "q_a_proj.weight",
        DType::E4M3FN,
        Glm5NextDsaSourceClass::PackedProjection,
        DsaRoleShape::QueryA,
    ),
    (
        Glm5NextDsaRole::QALayerNorm,
        "q_a_layernorm.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::QueryRank,
    ),
    (
        Glm5NextDsaRole::QBProj,
        "q_b_proj.weight",
        DType::E4M3FN,
        Glm5NextDsaSourceClass::PackedProjection,
        DsaRoleShape::QueryB,
    ),
    (
        Glm5NextDsaRole::KVAProjWithMqa,
        "kv_a_proj_with_mqa.weight",
        DType::E4M3FN,
        Glm5NextDsaSourceClass::PackedProjection,
        DsaRoleShape::KvA,
    ),
    (
        Glm5NextDsaRole::KVALayerNorm,
        "kv_a_layernorm.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::KvRank,
    ),
    (
        Glm5NextDsaRole::KVBProj,
        "kv_b_proj.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::KvB,
    ),
    (
        Glm5NextDsaRole::OProj,
        "o_proj.weight",
        DType::E4M3FN,
        Glm5NextDsaSourceClass::PackedProjection,
        DsaRoleShape::Output,
    ),
    (
        Glm5NextDsaRole::IndexerWQB,
        "indexer.wq_b.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexQuery,
    ),
    (
        Glm5NextDsaRole::IndexerWK,
        "indexer.wk.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexKey,
    ),
    (
        Glm5NextDsaRole::IndexerWeightsProj,
        "indexer.weights_proj.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexWeights,
    ),
    (
        Glm5NextDsaRole::IndexerKNormWeight,
        "indexer.k_norm.weight",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexWidth,
    ),
    (
        Glm5NextDsaRole::IndexerKNormBias,
        "indexer.k_norm.bias",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexWidth,
    ),
    (
        Glm5NextDsaRole::IndexerCompressGate,
        "indexer.index_kpool_compress_gate",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexGate,
    ),
    (
        Glm5NextDsaRole::IndexerCompressApe,
        "indexer.index_kpool_compress_ape",
        DType::BF16,
        Glm5NextDsaSourceClass::ExactDense,
        DsaRoleShape::IndexApe,
    ),
];

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Glm5NextDsaRoleSpec {
    pub role: Glm5NextDsaRole,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub source_class: Glm5NextDsaSourceClass,
}

#[derive(Clone, Debug)]
#[cfg(test)]
pub(crate) struct Glm5NextDsaOwnerBinding {
    name: String,
    owner: Arc<ExactSourceOwner>,
}

#[cfg(test)]
impl Glm5NextDsaOwnerBinding {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn owner(&self) -> &Arc<ExactSourceOwner> {
        &self.owner
    }
}

#[derive(Clone, Copy, Debug)]
#[cfg(test)]
pub(crate) struct Glm5NextDsaInputs {
    pub x: Traced,
    /// Per-batch 0/1 validity for the current query, `[B,1]`.
    pub query_validity: Traced,
    /// Per-batch causal/key visibility for every fixed cache slot, `[B,C]`.
    pub visibility: Traced,
    /// Runtime scalar I32 write position.
    pub position: Traced,
    pub k_state: Traced,
    pub v_state: Traced,
    pub indexer_state: Traced,
}

#[derive(Clone, Copy, Debug)]
#[cfg(test)]
pub(crate) struct Glm5NextDsaOutput {
    pub y: Traced,
    pub selection: Traced,
    pub pool_scores: Traced,
    pub k_state: Traced,
    pub v_state: Traced,
    pub indexer_state: Traced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum Glm5NextDsaSelectionLowering {
    BoundedPairwise,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) struct Glm5NextDsaSelectionEvidence {
    pub capacity: usize,
    pub lowering: Glm5NextDsaSelectionLowering,
}

#[cfg(test)]
impl Glm5NextDsaSelectionEvidence {
    #[cfg(test)]
    pub(crate) const fn production_capacity_admitted(self) -> bool {
        false
    }
}

#[derive(Clone, Copy)]
#[cfg(test)]
struct DsaDimensions {
    q_width: usize,
    v_width: usize,
    kv_width: usize,
    kv_head_width: usize,
    index_width: usize,
    indexer_lanes: usize,
    selection_width: usize,
}

#[cfg(test)]
fn checked_numel(field: &'static str, extents: &[usize]) -> Result<usize, Glm53FlashTraceError> {
    extents.iter().try_fold(1usize, |total, &extent| {
        total
            .checked_mul(extent)
            .ok_or(Glm53FlashTraceError::DsaShapeOverflow { field })
    })
}

#[cfg(test)]
fn valid_prefix(prefix: &str) -> bool {
    prefix
        .strip_prefix("model.language_model.layers.")
        .and_then(|suffix| suffix.strip_suffix(".self_attn"))
        .is_some_and(|layer| {
            layer == "{layer}"
                || (!layer.is_empty() && layer.bytes().all(|byte| byte.is_ascii_digit()))
        })
}

#[cfg(test)]
fn preflight_dsa_config(
    revision: &str,
    prefix: &str,
    scope: Glm5NextDsaScope,
    cfg: Glm5NextDsaConfig,
    capacity: usize,
) -> Result<DsaDimensions, Glm53FlashTraceError> {
    if revision != GLM53_FLASH_REVISION {
        return Err(Glm53FlashTraceError::DsaRevision {
            expected: GLM53_FLASH_REVISION,
            actual: revision.to_string(),
        });
    }
    if scope != Glm5NextDsaScope::Dsa {
        return Err(Glm53FlashTraceError::DsaUnsupportedScope { scope });
    }
    if !valid_prefix(prefix) {
        return Err(Glm53FlashTraceError::DsaSourcePrefix {
            prefix: prefix.to_string(),
        });
    }
    dsa_dimensions(cfg, capacity)
}

#[cfg(test)]
fn dsa_dimensions(
    cfg: Glm5NextDsaConfig,
    capacity: usize,
) -> Result<DsaDimensions, Glm53FlashTraceError> {
    for (field, value) in [
        ("hidden_size", cfg.hidden_size),
        ("query_rank", cfg.query_rank),
        ("kv_rank", cfg.kv_rank),
        ("num_heads", cfg.num_heads),
        ("qk_head_dim", cfg.qk_head_dim),
        ("v_head_dim", cfg.v_head_dim),
        ("index_n_heads", cfg.index_n_heads),
        ("index_head_dim", cfg.index_head_dim),
        ("index_topk", cfg.index_topk),
    ] {
        if value == 0 {
            return Err(Glm53FlashTraceError::DsaDimension {
                field,
                requirement: "nonzero".to_string(),
                actual: value,
            });
        }
    }
    if cfg.index_kpool != 4 {
        return Err(Glm53FlashTraceError::DsaDimension {
            field: "index_kpool",
            requirement: "exactly 4".to_string(),
            actual: cfg.index_kpool,
        });
    }
    if !cfg.index_topk.is_multiple_of(cfg.index_kpool) {
        return Err(Glm53FlashTraceError::DsaDimension {
            field: "index_topk",
            requirement: "divisible by index_kpool".to_string(),
            actual: cfg.index_topk,
        });
    }
    if cfg.pool_topk() > 512 {
        return Err(Glm53FlashTraceError::DsaDimension {
            field: "index_topk/index_kpool",
            requirement: "at most 512 pools".to_string(),
            actual: cfg.pool_topk(),
        });
    }
    if cfg.qk_rope_head_dim != 0 || !cfg.mla_use_nope {
        return Err(Glm53FlashTraceError::DsaRopeContract {
            rope_width: cfg.qk_rope_head_dim,
            mla_use_nope: cfg.mla_use_nope,
        });
    }
    if !cfg.index_kpool_always_select_tail
        || !cfg.index_kpool_compress
        || cfg.indexer_kind != Glm5NextIndexerKind::Full
    {
        return Err(Glm53FlashTraceError::DsaIndexerContract {
            kind: cfg.indexer_kind,
            compress: cfg.index_kpool_compress,
            always_select_tail: cfg.index_kpool_always_select_tail,
        });
    }
    if !(cfg.index_kpool..=GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT).contains(&capacity) {
        return Err(Glm53FlashTraceError::DsaPairwiseCapacity {
            capacity,
            limit: GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT,
        });
    }
    let q_width = cfg.num_heads.checked_mul(cfg.qk_head_dim).ok_or(
        Glm53FlashTraceError::DsaShapeOverflow {
            field: "query width",
        },
    )?;
    let v_width = cfg.num_heads.checked_mul(cfg.v_head_dim).ok_or(
        Glm53FlashTraceError::DsaShapeOverflow {
            field: "value width",
        },
    )?;
    let kv_head_width = cfg.qk_head_dim.checked_add(cfg.v_head_dim).ok_or(
        Glm53FlashTraceError::DsaShapeOverflow {
            field: "KV head width",
        },
    )?;
    let kv_width = cfg
        .num_heads
        .checked_mul(kv_head_width)
        .ok_or(Glm53FlashTraceError::DsaShapeOverflow { field: "KV width" })?;
    let index_width = cfg.index_n_heads.checked_mul(cfg.index_head_dim).ok_or(
        Glm53FlashTraceError::DsaShapeOverflow {
            field: "index query width",
        },
    )?;
    let indexer_lanes = cfg
        .index_head_dim
        .checked_mul(2)
        .and_then(|lanes| lanes.checked_add(1))
        .ok_or(Glm53FlashTraceError::DsaShapeOverflow {
            field: "indexer state lanes",
        })?;
    let selection_width =
        cfg.checked_selection_width()
            .ok_or(Glm53FlashTraceError::DsaShapeOverflow {
                field: "selection width",
            })?;
    for (field, extents) in [
        ("q_a projection", &[cfg.query_rank, cfg.hidden_size][..]),
        ("q_b projection", &[q_width, cfg.query_rank][..]),
        ("kv_a projection", &[cfg.kv_rank, cfg.hidden_size][..]),
        ("kv_b projection", &[kv_width, cfg.kv_rank][..]),
        ("output projection", &[cfg.hidden_size, v_width][..]),
        ("index query projection", &[index_width, cfg.query_rank][..]),
        (
            "index key projection",
            &[cfg.index_head_dim, cfg.hidden_size][..],
        ),
        (
            "index weights projection",
            &[cfg.index_n_heads, cfg.hidden_size][..],
        ),
        ("index APE", &[cfg.index_kpool, cfg.index_head_dim][..]),
        ("K state per batch row", &[capacity, q_width][..]),
        ("V state per batch row", &[capacity, v_width][..]),
        (
            "indexer state per batch row",
            &[capacity, indexer_lanes][..],
        ),
    ] {
        checked_numel(field, extents)?;
    }
    Ok(DsaDimensions {
        q_width,
        v_width,
        kv_width,
        kv_head_width,
        index_width,
        indexer_lanes,
        selection_width,
    })
}

/// Number of fixed decode-state tensors one DSA block carries.
#[cfg(test)]
pub const GLM5NEXT_DSA_STATE_PAIR_COUNT: usize = 3;

/// Fixed DSA decode state for `batch` rows and `capacity` cache slots, in [`Glm5NextDsaOutput`] order:
/// K, V, then indexer lanes.
#[cfg(test)]
pub(crate) fn glm5next_dsa_state_specs(
    cfg: Glm5NextDsaConfig,
    capacity: usize,
    batch: usize,
) -> Result<[Glm5NextStateSpec; GLM5NEXT_DSA_STATE_PAIR_COUNT], Glm53FlashTraceError> {
    let dims = dsa_dimensions(cfg, capacity)?;
    Ok([
        Glm5NextStateSpec {
            suffix: "k_state",
            ty: TensorType::f32(vec![batch, cfg.num_heads, capacity, cfg.qk_head_dim]),
        },
        Glm5NextStateSpec {
            suffix: "v_state",
            ty: TensorType::f32(vec![batch, cfg.num_heads, capacity, cfg.v_head_dim]),
        },
        Glm5NextStateSpec {
            suffix: "indexer_state",
            ty: TensorType::f32(vec![batch, capacity, dims.indexer_lanes]),
        },
    ])
}

#[cfg(test)]
fn role_shape(cfg: Glm5NextDsaConfig, dims: DsaDimensions, shape: DsaRoleShape) -> Vec<usize> {
    match shape {
        DsaRoleShape::QueryA => vec![cfg.query_rank, cfg.hidden_size],
        DsaRoleShape::QueryRank => vec![cfg.query_rank],
        DsaRoleShape::QueryB => vec![dims.q_width, cfg.query_rank],
        DsaRoleShape::KvA => vec![cfg.kv_rank, cfg.hidden_size],
        DsaRoleShape::KvRank => vec![cfg.kv_rank],
        DsaRoleShape::KvB => vec![dims.kv_width, cfg.kv_rank],
        DsaRoleShape::Output => vec![cfg.hidden_size, dims.v_width],
        DsaRoleShape::IndexQuery => vec![dims.index_width, cfg.query_rank],
        DsaRoleShape::IndexKey | DsaRoleShape::IndexGate => {
            vec![cfg.index_head_dim, cfg.hidden_size]
        }
        DsaRoleShape::IndexWeights => vec![cfg.index_n_heads, cfg.hidden_size],
        DsaRoleShape::IndexWidth => vec![cfg.index_head_dim],
        DsaRoleShape::IndexApe => vec![cfg.index_kpool, cfg.index_head_dim],
    }
}

#[cfg(test)]
pub(crate) fn glm5next_dsa_role_specs(
    revision: &str,
    prefix: &str,
    cfg: Glm5NextDsaConfig,
) -> Result<Vec<Glm5NextDsaRoleSpec>, Glm53FlashTraceError> {
    let dims = preflight_dsa_config(
        revision,
        prefix,
        Glm5NextDsaScope::Dsa,
        cfg,
        cfg.index_kpool,
    )?;
    Ok(DSA_ROLE_ROWS
        .iter()
        .map(
            |&(role, suffix, dtype, source_class, shape)| Glm5NextDsaRoleSpec {
                role,
                name: format!("{prefix}.{suffix}"),
                dtype,
                shape: role_shape(cfg, dims, shape),
                source_class,
            },
        )
        .collect())
}

/// Retain Card 367's exact BF16 owners in canonical role order. Packed projection owners remain Card 368 work.
#[cfg(test)]
pub(crate) fn glm5next_dsa_owner_table(
    loaded: &MixedLoadResult,
    revision: &str,
    prefix: &str,
    cfg: Glm5NextDsaConfig,
    roles: &[Glm5NextDsaRole],
) -> Result<Vec<Glm5NextDsaOwnerBinding>, Glm53FlashTraceError> {
    let specs = glm5next_dsa_role_specs(revision, prefix, cfg)?;
    if loaded.artifact.revision() != revision {
        return Err(Glm53FlashTraceError::DsaRevision {
            expected: GLM53_FLASH_REVISION,
            actual: loaded.artifact.revision().to_string(),
        });
    }
    let mut table = Vec::with_capacity(roles.len());
    for (ordinal, &role) in roles.iter().enumerate() {
        if roles[..ordinal].contains(&role) {
            return Err(Glm53FlashTraceError::DsaDuplicateRole { role });
        }
        let spec = specs
            .iter()
            .find(|spec| spec.role == role)
            .expect("DSA role table covers every Glm5NextDsaRole");
        if spec.source_class != Glm5NextDsaSourceClass::ExactDense {
            return Err(Glm53FlashTraceError::DsaPackedOwnerDeferred { role });
        }
        let source = loaded
            .exact_metadata
            .get(spec.name.as_str())
            .ok_or_else(|| Glm53FlashTraceError::DsaMissingSource {
                name: spec.name.clone(),
            })?;
        if source.kind != ExactSourceKind::Bf16 {
            return Err(Glm53FlashTraceError::DsaSourceKind {
                name: spec.name.clone(),
                expected: ExactSourceKind::Bf16,
                actual: source.kind,
            });
        }
        if source.owner.descriptor().dtype() != "BF16" {
            return Err(Glm53FlashTraceError::DsaSourceDtype {
                name: spec.name.clone(),
                expected: "BF16",
                actual: source.owner.descriptor().dtype().to_string(),
            });
        }
        if source.owner.descriptor().shape() != spec.shape.as_slice() {
            return Err(Glm53FlashTraceError::DsaSourceShape {
                name: spec.name.clone(),
                expected: spec.shape.clone(),
                actual: source.owner.descriptor().shape().to_vec(),
            });
        }
        table.push(Glm5NextDsaOwnerBinding {
            name: spec.name.clone(),
            owner: Arc::clone(&source.owner),
        });
    }
    Ok(table)
}

#[cfg(test)]
fn preflight_value(
    b: &Builder,
    value: Traced,
    role: &'static str,
    dtype: DType,
    shape: Vec<usize>,
) -> Result<(), Glm53FlashTraceError> {
    let actual = b.aval(value);
    if actual.dtype != dtype {
        return Err(Glm53FlashTraceError::DsaDtype {
            role,
            expected: dtype,
            actual: actual.dtype,
        });
    }
    if actual.shape != shape {
        return Err(Glm53FlashTraceError::DsaShape {
            role,
            expected: shape,
            actual: actual.shape,
        });
    }
    Ok(())
}

#[cfg(test)]
fn one_minus(b: &Builder, x: Traced) -> Traced {
    b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, x, Scalar::F32(-1.0)),
        Scalar::F32(1.0),
    )
}

#[cfg(test)]
fn equal(b: &Builder, left: Traced, right: Traced) -> Traced {
    b.binary(
        BinOp::Mul,
        b.binary(BinOp::Ge, left, right),
        b.binary(BinOp::Ge, right, left),
    )
}

#[cfg(test)]
fn clamp_cache_indices(b: &Builder, indices: Traced, capacity: usize) -> (Traced, Traced) {
    let outside = b.binary_scalar(BinOp::Ge, indices, Scalar::F32(capacity as f32));
    let in_range = one_minus(b, outside);
    let over_last = b.binary_scalar(BinOp::Ge, indices, Scalar::F32((capacity - 1) as f32));
    let excess = b.binary_scalar(BinOp::Sub, indices, Scalar::F32((capacity - 1) as f32));
    let clamped = b.binary(BinOp::Sub, indices, b.binary(BinOp::Mul, over_last, excess));
    (clamped, in_range)
}

#[cfg(test)]
fn mask_indices(b: &Builder, indices: Traced, keep: Traced) -> Traced {
    let shifted = b.binary_scalar(BinOp::Add, indices, Scalar::F32(1.0));
    b.binary_scalar(
        BinOp::Sub,
        b.binary(BinOp::Mul, shifted, keep),
        Scalar::F32(1.0),
    )
}

#[cfg(test)]
fn gather_batched_cache_rows(
    b: &Builder,
    data: Traced,
    indices: Traced,
    batch: usize,
    capacity: usize,
    trailing: &[usize],
) -> Traced {
    let index_shape = b.aval(indices).shape[1..].to_vec();
    let rows = (0..batch)
        .map(|row| {
            let mut source_shape = vec![capacity];
            source_shape.extend_from_slice(trailing);
            let source = b.reshape(b.slice(data, 0, row, row + 1), source_shape);
            let index = b.reshape(b.slice(indices, 0, row, row + 1), index_shape.clone());
            let gathered = b.gather(source, 0, index);
            let mut shape = vec![1];
            shape.extend_from_slice(&index_shape);
            shape.extend_from_slice(trailing);
            b.reshape(gathered, shape)
        })
        .collect::<Vec<_>>();
    b.concat(0, &rows)
}

#[cfg(test)]
fn structural_constant(b: &Builder, prefix: &str, name: &str, shape: Vec<usize>) -> Traced {
    b.slot_named(
        Slot::Activation,
        &format!("{prefix}.indexer.card367.{name}"),
        TensorType::f32(shape),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn compose_indexer(
    b: &Builder,
    prefix: &str,
    cfg: Glm5NextDsaConfig,
    capacity: usize,
    batch: usize,
    q_resid: Traced,
    x: Traced,
    query_validity: Traced,
    visibility: Traced,
    position: Traced,
    indexer_state: Traced,
    sources: &[(Glm5NextDsaRole, Traced)],
) -> (Traced, Traced, Traced) {
    let source = |role| {
        sources
            .iter()
            .find_map(|&(candidate, value)| (candidate == role).then_some(value))
            .expect("DSA dense source table covers every exact role")
    };
    let d = cfg.index_head_dim;
    let h = cfg.index_n_heads;
    let pools = capacity / cfg.index_kpool;
    let indexer_lanes = 2 * d + 1;

    let q = dense_source_linear(b, q_resid, source(Glm5NextDsaRole::IndexerWQB));
    let q = b.transpose(b.reshape(q, vec![batch, 1, h, d]), vec![0, 2, 1, 3]);
    let key = dense_source_linear(b, x, source(Glm5NextDsaRole::IndexerWK));
    let key = layernorm(
        b,
        key,
        b.cast(source(Glm5NextDsaRole::IndexerKNormWeight), DType::F32),
        b.cast(source(Glm5NextDsaRole::IndexerKNormBias), DType::F32),
        1.0e-6,
    );
    let gate = dense_source_linear(b, x, source(Glm5NextDsaRole::IndexerCompressGate));
    let current = b.concat(
        2,
        &[key, gate, b.reshape(query_validity, vec![batch, 1, 1])],
    );
    let indexer_state = b.dynamic_update_slice_dyn(indexer_state, current, position, 1);

    let valid = b.reshape(
        b.slice(indexer_state, 2, 2 * d, 2 * d + 1),
        vec![batch, capacity],
    );
    let first_rank = stable_descending_rank(b, valid);
    let first = b.arg_top_k(first_rank, 1);
    let first = b.reshape(first, vec![batch, 1, 1]);
    let offsets = structural_constant(b, prefix, "pool_offsets", vec![1, pools, cfg.index_kpool]);
    let raw_indices = b.binary(BinOp::Add, first, offsets);
    let (safe_indices, in_range) = clamp_cache_indices(b, raw_indices, capacity);
    let rows = gather_batched_cache_rows(
        b,
        indexer_state,
        safe_indices,
        batch,
        capacity,
        &[indexer_lanes],
    );
    let visible = gather_batched_cache_rows(b, visibility, safe_indices, batch, capacity, &[]);
    let lane_valid = b.reshape(
        b.slice(rows, 3, 2 * d, 2 * d + 1),
        vec![batch, pools, cfg.index_kpool],
    );
    let lane_keep = b.binary(BinOp::Mul, lane_valid, in_range);
    let lane_sum = b.reduce(RedOp::Sum, lane_keep, 2, false);
    let complete = b.binary_scalar(BinOp::Ge, lane_sum, Scalar::F32(cfg.index_kpool as f32));
    let pool_end_visible = b.reshape(
        b.slice(visible, 2, cfg.index_kpool - 1, cfg.index_kpool),
        vec![batch, pools],
    );
    let complete = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Mul, complete, pool_end_visible),
        query_validity,
    );

    let keys = b.slice(rows, 3, 0, d);
    let gates = b.slice(rows, 3, d, 2 * d);
    let ape = b.cast(source(Glm5NextDsaRole::IndexerCompressApe), DType::F32);
    let logits = b.binary(BinOp::Add, gates, ape);
    let weights = softmax(b, b.transpose(logits, vec![0, 1, 3, 2]));
    let keys = b.transpose(keys, vec![0, 1, 3, 2]);
    let pooled = b.reduce(RedOp::Sum, b.binary(BinOp::Mul, weights, keys), 3, false);

    let pool_keys = b.transpose(
        b.reshape(pooled, vec![batch, 1, pools, d]),
        vec![0, 1, 3, 2],
    );
    let dots = b.matmul(q, pool_keys);
    let dots = b.binary_scalar(
        BinOp::Mul,
        relu(b, dots),
        Scalar::F32(1.0 / (d as f32).sqrt()),
    );
    let head_weights = dense_source_linear(b, x, source(Glm5NextDsaRole::IndexerWeightsProj));
    let head_weights = b.binary_scalar(
        BinOp::Mul,
        b.reshape(head_weights, vec![batch, h, 1, 1]),
        Scalar::F32(1.0 / (h as f32).sqrt()),
    );
    let weighted = b.binary(BinOp::Mul, dots, head_weights);
    let weighted = b.transpose(weighted, vec![0, 2, 3, 1]);
    let scores = b.reshape(b.reduce(RedOp::Sum, weighted, 3, false), vec![batch, pools]);

    let rank = stable_descending_rank_masked(b, scores, complete);
    let selected_pool_count = cfg.pool_topk().min(pools);
    let selected_pools = b.arg_top_k(rank, selected_pool_count);
    let selected_rows = gather_batched_cache_rows(
        b,
        raw_indices,
        selected_pools,
        batch,
        pools,
        &[cfg.index_kpool],
    );
    let selected_keep = gather_batched_cache_rows(b, complete, selected_pools, batch, pools, &[]);
    let selected_keep = b.broadcast(
        b.reshape(selected_keep, vec![batch, selected_pool_count, 1]),
        vec![batch, selected_pool_count, cfg.index_kpool],
    );
    let selected_rows = mask_indices(b, selected_rows, selected_keep);
    let selected_rows = b.reshape(
        selected_rows,
        vec![batch, selected_pool_count * cfg.index_kpool],
    );

    let selected_padding = cfg.index_topk - selected_pool_count * cfg.index_kpool;
    let selected_rows = if selected_padding == 0 {
        selected_rows
    } else {
        let padding =
            structural_constant(b, prefix, "selection_padding", vec![1, selected_padding]);
        let padding = b.broadcast(padding, vec![batch, selected_padding]);
        b.concat(1, &[selected_rows, padding])
    };

    let complete_count = b.reduce(RedOp::Sum, complete, 1, true);
    let tail_start = b.binary(
        BinOp::Add,
        b.reshape(first, vec![batch, 1]),
        b.binary_scalar(
            BinOp::Mul,
            complete_count,
            Scalar::F32(cfg.index_kpool as f32),
        ),
    );
    let tail_offsets = structural_constant(b, prefix, "tail_offsets", vec![1, cfg.index_kpool - 1]);
    let tail_indices = b.binary(BinOp::Add, tail_start, tail_offsets);
    let (safe_tail, tail_in_range) = clamp_cache_indices(b, tail_indices, capacity);
    let tail_valid = gather_batched_cache_rows(b, valid, safe_tail, batch, capacity, &[]);
    let tail_visible = gather_batched_cache_rows(b, visibility, safe_tail, batch, capacity, &[]);
    let tail_keep = b.binary(
        BinOp::Mul,
        b.binary(
            BinOp::Mul,
            b.binary(BinOp::Mul, tail_valid, tail_visible),
            tail_in_range,
        ),
        query_validity,
    );
    let tail = mask_indices(b, tail_indices, tail_keep);
    let selection = b.reshape(
        b.concat(1, &[selected_rows, tail]),
        vec![
            batch,
            1,
            cfg.checked_selection_width()
                .expect("preflight checked selection width"),
        ],
    );
    (selection, scores, indexer_state)
}

#[cfg(test)]
fn selection_mask(
    b: &Builder,
    prefix: &str,
    selection: Traced,
    valid: Traced,
    visibility: Traced,
    query_validity: Traced,
) -> Traced {
    let selection_shape = b.aval(selection).shape;
    let batch = selection_shape[0];
    let width = selection_shape[2];
    let capacity = b.aval(valid).shape[1];
    let selected = b.reshape(selection, vec![batch, width, 1]);
    let positions = structural_constant(b, prefix, "cache_positions", vec![1, 1, capacity]);
    let selected = b.broadcast(selected, vec![batch, width, capacity]);
    let positions = b.broadcast(positions, vec![batch, width, capacity]);
    let equal = equal(b, selected, positions);
    let nonnegative = b.binary_scalar(BinOp::Ge, selected, Scalar::F32(0.0));
    let equal = b.binary(BinOp::Mul, equal, nonnegative);
    let equal = b.transpose(equal, vec![0, 2, 1]);
    let selected_any = b.reshape(b.reduce(RedOp::Max, equal, 2, true), vec![batch, capacity]);
    let keep = b.binary(
        BinOp::Mul,
        b.binary(
            BinOp::Mul,
            b.binary(BinOp::Mul, selected_any, valid),
            visibility,
        ),
        query_validity,
    );
    let additive = b.binary_scalar(BinOp::Sub, keep, Scalar::F32(1.0));
    let additive = b.binary_scalar(BinOp::Mul, additive, Scalar::F32(-MASK_NEG));
    b.reshape(additive, vec![batch, 1, 1, capacity])
}

/// Compose one GLM5Next NoPE DSA/MLA block. The callback owns only the four projection representations.
///
/// The dense oracle uses [`glm5next_dsa_block_dense`]; packed projection builders can be supplied through this
/// same function. MLA, pooling, masking, attention, and state semantics stay here.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn glm5next_dsa_block_with_projections<F>(
    b: &Builder,
    revision: &str,
    prefix: &str,
    scope: Glm5NextDsaScope,
    cfg: Glm5NextDsaConfig,
    capacity: usize,
    inputs: Glm5NextDsaInputs,
    mut project: F,
) -> Result<(Glm5NextDsaOutput, Glm5NextDsaSelectionEvidence), Glm53FlashTraceError>
where
    F: FnMut(&Builder, Glm5NextDsaRole, Traced) -> Result<Traced, Glm53FlashTraceError>,
{
    let dims = preflight_dsa_config(revision, prefix, scope, cfg, capacity)?;
    let batch = b.aval(inputs.x).shape.first().copied().unwrap_or(0);
    if batch == 0 {
        return Err(Glm53FlashTraceError::DsaDimension {
            field: "batch",
            requirement: "nonzero".to_string(),
            actual: batch,
        });
    }
    for (field, extents) in [
        ("batched input", &[batch, cfg.hidden_size][..]),
        ("batched K state", &[batch, capacity, dims.q_width][..]),
        ("batched V state", &[batch, capacity, dims.v_width][..]),
        (
            "batched indexer state",
            &[batch, capacity, dims.indexer_lanes][..],
        ),
        ("batched selection", &[batch, dims.selection_width][..]),
        (
            "batched pairwise pool rank",
            &[
                batch,
                capacity / cfg.index_kpool,
                capacity / cfg.index_kpool,
            ][..],
        ),
    ] {
        checked_numel(field, extents)?;
    }
    preflight_value(
        b,
        inputs.x,
        "input",
        DType::F32,
        vec![batch, 1, cfg.hidden_size],
    )?;
    preflight_value(
        b,
        inputs.query_validity,
        "query validity",
        DType::F32,
        vec![batch, 1],
    )?;
    preflight_value(
        b,
        inputs.visibility,
        "visibility",
        DType::F32,
        vec![batch, capacity],
    )?;
    preflight_value(b, inputs.position, "position", DType::I32, vec![])?;
    let [k_spec, v_spec, indexer_spec] = glm5next_dsa_state_specs(cfg, capacity, batch)?;
    for (role, value, spec) in [
        ("K state", inputs.k_state, k_spec),
        ("V state", inputs.v_state, v_spec),
        ("indexer state", inputs.indexer_state, indexer_spec),
    ] {
        preflight_value(b, value, role, DType::F32, spec.ty.shape)?;
    }

    let specs = glm5next_dsa_role_specs(revision, prefix, cfg)?;
    let dense_sources = specs
        .iter()
        .filter(|spec| spec.source_class == Glm5NextDsaSourceClass::ExactDense)
        .map(|spec| {
            (
                spec.role,
                b.constant(&spec.name, TensorType::new(spec.shape.clone(), spec.dtype)),
            )
        })
        .collect::<Vec<_>>();
    let source = |role| {
        dense_sources
            .iter()
            .find_map(|&(candidate, value)| (candidate == role).then_some(value))
            .expect("DSA dense source table covers every exact role")
    };

    let x = b.binary(
        BinOp::Mul,
        inputs.x,
        b.reshape(inputs.query_validity, vec![batch, 1, 1]),
    );

    let q_a = project(b, Glm5NextDsaRole::QAProj, x)?;
    preflight_value(
        b,
        q_a,
        "q_a projection",
        DType::F32,
        vec![batch, 1, cfg.query_rank],
    )?;
    let q_resid = rmsnorm(
        b,
        q_a,
        b.cast(source(Glm5NextDsaRole::QALayerNorm), DType::F32),
        1.0e-5,
    );
    let q = project(b, Glm5NextDsaRole::QBProj, q_resid)?;
    preflight_value(
        b,
        q,
        "q_b projection",
        DType::F32,
        vec![batch, 1, dims.q_width],
    )?;
    let q = b.transpose(
        b.reshape(q, vec![batch, 1, cfg.num_heads, cfg.qk_head_dim]),
        vec![0, 2, 1, 3],
    );

    let kv_a = project(b, Glm5NextDsaRole::KVAProjWithMqa, x)?;
    preflight_value(
        b,
        kv_a,
        "kv_a projection",
        DType::F32,
        vec![batch, 1, cfg.kv_rank],
    )?;
    let kv_resid = rmsnorm(
        b,
        kv_a,
        b.cast(source(Glm5NextDsaRole::KVALayerNorm), DType::F32),
        1.0e-5,
    );
    let kv = dense_source_linear(b, kv_resid, source(Glm5NextDsaRole::KVBProj));
    let kv = b.reshape(kv, vec![batch, 1, cfg.num_heads, dims.kv_head_width]);
    let k = b.slice(kv, 3, 0, cfg.qk_head_dim);
    let v = b.slice(kv, 3, cfg.qk_head_dim, dims.kv_head_width);
    let k = b.transpose(k, vec![0, 2, 1, 3]);
    let v = b.transpose(v, vec![0, 2, 1, 3]);
    let k_state = b.dynamic_update_slice_dyn(inputs.k_state, k, inputs.position, 2);
    let v_state = b.dynamic_update_slice_dyn(inputs.v_state, v, inputs.position, 2);

    let (selection, pool_scores, indexer_state) = compose_indexer(
        b,
        prefix,
        cfg,
        capacity,
        batch,
        q_resid,
        x,
        inputs.query_validity,
        inputs.visibility,
        inputs.position,
        inputs.indexer_state,
        &dense_sources,
    );

    let valid = b.reshape(
        b.slice(
            indexer_state,
            2,
            2 * cfg.index_head_dim,
            2 * cfg.index_head_dim + 1,
        ),
        vec![batch, capacity],
    );
    let mask = selection_mask(
        b,
        prefix,
        selection,
        valid,
        inputs.visibility,
        inputs.query_validity,
    );
    let attended = attention_masked(
        b,
        q,
        k_state,
        v_state,
        1,
        1.0 / (cfg.qk_head_dim as f32).sqrt(),
        mask,
    );
    let attended = b.reshape(
        b.transpose(attended, vec![0, 2, 1, 3]),
        vec![batch, 1, dims.v_width],
    );
    let attended = b.binary(
        BinOp::Mul,
        attended,
        b.reshape(inputs.query_validity, vec![batch, 1, 1]),
    );
    let y = project(b, Glm5NextDsaRole::OProj, attended)?;
    preflight_value(
        b,
        y,
        "output projection",
        DType::F32,
        vec![batch, 1, cfg.hidden_size],
    )?;

    Ok((
        Glm5NextDsaOutput {
            y,
            selection,
            pool_scores,
            k_state,
            v_state,
            indexer_state,
        },
        Glm5NextDsaSelectionEvidence {
            capacity,
            lowering: Glm5NextDsaSelectionLowering::BoundedPairwise,
        },
    ))
}

/// Card 367's bounded dense-oracle binding for the four projections.
#[cfg(test)]
pub(crate) fn glm5next_dsa_block_dense(
    b: &Builder,
    revision: &str,
    prefix: &str,
    scope: Glm5NextDsaScope,
    cfg: Glm5NextDsaConfig,
    capacity: usize,
    inputs: Glm5NextDsaInputs,
) -> Result<(Glm5NextDsaOutput, Glm5NextDsaSelectionEvidence), Glm53FlashTraceError> {
    let specs = glm5next_dsa_role_specs(revision, prefix, cfg)?;
    glm5next_dsa_block_with_projections(
        b,
        revision,
        prefix,
        scope,
        cfg,
        capacity,
        inputs,
        |b, role, x| {
            let spec = specs
                .iter()
                .find(|spec| spec.role == role)
                .expect("projection callback receives only canonical projection roles");
            let weight = b.constant(
                &format!("{}.card367_dense_oracle", spec.name),
                TensorType::new(spec.shape.clone(), DType::BF16),
            );
            Ok(dense_source_linear(b, x, weight))
        },
    )
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::fs;

    use poot_eval::exact_dense::DenseOwnerTensorView;
    use poot_eval::{EvalBudget, EvalOptions, Value, eval};
    use poot_graph_ir::{Graph, OpKind, Slot, Storage, ValueId};
    use poot_load::packed_safetensors::{
        AuthenticatedInventory, AuthenticatedSafetensorsHandleSet, ExactSourceOwnerCache,
        InventoryDecision, PackedArtifactManifest, PackedOwnerCache, PackedSafetensorsLimits,
        SourceSpan, TensorDisposition, sha256_digest,
    };
    use poot_tensor::HostTensor;
    use poot_test_util::{assert_close, max_abs_error};
    use serde_json::json;

    use super::*;
    use crate::test_support::safetensors::{SourceRow, TempDir, write_shard};

    const PREFIX: &str = "model.language_model.layers.3.self_attn";
    pub(in crate::architectures::glm53_flash) const CAPACITY: usize = 14;

    pub(in crate::architectures::glm53_flash) fn tiny_config() -> Glm5NextDsaConfig {
        Glm5NextDsaConfig {
            hidden_size: 4,
            query_rank: 3,
            kv_rank: 2,
            num_heads: 2,
            qk_head_dim: 2,
            v_head_dim: 2,
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk: 8,
            index_kpool: 4,
            qk_rope_head_dim: 0,
            mla_use_nope: true,
            index_kpool_always_select_tail: true,
            index_kpool_compress: true,
            indexer_kind: Glm5NextIndexerKind::Full,
        }
    }

    fn role_specs(cfg: Glm5NextDsaConfig) -> Vec<Glm5NextDsaRoleSpec> {
        glm5next_dsa_role_specs(GLM53_FLASH_REVISION, PREFIX, cfg).unwrap()
    }

    /// Every DSA checkpoint tensor below `self_attn`, spelled as the HF checkpoint names it, with its element
    /// count. This is independent of `DSA_ROLE_ROWS`, so a consistent role/name swap there fails the oracle.
    pub(in crate::architectures::glm53_flash) fn checkpoint_tensors(
        cfg: Glm5NextDsaConfig,
    ) -> [(&'static str, usize); 14] {
        let q_width = cfg.num_heads * cfg.qk_head_dim;
        let kv_width = cfg.num_heads * (cfg.qk_head_dim + cfg.v_head_dim);
        let v_width = cfg.num_heads * cfg.v_head_dim;
        let d = cfg.index_head_dim;
        [
            ("q_a_proj.weight", cfg.query_rank * cfg.hidden_size),
            ("q_a_layernorm.weight", cfg.query_rank),
            ("q_b_proj.weight", q_width * cfg.query_rank),
            ("kv_a_proj_with_mqa.weight", cfg.kv_rank * cfg.hidden_size),
            ("kv_a_layernorm.weight", cfg.kv_rank),
            ("kv_b_proj.weight", kv_width * cfg.kv_rank),
            ("o_proj.weight", cfg.hidden_size * v_width),
            (
                "indexer.wq_b.weight",
                cfg.index_n_heads * d * cfg.query_rank,
            ),
            ("indexer.wk.weight", d * cfg.hidden_size),
            (
                "indexer.weights_proj.weight",
                cfg.index_n_heads * cfg.hidden_size,
            ),
            ("indexer.k_norm.weight", d),
            ("indexer.k_norm.bias", d),
            ("indexer.index_kpool_compress_gate", d * cfg.hidden_size),
            ("indexer.index_kpool_compress_ape", cfg.index_kpool * d),
        ]
    }

    /// DSA weights keyed by checkpoint tensor name below `self_attn`. The Card 368 text oracle builds one per
    /// layer and reuses [`reference_step`].
    #[derive(Clone)]
    pub(in crate::architectures::glm53_flash) struct Fixture {
        pub(in crate::architectures::glm53_flash) cfg: Glm5NextDsaConfig,
        pub(in crate::architectures::glm53_flash) weights: BTreeMap<&'static str, Vec<f32>>,
    }

    impl Fixture {
        fn new() -> Self {
            let cfg = tiny_config();
            let weights = checkpoint_tensors(cfg)
                .into_iter()
                .enumerate()
                .map(|(ordinal, (name, len))| {
                    let values = match name {
                        "q_a_layernorm.weight"
                        | "kv_a_layernorm.weight"
                        | "indexer.k_norm.weight" => vec![1.0; len],
                        "indexer.k_norm.bias" => vec![0.125, -0.25],
                        "indexer.index_kpool_compress_ape" => {
                            vec![0.75, -0.5, -0.25, 0.625, 0.5, 0.25, -0.625, -0.375]
                        }
                        // `5 * ordinal` differs modulo 17 for every tensor, so no two tensors share values.
                        _ => (0..len)
                            .map(|index| {
                                let lane = (index * 7 + ordinal * 5) % 17;
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
                .unwrap_or_else(|| panic!("no DSA fixture tensor {name}"))
        }

        pub(in crate::architectures::glm53_flash) fn initial_state(
            &self,
            valid_positions: &[usize],
        ) -> ReferenceState {
            let cfg = self.cfg;
            let mut indexer = vec![0.0; CAPACITY * (2 * cfg.index_head_dim + 1)];
            let mut k = vec![0.0; cfg.num_heads * CAPACITY * cfg.qk_head_dim];
            let mut v = vec![0.0; cfg.num_heads * CAPACITY * cfg.v_head_dim];
            for position in 0..CAPACITY {
                let row = position * (2 * cfg.index_head_dim + 1);
                for lane in 0..cfg.index_head_dim {
                    indexer[row + lane] = (position as f32 + 1.0) * (lane as f32 + 1.0) * 0.071;
                    indexer[row + cfg.index_head_dim + lane] =
                        (((position * 3 + lane * 2) % 11) as f32 - 5.0) * 0.17;
                }
                indexer[row + 2 * cfg.index_head_dim] = if valid_positions.contains(&position) {
                    1.0
                } else {
                    0.0
                };
                for head in 0..cfg.num_heads {
                    for lane in 0..cfg.qk_head_dim {
                        k[(head * CAPACITY + position) * cfg.qk_head_dim + lane] =
                            (1 + head * 3 + position * 2 + lane) as f32 * 0.03125;
                    }
                    for lane in 0..cfg.v_head_dim {
                        v[(head * CAPACITY + position) * cfg.v_head_dim + lane] =
                            (2 + head * 5 + position + lane * 3) as f32 * -0.046875;
                    }
                }
            }
            ReferenceState { k, v, indexer }
        }

        fn graph_inputs(
            &self,
            graph: &Graph,
            input: &[f32],
            query_validity: f32,
            visibility: &[f32],
            position: usize,
            state: &ReferenceState,
        ) -> HashMap<ValueId, Value> {
            self.graph_inputs_batch(graph, input, &[query_validity], visibility, position, state)
        }

        fn graph_inputs_batch(
            &self,
            graph: &Graph,
            input: &[f32],
            query_validity: &[f32],
            visibility: &[f32],
            position: usize,
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
                        Storage::Slot(Slot::Activation) if name.ends_with("input") => {
                            HostTensor::f32(meta.aval.shape.clone(), input.to_vec())
                        }
                        Storage::Slot(Slot::Activation) if name.ends_with("query_validity") => {
                            HostTensor::f32(meta.aval.shape.clone(), query_validity.to_vec())
                        }
                        Storage::Slot(Slot::Mask) => {
                            HostTensor::f32(meta.aval.shape.clone(), visibility.to_vec())
                        }
                        Storage::Slot(Slot::Pos) => {
                            HostTensor::i32(meta.aval.shape.clone(), vec![position as i32])
                        }
                        Storage::State if name.ends_with("k_state") => {
                            HostTensor::f32(meta.aval.shape.clone(), state.k.clone())
                        }
                        Storage::State if name.ends_with("v_state") => {
                            HostTensor::f32(meta.aval.shape.clone(), state.v.clone())
                        }
                        Storage::State if name.ends_with("indexer_state") => {
                            HostTensor::f32(meta.aval.shape.clone(), state.indexer.clone())
                        }
                        Storage::Slot(Slot::Activation) if name.ends_with("pool_offsets") => {
                            HostTensor::f32(
                                meta.aval.shape.clone(),
                                (0..CAPACITY / self.cfg.index_kpool)
                                    .flat_map(|pool| {
                                        (0..self.cfg.index_kpool).map(move |lane| {
                                            (pool * self.cfg.index_kpool + lane) as f32
                                        })
                                    })
                                    .collect(),
                            )
                        }
                        Storage::Slot(Slot::Activation) if name.ends_with("tail_offsets") => {
                            HostTensor::f32(
                                meta.aval.shape.clone(),
                                (0..self.cfg.index_kpool - 1)
                                    .map(|index| index as f32)
                                    .collect(),
                            )
                        }
                        Storage::Slot(Slot::Activation) if name.ends_with("cache_positions") => {
                            HostTensor::f32(
                                meta.aval.shape.clone(),
                                (0..CAPACITY).map(|index| index as f32).collect(),
                            )
                        }
                        Storage::Slot(Slot::Activation) if name.ends_with("selection_padding") => {
                            HostTensor::f32(meta.aval.shape.clone(), vec![-1.0; meta.aval.numel()])
                        }
                        Storage::Const => {
                            let tensor = name
                                .strip_suffix(".card367_dense_oracle")
                                .unwrap_or(name)
                                .strip_prefix(PREFIX)
                                .and_then(|suffix| suffix.strip_prefix('.'))
                                .unwrap_or_else(|| panic!("unexpected DSA constant {name}"));
                            let values = self.weight(tensor);
                            if meta.aval.dtype == DType::BF16 {
                                HostTensor::bf16(
                                    meta.aval.shape.clone(),
                                    values
                                        .iter()
                                        .map(|&f| poot_runtime_common::f32_to_bf16(f))
                                        .collect(),
                                )
                            } else {
                                HostTensor::f32(meta.aval.shape.clone(), values.to_vec())
                            }
                        }
                        _ => panic!("unexpected DSA input {name}"),
                    };
                    (id, tensor.into())
                })
                .collect()
        }
    }

    #[derive(Clone)]
    pub(in crate::architectures::glm53_flash) struct ReferenceState {
        pub(in crate::architectures::glm53_flash) k: Vec<f32>,
        pub(in crate::architectures::glm53_flash) v: Vec<f32>,
        pub(in crate::architectures::glm53_flash) indexer: Vec<f32>,
    }

    #[derive(Clone, Copy)]
    enum PrimaryOutput {
        Y,
        Selection,
        PoolScores,
    }

    fn trace_dsa(
        cfg: Glm5NextDsaConfig,
        capacity: usize,
        batch: usize,
        primary: PrimaryOutput,
    ) -> (Graph, Glm5NextDsaSelectionEvidence) {
        let b = Builder::new();
        let x = b.slot_named(
            Slot::Activation,
            "glm5next_dsa_input",
            TensorType::f32(vec![batch, 1, cfg.hidden_size]),
        );
        let query_validity = b.slot_named(
            Slot::Activation,
            "glm5next_dsa_query_validity",
            TensorType::f32(vec![batch, 1]),
        );
        let visibility = b.slot_named(
            Slot::Mask,
            "glm5next_dsa_visibility",
            TensorType::f32(vec![batch, capacity]),
        );
        let position = b.slot(Slot::Pos, TensorType::scalar(DType::I32));
        let k_state = b.state_input(
            &format!("{PREFIX}.dsa.k_state"),
            TensorType::f32(vec![batch, cfg.num_heads, capacity, cfg.qk_head_dim]),
            StateRole::Recurrent,
        );
        let v_state = b.state_input(
            &format!("{PREFIX}.dsa.v_state"),
            TensorType::f32(vec![batch, cfg.num_heads, capacity, cfg.v_head_dim]),
            StateRole::Recurrent,
        );
        let indexer_state = b.state_input(
            &format!("{PREFIX}.dsa.indexer_state"),
            TensorType::f32(vec![batch, capacity, 2 * cfg.index_head_dim + 1]),
            StateRole::Recurrent,
        );
        let (output, evidence) = glm5next_dsa_block_dense(
            &b,
            GLM53_FLASH_REVISION,
            PREFIX,
            Glm5NextDsaScope::Dsa,
            cfg,
            capacity,
            Glm5NextDsaInputs {
                x,
                query_validity,
                visibility,
                position,
                k_state,
                v_state,
                indexer_state,
            },
        )
        .unwrap();
        let primary = match primary {
            PrimaryOutput::Y => output.y,
            PrimaryOutput::Selection => output.selection,
            PrimaryOutput::PoolScores => output.pool_scores,
        };
        (
            b.finish_with_state(
                primary,
                &[
                    (k_state, output.k_state),
                    (v_state, output.v_state),
                    (indexer_state, output.indexer_state),
                ],
            ),
            evidence,
        )
    }

    fn matvec(weight: &[f32], out: usize, input: &[f32]) -> Vec<f32> {
        let input_width = input.len();
        (0..out)
            .map(|row| {
                (0..input_width)
                    .map(|column| weight[row * input_width + column] * input[column])
                    .sum()
            })
            .collect()
    }

    fn rmsnorm_ref(values: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let denominator =
            (values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32 + eps)
                .sqrt();
        values
            .iter()
            .zip(weight)
            .map(|(value, weight)| value / denominator * weight)
            .collect()
    }

    // LayerNorm with a bias, shared with the other family tests via `crate::reference_ops` (R474-014).
    use crate::reference_ops::layernorm_ref;

    fn softmax_four(logits: [f32; 4]) -> [f32; 4] {
        let max = logits.into_iter().fold(f32::NEG_INFINITY, f32::max);
        let exp = logits.map(|value| (value - max).exp());
        let sum = exp.into_iter().sum::<f32>();
        exp.map(|value| value / sum)
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PoolMutation {
        None,
        OmitApe,
        SoftmaxFeatures,
    }

    struct ReferenceIndexer {
        scores: Vec<f32>,
        complete: Vec<bool>,
        selection: Vec<f32>,
        state: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn indexer_reference(
        fixture: &Fixture,
        input: &[f32],
        query_validity: f32,
        visibility: &[f32],
        position: usize,
        prior: &[f32],
        mutation: PoolMutation,
    ) -> ReferenceIndexer {
        let cfg = fixture.cfg;
        let d = cfg.index_head_dim;
        let h = cfg.index_n_heads;
        let lanes = 2 * d + 1;
        let q_a = matvec(fixture.weight("q_a_proj.weight"), cfg.query_rank, input);
        let q_resid = rmsnorm_ref(&q_a, fixture.weight("q_a_layernorm.weight"), 1.0e-5);
        let q = matvec(fixture.weight("indexer.wq_b.weight"), h * d, &q_resid);
        let raw_key = matvec(fixture.weight("indexer.wk.weight"), d, input);
        let key = layernorm_ref(
            &raw_key,
            fixture.weight("indexer.k_norm.weight"),
            fixture.weight("indexer.k_norm.bias"),
            raw_key.len(),
            1.0e-6,
        );
        let gate = matvec(
            fixture.weight("indexer.index_kpool_compress_gate"),
            d,
            input,
        );
        let head_weights = matvec(fixture.weight("indexer.weights_proj.weight"), h, input);
        let mut state = prior.to_vec();
        let current = position * lanes;
        state[current..current + d].copy_from_slice(&key);
        state[current + d..current + 2 * d].copy_from_slice(&gate);
        state[current + 2 * d] = query_validity;
        let first = (0..CAPACITY)
            .find(|&raw| state[raw * lanes + 2 * d] > 0.5)
            .unwrap_or(0);
        let pools = CAPACITY / cfg.index_kpool;
        let mut scores = vec![0.0; pools];
        let mut complete = vec![false; pools];
        for pool in 0..pools {
            let raw = (0..cfg.index_kpool)
                .map(|lane| first + pool * cfg.index_kpool + lane)
                .collect::<Vec<_>>();
            complete[pool] = query_validity > 0.5
                && raw
                    .iter()
                    .all(|&index| index < CAPACITY && state[index * lanes + 2 * d] > 0.5)
                && raw
                    .last()
                    .is_some_and(|&index| index < CAPACITY && visibility[index] > 0.5);
            if raw.iter().any(|&index| index >= CAPACITY) {
                continue;
            }
            let mut pooled = vec![0.0; d];
            if mutation == PoolMutation::SoftmaxFeatures {
                for (lane, &token) in raw.iter().enumerate() {
                    let logits = (0..d)
                        .map(|feature| {
                            state[token * lanes + d + feature]
                                + fixture.weight("indexer.index_kpool_compress_ape")
                                    [lane * d + feature]
                        })
                        .collect::<Vec<_>>();
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let denom = logits.iter().map(|value| (value - max).exp()).sum::<f32>();
                    for feature in 0..d {
                        pooled[feature] +=
                            (logits[feature] - max).exp() / denom * state[token * lanes + feature];
                    }
                }
            } else {
                for feature in 0..d {
                    let logits = std::array::from_fn(|lane| {
                        let token = raw[lane];
                        state[token * lanes + d + feature]
                            + if mutation == PoolMutation::OmitApe {
                                0.0
                            } else {
                                fixture.weight("indexer.index_kpool_compress_ape")
                                    [lane * d + feature]
                            }
                    });
                    let weights = softmax_four(logits);
                    pooled[feature] = (0..cfg.index_kpool)
                        .map(|lane| weights[lane] * state[raw[lane] * lanes + feature])
                        .sum();
                }
            }
            scores[pool] = (0..h)
                .map(|head| {
                    let dot = (0..d)
                        .map(|feature| q[head * d + feature] * pooled[feature])
                        .sum::<f32>()
                        / (d as f32).sqrt();
                    dot.max(0.0) * head_weights[head] / (h as f32).sqrt()
                })
                .sum();
        }
        let mut order = (0..pools).collect::<Vec<_>>();
        order.sort_by(|&left, &right| {
            complete[right]
                .cmp(&complete[left])
                .then_with(|| scores[right].total_cmp(&scores[left]))
                .then_with(|| left.cmp(&right))
        });
        let selected_pool_count = cfg.pool_topk().min(pools);
        let mut selection = Vec::with_capacity(cfg.checked_selection_width().unwrap());
        for &pool in &order[..selected_pool_count] {
            for lane in 0..cfg.index_kpool {
                selection.push(if complete[pool] {
                    (first + pool * cfg.index_kpool + lane) as f32
                } else {
                    -1.0
                });
            }
        }
        selection.resize(cfg.index_topk, -1.0);
        let tail_start = first + cfg.index_kpool * complete.iter().filter(|&&keep| keep).count();
        for raw in tail_start..tail_start + cfg.index_kpool - 1 {
            selection.push(
                if raw < CAPACITY
                    && query_validity > 0.5
                    && state[raw * lanes + 2 * d] > 0.5
                    && visibility[raw] > 0.5
                {
                    raw as f32
                } else {
                    -1.0
                },
            );
        }
        ReferenceIndexer {
            scores,
            complete,
            selection,
            state,
        }
    }

    pub(in crate::architectures::glm53_flash) struct ReferenceStep {
        pub(in crate::architectures::glm53_flash) output: Vec<f32>,
        selection: Vec<f32>,
        pool_scores: Vec<f32>,
        pub(in crate::architectures::glm53_flash) state: ReferenceState,
    }

    pub(in crate::architectures::glm53_flash) fn reference_step(
        fixture: &Fixture,
        input: &[f32],
        query_validity: f32,
        visibility: &[f32],
        position: usize,
        prior: &ReferenceState,
    ) -> ReferenceStep {
        let cfg = fixture.cfg;
        let masked_input = input
            .iter()
            .map(|value| value * query_validity)
            .collect::<Vec<_>>();
        let q_a = matvec(
            fixture.weight("q_a_proj.weight"),
            cfg.query_rank,
            &masked_input,
        );
        let q_resid = rmsnorm_ref(&q_a, fixture.weight("q_a_layernorm.weight"), 1.0e-5);
        let q = matvec(
            fixture.weight("q_b_proj.weight"),
            cfg.num_heads * cfg.qk_head_dim,
            &q_resid,
        );
        let kv_a = matvec(
            fixture.weight("kv_a_proj_with_mqa.weight"),
            cfg.kv_rank,
            &masked_input,
        );
        let kv_resid = rmsnorm_ref(&kv_a, fixture.weight("kv_a_layernorm.weight"), 1.0e-5);
        let kv = matvec(
            fixture.weight("kv_b_proj.weight"),
            cfg.num_heads * (cfg.qk_head_dim + cfg.v_head_dim),
            &kv_resid,
        );
        let mut state = prior.clone();
        for head in 0..cfg.num_heads {
            for lane in 0..cfg.qk_head_dim {
                state.k[(head * CAPACITY + position) * cfg.qk_head_dim + lane] =
                    kv[head * (cfg.qk_head_dim + cfg.v_head_dim) + lane];
            }
            for lane in 0..cfg.v_head_dim {
                state.v[(head * CAPACITY + position) * cfg.v_head_dim + lane] =
                    kv[head * (cfg.qk_head_dim + cfg.v_head_dim) + cfg.qk_head_dim + lane];
            }
        }
        let indexer = indexer_reference(
            fixture,
            &masked_input,
            query_validity,
            visibility,
            position,
            &prior.indexer,
            PoolMutation::None,
        );
        state.indexer = indexer.state;
        let selected = indexer
            .selection
            .iter()
            .copied()
            .filter(|index| *index >= 0.0)
            .map(|index| index as usize)
            .collect::<BTreeSet<_>>();
        let mut attended = vec![0.0; cfg.num_heads * cfg.v_head_dim];
        if query_validity > 0.5 && !selected.is_empty() {
            for head in 0..cfg.num_heads {
                let scores = selected
                    .iter()
                    .map(|&raw| {
                        (0..cfg.qk_head_dim)
                            .map(|lane| {
                                q[head * cfg.qk_head_dim + lane]
                                    * state.k[(head * CAPACITY + raw) * cfg.qk_head_dim + lane]
                            })
                            .sum::<f32>()
                            / (cfg.qk_head_dim as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp = scores
                    .iter()
                    .map(|score| (score - max).exp())
                    .collect::<Vec<_>>();
                let denom = exp.iter().sum::<f32>();
                for (ordinal, &raw) in selected.iter().enumerate() {
                    for lane in 0..cfg.v_head_dim {
                        attended[head * cfg.v_head_dim + lane] += exp[ordinal] / denom
                            * state.v[(head * CAPACITY + raw) * cfg.v_head_dim + lane];
                    }
                }
            }
        }
        let output = matvec(fixture.weight("o_proj.weight"), cfg.hidden_size, &attended);
        ReferenceStep {
            output,
            selection: indexer.selection,
            pool_scores: indexer.scores,
            state,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn graph_step(
        graph: &Graph,
        fixture: &Fixture,
        primary: PrimaryOutput,
        input: &[f32],
        query_validity: f32,
        visibility: &[f32],
        position: usize,
        state: &ReferenceState,
    ) -> ReferenceStep {
        let inputs =
            fixture.graph_inputs(graph, input, query_validity, visibility, position, state);
        let result = eval(graph, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let output = result.output.into_host().expect("dense output");
        let states: Vec<HostTensor> = result
            .state
            .into_iter()
            .map(|v| v.into_host().expect("dense state"))
            .collect();
        assert_eq!(states.len(), 3);
        ReferenceStep {
            output: matches!(primary, PrimaryOutput::Y)
                .then(|| output.as_f32().unwrap().to_vec())
                .unwrap_or_default(),
            selection: matches!(primary, PrimaryOutput::Selection)
                .then(|| output.as_f32().unwrap().to_vec())
                .unwrap_or_default(),
            pool_scores: matches!(primary, PrimaryOutput::PoolScores)
                .then(|| output.as_f32().unwrap().to_vec())
                .unwrap_or_default(),
            state: ReferenceState {
                k: states[0].as_f32().unwrap().to_vec(),
                v: states[1].as_f32().unwrap().to_vec(),
                indexer: states[2].as_f32().unwrap().to_vec(),
            },
        }
    }

    #[test]
    fn glm5next_nope_mla_matches_selected_attention() {
        let mut fixture = Fixture::new();
        fixture.cfg.index_topk = 4;
        let (graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::Y);
        graph.validate().unwrap();
        let state = fixture.initial_state(&(2..=10).collect::<Vec<_>>());
        let input = [0.75, -0.25, 0.5, 0.125];
        let visibility = (0..CAPACITY)
            .map(|position| if position <= 11 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let actual = graph_step(
            &graph,
            &fixture,
            PrimaryOutput::Y,
            &input,
            1.0,
            &visibility,
            11,
            &state,
        );
        let expected = reference_step(&fixture, &input, 1.0, &visibility, 11, &state);
        assert_close(&actual.output, &expected.output, 4.0e-5);
        assert_close(&actual.state.k, &expected.state.k, 3.0e-5);
        assert_close(&actual.state.v, &expected.state.v, 3.0e-5);
        let selected = expected
            .selection
            .iter()
            .copied()
            .filter(|&index| index >= 0.0)
            .map(|index| index as usize)
            .collect::<BTreeSet<_>>();
        assert!(selected.len() < 10);
        assert!((2..=11).any(|position| !selected.contains(&position)));
        assert!(expected.output.iter().any(|value| value.abs() > 1.0e-5));

        assert!(!graph.values.iter().any(|meta| {
            meta.name
                .as_deref()
                .is_some_and(|name| name.contains("rope") || name.contains("rotary"))
        }));
    }

    #[test]
    fn glm5next_indexer_pool_compression_matches_reference() {
        let fixture = Fixture::new();
        let (graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::PoolScores);
        let state = fixture.initial_state(&(2..=10).collect::<Vec<_>>());
        let input = [0.375, -0.625, 0.75, 0.25];
        let visibility = vec![1.0; CAPACITY];
        let actual = graph_step(
            &graph,
            &fixture,
            PrimaryOutput::PoolScores,
            &input,
            1.0,
            &visibility,
            11,
            &state,
        );
        let expected = indexer_reference(
            &fixture,
            &input,
            1.0,
            &visibility,
            11,
            &state.indexer,
            PoolMutation::None,
        );
        let no_ape = indexer_reference(
            &fixture,
            &input,
            1.0,
            &visibility,
            11,
            &state.indexer,
            PoolMutation::OmitApe,
        );
        let feature_softmax = indexer_reference(
            &fixture,
            &input,
            1.0,
            &visibility,
            11,
            &state.indexer,
            PoolMutation::SoftmaxFeatures,
        );
        assert_close(&actual.pool_scores, &expected.scores, 3.0e-5);
        assert!(max_abs_error(&actual.pool_scores, &no_ape.scores) > 1.0e-5);
        assert!(max_abs_error(&actual.pool_scores, &feature_softmax.scores) > 1.0e-5);
        assert_eq!(expected.complete, vec![true, true, false]);
    }

    #[test]
    fn glm5next_indexer_first_valid_and_tail_are_exact() {
        let fixture = Fixture::new();
        let (graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::Selection);
        let input = [0.5, -0.125, 0.625, -0.375];
        let visibility = vec![1.0; CAPACITY];
        for valid_positions in [
            (2..=10).collect::<Vec<_>>(),
            (3..=11).collect::<Vec<_>>(),
            Vec::new(),
            vec![5, 6, 7],
        ] {
            let position = valid_positions.last().copied().unwrap_or(0);
            let query_validity = if valid_positions.is_empty() { 0.0 } else { 1.0 };
            let state = fixture.initial_state(&valid_positions);
            let actual = graph_step(
                &graph,
                &fixture,
                PrimaryOutput::Selection,
                &input,
                query_validity,
                &visibility,
                position,
                &state,
            );
            let expected = indexer_reference(
                &fixture,
                &input
                    .iter()
                    .map(|value| value * query_validity)
                    .collect::<Vec<_>>(),
                query_validity,
                &visibility,
                position,
                &state.indexer,
                PoolMutation::None,
            );
            assert_eq!(actual.selection, expected.selection);
        }
        let state = fixture.initial_state(&(2..=10).collect::<Vec<_>>());
        let expected = indexer_reference(
            &fixture,
            &input,
            1.0,
            &visibility,
            10,
            &state.indexer,
            PoolMutation::None,
        );
        assert_eq!(&expected.selection[8..], &[10.0, -1.0, -1.0]);

        let (batched, _) = trace_dsa(fixture.cfg, CAPACITY, 2, PrimaryOutput::Selection);
        let left = fixture.initial_state(&(2..=10).collect::<Vec<_>>());
        let right = fixture.initial_state(&(3..=10).collect::<Vec<_>>());
        let batched_state = ReferenceState {
            k: [left.k.clone(), right.k.clone()].concat(),
            v: [left.v.clone(), right.v.clone()].concat(),
            indexer: [left.indexer.clone(), right.indexer.clone()].concat(),
        };
        let batched_input = [input, [-0.25, 0.75, 0.125, 0.5]].concat();
        let batched_visibility = vec![1.0; 2 * CAPACITY];
        let inputs = fixture.graph_inputs_batch(
            &batched,
            &batched_input,
            &[1.0, 1.0],
            &batched_visibility,
            11,
            &batched_state,
        );
        let batched_output = eval(&batched, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("dense output");
        let row_visibility = vec![1.0; CAPACITY];
        let left_expected = indexer_reference(
            &fixture,
            &input,
            1.0,
            &row_visibility,
            11,
            &left.indexer,
            PoolMutation::None,
        );
        let right_expected = indexer_reference(
            &fixture,
            &[-0.25, 0.75, 0.125, 0.5],
            1.0,
            &row_visibility,
            11,
            &right.indexer,
            PoolMutation::None,
        );
        let batched_expected = [left_expected.selection, right_expected.selection].concat();
        assert_eq!(
            batched_output.as_f32().unwrap(),
            batched_expected.as_slice()
        );
    }

    #[test]
    fn glm5next_indexer_visibility_is_causal() {
        let mut fixture = Fixture::new();
        fixture.cfg.index_topk = 12;
        let (graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::Selection);
        let state = fixture.initial_state(&(2..=13).collect::<Vec<_>>());
        let input = [0.625, 0.25, -0.5, 0.875];
        let mut visibility = vec![1.0; CAPACITY];
        visibility[11..].fill(0.0);
        let actual = graph_step(
            &graph,
            &fixture,
            PrimaryOutput::Selection,
            &input,
            1.0,
            &visibility,
            10,
            &state,
        );
        let expected = indexer_reference(
            &fixture,
            &input,
            1.0,
            &visibility,
            10,
            &state.indexer,
            PoolMutation::None,
        );
        assert_eq!(actual.selection, expected.selection);
        assert!(
            actual
                .selection
                .iter()
                .all(|&index| index < 0.0 || index <= 10.0)
        );
        assert!(expected.complete[0]);
        assert!(expected.complete[1]);
        assert!(!expected.complete[2]);
        assert_eq!(&actual.selection[8..12], &[-1.0; 4]);
        assert_eq!(&actual.selection[12..], &[10.0, -1.0, -1.0]);
    }

    #[test]
    fn glm5next_indexer_state_lane_order_is_exact() {
        let fixture = Fixture::new();
        let (graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::Selection);
        let state = fixture.initial_state(&(2..=6).collect::<Vec<_>>());
        let input = [0.25, 0.75, -0.5, 0.375];
        let visibility = vec![1.0; CAPACITY];
        let actual = graph_step(
            &graph,
            &fixture,
            PrimaryOutput::Selection,
            &input,
            1.0,
            &visibility,
            7,
            &state,
        );
        let expected = indexer_reference(
            &fixture,
            &input,
            1.0,
            &visibility,
            7,
            &state.indexer,
            PoolMutation::None,
        );
        assert_close(&actual.state.indexer, &expected.state, 3.0e-5);
        let lanes = 2 * fixture.cfg.index_head_dim + 1;
        let row = &actual.state.indexer[7 * lanes..8 * lanes];
        assert_ne!(
            &row[..fixture.cfg.index_head_dim],
            &row[fixture.cfg.index_head_dim..4]
        );
        assert_eq!(row[4], 1.0);
        assert_eq!(
            &actual.state.indexer[..7 * lanes],
            &state.indexer[..7 * lanes]
        );
    }

    #[test]
    fn glm5next_dsa_fixed_state_is_explicit() {
        let cfg = Glm5NextDsaConfig::exact();
        let (graph, evidence) = trace_dsa(cfg, 12, 2, PrimaryOutput::Y);
        graph.validate().unwrap();
        assert_eq!(graph.state.len(), 3);
        let expected = [
            TensorType::f32(vec![2, 64, 12, 256]),
            TensorType::f32(vec![2, 64, 12, 256]),
            TensorType::f32(vec![2, 12, 257]),
        ];
        for ((input, output), expected) in graph.state.iter().zip(expected) {
            assert_eq!(graph.aval(*input), &expected);
            assert_eq!(graph.aval(*output), &expected);
            assert_ne!(input, output);
        }
        assert_eq!(
            graph.aval(graph.output),
            &TensorType::f32(vec![2, 1, 4_096])
        );
        let (selection_graph, _) = trace_dsa(cfg, 12, 2, PrimaryOutput::Selection);
        assert_eq!(
            selection_graph.aval(selection_graph.output),
            &TensorType::f32(vec![2, 1, 2_051])
        );
        assert!(!evidence.production_capacity_admitted());
        assert!(!graph.eqns.iter().any(|eqn| {
            matches!(
                eqn.op,
                OpKind::PackedDequant { .. } | OpKind::PackedContraction { .. }
            )
        }));
        assert!(!graph.values.iter().any(|meta| {
            meta.name
                .as_deref()
                .is_some_and(|name| name.contains("rope") || name.contains("rotary"))
        }));

        let fixture = Fixture::new();
        let (tiny, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::Y);
        let state = fixture.initial_state(&(2..=6).collect::<Vec<_>>());
        let visibility = vec![1.0; CAPACITY];
        let first = graph_step(
            &tiny,
            &fixture,
            PrimaryOutput::Y,
            &[0.5, -0.25, 0.75, 0.125],
            1.0,
            &visibility,
            7,
            &state,
        );
        let second = graph_step(
            &tiny,
            &fixture,
            PrimaryOutput::Y,
            &[-0.375, 0.625, 0.25, -0.5],
            1.0,
            &visibility,
            8,
            &first.state,
        );
        let expected_first = reference_step(
            &fixture,
            &[0.5, -0.25, 0.75, 0.125],
            1.0,
            &visibility,
            7,
            &state,
        );
        let expected_second = reference_step(
            &fixture,
            &[-0.375, 0.625, 0.25, -0.5],
            1.0,
            &visibility,
            8,
            &expected_first.state,
        );
        assert_close(&first.state.k, &expected_first.state.k, 3.0e-5);
        assert_close(&first.state.v, &expected_first.state.v, 3.0e-5);
        assert_close(&first.state.indexer, &expected_first.state.indexer, 3.0e-5);
        assert_close(&second.state.k, &expected_second.state.k, 3.0e-5);
        assert_close(&second.state.v, &expected_second.state.v, 3.0e-5);
        assert_close(
            &second.state.indexer,
            &expected_second.state.indexer,
            3.0e-5,
        );
        assert_ne!(state.k, first.state.k);
        assert_ne!(first.state.k, second.state.k);
        assert_ne!(state.v, first.state.v);
        assert_ne!(first.state.v, second.state.v);
        assert_ne!(state.indexer, first.state.indexer);
        assert_ne!(first.state.indexer, second.state.indexer);
        let k_row = fixture.cfg.qk_head_dim;
        for head in 0..fixture.cfg.num_heads {
            for position in 0..CAPACITY {
                if position == 7 || position == 8 {
                    continue;
                }
                let start = (head * CAPACITY + position) * k_row;
                assert_eq!(
                    &second.state.k[start..start + k_row],
                    &state.k[start..start + k_row]
                );
            }
        }
        let indexer_row = 2 * fixture.cfg.index_head_dim + 1;
        for position in 0..CAPACITY {
            if position == 7 || position == 8 {
                continue;
            }
            let start = position * indexer_row;
            assert_eq!(
                &second.state.indexer[start..start + indexer_row],
                &state.indexer[start..start + indexer_row]
            );
        }
    }

    #[test]
    fn glm5next_indexer_tied_scores_prefer_lower_pool() {
        let mut fixture = Fixture::new();
        fixture.cfg.index_topk = 8;
        fixture.weights.insert(
            "indexer.weights_proj.weight",
            vec![0.0; fixture.cfg.index_n_heads * fixture.cfg.hidden_size],
        );
        let state = fixture.initial_state(&(0..=12).collect::<Vec<_>>());
        let input = [0.625, -0.375, 0.25, 0.75];
        let visibility = vec![1.0; CAPACITY];

        let (score_graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::PoolScores);
        let scored = graph_step(
            &score_graph,
            &fixture,
            PrimaryOutput::PoolScores,
            &input,
            1.0,
            &visibility,
            13,
            &state,
        );
        assert_eq!(scored.pool_scores, vec![0.0, 0.0, 0.0]);

        let (selection_graph, _) = trace_dsa(fixture.cfg, CAPACITY, 1, PrimaryOutput::Selection);
        let selected = graph_step(
            &selection_graph,
            &fixture,
            PrimaryOutput::Selection,
            &input,
            1.0,
            &visibility,
            13,
            &state,
        );
        assert_eq!(
            selected.selection,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0, 13.0, -1.0]
        );
    }

    #[test]
    fn glm5next_pairwise_rank_is_bounded_semantic_only() {
        let cfg = tiny_config();
        let (_, evidence) = trace_dsa(
            cfg,
            GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT,
            1,
            PrimaryOutput::Selection,
        );
        assert_eq!(evidence.capacity, GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT);
        assert_eq!(
            evidence.lowering,
            Glm5NextDsaSelectionLowering::BoundedPairwise
        );
        assert!(!evidence.production_capacity_admitted());
        assert!(matches!(
            preflight_dsa_config(
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextDsaScope::Dsa,
                cfg,
                GLM5NEXT_DSA_PAIRWISE_CAPACITY_LIMIT + 1,
            ),
            Err(Glm53FlashTraceError::DsaPairwiseCapacity { .. })
        ));
    }

    #[test]
    fn glm5next_dsa_manifest_rows_are_exact() {
        let manifest: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../../poot-load/src/glm53_flash_data/audit-manifest.json"
        ))
        .unwrap();
        let raw_index: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../../poot-load/src/glm53_flash_data/raw-index.json"
        ))
        .unwrap();
        let config: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../../poot-load/src/glm53_flash_data/config.json"
        ))
        .unwrap();
        let specs = glm5next_dsa_role_specs(
            GLM53_FLASH_REVISION,
            "model.language_model.layers.{layer}.self_attn",
            Glm5NextDsaConfig::exact(),
        )
        .unwrap();
        assert_eq!(specs.len(), 14);
        assert_eq!(
            specs
                .iter()
                .map(|spec| spec.role)
                .collect::<BTreeSet<_>>()
                .len(),
            14
        );
        let produced = specs
            .iter()
            .map(|spec| (spec.name.clone(), spec.shape.iter().product::<usize>()))
            .collect::<BTreeSet<_>>();
        let literal = checkpoint_tensors(Glm5NextDsaConfig::exact()).map(|(name, len)| {
            (
                format!("model.language_model.layers.{{layer}}.self_attn.{name}"),
                len,
            )
        });
        assert_eq!(produced, BTreeSet::from(literal));
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.source_class == Glm5NextDsaSourceClass::PackedProjection)
                .count(),
            4
        );
        let families = manifest["tensor_families"].as_array().unwrap();
        for spec in &specs {
            let family = families
                .iter()
                .find(|family| family["pattern"].as_str() == Some(spec.name.as_str()))
                .unwrap_or_else(|| panic!("missing committed DSA family {}", spec.name));
            let variant = family["variants"]
                .as_array()
                .unwrap()
                .iter()
                .find(|variant| {
                    variant["dtype"].as_str()
                        == Some(match spec.dtype {
                            DType::E4M3FN => "F8_E4M3",
                            DType::BF16 => "BF16",
                            _ => unreachable!(),
                        })
                        && variant["count"].as_u64() == Some(12)
                })
                .unwrap_or_else(|| panic!("missing committed DSA variant {}", spec.name));
            let shape = variant["physical_shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(shape, spec.shape, "committed shape for {}", spec.name);
            match spec.source_class {
                Glm5NextDsaSourceClass::PackedProjection => {
                    assert!(variant["pair_pattern"].as_str().is_some());
                    assert_eq!(variant["block_shape"], json!([128, 128]));
                }
                Glm5NextDsaSourceClass::ExactDense => {
                    assert!(variant["pair_pattern"].is_null());
                    assert!(variant["block_shape"].is_null());
                }
            }
        }

        let dsa_layers = config["text_config"]["layer_types"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(layer, kind)| {
                (kind.as_str() == Some("deepseek_sparse_attention")).then_some(layer)
            })
            .collect::<Vec<_>>();
        assert_eq!(dsa_layers, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
        assert_eq!(config["text_config"]["qk_rope_head_dim"], 0);
        assert_eq!(config["text_config"]["mla_use_nope"], true);
        assert!(
            config["text_config"]["indexer_types"]
                .as_array()
                .unwrap()
                .iter()
                .take(45)
                .all(|kind| kind.as_str() == Some("full"))
        );
        let weight_map = raw_index["weight_map"].as_object().unwrap();
        for layer in dsa_layers {
            for spec in glm5next_dsa_role_specs(
                GLM53_FLASH_REVISION,
                &format!("model.language_model.layers.{layer}.self_attn"),
                Glm5NextDsaConfig::exact(),
            )
            .unwrap()
            {
                assert!(weight_map.contains_key(&spec.name), "missing {}", spec.name);
            }
        }
    }

    fn classify_dense_fixture(
        inventory: AuthenticatedInventory<'_>,
    ) -> Result<Vec<InventoryDecision>, &'static str> {
        inventory
            .rows()
            .map(|row| {
                Ok(InventoryDecision::new(
                    row.key(),
                    TensorDisposition::DenseBf16,
                ))
            })
            .collect()
    }

    #[test]
    fn glm5next_dsa_synthetic_owner_handoff_is_zero_copy() {
        let cfg = tiny_config();
        let specs = role_specs(cfg);
        let selected_roles = [
            Glm5NextDsaRole::IndexerWK,
            Glm5NextDsaRole::IndexerKNormBias,
            Glm5NextDsaRole::IndexerCompressApe,
        ];
        let rows = selected_roles
            .iter()
            .enumerate()
            .map(|(ordinal, role)| {
                let spec = specs.iter().find(|spec| spec.role == *role).unwrap();
                SourceRow {
                    name: spec.name.clone(),
                    dtype: "BF16",
                    shape: spec.shape.clone(),
                    bytes: (0..spec.shape.iter().product::<usize>() * 2)
                        .map(|byte| (ordinal as u8 * 71).wrapping_add(byte as u8))
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
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
            .load_mixed(&mut packed_cache, &mut exact_cache, classify_dense_fixture)
            .unwrap();
        let table =
            glm5next_dsa_owner_table(&loaded, GLM53_FLASH_REVISION, PREFIX, cfg, &selected_roles)
                .unwrap();
        assert_eq!(table.len(), selected_roles.len());
        assert_eq!(loaded.exact_metadata.len(), selected_roles.len());
        assert_eq!(exact_cache.len(), selected_roles.len());
        assert!(matches!(
            glm5next_dsa_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextDsaRole::IndexerWK, Glm5NextDsaRole::IndexerWK],
            ),
            Err(Glm53FlashTraceError::DsaDuplicateRole {
                role: Glm5NextDsaRole::IndexerWK
            })
        ));
        assert!(matches!(
            glm5next_dsa_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextDsaRole::IndexerWQB],
            ),
            Err(Glm53FlashTraceError::DsaMissingSource { .. })
        ));
        let mut wrong_kind = loaded.clone();
        wrong_kind
            .exact_metadata
            .get_mut(rows[0].name.as_str())
            .unwrap()
            .kind = ExactSourceKind::F32;
        assert!(matches!(
            glm5next_dsa_owner_table(
                &wrong_kind,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextDsaRole::IndexerWK],
            ),
            Err(Glm53FlashTraceError::DsaSourceKind { .. })
        ));
        assert!(matches!(
            glm5next_dsa_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextDsaConfig {
                    hidden_size: cfg.hidden_size + 1,
                    ..cfg
                },
                &[Glm5NextDsaRole::IndexerWK],
            ),
            Err(Glm53FlashTraceError::DsaSourceShape { .. })
        ));
        let views = table
            .iter()
            .map(|binding| DenseOwnerTensorView::new(Arc::clone(binding.owner())).unwrap())
            .collect::<Vec<_>>();
        for ((binding, view), row) in table.iter().zip(&views).zip(&rows) {
            let loaded_source = loaded.exact_metadata.get(binding.name()).unwrap();
            assert!(Arc::ptr_eq(binding.owner(), &loaded_source.owner));
            assert!(Arc::ptr_eq(binding.owner(), view.owner()));
            assert_eq!(binding.owner().bytes(), row.bytes.as_slice());
            assert_eq!(
                binding.owner().bytes().as_ptr(),
                view.owner().bytes().as_ptr()
            );
            assert_eq!(view.dtype(), DType::BF16);
            assert_eq!(view.shape(), row.shape.as_slice());
        }
        assert_eq!(table[0].owner().descriptor().span(), SourceSpan::new(0, 16));

        let (graph, _) = trace_dsa(cfg, CAPACITY, 1, PrimaryOutput::Y);
        let dense_rows = table
            .iter()
            .zip(&views)
            .map(|(binding, view)| {
                let value = graph
                    .consts
                    .iter()
                    .copied()
                    .find(|&value| graph.meta(value).name.as_deref() == Some(binding.name()))
                    .unwrap();
                (value, view.clone())
            })
            .collect::<Vec<_>>();
        let bound: HashMap<ValueId, Value> = dense_rows
            .iter()
            .map(|(value, view)| (*value, Value::from(view.clone())))
            .collect();
        for (value, view) in &dense_rows {
            // Exact-value equality is owner `Arc` identity plus logical shape, and each view
            // already shares its table binding's owner above.
            assert_eq!(bound[value], Value::from(view.clone()));
        }
        assert!(packed_cache.is_empty());
        assert!(matches!(
            glm5next_dsa_owner_table(
                &loaded,
                GLM53_FLASH_REVISION,
                PREFIX,
                cfg,
                &[Glm5NextDsaRole::QAProj],
            ),
            Err(Glm53FlashTraceError::DsaPackedOwnerDeferred { .. })
        ));
    }

    #[test]
    fn glm5next_dsa_typed_preflight_rejects_mismatched_contracts() {
        let cfg = tiny_config();
        for scope in [
            Glm5NextDsaScope::ImageVideo,
            Glm5NextDsaScope::Mtp,
            Glm5NextDsaScope::Kda,
            Glm5NextDsaScope::Mhc,
            Glm5NextDsaScope::Ffn,
        ] {
            assert!(matches!(
                preflight_dsa_config(GLM53_FLASH_REVISION, PREFIX, scope, cfg, CAPACITY),
                Err(Glm53FlashTraceError::DsaUnsupportedScope { .. })
            ));
        }
        assert!(matches!(
            preflight_dsa_config("wrong", PREFIX, Glm5NextDsaScope::Dsa, cfg, CAPACITY),
            Err(Glm53FlashTraceError::DsaRevision { .. })
        ));
        assert!(matches!(
            preflight_dsa_config(
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextDsaScope::Dsa,
                Glm5NextDsaConfig {
                    qk_rope_head_dim: 2,
                    ..cfg
                },
                CAPACITY,
            ),
            Err(Glm53FlashTraceError::DsaRopeContract { .. })
        ));
        assert!(matches!(
            preflight_dsa_config(
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextDsaScope::Dsa,
                Glm5NextDsaConfig {
                    indexer_kind: Glm5NextIndexerKind::Shared,
                    ..cfg
                },
                CAPACITY,
            ),
            Err(Glm53FlashTraceError::DsaIndexerContract { .. })
        ));
        assert!(matches!(
            preflight_dsa_config(
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextDsaScope::Dsa,
                Glm5NextDsaConfig {
                    hidden_size: usize::MAX,
                    query_rank: 2,
                    ..cfg
                },
                CAPACITY,
            ),
            Err(Glm53FlashTraceError::DsaShapeOverflow { .. })
        ));

        let b = Builder::new();
        let malformed = Glm5NextDsaInputs {
            x: b.slot_named(
                Slot::Activation,
                "malformed_dsa_input",
                TensorType::new(vec![1, 1, cfg.hidden_size], DType::BF16),
            ),
            query_validity: b.slot(Slot::Activation, TensorType::f32(vec![1, 1])),
            visibility: b.slot(Slot::Mask, TensorType::f32(vec![1, CAPACITY])),
            position: b.slot(Slot::Pos, TensorType::scalar(DType::I32)),
            k_state: b.state_input(
                "malformed_dsa_k",
                TensorType::f32(vec![1, cfg.num_heads, CAPACITY, cfg.qk_head_dim]),
                StateRole::Recurrent,
            ),
            v_state: b.state_input(
                "malformed_dsa_v",
                TensorType::f32(vec![1, cfg.num_heads, CAPACITY, cfg.v_head_dim]),
                StateRole::Recurrent,
            ),
            indexer_state: b.state_input(
                "malformed_dsa_indexer",
                TensorType::f32(vec![1, CAPACITY, 2 * cfg.index_head_dim + 1]),
                StateRole::Recurrent,
            ),
        };
        assert!(matches!(
            glm5next_dsa_block_dense(
                &b,
                GLM53_FLASH_REVISION,
                PREFIX,
                Glm5NextDsaScope::Dsa,
                cfg,
                CAPACITY,
                malformed,
            ),
            Err(Glm53FlashTraceError::DsaDtype { role: "input", .. })
        ));
    }
}

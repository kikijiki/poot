//! One normalized disposition per authenticated MiniMax-M2 checkpoint name.
//!
//! [`MiniMaxM2SourceTable`] names every source of one configuration. This module turns that table
//! into a manifest: a name-keyed map whose every entry carries the card 359 [`TensorDisposition`]
//! the loader must assign, plus the dtype and shape the shard header has to present for it.
//!
//! A name is classified by the position its source occupies in the table, never by its suffix
//! (`model.norm.weight` ends in `.weight` and is a dense BF16 row). Classification is total:
//! omission, duplication and double disposition are admission errors, so the 96,103 names of the
//! pinned revision partition exactly.

use std::collections::{BTreeMap, BTreeSet};

use poot_quant::{OperandRole, SourceRole};

use super::{
    MINIMAX_M25_PACKED_FORMAT, MiniMaxM2AttentionProjection, MiniMaxM2Config, MiniMaxM2DenseRole,
    MiniMaxM2DenseSource, MiniMaxM2ExpertProjection, MiniMaxM2LayerSources,
    MiniMaxM2PackedComponent, MiniMaxM2PackedSource, MiniMaxM2SourceTable,
    MiniMaxM2SourceTableError,
};
use crate::packed_safetensors::{AuthenticatedInventory, InventoryDecision, TensorDisposition};

/// The pinned namespace one authenticated name belongs to.
///
/// The four classes spec 365 counts. A name outside them is not a MiniMax-M2 text source and is
/// rejected; this revision publishes no vision, MTP or auxiliary namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiniMaxM2Namespace {
    /// One component of an attention Q/K/V/O projection.
    AttentionPacked,
    /// One component of a routed expert `w1`/`w2`/`w3`.
    ExpertPacked,
    /// A router gate matrix or its selection-only correction bias, F32.
    RouterDense,
    /// The embedding, the untied LM head, or any RMSNorm weight, BF16.
    Bf16Dense,
}

impl MiniMaxM2Namespace {
    /// The dense role this namespace stores, or `None` for the two packed classes.
    const fn dense_role(self) -> Option<MiniMaxM2DenseRole> {
        match self {
            Self::AttentionPacked | Self::ExpertPacked => None,
            Self::RouterDense => Some(MiniMaxM2DenseRole::F32),
            Self::Bf16Dense => Some(MiniMaxM2DenseRole::Bf16),
        }
    }
}

/// One authenticated name's exact disposition and the header facts it must present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniMaxM2ManifestEntry {
    namespace: MiniMaxM2Namespace,
    disposition: TensorDisposition,
    dtype: &'static str,
    shape: Vec<usize>,
}

impl MiniMaxM2ManifestEntry {
    pub const fn namespace(&self) -> MiniMaxM2Namespace {
        self.namespace
    }

    pub const fn disposition(&self) -> &TensorDisposition {
        &self.disposition
    }

    /// The safetensors dtype string the shard header must carry for this name.
    pub const fn dtype(&self) -> &'static str {
        self.dtype
    }

    /// The shape the shard header must carry: the physical E4M3 grid for a packed weight, the block
    /// grid for a packed scale, and the logical shape for a dense row.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The linear id and component behind a packed entry.
    ///
    /// The checkpoint spelling comes from [`MiniMaxM2PackedComponent::checkpoint_suffix`]; the graph
    /// spelling is `poot_graph_ir::PackedSourceName::new(linear_id, component.source_role())` (card
    /// 379).
    pub fn packed_component(&self) -> Option<(&str, MiniMaxM2PackedComponent)> {
        match &self.disposition {
            TensorDisposition::PackedWeight { linear_id, .. } => {
                Some((linear_id.as_str(), MiniMaxM2PackedComponent::Weight))
            }
            TensorDisposition::PackedScale { linear_id } => {
                Some((linear_id.as_str(), MiniMaxM2PackedComponent::Scale))
            }
            _ => None,
        }
    }
}

/// The counted partition of one manifest.
///
/// Filled by checked increments while the rows are walked, then compared against
/// [`MiniMaxM2ManifestReport::expected`], which derives the same numbers from the configuration
/// alone, so a derivation defect and a constant typo cannot cancel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MiniMaxM2ManifestReport {
    pub total_names: usize,
    pub attention_packed_pairs: usize,
    pub expert_packed_pairs: usize,
    pub router_f32_names: usize,
    pub dense_bf16_names: usize,
}

impl MiniMaxM2ManifestReport {
    /// The partition the configuration requires: `62 * 4 = 248` attention pairs,
    /// `62 * 256 * 3 = 47,616` expert pairs, `62 * 2 = 124` router rows and `3 + 62 * 4 = 251` BF16
    /// rows for the pinned M2.5 profile, which is 96,103 names.
    pub fn expected(config: &MiniMaxM2Config) -> Result<Self, MiniMaxM2ManifestError> {
        let layers = config.num_hidden_layers;
        let attention_packed_pairs = checked_mul(layers, 4, "attention packed pairs")?;
        let expert_packed_pairs = checked_mul(
            checked_mul(layers, config.num_local_experts, "expert rows")?,
            3,
            "expert packed pairs",
        )?;
        let router_f32_names = checked_mul(layers, 2, "router F32 names")?;
        let dense_bf16_names = checked_add(
            3,
            checked_mul(layers, 4, "per-layer BF16 names")?,
            "dense BF16 names",
        )?;
        let packed_pairs = checked_add(
            attention_packed_pairs,
            expert_packed_pairs,
            "total packed pairs",
        )?;
        let total_names = checked_add(
            checked_add(router_f32_names, dense_bf16_names, "dense names")?,
            checked_mul(packed_pairs, 2, "packed names")?,
            "total names",
        )?;
        Ok(Self {
            total_names,
            attention_packed_pairs,
            expert_packed_pairs,
            router_f32_names,
            dense_bf16_names,
        })
    }

    /// Packed pairs of both classes, which is half the packed names.
    ///
    /// Checked: the fields are public and the struct derives `Default`, so an arbitrary instance
    /// (for example one inside [`MiniMaxM2ManifestError::PartitionMismatch`]) could wrap a plain `+`.
    pub const fn packed_pairs(&self) -> Option<usize> {
        self.attention_packed_pairs
            .checked_add(self.expert_packed_pairs)
    }
}

/// Every authenticated name of one MiniMax-M2 configuration, with its exact disposition.
#[derive(Clone, Debug)]
pub struct MiniMaxM2Manifest {
    entries: BTreeMap<String, MiniMaxM2ManifestEntry>,
    report: MiniMaxM2ManifestReport,
}

impl MiniMaxM2Manifest {
    /// Build the manifest from the configuration's source table.
    ///
    /// The walk is structural: globals, then each block's norms, router rows, attention projections
    /// and experts, in table order. No name is parsed to decide what it is.
    pub fn new(config: &MiniMaxM2Config) -> Result<Self, MiniMaxM2ManifestError> {
        let table = MiniMaxM2SourceTable::new(config)?;
        let mut builder = ManifestBuilder::default();

        for source in [table.embedding(), table.final_norm(), table.lm_head()] {
            builder.push_dense(source, MiniMaxM2Namespace::Bf16Dense)?;
        }
        for layer in table.layers() {
            builder.push_layer(layer)?;
        }

        builder.finish(MiniMaxM2ManifestReport::expected(config)?)
    }

    pub const fn report(&self) -> MiniMaxM2ManifestReport {
        self.report
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every entry in checkpoint-name order.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (&str, &MiniMaxM2ManifestEntry)> {
        self.entries
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
    }

    pub fn entry(&self, name: &str) -> Option<&MiniMaxM2ManifestEntry> {
        self.entries.get(name)
    }

    /// The disposition of one presented row, or the typed reason it has none.
    ///
    /// The MTP check comes first so an MTP row is reported as a boundary violation, not an unknown
    /// name. The pinned revision has no MTP namespace, so this arm is unreachable for it; it rejects
    /// a later revision that adds one.
    pub fn disposition_for(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<TensorDisposition, MiniMaxM2ManifestError> {
        if is_mtp_name(name) {
            return Err(MiniMaxM2ManifestError::MtpTensorPresented(name.to_string()));
        }
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| MiniMaxM2ManifestError::UnknownName(name.to_string()))?;
        if dtype != entry.dtype {
            return Err(MiniMaxM2ManifestError::Dtype {
                name: name.to_string(),
                expected: entry.dtype,
                observed: dtype.to_string(),
            });
        }
        if shape != entry.shape {
            return Err(MiniMaxM2ManifestError::Shape {
                name: name.to_string(),
                expected: entry.shape.clone(),
                observed: shape.to_vec(),
            });
        }
        Ok(entry.disposition.clone())
    }

    /// Assign one disposition to every row of an authenticated card 359 inventory.
    ///
    /// The per-row decision is [`Self::disposition_for`]; this adds the totality check so a
    /// missing name cannot yield a short decision list that reads as complete.
    pub fn classify(
        &self,
        inventory: AuthenticatedInventory<'_>,
    ) -> Result<Vec<InventoryDecision>, MiniMaxM2ManifestError> {
        let mut decisions = Vec::with_capacity(inventory.len());
        let mut seen = BTreeSet::new();
        for row in inventory.rows() {
            let disposition = self.disposition_for(row.name(), row.dtype(), row.shape())?;
            if !seen.insert(row.name().to_string()) {
                return Err(MiniMaxM2ManifestError::DuplicateName(
                    row.name().to_string(),
                ));
            }
            decisions.push(InventoryDecision::new(row.key(), disposition));
        }
        if let Some(name) = self.entries.keys().find(|name| !seen.contains(*name)) {
            return Err(MiniMaxM2ManifestError::MissingName(name.clone()));
        }
        Ok(decisions)
    }
}

/// Whether a dotted name lies in an MTP namespace.
///
/// Segment-prefixed, not substring: `model.mtp.0.x` and `model.mtp_layers.0.x` are MTP, a
/// hypothetical `deep_mtproj` segment is not. No derived name has such a segment
/// (`minimax_m2_mtp_is_not_fabricated`).
fn is_mtp_name(name: &str) -> bool {
    name.split('.').any(|segment| segment.starts_with("mtp"))
}

#[derive(Default)]
struct ManifestBuilder {
    entries: BTreeMap<String, MiniMaxM2ManifestEntry>,
    report: MiniMaxM2ManifestReport,
}

impl ManifestBuilder {
    fn insert(
        &mut self,
        name: String,
        entry: MiniMaxM2ManifestEntry,
    ) -> Result<(), MiniMaxM2ManifestError> {
        increment(&mut self.report.total_names, "total names")?;
        if self.entries.insert(name.clone(), entry).is_some() {
            return Err(MiniMaxM2ManifestError::DuplicateName(name));
        }
        Ok(())
    }

    fn push_dense(
        &mut self,
        source: &MiniMaxM2DenseSource,
        namespace: MiniMaxM2Namespace,
    ) -> Result<(), MiniMaxM2ManifestError> {
        let role = source.role();
        if namespace.dense_role() != Some(role) {
            return Err(MiniMaxM2ManifestError::RoleMismatch {
                name: source.name().to_string(),
                namespace,
                observed: role.source_dtype(),
            });
        }
        let disposition = match role {
            MiniMaxM2DenseRole::Bf16 => {
                increment(&mut self.report.dense_bf16_names, "dense BF16 names")?;
                TensorDisposition::DenseBf16
            }
            MiniMaxM2DenseRole::F32 => {
                increment(&mut self.report.router_f32_names, "router F32 names")?;
                TensorDisposition::DenseF32
            }
        };
        self.insert(
            source.name().to_string(),
            MiniMaxM2ManifestEntry {
                namespace,
                disposition,
                dtype: role.source_dtype(),
                shape: source.shape().to_vec(),
            },
        )
    }

    fn push_packed(
        &mut self,
        source: &MiniMaxM2PackedSource,
        namespace: MiniMaxM2Namespace,
    ) -> Result<(), MiniMaxM2ManifestError> {
        let counter = match namespace {
            MiniMaxM2Namespace::AttentionPacked => {
                (&mut self.report.attention_packed_pairs, "attention pairs")
            }
            MiniMaxM2Namespace::ExpertPacked => {
                (&mut self.report.expert_packed_pairs, "expert pairs")
            }
            MiniMaxM2Namespace::RouterDense | MiniMaxM2Namespace::Bf16Dense => {
                return Err(MiniMaxM2ManifestError::RoleMismatch {
                    name: source.linear_id().to_string(),
                    namespace,
                    observed: "packed",
                });
            }
        };
        increment(counter.0, counter.1)?;

        let descriptor = source.descriptor();
        let linear_id = source.linear_id().to_string();
        let scale_source_shape = descriptor.source_shape(SourceRole::Planar(OperandRole::Scale));
        let scale_bytes_per_element = MINIMAX_M25_PACKED_FORMAT
            .descriptor()
            .planar_operand(OperandRole::Scale)
            .expect("MINIMAX_M25_PACKED_FORMAT has a Scale operand")
            .element_bytes();
        self.insert(
            source.checkpoint_name(MiniMaxM2PackedComponent::Weight),
            MiniMaxM2ManifestEntry {
                namespace,
                disposition: TensorDisposition::PackedWeight {
                    linear_id: linear_id.clone(),
                    format: MINIMAX_M25_PACKED_FORMAT,
                    logical_shape: descriptor.shape(),
                },
                dtype: "F8_E4M3",
                shape: descriptor
                    .source_shape(SourceRole::Planar(OperandRole::Codes))
                    .to_vec(),
            },
        )?;
        self.insert(
            source.checkpoint_name(MiniMaxM2PackedComponent::Scale),
            MiniMaxM2ManifestEntry {
                namespace,
                disposition: TensorDisposition::PackedScale { linear_id },
                dtype: "F32",
                shape: vec![
                    scale_source_shape[0],
                    scale_source_shape[1] / scale_bytes_per_element,
                ],
            },
        )
    }

    fn push_layer(&mut self, layer: &MiniMaxM2LayerSources) -> Result<(), MiniMaxM2ManifestError> {
        for source in [
            layer.input_norm(),
            layer.post_attention_norm(),
            layer.q_norm(),
            layer.k_norm(),
        ] {
            self.push_dense(source, MiniMaxM2Namespace::Bf16Dense)?;
        }
        for source in [layer.router_gate(), layer.router_bias()] {
            self.push_dense(source, MiniMaxM2Namespace::RouterDense)?;
        }
        for projection in MiniMaxM2AttentionProjection::ALL {
            self.push_packed(
                layer.attention(projection),
                MiniMaxM2Namespace::AttentionPacked,
            )?;
        }
        let experts = layer.experts(MiniMaxM2ExpertProjection::W1).len();
        for expert in 0..experts {
            for projection in MiniMaxM2ExpertProjection::ALL {
                let sources = layer.experts(projection);
                let source =
                    sources
                        .get(expert)
                        .ok_or(MiniMaxM2ManifestError::RaggedExpertTable {
                            projection: projection.suffix(),
                            expected: experts,
                            observed: sources.len(),
                        })?;
                self.push_packed(source, MiniMaxM2Namespace::ExpertPacked)?;
            }
        }
        Ok(())
    }

    fn finish(
        self,
        expected: MiniMaxM2ManifestReport,
    ) -> Result<MiniMaxM2Manifest, MiniMaxM2ManifestError> {
        for (name, entry) in &self.entries {
            let Some((linear_id, component)) = entry.packed_component() else {
                continue;
            };
            let counterpart_component = component.other();
            let counterpart = format!("{linear_id}{}", counterpart_component.checkpoint_suffix());
            let reciprocal = self
                .entries
                .get(&counterpart)
                .and_then(MiniMaxM2ManifestEntry::packed_component)
                .is_some_and(|(paired_id, paired_component)| {
                    paired_id == linear_id && paired_component == counterpart_component
                });
            if !reciprocal {
                return Err(MiniMaxM2ManifestError::UnpairedPackedComponent {
                    name: name.clone(),
                    counterpart,
                });
            }
        }
        if self.report != expected {
            return Err(MiniMaxM2ManifestError::PartitionMismatch {
                observed: self.report,
                expected,
            });
        }
        Ok(MiniMaxM2Manifest {
            entries: self.entries,
            report: self.report,
        })
    }
}

fn increment(value: &mut usize, field: &'static str) -> Result<(), MiniMaxM2ManifestError> {
    *value = checked_add(*value, 1, field)?;
    Ok(())
}

fn checked_add(
    lhs: usize,
    rhs: usize,
    field: &'static str,
) -> Result<usize, MiniMaxM2ManifestError> {
    lhs.checked_add(rhs)
        .ok_or(MiniMaxM2ManifestError::CountOverflow { field })
}

fn checked_mul(
    lhs: usize,
    rhs: usize,
    field: &'static str,
) -> Result<usize, MiniMaxM2ManifestError> {
    lhs.checked_mul(rhs)
        .ok_or(MiniMaxM2ManifestError::CountOverflow { field })
}

#[derive(Debug, thiserror::Error)]
pub enum MiniMaxM2ManifestError {
    #[error(transparent)]
    SourceTable(#[from] MiniMaxM2SourceTableError),
    #[error("MiniMax-M2 manifest count overflow for {field}")]
    CountOverflow { field: &'static str },
    #[error("MiniMax-M2 manifest has a duplicate disposition for {0}")]
    DuplicateName(String),
    #[error("MiniMax-M2 inventory is missing the authenticated name {0}")]
    MissingName(String),
    #[error("MiniMax-M2 manifest has no disposition for {0}")]
    UnknownName(String),
    #[error(
        "MiniMax-M2 rejects the MTP tensor {0}: this revision publishes no MTP namespace and the config's MTP flags are not a layer count"
    )]
    MtpTensorPresented(String),
    #[error("MiniMax-M2 tensor {name} has dtype {observed}, expected {expected}")]
    Dtype {
        name: String,
        expected: &'static str,
        observed: String,
    },
    #[error("MiniMax-M2 tensor {name} has shape {observed:?}, expected {expected:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        observed: Vec<usize>,
    },
    #[error("MiniMax-M2 source {name} is {observed} but its table position is {namespace:?}")]
    RoleMismatch {
        name: String,
        namespace: MiniMaxM2Namespace,
        observed: &'static str,
    },
    #[error("MiniMax-M2 packed component {name} has no reciprocal counterpart {counterpart}")]
    UnpairedPackedComponent { name: String, counterpart: String },
    #[error("MiniMax-M2 expert projection {projection} has {observed} rows, expected {expected}")]
    RaggedExpertTable {
        projection: &'static str,
        expected: usize,
        observed: usize,
    },
    #[error("MiniMax-M2 manifest partition is {observed:?}, expected {expected:?}")]
    PartitionMismatch {
        observed: MiniMaxM2ManifestReport,
        expected: MiniMaxM2ManifestReport,
    },
}

#[cfg(test)]
mod tests {
    use super::super::tests::{exact_m25_config, small_config, small_config_json};
    use super::*;

    fn pinned_manifest() -> MiniMaxM2Manifest {
        MiniMaxM2Manifest::new(&exact_m25_config()).expect("pinned manifest")
    }

    /// Every pinned name gets exactly one disposition, the four namespaces partition the inventory
    /// by the counts spec 365 fixes, and the two packed classes are disjoint from the dense ones.
    ///
    /// Also proves `packed_pairs` is checked, since `MiniMaxM2ManifestReport` is constructible
    /// outside the builder.
    ///
    /// Red under: replacing `checked_add` with a wrapping or saturating sum - the first returns
    /// `Some(0)` and the second `Some(usize::MAX)`, neither of which is `None`.
    #[test]
    fn minimax_m2_packed_pairs_refuses_to_wrap() {
        let overflowing = MiniMaxM2ManifestReport {
            attention_packed_pairs: usize::MAX,
            expert_packed_pairs: 1,
            ..MiniMaxM2ManifestReport::default()
        };
        assert_eq!(
            overflowing.packed_pairs(),
            None,
            "a hand-built report that overflows must report no total, not a wrapped one"
        );

        let ordinary = MiniMaxM2ManifestReport {
            attention_packed_pairs: 248,
            expert_packed_pairs: 47_616,
            ..MiniMaxM2ManifestReport::default()
        };
        assert_eq!(ordinary.packed_pairs(), Some(47_864));
    }

    /// Red under: omitting a scale row, inserting any name twice, or giving a router row the BF16
    /// namespace - the first two move `total_names` and the last fails the role cross-check.
    #[test]
    fn minimax_m2_packed_manifest_is_complete() {
        let manifest = pinned_manifest();
        let report = manifest.report();

        assert_eq!(report.attention_packed_pairs, 248);
        assert_eq!(report.expert_packed_pairs, 47_616);
        assert_eq!(report.packed_pairs(), Some(47_864));
        assert_eq!(report.router_f32_names, 124);
        assert_eq!(report.dense_bf16_names, 251);
        assert_eq!(report.total_names, 96_103);
        assert_eq!(manifest.len(), 96_103, "one entry per name, none colliding");

        let mut counted = MiniMaxM2ManifestReport {
            total_names: manifest.len(),
            ..MiniMaxM2ManifestReport::default()
        };
        for (_, entry) in manifest.entries() {
            match (entry.namespace(), entry.disposition()) {
                (MiniMaxM2Namespace::AttentionPacked, TensorDisposition::PackedWeight { .. }) => {
                    counted.attention_packed_pairs += 1;
                }
                (MiniMaxM2Namespace::ExpertPacked, TensorDisposition::PackedWeight { .. }) => {
                    counted.expert_packed_pairs += 1;
                }
                (
                    MiniMaxM2Namespace::AttentionPacked | MiniMaxM2Namespace::ExpertPacked,
                    TensorDisposition::PackedScale { .. },
                ) => {}
                (MiniMaxM2Namespace::RouterDense, TensorDisposition::DenseF32) => {
                    counted.router_f32_names += 1;
                }
                (MiniMaxM2Namespace::Bf16Dense, TensorDisposition::DenseBf16) => {
                    counted.dense_bf16_names += 1;
                }
                (namespace, disposition) => {
                    panic!("{namespace:?} must not hold {disposition:?}");
                }
            }
        }
        assert_eq!(counted, report, "the walked rows reproduce the report");

        // The manifest's name set is exactly the source table's.
        let table = MiniMaxM2SourceTable::new(&exact_m25_config()).expect("source table");
        let names = table.names();
        assert_eq!(names.len(), 96_103);
        assert!(
            names.iter().all(|name| manifest.entry(name).is_some()),
            "every derived name has a disposition"
        );
    }

    /// A packed weight without its scale, or a scale without its weight, is an error.
    ///
    /// Red under: dropping the reciprocal check in `finish`, or pairing a weight with another
    /// linear's scale.
    #[test]
    fn minimax_m2_manifest_rejects_unpaired_packed_rows() {
        let config = small_config();
        let table = MiniMaxM2SourceTable::new(&config).expect("source table");
        let source = table.layers()[0].attention(MiniMaxM2AttentionProjection::Q);

        let mut builder = ManifestBuilder::default();
        builder
            .push_packed(source, MiniMaxM2Namespace::AttentionPacked)
            .expect("both components");
        builder
            .entries
            .remove(&source.checkpoint_name(MiniMaxM2PackedComponent::Scale));
        let error = builder
            .finish(MiniMaxM2ManifestReport::default())
            .expect_err("a weight without its scale must fail");
        assert!(
            matches!(
                error,
                MiniMaxM2ManifestError::UnpairedPackedComponent { ref counterpart, .. }
                    if counterpart.ends_with(".weight_scale_inv")
            ),
            "error was {error}"
        );
    }

    /// Classification reads the manifest row, not the suffix. `model.norm.weight` ends in `.weight`
    /// and is a dense BF16 row; the router gate ends in `.weight` and is dense F32.
    ///
    /// Red under: deciding the disposition from the name's suffix.
    #[test]
    fn minimax_m2_manifest_classifies_by_row_not_suffix() {
        let manifest = MiniMaxM2Manifest::new(&small_config()).expect("manifest");
        let config = small_config();

        let norm = manifest
            .disposition_for("model.norm.weight", "BF16", &[config.hidden_size])
            .expect("final norm is dense");
        assert_eq!(norm, TensorDisposition::DenseBf16);

        let gate = manifest
            .disposition_for(
                "model.layers.0.block_sparse_moe.gate.weight",
                "F32",
                &[config.num_local_experts, config.hidden_size],
            )
            .expect("router gate is dense F32");
        assert_eq!(gate, TensorDisposition::DenseF32);

        let attention = manifest
            .disposition_for(
                "model.layers.0.self_attn.q_proj.weight",
                "F8_E4M3",
                &[config.q_dim(), config.hidden_size],
            )
            .expect("attention weight is packed");
        assert!(matches!(attention, TensorDisposition::PackedWeight { .. }));

        let unknown = manifest
            .disposition_for("model.layers.0.self_attn.dense_fallback", "BF16", &[1])
            .expect_err("an unnamed row has no disposition");
        assert!(matches!(unknown, MiniMaxM2ManifestError::UnknownName(_)));
    }

    /// Each entry pins the dtype and shape its shard header must present: the physical E4M3 grid for
    /// a weight and the `[128, 128]` block grid for its scale, which are different shapes.
    ///
    /// Red under: accepting any dtype, or comparing the scale row against the weight's shape.
    #[test]
    fn minimax_m2_manifest_row_dtype_and_shape_are_pinned() {
        let manifest = pinned_manifest();
        let weight = manifest
            .entry("model.layers.0.self_attn.q_proj.weight")
            .expect("pinned q_proj weight");
        let scale = manifest
            .entry("model.layers.0.self_attn.q_proj.weight_scale_inv")
            .expect("pinned q_proj scale");
        assert_eq!(weight.dtype(), "F8_E4M3");
        assert_eq!(weight.shape(), [6144, 3072]);
        assert_eq!(scale.dtype(), "F32");
        assert_eq!(scale.shape(), [48, 24]);

        let wrong_dtype = manifest
            .disposition_for(
                "model.layers.0.self_attn.q_proj.weight",
                "BF16",
                &[6144, 3072],
            )
            .expect_err("a BF16 attention weight is not this checkpoint");
        assert!(matches!(wrong_dtype, MiniMaxM2ManifestError::Dtype { .. }));

        let wrong_shape = manifest
            .disposition_for(
                "model.layers.0.self_attn.q_proj.weight_scale_inv",
                "F32",
                &[6144, 3072],
            )
            .expect_err("a scale with the weight's shape is not this checkpoint");
        assert!(matches!(wrong_shape, MiniMaxM2ManifestError::Shape { .. }));
    }

    /// Every packed entry carries the linear id and component needed to spell the graph name (card
    /// 379), and the checkpoint spelling matches the real checkpoint.
    ///
    /// The suffixes are literals, not read from [`MiniMaxM2PackedComponent::checkpoint_suffix`],
    /// which would only prove a string equals itself.
    ///
    /// Red under: changing either checkpoint suffix, or storing an entry's name independently of the
    /// pair it is derived from. The graph half is proven in `poot-models`, where card 379's
    /// `PackedSourceName` is visible.
    #[test]
    fn minimax_m2_manifest_rows_carry_the_namespace_bridge() {
        let manifest = MiniMaxM2Manifest::new(&small_config()).expect("manifest");
        let mut packed = 0usize;
        for (name, entry) in manifest.entries() {
            let Some((linear_id, component)) = entry.packed_component() else {
                assert!(
                    entry.namespace() == MiniMaxM2Namespace::RouterDense
                        || entry.namespace() == MiniMaxM2Namespace::Bf16Dense
                );
                continue;
            };
            packed += 1;
            let expected = match component {
                MiniMaxM2PackedComponent::Weight => format!("{linear_id}.weight"),
                MiniMaxM2PackedComponent::Scale => format!("{linear_id}.weight_scale_inv"),
            };
            assert_eq!(
                name, expected,
                "the entry key is the checkpoint spelling of its own (linear id, component)"
            );
        }
        assert_eq!(
            packed,
            2 * manifest.report().packed_pairs().expect("pairs fit")
        );
    }

    /// The manifest depends on the text namespaces alone: MTP config fields are recorded but derive
    /// no row, and an MTP row presented for execution is a dedicated error.
    ///
    /// Red under: creating rows from `num_mtp_modules`, or dropping the MTP arm so the name falls
    /// through to the generic unknown-name error.
    #[test]
    fn minimax_m2_mtp_is_not_fabricated() {
        let mut value: serde_json::Value =
            serde_json::from_slice(&small_config_json(2, 3)).expect("fixture value");
        value["num_mtp_modules"] = serde_json::json!(3);
        let three = MiniMaxM2Config::from_slice(
            &serde_json::to_vec(&value).expect("serialize three-module config"),
        )
        .expect("config with three MTP modules");
        value["num_mtp_modules"] = serde_json::json!(7);
        value["mtp_transformer_layers"] = serde_json::json!(4);
        let seven = MiniMaxM2Config::from_slice(
            &serde_json::to_vec(&value).expect("serialize seven-module config"),
        )
        .expect("config with seven MTP modules");
        assert_ne!(three.num_mtp_modules, seven.num_mtp_modules);

        let three_names = MiniMaxM2Manifest::new(&three)
            .expect("manifest")
            .entries()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        let seven_names = MiniMaxM2Manifest::new(&seven)
            .expect("manifest")
            .entries()
            .map(|(name, _)| name.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            three_names, seven_names,
            "the MTP flags contribute no manifest row"
        );

        let manifest = pinned_manifest();
        assert!(
            manifest.entries().all(|(name, _)| !is_mtp_name(name)),
            "no pinned name lies in an MTP namespace"
        );
        for name in [
            "model.mtp.0.layers.0.input_layernorm.weight",
            "model.mtp_layers.0.eh_proj.weight",
        ] {
            let error = manifest
                .disposition_for(name, "BF16", &[3072])
                .expect_err("an MTP row must not be admitted");
            assert!(
                matches!(error, MiniMaxM2ManifestError::MtpTensorPresented(_)),
                "error was {error}"
            );
        }
    }
}

use super::*;
#[cfg(test)]
use std::collections::BTreeMap;

pub(crate) const GLM53_FLASH_TEXT_SELECTED_NAMES: usize = 74_001;

pub(crate) const GLM53_FLASH_TEXT_DEFERRED_MTP_NAMES: usize = 1_760;

pub(crate) const GLM53_FLASH_TEXT_EXCLUDED_VISION_NAMES: usize = 347;

pub(crate) const GLM53_FLASH_TEXT_PACKED_PAIRS: usize = 36_467;

pub(crate) const GLM53_FLASH_TEXT_DSA_PACKED_PAIRS: usize = 44;

pub(crate) const GLM53_FLASH_TEXT_DENSE_FFN_PACKED_PAIRS: usize = 9;

pub(crate) const GLM53_FLASH_TEXT_ROUTED_EXPERT_PACKED_PAIRS: usize = 36_288;

pub(crate) const GLM53_FLASH_TEXT_SHARED_EXPERT_PACKED_PAIRS: usize = 126;

pub(crate) const GLM53_FLASH_TEXT_DENSE_BF16_NAMES: usize = 777;

pub(crate) const GLM53_FLASH_TEXT_DENSE_F32_NAMES: usize = 290;

/// Exact committed-metadata accounting for the GLM-5.3-Flash main text selection.
///
/// Names and descriptors only; not source-owner, span, byte-accounting, or executor evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Glm53FlashTextInventoryReport {
    pub total_names: usize,
    pub selected_names: usize,
    pub deferred_mtp_names: usize,
    pub excluded_vision_names: usize,
    pub packed_pairs: usize,
    pub dsa_packed_pairs: usize,
    pub dense_ffn_packed_pairs: usize,
    pub routed_expert_packed_pairs: usize,
    pub shared_expert_packed_pairs: usize,
    pub dense_bf16_names: usize,
    pub dense_f32_names: usize,
    pub dense_i64_names: usize,
}

impl Glm53FlashTextInventoryReport {
    pub const fn exact() -> Self {
        Self {
            total_names: GLM53_FLASH_TENSOR_COUNT,
            selected_names: GLM53_FLASH_TEXT_SELECTED_NAMES,
            deferred_mtp_names: GLM53_FLASH_TEXT_DEFERRED_MTP_NAMES,
            excluded_vision_names: GLM53_FLASH_TEXT_EXCLUDED_VISION_NAMES,
            packed_pairs: GLM53_FLASH_TEXT_PACKED_PAIRS,
            dsa_packed_pairs: GLM53_FLASH_TEXT_DSA_PACKED_PAIRS,
            dense_ffn_packed_pairs: GLM53_FLASH_TEXT_DENSE_FFN_PACKED_PAIRS,
            routed_expert_packed_pairs: GLM53_FLASH_TEXT_ROUTED_EXPERT_PACKED_PAIRS,
            shared_expert_packed_pairs: GLM53_FLASH_TEXT_SHARED_EXPERT_PACKED_PAIRS,
            dense_bf16_names: GLM53_FLASH_TEXT_DENSE_BF16_NAMES,
            dense_f32_names: GLM53_FLASH_TEXT_DENSE_F32_NAMES,
            dense_i64_names: 0,
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct Glm53FlashExpectedTextRow {
    pub(crate) name: String,
    pub(crate) shard: String,
    pub(crate) kind: Glm53FlashExpectedTextKind,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) enum Glm53FlashExpectedTextKind {
    Deferred,
    Excluded,
    Selected {
        dtype: String,
        shape: Vec<usize>,
        logical_shape: Vec<usize>,
        quant_role: String,
    },
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum Glm53FlashPackedTextRole {
    Dsa,
    DenseFfn,
    RoutedExpert,
    SharedExpert,
}

#[cfg(test)]
pub(crate) fn glm53_flash_layer(name: &str) -> Option<usize> {
    name.strip_prefix("model.language_model.layers.")?
        .split_once('.')?
        .0
        .parse()
        .ok()
}

#[cfg(test)]
pub(crate) fn glm53_flash_text_scope(
    name: &str,
) -> Result<Option<TensorDisposition>, Glm53FlashMetadataError> {
    if name.starts_with("model.visual.") {
        return Ok(Some(TensorDisposition::Excluded));
    }
    if glm53_flash_layer(name) == Some(45) {
        return Ok(Some(TensorDisposition::Deferred));
    }
    if name == "lm_head.weight"
        || name == "model.language_model.embed_tokens.weight"
        || name == "model.language_model.norm.weight"
        || glm53_flash_layer(name).is_some_and(|layer| layer < 45)
    {
        return Ok(None);
    }
    Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
        "unrecognized namespace for {name}"
    )))
}

#[cfg(test)]
pub(crate) fn glm53_flash_expected_text_kind(
    name: &str,
    selected: impl FnOnce() -> Result<Glm53FlashExpectedTextKind, Glm53FlashMetadataError>,
) -> Result<Glm53FlashExpectedTextKind, Glm53FlashMetadataError> {
    match glm53_flash_text_scope(name)? {
        Some(TensorDisposition::Deferred) => Ok(Glm53FlashExpectedTextKind::Deferred),
        Some(TensorDisposition::Excluded) => Ok(Glm53FlashExpectedTextKind::Excluded),
        Some(other) => Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
            "invalid fixed disposition {other:?} for {name}"
        ))),
        None => selected(),
    }
}

#[cfg(test)]
pub(crate) fn glm53_flash_packed_text_role(
    linear_id: &str,
) -> Result<Glm53FlashPackedTextRole, Glm53FlashMetadataError> {
    if linear_id.contains(".self_attn.") {
        let layer = glm53_flash_layer(linear_id).ok_or_else(|| {
            Glm53FlashMetadataError::InvalidTextInventory(format!(
                "packed attention row has no layer: {linear_id}"
            ))
        })?;
        if layer < 45 && layer % 4 == 3 {
            return Ok(Glm53FlashPackedTextRole::Dsa);
        }
    } else if linear_id.contains(".mlp.experts.") {
        return Ok(Glm53FlashPackedTextRole::RoutedExpert);
    } else if linear_id.contains(".mlp.shared_experts.") {
        return Ok(Glm53FlashPackedTextRole::SharedExpert);
    } else if linear_id.contains(".mlp.") {
        return Ok(Glm53FlashPackedTextRole::DenseFfn);
    }
    Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
        "unrecognized selected packed role {linear_id}"
    )))
}

#[cfg(test)]
pub(crate) fn glm53_flash_selected_disposition(
    name: &str,
    logical_shape: &[usize],
    quant_role: &str,
    dtype: &str,
) -> Result<TensorDisposition, Glm53FlashMetadataError> {
    match (quant_role, dtype) {
        ("e4m3_weight", "F8_E4M3") => {
            let linear_id = name.strip_suffix(".weight").ok_or_else(|| {
                Glm53FlashMetadataError::InvalidTextInventory(format!(
                    "packed weight has invalid name {name}"
                ))
            })?;
            let logical_shape: [usize; 2] = logical_shape.try_into().map_err(|_| {
                Glm53FlashMetadataError::InvalidTextInventory(format!(
                    "packed weight {name} does not have rank two logical shape"
                ))
            })?;
            Ok(TensorDisposition::PackedWeight {
                linear_id: linear_id.to_string(),
                format: WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::F32,
                },
                logical_shape,
            })
        }
        ("f32_block_scale", "F32") => {
            let linear_id = name.strip_suffix(".weight_scale_inv").ok_or_else(|| {
                Glm53FlashMetadataError::InvalidTextInventory(format!(
                    "packed scale has invalid name {name}"
                ))
            })?;
            Ok(TensorDisposition::PackedScale {
                linear_id: linear_id.to_string(),
            })
        }
        ("unquantized", "BF16") => Ok(TensorDisposition::DenseBf16),
        ("unquantized", "F32") => Ok(TensorDisposition::DenseF32),
        ("unquantized", "I64") => Ok(TensorDisposition::DenseI64),
        _ => Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
            "unsupported selected descriptor {}: role={}, dtype={}",
            name, quant_role, dtype
        ))),
    }
}

#[cfg(test)]
pub(crate) fn glm53_flash_text_disposition(
    row: &Glm53FlashExpectedTextRow,
) -> Result<TensorDisposition, Glm53FlashMetadataError> {
    match &row.kind {
        Glm53FlashExpectedTextKind::Deferred => Ok(TensorDisposition::Deferred),
        Glm53FlashExpectedTextKind::Excluded => Ok(TensorDisposition::Excluded),
        Glm53FlashExpectedTextKind::Selected {
            dtype,
            logical_shape,
            quant_role,
            ..
        } => glm53_flash_selected_disposition(&row.name, logical_shape, quant_role, dtype),
    }
}

#[cfg(test)]
pub(crate) fn glm53_flash_text_report(
    rows: &[Glm53FlashExpectedTextRow],
    expected: Glm53FlashTextInventoryReport,
) -> Result<Glm53FlashTextInventoryReport, Glm53FlashMetadataError> {
    fn increment(value: &mut usize, field: &'static str) -> Result<(), Glm53FlashMetadataError> {
        *value = value
            .checked_add(1)
            .ok_or(Glm53FlashMetadataError::AccountingOverflow { field })?;
        Ok(())
    }

    let mut report = Glm53FlashTextInventoryReport::default();
    let mut dispositions = BTreeMap::new();
    for row in rows {
        let disposition = glm53_flash_text_disposition(row)?;
        if dispositions
            .insert(row.name.as_str(), disposition.clone())
            .is_some()
        {
            return Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
                "duplicate committed name {}",
                row.name
            )));
        }
        increment(&mut report.total_names, "text inventory total names")?;
        match &disposition {
            TensorDisposition::Deferred => {
                increment(&mut report.deferred_mtp_names, "deferred MTP names")?;
            }
            TensorDisposition::Excluded => {
                increment(&mut report.excluded_vision_names, "excluded vision names")?;
            }
            TensorDisposition::PackedWeight { linear_id, .. } => {
                increment(&mut report.selected_names, "selected text names")?;
                increment(&mut report.packed_pairs, "packed pairs")?;
                match glm53_flash_packed_text_role(linear_id)? {
                    Glm53FlashPackedTextRole::Dsa => {
                        increment(&mut report.dsa_packed_pairs, "DSA packed pairs")?;
                    }
                    Glm53FlashPackedTextRole::DenseFfn => {
                        increment(&mut report.dense_ffn_packed_pairs, "dense FFN packed pairs")?;
                    }
                    Glm53FlashPackedTextRole::RoutedExpert => {
                        increment(
                            &mut report.routed_expert_packed_pairs,
                            "routed expert packed pairs",
                        )?;
                    }
                    Glm53FlashPackedTextRole::SharedExpert => {
                        increment(
                            &mut report.shared_expert_packed_pairs,
                            "shared expert packed pairs",
                        )?;
                    }
                }
            }
            TensorDisposition::PackedScale { .. } => {
                increment(&mut report.selected_names, "selected text names")?;
            }
            TensorDisposition::DenseBf16 => {
                increment(&mut report.selected_names, "selected text names")?;
                increment(&mut report.dense_bf16_names, "dense BF16 names")?;
            }
            TensorDisposition::DenseF32 => {
                increment(&mut report.selected_names, "selected text names")?;
                increment(&mut report.dense_f32_names, "dense F32 names")?;
            }
            TensorDisposition::DenseI64 => {
                increment(&mut report.selected_names, "selected text names")?;
                increment(&mut report.dense_i64_names, "dense I64 names")?;
            }
            TensorDisposition::StandaloneE4m3 => {
                return Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
                    "unexpected standalone E4M3 row {}",
                    row.name
                )));
            }
        }
    }

    for (name, disposition) in &dispositions {
        let (linear_id, counterpart) = match disposition {
            TensorDisposition::PackedWeight { linear_id, .. } => {
                (linear_id, format!("{linear_id}.weight_scale_inv"))
            }
            TensorDisposition::PackedScale { linear_id } => {
                (linear_id, format!("{linear_id}.weight"))
            }
            _ => continue,
        };
        let paired = dispositions.get(counterpart.as_str());
        let valid = matches!(
            (disposition, paired),
            (
                TensorDisposition::PackedWeight { .. },
                Some(TensorDisposition::PackedScale { linear_id: paired_id })
            ) if paired_id == linear_id
        ) || matches!(
            (disposition, paired),
            (
                TensorDisposition::PackedScale { .. },
                Some(TensorDisposition::PackedWeight { linear_id: paired_id, .. })
            ) if paired_id == linear_id
        );
        if !valid {
            return Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
                "selected packed row {name} has no reciprocal counterpart {counterpart}"
            )));
        }
    }

    if report != expected {
        return Err(Glm53FlashMetadataError::InvalidTextInventory(format!(
            "committed partition mismatch: {report:?}"
        )));
    }
    Ok(report)
}

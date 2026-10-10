//! Bounded path, file, header, topology, selection, and digest validation.

use poot_quant::format::{FieldEncoding, FloatFormat, PlanarOperand};

use super::{
    ArtifactIdentityFields, BTreeMap, Component, Context, Deserialize, Deserializer, File,
    HASH_BUFFER_BYTES, INDEX_FILE, MapAccess, MapError, OperandRole, PackedArtifactManifest,
    PackedLimitKind, PackedLoadReport, PackedSafetensorsError, PackedSafetensorsLimits,
    PackedSelectionField, PackedSelectionRow, PackedWeight, Path, RetainedHandle, RetainedReader,
    SHA256, SeekFrom, SeqAccess, Sha256Digest, SourceRole, SourceSpan, StoredBytes, TensorEntry,
    ValidatedSelection, Visitor, de, fmt, sha256_digest,
};

pub(crate) fn validate_manifest_and_limits(
    manifest: &PackedArtifactManifest,
    limits: PackedSafetensorsLimits,
) -> Result<(), PackedSafetensorsError> {
    if manifest.repository.is_empty() || manifest.revision.is_empty() {
        return Err(PackedSafetensorsError::EmptyArtifactLabel);
    }
    if manifest.shards.is_empty() {
        return Err(PackedSafetensorsError::EmptyShardManifest);
    }
    for (kind, limit) in [
        (PackedLimitKind::ConfigBytes, limits.config_bytes),
        (PackedLimitKind::IndexBytes, limits.index_bytes),
        (
            PackedLimitKind::HeaderBytesPerShard,
            limits.header_bytes_per_shard,
        ),
        (PackedLimitKind::ShardCount, limits.shard_count),
        (PackedLimitKind::TensorEntries, limits.tensor_entries),
        (
            PackedLimitKind::SelectedSourceBytes,
            limits.selected_source_bytes,
        ),
        (
            PackedLimitKind::PackedSourceBytes,
            limits.packed_source_bytes,
        ),
    ] {
        if limit == 0 {
            return Err(PackedSafetensorsError::ZeroLimit { kind });
        }
    }
    enforce_limit(
        PackedLimitKind::ShardCount,
        limits.shard_count,
        manifest.shards.len(),
    )
}

pub(crate) fn enforce_limit(
    kind: PackedLimitKind,
    limit: usize,
    actual: usize,
) -> Result<(), PackedSafetensorsError> {
    if actual > limit {
        return Err(PackedSafetensorsError::LimitExceeded {
            kind,
            limit,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn validate_relative_path(path: &str) -> Result<(), PackedSafetensorsError> {
    let parsed = Path::new(path);
    if path.is_empty()
        || parsed.is_absolute()
        || !parsed
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(PackedSafetensorsError::InvalidShardPath {
            path: path.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn open_regular(
    root: &Path,
    relative: &str,
) -> Result<RetainedHandle, PackedSafetensorsError> {
    let path = root.join(relative);
    let file = File::open(&path).map_err(|error| PackedSafetensorsError::Io {
        file: relative.to_string(),
        error,
    })?;
    let metadata = file
        .metadata()
        .map_err(|error| PackedSafetensorsError::Io {
            file: relative.to_string(),
            error,
        })?;
    if !metadata.is_file() {
        return Err(PackedSafetensorsError::NotRegularFile {
            file: relative.to_string(),
        });
    }
    Ok(Box::new(file))
}

/// Maps one shard read-only, once: the single view every selected source of that shard slices.
pub(crate) fn map_shard(
    root: &Path,
    relative: &str,
) -> Result<StoredBytes, PackedSafetensorsError> {
    let file = File::open(root.join(relative)).map_err(|error| PackedSafetensorsError::Io {
        file: relative.to_string(),
        error,
    })?;
    StoredBytes::map(file).map_err(|error| match error {
        MapError::Io(error) => PackedSafetensorsError::Io {
            file: relative.to_string(),
            error,
        },
        MapError::NotRegularFile { .. } => PackedSafetensorsError::NotRegularFile {
            file: relative.to_string(),
        },
    })
}

pub(crate) fn file_length(
    file: &dyn RetainedReader,
    name: &str,
) -> Result<usize, PackedSafetensorsError> {
    let length = file
        .file_length()
        .map_err(|error| PackedSafetensorsError::Io {
            file: name.to_string(),
            error,
        })?;
    usize::try_from(length).map_err(|_| PackedSafetensorsError::ArithmeticOverflow {
        field: "file length conversion",
    })
}

pub(crate) fn read_exact_file(
    file: &mut dyn RetainedReader,
    name: &str,
    length: usize,
) -> Result<Vec<u8>, PackedSafetensorsError> {
    file.seek(SeekFrom::Start(0))
        .and_then(|_| {
            let mut bytes = vec![0u8; length];
            file.read_exact(&mut bytes)?;
            let mut extra = [0u8; 1];
            if file.read(&mut extra)? != 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "file grew while reading",
                ));
            }
            Ok(bytes)
        })
        .map_err(|error| PackedSafetensorsError::Io {
            file: name.to_string(),
            error,
        })
}

/// The header length and data start of a mapped shard, read off its first eight bytes.
pub(crate) fn read_header_length(
    shard: &[u8],
    name: &str,
) -> Result<(usize, usize), PackedSafetensorsError> {
    let Some(prefix) = shard.first_chunk::<8>() else {
        return Err(PackedSafetensorsError::InvalidTensorMetadata {
            file: name.to_string(),
            tensor: "<header>".to_string(),
            reason: "file is shorter than the eight-byte header prefix".to_string(),
        });
    };
    let header_length = usize::try_from(u64::from_le_bytes(*prefix)).map_err(|_| {
        PackedSafetensorsError::ArithmeticOverflow {
            field: "header length conversion",
        }
    })?;
    let data_start = checked_add(8, header_length, "header end")?;
    if data_start > shard.len() {
        return Err(PackedSafetensorsError::InvalidTensorMetadata {
            file: name.to_string(),
            tensor: "<header>".to_string(),
            reason: "declared header extends past end of file".to_string(),
        });
    }
    Ok((header_length, data_start))
}

pub(crate) fn parse_weight_map(
    root: &serde_json::Value,
) -> Result<BTreeMap<String, String>, PackedSafetensorsError> {
    let root = root
        .as_object()
        .ok_or_else(|| PackedSafetensorsError::Json {
            file: INDEX_FILE.to_string(),
            reason: "top level is not an object".to_string(),
        })?;
    let weight_map = root
        .get("weight_map")
        .and_then(serde_json::Value::as_object)
        .ok_or(PackedSafetensorsError::MissingWeightMap)?;
    let mut out = BTreeMap::new();
    for (name, shard) in weight_map {
        let shard =
            shard
                .as_str()
                .ok_or_else(|| PackedSafetensorsError::InvalidWeightMapValue {
                    tensor: name.clone(),
                })?;
        out.insert(name.clone(), shard.to_string());
    }
    Ok(out)
}

pub(crate) fn parse_header_entries(
    root: &serde_json::Value,
    filename: &str,
    data_length: usize,
) -> Result<BTreeMap<String, TensorEntry>, PackedSafetensorsError> {
    let root = root
        .as_object()
        .ok_or_else(|| PackedSafetensorsError::Json {
            file: filename.to_string(),
            reason: "top level is not an object".to_string(),
        })?;
    let mut out = BTreeMap::new();
    for (name, metadata) in root {
        if name == "__metadata__" {
            continue;
        }
        let object =
            metadata
                .as_object()
                .ok_or_else(|| PackedSafetensorsError::InvalidTensorMetadata {
                    file: filename.to_string(),
                    tensor: name.clone(),
                    reason: "metadata is not an object".to_string(),
                })?;
        let dtype = object
            .get("dtype")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| PackedSafetensorsError::InvalidTensorMetadata {
                file: filename.to_string(),
                tensor: name.clone(),
                reason: "dtype is missing or is not a string".to_string(),
            })?
            .to_string();
        let shape_values = object
            .get("shape")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| PackedSafetensorsError::InvalidTensorMetadata {
                file: filename.to_string(),
                tensor: name.clone(),
                reason: "shape is missing or is not an array".to_string(),
            })?;
        let mut shape = Vec::with_capacity(shape_values.len());
        for value in shape_values {
            let dimension =
                value
                    .as_u64()
                    .ok_or_else(|| PackedSafetensorsError::InvalidTensorMetadata {
                        file: filename.to_string(),
                        tensor: name.clone(),
                        reason: "shape contains a non-integer dimension".to_string(),
                    })?;
            shape.push(usize::try_from(dimension).map_err(|_| {
                PackedSafetensorsError::TensorByteCountOverflow {
                    file: filename.to_string(),
                    tensor: name.clone(),
                }
            })?);
        }
        let offsets = object
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| PackedSafetensorsError::InvalidTensorMetadata {
                file: filename.to_string(),
                tensor: name.clone(),
                reason: "data_offsets is missing or is not an array".to_string(),
            })?;
        if offsets.len() != 2 {
            return Err(PackedSafetensorsError::InvalidTensorMetadata {
                file: filename.to_string(),
                tensor: name.clone(),
                reason: "data_offsets must contain exactly two integers".to_string(),
            });
        }
        let start = parse_offset(&offsets[0], filename, name)?;
        let end = parse_offset(&offsets[1], filename, name)?;
        let span = SourceSpan { start, end };
        if start > end || end > data_length {
            return Err(PackedSafetensorsError::InvalidSpan {
                file: filename.to_string(),
                tensor: name.clone(),
                span,
                data_length,
            });
        }
        if let Some(width) = crate::dtype::element_width_bytes(&dtype) {
            let elements = shape
                .iter()
                .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
            let expected = elements
                .and_then(|count| count.checked_mul(width))
                .ok_or_else(|| PackedSafetensorsError::TensorByteCountOverflow {
                    file: filename.to_string(),
                    tensor: name.clone(),
                })?;
            let actual = end - start;
            if expected != actual {
                return Err(PackedSafetensorsError::TensorByteCountMismatch {
                    file: filename.to_string(),
                    tensor: name.clone(),
                    expected,
                    actual,
                });
            }
        }
        out.insert(
            name.clone(),
            TensorEntry {
                shard: filename.to_string(),
                dtype,
                shape,
                span,
            },
        );
    }
    Ok(out)
}

pub(crate) fn parse_offset(
    value: &serde_json::Value,
    filename: &str,
    tensor: &str,
) -> Result<usize, PackedSafetensorsError> {
    let offset = value
        .as_u64()
        .ok_or_else(|| PackedSafetensorsError::InvalidTensorMetadata {
            file: filename.to_string(),
            tensor: tensor.to_string(),
            reason: "data_offsets contains a non-integer".to_string(),
        })?;
    usize::try_from(offset).map_err(|_| PackedSafetensorsError::TensorByteCountOverflow {
        file: filename.to_string(),
        tensor: tensor.to_string(),
    })
}

pub(crate) fn validate_shard_overlaps(
    shard: &str,
    tensors: &BTreeMap<String, TensorEntry>,
) -> Result<(), PackedSafetensorsError> {
    let mut spans = tensors
        .iter()
        .filter(|(_, entry)| entry.shard == shard && entry.span.start != entry.span.end)
        .map(|(name, entry)| (entry.span, name.as_str()))
        .collect::<Vec<_>>();
    spans.sort_unstable_by(|(left_span, left_name), (right_span, right_name)| {
        left_span
            .start
            .cmp(&right_span.start)
            .then_with(|| left_span.end.cmp(&right_span.end))
            .then_with(|| left_name.cmp(right_name))
    });
    for adjacent in spans.windows(2) {
        let (first_span, first_name) = adjacent[0];
        let (second_span, second_name) = adjacent[1];
        if first_span.end > second_span.start {
            return Err(PackedSafetensorsError::OverlappingTensorSpans {
                shard: shard.to_string(),
                first: first_name.to_string(),
                second: second_name.to_string(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_index_header_bijection(
    index: &BTreeMap<String, String>,
    tensors: &BTreeMap<String, TensorEntry>,
) -> Result<(), PackedSafetensorsError> {
    for (name, shard) in index {
        let entry =
            tensors
                .get(name)
                .ok_or_else(|| PackedSafetensorsError::IndexTensorMissing {
                    tensor: name.clone(),
                    shard: shard.clone(),
                })?;
        if &entry.shard != shard {
            return Err(PackedSafetensorsError::IndexHeaderShardMismatch {
                tensor: name.clone(),
                index_shard: shard.clone(),
                header_shard: entry.shard.clone(),
            });
        }
    }
    for (name, entry) in tensors {
        if !index.contains_key(name) {
            return Err(PackedSafetensorsError::HeaderTensorMissingFromIndex {
                tensor: name.clone(),
                shard: entry.shard.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_selection_row(
    selection: &PackedSelectionRow,
    weight: &TensorEntry,
    scale: &TensorEntry,
) -> Result<(), PackedSafetensorsError> {
    compare_selection(
        selection,
        PackedSelectionField::Shard,
        &selection.shard,
        &weight.shard,
    )?;
    compare_selection(
        selection,
        PackedSelectionField::WeightSpan,
        format!("{:?}", selection.weight_span),
        format!("{:?}", weight.span),
    )?;
    compare_selection(
        selection,
        PackedSelectionField::ScaleSpan,
        format!("{:?}", selection.scale_span),
        format!("{:?}", scale.span),
    )?;
    compare_selection(
        selection,
        PackedSelectionField::WeightDtype,
        &selection.weight_dtype,
        &weight.dtype,
    )?;
    compare_selection(
        selection,
        PackedSelectionField::ScaleDtype,
        &selection.scale_dtype,
        &scale.dtype,
    )?;
    compare_selection(
        selection,
        PackedSelectionField::WeightShape,
        format!("{:?}", selection.weight_shape),
        format!("{:?}", weight.shape),
    )?;
    compare_selection(
        selection,
        PackedSelectionField::ScaleShape,
        format!("{:?}", selection.scale_shape),
        format!("{:?}", scale.shape),
    )?;

    let descriptor = selection.descriptor;
    compare_selection(
        selection,
        PackedSelectionField::WeightShape,
        format!(
            "{:?}",
            descriptor.source_shape(SourceRole::Planar(OperandRole::Codes))
        ),
        format!("{:?}", selection.weight_shape),
    )?;
    let scale_source_shape = descriptor.source_shape(SourceRole::Planar(OperandRole::Scale));
    let scale_bytes_per_element = planar_operand(descriptor, OperandRole::Scale).element_bytes();
    compare_selection(
        selection,
        PackedSelectionField::ScaleShape,
        format!(
            "{:?}",
            [
                scale_source_shape[0],
                scale_source_shape[1] / scale_bytes_per_element
            ]
        ),
        format!("{:?}", selection.scale_shape),
    )?;
    compare_selection(
        selection,
        PackedSelectionField::WeightDtype,
        expected_weight_dtype(descriptor),
        &selection.weight_dtype,
    )?;
    compare_selection(
        selection,
        PackedSelectionField::ScaleDtype,
        expected_scale_dtype(descriptor),
        &selection.scale_dtype,
    )?;
    compare_selection(
        selection,
        PackedSelectionField::WeightSourceBytes,
        descriptor.source_bytes(SourceRole::Planar(OperandRole::Codes)),
        selection.weight_span.len()?,
    )?;
    compare_selection(
        selection,
        PackedSelectionField::ScaleSourceBytes,
        descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)),
        selection.scale_span.len()?,
    )
}

/// `descriptor`'s planar operand of `role`, read off its `WeightFormat`'s own
/// [`poot_quant::format::FormatDescriptor`]. `PackedWeight::try_new` already proved every source
/// role a format's descriptor names is present, so a missing operand here is a
/// `poot-quant` descriptor bug, never a loader-admission decision (ADR-0104: whether a format is
/// admitted is the planner's decision, not the loader's - the loader admits any format with a
/// descriptor and a well-formed payload).
fn planar_operand(descriptor: PackedWeight, role: OperandRole) -> PlanarOperand {
    descriptor
        .format()
        .descriptor()
        .planar_operand(role)
        .unwrap_or_else(|error| {
            panic!("{:?} has no {role:?} operand: {error}", descriptor.format())
        })
}

/// The safetensors dtype string one planar operand's stored bytes must carry, derived directly
/// from its [`FieldEncoding`] and sub-byte packing - never a second table of registered cells
/// (card 542d, ADR-0104): a format with a descriptor this crate cannot decode is a `poot-quant`
/// gap, not a loader admission gate.
fn operand_stored_dtype(operand: PlanarOperand) -> &'static str {
    if let Some(packing) = operand.packing {
        return match packing.word_bytes {
            1 => "I8",
            2 => "I16",
            4 => "I32",
            8 => "I64",
            other => panic!("a packed operand with {other}-byte words has no safetensors dtype"),
        };
    }
    match operand.encoding {
        FieldEncoding::Float(FloatFormat::F32) => "F32",
        FieldEncoding::Float(FloatFormat::F16) => "F16",
        FieldEncoding::Float(FloatFormat::Bf16) => "BF16",
        FieldEncoding::Float(FloatFormat::E4m3Fn) => "F8_E4M3",
        FieldEncoding::Float(FloatFormat::E8m0) => "F8_E8M0",
        FieldEncoding::Float(FloatFormat::E2m1) => {
            panic!(
                "an unpacked E2M1 operand has no safetensors dtype: E2M1 is always sub-byte packed"
            )
        }
        FieldEncoding::Unsigned { .. } | FieldEncoding::Signed => match operand.bits {
            8 => "I8",
            16 => "I16",
            32 => "I32",
            64 => "I64",
            other => panic!("an unpacked {other}-bit integer operand has no safetensors dtype"),
        },
    }
}

pub(crate) fn compare_selection(
    selection: &PackedSelectionRow,
    field: PackedSelectionField,
    expected: impl fmt::Display,
    actual: impl fmt::Display,
) -> Result<(), PackedSafetensorsError> {
    let expected = expected.to_string();
    let actual = actual.to_string();
    if expected != actual {
        return Err(PackedSafetensorsError::SelectionMismatch {
            linear_id: selection.linear_id.clone(),
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

pub(crate) fn expected_weight_dtype(descriptor: PackedWeight) -> &'static str {
    operand_stored_dtype(planar_operand(descriptor, OperandRole::Codes))
}

pub(crate) fn expected_scale_dtype(descriptor: PackedWeight) -> &'static str {
    operand_stored_dtype(planar_operand(descriptor, OperandRole::Scale))
}

pub(crate) fn report_selected_accounting(
    report: &mut PackedLoadReport,
    selections: &[ValidatedSelection],
) -> Result<(), PackedSafetensorsError> {
    for row in selections {
        let descriptor = row.selection.descriptor;
        report.selected_weight_source_bytes = checked_add(
            report.selected_weight_source_bytes,
            descriptor.source_bytes(SourceRole::Planar(OperandRole::Codes)),
            "selected_weight_source_bytes",
        )?;
        report.selected_scale_source_bytes = checked_add(
            report.selected_scale_source_bytes,
            descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale)),
            "selected_scale_source_bytes",
        )?;
        report.source_padding_bits = checked_add(
            report.source_padding_bits,
            packed_source_padding_bits(descriptor)?,
            "source_padding_bits",
        )?;
        report.packed_source_bytes = checked_add(
            report.packed_source_bytes,
            descriptor.total_source_bytes(),
            "packed_source_bytes",
        )?;
        report.forbidden_f32_weight_bytes = checked_add(
            report.forbidden_f32_weight_bytes,
            checked_mul(
                descriptor.logical_values(),
                poot_tensor::DType::F32.byte_size(),
                "forbidden_f32_weight_bytes",
            )?,
            "forbidden_f32_weight_bytes",
        )?;
    }
    report.peak_unpublished_payload_bytes = report.packed_source_bytes;
    Ok(())
}

fn checked_mul(
    lhs: usize,
    rhs: usize,
    field: &'static str,
) -> Result<usize, PackedSafetensorsError> {
    lhs.checked_mul(rhs)
        .ok_or(PackedSafetensorsError::ArithmeticOverflow { field })
}

/// The unused high-nibble padding bits a ragged E2M1 row's last byte holds (zero for a format
/// with no sub-byte packing): the byte grid the `Codes` source stages minus the exact bits its
/// values need, both read off the descriptor - no layout fact restated.
fn packed_source_padding_bits(descriptor: PackedWeight) -> Result<usize, PackedSafetensorsError> {
    let codes = SourceRole::Planar(OperandRole::Codes);
    let codes_bits = descriptor
        .format()
        .descriptor()
        .planar_operand(OperandRole::Codes)
        .expect("a packed weight has a Codes operand")
        .bits as usize;
    let allocated_bits = checked_mul(descriptor.source_bytes(codes), 8, "source padding bits")?;
    let needed_bits = checked_mul(
        descriptor.logical_values(),
        codes_bits,
        "source padding bits",
    )?;
    allocated_bits
        .checked_sub(needed_bits)
        .ok_or(PackedSafetensorsError::ArithmeticOverflow {
            field: "source padding bits",
        })
}

/// One resident file's optional integrity pin (card 544, ADR-0103 decision 4): the exact length
/// and digest a manifest names for it. A manifest is optional integrity data, never the admission
/// gate - a file with no pin is still admitted, by whatever structural (capability) check the
/// caller applies to the bytes [`admit_file`] or [`admit_shard`] returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FilePin {
    pub length: usize,
    pub sha256: Sha256Digest,
}

/// The pin comparison at the heart of [`admit_file`]/[`admit_shard`] (card 544), for a caller
/// that already holds the bytes and needs no read of its own - the byte-buffer counterpart of
/// `admit_file` for exactly that case (a family module's own small pinned file, such as a
/// checkpoint's tensor index). `None` admits `bytes` unconditionally, by capability; nothing
/// outside this function and [`admit_file`]'s own length-probe fast path performs a pin
/// comparison, so every caller shares one definition of "matches its pin."
pub(crate) fn check_pin(
    bytes: &[u8],
    pin: Option<FilePin>,
    name: &str,
) -> Result<(), PackedSafetensorsError> {
    if let Some(pin) = pin {
        if bytes.len() != pin.length {
            return Err(PackedSafetensorsError::ArtifactChanged {
                file: name.to_string(),
            });
        }
        if sha256_digest(bytes) != pin.sha256 {
            return Err(PackedSafetensorsError::DigestMismatch {
                file: name.to_string(),
            });
        }
    }
    Ok(())
}

/// [`check_pin`] for a file too large to hold: the reader is hashed in fixed chunks, once, and its
/// total length and digest compared with `pin`. A short or long file is `ArtifactChanged`, a
/// different digest `DigestMismatch`, a failed read `Io`; the same errors as the byte form.
pub fn check_pin_reader(
    mut reader: impl std::io::Read,
    pin: FilePin,
    name: &str,
) -> Result<(), PackedSafetensorsError> {
    let mut context = Context::new(&SHA256);
    let mut buffer = [0u8; HASH_BUFFER_BYTES];
    let mut total = 0usize;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| PackedSafetensorsError::Io {
                file: name.to_string(),
                error,
            })?;
        if read == 0 {
            break;
        }
        context.update(&buffer[..read]);
        total += read;
    }
    if total != pin.length {
        return Err(PackedSafetensorsError::ArtifactChanged {
            file: name.to_string(),
        });
    }
    let mut digest = [0u8; 32];
    digest.copy_from_slice(context.finish().as_ref());
    if Sha256Digest(digest) != pin.sha256 {
        return Err(PackedSafetensorsError::DigestMismatch {
            file: name.to_string(),
        });
    }
    Ok(())
}

/// The one generic integrity step (card 544) every small file the loader reads - `config.json`,
/// the shard index, a tokenizer file, a chat template - goes through: admit it with exactly one
/// read of its content. A pin, when given, is checked against a digest computed from that same
/// read, never a second pass over the file (R475-014); its absence admits the bytes
/// unauthenticated, so a finetune or new revision with no pinned digest is not refused by
/// identity (R475-005, R476-018) - the caller's own structural check on the returned bytes is
/// what decides admission.
pub(crate) fn admit_file(
    file: &mut dyn RetainedReader,
    name: &str,
    pin: Option<FilePin>,
) -> Result<Vec<u8>, PackedSafetensorsError> {
    let actual_length = file_length(file, name)?;
    if let Some(pin) = pin
        && actual_length != pin.length
    {
        return Err(PackedSafetensorsError::ArtifactChanged {
            file: name.to_string(),
        });
    }
    let bytes = read_exact_file(file, name, actual_length)?;
    check_pin(&bytes, pin, name)?;
    Ok(bytes)
}

/// The shard counterpart of [`admit_file`], over a shard already mapped once. A pin, when given, is
/// compared by [`check_pin`] against the mapping itself (length, then digest), so one shard is hashed
/// at most once per load and never re-read; with no pin nothing is hashed and the shard is admitted
/// by capability, by the structural checks that parsed its header. Returns the bytes hashed.
pub(crate) fn admit_shard(
    shard: &StoredBytes,
    name: &str,
    pin: Option<FilePin>,
) -> Result<usize, PackedSafetensorsError> {
    let Some(pin) = pin else {
        return Ok(0);
    };
    check_pin(shard.as_slice(), Some(pin), name)?;
    Ok(shard.len())
}

/// The cache identity is content-derived, not pin-derived (card 544): `identity` carries the
/// repository/revision label plus every length and digest the reader actually observed - the
/// pinned value when a manifest names one, the bytes' own computed digest otherwise - so two
/// different capability-admitted checkpoints never collide in a shared owner cache even with no
/// manifest to tell them apart.
pub(crate) fn artifact_cache_identity(identity: &ArtifactIdentityFields) -> Sha256Digest {
    let mut context = Context::new(&SHA256);
    update_identity_string(&mut context, &identity.repository);
    update_identity_string(&mut context, &identity.revision);
    update_identity_usize(&mut context, identity.config_length);
    context.update(&identity.config_sha256.0);
    update_identity_usize(&mut context, identity.index_length);
    context.update(&identity.index_sha256.0);
    for shard in &identity.shards {
        update_identity_string(&mut context, &shard.filename);
        update_identity_usize(&mut context, shard.file_length);
        context.update(&shard.file_sha256.0);
        update_identity_usize(&mut context, shard.header_length);
        context.update(&shard.header_sha256.0);
    }
    Sha256Digest::from(
        <[u8; 32]>::try_from(context.finish().as_ref())
            .expect("SHA-256 output is exactly 32 bytes"),
    )
}

pub(crate) fn update_identity_string(context: &mut Context, value: &str) {
    update_identity_usize(context, value.len());
    context.update(value.as_bytes());
}

pub(crate) fn update_identity_usize(context: &mut Context, value: usize) {
    context.update(&(value as u128).to_le_bytes());
}

pub(crate) fn checked_add(
    lhs: usize,
    rhs: usize,
    field: &'static str,
) -> Result<usize, PackedSafetensorsError> {
    lhs.checked_add(rhs)
        .ok_or(PackedSafetensorsError::ArithmeticOverflow { field })
}

pub(crate) struct UniqueJson(pub(crate) serde_json::Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct UniqueJsonVisitor;

        impl<'de> Visitor<'de> for UniqueJsonVisitor {
            type Value = UniqueJson;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("non-finite JSON number"))?;
                Ok(UniqueJson(serde_json::Value::Number(number)))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                self.visit_string(value.to_string())
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::String(value)))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Null))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(serde_json::Value::Null))
            }

            fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
            where
                D: Deserializer<'de>,
            {
                UniqueJson::deserialize(deserializer)
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(UniqueJson(value)) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(UniqueJson(serde_json::Value::Array(values)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(<A::Error as de::Error>::custom(format!(
                            "duplicate JSON key {key:?}"
                        )));
                    }
                    let UniqueJson(value) = map.next_value()?;
                    values.insert(key, value);
                }
                Ok(UniqueJson(serde_json::Value::Object(values)))
            }
        }

        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

/// `bytes` as a JSON value, refusing an object that repeats a key at any depth: `serde_json::Value`
/// alone keeps the last of a repeated key, so a config's earlier setting would be dropped unseen.
pub fn parse_json_unique_keys(bytes: &[u8]) -> Result<serde_json::Value, serde_json::Error> {
    serde_json::from_slice::<UniqueJson>(bytes).map(|value| value.0)
}

pub(crate) fn parse_unique_json(
    bytes: &[u8],
    filename: &str,
) -> Result<serde_json::Value, PackedSafetensorsError> {
    parse_json_unique_keys(bytes).map_err(|error| {
        let reason = error.to_string();
        if let Some(rest) = reason.strip_prefix("duplicate JSON key \"") {
            let key = rest.split('"').next().unwrap_or(rest).to_string();
            PackedSafetensorsError::DuplicateJsonKey {
                file: filename.to_string(),
                key,
            }
        } else {
            PackedSafetensorsError::Json {
                file: filename.to_string(),
                reason,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use poot_quant::format::{ScaleEncoding, WeightFormat};

    use super::*;

    fn header(dtype: &str, shape: &[usize], data_offsets: [usize; 2]) -> serde_json::Value {
        serde_json::json!({
            "w": {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": data_offsets,
            }
        })
    }

    /// SC-001: `crate::dtype::element_width_bytes` (the one width table) backs `parse_header_entries`'s
    /// byte-count check, the "validation row" that must move with the safetensors reader's own
    /// byte-count check when a dtype's width changes (R484-006).
    #[test]
    fn parse_header_entries_enforces_the_shared_width_table() {
        // F32 x 4 elements = 16 bytes: matches element_width_bytes("F32") * 4.
        let ok = parse_header_entries(&header("F32", &[4], [0, 16]), "shard.safetensors", 16)
            .expect("16 bytes matches F32 x 4");
        assert_eq!(ok.get("w").unwrap().shape, vec![4]);

        // One byte short of element_width_bytes("F32") * 4 = 16.
        let err = parse_header_entries(&header("F32", &[4], [0, 15]), "shard.safetensors", 15)
            .expect_err("15 bytes must mismatch F32 x 4 = 16 bytes");
        assert!(
            matches!(
                err,
                PackedSafetensorsError::TensorByteCountMismatch {
                    expected: 16,
                    actual: 15,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// A dtype string `element_width_bytes` does not recognize skips the byte-count check instead of
    /// panicking or silently mismatching (matches `poot_tensor::DType::parse`'s
    /// typed-error contract: the unrecognized string never reaches a width comparison).
    #[test]
    fn parse_header_entries_skips_the_byte_count_check_for_an_unrecognized_dtype() {
        let ok = parse_header_entries(&header("COMPLEX64", &[4], [0, 1]), "shard.safetensors", 1)
            .expect("an unrecognized dtype must not fail the byte-count check");
        assert_eq!(ok.get("w").unwrap().dtype, "COMPLEX64");
    }

    /// ADR-0104 (card 542d): the loader admits any format with a descriptor
    /// and a well-formed payload - whether a format executes is decided at planning, never by a
    /// loader-side registry. `E4m3PerChannel` was never one of the four `PackedLinearFormat`
    /// cells card 542d deleted; its expected weight/scale dtypes are still derived here, straight
    /// off the descriptor, with no second table and no `.expect("registered")`.
    ///
    /// Mutation (card 542d, run 2026-09-29): reintroduced a loader-side registry gate in
    /// `expected_weight_dtype`/`expected_scale_dtype` (`if !matches!(descriptor.format(),
    /// WeightFormat::E4m3Block128 { .. } | WeightFormat::E2m1Row32) { panic!("unregistered packed
    /// format: {:?}", descriptor.format()) }`) -> this test panicked with "unregistered packed
    /// format: E4m3PerChannel { scale: F32 }" instead of returning `"F8_E4M3"`/`"F32"`. Reverted.
    #[test]
    fn expected_dtype_derives_from_the_descriptor_for_an_unregistered_planar_format() {
        let format = WeightFormat::E4m3PerChannel {
            scale: ScaleEncoding::F32,
        };
        let descriptor = PackedWeight::try_new(format, [4, 8]).expect("a well-formed descriptor");
        assert_eq!(expected_weight_dtype(descriptor), "F8_E4M3");
        assert_eq!(expected_scale_dtype(descriptor), "F32");
    }

    struct CursorReader(std::io::Cursor<Vec<u8>>);

    impl CursorReader {
        fn new(bytes: Vec<u8>) -> Self {
            Self(std::io::Cursor::new(bytes))
        }
    }

    impl std::io::Read for CursorReader {
        fn read(&mut self, buffer: &mut [u8]) -> Result<usize, std::io::Error> {
            std::io::Read::read(&mut self.0, buffer)
        }
    }

    impl std::io::Seek for CursorReader {
        fn seek(&mut self, position: SeekFrom) -> Result<u64, std::io::Error> {
            std::io::Seek::seek(&mut self.0, position)
        }
    }

    impl RetainedReader for CursorReader {
        fn file_length(&self) -> Result<u64, std::io::Error> {
            Ok(self.0.get_ref().len() as u64)
        }
    }

    /// Card 544 SC-003: a pinned file (standing in for a tokenizer or chat-template file - every
    /// small file the loader reads goes through this same `admit_file`, already the real path
    /// `authenticate_retained_config_bytes` uses to admit the shard index) with a wrong digest is
    /// refused.
    ///
    /// Mutation (as SC-003 names it): read the file outside the integrity step - call
    /// `read_exact_file` directly instead of `admit_file`. The row that depends on going through
    /// `admit_file` goes red under that bypass, which the second half of this test demonstrates
    /// directly: the bypass admits exactly the tampered bytes the integrity step refused.
    #[test]
    fn admit_file_refuses_a_wrong_digest_and_a_bypass_would_miss_it() {
        let expected = b"tokenizer.json: the pinned contents".to_vec();
        let tampered = b"tokenizer.json: TAMPERED CONTENTS!!".to_vec();
        assert_eq!(expected.len(), tampered.len(), "mutation preserves length");
        assert_ne!(expected, tampered);
        let pin = FilePin {
            length: expected.len(),
            sha256: sha256_digest(&expected),
        };

        let mut through_the_integrity_step = CursorReader::new(tampered.clone());
        let error = admit_file(&mut through_the_integrity_step, "tokenizer.json", Some(pin))
            .expect_err("a pinned file with a wrong digest must be refused");
        assert!(
            matches!(error, PackedSafetensorsError::DigestMismatch { .. }),
            "{error:?}"
        );

        let mut bypassing_the_integrity_step = CursorReader::new(tampered.clone());
        let bypassed = read_exact_file(
            &mut bypassing_the_integrity_step,
            "tokenizer.json",
            pin.length,
        )
        .expect("a direct read outside the integrity step performs no digest check");
        assert_eq!(
            bypassed, tampered,
            "the bypass admits exactly the tampered bytes admit_file refused above"
        );
    }

    /// The streamed pin check agrees with the whole-buffer one: an input several chunks long with a
    /// partial last chunk, one exactly a chunk long, and an empty file are admitted under their own
    /// pin, refused as `DigestMismatch` when one byte differs and as `ArtifactChanged` under a wrong
    /// length. Mutation: drop the last byte of every read from the hash in `check_pin_reader`
    /// (`context.update(&buffer[..read - 1])`); the multi-chunk row fails with `DigestMismatch` for
    /// the file it should admit.
    #[test]
    fn the_streamed_pin_check_equals_the_whole_buffer_one() {
        for len in [0, HASH_BUFFER_BYTES, 3 * HASH_BUFFER_BYTES + 17] {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
            let pin = FilePin {
                length: len,
                sha256: sha256_digest(&bytes),
            };
            check_pin_reader(bytes.as_slice(), pin, "f").unwrap_or_else(|e| panic!("{len}: {e:?}"));
            check_pin(&bytes, Some(pin), "f").unwrap();
            let mut other = bytes.clone();
            if let Some(last) = other.last_mut() {
                *last ^= 1;
                let error = check_pin_reader(other.as_slice(), pin, "f").unwrap_err();
                assert!(matches!(
                    error,
                    PackedSafetensorsError::DigestMismatch { .. }
                ));
            }
            let short = FilePin {
                length: len + 1,
                ..pin
            };
            let error = check_pin_reader(bytes.as_slice(), short, "f").unwrap_err();
            assert!(matches!(
                error,
                PackedSafetensorsError::ArtifactChanged { .. }
            ));
        }
    }
}

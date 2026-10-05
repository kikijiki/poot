use poot_tensor::DType;
use std::collections::HashSet;
use std::sync::Arc;

use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore, WeightStoreBuilder};

use super::*;

struct UniqueSafetensorsHeader(serde_json::Map<String, serde_json::Value>);

impl<'de> serde::Deserialize<'de> for UniqueSafetensorsHeader {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> serde::de::Visitor<'de> for HeaderVisitor {
            type Value = UniqueSafetensorsHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors JSON object with unique tensor names")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some((name, value)) = map.next_entry::<String, serde_json::Value>()? {
                    if values.insert(name.clone(), value).is_some() {
                        return Err(<A::Error as serde::de::Error>::custom(format!(
                            "duplicate safetensors tensor name {name:?} within one header"
                        )));
                    }
                }
                Ok(UniqueSafetensorsHeader(values))
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

fn parse_safetensors_header(
    header_bytes: &[u8],
) -> Result<serde_json::Map<String, serde_json::Value>, LoadError> {
    serde_json::from_slice::<UniqueSafetensorsHeader>(header_bytes)
        .map(|header| header.0)
        .map_err(|error| {
            let text = error.to_string();
            if let Some(rest) = text.strip_prefix("duplicate safetensors tensor name ") {
                let name = rest
                    .split(" within one header")
                    .next()
                    .unwrap_or(rest)
                    .trim_matches('"')
                    .to_string();
                LoadError::DuplicateTensorName {
                    name,
                    scope: "within one header".to_string(),
                }
            } else {
                LoadError::Json(error)
            }
        })
}

/// One safetensors header entry, parsed and validated against `data_len` (ADR-0103's
/// untrusted-header hardening: `data_offsets` in range, the shape's element count and its
/// dtype-implied byte count fit `usize` and agree with the actual span).
struct ParsedTensorHeaderEntry {
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

fn parse_tensor_header_entry(
    name: &str,
    meta: &serde_json::Value,
    data_len: usize,
) -> Result<ParsedTensorHeaderEntry, LoadError> {
    let dtype_str = meta["dtype"].as_str().ok_or_else(|| miss(name, "dtype"))?;
    let shape: Vec<usize> = meta["shape"]
        .as_array()
        .ok_or_else(|| miss(name, "shape"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let dim = value.as_u64().ok_or_else(|| {
                LoadError::SafeTensors(format!(
                    "{name}: shape[{index}] is not a non-negative integer"
                ))
            })?;
            usize::try_from(dim).map_err(|_| {
                LoadError::SafeTensors(format!("{name}: shape[{index}] does not fit usize"))
            })
        })
        .collect::<Result<_, _>>()?;
    let offs = meta["data_offsets"]
        .as_array()
        .ok_or_else(|| miss(name, "data_offsets"))?;
    // Untrusted header: `data_offsets` must be exactly [start, end], both integers, within
    // the data section. Otherwise a malformed file panics on `offs[1]`, the `.as_u64()`
    // unwrap, or the `data[start..end]` slice; each is a `LoadError` instead.
    if offs.len() != 2 {
        return Err(LoadError::SafeTensors(format!(
            "{name}: data_offsets must be [start, end]"
        )));
    }
    let start_u64 = offs[0]
        .as_u64()
        .ok_or_else(|| LoadError::SafeTensors(format!("{name}: data_offsets[0] not a u64")))?;
    let end_u64 = offs[1]
        .as_u64()
        .ok_or_else(|| LoadError::SafeTensors(format!("{name}: data_offsets[1] not a u64")))?;
    let start = usize::try_from(start_u64).map_err(|_| {
        LoadError::SafeTensors(format!("{name}: data_offsets[0] does not fit usize"))
    })?;
    let end = usize::try_from(end_u64).map_err(|_| {
        LoadError::SafeTensors(format!("{name}: data_offsets[1] does not fit usize"))
    })?;
    if start > end || end > data_len {
        return Err(LoadError::SafeTensors(format!(
            "{name}: data_offsets [{start}, {end}] out of range for {data_len} data bytes"
        )));
    }
    // Reject a shape whose element count overflows usize: the count only feeds the checks
    // below, but a wrap could coincidentally match and admit a wrong-shaped tensor. `raw`
    // is already bounded; this is defense in depth, as in the GGUF path.
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            LoadError::SafeTensors(format!("{name}: shape element count overflows usize"))
        })?;
    let dtype = parse_dtype(dtype_str)?;
    let expected_bytes = numel.checked_mul(dtype.byte_size()).ok_or_else(|| {
        LoadError::SafeTensors(format!(
            "{name}: byte count for {dtype_str} shape {shape:?} overflows usize"
        ))
    })?;
    let actual_bytes = end - start;
    if actual_bytes != expected_bytes {
        return Err(LoadError::SafeTensors(format!(
            "{name}: {actual_bytes} bytes for {dtype_str} shape {shape:?}, expected {expected_bytes}"
        )));
    }
    Ok(ParsedTensorHeaderEntry {
        dtype,
        shape,
        start,
        end,
    })
}

/// Every non-metadata header entry's stored bytes, inserted as a [`WeightEntry::Dense`] keyed by
/// its checkpoint tensor name; `read_range` supplies the raw bytes exactly as the checkpoint
/// stored them (ADR-0103 decision 1: no widening, no decode). `I64` entries (index/position
/// buffers, e.g. BERT's `embeddings.position_ids`) are skipped: no consumer reads them and they
/// are never a weight (matches the reader's historical behavior).
/// `insert_header_entries`'s cross-call duplicate-name tracking: every non-metadata header key seen
/// so far, regardless of dtype (an I64 name never becomes a store entry, but a second file naming
/// it is still the same real-world duplicate-tensor bug the store-keyed check below catches for
/// every other dtype).
fn insert_header_entries<'a>(
    header_bytes: &[u8],
    data_len: usize,
    mut read_range: impl FnMut(usize, usize) -> Result<Cow<'a, [u8]>, LoadError>,
    builder: &mut WeightStoreBuilder,
    seen_names: &mut HashSet<String>,
) -> Result<(), LoadError> {
    let obj = parse_safetensors_header(header_bytes)?;
    for (name, meta) in &obj {
        if name == "__metadata__" {
            continue;
        }
        if !seen_names.insert(name.clone()) {
            return Err(LoadError::DuplicateTensorName {
                name: name.clone(),
                scope: "across model shards".to_string(),
            });
        }
        let ParsedTensorHeaderEntry {
            dtype,
            shape,
            start,
            end,
        } = parse_tensor_header_entry(name, meta, data_len)?;
        if dtype == DType::I64 {
            continue;
        }
        let raw = read_range(start, end)?;
        let bytes: Arc<[u8]> = Arc::from(raw.into_owned());
        let dense = DenseWeight::try_new(dtype, shape, bytes)
            .map_err(|error| LoadError::SafeTensors(format!("{name}: {error}")))?;
        // `seen_names` above already proves this name is unique across every call into this
        // builder (within a header, `UniqueSafetensorsHeader` also rejects a repeat), so this
        // insert cannot collide.
        builder
            .insert(name.clone(), WeightEntry::Dense(dense))
            .expect("seen_names already proved this key unique");
    }
    Ok(())
}

fn insert_file(
    path: &Path,
    builder: &mut WeightStoreBuilder,
    seen_names: &mut HashSet<String>,
) -> Result<(), LoadError> {
    let mut file = File::open(path)?;
    let file_len = usize::try_from(file.metadata()?.len())
        .map_err(|_| LoadError::SafeTensors("file length does not fit usize".into()))?;
    if file_len < 8 {
        return Err(LoadError::SafeTensors("file too small".into()));
    }
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_end = checked_header_end(u64::from_le_bytes(len_bytes), file_len)?;
    let mut header_bytes = vec![0u8; header_end - 8];
    file.read_exact(&mut header_bytes)?;
    let data_len = file_len - header_end;
    insert_header_entries(
        &header_bytes,
        data_len,
        |start, end| {
            let absolute = header_end.checked_add(start).ok_or_else(|| {
                LoadError::SafeTensors("tensor file offset overflows usize".into())
            })?;
            file.seek(SeekFrom::Start(u64::try_from(absolute).map_err(|_| {
                LoadError::SafeTensors("tensor file offset does not fit u64".into())
            })?))?;
            let mut raw = vec![0u8; end - start];
            file.read_exact(&mut raw)?;
            Ok(Cow::Owned(raw))
        },
        builder,
        seen_names,
    )
}

#[cfg(any(test, feature = "test-support"))]
fn insert_bytes(bytes: &[u8], builder: &mut WeightStoreBuilder) -> Result<(), LoadError> {
    if bytes.len() < 8 {
        return Err(LoadError::SafeTensors("file too small".into()));
    }
    let header_end = checked_header_end(
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        bytes.len(),
    )?;
    let data = &bytes[header_end..];
    let mut seen_names = HashSet::new();
    insert_header_entries(
        &bytes[8..header_end],
        data.len(),
        |start, end| Ok(Cow::Borrowed(&data[start..end])),
        builder,
        &mut seen_names,
    )
}

/// Unique shard file names a `model.safetensors.index.json`'s `weight_map` names, sorted so shards
/// are read in a deterministic order (each shard is read exactly once regardless of how many
/// tensors it contributes).
fn shard_files(index: &serde_json::Value) -> Result<Vec<String>, LoadError> {
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| LoadError::SafeTensors("index has no weight_map".into()))?;
    let mut shards: Vec<String> = weight_map
        .values()
        .map(|value| {
            value.as_str().map(str::to_string).ok_or_else(|| {
                LoadError::SafeTensors("index weight_map shard is not a string".into())
            })
        })
        .collect::<Result<_, _>>()?;
    shards.sort();
    shards.dedup();
    Ok(shards)
}

/// Load a model's weights from a directory (single `model.safetensors` or the sharded
/// `model.safetensors.index.json` layout) into a [`WeightStore`]: every tensor's bytes exactly as
/// the checkpoint stored them, keyed by its checkpoint name (ADR-0103 decisions 1 and 3, card
/// 540b). No f32 widening happens here; a consumer that needs f32 (the CPU oracle, or a GPU
/// backend with no native upload lane for the stored dtype) decodes on demand from the returned
/// entries.
pub fn load_weight_store(dir: impl AsRef<Path>) -> Result<WeightStore, LoadError> {
    let dir = dir.as_ref();
    let mut builder = WeightStore::builder();
    let mut seen_names = HashSet::new();
    let single = dir.join("model.safetensors");
    if single.exists() {
        insert_file(&single, &mut builder, &mut seen_names)?;
        return Ok(builder.build());
    }
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::SafeTensors(
            "no model.safetensors or model.safetensors.index.json in model dir".into(),
        ));
    }
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    for shard in shard_files(&index)? {
        insert_file(&dir.join(shard), &mut builder, &mut seen_names)?;
    }
    Ok(builder.build())
}

/// Card 545a: `store` with every quantized linear of a `scheme`
/// checkpoint re-keyed as ONE packed weight, `{prefix}.weight` -> [`WeightEntry::Packed`], whose
/// sources are the checkpoint's own tensors byte for byte (the `poot-quant` planar descriptors state
/// exactly the on-disk layouts): GPTQ `qweight`/`qzeros`/`scales` (+ `g_idx` when act-order is real),
/// AWQ `qweight`/`qzeros`/`scales`, FP8 `weight` with a per-output-row `weight_scale` (compressed-tensors)
/// or a 128x128-block `weight_scale_inv` (DeepSeek style; F32, BF16 or E8M0 scales).
/// Nothing is decoded; the payload's own content checks run here, so a bad group index or a
/// non-finite scale is refused at load. Every other entry passes through unchanged.
///
/// This is the one reader of a block-FP8 linear (Card 654): the host dequantizer that widened one to
/// f32 is gone.
///
/// ```compile_fail,E0425
/// let _ = poot_load::safetensors::dequant_fp8_blockwise;
/// ```
pub fn pack_quantized_linears(
    store: &WeightStore,
    scheme: QuantScheme,
) -> Result<WeightStore, LoadError> {
    use poot_quant::format::{GroupMap, OperandRole, ScaleEncoding, WeightFormat};
    use poot_quant::{PackedPayload, PackedWeight, SourceRole};

    // Every layout refusal names the checkpoint tensor it is about (SC-008).
    let layout =
        |tensor: String, reason: String| LoadError::QuantizedLinearLayout { tensor, reason };
    let part = |prefix: &str, suffix: &str, dtype: DType| -> Result<DenseWeight, LoadError> {
        let name = format!("{prefix}{suffix}");
        match store.get(&name) {
            Some(WeightEntry::Dense(dense)) if dense.dtype() == dtype => Ok(dense.clone()),
            Some(WeightEntry::Dense(dense)) => Err(layout(
                name,
                format!("stored {:?}, expected {dtype:?}", dense.dtype()),
            )),
            _ => Err(layout(name, "missing".to_string())),
        }
    };
    let nonzero = |tensor: String, n: usize, what: &str| {
        std::num::NonZeroUsize::new(n).ok_or_else(|| layout(tensor, format!("{what} is 0")))
    };
    // The checkpoint's `quantization_config.group_size` is one value for the whole checkpoint, not
    // per tensor (card 545b): validated once here, not once per prefix inside the
    // loop below, and refused through a typed config error (not `layout`, whose `tensor` field is
    // documented as the checkpoint tensor a refusal is about, not a config key).
    //
    // Card 545b: this enum, not `Option<NonZeroUsize>` plus an `.expect()`
    // per kind inside the loop, is what proves a Gptq/Awq scheme always carries its validated
    // `group_size` and an Fp8 scheme never does - a mismatch is a compile error, not a runtime panic
    // path standing in for one.
    #[derive(Clone, Copy)]
    enum ValidatedScheme {
        Gptq(std::num::NonZeroUsize),
        Awq(std::num::NonZeroUsize),
        Fp8,
    }
    let nonzero_group_size = || {
        std::num::NonZeroUsize::new(scheme.group_size).ok_or(LoadError::InvalidQuantConfig {
            field: "group_size",
            value: scheme.group_size,
        })
    };
    let validated_scheme = match scheme.kind {
        QuantKind::Gptq => ValidatedScheme::Gptq(nonzero_group_size()?),
        QuantKind::Awq => ValidatedScheme::Awq(nonzero_group_size()?),
        QuantKind::Fp8 => ValidatedScheme::Fp8,
    };
    // One prefix per linear (a set: a checkpoint naming both FP8 scale spellings is refused below).
    let prefixes: std::collections::BTreeSet<String> = store
        .keys()
        .filter_map(|key| match scheme.kind {
            QuantKind::Gptq | QuantKind::Awq => key.as_str().strip_suffix(".qweight"),
            QuantKind::Fp8 => {
                let key = key.as_str();
                key.strip_suffix(".weight_scale")
                    .or_else(|| key.strip_suffix(".weight_scale_inv"))
                    .filter(|prefix| store.contains(&format!("{prefix}.weight")))
            }
        })
        .map(str::to_string)
        .collect();
    // Fail closed (Card 654 review): an `F8_E4M3` linear with no recognized scale sibling (none, or
    // DeepSeek-V4's `.scale`) is not a packed linear, and its raw codes must never pass through as
    // dense unscaled weights.
    if let Some(unscaled) = store.iter().find_map(|(key, entry)| {
        let prefix = key.as_str().strip_suffix(".weight")?;
        let WeightEntry::Dense(dense) = entry else {
            return None;
        };
        (dense.dtype() == DType::E4M3FN && !prefixes.contains(prefix))
            .then(|| key.as_str().to_string())
    }) {
        return Err(layout(
            unscaled,
            "F8_E4M3 weight has no .weight_scale or .weight_scale_inv sibling".to_string(),
        ));
    }
    let mut consumed: HashSet<String> = HashSet::new();
    let mut packed: Vec<(String, PackedPayload)> = Vec::with_capacity(prefixes.len());
    for prefix in &prefixes {
        let prefix = prefix.as_str();
        // The format, logical `[out, K]`, the payload sources and the checkpoint tensors they consume.
        type Linear = (
            WeightFormat,
            [usize; 2],
            Vec<(SourceRole, Arc<[u8]>)>,
            Vec<&'static str>,
        );
        let (format, shape, sources, parts): Linear = match validated_scheme {
            ValidatedScheme::Gptq(group_size) => {
                let qweight = part(prefix, ".qweight", DType::I32)?;
                let qzeros = part(prefix, ".qzeros", DType::I32)?;
                let scales = part(prefix, ".scales", DType::F16)?;
                let g_idx = part(prefix, ".g_idx", DType::I32)?;
                let [k8, out] = qweight.shape()[..] else {
                    return Err(layout(
                        format!("{prefix}.qweight"),
                        format!("shape {:?}, expected [K/8, out]", qweight.shape()),
                    ));
                };
                let k = k8 * 8;
                let [groups, _] = scales.shape()[..] else {
                    return Err(layout(
                        format!("{prefix}.scales"),
                        format!("shape {:?}, expected [groups, out]", scales.shape()),
                    ));
                };
                if g_idx.shape() != [k] {
                    return Err(layout(
                        format!("{prefix}.g_idx"),
                        format!("shape {:?}, expected [{k}]", g_idx.shape()),
                    ));
                }
                // A g_idx that is exactly `k / group_size` (no act-order) is the contiguous
                // map; any other order is read per K through the group-index source.
                let contiguous =
                    g_idx
                        .bytes()
                        .as_slice()
                        .chunks_exact(4)
                        .enumerate()
                        .all(|(k, word)| {
                            i32::from_le_bytes(word.try_into().unwrap()) as usize
                                == k / group_size.get()
                        });
                let mut sources = vec![
                    (
                        SourceRole::Planar(OperandRole::Codes),
                        qweight.bytes().as_arc(),
                    ),
                    (
                        SourceRole::Planar(OperandRole::Zero),
                        qzeros.bytes().as_arc(),
                    ),
                    (
                        SourceRole::Planar(OperandRole::Scale),
                        scales.bytes().as_arc(),
                    ),
                ];
                let groups = if contiguous {
                    GroupMap::Contiguous { size: group_size }
                } else {
                    sources.push((
                        SourceRole::Planar(OperandRole::GroupIndex),
                        g_idx.bytes().as_arc(),
                    ));
                    GroupMap::Indexed {
                        groups: nonzero(format!("{prefix}.scales"), groups, "the group count")?,
                    }
                };
                (
                    WeightFormat::Gptq { groups },
                    [out, k],
                    sources,
                    vec![".qweight", ".qzeros", ".scales", ".g_idx"],
                )
            }
            ValidatedScheme::Awq(group_size) => {
                let qweight = part(prefix, ".qweight", DType::I32)?;
                let qzeros = part(prefix, ".qzeros", DType::I32)?;
                let scales = part(prefix, ".scales", DType::F16)?;
                let [k, out8] = qweight.shape()[..] else {
                    return Err(layout(
                        format!("{prefix}.qweight"),
                        format!("shape {:?}, expected [K, out/8]", qweight.shape()),
                    ));
                };
                (
                    WeightFormat::Awq { group_size },
                    [out8 * 8, k],
                    vec![
                        (
                            SourceRole::Planar(OperandRole::Codes),
                            qweight.bytes().as_arc(),
                        ),
                        (
                            SourceRole::Planar(OperandRole::Zero),
                            qzeros.bytes().as_arc(),
                        ),
                        (
                            SourceRole::Planar(OperandRole::Scale),
                            scales.bytes().as_arc(),
                        ),
                    ],
                    vec![".qweight", ".qzeros", ".scales"],
                )
            }
            ValidatedScheme::Fp8 => {
                let weight = part(prefix, ".weight", DType::E4M3FN)?;
                let [out, k] = weight.shape()[..] else {
                    return Err(layout(
                        format!("{prefix}.weight"),
                        format!("shape {:?}, expected [out, K]", weight.shape()),
                    ));
                };
                // compressed-tensors per-channel (`weight_scale [out, 1]`) or DeepSeek-style block
                // (`weight_scale_inv [ceil(out/128), ceil(K/128)]`); naming both is ambiguous.
                let (per_row, per_block) = (
                    format!("{prefix}.weight_scale"),
                    format!("{prefix}.weight_scale_inv"),
                );
                let (suffix, block) = match (store.contains(&per_row), store.contains(&per_block)) {
                    (true, true) => {
                        return Err(LoadError::Fp8AmbiguousScale {
                            prefix: prefix.to_string(),
                            first: per_row,
                            second: per_block,
                        });
                    }
                    (_, true) => (".weight_scale_inv", true),
                    _ => (".weight_scale", false),
                };
                let scale_name = format!("{prefix}{suffix}");
                let (scale, encoding) = match store.get(&scale_name) {
                    Some(WeightEntry::Dense(dense)) => match dense.dtype() {
                        DType::F32 => (dense.clone(), ScaleEncoding::F32),
                        DType::BF16 => (dense.clone(), ScaleEncoding::Bf16),
                        DType::E8M0 => (dense.clone(), ScaleEncoding::E8m0),
                        other => {
                            return Err(layout(
                                scale_name,
                                format!("stored {other:?}, expected F32, BF16 or E8M0"),
                            ));
                        }
                    },
                    _ => return Err(layout(scale_name, "missing".to_string())),
                };
                let (format, expected) = if block {
                    (
                        WeightFormat::E4m3Block128 { scale: encoding },
                        vec![out.div_ceil(128), k.div_ceil(128)],
                    )
                } else {
                    (
                        WeightFormat::E4m3PerChannel { scale: encoding },
                        vec![out, 1],
                    )
                };
                if scale.shape() != expected.as_slice() {
                    return Err(layout(
                        scale_name,
                        format!("shape {:?}, expected {expected:?}", scale.shape()),
                    ));
                }
                (
                    format,
                    [out, k],
                    vec![
                        (
                            SourceRole::Planar(OperandRole::Codes),
                            weight.bytes().as_arc(),
                        ),
                        (
                            SourceRole::Planar(OperandRole::Scale),
                            scale.bytes().as_arc(),
                        ),
                    ],
                    vec![".weight", suffix],
                )
            }
        };
        let payload = PackedWeight::try_new(format, shape)
            .and_then(|weight| PackedPayload::try_new(weight, sources))
            .map_err(|source| LoadError::QuantizedLinearPayload {
                linear: format!("{prefix}.weight"),
                source,
            })?;
        consumed.extend(parts.iter().map(|suffix| format!("{prefix}{suffix}")));
        packed.push((format!("{prefix}.weight"), payload));
    }
    let mut builder = WeightStore::builder();
    for (key, entry) in store.iter() {
        if !consumed.contains(key.as_str()) {
            builder.insert(key.clone(), entry.clone())?;
        }
    }
    for (key, payload) in packed {
        builder.insert(key, WeightEntry::Packed(Arc::new(payload)))?;
    }
    Ok(builder.build())
}

/// Parse a SafeTensors archive from bytes in memory (same format as a file; used by tests) into a
/// [`WeightStore`]. See [`load_weight_store`].
///
/// Card 671: no production caller anywhere in the workspace loads a SafeTensors archive straight from an
/// in-memory byte slice - every non-test reference is poot-test-util's forwarder (reached only as a
/// dev-dependency); this crate's own tests use it directly too. Gated so a default build never has it,
/// instead of leaving it unconditionally `pub` with poot-test-util as the only non-test "caller" keeping it
/// off the dead-pub scanner.
#[cfg(any(test, feature = "test-support"))]
pub fn load_weight_store_bytes(bytes: &[u8]) -> Result<WeightStore, LoadError> {
    let mut builder = WeightStore::builder();
    insert_bytes(bytes, &mut builder)?;
    Ok(builder.build())
}

/// Load one safetensors file (not a model directory: no sharding, no fixed `model.safetensors`
/// name) into a [`WeightStore`]. Used by loaders whose single archive has its own name (the LoRA
/// adapter loader's `adapter_model.safetensors`, e.g.). See [`load_weight_store`].
pub fn load_weight_store_file(path: impl AsRef<Path>) -> Result<WeightStore, LoadError> {
    let mut builder = WeightStore::builder();
    let mut seen_names = HashSet::new();
    insert_file(path.as_ref(), &mut builder, &mut seen_names)?;
    Ok(builder.build())
}

/// The dense entry named `name`, or a typed error for a missing/packed entry. Exposed beyond this
/// module (not `pub(crate)`) so `poot_load::lora` and `poot_eval`'s dense-materialization helpers
/// (card 543: `poot-eval` owns `Tensor`, so the WeightStore->Tensor step lives there, not here) can
/// reuse the same lookup instead of duplicating it.
pub fn dense_bytes<'a>(store: &'a WeightStore, name: &str) -> Result<&'a DenseWeight, LoadError> {
    match store.get(name) {
        Some(WeightEntry::Dense(dense)) => Ok(dense),
        Some(WeightEntry::Packed(_)) => Err(LoadError::SafeTensors(format!(
            "{name}: expected a dense stored tensor, found a packed entry"
        ))),
        None => Err(LoadError::SafeTensors(format!("missing tensor {name}"))),
    }
}

/// Decode a dense entry's stored bytes to row-major f32 (the CPU-oracle payload). See
/// [`dense_bytes`]'s doc for why this is `pub`.
pub fn decode_dense(dense: &DenseWeight) -> Result<(Vec<usize>, Vec<f32>), LoadError> {
    let dtype_str = match dense.dtype() {
        DType::F32 => "F32",
        DType::BF16 => "BF16",
        DType::F16 => "F16",
        DType::E4M3FN => "F8_E4M3",
        DType::E8M0 => "F8_E8M0",
        other => {
            return Err(LoadError::UndecodableCheckpointDtype {
                name: format!("{other:?}"),
            });
        }
    };
    let data = decode(dtype_str, dense.bytes().as_slice())?;
    Ok((dense.shape().to_vec(), data))
}

//! Safetensors checkpoint loading: map a directory's weight files, validate each header against its file,
//! and build the [`WeightStore`] (dense tensors as stored, quantized linears packed).

use poot_tensor::DType;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use std::num::NonZeroUsize;

use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};

use super::{LoadError, QuantKind, QuantScheme, checked_header_end, miss, parse_dtype};
use crate::plan::{Container, JobKind, LoadPlan, TensorJob, build_store, default_workers};

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

/// Plan every non-metadata header entry of `files[file]` as a dense job keyed by its checkpoint tensor
/// name: the entry's stored bytes exactly as the checkpoint stored them (ADR-0103 decision 1: no
/// widening, no decode, no copy). Every dtype is kept, `I64` index tables included (DeepSeek-V4's
/// `tid2eid` is one, read through `WeightView::IndexNarrow`); a tensor no family claims is just an
/// unused entry. `seen_names` is the cross-file duplicate-name tracking: every non-metadata header key
/// seen so far, the same real-world duplicate-tensor bug the store-keyed check also catches.
fn plan_header_entries(
    plan: &mut LoadPlan,
    file: usize,
    header_bytes: &[u8],
    data_start: usize,
    data_len: usize,
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
        plan.push(TensorJob {
            key: WeightKey::from(name.as_str()),
            file,
            range: data_start + start..data_start + end,
            kind: JobKind::Dense { dtype, shape },
        });
    }
    Ok(())
}

/// Plan one safetensors file: the header is parsed from the mapping and every tensor is a range of it.
fn plan_file(
    plan: &mut LoadPlan,
    file: usize,
    stored: &StoredBytes,
    seen_names: &mut HashSet<String>,
) -> Result<(), LoadError> {
    let bytes = stored.as_slice();
    if bytes.len() < 8 {
        return Err(LoadError::SafeTensors("file too small".into()));
    }
    let header_end = checked_header_end(
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        bytes.len(),
    )?;
    plan_header_entries(
        plan,
        file,
        &bytes[8..header_end],
        header_end,
        bytes.len() - header_end,
        seen_names,
    )
}

/// The plan for `files` (mapped safetensors files, [`map_weight_files`]) in the order given.
fn safetensors_plan(files: &[(String, StoredBytes)]) -> Result<LoadPlan, LoadError> {
    let mut plan = LoadPlan::new(
        Container::Safetensors,
        files.iter().map(|(_, file)| file.clone()).collect(),
    );
    let mut seen_names = HashSet::new();
    for (index, (_, file)) in files.iter().enumerate() {
        plan_file(&mut plan, index, file, &mut seen_names)?;
    }
    Ok(plan)
}

/// Map one safetensors file for [`load_mapped_weight_store`].
fn map_file(path: &Path) -> Result<(String, StoredBytes), LoadError> {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok((name, StoredBytes::map(File::open(path)?)?))
}

/// Whether `name` is exactly one normal path component: no separator, no `.`/`..`, not absolute.
fn is_plain_file_name(name: &str) -> bool {
    let mut components = Path::new(name).components();
    matches!(
        (components.next(), components.next()),
        (Some(std::path::Component::Normal(_)), None)
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
    // The index is untrusted: a shard is a plain file name in the model directory, never a path.
    if let Some(unsafe_name) = shards.iter().find(|name| !is_plain_file_name(name)) {
        return Err(LoadError::UnsafeShardName {
            name: unsafe_name.clone(),
        });
    }
    shards.sort();
    shards.dedup();
    Ok(shards)
}

/// Map the weight files of a model directory (single `model.safetensors` or the sharded
/// `model.safetensors.index.json` layout), each exactly once, as `(file name, mapping)` in load order.
/// A caller that pins the files checks the pins over these mappings and then builds the store from the
/// same mappings ([`load_mapped_weight_store`]), so digest, header and tensors come from one inode
/// (ADR-0103 amendment). Nothing is read from a mapping here.
pub fn map_weight_files(dir: impl AsRef<Path>) -> Result<Vec<(String, StoredBytes)>, LoadError> {
    let dir = dir.as_ref();
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![map_file(&single)?]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Err(LoadError::SafeTensors(
            "no model.safetensors or model.safetensors.index.json in model dir".into(),
        ));
    }
    let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&index_path)?)?;
    shard_files(&index)?
        .into_iter()
        .map(|shard| map_file(&dir.join(shard)))
        .collect()
}

/// Build a [`WeightStore`] from mapped safetensors files ([`map_weight_files`]): every tensor's bytes a
/// view of its file exactly as the checkpoint stored them, keyed by its checkpoint name (ADR-0103
/// decisions 1 and 3, card 540b). No f32 widening happens here; a consumer that needs f32 (the CPU
/// oracle, or a GPU backend with no native upload lane for the stored dtype) decodes on demand from the
/// returned entries. Once the last file is read, every file must still be the one that was mapped
/// (ADR-0103 amendment, boundary 1) or the load is [`LoadError::SourceChanged`]. The tensors are
/// validated on `workers` threads ([`build_store`]).
pub fn load_mapped_weight_store(
    files: &[(String, StoredBytes)],
    workers: NonZeroUsize,
) -> Result<WeightStore, LoadError> {
    build_store(safetensors_plan(files)?, workers)
}

/// [`map_weight_files`] then [`load_mapped_weight_store`] on [`default_workers`], for a test with no
/// files to pin.
#[cfg(any(test, feature = "test-support"))]
pub fn load_weight_store(dir: impl AsRef<Path>) -> Result<WeightStore, LoadError> {
    load_mapped_weight_store(&map_weight_files(dir)?, default_workers())
}

/// Whether `prefix` (a `.weight` tensor's name without that suffix) names one numbered shard of a
/// Qwen4Exp n-gram PLE table: `<table>.shard_<ordinal>` where the checkpoint stores one
/// `<table>.weight_scale` for the whole table, not one per shard. The shards are E4M3 lookup rows
/// the model dequantizes with that shared scale, never packed linears; without the shared scale a
/// shard is an unscaled E4M3 tensor like any other and stays refused.
fn is_ple_table_shard(store: &WeightStore, prefix: &str) -> bool {
    prefix
        .rsplit_once(".shard_")
        .is_some_and(|(table, ordinal)| {
            !ordinal.is_empty()
                && ordinal.bytes().all(|b| b.is_ascii_digit())
                && table.ends_with(".ngram_embedding")
                && store.contains(&format!("{table}.weight_scale"))
        })
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
    // A header's extent times the 8 codes a packed i32 word holds: refused, not wrapped, when it
    // overflows `usize` (the header is untrusted).
    let pack_extent = |tensor: String, words: usize, what: &str| {
        words.checked_mul(8).ok_or_else(|| {
            layout(
                tensor,
                format!("{what} extent {words} times 8 overflows usize"),
            )
        })
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
        (dense.dtype() == DType::E4M3FN
            && !prefixes.contains(prefix)
            && !is_ple_table_shard(store, prefix))
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
            Vec<(SourceRole, StoredBytes)>,
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
                let k = pack_extent(format!("{prefix}.qweight"), k8, "K/8")?;
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
                let contiguous = g_idx
                    .bytes()
                    .as_slice()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .enumerate()
                    .all(|(k, word)| i32::from_le_bytes(*word) as usize == k / group_size);
                let mut sources = vec![
                    (
                        SourceRole::Planar(OperandRole::Codes),
                        qweight.bytes().clone(),
                    ),
                    (
                        SourceRole::Planar(OperandRole::Zero),
                        qzeros.bytes().clone(),
                    ),
                    (
                        SourceRole::Planar(OperandRole::Scale),
                        scales.bytes().clone(),
                    ),
                ];
                let groups = if contiguous {
                    GroupMap::Contiguous { size: group_size }
                } else {
                    sources.push((
                        SourceRole::Planar(OperandRole::GroupIndex),
                        g_idx.bytes().clone(),
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
                    [pack_extent(format!("{prefix}.qweight"), out8, "out/8")?, k],
                    vec![
                        (
                            SourceRole::Planar(OperandRole::Codes),
                            qweight.bytes().clone(),
                        ),
                        (
                            SourceRole::Planar(OperandRole::Zero),
                            qzeros.bytes().clone(),
                        ),
                        (
                            SourceRole::Planar(OperandRole::Scale),
                            scales.bytes().clone(),
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
                            weight.bytes().clone(),
                        ),
                        (
                            SourceRole::Planar(OperandRole::Scale),
                            scale.bytes().clone(),
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

/// Load one safetensors file (not a model directory: no sharding, no fixed `model.safetensors`
/// name) into a [`WeightStore`]. Used by loaders whose single archive has its own name (the LoRA
/// adapter loader's `adapter_model.safetensors`, e.g.). See [`load_weight_store`].
pub fn load_weight_store_file(path: impl AsRef<Path>) -> Result<WeightStore, LoadError> {
    load_mapped_weight_store(&[map_file(path.as_ref())?], default_workers())
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

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    /// Parse a SafeTensors archive from bytes in memory (same format as a file) into a [`WeightStore`]. See
    /// [`load_weight_store`]. Only this crate's tests build an archive in memory.
    pub(crate) fn load_weight_store_bytes(bytes: &[u8]) -> Result<WeightStore, LoadError> {
        let files = [(String::new(), StoredBytes::from(bytes.to_vec()))];
        build_store(safetensors_plan(&files)?, default_workers())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model directory whose `model.safetensors.index.json` maps one tensor to `shard`.
    fn map_dir_with_shard_named(label: &str, shard: &str) -> Result<usize, LoadError> {
        let dir = poot_test_util::unique_temp_path(label);
        std::fs::create_dir_all(&*dir).unwrap();
        let index = serde_json::json!({"weight_map": {"w": shard}});
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
        map_weight_files(&*dir).map(|files| files.len())
    }

    /// POOT-1196 SC-006: a shard index naming a path (parent component, separator, absolute) is
    /// refused by name before any file is opened; a plain name proceeds (to the missing file).
    #[test]
    fn shard_names_must_be_plain_file_names() {
        for name in [
            "../x.safetensors",
            "sub/x.safetensors",
            "/etc/x.safetensors",
            "..",
        ] {
            let error = map_dir_with_shard_named("shard_name", name).unwrap_err();
            assert!(
                matches!(&error, LoadError::UnsafeShardName { name: n } if n == name),
                "{name}: {error}"
            );
        }
        let error = map_dir_with_shard_named("shard_name_plain", "x.safetensors").unwrap_err();
        assert!(matches!(error, LoadError::Io(_)), "{error}");
    }

    /// An archive of the given `(name, dtype, shape)` entries, all zero bytes sized by the shape.
    fn archive(entries: &[(&str, &str, Vec<usize>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data_len = 0usize;
        for (name, dtype, shape) in entries {
            let width = if *dtype == "F16" { 2 } else { 4 };
            let bytes = shape.iter().product::<usize>() * width;
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [data_len, data_len + bytes]}),
            );
            data_len += bytes;
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend(header);
        blob.resize(blob.len() + data_len, 0);
        blob
    }

    fn pack(
        kind: QuantKind,
        entries: &[(&str, &str, Vec<usize>)],
    ) -> Result<WeightStore, LoadError> {
        let store = test_only::load_weight_store_bytes(&archive(entries))?;
        pack_quantized_linears(
            &store,
            QuantScheme {
                kind,
                group_size: 8,
            },
        )
    }

    /// POOT-1196 SC-005: header extents of a quantized linear that overflow on multiply, or are
    /// zero, are a typed refusal naming the tensor; none panics or wraps.
    #[test]
    fn quantized_linear_extents_are_checked() {
        let huge = usize::MAX / 8 + 1;
        // GPTQ qweight [K/8, out] with K/8 * 8 overflowing; zero elements keep the header valid.
        let gptq = |k8: usize| {
            pack(
                QuantKind::Gptq,
                &[
                    ("l.qweight", "I32", vec![k8, 0]),
                    ("l.qzeros", "I32", vec![1, 1]),
                    ("l.scales", "F16", vec![1, 1]),
                    ("l.g_idx", "I32", vec![0]),
                ],
            )
        };
        // AWQ qweight [K, out/8] with out/8 * 8 overflowing.
        let awq = |out8: usize| {
            pack(
                QuantKind::Awq,
                &[
                    ("l.qweight", "I32", vec![0, out8]),
                    ("l.qzeros", "I32", vec![1, 1]),
                    ("l.scales", "F16", vec![1, 1]),
                ],
            )
        };
        for (what, result) in [("gptq", gptq(huge)), ("awq", awq(huge))] {
            match result {
                Err(LoadError::QuantizedLinearLayout { tensor, reason }) => {
                    assert_eq!(tensor, "l.qweight", "{what}");
                    assert!(reason.contains("overflows"), "{what}: {reason}");
                }
                other => panic!("{what}: expected a layout refusal, got {other:?}"),
            }
        }
        // A zero extent is the packed weight's own typed refusal, naming the linear's `.weight` key.
        for (what, result) in [("gptq", gptq(0)), ("awq", awq(0))] {
            match result {
                Err(LoadError::QuantizedLinearPayload { linear, source }) => {
                    assert_eq!(linear, "l.weight", "{what}");
                    assert_eq!(
                        source,
                        poot_quant::PackedWeightError::ZeroExtent {
                            axis: poot_quant::LogicalAxis::Out
                        },
                        "{what}"
                    );
                }
                other => panic!("{what}: expected a zero-extent payload refusal, got {other:?}"),
            }
        }
    }
}

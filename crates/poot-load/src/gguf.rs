//! GGUF container reader (spec 022): the llama.cpp/unsloth binary format. Parses the header, the typed
//! metadata key-value pairs, and the tensor info table, and dequantizes the ggml tensor types F32, F16, BF16,
//! the legacy blocks (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0), the K-quant super-blocks (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K), and the
//! IQ4_NL / IQ4_XS codebook quants to f32.
//!
//! Layout (little-endian): magic `GGUF`, `version: u32`, `tensor_count: u64`, `metadata_kv_count: u64`,
//! then the KV pairs, then the tensor infos, then padding to `general.alignment` (default 32), then the
//! tensor data. A GGUF string is a `u64` length followed by that many UTF-8 bytes.

mod format;
mod quantization;

pub use format::*;
pub use quantization::*;

use poot_tensor::DType;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use poot_quant::format::WeightFormat;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_quant::{PackedPayload, PackedWeight, SourceRole};

use crate::LoadError;
use format::gerr;

/// Which GGUF tensor names a family's tracer reads, and the [`WeightKey`] to store each one under
/// in the [`WeightStore`] [`read_gguf`] builds. The reader itself names no family (ADR-0103
/// decision 3): this is the one piece of family-specific data it takes.
pub trait TensorNameMap {
    /// The key to store `gguf_name` under, or `None` to skip a tensor this family's tracer never
    /// reads.
    fn weight_key(&self, gguf_name: &str) -> Option<WeightKey>;
}

/// A [`TensorNameMap`] that keeps every GGUF tensor name as its own [`WeightKey`], unchanged.
pub struct IdentityNames;

impl TensorNameMap for IdentityNames {
    fn weight_key(&self, gguf_name: &str) -> Option<WeightKey> {
        Some(WeightKey::from(gguf_name))
    }
}

/// The [`DType`] of a dense (non-quantized) [`WeightFormat`]. Only called once
/// [`poot_quant::format::FormatDescriptor::has_scale`] is known `false`, which holds only for the
/// three plain float formats (`format.rs` module docs), so every other variant is unreachable here.
fn dense_stored_dtype(format: WeightFormat) -> DType {
    match format {
        WeightFormat::F32 => DType::F32,
        WeightFormat::F16 => DType::F16,
        WeightFormat::Bf16 => DType::BF16,
        other => unreachable!("{other:?} has no Scale operand but is not a dense float format"),
    }
}

/// The byte source [`read_gguf`] range-reads tensor data from: a real file (production - one open file
/// descriptor, positional reads, no cursor state to serialize) or an in-memory buffer (a fixture or test
/// that already has the whole small GGUF in memory and would otherwise need a throwaway temp file just
/// to call [`read_gguf`]).
pub enum GgufSource<'a> {
    File(&'a File),
    Bytes(&'a [u8]),
}

impl GgufSource<'_> {
    /// `len` bytes at absolute byte offset `offset` from the start of the source, as one owned buffer.
    fn read_span(&self, offset: u64, len: usize) -> Result<Arc<[u8]>, LoadError> {
        match self {
            Self::File(file) => {
                let mut buffer: Arc<[u8]> = Arc::from(vec![0u8; len]);
                file.read_exact_at(
                    Arc::get_mut(&mut buffer)
                        .expect("freshly allocated buffer has exactly one owner"),
                    offset,
                )?;
                Ok(buffer)
            }
            Self::Bytes(bytes) => {
                let start =
                    usize::try_from(offset).map_err(|_| gerr("tensor offset overflows usize"))?;
                let end = start
                    .checked_add(len)
                    .ok_or_else(|| gerr("tensor range overflows usize"))?;
                bytes
                    .get(start..end)
                    .map(Arc::from)
                    .ok_or_else(|| gerr("tensor range past end of in-memory buffer"))
            }
        }
    }
}

impl<'a> From<&'a File> for GgufSource<'a> {
    fn from(file: &'a File) -> Self {
        Self::File(file)
    }
}

impl<'a> From<&'a [u8]> for GgufSource<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        Self::Bytes(bytes)
    }
}

/// `len` bytes starting at `data_start + offset` of `source`, as one owned buffer.
fn read_span(
    source: &GgufSource<'_>,
    data_start: usize,
    offset: u64,
    len: usize,
) -> Result<Arc<[u8]>, LoadError> {
    let start = data_start
        .checked_add(usize::try_from(offset).map_err(|_| gerr("tensor offset overflows usize"))?)
        .ok_or_else(|| gerr("tensor data offset overflows usize"))?;
    let start = u64::try_from(start).map_err(|_| gerr("tensor data offset overflows u64"))?;
    source.read_span(start, len)
}

/// One per-tensor scan of a GGUF file into a [`WeightStore`] (dquant.md 8.1): dense types (F32, F16,
/// BF16) become [`WeightEntry::Dense`]; a quantized rank-2 tensor becomes one
/// [`WeightEntry::Packed`] with a single `SourceRole::Blocks` source; a quantized rank-3 expert
/// tensor (`[E, out, K]`) becomes `E` separate packed owners, one per expert, keyed
/// `"{key}.{expert}"` (ADR-0109: experts stay separate owners, never stacked on the host). Every
/// tensor loads by range read of exactly its own bytes; nothing here reads the whole file, and no
/// byte is transcoded, widened or repacked (ADR-0103 decision 1) - the stored payload is bit-for-bit
/// the file's tensor bytes. A tensor `names` does not map is skipped entirely, including its type
/// check; a quantized tensor of a type [`weight_format_of`] does not know is refused by name
/// ([`LoadError::UnsupportedGgufType`]), never loaded dense (ADR-0103 decision 2).
pub fn read_gguf<'a>(
    index: &GgufIndex,
    source: impl Into<GgufSource<'a>>,
    names: &dyn TensorNameMap,
) -> Result<WeightStore, LoadError> {
    let source = source.into();
    let source = &source;
    let mut builder = WeightStore::builder();
    let data_start = index.data_start();
    for (name, info) in &index.tensors {
        let Some(key) = names.weight_key(name) else {
            continue;
        };
        // ggml `ne` is fastest-varying first; the logical row-major shape is the reverse.
        let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
        let format =
            weight_format_of(info.ggml_type).ok_or_else(|| LoadError::UnsupportedGgufType {
                tensor: name.clone(),
                ggml_type: info.ggml_type,
            })?;
        let descriptor = format.descriptor();
        if !descriptor.has_scale() {
            let dtype = dense_stored_dtype(format);
            let numel: usize = shape.iter().product();
            let byte_len = numel
                .checked_mul(dtype.byte_size())
                .ok_or_else(|| gerr(&format!("{name}: dense byte length overflows usize")))?;
            let bytes = read_span(source, data_start, info.offset, byte_len)?;
            let dense = DenseWeight::try_new(dtype, shape, bytes)?;
            builder.insert(key, WeightEntry::Dense(dense))?;
            continue;
        }
        match shape.len() {
            2 => {
                let weight = PackedWeight::try_new(format, [shape[0], shape[1]])?;
                let len = weight.source_bytes(SourceRole::Blocks);
                let bytes = read_span(source, data_start, info.offset, len)?;
                let payload = PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes)])?;
                builder.insert(key, WeightEntry::Packed(Arc::new(payload)))?;
            }
            3 => {
                // ggml `ne = [K, out, E]`; the reversed logical shape is `[E, out, K]`.
                let (experts, out, k) = (shape[0], shape[1], shape[2]);
                let weight = PackedWeight::try_new(format, [out, k])?;
                let expert_len = weight.source_bytes(SourceRole::Blocks);
                for expert in 0..experts {
                    let expert_offset = expert_len
                        .checked_mul(expert)
                        .and_then(|delta| info.offset.checked_add(delta as u64))
                        .ok_or_else(|| gerr(&format!("{name}: expert offset overflows")))?;
                    let bytes = read_span(source, data_start, expert_offset, expert_len)?;
                    let payload = PackedPayload::try_new(weight, [(SourceRole::Blocks, bytes)])?;
                    builder.insert(
                        format!("{key}.{expert}"),
                        WeightEntry::Packed(Arc::new(payload)),
                    )?;
                }
            }
            rank => {
                return Err(LoadError::UnsupportedGgufTensorRank {
                    tensor: name.clone(),
                    rank,
                });
            }
        }
    }
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    use poot_quant::format::Storage;
    use poot_quant::scalar::{bf16_to_f32, e2m1_to_f32, e8m0_to_f32, f16_to_f32};

    // ggml tensor type ids the fixtures below write.
    const GGML_F32: u32 = 0;
    const GGML_F16: u32 = 1;
    const GGML_Q4_0: u32 = 2;
    const GGML_Q8_0: u32 = 8;
    const GGML_Q2_K: u32 = 10;
    const GGML_Q4_K: u32 = 12;
    const GGML_BF16: u32 = 30;

    /// Test oracle: the first `numel` values of `raw` in `format`'s blocks, decoded through
    /// poot-quant's one scalar decoder.
    fn dequant(format: WeightFormat, raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        let Storage::Blocks(layout) = format.descriptor().storage else {
            unreachable!("every GGUF format is a block format")
        };
        assert!(
            numel.is_multiple_of(layout.values),
            "{format:?} partial block"
        );
        let need = numel / layout.values * layout.bytes;
        bytes_ok(raw, need)?;
        let mut out = vec![0.0f32; numel];
        format
            .descriptor()
            .decode_blocks(&raw[..need], &mut out)
            .map_err(|error| gerr(&error.to_string()))?;
        Ok(out)
    }
    fn dequant_q4_0(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q4_0, raw, numel)
    }
    fn dequant_q4_1(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q4_1, raw, numel)
    }
    fn dequant_q5_0(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q5_0, raw, numel)
    }
    fn dequant_q5_1(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q5_1, raw, numel)
    }
    fn dequant_q8_0(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q8_0, raw, numel)
    }
    fn dequant_q2_k(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q2_K, raw, numel)
    }
    fn dequant_q3_k(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q3_K, raw, numel)
    }
    fn dequant_q4_k(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q4_K, raw, numel)
    }
    fn dequant_q5_k(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q5_K, raw, numel)
    }
    fn dequant_q6_k(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Q6_K, raw, numel)
    }
    fn dequant_iq4_nl(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Iq4_Nl, raw, numel)
    }
    fn dequant_iq4_xs(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Iq4_Xs, raw, numel)
    }
    fn dequant_mxfp4(raw: &[u8], numel: usize) -> Result<Vec<f32>, LoadError> {
        dequant(WeightFormat::Mxfp4, raw, numel)
    }

    /// Test fixture: a whole (small) GGUF in memory, its index plus its bytes, decoding a named
    /// tensor through [`read_gguf`] and the oracle above. The production loader range-reads through
    /// [`read_gguf`] and never holds the file whole.
    struct WholeGguf {
        index: GgufIndex,
        bytes: Vec<u8>,
    }

    impl std::ops::Deref for WholeGguf {
        type Target = GgufIndex;
        fn deref(&self) -> &GgufIndex {
            &self.index
        }
    }

    impl WholeGguf {
        fn load(path: impl AsRef<std::path::Path>) -> Result<Self, LoadError> {
            Self::from_bytes(std::fs::read(path)?)
        }

        fn from_bytes(bytes: Vec<u8>) -> Result<Self, LoadError> {
            Ok(Self {
                index: GgufIndex::from_bytes(&bytes)?,
                bytes,
            })
        }

        /// Tensor `name` decoded to f32, in logical row-major order.
        fn dequant(&self, name: &str) -> Result<poot_tensor::HostTensor, LoadError> {
            let info = self
                .index
                .tensors
                .get(name)
                .ok_or_else(|| gerr(&format!("missing tensor {name}")))?;
            let shape: Vec<usize> = info.dims.iter().rev().map(|&d| d as usize).collect();
            let numel: usize = shape.iter().product();
            let start = self.index.data_start() + info.offset as usize;
            let raw = &self.bytes[start..];
            let format = weight_format_of(info.ggml_type)
                .ok_or_else(|| gerr(&format!("unsupported ggml type {}", info.ggml_type)))?;
            let data: Vec<f32> = match format {
                WeightFormat::F32 => raw[..numel * 4]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                WeightFormat::F16 => raw[..numel * 2]
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                WeightFormat::Bf16 => raw[..numel * 2]
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                format => dequant(format, raw, numel)?,
            };
            Ok(poot_tensor::HostTensor::f32(shape, data))
        }
    }

    fn q8_0_roundtrip_block(vals: &[f32; 32]) -> ([u8; 34], f32) {
        // Encode a Q8_0 block as ggml does: d = max|x|/127, q = round(x/d).
        let amax = vals.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
        let d = amax / 127.0;
        let mut buf = [0u8; 34];
        // Store d as a real f16.
        let h = f32_to_f16(d);
        buf[0..2].copy_from_slice(&h.to_le_bytes());
        for i in 0..32 {
            let q = if d != 0.0 {
                (vals[i] / d).round().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            buf[2 + i] = q as u8;
        }
        (buf, d)
    }

    #[test]
    fn q8_0_dequant_matches_block_encoding() {
        let vals: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
        let (buf, _d) = q8_0_roundtrip_block(&vals);
        let got = dequant_q8_0(&buf, 32).unwrap();
        // Dequant recovers the values within the Q8_0 step (d ~= amax/127).
        for (g, w) in got.iter().zip(vals.iter()) {
            assert!((g - w).abs() <= 0.02, "{g} vs {w}");
        }
    }

    #[test]
    fn encode_q8_0_round_trips_through_dequant() {
        // Fixture encoder: encode 96 elements (3 blocks), dequant, and check within the Q8_0 step.
        let vals: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.05).collect();
        let raw = encode_q8_0(&vals);
        assert_eq!(raw.len(), 3 * 34);
        let got = dequant_q8_0(&raw, 96).unwrap();
        for (g, w) in got.iter().zip(&vals) {
            assert!((g - w).abs() <= 0.02, "{g} vs {w}");
        }
    }

    #[test]
    fn encode_q5_0_round_trips_through_dequant() {
        // Card 179 fixture encoder: encode 96 elements (3 blocks spanning several sign/scale
        // ranges), dequant, and check within the Q5_0 step (per block d = amax/16). The element at
        // amax hits the encoder's clamp (x/d = 16 rounds to code 32, clamped to 31, the same quirk
        // as `encode_q4_0`), so worst-case error is a full step `d`; amax is up to ~2.4, so <= ~0.15.
        let vals: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.05).collect();
        let raw = encode_q5_0(&vals);
        assert_eq!(raw.len(), 3 * 22);
        let got = dequant_q5_0(&raw, 96).unwrap();
        for (g, w) in got.iter().zip(&vals) {
            assert!((g - w).abs() <= 0.16, "{g} vs {w}");
        }
        assert_block_scales(&raw, 22, &vals, 16.0);
    }

    #[test]
    fn encode_q4_0_round_trips_through_dequant() {
        // Same shape as the Q5_0 case: per block d = amax/8, and the element at amax hits the clamp
        // (x/d = 8 rounds to code 16, clamped to 15), so the worst-case error is a full step `d`.
        let vals: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.05).collect();
        let raw = encode_q4_0(&vals);
        assert_eq!(raw.len(), 3 * 18);
        let got = dequant_q4_0(&raw, 96).unwrap();
        for (g, w) in got.iter().zip(&vals) {
            assert!((g - w).abs() <= 0.31, "{g} vs {w}");
        }
        assert_block_scales(&raw, 18, &vals, 8.0);
    }

    /// Every block's leading f16 scale must be `amax / divisor` of that block's source values, computed
    /// here independently of the encoder. The dequant round trip alone cannot see a wrong scale: both
    /// sides read the same bytes, and a coarser scale still lands inside the loose per-element bound.
    fn assert_block_scales(raw: &[u8], block_bytes: usize, vals: &[f32], divisor: f32) {
        for (blk, chunk) in vals.chunks(32).enumerate() {
            let amax = chunk.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let d = f16_to_f32(u16::from_le_bytes([
                raw[blk * block_bytes],
                raw[blk * block_bytes + 1],
            ]));
            let want = amax / divisor;
            assert!(
                (d - want).abs() <= want * 1e-3,
                "block {blk}: scale {d} vs amax/{divisor} = {want}"
            );
        }
    }

    #[test]
    fn parses_real_qwen2_gguf() {
        // Uses the Qwen2.5-0.5B Q8_0 GGUF under POOT_MODELS_DIR; skips if absent.
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
        )) else {
            return;
        };
        let g = WholeGguf::load(&path).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("qwen2"));
        assert_eq!(
            g.get("qwen2.block_count").and_then(|v| v.as_u64()),
            Some(24)
        );
        assert_eq!(
            g.get("qwen2.embedding_length").and_then(|v| v.as_u64()),
            Some(896)
        );
        assert_eq!(g.tensors.len(), 291);
        // The token embedding is Q8_0 [vocab, 896]; dequant the small F32 output norm and check finiteness.
        let norm = g.dequant("output_norm.weight").unwrap();
        assert_eq!(norm.shape(), [896]);
        assert!(norm.as_f32().unwrap().iter().all(|v| v.is_finite()));
        // A Q8_0 weight dequants to the right element count and is finite.
        let tok = g.dequant("token_embd.weight").unwrap();
        assert_eq!(tok.shape(), [151936, 896]);
        assert!(tok.as_f32().unwrap()[..896].iter().all(|v| v.is_finite()));
    }

    const F16_ONE: [u8; 2] = [0x00, 0x3C]; // 1.0 in f16, little-endian

    // Differential probe: the same weights live in the Q8_0 (near-lossless) and Q4_K_M GGUFs.
    // Dequants shared tensors from each and reports cosine similarity and type; a wrong Q4_K/Q6_K
    // dequant shows as low correlation (uniform-value unit tests cannot catch a position bug).
    // Run: cargo test -p poot-load q4km_vs_q8 -- --ignored --nocapture
    #[test]
    #[ignore = "differential debug probe; needs both GGUFs under POOT_MODELS_DIR"]
    fn q4km_vs_q8_tensor_correlation() {
        let Some(q8) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
        )) else {
            return;
        };
        // Quant file to check against the Q8_0 reference.
        let Some(q4) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )) else {
            return;
        };
        let (g8, g4) = (WholeGguf::load(&q8).unwrap(), WholeGguf::load(&q4).unwrap());
        // Type distribution in the q4_k_m file.
        let mut counts: std::collections::BTreeMap<u32, usize> = Default::default();
        for t in g4.tensors.values() {
            *counts.entry(t.ggml_type).or_default() += 1;
        }
        eprintln!("q4_k_m ggml_type counts: {counts:?}");
        let names = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_v.weight",
        ];
        for n in names {
            let (Some(a), Some(b)) = (g4.tensors.get(n), g8.tensors.get(n)) else {
                eprintln!(
                    "{n}: missing in one file (q4={:?})",
                    g4.tensors.get(n).map(|t| t.ggml_type)
                );
                continue;
            };
            let (ty4, ty8) = (a.ggml_type, b.ggml_type);
            let da = g4.dequant(n).unwrap().as_f32().unwrap().to_vec();
            let db = g8.dequant(n).unwrap().as_f32().unwrap().to_vec();
            let cos = cosine(&da, &db);
            eprintln!(
                "{n}: q4_type={ty4} q8_type={ty8} n={} cos={cos:.4} | q4[0..4]={:?} q8[0..4]={:?}",
                da.len(),
                &da[..4.min(da.len())],
                &db[..4.min(db.len())],
            );
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    #[test]
    fn q4_k_block_dequants_to_known_values() {
        // One super-block: d=1, dmin=0, every sub-block scale=1 min=0, every q4 nibble=1 -> all weights 1.
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&F16_ONE); // d = 1
        // dmin stays 0. scales (blk[4..16]) encode (scale=1, min=0) for all 8 sub-blocks (see
        // get_scale_min_k4): scales[0..4]=1, scales[4..8]=0, scales[8..12]=1.
        for i in 0..4 {
            blk[4 + i] = 1; // scales[0..4]
            blk[12 + i] = 1; // scales[8..12]
        }
        for b in blk.iter_mut().skip(16) {
            *b = 0x11; // qs: both nibbles = 1
        }
        let out = dequant_q4_k(&blk, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-4, "q4_k[{i}] = {v}, want 1.0");
        }
    }

    #[test]
    fn q5_k_block_dequants_to_known_values() {
        // One super-block: d=1, dmin=0, scale=1, min=0, every low nibble 1, every qh bit set ->
        // weight = d*scale*(1 + 16) - 0 = 17.
        let mut blk = vec![0u8; 176];
        blk[0..2].copy_from_slice(&F16_ONE); // d = 1; dmin stays 0
        for i in 0..4 {
            blk[4 + i] = 1; // scales[0..4] = 1
            blk[12 + i] = 1; // scales[8..12] = 1  (scale=1, min=0 for all 8 sub-blocks)
        }
        for b in blk[16..48].iter_mut() {
            *b = 0xFF; // qh: every 5th bit set -> +16
        }
        for b in blk[48..176].iter_mut() {
            *b = 0x11; // qs: both nibbles = 1
        }
        let out = dequant_q5_k(&blk, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 17.0).abs() < 1e-3, "q5_k[{i}] = {v}, want 17.0");
        }
    }

    #[test]
    fn q6_k_block_dequants_to_known_values() {
        // One super-block: d=1, every scale=-2 (i8 0xFE), ql=qh=0 -> q = -32 -> weight =
        // 1 * (-2) * (-32) = 64. The negative scale guards against reading Q6_K's int8_t scales as
        // u8 (254*-32).
        let mut blk = vec![0u8; 210];
        for i in 0..16 {
            blk[192 + i] = 0xFE; // scales[16] = -2 (i8)
        }
        blk[208..210].copy_from_slice(&F16_ONE); // d = 1
        let out = dequant_q6_k(&blk, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 64.0).abs() < 1e-3, "q6_k[{i}] = {v}, want 64.0");
        }
    }

    #[test]
    fn mxfp4_block_dequants_to_the_ocp_e2m1_values() {
        // Codes 0..15 laid out so element j (low nibble of byte j) = j and element j+16 = j (again),
        // i.e. bytes 0..15 = 0x00, 0x11, ... 0xFF gives element j = code j and element j+16 = code j.
        let mut raw = vec![127u8];
        for j in 0..16u8 {
            raw.push(j | (j << 4));
        }
        let got = dequant_mxfp4(&raw, 32).unwrap();
        let e2m1 = [
            0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        for j in 0..16 {
            assert_eq!(got[j], e2m1[j], "low-nibble element {j}");
            assert_eq!(got[j + 16], e2m1[j], "high-nibble element {}", j + 16);
        }
        // The shared scale is a power of two: e = 129 -> X = 2^2 = 4, every value scaled by 4.
        raw[0] = 129;
        let got4 = dequant_mxfp4(&raw, 32).unwrap();
        for j in 0..32 {
            assert_eq!(got4[j], got[j] * 4.0, "e=129 must scale element {j} by 2^2");
        }
        // An all-zero block (e = 0, the encoder's amax == 0 case) must dequantize to exact zeros.
        let zero = vec![0u8; 17];
        assert_eq!(dequant_mxfp4(&zero, 32).unwrap(), vec![0.0f32; 32]);
    }

    /// Cross-check poot's MXFP4 dequant against the OCP MX v1.0 spec in its own terms (shared
    /// scale `X = 2^(e-127)` times an E2M1 value in `+-{0, 0.5, 1, 1.5, 2, 3, 4, 6}`) rather than
    /// ggml's doubled-table/halved-scale form. Both are implemented independently and must agree
    /// exactly over every code and a wide exponent range; a ggml-only transcription could be
    /// self-consistent and 2x wrong.
    ///
    /// The same comparison was run once on 11520 real weights fetched by HTTP range request from
    /// `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf` (`blk.0.ffn_gate_exps.weight`, type 39):
    /// bit-identical, block exponents 120-122 (scales 2^-8..2^-6), values in +-0.125. The receipt
    /// is in `docs/updates/`; it is not committed because that would mean checking in model weights.
    #[test]
    fn mxfp4_dequant_agrees_with_the_ocp_spec_stated_independently() {
        /// The OCP MX v1.0 formulation from the spec, not ggml: magnitude table for codes 0..7,
        /// sign in bit 3, times `X = 2^(e-127)`.
        fn ocp_value(code: u8, e: u8) -> f32 {
            const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let v = MAG[(code & 0x7) as usize];
            let signed = if code & 0x8 != 0 { -v } else { v };
            // 2^(e-127) by exponent arithmetic (all `e` below are in the normal range).
            signed * f32::from_bits(((e as u32) << 23).max(1))
        }
        // Sweep every code against a wide, all-normal exponent range (X = 2^-27 .. 2^28).
        for e in 100u8..=155 {
            let mut raw = vec![e];
            for j in 0..16u8 {
                raw.push(j | (j << 4)); // element j = code j, element j+16 = code j
            }
            let got = dequant_mxfp4(&raw, 32).unwrap();
            for j in 0..16usize {
                let want = ocp_value(j as u8, e);
                assert_eq!(got[j], want, "e={e} code={j} (low nibble)");
                assert_eq!(got[j + 16], want, "e={e} code={j} (high nibble)");
            }
        }
    }

    /// Pins MXFP4's nibble interleave (byte `j` = element `j` low / `j+16` high, not natural
    /// order): a block whose byte 0 is `0x51` must put code 1 at element 0 and code 5 at element
    /// 16, and nothing else may be non-zero.
    #[test]
    fn mxfp4_nibble_interleave_is_low_half_high_half() {
        let mut raw = vec![0u8; 17];
        raw[0] = 127; // scale 2^0
        raw[1] = 0x51; // low nibble 1 -> element 0; high nibble 5 -> element 16
        let got = dequant_mxfp4(&raw, 32).unwrap();
        assert_eq!(got[0], 0.5, "code 1 (E2M1 0.5) belongs at element 0");
        assert_eq!(got[16], 3.0, "code 5 (E2M1 3.0) belongs at element 16");
        for (j, &v) in got.iter().enumerate() {
            if j != 0 && j != 16 {
                assert_eq!(v, 0.0, "element {j} must stay zero");
            }
        }
    }

    /// `pack_mxfp4` must reproduce `dequant_mxfp4` bit-exactly once its packed nibbles/scales are
    /// unpacked as the kernel/CPU oracle does (`w = e2m1(code) * e8m0(scales, k/32)`,
    /// 8 codes per i32 word in natural element order), like every other packer here. Multi-row,
    /// multi-block, varying exponents and codes.
    #[test]
    fn encode_mxfp4_round_trips_through_dequant() {
        // On-grid signal: exact E2M1 multiples of one power of two, so quantization is lossless.
        let on_grid: Vec<f32> = (0..64)
            .map(|i| {
                let e2m1 = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                let v = e2m1[i % 8] * 2.0;
                if i % 16 >= 8 { -v } else { v }
            })
            .collect();
        let bytes = encode_mxfp4(&on_grid);
        assert_eq!(bytes.len(), 2 * 17);
        let back = dequant_mxfp4(&bytes, 64).unwrap();
        for (i, (&a, &b)) in on_grid.iter().zip(back.iter()).enumerate() {
            assert_eq!(a, b, "on-grid value {i} must round-trip exactly");
        }
        // Off-grid signal: every decoded value must be exactly representable (kvalue * scale)
        // and within one grid step of the source.
        let off_grid: Vec<f32> = (0..32).map(|i| (i as f32) * 0.037 - 0.4).collect();
        let bytes = encode_mxfp4(&off_grid);
        let back = dequant_mxfp4(&bytes, 32).unwrap();
        let d = e8m0_to_f32(bytes[0]);
        let grid: Vec<f32> = (0..16u8).map(|code| e2m1_to_f32(code) * d).collect();
        for (i, &b) in back.iter().enumerate() {
            assert!(
                grid.contains(&b),
                "decoded value {i} ({b}) must be on the MXFP4 grid for scale {d}"
            );
            // ...and the nearest grid point to the source (as ggml's `best_index_mxfp4`). A plain
            // absolute tolerance would be wrong: the grid is non-uniform (kvalue gaps
            // 1,1,1,2,2,4), so the worst-case step is 4*d.
            let best = grid
                .iter()
                .copied()
                .min_by(|x, y| (x - off_grid[i]).abs().total_cmp(&(y - off_grid[i]).abs()))
                .unwrap();
            assert_eq!(
                b, best,
                "decoded value {i} must be the nearest MXFP4 grid point to {}",
                off_grid[i]
            );
        }
    }

    #[test]
    fn q2_k_block_dequants_to_known_values() {
        // Q2_K layout: scales[16], qs[64], d f16, dmin f16 (84 bytes). x = d*(sc&0xF)*q - dmin*(sc>>4).
        // case A: d=1, dmin=0, every scale byte 0x0F (scale=15, min=0), qs=0xFF (every 2-bit q=3)
        //   -> x = 1*15*3 - 0 = 45. Exercises the scale low-nibble + the 2-bit lanes at all 4 shifts.
        let mut a = vec![0u8; 84];
        for b in a[0..16].iter_mut() {
            *b = 0x0F;
        }
        for b in a[16..80].iter_mut() {
            *b = 0xFF;
        }
        a[80..82].copy_from_slice(&F16_ONE); // d = 1; dmin stays 0
        let out = dequant_q2_k(&a, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 45.0).abs() < 1e-3, "q2_k.A[{i}] = {v}, want 45.0");
        }
        // case B: d=1, dmin=1, every scale byte 0xF0 (scale=0, min=15), qs=0 -> x = 0 - 1*15 = -15.
        //   Exercises the min (scale high-nibble) path independent of the quants.
        let mut b = vec![0u8; 84];
        for x in b[0..16].iter_mut() {
            *x = 0xF0;
        }
        b[80..82].copy_from_slice(&F16_ONE); // d = 1
        b[82..84].copy_from_slice(&F16_ONE); // dmin = 1
        let out = dequant_q2_k(&b, 256).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v + 15.0).abs() < 1e-3, "q2_k.B[{i}] = {v}, want -15.0");
        }
    }

    #[test]
    fn q3_k_block_dequants_to_known_values() {
        // Q3_K layout: hmask[32], qs[64], scales[12], d f16 (110 bytes). The 12 scale bytes all 0xFF unpack
        // (via the kmask shuffle) to all-63 6-bit scales, so scale-32 = 31. x = d*(scale-32)*(q_lo2 - hbit),
        // hbit = (hmask&m ? 0 : 4).
        // case A: d=1, qs=0, hmask=0 -> hbit=4 -> x = 1*31*(0-4) = -124. Exercises the scale unpack + the
        //   high-bit "subtract 4" branch.
        let mut a = vec![0u8; 110];
        for b in a[96..108].iter_mut() {
            *b = 0xFF; // scales -> all 63
        }
        a[108..110].copy_from_slice(&F16_ONE); // d = 1
        let out = dequant_q3_k(&a, 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert!((v + 124.0).abs() < 1e-3, "q3_k.A[{i}] = {v}, want -124.0");
        }
        // case B: d=1, qs=0xFF (q_lo2=3 at every shift), hmask=0xFF (high bit set -> no subtract)
        //   -> x = 1*31*(3-0) = 93. Exercises the 2-bit lanes at all shifts + the high-bit "no offset" branch.
        let mut b = vec![0u8; 110];
        for x in b[0..32].iter_mut() {
            *x = 0xFF; // hmask all set
        }
        for x in b[32..96].iter_mut() {
            *x = 0xFF; // qs all set
        }
        for x in b[96..108].iter_mut() {
            *x = 0xFF; // scales -> all 63
        }
        b[108..110].copy_from_slice(&F16_ONE); // d = 1
        let out = dequant_q3_k(&b, 256).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 93.0).abs() < 1e-3, "q3_k.B[{i}] = {v}, want 93.0");
        }
    }

    #[test]
    fn bf16_widens_to_f32() {
        // bf16 is the high 16 bits of an f32. Known patterns: sign|exp(8)|mant(7).
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3F80), 1.0); // exp 127, mant 0
        assert_eq!(bf16_to_f32(0xBF80), -1.0); // sign bit set
        assert_eq!(bf16_to_f32(0x4000), 2.0);
        assert_eq!(bf16_to_f32(0x4049), 3.140625); // ~pi truncated to bf16
        assert!(bf16_to_f32(0x7FC0).is_nan()); // a quiet NaN carries through
        assert_eq!(bf16_to_f32(0x7F80), f32::INFINITY);
    }

    #[test]
    fn q4_0_block_dequants_to_known_values() {
        // one block: d=1, every nibble=0x0F -> out = (15 - 8) = 7.
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&F16_ONE); // d = 1
        for b in blk.iter_mut().skip(2) {
            *b = 0xFF; // both nibbles = 15
        }
        let out = dequant_q4_0(&blk, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 7.0).abs() < 1e-4, "q4_0[{i}] = {v}, want 7.0");
        }
    }

    #[test]
    fn iq4_nl_block_dequants_to_known_codebook_values() {
        // IQ4_NL: like Q4_0 (32-elem block, f16 d + 16 qs bytes) but each nibble indexes the kvalues_iq4nl
        // codebook. d=1.
        // case A: every nibble = 0 -> kvalues[0] = -127 for all 32.
        let mut a = vec![0u8; 18];
        a[0..2].copy_from_slice(&F16_ONE); // d = 1; qs all zero
        let out = dequant_iq4_nl(&a, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            assert!((v + 127.0).abs() < 1e-4, "iq4_nl.A[{i}] = {v}, want -127.0");
        }
        // case B: every qs byte 0x8F -> low nibble 0xF -> kvalues[15]=113 (first half), high nibble 0x8 ->
        // kvalues[8]=1 (second half). Verifies the low/high split + a non-trivial codebook lookup.
        let mut b = vec![0u8; 18];
        b[0..2].copy_from_slice(&F16_ONE);
        for x in b.iter_mut().skip(2) {
            *x = 0x8F;
        }
        let out = dequant_iq4_nl(&b, 32).unwrap();
        for (i, &v) in out.iter().enumerate().take(16) {
            assert!(
                (v - 113.0).abs() < 1e-4,
                "iq4_nl.B low[{i}] = {v}, want 113.0"
            );
        }
        for (i, &v) in out.iter().enumerate().skip(16) {
            assert!((v - 1.0).abs() < 1e-4, "iq4_nl.B high[{i}] = {v}, want 1.0");
        }
    }

    #[test]
    fn iq4_xs_superblock_dequants_to_known_values() {
        // IQ4_XS layout: d f16, scales_h u16, scales_l[4], qs[128] (136 bytes / 256 elems). Each of 8
        // sub-blocks (32 elems) has a 6-bit scale ls = low4(scales_l) | high2(scales_h)<<4, then dl=d*(ls-32);
        // x = dl * kvalues[nibble]. Build a block with d=1, all qs=0 (nibble 0 -> kvalues[0] = -127).
        // scales_h = 0xAAAA -> every sub-block's high 2 bits = 0b10 = 2.
        let mk = |scales_l: [u8; 4]| -> Vec<u8> {
            let mut b = vec![0u8; 136];
            b[0..2].copy_from_slice(&F16_ONE); // d = 1
            b[2..4].copy_from_slice(&0xAAAAu16.to_le_bytes()); // scales_h
            b[4..8].copy_from_slice(&scales_l);
            // qs (b[8..136]) stay 0 -> nibble 0
            b
        };
        // case A: all scales_l nibbles = 1 -> every sub-block ls = 1 | 32 = 33, dl = 1 -> all 256 = -127.
        let out = dequant_iq4_xs(&mk([0x11; 4]), 256).unwrap();
        assert_eq!(out.len(), 256);
        for (i, &v) in out.iter().enumerate() {
            assert!((v + 127.0).abs() < 1e-4, "iq4_xs.A[{i}] = {v}, want -127.0");
        }
        // case B: scales_l[0] = 0x21 -> sub-block 0 low nibble 1 (ls 33, dl 1), sub-block 1 low nibble 2
        // (ls 34, dl 2); the rest dl 1. So elems [0,32) = -127, [32,64) = -254, [64,256) = -127. This pins the
        // PER-SUB-BLOCK scale indexing (the low/high bit unpacking + which sub-block reads which bits).
        let out = dequant_iq4_xs(&mk([0x21, 0x11, 0x11, 0x11]), 256).unwrap();
        for (i, &v) in out.iter().enumerate() {
            let want = if (32..64).contains(&i) {
                -254.0
            } else {
                -127.0
            };
            assert!((v - want).abs() < 1e-4, "iq4_xs.B[{i}] = {v}, want {want}");
        }
    }

    #[test]
    fn q5_0_block_dequants_to_known_values() {
        // one block: d=1, qh all-ones (5th bit set for every element), nibbles=0x0F ->
        // q = (0x0F | 0x10) - 16 = 31 - 16 = 15.
        let mut blk = vec![0u8; 22];
        blk[0..2].copy_from_slice(&F16_ONE); // d = 1
        for i in 0..4 {
            blk[2 + i] = 0xFF; // qh: all 32 high-bits set
        }
        for b in blk.iter_mut().skip(6) {
            *b = 0xFF; // both nibbles = 15
        }
        let out = dequant_q5_0(&blk, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 15.0).abs() < 1e-4, "q5_0[{i}] = {v}, want 15.0");
        }
    }

    const F16_FIVE: [u8; 2] = [0x00, 0x45]; // 5.0 in f16, little-endian

    #[test]
    fn q4_1_block_dequants_to_known_values() {
        // d=1, m=5, every nibble=3 -> out = d*3 + m = 8. The explicit min (no -8 offset) is the Q4_1 vs
        // Q4_0 difference.
        let mut blk = vec![0u8; 20];
        blk[0..2].copy_from_slice(&F16_ONE); // d = 1
        blk[2..4].copy_from_slice(&F16_FIVE); // m = 5
        for b in blk.iter_mut().skip(4) {
            *b = 0x33; // both nibbles = 3
        }
        let out = dequant_q4_1(&blk, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 8.0).abs() < 1e-3, "q4_1[{i}] = {v}, want 8.0");
        }
    }

    #[test]
    fn q5_1_block_dequants_to_known_values() {
        // d=1, m=5, qh all-ones (5th bit set), nibbles=3 -> q = (3 | 16) = 19, out = d*19 + m = 24.
        let mut blk = vec![0u8; 24];
        blk[0..2].copy_from_slice(&F16_ONE); // d = 1
        blk[2..4].copy_from_slice(&F16_FIVE); // m = 5
        for i in 0..4 {
            blk[4 + i] = 0xFF; // qh: all 32 high-bits set
        }
        for b in blk.iter_mut().skip(8) {
            *b = 0x33; // both nibbles = 3
        }
        let out = dequant_q5_1(&blk, 32).unwrap();
        assert_eq!(out.len(), 32);
        for (i, &v) in out.iter().enumerate() {
            assert!((v - 24.0).abs() < 1e-3, "q5_1[{i}] = {v}, want 24.0");
        }
    }

    #[test]
    fn gguf_writer_round_trips_through_the_reader() {
        // Build a GGUF in memory and read it back: container parsing (header, KV types, tensor
        // table, alignment/offsets) plus F32/F16/BF16/Q8_0 dequant.
        let f32_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
        let f32_bytes: Vec<u8> = f32_data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let f16_vals: Vec<f32> = (0..8).map(|i| i as f32 - 4.0).collect();
        let f16_bytes: Vec<u8> = f16_vals
            .iter()
            .flat_map(|&x| f32_to_f16(x).to_le_bytes())
            .collect();
        let bf16_words = [0x3f80u16, 0xc020, 0x3e00, 0x42c8];
        let bf16_bytes: Vec<u8> = bf16_words.iter().flat_map(|x| x.to_le_bytes()).collect();
        let bf16_vals: Vec<f32> = bf16_words.iter().map(|&x| bf16_to_f32(x)).collect();
        let q8_vals: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) * 0.1);
        let (q8_block, _) = q8_0_roundtrip_block(&q8_vals);

        let kvs = vec![
            ("general.architecture", GgufValue::Str("test".into())),
            ("general.alignment", GgufValue::U32(32)),
            ("test.block_count", GgufValue::U32(7)),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(vec![GgufValue::Str("a".into()), GgufValue::Str("b".into())]),
            ),
        ];
        // ggml ne-dims are the REVERSE of logical: F32 logical [2,3,4] -> ne [4,3,2].
        let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
            ("a_f32", vec![4, 3, 2], GGML_F32, f32_bytes),
            ("b_f16", vec![8], GGML_F16, f16_bytes.clone()),
            ("c_q8", vec![32, 1], GGML_Q8_0, q8_block.to_vec()),
            ("d_bf16", vec![4], GGML_BF16, bf16_bytes.clone()),
        ];
        let g = WholeGguf::from_bytes(write_gguf(&kvs, &tensors)).expect("parse written gguf");

        // metadata + array round-trip.
        assert_eq!(g.version, 3);
        assert_eq!(
            g.get("general.architecture").and_then(|v| v.as_str()),
            Some("test")
        );
        assert_eq!(g.get("test.block_count").and_then(|v| v.as_u64()), Some(7));
        assert_eq!(
            g.get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .map(|a| a.len()),
            Some(2)
        );

        // F32 is exact (and the dims reverse to the logical shape).
        let a = g.dequant("a_f32").unwrap();
        assert_eq!(a.shape(), vec![2, 3, 4]);
        assert_eq!(a.as_f32().unwrap(), f32_data);
        // F16 within its rounding.
        let b = g.dequant("b_f16").unwrap();
        assert_eq!(b.shape(), vec![8]);
        for (got, want) in b.as_f32().unwrap().iter().zip(&f16_vals) {
            assert!((got - want).abs() < 1e-2, "f16 {got} vs {want}");
        }
        // BF16 widens from, and retains, the exact same little-endian source words.
        let d = g.dequant("d_bf16").unwrap();
        assert_eq!(d.shape(), vec![4]);
        assert_eq!(
            d.as_f32().unwrap(),
            bf16_vals,
            "bf16 widening must match source words"
        );
        // Q8_0 within its step.
        let c = g.dequant("c_q8").unwrap();
        for (got, want) in c.as_f32().unwrap().iter().zip(&q8_vals) {
            assert!((got - want).abs() <= 0.02, "q8_0 {got} vs {want}");
        }
        // The one reader keeps each dense tensor's stored bytes exactly (spec 135 Phase 2, FR-006):
        // a GPU upload reads the source, not a re-encode.
        let store = read_gguf(&g, g.bytes.as_slice(), &IdentityNames).expect("read_gguf");
        let stored = |name: &str| match store.get(name) {
            Some(WeightEntry::Dense(dense)) => (dense.dtype(), dense.bytes().as_slice().to_vec()),
            other => panic!("{name}: {other:?}"),
        };
        assert_eq!(stored("b_f16"), (DType::F16, f16_bytes));
        assert_eq!(stored("d_bf16"), (DType::BF16, bf16_bytes));
        assert!(matches!(store.get("c_q8"), Some(WeightEntry::Packed(_))));
    }

    #[test]
    fn gguf_writer_round_trips_bool_metadata() {
        // Card 223: `gguf_encode_value` panicked on GgufValue::Bool although the reader
        // (`Cursor::value`, type id 7) decodes it; real GGUFs use it for
        // tokenizer.ggml.add_bos_token. No tensors are needed.
        let kvs = vec![
            ("tokenizer.ggml.add_bos_token", GgufValue::Bool(true)),
            ("tokenizer.ggml.add_eos_token", GgufValue::Bool(false)),
        ];
        let bytes = write_gguf(&kvs, &[]);
        let g = WholeGguf::from_bytes(bytes).expect("parse written gguf");
        assert_eq!(
            g.get("tokenizer.ggml.add_bos_token"),
            Some(&GgufValue::Bool(true))
        );
        assert_eq!(
            g.get("tokenizer.ggml.add_eos_token"),
            Some(&GgufValue::Bool(false))
        );
    }

    #[test]
    fn gguf_rejects_malformed_tensor_dims_instead_of_panicking() {
        // A GGUF is an untrusted file. A crafted header must be rejected at parse, not reach
        // `dequant`/`pack_*` where the dims-derived count feeds `vec![0.0; numel]` (capacity
        // overflow) or a wrapped block size that bypasses `bytes_ok`. `write_gguf` emits whatever
        // dims it is given, independent of the small data blob.

        // (1) A single huge dim: numel exceeds MAX_TENSOR_ELEMS -> error, no panic.
        let huge = write_gguf(
            &[],
            &[(
                "token_embd.weight",
                vec![1u64 << 50],
                GGML_F32,
                vec![0u8; 8],
            )],
        );
        assert!(
            WholeGguf::from_bytes(huge).is_err(),
            "a tensor with an astronomically large element count must be rejected"
        );

        // (2) Dims whose product overflows usize: the checked product errors rather than wrapping
        // to a small numel that slips past the byte-length guard.
        let overflow = write_gguf(&[], &[("w", vec![u64::MAX, 2, 2], GGML_F32, vec![0u8; 8])]);
        assert!(
            WholeGguf::from_bytes(overflow).is_err(),
            "dims whose product overflows usize must be rejected"
        );

        // (3) Excessive rank: guarded before `Vec::with_capacity(n_dims)` (rank 9 >
        // MAX_TENSOR_DIMS = 8; each dim 1 so numel is valid).
        let deep = write_gguf(&[], &[("w", vec![1u64; 9], GGML_F32, vec![0u8; 8])]);
        assert!(
            WholeGguf::from_bytes(deep).is_err(),
            "a tensor rank above the maximum must be rejected"
        );

        // A small legitimate tensor still parses.
        let ok = write_gguf(&[], &[("w", vec![4u64, 2], GGML_F32, vec![0u8; 32])]);
        assert!(
            WholeGguf::from_bytes(ok).is_ok(),
            "a valid 2-D tensor must still parse"
        );
    }

    #[test]
    #[ignore = "diagnostic: print tensor ggml types for each layer; needs POOT_MODELS_DIR (and optionally POOT_DIAG_LAYERS)"]
    fn diag_gguf_tensor_types() {
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )) else {
            return;
        };
        let n_layers: usize = std::env::var("POOT_DIAG_LAYERS")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .expect("numeric POOT_DIAG_LAYERS");
        let g = WholeGguf::load(&path).expect("load gguf");
        let type_name = |t: u32| {
            match t {
                0 => "F32",
                1 => "F16",
                2 => "Q4_0",
                3 => "Q4_1",
                6 => "Q5_0",
                7 => "Q5_1",
                8 => "Q8_0",
                10 => "Q2_K",
                11 => "Q3_K",
                12 => "Q4_K",
                13 => "Q5_K",
                14 => "Q6_K",
                15 => "Q8_K",
                30 => "BF16",
                x => return format!("?({})", x),
            }
            .to_string()
        };
        let projs = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ];
        for i in 0..n_layers {
            for p in &projs {
                let name = format!("blk.{i}.{p}");
                if let Some(info) = g.tensors.get(&name) {
                    eprintln!(
                        "blk.{i}.{}: {} dims={:?}",
                        p,
                        type_name(info.ggml_type),
                        info.dims
                    );
                } else {
                    eprintln!("blk.{i}.{p}: MISSING");
                }
            }
        }
    }

    /// Card 543 SC-004: every ggml type id 0..=39 maps through [`weight_format_of`] to its format or
    /// `None`, pinned against a literal table (not derived, so it can be checked against `ggml.h`
    /// line by line). MUTATION (card 543, manually applied and reverted, never left in the tree):
    /// mapping id 7 (Q5_1) to `Some(WeightFormat::Q5_0)` in the `weight_format_of` table makes this
    /// row's `assert_eq!` fail (`Some(Q5_0) != Some(Q5_1)`) - the acceptance mutation the card names
    /// ("map Q5_1 to Q5_0") shows up here, not in `read_gguf_refuses_...` (that row is about an id
    /// with no packed path at all, e.g. Q8_1).
    #[test]
    fn weight_format_of_matches_the_pinned_ggml_type_table() {
        let mapped: &[(u32, WeightFormat)] = &[
            (0, WeightFormat::F32),
            (1, WeightFormat::F16),
            (2, WeightFormat::Q4_0),
            (3, WeightFormat::Q4_1),
            (6, WeightFormat::Q5_0),
            (7, WeightFormat::Q5_1),
            (8, WeightFormat::Q8_0),
            (10, WeightFormat::Q2_K),
            (11, WeightFormat::Q3_K),
            (12, WeightFormat::Q4_K),
            (13, WeightFormat::Q5_K),
            (14, WeightFormat::Q6_K),
            (20, WeightFormat::Iq4_Nl),
            (23, WeightFormat::Iq4_Xs),
            (30, WeightFormat::Bf16),
            (39, WeightFormat::Mxfp4),
        ];
        for &(id, format) in mapped {
            assert_eq!(weight_format_of(id), Some(format), "ggml type {id}");
        }
        let unmapped: &[u32] = &[
            4, 5, 9, 15, 16, 17, 18, 19, 21, 22, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36,
            37, 38,
        ];
        for &id in unmapped {
            assert_eq!(weight_format_of(id), None, "ggml type {id}");
        }
        // Every id 0..=39 is exactly one of the two lists above; nothing pinned falls through.
        assert_eq!(mapped.len() + unmapped.len(), 40);
        assert_eq!(
            weight_format_of(40),
            None,
            "40 is past the pinned 0..=39 table"
        );
    }

    /// `read_gguf`'s block-format fixtures below need blocks of the right byte length; this reads
    /// that length off `format`'s own descriptor rather than a restated literal, per format.rs's
    /// module docs ("the layout of a format is written once, here, and nowhere else").
    fn block_bytes_and_values(format: WeightFormat) -> (usize, usize) {
        let poot_quant::format::Storage::Blocks(layout) = format.descriptor().storage else {
            panic!("{format:?} is not a block format");
        };
        (layout.bytes, layout.values)
    }

    /// `rows` rows of `k` values of `format`'s native blocks, as all-zero bytes. Every float-encoded
    /// field of a zero block decodes to `0.0` (finite), so `PackedPayload::try_new`'s content
    /// validation always accepts it: these fixtures test the reader's byte routing and accounting,
    /// not decode correctness (covered by 541/542a/542b's oracles).
    fn zero_blocks(format: WeightFormat, rows: usize, k: usize) -> Vec<u8> {
        let (block_bytes, block_values) = block_bytes_and_values(format);
        assert!(k.is_multiple_of(block_values));
        vec![0u8; rows * (k / block_values) * block_bytes]
    }

    fn write_temp_gguf(bytes: &[u8]) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "poot-load-card543-gguf-{}-{nanos}-{id}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, bytes).expect("write temp gguf fixture");
        path
    }

    /// A dropped-on-scope-exit temp file, so an assertion failure mid-test still cleans up.
    struct TempGguf(std::path::PathBuf);

    impl TempGguf {
        fn write(bytes: &[u8]) -> Self {
            Self(write_temp_gguf(bytes))
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Card 543 SC-003 (packed load) / SC-002 (no transcoding): a Q2_K fixture loads as one packed
    /// owner whose stored bytes equal the file's tensor bytes exactly - no transcode, no widened
    /// scale. MUTATION (card 543, manually applied and reverted, never left in the tree): routing
    /// Q2_K through the dense branch (temporarily forcing `descriptor.has_scale()` to `false` for
    /// `WeightFormat::Q2_K` in `read_gguf`) makes this row red: `DenseWeight::try_new` fails with
    /// `ByteLength { dtype: F32, ... }` because the block bytes are not a multiple of 4 f32s per row.
    #[test]
    fn read_gguf_loads_a_q2_k_tensor_packed_with_stored_bytes_equal_to_the_file_bytes() {
        let (out, k) = (2usize, 256usize);
        let data = zero_blocks(WeightFormat::Q2_K, out, k);
        let bytes = write_gguf(
            &[],
            &[(
                "blk.0.attn_v.weight",
                vec![k as u64, out as u64],
                GGML_Q2_K,
                data.clone(),
            )],
        );
        let fixture = TempGguf::write(&bytes);
        let index = GgufIndex::open(fixture.path()).expect("open index");
        let file = File::open(fixture.path()).expect("open file");
        let store = read_gguf(&index, &file, &IdentityNames).expect("read_gguf");

        assert_eq!(store.len(), 1);
        let entry = store.get("blk.0.attn_v.weight").expect("missing entry");
        let WeightEntry::Packed(payload) = entry else {
            panic!("expected a packed entry, got {entry:?}");
        };
        assert_eq!(payload.weight().format(), WeightFormat::Q2_K);
        assert_eq!(payload.weight().shape(), [out, k]);
        assert_eq!(payload.bytes(SourceRole::Blocks), data.as_slice());
        assert_eq!(store.total_stored_bytes(), data.len());
    }

    /// Card 543 SC-003/SC-004 (refusal): a scheme with no packed path (`Q8_1`, ggml id 9) is refused
    /// by name, naming the tensor - never loaded dense silently.
    #[test]
    fn read_gguf_refuses_a_scheme_with_no_packed_path_by_name() {
        const GGML_Q8_1: u32 = 9;
        let bytes = write_gguf(
            &[],
            &[("blk.0.attn_v.weight", vec![32], GGML_Q8_1, vec![0u8; 34])],
        );
        let fixture = TempGguf::write(&bytes);
        let index = GgufIndex::open(fixture.path()).expect("open index");
        let file = File::open(fixture.path()).expect("open file");
        let error = read_gguf(&index, &file, &IdentityNames).expect_err("must refuse Q8_1");
        match error {
            LoadError::UnsupportedGgufType { tensor, ggml_type } => {
                assert_eq!(tensor, "blk.0.attn_v.weight");
                assert_eq!(ggml_type, GGML_Q8_1);
            }
            other => panic!("expected UnsupportedGgufType, got {other:?}"),
        }
    }

    /// A tensor `names` does not map is skipped before its type is even looked at, so an unsupported
    /// ggml type on a tensor the caller never reads does not refuse the whole load.
    #[test]
    fn read_gguf_skips_a_tensor_the_name_map_does_not_map() {
        struct SkipEverything;
        impl TensorNameMap for SkipEverything {
            fn weight_key(&self, _gguf_name: &str) -> Option<WeightKey> {
                None
            }
        }
        const GGML_Q8_1: u32 = 9; // no packed path; would refuse if not skipped first
        let bytes = write_gguf(
            &[],
            &[("blk.0.attn_v.weight", vec![32], GGML_Q8_1, vec![0u8; 34])],
        );
        let fixture = TempGguf::write(&bytes);
        let index = GgufIndex::open(fixture.path()).expect("open index");
        let file = File::open(fixture.path()).expect("open file");
        let store = read_gguf(&index, &file, &SkipEverything).expect("skips, does not refuse");
        assert!(store.is_empty());
    }

    /// Card 543 SC-005: a rank-3 expert fixture loads as `E` separate packed owners (ADR-0109: never
    /// stacked on the host), each owner's bytes equal to that expert's own byte range of the file.
    /// MUTATION (card 543, manually applied and reverted, never left in the tree): an off-by-one
    /// expert stride (`expert_len.checked_mul(expert + 1)` instead of `expert`) turns this row red:
    /// observed failure `read_gguf: Io(Error { kind: UnexpectedEof, message: "failed to fill whole
    /// buffer" })`, since the last expert's shifted range reads past the file's tensor-data end.
    #[test]
    fn read_gguf_loads_rank3_experts_as_separate_owners_matching_the_files_per_expert_bytes() {
        let (experts, out, k) = (3usize, 2usize, 256usize);
        let (block_bytes, block_values) = block_bytes_and_values(WeightFormat::Q4_K);
        assert!(k.is_multiple_of(block_values));
        let per_expert = out * (k / block_values) * block_bytes;
        let mut data = vec![0u8; experts * per_expert];
        for expert in 0..experts {
            // < 16 keeps every f16-interpreted word in the block subnormal (exponent bits all 0),
            // hence finite, while still distinguishing each expert's byte range.
            let fill = (expert + 1) as u8;
            data[expert * per_expert..(expert + 1) * per_expert].fill(fill);
        }
        // ggml ne = [K, out, E] (fastest first); the reversed logical shape is [E, out, K].
        let bytes = write_gguf(
            &[],
            &[(
                "blk.0.ffn_gate_exps.weight",
                vec![k as u64, out as u64, experts as u64],
                GGML_Q4_K,
                data.clone(),
            )],
        );
        let fixture = TempGguf::write(&bytes);
        let index = GgufIndex::open(fixture.path()).expect("open index");
        let file = File::open(fixture.path()).expect("open file");
        let store = read_gguf(&index, &file, &IdentityNames).expect("read_gguf");

        assert_eq!(store.len(), experts);
        for expert in 0..experts {
            let key = format!("blk.0.ffn_gate_exps.weight.{expert}");
            let entry = store
                .get(&key)
                .unwrap_or_else(|| panic!("missing owner {key}"));
            let WeightEntry::Packed(payload) = entry else {
                panic!("expected a packed entry for {key}, got {entry:?}");
            };
            assert_eq!(payload.weight().format(), WeightFormat::Q4_K);
            assert_eq!(payload.weight().shape(), [out, k]);
            let expected = &data[expert * per_expert..(expert + 1) * per_expert];
            assert_eq!(
                payload.bytes(SourceRole::Blocks),
                expected,
                "expert {expert}"
            );
        }
    }

    /// Dense GGUF types (F32, F16, BF16) load as [`WeightEntry::Dense`], stored bytes equal to the
    /// file's tensor bytes, no widening: the other half of SC-002 alongside the packed row above.
    #[test]
    fn read_gguf_loads_dense_types_with_no_widening() {
        let f32_vals = [1.0f32, -2.5, 0.0, 3.25];
        let f32_bytes: Vec<u8> = f32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16_bytes: Vec<u8> = [0x3c00u16, 0xbc00, 0x0000, 0x4200] // 1.0, -1.0, 0.0, 3.0
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = write_gguf(
            &[],
            &[
                (
                    "norm.weight",
                    vec![f32_vals.len() as u64],
                    GGML_F32,
                    f32_bytes.clone(),
                ),
                (
                    "blk.0.attn_norm.weight",
                    vec![4],
                    GGML_F16,
                    f16_bytes.clone(),
                ),
            ],
        );
        let fixture = TempGguf::write(&bytes);
        let index = GgufIndex::open(fixture.path()).expect("open index");
        let file = File::open(fixture.path()).expect("open file");
        let store = read_gguf(&index, &file, &IdentityNames).expect("read_gguf");

        let WeightEntry::Dense(f32_entry) = store.get("norm.weight").expect("missing f32 entry")
        else {
            panic!("expected a dense entry");
        };
        assert_eq!(f32_entry.dtype(), DType::F32);
        assert_eq!(f32_entry.bytes().as_slice(), f32_bytes.as_slice());

        let WeightEntry::Dense(f16_entry) = store
            .get("blk.0.attn_norm.weight")
            .expect("missing f16 entry")
        else {
            panic!("expected a dense entry");
        };
        assert_eq!(f16_entry.dtype(), DType::F16);
        assert_eq!(f16_entry.bytes().as_slice(), f16_bytes.as_slice());
        assert_eq!(
            store.total_stored_bytes(),
            f32_bytes.len() + f16_bytes.len()
        );
    }

    /// `GgufIndex::from_bytes` + `read_gguf` over an in-memory `[u8]` buffer (the `GgufSource` impl for
    /// `[u8]`, card 543's twin of 540b): a fixture or test with the whole small GGUF already in memory
    /// needs no temp file. Same content check as the packed on-disk test above, over the in-memory path.
    #[test]
    fn read_gguf_works_over_an_in_memory_buffer_with_no_temp_file() {
        let (out, k) = (2usize, 32usize);
        let data = zero_blocks(WeightFormat::Q4_0, out, k);
        let bytes = write_gguf(
            &[],
            &[(
                "blk.0.attn_v.weight",
                vec![k as u64, out as u64],
                GGML_Q4_0,
                data.clone(),
            )],
        );
        let index = GgufIndex::from_bytes(&bytes).expect("parse in-memory gguf");
        let store = read_gguf(&index, bytes.as_slice(), &IdentityNames).expect("read_gguf");
        let entry = store.get("blk.0.attn_v.weight").expect("missing entry");
        let WeightEntry::Packed(payload) = entry else {
            panic!("expected a packed entry, got {entry:?}");
        };
        assert_eq!(payload.weight().format(), WeightFormat::Q4_0);
        assert_eq!(payload.bytes(SourceRole::Blocks), data.as_slice());
    }

    /// M4 mutation sample, M3: the path reader's 256 MB header cap admits a real-sized tokenizer
    /// table. A GGUF whose one string-array KV is over 2 MiB (300k short tokens) parses through
    /// `GgufIndex::open` and round-trips the array length (a cap cut to ~1 MiB or 0 turns it red).
    #[test]
    fn a_multi_megabyte_tokenizer_table_parses_through_the_path_reader() {
        const TOKENS: usize = 300_000;
        let tokens: Vec<GgufValue> = (0..TOKENS)
            .map(|t| GgufValue::Str(format!("tok{t}")))
            .collect();
        let bytes = write_gguf(&[("tokenizer.ggml.tokens", GgufValue::Array(tokens))], &[]);
        assert!(bytes.len() > 2 << 20, "{} bytes", bytes.len());
        let path = poot_test_util::unique_temp_path("poot_gguf_big_header.gguf");
        std::fs::write(&path, &bytes).unwrap();
        let index = GgufIndex::open(&path).expect("parse a multi-megabyte header");
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            index
                .get("tokenizer.ggml.tokens")
                .and_then(GgufValue::as_array)
                .map(<[GgufValue]>::len),
            Some(TOKENS)
        );
    }

    /// M4: a GGUF cut anywhere inside its KV section or tensor directory is a typed error, never a
    /// panic; a header-only GGUF whose last KV ends exactly at end of file parses. (`Cursor::take`
    /// refusing a read that ends at EOF turns the second half red; reading past it would panic.)
    #[test]
    fn a_truncated_gguf_is_a_typed_error() {
        let q8: [f32; 32] = std::array::from_fn(|i| i as f32 * 0.25);
        let (block, _) = q8_0_roundtrip_block(&q8);
        let bytes = write_gguf(
            &[
                ("general.architecture", GgufValue::Str("test".into())),
                ("test.block_count", GgufValue::U32(3)),
                (
                    "test.names",
                    GgufValue::Array(vec![GgufValue::Str("x".into()); 4]),
                ),
            ],
            &[("w", vec![32, 1], GGML_Q8_0, block.to_vec())],
        );
        let header_end = GgufIndex::from_bytes(&bytes).unwrap().data_start();
        for len in 0..header_end {
            let prefix = &bytes[..len];
            assert!(
                std::panic::catch_unwind(|| GgufIndex::from_bytes(prefix).is_err())
                    .unwrap_or(false),
                "a GGUF cut at {len} of {header_end} header bytes must be a typed error"
            );
        }
        let header_only = write_gguf(
            &[
                ("general.alignment", GgufValue::U32(1)),
                ("test.name", GgufValue::Str("last".into())),
            ],
            &[],
        );
        let end = GgufIndex::from_bytes(&header_only).unwrap().data_start();
        let index = GgufIndex::from_bytes(&header_only[..end])
            .expect("a header whose last KV ends at end of file parses");
        assert_eq!(
            index.get("test.name").and_then(GgufValue::as_str),
            Some("last")
        );
    }

    /// M6: the typed metadata accessors (rope base, add_bos, hyperparameters) read exactly their
    /// own variants: a float accessor answering 0.0/1.0, or an integer accessor losing a width,
    /// turns this red.
    #[test]
    fn gguf_value_accessors_are_typed() {
        assert_eq!(GgufValue::F32(10000.0).as_f32(), Some(10000.0));
        assert_eq!(GgufValue::F64(10000.0).as_f32(), Some(10000.0));
        assert_eq!(GgufValue::Str("1e4".into()).as_f32(), None);
        assert_eq!(GgufValue::U32(7).as_f32(), None);
        assert_eq!(GgufValue::Bool(false).as_bool(), Some(false));
        assert_eq!(GgufValue::Bool(true).as_bool(), Some(true));
        assert_eq!(GgufValue::U8(1).as_bool(), None);
        for value in [
            GgufValue::U8(7),
            GgufValue::I8(7),
            GgufValue::U16(7),
            GgufValue::I16(7),
            GgufValue::U32(7),
            GgufValue::I32(7),
            GgufValue::U64(7),
            GgufValue::I64(7),
        ] {
            assert_eq!(value.as_u64(), Some(7), "{value:?}");
        }
        assert_eq!(GgufValue::F32(7.0).as_u64(), None);
        assert_eq!(GgufValue::Str("x".into()).as_str(), Some("x"));
        assert_eq!(GgufValue::U8(7).as_str(), None);
        assert_eq!(GgufValue::U8(7).as_array(), None);
        // The read path: an F32 and a Bool KV come back through the index.
        let bytes = write_gguf(
            &[
                ("test.rope.freq_base", GgufValue::F32(500000.0)),
                ("tokenizer.ggml.add_bos_token", GgufValue::Bool(false)),
            ],
            &[],
        );
        let index = GgufIndex::from_bytes(&bytes).unwrap();
        assert_eq!(
            index.get("test.rope.freq_base").and_then(GgufValue::as_f32),
            Some(500000.0)
        );
        assert_eq!(
            index
                .get("tokenizer.ggml.add_bos_token")
                .and_then(GgufValue::as_bool),
            Some(false)
        );
    }

    /// L5: a tensor of exactly `MAX_TENSOR_ELEMS` elements is admitted (header only, no data read);
    /// one more row is refused.
    #[test]
    fn the_tensor_element_cap_is_inclusive() {
        let at_cap = write_gguf(&[], &[("w", vec![1 << 20, 1 << 20], GGML_F32, Vec::new())]);
        assert!(GgufIndex::from_bytes(&at_cap).is_ok());
        let over = write_gguf(
            &[],
            &[("w", vec![1 << 20, (1 << 20) + 1], GGML_F32, Vec::new())],
        );
        assert!(GgufIndex::from_bytes(&over).is_err());
    }
}

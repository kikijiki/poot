//! Per-tensor dequant parity for Qwen3.8-Flash-Next's UD-Q2_K_XL GGUF against an independent
//! reference. `data/flash_next_dequant_samples.tsv` holds, for one real tensor of each ggml type the
//! checkpoint stores (Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, BF16, F32, and the lattice quants IQ2_XS and
//! IQ3_XXS of the expert gate stacks), three runs of consecutive blocks (start, middle, end of the
//! tensor) cut from the published shards with ranged HTTP reads, and the
//! values llama.cpp's gguf-py (`gguf.quants.dequantize`, numpy) decodes them to. Each row is written
//! as a one-row tensor of a GGUF and driven through the production entries (`read_gguf`, then the
//! store entry's own decode); nothing is decoded by test code. Model-free: no checkpoint is read.
//!
//! One extra row cuts the last expert (511) of a stacked IQ4_NL expert tensor at its real byte offset
//! (`511 * 921600`, the stride the reader's per-expert expansion uses); the qwen38 loader test checks
//! that stride against the plan. The IQ2_XS and IQ3_XXS tensors (`blk.0` and `blk.2`
//! `ffn_gate_exps`, four blocks per run) have the same extra row at `511 * 473600` and
//! `511 * 627200`.
//!
//! The last six rows are from the re-quantised UD-Q2_K_XL (upstream revision `084c631a`, 2026-11-07), whose
//! expert gate stacks use `IQ2_XXS` (`blk.0`, ggml type 16) and `IQ2_S` (`blk.2`, type 22); their expected
//! values are llama.cpp b9190's own `to_float` of the same bytes (`ggml_reference`), three runs of four blocks
//! each cut by ranged reads. That revision also stores ggml type 42 on its `ffn_down_exps` stacks, a type
//! newer than the b9190 pin that poot refuses by name.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1189 report):
//! DQ-1 one entry of the IQ4_NL codebook is changed; DQ-2 the Q6_K high-bit plane is read from the
//! wrong offset; DQ-3 the BF16 widening drops its shift; DQ-4 two entries of the IQ2_XS grid are
//! swapped; DQ-5 two entries of the IQ3_XXS grid are swapped.
//! POOT-1540: DQ-6 `iq2xxs_grid[0]` and DQ-7 `iq2s_grid[0]` are changed (red on the type 16 and 22 rows).

use poot_load::gguf::{GgufIndex, IdentityNames, read_gguf, write_gguf};
use poot_quant::StoredBytes;
use poot_quant::format::Storage;
use poot_quant::weights::WeightEntry;

use poot_load::gguf::weight_format_of;

const SAMPLES: &str = include_str!("data/flash_next_dequant_samples.tsv");
const TQ2_0_SAMPLES: &str = include_str!("data/bitnet_tq2_0_dequant_samples.tsv");

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

struct Sample {
    tensor: String,
    ggml_type: u32,
    offset: usize,
    data: Vec<u8>,
    expected: Vec<f32>,
}

fn samples(text: &str) -> Vec<Sample> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let mut cols = line.split('\t');
            let mut next = || cols.next().expect("five columns");
            let (tensor, ggml_type, offset) = (next(), next(), next());
            let (data, expected) = (unhex(next()), unhex(next()));
            Sample {
                tensor: tensor.to_string(),
                ggml_type: ggml_type.parse().unwrap(),
                offset: offset.parse().unwrap(),
                expected: expected
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect(),
                data,
            }
        })
        .collect()
}

/// The values `data` decodes to when stored as one row of `ggml_type`, through `read_gguf`.
fn decode(sample: &Sample) -> Vec<f32> {
    let format = weight_format_of(sample.ggml_type).expect("a supported type");
    let values = match format.descriptor().storage {
        Storage::Blocks(layout) => sample.data.len() / layout.bytes * layout.values,
        _ => sample.data.len() / sample.ggml_type_bytes(),
    };
    let bytes = write_gguf(
        &[],
        &[(
            "w",
            vec![values as u64, 1],
            sample.ggml_type,
            sample.data.clone(),
        )],
    );
    let file = StoredBytes::from(bytes);
    let index = GgufIndex::parse(&file).unwrap();
    let store = read_gguf(&index, &IdentityNames).unwrap();
    match store.get("w").expect("the tensor is stored") {
        WeightEntry::Packed(payload) => {
            let mut out = vec![0.0; values];
            payload.decode_row(0, &mut out).unwrap();
            out
        }
        WeightEntry::Dense(dense) => dense.decode_f32().unwrap(),
    }
}

impl Sample {
    /// The byte width of one value of a dense type (F32 or BF16).
    fn ggml_type_bytes(&self) -> usize {
        match self.ggml_type {
            0 => 4,
            30 => 2,
            other => panic!("ggml type {other} is not dense"),
        }
    }
}

/// DQ-1..3: every sampled run of every stored type decodes to gguf-py's values, bit for bit.
#[test]
fn dq1_every_stored_type_decodes_to_the_reference() {
    let samples = samples(SAMPLES);
    // Twelve tensors of eleven types, three runs each, plus a last-expert run of three of them.
    assert_eq!(samples.len(), 12 * 3 + 3);
    let mut seen = std::collections::BTreeSet::new();
    for sample in &samples {
        seen.insert(sample.ggml_type);
        let got = decode(sample);
        assert_eq!(got.len(), sample.expected.len(), "{}", sample.tensor);
        if let Some(at) = (0..got.len()).find(|&i| got[i].to_bits() != sample.expected[i].to_bits())
        {
            panic!(
                "{} (ggml type {}) at byte {}: value {at} decodes to {} ({:#010x}), the reference says {} ({:#010x})",
                sample.tensor,
                sample.ggml_type,
                sample.offset,
                got[at],
                got[at].to_bits(),
                sample.expected[at],
                sample.expected[at].to_bits()
            );
        }
    }
    // F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ2_S and BF16: every type the set stores.
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        [0, 8, 12, 13, 14, 16, 17, 18, 20, 22, 30]
    );
}

/// `TQ2_0` has no tensor in the Flash-Next checkpoint, so its real blocks come from the BitNet b1.58 2B-4T
/// `TQ2_0` GGUF (`data/bitnet_tq2_0_dequant_samples.tsv`: three runs of four blocks of one tensor, the values
/// llama.cpp b9190's `to_float` decodes them to). Mutation: change the `TQ2_0` code offset in
/// `blocks/kquant.rs` from -1 to 0; red with `blk.0.ffn_down.weight (ggml type 35) at byte 0: value 0 decodes to 0
/// (0x00000000), the reference says -2.1640625 (0xc00a8000)`.
#[test]
fn dq4_tq2_0_real_blocks_decode_to_the_reference() {
    let samples = samples(TQ2_0_SAMPLES);
    assert_eq!(samples.len(), 3);
    for sample in &samples {
        assert_eq!(sample.ggml_type, 35);
        let got = decode(sample);
        assert_eq!(got.len(), sample.expected.len(), "{}", sample.tensor);
        if let Some(at) = (0..got.len()).find(|&i| got[i].to_bits() != sample.expected[i].to_bits())
        {
            panic!(
                "{} (ggml type 35) at byte {}: value {at} decodes to {} ({:#010x}), the reference says {} ({:#010x})",
                sample.tensor,
                sample.offset,
                got[at],
                got[at].to_bits(),
                sample.expected[at],
                sample.expected[at].to_bits()
            );
        }
    }
}

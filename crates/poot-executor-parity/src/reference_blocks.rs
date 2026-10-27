//! The packed decode GEMV against llama.cpp's own dequantisation, on the real blocks of the Flash-Next
//! checkpoint (POOT-1337).
//!
//! `poot-load`'s `flash_next_dequant_samples.tsv` holds, for one real tensor of each ggml type the checkpoint
//! stores, runs of consecutive blocks cut from the published shards and the values llama.cpp's gguf-py
//! (`gguf.quants.dequantize`) decodes them to. Each block-format run becomes a packed weight of one block per
//! row (`[blocks, values-per-block]`), and the fixture is the packed linear at one activation row (the decode
//! Gemv) stepped once per column with a one-hot activation, so step `c`'s output row is column `c` of every
//! block, a single product `1.0 * w` plus zeros: exact on any executor whatever its reduction order. Every
//! element must equal gguf-py's value bit for bit. The reference is independent of poot's own decoder
//! (`PackedPayload::decode_row`, the CPU oracle every other parity row compares with), and the blocks are
//! the real tensors' (the IQ2_XS grid and signs, the IQ4_NL codebook, the K-quant sub-scales).

use std::sync::Arc;

use poot_eval::test_support::StepFixture;
use poot_executor::Executor;
use poot_graph_ir::test_support::packed_graphs::packed_linear;
use poot_graph_ir::{Builder, Slot, TensorType};
use poot_graph_plan::Target;
use poot_quant::StoredBytes;
use poot_quant::format::{Storage, WeightFormat};
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_quant::{PackedPayload, PackedWeight, SourceRole};
use poot_tensor::HostTensor;

use crate::{Fixture, run_outputs};

const SAMPLES: &str = include_str!("../../poot-load/tests/data/flash_next_dequant_samples.tsv");

/// One reference run: its fixture and the values gguf-py decodes its blocks to, `blocks * values` of them,
/// block-major.
pub struct ReferenceRow {
    /// The decode Gemv over the run, one one-hot step per column.
    pub fixture: Fixture,
    /// The tensor the blocks were cut from, for a failure message.
    pub tensor: String,
    /// The byte offset of the run in the tensor.
    pub offset: usize,
    /// Values per block: the contraction length and the number of steps.
    pub values: usize,
    /// gguf-py's decode of the blocks.
    pub expected: Vec<f32>,
}

/// The block formats the checkpoint stores, by ggml type id, with a fixture name.
fn format_of(ggml_type: u32) -> Option<(WeightFormat, &'static str)> {
    Some(match ggml_type {
        8 => (WeightFormat::Q8_0, "reference_q8_0"),
        12 => (WeightFormat::Q4_K, "reference_q4_k"),
        13 => (WeightFormat::Q5_K, "reference_q5_k"),
        14 => (WeightFormat::Q6_K, "reference_q6_k"),
        17 => (WeightFormat::Iq2_Xs, "reference_iq2_xs"),
        18 => (WeightFormat::Iq3_Xxs, "reference_iq3_xxs"),
        20 => (WeightFormat::Iq4_Nl, "reference_iq4_nl"),
        _ => return None,
    })
}

fn unhex(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// Every block-format run of the reference samples as a fixture.
pub fn reference_block_rows() -> Vec<ReferenceRow> {
    SAMPLES
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let mut cols = line.split('\t');
            let mut next = || cols.next().expect("five columns");
            let (tensor, ggml_type, offset) = (next(), next(), next());
            let (data, expected) = (unhex(next()), unhex(next()));
            let (format, name) = format_of(ggml_type.parse().unwrap())?;
            let Storage::Blocks(layout) = format.descriptor().storage else {
                panic!("{name}: not a block format");
            };
            let blocks = data.len() / layout.bytes;
            let weight = PackedWeight::try_new(format, [blocks, layout.values]).unwrap();
            let payload =
                PackedPayload::try_new(weight, vec![(SourceRole::Blocks, StoredBytes::from(data))])
                    .unwrap_or_else(|e| panic!("{tensor}: {e:?}"));
            let expected: Vec<f32> = expected
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            assert_eq!(expected.len(), blocks * layout.values, "{tensor}");
            let b = Builder::for_tests();
            let x = b
                .slot_named(
                    Slot::Activation,
                    "x",
                    TensorType::f32(vec![1, layout.values]),
                )
                .unwrap();
            let y = packed_linear(&b, x, "layer", weight, None, None).unwrap();
            let graph = b.finish(y).unwrap();
            let key = graph.meta(x.id()).slot_key().unwrap().clone();
            let steps = (0..layout.values)
                .map(|column| {
                    let mut one_hot = vec![0.0f32; layout.values];
                    one_hot[column] = 1.0;
                    vec![StepFixture {
                        key: key.clone(),
                        tensor: HostTensor::f32(vec![1, layout.values], one_hot),
                    }]
                })
                .collect();
            let mut store = WeightStore::builder();
            store
                .insert("layer".to_string(), WeightEntry::Packed(Arc::new(payload)))
                .unwrap();
            Some(ReferenceRow {
                fixture: Fixture {
                    name,
                    graph,
                    store: store.build(),
                    steps,
                },
                tensor: tensor.to_string(),
                offset: offset.parse().unwrap(),
                values: layout.values,
                expected,
            })
        })
        .collect()
}

/// Run every reference row on `exec` and assert each decoded element equals gguf-py's, bit for bit (a zero
/// compares equal whatever its sign).
pub fn assert_reference_blocks(exec: &mut dyn Executor, target: Target) {
    let rows = reference_block_rows();
    assert!(!rows.is_empty(), "no block-format reference samples");
    for row in &rows {
        let outputs = run_outputs(exec, target, &row.fixture).unwrap_or_else(|e| panic!("{e}"));
        for (column, output) in outputs.iter().enumerate() {
            let got = output.as_f32().expect("F32 output");
            for (block, &got) in got.iter().enumerate() {
                let want = row.expected[block * row.values + column];
                assert_eq!(
                    (got + 0.0).to_bits(),
                    (want + 0.0).to_bits(),
                    "{} ({}) at byte {}: block {block} column {column} decodes on the executor to {got} \
                     ({:#010x}), llama.cpp's gguf-py says {want} ({:#010x})",
                    row.tensor,
                    row.fixture.name,
                    row.offset,
                    got.to_bits(),
                    want.to_bits()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_eval::test_support::weight_map_oracle::oracle_for;

    /// The fixture is sound: poot's CPU oracle, run over the same graph and store, reproduces gguf-py's value for
    /// every element of every row, so a device that differs from gguf-py differs from a decode already proven
    /// right here (and `poot-load`'s `flash_next_dequant` proves the oracle's decode itself on the same blocks).
    /// Mutation: take `expected` of the wrong block (`block * values + column` -> `column * blocks + block`);
    /// this row goes red.
    #[test]
    fn the_cpu_oracle_decodes_every_reference_row_to_gguf_py() {
        let rows = reference_block_rows();
        // Q8_0, Q4_K, Q5_K and Q6_K three runs each; IQ2_XS and IQ3_XXS four (a last-expert run too); IQ4_NL seven.
        assert_eq!(rows.len(), 27);
        for row in &rows {
            let mut oracle = oracle_for(&row.fixture.graph, &row.fixture.store);
            for (column, step) in row.fixture.steps.iter().enumerate() {
                let out = oracle(step);
                for (block, &got) in out.as_f32().unwrap().iter().enumerate() {
                    let want = row.expected[block * row.values + column];
                    assert_eq!(
                        (got + 0.0).to_bits(),
                        (want + 0.0).to_bits(),
                        "{} ({}) block {block} column {column}: oracle {got}, gguf-py {want}",
                        row.tensor,
                        row.fixture.name
                    );
                }
            }
        }
    }
}

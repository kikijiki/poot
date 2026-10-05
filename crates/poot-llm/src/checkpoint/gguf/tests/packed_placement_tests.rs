//! Card 545a: the GGUF loader keeps quantized tensors packed, and every loader row op
//! (llama's q/k un-permute, phi3's fused q|k|v and gate|up slices, the MoE gate||up concatenation and
//! gpt-oss's gate/up interleave) moves whole stored rows. Each test writes a GGUF whose tensors are
//! random Q8_0 payloads, runs `gguf_weights`, and checks every placed owner's decoded rows against the
//! stored rows at indices derived here from the HF layout, independently of the loader: a wrong slice
//! offset, permutation or concatenation order turns it red.

use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::packed_source::PackedSourceName;
use poot_graph_plan::{PackedLayout, WeightFormats};
use poot_load::gguf::{GgufIndex, GgufValue, IdentityNames, read_gguf, write_gguf};
use poot_quant::format::WeightFormat;
use poot_quant::{PackedPayload, SourceRole};

use crate::checkpoint::gguf::arch_config::gguf_weights;
use crate::checkpoint::gguf::gguf_config;

const H: usize = 32;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HD: usize = H / HEADS;
const Q_DIM: usize = HEADS * HD;
const KV_DIM: usize = KV_HEADS * HD;
const INTER: usize = 64;
const EXPERTS: usize = 2;
const VOCAB: usize = 8;
const F32: u32 = 0;
const Q8_0: u32 = 8;

/// A GGUF under construction: every weight tensor a random Q8_0 payload (kept for the expected
/// rows), every norm/bias F32.
struct Fixture {
    /// The format every packed tensor is stored in, and its GGUF tensor type.
    format: (WeightFormat, u32),
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
    stored: HashMap<String, PackedPayload>,
}

impl Default for Fixture {
    fn default() -> Self {
        Self::stored_as(WeightFormat::Q8_0, Q8_0)
    }
}

impl Fixture {
    fn stored_as(format: WeightFormat, ggml_type: u32) -> Self {
        Self {
            format: (format, ggml_type),
            tensors: Vec::new(),
            stored: HashMap::new(),
        }
    }

    /// A stored `[out, k]` weight in the fixture's format (`[E, out, k]` with `experts`).
    fn packed(&mut self, name: &str, out: usize, k: usize, experts: Option<usize>) {
        let seed = self.tensors.len() as u64;
        let mut bytes = Vec::new();
        for e in 0..experts.unwrap_or(1) {
            let payload = poot_test_util::packed::random_payload(
                self.format.0,
                [out, k],
                seed * 16 + e as u64,
            );
            bytes.extend_from_slice(payload.bytes(SourceRole::Blocks));
            let key = match experts {
                Some(_) => format!("{name}.{e}"),
                None => name.to_string(),
            };
            self.stored.insert(key, payload);
        }
        let dims = match experts {
            Some(e) => vec![k as u64, out as u64, e as u64],
            None => vec![k as u64, out as u64],
        };
        self.tensors
            .push((name.to_string(), dims, self.format.1, bytes));
    }

    fn dense(&mut self, name: &str, dims: Vec<u64>) {
        let n: u64 = dims.iter().product();
        let bytes = (0..n)
            .flat_map(|i| (1.0 + i as f32 * 1e-3).to_le_bytes())
            .collect();
        self.tensors.push((name.to_string(), dims, F32, bytes));
    }

    fn attention(&mut self, fused_qkv: bool) {
        self.dense("blk.0.attn_norm.weight", vec![H as u64]);
        self.dense("blk.0.ffn_norm.weight", vec![H as u64]);
        if fused_qkv {
            self.packed("blk.0.attn_qkv.weight", Q_DIM + 2 * KV_DIM, H, None);
        } else {
            self.packed("blk.0.attn_q.weight", Q_DIM, H, None);
            self.packed("blk.0.attn_k.weight", KV_DIM, H, None);
            self.packed("blk.0.attn_v.weight", KV_DIM, H, None);
        }
        self.packed("blk.0.attn_output.weight", H, Q_DIM, None);
    }

    fn model(&mut self, tied: bool) {
        self.packed("token_embd.weight", VOCAB, H, None);
        self.dense("output_norm.weight", vec![H as u64]);
        if !tied {
            self.packed("output.weight", VOCAB, H, None);
        }
    }

    /// `gguf_weights` over the written file, for `arch` with `extra` metadata.
    fn load(
        &self,
        arch: &str,
        extra: Vec<(String, GgufValue)>,
    ) -> (HashMap<String, Value>, WeightFormats) {
        let mut kvs: Vec<(String, GgufValue)> = vec![
            ("general.architecture".into(), GgufValue::Str(arch.into())),
            (format!("{arch}.embedding_length"), GgufValue::U32(H as u32)),
            (format!("{arch}.block_count"), GgufValue::U32(1)),
            (
                format!("{arch}.attention.head_count"),
                GgufValue::U32(HEADS as u32),
            ),
            (
                format!("{arch}.attention.head_count_kv"),
                GgufValue::U32(KV_HEADS as u32),
            ),
            (
                format!("{arch}.feed_forward_length"),
                GgufValue::U32(INTER as u32),
            ),
            (format!("{arch}.context_length"), GgufValue::U32(16)),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array(
                    (0..VOCAB)
                        .map(|t| GgufValue::Str(format!("t{t}")))
                        .collect(),
                ),
            ),
        ];
        kvs.extend(extra);
        let kvs: Vec<(&str, GgufValue)> =
            kvs.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = self
            .tensors
            .iter()
            .map(|(n, d, t, b)| (n.as_str(), d.clone(), *t, b.clone()))
            .collect();
        let bytes = write_gguf(&kvs, &tensors);
        let g = GgufIndex::from_bytes(&bytes).expect("parse the written gguf");
        let store = read_gguf(&g, bytes.as_slice(), &IdentityNames).expect("read the written gguf");
        let cfg = gguf_config(&g, arch, false).expect("gguf config");
        gguf_weights(&g, &store, &cfg, arch).expect("gguf_weights")
    }

    /// Stored row `row` of `name`, decoded.
    fn stored_row(&self, name: &str, row: usize) -> Vec<u32> {
        decode_row(&self.stored[name], row)
    }
}

fn decode_row(payload: &PackedPayload, row: usize) -> Vec<u32> {
    let mut out = vec![0.0f32; payload.weight().shape()[1]];
    payload.decode_row(row, &mut out).unwrap();
    out.into_iter().map(f32::to_bits).collect()
}

/// The rows of the owner a placed constant's `linear_id` names, decoded.
fn placed_rows(weights: &HashMap<String, Value>, linear_id: &str) -> Vec<Vec<u32>> {
    let carrier = PackedSourceName::new(linear_id, SourceRole::Blocks);
    let Some(Value::Packed(component)) = weights.get(carrier.as_ref()) else {
        panic!("{carrier} is not a packed carrier");
    };
    let owner = component.owner();
    (0..owner.weight().shape()[0])
        .map(|r| decode_row(owner, r))
        .collect()
}

/// Assert `name` is recorded packed with `layout` and its (single or per-expert) owners' rows are
/// `expected[owner]`.
fn assert_placed(
    weights: &HashMap<String, Value>,
    formats: &WeightFormats,
    name: &str,
    layout: PackedLayout,
    expected: &[Vec<Vec<u32>>],
) {
    let packed = formats
        .get(name)
        .unwrap_or_else(|| panic!("{name} should be recorded packed"));
    assert_eq!(packed.layout, layout, "{name}");
    assert_eq!(packed.linear_ids.len(), expected.len(), "{name} owners");
    for (owner, (linear_id, rows)) in packed.linear_ids.iter().zip(expected).enumerate() {
        assert_eq!(
            &placed_rows(weights, linear_id),
            rows,
            "{name} owner {owner}"
        );
    }
}

/// llama.cpp's rope permute, re-derived as "HF row r is stored row s": `LlamaModel.permute`
/// reshapes each head `(2, hd/2)` and swaps to `(hd/2, 2)`, so HF row `h*hd + t*half + j` is stored
/// at `h*hd + 2*j + t`.
fn llama_stored_row(hf_row: usize) -> usize {
    let (h, rest) = (hf_row / HD, hf_row % HD);
    let (t, j) = (rest / (HD / 2), rest % (HD / 2));
    h * HD + 2 * j + t
}

/// llama's q/k un-permute on packed rows, the packed tied lm_head, and the plain projections.
/// Mutation: an un-permute that keeps the stored order (or swaps the pair halves) turns the q/k
/// asserts red; a transposed layout for the embedding turns the table assert red.
#[test]
fn llama_q_and_k_are_unpermuted_row_by_row_on_the_stored_blocks() {
    let mut f = Fixture::default();
    f.model(true);
    f.attention(false);
    f.packed("blk.0.ffn_gate.weight", INTER, H, None);
    f.packed("blk.0.ffn_up.weight", INTER, H, None);
    f.packed("blk.0.ffn_down.weight", H, INTER, None);
    let (w, formats) = f.load("llama", vec![]);
    let hf = |name: &str, rows: usize| -> Vec<Vec<u32>> {
        (0..rows)
            .map(|r| f.stored_row(name, llama_stored_row(r)))
            .collect()
    };
    let as_stored = |name: &str, rows: usize| -> Vec<Vec<u32>> {
        (0..rows).map(|r| f.stored_row(name, r)).collect()
    };
    let l = |s: &str| format!("model.layers.0.{s}");
    assert_placed(
        &w,
        &formats,
        &l("self_attn.q_proj.weight"),
        PackedLayout::Columns,
        &[hf("blk.0.attn_q.weight", Q_DIM)],
    );
    assert_placed(
        &w,
        &formats,
        &l("self_attn.k_proj.weight"),
        PackedLayout::Columns,
        &[hf("blk.0.attn_k.weight", KV_DIM)],
    );
    assert_placed(
        &w,
        &formats,
        &l("self_attn.v_proj.weight"),
        PackedLayout::Columns,
        &[as_stored("blk.0.attn_v.weight", KV_DIM)],
    );
    assert_placed(
        &w,
        &formats,
        &l("mlp.down_proj.weight"),
        PackedLayout::Columns,
        &[as_stored("blk.0.ffn_down.weight", H)],
    );
    let table = as_stored("token_embd.weight", VOCAB);
    assert_placed(
        &w,
        &formats,
        "model.embed_tokens.weight",
        PackedLayout::Rows,
        std::slice::from_ref(&table),
    );
    assert_placed(
        &w,
        &formats,
        "lm_head.weight",
        PackedLayout::Columns,
        &[table],
    );
}

/// A tied lm_head is the embedding's own stored payload, shared: no byte copy.
#[test]
fn a_tied_lm_head_shares_the_embedding_payload() {
    let mut f = Fixture::default();
    f.model(true);
    f.attention(false);
    f.packed("blk.0.ffn_gate.weight", INTER, H, None);
    f.packed("blk.0.ffn_up.weight", INTER, H, None);
    f.packed("blk.0.ffn_down.weight", H, INTER, None);
    let (w, _) = f.load("qwen2", vec![]);
    let component =
        |linear_id: &str| match &w[PackedSourceName::new(linear_id, SourceRole::Blocks).as_ref()] {
            Value::Packed(component) => component.clone(),
            other => panic!("{linear_id}: {other:?}"),
        };
    assert!(
        component("model.embed_tokens").same_owner(&component("lm_head")),
        "the tied lm_head must reuse the embedding's payload"
    );
}

/// phi3's fused q|k|v (HF order, no rope permute) and gate|up slices. Mutation: swapping the k and
/// v offsets, or reading gate from the second half, turns this red.
#[test]
fn phi3_fused_projections_split_at_their_hf_offsets() {
    let mut f = Fixture::default();
    f.model(false);
    f.attention(true);
    f.packed("blk.0.ffn_up.weight", 2 * INTER, H, None);
    f.packed("blk.0.ffn_down.weight", H, INTER, None);
    let (w, formats) = f.load(
        "phi3",
        vec![(
            "phi3.rope.dimension_count".into(),
            GgufValue::U32(HD as u32),
        )],
    );
    let rows = |name: &str, lo: usize, hi: usize| -> Vec<Vec<u32>> {
        (lo..hi).map(|r| f.stored_row(name, r)).collect()
    };
    let l = |s: &str| format!("model.layers.0.{s}");
    let qkv = "blk.0.attn_qkv.weight";
    assert_placed(
        &w,
        &formats,
        &l("self_attn.q_proj.weight"),
        PackedLayout::Columns,
        &[rows(qkv, 0, Q_DIM)],
    );
    assert_placed(
        &w,
        &formats,
        &l("self_attn.k_proj.weight"),
        PackedLayout::Columns,
        &[rows(qkv, Q_DIM, Q_DIM + KV_DIM)],
    );
    assert_placed(
        &w,
        &formats,
        &l("self_attn.v_proj.weight"),
        PackedLayout::Columns,
        &[rows(qkv, Q_DIM + KV_DIM, Q_DIM + 2 * KV_DIM)],
    );
    let gu = "blk.0.ffn_up.weight";
    assert_placed(
        &w,
        &formats,
        &l("mlp.gate_proj.weight"),
        PackedLayout::Columns,
        &[rows(gu, 0, INTER)],
    );
    assert_placed(
        &w,
        &formats,
        &l("mlp.up_proj.weight"),
        PackedLayout::Columns,
        &[rows(gu, INTER, 2 * INTER)],
    );
}

fn moe_fixture(arch: &str) -> Fixture {
    let mut f = Fixture::default();
    f.model(false);
    f.attention(false);
    f.dense("blk.0.ffn_gate_inp.weight", vec![H as u64, EXPERTS as u64]);
    f.packed("blk.0.ffn_gate_exps.weight", INTER, H, Some(EXPERTS));
    f.packed("blk.0.ffn_up_exps.weight", INTER, H, Some(EXPERTS));
    f.packed("blk.0.ffn_down_exps.weight", H, INTER, Some(EXPERTS));
    if arch == "olmoe" {
        f.dense("blk.0.attn_q_norm.weight", vec![Q_DIM as u64]);
        f.dense("blk.0.attn_k_norm.weight", vec![KV_DIM as u64]);
    }
    if arch == "gpt-oss" {
        for bias in ["attn_q.bias", "attn_k.bias", "attn_v.bias"] {
            let n = if bias == "attn_q.bias" { Q_DIM } else { KV_DIM };
            f.dense(&format!("blk.0.{bias}"), vec![n as u64]);
        }
        f.dense("blk.0.attn_output.bias", vec![H as u64]);
        f.dense("blk.0.attn_sinks.weight", vec![HEADS as u64]);
        f.dense("blk.0.post_attention_norm.weight", vec![H as u64]);
        f.dense("blk.0.ffn_gate_inp.bias", vec![EXPERTS as u64]);
        f.dense(
            "blk.0.ffn_gate_exps.bias",
            vec![INTER as u64, EXPERTS as u64],
        );
        f.dense("blk.0.ffn_up_exps.bias", vec![INTER as u64, EXPERTS as u64]);
        f.dense("blk.0.ffn_down_exps.bias", vec![H as u64, EXPERTS as u64]);
    }
    f
}

/// Each expert's fused gate_up owner is its gate rows then its up rows (qwen3moe/olmoe/granitemoe/
/// Mixtral), and down stays per expert. Mutation: concatenating up before gate, or pairing expert
/// 0's gate with expert 1's up, turns this red.
#[test]
fn moe_gate_up_concatenates_each_experts_gate_then_up_rows() {
    let f = moe_fixture("olmoe");
    let (w, formats) = f.load(
        "olmoe",
        vec![
            ("olmoe.expert_count".into(), GgufValue::U32(EXPERTS as u32)),
            ("olmoe.expert_used_count".into(), GgufValue::U32(1)),
        ],
    );
    let expert_rows = |name: &str, e: usize, rows: usize| -> Vec<Vec<u32>> {
        (0..rows)
            .map(|r| f.stored_row(&format!("{name}.{e}"), r))
            .collect()
    };
    let gate_up: Vec<Vec<Vec<u32>>> = (0..EXPERTS)
        .map(|e| {
            let mut rows = expert_rows("blk.0.ffn_gate_exps.weight", e, INTER);
            rows.extend(expert_rows("blk.0.ffn_up_exps.weight", e, INTER));
            rows
        })
        .collect();
    let down: Vec<Vec<Vec<u32>>> = (0..EXPERTS)
        .map(|e| expert_rows("blk.0.ffn_down_exps.weight", e, H))
        .collect();
    let l = |s: &str| format!("model.layers.0.{s}");
    assert_placed(
        &w,
        &formats,
        &l("mlp.experts.gate_up_proj.weight"),
        PackedLayout::StackedColumns,
        &gate_up,
    );
    assert_placed(
        &w,
        &formats,
        &l("mlp.experts.down_proj.weight"),
        PackedLayout::StackedColumns,
        &down,
    );
}

/// gpt-oss re-interleaves each expert's gate/up rows (even rows gate, odd rows up) to its native
/// fused layout. Mutation: concatenating instead of interleaving turns this red.
#[test]
fn gptoss_gate_up_interleaves_each_experts_gate_and_up_rows() {
    let f = moe_fixture("gpt-oss");
    let (w, formats) = f.load(
        "gpt-oss",
        vec![
            (
                "gpt-oss.expert_count".into(),
                GgufValue::U32(EXPERTS as u32),
            ),
            ("gpt-oss.expert_used_count".into(), GgufValue::U32(1)),
        ],
    );
    let gate_up: Vec<Vec<Vec<u32>>> = (0..EXPERTS)
        .map(|e| {
            (0..INTER)
                .flat_map(|r| {
                    [
                        f.stored_row(&format!("blk.0.ffn_gate_exps.weight.{e}"), r),
                        f.stored_row(&format!("blk.0.ffn_up_exps.weight.{e}"), r),
                    ]
                })
                .collect()
        })
        .collect();
    assert_placed(
        &w,
        &formats,
        "model.layers.0.mlp.experts.gate_up_proj",
        PackedLayout::StackedColumns,
        &gate_up,
    );
}

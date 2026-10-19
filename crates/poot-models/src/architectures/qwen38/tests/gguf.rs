//! Qwen3.8-Flash-Next from a GGUF (arch `qwen4exp`, the unsloth quants): the typed config read from
//! metadata, the tensor-name table, and the stored conventions, against the real UD-Q2_K_XL
//! directory and against the HF fixture converted the way llama.cpp's converter writes it.
//!
//! - The real directory (`tests/data/flash_next_ud_q2_k_xl_tensors.tsv`: all 1224 tensors of the
//!   three shards with their ggml dims, types and offsets, cut from the published files) and the real
//!   `qwen4exp.*` metadata are fed through the production entries: [`Qwen4ExpParams::from_raw`],
//!   `weights_of`, `GgufIndex::parse_shards` and `gguf_plan` (over sparse files of the shards' real
//!   sizes, so no tensor byte is read). Every tensor is mapped by the name table or refused typed
//!   by type; the shapes the table implies are the real ones.
//! - The fixture converter ([`convert`]) is the converter's `modify_tensors` written again over the
//!   fixture: norm scales folded to `1 + w` (all but the GDN's gated norm), the value heads of every
//!   GDN tensor reordered to tiled order, `A_log` stored as `-exp(A_log)`, the indexer's fused
//!   projection split, the convs squeezed, the n-gram table renamed to the model-level tensor, and the
//!   hash constants moved to metadata. The converted checkpoint, loaded through the GGUF path, must
//!   produce the HF path's logits and carried state, which the independent f64 reference matches.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1189 report):
//! GQ-1 the GGUF layout reads the norms as `1 + w` ([`NormKind::RmsPlusOne`]); GQ-2 it reads the
//! GDN's value heads grouped; GQ-3 it applies `exp` to `ssm_a` (log decay); GQ-4 the indexer's key
//! rows are mapped from the query tensor; GQ-5 the stored PLE multipliers are replaced by ones; GQ-6
//! the layer kinds ignore `compress_ratios`; GQ-7 the name table maps `ffn_up_exps` to the gate
//! stack; GQ-8 the PLE table name loses its `.weight`; GQ-9 the interval cross-check is skipped;
//! GQ-10 (poot-load) the per-expert stride is one block short.

use std::collections::{BTreeMap, BTreeSet};

use poot_load::gguf::{
    GgufIndex, GgufTensorInfo, GgufValue, IdentityNames, TensorNameMap, gguf_plan, read_gguf,
    write_gguf,
};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_tensor::DType;

use super::super::config::{Layout, expert_layout, weights_of};
use super::*;
use crate::components::norm::NormKind;
use crate::components::ple::{NgramHash, PleParams, PleSpec};
use crate::model::ConfigReason;

// ---------------------------------------------------------------------------------------------
// The real checkpoint's directory and metadata.
// ---------------------------------------------------------------------------------------------

const DIRECTORY: &str = include_str!("../../../../tests/data/flash_next_ud_q2_k_xl_tensors.tsv");

/// One tensor of the real directory.
#[derive(Clone, Debug)]
struct Listed {
    name: String,
    /// ggml `ne`: fastest-varying first.
    ne: Vec<u64>,
    ggml_type: u32,
    shard: usize,
    offset: u64,
}

impl Listed {
    /// The logical row-major shape.
    fn shape(&self) -> Vec<usize> {
        self.ne.iter().rev().map(|&d| d as usize).collect()
    }
}

fn listed() -> Vec<Listed> {
    DIRECTORY
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|line| {
            let c: Vec<&str> = line.split('\t').collect();
            Listed {
                name: c[0].to_string(),
                ne: c[1].split(',').map(|d| d.parse().unwrap()).collect(),
                ggml_type: c[2].parse().unwrap(),
                shard: c[3].parse().unwrap(),
                offset: c[4].parse().unwrap(),
            }
        })
        .collect()
}

/// The data bytes of shard `shard` (the file size minus its header), from the directory's comments.
fn shard_data_bytes(shard: usize) -> u64 {
    DIRECTORY
        .lines()
        .find_map(|l| {
            let c: Vec<&str> = l.strip_prefix('#')?.split('\t').collect();
            (c[0] == "shard" && c[1].parse::<usize>().ok()? == shard).then(|| c[2].parse().unwrap())
        })
        .expect("the shard's size is listed")
}

type Kvs = Vec<(&'static str, GgufValue)>;

fn u32s(values: impl IntoIterator<Item = u32>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufValue::U32).collect())
}

fn u64s(values: impl IntoIterator<Item = u64>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufValue::U64).collect())
}

/// The `qwen4exp.*` and tokenizer metadata of the real first shard (values as the published file
/// states them; the 248,320 token strings are stood in by one string each).
fn real_kvs() -> Kvs {
    let n = |v: u32| GgufValue::U32(v);
    vec![
        ("general.architecture", GgufValue::Str("qwen4exp".into())),
        ("qwen4exp.block_count", n(48)),
        ("qwen4exp.context_length", n(262_144)),
        ("qwen4exp.embedding_length", n(2560)),
        ("qwen4exp.attention.head_count", n(24)),
        ("qwen4exp.attention.head_count_kv", n(2)),
        ("qwen4exp.rope.freq_base", GgufValue::F32(10_000_000.0)),
        (
            "qwen4exp.attention.layer_norm_rms_epsilon",
            GgufValue::F32(1e-6),
        ),
        ("qwen4exp.expert_count", n(512)),
        ("qwen4exp.expert_used_count", n(10)),
        ("qwen4exp.attention.key_length", n(256)),
        ("qwen4exp.attention.value_length", n(256)),
        ("qwen4exp.expert_feed_forward_length", n(640)),
        ("qwen4exp.expert_shared_feed_forward_length", n(640)),
        ("qwen4exp.ssm.conv_kernel", n(4)),
        ("qwen4exp.ssm.state_size", n(128)),
        ("qwen4exp.ssm.group_count", n(16)),
        ("qwen4exp.ssm.time_step_rank", n(48)),
        ("qwen4exp.ssm.inner_size", n(6144)),
        ("qwen4exp.full_attention_interval", n(4)),
        ("qwen4exp.rope.dimension_count", n(64)),
        ("qwen4exp.hyper_connection.count", n(4)),
        ("qwen4exp.hyper_connection.low_rank", n(320)),
        ("qwen4exp.attention.indexer.head_count", n(4)),
        ("qwen4exp.attention.indexer.key_length", n(128)),
        ("qwen4exp.attention.indexer.top_k", n(2048)),
        (
            "qwen4exp.attention.compress_ratios",
            u32s((0..48).map(|l| if l % 4 == 3 { 4 } else { 0 })),
        ),
        ("qwen4exp.ple.layers", u32s([1])),
        ("qwen4exp.ple.ngram_size", n(3)),
        ("qwen4exp.ple.heads_per_ngram", n(8)),
        ("qwen4exp.ple.conv_kernel", n(4)),
        ("qwen4exp.ple.eos_token_id", n(248_044)),
        ("qwen4exp.embedding_length_per_layer_input", n(160)),
        (
            "qwen4exp.ple.layer_multipliers",
            u64s([23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]),
        ),
        (
            "qwen4exp.ple.head_offsets",
            u64s([
                0,
                20_000_003,
                40_000_026,
                60_000_059,
                80_000_106,
                100_000_165,
                120_000_228,
                140_000_297,
                160_000_374,
                180_000_455,
                200_000_548,
                220_000_655,
                240_000_802,
                260_000_955,
                280_001_114,
                300_001_275,
            ]),
        ),
        (
            "qwen4exp.ple.head_vocab_sizes",
            u64s([
                20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
                20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
                20_000_161, 20_000_171,
            ]),
        ),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![GgufValue::Str("t".into()); 248_320]),
        ),
        ("tokenizer.ggml.eos_token_id", n(248_046)),
        ("tokenizer.ggml.bos_token_id", n(248_044)),
    ]
}

/// An index with `kvs` as its metadata and `tensors` as its directory (no tensor bytes).
fn index_of(kvs: &Kvs, tensors: impl IntoIterator<Item = Listed>) -> GgufIndex {
    let mut index = GgufIndex::parse(&StoredBytes::from(write_gguf(kvs, &[]))).unwrap();
    for t in tensors {
        index.tensors.insert(
            t.name,
            GgufTensorInfo {
                dims: t.ne,
                ggml_type: t.ggml_type,
                offset: t.offset,
                shard: t.shard,
            },
        );
    }
    index
}

fn real_index() -> GgufIndex {
    index_of(&real_kvs(), listed())
}

fn params_of(index: &GgufIndex) -> Result<Qwen4ExpParams, ModelError> {
    Qwen4ExpParams::from_raw(&RawConfig::Gguf(index))
}

// ---------------------------------------------------------------------------------------------
// What the name table implies: every tensor, with its logical shape, from the metadata alone.
// ---------------------------------------------------------------------------------------------

/// The facts the shape table needs, read from the metadata by key (not through the family's
/// reader, so a reader that misreads a key disagrees with this).
struct Facts<'a>(&'a Kvs);

impl Facts<'_> {
    fn n(&self, key: &str) -> usize {
        match self.0.iter().find(|(k, _)| *k == key) {
            Some((_, v)) => v.as_u64().unwrap_or_else(|| panic!("{key} is a number")) as usize,
            None => panic!("{key} is absent"),
        }
    }

    fn list(&self, key: &str) -> Vec<u64> {
        let (_, v) = self.0.iter().find(|(k, _)| *k == key).expect(key);
        v.as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect()
    }
}

/// Every tensor a checkpoint with `kvs` holds, by name, with its logical row-major shape: the name
/// table of the llama.cpp `qwen4exp` loader, written from `llm_build`'s `create_tensor` calls.
fn expected_shapes(kvs: &Kvs, ple_rows: usize) -> BTreeMap<String, Vec<usize>> {
    let f = Facts(kvs);
    let k = |s: &str| format!("qwen4exp.{s}");
    let (w, layers, vocab) = (f.n(&k("embedding_length")), f.n(&k("block_count")), {
        // The vocabulary is the token list's length.
        match &kvs
            .iter()
            .find(|(k, _)| *k == "tokenizer.ggml.tokens")
            .unwrap()
            .1
        {
            GgufValue::Array(tokens) => tokens.len(),
            _ => unreachable!(),
        }
    });
    let (heads, kv, hd) = (
        f.n(&k("attention.head_count")),
        f.n(&k("attention.head_count_kv")),
        f.n(&k("attention.key_length")),
    );
    let (hk, hv, d, kernel) = (
        f.n(&k("ssm.group_count")),
        f.n(&k("ssm.time_step_rank")),
        f.n(&k("ssm.state_size")),
        f.n(&k("ssm.conv_kernel")),
    );
    let (conv, value_dim) = (2 * hk * d + hv * d, hv * d);
    let (experts, inter, shared) = (
        f.n(&k("expert_count")),
        f.n(&k("expert_feed_forward_length")),
        f.n(&k("expert_shared_feed_forward_length")),
    );
    let (hc, rank) = (
        f.n(&k("hyper_connection.count")),
        f.n(&k("hyper_connection.low_rank")),
    );
    let (hi, di) = (
        f.n(&k("attention.indexer.head_count")),
        f.n(&k("attention.indexer.key_length")),
    );
    let stream = hc * w;
    let ratios = f.list(&k("attention.compress_ratios"));
    let ple_layers = f.list(&k("ple.layers"));

    let mut shapes = BTreeMap::new();
    let mut put = |name: String, shape: Vec<usize>| {
        assert!(shapes.insert(name.clone(), shape).is_none(), "{name} twice");
    };
    put("token_embd.weight".into(), vec![vocab, w]);
    put("output.weight".into(), vec![vocab, w]);
    put("output_hc_norm.weight".into(), vec![stream]);
    put("output_hc_down.weight".into(), vec![rank, stream]);
    put("output_hc_up.weight".into(), vec![stream, rank]);
    if !ple_layers.is_empty() {
        let row = f.n(&k("embedding_length_per_layer_input"));
        put("per_layer_token_embd.weight".into(), vec![ple_rows, row]);
    }
    assert_eq!(ratios.len(), layers);
    for (l, ratio) in ratios.iter().enumerate() {
        let b = |s: &str| format!("blk.{l}.{s}");
        for site in ["attn", "ffn"] {
            put(b(&format!("hc_{site}_norm.weight")), vec![stream]);
            put(b(&format!("hc_{site}_down.weight")), vec![rank, stream]);
            put(b(&format!("hc_{site}_up.weight")), vec![stream, rank]);
            put(b(&format!("hc_{site}_inject.weight")), vec![hc, stream]);
        }
        if *ratio != 0 {
            put(b("attn_q.weight"), vec![2 * heads * hd, w]);
            put(b("attn_k.weight"), vec![kv * hd, w]);
            put(b("attn_v.weight"), vec![kv * hd, w]);
            put(b("attn_output.weight"), vec![w, heads * hd]);
            put(b("attn_q_norm.weight"), vec![hd]);
            put(b("attn_k_norm.weight"), vec![hd]);
            put(b("indexer.q_proj.weight"), vec![hi * di, w]);
            put(b("indexer.k_proj.weight"), vec![di, w]);
            put(b("indexer.q_norm.weight"), vec![di]);
            put(b("indexer.k_norm.weight"), vec![di]);
        } else {
            put(b("attn_qkv.weight"), vec![conv, w]);
            put(b("attn_gate.weight"), vec![value_dim, w]);
            put(b("ssm_beta.weight"), vec![hv, w]);
            put(b("ssm_alpha.weight"), vec![hv, w]);
            put(b("ssm_conv1d.weight"), vec![conv, kernel]);
            put(b("ssm_dt.bias"), vec![hv]);
            put(b("ssm_a"), vec![hv]);
            put(b("ssm_norm.weight"), vec![d]);
            put(b("ssm_out.weight"), vec![w, value_dim]);
        }
        if ple_layers.contains(&(l as u64)) {
            let embed = f.n(&k("embedding_length_per_layer_input"))
                * (f.n(&k("ple.ngram_size")) - 1)
                * f.n(&k("ple.heads_per_ngram"));
            put(
                b("ple_conv1d.weight"),
                vec![stream, f.n(&k("ple.conv_kernel"))],
            );
            put(b("ple_key.weight"), vec![stream, embed]);
            put(b("ple_value.weight"), vec![w, embed]);
            for norm in ["key", "query", "conv"] {
                put(b(&format!("ple_norm_{norm}.weight")), vec![stream]);
            }
        }
        put(b("ffn_gate_inp.weight"), vec![experts, w]);
        put(b("ffn_gate_inp_shexp.weight"), vec![w]);
        put(b("ffn_gate_shexp.weight"), vec![shared, w]);
        put(b("ffn_up_shexp.weight"), vec![shared, w]);
        put(b("ffn_down_shexp.weight"), vec![w, shared]);
        put(b("ffn_gate_exps.weight"), vec![experts, inter, w]);
        put(b("ffn_up_exps.weight"), vec![experts, inter, w]);
        put(b("ffn_down_exps.weight"), vec![experts, w, inter]);
    }
    shapes
}

/// SC-001 (the table): the name table and the shapes it implies are exactly the real checkpoint's
/// directory, tensor for tensor: no tensor the table lacks, none it names that the file does not
/// hold, and the logical shape of every one equal. Mutation GQ-8 (a renamed PLE table) and GQ-7
/// (a dropped expert stack) are red here and in [`every_real_tensor_is_mapped_by_the_name_table`].
#[test]
fn the_shape_table_is_the_real_directory() {
    let real = listed();
    assert_eq!(real.len(), 1224, "split.tensors.count of the real set");
    let rows = real
        .iter()
        .find(|t| t.name == "per_layer_token_embd.weight")
        .unwrap()
        .shape()[0];
    let expected = expected_shapes(&real_kvs(), rows);
    let listed: BTreeMap<String, Vec<usize>> =
        real.iter().map(|t| (t.name.clone(), t.shape())).collect();
    let missing: Vec<_> = listed
        .keys()
        .filter(|k| !expected.contains_key(*k))
        .collect();
    let extra: Vec<_> = expected
        .keys()
        .filter(|k| !listed.contains_key(*k))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "tensors the table lacks: {missing:?}; tensors it names that the file lacks: {extra:?}"
    );
    for (name, shape) in &expected {
        assert_eq!(shape, &listed[name], "{name}");
    }
}

// ---------------------------------------------------------------------------------------------
// The typed config from the real metadata.
// ---------------------------------------------------------------------------------------------

/// SC-002: the real metadata reads into the config the checkpoint is: 48 layers whose full-attention
/// layers are those the compress ratios name, the indexer's block width four, 512 experts of which
/// ten route, four hyper-connection streams, the one PLE layer (zero-indexed layer 1) with its
/// 320,001,536-row table, and the stop and start ids. Mutation GQ-6: the kinds ignore the ratios.
#[test]
fn the_real_metadata_reads_into_the_typed_config() {
    let index = real_index();
    let p = params_of(&index).expect("the real metadata is a valid config");
    assert_eq!(p.layout, Layout::Gguf);
    assert_eq!(
        (p.vocab, p.width, p.max_positions),
        (248_320, 2560, 262_144)
    );
    assert_eq!(p.kinds.len(), 48);
    for (l, kind) in p.kinds.iter().enumerate() {
        let want = if l % 4 == 3 {
            LayerKind::FullAttention
        } else {
            LayerKind::LinearAttention
        };
        assert_eq!(*kind, want, "layer {l}");
    }
    assert_eq!(p.ratio.get(), 4);
    assert_eq!(p.hyper.streams(), 4);
    assert_eq!(p.moe.experts(), 512);
    assert_eq!(p.eos, BTreeSet::from([248_046]));
    assert_eq!(p.bos, Some(248_044));
    assert!(!p.head_required, "a GGUF's head may be tied");

    let ple: Vec<usize> = (0..48).filter(|&l| p.ple[l].is_some()).collect();
    assert_eq!(ple, [1]);
    let ple = p.ple[1].as_ref().unwrap();
    assert_eq!(
        (ple.heads(), ple.head_dim(), ple.rows()),
        (16, 160, 320_001_536)
    );
    let spec = ple.spec();
    assert_eq!(
        (spec.embed_dim, spec.ngram_size, spec.conv_kernel, spec.eos),
        (2560, 3, 4, 248_044)
    );
    assert_eq!(
        spec.norm.kind(),
        NormKind::Rms,
        "the converter folded the one in"
    );
}

/// SC-003: the hash constants the GGUF stores are the ones the reference model derives from the HF
/// config (`ngram_vocab_size_base` 20,000,000, divisible by 128, seed 1234): reading them as stored
/// is the same hash, and the table's rows are the derivation's padded rows.
#[test]
fn the_stored_hash_constants_equal_the_derivation() {
    let index = real_index();
    let p = params_of(&index).unwrap();
    let stored = p.ple[1].as_ref().unwrap().spec().clone();
    let derived = PleSpec {
        hash: NgramHash::Derived {
            vocab_base: 20_000_000,
            vocab_divisible_by: 128,
            seed: 1234,
        },
        ..stored.clone()
    };
    let (want, got) = (ple_oracle::tables(&derived), ple_oracle::tables(&stored));
    assert_eq!(got.multipliers, want.multipliers, "multipliers");
    assert_eq!(got.head_offset, want.head_offset, "head offsets");
    assert_eq!(got.head_vocab, want.head_vocab, "head vocab sizes");
    assert_eq!(got.rows, want.rows, "rows");
    // And the component's own tables agree with the oracle's for the stored hash.
    let component = PleParams::new(stored).unwrap();
    assert_eq!(component.rows(), want.rows);
}

/// The real set refuses nothing: every ggml type the 1224 tensors of UD-Q2_K_XL use has a format,
/// the IQ2_XS (17) and IQ3_XXS (18) expert gate and up stacks of the 48 layers included.
/// Mutation: map ggml type 18 to `None` in `weight_format_of`; the stacks of layer 2 are listed.
#[test]
fn every_type_of_the_real_set_has_a_format() {
    let refused: BTreeSet<String> = listed()
        .into_iter()
        .filter(|t| poot_load::gguf::weight_format_of(t.ggml_type).is_none())
        .map(|t| t.name)
        .collect();
    assert_eq!(refused, BTreeSet::new());
    let lattice = listed()
        .into_iter()
        .filter(|t| matches!(t.ggml_type, 17 | 18))
        .count();
    assert_eq!(lattice, 96, "the gate and up stack of each of 48 layers");
}

/// The name table maps every tensor the real set holds: a store keyed as the loader keys it (a
/// stacked tensor is one owner per expert) is mapped with no tensor missing, and no stored tensor is
/// left unread. Mutations GQ-7, GQ-8.
#[test]
fn every_real_tensor_is_mapped_by_the_name_table() {
    let index = real_index();
    let p = params_of(&index).unwrap();
    let experts = p.moe.experts();
    let placeholder = || {
        WeightEntry::Dense(
            DenseWeight::try_new(DType::F32, vec![1], 0f32.to_le_bytes().to_vec().into()).unwrap(),
        )
    };
    let mut store = WeightStore::builder();
    for t in listed() {
        if t.ne.len() == 3 {
            for e in 0..experts {
                store
                    .insert(format!("{}.{e}", t.name), placeholder())
                    .unwrap();
            }
        } else {
            store.insert(t.name, placeholder()).unwrap();
        }
    }
    let store = store.build();
    let (names, _) = expert_layout(p.layout, &store);
    let map = weights_of(&p, &store, names).unwrap_or_else(|e| panic!("{e}"));
    let report = map.report(&store);
    assert!(
        report.unknown.is_empty(),
        "stored tensors no weight reads: {:?}",
        report.unknown.iter().take(5).collect::<Vec<_>>()
    );
    assert_eq!(report.loaded.len(), store.keys().count());
}

// ---------------------------------------------------------------------------------------------
// The loader over the real shard set: sparse files of the real sizes, no tensor byte read.
// ---------------------------------------------------------------------------------------------

/// A GGUF header with `kvs` and `tensors` (ggml dims, type, offset), data starting at the next
/// multiple of 32; returns the bytes and the data start.
fn header(kvs: &Kvs, tensors: &[&Listed]) -> Vec<u8> {
    fn gstr(out: &mut Vec<u8>, s: &str) {
        out.extend((s.len() as u64).to_le_bytes());
        out.extend(s.as_bytes());
    }
    fn scalar(out: &mut Vec<u8>, v: &GgufValue) {
        match v {
            GgufValue::U32(x) => out.extend(x.to_le_bytes()),
            GgufValue::U64(x) => out.extend(x.to_le_bytes()),
            GgufValue::F32(x) => out.extend(x.to_le_bytes()),
            GgufValue::Str(s) => gstr(out, s),
            other => panic!("{other:?}"),
        }
    }
    let tag = |v: &GgufValue| -> u32 {
        match v {
            GgufValue::U32(_) => 4,
            GgufValue::F32(_) => 6,
            GgufValue::Str(_) => 8,
            GgufValue::U64(_) => 10,
            other => panic!("{other:?}"),
        }
    };
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend((kvs.len() as u64).to_le_bytes());
    for (key, value) in kvs {
        gstr(&mut out, key);
        match value {
            GgufValue::Array(items) => {
                out.extend(9u32.to_le_bytes());
                out.extend(tag(&items[0]).to_le_bytes());
                out.extend((items.len() as u64).to_le_bytes());
                items.iter().for_each(|i| scalar(&mut out, i));
            }
            other => {
                out.extend(tag(other).to_le_bytes());
                scalar(&mut out, other);
            }
        }
    }
    for t in tensors {
        gstr(&mut out, &t.name);
        out.extend((t.ne.len() as u32).to_le_bytes());
        t.ne.iter().for_each(|d| out.extend(d.to_le_bytes()));
        out.extend(t.ggml_type.to_le_bytes());
        out.extend(t.offset.to_le_bytes());
    }
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    out
}

/// The three shards of the real set as sparse files: the real directory, every tensor's range inside
/// a file of the shard's real data size, the data zero until a test writes some.
struct SparseSet {
    paths: Vec<std::path::PathBuf>,
    data_start: Vec<u64>,
    _dir: poot_test_util::UniqueTempPath,
}

impl SparseSet {
    fn new() -> Self {
        use std::os::unix::fs::FileExt;
        let dir = poot_test_util::unique_temp_path("shards");
        std::fs::create_dir_all(std::path::Path::new(&*dir).parent().unwrap()).unwrap();
        let root = std::path::Path::new(&*dir).parent().unwrap().to_path_buf();
        let real = listed();
        let split = |no: u32| {
            [
                ("split.no", GgufValue::U32(no)),
                ("split.count", GgufValue::U32(3)),
                ("split.tensors.count", GgufValue::U32(1224)),
            ]
        };
        let (mut paths, mut data_start) = (Vec::new(), Vec::new());
        for shard in 0..3 {
            let mut kvs: Kvs = if shard == 0 { real_kvs() } else { Vec::new() };
            kvs.extend(split(shard as u32));
            let mine: Vec<&Listed> = real.iter().filter(|t| t.shard == shard).collect();
            let head = header(&kvs, &mine);
            let path = root.join(format!("Q-0000{}-of-00003.gguf", shard + 1));
            let file = std::fs::File::create(&path).unwrap();
            file.write_all_at(&head, 0).unwrap();
            let data = if shard == 0 {
                0
            } else {
                shard_data_bytes(shard)
            };
            file.set_len(head.len() as u64 + data).unwrap();
            data_start.push(head.len() as u64);
            paths.push(path);
        }
        Self {
            paths,
            data_start,
            _dir: dir,
        }
    }

    fn index(&self) -> GgufIndex {
        let files: Vec<StoredBytes> = self
            .paths
            .iter()
            .map(|p| StoredBytes::map(std::fs::File::open(p).unwrap()).unwrap())
            .collect();
        GgufIndex::parse_shards(&files).unwrap()
    }
}

/// A name map that keeps the named tensors and no others.
struct Keep<'a>(&'a [&'a str]);

impl TensorNameMap for Keep<'_> {
    fn weight_key(&self, name: &str) -> Option<WeightKey> {
        self.0.contains(&name).then(|| WeightKey::from(name))
    }
}

/// SC-004: the loader plans the whole real set with zero refusals: every one of the 1224 tensors,
/// the IQ2_XS and IQ3_XXS expert stacks included (`blk.0.ffn_gate_exps.weight` is IQ2_XS,
/// `blk.2.ffn_gate_exps.weight` IQ3_XXS), becomes owners, a stacked tensor one owner per expert.
/// Mutation: map ggml type 17 to `None` in `weight_format_of`; the plan refuses
/// `blk.0.ffn_gate_exps.weight`.
#[test]
fn the_loader_plans_the_whole_real_set_with_no_refusal() {
    let set = SparseSet::new();
    let index = set.index();
    assert_eq!(index.tensors.len(), 1224);
    let types: std::collections::BTreeSet<u32> = listed().iter().map(|t| t.ggml_type).collect();
    assert!(
        types.contains(&17) && types.contains(&18),
        "the set holds IQ2_XS and IQ3_XXS: {types:?}"
    );
    let plan = gguf_plan(&index, &IdentityNames).unwrap_or_else(|e| panic!("{e}"));
    let expected: usize = listed()
        .iter()
        .map(|t| if t.ne.len() == 3 { t.ne[2] as usize } else { 1 })
        .sum();
    assert_eq!(plan.len(), expected);
}

/// SC-005: a stacked expert tensor of the real set (`blk.47.ffn_down_exps.weight`, IQ4_NL
/// `[512, 2560, 640]` in shard 3) is cut into 512 owners at `expert * 921600` bytes, at the real
/// offset: three experts are marked in the sparse shard and each owner starts with its own mark.
/// Mutation: the per-expert stride is one block short.
#[test]
fn the_real_expert_stack_is_cut_at_the_real_stride() {
    use std::os::unix::fs::FileExt;
    let set = SparseSet::new();
    let down = listed()
        .into_iter()
        .find(|t| t.name == "blk.47.ffn_down_exps.weight")
        .unwrap();
    assert_eq!(
        (down.ggml_type, down.ne.clone(), down.shard),
        (20, vec![640, 2560, 512], 2)
    );
    let stride = 640 * 2560 / 32 * 18;
    assert_eq!(stride, 921_600);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&set.paths[2])
        .unwrap();
    // An IQ4_NL block: f16 scale 1.0 (0x3c00), then 16 bytes of nibbles. Each marked expert's first
    // block carries its own nibble byte; an unmarked expert's blocks are zero.
    let marked = [(0usize, 0x11u8), (1, 0x22), (511, 0x33)];
    let block = |fill: u8| -> Vec<u8> {
        let mut block = vec![0x00, 0x3c];
        block.extend([fill; 16]);
        block
    };
    for (expert, fill) in marked {
        file.write_all_at(
            &block(fill),
            set.data_start[2] + down.offset + (expert * stride) as u64,
        )
        .unwrap();
    }
    let index = set.index();
    let store = read_gguf(
        &index,
        &Keep(&["blk.47.ffn_down_exps.weight"]),
        poot_load::plan::default_workers(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(store.keys().count(), 512);
    for expert in [0usize, 1, 255, 511] {
        let key = format!("blk.47.ffn_down_exps.weight.{expert}");
        let Some(WeightEntry::Packed(payload)) = store.get(&key) else {
            panic!("{key} is a packed owner");
        };
        let bytes = payload.bytes(poot_quant::SourceRole::Blocks);
        assert_eq!(bytes.len(), stride, "{key}");
        let want = match marked.iter().find(|(e, _)| *e == expert) {
            Some(&(_, fill)) => block(fill),
            None => vec![0u8; 18],
        };
        assert_eq!(&bytes[..18], &want[..], "{key}");
    }
}

// ---------------------------------------------------------------------------------------------
// The converted fixture: the HF checkpoint, written the way the converter writes it.
// ---------------------------------------------------------------------------------------------

/// A tensor of the converted checkpoint: `[shape]` row-major f32.
struct Converted {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// `_reorder_v_heads` over an axis of `hk * vpk * hd` entries of a `[outer, axis, inner]` tensor:
/// grouped-by-key-head order to tiled order, `[hk, vpk, hd] -> [vpk, hk, hd]`.
fn tile(data: &[f32], outer: usize, inner: usize, hk: usize, vpk: usize, hd: usize) -> Vec<f32> {
    let axis = hk * vpk * hd;
    assert_eq!(data.len(), outer * axis * inner);
    let mut out = vec![0.0; data.len()];
    for o in 0..outer {
        for g in 0..hk {
            for j in 0..vpk {
                for d in 0..hd {
                    let (from, to) = ((g * vpk + j) * hd + d, (j * hk + g) * hd + d);
                    let (from, to) = ((o * axis + from) * inner, (o * axis + to) * inner);
                    out[to..to + inner].copy_from_slice(&data[from..from + inner]);
                }
            }
        }
    }
    out
}

/// The HF fixture's tensors as llama.cpp's converter writes them for `qwen4exp`.
fn convert(fixture: &Fixture) -> Vec<Converted> {
    let text = &fixture.config["text_config"];
    let n = |f: &str| Fixture::count(text, f);
    let (hk, hv, d) = (
        n("linear_num_key_heads"),
        n("linear_num_value_heads"),
        n("linear_key_head_dim"),
    );
    let vpk = hv / hk;
    let (hi, di) = (n("indexer_n_heads"), n("indexer_head_dim"));
    let (experts, inter, w) = (
        n("num_experts"),
        n("moe_intermediate_size"),
        n("hidden_size"),
    );
    let mut out = Vec::new();
    for (key, entry) in fixture.store.iter() {
        let key = key.as_str();
        let (shape, data) = (entry.shape(), values(entry));
        let mut add = |name: &str, shape: Vec<usize>, data: Vec<f32>| {
            assert_eq!(shape.iter().product::<usize>(), data.len(), "{name}");
            out.push(Converted {
                name: name.to_string(),
                shape,
                data,
            });
        };
        let plus_one = |data: &[f32]| data.iter().map(|v| v + 1.0).collect::<Vec<f32>>();
        let (layer, rest) = match key
            .strip_prefix("model.language_model.layers.")
            .and_then(|r| r.split_once('.'))
        {
            Some((l, rest)) => (Some(l.parse::<usize>().unwrap()), rest),
            None => (None, key),
        };
        let blk = |s: &str| format!("blk.{}.{s}", layer.unwrap());
        match (layer, rest) {
            (None, "model.language_model.embed_tokens.weight") => {
                add("token_embd.weight", shape, data)
            }
            (None, "lm_head.weight") => add("output.weight", shape, data),
            (None, "model.language_model.hyper_connection_mixer.hc_norm.weight") => {
                add("output_hc_norm.weight", shape, plus_one(&data))
            }
            (None, "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight") => {
                add("output_hc_down.weight", shape, data)
            }
            (None, "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight") => {
                add("output_hc_up.weight", shape, data)
            }
            (Some(_), _) => {
                let site = |s: &str| if s == "attn" { "attn" } else { "ffn" };
                if let Some(tail) = rest
                    .strip_prefix("attn_hyper_connection.")
                    .map(|t| ("attn", t))
                    .or_else(|| {
                        rest.strip_prefix("mlp_hyper_connection.")
                            .map(|t| ("mlp", t))
                    })
                {
                    let (s, tail) = tail;
                    let (name, data) = match tail {
                        "hc_norm.weight" => ("norm", plus_one(&data)),
                        "input_mix_weight_down.weight" => ("down", data),
                        "input_mix_weight_up.weight" => ("up", data),
                        "block_inject_weight.weight" => ("inject", data),
                        other => panic!("{other}"),
                    };
                    add(&blk(&format!("hc_{}_{name}.weight", site(s))), shape, data);
                    continue;
                }
                match rest {
                    "ple.ple_embedding.ngram_embedding.weight" => {
                        add("per_layer_token_embd.weight", shape, data)
                    }
                    "ple.key_proj.weight" => add(&blk("ple_key.weight"), shape, data),
                    "ple.value_proj.weight" => add(&blk("ple_value.weight"), shape, data),
                    "ple.norm_key.weight" => {
                        add(&blk("ple_norm_key.weight"), shape, plus_one(&data))
                    }
                    "ple.norm_query.weight" => {
                        add(&blk("ple_norm_query.weight"), shape, plus_one(&data))
                    }
                    "ple.norm_conv.weight" => {
                        add(&blk("ple_norm_conv.weight"), shape, plus_one(&data))
                    }
                    "ple.conv1d.weight" => {
                        add(&blk("ple_conv1d.weight"), vec![shape[0], shape[2]], data)
                    }
                    "self_attn.q_proj.weight" => add(&blk("attn_q.weight"), shape, data),
                    "self_attn.k_proj.weight" => add(&blk("attn_k.weight"), shape, data),
                    "self_attn.v_proj.weight" => add(&blk("attn_v.weight"), shape, data),
                    "self_attn.o_proj.weight" => add(&blk("attn_output.weight"), shape, data),
                    "self_attn.q_norm.weight" => {
                        add(&blk("attn_q_norm.weight"), shape, plus_one(&data))
                    }
                    "self_attn.k_norm.weight" => {
                        add(&blk("attn_k_norm.weight"), shape, plus_one(&data))
                    }
                    "self_attn.indexer.index_qk_proj.weight" => {
                        // Query rows first, then the one key row group.
                        let q = hi * di;
                        let cols = shape[1];
                        add(
                            &blk("indexer.q_proj.weight"),
                            vec![q, cols],
                            data[..q * cols].to_vec(),
                        );
                        add(
                            &blk("indexer.k_proj.weight"),
                            vec![di, cols],
                            data[q * cols..].to_vec(),
                        );
                    }
                    "self_attn.indexer.q_layernorm.weight" => {
                        add(&blk("indexer.q_norm.weight"), shape, plus_one(&data))
                    }
                    "self_attn.indexer.k_layernorm.weight" => {
                        add(&blk("indexer.k_norm.weight"), shape, plus_one(&data))
                    }
                    "linear_attn.in_proj_qkv.weight" => {
                        // Query and key rows stay; the value rows are tiled.
                        let (qk, cols) = (2 * hk * d, shape[1]);
                        let mut converted = data[..qk * cols].to_vec();
                        converted.extend(tile(&data[qk * cols..], 1, cols, hk, vpk, d));
                        add(&blk("attn_qkv.weight"), shape, converted);
                    }
                    "linear_attn.in_proj_z.weight" => add(
                        &blk("attn_gate.weight"),
                        shape.clone(),
                        tile(&data, 1, shape[1], hk, vpk, d),
                    ),
                    "linear_attn.in_proj_b.weight" => add(
                        &blk("ssm_beta.weight"),
                        shape.clone(),
                        tile(&data, 1, shape[1], hk, vpk, 1),
                    ),
                    "linear_attn.in_proj_a.weight" => add(
                        &blk("ssm_alpha.weight"),
                        shape.clone(),
                        tile(&data, 1, shape[1], hk, vpk, 1),
                    ),
                    "linear_attn.conv1d.weight" => {
                        // `[channels, 1, kernel]` squeezed; only the value channels are tiled.
                        let (qk, kernel) = (2 * hk * d, shape[2]);
                        let mut converted = data[..qk * kernel].to_vec();
                        converted.extend(tile(&data[qk * kernel..], 1, kernel, hk, vpk, d));
                        add(&blk("ssm_conv1d.weight"), vec![shape[0], kernel], converted);
                    }
                    "linear_attn.dt_bias" => {
                        add(&blk("ssm_dt.bias"), shape, tile(&data, 1, 1, hk, vpk, 1))
                    }
                    "linear_attn.A_log" => add(
                        &blk("ssm_a"),
                        shape,
                        tile(&data, 1, 1, hk, vpk, 1)
                            .into_iter()
                            .map(|a| -a.exp())
                            .collect(),
                    ),
                    "linear_attn.norm.weight" => add(&blk("ssm_norm.weight"), shape, data),
                    "linear_attn.out_proj.weight" => add(
                        &blk("ssm_out.weight"),
                        shape.clone(),
                        tile(&data, shape[0], 1, hk, vpk, d),
                    ),
                    "mlp.gate.weight" => add(&blk("ffn_gate_inp.weight"), shape, data),
                    "mlp.shared_expert.gate_proj.weight" => {
                        add(&blk("ffn_gate_shexp.weight"), shape, data)
                    }
                    "mlp.shared_expert.up_proj.weight" => {
                        add(&blk("ffn_up_shexp.weight"), shape, data)
                    }
                    "mlp.shared_expert.down_proj.weight" => {
                        add(&blk("ffn_down_shexp.weight"), shape, data)
                    }
                    "mlp.shared_expert_gate.weight" => {
                        add(&blk("ffn_gate_inp_shexp.weight"), vec![shape[1]], data)
                    }
                    "mlp.experts.gate_up_proj" => {
                        // `[experts, 2 * inter, width]`, gate rows first.
                        let (gate, up): (Vec<_>, Vec<_>) = data
                            .chunks(2 * inter * w)
                            .map(|e| (e[..inter * w].to_vec(), e[inter * w..].to_vec()))
                            .unzip();
                        add(
                            &blk("ffn_gate_exps.weight"),
                            vec![experts, inter, w],
                            gate.concat(),
                        );
                        add(
                            &blk("ffn_up_exps.weight"),
                            vec![experts, inter, w],
                            up.concat(),
                        );
                    }
                    "mlp.experts.down_proj" => add(&blk("ffn_down_exps.weight"), shape, data),
                    other => panic!("an HF tensor the converter test does not know: {other}"),
                }
            }
            (None, other) => panic!("an HF tensor the converter test does not know: {other}"),
        }
    }
    out
}

/// The metadata the converter writes for `fixture` (the inverse of the HF config), with the PLE
/// hash constants from the reference derivation.
fn fixture_kvs(fixture: &Fixture, p: &Qwen4ExpParams) -> Kvs {
    let text = &fixture.config["text_config"];
    let n = |f: &str| Fixture::count(text, f) as u32;
    let layers = n("num_hidden_layers") as usize;
    let kinds: Vec<bool> = text["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| k == "full_attention")
        .collect();
    let ratio = n("indexer_compress_ratio");
    let head_dim = n("head_dim");
    let rotary = (head_dim as f64 * text["partial_rotary_factor"].as_f64().unwrap()).round() as u32;
    let mut kvs: Kvs = vec![
        ("general.architecture", GgufValue::Str("qwen4exp".into())),
        ("qwen4exp.block_count", GgufValue::U32(layers as u32)),
        (
            "qwen4exp.context_length",
            GgufValue::U32(n("max_position_embeddings")),
        ),
        (
            "qwen4exp.embedding_length",
            GgufValue::U32(n("hidden_size")),
        ),
        (
            "qwen4exp.attention.head_count",
            GgufValue::U32(n("num_attention_heads")),
        ),
        (
            "qwen4exp.attention.head_count_kv",
            GgufValue::U32(n("num_key_value_heads")),
        ),
        (
            "qwen4exp.rope.freq_base",
            GgufValue::F32(text["rope_parameters"]["rope_theta"].as_f64().unwrap() as f32),
        ),
        (
            "qwen4exp.attention.layer_norm_rms_epsilon",
            GgufValue::F32(text["rms_norm_eps"].as_f64().unwrap() as f32),
        ),
        ("qwen4exp.expert_count", GgufValue::U32(n("num_experts"))),
        (
            "qwen4exp.expert_used_count",
            GgufValue::U32(n("num_experts_per_tok")),
        ),
        ("qwen4exp.attention.key_length", GgufValue::U32(head_dim)),
        ("qwen4exp.attention.value_length", GgufValue::U32(head_dim)),
        (
            "qwen4exp.expert_feed_forward_length",
            GgufValue::U32(n("moe_intermediate_size")),
        ),
        (
            "qwen4exp.expert_shared_feed_forward_length",
            GgufValue::U32(n("shared_expert_intermediate_size")),
        ),
        (
            "qwen4exp.ssm.conv_kernel",
            GgufValue::U32(n("linear_conv_kernel_dim")),
        ),
        (
            "qwen4exp.ssm.state_size",
            GgufValue::U32(n("linear_key_head_dim")),
        ),
        (
            "qwen4exp.ssm.group_count",
            GgufValue::U32(n("linear_num_key_heads")),
        ),
        (
            "qwen4exp.ssm.time_step_rank",
            GgufValue::U32(n("linear_num_value_heads")),
        ),
        (
            "qwen4exp.ssm.inner_size",
            GgufValue::U32(n("linear_value_head_dim") * n("linear_num_value_heads")),
        ),
        ("qwen4exp.rope.dimension_count", GgufValue::U32(rotary)),
        (
            "qwen4exp.hyper_connection.count",
            GgufValue::U32(n("hc_count")),
        ),
        (
            "qwen4exp.hyper_connection.low_rank",
            GgufValue::U32(n("hc_lowrank")),
        ),
        (
            "qwen4exp.attention.indexer.head_count",
            GgufValue::U32(n("indexer_n_heads")),
        ),
        (
            "qwen4exp.attention.indexer.key_length",
            GgufValue::U32(n("indexer_head_dim")),
        ),
        (
            "qwen4exp.attention.indexer.top_k",
            GgufValue::U32(n("indexer_budget")),
        ),
        (
            "qwen4exp.attention.compress_ratios",
            u32s(kinds.iter().map(|&full| if full { ratio } else { 0 })),
        ),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(vec![GgufValue::Str("t".into()); n("vocab_size") as usize]),
        ),
        (
            "tokenizer.ggml.eos_token_id",
            GgufValue::U32(n("eos_token_id")),
        ),
    ];
    // The converter writes the interval only as a default; state it when the pattern is regular.
    if let Some(interval) = kinds.iter().position(|&full| full).map(|i| i + 1)
        && kinds
            .iter()
            .enumerate()
            .all(|(l, &full)| full == (l + 1).is_multiple_of(interval))
    {
        kvs.push((
            "qwen4exp.full_attention_interval",
            GgufValue::U32(interval as u32),
        ));
    }
    let ple: Vec<u32> = (0..layers)
        .filter(|&l| p.ple[l].is_some())
        .map(|l| l as u32)
        .collect();
    if let [layer] = ple[..] {
        let spec = p.ple[layer as usize].as_ref().unwrap().spec().clone();
        let tables = ple_oracle::tables(&spec);
        let as_u64 = |v: &[i64]| u64s(v.iter().map(|&x| x as u64));
        kvs.extend([
            ("qwen4exp.ple.layers", u32s([layer])),
            (
                "qwen4exp.ple.ngram_size",
                GgufValue::U32(spec.ngram_size as u32),
            ),
            (
                "qwen4exp.ple.heads_per_ngram",
                GgufValue::U32(spec.heads_per_ngram as u32),
            ),
            (
                "qwen4exp.ple.conv_kernel",
                GgufValue::U32(spec.conv_kernel as u32),
            ),
            ("qwen4exp.ple.eos_token_id", GgufValue::U32(spec.eos)),
            (
                "qwen4exp.embedding_length_per_layer_input",
                GgufValue::U32(
                    (spec.embed_dim / ((spec.ngram_size - 1) * spec.heads_per_ngram)) as u32,
                ),
            ),
            (
                "qwen4exp.ple.layer_multipliers",
                as_u64(&tables.multipliers),
            ),
            ("qwen4exp.ple.head_offsets", as_u64(&tables.head_offset)),
            ("qwen4exp.ple.head_vocab_sizes", as_u64(&tables.head_vocab)),
        ]);
    }
    kvs
}

/// The converted checkpoint: its index (metadata and directory) and its F32 weights.
fn converted(fixture: &Fixture) -> (GgufIndex, WeightStore, Kvs) {
    let hf = Qwen4ExpParams::from_raw(&fixture.raw()).unwrap();
    let kvs = fixture_kvs(fixture, &hf);
    let tensors = convert(fixture);
    let directory = tensors.iter().map(|t| Listed {
        name: t.name.clone(),
        ne: t.shape.iter().rev().map(|&d| d as u64).collect(),
        ggml_type: 0,
        shard: 0,
        offset: 0,
    });
    let index = index_of(&kvs, directory);
    let mut store = WeightStore::builder();
    for t in tensors {
        let bytes: Vec<u8> = t.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        store
            .insert(
                t.name,
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::F32, t.shape, bytes.into()).unwrap(),
                ),
            )
            .unwrap();
    }
    (index, store.build(), kvs)
}

fn built_from_gguf(
    index: &GgufIndex,
    store: &WeightStore,
) -> Result<Box<dyn Model>, crate::registry::LoadModelError> {
    Registry::builtin()
        .expect("the builtin registry")
        .build(&RawConfig::Gguf(index), store)
        .map(|parts| parts.model)
}

/// SC-006: the converted fixture has exactly the tensors, with exactly the shapes, the shape table
/// gives for its metadata: the converter test and the table agree, so the table's agreement with the
/// real directory ([`the_shape_table_is_the_real_directory`]) is a statement about the converter's
/// layout too.
#[test]
fn the_converted_fixture_has_the_table_s_tensors_and_shapes() {
    let fixture = super::super::fixture();
    let (index, store, kvs) = converted(&fixture);
    let rows = index.tensors["per_layer_token_embd.weight"].dims[1] as usize;
    let expected = expected_shapes(&kvs, rows);
    let got: BTreeMap<String, Vec<usize>> = store
        .iter()
        .map(|(k, e)| (k.as_str().to_string(), e.shape()))
        .collect();
    assert_eq!(got, expected);
}

/// SC-007: the converted fixture, loaded through the GGUF path, produces the HF path's logits at
/// every position and every carried element (state), which the independent f64 reference matches:
/// the prompt as one prefill, and as a prefill followed by decode steps. This pins every stored
/// convention at once: the folded norm scales, the tiled value heads, `-exp(A_log)`, the split
/// indexer, the squeezed convs, the model-level n-gram table and the stored hash constants.
/// Mutations GQ-1 (norms read as `1 + w`), GQ-2 (value heads read grouped), GQ-3 (`exp` applied to
/// `ssm_a`), GQ-4 (the indexer key rows mapped from the query tensor) and GQ-5 (the hash re-derived
/// from the stored constants' own base) are each red here.
#[test]
fn the_converted_checkpoint_matches_the_hf_logits_and_state() {
    let fixture = super::super::fixture();
    let (index, gguf_store, _) = converted(&fixture);
    let model = built_from_gguf(&index, &gguf_store).unwrap_or_else(|e| panic!("{e}"));
    let case = Case::new(fixture.clone());
    let want = Reference::new(&fixture).forward(&case.tokens, case.cap);
    let names = case.state_names(&*model);
    assert_eq!(
        names,
        case.state_names(&*case.model()),
        "the GGUF model carries the state the HF model carries"
    );
    let tokens = &case.tokens;
    let half = tokens.len() / 2;
    for plan in [
        vec![tokens.len()],
        std::iter::once(half)
            .chain(std::iter::repeat_n(1, tokens.len() - half))
            .collect(),
    ] {
        let (mut logits, mut state, mut at) = (Vec::new(), Vec::new(), 0);
        for len in plan.iter().copied() {
            let (out, next) = run_step(
                &*model,
                &gguf_store,
                case.cap,
                &tokens[at..at + len],
                at as i32,
                &state,
            );
            logits.extend(out);
            state = next;
            at += len;
        }
        let label = format!("gguf {plan:?}");
        assert_logits(&label, case.vocab(), &logits, &want.logits, 0);
        // The GDN's carried state is in the stored head order (tiled), so it is the HF state
        // permuted; the logits of the chained plan already depend on it. Every other element a step
        // carries (the KV and indexer caches, the PLE history and conv window) is in one order.
        let keep: Vec<usize> = (0..names.len())
            .filter(|&i| !names[i].starts_with("gdn."))
            .collect();
        let kept: Vec<String> = keep.iter().map(|&i| names[i].clone()).collect();
        let got: Vec<Vec<f32>> = keep.iter().map(|&i| state[i].clone()).collect();
        let reference: BTreeMap<String, Vec<f64>> = want
            .state
            .iter()
            .filter(|(name, _)| !name.starts_with("gdn."))
            .map(|(name, v)| (name.clone(), v.clone()))
            .collect();
        assert_state(&label, &kept, &got, &reference);
    }
}

/// SC-008: a GGUF config the family cannot configure is a typed error naming the key the file used,
/// never a panic or a silent default: the layer kinds come from the compress ratios (absent, short,
/// two block widths, no full-attention layer, or disagreeing with the stated interval), the PLE
/// block names one linear-attention layer in range with every constant it needs, the stored hash
/// constants fit the layer and its table, and the table tensor exists. Mutation GQ-6 (the ratios
/// ignored) turns the first rows red.
#[test]
fn a_gguf_config_the_family_cannot_configure_is_refused_typed() {
    let fixture = super::super::fixture();
    let (index, _, kvs) = converted(&fixture);
    let refuse = |edit: &dyn Fn(&mut Kvs, &mut BTreeMap<String, GgufTensorInfo>)| {
        let mut kvs = kvs.clone();
        let mut tensors: BTreeMap<String, GgufTensorInfo> = index
            .tensors
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        edit(&mut kvs, &mut tensors);
        let mut mutated = index_of(&kvs, std::iter::empty());
        mutated.tensors = tensors.into_iter().collect();
        match params_of(&mutated).expect_err("the config is refused") {
            ModelError::Config { field, reason, .. } => (field, reason),
            other => panic!("{other:?}"),
        }
    };
    let set = |kvs: &mut Kvs, key: &'static str, value: GgufValue| {
        kvs.retain(|(k, _)| *k != key);
        kvs.push((key, value));
    };
    let ratios = "qwen4exp.attention.compress_ratios";
    let interval = "qwen4exp.full_attention_interval";
    let layers = "qwen4exp.ple.layers";

    assert_eq!(
        refuse(&|kvs, _| kvs.retain(|(k, _)| *k != ratios)),
        (ratios, ConfigReason::Missing)
    );
    assert_eq!(
        refuse(&|kvs, _| set(kvs, ratios, u32s([0, 0]))),
        (ratios, ConfigReason::Length { expected: 3 })
    );
    // Two block widths: the prefill granule and the indexer's pooling share one.
    assert_eq!(
        refuse(&|kvs, _| set(kvs, ratios, u32s([0, 4, 2]))),
        (ratios, ConfigReason::Unsupported)
    );
    assert_eq!(
        refuse(&|kvs, _| set(kvs, ratios, u32s([0, 0, 0]))),
        (ratios, ConfigReason::Unsupported)
    );
    // The stated interval must agree with the ratios: interval 2 makes layer 1 full attention.
    assert_eq!(
        refuse(&|kvs, _| set(kvs, interval, GgufValue::U32(2))),
        (interval, ConfigReason::Unsupported)
    );
    // The PLE: two layers (one table, one set of constants), a full-attention layer, out of range.
    assert_eq!(
        refuse(&|kvs, _| set(kvs, layers, u32s([0, 1]))),
        (layers, ConfigReason::Unsupported)
    );
    assert_eq!(
        refuse(&|kvs, _| set(kvs, layers, u32s([2]))),
        (layers, ConfigReason::Unsupported)
    );
    assert_eq!(
        refuse(&|kvs, _| set(kvs, layers, u32s([3]))),
        (layers, ConfigReason::Exceeds { max: 2 })
    );
    // The stored hash constants: absent, the wrong count, an empty slice, a slice past the table.
    let multipliers = "qwen4exp.ple.layer_multipliers";
    let vocabs = "qwen4exp.ple.head_vocab_sizes";
    assert_eq!(
        refuse(&|kvs, _| kvs.retain(|(k, _)| *k != multipliers)),
        (multipliers, ConfigReason::Missing)
    );
    assert_eq!(
        refuse(&|kvs, _| set(kvs, multipliers, u64s([3, 5]))),
        (
            "ple.layer_multipliers",
            ConfigReason::Length { expected: 3 }
        )
    );
    assert_eq!(
        refuse(&|kvs, _| set(kvs, vocabs, u64s([0, 0, 0, 0]))),
        ("ple.head_vocab_sizes", ConfigReason::Zero)
    );
    assert_eq!(
        refuse(&|_, tensors| {
            tensors.get_mut("per_layer_token_embd.weight").unwrap().dims[1] = 1;
        }),
        ("ple.head_offsets", ConfigReason::Exceeds { max: 1 })
    );
    assert_eq!(
        refuse(&|_, tensors| {
            tensors.remove("per_layer_token_embd.weight");
        }),
        ("per_layer_token_embd.weight", ConfigReason::Missing)
    );
    // The GDN's inner size is the value heads' total width.
    assert_eq!(
        refuse(&|kvs, _| set(kvs, "qwen4exp.ssm.inner_size", GgufValue::U32(7))),
        ("qwen4exp.ssm.inner_size", ConfigReason::Unsupported)
    );
}

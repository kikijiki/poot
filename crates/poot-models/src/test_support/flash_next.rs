//! Qwen3.8-Flash-Next UD-Q2_K_XL as it is published: the real tensor directory of its three GGUF shards
//! (`tests/data/flash_next_ud_q2_k_xl_tensors.tsv`), the real `qwen4exp.*` metadata, and the shard set as
//! sparse files of the real sizes. A test that loads or plans the checkpoint reads no tensor byte: the data
//! is zero until a test writes some.

use poot_load::gguf::{GgufIndex, GgufValue};
use poot_quant::StoredBytes;

/// The directory file: comment lines carry each shard's data size, the rest one tensor per line.
const DIRECTORY: &str = include_str!("../../tests/data/flash_next_ud_q2_k_xl_tensors.tsv");

/// One tensor of the real directory.
#[derive(Clone, Debug)]
pub struct Listed {
    /// The tensor's name.
    pub name: String,
    /// ggml `ne`: fastest-varying first.
    pub ne: Vec<u64>,
    /// The ggml type id.
    pub ggml_type: u32,
    /// The 0-based shard holding it.
    pub shard: usize,
    /// Its offset in the shard's data section.
    pub offset: u64,
}

impl Listed {
    /// The logical row-major shape.
    pub fn shape(&self) -> Vec<usize> {
        self.ne.iter().rev().map(|&d| d as usize).collect()
    }
}

/// Every tensor of the real directory, in file order.
pub fn listed() -> Vec<Listed> {
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

/// GGUF metadata as the writer takes it.
pub type Kvs = Vec<(&'static str, GgufValue)>;

/// A GGUF array of `u32`.
pub fn u32s(values: impl IntoIterator<Item = u32>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufValue::U32).collect())
}

/// A GGUF array of `u64`.
pub fn u64s(values: impl IntoIterator<Item = u64>) -> GgufValue {
    GgufValue::Array(values.into_iter().map(GgufValue::U64).collect())
}

/// The `qwen4exp.*` and tokenizer metadata of the real first shard (values as the published file
/// states them; the 248,320 token strings are stood in by `t<id>` and the merges by one).
pub fn real_kvs() -> Kvs {
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
        ("tokenizer.ggml.model", GgufValue::Str("gpt2".into())),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(
                (0..248_320)
                    .map(|id| GgufValue::Str(format!("t{id}")))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges",
            GgufValue::Array(vec![GgufValue::Str("t1 t2".into())]),
        ),
        ("tokenizer.ggml.eos_token_id", n(248_046)),
        ("tokenizer.ggml.bos_token_id", n(248_044)),
    ]
}

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
pub struct SparseSet {
    /// The three shard files, in set order.
    pub paths: Vec<std::path::PathBuf>,
    /// Where each shard's data section starts.
    pub data_start: Vec<u64>,
    _dir: poot_test_util::UniqueTempPath,
}

impl SparseSet {
    /// Write the three sparse shards.
    pub fn create() -> Self {
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

    /// The shard set parsed as one index.
    pub fn index(&self) -> GgufIndex {
        let files: Vec<StoredBytes> = self
            .paths
            .iter()
            .map(|p| StoredBytes::map(std::fs::File::open(p).unwrap()).unwrap())
            .collect();
        GgufIndex::parse_shards(&files).unwrap()
    }
}

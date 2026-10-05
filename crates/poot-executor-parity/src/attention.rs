//! Attention fixtures: flash decode and prefill under a per-head ALiBi mask, and decode over the
//! query/key-value head layouts (multi-head, multi-query) the shared qwen2 decode fixture does not
//! cover.

use poot_graph_ir::ops::{attention_masked, attention_prefill_softcap};
use poot_graph_ir::{Builder, TensorType};
use poot_tensor::HostTensor;

use crate::dense::{Dense, Family};
use crate::{Fixture, const_fixture, decode_fixture, store_for};

const HQ: usize = 4;
const HKV: usize = 2;
const D: usize = 4;

/// One ALiBi case: the same attention over the same q/k/v, once with the per-head mask and once
/// with head 0's mask row broadcast to every head.
pub struct AlibiCase {
    /// The mask holds a different slope for every head.
    pub per_head: Fixture,
    /// The same program with a broadcast mask (`Hm = 1`) bound to head 0's row. A device that read
    /// the per-head mask at head stride 0 would produce this output, so the oracle's output for
    /// `per_head` must differ from this fixture's: a row asserts it, which shows `per_head` can tell
    /// the two apart.
    pub head0_broadcast: Fixture,
    /// The prefix of the planned op name this case runs as (`flash_attn_decode` or
    /// `flash_attn_prefill`).
    pub flash_op: &'static str,
}

/// Per-head ALiBi slopes: a distinct power of two for each head.
const SLOPES: [f32; HQ] = [0.25, 0.0625, 0.015625, 0.00390625];

fn centered(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// The three flash kernels the planner picks between: the synthesized region decode, the
/// synthesized region prefill, and the imported flash prefill (chosen for a softcapped prefill,
/// which the region prefill has no softcap for). The per-head mask is a real ALiBi mask:
/// visibility plus `-slope[h] * (query - key)`.
pub fn alibi_flash_cases() -> Vec<AlibiCase> {
    vec![
        alibi_decode_case(),
        alibi_prefill_case("alibi_flash_prefill_region", None),
        alibi_prefill_case("alibi_flash_prefill_imported", Some(50.0)),
    ]
}

fn alibi_decode_case() -> AlibiCase {
    let (cap, pos) = (6usize, 3usize);
    let mask: Vec<f32> = (0..HQ)
        .flat_map(|h| {
            (0..cap).map(move |t| {
                if t <= pos {
                    -SLOPES[h] * (pos as f32 - t as f32)
                } else {
                    -1.0e9
                }
            })
        })
        .collect();
    let build = |name: &'static str, mask_heads: usize, mask: Vec<f32>| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, HQ, 1, D]));
        let k = b.constant("k", TensorType::f32(vec![1, HKV, cap, D]));
        let v = b.constant("v", TensorType::f32(vec![1, HKV, cap, D]));
        let m = b.constant("mask", TensorType::f32(vec![1, mask_heads, 1, cap]));
        let y = attention_masked(&b, q, k, v, HQ / HKV, 1.0 / (D as f32).sqrt(), m);
        let graph = b.finish(y);
        const_fixture(
            name,
            graph,
            vec![
                ("q", HostTensor::f32(vec![1, HQ, 1, D], centered(1, HQ * D))),
                (
                    "k",
                    HostTensor::f32(vec![1, HKV, cap, D], centered(2, HKV * cap * D)),
                ),
                (
                    "v",
                    HostTensor::f32(vec![1, HKV, cap, D], centered(3, HKV * cap * D)),
                ),
                ("mask", HostTensor::f32(vec![1, mask_heads, 1, cap], mask)),
            ],
            2,
        )
    };
    AlibiCase {
        head0_broadcast: build("alibi_flash_decode_head0", 1, mask[..cap].to_vec()),
        per_head: build("alibi_flash_decode", HQ, mask),
        flash_op: "flash_attn_decode",
    }
}

fn alibi_prefill_case(name: &'static str, softcap: Option<f32>) -> AlibiCase {
    let l = 6usize;
    let mask: Vec<f32> = (0..HQ)
        .flat_map(|h| {
            (0..l).flat_map(move |i| {
                (0..l).map(move |j| {
                    if j <= i {
                        -SLOPES[h] * (i as f32 - j as f32)
                    } else {
                        -1.0e30
                    }
                })
            })
        })
        .collect();
    let build = |name: &'static str, mask_heads: usize, mask: Vec<f32>| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, HQ, l, D]));
        let k = b.constant("k", TensorType::f32(vec![1, HKV, l, D]));
        let v = b.constant("v", TensorType::f32(vec![1, HKV, l, D]));
        let m = b.constant("m", TensorType::f32(vec![1, mask_heads, l, l]));
        let y =
            attention_prefill_softcap(&b, q, k, v, HQ / HKV, 1.0 / (D as f32).sqrt(), m, softcap);
        let graph = b.finish(y);
        const_fixture(
            name,
            graph,
            vec![
                (
                    "q",
                    HostTensor::f32(vec![1, HQ, l, D], centered(4, HQ * l * D)),
                ),
                (
                    "k",
                    HostTensor::f32(vec![1, HKV, l, D], centered(5, HKV * l * D)),
                ),
                (
                    "v",
                    HostTensor::f32(vec![1, HKV, l, D], centered(6, HKV * l * D)),
                ),
                ("m", HostTensor::f32(vec![1, mask_heads, l, l], mask)),
            ],
            2,
        )
    };
    AlibiCase {
        head0_broadcast: build("alibi_flash_prefill_head0", 1, mask[..l * l].to_vec()),
        per_head: build(name, HQ, mask),
        flash_op: "flash_attn_prefill",
    }
}

/// A qwen2 decode over `n_heads` query heads sharing `n_kv_heads` key-value heads: `4 x 2` (the
/// grouped-query layout) is [`crate::qwen2_decode_fixture`]; these add multi-head (`2 x 2`, every
/// query head its own key-value head) and multi-query (`2 x 1`, one key-value head shared by all).
pub fn head_layout_decode_fixtures(cap: usize) -> Vec<Fixture> {
    [
        ("qwen2_decode_multi_head", 2, 2),
        ("qwen2_decode_multi_query", 2, 1),
    ]
    .into_iter()
    .map(|(name, n_heads, n_kv_heads)| {
        let dense = Dense::new(Family::Qwen2)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(n_heads, n_kv_heads)
            .head_dim(4)
            .max_positions(16);
        decode_fixture(name, &dense, cap, store_for)
    })
    .collect()
}

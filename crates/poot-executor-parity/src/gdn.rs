//! The chunked Gated-DeltaNet prefill: many small chained dependent dispatches in one recorded
//! program, carrying the recurrent state across calls.

use poot_graph_ir::ops::gdn_prefill_chunked;
use poot_graph_ir::{Builder, Slot, StateRole, TensorType};
use poot_graph_plan::FusionPolicy;
use poot_tensor::HostTensor;

use poot_test_util::StepFixture;

use crate::{Fixture, store_from_tensors};

/// The toy scale of the chunked prefill: `LEN` positions in `CHUNK`-position chunks, `H_K` key heads
/// repeated (tiled, `hv % H_K`) to `H_V` value heads, each of head dim `D`.
pub const LEN: usize = 8;
pub const CHUNK: usize = 4;
pub const H_K: usize = 2;
pub const H_V: usize = 4;
pub const D: usize = 3;

/// One call's raw inputs, row-major: `q`, `k` are `[H_K, LEN, D]` (L2-normalized, unscaled), `v` is
/// `[H_V, LEN, D]`, `g` (the log-domain forget gate, negative) and `beta` (the delta weight, in
/// `(0, 1)`) are `[H_V, LEN]`.
pub struct GdnInputs {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub beta: Vec<f32>,
}

/// The chunked prefill fixture and the inputs of each of its steps. Step 1 starts from the state step
/// 0 left, which is nonzero, so both state-dependent cross terms of the chunked algebra are exercised
/// from step 1 on.
pub struct GdnChunkedCase {
    pub fixture: Fixture,
    pub inputs: Vec<GdnInputs>,
}

fn l2_normalized(mut raw: Vec<f32>, rows: usize, dim: usize) -> Vec<f32> {
    for r in 0..rows {
        let row = &mut raw[r * dim..(r + 1) * dim];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        row.iter_mut().for_each(|x| *x /= norm);
    }
    raw
}

fn unit_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn inputs(seed: u64) -> GdnInputs {
    GdnInputs {
        q: l2_normalized(unit_fill(H_K * LEN * D, seed + 101), H_K * LEN, D),
        k: l2_normalized(unit_fill(H_K * LEN * D, seed + 202), H_K * LEN, D),
        v: unit_fill(H_V * LEN * D, seed + 303),
        g: (0..H_V * LEN)
            .map(|i| -0.05 - 0.03 * ((i * 5 + i / LEN) % 4) as f32)
            .collect(),
        beta: (0..H_V * LEN)
            .map(|i| 0.3 + 0.5 * (((i * 3 + i / LEN) % 4) as f32 / 4.0))
            .collect(),
    }
}

/// Two chunked prefill calls over one carried state.
pub fn gdn_prefill_chunked_case() -> GdnChunkedCase {
    let b = Builder::new();
    let slot =
        |tag: &str, shape: Vec<usize>| b.slot_named(Slot::Activation, tag, TensorType::f32(shape));
    let q = slot("gdn.q", vec![1, H_K, LEN, D]);
    let k = slot("gdn.k", vec![1, H_K, LEN, D]);
    let v = slot("gdn.v", vec![1, H_V, LEN, D]);
    let g = slot("gdn.g", vec![1, H_V, LEN, 1]);
    let beta = slot("gdn.beta", vec![1, H_V, LEN, 1]);
    let state = b.state_input(
        "gdn.state",
        TensorType::f32(vec![1, H_V, D, D]),
        StateRole::Recurrent,
    );
    let tril_incl = b.constant("gdn.tril_incl", TensorType::f32(vec![1, 1, CHUNK, CHUNK]));
    let tril_strict = b.constant("gdn.tril_strict", TensorType::f32(vec![1, 1, CHUNK, CHUNK]));
    let (out, state_out) =
        gdn_prefill_chunked(&b, q, k, v, g, beta, state, tril_incl, tril_strict, CHUNK);
    let graph = b.finish_with_state(out, &[(state, state_out)]);

    let triangle = |keep: fn(usize, usize) -> bool| {
        let data = (0..CHUNK * CHUNK)
            .map(|i| if keep(i / CHUNK, i % CHUNK) { 1.0 } else { 0.0 })
            .collect();
        HostTensor::f32(vec![1, 1, CHUNK, CHUNK], data)
    };
    let store = store_from_tensors(vec![
        ("gdn.tril_incl", triangle(|i, j| j <= i)),
        ("gdn.tril_strict", triangle(|i, j| j < i)),
    ]);

    let calls = vec![inputs(0), inputs(1000)];
    let key = |id| graph.meta(id).slot_key().unwrap().clone();
    let steps = calls
        .iter()
        .map(|call| {
            [
                (q.id, vec![1, H_K, LEN, D], &call.q),
                (k.id, vec![1, H_K, LEN, D], &call.k),
                (v.id, vec![1, H_V, LEN, D], &call.v),
                (g.id, vec![1, H_V, LEN, 1], &call.g),
                (beta.id, vec![1, H_V, LEN, 1], &call.beta),
            ]
            .into_iter()
            .map(|(id, shape, data)| StepFixture {
                key: key(id),
                tensor: HostTensor::f32(shape, data.clone()),
            })
            .collect()
        })
        .collect();
    GdnChunkedCase {
        fixture: Fixture {
            name: "gdn_prefill_chunked",
            graph,
            store,
            steps,
            fusion: FusionPolicy::Full,
        },
        inputs: calls,
    }
}

//! A decode graph over the guarded live-capacity attention composite, with its operands and an oracle.
//!
//! The graph derives its length in the graph from `Slot::Pos` (`len = pos + 1`, never a host-bound length),
//! masks with the in-graph causal mask of the same position, and declares the composite's length witness as
//! a validation output. [`LiveAttentionFixture::oracle`] is the plain materialized attention over the first
//! `len[b]` keys of each batch row, built from the ordinary (unguarded) composite, so it shares nothing with
//! the guarded one but the arithmetic.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::ops::{
    AttentionKeys, attention_live, attention_masked_softcap, causal_mask_from_pos,
};
use poot_graph_ir::{
    Builder, Graph, IndexRange, Outputs, Slot, TensorType, Traced, ValidationId, ValueId,
};
use poot_tensor::{DType, HostTensor};

use crate::fill;

/// The shape of one live-capacity attention decode.
#[derive(Clone, Copy, Debug)]
pub struct LiveAttentionFixture {
    pub batch: usize,
    pub hq: usize,
    pub hkv: usize,
    pub cap: usize,
    /// The query and key width (a split key's per-head part and shared part together).
    pub dk: usize,
    pub dv: usize,
    /// `Some(s)`: the key is `k_head [.., dk - s]` plus `k_shared [B,1,Lk,s]`.
    pub shared_dim: Option<usize>,
    pub softcap: Option<f32>,
    /// The dtype the attention computes in. The operands and the output stay F32 so every executor binds and
    /// reads them the same way: a narrow fixture narrows each operand (and the mask) with a `Cast` first and
    /// widens the output back, so the composite itself runs at `dtype`.
    pub dtype: DType,
}

/// `x` at `dtype`: itself when it already is, else a `Cast`.
fn cast_to(b: &Builder, x: Traced, dtype: DType) -> Traced {
    if b.aval(x).dtype == dtype {
        x
    } else {
        b.cast(x, dtype)
    }
}

/// The built graph and the value ids its operands bind to.
pub struct LiveAttentionGraph {
    pub graph: Graph,
    pub q: ValueId,
    /// The key, or the per-head part of a split key.
    pub k: ValueId,
    pub k_shared: Option<ValueId>,
    pub v: ValueId,
    pub pos: ValueId,
}

impl LiveAttentionFixture {
    fn scale(&self) -> f32 {
        1.0 / (self.dk as f32).sqrt()
    }

    /// The decode graph over the composite; `validated` declares the length witness as a validation output.
    pub fn build(&self, validated: bool) -> LiveAttentionGraph {
        let Self {
            batch,
            hq,
            hkv,
            cap,
            dk,
            dv,
            shared_dim,
            softcap,
            dtype,
        } = *self;
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![batch, hq, 1, dk]));
        let v = b.constant("v", TensorType::f32(vec![batch, hkv, cap, dv]));
        // Every position an `f32` holds exactly binds, so the in-graph length guard (not the binder)
        // is what rejects a length outside `1..=cap`; the mask's `I32 -> F32` cast of `pos` is proven
        // exact from this range.
        let pos = b.slot_ranged(
            Slot::Pos,
            TensorType::new(vec![batch, 1], DType::I32),
            IndexRange::F32_EXACT,
        );
        let head_dim = dk - shared_dim.unwrap_or(0);
        let k = b.constant("k", TensorType::f32(vec![batch, hkv, cap, head_dim]));
        let shared = shared_dim
            .map(|width| b.constant("k_shared", TensorType::f32(vec![batch, 1, cap, width])));
        let narrow = |x| cast_to(&b, x, dtype);
        let keys = match shared {
            Some(shared) => AttentionKeys::Split {
                head: narrow(k),
                shared: narrow(shared),
            },
            None => AttentionKeys::Single(narrow(k)),
        };
        let mask = narrow(causal_mask_from_pos(&b, pos, cap, None));
        let len = b.binary_scalar(
            poot_graph_ir::BinOp::Add,
            b.reshape(pos, vec![batch]),
            poot_graph_ir::Scalar::I32(1),
        );
        let live = attention_live(
            &b,
            narrow(q),
            keys,
            narrow(v),
            hq / hkv,
            self.scale(),
            mask,
            softcap,
            len,
        )
        .expect("the live composite builds");
        let out = cast_to(&b, live.out, DType::F32);
        let outputs = if validated {
            Outputs::new(out).with_witness(ValidationId(1), "attention.len", live.witness)
        } else {
            Outputs::new(out)
        };
        let graph = b
            .finish(outputs)
            .expect("the live composite is a valid graph");
        LiveAttentionGraph {
            graph,
            q: q.id(),
            k: k.id(),
            k_shared: shared.map(|shared| shared.id()),
            v: v.id(),
            pos: pos.id(),
        }
    }

    /// The finite operands for per-row live lengths `len` (the bound `pos` is `len - 1`). With `poison_tail`,
    /// every key and value row at or past a row's length holds NaN.
    pub fn inputs(&self, graph: &LiveAttentionGraph, len: &[i32], poison_tail: bool) -> Inputs {
        let Self {
            batch,
            hq,
            hkv,
            cap,
            dk,
            dv,
            shared_dim,
            ..
        } = *self;
        let head_dim = dk - shared_dim.unwrap_or(0);
        let mut k = fill(batch * hkv * cap * head_dim, 2);
        let mut v = fill(batch * hkv * cap * dv, 3);
        let mut shared = shared_dim.map(|width| fill(batch * cap * width, 4));
        if poison_tail {
            for (b, &live) in len.iter().enumerate() {
                for t in (live.max(0) as usize).min(cap)..cap {
                    for h in 0..hkv {
                        let row = (b * hkv + h) * cap + t;
                        k[row * head_dim..(row + 1) * head_dim].fill(f32::NAN);
                        v[row * dv..(row + 1) * dv].fill(f32::NAN);
                    }
                    if let (Some(shared), Some(width)) = (shared.as_mut(), shared_dim) {
                        let row = b * cap + t;
                        shared[row * width..(row + 1) * width].fill(f32::NAN);
                    }
                }
            }
        }
        let mut tensors = HashMap::from([
            (
                graph.q,
                HostTensor::f32(vec![batch, hq, 1, dk], fill(batch * hq * dk, 1)),
            ),
            (graph.k, HostTensor::f32(vec![batch, hkv, cap, head_dim], k)),
            (graph.v, HostTensor::f32(vec![batch, hkv, cap, dv], v)),
            (
                graph.pos,
                HostTensor::i32(
                    vec![batch, 1],
                    len.iter().map(|&len| len.wrapping_sub(1)).collect(),
                ),
            ),
        ]);
        if let (Some(id), Some(data), Some(width)) = (graph.k_shared, shared, shared_dim) {
            tensors.insert(id, HostTensor::f32(vec![batch, 1, cap, width], data));
        }
        Inputs(tensors)
    }

    /// The plain attention over the first `len[b]` keys of each batch row (`1 <= len[b] <= cap`): one ordinary
    /// materialized graph per row, over that row's key prefix with the split key written out in full.
    pub fn oracle(&self, inputs: &Inputs, graph: &LiveAttentionGraph, len: &[i32]) -> Vec<f32> {
        let Self {
            hq,
            hkv,
            cap,
            dk,
            dv,
            shared_dim,
            softcap,
            dtype,
            ..
        } = *self;
        let head_dim = dk - shared_dim.unwrap_or(0);
        let f32s = |id: ValueId| inputs.0[&id].as_f32().expect("an f32 operand").to_vec();
        let (q, k, v) = (f32s(graph.q), f32s(graph.k), f32s(graph.v));
        let shared = graph.k_shared.map(f32s);
        let mut out = Vec::new();
        for (b, &live) in len.iter().enumerate() {
            let n = live as usize;
            assert!((1..=cap).contains(&n), "the oracle needs 1 <= len <= cap");
            let mut keys = Vec::with_capacity(hkv * n * dk);
            let mut values = Vec::with_capacity(hkv * n * dv);
            for h in 0..hkv {
                for t in 0..n {
                    let row = (b * hkv + h) * cap + t;
                    keys.extend_from_slice(&k[row * head_dim..(row + 1) * head_dim]);
                    if let (Some(shared), Some(width)) = (&shared, shared_dim) {
                        let row = b * cap + t;
                        keys.extend_from_slice(&shared[row * width..(row + 1) * width]);
                    }
                    values.extend_from_slice(&v[row * dv..(row + 1) * dv]);
                }
            }
            let g = Builder::new();
            let qi = g.constant("q", TensorType::f32(vec![1, hq, 1, dk]));
            let ki = g.constant("k", TensorType::f32(vec![1, hkv, n, dk]));
            let vi = g.constant("v", TensorType::f32(vec![1, hkv, n, dv]));
            let mi = g.constant("mask", TensorType::f32(vec![1, 1, 1, n]));
            let attended = attention_masked_softcap(
                &g,
                cast_to(&g, qi, dtype),
                cast_to(&g, ki, dtype),
                cast_to(&g, vi, dtype),
                hq / hkv,
                self.scale(),
                cast_to(&g, mi, dtype),
                softcap,
            );
            let attended = cast_to(&g, attended, DType::F32);
            let graph = g.finish(attended).unwrap();
            let bound = HashMap::from([
                (
                    qi.id(),
                    Value::from(HostTensor::f32(
                        vec![1, hq, 1, dk],
                        q[b * hq * dk..(b + 1) * hq * dk].to_vec(),
                    )),
                ),
                (
                    ki.id(),
                    Value::from(HostTensor::f32(vec![1, hkv, n, dk], keys)),
                ),
                (
                    vi.id(),
                    Value::from(HostTensor::f32(vec![1, hkv, n, dv], values)),
                ),
                (
                    mi.id(),
                    Value::from(HostTensor::f32(vec![1, 1, 1, n], vec![0.0; n])),
                ),
            ]);
            let row = eval(&graph, &bound, EvalOptions::new(EvalBudget::UNBOUNDED))
                .expect("the oracle row evaluates")
                .output
                .into_host()
                .expect("a dense oracle row");
            out.extend_from_slice(row.as_f32().expect("an f32 oracle row"));
        }
        out
    }
}

/// The host tensors one fixture run binds, keyed by graph value.
pub struct Inputs(pub HashMap<ValueId, HostTensor>);

impl Inputs {
    /// The same tensors as the oracle's `Value` bindings.
    pub fn values(&self) -> HashMap<ValueId, Value> {
        self.0
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect()
    }
}

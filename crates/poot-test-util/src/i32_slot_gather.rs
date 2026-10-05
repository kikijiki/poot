//! Spike 562 F-9 (card 643): one I32 `Slot::Pos` that is a `Gather` index and also feeds another use (an
//! I32->F32 `Cast`, a `DynamicUpdateSlice` start, or both). Each backend's device row hands [`run_all`] a
//! closure that compiles a variant for its target and runs it on its resident executor; `run_all` compares
//! every output bit for bit.
//!
//! The ops are exact (a gather, an integer-valued cast, one add, a dynamic update slice and a slice), so
//! device, CPU oracle and the plain-Rust ground truth must agree bit for bit.

use poot_tensor::DType;
use std::collections::HashMap;

use poot_eval::Value;
use poot_graph_ir::{BinOp, Builder, Graph, Slot, StateRole, Storage, TensorType, ValueId};
use poot_tensor::HostTensor;

/// The step's positions: the `Slot::Pos` payload, the Gather rows and the DUS start (`POSITIONS[0]`).
const POSITIONS: [i32; 4] = [3, 4, 5, 6];
const ROWS: usize = 16;
const WIDTH: usize = 8;
const CACHE_ROWS: usize = 32;
/// The rows of the updated cache each state variant outputs (`slice(out, 0, 0, 8)`).
const OUT_ROWS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Use {
    /// v0: `gather + broadcast(cast(idx))`.
    Cast,
    /// v1: `dus(cache, gather, idx[0])`, sliced.
    Dus,
    /// v2: `dus(cache, gather + broadcast(cast(idx)), idx[0])`, sliced.
    CastDus,
}

/// One spike-562 micro graph with its input payloads.
pub struct I32SlotGatherVariant {
    name: &'static str,
    /// The traced graph. A backend compiles it for its own target.
    pub graph: Graph,
    kind: Use,
    table: Vec<f32>,
    cache: Vec<f32>,
}

/// The three spike-562 `slot_gather_plus_other_use` variants (v0, v1, v2).
fn i32_slot_gather_variants() -> Vec<I32SlotGatherVariant> {
    [
        ("slot_gather_plus_cast_v0", Use::Cast),
        ("slot_gather_plus_dus_v1", Use::Dus),
        ("slot_gather_plus_cast_and_dus_v2", Use::CastDus),
    ]
    .into_iter()
    .map(|(name, kind)| I32SlotGatherVariant {
        name,
        graph: build(kind),
        kind,
        table: (0..ROWS * WIDTH).map(|i| (i as f32 * 0.37).sin()).collect(),
        cache: (0..CACHE_ROWS * WIDTH)
            .map(|i| (i as f32 * 0.05).sin())
            .collect(),
    })
    .collect()
}

fn build(kind: Use) -> Graph {
    let t = POSITIONS.len();
    let b = Builder::new();
    let idx = b.slot(Slot::Pos, TensorType::new(vec![t], DType::I32));
    let table = b.constant("table", TensorType::f32(vec![ROWS, WIDTH]));
    let rows = b.gather(table, 0, idx);
    let update = if kind == Use::Dus {
        rows
    } else {
        let q = b.cast(idx, DType::F32);
        b.binary(
            BinOp::Add,
            rows,
            b.broadcast(b.reshape(q, vec![t, 1]), vec![t, WIDTH]),
        )
    };
    if kind == Use::Cast {
        return b.finish(update);
    }
    let start = b.reshape(b.slice(idx, 0, 0, 1), vec![]);
    let cache = b.state_input(
        "cache",
        TensorType::f32(vec![CACHE_ROWS, WIDTH]),
        StateRole::Recurrent,
    );
    let written = b.dynamic_update_slice_dyn(cache, update, start, 0);
    let out = b.slice(written, 0, 0, OUT_ROWS);
    b.finish_with_state(out, &[(cache, written)])
}

impl I32SlotGatherVariant {
    /// `graph`'s inputs: the traced graph or a compiled one, which may renumber values, so each input is
    /// bound by what it is, not by id.
    pub fn inputs(&self, graph: &Graph) -> HashMap<ValueId, Value> {
        graph
            .inputs
            .iter()
            .map(|&id| {
                let meta = graph.meta(id);
                let tensor = match (&meta.storage, meta.name.as_deref()) {
                    (Storage::Slot(Slot::Pos), _) => {
                        HostTensor::i32(vec![POSITIONS.len()], POSITIONS.to_vec())
                    }
                    (Storage::Const, Some("table")) => {
                        HostTensor::f32(vec![ROWS, WIDTH], self.table.clone())
                    }
                    (Storage::State, _) => {
                        HostTensor::f32(vec![CACHE_ROWS, WIDTH], self.cache.clone())
                    }
                    (Storage::Computed(c), _) => HostTensor::f32(c.shape(), c.values_f32()),
                    other => panic!("{}: unexpected input v{id}: {other:?}", self.name),
                };
                (id, Value::from(tensor))
            })
            .collect()
    }

    /// The seed of every carried state buffer, in `graph.state` order (the cache, or none for v0).
    pub fn state(&self) -> Vec<Vec<f32>> {
        match self.kind {
            Use::Cast => Vec::new(),
            Use::Dus | Use::CastDus => vec![self.cache.clone()],
        }
    }

    /// The expected output, from the variant's definition alone: gathered table rows (plus their position
    /// for the cast variants), written over cache rows `POSITIONS[0]..` for the DUS variants.
    fn ground_truth(&self) -> Vec<f32> {
        let row = |p: i32| {
            let p = p as usize;
            let add = if self.kind == Use::Dus { 0.0 } else { p as f32 };
            self.table[p * WIDTH..(p + 1) * WIDTH]
                .iter()
                .map(move |&x| x + add)
        };
        match self.kind {
            Use::Cast => POSITIONS.iter().flat_map(|&p| row(p)).collect(),
            Use::Dus | Use::CastDus => {
                let mut out = self.cache[..OUT_ROWS * WIDTH].to_vec();
                let start = POSITIONS[0] as usize;
                for (r, &p) in POSITIONS.iter().enumerate() {
                    let at = (start + r) * WIDTH;
                    for (dst, value) in out[at..at + WIDTH].iter_mut().zip(row(p)) {
                        *dst = value;
                    }
                }
                out
            }
        }
    }

    /// Compare `got` (a `backend` run of this variant) with the CPU oracle bit for bit, and the oracle with
    /// the ground truth bit for bit; the first difference is the error. `to_bits` equality also fails on NaN.
    fn check_bit_exact(&self, backend: &str, got: &HostTensor) -> Result<(), String> {
        let oracle = poot_eval::eval(
            &self.graph,
            &self.inputs(&self.graph),
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .map_err(|e| format!("CPU oracle: {e}"))?
        .output
        .into_host()
        .map_err(|e| format!("CPU oracle: {e}"))?;
        let truth = self.ground_truth();
        if oracle.as_f32().unwrap().len() != truth.len() {
            return Err(format!(
                "oracle has {} elements, ground truth {}",
                oracle.as_f32().unwrap().len(),
                truth.len()
            ));
        }
        if let Some((i, (o, t))) = oracle
            .as_f32()
            .unwrap()
            .iter()
            .zip(truth.iter())
            .enumerate()
            .find(|(_, (o, t))| o.to_bits() != t.to_bits())
        {
            return Err(format!("oracle elem {i}: {o} vs ground truth {t}"));
        }
        if got.shape() != oracle.shape() {
            return Err(format!(
                "{backend} shape {:?} vs cpu {:?}",
                got.shape(),
                oracle.shape()
            ));
        }
        match got
            .as_f32()
            .unwrap()
            .iter()
            .zip(oracle.as_f32().unwrap().iter())
            .enumerate()
            .find(|(_, (g, o))| g.to_bits() != o.to_bits())
        {
            Some((i, (g, o))) => Err(format!("{backend} elem {i}: {g} vs cpu {o}")),
            None => Ok(()),
        }
    }
}

/// Run every variant through `run` (the backend's compile + resident run of the variant's graph, returning
/// the output) and check each bit for bit. Every variant runs even after one fails, so the panic names each
/// failing variant with its first difference or its compile/run error.
pub fn run_all(
    backend: &str,
    mut run: impl FnMut(&I32SlotGatherVariant) -> Result<HostTensor, String>,
) {
    let failures: Vec<String> = i32_slot_gather_variants()
        .iter()
        .filter_map(|variant| {
            run(variant)
                .and_then(|got| variant.check_bit_exact(backend, &got))
                .err()
                .map(|e| format!("{}: {e}", variant.name))
        })
        .collect();
    assert!(failures.is_empty(), "{backend}: {failures:#?}");
}

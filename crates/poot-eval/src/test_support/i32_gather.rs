//! POOT-725: `Gather` with I32 indices, one graph per case, bound and compared bit for bit.
//!
//! The cases cover the movement generator's index forms: axis 0 and axis 1, F32, BF16 and I32 data, and
//! indices that repeat and sit on both edges of the axis. The index is a `Slot::Token` input, bound as raw
//! I32 words; the table is the `"table"` const. A device row compiles each [`GatherCase::graph`] for its
//! target, runs it through the executor contract and hands the output to [`GatherCase::check`], which
//! compares the output words with the CPU oracle's and with plain Rust ground truth.
//!
//! The table words are chosen so a read through the wrong lane differs visibly: F32 words include `-0.0` and
//! a subnormal, BF16 words are arbitrary 16-bit patterns, I32 words need all 32 bits.

use std::collections::HashMap;

use crate::{EvalBudget, EvalOptions, Value};
use poot_graph_ir::{Builder, Graph, Slot, Storage, TensorType, ValueId};
use poot_tensor::{DType, HostTensor};

/// One gather: its traced graph, the table it reads and the index slot's payload.
pub struct GatherCase {
    pub name: String,
    /// The traced graph. A backend compiles it for its own target.
    pub graph: Graph,
    dtype: DType,
    table_shape: Vec<usize>,
    /// The table's elements, one `u32` per element: the bits of an F32 or I32 word, the 16 bits of a BF16 one.
    table: Vec<u32>,
    axis: usize,
    index: Vec<i32>,
}

/// The table words of one data dtype: `count` distinct patterns, none a NaN.
fn words(dtype: DType, count: usize) -> Vec<u32> {
    (0..count)
        .map(|i| match dtype {
            DType::F32 => match i {
                0 => (-0.0f32).to_bits(),
                1 => 1u32,
                _ => (((i as f32) * 0.37).sin() * 100.0).to_bits(),
            },
            DType::BF16 => 0x3c00 + 0x0111 * i as u32,
            DType::I32 => match i {
                0 => i32::MIN as u32,
                n if n + 1 == count => i32::MAX as u32,
                n => (1u32 << 24) + 1 + 2 * n as u32,
            },
            other => unreachable!("no gather case over {other:?}"),
        })
        .collect()
}

fn build(dtype: DType, table_shape: &[usize], axis: usize, index: &[i32]) -> Graph {
    let b = Builder::for_tests();
    let indices = b
        .slot(Slot::Token, TensorType::new(vec![index.len()], DType::I32))
        .unwrap();
    let table = b
        .constant("table", TensorType::new(table_shape.to_vec(), dtype))
        .unwrap();
    let rows = b.gather(table, axis, indices).unwrap();
    // The oracle holds a BF16 gather's result as an exact value, not a host tensor, and a real BF16
    // table is read through a widening cast (`fold_dense_bf16_row_gathers`): widen the rows, exactly.
    let out = if dtype == DType::BF16 {
        b.cast(rows, DType::F32).unwrap()
    } else {
        rows
    };
    b.finish(out).unwrap()
}

/// Every case: axis 0 and axis 1 over F32 and I32 data, and over BF16 data when `bf16` (a backend whose
/// BF16 table const has a native lane; wgpu keeps BF16 consts packed and refuses a plain gather of one).
pub fn cases(bf16: bool) -> Vec<GatherCase> {
    let mut dtypes = vec![DType::F32, DType::I32];
    if bf16 {
        dtypes.push(DType::BF16);
    }
    let mut cases = Vec::new();
    for dtype in dtypes {
        // Axis 0: [6, 4] rows, repeated (5, 0, 5) and edge (0, 5) indices.
        // Axis 1: [3, 6, 2], the same edges on the middle axis.
        for (axis, table_shape, index) in [
            (0usize, vec![6usize, 4], vec![5, 0, 5, 2, 0]),
            (1usize, vec![3, 6, 2], vec![5, 0, 5, 3]),
        ] {
            let count = table_shape.iter().product();
            cases.push(GatherCase {
                name: format!("gather_axis{axis}_{dtype}_i32_index"),
                graph: build(dtype, &table_shape, axis, &index),
                dtype,
                table_shape,
                table: words(dtype, count),
                axis,
                index,
            });
        }
    }
    cases
}

impl GatherCase {
    fn table_tensor(&self) -> HostTensor {
        match self.dtype {
            DType::F32 => HostTensor::f32(
                self.table_shape.clone(),
                self.table.iter().map(|&w| f32::from_bits(w)).collect(),
            ),
            DType::I32 => HostTensor::i32(
                self.table_shape.clone(),
                self.table.iter().map(|&w| w as i32).collect(),
            ),
            DType::BF16 => HostTensor::bf16(
                self.table_shape.clone(),
                self.table.iter().map(|&w| w as u16).collect(),
            ),
            other => unreachable!("no gather case over {other:?}"),
        }
    }

    /// `graph`'s inputs, each bound by what it is: the index slot and the `"table"` const.
    pub fn inputs(&self, graph: &Graph) -> HashMap<ValueId, HostTensor> {
        graph
            .inputs()
            .iter()
            .map(|&id| {
                let tensor = match &graph.meta(id).storage {
                    Storage::Slot(Slot::Token) => {
                        HostTensor::i32(vec![self.index.len()], self.index.clone())
                    }
                    Storage::Const => self.table_tensor(),
                    other => panic!("{}: unexpected input v{id}: {other:?}", self.name),
                };
                (id, tensor)
            })
            .collect()
    }

    /// The gathered elements as output words, from the case's definition alone: out[outer, g, inner] =
    /// table[outer, index[g], inner] along `axis`.
    fn ground_truth(&self) -> Vec<u32> {
        let outer: usize = self.table_shape[..self.axis].iter().product();
        let extent = self.table_shape[self.axis];
        let inner: usize = self.table_shape[self.axis + 1..].iter().product();
        let mut out = Vec::with_capacity(outer * self.index.len() * inner);
        for o in 0..outer {
            for &i in &self.index {
                let base = (o * extent + i as usize) * inner;
                let row = &self.table[base..base + inner];
                if self.dtype == DType::BF16 {
                    // A BF16 word widened to F32 is the word in the high half.
                    out.extend(row.iter().map(|&w| w << 16));
                } else {
                    out.extend_from_slice(row);
                }
            }
        }
        out
    }

    /// The graph's output dtype: the table's, or F32 for BF16 data (see [`build`]).
    fn output_dtype(&self) -> DType {
        if self.dtype == DType::BF16 {
            DType::F32
        } else {
            self.dtype
        }
    }

    /// The elements of `t` as raw 32-bit words.
    fn words_of(&self, t: &HostTensor) -> Result<Vec<u32>, String> {
        if t.dtype() != self.output_dtype() {
            return Err(format!("dtype {}, want {}", t.dtype(), self.output_dtype()));
        }
        Ok(t.view()
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect())
    }

    /// Compare `got` (a `backend` run of [`Self::graph`]) with the ground truth word for word, and the CPU
    /// oracle with the same; the first difference is the error. Comparing words rather than values makes a
    /// NaN, `-0.0` or subnormal difference visible.
    pub fn check(&self, backend: &str, got: &HostTensor) -> Result<(), String> {
        let truth = self.ground_truth();
        let oracle = crate::eval(
            &self.graph,
            &self
                .inputs(&self.graph)
                .into_iter()
                .map(|(id, t)| (id, Value::from(t)))
                .collect(),
            EvalOptions::new(EvalBudget::UNBOUNDED),
        )
        .map_err(|e| format!("CPU oracle: {e}"))?
        .output
        .into_host()
        .map_err(|e| format!("CPU oracle: {e}"))?;
        let first_difference = |label: &str, t: &HostTensor| -> Result<(), String> {
            let words = self.words_of(t).map_err(|e| format!("{label}: {e}"))?;
            if words.len() != truth.len() {
                return Err(format!(
                    "{label} has {} elements, want {}",
                    words.len(),
                    truth.len()
                ));
            }
            match words
                .iter()
                .zip(&truth)
                .enumerate()
                .find(|(_, (g, t))| g != t)
            {
                Some((i, (g, t))) => Err(format!(
                    "{label} element {i}: {g:#010x} vs ground truth {t:#010x}"
                )),
                None => Ok(()),
            }
        };
        first_difference("CPU oracle", &oracle)?;
        first_difference(backend, got)
    }
}

/// Run every case through `run` (a backend's compile + executor-contract run of the case's graph) and check
/// each. Every case runs even after one fails, so the panic names each failing case with its first
/// difference or its compile/run error.
pub fn run_all(
    backend: &str,
    bf16: bool,
    mut run: impl FnMut(&GatherCase) -> Result<HostTensor, String>,
) {
    let failures: Vec<String> = cases(bf16)
        .iter()
        .filter_map(|case| {
            run(case)
                .and_then(|got| case.check(backend, &got))
                .err()
                .map(|e| format!("{}: {e}", case.name))
        })
        .collect();
    assert!(failures.is_empty(), "{backend}: {failures:#?}");
}

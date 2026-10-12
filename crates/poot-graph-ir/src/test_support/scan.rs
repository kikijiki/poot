//! The recurrence-scan fixture (POOT-634): a graph over the one `OpKind::Scan` equation whose `y` is the graph
//! output and whose `s_out` is the carried state, seeded operands for either family, and an independent
//! plain-Rust f64 reference (the token-by-token recurrence) that shares no code with the scan's chunked
//! definition.

use std::collections::HashMap;

use poot_tensor::HostTensor;
use poot_test_util::fill;

use crate::{Builder, Graph, ScanAttrs, ScanFamily, Slot, StateRole, TensorType, Traced, ValueId};

/// The shape of one scan. `width` is the head width (`D` for the delta rule, `P` for SSD); `state_rows` is
/// `D` for the delta rule (ignored: it equals `width`) and `N` for SSD.
#[derive(Clone, Copy, Debug)]
pub struct ScanFixture {
    pub family: ScanFamily,
    pub heads: usize,
    pub len: usize,
    pub width: usize,
    pub state_rows: usize,
    pub chunk: usize,
}

/// The operands of one scan call, flattened row-major, in the family's operand order (the state last).
#[derive(Clone, Debug)]
pub struct ScanInputs {
    pub operands: Vec<Vec<f32>>,
}

pub struct ScanGraph {
    pub graph: Graph,
    /// The operand ids in the family's operand order; the last is the carried state.
    pub operands: Vec<ValueId>,
}

/// How [`ScanFixture::graph_with`] declares the sequence operands.
#[derive(Clone, Copy)]
enum Inputs {
    Constants,
    Slots,
}

impl ScanFixture {
    pub fn delta_rule(heads: usize, len: usize, d: usize, chunk: usize) -> Self {
        Self {
            family: ScanFamily::DeltaRule,
            heads,
            len,
            width: d,
            state_rows: d,
            chunk,
        }
    }

    pub fn ssd(heads: usize, len: usize, p: usize, n: usize, chunk: usize) -> Self {
        Self {
            family: ScanFamily::Ssd,
            heads,
            len,
            width: p,
            state_rows: n,
            chunk,
        }
    }

    /// The carried state's rows: `D` for the delta rule, `N` for SSD.
    pub fn state_rows_of(&self) -> usize {
        match self.family {
            ScanFamily::DeltaRule => self.width,
            ScanFamily::Ssd => self.state_rows,
        }
    }

    fn attrs(&self) -> ScanAttrs {
        ScanAttrs {
            family: self.family,
            chunk: self.chunk,
        }
    }

    /// The operand names and shapes, in the family's operand order; the last is the state.
    pub fn operand_shapes(&self) -> Vec<(&'static str, Vec<usize>)> {
        let (h, l, w, r) = (self.heads, self.len, self.width, self.state_rows);
        match self.family {
            ScanFamily::DeltaRule => vec![
                ("q", vec![1, h, l, w]),
                ("k", vec![1, h, l, w]),
                ("v", vec![1, h, l, w]),
                ("g", vec![1, h, l, 1]),
                ("beta", vec![1, h, l, 1]),
                ("s", vec![1, h, w, w]),
            ],
            ScanFamily::Ssd => vec![
                ("x", vec![1, h, l, w]),
                ("b", vec![1, h, l, r]),
                ("c", vec![1, h, l, r]),
                ("delta", vec![1, h, l, 1]),
                ("a", vec![1, h, 1, 1]),
                ("d", vec![1, h, 1, 1]),
                ("s", vec![1, h, r, w]),
            ],
        }
    }

    /// Realistic operands: L2-normed q/k, a mild negative log-gate and beta in (0,1) for the delta rule; a
    /// small positive `delta` with a negative per-head `A` for SSD; a nonzero carried state either way.
    pub fn inputs(&self, seed: u64) -> ScanInputs {
        let Self { heads, len, .. } = *self;
        let unit_rows = |mut raw: Vec<f32>, width: usize| {
            for row in raw.chunks_mut(width) {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                row.iter_mut().for_each(|x| *x /= norm);
            }
            raw
        };
        let shapes = self.operand_shapes();
        let numel = |i: usize| shapes[i].1.iter().product::<usize>();
        let width = |i: usize| *shapes[i].1.last().unwrap();
        let operands = match self.family {
            ScanFamily::DeltaRule => vec![
                unit_rows(fill(numel(0), seed + 1), width(0)),
                unit_rows(fill(numel(1), seed + 2), width(1)),
                fill(numel(2), seed + 3),
                (0..heads * len)
                    .map(|i| -0.05 - 0.03 * ((i * 5 + i / len) % 4) as f32)
                    .collect(),
                (0..heads * len)
                    .map(|i| 0.3 + 0.5 * (((i * 3 + i / len) % 4) as f32 / 4.0))
                    .collect(),
                fill(numel(5), seed + 4),
            ],
            ScanFamily::Ssd => vec![
                fill(numel(0), seed + 1),
                fill(numel(1), seed + 2)
                    .into_iter()
                    .map(|x| x * 0.5)
                    .collect(),
                fill(numel(2), seed + 3)
                    .into_iter()
                    .map(|x| x * 0.5)
                    .collect(),
                (0..heads * len)
                    .map(|i| 0.05 + 0.1 * (((i * 7 + i / len) % 5) as f32 / 5.0))
                    .collect(),
                (0..heads).map(|h| -(0.4 + 0.3 * h as f32)).collect(),
                (0..heads).map(|h| 0.5 + 0.1 * h as f32).collect(),
                fill(numel(6), seed + 4)
                    .into_iter()
                    .map(|x| x * 0.3)
                    .collect(),
            ],
        };
        ScanInputs { operands }
    }

    /// `y = scan(..)` as the output, `s_out` carried over the state operand; every operand a constant.
    pub fn graph(&self) -> ScanGraph {
        self.graph_with(Inputs::Constants, false)
    }

    /// [`Self::graph`] with the sequence operands as per-step activation slots (so a device entry can be
    /// stepped with fresh sequence data each time) and the state carried.
    pub fn slot_graph(&self) -> ScanGraph {
        self.graph_with(Inputs::Slots, false)
    }

    /// [`Self::graph`] publishing the whole packed result (`y` rows then `s_out` rows) as the output, so one
    /// comparison covers every output and carried-state element.
    pub fn packed_graph(&self) -> ScanGraph {
        self.graph_with(Inputs::Constants, true)
    }

    /// [`Self::slot_graph`] publishing the whole packed result.
    pub fn packed_slot_graph(&self) -> ScanGraph {
        self.graph_with(Inputs::Slots, true)
    }

    fn graph_with(&self, inputs: Inputs, packed: bool) -> ScanGraph {
        let b = Builder::for_tests();
        let shapes = self.operand_shapes();
        let (state, sequence) = shapes.split_last().expect("a scan has operands");
        let mut operands: Vec<Traced> = sequence
            .iter()
            .map(|(name, shape)| {
                let ty = TensorType::f32(shape.clone());
                match inputs {
                    Inputs::Constants => b.constant(name, ty).unwrap(),
                    Inputs::Slots => b.slot_named(Slot::Activation, name, ty).unwrap(),
                }
            })
            .collect();
        let s_in = b
            .state_input(
                state.0,
                TensorType::f32(state.1.clone()),
                StateRole::Recurrent,
            )
            .unwrap();
        operands.push(s_in);
        let result = b.scan(self.attrs(), &operands).unwrap();
        let ids = operands.iter().map(|t| t.id()).collect();
        let output = if packed { result.packed } else { result.y };
        let graph = b
            .finish(crate::Outputs::new(output).with_state(&[(s_in, result.s_out)]))
            .unwrap();
        ScanGraph {
            graph,
            operands: ids,
        }
    }

    /// The host tensors for `graph`'s operands.
    pub fn bind(&self, graph: &ScanGraph, inputs: &ScanInputs) -> HashMap<ValueId, HostTensor> {
        self.operand_shapes()
            .into_iter()
            .zip(&graph.operands)
            .zip(&inputs.operands)
            .map(|(((_, shape), &id), data)| (id, HostTensor::f32(shape, data.clone())))
            .collect()
    }

    /// `inputs` with the sequence operands cut to the `count` positions from `from` (the carried state and
    /// the per-head scalars are kept), for stepping a long sequence in pieces.
    pub fn window(&self, inputs: &ScanInputs, from: usize, count: usize) -> ScanInputs {
        let per_position = match self.family {
            ScanFamily::DeltaRule => 5,
            ScanFamily::Ssd => 4,
        };
        let operands = self
            .operand_shapes()
            .iter()
            .zip(&inputs.operands)
            .enumerate()
            .map(|(i, ((_, shape), data))| {
                if i >= per_position {
                    return data.clone();
                }
                let width = shape[3];
                (0..self.heads)
                    .flat_map(|h| {
                        let start = (h * self.len + from) * width;
                        data[start..start + count * width].to_vec()
                    })
                    .collect()
            })
            .collect();
        ScanInputs { operands }
    }

    /// The token-by-token recurrence in f64. Returns `(y [H,L,W], s_out [H,R,W])`, flattened.
    pub fn reference(&self, inputs: &ScanInputs) -> (Vec<f32>, Vec<f32>) {
        let Self {
            heads,
            len,
            width,
            state_rows: rows,
            ..
        } = *self;
        let rows = match self.family {
            ScanFamily::DeltaRule => width,
            ScanFamily::Ssd => rows,
        };
        let op = &inputs.operands;
        let at = |i: usize, j: usize| f64::from(op[i][j]);
        let state_index = op.len() - 1;
        let mut state: Vec<f64> = op[state_index].iter().map(|&x| f64::from(x)).collect();
        let mut y = vec![0.0f32; heads * len * width];
        for h in 0..heads {
            let s = &mut state[h * rows * width..(h + 1) * rows * width];
            for t in 0..len {
                let pos = h * len + t;
                match self.family {
                    ScanFamily::DeltaRule => {
                        let row = pos * width;
                        let scale = 1.0 / (width as f64).sqrt();
                        let decay = at(3, pos).exp();
                        let beta = at(4, pos);
                        s.iter_mut().for_each(|x| *x *= decay);
                        for j in 0..width {
                            let kv: f64 =
                                (0..width).map(|r| at(1, row + r) * s[r * width + j]).sum();
                            let delta = beta * (at(2, row + j) - kv);
                            for r in 0..width {
                                s[r * width + j] += at(1, row + r) * delta;
                            }
                        }
                        for j in 0..width {
                            let o: f64 = (0..width)
                                .map(|r| at(0, row + r) * scale * s[r * width + j])
                                .sum();
                            y[row + j] = o as f32;
                        }
                    }
                    ScanFamily::Ssd => {
                        let (x_row, bc_row) = (pos * width, pos * rows);
                        let delta = at(3, pos);
                        let abar = (delta * at(4, h)).exp();
                        for j in 0..width {
                            let mut acc = 0.0;
                            for r in 0..rows {
                                let cell = &mut s[r * width + j];
                                *cell = abar * *cell + delta * at(1, bc_row + r) * at(0, x_row + j);
                                acc += at(2, bc_row + r) * *cell;
                            }
                            y[x_row + j] = (acc + at(5, h) * at(0, x_row + j)) as f32;
                        }
                    }
                }
            }
        }
        (y, state.into_iter().map(|x| x as f32).collect())
    }
}

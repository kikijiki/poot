//! The selected-attention fixture (POOT-1143 stage 3): an independent f64 attention over exactly the keys a
//! block selection names, which shares no code with the op's dense definition or with the device kernel,
//! and the selection cases the definition, the kernel and every executor are exercised on.

use super::block_top_k::{grid_scores, reference_row};

/// A selected-attention case: shapes and the per-query selection. Single dense key, `dk == dv == d`.
#[derive(Clone, Debug)]
pub struct Case {
    /// What the case exercises.
    pub label: String,
    /// Batch rows.
    pub batch: usize,
    /// Query heads.
    pub hq: usize,
    /// Query heads per key/value head.
    pub n_rep: usize,
    /// Query rows per batch row.
    pub lq: usize,
    /// Stored key positions.
    pub lk: usize,
    /// Head width.
    pub d: usize,
    /// Positions per block.
    pub ratio: usize,
    /// Selected blocks per query.
    pub k: usize,
    /// `[batch * lq]` absolute positions.
    pub pos: Vec<i32>,
    /// `[batch * lq * k]` selected block ids, ascending per query, `-1` padded.
    pub ids: Vec<i32>,
}

impl Case {
    /// Key/value heads.
    pub fn hkv(&self) -> usize {
        self.hq / self.n_rep
    }

    /// The `1 / sqrt(d)` scale the cases use.
    pub fn scale(&self) -> f32 {
        1.0 / (self.d as f32).sqrt()
    }

    /// Deterministic `q`, `k`, `v` in `[-1, 1)`, three different tensors: `fill` makes a seed odd, so seeds
    /// that are one apart would draw the same values twice and a swapped `k` and `v` would go unseen.
    pub fn operands(&self, seed: u64) -> [Vec<f32>; 3] {
        [
            fill(self.batch * self.hq * self.lq * self.d, seed),
            fill(self.batch * self.hkv() * self.lk * self.d, seed + 2),
            fill(self.batch * self.hkv() * self.lk * self.d, seed + 4),
        ]
    }

    /// The keys query `q` (a flat index into `pos`) reads: the positions of its selected blocks and of its
    /// own unfinished block, at or before it and inside the stored keys.
    pub fn visible(&self, q: usize) -> Vec<usize> {
        let p = self.pos[q].max(0) as usize;
        let tail = (p + 1) / self.ratio * self.ratio;
        let mut keys: Vec<usize> = self.ids[q * self.k..(q + 1) * self.k]
            .iter()
            .filter(|&&id| id >= 0)
            .flat_map(|&id| (0..self.ratio).map(move |o| id as usize * self.ratio + o))
            .chain(tail..=p)
            .filter(|&j| j <= p && j < self.lk)
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// The f64 reference output `[batch, hq, lq, d]`: softmax over the visible keys of `q . k * scale`,
    /// weighting the value rows.
    pub fn reference(&self, q_all: &[f32], k_all: &[f32], v_all: &[f32]) -> Vec<f64> {
        let (d, scale) = (self.d, 1.0 / (self.d as f64).sqrt());
        let mut out = vec![0.0; self.batch * self.hq * self.lq * d];
        for b in 0..self.batch {
            for h in 0..self.hq {
                for i in 0..self.lq {
                    let row = (b * self.hq + h) * self.lq + i;
                    let keys = self.visible(b * self.lq + i);
                    let kv = b * self.hkv() + h / self.n_rep;
                    let scores: Vec<f64> = keys
                        .iter()
                        .map(|&j| {
                            (0..d)
                                .map(|c| {
                                    f64::from(q_all[row * d + c])
                                        * f64::from(k_all[(kv * self.lk + j) * d + c])
                                })
                                .sum::<f64>()
                                * scale
                        })
                        .collect();
                    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for c in 0..d {
                        out[row * d + c] = keys
                            .iter()
                            .zip(&weights)
                            .map(|(&j, w)| w / total * f64::from(v_all[(kv * self.lk + j) * d + c]))
                            .sum();
                    }
                }
            }
        }
        out
    }
}

/// Deterministic values in `[-1, 1)`.
pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// The ids the plain selection keeps, per query, for scores seeded per query: ascending, `-1` padded.
pub fn selected_ids(pos: &[i32], blocks: usize, ratio: usize, k: usize, levels: u32) -> Vec<i32> {
    pos.iter()
        .enumerate()
        .flat_map(|(q, &p)| {
            let scores = grid_scores(blocks, levels, 7 + q as u64);
            reference_row(&scores, p, ratio, k)[..k].to_vec()
        })
        .collect()
}

/// The cases: decode at two depths over a GQA pair, a prefill whose first queries have no complete block
/// (an empty selection) or sit on a block boundary, a selection of every complete block (dense causal), and a
/// `-1`-padded selection that is shorter than `k`.
pub fn cases() -> Vec<Case> {
    let (ratio, blocks) = (4, 10);
    let lk = ratio * blocks;
    let mut cases = Vec::new();
    let pos = vec![37, 22];
    cases.push(Case {
        label: "decode".into(),
        batch: 2,
        hq: 4,
        n_rep: 2,
        lq: 1,
        lk,
        d: 6,
        ratio,
        k: 3,
        ids: selected_ids(&pos, blocks, ratio, 3, 1 << 20),
        pos,
    });
    let pos = vec![0, 2, 3, 4, 7, 8, 39];
    cases.push(Case {
        label: "prefill with short selections".into(),
        batch: 1,
        hq: 2,
        n_rep: 1,
        lq: pos.len(),
        lk,
        d: 5,
        ratio,
        k: 2,
        ids: selected_ids(&pos, blocks, ratio, 2, 1 << 20),
        pos,
    });
    let pos: Vec<i32> = (0..lk as i32).collect();
    cases.push(Case {
        label: "every block selected".into(),
        batch: 1,
        hq: 2,
        n_rep: 2,
        lq: lk,
        lk,
        d: 4,
        ratio,
        k: blocks,
        ids: selected_ids(&pos, blocks, ratio, blocks, 1 << 20),
        pos,
    });
    // A prefill chunk of a wide GQA group (six query heads per key/value head, as a block-selected layer's
    // workgroup serves them together): two batch rows of five queries, one on the first block, one before any
    // block completes, the rest deep.
    let pos = vec![0, 2, 19, 20, 39, 3, 11, 24, 31, 38];
    cases.push(Case {
        label: "prefill of a wide GQA group".into(),
        batch: 2,
        hq: 12,
        n_rep: 6,
        lq: 5,
        lk,
        d: 8,
        ratio,
        k: 3,
        ids: selected_ids(&pos, blocks, ratio, 3, 1 << 20),
        pos,
    });
    cases
}

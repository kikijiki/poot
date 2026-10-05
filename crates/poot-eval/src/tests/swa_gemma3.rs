//! B2 sliding-window attention oracle, gemma3 per-layer mask selection, softcap config wiring.

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::attention_prefill;
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::max_abs_error;

// ---- B2: sliding-window attention correctness (eval oracle) ----

/// Build and evaluate a tiny prefill attention graph with a given additive mask tensor.
/// Q/K/V are `[1, hq, l, d]`; mask is `[1, 1, l, l]`. Returns the `[1,hq,l,d]` output.
#[allow(clippy::too_many_arguments)]
fn eval_prefill_attn(
    q_data: Vec<f32>,
    k_data: Vec<f32>,
    v_data: Vec<f32>,
    mask_data: Vec<f32>,
    hq: usize,
    hkv: usize,
    l: usize,
    d: usize,
) -> Vec<f32> {
    let b = Builder::new();
    let scale = 1.0 / (d as f32).sqrt();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let n_rep = hq / hkv;
    let out = attention_prefill(&b, q, k, v, n_rep, scale, mask);
    let g = b.finish(out);
    let mut inputs = HashMap::new();
    inputs.insert(q.id, HostTensor::f32(vec![1, hq, l, d], q_data));
    inputs.insert(k.id, HostTensor::f32(vec![1, hkv, l, d], k_data));
    inputs.insert(v.id, HostTensor::f32(vec![1, hkv, l, d], v_data));
    inputs.insert(mask.id, HostTensor::f32(vec![1, 1, l, l], mask_data));
    let out_t = (|| -> Result<HostTensor, EvalError> {
        let values: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?
            .output
            .into_host()
    })()
    .unwrap();
    out_t.as_f32().unwrap().to_vec()
}

/// Hand-roll softmax-attention for one query row `i` over all keys, given the additive mask row.
fn ref_attn_row(
    q_row: &[f32],    // [d]
    k_mat: &[f32],    // [l, d]
    v_mat: &[f32],    // [l, d]
    mask_row: &[f32], // [l] additive mask for query i
    l: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0 / (d as f32).sqrt();
    // raw scores: q @ k^T
    let mut scores: Vec<f32> = (0..l)
        .map(|j| {
            let kj = &k_mat[j * d..(j + 1) * d];
            q_row.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale + mask_row[j]
        })
        .collect();
    // softmax
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for s in &mut scores {
        *s = (*s - max).exp();
    }
    let denom: f32 = scores.iter().sum();
    for s in &mut scores {
        *s /= denom;
    }
    // weighted sum of V rows
    let mut out = vec![0.0f32; d];
    for (j, &w) in scores.iter().enumerate() {
        let vj = &v_mat[j * d..(j + 1) * d];
        for (o, &val) in out.iter_mut().zip(vj) {
            *o += w * val;
        }
    }
    out
}

/// B2 eval oracle: a windowed causal mask gives correct outputs that differ from full causal.
///
/// L=6, 1 head, head_dim=4, window=3. Checks:
/// 1. Windowed output differs from full-causal at query rows i >= w (out-of-window keys excluded).
/// 2. Windowed output matches a hand-rolled reference for every row.
/// 3. Rows 0..w-1, where the window covers the whole causal range, are identical to full-causal.
#[test]
fn swa_windowed_mask_eval_oracle_correctness() {
    let (l, hq, hkv, d) = (6usize, 2, 2, 4);
    let w = 3usize; // window width

    // Deterministic fill (same as the `fill` helper above).
    let seed_fill = |n: usize, seed: u64| -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    };

    let q_data = seed_fill(hq * l * d, 1);
    let k_data = seed_fill(hkv * l * d, 2);
    let v_data = seed_fill(hkv * l * d, 3);

    // Full-causal mask [1,1,l,l]: 0 on/below diagonal, -1e30 above.
    let full_mask: Vec<f32> = (0..l * l)
        .map(|idx| {
            let (i, j) = (idx / l, idx % l);
            if j <= i { 0.0 } else { -1e30 }
        })
        .collect();

    // Windowed mask [1,1,l,l]: 0 in [i-w+1, i], -1e30 elsewhere.
    let win_mask: Vec<f32> = (0..l * l)
        .map(|idx| {
            let (i, j) = (idx / l, idx % l);
            if j <= i && i - j < w { 0.0 } else { -1e30 }
        })
        .collect();

    let out_full = eval_prefill_attn(
        q_data.clone(),
        k_data.clone(),
        v_data.clone(),
        full_mask.clone(),
        hq,
        hkv,
        l,
        d,
    );
    let out_win = eval_prefill_attn(
        q_data.clone(),
        k_data.clone(),
        v_data.clone(),
        win_mask.clone(),
        hq,
        hkv,
        l,
        d,
    );

    assert_eq!(out_full.len(), hq * l * d);
    assert_eq!(out_win.len(), hq * l * d);

    // Check 1: windowed output differs from full-causal for rows i >= w. Row i is the range
    // [i*d..(i+1)*d] inside each head slice [h*l*d..(h+1)*l*d].
    let mut found_diff = false;
    for h in 0..hq {
        let base = h * l * d;
        for i in w..l {
            // Row i: at least one out-of-window key (j < i-w+1) was visible in full but not windowed.
            let full_row = &out_full[base + i * d..base + (i + 1) * d];
            let win_row = &out_win[base + i * d..base + (i + 1) * d];
            let diff: f32 = full_row
                .iter()
                .zip(win_row)
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff > 1e-6,
                "head {h} row {i}: windowed must differ from full-causal (diff={diff:.2e})"
            );
            found_diff = true;
        }
    }
    assert!(found_diff, "at least one row should differ");

    // Check 2: windowed output matches the hand-rolled reference, applying the window per (head, query_row).
    let tol = 1e-5f32;
    for h in 0..hq {
        let q_h = &q_data[h * l * d..(h + 1) * l * d];
        let k_h = &k_data[h * l * d..(h + 1) * l * d]; // hkv == hq so no GQA here
        let v_h = &v_data[h * l * d..(h + 1) * l * d];
        for i in 0..l {
            let win_row = &win_mask[i * l..(i + 1) * l];
            let ref_row = ref_attn_row(&q_h[i * d..(i + 1) * d], k_h, v_h, win_row, l, d);
            let got_row = &out_win[h * l * d + i * d..h * l * d + (i + 1) * d];
            for (c, (&r, &g)) in ref_row.iter().zip(got_row).enumerate() {
                let err = (r - g).abs();
                assert!(
                    err <= tol,
                    "head {h} row {i} col {c}: ref={r:.6} got={g:.6} err={err:.2e} > tol={tol}"
                );
            }
        }
    }

    // Check 3: for i < w the window [i-w+1, i] covers the whole causal range [0, i], so the rows are equal.
    for h in 0..hq {
        let base = h * l * d;
        for i in 0..w.min(l) {
            let full_row = &out_full[base + i * d..base + (i + 1) * d];
            let win_row = &out_win[base + i * d..base + (i + 1) * d];
            for (c, (&f, &wv)) in full_row.iter().zip(win_row).enumerate() {
                let err = (f - wv).abs();
                assert!(
                    err <= tol,
                    "head {h} row {i} col {c}: early rows must be equal (diff={err:.2e})"
                );
            }
        }
    }
}

/// Card 550 SC-004: `causal_mask_from_pos`'s single-query-position output (`[1,1,1,cap]`, the decode shape)
/// equals the pre-card host-built mask row (poot-llm's old `decode_mask_row`: 0 for `t <= position &&
/// position - t < window`, else `-1e9`) bit for bit, over a table of `(position, capacity, window)` cases
/// covering: window=None (full causal), a real window strictly inside `[0, position]`, a window exactly
/// covering the causal range (`window == position + 1`, the boundary where windowing becomes a no-op), a
/// window wider than the causal range, `position == 0` and `position == capacity - 1` (the edges), and
/// `window == 1` (self-only).
#[test]
fn causal_mask_from_pos_matches_host_built_mask_row_table() {
    use poot_graph_ir::Slot;
    use poot_tensor::DType;

    let cases: &[(usize, usize, Option<usize>)] = &[
        (0, 8, None),
        (3, 8, None),
        (7, 8, None),
        (5, 8, Some(3)),   // real window strictly inside [0, position]
        (5, 8, Some(6)),   // window == position + 1: boundary, no-op vs full causal
        (5, 8, Some(100)), // window wider than the causal range: also a no-op
        (0, 8, Some(3)),   // position == 0 (edge)
        (7, 8, Some(3)),   // position == capacity - 1 (edge)
        (5, 8, Some(1)),   // self-only
        (0, 1, Some(1)),   // capacity 1
    ];

    for &(position, cap, window) in cases {
        let b = Builder::new();
        let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32));
        let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, cap, window);
        let posi = pos.id;
        let g = b.finish(mask);

        let mut inputs: HashMap<usize, Value> = HashMap::new();
        inputs.insert(
            posi,
            HostTensor::i32(vec![1, 1], vec![position as i32]).into(),
        );
        let got = crate::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();

        assert_eq!(
            got.shape(),
            vec![1, 1, 1, cap],
            "case {position},{cap},{window:?}: shape"
        );

        // The pre-card host-built mask row (poot-llm's `decode_mask_row`): 0 for a visible key slot,
        // -1e9 for a masked one.
        let want: Vec<f32> = (0..cap)
            .map(|t| {
                let visible = t <= position && window.is_none_or(|w| position - t < w);
                if visible { 0.0 } else { -1.0e9 }
            })
            .collect();

        assert_eq!(
            got.as_f32().unwrap(),
            want.as_slice(),
            "case (position={position}, cap={cap}, window={window:?}): in-graph mask != host-built mask row"
        );
    }
}

/// Card 550: the decode-row table above pins `-1.0e9` (`decode_mask_row`'s historical
/// magnitude), but the pre-card host-built PREFILL mask (`prefill_causal_mask`) used `-1.0e30` - a
/// different masked value for the same visibility pattern. `causal_mask_from_pos` unifies both call
/// sites on one magnitude (`-1.0e9`), so "equals the previous host-built mask bit for bit" cannot hold
/// for prefill's masked entries literally; what carries over is the VISIBILITY PATTERN (which entries
/// are 0 vs masked) plus the softmax-flushing argument that the exact masked magnitude is not
/// load-bearing. This test covers the multi-row prefill shape (`[1,1,L,L]`, `Pos = [0,1,..,L-1]`, the
/// one-shot prefill convention) that the decode-row table does not, and proves the flushing argument
/// numerically rather than asserting it in prose: `exp(mask - row_max)` underflows to exactly `0.0f32`
/// for BOTH magnitudes at every realistic row max (f32's `exp` underflows below `x <~ -103.97`, and both
/// `-1e9` and `-1e30` sit there by roughly nine and twenty-one orders of magnitude respectively, so
/// subtracting any `row_max` a real attention score could produce changes nothing).
#[test]
fn causal_mask_from_pos_prefill_shape_matches_visibility_pattern_and_flushes_either_magnitude() {
    use poot_graph_ir::Slot;
    use poot_tensor::DType;

    let cases: &[(usize, Option<usize>)] = &[
        (6, None),    // full causal, multi-row
        (6, Some(3)), // real window strictly inside the causal range, multi-row
        (6, Some(6)), // window == L: boundary, no-op vs full causal
        (1, None),    // L=1 (single row, degenerate prefill)
    ];

    for &(l, window) in cases {
        let b = Builder::new();
        let pos = b.slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32));
        let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, l, window);
        let posi = pos.id;
        let g = b.finish(mask);

        let mut inputs: HashMap<usize, Value> = HashMap::new();
        inputs.insert(
            posi,
            HostTensor::i32(vec![1, l], (0..l as i32).collect()).into(),
        );
        let got = crate::eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap();
        assert_eq!(
            got.shape(),
            vec![1, 1, l, l],
            "case l={l},window={window:?}: shape"
        );

        // The pre-card host-built PREFILL mask's visibility pattern (`prefill_causal_mask`'s formula,
        // historical magnitude `-1e30`): row i, col j visible iff j <= i (and i - j < window).
        for i in 0..l {
            for j in 0..l {
                let visible = j <= i && window.is_none_or(|w| i - j < w);
                let idx = i * l + j;
                if visible {
                    assert_eq!(
                        got.as_f32().unwrap()[idx],
                        0.0,
                        "case l={l},window={window:?}: row {i} col {j} must be visible (0.0)"
                    );
                } else {
                    assert_eq!(
                        got.as_f32().unwrap()[idx],
                        -1.0e9,
                        "case l={l},window={window:?}: row {i} col {j} must be masked (-1e9, card \
                         550's unified magnitude)"
                    );
                }
            }
        }
    }

    // The flushing argument itself, numerically: softmax always subtracts the row max before `exp`, so
    // every masked entry's contribution is `exp(mask - row_max)`. Both magnitudes underflow to exactly
    // 0.0f32 for any row max a real attention score could produce (tested here up to 1e6, far beyond
    // what any realistic Q.K^T/sqrt(d) scale produces).
    for row_max in [0.0f32, 50.0, -50.0, 1.0e6, -1.0e6] {
        assert_eq!(
            (-1.0e9f32 - row_max).exp(),
            0.0,
            "exp(-1e9 - {row_max}) must flush to exactly 0.0"
        );
        assert_eq!(
            (-1.0e30f32 - row_max).exp(),
            0.0,
            "exp(-1e30 - {row_max}) must flush to exactly 0.0"
        );
    }
}

/// Card 151 SWA fuzz: windowed-mask prefill attention equals attention with a hand-built windowed mask,
/// across random window widths and shapes (incl. GQA). Production feeds an additive `[1,1,L,L]` mask into the
/// ordinary `attention_prefill` composition; the composition fed the windowed mask must reproduce the
/// per-(head,row) softmax reference `ref_attn_row` fed the same mask row.
///
/// Fixed seed, 60 cases: L in 2..24, head_dim in {2,4,8,16}, hkv in 1..4 with n_rep in {1,2,4}, window w in
/// 1..=L+1 (w>=L is full causal, w=1 is self-only). Asserts that true-windowing (w<L) and GQA (n_rep>1) cases
/// both occurred.
#[test]
fn swa_windowed_mask_fuzz_matches_hand_windowed_reference() {
    let mut seed = 0x51a_c0de_5147_0011u64;
    let dims = [2usize, 4, 8, 16];
    let reps = [1usize, 2, 4];
    let mut cases = 0usize;
    let (mut windowed_cases, mut gqa_cases) = (0usize, 0usize);

    for _ in 0..60u64 {
        let l = 2 + (fuzz_u64(&mut seed) % 22) as usize;
        let d = dims[(fuzz_u64(&mut seed) % dims.len() as u64) as usize];
        let hkv = 1 + (fuzz_u64(&mut seed) % 4) as usize;
        let n_rep = reps[(fuzz_u64(&mut seed) % reps.len() as u64) as usize];
        let hq = hkv * n_rep;
        // window in 1..=L+1 (L+1 is full-causal).
        let w = 1 + (fuzz_u64(&mut seed) % (l as u64 + 1)) as usize;

        let q_data = fill(hq * l * d, fuzz_u64(&mut seed));
        let k_data = fill(hkv * l * d, fuzz_u64(&mut seed));
        let v_data = fill(hkv * l * d, fuzz_u64(&mut seed));

        // Windowed causal mask [1,1,L,L]: visible iff j<=i AND i-j<w.
        let win_mask: Vec<f32> = (0..l * l)
            .map(|idx| {
                let (i, j) = (idx / l, idx % l);
                if j <= i && i - j < w { 0.0 } else { -1e30 }
            })
            .collect();

        let out_win = eval_prefill_attn(
            q_data.clone(),
            k_data.clone(),
            v_data.clone(),
            win_mask.clone(),
            hq,
            hkv,
            l,
            d,
        );
        assert_eq!(out_win.len(), hq * l * d, "l={l} hq={hq} d={d}: out len");

        let tol = 1e-4f32;
        for h in 0..hq {
            let kv = h / n_rep; // GQA: query head h reads kv head h/n_rep
            let q_h = &q_data[h * l * d..(h + 1) * l * d];
            let k_h = &k_data[kv * l * d..(kv + 1) * l * d];
            let v_h = &v_data[kv * l * d..(kv + 1) * l * d];
            for i in 0..l {
                let win_row = &win_mask[i * l..(i + 1) * l];
                let ref_row = ref_attn_row(&q_h[i * d..(i + 1) * d], k_h, v_h, win_row, l, d);
                let got_row = &out_win[h * l * d + i * d..h * l * d + (i + 1) * d];
                for (c, (&r, &g)) in ref_row.iter().zip(got_row).enumerate() {
                    let err = (r - g).abs();
                    assert!(
                        err <= tol,
                        "l={l} hq={hq} hkv={hkv} d={d} w={w} head {h} row {i} col {c}: \
                         ref={r:.6} got={g:.6} err={err:.2e} > tol={tol}"
                    );
                }
            }
        }

        if w < l {
            windowed_cases += 1;
        }
        if n_rep > 1 {
            gqa_cases += 1;
        }
        cases += 1;
    }

    eprintln!(
        "swa windowed-mask fuzz: {cases} cases, {windowed_cases} true-windowed (w<L), \
         {gqa_cases} GQA (n_rep>1); all match hand-windowed reference"
    );
    assert!(
        windowed_cases > 0 && gqa_cases > 0,
        "fuzz failed to cover windowing/GQA (windowed={windowed_cases}, gqa={gqa_cases})"
    );
}

/// One step of a `Model::trace` graph on the CPU oracle: `Slot::Token` and `Slot::Pos` bound from `tokens` at
/// absolute position `pos0`, every const filled by `weight(name, shape)`, every carried state from `state` (zeros
/// when empty). Returns the output and the state outputs.
pub(super) fn run_step(
    g: &poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>,
    tokens: &[i32],
    pos0: usize,
    weight: &dyn Fn(&str, &[usize]) -> Vec<f32>,
    state: &[HostTensor],
) -> (HostTensor, Vec<HostTensor>) {
    use poot_graph_ir::{Slot, Storage};

    let mut inputs: HashMap<usize, Value> = HashMap::new();
    for &id in &g.inputs {
        let m = g.meta(id);
        let shape = m.aval.shape.clone();
        let t = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(shape, tokens.to_vec()),
            Storage::Slot(Slot::Pos) => HostTensor::i32(
                shape,
                (pos0..pos0 + tokens.len()).map(|p| p as i32).collect(),
            ),
            Storage::Slot(other) => panic!("unexpected slot {other:?} in a dense step graph"),
            Storage::Const => {
                HostTensor::f32(shape.clone(), weight(m.name.as_deref().unwrap(), &shape))
            }
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()),
            Storage::State => continue,
            Storage::Device => panic!("a graph input is never a device intermediate"),
        };
        inputs.insert(id, Value::from(t));
    }
    for (i, &(input, _)) in g.state.iter().enumerate() {
        let t = state
            .get(i)
            .cloned()
            .unwrap_or_else(|| HostTensor::zeros(g.aval(input).shape.clone()));
        inputs.insert(input, Value::from(t));
    }
    let evaluation = crate::eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
    let state = evaluation
        .state
        .into_iter()
        .map(|v| v.into_host().unwrap())
        .collect();
    (evaluation.output.into_host().unwrap(), state)
}

/// Deterministic name-seeded weights of scale 0.1, the fill the rest of this file uses.
pub(super) fn named_weights(name: &str, shape: &[usize]) -> Vec<f32> {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    fill(shape.iter().product::<usize>().max(1), seed)
        .iter()
        .map(|v| v * 0.1)
        .collect()
}

/// The tiny 4-layer Gemma 3 of the per-layer mask rows: GQA (`n_rep = 2`, so the SWA rows reach the grouped
/// path), a query scale of 0.5, a local window of `window` (`None`: no window) and every `every`-th layer global.
fn gemma3(every: usize, window: Option<usize>) -> poot_executor_parity::weight_map::MappedModel {
    use poot_executor_parity::dense::{Dense, Family};

    Dense::new(Family::Gemma3)
        .vocab(16)
        .dims(8, 16, 4)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .with("rms_norm_eps", 1e-5)
        .with("query_pre_attn_scalar", 4.0)
        .with("sliding_window_pattern", every)
        .with("sliding_window", window)
        .f32_model()
}

fn gemma3_prefill_logits(
    every: usize,
    window: Option<usize>,
    (n, cap): (usize, usize),
) -> Vec<f32> {
    use poot_executor_parity::dense::step;
    use poot_models::model::{LogitRows, Phase};

    let m = gemma3(every, window);
    let g = m
        .model
        .trace(Phase::Prefill, step(1, n, cap, LogitRows::Last))
        .unwrap();
    // Distinct token ids per position: V has no RoPE, so identical tokens would make every V row identical and
    // the windowed mask a no-op (softmax over identical values returns that value).
    let tokens: Vec<i32> = (0..n as i32).collect();
    run_step(&g, &tokens, 0, &named_weights, &[])
        .0
        .as_f32()
        .unwrap()
        .to_vec()
}

/// Card 151 Part 2 (updated for card 550 and POOT-1015): gemma3's per-layer global/local mask split through
/// `Model::trace` of the registry's Gemma 3. Tests the family's selection, not the windowed-mask arithmetic
/// (`swa_windowed_mask_eval_oracle_correctness` and poot-llm's `mask_tests`): a global layer's mask must always
/// be the unwindowed causal mask, a local layer's always the windowed one, never swapped or shared. Both masks are
/// in-graph computations over `Slot::Pos`, so this drives the selection by varying `sliding_window` (the only knob
/// that can change the local mask's content).
///
/// Tiny synthetic 4-layer config with `sliding_window_pattern = 2`: layers 1,3 global, 0,2 local. Three checks:
/// 1. All-global (`sliding_window_pattern = 1`): changing the window must not change the output (no layer is
///    local, so the windowed mask is never selected).
/// 2. Mixed, W < seq_len: narrowing the window from none (full-causal) to `Some(3)` (seq_len=6) must change the
///    output.
/// 3. Mixed, W >= seq_len: `Some(seq_len)` and `Some(seq_len+10)` must both be byte-identical to no window.
#[test]
fn gemma3_prefill_per_layer_mask_selection() {
    let l = 6usize; // seq_len
    let run = |every, window| gemma3_prefill_logits(every, window, (l, l));

    // --- Check 1: all-global, the window must not affect the output. ---
    assert_eq!(
        run(1, None),
        run(1, Some(3)),
        "all-global: the window must not affect output"
    );

    // --- Checks 2+3: mixed (pattern 2 -> layers 1,3 global, 0,2 local). ---
    let out_win = run(2, Some(3));
    let out_full = run(2, None);
    let max_diff = max_abs_error(&out_win, &out_full);
    assert!(
        max_diff > 1e-5,
        "mixed layers: W=3 window on local layers must change the output vs full-causal (max_diff={max_diff:.2e})"
    );
    assert_eq!(
        out_full,
        run(2, Some(l)),
        "mixed layers: window==seq_len must be byte-identical to full-causal"
    );
    assert_eq!(
        out_full,
        run(2, Some(l + 10)),
        "mixed layers: window>seq_len must be byte-identical to full-causal"
    );
}

/// Companion to [`gemma3_prefill_per_layer_mask_selection`] over a cache larger than the prompt (prefill that also
/// fills the K/V caches the decode continues from): same mixed (2 global + 2 local) config. Checks only real
/// windowing (check 3 above).
#[test]
fn gemma3_prefill_kv_per_layer_mask_selection() {
    let (n, cap) = (6usize, 8usize); // prompt length n, cache capacity cap >= n
    let run = |window| gemma3_prefill_logits(2, window, (n, cap));

    let out_win = run(Some(3));
    let out_full = run(None);
    let max_diff = max_abs_error(&out_win, &out_full);
    assert!(
        max_diff > 1e-5,
        "gemma prefill-kv: W=3 window on local layers must change output vs full-causal (max_diff={max_diff:.2e})"
    );

    // W >= n is a no-op end to end.
    assert_eq!(
        out_full,
        run(Some(n)),
        "gemma prefill-kv: window==seq_len must be byte-identical to full-causal"
    );
}

/// Card 151 Part 2 (decode half, updated for card 550 and POOT-1015): the Gemma 3 decode step's per-layer window
/// through `Model::trace`. Both the global and local-layer masks are in-graph computations over `Slot::Pos`
/// (`[1,1]`) and `iota(cap)`, so there is no host-built mask to swap or to pin the window arithmetic against
/// independently - that arithmetic is `swa_windowed_mask_eval_oracle_correctness`'s job. Three checks at a position
/// past the window (pos=5, w=2, so key slots t<=3 are out of window on local layers):
/// 1. Window active, mixed (pattern 2): `sliding_window = Some(2)` must differ from none.
/// 2. Window inert (pos < w): at pos=1 with w=2 nothing is windowed out, so `Some(2)` == none byte-identical.
/// 3. All-global (pattern 1): the local window must reach no layer, so `Some(2)` == none byte-identical even at
///    pos=5.
#[test]
fn gemma3_decode_per_layer_mask_selection() {
    use poot_executor_parity::dense::step;
    use poot_models::model::{LogitRows, Phase};

    let cap = 8usize;
    // Deterministic non-zero cache contents (per state index), so different windows give different outputs.
    let logits = |every: usize, window: Option<usize>, pos: usize| -> Vec<f32> {
        let m = gemma3(every, window);
        let g = m
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .unwrap();
        let state: Vec<HostTensor> = g
            .state
            .iter()
            .enumerate()
            .map(|(i, &(input, _))| {
                let shape = g.aval(input).shape.clone();
                HostTensor::f32(shape.clone(), named_weights(&format!("kv.{i}"), &shape))
            })
            .collect();
        run_step(&g, &[1], pos, &named_weights, &state)
            .0
            .as_f32()
            .unwrap()
            .to_vec()
    };

    // --- Check 1: window active, mixed (layers 1,3 global; 0,2 local). pos=5 > w=2. ---
    let out_win = logits(2, Some(2), 5);
    let out_none = logits(2, None, 5);
    let max_diff = max_abs_error(&out_win, &out_none);
    assert!(
        max_diff > 1e-5,
        "mixed decode: local window w=2 at pos=5 must change output vs full-causal (max_diff={max_diff:.2e})"
    );

    // --- Check 2: window inert at pos=1 (< w=2): nothing windowed out, Some(2) == none. ---
    assert_eq!(
        logits(2, Some(2), 1),
        logits(2, None, 1),
        "decode: at pos < window the window is inert, must be byte-identical to full-causal"
    );

    // --- Check 3: all-global: the local window reaches no layer even at pos=5. ---
    assert_eq!(
        logits(1, Some(2), 5),
        logits(1, None, 5),
        "all-global decode: the window must not affect any layer's output"
    );
}

/// Config wiring: `attn_logit_softcapping` and `final_logit_softcapping` flow from a Gemma 2 config into the traced
/// graph and change the eval output; a config without them traces fewer equations.
///
/// Checks:
///   1. The uncapped graph has fewer equations than the capped one, by one softcap (3 ops) per layer plus one
///      after the head.
///   2. Both evaluated with the same weights give different logits.
///   3. The window changes mask values only: a different `sliding_window` traces the same number of equations.
#[test]
fn config_wiring_softcap_changes_graph_and_output() {
    use poot_executor_parity::dense::{Dense, Family, step};
    use poot_models::model::{LogitRows, Phase};

    let pos = 4usize;
    let gemma2 = |attn: Option<f32>, last: Option<f32>, window: usize| {
        let m = Dense::new(Family::Gemma2)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(4, 2)
            .head_dim(4)
            .max_positions(16)
            .with("attn_logit_softcapping", attn)
            .with("final_logit_softcapping", last)
            .with("sliding_window", window)
            .f32_model();
        m.model
            .trace(Phase::Decode, step(1, 1, 8, LogitRows::Last))
            .unwrap()
    };
    let layers = 2usize;
    let g_none = gemma2(None, None, 4);

    // Tight caps that clamp the tiny test logits. With 0.1-scaled weights and hidden=16, logits are ~0.1-0.3; a cap
    // of 0.05 saturates them to ~0.05 (relative diff > 0.1).
    let g_cap = gemma2(Some(0.05), Some(0.05), 4);

    // Check 1: softcap adds ops to the graph. Each softcap call emits 3 ops: mul_scalar + tanh + mul_scalar. The
    // attention softcap fires once per layer, the final one once after the head: extra = (layers + 1) * 3.
    let expected_extra = (layers + 1) * 3;
    assert_eq!(
        g_cap.eqns.len() - g_none.eqns.len(),
        expected_extra,
        "unexpected op delta: capped {} vs uncapped {}",
        g_cap.eqns.len(),
        g_none.eqns.len()
    );

    let out = |g: &poot_graph_ir::Graph<poot_graph_ir::ValidationOutputs>| {
        run_step(g, &[5], pos, &named_weights, &[]).0
    };
    // Check 2: softcap changes the output, not just the graph structure.
    let max_diff = max_abs_error(
        out(&g_none).as_f32().unwrap(),
        out(&g_cap).as_f32().unwrap(),
    );
    assert!(
        max_diff > 1e-4,
        "softcap config must produce different logits; max_diff={max_diff:.2e}"
    );

    // Check 3: the window only affects mask values at run time.
    assert_eq!(
        gemma2(None, None, 2).eqns.len(),
        g_none.eqns.len(),
        "sliding_window does not change graph structure"
    );
}

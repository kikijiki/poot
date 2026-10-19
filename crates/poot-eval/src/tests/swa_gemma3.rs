//! B2 sliding-window attention oracle (hand-built graphs). The gemma3 per-layer mask selection and softcap
//! config-wiring rows trace a model and live in `poot-models/tests/oracle_traces/swa_gemma3.rs`.

use crate::{EvalError, EvalOptions, Value};
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::attention_prefill;
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;

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
    let b = Builder::for_tests();
    let scale = 1.0 / (d as f32).sqrt();
    let q = b.constant("q", TensorType::f32(vec![1, hq, l, d])).unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![1, hkv, l, d]))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, l, l]))
        .unwrap();
    let n_rep = hq / hkv;
    let out = attention_prefill(&b, q, k, v, n_rep, scale, mask).unwrap();
    let g = b.finish(out).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(q.id(), HostTensor::f32(vec![1, hq, l, d], q_data));
    inputs.insert(k.id(), HostTensor::f32(vec![1, hkv, l, d], k_data));
    inputs.insert(v.id(), HostTensor::f32(vec![1, hkv, l, d], v_data));
    inputs.insert(mask.id(), HostTensor::f32(vec![1, 1, l, l], mask_data));
    let out_t = (|| -> Result<HostTensor, EvalError> {
        let values: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        crate::eval(&g, &values, EvalOptions::default())?
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
        let b = Builder::for_tests();
        let pos = b
            .slot(Slot::Pos, TensorType::new(vec![1, 1], DType::I32))
            .unwrap();
        let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, cap, window).unwrap();
        let posi = pos.id();
        let g = b.finish(mask).unwrap();

        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        inputs.insert(
            posi,
            HostTensor::i32(vec![1, 1], vec![position as i32]).into(),
        );
        let got = crate::eval(&g, &inputs, EvalOptions::default())
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
        let b = Builder::for_tests();
        let pos = b
            .slot(Slot::Pos, TensorType::new(vec![1, l], DType::I32))
            .unwrap();
        let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, l, window).unwrap();
        let posi = pos.id();
        let g = b.finish(mask).unwrap();

        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        inputs.insert(
            posi,
            HostTensor::i32(vec![1, l], (0..l as i32).collect()).into(),
        );
        let got = crate::eval(&g, &inputs, EvalOptions::default())
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

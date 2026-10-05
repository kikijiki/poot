//! Verifies that `flash_attention_decode`'s Body computes `flash_decode_ref` on the kernel-IR reference
//! interpreter, without a GPU. The flash kernel is NVPTX-only (private `o[D]` array), so its only hardware
//! target is RunPod; interpreting the Body here confirms it is a faithful translation of the recurrence,
//! leaving only NVPTX codegen lowering for the RunPod run.

use poot_kernel_ir::Body;
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_test_util::kernel_fixtures::workgroups_covering;
use poot_test_util::{assert_close, max_abs_error};

/// Run `body` with one thread per head over `params` (param local i -> params[i-1]) and return the output,
/// the last param.
fn interpret(body: &Body, params: [&[f32]; 5], threads: usize) -> Vec<f32> {
    let mut buffers = params.map(Buffer::from_f32s);
    run(body, workgroups_covering(body, threads), &mut buffers).unwrap();
    buffers[4].to_f32s().unwrap()
}

#[allow(clippy::too_many_arguments)]
fn flash_decode_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
    // 0 for a broadcast `[cap]` mask, `cap` for a per-head `[Hq, cap]` (ALiBi) mask.
    mask_head_stride: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kv = h / n_rep;
        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut o = vec![0.0f32; d];
        for t in 0..cap {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[kv * cap * d + t * d + dd];
            }
            s = s * scale + mask[h * mask_head_stride + t];
            let m_new = m.max(s);
            // card 674: `m_new == -inf` means every position up to and including this one is masked;
            // `(m - m_new).exp()`/`(s - m_new).exp()` would be `(-inf - -inf).exp() = NaN.exp() = NaN`.
            // Nothing has been seen yet, so this step contributes zero weight.
            let (corr, e) = if m_new == f32::NEG_INFINITY {
                (0.0, 0.0)
            } else {
                ((m - m_new).exp(), (s - m_new).exp())
            };
            l = l * corr + e;
            for dd in 0..d {
                o[dd] = o[dd] * corr + e * v[kv * cap * d + t * d + dd];
            }
            m = m_new;
        }
        for dd in 0..d {
            out[h * d + dd] = o[dd] / l;
        }
    }
    out
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
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

#[test]
fn flash_decode_body_interprets_to_reference() {
    // Interpret the flash_attention_decode Body on CPU and confirm it equals flash_decode_ref (the RunPod run then
    // only confirms NVPTX lowering).
    let (hq, n_rep, cap, d) = (4usize, 2usize, 6usize, 8usize);
    let hkv = hq / n_rep;
    let scale = 1.0f32 / (d as f32).sqrt();
    let q = fill(hq * d, 1);
    let k = fill(hkv * cap * d, 2);
    let v = fill(hkv * cap * d, 3);
    let mask: Vec<f32> = (0..cap)
        .map(|t| if t <= 3 { 0.0 } else { -1.0e9 })
        .collect();

    let body = kg::flash_attention_decode("flash_decode", hq, n_rep, cap, d, scale, false);
    // params: 1=q, 2=k, 3=v, 4=mask, 5=out.
    let got = interpret(&body, [&q, &k, &v, &mask, &vec![0.0f32; hq * d]], hq);
    let want = flash_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale, 0);
    assert_close(&got, &want, 1e-5);
}

/// The same Body-level check for the per-head mask (`mask_per_head = true`), the layout ALiBi needs. The
/// generated Body must read `mask[h*cap + t]`, so head `h`'s own bias row enters the online softmax inside the
/// running-max/rescale recurrence.
///
/// The mutation this pins: a Body that kept the broadcast `mask[t]` index would score every head with head 0's
/// row. The second assertion checks the interpreted result differs from that.
#[test]
fn flash_decode_body_per_head_mask_interprets_to_reference() {
    let (hq, n_rep, cap, d) = (4usize, 2usize, 6usize, 8usize);
    let hkv = hq / n_rep;
    let scale = 1.0f32 / (d as f32).sqrt();
    let q = fill(hq * d, 1);
    let k = fill(hkv * cap * d, 2);
    let v = fill(hkv * cap * d, 3);
    // One ALiBi row per head: visibility (0 / -1e9) plus -slope[h]*(pos - t), pos = 3.
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];
    let pos = 3usize;
    let mask: Vec<f32> = (0..hq)
        .flat_map(|h| {
            (0..cap).map(move |t| {
                if t <= pos {
                    -slopes[h] * (pos as f32 - t as f32)
                } else {
                    -1.0e9
                }
            })
        })
        .collect();

    let body = kg::flash_attention_decode("flash_decode", hq, n_rep, cap, d, scale, true);
    let got = interpret(&body, [&q, &k, &v, &mask, &vec![0.0f32; hq * d]], hq);
    let want = flash_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale, cap);
    assert_close(&got, &want, 1e-5);

    // MUTATION: head 0's row broadcast over every head (a Body that ignored the head stride).
    let head0: Vec<f32> = mask[..cap].to_vec();
    let broadcast = flash_decode_ref(&q, &k, &v, &head0, hq, n_rep, cap, d, scale, 0);
    let max_diff = max_abs_error(&got, &broadcast);
    assert!(
        max_diff > 1e-4,
        "the per-head Body must not collapse to head 0's mask row; max_diff={max_diff:.2e}"
    );
}

/// card 674: every other mask-construction site in the repo uses the `-1.0e9` finite sentinel for a
/// masked position; these three tests use the mathematically literal `f32::NEG_INFINITY` instead -
/// what the quantized-KV decode acceptance tests (`poot-gpu/tests/qwen2.rs`) also use. Mutation: drop
/// the `m_new == f32::NEG_INFINITY` guards added to `flash_attention_decode`/`flash_region_decode` in
/// `poot-kernelgen/src/flash.rs` (card 674) and these three go red (NaN output, confirmed by hand
/// before the fix landed).
#[test]
fn flash_decode_body_leading_true_inf_mask_does_not_nan() {
    // Position 0 is masked with TRUE -inf while the running max is still -inf (nothing seen yet):
    // `m_new = max(-inf, -inf) = -inf`, and the un-guarded recurrence computed
    // `exp(-inf - -inf) = NaN`, poisoning the whole row even though position 1 is unmasked and should
    // carry all the weight. A causal mask (`t <= pos` valid) never hits this for a single-thread
    // sequential loop (position 0 is always valid), so this needs an explicit non-causal-shaped mask.
    let (hq, n_rep, cap, d) = (2usize, 1usize, 2usize, 4usize);
    let hkv = hq / n_rep;
    let scale = 0.5f32;
    let q = fill(hq * d, 21);
    let k = fill(hkv * cap * d, 22);
    let v = fill(hkv * cap * d, 23);
    let mask = vec![f32::NEG_INFINITY, 0.0];

    let body = kg::flash_attention_decode("flash_decode", hq, n_rep, cap, d, scale, false);
    let got = interpret(&body, [&q, &k, &v, &mask, &vec![0.0f32; hq * d]], hq);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "flash_attention_decode produced non-finite output with a leading true-inf mask: {got:?}"
    );
    let want = flash_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale, 0);
    assert_close(&got, &want, 1e-5);
}

/// The textbook two-pass reference (`softmax(scale*qkᵀ+mask) @ v`, one pass over the whole row): never
/// subtracts two infinities as long as at least one position is unmasked, so it is a safe oracle for
/// [`flash_region_decode_body_fully_masked_lane_does_not_nan`]'s partially-masked row.
#[allow(clippy::too_many_arguments)]
fn direct_decode_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        let kv = h / n_rep;
        let mut scores = vec![0.0f32; cap];
        for (t, sc) in scores.iter_mut().enumerate() {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[kv * cap * d + t * d + dd];
            }
            *sc = s * scale + mask[t];
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut denom = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - m).exp();
            denom += *sc;
        }
        for dd in 0..d {
            let mut acc = 0.0f32;
            for (t, &sc) in scores.iter().enumerate() {
                acc += sc / denom * v[kv * cap * d + t * d + dd];
            }
            out[h * d + dd] = acc;
        }
    }
    out
}

/// card 674: `flash_region_decode`'s LDS-cooperative kernel strides `cap` KV positions over `w` lanes
/// (`t = lane, lane+w, ..`); with `cap=3 < w=8`, lanes 3..7 get no position at all (idle: `m` stays its
/// initial `-inf`) and lane 0's one assigned position (`t=0`) is masked with true `-inf` (so lane 0's
/// own running max also stays `-inf`). Both the per-lane update and the cross-lane merge must fold
/// these "nothing seen yet" states in without computing `exp(-inf - -inf)`. Runs on the kernel-IR
/// interpreter (`poot_kernel_ir::interp::run`, which supports the workgroup-local arrays and barrier
/// this Body uses) - no GPU needed.
#[test]
fn flash_region_decode_body_fully_masked_lane_does_not_nan() {
    let (hq, n_rep, cap, d, w) = (2usize, 1usize, 3usize, 4usize, 8usize);
    let hkv = hq / n_rep;
    let scale = 0.5f32;
    let q = fill(hq * d, 31);
    let k = fill(hkv * cap * d, 32);
    let v = fill(hkv * cap * d, 33);
    // Lanes 0 and 1 (t=0, t=1) are masked - two LEADING lanes empty, not just one. The merge folds
    // lane ii=1 into the running state seeded from lane 0 first (`ii` starts at 1): with only lane 0
    // masked, lane 1 would already be finite and the running max would never be `-inf` at a merge
    // step, leaving the merge's own `m_merged == -inf` branch dead. Masking both forces that merge
    // step itself to combine two still-empty partials (card 674).
    let mask = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0];

    let body =
        kg::flash_region_decode("flash_region_decode", 1, hq, n_rep, cap, d, scale, w, false);
    let mut buffers = [
        Buffer::from_f32s(&q),
        Buffer::from_f32s(&k),
        Buffer::from_f32s(&v),
        Buffer::from_f32s(&mask),
        Buffer::from_f32s(&vec![0.0f32; hq * d]),
    ];
    run(&body, workgroups_covering(&body, hq * w), &mut buffers).unwrap();
    let got = buffers[4].to_f32s().unwrap();
    assert!(
        got.iter().all(|x| x.is_finite()),
        "flash_region_decode produced non-finite output with a fully -inf-masked lane: {got:?}"
    );
    let want = direct_decode_ref(&q, &k, &v, &mask, hq, n_rep, cap, d, scale);
    assert_close(&got, &want, 1e-5);
}

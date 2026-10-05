//! Spec 267: sectioned ("mRoPE") RoPE, `poot_graph_ir::ops::rope_sectioned`. CPU-oracle proof: a hand-rolled independent
//! reference for the distinct-per-section-position case, plus backward-compatibility checks (equal section positions
//! must reduce bit-exact to plain `rope`/`rope_prefill`).

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{rope, rope_prefill, rope_sectioned, rope_sectioned_batched};
use poot_graph_ir::types::TensorType;
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::fill;
use poot_test_util::assert_close_rel;

/// Independent reference: for each half-dim frequency index, pick the position id of the section that index falls in,
/// gather that section's cos/sin row at that position, then the usual half-split rotate. `xd`/`cosd`/`sind` only need
/// `rot` (resp. `max_pos*rot`) leading elements populated. `cosd`/`sind` are flattened `[max_pos, rot]`.
fn sectioned_reference(
    xd: &[f32],
    cosd: &[f32],
    sind: &[f32],
    rot: usize,
    positions: &[usize],
    sections: &[usize],
) -> Vec<f32> {
    let half = rot / 2;
    assert_eq!(sections.iter().sum::<usize>(), half);
    let mut sec_of = vec![0usize; half];
    let mut start = 0;
    for (j, &w) in sections.iter().enumerate() {
        for slot in sec_of.iter_mut().take(start + w).skip(start) {
            *slot = j;
        }
        start += w;
    }
    let mut cos_row = vec![0.0f32; rot];
    let mut sin_row = vec![0.0f32; rot];
    for i in 0..half {
        let p = positions[sec_of[i]];
        cos_row[i] = cosd[p * rot + i];
        cos_row[i + half] = cosd[p * rot + i + half];
        sin_row[i] = sind[p * rot + i];
        sin_row[i + half] = sind[p * rot + i + half];
    }
    let mut rh = vec![0.0f32; rot];
    for i in 0..half {
        rh[i] = -xd[half + i];
        rh[half + i] = xd[i];
    }
    (0..rot)
        .map(|i| xd[i] * cos_row[i] + rh[i] * sin_row[i])
        .collect()
}

#[test]
fn mrope_matches_direct_reference_with_distinct_section_positions() {
    // full rotary (rot == d == 16), 3 sections [2,3,3] summing to half=8, three distinct position ids (the general
    // multimodal case: temporal/height/width all differ).
    let b = Builder::new();
    let d = 16usize;
    let max_pos = 16usize;
    let sections = [2usize, 3, 3];
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos_t = b.constant("pos_t", TensorType::scalar(DType::I32));
    let pos_h = b.constant("pos_h", TensorType::scalar(DType::I32));
    let pos_w = b.constant("pos_w", TensorType::scalar(DType::I32));
    let out = rope_sectioned(&b, x, cos, sin, &[pos_t, pos_h, pos_w], &sections);
    let (xi, ci, si, pti, phi, pwi) = (x.id, cos.id, sin.id, pos_t.id, pos_h.id, pos_w.id);
    let g = b.finish(out);

    let xd = fill(d, 201);
    let cosd = fill(max_pos * d, 202);
    let sind = fill(max_pos * d, 203);
    let (pt, ph, pw) = (2usize, 9usize, 13usize);
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
    );
    inputs.insert(pti, Value::from(HostTensor::i32(vec![], vec![pt as i32])));
    inputs.insert(phi, Value::from(HostTensor::i32(vec![], vec![ph as i32])));
    inputs.insert(pwi, Value::from(HostTensor::i32(vec![], vec![pw as i32])));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let want = sectioned_reference(&xd, &cosd, &sind, d, &[pt, ph, pw], &sections);
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

#[test]
fn mrope_batched_matches_independent_per_row_reference() {
    let b = Builder::new();
    let batch = 2usize;
    let d = 16usize;
    let max_pos = 16usize;
    let sections = [2usize, 3, 3];
    let x = b.constant("x", TensorType::f32(vec![batch, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos_t = b.constant("pos_t", TensorType::new(vec![batch], DType::I32));
    let pos_h = b.constant("pos_h", TensorType::new(vec![batch], DType::I32));
    let pos_w = b.constant("pos_w", TensorType::new(vec![batch], DType::I32));
    let out = rope_sectioned_batched(&b, x, cos, sin, &[pos_t, pos_h, pos_w], &sections, batch);
    let (xi, ci, si, pti, phi, pwi) = (x.id, cos.id, sin.id, pos_t.id, pos_h.id, pos_w.id);
    let g = b.finish(out);
    g.validate()
        .expect("batched sectioned RoPE graph validates");

    let xd = fill(batch * d, 211);
    let cosd = fill(max_pos * d, 212);
    let sind = fill(max_pos * d, 213);
    let axes = [[2usize, 7], [5usize, 3], [9usize, 12]];
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![batch, 1, 1, d], xd.clone())),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
    );
    inputs.insert(
        pti,
        Value::from(HostTensor::i32(
            vec![batch],
            axes[0].iter().map(|&p| p as i32).collect(),
        )),
    );
    inputs.insert(
        phi,
        Value::from(HostTensor::i32(
            vec![batch],
            axes[1].iter().map(|&p| p as i32).collect(),
        )),
    );
    inputs.insert(
        pwi,
        Value::from(HostTensor::i32(
            vec![batch],
            axes[2].iter().map(|&p| p as i32).collect(),
        )),
    );
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let mut want = Vec::with_capacity(batch * d);
    for row in 0..batch {
        want.extend(sectioned_reference(
            &xd[row * d..(row + 1) * d],
            &cosd,
            &sind,
            d,
            &[axes[0][row], axes[1][row], axes[2][row]],
            &sections,
        ));
    }
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);
}

#[test]
fn mrope_matches_rope_bit_exact_when_section_positions_equal_decode() {
    // backward compatibility (decode/scalar-position shape): all 3 sections given the same position id must reduce to
    // bit-exact plain `rope` output, since no arithmetic separates them (pure slice + concat of identical source rows).
    let d = 16usize;
    let max_pos = 16usize;
    let sections = [3usize, 2, 3];
    let xd = fill(d, 301);
    let cosd = fill(max_pos * d, 302);
    let sind = fill(max_pos * d, 303);
    let posv = 5usize;

    let got_sectioned = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
        let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
        let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
        let p0 = b.constant("p0", TensorType::scalar(DType::I32));
        let p1 = b.constant("p1", TensorType::scalar(DType::I32));
        let p2 = b.constant("p2", TensorType::scalar(DType::I32));
        let out = rope_sectioned(&b, x, cos, sin, &[p0, p1, p2], &sections);
        let (xi, ci, si, p0i, p1i, p2i) = (x.id, cos.id, sin.id, p0.id, p1.id, p2.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(
            xi,
            Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
        );
        inputs.insert(
            ci,
            Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
        );
        inputs.insert(p0i, Value::from(HostTensor::i32(vec![], vec![posv as i32])));
        inputs.insert(p1i, Value::from(HostTensor::i32(vec![], vec![posv as i32])));
        inputs.insert(p2i, Value::from(HostTensor::i32(vec![], vec![posv as i32])));
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };

    let got_rope = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
        let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
        let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
        let pos = b.constant("pos", TensorType::scalar(DType::I32));
        let out = rope(&b, x, cos, sin, pos);
        let (xi, ci, si, pi) = (x.id, cos.id, sin.id, pos.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(
            xi,
            Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
        );
        inputs.insert(
            ci,
            Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
        );
        inputs.insert(pi, Value::from(HostTensor::i32(vec![], vec![posv as i32])));
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };

    assert_eq!(got_sectioned.shape(), got_rope.shape());
    let sb: Vec<u32> = got_sectioned
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let rb: Vec<u32> = got_rope
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        sb, rb,
        "equal-position sectioned rope must be bit-exact to plain rope"
    );
}

#[test]
fn mrope_matches_rope_prefill_bit_exact_when_section_positions_equal_whole_sequence() {
    // backward compatibility (whole-sequence shape): a text-only prompt gives every section the same running position
    // counter 0..L, so sectioned rope over explicit position vectors must be bit-exact to `rope_prefill`'s implicit 0..L.
    let d = 16usize;
    let max_pos = 16usize;
    let l = 6usize;
    let sections = [1usize, 3, 4];
    let xd = fill(d * l, 401);
    let cosd = fill(max_pos * d, 402);
    let sind = fill(max_pos * d, 403);
    let posv: Vec<i32> = (0..l).map(|i| i as i32).collect();

    let got_sectioned = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, l, d]));
        let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
        let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
        let p0 = b.constant("p0", TensorType::new(vec![l], DType::I32));
        let p1 = b.constant("p1", TensorType::new(vec![l], DType::I32));
        let p2 = b.constant("p2", TensorType::new(vec![l], DType::I32));
        let out = rope_sectioned(&b, x, cos, sin, &[p0, p1, p2], &sections);
        let (xi, ci, si, p0i, p1i, p2i) = (x.id, cos.id, sin.id, p0.id, p1.id, p2.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(
            xi,
            Value::from(HostTensor::f32(vec![1, 1, l, d], xd.clone())),
        );
        inputs.insert(
            ci,
            Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
        );
        inputs.insert(p0i, Value::from(HostTensor::i32(vec![l], posv.clone())));
        inputs.insert(p1i, Value::from(HostTensor::i32(vec![l], posv.clone())));
        inputs.insert(p2i, Value::from(HostTensor::i32(vec![l], posv.clone())));
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };

    let got_prefill = {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![1, 1, l, d]));
        let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
        let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
        let out = rope_prefill(&b, x, cos, sin, l);
        let (xi, ci, si) = (x.id, cos.id, sin.id);
        let g = b.finish(out);
        let mut inputs = HashMap::new();
        inputs.insert(
            xi,
            Value::from(HostTensor::f32(vec![1, 1, l, d], xd.clone())),
        );
        inputs.insert(
            ci,
            Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
        );
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };

    assert_eq!(got_sectioned.shape(), got_prefill.shape());
    let sb: Vec<u32> = got_sectioned
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let rb: Vec<u32> = got_prefill
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(
        sb, rb,
        "equal-position sectioned rope must be bit-exact to rope_prefill over a whole sequence"
    );
}

#[test]
fn mrope_partial_rotary_passes_through_untouched_dims() {
    // phi3-style partial rotary composed with sectioning: head_dim d=12, only the leading rot=8 dims rotate (sections
    // partition rot/2=4, not d/2), [rot,d) passes through unrotated, as `rope_partial`.
    let b = Builder::new();
    let (d, rot) = (12usize, 8usize);
    let max_pos = 16usize;
    let sections = [1usize, 3];
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, rot]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, rot]));
    let pos_t = b.constant("pos_t", TensorType::scalar(DType::I32));
    let pos_h = b.constant("pos_h", TensorType::scalar(DType::I32));
    let out = rope_sectioned(&b, x, cos, sin, &[pos_t, pos_h], &sections);
    let (xi, ci, si, pti, phi) = (x.id, cos.id, sin.id, pos_t.id, pos_h.id);
    let g = b.finish(out);

    let xd = fill(d, 501);
    let cosd = fill(max_pos * rot, 502);
    let sind = fill(max_pos * rot, 503);
    let (pt, ph) = (2usize, 7usize);
    let mut inputs = HashMap::new();
    inputs.insert(
        xi,
        Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
    );
    inputs.insert(
        ci,
        Value::from(HostTensor::f32(vec![max_pos, rot], cosd.clone())),
    );
    inputs.insert(
        si,
        Value::from(HostTensor::f32(vec![max_pos, rot], sind.clone())),
    );
    inputs.insert(pti, Value::from(HostTensor::i32(vec![], vec![pt as i32])));
    inputs.insert(phi, Value::from(HostTensor::i32(vec![], vec![ph as i32])));
    let got = eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let want_rot = sectioned_reference(&xd, &cosd, &sind, rot, &[pt, ph], &sections);
    assert_close_rel(&got.as_f32().unwrap()[..rot], &want_rot, 1e-5);
    assert_eq!(
        &got.as_f32().unwrap()[rot..d],
        &xd[rot..d],
        "passthrough dims [rot,d) must be bit-exact copies of x"
    );
}

#[test]
fn mrope_zero_width_section_traces_and_is_inert() {
    // Edge case: a section width of 0 is a degenerate but legal partition (e.g. folding the temporal axis into
    // height/width). `Builder::slice` and `Concat` both allow zero-length ranges, so this must trace and evaluate.
    //
    // `rope_sectioned` gathers every section's position id unconditionally (FR-005), even for width 0, so the position id
    // must still be in-bounds for `cos_table`/`sin_table` or eval panics on the gather. Production callers
    // (`trace_prefill_mrope`/`trace_decode_mrope`) always supply an in-range id per section.
    let b = Builder::new();
    let d = 16usize;
    let max_pos = 16usize;
    let sections = [0usize, 3, 5]; // sums to half=8; section 0 is empty
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos_t = b.constant("pos_t", TensorType::scalar(DType::I32));
    let pos_h = b.constant("pos_h", TensorType::scalar(DType::I32));
    let pos_w = b.constant("pos_w", TensorType::scalar(DType::I32));
    let out = rope_sectioned(&b, x, cos, sin, &[pos_t, pos_h, pos_w], &sections);
    let (xi, ci, si, pti, phi, pwi) = (x.id, cos.id, sin.id, pos_t.id, pos_h.id, pos_w.id);
    let g = b.finish(out);

    let xd = fill(d, 601);
    let cosd = fill(max_pos * d, 602);
    let sind = fill(max_pos * d, 603);
    let (ph, pw) = (4usize, 11usize);
    let eval_with_pt = |pt: usize| -> Vec<f32> {
        let mut inputs = HashMap::new();
        inputs.insert(
            xi,
            Value::from(HostTensor::f32(vec![1, 1, 1, d], xd.clone())),
        );
        inputs.insert(
            ci,
            Value::from(HostTensor::f32(vec![max_pos, d], cosd.clone())),
        );
        inputs.insert(
            si,
            Value::from(HostTensor::f32(vec![max_pos, d], sind.clone())),
        );
        inputs.insert(pti, Value::from(HostTensor::i32(vec![], vec![pt as i32])));
        inputs.insert(phi, Value::from(HostTensor::i32(vec![], vec![ph as i32])));
        inputs.insert(pwi, Value::from(HostTensor::i32(vec![], vec![pw as i32])));
        eval(&g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_f32()
            .unwrap()
            .to_vec()
    };

    // Two different in-bounds position ids for the empty section must produce identical output, proving the zero-width
    // section's position never reaches the result (the reference never indexes section 0 when its width is 0).
    let got_a = eval_with_pt(2);
    let got_b = eval_with_pt(9);
    assert_eq!(
        got_a, got_b,
        "a zero-width section's position id must not affect the output"
    );
    let want = sectioned_reference(&xd, &cosd, &sind, d, &[2, ph, pw], &sections);
    assert_close_rel(&got_a, &want, 1e-5);
}

#[test]
#[should_panic(expected = "section widths must sum to rot/2")]
fn mrope_panics_when_section_widths_do_not_sum_to_half_rot() {
    let b = Builder::new();
    let d = 16usize;
    let max_pos = 4usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos_t = b.constant("pos_t", TensorType::scalar(DType::I32));
    let pos_h = b.constant("pos_h", TensorType::scalar(DType::I32));
    // sections sum to 5, half is 8: must panic before any eval.
    let _ = rope_sectioned(&b, x, cos, sin, &[pos_t, pos_h], &[2, 3]);
}

#[test]
#[should_panic(expected = "one position id per section")]
fn mrope_panics_when_position_count_does_not_match_section_count() {
    let b = Builder::new();
    let d = 16usize;
    let max_pos = 4usize;
    let x = b.constant("x", TensorType::f32(vec![1, 1, 1, d]));
    let cos = b.constant("cos", TensorType::f32(vec![max_pos, d]));
    let sin = b.constant("sin", TensorType::f32(vec![max_pos, d]));
    let pos_t = b.constant("pos_t", TensorType::scalar(DType::I32));
    // 1 position for 3 declared sections: must panic before any eval.
    let _ = rope_sectioned(&b, x, cos, sin, &[pos_t], &[2, 3, 3]);
}

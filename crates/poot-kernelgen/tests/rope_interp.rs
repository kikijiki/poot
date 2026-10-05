//! Verifies that `kernelgen::rope`'s Body computes the half-split rotate reference on the kernel-IR reference
//! interpreter, without a GPU. Confirms the Body is a faithful translation of the `ops::rope_partial`
//! decomposition (full and partial rotary, cos/sin broadcast over the head axis), leaving only backend codegen
//! lowering for the on-device / RunPod run.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_test_util::kernel_fixtures::workgroups_covering;

/// The half-split rotate reference (`ops::rope_partial`). `x` is `[hq, d]` flat; `cos`/`sin` are `[rot]`
/// (one position, broadcast over the head axis). Passthrough `[rot, d)` for a partial rope.
fn rope_ref(x: &[f32], cos: &[f32], sin: &[f32], hq: usize, d: usize, rot: usize) -> Vec<f32> {
    let half = rot / 2;
    let mut out = vec![0.0f32; hq * d];
    for h in 0..hq {
        for j in 0..d {
            let i = h * d + j;
            if j >= rot {
                out[i] = x[i];
            } else if j < half {
                out[i] = x[i] * cos[j] + (-x[i + half]) * sin[j];
            } else {
                out[i] = x[i] * cos[j] + x[i - half] * sin[j];
            }
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

fn check(d: usize, rot: usize) {
    let hq = 3usize; // multi-head: exercises cos/sin broadcast over the head axis (leading dims)
    let x_shape = vec![1usize, hq, 1, d];
    let cos_shape = vec![rot]; // decode: one gathered [rot] row, broadcast against x
    let x = fill(hq * d, 1);
    let cos = fill(rot, 2);
    let sin = fill(rot, 3);
    let body = kg::rope("rope", &x_shape, &cos_shape, rot).expect("rope precondition");
    // params: 1=x, 2=cos, 3=sin, 4=out.
    let mut buffers = [
        Buffer::from_f32s(&x),
        Buffer::from_f32s(&cos),
        Buffer::from_f32s(&sin),
        Buffer::from_f32s(&vec![0.0f32; hq * d]),
    ];
    run(&body, workgroups_covering(&body, hq * d), &mut buffers).unwrap();
    let got = buffers[3].to_f32s().unwrap();
    let want = rope_ref(&x, &cos, &sin, hq, d, rot);
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - b).abs() <= 1e-6,
            "rope Body interp vs ref at {i} (d={d} rot={rot}): {a} vs {b}"
        );
    }
}

#[test]
fn rope_body_interprets_to_reference_full() {
    // full rotary: rot == D, no passthrough.
    check(8, 8);
    check(16, 16);
}

#[test]
fn rope_body_interprets_to_reference_partial() {
    // partial rotary (phi3/qwen3next): only the leading rot dims rotate, [rot, D) passes through.
    check(8, 4);
    check(16, 4);
}

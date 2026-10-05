//! Card 630 (R466-009): `Tanh` and `Erf` join the transcendental primitives; `silu`, `gelu` and `gelu_erf`
//! are `ops` compositions over them. Every literal below is the pre-card CPU oracle's output for the same
//! input, recorded from master `061a8896` (where `Silu`, `Gelu` and `GeluErf` were primitives), so the
//! compositions are pinned against the old oracle rather than against themselves.
//!
//! Integration test (not `#[cfg(test)]`) because the crate dev-depends on `poot-eval`, which depends back
//! on it; see `fusion_soundness_536b.rs`.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::ops::{geglu, gelu, gelu_erf, sigmoid, silu, softcap, softplus, swiglu};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, Traced};
use poot_graph_plan::passes_without_target as optimize;
use poot_tensor::HostTensor;

/// The first-row value table: zeros, tiny, unit, mid, saturating and non-finite inputs.
const TABLE: [f32; 24] = [
    0.0,
    1e-30,
    -1e-30,
    1e-6,
    -1e-6,
    1.0,
    -1.0,
    3.0,
    -3.0,
    5.0,
    -5.0,
    7.0,
    -7.0,
    10.0,
    -10.0,
    88.0,
    -88.0,
    100.0,
    -100.0,
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.5,
    -0.5,
];

// Pre-card oracle outputs over `TABLE`, as f32 bit patterns.
const SILU: [u32; 24] = [
    0x00000000, 0x0d224260, 0x8d224260, 0x350637c1, 0xb50637b9, 0x3f3b26a8, 0xbe89b2b1, 0x4036e4ed,
    0xbe11b139, 0x409eeddc, 0xbd0911d0, 0x40dfcbc2, 0xbbd0f929, 0x411ffe24, 0xb9ee03fe, 0x42b00000,
    0x83354ddc, 0x42c80000, 0x80000000, 0x7fc00000, 0x7f800000, 0xffc00000, 0x3e9f597f, 0xbe414d03,
];
const GELU: [u32; 24] = [
    0x00000000, 0x0d224260, 0x8d224260, 0x350637c4, 0xb50637b7, 0x3f57585c, 0xbe229e91, 0x403fc468,
    0xbb6e6153, 0x40a00000, 0xb4761491, 0x40e00000, 0xa726d86f, 0x41200000, 0x8223e4cf, 0x42b00000,
    0x80000000, 0x42c80000, 0x80000000, 0x7fc00000, 0x7f800000, 0xffc00000, 0x3eb1016d, 0xbe1dfd26,
];
const GELU_ERF: [u32; 24] = [
    0x00000000, 0x0d224260, 0x8d224260, 0x350637c4, 0xb50637b6, 0x3f57625f, 0xbe227686, 0x403fbda6,
    0xbb84b34c, 0x409ffffd, 0xb5c05e5d, 0x40e00000, 0xad1d9a40, 0x41200000, 0x80000000, 0x42b00000,
    0x80000000, 0x42c80000, 0x80000000, 0x7fc00000, 0x7f800000, 0xffc00000, 0x3eb103af, 0xbe1df8a2,
];
const SIGMOID: [u32; 24] = [
    0x3f000000, 0x3f000000, 0x3f000000, 0x3f000004, 0x3efffff8, 0x3f3b26a8, 0x3e89b2b1, 0x3f73dbe6,
    0x3d4241a2, 0x3f7e4961, 0x3bdb4fb4, 0x3f7fc44c, 0x3a6ed39c, 0x3f7ffd06, 0x383e6998, 0x3f800000,
    0x0041edc4, 0x3f800000, 0x00000000, 0xffc00000, 0x3f800000, 0x00000000, 0x3f1f597f, 0x3ec14d03,
];
const SOFTPLUS: [u32; 24] = [
    0x3f317218, 0x3f317218, 0x3f317218, 0x3f317220, 0x3f317210, 0x3fa818f5, 0x3ea063d5, 0x40431c0e,
    0x3d470382, 0x40a03703, 0x3bdc0c6e, 0x40e00778, 0x3a6eec1e, 0x41200030, 0x383e7ee4, 0x42b00000,
    0x00000000, 0x7f800000, 0x00000000, 0x7fc00000, 0x7f800000, 0x00000000, 0x3f795d1c, 0x3ef2ba38,
];
/// `ops::softcap(x, 30.0)` with the pre-card exp-composition tanh.
const SOFTCAP_30: [u32; 24] = [
    0x00000000, 0x00000000, 0x00000000, 0x35700000, 0xb5700000, 0x3f7fe7b9, 0xbf7fe7b9, 0x403f5cd3,
    0xc03f5cd3, 0x409e88ea, 0xc09e88ea, 0x40dc057e, 0xc0dc057e, 0x411a537c, 0xc11a537c, 0x41eea4f3,
    0xc1eea4f3, 0x41ef63d1, 0xc1ef63d1, 0x7fc00000, 0xffc00000, 0xffc00000, 0x3efff9fe, 0xbefff9fe,
];

// Pre-card `activation(gate) * up` over the MLP fixture below (`eval(g)` and `eval(passes(g))` agreed).
const MLP_SILU: [u32; 32] = [
    0x80000000, 0xbca2e87c, 0xbd4b4c3d, 0xbdb98cff, 0xbe12581a, 0xbe517e43, 0xbe8a8641, 0xbea9fdd3,
    0xbec0c306, 0xbec77293, 0xbeb7b72a, 0xbe8f8af1, 0xbe29c653, 0xbd403c77, 0x3cd7671a, 0xbb7e29d0,
    0xbe3fc6fa, 0xbf0d095a, 0xbf8b99db, 0xbfe2b1ab, 0xc0220ada, 0xc0530684, 0xc07f0758, 0xc0907332,
    0xc09a0353, 0xc09a5545, 0xc0901ab7, 0xc0755ba2, 0xc034403a, 0xbfbd016d, 0x3e0d3348, 0x3ffaabfd,
];
const MLP_GELU: [u32; 32] = [
    0x80000000, 0xb72204a4, 0xb8c22c42, 0xba0e3068, 0xbb18d3ae, 0xbc00ae9e, 0xbcaf96cd, 0xbd45f3bf,
    0xbdba2371, 0xbe11e841, 0xbe3bfb08, 0xbe3f5bb8, 0xbe0a7906, 0xbd35a089, 0x3ce0f150, 0xbb8d11b5,
    0xbe5b5e0c, 0xbf226cfa, 0xbf9f5bf9, 0xbffe40d8, 0xc031f801, 0xc0630ff0, 0xc086c5f2, 0xc0967b5d,
    0xc09eac18, 0xc09db1a1, 0xc0925b59, 0xc0781ba5, 0xc035b2f9, 0xbfbe1813, 0x3e0dc87e, 0x3ffb69db,
];
const MLP_GELU_ERF: [u32; 32] = [
    0x80000000, 0xb7a8abee, 0xb91b01c6, 0xba3bf76a, 0xbb31f997, 0xbc0a79d5, 0xbcb4be4f, 0xbd476dbb,
    0xbdba052e, 0xbe119747, 0xbe3bb4fb, 0xbe3f3fd7, 0xbe0a74f7, 0xbd35a05c, 0x3ce0f174, 0xbb8d1376,
    0xbe5b65fa, 0xbf22754e, 0xbf9f6264, 0xbffe4285, 0xc031f2fc, 0xc06305ac, 0xc086c04d, 0xc096771f,
    0xc09ea9c1, 0xc09db0a8, 0xc0925b0a, 0xc0781b7f, 0xc035b2f2, 0xbfbe1812, 0x3e0dc87e, 0x3ffb69db,
];

fn eval_bits(g: &Graph, inputs: HashMap<usize, HostTensor>) -> Vec<u32> {
    let inputs: HashMap<usize, Value> = inputs
        .into_iter()
        .map(|(k, v)| (k, Value::from(v)))
        .collect();
    eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap()
        .as_f32()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `eval(g)` and `eval(passes(g))` over `TABLE` for an elementwise activation.
fn run_table(f: impl Fn(&Builder, Traced) -> Traced) -> (Vec<u32>, Vec<u32>) {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![TABLE.len()]));
    let y = f(&b, x);
    let xi = x.id;
    let g = b.finish(y);
    let input = || HashMap::from([(xi, HostTensor::f32(vec![TABLE.len()], TABLE.to_vec()))]);
    (eval_bits(&g, input()), eval_bits(&optimize(&g), input()))
}

/// The MLP fixture inputs: a gate sweeping `[-5, 5)` and an oscillating up projection.
fn mlp_inputs(gate: usize, up: usize) -> HashMap<usize, HostTensor> {
    let n = 32usize;
    let gd: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 - 5.0) * 0.9).collect();
    let ud: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.21).sin() * 1.5).collect();
    HashMap::from([
        (gate, HostTensor::f32(vec![1, 1, n], gd)),
        (up, HostTensor::f32(vec![1, 1, n], ud)),
    ])
}

/// `eval(g)` and `eval(passes(g))` of `act(gate) * up` (`swiglu` / `geglu` shape; `gelu_erf` has no GLU
/// wrapper, so its MLP multiplies by hand).
fn run_mlp(act: fn(&Builder, Traced) -> Traced) -> (Vec<u32>, Vec<u32>) {
    let b = Builder::new();
    let gate = b.constant("gate", TensorType::f32(vec![1, 1, 32]));
    let up = b.constant("up", TensorType::f32(vec![1, 1, 32]));
    let out = b.binary(poot_graph_ir::BinOp::Mul, act(&b, gate), up);
    let (gi, ui) = (gate.id, up.id);
    let g = b.finish(out);
    (
        eval_bits(&g, mlp_inputs(gi, ui)),
        eval_bits(&optimize(&g), mlp_inputs(gi, ui)),
    )
}

fn assert_bits(name: &str, got: &[u32], want: &[u32; 24]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g,
            w,
            "{name}(x={}) row {i}: got {:#010x} ({}) want {w:#010x} ({})",
            TABLE[i],
            g,
            f32::from_bits(*g),
            f32::from_bits(*w)
        );
    }
}

/// Distance in f32 units in the last place between two finite values (sign-aware).
fn ulps(a: f32, b: f32) -> u32 {
    let key = |v: f32| {
        let bits = v.to_bits() as i32;
        if bits < 0 { i32::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

/// ADR-0101 tier 2 elementwise check: every element, NaN where the pre-card oracle has NaN and nowhere
/// else, otherwise within `abs + rel * |want|`. Returns the largest absolute error and the largest error in ulps
/// over the rows whose expected magnitude is at least `1e-3` (ulps of a near-zero tail are noise).
fn assert_tier2(name: &str, got: &[u32], want: &[u32], abs: f32, rel: f32) -> (f32, u32) {
    let (mut max_abs, mut max_ulp) = (0.0f32, 0);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (f32::from_bits(*g), f32::from_bits(*w));
        if w.is_nan() {
            assert!(g.is_nan(), "{name} row {i}: want NaN, got {g}");
        } else {
            assert!(!g.is_nan(), "{name} row {i}: NaN where {w} expected");
            if g == w {
                continue; // equal, including matching infinities
            }
            assert!(
                (g - w).abs() <= abs + rel * w.abs(),
                "{name} row {i}: got {g} want {w} (diff {})",
                (g - w).abs()
            );
            max_abs = max_abs.max((g - w).abs());
            if w.abs() >= 1e-3 {
                max_ulp = max_ulp.max(ulps(g, w));
            }
        }
    }
    (max_abs, max_ulp)
}

/// SC-001: `silu` and `gelu` equal the pre-card oracle `to_bits` over the value table; `sigmoid` and
/// `softplus` are unchanged `to_bits`; `gelu_erf` is within tier 2 of the pre-card literals; and each
/// activation's `eval(g)` equals `eval(passes(g))` `to_bits`.
/// Mutation: write `silu` as `x * sigmoid(x)` (`ops/activation.rs`); rows 3, -3, 5 and -7 differ in the
/// last bit.
#[test]
fn activation_compositions_match_the_pre_card_oracle() {
    for (name, f, want) in [
        ("silu", silu as fn(&Builder, Traced) -> Traced, &SILU),
        ("gelu", gelu, &GELU),
        ("sigmoid", sigmoid, &SIGMOID),
        ("softplus", softplus, &SOFTPLUS),
    ] {
        let (raw, fused) = run_table(f);
        assert_bits(name, &raw, want);
        assert_eq!(raw, fused, "{name}: eval(g) vs eval(passes(g))");
    }

    let (raw, fused) = run_table(gelu_erf);
    assert_eq!(raw, fused, "gelu_erf: eval(g) vs eval(passes(g))");
    // Per-op f32 rounding replaces the old f64 path: near the left tail `1 + erf` cancels, so the bound
    // is absolute there.
    let (max_abs, max_ulp) = assert_tier2("gelu_erf", &raw, &GELU_ERF, 1e-6, 1e-5);
    eprintln!(
        "gelu_erf vs pre-card oracle: max abs {max_abs:e}, max {max_ulp} ulp (|want| >= 1e-3)"
    );
}

/// SC-003: a qwen2-class SwiGLU and a gemma-class GeGLU MLP equal the pre-card oracle `to_bits` before
/// and after the pass pipeline; an MPT-class exact-GELU MLP is within tier 2 and `to_bits` between the
/// two evaluations.
/// Mutation: swap the `0.044715` association in `gelu` (`x * x * x * 0.044715`); the gemma row goes red.
#[test]
fn mlp_fixtures_match_the_pre_card_oracle() {
    let (raw, fused) = run_mlp(silu);
    assert_eq!(raw, MLP_SILU, "swiglu eval(g)");
    assert_eq!(fused, MLP_SILU, "swiglu eval(passes(g))");

    let (raw, fused) = run_mlp(gelu);
    assert_eq!(raw, MLP_GELU, "geglu eval(g)");
    assert_eq!(fused, MLP_GELU, "geglu eval(passes(g))");

    let (raw, fused) = run_mlp(gelu_erf);
    assert_eq!(raw, fused, "gelu_erf MLP: eval(g) vs eval(passes(g))");
    let (max_abs, max_ulp) = assert_tier2("gelu_erf MLP", &raw, &MLP_GELU_ERF, 1e-6, 1e-5);
    eprintln!(
        "gelu_erf MLP vs pre-card oracle: max abs {max_abs:e}, max {max_ulp} ulp (|want| >= 1e-3)"
    );
}

/// `swiglu`/`geglu` build the same graph as `act(gate) * up` by hand (the GLU wrappers add nothing).
#[test]
fn glu_wrappers_equal_activation_times_up() {
    for (glu, act) in [
        (
            swiglu as fn(&Builder, Traced, Traced) -> Traced,
            silu as fn(&Builder, Traced) -> Traced,
        ),
        (geglu, gelu),
    ] {
        let b = Builder::new();
        let gate = b.constant("gate", TensorType::f32(vec![1, 1, 32]));
        let up = b.constant("up", TensorType::f32(vec![1, 1, 32]));
        let (gi, ui) = (gate.id, up.id);
        let out = glu(&b, gate, up);
        let g = b.finish(out);
        let want = run_mlp(act).0;
        assert_eq!(eval_bits(&g, mlp_inputs(gi, ui)), want);
    }
}

/// `ops::softcap` over the `Tanh` primitive stays within tier 2 of the pre-card exp-composition
/// oracle; `tanh(+-inf)` is now the exact cap where the old composition was NaN.
#[test]
fn softcap_on_the_tanh_primitive_stays_within_tier_two_of_the_pre_card_oracle() {
    let (raw, fused) = run_table(|b, x| softcap(b, x, 30.0));
    assert_eq!(raw, fused, "softcap: eval(g) vs eval(passes(g))");
    let finite: Vec<usize> = (0..TABLE.len()).filter(|&i| TABLE[i].is_finite()).collect();
    let got: Vec<u32> = finite.iter().map(|&i| raw[i]).collect();
    let want: Vec<u32> = finite.iter().map(|&i| SOFTCAP_30[i]).collect();
    let (max_abs, max_ulp) = assert_tier2("softcap", &got, &want, 1e-6, 1e-5);
    eprintln!(
        "softcap(30) vs pre-card oracle, finite rows: max abs {max_abs:e}, max {max_ulp} ulp (|want| >= 1e-3)"
    );
    for (i, cap) in [(20usize, 30.0f32), (21, -30.0)] {
        assert_eq!(
            f32::from_bits(raw[i]),
            cap,
            "softcap({}) saturates",
            TABLE[i]
        );
    }
}

//! Card 536a (R467-005, R486-014): the attention/rope/flash matchers are legality checks, proven
//! against the real CPU oracle (`poot_eval::eval`) rather than a duplicated one.
//!
//! This is an integration test (`tests/`), not a `#[cfg(test)] mod` inside `src/`: this crate
//! dev-depends on `poot-eval`, which depends back on this crate (an allowed dev-dependency cycle, see
//! `crates/poot-graph-plan/Cargo.toml`). A unit-test module compiled with `--cfg test` recompiles this
//! crate's own types as a distinct unit from the one `poot-eval` was built against, so
//! `poot_eval::eval` would reject this crate's `Graph` as a foreign type ("multiple different versions
//! of crate `poot_graph_plan`", a standard Cargo limitation for this dependency shape). An integration
//! test links against the crate's normal library unit instead - the same one `poot-eval` sees - so the
//! types match.
//!
//! Card 626: moved here from `poot-graph-ir/tests/` with the transform passes themselves - `optimize`
//! (the standalone pipeline) is gone, reconstructed below from its passes (`compile` is the one
//! pipeline now, but needs a `Target` this oracle-comparison suite never had).
//!
//! Only the oracle-comparison tests live here, and only through already-public APIs (`optimize`,
//! `dispatch_count`, `OpKind` matching, the `ops::` builders). The typed decline-reason
//! assertions (which need the matcher internals `match_attention`, `match_rope_core`, `canonicalize`,
//! `producer_map`, `AttentionDecline`, `RopeDecline`) don't need the oracle at all, so they stay
//! `pub(super)` and live as ordinary unit tests in `crates/poot-graph-plan/src/passes/tests.rs` (536a
//! review: widening those to `pub` for a handful of test-only integration-crate callers would have
//! been the exact "public only for tests" pattern the dead-pub remediation removed elsewhere).

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::analysis::dispatch_count;
use poot_graph_ir::ops::{attention_masked, attention_masked_softcap};
use poot_graph_ir::{BinOp, Builder, Graph, OpKind, Operand, RedOp, Scalar, TensorType, UnOp};
use poot_graph_plan::passes_without_target as optimize;
use poot_tensor::HostTensor;

/// Bind every graph input to a deterministic tensor of its declared shape.
fn bind_inputs(g: &Graph) -> HashMap<usize, HostTensor> {
    g.inputs
        .iter()
        .map(|&id| {
            let shape = g.aval(id).shape.clone();
            let n = shape.iter().product::<usize>().max(1);
            let data = (0..n)
                .map(|i| ((i as f32) * 0.171 - 0.4).sin() * 0.5)
                .collect();
            (id, HostTensor::f32(shape, data))
        })
        .collect()
}

fn eval_graph(g: &Graph) -> HostTensor {
    let inputs: HashMap<usize, Value> = bind_inputs(g)
        .into_iter()
        .map(|(k, v)| (k, Value::from(v)))
        .collect();
    eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap_or_else(|e| panic!("oracle eval failed: {e}"))
        .output
        .into_host()
        .unwrap_or_else(|e| panic!("oracle output not dense: {e}"))
}

/// Tier-1 semantic comparison: finite outputs must agree word for word.
fn assert_same_bits(a: &HostTensor, b: &HostTensor) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    assert_eq!(
        a.as_f32().unwrap().len(),
        b.as_f32().unwrap().len(),
        "element count mismatch"
    );
    for (i, (x, y)) in a
        .as_f32()
        .unwrap()
        .iter()
        .zip(b.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite element {i}: {x} vs {y}"
        );
        assert_eq!(x.to_bits(), y.to_bits(), "element {i}: {x} vs {y}");
    }
}

fn flash_count(g: &Graph) -> usize {
    g.eqns
        .iter()
        .filter(|e| {
            matches!(
                e.op,
                OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
            )
        })
        .count()
}

fn rope_count(g: &Graph) -> usize {
    g.eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::Rope { .. }))
        .count()
}

/// The decode-shaped hand-written attention (q `[1,4,1,16]`, k/v `[1,4,8,16]`, mask `[1,1,1,8]`)
/// with knobs for the equivalent spellings R467-005 lists: the commuted mask add, the scalar placed
/// on the left, the scale folded into `q` before `Q K^T`, and the softmax normalizer as
/// `e * (1 / sum)`.
#[derive(Clone, Copy, Default)]
struct AttentionSpelling {
    mask_first: bool,
    scale_q_first: bool,
    recip_softmax: bool,
}

fn hand_attention_decode(spelling: AttentionSpelling) -> Graph {
    let (hq, cap, d) = (4usize, 8usize, 16usize);
    let scale = 1.0 / (d as f32).sqrt();
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let kt = b.transpose(k, vec![0, 1, 3, 2]);
    let scores = if spelling.scale_q_first {
        let scaled_q = b.binary_scalar(BinOp::Mul, q, Scalar::F32(scale));
        b.matmul(scaled_q, kt)
    } else {
        let qk = b.matmul(q, kt);
        b.binary_scalar(BinOp::Mul, qk, Scalar::F32(scale))
    };
    let scores = if spelling.mask_first {
        b.binary(BinOp::Add, mask, scores)
    } else {
        b.binary(BinOp::Add, scores, mask)
    };
    let last = b.aval(scores).rank() - 1;
    let m = b.reduce(RedOp::Max, scores, last, true);
    let shifted = b.binary(BinOp::Sub, scores, m);
    let e = b.unary(UnOp::Exp, shifted);
    let denom = b.reduce(RedOp::Sum, e, last, true);
    let p = if spelling.recip_softmax {
        let recip = b.unary(UnOp::Recip, denom);
        b.binary(BinOp::Mul, e, recip)
    } else {
        b.binary(BinOp::Div, e, denom)
    };
    let out = b.matmul(p, v);
    b.finish(out)
}

/// A full-rotary rotate-half rope with free `x`/`cos`/`sin` shapes; `commute` writes `Mul(cos, x)`.
fn hand_rope(x_shape: Vec<usize>, cos_shape: Vec<usize>, commute: bool) -> Graph {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(x_shape.clone()));
    let cos = b.constant("cos", TensorType::f32(cos_shape.clone()));
    let sin = b.constant("sin", TensorType::f32(cos_shape));
    let last = x_shape.len() - 1;
    let d = x_shape[last];
    let half = d / 2;
    let x1 = b.slice(x, last, 0, half);
    let x2 = b.slice(x, last, half, d);
    let neg_x2 = b.unary(UnOp::Neg, x2);
    let rotate_half = b.concat(last, &[neg_x2, x1]);
    let xc = if commute {
        b.binary(BinOp::Mul, cos, x)
    } else {
        b.binary(BinOp::Mul, x, cos)
    };
    let rs = b.binary(BinOp::Mul, rotate_half, sin);
    let out = b.binary(BinOp::Add, xc, rs);
    b.finish(out)
}

/// Legal attention spellings fuse without changing arithmetic boundaries. Pre-scaling Q remains
/// its own operation; reciprocal normalization remains outside the division-based flash region.
/// Every spelling is compared bitwise with its own primitive computation.
#[test]
fn card536a_attention_spellings_fuse_like_the_helper() {
    let (hq, cap, d) = (4usize, 8usize, 16usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
    let helper_out = attention_masked(&b, q, k, v, 1, 0.25, mask);
    let helper = b.finish(helper_out);

    let optimized_helper = optimize(&helper);
    assert_eq!(
        flash_count(&optimized_helper),
        1,
        "the helper spelling fuses"
    );
    let reference = dispatch_count(&optimized_helper);
    assert_same_bits(&eval_graph(&helper), &eval_graph(&optimized_helper));

    let spellings = [
        (
            "commuted mask add",
            AttentionSpelling {
                mask_first: true,
                ..Default::default()
            },
        ),
        (
            "scale folded into q",
            AttentionSpelling {
                scale_q_first: true,
                ..Default::default()
            },
        ),
        (
            "softmax e * (1/sum)",
            AttentionSpelling {
                recip_softmax: true,
                ..Default::default()
            },
        ),
    ];
    for (name, spelling) in spellings {
        let graph = hand_attention_decode(spelling);
        let optimized = optimize(&graph);
        let expected_flash = usize::from(!spelling.recip_softmax);
        assert_eq!(
            flash_count(&optimized),
            expected_flash,
            "{name}: incorrect region selection"
        );
        if !spelling.recip_softmax {
            assert_eq!(
                dispatch_count(&optimized),
                reference + usize::from(spelling.scale_q_first),
                "{name}: pre-scaling must retain its dispatch and rounding boundary"
            );
        }
        assert_same_bits(&eval_graph(&graph), &eval_graph(&optimized));
    }
}

/// SC-001 metamorphic rope: `Mul(cos, x)` is the same rotation as the helper's `Mul(x, cos)` and
/// fuses to one `Rope`, with the oracle unchanged. Mutation: the rope matcher's structural `a`/`cos`
/// role detection (or `canonicalize`); the commuted fixture stays the decomposition.
#[test]
fn card536a_rope_commuted_spelling_fuses_like_the_helper() {
    let helper = hand_rope(vec![1, 4, 3, 8], vec![1, 1, 3, 8], false);
    let commuted = hand_rope(vec![1, 4, 3, 8], vec![1, 1, 3, 8], true);
    for (name, graph) in [("helper", &helper), ("Mul(cos, x)", &commuted)] {
        let optimized = optimize(graph);
        assert_eq!(rope_count(&optimized), 1, "{name}: one fused Rope expected");
        assert_same_bits(&eval_graph(graph), &eval_graph(&optimized));
    }
    assert_eq!(
        dispatch_count(&optimize(&helper)),
        dispatch_count(&optimize(&commuted)),
        "commuted rope must reach the helper's dispatch count"
    );
}

/// SC-002 near-miss attention, by semantic detail: `optimize` declines and the oracle result is
/// unchanged (the decomposition stays). The typed decline-reason assertion for each of these same
/// fixtures is `card536a_attention_near_misses_decline_with_a_reason` in
/// `src/transform/tests.rs`. Mutation observed red: widen the matching check; the fixture fuses and
/// `flash_count` stops being zero.
#[test]
fn card536a_attention_near_misses_decline_with_a_reason() {
    let mut cases: Vec<(&str, Graph)> = Vec::new();

    // Axis: the row softmax reduces the wrong axis.
    let mut wrong_axis = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask);
        b.finish(out)
    };
    // Axis 0 is the batch dim and is always 1 in these fixtures, so forcing the reduce there with
    // `keepdim: true` leaves the shape unchanged - the declared aval is still correct, it just needs
    // recomputing from each reduce's own input (not left at its pre-mutation value).
    let reduce_input_avals: Vec<(usize, TensorType)> = wrong_axis
        .eqns
        .iter()
        .filter_map(|eqn| match (&eqn.op, eqn.inputs.first()) {
            (OpKind::Reduce { .. }, Some(Operand::Value(input))) => {
                Some((eqn.out, wrong_axis.aval(*input).clone()))
            }
            _ => None,
        })
        .collect();
    for eqn in wrong_axis.eqns.iter_mut() {
        if let OpKind::Reduce { axis, .. } = &mut eqn.op {
            *axis = 0;
        }
    }
    for (out, aval) in reduce_input_avals {
        wrong_axis.values[out].aval = aval;
    }
    cases.push(("wrong reduce axis", wrong_axis));

    // Scale: a softcap chain whose two constants are not reciprocals.
    let mut wrong_softcap = {
        let (hq, cap, d) = (4usize, 6usize, 16usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
        let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
        let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, cap]));
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.25, mask, Some(2.0));
        b.finish(out)
    };
    let mut rewrote = false;
    for eqn in wrong_softcap.eqns.iter_mut() {
        if let OpKind::Binary(BinOp::Mul) = eqn.op
            && let Some(Operand::Lit(Scalar::F32(0.5))) = eqn.inputs.get(1)
        {
            eqn.inputs[1] = Operand::Lit(Scalar::F32(0.6));
            rewrote = true;
        }
    }
    assert!(rewrote, "the softcap reciprocal literal must be present");
    cases.push(("softcap constants not reciprocal", wrong_softcap));

    // Mask: the value width differs from the QK width (MLA); the fused op indexes one width.
    let mla = {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 3]));
        let k = b.constant("k", TensorType::f32(vec![1, 2, 5, 3]));
        let v = b.constant("v", TensorType::f32(vec![1, 2, 5, 2]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, 1, 5]));
        let out = attention_masked(&b, q, k, v, 1, 1.0, mask);
        b.finish(out)
    };
    cases.push(("V width differs from QK", mla));

    for (name, graph) in cases {
        let optimized = optimize(&graph);
        assert_eq!(flash_count(&optimized), 0, "{name}: must stay unfused");
        assert_same_bits(&eval_graph(&graph), &eval_graph(&optimized));
    }
}

/// Head-dim awareness (R480-009): a decode match at or under `FLASH_LDS_CAP` fuses; wider than it
/// stays unfused with the oracle result unchanged. A prefill at D=512 still fuses through the
/// prefill cap. The typed `HeadDimExceedsCap` decline reason for the D=512 decode case is
/// `card536a_flash_head_dim_is_aware_decline_reason` in `src/transform/tests.rs`. Mutation: remove
/// the head-dim gate; the widening op is emitted (and the D=512 assertion fails).
#[test]
fn card536a_flash_head_dim_is_aware() {
    let build = |m: usize, d: usize| {
        let b = Builder::new();
        let q = b.constant("q", TensorType::f32(vec![1, 2, m, d]));
        let k = b.constant("k", TensorType::f32(vec![1, 2, 5, d]));
        let v = b.constant("v", TensorType::f32(vec![1, 2, 5, d]));
        let mask = b.constant("mask", TensorType::f32(vec![1, 1, m, 5]));
        let out = attention_masked(&b, q, k, v, 1, 0.25, mask);
        b.finish(out)
    };

    let narrow = build(1, 256);
    assert_eq!(flash_count(&optimize(&narrow)), 1, "D=256 decode must fuse");

    let wide = build(1, 512);
    let optimized = optimize(&wide);
    assert_eq!(flash_count(&optimized), 0, "D=512 decode must stay unfused");
    assert_same_bits(&eval_graph(&wide), &eval_graph(&optimized));

    let prefill_wide = build(5, 512);
    assert_eq!(
        flash_count(&optimize(&prefill_wide)),
        1,
        "D=512 prefill must fuse through the prefill cap"
    );
}

/// R486-014: transform soundness is proven inside this crate's own test suite through the oracle,
/// including a tileable contraction (M=16, rank-2 weight). Card 557: tiling is the planner's kernel
/// choice, not a graph rewrite, so the passes leave the contraction a plain `MatMul` and the oracle
/// evaluates raw and optimized identically.
#[test]
fn card536a_contraction_is_covered_by_the_in_crate_oracle() {
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, 16, 32]));
    let w = b.constant("w", TensorType::f32(vec![32, 16]));
    let raw_out = b.matmul(x, w);
    let raw = b.finish(raw_out);

    let optimized = optimize(&raw);
    assert!(
        optimized
            .eqns
            .iter()
            .any(|eqn| matches!(eqn.op, OpKind::MatMul)),
        "the contraction stays a plain MatMul through the passes"
    );
    assert_eq!(
        eval_graph(&raw).shape(),
        eval_graph(&optimized).shape(),
        "optimize keeps the contraction's shape"
    );
    assert_same_bits(&eval_graph(&raw), &eval_graph(&optimized));
}

/// The attention flash decode's own typing oracle, kept with the transform soundness suite
/// (R486-014): `FlashAttentionDecode` with B>1 (`q [B,Hq,1,D]`, `k/v [B,Hkv,cap,D]`,
/// `mask [B,1,1,cap]`) equals the materialized `attention_masked` per batch row over random shapes.
/// Card 038 originally, moved from `poot-eval`'s `flash_optimize.rs` (card 536a): unlike its neighbors
/// there, it needs only `poot_graph_ir`/`poot_eval`, not a real model's traces.
#[test]
fn flash_decode_batched_matches_attention_masked() {
    let mut s: u64 = 0xBA7C_4ED0_1234_5678;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rf = |r: &mut dyn FnMut() -> u64| ((r() % 1000) as f32 / 1000.0) * 1.4 - 0.7;
    for _ in 0..20 {
        let bsz = 2 + (rng() % 3) as usize;
        let hkv = 1 + (rng() % 2) as usize;
        let n_rep = 1 + (rng() % 3) as usize;
        let hq = hkv * n_rep;
        let d = 2 + (rng() % 5) as usize;
        let cap = 1 + (rng() % 6) as usize;
        let scale = 0.2 + 0.1 * (rng() % 5) as f32;
        let qd: Vec<f32> = (0..bsz * hq * d).map(|_| rf(&mut rng)).collect();
        let kd: Vec<f32> = (0..bsz * hkv * cap * d).map(|_| rf(&mut rng)).collect();
        let vd: Vec<f32> = (0..bsz * hkv * cap * d).map(|_| rf(&mut rng)).collect();
        let md: Vec<f32> = (0..bsz * cap)
            .map(|i| {
                if i % cap == 0 || rng() % 2 == 0 {
                    0.0
                } else {
                    -1.0e9
                }
            })
            .collect();

        let build = |flash: bool| -> HostTensor {
            let b = Builder::new();
            let q = b.constant("q", TensorType::f32(vec![bsz, hq, 1, d]));
            let k = b.constant("k", TensorType::f32(vec![bsz, hkv, cap, d]));
            let v = b.constant("v", TensorType::f32(vec![bsz, hkv, cap, d]));
            let mask = b.constant("m", TensorType::f32(vec![bsz, 1, 1, cap]));
            let out = attention_masked(&b, q, k, v, n_rep, scale, mask);
            let materialized = b.finish(out);
            // Card 557: tracers cannot build the flash op; the matcher forms it from the chain.
            let g = if flash {
                let fused = optimize(&materialized);
                assert_eq!(flash_count(&fused), 1, "the batched decode chain must fuse");
                fused
            } else {
                materialized
            };
            let mut inp: HashMap<usize, Value> = HashMap::new();
            inp.insert(
                q.id,
                Value::from(HostTensor::f32(vec![bsz, hq, 1, d], qd.clone())),
            );
            inp.insert(
                k.id,
                Value::from(HostTensor::f32(vec![bsz, hkv, cap, d], kd.clone())),
            );
            inp.insert(
                v.id,
                Value::from(HostTensor::f32(vec![bsz, hkv, cap, d], vd.clone())),
            );
            inp.insert(
                mask.id,
                Value::from(HostTensor::f32(vec![bsz, 1, 1, cap], md.clone())),
            );
            eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        };
        let flash = build(true);
        let materialized = build(false);
        assert_eq!(flash.shape(), vec![bsz, hq, 1, d]);
        assert_same_bits(&flash, &materialized);
    }
}

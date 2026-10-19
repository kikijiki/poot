//! `fuse` preserves the exact-I32 bitwise chain's result bit for bit.
//!
//! The chain (And/Or/Xor/Shl/Shr/Not/Clz/Select over wrapping I32 words) is the one
//! `poot-eval/src/tests/exact_i32.rs` pins against hand-computed words; that crate must never depend on
//! `poot-graph-plan`, so the fusion half of the check lives here, driving the real `fuse` pass.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{BinOp, Builder, Scalar, TensorType, UnOp};
use poot_tensor::DType;
use poot_tensor::HostTensor;

fn ints(value: Value) -> Option<Vec<i32>> {
    value.into_host().unwrap().as_i32().map(<[i32]>::to_vec)
}

#[test]
fn fuse_preserves_the_exact_i32_bit_select_and_clz_chain() {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![4], DType::I32))
        .unwrap();
    let y = b
        .constant("y", TensorType::new(vec![4], DType::I32))
        .unwrap();
    let and = b.binary(BinOp::And, x, y).unwrap();
    let or = b.binary(BinOp::Or, x, y).unwrap();
    let xor = b.binary(BinOp::Xor, x, y).unwrap();
    let shl = b.binary_scalar(BinOp::Shl, x, Scalar::I32(33)).unwrap();
    let shr = b.binary(BinOp::Shr, x, y).unwrap();
    let not = b.unary(UnOp::Not, x).unwrap();
    let clz = b.unary(UnOp::Clz, x).unwrap();
    let selected = b.select(and, or, xor).unwrap();
    // Every intermediate feeds the output, so each op is live in the region `fuse` rewrites.
    let mut out = selected;
    for term in [shl, shr, not, clz] {
        out = b.binary(BinOp::Or, out, term).unwrap();
    }
    let g = b.finish(out).unwrap();

    let inputs = HashMap::from([
        (
            x.id(),
            Value::from(HostTensor::i32(vec![4], vec![0, -1, i32::MIN, 1])),
        ),
        (
            y.id(),
            Value::from(HostTensor::i32(vec![4], vec![7, 0, 1, 33])),
        ),
    ]);
    let options = || EvalOptions::default();

    let unfused = ints(eval(&g, &inputs, options()).unwrap().output);
    // Lane 0 by hand: select(and=0, or=7, xor=7) = 7; or-ed with shl 0, shr 0, not -1, clz 32 = -1.
    assert_eq!(unfused.as_deref().map(|words| words[0]), Some(-1));

    let fused = poot_graph_plan::fuse(&g).unwrap();
    assert!(
        fused.eqns().len() < g.eqns().len(),
        "fuse must actually rewrite this chain ({} eqns before, {} after)",
        g.eqns().len(),
        fused.eqns().len()
    );
    assert_eq!(
        ints(eval(&fused, &inputs, options()).unwrap().output),
        unfused
    );
}

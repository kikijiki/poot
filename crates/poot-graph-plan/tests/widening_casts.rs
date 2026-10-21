//! `fuse_widening_casts`: the `Cast` of a narrow weight becomes a step of the fused region that reads it, so
//! the weight's two stored bytes are the only copy and no dispatch widens it (POOT-1022). Every fold is also
//! checked against the evaluation of the graph as traced, bit for bit.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::op::{BinOp, FusedOp, LocalId, OpKind, RowStep, UnOp};
use poot_graph_ir::ops::rmsnorm;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Builder, Graph, ValueId};
use poot_graph_plan::passes_without_target;
use poot_tensor::{DType, HostTensor};

/// BF16 bit patterns of `n` small values, and the f32 values they widen to.
fn bf16_weight(n: usize, seed: f32) -> (Vec<u16>, Vec<f32>) {
    let bits: Vec<u16> = (0..n)
        .map(|i| ((0.5 + seed + i as f32 * 0.3125).to_bits() >> 16) as u16)
        .collect();
    let widened = bits
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    (bits, widened)
}

fn f32_values(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as u64 * 31 + seed * 17) as f32 * 0.37).sin() * 2.0)
        .collect()
}

fn output_bits(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<u32> {
    eval(g, inputs, EvalOptions::default())
        .expect("the graph evaluates")
        .output
        .into_host()
        .expect("a dense graph")
        .as_f32()
        .expect("an f32 output")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn casts(g: &Graph) -> usize {
    g.eqns()
        .iter()
        .filter(|e| matches!(e.op, OpKind::Cast { .. }))
        .count()
}

/// The one eqn of `g` that `pick` selects.
fn only(g: &Graph, pick: impl Fn(&OpKind) -> bool) -> &poot_graph_ir::Eqn {
    let mut found = g.eqns().iter().filter(|e| pick(&e.op));
    let eqn = found.next().expect("one such eqn");
    assert!(found.next().is_none(), "exactly one such eqn");
    eqn
}

#[test]
fn a_norm_scale_cast_folds_into_the_row_region_and_evaluates_identically() {
    let n = 8usize;
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n], DType::BF16))
        .unwrap();
    let out = rmsnorm(&b, x, b.cast(w, DType::F32).unwrap(), 1e-6).unwrap();
    let (xi, wi) = (x.id(), w.id());
    let g = b.finish(out).unwrap();

    let og = passes_without_target(&g).unwrap();
    assert_eq!(casts(&g), 1);
    assert_eq!(casts(&og), 0, "the widening cast is a step, not an eqn");
    let row = only(&og, |op| matches!(op, OpKind::FusedRow(_)));
    let OpKind::FusedRow(region) = &row.op else {
        unreachable!()
    };
    assert_eq!(row.inputs.len(), 2, "the row reads x and the narrow w");
    assert_eq!(og.aval(og.eqns()[0].out).dtype, DType::F32);
    assert!(
        row.inputs
            .iter()
            .any(|input| matches!(input, poot_graph_ir::Operand::Value(v) if og.aval(*v).dtype == DType::BF16)),
        "the region reads the BF16 weight itself"
    );
    assert!(
        (0..region.n_inputs).any(|input| region.widens_input(LocalId::new(input))),
        "a Cast step widens that input"
    );
    assert!(matches!(
        region.steps[0],
        RowStep::Pointwise {
            op: FusedOp::Cast { to: DType::F32 },
            ..
        }
    ));

    let (bits, _) = bf16_weight(n, 0.0);
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            xi,
            Value::from(HostTensor::f32(vec![1, 1, n], f32_values(n, 1))),
        ),
        (wi, Value::from(HostTensor::bf16(vec![n], bits))),
    ]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

#[test]
fn a_bias_cast_folds_into_the_contraction_epilogue_and_evaluates_identically() {
    let (k, n) = (8usize, 4usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, k])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![k, n])).unwrap();
    let bias = b
        .constant("bias", TensorType::new(vec![n], DType::BF16))
        .unwrap();
    let y = b.matmul(x, w).unwrap();
    let bias = b
        .broadcast(b.cast(bias, DType::F32).unwrap(), vec![1, 1, n])
        .unwrap();
    let out = b.binary(BinOp::Add, y, bias).unwrap();
    let (xi, wi) = (x.id(), w.id());
    let g = b.finish(out).unwrap();
    let bias_const = g
        .values()
        .iter_enumerated()
        .find(|(_, v)| v.name().as_deref() == Some("bias"))
        .map(|(id, _)| id)
        .expect("the bias const");

    let og = passes_without_target(&g).unwrap();
    assert_eq!(casts(&og), 0, "the widening cast is a step, not an eqn");
    let epilogue = only(&og, |op| matches!(op, OpKind::ContractionEpilogue { .. }));
    let OpKind::ContractionEpilogue { region, .. } = &epilogue.op else {
        unreachable!()
    };
    assert!(
        epilogue
            .inputs
            .iter()
            .any(|input| matches!(input, poot_graph_ir::Operand::Value(v) if *v == bias_const)),
        "the epilogue reads the BF16 bias itself"
    );
    assert!((1..region.n_inputs).any(|input| region.widens_input(LocalId::new(input))));

    let (bits, _) = bf16_weight(n, 0.25);
    let inputs: HashMap<ValueId, Value> = HashMap::from([
        (
            xi,
            Value::from(HostTensor::f32(vec![1, 1, k], f32_values(k, 2))),
        ),
        (
            wi,
            Value::from(HostTensor::f32(vec![k, n], f32_values(k * n, 3))),
        ),
        (bias_const, Value::from(HostTensor::bf16(vec![n], bits))),
    ]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

#[test]
fn a_cast_nothing_else_may_see_stays_a_cast() {
    let n = 8usize;
    // The narrow const has a second reader: it must stay a buffer every reader can read.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n], DType::BF16))
        .unwrap();
    let scale = b.cast(w, DType::F32).unwrap();
    let normed = rmsnorm(&b, x, scale, 1e-6).unwrap();
    let other = b
        .cast(b.reshape(w, vec![1, 1, n]).unwrap(), DType::F32)
        .unwrap();
    let out = b.binary(BinOp::Add, normed, other).unwrap();
    let g = b.finish(out).unwrap();
    let og = passes_without_target(&g).unwrap();
    assert_eq!(casts(&og), 2, "a const with a second reader folds nothing");

    // The cast result has a second reader: the widened f32 value is still needed outside the region.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n], DType::BF16))
        .unwrap();
    let scale = b.cast(w, DType::F32).unwrap();
    let normed = rmsnorm(&b, x, scale, 1e-6).unwrap();
    let out = b
        .binary(BinOp::Add, normed, b.reshape(scale, vec![1, 1, n]).unwrap())
        .unwrap();
    let g = b.finish(out).unwrap();
    let og = passes_without_target(&g).unwrap();
    assert_eq!(casts(&og), 1, "a cast with a second reader stays a cast");

    // A pointwise region's kernels read every leaf as f32: its cast is not folded.
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![1, 1, n])).unwrap();
    let w = b
        .constant("w", TensorType::new(vec![n], DType::BF16))
        .unwrap();
    let widened = b
        .broadcast(b.cast(w, DType::F32).unwrap(), vec![1, 1, n])
        .unwrap();
    let out = b
        .binary(BinOp::Add, b.unary(UnOp::Neg, x).unwrap(), widened)
        .unwrap();
    let g = b.finish(out).unwrap();
    let og = passes_without_target(&g).unwrap();
    assert_eq!(casts(&og), 1, "a pointwise region keeps its widening cast");
}

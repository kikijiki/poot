//! `fuse_index_casts`: the `Cast` of an I32 word to F32 becomes a step of the pointwise region that reads it
//! or the one that writes it, so no `I32ToF32` kernel launches (POOT-1108). Every fold is also checked against
//! the evaluation of the graph as traced, bit for bit.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::op::{BinOp, FusedOp, OpKind, UnOp};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Builder, Eqn, Graph, Operand, Slot, ValueId};
use poot_graph_plan::passes_without_target as optimize;
use poot_tensor::{DType, HostTensor};

fn output_bits(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<u32> {
    eval(g, inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
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

fn i32_value(shape: Vec<usize>, words: Vec<i32>) -> Value {
    Value::from(HostTensor::i32(shape, words))
}

fn f32_value(shape: Vec<usize>, values: Vec<f32>) -> Value {
    Value::from(HostTensor::f32(shape, values))
}

fn casts(g: &Graph) -> usize {
    g.eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::Cast { .. }))
        .count()
}

fn regions(g: &Graph) -> Vec<&Eqn> {
    g.eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::Fused(_)))
        .collect()
}

fn cast_steps(eqn: &Eqn) -> usize {
    let OpKind::Fused(region) = &eqn.op else {
        return 0;
    };
    region
        .steps
        .iter()
        .filter(|step| matches!(step.op, FusedOp::Cast { to: DType::F32 }))
        .count()
}

/// The position of a decode step, cast to f32 and compared with an iota: the sliding-window mask shape every
/// attention layer traces. The cast reaches the compare through a reshape and a broadcast.
#[test]
fn a_position_cast_becomes_a_step_of_the_mask_region_that_reads_it() {
    let n = 8usize;
    let b = Builder::new();
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1], DType::I32));
    let x = b.constant("x", TensorType::f32(vec![1, n]));
    let pos_f = b.broadcast(b.reshape(b.cast(pos, DType::F32), vec![1, 1]), vec![1, n]);
    let cols = b.reshape(b.iota(n), vec![1, n]);
    let keep = b.binary(BinOp::Ge, pos_f, cols);
    let mask = b.binary_scalar(BinOp::Sub, keep, poot_graph_ir::Scalar::F32(1.0));
    let out = b.binary(BinOp::Add, x, mask);
    let (pos_id, x_id) = (pos.id(), x.id());
    let g = b.finish(out);

    let og = optimize(&g);
    og.validate().expect("the folded graph is valid");
    assert_eq!(casts(&g), 1);
    assert_eq!(
        casts(&og),
        0,
        "the position cast is a step, not an equation"
    );
    let reading: Vec<&Eqn> = regions(&og)
        .into_iter()
        .filter(|eqn| cast_steps(eqn) == 1)
        .collect();
    assert_eq!(reading.len(), 1, "one region casts the position itself");
    assert!(
        reading[0]
            .inputs
            .iter()
            .any(|input| matches!(input, Operand::Value(v) if og.aval(*v).dtype == DType::I32)),
        "the region reads an I32 word"
    );

    for position in [0, 3, 7] {
        let inputs = HashMap::from([
            (pos_id, i32_value(vec![1], vec![position])),
            (
                x_id,
                f32_value(vec![1, n], (0..n).map(|i| i as f32 * 0.25 - 1.0).collect()),
            ),
        ]);
        assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
    }
}

/// A witness flag: an exact-I32 compare, cast to f32, then moved by a transpose (no pointwise reader to take
/// the cast). The I32 producer writes the f32 result itself.
#[test]
fn an_i32_producer_writes_the_f32_the_cast_would_have_made() {
    let b = Builder::new();
    let a = b.slot_named(Slot::Token, "a", TensorType::new(vec![2, 3], DType::I32));
    let c = b.slot_named(Slot::Token, "c", TensorType::new(vec![2, 3], DType::I32));
    let flag = b.cast(b.binary(BinOp::GeU, a, c), DType::F32);
    let out = b.transpose(flag, vec![1, 0]);
    let (a_id, c_id) = (a.id(), c.id());
    let g = b.finish(out);

    let og = optimize(&g);
    og.validate().expect("the folded graph is valid");
    assert_eq!(casts(&g), 1);
    assert_eq!(casts(&og), 0);
    let producer = regions(&og);
    assert_eq!(producer.len(), 1, "the compare is now one region");
    assert_eq!(cast_steps(producer[0]), 1);
    assert_eq!(
        og.aval(producer[0].out).dtype,
        DType::F32,
        "the region writes the f32 flag"
    );

    // -1 and i32::MIN are the largest words unsigned: a signed compare would get these rows wrong.
    let inputs = HashMap::from([
        (a_id, i32_value(vec![2, 3], vec![-1, 0, 5, 7, i32::MIN, 2])),
        (c_id, i32_value(vec![2, 3], vec![1, 0, 5, 9, 1, 3])),
    ]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

/// An exact-I32 chain, its cast, and the float arithmetic that reads it are one region: one launch, as when the
/// whole chain computed in f32.
#[test]
fn an_i32_chain_its_cast_and_the_float_arithmetic_after_it_are_one_region() {
    let b = Builder::new();
    let a = b.slot_named(Slot::Token, "a", TensorType::new(vec![2, 4], DType::I32));
    let c = b.slot_named(Slot::Token, "c", TensorType::new(vec![2, 4], DType::I32));
    let w = b.constant("w", TensorType::f32(vec![2, 4]));
    let hit = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Ge, a, c),
        b.binary(BinOp::Ge, c, a),
    );
    let out = b.binary(BinOp::Mul, b.cast(hit, DType::F32), w);
    let (a_id, c_id, w_id) = (a.id(), c.id(), w.id());
    let g = b.finish(out);

    let og = optimize(&g);
    og.validate().expect("the folded graph is valid");
    assert_eq!(casts(&og), 0);
    let all = regions(&og);
    assert_eq!(
        all.len(),
        1,
        "compare, product, cast and scale are one region"
    );
    assert_eq!(og.eqns.len(), 1, "nothing else launches");
    assert_eq!(cast_steps(all[0]), 1);

    let inputs = HashMap::from([
        (a_id, i32_value(vec![2, 4], vec![1, 2, 3, 4, 5, 6, 7, 8])),
        (c_id, i32_value(vec![2, 4], vec![1, 3, 3, 0, 5, 7, 7, 9])),
        (
            w_id,
            f32_value(vec![2, 4], vec![0.5, -1.5, 2.0, 8.0, -0.0, 3.0, 1.0, 9.0]),
        ),
    ]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

/// A cast read by two standalone pointwise equations (a compare and a scale): each becomes a one-step region
/// that casts the I32 word, so the cast launches nowhere.
#[test]
fn a_cast_read_by_lone_pointwise_equations_folds_into_each() {
    let n = 6usize;
    let b = Builder::new();
    let pos = b.slot(Slot::Pos, TensorType::new(vec![n], DType::I32));
    let cols = b.iota(n);
    let pos_f = b.cast(pos, DType::F32);
    let keep = b.binary(BinOp::Ge, pos_f, cols);
    let scaled = b.binary_scalar(BinOp::Mul, pos_f, poot_graph_ir::Scalar::F32(0.5));
    // A reduce keeps the two readers from becoming one region.
    let out = b.binary(
        BinOp::Add,
        b.reduce(poot_graph_ir::RedOp::Sum, keep, 0, true),
        b.reduce(poot_graph_ir::RedOp::Sum, scaled, 0, true),
    );
    let pos_id = pos.id();
    let g = b.finish(out);

    let og = optimize(&g);
    og.validate().expect("the folded graph is valid");
    assert_eq!(casts(&og), 0);
    assert_eq!(
        regions(&og).iter().filter(|e| cast_steps(e) == 1).count(),
        2,
        "each reader casts the word itself"
    );
    let inputs = HashMap::from([(pos_id, i32_value(vec![n], vec![0, 1, 4, 5, 2, 9]))]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

/// What a fold must not touch: a cast with a reader that is not a pointwise equation, a cast result that is
/// the graph output, and a reader that already reads the I32 word as itself (a select condition shares its
/// leaf with the cast).
#[test]
fn a_cast_no_pointwise_kernel_can_absorb_stays_a_cast() {
    // A contraction reads the cast: it has no I32 lane, so the cast stays a launch.
    let b = Builder::new();
    let pos = b.slot(Slot::Pos, TensorType::new(vec![1, 4], DType::I32));
    let w = b.constant("w", TensorType::f32(vec![4, 3]));
    let out = b.matmul(b.cast(pos, DType::F32), w);
    let g = b.finish(out);
    assert_eq!(casts(&optimize(&g)), 1, "a matmul operand keeps its cast");

    // The graph output is the cast itself: its value is read from outside, with no reader to fold into.
    let b = Builder::new();
    let pos = b.slot(Slot::Pos, TensorType::new(vec![4], DType::I32));
    let out = b.cast(pos, DType::F32);
    let g = b.finish(out);
    assert_eq!(casts(&optimize(&g)), 1, "a cast that is the output stays");

    // The select's condition is the I32 word, its branch the cast of the same word: one leaf, two meanings.
    let b = Builder::new();
    let pos = b.slot(Slot::Pos, TensorType::new(vec![4], DType::I32));
    let x = b.constant("x", TensorType::f32(vec![4]));
    let (pos_id, x_id) = (pos.id(), x.id());
    let out = b.select(pos, b.cast(pos, DType::F32), x);
    let g = b.finish(out);
    let og = optimize(&g);
    assert_eq!(casts(&og), 1, "a reader of the word itself keeps the cast");
    let inputs = HashMap::from([
        (pos_id, i32_value(vec![4], vec![0, 2, 0, -3])),
        (x_id, f32_value(vec![4], vec![1.5, 2.5, 3.5, 4.5])),
    ]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

/// A producer with another reader keeps its I32 words for them: the cast folds into the float reader only.
#[test]
fn a_producer_another_equation_reads_is_not_rewritten() {
    let b = Builder::new();
    let a = b.slot_named(Slot::Token, "a", TensorType::new(vec![4], DType::I32));
    let c = b.slot_named(Slot::Token, "c", TensorType::new(vec![4], DType::I32));
    let w = b.constant("w", TensorType::f32(vec![4]));
    let flag = b.binary(BinOp::GeU, a, c);
    let scaled = b.binary(BinOp::Mul, b.cast(flag, DType::F32), w);
    // The same I32 flag, summed as words by an exact-I32 op the cast does not touch.
    let other = b.cast(b.unary(UnOp::Not, flag), DType::F32);
    let out = b.binary(BinOp::Add, scaled, other);
    let (a_id, c_id, w_id) = (a.id(), c.id(), w.id());
    let g = b.finish(out);

    let og = optimize(&g);
    og.validate().expect("the folded graph is valid");
    assert_eq!(casts(&og), 0);
    let inputs = HashMap::from([
        (a_id, i32_value(vec![4], vec![1, 5, 2, -1])),
        (c_id, i32_value(vec![4], vec![1, 6, 0, 3])),
        (w_id, f32_value(vec![4], vec![1.0, 2.0, 3.0, 4.0])),
    ]);
    assert_eq!(output_bits(&g, &inputs), output_bits(&og, &inputs));
}

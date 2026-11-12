//! POOT-1564 SC-j: `slice_of_concat` is exact. A slice whose range lies inside one part of the concat it
//! reads becomes a read of that part, and the oracle (`poot_eval::eval`) returns the same bits before and
//! after; a range that straddles two parts keeps the concat.
//!
//! Integration test for the reason `fusion_soundness.rs` gives: the oracle crate depends back on this one.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{BinOp, Builder, Graph, OpKind, Operand, TensorType, ValueId};
use poot_graph_plan::{dce, slice_of_concat};
use poot_tensor::HostTensor;

/// A conv window `[2, 3, 4]` and the new row `[2, 1, 4]` concatenated on axis 1, as the windowed causal
/// conv builds them, read by `slice(start..end)` of axis 1 and doubled, so the slice is an interior value.
fn windowed(start: usize, end: usize) -> Graph {
    let b = Builder::for_tests();
    let window = b
        .constant("window", TensorType::f32(vec![2, 3, 4]))
        .unwrap();
    let x = b.constant("x", TensorType::f32(vec![2, 1, 4])).unwrap();
    let extended = b.concat(1, &[window, x]).unwrap();
    let tap = b.slice(extended, 1, start, end).unwrap();
    let out = b.binary(BinOp::Add, tap, tap).unwrap();
    b.finish(out).unwrap()
}

fn output(g: &Graph) -> Vec<u32> {
    let inputs: HashMap<ValueId, Value> = g
        .inputs()
        .iter()
        .map(|&id| {
            let shape = g.aval(id).shape.clone();
            let n: usize = shape.iter().product();
            let data = (0..n)
                .map(|i| ((i as f32) * 0.37 + id.index() as f32).sin())
                .collect();
            (id, Value::from(HostTensor::f32(shape, data)))
        })
        .collect();
    let host = eval(g, &inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    host.as_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}

fn concats(g: &Graph) -> usize {
    g.eqns()
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::Concat { .. }))
        .count()
}

/// Mutation: let `part_containing` accept a range that ends one element past the part (`range.end <= to`
/// becomes `range.end <= to + 1`): the `2..4` straddle is rewritten to a slice of the window past its end
/// and the oracle comparison (or the slice's own shape check) goes red.
#[test]
fn slice_of_concat_simplifies_exactly() {
    // Inside the window (0..1, 1..3), exactly the new row (3..4, a whole part), and straddling (2..4).
    for (start, end, simplified) in [(0, 1, true), (1, 3, true), (3, 4, true), (2, 4, false)] {
        let g = windowed(start, end);
        let rewritten = slice_of_concat(&g).unwrap();
        assert_eq!(
            output(&rewritten),
            output(&g),
            "{start}..{end}: the rewrite changes the value"
        );
        let rewritten = dce(&rewritten).unwrap();
        assert_eq!(
            concats(&rewritten),
            usize::from(!simplified),
            "{start}..{end}: the concat is gone exactly when the range is inside one part"
        );
    }
}

/// A slice that is the whole part and also a pinned value (the graph output) keeps its own buffer: the
/// pass leaves it a slice of the concat rather than aliasing the part.
#[test]
fn a_pinned_whole_part_slice_keeps_its_equation() {
    let b = Builder::for_tests();
    let window = b
        .constant("window", TensorType::f32(vec![2, 3, 4]))
        .unwrap();
    let x = b.constant("x", TensorType::f32(vec![2, 1, 4])).unwrap();
    let extended = b.concat(1, &[window, x]).unwrap();
    let last = b.slice(extended, 1, 3, 4).unwrap();
    let g = b.finish(last).unwrap();

    let rewritten = slice_of_concat(&g).unwrap();
    assert_eq!(output(&rewritten), output(&g));
    let slice = rewritten
        .eqns()
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::Slice { .. }))
        .expect("the pinned slice keeps its equation");
    assert!(
        matches!(slice.inputs.as_slice(), [Operand::Value(source)] if *source == extended.id()),
        "the pinned slice still reads the concat"
    );
}

//! POOT-726: what the interval analysis proves for each rule, on graphs built through the production
//! `Builder`. The `compile` verdicts over the same facts are `poot-graph-plan`'s (SC-001, SC-002).

use super::*;
use crate::builder::Builder;
use crate::graph::{Operand, Slot};
use crate::types::{DType, TensorType};

const UPPER: usize = 4;

fn range_of(builder: Builder, value: crate::Traced) -> IndexRange {
    let id = value.id();
    let graph = builder.finish(value).unwrap();
    value_ranges(&graph)[id].expect("an I32 value has a range")
}

fn r(lo: i64, hi_exclusive: i64) -> IndexRange {
    IndexRange::new(lo, hi_exclusive).unwrap()
}

#[test]
fn a_float_value_has_no_range_and_an_undeclared_i32_input_is_every_word() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![2])).unwrap();
    let word = b
        .constant("w", TensorType::new(vec![2], DType::I32))
        .unwrap();
    let sum = b.binary_scalar(BinOp::Add, x, Scalar::F32(1.0)).unwrap();
    let out = b.cast(word, DType::F32).unwrap();
    let graph = b.finish(out).unwrap();
    let ranges = value_ranges(&graph);
    assert_eq!(ranges[x.id()], None);
    assert_eq!(ranges[sum.id()], None);
    assert_eq!(ranges[word.id()], Some(IndexRange::I32));
}

#[test]
fn iota_argtopk_and_comparisons_are_bounded_by_their_definitions() {
    let b = Builder::for_tests();
    let iota = b.iota_i32(7).unwrap();
    assert_eq!(range_of(b, iota), r(0, 7));

    let b = Builder::for_tests();
    let scores = b.constant("scores", TensorType::f32(vec![3, 6])).unwrap();
    let ids = b.arg_top_k(scores, 2).unwrap();
    assert_eq!(range_of(b, ids), r(0, 6));

    for op in [BinOp::Ge, BinOp::GeU] {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![3], DType::I32))
            .unwrap();
        let flag = b.binary_scalar(op, x, Scalar::I32(5)).unwrap();
        assert_eq!(range_of(b, flag), r(0, 2), "{op:?}");
    }
}

#[test]
fn arithmetic_is_interval_arithmetic_and_a_possible_wrap_is_every_word() {
    let b = Builder::for_tests();
    let pos = b
        .slot_ranged(
            Slot::Pos,
            TensorType::new(vec![4], DType::I32),
            IndexRange::below(10).unwrap(),
        )
        .unwrap();
    let add = b.binary_scalar(BinOp::Add, pos, Scalar::I32(5)).unwrap();
    let sub = b.binary_scalar(BinOp::Sub, pos, Scalar::I32(5)).unwrap();
    let mul = b.binary_scalar(BinOp::Mul, pos, Scalar::I32(-3)).unwrap();
    let max = b.binary_scalar(BinOp::Max, pos, Scalar::I32(7)).unwrap();
    let wraps = b
        .binary_scalar(BinOp::Add, pos, Scalar::I32(i32::MAX))
        .unwrap();
    let out = b.concat(0, &[add, sub, mul, max, wraps]).unwrap();
    let graph = b.finish(out).unwrap();
    let ranges = value_ranges(&graph);
    assert_eq!(ranges[pos.id()], Some(r(0, 10)));
    assert_eq!(ranges[add.id()], Some(r(5, 15)));
    assert_eq!(ranges[sub.id()], Some(r(-5, 5)));
    assert_eq!(ranges[mul.id()], Some(r(-27, 1)));
    assert_eq!(ranges[max.id()], Some(r(7, 10)));
    // `9 + i32::MAX` wraps modulo 2^32: the sum is not bounded by its interval.
    assert_eq!(ranges[wraps.id()], Some(IndexRange::I32));
    assert_eq!(ranges[out.id()], Some(IndexRange::I32));
}

#[test]
fn select_is_the_union_of_its_branches_and_movement_keeps_the_range() {
    let b = Builder::for_tests();
    let low = b.iota_i32(3).unwrap();
    let high = b.binary_scalar(BinOp::Add, low, Scalar::I32(10)).unwrap();
    let c = b
        .constant("c", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let flag = b.binary_scalar(BinOp::Ge, c, Scalar::I32(1)).unwrap();
    let picked = b.select(flag, low, high).unwrap();
    let moved = b.reshape(picked, vec![3, 1]).unwrap();
    let moved = b.transpose(moved, vec![1, 0]).unwrap();
    let moved = b.slice(moved, 1, 0, 2).unwrap();
    assert_eq!(range_of(b, moved), r(0, 13));
}

#[test]
fn a_gather_reads_a_subset_of_its_table() {
    let b = Builder::for_tests();
    let table = b
        .binary_scalar(BinOp::Add, b.iota_i32(4).unwrap(), Scalar::I32(100))
        .unwrap();
    let index = b
        .constant("index", TensorType::new(vec![5], DType::I32))
        .unwrap();
    let gathered = b.gather(table, 0, index).unwrap();
    assert_eq!(range_of(b, gathered), r(100, 104));
}

#[test]
fn the_guard_output_is_below_its_bound_without_matching_the_chain() {
    let b = Builder::for_tests();
    let raw = b
        .constant("raw", TensorType::new(vec![6], DType::I32))
        .unwrap();
    let guard = b.guard_index_bounds(raw, UPPER).unwrap();
    // The raw selector proves nothing; the guarded one is `[0, UPPER)` and its successor `[1, UPPER]`.
    let successor = b
        .binary_scalar(BinOp::Add, guard.guarded, Scalar::I32(1))
        .unwrap();
    let (raw_id, guarded, succ) = (raw.id(), guard.guarded.id(), successor.id());
    let graph = b.finish(successor).unwrap();
    let ranges = value_ranges(&graph);
    assert_eq!(ranges[raw_id], Some(IndexRange::I32));
    assert_eq!(ranges[guarded], Some(r(0, UPPER as i64)));
    assert_eq!(ranges[succ], Some(r(1, UPPER as i64 + 1)));
}

/// A range that rests on an index guard's clamp names the guard's condition; one that rests on a declared
/// slot or a definition does not, and a value computed from a clamped one inherits the dependency.
#[test]
fn clamped_ranges_name_the_guard_condition_they_rest_on() {
    let b = Builder::for_tests();
    let raw = b
        .constant("raw", TensorType::new(vec![4], DType::I32))
        .unwrap();
    let guard = b.guard_index_bounds(raw, UPPER).unwrap();
    let successor = b
        .binary_scalar(BinOp::Add, guard.guarded, Scalar::I32(1))
        .unwrap();
    let pos = b
        .slot_ranged(
            Slot::Pos,
            TensorType::new(vec![4], DType::I32),
            IndexRange::below(UPPER).unwrap(),
        )
        .unwrap();
    let declared = b.binary_scalar(BinOp::Add, pos, Scalar::I32(1)).unwrap();
    let iota = b.iota_i32(UPPER).unwrap();
    let out = b.concat(0, &[successor, declared, iota]).unwrap();
    let graph = b.finish(out).unwrap();
    // The guard's condition is the comparison its `Select` reads.
    let condition = graph
        .eqns()
        .iter()
        .find_map(|eqn| match (&eqn.op, eqn.inputs.as_slice()) {
            (OpKind::Select, [Operand::Value(condition), ..]) => Some(*condition),
            _ => None,
        })
        .expect("the guard's select");
    let facts = value_ranges(&graph);
    assert_eq!(facts.clamps(guard.guarded.id()), [condition]);
    assert_eq!(facts.clamps(successor.id()), [condition]);
    assert!(facts.clamps(raw.id()).is_empty());
    assert!(facts.clamps(pos.id()).is_empty());
    assert!(facts.clamps(declared.id()).is_empty());
    assert!(facts.clamps(iota.id()).is_empty());
}

#[test]
fn a_comparison_narrows_only_the_value_it_compares() {
    // Signed: `x < 4` where `Ge(x, 4)` is zero, `x >= 4` where it is nonzero.
    let b = Builder::for_tests();
    let x = b
        .slot_ranged(
            Slot::Token,
            TensorType::new(vec![3], DType::I32),
            r(-10, 10),
        )
        .unwrap();
    let other = b
        .constant("other", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let flag = b.binary_scalar(BinOp::Ge, x, Scalar::I32(4)).unwrap();
    let clamped_below = b.select(flag, other, x).unwrap();
    let clamped_above = b.select(flag, x, other).unwrap();
    let (below, above) = (clamped_below.id(), clamped_above.id());
    let out = b.concat(0, &[clamped_below, clamped_above]).unwrap();
    let graph = b.finish(out).unwrap();
    let ranges = value_ranges(&graph);
    // The other branch is unbounded, so each union is every word: only the narrowed branch is checked
    // through the guard-shaped case below.
    assert_eq!(ranges[below], Some(IndexRange::I32));
    assert_eq!(ranges[above], Some(IndexRange::I32));

    // `Select(GeU(x, 4), 1, x)`: the false branch is `x` under `x <u 4`.
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let invalid = b.binary_scalar(BinOp::GeU, x, Scalar::I32(4)).unwrap();
    let one = b
        .binary_scalar(BinOp::Max, b.iota_i32(3).unwrap(), Scalar::I32(1))
        .unwrap();
    let picked = b.select(invalid, one, x).unwrap();
    assert_eq!(range_of(b, picked), r(0, 4));

    // The condition compares another value than the branch: nothing is narrowed.
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let y = b
        .constant("y", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let invalid = b.binary_scalar(BinOp::GeU, y, Scalar::I32(4)).unwrap();
    let picked = b.select(invalid, b.iota_i32(3).unwrap(), x).unwrap();
    assert_eq!(range_of(b, picked), IndexRange::I32);
}

#[test]
fn a_slot_range_is_declared_per_slot_and_must_bound_an_i32() {
    let b = Builder::for_tests();
    let ranged = b
        .slot_ranged(
            Slot::Token,
            TensorType::new(vec![2], DType::I32),
            r(0, 32_000),
        )
        .unwrap();
    let plain = b
        .slot(Slot::Pos, TensorType::new(vec![2], DType::I32))
        .unwrap();
    let out = b.concat(0, &[ranged, plain]).unwrap();
    let graph = b.finish(out).unwrap();
    let ranges = value_ranges(&graph);
    assert_eq!(ranges[ranged.id()], Some(r(0, 32_000)));
    assert_eq!(ranges[plain.id()], Some(IndexRange::I32));
    assert_eq!(graph.meta(ranged.id()).range(), Some(r(0, 32_000)));
    assert_eq!(graph.meta(plain.id()).range(), None);
}

#[test]
fn a_float_slot_cannot_declare_a_range() {
    let b = Builder::for_tests();
    assert!(matches!(
        b.slot_ranged(Slot::Token, TensorType::f32(vec![2]), r(0, 4)),
        Err(crate::BuilderError::RangeNotI32 {
            dtype: DType::F32,
            ..
        })
    ));
    assert_eq!(b.generation(), 0, "a refused declaration appends nothing");
}

#[test]
fn range_values_are_checked_against_i32_words() {
    assert_eq!(IndexRange::new(3, 3), None);
    assert_eq!(IndexRange::new(0, i64::from(i32::MAX) + 2), None);
    assert!(r(0, 4).contains(3) && !r(0, 4).contains(4) && !r(0, 4).contains(-1));
    assert!(r(0, 4).within(&IndexRange::F32_EXACT));
    assert!(!r(0, (1 << 24) + 2).within(&IndexRange::F32_EXACT));
    assert!(
        IndexRange::new(-(1 << 24), (1 << 24) + 1)
            .unwrap()
            .within(&IndexRange::F32_EXACT)
    );
}

/// POOT-1114: a const table declares its range at construction, and a `Gather` of it inherits that
/// range through the movement rule. Mutation: have `Builder::declare_constant` record `None`; the
/// table and the gathered words are every `i32` word and this row goes red.
#[test]
fn a_ranged_const_table_declares_its_range_and_a_gather_inherits_it() {
    let b = Builder::for_tests();
    let table = b
        .constant_ranged(
            "route.table",
            TensorType::new(vec![5, 2], DType::I32),
            r(0, 8),
        )
        .unwrap();
    let tokens = b
        .constant("tokens", TensorType::new(vec![3], DType::I32))
        .unwrap();
    let rows = b.gather(table, 0, tokens).unwrap();
    let graph = b.finish(rows).unwrap();
    let ranges = value_ranges(&graph);
    assert_eq!(ranges[table.id()], Some(r(0, 8)));
    assert_eq!(ranges[rows.id()], Some(r(0, 8)));
}

#[test]
fn a_ranged_const_must_be_i32() {
    let b = Builder::for_tests();
    assert!(matches!(
        b.constant_ranged("w", TensorType::f32(vec![2]), r(0, 4)),
        Err(crate::BuilderError::RangeNotI32 {
            dtype: DType::F32,
            ..
        })
    ));
    assert_eq!(b.generation(), 0, "a refused declaration appends nothing");
}

/// The I32 range of `F32 -> I32` casts of expressions over two F32 inputs, to test what `f32_bounds` proves.
fn cast_range(build: impl FnOnce(&Builder, crate::Traced) -> crate::Traced) -> IndexRange {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![4])).unwrap();
    let indicator = b.binary(BinOp::Ge, x, y).unwrap();
    let expr = build(&b, indicator);
    let cast = b.cast(expr, DType::I32).unwrap();
    range_of(b, cast)
}

/// A comparison is `{0, 1}`, so an indicator times its complement is in `[0, 1]` and its cast is proven
/// `[0, 2)`. Mutation: widen the `Ge` bounds in `f32_bounds` to `(0.0, 2.0)`; the product is `[0, 4]` and this
/// row goes red (the control row of the next test reads `[0, 7)` for `[0, 4)` too).
#[test]
fn an_indicator_times_its_complement_casts_to_zero_or_one() {
    let range = cast_range(|b, ind| {
        let not = b
            .binary_scalar(
                BinOp::Add,
                b.binary_scalar(BinOp::Mul, ind, Scalar::F32(-1.0)).unwrap(),
                Scalar::F32(1.0),
            )
            .unwrap();
        b.binary(BinOp::Mul, ind, not).unwrap()
    });
    assert_eq!(range, r(0, 2));
}

/// Bounds the F32 arithmetic could round (above 2^24) or that are not whole numbers prove nothing: the cast is
/// every word. Mutation: drop the `EXACT_F32` check in `exact_bounds`; the first case is proven `[0, 20000001)`
/// and this row goes red.
#[test]
fn bounds_past_the_exact_f32_integers_or_not_whole_prove_nothing() {
    let scaled = |scale: f32| {
        cast_range(move |b, ind| {
            b.binary_scalar(BinOp::Mul, ind, Scalar::F32(scale))
                .unwrap()
        })
    };
    assert_eq!(scaled(20_000_000.0), IndexRange::I32, "above 2^24");
    assert_eq!(scaled(0.5), IndexRange::I32, "not whole");
    assert_eq!(
        scaled(3.0),
        r(0, 4),
        "control: a whole bound within 2^24 is proven"
    );
}

/// An operand nothing bounds (an F32 input) makes the product unbounded: the cast is every word.
#[test]
fn an_unbounded_f32_operand_proves_nothing() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![4])).unwrap();
    let ind = b.binary(BinOp::Ge, x, y).unwrap();
    let product = b.binary(BinOp::Mul, ind, x).unwrap();
    let cast = b.cast(product, DType::I32).unwrap();
    assert_eq!(range_of(b, cast), IndexRange::I32);
}

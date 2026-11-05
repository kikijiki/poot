//! POOT-1451 oracle rows: the block-aligned expert grouping (`ExpertGroup { experts, block }`) and the indexed
//! contraction over it (`IndexedMatMul { rows: Grouped }`). The grouped contraction is the per-row one, bit for bit,
//! at every block; an id outside `0..E` is the typed refusal it is per row; a grouping that is not a partition of
//! the rows has no defined result and is refused by name.

use std::collections::HashMap;

use crate::{EvalError, EvalOptions, GroupingFault, IndexFault, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{ExpertGrouping, Graph, ValueId};
use poot_tensor::{DType, HostTensor};

const EXPERTS: usize = 4;
const K: usize = 5;
const N: usize = 3;

struct Fixture {
    graph: Graph,
    inputs: HashMap<ValueId, Value>,
}

/// `x [rows, K]` against an `[E, K, N]` stack through `ids`: per row when `block` is `None`, else through the
/// grouping of the ids in tiles of `block`.
fn contraction(ids: &[i32], block: Option<usize>) -> Fixture {
    let rows = ids.len();
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, K])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![EXPERTS, K, N]))
        .unwrap();
    let id = b
        .constant("ids", TensorType::new(vec![rows], DType::I32))
        .unwrap();
    let out = match block {
        None => b.indexed_matmul(x, w, id).unwrap(),
        Some(block) => {
            let groups = b.expert_group(id, EXPERTS, block).unwrap();
            b.grouped_indexed_matmul(x, w, groups, block).unwrap()
        }
    };
    let graph = b.finish(out).unwrap();
    let inputs = HashMap::from([
        (
            x.id(),
            Value::from(HostTensor::f32(
                vec![rows, K],
                (0..rows * K)
                    .map(|i| ((i * 7) % 13) as f32 * 0.37 - 2.0)
                    .collect(),
            )),
        ),
        (
            w.id(),
            Value::from(HostTensor::f32(
                vec![EXPERTS, K, N],
                (0..EXPERTS * K * N)
                    .map(|i| ((i * 5) % 11) as f32 * 0.29 - 1.4)
                    .collect(),
            )),
        ),
        (
            id.id(),
            Value::from(HostTensor::i32(vec![rows], ids.to_vec())),
        ),
    ]);
    Fixture { graph, inputs }
}

fn run(fixture: &Fixture) -> Result<HostTensor, EvalError> {
    Ok(
        eval(&fixture.graph, &fixture.inputs, EvalOptions::default())?
            .output
            .into_host()
            .expect("a dense result"),
    )
}

fn bits(t: &HostTensor) -> Vec<u32> {
    t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect()
}

/// The grouped contraction is the per-row contraction bit for bit at every block (one row a tile, partial tiles,
/// one tile per expert), over routings with empty experts, one expert taking every row and more experts than rows.
///
/// Mutation (observed red): in `row_experts` read a grouped row's expert from the tile after its own (`position /
/// block + 1`); the rows read another expert's weight and the bits differ.
#[test]
fn a_grouped_contraction_is_the_per_row_one_at_every_block() {
    for ids in [
        vec![2, 0, 3, 3, 1, 0, 2, 2, 2, 1, 0],
        vec![1; 9],
        vec![3, 0],
    ] {
        let direct = run(&contraction(&ids, None)).unwrap();
        for block in [1, 2, 4, 16] {
            let grouped = run(&contraction(&ids, Some(block))).unwrap();
            assert_eq!(bits(&grouped), bits(&direct), "{ids:?} at block {block}");
        }
    }
}

/// An id outside `0..E` is the per-row contraction's typed refusal through the grouping too: its row lands in the
/// invalid bucket, whose tile names no expert, and the contraction refuses it as an index fault (on a device, the
/// row is NaN).
///
/// Mutation (observed red): send invalid ids to bucket 0 in `expert_group` (`unwrap_or(0)`); the row reads expert
/// 0 and the grouped contraction evaluates instead of refusing.
#[test]
fn an_invalid_id_is_refused_through_the_grouping() {
    for bad in [-1, EXPERTS as i32, i32::MAX] {
        let ids = [1, bad, 0, 2];
        assert!(matches!(
            run(&contraction(&ids, None)),
            Err(EvalError::Index(_))
        ));
        let refusal = run(&contraction(&ids, Some(2))).expect_err("an invalid id has no expert");
        let EvalError::Index(fault) = &refusal else {
            panic!("{bad}: {refusal}");
        };
        let IndexFault { len, .. } = **fault;
        assert_eq!(len, EXPERTS, "{bad}: {refusal}");
    }
}

/// The grouping's own layout: buckets in order, each bucket's rows in row order padded with the sentinel to whole
/// tiles, invalid ids last, the tail past the padded total all sentinels and tiles of no expert.
///
/// Mutation (observed red): pad a bucket's run to `placed / block * block` in `expert_group` (round down); expert
/// 0's three rows at block 2 leave its second tile out and the layout differs.
#[test]
fn the_grouping_pads_each_buckets_rows_to_whole_tiles() {
    let ids = [2, 0, -3, 0, 2, 0];
    let b = Builder::for_tests();
    let id = b
        .constant("ids", TensorType::new(vec![ids.len()], DType::I32))
        .unwrap();
    let grouping = b.expert_group(id, 3, 2).unwrap();
    let graph = b.finish(grouping).unwrap();
    let inputs = HashMap::from([(
        id.id(),
        Value::from(HostTensor::i32(vec![ids.len()], ids.to_vec())),
    )]);
    let out = eval(&graph, &inputs, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    let layout = ExpertGrouping {
        rows: 6,
        experts: 3,
        block: 2,
    };
    assert_eq!(layout.tiles(), 5, "(6 rows + 4 buckets * 1) / 2");
    let s = 6;
    assert_eq!(
        out.as_i32().unwrap(),
        [
            // bucket 0: rows 1 3 5; bucket 2: rows 0 4; bucket 3 (invalid): row 2; then the empty tail
            1, 3, 5, s, 0, 4, 2, s, s, s, //
            // tile buckets, the last an empty tile of no expert
            0, 0, 2, 3, 3,
        ]
    );
}

/// A grouping that is not a partition of the rows has no defined contraction: a row placed twice, a row never
/// placed and a word outside `0..=rows` are each refused by name.
///
/// Mutation (observed red): skip the `Repeated` check in `row_experts`; the twice-placed row evaluates and the first
/// case goes red.
#[test]
fn a_grouping_that_is_not_a_partition_is_refused_by_name() {
    let rows = 3;
    let layout = ExpertGrouping {
        rows,
        experts: EXPERTS,
        block: 2,
    };
    let run_with = |order: [i32; 6]| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![rows, K])).unwrap();
        let w = b
            .constant("w", TensorType::f32(vec![EXPERTS, K, N]))
            .unwrap();
        let groups = b
            .constant("groups", TensorType::new(vec![layout.len()], DType::I32))
            .unwrap();
        let out = b.grouped_indexed_matmul(x, w, groups, 2).unwrap();
        let graph = b.finish(out).unwrap();
        let mut words = order.to_vec();
        words.resize(layout.positions(), rows as i32);
        words.extend((0..layout.tiles()).map(|t| (t % EXPERTS) as i32));
        let inputs = HashMap::from([
            (
                x.id(),
                Value::from(HostTensor::f32(vec![rows, K], vec![1.0; rows * K])),
            ),
            (
                w.id(),
                Value::from(HostTensor::f32(
                    vec![EXPERTS, K, N],
                    vec![1.0; EXPERTS * K * N],
                )),
            ),
            (
                groups.id(),
                Value::from(HostTensor::i32(vec![layout.len()], words)),
            ),
        ]);
        eval(&graph, &inputs, EvalOptions::default()).map(|_| ())
    };
    let fault = |order| match run_with(order) {
        Err(EvalError::ExpertGrouping { fault, .. }) => fault,
        other => panic!("{order:?}: {other:?}"),
    };
    let s = rows as i32;
    assert_eq!(
        fault([0, 1, 1, s, 2, s]),
        GroupingFault::Repeated { row: 1 }
    );
    assert_eq!(fault([0, 1, s, s, s, s]), GroupingFault::Missing { row: 2 });
    assert_eq!(
        fault([0, 1, 2, 9, s, s]),
        GroupingFault::Word {
            position: 3,
            word: 9
        }
    );
    assert!(
        run_with([0, 1, 2, s, s, s]).is_ok(),
        "a partition evaluates"
    );
}

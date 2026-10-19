//! The block selection composite (POOT-1143 stage 2) through the real Builder, inference, decomposition and
//! evaluator seams: its primitive definition against an independent plain-Rust selection, the typed
//! projections of its one packed result, its typing refusals, and the oracle's bound on the quadratic
//! definition.

use std::collections::HashMap;

use poot_graph_ir::test_support::block_top_k::{reference_row, selection_cases};
use poot_graph_ir::{BlockTopKAttrs, Builder, Outputs, ShapeError, TensorType, Traced, ValueId};
use poot_tensor::{DType, HostTensor};

use crate::{EvalError, EvalOptions, Value, eval};

/// A graph over scores `[rows, 1, tokens, blocks]` and positions `[rows, tokens]` whose output is the
/// op's packed result.
struct Fixture {
    graph: poot_graph_ir::Graph,
    scores: ValueId,
    pos: ValueId,
    packed: Traced,
}

fn fixture(rows: usize, tokens: usize, blocks: usize, attrs: BlockTopKAttrs) -> Fixture {
    let b = Builder::for_tests();
    let scores = b
        .constant("scores", TensorType::f32(vec![rows, 1, tokens, blocks]))
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![rows, tokens], DType::I32))
        .unwrap();
    let result = b.block_top_k(attrs, scores, pos).unwrap();
    let graph = b.finish(Outputs::new(result.packed)).unwrap();
    Fixture {
        graph,
        scores: scores.id(),
        pos: pos.id(),
        packed: result.packed,
    }
}

/// The packed rows of `scores` (one `[blocks]` row per query, `rows * tokens` queries, row-major) at `pos`.
fn run(
    rows: usize,
    tokens: usize,
    scores: &[Vec<f32>],
    pos: &[i32],
    attrs: BlockTopKAttrs,
) -> Result<Vec<Vec<i32>>, EvalError> {
    let blocks = scores[0].len();
    let fixture = fixture(rows, tokens, blocks, attrs);
    let values: HashMap<ValueId, Value> = HashMap::from([
        (
            fixture.scores,
            HostTensor::f32(vec![rows, 1, tokens, blocks], scores.concat()).into(),
        ),
        (
            fixture.pos,
            HostTensor::i32(vec![rows, tokens], pos.to_vec()).into(),
        ),
    ]);
    let result = eval(&fixture.graph, &values, EvalOptions::default())?;
    let packed = result.output.into_host().expect("dense result");
    assert_eq!(
        packed.shape(),
        fixture.graph.aval(fixture.packed.id()).shape
    );
    Ok(packed
        .as_i32()
        .expect("an I32 result")
        .chunks(attrs.k + blocks)
        .map(<[i32]>::to_vec)
        .collect())
}

/// `got` equals `want` word for word; the failure names the first word that differs: an id slot or a block's
/// keep indicator.
fn assert_row(label: &str, got: &[i32], want: &[i32], k: usize) {
    assert_eq!(got.len(), want.len(), "{label}: row length");
    if let Some(at) = got.iter().zip(want).position(|(g, w)| g != w) {
        let what = if at < k {
            format!("id slot {at}")
        } else {
            format!("keep indicator of block {}", at - k)
        };
        panic!("{label}: {what}: got {}, want {}", got[at], want[at]);
    }
}

/// The definition equals the plain selection on every row of the selection cases: ids, `-1` padding and the
/// keep indicator, bit for bit (design section 5, tier 1). Mutations in `ops/block_top_k.rs`: rank all blocks
/// instead of the eligible ones (`stable_descending_rank` for `_masked`); the eligibility `Ge(pos, first)`
/// for `Ge(pos, last)`; ids ranked descending instead of ascending.
#[test]
fn the_definition_selects_what_the_plain_selection_selects() {
    for (label, scores, positions, ratio, k) in selection_cases() {
        let got = run(
            positions.len(),
            1,
            &scores,
            &positions,
            BlockTopKAttrs { ratio, k },
        )
        .unwrap();
        for (q, row) in got.iter().enumerate() {
            assert_row(
                &format!("{label}: query {q} at position {}", positions[q]),
                row,
                &reference_row(&scores[q], positions[q], ratio, k),
                k,
            );
        }
    }
}

/// Rows and tokens are separate axes: query `(r, t)` reads row `r`'s score row `t` and position `pos[r][t]`.
#[test]
fn queries_are_rows_by_tokens() {
    let (rows, tokens, ratio, k) = (2usize, 3usize, 2usize, 3usize);
    let scores: Vec<Vec<f32>> = (0..rows * tokens)
        .map(|q| poot_graph_ir::test_support::block_top_k::grid_scores(9, 1 << 20, q as u64))
        .collect();
    let pos = [3, 5, 9, 1, 7, 17];
    let got = run(rows, tokens, &scores, &pos, BlockTopKAttrs { ratio, k }).unwrap();
    for (q, row) in got.iter().enumerate() {
        assert_row(
            &format!("query {q}"),
            row,
            &reference_row(&scores[q], pos[q], ratio, k),
            k,
        );
    }
}

/// `ids` and `keep` are two distinct typed projections of the packed result: slices at the columns its
/// layout names, I32 throughout.
#[test]
fn ids_and_keep_are_typed_projections_of_one_packed_result() {
    let b = Builder::for_tests();
    let scores = b
        .constant("scores", TensorType::f32(vec![2, 1, 3, 9]))
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![2, 3], DType::I32))
        .unwrap();
    let result = b
        .block_top_k(BlockTopKAttrs { ratio: 2, k: 4 }, scores, pos)
        .unwrap();
    let g = b.finish(Outputs::new(result.packed)).unwrap();
    assert_eq!(g.aval(result.packed.id()).shape, vec![2, 1, 3, 13]);
    assert_eq!(g.aval(result.packed.id()).dtype, DType::I32);
    assert_eq!(g.aval(result.ids.id()).shape, vec![2, 1, 3, 4]);
    assert_eq!(g.aval(result.keep.id()).shape, vec![2, 1, 3, 9]);
    assert_ne!(result.ids.id(), result.keep.id());
}

/// The typing rule refuses what the definition cannot rank: a `k` above the blocks or zero, a zero ratio, a
/// score tensor that is not F32 `[R, 1, T, NB]`, positions that are not I32 `[R, T]`.
#[test]
fn the_typing_rule_refuses_what_it_cannot_rank() {
    let attrs = BlockTopKAttrs { ratio: 4, k: 3 };
    let ok = [
        TensorType::f32(vec![2, 1, 3, 8]),
        TensorType::new(vec![2, 3], DType::I32),
    ];
    assert!(attrs.layout(&ok).is_ok());
    let bad = |attrs: BlockTopKAttrs, ins: [TensorType; 2]| attrs.layout(&ins).unwrap_err();
    assert!(matches!(
        bad(BlockTopKAttrs { ratio: 4, k: 9 }, ok.clone()),
        ShapeError::BlockTopK {
            k: 9,
            blocks: 8,
            ..
        }
    ));
    assert!(matches!(
        bad(BlockTopKAttrs { ratio: 4, k: 0 }, ok.clone()),
        ShapeError::BlockTopK { k: 0, .. }
    ));
    assert!(matches!(
        bad(BlockTopKAttrs { ratio: 0, k: 3 }, ok.clone()),
        ShapeError::BlockTopK { ratio: 0, .. }
    ));
    assert!(matches!(
        bad(
            attrs,
            [TensorType::new(vec![2, 1, 3, 8], DType::I32), ok[1].clone()]
        ),
        ShapeError::DtypeOp { .. }
    ));
    assert!(matches!(
        bad(attrs, [ok[0].clone(), TensorType::f32(vec![2, 3])]),
        ShapeError::DtypeOp { .. }
    ));
    assert!(matches!(
        bad(attrs, [TensorType::f32(vec![2, 3, 8]), ok[1].clone()]),
        ShapeError::OperandRank {
            operand: "scores",
            ..
        }
    ));
    assert!(matches!(
        bad(attrs, [TensorType::f32(vec![2, 2, 3, 8]), ok[1].clone()]),
        ShapeError::OperandDim {
            operand: "scores",
            ..
        }
    ));
    assert!(matches!(
        bad(
            attrs,
            [ok[0].clone(), TensorType::new(vec![2, 4], DType::I32)]
        ),
        ShapeError::OperandDim { operand: "pos", .. }
    ));
}

/// The oracle ranks every pair of blocks, so it refuses a row past its bound with a typed error naming the
/// op instead of allocating the pairwise tensors (16384 blocks of one query is 2^28 pairs, 1 GiB per
/// tensor). This is the 128K-position case (32768 blocks) of the design; the device kernel has no such
/// bound. Mutation: raise `MAX_BLOCK_TOP_K_PAIRS` past 2^28; the row allocates instead of refusing.
#[test]
fn the_oracle_refuses_a_row_it_cannot_rank_quadratically() {
    let blocks = 16384;
    let error = run(
        1,
        1,
        &[vec![0.0; blocks]],
        &[4 * blocks as i32],
        BlockTopKAttrs { ratio: 4, k: 512 },
    )
    .unwrap_err();
    assert!(
        matches!(&error, EvalError::Unsupported { op, detail, .. } if *op == "block_top_k" && detail.contains("quadratic")),
        "{error:?}"
    );
}

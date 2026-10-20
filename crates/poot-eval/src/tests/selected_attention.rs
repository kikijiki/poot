//! The selected-attention composite (POOT-1143 stage 3) through the real Builder, inference, decomposition and
//! evaluator seams: its dense definition against an independent f64 attention over exactly the selected keys,
//! the d0 case (every complete block selected) against plain causal attention bit for bit, its typing
//! refusals, and the oracle's bound on the mask it builds.

use std::collections::HashMap;

use poot_graph_ir::test_support::composites;
use poot_graph_ir::test_support::selected_attention::{Case, cases, fill};
use poot_graph_ir::{
    Builder, Outputs, SelectedAttentionAttrs, ShapeError, TensorType, ValueId, ops,
};
use poot_tensor::{DType, HostTensor};

use crate::{EvalError, EvalOptions, Value, eval};

/// A graph over `q`, `k`, `v`, `ids` and `pos` constants whose output is the op's.
struct Fixture {
    graph: poot_graph_ir::Graph,
    ids: [ValueId; 5],
}

fn attrs(case: &Case) -> SelectedAttentionAttrs {
    SelectedAttentionAttrs {
        n_rep: case.n_rep,
        scale: case.scale(),
        softcap: None,
        ratio: case.ratio,
        k: case.k,
    }
}

fn fixture(case: &Case) -> Fixture {
    let b = Builder::for_tests();
    let (batch, hq, lq, lk, d) = (case.batch, case.hq, case.lq, case.lk, case.d);
    let hkv = case.hkv();
    let q = b
        .constant("q", TensorType::f32(vec![batch, hq, lq, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let ids = b
        .constant(
            "ids",
            TensorType::new(vec![batch, 1, lq, case.k], DType::I32),
        )
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![batch, lq], DType::I32))
        .unwrap();
    let out = composites::selected_attention(&b, attrs(case), [q, k, v, ids, pos]).unwrap();
    Fixture {
        graph: b.finish(Outputs::new(out)).unwrap(),
        ids: [q.id(), k.id(), v.id(), ids.id(), pos.id()],
    }
}

fn run(case: &Case, [q, k, v]: &[Vec<f32>; 3]) -> Result<Vec<f32>, EvalError> {
    let f = fixture(case);
    let (batch, hq, lq, lk, d, hkv) = (case.batch, case.hq, case.lq, case.lk, case.d, case.hkv());
    let values: HashMap<ValueId, Value> = HashMap::from([
        (
            f.ids[0],
            HostTensor::f32(vec![batch, hq, lq, d], q.clone()).into(),
        ),
        (
            f.ids[1],
            HostTensor::f32(vec![batch, hkv, lk, d], k.clone()).into(),
        ),
        (
            f.ids[2],
            HostTensor::f32(vec![batch, hkv, lk, d], v.clone()).into(),
        ),
        (
            f.ids[3],
            HostTensor::i32(vec![batch, 1, lq, case.k], case.ids.clone()).into(),
        ),
        (
            f.ids[4],
            HostTensor::i32(vec![batch, lq], case.pos.clone()).into(),
        ),
    ]);
    let result = eval(&f.graph, &values, EvalOptions::default())?;
    Ok(result
        .output
        .into_host()
        .expect("dense output")
        .as_f32()
        .unwrap()
        .to_vec())
}

/// The dense definition equals the independent attention over the selected keys within the f32 softmax's
/// rounding (the sums differ in order only), on every case. Mutations in `ops/selected_attention.rs`: drop the
/// causal penalty (the queries take later keys); read the unfinished block as complete (`Ge(at, first)` for
/// `Sub(started, complete)`); compare the ids against block `b + 1`.
#[test]
fn the_definition_attends_over_exactly_the_selected_keys() {
    for case in cases() {
        let operands = case.operands(11);
        let got = run(&case, &operands).unwrap();
        let want = case.reference(&operands[0], &operands[1], &operands[2]);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() <= 1e-5 * (1.0 + w.abs()),
                "{}: output element {i} (row {}, channel {}): got {g}, reference {w}",
                case.label,
                i / case.d,
                i % case.d
            );
        }
    }
}

/// SC-1143-11, the d0 case: with every complete block selected the op equals plain causal attention, bit for
/// bit on the oracle (the dropped keys weigh exactly zero in both). Mutation: drop one selected block from
/// `ids`; the queries past it differ.
#[test]
fn selecting_every_block_is_plain_causal_attention() {
    let case = cases()
        .into_iter()
        .find(|c| c.label == "every block selected")
        .unwrap();
    let operands = case.operands(5);
    let got = run(&case, &operands).unwrap();

    let b = Builder::for_tests();
    let (batch, hq, lq, lk, d, hkv) = (case.batch, case.hq, case.lq, case.lk, case.d, case.hkv());
    let q = b
        .constant("q", TensorType::f32(vec![batch, hq, lq, d]))
        .unwrap();
    let k = b
        .constant("k", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let v = b
        .constant("v", TensorType::f32(vec![batch, hkv, lk, d]))
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![batch, lq], DType::I32))
        .unwrap();
    // The causal mask over the capacity: the plain attention a dense layer runs.
    let mask = ops::causal_mask_from_pos(&b, pos, lk, None).unwrap();
    let out =
        ops::attention_masked_softcap(&b, q, k, v, case.n_rep, case.scale(), mask, None).unwrap();
    let graph = b.finish(Outputs::new(out)).unwrap();
    let values: HashMap<ValueId, Value> = HashMap::from([
        (
            q.id(),
            HostTensor::f32(vec![batch, hq, lq, d], operands[0].clone()).into(),
        ),
        (
            k.id(),
            HostTensor::f32(vec![batch, hkv, lk, d], operands[1].clone()).into(),
        ),
        (
            v.id(),
            HostTensor::f32(vec![batch, hkv, lk, d], operands[2].clone()).into(),
        ),
        (
            pos.id(),
            HostTensor::i32(vec![batch, lq], case.pos.clone()).into(),
        ),
    ]);
    let plain = eval(&graph, &values, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();
    for (i, (g, w)) in got.iter().zip(plain.as_f32().unwrap()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "output element {i}: got {g}, plain {w}"
        );
    }
}

/// The typing rule refuses what the definition cannot read: a zero ratio or `k`, an `ids` operand whose width
/// is not `k`, a position operand that is not `[B, Lq]`, ids or positions that are not I32, and a key
/// whose extents differ from the query's.
#[test]
fn the_typing_rule_refuses_what_it_cannot_read() {
    let attrs = SelectedAttentionAttrs {
        n_rep: 2,
        scale: 0.5,
        softcap: None,
        ratio: 4,
        k: 3,
    };
    let ok = [
        TensorType::f32(vec![2, 4, 5, 8]),
        TensorType::f32(vec![2, 2, 16, 8]),
        TensorType::f32(vec![2, 2, 16, 6]),
        TensorType::new(vec![2, 1, 5, 3], DType::I32),
        TensorType::new(vec![2, 5], DType::I32),
    ];
    let layout = attrs.layout(&ok).unwrap();
    assert_eq!((layout.lk, layout.dv), (16, 6));
    let with = |attrs: &SelectedAttentionAttrs, at: usize, t: TensorType| {
        let mut ins = ok.to_vec();
        ins[at] = t;
        attrs.layout(&ins).unwrap_err()
    };
    let zero_ratio = SelectedAttentionAttrs {
        ratio: 0,
        ..attrs.clone()
    };
    assert!(matches!(
        zero_ratio.layout(&ok).unwrap_err(),
        ShapeError::SelectedAttention { ratio: 0, .. }
    ));
    assert!(matches!(
        with(&attrs, 3, TensorType::new(vec![2, 1, 5, 4], DType::I32)),
        ShapeError::AttentionOperand { operand: "ids", .. }
    ));
    assert!(matches!(
        with(&attrs, 4, TensorType::new(vec![2, 4], DType::I32)),
        ShapeError::AttentionOperand { operand: "pos", .. }
    ));
    assert!(matches!(
        with(&attrs, 3, TensorType::f32(vec![2, 1, 5, 3])),
        ShapeError::AttentionDtype { operand: "ids", .. }
    ));
    assert!(matches!(
        with(&attrs, 4, TensorType::f32(vec![2, 5])),
        ShapeError::AttentionDtype { operand: "pos", .. }
    ));
    assert!(matches!(
        with(&attrs, 1, TensorType::f32(vec![2, 2, 16, 7])),
        ShapeError::AttentionOperand { operand: "k", .. }
    ));
}

/// The oracle compares every query's ids with every block, so a selection past its bound is refused with a
/// typed error naming the op instead of allocating the indicator. Mutation: raise `MAX_SELECTED_BLOCK_PAIRS`
/// past 2^34; the row allocates (and runs out of memory) instead of refusing.
#[test]
fn the_oracle_refuses_a_selection_it_cannot_expand() {
    // 1 query, 8192 ids, 1 << 21 blocks: 2^34 comparisons.
    let case = Case {
        label: "bound".into(),
        batch: 1,
        hq: 1,
        n_rep: 1,
        lq: 1,
        lk: 4 << 21,
        d: 1,
        ratio: 4,
        k: 8192,
        pos: vec![0],
        ids: vec![-1; 8192],
    };
    let operands = [fill(1, 1), fill(case.lk, 2), fill(case.lk, 3)];
    let error = run(&case, &operands).unwrap_err();
    assert!(
        matches!(&error, EvalError::Unsupported { op, .. } if *op == "selected_attention"),
        "{error:?}"
    );
}

/// Check the dense definition against the independent f64 attention over the selected keys on the design's
/// shapes (24 query heads over 2 key/value heads of 256, blocks of four, 512 selected) at each of `depths`
/// stored positions, the selection being the plain top-k of seeded scores (exact set equality is the block
/// top-k's: `block_top_k::tests`). Only the f32 sums differ, in order.
fn assert_definition_at_depths(depths: &[usize]) {
    use poot_graph_ir::test_support::block_top_k::{grid_scores, reference_row};

    for &lk in depths {
        let (ratio, k) = (4, 512);
        let pos = vec![(lk - 3) as i32];
        let ids =
            reference_row(&grid_scores(lk / ratio, 1 << 20, 99), pos[0], ratio, k)[..k].to_vec();
        let case = Case {
            label: format!("{lk} positions"),
            batch: 1,
            hq: 24,
            n_rep: 12,
            lq: 1,
            lk,
            d: 256,
            ratio,
            k,
            pos,
            ids,
        };
        let operands = case.operands(17);
        let got = run(&case, &operands).unwrap();
        let want = case.reference(&operands[0], &operands[1], &operands[2]);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() <= 2e-5 * (1.0 + w.abs()),
                "{}: output element {i} (head {}, channel {}): got {g}, reference {w}",
                case.label,
                i / case.d,
                i % case.d
            );
        }
    }
}

/// SC-1143-12, the depth sweep at 8K stored positions, in the gate. The 32K and 128K rows are
/// `the_definition_attends_over_the_selected_keys_at_32k_and_128k_real_dims`.
#[test]
fn the_definition_attends_over_the_selected_keys_at_depth_real_dims() {
    assert_definition_at_depths(&[8192]);
}

/// The 32K and 128K rows of the sweep: the 128K one takes minutes in the oracle (the three-depth row took 1157 s in a debug lane), so it is run by name
/// (`--release`), not in the gate; the kernel's own rows reach 65536 blocks in `poot-kernelgen`.
#[test]
#[ignore = "minutes on the CPU oracle; run by name with --release: -E 'test(=the_definition_attends_over_the_selected_keys_at_32k_and_128k_real_dims)'"]
fn the_definition_attends_over_the_selected_keys_at_32k_and_128k_real_dims() {
    assert_definition_at_depths(&[32_768, 131_072]);
}

//! Routing fixtures: `ArgTopK` over a length that leaves a partial last workgroup, and MoE routing
//! on tied router scores.
//!
//! Each case carries the values the device must produce, derived here without the oracle (the
//! inverse of a rank permutation; a stable descending sort), so a row can assert the selection
//! itself and not only agreement with `eval`.

use poot_eval::test_support::StepFixture;
use poot_graph_ir::{Builder, Graph, TensorType, ops};
use poot_tensor::HostTensor;

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::Executor;
use poot_graph_plan::Target;

use crate::{Fixture, const_fixture};

/// One `ArgTopK` case: `rank [rows, e] -> ids [rows, k]`, `ids[l, r]` the index `i` with
/// `rank[l, i] == r`.
pub struct ArgTopKCase {
    /// The `ArgTopK` program and the rank input of each step.
    pub fixture: Fixture,
    /// The `[rows, k]` ids `fixture` must produce, row-major.
    pub expected: Vec<i32>,
}

/// `ArgTopK` runs one thread per output element, so `rows * k` lanes. The cases:
///
/// - `3 x 16, k = 4`: 12 lanes, one partial workgroup (the shape that crashed ROCm with an HSA
///   aperture violation when the idle lanes of a partial workgroup read uninitialized state).
/// - `17 x 16, k = 4`: 68 lanes, one full 64-lane workgroup and a 4-lane tail. The tail lanes are
///   the last row's four outputs, so that row's winner (`r = 0`, lane 64) is found by a tail lane;
///   the last row ranks the last expert first, so the winner is also the row's final element.
pub fn arg_top_k_tail_cases() -> Vec<ArgTopKCase> {
    vec![
        arg_top_k_case("arg_top_k_l3_e16_k4", 3, 16, 4),
        arg_top_k_case("arg_top_k_l17_e16_k4", 17, 16, 4),
    ]
}

fn arg_top_k_case(name: &'static str, rows: usize, e: usize, k: usize) -> ArgTopKCase {
    // Row `r` ranks expert `i` at `(7 i + 3 r) mod e` (a permutation: gcd(7, e) = 1), so rows are
    // different permutations. The last row is the reversal, ranking expert `e - 1` first.
    let rank_of = |row: usize, i: usize| {
        if row + 1 == rows {
            e - 1 - i
        } else {
            (i * 7 + row * 3) % e
        }
    };
    let rank: Vec<f32> = (0..rows)
        .flat_map(|row| (0..e).map(move |i| (row, i)))
        .map(|(row, i)| rank_of(row, i) as f32)
        .collect();
    let expected = (0..rows)
        .flat_map(|row| {
            (0..k).map(move |r| {
                (0..e)
                    .find(|&i| rank_of(row, i) == r)
                    .expect("a permutation ranks every expert") as i32
            })
        })
        .collect();

    let b = Builder::for_tests();
    let rank_in = b.constant("rank", TensorType::f32(vec![rows, e])).unwrap();
    let ids = b.arg_top_k(rank_in, k).unwrap();
    let graph = b.finish(ids).unwrap();
    ArgTopKCase {
        fixture: const_fixture(
            name,
            graph,
            vec![("rank", HostTensor::f32(vec![rows, e], rank))],
            2,
        ),
        expected,
    }
}

/// One tied-routing case: the router scores of `rows` tokens over `e` experts, routed top-`k`.
pub struct TieCase {
    /// The routing program over the tied router scores.
    pub fixture: Fixture,
    /// Number of experts routed per token.
    pub k: usize,
    /// The `[rows, k]` expert ids a stable descending order selects, row-major: tied experts in
    /// ascending index order.
    pub expected_ids: Vec<f32>,
}

/// The routed output is `[rows, k + 2 e]`: the selected ids, the keep mask and the gate, concatenated.
pub fn tie_cases() -> Vec<TieCase> {
    let cases: [(&'static str, &[[f32; 5]], usize); 7] = [
        ("moe_tie_all_equal", &[[1.0; 5]], 3),
        ("moe_tie_at_the_cut", &[[3.0, 5.0, 5.0, 5.0, 1.0]], 2),
        ("moe_tie_maxima_split", &[[9.0, 9.0, 2.0, 9.0, 0.0]], 2),
        ("moe_tie_signed_zero", &[[-0.0, 0.0, -0.0, 0.0, -1.0]], 3),
        ("moe_tie_k_one", &[[1.0, 9.0, 2.0, 9.0, 0.0]], 1),
        (
            "moe_tie_k_equals_experts",
            &[[f32::MAX, f32::MIN, 3.0, -0.0, f32::MIN]],
            5,
        ),
        (
            "moe_tie_rows",
            &[
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [0.5, 4.0, 4.0, 0.5, 4.0],
                [-1.0, -1.0, 3.0, 3.0, -1.0],
                [7.0, 1.0, 7.0, 1.0, 7.0],
            ],
            3,
        ),
    ];
    cases
        .into_iter()
        .map(|(name, rows, k)| tie_case(name, rows, k))
        .collect()
}

fn tie_case(name: &'static str, rows: &[[f32; 5]], k: usize) -> TieCase {
    let (l, e) = (rows.len(), 5);
    let scores: Vec<f32> = rows.iter().flatten().copied().collect();
    let expected_ids = rows
        .iter()
        .flat_map(|row| {
            let mut order: Vec<usize> = (0..e).collect();
            // Descending score; `sort_by` is stable, so tied experts keep ascending index order.
            order.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).expect("finite scores"));
            order.into_iter().take(k).map(|i| i as f32)
        })
        .collect();

    let b = Builder::for_tests();
    let scores_in = b.constant("scores", TensorType::f32(vec![l, e])).unwrap();
    let rank = ops::stable_descending_rank(&b, scores_in).unwrap();
    let ids = b
        .cast(b.arg_top_k(rank, k).unwrap(), poot_tensor::DType::F32)
        .unwrap();
    let mask = ops::top_k_keep_mask(&b, rank, k).unwrap();
    let gate = ops::top_k_gate(&b, scores_in, k).unwrap();
    let out = b.concat(1, &[ids, mask, gate]).unwrap();
    let graph = b.finish(out).unwrap();
    TieCase {
        fixture: const_fixture(
            name,
            graph,
            vec![("scores", HostTensor::f32(vec![l, e], scores))],
            2,
        ),
        k,
        expected_ids,
    }
}

/// The expert ids of a routing: an `I32` step input, since the weight store holds floats only. `experts` declares
/// the ids in `0..experts`, as a router's `ArgTopK` proves them; `compile` refuses an indexed contraction whose ids
/// it cannot prove in range, so an invalid id never reaches a compiled contraction, only the grouping on its own.
fn ids_slot(b: &Builder, rows: usize, experts: Option<usize>) -> poot_graph_ir::Traced {
    let aval = TensorType::new(vec![rows], poot_tensor::DType::I32);
    match experts {
        Some(experts) => b.slot_named_ranged(
            poot_graph_ir::Slot::Activation,
            "ids",
            aval,
            poot_graph_ir::IndexRange::below(experts)
                .expect("an expert count fits the index range"),
        ),
        None => b.slot_named(poot_graph_ir::Slot::Activation, "ids", aval),
    }
    .unwrap()
}

/// A fixture with named float constants and the expert `ids` bound at each of two steps (the entry records on
/// the first and replays on the second).
fn ids_fixture(
    name: &'static str,
    graph: Graph,
    ids_in: poot_graph_ir::Traced,
    consts: Vec<(&str, HostTensor)>,
    ids: Vec<i32>,
) -> Fixture {
    let key = graph
        .meta(ids_in.id())
        .slot_key()
        .expect("a slot input has a key")
        .clone();
    let step = || {
        vec![StepFixture {
            key: key.clone(),
            tensor: HostTensor::i32(vec![ids.len()], ids.clone()),
        }]
    };
    Fixture {
        name,
        graph,
        store: crate::store_from_tensors(consts),
        steps: vec![step(), step()],
    }
}

/// One expert-grouping case: `ids [rows]` through `ExpertGroup`, whose `[2, rows]` output is the sorted order and
/// its inverse.
pub struct ExpertGroupCase {
    /// The grouping program over the ids.
    pub fixture: Fixture,
    /// The `[2, rows]` output the device must produce, row-major: the stable order by expert, then each row's
    /// sorted position. Derived here by a stable sort, not by the oracle.
    pub expected: Vec<i32>,
}

/// The routings that break a naive grouping: every row on one expert, experts without rows, tied rows
/// (a repeated expert keeps its row order), ids outside the table (sorted after every expert, never dropped), a
/// length that leaves a partial last workgroup of rows, and more experts than rows.
pub fn expert_group_cases() -> Vec<ExpertGroupCase> {
    let cycle = |rows: usize, experts: i32| -> Vec<i32> {
        (0..rows as i32).map(|r| (r * 5 + 3) % experts).collect()
    };
    vec![
        expert_group_case("expert_group_one_expert", vec![2; 37], 4),
        expert_group_case(
            "expert_group_empty_experts",
            vec![3, 3, 0, 3, 0, 0, 3, 5, 5],
            6,
        ),
        expert_group_case("expert_group_ties", vec![1, 0, 1, 0, 1, 0, 1, 0], 2),
        expert_group_case("expert_group_mixed", cycle(301, 7), 7),
        expert_group_case("expert_group_more_experts_than_rows", cycle(5, 40), 40),
        expert_group_case(
            "expert_group_invalid_ids",
            vec![2, -1, 0, 4, i32::MIN, 1, i32::MAX, 0, 1_000_000_000, 3, -7],
            4,
        ),
    ]
}

fn expert_group_case(name: &'static str, ids: Vec<i32>, experts: usize) -> ExpertGroupCase {
    let bucket = |id: i32| {
        usize::try_from(id)
            .ok()
            .filter(|&e| e < experts)
            .unwrap_or(experts)
    };
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&row| bucket(ids[row]));
    let mut dest = vec![0i32; ids.len()];
    for (position, &row) in order.iter().enumerate() {
        dest[row] = position as i32;
    }
    let expected = order
        .into_iter()
        .map(|row| row as i32)
        .chain(dest)
        .collect();

    let b = Builder::for_tests();
    let ids_in = ids_slot(&b, ids.len(), None);
    let out = b.expert_group(ids_in, experts).unwrap();
    let graph = b.finish(out).unwrap();
    ExpertGroupCase {
        fixture: ids_fixture(name, graph, ids_in, Vec::new(), ids),
        expected,
    }
}

/// One routed-contraction case: the indexed contraction of `rows` activation rows through their expert ids, run
/// twice in one program, directly and through the expert grouping (rows gathered into expert order, contracted,
/// gathered back). The `[rows, 2 * N]` output is the direct result, then the grouped one.
pub struct GroupedContractionCase {
    /// The program holding both contractions.
    pub fixture: Fixture,
    /// The output columns of one contraction (`N`).
    pub n: usize,
}

/// More rows than a Gemv serves (so the register-tiled expert walk runs on every device), over routings where a tile
/// holds one expert, many, or none of an expert.
pub fn grouped_contraction_cases() -> Vec<GroupedContractionCase> {
    const EXPERTS: usize = 8;
    let rows = 100usize;
    let mixed: Vec<i32> = (0..rows as i32).map(|r| (r * 5 + 3) % 8).collect();
    vec![
        grouped_contraction_case("grouped_one_expert", vec![5; rows], EXPERTS),
        grouped_contraction_case(
            "grouped_empty_experts",
            (0..rows as i32).map(|r| [1, 6][(r % 2) as usize]).collect(),
            EXPERTS,
        ),
        grouped_contraction_case("grouped_mixed", mixed, EXPERTS),
    ]
}

fn grouped_contraction_case(
    name: &'static str,
    ids: Vec<i32>,
    experts: usize,
) -> GroupedContractionCase {
    let (rows, k, n) = (ids.len(), 12usize, 20usize);
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, k])).unwrap();
    let w = b
        .constant("w", TensorType::f32(vec![experts, k, n]))
        .unwrap();
    let ids_in = ids_slot(&b, rows, Some(experts));
    let direct = b.indexed_matmul(x, w, ids_in).unwrap();
    let group = b.expert_group(ids_in, experts).unwrap();
    let row_of = |i| {
        b.reshape(b.slice(group, 0, i, i + 1).unwrap(), vec![rows])
            .unwrap()
    };
    let (order, dest) = (row_of(0), row_of(1));
    let sorted = b
        .indexed_matmul(
            b.gather(x, 0, order).unwrap(),
            w,
            b.gather(ids_in, 0, order).unwrap(),
        )
        .unwrap();
    let grouped = b.gather(sorted, 0, dest).unwrap();
    let out = b.concat(1, &[direct, grouped]).unwrap();
    let graph = b.finish(out).unwrap();
    let x_data: Vec<f32> = (0..rows * k).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let w_data: Vec<f32> = (0..experts * k * n)
        .map(|i| ((i * 13) % 9) as f32 * 0.25 - 1.0)
        .collect();
    GroupedContractionCase {
        fixture: ids_fixture(
            name,
            graph,
            ids_in,
            vec![
                ("x", HostTensor::f32(vec![rows, k], x_data)),
                ("w", HostTensor::f32(vec![experts, k, n], w_data)),
            ],
            ids,
        ),
        n,
    }
}

/// Assert the two halves of a grouped-contraction output (`[rows, 2 * n]`) are bit-identical row by row: grouping
/// changes no row's dot product.
pub fn assert_grouping_changes_no_row(case: &GroupedContractionCase, got: &HostTensor) {
    let name = case.fixture.name;
    let width = 2 * case.n;
    let values = got.as_f32().expect("the contraction output is F32");
    for (row, pair) in values.chunks(width).enumerate() {
        let (direct, grouped) = pair.split_at(case.n);
        for (col, (d, g)) in direct.iter().zip(grouped).enumerate() {
            assert_eq!(
                d.to_bits(),
                g.to_bits(),
                "{name} row {row} col {col}: direct {d} vs grouped {g}"
            );
        }
    }
}

/// The expert grouping on a device, over [`expert_group_cases`] and [`grouped_contraction_cases`]: the grouping
/// equals a stable sort derived here and the oracle at every replayed step; a contraction through the grouping is
/// bit-identical per row to the direct one and equals the oracle. (An id outside the table never reaches a compiled
/// contraction: `compile` refuses ids it cannot prove in range, so the invalid-id contraction is covered on the
/// kernel interpreter, `poot-kernelgen/tests/expert_group_interp.rs`.)
pub fn assert_expert_grouping_on_device(exec: &mut dyn Executor, target: Target) {
    for case in expert_group_cases() {
        let name = case.fixture.name;
        let outputs =
            crate::run_outputs(exec, target, &case.fixture).unwrap_or_else(|e| panic!("{e}"));
        for (step, got) in outputs.iter().enumerate() {
            assert_eq!(
                got.as_i32().expect("the grouping is I32"),
                case.expected.as_slice(),
                "{name} step {step}: the order and its inverse differ from a stable sort by expert"
            );
        }
        let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
        crate::run_parity(exec, target, &case.fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    for case in grouped_contraction_cases() {
        let name = case.fixture.name;
        let outputs =
            crate::run_outputs(exec, target, &case.fixture).unwrap_or_else(|e| panic!("{e}"));
        for got in &outputs {
            assert_grouping_changes_no_row(&case, got);
        }
        let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
        crate::run_parity(exec, target, &case.fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

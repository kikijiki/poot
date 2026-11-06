//! Routing fixtures: `ArgTopK` over a length that leaves a partial last workgroup, and MoE routing
//! on tied router scores.
//!
//! Each case carries the values the device must produce, derived here without the oracle (the
//! inverse of a rank permutation; a stable descending sort), so a row can assert the selection
//! itself and not only agreement with `eval`.

use poot_eval::test_support::StepFixture;
use poot_graph_ir::{Builder, ExpertGrouping, Graph, TensorType, ops};
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

/// One expert-grouping case: `ids [rows]` through `ExpertGroup` in tiles of `block`, whose output is the
/// [`ExpertGrouping`] layout (POOT-1451).
pub struct ExpertGroupCase {
    /// The grouping program over the ids.
    pub fixture: Fixture,
    /// The positions per tile.
    pub block: usize,
    /// The output the device must produce: the padded order, then each tile's bucket. Derived here bucket by
    /// bucket, not by the oracle.
    pub expected: Vec<i32>,
}

/// The routings that break a naive grouping, each at every [`GROUPING_BLOCKS`] size: every row on one expert,
/// experts without rows, tied rows (a repeated expert keeps its row order), ids outside the table (in the last
/// bucket, never dropped), a length that leaves partial tiles, more experts than rows, and more buckets than the
/// grouping kernel's lanes.
pub fn expert_group_cases() -> Vec<ExpertGroupCase> {
    let cycle = |rows: usize, experts: i32| -> Vec<i32> {
        (0..rows as i32).map(|r| (r * 5 + 3) % experts).collect()
    };
    let cases = [
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
        expert_group_case("expert_group_more_buckets_than_lanes", cycle(700, 300), 300),
    ];
    cases.into_iter().flatten().collect()
}

/// The tile sizes every grouping case runs at: one row a tile, a few, and a fragment tile's 16.
const GROUPING_BLOCKS: [usize; 3] = [1, 4, 16];

/// The block-aligned grouping of `ids` over `experts` in tiles of `block` ([`ExpertGrouping`]), built bucket by
/// bucket: each bucket's rows in row order, padded with the sentinel `rows` to whole tiles, then each tile's bucket.
fn grouping_layout(ids: &[i32], experts: usize, block: usize) -> Vec<i32> {
    let layout = ExpertGrouping {
        rows: ids.len(),
        experts,
        block,
    };
    let bucket = |id: i32| {
        usize::try_from(id)
            .ok()
            .filter(|&e| e < experts)
            .unwrap_or(experts)
    };
    let (mut order, mut tiles) = (Vec::new(), Vec::new());
    for b in 0..=experts {
        let rows: Vec<i32> = (0..ids.len())
            .filter(|&r| bucket(ids[r]) == b)
            .map(|r| r as i32)
            .collect();
        for chunk in rows.chunks(block) {
            order.extend_from_slice(chunk);
            order.resize(order.len() + block - chunk.len(), ids.len() as i32);
            tiles.push(b as i32);
        }
    }
    order.resize(layout.positions(), ids.len() as i32);
    tiles.resize(layout.tiles(), experts as i32);
    order.extend(tiles);
    order
}

fn expert_group_case(name: &'static str, ids: Vec<i32>, experts: usize) -> Vec<ExpertGroupCase> {
    GROUPING_BLOCKS
        .into_iter()
        .map(|block| {
            let b = Builder::for_tests();
            let ids_in = ids_slot(&b, ids.len(), None);
            let out = b.expert_group(ids_in, experts, block).unwrap();
            let graph = b.finish(out).unwrap();
            ExpertGroupCase {
                fixture: ids_fixture(name, graph, ids_in, Vec::new(), ids.clone()),
                block,
                expected: grouping_layout(&ids, experts, block),
            }
        })
        .collect()
}

/// One routed-contraction case: the indexed contraction of `rows` activation rows through their expert ids, pinned to
/// the grouping (`Policy::grouped`), so `compile` groups them by expert (`group_expert_rows`) and the device runs the
/// tiled contraction over the grouping. The `[rows, N]` output is in row order.
pub struct GroupedContractionCase {
    /// The program holding the contraction.
    pub fixture: Fixture,
    /// The output columns (`N`).
    pub n: usize,
    /// Each row's own product `x[r] . W[ids[r]]`, `[rows, N]` row-major, summed here (the operands are small
    /// multiples of a quarter, so every summation order gives these bits).
    pub expected: Vec<f32>,
}

/// Rows pinned to the grouping (so every device runs the tiled contraction over the rows' grouping), over routings
/// where a tile holds one expert's rows, one expert takes every row, or experts take none.
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

/// A routed prefill projection under the narrow policy (POOT-1451): 512 rows over 8 experts of a checkpoint-order
/// BF16 `[E, N, K]` stack read through its transposed view. `compile` groups the rows once, the tiling's tile rows
/// are the grouping's block (every tile one expert's rows), and on a matrix unit that takes BF16 the tile is the
/// fragment one; the output is each row's own product, bit for bit. The operands are integers in `[-3, 3]`, exact in BF16 and F16, and every partial sum is an
/// exact integer, so the narrowing changes nothing and every accumulation order gives the expected bits.
pub fn assert_grouped_fragment_contraction_on_device(exec: &mut dyn Executor, target: Target) {
    const EXPERTS: usize = 8;
    let (rows, k, n) = (512usize, 64usize, 48usize);
    let ids: Vec<i32> = (0..rows as i32)
        .map(|r| (r * 5 + 3) % 7 + (r % 3 == 0) as i32)
        .collect();
    let int = |seed: usize, count: usize| -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 7 + seed * 3) % 7) as f32 - 3.0)
            .collect()
    };
    let (x_data, w_data) = (int(1, rows * k), int(2, EXPERTS * n * k));
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![rows, k])).unwrap();
    let stack = b
        .constant(
            "w",
            TensorType::new(vec![EXPERTS, n, k], poot_tensor::DType::BF16),
        )
        .unwrap();
    let ids_in = ids_slot(&b, rows, Some(EXPERTS));
    let out = b
        .indexed_matmul(x, b.transpose(stack, vec![0, 2, 1]).unwrap(), ids_in)
        .unwrap();
    let graph = b.finish(out).unwrap();
    let words: Vec<u16> = w_data.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
    let fixture = ids_fixture(
        "grouped_fragment_bf16",
        graph,
        ids_in,
        vec![
            ("x", HostTensor::f32(vec![rows, k], x_data.clone())),
            ("w", HostTensor::bf16(vec![EXPERTS, n, k], words)),
        ],
        ids.clone(),
    );
    // A matrix unit that takes the stack's BF16 (ROCm WMMA, NVPTX `wmma`) runs the fragment tile; an F16-only one
    // (a Vulkan cooperative matrix) leaves the tiling to the cost model.
    let bf16_units = match target {
        Target::Gpu { caps, .. } => caps
            .matrix_units
            .known()
            .is_some_and(|units| units.config(poot_graph_ir::FragmentDtype::Bf16).is_some()),
        Target::Npu(_) => false,
    };
    // The rows are pinned to the grouping: at these dimensions the cost model plans the Gemv. On a BF16 unit the
    // narrowing is declared too: this row is about the grouped fragment path, and whether a unit's issue rate pays at
    // 48 columns and 64 deep is the cost model's own call (an sm_89 part, at the GeForce floor rate, keeps the
    // register tile), not what this row asserts.
    let policy = crate::Policy {
        declared: bf16_units,
        ..crate::Policy::grouped(poot_graph_ir::MatmulPrecision::Narrow)
    };
    let program = crate::staged_under(exec, &fixture, target, policy);
    let (_, _, stage) = program.stages().next().expect("one stage");
    let blocks: Vec<usize> = stage
        .planned()
        .filter_map(|(eqn, _)| match eqn.op {
            poot_graph_ir::OpKind::ExpertGroup { block, .. } => Some(block),
            _ => None,
        })
        .collect();
    let schedules: Vec<poot_kernelgen::Schedule> = stage
        .planned()
        .filter_map(|(eqn, _)| match stage.kernel_choice(eqn) {
            poot_graph_plan::KernelChoice::Generated(
                poot_kernelgen::KernelRequest::Contraction(
                    poot_kernelgen::ContractionSpec::Indexed(request),
                ),
            ) => Some(request.schedule),
            _ => None,
        })
        .collect();
    assert_eq!(schedules.len(), 1, "one indexed contraction: {schedules:?}");
    assert_eq!(
        blocks,
        [schedules[0].output_tile().expect("a tiled schedule").0],
        "one grouping, its block the tiling's tile rows: {schedules:?}"
    );
    if bf16_units {
        assert!(
            matches!(schedules[0], poot_kernelgen::Schedule::FragmentTiled { .. }),
            "the grouped rows run the fragment tile on a BF16 matrix unit: {schedules:?}"
        );
    }
    let expected: Vec<f32> = (0..rows)
        .flat_map(|row| {
            let (x, w, e) = (&x_data, &w_data, ids[row] as usize);
            (0..n).map(move |col| {
                (0..k)
                    .map(|kk| x[row * k + kk] * w[(e * n + col) * k + kk])
                    .sum::<f32>()
            })
        })
        .collect();
    let outputs =
        crate::run_outputs_under(exec, target, &fixture, policy).unwrap_or_else(|e| panic!("{e}"));
    for (step, got) in outputs.iter().enumerate() {
        let got = got.as_f32().expect("the contraction output is F32");
        if let Some(at) = (0..got.len()).find(|&i| got[i].to_bits() != expected[i].to_bits()) {
            panic!(
                "step {step}: row {} col {} (expert {}) is {} on the device, {} its expert's product",
                at / n,
                at % n,
                ids[at / n],
                got[at],
                expected[at]
            );
        }
    }
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
    let out = b.indexed_matmul(x, w, ids_in).unwrap();
    let graph = b.finish(out).unwrap();
    let x_data: Vec<f32> = (0..rows * k).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let w_data: Vec<f32> = (0..experts * k * n)
        .map(|i| ((i * 13) % 9) as f32 * 0.25 - 1.0)
        .collect();
    let expected = (0..rows)
        .flat_map(|row| {
            let (x, w, expert) = (&x_data, &w_data, ids[row] as usize);
            (0..n).map(move |col| {
                (0..k)
                    .map(|kk| x[row * k + kk] * w[(expert * k + kk) * n + col])
                    .sum::<f32>()
            })
        })
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
        expected,
    }
}

/// Assert a grouped-contraction output (`[rows, N]`) is each row's own product, bit for bit: the grouping drops,
/// moves and reroutes no row.
pub fn assert_every_row_is_its_experts_product(case: &GroupedContractionCase, got: &HostTensor) {
    let name = case.fixture.name;
    let values = got.as_f32().expect("the contraction output is F32");
    assert_eq!(values.len(), case.expected.len(), "{name}");
    for (at, (g, want)) in values.iter().zip(&case.expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            want.to_bits(),
            "{name} row {} col {}: device {g} vs its expert's product {want}",
            at / case.n,
            at % case.n
        );
    }
}

/// The expert grouping on a device, over [`expert_group_cases`] and [`grouped_contraction_cases`]: the grouping
/// equals the layout derived here and the oracle at every replayed step; a contraction `compile` groups writes every
/// row's own product in row order and equals the oracle. (An id outside the table never reaches a compiled
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
                "{name} block {} step {step}: the grouping differs from the layout",
                case.block
            );
        }
        let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
        crate::run_parity(exec, target, &case.fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
    // The contractions are pinned to the grouping: at these dimensions the cost model plans the Gemv.
    let grouped = crate::Policy::grouped(poot_graph_ir::MatmulPrecision::Exact);
    for case in grouped_contraction_cases() {
        let name = case.fixture.name;
        let outputs = crate::run_outputs_under(exec, target, &case.fixture, grouped)
            .unwrap_or_else(|e| panic!("{e}"));
        for got in &outputs {
            assert_every_row_is_its_experts_product(&case, got);
        }
        let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
        crate::run_parity_under(exec, target, &case.fixture, grouped, &mut oracle)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

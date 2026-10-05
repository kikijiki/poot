//! Routing fixtures: `ArgTopK` over a length that leaves a partial last workgroup, and MoE routing
//! on tied router scores.
//!
//! Each case carries the values the device must produce, derived here without the oracle (the
//! inverse of a rank permutation; a stable descending sort), so a row can assert the selection
//! itself and not only agreement with `eval`.

use poot_graph_ir::{Builder, TensorType, ops};
use poot_tensor::HostTensor;

use crate::{Fixture, const_fixture};

/// One `ArgTopK` case: `rank [rows, e] -> ids [rows, k]`, `ids[l, r]` the index `i` with
/// `rank[l, i] == r`.
pub struct ArgTopKCase {
    pub fixture: Fixture,
    /// The `[rows, k]` ids `fixture` must produce, row-major.
    pub expected: Vec<f32>,
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
                    .expect("a permutation ranks every expert") as f32
            })
        })
        .collect();

    let b = Builder::new();
    let rank_in = b.constant("rank", TensorType::f32(vec![rows, e]));
    let ids = b.arg_top_k(rank_in, k);
    let graph = b.finish(ids);
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
    pub fixture: Fixture,
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

    let b = Builder::new();
    let scores_in = b.constant("scores", TensorType::f32(vec![l, e]));
    let rank = ops::stable_descending_rank(&b, scores_in);
    let ids = b.arg_top_k(rank, k);
    let mask = ops::top_k_keep_mask(&b, rank, k);
    let gate = ops::top_k_gate(&b, scores_in, k);
    let out = b.concat(1, &[ids, mask, gate]);
    let graph = b.finish(out);
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

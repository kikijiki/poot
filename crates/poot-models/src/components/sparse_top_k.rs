//! Top-k selection for attention-side selectors (DeepSeek sparse attention, the compressed-attention
//! indexers). The router keeps exactly `k` entries ([`poot_graph_ir::ops::stable_descending_rank`]
//! breaks ties by index); these selectors keep every entry with fewer than `k` strictly greater
//! ones, so exact ties at the cut are all kept and more than `k` may survive. That is the
//! established selector behavior of these attention families, kept distinct from the router's, but
//! it is computed from the router's rank so the same `StableRank` composite serves both.

use crate::model::TraceError;
use crate::model::last_axis;
use poot_graph_ir::{BinOp, Builder, RedOp, Requirement, Scalar, Traced, ops};

/// `1.0` where a score is among the `k` best of its row (trailing axis of `x`, shape `[.., N]`),
/// `0.0` otherwise, with ties kept together: an entry is kept when fewer than `k` entries are
/// strictly greater, so a tie at the cut keeps every tied entry. `k >= N` keeps the whole row.
///
/// The selection is a threshold, not a pairwise rank: the entry of stable rank `k - 1` is the
/// `k`-th best score, and an entry has fewer than `k` strictly greater entries exactly when it is
/// at least that score. The stable rank is [`ops::stable_descending_rank`], which `compile` forms
/// into one `StableRank` composite, so no `[.., N, N]` pairwise value is materialized. Scores are
/// canonicalized as the router's are (NaN and `-inf` become `f32::MIN`, `+inf` `f32::MAX`), so
/// every row has a defined cut.
pub(crate) fn keep_top_k_mask(b: &Builder, x: Traced, k: usize) -> Result<Traced, TraceError> {
    let shape = b.aval(x)?.shape;
    let n = last_axis("keep_top_k_mask", &shape)?;
    if k == 0 {
        return Err(poot_graph_ir::BuilderError::Requirement {
            operation: "keep_top_k_mask",
            requirement: Requirement::TopKRange { k, experts: n },
        }
        .into());
    }
    let scores = ops::canonical_router_scores(b, x)?;
    if k >= n {
        // Every canonical score is at least `f32::MIN`.
        return Ok(b.binary_scalar(BinOp::Ge, scores, Scalar::F32(f32::MIN))?);
    }
    let last = shape.len() - 1;
    let rank = ops::stable_descending_rank(b, x)?;
    // One-hot at stable rank `k - 1`: `rank >= k - 1` and not `rank >= k`.
    let from = b.binary_scalar(BinOp::Ge, rank, Scalar::F32((k - 1) as f32))?;
    let past = b.binary_scalar(BinOp::Ge, rank, Scalar::F32(k as f32))?;
    let cut = b.binary(BinOp::Sub, from, past)?;
    let kth = b.reduce(RedOp::Sum, b.binary(BinOp::Mul, scores, cut)?, last, true)?;
    let kth = b.broadcast(kth, shape)?;
    Ok(b.binary(BinOp::Ge, scores, kth)?)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::TensorType;

    use super::*;
    use crate::components::standard::oracle::Fx;

    const ROWS: usize = 3;
    const N: usize = 7;

    /// Rows with distinct values, exact ties and a negative run.
    fn input() -> Vec<f32> {
        vec![
            0.5, -1.0, 2.0, 0.25, 3.0, -0.5, 1.0, //
            1.0, 1.0, 0.0, 1.0, -2.0, 0.0, 2.0, //
            -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0,
        ]
    }

    fn run(x: &[f32], pick: impl Fn(&Builder, Traced) -> Result<Traced, TraceError>) -> Vec<f32> {
        let b = Builder::for_tests();
        let xs = b.constant("x", TensorType::f32(vec![ROWS, N])).unwrap();
        let out = pick(&b, xs).unwrap();
        Fx::new(1)
            .eval(&b.finish(out).unwrap(), &[("x", x)], &[], &[])
            .0
    }

    /// The keep mask written from its definition: an entry is kept when fewer than `k` entries of
    /// its row are strictly greater.
    fn reference_mask(x: &[f32], k: usize) -> Vec<f32> {
        x.chunks(N)
            .flat_map(|row| {
                (0..N).map(move |i| {
                    let greater = row.iter().filter(|&&v| v > row[i]).count();
                    if greater < k { 1.0 } else { 0.0 }
                })
            })
            .collect()
    }

    /// SC-002: the mask keeps exactly the entries with fewer than `k` strictly greater ones, for every
    /// `k` from one past the row (keeps all) down to one: `k` of a distinct row, every tied entry of
    /// an all-equal row, a whole tie group at the cut. Mutation: an off-by-one in the cut
    /// (`k` for `k - 1` in the one-hot); the kept counts change and this row goes red.
    #[test]
    fn the_mask_keeps_entries_with_fewer_than_k_strictly_greater() {
        let x = input();
        for k in 1..=N + 1 {
            let got = run(&x, |b, xs| keep_top_k_mask(b, xs, k));
            assert_eq!(got, reference_mask(&x, k), "k = {k}");
        }
        // The distinct row keeps exactly k; the all-equal row keeps all of them at any k.
        let two = run(&x, |b, xs| keep_top_k_mask(b, xs, 2));
        assert_eq!(two[..N].iter().sum::<f32>(), 2.0);
        assert_eq!(two[2 * N..].iter().sum::<f32>(), N as f32);
        // Row two ties three entries at 1.0 below the maximum: k = 2 keeps the 2.0 and all three.
        assert_eq!(two[N..2 * N].iter().sum::<f32>(), 4.0);
    }

    /// A non-finite score gets the router's canonical value instead of an undefined cut: NaN ranks
    /// last, `+inf` first, and `k` still selects `k` entries of a distinct row.
    #[test]
    fn non_finite_scores_have_a_defined_cut() {
        let mut x = vec![0.0; ROWS * N];
        x[..N].copy_from_slice(&[1.0, f32::NAN, 3.0, f32::INFINITY, -1.0, 2.0, 0.5]);
        let got = run(&x, |b, xs| keep_top_k_mask(b, xs, 3));
        assert_eq!(got[..N], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
    }

    /// `k = 0` selects nothing, which is no selection: it is refused, not traced.
    #[test]
    fn a_zero_k_is_refused() {
        let b = Builder::for_tests();
        let xs = b.constant("x", TensorType::f32(vec![ROWS, N])).unwrap();
        assert!(keep_top_k_mask(&b, xs, 0).is_err());
    }

    /// The production path: `compile` turns the mask's rank into one `StableRank` on every backend
    /// and the `[.., N, N]` pairwise workspace is gone. A `k` at or past the row never ranks. Mutation:
    /// rank with a pairwise decomposition of the mask's own (or skip the rank arm of `form_selection`);
    /// no `StableRank` forms, or a pairwise value stays, and this row goes red.
    #[test]
    fn compile_ranks_the_selector_through_one_stable_rank_without_a_pairwise_value() {
        use poot_graph_ir::OpKind;
        use poot_graph_plan::{CompileLimits, CompileOptions, Target, compile};
        use poot_target::{AmdArch, Backend, DeviceCaps};

        let options = CompileOptions::new(CompileLimits::STANDARD);
        for (backend, caps) in [
            (
                Backend::AmdGcn(AmdArch::gfx1151()),
                DeviceCaps::rocm_default(),
            ),
            (Backend::Nvptx, DeviceCaps::ptx_default()),
            (
                Backend::SpirvVulkan,
                poot_test_util::device_caps::wgpu_rdna3_igpu(),
            ),
        ] {
            for (k, ranks) in [(2, 1), (N, 0)] {
                let b = Builder::for_tests();
                let xs = b.constant("x", TensorType::f32(vec![ROWS, N])).unwrap();
                let out = keep_top_k_mask(&b, xs, k).unwrap();
                let graph = b.finish(out).unwrap();
                let program = compile(&graph, &Target::Gpu { backend, caps }, &options).unwrap();
                let g = program.graph();
                let formed = g
                    .eqns()
                    .iter()
                    .filter(|eqn| matches!(eqn.op, OpKind::StableRank(_)))
                    .count();
                assert_eq!(formed, ranks, "{backend:?} k={k}: StableRank count");
                for eqn in g.eqns() {
                    assert_ne!(
                        g.aval(eqn.out).shape,
                        [ROWS, N, N],
                        "{backend:?} k={k}: {} is a pairwise value",
                        eqn.op.name()
                    );
                }
            }
        }
    }
}

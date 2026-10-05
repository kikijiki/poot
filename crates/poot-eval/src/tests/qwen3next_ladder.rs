//! Card 158 ArgTopK tie regression (the collapse-repro ladder and the qwen2 mixed K-quant reference this
//! file used to carry were deleted with the quant-producer tracers, card 545a).

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, RedOp};
use poot_graph_ir::types::TensorType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

/// Card 158 regression: on rank ties, prefill MoE routing must emit only in-range expert ids.
///
/// The GPU kernel `arg_top_k_dt` (fed by `moe_grouped_prep`'s `ArgTopK` in the L>1 prefill path) inverted
/// rank->expert-id with a sum one-hot (`acc = sum_ee ee*eq(rank[ee], r)`), correct only when rank is a
/// permutation. When two experts share a rank r<k (256-expert router logits hit f32 ties), the sum added both
/// ids, giving an id that can exceed E and index the expert weight buffer of the router's consumer (at the
/// time, the now-deleted `indexed_matmul_dequant`) out of bounds (prefill device loss). Decode (`moe_sparse`)
/// uses `Scatter` (`out[rank[i]]=i`, last-wins), always
/// in range. `arg_top_k_dt` now uses the same last-wins select as the eval oracle `arg_top_k`. This test
/// checks that the old sum goes OOB on ties and that the fixed select equals the eval oracle and stays in
/// [0,E).
#[test]
fn card158_arg_top_k_ties_stay_in_range() {
    let (e, k) = (256usize, 8usize);

    // Compute rank[E] from logits with the pairwise-Ge composition moe_grouped_prep uses.
    let rank_of = |logits: &[f32]| -> Vec<f32> {
        let br = Builder::new();
        let lg = br.constant("logits", TensorType::f32(vec![1, e]));
        let p = br.broadcast(br.reshape(lg, vec![1, e, 1]), vec![1, e, e]);
        let q = br.broadcast(br.reshape(lg, vec![1, 1, e]), vec![1, e, e]);
        let ge = br.binary(BinOp::Ge, p, q);
        let gt = br.binary_scalar(BinOp::Mul, ge, poot_graph_ir::types::Scalar::F32(-1.0));
        let gt = br.binary_scalar(BinOp::Add, gt, poot_graph_ir::types::Scalar::F32(1.0));
        let rank = br.reduce(RedOp::Sum, gt, 2, false);
        let lgi = lg.id;
        let g = br.finish(rank);
        let mut inp = HashMap::new();
        inp.insert(
            lgi,
            Value::from(HostTensor::f32(vec![1, e], logits.to_vec())),
        );
        eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_f32()
            .unwrap()
            .to_vec()
    };
    // Old GPU inversion: acc = sum_ee ee*eq(rank[ee], r). Kept only to prove it went OOB.
    let sum_inversion = |rank: &[f32]| -> Vec<usize> {
        (0..k)
            .map(|r| {
                let mut acc = 0.0f32;
                for (ee, &rv) in rank.iter().enumerate() {
                    if rv == r as f32 {
                        acc += ee as f32;
                    }
                }
                acc as usize
            })
            .collect()
    };
    // Fixed GPU inversion, replicating arg_top_k_dt's last-wins select (`acc = eqf ? eif : acc`, the largest
    // ee with rank==r): always a single in-range id.
    let select_inversion = |rank: &[f32]| -> Vec<usize> {
        (0..k)
            .map(|r| {
                let mut acc = 0.0f32;
                for (ee, &rv) in rank.iter().enumerate() {
                    if rv == r as f32 {
                        acc = ee as f32; // overwrite -> last (largest ee) wins
                    }
                }
                acc as usize
            })
            .collect()
    };
    // Eval oracle arg_top_k (scatter out[rank[i]]=i, last-wins); the fixed kernel must equal this.
    let oracle = |rank_data: &[f32]| -> Vec<usize> {
        let ba = Builder::new();
        let rk = ba.constant("rank", TensorType::f32(vec![1, e]));
        let out = ba.arg_top_k(rk, k);
        let rki = rk.id;
        let g = ba.finish(out);
        let mut inp = HashMap::new();
        inp.insert(
            rki,
            Value::from(HostTensor::f32(vec![1, e], rank_data.to_vec())),
        );
        eval(&g, &inp, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
            .as_f32()
            .unwrap()
            .iter()
            .map(|&v| v as usize)
            .collect()
    };

    let check = |label: &str, logits: &[f32]| {
        let rank = rank_of(logits);
        let old = sum_inversion(&rank);
        let fixed = select_inversion(&rank);
        let orc = oracle(&rank);
        eprintln!("{label}: OLD-sum={old:?} FIXED-select={fixed:?} oracle={orc:?}");
        assert_eq!(
            fixed, orc,
            "{label}: fixed select must equal the eval oracle"
        );
        assert!(
            fixed.iter().all(|&x| x < e),
            "{label}: fixed ids must be in [0,E)"
        );
        (old, fixed)
    };

    // (A) Varied, distinct logits -> rank is a permutation (no ties) -> old==fixed, both in range.
    let varied: Vec<f32> = (0..e)
        .map(|i| ((i * 977 % e) as f32) * 0.013 - 1.7)
        .collect();
    let mut sorted = varied.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup();
    assert_eq!(sorted.len(), e, "varied logits must be distinct (no ties)");
    let (old_a, fixed_a) = check("VARIED-notie", &varied);
    assert_eq!(old_a, fixed_a, "no-tie: sum and select agree");

    // (B) Degenerate, all-equal -> all rank 0. Old sum slot0 = 0+..+255 = 32640 (OOB); fixed = 255.
    let degen = vec![0.0f32; e];
    let (old_b, fixed_b) = check("DEGENERATE", &degen);
    assert_eq!(
        old_b[0],
        (0..e).sum::<usize>(),
        "OLD sum slot0 = sum of all ids (32640)"
    );
    assert!(old_b[0] >= e, "OLD degenerate id was OUT OF RANGE");
    assert_eq!(
        fixed_b[0],
        e - 1,
        "FIXED degenerate slot0 = last (255) expert"
    );

    // (C) High tie at experts 200,201 (both top -> rank 0). Old sum slot0 = 401 > E (OOB); fixed = 201.
    let mut tie = varied.clone();
    let top = tie.iter().cloned().fold(f32::MIN, f32::max) + 1.0;
    tie[200] = top;
    tie[201] = top;
    let (old_c, fixed_c) = check("HIGHTIE-200-201", &tie);
    assert_eq!(old_c[0], 401, "OLD sum slot0 = 200+201 = 401 (>E=256, OOB)");
    assert!(old_c[0] >= e, "OLD high-tie id was OUT OF RANGE");
    assert_eq!(
        fixed_c[0], 201,
        "FIXED high-tie slot0 = last (201) tied expert"
    );
}

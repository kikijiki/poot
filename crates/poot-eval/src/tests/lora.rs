//! LoRA-adapted linear (spec 248): [`poot_graph_ir::ops::lora_linear`] and `lora_linear_batched` in isolation. The
//! full-model wiring rows are parked under POOT-579 SC-003.

use crate::{EvalBudget, EvalOptions, Value, eval};
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error};

/// `lora_linear`'s primitive composition matches a hand-computed `W + scaling*B@A` reference on a tiny synthetic case,
/// and visibly differs from the un-adapted base output (a zero-correction bug would pass the shape checks but not this).
#[test]
fn lora_linear_matches_hand_computed_reference() {
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::ops::lora_linear;
    use poot_graph_ir::types::TensorType;

    let (m, in_dim, out_dim, r) = (2usize, 3usize, 2usize, 2usize);
    let scaling = 1.5f32;

    let xd = fill(m * in_dim, 1);
    let wd = fill(in_dim * out_dim, 2);
    let bias_d = fill(out_dim, 3);
    let ad = fill(in_dim * r, 4); // lora_a [in, r]
    let bd = fill(r * out_dim, 5); // lora_b [r, out]

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![m, in_dim]));
    let w = b.constant("w", TensorType::f32(vec![in_dim, out_dim]));
    let bias = b.constant("bias", TensorType::f32(vec![out_dim]));
    let la = b.constant("la", TensorType::f32(vec![in_dim, r]));
    let lb = b.constant("lb", TensorType::f32(vec![r, out_dim]));
    let y = lora_linear(&b, x, w, Some(bias), la, lb, scaling);
    let (xi, wi, bi, lai, lbi) = (x.id, w.id, bias.id, la.id, lb.id);
    let g = b.finish(y);

    let mut inputs = HashMap::new();
    inputs.insert(xi, HostTensor::f32(vec![m, in_dim], xd.clone()));
    inputs.insert(wi, HostTensor::f32(vec![in_dim, out_dim], wd.clone()));
    inputs.insert(bi, HostTensor::f32(vec![out_dim], bias_d.clone()));
    inputs.insert(lai, HostTensor::f32(vec![in_dim, r], ad.clone()));
    inputs.insert(lbi, HostTensor::f32(vec![r, out_dim], bd.clone()));
    let got = {
        let values: HashMap<_, Value> = inputs
            .iter()
            .map(|(&id, t)| (id, Value::from(t.clone())))
            .collect();
        eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .expect("lora tests evaluate dense graphs")
    };

    // Independent reference: base = x@w + bias; correction = scaling * (x@lora_a) @ lora_b.
    let base = matmul_ref(&xd, &wd, m, in_dim, out_dim);
    let base_biased: Vec<f32> = (0..m)
        .flat_map(|i| (0..out_dim).map(move |j| (i, j)))
        .map(|(i, j)| base[i * out_dim + j] + bias_d[j])
        .collect();
    let xa = matmul_ref(&xd, &ad, m, in_dim, r);
    let xab = matmul_ref(&xa, &bd, m, r, out_dim);
    let want: Vec<f32> = base_biased
        .iter()
        .zip(xab.iter())
        .map(|(bv, cv)| bv + scaling * cv)
        .collect();
    assert_close_rel(got.as_f32().unwrap(), &want, 1e-5);

    // The adapter must visibly change the output vs the un-adapted base (catches a correction that traces to a no-op).
    let max_diff = max_abs_error(&base_biased, got.as_f32().unwrap());
    assert!(
        max_diff > 1e-3,
        "lora correction should visibly change the output, got max_diff={max_diff}"
    );
}

/// `lora_linear_batched` batches `M` rows, each with its own adapter id, into one dispatch (two `IndexedMatMul`s, the
/// same primitive `moe_sparse`/`moe_grouped` use). Builds a 3-row batch (row 0 -> "small_r" [r=2], row 1 -> "big_r"
/// [r=3, different rank, proving the pool's zero-padding is exact], row 2 -> `LoraAdapterPool::NO_ADAPTER`) via
/// `poot_load::lora::LoraAdapterPool::stack_for_module`, and checks the output against each row run through the
/// sequential path (`lora_linear` for adapted rows, plain `linear` for the other).
mod lora_linear_batched_tests {
    use super::*;
    use poot_graph_ir::builder::Builder;
    use poot_graph_ir::ops::{lora_linear, lora_linear_batched};
    use poot_graph_ir::types::TensorType;
    use poot_load::lora::{LoraAdapter, LoraAdapterConfig, LoraAdapterPool, LoraWeight};

    fn raw(shape: Vec<usize>, data: Vec<f32>) -> poot_tensor::HostTensor {
        poot_tensor::HostTensor::f32(shape, data)
    }

    /// Synthetic adapter targeting `module` with rank `r`, already in poot's post-transpose `[in,r]`/`[r,out]` layout.
    fn adapter(r: usize, in_dim: usize, out_dim: usize, module: &str, seed: u64) -> LoraAdapter {
        let a = fill(in_dim * r, seed);
        let b = fill(r * out_dim, seed + 1);
        let mut weights = std::collections::HashMap::new();
        weights.insert(
            module.to_string(),
            LoraWeight {
                a: raw(vec![in_dim, r], a),
                b: raw(vec![r, out_dim], b),
            },
        );
        LoraAdapter {
            config: LoraAdapterConfig {
                r,
                lora_alpha: (3 * r) as f32, // scaling = alpha/r = 3.0, easy to sanity-check by eye
                target_modules: vec![module.to_string()],
                use_rslora: false,
            },
            weights,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eval_lora_linear_batched(
        x: &[f32],
        m: usize,
        in_dim: usize,
        out_dim: usize,
        w: &[f32],
        bias: &[f32],
        stacked: &poot_load::lora::LoraStackedModule,
        idx: &[f32],
    ) -> Vec<f32> {
        let b = Builder::new();
        let xt = b.constant("x", TensorType::f32(vec![m, in_dim]));
        let wt = b.constant("w", TensorType::f32(vec![in_dim, out_dim]));
        let biast = b.constant("bias", TensorType::f32(vec![out_dim]));
        let at = b.constant("a", TensorType::f32(stacked.a.shape().to_vec()));
        let bt = b.constant("bstack", TensorType::f32(stacked.b.shape().to_vec()));
        let st = b.constant("scaling", TensorType::f32(stacked.scaling.shape().to_vec()));
        let it = b.constant("idx", TensorType::f32(vec![m]));
        let y = lora_linear_batched(&b, xt, wt, Some(biast), at, bt, st, it);
        let (xi, wi, bi, ai, bsi, sci, idxi) = (xt.id, wt.id, biast.id, at.id, bt.id, st.id, it.id);
        let g = b.finish(y);

        let mut inputs = HashMap::new();
        inputs.insert(xi, HostTensor::f32(vec![m, in_dim], x.to_vec()));
        inputs.insert(wi, HostTensor::f32(vec![in_dim, out_dim], w.to_vec()));
        inputs.insert(bi, HostTensor::f32(vec![out_dim], bias.to_vec()));
        inputs.insert(
            ai,
            HostTensor::f32(
                stacked.a.shape().to_vec(),
                stacked.a.as_f32().unwrap().to_vec(),
            ),
        );
        inputs.insert(
            bsi,
            HostTensor::f32(
                stacked.b.shape().to_vec(),
                stacked.b.as_f32().unwrap().to_vec(),
            ),
        );
        inputs.insert(
            sci,
            HostTensor::f32(
                stacked.scaling.shape().to_vec(),
                stacked.scaling.as_f32().unwrap().to_vec(),
            ),
        );
        inputs.insert(idxi, HostTensor::f32(vec![m], idx.to_vec()));
        {
            let values: HashMap<_, Value> = inputs
                .iter()
                .map(|(&id, t)| (id, Value::from(t.clone())))
                .collect();
            eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .expect("lora tests evaluate dense graphs")
        }
        .as_f32()
        .unwrap()
        .to_vec()
    }

    /// One row's reference via `lora_linear`, or plain `linear` for the no-adapter row.
    fn eval_one_row(
        x_row: &[f32],
        in_dim: usize,
        out_dim: usize,
        w: &[f32],
        bias: &[f32],
        weight: Option<(&LoraWeight, f32)>, // (a/b, scaling), None = no adapter
    ) -> Vec<f32> {
        use poot_graph_ir::ops::linear;
        let b = Builder::new();
        let xt = b.constant("x", TensorType::f32(vec![1, in_dim]));
        let wt = b.constant("w", TensorType::f32(vec![in_dim, out_dim]));
        let biast = b.constant("bias", TensorType::f32(vec![out_dim]));
        let mut inputs = HashMap::new();
        inputs.insert(xt.id, HostTensor::f32(vec![1, in_dim], x_row.to_vec()));
        inputs.insert(wt.id, HostTensor::f32(vec![in_dim, out_dim], w.to_vec()));
        inputs.insert(biast.id, HostTensor::f32(vec![out_dim], bias.to_vec()));
        let y = match weight {
            Some((lw, scaling)) => {
                let r = lw.a.shape()[1];
                let la = b.constant("la", TensorType::f32(vec![in_dim, r]));
                let lb = b.constant("lb", TensorType::f32(vec![r, out_dim]));
                inputs.insert(
                    la.id,
                    HostTensor::f32(vec![in_dim, r], lw.a.as_f32().unwrap().to_vec()),
                );
                inputs.insert(
                    lb.id,
                    HostTensor::f32(vec![r, out_dim], lw.b.as_f32().unwrap().to_vec()),
                );
                lora_linear(&b, xt, wt, Some(biast), la, lb, scaling)
            }
            None => linear(&b, xt, wt, Some(biast)),
        };
        let g = b.finish(y);
        {
            let values: HashMap<_, Value> = inputs
                .iter()
                .map(|(&id, t)| (id, Value::from(t.clone())))
                .collect();
            eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .expect("lora tests evaluate dense graphs")
        }
        .as_f32()
        .unwrap()
        .to_vec()
    }

    #[test]
    fn lora_linear_batched_matches_sequential_single_adapter_dispatch() {
        let module = "model.layers.0.self_attn.q_proj";
        let (in_dim, out_dim) = (5usize, 4usize);
        let m = 3;

        let adapter_small = adapter(2, in_dim, out_dim, module, 10);
        let adapter_big = adapter(3, in_dim, out_dim, module, 20);

        let mut pool = LoraAdapterPool::new();
        let idx_small = pool.register("small_r", adapter_small.clone());
        let idx_big = pool.register("big_r", adapter_big.clone());
        let stacked = pool
            .stack_for_module(module)
            .unwrap()
            .expect("both adapters target this module");

        let w = fill(in_dim * out_dim, 1);
        let bias = fill(out_dim, 2);
        let x = fill(m * in_dim, 3);
        // row 0 -> small_r, row 1 -> big_r, row 2 -> NO_ADAPTER.
        let idx = vec![
            idx_small as f32,
            idx_big as f32,
            LoraAdapterPool::NO_ADAPTER as f32,
        ];

        let got = eval_lora_linear_batched(&x, m, in_dim, out_dim, &w, &bias, &stacked, &idx);
        assert_eq!(got.len(), m * out_dim);

        // Independent per-row reference: each row through the sequential single-adapter path alone, using the same adapter structs.
        let want_row0 = eval_one_row(
            &x[0..in_dim],
            in_dim,
            out_dim,
            &w,
            &bias,
            Some((
                adapter_small.weight_for(module).unwrap(),
                adapter_small.config.scaling(),
            )),
        );
        let want_row1 = eval_one_row(
            &x[in_dim..2 * in_dim],
            in_dim,
            out_dim,
            &w,
            &bias,
            Some((
                adapter_big.weight_for(module).unwrap(),
                adapter_big.config.scaling(),
            )),
        );
        let want_row2 = eval_one_row(&x[2 * in_dim..3 * in_dim], in_dim, out_dim, &w, &bias, None);

        assert_close_rel(&got[0..out_dim], &want_row0, 1e-5);
        assert_close_rel(&got[out_dim..2 * out_dim], &want_row1, 1e-5);
        assert_close_rel(&got[2 * out_dim..3 * out_dim], &want_row2, 1e-5);

        // Sanity: the three rows must visibly differ from each other.
        let d01 = max_abs_error(&got[0..out_dim], &got[out_dim..2 * out_dim]);
        let d02 = max_abs_error(&got[0..out_dim], &got[2 * out_dim..3 * out_dim]);
        assert!(
            d01 > 1e-3,
            "row0 (small_r) vs row1 (big_r) should differ, got {d01}"
        );
        assert!(
            d02 > 1e-3,
            "row0 (small_r) vs row2 (no adapter) should differ, got {d02}"
        );
    }

    /// No cross-row leakage: perturbing row 1's `x` and re-running the same batched dispatch must leave every other row's
    /// output bit-for-bit unchanged.
    #[test]
    fn lora_linear_batched_rows_are_independent_no_cross_row_leakage() {
        let module = "model.layers.2.self_attn.v_proj";
        let (in_dim, out_dim) = (4usize, 3usize);
        let m = 3;

        let mut pool = LoraAdapterPool::new();
        let idx_a = pool.register("adapter_a", adapter(2, in_dim, out_dim, module, 30));
        let idx_b = pool.register("adapter_b", adapter(4, in_dim, out_dim, module, 40));
        let stacked = pool.stack_for_module(module).unwrap().unwrap();

        let w = fill(in_dim * out_dim, 5);
        let bias = fill(out_dim, 6);
        let idx = vec![
            idx_a as f32,
            idx_b as f32,
            LoraAdapterPool::NO_ADAPTER as f32,
        ];

        let x1 = fill(m * in_dim, 7);
        let mut x2 = x1.clone();
        // Perturb ONLY row 1's activation (elements [in_dim..2*in_dim)).
        for v in &mut x2[in_dim..2 * in_dim] {
            *v += 100.0;
        }

        let got1 = eval_lora_linear_batched(&x1, m, in_dim, out_dim, &w, &bias, &stacked, &idx);
        let got2 = eval_lora_linear_batched(&x2, m, in_dim, out_dim, &w, &bias, &stacked, &idx);

        // Row 0 and row 2 must be exactly unchanged (bit-for-bit).
        assert_eq!(
            &got1[0..out_dim],
            &got2[0..out_dim],
            "row 0 leaked row 1's perturbation"
        );
        assert_eq!(
            &got1[2 * out_dim..3 * out_dim],
            &got2[2 * out_dim..3 * out_dim],
            "row 2 leaked row 1's perturbation"
        );
        // Row 1 (the perturbed row) must differ, otherwise the no-leakage check above would be vacuous.
        let d1 = max_abs_error(&got1[out_dim..2 * out_dim], &got2[out_dim..2 * out_dim]);
        assert!(
            d1 > 1.0,
            "row 1 should reflect its own perturbation, got diff {d1}"
        );
    }
}

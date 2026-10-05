//! Qwen3Next GDN prefill blocks, batched decode/admission, gated attention prefill, mamba core paths.

use crate::{EvalBudget, EvalError, EvalOptions, Value};
use poot_graph_ir::ValueId;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::graph::StateRole;
use poot_graph_ir::ops::{gated_delta_net_decode, gdn_prefill_chunked, mamba2_ssd_decode};
use poot_graph_ir::types::TensorType;
use poot_tensor::DType;
use poot_tensor::HostTensor;
use std::collections::HashMap;

use super::helpers::*;
use poot_test_util::{assert_close_rel, max_abs_error, seed_of};

/// Whole-block analogue of `gdn_prefill_chunked_matches_decode_recurrence`: `qwen3next_gdn_prefill_block`
/// must reproduce `L` sequential `qwen3next_gdn_block` calls in per-position output and in the final
/// carried state (conv cache + GDN state). `h_k=2, h_v=4` (tiled GQA), `L=8, chunk=4`, zero initial
/// cache/state, shared weights.
#[test]
fn qwen3next_gdn_prefill_block_matches_decode() {
    use poot_models::qwen3next::{GdnHeadOrder, qwen3next_gdn_block, qwen3next_gdn_prefill_block};

    let (h, hk, hv, d, ck, l, chunk) = (5usize, 2usize, 4usize, 3usize, 4usize, 8usize, 4usize);
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6;

    let xd = fill(l * h, 111); // [1, L, H]
    let w_qkv_d = fill(h * conv_dim, 112);
    let w_gate_d = fill(h * value_dim, 113);
    let w_conv_d = fill(ck * conv_dim, 114);
    let w_beta_d = fill(h * hv, 115);
    let w_alpha_d = fill(h * hv, 116);
    let dt_bias_d = fill(hv, 117);
    // ssm_a strictly negative so g = softplus(..)*ssm_a is a real decay for any x.
    let ssm_a_d: Vec<f32> = fill(hv, 118).into_iter().map(|v| -0.2 - v.abs()).collect();
    let norm_w_d = fill(d, 119);
    let w_out_d = fill(value_dim * h, 120);

    // --- sequential decode reference ---
    let db = Builder::new();
    let dx = db.constant("x", TensorType::f32(vec![1, 1, h]));
    let dw_qkv = db.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let dw_gate = db.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let dw_conv = db.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let dw_beta = db.constant("w_beta", TensorType::f32(vec![h, hv]));
    let dw_alpha = db.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let ddt_bias = db.constant("dt_bias", TensorType::f32(vec![hv]));
    let dssm_a = db.constant("ssm_a", TensorType::f32(vec![hv]));
    let dnorm_w = db.constant("norm_w", TensorType::f32(vec![d]));
    let dw_out = db.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let dcache_in = db.state_input(
        "conv_cache",
        TensorType::f32(vec![1, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ds_in = db.state_input(
        "s",
        TensorType::f32(vec![1, hv, d, d]),
        StateRole::Recurrent,
    );
    let (dout, dcache_out, ds_out) = qwen3next_gdn_block(
        &db,
        dx,
        dw_qkv,
        dw_gate,
        dw_conv,
        dw_beta,
        dw_alpha,
        ddt_bias,
        dssm_a,
        dnorm_w,
        dw_out,
        dcache_in,
        ds_in,
        hk,
        hv,
        d,
        ck,
        eps,
        GdnHeadOrder::Tiled,
    );
    let dg = db.finish_with_state(dout, &[(dcache_in, dcache_out), (ds_in, ds_out)]);

    let mut cache = HostTensor::f32(vec![1, ck - 1, conv_dim], vec![0.0f32; (ck - 1) * conv_dim]);
    let mut state = HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]);
    let mut want_o = vec![0.0f32; l * h];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            dx.id,
            HostTensor::f32(vec![1, 1, h], xd[t * h..(t + 1) * h].to_vec()),
        );
        inp.insert(
            dw_qkv.id,
            HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
        );
        inp.insert(
            dw_gate.id,
            HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
        );
        inp.insert(
            dw_conv.id,
            HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
        );
        inp.insert(dw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
        inp.insert(dw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
        inp.insert(ddt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
        inp.insert(dssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
        inp.insert(dnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
        inp.insert(
            dw_out.id,
            HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
        );
        inp.insert(dcache_in.id, cache.clone());
        inp.insert(ds_in.id, state.clone());

        let (ot, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        assert_eq!(ot.shape(), vec![1, 1, h]);
        want_o[t * h..(t + 1) * h].copy_from_slice(ot.as_f32().unwrap());
        let mut it = new_states.into_iter();
        cache = it.next().unwrap();
        state = it.next().unwrap();
    }
    let want_cache = cache.as_f32().unwrap().to_vec();
    let want_state = state.as_f32().unwrap().to_vec();

    // --- prefill block: one call over all L positions ---
    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c]; // lower-triangular INCLUDING the diagonal (S2)
    let mut tril_strict = vec![0.0f32; c * c]; // STRICTLY lower-triangular, zero diagonal (S2)
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, h]));
    let pw_qkv = pb.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let pw_gate = pb.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let pw_conv = pb.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let pw_beta = pb.constant("w_beta", TensorType::f32(vec![h, hv]));
    let pw_alpha = pb.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let pdt_bias = pb.constant("dt_bias", TensorType::f32(vec![hv]));
    let pssm_a = pb.constant("ssm_a", TensorType::f32(vec![hv]));
    let pnorm_w = pb.constant("norm_w", TensorType::f32(vec![d]));
    let pw_out = pb.constant("w_out", TensorType::f32(vec![value_dim, h]));
    // s_in: zero GDN state (fresh prefill).
    let ps_in = pb.state_input(
        "s",
        TensorType::f32(vec![1, hv, d, d]),
        StateRole::Recurrent,
    );
    // A fresh prefill has no incoming conv cache; pair conv_cache_out with a placeholder state_input so
    // eval_with_state returns it.
    let pconv_cache_ph = pb.state_input(
        "conv_cache_ph",
        TensorType::f32(vec![1, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ptril_incl = pb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let ptril_strict = pb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (pout, pcache_out, ps_out) = qwen3next_gdn_prefill_block(
        &pb,
        px,
        pw_qkv,
        pw_gate,
        pw_conv,
        pw_beta,
        pw_alpha,
        pdt_bias,
        pssm_a,
        pnorm_w,
        pw_out,
        ps_in,
        ptril_incl,
        ptril_strict,
        hk,
        hv,
        d,
        ck,
        c,
        eps,
        GdnHeadOrder::Tiled,
    );
    assert_eq!(
        pb.aval(pout).shape,
        vec![1, l, h],
        "prefill block output shape [1,L,H]"
    );
    assert_eq!(
        pb.aval(pcache_out).shape,
        vec![1, ck - 1, conv_dim],
        "prefill block conv cache shape"
    );
    assert_eq!(
        pb.aval(ps_out).shape,
        vec![1, hv, d, d],
        "prefill block state shape"
    );
    let pg = pb.finish_with_state(pout, &[(ps_in, ps_out), (pconv_cache_ph, pcache_out)]);

    let mut pinp = HashMap::new();
    pinp.insert(px.id, HostTensor::f32(vec![1, l, h], xd));
    pinp.insert(pw_qkv.id, HostTensor::f32(vec![h, conv_dim], w_qkv_d));
    pinp.insert(pw_gate.id, HostTensor::f32(vec![h, value_dim], w_gate_d));
    pinp.insert(pw_conv.id, HostTensor::f32(vec![ck, conv_dim], w_conv_d));
    pinp.insert(pw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d));
    pinp.insert(pw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d));
    pinp.insert(pdt_bias.id, HostTensor::f32(vec![hv], dt_bias_d));
    pinp.insert(pssm_a.id, HostTensor::f32(vec![hv], ssm_a_d));
    pinp.insert(pnorm_w.id, HostTensor::f32(vec![d], norm_w_d));
    pinp.insert(pw_out.id, HostTensor::f32(vec![value_dim, h], w_out_d));
    pinp.insert(
        ps_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    pinp.insert(
        pconv_cache_ph.id,
        HostTensor::f32(vec![1, ck - 1, conv_dim], vec![0.0f32; (ck - 1) * conv_dim]),
    );
    pinp.insert(ptril_incl.id, HostTensor::f32(vec![1, 1, c, c], tril_incl));
    pinp.insert(
        ptril_strict.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict),
    );

    let (got_o, got_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = pinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&pg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    assert_eq!(got_o.shape(), vec![1, l, h]);
    let mut it = got_states.into_iter();
    let got_state = it.next().unwrap();
    let got_cache = it.next().unwrap();
    assert_eq!(got_state.shape(), vec![1, hv, d, d]);
    assert_eq!(got_cache.shape(), vec![1, ck - 1, conv_dim]);

    let o_err = max_abs_error(got_o.as_f32().unwrap(), &want_o);
    let cache_err = max_abs_error(got_cache.as_f32().unwrap(), &want_cache);
    let s_err = max_abs_error(got_state.as_f32().unwrap(), &want_state);
    eprintln!(
        "qwen3next_gdn_prefill_block_matches_decode: output_max_abs_err={o_err:.2e} \
         conv_cache_max_abs_err={cache_err:.2e} state_max_abs_err={s_err:.2e}"
    );
    assert_close_rel(got_o.as_f32().unwrap(), &want_o, 1e-4);
    assert_close_rel(got_cache.as_f32().unwrap(), &want_cache, 1e-4);
    assert_close_rel(got_state.as_f32().unwrap(), &want_state, 1e-4);
}

/// [`qwen3next_gdn_prefill_into_slot`] writing pool slot 2 of a 4-slot pool must leave every other slot's
/// `conv_cache`/`ssm_state` byte-unchanged.
/// Uses real-magnitude `ssm_a` (`-41..-1`) because the chunked GDN region is numerically delicate.
#[test]
fn qwen3next_gdn_prefill_into_slot_touches_only_target_slot() {
    use poot_models::qwen3next::{GdnHeadOrder, qwen3next_gdn_prefill_into_slot};

    let (h, hk, hv, d, ck, l, chunk) = (6usize, 2usize, 6usize, 4usize, 4usize, 12usize, 4usize);
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6;
    let n_slots = 4usize;
    let target_slot = 2usize;

    let xd = fill(l * h, 201);
    let w_qkv_d = fill(h * conv_dim, 202);
    let w_gate_d = fill(h * value_dim, 203);
    let w_conv_d = fill(ck * conv_dim, 204);
    let w_beta_d = fill(h * hv, 205);
    let w_alpha_d = fill(h * hv, 206);
    let dt_bias_d = fill(hv, 207);
    // Real-magnitude decay (real 35B ssm_a spans roughly [-41, -1]).
    let ssm_a_d: Vec<f32> = (0..hv)
        .map(|i| -41.0 + i as f32 * (40.0 / (hv - 1) as f32))
        .collect();
    let norm_w_d = fill(d, 208);
    let w_out_d = fill(value_dim * h, 209);

    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c];
    let mut tril_strict = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    let conv_row = (ck - 1) * conv_dim;
    let ssm_row = hv * d * d;
    // Distinct nonzero state in slots 0, 1, 3 (slot 2 is the target and stays zero) so a broadcast or
    // aliasing bug is caught.
    let mut conv_pool_d = vec![0.0f32; n_slots * conv_row];
    let mut ssm_pool_d = vec![0.0f32; n_slots * ssm_row];
    for slot in 0..n_slots {
        if slot == target_slot {
            continue;
        }
        let cv = fill(conv_row, 300 + slot as u64);
        conv_pool_d[slot * conv_row..(slot + 1) * conv_row].copy_from_slice(&cv);
        let sv = fill(ssm_row, 400 + slot as u64);
        ssm_pool_d[slot * ssm_row..(slot + 1) * ssm_row].copy_from_slice(&sv);
    }
    let conv_pool_before = conv_pool_d.clone();
    let ssm_pool_before = ssm_pool_d.clone();

    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, l, h]));
    let w_qkv = b.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let w_gate = b.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let w_conv = b.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let w_beta = b.constant("w_beta", TensorType::f32(vec![h, hv]));
    let w_alpha = b.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let dt_bias = b.constant("dt_bias", TensorType::f32(vec![hv]));
    let ssm_a = b.constant("ssm_a", TensorType::f32(vec![hv]));
    let norm_w = b.constant("norm_w", TensorType::f32(vec![d]));
    let w_out = b.constant("w_out", TensorType::f32(vec![value_dim, h]));
    // s_in: zero (fresh admission).
    let s_in = b.constant("s_in", TensorType::f32(vec![1, hv, d, d]));
    let tril_incl_t = b.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let tril_strict_t = b.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let conv_pool_in = b.state_input(
        "conv_pool",
        TensorType::f32(vec![n_slots, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ssm_pool_in = b.state_input(
        "ssm_pool",
        TensorType::f32(vec![n_slots, hv, d, d]),
        StateRole::Recurrent,
    );
    let slot_t = b.constant("slot", TensorType::scalar(DType::I32));

    let (cur, conv_pool_out, ssm_pool_out) = qwen3next_gdn_prefill_into_slot(
        &b,
        x,
        w_qkv,
        w_gate,
        w_conv,
        w_beta,
        w_alpha,
        dt_bias,
        ssm_a,
        norm_w,
        w_out,
        s_in,
        tril_incl_t,
        tril_strict_t,
        conv_pool_in,
        ssm_pool_in,
        slot_t,
        hk,
        hv,
        d,
        ck,
        chunk,
        eps,
        GdnHeadOrder::Tiled,
    );
    assert_eq!(
        b.aval(conv_pool_out).shape,
        vec![n_slots, ck - 1, conv_dim],
        "conv pool output keeps the pool shape"
    );
    assert_eq!(
        b.aval(ssm_pool_out).shape,
        vec![n_slots, hv, d, d],
        "ssm pool output keeps the pool shape"
    );
    let g = b.finish_with_state(
        cur,
        &[(conv_pool_in, conv_pool_out), (ssm_pool_in, ssm_pool_out)],
    );

    let mut inp = HashMap::new();
    inp.insert(x.id, HostTensor::f32(vec![1, l, h], xd));
    inp.insert(w_qkv.id, HostTensor::f32(vec![h, conv_dim], w_qkv_d));
    inp.insert(w_gate.id, HostTensor::f32(vec![h, value_dim], w_gate_d));
    inp.insert(w_conv.id, HostTensor::f32(vec![ck, conv_dim], w_conv_d));
    inp.insert(w_beta.id, HostTensor::f32(vec![h, hv], w_beta_d));
    inp.insert(w_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d));
    inp.insert(dt_bias.id, HostTensor::f32(vec![hv], dt_bias_d));
    inp.insert(ssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
    inp.insert(norm_w.id, HostTensor::f32(vec![d], norm_w_d));
    inp.insert(w_out.id, HostTensor::f32(vec![value_dim, h], w_out_d));
    inp.insert(
        s_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    inp.insert(tril_incl_t.id, HostTensor::f32(vec![1, 1, c, c], tril_incl));
    inp.insert(
        tril_strict_t.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict),
    );
    inp.insert(
        conv_pool_in.id,
        HostTensor::f32(vec![n_slots, ck - 1, conv_dim], conv_pool_d.clone()),
    );
    inp.insert(
        ssm_pool_in.id,
        HostTensor::f32(vec![n_slots, hv, d, d], ssm_pool_d.clone()),
    );
    inp.insert(slot_t.id, HostTensor::i32(vec![], vec![target_slot as i32]));

    let (_cur_out, states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = inp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    let mut it = states.into_iter();
    let conv_pool_after = it.next().unwrap();
    let ssm_pool_after = it.next().unwrap();
    assert_eq!(conv_pool_after.shape(), vec![n_slots, ck - 1, conv_dim]);
    assert_eq!(ssm_pool_after.shape(), vec![n_slots, hv, d, d]);

    for slot in 0..n_slots {
        let cbefore = &conv_pool_before[slot * conv_row..(slot + 1) * conv_row];
        let cafter = &conv_pool_after.as_f32().unwrap()[slot * conv_row..(slot + 1) * conv_row];
        let sbefore = &ssm_pool_before[slot * ssm_row..(slot + 1) * ssm_row];
        let safter = &ssm_pool_after.as_f32().unwrap()[slot * ssm_row..(slot + 1) * ssm_row];
        if slot == target_slot {
            // The written slot must change, otherwise the isolation check below passes vacuously.
            let changed = cafter.iter().any(|&v| v != 0.0) || safter.iter().any(|&v| v != 0.0);
            assert!(
                changed,
                "slot {target_slot} should have been written (nonzero)"
            );
        } else {
            let conv_err = max_abs_error(cafter, cbefore);
            let ssm_err = max_abs_error(safter, sbefore);
            assert_eq!(
                conv_err, 0.0,
                "slot {slot} conv_cache must be BYTE-UNCHANGED by a write to slot {target_slot}, \
                 got max_abs_err={conv_err}"
            );
            assert_eq!(
                ssm_err, 0.0,
                "slot {slot} ssm_state must be BYTE-UNCHANGED by a write to slot {target_slot}, \
                 got max_abs_err={ssm_err}"
            );
        }
    }
    eprintln!(
        "qwen3next_gdn_prefill_into_slot_touches_only_target_slot: slots {{0,1,3}} \
         max_abs_err=0.0 (exact byte-unchanged), slot {target_slot} written, real-magnitude \
         ssm_a range=[{:.1},{:.1}]",
        ssm_a_d[0],
        ssm_a_d[hv - 1]
    );
}

/// The value [`qwen3next_gdn_prefill_into_slot`] writes into slot `target_slot` must match a bare
/// [`qwen3next_gdn_prefill_block`] call on the same inputs exactly: the `DynamicUpdateSlice` wrapper only
/// places values. Real-magnitude `ssm_a` (`-41..-1`), `L=12, chunk=4` (3 chunk boundaries); other slots
/// hold nonzero garbage.
#[test]
fn qwen3next_gdn_prefill_into_slot_matches_bare_prefill_block_real_magnitude_ssm_a() {
    use poot_models::qwen3next::{
        GdnHeadOrder, qwen3next_gdn_prefill_block, qwen3next_gdn_prefill_into_slot,
    };

    let (h, hk, hv, d, ck, l, chunk) = (6usize, 2usize, 6usize, 4usize, 4usize, 12usize, 4usize);
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6;
    let n_slots = 4usize;
    let target_slot = 2usize;

    let xd = fill(l * h, 701);
    let w_qkv_d = fill(h * conv_dim, 702);
    let w_gate_d = fill(h * value_dim, 703);
    let w_conv_d = fill(ck * conv_dim, 704);
    let w_beta_d = fill(h * hv, 705);
    let w_alpha_d = fill(h * hv, 706);
    let dt_bias_d = fill(hv, 707);
    let ssm_a_d: Vec<f32> = (0..hv)
        .map(|i| -41.0 + i as f32 * (40.0 / (hv - 1) as f32))
        .collect();
    let norm_w_d = fill(d, 708);
    let w_out_d = fill(value_dim * h, 709);

    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c];
    let mut tril_strict = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    // --- reference: bare qwen3next_gdn_prefill_block, no pool, zero s_in. ds_in/dcache_ph are placeholder
    // state_inputs so finish_with_state returns cache_out/s_out. ---
    let db = Builder::new();
    let dx = db.constant("x", TensorType::f32(vec![1, l, h]));
    let dw_qkv = db.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let dw_gate = db.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let dw_conv = db.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let dw_beta = db.constant("w_beta", TensorType::f32(vec![h, hv]));
    let dw_alpha = db.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let ddt_bias = db.constant("dt_bias", TensorType::f32(vec![hv]));
    let dssm_a = db.constant("ssm_a", TensorType::f32(vec![hv]));
    let dnorm_w = db.constant("norm_w", TensorType::f32(vec![d]));
    let dw_out = db.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let ds_in = db.state_input(
        "s_in_ph",
        TensorType::f32(vec![1, hv, d, d]),
        StateRole::Recurrent,
    );
    let dcache_ph = db.state_input(
        "cache_ph",
        TensorType::f32(vec![1, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let dtril_incl = db.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let dtril_strict = db.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (dcur, dcache_out, ds_out) = qwen3next_gdn_prefill_block(
        &db,
        dx,
        dw_qkv,
        dw_gate,
        dw_conv,
        dw_beta,
        dw_alpha,
        ddt_bias,
        dssm_a,
        dnorm_w,
        dw_out,
        ds_in,
        dtril_incl,
        dtril_strict,
        hk,
        hv,
        d,
        ck,
        chunk,
        eps,
        GdnHeadOrder::Tiled,
    );
    let dg = db.finish_with_state(dcur, &[(ds_in, ds_out), (dcache_ph, dcache_out)]);
    let mut dinp = HashMap::new();
    dinp.insert(dx.id, HostTensor::f32(vec![1, l, h], xd.clone()));
    dinp.insert(
        dw_qkv.id,
        HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
    );
    dinp.insert(
        dw_gate.id,
        HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
    );
    dinp.insert(
        dw_conv.id,
        HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
    );
    dinp.insert(dw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
    dinp.insert(dw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
    dinp.insert(ddt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
    dinp.insert(dssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
    dinp.insert(dnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
    dinp.insert(
        dw_out.id,
        HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
    );
    dinp.insert(
        ds_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    dinp.insert(
        dcache_ph.id,
        HostTensor::f32(vec![1, ck - 1, conv_dim], vec![0.0f32; (ck - 1) * conv_dim]),
    );
    dinp.insert(
        dtril_incl.id,
        HostTensor::f32(vec![1, 1, c, c], tril_incl.clone()),
    );
    dinp.insert(
        dtril_strict.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict.clone()),
    );
    let (want_cur, dstates) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = dinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    let mut dit = dstates.into_iter();
    let want_s_out = dit.next().unwrap();
    let want_cache_out = dit.next().unwrap();

    // --- slot-addressed: same inputs written into slot `target_slot` of a 4-slot pool whose other slots
    // hold nonzero garbage. ---
    let conv_row = (ck - 1) * conv_dim;
    let ssm_row = hv * d * d;
    let mut conv_pool_d = vec![0.0f32; n_slots * conv_row];
    let mut ssm_pool_d = vec![0.0f32; n_slots * ssm_row];
    for slot in 0..n_slots {
        if slot == target_slot {
            continue;
        }
        conv_pool_d[slot * conv_row..(slot + 1) * conv_row]
            .copy_from_slice(&fill(conv_row, 800 + slot as u64));
        ssm_pool_d[slot * ssm_row..(slot + 1) * ssm_row]
            .copy_from_slice(&fill(ssm_row, 900 + slot as u64));
    }

    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, h]));
    let pw_qkv = pb.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let pw_gate = pb.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let pw_conv = pb.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let pw_beta = pb.constant("w_beta", TensorType::f32(vec![h, hv]));
    let pw_alpha = pb.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let pdt_bias = pb.constant("dt_bias", TensorType::f32(vec![hv]));
    let pssm_a = pb.constant("ssm_a", TensorType::f32(vec![hv]));
    let pnorm_w = pb.constant("norm_w", TensorType::f32(vec![d]));
    let pw_out = pb.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let ps_in = pb.constant("s_in", TensorType::f32(vec![1, hv, d, d]));
    let ptril_incl = pb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let ptril_strict = pb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let pconv_pool_in = pb.state_input(
        "conv_pool",
        TensorType::f32(vec![n_slots, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let pssm_pool_in = pb.state_input(
        "ssm_pool",
        TensorType::f32(vec![n_slots, hv, d, d]),
        StateRole::Recurrent,
    );
    let pslot = pb.constant("slot", TensorType::scalar(DType::I32));
    let (pcur, pconv_pool_out, pssm_pool_out) = qwen3next_gdn_prefill_into_slot(
        &pb,
        px,
        pw_qkv,
        pw_gate,
        pw_conv,
        pw_beta,
        pw_alpha,
        pdt_bias,
        pssm_a,
        pnorm_w,
        pw_out,
        ps_in,
        ptril_incl,
        ptril_strict,
        pconv_pool_in,
        pssm_pool_in,
        pslot,
        hk,
        hv,
        d,
        ck,
        chunk,
        eps,
        GdnHeadOrder::Tiled,
    );
    let pg = pb.finish_with_state(
        pcur,
        &[
            (pconv_pool_in, pconv_pool_out),
            (pssm_pool_in, pssm_pool_out),
        ],
    );

    let mut pinp = HashMap::new();
    pinp.insert(px.id, HostTensor::f32(vec![1, l, h], xd));
    pinp.insert(pw_qkv.id, HostTensor::f32(vec![h, conv_dim], w_qkv_d));
    pinp.insert(pw_gate.id, HostTensor::f32(vec![h, value_dim], w_gate_d));
    pinp.insert(pw_conv.id, HostTensor::f32(vec![ck, conv_dim], w_conv_d));
    pinp.insert(pw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d));
    pinp.insert(pw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d));
    pinp.insert(pdt_bias.id, HostTensor::f32(vec![hv], dt_bias_d));
    pinp.insert(pssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
    pinp.insert(pnorm_w.id, HostTensor::f32(vec![d], norm_w_d));
    pinp.insert(pw_out.id, HostTensor::f32(vec![value_dim, h], w_out_d));
    pinp.insert(
        ps_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    pinp.insert(ptril_incl.id, HostTensor::f32(vec![1, 1, c, c], tril_incl));
    pinp.insert(
        ptril_strict.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict),
    );
    pinp.insert(
        pconv_pool_in.id,
        HostTensor::f32(vec![n_slots, ck - 1, conv_dim], conv_pool_d),
    );
    pinp.insert(
        pssm_pool_in.id,
        HostTensor::f32(vec![n_slots, hv, d, d], ssm_pool_d),
    );
    pinp.insert(pslot.id, HostTensor::i32(vec![], vec![target_slot as i32]));

    let (got_cur, pstates) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = pinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&pg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    let mut pit = pstates.into_iter();
    let got_conv_pool = pit.next().unwrap();
    let got_ssm_pool = pit.next().unwrap();

    let got_cache_at_slot =
        &got_conv_pool.as_f32().unwrap()[target_slot * conv_row..(target_slot + 1) * conv_row];
    let got_state_at_slot =
        &got_ssm_pool.as_f32().unwrap()[target_slot * ssm_row..(target_slot + 1) * ssm_row];

    let cur_err = max_abs_error(got_cur.as_f32().unwrap(), want_cur.as_f32().unwrap());
    let cache_err = max_abs_error(got_cache_at_slot, want_cache_out.as_f32().unwrap());
    let state_err = max_abs_error(got_state_at_slot, want_s_out.as_f32().unwrap());
    eprintln!(
        "qwen3next_gdn_prefill_into_slot_matches_bare_prefill_block_real_magnitude_ssm_a: \
         cur_max_abs_err={cur_err:.2e} conv_cache_max_abs_err={cache_err:.2e} \
         ssm_state_max_abs_err={state_err:.2e} (real-magnitude ssm_a range=[{:.1},{:.1}], L={l} \
         chunk={chunk} => {} chunks)",
        ssm_a_d[0],
        ssm_a_d[hv - 1],
        l.div_ceil(chunk)
    );
    // DynamicUpdateSlice is a pure copy after the identical prefill subgraph: require bit-exact equality.
    assert_eq!(cur_err, 0.0, "cur output must exactly match the bare call");
    assert_eq!(
        cache_err, 0.0,
        "slot {target_slot}'s conv_cache must exactly match the bare call's conv_cache_out"
    );
    assert_eq!(
        state_err, 0.0,
        "slot {target_slot}'s ssm_state must exactly match the bare call's s_out"
    );
}

// --- Batched qwen3next decode tracer (attention shared-pool KV + GDN pool + grouped MoE), CPU oracle.
// Shared config/weights helpers for the tests below. ---

/// Tiny hybrid Qwen3-Next config (3 GDN layers + 1 full-attention layer per `full_attention_interval=4`),
/// GQA on both mixer kinds, MoE top-k < n_experts.
fn qwen3next_batched_test_config() -> poot_models::qwen3next::Qwen3NextConfig {
    poot_models::qwen3next::Qwen3NextConfig {
        vocab: 6,
        hidden: 12,
        n_layers: 4,
        full_attention_interval: 4, // is_attn_layer(li) = (li+1)%4==0 -> only li=3 is full-attention.
        eps: 1e-5,
        max_pos: 32,
        rotary_dim: 4,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        gdn_num_k_heads: 2,
        gdn_num_v_heads: 4,
        gdn_head_dim: 4,
        conv_k: 3,
        n_experts: 4,
        top_k: 2,
        expert_inter: 8,
        shared_inter: 8,
    }
}

/// Deterministic dense f32 weights for [`qwen3next_batched_test_config`], keyed by the constant names shared
/// by `qwen3next_decode_trace` (reference) and `qwen3next_decode_trace_batched_shared_pool` (under test).
/// `ssm_a` uses the real 35B range (`-41..-1`), since the chunked GDN cancellation bug only shows at real
/// magnitude.
fn qwen3next_batched_test_weights(
    cfg: &poot_models::qwen3next::Qwen3NextConfig,
) -> HashMap<String, Vec<f32>> {
    let h = cfg.hidden;
    let mut m: HashMap<String, Vec<f32>> = HashMap::new();
    let put = |m: &mut HashMap<String, Vec<f32>>, name: String, n: usize| {
        let v = fill(n, seed_of(&name));
        m.insert(name, v);
    };

    put(&mut m, "rope.cos".to_string(), cfg.max_pos * cfg.rotary_dim);
    put(&mut m, "rope.sin".to_string(), cfg.max_pos * cfg.rotary_dim);
    put(
        &mut m,
        "model.embed_tokens.weight".to_string(),
        cfg.vocab * h,
    );
    put(&mut m, "model.norm.weight".to_string(), h);
    put(&mut m, "lm_head.weight".to_string(), h * cfg.vocab);

    for li in 0..cfg.n_layers {
        let p = |s: &str| format!("model.layers.{li}.{s}");
        put(&mut m, p("attn_norm.weight"), h);
        put(&mut m, p("post_attention_norm.weight"), h);
        put(&mut m, p("ffn_gate_inp.weight"), h * cfg.n_experts);
        put(
            &mut m,
            p("ffn.w_in"),
            cfg.n_experts * h * 2 * cfg.expert_inter,
        );
        put(&mut m, p("ffn.w_out"), cfg.n_experts * cfg.expert_inter * h);
        put(&mut m, p("ffn_gate_shexp.weight"), h * cfg.shared_inter);
        put(&mut m, p("ffn_up_shexp.weight"), h * cfg.shared_inter);
        put(&mut m, p("ffn_down_shexp.weight"), cfg.shared_inter * h);
        put(&mut m, p("ffn_gate_inp_shexp.weight"), h);

        if cfg.is_attn_layer(li) {
            let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
            put(&mut m, p("attn_q.weight"), h * nh * 2 * hd);
            put(&mut m, p("attn_k.weight"), h * nkv * hd);
            put(&mut m, p("attn_v.weight"), h * nkv * hd);
            put(&mut m, p("attn_output.weight"), nh * hd * h);
            put(&mut m, p("attn_q_norm.weight"), hd);
            put(&mut m, p("attn_k_norm.weight"), hd);
        } else {
            let (hk, hv, hd, ck) = (
                cfg.gdn_num_k_heads,
                cfg.gdn_num_v_heads,
                cfg.gdn_head_dim,
                cfg.conv_k,
            );
            let key_dim = hk * hd;
            let value_dim = hv * hd;
            let conv_dim = 2 * key_dim + value_dim;
            put(&mut m, p("attn_qkv.weight"), h * conv_dim);
            put(&mut m, p("attn_gate.weight"), h * value_dim);
            put(&mut m, p("ssm_conv1d.weight"), ck * conv_dim);
            put(&mut m, p("ssm_beta.weight"), h * hv);
            put(&mut m, p("ssm_alpha.weight"), h * hv);
            put(&mut m, p("ssm_dt.bias"), hv);
            // Real-magnitude decay (real 35B ssm_a spans roughly [-41,-1]).
            let ssm_a: Vec<f32> = (0..hv)
                .map(|i| -41.0 + i as f32 * (40.0 / (hv - 1).max(1) as f32))
                .collect();
            m.insert(p("ssm_a"), ssm_a);
            put(&mut m, p("ssm_norm.weight"), hd);
            put(&mut m, p("ssm_out.weight"), value_dim * h);
        }
    }
    m
}

fn qwen3next_const_tensor(
    weights: &HashMap<String, Vec<f32>>,
    name: &str,
    shape: &[usize],
) -> HostTensor {
    let data = weights
        .get(name)
        .unwrap_or_else(|| panic!("qwen3next test weights missing constant: {name}"));
    assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "constant {name}: data len {} != shape {shape:?}'s product",
        data.len()
    );
    HostTensor::f32(shape.to_vec(), data.clone())
}

/// A batched decode step via [`qwen3next_decode_trace_batched_shared_pool`] must give each row exactly what an
/// independent single-sequence [`qwen3next_decode_trace`] run over that row's token stream produces, for the
/// hybrid GDN + attention mix feeding a grouped MoE FFN.
///
/// `pool_slots == batch*n_steps` with an interleaved slot map (row r, position t -> slot `t*batch+r`), so the
/// shared-pool indirection matters. Different token streams per row expose a batch-axis mixup.
#[test]
fn qwen3next_batched_decode_matches_sequential_single_seq() {
    use poot_graph_ir::{Slot, Storage};
    use poot_models::qwen3next::{
        qwen3next_decode_trace, qwen3next_decode_trace_batched_shared_pool,
    };

    let cfg = qwen3next_batched_test_config();
    let batch = 2usize;
    let cap = 6usize;
    let n_steps = 4usize; // < cap, so every row still has headroom past its last generated position
    let pool_slots = batch * n_steps; // tight: forces the shared-pool slot-map indirection to matter
    let gdn_n_slots = batch; // one GDN pool slot per physical row, no reuse in this test
    let weights = qwen3next_batched_test_weights(&cfg);
    // Different token streams per row (vocab=6).
    let row_tokens: [[usize; 4]; 2] = [[1, 3, 2, 0], [4, 5, 1, 2]];

    // --- Reference: per row, n_steps sequential qwen3next_decode_trace steps from zero state. ---
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(batch);
    for tokens in row_tokens.iter().take(batch) {
        let mut caches: Option<Vec<HostTensor>> = None;
        let mut last_logits = HostTensor::zeros(vec![1, 1, cfg.vocab]);
        for (pos, &tok) in tokens.iter().enumerate() {
            let g = qwen3next_decode_trace(&cfg, pos, cap);
            let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
            for &id in &g.inputs {
                let m = g.meta(id);
                match m.storage {
                    Storage::Slot(Slot::Token) => {
                        inputs.insert(id, HostTensor::i32(vec![], vec![tok as i32]));
                    }
                    Storage::Slot(Slot::Pos) => {
                        inputs.insert(id, HostTensor::i32(vec![], vec![pos as i32]));
                    }
                    Storage::Const => {
                        let name = m.name.as_deref().unwrap();
                        let shape = g.aval(id).shape.clone();
                        inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
                    }
                    Storage::State => {}
                    other => panic!("unexpected storage {other:?} for single-seq decode input"),
                }
            }
            match &caches {
                Some(c) => {
                    for (ci, &(sid, _)) in g.state.iter().enumerate() {
                        inputs.insert(sid, c[ci].clone());
                    }
                }
                None => {
                    for &(sid, _) in &g.state {
                        let shape = g.aval(sid).shape.clone();
                        inputs.insert(sid, HostTensor::zeros(shape));
                    }
                }
            }
            let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: HashMap<ValueId, Value> = inputs
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .expect("qwen3next decode eval");
            last_logits = logits;
            caches = Some(new_states);
        }
        assert_eq!(last_logits.shape(), vec![1, 1, cfg.vocab]);
        ref_logits.push(last_logits.as_f32().unwrap().to_vec());
    }

    // --- Batched shared-pool trace over both rows, zero-start state, interleaved slot map for attention KV. ---
    let g = qwen3next_decode_trace_batched_shared_pool(&cfg, batch, cap, pool_slots, gdn_n_slots);
    g.validate()
        .expect("qwen3next batched decode graph validates");
    let mut caches: Option<Vec<HostTensor>> = None;
    let mut batched_logits = HostTensor::zeros(vec![batch, 1, cfg.vocab]);
    // `step` is the shared time coordinate across every row of the row-major token fixture.
    #[allow(clippy::needless_range_loop)]
    for step in 0..n_steps {
        let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
        let global_slotmap: Vec<i32> = (0..batch)
            .flat_map(|row| {
                (0..cap).map(move |t| {
                    if t < n_steps {
                        (t * batch + row) as i32
                    } else {
                        0
                    }
                })
            })
            .collect();
        let mask: Vec<f32> = (0..batch)
            .flat_map(|_| (0..cap).map(|t| if t <= step { 0.0 } else { -1.0e9 }))
            .collect();
        let tokens: Vec<i32> = (0..batch).map(|row| row_tokens[row][step] as i32).collect();
        for &id in &g.inputs {
            let m = g.meta(id);
            let shape = g.aval(id).shape.clone();
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], tokens.clone()));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], vec![step as i32; batch]));
                }
                Storage::Slot(Slot::Mask) => {
                    inputs.insert(id, HostTensor::f32(vec![batch, cap], mask.clone()));
                }
                Storage::Slot(Slot::SlotMap) => {
                    // Attention shared-pool KV slot map.
                    inputs.insert(
                        id,
                        HostTensor::i32(vec![batch, cap], global_slotmap.clone()),
                    );
                }
                Storage::Slot(Slot::GdnSlotMap) => {
                    // GDN pool row map: row r's GDN pool slot = r (fixed, no reuse in this test).
                    inputs.insert(
                        id,
                        HostTensor::i32(vec![batch], (0..batch).map(|r| r as i32).collect()),
                    );
                }
                Storage::Const => {
                    let name = m.name.as_deref().unwrap();
                    inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
                }
                Storage::State => {}
                other => panic!("unexpected storage {other:?} for batched decode input"),
            }
        }
        match &caches {
            Some(c) => {
                for (ci, &(sid, _)) in g.state.iter().enumerate() {
                    inputs.insert(sid, c[ci].clone());
                }
            }
            None => {
                for &(sid, _) in &g.state {
                    let shape = g.aval(sid).shape.clone();
                    inputs.insert(sid, HostTensor::zeros(shape));
                }
            }
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("qwen3next batched decode eval");
        batched_logits = logits;
        caches = Some(new_states);
    }
    assert_eq!(batched_logits.shape(), vec![batch, 1, cfg.vocab]);

    let mut max_err = 0.0f32;
    for (row, want) in ref_logits.iter().enumerate().take(batch) {
        let got = &batched_logits.as_f32().unwrap()[row * cfg.vocab..(row + 1) * cfg.vocab];
        let err = max_abs_error(got, want);
        max_err = max_err.max(err);
        assert!(
            err < 1e-3,
            "row {row}: batched vs sequential logits max_abs={err:.3e} >= 1e-3"
        );
    }
    eprintln!(
        "qwen3next_batched_decode_matches_sequential_single_seq: batch={batch} n_steps={n_steps} \
         logits_max_abs_err={max_err:.2e}"
    );
}

/// Regression: the attention shared-pool KV map (`[batch,cap]`) and the GDN pool row map (`[batch]`) must not
/// share a `Slot` kind. `poot-eval` resolves inputs per `ValueId`, so sharing was harmless here, but
/// `poot-llm`'s `Runner::bind_decode_batched` stamps one buffer onto every id of a `Slot` kind.
///
/// The test builds that one-buffer-per-kind binder against the production tracer and checks each buffer's
/// length against every id of its kind. With `Slot::GdnSlotMap` distinct from `Slot::SlotMap`, each kind
/// has one id and the checks pass. Tagging `gdn_slot_map` with `Slot::SlotMap` again gives a length
/// mismatch (12 vs 2 at `batch=2, cap=6`) caught by the `assert_eq!` naming the collision.
#[test]
fn qwen3next_batched_decode_slotmap_kinds_bind_independently_via_kind_keyed_binder() {
    use poot_graph_ir::{Slot, Storage};
    use poot_models::qwen3next::qwen3next_decode_trace_batched_shared_pool;

    let cfg = qwen3next_batched_test_config();
    let weights = qwen3next_batched_test_weights(&cfg);
    // batch != cap and both != gdn_n_slots, so no two of {batch, batch*cap, gdn_n_slots} coincide.
    let (batch, cap, pool_slots, gdn_n_slots) = (2usize, 6usize, 8usize, 3usize);
    let g = qwen3next_decode_trace_batched_shared_pool(&cfg, batch, cap, pool_slots, gdn_n_slots);
    g.validate().expect("batched decode graph validates");

    // Distinct buffers with distinct lengths (batch*cap=12 vs batch=2); values are valid pool indices so a
    // successful bind runs the real gather/scatter.
    let attn_slot_map: Vec<i32> = (0..batch * cap).map(|i| (i % pool_slots) as i32).collect();
    let gdn_slot_map: Vec<i32> = (0..batch).map(|r| (r % gdn_n_slots) as i32).collect();

    // Mirrors `poot_llm::Runner::bind_decode_batched`: one buffer per Slot kind, inserted (reshaped to the
    // id's declared shape) under every id of that kind.
    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    for &id in &g.inputs {
        let meta = g.meta(id);
        let shape = meta.aval.shape.clone();
        match meta.storage {
            Storage::Slot(Slot::SlotMap) => {
                assert_eq!(
                    shape.iter().product::<usize>(),
                    attn_slot_map.len(),
                    "kind-keyed binder: Slot::SlotMap id v{id} (shape {shape:?}) does not match the \
                     single attention-map buffer this binder supplies for the WHOLE Slot::SlotMap kind - \
                     this is the exact collision card 188 Increment 4 flagged and Increment 5 fixes by \
                     giving the GDN row map its own Slot::GdnSlotMap kind"
                );
                inputs.insert(id, HostTensor::i32(shape, attn_slot_map.clone()));
            }
            Storage::Slot(Slot::GdnSlotMap) => {
                assert_eq!(
                    shape.iter().product::<usize>(),
                    gdn_slot_map.len(),
                    "kind-keyed binder: Slot::GdnSlotMap id v{id} (shape {shape:?}) does not match the \
                     GDN pool row map buffer"
                );
                inputs.insert(id, HostTensor::i32(shape, gdn_slot_map.clone()));
            }
            Storage::Slot(Slot::Token) => {
                inputs.insert(id, HostTensor::i32(shape, vec![0; batch]));
            }
            Storage::Slot(Slot::Activation) => {
                unreachable!("standalone Activation slots are not model decode inputs")
            }
            Storage::Slot(Slot::Pos) => {
                inputs.insert(id, HostTensor::i32(shape, vec![0; batch]));
            }
            Storage::Slot(Slot::MropePosition) => {
                unreachable!("qwen3next batched decode has no mRoPE position slot")
            }
            Storage::Slot(Slot::SeqLen) => {
                unreachable!("qwen3next_decode_trace_batched_shared_pool has no SeqLen slot")
            }
            Storage::Slot(Slot::Mask) => {
                let n: usize = shape.iter().product();
                inputs.insert(id, HostTensor::f32(shape, vec![0.0; n]));
            }
            Storage::Slot(Slot::TokenEmbed) => {
                unreachable!("qwen3next_decode_trace_batched_shared_pool uses a dense embed gather")
            }
            Storage::Slot(Slot::LoraIdx) => {
                unreachable!(
                    "LoraIdx binds only on the batched shared-pool LoRA decode path (spec 248 Phase 2)"
                )
            }
            Storage::Slot(Slot::ExpertPoolMap) => {
                unreachable!(
                    "ExpertPoolMap binds only on the pooled-MoE decode path (spec 266 phase 1), and is \
                     resolved per value NAME (Builder::slot_named), never per slot kind"
                )
            }
            Storage::Const => {
                let name = meta.name.as_deref().unwrap();
                inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
            }
            Storage::Computed(computed) => {
                inputs.insert(id, HostTensor::f32(computed.shape(), computed.values_f32()));
            }
            Storage::State => {
                inputs.insert(id, HostTensor::zeros(shape));
            }
            Storage::Slot(Slot::Sampler) => {
                unreachable!("Sampler binds only on a graph with a card-551b-appended suffix")
            }
            Storage::Device => unreachable!("device value in input set"),
        }
    }
    let (logits, _) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .expect("kind-keyed-bound batched decode eval");
    assert_eq!(logits.shape(), vec![batch, 1, cfg.vocab]);
}

/// `qwen3next_decode_trace_batched_shared_pool` at `batch=1`/`gdn_n_slots=1` must match the single-sequence
/// [`qwen3next_decode_trace`] step for step. Reports the observed `max_abs_err` instead of asserting bit
/// equality: the batched attention goes through `attention_masked` (softmax over the full `cap` with an
/// additive mask) plus pool indirection, while single-seq uses `attention` over an exact `pos+1` prefix
/// slice. Equivalent mathematically, not bitwise; tolerance `1e-3` as in
/// `gemma4_batched_decode_matches_sequential_single_seq`.
#[test]
fn qwen3next_batched_decode_n_slots_1_matches_single_seq_trace() {
    use poot_graph_ir::{Slot, Storage};
    use poot_models::qwen3next::{
        qwen3next_decode_trace, qwen3next_decode_trace_batched_shared_pool,
    };

    let cfg = qwen3next_batched_test_config();
    let batch = 1usize;
    let cap = 6usize;
    let n_steps = 4usize;
    let pool_slots = cap; // batch=1: exactly enough for one row's whole run, no indirection needed
    let gdn_n_slots = 1usize;
    let weights = qwen3next_batched_test_weights(&cfg);
    let tokens = [2usize, 0, 5, 3];

    // --- Reference: qwen3next_decode_trace, sequential. ---
    let mut ref_caches: Option<Vec<HostTensor>> = None;
    let mut ref_logits_by_step: Vec<Vec<f32>> = Vec::with_capacity(n_steps);
    for (pos, &tok) in tokens.iter().enumerate() {
        let g = qwen3next_decode_trace(&cfg, pos, cap);
        let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, HostTensor::i32(vec![], vec![tok as i32]));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, HostTensor::i32(vec![], vec![pos as i32]));
                }
                Storage::Const => {
                    let name = m.name.as_deref().unwrap();
                    let shape = g.aval(id).shape.clone();
                    inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
                }
                Storage::State => {}
                other => panic!("unexpected storage {other:?}"),
            }
        }
        match &ref_caches {
            Some(c) => {
                for (ci, &(sid, _)) in g.state.iter().enumerate() {
                    inputs.insert(sid, c[ci].clone());
                }
            }
            None => {
                for &(sid, _) in &g.state {
                    let shape = g.aval(sid).shape.clone();
                    inputs.insert(sid, HostTensor::zeros(shape));
                }
            }
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("qwen3next decode eval");
        ref_logits_by_step.push(logits.as_f32().unwrap().to_vec());
        ref_caches = Some(new_states);
    }

    // --- Under test: batched tracer at batch=1, one active row, slot 0. ---
    let g = qwen3next_decode_trace_batched_shared_pool(&cfg, batch, cap, pool_slots, gdn_n_slots);
    g.validate()
        .expect("qwen3next batched decode graph (batch=1) validates");
    let mut caches: Option<Vec<HostTensor>> = None;
    let mut max_err = 0.0f32;
    for (step, &tok) in tokens.iter().enumerate() {
        let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
        let slotmap: Vec<i32> = (0..cap)
            .map(|t| if t <= step { t as i32 } else { 0 })
            .collect();
        let mask: Vec<f32> = (0..cap)
            .map(|t| if t <= step { 0.0 } else { -1.0e9 })
            .collect();
        for &id in &g.inputs {
            let m = g.meta(id);
            let shape = g.aval(id).shape.clone();
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], vec![tok as i32]));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], vec![step as i32]));
                }
                Storage::Slot(Slot::Mask) => {
                    inputs.insert(id, HostTensor::f32(vec![batch, cap], mask.clone()));
                }
                Storage::Slot(Slot::SlotMap) => {
                    inputs.insert(id, HostTensor::i32(vec![batch, cap], slotmap.clone()));
                }
                Storage::Slot(Slot::GdnSlotMap) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], vec![0]));
                }
                Storage::Const => {
                    let name = m.name.as_deref().unwrap();
                    inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
                }
                Storage::State => {}
                other => panic!("unexpected storage {other:?}"),
            }
        }
        match &caches {
            Some(c) => {
                for (ci, &(sid, _)) in g.state.iter().enumerate() {
                    inputs.insert(sid, c[ci].clone());
                }
            }
            None => {
                for &(sid, _) in &g.state {
                    let shape = g.aval(sid).shape.clone();
                    inputs.insert(sid, HostTensor::zeros(shape));
                }
            }
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("qwen3next batched decode eval (batch=1)");
        let err = max_abs_error(logits.as_f32().unwrap(), &ref_logits_by_step[step]);
        max_err = max_err.max(err);
        assert!(
            err < 1e-3,
            "step {step}: batch=1 batched vs single-seq logits max_abs={err:.3e} >= 1e-3"
        );
        caches = Some(new_states);
    }
    eprintln!(
        "qwen3next_batched_decode_n_slots_1_matches_single_seq_trace: n_steps={n_steps} \
         logits_max_abs_err={max_err:.2e}"
    );
}

/// Chunked slot-addressed prefill ([`qwen3next_gdn_prefill_into_slot`]) followed by pooled decode
/// ([`qwen3next_gdn_decode_batched_pool`]) must match a reference that runs the same bare chunked prefill and
/// then sequential `qwen3next_gdn_block` decode steps. `L=8, chunk=4` (2 chunk boundaries), real-magnitude
/// `ssm_a` in `[-41,-1]`. Three other pool slots hold distinct nonzero garbage and must stay byte-unchanged
/// across the admission write and every decode step.
#[test]
fn qwen3next_gdn_decode_batched_pool_continues_correctly_after_chunked_prefill_admission() {
    use poot_models::qwen3next::{
        GdnHeadOrder, qwen3next_gdn_block, qwen3next_gdn_decode_batched_pool,
        qwen3next_gdn_prefill_block, qwen3next_gdn_prefill_into_slot,
    };

    let (h, hk, hv, d, ck, l, chunk) = (6usize, 2usize, 6usize, 4usize, 4usize, 8usize, 4usize);
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6;
    let n_slots = 4usize;
    let target_slot = 2usize;
    let n_decode = 3usize;

    let xd = fill(l * h, 501);
    let w_qkv_d = fill(h * conv_dim, 502);
    let w_gate_d = fill(h * value_dim, 503);
    let w_conv_d = fill(ck * conv_dim, 504);
    let w_beta_d = fill(h * hv, 505);
    let w_alpha_d = fill(h * hv, 506);
    let dt_bias_d = fill(hv, 507);
    // Real-magnitude decay (real 35B ssm_a spans roughly [-41,-1]); the chunked cumsum/exp cancellation
    // only shows at this magnitude.
    let ssm_a_d: Vec<f32> = (0..hv)
        .map(|i| -41.0 + i as f32 * (40.0 / (hv - 1) as f32))
        .collect();
    let norm_w_d = fill(d, 508);
    let w_out_d = fill(value_dim * h, 509);
    // decode-step tokens: [1,1,H] each.
    let decode_x_d: Vec<Vec<f32>> = (0..n_decode).map(|i| fill(h, 600 + i as u64)).collect();

    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c];
    let mut tril_strict = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }

    // ---------- Reference: bare prefill, then n_decode bare sequential decode steps. ----------
    let rb = Builder::new();
    let rx = rb.constant("x", TensorType::f32(vec![1, l, h]));
    let rw_qkv = rb.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let rw_gate = rb.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let rw_conv = rb.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let rw_beta = rb.constant("w_beta", TensorType::f32(vec![h, hv]));
    let rw_alpha = rb.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let rdt_bias = rb.constant("dt_bias", TensorType::f32(vec![hv]));
    let rssm_a = rb.constant("ssm_a", TensorType::f32(vec![hv]));
    let rnorm_w = rb.constant("norm_w", TensorType::f32(vec![d]));
    let rw_out = rb.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let rs_in = rb.constant("s_in", TensorType::f32(vec![1, hv, d, d]));
    let rtril_incl = rb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let rtril_strict = rb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (_ref_prefill_cur, ref_cc0, ref_s0) = qwen3next_gdn_prefill_block(
        &rb,
        rx,
        rw_qkv,
        rw_gate,
        rw_conv,
        rw_beta,
        rw_alpha,
        rdt_bias,
        rssm_a,
        rnorm_w,
        rw_out,
        rs_in,
        rtril_incl,
        rtril_strict,
        hk,
        hv,
        d,
        ck,
        chunk,
        eps,
        GdnHeadOrder::Tiled,
    );
    let rg_prefill = rb.finish_with_state(_ref_prefill_cur, &[]);
    let mut rinp = HashMap::new();
    rinp.insert(rx.id, HostTensor::f32(vec![1, l, h], xd.clone()));
    rinp.insert(
        rw_qkv.id,
        HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
    );
    rinp.insert(
        rw_gate.id,
        HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
    );
    rinp.insert(
        rw_conv.id,
        HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
    );
    rinp.insert(rw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
    rinp.insert(rw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
    rinp.insert(rdt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
    rinp.insert(rssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
    rinp.insert(rnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
    rinp.insert(
        rw_out.id,
        HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
    );
    rinp.insert(
        rs_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    rinp.insert(
        rtril_incl.id,
        HostTensor::f32(vec![1, 1, c, c], tril_incl.clone()),
    );
    rinp.insert(
        rtril_strict.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict.clone()),
    );
    let ref_env = (|| -> Result<Vec<Option<HostTensor>>, EvalError> {
        let values: HashMap<ValueId, Value> = rinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let environment = crate::eval(
            &rg_prefill,
            &values,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )?
        .environment
        .expect("keep_environment was set");
        environment
            .into_iter()
            .map(|slot| slot.map(Value::into_host).transpose())
            .collect()
    })()
    .expect("reference prefill eval");
    let mut ref_conv = ref_env[ref_cc0.id]
        .clone()
        .expect("conv_cache_out computed");
    let mut ref_ssm = ref_env[ref_s0.id].clone().expect("s_out computed");

    let mut ref_final_cur = HostTensor::zeros(vec![1, 1, h]);
    for x_d in decode_x_d.iter().take(n_decode) {
        let db = Builder::new();
        let dx = db.constant("x", TensorType::f32(vec![1, 1, h]));
        let dw_qkv = db.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
        let dw_gate = db.constant("w_gate", TensorType::f32(vec![h, value_dim]));
        let dw_conv = db.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
        let dw_beta = db.constant("w_beta", TensorType::f32(vec![h, hv]));
        let dw_alpha = db.constant("w_alpha", TensorType::f32(vec![h, hv]));
        let ddt_bias = db.constant("dt_bias", TensorType::f32(vec![hv]));
        let dssm_a = db.constant("ssm_a", TensorType::f32(vec![hv]));
        let dnorm_w = db.constant("norm_w", TensorType::f32(vec![d]));
        let dw_out = db.constant("w_out", TensorType::f32(vec![value_dim, h]));
        let dconv_in = db.constant("conv_in", TensorType::f32(vec![1, ck - 1, conv_dim]));
        let ds_in = db.constant("s_in", TensorType::f32(vec![1, hv, d, d]));
        let (cur, cc_out, s_out) = qwen3next_gdn_block(
            &db,
            dx,
            dw_qkv,
            dw_gate,
            dw_conv,
            dw_beta,
            dw_alpha,
            ddt_bias,
            dssm_a,
            dnorm_w,
            dw_out,
            dconv_in,
            ds_in,
            hk,
            hv,
            d,
            ck,
            eps,
            GdnHeadOrder::Tiled,
        );
        let dg = db.finish_with_state(cur, &[]);
        let mut dinp = HashMap::new();
        dinp.insert(dx.id, HostTensor::f32(vec![1, 1, h], x_d.clone()));
        dinp.insert(
            dw_qkv.id,
            HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
        );
        dinp.insert(
            dw_gate.id,
            HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
        );
        dinp.insert(
            dw_conv.id,
            HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
        );
        dinp.insert(dw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
        dinp.insert(dw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
        dinp.insert(ddt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
        dinp.insert(dssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
        dinp.insert(dnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
        dinp.insert(
            dw_out.id,
            HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
        );
        dinp.insert(dconv_in.id, ref_conv.clone());
        dinp.insert(ds_in.id, ref_ssm.clone());
        let d_env = (|| -> Result<Vec<Option<HostTensor>>, EvalError> {
            let values: HashMap<ValueId, Value> = dinp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let environment = crate::eval(
                &dg,
                &values,
                EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            )?
            .environment
            .expect("keep_environment was set");
            environment
                .into_iter()
                .map(|slot| slot.map(Value::into_host).transpose())
                .collect()
        })()
        .expect("reference decode eval");
        ref_final_cur = d_env[cur.id].clone().expect("cur computed");
        ref_conv = d_env[cc_out.id].clone().expect("conv_cache_out computed");
        ref_ssm = d_env[s_out.id].clone().expect("s_out computed");
    }

    // ---------- Under test: admission via qwen3next_gdn_prefill_into_slot into a 4-slot pool (slots 0,1,3
    // hold distinct nonzero garbage), then n_decode steps via qwen3next_gdn_decode_batched_pool at
    // gdn_slot_map=[target_slot]. ----------
    let conv_row = (ck - 1) * conv_dim;
    let ssm_row = hv * d * d;
    let mut conv_pool_d = vec![0.0f32; n_slots * conv_row];
    let mut ssm_pool_d = vec![0.0f32; n_slots * ssm_row];
    for slot in 0..n_slots {
        if slot == target_slot {
            continue;
        }
        let cv = fill(conv_row, 700 + slot as u64);
        conv_pool_d[slot * conv_row..(slot + 1) * conv_row].copy_from_slice(&cv);
        let sv = fill(ssm_row, 800 + slot as u64);
        ssm_pool_d[slot * ssm_row..(slot + 1) * ssm_row].copy_from_slice(&sv);
    }
    let other_slots_before_admission = (conv_pool_d.clone(), ssm_pool_d.clone());

    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, h]));
    let pw_qkv = pb.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let pw_gate = pb.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let pw_conv = pb.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let pw_beta = pb.constant("w_beta", TensorType::f32(vec![h, hv]));
    let pw_alpha = pb.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let pdt_bias = pb.constant("dt_bias", TensorType::f32(vec![hv]));
    let pssm_a = pb.constant("ssm_a", TensorType::f32(vec![hv]));
    let pnorm_w = pb.constant("norm_w", TensorType::f32(vec![d]));
    let pw_out = pb.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let ps_in = pb.constant("s_in", TensorType::f32(vec![1, hv, d, d]));
    let ptril_incl = pb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let ptril_strict = pb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let pconv_pool_in = pb.state_input(
        "conv_pool",
        TensorType::f32(vec![n_slots, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let pssm_pool_in = pb.state_input(
        "ssm_pool",
        TensorType::f32(vec![n_slots, hv, d, d]),
        StateRole::Recurrent,
    );
    let pslot = pb.constant("slot", TensorType::scalar(DType::I32));
    let (_p_cur, pconv_pool_out, pssm_pool_out) = qwen3next_gdn_prefill_into_slot(
        &pb,
        px,
        pw_qkv,
        pw_gate,
        pw_conv,
        pw_beta,
        pw_alpha,
        pdt_bias,
        pssm_a,
        pnorm_w,
        pw_out,
        ps_in,
        ptril_incl,
        ptril_strict,
        pconv_pool_in,
        pssm_pool_in,
        pslot,
        hk,
        hv,
        d,
        ck,
        chunk,
        eps,
        GdnHeadOrder::Tiled,
    );
    let pg = pb.finish_with_state(
        _p_cur,
        &[
            (pconv_pool_in, pconv_pool_out),
            (pssm_pool_in, pssm_pool_out),
        ],
    );
    let mut pinp = HashMap::new();
    pinp.insert(px.id, HostTensor::f32(vec![1, l, h], xd));
    pinp.insert(
        pw_qkv.id,
        HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
    );
    pinp.insert(
        pw_gate.id,
        HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
    );
    pinp.insert(
        pw_conv.id,
        HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
    );
    pinp.insert(pw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
    pinp.insert(pw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
    pinp.insert(pdt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
    pinp.insert(pssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
    pinp.insert(pnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
    pinp.insert(
        pw_out.id,
        HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
    );
    pinp.insert(
        ps_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    pinp.insert(ptril_incl.id, HostTensor::f32(vec![1, 1, c, c], tril_incl));
    pinp.insert(
        ptril_strict.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict),
    );
    pinp.insert(
        pconv_pool_in.id,
        HostTensor::f32(vec![n_slots, ck - 1, conv_dim], conv_pool_d),
    );
    pinp.insert(
        pssm_pool_in.id,
        HostTensor::f32(vec![n_slots, hv, d, d], ssm_pool_d),
    );
    pinp.insert(pslot.id, HostTensor::i32(vec![], vec![target_slot as i32]));
    let (_p_cur_out, p_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = pinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&pg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .expect("admission eval");
    let mut conv_pool = p_states[0].clone();
    let mut ssm_pool = p_states[1].clone();

    // n_decode steps, batch=1, gdn_slot_map=[target_slot].
    let mut sut_final_cur = HostTensor::zeros(vec![1, 1, h]);
    for (step, x_d) in decode_x_d.iter().enumerate().take(n_decode) {
        let db = Builder::new();
        let dx = db.constant("x", TensorType::f32(vec![1, 1, h]));
        let dw_qkv = db.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
        let dw_gate = db.constant("w_gate", TensorType::f32(vec![h, value_dim]));
        let dw_conv = db.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
        let dw_beta = db.constant("w_beta", TensorType::f32(vec![h, hv]));
        let dw_alpha = db.constant("w_alpha", TensorType::f32(vec![h, hv]));
        let ddt_bias = db.constant("dt_bias", TensorType::f32(vec![hv]));
        let dssm_a = db.constant("ssm_a", TensorType::f32(vec![hv]));
        let dnorm_w = db.constant("norm_w", TensorType::f32(vec![d]));
        let dw_out = db.constant("w_out", TensorType::f32(vec![value_dim, h]));
        let dconv_pool_in = db.state_input(
            "conv_pool",
            TensorType::f32(vec![n_slots, ck - 1, conv_dim]),
            StateRole::Recurrent,
        );
        let dssm_pool_in = db.state_input(
            "ssm_pool",
            TensorType::f32(vec![n_slots, hv, d, d]),
            StateRole::Recurrent,
        );
        let dgdn_slot_map = db.constant("gdn_slot_map", TensorType::new(vec![1], DType::I32));
        let (cur, conv_pool_out, ssm_pool_out) = qwen3next_gdn_decode_batched_pool(
            &db,
            dx,
            dw_qkv,
            dw_gate,
            dw_conv,
            dw_beta,
            dw_alpha,
            ddt_bias,
            dssm_a,
            dnorm_w,
            dw_out,
            dconv_pool_in,
            dssm_pool_in,
            dgdn_slot_map,
            hk,
            hv,
            d,
            ck,
            eps,
            1,
            GdnHeadOrder::Tiled,
        );
        let dg = db.finish_with_state(
            cur,
            &[(dconv_pool_in, conv_pool_out), (dssm_pool_in, ssm_pool_out)],
        );
        let mut dinp = HashMap::new();
        dinp.insert(dx.id, HostTensor::f32(vec![1, 1, h], x_d.clone()));
        dinp.insert(
            dw_qkv.id,
            HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
        );
        dinp.insert(
            dw_gate.id,
            HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
        );
        dinp.insert(
            dw_conv.id,
            HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
        );
        dinp.insert(dw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
        dinp.insert(dw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
        dinp.insert(ddt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
        dinp.insert(dssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
        dinp.insert(dnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
        dinp.insert(
            dw_out.id,
            HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
        );
        dinp.insert(dconv_pool_in.id, conv_pool.clone());
        dinp.insert(dssm_pool_in.id, ssm_pool.clone());
        dinp.insert(
            dgdn_slot_map.id,
            HostTensor::i32(vec![1], vec![target_slot as i32]),
        );
        let (_sut_cur, sut_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = dinp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("pooled decode eval");
        sut_final_cur = _sut_cur;
        conv_pool = sut_states[0].clone();
        ssm_pool = sut_states[1].clone();

        // Isolation must hold at every decode step.
        for slot in 0..n_slots {
            if slot == target_slot {
                continue;
            }
            let cbefore = &other_slots_before_admission.0[slot * conv_row..(slot + 1) * conv_row];
            let cafter = &conv_pool.as_f32().unwrap()[slot * conv_row..(slot + 1) * conv_row];
            let sbefore = &other_slots_before_admission.1[slot * ssm_row..(slot + 1) * ssm_row];
            let safter = &ssm_pool.as_f32().unwrap()[slot * ssm_row..(slot + 1) * ssm_row];
            assert_eq!(
                max_abs_error(cafter, cbefore),
                0.0,
                "decode step {step}: slot {slot} conv_cache must stay byte-unchanged"
            );
            assert_eq!(
                max_abs_error(safter, sbefore),
                0.0,
                "decode step {step}: slot {slot} ssm_state must stay byte-unchanged"
            );
        }
    }

    let cur_err = max_abs_error(
        sut_final_cur.as_f32().unwrap(),
        ref_final_cur.as_f32().unwrap(),
    );
    let conv_err = max_abs_error(
        &conv_pool.as_f32().unwrap()[target_slot * conv_row..(target_slot + 1) * conv_row],
        ref_conv.as_f32().unwrap(),
    );
    let ssm_err = max_abs_error(
        &ssm_pool.as_f32().unwrap()[target_slot * ssm_row..(target_slot + 1) * ssm_row],
        ref_ssm.as_f32().unwrap(),
    );
    eprintln!(
        "qwen3next_gdn_decode_batched_pool_continues_correctly_after_chunked_prefill_admission: \
         cur_max_abs_err={cur_err:.2e} conv_cache_max_abs_err={conv_err:.2e} \
         ssm_state_max_abs_err={ssm_err:.2e} (real-magnitude ssm_a range=[{:.1},{:.1}], L={l} chunk={chunk} \
         => {} chunk boundaries, n_decode={n_decode})",
        ssm_a_d[0],
        ssm_a_d[hv - 1],
        l.div_ceil(chunk) - 1,
    );
    assert!(cur_err < 1e-4, "final decode cur mismatch: {cur_err:.3e}");
    assert!(conv_err < 1e-4, "final conv_cache mismatch: {conv_err:.3e}");
    assert!(ssm_err < 1e-4, "final ssm_state mismatch: {ssm_err:.3e}");
}

/// Admit/evict mid-run: sequence A occupies batch row 0 for a short life, is evicted, and sequence C is admitted
/// into the same row; sequence B occupies row 1 throughout. Every row's output must match its own independent
/// [`qwen3next_decode_trace`] run, including B's, so cross-row isolation holds through admit/evict on the
/// other row.
///
/// Attention KV needs no clearing on admission (stale positions are masked out of the softmax), but the GDN
/// pool row must be zeroed before C's first step since recurrent state has no mask downstream. Skipping the zero would leak A's state into C's first decode step.
#[test]
fn qwen3next_batched_decode_admit_evict_reuses_freed_slot() {
    use poot_graph_ir::{Slot, Storage};
    use poot_models::qwen3next::{
        qwen3next_decode_trace, qwen3next_decode_trace_batched_shared_pool,
    };

    let cfg = qwen3next_batched_test_config();
    let batch = 2usize;
    let cap = 6usize;
    let pool_slots = batch * cap; // simple row-contiguous scheme: row r's KV lives in pool rows [r*cap,(r+1)*cap)
    let gdn_n_slots = batch; // gdn_slot_map is fixed = physical row index (identity), no GDN-side reuse map
    let weights = qwen3next_batched_test_weights(&cfg);

    let a_tokens = [1usize, 3, 2]; // row 0, steps [0,3)
    let c_tokens = [4usize, 0, 5]; // row 0, steps [3,6) - a DIFFERENT sequence reusing row 0's slot
    let b_tokens = [2usize, 5, 1, 0, 3, 4]; // row 1, steps [0,6) - never moves
    let n_steps = 6usize;
    let switch_step = 3usize; // A's last step is 2; C's first step is 3

    // row 0's (token, local_pos) per global step: A while step<switch_step, else C (local pos resets).
    let row0_at = |step: usize| -> (usize, usize) {
        if step < switch_step {
            (a_tokens[step], step)
        } else {
            (c_tokens[step - switch_step], step - switch_step)
        }
    };
    let row1_at = |step: usize| -> (usize, usize) { (b_tokens[step], step) };

    // --- References: B and C run independently through qwen3next_decode_trace from zero state. ---
    let run_reference = |tokens: &[usize]| -> Vec<f32> {
        let mut caches: Option<Vec<HostTensor>> = None;
        let mut last_logits = HostTensor::zeros(vec![1, 1, cfg.vocab]);
        for (pos, &tok) in tokens.iter().enumerate() {
            let g = qwen3next_decode_trace(&cfg, pos, cap);
            let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
            for &id in &g.inputs {
                let m = g.meta(id);
                match m.storage {
                    Storage::Slot(Slot::Token) => {
                        inputs.insert(id, HostTensor::i32(vec![], vec![tok as i32]));
                    }
                    Storage::Slot(Slot::Pos) => {
                        inputs.insert(id, HostTensor::i32(vec![], vec![pos as i32]));
                    }
                    Storage::Const => {
                        let name = m.name.as_deref().unwrap();
                        let shape = g.aval(id).shape.clone();
                        inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
                    }
                    Storage::State => {}
                    other => panic!("unexpected storage {other:?}"),
                }
            }
            match &caches {
                Some(c) => {
                    for (ci, &(sid, _)) in g.state.iter().enumerate() {
                        inputs.insert(sid, c[ci].clone());
                    }
                }
                None => {
                    for &(sid, _) in &g.state {
                        let shape = g.aval(sid).shape.clone();
                        inputs.insert(sid, HostTensor::zeros(shape));
                    }
                }
            }
            let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: HashMap<ValueId, Value> = inputs
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .expect("qwen3next reference decode eval");
            last_logits = logits;
            caches = Some(new_states);
        }
        last_logits.as_f32().unwrap().to_vec()
    };
    let ref_b = run_reference(&b_tokens);
    let ref_c = run_reference(&c_tokens);

    // g.state indices of the GDN pool (conv_pool, ssm_pool) pairs: per layer, GDN layers push (conv_pool,
    // ssm_pool), attention layers push (k, v).
    let gdn_state_indices: Vec<usize> = (0..cfg.n_layers)
        .filter(|&li| !cfg.is_attn_layer(li))
        .flat_map(|li| [2 * li, 2 * li + 1])
        .collect();

    let zero_pool_row = |t: &HostTensor, n_slots: usize, row: usize| -> HostTensor {
        let mut data: Vec<f32> = t.as_f32().unwrap().to_vec();
        let row_len = data.len() / n_slots;
        let start = row * row_len;
        for v in &mut data[start..start + row_len] {
            *v = 0.0;
        }
        HostTensor::f32(t.shape().to_vec(), data)
    };

    // --- Batched shared-pool trace: row 0 = A then C (slot reuse at switch_step), row 1 = B throughout. ---
    let g = qwen3next_decode_trace_batched_shared_pool(&cfg, batch, cap, pool_slots, gdn_n_slots);
    g.validate()
        .expect("qwen3next batched decode graph (admit/evict) validates");
    let mut caches: Option<Vec<HostTensor>> = None;
    let mut batched_logits = HostTensor::zeros(vec![batch, 1, cfg.vocab]);
    for step in 0..n_steps {
        let (tok0, pos0) = row0_at(step);
        let (tok1, pos1) = row1_at(step);
        let tokens_v = vec![tok0 as i32, tok1 as i32];
        let pos_v = vec![pos0 as i32, pos1 as i32];
        // Row-contiguous slot map: row r's position t lands at pool row r*cap+t whichever sequence occupies row
        // r (safe because of the mask below).
        let slotmap: Vec<i32> = (0..batch)
            .flat_map(|row| (0..cap).map(move |t| (row * cap + t) as i32))
            .collect();
        let mask: Vec<f32> = [pos0, pos1]
            .iter()
            .flat_map(|&p| (0..cap).map(move |t| if t <= p { 0.0 } else { -1.0e9 }))
            .collect();

        // Admission: right before C's first step, zero row 0 of the GDN pool (conv_cache and ssm_state); the
        // recurrent state has no mask.
        if step == switch_step
            && let Some(c) = &mut caches
        {
            for &ci in &gdn_state_indices {
                c[ci] = zero_pool_row(&c[ci], gdn_n_slots, 0);
            }
        }

        let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
        for &id in &g.inputs {
            let m = g.meta(id);
            let shape = g.aval(id).shape.clone();
            match m.storage {
                Storage::Slot(Slot::Token) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], tokens_v.clone()));
                }
                Storage::Slot(Slot::Pos) => {
                    inputs.insert(id, HostTensor::i32(vec![batch], pos_v.clone()));
                }
                Storage::Slot(Slot::Mask) => {
                    inputs.insert(id, HostTensor::f32(vec![batch, cap], mask.clone()));
                }
                Storage::Slot(Slot::SlotMap) => {
                    inputs.insert(id, HostTensor::i32(vec![batch, cap], slotmap.clone()));
                }
                Storage::Slot(Slot::GdnSlotMap) => {
                    // GDN pool row map: identity; C reuses row 0's slot (the zeroing above makes reuse safe).
                    inputs.insert(
                        id,
                        HostTensor::i32(vec![batch], (0..batch).map(|r| r as i32).collect()),
                    );
                }
                Storage::Const => {
                    let name = m.name.as_deref().unwrap();
                    inputs.insert(id, qwen3next_const_tensor(&weights, name, &shape));
                }
                Storage::State => {}
                other => panic!("unexpected storage {other:?}"),
            }
        }
        match &caches {
            Some(c) => {
                for (ci, &(sid, _)) in g.state.iter().enumerate() {
                    inputs.insert(sid, c[ci].clone());
                }
            }
            None => {
                for &(sid, _) in &g.state {
                    let shape = g.aval(sid).shape.clone();
                    inputs.insert(sid, HostTensor::zeros(shape));
                }
            }
        }
        let (logits, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .expect("qwen3next batched decode eval (admit/evict)");
        batched_logits = logits;
        caches = Some(new_states);
    }
    assert_eq!(batched_logits.shape(), vec![batch, 1, cfg.vocab]);

    let got_c = &batched_logits.as_f32().unwrap()[..cfg.vocab];
    let got_b = &batched_logits.as_f32().unwrap()[cfg.vocab..2 * cfg.vocab];
    let err_c = max_abs_error(got_c, &ref_c);
    let err_b = max_abs_error(got_b, &ref_b);
    eprintln!(
        "qwen3next_batched_decode_admit_evict_reuses_freed_slot: row0(A->C)_max_abs_err={err_c:.2e} \
         row1(B, unmoved)_max_abs_err={err_b:.2e}"
    );
    assert!(
        err_c < 1e-3,
        "row 0 (C, reused A's freed slot) vs its own independent reference: max_abs={err_c:.3e} >= 1e-3"
    );
    assert!(
        err_b < 1e-3,
        "row 1 (B, never moved) vs its own independent reference: max_abs={err_b:.3e} >= 1e-3 - \
         cross-row isolation broke across the admit/evict event on row 0"
    );
}

/// [`qwen3next_zero_gdn_slot_trace`] must zero only the target slot's GDN state (every GDN layer) and leave
/// every other GDN slot and the whole attention KV pool byte-unchanged (the passthrough Reshape must preserve
/// content).
#[test]
fn qwen3next_zero_gdn_slot_trace_touches_only_target_slot_and_leaves_kv_unchanged() {
    use poot_models::qwen3next::qwen3next_zero_gdn_slot_trace;

    let cfg = qwen3next_batched_test_config();
    let pool_slots = 6usize;
    let gdn_n_slots = 4usize;
    let target_slot = 2usize;

    let g = qwen3next_zero_gdn_slot_trace(&cfg, pool_slots, gdn_n_slots, target_slot);
    g.validate().expect("zero-gdn-slot graph validates");

    // Distinct nonzero data in every state entry (attention KV rows and every GDN slot, including the target)
    // so zeroing everything or the wrong slot is caught.
    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    let mut before: HashMap<ValueId, Vec<f32>> = HashMap::new();
    for (i, &(sid, _)) in g.state.iter().enumerate() {
        let shape = g.aval(sid).shape.clone();
        let n: usize = shape.iter().product();
        let data = fill(n, 900 + i as u64);
        before.insert(sid, data.clone());
        inputs.insert(sid, HostTensor::f32(shape, data));
    }

    let (_out, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .expect("zero-gdn-slot eval");
    assert_eq!(new_states.len(), g.state.len());

    let ck = cfg.conv_k;
    let (hk, hv, hd) = (cfg.gdn_num_k_heads, cfg.gdn_num_v_heads, cfg.gdn_head_dim);
    let key_dim = hk * hd;
    let value_dim = hv * hd;
    let conv_dim = 2 * key_dim + value_dim;
    let conv_row = (ck - 1) * conv_dim;
    let ssm_row = hv * hd * hd;

    let mut checked_gdn_layer = false;
    let mut checked_kv_layer = false;
    for (i, &(sid, _)) in g.state.iter().enumerate() {
        let name = g.meta(sid).name.clone().unwrap();
        let shape = g.aval(sid).shape.clone();
        let before_data = &before[&sid];
        let after_data = new_states[i].as_f32().unwrap();
        if name.ends_with("kv.k_cache") || name.ends_with("kv.v_cache") {
            // Attention KV pool: byte-unchanged in every row.
            assert_eq!(
                after_data,
                before_data.as_slice(),
                "{name}: attention KV pool must pass through byte-unchanged"
            );
            checked_kv_layer = true;
        } else if name.ends_with("gdn.conv_cache") {
            for slot in 0..gdn_n_slots {
                let want_zero = slot == target_slot;
                let got = &after_data[slot * conv_row..(slot + 1) * conv_row];
                if want_zero {
                    assert!(
                        got.iter().all(|&x| x == 0.0),
                        "{name} slot {slot} (target): conv_cache must be all-zero, got {got:?}"
                    );
                } else {
                    let want = &before_data[slot * conv_row..(slot + 1) * conv_row];
                    assert_eq!(
                        got, want,
                        "{name} slot {slot} (not target): conv_cache must be byte-unchanged"
                    );
                }
            }
            checked_gdn_layer = true;
        } else if name.ends_with("gdn.ssm_state") {
            for slot in 0..gdn_n_slots {
                let want_zero = slot == target_slot;
                let got = &after_data[slot * ssm_row..(slot + 1) * ssm_row];
                if want_zero {
                    assert!(
                        got.iter().all(|&x| x == 0.0),
                        "{name} slot {slot} (target): ssm_state must be all-zero, got {got:?}"
                    );
                } else {
                    let want = &before_data[slot * ssm_row..(slot + 1) * ssm_row];
                    assert_eq!(
                        got, want,
                        "{name} slot {slot} (not target): ssm_state must be byte-unchanged"
                    );
                }
            }
        } else {
            panic!("unexpected state entry name {name} shape {shape:?}");
        }
    }
    assert!(
        checked_gdn_layer,
        "must have at least one GDN layer to check"
    );
    assert!(
        checked_kv_layer,
        "must have at least one attention layer to check"
    );
}

/// Diagnostic (card 158): feed the chunked core (`gdn_prefill_chunked`) and the sequential core
/// (`gated_delta_net_decode`) the same core inputs a real GDN block produces (q/k = l2norm of conv(proj),
/// v = silu(conv(proj)), g = softplus(proj)*ssm_a, beta = sigmoid(proj)) at real dims, and report per-head
/// state error with each head's decay magnitude. Sweeps dt magnitude to test the cumulative-log-decay
/// cancellation hypothesis. Prints, asserts nothing.
#[test]
fn gdn_core_on_realistic_frontend_inputs_per_head_real_dims() {
    use poot_graph_ir::op::{BinOp, RedOp, UnOp};
    use poot_graph_ir::ops::{causal_conv1d_prefill, linear, sigmoid, softplus};
    use poot_graph_ir::types::Scalar;

    let (h, hk, hv, d, ck, l) = (2048usize, 16usize, 32usize, 128usize, 4usize, 115usize);
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;
    let ssm_a_d: Vec<f32> = (0..hv)
        .map(|i| -0.02 - 72.0 * (i as f32 / (hv - 1) as f32))
        .collect();

    for &dt_scale in &[1.0f32, 8.0] {
        // Front-end graph outputting q(H_k), k(H_k), v(H_v), g(H_v per pos), beta(H_v per pos).
        let fb = Builder::new();
        let x = fb.constant("x", TensorType::f32(vec![1, l, h]));
        let w_qkv = fb.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
        let w_conv = fb.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
        let w_beta = fb.constant("w_beta", TensorType::f32(vec![h, hv]));
        let w_alpha = fb.constant("w_alpha", TensorType::f32(vec![h, hv]));
        let dt_bias = fb.constant("dt_bias", TensorType::f32(vec![hv]));
        let ssm_a = fb.constant("ssm_a", TensorType::f32(vec![hv]));
        let qkv_mixed = linear(&fb, x, w_qkv, None);
        let conv_out =
            poot_graph_ir::ops::silu(&fb, causal_conv1d_prefill(&fb, qkv_mixed, w_conv, ck));
        let q_flat = fb.slice(conv_out, 2, 0, key_dim);
        let k_flat = fb.slice(conv_out, 2, key_dim, 2 * key_dim);
        let v_flat = fb.slice(conv_out, 2, 2 * key_dim, conv_dim);
        let q = fb.transpose(fb.reshape(q_flat, vec![1, l, hk, d]), vec![0, 2, 1, 3]);
        let k = fb.transpose(fb.reshape(k_flat, vec![1, l, hk, d]), vec![0, 2, 1, 3]);
        let v = fb.transpose(fb.reshape(v_flat, vec![1, l, hv, d]), vec![0, 2, 1, 3]);
        // l2 norm of q,k over the last axis.
        let l2 = |t| {
            let sq = fb.binary(BinOp::Mul, t, t);
            let ss = fb.reduce(RedOp::Sum, sq, 3, true);
            let nrm = fb.unary(
                UnOp::Sqrt,
                fb.binary_scalar(BinOp::Add, ss, Scalar::F32(eps)),
            );
            fb.binary(BinOp::Div, t, nrm)
        };
        let q = l2(q);
        let k = l2(k);
        let beta = sigmoid(&fb, linear(&fb, x, w_beta, None));
        let alpha = softplus(
            &fb,
            fb.binary(BinOp::Add, linear(&fb, x, w_alpha, None), dt_bias),
        );
        let g = fb.binary(BinOp::Mul, alpha, ssm_a);
        let g4 = fb.transpose(fb.reshape(g, vec![1, l, hv, 1]), vec![0, 2, 1, 3]); // [1,Hv,L,1]
        let beta4 = fb.transpose(fb.reshape(beta, vec![1, l, hv, 1]), vec![0, 2, 1, 3]);
        // Multiple outputs via eval_all and reading ids.
        let fg = fb.finish(q); // primary irrelevant; use eval_all to read every tap.
        let mut fin = HashMap::new();
        fin.insert(x.id, HostTensor::f32(vec![1, l, h], fill(l * h, 111)));
        fin.insert(
            w_qkv.id,
            HostTensor::f32(vec![h, conv_dim], fill(h * conv_dim, 112)),
        );
        fin.insert(
            w_conv.id,
            HostTensor::f32(vec![ck, conv_dim], fill(ck * conv_dim, 114)),
        );
        fin.insert(w_beta.id, HostTensor::f32(vec![h, hv], fill(h * hv, 115)));
        // dt_scale scales w_alpha to raise/lower the softplus argument (dt magnitude).
        fin.insert(
            w_alpha.id,
            HostTensor::f32(
                vec![h, hv],
                fill(h * hv, 116)
                    .into_iter()
                    .map(|x| x * dt_scale)
                    .collect(),
            ),
        );
        fin.insert(dt_bias.id, HostTensor::f32(vec![hv], fill(hv, 117)));
        fin.insert(ssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
        let env = (|| -> Result<Vec<Option<HostTensor>>, EvalError> {
            let values: HashMap<ValueId, Value> = fin
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let environment = crate::eval(
                &fg,
                &values,
                EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
            )?
            .environment
            .expect("keep_environment was set");
            environment
                .into_iter()
                .map(|slot| slot.map(Value::into_host).transpose())
                .collect()
        })()
        .unwrap();
        let read = |t: poot_graph_ir::Traced| env[t.id].clone().unwrap();
        let (qd, kd, vd, gd, betad) = (
            read(q).as_f32().unwrap().to_vec(),
            read(k).as_f32().unwrap().to_vec(),
            read(v).as_f32().unwrap().to_vec(),
            read(g4).as_f32().unwrap().to_vec(),
            read(beta4).as_f32().unwrap().to_vec(),
        );
        // g magnitude report.
        let gmin = gd.iter().cloned().fold(f32::INFINITY, f32::min);
        let gmax = gd.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Tiled repeat of q,k H_k->H_v for the sequential reference.
        let tile = |src: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; hv * l * d];
            for h2 in 0..hv {
                let hk2 = h2 % hk;
                out[h2 * l * d..(h2 + 1) * l * d]
                    .copy_from_slice(&src[hk2 * l * d..(hk2 + 1) * l * d]);
            }
            out
        };
        let qv = tile(&qd);
        let kv = tile(&kd);

        // Sequential core (chunk-independent reference), computed first so the chunk sweep can diff against it.
        let db = Builder::new();
        let q_in = db.constant("q", TensorType::f32(vec![1, hv, 1, d]));
        let k_in = db.constant("k", TensorType::f32(vec![1, hv, 1, d]));
        let v_in = db.constant("v", TensorType::f32(vec![1, hv, 1, d]));
        let g_in = db.constant("g", TensorType::f32(vec![1, hv, 1, 1]));
        let bt_in = db.constant("beta", TensorType::f32(vec![1, hv, 1, 1]));
        let ds_in = db.state_input(
            "s",
            TensorType::f32(vec![1, hv, d, d]),
            StateRole::Recurrent,
        );
        let (_o, s_out) = gated_delta_net_decode(&db, q_in, k_in, v_in, g_in, bt_in, ds_in);
        let dg = db.finish_with_state(_o, &[(ds_in, s_out)]);
        let mut state = HostTensor::f32(vec![1, hv, d, d], vec![0.0; hv * d * d]);
        for t in 0..l {
            let qt: Vec<f32> = (0..hv * d)
                .map(|i| qv[(i / d * l + t) * d + i % d])
                .collect();
            let kt: Vec<f32> = (0..hv * d)
                .map(|i| kv[(i / d * l + t) * d + i % d])
                .collect();
            let vt: Vec<f32> = (0..hv * d)
                .map(|i| vd[(i / d * l + t) * d + i % d])
                .collect();
            let gt: Vec<f32> = (0..hv).map(|hh| gd[hh * l + t]).collect();
            let btt: Vec<f32> = (0..hv).map(|hh| betad[hh * l + t]).collect();
            let mut inp = HashMap::new();
            inp.insert(q_in.id, HostTensor::f32(vec![1, hv, 1, d], qt));
            inp.insert(k_in.id, HostTensor::f32(vec![1, hv, 1, d], kt));
            inp.insert(v_in.id, HostTensor::f32(vec![1, hv, 1, d], vt));
            inp.insert(g_in.id, HostTensor::f32(vec![1, hv, 1, 1], gt));
            inp.insert(bt_in.id, HostTensor::f32(vec![1, hv, 1, 1], btt));
            inp.insert(ds_in.id, state.clone());
            state = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: HashMap<ValueId, Value> = inp
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation =
                    crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .unwrap()
            .1
            .into_iter()
            .next()
            .unwrap();
        }
        let want_state = state.as_f32().unwrap().to_vec();

        // Chunk-size sweep of the chunked core on the same inputs. Cancellation in the within-chunk cumulative
        // log-decay prefix sum G_i (tril_incl @ g) scales with the chunk's cumulative magnitude, so a smaller
        // chunk should reduce the error if cancellation is the cause.
        eprintln!("--- dt_scale={dt_scale} g_range=[{gmin:.2},{gmax:.2}] ---");
        for &c in &[64usize, 32, 16, 8] {
            let mut ti = vec![0.0f32; c * c];
            let mut ts = vec![0.0f32; c * c];
            for i in 0..c {
                for j in 0..c {
                    if j <= i {
                        ti[i * c + j] = 1.0;
                    }
                    if j < i {
                        ts[i * c + j] = 1.0;
                    }
                }
            }
            let cb = Builder::new();
            let cq = cb.constant("q", TensorType::f32(vec![1, hk, l, d]));
            let ckk = cb.constant("k", TensorType::f32(vec![1, hk, l, d]));
            let cv = cb.constant("v", TensorType::f32(vec![1, hv, l, d]));
            let cg = cb.constant("g", TensorType::f32(vec![1, hv, l, 1]));
            let cbeta = cb.constant("beta", TensorType::f32(vec![1, hv, l, 1]));
            let cs = cb.state_input(
                "s",
                TensorType::f32(vec![1, hv, d, d]),
                StateRole::Recurrent,
            );
            let cti = cb.constant("ti", TensorType::f32(vec![1, 1, c, c]));
            let cts = cb.constant("ts", TensorType::f32(vec![1, 1, c, c]));
            let (co, cso) = gdn_prefill_chunked(&cb, cq, ckk, cv, cg, cbeta, cs, cti, cts, c);
            let cgraph = cb.finish_with_state(co, &[(cs, cso)]);
            let mut cin = HashMap::new();
            cin.insert(cq.id, HostTensor::f32(vec![1, hk, l, d], qd.clone()));
            cin.insert(ckk.id, HostTensor::f32(vec![1, hk, l, d], kd.clone()));
            cin.insert(cv.id, HostTensor::f32(vec![1, hv, l, d], vd.clone()));
            cin.insert(cg.id, HostTensor::f32(vec![1, hv, l, 1], gd.clone()));
            cin.insert(cbeta.id, HostTensor::f32(vec![1, hv, l, 1], betad.clone()));
            cin.insert(
                cs.id,
                HostTensor::f32(vec![1, hv, d, d], vec![0.0; hv * d * d]),
            );
            cin.insert(cti.id, HostTensor::f32(vec![1, 1, c, c], ti));
            cin.insert(cts.id, HostTensor::f32(vec![1, 1, c, c], ts));
            let (_go, gss) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
                let values: HashMap<ValueId, Value> = cin
                    .iter()
                    .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                    .collect();
                let evaluation =
                    crate::eval(&cgraph, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
                let state = evaluation
                    .state
                    .into_iter()
                    .map(Value::into_host)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((evaluation.output.into_host()?, state))
            })()
            .unwrap();
            let got_state = gss.into_iter().next().unwrap().as_f32().unwrap().to_vec();
            let mut worst = (0usize, 0.0f32);
            for hh in 0..hv {
                let head = hh * d * d..(hh + 1) * d * d;
                // A probe that prints rather than asserts, so a non-finite chunked state is reported as an infinite
                // error instead of failing the test. At dt_scale=8 the front end's `ops::softplus` (`ln(1 + exp(x))`)
                // overflows for x > 88.7, g becomes -inf, and the chunked cumulative-decay sum yields NaN: the NaN this
                // sweep printed as 0 while the error was a NaN-blind fold (R482-002).
                let he = if got_state[head.clone()].iter().all(|v| v.is_finite()) {
                    max_abs_error(&got_state[head.clone()], &want_state[head])
                } else {
                    f32::INFINITY
                };
                if he > worst.1 {
                    worst = (hh, he);
                }
            }
            eprintln!(
                "  chunk={c:2}: WORST head {} state_err={:.3e}",
                worst.0, worst.1
            );
        }
    }
}

/// Diagnostic (card 158): the whole GDN block (conv + l2norm + g/beta shaping + delta-rule core + gated
/// output RMSNorm + out-projection) at real Qwen3-Next dims (h=2048, H_k=16, H_v=32, D=128, K=4, C=64) with
/// real-magnitude decay (ssm_a spanning -0.02..-72 like blk.0), comparing `qwen3next_gdn_prefill_block`
/// against L sequential `qwen3next_gdn_block` calls. `L=70` (2 chunks: 64+6). Prints errors, asserts
/// nothing.
#[test]
fn qwen3next_gdn_block_prefill_vs_decode_real_dims() {
    use poot_models::qwen3next::{GdnHeadOrder, qwen3next_gdn_block, qwen3next_gdn_prefill_block};

    let (h, hk, hv, d, ck, l, chunk) = (
        2048usize, 16usize, 32usize, 128usize, 4usize, 70usize, 64usize,
    );
    let key_dim = hk * d;
    let value_dim = hv * d;
    let conv_dim = 2 * key_dim + value_dim;
    let eps = 1e-6f32;

    let xd = fill(l * h, 111);
    let w_qkv_d = fill(h * conv_dim, 112);
    let w_gate_d = fill(h * value_dim, 113);
    let w_conv_d = fill(ck * conv_dim, 114);
    let w_beta_d = fill(h * hv, 115);
    let w_alpha_d = fill(h * hv, 116);
    let dt_bias_d = fill(hv, 117);
    // Real-magnitude ssm_a: -0.02 .. -72 across heads (blk.0 range).
    let ssm_a_d: Vec<f32> = (0..hv)
        .map(|i| -0.02 - 72.0 * (i as f32 / (hv - 1) as f32))
        .collect();
    let norm_w_d = fill(d, 119);
    let w_out_d = fill(value_dim * h, 120);

    // Sequential decode reference.
    let db = Builder::new();
    let dx = db.constant("x", TensorType::f32(vec![1, 1, h]));
    let dw_qkv = db.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let dw_gate = db.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let dw_conv = db.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let dw_beta = db.constant("w_beta", TensorType::f32(vec![h, hv]));
    let dw_alpha = db.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let ddt_bias = db.constant("dt_bias", TensorType::f32(vec![hv]));
    let dssm_a = db.constant("ssm_a", TensorType::f32(vec![hv]));
    let dnorm_w = db.constant("norm_w", TensorType::f32(vec![d]));
    let dw_out = db.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let dcache_in = db.state_input(
        "conv_cache",
        TensorType::f32(vec![1, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ds_in = db.state_input(
        "s",
        TensorType::f32(vec![1, hv, d, d]),
        StateRole::Recurrent,
    );
    let (dout, dcache_out, ds_out) = qwen3next_gdn_block(
        &db,
        dx,
        dw_qkv,
        dw_gate,
        dw_conv,
        dw_beta,
        dw_alpha,
        ddt_bias,
        dssm_a,
        dnorm_w,
        dw_out,
        dcache_in,
        ds_in,
        hk,
        hv,
        d,
        ck,
        eps,
        GdnHeadOrder::Tiled,
    );
    let dg = db.finish_with_state(dout, &[(dcache_in, dcache_out), (ds_in, ds_out)]);
    let mut cache = HostTensor::f32(vec![1, ck - 1, conv_dim], vec![0.0f32; (ck - 1) * conv_dim]);
    let mut state = HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]);
    let mut want_o = vec![0.0f32; l * h];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            dx.id,
            HostTensor::f32(vec![1, 1, h], xd[t * h..(t + 1) * h].to_vec()),
        );
        inp.insert(
            dw_qkv.id,
            HostTensor::f32(vec![h, conv_dim], w_qkv_d.clone()),
        );
        inp.insert(
            dw_gate.id,
            HostTensor::f32(vec![h, value_dim], w_gate_d.clone()),
        );
        inp.insert(
            dw_conv.id,
            HostTensor::f32(vec![ck, conv_dim], w_conv_d.clone()),
        );
        inp.insert(dw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d.clone()));
        inp.insert(dw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d.clone()));
        inp.insert(ddt_bias.id, HostTensor::f32(vec![hv], dt_bias_d.clone()));
        inp.insert(dssm_a.id, HostTensor::f32(vec![hv], ssm_a_d.clone()));
        inp.insert(dnorm_w.id, HostTensor::f32(vec![d], norm_w_d.clone()));
        inp.insert(
            dw_out.id,
            HostTensor::f32(vec![value_dim, h], w_out_d.clone()),
        );
        inp.insert(dcache_in.id, cache.clone());
        inp.insert(ds_in.id, state.clone());
        let (ot, ns) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        want_o[t * h..(t + 1) * h].copy_from_slice(ot.as_f32().unwrap());
        let mut it = ns.into_iter();
        cache = it.next().unwrap();
        state = it.next().unwrap();
    }
    let want_state = state.as_f32().unwrap().to_vec();

    // Batched prefill block.
    let c = chunk;
    let mut tril_incl = vec![0.0f32; c * c];
    let mut tril_strict = vec![0.0f32; c * c];
    for i in 0..c {
        for j in 0..c {
            if j <= i {
                tril_incl[i * c + j] = 1.0;
            }
            if j < i {
                tril_strict[i * c + j] = 1.0;
            }
        }
    }
    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, h]));
    let pw_qkv = pb.constant("w_qkv", TensorType::f32(vec![h, conv_dim]));
    let pw_gate = pb.constant("w_gate", TensorType::f32(vec![h, value_dim]));
    let pw_conv = pb.constant("w_conv", TensorType::f32(vec![ck, conv_dim]));
    let pw_beta = pb.constant("w_beta", TensorType::f32(vec![h, hv]));
    let pw_alpha = pb.constant("w_alpha", TensorType::f32(vec![h, hv]));
    let pdt_bias = pb.constant("dt_bias", TensorType::f32(vec![hv]));
    let pssm_a = pb.constant("ssm_a", TensorType::f32(vec![hv]));
    let pnorm_w = pb.constant("norm_w", TensorType::f32(vec![d]));
    let pw_out = pb.constant("w_out", TensorType::f32(vec![value_dim, h]));
    let ps_in = pb.state_input(
        "s",
        TensorType::f32(vec![1, hv, d, d]),
        StateRole::Recurrent,
    );
    let pconv_cache_ph = pb.state_input(
        "conv_cache_ph",
        TensorType::f32(vec![1, ck - 1, conv_dim]),
        StateRole::Recurrent,
    );
    let ptril_incl = pb.constant("tril_incl", TensorType::f32(vec![1, 1, c, c]));
    let ptril_strict = pb.constant("tril_strict", TensorType::f32(vec![1, 1, c, c]));
    let (pout, _pcache_out, ps_out) = qwen3next_gdn_prefill_block(
        &pb,
        px,
        pw_qkv,
        pw_gate,
        pw_conv,
        pw_beta,
        pw_alpha,
        pdt_bias,
        pssm_a,
        pnorm_w,
        pw_out,
        ps_in,
        ptril_incl,
        ptril_strict,
        hk,
        hv,
        d,
        ck,
        c,
        eps,
        GdnHeadOrder::Tiled,
    );
    let pg = pb.finish_with_state(pout, &[(ps_in, ps_out), (pconv_cache_ph, _pcache_out)]);
    let mut pinp = HashMap::new();
    pinp.insert(px.id, HostTensor::f32(vec![1, l, h], xd));
    pinp.insert(pw_qkv.id, HostTensor::f32(vec![h, conv_dim], w_qkv_d));
    pinp.insert(pw_gate.id, HostTensor::f32(vec![h, value_dim], w_gate_d));
    pinp.insert(pw_conv.id, HostTensor::f32(vec![ck, conv_dim], w_conv_d));
    pinp.insert(pw_beta.id, HostTensor::f32(vec![h, hv], w_beta_d));
    pinp.insert(pw_alpha.id, HostTensor::f32(vec![h, hv], w_alpha_d));
    pinp.insert(pdt_bias.id, HostTensor::f32(vec![hv], dt_bias_d));
    pinp.insert(pssm_a.id, HostTensor::f32(vec![hv], ssm_a_d));
    pinp.insert(pnorm_w.id, HostTensor::f32(vec![d], norm_w_d));
    pinp.insert(pw_out.id, HostTensor::f32(vec![value_dim, h], w_out_d));
    pinp.insert(
        ps_in.id,
        HostTensor::f32(vec![1, hv, d, d], vec![0.0f32; hv * d * d]),
    );
    pinp.insert(
        pconv_cache_ph.id,
        HostTensor::f32(vec![1, ck - 1, conv_dim], vec![0.0f32; (ck - 1) * conv_dim]),
    );
    pinp.insert(ptril_incl.id, HostTensor::f32(vec![1, 1, c, c], tril_incl));
    pinp.insert(
        ptril_strict.id,
        HostTensor::f32(vec![1, 1, c, c], tril_strict),
    );
    let (got_o, got_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = pinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&pg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    let got_state = got_states.into_iter().next().unwrap();

    // Last-position output error (position L-1).
    let last_row = (l - 1) * h..l * h;
    let last_err = max_abs_error(
        &got_o.as_f32().unwrap()[last_row.clone()],
        &want_o[last_row],
    );
    let o_err = max_abs_error(got_o.as_f32().unwrap(), &want_o);
    let s_err = max_abs_error(got_state.as_f32().unwrap(), &want_state);
    eprintln!(
        "gdn_block REAL dims (h=2048,Hv=32,D=128,C=64,L=70, ssm_a -0.02..-72): \
         output_max_abs_err(all)={o_err:.3e} last_pos={last_err:.3e} state_max_abs_err={s_err:.3e}"
    );
}

/// Whole-block analogue of `qwen3next_gdn_prefill_block_matches_decode` for the gated full-attention
/// block: `qwen3next_gated_attention_prefill` must reproduce `L` sequential `qwen3next_gated_attention` calls
/// exactly, in per-position output and final KV cache. The decode block's `pos_idx` is a baked literal, so
/// the reference retraces a fresh decode graph per step.
///
/// `n_heads=4, n_kv_heads=2`, `head_dim=4`, `rotary_dim=2` (partial RoPE), `hidden=6`, `L=cap=8`. Both paths
/// start from a zero KV cache and share all weights (RoPE tables and causal mask included).
#[test]
fn qwen3next_gated_attention_prefill_matches_decode() {
    use poot_models::qwen3next::{qwen3next_gated_attention, qwen3next_gated_attention_prefill};

    let (h, nh, nkv, hd) = (6usize, 4usize, 2usize, 4usize); // hidden, n_heads, n_kv_heads, head_dim
    let rotary_dim = 2usize;
    let l = 8usize;
    let cap = l; // cache capacity == L, so decode's final cache == prefill's filled cache exactly
    let max_pos = l;
    let eps = 1e-6f32;
    let q_dim = nh * 2 * hd; // wq output width (interleaved query|gate)
    let kv_dim = nkv * hd;

    // Deterministic synthetic weights shared between the two paths.
    let x_data = fill(l * h, 201); // [1, L, H] row-major [L, H]
    let wq_data = fill(h * q_dim, 202); // [H, n_heads*2*head_dim]
    let wk_data = fill(h * kv_dim, 203); // [H, n_kv_heads*head_dim]
    let wv_data = fill(h * kv_dim, 204); // [H, n_kv_heads*head_dim]
    let wo_data = fill(nh * hd * h, 205); // [n_heads*head_dim, H]
    let q_norm_w_data = fill(hd, 206); // [head_dim]
    let k_norm_w_data = fill(hd, 207); // [head_dim]
    let cos_data = fill(max_pos * rotary_dim, 208); // [max_pos, rotary_dim]
    let sin_data = fill(max_pos * rotary_dim, 209); // [max_pos, rotary_dim]

    // --- sequential decode reference: L calls of qwen3next_gated_attention from a zero cache, retracing a
    // graph per step since pos_idx is a baked literal. ---
    let mut cache_k = poot_tensor::HostTensor::zeros(vec![1, nkv, cap, hd]);
    let mut cache_v = poot_tensor::HostTensor::zeros(vec![1, nkv, cap, hd]);
    let mut want_o = vec![0.0f32; l * h];
    for t in 0..l {
        let db = Builder::new();
        let dx = db.constant("x", TensorType::f32(vec![1, 1, h]));
        let dwq = db.constant("wq", TensorType::f32(vec![h, q_dim]));
        let dwk = db.constant("wk", TensorType::f32(vec![h, kv_dim]));
        let dwv = db.constant("wv", TensorType::f32(vec![h, kv_dim]));
        let dwo = db.constant("wo", TensorType::f32(vec![nh * hd, h]));
        let dqn = db.constant("qn", TensorType::f32(vec![hd]));
        let dkn = db.constant("kn", TensorType::f32(vec![hd]));
        let dcos = db.constant("cos", TensorType::f32(vec![max_pos, rotary_dim]));
        let dsin = db.constant("sin", TensorType::f32(vec![max_pos, rotary_dim]));
        let dpos = db.constant("pos", TensorType::f32(vec![]));
        let dkc_in = db.state_input(
            "k_cache",
            TensorType::f32(vec![1, nkv, cap, hd]),
            StateRole::Recurrent,
        );
        let dvc_in = db.state_input(
            "v_cache",
            TensorType::f32(vec![1, nkv, cap, hd]),
            StateRole::Recurrent,
        );
        let (dout, dkc_out, dvc_out) = qwen3next_gated_attention(
            &db, dx, dwq, dwk, dwv, dwo, dqn, dkn, dcos, dsin, dpos, dkc_in, dvc_in, nh, nkv, hd,
            t, eps,
        );
        assert_eq!(db.aval(dout).shape, vec![1, 1, h], "decode step {t} shape");
        let dg = db.finish_with_state(dout, &[(dkc_in, dkc_out), (dvc_in, dvc_out)]);

        let mut inp = HashMap::new();
        inp.insert(
            dx.id,
            poot_tensor::HostTensor::f32(vec![1, 1, h], x_data[t * h..(t + 1) * h].to_vec()),
        );
        inp.insert(
            dwq.id,
            poot_tensor::HostTensor::f32(vec![h, q_dim], wq_data.clone()),
        );
        inp.insert(
            dwk.id,
            poot_tensor::HostTensor::f32(vec![h, kv_dim], wk_data.clone()),
        );
        inp.insert(
            dwv.id,
            poot_tensor::HostTensor::f32(vec![h, kv_dim], wv_data.clone()),
        );
        inp.insert(
            dwo.id,
            poot_tensor::HostTensor::f32(vec![nh * hd, h], wo_data.clone()),
        );
        inp.insert(
            dqn.id,
            poot_tensor::HostTensor::f32(vec![hd], q_norm_w_data.clone()),
        );
        inp.insert(
            dkn.id,
            poot_tensor::HostTensor::f32(vec![hd], k_norm_w_data.clone()),
        );
        inp.insert(
            dcos.id,
            poot_tensor::HostTensor::f32(vec![max_pos, rotary_dim], cos_data.clone()),
        );
        inp.insert(
            dsin.id,
            poot_tensor::HostTensor::f32(vec![max_pos, rotary_dim], sin_data.clone()),
        );
        inp.insert(
            dpos.id,
            poot_tensor::HostTensor::f32(vec![], vec![t as f32]),
        );
        inp.insert(dkc_in.id, cache_k.clone());
        inp.insert(dvc_in.id, cache_v.clone());

        let (ot, new_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&dg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        assert_eq!(ot.shape(), vec![1, 1, h]);
        want_o[t * h..(t + 1) * h].copy_from_slice(ot.as_f32().unwrap());
        let mut it = new_states.into_iter();
        cache_k = it.next().unwrap();
        cache_v = it.next().unwrap();
    }
    let want_cache_k = cache_k.as_f32().unwrap().to_vec();
    let want_cache_v = cache_v.as_f32().unwrap().to_vec();

    // --- prefill block: one call over all L positions, zero cache start. ---
    // causal mask [1,1,L,L]: 0 on/below the diagonal, -1e30 above.
    let mask_data: Vec<f32> = (0..l * l)
        .map(|idx| {
            let (i, j) = (idx / l, idx % l);
            if j <= i { 0.0 } else { -1e30 }
        })
        .collect();

    let pb = Builder::new();
    let px = pb.constant("x", TensorType::f32(vec![1, l, h]));
    let pwq = pb.constant("wq", TensorType::f32(vec![h, q_dim]));
    let pwk = pb.constant("wk", TensorType::f32(vec![h, kv_dim]));
    let pwv = pb.constant("wv", TensorType::f32(vec![h, kv_dim]));
    let pwo = pb.constant("wo", TensorType::f32(vec![nh * hd, h]));
    let pqn = pb.constant("qn", TensorType::f32(vec![hd]));
    let pkn = pb.constant("kn", TensorType::f32(vec![hd]));
    let pcos = pb.constant("cos", TensorType::f32(vec![max_pos, rotary_dim]));
    let psin = pb.constant("sin", TensorType::f32(vec![max_pos, rotary_dim]));
    let pmask = pb.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let pkc_in = pb.state_input(
        "k_cache",
        TensorType::f32(vec![1, nkv, cap, hd]),
        StateRole::Recurrent,
    );
    let pvc_in = pb.state_input(
        "v_cache",
        TensorType::f32(vec![1, nkv, cap, hd]),
        StateRole::Recurrent,
    );
    let (pout, pkc_out, pvc_out) = qwen3next_gated_attention_prefill(
        &pb, px, pwq, pwk, pwv, pwo, pqn, pkn, pcos, psin, pmask, pkc_in, pvc_in, nh, nkv, hd, eps,
    );
    assert_eq!(
        pb.aval(pout).shape,
        vec![1, l, h],
        "prefill block output shape [1,L,H]"
    );
    assert_eq!(
        pb.aval(pkc_out).shape,
        vec![1, nkv, cap, hd],
        "prefill block k cache shape"
    );
    assert_eq!(
        pb.aval(pvc_out).shape,
        vec![1, nkv, cap, hd],
        "prefill block v cache shape"
    );
    let pg = pb.finish_with_state(pout, &[(pkc_in, pkc_out), (pvc_in, pvc_out)]);

    let mut pinp = HashMap::new();
    pinp.insert(px.id, poot_tensor::HostTensor::f32(vec![1, l, h], x_data));
    pinp.insert(
        pwq.id,
        poot_tensor::HostTensor::f32(vec![h, q_dim], wq_data),
    );
    pinp.insert(
        pwk.id,
        poot_tensor::HostTensor::f32(vec![h, kv_dim], wk_data),
    );
    pinp.insert(
        pwv.id,
        poot_tensor::HostTensor::f32(vec![h, kv_dim], wv_data),
    );
    pinp.insert(
        pwo.id,
        poot_tensor::HostTensor::f32(vec![nh * hd, h], wo_data),
    );
    pinp.insert(
        pqn.id,
        poot_tensor::HostTensor::f32(vec![hd], q_norm_w_data),
    );
    pinp.insert(
        pkn.id,
        poot_tensor::HostTensor::f32(vec![hd], k_norm_w_data),
    );
    pinp.insert(
        pcos.id,
        poot_tensor::HostTensor::f32(vec![max_pos, rotary_dim], cos_data),
    );
    pinp.insert(
        psin.id,
        poot_tensor::HostTensor::f32(vec![max_pos, rotary_dim], sin_data),
    );
    pinp.insert(
        pmask.id,
        poot_tensor::HostTensor::f32(vec![1, 1, l, l], mask_data),
    );
    pinp.insert(
        pkc_in.id,
        poot_tensor::HostTensor::zeros(vec![1, nkv, cap, hd]),
    );
    pinp.insert(
        pvc_in.id,
        poot_tensor::HostTensor::zeros(vec![1, nkv, cap, hd]),
    );

    let (got_o, got_states) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
        let values: HashMap<ValueId, Value> = pinp
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = crate::eval(&pg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()?;
        Ok((evaluation.output.into_host()?, state))
    })()
    .unwrap();
    assert_eq!(got_o.shape(), vec![1, l, h]);
    let mut it = got_states.into_iter();
    let got_kc = it.next().unwrap();
    let got_vc = it.next().unwrap();
    assert_eq!(got_kc.shape(), vec![1, nkv, cap, hd]);
    assert_eq!(got_vc.shape(), vec![1, nkv, cap, hd]);

    let o_err = max_abs_error(got_o.as_f32().unwrap(), &want_o);
    let kc_err = max_abs_error(got_kc.as_f32().unwrap(), &want_cache_k);
    let vc_err = max_abs_error(got_vc.as_f32().unwrap(), &want_cache_v);
    eprintln!(
        "qwen3next_gated_attention_prefill_matches_decode: output_max_abs_err={o_err:.2e} \
         k_cache_max_abs_err={kc_err:.2e} v_cache_max_abs_err={vc_err:.2e}"
    );
    assert_close_rel(got_o.as_f32().unwrap(), &want_o, 1e-4);
    assert_close_rel(got_kc.as_f32().unwrap(), &want_cache_k, 1e-4);
    assert_close_rel(got_vc.as_f32().unwrap(), &want_cache_v, 1e-4);
}

#[test]
fn mamba_core_path_decode_loop_matches_reference() {
    // Mamba2 SSM core path: causal conv1d -> SiLU -> SSD with Delta = softplus(dt), run step by step with conv
    // cache and SSM state carried through the State mechanism, must equal a hand-rolled Mamba reference.
    // Hq=2 heads, P=4 head dim (conv channels = Hq*P = 8), N=3 state, K=4 conv.
    use poot_graph_ir::ops::{causal_conv1d_decode, mamba2_ssd_decode, softplus};
    let (l, hq, pdim, n, kk) = (4usize, 2usize, 4usize, 3usize, 4usize);
    let ch = hq * pdim; // conv channels = 8
    let xcv = |t: usize, c: usize| ((t * 3 + c) % 7) as f32 * 0.1 - 0.3; // conv input [L, ch]
    let bv = |t: usize, h: usize, i: usize| ((t + h * 2 + i) % 5) as f32 * 0.1 - 0.2; // B [L,Hq,N]
    let cvv = |t: usize, h: usize, i: usize| ((t * 2 + h + i) % 6) as f32 * 0.1 - 0.25; // C
    let dtv = |t: usize, h: usize| 0.1 + 0.15 * ((t + h) % 3) as f32; // dt pre-softplus [L,Hq]
    let wcv = |k: usize, c: usize| ((k + c) % 5) as f32 * 0.1 - 0.2; // conv kernel [K, ch]
    let av = |h: usize| -0.5 - 0.2 * h as f32;
    let dv = |h: usize| 0.2 + 0.1 * h as f32;
    let silu = |x: f32| x / (1.0 + (-x).exp());

    // Decode graph (one step), carrying conv cache + SSM state.
    let b = Builder::new();
    let xc = b.constant("xc", TensorType::f32(vec![1, 1, ch])); // conv input
    let wc = b.constant("wc", TensorType::f32(vec![kk, ch])); // conv kernel
    let bb = b.constant("b", TensorType::f32(vec![1, hq, 1, n]));
    let cc = b.constant("c", TensorType::f32(vec![1, hq, 1, n]));
    let dt = b.constant("dt", TensorType::f32(vec![1, hq, 1, 1])); // pre-softplus
    let ap = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
    let dsk = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
    let conv_cache = b.state_input(
        "cv",
        TensorType::f32(vec![1, kk - 1, ch]),
        StateRole::Recurrent,
    );
    let h_in = b.state_input(
        "h",
        TensorType::f32(vec![1, hq, n, pdim]),
        StateRole::Recurrent,
    );
    // conv -> silu, reshape [1,1,ch] -> [1,Hq,1,P]
    let (conv_out, conv_cache_out) = causal_conv1d_decode(&b, xc, wc, conv_cache, kk);
    let xact = poot_graph_ir::ops::silu(&b, conv_out);
    let xheads = b.reshape(xact, vec![1, hq, 1, pdim]);
    let delta = softplus(&b, dt); // Delta = softplus(dt)
    let (y, h_out) = mamba2_ssd_decode(&b, xheads, bb, cc, delta, ap, dsk, h_in);
    let g = b.finish_with_state(y, &[(conv_cache, conv_cache_out), (h_in, h_out)]);

    let bind_scalars = |inp: &mut HashMap<usize, HostTensor>| {
        inp.insert(
            wc.id,
            HostTensor::f32(
                vec![kk, ch],
                (0..kk * ch).map(|i| wcv(i / ch, i % ch)).collect(),
            ),
        );
        inp.insert(
            ap.id,
            HostTensor::f32(vec![1, hq, 1, 1], (0..hq).map(av).collect()),
        );
        inp.insert(
            dsk.id,
            HostTensor::f32(vec![1, hq, 1, 1], (0..hq).map(dv).collect()),
        );
    };

    let mut cv_cache = HostTensor::f32(vec![1, kk - 1, ch], vec![0.0f32; (kk - 1) * ch]);
    let mut h_state = HostTensor::f32(vec![1, hq, n, pdim], vec![0.0f32; hq * n * pdim]);
    let mut got = vec![0.0f32; hq * l * pdim];
    for t in 0..l {
        let mut inp = HashMap::new();
        inp.insert(
            xc.id,
            HostTensor::f32(vec![1, 1, ch], (0..ch).map(|c| xcv(t, c)).collect()),
        );
        inp.insert(
            bb.id,
            HostTensor::f32(
                vec![1, hq, 1, n],
                (0..hq * n).map(|i| bv(t, i / n, i % n)).collect(),
            ),
        );
        inp.insert(
            cc.id,
            HostTensor::f32(
                vec![1, hq, 1, n],
                (0..hq * n).map(|i| cvv(t, i / n, i % n)).collect(),
            ),
        );
        inp.insert(
            dt.id,
            HostTensor::f32(vec![1, hq, 1, 1], (0..hq).map(|h| dtv(t, h)).collect()),
        );
        bind_scalars(&mut inp);
        inp.insert(conv_cache.id, cv_cache.clone());
        inp.insert(h_in.id, h_state.clone());
        let (yt, new) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        cv_cache = new[0].clone();
        h_state = new[1].clone();
        for h in 0..hq {
            for p in 0..pdim {
                got[(h * l + t) * pdim + p] = yt.as_f32().unwrap()[h * pdim + p];
            }
        }
    }

    // Hand reference: conv -> silu per channel, then SSD per head with Delta=softplus(dt).
    // Conv channel c maps to (head = c/P, p = c%P).
    let mut want = vec![0.0f32; hq * l * pdim];
    let mut state = vec![0.0f32; hq * n * pdim];
    for t in 0..l {
        // conv + silu, per channel.
        let mut xc_act = vec![0.0f32; ch];
        for (c, slot) in xc_act.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..kk {
                let src = t as isize - (kk as isize - 1) + k as isize;
                if src >= 0 {
                    acc += wcv(k, c) * xcv(src as usize, c);
                }
            }
            *slot = silu(acc);
        }
        for h in 0..hq {
            let delta_h = (1.0 + dtv(t, h).exp()).ln(); // softplus
            let a_bar = (delta_h * av(h)).exp();
            for nn in 0..n {
                let b_bar = delta_h * bv(t, h, nn);
                for p in 0..pdim {
                    let xval = xc_act[h * pdim + p];
                    state[(h * n + nn) * pdim + p] =
                        a_bar * state[(h * n + nn) * pdim + p] + b_bar * xval;
                }
            }
            for p in 0..pdim {
                let mut acc = 0.0f32;
                for nn in 0..n {
                    acc += cvv(t, h, nn) * state[(h * n + nn) * pdim + p];
                }
                want[(h * l + t) * pdim + p] = acc + dv(h) * xc_act[h * pdim + p];
            }
        }
    }
    assert_close_rel(&got, &want, 1e-4);
}

#[test]
fn mamba2_ssd_decode_matches_recurrence() {
    // Mamba2 / SSD recurrence (h_t = exp(Delta*A)*h_{t-1} + (Delta*B)^T x_t; y = C@h + D*x) run step by step
    // via the State mechanism must equal the hand-rolled SSM recurrence. Hq=2 heads, N=3, P=4, L=4.
    let (l, hq, n, p) = (4usize, 2usize, 3usize, 4usize);
    let xv = |h: usize, t: usize, i: usize| ((h * 3 + t * 2 + i) % 7) as f32 * 0.1 - 0.3;
    let bv = |h: usize, t: usize, i: usize| ((h * 5 + t + i * 2) % 5) as f32 * 0.1 - 0.2;
    let cv = |h: usize, t: usize, i: usize| ((h + t * 3 + i) % 6) as f32 * 0.1 - 0.25;
    let dlt = |h: usize, t: usize| 0.3 + 0.2 * ((h + t) % 3) as f32; // timestep in (0.3, 0.7)
    let av = |h: usize| -0.5 - 0.3 * h as f32; // learned A (negative -> Abar in (0,1))
    let dv = |h: usize| 0.2 + 0.1 * h as f32; // skip D

    // build the decode graph (one step), carry h.
    let b = Builder::new();
    let x = b.constant("x", TensorType::f32(vec![1, hq, 1, p]));
    let bin = b.constant("b", TensorType::f32(vec![1, hq, 1, n]));
    let cin = b.constant("c", TensorType::f32(vec![1, hq, 1, n]));
    let delta = b.constant("dl", TensorType::f32(vec![1, hq, 1, 1]));
    let a_param = b.constant("a", TensorType::f32(vec![1, hq, 1, 1]));
    let d_skip = b.constant("d", TensorType::f32(vec![1, hq, 1, 1]));
    let h_in = b.state_input(
        "h",
        TensorType::f32(vec![1, hq, n, p]),
        StateRole::Recurrent,
    );
    let (y, h_out) = mamba2_ssd_decode(&b, x, bin, cin, delta, a_param, d_skip, h_in);
    let g = b.finish_with_state(y, &[(h_in, h_out)]);
    let ad: Vec<f32> = (0..hq).map(av).collect();
    let dd: Vec<f32> = (0..hq).map(dv).collect();

    let mut caches = vec![HostTensor::f32(vec![1, hq, n, p], vec![0.0f32; hq * n * p])];
    let mut got = vec![0.0f32; hq * l * p];
    for t in 0..l {
        let xt: Vec<f32> = (0..hq)
            .flat_map(|h| (0..p).map(move |i| xv(h, t, i)))
            .collect();
        let bt: Vec<f32> = (0..hq)
            .flat_map(|h| (0..n).map(move |i| bv(h, t, i)))
            .collect();
        let ct: Vec<f32> = (0..hq)
            .flat_map(|h| (0..n).map(move |i| cv(h, t, i)))
            .collect();
        let dlt_t: Vec<f32> = (0..hq).map(|h| dlt(h, t)).collect();
        let mut inp = HashMap::new();
        inp.insert(x.id, HostTensor::f32(vec![1, hq, 1, p], xt));
        inp.insert(bin.id, HostTensor::f32(vec![1, hq, 1, n], bt));
        inp.insert(cin.id, HostTensor::f32(vec![1, hq, 1, n], ct));
        inp.insert(delta.id, HostTensor::f32(vec![1, hq, 1, 1], dlt_t));
        inp.insert(a_param.id, HostTensor::f32(vec![1, hq, 1, 1], ad.clone()));
        inp.insert(d_skip.id, HostTensor::f32(vec![1, hq, 1, 1], dd.clone()));
        inp.insert(h_in.id, caches[0].clone());
        let (yt, new) = (|| -> Result<(HostTensor, Vec<HostTensor>), EvalError> {
            let values: HashMap<ValueId, Value> = inp
                .iter()
                .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
                .collect();
            let evaluation = crate::eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))?;
            let state = evaluation
                .state
                .into_iter()
                .map(Value::into_host)
                .collect::<Result<Vec<_>, _>>()?;
            Ok((evaluation.output.into_host()?, state))
        })()
        .unwrap();
        caches = new;
        for h in 0..hq {
            for i in 0..p {
                got[(h * l + t) * p + i] = yt.as_f32().unwrap()[h * p + i];
            }
        }
    }

    // hand SSM recurrence per head.
    let mut want = vec![0.0f32; hq * l * p];
    for h in 0..hq {
        let mut state = vec![0.0f32; n * p]; // h [N, P]
        for t in 0..l {
            let a_bar = (dlt(h, t) * av(h)).exp();
            for nn in 0..n {
                let b_bar = dlt(h, t) * bv(h, t, nn);
                for pp in 0..p {
                    state[nn * p + pp] = a_bar * state[nn * p + pp] + b_bar * xv(h, t, pp);
                }
            }
            for pp in 0..p {
                let mut acc = 0.0f32;
                for nn in 0..n {
                    acc += cv(h, t, nn) * state[nn * p + pp];
                }
                want[(h * l + t) * p + pp] = acc + dv(h) * xv(h, t, pp);
            }
        }
    }
    assert_close_rel(&got, &want, 1e-5);
}

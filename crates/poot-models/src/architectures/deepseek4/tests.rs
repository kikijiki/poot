use super::*;
use poot_graph_ir::Storage;

fn tiny_cfg() -> DeepseekV4Config {
    DeepseekV4Config {
        vocab: 10,
        hidden: 8,
        layers: 2,
        num_heads: 2,
        head_dim: 6,
        qk_rope_head_dim: 2,
        q_lora_rank: 5,
        o_groups: 2,
        o_lora_rank: 3,
        intermediate: 7,
        // 0.5, not the real 10.0: at 10.0 no fixture activation reaches the clamp, so the w1/w3 asymmetry
        // would be inert and swapping `clamp_max` for `clamp_sym` would stay green.
        swiglu_limit: 0.5,
        sliding_window: 3,
        eps: 1e-5,
        max_pos: 16,
        rope_theta: 10_000.0,
        hc_mult: 2,
        hc_sinkhorn_iters: 4,
        hc_eps: 1e-6,
        hca_compress_rate: 2,
        compress_rope_theta: 20_000.0,
        csa_compress_rate: 2,
        index_n_heads: 2,
        index_head_dim: 4,
        index_topk: 2,
        routed_experts: 4,
        experts_per_tok: 2,
        moe_intermediate: 4,
        // layer 0 routes from `tid2eid`, every later layer from `ffn.gate.bias`, so a two-layer
        // fixture exercises both rows.
        hash_router_layers: 1,
        route_scale: 1.5,
    }
}

#[test]
fn deepseek4_sliding_decode_validates_and_has_one_state_tensor_per_layer() {
    let cfg = tiny_cfg();
    let cap = 6;
    let g = trace_deepseek4_sliding_decode(cfg, cap)
        .expect("trace_deepseek4_sliding_decode should trace");
    g.validate()
        .expect("deepseek4 sliding decode graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    assert_eq!(
        g.state.len(),
        cfg.layers,
        "SC-001: one K==V cache per layer"
    );
    for (si, _so) in &g.state {
        assert_eq!(
            g.aval(*si).shape,
            vec![1, 1, cap, cfg.head_dim],
            "kv_cache shape"
        );
    }
}

#[test]
fn deepseek4_sliding_prefill_validates() {
    let cfg = tiny_cfg();
    let g = trace_deepseek4_sliding_prefill(cfg, 6)
        .expect("trace_deepseek4_sliding_prefill should trace");
    g.validate()
        .expect("deepseek4 sliding prefill graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
}

#[test]
fn deepseek4_hca_decode_validates_and_has_three_state_tensors_per_layer() {
    let cfg = tiny_cfg();
    let cap = 6; // multiple of hca_compress_rate=2
    let g = trace_deepseek4_hca_decode(cfg, cap).expect("trace_deepseek4_hca_decode should trace");
    g.validate()
        .expect("deepseek4 HCA decode graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    assert_eq!(
        g.state.len(),
        3 * cfg.layers,
        "local K==V cache + hca raw kv cache + hca raw gate cache, per layer"
    );
}

#[test]
fn deepseek4_hca_prefill_validates() {
    let cfg = tiny_cfg();
    let g = trace_deepseek4_hca_prefill(cfg, 6).expect("trace_deepseek4_hca_prefill should trace"); // multiple of hca_compress_rate=2
    g.validate()
        .expect("deepseek4 HCA prefill graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
}

#[test]
fn deepseek4_csa_decode_validates_and_has_five_state_tensors_per_layer() {
    let cfg = tiny_cfg();
    let cap = 8; // multiple of csa_compress_rate=2
    let g = trace_deepseek4_csa_decode(cfg, cap).expect("trace_deepseek4_csa_decode should trace");
    g.validate()
        .expect("deepseek4 CSA decode graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    assert_eq!(
        g.state.len(),
        5 * cfg.layers,
        "local K==V cache + csa compressor kv/gate raw cache + csa indexer kv/gate raw cache, per layer"
    );
}

#[test]
fn deepseek4_csa_prefill_validates() {
    let cfg = tiny_cfg();
    let g = trace_deepseek4_csa_prefill(cfg, 8).expect("trace_deepseek4_csa_prefill should trace"); // multiple of csa_compress_rate=2
    g.validate()
        .expect("deepseek4 CSA prefill graph should validate");
    assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
}

mod cpu_oracle {
    use super::*;
    use crate::deepseek2::deepseek2_rope_tables_interleaved;
    use std::collections::HashMap;

    /// Card 386: an absolute floor added to (not replacing) the relative `1e-4 * want.abs().max(1e-3)` check,
    /// for oracle pairs that chain 3 or more V4 layers. Derived, not fitted: a Higham-style `sqrt(N) * u`
    /// estimate over this fixture's ~711 order-sensitive f32 operations (rmsnorm reciprocal-vs-divide, sink-
    /// softmax accumulation order) gives ~3.2e-6; the floor is ~3.1x that. See
    /// specs/386-deepseek4-hybrid-stack-reference-disagreement/spec.md. Applied to every oracle pair that
    /// chains this many layers.
    const V4_HYBRID_STACK_ABS_FLOOR: f32 = 1e-5;

    /// Card 386 bisection tool: [`trace_deepseek4_hybrid_stack_prefill`] truncated after `upto` layers
    /// (`1..=schedule.len()`), outputting the widened residual `[1, l, hc, hidden]` (`l` the padded length the
    /// production tracer would use) instead of running the closing hyper_head/norm/lm_head chain.
    ///
    /// A near-verbatim copy of the production dispatch, so a rewrite of that function has one place to also
    /// update. Pairs with [`deepseek4_hybrid_stack_decode_ref`]'s `layer_residuals` capture: both carry the
    /// residual right after a layer's FFN combine, so a test can compare every position and layer count, not
    /// only a final logit. Only real positions (`0..seq_len`) are meaningful when `l > seq_len`.
    fn trace_deepseek4_hybrid_stack_prefill_residual_upto(
        cfg: DeepseekV4Config,
        seq_len: usize,
        schedule: &[V4LayerKind],
        upto: usize,
    ) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
        assert_eq!(
            schedule.len(),
            cfg.layers,
            "trace_deepseek4_hybrid_stack_prefill_residual_upto: schedule length must equal cfg.layers"
        );
        assert!(
            (1..=schedule.len()).contains(&upto),
            "trace_deepseek4_hybrid_stack_prefill_residual_upto: upto must select between one and \
every layer"
        );
        let active = &schedule[..upto];
        let has_sliding = active.contains(&V4LayerKind::Sliding);
        let has_hca = active.contains(&V4LayerKind::Hca);
        let has_csa = active.contains(&V4LayerKind::Csa);
        let l = deepseek4_prefill_pad_len(seq_len, &cfg, has_hca, has_csa);

        let b = Builder::new();
        let (h, hc, rd) = (cfg.hidden, cfg.hc_mult, cfg.qk_rope_head_dim);

        let tokens = b.slot(Slot::Token, TensorType::new(vec![l], DType::I32));
        // The plan is built from the full schedule for every `upto`; only fewer per-layer bodies are called.
        let (plan, dims, tokens, mut moe) =
            deepseek4_moe_preamble(&b, &cfg, schedule, tokens, V4MoePhase::Prefill)?;
        let mask = b.slot_named(Slot::Mask, "prefill", TensorType::f32(vec![1, 1, l, l]));

        let main_rope = has_sliding.then(|| {
            (
                b.constant(V4_ROPE_COS, TensorType::f32(vec![cfg.max_pos, rd / 2])),
                b.constant(V4_ROPE_SIN, TensorType::f32(vec![cfg.max_pos, rd / 2])),
            )
        });
        let compress_rope = (has_hca || has_csa).then(|| {
            (
                b.constant(
                    V4_COMPRESS_ROPE_COS,
                    TensorType::f32(vec![cfg.max_pos, rd / 2]),
                ),
                b.constant(
                    V4_COMPRESS_ROPE_SIN,
                    TensorType::f32(vec![cfg.max_pos, rd / 2]),
                ),
            )
        });

        let hca_info = has_hca.then(|| {
            let cr = cfg.hca_compress_rate;
            let n_windows = l / cr;
            let block_bias = b.slot_named(
                Slot::Activation,
                V4_HCA_BLOCK_BIAS,
                TensorType::f32(vec![1, 1, l, n_windows]),
            );
            let win_idx = b.slot_named(
                Slot::Activation,
                V4_HCA_WINDOW_POSITIONS,
                TensorType::f32(vec![n_windows]),
            );
            (block_bias, win_idx, n_windows)
        });
        let csa_info = has_csa.then(|| {
            let cr = cfg.csa_compress_rate;
            let n_windows = l / cr;
            let block_bias = b.slot_named(
                Slot::Activation,
                V4_CSA_BLOCK_BIAS,
                TensorType::f32(vec![1, 1, l, n_windows]),
            );
            let win_idx = b.slot_named(
                Slot::Activation,
                V4_CSA_WINDOW_POSITIONS,
                TensorType::f32(vec![n_windows]),
            );
            (block_bias, win_idx, n_windows)
        });

        let embed = v4_source(&b, plan.top_dense(V4TopDense::Embedding)?);
        let emb = b.cast(b.gather(embed, 0, tokens), DType::F32); // [L, hidden]
        let x0 = b.reshape(emb, vec![1, l, 1, h]);
        let mut x = b.broadcast(x0, vec![1, l, hc, h]);

        for (li, kind) in active.iter().enumerate() {
            x = match kind {
                V4LayerKind::Sliding => {
                    let (cos, sin) = main_rope.expect("has_sliding implies main_rope is built");
                    deepseek4_sliding_layer_prefill(
                        &b,
                        &cfg,
                        li,
                        x,
                        l,
                        mask,
                        cos,
                        sin,
                        &plan,
                        dims,
                        &mut moe.context(),
                    )?
                }
                V4LayerKind::Hca => {
                    let (cos, sin) = compress_rope.expect("has_hca implies compress_rope is built");
                    let (block_bias, win_idx, n_windows) =
                        hca_info.expect("has_hca implies hca_info is built");
                    let combined_mask = b.concat(3, &[mask, block_bias]);
                    deepseek4_hca_layer_prefill(
                        &b,
                        &cfg,
                        li,
                        x,
                        l,
                        cos,
                        sin,
                        combined_mask,
                        win_idx,
                        n_windows,
                        &plan,
                        dims,
                        &mut moe.context(),
                    )?
                }
                V4LayerKind::Csa => {
                    let (cos, sin) = compress_rope.expect("has_csa implies compress_rope is built");
                    let (block_bias, win_idx, n_windows) =
                        csa_info.expect("has_csa implies csa_info is built");
                    deepseek4_csa_layer_prefill(
                        &b,
                        &cfg,
                        li,
                        x,
                        l,
                        cos,
                        sin,
                        mask,
                        block_bias,
                        win_idx,
                        n_windows,
                        &plan,
                        dims,
                        &mut moe.context(),
                    )?
                }
            };
        }

        // No hyper_head/norm/lm_head: the bisection value is the raw residual after `upto` layers.
        moe.finish(b, x, &[])
    }

    /// Card 386 bisection tool, second half: the closing hyper_head -> norm -> lm_head chain as its own graph,
    /// taking the widened residual as an F32 constant input. Paired with
    /// [`trace_deepseek4_hybrid_stack_prefill_residual_upto`] at `upto = schedule.len()`, it gives logits for
    /// every position, not only the last. No I32 inputs, so plain `poot_eval::eval` runs it. It builds its own
    /// `DeepseekV4SourcePlan` (the top-level dense sources depend only on `cfg`, but the constructor wants a
    /// full schedule).
    fn trace_deepseek4_hybrid_stack_head(
        cfg: &DeepseekV4Config,
        schedule: &[V4LayerKind],
        l: usize,
    ) -> Result<Graph, DeepseekV4MoeError> {
        let plan = DeepseekV4SourcePlan::new(cfg, schedule)?;
        let b = Builder::new();
        let (h, hc) = (cfg.hidden, cfg.hc_mult);
        let x = b.constant("resid", TensorType::f32(vec![1, l, hc, h]));
        let hh_fn = v4_mhc_projection(&b, plan.top_dense(V4TopDense::HcHeadProjection)?);
        let hh_base = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadBase)?);
        let hh_scale = v4_source_f32(&b, plan.top_dense(V4TopDense::HcHeadScale)?);
        let collapsed_final = hyper_head(
            &b, x, hh_fn, hh_base, hh_scale, hc, h, l, cfg.eps, cfg.hc_eps,
        );
        let norm = v4_source_f32(&b, plan.top_dense(V4TopDense::FinalNorm)?);
        let xf = rmsnorm(&b, collapsed_final, norm, cfg.eps);
        let logits = v4_source_linear(&b, xf, plan.top_dense(V4TopDense::Head)?);
        Ok(b.finish(logits))
    }

    /// Card 364a MoE fixtures: packed expert owners, an independent packed decoder, the hash table, and the
    /// binder every full-model oracle uses to turn a graph constant into a `poot_eval::Value`. The decoder is
    /// written from the two format definitions, not the production packed path, so comparing against it
    /// compares two independent implementations.
    mod moe_fixture {
        use super::*;
        use poot_eval::Value;
        use poot_graph_ir::op::{PackedWeight, WeightFormat};
        use poot_quant::{OperandRole, PackedComponentRef, PackedPayload, SourceRole};
        use std::sync::Arc;

        /// The eight E2M1 magnitudes, sign in bit 3. Codes chosen so every fixture weight is exact.
        const E2M1_MAGNITUDE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

        /// E8M0 exponent byte `127` is exactly `2^0`, so a fixture scale never perturbs a weight.
        const E8M0_ONE: u8 = 127;

        /// Finite E4M3 codes: +-0.5, +-0.75, +-1.0, +-1.5, and zero.
        const E4M3_CODES: [u8; 9] = [0x00, 0x30, 0x34, 0x38, 0x3c, 0xb0, 0xb4, 0xb8, 0xbc];

        /// Bind a `Storage::Const` f32 value as its declared dtype. A plain `Value::Host` f32 mirror is fine
        /// for F32 (and for a BF16 value a `Cast(BF16, F32)` or a `MatMul`/`DenseContraction` weight operand
        /// reads, the dense carve-out) but a raw `Transpose`/`Gather`/`Reshape` of a
        /// declared-BF16 value reads it through `operand::bf16_view`, which has no `Value::Host` arm at all -
        /// every V4 dense weight this fixture binds reaches exactly such a `Transpose` through `v4_source` /
        /// `v4_source_linear` before any cast, so a BF16 const needs a real Card 370 owner view, not a
        /// `.with_bf16`-decorated `Tensor` (Card 554d, the BF16 sibling of the I32-slot binding regression).
        pub(super) fn bind_const(
            name: &str,
            shape: Vec<usize>,
            dtype: poot_tensor::DType,
            data: Vec<f32>,
        ) -> Value {
            if dtype == poot_tensor::DType::BF16 {
                bf16_owner_value(name, shape, &data)
            } else {
                Value::Host(poot_tensor::HostTensor::f32(shape, data))
            }
        }

        /// An owner-backed exact BF16 [`Value`] for `data`, which must already be BF16-exact (every plan-sourced
        /// dense row [`plan_dense_weights`] truncates, and [`routed_moe::router_weights`] truncates its one BF16
        /// row the same way): a one-row synthetic checkpoint, authenticated exactly like a real one, so the
        /// returned [`poot_eval::exact_dense::DenseOwnerTensorView`] is indistinguishable from a production
        /// Card 359/370 owner to every consumer.
        fn bf16_owner_value(name: &str, shape: Vec<usize>, data: &[f32]) -> Value {
            let bytes = crate::test_support::safetensors::bf16_bytes(name, data);
            let row = crate::test_support::safetensors::SourceRow {
                name: name.to_string(),
                dtype: "BF16",
                shape,
                bytes,
            };
            let checkpoint = crate::test_support::safetensors::load_checkpoint(
                "poot-models/card554d-bf16-fixture",
                "tiny",
                b"{}",
                vec![(
                    row,
                    poot_load::packed_safetensors::TensorDisposition::DenseBf16,
                )],
            );
            let owner = Arc::clone(
                &checkpoint
                    .mixed
                    .exact_metadata
                    .get(name)
                    .unwrap_or_else(|| panic!("no exact owner metadata for {name}"))
                    .owner,
            );
            Value::from(
                poot_eval::exact_dense::DenseOwnerTensorView::new(owner)
                    .unwrap_or_else(|error| panic!("{name}: bf16 owner view: {error}")),
            )
        }

        fn hash(name: &str, index: usize) -> u64 {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for byte in name.as_bytes().iter().chain(&index.to_le_bytes()) {
                h ^= u64::from(*byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
            h
        }

        fn e2m1_value(code: u8) -> f32 {
            let magnitude = E2M1_MAGNITUDE[usize::from(code & 0x07)];
            if code & 0x08 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }

        fn e4m3_value(byte: u8) -> f32 {
            let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
            let exponent = i32::from((byte >> 3) & 0x0f);
            let mantissa = f32::from(byte & 0x07);
            if exponent == 0 {
                sign * mantissa * 2.0f32.powi(-9)
            } else {
                sign * (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
            }
        }

        /// Deterministic packed payload for one linear id. Every scale byte is `E8M0_ONE`, so the logical
        /// weight is exactly the code table value and the reference needs no tolerance of its own.
        pub(super) fn packed_owner(
            linear_id: &str,
            descriptor: PackedWeight,
        ) -> Arc<PackedPayload> {
            let is_fp4 = descriptor.format() == WeightFormat::E2m1Row32;
            let weights: Vec<u8> = (0..descriptor
                .source_bytes(SourceRole::Planar(OperandRole::Codes)))
                .map(|index| {
                    if is_fp4 {
                        let low = (hash(linear_id, index) % 16) as u8;
                        let high = (hash(linear_id, index + 0x5000) % 16) as u8;
                        low | (high << 4)
                    } else {
                        E4M3_CODES[(hash(linear_id, index) % E4M3_CODES.len() as u64) as usize]
                    }
                })
                .collect();
            let scales =
                vec![E8M0_ONE; descriptor.source_bytes(SourceRole::Planar(OperandRole::Scale))];
            Arc::new(
                PackedPayload::try_new(
                    descriptor,
                    [
                        (SourceRole::Planar(OperandRole::Codes), weights.into()),
                        (SourceRole::Planar(OperandRole::Scale), scales.into()),
                    ],
                )
                .expect("fixture packed payload"),
            )
        }

        /// Decode one packed owner to its `[out, K]` logical f32 weight, row-major.
        ///
        /// Adjacent-K FP4 stores two codes per byte, the even `k` in the low nibble, with one E8M0 byte per
        /// row per 32 `k` values. E4M3 stores one code per byte with one E8M0 byte per 128x128 block.
        pub(super) fn decoded_weight(owner: &PackedPayload) -> Vec<f32> {
            let descriptor = owner.weight();
            let [out, k] = descriptor.shape();
            let weight_columns = descriptor.source_shape(SourceRole::Planar(OperandRole::Codes))[1];
            let scale_columns = descriptor.source_shape(SourceRole::Planar(OperandRole::Scale))[1];
            let weights = owner.bytes(SourceRole::Planar(OperandRole::Codes));
            let scales = owner.bytes(SourceRole::Planar(OperandRole::Scale));
            let fp4 = descriptor.format() == WeightFormat::E2m1Row32;
            let mut decoded = Vec::with_capacity(out * k);
            for row in 0..out {
                for column in 0..k {
                    let (value, scale_byte) = if fp4 {
                        let byte = weights[row * weight_columns + column / 2];
                        let code = if column.is_multiple_of(2) {
                            byte & 0x0f
                        } else {
                            byte >> 4
                        };
                        (e2m1_value(code), scales[row * scale_columns + column / 32])
                    } else {
                        (
                            e4m3_value(weights[row * weight_columns + column]),
                            scales[(row / 128) * scale_columns + column / 128],
                        )
                    };
                    decoded.push(value * 2.0f32.powi(i32::from(scale_byte) - 127));
                }
            }
            decoded
        }

        /// Every packed owner a plan names, keyed by linear id.
        ///
        /// Driven by the plan rather than by a hand-written list, so the attention linears, the CSA
        /// indexer weight and the experts all get owners from one table and a new packed row cannot
        /// be forgotten here.
        pub(super) fn plan_owners(
            plan: &DeepseekV4SourcePlan,
        ) -> HashMap<String, Arc<PackedPayload>> {
            plan.packed_sources()
                .map(|row| {
                    (
                        row.linear_id.clone(),
                        packed_owner(&row.linear_id, row.descriptor),
                    )
                })
                .collect()
        }

        /// `attn.wo_a`'s host-side grouped weight view for the reference oracle, from the plan's owner pair:
        /// decode `[o_groups * o_lora_rank, in_per_group]`, then split and transpose into
        /// `[o_groups, in_per_group, o_lora_rank]`. An oracle wants its own derivation. Reshaping straight into the grouped extents would swap the last two axes and stay
        /// shape-valid (the trap spec 281 warned about).
        pub(super) fn grouped_dequant(owner: &PackedPayload, o_groups: usize) -> Vec<f32> {
            let decoded = decoded_weight(owner);
            let [out, in_per_group] = owner.weight().shape();
            assert!(o_groups > 0 && out.is_multiple_of(o_groups));
            let o_lora_rank = out / o_groups;
            regroup_out_proj(&decoded, o_groups, o_lora_rank, in_per_group)
        }

        /// The pure remap `grouped_dequant` performs, split out so it can be pinned against a hand-written
        /// expectation (`deepseek4_grouped_dequant_splits_then_transposes`): split-then-transpose a row-major
        /// `[o_groups * o_lora_rank, in_per_group]` weight into `[o_groups, in_per_group, o_lora_rank]`.
        pub(super) fn regroup_out_proj(
            decoded: &[f32],
            o_groups: usize,
            o_lora_rank: usize,
            in_per_group: usize,
        ) -> Vec<f32> {
            let mut data = vec![0.0f32; o_groups * o_lora_rank * in_per_group];
            for g in 0..o_groups {
                for d in 0..o_lora_rank {
                    for i in 0..in_per_group {
                        data[(g * in_per_group + i) * o_lora_rank + d] =
                            decoded[(g * o_lora_rank + d) * in_per_group + i];
                    }
                }
            }
            data
        }

        /// `grouped_dequant`'s numeric content, pinned against a hand-written expectation; an axis-order bug
        /// would otherwise need a real end-to-end run to surface.
        #[test]
        fn deepseek4_grouped_dequant_splits_then_transposes() {
            // Stored [o_groups * o_lora_rank, in_per_group] = [2 * 3, 2], values 0..11 in checkpoint
            // order. Grouped view [o_groups, in_per_group, o_lora_rank] = [2, 2, 3].
            let decoded: Vec<f32> = (0..12).map(|v| v as f32).collect();
            let got = regroup_out_proj(&decoded, 2, 3, 2);
            assert_eq!(
                got,
                vec![0., 2., 4., 1., 3., 5., 6., 8., 10., 7., 9., 11.],
                "a direct reshape would give 0..11 unchanged - that is the spec-281 trap"
            );
        }

        /// The hash table one fixture uses: expert `(token + slot) % routed_experts`, so neighbouring
        /// token rows differ and a test that reads the wrong row fails.
        pub(super) fn tid2eid(cfg: &DeepseekV4Config) -> Vec<i32> {
            (0..cfg.vocab)
                .flat_map(|token| {
                    (0..cfg.experts_per_tok)
                        .map(move |slot| ((token + slot) % cfg.routed_experts) as i32)
                })
                .collect()
        }

        /// The per-step integer inputs a V4 graph needs. Decode passes the one token it is stepping;
        /// prefill passes the whole (possibly padded) prompt and leaves `position` unread.
        #[derive(Clone, Copy)]
        pub(super) struct V4StepInputs<'a> {
            pub tokens: &'a [usize],
            pub position: usize,
        }

        /// Bind one I32 input from its integer values, through Card 371's exact carrier.
        pub(super) fn exact_i32(shape: Vec<usize>, words: Vec<i32>) -> Value {
            Value::from(
                poot_eval::ExactI32TensorView::try_from_words(shape, Arc::from(words))
                    .expect("fixture exact I32 input"),
            )
        }

        /// The role of each `Cast I32 -> F32` in a V4 graph, which the exact evaluator needs and the planner
        /// would normally supply. A model test cannot call `poot-graph-plan`, so roles are classified from the
        /// graph: the cast a declared validation output reduces is the guard's witness, and every other
        /// I32-to-F32 cast is the router selector. `pub(super)` because `tests::cpu_oracle::routed_moe` (Card
        /// 424) needs the same classification to drive `poot_llm::Runner::run_exact_cpu_step_routed`.
        pub(super) fn cast_roles<'g>(
            g: &'g Graph<ValidationOutputs>,
            cfg: &DeepseekV4Config,
            ceiling: usize,
        ) -> impl FnMut(poot_graph_ir::ValueId) -> Option<poot_eval::cast_authority::ExactI32CastRole> + 'g
        {
            let expert_count = cfg.routed_experts;
            move |cast: poot_graph_ir::ValueId| {
                let witness = g.eqns.iter().any(|eqn| {
                    matches!(eqn.op, poot_graph_ir::OpKind::Reduce { .. })
                        && eqn.inputs.iter().any(|operand| {
                            matches!(operand, poot_graph_ir::Operand::Value(value) if *value == cast)
                        })
                        && g
                            .validation_outputs()
                            .iter()
                            .any(|declared| declared.value == eqn.out)
                });
                Some(if witness {
                    poot_eval::cast_authority::ExactI32CastRole::Witness {
                        selected_element_ceiling: ceiling,
                    }
                } else {
                    poot_eval::cast_authority::ExactI32CastRole::HostCheckedSelector(
                        poot_eval::cast_authority::ExactI32CastBounds {
                            selected_element_ceiling: ceiling,
                            expert_count,
                        },
                    )
                })
            }
        }

        /// Evaluate a stateless V4 graph on the CPU through Card 371's exact-I32 evaluator. (`eval_value_with_state`
        /// cannot run these graphs: its storage admission has no arm for an I32 input.)
        pub(super) fn eval_graph(
            cfg: &DeepseekV4Config,
            g: &Graph<ValidationOutputs>,
            inputs: &HashMap<poot_graph_ir::ValueId, Value>,
        ) -> Result<Value, poot_eval::EvalError> {
            let mut roles = cast_roles(g, cfg, g.values.len());
            let evaluation = poot_eval::eval(
                g,
                inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED)
                    .cast_authority(&mut roles),
            )?;
            Ok(evaluation.output)
        }

        /// [`eval_graph`] for a graph that carries state (Card 382): a decode step needs the exact-I32 bindings and
        /// its caches back.
        pub(super) fn eval_graph_with_state(
            cfg: &DeepseekV4Config,
            g: &Graph<ValidationOutputs>,
            inputs: &HashMap<poot_graph_ir::ValueId, Value>,
        ) -> Result<(Value, Vec<Value>), poot_eval::EvalError> {
            let mut roles = cast_roles(g, cfg, g.values.len());
            let evaluation = poot_eval::eval(
                g,
                inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED)
                    .cast_authority(&mut roles),
            )?;
            Ok((evaluation.output, evaluation.state))
        }

        /// The additive decode attention mask a V4 step binds over its fixed-capacity cache: the
        /// sliding window ending at `position`, everything else masked out.
        fn decode_window_mask(
            cfg: &DeepseekV4Config,
            capacity: usize,
            position: usize,
        ) -> Vec<f32> {
            let start = position.saturating_sub(cfg.sliding_window - 1);
            (0..capacity)
                .map(|slot| {
                    if start <= slot && slot <= position {
                        0.0
                    } else {
                        -1.0e9
                    }
                })
                .collect()
        }

        /// Step a V4 decode graph through `tokens`, feeding each step's published state into the next, and return
        /// every step's logits. `named_const` supplies only the f32 constants a tracer renames or computes;
        /// anything it declines falls back to `weights`. Packed components, the exact route table and every I32
        /// slot come from `bind_exact`; the `hca`/`csa.window_positions` step inputs and the additive mask are built
        /// here, and `causal.iota` is an in-graph `iota` (card 550a).
        pub(super) fn decode_logits(
            cfg: &DeepseekV4Config,
            g: &Graph<ValidationOutputs>,
            moe: &MoeFixture,
            weights: &HashMap<String, Vec<f32>>,
            tokens: &[usize],
            mut named_const: impl FnMut(&str) -> Option<Vec<f32>>,
        ) -> Vec<Vec<f32>> {
            let mut state: Vec<Value> = g
                .state
                .iter()
                .map(|&(state_in, _)| {
                    Value::Host(poot_tensor::HostTensor::zeros(
                        g.aval(state_in).shape.clone(),
                    ))
                })
                .collect();
            let mut all_logits = Vec::with_capacity(tokens.len());
            for (position, &token) in tokens.iter().enumerate() {
                let stepped = [token];
                let step = V4StepInputs {
                    tokens: &stepped,
                    position,
                };
                let mut inputs: HashMap<poot_graph_ir::ValueId, Value> = HashMap::new();
                for &id in &g.inputs {
                    let meta = g.meta(id);
                    if let Some(value) = moe.bind_exact(meta, step) {
                        inputs.insert(id, value);
                        continue;
                    }
                    let value = match meta.storage {
                        Storage::Slot(Slot::Mask) => Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            decode_window_mask(cfg, meta.aval.numel(), position),
                        )),
                        Storage::Slot(Slot::Activation) => {
                            let name = meta.name.as_deref().expect("activation without a name");
                            let rate = if name == "activation.hca.window_positions" {
                                cfg.hca_compress_rate
                            } else {
                                assert_eq!(
                                    name, "activation.csa.window_positions",
                                    "unexpected activation slot {name}"
                                );
                                cfg.csa_compress_rate
                            };
                            let windows = meta.aval.shape[0];
                            Value::Host(poot_tensor::HostTensor::f32(
                                meta.aval.shape.clone(),
                                (0..windows).map(|wi| (wi * rate) as f32).collect(),
                            ))
                        }
                        Storage::State => continue,
                        Storage::Const => {
                            let name = meta.name.as_deref().expect("const without a name");
                            bind_const(
                                name,
                                meta.aval.shape.clone(),
                                meta.aval.dtype,
                                named_const(name)
                                    .unwrap_or_else(|| crate::model_fixture_data(weights, name)),
                            )
                        }
                        other => panic!("unexpected storage {other:?}"),
                    };
                    inputs.insert(id, value);
                }
                for (index, &(state_in, _)) in g.state.iter().enumerate() {
                    inputs.insert(state_in, state[index].clone());
                }

                let (logits, next) = eval_graph_with_state(cfg, g, &inputs)
                    .unwrap_or_else(|error| panic!("decode step {position}: {error}"));
                state = next;
                let logits = dense(&logits);
                assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
                all_logits.push(logits.as_f32().unwrap().to_vec());
            }
            all_logits
        }

        /// Compare a decode run's per-step logits against the independent reference at the tolerance every V4
        /// oracle uses, plus `abs_floor` (`V4_HYBRID_STACK_ABS_FLOOR` for the hybrid 3-layer chain, `0.0` for
        /// every other caller).
        pub(super) fn assert_decode_logits(
            got: &[Vec<f32>],
            want: &[Vec<f32>],
            what: &str,
            abs_floor: f32,
        ) {
            assert_eq!(got.len(), want.len(), "{what}: one logits row per step");
            for (position, (got, want)) in got.iter().zip(want).enumerate() {
                assert_eq!(got.len(), want.len(), "{what} step {position}: vocab");
                for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-4 * want.abs().max(1e-3) + abs_floor,
                        "{what} step {position} logit {index}: graph={got} ref={want}"
                    );
                }
            }
        }

        /// The dense half of an evaluated `Value`. A V4 logits root is always dense.
        pub(super) fn dense(value: &Value) -> &poot_tensor::HostTensor {
            match value {
                Value::Host(tensor) => tensor,
                other => panic!("expected a dense result, got {other:?}"),
            }
        }

        /// One fixture's source plan, packed owners and hash table, so an oracle and a graph binder each take one
        /// parameter. The plan is the one the tracer builds, so names, roles and descriptors agree by
        /// construction.
        pub(super) struct MoeFixture {
            plan: DeepseekV4SourcePlan,
            owners: HashMap<String, Arc<PackedPayload>>,
            table: Vec<i32>,
            o_groups: usize,
        }

        impl MoeFixture {
            /// A fixture for a single-attention-kind model. The hybrid-stack tests pass their own
            /// schedule through [`MoeFixture::for_schedule`].
            pub(super) fn new(cfg: &DeepseekV4Config) -> Self {
                Self::for_schedule(cfg, &vec![V4LayerKind::Sliding; cfg.layers])
            }

            pub(super) fn for_schedule(cfg: &DeepseekV4Config, schedule: &[V4LayerKind]) -> Self {
                let plan = DeepseekV4SourcePlan::new(cfg, schedule).expect("fixture plan");
                Self {
                    owners: plan_owners(&plan),
                    plan,
                    table: tid2eid(cfg),
                    o_groups: cfg.o_groups,
                }
            }

            pub(super) fn owner(&self, linear_id: &str) -> &Arc<PackedPayload> {
                self.owners
                    .get(linear_id)
                    .unwrap_or_else(|| panic!("no packed owner for {linear_id}"))
            }

            /// One packed row's decoded `[out, K]` logical weight, for a reference oracle.
            pub(super) fn weight(&self, linear_id: &str) -> Vec<f32> {
                decoded_weight(self.owner(linear_id))
            }

            /// `attn.wo_a`'s grouped weight view for `layer`, re-derived on the host from the owner pair the graph
            /// consumes through Card 385's `ops::packed_block_diagonal_linear`.
            pub(super) fn grouped_wo_a(&self, layer: usize) -> Vec<f32> {
                let row = self
                    .plan
                    .packed(layer, V4PackedRole::WoA)
                    .expect("wo_a row");
                grouped_dequant(self.owner(&row.linear_id), self.o_groups)
            }

            pub(super) fn table(&self) -> &[i32] {
                &self.table
            }

            /// The independent routed-plus-shared reference for one layer and token.
            pub(super) fn ffn_ref(
                &self,
                cfg: &DeepseekV4Config,
                dense: &HashMap<String, Vec<f32>>,
                li: usize,
                token: usize,
                x: &[f32],
            ) -> Vec<f32> {
                routed_moe_ffn_ref(cfg, li, token, x, dense, &self.owners, &self.table)
            }

            /// Bind one graph constant only the MoE surface can supply: a packed source component, the exact I32
            /// hash table, or a host iota. Every other constant stays on the fixture's dense path.
            pub(super) fn bind_exact(
                &self,
                meta: &poot_graph_ir::ValueMeta,
                step: V4StepInputs<'_>,
            ) -> Option<Value> {
                // An I32 graph input needs the exact carrier on the `Value` path. Only `Slot::Token`/`Slot::Pos`
                // have an f32 escape hatch, which would leave `Slot::SeqLen` unbindable; answer every I32 slot here.
                match meta.storage {
                    Storage::Slot(Slot::Token) => {
                        let words = step.tokens.iter().map(|&t| t as i32).collect();
                        return Some(exact_i32(meta.aval.shape.clone(), words));
                    }
                    Storage::Slot(Slot::Pos) => {
                        return Some(exact_i32(
                            meta.aval.shape.clone(),
                            vec![step.position as i32],
                        ));
                    }
                    Storage::Slot(Slot::SeqLen) => {
                        return Some(exact_i32(
                            meta.aval.shape.clone(),
                            vec![step.position as i32 + 1],
                        ));
                    }
                    Storage::Const => {}
                    _ => return None,
                }
                let name = meta.name.as_deref()?;
                if let Some(source) = poot_graph_ir::PackedSourceName::parse(name) {
                    return Some(Value::Packed(PackedComponentRef::new(
                        Arc::clone(self.owner(source.linear_id())),
                        source.role(),
                    )));
                }
                if name.ends_with("ffn.gate.tid2eid") {
                    let view = poot_eval::ExactI32TensorView::try_from_words(
                        meta.aval.shape.clone(),
                        Arc::from(self.table.clone()),
                    )
                    .expect("fixture exact I32 table");
                    return Some(Value::from(view));
                }
                None
            }
        }

        /// The dense MoE rows one layer carries: the router projection and, for a score-routed layer, its
        /// correction bias. Only the router-focused fixtures use this; full-graph builders take every dense row
        /// from the plan ([`super::plan_dense_weights`]).
        pub(super) fn dense_rows(
            cfg: &DeepseekV4Config,
            li: usize,
            make: impl Fn(&str, usize) -> Vec<f32>,
        ) -> Vec<(String, Vec<f32>)> {
            let gate = format!("layers.{li}.ffn.gate.weight");
            let mut rows = vec![(gate.clone(), make(&gate, cfg.hidden * cfg.routed_experts))];
            if cfg.router_kind(li) == V4RouterKind::Score {
                let bias = format!("layers.{li}.ffn.gate.bias");
                rows.push((bias.clone(), make(&bias, cfg.routed_experts)));
            }
            rows
        }

        /// The independent routed-plus-shared MoE reference for one layer and token, written from the semantic
        /// contract: `sqrt(softplus(logits))`, ids from the table or a bias-selected stable top-k, weights
        /// gathered from the UNBIASED scores, normalized over the selected ids and scaled, then
        /// `w2(silu(clamp_max(w1(x),L)) * clamp(w3(x),-L,L))` per expert, accumulated, plus one unweighted shared
        /// expert.
        pub(super) fn routed_moe_ffn_ref(
            cfg: &DeepseekV4Config,
            li: usize,
            token: usize,
            x: &[f32],
            dense: &HashMap<String, Vec<f32>>,
            owners: &HashMap<String, Arc<PackedPayload>>,
            table: &[i32],
        ) -> Vec<f32> {
            let (h, e, k, inter) = (
                cfg.hidden,
                cfg.routed_experts,
                cfg.experts_per_tok,
                cfg.moe_intermediate,
            );
            // `ffn.gate.weight` is the checkpoint's own `[routed_experts, hidden]` row.
            let gate_w = &dense[&format!("layers.{li}.ffn.gate.weight")];
            let logits: Vec<f32> = (0..e)
                .map(|expert| (0..h).map(|i| x[i] * gate_w[expert * h + i]).sum::<f32>())
                .collect();
            let raw: Vec<f32> = logits
                .iter()
                .map(|&v| (1.0 + v.exp()).ln().sqrt())
                .collect();

            let ids: Vec<usize> = match cfg.router_kind(li) {
                V4RouterKind::Hash => (0..k)
                    .map(|slot| table[token * k + slot] as usize)
                    .collect(),
                V4RouterKind::Score => {
                    let bias = &dense[&format!("layers.{li}.ffn.gate.bias")];
                    let mut order: Vec<usize> = (0..e).collect();
                    order.sort_by(|&a, &b| {
                        let (sa, sb) = (raw[a] + bias[a], raw[b] + bias[b]);
                        sb.partial_cmp(&sa)
                            .expect("finite fixture scores")
                            .then(a.cmp(&b))
                    });
                    order[..k].to_vec()
                }
            };
            let selected: Vec<f32> = ids.iter().map(|&expert| raw[expert]).collect();
            let sum: f32 = selected.iter().sum();
            let weights: Vec<f32> = selected
                .iter()
                .map(|&value| value / sum * cfg.route_scale)
                .collect();

            let expert_out = |linear_id: &str, act: &[f32], out_dim: usize, in_dim: usize| {
                let w = decoded_weight(&owners[linear_id]);
                (0..out_dim)
                    .map(|o| (0..in_dim).map(|i| act[i] * w[o * in_dim + i]).sum::<f32>())
                    .collect::<Vec<f32>>()
            };
            let swiglu = |gate: Vec<f32>, up: Vec<f32>| -> Vec<f32> {
                let limit = cfg.swiglu_limit;
                gate.iter()
                    .zip(&up)
                    .map(|(&g, &u)| silu_ref(g.min(limit)) * u.clamp(-limit, limit))
                    .collect()
            };

            let mut out = vec![0.0f32; h];
            for (slot, &expert) in ids.iter().enumerate() {
                let prefix = format!("layers.{li}.ffn.experts.{expert}");
                let act = swiglu(
                    expert_out(&format!("{prefix}.w1"), x, inter, h),
                    expert_out(&format!("{prefix}.w3"), x, inter, h),
                );
                let down = expert_out(&format!("{prefix}.w2"), &act, h, inter);
                for (o, value) in down.into_iter().enumerate() {
                    out[o] += weights[slot] * value;
                }
            }
            let shared = format!("layers.{li}.ffn.shared_experts");
            let act = swiglu(
                expert_out(&format!("{shared}.w1"), x, inter, h),
                expert_out(&format!("{shared}.w3"), x, inter, h),
            );
            for (o, value) in expert_out(&format!("{shared}.w2"), &act, h, inter)
                .into_iter()
                .enumerate()
            {
                out[o] += value;
            }
            out
        }
    }

    use poot_test_util::fill;

    use poot_test_util::seed_of;

    fn weight(name: &str, n: usize, ln_gamma: bool) -> Vec<f32> {
        let raw = fill(n, seed_of(name));
        if ln_gamma {
            raw.iter().map(|v| 1.0 + v * 0.05).collect()
        } else {
            raw.iter().map(|v| v * 0.1).collect()
        }
    }

    // ---- Independent, hand-written mHC reference (plain nested loops; does not call
    // `hyper_connection`/`hyper_connection_combine`/`hyper_head`, per spec 281 SC-002/SC-003). Operates on
    // `Vec<Vec<f32>>` streams (`[hc][hidden]`), one token at a time.

    fn sigmoid_ref(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn rmsnorm_no_weight_ref(x: &[f32], eps: f32) -> Vec<f32> {
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let den = (ms + eps).sqrt();
        x.iter().map(|v| v / den).collect()
    }

    fn rmsnorm_ref(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let den = (ms + eps).sqrt();
        (0..x.len()).map(|i| x[i] / den * w[i]).collect()
    }

    use poot_test_util::linear_ref;

    /// One stored `[out, in]` row as the `[in, out]` matrix a tracer-layout reference helper reads: the
    /// reference twin of the graph's transpose equation ([`v4_mhc_projection`], [`v4_source_linear`]), for
    /// [`hyper_connection_ref`] and [`hyper_head_ref`].
    fn transpose_ref(w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                t[i * out_dim + o] = w[o * in_dim + i];
            }
        }
        t
    }

    /// One mHC site's projection from a fixture map, in the `[in, out]` orientation
    /// [`hyper_connection_ref`] and [`hyper_head_ref`] read. The plan stores `[mix, streams * hidden]`.
    fn mhc_projection_ref(
        w: &HashMap<String, Vec<f32>>,
        name: &str,
        mix_dim: usize,
        stream_width: usize,
    ) -> Vec<f32> {
        transpose_ref(w.get(name).unwrap(), mix_dim, stream_width)
    }

    /// [`linear_ref`]'s checkpoint-orientation twin: `w` is the stored `[out, in]` row-major weight, which the
    /// graph transposes before the matmul. A separate function rather than a flag so call sites cannot read as
    /// the other layout.
    fn linear_ref_out_in(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        (0..out_dim)
            .map(|o| (0..in_dim).map(|i| x[i] * w[o * in_dim + i]).sum())
            .collect()
    }

    use poot_test_util::silu_ref;

    /// Independent mHC reference: `streams[hc][hidden]` -> `(pre[hc], post[hc], comb[hc][hc],
    /// collapsed[hidden])`. `fn_w[hc*hidden][(2+hc)*hc]`, `base[(2+hc)*hc]`, `scale[3]` use the flat row-major
    /// `[in,out]` layout of `linear_ref`.
    ///
    /// Written from HF `transformers` `models/deepseek_v4/modeling_deepseek_v4.py` at `0b05eec`,
    /// `DeepseekV4HyperConnection.forward`: split `[pre, post, comb]`, `softmax(-1) + eps`, one column
    /// normalization, then `hc_sinkhorn_iters - 1` rounds of rows then columns. HF stores `fn` as
    /// `[mix, hc*hidden]`; this reference takes the graph constant's `[in, out]` layout.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn hyper_connection_ref(
        streams: &[Vec<f32>],
        fn_w: &[f32],
        base: &[f32],
        scale: &[f32],
        hc: usize,
        hidden: usize,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>) {
        let flat: Vec<f32> = streams.iter().flatten().copied().collect();
        let normed = rmsnorm_no_weight_ref(&flat, rms_eps);
        let mix = linear_ref(&normed, fn_w, hc * hidden, (2 + hc) * hc);

        let pre: Vec<f32> = (0..hc)
            .map(|i| sigmoid_ref(mix[i] * scale[0] + base[i]) + hc_eps)
            .collect();
        let post: Vec<f32> = (0..hc)
            .map(|i| 2.0 * sigmoid_ref(mix[hc + i] * scale[1] + base[hc + i]))
            .collect();

        let mut comb_logits = vec![vec![0.0f32; hc]; hc];
        for j in 0..hc {
            for k in 0..hc {
                let idx = 2 * hc + j * hc + k;
                comb_logits[j][k] = mix[idx] * scale[2] + base[idx];
            }
        }
        let mut comb = vec![vec![0.0f32; hc]; hc];
        for j in 0..hc {
            let m = comb_logits[j].iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = comb_logits[j].iter().map(|v| (v - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for k in 0..hc {
                comb[j][k] = exps[k] / sum + hc_eps;
            }
        }
        // Initial column-normalize, then `sinkhorn_iters - 1` rounds of (row-normalize, column-normalize),
        // matching the reference's asymmetric round count.
        let col_normalize = |m: &mut Vec<Vec<f32>>| {
            for k in 0..hc {
                let s: f32 = (0..hc).map(|j| m[j][k]).sum::<f32>() + hc_eps;
                for j in 0..hc {
                    m[j][k] /= s;
                }
            }
        };
        let row_normalize = |m: &mut Vec<Vec<f32>>| {
            for j in 0..hc {
                let s: f32 = m[j].iter().sum::<f32>() + hc_eps;
                for k in 0..hc {
                    m[j][k] /= s;
                }
            }
        };
        col_normalize(&mut comb);
        for _ in 0..sinkhorn_iters.saturating_sub(1) {
            row_normalize(&mut comb);
            col_normalize(&mut comb);
        }

        let mut collapsed = vec![0.0f32; hidden];
        for (j, stream) in streams.iter().enumerate() {
            for (d, &v) in stream.iter().enumerate() {
                collapsed[d] += pre[j] * v;
            }
        }

        (pre, post, comb, collapsed)
    }

    /// HF `DeepseekV4DecoderLayer.forward` (same file and revision as [`hyper_connection_ref`]):
    /// `post[..., None] * sublayer_out[..., None, :] + matmul(comb.transpose(-1, -2), residual)`.
    fn hyper_connection_combine_ref(
        post: &[f32],
        comb: &[Vec<f32>],
        sublayer_out: &[f32],
        residual: &[Vec<f32>],
        hc: usize,
        hidden: usize,
    ) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0f32; hidden]; hc];
        for k in 0..hc {
            for d in 0..hidden {
                let outer = post[k] * sublayer_out[d];
                // comb^T: sum_j comb[j][k] * residual[j][d]
                let mix: f32 = (0..hc).map(|j| comb[j][k] * residual[j][d]).sum();
                out[k][d] = outer + mix;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn hyper_head_ref(
        streams: &[Vec<f32>],
        fn_w: &[f32],
        base: &[f32],
        scale: f32,
        hc: usize,
        hidden: usize,
        rms_eps: f32,
        hc_eps: f32,
    ) -> Vec<f32> {
        let flat: Vec<f32> = streams.iter().flatten().copied().collect();
        let normed = rmsnorm_no_weight_ref(&flat, rms_eps);
        let mixes = linear_ref(&normed, fn_w, hc * hidden, hc);
        let pre: Vec<f32> = (0..hc)
            .map(|i| sigmoid_ref(mixes[i] * scale + base[i]) + hc_eps)
            .collect();
        let mut out = vec![0.0f32; hidden];
        for (j, stream) in streams.iter().enumerate() {
            for (d, &v) in stream.iter().enumerate() {
                out[d] += pre[j] * v;
            }
        }
        out
    }

    #[test]
    fn hyper_connection_matches_independent_reference_and_comb_is_doubly_stochastic() {
        let hc = 3usize;
        let hidden = 5usize;
        let rms_eps = 1e-5f32;
        let hc_eps = 1e-6f32;
        for sinkhorn_iters in [1usize, 3, 20] {
            let mix_dim = (2 + hc) * hc;
            let fn_w = weight("hc.fn", hc * hidden * mix_dim, false);
            let base = weight("hc.base", mix_dim, false);
            let scale = weight("hc.scale", 3, false);
            let streams: Vec<Vec<f32>> = (0..hc)
                .map(|j| fill(hidden, seed_of(&format!("stream{j}"))))
                .collect();

            let (_pre, post_ref, comb_ref, collapsed_ref) = hyper_connection_ref(
                &streams,
                &fn_w,
                &base,
                &scale,
                hc,
                hidden,
                rms_eps,
                hc_eps,
                sinkhorn_iters,
            );

            // Graph side: a 1-layer graph that runs `hyper_connection` and outputs `post`/`comb`/`collapsed`
            // concatenated, so `poot-eval` executes the same composition the tracers use.
            let b = Builder::new();
            let streams_c = b.constant("streams", TensorType::f32(vec![1, 1, hc, hidden]));
            let fn_c = b.constant("fn_w", TensorType::f32(vec![hc * hidden, mix_dim]));
            let base_c = b.constant("base", TensorType::f32(vec![mix_dim]));
            let scale_c = b.constant("scale", TensorType::f32(vec![3]));
            let (post_t, comb_t, collapsed_t) = hyper_connection(
                &b,
                streams_c,
                fn_c,
                base_c,
                scale_c,
                hc,
                hidden,
                1,
                rms_eps,
                hc_eps,
                sinkhorn_iters,
            );
            let post_flat = b.reshape(post_t, vec![hc]);
            let comb_flat = b.reshape(comb_t, vec![hc * hc]);
            let collapsed_flat = b.reshape(collapsed_t, vec![hidden]);
            let out = b.concat(0, &[post_flat, comb_flat, collapsed_flat]);
            let g = b.finish(out);

            let mut w: HashMap<String, Vec<f32>> = HashMap::new();
            w.insert(
                "streams".to_string(),
                streams.iter().flatten().copied().collect(),
            );
            w.insert("fn_w".to_string(), fn_w.clone());
            w.insert("base".to_string(), base.clone());
            w.insert("scale".to_string(), scale.clone());

            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let name = meta.name.as_deref().expect("const without a name");
                let data = w
                    .get(name)
                    .unwrap_or_else(|| panic!("no weight for {name}"));
                inputs.insert(
                    id,
                    poot_eval::Value::from(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        data.clone(),
                    )),
                );
            }
            let result = poot_eval::eval(
                &g,
                &inputs,
                poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
            )
            .expect("eval hyper_connection graph")
            .output
            .into_host()
            .expect("hyper_connection output is dense");
            let got = &result.as_f32().unwrap();

            for (i, &v) in post_ref.iter().enumerate() {
                let g = got[i];
                assert!(
                    (g - v).abs() <= 1e-4 * v.abs().max(1e-3),
                    "post[{i}] iters={sinkhorn_iters}: graph={g} ref={v}"
                );
            }
            for j in 0..hc {
                for k in 0..hc {
                    let v = comb_ref[j][k];
                    let g = got[hc + j * hc + k];
                    assert!(
                        (g - v).abs() <= 1e-4 * v.abs().max(1e-3),
                        "comb[{j}][{k}] iters={sinkhorn_iters}: graph={g} ref={v}"
                    );
                }
            }
            for (i, &v) in collapsed_ref.iter().enumerate() {
                let g = got[hc + hc * hc + i];
                assert!(
                    (g - v).abs() <= 1e-4 * v.abs().max(1e-3),
                    "collapsed[{i}] iters={sinkhorn_iters}: graph={g} ref={v}"
                );
            }

            // Doubly-stochastic property (SC-002): at the real `hc_sinkhorn_iters=20` every row and column of
            // `comb_ref` sums to ~1.0. Not checked at iters=1/3 (not converged there); the iters=1 case exists
            // to catch a symmetric-round-count bug.
            if sinkhorn_iters == 20 {
                for (j, row) in comb_ref.iter().enumerate() {
                    let row_sum: f32 = row.iter().sum();
                    assert!((row_sum - 1.0).abs() < 1e-3, "row {j} sum={row_sum}");
                }
                for k in 0..hc {
                    let col_sum: f32 = comb_ref.iter().map(|row| row[k]).sum();
                    assert!((col_sum - 1.0).abs() < 1e-3, "col {k} sum={col_sum}");
                }
            }
        }
    }

    /// One whole mHC site (`hyper_connection` then `hyper_connection_combine`) against the plain-loop
    /// reference. The fixture is built so the mixing direction shows: `hc = 4` as in the real config (a
    /// 2x2 doubly-stochastic matrix is symmetric, so `tiny_cfg`'s `hc_mult = 2` hides `comb` vs `comb.T`),
    /// distinct streams per token, wide `comb` logits, and three Sinkhorn rounds, far from converged. The
    /// sublayer is the identity on `collapsed`, so the output also depends on `pre`.
    #[test]
    fn hyper_connection_site_mixes_residual_through_comb_transpose() {
        let hc = 4usize;
        let hidden = 5usize;
        let seq = 2usize;
        let rms_eps = 1e-5f32;
        let hc_eps = 1e-6f32;
        let sinkhorn_iters = 3usize;
        let mix_dim = (2 + hc) * hc;
        let fn_w = weight("site.fn", hc * hidden * mix_dim, false);
        let base: Vec<f32> = fill(mix_dim, seed_of("site.base"))
            .iter()
            .map(|v| 2.0 * v)
            .collect();
        let scale = vec![0.8f32, -0.6, 1.5];
        let streams: Vec<Vec<Vec<f32>>> = (0..seq)
            .map(|t| {
                (0..hc)
                    .map(|j| fill(hidden, seed_of(&format!("site.stream{t}.{j}"))))
                    .collect()
            })
            .collect();

        // Output `[seq][hc][hidden]`, flattened. With `transpose_comb` the reference combine receives
        // `comb.T`, so it mixes through `comb`: the wrong direction.
        let site_ref = |iters: usize, transpose_comb: bool| -> Vec<f32> {
            streams
                .iter()
                .flat_map(|token| {
                    let (_pre, post, comb, collapsed) = hyper_connection_ref(
                        token, &fn_w, &base, &scale, hc, hidden, rms_eps, hc_eps, iters,
                    );
                    let comb: Vec<Vec<f32>> = if transpose_comb {
                        (0..hc)
                            .map(|j| (0..hc).map(|k| comb[k][j]).collect())
                            .collect()
                    } else {
                        comb
                    };
                    hyper_connection_combine_ref(&post, &comb, &collapsed, token, hc, hidden)
                        .into_iter()
                        .flatten()
                })
                .collect()
        };
        // The graph comparison below allows `tol(v)` per element.
        let tol = |v: f32| 1e-4 * v.abs().max(1.0);
        let want = site_ref(sinkhorn_iters, false);

        // The fixture must separate the mutations this test guards: a mutated reference must miss the
        // comparison tolerance by at least 10x somewhere. Guarded here: the wrong mixing direction and
        // one fewer Sinkhorn round. Row-first normalization ends on rows summing to one while
        // column-first ends on columns, so the rows must still be off by 10x the tolerance floor.
        let separation = |mutated: &[f32]| {
            want.iter()
                .zip(mutated)
                .map(|(&v, &m)| (v - m).abs() / tol(v))
                .fold(0.0f32, f32::max)
        };
        let direction = separation(&site_ref(sinkhorn_iters, true));
        assert!(
            direction > 10.0,
            "comb vs comb.T separation {direction}x tol"
        );
        let rounds = separation(&site_ref(sinkhorn_iters - 1, false));
        assert!(rounds > 10.0, "one-fewer-round separation {rounds}x tol");
        for (t, token) in streams.iter().enumerate() {
            let (_pre, _post, comb, _collapsed) = hyper_connection_ref(
                token,
                &fn_w,
                &base,
                &scale,
                hc,
                hidden,
                rms_eps,
                hc_eps,
                sinkhorn_iters,
            );
            let row_error = comb
                .iter()
                .map(|row| (row.iter().sum::<f32>() - 1.0).abs())
                .fold(0.0f32, f32::max);
            assert!(
                row_error > 10.0 * tol(0.0),
                "token {t}: comb rows converged, error {row_error}"
            );
        }

        let b = Builder::new();
        let streams_c = b.constant("streams", TensorType::f32(vec![1, seq, hc, hidden]));
        let fn_c = b.constant("fn_w", TensorType::f32(vec![hc * hidden, mix_dim]));
        let base_c = b.constant("base", TensorType::f32(vec![mix_dim]));
        let scale_c = b.constant("scale", TensorType::f32(vec![3]));
        let (post_t, comb_t, collapsed_t) = hyper_connection(
            &b,
            streams_c,
            fn_c,
            base_c,
            scale_c,
            hc,
            hidden,
            seq,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
        );
        let out =
            hyper_connection_combine(&b, post_t, comb_t, collapsed_t, streams_c, hc, seq, hidden);
        let g = b.finish(out);
        assert_eq!(g.aval(g.output).shape, vec![1, seq, hc, hidden]);

        let mut w: HashMap<String, Vec<f32>> = HashMap::new();
        w.insert(
            "streams".to_string(),
            streams.iter().flatten().flatten().copied().collect(),
        );
        w.insert("fn_w".to_string(), fn_w.clone());
        w.insert("base".to_string(), base.clone());
        w.insert("scale".to_string(), scale.clone());

        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data = w
                .get(name)
                .unwrap_or_else(|| panic!("no weight for {name}"));
            inputs.insert(
                id,
                poot_eval::Value::from(poot_tensor::HostTensor::f32(
                    meta.aval.shape.clone(),
                    data.clone(),
                )),
            );
        }
        let result = poot_eval::eval(
            &g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval mHC site graph")
        .output
        .into_host()
        .expect("mHC site output is dense");

        assert_eq!(result.as_f32().unwrap().len(), want.len());
        for (i, (&got, &v)) in result.as_f32().unwrap().iter().zip(&want).enumerate() {
            let (t, k, d) = (i / (hc * hidden), i / hidden % hc, i % hidden);
            assert!(
                (got - v).abs() <= tol(v),
                "token {t} stream {k} dim {d}: graph={got} ref={v}"
            );
        }
    }

    /// Independent reference for [`hca_softmax_gate_pool`] (`DeepseekV4HCACompressor.forward` eqs. 20-23, layout
    /// `[n_windows][compress_rate][head_dim]` nested `Vec`s; does not call the graph function): softmax over
    /// the window-position axis independently per `head_dim` channel (`dim=2`), not over `head_dim`.
    fn hca_softmax_gate_pool_ref(
        kv: &[Vec<Vec<f32>>],
        gate: &[Vec<Vec<f32>>],
        position_bias: &[f32], // [compress_rate * head_dim], row-major [compress_rate][head_dim]
        kv_norm_w: &[f32],
        compress_rate: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<Vec<f32>> {
        kv.iter()
            .zip(gate.iter())
            .map(|(kv_win, gate_win)| {
                let mut pooled = vec![0.0f32; head_dim];
                for ch in 0..head_dim {
                    let logits: Vec<f32> = (0..compress_rate)
                        .map(|j| gate_win[j][ch] + position_bias[j * head_dim + ch])
                        .collect();
                    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                    let exps: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
                    let sum: f32 = exps.iter().sum();
                    let mut acc = 0.0f32;
                    for j in 0..compress_rate {
                        acc += (exps[j] / sum) * kv_win[j][ch];
                    }
                    pooled[ch] = acc;
                }
                rmsnorm_ref(&pooled, kv_norm_w, eps)
            })
            .collect()
    }

    #[test]
    fn hca_softmax_gate_pool_matches_independent_reference() {
        let n_windows = 3usize;
        let compress_rate = 4usize;
        let head_dim = 6usize;
        let eps = 1e-5f32;

        let kv: Vec<Vec<Vec<f32>>> = (0..n_windows)
            .map(|w| {
                (0..compress_rate)
                    .map(|j| fill(head_dim, seed_of(&format!("kv{w}_{j}"))))
                    .collect()
            })
            .collect();
        let gate: Vec<Vec<Vec<f32>>> = (0..n_windows)
            .map(|w| {
                (0..compress_rate)
                    .map(|j| fill(head_dim, seed_of(&format!("gate{w}_{j}"))))
                    .collect()
            })
            .collect();
        let position_bias = weight("pb", compress_rate * head_dim, false);
        let kv_norm_w = weight("kvnorm", head_dim, true);

        let want = hca_softmax_gate_pool_ref(
            &kv,
            &gate,
            &position_bias,
            &kv_norm_w,
            compress_rate,
            head_dim,
            eps,
        );

        let b = Builder::new();
        let kv_c = b.constant(
            "kv",
            TensorType::f32(vec![1, n_windows, compress_rate, head_dim]),
        );
        let gate_c = b.constant(
            "gate",
            TensorType::f32(vec![1, n_windows, compress_rate, head_dim]),
        );
        let pb_c = b.constant("pb", TensorType::f32(vec![compress_rate, head_dim]));
        let kvnorm_c = b.constant("kvnorm", TensorType::f32(vec![head_dim]));
        let pooled = hca_softmax_gate_pool(
            &b,
            kv_c,
            gate_c,
            pb_c,
            kvnorm_c,
            compress_rate,
            head_dim,
            eps,
        );
        let g = b.finish(pooled);

        let kv_flat: Vec<f32> = kv.iter().flatten().flatten().copied().collect();
        let gate_flat: Vec<f32> = gate.iter().flatten().flatten().copied().collect();
        let mut w: HashMap<String, Vec<f32>> = HashMap::new();
        w.insert("kv".to_string(), kv_flat);
        w.insert("gate".to_string(), gate_flat);
        w.insert("pb".to_string(), position_bias);
        w.insert("kvnorm".to_string(), kv_norm_w);

        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data = w
                .get(name)
                .unwrap_or_else(|| panic!("no weight for {name}"));
            inputs.insert(
                id,
                poot_eval::Value::from(poot_tensor::HostTensor::f32(
                    meta.aval.shape.clone(),
                    data.clone(),
                )),
            );
        }
        let result = poot_eval::eval(
            &g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval hca_softmax_gate_pool graph")
        .output
        .into_host()
        .expect("hca_softmax_gate_pool output is dense");
        let got = &result.as_f32().unwrap();

        for w_i in 0..n_windows {
            for ch in 0..head_dim {
                let (gv, wv) = (got[w_i * head_dim + ch], want[w_i][ch]);
                assert!(
                    (gv - wv).abs() <= 1e-4 * wv.abs().max(1e-3),
                    "window {w_i} channel {ch}: graph={gv} ref={wv}"
                );
            }
        }
    }

    // ---- Full-model independent reference (attention core + mHC + dense FFN), toy dims of `tiny_cfg()`,
    // hand-rolled without calling any graph-side composition.

    /// Every dense checkpoint row a plan names, with deterministic fixture values. Driven by the plan, so a
    /// fixture cannot miss a row, misspell one, or give an HCA layer a CSA layer's compressor width. Norm rows
    /// are one-centered. BF16 rows are truncated to what a BF16 checkpoint can store, so a reference reads
    /// the value the row holds. Packed rows and the exact-I32 hash table come from [`moe_fixture::MoeFixture`].
    fn plan_dense_weights(
        cfg: &DeepseekV4Config,
        schedule: &[V4LayerKind],
    ) -> HashMap<String, Vec<f32>> {
        let plan = DeepseekV4SourcePlan::new(cfg, schedule).expect("fixture source plan");
        plan.dense_sources()
            .filter(|row| !row.is_exact_i32())
            .map(|row| {
                let numel: usize = row.shape.iter().product();
                let values = weight(&row.name, numel, row.name.ends_with("norm.weight"));
                let values = if row.kind == poot_load::packed_safetensors::ExactSourceKind::Bf16 {
                    values
                        .into_iter()
                        .map(crate::test_support::safetensors::truncate_to_bf16)
                        .collect()
                } else {
                    values
                };
                (row.name.clone(), values)
            })
            .collect()
    }

    /// Every constant a V4 graph reads apart from its packed owners: the plan's dense rows for this schedule,
    /// plus the host tables.
    ///
    /// The schedule picks each layer's compressor width, since HCA and CSA share the compressor names and
    /// differ only in pooled width; an all-CSA map given to a mixed-schedule graph would hand an HCA layer a
    /// `[2 * head_dim, hidden]` row where the graph declares `[head_dim, hidden]`.
    ///
    /// `rope_theta` is the table under the main rope keys (one table per fixture, shared with
    /// [`attention_core_ref`] and redirected to the compress keys by the binder). A mixed-schedule test
    /// overrides it per layer.
    fn schedule_weights(
        cfg: &DeepseekV4Config,
        schedule: &[V4LayerKind],
        rope_theta: f32,
    ) -> HashMap<String, Vec<f32>> {
        let mut w = plan_dense_weights(cfg, schedule);
        let (cos, sin) =
            deepseek2_rope_tables_interleaved(cfg.max_pos, cfg.qk_rope_head_dim, rope_theta, None);
        w.insert(V4_ROPE_COS.to_string(), cos);
        w.insert(V4_ROPE_SIN.to_string(), sin);
        w
    }

    fn all_weights(cfg: &DeepseekV4Config) -> HashMap<String, Vec<f32>> {
        schedule_weights(cfg, &vec![V4LayerKind::Sliding; cfg.layers], cfg.rope_theta)
    }

    /// Independent reference for the interleaved-pair partial RoPE (trailing `rot`-wide slice, `nope`-leading
    /// passthrough), re-derived without `crate::deepseek2`, as in `crate::deepseek32`'s `cpu_oracle`.
    /// `inverse=true` negates `sin` (the output derotation).
    fn rope_interleaved_ref(
        row: &[f32],
        cos: &[f32],
        sin: &[f32],
        pos: usize,
        rot: usize,
        inverse: bool,
    ) -> Vec<f32> {
        let half = rot / 2;
        let sign = if inverse { -1.0 } else { 1.0 };
        let mut out = row.to_vec();
        for i in 0..half {
            let c = cos[pos * half + i];
            let s = sin[pos * half + i] * sign;
            let (a, bb) = (row[2 * i], row[2 * i + 1]);
            out[2 * i] = a * c - bb * s;
            out[2 * i + 1] = a * s + bb * c;
        }
        out
    }

    fn softmax_with_sink_ref(scores: &mut [f32], sink: f32) -> f32 {
        let m = scores.iter().cloned().fold(sink, f32::max);
        let mut denom = (sink - m).exp();
        for v in scores.iter_mut() {
            *v = (*v - m).exp();
            denom += *v;
        }
        denom
    }

    /// One `sliding_attention` layer's full attention core (independent reference): low-rank Q,
    /// single shared K==V head, trailing-slice interleaved partial RoPE (+ inverse derotation on the
    /// output), attention sinks, grouped output projection. `cache` is the causally-valid K==V prefix
    /// (already windowed by the caller to `sliding_window`), `[prefix][head_dim]`.
    #[allow(clippy::too_many_arguments)]
    fn attention_core_ref(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Vec<f32>>,
        moe: &moe_fixture::MoeFixture,
        li: usize,
        x_normed: &[f32],
        cache: &[Vec<f32>],
        qpos: usize,
    ) -> Vec<f32> {
        let (h, hq, d, rd) = (
            cfg.hidden,
            cfg.num_heads,
            cfg.head_dim,
            cfg.qk_rope_head_dim,
        );
        let nope_w = d - rd;
        let p = |s: &str| format!("layers.{li}.{s}");
        let cos = w.get("rope.cos").unwrap();
        let sin = w.get("rope.sin").unwrap();

        let q_a = linear_ref_out_in(x_normed, &moe.weight(&p("attn.wq_a")), h, cfg.q_lora_rank);
        let q_res = rmsnorm_ref(&q_a, w.get(&p("attn.q_norm.weight")).unwrap(), cfg.eps);
        let q_full = linear_ref_out_in(
            &q_res,
            &moe.weight(&p("attn.wq_b")),
            cfg.q_lora_rank,
            hq * d,
        );

        let mut heads_out = vec![vec![0.0f32; d]; hq];
        let sinks = w.get(&p("attn.attn_sink")).unwrap();
        for hix in 0..hq {
            let mut qh = q_full[hix * d..(hix + 1) * d].to_vec();
            qh = rmsnorm_no_weight_ref(&qh, cfg.eps); // q_b_norm, per head
            let mut q_rope = qh[nope_w..].to_vec();
            q_rope = rope_interleaved_ref(&q_rope, cos, sin, qpos, rd, false);
            let mut q_final = qh[..nope_w].to_vec();
            q_final.extend(q_rope);

            let mut scores: Vec<f32> = cache
                .iter()
                .map(|k| {
                    let dot: f32 = q_final.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                    dot * cfg.attn_scale()
                })
                .collect();
            let denom = softmax_with_sink_ref(&mut scores, sinks[hix]);
            let mut acc = vec![0.0f32; d];
            for (kpos, v) in cache.iter().enumerate() {
                let p_kv = scores[kpos] / denom;
                for dd in 0..d {
                    acc[dd] += p_kv * v[dd];
                }
            }
            heads_out[hix] = acc;
        }

        // Inverse derotation on the trailing rope slice of the attention OUTPUT.
        let mut attn_flat = vec![0.0f32; hq * d];
        for hix in 0..hq {
            let mut o_rope = heads_out[hix][nope_w..].to_vec();
            o_rope = rope_interleaved_ref(&o_rope, cos, sin, qpos, rd, true);
            for dd in 0..nope_w {
                attn_flat[hix * d + dd] = heads_out[hix][dd];
            }
            for dd in 0..rd {
                attn_flat[hix * d + nope_w + dd] = o_rope[dd];
            }
        }

        // Grouped output projection: the `attn.wo_a` owner pair the graph consumes through Card 385's
        // `ops::packed_block_diagonal_linear`, dequantized on the host into the grouped layout. Within one
        // group the slice is `[in_per_group, o_lora_rank]` row-major, the orientation `linear_ref` reads.
        let in_per_group = cfg.in_per_group();
        let wo_a = moe.grouped_wo_a(li);
        let mut grouped_out = vec![0.0f32; cfg.o_groups * cfg.o_lora_rank];
        for g in 0..cfg.o_groups {
            let x_g = &attn_flat[g * in_per_group..(g + 1) * in_per_group];
            let w_g =
                &wo_a[g * in_per_group * cfg.o_lora_rank..(g + 1) * in_per_group * cfg.o_lora_rank];
            let y_g = linear_ref(x_g, w_g, in_per_group, cfg.o_lora_rank);
            grouped_out[g * cfg.o_lora_rank..(g + 1) * cfg.o_lora_rank].copy_from_slice(&y_g);
        }
        linear_ref_out_in(
            &grouped_out,
            &moe.weight(&p("attn.wo_b")),
            cfg.o_groups * cfg.o_lora_rank,
            h,
        )
    }

    /// Full independent-reference decode: `n_steps` tokens, window `cfg.sliding_window` enforced by
    /// only keeping the last `sliding_window` cache entries.
    fn deepseek4_decode_ref(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Vec<f32>>,
        moe: &moe_fixture::MoeFixture,
        tokens: &[usize],
    ) -> Vec<Vec<f32>> {
        let (h, hc) = (cfg.hidden, cfg.hc_mult);
        let embed = w.get("embed.weight").unwrap();

        let mut caches: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut all_logits = Vec::new();

        for (qpos, &tok) in tokens.iter().enumerate() {
            let e = embed[tok * h..(tok + 1) * h].to_vec();
            let mut streams: Vec<Vec<f32>> = (0..hc).map(|_| e.clone()).collect();

            for (li, cache) in caches.iter_mut().enumerate() {
                let p = |s: &str| format!("layers.{li}.{s}");
                let (_pre, post, comb, collapsed) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_attn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_attn_base")).unwrap(),
                    w.get(&p("hc_attn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed =
                    rmsnorm_ref(&collapsed, w.get(&p("attn_norm.weight")).unwrap(), cfg.eps);

                let wkv = moe.weight(&p("attn.wkv"));
                let kv_raw = linear_ref_out_in(&normed, &wkv, cfg.hidden, cfg.head_dim);
                let kv_n = rmsnorm_ref(&kv_raw, w.get(&p("attn.kv_norm.weight")).unwrap(), cfg.eps);
                let nope_w = cfg.head_dim - cfg.qk_rope_head_dim;
                let mut kv_rope = kv_n[nope_w..].to_vec();
                kv_rope = rope_interleaved_ref(
                    &kv_rope,
                    w.get("rope.cos").unwrap(),
                    w.get("rope.sin").unwrap(),
                    qpos,
                    cfg.qk_rope_head_dim,
                    false,
                );
                let mut kv_new = kv_n[..nope_w].to_vec();
                kv_new.extend(kv_rope);
                cache.push(kv_new);
                let window_start = qpos.saturating_sub(cfg.sliding_window - 1);
                let cache_slice = &cache[window_start..=qpos];

                let attn_out = attention_core_ref(cfg, w, moe, li, &normed, cache_slice, qpos);
                streams = hyper_connection_combine_ref(&post, &comb, &attn_out, &streams, hc, h);

                let (_pre2, post2, comb2, collapsed2) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_ffn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_ffn_base")).unwrap(),
                    w.get(&p("hc_ffn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed2 =
                    rmsnorm_ref(&collapsed2, w.get(&p("ffn_norm.weight")).unwrap(), cfg.eps);
                let mlp_out = moe.ffn_ref(cfg, w, li, tok, &normed2);
                streams = hyper_connection_combine_ref(&post2, &comb2, &mlp_out, &streams, hc, h);
            }

            let hh_scale = w.get("hc_head_scale").unwrap()[0];
            let collapsed_final = hyper_head_ref(
                &streams,
                &mhc_projection_ref(w, "hc_head_fn", hc, hc * h),
                w.get("hc_head_base").unwrap(),
                hh_scale,
                hc,
                h,
                cfg.eps,
                cfg.hc_eps,
            );
            let xf = rmsnorm_ref(&collapsed_final, w.get("norm.weight").unwrap(), cfg.eps);
            let logits = linear_ref_out_in(&xf, w.get("head.weight").unwrap(), h, cfg.vocab);
            all_logits.push(logits);
        }
        all_logits
    }

    /// SC-004: a stepwise decode run against the independent reference, on Card 382's state-carrying exact-I32
    /// evaluator. Five steps with `sliding_window = 3` and physical capacity six, so the window floor is
    /// exercised past the window and the cache is written at five distinct exact positions.
    ///
    /// "window capacity" in the name is the sliding window, not the cache: nothing wraps (`Slot::Pos` is the
    /// absolute step, and a step at `cap` is a typed error). See the spec's "No wrap, and what one would cost".
    #[test]
    fn deepseek4_sliding_decode_matches_independent_reference_past_window_capacity() {
        let cfg = tiny_cfg();
        let cap = 6;
        let tokens = [1usize, 2, 3, 4, 5];
        assert!(
            tokens.len() > cfg.sliding_window,
            "the window floor must fire"
        );
        let w = all_weights(&cfg);
        let moe = moe_fixture::MoeFixture::new(&cfg);
        let want = deepseek4_decode_ref(&cfg, &w, &moe, &tokens);

        let g = trace_deepseek4_sliding_decode(cfg, cap).expect("sliding decode traces");
        let got = moe_fixture::decode_logits(&cfg, &g, &moe, &w, &tokens, |_| None);
        moe_fixture::assert_decode_logits(&got, &want, "sliding decode", 0.0);
    }

    /// Checked at every position, not only the last (Card 386 P2): prefill's mask/RoPE-table construction is
    /// code the per-step-checked decode oracle never runs. Reuses
    /// `trace_deepseek4_hybrid_stack_prefill_residual_upto`/`trace_deepseek4_hybrid_stack_head` at an
    /// all-Sliding schedule, the per-layer dispatch `trace_deepseek4_sliding_prefill` reaches too. No
    /// divergence is known here, so it is not `#[ignore]`d; a red result would be a new prefill-only bug
    /// (Card 386 scope note) with its own card.
    #[test]
    fn deepseek4_sliding_prefill_matches_independent_reference() {
        let cfg = tiny_cfg();
        let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Sliding);
        let l = 5usize;
        let tokens = [1usize, 2, 3, 4, 5];
        let w = all_weights(&cfg);
        let moe = moe_fixture::MoeFixture::new(&cfg);
        let ref_logits = deepseek4_decode_ref(&cfg, &w, &moe, &tokens);

        let g = trace_deepseek4_hybrid_stack_prefill_residual_upto(cfg, l, &schedule, cfg.layers)
            .expect("trace_deepseek4_hybrid_stack_prefill_residual_upto should trace");
        let mut mask = vec![-1.0e9f32; l * l];
        for i in 0..l {
            let start = i.saturating_sub(cfg.sliding_window - 1);
            for j in start..=i {
                mask[i * l + j] = 0.0;
            }
        }

        let step = moe_fixture::V4StepInputs {
            tokens: &tokens,
            position: 0,
        };
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            if let Some(value) = moe.bind_exact(meta, step) {
                inputs.insert(id, value);
                continue;
            }
            let value = match &meta.storage {
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        mask.clone(),
                    ))
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data = w
                        .get(name)
                        .unwrap_or_else(|| panic!("no weight for {name}"));
                    moe_fixture::bind_const(
                        name,
                        meta.aval.shape.clone(),
                        meta.aval.dtype,
                        data.clone(),
                    )
                }
                other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
            };
            inputs.insert(id, value);
        }
        let residual = moe_fixture::eval_graph(&cfg, &g, &inputs).expect("eval prefill residual");
        let residual = moe_fixture::dense(&residual);
        assert_eq!(residual.shape(), vec![1, l, cfg.hc_mult, cfg.hidden]);

        let head = trace_deepseek4_hybrid_stack_head(&cfg, &schedule, l)
            .expect("trace_deepseek4_hybrid_stack_head should trace");
        let mut head_inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &head.inputs {
            let meta = head.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data: Vec<f32> = if name == "resid" {
                residual.as_f32().unwrap().to_vec()
            } else {
                w.get(name)
                    .unwrap_or_else(|| panic!("no weight for {name}"))
                    .clone()
            };
            head_inputs.insert(
                id,
                moe_fixture::bind_const(name, meta.aval.shape.clone(), meta.aval.dtype, data),
            );
        }
        let logits = poot_eval::eval(
            &head,
            &head_inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval prefill head")
        .output
        .into_host()
        .expect("prefill head output is dense");
        assert_eq!(logits.shape(), vec![1, l, cfg.vocab]);

        // Card 422: `qpos` indexes the nested `ref_logits` and the flat stride into `logits.data`
        // (`qpos * cfg.vocab + i`); `i` does the same for `want`. Clippy's `.iter().enumerate()` suggestion only
        // removes the nested indexing; a full rewrite would need `logits.data.chunks(cfg.vocab)` zipped against
        // `ref_logits`, which this comparison code should not carry unproven.
        #[allow(clippy::needless_range_loop)]
        for qpos in 0..tokens.len() {
            let want = &ref_logits[qpos];
            #[allow(clippy::needless_range_loop)]
            for i in 0..cfg.vocab {
                let gv = logits.as_f32().unwrap()[qpos * cfg.vocab + i];
                let wv = want[i];
                assert!(
                    (gv - wv).abs() <= 1e-4 * wv.abs().max(1e-3) + V4_HYBRID_STACK_ABS_FLOOR,
                    "prefill position {qpos} logit {i}: graph={gv} ref={wv}"
                );
            }
        }
    }

    /// Card 364a acceptance: the routed-plus-shared MoE block alone, so a failure names the router rather than
    /// a rope table.
    mod routed_moe {
        use super::*;
        use poot_eval::{EvalError, Value};

        /// One layer's MoE block as its own graph: the real token slot, Card 372c guard and composition, with the
        /// block input as an ordinary constant.
        fn probe(
            cfg: &DeepseekV4Config,
            li: usize,
            t: usize,
            phase: V4MoePhase,
        ) -> Graph<ValidationOutputs> {
            let b = Builder::new();
            let tokens = b.slot(Slot::Token, TensorType::new(vec![t], DType::I32));
            let schedule = vec![V4LayerKind::Sliding; cfg.layers];
            let (plan, dims, _embed_ids, mut moe) =
                deepseek4_moe_preamble(&b, cfg, &schedule, tokens, phase).expect("probe preamble");
            let x = b.constant("probe.x", TensorType::f32(vec![1, t, cfg.hidden]));
            let y = deepseek4_routed_moe_ffn(&b, cfg, &plan, dims, li, x, &mut moe.context())
                .expect("probe MoE block");
            moe.finish(b, y, &[]).expect("probe graph closes")
        }

        /// Deterministic block inputs, spread so no two tokens share a route.
        fn probe_x(cfg: &DeepseekV4Config, t: usize) -> Vec<f32> {
            (0..t * cfg.hidden)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.25)
                .collect()
        }

        fn router_weights(cfg: &DeepseekV4Config, li: usize) -> HashMap<String, Vec<f32>> {
            moe_fixture::dense_rows(cfg, li, |name, numel| {
                (0..numel)
                    .map(|index| {
                        let seed = name.len() + index;
                        ((seed % 11) as f32 - 5.0) * 0.1
                    })
                    .collect()
            })
            .into_iter()
            // `ffn.gate.weight` is a BF16 checkpoint row (Card 554d's `bind_const` owner path needs a
            // BF16-exact value, same as every plan-sourced row `plan_dense_weights` truncates); `ffn.gate.bias`
            // (Score router only) is F32 and needs no truncation.
            .map(|(name, values)| {
                let values = if name.ends_with("ffn.gate.weight") {
                    values
                        .into_iter()
                        .map(crate::test_support::safetensors::truncate_to_bf16)
                        .collect()
                } else {
                    values
                };
                (name, values)
            })
            .collect()
        }

        fn bind(
            g: &Graph<ValidationOutputs>,
            moe: &moe_fixture::MoeFixture,
            dense: &HashMap<String, Vec<f32>>,
            x: &[f32],
            tokens: &[usize],
        ) -> HashMap<poot_graph_ir::ValueId, Value> {
            let step = moe_fixture::V4StepInputs {
                tokens,
                position: 0,
            };
            let mut inputs = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                if let Some(value) = moe.bind_exact(meta, step) {
                    inputs.insert(id, value);
                    continue;
                }
                let value = match meta.storage {
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("named const");
                        if name == "probe.x" {
                            Value::Host(poot_tensor::HostTensor::f32(
                                meta.aval.shape.clone(),
                                x.to_vec(),
                            ))
                        } else {
                            let data = dense
                                .get(name)
                                .unwrap_or_else(|| panic!("no weight for {name}"))
                                .clone();
                            moe_fixture::bind_const(
                                name,
                                meta.aval.shape.clone(),
                                meta.aval.dtype,
                                data,
                            )
                        }
                    }
                    other => panic!("unexpected storage {other:?} in a MoE probe"),
                };
                inputs.insert(id, value);
            }
            inputs
        }

        fn run(
            cfg: &DeepseekV4Config,
            li: usize,
            phase: V4MoePhase,
            tokens: &[usize],
        ) -> (Vec<f32>, Vec<f32>) {
            let t = tokens.len();
            let moe = moe_fixture::MoeFixture::new(cfg);
            let dense = router_weights(cfg, li);
            let x = probe_x(cfg, t);
            let g = probe(cfg, li, t, phase);
            let inputs = bind(&g, &moe, &dense, &x, tokens);
            let got = moe_fixture::eval_graph(cfg, &g, &inputs).expect("probe evaluates");
            let got = moe_fixture::dense(&got).as_f32().unwrap().to_vec();

            let mut want = Vec::with_capacity(t * cfg.hidden);
            for (row, &token) in tokens.iter().enumerate() {
                let slice = &x[row * cfg.hidden..(row + 1) * cfg.hidden];
                want.extend(moe.ffn_ref(cfg, &dense, li, token, slice));
            }
            (got, want)
        }

        fn assert_close(got: &[f32], want: &[f32], what: &str) {
            assert_eq!(got.len(), want.len(), "{what}: length");
            for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-4 * w.abs().max(1e-3),
                    "{what} lane {index}: graph={g} ref={w}"
                );
            }
        }

        /// SC-001 (hash row): the table picks the ids and the unbiased scores supply the weights. Red under
        /// "choose score top-k" and under "read the neighbouring token row", since the fixture table sends each
        /// token to a different expert set.
        #[test]
        fn deepseek4_hash_router_matches_independent_oracle() {
            let cfg = tiny_cfg();
            assert_eq!(cfg.router_kind(0), V4RouterKind::Hash);
            let (got, want) = run(&cfg, 0, V4MoePhase::Decode, &[3]);
            assert_close(&got, &want, "hash router");

            let (other, _) = run(&cfg, 0, V4MoePhase::Decode, &[4]);
            assert!(
                poot_test_util::max_abs_error(&got, &other) > 1e-6,
                "a hash layer must route token 3 and token 4 differently"
            );
        }

        /// SC-001 (score row): the correction bias moves the selection and never a weight. The reference gathers
        /// from unbiased scores, so gathering biased scores as weights is red.
        #[test]
        fn deepseek4_score_router_bias_is_selection_only() {
            let cfg = tiny_cfg();
            assert_eq!(cfg.router_kind(1), V4RouterKind::Score);
            let (got, want) = run(&cfg, 1, V4MoePhase::Decode, &[2]);
            assert_close(&got, &want, "score router");

            // A bias large enough to invert the order must change the output, else selection ignores the bias.
            let moe = moe_fixture::MoeFixture::new(&cfg);
            let mut dense = router_weights(&cfg, 1);
            let x = probe_x(&cfg, 1);
            let g = probe(&cfg, 1, 1, V4MoePhase::Decode);
            let baseline = {
                let inputs = bind(&g, &moe, &dense, &x, &[2]);
                moe_fixture::dense(&moe_fixture::eval_graph(&cfg, &g, &inputs).expect("baseline"))
                    .as_f32()
                    .unwrap()
                    .to_vec()
            };
            let bias = dense
                .get_mut("layers.1.ffn.gate.bias")
                .expect("score layers carry a correction bias");
            bias.iter_mut().enumerate().for_each(|(expert, value)| {
                *value = -(expert as f32);
            });
            let inputs = bind(&g, &moe, &dense, &x, &[2]);
            let reordered =
                moe_fixture::dense(&moe_fixture::eval_graph(&cfg, &g, &inputs).expect("reordered"))
                    .as_f32()
                    .unwrap()
                    .to_vec();
            assert!(
                poot_test_util::max_abs_error(&baseline, &reordered) > 1e-6,
                "a correction bias that reverses the expert order must change the selection"
            );
        }

        /// SC-002: a multi-token prefill block with repeated and non-monotonic route ids and at least
        /// one unused expert, against the hand-written reference. Red if the shared branch is dropped,
        /// the route scale is omitted, or the w1/w3 clamp rules are swapped.
        #[test]
        fn deepseek4_routed_shared_ffn_matches_independent_oracle() {
            let cfg = tiny_cfg();
            let tokens = [1usize, 1, 5, 2];
            let (got, want) = run(&cfg, 0, V4MoePhase::Prefill, &tokens);
            assert_close(&got, &want, "routed plus shared FFN");

            let moe = moe_fixture::MoeFixture::new(&cfg);
            let routed: std::collections::BTreeSet<i32> = tokens
                .iter()
                .flat_map(|&token| {
                    let row = token * cfg.experts_per_tok;
                    moe.table()[row..row + cfg.experts_per_tok].to_vec()
                })
                .collect();
            assert!(
                routed.len() < cfg.routed_experts,
                "the fixture must leave at least one expert unused, routed {routed:?}"
            );
            assert!(
                tokens.windows(2).any(|pair| pair[0] == pair[1]),
                "the fixture must route a repeated token"
            );
        }

        /// FR-013: decode reaches the indexed helper and prefill the grouped one. Only the grouped form builds the
        /// canonical sort, so a `Scatter` is present in exactly one. Red if either phase calls the other helper.
        #[test]
        fn deepseek4_decode_is_indexed_and_prefill_is_grouped() {
            let cfg = tiny_cfg();
            let scatters = |phase| {
                probe(&cfg, 0, 2, phase)
                    .eqns
                    .iter()
                    .filter(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Scatter { .. }))
                    .count()
            };
            assert_eq!(
                scatters(V4MoePhase::Decode),
                0,
                "decode must reach packed_indexed_linear, which builds no sort"
            );
            assert!(
                scatters(V4MoePhase::Prefill) > 0,
                "prefill must reach packed_grouped_linear, which builds the canonical sort"
            );
        }

        /// FR-021: the runtime token index is bounded by Card 372c's guard and its witness is declared, so an
        /// out-of-range id fails before any result is published. Red if the gather reads the raw slot or the
        /// witness declaration is dropped.
        #[test]
        fn deepseek4_token_index_is_guarded() {
            let cfg = tiny_cfg();
            let moe = moe_fixture::MoeFixture::new(&cfg);
            let dense = router_weights(&cfg, 0);
            let x = probe_x(&cfg, 1);
            let g = probe(&cfg, 0, 1, V4MoePhase::Decode);

            let inputs = bind(&g, &moe, &dense, &x, &[cfg.vocab - 1]);
            moe_fixture::eval_graph(&cfg, &g, &inputs)
                .expect("the last in-range token id evaluates");

            let inputs = bind(&g, &moe, &dense, &x, &[cfg.vocab]);
            let failure = match moe_fixture::eval_graph(&cfg, &g, &inputs) {
                Err(EvalError::Validation(failure)) => failure,
                other => panic!("an out-of-range token id must fail validation, got {other:?}"),
            };
            assert_eq!(
                deepseek4_router_validation_error(&failure),
                Some(DeepseekV4RouterError::TokenIndexOutOfRange)
            );
        }

        /// FR-016: nonfinite logits and a nonpositive selected-score sum each fail deterministically
        /// through their own named validation output, publishing no result. Red if either witness is
        /// dropped, or if the two map to the same typed failure.
        #[test]
        fn deepseek4_router_cpu_errors_are_transactional() {
            let cfg = tiny_cfg();
            let moe = moe_fixture::MoeFixture::new(&cfg);
            let x = probe_x(&cfg, 1);
            let g = probe(&cfg, 0, 1, V4MoePhase::Decode);

            // Every logit far below the f32 exp floor makes `sqrt(softplus(logit))` exactly zero, so the selected
            // scores sum to zero while staying finite.
            let mut dense = router_weights(&cfg, 0);
            dense.insert(
                "layers.0.ffn.gate.weight".to_string(),
                vec![-1.0e3; cfg.hidden * cfg.routed_experts],
            );
            let inputs = bind(&g, &moe, &dense, &[1.0; 8], &[1]);
            let failure = match moe_fixture::eval_graph(&cfg, &g, &inputs) {
                Err(EvalError::Validation(failure)) => failure,
                other => panic!("a zero score sum must fail validation, got {other:?}"),
            };
            assert_eq!(
                deepseek4_router_validation_error(&failure),
                Some(DeepseekV4RouterError::UnusableSelectedScoreSum { layer: 0 })
            );

            // A finite logit above the f32 `exp` overflow point (~88.7): `softplus` is the naive `log(1 + exp(x))`,
            // so `raw` is `+inf`, the sum is `+inf`, and the weights would be NaN. The nonfinite-logit witness
            // cannot see it, so the sum witness's finiteness half must. Red if that half is dropped: the graph
            // publishes NaN with both witnesses reading zero.
            let mut dense = router_weights(&cfg, 0);
            dense.insert(
                "layers.0.ffn.gate.weight".to_string(),
                vec![1.0e3; cfg.hidden * cfg.routed_experts],
            );
            let inputs = bind(&g, &moe, &dense, &[1.0; 8], &[1]);
            let failure = match moe_fixture::eval_graph(&cfg, &g, &inputs) {
                Err(EvalError::Validation(failure)) => failure,
                other => panic!(
                    "an overflowing softplus must fail validation, not publish NaN, got {other:?}"
                ),
            };
            assert_eq!(
                deepseek4_router_validation_error(&failure),
                Some(DeepseekV4RouterError::UnusableSelectedScoreSum { layer: 0 })
            );

            let mut dense = router_weights(&cfg, 0);
            dense.insert(
                "layers.0.ffn.gate.weight".to_string(),
                vec![f32::NAN; cfg.hidden * cfg.routed_experts],
            );
            let inputs = bind(&g, &moe, &dense, &x, &[1]);
            let failure = match moe_fixture::eval_graph(&cfg, &g, &inputs) {
                Err(EvalError::Validation(failure)) => failure,
                other => panic!("nonfinite logits must fail validation, got {other:?}"),
            };
            assert_eq!(
                deepseek4_router_validation_error(&failure),
                Some(DeepseekV4RouterError::NonfiniteRouterLogits { layer: 0 })
            );
        }

        /// What only a trace can prove about a V4 DECODE graph: it builds, validates, declares exactly the
        /// witnesses its layer count implies, and carries one state pair per cache. Numeric decode oracles
        /// compare logits, which do not change if a witness is missing. Red if a tracer stops declaring a
        /// witness or the per-layer witness pair count drifts from the layer count.
        #[test]
        fn deepseek4_decode_graphs_validate_and_declare_their_witnesses() {
            let cfg = tiny_cfg();
            let hybrid = DeepseekV4Config {
                layers: 3,
                ..tiny_cfg()
            };
            let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
            let graphs: Vec<(&str, usize, Graph<ValidationOutputs>)> = vec![
                (
                    "sliding",
                    cfg.layers,
                    trace_deepseek4_sliding_decode(cfg, 6).expect("sliding decode"),
                ),
                (
                    "hca",
                    cfg.layers,
                    trace_deepseek4_hca_decode(cfg, 6).expect("hca decode"),
                ),
                (
                    "csa",
                    cfg.layers,
                    trace_deepseek4_csa_decode(cfg, 6).expect("csa decode"),
                ),
                (
                    "hybrid",
                    hybrid.layers,
                    trace_deepseek4_hybrid_stack_decode(hybrid, 8, &schedule)
                        .expect("hybrid decode"),
                ),
            ];
            for (what, layers, g) in &graphs {
                g.validate()
                    .unwrap_or_else(|e| panic!("{what} decode: {e}"));
                // One shared token-index bound, then a finite-logit and a score-sum witness per layer.
                let declared = g.validation_outputs();
                assert_eq!(
                    declared.len(),
                    1 + 2 * layers,
                    "{what} decode declares the wrong witness count"
                );
                assert_eq!(
                    declared[0].name, "deepseek4.token_index",
                    "{what} decode must bound its token index first"
                );
                for li in 0..*layers {
                    assert!(
                        declared
                            .iter()
                            .any(|d| d.name == format!("deepseek4.layer{li}.router_logits_finite")),
                        "{what} decode is missing layer {li}'s finite-logit witness"
                    );
                    assert!(
                        declared
                            .iter()
                            .any(|d| d.name == format!("deepseek4.layer{li}.router_score_sum")),
                        "{what} decode is missing layer {li}'s score-sum witness"
                    );
                }
                assert!(!g.state.is_empty(), "{what} decode carries no state");
            }
        }

        /// SC-003: every layer kind reaches the same routed-plus-shared composition, in both phases. Red if any
        /// call site reverts to a shared-only or dense helper (the routed expert source names would vanish).
        #[test]
        fn deepseek4_all_layer_kinds_use_routed_moe() {
            let cfg = tiny_cfg();
            let graphs: Vec<(&str, Graph<ValidationOutputs>)> = vec![
                (
                    "sliding decode",
                    trace_deepseek4_sliding_decode(cfg, 6).expect("sliding decode"),
                ),
                (
                    "sliding prefill",
                    trace_deepseek4_sliding_prefill(cfg, 6).expect("sliding prefill"),
                ),
                (
                    "hca decode",
                    trace_deepseek4_hca_decode(cfg, 6).expect("hca decode"),
                ),
                (
                    "hca prefill",
                    trace_deepseek4_hca_prefill(cfg, 6).expect("hca prefill"),
                ),
                (
                    "csa decode",
                    trace_deepseek4_csa_decode(cfg, 6).expect("csa decode"),
                ),
                (
                    "csa prefill",
                    trace_deepseek4_csa_prefill(cfg, 8).expect("csa prefill"),
                ),
            ];
            for (what, g) in &graphs {
                let names: Vec<&str> = g
                    .values
                    .iter()
                    .filter_map(|value| value.name.as_deref())
                    .collect();
                for li in 0..cfg.layers {
                    for expert in 0..cfg.routed_experts {
                        for projection in ["w1", "w2", "w3"] {
                            let id = format!("layers.{li}.ffn.experts.{expert}.{projection}");
                            let weight = poot_graph_ir::PackedSourceName::weight(&id);
                            assert!(
                                names.contains(&weight.as_str()),
                                "{what} is missing routed expert source {}",
                                weight.as_str()
                            );
                        }
                    }
                    let shared = poot_graph_ir::PackedSourceName::weight(&format!(
                        "layers.{li}.ffn.shared_experts.w2"
                    ));
                    assert!(
                        names.contains(&shared.as_str()),
                        "{what} is missing the shared expert"
                    );
                }
                assert!(
                    !names.iter().any(|name| name.contains("mlp.gate_proj")),
                    "{what} still declares a dense FFN weight"
                );
            }
        }
    }

    /// Card 452: the production hash router's response to `token_id`, observed through
    /// [`trace_deepseek4_hash_router_ids`] with no activation, embedding or residual in the value. The graph
    /// shares its preamble and its id gather ([`deepseek4_hash_router_ids`]) with `deepseek4_routed_moe_ffn`.
    mod hash_router_ids {
        use super::*;
        use poot_eval::Value;
        use std::sync::Arc;

        /// Three layers, the first two hash-routed, four experts, two slots, ten token ids.
        fn cfg() -> DeepseekV4Config {
            DeepseekV4Config {
                layers: 3,
                hash_router_layers: 2,
                routed_experts: 4,
                experts_per_tok: 2,
                ..tiny_cfg()
            }
        }

        /// Hand-written `tid2eid` rows, one per token id, for hash layers 0 and 1. Rows differ across
        /// tokens, and layer 1's row differs from layer 0's for every token, so neither a wrong token nor a
        /// wrong layer's table can produce the right ids.
        const TABLES: [[[i32; 2]; 10]; 2] = [
            [
                [1, 3],
                [0, 2],
                [3, 1],
                [2, 0],
                [1, 2],
                [3, 0],
                [0, 3],
                [2, 1],
                [1, 0],
                [3, 2],
            ],
            [
                [2, 0],
                [3, 1],
                [0, 2],
                [1, 3],
                [3, 0],
                [1, 2],
                [2, 1],
                [0, 3],
                [3, 2],
                [1, 0],
            ],
        ];

        /// The prompt the rows observe: tokens 7, 2, 8, 4, and the ids each hash layer must select for them,
        /// written out rather than read back from [`TABLES`].
        const TOKENS: [usize; 4] = [7, 2, 8, 4];
        const EXPECTED: [[[i32; 2]; 4]; 2] = [
            [[2, 1], [3, 1], [1, 0], [1, 2]],
            [[0, 3], [0, 2], [3, 2], [3, 0]],
        ];

        fn table_words(layer: usize, swap: Option<(usize, usize)>) -> Vec<i32> {
            let mut rows = TABLES[layer];
            if let Some((a, b)) = swap {
                rows.swap(a, b);
            }
            rows.iter().flatten().copied().collect()
        }

        fn view(layer: usize, swap: Option<(usize, usize)>) -> poot_eval::ExactI32TensorView {
            poot_eval::ExactI32TensorView::try_from_words(
                vec![10, 2],
                Arc::from(table_words(layer, swap)),
            )
            .expect("table view")
        }

        /// Trace layer `layer`, bind `table` to its `tid2eid` constant, and return the observed ids per token.
        fn observe(
            layer: usize,
            tokens: &[usize],
            table: poot_eval::ExactI32TensorView,
        ) -> Vec<[i32; 2]> {
            let cfg = cfg();
            let g =
                trace_deepseek4_hash_router_ids(cfg, layer, tokens.len()).expect("router trace");
            let mut inputs = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                let value = match (&meta.storage, meta.name.as_deref()) {
                    (Storage::Slot(Slot::Token), _) => moe_fixture::exact_i32(
                        meta.aval.shape.clone(),
                        tokens.iter().map(|&t| t as i32).collect(),
                    ),
                    (Storage::Const, Some(name)) => {
                        assert_eq!(name, format!("layers.{layer}.ffn.gate.tid2eid"));
                        Value::from(table.clone())
                    }
                    other => panic!("unexpected router-observation input {other:?}"),
                };
                inputs.insert(id, value);
            }
            let out = moe_fixture::eval_graph(&cfg, &g, &inputs).expect("router observation");
            let out = moe_fixture::dense(&out);
            assert_eq!(out.shape(), vec![tokens.len(), 2]);
            out.as_f32()
                .unwrap()
                .chunks_exact(2)
                .map(|row| [row[0] as i32, row[1] as i32])
                .collect()
        }

        fn expected(layer: usize) -> Vec<[i32; 2]> {
            EXPECTED[layer].to_vec()
        }

        /// SC-001: each hash layer selects the independently expected ids for each token, and the ids differ
        /// across tokens and across the two layers. Red under a fixed token id (row 0 for every token) and
        /// under a neighbouring-row read.
        #[test]
        fn card452_hash_router_selects_expected_ids_per_token_at_two_layers() {
            for layer in 0..2 {
                let got = observe(layer, &TOKENS, view(layer, None));
                eprintln!("card452 SC-001 layer {layer} tokens {TOKENS:?} -> ids {got:?}");
                assert_eq!(got, expected(layer), "layer {layer} observed ids");
                let distinct: std::collections::BTreeSet<_> = got.iter().collect();
                assert!(
                    distinct.len() >= 3,
                    "layer {layer}: the tokens must select at least three different id rows, got {got:?}"
                );
            }
            assert_ne!(
                observe(0, &TOKENS, view(0, None)),
                observe(1, &TOKENS, view(1, None)),
                "the two hash layers must read their own tables"
            );
        }

        /// SC-003: swapping two non-identical `tid2eid` rows changes the observed ids at exactly the swapped
        /// tokens, so a production path that swapped rows would fail SC-001. The printed rows show the swap
        /// is not a no-op at this data (tokens 7 and 2 have different rows in both layers).
        #[test]
        fn card452_swapped_table_rows_are_observed_through_the_router() {
            #[allow(clippy::needless_range_loop)] // `layer` also names the traced layer
            for layer in 0..2 {
                let (a, b) = (TOKENS[0], TOKENS[1]);
                assert_ne!(TABLES[layer][a], TABLES[layer][b], "rows must differ");
                let swapped = observe(layer, &TOKENS, view(layer, Some((a, b))));
                let plain = observe(layer, &TOKENS, view(layer, None));
                eprintln!(
                    "card452 SC-003 layer {layer} swap rows {a}<->{b}: plain {plain:?} swapped {swapped:?}"
                );
                assert_ne!(swapped, expected(layer), "the swap must be observable");
                assert_eq!(swapped[0], plain[1], "token {a} now reads token {b}'s row");
                assert_eq!(swapped[1], plain[0], "token {b} now reads token {a}'s row");
                assert_eq!(&swapped[2..], &plain[2..], "other tokens are untouched");
            }
        }

        /// A layer that is not hash-routed has no `tid2eid` to observe.
        #[test]
        fn card452_observation_rejects_a_score_routed_layer() {
            assert!(matches!(
                trace_deepseek4_hash_router_ids(cfg(), 2, 1),
                Err(DeepseekV4MoeError::Config { field: "layer", .. })
            ));
        }
    }

    /// Card 364b acceptance: every graph source name comes from the plan, with the checkpoint's role, dtype and
    /// orientation.
    mod exact_sources {
        use super::*;
        use crate::deepseek4::sources::V4SourceClass;
        use std::collections::BTreeSet;

        /// One traced graph per entry point with its schedule. A three-layer mixed stack, so the hybrid rows,
        /// compressor widths and the CSA-only indexer all appear.
        fn all_tracer_graphs() -> Vec<(
            &'static str,
            DeepseekV4Config,
            Vec<V4LayerKind>,
            Graph<ValidationOutputs>,
        )> {
            let cfg = tiny_cfg();
            let mixed_cfg = DeepseekV4Config {
                layers: 3,
                ..tiny_cfg()
            };
            let mixed = vec![V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
            let uniform = |kind| vec![kind; cfg.layers];
            vec![
                (
                    "sliding decode",
                    cfg,
                    uniform(V4LayerKind::Sliding),
                    trace_deepseek4_sliding_decode(cfg, 6).expect("sliding decode"),
                ),
                (
                    "sliding prefill",
                    cfg,
                    uniform(V4LayerKind::Sliding),
                    trace_deepseek4_sliding_prefill(cfg, 6).expect("sliding prefill"),
                ),
                (
                    "hca decode",
                    cfg,
                    uniform(V4LayerKind::Hca),
                    trace_deepseek4_hca_decode(cfg, 6).expect("hca decode"),
                ),
                (
                    "hca prefill",
                    cfg,
                    uniform(V4LayerKind::Hca),
                    trace_deepseek4_hca_prefill(cfg, 6).expect("hca prefill"),
                ),
                (
                    "csa decode",
                    cfg,
                    uniform(V4LayerKind::Csa),
                    trace_deepseek4_csa_decode(cfg, 6).expect("csa decode"),
                ),
                (
                    "csa prefill",
                    cfg,
                    uniform(V4LayerKind::Csa),
                    trace_deepseek4_csa_prefill(cfg, 8).expect("csa prefill"),
                ),
                (
                    "hybrid decode",
                    mixed_cfg,
                    mixed.clone(),
                    trace_deepseek4_hybrid_stack_decode(mixed_cfg, 8, &mixed)
                        .expect("hybrid decode"),
                ),
                (
                    "hybrid prefill",
                    mixed_cfg,
                    mixed.clone(),
                    trace_deepseek4_hybrid_stack_prefill(mixed_cfg, 8, &mixed)
                        .expect("hybrid prefill"),
                ),
            ]
        }

        /// The named constants of one graph, in declaration order.
        fn const_names(g: &Graph<ValidationOutputs>) -> Vec<&str> {
            g.inputs
                .iter()
                .filter(|&&id| g.meta(id).storage == Storage::Const)
                .filter_map(|&id| g.meta(id).name.as_deref())
                .collect()
        }

        /// SC-001, both directions. Every `Storage::Const` input of every tracer resolves to exactly one plan row
        /// (a packed component, an exact dense or exact-I32 row, or a named host table), and every dense row the
        /// schedule's layer kinds imply is staged. Walking every constant means a row staged under its old name
        /// cannot slip past. The reverse direction matters because a dense row no tracer reads would otherwise
        /// be visible only to `check_exact_counts`, at the exact config (packed rows have
        /// `deepseek4_every_packed_row_is_consumed`).
        #[test]
        fn deepseek4_tower_sources_are_packed_or_exact_dense() {
            for (what, cfg, schedule, g) in all_tracer_graphs() {
                let plan = DeepseekV4SourcePlan::new(&cfg, &schedule).expect("plan");
                let mut packed = 0;
                let mut derived = 0;
                let mut staged_dense = BTreeSet::new();
                for name in const_names(&g) {
                    match plan.classify(name) {
                        Some(V4SourceClass::Packed(_)) => packed += 1,
                        Some(V4SourceClass::Dense(row)) => {
                            staged_dense.insert(row.name.as_str());
                        }
                        Some(V4SourceClass::Derived) => derived += 1,
                        None => panic!(
                            "{what} stages {name}, which the source plan does not name - a \
                             placeholder, a typo, or a host table nobody declared"
                        ),
                    }
                }
                assert!(
                    packed > 0 && !staged_dense.is_empty() && derived > 0,
                    "{what} counts"
                );

                let planned_dense: BTreeSet<&str> =
                    plan.dense_sources().map(|row| row.name.as_str()).collect();
                let unstaged: Vec<&str> =
                    planned_dense.difference(&staged_dense).copied().collect();
                assert!(
                    unstaged.is_empty(),
                    "{what}: the plan names dense rows this graph never stages: {unstaged:?}"
                );
            }
        }

        /// FR-009, Card 385. Every packed row the plan names is consumed as a packed linear, both components
        /// staged together. A new exception (`attn.wo_a` was one before Card 385) is red.
        #[test]
        fn deepseek4_every_packed_row_is_consumed() {
            for (what, cfg, schedule, g) in all_tracer_graphs() {
                let plan = DeepseekV4SourcePlan::new(&cfg, &schedule).expect("plan");
                let names = const_names(&g);
                let staged: BTreeSet<&str> = names.iter().copied().collect();
                for row in plan.packed_sources() {
                    let constants = row.source_constants();
                    let consumed = staged.contains(constants[0].0.as_str());
                    assert!(
                        constants
                            .iter()
                            .all(|(name, _)| staged.contains(name.as_str()) == consumed),
                        "{what}: {} stages some components without the others",
                        row.linear_id
                    );
                    assert!(
                        consumed,
                        "{what}: {} ({:?} on layer {}) is never consumed as a packed linear",
                        row.linear_id, row.role, row.layer
                    );
                }
            }
        }

        /// FR-005. A dense constant is staged at the plan row's own shape and dtype: nothing pre-transposed or
        /// silently widened to f32. The per-block cast and transpose equations are not constrained here.
        #[test]
        fn deepseek4_dense_rows_keep_the_checkpoint_orientation() {
            let mut seen_bf16 = 0;
            let mut seen_f32 = 0;
            let mut seen_i32 = 0;
            for (what, cfg, schedule, g) in all_tracer_graphs() {
                let plan = DeepseekV4SourcePlan::new(&cfg, &schedule).expect("plan");
                for &id in &g.inputs {
                    let meta = g.meta(id);
                    if meta.storage != Storage::Const {
                        continue;
                    }
                    let name = meta.name.as_deref().expect("named constant");
                    let Some(V4SourceClass::Dense(row)) = plan.classify(name) else {
                        continue;
                    };
                    assert_eq!(
                        (meta.aval.shape.clone(), meta.aval.dtype),
                        (row.shape.clone(), row.dtype),
                        "{what}: {name} is staged with a shape or dtype the plan does not declare"
                    );
                    match row.dtype {
                        DType::BF16 => seen_bf16 += 1,
                        DType::F32 => seen_f32 += 1,
                        DType::I32 => seen_i32 += 1,
                        other => panic!("{name} has dense dtype {other}"),
                    }
                }
            }
            // All three dense lanes are exercised, so the assertion is not vacuous for any: BF16 weights and norms,
            // F32 mHC bases and scales, and the Card 371 hash table.
            assert!(seen_bf16 > 0 && seen_f32 > 0 && seen_i32 > 0);
        }
    }

    // ---- HCA full-model independent reference. Reuses `attention_core_ref`/`dense_ffn_ref`/
    // `hyper_connection_ref`/`hyper_connection_combine_ref`/`hyper_head_ref`/`rope_interleaved_ref` unchanged:
    // HCA only changes which rope table the layer uses (the compress-theta table stored under the same
    // "rope.cos"/"rope.sin" keys) and appends a compressed-KV branch to `attention_core_ref`'s generic
    // `cache: &[Vec<f32>]` argument.

    /// Every constant an all-HCA V4 graph reads apart from its packed owners. Every non-sliding layer uses
    /// `compress_rope_theta` for Q/K/V and the compressed entries, so that is the table under the main rope keys.
    fn hca_all_weights(cfg: &DeepseekV4Config) -> HashMap<String, Vec<f32>> {
        schedule_weights(
            cfg,
            &vec![V4LayerKind::Hca; cfg.layers],
            cfg.compress_rope_theta,
        )
    }

    /// Independent reference for [`hca_softmax_gate_pool`] + [`hca_window_rope`] together, over a growing
    /// per-token history (`kv_hist`/`gate_hist`) rather than the graph's fixed-capacity padded buffer. Returns
    /// only the `kv_hist.len() / compress_rate` complete windows, which are exactly the causally valid ones
    /// (see [`hca_decode_validity_mask`]).
    fn hca_pool_windows_ref(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Vec<f32>>,
        li: usize,
        kv_hist: &[Vec<f32>],
        gate_hist: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let p = |s: &str| format!("layers.{li}.{s}");
        let cr = cfg.hca_compress_rate;
        let d = cfg.head_dim;
        let n_windows = kv_hist.len() / cr;
        if n_windows == 0 {
            return Vec::new();
        }
        let position_bias = w.get(&p("attn.compressor.ape")).unwrap();
        let kv_norm_w = w.get(&p("attn.compressor.norm.weight")).unwrap();
        let cos = w.get("rope.cos").unwrap();
        let sin = w.get("rope.sin").unwrap();
        let nope_w = cfg.head_dim - cfg.qk_rope_head_dim;
        let mut out = Vec::with_capacity(n_windows);
        for win in 0..n_windows {
            let mut pooled = vec![0.0f32; d];
            for ch in 0..d {
                let logits: Vec<f32> = (0..cr)
                    .map(|j| gate_hist[win * cr + j][ch] + position_bias[j * d + ch])
                    .collect();
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let mut acc = 0.0f32;
                for j in 0..cr {
                    acc += (exps[j] / sum) * kv_hist[win * cr + j][ch];
                }
                pooled[ch] = acc;
            }
            let normed = rmsnorm_ref(&pooled, kv_norm_w, cfg.eps);
            let pos = win * cr;
            let mut rope_part = normed[nope_w..].to_vec();
            rope_part =
                rope_interleaved_ref(&rope_part, cos, sin, pos, cfg.qk_rope_head_dim, false);
            let mut entry = normed[..nope_w].to_vec();
            entry.extend(rope_part);
            out.push(entry);
        }
        out
    }

    /// Full independent-reference decode for an all-HCA model: [`deepseek4_decode_ref`]'s local-sliding-K==V,
    /// mHC and FFN structure, with HCA's compressed branch appended to the local `cache` slice before
    /// [`attention_core_ref`] runs (`kv = torch.cat([kv, compressed_kv], dim=2)`; `attention_core_ref` treats
    /// `cache` generically).
    fn deepseek4_hca_decode_ref(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Vec<f32>>,
        moe: &moe_fixture::MoeFixture,
        tokens: &[usize],
    ) -> Vec<Vec<f32>> {
        let (h, hc) = (cfg.hidden, cfg.hc_mult);
        let embed = w.get("embed.weight").unwrap();

        let mut caches: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut hca_kv_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut hca_gate_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut all_logits = Vec::new();

        for (qpos, &tok) in tokens.iter().enumerate() {
            let e = embed[tok * h..(tok + 1) * h].to_vec();
            let mut streams: Vec<Vec<f32>> = (0..hc).map(|_| e.clone()).collect();

            for (li, cache) in caches.iter_mut().enumerate() {
                let p = |s: &str| format!("layers.{li}.{s}");
                let (_pre, post, comb, collapsed) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_attn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_attn_base")).unwrap(),
                    w.get(&p("hc_attn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed =
                    rmsnorm_ref(&collapsed, w.get(&p("attn_norm.weight")).unwrap(), cfg.eps);

                let wkv = moe.weight(&p("attn.wkv"));
                let kv_raw = linear_ref_out_in(&normed, &wkv, cfg.hidden, cfg.head_dim);
                let kv_n = rmsnorm_ref(&kv_raw, w.get(&p("attn.kv_norm.weight")).unwrap(), cfg.eps);
                let nope_w = cfg.head_dim - cfg.qk_rope_head_dim;
                let mut kv_rope = kv_n[nope_w..].to_vec();
                kv_rope = rope_interleaved_ref(
                    &kv_rope,
                    w.get("rope.cos").unwrap(),
                    w.get("rope.sin").unwrap(),
                    qpos,
                    cfg.qk_rope_head_dim,
                    false,
                );
                let mut kv_new = kv_n[..nope_w].to_vec();
                kv_new.extend(kv_rope);
                cache.push(kv_new);
                let window_start = qpos.saturating_sub(cfg.sliding_window - 1);
                let mut combined_cache = cache[window_start..=qpos].to_vec();

                let hkv = w.get(&p("attn.compressor.wkv.weight")).unwrap();
                let hgate = w.get(&p("attn.compressor.wgate.weight")).unwrap();
                let raw_kv = linear_ref_out_in(&normed, hkv, cfg.hidden, cfg.head_dim);
                let raw_gate = linear_ref_out_in(&normed, hgate, cfg.hidden, cfg.head_dim);
                hca_kv_hist[li].push(raw_kv);
                hca_gate_hist[li].push(raw_gate);
                let compressed =
                    hca_pool_windows_ref(cfg, w, li, &hca_kv_hist[li], &hca_gate_hist[li]);
                combined_cache.extend(compressed);

                let attn_out = attention_core_ref(cfg, w, moe, li, &normed, &combined_cache, qpos);
                streams = hyper_connection_combine_ref(&post, &comb, &attn_out, &streams, hc, h);

                let (_pre2, post2, comb2, collapsed2) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_ffn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_ffn_base")).unwrap(),
                    w.get(&p("hc_ffn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed2 =
                    rmsnorm_ref(&collapsed2, w.get(&p("ffn_norm.weight")).unwrap(), cfg.eps);
                let mlp_out = moe.ffn_ref(cfg, w, li, tok, &normed2);
                streams = hyper_connection_combine_ref(&post2, &comb2, &mlp_out, &streams, hc, h);
            }

            let hh_scale = w.get("hc_head_scale").unwrap()[0];
            let collapsed_final = hyper_head_ref(
                &streams,
                &mhc_projection_ref(w, "hc_head_fn", hc, hc * h),
                w.get("hc_head_base").unwrap(),
                hh_scale,
                hc,
                h,
                cfg.eps,
                cfg.hc_eps,
            );
            let xf = rmsnorm_ref(&collapsed_final, w.get("norm.weight").unwrap(), cfg.eps);
            let logits = linear_ref_out_in(&xf, w.get("head.weight").unwrap(), h, cfg.vocab);
            all_logits.push(logits);
        }
        all_logits
    }

    /// SC-004 for HCA: six steps at `hca_compress_rate = 2` and capacity six, so all three compressed windows
    /// are built and the `sliding_window = 3` local floor fires past capacity in one run. Three caches per
    /// layer are carried, each written at its own exact position.
    #[test]
    fn deepseek4_hca_decode_matches_independent_reference_across_multiple_windows() {
        let cfg = tiny_cfg();
        let cap = 6;
        let tokens = [1usize, 2, 3, 4, 5, 6];
        let w = hca_all_weights(&cfg);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &vec![V4LayerKind::Hca; cfg.layers]);
        let want = deepseek4_hca_decode_ref(&cfg, &w, &moe, &tokens);

        let g = trace_deepseek4_hca_decode(cfg, cap).expect("hca decode traces");
        let got = moe_fixture::decode_logits(&cfg, &g, &moe, &w, &tokens, |name| match name {
            "hca.rope.cos" => Some(w.get("rope.cos").unwrap().clone()),
            "hca.rope.sin" => Some(w.get("rope.sin").unwrap().clone()),
            _ => None,
        });
        moe_fixture::assert_decode_logits(&got, &want, "hca decode", 0.0);
    }

    /// Checked at every position, not only the last (Card 386 P2; see
    /// `deepseek4_sliding_prefill_matches_independent_reference`).
    #[test]
    fn deepseek4_hca_prefill_matches_independent_reference() {
        let cfg = tiny_cfg();
        let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Hca);
        let l = 6usize; // multiple of hca_compress_rate=2
        let tokens = [1usize, 2, 3, 4, 5, 6];
        let w = hca_all_weights(&cfg);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &schedule);
        let ref_logits = deepseek4_hca_decode_ref(&cfg, &w, &moe, &tokens);

        let g = trace_deepseek4_hybrid_stack_prefill_residual_upto(cfg, l, &schedule, cfg.layers)
            .expect("trace_deepseek4_hybrid_stack_prefill_residual_upto should trace");
        let mut mask = vec![-1.0e9f32; l * l];
        for i in 0..l {
            let start = i.saturating_sub(cfg.sliding_window - 1);
            for j in start..=i {
                mask[i * l + j] = 0.0;
            }
        }
        let n_windows = l / cfg.hca_compress_rate;
        let win_positions: Vec<f32> = (0..n_windows)
            .map(|w_i| (w_i * cfg.hca_compress_rate) as f32)
            .collect();
        // Window `wi` is valid only once its last source token (`win_positions[wi] + cr - 1`) has been
        // processed (see `hca_decode_validity_mask`).
        let mut block_bias = vec![0.0f32; l * n_windows];
        for i in 0..l {
            for wi in 0..n_windows {
                if (i as f32) < win_positions[wi] + (cfg.hca_compress_rate - 1) as f32 {
                    block_bias[i * n_windows + wi] = HCA_MASK_NEG;
                }
            }
        }

        let step = moe_fixture::V4StepInputs {
            tokens: &tokens,
            position: 0,
        };
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            if let Some(value) = moe.bind_exact(meta, step) {
                inputs.insert(id, value);
                continue;
            }
            let value = match &meta.storage {
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        mask.clone(),
                    ))
                }
                Storage::Slot(Slot::Activation) => {
                    let name = meta.name.as_deref().expect("activation without a name");
                    let data = if name == "activation.hca.block_bias" {
                        block_bias.clone()
                    } else if name == "activation.hca.window_positions" {
                        win_positions.clone()
                    } else {
                        panic!("unexpected activation slot {name} in HCA prefill")
                    };
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        data,
                    ))
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    match name {
                        "hca.rope.cos" => poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            w.get("rope.cos").unwrap().clone(),
                        )),
                        "hca.rope.sin" => poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            w.get("rope.sin").unwrap().clone(),
                        )),
                        _ => {
                            let data = w
                                .get(name)
                                .unwrap_or_else(|| panic!("no weight for {name}"))
                                .clone();
                            moe_fixture::bind_const(
                                name,
                                meta.aval.shape.clone(),
                                meta.aval.dtype,
                                data,
                            )
                        }
                    }
                }
                other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
            };
            inputs.insert(id, value);
        }
        let residual = moe_fixture::eval_graph(&cfg, &g, &inputs).expect("eval HCA residual");
        let residual = moe_fixture::dense(&residual);
        assert_eq!(residual.shape(), vec![1, l, cfg.hc_mult, cfg.hidden]);

        let head = trace_deepseek4_hybrid_stack_head(&cfg, &schedule, l)
            .expect("trace_deepseek4_hybrid_stack_head should trace");
        let mut head_inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &head.inputs {
            let meta = head.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data: Vec<f32> = if name == "resid" {
                residual.as_f32().unwrap().to_vec()
            } else {
                w.get(name)
                    .unwrap_or_else(|| panic!("no weight for {name}"))
                    .clone()
            };
            head_inputs.insert(
                id,
                moe_fixture::bind_const(name, meta.aval.shape.clone(), meta.aval.dtype, data),
            );
        }
        let logits = poot_eval::eval(
            &head,
            &head_inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval HCA head")
        .output
        .into_host()
        .expect("HCA head output is dense");
        assert_eq!(logits.shape(), vec![1, l, cfg.vocab]);

        // Card 422: same dual-purpose indexing as the plain prefill oracle; see that comment.
        #[allow(clippy::needless_range_loop)]
        for qpos in 0..tokens.len() {
            let want = &ref_logits[qpos];
            #[allow(clippy::needless_range_loop)]
            for i in 0..cfg.vocab {
                let gv = logits.as_f32().unwrap()[qpos * cfg.vocab + i];
                let wv = want[i];
                assert!(
                    (gv - wv).abs() <= 1e-4 * wv.abs().max(1e-3) + V4_HYBRID_STACK_ABS_FLOOR,
                    "HCA prefill position {qpos} logit {i}: graph={gv} ref={wv}"
                );
            }
        }
    }

    // ---- CSA independent reference. `csa_overlap_pool_ref` verifies the overlapping Ca/Cb window construction
    // standalone (as `hca_softmax_gate_pool_matches_independent_reference` does for HCA). The full-model
    // reference extends `deepseek4_hca_decode_ref` with the compressor's overlapping pool plus the indexer's
    // pool and top-k selection, reusing `attention_core_ref`/`dense_ffn_ref`/`hyper_connection_ref`/etc.
    // unchanged. A not-selected compressed entry is left out of `cache` instead of masked, which is
    // equivalent: `exp(score + HCA_MASK_NEG - max)` underflows to `0.0` in f32, the same as an omitted row.

    /// Independent reference for [`csa_overlap_pool`] (overlapping Ca/Cb windows plus softmax-gated pool), over a
    /// growing per-token history like [`hca_pool_windows_ref`]. `kv_hist`/`gate_hist` rows are `2*head_dim`
    /// wide (`[..head_dim]` = Ca, `[head_dim..]` = Cb). Window `0` pools only its own Cb half (zero-kv /
    /// `HCA_MASK_NEG`-gate prior-Ca pad), as in [`csa_overlap_pool`].
    fn csa_overlap_pool_ref(
        kv_hist: &[Vec<f32>],
        gate_hist: &[Vec<f32>],
        position_bias: &[f32], // [compress_rate * 2*head_dim], row-major [compress_rate][2*head_dim]
        kv_norm_w: &[f32],
        compress_rate: usize,
        head_dim: usize,
        eps: f32,
    ) -> Vec<Vec<f32>> {
        let n_windows = kv_hist.len() / compress_rate;
        let mut out = Vec::with_capacity(n_windows);
        for win in 0..n_windows {
            let cb_kv: Vec<&[f32]> = (0..compress_rate)
                .map(|j| &kv_hist[win * compress_rate + j][head_dim..])
                .collect();
            let cb_gate: Vec<Vec<f32>> = (0..compress_rate)
                .map(|j| {
                    (0..head_dim)
                        .map(|ch| {
                            gate_hist[win * compress_rate + j][head_dim + ch]
                                + position_bias[j * 2 * head_dim + head_dim + ch]
                        })
                        .collect()
                })
                .collect();
            let (ca_kv, ca_gate): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if win == 0 {
                (
                    vec![vec![0.0f32; head_dim]; compress_rate],
                    vec![vec![HCA_MASK_NEG; head_dim]; compress_rate],
                )
            } else {
                let prev = win - 1;
                let kv: Vec<Vec<f32>> = (0..compress_rate)
                    .map(|j| kv_hist[prev * compress_rate + j][..head_dim].to_vec())
                    .collect();
                let gate: Vec<Vec<f32>> = (0..compress_rate)
                    .map(|j| {
                        (0..head_dim)
                            .map(|ch| {
                                gate_hist[prev * compress_rate + j][ch]
                                    + position_bias[j * 2 * head_dim + ch]
                            })
                            .collect()
                    })
                    .collect();
                (kv, gate)
            };

            let mut pooled = vec![0.0f32; head_dim];
            for ch in 0..head_dim {
                let logits: Vec<f32> = ca_gate
                    .iter()
                    .chain(cb_gate.iter())
                    .map(|row| row[ch])
                    .collect();
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
                let sum: f32 = exps.iter().sum();
                let mut acc = 0.0f32;
                for j in 0..compress_rate {
                    acc += (exps[j] / sum) * ca_kv[j][ch];
                }
                for j in 0..compress_rate {
                    acc += (exps[compress_rate + j] / sum) * cb_kv[j][ch];
                }
                pooled[ch] = acc;
            }
            out.push(rmsnorm_ref(&pooled, kv_norm_w, eps));
        }
        out
    }

    #[test]
    fn csa_overlap_pool_matches_independent_reference() {
        let n_windows = 4usize; // >=3 so window 0 (zero/-inf pad), window 1 (Ca of a REAL window 0),
        // and window >=2 (Ca of a non-zero prior window) are all exercised.
        let compress_rate = 3usize;
        let head_dim = 5usize;
        let eps = 1e-5f32;

        let kv_hist: Vec<Vec<f32>> = (0..n_windows * compress_rate)
            .map(|i| fill(2 * head_dim, seed_of(&format!("csakv{i}"))))
            .collect();
        let gate_hist: Vec<Vec<f32>> = (0..n_windows * compress_rate)
            .map(|i| fill(2 * head_dim, seed_of(&format!("csagate{i}"))))
            .collect();
        let position_bias = weight("csapb", compress_rate * 2 * head_dim, false);
        let kv_norm_w = weight("csakvnorm", head_dim, true);

        let want = csa_overlap_pool_ref(
            &kv_hist,
            &gate_hist,
            &position_bias,
            &kv_norm_w,
            compress_rate,
            head_dim,
            eps,
        );

        let b = Builder::new();
        let kv_c = b.constant(
            "kv",
            TensorType::f32(vec![1, n_windows, compress_rate, 2 * head_dim]),
        );
        let gate_c = b.constant(
            "gate",
            TensorType::f32(vec![1, n_windows, compress_rate, 2 * head_dim]),
        );
        let pb_c = b.constant("pb", TensorType::f32(vec![compress_rate, 2 * head_dim]));
        let kvnorm_c = b.constant("kvnorm", TensorType::f32(vec![head_dim]));
        let pooled = csa_overlap_pool(
            &b,
            kv_c,
            gate_c,
            pb_c,
            kvnorm_c,
            n_windows,
            compress_rate,
            head_dim,
            eps,
        );
        let g = b.finish(pooled);

        let kv_flat: Vec<f32> = kv_hist.iter().flatten().copied().collect();
        let gate_flat: Vec<f32> = gate_hist.iter().flatten().copied().collect();
        let mut w: HashMap<String, Vec<f32>> = HashMap::new();
        w.insert("kv".to_string(), kv_flat);
        w.insert("gate".to_string(), gate_flat);
        w.insert("pb".to_string(), position_bias);
        w.insert("kvnorm".to_string(), kv_norm_w);

        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data = w
                .get(name)
                .unwrap_or_else(|| panic!("no weight for {name}"));
            inputs.insert(
                id,
                poot_eval::Value::from(poot_tensor::HostTensor::f32(
                    meta.aval.shape.clone(),
                    data.clone(),
                )),
            );
        }
        let result = poot_eval::eval(
            &g,
            &inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval csa_overlap_pool graph")
        .output
        .into_host()
        .expect("csa_overlap_pool output is dense");
        let got = &result.as_f32().unwrap();

        for w_i in 0..n_windows {
            for ch in 0..head_dim {
                let (gv, wv) = (got[w_i * head_dim + ch], want[w_i][ch]);
                assert!(
                    (gv - wv).abs() <= 1e-4 * wv.abs().max(1e-3),
                    "window {w_i} channel {ch}: graph={gv} ref={wv}"
                );
            }
        }
    }

    /// Every constant an all-CSA V4 graph reads apart from its packed owners; the schedule picks each
    /// compressor's width.
    fn csa_all_weights(cfg: &DeepseekV4Config) -> HashMap<String, Vec<f32>> {
        schedule_weights(
            cfg,
            &vec![V4LayerKind::Csa; cfg.layers],
            cfg.compress_rope_theta,
        )
    }

    /// Independent reference for [`csa_overlap_pool`] + RoPE at window positions, `head_dim`- or
    /// `index_head_dim`-wide (one function serves the outer compressor and the indexer's smaller one).
    #[allow(clippy::too_many_arguments)]
    fn csa_pool_and_rope_ref(
        kv_hist: &[Vec<f32>],
        gate_hist: &[Vec<f32>],
        position_bias: &[f32],
        kv_norm_w: &[f32],
        compress_rate: usize,
        width: usize,
        eps: f32,
        cos: &[f32],
        sin: &[f32],
        rope_d: usize,
    ) -> Vec<Vec<f32>> {
        let pooled = csa_overlap_pool_ref(
            kv_hist,
            gate_hist,
            position_bias,
            kv_norm_w,
            compress_rate,
            width,
            eps,
        );
        let nope_w = width - rope_d;
        pooled
            .iter()
            .enumerate()
            .map(|(win, row)| {
                let pos = win * compress_rate;
                let mut rope_part = row[nope_w..].to_vec();
                rope_part = rope_interleaved_ref(&rope_part, cos, sin, pos, rope_d, false);
                let mut entry = row[..nope_w].to_vec();
                entry.extend(rope_part);
                entry
            })
            .collect()
    }

    /// Independent reference for CSA's per-query compressed-entry selection: causal-completeness validity
    /// (`query_pos >= win*cr + cr - 1`, as [`hca_decode_validity_mask`]) combined with the indexer's top-`k`
    /// over `index_scores + causal bias`, using the rank definition of `crate::deepseek3::pairwise_rank`/
    /// `keep_top_k_mask` (`rank[i] = count_j(ranked[j] > ranked[i])`, strict, so ties share a rank and are
    /// kept together; `rank < k`). Returns the indices of entries this query may attend to. Dropping others
    /// from `cache` matches the graph's additive `HCA_MASK_NEG` masking (see the section header).
    #[allow(clippy::too_many_arguments)]
    fn csa_selected_window_indices_ref(
        query_pos: usize,
        n_windows: usize,
        compress_rate: usize,
        index_scores: &[f32],
        topk: usize,
    ) -> Vec<usize> {
        let ranked: Vec<f32> = (0..n_windows)
            .map(|win| {
                let causal_bias = if query_pos >= win * compress_rate + compress_rate - 1 {
                    0.0
                } else {
                    HCA_MASK_NEG
                };
                index_scores[win] + causal_bias
            })
            .collect();
        (0..n_windows)
            .filter(|&i| {
                let rank = (0..n_windows).filter(|&j| ranked[j] > ranked[i]).count();
                let causally_valid = query_pos >= i * compress_rate + compress_rate - 1;
                rank < topk && causally_valid
            })
            .collect()
    }

    /// Independent reference for [`csa_indexer_scores`]: `sum_h softmax_scale * ReLU(q_h . k_win) *
    /// weights_scaling * w_idx_h`, per window.
    #[allow(clippy::too_many_arguments)]
    fn csa_indexer_scores_ref(
        q_idx_heads: &[Vec<f32>],    // [Hi][Di]
        idx_compressed: &[Vec<f32>], // [n_windows][Di]
        w_idx: &[f32],               // [Hi]
        softmax_scale: f32,
        weights_scaling: f32,
    ) -> Vec<f32> {
        idx_compressed
            .iter()
            .map(|k_win| {
                q_idx_heads
                    .iter()
                    .zip(w_idx.iter())
                    .map(|(q_h, &w_h)| {
                        let dot: f32 = q_h.iter().zip(k_win.iter()).map(|(a, b)| a * b).sum();
                        dot.max(0.0) * softmax_scale * (w_h * weights_scaling)
                    })
                    .sum::<f32>()
            })
            .collect()
    }

    /// Full independent-reference decode for an all-CSA model: [`deepseek4_hca_decode_ref`]'s structure with
    /// CSA's top-`k`-selected overlapping-window branch appended to the local `cache` slice before
    /// [`attention_core_ref`] runs.
    fn deepseek4_csa_decode_ref(
        cfg: &DeepseekV4Config,
        w: &HashMap<String, Vec<f32>>,
        moe: &moe_fixture::MoeFixture,
        tokens: &[usize],
    ) -> Vec<Vec<f32>> {
        let (h, hc) = (cfg.hidden, cfg.hc_mult);
        let embed = w.get("embed.weight").unwrap();
        let cr = cfg.csa_compress_rate;
        let (hi, di) = (cfg.index_n_heads, cfg.index_head_dim);
        let softmax_scale = 1.0 / (di as f32).sqrt();
        let weights_scaling = 1.0 / (hi as f32).sqrt();
        let cos = w.get("rope.cos").unwrap().clone();
        let sin = w.get("rope.sin").unwrap().clone();

        let mut caches: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut csa_kv_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut csa_gate_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut idx_kv_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut idx_gate_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut all_logits = Vec::new();

        for (qpos, &tok) in tokens.iter().enumerate() {
            let e = embed[tok * h..(tok + 1) * h].to_vec();
            let mut streams: Vec<Vec<f32>> = (0..hc).map(|_| e.clone()).collect();

            for (li, cache) in caches.iter_mut().enumerate() {
                let p = |s: &str| format!("layers.{li}.{s}");
                let (_pre, post, comb, collapsed) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_attn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_attn_base")).unwrap(),
                    w.get(&p("hc_attn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed =
                    rmsnorm_ref(&collapsed, w.get(&p("attn_norm.weight")).unwrap(), cfg.eps);

                let wkv = moe.weight(&p("attn.wkv"));
                let kv_raw = linear_ref_out_in(&normed, &wkv, cfg.hidden, cfg.head_dim);
                let kv_n = rmsnorm_ref(&kv_raw, w.get(&p("attn.kv_norm.weight")).unwrap(), cfg.eps);
                let nope_w = cfg.head_dim - cfg.qk_rope_head_dim;
                let mut kv_rope = kv_n[nope_w..].to_vec();
                kv_rope =
                    rope_interleaved_ref(&kv_rope, &cos, &sin, qpos, cfg.qk_rope_head_dim, false);
                let mut kv_new = kv_n[..nope_w].to_vec();
                kv_new.extend(kv_rope);
                cache.push(kv_new);
                let window_start = qpos.saturating_sub(cfg.sliding_window - 1);
                let mut combined_cache = cache[window_start..=qpos].to_vec();

                // Compressor's overlapping-window pooled KV, at head_dim.
                let ckv = w.get(&p("attn.compressor.wkv.weight")).unwrap();
                let cgate = w.get(&p("attn.compressor.wgate.weight")).unwrap();
                let raw_kv = linear_ref_out_in(&normed, ckv, cfg.hidden, 2 * cfg.head_dim);
                let raw_gate = linear_ref_out_in(&normed, cgate, cfg.hidden, 2 * cfg.head_dim);
                csa_kv_hist[li].push(raw_kv);
                csa_gate_hist[li].push(raw_gate);
                let compressed = csa_pool_and_rope_ref(
                    &csa_kv_hist[li],
                    &csa_gate_hist[li],
                    w.get(&p("attn.compressor.ape")).unwrap(),
                    w.get(&p("attn.compressor.norm.weight")).unwrap(),
                    cr,
                    cfg.head_dim,
                    cfg.eps,
                    &cos,
                    &sin,
                    cfg.qk_rope_head_dim,
                );

                // Indexer's overlapping-window pooled keys, at index_head_dim.
                let ikv = w.get(&p("attn.indexer.compressor.wkv.weight")).unwrap();
                let igate = w.get(&p("attn.indexer.compressor.wgate.weight")).unwrap();
                let idx_raw_kv = linear_ref_out_in(&normed, ikv, cfg.hidden, 2 * di);
                let idx_raw_gate = linear_ref_out_in(&normed, igate, cfg.hidden, 2 * di);
                idx_kv_hist[li].push(idx_raw_kv);
                idx_gate_hist[li].push(idx_raw_gate);
                let idx_compressed = csa_pool_and_rope_ref(
                    &idx_kv_hist[li],
                    &idx_gate_hist[li],
                    w.get(&p("attn.indexer.compressor.ape")).unwrap(),
                    w.get(&p("attn.indexer.compressor.norm.weight")).unwrap(),
                    cr,
                    di,
                    cfg.eps,
                    &cos,
                    &sin,
                    cfg.qk_rope_head_dim,
                );
                let n_windows = idx_compressed.len();

                // Indexer's query, from q_residual (the `q_a_norm` output `attention_core_ref` also recomputes for
                // the main Q; duplicated here rather than reaching into its internals).
                let q_a =
                    linear_ref_out_in(&normed, &moe.weight(&p("attn.wq_a")), h, cfg.q_lora_rank);
                let q_res = rmsnorm_ref(&q_a, w.get(&p("attn.q_norm.weight")).unwrap(), cfg.eps);
                let iwqb = moe.weight(&p("attn.indexer.wq_b"));
                let q_idx_flat = linear_ref_out_in(&q_res, &iwqb, cfg.q_lora_rank, hi * di);
                let idx_nope_w = di - cfg.qk_rope_head_dim;
                let q_idx_heads: Vec<Vec<f32>> = (0..hi)
                    .map(|hix| {
                        let qh = &q_idx_flat[hix * di..(hix + 1) * di];
                        let mut rope_part = qh[idx_nope_w..].to_vec();
                        rope_part = rope_interleaved_ref(
                            &rope_part,
                            &cos,
                            &sin,
                            qpos,
                            cfg.qk_rope_head_dim,
                            false,
                        );
                        let mut out = qh[..idx_nope_w].to_vec();
                        out.extend(rope_part);
                        out
                    })
                    .collect();

                let iww = w.get(&p("attn.indexer.weights_proj.weight")).unwrap();
                let w_idx = linear_ref_out_in(&normed, iww, h, hi);

                let index_scores = csa_indexer_scores_ref(
                    &q_idx_heads,
                    &idx_compressed,
                    &w_idx,
                    softmax_scale,
                    weights_scaling,
                );
                let topk = cfg.index_topk.min(n_windows);
                let selected =
                    csa_selected_window_indices_ref(qpos, n_windows, cr, &index_scores, topk);
                combined_cache.extend(selected.into_iter().map(|win| compressed[win].clone()));

                let attn_out = attention_core_ref(cfg, w, moe, li, &normed, &combined_cache, qpos);
                streams = hyper_connection_combine_ref(&post, &comb, &attn_out, &streams, hc, h);

                let (_pre2, post2, comb2, collapsed2) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_ffn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_ffn_base")).unwrap(),
                    w.get(&p("hc_ffn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed2 =
                    rmsnorm_ref(&collapsed2, w.get(&p("ffn_norm.weight")).unwrap(), cfg.eps);
                let mlp_out = moe.ffn_ref(cfg, w, li, tok, &normed2);
                streams = hyper_connection_combine_ref(&post2, &comb2, &mlp_out, &streams, hc, h);
            }

            let hh_scale = w.get("hc_head_scale").unwrap()[0];
            let collapsed_final = hyper_head_ref(
                &streams,
                &mhc_projection_ref(w, "hc_head_fn", hc, hc * h),
                w.get("hc_head_base").unwrap(),
                hh_scale,
                hc,
                h,
                cfg.eps,
                cfg.hc_eps,
            );
            let xf = rmsnorm_ref(&collapsed_final, w.get("norm.weight").unwrap(), cfg.eps);
            let logits = linear_ref_out_in(&xf, w.get("head.weight").unwrap(), h, cfg.vocab);
            all_logits.push(logits);
        }
        all_logits
    }

    /// SC-004 for CSA: eight steps at `csa_compress_rate = 2` and capacity eight, so all four compressed windows
    /// are built with `index_topk = 2` restricting the selection, while the `sliding_window = 3` local floor
    /// fires past capacity.
    #[test]
    fn deepseek4_csa_decode_matches_independent_reference_across_multiple_windows() {
        let cfg = tiny_cfg();
        let cap = 8;
        let tokens = [1usize, 2, 3, 4, 5, 6, 7, 8];
        assert!(
            cfg.index_topk < cap / cfg.csa_compress_rate,
            "the indexer top-k must restrict the windows it picks from"
        );
        let w = csa_all_weights(&cfg);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &vec![V4LayerKind::Csa; cfg.layers]);
        let want = deepseek4_csa_decode_ref(&cfg, &w, &moe, &tokens);

        let g = trace_deepseek4_csa_decode(cfg, cap).expect("csa decode traces");
        let got = moe_fixture::decode_logits(&cfg, &g, &moe, &w, &tokens, |name| match name {
            "hca.rope.cos" => Some(w.get("rope.cos").unwrap().clone()),
            "hca.rope.sin" => Some(w.get("rope.sin").unwrap().clone()),
            _ => None,
        });
        moe_fixture::assert_decode_logits(&got, &want, "csa decode", 0.0);
    }

    /// Checked at every position, not only the last (Card 386 P2; see
    /// `deepseek4_sliding_prefill_matches_independent_reference`).
    #[test]
    fn deepseek4_csa_prefill_matches_independent_reference() {
        let cfg = tiny_cfg();
        let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Csa);
        let l = 8usize; // multiple of csa_compress_rate=2
        let tokens = [1usize, 2, 3, 4, 5, 6, 7, 8];
        let w = csa_all_weights(&cfg);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &schedule);
        let ref_logits = deepseek4_csa_decode_ref(&cfg, &w, &moe, &tokens);

        let g = trace_deepseek4_hybrid_stack_prefill_residual_upto(cfg, l, &schedule, cfg.layers)
            .expect("trace_deepseek4_hybrid_stack_prefill_residual_upto should trace");
        let mut mask = vec![-1.0e9f32; l * l];
        for i in 0..l {
            let start = i.saturating_sub(cfg.sliding_window - 1);
            for j in start..=i {
                mask[i * l + j] = 0.0;
            }
        }
        let n_windows = l / cfg.csa_compress_rate;
        let win_positions: Vec<f32> = (0..n_windows)
            .map(|w_i| (w_i * cfg.csa_compress_rate) as f32)
            .collect();
        let mut block_bias = vec![0.0f32; l * n_windows];
        for i in 0..l {
            for wi in 0..n_windows {
                if (i as f32) < win_positions[wi] + (cfg.csa_compress_rate - 1) as f32 {
                    block_bias[i * n_windows + wi] = HCA_MASK_NEG;
                }
            }
        }

        let step = moe_fixture::V4StepInputs {
            tokens: &tokens,
            position: 0,
        };
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            if let Some(value) = moe.bind_exact(meta, step) {
                inputs.insert(id, value);
                continue;
            }
            let value = match &meta.storage {
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        mask.clone(),
                    ))
                }
                Storage::Slot(Slot::Activation) => {
                    let name = meta.name.as_deref().expect("activation without a name");
                    let data = if name == "activation.csa.block_bias" {
                        block_bias.clone()
                    } else if name == "activation.csa.window_positions" {
                        win_positions.clone()
                    } else {
                        panic!("unexpected activation slot {name} in CSA prefill")
                    };
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        data,
                    ))
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    match name {
                        "hca.rope.cos" => poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            w.get("rope.cos").unwrap().clone(),
                        )),
                        "hca.rope.sin" => poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            w.get("rope.sin").unwrap().clone(),
                        )),
                        _ => {
                            let data = crate::model_fixture_data(&w, name);
                            moe_fixture::bind_const(
                                name,
                                meta.aval.shape.clone(),
                                meta.aval.dtype,
                                data,
                            )
                        }
                    }
                }
                other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
            };
            inputs.insert(id, value);
        }
        let residual = moe_fixture::eval_graph(&cfg, &g, &inputs).expect("eval CSA residual");
        let residual = moe_fixture::dense(&residual);
        assert_eq!(residual.shape(), vec![1, l, cfg.hc_mult, cfg.hidden]);

        let head = trace_deepseek4_hybrid_stack_head(&cfg, &schedule, l)
            .expect("trace_deepseek4_hybrid_stack_head should trace");
        let mut head_inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &head.inputs {
            let meta = head.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data: Vec<f32> = if name == "resid" {
                residual.as_f32().unwrap().to_vec()
            } else {
                crate::model_fixture_data(&w, name)
            };
            head_inputs.insert(
                id,
                moe_fixture::bind_const(name, meta.aval.shape.clone(), meta.aval.dtype, data),
            );
        }
        let logits = poot_eval::eval(
            &head,
            &head_inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval CSA head")
        .output
        .into_host()
        .expect("CSA head output is dense");
        assert_eq!(logits.shape(), vec![1, l, cfg.vocab]);

        // Card 422: same dual-purpose indexing as the plain prefill oracle; see that comment.
        #[allow(clippy::needless_range_loop)]
        for qpos in 0..tokens.len() {
            let want = &ref_logits[qpos];
            #[allow(clippy::needless_range_loop)]
            for i in 0..cfg.vocab {
                let gv = logits.as_f32().unwrap()[qpos * cfg.vocab + i];
                let wv = want[i];
                assert!(
                    (gv - wv).abs() <= 1e-4 * wv.abs().max(1e-3) + V4_HYBRID_STACK_ABS_FLOOR,
                    "CSA prefill position {qpos} logit {i}: graph={gv} ref={wv}"
                );
            }
        }
    }

    // ---- Mixed-schedule (spec 281) independent reference. Reuses every per-kind reference helper above
    // (`attention_core_ref`, `hca_pool_windows_ref`, `csa_pool_and_rope_ref`, `csa_indexer_scores_ref`,
    // `csa_selected_window_indices_ref`, `dense_ffn_ref`, `hyper_connection_ref`/`_combine_ref`/
    // `hyper_head_ref`) unchanged. `attention_core_ref`/`hca_pool_windows_ref` read `w["rope.cos"]`/
    // `w["rope.sin"]`, which suits a single-table model; a mixed schedule needs a different table per layer,
    // so this function swaps those two keys to the layer's table (from `main_cos`/`main_sin` and the compress
    // table `csa_all_weights` stores) before each layer's helper calls.
    //
    // Card 434: `too_many_arguments` (9/7) is left as an `#[allow]` rather than a params struct: every
    // parameter is an independent typed input, the last two are opt-in Card 386 diagnostic capture slots
    // (`None` for ordinary oracle calls), and this test-only function has a handful of call sites.
    #[allow(clippy::too_many_arguments)]
    fn deepseek4_hybrid_stack_decode_ref(
        cfg: &DeepseekV4Config,
        schedule: &[V4LayerKind],
        w: &mut HashMap<String, Vec<f32>>,
        moe: &moe_fixture::MoeFixture,
        main_cos: &[f32],
        main_sin: &[f32],
        tokens: &[usize],
        // Card 386 bisection tool: when `Some`, every position's widened residual (`[stream][hidden]`) is
        // snapshotted after each layer's FFN combine, not just the final logits. `None` for ordinary oracles.
        mut layer_residuals: Option<&mut Vec<Vec<Vec<Vec<f32>>>>>,
        // Card 386 hypothesis 4 (MoE amplification): when `Some`, every position's residual is also snapshotted
        // right after each layer's attention combine, before the FFN/MoE half, to show whether a divergence
        // exists before the router's `sqrt(softplus(.))`. No graph-side counterpart exists (the per-layer
        // tracers run attention and FFN as one call; exposing the midpoint would duplicate half of each body or
        // change shared production functions). Same `None`-by-default rule.
        mut attn_residuals: Option<&mut Vec<Vec<Vec<Vec<f32>>>>>,
    ) -> Vec<Vec<f32>> {
        let (h, hc) = (cfg.hidden, cfg.hc_mult);
        let embed = w.get("embed.weight").unwrap().clone();
        let compress_cos = w.get("rope.cos").unwrap().clone();
        let compress_sin = w.get("rope.sin").unwrap().clone();
        let cr_hca = cfg.hca_compress_rate;
        let cr_csa = cfg.csa_compress_rate;
        let (hi, di) = (cfg.index_n_heads, cfg.index_head_dim);
        let softmax_scale = 1.0 / (di as f32).sqrt();
        let weights_scaling = 1.0 / (hi as f32).sqrt();
        let nope_w = cfg.head_dim - cfg.qk_rope_head_dim;

        let mut caches: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut hca_kv_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut hca_gate_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut csa_kv_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut csa_gate_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut idx_kv_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut idx_gate_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); cfg.layers];
        let mut all_logits = Vec::new();

        for (qpos, &tok) in tokens.iter().enumerate() {
            let e = embed[tok * h..(tok + 1) * h].to_vec();
            let mut streams: Vec<Vec<f32>> = (0..hc).map(|_| e.clone()).collect();
            let mut residuals_this_position: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut attn_residuals_this_position: Vec<Vec<Vec<f32>>> = Vec::new();

            for (li, kind) in schedule.iter().enumerate() {
                let p = |s: &str| format!("layers.{li}.{s}");
                let (cos, sin): (&[f32], &[f32]) = match kind {
                    V4LayerKind::Sliding => (main_cos, main_sin),
                    V4LayerKind::Hca | V4LayerKind::Csa => (&compress_cos, &compress_sin),
                };
                w.insert("rope.cos".to_string(), cos.to_vec());
                w.insert("rope.sin".to_string(), sin.to_vec());

                let (_pre, post, comb, collapsed) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_attn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_attn_base")).unwrap(),
                    w.get(&p("hc_attn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed =
                    rmsnorm_ref(&collapsed, w.get(&p("attn_norm.weight")).unwrap(), cfg.eps);

                let wkv = moe.weight(&p("attn.wkv"));
                let kv_raw = linear_ref_out_in(&normed, &wkv, cfg.hidden, cfg.head_dim);
                let kv_n = rmsnorm_ref(&kv_raw, w.get(&p("attn.kv_norm.weight")).unwrap(), cfg.eps);
                let mut kv_rope = kv_n[nope_w..].to_vec();
                kv_rope =
                    rope_interleaved_ref(&kv_rope, cos, sin, qpos, cfg.qk_rope_head_dim, false);
                let mut kv_new = kv_n[..nope_w].to_vec();
                kv_new.extend(kv_rope);
                caches[li].push(kv_new);
                let window_start = qpos.saturating_sub(cfg.sliding_window - 1);
                let mut combined_cache = caches[li][window_start..=qpos].to_vec();

                match kind {
                    V4LayerKind::Sliding => {}
                    V4LayerKind::Hca => {
                        let hkv = w.get(&p("attn.compressor.wkv.weight")).unwrap();
                        let hgate = w.get(&p("attn.compressor.wgate.weight")).unwrap();
                        let raw_kv = linear_ref_out_in(&normed, hkv, cfg.hidden, cfg.head_dim);
                        let raw_gate = linear_ref_out_in(&normed, hgate, cfg.hidden, cfg.head_dim);
                        hca_kv_hist[li].push(raw_kv);
                        hca_gate_hist[li].push(raw_gate);
                        let compressed =
                            hca_pool_windows_ref(cfg, w, li, &hca_kv_hist[li], &hca_gate_hist[li]);
                        combined_cache.extend(compressed);
                    }
                    V4LayerKind::Csa => {
                        let ckv = w.get(&p("attn.compressor.wkv.weight")).unwrap();
                        let cgate = w.get(&p("attn.compressor.wgate.weight")).unwrap();
                        let raw_kv = linear_ref_out_in(&normed, ckv, cfg.hidden, 2 * cfg.head_dim);
                        let raw_gate =
                            linear_ref_out_in(&normed, cgate, cfg.hidden, 2 * cfg.head_dim);
                        csa_kv_hist[li].push(raw_kv);
                        csa_gate_hist[li].push(raw_gate);
                        let compressed = csa_pool_and_rope_ref(
                            &csa_kv_hist[li],
                            &csa_gate_hist[li],
                            w.get(&p("attn.compressor.ape")).unwrap(),
                            w.get(&p("attn.compressor.norm.weight")).unwrap(),
                            cr_csa,
                            cfg.head_dim,
                            cfg.eps,
                            cos,
                            sin,
                            cfg.qk_rope_head_dim,
                        );

                        let ikv = w.get(&p("attn.indexer.compressor.wkv.weight")).unwrap();
                        let igate = w.get(&p("attn.indexer.compressor.wgate.weight")).unwrap();
                        let idx_raw_kv = linear_ref_out_in(&normed, ikv, cfg.hidden, 2 * di);
                        let idx_raw_gate = linear_ref_out_in(&normed, igate, cfg.hidden, 2 * di);
                        idx_kv_hist[li].push(idx_raw_kv);
                        idx_gate_hist[li].push(idx_raw_gate);
                        let idx_compressed = csa_pool_and_rope_ref(
                            &idx_kv_hist[li],
                            &idx_gate_hist[li],
                            w.get(&p("attn.indexer.compressor.ape")).unwrap(),
                            w.get(&p("attn.indexer.compressor.norm.weight")).unwrap(),
                            cr_csa,
                            di,
                            cfg.eps,
                            cos,
                            sin,
                            cfg.qk_rope_head_dim,
                        );
                        let n_windows = idx_compressed.len();

                        let q_a = linear_ref_out_in(
                            &normed,
                            &moe.weight(&p("attn.wq_a")),
                            h,
                            cfg.q_lora_rank,
                        );
                        let q_res =
                            rmsnorm_ref(&q_a, w.get(&p("attn.q_norm.weight")).unwrap(), cfg.eps);
                        let iwqb = moe.weight(&p("attn.indexer.wq_b"));
                        let q_idx_flat = linear_ref_out_in(&q_res, &iwqb, cfg.q_lora_rank, hi * di);
                        let idx_nope_w = di - cfg.qk_rope_head_dim;
                        let q_idx_heads: Vec<Vec<f32>> = (0..hi)
                            .map(|hix| {
                                let qh = &q_idx_flat[hix * di..(hix + 1) * di];
                                let mut rope_part = qh[idx_nope_w..].to_vec();
                                rope_part = rope_interleaved_ref(
                                    &rope_part,
                                    cos,
                                    sin,
                                    qpos,
                                    cfg.qk_rope_head_dim,
                                    false,
                                );
                                let mut out = qh[..idx_nope_w].to_vec();
                                out.extend(rope_part);
                                out
                            })
                            .collect();

                        let iww = w.get(&p("attn.indexer.weights_proj.weight")).unwrap();
                        let w_idx = linear_ref_out_in(&normed, iww, h, hi);

                        let index_scores = csa_indexer_scores_ref(
                            &q_idx_heads,
                            &idx_compressed,
                            &w_idx,
                            softmax_scale,
                            weights_scaling,
                        );
                        let topk = cfg.index_topk.min(n_windows);
                        let selected = csa_selected_window_indices_ref(
                            qpos,
                            n_windows,
                            cr_csa,
                            &index_scores,
                            topk,
                        );
                        combined_cache
                            .extend(selected.into_iter().map(|win| compressed[win].clone()));
                    }
                }

                let attn_out = attention_core_ref(cfg, w, moe, li, &normed, &combined_cache, qpos);
                streams = hyper_connection_combine_ref(&post, &comb, &attn_out, &streams, hc, h);
                if attn_residuals.is_some() {
                    attn_residuals_this_position.push(streams.clone());
                }

                let (_pre2, post2, comb2, collapsed2) = hyper_connection_ref(
                    &streams,
                    &mhc_projection_ref(w, &p("hc_ffn_fn"), cfg.hc_mix_dim(), hc * h),
                    w.get(&p("hc_ffn_base")).unwrap(),
                    w.get(&p("hc_ffn_scale")).unwrap(),
                    hc,
                    h,
                    cfg.eps,
                    cfg.hc_eps,
                    cfg.hc_sinkhorn_iters,
                );
                let normed2 =
                    rmsnorm_ref(&collapsed2, w.get(&p("ffn_norm.weight")).unwrap(), cfg.eps);
                let mlp_out = moe.ffn_ref(cfg, w, li, tok, &normed2);
                streams = hyper_connection_combine_ref(&post2, &comb2, &mlp_out, &streams, hc, h);
                if layer_residuals.is_some() {
                    residuals_this_position.push(streams.clone());
                }
            }
            if let Some(out) = layer_residuals.as_deref_mut() {
                out.push(residuals_this_position);
            }
            if let Some(out) = attn_residuals.as_deref_mut() {
                out.push(attn_residuals_this_position);
            }

            let hh_scale = w.get("hc_head_scale").unwrap()[0];
            let collapsed_final = hyper_head_ref(
                &streams,
                &mhc_projection_ref(w, "hc_head_fn", hc, hc * h),
                w.get("hc_head_base").unwrap(),
                hh_scale,
                hc,
                h,
                cfg.eps,
                cfg.hc_eps,
            );
            let xf = rmsnorm_ref(&collapsed_final, w.get("norm.weight").unwrap(), cfg.eps);
            let logits = linear_ref_out_in(&xf, w.get("head.weight").unwrap(), h, cfg.vocab);
            all_logits.push(logits);
        }
        // `hca_compress_rate` is read only via `cr_hca` in `hca_pool_windows_ref`'s caller; kept as a local for
        // symmetry with `cr_csa`.
        let _ = cr_hca;
        all_logits
    }

    /// SC-001-style graph-validation check for the mixed-schedule stack: a 3-layer [Sliding, Hca, Csa] schedule
    /// produces a valid graph with the expected per-kind state count (1 + 3 + 5 = 9 state tensors).
    #[test]
    fn deepseek4_hybrid_stack_decode_validates_with_mixed_state_tensor_count() {
        let cfg = DeepseekV4Config {
            layers: 3,
            ..tiny_cfg()
        };
        let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
        let cap = 8; // multiple of both hca_compress_rate=2 and csa_compress_rate=2
        let g = trace_deepseek4_hybrid_stack_decode(cfg, cap, &schedule)
            .expect("trace_deepseek4_hybrid_stack_decode should trace");
        g.validate()
            .expect("deepseek4 hybrid stack decode graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
        assert_eq!(
            g.state.len(),
            1 + 3 + 5,
            "sliding(1) + hca(3) + csa(5) state tensors, one layer of each kind"
        );
    }

    #[test]
    fn deepseek4_hybrid_stack_prefill_validates() {
        let cfg = DeepseekV4Config {
            layers: 3,
            ..tiny_cfg()
        };
        let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
        let g = trace_deepseek4_hybrid_stack_prefill(cfg, 8, &schedule)
            .expect("trace_deepseek4_hybrid_stack_prefill should trace"); // multiple of both rates
        g.validate()
            .expect("deepseek4 hybrid stack prefill graph should validate");
        assert_eq!(g.aval(g.output).shape, vec![1, 1, cfg.vocab]);
    }

    /// `parse_compress_ratios` against the real `DeepSeek-V4-Flash-0731` `compress_ratios` schedule (46 entries:
    /// `sliding_attention` for the first 2 and last 3 layers, mostly alternating CSA(4)/HCA(128) between).
    #[test]
    fn parse_compress_ratios_matches_real_flash_0731_schedule() {
        let cfg = DeepseekV4Config {
            csa_compress_rate: 4,
            hca_compress_rate: 128,
            ..tiny_cfg()
        };
        #[rustfmt::skip]
        let ratios: [usize; 46] = [
            0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
            128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0,
        ];
        let schedule = parse_compress_ratios(&cfg, &ratios);
        assert_eq!(schedule.len(), 46);
        assert_eq!(
            &schedule[0..2],
            [V4LayerKind::Sliding, V4LayerKind::Sliding]
        );
        assert_eq!(
            &schedule[43..46],
            [
                V4LayerKind::Sliding,
                V4LayerKind::Sliding,
                V4LayerKind::Sliding
            ]
        );
        let csa = schedule.iter().filter(|k| **k == V4LayerKind::Csa).count();
        let hca = schedule.iter().filter(|k| **k == V4LayerKind::Hca).count();
        let sliding = schedule
            .iter()
            .filter(|k| **k == V4LayerKind::Sliding)
            .count();
        assert_eq!(
            csa, 21,
            "21 real '4' entries (20 alternating + 1 trailing before the last 3 zeros)"
        );
        assert_eq!(hca, 20, "20 real '128' entries");
        assert_eq!(sliding, 5, "the real first-2 + last-3 '0' entries");
        assert_eq!(csa + hca + sliding, 46);
    }

    #[test]
    #[should_panic(expected = "matches neither 0")]
    fn parse_compress_ratios_rejects_unknown_ratio() {
        let cfg = tiny_cfg();
        parse_compress_ratios(&cfg, &[0, 4, 128, 7]);
    }

    /// SC-004 for a mixed schedule: one Sliding, one HCA and one CSA layer, eight steps at a capacity both
    /// compress rates divide. Every cache kind is carried at once, so a layer that wrote its cache at the wrong
    /// position or read a stale one shows up here and in no single-kind row.
    ///
    /// Card 386: a ~5e-6 absolute divergence at position 1 once failed the `1e-4 * want.abs().max(1e-3)` bar on
    /// small logits. It came from two universal f32 non-bit-exactness sources (`rmsnorm_core`'s
    /// reciprocal-vs-divide and `attention_prefill_with_sink`'s sink-term order) and no longer reproduces. The
    /// per-`(layer, position)` bisection (`deepseek4_hybrid_stack_prefill_bisects_by_layer_and_position`)
    /// shows every cell at or under `V4_HYBRID_STACK_ABS_FLOOR`, i.e. ordinary rounding. See
    /// specs/386-deepseek4-hybrid-stack-reference-disagreement/spec.md.
    #[test]
    fn deepseek4_hybrid_stack_decode_matches_independent_reference_mixed_schedule() {
        let cfg = DeepseekV4Config {
            layers: 3,
            ..tiny_cfg()
        };
        let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
        let cap = 8;
        let tokens = [1usize, 3, 5, 2, 7, 4, 6, 0];

        let mut w = schedule_weights(&cfg, &schedule, cfg.compress_rope_theta);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &schedule);
        let (main_cos, main_sin) = deepseek2_rope_tables_interleaved(
            cfg.max_pos,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            None,
        );
        let want = deepseek4_hybrid_stack_decode_ref(
            &cfg, &schedule, &mut w, &moe, &main_cos, &main_sin, &tokens, None, None,
        );

        let g = trace_deepseek4_hybrid_stack_decode(cfg, cap, &schedule)
            .expect("hybrid stack decode traces");
        let got = moe_fixture::decode_logits(&cfg, &g, &moe, &w, &tokens, |name| match name {
            "rope.cos" => Some(main_cos.clone()),
            "rope.sin" => Some(main_sin.clone()),
            "hca.rope.cos" => Some(w.get("rope.cos").unwrap().clone()),
            "hca.rope.sin" => Some(w.get("rope.sin").unwrap().clone()),
            _ => None,
        });
        moe_fixture::assert_decode_logits(
            &got,
            &want,
            "hybrid stack decode",
            V4_HYBRID_STACK_ABS_FLOOR,
        );
    }

    /// SC-009-style prefill twin: full 8-token prefill (a multiple of both compress rates), cross-checked at
    /// every position, not only the last. A last-position-only check once stayed green through a real
    /// position-1 divergence (Card 386).
    ///
    /// Built from `trace_deepseek4_hybrid_stack_prefill_residual_upto` at the full layer count plus
    /// `trace_deepseek4_hybrid_stack_head`, so a full-depth residual becomes logits for every position in one
    /// more small eval. Positions are checked in ascending order, so a failing `assert!` names the first
    /// diverging position.
    ///
    /// Card 386: not `#[ignore]`d. The reported ~5e-6 divergence does not reproduce. `V4_HYBRID_STACK_ABS_FLOOR`
    /// is applied anyway: the bisection tool still finds two residual cells (layer 3, position 4) above the bare
    /// `1e-4` relative bar, at up to ~7.7e-6 absolute, under the derived floor and consistent with f32
    /// rounding. See specs/386-deepseek4-hybrid-stack-reference-disagreement/spec.md.
    #[test]
    fn deepseek4_hybrid_stack_prefill_matches_independent_reference_mixed_schedule() {
        let cfg = DeepseekV4Config {
            layers: 3,
            ..tiny_cfg()
        };
        let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
        let l = 8usize;
        let tokens = [1usize, 3, 5, 2, 7, 4, 6, 0];

        let mut w = schedule_weights(&cfg, &schedule, cfg.compress_rope_theta);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &schedule);
        let (main_cos, main_sin) = deepseek2_rope_tables_interleaved(
            cfg.max_pos,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            None,
        );
        let ref_logits = deepseek4_hybrid_stack_decode_ref(
            &cfg, &schedule, &mut w, &moe, &main_cos, &main_sin, &tokens, None, None,
        );

        let g =
            trace_deepseek4_hybrid_stack_prefill_residual_upto(cfg, l, &schedule, schedule.len())
                .expect("trace_deepseek4_hybrid_stack_prefill_residual_upto should trace");

        let mut mask = vec![-1.0e9f32; l * l];
        for i in 0..l {
            let start = i.saturating_sub(cfg.sliding_window - 1);
            for j in start..=i {
                mask[i * l + j] = 0.0;
            }
        }
        let n_windows_hca = l / cfg.hca_compress_rate;
        let hca_win_positions: Vec<f32> = (0..n_windows_hca)
            .map(|w_i| (w_i * cfg.hca_compress_rate) as f32)
            .collect();
        let mut hca_block_bias = vec![0.0f32; l * n_windows_hca];
        for i in 0..l {
            for wi in 0..n_windows_hca {
                if (i as f32) < hca_win_positions[wi] + (cfg.hca_compress_rate - 1) as f32 {
                    hca_block_bias[i * n_windows_hca + wi] = HCA_MASK_NEG;
                }
            }
        }
        let n_windows_csa = l / cfg.csa_compress_rate;
        let csa_win_positions: Vec<f32> = (0..n_windows_csa)
            .map(|w_i| (w_i * cfg.csa_compress_rate) as f32)
            .collect();
        let mut csa_block_bias = vec![0.0f32; l * n_windows_csa];
        for i in 0..l {
            for wi in 0..n_windows_csa {
                if (i as f32) < csa_win_positions[wi] + (cfg.csa_compress_rate - 1) as f32 {
                    csa_block_bias[i * n_windows_csa + wi] = HCA_MASK_NEG;
                }
            }
        }

        let step = moe_fixture::V4StepInputs {
            tokens: &tokens,
            position: 0,
        };
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            if let Some(value) = moe.bind_exact(meta, step) {
                inputs.insert(id, value);
                continue;
            }
            let value = match &meta.storage {
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        mask.clone(),
                    ))
                }
                Storage::Slot(Slot::Activation) => {
                    let name = meta.name.as_deref().expect("activation without a name");
                    let data: Vec<f32> = match name {
                        "activation.hca.block_bias" => hca_block_bias.clone(),
                        "activation.csa.block_bias" => csa_block_bias.clone(),
                        "activation.hca.window_positions" => hca_win_positions.clone(),
                        "activation.csa.window_positions" => csa_win_positions.clone(),
                        _ => panic!("unexpected activation slot {name} in hybrid prefill"),
                    };
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        data,
                    ))
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data: Vec<f32> = match name {
                        "rope.cos" => main_cos.clone(),
                        "rope.sin" => main_sin.clone(),
                        "hca.rope.cos" => w.get("rope.cos").unwrap().clone(),
                        "hca.rope.sin" => w.get("rope.sin").unwrap().clone(),
                        _ => crate::model_fixture_data(&w, name),
                    };
                    moe_fixture::bind_const(name, meta.aval.shape.clone(), meta.aval.dtype, data)
                }
                other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
            };
            inputs.insert(id, value);
        }
        let residual =
            moe_fixture::eval_graph(&cfg, &g, &inputs).expect("eval hybrid stack residual");
        let residual = moe_fixture::dense(&residual);
        assert_eq!(residual.shape(), vec![1, l, cfg.hc_mult, cfg.hidden]);

        let head = trace_deepseek4_hybrid_stack_head(&cfg, &schedule, l)
            .expect("trace_deepseek4_hybrid_stack_head should trace");
        let mut head_inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &head.inputs {
            let meta = head.meta(id);
            let name = meta.name.as_deref().expect("const without a name");
            let data: Vec<f32> = if name == "resid" {
                residual.as_f32().unwrap().to_vec()
            } else {
                crate::model_fixture_data(&w, name)
            };
            head_inputs.insert(
                id,
                moe_fixture::bind_const(name, meta.aval.shape.clone(), meta.aval.dtype, data),
            );
        }
        let logits = poot_eval::eval(
            &head,
            &head_inputs,
            poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
        )
        .expect("eval hybrid stack head")
        .output
        .into_host()
        .expect("hybrid stack head output is dense");
        assert_eq!(logits.shape(), vec![1, l, cfg.vocab]);

        // Card 422: same dual-purpose indexing as the plain prefill oracle; see that comment.
        #[allow(clippy::needless_range_loop)]
        for qpos in 0..tokens.len() {
            let want = &ref_logits[qpos];
            #[allow(clippy::needless_range_loop)]
            for i in 0..cfg.vocab {
                let gv = logits.as_f32().unwrap()[qpos * cfg.vocab + i];
                let wv = want[i];
                assert!(
                    (gv - wv).abs() <= 1e-4 * wv.abs().max(1e-3) + V4_HYBRID_STACK_ABS_FLOOR,
                    "hybrid stack prefill position {qpos} logit {i}: graph={gv} ref={wv}"
                );
            }
        }
    }

    /// Card 386 bisection tool: compares the widened residual carried between layers by the tracer
    /// (`trace_deepseek4_hybrid_stack_prefill_residual_upto`) and the reference
    /// (`deepseek4_hybrid_stack_decode_ref`'s `layer_residuals`) at every position and layer count, to find the
    /// (layer, position) where a divergence first exceeds tolerance, instead of guessing from a downstream logit.
    ///
    /// Card 386: the 24-cell table showed every cell at or under `V4_HYBRID_STACK_ABS_FLOOR`, two cells (layer 3,
    /// position 4) exceeding the bare `1e-4` relative bar before the floor. Kept `#[ignore]`d and checked against
    /// the same floor as the production oracles: a manual diagnostic, not a CI gate. Run with
    /// `cargo test -p poot-models deepseek4_hybrid_stack_prefill_bisects_by_layer_and_position -- --ignored
    /// --nocapture` (prints per cell).
    #[test]
    #[ignore = "card 386: bisection tool - diagnostic table, see the task doc"]
    fn deepseek4_hybrid_stack_prefill_bisects_by_layer_and_position() {
        let cfg = DeepseekV4Config {
            layers: 3,
            ..tiny_cfg()
        };
        let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
        let l = 8usize;
        let tokens = [1usize, 3, 5, 2, 7, 4, 6, 0];
        let (hc, h) = (cfg.hc_mult, cfg.hidden);

        let mut w = schedule_weights(&cfg, &schedule, cfg.compress_rope_theta);
        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &schedule);
        let (main_cos, main_sin) = deepseek2_rope_tables_interleaved(
            cfg.max_pos,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            None,
        );
        let mut ref_residuals: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();
        let _ = deepseek4_hybrid_stack_decode_ref(
            &cfg,
            &schedule,
            &mut w,
            &moe,
            &main_cos,
            &main_sin,
            &tokens,
            Some(&mut ref_residuals),
            // Hypothesis 4's pre-MoE capture is not wired up: no graph-side counterpart exists to compare against
            // (see the parameter's doc comment on `deepseek4_hybrid_stack_decode_ref`).
            None,
        );

        let mut mask = vec![-1.0e9f32; l * l];
        for i in 0..l {
            let start = i.saturating_sub(cfg.sliding_window - 1);
            for j in start..=i {
                mask[i * l + j] = 0.0;
            }
        }
        let n_windows_hca = l / cfg.hca_compress_rate;
        let hca_win_positions: Vec<f32> = (0..n_windows_hca)
            .map(|w_i| (w_i * cfg.hca_compress_rate) as f32)
            .collect();
        let mut hca_block_bias = vec![0.0f32; l * n_windows_hca];
        for i in 0..l {
            for wi in 0..n_windows_hca {
                if (i as f32) < hca_win_positions[wi] + (cfg.hca_compress_rate - 1) as f32 {
                    hca_block_bias[i * n_windows_hca + wi] = HCA_MASK_NEG;
                }
            }
        }
        let n_windows_csa = l / cfg.csa_compress_rate;
        let csa_win_positions: Vec<f32> = (0..n_windows_csa)
            .map(|w_i| (w_i * cfg.csa_compress_rate) as f32)
            .collect();
        let mut csa_block_bias = vec![0.0f32; l * n_windows_csa];
        for i in 0..l {
            for wi in 0..n_windows_csa {
                if (i as f32) < csa_win_positions[wi] + (cfg.csa_compress_rate - 1) as f32 {
                    csa_block_bias[i * n_windows_csa + wi] = HCA_MASK_NEG;
                }
            }
        }

        let step = moe_fixture::V4StepInputs {
            tokens: &tokens,
            position: 0,
        };

        let mut failures: Vec<String> = Vec::new();
        for upto in 1..=schedule.len() {
            let g = trace_deepseek4_hybrid_stack_prefill_residual_upto(cfg, l, &schedule, upto)
                .unwrap_or_else(|e| panic!("residual prefill upto={upto} should trace: {e}"));
            let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
            for &id in &g.inputs {
                let meta = g.meta(id);
                if let Some(value) = moe.bind_exact(meta, step) {
                    inputs.insert(id, value);
                    continue;
                }
                let value = match &meta.storage {
                    Storage::Slot(Slot::Mask) => {
                        let name = meta.name.as_deref().expect("mask slot without a name");
                        assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                        poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            mask.clone(),
                        ))
                    }
                    Storage::Slot(Slot::Activation) => {
                        let name = meta.name.as_deref().expect("activation without a name");
                        let data: Vec<f32> = match name {
                            "activation.hca.block_bias" => hca_block_bias.clone(),
                            "activation.csa.block_bias" => csa_block_bias.clone(),
                            "activation.hca.window_positions" => hca_win_positions.clone(),
                            "activation.csa.window_positions" => csa_win_positions.clone(),
                            _ => panic!("unexpected activation slot {name} in hybrid prefill"),
                        };
                        poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                            meta.aval.shape.clone(),
                            data,
                        ))
                    }
                    Storage::Const => {
                        let name = meta.name.as_deref().expect("const without a name");
                        let data: Vec<f32> = match name {
                            "rope.cos" => main_cos.clone(),
                            "rope.sin" => main_sin.clone(),
                            "hca.rope.cos" => w.get("rope.cos").unwrap().clone(),
                            "hca.rope.sin" => w.get("rope.sin").unwrap().clone(),
                            _ => crate::model_fixture_data(&w, name),
                        };
                        moe_fixture::bind_const(
                            name,
                            meta.aval.shape.clone(),
                            meta.aval.dtype,
                            data,
                        )
                    }
                    other => {
                        panic!("unexpected storage {other:?} in a stateless prefill graph")
                    }
                };
                inputs.insert(id, value);
            }
            let got = moe_fixture::eval_graph(&cfg, &g, &inputs)
                .unwrap_or_else(|e| panic!("eval residual prefill upto={upto}: {e}"));
            let got = moe_fixture::dense(&got);
            assert_eq!(got.shape(), vec![1, l, hc, h]);

            // Card 422: `qpos`/`stream`/`d` each walk the nested `ref_residuals[qpos][upto - 1]` and compute the
            // flat stride `(qpos * hc + stream) * h + d` into `got.data`. An iterator rewrite needs two levels of
            // `.chunks()` zipped against a nested structure, more error-prone than the oracles above. See Card 422.
            #[allow(clippy::needless_range_loop)]
            for qpos in 0..tokens.len() {
                let want = &ref_residuals[qpos][upto - 1];
                let mut graph_flat = Vec::with_capacity(hc * h);
                let mut ref_flat = Vec::with_capacity(hc * h);
                #[allow(clippy::needless_range_loop)]
                for stream in 0..hc {
                    #[allow(clippy::needless_range_loop)]
                    for d in 0..h {
                        let gv = got.as_f32().unwrap()[(qpos * hc + stream) * h + d];
                        let wv = want[stream][d];
                        graph_flat.push(gv);
                        ref_flat.push(wv);
                        let err = (gv - wv).abs();
                        if err > 1e-4 * wv.abs().max(1e-3) + V4_HYBRID_STACK_ABS_FLOOR {
                            failures.push(format!(
                                "layer {upto} residual pos {qpos} stream {stream} dim {d}: \
graph={gv} ref={wv} abs_err={err:e}"
                            ));
                        }
                    }
                }
                // Panics naming the element on a NaN, which the threshold test above cannot see.
                let max_abs_err = poot_test_util::max_abs_error(&graph_flat, &ref_flat);
                eprintln!("386-bisection layer={upto} pos={qpos} max_abs_err={max_abs_err:e}");
            }
        }
        assert!(
            failures.is_empty(),
            "{} out-of-tolerance cell(s), first few:\n{}",
            failures.len(),
            failures
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Re-prefill at the real `DeepSeek-V4-Flash-0731` rates (`csa_compress_rate=4`, `hca_compress_rate=128`, as
    /// pinned by [`parse_compress_ratios_matches_real_flash_0731_schedule`]). `seq_len=5` is a multiple of
    /// neither rate, the shape `Runner::generate`'s CPU re-prefill loop hits on every step past the first (two
    /// distinct rates > 1 never both divide two consecutive integers; only 1 in `lcm(4,128)=128` lengths passed
    /// the old assert, which panicked with `"seq_len must be an exact multiple of hca_compress_rate ..."`).
    ///
    /// Ground truth is [`deepseek4_hybrid_stack_decode_ref`], an independent step-by-step reference fed only the
    /// 5 real tokens: it pools only complete windows from history (`kv_hist.len() / cr`, see
    /// [`hca_pool_windows_ref`]) and gates every compressed entry by `query_pos >= win*cr + cr - 1` (see
    /// [`csa_selected_window_indices_ref`]), so it has no divisibility assumption and is unaffected by the
    /// tracer's padding. That shows the padded tracer's last-token output is correct, not just finite.
    #[test]
    fn deepseek4_hybrid_stack_prefill_matches_independent_reference_real_rates_non_multiple_seq_len()
     {
        let cfg = DeepseekV4Config {
            layers: 3,
            hca_compress_rate: 128,
            csa_compress_rate: 4,
            max_pos: 256, // must cover the padded length (128)
            ..tiny_cfg()
        };
        let schedule = [V4LayerKind::Sliding, V4LayerKind::Hca, V4LayerKind::Csa];
        let seq_len = 5usize;
        assert!(!seq_len.is_multiple_of(cfg.csa_compress_rate));
        assert!(!seq_len.is_multiple_of(cfg.hca_compress_rate));
        let real_tokens = [1usize, 3, 5, 2, 7];
        assert_eq!(real_tokens.len(), seq_len);

        let mut w = schedule_weights(&cfg, &schedule, cfg.compress_rope_theta);

        let moe = moe_fixture::MoeFixture::for_schedule(&cfg, &schedule);
        let (main_cos, main_sin) = deepseek2_rope_tables_interleaved(
            cfg.max_pos,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            None,
        );
        let ref_logits = deepseek4_hybrid_stack_decode_ref(
            &cfg,
            &schedule,
            &mut w,
            &moe,
            &main_cos,
            &main_sin,
            &real_tokens,
            None,
            None,
        );
        let want = &ref_logits[seq_len - 1];

        // The call that used to panic.
        let g = trace_deepseek4_hybrid_stack_prefill(cfg, seq_len, &schedule)
            .expect("trace_deepseek4_hybrid_stack_prefill should trace");

        let l = deepseek4_prefill_pad_len(seq_len, &cfg, true, true);
        assert_eq!(
            l, 128,
            "lcm(4, 128) = 128, the padded length this tracer must use"
        );
        let mut tokens = real_tokens.to_vec();
        tokens.resize(l, 0); // pad token id, same convention Runner::bind uses

        let mut mask = vec![-1.0e9f32; l * l];
        for i in 0..l {
            let start = i.saturating_sub(cfg.sliding_window - 1);
            for j in start..=i {
                mask[i * l + j] = 0.0;
            }
        }
        let n_windows_hca = l / cfg.hca_compress_rate;
        let hca_win_positions: Vec<f32> = (0..n_windows_hca)
            .map(|w_i| (w_i * cfg.hca_compress_rate) as f32)
            .collect();
        let mut hca_block_bias = vec![0.0f32; l * n_windows_hca];
        for i in 0..l {
            for wi in 0..n_windows_hca {
                if (i as f32) < hca_win_positions[wi] + (cfg.hca_compress_rate - 1) as f32 {
                    hca_block_bias[i * n_windows_hca + wi] = HCA_MASK_NEG;
                }
            }
        }
        let n_windows_csa = l / cfg.csa_compress_rate;
        let csa_win_positions: Vec<f32> = (0..n_windows_csa)
            .map(|w_i| (w_i * cfg.csa_compress_rate) as f32)
            .collect();
        let mut csa_block_bias = vec![0.0f32; l * n_windows_csa];
        for i in 0..l {
            for wi in 0..n_windows_csa {
                if (i as f32) < csa_win_positions[wi] + (cfg.csa_compress_rate - 1) as f32 {
                    csa_block_bias[i * n_windows_csa + wi] = HCA_MASK_NEG;
                }
            }
        }

        let step = moe_fixture::V4StepInputs {
            tokens: &tokens,
            position: 0,
        };
        let mut inputs: HashMap<poot_graph_ir::ValueId, poot_eval::Value> = HashMap::new();
        for &id in &g.inputs {
            let meta = g.meta(id);
            if let Some(value) = moe.bind_exact(meta, step) {
                inputs.insert(id, value);
                continue;
            }
            let value = match &meta.storage {
                Storage::Slot(Slot::Mask) => {
                    let name = meta.name.as_deref().expect("mask slot without a name");
                    assert_eq!(name, "mask.prefill", "unexpected mask slot {name}");
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        mask.clone(),
                    ))
                }
                Storage::Slot(Slot::Activation) => {
                    let name = meta.name.as_deref().expect("activation without a name");
                    let data: Vec<f32> = match name {
                        "activation.hca.block_bias" => hca_block_bias.clone(),
                        "activation.csa.block_bias" => csa_block_bias.clone(),
                        "activation.hca.window_positions" => hca_win_positions.clone(),
                        "activation.csa.window_positions" => csa_win_positions.clone(),
                        _ => panic!("unexpected activation slot {name} in hybrid prefill"),
                    };
                    poot_eval::Value::Host(poot_tensor::HostTensor::f32(
                        meta.aval.shape.clone(),
                        data,
                    ))
                }
                Storage::Const => {
                    let name = meta.name.as_deref().expect("const without a name");
                    let data: Vec<f32> = match name {
                        "rope.cos" => main_cos.clone(),
                        "rope.sin" => main_sin.clone(),
                        "hca.rope.cos" => w.get("rope.cos").unwrap().clone(),
                        "hca.rope.sin" => w.get("rope.sin").unwrap().clone(),
                        _ => crate::model_fixture_data(&w, name),
                    };
                    moe_fixture::bind_const(name, meta.aval.shape.clone(), meta.aval.dtype, data)
                }
                other => panic!("unexpected storage {other:?} in a stateless prefill graph"),
            };
            inputs.insert(id, value);
        }
        let logits = moe_fixture::eval_graph(&cfg, &g, &inputs)
            .expect("eval hybrid stack prefill at real rates");
        let logits = moe_fixture::dense(&logits);
        let got = &logits.as_f32().unwrap();
        assert_eq!(logits.shape(), vec![1, 1, cfg.vocab]);
        for i in 0..cfg.vocab {
            let (gv, wv) = (got[i], want[i]);
            assert!(
                (gv - wv).abs() <= 1e-3 * wv.abs().max(1e-3),
                "hybrid stack prefill (real rates, padded) logit {i}: graph={gv} ref={wv}"
            );
        }
    }
}

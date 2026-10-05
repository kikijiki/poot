//! `optimize` oracle checks, dispatch-count regressions, and flash-attention pass fusion equivalence
//! over real qwen2-shaped graphs, moved from `poot-eval/src/tests/flash_optimize.rs` with the passes
//! themselves (card 626) - poot-eval must never depend on poot-graph-plan (its own architecture
//! test); this crate already dev-depends on poot-eval, so an integration test here can drive both.
//! (`masked_decode_matches_sliced`/`flash_prefill_tracer_matches_decomposed_prefill`/
//! `flash_decode_op_matches_decomposed_masked` stayed behind in poot-eval: they exercise the
//! `FlashAttentionDecode`/`FlashAttentionPrefill` ops directly through dedicated model tracers, never
//! a poot-graph-plan pass.)

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_executor_parity::weight_map::MappedModel;
use poot_graph_ir::{Graph, OpKind, Slot, Storage};
use poot_models::model::{LogitRows, Phase};
use poot_tensor::{DType, HostTensor};
use std::collections::HashMap as Map;

/// ADR-0101 tier 1 (Card 556): the oracle evaluates a composite from its decomposition, so a pass's
/// output graph evaluates to exactly the input graph's bits, on every element. A non-finite element
/// fails even where both sides agree.
fn assert_bits_equal(got: &HostTensor, want: &HostTensor, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (got, want) = (got.as_f32().unwrap(), want.as_f32().unwrap());
    for (index, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.is_finite() && w.is_finite(),
            "{what}: element {index} is not finite: {g} vs {w}"
        );
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: element {index}: optimized {g} vs raw {w}"
        );
    }
}

/// How many equations of `g` satisfy `is_op`; asserted above zero so a bit-exact row is not vacuous.
fn count(g: &Graph, is_op: impl Fn(&OpKind) -> bool) -> usize {
    g.eqns.iter().filter(|eqn| is_op(&eqn.op)).count()
}

/// Deterministic pseudo-random fill in [-1, 1), no rng dependency (poot-eval's own test helper of the
/// same name, duplicated: this is an external integration test, so it cannot reach poot-eval's
/// `pub(super)` test helpers).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

use poot_graph_plan::passes_without_target as optimize;

/// A weight bound by name: a name-seeded fill scaled by 0.1.
fn weight(name: &str, shape: &[usize]) -> HostTensor {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    HostTensor::f32(
        shape.to_vec(),
        fill(shape.iter().product::<usize>().max(1), seed)
            .iter()
            .map(|v| v * 0.1)
            .collect(),
    )
}

/// The tiny qwen2 (GQA 4 over 2, head dim 4) most rows trace, through the registry.
fn tiny_qwen2(layers: usize, max_positions: usize) -> MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, layers)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(max_positions)
        .f32_model()
}

/// `m`'s step graph: `rows` sequences of `tokens` new tokens over `cap` positions, logits at the last
/// token of each. A whole-prompt prefill has `cap == tokens` (the only shape whose attention mask is
/// square, which the flash matcher requires).
fn trace(m: &MappedModel, phase: Phase, rows: usize, tokens: usize, cap: usize) -> Graph {
    plain(
        m.model
            .trace(phase, step(rows, tokens, cap, LogitRows::Last))
            .unwrap(),
    )
}

/// Every input of `g` bound: the token slot cycling through `tokens`, the position slot counting up
/// from `start` along the token axis, weights by [`weight`], carried state from `state(k, shape)`.
fn bind(
    g: &Graph,
    tokens: &[i32],
    start: i32,
    state: &dyn Fn(usize, &[usize]) -> HostTensor,
) -> Map<usize, Value> {
    let mut inputs: Map<usize, Value> = Map::new();
    for &id in &g.inputs {
        let m = &g.values[id];
        let shape = m.aval.shape.clone();
        let numel = shape.iter().product::<usize>().max(1);
        let t = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(
                shape,
                (0..numel).map(|i| tokens[i % tokens.len()]).collect(),
            ),
            Storage::Slot(Slot::Pos) => {
                let axis = *shape.last().unwrap();
                HostTensor::i32(
                    shape,
                    (0..numel).map(|i| start + (i % axis) as i32).collect(),
                )
            }
            Storage::Slot(other) => panic!("unexpected slot {other:?}"),
            Storage::Const => weight(m.name.as_deref().unwrap(), &shape),
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()),
            Storage::State => continue,
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, Value::from(t));
    }
    for (k, &(si, _)) in g.state.iter().enumerate() {
        let shape = g.aval(si).shape.clone();
        inputs.insert(si, Value::from(state(k, &shape)));
    }
    inputs
}

fn zeros(_: usize, shape: &[usize]) -> HostTensor {
    HostTensor::zeros(shape.to_vec())
}

/// `g` evaluated: the output and every carried state, dense.
fn run(g: &Graph, inputs: &Map<usize, Value>) -> (HostTensor, Vec<HostTensor>) {
    let r = eval(g, inputs, EvalOptions::new(EvalBudget::UNBOUNDED)).expect("the graph evaluates");
    let state = r
        .state
        .into_iter()
        .map(|v| v.into_host().expect("flash_optimize state is dense"))
        .collect();
    (
        r.output
            .into_host()
            .expect("flash_optimize tests evaluate dense graphs"),
        state,
    )
}

#[test]
fn optimize_preserves_results_over_random_configs() {
    // card 035: the pass pipeline (cse -> flash_attention -> dce -> fuse) must preserve results
    // across the model config space (GQA ratios, qk-norm, qkv-bias, layer counts, head
    // dims): eval(raw) == eval(optimize) for a carried decode step and a prefill forward. Pure CPU.
    // Card 556 SC-001: bit for bit on the output and every carried state (ADR-0101 tier 1), with the
    // flash, rope and fused composites formed (counted above zero), since the oracle evaluates each
    // composite from its decomposition. The bias / qk-norm axes pick the family: qwen2 (bias), qwen3
    // (qk-norm, no bias) or llama (neither); a qwen2 with qk-norm is no shipped family.
    let mut s: u64 = 0x4F50_5431_4D5A_4521;
    let mut rng = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    for _ in 0..24 {
        let head_dim = [2usize, 4, 8][(rng() % 3) as usize];
        let n_kv_heads = 1 + (rng() % 2) as usize; // 1..=2
        let n_rep = 1 + (rng() % 3) as usize; // 1..=3
        let n_heads = n_kv_heads * n_rep;
        let layers = 1 + (rng() % 3) as usize; // 1..=3
        let qkv_bias = rng() % 2 == 0;
        let qk_norm = rng() % 2 == 0;
        let family = match (qkv_bias, qk_norm) {
            (true, _) => Family::Qwen2,
            (false, true) => Family::Qwen3,
            (false, false) => Family::Llama,
        };
        let vocab = 32usize;
        let m = Dense::new(family)
            .vocab(vocab)
            .dims(n_heads * head_dim, n_heads * head_dim * 2, layers)
            .heads(n_heads, n_kv_heads)
            .head_dim(head_dim)
            .max_positions(32)
            .f32_model();
        let (cap, l) = (6usize, 5usize);
        let tokens: Vec<i32> = (0..l as i32).map(|i| (i * 3 + 1) % vocab as i32).collect();

        // --- decode: a single step, random weights + caches, raw vs optimize on identical inputs ---
        let gd = trace(&m, Phase::Decode, 1, 1, cap);
        let ogd = optimize(&gd);
        assert!(count(&ogd, |op| matches!(op, OpKind::FlashAttentionDecode { .. })) > 0);
        assert!(count(&ogd, |op| matches!(op, OpKind::Rope { .. })) > 0);
        assert!(
            count(&ogd, |op| matches!(
                op,
                OpKind::Fused(_) | OpKind::FusedRow(_)
            )) > 0
        );
        // caches seeded by state index (same for both graphs).
        let caches = |k: usize, sh: &[usize]| {
            HostTensor::f32(
                sh.to_vec(),
                fill(sh.iter().product::<usize>().max(1), 7000 + k as u64),
            )
        };
        let (raw_d, raw_state) = run(&gd, &bind(&gd, &[5], 3, &caches));
        let (opt_d, opt_state) = run(&ogd, &bind(&ogd, &[5], 3, &caches));
        assert_bits_equal(&opt_d, &raw_d, "decode logits");
        for (index, (opt, raw)) in opt_state.iter().zip(&raw_state).enumerate() {
            assert_bits_equal(opt, raw, &format!("decode state {index}"));
        }

        // --- prefill: a whole forward from empty caches, raw vs optimize on identical inputs ---
        let gp = trace(&m, Phase::Prefill, 1, l, l);
        let ogp = optimize(&gp);
        assert!(
            count(&ogp, |op| matches!(
                op,
                OpKind::FlashAttentionPrefill { .. }
            )) > 0
        );
        let (raw_p, _) = run(&gp, &bind(&gp, &tokens, 0, &zeros));
        let (opt_p, _) = run(&ogp, &bind(&ogp, &tokens, 0, &zeros));
        assert_bits_equal(&opt_p, &raw_p, "prefill logits");
    }
}

/// Card 556 SC-001 (gemma2): the traced Gemma2 prefill, whose attention carries the
/// `attn_logit_softcap`, fuses to `FlashAttentionPrefill { softcap: Some(_) }` in every layer, and
/// the optimized graph's logits and carried caches equal the traced graph's bit for bit.
#[test]
fn optimize_is_bit_exact_on_traced_gemma2_softcap_prefill() {
    let layers = 4;
    let m = Dense::new(Family::Gemma2)
        .vocab(64)
        .dims(64, 128, layers)
        .heads(4, 2)
        .head_dim(16)
        .max_positions(32)
        .with("sliding_window", 4)
        .with("attn_logit_softcapping", 50.0)
        .with("final_logit_softcapping", 30.0)
        .with("query_pre_attn_scalar", serde_json::Value::Null)
        .f32_model();
    let n = 6usize;
    let g = trace(&m, Phase::Prefill, 1, n, n);
    let og = optimize(&g);
    let softcapped = count(&og, |op| {
        matches!(
            op,
            OpKind::FlashAttentionPrefill {
                softcap: Some(_),
                ..
            }
        )
    });
    assert_eq!(
        softcapped, layers,
        "one softcapped FlashAttentionPrefill per layer"
    );

    let tokens: Vec<i32> = (0..n as i32).map(|i| (i * 7 + 3) % 64).collect();
    let (raw, raw_state) = run(&g, &bind(&g, &tokens, 0, &zeros));
    let (opt, opt_state) = run(&og, &bind(&og, &tokens, 0, &zeros));
    assert_bits_equal(&opt, &raw, "gemma2 prefill logits");
    for (index, (opt, raw)) in opt_state.iter().zip(&raw_state).enumerate() {
        assert_bits_equal(opt, raw, &format!("gemma2 prefill state {index}"));
    }
}

/// The real Qwen2.5-0.5B (hidden 896, inter 4864, head_dim 64, GQA 14/2, 24 layers) and Qwen3-0.6B
/// (hidden 1024, inter 3072, head_dim 128, GQA 16/8, qk_norm, 28 layers) dimensions, `vocab` shrunk
/// to 64 since it only affects embed/lm_head width, not fused regions, and 151936 would allocate
/// ~550 MB per weight.
fn real_configs() -> [(&'static str, Dense); 2] {
    [
        (
            "qwen2_0_5b",
            Dense::new(Family::Qwen2)
                .vocab(64)
                .dims(896, 4864, 24)
                .heads(14, 2)
                .head_dim(64)
                .max_positions(32768),
        ),
        (
            "qwen3_0_6b",
            Dense::new(Family::Qwen3)
                .vocab(64)
                .dims(1024, 3072, 28)
                .heads(16, 8)
                .head_dim(128)
                .max_positions(40960),
        ),
    ]
}

#[test]
fn optimize_matches_oracle_on_real_qwen2_decode_configs_real_dims() {
    // `optimize_preserves_results_over_random_configs` only exercises toy shapes (head_dim <= 8,
    // hidden <= 24), so a mis-fusion that appears at real width (row-region axis at hidden=896, GQA
    // repeat at n_rep=7) would pass. Run the full pass pipeline (canonicalize -> cse ->
    // fold_iota -> rope_fusion -> flash_attention_capped -> dce -> fuse_bias_epilogues -> fuse; Card
    // 557 moved the tiling and region-flash choices into the planner) against the CPU oracle on the
    // real configs of `real_configs`; eval(raw) must equal eval(optimize) bit for bit (Card 556:
    // the oracle evaluates the flash op from its decomposition, so nothing reassociates).
    for (label, dense) in real_configs() {
        let m = dense.zeroed_model(DType::F32);
        let cap = 16usize;
        let g = trace(&m, Phase::Decode, 1, 1, cap);
        let og = optimize(&g);
        assert!(
            og.eqns.len() < g.eqns.len(),
            "{label}: optimize should shrink the real-config decode graph ({} -> {} eqns)",
            g.eqns.len(),
            og.eqns.len()
        );

        let caches = |k: usize, sh: &[usize]| {
            HostTensor::f32(
                sh.to_vec(),
                fill(sh.iter().product::<usize>().max(1), 9000 + k as u64),
            )
        };
        let (raw, _) = run(&g, &bind(&g, &[3], 4, &caches));
        let (opt, _) = run(&og, &bind(&og, &[3], 4, &caches));
        assert_eq!(
            raw.shape(),
            opt.shape(),
            "{label}: shape mismatch raw vs optimize"
        );
        assert_bits_equal(&opt, &raw, label);
    }
}

#[test]
fn dispatch_count_regression_on_real_qwen2_decode_configs() {
    // The real-model dispatch test (`fusion_dispatch_reduction_on_real_qwen2_config`,
    // poot-gpu/tests/qwen2.rs) runs only `cse+fuse` and asserts a direction (after < before), so a
    // regression that stays below the un-fused baseline would pass. This pins `dispatch_count`
    // (static IR metric: every eqn except a pure-aliasing Reshape is one dispatch) through the full
    // `optimize()` pipeline on the real Qwen2.5-0.5B and Qwen3-0.6B decode graphs at KV capacity 64
    // (matching that poot-gpu test). The pins are exact, not upper bounds: a bound above the real
    // count still passes with rope fusion deleted, so a change in either direction fails here and the
    // pin moves with the change.
    use poot_graph_ir::analysis::dispatch_count;

    let cap = 64usize;
    let [(label2, qwen2), (label3, qwen3)] = real_configs();
    for (label, dense, pin) in [(label2, qwen2, PIN_QWEN2), (label3, qwen3, PIN_QWEN3)] {
        let g = trace(&dense.zeroed_model(DType::F32), Phase::Decode, 1, 1, cap);
        assert_eq!(
            dispatch_count(&optimize(&g)),
            pin,
            "{label} real-config optimize() dispatch_count moved off its pin"
        );
    }
}

/// `optimize()` dispatch counts of the real-config decode graphs at capacity 64. The dense decode
/// body plans more dispatches per layer than it should: POOT-1022 brings these pins down.
const PIN_QWEN2: usize = 775;
const PIN_QWEN3: usize = 791;

#[test]
fn flash_default_cuts_prefill_dispatch_count() {
    // card 030/038: flash also collapses each prefill layer's attention chain to one
    // FlashAttentionPrefill op, so the prefill (TTFT) dispatch count drops. Validates the
    // Runner::prefill_graph_cost path at a representative seq_len.
    use poot_graph_ir::analysis::dispatch_count;
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{cse, flash_attention_capped, fuse};
    let layers = 4;
    let g = trace(&tiny_qwen2(layers, 64), Phase::Prefill, 1, 16, 16);
    let n_flash = flash_attention_capped(&cse(&g), None)
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionPrefill { .. }))
        .count();
    assert_eq!(n_flash, layers, "one FlashAttentionPrefill per layer");
    let no_flash = dispatch_count(&fuse(&cse(&g)));
    let with_flash = dispatch_count(&optimize(&g));
    assert!(
        with_flash < no_flash,
        "flash reduces prefill dispatches: {no_flash} -> {with_flash}"
    );
}

#[test]
fn flash_default_cuts_decode_dispatch_count() {
    // card 038: flash collapses each layer's ~10-op softmax-attention chain (transpose, QKᵀ, scale,
    // mask-add, max/sub/exp/sum/div, PV) into one FlashAttentionDecode dispatch. Compare
    // dispatch_count for the optimized decode graph without flash (fuse(cse)) and with
    // (optimize = cse->flash->dce->fuse) and assert a per-layer reduction. No GPU.
    use poot_graph_ir::analysis::dispatch_count;
    use poot_graph_plan::{cse, fuse};
    let layers = 4;
    let cap = 8;
    let g = trace(&tiny_qwen2(layers, 16), Phase::Decode, 1, 1, cap);
    let no_flash = dispatch_count(&fuse(&cse(&g)));
    let with_flash = dispatch_count(&optimize(&g));
    let saved = no_flash - with_flash;
    eprintln!(
        "decode dispatch count: no-flash={no_flash}, flash={with_flash} (saved {saved} over {layers} layers, {} per layer)",
        saved / layers
    );
    assert!(with_flash < no_flash, "flash must reduce dispatches");
    // each layer's attention collapses to one op, saving several dispatches per layer.
    assert!(
        saved >= 4 * layers,
        "expected several dispatches saved per layer, saved {saved}"
    );
}

#[test]
fn flash_attention_shrinks_prefill_peak_memory_quadratically() {
    // card 038: the materialized attention_prefill holds the [1,Hq,L,L] scores (O(L^2) transient
    // bytes); the flash_attention pass removes them (O(L*D)). Measure peak_transient_bytes for both
    // at growing L: materialized grows ~quadratically, flash ~linearly, so the win ratio widens with
    // context length. No GPU.
    use poot_graph_ir::analysis::peak_transient_bytes;
    use poot_graph_plan::{cse, dce, flash_attention_capped};
    let m = tiny_qwen2(2, 256);
    let measure = |l: usize| -> (usize, usize) {
        let g = trace(&m, Phase::Prefill, 1, l, l);
        let mat = peak_transient_bytes(&g);
        let flash = peak_transient_bytes(&dce(&flash_attention_capped(&cse(&g), None)));
        (mat, flash)
    };
    let (m16, f16) = measure(16);
    let (m64, f64) = measure(64);
    eprintln!(
        "prefill peak transient bytes: L=16 mat={m16} flash={f16} ({}x); L=64 mat={m64} flash={f64} ({}x)",
        m16 / f16.max(1),
        m64 / f64.max(1)
    );
    // flash always wins (removes the scores).
    assert!(f16 < m16 && f64 < m64, "flash peak < materialized peak");
    // materialized grows super-linearly with L (the L^2 scores dominate): 4x L -> well over 4x bytes.
    assert!(
        m64 >= 8 * m16,
        "materialized peak should grow super-linearly: {m16} -> {m64}"
    );
    // flash still grows far more slowly than materialized (4x L -> at most ~6x bytes), but no longer
    // purely linearly: card 550 moved the causal mask from a host-bound input (excluded from this
    // transient-bytes ledger) to a graph computation (`causal_mask_from_pos`), whose own `[1,1,L,L]`
    // visibility/bias chain is inherently O(L^2) - the same bound the mask always had, now visible to
    // this measurement instead of hidden in an external buffer. This narrow `cse -> flash_attention ->
    // dce` pipeline (unlike the full `optimize()` tested elsewhere) has no elementwise-fusion pass to
    // collapse that chain's several L*L intermediates into one.
    assert!(
        f64 <= 6 * f16,
        "flash peak should grow far more slowly than materialized's L^2: {f16} -> {f64}"
    );
    // the win ratio widens with context length.
    assert!(
        m64 / f64.max(1) > m16 / f16.max(1),
        "flash win should grow with L"
    );
}

#[test]
fn flash_attention_pass_fuses_prefill_attention() {
    // card 038: the flash-attention pass pattern-matches the materialized
    // softmax-attention subgraph in a raw prefill trace and rewrites it to FlashAttentionPrefill, once
    // per layer, without changing the logits. cse -> flash_attention -> dce; tiny config, synthetic weights.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{cse, dce, flash_attention_capped};
    let layers = 2;
    let tokens: Vec<i32> = vec![5, 9, 2, 14, 7, 1];
    let l = tokens.len();
    let g = trace(&tiny_qwen2(layers, 16), Phase::Prefill, 1, l, l);
    let fused = dce(&flash_attention_capped(&cse(&g), None));
    let n_flash = fused
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionPrefill { .. }))
        .count();
    assert_eq!(n_flash, layers, "one FlashAttentionPrefill per layer");
    let (base, _) = run(&g, &bind(&g, &tokens, 0, &zeros));
    let (after, _) = run(&fused, &bind(&fused, &tokens, 0, &zeros));
    assert_eq!(base.shape(), after.shape());
    assert_bits_equal(&after, &base, "prefill logits");
}

#[test]
fn flash_attention_pass_fuses_batched_decode() {
    // card 038: a batched decode trace (q [B,Hq,1,D], B>1) flashes too: FlashAttentionDecode carries the
    // batch (one workgroup per (batch,head)). One op per layer.
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{cse, flash_attention_capped};
    let layers = 2;
    let g = flash_attention_capped(
        &cse(&trace(&tiny_qwen2(layers, 16), Phase::Decode, 3, 1, 5)),
        None,
    );
    let n_dec = g
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionDecode { .. }))
        .count();
    assert_eq!(n_dec, layers, "batched decode flashes once per layer");
}

#[test]
fn flash_attention_pass_fuses_decode_attention() {
    // card 038: the same pass fires on a decode trace (M=1 -> FlashAttentionDecode), once per layer,
    // and the rewritten graph evals identically across a carried decode. The decode/prefill split is by
    // q's M dim (1 vs L).
    use poot_graph_ir::op::OpKind;
    use poot_graph_plan::{cse, dce, flash_attention_capped};
    let layers = 2;
    let tokens: Vec<i32> = vec![5, 9, 2, 14, 7];
    let cap = tokens.len();
    let g = trace(&tiny_qwen2(layers, 16), Phase::Decode, 1, 1, cap);
    let fused = dce(&flash_attention_capped(&cse(&g), None));
    let n_flash = fused
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FlashAttentionDecode { .. }))
        .count();
    assert_eq!(n_flash, layers, "one FlashAttentionDecode per layer");

    let mut base_caches: Vec<HostTensor> = g
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(g.aval(si).shape.clone()))
        .collect();
    let mut fused_caches: Vec<HostTensor> = fused
        .state
        .iter()
        .map(|&(si, _)| HostTensor::zeros(fused.aval(si).shape.clone()))
        .collect();
    for (pos, &token) in tokens.iter().enumerate() {
        let token = [token];
        let (bl, bn) = run(
            &g,
            &bind(&g, &token, pos as i32, &|k, _| base_caches[k].clone()),
        );
        base_caches = bn;
        let (fl, fnew) = run(
            &fused,
            &bind(&fused, &token, pos as i32, &|k, _| fused_caches[k].clone()),
        );
        fused_caches = fnew;
        assert_bits_equal(&fl, &bl, &format!("decode logits at position {pos}"));
    }
}

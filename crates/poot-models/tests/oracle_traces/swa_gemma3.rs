//! Gemma 3's per-layer global/local mask selection and Gemma 2's softcap config wiring, traced through the
//! registry's dense family and evaluated on the CPU oracle. (Moved from `poot-eval/src/tests/swa_gemma3.rs`;
//! the windowed-mask arithmetic rows stay with the oracle.)

use poot_eval::{EvalOptions, Value};
use poot_graph_ir::ValueId;
use poot_tensor::HostTensor;
use poot_test_util::{fill, max_abs_error};
use std::collections::HashMap;

/// One step of a `Model::trace` graph on the CPU oracle: `Slot::Token` and `Slot::Pos` bound from `tokens` at
/// absolute position `pos0`, every const filled by `weight(name, shape)`, every carried state from `state` (zeros
/// when empty). Returns the output and the state outputs.
pub(super) fn run_step(
    g: &poot_graph_ir::Graph,
    tokens: &[i32],
    pos0: usize,
    weight: &dyn Fn(&str, &[usize]) -> Vec<f32>,
    state: &[HostTensor],
) -> (HostTensor, Vec<HostTensor>) {
    use poot_graph_ir::{Slot, Storage};

    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in g.inputs() {
        let m = g.meta(id);
        let shape = m.aval.shape.clone();
        let t = match m.storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(shape, tokens.to_vec()),
            Storage::Slot(Slot::Pos) => HostTensor::i32(
                shape,
                (pos0..pos0 + tokens.len()).map(|p| p as i32).collect(),
            ),
            Storage::Slot(other) => panic!("unexpected slot {other:?} in a dense step graph"),
            Storage::Const => {
                HostTensor::f32(shape.clone(), weight(m.name.as_deref().unwrap(), &shape))
            }
            Storage::Computed(c) => c.tensor(),
            Storage::State => continue,
            Storage::Device => panic!("a graph input is never a device intermediate"),
        };
        inputs.insert(id, Value::from(t));
    }
    for (i, (input, _)) in g.state().enumerate() {
        let t = state
            .get(i)
            .cloned()
            .unwrap_or_else(|| HostTensor::zeros(g.aval(input).shape.clone()));
        inputs.insert(input, Value::from(t));
    }
    let evaluation = poot_eval::eval(g, &inputs, EvalOptions::default()).unwrap();
    let state = evaluation
        .state
        .into_iter()
        .map(|v| v.into_host().unwrap())
        .collect();
    (evaluation.output.into_host().unwrap(), state)
}

/// Deterministic name-seeded weights of scale 0.1, the fill the rest of this file uses.
pub(super) fn named_weights(name: &str, shape: &[usize]) -> Vec<f32> {
    let seed: u64 = name.bytes().fold(1469598103934665603u64, |h, c| {
        (h ^ c as u64).wrapping_mul(1099511628211)
    });
    fill(shape.iter().product::<usize>().max(1), seed)
        .iter()
        .map(|v| v * 0.1)
        .collect()
}

/// The tiny 4-layer Gemma 3 of the per-layer mask rows: GQA (`n_rep = 2`, so the SWA rows reach the grouped
/// path), a query scale of 0.5, a local window of `window` (`None`: no window) and every `every`-th layer global.
fn gemma3(
    every: usize,
    window: Option<usize>,
) -> poot_models::test_support::weight_map::MappedModel {
    use poot_models::test_support::dense::{Dense, Family};

    Dense::new(Family::Gemma3)
        .vocab(16)
        .dims(8, 16, 4)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .with("rms_norm_eps", 1e-5)
        .with("query_pre_attn_scalar", 4.0)
        .with("sliding_window_pattern", every)
        .with("sliding_window", window)
        .f32_model()
}

fn gemma3_prefill_logits(
    every: usize,
    window: Option<usize>,
    (n, cap): (usize, usize),
) -> Vec<f32> {
    use poot_models::model::{LogitRows, StepSpec};
    use poot_models::test_support::dense::step;

    let m = gemma3(every, window);
    let g = m
        .model
        .trace(
            &StepSpec::step(step(1, n, cap, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    // Distinct token ids per position: V has no RoPE, so identical tokens would make every V row identical and
    // the windowed mask a no-op (softmax over identical values returns that value).
    let tokens: Vec<i32> = (0..n as i32).collect();
    run_step(&g, &tokens, 0, &named_weights, &[])
        .0
        .as_f32()
        .unwrap()
        .to_vec()
}

/// Card 151 Part 2 (updated for card 550 and POOT-1015): gemma3's per-layer global/local mask split through
/// `Model::trace` of the registry's Gemma 3. Tests the family's selection, not the windowed-mask arithmetic
/// (`swa_windowed_mask_eval_oracle_correctness` and poot-llm's `mask_tests`): a global layer's mask must always
/// be the unwindowed causal mask, a local layer's always the windowed one, never swapped or shared. Both masks are
/// in-graph computations over `Slot::Pos`, so this drives the selection by varying `sliding_window` (the only knob
/// that can change the local mask's content).
///
/// Tiny synthetic 4-layer config with `sliding_window_pattern = 2`: layers 1,3 global, 0,2 local. Three checks:
/// 1. All-global (`sliding_window_pattern = 1`): changing the window must not change the output (no layer is
///    local, so the windowed mask is never selected).
/// 2. Mixed, W < seq_len: narrowing the window from none (full-causal) to `Some(3)` (seq_len=6) must change the
///    output.
/// 3. Mixed, W >= seq_len: `Some(seq_len)` and `Some(seq_len+10)` must both be byte-identical to no window.
#[test]
fn gemma3_prefill_per_layer_mask_selection() {
    let l = 6usize; // seq_len
    let run = |every, window| gemma3_prefill_logits(every, window, (l, l));

    // --- Check 1: all-global, the window must not affect the output. ---
    assert_eq!(
        run(1, None),
        run(1, Some(3)),
        "all-global: the window must not affect output"
    );

    // --- Checks 2+3: mixed (pattern 2 -> layers 1,3 global, 0,2 local). ---
    let out_win = run(2, Some(3));
    let out_full = run(2, None);
    let max_diff = max_abs_error(&out_win, &out_full);
    assert!(
        max_diff > 1e-5,
        "mixed layers: W=3 window on local layers must change the output vs full-causal (max_diff={max_diff:.2e})"
    );
    assert_eq!(
        out_full,
        run(2, Some(l)),
        "mixed layers: window==seq_len must be byte-identical to full-causal"
    );
    assert_eq!(
        out_full,
        run(2, Some(l + 10)),
        "mixed layers: window>seq_len must be byte-identical to full-causal"
    );
}

/// Companion to [`gemma3_prefill_per_layer_mask_selection`] over a cache larger than the prompt (prefill that also
/// fills the K/V caches the decode continues from): same mixed (2 global + 2 local) config. Checks only real
/// windowing (check 3 above).
#[test]
fn gemma3_prefill_kv_per_layer_mask_selection() {
    let (n, cap) = (6usize, 8usize); // prompt length n, cache capacity cap >= n
    let run = |window| gemma3_prefill_logits(2, window, (n, cap));

    let out_win = run(Some(3));
    let out_full = run(None);
    let max_diff = max_abs_error(&out_win, &out_full);
    assert!(
        max_diff > 1e-5,
        "gemma prefill-kv: W=3 window on local layers must change output vs full-causal (max_diff={max_diff:.2e})"
    );

    // W >= n is a no-op end to end.
    assert_eq!(
        out_full,
        run(Some(n)),
        "gemma prefill-kv: window==seq_len must be byte-identical to full-causal"
    );
}

/// Card 151 Part 2 (decode half, updated for card 550 and POOT-1015): the Gemma 3 decode step's per-layer window
/// through `Model::trace`. Both the global and local-layer masks are in-graph computations over `Slot::Pos`
/// (`[1,1]`) and `iota(cap)`, so there is no host-built mask to swap or to pin the window arithmetic against
/// independently - that arithmetic is `swa_windowed_mask_eval_oracle_correctness`'s job. Three checks at a position
/// past the window (pos=5, w=2, so key slots t<=3 are out of window on local layers):
/// 1. Window active, mixed (pattern 2): `sliding_window = Some(2)` must differ from none.
/// 2. Window inert (pos < w): at pos=1 with w=2 nothing is windowed out, so `Some(2)` == none byte-identical.
/// 3. All-global (pattern 1): the local window must reach no layer, so `Some(2)` == none byte-identical even at
///    pos=5.
#[test]
fn gemma3_decode_per_layer_mask_selection() {
    use poot_models::model::{LogitRows, StepSpec};
    use poot_models::test_support::dense::step;

    let cap = 8usize;
    // Deterministic non-zero cache contents (per state index), so different windows give different outputs.
    let logits = |every: usize, window: Option<usize>, pos: usize| -> Vec<f32> {
        let m = gemma3(every, window);
        let g = m
            .model
            .trace(
                &StepSpec::step(step(1, 1, cap, LogitRows::Last)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let state: Vec<HostTensor> = g
            .state()
            .enumerate()
            .map(|(i, (input, _))| {
                let shape = g.aval(input).shape.clone();
                HostTensor::f32(shape.clone(), named_weights(&format!("kv.{i}"), &shape))
            })
            .collect();
        run_step(&g, &[1], pos, &named_weights, &state)
            .0
            .as_f32()
            .unwrap()
            .to_vec()
    };

    // --- Check 1: window active, mixed (layers 1,3 global; 0,2 local). pos=5 > w=2. ---
    let out_win = logits(2, Some(2), 5);
    let out_none = logits(2, None, 5);
    let max_diff = max_abs_error(&out_win, &out_none);
    assert!(
        max_diff > 1e-5,
        "mixed decode: local window w=2 at pos=5 must change output vs full-causal (max_diff={max_diff:.2e})"
    );

    // --- Check 2: window inert at pos=1 (< w=2): nothing windowed out, Some(2) == none. ---
    assert_eq!(
        logits(2, Some(2), 1),
        logits(2, None, 1),
        "decode: at pos < window the window is inert, must be byte-identical to full-causal"
    );

    // --- Check 3: all-global: the local window reaches no layer even at pos=5. ---
    assert_eq!(
        logits(1, Some(2), 5),
        logits(1, None, 5),
        "all-global decode: the window must not affect any layer's output"
    );
}

/// Config wiring: `attn_logit_softcapping` and `final_logit_softcapping` flow from a Gemma 2 config into the traced
/// graph and change the eval output; a config without them traces fewer equations.
///
/// Checks:
///   1. The uncapped graph has fewer equations than the capped one, by one softcap (3 ops) per layer plus one
///      after the head.
///   2. Both evaluated with the same weights give different logits.
///   3. The window changes mask values only: a different `sliding_window` traces the same number of equations.
#[test]
fn config_wiring_softcap_changes_graph_and_output() {
    use poot_models::model::{LogitRows, StepSpec};
    use poot_models::test_support::dense::{Dense, Family, step};

    let pos = 4usize;
    let gemma2 = |attn: Option<f32>, last: Option<f32>, window: usize| {
        let m = Dense::new(Family::Gemma2)
            .vocab(32)
            .dims(16, 32, 2)
            .heads(4, 2)
            .head_dim(4)
            .max_positions(16)
            .with("attn_logit_softcapping", attn)
            .with("final_logit_softcapping", last)
            .with("sliding_window", window)
            .f32_model();
        m.model
            .trace(
                &StepSpec::step(step(1, 1, 8, LogitRows::Last)),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap()
    };
    let layers = 2usize;
    let g_none = gemma2(None, None, 4);

    // Tight caps that clamp the tiny test logits. With 0.1-scaled weights and hidden=16, logits are ~0.1-0.3; a cap
    // of 0.05 saturates them to ~0.05 (relative diff > 0.1).
    let g_cap = gemma2(Some(0.05), Some(0.05), 4);

    // Check 1: softcap adds ops to the graph. Each softcap call emits 3 ops: mul_scalar + tanh + mul_scalar. The
    // attention softcap fires once per layer, the final one once after the head: extra = (layers + 1) * 3.
    let expected_extra = (layers + 1) * 3;
    assert_eq!(
        g_cap.eqns().len() - g_none.eqns().len(),
        expected_extra,
        "unexpected op delta: capped {} vs uncapped {}",
        g_cap.eqns().len(),
        g_none.eqns().len()
    );

    let out = |g: &poot_graph_ir::Graph| run_step(g, &[5], pos, &named_weights, &[]).0;
    // Check 2: softcap changes the output, not just the graph structure.
    let max_diff = max_abs_error(
        out(&g_none).as_f32().unwrap(),
        out(&g_cap).as_f32().unwrap(),
    );
    assert!(
        max_diff > 1e-4,
        "softcap config must produce different logits; max_diff={max_diff:.2e}"
    );

    // Check 3: the window only affects mask values at run time.
    assert_eq!(
        gemma2(None, None, 2).eqns().len(),
        g_none.eqns().len(),
        "sliding_window does not change graph structure"
    );
}

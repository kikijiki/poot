//! The attention generator takes the softcap decode, the split-key MLA shape and the guarded live-capacity
//! composite (POOT-730), proven on the CPU oracle and the planner (no device).
//!
//! The device rows (`poot-gpu`, `poot-rocm-gpu`) run the same fixture through the executor contract.
//! This is an integration test: the crate dev-depends on `poot-eval`, which depends back on it.

use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalError, EvalOptions, Value, eval};
use poot_graph_ir::ops::{attention_masked_softcap, repeat_kv};
use poot_graph_ir::{
    AttentionAttrs, BinOp, Builder, Eqn, Graph, KeyLayout, KvBound, OpKind, Operand, Scalar,
    TensorType, ValidationOutputs, ValueId,
};
use poot_graph_plan::{
    Capability, CompileError, CompileOptions, KernelChoice, Plan, PlanError, Submission, Target,
    compile, passes_without_target as optimize,
};
use poot_kernelgen::{AttentionSchedule, AttentionSpec, KernelRequest, ScoreTransform};
use poot_target::{AmdArch, Backend};
use poot_test_util::assert_close;
use poot_test_util::live_attention::{Inputs, LiveAttentionFixture, LiveAttentionGraph};

fn evaluate(
    graph: &Graph<ValidationOutputs>,
    inputs: &HashMap<ValueId, Value>,
) -> Result<Vec<f32>, EvalError> {
    let output = eval(graph, inputs, EvalOptions::new(EvalBudget::UNBOUNDED))?
        .output
        .into_host()
        .expect("a dense attention output");
    Ok(output.as_f32().expect("an f32 output").to_vec())
}

/// Word-for-word equality that treats a NaN as equal to a NaN: two evaluations of the same arithmetic.
fn assert_same_words(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: element count");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let same = (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
        assert!(same, "{label}: element {i}: {x} vs {y}");
    }
}

fn attentions(graph: &Graph<ValidationOutputs>) -> Vec<&AttentionAttrs> {
    graph
        .eqns
        .iter()
        .filter_map(|eqn| match &eqn.op {
            OpKind::Attention(attrs) => Some(attrs),
            _ => None,
        })
        .collect()
}

fn single_key() -> LiveAttentionFixture {
    LiveAttentionFixture {
        batch: 3,
        hq: 4,
        hkv: 2,
        cap: 13,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: Some(30.0),
        dtype: poot_tensor::DType::F32,
    }
}

fn split_key() -> LiveAttentionFixture {
    LiveAttentionFixture {
        dk: 12,
        dv: 6,
        shared_dim: Some(4),
        ..single_key()
    }
}

/// The same graph, with the float select at `index` (construction order: key sources, value, scores) replaced
/// by a reshape of its data operand: that source is no longer guarded.
fn without_guard(mut graph: LiveAttentionGraph, which: usize) -> LiveAttentionGraph {
    let selects: Vec<usize> = graph
        .graph
        .eqns
        .iter()
        .enumerate()
        .filter(|(_, eqn)| {
            matches!(eqn.op, OpKind::Select)
                && graph.graph.aval(eqn.out).dtype == poot_tensor::DType::F32
        })
        .map(|(index, _)| index)
        .collect();
    let shape = graph
        .graph
        .aval(graph.graph.eqns[selects[which]].out)
        .shape
        .clone();
    let eqn: &mut Eqn = &mut graph.graph.eqns[selects[which]];
    let data = eqn.inputs[1];
    eqn.op = OpKind::Reshape { shape };
    eqn.inputs = vec![data];
    graph
}

/// SC-003: the guarded graph and the matched `Live` composite evaluate bit for bit on the CPU oracle, with NaN
/// in every inactive key and value, for a dense and a split key. Mutation: emit `Live` from the score select
/// alone (accept an unguarded source in the matcher's `peel_live_rows`); the poisoned-tail near-miss rows
/// below then fuse, and their evaluation changes (the unguarded graph is NaN, the composite is finite).
#[test]
fn the_guarded_graph_and_the_matched_live_composite_evaluate_bit_identically() {
    let len = [4, 13, 1];
    for fixture in [single_key(), split_key()] {
        let built = fixture.build(true);
        let inputs = fixture.inputs(&built, &len, true).values();
        let optimized = optimize(&built.graph);
        let [attrs] = attentions(&optimized)[..] else {
            panic!(
                "the guarded graph fuses to one attention op: {:?}",
                attentions(&optimized)
            );
        };
        assert_eq!(attrs.bound, KvBound::Live);
        assert_eq!(
            matches!(attrs.key, KeyLayout::Split { .. }),
            fixture.shared_dim.is_some()
        );
        let unfused = evaluate(&built.graph, &inputs).expect("the guarded graph evaluates");
        assert!(
            unfused.iter().all(|x| x.is_finite()),
            "poison reached {unfused:?}"
        );
        assert_same_words(
            "guarded graph vs matched live composite",
            &unfused,
            &evaluate(&optimized, &inputs).expect("the matched graph evaluates"),
        );
    }
}

/// SC-003, the near misses: a score guard without both key sources guarded, a caller-supplied length no
/// declared witness bounds, and a condition one key off stay generic primitives (no `Attention` op) and
/// evaluate exactly as written, poisoned tail included; a score-only additive mask (the production
/// `causal_mask_from_pos`) stays a `Capacity` attention.
#[test]
fn unproved_guards_stay_generic_and_a_score_mask_stays_capacity() {
    let len = [4, 13, 1];
    let fixture = single_key();
    let mut rows: Vec<(&str, LiveAttentionGraph)> = vec![
        ("V unguarded", without_guard(fixture.build(true), 1)),
        ("K unguarded", without_guard(fixture.build(true), 0)),
        ("length without a declared witness", fixture.build(false)),
    ];
    let mut off_by_one = fixture.build(true);
    let iotas: Vec<ValueId> = off_by_one
        .graph
        .eqns
        .iter()
        .filter(|eqn| {
            matches!(
                eqn.op,
                OpKind::Iota {
                    dtype: poot_tensor::DType::I32,
                    ..
                }
            )
        })
        .map(|eqn| eqn.out)
        .collect();
    assert_eq!(
        iotas.len(),
        1,
        "the live bound's key ends come from one exact I32 iota"
    );
    for eqn in &mut off_by_one.graph.eqns {
        let key_ends = matches!(eqn.op, OpKind::Binary(BinOp::Add))
            && matches!(eqn.inputs.first(), Some(Operand::Value(id)) if iotas.contains(id))
            && matches!(eqn.inputs.get(1), Some(Operand::Lit(Scalar::I32(1))));
        if key_ends {
            eqn.inputs[1] = Operand::Lit(Scalar::I32(2));
        }
    }
    rows.push(("conditions one key off", off_by_one));
    for (name, built) in rows {
        let inputs = fixture.inputs(&built, &len, true).values();
        let optimized = optimize(&built.graph);
        // First what the rewrite would change: the unguarded graph is NaN on the poisoned tail and a `Live`
        // composite is finite, so a wrongly emitted `Live` shows up as a different result.
        assert_same_words(
            name,
            &evaluate(&built.graph, &inputs).expect("the near miss evaluates"),
            &evaluate(&optimized, &inputs).expect("the optimized near miss evaluates"),
        );
        assert!(
            attentions(&optimized).is_empty(),
            "{name}: must stay generic, fused to {:?}",
            attentions(&optimized)
        );
    }

    // The production mask: additive `-1e9` on the scores, no live bound. It fuses as a capacity attention.
    let b = Builder::new();
    let (hq, cap, d) = (2usize, 6usize, 4usize);
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hq, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, d]));
    let pos = b.slot(
        poot_graph_ir::Slot::Pos,
        TensorType::new(vec![1, 1], poot_tensor::DType::I32),
    );
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, cap, None);
    let out = attention_masked_softcap(&b, q, k, v, 1, 0.5, mask, None);
    let graph = b.finish(out).with_validations(Vec::new());
    let optimized = optimize(&graph);
    let [attrs] = attentions(&optimized)[..] else {
        panic!("the production mask fuses to one attention op");
    };
    assert_eq!(
        (attrs.bound, attrs.key),
        (KvBound::Capacity, KeyLayout::Single)
    );
}

/// SC-002 on the oracle: capacity 4096, the length derived in the graph as 64 from `Pos`, NaN in every key and
/// value from 64 on. The guarded graph and the first-64-key oracle agree, all finite. Mutation: drop the
/// decomposition's V guard (`guard_rows(v)` in `ops::attention_over_keys`); `0 * NaN` reaches the value
/// matmul and the finiteness assertion fails.
#[test]
fn a_poisoned_tail_past_the_live_length_never_reaches_the_output() {
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 4096,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: None,
        dtype: poot_tensor::DType::F32,
    };
    let built = fixture.build(true);
    let inputs = fixture.inputs(&built, &[64], true);
    let got = evaluate(&built.graph, &inputs.values()).expect("the guarded graph evaluates");
    assert!(got.iter().all(|x| x.is_finite()), "poison reached {got:?}");
    assert_close(&got, &fixture.oracle(&inputs, &built, &[64]), 1e-5);
    // The fixture must put NaN where a capacity-long read goes.
    let poisoned = |inputs: &Inputs| {
        inputs.0[&built.v]
            .as_f32()
            .unwrap()
            .iter()
            .any(|x| x.is_nan())
    };
    assert!(poisoned(&inputs));
}

/// SC-004 on the production path: lengths 0, -1 and capacity + 1 fail the validation channel (`eval` publishes
/// nothing: it returns the validation failure, not an output), for the guarded graph and the matched live
/// graph alike; the valid endpoints 1 and capacity evaluate and match the oracle. Mutation: widen the
/// guard's bound in `ops::attention_live` (`guard_index_bounds(last, lk + 8)`); capacity + 1 then evaluates.
#[test]
fn an_out_of_range_live_length_fails_validation_and_the_endpoints_run() {
    let fixture = LiveAttentionFixture {
        batch: 1,
        cap: 9,
        ..single_key()
    };
    let built = fixture.build(true);
    let optimized = optimize(&built.graph);
    for len in [0, -1, 10] {
        let inputs = fixture.inputs(&built, &[len], false).values();
        for (name, graph) in [
            ("guarded graph", &built.graph),
            ("matched graph", &optimized),
        ] {
            let outcome = evaluate(graph, &inputs);
            assert!(
                matches!(outcome, Err(EvalError::Validation(_))),
                "{name}, len {len}: expected a validation failure, got {outcome:?}"
            );
        }
    }
    // Extreme lengths are not exact in the F32 cast the length guard reads: a typed cast fault, never a
    // result. Mutation: let them through (clamp `len` before the guard); the row gets `Ok`.
    for len in [i32::MIN + 1, i32::MAX] {
        let inputs = fixture.inputs(&built, &[len], false).values();
        for (name, graph) in [
            ("guarded graph", &built.graph),
            ("matched graph", &optimized),
        ] {
            let outcome = evaluate(graph, &inputs);
            assert!(
                matches!(outcome, Err(EvalError::Cast(_) | EvalError::Validation(_))),
                "{name}, len {len}: expected a typed cast or validation failure, got {outcome:?}"
            );
        }
    }
    assert_eq!(
        attentions(&optimized).len(),
        1,
        "the live attention is matched"
    );
    for len in [1, 9] {
        let inputs = fixture.inputs(&built, &[len], true);
        let expected = fixture.oracle(&inputs, &built, &[len]);
        for (name, graph) in [
            ("guarded graph", &built.graph),
            ("matched graph", &optimized),
        ] {
            let got = evaluate(graph, &inputs.values()).expect("a valid length publishes");
            assert_close(&got, &expected, 1e-5);
            assert!(got.iter().all(|x| x.is_finite()), "{name}, len {len}");
        }
    }
}

fn targets() -> [(&'static str, Target); 3] {
    [
        Backend::SpirvVulkan,
        Backend::Nvptx,
        Backend::AmdGcn(AmdArch::gfx1151()),
    ]
    .map(|backend| {
        (
            match backend {
                Backend::SpirvVulkan => "spirv",
                Backend::Nvptx => "nvptx",
                _ => "amdgcn",
            },
            Target {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            },
        )
    })
}

/// The one generated attention request of `graph` compiled for `target`, and how many launches its plan makes.
fn planned_attention(graph: &Graph<ValidationOutputs>, target: &Target) -> (AttentionSpec, usize) {
    let program = compile(
        graph,
        target,
        &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
    )
    .expect("the fixture compiles");
    let mut found = program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)));
    let (eqn, plan) = found
        .next()
        .expect("the compiled program keeps the attention");
    assert!(found.next().is_none(), "exactly one attention equation");
    let KernelChoice::Generated(KernelRequest::Attention(spec)) = program.kernel_choice(eqn) else {
        panic!(
            "the attention is not generated: {:?}",
            program.kernel_choice(eqn)
        );
    };
    let Plan::Dispatches(dispatches) = plan else {
        panic!("the attention launches: {plan:?}");
    };
    (spec.clone(), dispatches.len())
}

/// SC-001: Gemma2's softcapped decode and a DeepSeek-V2 MLA decode (`dk` 192, `dv` 128, split key) select the
/// initial streaming attention schedule on every target, as one dispatch each. Mutations: restore the decode
/// softcap decline in `match_attention` (the Gemma2 row plans no attention), and drop the split-key
/// recognition in `peel_key` (the MLA row plans no attention).
#[test]
fn gemma2_softcap_decode_and_mla_select_the_streaming_attention_schedule() {
    // Gemma2: GQA decode with the logit softcap, over the in-graph position mask.
    let (hq, hkv, cap, d) = (4usize, 2usize, 16usize, 32usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, cap, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, cap, d]));
    let pos = b.slot(
        poot_graph_ir::Slot::Pos,
        TensorType::new(vec![1, 1], poot_tensor::DType::I32),
    );
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, cap, None);
    let out = attention_masked_softcap(&b, q, k, v, hq / hkv, 0.25, mask, Some(50.0));
    let gemma2 = b.finish(out).with_validations(Vec::new());

    // DeepSeek-V2 MLA: per-head `k_nope` (128) and the one shared `k_pe` (64) the tracer broadcasts and
    // concatenates, a 128-wide value.
    let (hq, cap, nope, rope, dv) = (2usize, 16usize, 128usize, 64usize, 128usize);
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hq, 1, nope + rope]));
    let k_nope = b.constant("k_nope", TensorType::f32(vec![1, hq, cap, nope]));
    let k_pe = b.constant("k_pe", TensorType::f32(vec![1, 1, cap, rope]));
    let v = b.constant("v", TensorType::f32(vec![1, hq, cap, dv]));
    let pos = b.slot(
        poot_graph_ir::Slot::Pos,
        TensorType::new(vec![1, 1], poot_tensor::DType::I32),
    );
    let mask = poot_graph_ir::ops::causal_mask_from_pos(&b, pos, cap, None);
    let key = b.concat(3, &[k_nope, repeat_kv(&b, k_pe, hq)]);
    let out = attention_masked_softcap(&b, q, key, v, 1, 0.07, mask, None);
    let mla = b.finish(out).with_validations(Vec::new());

    for (name, target) in targets() {
        let (spec, launches) = planned_attention(&gemma2, &target);
        assert!(
            matches!(spec.score, ScoreTransform::ScaleSoftcap { .. }),
            "{name}: Gemma2 carries its softcap: {spec:?}"
        );
        assert_eq!((spec.lq, spec.dk, spec.dv), (1, 32, 32), "{name}");
        assert!(
            matches!(spec.schedule, AttentionSchedule::Streaming { .. }),
            "{name}"
        );
        assert_eq!(launches, 1, "{name}: one dispatch");

        let (spec, launches) = planned_attention(&mla, &target);
        assert_eq!((spec.dk, spec.dv), (192, 128), "{name}: MLA widths");
        assert_eq!(
            spec.key,
            poot_kernelgen::KeyLayout::Split { shared_dim: 64 },
            "{name}"
        );
        assert!(
            matches!(spec.schedule, AttentionSchedule::Streaming { .. }),
            "{name}"
        );
        assert_eq!(launches, 1, "{name}: one dispatch");
    }
}

/// SC-002, planning half: the compiled live decode plans `bound=live` over the length operand the graph derived
/// from `Pos`, with one dispatch whose grid does not depend on the length. Mutation: plan the bound as
/// `Static` in `planner/attention.rs`; the assertion on `spec.bound` fails.
#[test]
fn the_compiled_live_decode_plans_a_live_bound_and_a_length_independent_launch() {
    let fixture = LiveAttentionFixture {
        batch: 1,
        hq: 2,
        hkv: 1,
        cap: 4096,
        dk: 8,
        dv: 8,
        shared_dim: None,
        softcap: None,
        dtype: poot_tensor::DType::F32,
    };
    let built = fixture.build(true);
    for (name, target) in targets() {
        let (spec, launches) = planned_attention(&built.graph, &target);
        assert_eq!(spec.bound, poot_kernelgen::KeyBound::Live, "{name}");
        assert_eq!(launches, 1, "{name}");
        assert_eq!(
            spec.workgroups(),
            2,
            "{name}: the grid is (batch, head, row)"
        );
    }
}

/// POOT-1040: the live bound builds its key ends in exact I32 (an I32 `Iota` plus an I32 literal) and never
/// round-trips through F32: the guarded graph holds no F32 -> I32 `Cast`. Mutation: build the ends from an F32
/// iota plus a `Cast` to I32 (`LiveKeys::new` in `ops::attention_over_keys`); the iota assertion and the cast
/// assertion both fail.
#[test]
fn the_live_bound_builds_its_key_ends_in_exact_i32() {
    let built = single_key().build(true);
    let g = &built.graph;
    let i32_iotas = g
        .eqns
        .iter()
        .filter(|eqn| {
            matches!(
                eqn.op,
                OpKind::Iota {
                    dtype: poot_tensor::DType::I32,
                    ..
                }
            )
        })
        .count();
    assert_eq!(i32_iotas, 1, "one exact I32 iota feeds the key ends");
    let float_to_int_casts: Vec<_> = g
        .eqns
        .iter()
        .filter(|eqn| {
            matches!(
                eqn.op,
                OpKind::Cast {
                    to: poot_tensor::DType::I32
                }
            ) && g
                .aval(match eqn.inputs[0] {
                    Operand::Value(id) => id,
                    Operand::Lit(_) => unreachable!("a cast reads a value"),
                })
                .dtype
                == poot_tensor::DType::F32
        })
        .collect();
    assert!(
        float_to_int_casts.is_empty(),
        "no F32 -> I32 cast on the live path: {float_to_int_casts:?}"
    );
}

/// POOT-1040: a guarded live graph that fails the matcher's proof (no declared length witness, so the length is
/// not bounded) stays generic primitives, and a generic graph with no exact-I32 iota would be refused by every
/// device plan. It compiles for every target with no attention kernel. Mutation: build the key ends from an
/// F32 iota plus a `Cast` to I32; every target refuses the cast (`DtypeLowering`) and the compile fails.
#[test]
fn a_declined_live_graph_plans_on_every_target() {
    let built = single_key().build(false);
    for (name, target) in targets() {
        let program = compile(
            &built.graph,
            &target,
            &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
        )
        .unwrap_or_else(|e| panic!("{name}: the declined live graph must plan: {e}"));
        assert!(
            program
                .planned()
                .all(|(eqn, _)| !matches!(eqn.op, OpKind::Attention(_))),
            "{name}: the unproved graph stays generic"
        );
    }
}

/// POOT-1040: a declined live graph (no declared length witness) cannot publish a failure, so the index guard's
/// in-range fallback decides: a length outside `1..=cap` (0, -1, cap, cap + 91) selects the fallback index 0, which
/// is exactly a live length of 1 (bit for bit, never everything and never nothing); a length the guard's F32 cast
/// cannot hold (`i32::MIN`, `i32::MIN + 1`, `i32::MAX`) is a typed `Cast` fault, never a result. Mutation: bound
/// the guard by `lk + 8` in `ops::attention_live`; length 10 then keeps all nine keys and the length-1 comparison
/// fails.
#[test]
fn a_declined_live_graph_treats_an_out_of_range_length_as_length_one() {
    let fixture = LiveAttentionFixture {
        batch: 1,
        cap: 9,
        ..single_key()
    };
    let built = fixture.build(false);
    let run = |len: i32| {
        evaluate(
            &built.graph,
            &fixture.inputs(&built, &[len], false).values(),
        )
    };
    let one = run(1).expect("length 1 evaluates");
    assert_ne!(one, run(9).expect("length 9 evaluates"), "the rows differ");
    for len in [0, -1, 10, 100] {
        assert_same_words(
            &format!("len {len} vs len 1"),
            &run(len).expect("an out-of-range length falls back"),
            &one,
        );
    }
    for len in [i32::MIN, i32::MIN + 1, i32::MAX] {
        let outcome = run(len);
        assert!(
            matches!(outcome, Err(EvalError::Cast(_))),
            "len {len}: expected a typed cast fault, got {outcome:?}"
        );
    }
}

/// The two-byte float dtypes the live bound runs in.
const NARROW: [poot_tensor::DType; 2] = [poot_tensor::DType::BF16, poot_tensor::DType::F16];

fn narrow(fixture: LiveAttentionFixture, dtype: poot_tensor::DType) -> LiveAttentionFixture {
    LiveAttentionFixture { dtype, ..fixture }
}

/// POOT-1041, the oracle: the guarded live composite over BF16 and F16 operands (dense and split key, NaN in
/// every inactive key and value) evaluates to the first-`len` keys' plain attention at the same dtype, bit
/// for bit: the bit-moving `Select` lets nothing of the tail through, and the same-dtype arithmetic matches.
/// Mutation: drop the V guard (`guard_rows(v)` in `ops::attention_over_keys`); `0 * NaN` reaches the value
/// matmul and the finiteness assertion fails.
#[test]
fn a_narrow_live_composite_evaluates_to_the_first_keys_oracle_with_a_poisoned_tail() {
    let len = [4, 13, 1];
    for dtype in NARROW {
        for fixture in [single_key(), split_key()] {
            let fixture = narrow(fixture, dtype);
            let built = fixture.build(true);
            let inputs = fixture.inputs(&built, &len, true);
            let got =
                evaluate(&built.graph, &inputs.values()).expect("the guarded graph evaluates");
            assert!(
                got.iter().all(|x| x.is_finite()),
                "{dtype:?}: poison reached {got:?}"
            );
            assert_same_words(
                &format!("{dtype:?} guarded graph vs first-keys oracle"),
                &got,
                &fixture.oracle(&inputs, &built, &len),
            );
        }
    }
}

/// POOT-1041, the planner: a BF16 or F16 live graph never fuses into the F32 attention kernel (the guarded
/// graph keeps its primitives), and its float selects plan as pointwise kernels over native two-byte branch
/// leaves on ROCm and PTX. SPIR-V has no native two-byte lane, so the same graph is refused there with the
/// typed `DtypeLowering` on the select. Mutations: drop the `NotF32` decline in `match_attention` (the BF16
/// capacity row fuses to an `Attention` op); drop the BF16/F16 arm of the `Select` planning in
/// `planner/elementwise.rs` (the ROCm and PTX plans fail `DtypeLowering`, and the SPIR-V refusal row stays).
#[test]
fn a_narrow_live_graph_plans_its_selects_natively_and_spirv_refuses_them() {
    for dtype in NARROW {
        let built = narrow(single_key(), dtype).build(true);
        assert!(
            attentions(&optimize(&built.graph)).is_empty(),
            "{dtype:?}: the F32 attention kernel must not claim a narrow chain"
        );
        let leaf = match dtype {
            poot_tensor::DType::BF16 => poot_kernelgen::LeafTy::BF16,
            _ => poot_kernelgen::LeafTy::F16,
        };
        for (name, target) in targets() {
            let result = compile(
                &built.graph,
                &target,
                &CompileOptions::new(Submission::Replay, poot_graph_plan::CompileLimits::STANDARD),
            );
            if name == "spirv" {
                let Err(CompileError::Plan(error)) = result else {
                    panic!("{dtype:?} {name}: the narrow select must be refused, got {result:?}");
                };
                let PlanError::Refused(refusal) = *error else {
                    panic!("{dtype:?} {name}: expected a typed refusal, got {error:?}");
                };
                assert_eq!(refusal.missing, Capability::DtypeLowering, "{dtype:?}");
                assert!(
                    matches!(refusal.op, OpKind::Select),
                    "{dtype:?}: {refusal:?}"
                );
                continue;
            }
            let program = result.unwrap_or_else(|e| panic!("{dtype:?} {name}: must plan: {e}"));
            let selects: Vec<_> = program
                .planned()
                .filter(|(eqn, _)| {
                    matches!(eqn.op, OpKind::Select) && program.graph().aval(eqn.out).dtype == dtype
                })
                .collect();
            // The K and V row guards, then the score guard.
            assert_eq!(selects.len(), 3, "{dtype:?} {name}");
            for (eqn, _) in selects {
                let KernelChoice::Generated(KernelRequest::Pointwise(spec)) =
                    program.kernel_choice(eqn)
                else {
                    panic!("{dtype:?} {name}: a select is a generated pointwise kernel");
                };
                assert_eq!(
                    spec.kernel.leaf_tys,
                    [poot_kernelgen::LeafTy::I32, leaf, leaf],
                    "{dtype:?} {name}"
                );
            }
        }
    }
}

/// POOT-1041: the generated attention kernel is F32 only, so a BF16 capacity chain (no live bound) stays the
/// decomposition too instead of fusing to an `Attention` op the planner would refuse. Mutation: drop the
/// `NotF32` decline in `match_attention`; the chain fuses.
#[test]
fn a_narrow_capacity_attention_chain_stays_the_decomposition() {
    for dtype in NARROW {
        let b = Builder::new();
        let (hq, cap, d) = (2usize, 6usize, 4usize);
        let input = |name, shape: Vec<usize>| b.constant(name, TensorType::new(shape, dtype));
        let q = input("q", vec![1, hq, 1, d]);
        let k = input("k", vec![1, hq, cap, d]);
        let v = input("v", vec![1, hq, cap, d]);
        let mask = input("mask", vec![1, 1, 1, cap]);
        let out = attention_masked_softcap(&b, q, k, v, 1, 0.5, mask, None);
        let graph = b.finish(out).with_validations(Vec::new());
        assert!(
            attentions(&optimize(&graph)).is_empty(),
            "{dtype:?}: the chain must stay generic"
        );
    }
}

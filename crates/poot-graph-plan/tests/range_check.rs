//! The opt-in range detector (POOT-1475, `receipts/1471-narrow-without-range-guards/design.md` section 6.2):
//! `CompileOptions::range_check` gives every operand an f16-narrowed op rounds a validation witness, so a step that
//! leaves f16's range is a typed validation fault naming the first op, its layer and the operand's role.
//!
//! Every row drives the production `compile` on RADV caps (f16 fragments only) and on gfx1151 caps, and evaluates the
//! compiled graph with the CPU oracle, which gates the result against the graph's validation packet exactly as an
//! executor does: a failing witness is an `EvalError::Validation` naming it.

use std::collections::HashMap;

use poot_eval::{EvalError, EvalOptions, Value, eval};
use poot_graph_ir::ops::rmsnorm;
use poot_graph_ir::{
    AttentionAttrs, Builder, Graph, MatmulPrecision, OpKind, Operand, OperandPrecision, Slot,
    TensorType, ValueId,
};
use poot_graph_plan::{CompileLimits, CompileOptions, Program, Target, compile};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::{DType, HostTensor};

const M: usize = 48;
const K: usize = 64;

fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
    caps.register_file = AmdArch::gfx1151().register_file();
    caps.subgroup = Queried::Known(SubgroupSupport::Present {
        min_size: 32,
        max_size: 32,
    });
    Target::Gpu {
        backend: Backend::AmdGcn(AmdArch::gfx1151()),
        caps,
    }
}

fn radv() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::wgpu_rdna3_igpu(),
    }
}

/// The two targets on which the narrow policy rounds to f16.
fn f16_targets() -> [(&'static str, Target); 2] {
    [("radv", radv()), ("gfx1151", rocm_wmma())]
}

fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

/// `g` compiled under `policy`. The detector's subject is the narrowed operand, so the narrowing is declared on every op
/// the dtype rule admits (`declare_precision`) rather than left to the planner's cost model, which keeps radv exact.
fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision, range_check: bool) -> Program {
    let options = CompileOptions {
        declare_precision: true,
        ..CompileOptions::new(CompileLimits::STANDARD)
            .with_matmul_precision(policy)
            .with_range_check(range_check)
    };
    compile(g, target, &options).expect("the graph compiles")
}

fn run(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Result<Vec<f32>, EvalError> {
    Ok(eval(g, inputs, EvalOptions::default())?
        .output
        .into_host()
        .expect("a dense graph")
        .as_f32()
        .expect("an f32 output")
        .to_vec())
}

/// The name of the witness a failing evaluation names.
fn failing_witness(result: Result<Vec<f32>, EvalError>) -> String {
    match result {
        Err(EvalError::Validation(failure)) => failure.name,
        Err(other) => panic!("expected a validation fault, got {other}"),
        Ok(_) => panic!("expected a validation fault, the run passed"),
    }
}

/// Two decoder layers over `[K, K]` f32 weights: `x = x + rmsnorm(x) W` per layer. `norm_scales` scales each layer's
/// norm weight, which sets the size of the activation its projection reads (a norm cancels the scale of its input).
fn layers(norm_scales: [f32; 2]) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::for_tests();
    let mut inputs = HashMap::new();
    let mut constant = |name: &str, shape: Vec<usize>, seed: u64, scale: f32| {
        let n = shape.iter().product();
        let t = b.constant(name, TensorType::f32(shape.clone())).unwrap();
        inputs.insert(
            t.id(),
            Value::Host(HostTensor::f32(shape, values(n, seed, scale))),
        );
        t
    };
    let mut x = constant("x", vec![M, K], 1, 1.0);
    for (layer, scale) in norm_scales.into_iter().enumerate() {
        let norm = constant(&format!("norm{layer}"), vec![K], 2 + layer as u64, scale);
        let w = constant(&format!("w{layer}"), vec![K, K], 5 + layer as u64, 0.25);
        let _layer = b.layer_scope(layer);
        let h = rmsnorm(&b, x, norm, 1e-6).unwrap();
        x = b
            .binary(poot_graph_ir::op::BinOp::Add, x, b.matmul(h, w).unwrap())
            .unwrap();
    }
    (b.finish(x).unwrap(), inputs)
}

/// An activation past f16's range in the second layer is a validation fault that names that layer, the op and the
/// operand; the first layer's operand, in range, is not named. Mutation: in `check_f16_ranges` skip the equations of
/// `layer` 1 (or compare against the f16 maximum instead of the rounding edge's complement) and the run passes or
/// names another witness.
#[test]
fn an_overflow_is_a_fault_naming_its_layer_op_and_operand() {
    let (g, inputs) = layers([1.0, 1.0e6]);
    for (name, target) in f16_targets() {
        let program = compiled(&g, &target, MatmulPrecision::Narrow, true);
        let witness = failing_witness(run(program.graph(), &inputs));
        assert!(
            witness.starts_with("f16_range.") && witness.contains(".l1."),
            "{name}: the overflow is in layer 1, got {witness}"
        );
        assert!(
            witness.ends_with(".matmul.activation"),
            "{name}: the operand is a contraction's activation, got {witness}"
        );
    }
}

/// With every activation in range the detector stays quiet and the program publishes the output it publishes
/// without the detector, bit for bit: a witness reads a value, it never changes one. Mutation: in `check_f16_ranges`
/// compare `-|x|` against `-1.0` (the check fires on ordinary values) and this row fails with the layer-0 witness.
#[test]
fn in_range_data_passes_and_changes_nothing() {
    let (g, inputs) = layers([1.0, 1.0]);
    for (name, target) in f16_targets() {
        let off = run(
            compiled(&g, &target, MatmulPrecision::Narrow, false).graph(),
            &inputs,
        )
        .expect("the program without the detector runs");
        let on = run(
            compiled(&g, &target, MatmulPrecision::Narrow, true).graph(),
            &inputs,
        )
        .expect("an in-range run passes the detector");
        assert_eq!(on.len(), off.len());
        assert!(
            on.iter().zip(&off).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: the detector changed the output"
        );
    }
}

/// Off costs nothing: the program has no witness, no extra equation and does not name the pass; on, it has exactly
/// one witness per distinct watched operand (the two layers' activations) and the equations that compute them; and
/// when the policy narrows nothing there is nothing to watch, so even on the option adds nothing. Mutation: run
/// `check_f16_ranges` whatever `range_check` says (in `compile`) and the off assertions fail.
#[test]
fn off_adds_no_witness_and_no_equation() {
    let (g, _) = layers([1.0, 1.0]);
    for (name, target) in f16_targets() {
        let off = compiled(&g, &target, MatmulPrecision::Narrow, false);
        let on = compiled(&g, &target, MatmulPrecision::Narrow, true);
        let exact = compiled(&g, &target, MatmulPrecision::Exact, true);
        assert_eq!(off.graph().witnesses().count(), 0, "{name}: off");
        assert!(
            !off.passes().any(|pass| pass == "check_f16_ranges"),
            "{name}: off runs no detector pass"
        );
        assert_eq!(
            on.graph().witnesses().count(),
            2,
            "{name}: one witness per watched activation"
        );
        assert!(on.passes().any(|pass| pass == "check_f16_ranges"), "{name}");
        assert!(
            on.graph().eqns().len() > off.graph().eqns().len(),
            "{name}: the witnesses are computed"
        );
        assert_eq!(
            exact.graph().witnesses().count(),
            0,
            "{name}: nothing narrows under Exact, so nothing is watched"
        );
        assert_eq!(
            exact.graph().eqns().len(),
            compiled(&g, &target, MatmulPrecision::Exact, false)
                .graph()
                .eqns()
                .len(),
            "{name}: Exact with the option is the Exact program"
        );
    }
}

/// One causal attention in layer 17 over an f16 cache of positive keys and values, a softcap and a positive query
/// of magnitude up to `q_scale`. The query is an activation (a slot), as in a served model. All entries are
/// positive so an overflowed query gives `+inf` scores and never `inf - inf`: the case the logits refusal cannot see.
fn softcapped_attention(q_scale: f32) -> (Graph, HashMap<ValueId, Value>) {
    const HQ: usize = 4;
    const HKV: usize = 2;
    const LQ: usize = 512;
    const LK: usize = 512;
    const D: usize = 64;
    let b = Builder::for_tests();
    let mut inputs = HashMap::new();
    let q_shape = vec![1, HQ, LQ, D];
    let kv_shape = vec![1, HKV, LK, D];
    let q = b
        .slot_named(Slot::Activation, "q", TensorType::f32(q_shape.clone()))
        .unwrap();
    let positive = |n: usize, seed: u64, scale: f32| -> Vec<f32> {
        values(n, seed, 0.5)
            .into_iter()
            .map(|v| (v.abs() + 0.05) * scale)
            .collect()
    };
    inputs.insert(
        q.id(),
        Value::Host(HostTensor::f32(q_shape, positive(HQ * LQ * D, 1, q_scale))),
    );
    let mut stored = |name: &str, seed: u64| {
        let t = b
            .constant(name, TensorType::new(kv_shape.clone(), DType::F16))
            .unwrap();
        let words = positive(HKV * LK * D, seed, 1.0)
            .into_iter()
            .map(poot_quant::scalar::f32_to_f16)
            .collect();
        inputs.insert(
            t.id(),
            Value::Host(HostTensor::f16(kv_shape.clone(), words)),
        );
        t
    };
    let (k, v) = (stored("k", 2), stored("v", 3));
    let mask = b
        .constant("mask", TensorType::f32(vec![1, 1, LQ, LK]))
        .unwrap();
    let causal = (0..LQ)
        .flat_map(|i| (0..LK).map(move |j| if j <= i + LK - LQ { 0.0 } else { -1.0e9 }))
        .collect();
    inputs.insert(
        mask.id(),
        Value::Host(HostTensor::f32(vec![1, 1, LQ, LK], causal)),
    );
    let _layer = b.layer_scope(17);
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs {
                precision: OperandPrecision::Exact,
                ..AttentionAttrs::new(HQ / HKV, 1.0e-6, Some(30.0))
            }),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    (b.finish(out).unwrap(), inputs)
}

/// The known limit of the default (design 4.3) and the tool that covers it: under a score softcap an overflowed query
/// gives `c * tanh(inf / c) = c`, so the narrowed run is finite (and wrong: it differs from the exact run), and the
/// logits refusal never fires. With `range_check` the same run is a validation fault naming the attention's layer and
/// its query. Mutation: in `rounded_operands` leave an attention's query out and the second half fails (the run
/// passes).
#[test]
fn a_softcapped_attention_overflows_finitely_and_the_detector_names_it() {
    let (g, inputs) = softcapped_attention(1.0e6);
    // The attention is declared narrowed, which radv's one-wave fragments cannot plan (the planner refuses it), so
    // the row runs where a matrix unit schedules it.
    let (name, target) = ("gfx1151", rocm_wmma());
    let exact = run(
        compiled(&g, &target, MatmulPrecision::Exact, false).graph(),
        &inputs,
    )
    .expect("the exact run");
    let narrow = run(
        compiled(&g, &target, MatmulPrecision::Narrow, false).graph(),
        &inputs,
    )
    .expect("the narrowed run publishes");
    assert!(
        narrow.iter().all(|v| v.is_finite()),
        "{name}: under a softcap the overflow is finite (the documented limit)"
    );
    assert!(
        narrow
            .iter()
            .zip(&exact)
            .any(|(n, e)| (n - e).abs() > 1.0e-3),
        "{name}: and it is wrong: the capped scores are uniform"
    );
    let witness = failing_witness(run(
        compiled(&g, &target, MatmulPrecision::Narrow, true).graph(),
        &inputs,
    ));
    assert!(
        witness.contains(".l17.") && witness.ends_with(".attention.query"),
        "{name}: the detector names layer 17's attention query, got {witness}"
    );
}

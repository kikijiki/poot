//! The f16 range of attention under the default `Narrow` (POOT-1474,
//! `receipts/1471-narrow-without-range-guards/design.md` section 7): an attention the narrow policy rounds to f16
//! reads its query unscaled, and a query past f16's range reaches the output as a non-finite value.
//!
//! Every row drives the production `compile` on RADV caps (f16 fragments only) and on gfx1151 caps (bf16 and f16;
//! the narrow rule takes f16 there since POOT-1458), and evaluates the compiled graph with the CPU oracle, which
//! computes the program's declared rounding.

use std::collections::HashMap;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{
    AttentionAttrs, Builder, FragmentDtype, Graph, MatmulPrecision, OpKind, Operand,
    OperandPrecision, TensorType, ValueId,
};
use poot_graph_plan::{CompileLimits, CompileOptions, Program, Target, compile};
use poot_target::{AmdArch, Backend, DeviceCaps, MatrixUnits, Queried, SubgroupSupport};
use poot_tensor::{DType, HostTensor};

const HQ: usize = 4;
const HKV: usize = 2;
const LQ: usize = 32;
const LK: usize = 64;
const D: usize = 16;

const F16: OperandPrecision = OperandPrecision::Narrow(FragmentDtype::F16);

fn rocm_wmma() -> Target {
    let mut caps = DeviceCaps::rocm_default();
    caps.matrix_units = Queried::Known(MatrixUnits::amd("gfx1151", 32));
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

/// RADV, and gfx1151 with both fragment dtypes: the two targets on which the narrow policy rounds attention to f16.
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

/// One causal prefill attention into a 64-key cache stored f16 (the default KV dtype), over an f32 query of
/// magnitude up to `q_scale`. `precision` is the op's own: `Exact` lets the policy choose, a declared `Narrow(f16)`
/// stands as it is (the f16 program without the guard). `scale` multiplies the scores; it is chosen with
/// `q_scale` so the scores stay of order one and the softmax is not one-hot, which would make a single rounding
/// decide the output.
fn attention(
    q_scale: f32,
    scale: f32,
    precision: OperandPrecision,
) -> (Graph, HashMap<ValueId, Value>) {
    let b = Builder::for_tests();
    let mut inputs = HashMap::new();
    let q_shape = vec![1, HQ, LQ, D];
    let kv_shape = vec![1, HKV, LK, D];
    let q = b.constant("q", TensorType::f32(q_shape.clone())).unwrap();
    inputs.insert(
        q.id(),
        Value::Host(HostTensor::f32(
            q_shape.clone(),
            values(HQ * LQ * D, 1, q_scale),
        )),
    );
    let mut stored = |name: &str, seed: u64| {
        let t = b
            .constant(name, TensorType::new(kv_shape.clone(), DType::F16))
            .unwrap();
        let words = values(HKV * LK * D, seed, 1.0)
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
    let mut plan = b.append_plan(0).unwrap();
    let out = plan
        .equation(
            OpKind::Attention(AttentionAttrs {
                precision,
                ..AttentionAttrs::new(HQ / HKV, scale, None)
            }),
            [q, k, v, mask].map(|t| Operand::Value(t.id())).to_vec(),
        )
        .expect("the attention types");
    plan.declare_result(out).unwrap();
    let mut prepared = b.preflight_append(plan).unwrap();
    let out = b.commit_append(&mut prepared).unwrap();
    (b.finish(out).unwrap(), inputs)
}

fn compiled(g: &Graph, target: &Target, policy: MatmulPrecision) -> Program {
    let options = CompileOptions::new(CompileLimits::STANDARD).with_matmul_precision(policy);
    compile(g, target, &options).expect("the attention compiles")
}

fn output(g: &Graph, inputs: &HashMap<ValueId, Value>) -> Vec<f32> {
    eval(g, inputs, EvalOptions::default())
        .expect("the graph evaluates")
        .output
        .into_host()
        .expect("a dense graph")
        .as_f32()
        .expect("an f32 output")
        .to_vec()
}

/// The precision of the program's attention.
fn attention_precision(program: &Program) -> OperandPrecision {
    program
        .planned()
        .find_map(|(eqn, _)| match &eqn.op {
            OpKind::Attention(attrs) => Some(attrs.precision),
            _ => None,
        })
        .expect("the program plans an attention")
}

/// The default's failure mode: a query reaching 1e6 (past f16's 65504 on every row) compiles under the narrow policy
/// to an f16 attention that reads it unscaled, so the query rounds to infinity and the output is not finite (it is
/// not clamped and not finite-and-wrong); the exact program over the same data is finite. Serving refuses a
/// non-finite logit with a typed error (`receipts/1471-narrow-without-range-guards/design.md` section 4.2).
/// Mutation: make the narrowing saturate in the oracle (`poot-eval` f16 rounding clamps to 65504); the output turns
/// finite and this row fails.
#[test]
fn a_query_past_the_f16_range_gives_a_non_finite_output() {
    let (g, inputs) = attention(1.0e6, 1.0e-6, OperandPrecision::Exact);
    let exact = output(
        compiled(&g, &rocm_wmma(), MatmulPrecision::Exact).graph(),
        &inputs,
    );
    assert!(
        exact.iter().all(|v| v.is_finite()),
        "the exact program is finite over the same data"
    );
    for (name, target) in f16_targets() {
        let program = compiled(&g, &target, MatmulPrecision::Narrow);
        assert_eq!(
            attention_precision(&program),
            F16,
            "{name}: the attention rounds to f16"
        );
        let narrow = output(program.graph(), &inputs);
        assert!(
            narrow.iter().any(|v| !v.is_finite()),
            "{name}: the overflowed query reached the output as a finite value"
        );
    }
}

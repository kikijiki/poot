//! POOT-1175: the part of a Qwen2.5-0.5B decoder block the Intel NPU lowering admits, run on the device.
//!
//! The block is the Qwen2 tracer's at the real dimensions (hidden 896, 14 query heads over 2 key/value heads of
//! 64, feed-forward 4864, 8 tokens); `poot-models`' `npu_qwen_refusals` row lists what the lowering refuses of
//! it. What it admits is run here in F16 through the ordinary route (planner, `emit_vpu`, vendor graph
//! compiler, Level Zero): the attention core (projections with biases, grouped-query expansion, scores, the
//! softmax numerator, output projection) and the MLP with its residual, each against the CPU oracle within a
//! stated tolerance, with a timing receipt. Weights are synthetic; no checkpoint is loaded.
//!
//! ```text
//! nix develop .#npu-intel -c cargo nextest run -p poot-npu --features vpu --test vpu_qwen --test-threads=1 --no-capture
//! ```

#![cfg(feature = "vpu")]

mod vpu_common;

use std::time::{Duration, Instant};

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, NoSync};
use poot_executor_parity::{Fixture, fixture_weights, staged_on};
use poot_graph_ir::{BinOp, Builder, Slot, TensorType, Traced, UnOp};
use poot_tensor::DType;

use vpu_common::{constant, npu, store, x_step};

/// Qwen2.5-0.5B-Instruct: hidden width, feed-forward width, query heads, key/value heads, head dimension.
const HIDDEN: usize = 896;
const INTER: usize = 4864;
const HEADS: usize = 14;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
/// Prompt tokens of the prefill.
const TOKENS: usize = 8;

/// `x @ w^T` for an HF-layout `[out, in]` F16 weight.
fn linear(b: &Builder, x: Traced, w: Traced) -> Traced {
    let wt = b.transpose(w, vec![1, 0]).unwrap();
    b.matmul(x, wt).unwrap()
}

fn f16(shape: &[usize]) -> TensorType {
    TensorType::new(shape.to_vec(), DType::F16)
}

/// Weights are `scale * u` with `u` a deterministic value in about `[-1, 1]`, so a K-long dot product stays well
/// inside FP16 at the real widths.
fn weight(name: &'static str, shape: &[usize], seed: usize, scale: f32) -> vpu_common::Const {
    constant(name, shape, move |i| {
        let h = (i.wrapping_mul(2_654_435_761).wrapping_add(seed * 40_503)) % 2003;
        scale * (h as f32 / 1001.0 - 1.0)
    })
}

/// The MLP of the block with its residual: `x + down(gate(x) * up(x))`. The SiLU the model puts on `gate`
/// (a divide) has no opset op, so it is absent here and listed in the refusal table.
fn mlp_fragment() -> Fixture {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", f16(&[TOKENS, HIDDEN]))
        .unwrap();
    let gate_w = b.constant("gate", f16(&[INTER, HIDDEN])).unwrap();
    let up_w = b.constant("up", f16(&[INTER, HIDDEN])).unwrap();
    let down_w = b.constant("down", f16(&[HIDDEN, INTER])).unwrap();
    let gate = linear(&b, x, gate_w);
    let up = linear(&b, x, up_w);
    let hidden = b.binary(BinOp::Mul, gate, up).unwrap();
    let down = linear(&b, hidden, down_w);
    let out = b.binary(BinOp::Add, down, x).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[
        weight("gate", &[INTER, HIDDEN], 1, 0.05),
        weight("up", &[INTER, HIDDEN], 2, 0.05),
        weight("down", &[HIDDEN, INTER], 3, 0.02),
    ]);
    let steps = (0..3)
        .map(|s| {
            x_step("x", &[TOKENS, HIDDEN], move |i| {
                ((i * (s + 3)) % 17) as f32 / 8.0 - 1.0
            })
        })
        .collect();
    Fixture {
        name: "qwen2_mlp",
        graph,
        store,
        steps,
    }
}

/// The attention core with its Q/K/V biases, grouped-query expansion of K and V, the score scale as a constant,
/// the softmax numerator `exp(s)` (the max subtraction and the divide have no opset op) and the output
/// projection: `o(exp(q k^T / 8) v)`.
fn attention_fragment() -> Fixture {
    let b = Builder::for_tests();
    let (q_width, kv_width) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
    let group = HEADS / KV_HEADS;
    let x = b
        .slot_named(Slot::Activation, "x", f16(&[TOKENS, HIDDEN]))
        .unwrap();
    let project = |name: &str, width: usize| {
        let w = b
            .constant(&format!("{name}_w"), f16(&[width, HIDDEN]))
            .unwrap();
        let bias = b.constant(&format!("{name}_b"), f16(&[1, width])).unwrap();
        let bias = b.broadcast(bias, vec![TOKENS, width]).unwrap();
        b.binary(BinOp::Add, linear(&b, x, w), bias).unwrap()
    };
    let q = project("q", q_width);
    let k = project("k", kv_width);
    let v = project("v", kv_width);
    // [tokens, heads * d] -> [1, heads, tokens, d]
    let to_heads = |t: Traced, heads: usize| {
        let t = b.reshape(t, vec![1, TOKENS, heads, HEAD_DIM]).unwrap();
        b.transpose(t, vec![0, 2, 1, 3]).unwrap()
    };
    // [1, kv, tokens, d] -> [1, heads, tokens, d]: each key/value head serves `group` query heads.
    let expand = |t: Traced| {
        let t = b
            .reshape(t, vec![1, KV_HEADS, 1, TOKENS, HEAD_DIM])
            .unwrap();
        let t = b
            .broadcast(t, vec![1, KV_HEADS, group, TOKENS, HEAD_DIM])
            .unwrap();
        b.reshape(t, vec![1, HEADS, TOKENS, HEAD_DIM]).unwrap()
    };
    let q = to_heads(q, HEADS);
    let k = expand(to_heads(k, KV_HEADS));
    let v = expand(to_heads(v, KV_HEADS));
    let kt = b.transpose(k, vec![0, 1, 3, 2]).unwrap();
    let scores = b.matmul(q, kt).unwrap();
    let scale = b.constant("scale", f16(&[1, 1, 1, 1])).unwrap();
    let scale = b.broadcast(scale, vec![1, HEADS, TOKENS, TOKENS]).unwrap();
    let scores = b.binary(BinOp::Mul, scores, scale).unwrap();
    let weights = b.unary(UnOp::Exp, scores).unwrap();
    let mixed = b.matmul(weights, v).unwrap();
    let mixed = b.transpose(mixed, vec![0, 2, 1, 3]).unwrap();
    let mixed = b.reshape(mixed, vec![TOKENS, q_width]).unwrap();
    let o_w = b.constant("o_w", f16(&[HIDDEN, q_width])).unwrap();
    let out = linear(&b, mixed, o_w);
    let graph = b.finish(out).unwrap();
    let store = store(&[
        weight("q_w", &[q_width, HIDDEN], 4, 0.05),
        weight("q_b", &[1, q_width], 5, 0.1),
        weight("k_w", &[kv_width, HIDDEN], 6, 0.05),
        weight("k_b", &[1, kv_width], 7, 0.1),
        weight("v_w", &[kv_width, HIDDEN], 8, 0.05),
        weight("v_b", &[1, kv_width], 9, 0.1),
        weight("o_w", &[HIDDEN, q_width], 10, 0.02),
        constant("scale", &[1, 1, 1, 1], |_| 0.125),
    ]);
    let steps = (0..3)
        .map(|s| {
            x_step("x", &[TOKENS, HIDDEN], move |i| {
                ((i * (s + 5)) % 19) as f32 / 9.0 - 1.0
            })
        })
        .collect();
    Fixture {
        name: "qwen2_attention_core",
        graph,
        store,
        steps,
    }
}

/// Elements of `got` further from `want` than `RELATIVE_TOLERANCE` of the oracle tensor's largest magnitude.
/// The output is a K = 896 to 4864 long dot product accumulated in FP16 storage between ops, so the bound is
/// relative to the tensor's scale, not per element.
const RELATIVE_TOLERANCE: f32 = 2.0e-2;

/// The fragments take only dense activation slots, so a step's inputs are its raw views.
fn step_inputs(step: &[poot_eval::test_support::StepFixture]) -> poot_executor::StepInputs<'_> {
    let mut inputs = poot_executor::StepInputs::new();
    for v in step {
        inputs.push(v.key.clone(), v.tensor.shape(), v.tensor.view());
    }
    inputs
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

/// Run `fixture` on the NPU against the oracle, every step; then time `REPLAYS` steps on the NPU and on the CPU.
fn run_fragment(fixture: &Fixture) {
    const REPLAYS: usize = 20;
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let exec = &mut npu.engine;
    let exe = poot_executor::Executor::load_weights(
        exec,
        std::sync::Arc::new(fixture.store.clone()),
        fixture_weights(fixture),
    )
    .unwrap();
    let compile_started = Instant::now();
    let program = staged_on(&*exec, fixture, target);
    let entry = poot_executor::Executor::add_entry(exec, exe, &program).unwrap();
    let compile = compile_started.elapsed();
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let mut worst = 0.0f32;
    let mut npu_times = Vec::new();
    for (i, step) in fixture.steps.iter().enumerate() {
        let inputs = step_inputs(step);
        let started = Instant::now();
        let got = poot_executor::Executor::step(exec, exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        npu_times.push(started.elapsed());
        let want = oracle(step);
        assert_eq!(
            got.shape(),
            want.shape(),
            "{} step {i}: shape",
            fixture.name
        );
        let (got, want) = (got.to_f32().unwrap(), want.to_f32().unwrap());
        let scale = want.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        for (j, (&g, &w)) in got.iter().zip(&*want).enumerate() {
            let error = (g - w).abs();
            worst = worst.max(error / scale);
            assert!(
                error <= RELATIVE_TOLERANCE * scale && !g.is_nan(),
                "{} step {i} element {j}: the NPU computed {g}, the oracle {w} (scale {scale})",
                fixture.name
            );
        }
    }
    // Steady-state replays: the same recorded graph over the first step's inputs.
    let inputs = step_inputs(&fixture.steps[0]);
    let mut replays = Vec::new();
    for _ in 0..REPLAYS {
        let started = Instant::now();
        poot_executor::Executor::step(exec, exe, entry, &inputs, &mut NoSync)
            .unwrap()
            .to_host()
            .unwrap();
        replays.push(started.elapsed());
    }
    let mut cpu = Vec::new();
    for _ in 0..REPLAYS.min(5) {
        let mut cpu_oracle = oracle_for(&fixture.graph, &fixture.store);
        let started = Instant::now();
        cpu_oracle(&fixture.steps[0]);
        cpu.push(started.elapsed());
    }
    println!(
        "RECEIPT {}: vendor compile {compile:?}; worst error {:.3e} of the output scale (bound {RELATIVE_TOLERANCE}); \
         NPU replay median {:?} (host wall incl. readback, {REPLAYS} replays), CPU oracle median {:?} \
         (host wall of the poot-eval reference evaluator, including its setup: a correctness oracle, not an optimized CPU baseline)",
        fixture.name,
        worst,
        median(replays),
        median(cpu),
    );
    poot_executor::Executor::unload(exec, exe).unwrap();
}

#[test]
fn the_mlp_of_the_block_matches_the_oracle_on_the_npu() {
    run_fragment(&mlp_fragment());
}

#[test]
fn the_attention_core_of_the_block_matches_the_oracle_on_the_npu() {
    run_fragment(&attention_fragment());
}

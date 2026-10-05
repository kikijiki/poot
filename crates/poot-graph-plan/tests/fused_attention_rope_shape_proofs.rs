//! Card 496 (R467-001, R467-002): the flash-attention and rope matchers fuse only operand shapes the
//! fused op is defined for. Each case checks the optimized graph against the CPU oracle on the raw
//! graph, element by element and bit for bit, and plans every optimized equation on each backend.
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::analysis::dispatch_count;
use poot_graph_ir::builder::Builder;
use poot_graph_ir::op::{BinOp, OpKind, UnOp};
use poot_graph_ir::ops::{attention_masked, attention_prefill};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Graph, ValueId};
use poot_target::AmdArch;
use poot_target::Backend;
use poot_tensor::HostTensor;
use poot_test_util::graph_fixtures::plan_eqn;

use poot_graph_plan::passes_without_target as optimize;

/// Deterministic inputs in `[-0.5, 0.5)`, one stream per input id.
fn inputs(g: &Graph) -> HashMap<ValueId, Value> {
    g.inputs
        .iter()
        .map(|&id| {
            let shape = g.aval(id).shape.clone();
            let mut s = (id as u64 + 7).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let data = (0..shape.iter().product::<usize>())
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect();
            (id, Value::from(HostTensor::f32(shape, data)))
        })
        .collect()
}

/// The optimized graph validates and its oracle result equals the raw graph's in every element, bit for
/// bit (a NaN never equals).
fn assert_optimize_preserves(name: &str, raw: &Graph, optimized: &Graph) {
    optimized
        .validate()
        .unwrap_or_else(|e| panic!("{name}: optimized graph invalid: {e}"));
    let ins = inputs(raw);
    let want = eval(raw, &ins, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap_or_else(|e| panic!("{name}: raw eval: {e}"))
        .output
        .into_host()
        .unwrap_or_else(|e| panic!("{name}: raw output not dense: {e}"));
    let got = eval(optimized, &ins, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap_or_else(|e| panic!("{name}: optimized eval: {e}"))
        .output
        .into_host()
        .unwrap_or_else(|e| panic!("{name}: optimized output not dense: {e}"));
    assert_eq!(got.shape(), want.shape(), "{name}: output shape");
    for (i, (g, w)) in got
        .as_f32()
        .unwrap()
        .iter()
        .zip(want.as_f32().unwrap().iter())
        .enumerate()
    {
        assert!(
            !w.is_nan() && g.to_bits() == w.to_bits(),
            "{name}: element {i}: optimized {g} != raw {w}"
        );
    }
}

fn flash_ops(g: &Graph) -> Vec<String> {
    g.eqns
        .iter()
        .filter(|e| {
            matches!(
                e.op,
                OpKind::FlashAttentionDecode { .. } | OpKind::FlashAttentionPrefill { .. }
            )
        })
        .map(|e| e.op.name())
        .collect()
}

/// `plan_eqn` over every equation, on every backend: zero refusals.
fn assert_plans_everywhere(name: &str, g: &Graph) {
    let backends = [
        ("SpirvVulkan", Backend::SpirvVulkan),
        ("AmdGcn", Backend::AmdGcn(AmdArch::gfx1151())),
        ("Nvptx", Backend::Nvptx),
    ];
    for (backend_name, backend) in backends {
        for (i, eqn) in g.eqns.iter().enumerate() {
            if let Err(e) = plan_eqn(
                g,
                eqn,
                backend,
                &poot_test_util::device_caps::default_caps_for(backend),
            ) {
                panic!(
                    "{name}: {backend_name} refuses eqn {i} ({}): {e}",
                    eqn.op.name()
                );
            }
        }
    }
}

/// A decode attention (`ops::attention_masked`) over constant operands of the given shapes.
fn decode_attention(q: [usize; 4], kv: [usize; 4], mask: [usize; 4], n_rep: usize) -> Graph {
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(q.to_vec()));
    let k = b.constant("k", TensorType::f32(kv.to_vec()));
    let v = b.constant("v", TensorType::f32(kv.to_vec()));
    let mask = b.constant("mask", TensorType::f32(mask.to_vec()));
    let out = attention_masked(&b, q, k, v, n_rep, 0.25, mask);
    b.finish(out)
}

/// A prefill attention (`ops::attention_prefill`) with `hkv` KV heads repeated `n_rep` times.
fn prefill_attention(hkv: usize, n_rep: usize, l: usize, d: usize) -> Graph {
    let b = Builder::new();
    let q = b.constant("q", TensorType::f32(vec![1, hkv * n_rep, l, d]));
    let k = b.constant("k", TensorType::f32(vec![1, hkv, l, d]));
    let v = b.constant("v", TensorType::f32(vec![1, hkv, l, d]));
    let mask = b.constant("mask", TensorType::f32(vec![1, 1, l, l]));
    let out = attention_prefill(&b, q, k, v, n_rep, 0.25, mask);
    b.finish(out)
}

/// SC-001 (R467-001, 467 probe p10): K/V with a size-1 head axis (MQA by broadcast) or a size-1 batch axis
/// (one KV shared by every batch row) is a valid decomposition that the fused decode op would index past.
/// It stays the decomposition: no flash op, the two attention MatMuls and the unfused dispatch count
/// remain, and the oracle result equals the raw graph's.
#[test]
fn broadcast_kv_attention_stays_decomposed_and_matches_the_oracle() {
    let cases = [
        (
            "MQA by broadcast",
            [1, 4, 1, 16],
            [1, 1, 8, 16],
            [1, 1, 1, 8],
        ),
        (
            "KV shared over batch",
            [2, 4, 1, 16],
            [1, 4, 8, 16],
            [2, 1, 1, 8],
        ),
    ];
    for (name, q, kv, mask) in cases {
        let raw = decode_attention(q, kv, mask, 1);
        let optimized = optimize(&raw);
        assert_eq!(flash_ops(&optimized), Vec::<String>::new(), "{name}: fused");
        let kinds: Vec<&str> = optimized
            .eqns
            .iter()
            .map(|e| match e.op {
                OpKind::Transpose { .. } => "transpose",
                OpKind::MatMul => "matmul",
                OpKind::Fused(_) => "fused",
                OpKind::FusedRow(_) => "fused_row",
                _ => "other",
            })
            .collect();
        // K^T, Q K^T, scale + mask, the softmax max/exp rows, P V.
        assert_eq!(
            kinds,
            [
                "transpose",
                "matmul",
                "fused",
                "fused_row",
                "fused_row",
                "matmul"
            ],
            "{name}: optimized op kinds"
        );
        assert_eq!(
            dispatch_count(&optimized),
            6,
            "{name}: optimized dispatches"
        );
        assert_optimize_preserves(name, &raw, &optimized);
    }
}

/// SC-003 (R467-002, 467 probe p04): a rotate-half chain whose per-head cos/sin broadcast a shared x up
/// (`x[1,1,S,D]`, `cos[1,H,S,D]`) optimizes to a graph that validates, equals the raw oracle, and plans on
/// every backend. Before the fix `rope_fusion` emitted a `Rope` typed as x (`[1,1,S,D]`) against the
/// declared `[1,H,S,D]`: `Graph::validate` refuses it, while the planner alone (`plan_eqn_analyzed`,
/// then named `plan_eqn`) accepts it and plans a rope body over x's shape launched over the larger
/// output.
#[test]
fn rope_chain_that_broadcasts_x_up_optimizes_to_a_plannable_graph() {
    let b = Builder::new();
    let (s, d) = (3, 8);
    let x = b.constant("x", TensorType::f32(vec![1, 1, s, d]));
    let cos = b.constant("cos", TensorType::f32(vec![1, 4, s, d]));
    let sin = b.constant("sin", TensorType::f32(vec![1, 4, s, d]));
    let x1 = b.slice(x, 3, 0, d / 2);
    let x2 = b.slice(x, 3, d / 2, d);
    let neg_x2 = b.unary(UnOp::Neg, x2);
    let rotate_half = b.concat(3, &[neg_x2, x1]);
    let xc = b.binary(BinOp::Mul, x, cos);
    let rs = b.binary(BinOp::Mul, rotate_half, sin);
    let out = b.binary(BinOp::Add, xc, rs);
    let raw = b.finish(out);
    let optimized = optimize(&raw);
    assert_plans_everywhere("rope broadcasting x", &optimized);
    assert_optimize_preserves("rope broadcasting x", &raw, &optimized);
}

/// SC-004: `optimize` over attention head layouts. The layouts the fused op is defined for (MHA, GQA and
/// MQA via `repeat_kv`, decode and prefill) fuse to one flash op; the broadcast layouts (MQA by broadcast,
/// KV shared over batch) stay decomposed. Every case validates and equals the raw oracle bit for bit.
#[test]
fn optimize_preserves_attention_over_head_layouts() {
    let decode = |hkv: usize, n_rep: usize, bsz: usize| {
        let hq = hkv * n_rep;
        decode_attention([bsz, hq, 1, 8], [bsz, hkv, 6, 8], [bsz, 1, 1, 6], n_rep)
    };
    let cases: [(&str, Graph, bool); 9] = [
        ("decode MHA", decode(4, 1, 1), true),
        ("decode GQA", decode(2, 2, 1), true),
        ("decode MQA via repeat_kv", decode(1, 4, 1), true),
        ("decode batched GQA", decode(2, 2, 2), true),
        ("prefill MHA", prefill_attention(4, 1, 5, 8), true),
        ("prefill GQA", prefill_attention(2, 2, 5, 8), true),
        (
            "prefill MQA via repeat_kv",
            prefill_attention(1, 4, 5, 8),
            true,
        ),
        (
            "decode MQA by broadcast",
            decode_attention([1, 4, 1, 8], [1, 1, 6, 8], [1, 1, 1, 6], 1),
            false,
        ),
        (
            "decode KV shared over batch",
            decode_attention([2, 4, 1, 8], [1, 4, 6, 8], [2, 1, 1, 6], 1),
            false,
        ),
    ];
    for (name, raw, fuses) in cases {
        let optimized = optimize(&raw);
        assert_eq!(
            flash_ops(&optimized).len(),
            usize::from(fuses),
            "{name}: flash ops {:?}",
            flash_ops(&optimized)
        );
        assert_optimize_preserves(name, &raw, &optimized);
    }
}

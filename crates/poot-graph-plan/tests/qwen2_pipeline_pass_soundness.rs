//! CSE/fuse preservation and `legalize`'s host-embed rewrite on a real qwen2 decomposition, moved
//! from `poot-eval/src/tests/qwen2_pipeline.rs` with the passes themselves (card 626) - poot-eval
//! must never depend on poot-graph-plan (its own architecture test); this crate already dev-depends
//! on poot-eval, so an integration test here can drive both.

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::{Graph, Slot, Storage, ValueId};
use poot_graph_plan::{cse, dce, fuse, legalize};
use poot_models::model::{LogitRows, Phase};
use poot_tensor::HostTensor;
use poot_test_util::assert_close_rel;
use std::collections::HashMap;

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

/// The tiny qwen2 the pass tests trace (through the registry): the same dimensions the decomposition
/// tests always used.
fn tiny_qwen2() -> poot_executor_parity::weight_map::MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, 2)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(16)
        .f32_model()
}

/// `model`'s one-token decode step over `cap` positions.
fn decode_graph(model: &poot_executor_parity::weight_map::MappedModel, cap: usize) -> Graph {
    plain(
        model
            .model
            .trace(Phase::Decode, step(1, 1, cap, LogitRows::Last))
            .unwrap(),
    )
}

/// Every input of `g` bound: the token and position slots as one decode step at `pos`, state and
/// weights filled by input index (weights scaled by 0.1).
fn bind_decode(g: &Graph, token: i32, pos: i32) -> HashMap<ValueId, HostTensor> {
    let mut inputs: HashMap<ValueId, HostTensor> = HashMap::new();
    for (i, &id) in g.inputs.iter().enumerate() {
        let aval = g.aval(id).clone();
        let t = match g.values[id].storage {
            Storage::Slot(Slot::Token) => HostTensor::i32(aval.shape, vec![token]),
            Storage::Slot(Slot::Pos) => HostTensor::i32(aval.shape, vec![pos]),
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Computed(c) => HostTensor::f32(c.shape(), c.values_f32()),
            Storage::Const | Storage::State => {
                let data: Vec<f32> = fill(aval.numel(), 100 + i as u64)
                    .iter()
                    .map(|v| v * 0.1)
                    .collect();
                HostTensor::f32(aval.shape, data)
            }
            Storage::Device => unreachable!(),
        };
        inputs.insert(id, t);
    }
    inputs
}

#[test]
fn cse_preserves_results() {
    // CSE only removes redundant identical computations, so eval(cse(g)) is bit-identical to eval(g).
    let pos = 3usize;
    let g = decode_graph(&tiny_qwen2(), 8);
    let c = cse(&g);
    assert!(c.eqns.len() < g.eqns.len());

    let inputs = bind_decode(&g, 5, pos as i32);
    let a = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let b = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&c, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    assert_eq!(
        a.as_f32().unwrap(),
        b.as_f32().unwrap(),
        "CSE changed the result"
    );
}

#[test]
fn fuse_preserves_qwen2_results_and_shrinks() {
    // G5 executor-equivalence oracle on a real decomposition: eval(fuse(g)) is bit-identical to eval(g) on the tiny qwen2
    // decode graph, with fewer eqns, and both fusion kinds fired: pointwise Fused regions (rope / residual chains) and
    // reduction-rooted FusedRow regions (each layer's RMSNorm + softmax).
    let pos = 3usize;
    let g = decode_graph(&tiny_qwen2(), 8);
    let fg = fuse(&g);
    fg.validate().expect("fused graph valid");
    assert!(
        fg.eqns.len() < g.eqns.len(),
        "fusion should drop eqns: {} -> {}",
        g.eqns.len(),
        fg.eqns.len()
    );
    let n_fused = fg
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::Fused(_)))
        .count();
    let n_row = fg
        .eqns
        .iter()
        .filter(|e| matches!(e.op, OpKind::FusedRow(_)))
        .count();
    assert!(n_fused > 0, "pointwise fusion should fire");
    assert!(
        n_row > 0,
        "reduction-rooted fusion should fire (RMSNorm + softmax)"
    );
    eprintln!(
        "qwen2 tiny fuse: {} eqns -> {} ({} Fused, {} FusedRow)",
        g.eqns.len(),
        fg.eqns.len(),
        n_fused,
        n_row
    );

    let inputs = bind_decode(&g, 5, pos as i32);
    let a = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&g, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    let b = {
        let values: HashMap<ValueId, Value> = inputs
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&fg, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };
    assert_eq!(
        a.as_f32().unwrap(),
        b.as_f32().unwrap(),
        "fusion changed the result"
    );
}

/// The vocabulary and width of [`tiny_legalize_model`].
const VOCAB: usize = 64;
const HIDDEN: usize = 8;

/// A tiny qwen2 whose embed table (`vocab * hidden * 4` bytes) is strictly the largest constant the
/// tracer declares - every projection weight fits `hidden * inter` or smaller - so a
/// [`poot_target::DeviceCaps::max_buffer_bytes`] between the two hosts the embed table and nothing else.
fn tiny_legalize_model() -> poot_executor_parity::weight_map::MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(HIDDEN, 8, 2)
        .heads(2, 1)
        .head_dim(4)
        .max_positions(8)
        .f32_model()
}

/// [`tiny_legalize_model`]'s step graph: `tokens` new tokens over `cap` positions, logits at every
/// position so the graph ends in the lm_head `MatMul` [`without_lm_head`] cuts at.
fn legalize_graph(phase: Phase, tokens: usize, cap: usize) -> Graph {
    plain(
        tiny_legalize_model()
            .model
            .trace(phase, step(1, tokens, cap, LogitRows::All))
            .unwrap(),
    )
}

/// A [`poot_target::DeviceCaps`] whose `max_buffer_bytes` sits strictly between the largest
/// non-embed constant [`tiny_legalize_model`]'s traced graphs declare (a projection weight, at most
/// `hidden * inter * 4` bytes) and the embed table (`vocab * hidden * 4` bytes: 2048), so `legalize`
/// hosts the embed gather and refuses nothing else in that fixture. Both tests run their graph through
/// [`without_lm_head`] first: lm_head has the same `vocab * hidden` element count as the embed table
/// (no smaller), so comparing final logits would need lm_head to fit under the same limit that must
/// exceed the embed table to trigger the rewrite, which no scalar limit can do for a tied dense model.
fn caps_hosting_only_the_tiny_embed() -> poot_target::DeviceCaps {
    poot_target::DeviceCaps {
        max_buffer_bytes: 512,
        lds_bytes: 64 * 1024,
        max_grid: [65_535, 65_535, 65_535],
        watchdog_budget: None,
        tensor_core: poot_target::TensorCoreSupport::None,
        known_miscompiles: poot_target::KnownMiscompiles::default(),
        compute_units: poot_target::STRIX_HALO_COMPUTE_UNITS,
        ..poot_target::DeviceCaps::rocm_default()
    }
}

/// [`tiny_legalize_model`]'s decode graph, stopped at the final RMSNorm (no lm_head matmul): the
/// dense embed gather the step always emits, with everything after it that
/// [`legalize`]'s rewrite cannot affect. This crate has no direct hook to truncate the tracer, so
/// this walks the graph back from the traced output to the last `Rmsnorm`-shaped value so lm_head's
/// `MatMul` against the embed-sized weight never enters either compared graph.
fn without_lm_head(mut g: Graph) -> Graph {
    use poot_graph_ir::Operand;
    // a decode or prefill step with all-position logits ends with `linear(rmsnorm_output, lm_head)`: a `MatMul`
    // (bias-free, since lm_head passes no bias) whose output is the graph's declared output. Retarget
    // the output at that `MatMul`'s own first operand (the pre-lm_head hidden state) and drop it from
    // `eqns` with ordinary DCE, so nothing downstream of it - including the embed-sized
    // `w.head` const - survives.
    let out_eqn = g
        .eqns
        .iter()
        .find(|e| e.out == g.output)
        .expect("the graph's declared output is produced by an equation");
    assert!(
        matches!(out_eqn.op, OpKind::MatMul),
        "expected the traced graph to end in the lm_head MatMul, found {:?}",
        out_eqn.op
    );
    let Operand::Value(hidden) = out_eqn.inputs[0] else {
        panic!("lm_head MatMul's first operand must be a value")
    };
    g.output = hidden;
    let g = dce(&g);

    // `dce` only prunes `eqns` ("the value table is preserved, dead entries are harmless" - true for
    // an ordinary unused intermediate, but `lm_head.weight` staying in `consts` here would still be an
    // embed-sized constant `legalize`'s refusal path sees). Prune the input tables to what the
    // DCE'd graph actually reads.
    let referenced: std::collections::HashSet<_> = g
        .eqns
        .iter()
        .flat_map(|e| {
            e.inputs.iter().filter_map(|&o| match o {
                Operand::Value(v) => Some(v),
                Operand::Lit(_) => None,
            })
        })
        .chain(std::iter::once(g.output))
        .chain(g.state.iter().flat_map(|&(a, b)| [a, b]))
        .collect();
    poot_graph_ir::Graph {
        inputs: g
            .inputs
            .iter()
            .copied()
            .filter(|id| referenced.contains(id))
            .collect(),
        consts: g
            .consts
            .iter()
            .copied()
            .filter(|id| referenced.contains(id))
            .collect(),
        slots: g
            .slots
            .iter()
            .copied()
            .filter(|(id, _)| referenced.contains(id))
            .collect(),
        ..g
    }
}

#[test]
fn legalize_hosts_the_decode_embed_gather_without_changing_hidden_state() {
    // Card 523a: `legalize`, given a target whose buffer limit is below the embed table, rewrites the
    // dense in-graph gather `trace_decode_kv_masked` always emits into a host-computed
    // `Slot::TokenEmbed` row. The two graphs (traced once, legalized only for the second run) must be
    // numerically equivalent: this was card 168's `host_embed=true`/`false` tracer-flag equivalence,
    // now a property of the one rewrite instead of two tracer branches. See [`without_lm_head`] for why
    // this compares the pre-lm_head hidden state, not final logits.
    use std::collections::HashMap as Map;

    let cap = 6;
    let pos = 3; // an arbitrary mid-cache decode step.
    let token: u32 = 11;
    let weight = |name: &str, shape: &[usize]| -> HostTensor {
        let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
            (acc ^ c as u64).wrapping_mul(1099511628211)
        });
        HostTensor::f32(
            shape.to_vec(),
            fill(shape.iter().product::<usize>().max(1), seed)
                .iter()
                .map(|v| v * 0.1)
                .collect(),
        )
    };
    let gd = without_lm_head(legalize_graph(Phase::Decode, 1, cap));
    let mut di: Map<ValueId, HostTensor> = Map::new();
    for &id in &gd.inputs {
        let m = gd.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                di.insert(id, HostTensor::i32(vec![1, 1], vec![token as i32]))
            }
            Storage::Slot(Slot::Pos) => di.insert(id, HostTensor::i32(vec![1, 1], vec![pos])),
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => di.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            Storage::Computed(computed) => {
                Some(HostTensor::f32(computed.shape(), computed.values_f32()))
            }
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gd.state {
        di.insert(si, HostTensor::zeros(gd.aval(si).shape.clone()));
    }
    let (hidden_dense, _) = {
        let values: HashMap<ValueId, Value> = di
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = eval(&gd, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (evaluation.output.into_host().unwrap(), state)
    };

    let gh = legalize(
        &gd,
        &caps_hosting_only_the_tiny_embed(),
        &poot_graph_plan::CompileLimits::STANDARD,
    )
    .expect("the embed table hosts; nothing else in this fixture is oversized");
    assert!(
        gh.consts
            .iter()
            .all(|&id| gh.meta(id).name.as_deref() != Some("w.embed")),
        "legalize must drop the oversized embed const, not merely refuse it"
    );
    let embed_w = weight("w.embed", &[VOCAB, HIDDEN]);
    let h = HIDDEN;
    let token_row =
        embed_w.as_f32().unwrap()[token as usize * h..(token as usize + 1) * h].to_vec();
    let mut hi: Map<ValueId, HostTensor> = Map::new();
    for &id in &gh.inputs {
        let m = gh.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                unreachable!("legalize must drop the Token slot once it hosts the gather")
            }
            Storage::Slot(Slot::Pos) => hi.insert(id, HostTensor::i32(vec![1, 1], vec![pos])),
            Storage::Slot(Slot::TokenEmbed) => {
                assert_eq!(m.aval.shape, vec![1, 1, h], "single-row TokenEmbed shape");
                hi.insert(id, HostTensor::f32(vec![1, 1, h], token_row.clone()))
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => hi.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            Storage::Computed(computed) => {
                Some(HostTensor::f32(computed.shape(), computed.values_f32()))
            }
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gh.state {
        hi.insert(si, HostTensor::zeros(gh.aval(si).shape.clone()));
    }
    let (hidden_host, _) = {
        let values: HashMap<ValueId, Value> = hi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        let evaluation = eval(&gh, &values, EvalOptions::new(EvalBudget::UNBOUNDED)).unwrap();
        let state = evaluation
            .state
            .into_iter()
            .map(Value::into_host)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (evaluation.output.into_host().unwrap(), state)
    };

    assert_close_rel(
        hidden_host.as_f32().unwrap(),
        hidden_dense.as_f32().unwrap(),
        1e-5,
    );
}

#[test]
fn legalize_hosts_the_prefill_embed_gather_without_changing_hidden_state() {
    // Same equivalence as `legalize_hosts_the_decode_embed_gather_without_changing_hidden_state`, for
    // the multi-row prefill step (`Phase::Prefill`, L>1): the hosted `Slot::TokenEmbed` shape is
    // `[l,hidden]` (one row per prompt position).
    use std::collections::HashMap as Map;

    let tokens: Vec<u32> = vec![5, 9, 2, 14, 7];
    let l = tokens.len();
    let weight = |name: &str, shape: &[usize]| -> HostTensor {
        let seed: u64 = name.bytes().fold(1469598103934665603u64, |acc, c| {
            (acc ^ c as u64).wrapping_mul(1099511628211)
        });
        HostTensor::f32(
            shape.to_vec(),
            fill(shape.iter().product::<usize>().max(1), seed)
                .iter()
                .map(|v| v * 0.1)
                .collect(),
        )
    };

    let gd = without_lm_head(legalize_graph(Phase::Prefill, l, l));
    let mut di: Map<ValueId, HostTensor> = Map::new();
    for &id in &gd.inputs {
        let m = gd.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => di.insert(
                id,
                HostTensor::i32(vec![1, l], tokens.iter().map(|&t| t as i32).collect()),
            ),
            Storage::Slot(Slot::Pos) => {
                di.insert(id, HostTensor::i32(vec![1, l], (0..l as i32).collect()))
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => di.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            Storage::Computed(computed) => {
                Some(HostTensor::f32(computed.shape(), computed.values_f32()))
            }
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gd.state {
        di.insert(si, HostTensor::zeros(gd.aval(si).shape.clone()));
    }
    let hidden_dense = {
        let values: HashMap<ValueId, Value> = di
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&gd, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };

    let gh = legalize(
        &gd,
        &caps_hosting_only_the_tiny_embed(),
        &poot_graph_plan::CompileLimits::STANDARD,
    )
    .expect("the embed table hosts; nothing else in this fixture is oversized");
    let embed_w = weight("w.embed", &[VOCAB, HIDDEN]);
    let h = HIDDEN;
    let mut rows = Vec::with_capacity(l * h);
    for &t in &tokens {
        rows.extend_from_slice(&embed_w.as_f32().unwrap()[t as usize * h..(t as usize + 1) * h]);
    }
    let mut hi: Map<ValueId, HostTensor> = Map::new();
    for &id in &gh.inputs {
        let m = gh.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                unreachable!("legalize must drop the Token slot once it hosts the gather")
            }
            Storage::Slot(Slot::TokenEmbed) => {
                assert_eq!(m.aval.shape, vec![1, l, h], "prefill TokenEmbed shape");
                hi.insert(id, HostTensor::f32(vec![1, l, h], rows.clone()))
            }
            Storage::Slot(Slot::Pos) => {
                hi.insert(id, HostTensor::i32(vec![1, l], (0..l as i32).collect()))
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => hi.insert(id, weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            Storage::Computed(computed) => {
                Some(HostTensor::f32(computed.shape(), computed.values_f32()))
            }
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gh.state {
        hi.insert(si, HostTensor::zeros(gh.aval(si).shape.clone()));
    }
    let hidden_host = {
        let values: HashMap<ValueId, Value> = hi
            .iter()
            .map(|(&id, tensor)| (id, Value::from(tensor.clone())))
            .collect();
        eval(&gh, &values, EvalOptions::new(EvalBudget::UNBOUNDED))
            .unwrap()
            .output
            .into_host()
            .unwrap()
    };

    assert_close_rel(
        hidden_host.as_f32().unwrap(),
        hidden_dense.as_f32().unwrap(),
        1e-5,
    );
}

//! Card 523a: `legalize`'s host-embed rewrite, on the real (not
//! `Builder`-hand-assembled) qwen2 architecture, driven through `poot_graph_plan::compile` - the real
//! graph-to-program entry point - instead of a bare `legalize` call. `compile` also runs
//! canonicalize/cse/rope-fusion/flash-attention/fuse and planning after legalize, so this confirms the
//! host rewrite survives everything the rest of the pipeline does to it. `poot-eval` cannot host these
//! tests itself: it must never depend on `poot-graph-plan` even for tests (its own
//! `poot_eval_does_not_depend_on_graph_plan` guards this), so they live here, where `compile` itself is
//! defined and both `poot-eval` (the CPU oracle) and `poot-models` (the qwen2 tracer) are already
//! dev-dependencies.

use std::collections::HashMap as Map;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::{Slot, Storage, ValueId};
use poot_graph_plan::{CompileOptions, FusionPolicy, Submission, Target, compile};
use poot_models::model::{LogitRows, Phase};
use poot_target::Backend;
use poot_tensor::HostTensor;

/// deterministic pseudo-random fill in [-1, 1), no rng dependency.
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

/// The vocabulary and width of [`tiny_legalize_model`].
const VOCAB: usize = 64;
const HIDDEN: usize = 8;

/// A tiny qwen2 (through the registry) whose embed table (`vocab * hidden * 4` bytes) is strictly the
/// largest constant the tracer declares - every projection weight fits `hidden * inter` or smaller -
/// so a [`poot_target::DeviceCaps::max_buffer_bytes`] between the two hosts the embed table and nothing
/// else.
fn tiny_legalize_model() -> poot_executor_parity::weight_map::MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(HIDDEN, 8, 2)
        .heads(2, 1)
        .head_dim(4)
        .max_positions(8)
        .f32_model()
}

/// `tiny_legalize_model`'s step graph for `phase`: `tokens` new tokens over `cap` positions, logits
/// at every position so the graph ends in the lm_head `MatMul` [`without_lm_head`] cuts at.
fn trace(phase: Phase, tokens: usize, cap: usize) -> poot_graph_ir::Graph {
    plain(
        tiny_legalize_model()
            .model
            .trace(phase, step(1, tokens, cap, LogitRows::All))
            .unwrap(),
    )
}

/// A [`poot_target::DeviceCaps`] whose `max_buffer_bytes` sits strictly between the largest non-embed
/// constant [`tiny_legalize_cfg`]'s traced graphs declare (a projection weight, at most
/// `hidden * inter * 4` bytes) and the embed table (`vocab * hidden * 4` bytes: 2048), so `legalize`
/// hosts the embed gather and refuses nothing else in that fixture. Both tests run their graph through
/// [`without_lm_head`] first: lm_head has the same `vocab * hidden` element count as the embed table (no
/// smaller), so comparing final logits would need lm_head to fit under the same limit that must exceed
/// the embed table to trigger the rewrite, which no scalar limit can do for a tied dense model.
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

/// [`tiny_legalize_cfg`]'s decode/prefill graph, stopped at the final RMSNorm (no lm_head matmul): the
/// dense embed gather always emits, with everything after it that `legalize`'s rewrite cannot affect.
fn without_lm_head(mut g: poot_graph_ir::Graph) -> poot_graph_ir::Graph {
    use poot_graph_ir::{OpKind, Operand};
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
    let g = poot_graph_plan::dce(&g);

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
fn compile_hosts_the_decode_embed_gather_without_changing_hidden_state() {
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
    let gd = without_lm_head(trace(Phase::Decode, 1, cap));
    let mut di: Map<ValueId, Value> = Map::new();
    for &id in &gd.inputs {
        let m = gd.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => di.insert(
                id,
                Value::from(HostTensor::i32(vec![1, 1], vec![token as i32])),
            ),
            Storage::Slot(Slot::Pos) => {
                di.insert(id, Value::from(HostTensor::i32(vec![1, 1], vec![pos])))
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => di.insert(
                id,
                Value::from(weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            ),
            Storage::Computed(computed) => Some(Value::from(HostTensor::f32(
                computed.shape(),
                computed.values_f32(),
            ))),
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gd.state {
        di.insert(
            si,
            Value::from(HostTensor::zeros(gd.aval(si).shape.clone())),
        );
    }
    let hidden_dense = eval(&gd, &di, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let target = Target {
        backend: Backend::SpirvVulkan,
        caps: caps_hosting_only_the_tiny_embed(),
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let program = compile(&gd, &target, &options)
        .expect("the embed table hosts; nothing else in this fixture is oversized");
    let gh = program.graph();
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
    let mut hi: Map<ValueId, Value> = Map::new();
    for &id in &gh.inputs {
        let m = gh.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                unreachable!("legalize must drop the Token slot once it hosts the gather")
            }
            Storage::Slot(Slot::Pos) => {
                hi.insert(id, Value::from(HostTensor::i32(vec![1, 1], vec![pos])))
            }
            Storage::Slot(Slot::TokenEmbed) => {
                assert_eq!(m.aval.shape, vec![1, 1, h], "single-row TokenEmbed shape");
                hi.insert(
                    id,
                    Value::from(HostTensor::f32(vec![1, 1, h], token_row.clone())),
                )
            }
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => hi.insert(
                id,
                Value::from(weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            ),
            Storage::Computed(computed) => Some(Value::from(HostTensor::f32(
                computed.shape(),
                computed.values_f32(),
            ))),
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gh.state {
        hi.insert(
            si,
            Value::from(HostTensor::zeros(gh.aval(si).shape.clone())),
        );
    }
    let hidden_host = eval(gh, &hi, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    poot_test_util::assert_close_rel(
        hidden_host.as_f32().unwrap(),
        hidden_dense.as_f32().unwrap(),
        1e-5,
    );
}

/// Same as [`compile_hosts_the_decode_embed_gather_without_changing_hidden_state`], for the multi-row
/// prefill step (`Phase::Prefill`, L>1) - the hosted `Slot::TokenEmbed` shape is `[l,hidden]` (one row
/// per prompt position), driven through `compile`.
#[test]
fn compile_hosts_the_prefill_embed_gather_without_changing_hidden_state() {
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

    let gd = without_lm_head(trace(Phase::Prefill, l, l));
    let mut di: Map<ValueId, Value> = Map::new();
    for &id in &gd.inputs {
        let m = gd.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => di.insert(
                id,
                Value::from(HostTensor::i32(
                    vec![1, l],
                    tokens.iter().map(|&t| t as i32).collect(),
                )),
            ),
            Storage::Slot(Slot::Pos) => di.insert(
                id,
                Value::from(HostTensor::i32(vec![1, l], (0..l as i32).collect())),
            ),
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => di.insert(
                id,
                Value::from(weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            ),
            Storage::Computed(computed) => Some(Value::from(HostTensor::f32(
                computed.shape(),
                computed.values_f32(),
            ))),
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gd.state {
        di.insert(
            si,
            Value::from(HostTensor::zeros(gd.aval(si).shape.clone())),
        );
    }
    let hidden_dense = eval(&gd, &di, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let target = Target {
        backend: Backend::SpirvVulkan,
        caps: caps_hosting_only_the_tiny_embed(),
    };
    let options = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: poot_graph_plan::CompileLimits::STANDARD,
    };
    let program = compile(&gd, &target, &options)
        .expect("the embed table hosts; nothing else in this fixture is oversized");
    let gh = program.graph();
    let embed_w = weight("w.embed", &[VOCAB, HIDDEN]);
    let h = HIDDEN;
    let mut rows = Vec::with_capacity(l * h);
    for &t in &tokens {
        rows.extend_from_slice(&embed_w.as_f32().unwrap()[t as usize * h..(t as usize + 1) * h]);
    }
    let mut hi: Map<ValueId, Value> = Map::new();
    for &id in &gh.inputs {
        let m = gh.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => {
                unreachable!("legalize must drop the Token slot once it hosts the gather")
            }
            Storage::Slot(Slot::TokenEmbed) => {
                assert_eq!(m.aval.shape, vec![1, l, h], "prefill TokenEmbed shape");
                hi.insert(
                    id,
                    Value::from(HostTensor::f32(vec![1, l, h], rows.clone())),
                )
            }
            Storage::Slot(Slot::Pos) => hi.insert(
                id,
                Value::from(HostTensor::i32(vec![1, l], (0..l as i32).collect())),
            ),
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => hi.insert(
                id,
                Value::from(weight(m.name.as_deref().unwrap(), &m.aval.shape)),
            ),
            Storage::Computed(computed) => Some(Value::from(HostTensor::f32(
                computed.shape(),
                computed.values_f32(),
            ))),
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for &(si, _) in &gh.state {
        hi.insert(
            si,
            Value::from(HostTensor::zeros(gh.aval(si).shape.clone())),
        );
    }
    let hidden_host = eval(gh, &hi, EvalOptions::new(EvalBudget::UNBOUNDED))
        .unwrap()
        .output
        .into_host()
        .unwrap();

    poot_test_util::assert_close_rel(
        hidden_host.as_f32().unwrap(),
        hidden_dense.as_f32().unwrap(),
        1e-5,
    );
}

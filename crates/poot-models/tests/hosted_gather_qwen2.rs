//! POOT-1423: the hosted embed gather, on the real (not `Builder`-hand-assembled) qwen2 architecture, driven
//! through `poot_graph_plan::compile_staged` - the real graph-to-program entry point. The planner hosts a
//! table only the embed gather reads whatever the target's buffer limit is, and `compile` then runs
//! canonicalize/cse/rope-fusion/flash-attention/fuse and planning on the stage, so this confirms the hosted
//! rewrite survives everything the rest of the pipeline does to it. `poot-eval` cannot host these tests
//! itself: it must never depend on `poot-graph-plan` even for tests (its own
//! `poot_eval_does_not_depend_on_graph_plan` guards this), so they live here, where both `poot-eval` (the CPU
//! oracle) and `poot-models` (the qwen2 tracer) are dev-dependencies.

use std::collections::HashMap as Map;

use poot_eval::test_support::stage_chain::chain;
use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{Slot, SlotKey, Storage, ValueId};
use poot_graph_plan::multi_device::topology::DeviceId;
use poot_graph_plan::staged::{
    DevicePlacement, ExpertPlacement, Partition, TargetSet, compile_staged,
};
use poot_graph_plan::{CompileLimits, CompileOptions, IndexSource, Target};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::test_support::dense::{Dense, Family, step};
use poot_target::Backend;
use poot_tensor::HostTensor;

#[path = "support/hosted.rs"]
mod hosted;
use hosted::hosted_rows;

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

/// The vocabulary and width of [`tiny_model`].
const VOCAB: usize = 64;
const HIDDEN: usize = 8;

/// A tiny qwen2 (through the registry) with an untied embed table, which only the embed gather reads once the
/// graph stops before the head.
fn tiny_model() -> poot_models::test_support::weight_map::MappedModel {
    Dense::new(Family::Qwen2)
        .vocab(VOCAB)
        .dims(HIDDEN, 8, 2)
        .heads(2, 1)
        .head_dim(4)
        .max_positions(8)
        .f32_model()
}

/// `tiny_model`'s step graph: `tokens` new tokens over `cap` positions, logits
/// at every position so the graph ends in the lm_head `MatMul` [`without_lm_head`] cuts at.
fn trace(tokens: usize, cap: usize) -> poot_graph_ir::Graph {
    tiny_model()
        .model
        .trace(
            &StepSpec::step(step(1, tokens, cap, LogitRows::All)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap()
}

/// The decode/prefill graph, stopped at the final RMSNorm (no lm_head matmul): the dense embed gather always
/// emits, so the table has no reader but the gather.
fn without_lm_head(g: poot_graph_ir::Graph) -> poot_graph_ir::Graph {
    use poot_graph_ir::{OpKind, Operand};
    let out_eqn = g
        .eqns()
        .iter()
        .find(|e| e.out == g.result())
        .expect("the graph's declared output is produced by an equation");
    assert!(
        matches!(out_eqn.op, OpKind::MatMul { .. }),
        "expected the traced graph to end in the lm_head MatMul, found {:?}",
        out_eqn.op
    );
    let Operand::Value(hidden) = out_eqn.inputs[0] else {
        panic!("lm_head MatMul's first operand must be a value")
    };
    let g =
        poot_graph_ir::test_support::graph_fixtures::derive(&g, |draft| draft.set_result(hidden))
            .unwrap();
    let g = poot_graph_plan::dce(&g).unwrap();

    let referenced: std::collections::HashSet<_> = g
        .eqns()
        .iter()
        .flat_map(|e| {
            e.inputs.iter().filter_map(|&o| match o {
                Operand::Value(v) => Some(v),
                Operand::Lit(_) => None,
            })
        })
        .chain(std::iter::once(g.result()))
        .chain(g.state().flat_map(|(a, b)| [a, b]))
        .collect();
    let mut pruned = g.edit();
    pruned.inputs_mut().retain(|id| referenced.contains(id));
    pruned.consts_mut().retain(|id| referenced.contains(id));
    pruned.slots_mut().retain(|(id, _)| referenced.contains(id));
    pruned.commit().unwrap()
}

/// The seeded value of weight `name`, the same on every call so the unhosted and the hosted run read one table.
fn weight(name: &str, shape: &[usize]) -> HostTensor {
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
}

/// `tokens` at consecutive positions from `pos`, over a `cap`-position cache, planned for Vulkan through
/// `compile_staged` and evaluated stage by stage on the oracle with the rows the host would copy, against the
/// unhosted traced graph evaluated whole. Returns the number of hosted gathers the program has.
fn hosted_hidden_equals_unhosted(tokens: &[i32], pos: i32, cap: usize) -> usize {
    let l = tokens.len();
    let gd = without_lm_head(trace(l, cap));
    let mut di: Map<ValueId, Value> = Map::new();
    for &id in gd.inputs() {
        let m = gd.meta(id);
        match m.storage {
            Storage::Slot(Slot::Token) => di.insert(
                id,
                Value::from(HostTensor::i32(vec![1, l], tokens.to_vec())),
            ),
            Storage::Slot(Slot::Pos) => di.insert(
                id,
                Value::from(HostTensor::i32(
                    vec![1, l],
                    (0..l as i32).map(|i| pos + i).collect(),
                )),
            ),
            Storage::Slot(other) => unreachable!("unexpected slot {other:?}"),
            Storage::Const => di.insert(id, Value::from(weight(&m.name().unwrap(), &m.aval.shape))),
            Storage::Computed(computed) => Some(Value::from(computed.tensor())),
            Storage::State => None,
            Storage::Device => unreachable!(),
        };
    }
    for (si, _) in gd.state() {
        di.insert(
            si,
            Value::from(HostTensor::zeros_of(
                gd.aval(si).dtype,
                gd.aval(si).shape.clone(),
            )),
        );
    }
    let hidden_dense = eval(&gd, &di, EvalOptions::default())
        .unwrap()
        .output
        .into_host()
        .unwrap();

    let target = Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: poot_test_util::device_caps::default_caps_for(Backend::SpirvVulkan),
    };
    let program = compile_staged(
        &gd,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(CompileLimits::STANDARD)
            .with_matmul_precision(poot_graph_ir::MatmulPrecision::Exact),
    )
    .expect("the embed table hosts; nothing else in this fixture is oversized");
    let stages: Vec<&poot_graph_ir::Graph> = program
        .stages()
        .map(|(_, _, stage)| stage.graph())
        .collect();
    assert_eq!(stages.len(), 1, "a step-indexed gather adds no cut");
    let gh = stages[0];
    assert!(
        gh.consts()
            .iter()
            .all(|&id| gh.meta(id).name().as_deref() != Some("w.embed")),
        "the planner must drop the embed table from the graph, not merely refuse it"
    );
    assert!(
        gh.slots().iter().all(|&(_, slot)| slot != Slot::Token),
        "the token slot is dropped once the gather is hosted"
    );
    let [site] = program.hosted() else {
        panic!("one hosted gather, got {:?}", program.hosted())
    };
    assert_eq!(
        site.index,
        IndexSource::Step(SlotKey::new(Slot::Token, None))
    );
    assert_eq!(site.ids, l);
    let rows = &site.rows[0];
    assert_eq!(rows.source, "w.embed");
    assert_eq!(rows.aval.shape, vec![1, l, HIDDEN]);

    let chained = chain(&gd, &stages, &di, &mut hosted_rows(&program, &gd, &di));
    let hidden_host = chained.output.into_host().unwrap();
    poot_test_util::assert_close_rel(
        hidden_host.as_f32().unwrap(),
        hidden_dense.as_f32().unwrap(),
        1e-5,
    );
    program.hosted().len()
}

/// A decode step (one token at a mid-cache position): the hosted program's hidden state equals the unhosted
/// graph's. Mutation: in `hosted_gather`, bind the rows of `id + 1`; the hidden state differs.
#[test]
fn compile_staged_hosts_the_decode_embed_gather_without_changing_hidden_state() {
    assert_eq!(hosted_hidden_equals_unhosted(&[11], 3, 6), 1);
}

/// The multi-row prefill step: the hosted rows are `[1, l, hidden]`, one row per prompt position.
#[test]
fn compile_staged_hosts_the_prefill_embed_gather_without_changing_hidden_state() {
    assert_eq!(hosted_hidden_equals_unhosted(&[5, 9, 2, 14, 7], 0, 5), 1);
}

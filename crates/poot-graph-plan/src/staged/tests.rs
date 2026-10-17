//! POOT-745 at the compile level: what `compile_staged` produces for a pipeline, checked on the CPU
//! oracle (ADR-0101 tier 1: a cut graph chained stage by stage equals the uncut graph bit for bit).
//! The executor and the device lanes check the same programs run.

use crate::production_plan::Submission;
use std::collections::HashMap;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{
    BinOp, Builder, Graph, Scalar, Slot, SlotKey, StateRole, Storage, TensorType, UnOp,
    ValidationId, ValueId,
};
use poot_target::Backend;
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_caps::default_caps_for;

use super::*;
use poot_eval::test_support::stage_chain::{bind_decode, bits, chain};
use poot_test_util::fill;

use crate::CompileLimits;

fn target() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: default_caps_for(Backend::SpirvVulkan),
    }
}

fn options() -> CompileOptions {
    CompileOptions::new(Submission::Replay, CompileLimits::STANDARD)
}

fn partition(devices: DevicePlacement) -> Partition {
    Partition {
        experts: ExpertPlacement::AllResident,
        devices,
    }
}

fn pipeline(after_layers: &[usize]) -> DevicePlacement {
    DevicePlacement::Pipeline {
        after_layers: after_layers.to_vec(),
        devices: vec![DeviceId(0); after_layers.len() + 1],
    }
}

fn staged(g: &Graph, devices: DevicePlacement, options: &CompileOptions) -> StagedProgram {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target()),
        &partition(devices),
        options,
    )
    .unwrap_or_else(|error| panic!("compile_staged: {error}"))
}

fn stage_graphs(program: &StagedProgram) -> Vec<&Graph> {
    program
        .stages()
        .map(|(_, _, stage)| stage.graph())
        .collect()
}

fn uncut(g: &Graph, inputs: &HashMap<ValueId, Value>) -> poot_eval::Evaluation {
    eval(
        g,
        inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .expect("the uncut graph evaluates")
}

/// A graph whose boundaries stress what the exports must survive: four independent values leave
/// layer 0 (`a`, a structural duplicate `a2` that CSE merges into `a`, `c`, and `d`), `c` fans out to
/// both later stages, layer 0's carried state is fed by a boundary value, and a validation witness
/// hangs off another boundary.
fn export_fixture() -> Graph {
    let b = Builder::for_tests();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let v = b.constant("v", TensorType::f32(vec![4])).unwrap();
    let (a, a2, c, d, witness);
    let (k0_in, k0_out);
    {
        let _layer = b.layer_scope(0);
        a = b.binary(BinOp::Mul, x, w).unwrap();
        a2 = b.binary(BinOp::Mul, x, w).unwrap();
        c = b.binary(BinOp::Add, a, v).unwrap();
        d = b.binary(BinOp::Sub, a, v).unwrap();
        k0_in = b
            .state_input("k0", TensorType::f32(vec![4]), StateRole::Recurrent)
            .unwrap();
        k0_out = b.binary(BinOp::Add, k0_in, c).unwrap();
        let total = b.reduce(poot_graph_ir::RedOp::Sum, d, 0, true).unwrap();
        let negated = b.unary(UnOp::Neg, total).unwrap();
        witness = b
            .binary_scalar(BinOp::Ge, negated, Scalar::F32(1.0e9))
            .unwrap();
    }
    let (k1_in, k1_out, g);
    {
        let _layer = b.layer_scope(1);
        let e = b.binary(BinOp::Mul, a, a2).unwrap();
        let f = b.binary(BinOp::Add, e, c).unwrap();
        g = b.binary(BinOp::Mul, f, d).unwrap();
        k1_in = b
            .state_input("k1", TensorType::f32(vec![4]), StateRole::Recurrent)
            .unwrap();
        k1_out = b.binary(BinOp::Add, k1_in, g).unwrap();
    }
    let out;
    {
        let _layer = b.layer_scope(2);
        out = b.binary(BinOp::Add, g, c).unwrap();
    }
    poot_graph_ir::test_support::graph_fixtures::finish_with_state_and_validations(
        b,
        out,
        &[(k0_in, k0_out), (k1_in, k1_out)],
        &[(ValidationId(1), "d_not_positive", witness)],
    )
    .expect("the fixture validates")
}

fn export_inputs(g: &Graph) -> HashMap<ValueId, Value> {
    let mut inputs = HashMap::new();
    for (index, &id) in g.inputs().iter().enumerate() {
        let shape = g.aval(id).shape.clone();
        let data = fill(shape.iter().product(), 100 + index as u64);
        inputs.insert(id, Value::from(HostTensor::f32(shape, data)));
    }
    inputs
}

/// SC-007: a production `compile_staged` exports four independent values, two of which CSE merges,
/// fans `c` out to both later stages, carries state through a boundary-fed pair and keeps a witness
/// root. After optimization and renumbering every consumer, the output, both state pairs and the
/// witness equal the uncut oracle, and every export still resolves.
///
/// Mutations (each makes this red): retain only the output root (exports out of `liveness_roots`:
/// DCE drops the boundary producers and compile or the chain fails); drop one remapping (CSE's
/// `remap_results` skipping exports: the merged export keeps a value CSE removed); keep a stale
/// pre-CSE `ValueId` (same site); bind only one consumer (the second `Link` of `c`).
#[test]
fn exports_survive_cse_dce_and_renumbering_and_every_consumer_sees_its_value() {
    let graph = export_fixture();
    let inputs = export_inputs(&graph);
    let whole = uncut(&graph, &inputs);
    let program = staged(&graph, pipeline(&[0, 1]), &options());

    // Stage 0 exports a, a2, c and d; a2 was merged into a by CSE, so two exports share one value.
    let (_, _, first) = program.stages().next().unwrap();
    let exports: Vec<_> = first.exports().collect();
    assert_eq!(exports.len(), 4, "a, a2, c and d all leave stage 0");
    let distinct: std::collections::BTreeSet<ValueId> =
        exports.iter().map(|export| export.value).collect();
    assert_eq!(
        distinct.len(),
        3,
        "CSE merged the duplicate into one compiled value, each export keeps its own id"
    );
    for export in &exports {
        assert!(
            first
                .graph()
                .eqns()
                .iter()
                .any(|eqn| eqn.out == export.value)
                || first.graph().inputs().contains(&export.value),
            "{:?} resolves to a value the compiled stage defines",
            export.id
        );
    }
    // `c` is read by stages 1 and 2: one link per consumer.
    let c_links = program
        .links()
        .iter()
        .filter(|link| link.from == StageId(0))
        .count();
    assert_eq!(
        c_links, 5,
        "a, a2, c, d into stage 1 and c again into stage 2"
    );
    assert!(
        program
            .links()
            .iter()
            .any(|link| link.from == StageId(0) && link.to == StageId(2)),
        "a boundary fans out past the next stage"
    );

    let chained = chain(&graph, &stage_graphs(&program), &inputs);
    assert_eq!(bits(&chained.output), bits(&whole.output), "the output");
    for (pair, (got, expected)) in graph.state().zip(chained.state.iter().zip(&whole.state)) {
        assert_eq!(bits(got), bits(expected), "state {} -> {}", pair.0, pair.1);
    }
    let environment = whole.environment.as_ref().unwrap();
    let declared = graph.witnesses().next().unwrap();
    assert_eq!(
        bits(&chained.witnesses[&declared.id]),
        bits(environment[declared.value].as_ref().unwrap()),
        "the witness root survives the cut"
    );
    for (id, value) in &chained.boundaries {
        assert_eq!(
            bits(value),
            bits(environment[id.0].as_ref().unwrap()),
            "{id:?} handed on a different value than the uncut graph computes"
        );
    }
}

/// A chain of `blocks` blocks, three equations each, tagged by layer, with one shared constant and one
/// shared activation slot read by every block.
fn long_chain(blocks: usize) -> Graph {
    let b = Builder::for_tests();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![4])).unwrap();
    let mut hidden = x;
    for layer in 0..blocks {
        let _scope = b.layer_scope(layer);
        let scaled = b.binary(BinOp::Mul, hidden, w).unwrap();
        let shifted = b.binary(BinOp::Add, scaled, x).unwrap();
        hidden = b.binary(BinOp::Mul, shifted, scaled).unwrap();
    }
    b.finish(hidden).unwrap()
}

/// SC-006 (metadata half): the value tables the stages own add up to the graph plus the explicit
/// boundary copies plus the binders every stage replicates, whatever the stage count; they do not
/// grow as stages times values.
///
/// Mutation: give each stage the whole value table (`values: g.values.clone()` in `split_stages`):
/// the total becomes stages times values and this goes red.
#[test]
fn stage_metadata_is_linear_in_the_graph_plus_links_not_stages_times_values() {
    let blocks = 64;
    let graph = long_chain(blocks);
    let values = graph.values().len();
    assert!(
        values > 3 * blocks,
        "a long graph, or the bound proves nothing"
    );
    // The binders each stage reads itself: the shared constant and the shared slot.
    let replicated = 2;
    for stages in [2usize, 4, 8, 16, 32] {
        let cuts: Vec<usize> = (1..stages).map(|s| s * blocks / stages - 1).collect();
        let split = crate::unit_helpers::split_after_layers(&graph, &cuts).unwrap();
        assert_eq!(split.stages.len(), stages);
        let owned: usize = split.stages.iter().map(|s| s.graph.values().len()).sum();
        let links: usize = split
            .boundaries
            .iter()
            .map(|boundary| boundary.consuming_stages.len())
            .sum();
        assert!(
            owned <= values + links + stages * replicated,
            "{stages} stages own {owned} values; the graph has {values} and the cuts add {links} links \
             and {} replicated binders",
            stages * replicated
        );

        // The compiled programs, too: bounded by the same terms (compile adds no per-stage copy).
        let program = staged(&graph, pipeline(&cuts), &options());
        let compiled: usize = program
            .stages()
            .map(|(_, _, stage)| stage.graph().values().len())
            .sum();
        assert!(
            compiled <= values + links + stages * replicated,
            "{stages} compiled stages own {compiled} values against a bound of {}",
            values + links + stages * replicated
        );
    }
}

/// `ids = token + token` in layer 0, which layer 0 also reads as the index of a gather, then handed to
/// layer 1. Layer 1 either only gathers by it or does arithmetic on it that is itself the graph output.
fn i32_boundary(consumer_does_arithmetic: bool) -> Graph {
    let b = Builder::for_tests();
    let token = b
        .slot(Slot::Token, TensorType::new(vec![4], DType::I32))
        .unwrap();
    let table = b.constant("table", TensorType::f32(vec![8, 4])).unwrap();
    let (ids, rows);
    {
        let _layer = b.layer_scope(0);
        ids = b.binary(BinOp::Add, token, token).unwrap();
        rows = b.gather(table, 0, ids).unwrap();
    }
    let k_in = b
        .state_input("k", TensorType::f32(vec![4, 4]), StateRole::Recurrent)
        .unwrap();
    let (k_out, out);
    {
        let _layer = b.layer_scope(1);
        k_out = b.binary(BinOp::Add, k_in, rows).unwrap();
        out = if consumer_does_arithmetic {
            b.binary(BinOp::Add, ids, ids).unwrap()
        } else {
            b.gather(table, 0, ids).unwrap()
        };
    }
    b.finish(poot_graph_ir::Outputs::new(out).with_state(&[(k_in, k_out)]))
        .unwrap()
}

/// SC-008, POOT-725: an I32 boundary value has one representation, dense I32 words, however its consumer
/// reads it. A consumer that only gathers by it and one that does arithmetic on it both compile across the
/// cut, with the ids and the gathered rows crossing (before, the gather-only consumer read an F32 mirror and
/// the cut was a `LinkSchema` refusal).
///
/// Mutation: make `value_storage_kind` store an I32 value no `Binary` consumes as F32; the gather-only
/// boundary is refused as a `LinkSchema` mismatch and the first loop goes red.
#[test]
fn an_i32_boundary_has_one_representation_whatever_its_consumer_reads() {
    for arithmetic in [false, true] {
        let program = staged(&i32_boundary(arithmetic), pipeline(&[0]), &options());
        assert_eq!(
            program.links().len(),
            2,
            "arithmetic {arithmetic}: the ids and the gathered rows both cross"
        );
    }
}

/// The partition's refusals are typed and come before any compile work.
#[test]
fn unsupported_partitions_are_typed_refusals() {
    let graph = long_chain(3);
    let targets = TargetSet::single(DeviceId(0), target());
    let refused = |partition: Partition| {
        compile_staged(&graph, &targets, &partition, &options())
            .map(|_| ())
            .expect_err("the partition is refused")
    };
    assert!(matches!(
        refused(partition(DevicePlacement::Pipeline {
            after_layers: vec![0],
            devices: vec![DeviceId(0), DeviceId(1)],
        })),
        StagedCompileError::UnknownDevice(DeviceId(1))
    ));
    assert!(matches!(
        refused(Partition {
            experts: ExpertPlacement::Pooled { slots_per_layer: 2 },
            devices: DevicePlacement::Single(DeviceId(0)),
        }),
        StagedCompileError::Pool(PoolError::NoRoutedExperts)
    ));
    assert!(matches!(
        refused(partition(DevicePlacement::Single(DeviceId(5)))),
        StagedCompileError::UnknownDevice(DeviceId(5))
    ));
    assert!(matches!(
        refused(partition(DevicePlacement::Pipeline {
            after_layers: vec![0, 1],
            devices: vec![DeviceId(0)],
        })),
        StagedCompileError::DeviceCount {
            cuts: 2,
            stages: 3,
            devices: 1
        }
    ));
}

/// A boundary that is a view of carried state is refused at compile: the producing stage commits the
/// state before the consumer would read the view, so the consumer would see the committed value, not
/// the one the uncut graph reads. The refusal is `compile_staged`'s, typed, before any executor
/// loads the program.
///
/// Mutation: skip the `export_aliasing_state` check in `compile_staged`; the cut is admitted and
/// this goes red.
#[test]
fn a_boundary_that_views_carried_state_is_refused_at_compile() {
    let b = Builder::for_tests();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4])).unwrap();
    let k_in = b
        .state_input("k", TensorType::f32(vec![4]), StateRole::Recurrent)
        .unwrap();
    let (view, k_out);
    {
        let _layer = b.layer_scope(0);
        view = b.reshape(k_in, vec![2, 2]).unwrap();
        k_out = b.binary(BinOp::Add, k_in, x).unwrap();
    }
    let out;
    {
        let _layer = b.layer_scope(1);
        out = b.binary(BinOp::Add, view, view).unwrap();
    }
    let graph = b
        .finish(poot_graph_ir::Outputs::new(out).with_state(&[(k_in, k_out)]))
        .unwrap();
    match compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target()),
        &partition(pipeline(&[0])),
        &options(),
    ) {
        Err(StagedCompileError::BoundaryAliasesState { stage, .. }) => {
            assert_eq!(
                stage,
                StageId(0),
                "the producing stage owns the carried state"
            );
        }
        other => panic!(
            "expected a BoundaryAliasesState refusal, got {:?}",
            other.map(|_| ())
        ),
    }
}

// ---- POOT-746: the expert pool ---------------------------------------------------------------

const EXPERTS: usize = 8;
const TOP_K: usize = 2;
const HIDDEN: usize = 4;
const INTER: usize = 3;

/// `layers` routed MoE blocks over `tokens` tokens: each layer's up projection is held in the
/// checkpoint's `[E, N, K]` orientation and read through the transpose tracers spell, its down
/// projection as the logical `[E, K, N]` stack.
fn moe_model(tokens: usize, layers: usize) -> Graph {
    let b = Builder::for_tests();
    let mut x = b
        .slot(Slot::Activation, TensorType::f32(vec![1, tokens, HIDDEN]))
        .unwrap();
    for layer in 0..layers {
        let _scope = b.layer_scope(layer);
        let router = b
            .constant(
                &format!("w.l{layer}.router"),
                TensorType::f32(vec![HIDDEN, EXPERTS]),
            )
            .unwrap();
        let stored_in = b
            .constant(
                &format!("w.l{layer}.moe.in"),
                TensorType::f32(vec![EXPERTS, 2 * INTER, HIDDEN]),
            )
            .unwrap();
        let w_in = b.transpose(stored_in, vec![0, 2, 1]).unwrap();
        let w_out = b
            .constant(
                &format!("w.l{layer}.moe.out"),
                TensorType::f32(vec![EXPERTS, INTER, HIDDEN]),
            )
            .unwrap();
        let y = poot_graph_ir::ops::moe(&b, x, router, w_in, w_out, EXPERTS, TOP_K, INTER).unwrap();
        x = b.binary(BinOp::Add, x, y).unwrap();
    }
    b.finish(x).unwrap()
}

fn pooled_partition(slots_per_layer: usize) -> Partition {
    Partition {
        experts: ExpertPlacement::Pooled { slots_per_layer },
        devices: DevicePlacement::Single(DeviceId(0)),
    }
}

fn compile_pooled(g: &Graph, slots_per_layer: usize) -> Result<StagedProgram, StagedCompileError> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target()),
        &pooled_partition(slots_per_layer),
        &options(),
    )
}

/// A host that makes exactly the experts a step selected resident, one row each in ascending expert
/// order, and rewrites each pooled table and map: the contract of a sync point with none of the LRU
/// (the production manager's, and its tests, live with the driver).
struct PoolHost<'g> {
    /// The unpooled graph's inputs by const name: the full `[E, ..]` tables.
    tables: HashMap<String, (Vec<usize>, Vec<f32>)>,
    graph: &'g Graph,
    /// Experts made resident per sync, for the rows that need to see eviction.
    resident: Vec<Vec<usize>>,
    /// A sync that does not hold: the lowest selected expert is left out of the pool (its map entry stays
    /// `-1`), as a host that mis-counted the step's selection would.
    leave_out_lowest: bool,
}

impl PoolHost<'_> {
    fn sync(&mut self, point: &SyncPoint, ids: &[i32]) -> HashMap<SlotKey, Value> {
        let mut needed: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
        needed.sort_unstable();
        needed.dedup();
        assert!(needed.len() <= point.site.slots);
        if self.leave_out_lowest {
            needed.remove(0);
        }
        self.resident.push(needed.clone());
        let mut map = vec![-1i32; point.site.experts];
        for (row, &expert) in needed.iter().enumerate() {
            map[expert] = row as i32;
        }
        let mut writes = HashMap::new();
        writes.insert(
            point.site.map.clone(),
            Value::from(HostTensor::i32(vec![point.site.experts], map)),
        );
        for weight in &point.site.weights {
            let (shape, data) = &self.tables[&weight.name];
            assert_eq!(shape[0], point.site.experts);
            let per_expert = data.len() / shape[0];
            let mut pool = vec![0.0f32; weight.pool.numel()];
            for (row, &expert) in needed.iter().enumerate() {
                pool[row * per_expert..(row + 1) * per_expert]
                    .copy_from_slice(&data[expert * per_expert..(expert + 1) * per_expert]);
            }
            writes.insert(
                weight.slot.clone(),
                Value::from(HostTensor::f32(weight.pool.shape.clone(), pool)),
            );
        }
        let _ = self.graph;
        writes
    }
}

/// Evaluate a pooled program stage by stage on the oracle: each stage binds the unpooled graph's
/// inputs by name, the boundaries earlier stages exported, and whatever the sync before it wrote.
fn eval_pooled(
    program: &StagedProgram,
    graph: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> (Value, Vec<Vec<usize>>) {
    eval_pooled_syncing(program, graph, inputs, false)
        .unwrap_or_else(|(stage, error)| panic!("stage {stage:?} did not evaluate: {error}"))
}

/// The output of a pooled run and the experts each sync made resident.
type PooledRun = (Value, Vec<Vec<usize>>);

/// [`eval_pooled`], with the stage that fails and why returned. With `leave_out_lowest` the host's syncs do
/// not hold: the lowest selected expert is never made resident.
fn eval_pooled_syncing(
    program: &StagedProgram,
    graph: &Graph,
    inputs: &HashMap<ValueId, Value>,
    leave_out_lowest: bool,
) -> Result<PooledRun, (StageId, poot_eval::EvalError)> {
    let mut by_name: HashMap<String, Value> = HashMap::new();
    let mut tables = HashMap::new();
    let mut activations: HashMap<SlotKey, Value> = HashMap::new();
    for &id in graph.inputs() {
        let meta = graph.meta(id);
        let value = inputs[&id].clone();
        match meta.storage {
            Storage::Const => {
                let host = value
                    .as_host()
                    .expect("dense")
                    .as_f32()
                    .expect("f32")
                    .to_vec();
                tables.insert(
                    meta.name.clone().expect("named"),
                    (meta.aval.shape.clone(), host),
                );
                by_name.insert(meta.name.clone().expect("named"), value);
            }
            Storage::Slot(_) => {
                activations.insert(meta.slot_key().cloned().expect("keyed"), value);
            }
            _ => {}
        }
    }
    let mut host = PoolHost {
        tables,
        graph,
        resident: Vec::new(),
        leave_out_lowest,
    };
    let mut resident_slots: HashMap<SlotKey, Value> = HashMap::new();
    let mut boundaries: HashMap<SlotKey, Value> = HashMap::new();
    let mut output = None;
    for (stage, _, program_stage) in program.stages() {
        let g = program_stage.graph();
        let stage_inputs: HashMap<ValueId, Value> = g
            .inputs()
            .iter()
            .map(|&id| {
                let meta = g.meta(id);
                let value = match meta.storage {
                    Storage::Computed(payload) => Value::from(payload.tensor()),
                    Storage::Const => by_name[meta.name.as_deref().expect("named")].clone(),
                    Storage::Slot(Slot::ExpertPool | Slot::ExpertPoolMap) => {
                        resident_slots[meta.slot_key().expect("keyed")].clone()
                    }
                    Storage::Slot(_) => {
                        let key = meta.slot_key().expect("keyed");
                        boundaries
                            .get(key)
                            .or_else(|| activations.get(key))
                            .unwrap_or_else(|| panic!("stage {stage:?}: {key:?} is unbound"))
                            .clone()
                    }
                    other => panic!("unexpected input storage {other:?}"),
                };
                (id, value)
            })
            .collect();
        let result = eval(
            g,
            &stage_inputs,
            EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
        )
        .map_err(|error| (stage, error))?;
        let environment = result.environment.expect("the environment was kept");
        for export in g.exports() {
            boundaries.insert(
                export.id.slot_key(),
                environment[export.value].clone().expect("exported"),
            );
        }
        for point in program.sync_points().iter().filter(|p| p.after == stage) {
            let &[read] = point.reads.as_slice() else {
                panic!("a routed sync reads the ids")
            };
            let ids = boundaries[&read.slot_key()]
                .as_host()
                .expect("dense")
                .as_i32()
                .expect("i32")
                .to_vec();
            resident_slots.extend(host.sync(point, &ids));
        }
        output = Some(result.output);
    }
    Ok((output.expect("a stage ran"), host.resident))
}

/// SC-001 (oracle half): the pooled program's cut graphs, with a host that makes the selected experts
/// resident at each sync, equal the unpooled graph bit for bit, at decode (one token) and at a
/// prefill whose demand fills the pool. Eviction happens: across the layers' syncs different experts
/// are resident in the same `P = 4 < E = 8` rows.
///
/// Mutation (observed red): in `pool_experts`, point the contractions at the untranslated ids instead
/// of the gather's output; the pooled tables are indexed by global ids and the output differs.
#[test]
fn a_pooled_moe_program_equals_the_unpooled_oracle_bit_for_bit() {
    for tokens in [1, 2] {
        let graph = moe_model(tokens, 3);
        let inputs = bind_decode(&graph);
        let whole = uncut(&graph, &inputs);
        let program = compile_pooled(&graph, 4).expect("the pool fits the step's demand");
        assert_eq!(program.stages().count(), 4, "one cut per routed layer");
        assert_eq!(program.sync_points().len(), 3);
        for (layer, point) in program.sync_points().iter().enumerate() {
            assert_eq!(point.after, StageId(layer as u32));
            assert_eq!(point.site.layer, Some(layer));
            assert_eq!((point.site.experts, point.site.slots), (EXPERTS, 4));
            assert_eq!(point.site.weights.len(), 2, "the up and the down table");
            assert_eq!(point.reads, vec![point.site.ids]);
            for weight in &point.site.weights {
                assert_eq!(weight.pool.shape[0], 4);
            }
        }
        let (pooled, resident) = eval_pooled(&program, &graph, &inputs);
        assert_eq!(
            bits(&pooled),
            bits(&whole.output),
            "tokens={tokens}: pooled output differs from the unpooled graph"
        );
        assert!(
            resident.iter().any(|experts| experts.len() > 1),
            "the host paged real selections"
        );
    }
}

/// POOT-1119: the translated ids of every pooled site pass through the canonical index guard `[0, P)`, whose
/// witness is declared as a validation output of the stage that reads the pool. A host whose sync does not
/// hold (a selected expert left at map entry `-1`) therefore fails that stage's validation instead of
/// reading a wrong expert's row; the exact sync passes.
///
/// Mutation (observed red): in `pool_experts` point the contractions at `raw` (the unguarded gather) and
/// drop the `pooled.push_witness(..)` call with its `added.push(id)`. `compile_pooled` itself fails,
/// `Compile(ExpertIds(.. op: "indexed_matmul" .. OutOfRange))`: the proof refuses the unguarded ids before
/// the declarations assertion is reached. Dropping only the `push_witness` + `added.push(id)` pair (ids
/// still guarded) leaves the declarations assertion red instead.
#[test]
fn a_pooled_selection_not_made_resident_is_a_validation_failure() {
    let graph = moe_model(2, 3);
    let inputs = bind_decode(&graph);
    let program = compile_pooled(&graph, 4).expect("the pool fits the step's demand");

    // One witness per site, in the stage that follows its sync.
    for (layer, point) in program.sync_points().iter().enumerate() {
        let stage = program
            .stages()
            .find(|(id, ..)| id.0 as usize == layer + 1)
            .map(|(_, _, program)| program)
            .expect("the stage after the sync");
        let names: Vec<&str> = stage
            .graph()
            .witnesses()
            .map(|witness| witness.name)
            .collect();
        assert_eq!(
            names,
            [format!("pool.site.{layer}.resident")],
            "stage after sync {layer}"
        );
        assert_eq!(point.after, StageId(layer as u32));
    }

    eval_pooled_syncing(&program, &graph, &inputs, false).expect("an exact sync validates");
    let (stage, error) = eval_pooled_syncing(&program, &graph, &inputs, true)
        .expect_err("a selected expert that is not resident must fail validation");
    assert_eq!(stage, StageId(1), "the stage after the first sync");
    assert!(
        error.to_string().contains("pool.site.0.resident"),
        "the failure names the witness: {error}"
    );
}

/// SC-003 (compile half): a step whose selection can need more experts than the pool holds, and a pool
/// that is not smaller than the experts, are the two typed refusals from `compile_staged`.
///
/// Mutation (observed red): compute the demand from the last axis of the ids only (without the rows
/// of the step); the three-token prefill's six ids pass as a demand of two and the first assertion
/// goes red.
#[test]
fn a_pool_below_the_step_demand_or_not_below_the_experts_is_a_typed_refusal() {
    let prefill = moe_model(3, 1);
    match compile_pooled(&prefill, 4) {
        Err(StagedCompileError::Pool(PoolError::CapacityBelowStepDemand {
            layer: Some(0),
            demand: 6,
            slots: 4,
        })) => {}
        other => panic!(
            "expected CapacityBelowStepDemand, got {:?}",
            other.map(|_| ())
        ),
    }
    let decode = moe_model(1, 1);
    match compile_pooled(&decode, EXPERTS) {
        Err(StagedCompileError::Pool(PoolError::CapacityNotBelowExperts {
            slots: 8,
            experts: 8,
        })) => {}
        other => panic!(
            "expected CapacityNotBelowExperts, got {:?}",
            other.map(|_| ())
        ),
    }
}

// ---- POOT-1140: the plan's read traffic ------------------------------------------------------

const TRAFFIC_VOCAB: usize = 16;
const TRAFFIC_HIDDEN: usize = 8;
const TRAFFIC_KV_HEADS: usize = 2;
const TRAFFIC_CAPACITY: usize = 8;
const TRAFFIC_HEAD_DIM: usize = 4;

/// A one-layer decode shape: a tied embedding (`embed` is both gathered by token and read, transposed,
/// as the logits head), an untied table only a gather reads, one projection, and a positional cache the
/// step folds in place. f32 throughout, so every expected byte count is a product of the dims.
fn traffic_fixture() -> Graph {
    let b = Builder::for_tests();
    let token = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
    let embed = b
        .constant(
            "embed",
            TensorType::f32(vec![TRAFFIC_VOCAB, TRAFFIC_HIDDEN]),
        )
        .unwrap();
    let side = b
        .constant(
            "side_table",
            TensorType::f32(vec![TRAFFIC_VOCAB, TRAFFIC_HIDDEN]),
        )
        .unwrap();
    let proj = b
        .constant(
            "proj",
            TensorType::f32(vec![TRAFFIC_HIDDEN, TRAFFIC_HIDDEN]),
        )
        .unwrap();
    let cache_in = b
        .state_input(
            "kv",
            TensorType::f32(vec![
                1,
                TRAFFIC_KV_HEADS,
                TRAFFIC_CAPACITY,
                TRAFFIC_HEAD_DIM,
            ]),
            StateRole::Positional { axis: 2 },
        )
        .unwrap();
    let row = b.gather(embed, 0, token).unwrap();
    let side_row = b.gather(side, 0, token).unwrap();
    let x = b
        .reshape(
            b.binary(BinOp::Add, row, side_row).unwrap(),
            vec![1, TRAFFIC_HIDDEN],
        )
        .unwrap();
    // The projection is read through a reshape, which plans as an alias of its buffer.
    let flat = b
        .reshape(proj, vec![TRAFFIC_HIDDEN * TRAFFIC_HIDDEN])
        .unwrap();
    let proj = b
        .reshape(flat, vec![TRAFFIC_HIDDEN, TRAFFIC_HIDDEN])
        .unwrap();
    let hidden = b.matmul(x, proj).unwrap();
    let head = b
        .transpose(embed, vec![1, 0])
        .expect("the tied head is the embedding transposed");
    let logits = b.matmul(hidden, head).unwrap();
    let cache_out = b.binary(BinOp::Mul, cache_in, cache_in).unwrap();
    b.finish(poot_graph_ir::Outputs::new(logits).with_state(&[(cache_in, cache_out)]))
        .expect("the fixture validates")
}

/// The traffic of a decode step is what its dispatches read: the tied embedding once at its stored
/// width (read both as a gathered row and as the head, so counted whole once), the projection read
/// through an aliasing reshape, and the cache per attended position; a table only a gather reads
/// contributes nothing. Mutations: count the head's weight as f32 beside the stored embedding
/// (`weight_bytes += vocab * hidden * 4`), count a gather-only table whole, or stop resolving aliases to
/// their buffer.
#[test]
fn read_traffic_counts_each_streamed_weight_once_and_the_cache_per_position() {
    let program = staged(&traffic_fixture(), pipeline(&[]), &options());
    let traffic = program.read_traffic();
    let f32_bytes = 4;
    let embed = TRAFFIC_VOCAB * TRAFFIC_HIDDEN * f32_bytes;
    let proj = TRAFFIC_HIDDEN * TRAFFIC_HIDDEN * f32_bytes;
    assert_eq!(
        traffic.weight_bytes,
        (embed + proj) as u64,
        "the tied head and the gathered embedding are one weight, the gather-only table none"
    );
    assert_eq!(traffic.recurrent_state_bytes, 0);
    let per_position = TRAFFIC_KV_HEADS * TRAFFIC_HEAD_DIM * f32_bytes;
    assert_eq!(traffic.kv_bytes_per_position, per_position as u64);
    assert_eq!(
        traffic.step_bytes([2, 3]),
        (embed + proj + 5 * per_position) as u64
    );
}

/// A block-granular cache counts each position's share of an entry: a `[rows, 1, blocks, dim]` cache
/// of `ratio`-position blocks reads `dim * 4 / ratio` bytes per attended position per row, not
/// `dim * 4` as the same shape declared positional would (POOT-1143). Mutation: handle `Blocked` like
/// `Positional` in `read_traffic` (drop the `ratio` factor); the per-position count is four times too large.
#[test]
fn a_blocked_cache_counts_its_entry_over_the_positions_it_covers() {
    const ROWS: usize = 2;
    const BLOCKS: usize = 4;
    const DIM: usize = 8;
    const RATIO: u32 = 4;
    let b = Builder::for_tests();
    let q = b
        .constant("q", TensorType::f32(vec![ROWS, 1, 1, DIM]))
        .unwrap();
    let blocks = b
        .state_input(
            "blocks",
            TensorType::f32(vec![ROWS, 1, BLOCKS, DIM]),
            StateRole::Blocked {
                axis: 2,
                ratio: std::num::NonZeroU32::new(RATIO).unwrap(),
            },
        )
        .unwrap();
    let scores = b
        .matmul(q, b.transpose(blocks, vec![0, 1, 3, 2]).unwrap())
        .unwrap();
    let blocks_out = b.binary(BinOp::Mul, blocks, blocks).unwrap();
    let graph = b
        .finish(poot_graph_ir::Outputs::new(scores).with_state(&[(blocks, blocks_out)]))
        .expect("the fixture validates");
    let traffic = staged(&graph, pipeline(&[]), &options()).read_traffic();
    assert_eq!(
        traffic.kv_bytes_per_position,
        (DIM * 4 / RATIO as usize) as u64
    );
    assert_eq!(traffic.recurrent_state_bytes, 0);
}

/// A cache a block-selected attention reads is counted at no more than its `(K + 1) * ratio` positions per
/// token, not at every attended position (POOT-1143): a token attending 40 positions of a cache selected from
/// at 12 reads 12, one attending 5 reads 5. A dense cache beside it keeps counting every attended position.
/// Mutation: count the selected cache in `kv_bytes_per_position`; the long token's bytes triple.
#[test]
fn a_block_selected_cache_is_counted_at_its_selected_positions() {
    let b = Builder::for_tests();
    let q = b.constant("q", TensorType::f32(vec![1, 2, 1, 4])).unwrap();
    let ids = b
        .constant("ids", TensorType::new(vec![1, 1, 1, 2], DType::I32))
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![1, 1], DType::I32))
        .unwrap();
    let cache = |name: &str| {
        b.state_input(
            name,
            TensorType::f32(vec![1, 1, 64, 4]),
            StateRole::Positional { axis: 2 },
        )
        .unwrap()
    };
    let (k, v) = (cache("k"), cache("v"));
    let out = b
        .selected_attention(
            poot_graph_ir::SelectedAttentionAttrs {
                n_rep: 2,
                scale: 0.5,
                softcap: None,
                ratio: 4,
                k: 2,
            },
            q,
            k,
            v,
            ids,
            pos,
        )
        .unwrap();
    let k_out = b.binary(BinOp::Mul, k, k).unwrap();
    let v_out = b.binary(BinOp::Mul, v, v).unwrap();
    let graph = b
        .finish(poot_graph_ir::Outputs::new(out).with_state(&[(k, k_out), (v, v_out)]))
        .expect("the fixture validates");
    let traffic = staged(&graph, pipeline(&[]), &options()).read_traffic();
    let per_position = 2 * 4 * 4; // k and v: four f32 per position each
    assert_eq!(traffic.kv_bytes_per_position, 0);
    assert_eq!(traffic.selected_kv_bytes_per_position, per_position);
    assert_eq!(traffic.selected_positions, 12);
    let base = traffic.step_bytes([]);
    assert_eq!(traffic.step_bytes([5]) - base, 5 * per_position);
    assert_eq!(traffic.step_bytes([40]) - base, 12 * per_position);
}

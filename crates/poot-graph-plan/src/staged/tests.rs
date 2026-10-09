//! POOT-745 at the compile level: what `compile_staged` produces for a pipeline, checked on the CPU
//! oracle (ADR-0101 tier 1: a cut graph chained stage by stage equals the uncut graph bit for bit).
//! The executor and the device lanes check the same programs run.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_eval::{EvalBudget, EvalOptions, Value, eval};
use poot_graph_ir::{
    BinOp, Builder, Graph, Scalar, Slot, StateRole, TensorType, UnOp, ValidationChannel,
    ValidationId, ValidationOutputs, ValueId,
};
use poot_quant::weights::LoraTarget;
use poot_target::Backend;
use poot_tensor::{DType, HostTensor};
use poot_test_util::device_caps::default_caps_for;

use super::*;
use crate::test_support::{
    LAYERS, bind_decode, bits, chain, fill, qwen2_decode, qwen2_decode_rows,
};
use crate::{CompileLimits, LoraPlan, Submission};

fn target() -> Target {
    Target {
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

fn staged<V: ValidationChannel>(
    g: &Graph<V>,
    devices: DevicePlacement,
    options: &CompileOptions,
) -> StagedProgram<V> {
    compile_staged(
        g,
        &TargetSet::single(DeviceId(0), target()),
        &partition(devices),
        options,
    )
    .unwrap_or_else(|error| panic!("compile_staged: {error}"))
}

fn stage_graphs<V: ValidationChannel>(program: &StagedProgram<V>) -> Vec<&Graph<V>> {
    program
        .stages()
        .map(|(_, _, stage)| stage.graph())
        .collect()
}

fn uncut<V: ValidationChannel>(
    g: &Graph<V>,
    inputs: &HashMap<ValueId, Value>,
) -> poot_eval::Evaluation {
    eval(
        g,
        inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED).keep_environment(),
    )
    .expect("the uncut graph evaluates")
}

/// SC-001 (oracle half): qwen2's decode step cut after layers 0 and 1, compiled stage by stage and
/// chained, equals the uncut traced graph on the logits and on every carried KV pair, bit for bit.
#[test]
fn a_qwen2_pipeline_cut_after_layers_zero_and_one_equals_the_uncut_oracle() {
    let graph = qwen2_decode(LAYERS);
    let inputs = bind_decode(&graph);
    let whole = uncut(&graph, &inputs);

    let program = staged(&graph, pipeline(&[0, 1]), &options());
    assert_eq!(program.stages().count(), 3);
    assert!(
        !program.links().is_empty(),
        "the cuts cross values, or this row proves nothing"
    );
    let chained = chain(&graph, &stage_graphs(&program), &inputs);
    assert_eq!(bits(&chained.output), bits(&whole.output));
    assert_eq!(chained.state.len(), graph.state.len());
    for (pair, (got, expected)) in graph
        .state
        .iter()
        .zip(chained.state.iter().zip(&whole.state))
    {
        assert_eq!(
            bits(got),
            bits(expected),
            "carried state v{} -> v{} differs",
            pair.0,
            pair.1
        );
    }
}

/// A graph whose boundaries stress what the exports must survive: four independent values leave
/// layer 0 (`a`, a structural duplicate `a2` that CSE merges into `a`, `c`, and `d`), `c` fans out to
/// both later stages, layer 0's carried state is fed by a boundary value, and a validation witness
/// hangs off another boundary.
fn export_fixture() -> Graph<ValidationOutputs> {
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let w = b.constant("w", TensorType::f32(vec![4]));
    let v = b.constant("v", TensorType::f32(vec![4]));
    let (a, a2, c, d, witness);
    let (k0_in, k0_out);
    {
        let _layer = b.layer_scope(0);
        a = b.binary(BinOp::Mul, x, w);
        a2 = b.binary(BinOp::Mul, x, w);
        c = b.binary(BinOp::Add, a, v);
        d = b.binary(BinOp::Sub, a, v);
        k0_in = b.state_input("k0", TensorType::f32(vec![4]), StateRole::Recurrent);
        k0_out = b.binary(BinOp::Add, k0_in, c);
        let total = b.reduce(poot_graph_ir::RedOp::Sum, d, 0, true);
        let negated = b.unary(UnOp::Neg, total);
        witness = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(1.0e9));
    }
    let (k1_in, k1_out, g);
    {
        let _layer = b.layer_scope(1);
        let e = b.binary(BinOp::Mul, a, a2);
        let f = b.binary(BinOp::Add, e, c);
        g = b.binary(BinOp::Mul, f, d);
        k1_in = b.state_input("k1", TensorType::f32(vec![4]), StateRole::Recurrent);
        k1_out = b.binary(BinOp::Add, k1_in, g);
    }
    let out;
    {
        let _layer = b.layer_scope(2);
        out = b.binary(BinOp::Add, g, c);
    }
    crate::test_support::finish_with_state_and_validations(
        b,
        out,
        &[(k0_in, k0_out), (k1_in, k1_out)],
        &[(ValidationId(1), "d_not_positive", witness)],
    )
    .expect("the fixture validates")
}

fn export_inputs(g: &Graph<ValidationOutputs>) -> HashMap<ValueId, Value> {
    let mut inputs = HashMap::new();
    for (index, &id) in g.inputs.iter().enumerate() {
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
    let exports = first.exports();
    assert_eq!(exports.len(), 4, "a, a2, c and d all leave stage 0");
    let distinct: std::collections::BTreeSet<ValueId> =
        exports.iter().map(|export| export.value).collect();
    assert_eq!(
        distinct.len(),
        3,
        "CSE merged the duplicate into one compiled value, each export keeps its own id"
    );
    for export in exports {
        assert!(
            first.graph().eqns.iter().any(|eqn| eqn.out == export.value)
                || first.graph().inputs.contains(&export.value),
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
    for (pair, (got, expected)) in graph
        .state
        .iter()
        .zip(chained.state.iter().zip(&whole.state))
    {
        assert_eq!(
            bits(got),
            bits(expected),
            "state v{} -> v{}",
            pair.0,
            pair.1
        );
    }
    let environment = whole.environment.as_ref().unwrap();
    let declared = &graph.validation_outputs()[0];
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
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let w = b.constant("w", TensorType::f32(vec![4]));
    let mut hidden = x;
    for layer in 0..blocks {
        let _scope = b.layer_scope(layer);
        let scaled = b.binary(BinOp::Mul, hidden, w);
        let shifted = b.binary(BinOp::Add, scaled, x);
        hidden = b.binary(BinOp::Mul, shifted, scaled);
    }
    b.finish(hidden)
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
    let values = graph.values.len();
    assert!(
        values > 3 * blocks,
        "a long graph, or the bound proves nothing"
    );
    // The binders each stage reads itself: the shared constant and the shared slot.
    let replicated = 2;
    for stages in [2usize, 4, 8, 16, 32] {
        let cuts: Vec<usize> = (1..stages).map(|s| s * blocks / stages - 1).collect();
        let split = crate::test_support::split_after_layers(&graph, &cuts).unwrap();
        assert_eq!(split.stages.len(), stages);
        let owned: usize = split.stages.iter().map(|s| s.graph.values.len()).sum();
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
            .map(|(_, _, stage)| stage.graph().values.len())
            .sum();
        assert!(
            compiled <= values + links + stages * replicated,
            "{stages} compiled stages own {compiled} values against a bound of {}",
            values + links + stages * replicated
        );
    }
}

/// SC-005 (oracle half): a nonzero LoRA stack applied once to the whole graph, then cut after
/// layer 0 on one device, equals the same adapted graph compiled as one stage on every output and
/// carried-state element, and both differ from the unadapted graph (the adapter matters).
///
/// Mutation: compile each stage with the caller's options instead of the LoRA-free copy
/// (`compile_staged` stage compile): the correction is applied a second time to every
/// stage and the chain disagrees with the single stage.
#[test]
fn a_lora_graph_cut_across_stages_equals_the_unpartitioned_adapted_graph() {
    let rows = 2;
    let graph = qwen2_decode_rows(2, rows);
    let inputs = bind_decode(&graph);
    let lora = LoraPlan {
        rows: NonZeroUsize::new(rows).unwrap(),
        adapters: NonZeroUsize::new(1).unwrap(),
        rank: NonZeroUsize::new(2).unwrap(),
        projections: LoraTarget::ALL.to_vec(),
    };
    let with_lora = options().with_lora(lora);

    let single = staged(&graph, DevicePlacement::Single(DeviceId(0)), &with_lora);
    let cut = staged(&graph, pipeline(&[0]), &with_lora);

    // The adapted graph is the one the oracle evaluates, so bind its inputs by declared identity.
    let single_graph = stage_graphs(&single)[0];
    let adapted_inputs: HashMap<ValueId, Value> = single_graph
        .inputs
        .iter()
        .map(|&id| (id, adapted_input(single_graph, id, &graph, &inputs)))
        .collect();
    let one = eval(
        single_graph,
        &adapted_inputs,
        EvalOptions::new(EvalBudget::UNBOUNDED),
    )
    .expect("the adapted graph evaluates");

    let chained = chain(single_graph, &stage_graphs(&cut), &adapted_inputs);
    assert_eq!(
        bits(&chained.output),
        bits(&one.output),
        "the cut adapted graph's logits"
    );
    for (got, expected) in chained.state.iter().zip(&one.state) {
        assert_eq!(bits(got), bits(expected), "the cut adapted graph's state");
    }

    let unadapted = uncut(&graph, &inputs);
    assert_ne!(
        bits(&one.output),
        bits(&unadapted.output),
        "the adapters change the logits, or this row proves nothing"
    );
}

/// A binder of the compiled adapted graph: the uncut graph's own value where it has one, else a
/// seeded fill (the stacked adapter factors and scale exist only in the adapted graph).
fn adapted_input(
    adapted: &Graph,
    id: ValueId,
    original: &Graph,
    inputs: &HashMap<ValueId, Value>,
) -> Value {
    let meta = adapted.meta(id);
    let found = original.inputs.iter().find(|&&candidate| {
        let theirs = original.meta(candidate);
        theirs.storage == meta.storage
            && theirs.name == meta.name
            && theirs.slot_key() == meta.slot_key()
    });
    if let Some(&other) = found {
        return inputs[&other].clone();
    }
    match meta.storage {
        poot_graph_ir::Storage::Slot(Slot::LoraIdx) => {
            let numel = meta.aval.numel();
            Value::from(HostTensor::f32(
                meta.aval.shape.clone(),
                (0..numel)
                    .map(|row| if row == 0 { 1.0 } else { 0.0 })
                    .collect(),
            ))
        }
        poot_graph_ir::Storage::Computed(payload) => Value::from(payload.tensor()),
        poot_graph_ir::Storage::Const => {
            let name = meta.name.as_deref().unwrap();
            let seed = name
                .bytes()
                .fold(7u64, |h, c| (h ^ u64::from(c)).wrapping_mul(1099511628211));
            Value::from(HostTensor::f32(
                meta.aval.shape.clone(),
                fill(meta.aval.numel(), seed)
                    .iter()
                    .map(|v| v * 0.3)
                    .collect(),
            ))
        }
        other => panic!("no binding for {other:?} {:?}", meta.name),
    }
}

/// `ids = token + token` in layer 0, which layer 0 also reads as the index of a gather (an exact
/// I32 read), then handed to layer 1. Layer 1 either only gathers by it (a float-mirror read: the
/// consumer's component has no arithmetic that reaches an output) or does arithmetic on it that is
/// itself the graph output (an exact read).
fn i32_boundary(consumer_agrees: bool) -> Graph {
    let b = Builder::new();
    let token = b.slot(Slot::Token, TensorType::new(vec![4], DType::I32));
    let table = b.constant("table", TensorType::f32(vec![8, 4]));
    let (ids, rows);
    {
        let _layer = b.layer_scope(0);
        ids = b.binary(BinOp::Add, token, token);
        rows = b.gather(table, 0, ids);
    }
    let k_in = b.state_input("k", TensorType::f32(vec![4, 4]), StateRole::Recurrent);
    let (k_out, out);
    {
        let _layer = b.layer_scope(1);
        k_out = b.binary(BinOp::Add, k_in, rows);
        out = if consumer_agrees {
            b.binary(BinOp::Add, ids, ids)
        } else {
            b.gather(table, 0, ids)
        };
    }
    b.finish_with_state(out, &[(k_in, k_out)])
}

/// SC-008: the two sides of a boundary must agree on dtype, shape and exact storage. An I32 value
/// the producer computes arithmetically and exports is exact words; a consumer that only gathers by
/// it reads the float mirror lane. Both are four 32-bit words, the same byte count and the same
/// logical extent, yet different representations: compile refuses the cut with both schemas named.
/// A consumer that also does arithmetic on it agrees with the producer and compiles.
///
/// Mutation: compare only the byte length in `check_link`: the mismatched cut is admitted and the
/// first assertion goes red.
#[test]
fn a_boundary_the_two_stages_represent_differently_is_refused_at_compile() {
    let mismatched = i32_boundary(false);
    match compile_staged(
        &mismatched,
        &TargetSet::single(DeviceId(0), target()),
        &partition(pipeline(&[0])),
        &options(),
    ) {
        Err(StagedCompileError::LinkSchema {
            producer, consumer, ..
        }) => {
            assert_eq!(producer.aval, consumer.aval, "same dtype and shape");
            assert_eq!(
                producer.storage.element().byte_width(),
                consumer.storage.element().byte_width(),
                "same bytes per element"
            );
            assert_ne!(
                producer.storage, consumer.storage,
                "different representation"
            );
        }
        other => panic!(
            "expected a LinkSchema refusal for the mirror/exact boundary, got {:?}",
            other.map(|_| ())
        ),
    }

    let agreeing = i32_boundary(true);
    let program = staged(&agreeing, pipeline(&[0]), &options());
    assert_eq!(
        program.links().len(),
        2,
        "the ids and the gathered rows both cross"
    );
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
        StagedCompileError::Unsupported(Unsupported::MultiDevice)
    ));
    assert!(matches!(
        refused(Partition {
            experts: ExpertPlacement::Pooled,
            devices: DevicePlacement::Single(DeviceId(0)),
        }),
        StagedCompileError::Unsupported(Unsupported::ExpertPool)
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
    let b = Builder::new();
    let x = b.slot(Slot::Activation, TensorType::f32(vec![4]));
    let k_in = b.state_input("k", TensorType::f32(vec![4]), StateRole::Recurrent);
    let (view, k_out);
    {
        let _layer = b.layer_scope(0);
        view = b.reshape(k_in, vec![2, 2]);
        k_out = b.binary(BinOp::Add, k_in, x);
    }
    let out;
    {
        let _layer = b.layer_scope(1);
        out = b.binary(BinOp::Add, view, view);
    }
    let graph = b.finish_with_state(out, &[(k_in, k_out)]);
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

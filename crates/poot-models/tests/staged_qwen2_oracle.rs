//! POOT-745 SC-001/SC-005 oracle halves and the POOT-747 placement half on the real qwen2 trace: a decode
//! step cut after layers, compiled stage by stage through `compile_staged` and chained on the CPU oracle,
//! equals the uncut graph bit for bit (ADR-0101 tier 1), and placed on two PTX devices puts each layer's
//! weights on its stage. They trace a model, so they sit above poot-graph-plan, not in it (POOT-1130); its
//! own unit tests run the same rows over a hand-built layered decode graph.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_eval::test_support::stage_chain::{bind_decode, bits, chain};
use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::{Graph, Slot, ValueId};
use poot_graph_plan::staged::{
    DeviceId, DevicePlacement, ExpertPlacement, Partition, StagedProgram, TargetSet, compile_staged,
};
use poot_graph_plan::{CollectiveKind, CompileLimits, CompileOptions, LoraPlan, Target};
use poot_models::model::{LogitRows, StepSpec};
use poot_models::test_support::dense::{Dense, Family, step};
use poot_target::Backend;
use poot_tensor::HostTensor;
use poot_test_util::{device_caps::default_caps_for, fill};
use poot_weights::LoraTarget;

/// Layers of the traced qwen2.
const LAYERS: usize = 3;

/// The tiny qwen2 decode graph (GQA 4 over 2) of `layers` layers over a six-position cache, stepping `rows`
/// sequences at once, traced through the registry.
fn qwen2_decode_rows(layers: usize, rows: usize) -> Graph {
    Dense::new(Family::Qwen2)
        .vocab(32)
        .dims(16, 32, layers)
        .heads(4, 2)
        .head_dim(4)
        .max_positions(32)
        .f32_model()
        .model
        .trace(
            &StepSpec::step(step(rows, 1, 6, LogitRows::Last)),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the qwen2 fixture traces a decode step")
}

fn qwen2_decode(layers: usize) -> Graph {
    qwen2_decode_rows(layers, 1)
}

fn target() -> Target {
    Target::Gpu {
        backend: Backend::SpirvVulkan,
        caps: default_caps_for(Backend::SpirvVulkan),
    }
}

fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
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
    eval(g, inputs, EvalOptions::default().keep_environment()).expect("the uncut graph evaluates")
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
    assert_eq!(chained.state.len(), graph.state().count());
    for (pair, (got, expected)) in graph.state().zip(chained.state.iter().zip(&whole.state)) {
        assert_eq!(
            bits(got),
            bits(expected),
            "carried state {} -> {} differs",
            pair.0,
            pair.1
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
        factor_dtype: poot_tensor::DType::F32,
        projections: LoraTarget::ALL.to_vec(),
    };
    let with_lora = options().with_lora(lora);

    let single = staged(&graph, DevicePlacement::Single(DeviceId(0)), &with_lora);
    let cut = staged(&graph, pipeline(&[0]), &with_lora);

    // The adapted graph is the one the oracle evaluates, so bind its inputs by declared identity.
    let single_graph = stage_graphs(&single)[0];
    let adapted_inputs: HashMap<ValueId, Value> = single_graph
        .inputs()
        .iter()
        .map(|&id| (id, adapted_input(single_graph, id, &graph, &inputs)))
        .collect();
    let one = eval(single_graph, &adapted_inputs, EvalOptions::default())
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

/// The names of the weights (named `Const` inputs) a stage graph reads, sorted.
fn weight_names(graph: &Graph) -> Vec<String> {
    let mut names: Vec<String> = graph
        .consts()
        .iter()
        .map(|&value| graph.meta(value))
        .filter(|meta| meta.storage == poot_graph_ir::Storage::Const)
        .filter_map(|meta| meta.name.clone())
        .collect();
    names.sort();
    names
}

/// POOT-747 placement half on the real qwen2 trace, host-only: the decode step cut after layer 0 and
/// placed on two PTX devices compiles with no device. Stage 0 on device 0 reads the embedding and layer 0's weights,
/// stage 1 on device 1 reads layers 1 and 2 and the head, every weight of the uncut graph is read by one
/// of them, and each crossing value is one `PointToPoint` row from device 0 to device 1 with no link.
///
/// Mutation: resolve the cut one layer late in `StageAssignment::from_sites` (`runs.runs.get(layer + 2)`):
/// stage 0 then owns layer 1 and its weights, and the stage-0 weight set goes red.
#[test]
fn a_qwen2_pipeline_on_two_ptx_devices_places_each_layers_weights_on_its_stage() {
    const GIB: u64 = 1 << 30;
    let ptx = |device: u32| {
        let mut caps = default_caps_for(Backend::Nvptx);
        caps.memory_budget_bytes = Some(GIB);
        (
            DeviceId(device),
            Target::Gpu {
                backend: Backend::Nvptx,
                caps,
            },
        )
    };
    let targets = TargetSet::new(vec![ptx(0), ptx(1)]).expect("distinct devices");
    let graph = qwen2_decode(LAYERS);
    let program = compile_staged(
        &graph,
        &targets,
        &partition(DevicePlacement::Pipeline {
            after_layers: vec![0],
            devices: vec![DeviceId(0), DeviceId(1)],
        }),
        &options(),
    )
    .unwrap_or_else(|error| panic!("a two-device PTX pipeline compiles host-only: {error}"));

    let stages: Vec<_> = program.stages().collect();
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].1, DeviceId(0));
    assert_eq!(stages[1].1, DeviceId(1));

    let layer = |index: usize| -> Vec<String> {
        weight_names(&graph)
            .into_iter()
            .filter(|name| name.starts_with(&format!("w.l{index}.")))
            .collect()
    };
    assert!(!layer(0).is_empty(), "the trace names layer 0's weights");
    let first = weight_names(stages[0].2.graph());
    let mut first_expected: Vec<String> =
        layer(0).into_iter().chain(["w.embed".to_owned()]).collect();
    first_expected.sort();
    assert_eq!(
        first, first_expected,
        "stage 0 reads the embedding and exactly layer 0's weights"
    );
    let mut second_expected: Vec<String> = layer(1)
        .into_iter()
        .chain(layer(2))
        .chain(["w.final_norm".to_owned(), "w.head".to_owned()])
        .collect();
    second_expected.sort();
    assert_eq!(
        weight_names(stages[1].2.graph()),
        second_expected,
        "stage 1 reads layers 1 and 2 and the head"
    );
    let mut placed: Vec<String> = first
        .into_iter()
        .chain(weight_names(stages[1].2.graph()))
        .collect();
    placed.sort();
    assert_eq!(placed, weight_names(&graph), "every weight is placed once");

    assert!(program.links().is_empty(), "no hop stays on one device");
    assert!(!program.transfers().is_empty(), "the cut crosses values");
    assert_eq!(program.comm().rows.len(), program.transfers().len());
    for transfer in program.transfers() {
        assert_eq!((transfer.from, transfer.to), (stages[0].0, stages[1].0));
    }
    for row in &program.comm().rows {
        assert_eq!(row.kind, CollectiveKind::PointToPoint);
        assert_eq!(row.participants, [DeviceId(0), DeviceId(1)]);
    }
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
    let found = original.inputs().iter().find(|&&candidate| {
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
            Value::from(HostTensor::i32(
                meta.aval.shape.clone(),
                (0..numel).map(|row| i32::from(row == 0)).collect(),
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

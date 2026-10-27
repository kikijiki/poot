//! POOT-1395: the KV cache's storage dtype through the tracer, the planner and the oracle. A cache traced in a
//! two-byte dtype is stored narrow and read by the fused attention through its own lane (no widened copy of
//! the cache exists); f32 stays the exact reference and traces no narrowing at all; a backend that cannot
//! hold the dtype refuses it by name; and the oracle evaluates the narrowed state the device stores.

use poot_graph_ir::{Graph, OpKind, StateRole};
use poot_graph_plan::{CompileError, KernelChoice, PlanError, Program, Target, compile};
use poot_kernelgen::{KernelRequest, KvLane};
use poot_models::model::{KvDtype, LogitRows, StepShape, StepSpec};
use poot_models::test_support::dense::{Dense, Family, paged_step, step};
use poot_models::test_support::weight_map::{MappedModel, Step};
use poot_target::{AmdArch, Backend};
use poot_tensor::{DType, HostTensor};

#[path = "support/targets.rs"]
mod targets;

const CAPACITY: usize = 16;
const PROMPT: [i32; 5] = [3, 1, 4, 1, 5];

fn model() -> MappedModel {
    Dense::new(Family::Qwen3).model()
}

fn trace(m: &MappedModel, shape: StepShape) -> Graph {
    m.model
        .trace(
            &StepSpec::step(shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .expect("the family traces a step")
}

/// A prefill of the prompt and a decode step, contiguous and paged, each over a cache of `kv`.
fn shapes(kv: KvDtype) -> Vec<StepShape> {
    [
        step(1, PROMPT.len(), CAPACITY, LogitRows::Last),
        step(1, 1, CAPACITY, LogitRows::Last),
        paged_step(1, PROMPT.len(), CAPACITY, 2 * CAPACITY, LogitRows::Last),
        paged_step(1, 1, CAPACITY, 2 * CAPACITY, LogitRows::Last),
    ]
    .into_iter()
    .map(|shape| shape.with_kv_dtype(kv))
    .collect()
}

/// The dtype of every KV cache state of `graph` (the contiguous caches and the paged pools).
fn cache_dtypes(graph: &Graph) -> Vec<DType> {
    graph
        .state_pairs()
        .filter(|pair| matches!(pair.role, StateRole::Positional { .. }))
        .map(|pair| graph.aval(pair.input).dtype)
        .collect()
}

/// The lane of every attention dispatch of `program`.
fn attention_lanes(program: &Program) -> Vec<KvLane> {
    program
        .planned()
        .filter(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
        .flat_map(|(_, plan)| plan.dispatches())
        .filter_map(|dispatch| match dispatch.choice() {
            KernelChoice::Generated(KernelRequest::Attention(spec)) => Some(spec.kv),
            KernelChoice::Generated(KernelRequest::AttentionRows(rows)) => Some(rows.spec.kv),
            _ => None,
        })
        .collect()
}

fn gpu(backend: Backend) -> Target {
    Target::Gpu {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    }
}

/// An f16 cache is stored as f16 and read by the fused attention through the f16 lane on every GPU target,
/// prefill and decode, contiguous and paged: the write stores the f32 entries narrowed itself (no narrowing
/// `Cast` is traced) and the widening `Cast` the tracer writes on the read is part of the attention, so no
/// copy of the new entries or of the cache is left in the compiled graph. Mutation: drop the widening peel in
/// the attention matcher (`peel_widen` returning its input); the chain stays the decomposition and no
/// attention equation reads the cache.
#[test]
fn an_f16_cache_is_read_by_the_fused_attention_through_the_f16_lane() {
    let m = model();
    let targets = [
        gpu(Backend::AmdGcn(AmdArch::gfx1151())),
        gpu(Backend::Nvptx),
        gpu(Backend::SpirvVulkan),
    ];
    for shape in shapes(KvDtype::F16) {
        let graph = trace(&m, shape);
        let caches = cache_dtypes(&graph);
        assert!(!caches.is_empty(), "{shape:?}: the step carries caches");
        assert!(
            caches.iter().all(|&dtype| dtype == DType::F16),
            "{shape:?}: {caches:?}"
        );
        assert!(
            !graph
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Cast { to: DType::F16 })),
            "{shape:?}: the write narrows, no cast does"
        );
        for target in &targets {
            let program = compile(&graph, target, &targets::options())
                .unwrap_or_else(|e| panic!("{shape:?} on {:?}: {e}", target.id()));
            let lanes = attention_lanes(&program);
            assert!(
                !lanes.is_empty(),
                "{shape:?} on {:?}: an attention reads the cache",
                target.id()
            );
            assert!(
                lanes.iter().all(|&lane| lane == KvLane::F16),
                "{shape:?} on {:?}: {lanes:?}",
                target.id()
            );
            let widened = program.graph().eqns().iter().any(|eqn| {
                matches!(eqn.op, OpKind::Cast { to: DType::F32 })
                    && eqn.inputs.iter().any(|operand| match operand {
                        poot_graph_ir::Operand::Value(v) => {
                            program.graph().aval(*v).dtype == DType::F16
                        }
                        poot_graph_ir::Operand::Lit(_) => false,
                    })
            });
            assert!(
                !widened,
                "{shape:?} on {:?}: no f32 copy of the f16 cache is compiled",
                target.id()
            );
        }
    }
}

/// A bf16 cache compiles on a device with native two-byte lanes and is read through the bf16 lane; SPIR-V
/// has no bf16 type, so there the cache is refused by name, at planning, naming the bf16 equation and the
/// missing lowering. Mutation: drop the cast family's `(BF16, F32)` refusal for a device without two-byte
/// lanes; the widening read is admitted and the row goes red.
#[test]
fn a_bf16_cache_compiles_where_bf16_lanes_exist_and_is_refused_by_name_on_spirv() {
    let m = model();
    let decode = trace(
        &m,
        step(1, 1, CAPACITY, LogitRows::Last).with_kv_dtype(KvDtype::Bf16),
    );
    let rocm = compile(
        &decode,
        &gpu(Backend::AmdGcn(AmdArch::gfx1151())),
        &targets::options(),
    )
    .expect("ROCm has bf16 lanes");
    let lanes = attention_lanes(&rocm);
    assert!(
        !lanes.is_empty() && lanes.iter().all(|&lane| lane == KvLane::Bf16),
        "{lanes:?}"
    );
    let refusal = match compile(&decode, &gpu(Backend::SpirvVulkan), &targets::options()) {
        Err(CompileError::Plan(error)) => match *error {
            PlanError::Refused(refusal) => refusal,
            other => panic!("a typed refusal, not {other:?}"),
        },
        Err(other) => panic!("a planning refusal, not {other:?}"),
        Ok(_) => panic!("a bf16 cache on SPIR-V is refused"),
    };
    // The first bf16 equation the planner reaches is the cache's widening read.
    assert_eq!(refusal.op, OpKind::Cast { to: DType::F32 });
    assert_eq!(refusal.dtypes.operands, [DType::BF16]);
    assert_eq!(refusal.missing, poot_graph_plan::Capability::DtypeLowering);
}

/// An f32 cache is the exact reference: its graph holds no cast from or to a two-byte float and every cache
/// is f32, so it is the graph the tracer wrote before narrow storage existed. Mutation: make the tracer's
/// `cast_to` always emit a `Cast` (an identity f32 cast on the cache read); the row goes red.
#[test]
fn an_f32_cache_traces_no_narrowing() {
    let m = model();
    for shape in shapes(KvDtype::F32) {
        let graph = trace(&m, shape);
        assert!(
            cache_dtypes(&graph)
                .iter()
                .all(|&dtype| dtype == DType::F32)
        );
        let cache_casts = graph.eqns().iter().filter(|eqn| {
            matches!(eqn.op, OpKind::Cast { .. })
                && eqn.inputs.iter().any(|operand| match operand {
                    poot_graph_ir::Operand::Value(v) => {
                        graph.aval(*v).dtype == graph.aval(eqn.out).dtype
                    }
                    poot_graph_ir::Operand::Lit(_) => false,
                })
        });
        assert_eq!(cache_casts.count(), 0, "{shape:?}: no identity cast");
        assert!(
            !graph.eqns().iter().any(|eqn| matches!(
                eqn.op,
                OpKind::Cast {
                    to: DType::F16 | DType::BF16
                }
            )),
            "{shape:?}: nothing narrows"
        );
    }
}

/// The oracle evaluates the narrowed state, so an exact-token gate against it compares like with like: after
/// the prompt's prefill the first layer's f16 cache holds exactly the f32 cache's values rounded to f16 (its
/// keys and values are computed in f32 from the same embeddings either way; only their storage differs), a
/// later layer's within the drift the rounding feeds forward, and the logits, which read the rounded cache,
/// move off the f32 run's. Mutation: narrow to bf16 when f16 is asked
/// (`KvDtype::dtype` mapping `F16` to `DType::BF16`); the state is no longer the f16 rounding and the row
/// goes red.
#[test]
fn the_oracle_evaluates_the_cache_the_device_stores() {
    let m = model();
    let prefill = Step::new(&PROMPT, 0);
    let run = |kv: KvDtype| {
        let graph = trace(
            &m,
            step(1, PROMPT.len(), CAPACITY, LogitRows::Last).with_kv_dtype(kv),
        );
        let zeros: Vec<HostTensor> = graph
            .state()
            .map(|(input, _)| {
                HostTensor::zeros_of(graph.aval(input).dtype, graph.aval(input).shape.clone())
            })
            .collect();
        let (logits, state) = poot_eval::test_support::weight_map_oracle::eval_mapped(
            &graph,
            &m.store,
            &m.map,
            &prefill.slots(),
            &zeros,
        );
        (logits.as_f32().unwrap().to_vec(), state)
    };
    let (exact, exact_state) = run(KvDtype::F32);
    let (narrow, narrow_state) = run(KvDtype::F16);
    let traced = trace(&m, step(1, PROMPT.len(), CAPACITY, LogitRows::Last));
    let names: Vec<String> = traced
        .state()
        .map(|(input, _)| {
            traced
                .meta(input)
                .name()
                .expect("a named state")
                .to_string()
        })
        .collect();
    assert_eq!(exact_state.len(), narrow_state.len());
    let mut first_layer = 0;
    for ((name, f32_cache), f16_cache) in names.iter().zip(&exact_state).zip(&narrow_state) {
        assert_eq!(f16_cache.dtype(), DType::F16, "{name}");
        let f32_values = f32_cache.as_f32().expect("the f32 cache");
        let f16_values = f16_cache.to_f32().expect("the f16 cache's values");
        if name.starts_with("kv.l0.") {
            first_layer += 1;
            let rounded: Vec<u16> = f32_values
                .iter()
                .map(|&v| poot_quant::scalar::f32_to_f16(v))
                .collect();
            assert_eq!(
                f16_cache.as_half().expect("the f16 cache's words"),
                rounded.as_slice(),
                "{name}: the f16 cache is the f32 cache rounded"
            );
        } else {
            for (a, b) in f32_values.iter().zip(f16_values.iter()) {
                assert!(
                    (a - b).abs() <= 1e-2 * (1.0 + a.abs()),
                    "{name}: {a} against {b}"
                );
            }
        }
    }
    assert_eq!(
        first_layer, 2,
        "the first layer's key and value caches were compared"
    );
    let moved = poot_test_util::max_abs_error(&narrow, &exact);
    assert!(
        moved > 0.0 && moved < 1e-2,
        "the narrowed cache moves the logits, by little: {moved}"
    );
}

//! POOT-664: the AIE sequential schedule is a choice inside the ordinary planner; what the NPU lowering adds is
//! the shape ABI and the limit checks. Model-free.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::PathBuf;

use crate::aie_kernel_handle;
use poot_codegen::{KernelCache, Target as CodegenTarget};
use poot_graph_ir::op::RedOp;
use poot_graph_ir::{Builder, Graph, TensorType};
use poot_graph_plan::test_support::roomy_body_limits;
use poot_graph_plan::{
    AieAbiRefusal, Capability, KernelChoice, NpuTarget, PlanError, Planned, Target, plan_eqn_choice,
};
use poot_npu_artifacts::{
    AieArtifactError, AieBindError, AieBound, AieLimitRefusal, check_aie_bindings,
};
use poot_target::aie::RegionLimits;
use poot_target::{AieCaps, AieTargetId, Queried, TargetId};
use poot_tensor::DType;

use super::{aie_shape_abi, validate_linked_artifact};
use crate::test_support::{scripted_toolchain, single_tile_recipe};
use crate::{AieCompileError, compile_aie, npu_lowering};

const AIE: TargetId = TargetId::Aie(AieTargetId::Xdna2);

fn aie_caps() -> AieCaps {
    // SYNTHETIC limits: roomy enough for a [2,3] kernel, not a device measurement.
    AieCaps {
        target: AieTargetId::Xdna2,
        tile_local_bytes: Queried::Known(1 << 16),
        host_bindings: Queried::Known(8),
        instruction_bytes: Queried::Known(1 << 20),
        memory_budget_bytes: Queried::Known(1 << 30),
        region: RegionLimits {
            compute_tiles: Queried::Known(32),
            tile_input_channels: Queried::Known(2),
            tile_output_channels: Queried::Known(2),
            host_input_streams: Queried::Known(16),
            host_output_streams: Queried::Known(16),
        },
    }
}

/// A target nobody has measured: every bound unknown, so no plan that needs one is admitted.
fn unmeasured_caps() -> AieCaps {
    AieCaps {
        tile_local_bytes: Queried::Unknown,
        host_bindings: Queried::Unknown,
        instruction_bytes: Queried::Unknown,
        memory_budget_bytes: Queried::Unknown,
        region: RegionLimits {
            compute_tiles: Queried::Unknown,
            tile_input_channels: Queried::Unknown,
            tile_output_channels: Queried::Unknown,
            host_input_streams: Queried::Unknown,
            host_output_streams: Queried::Unknown,
        },
        ..aie_caps()
    }
}

fn limits() -> poot_kernelgen::BodyLimits {
    roomy_body_limits()
}

fn graph_of(build: impl FnOnce(&Builder) -> poot_graph_ir::Traced) -> Graph {
    let b = Builder::for_tests();
    let out = build(&b);
    b.finish(out).unwrap()
}

fn add_graph(a_shape: Vec<usize>, b_shape: Vec<usize>, dtype: DType) -> Graph {
    graph_of(|b| {
        let x = b.constant("a", TensorType::new(a_shape, dtype)).unwrap();
        let y = b.constant("b", TensorType::new(b_shape, dtype)).unwrap();
        b.binary(poot_graph_ir::BinOp::Add, x, y).unwrap()
    })
}

fn last_eqn(g: &Graph) -> &poot_graph_ir::Eqn {
    g.eqns().last().expect("the graph has an equation")
}

/// The ordinary planner entry for an AIE target carrying `caps`.
fn plan_aie_with(g: &Graph, caps: AieCaps) -> Result<Planned, PlanError> {
    let target = Target::Npu(NpuTarget::Aie {
        target: AieTargetId::Xdna2,
        caps,
    });
    let npu = npu_lowering();
    plan_eqn_choice(
        g,
        last_eqn(g),
        &target,
        &HashMap::new(),
        &limits(),
        Some(npu.lowering()),
    )
}

fn plan_aie(g: &Graph) -> Result<Planned, PlanError> {
    plan_aie_with(g, aie_caps())
}

fn aie_refusal(g: &Graph) -> Capability {
    match plan_aie(g) {
        Err(PlanError::Refused(refusal)) => {
            assert_eq!(refusal.target, AIE);
            refusal.missing
        }
        Err(other) => panic!("expected a typed refusal, got {other:?}"),
        Ok(_) => panic!("the equation must be refused for an AIE core"),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("poot-aie-plan-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The planner positive: `plan_eqn` itself plans an F32 `[2,3]` add for an AIE target, as the ordinary
/// elementwise request under the serialized schedule on the existing plan types (one dispatch, one
/// invocation, a one-thread workgroup). The compiler links its body against the ABI derived from that
/// request, the ordinary handle wraps it, and a fake loader sees three ordered arguments and core
/// lengths `[6, 6, 6]`. A fake loader does not claim numerical execution or hardware.
/// Mutation: refuse the elementwise family in `plan_aie_eqn` or give `build_serialized` a GPU grid; the
/// plan or the dispatch-shape assertion fails.
#[test]
fn plan_eqn_plans_an_f32_add_under_the_aie_serialized_schedule_and_it_links() {
    let g = add_graph(vec![2, 3], vec![2, 3], DType::F32);
    let planned = plan_aie(&g).unwrap();
    let dispatch = planned.plan.single().expect("one dispatch");
    assert_eq!(dispatch.body().workgroup_size, [1, 1, 1]);
    assert_eq!(dispatch.launch().grid, [1, 1, 1]);
    assert!(dispatch.launch().meta.is_empty());

    let abi = aie_shape_abi(&planned.choice).unwrap();
    assert_eq!(abi.core_lengths(), vec![6, 6, 6]);
    // The ABI is read from the plan's own generation: the dimensions are the graph's, in order.
    for binding in abi.bindings() {
        assert_eq!(binding.dims(), &[2, 3]);
    }

    let dir = scratch("positive");
    let tc = scripted_toolchain(&dir, true);
    let cache = KernelCache::at(dir.join("cache"), CodegenTarget::AieCore);
    let compiled = compile_aie(
        &[dispatch.body()],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        NonZeroU64::new(1 << 20).unwrap(),
    )
    .unwrap();
    validate_linked_artifact(&compiled.artifact, &aie_caps()).unwrap();
    let artifact = compiled.artifact.clone();
    let handle = aie_kernel_handle(compiled.artifact);
    assert_eq!(handle.target(), AIE);
    assert_eq!(handle.args().len(), 3);

    let dims = [2u64, 3];
    let bound = |writable| AieBound {
        storage: poot_target::BufferStorage::f32(),
        dims: &dims,
        writable,
    };
    assert_eq!(
        check_aie_bindings(&artifact, &[bound(false), bound(false), bound(true)]),
        Ok(())
    );
    assert_eq!(
        check_aie_bindings(&artifact, &[bound(true), bound(false), bound(false)]),
        Err(AieBindError::ReadOnlyWritten { index: 2 })
    );
}

/// A unary F32 equation plans for an AIE target as the ordinary request under the serialized schedule:
/// two bindings (the operand it reads, the output it writes), one loop over every element, and the op's
/// numerical contract carried in the choice, so the plan key tells `exp` from `tanh`.
/// Mutation: give `exp` the `tanh` contract in `AieOp::contract`; the contract assertion fails, and the
/// two plan keys no longer differ by more than the request.
#[test]
fn plan_eqn_plans_a_unary_f32_under_its_numerical_contract() {
    let unary = |op| {
        graph_of(|b| {
            let x = b.constant("x", TensorType::f32(vec![3, 4])).unwrap();
            b.unary(op, x).unwrap()
        })
    };
    let mut keys = Vec::new();
    for (op, id) in [
        (poot_graph_ir::UnOp::Exp, "f32-exp-software-v1"),
        (poot_graph_ir::UnOp::Tanh, "f32-tanh-software-v1"),
        (poot_graph_ir::UnOp::Neg, "f32-neg-v1"),
    ] {
        let planned = plan_aie(&unary(op)).unwrap();
        let KernelChoice::Serialized {
            contract, bindings, ..
        } = &planned.choice
        else {
            panic!("{op:?} is planned as a serialized schedule");
        };
        assert_eq!(contract.id, id);
        assert_eq!(
            bindings.iter().map(|b| b.writes).collect::<Vec<_>>(),
            [false, true]
        );
        // The buffers are bound at the graph's own dimensions.
        assert!(
            bindings
                .iter()
                .all(|b| b.dims == [3, 4] && b.accessed == 12)
        );
        let dispatch = planned.plan.single().expect("one dispatch");
        assert_eq!(dispatch.body().workgroup_size, [1, 1, 1]);
        assert_eq!(
            aie_shape_abi(&planned.choice).unwrap().core_lengths(),
            [12, 12]
        );
        keys.push(dispatch.key().to_string());
    }
    keys.sort();
    keys.dedup();
    assert_eq!(keys.len(), 3, "each op is its own kernel");
}

/// A last-axis reduction and a gather plan for an AIE target as the ordinary requests under the
/// serialized schedule, bound at the graph's own dimensions: the reduction's output keeps its unit
/// axis, the gather's index is an I32 buffer in an arena of its own, and the plan's contract is the
/// op's. Mutation: bind the gather's index as `ElementKind::F32` in `serialized_movement`, or take the
/// dimensions from the request instead of the graph in `build_serialized`; the assertions fail.
#[test]
fn plan_eqn_plans_a_row_reduction_and_a_gather_with_the_graphs_dimensions() {
    use poot_target::ElementKind;
    let sum = graph_of(|b| {
        let x = b.constant("x", TensorType::f32(vec![5, 8])).unwrap();
        b.reduce(RedOp::Sum, x, 1, true).unwrap()
    });
    let planned = plan_aie(&sum).unwrap();
    let KernelChoice::Serialized {
        contract, bindings, ..
    } = &planned.choice
    else {
        panic!("a row sum is planned as a serialized schedule");
    };
    assert_eq!(contract.id, "f32-row-sum-sequential-v1");
    let seen: Vec<_> = bindings.iter().map(|b| (&b.dims[..], b.accessed)).collect();
    assert_eq!(seen, [(&[5u64, 8][..], 40), (&[5u64, 1][..], 5)]);

    let gather = graph_of(|b| {
        let table = b.constant("table", TensorType::f32(vec![10, 3])).unwrap();
        let ids = b
            .constant("ids", TensorType::new(vec![4], DType::I32))
            .unwrap();
        b.gather(table, 0, ids).unwrap()
    });
    let planned = plan_aie(&gather).unwrap();
    let KernelChoice::Serialized {
        contract, bindings, ..
    } = &planned.choice
    else {
        panic!("a gather is planned as a serialized schedule");
    };
    assert_eq!(contract.id, "gather-copy-v1");
    let seen: Vec<_> = bindings
        .iter()
        .map(|b| (&b.dims[..], b.element, b.writes))
        .collect();
    assert_eq!(
        seen,
        [
            (&[10u64, 3][..], ElementKind::F32, false),
            (&[4u64][..], ElementKind::I32, false),
            (&[4u64, 3][..], ElementKind::F32, true),
        ]
    );
    let abi = aie_shape_abi(&planned.choice).unwrap();
    let arenas: Vec<_> = abi
        .arenas()
        .iter()
        .map(|a| (a.access(), a.element(), a.words()))
        .collect();
    assert_eq!(
        arenas,
        [
            (poot_target::ArgAccess::Read, ElementKind::F32, 30),
            (poot_target::ArgAccess::Read, ElementKind::I32, 4),
            (poot_target::ArgAccess::Write, ElementKind::F32, 12),
        ]
    );
}

/// The same ordinary request path refuses what has no serialized schedule, each with the planner's own
/// typed refusal: a family the generator has no loop for, a broadcast operand, a non-F32 element, an
/// op not proved on the core. Mutation: make `serialized_schedule` fall back to the GPU generator, or
/// drop its operand-shape check; the matching row returns `Ok`.
#[test]
fn plan_eqn_refuses_for_aie_what_has_no_serialized_schedule() {
    let kernel_gen = |capability: Capability| match capability {
        Capability::KernelGen(poot_kernelgen::KernelGenError::Unsupported { request, reason }) => {
            (request, reason)
        }
        other => panic!("expected a generator refusal, got {other:?}"),
    };
    let broadcast = add_graph(vec![2, 3], vec![1, 3], DType::F32);
    assert!(kernel_gen(aie_refusal(&broadcast)).1.contains("broadcast"));
    let ints = add_graph(vec![2, 3], vec![2, 3], DType::I32);
    assert_eq!(kernel_gen(aie_refusal(&ints)).0, "serialized_schedule");
    // A reduction over a leading axis has no row to loop over.
    let leading = graph_of(|b| {
        let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
        b.reduce(RedOp::Sum, x, 0, false).unwrap()
    });
    assert_eq!(
        aie_refusal(&leading),
        Capability::NonLastAxisReduce { axis: 0 }
    );
    // A half-precision reduction is not a row of F32 words.
    let bf16_rows = graph_of(|b| {
        let x = b
            .constant("x", TensorType::new(vec![2, 3], DType::BF16))
            .unwrap();
        b.reduce(RedOp::Sum, x, 1, false).unwrap()
    });
    assert_eq!(kernel_gen(aie_refusal(&bf16_rows)).0, "serialized_schedule");
    // A data movement other than a gather reads GPU capabilities.
    let transpose = graph_of(|b| {
        let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
        b.transpose(x, vec![1, 0]).unwrap()
    });
    assert_eq!(aie_refusal(&transpose), Capability::GpuOnly);
    // A collective is a cut between stage programs on every target.
    let reduce = graph_of(|b| {
        let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
        b.all_reduce(x, RedOp::Sum, 0).unwrap()
    });
    assert!(matches!(
        aie_refusal(&reduce),
        Capability::CollectiveInStage
    ));
}

/// A lowering that reads GPU capabilities (a shipped GPU template, a contraction) is a typed
/// [`Capability::GpuOnly`] refusal for an AIE target, through the ordinary planner, on a deliberately
/// unsupported fixture: the AIE target has no `DeviceCaps` to read, and the refusal names the target.
/// Mutation: dispatch the family to its GPU planner for an AIE target (give `plan_aie_eqn` a `matmul`
/// arm that plans with a fixture `DeviceCaps`); the contraction is planned as a GPU kernel and the row
/// returns `Ok`.
#[test]
fn a_gpu_only_lowering_is_a_typed_refusal_for_an_aie_target() {
    let template = graph_of(|b| {
        let seed = b
            .constant("seed", TensorType::new(vec![2], DType::I32))
            .unwrap();
        b.random_uniform(seed, 4).unwrap()
    });
    assert_eq!(aie_refusal(&template), Capability::GpuOnly);
    let contraction = graph_of(|b| {
        let x = b.constant("x", TensorType::f32(vec![2, 3])).unwrap();
        let w = b.constant("w", TensorType::f32(vec![3, 4])).unwrap();
        b.matmul(x, w).unwrap()
    });
    assert_eq!(aie_refusal(&contraction), Capability::GpuOnly);
    // The expert grouping of the routed MoE rows is a GPU kernel too: refused, never run unsorted.
    let grouping = graph_of(|b| {
        let ids = b
            .constant("ids", TensorType::new(vec![6], DType::I32))
            .unwrap();
        b.expert_group(ids, 4).unwrap()
    });
    assert_eq!(aie_refusal(&grouping), Capability::GpuOnly);
}

/// The planner finalizes an AIE plan against the target's own limits: the shape ABI it derives from the
/// plan's choice lays its three bindings out in two host arenas (the reads, the write), an unmeasured
/// limit refuses (never a guess), and a measured one the ABI exceeds refuses with both numbers.
/// Mutation: drop `check_shape_abi_limits` from `finalize_aie`; both rows plan.
#[test]
fn the_planner_refuses_a_shape_abi_the_target_cannot_bind() {
    let add = add_graph(vec![2, 3], vec![2, 3], DType::F32);
    let refusal = |caps| match plan_aie_with(&add, caps) {
        Err(PlanError::Refused(refusal)) => refusal.missing,
        other => panic!("expected a typed refusal, got {:?}", other.map(|_| ())),
    };
    assert_eq!(
        refusal(unmeasured_caps()),
        Capability::AieLimit(AieLimitRefusal::Unknown {
            limit: "host buffers"
        })
    );
    let one = AieCaps {
        host_bindings: Queried::Known(1),
        ..aie_caps()
    };
    assert_eq!(
        refusal(one),
        Capability::AieLimit(AieLimitRefusal::Exceeded {
            resource: "host buffers",
            needed: 2,
            available: 1
        })
    );
    assert!(plan_aie(&add).is_ok(), "control: the roomy target plans it");
}

/// Finalization of the linked artifact reads the target's limits: an unmeasured limit refuses (never
/// a guess), and a reservation past a measured one refuses with both numbers. Mutation: treat
/// `Unknown` as unbounded, or drop the `needed > available` comparison.
#[test]
fn a_linked_artifact_is_refused_against_unknown_and_exceeded_target_limits() {
    let add = add_graph(vec![2, 3], vec![2, 3], DType::F32);
    let planned = plan_aie(&add).unwrap();
    let body = planned.plan.single().unwrap().body().clone();
    let abi = aie_shape_abi(&planned.choice).unwrap();
    let dir = scratch("limits");
    let compiled = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &scripted_toolchain(&dir, true),
        &KernelCache::at(dir.join("cache"), CodegenTarget::AieCore),
        NonZeroU64::new(1 << 20).unwrap(),
    )
    .unwrap();
    let artifact = &compiled.artifact;
    assert_eq!(validate_linked_artifact(artifact, &aie_caps()), Ok(()));
    let unmeasured = unmeasured_caps();
    assert_eq!(
        validate_linked_artifact(artifact, &unmeasured),
        Err(AieLimitRefusal::Unknown {
            limit: "tile-local bytes"
        })
    );
    // Three FIFOs of 6 words at depth 2.
    let tiny = AieCaps {
        tile_local_bytes: Queried::Known(143),
        ..aie_caps()
    };
    assert_eq!(
        validate_linked_artifact(artifact, &tiny),
        Err(AieLimitRefusal::Exceeded {
            resource: "tile-local bytes",
            needed: 144,
            available: 143
        })
    );
    let one_buffer = AieCaps {
        host_bindings: Queried::Known(1),
        ..aie_caps()
    };
    assert_eq!(
        validate_linked_artifact(artifact, &one_buffer),
        Err(AieLimitRefusal::Exceeded {
            resource: "host buffers",
            needed: 2,
            available: 1
        })
    );
    let no_room = AieCaps {
        instruction_bytes: Queried::Known(3),
        ..aie_caps()
    };
    assert_eq!(
        validate_linked_artifact(artifact, &no_room),
        Err(AieLimitRefusal::Exceeded {
            resource: "instruction bytes",
            needed: 4,
            available: 3
        })
    );
}

/// A planned AIE kernel without the IRON driver fails the ordinary compile with the named stage and no
/// entry: the planner admitted the plan, the toolchain boundary is the compiler's.
#[test]
fn a_planned_aie_kernel_without_the_iron_driver_is_a_typed_compile_error() {
    let add = add_graph(vec![2, 3], vec![2, 3], DType::F32);
    let planned = plan_aie(&add).unwrap();
    let body = planned.plan.single().unwrap().body().clone();
    let abi = aie_shape_abi(&planned.choice).unwrap();
    let dir = scratch("no-iron");
    let cache = KernelCache::at(dir.join("cache"), CodegenTarget::AieCore);
    let error = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &scripted_toolchain(&dir, false),
        &cache,
        NonZeroU64::new(1 << 20).unwrap(),
    )
    .map(|_| ())
    .unwrap_err();
    assert!(
        matches!(
            error,
            AieCompileError::ToolchainMissing {
                stage: crate::AieStage::Iron,
                ..
            }
        ),
        "got {error:?}"
    );
    let dir = cache
        .dir()
        .expect("the cache sits at an explicit directory");
    assert!(!dir.exists() || std::fs::read_dir(dir).unwrap().next().is_none());
}

/// An AIE core binds each tensor as a flat word buffer of at least one element; a zero-element tensor has
/// no such buffer, and the shape ABI the planner derives refuses it, naming the binding.
///
/// Mutation: make the `dims.contains(&0)` check in `AieShapeAbi::new` never fire; the empty add plans for
/// the core and this row goes red (the `AieAbi` match fails).
#[test]
fn a_zero_element_tensor_on_an_aie_core_is_an_aie_abi_refusal() {
    let add = |rows: usize| add_graph(vec![rows, 4], vec![rows, 4], DType::F32);
    plan_aie(&add(2)).expect("a non-empty add plans for the core");
    let Err(PlanError::Refused(refusal)) = plan_aie(&add(0)) else {
        panic!("a zero-element add must be refused");
    };
    assert!(matches!(
        refusal.missing,
        Capability::AieAbi(AieAbiRefusal::Abi(AieArtifactError::ZeroDimension {
            index: 0
        }))
    ));
}

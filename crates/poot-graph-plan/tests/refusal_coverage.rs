//! POOT-1289: each `Capability` refusal the variant-coverage list (scripts/variant-coverage.tsv) recorded
//! as a gap, made to fire through the planner's per-equation entry (`plan_eqn_choice`, the one entry
//! `compile` itself plans through). Model-free.
//!
//! `IndexOperand` and `FusedStep` are not reachable through the `Builder` (it takes an unsigned index and
//! never builds a fused region; the fusion pass admits only ops the kernel computes), so their graphs are
//! forged with the test-support `derive` helper: the planner is the second line of defence for a graph some
//! other path built. `OperandCount` needs no forged graph: `plan_eqn_choice` takes the equation apart from
//! its graph, and an equation's operand list is public.

use std::collections::HashMap;

use poot_graph_ir::UnOp;
use poot_graph_ir::op::{FusedOp, FusedOperand, FusedRegion, FusedStep, LocalId};
use poot_graph_ir::test_support::composites;
use poot_graph_ir::test_support::graph_fixtures::derive;
use poot_graph_ir::{
    BinOp, BlockTopKAttrs, Builder, Eqn, Graph, OpKind, Operand, Scalar, TensorType,
};
use poot_graph_plan::test_support::roomy_body_limits;
use poot_graph_plan::{
    AieAbiRefusal, Capability, FusedStepGap, PlanError, Target, plan_eqn_choice,
};
use poot_runtime_common::AieArtifactError;
use poot_target::DeviceCaps;
use poot_target::aie::RegionLimits;
use poot_target::{AieCaps, AieTargetId, Queried};
use poot_tensor::DType;

fn wgpu() -> Target {
    Target::Gpu {
        backend: poot_target::Backend::SpirvVulkan,
        caps: DeviceCaps::wgpu_rdna3_igpu(),
    }
}

/// The capability the planner refuses the last equation of `g` for, on `target`.
fn refusal_on(g: &Graph, target: &Target) -> Capability {
    let eqn = g.eqns().last().expect("the graph has an equation");
    refusal_of(g, eqn, target)
}

/// The capability the planner refuses `eqn` of `g` for, on `target`.
fn refusal_of(g: &Graph, eqn: &Eqn, target: &Target) -> Capability {
    match plan_eqn_choice(g, eqn, target, &HashMap::new(), &roomy_body_limits()) {
        Err(PlanError::Refused(refusal)) => refusal.missing,
        Err(other) => panic!("expected a typed refusal, got {other:?}"),
        Ok(_) => panic!("the equation must be refused"),
    }
}

/// The capability the planner refuses the last equation of `g` for, or `None` when it plans.
fn refusal_or_ok(g: &Graph) -> Option<Capability> {
    let eqn = g.eqns().last().expect("the graph has an equation");
    match plan_eqn_choice(g, eqn, &wgpu(), &HashMap::new(), &roomy_body_limits()) {
        Err(PlanError::Refused(refusal)) => Some(refusal.missing),
        Err(other) => panic!("expected a typed refusal or a plan, got {other:?}"),
        Ok(_) => None,
    }
}

fn refusal(g: &Graph) -> Capability {
    refusal_on(g, &wgpu())
}

/// A standalone `PackedDequant` has no kernel (the escape gate refuses it in `compile`; one that reaches
/// planning is refused by name).
///
/// Mutation: refuse the standalone `PackedDequant` in `packed_dequant::plan` with `Capability::GpuOnly`
/// instead; this row goes red (`left: GpuOnly, right: ImportedKernelPlanning`).
#[test]
fn a_standalone_packed_dequant_is_an_imported_kernel_planning_refusal() {
    let descriptor =
        poot_quant::PackedWeight::try_new(poot_quant::format::WeightFormat::Q8_0, [4, 64]).unwrap();
    let b = Builder::for_tests();
    let sources: Vec<_> = poot_graph_ir::packed_source_constants("table", descriptor)
        .into_iter()
        .map(|(name, tensor_type)| b.constant(name.as_str(), tensor_type).unwrap())
        .collect();
    let table = b.packed_dequant(&sources, descriptor).unwrap();
    let g = b.finish(table).unwrap();
    assert_eq!(refusal(&g), Capability::ImportedKernelPlanning);
}

/// An I32 remainder by a literal zero has no exact kernel (the remainder is unsigned and a zero divisor
/// has no result): the planner refuses it by name instead of emitting a division by zero.
///
/// Mutation: change the `RemU` by `Lit(I32(0))` match in `elementwise::plan` to another literal; the
/// equation plans and this row goes red (`the equation must be refused`).
#[test]
fn an_i32_remainder_by_literal_zero_is_an_exact_i32_op_refusal() {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![4], DType::I32))
        .unwrap();
    let rem = b
        .binary_scalar(BinOp::RemU, x, Scalar::I32(0))
        .expect("the Builder admits an I32 remainder by a literal");
    let g = b.finish(rem).unwrap();
    assert_eq!(refusal(&g), Capability::ExactI32Op);
    // The control: the same remainder by a nonzero literal plans.
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![4], DType::I32))
        .unwrap();
    let rem = b.binary_scalar(BinOp::RemU, x, Scalar::I32(3)).unwrap();
    let g = b.finish(rem).unwrap();
    let eqn = g.eqns().last().unwrap();
    plan_eqn_choice(&g, eqn, &wgpu(), &HashMap::new(), &roomy_body_limits())
        .expect("a nonzero literal divisor plans");
}

/// A dynamic-update-slice whose literal start index is negative has no static window. The Builder takes a
/// `usize` index, so the graph is a well-formed update whose literal is replaced after the fact with
/// `derive` (the graph check does not read the value of an index literal).
///
/// Mutation: let `usize::try_from(*index)` in `movement::plan` fall back to `0` instead of refusing; the
/// update plans at the wrong window and this row goes red (`the equation must be refused`).
#[test]
fn a_negative_literal_update_index_is_an_index_operand_refusal() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4, 3])).unwrap();
    let update = b.constant("u", TensorType::f32(vec![1, 3])).unwrap();
    let written = b.dynamic_update_slice(x, update, 1, 0).unwrap();
    let g = b.finish(written).unwrap();
    assert_eq!(
        refusal_or_ok(&g),
        None,
        "the control update at index 1 plans"
    );
    let forged = derive(&g, |draft| {
        draft.eqns_mut()[0].inputs[2] = Operand::Lit(Scalar::I32(-1));
    })
    .expect("the graph check does not read an index literal's value");
    assert_eq!(refusal(&forged), Capability::IndexOperand);
}

/// The block selection runs as one workgroup of `BLOCK_TOP_K_LANES` lanes: a device whose workgroup is
/// smaller refuses with the lanes it needed and the limits it has, never a quadratic fallback.
///
/// Mutation: make the `fits` check in `block_top_k::plan` never fire; the request reaches the generic
/// resource check and this row goes red (`expected BlockTopKWorkgroup, got KernelResources(StaticLds ..)`).
#[test]
fn a_device_below_the_block_top_k_workgroup_is_refused_with_both_numbers() {
    let b = Builder::for_tests();
    let scores = b
        .constant("scores", TensorType::f32(vec![1, 1, 2, 8]))
        .unwrap();
    let pos = b
        .constant("pos", TensorType::new(vec![1, 2], DType::I32))
        .unwrap();
    let selection =
        composites::block_top_k(&b, BlockTopKAttrs { ratio: 2, k: 2 }, scores, pos).unwrap();
    let g = b.finish(selection.packed).unwrap();
    let mut caps = DeviceCaps::wgpu_rdna3_igpu();
    caps.lds_bytes = 16;
    let target = Target::Gpu {
        backend: poot_target::Backend::SpirvVulkan,
        caps,
    };
    let selection_eqn = g
        .eqns()
        .iter()
        .find(|eqn| matches!(eqn.op, OpKind::BlockTopK(_)))
        .expect("the graph selects blocks");
    match refusal_of(&g, selection_eqn, &target) {
        Capability::BlockTopKWorkgroup {
            lds_needed,
            lds_bytes,
            ..
        } => assert_eq!((lds_bytes, lds_needed > 16), (16, true)),
        other => panic!("expected BlockTopKWorkgroup, got {other:?}"),
    }
}

/// The same refusal through `compile`, from the composition a tracer states: the selection pass forms the
/// `BlockTopK` and the small device refuses that op with its workgroup numbers, not the first primitive of the
/// quadratic rank chain it replaced (which no device lowers). Mutation: make `compile` run the selection pass
/// as a revertible rewrite (`pipeline.run`); the refusal reverts to the composition and this row goes red
/// (`expected BlockTopKWorkgroup, got DtypeLowering` on the chain's `Cast`).
#[test]
fn a_traced_block_selection_on_a_small_device_is_refused_as_the_block_top_k() {
    let b = Builder::for_tests();
    let scores = b
        .constant("scores", TensorType::f32(vec![1, 1, 2, 8]))
        .unwrap();
    let pos = b
        .slot_ranged(
            poot_graph_ir::Slot::Pos,
            TensorType::new(vec![1, 2], DType::I32),
            poot_graph_ir::IndexRange::below(16).unwrap(),
        )
        .unwrap();
    let packed = poot_graph_ir::ops::block_top_k_packed(&b, scores, pos, 2, 2).unwrap();
    let g = b.finish(packed).unwrap();
    let mut caps = DeviceCaps::wgpu_rdna3_igpu();
    caps.lds_bytes = 16;
    let target = Target::Gpu {
        backend: poot_target::Backend::SpirvVulkan,
        caps,
    };
    let options = poot_graph_plan::CompileOptions::new(poot_graph_plan::CompileLimits::STANDARD);
    let refused = poot_graph_plan::compile(&g, &target, &options).map(|_| ());
    let Err(poot_graph_plan::CompileError::Plan(plan)) = refused else {
        panic!("expected the block top-k's refusal, got {refused:?}")
    };
    match *plan {
        PlanError::Refused(refusal) => assert!(
            matches!(refusal.missing, Capability::BlockTopKWorkgroup { .. }),
            "expected BlockTopKWorkgroup, got {:?}",
            refusal.missing
        ),
        other => panic!("expected a typed refusal, got {other:?}"),
    }
}

/// A binary equation handed to the planner with three operands: the planner names the count it expected
/// and the count it got, rather than indexing past a slice pattern. The equation is the graph's own add
/// with its first operand repeated (the graph check never sees it: the planner entry takes the equation
/// separately from the graph).
///
/// Mutation: change the `OperandCount` arm of the operand match in `elementwise::plan` to refuse with
/// `Capability::DtypeLowering`; this row goes red (`left: DtypeLowering`).
#[test]
fn a_binary_equation_with_three_operands_is_an_operand_count_refusal() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![4])).unwrap();
    let y = b.constant("y", TensorType::f32(vec![4])).unwrap();
    let sum = b.binary(BinOp::Add, x, y).unwrap();
    let add = b.finish(sum).unwrap();
    let mut eqn = add.eqns()[0].clone();
    eqn.inputs.push(eqn.inputs[0]);
    assert_eq!(
        refusal_of(&add, &eqn, &wgpu()),
        Capability::OperandCount {
            expected: 2,
            actual: 3
        }
    );
}

/// A float fused region holds the ops its kernel computes; a narrowing `Cast` is not one of them (the fusion
/// pass never admits it, so no compile produces this region). A graph some other path built still meets a typed
/// refusal naming the step's op, not a kernel that silently skips it. The graph is an add whose operation
/// is replaced by a two-step region.
///
/// Mutation: make the final `FusedOp::Cast { .. }` arm of `map_fused_op` return `kg::FusedScalarOp::Widen`; the
/// region plans and this row goes red (`the equation must be refused`).
#[test]
fn a_float_region_step_the_kernel_cannot_compute_is_a_fused_step_refusal() {
    let region = |first: FusedOp| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 8])).unwrap();
        let y = b.constant("y", TensorType::f32(vec![2, 8])).unwrap();
        let sum = b.binary(BinOp::Add, x, y).unwrap();
        let add = b.finish(sum).unwrap();
        derive(&add, |draft| {
            draft.eqns_mut()[0].op = OpKind::Fused(FusedRegion {
                n_inputs: 2,
                steps: vec![
                    FusedStep {
                        op: first,
                        inputs: vec![FusedOperand::Local(LocalId::new(0))],
                    },
                    FusedStep {
                        op: FusedOp::Binary(BinOp::Add),
                        inputs: vec![
                            FusedOperand::Local(LocalId::new(0)),
                            FusedOperand::Local(LocalId::new(1)),
                        ],
                    },
                ],
                output: LocalId::new(3),
                pack: Vec::new(),
            });
        })
        .expect("a region of pointwise steps is a well-formed graph")
    };
    assert_eq!(
        refusal_or_ok(&region(FusedOp::Unary(UnOp::Neg))),
        None,
        "a negation plans"
    );
    let narrowing = FusedOp::Cast { to: DType::F16 };
    assert_eq!(
        refusal(&region(narrowing)),
        Capability::FusedStep(FusedStepGap::OpInFloatRegion(narrowing))
    );
}

/// An AIE core binds each tensor as a flat word buffer of at least one element; a zero-element tensor has
/// no such buffer, and the shape ABI the planner derives refuses it, naming the binding.
///
/// Mutation: make the `dims.contains(&0)` check in `AieShapeAbi::new` never fire; the empty add plans for
/// the core and this row goes red (the `AieAbi` match fails).
#[test]
fn a_zero_element_tensor_on_an_aie_core_is_an_aie_abi_refusal() {
    let target = Target::Aie {
        target: AieTargetId::Xdna2,
        caps: AieCaps {
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
        },
    };
    let add = |rows: usize| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![rows, 4])).unwrap();
        let y = b.constant("y", TensorType::f32(vec![rows, 4])).unwrap();
        let sum = b.binary(BinOp::Add, x, y).unwrap();
        b.finish(sum).unwrap()
    };
    let control = add(2);
    plan_eqn_choice(
        &control,
        control.eqns().last().unwrap(),
        &target,
        &HashMap::new(),
        &roomy_body_limits(),
    )
    .expect("a non-empty add plans for the core");
    assert!(matches!(
        refusal_on(&add(0), &target),
        Capability::AieAbi(AieAbiRefusal::Abi(AieArtifactError::ZeroDimension {
            index: 0
        }))
    ));
}

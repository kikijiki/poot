//! POOT-1164: the Intel NPU `Device` under the shared `Engine`, on the real device.
//!
//! Every row here runs the ordinary route: the planner lowers an F16 graph to OpenVINO opset ops for the
//! Intel NPU target, `poot-codegen` emits the IR, the engine hands the region to the device, the driver's
//! vendor graph compiler builds the blob and the NPU runs it through Level Zero. A box without the NPU or the
//! driver stack fails these rows when `POOT_REQUIRE_INTEL_NPU=1`; it never passes on them. Rows take
//! `/tmp/poot-vpu.lock` so two runs never share the device.
//!
//! ```text
//! nix develop .#npu-intel -c cargo nextest run -p poot-npu --features vpu --test-threads=1
//! ```

#![cfg(feature = "vpu")]

mod vpu_common;

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::Device;
use poot_executor_parity::Fixture;
use poot_graph_ir::{
    BinOp, Builder, Graph, Outputs, RedOp, Scalar, Slot, StateRole, TensorType, UnOp,
};
use poot_quant::weights::WeightStore;
use poot_tensor::DType;

use vpu_common::{Row, assert_matches, constant, npu, refusal_of, store, x_step};

fn f16(shape: &[usize]) -> TensorType {
    TensorType::new(shape.to_vec(), DType::F16)
}

/// `a + b` over two `[2, 3]` F16 constants.
fn add_consts() -> Row {
    let b = Builder::for_tests();
    let x = b.constant("a", f16(&[2, 3])).unwrap();
    let y = b.constant("b", f16(&[2, 3])).unwrap();
    let out = b.binary(BinOp::Add, x, y).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("a", &[2, 3], |i| i as f32 + 1.0),
        constant("b", &[2, 3], |i| 10.0 * (i as f32 + 1.0)),
    ]);
    Row {
        fixture: Fixture {
            name: "add_consts",
            graph,
            store,
            steps: vec![Vec::new(), Vec::new()],
        },
        exact: true,
    }
}

/// `x * w + b` over a `[4, 8]` F16 activation slot and two constants: three steps with different `x`, so the
/// recording is replayed over inputs the device had not seen when it recorded.
fn scale_shift_slot() -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[4, 8])).unwrap();
    let w = b.constant("w", f16(&[4, 8])).unwrap();
    let bias = b.constant("bias", f16(&[4, 8])).unwrap();
    let scaled = b.binary(BinOp::Mul, x, w).unwrap();
    let out = b.binary(BinOp::Add, scaled, bias).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[
        constant("w", &[4, 8], |i| (i % 7) as f32 - 3.0),
        constant("bias", &[4, 8], |i| (i % 5) as f32),
    ]);
    let steps = (0..3)
        .map(|s| x_step("x", &[4, 8], move |i| ((i * (s + 2)) % 11) as f32 - 5.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "scale_shift_slot",
            graph,
            store,
            steps,
        },
        exact: true,
    }
}

/// A carried running sum: `state' = state + x`, output `state'`. Every step reads the state the previous
/// step committed, so a dropped commit or a stale read shows in the output.
fn running_sum_state() -> Row {
    let b = Builder::for_tests();
    let state = b
        .state_input("acc", f16(&[2, 4]), StateRole::Recurrent)
        .unwrap();
    let x = b.slot_named(Slot::Activation, "x", f16(&[2, 4])).unwrap();
    let next = b.binary(BinOp::Add, state, x).unwrap();
    let graph: Graph = b
        .finish(Outputs::new(next).with_state(&[(state, next)]))
        .unwrap();
    let steps = (0..4)
        .map(|s| x_step("x", &[2, 4], move |i| (i + 1) as f32 * (s + 1) as f32))
        .collect();
    Row {
        fixture: Fixture {
            name: "running_sum_state",
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact: true,
    }
}

/// `exp(x)` over a `[3, 5]` slot: a transcendental, so compared within FP16 tolerance, not bit for bit.
fn exp_slot() -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[3, 5])).unwrap();
    let out = b.unary(UnOp::Exp, x).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..2)
        .map(|s| x_step("x", &[3, 5], move |i| ((i + 4 * s) % 9) as f32 * 0.25 - 1.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "exp_slot",
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact: false,
    }
}

/// Row sums over a `[4, 6]` slot, reduced along the last axis and along the first.
fn reduce_sum_rows() -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[4, 6])).unwrap();
    let rows = b.reduce(RedOp::Sum, x, 1, false).unwrap();
    let cols = b.reduce(RedOp::Sum, x, 0, false).unwrap();
    let total_rows = b.reduce(RedOp::Sum, rows, 0, false).unwrap();
    let total_cols = b.reduce(RedOp::Sum, cols, 0, false).unwrap();
    let out = b.binary(BinOp::Add, total_rows, total_cols).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..2)
        .map(|s| x_step("x", &[4, 6], move |i| ((i * (s + 3)) % 7) as f32 - 3.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "reduce_sum_rows",
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact: true,
    }
}

/// `x @ w` over a `[3, 8]` slot and a `[8, 5]` constant, small integers throughout, so every product and
/// partial sum is exact in FP16.
fn matmul_const() -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[3, 8])).unwrap();
    let w = b.constant("w", f16(&[8, 5])).unwrap();
    let out = b.matmul(x, w).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[constant("w", &[8, 5], |i| (i % 5) as f32 - 2.0)]);
    let steps = (0..2)
        .map(|s| x_step("x", &[3, 8], move |i| ((i + s) % 4) as f32 - 1.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "matmul_const",
            graph,
            store,
            steps,
        },
        exact: true,
    }
}

/// Layout and movement ops in one chain over a `[2, 3, 4]` slot: transpose, reshape, slice, concat and a
/// broadcast, each moving bits without changing them.
fn movement_chain() -> Row {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", f16(&[2, 3, 4]))
        .unwrap();
    let t = b.transpose(x, vec![2, 0, 1]).unwrap();
    let flat = b.reshape(t, vec![4, 6]).unwrap();
    let head = b.slice(flat, 1, 0, 3).unwrap();
    let tail = b.slice(flat, 1, 3, 6).unwrap();
    let swapped = b.concat(1, &[tail, head]).unwrap();
    let bias = b.constant("bias", f16(&[1, 6])).unwrap();
    let wide = b.broadcast(bias, vec![4, 6]).unwrap();
    let out = b.binary(BinOp::Add, swapped, wide).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[constant("bias", &[1, 6], |i| 100.0 * i as f32)]);
    let steps = (0..3)
        .map(|s| x_step("x", &[2, 3, 4], move |i| ((i * (s + 1)) % 13) as f32))
        .collect();
    Row {
        fixture: Fixture {
            name: "movement_chain",
            graph,
            store,
            steps,
        },
        exact: true,
    }
}

/// A four-dimensional add: the driver pads every argument to five dimensions, and the row layout must survive.
fn rank_four_add() -> Row {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", f16(&[2, 3, 2, 2]))
        .unwrap();
    let w = b.constant("w", f16(&[2, 3, 2, 2])).unwrap();
    let out = b.binary(BinOp::Add, x, w).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[constant("w", &[2, 3, 2, 2], |i| 10.0 * i as f32)]);
    Row {
        fixture: Fixture {
            name: "rank_four_add",
            graph,
            store,
            steps: vec![x_step("x", &[2, 3, 2, 2], |i| i as f32)],
        },
        exact: true,
    }
}

/// F32 data crossing the boundary: an F32 slot is converted to F16, added, and converted back, so the region's
/// arguments are F32 and its arithmetic F16.
fn f32_boundary() -> Row {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![2, 4]))
        .unwrap();
    let narrow = b.cast(x, DType::F16).unwrap();
    let w = b.constant("w", f16(&[2, 4])).unwrap();
    let sum = b.binary(BinOp::Add, narrow, w).unwrap();
    let out = b.cast(sum, DType::F32).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[constant("w", &[2, 4], |i| i as f32)]);
    let steps = (0..2)
        .map(|s| {
            vec![poot_eval::test_support::StepFixture {
                key: vpu_common::slot_key("x"),
                tensor: poot_tensor::HostTensor::f32(
                    vec![2, 4],
                    (0..8).map(|i| (i as f32 - 3.0) * (s + 1) as f32).collect(),
                ),
            }]
        })
        .collect();
    Row {
        fixture: Fixture {
            name: "f32_boundary",
            graph,
            store,
            steps,
        },
        exact: true,
    }
}

/// `op(x, y)` over two `[3, 5]` F16 slots whose values stay small enough that every step is exactly
/// representable (or, when `exact` is false, within FP16 tolerance). `y` is never zero, so a division is defined.
fn binary_slots(name: &'static str, op: BinOp, exact: bool) -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[3, 5])).unwrap();
    let y = b.slot_named(Slot::Activation, "y", f16(&[3, 5])).unwrap();
    let out = b.binary(op, x, y).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..2)
        .map(|s| {
            let mut step = x_step("x", &[3, 5], move |i| ((i + 3 * s) % 9) as f32 - 4.0);
            step.extend(x_step("y", &[3, 5], move |i| {
                ((i * 2 + s) % 7) as f32 * 0.5 + 0.5
            }));
            step
        })
        .collect();
    Row {
        fixture: Fixture {
            name,
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact,
    }
}

/// `op(x)` over a `[3, 5]` F16 slot of positive values, so a square root is defined and a negation has no zero
/// to disagree on: measured on Lunar Lake, the NPU's `Negative` of `+0` is `+0` where the oracle's is `-0`
/// (the sign of a zero, nothing else).
fn unary_slot(name: &'static str, op: UnOp, exact: bool) -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[3, 5])).unwrap();
    let out = b.unary(op, x).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..2)
        .map(|s| {
            x_step("x", &[3, 5], move |i| {
                ((i + 5 * s) % 12) as f32 * 0.75 + 0.25
            })
        })
        .collect();
    Row {
        fixture: Fixture {
            name,
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact,
    }
}

/// Row and column maxima of a `[4, 6]` slot, kept apart by `keepdim` and combined by a broadcasting subtract.
fn reduce_max_rows() -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[4, 6])).unwrap();
    let rows = b.reduce(RedOp::Max, x, 1, true).unwrap();
    let cols = b.reduce(RedOp::Max, x, 0, true).unwrap();
    let out = b.binary(BinOp::Sub, rows, cols).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..3)
        .map(|s| x_step("x", &[4, 6], move |i| ((i * (s + 3)) % 13) as f32 - 6.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "reduce_max_rows",
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact: true,
    }
}

/// Inline scalar literals as operands: `max(x * 0.5 + 1.5 - 2.0, 0.0)` and the indicator `ge(x, 1.0)`, added.
/// Every literal is a constant the planner defines for its consumer.
fn literal_operands() -> Row {
    let b = Builder::for_tests();
    let x = b.slot_named(Slot::Activation, "x", f16(&[3, 5])).unwrap();
    let scaled = b.binary_scalar(BinOp::Mul, x, Scalar::F32(0.5)).unwrap();
    let shifted = b
        .binary_scalar(BinOp::Add, scaled, Scalar::F32(1.5))
        .unwrap();
    let lowered = b
        .binary_scalar(BinOp::Sub, shifted, Scalar::F32(2.0))
        .unwrap();
    let relu = b
        .binary_scalar(BinOp::Max, lowered, Scalar::F32(0.0))
        .unwrap();
    let flag = b.binary_scalar(BinOp::Ge, x, Scalar::F32(1.0)).unwrap();
    let out = b.binary(BinOp::Add, relu, flag).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..2)
        .map(|s| x_step("x", &[3, 5], move |i| ((i + 4 * s) % 11) as f32 - 5.0))
        .collect();
    Row {
        fixture: Fixture {
            name: "literal_operands",
            graph,
            store: WeightStore::builder().build(),
            steps,
        },
        exact: true,
    }
}

/// An iota as a gather index: rows `0..4` of a `[6, 4]` table, the index range a constant the graph defines.
fn iota_gather() -> Row {
    let b = Builder::for_tests();
    let table = b.constant("table", f16(&[6, 4])).unwrap();
    let ids = b.iota_i32(4).unwrap();
    let out = b.gather(table, 0, ids).unwrap();
    let graph = b.finish(out).unwrap();
    Row {
        fixture: Fixture {
            name: "iota_gather",
            graph,
            store: store(&[constant("table", &[6, 4], |i| i as f32 * 0.5)]),
            steps: vec![Vec::new(), Vec::new()],
        },
        exact: true,
    }
}

/// The KV-cache slot write: a two-row update written into a `[6, 4]` cache at a literal row index.
fn update_slice_literal() -> Row {
    let b = Builder::for_tests();
    let cache = b.constant("cache", f16(&[6, 4])).unwrap();
    let row = b.slot_named(Slot::Activation, "row", f16(&[2, 4])).unwrap();
    let out = b.dynamic_update_slice(cache, row, 3, 0).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = (0..2)
        .map(|s| x_step("row", &[2, 4], move |i| (i + 1 + 10 * s) as f32))
        .collect();
    Row {
        fixture: Fixture {
            name: "update_slice_literal",
            graph,
            store: store(&[constant("cache", &[6, 4], |i| -(i as f32))]),
            steps,
        },
        exact: true,
    }
}

/// The same write at an I32 position slot that differs on every step: one recording, three positions.
fn update_slice_runtime() -> Row {
    let b = Builder::for_tests();
    let cache = b.constant("cache", f16(&[6, 4])).unwrap();
    let row = b.slot_named(Slot::Activation, "row", f16(&[2, 4])).unwrap();
    let pos = b
        .slot_named(
            Slot::Activation,
            "pos",
            TensorType::new(Vec::new(), DType::I32),
        )
        .unwrap();
    let out = b.dynamic_update_slice_dyn(cache, row, pos, 0).unwrap();
    let graph = b.finish(out).unwrap();
    let steps = [0, 2, 4]
        .into_iter()
        .map(|at: i32| {
            let mut step = x_step("row", &[2, 4], move |i| (i + 1) as f32 + 10.0 * at as f32);
            step.push(poot_eval::test_support::StepFixture {
                key: vpu_common::slot_key("pos"),
                tensor: poot_tensor::HostTensor::i32(Vec::new(), vec![at]),
            });
            step
        })
        .collect();
    Row {
        fixture: Fixture {
            name: "update_slice_runtime",
            graph,
            store: store(&[constant("cache", &[6, 4], |i| -(i as f32))]),
            steps,
        },
        exact: true,
    }
}

fn bank() -> Vec<Row> {
    vec![
        add_consts(),
        scale_shift_slot(),
        running_sum_state(),
        exp_slot(),
        reduce_sum_rows(),
        matmul_const(),
        movement_chain(),
        rank_four_add(),
        f32_boundary(),
        binary_slots("sub_slots", BinOp::Sub, true),
        binary_slots("max_slots", BinOp::Max, true),
        binary_slots("ge_slots", BinOp::Ge, true),
        binary_slots("div_slots", BinOp::Div, false),
        unary_slot("neg_slot", UnOp::Neg, true),
        unary_slot("sqrt_slot", UnOp::Sqrt, false),
        reduce_max_rows(),
        literal_operands(),
        iota_gather(),
        update_slice_literal(),
        update_slice_runtime(),
    ]
}

/// The declared fixture bank, through the ordinary route onto the NPU, every step's output (which is the
/// carried state for the running sum) against the shared CPU oracle, element by element.
#[test]
fn the_fixture_bank_matches_the_oracle_on_the_npu() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    for row in bank() {
        let fixture = &row.fixture;
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, fixture)
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
        assert_eq!(outputs.len(), fixture.steps.len());
        for (i, (got, step)) in outputs.iter().zip(&fixture.steps).enumerate() {
            let want = oracle(step);
            assert_matches(fixture.name, i, got, &want, row.exact);
        }
    }
}

/// Rows of an embedding lookup: a `[6, 4]` F16 table gathered by an I32 index slot, two steps with different
/// indices so the recording replays over indices it had not seen.
fn gather_rows() -> Row {
    let b = Builder::for_tests();
    let table = b.constant("table", f16(&[6, 4])).unwrap();
    let ids = b
        .slot_named(
            Slot::Activation,
            "ids",
            TensorType::new(vec![3], DType::I32),
        )
        .unwrap();
    let out = b.gather(table, 0, ids).unwrap();
    let graph = b.finish(out).unwrap();
    let store = store(&[constant("table", &[6, 4], |i| i as f32)]);
    let step = |ids: [i32; 3]| {
        vec![poot_eval::test_support::StepFixture {
            key: vpu_common::slot_key("ids"),
            tensor: poot_tensor::HostTensor::i32(vec![3], ids.to_vec()),
        }]
    };
    Row {
        fixture: Fixture {
            name: "gather_rows",
            graph,
            store,
            steps: vec![step([4, 0, 5]), step([1, 1, 2])],
        },
        exact: true,
    }
}

/// SC-gather: an embedding gather by I32 indices, F16 rows moved bit for bit.
///
/// Mutation (planner): lower `Gather` to `VpuOp::Reshape` in `opset_op`; the rows differ.
#[test]
fn a_gather_by_i32_indices_returns_its_rows_on_the_npu() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let row = gather_rows();
    let fixture = &row.fixture;
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, fixture)
        .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
    assert_matches(
        fixture.name,
        0,
        &outputs[0],
        &oracle(&fixture.steps[0]),
        true,
    );
}

/// The whole program is one graph: the vendor compiler runs once however many equations it holds, and each
/// step is one replay of one recording. The movement chain has eight equations; three steps are three
/// replays.
///
/// Mutation: make `poot-graph-plan`'s `package` return no region for `ArtifactGranularity::Graph`. The
/// engine finds a graph node outside any region and the load is refused ("a graph node outside any region").
#[test]
fn a_program_is_one_compiled_graph_and_one_replay_per_step() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let fixture = movement_chain().fixture;
    assert_eq!(fixture.steps.len(), 3);
    poot_executor_parity::run_outputs(&mut npu.engine, target, &fixture)
        .unwrap_or_else(|e| panic!("{e}"));
    let device = npu.engine.device();
    assert_eq!(device.compiles(), 1, "eight equations compile as one graph");
    assert_eq!(device.replays(), 3, "one replay per step");
}

/// Equations the NPU's lowering cannot take are typed refusals naming the equation and the target, before any
/// buffer is allocated: a BF16 add is not computed in FP16, an F32 add is refused naming the missing contract
/// once the target records none, `Log` has no opset op, a literal FP16 cannot hold is not saturated, a
/// conversion through BF16 changes the value.
///
/// Mutation: drop the arithmetic check in `lower_eqn` (poot-graph-plan); the BF16 add is admitted and the
/// first assertion fails.
#[test]
fn equations_the_npu_cannot_lower_are_refused_with_typed_capabilities() {
    use poot_graph_plan::Capability;
    let Some(npu) = npu() else { return };
    let target = npu.engine.device().target();
    let poot_graph_plan::Target::Npu(poot_graph_plan::NpuTarget::Vpu { target: id, caps }) = target
    else {
        panic!("the NPU device plans for a Vpu target");
    };
    let add_of = |dtype: DType| {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::new(vec![2, 3], dtype)).unwrap();
        let y = b.constant("y", TensorType::new(vec![2, 3], dtype)).unwrap();
        let out = b.binary(BinOp::Add, x, y).unwrap();
        b.finish(out).unwrap()
    };
    let refusal = refusal_of(add_of(DType::BF16), target);
    assert_eq!(
        refusal.missing,
        Capability::Arithmetic { dtype: DType::BF16 }
    );
    assert_eq!(
        refusal.target,
        poot_target::TargetId::Vpu(poot_target::VpuTargetId::Npu4)
    );
    let uncontracted = poot_graph_plan::Target::Npu(poot_graph_plan::NpuTarget::Vpu {
        target: id,
        caps: poot_target::VpuCaps {
            narrowing: None,
            ..caps
        },
    });
    assert_eq!(
        refusal_of(add_of(DType::F32), uncontracted).missing,
        Capability::NarrowingContract { dtype: DType::F32 }
    );

    let log = {
        let b = Builder::for_tests();
        let x = b.constant("x", f16(&[2, 3])).unwrap();
        let out = b.unary(UnOp::Log, x).unwrap();
        b.finish(out).unwrap()
    };
    let log_refusal = refusal_of(log, target);
    assert_eq!(log_refusal.missing, Capability::VpuOpset);

    let literal = {
        let b = Builder::for_tests();
        let x = b.constant("x", f16(&[2, 3])).unwrap();
        let out = b
            .binary_scalar(BinOp::Add, x, poot_graph_ir::Scalar::F32(1.0e10))
            .unwrap();
        b.finish(out).unwrap()
    };
    assert_eq!(
        refusal_of(literal, target).missing,
        Capability::DtypeLowering
    );

    let through_bf16 = {
        let b = Builder::for_tests();
        let x = b.constant("x", f16(&[2, 3])).unwrap();
        let out = b.cast(x, DType::BF16).unwrap();
        b.finish(out).unwrap()
    };
    assert_eq!(
        refusal_of(through_bf16, target).missing,
        Capability::DtypeLowering
    );
}

/// Unloading an executable releases every buffer it allocated: the device's live byte count returns to zero
/// for every buffer role.
///
/// Mutation: in `VpuDevice::allocate` forget the charge (`std::mem::forget(charge)`); live bytes stay
/// non-zero after unload.
#[test]
fn every_buffer_is_released_when_the_executable_is_unloaded() {
    use poot_executor::{Executor, NoSync, StepInputs};
    use std::sync::Arc;

    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    let row = scale_shift_slot();
    let fixture = &row.fixture;
    let live = |engine: &poot_executor::Engine<poot_npu::VpuDevice>| -> u64 {
        engine
            .device()
            .memory()
            .iter()
            .map(|(_, counters)| counters.live_bytes)
            .sum()
    };
    let exe = npu
        .engine
        .load_weights(
            Arc::new(fixture.store.clone()),
            poot_executor_parity::fixture_weights(fixture),
        )
        .expect("the weights load");
    let program = poot_executor_parity::staged_on(&npu.engine, fixture, target);
    let entry = npu
        .engine
        .add_entry(exe, &program)
        .expect("the entry compiles and loads");
    let mut inputs = StepInputs::new();
    for step in &fixture.steps[0] {
        inputs.push(step.key.clone(), step.tensor.shape(), step.tensor.view());
    }
    npu.engine
        .step(exe, entry, &inputs, &mut NoSync)
        .expect("the step runs");
    assert!(
        live(&npu.engine) > 0,
        "a loaded and stepped entry holds device buffers"
    );
    npu.engine.unload(exe).expect("the executable unloads");
    let memory = npu.engine.device().memory();
    assert!(!memory.is_empty(), "the device counted its allocations");
    for (role, counters) in &memory {
        assert!(counters.allocations > 0, "{role:?}");
        assert_eq!(
            counters.live_bytes, 0,
            "{role:?} holds {} bytes after unload",
            counters.live_bytes
        );
    }
}

/// Capture-and-replay: the second pass replays the recording over reset state and equals the first pass bit
/// for bit, for a stateless graph over changing inputs and for the carried running sum.
///
/// Mutation: in `VpuDevice::write_with` skip the `fill`, so a reset or an input write never reaches the
/// buffer; the second pass reads stale state and differs.
#[test]
fn replaying_a_recording_over_reset_state_is_bit_identical() {
    let Some(mut npu) = npu() else { return };
    let target = npu.engine.device().target();
    for row in [scale_shift_slot(), running_sum_state()] {
        let fixture = &row.fixture;
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        poot_executor_parity::run_capture_replay(&mut npu.engine, target, fixture, &mut oracle)
            .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// The one blob file in a cache directory.
fn only_blob(dir: &std::path::Path) -> std::path::PathBuf {
    let mut blobs: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "blob"))
        .collect();
    assert_eq!(
        blobs.len(),
        1,
        "the cache holds exactly one compiled region: {blobs:?}"
    );
    blobs.pop().unwrap()
}

/// Run the scale-and-shift fixture on a fresh engine over a device opened with `cache_dir`, comparing every
/// step with the oracle, and return `(compiles, cache_hits)`.
fn run_scale_shift_with_cache(cache_dir: &std::path::Path) -> Option<(u64, u64)> {
    let mut npu = vpu_common::npu_with(poot_npu::VpuConfig {
        cache_dir: Some(cache_dir.to_owned()),
        ..poot_npu::VpuConfig::default()
    })?;
    let target = npu.engine.device().target();
    let row = scale_shift_slot();
    let fixture = &row.fixture;
    let mut oracle = oracle_for(&fixture.graph, &fixture.store);
    let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, fixture)
        .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
    for (i, (got, step)) in outputs.iter().zip(&fixture.steps).enumerate() {
        assert_matches(fixture.name, i, got, &oracle(step), true);
    }
    let device = npu.engine.device();
    Some((device.compiles(), device.cache_hits()))
}

/// The blob cache: the first device compiles the region and leaves its blob on disk; a second device over the
/// same directory loads that blob and never runs the vendor compiler, and its output still equals the oracle.
///
/// Mutation: make `BlobCache::get` return `Ok(None)`; the second device compiles and the hit count is zero.
#[test]
fn a_second_device_loads_the_cached_blob_without_compiling() {
    let dir = vpu_common::fresh_cache_dir("hit");
    let Some(first) = run_scale_shift_with_cache(&dir) else {
        return;
    };
    assert_eq!(first, (1, 0), "an empty cache compiles once and hits never");
    let blob = only_blob(&dir);
    assert!(
        std::fs::metadata(&blob).unwrap().len() > 1000,
        "a whole blob is on disk"
    );
    let second = run_scale_shift_with_cache(&dir).expect("the device is still there");
    assert_eq!(
        second,
        (0, 1),
        "a warm cache loads the blob and never compiles"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// An entry that is not a whole blob for this exact region is a miss, never a wrong blob: a truncated file, a
/// foreign header, and an entry whose stored key is another region's are all recompiled, and the recompile
/// replaces the entry so the next device hits.
///
/// Mutation: skip the stored-key comparison in `decode`; the foreign-key entry is loaded as this region's
/// blob and the row sees a hit where a compile is owed.
#[test]
fn a_cache_entry_that_is_not_this_regions_blob_is_recompiled_and_replaced() {
    let dir = vpu_common::fresh_cache_dir("corrupt");
    if run_scale_shift_with_cache(&dir).is_none() {
        return;
    }
    let blob = only_blob(&dir);
    let valid = std::fs::read(&blob).unwrap();
    const MAGIC_LEN: usize = "poot-vpu-blob-v1\n".len();
    let key_len = u64::from_le_bytes(valid[MAGIC_LEN..MAGIC_LEN + 8].try_into().unwrap()) as usize;
    let key_at = MAGIC_LEN + 8;

    let mut foreign_key = valid.clone();
    foreign_key[key_at + key_len - 2] ^= 1; // another region's key text
    let damages: [(&str, Vec<u8>); 3] = [
        ("truncated", valid[..40].to_vec()),
        (
            "foreign header",
            [b"not a poot cache file".as_slice(), &valid[MAGIC_LEN..]].concat(),
        ),
        ("foreign key", foreign_key),
    ];
    for (what, bytes) in damages {
        std::fs::write(&blob, bytes).unwrap();
        let counts = run_scale_shift_with_cache(&dir).unwrap();
        assert_eq!(
            counts,
            (1, 0),
            "{what}: the entry is recompiled, not loaded"
        );
        assert_eq!(
            run_scale_shift_with_cache(&dir).unwrap(),
            (0, 1),
            "{what}: the recompile replaced the entry"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A blob the driver refuses to load (the entry is whole and keyed right, but its bytes are not a blob this
/// driver reads) is a miss: the region is recompiled and the bad entry replaced, with a warning, not an error.
///
/// Mutation: propagate the `load_native_graph` error in `VpuDevice::graph_of` instead of recompiling; the row
/// sees the load fail.
#[test]
fn a_cached_blob_the_driver_rejects_is_recompiled_and_replaced() {
    let dir = vpu_common::fresh_cache_dir("rejected");
    if run_scale_shift_with_cache(&dir).is_none() {
        return;
    }
    let blob = only_blob(&dir);
    let valid = std::fs::read(&blob).unwrap();
    const MAGIC_LEN: usize = "poot-vpu-blob-v1\n".len();
    let key_len = u64::from_le_bytes(valid[MAGIC_LEN..MAGIC_LEN + 8].try_into().unwrap()) as usize;
    let header = MAGIC_LEN + 8 + key_len;
    let mut garbage = valid[..header].to_vec();
    garbage.extend_from_slice(&[0xA5; 4096]);
    std::fs::write(&blob, garbage).unwrap();
    assert_eq!(run_scale_shift_with_cache(&dir).unwrap(), (1, 0));
    assert_eq!(run_scale_shift_with_cache(&dir).unwrap(), (0, 1));
    let _ = std::fs::remove_dir_all(&dir);
}

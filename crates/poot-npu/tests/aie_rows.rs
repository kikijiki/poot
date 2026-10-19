//! POOT-1174: row reductions, norms and gather on the XDNA2 NPU, each against its numerical contract.
//!
//! The AIE schedule of a row kernel is the ordinary row generator with one lane per row, and of a gather the
//! ordinary movement generator, both with their thread grid serialized into loops. Each row here drives the
//! production route: the graph is compiled for the AIE target (`poot_graph_plan::compile`, which plans every
//! equation through `poot_kernelgen::serialized_schedule`), linked by the real Peano and IRON driver, loaded
//! by the engine and run through XRT. The result is compared with the CPU oracle's under the contract the
//! planner attached to the equation (read back from the compiled program, not restated here), so a contract
//! looser than the core's behaviour is a red row. A box without the NPU or the toolchain fails these rows
//! when `POOT_REQUIRE_NPU=1`; it never passes them silently.
//!
//! ```text
//! nix develop .#npu -c cargo nextest run -p poot-npu --features xdna --test aie_rows --test-threads=1
//! ```

#![cfg(feature = "xdna")]

mod common;

use poot_eval::test_support::StepFixture;
use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::Device;
use poot_executor_parity::Fixture;
use poot_graph_ir::op::OpKind;
use poot_graph_ir::{Builder, Eqn, RedOp, Slot, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, KernelChoice, Target, compile};
use poot_kernelgen::{ErrorBound, NumericalContract};
use poot_tensor::{DType, HostTensor};

use common::{Stream, constant, npu, slot_key, store, worst_against};

const ROWS: usize = 32;
const COLS: usize = 64;

/// The contract the planner attached to the one equation of `fixture`'s graph that `is_row_op` selects.
fn planned_contract(
    fixture: &Fixture,
    target: &Target,
    is_row_op: impl Fn(&OpKind) -> bool,
) -> NumericalContract {
    let program = compile(
        &fixture.graph,
        target,
        &CompileOptions::new(CompileLimits::STANDARD),
    )
    .expect("the graph compiles for the NPU");
    let selected: Vec<&Eqn> = program
        .graph()
        .eqns()
        .iter()
        .filter(|eqn| is_row_op(&eqn.op))
        .collect();
    let [eqn] = selected.as_slice() else {
        panic!(
            "{}: expected one row equation, found {}",
            fixture.name,
            selected.len()
        );
    };
    match program.kernel_choice(eqn) {
        KernelChoice::Serialized { contract, .. } => *contract,
        other => panic!("{}: the row equation is planned as {other:?}", fixture.name),
    }
}

/// One case: a fixture, the equation that carries the contract, and for a running sum the magnitude sum
/// each result adds (`TermsUlps` is relative to it).
struct Case {
    name: &'static str,
    fixture: Fixture,
    is_row_op: fn(&OpKind) -> bool,
    terms: Vec<f32>,
}

fn is_reduce(op: &OpKind) -> bool {
    matches!(op, OpKind::Reduce { .. })
}

fn is_fused_row(op: &OpKind) -> bool {
    matches!(op, OpKind::FusedRow(_))
}

/// `values` as the dense `[ROWS, COLS]` constant `x`, reduced over the last axis with `op`; the magnitude
/// sums are what the sum's bound is relative to.
fn reduction(name: &'static str, op: RedOp, values: Vec<f32>) -> Case {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![ROWS, COLS])).unwrap();
    let out = b.reduce(op, x, 1, true).unwrap();
    let terms = values
        .chunks(COLS)
        .map(|row| row.iter().map(|v| v.abs()).sum())
        .collect();
    let data = values.clone();
    Case {
        name,
        fixture: Fixture {
            name,
            graph: b.finish(out).unwrap(),
            store: store(&[constant("x", &[ROWS, COLS], |i| data[i])]),
            steps: vec![Vec::new()],
        },
        is_row_op: is_reduce,
        terms,
    }
}

/// RMSNorm of `[ROWS, COLS]` constants, with a weight vector or without.
fn norm(name: &'static str, x: Vec<f32>, w: Option<Vec<f32>>) -> Case {
    let b = Builder::for_tests();
    let tx = b.constant("x", TensorType::f32(vec![ROWS, COLS])).unwrap();
    let mut consts = vec![constant("x", &[ROWS, COLS], |i| x[i])];
    let out = match &w {
        Some(w) => {
            let tw = b.constant("w", TensorType::f32(vec![COLS])).unwrap();
            consts.push(constant("w", &[COLS], |i| w[i]));
            poot_graph_ir::ops::rmsnorm(&b, tx, tw, 1e-5).unwrap()
        }
        None => poot_graph_ir::ops::rmsnorm_no_weight(&b, tx, 1e-5).unwrap(),
    };
    Case {
        name,
        fixture: Fixture {
            name,
            graph: b.finish(out).unwrap(),
            store: store(&consts),
            steps: vec![Vec::new()],
        },
        is_row_op: is_fused_row,
        terms: Vec::new(),
    }
}

fn cases() -> Vec<Case> {
    let mut s = Stream(0x9E37_79B9_7F4A_7C15);
    let n = ROWS * COLS;
    let signed_wide = |s: &mut Stream| -> Vec<f32> {
        s.logarithmic(n, 1e-3, 1e3)
            .into_iter()
            .map(|v| if s.next() < 0.5 { -v } else { v })
            .collect()
    };
    vec![
        reduction("row sum", RedOp::Sum, s.linear(n, -1e3, 1e3)),
        reduction("row sum (wide magnitudes)", RedOp::Sum, signed_wide(&mut s)),
        reduction("row sum (positive)", RedOp::Sum, s.linear(n, 0.0, 1.0)),
        reduction("row max", RedOp::Max, s.linear(n, -1e6, 1e6)),
        reduction("row max (wide magnitudes)", RedOp::Max, signed_wide(&mut s)),
        norm(
            "rmsnorm",
            s.linear(n, -8.0, 8.0),
            Some(s.linear(COLS, 0.5, 1.5)),
        ),
        norm(
            "rmsnorm (wide magnitudes)",
            signed_wide(&mut s),
            Some(s.linear(COLS, -2.0, 2.0)),
        ),
        norm("rmsnorm (no weight)", s.linear(n, -3.0, 3.0), None),
        norm("rmsnorm (small row values)", s.linear(n, -1e-4, 1e-4), None),
    ]
}

/// Every row reduction and norm, through the production route onto the NPU, against the CPU oracle: each
/// element within the contract the planner attached to the equation. The wide-magnitude rows reach the edge
/// of the sum's bound (partial sums that cancel) and of the norm's (squares spanning twelve decades).
///
/// Mutation: tighten the `Sum` bound in `serialized_row` to `TermsUlps(1)`, or the software square root's
/// `SQRT_REL` to `1e-9`; the row names the case and the element and fails on the measured error.
#[test]
fn every_row_op_meets_its_numerical_contract_on_the_npu() {
    let mut failures = Vec::new();
    for case in cases() {
        let Some(mut npu) = npu() else { return };
        let target = npu.engine.device().target();
        let contract = planned_contract(&case.fixture, &target, case.is_row_op);
        let mut oracle = oracle_for(&case.fixture.graph, &case.fixture.store);
        let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, &case.fixture)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let want = oracle(&case.fixture.steps[0]);
        assert_eq!(outputs[0].shape(), want.shape(), "{}: shape", case.name);
        let (got, want) = (outputs[0].to_f32().unwrap(), want.to_f32().unwrap());
        let (worst, at) = worst_against(&contract, &[], &case.terms, &got, &want);
        eprintln!(
            "{}: {} worst error/allowed {worst:.3}",
            case.name, contract.id
        );
        if worst > 1.0 {
            failures.push(format!(
                "{} ({}): element {at} of {}: the NPU computed {}, the exact result is {}, \
                 the contract allows {:?}",
                case.name,
                contract.id,
                got.len(),
                got[at],
                want[at],
                contract.bound
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The index operand of a gather: `L` positions into the `V` axis, a different set at each step so the
/// recording replays over indices it had not seen.
fn ids(step: usize, vocab: usize, len: usize) -> Vec<i32> {
    (0..len)
        .map(|i| ((i * 7 + step * 13 + 3) % vocab) as i32)
        .collect()
}

/// A gather of a `[V, D]` table by an `[L]` index slot (rows along axis 0), and of a `[D, V]` table along
/// axis 1, for `dtype` data. The table is a slot too for I32, whose constants the oracle binds as F32.
fn gather(name: &'static str, dtype: DType, axis: usize) -> (Fixture, usize) {
    const V: usize = 48;
    const D: usize = 16;
    const L: usize = 24;
    let table_shape = if axis == 0 { vec![V, D] } else { vec![D, V] };
    let b = Builder::for_tests();
    let table = match dtype {
        DType::I32 => b
            .slot_named(
                Slot::Activation,
                "table",
                TensorType::new(table_shape.clone(), DType::I32),
            )
            .unwrap(),
        _ => b
            .constant("table", TensorType::f32(table_shape.clone()))
            .unwrap(),
    };
    let index = b
        .slot_named(
            Slot::Activation,
            "ids",
            TensorType::new(vec![L], DType::I32),
        )
        .unwrap();
    let out = b.gather(table, axis, index).unwrap();
    let numel = V * D;
    let steps = (0..3)
        .map(|step| {
            let mut inputs = vec![StepFixture {
                key: slot_key("ids"),
                tensor: HostTensor::i32(vec![L], ids(step, V, L)),
            }];
            if dtype == DType::I32 {
                inputs.push(StepFixture {
                    key: slot_key("table"),
                    tensor: HostTensor::i32(
                        table_shape.clone(),
                        (0..numel as i32)
                            .map(|i| i * 31 - 7000 + step as i32)
                            .collect(),
                    ),
                });
            }
            inputs
        })
        .collect();
    let consts = if dtype == DType::I32 {
        Vec::new()
    } else {
        vec![constant("table", &table_shape, |i| {
            // Distinct bit patterns, including a negative zero and a NaN payload: a gather moves bits.
            match i % 97 {
                0 => -0.0,
                1 => f32::from_bits(0x7fc0_1234),
                _ => i as f32 * 0.37 - 400.0,
            }
        })]
    };
    (
        Fixture {
            name,
            graph: b.finish(out).unwrap(),
            store: store(&consts),
            steps,
        },
        L,
    )
}

/// A gather moves bits whatever they are: a table with a negative zero and a NaN payload comes back
/// bit for bit, F32 and I32 data, along either axis, over three steps whose indices differ.
///
/// Mutation: in `serialized_movement` bind the data as `ElementKind::I32` for an F32 gather, or lower the
/// gather's `index_numel` by one; the rows differ.
#[test]
fn a_gather_returns_its_rows_bit_for_bit_on_the_npu() {
    for (fixture, _) in [
        gather("gather rows (f32)", DType::F32, 0),
        gather("gather columns (f32)", DType::F32, 1),
        gather("gather rows (i32)", DType::I32, 0),
    ] {
        let Some(mut npu) = npu() else { return };
        let target = npu.engine.device().target();
        let contract =
            planned_contract(&fixture, &target, |op| matches!(op, OpKind::Gather { .. }));
        assert_eq!(contract.bound, ErrorBound::Exact, "{}", fixture.name);
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        let outputs = poot_executor_parity::run_outputs(&mut npu.engine, target, &fixture)
            .unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
        for (step, got) in outputs.iter().enumerate() {
            let want = oracle(&fixture.steps[step]);
            assert_eq!(
                got.shape(),
                want.shape(),
                "{} step {step}: shape",
                fixture.name
            );
            assert_eq!(
                got.dtype(),
                want.dtype(),
                "{} step {step}: dtype",
                fixture.name
            );
            let (got, want) = (got.view().bytes(), want.view().bytes());
            assert_eq!(
                got, want,
                "{} step {step}: the gathered bits differ",
                fixture.name
            );
        }
    }
}

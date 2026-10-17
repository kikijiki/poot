//! POOT-1161: pointwise F32 unary and binary ops on the XDNA2 NPU, each against its numerical contract.
//!
//! The AIE schedule is the ordinary generator's body with its thread grid serialized into loops. These rows
//! drive the production route for every op that has a contract: the graph is planned for the AIE target
//! (`poot_kernelgen::serialized_schedule`), linked by the real Peano and IRON driver, loaded by the engine and
//! run through XRT. The result is compared with the CPU oracle's exact result under the contract the planner
//! attached (`AieOp::contract`), so a contract that is looser than the core's behaviour, or an op whose
//! lowering is wrong, is a red row. A box without the NPU or the toolchain fails these rows when
//! `POOT_REQUIRE_NPU=1`; it never passes them silently.
//!
//! ```text
//! nix develop .#npu -c cargo nextest run -p poot-npu --features xdna --test aie_pointwise --test-threads=1
//! ```

#![cfg(feature = "xdna")]

mod common;

use std::fs::File;

use poot_eval::test_support::weight_map_oracle::oracle_for;
use poot_executor::{Device, Engine};
use poot_executor_parity::Fixture;
use poot_graph_ir::{BinOp, Builder, TensorType, UnOp};
use poot_kernel_ir::MathOp;
use poot_kernelgen::{AieOp, UnaryOp};
use poot_npu::{XdnaConfig, XdnaDevice};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::DType;

use common::{Stream, worst_against};

/// The tensor every case runs over: 2048 elements, laid out `[32, 64]`, so the schedule's loop is checked
/// against a shape the ordinary generator knows nothing of. Three four-byte FIFOs of it fit one tile.
const ROWS: usize = 32;
const COLS: usize = 64;
const N: usize = ROWS * COLS;

/// Hold `/tmp/poot-npu.lock` until the returned file drops: two runs never share the device.
fn lock() -> File {
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open("/tmp/poot-npu.lock")
        .expect("open the NPU lock file");
    lock.lock().expect("take the NPU lock");
    lock
}

/// A fresh engine over the NPU, or `None` (a printed SKIP) when the device or toolchain is absent and
/// `POOT_REQUIRE_NPU` is not `1`. Every case gets its own engine only to keep the cases independent: an engine
/// keys a resident constant by the store entry's identity, not its name or shape, so executables over one graph
/// and different data do not share data (`executables_over_one_graph_and_different_data_...`).
fn engine() -> Option<Engine<XdnaDevice>> {
    let get = |name: &str| std::env::var(name).ok();
    for variable in ["POOT_AIE_LLC", "POOT_AIE_IRON"] {
        if get(variable).is_none() {
            let missing = format!("${variable} is unset (run under `nix develop .#npu`)");
            let reason = DeviceBackend::Npu.fail_if_required(get, missing);
            eprintln!("SKIP: NPU toolchain unavailable: {reason}");
            return None;
        }
    }
    let device = open_or_skip(
        DeviceBackend::Npu,
        XdnaDevice::open_with(XdnaConfig::default()),
    )?;
    Some(Engine::new(device))
}

fn store(consts: &[(&str, &[f32])]) -> WeightStore {
    let mut builder = WeightStore::builder();
    for (name, values) in consts {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense =
            DenseWeight::try_new(DType::F32, vec![ROWS, COLS], StoredBytes::from(bytes)).unwrap();
        builder.insert(*name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

/// The graph op a case runs, the AIE op the planner maps it to, and its operands' data.
enum Case {
    Unary(UnOp, AieOp, Vec<f32>),
    Binary(BinOp, AieOp, Vec<f32>, Vec<f32>),
}

impl Case {
    fn unary(op: UnOp, aie: UnaryOp, x: Vec<f32>) -> Self {
        Case::Unary(op, AieOp::Unary(aie), x)
    }

    fn binary(op: BinOp, aie: poot_kernel_ir::BinOp, x: Vec<f32>, y: Vec<f32>) -> Self {
        Case::Binary(op, AieOp::Binary(aie), x, y)
    }

    fn aie(&self) -> AieOp {
        match self {
            Case::Unary(_, aie, _) | Case::Binary(_, aie, ..) => *aie,
        }
    }

    fn operands(&self) -> Vec<&[f32]> {
        match self {
            Case::Unary(_, _, x) => vec![x],
            Case::Binary(_, _, x, y) => vec![x, y],
        }
    }

    fn fixture(&self) -> Fixture {
        let b = Builder::for_tests();
        let tensor = || TensorType::f32(vec![ROWS, COLS]);
        let (out, store) = match self {
            Case::Unary(op, _, x) => {
                let t = b.constant("x", tensor()).unwrap();
                (b.unary(*op, t).unwrap(), store(&[("x", x)]))
            }
            Case::Binary(op, _, x, y) => {
                let tx = b.constant("x", tensor()).unwrap();
                let ty = b.constant("y", tensor()).unwrap();
                (b.binary(*op, tx, ty).unwrap(), store(&[("x", x), ("y", y)]))
            }
        };
        Fixture {
            name: "pointwise",
            graph: b.finish(out).unwrap(),
            store,
            steps: vec![Vec::new()],
        }
    }
}

/// The cases: every op that has a contract, over input ranges that include the edges of the contract's domain
/// (exp's clamp, log's whole normal range, the saturation of tanh and erf, cancellation in add and sub).
fn cases() -> Vec<(&'static str, Case)> {
    use poot_kernel_ir::BinOp as K;
    let mut s = Stream(0x9E37_79B9_7F4A_7C15);
    let math = UnaryOp::Math;
    vec![
        (
            "add",
            Case::binary(
                BinOp::Add,
                K::Add,
                s.linear(N, -1e6, 1e6),
                s.linear(N, -1e6, 1e6),
            ),
        ),
        (
            "add (wide magnitudes)",
            Case::binary(
                BinOp::Add,
                K::Add,
                s.logarithmic(N, 1e-20, 1e20),
                s.logarithmic(N, 1e-20, 1e20),
            ),
        ),
        (
            "add (cancellation)",
            Case::binary(
                BinOp::Add,
                K::Add,
                s.linear(N, 1.0, 2.0),
                s.linear(N, -2.0, -1.0),
            ),
        ),
        (
            "sub",
            Case::binary(
                BinOp::Sub,
                K::Sub,
                s.linear(N, -1e6, 1e6),
                s.linear(N, -1e6, 1e6),
            ),
        ),
        (
            "sub (cancellation)",
            Case::binary(
                BinOp::Sub,
                K::Sub,
                s.linear(N, 1.0, 2.0),
                s.linear(N, 1.0, 2.0),
            ),
        ),
        (
            "mul",
            Case::binary(
                BinOp::Mul,
                K::Mul,
                s.logarithmic(N, 1e-15, 1e15),
                s.logarithmic(N, 1e-15, 1e15),
            ),
        ),
        (
            "div",
            Case::binary(
                BinOp::Div,
                K::Div,
                s.logarithmic(N, 1e-15, 1e15),
                s.logarithmic(N, 1e-15, 1e15),
            ),
        ),
        (
            "max",
            Case::binary(
                BinOp::Max,
                K::Max,
                s.linear(N, -1e6, 1e6),
                s.linear(N, -1e6, 1e6),
            ),
        ),
        (
            "ge",
            Case::binary(
                BinOp::Ge,
                K::Ge,
                s.linear(N, -5.0, 5.0),
                s.linear(N, -5.0, 5.0),
            ),
        ),
        (
            "neg",
            Case::unary(
                UnOp::Neg,
                UnaryOp::Basic(poot_kernel_ir::UnOp::Neg),
                s.linear(N, -1e30, 1e30),
            ),
        ),
        (
            "recip",
            Case::unary(UnOp::Recip, UnaryOp::Recip, s.logarithmic(N, 1e-30, 1e30)),
        ),
        (
            "exp",
            Case::unary(UnOp::Exp, math(MathOp::Exp), s.linear(N, -87.0, 88.0)),
        ),
        (
            "exp (near zero)",
            Case::unary(UnOp::Exp, math(MathOp::Exp), s.linear(N, -10.0, 10.0)),
        ),
        (
            "log",
            Case::unary(
                UnOp::Log,
                math(MathOp::Log),
                s.logarithmic(N, f32::MIN_POSITIVE, 3.0e38),
            ),
        ),
        (
            "log (near one)",
            Case::unary(UnOp::Log, math(MathOp::Log), s.linear(N, 0.5, 2.0)),
        ),
        (
            "sqrt",
            Case::unary(
                UnOp::Sqrt,
                math(MathOp::Sqrt),
                s.logarithmic(N, f32::MIN_POSITIVE, 3.0e38),
            ),
        ),
        (
            "sqrt (near one)",
            Case::unary(UnOp::Sqrt, math(MathOp::Sqrt), s.linear(N, 0.5, 2.0)),
        ),
        (
            "tanh",
            Case::unary(UnOp::Tanh, UnaryOp::Tanh, s.linear(N, -60.0, 60.0)),
        ),
        (
            "tanh (near zero)",
            Case::unary(UnOp::Tanh, UnaryOp::Tanh, s.linear(N, -1e-3, 1e-3)),
        ),
        (
            "tanh (saturated)",
            Case::unary(UnOp::Tanh, UnaryOp::Tanh, s.linear(N, -1e30, 1e30)),
        ),
        (
            "erf",
            Case::unary(UnOp::Erf, UnaryOp::Erf, s.linear(N, -6.0, 6.0)),
        ),
        (
            "erf (near zero)",
            Case::unary(UnOp::Erf, UnaryOp::Erf, s.linear(N, -1e-3, 1e-3)),
        ),
        (
            "erf (saturated)",
            Case::unary(UnOp::Erf, UnaryOp::Erf, s.linear(N, -1e30, 1e30)),
        ),
    ]
}

/// Every op that has a numerical contract, through the production route onto the NPU, against the CPU
/// oracle's result: each element is within the contract the planner attached to the op, over inputs that
/// reach the edges of the contract's domain. The `[32, 64]` shape and the unary ops' flat buffers check the
/// serialized loop against a shape the generator does not know.
///
/// Mutation: tighten the `exp` contract in `AieOp::contract` to `rel: 1e-7`, or make add and sub's
/// `OperandUlps(2)` `OperandUlps(1)`; the row names the op and the element and fails on the measured error.
#[test]
fn every_pointwise_op_meets_its_numerical_contract_on_the_npu() {
    let _lock = lock();
    let mut failures = Vec::new();
    for (name, case) in cases() {
        let Some(mut engine) = engine() else { return };
        let target = engine.device().target();
        let contract = case
            .aie()
            .contract()
            .expect("the case is an op with a contract");
        let fixture = case.fixture();
        let operands = case.operands();
        let (lo, hi) = contract.domain;
        assert!(
            operands
                .iter()
                .all(|op| op.iter().all(|v| (lo..=hi).contains(v))),
            "{name}: the case's inputs leave the contract's domain {:?}",
            contract.domain
        );
        let mut oracle = oracle_for(&fixture.graph, &fixture.store);
        let outputs = poot_executor_parity::run_outputs(&mut engine, target, &fixture)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = oracle(&fixture.steps[0]);
        assert_eq!(outputs[0].shape(), [ROWS, COLS], "{name}: shape");
        let (got, want) = (outputs[0].to_f32().unwrap(), want.to_f32().unwrap());
        let (worst, at) = worst_against(&contract, &operands, &[], &got, &want);
        eprintln!("{name}: {} worst error/allowed {worst:.3}", contract.id);
        if worst > 1.0 {
            failures.push(format!(
                "{name} ({}): element {at} of {N}: the NPU computed {}, the exact result is {}, \
                 the contract allows {:?}",
                contract.id, got[at], want[at], contract.bound
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// POOT-1165: executables of one engine over the same graph and different constants each compute their own,
/// while all of them stay loaded and alternate (the plan key, kernel and artifact are shared; only data differs).
#[test]
fn executables_over_one_graph_and_different_data_each_compute_their_own_on_the_npu() {
    let _lock = lock();
    let Some(mut engine) = engine() else { return };
    let target = engine.device().target();
    let fixtures =
        poot_executor_parity::coresident::same_graph_different_data_fixtures(&[0.0, 3.0, -7.0]);
    poot_executor_parity::coresident::run_coresident(&mut engine, target, &fixtures)
        .unwrap_or_else(|e| panic!("{e}"));
}

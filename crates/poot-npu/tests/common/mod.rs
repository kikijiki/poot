//! Shared support for the XDNA2 device rows: the device lock, an engine over the NPU, fixture constants and
//! the oracle comparison. `tests/common/mod.rs` compiles once per test binary that declares `mod common;`,
//! and each binary uses a different subset, so the blanket allow below is the only way to share them.
#![allow(
    dead_code,
    reason = "compiled once per test binary, each using a different subset"
)]

use std::fs::File;

use poot_eval::test_support::StepFixture;
use poot_executor::Engine;
use poot_executor_parity::Fixture;
use poot_graph_ir::{Graph, Slot, SlotKey};
use poot_npu::{XdnaConfig, XdnaDevice};
use poot_quant::StoredBytes;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::{DType, HostTensor};

/// The held lock and an engine over the NPU. The lock is released when this drops.
pub(crate) struct Npu {
    pub(crate) engine: Engine<XdnaDevice>,
    _lock: File,
}

/// Hold `/tmp/poot-npu.lock` until the returned file drops: two runs never share the device.
pub(crate) fn lock() -> File {
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open("/tmp/poot-npu.lock")
        .expect("open the NPU lock file");
    lock.lock().expect("take the NPU lock");
    lock
}

/// The lock, and the device opened under it with `config`; `None` (a printed SKIP) when the device or the
/// Peano/IRON toolchain is absent and `POOT_REQUIRE_NPU` is not `1`. With it set, either absence fails the
/// row: the lane never passes on skips.
pub(crate) fn device(config: XdnaConfig) -> Option<(File, XdnaDevice)> {
    let lock = lock();
    let get = |name: &str| std::env::var(name).ok();
    for variable in ["POOT_AIE_LLC", "POOT_AIE_IRON"] {
        if get(variable).is_none() {
            let missing = format!("${variable} is unset (run under `nix develop .#npu`)");
            let reason = DeviceBackend::Npu.fail_if_required(get, missing);
            eprintln!("SKIP: NPU toolchain unavailable: {reason}");
            return None;
        }
    }
    let device = open_or_skip(DeviceBackend::Npu, XdnaDevice::open_with(config))?;
    Some((lock, device))
}

/// An engine over the NPU (see [`device`]).
pub(crate) fn npu() -> Option<Npu> {
    let (lock, device) = device(XdnaConfig::default())?;
    Some(Npu {
        engine: Engine::new(device),
        _lock: lock,
    })
}

/// A named dense F32 constant of the fixture's weight store.
pub(crate) struct Const {
    name: &'static str,
    shape: Vec<usize>,
    values: Vec<f32>,
}

pub(crate) fn constant(name: &'static str, shape: &[usize], f: impl Fn(usize) -> f32) -> Const {
    let numel = shape.iter().product();
    Const {
        name,
        shape: shape.to_vec(),
        values: (0..numel).map(f).collect(),
    }
}

pub(crate) fn store(consts: &[Const]) -> WeightStore {
    let mut builder = WeightStore::builder();
    for c in consts {
        let bytes: Vec<u8> = c.values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense =
            DenseWeight::try_new(DType::F32, c.shape.clone(), StoredBytes::from(bytes)).unwrap();
        builder.insert(c.name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

pub(crate) fn slot_key(name: &str) -> SlotKey {
    SlotKey::new(Slot::Activation, Some(name))
}

pub(crate) fn x_step(name: &str, shape: &[usize], f: impl Fn(usize) -> f32) -> Vec<StepFixture> {
    let numel: usize = shape.iter().product();
    vec![StepFixture {
        key: slot_key(name),
        tensor: HostTensor::f32(shape.to_vec(), (0..numel).map(f).collect()),
    }]
}

/// One fixture of the bank: whether its values are small integers, which every IEEE or AIE rounding of
/// `+ - *` and `max` reproduces exactly, so the device must equal the oracle bit for bit.
pub(crate) struct Row {
    pub(crate) fixture: Fixture,
    pub(crate) exact: bool,
}

pub(crate) fn assert_matches(
    name: &str,
    step: usize,
    got: &HostTensor,
    want: &HostTensor,
    exact: bool,
) {
    assert_eq!(got.dtype(), want.dtype(), "{name} step {step}: dtype");
    assert_eq!(got.shape(), want.shape(), "{name} step {step}: shape");
    let (got, want) = (got.to_f32().unwrap(), want.to_f32().unwrap());
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let ok = if exact {
            g.to_bits() == w.to_bits()
        } else {
            (g - w).abs() <= 1e-5 * w.abs().max(1.0)
        };
        assert!(
            ok && !g.is_nan(),
            "{name} step {step} element {i}: the NPU computed {g}, the oracle {w}"
        );
    }
}

/// Compile `graph` for `target` the way every executor entry does, returning the error the planner raised.
pub(crate) fn refusal_of(
    graph: Graph,
    target: poot_graph_plan::Target,
) -> poot_graph_plan::Refusal {
    use poot_graph_plan::multi_device::topology::DeviceId;
    use poot_graph_plan::production_plan::Submission;
    use poot_graph_plan::staged::{
        DevicePlacement, ExpertPlacement, Partition, StagedCompileError, TargetSet, compile_staged,
    };
    use poot_graph_plan::{CompileError, CompileLimits, CompileOptions, PlanError};
    let Err(error) = compile_staged(
        &graph,
        &TargetSet::single(DeviceId(0), target),
        &Partition {
            experts: ExpertPlacement::AllResident,
            devices: DevicePlacement::Single(DeviceId(0)),
        },
        &CompileOptions::new(Submission::Replay, CompileLimits::STANDARD),
    ) else {
        panic!("the equation is admitted for the NPU");
    };
    match error {
        StagedCompileError::Compile(CompileError::Plan(plan)) => match *plan {
            PlanError::Refused(refusal) => *refusal,
            other => panic!("expected a typed refusal, got {other:?}"),
        },
        other => panic!("expected a planner refusal, got {other:?}"),
    }
}

/// A deterministic stream of values in `[0, 1)`.
pub(crate) struct Stream(pub(crate) u64);

impl Stream {
    pub(crate) fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// `n` values uniform in `[lo, hi)`.
    pub(crate) fn linear(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| lo + (hi - lo) * self.next()).collect()
    }

    /// `n` values whose logarithm is uniform in `[ln lo, ln hi)`.
    pub(crate) fn logarithmic(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (lo.ln() + (hi.ln() - lo.ln()) * self.next()).exp())
            .collect()
    }
}

/// The worst `error / allowed` over a result's elements, with the element that attains it. A ratio above 1
/// is outside the contract; an `Exact` contract allows no error, so any difference is infinite. `operands`
/// are the pointwise operands an `OperandUlps` bound is relative to; `terms[i]` is the sum of the magnitudes
/// the `i`th result adds, which a `TermsUlps` bound is relative to.
pub(crate) fn worst_against(
    contract: &poot_kernelgen::NumericalContract,
    operands: &[&[f32]],
    terms: &[f32],
    got: &[f32],
    want: &[f32],
) -> (f32, usize) {
    use poot_kernelgen::ErrorBound;
    let ratio = |i: usize| -> f32 {
        let (g, w) = (got[i], want[i]);
        if !g.is_finite() {
            return f32::INFINITY;
        }
        let error = (g - w).abs();
        let allowed = match contract.bound {
            ErrorBound::Exact => {
                return if g.to_bits() == w.to_bits() {
                    0.0
                } else {
                    f32::INFINITY
                };
            }
            ErrorBound::OperandUlps(ulps) => {
                let larger = operands.iter().map(|op| op[i].abs()).fold(0.0f32, f32::max);
                ulps as f32 * larger * f32::EPSILON
            }
            ErrorBound::TermsUlps(ulps) => ulps as f32 * terms[i] * f32::EPSILON,
            ErrorBound::Result { abs, rel } => abs + rel * w.abs(),
        };
        if error == 0.0 { 0.0 } else { error / allowed }
    };
    (0..got.len())
        .map(|i| (ratio(i), i))
        .fold(
            (0.0, 0),
            |worst, next| {
                if next.0 > worst.0 { next } else { worst }
            },
        )
}

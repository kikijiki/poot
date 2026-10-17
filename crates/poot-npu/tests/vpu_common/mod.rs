//! Shared support for the Intel NPU device rows: the device lock, an engine over the NPU, F16 fixture
//! constants and the oracle comparison. Compiled once per test binary that declares `mod vpu_common;`.
#![allow(
    dead_code,
    reason = "compiled once per test binary, each using a different subset"
)]

use std::fs::File;
use std::path::PathBuf;

use poot_eval::test_support::StepFixture;
use poot_executor::Engine;
use poot_graph_ir::{Slot, SlotKey};
use poot_npu::{VpuConfig, VpuDevice};
use poot_quant::StoredBytes;
use poot_quant::scalar::f32_to_f16;
use poot_quant::weights::{DenseWeight, WeightEntry, WeightStore};
use poot_runtime_common::DeviceBackend;
use poot_runtime_common::device_skip::open_or_skip;
use poot_tensor::{DType, HostTensor};

/// The held lock and an engine over the NPU. The lock is released when this drops.
pub(crate) struct Npu {
    pub(crate) engine: Engine<VpuDevice>,
    _lock: File,
}

/// Hold `/tmp/poot-vpu.lock` until the returned file drops: two runs never share the device.
pub(crate) fn lock() -> File {
    let lock = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open("/tmp/poot-vpu.lock")
        .expect("open the NPU lock file");
    lock.lock().expect("take the NPU lock");
    lock
}

/// The lock, and the device opened under it with `config`; `None` (a printed SKIP) when the device or its
/// driver stack is absent and `POOT_REQUIRE_INTEL_NPU` is not `1`. With it set, absence fails the row: the lane
/// never passes on skips.
pub(crate) fn device(config: VpuConfig) -> Option<(File, VpuDevice)> {
    let lock = lock();
    let device = open_or_skip(DeviceBackend::IntelNpu, VpuDevice::open_with(config))?;
    Some((lock, device))
}

/// An engine over the NPU (see [`device`]) with no blob cache.
pub(crate) fn npu() -> Option<Npu> {
    npu_with(VpuConfig::default())
}

/// An engine over the NPU opened with `config`.
pub(crate) fn npu_with(config: VpuConfig) -> Option<Npu> {
    let (lock, device) = device(config)?;
    Some(Npu {
        engine: Engine::new(device),
        _lock: lock,
    })
}

/// A fresh, empty blob cache directory under the system temp dir.
pub(crate) fn fresh_cache_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("poot-vpu-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A named dense F16 constant of the fixture's weight store.
pub(crate) struct Const {
    name: &'static str,
    shape: Vec<usize>,
    words: Vec<u16>,
}

/// The F16 constant `name` of `shape` whose element `i` is `f(i)` (rounded to F16).
pub(crate) fn constant(name: &'static str, shape: &[usize], f: impl Fn(usize) -> f32) -> Const {
    let numel = shape.iter().product();
    Const {
        name,
        shape: shape.to_vec(),
        words: (0..numel).map(|i| f32_to_f16(f(i))).collect(),
    }
}

pub(crate) fn store(consts: &[Const]) -> WeightStore {
    let mut builder = WeightStore::builder();
    for c in consts {
        let bytes: Vec<u8> = c.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let dense =
            DenseWeight::try_new(DType::F16, c.shape.clone(), StoredBytes::from(bytes)).unwrap();
        builder.insert(c.name, WeightEntry::Dense(dense)).unwrap();
    }
    builder.build()
}

pub(crate) fn slot_key(name: &str) -> SlotKey {
    SlotKey::new(Slot::Activation, Some(name))
}

/// One step's F16 activation `name` of `shape`, element `i` being `f(i)` rounded to F16.
pub(crate) fn x_step(name: &str, shape: &[usize], f: impl Fn(usize) -> f32) -> Vec<StepFixture> {
    let numel: usize = shape.iter().product();
    vec![StepFixture {
        key: slot_key(name),
        tensor: HostTensor::f16(
            shape.to_vec(),
            (0..numel).map(|i| f32_to_f16(f(i))).collect(),
        ),
    }]
}

/// One fixture of the bank: whether its values are small integers, which an FP16 add, multiply or sum reproduces
/// exactly, so the device must equal the oracle bit for bit.
pub(crate) struct Row {
    pub(crate) fixture: poot_executor_parity::Fixture,
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
        // FP16 carries 11 significant bits: a tolerance of two units in the last place of the magnitude.
        let ok = if exact {
            g.to_bits() == w.to_bits()
        } else {
            (g - w).abs() <= 2.0e-3 * w.abs().max(1.0)
        };
        assert!(
            ok && !g.is_nan(),
            "{name} step {step} element {i}: the NPU computed {g}, the oracle {w}"
        );
    }
}

/// Compile `graph` for `target` the way every executor entry does, returning the refusal the planner raised.
pub(crate) fn refusal_of(
    graph: poot_graph_ir::Graph,
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
        panic!("the equation is refused for the NPU");
    };
    match error {
        StagedCompileError::Compile(CompileError::Plan(plan)) => match *plan {
            PlanError::Refused(refusal) => *refusal,
            other => panic!("expected a typed refusal, got {other:?}"),
        },
        other => panic!("expected a planner refusal, got {other:?}"),
    }
}

/// The region of `a + b` over two F16 `[1, 16]` tensors, emitted for the Intel NPU: the IR and ABI the
/// engine would hand the device. `a_name` is the first input's declared name, which a row changes to make the
/// artifact's ABI disagree with the IR.
pub(crate) fn add_artifact(a_name: &str, shape: &[usize]) -> poot_runtime_common::VpuGraphArtifact {
    use poot_codegen::{
        VpuElem, VpuInput, VpuNode, VpuOp, VpuRegion, VpuTensor, VpuValue, emit_vpu,
    };
    use poot_runtime_common::{VpuBinding, VpuGraphArtifact, VpuLayer};
    use poot_target::{BufferStorage, VpuTargetId};
    let tensor = VpuTensor {
        elem: VpuElem::F16,
        shape: shape.to_vec(),
    };
    let region = VpuRegion {
        name: "add".into(),
        inputs: ["a", "b"]
            .into_iter()
            .map(|name| VpuInput {
                name: name.into(),
                tensor: tensor.clone(),
            })
            .collect(),
        nodes: vec![VpuNode {
            origin: 2,
            op: VpuOp::Add,
            inputs: vec![VpuValue(0), VpuValue(1)],
            out: tensor,
        }],
        outputs: vec![VpuValue(2)],
    };
    let ir = emit_vpu(&region).unwrap();
    let binding = |name: &str| VpuBinding::new(name, BufferStorage::f16(), shape.to_vec());
    VpuGraphArtifact::new(
        VpuTargetId::Npu4,
        ir.xml.into_bytes(),
        ir.weights,
        vec![binding(a_name), binding("b")],
        vec![binding("v2")],
        vec![VpuLayer::new("v2", 2)],
    )
    .unwrap()
}

//! POOT-1164: the Intel NPU `Device`'s own contract, on the real device and driver: the checks only the device
//! can make before it touches the driver, the failure it owes when a replay does not finish, and the vendor
//! compiler's refusals. Rows drive `VpuDevice` directly with the artifacts the engine would hand it, so a
//! refusal is the device's own and not an earlier layer's.
//!
//! ```text
//! nix develop .#npu-intel -c cargo nextest run -p poot-npu --features vpu --test-threads=1
//! ```

#![cfg(feature = "vpu")]

mod vpu_common;

use std::time::Duration;

use poot_codegen::{
    VpuElem, VpuInput, VpuNode, VpuOp, VpuRegion, VpuTensor, VpuValue, emit_vpu, vpu_kernel_handle,
};
use poot_executor::{AieInvocation, Arg, Device, Dispatch, LaunchKind, VpuInvocation};
use poot_graph_plan::production_plan::Submission;
use poot_l0_runtime::L0Error;
use poot_npu::{VpuConfig, VpuDevice, VpuError};
use poot_quant::scalar::f32_to_f16;
use poot_runtime_common::{BufferRole, VpuBinding, VpuGraphArtifact, VpuLayer};
use poot_target::{BufferStorage, VpuTargetId};

use vpu_common::{add_artifact, device};

const N: usize = 16;

fn words(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|v| f32_to_f16(v).to_le_bytes()).collect()
}

/// The artifact `a + b` over `[1, 16]`, loaded on `device`.
fn load_add(
    device: &mut VpuDevice,
    artifact: VpuGraphArtifact,
) -> Result<poot_npu::VpuKernel, VpuError> {
    device.load_kernel("add", vpu_kernel_handle(artifact))
}

/// The compiled graph's arguments are compared with the region's ABI, argument by argument, and each way they can
/// disagree is its own refusal with its own reason: the name (this row), the argument count, the direction, the
/// extents and the precision (the rows below). The same region with an honest ABI loads.
///
/// Mutation: drop the name comparison in `check_abi`; the graph's `a` is accepted as `zz` and the load succeeds.
#[test]
fn a_compiled_graph_that_disagrees_with_the_regions_abi_is_refused_before_it_runs() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    // The IR names its first input `a`; the region's ABI claims `zz`. The compiled graph reports `a`.
    let Err(error) = load_add(&mut device, add_artifact("zz", &[1, N])) else {
        panic!("a graph whose argument names differ from the region's ABI must be refused");
    };
    let VpuError::Abi { key, reason } = error else {
        panic!("expected an ABI refusal, got {error:?}");
    };
    assert_eq!(key, "add");
    assert!(
        reason.contains("argument 0 is `a` in the graph, `zz` in the region"),
        "{reason}"
    );
    // The same region with an honest ABI loads: the refusal is the disagreement, not the region.
    load_add(&mut device, add_artifact("a", &[1, N])).expect("the honest ABI loads");
}

/// A region of more than five dimensions is refused at load, naming the argument: the driver reports every
/// argument in five, so a sixth cannot be checked against the ABI.
///
/// Mutation: drop the rank check in `VpuDevice::load_kernel`; the region reaches the compiler instead.
#[test]
fn a_region_past_five_dimensions_is_refused_at_load() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let Err(error) = load_add(&mut device, add_artifact("a", &[1, 1, 1, 1, 1, N])) else {
        panic!("a rank-six region must be refused");
    };
    let VpuError::Abi { reason, .. } = error else {
        panic!("expected an ABI refusal, got {error:?}");
    };
    assert!(reason.contains("more than 5 dimensions"), "{reason}");
}

/// The vendor compiler's rejection is a typed error carrying the compiler's own log, not a panic or a bare
/// status: here a layer type the compiler's opset does not contain.
#[test]
fn a_vendor_compile_failure_is_a_typed_error_with_the_compilers_log() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let good = add_artifact("a", &[1, N]);
    let xml = String::from_utf8(good.xml().to_vec())
        .unwrap()
        .replace("type=\"Add\"", "type=\"Bucketize\"");
    let bad = VpuGraphArtifact::new(
        good.target(),
        xml.into_bytes(),
        good.weights().to_vec(),
        good.inputs().to_vec(),
        good.outputs().to_vec(),
        good.layers().to_vec(),
    )
    .unwrap();
    let Err(error) = load_add(&mut device, bad) else {
        panic!("a layer the compiler does not know must be refused");
    };
    let VpuError::L0 { source, .. } = &error else {
        panic!("expected a Level Zero error carrying the compiler's log, got {error:?}");
    };
    let L0Error::GraphBuild { log, .. } = source else {
        panic!("expected a graph build failure, got {source:?}");
    };
    assert!(log.contains("Bucketize"), "the log names the layer: {log}");
}

/// A dispatch that is not the loaded graph's ABI is refused by the device before anything is recorded: a launch
/// of another domain, an invocation of another artifact, a wrong argument count, a buffer of another storage, a
/// wrong element count, a buffer too short. The honest dispatch then still runs and returns the sum, so the
/// refusals changed nothing.
///
/// Mutation: drop `check_invocation` in `VpuDevice::dispatch`; the invocation row accepts a foreign digest.
#[test]
fn a_dispatch_that_is_not_the_graphs_abi_is_refused_before_it_runs() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let artifact = add_artifact("a", &[1, N]);
    let digest = artifact.digest();
    let kernel = load_add(&mut device, artifact).expect("the add loads");
    let f16 = BufferStorage::f16();
    let alloc = |device: &mut VpuDevice, storage: BufferStorage, elems: usize| {
        device
            .allocate(BufferRole::Staging, storage, elems)
            .unwrap()
    };
    let a = alloc(&mut device, f16, N);
    let b = alloc(&mut device, f16, N);
    let out = alloc(&mut device, f16, N);
    let short = alloc(&mut device, f16, N - 1);
    let wide = alloc(&mut device, BufferStorage::f32(), N);
    let honest = LaunchKind::VpuGraph(VpuInvocation::new(VpuTargetId::Npu4, digest));
    let arg = |buffer, elems| Arg { buffer, elems };

    device.begin(Submission::Replay).unwrap();
    let mut dispatch = |launch: LaunchKind, inputs: &[Arg<'_, VpuDevice>], output| {
        device.dispatch(Dispatch {
            kernel: &kernel,
            inputs,
            output,
            launch,
            work: 1,
        })
    };
    let two = [arg(&a, N as u32), arg(&b, N as u32)];

    assert!(matches!(
        dispatch(
            LaunchKind::Grid {
                threads: [1, 1, 1],
                workgroup: [1, 1, 1]
            },
            &two,
            arg(&out, N as u32)
        ),
        Err(VpuError::GridLaunch { .. })
    ));
    assert!(matches!(
        dispatch(
            LaunchKind::Aie(AieInvocation::new(
                poot_target::AieTargetId::Xdna2,
                std::num::NonZeroU32::new(8).unwrap()
            )),
            &two,
            arg(&out, N as u32)
        ),
        Err(VpuError::AieLaunch)
    ));
    assert!(matches!(
        dispatch(
            LaunchKind::VpuGraph(VpuInvocation::new(VpuTargetId::Npu4, digest ^ 1)),
            &two,
            arg(&out, N as u32)
        ),
        Err(VpuError::InvocationMismatch { .. })
    ));
    assert!(matches!(
        dispatch(honest, &two[..1], arg(&out, N as u32)),
        Err(VpuError::Bindings { reason, .. }) if reason.contains("graph has 3 arguments, the dispatch binds 2")
    ));
    assert!(matches!(
        dispatch(honest, &[arg(&wide, N as u32), arg(&b, N as u32)], arg(&out, N as u32)),
        Err(VpuError::Bindings { reason, .. }) if reason.contains("argument 0")
    ));
    assert!(matches!(
        dispatch(honest, &two, arg(&out, N as u32 - 1)),
        Err(VpuError::Bindings { reason, .. }) if reason.contains("argument 2 is 16 elements, the dispatch binds 15")
    ));
    assert!(matches!(
        dispatch(honest, &two, arg(&short, N as u32)),
        Err(VpuError::Bindings { reason, .. }) if reason.contains("needs 32 bytes, the bound buffer holds 30")
    ));

    // The honest dispatch runs after all of those.
    dispatch(honest, &two, arg(&out, N as u32)).expect("the honest dispatch records");
    device
        .write_with(&a, 0, 32, &mut |_, bytes| {
            bytes.copy_from_slice(&words((0..N).map(|i| i as f32)))
        })
        .unwrap();
    device
        .write_with(&b, 0, 32, &mut |_, bytes| {
            bytes.copy_from_slice(&words((0..N).map(|i| 100.0 + i as f32)))
        })
        .unwrap();
    let recording = device.finish().unwrap().expect("a recording");
    device.replay(&recording).unwrap();
    let mut got = [0u8; 32];
    device.read(&out, &mut got).unwrap();
    assert_eq!(got.to_vec(), words((0..N).map(|i| 100.0 + 2.0 * i as f32)));
}

/// A replay that does not finish in time is a typed failure, and it poisons the device: the graph may still be
/// running, so every later call is refused and the buffers stay allocated. A `[2048, 2048]` F16 matmul takes far
/// longer than the microsecond the deadline allows.
///
/// Mutation: drop the `self.poison(..)` call in `VpuDevice::replay`; the later calls succeed.
#[test]
fn a_replay_that_misses_its_deadline_poisons_the_device() {
    let config = VpuConfig {
        run_timeout: Duration::from_micros(1),
        ..VpuConfig::default()
    };
    let Some((_lock, mut device)) = device(config) else {
        return;
    };
    let dim = 2048;
    let tensor = VpuTensor {
        elem: VpuElem::F16,
        shape: vec![dim, dim],
    };
    let region = VpuRegion {
        name: "heavy".into(),
        inputs: ["a", "b"]
            .into_iter()
            .map(|name| VpuInput {
                name: name.into(),
                tensor: tensor.clone(),
            })
            .collect(),
        nodes: vec![VpuNode {
            origin: 2,
            op: VpuOp::MatMul,
            inputs: vec![VpuValue(0), VpuValue(1)],
            out: tensor,
        }],
        outputs: vec![VpuValue(2)],
    };
    let ir = emit_vpu(&region).unwrap();
    let binding = |name: &str| VpuBinding::new(name, BufferStorage::f16(), vec![dim, dim]);
    let artifact = VpuGraphArtifact::new(
        VpuTargetId::Npu4,
        ir.xml.into_bytes(),
        ir.weights,
        vec![binding("a"), binding("b")],
        vec![binding("v2")],
        vec![VpuLayer::new("v2", 2)],
    )
    .unwrap();
    let digest = artifact.digest();
    let kernel = load_add(&mut device, artifact).expect("the matmul compiles");
    let elems = dim * dim;
    let buffers: Vec<_> = (0..3)
        .map(|_| {
            device
                .allocate(BufferRole::Staging, BufferStorage::f16(), elems)
                .unwrap()
        })
        .collect();
    device.begin(Submission::Replay).unwrap();
    let n = elems as u32;
    device
        .dispatch(Dispatch {
            kernel: &kernel,
            inputs: &[
                Arg {
                    buffer: &buffers[0],
                    elems: n,
                },
                Arg {
                    buffer: &buffers[1],
                    elems: n,
                },
            ],
            output: Arg {
                buffer: &buffers[2],
                elems: n,
            },
            launch: LaunchKind::VpuGraph(VpuInvocation::new(VpuTargetId::Npu4, digest)),
            work: 1,
        })
        .unwrap();
    let recording = device.finish().unwrap().expect("a recording");

    let Err(VpuError::L0 { source, .. }) = device.replay(&recording) else {
        panic!("a 1 microsecond deadline cannot hold a 2048-cube matmul");
    };
    assert!(matches!(source, L0Error::Timeout { .. }), "{source:?}");
    assert!(matches!(device.replay(&recording), Err(VpuError::Poisoned)));
    assert!(matches!(
        device.allocate(BufferRole::Staging, BufferStorage::f16(), 1),
        Err(VpuError::Poisoned)
    ));
    let mut out = [0u8; 2];
    assert!(matches!(
        device.read(&buffers[2], &mut out),
        Err(VpuError::Poisoned)
    ));
    let live: u64 = device
        .memory()
        .iter()
        .map(|(_, counters)| counters.live_bytes)
        .sum();
    assert!(live > 0, "the failed replay's buffers stay allocated");
}

/// An artifact with `good`'s IR and the given declared ABI.
fn rebuilt(
    good: &VpuGraphArtifact,
    inputs: Vec<VpuBinding>,
    outputs: Vec<VpuBinding>,
) -> VpuGraphArtifact {
    VpuGraphArtifact::new(
        good.target(),
        good.xml().to_vec(),
        good.weights().to_vec(),
        inputs,
        outputs,
        good.layers().to_vec(),
    )
    .unwrap()
}

/// Load `artifact` and return the reason of the ABI refusal it must get.
fn abi_reason(device: &mut VpuDevice, artifact: VpuGraphArtifact) -> String {
    match load_add(device, artifact) {
        Err(VpuError::Abi { key, reason }) => {
            assert_eq!(key, "add");
            reason
        }
        Err(other) => panic!("expected an ABI refusal, got {other:?}"),
        Ok(_) => panic!("a graph that disagrees with the region's ABI must be refused"),
    }
}

/// A region that declares fewer arguments than the compiled graph has is refused for its count.
///
/// Mutation: drop the argument-count comparison in `check_abi`; the zip stops at the shorter list and the load
/// succeeds.
#[test]
fn a_graph_with_more_arguments_than_the_region_declares_is_refused_for_its_count() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let good = add_artifact("a", &[1, N]);
    let missing_input = rebuilt(
        &good,
        vec![good.inputs()[0].clone()],
        good.outputs().to_vec(),
    );
    let reason = abi_reason(&mut device, missing_input);
    assert!(
        reason.contains("the region has 2 arguments, the graph 3"),
        "{reason}"
    );
}

/// An argument the region calls an output but the graph reads is refused for its direction. The names and the
/// count agree (`a`, `b`, `v2`), so only the direction differs.
///
/// Mutation: drop the direction comparison in `check_abi`; the load succeeds.
#[test]
fn an_argument_read_by_the_graph_but_declared_an_output_is_refused_for_its_direction() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let good = add_artifact("a", &[1, N]);
    let swapped = rebuilt(
        &good,
        vec![good.inputs()[0].clone()],
        vec![good.inputs()[1].clone(), good.outputs()[0].clone()],
    );
    let reason = abi_reason(&mut device, swapped);
    assert!(
        reason.contains("argument 1 (`b`) is Input in the graph, Output in the region"),
        "{reason}"
    );
}

/// A region that declares other extents than the IR compiled with is refused for its dimensions: the compiled
/// graph reports `[1, 16]` padded to five, the region says `[1, 8]`.
///
/// Mutation: drop the extents comparison in `check_abi` (`if false && argument.dims != padded`); the load
/// succeeds.
#[test]
fn a_region_whose_extents_differ_from_the_compiled_graphs_is_refused_for_its_dimensions() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let good = add_artifact("a", &[1, N]);
    let narrow = VpuBinding::new("a", BufferStorage::f16(), vec![1, N / 2]);
    let wrong = rebuilt(
        &good,
        vec![narrow, good.inputs()[1].clone()],
        good.outputs().to_vec(),
    );
    let reason = abi_reason(&mut device, wrong);
    assert!(
        reason.contains(
            "argument 0 (`a`) has dimensions [1, 16, 1, 1, 1] in the graph, [1, 8] in the region"
        ),
        "{reason}"
    );
}

/// A region that declares another precision than the IR's is refused for it, for an input and for an output: the
/// vendor compiler keeps the IR's FP16 at the boundary whatever the option string asks, and the device does not
/// accept a conversion it did not plan.
///
/// Mutation: drop the precision comparison in `check_abi`; both loads succeed.
#[test]
fn a_region_whose_precision_differs_from_the_compiled_graphs_is_refused_for_its_precision() {
    let Some((_lock, mut device)) = device(VpuConfig::default()) else {
        return;
    };
    let good = add_artifact("a", &[1, N]);
    let wide_input = rebuilt(
        &good,
        vec![
            VpuBinding::new("a", BufferStorage::f32(), vec![1, N]),
            good.inputs()[1].clone(),
        ],
        good.outputs().to_vec(),
    );
    let reason = abi_reason(&mut device, wide_input);
    assert!(
        reason.contains("argument 0 (`a`) is Fp16 in the compiled graph, Fp32 in the region"),
        "{reason}"
    );
    let int_output = rebuilt(
        &good,
        good.inputs().to_vec(),
        vec![VpuBinding::new("v2", BufferStorage::i32(), vec![1, N])],
    );
    let reason = abi_reason(&mut device, int_output);
    assert!(
        reason.contains("argument 2 (`v2`) is Fp16 in the compiled graph, Int32 in the region"),
        "{reason}"
    );
}

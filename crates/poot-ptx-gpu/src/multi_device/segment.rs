//! ADR-0099 items 1 and 2: compute segments, the capture-scoped buffer view, and the
//! compute-segment capture entry. `executor.rs` invokes these inside a node's capturable
//! phase; nothing here opens a device on its own.

use std::collections::HashMap;

use cudarc::driver::sys;
use poot_graph_plan::multi_device::replay::{ReplayContract, StaticBuffer, StaticBufferId};
#[cfg(test)]
use poot_graph_plan::multi_device::{ByteCategories, OwnerId, PlacementRow, Topology};
use poot_graph_plan::multi_device::{DeviceId, PlanNodeId};

use super::error::CapturedMultiDeviceError;
#[cfg(test)]
use super::executor::{
    CaptureInputs, PtxCommunicationBufferBinding, PtxDeviceCapture, PtxMultiDeviceReplayCounters,
};
use super::executor::{PtxCapturedMultiDeviceExecutor, ResidentBuffer};
use crate::p2p::PtxP2PTransport;

/// One PTX program a compute segment may record on a rank's capture stream (ADR-0099 item 1).
///
/// Kernel parameters follow poot's shared dispatch ABI: every buffer operand is packed as the
/// interleaved pair `(device address, element count)`, exactly as `PtxContext::dispatch_dev` packs
/// `ins` then `out`, so a `poot-codegen` program compiled for a chain of buffer operands launches
/// unchanged. The count comes from `buffer.bytes / elem_bytes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentProgram {
    /// PTX text. Loaded into the segment device's context before the first capturable phase.
    pub ptx: String,
    /// Kernel entry point inside `ptx`.
    pub entry: String,
    /// Threads per block; the grid is `ceil(threads / block)` per axis.
    pub block: [u32; 3],
    /// Element size of every buffer operand, in bytes. A buffer whose byte count is not a whole
    /// number of elements is rejected before the launch.
    pub elem_bytes: u32,
}

/// Capture-scoped, typed access to a capture's resident buffers, keyed by [`StaticBufferId`]
/// (ADR-0099 item 2).
///
/// The view hands out only Card 360's own [`StaticBuffer`] metadata plus the resident [`DeviceId`],
/// never a device address: the one raw value it can resolve stays private to this module and leaves
/// it only inside a recorded kernel launch. The executor builds it per segment invocation, so its
/// lifetime is that invocation's and it cannot be stored past the phase.
pub struct CaptureBufferView<'capture> {
    contract: &'capture ReplayContract,
    resident: &'capture HashMap<StaticBufferId, ResidentBuffer>,
}

impl<'capture> CaptureBufferView<'capture> {
    /// The contract entry for `id` and the device it is resident on, or `None` when this capture
    /// never uploaded `id`.
    pub fn get(&self, id: &StaticBufferId) -> Option<(&'capture StaticBuffer, DeviceId)> {
        let buffer = self
            .contract
            .buffers
            .iter()
            .find(|buffer| &buffer.id == id)?;
        let resident = self.resident.get(id)?;
        Some((buffer, resident.device))
    }

    /// [`Self::get`] as a typed [`CapturedMultiDeviceError`].
    pub fn require(
        &self,
        id: &StaticBufferId,
    ) -> Result<(&'capture StaticBuffer, DeviceId), CapturedMultiDeviceError> {
        self.get(id)
            .ok_or_else(|| CapturedMultiDeviceError::UnknownBufferUpload { id: id.clone() })
    }

    /// Device address of the resident buffer behind `id`. Private: this is the only raw value the
    /// view carries.
    fn device_ptr(&self, id: &StaticBufferId) -> Option<sys::CUdeviceptr> {
        self.resident.get(id).map(|resident| resident.buffer.ptr())
    }
}

/// Model-free admission for one segment launch's program index: the launch must name a program the
/// segment declared at capture time, and that program must carry a usable element size. The
/// returned reference is the only program the rest of the launch can bind, so nothing downstream can
/// see a program this function rejected.
pub(crate) fn admit_segment_program(
    node: PlanNodeId,
    programs: &[SegmentProgram],
    index: usize,
) -> Result<&SegmentProgram, CapturedMultiDeviceError> {
    let program = programs
        .get(index)
        .ok_or(CapturedMultiDeviceError::SegmentProgramIndex {
            node,
            index,
            programs: programs.len(),
        })?;
    if program.elem_bytes == 0 {
        return Err(CapturedMultiDeviceError::SegmentProgramElementBytes {
            node,
            program: index,
            elem_bytes: program.elem_bytes,
        });
    }
    Ok(program)
}

/// Model-free admission for one segment launch's operands: every buffer must be resident on this
/// segment's device and hold a whole number of the program's elements. Returns each operand's
/// element count in order - the length half of every buffer parameter in poot's shared dispatch ABI.
pub(crate) fn admit_segment_operands(
    node: PlanNodeId,
    program: &SegmentProgram,
    expected_device: DeviceId,
    operands: &[(&StaticBuffer, DeviceId)],
) -> Result<Vec<i64>, CapturedMultiDeviceError> {
    let elem_bytes = u64::from(program.elem_bytes);
    let mut lengths = Vec::with_capacity(operands.len());
    for (buffer, device) in operands {
        if *device != expected_device {
            return Err(CapturedMultiDeviceError::SegmentBufferDevice {
                node,
                id: buffer.id.clone(),
                expected: expected_device,
                actual: *device,
            });
        }
        if buffer.bytes % elem_bytes != 0 {
            return Err(CapturedMultiDeviceError::SegmentBufferElementSize {
                node,
                id: buffer.id.clone(),
                bytes: buffer.bytes,
                elem_bytes: program.elem_bytes,
            });
        }
        lengths.push((buffer.bytes / elem_bytes) as i64);
    }
    Ok(lengths)
}

/// One launch's `(grid, block)`, in the tuple shape `cuLaunchKernel` takes.
pub(crate) type SegmentLaunchGeometry = ((u32, u32, u32), (u32, u32, u32));

/// Model-free admission for one segment launch's dispatch shape: rounding `threads` up to the
/// program's block shape must leave a non-empty grid on every axis. Returns the `(grid, block)` the
/// launch uses.
pub(crate) fn admit_segment_extent(
    node: PlanNodeId,
    program: &SegmentProgram,
    threads: [u32; 3],
) -> Result<SegmentLaunchGeometry, CapturedMultiDeviceError> {
    let block = (
        program.block[0].max(1),
        program.block[1].max(1),
        program.block[2].max(1),
    );
    let grid = (
        threads[0].div_ceil(block.0),
        threads[1].div_ceil(block.1),
        threads[2].div_ceil(block.2),
    );
    if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
        return Err(CapturedMultiDeviceError::SegmentLaunchExtent { node, threads });
    }
    Ok((grid, block))
}

/// The context the executor hands one compute segment inside its node's capturable phase
/// (ADR-0099 item 1): after every `wait_for_node` for that node, before the node's recorded event,
/// on the segment device's capture stream.
///
/// Private fields make the executor the only constructor, and `'capture` ties it to the phase, so a
/// segment cannot keep it.
pub struct SegmentCapture<'capture> {
    transport: &'capture PtxP2PTransport,
    rank: usize,
    device: DeviceId,
    node: PlanNodeId,
    programs: &'capture [SegmentProgram],
    functions: &'capture [sys::CUfunction],
    buffers: CaptureBufferView<'capture>,
}

impl<'capture> SegmentCapture<'capture> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        transport: &'capture PtxP2PTransport,
        contract: &'capture ReplayContract,
        rank: usize,
        device: DeviceId,
        node: PlanNodeId,
        programs: &'capture [SegmentProgram],
        functions: &'capture [sys::CUfunction],
        resident: &'capture HashMap<StaticBufferId, ResidentBuffer>,
    ) -> Self {
        debug_assert_eq!(
            programs.len(),
            functions.len(),
            "a segment's programs and loaded entry points are index-aligned"
        );
        Self {
            transport,
            rank,
            device,
            node,
            programs,
            functions,
            buffers: CaptureBufferView { contract, resident },
        }
    }

    /// The device whose rank records this segment.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The opaque plan node whose phase is being captured.
    pub fn node(&self) -> PlanNodeId {
        self.node
    }

    /// The capture-scoped buffer view keyed by [`StaticBufferId`].
    pub fn buffers(&self) -> &CaptureBufferView<'capture> {
        &self.buffers
    }

    /// Record one declared program on this segment's capture stream, binding `operands` in order.
    ///
    /// Every operand must be resident on this segment's device and a whole number of the program's
    /// elements; each contributes `(device address, element count)` to the kernel's parameters.
    /// `threads` is the dispatch shape, rounded up to the program's block shape.
    pub fn launch(
        &mut self,
        program: usize,
        operands: &[&StaticBufferId],
        threads: [u32; 3],
    ) -> Result<(), CapturedMultiDeviceError> {
        let meta = admit_segment_program(self.node, self.programs, program)?;
        // `new` binds `functions` index-aligned with `programs`, so the admitted index is safe.
        debug_assert_eq!(self.functions.len(), self.programs.len());
        let function = self.functions[program];

        let mut resolved = Vec::with_capacity(operands.len());
        for id in operands {
            resolved.push(self.buffers.require(id)?);
        }
        let lengths = admit_segment_operands(self.node, meta, self.device, &resolved)?;

        let mut addresses = Vec::with_capacity(operands.len());
        for id in operands {
            addresses.push(
                self.buffers
                    .device_ptr(id)
                    .ok_or(CapturedMultiDeviceError::UnknownBufferUpload { id: (*id).clone() })?,
            );
        }

        let (grid, block) = admit_segment_extent(self.node, meta, threads)?;

        let mut params: Vec<*mut std::ffi::c_void> = Vec::with_capacity(addresses.len() * 2);
        for index in 0..addresses.len() {
            params.push(&addresses[index] as *const _ as *mut std::ffi::c_void);
            params.push(&lengths[index] as *const _ as *mut std::ffi::c_void);
        }
        self.transport
            .launch_rank_program(self.rank, function, grid, block, &mut params)?;
        Ok(())
    }
}

/// ADR-0099 item 1: caller-supplied compute the executor records inside one opaque plan node's
/// capturable phase on one rank's capture stream.
///
/// A trait object rather than a bare closure type, and deliberately so: the public capture entry
/// has to name the type it stores and document against it, a real model stage is a struct that owns
/// its compiled programs and its bindings (a closure would have to capture all of that to say the
/// same thing), and the trait is the one place the ordering contract can be stated. Boxing is
/// identical either way, so choosing closures would buy ergonomics only.
pub trait PtxComputeSegment {
    /// Record this segment's work. Called inside the node's capturable phase, after the waits for
    /// its producers and before the node's recorded event, on the capture stream of
    /// [`SegmentCapture::device`]'s rank.
    fn record(&mut self, capture: &mut SegmentCapture<'_>) -> Result<(), CapturedMultiDeviceError>;
}

/// One compute-segment capture entry: a segment bound to one opaque plan node on one device.
pub struct PtxDeviceSegment {
    /// Topology device whose rank records this segment.
    pub device: DeviceId,
    /// Opaque plan node whose phase records it. A communication-row id is rejected.
    pub node: PlanNodeId,
    /// Programs the segment may launch, in [`SegmentCapture::launch`] index order. Loaded into
    /// `device`'s context before the first capturable phase.
    pub programs: Vec<SegmentProgram>,
    /// The segment itself.
    pub segment: Box<dyn PtxComputeSegment>,
}

impl PtxCapturedMultiDeviceExecutor {
    /// The ADR-0099 item 1 capture entry: [`Self::capture_with_counters`] plus the compute segments
    /// that record model work at opaque plan nodes. An empty `segments` list is exactly
    /// [`Self::capture_with_counters`]'s behavior, down to the replay counters.
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn capture_with_segments(
        topology: Topology,
        placements: Vec<PlacementRow>,
        expected_owners: Vec<OwnerId>,
        device_usage: HashMap<DeviceId, ByteCategories>,
        replay: ReplayContract,
        device_captures: Vec<PtxDeviceCapture>,
        communication_buffers: Vec<PtxCommunicationBufferBinding>,
        segments: Vec<PtxDeviceSegment>,
        counters: &mut PtxMultiDeviceReplayCounters,
    ) -> Result<Self, CapturedMultiDeviceError> {
        Self::capture_inputs_with_counters(
            CaptureInputs {
                topology,
                placements,
                expected_owners,
                device_usage,
                replay,
                device_captures,
                communication_buffers,
            },
            segments,
            counters,
        )
    }
}

#[cfg(test)]
mod tests {

    use poot_graph_plan::CollectiveKind;
    use poot_graph_plan::multi_device::replay::{ReplayContract, StaticBufferId, StaticBufferKind};
    use poot_graph_plan::multi_device::{ByteCategories, CommunicationPlan};
    use poot_kernel_ir::BinOp;

    use super::super::executor::tests::{
        communication_row, p2p_inputs, replay_input_buffer, static_buffer, topology,
    };
    use super::super::executor::{
        CaptureInputs, DeviceCommunicationBuffers, PtxCommunicationBufferBinding, PtxDeviceCapture,
        PtxMultiDeviceReplayCounters, StaticBufferUpload, admit_captured_multi_device_plan,
    };
    use super::*;

    fn capture_segments(
        inputs: CaptureInputs,
        segments: Vec<PtxDeviceSegment>,
        counters: &mut PtxMultiDeviceReplayCounters,
    ) -> Result<PtxCapturedMultiDeviceExecutor, CapturedMultiDeviceError> {
        let CaptureInputs {
            topology,
            placements,
            expected_owners,
            device_usage,
            replay,
            device_captures,
            communication_buffers,
        } = inputs;
        PtxCapturedMultiDeviceExecutor::capture_with_segments(
            topology,
            placements,
            expected_owners,
            device_usage,
            replay,
            device_captures,
            communication_buffers,
            segments,
            counters,
        )
    }

    /// A segment that records nothing, so admission tests can bind one without a device.
    struct SilentSegment;

    impl PtxComputeSegment for SilentSegment {
        fn record(
            &mut self,
            _capture: &mut SegmentCapture<'_>,
        ) -> Result<(), CapturedMultiDeviceError> {
            Ok(())
        }
    }

    fn segment(device: DeviceId, node: PlanNodeId) -> PtxDeviceSegment {
        PtxDeviceSegment {
            device,
            node,
            programs: vec![SegmentProgram {
                ptx: String::new(),
                entry: "unused".into(),
                block: [64, 1, 1],
                elem_bytes: 4,
            }],
            segment: Box::new(SilentSegment),
        }
    }

    /// `p2p_inputs` plus one opaque producer and one opaque consumer, so the schedule's execution
    /// order contains nodes a segment may legally bind.
    fn opaque_node_inputs() -> CaptureInputs {
        let mut inputs = p2p_inputs();
        inputs.replay.communication.rows[0].producers = vec![PlanNodeId(1001)];
        inputs.replay.communication.rows[0].consumers = vec![PlanNodeId(1002)];
        inputs
    }

    // -- ADR-0099 item 1: model-free segment admission -------------------------------

    #[test]
    fn captured_executor_segment_admission_accepts_an_opaque_plan_node() {
        let admitted = admit_captured_multi_device_plan(
            opaque_node_inputs(),
            &[segment(DeviceId(41), PlanNodeId(1001))],
        )
        .expect("an opaque producer of an admitted row must take a segment");
        assert_eq!(admitted.rank_by_device[&DeviceId(41)], 0);
        assert!(
            admitted.execution_order.contains(&PlanNodeId(1001)),
            "the bound node must be part of the schedule's execution order"
        );
    }

    /// Red under: the admission loop's `SegmentOnCommunicationRow` branch disabled. Observed
    /// 2026-09-26: the typed error is never produced, so control falls through to
    /// `PtxP2PTransport::new` and the test panics in cudarc's driver load ("Unable to dynamically
    /// load the \"cuda\" shared library") instead of matching
    /// `SegmentOnCommunicationRow { node: PlanNodeId(5) }`. Green without the mutation.
    #[test]
    fn captured_executor_segment_on_a_communication_row_rejects_with_zero_device_counters() {
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_segments(
            p2p_inputs(),
            vec![segment(DeviceId(41), PlanNodeId(5))],
            &mut counters,
        )
        .err()
        .expect("a segment bound to communication row 5 must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentOnCommunicationRow {
                    node: PlanNodeId(5)
                }
            ),
            "expected SegmentOnCommunicationRow, got {error:?}"
        );
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());
    }

    /// Red under: the admission loop's `SegmentUnknownNode` branch disabled. Observed 2026-09-26:
    /// the typed error is never produced, so control falls through to `PtxP2PTransport::new` and
    /// the test panics in cudarc's driver load ("Unable to dynamically load the \"cuda\" shared
    /// library") instead of matching `SegmentUnknownNode { node: PlanNodeId(4242) }`. Green
    /// without the mutation.
    #[test]
    fn captured_executor_segment_on_an_unknown_node_rejects_with_zero_device_counters() {
        let mut counters = PtxMultiDeviceReplayCounters::default();
        let error = capture_segments(
            p2p_inputs(),
            vec![segment(DeviceId(41), PlanNodeId(4242))],
            &mut counters,
        )
        .err()
        .expect("a segment bound to a node no row or edge declares must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentUnknownNode {
                    node: PlanNodeId(4242)
                }
            ),
            "expected SegmentUnknownNode, got {error:?}"
        );
        assert_eq!(counters, PtxMultiDeviceReplayCounters::default());
    }

    /// Red under: the admission loop's `elem_bytes == 0` branch disabled. Observed 2026-09-26:
    /// admission returns `Ok`, so `.err().expect` panics "a program with no element size must
    /// reject". Green without the mutation.
    #[test]
    fn captured_executor_segment_program_admission_rejects_a_zero_element_size() {
        let mut broken = segment(DeviceId(41), PlanNodeId(1001));
        broken.programs[0].elem_bytes = 0;
        let error = admit_captured_multi_device_plan(opaque_node_inputs(), &[broken])
            .err()
            .expect("a program with no element size must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentProgramElementBytes {
                    node: PlanNodeId(1001),
                    program: 0,
                    elem_bytes: 0,
                }
            ),
            "expected SegmentProgramElementBytes, got {error:?}"
        );
    }

    /// Red under: the admission loop's `block.contains(&0)` branch disabled. Observed 2026-09-26:
    /// admission returns `Ok`, so `.err().expect` panics "a program with a zero workgroup extent
    /// must reject". Green without the mutation.
    #[test]
    fn captured_executor_segment_program_admission_rejects_a_zero_block_extent() {
        let mut flat_block = segment(DeviceId(41), PlanNodeId(1001));
        flat_block.programs[0].block = [64, 0, 1];
        let error = admit_captured_multi_device_plan(opaque_node_inputs(), &[flat_block])
            .err()
            .expect("a program with a zero workgroup extent must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentProgramBlockShape { program: 0, .. }
            ),
            "expected SegmentProgramBlockShape, got {error:?}"
        );
    }

    // -- ADR-0099 S1 device row ------------------------------------------------------

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn read_f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    /// Device 0's segment: `carrier:a = x + x`, then `carrier:a += step`, so `a` holds
    /// `2*x + step` when the captured transfer picks it up.
    struct PipelineStageZero;

    impl PtxComputeSegment for PipelineStageZero {
        fn record(
            &mut self,
            capture: &mut SegmentCapture<'_>,
        ) -> Result<(), CapturedMultiDeviceError> {
            let x = StaticBufferId("input:x".into());
            let step = StaticBufferId("fixed:step".into());
            let a = StaticBufferId("carrier:a".into());
            let threads = [1000, 1, 1];
            capture.launch(0, &[&x, &x, &a], threads)?;
            capture.launch(0, &[&a, &step, &a], threads)?;
            Ok(())
        }
    }

    /// Device 1's segment: `comm:z = z + 1`, so `z` holds `y + 1` for the received `y`.
    struct PipelineStageOne;

    impl PtxComputeSegment for PipelineStageOne {
        fn record(
            &mut self,
            capture: &mut SegmentCapture<'_>,
        ) -> Result<(), CapturedMultiDeviceError> {
            let z = StaticBufferId("comm:z".into());
            let ones = StaticBufferId("fixed:ones".into());
            capture.launch(0, &[&z, &ones, &z], [1000, 1, 1])?;
            Ok(())
        }
    }

    /// ADR-0099 S1 evidence: a synthetic two-device pipeline. Device 0's segment computes
    /// `y = 2*x + step` into `carrier:a` (`x` is a per-replay input), a captured point-to-point
    /// transfer moves `a` to device 1, and device 1's segment computes `z = y + 1`. Three replays
    /// use three different `x` values and every downloaded `z` must equal the host reference
    /// exactly.
    #[test]
    #[ignore = "requires two peer-capable NVIDIA GPUs"]
    fn probe_adr0099_segment_pipeline_matches_host_reference_over_three_replays() {
        const ELEMENTS: usize = 1000;
        let element_bytes = (ELEMENTS * 4) as u64;
        let device_count = crate::p2p::device_count().expect("ADR-0099 needs the NVIDIA driver");
        assert!(
            device_count >= 2,
            "ADR-0099 needs two NVIDIA GPUs, found {device_count}"
        );

        let x_id = StaticBufferId("input:x".into());
        let step_id = StaticBufferId("fixed:step".into());
        let a_id = StaticBufferId("carrier:a".into());
        let z_id = StaticBufferId("comm:z".into());
        let ones_id = StaticBufferId("fixed:ones".into());

        let add_body = poot_kernelgen::binary("card451_segment_add", BinOp::Add);
        let add_program = SegmentProgram {
            ptx: crate::p2p::compile_add_ptx(&add_body).expect("compile the segment add program"),
            entry: "card451_segment_add".into(),
            block: add_body.workgroup_size,
            elem_bytes: 4,
        };

        let step: Vec<f32> = (0..ELEMENTS).map(|index| (100 + index) as f32).collect();
        let ones: Vec<f32> = vec![1.0; ELEMENTS];
        let zeros = vec![0u8; ELEMENTS * 4];

        let mut transfer_row = communication_row(30, CollectiveKind::PointToPoint, ELEMENTS * 4);
        transfer_row.producers = vec![PlanNodeId(1000)];
        transfer_row.consumers = vec![PlanNodeId(1001)];
        let contract = ReplayContract {
            buffers: vec![
                replay_input_buffer(&x_id.0, ELEMENTS * 4),
                static_buffer(&step_id.0, StaticBufferKind::DenseConstant, ELEMENTS * 4),
                static_buffer(&a_id.0, StaticBufferKind::Carrier, ELEMENTS * 4),
                static_buffer(&z_id.0, StaticBufferKind::Communication, ELEMENTS * 4),
                static_buffer(&ones_id.0, StaticBufferKind::DenseConstant, ELEMENTS * 4),
            ],
            dispatch_count: 3,
            communication: CommunicationPlan {
                rows: vec![transfer_row],
            },
        };

        let mut device_usage = HashMap::new();
        device_usage.insert(
            DeviceId(41),
            ByteCategories {
                source_or_carrier: 3 * element_bytes,
                ..ByteCategories::default()
            },
        );
        device_usage.insert(
            DeviceId(7),
            ByteCategories {
                source_or_carrier: 2 * element_bytes,
                ..ByteCategories::default()
            },
        );

        let inputs = CaptureInputs {
            topology: topology(&[CollectiveKind::PointToPoint]),
            placements: Vec::new(),
            expected_owners: Vec::new(),
            device_usage,
            replay: contract.clone(),
            device_captures: vec![
                PtxDeviceCapture {
                    device: DeviceId(41),
                    graph_identity: "segment-stage-0".into(),
                    static_uploads: vec![
                        StaticBufferUpload {
                            id: x_id.clone(),
                            bytes: zeros.clone(),
                        },
                        StaticBufferUpload {
                            id: step_id.clone(),
                            bytes: f32_bytes(&step),
                        },
                        StaticBufferUpload {
                            id: a_id.clone(),
                            bytes: zeros.clone(),
                        },
                    ],
                },
                PtxDeviceCapture {
                    device: DeviceId(7),
                    graph_identity: "segment-stage-1".into(),
                    static_uploads: vec![
                        StaticBufferUpload {
                            id: ones_id.clone(),
                            bytes: f32_bytes(&ones),
                        },
                        StaticBufferUpload {
                            id: z_id.clone(),
                            bytes: zeros.clone(),
                        },
                    ],
                },
            ],
            communication_buffers: vec![PtxCommunicationBufferBinding {
                row: PlanNodeId(30),
                devices: vec![
                    DeviceCommunicationBuffers {
                        device: DeviceId(41),
                        source: Some(a_id.clone()),
                        destination: None,
                    },
                    DeviceCommunicationBuffers {
                        device: DeviceId(7),
                        source: None,
                        destination: Some(z_id.clone()),
                    },
                ],
            }],
        };
        let segments = vec![
            PtxDeviceSegment {
                device: DeviceId(41),
                node: PlanNodeId(1000),
                programs: vec![add_program.clone()],
                segment: Box::new(PipelineStageZero),
            },
            PtxDeviceSegment {
                device: DeviceId(7),
                node: PlanNodeId(1001),
                programs: vec![add_program],
                segment: Box::new(PipelineStageOne),
            },
        ];

        let mut counters = PtxMultiDeviceReplayCounters::default();
        let mut executor = capture_segments(inputs, segments, &mut counters)
            .expect("the segment pipeline captures on two GPUs");

        for replay in 0..3usize {
            let x: Vec<f32> = (0..ELEMENTS)
                .map(|index| ((index + replay * 7) % 13) as f32)
                .collect();
            let x_bytes = f32_bytes(&x);
            executor
                .write_replay_inputs(&[(x_id.clone(), x_bytes.as_slice())])
                .expect("the per-replay input writes");
            executor.replay(&contract).expect("replay");

            let z = read_f32s(
                &executor
                    .download(&z_id)
                    .expect("download the pipeline output"),
            );
            assert_eq!(z.len(), ELEMENTS, "replay {replay} must fill every element");
            for (index, (&got, &input)) in z.iter().zip(x.iter()).enumerate() {
                let expected = ((input + input) + step[index]) + 1.0;
                assert_eq!(got, expected, "replay {replay} element {index}");
            }
        }

        // Checked last so a dropped transfer reddens on the values above first.
        assert_eq!(counters.captured_transfer_nodes, 1);
    }

    // -- ADR-0099 repair 2: model-free launch admission --------------------------------

    fn sample_program() -> SegmentProgram {
        SegmentProgram {
            ptx: String::new(),
            entry: "unused".into(),
            block: [64, 1, 1],
            elem_bytes: 4,
        }
    }

    fn sample_buffer(id: &str, bytes: u64) -> StaticBuffer {
        StaticBuffer {
            id: StaticBufferId(id.into()),
            kind: StaticBufferKind::DenseConstant,
            bytes,
            fingerprint: 0,
            per_replay_input: false,
        }
    }

    /// Red under: the `programs.get(index)` lookup falling back to the first declared program.
    /// Observed 2026-09-26: the bad index admits, so `.expect_err` panics "an index past the
    /// declared programs must reject". Green without the mutation.
    #[test]
    fn segment_launch_admission_rejects_an_undeclared_program_index() {
        let programs = [sample_program()];
        admit_segment_program(PlanNodeId(1001), &programs, 0)
            .expect("a declared program index admits");

        let error = admit_segment_program(PlanNodeId(1001), &programs, 3)
            .expect_err("an index past the declared programs must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentProgramIndex {
                    node: PlanNodeId(1001),
                    index: 3,
                    programs: 1,
                }
            ),
            "expected SegmentProgramIndex, got {error:?}"
        );
    }

    /// Red under: the `*device != expected_device` branch of `admit_segment_operands` disabled.
    /// Observed 2026-09-26: the foreign operand admits, so `.expect_err` panics "an operand on
    /// another device must reject". Green without the mutation.
    #[test]
    fn segment_launch_admission_rejects_an_operand_on_another_device() {
        let program = sample_program();
        let buffer = sample_buffer("resident:a", 64);
        let lengths = admit_segment_operands(
            PlanNodeId(1001),
            &program,
            DeviceId(41),
            &[(&buffer, DeviceId(41))],
        )
        .expect("an operand on the segment's own device admits");
        assert_eq!(lengths, vec![16]);

        let error = admit_segment_operands(
            PlanNodeId(1001),
            &program,
            DeviceId(41),
            &[(&buffer, DeviceId(7))],
        )
        .expect_err("an operand on another device must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentBufferDevice {
                    node: PlanNodeId(1001),
                    ref id,
                    expected: DeviceId(41),
                    actual: DeviceId(7),
                } if *id == StaticBufferId("resident:a".into())
            ),
            "expected SegmentBufferDevice, got {error:?}"
        );
    }

    /// Red under: the `buffer.bytes % elem_bytes != 0` branch of `admit_segment_operands`
    /// disabled. Observed 2026-09-26: the partial buffer admits, so `.expect_err` panics "a
    /// buffer that is not a whole number of elements must reject". Green without the mutation.
    #[test]
    fn segment_launch_admission_rejects_a_partial_element_buffer() {
        let program = sample_program();
        let whole = sample_buffer("resident:b", 64);
        let lengths = admit_segment_operands(
            PlanNodeId(1001),
            &program,
            DeviceId(41),
            &[(&whole, DeviceId(41))],
        )
        .expect("a whole number of elements admits");
        assert_eq!(lengths, vec![16]);

        let partial = sample_buffer("resident:odd", 6);
        let error = admit_segment_operands(
            PlanNodeId(1001),
            &program,
            DeviceId(41),
            &[(&partial, DeviceId(41))],
        )
        .expect_err("a buffer that is not a whole number of elements must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentBufferElementSize {
                    node: PlanNodeId(1001),
                    ref id,
                    bytes: 6,
                    elem_bytes: 4,
                } if *id == StaticBufferId("resident:odd".into())
            ),
            "expected SegmentBufferElementSize, got {error:?}"
        );
    }

    /// Red under: the zero-grid branch of `admit_segment_extent` disabled. Observed 2026-09-26:
    /// the empty shape admits, so `.expect_err` panics "an empty dispatch extent must reject".
    /// Green without the mutation.
    #[test]
    fn segment_launch_admission_rejects_an_empty_dispatch_extent() {
        let program = sample_program();
        let (grid, block) = admit_segment_extent(PlanNodeId(1001), &program, [1000, 1, 1])
            .expect("a non-empty dispatch shape admits");
        assert_eq!(block, (64, 1, 1));
        assert_eq!(grid, (16, 1, 1));

        let error = admit_segment_extent(PlanNodeId(1001), &program, [0, 1, 1])
            .expect_err("an empty dispatch extent must reject");
        assert!(
            matches!(
                error,
                CapturedMultiDeviceError::SegmentLaunchExtent {
                    node: PlanNodeId(1001),
                    threads: [0, 1, 1],
                }
            ),
            "expected SegmentLaunchExtent, got {error:?}"
        );
    }
}

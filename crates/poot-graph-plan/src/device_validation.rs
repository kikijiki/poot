//! Device witness admission for validation-bearing graphs (Card 375b).
//!
//! A device executor computes validation witnesses with its own kernels. A packet judges the values that
//! backend computed and would publish (Spec 375, self-consistent gating). This module admits only witness
//! heads whose computation is exact on any IEEE-conforming device: after an ordered `Ge` observation, every
//! lane stays a small integer, so reduction order, parallel sums, and FMA contraction cannot change the
//! packet bits.
//!
//! The admission is a structural classifier. `witness_op` maps every `OpKind`, `BinOp`, `RedOp`, and
//! `DType` with an exhaustive `match` and no wildcard arm, so a new operation does not compile until it is
//! classified.

use poot_graph_ir::op::{BinOp, OpKind, RedOp};
use poot_graph_ir::{
    Eqn, Graph, Operand, Scalar, ValidationChannel, ValidationPacketLayout, ValueId,
};
use poot_kernel_ir::{Body, WmmaShape};
use poot_kernelgen::{BodyUse, FragmentUse, KernelRequirements, UnmeasurableLds};
use poot_target::{DeviceCaps, Queried, SubgroupSupport, TensorCoreSupport};
use poot_tensor::DType;

use crate::{PlanError, ValidationPacketPlan, ValidationPacketSource};

/// The largest lane magnitude a witness head may reach. Every integer up to 2^24 is exact in f32.
pub const DEVICE_WITNESS_MAGNITUDE_BOUND: u64 = 1 << 24;

/// Why a validation value is not a canonical device witness.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum DeviceWitnessRejection {
    #[error("v{value} is a graph input, not a device computation")]
    GraphInput { value: ValueId },
    #[error("v{value} has dtype {dtype}, which cannot hold an exact count")]
    Dtype { value: ValueId, dtype: DType },
    #[error("v{value} is produced by {op}, which has no exact witness-head role")]
    UnsupportedOp { value: ValueId, op: String },
    #[error("v{value} casts {from} to {to}; only F32 and I32 counts are exact")]
    CastDtype {
        value: ValueId,
        from: DType,
        to: DType,
    },
    #[error("v{value} uses literal {literal:?}, which is not an integer of magnitude at most 2^24")]
    Literal { value: ValueId, literal: Scalar },
    #[error("v{value} can reach magnitude {bound}, above the exact bound 2^24")]
    Bound { value: ValueId, bound: u64 },
    #[error("the magnitude bound of v{value} overflowed u64")]
    BoundOverflow { value: ValueId },
    #[error("v{value} reduces axis {axis}, which its operand does not have")]
    ReduceAxis { value: ValueId, axis: usize },
}

/// The role one operation plays in a witness head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WitnessOp {
    /// An ordered comparison: 0 or 1 for any operands, including NaN.
    Observation,
    Add,
    Sub,
    Mul,
    Max,
    /// `f + c * (t - f)` over I32 counts.
    Select,
    ReduceSum {
        axis: usize,
    },
    ReduceMax,
    /// F32 <-> I32 of a count.
    Cast {
        to: DType,
    },
    /// Rearranges lanes without changing any value.
    Movement,
    Unsupported,
}

fn witness_bin_op(op: BinOp) -> WitnessOp {
    match op {
        BinOp::Ge | BinOp::GeU => WitnessOp::Observation,
        BinOp::Add => WitnessOp::Add,
        BinOp::Sub => WitnessOp::Sub,
        BinOp::Mul => WitnessOp::Mul,
        BinOp::Max => WitnessOp::Max,
        // Division/remainder and bitwise operations do not preserve the count domain.
        BinOp::Div
        | BinOp::RemU
        | BinOp::And
        | BinOp::Or
        | BinOp::Xor
        | BinOp::Shl
        | BinOp::Shr => WitnessOp::Unsupported,
    }
}

fn witness_red_op(op: RedOp, axis: usize) -> WitnessOp {
    match op {
        RedOp::Sum => WitnessOp::ReduceSum { axis },
        RedOp::Max => WitnessOp::ReduceMax,
    }
}

fn count_dtype(dtype: DType) -> bool {
    match dtype {
        DType::F32 | DType::I32 => true,
        DType::BF16 | DType::F16 | DType::I8 | DType::E4M3FN => false,
        DType::Bool
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F64
        | DType::E8M0 => unreachable!(
            "{dtype:?} is a checkpoint-storage-only dtype; no traced graph value declares it"
        ),
    }
}

fn witness_op(op: &OpKind) -> WitnessOp {
    match op {
        OpKind::Binary(op) => witness_bin_op(*op),
        OpKind::Select => WitnessOp::Select,
        OpKind::Reduce { op, axis, .. } => witness_red_op(*op, *axis),
        OpKind::Cast { to } => WitnessOp::Cast { to: *to },
        OpKind::Reshape { .. }
        | OpKind::Transpose { .. }
        | OpKind::Slice { .. }
        | OpKind::Broadcast { .. }
        | OpKind::Concat { .. } => WitnessOp::Movement,
        // Unary operations are transcendental, rounding, or bitwise; gather and scatter read a data-dependent
        // lane; contractions, attention, quantization, and fused regions may reassociate or contract.
        OpKind::Unary(_)
        | OpKind::Gather { .. }
        | OpKind::Scatter { .. }
        | OpKind::ScatterUpdate
        | OpKind::MatMul
        | OpKind::PackedDequant { .. }
        | OpKind::PackedContraction { .. }
        | OpKind::PackedRowGather { .. }
        | OpKind::DenseContraction { .. }
        | OpKind::DenseRowGather { .. }
        | OpKind::MatMulBias
        | OpKind::ArgTopK { .. }
        | OpKind::IndexedMatMul
        | OpKind::DynamicUpdateSlice { .. }
        | OpKind::PackI8
        | OpKind::UnpackI8 { .. }
        | OpKind::Fused(_)
        | OpKind::FusedRow(_)
        | OpKind::FlashAttentionDecode { .. }
        | OpKind::FlashAttentionPrefill { .. }
        | OpKind::Rope { .. }
        | OpKind::AllReduce { .. }
        | OpKind::AllGather { .. }
        | OpKind::Iota { .. }
        | OpKind::RandomUniform { .. }
        | OpKind::SampleToken { .. } => WitnessOp::Unsupported,
    }
}

struct WitnessWalk<'g, V: ValidationChannel> {
    graph: &'g Graph<V>,
    producers: Vec<Option<&'g Eqn>>,
    bounds: Vec<Option<u64>>,
}

impl<'g, V: ValidationChannel> WitnessWalk<'g, V> {
    fn new(graph: &'g Graph<V>) -> Self {
        let mut producers = vec![None; graph.values.len()];
        for eqn in &graph.eqns {
            producers[eqn.out] = Some(eqn);
        }
        Self {
            graph,
            producers,
            bounds: vec![None; graph.values.len()],
        }
    }

    fn checked(value: ValueId, bound: Option<u64>) -> Result<u64, DeviceWitnessRejection> {
        match bound {
            Some(bound) if bound <= DEVICE_WITNESS_MAGNITUDE_BOUND => Ok(bound),
            Some(bound) => Err(DeviceWitnessRejection::Bound { value, bound }),
            None => Err(DeviceWitnessRejection::BoundOverflow { value }),
        }
    }

    fn operand(
        &mut self,
        value: ValueId,
        operand: &Operand,
    ) -> Result<u64, DeviceWitnessRejection> {
        match *operand {
            Operand::Value(input) => self.bound(input),
            Operand::Lit(literal) => {
                let magnitude = match literal {
                    Scalar::I32(v) => Some(u64::from(v.unsigned_abs())),
                    Scalar::F32(v) if v.is_finite() && v.fract() == 0.0 => Some(v.abs() as u64),
                    Scalar::F32(_) => None,
                };
                magnitude
                    .filter(|&magnitude| magnitude <= DEVICE_WITNESS_MAGNITUDE_BOUND)
                    .ok_or(DeviceWitnessRejection::Literal { value, literal })
            }
        }
    }

    fn operands(&mut self, eqn: &Eqn) -> Result<Vec<u64>, DeviceWitnessRejection> {
        eqn.inputs
            .iter()
            .map(|operand| self.operand(eqn.out, operand))
            .collect()
    }

    /// The static magnitude bound of a count-domain value, or why it is not one.
    fn bound(&mut self, value: ValueId) -> Result<u64, DeviceWitnessRejection> {
        if let Some(bound) = self.bounds[value] {
            return Ok(bound);
        }
        // Every value in a head holds an exact count, so its own dtype must be able to represent one.
        // Checked per value, not only where a `Cast` names a dtype, so the invariant is local.
        let dtype = self.graph.aval(value).dtype;
        if !count_dtype(dtype) {
            return Err(DeviceWitnessRejection::Dtype { value, dtype });
        }
        let eqn = self.producers[value].ok_or(DeviceWitnessRejection::GraphInput { value })?;
        let bound = match witness_op(&eqn.op) {
            WitnessOp::Observation => 1,
            WitnessOp::Add | WitnessOp::Sub => {
                let bounds = self.operands(eqn)?;
                Self::checked(
                    value,
                    bounds.iter().try_fold(0u64, |sum, &b| sum.checked_add(b)),
                )?
            }
            WitnessOp::Mul => {
                let bounds = self.operands(eqn)?;
                Self::checked(
                    value,
                    bounds
                        .iter()
                        .try_fold(1u64, |product, &b| product.checked_mul(b)),
                )?
            }
            WitnessOp::Max | WitnessOp::ReduceMax | WitnessOp::Movement => {
                self.operands(eqn)?.into_iter().max().unwrap_or(0)
            }
            WitnessOp::Select => {
                let bounds = self.operands(eqn)?;
                let &[condition, if_true, if_false] = bounds.as_slice() else {
                    return Err(DeviceWitnessRejection::UnsupportedOp {
                        value,
                        op: eqn.op.name(),
                    });
                };
                Self::checked(
                    value,
                    if_true
                        .checked_add(if_false)
                        .and_then(|sum| condition.checked_mul(sum))
                        .and_then(|scaled| scaled.checked_add(if_false)),
                )?
            }
            WitnessOp::ReduceSum { axis } => {
                let bounds = self.operands(eqn)?;
                let extent = match eqn.inputs.first() {
                    Some(Operand::Value(input)) => self
                        .graph
                        .aval(*input)
                        .shape
                        .get(axis)
                        .copied()
                        .ok_or(DeviceWitnessRejection::ReduceAxis { value, axis })?,
                    _ => return Err(DeviceWitnessRejection::ReduceAxis { value, axis }),
                };
                Self::checked(
                    value,
                    bounds
                        .first()
                        .copied()
                        .and_then(|bound| bound.checked_mul(extent as u64)),
                )?
            }
            WitnessOp::Cast { to } => {
                let from = match eqn.inputs.first() {
                    Some(Operand::Value(input)) => self.graph.aval(*input).dtype,
                    Some(Operand::Lit(literal)) => literal.dtype(),
                    None => to,
                };
                if !count_dtype(from) || !count_dtype(to) {
                    return Err(DeviceWitnessRejection::CastDtype { value, from, to });
                }
                self.operands(eqn)?.into_iter().max().unwrap_or(0)
            }
            WitnessOp::Unsupported => {
                return Err(DeviceWitnessRejection::UnsupportedOp {
                    value,
                    op: eqn.op.name(),
                });
            }
        };
        self.bounds[value] = Some(bound);
        Ok(bound)
    }
}

/// Admit every validation declaration in `graph` as a canonical device witness, on the unfused
/// graph. [`compile`](crate::compile) calls this immediately before `fuse`: fusion folds a witness
/// producer chain into one `OpKind::Fused` region, which this structural classifier does not see
/// through, so the walk must run on the graph the classifier is defined over. A graph with no
/// validation declarations returns immediately, with no walk.
pub(crate) fn admit_device_witnesses<V: ValidationChannel>(
    graph: &Graph<V>,
) -> Result<(), PlanError> {
    if graph.validation_outputs().is_empty() {
        return Ok(());
    }
    let mut walk = WitnessWalk::new(graph);
    for validation in graph.validation_outputs() {
        walk.bound(validation.value).map_err(|reason| {
            PlanError::ValidationWitnessNotCanonical {
                id: validation.id,
                value: validation.value,
                reason,
            }
        })?;
    }
    Ok(())
}

/// The declaration-ordered packet plan for any validation channel.
pub(crate) fn validation_packet_plan<V: ValidationChannel>(
    graph: &Graph<V>,
) -> Result<ValidationPacketPlan, PlanError> {
    graph.validate()?;
    let layout = ValidationPacketLayout::for_graph(graph)?;
    let sources = graph
        .validation_outputs()
        .iter()
        .zip(&layout.entries)
        .map(|(validation, entry)| ValidationPacketSource {
            value: validation.value,
            first_lane: entry.first_lane,
            lane_count: entry.lane_count,
        })
        .collect();
    Ok(ValidationPacketPlan { layout, sources })
}

/// Why a planned kernel cannot run on a device: its body, launch or declared requirements need more
/// than the device's [`DeviceCaps`] state, or need something the caps cannot confirm.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ResourceRefusal {
    #[error(transparent)]
    UnmeasurableLds(#[from] UnmeasurableLds),
    #[error("the body uses the {fragment:?} matrix fragment but declares no subgroup requirement")]
    FragmentWithoutSubgroup { fragment: FragmentUse },
    #[error("the device's {tensor_core:?} matrix hardware does not run the {fragment:?} fragment")]
    FragmentUnsupported {
        fragment: FragmentUse,
        tensor_core: TensorCoreSupport,
    },
    #[error(
        "the device's matrix hardware is not exposed by its API, so the {fragment:?} fragment is unproven"
    )]
    FragmentCapabilityUnknown { fragment: FragmentUse },
    #[error("the kernel needs {lanes}-lane subgroups and the device has no subgroup support")]
    SubgroupAbsent { lanes: u32 },
    #[error(
        "the kernel needs {lanes}-lane subgroups and the device's subgroup support is not exposed by its API"
    )]
    SubgroupUnknown { lanes: u32 },
    #[error("the kernel needs {lanes}-lane subgroups; the device runs {min_size} to {max_size}")]
    SubgroupSize {
        lanes: u32,
        min_size: u32,
        max_size: u32,
    },
    #[error(
        "the {invocations}-invocation workgroup is not a whole number of {lanes}-lane subgroups"
    )]
    WorkgroupNotSubgroupMultiple { invocations: u64, lanes: u32 },
    #[error("workgroup axis {axis} has extent zero")]
    WorkgroupExtentZero { axis: usize },
    #[error("workgroup axis {axis} extent {size} exceeds the device limit {limit}")]
    WorkgroupDimension { axis: usize, size: u32, limit: u32 },
    #[error("the {invocations}-invocation workgroup exceeds the device limit {limit}")]
    WorkgroupInvocations { invocations: u64, limit: u32 },
    #[error(
        "the body declares {needed} bytes of static workgroup memory; the device offers {available}"
    )]
    StaticLds { needed: u64, available: u32 },
    #[error("one dispatch carries {work} serial steps; the device bounds a dispatch at {budget}")]
    DispatchWork { work: u64, budget: u64 },
}

/// Check one kernel against `caps` before any loader sees it. `body` states the resources the kernel
/// uses; `declared` states what the body cannot (see [`KernelRequirements`]). The
/// resources the body states are measured from it, so a declaration that understates one is refused, not
/// believed. An `Unknown` subgroup or matrix-hardware capability cannot prove a requirement and refuses; an
/// `Unknown` `max_dispatch_work` means no calibrated bound exists, and the dispatch is admitted. Every
/// production `DeviceCaps` default keeps `max_dispatch_work` `Unknown` until POOT-634 calibrates it, so only
/// fixture caps exercise the work refusal today. The launch's
/// workgroup count is not checked here: a host folds an oversized 1-D launch onto Y (`fold_grid`), so
/// whether a count fits is that fold's rule, and the host refuses with its own typed `GridCap`.
pub(crate) fn validate_kernel_resources(
    body: &Body,
    declared: KernelRequirements,
    caps: &DeviceCaps,
) -> Result<(), ResourceRefusal> {
    let used = BodyUse::measure(body)?;
    let invocations: u64 = used
        .workgroup
        .iter()
        .map(|&extent| u64::from(extent))
        .product();

    for &fragment in &used.fragments {
        if declared.subgroup_lanes.is_none() {
            return Err(ResourceRefusal::FragmentWithoutSubgroup { fragment });
        }
        check_fragment(fragment, caps.tensor_core)?;
    }
    if let Some(lanes) = declared.subgroup_lanes {
        match caps.subgroup {
            Queried::Unknown => return Err(ResourceRefusal::SubgroupUnknown { lanes }),
            Queried::Known(SubgroupSupport::Absent) => {
                return Err(ResourceRefusal::SubgroupAbsent { lanes });
            }
            Queried::Known(SubgroupSupport::Present { min_size, max_size }) => {
                if !(min_size..=max_size).contains(&lanes) {
                    return Err(ResourceRefusal::SubgroupSize {
                        lanes,
                        min_size,
                        max_size,
                    });
                }
            }
        }
        if !invocations.is_multiple_of(u64::from(lanes)) {
            return Err(ResourceRefusal::WorkgroupNotSubgroupMultiple { invocations, lanes });
        }
    }

    for (axis, &size) in used.workgroup.iter().enumerate() {
        if size == 0 {
            return Err(ResourceRefusal::WorkgroupExtentZero { axis });
        }
        let limit = caps.max_workgroup_size[axis];
        if size > limit {
            return Err(ResourceRefusal::WorkgroupDimension { axis, size, limit });
        }
    }
    if invocations > u64::from(caps.max_workgroup_invocations) {
        return Err(ResourceRefusal::WorkgroupInvocations {
            invocations,
            limit: caps.max_workgroup_invocations,
        });
    }
    if used.lds_bytes > u64::from(caps.lds_bytes) {
        return Err(ResourceRefusal::StaticLds {
            needed: used.lds_bytes,
            available: caps.lds_bytes,
        });
    }
    if let (Some(work), Queried::Known(budget)) = (declared.serial_work, caps.max_dispatch_work)
        && work.total() > budget
    {
        return Err(ResourceRefusal::DispatchWork {
            work: work.total(),
            budget,
        });
    }
    Ok(())
}

/// Whether a device's matrix hardware runs `fragment`: a 16x16x16 tile on the families whose layout the
/// emitters implement. A family the emitters do not implement is a refusal, not a guess.
fn check_fragment(
    fragment: FragmentUse,
    tensor_core: TensorCoreSupport,
) -> Result<(), ResourceRefusal> {
    match tensor_core {
        TensorCoreSupport::Wmma16x16x16Rdna3 | TensorCoreSupport::NvidiaWmma16x16x16Sm80
            if fragment.shape == WmmaShape::M16N16K16 =>
        {
            Ok(())
        }
        TensorCoreSupport::UnknownNotExposedByApi => {
            Err(ResourceRefusal::FragmentCapabilityUnknown { fragment })
        }
        TensorCoreSupport::None
        | TensorCoreSupport::Wmma16x16x16Rdna3
        | TensorCoreSupport::NvidiaWmma16x16x16Sm80
        | TensorCoreSupport::Rdna4
        | TensorCoreSupport::CdnaMfma => Err(ResourceRefusal::FragmentUnsupported {
            fragment,
            tensor_core,
        }),
    }
}

#[cfg(test)]
mod tests;

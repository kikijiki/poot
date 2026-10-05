//! Graph-structure, execution-lane, and validation-packet admission checks.

#[cfg(test)]
use poot_target::Backend;

use crate::*;

/// One graph value copied into the canonical validation packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketSource {
    pub value: ValueId,
    pub first_lane: usize,
    pub lane_count: usize,
}

/// Backend-neutral, declaration-ordered packet layout and source plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationPacketPlan {
    pub layout: ValidationPacketLayout,
    pub sources: Vec<ValidationPacketSource>,
}

/// Plan the packet for a validation-bearing graph. Only `Graph<ValidationOutputs>` has a packet.
#[cfg(test)]
pub(crate) fn plan_validation_packet(
    g: &Graph<ValidationOutputs>,
) -> Result<ValidationPacketPlan, PlanError> {
    device_validation::validation_packet_plan(g)
}

/// Reject graph execution while generic executor buffers cannot represent authoritative typed bytes.
///
/// This is graph-wide rather than per-equation: an identity graph can return an input or constant with
/// zero equations, so `plan_eqn` is never called. Every legacy tensor-only executor called this before
/// binding host tensors. Storage-aware value entry points must instead validate their typed raw
/// representation. The portable host layout oracle and per-byte kernel conversions do not make an
/// `Arc<[f32]>` executor tensor safe to reinterpret as packed E4M3FN storage. Exact-I32 planning and
/// cross-target compilation are available, but production bind/replay and typed readback remain gated
/// until that representation is wired end to end.
///
/// `pub(crate)`/test-only: no production caller survives Card 546b's `GpuExecutor` deletion; kept for
/// the exact-i32/structural gating coverage in `tests.rs`.
#[cfg(test)]
pub(crate) fn validate_graph_execution<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<(), PlanError> {
    // Several executor entry points call this on a caller-supplied Graph before any prepared-graph
    // constructor has validated it. Validate first so capability inspection cannot mask the structural
    // error (or panic on a malformed value id).
    validate_graph_structure(g)?;
    if let Some(value) = g
        .values
        .iter()
        .position(|value| value.aval.dtype == DType::E4M3FN)
    {
        return Err(PlanError::UnrepresentableValue {
            value,
            dtype: DType::E4M3FN,
            gap: StorageGap::E4m3RawBytes,
        });
    }
    validate_graph_exact_i32(g)
}

/// Card 380: sibling to [`validate_graph_execution`] for the wgpu entry points that bind every value as a
/// plain `Arc<[f32]>` executor tensor: the (deleted) `GpuExecutor::run`/`bind_resident`, and so
/// `run_resident_kv`, `run_resident_kv_argmax`, `run_resident_prefill` and the cached decode paths.
///
/// It adds one rejection: a BF16-typed value. Card 380 gave BF16 a real wgpu representation (packed
/// `u32` lanes holding the checkpoint bytes verbatim), but only on the typed value walk, which uploads
/// it through `upload_typed_input` and checks every planned kernel against it with
/// `validate_typed_plan_storage`. A tensor-bound executor has neither half of that contract: it uploads
/// `numel` four-byte f32 words, so the card 380 contraction kernel would read an f32 buffer as packed
/// BF16 lanes. That buffer is larger than the kernel expects, so its bounds guard passes and the answer
/// is silently wrong rather than an out-of-bounds read.
///
/// Same reasoning as the E4M3FN rejection in [`validate_graph_execution`]: the generic executor tensor
/// does not carry the storage the dtype names, so reject by name rather than let a kernel
/// read a buffer at the wrong element width. It also covers the BF16 values
/// `widen_mismatched_matmul_dtypes` can introduce, via the `Cast(F32 -> BF16)` it puts on a
/// `Gather`/`Slice` table whose output stayed BF16; those reached SPIR-V codegen and failed with an
/// opaque "bf16 compute is NVPTX-only" and now fail here, naming the value.
///
/// Not part of [`validate_graph_execution`], which PTX and ROCm also call: PTX has actual two-byte BF16
/// storage and ROCm has `ConstLayout::Bf16Wmma`, so both run real BF16 graphs and must not be gated by
/// a wgpu limitation.
/// Card 380 P1 + the decode bf16 GEMV lane: reject a BF16 value on the wgpu tensor-bound entry points
/// unless every consumer reads it as packed u32 lanes the resident binder now stores that way.
///
/// Two narrow cases survive:
/// - a decode-GEMV weight ([`bf16_const_feeds_decode_bf16_gemv`]): `bind_resident` uploads packed u32
///   lanes and the planned `gemv*:bf16` body widens in-register;
/// - nothing else on this lane. DenseContraction/DenseRowGather weights stay typed-walk only (the
///   card 380 contract): the tensor lane does not plan those ops for a resident decode.
///
/// Every other BF16 value (a computed value, a const whose consumers are not both packed-lane) is
/// rejected by name rather than uploaded as four-byte f32 words for a kernel that reads two-byte
/// elements packed into u32 lanes.
///
/// `pub(crate)`/test-only: no production caller survives Card 546b's `GpuExecutor` deletion; kept for
/// the packed-BF16 contract coverage in `tests.rs`.
#[cfg(test)]
pub(crate) fn validate_wgpu_tensor_execution<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<(), PlanError> {
    validate_graph_execution(g)?;
    let caps = poot_target::DeviceCaps::wgpu_rdna3_igpu();
    if let Some(value) = (0..g.values.len()).find(|&id| {
        g.values[id].aval.dtype == DType::BF16
            && !bf16_const_feeds_decode_bf16_gemv(g, id, Backend::SpirvVulkan, &caps)
    }) {
        return Err(PlanError::UnrepresentableValue {
            value,
            dtype: DType::BF16,
            gap: StorageGap::Bf16PackedLanes,
        });
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_graph_structure<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<(), PlanError> {
    g.validate().map_err(PlanError::InvalidGraph)
}

#[cfg(test)]
pub(crate) fn validate_graph_exact_i32<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<(), PlanError> {
    let exact_i32 = ExactI32StorageAnalysis::new(g);
    if let Some(value) = (0..g.values.len()).find(|&id| exact_i32.required(id)) {
        return Err(PlanError::UnrepresentableValue {
            value,
            dtype: DType::I32,
            gap: StorageGap::ExactI32BindReplay,
        });
    }
    Ok(())
}

/// Card 259: is this flash-attention op's mask per head (one bias row/plane per query head, which ALiBi
/// needs) or the broadcast single row/plane every non-ALiBi caller passes? Read off the mask tensor's
/// shape, `[B, Hm, ..]`: `Hm == 1` is broadcast, `Hm == Hq` is per-head, anything else is a shape error
/// (reject clearly rather than misread a hot shared path).
///
/// The CPU oracle needs no copy of this rule: it evaluates the flash op from its decomposition, whose
/// additive mask broadcasts over the head axis exactly when `Hm == 1` (Card 556). The four flash
/// lowering arms turn the answer into element strides: a metadata word for the imported kernels, a
/// baked constant for the synthesized ones.
pub(crate) fn flash_mask_per_head(
    mask_shape: &[usize],
    hq: usize,
    what: &str,
) -> Result<bool, PlanError> {
    let hm = mask_shape.get(1).copied().unwrap_or(1);
    if hm == 1 {
        Ok(false)
    } else if hm == hq {
        Ok(true)
    } else {
        Err(PlanError::BadShape(format!(
            "{what} mask head axis must be 1 (broadcast) or Hq={hq} (per-head, ALiBi), got {hm} in \
             mask shape {mask_shape:?}"
        )))
    }
}

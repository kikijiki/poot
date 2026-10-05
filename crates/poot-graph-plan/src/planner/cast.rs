//! Dtype Cast equation planning.

use super::*;
use poot_target::{Backend, DeviceCaps};

/// Plan one Cast equation, matching the (input, output) dtype pair exhaustively.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    analysis: ExactI32Requirements<'_>,
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    caps: &DeviceCaps,
    limits: BodyLimits,
    packed_source: bool,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let planned = match &eqn.op {
        OpKind::Cast { to } => {
            // Each cast kernel hard-codes both element types, so this arm must match the (input,
            // output) dtype pair exhaustively. Dispatching on the target alone routed every non-F16
            // source of a Cast-to-F32 through `cast_bf16_to_f32`: an i32/i8 source's bytes
            // reinterpreted as packed bf16 halves, silent garbage on bf16-capable backends (the
            // "cast(i32->f32) emits a bf16 kernel" bug), and the BF16/F16 targets assumed an F32 source
            // the same way. A same-dtype cast is the identity, so it aliases like Reshape. Unwired
            // pairs are rejected by name.
            let in_dt = g.aval(ids[0]).dtype;
            let cast = |spec: CastSpec| {
                Planned::generated(
                    site,
                    KernelRequest::Elementwise(ElementwiseSpec::Cast(spec)),
                    backend,
                    caps,
                )
            };
            match (in_dt, *to) {
                (a, b) if a == b => Planned::alias(ids[0]),
                (DType::F32, DType::E4M3FN) => {
                    let row_len = out_shape.last().copied().unwrap_or(1);
                    if out_shape.contains(&0) || row_len == 0 {
                        return Err(site.refuse(Capability::ZeroElementE4m3));
                    }
                    // Spec 149 SC-005: the packed-row writer owns one physical u32 output word per
                    // thread, so the launch uses the row-padded word count, not out_numel.
                    cast(CastSpec::F32ToE4m3 {
                        out_shape: out_shape.to_vec(),
                        row_len,
                    })?
                }
                (DType::E4M3FN, DType::F32) => {
                    let input_shape = shape(ids[0]);
                    let row_len = input_shape.last().copied().unwrap_or(1);
                    if out_shape.contains(&0) || row_len == 0 {
                        return Err(site.refuse(Capability::ZeroElementE4m3));
                    }
                    cast(CastSpec::E4m3ToF32 {
                        row_len,
                        numel: out_numel,
                    })?
                }
                (DType::F32, DType::BF16) => cast(CastSpec::F32ToBf16 { numel: out_numel })?,
                (DType::F32, DType::F16) => cast(CastSpec::F32ToF16 { numel: out_numel })?,
                (DType::F16, DType::F32) => cast(CastSpec::F16ToF32 { numel: out_numel })?,
                // Card 1011: a BF16 const read only by packed readers is stored as packed `u32` lanes (the
                // storage plan's `bf16_const_is_packed`), so the cast reads those lanes and widens in
                // register; no F32 copy of the const is uploaded. Any other BF16 source (a computed value,
                // or a native-lane const) is the native two-byte cast.
                (DType::BF16, DType::F32) if packed_source => Planned::imported(
                    ImportedKernel::PackedBf16ToF32,
                    backend,
                    None,
                    [out_numel as u32, 1, 1],
                ),
                (DType::BF16, DType::F32) => cast(CastSpec::Bf16ToF32 { numel: out_numel })?,
                // Card 372c: the guarded selector Cast and the witness flag Cast. Exact for every lane
                // at or below 2^24, which the guard's bound and the device witness admission both hold.
                // The reverse pair stays unwired; nothing needs it.
                //
                // Like a Gather index, the source is read in the representation
                // `ExactI32StorageAnalysis` chose for it. Outside an exact component the source lives on
                // the f32-mirror lane, whose bytes already are the F32 result, so the cast aliases.
                // Reading it through `Slice<i32>` regardless voted the value to dense I32 storage while
                // its other readers (a mirror Gather index, a movement op) still read `Slice<f32>`
                // (spike-562 F-9).
                (DType::I32, DType::F32) if !analysis.required(ids[0]) => Planned::alias(ids[0]),
                (DType::I32, DType::F32) => cast(CastSpec::I32ToF32 { numel: out_numel })?,
                (_, _) => {
                    return Err(site.refuse(Capability::DtypeLowering));
                }
            }
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | moe_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only Cast here"),
    };
    Ok(planned)
}

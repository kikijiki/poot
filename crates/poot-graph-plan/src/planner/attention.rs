//! Attention equation planning: the kernel choice between the synthesized and imported flash
//! decode/prefill kernels.

use super::*;
use poot_target::{Backend, DeviceCaps, FLASH_LDS_CAP, FLASH_PREFILL_LDS_CAP};

/// Plan one attention equation (FlashAttentionDecode/FlashAttentionPrefill) and record which kernel it
/// takes.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    backend: Backend,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let planned = match &eqn.op {
        OpKind::FlashAttentionDecode { n_rep, scale } => {
            // out [B,Hq,1,D]; inputs q[B,Hq,1,D], k/v[B,Hkv,cap,D], mask[B,Hm,1,cap]. GQA + masked online
            // softmax.
            let bsz = out_shape[0];
            let hq = out_shape[1];
            let d = out_shape[out_shape.len() - 1];
            let cap = shape(ids[1])[2];
            let mask_per_head = flash_mask_per_head(&shape(ids[3]), hq, "flash decode")?;
            if d <= FLASH_LDS_CAP {
                // card 099d / spec 061, card 143 lever 1a: within the LDS cap the synthesizer generates
                // the flash decode Body (LDS-cooperative online softmax, `kg::flash_region_decode`),
                // batched (B>1, card 038) included: `B*Hq` workgroups of `flash_decode_width(d)` lanes
                // each. Card 557 moved this choice here from the `flash_region_synth` retag pass.
                Planned::generated(
                    site,
                    KernelRequest::Attention(AttentionSpec::RegionDecode {
                        bsz,
                        hq,
                        n_rep: *n_rep,
                        cap,
                        d,
                        scale_bits: scale.to_bits(),
                        width: flash_decode_width(d),
                        mask_per_head,
                    }),
                    backend,
                    caps,
                )?
            } else if bsz > 1 {
                // the kernelgen flash fallback (D > LDS cap) is single-sequence only; a batched decode with
                // such a head dim is not a real config (head dims are <= 128). Reject clearly.
                return Err(site.refuse(Capability::HeadDimExceedsLdsCap {
                    head_dim: d,
                    lds_cap: FLASH_LDS_CAP,
                }));
            } else {
                // The kernelgen flash fallback (D > LDS cap) holds its running output in a private
                // array: NVPTX is its intended target, and SPIR-V cannot state it (unimportable, and it
                // crashes SPIR-V codegen: wgpu and the raw-Vulkan runtime both go through
                // `Backend::SpirvVulkan`). `generate` reports that as `Unsupported`; it is refused here
                // by name instead of emitting a kernel that hangs the GPU.
                Planned::generated(
                    site,
                    KernelRequest::Attention(AttentionSpec::DecodeSingle {
                        bsz,
                        hq,
                        n_rep: *n_rep,
                        cap,
                        d,
                        scale_bits: scale.to_bits(),
                        mask_per_head,
                    }),
                    backend,
                    caps,
                )
                .map_err(|error| match error {
                    PlanError::Refused(refusal)
                        if matches!(
                            refusal.missing,
                            Capability::KernelGen(kg::KernelGenError::Unsupported { .. })
                        ) =>
                    {
                        site.refuse(Capability::HeadDimExceedsLdsCap {
                            head_dim: d,
                            lds_cap: FLASH_LDS_CAP,
                        })
                    }
                    other => other,
                })?
            }
        }
        OpKind::FlashAttentionPrefill {
            n_rep,
            scale,
            softcap,
        } => {
            // out [1,Hq,L,D]; inputs q[1,Hq,L,D], k/v[1,Hkv,L,D], mask[1,Hm,L,L] (`infer` admits only a
            // single sequence). GQA + masked online softmax, one workgroup per (head, query row): Hq*L
            // workgroups laid over a 2-D grid, never materializing the [1,Hq,L,L] scores.
            let hq = out_shape[1];
            let l = out_shape[2];
            let d = out_shape[out_shape.len() - 1];
            let mask_per_head = flash_mask_per_head(&shape(ids[3]), hq, "flash prefill")?;
            if d <= FLASH_PREFILL_LDS_CAP && softcap.is_none() {
                // card 099d / spec 061: the synthesizer generates the flash prefill Body (online softmax,
                // o[D] in LDS) via `kg::flash_region_prefill`. Its LDS `o[D]` scratch sizes dynamically to
                // `d`, verified up to the separate, higher `FLASH_PREFILL_LDS_CAP` (512, card 185); the
                // `x_groups` grid X width is baked into the kernel, so it is a plain Plan::Compute. Card
                // 198: the generated body has no softcap, so a softcapped prefill takes the imported
                // kernel below. Card 557 moved this choice here from the `flash_region_synth` retag pass.
                Planned::generated(
                    site,
                    KernelRequest::Attention(AttentionSpec::RegionPrefill {
                        hq,
                        l,
                        d,
                        n_rep: *n_rep,
                        scale_bits: scale.to_bits(),
                        mask_per_head,
                    }),
                    backend,
                    caps,
                )?
            } else {
                // The imported kernel holds each (head,row)'s running output `o[D]` in a fixed
                // 256-entry LDS array (`const LDS_SIZE: usize = 256` in `flash_prefill.rs`), so it runs
                // on both backends but `D` must fit the shared FLASH_LDS_CAP (256), not
                // FLASH_PREFILL_LDS_CAP. The kernel forms `gi = GroupY*x_groups + GroupX`, so `x_groups`
                // (the 2-D grid X width, <= the wgpu 65535 gridDim.x cap) rides in the meta and
                // long-context Hq*L spills onto Y, with no cap on context length on either backend.
                if d > FLASH_LDS_CAP {
                    return Err(site.refuse(Capability::HeadDimExceedsLdsCap {
                        head_dim: d,
                        lds_cap: FLASH_LDS_CAP,
                    }));
                }
                let (x_groups, y_groups) = gemv_grid(hq * l, caps);
                // Card 198: `has_softcap` + `cap_bits` ride as two extra meta words (indices 6-7); a
                // `softcap: None` op gets `[0, 0]` (has_softcap=0), so the kernel skips the softcap
                // branch.
                let (has_softcap, cap_bits) = match softcap {
                    Some(c) => (1u32, c.to_bits()),
                    None => (0u32, 0u32),
                };
                Planned::imported(
                    ImportedKernel::FlashPrefill,
                    backend,
                    Some(vec![
                        hq as u32,
                        l as u32,
                        d as u32,
                        *n_rep as u32,
                        scale.to_bits(),
                        x_groups as u32,
                        has_softcap,
                        cap_bits,
                        // card 259: mask head stride in elements: 0 for the broadcast `[1,1,L,L]` mask
                        // (`h*0 + row*L`), `L*L` for the per-head `[1,Hq,L,L]` ALiBi mask.
                        if mask_per_head { (l * l) as u32 } else { 0 },
                    ]),
                    [x_groups as u32, y_groups as u32, 1],
                )
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
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only attention ops here"),
    };
    Ok(planned)
}

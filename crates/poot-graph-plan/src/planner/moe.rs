//! MoE equation planning: the indexed matmul and the ArgTopK rank inversion.

use super::*;
use poot_target::{Backend, DeviceCaps};

/// Plan one MoE-family equation (IndexedMatMul/ArgTopK).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
    limits: BodyLimits,
) -> Result<Planned, PlanError> {
    let site = Site::new(g, eqn, backend, limits);
    let shape = |vid: ValueId| g.aval(vid).shape.clone();
    let ids = value_ids(eqn);
    let fty = fty_ty;
    let mut planned = match &eqn.op {
        OpKind::IndexedMatMul => {
            // card 088: x[M,K], W[E,K,N], idx[M] -> [M,N]; each row reads its own expert's rows in-kernel.
            let (x, w) = (shape(ids[0]), shape(ids[1]));
            let (m, k, n) = (x[0], x[1], w[2]);
            let spec = if is_indexed_gemv_in(g, backend, eqn, out_numel, caps) {
                // wgpu sparse-MoE decode: an LDS-reduction GEMV (GEMV_WIDTH lanes per output, coalesced
                // expert-weight reads) instead of the naive one-thread-per-output.
                ContractionSpec::IndexedGemv {
                    k,
                    n,
                    width: GEMV_WIDTH,
                    numel: out_numel,
                }
            } else {
                ContractionSpec::IndexedSerial {
                    dt: fty(odt),
                    m,
                    k,
                    n,
                    fold: elementwise_fold(g, backend, eqn, out_shape, out_numel, caps),
                }
            };
            Planned::generated(site, KernelRequest::Contraction(spec), backend, caps)?
        }
        OpKind::ArgTopK { k } => {
            // spec 136 P2: rank[..,E] -> [..,k] expert ids, one thread per output element, a naive
            // serial scan over E per thread (FR-003; routing is not the hot path, no perf work here).
            // `e` and `k` are both baked (trace-time constants); F32-only (fork c), so no `dt`/`fty`.
            let rank_shape = shape(ids[0]);
            let Some(&e) = rank_shape.last() else {
                return Err(PlanError::BadShape(
                    "ArgTopK rank operand needs a last (expert) axis, got a 0-D shape".into(),
                ));
            };
            Planned::generated(
                site,
                KernelRequest::TopK(TopKSpec {
                    e,
                    k: *k,
                    numel: out_numel,
                }),
                backend,
                caps,
            )?
        }
        OpKind::Iota { .. } => {
            unreachable!("plan_eqn refuses an unfolded Iota before dispatching to a family")
        }
        imported_ops!()
        | packed_ops!()
        | elementwise_ops!()
        | matmul_ops!()
        | attention_ops!()
        | cast_ops!()
        | movement_ops!()
        | sampling_ops!() => unreachable!("plan_eqn routes only MoE ops here"),
    };
    finalize(
        &mut planned,
        g,
        eqn,
        out_shape,
        out_numel,
        backend,
        odt,
        caps,
    )?;
    Ok(planned)
}

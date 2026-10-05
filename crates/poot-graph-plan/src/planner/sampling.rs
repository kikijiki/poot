//! Sampling equation planning (card 551a, R472-007): `RandomUniform` and `SampleToken` lower onto the
//! batched sampler bodies in `imported::sampling` (Card 447's original consolidation). Every
//! `SampleToken` rule dispatches one workgroup of 64 lanes per row (the bodies' own fixed shape; see
//! `finalize`'s `custom_grid` guard in the parent module, which keeps the shared launch postamble from
//! bumping this workgroup size). `RandomUniform` is a plain one-thread-per-output elementwise kernel.
//!
//! Small rows and the top-k/top-p/noise stages run inside these same bodies (bisection, not a separate
//! dispatch); there is no separate large-vocabulary two-stage schedule here (out of scope for this card -
//! see the card's final report).

use super::*;
use poot_graph_ir::op::SampleRule;
use poot_target::{Backend, DeviceCaps};

/// Plan one sampling-family equation (`RandomUniform`/`SampleToken`).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan(
    g: GraphTables<'_>,
    eqn: &Eqn,
    _out_shape: &[usize],
    out_numel: usize,
    backend: Backend,
    odt: DType,
    caps: &DeviceCaps,
) -> Result<Planned, PlanError> {
    let _ = caps;
    let ids = value_ids(eqn);
    let planned = match &eqn.op {
        OpKind::RandomUniform { cols } => Planned::imported(
            ImportedKernel::RandomUniform,
            backend,
            Some(vec![*cols as u32]),
            [out_numel as u32, 1, 1],
        ),
        OpKind::SampleToken { rule } => {
            // out_shape is [.., 2]; rows is every leading-axis element flattened, vocab the logits'
            // last-axis extent (not out_shape's - the output's own last axis is always 2).
            debug_assert_eq!(odt, DType::I32, "SampleToken output is always I32");
            let rows = out_numel / 2;
            let logits_shape = g.aval(ids[0]).shape.clone();
            let vocab = *logits_shape.last().unwrap_or(&0);
            let kernel = match rule {
                SampleRule::Greedy => ImportedKernel::ArgmaxBatched,
                SampleRule::Gumbel => ImportedKernel::SampleGumbelArgmaxBatched,
                SampleRule::GumbelTopK => ImportedKernel::SampleTruncatedGumbelArgmaxBatched,
                SampleRule::GumbelTopKTopP => ImportedKernel::SampleToppGumbelArgmaxBatched,
            };
            Planned::imported(
                kernel,
                backend,
                Some(vec![vocab as u32]),
                [(rows * 64) as u32, 1, 1],
            )
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
        | cast_ops!()
        | movement_ops!() => unreachable!("plan_eqn routes only sampling ops here"),
    };
    Ok(planned)
}

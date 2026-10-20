//! GLM-5.3-Flash's multi-token-prediction layer behind the `draft` program: the layer the checkpoint
//! stores after the text stack, which guesses the token after next from a position's hidden state
//! and the host-known token that follows it.
//!
//! The layer is the reference implementation's `Glm5NextMultiTokenPredictorLayer` (vLLM v0.31.0,
//! tag commit `db9527a4`, `vllm/models/glm5next/common/mtp.py`, read 2026-10-11; the block is
//! `Glm5NextDecoderLayer(is_mtp_layer=True)` in `model.py` beside it):
//!
//! 1. the token's embedding is zeroed at absolute position 0 (`fused_eh_norm`), then normalized by
//!    `enorm`; the position's hidden state is normalized by `hnorm`, and `eh_proj` maps
//!    `concat(enorm(embed), hnorm(hidden))`, embedding first, to the width;
//! 2. one DSA block (latent attention under the pooled indexer, its own cache) and the feed-forward
//!    of the last text layer's kind, each behind its pre-norm and joined to the residual, with no
//!    hyper-connection wrapper (`is_mtp_layer`);
//! 3. `shared_head.norm` over the block's output, then the target's head. That post-norm state is
//!    both the draft logits' input and the hidden state a chain recycles (vLLM PR 45895: recycling
//!    the pre-norm state lowers acceptance).
//!
//! The hidden state `h[p]` it reads is the target's closing-norm output at `p`, the mean of the
//! `hc_mult` streams normalized by the model's final norm (the hidden-state carry, POOT-1109). The
//! layer's own sparse indices are computed by every program (vLLM's `index_share_for_mtp_iteration`
//! is off by default and is not implemented: a chain step recomputes them).

use poot_graph_ir::{BinOp, Builder, Scalar, Traced};
use poot_tensor::DType;
use poot_weights::{MtpRole, WeightId, WeightMap, WeightRole};

use super::config::{Ffn, Glm53Config};
use super::stack::FfnWeights;
use crate::components::dsa::{PooledDsaWeights, pooled_dsa};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::norm;
use crate::components::standard::{LayerNorms, LayerParams, StackWeights, Step, standard_layer};
use crate::model::{LogitRows, ParamBlock, TraceError, WiringError};

/// The MTP layer's weights.
#[derive(Debug)]
pub(super) struct MtpLayer {
    /// The layer index past the text stack: its weight ids and its cache names.
    layer: usize,
    eh_proj: Weight,
    enorm: Weight,
    hnorm: Weight,
    head_norm: Weight,
    norms: LayerNorms,
    dsa: PooledDsaWeights,
    ffn: FfnWeights,
}

impl MtpLayer {
    pub(super) fn new(
        map: &WeightMap,
        layer: usize,
        ffn: Ffn,
        cfg: &Glm53Config,
    ) -> Result<Self, WeightError> {
        let width = cfg.width;
        let weight = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Mtp(role)), shape)
        };
        Ok(Self {
            layer,
            eh_proj: weight(MtpRole::EhProj, &[width, 2 * width])?,
            enorm: weight(MtpRole::ENorm, &[width])?,
            hnorm: weight(MtpRole::HNorm, &[width])?,
            head_norm: weight(MtpRole::HeadNorm, &[width])?,
            norms: LayerNorms::new(map, layer, width)?,
            dsa: PooledDsaWeights::new(
                map,
                layer,
                width,
                WiringError::require(&cfg.dsa, ParamBlock::Dsa)?,
            )?,
            ffn: FfnWeights::new(map, layer, ffn, cfg)?,
        })
    }

    /// The step's logits: the layer over the carry's first `tokens` rows and the step's host tokens.
    /// The step shape names the carry (`<dyn Model>::trace` refuses a draft step that does not).
    /// The post-norm state of the last position goes back into row 0 of the carry, the row a
    /// following one-token step reads; under [`LogitRows::Last`] only that position has logits.
    pub(super) fn trace(
        &self,
        b: &Builder,
        step: &Step,
        stack: &StackWeights,
        cfg: &Glm53Config,
    ) -> Result<Traced, TraceError> {
        let shape = step.shape();
        let (rows, tokens, width) = (shape.rows.get(), shape.tokens.get(), cfg.width);
        let carry = step
            .hidden_carry(b, width)?
            .ok_or(TraceError::Wiring(WiringError::MissingHiddenCarry))?;
        let hidden = step.carried_hidden(b, &carry)?;

        // The first position of a sequence has no token before it to follow: its embedding is
        // zero, whatever the row holds. The zeros come from the I32 positions (`pos * 0`, cast), not
        // from the embedding row, so a non-finite row at position 0 cannot reach the result.
        let embedded = stack.embed.embed(b, step.tokens())?;
        let pos = step.pos();
        let wide = |value: Traced| {
            Ok::<_, TraceError>(b.broadcast(
                b.reshape(value, vec![rows, tokens, 1])?,
                vec![rows, tokens, width],
            )?)
        };
        let zeros = b.cast(
            b.binary_scalar(BinOp::Mul, pos, Scalar::I32(0))?,
            DType::F32,
        )?;
        let masked = b.select(wide(pos)?, embedded, wide(zeros)?)?;
        let fused = linear(
            b,
            b.concat(
                2,
                &[
                    norm(b, masked, &self.enorm, cfg.norm)?,
                    norm(b, hidden, &self.hnorm, cfg.norm)?,
                ],
            )?,
            &self.eh_proj,
            None,
        )?;

        let _scope = b.layer_scope(self.layer);
        let block = standard_layer(
            b,
            fused,
            &self.norms,
            &LayerParams::new(cfg.norm),
            |b, normed| {
                pooled_dsa(
                    b,
                    step,
                    self.layer,
                    normed,
                    &self.dsa,
                    WiringError::require(&cfg.dsa, ParamBlock::Dsa)?,
                )
            },
            |b, input| self.ffn.trace(b, step, self.layer, input.block, cfg),
        )?;
        let normed = norm(b, block, &self.head_norm, cfg.norm)?;
        step.carry_last_hidden(b, &carry, normed)?;
        let normed = if shape.logits == LogitRows::Last && tokens > 1 {
            b.slice(normed, 1, tokens - 1, tokens)?
        } else {
            normed
        };
        stack.head.logits(b, normed)
    }
}

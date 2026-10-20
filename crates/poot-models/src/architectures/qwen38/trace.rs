//! Qwen4Exp's body: one function over a step, for a prompt, a chunk and one token alike.
//!
//! ```text
//! x = widen(embed(tokens))                                   // `streams` copies of the hidden state
//! for each layer:
//!     x += ple(x)                                            // the layers `ple_layer_ids` names
//!     x  = inject(x, mixer(input(x)))                        // GDN, or QSA attention
//!     x  = inject(x, moe(input(x)))
//! logits = head(collapse(x))
//! ```
//!
//! Every block calls its shared component: `components::gdn`, `components::attention` under the
//! QSA mask ([`qsa_select`](super::qsa::qsa_select)), `components::moe` and `components::ple`; the
//! gated residual around them is the family's own ([`gated_residual`](super::gated_residual)).

use poot_graph_ir::{BinOp, Builder, Traced};
use poot_weights::{HyperSite, WeightId, WeightMap, WeightRole};

use super::LayerKind;
use super::config::Qwen4ExpParams;
use super::gated_residual::{self, HyperWeights};
use super::qsa::{QsaWeights, qsa_select};
use crate::components::attention::{AttentionWeights, attention_selecting};
use crate::components::embed::Embedding;
use crate::components::gdn::{GdnWeights, gdn};
use crate::components::head::Head;
use crate::components::linear::WeightError;
use crate::components::moe::{self, MoeWeights};
use crate::components::ple::{PleWeights, ple};
use crate::components::standard::Step;
use crate::model::LogitRows;
use crate::model::TraceError;

/// A full-attention layer's mixer: the gated GQA attention and the indexer that narrows it.
#[derive(Debug)]
struct Full {
    attention: AttentionWeights,
    qsa: QsaWeights,
}

#[derive(Debug)]
enum Mixer {
    Gdn(Box<GdnWeights>),
    Full(Box<Full>),
}

#[derive(Debug)]
struct Layer {
    ple: Option<PleWeights>,
    attn_gate: HyperWeights,
    mlp_gate: HyperWeights,
    mixer: Mixer,
    moe: MoeWeights,
}

/// Every weight the body reads, shape-checked against the config.
#[derive(Debug)]
pub(super) struct Weights {
    embed: Embedding,
    head: Head,
    collapse: HyperWeights,
    layers: Vec<Layer>,
}

impl Weights {
    pub(super) fn new(map: &WeightMap, p: &Qwen4ExpParams) -> Result<Self, WeightError> {
        let gate = |site: HyperSite, layer: Option<usize>, inject: bool| {
            HyperWeights::new(
                map,
                |role| {
                    let role = WeightRole::Hyper(site, role);
                    match layer {
                        Some(l) => WeightId::layer(l, role),
                        None => WeightId::model(role),
                    }
                },
                &p.hyper,
                inject,
            )
        };
        let layers = p
            .kinds
            .iter()
            .enumerate()
            .map(|(l, kind)| {
                let mixer = match kind {
                    LayerKind::LinearAttention => {
                        Mixer::Gdn(Box::new(GdnWeights::new(map, l, p.width, &p.gdn)?))
                    }
                    LayerKind::FullAttention => Mixer::Full(Box::new(Full {
                        attention: AttentionWeights::new(map, l, p.width, &p.attention)?,
                        qsa: QsaWeights::new(map, l, p.width, &p.qsa)?,
                    })),
                };
                Ok(Layer {
                    ple: p.ple[l]
                        .as_ref()
                        .map(|ple| PleWeights::new(map, l, ple))
                        .transpose()?,
                    attn_gate: gate(HyperSite::Attn, Some(l), true)?,
                    mlp_gate: gate(HyperSite::Mlp, Some(l), true)?,
                    mixer,
                    moe: MoeWeights::new(map, l, &p.moe)?,
                })
            })
            .collect::<Result<Vec<_>, WeightError>>()?;
        Ok(Self {
            embed: Embedding::new(map, p.vocab, p.width)?,
            head: Head::new(map, p.vocab, p.width)?,
            collapse: gate(HyperSite::Final, None, false)?,
            layers,
        })
    }
}

/// The step's logits: `[rows, tokens, vocab]`, or the last token's under [`LogitRows::Last`].
pub(super) fn trace_step(
    b: &Builder,
    step: &Step,
    w: &Weights,
    p: &Qwen4ExpParams,
) -> Result<Traced, TraceError> {
    let hyper = &p.hyper;
    let mut x = hyper.widen(b, w.embed.embed(b, step.tokens())?)?;
    for (l, layer) in w.layers.iter().enumerate() {
        let _scope = b.layer_scope(l);
        // The PLE reads the raw multi-stream state, before the layer's first gate.
        if let (Some(weights), Some(params)) = (&layer.ple, &p.ple[l]) {
            x = b.binary(BinOp::Add, x, ple(b, step, l, x, weights, params)?)?;
        }
        let (input, weights) = gated_residual::input(b, x, &layer.attn_gate, hyper)?;
        let mixed = match &layer.mixer {
            Mixer::Gdn(gdn_weights) => gdn(b, step, l, input, gdn_weights, &p.gdn)?,
            Mixer::Full(full) => {
                attention_selecting(b, step, l, input, &full.attention, &p.attention, &|b| {
                    qsa_select(b, step, l, input, &full.qsa, &p.qsa)
                })?
            }
        };
        x = gated_residual::inject(b, x, mixed, weights)?;
        let (input, weights) = gated_residual::input(b, x, &layer.mlp_gate, hyper)?;
        let routed = moe::moe(b, step, l, input, None, &layer.moe, &p.moe)?;
        x = gated_residual::inject(b, x, routed, weights)?;
    }
    let tokens = step.shape().tokens.get();
    if step.shape().logits == LogitRows::Last && tokens > 1 {
        x = b.slice(x, 1, tokens - 1, tokens)?;
    }
    w.head
        .logits(b, gated_residual::collapse(b, x, &w.collapse, hyper)?)
}

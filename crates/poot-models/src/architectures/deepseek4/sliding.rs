//! The attention every DeepSeek-V4 layer shares: a low-rank query, one key-value head read as both
//! key and value, a sliding window over a carried cache, per-head attention sinks and a grouped
//! low-rank output projection. A compressed layer adds entries to what the query attends to
//! ([`super::hca`], [`super::csa`]); this file owns everything else.
//!
//! The last `rope_dim` dims of each head rotate (interleaved pairs). The key-value head is read as
//! the value too, so the attention output carries the rotation of the positions it attended to; the
//! output's rotated slice is rotated back by the query's position before the output projection.

use poot_graph_ir::{Builder, Traced, ops};
use poot_quant::weights::{AttnRole, MlaRole, WeightId, WeightMap, WeightRole};

use super::config::DeepseekV4Config;
use crate::components::attention::{carried_cache, first_writes};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::norm;
use crate::components::rope::{RopeRows, apply_rope};
use crate::components::standard::Step;
use crate::model::{TraceError, last_axis};

/// One layer's attention weights (the compressed layers' extras are [`super::hca`] and
/// [`super::csa`]'s).
#[derive(Debug)]
pub(super) struct LocalWeights {
    q_a: Weight,
    q_norm: Weight,
    q_b: Weight,
    kv: Weight,
    kv_norm: Weight,
    sinks: Weight,
    o_a: Weight,
    o_b: Weight,
}

impl LocalWeights {
    pub(super) fn new(
        map: &WeightMap,
        layer: usize,
        cfg: &DeepseekV4Config,
    ) -> Result<Self, WeightError> {
        let (width, a) = (cfg.width, &cfg.attn);
        let w = |role, shape: &[usize]| Weight::new(map, WeightId::layer(layer, role), shape);
        let heads_width = a.heads * a.head_dim;
        Ok(Self {
            q_a: w(WeightRole::Mla(MlaRole::QA), &[a.q_lora, width])?,
            q_norm: w(WeightRole::Mla(MlaRole::QANorm), &[a.q_lora])?,
            q_b: w(WeightRole::Mla(MlaRole::QB), &[heads_width, a.q_lora])?,
            kv: w(WeightRole::Attn(AttnRole::K), &[a.head_dim, width])?,
            kv_norm: w(WeightRole::Attn(AttnRole::KNorm), &[a.head_dim])?,
            sinks: w(WeightRole::Attn(AttnRole::Sinks), &[a.heads])?,
            o_a: w(
                WeightRole::Attn(AttnRole::OA),
                &[a.o_groups * a.o_lora, a.in_per_group()],
            )?,
            o_b: w(
                WeightRole::Attn(AttnRole::O),
                &[width, a.o_groups * a.o_lora],
            )?,
        })
    }
}

/// What a step computes of a layer's local branch and the compressed branches read: the rotated
/// queries, the normalized low-rank query the indexer also projects, the carried key-value cache and
/// the layer's rotation rows.
pub(super) struct Local {
    /// `[rows, heads, tokens, head_dim]`.
    pub q: Traced,
    /// `[rows, tokens, q_lora]`.
    pub q_res: Traced,
    /// `[rows, 1, capacity, head_dim]`: the cache after this step's keys are written.
    pub cache: Traced,
    pub rows: RopeRows,
    /// Where each row of the step starts writing (contiguous layouts), shared by every cache of the
    /// layer.
    pub write_at: Vec<Traced>,
}

/// Rotate the trailing `rotary` dims of `x` (`[rows, heads, tokens, dim]`) with `rows`: the leading
/// dims pass through. `rows` must have been made for a rotary width of `rotary`.
pub(super) fn rotate_tail(
    b: &Builder,
    x: Traced,
    rows: &RopeRows,
    rotary: usize,
) -> Result<Traced, TraceError> {
    let dim = last_axis("rotate_tail", &b.aval(x)?.shape)?;
    if dim == rotary {
        return apply_rope(b, x, rows);
    }
    let head = b.slice(x, 3, 0, dim - rotary)?;
    let tail = b.slice(x, 3, dim - rotary, dim)?;
    Ok(b.concat(3, &[head, apply_rope(b, tail, rows)?])?)
}

/// The local branch of layer `layer` over the normalized block input `normed` (`[rows, tokens,
/// width]`): queries, and this step's key-value written into the layer's carried cache.
pub(super) fn local(
    b: &Builder,
    step: &Step,
    layer: usize,
    normed: Traced,
    w: &LocalWeights,
    cfg: &DeepseekV4Config,
    rows: RopeRows,
) -> Result<Local, TraceError> {
    let shape = step.shape();
    let (r, t) = (shape.rows.get(), shape.tokens.get());
    let a = &cfg.attn;

    let q_res = norm(b, linear(b, normed, &w.q_a, None)?, &w.q_norm, cfg.norm)?;
    let q = linear(b, q_res, &w.q_b, None)?;
    let q = b.transpose(
        b.reshape(q, vec![r, t, a.heads, a.head_dim])?,
        vec![0, 2, 1, 3],
    )?;
    let q = rotate_tail(
        b,
        ops::rmsnorm_no_weight(b, q, cfg.norm.eps())?,
        &rows,
        a.rope_dim,
    )?;

    let kv = norm(b, linear(b, normed, &w.kv, None)?, &w.kv_norm, cfg.norm)?;
    let kv = rotate_tail(
        b,
        b.reshape(kv, vec![r, 1, t, a.head_dim])?,
        &rows,
        a.rope_dim,
    )?;
    let write_at = first_writes(b, step)?;
    let cache = carried_cache(b, step, &write_at, layer, "kv", kv, 1, a.head_dim)?;
    Ok(Local {
        q,
        q_res,
        cache,
        rows,
        write_at,
    })
}

/// The compressed entries a layer's queries also attend to: `[rows, 1, windows, head_dim]` keys
/// (read as values too) and the additive `[rows, 1, tokens, windows]` mask of the windows each query
/// token may read.
pub(super) struct Compressed {
    pub entries: Traced,
    pub mask: Traced,
}

/// Attend `local.q` over the layer's cache and, when the layer has any, its compressed entries, then
/// project the output: `[rows, tokens, width]`.
pub(super) fn attend(
    b: &Builder,
    step: &Step,
    local: &Local,
    compressed: Option<&Compressed>,
    w: &LocalWeights,
    cfg: &DeepseekV4Config,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (r, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let a = &cfg.attn;
    let window = ops::causal_mask_from_pos(b, step.pos(), cap, Some(a.window))?;
    let (keys, mask) = match compressed {
        None => (local.cache, window),
        Some(c) => (
            b.concat(2, &[local.cache, c.entries])?,
            b.concat(3, &[window, c.mask])?,
        ),
    };
    let out = ops::attention_prefill_with_sink(
        b,
        local.q,
        keys,
        keys,
        a.heads,
        a.scale,
        mask,
        w.sinks.declare_f32(b)?,
        a.heads,
        t,
    )?;
    // Undo the rotation the value picked up from the key it shares.
    let out = rotate_tail(b, out, &local.rows.inverse(b)?, a.rope_dim)?;
    let out = b.transpose(out, vec![0, 2, 1, 3])?;

    // Grouped first stage: `o_groups` independent `[in_per_group] -> [o_lora]` projections, the
    // stored `[groups * o_lora, in_per_group]` read as `groups` blocks of rows.
    let groups = b.transpose(
        b.reshape(out, vec![r * t, a.o_groups, a.in_per_group()])?,
        vec![1, 0, 2],
    )?;
    let blocks = b.reshape(
        w.o_a.declare(b)?,
        vec![a.o_groups, a.o_lora, a.in_per_group()],
    )?;
    let projected = b.matmul(groups, b.transpose(blocks, vec![0, 2, 1])?)?;
    let mixed = b.reshape(
        b.transpose(projected, vec![1, 0, 2])?,
        vec![r, t, a.o_groups * a.o_lora],
    )?;
    linear(b, mixed, &w.o_b, None)
}

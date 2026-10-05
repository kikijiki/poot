use super::*;

/// RoPE (non-interleaved half-split) on a per-head tensor `x[.., D]`. cos/sin are `[max_pos, D]`
/// constants; the row at `pos` is gathered (the only per-token input here).
pub fn rope(b: &Builder, x: Traced, cos_table: Traced, sin_table: Traced, pos: Traced) -> Traced {
    let shape = b.aval(x).shape;
    let d = *shape.last().expect("rope on a scalar");
    let last = shape.len() - 1;
    // Rotary width = cos/sin table width. Equals head_dim for every arch except phi3's partial rotary,
    // where only the leading `rot` dims rotate and `[rot, d)` passes through (see rope_partial).
    let rot = *b.aval(cos_table).shape.last().expect("rope table rank>=1");
    let cos = b.gather_scalar(cos_table, 0, pos); // [rot]
    let sin = b.gather_scalar(sin_table, 0, pos); // [rot]
    rope_partial(b, x, cos, sin, d, rot, last)
}

/// The shared rope core: apply (already-position-selected) `cos`/`sin` of width `rot` to `x`'s leading
/// `rot` dims via the half-split rotate, passing `[rot, d)` through unrotated. `rot == d` is the common
/// full-rotary case (no passthrough). `last` is x's last axis. cos/sin must broadcast against x[.., 0:rot].
///
/// The one rope definition (Card 556): [`rope`], [`rope_batched`], [`rope_prefill`] and, through
/// [`rope_sectioned_partial`], the sectioned variants all end here, and it is the decomposition of
/// `OpKind::Rope { rot }` (`crate::decompose`), so the half-split pairing is stated once.
pub fn rope_partial(
    b: &Builder,
    x: Traced,
    cos: Traced,
    sin: Traced,
    d: usize,
    rot: usize,
    last: usize,
) -> Traced {
    let half = rot / 2;
    let xr = if rot < d { b.slice(x, last, 0, rot) } else { x };
    let x1 = b.slice(xr, last, 0, half);
    let x2 = b.slice(xr, last, half, rot);
    let neg_x2 = b.unary(UnOp::Neg, x2);
    let rh = b.concat(last, &[neg_x2, x1]); // rotate_half
    let xc = b.binary(BinOp::Mul, xr, cos);
    let rs = b.binary(BinOp::Mul, rh, sin);
    let out = b.binary(BinOp::Add, xc, rs);
    if rot < d {
        let xp = b.slice(x, last, rot, d); // pass-through dims
        b.concat(last, &[out, xp])
    } else {
        out
    }
}

/// Batched-decode RoPE: like [`rope`] but `pos` is a `[B]` vector (one position per batch row) and `x` is
/// `[B, H, 1, D]`. The cos/sin gather by a `[B]` index yields `[B, D]`, reshaped to `[B, 1, 1, D]` to
/// broadcast over the head and query axes. Same primitives as [`rope`].
pub fn rope_batched(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos: Traced,
    batch: usize,
) -> Traced {
    let shape = b.aval(x).shape;
    let d = *shape.last().expect("rope on a scalar");
    let last = shape.len() - 1;
    let rot = *b.aval(cos_table).shape.last().expect("rope table rank>=1");
    let cos = b.gather(cos_table, 0, pos); // [B, rot]
    let sin = b.gather(sin_table, 0, pos); // [B, rot]
    let cos = b.reshape(cos, vec![batch, 1, 1, rot]);
    let sin = b.reshape(sin, vec![batch, 1, 1, rot]);
    rope_partial(b, x, cos, sin, d, rot, last)
}

/// Batched-decode sectioned ("mRoPE") RoPE: the `[B,H,1,D]` counterpart to [`rope_sectioned`].
/// Each entry in `pos` is one section's `[B]` position vector, reshaped to `[B,1,1]` so the shared
/// composition's gathers produce `[B,1,1,rot]` rows that broadcast over the head axis. The section
/// math stays in [`rope_sectioned`].
pub fn rope_sectioned_batched(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos: &[Traced],
    sections: &[usize],
    batch: usize,
) -> Traced {
    let pos: Vec<Traced> = pos
        .iter()
        .map(|&position| b.reshape(position, vec![batch, 1, 1]))
        .collect();
    rope_sectioned(b, x, cos_table, sin_table, &pos, sections)
}

/// Sectioned ("mRoPE") RoPE on a per-head tensor `x[.., D]`, generalizing [`rope`] from one shared
/// position id to `sections.len()` independent ones (Qwen2-VL/2.5-VL temporal/height/width position
/// ids; spec 267). `cos_table`/`sin_table` are the same `[max_pos, rot]` tables `rope` uses (one
/// `inv_freq` for all sections); only the gathered position id differs per section. `sections`
/// partitions the half-dim frequency axis `[0, rot/2)` into contiguous groups (must sum to `rot/2`);
/// `pos[j]` is section `j`'s per-token position id, one `Traced` per section
/// (`pos.len() == sections.len()`): a scalar for decode (as in `rope`) or a vector for a whole
/// sequence. Unlike `rope_prefill`'s implicit `0..L`, mRoPE height/width ids are not contiguous
/// within an image grid, so they are passed explicitly. All positions equal reduces bit-exactly to
/// `rope`/`rope_prefill` (see [`rope_sectioned_partial`]).
pub fn rope_sectioned(
    b: &Builder,
    x: Traced,
    cos_table: Traced,
    sin_table: Traced,
    pos: &[Traced],
    sections: &[usize],
) -> Traced {
    assert_eq!(
        pos.len(),
        sections.len(),
        "rope_sectioned: one position id per section (got {} positions, {} sections)",
        pos.len(),
        sections.len()
    );
    let shape = b.aval(x).shape;
    let d = *shape.last().expect("rope on a scalar");
    let last = shape.len() - 1;
    let rot = *b.aval(cos_table).shape.last().expect("rope table rank>=1");
    let cos_secs: Vec<Traced> = pos.iter().map(|&p| b.gather(cos_table, 0, p)).collect();
    let sin_secs: Vec<Traced> = pos.iter().map(|&p| b.gather(sin_table, 0, p)).collect();
    rope_sectioned_partial(b, x, &cos_secs, &sin_secs, sections, d, rot, last)
}

/// The sectioned-rope core: reassemble one `rot`-wide cos/sin row from `sections.len()` independently
/// gathered rows (`cos_secs[j]`/`sin_secs[j]`, each position-selected for section `j` at full `rot`
/// width), then hand off to [`rope_partial`]. Section `j` covering half-dim columns
/// `[start_j, start_j+w_j)` contributes `cos_secs[j]`'s columns at `[start_j, start_j+w_j)` (first
/// half) and `[half+start_j, half+start_j+w_j)` (the mirrored second half). Concatenating all
/// first-half slices then all second-half slices in section order restores the original column order.
///
/// If every `cos_secs[j]` is the same row (all position ids equal), the slices partition `[0, half)`
/// and `[half, rot)` exactly once with no arithmetic, so the row is reconstructed bit-exactly and the
/// result matches [`rope`]/[`rope_prefill`].
#[allow(clippy::too_many_arguments)]
pub fn rope_sectioned_partial(
    b: &Builder,
    x: Traced,
    cos_secs: &[Traced],
    sin_secs: &[Traced],
    sections: &[usize],
    d: usize,
    rot: usize,
    last: usize,
) -> Traced {
    assert_eq!(cos_secs.len(), sections.len(), "one cos row per section");
    assert_eq!(sin_secs.len(), sections.len(), "one sin row per section");
    let half = rot / 2;
    let total: usize = sections.iter().sum();
    assert_eq!(
        total, half,
        "rope_sectioned: section widths must sum to rot/2 ({half}), got {total}"
    );
    let cos_last = b.aval(cos_secs[0]).shape.len() - 1;
    let sin_last = b.aval(sin_secs[0]).shape.len() - 1;

    let mut cos_first = Vec::with_capacity(sections.len());
    let mut cos_second = Vec::with_capacity(sections.len());
    let mut sin_first = Vec::with_capacity(sections.len());
    let mut sin_second = Vec::with_capacity(sections.len());
    let mut start = 0;
    for (j, &w) in sections.iter().enumerate() {
        cos_first.push(b.slice(cos_secs[j], cos_last, start, start + w));
        cos_second.push(b.slice(cos_secs[j], cos_last, half + start, half + start + w));
        sin_first.push(b.slice(sin_secs[j], sin_last, start, start + w));
        sin_second.push(b.slice(sin_secs[j], sin_last, half + start, half + start + w));
        start += w;
    }
    cos_first.extend(cos_second);
    sin_first.extend(sin_second);
    let cos = b.concat(cos_last, &cos_first);
    let sin = b.concat(sin_last, &sin_first);
    rope_partial(b, x, cos, sin, d, rot, last)
}

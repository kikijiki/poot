use super::*;

/// Add the local sliding-window floor to a full-causal decode mask in-graph (card 151), so LOCAL layers window
/// while GLOBAL layers keep the raw mask, with no second `Slot::Mask` (GPU-capture aliasing hazard) and no
/// per-backend binder change. The window `w` is a compile-time constant: `floor[.,t] = -1e9 * max(0,
/// (pos - w + 1) - t)` is large-negative where key slot `t <= pos - w` is too old. `pos_f` (f32 position(s)) is
/// gathered from the in-graph `iota` at `Slot::Pos` (`iota[pos] == pos`, card 550a), avoiding an i32->f32 `Cast`, which
/// mis-lowers to bf16 on wgpu. `iota_cap` is `[0,1,...,cap-1]`. Returns `mask_4d + floor`. `rows` is 1 for
/// single-seq decode (scalar pos) or `batch` for batched shared-pool decode (`[batch]` pos). Same arithmetic as
/// the Gemma 3 decode per-layer window floor (verified by `gemma3_decode_per_layer_mask_selection`). Inert
/// (all-zero floor) whenever `w >= cap`.
pub fn gemma4_local_window_floor(
    b: &Builder,
    mask_4d: Traced,
    pos_f: Traced,
    iota_cap: Traced,
    window: usize,
    rows: usize,
    cap: usize,
) -> Traced {
    let pos_f = b.reshape(pos_f, vec![rows, 1]);
    let thresh = b.binary_scalar(BinOp::Sub, pos_f, Scalar::F32(window as f32 - 1.0));
    let thresh = b.broadcast(thresh, vec![rows, cap]);
    let iota_row = b.reshape(iota_cap, vec![1, cap]);
    let iota_row = b.broadcast(iota_row, vec![rows, cap]);
    let arg = b.binary(BinOp::Sub, thresh, iota_row); // (pos - w + 1) - t
    let relu = b.binary_scalar(BinOp::Max, arg, Scalar::F32(0.0));
    let floor = b.binary_scalar(BinOp::Mul, relu, Scalar::F32(-1e9));
    let floor = b.reshape(floor, vec![rows, 1, 1, cap]);
    b.binary(BinOp::Add, mask_4d, floor)
}

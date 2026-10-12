//! The Qwen2.5-VL vision tower behind [`Model`], a model whose only phase is [`Phase::Encode`]: the
//! 3-D patch embedding, a stack of RMSNorm transformer blocks with 2-D rotary attention (windowed in
//! all but a few blocks, full per frame in those), and the patch merger. One step is `rows` images
//! (or videos) of the same patch count; each returns the `patches / merge^2` embedding rows the text
//! tower takes in place of the prompt's `<|image_pad|>` / `<|video_pad|>` placeholders
//! (`StepInput::TokensWithEmbeds`).
//!
//! The checkpoint holds the tower beside the text tower (`visual.` beside `model.`), which a second
//! registry resolve builds as the `qwen2_5_vl` family; this family reads its own rows (`visual.`) and
//! ignores the rest. Its config is the checkpoint's `vision_config` with `model_type` set to
//! [`FAMILY`]'s key by the caller.
//!
//! # Inputs
//!
//! An image's patches arrive flattened by the processor in merge-cell-major order (each consecutive
//! `merge^2` patches are one `merge x merge` cell of the patch grid, cells in row-major order). The
//! tower attends in windows, and the windows are not contiguous in that order, so the caller hands the
//! step the patches already permuted into window order together with the grid facts the graph needs:
//! each patch's grid row and column ([`POS_H_SLOT`], [`POS_W_SLOT`], the 2-D rotary positions), the
//! window it sits in ([`WINDOW_SLOT`]) and its frame ([`FRAME_SLOT`]). All of that is
//! [`PatchGeometry`], computed from the grid by [`Qwen25VlVisionParams::geometry`], which also maps
//! the tower's rows back to the processor's order. Windows are the block-diagonal attention mask of
//! the windowed blocks and frames the mask of the full-attention blocks; the graph builds both from
//! the ids.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;

use poot_graph_ir::rope_table::{RopeFlavor, RopeSpec, RopeTable};
use poot_graph_ir::{
    BinOp, Builder, ComputedConst, Graph, Scalar, Slot, TensorType, Traced, UnOp, ops,
};
use poot_quant::weights::{
    AttnRole, FfnRole, NormRole, WeightId, WeightMap, WeightRole, WeightStore,
};
use poot_tensor::DType;

use super::idefics3::PIXELS_SLOT;
use crate::components::dims::{Field, Fields};
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::{NormKind, NormParams, norm};
use crate::model::WiringError;
use crate::model::{
    ConfigReason, FamilyKey, Model, ModelConfig, ModelError, ModelOutput, Phase, ShapeReason,
    StepShape, TraceError,
};
use crate::names::{NameRow, Presence, Source, family_weights};
use crate::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig};
use poot_text::ChatFormat;

/// The Qwen2.5-VL vision tower's [`FamilyKey`]: a family of its own beside the text tower's.
pub const FAMILY: FamilyKey = FamilyKey::new("qwen2_5_vl_vision");

/// The `Slot::Activation` inputs of a step beside [`PIXELS_SLOT`] (`[rows, patches, patch_dim]` F32,
/// each patch flattened `[channel, t, i, j]`, the convolution weight's own layout): the patch's grid
/// row and column (`[rows, patches]` I32), its attention window and its frame (`[rows, patches]` F32
/// ids, equal ids attend to each other).
pub const POS_H_SLOT: &str = "pos_h";
/// The grid column of each patch, `[rows, patches]` I32 (the width-axis rotary position).
pub const POS_W_SLOT: &str = "pos_w";
/// The attention window id of each patch, `[rows, patches]` F32.
pub const WINDOW_SLOT: &str = "window";
/// The frame id of each patch, `[rows, patches]` F32 (the mask of the full-attention blocks).
pub const FRAME_SLOT: &str = "frame";

/// The base of the vision tower's rotary frequencies (`VisionRotaryEmbedding`'s default).
const ROTARY_THETA: f32 = 10_000.0;

/// The score added where an attention mask forbids a key: large enough to weigh zero after the softmax
/// of any score the tower produces, finite so no NaN can arise.
const MASKED: f32 = -1.0e9;

/// The most patches one step takes. Attention is dense over a step's patches (the windows are a mask),
/// so a step's score matrices grow with the square of this; a larger image is the caller's to refuse
/// or downscale. A step's graph is traced per patch count (the grid's shape is bound, not traced), so the
/// driver prepares one entry per distinct patch count and a stream of differently sized images compiles a
/// program per size, bounded by the driver's prepared-set limits.
pub const MAX_PATCHES: usize = 4096;

pub(crate) const ENTRY: FamilyEntry = FamilyEntry {
    family: FAMILY,
    keys: &[(ConfigSource::HfModelType, "qwen2_5_vl_vision")],
    build,
    fixture,
    variants: &[],
};

const fn tensor(role: WeightRole, name: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::Tensor(name),
        presence: Presence::Required,
    }
}

const fn linear_row(role: WeightRole, stem: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerLinear {
            prefix: "visual.blocks.",
            stem,
        },
        presence: Presence::Required,
    }
}

const fn layer_row(role: WeightRole, suffix: &'static str) -> NameRow {
    NameRow {
        role,
        source: Source::LayerTensor {
            prefix: "visual.blocks.",
            suffix,
        },
        presence: Presence::Required,
    }
}

const HF_ROWS: &[NameRow] = &[
    tensor(WeightRole::Patch, "visual.patch_embed.proj.weight"),
    tensor(WeightRole::FinalNorm, "visual.merger.ln_q.weight"),
    tensor(WeightRole::Connector, "visual.merger.mlp.0.weight"),
    tensor(WeightRole::ConnectorBias, "visual.merger.mlp.0.bias"),
    tensor(WeightRole::ConnectorOut, "visual.merger.mlp.2.weight"),
    tensor(WeightRole::ConnectorOutBias, "visual.merger.mlp.2.bias"),
    layer_row(WeightRole::Norm(NormRole::Attn), ".norm1.weight"),
    layer_row(WeightRole::Norm(NormRole::Ffn), ".norm2.weight"),
    linear_row(WeightRole::Attn(AttnRole::Qkv), ".attn.qkv"),
    layer_row(WeightRole::Attn(AttnRole::QkvBias), ".attn.qkv.bias"),
    linear_row(WeightRole::Attn(AttnRole::O), ".attn.proj"),
    layer_row(WeightRole::Attn(AttnRole::OBias), ".attn.proj.bias"),
    linear_row(WeightRole::Ffn(FfnRole::Gate), ".mlp.gate_proj"),
    layer_row(WeightRole::Ffn(FfnRole::GateBias), ".mlp.gate_proj.bias"),
    linear_row(WeightRole::Ffn(FfnRole::Up), ".mlp.up_proj"),
    layer_row(WeightRole::Ffn(FfnRole::UpBias), ".mlp.up_proj.bias"),
    linear_row(WeightRole::Ffn(FfnRole::Down), ".mlp.down_proj"),
    layer_row(WeightRole::Ffn(FfnRole::DownBias), ".mlp.down_proj.bias"),
];

const DEPTH: Field = Field::hf("depth");
const WIDTH: Field = Field::hf("hidden_size");
const INTER: Field = Field::hf("intermediate_size");
const HEADS: Field = Field::hf("num_heads");
const CHANNELS: Field = Field::hf("in_chans");
const PATCH: Field = Field::hf("patch_size");
const TEMPORAL: Field = Field::hf("temporal_patch_size");
const MERGE: Field = Field::hf("spatial_merge_size");
const WINDOW: Field = Field::hf("window_size");
const OUT_WIDTH: Field = Field::hf("out_hidden_size");
const FULL_ATTENTION: &str = "fullatt_block_indexes";

/// The tower's typed config: every value checked once, here, so tracing cannot fail on it.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen25VlVisionParams {
    width: usize,
    heads: usize,
    head_dim: usize,
    inter: usize,
    layers: usize,
    channels: usize,
    patch_size: usize,
    temporal: usize,
    merge: usize,
    /// Merge cells along each side of a window.
    window_cells: usize,
    full_attention: BTreeSet<usize>,
    out_width: usize,
    norm: NormParams,
}

impl Qwen25VlVisionParams {
    /// Reads the typed config from the checkpoint's `vision_config`; a missing or inconsistent
    /// field, or a full-attention block index past the layer count, is a typed [`ModelError::Config`].
    pub fn from_raw(raw: &RawConfig<'_>) -> Result<Self, ModelError> {
        let RawConfig::HfJson { config, .. } = raw else {
            return Err(ModelError::Config {
                family: FAMILY,
                field: "general.architecture",
                reason: ConfigReason::Unsupported,
            });
        };
        let fields = Fields::new(raw, FAMILY);
        let width = fields.count(WIDTH, None)?;
        let heads = fields.count(HEADS, None)?;
        if !width.is_multiple_of(heads) {
            return Err(fields.error(fields.key(HEADS), ConfigReason::NotDivisible { by: heads }));
        }
        let head_dim = width / heads;
        // The rotary width is half the head, split evenly between the two grid axes, and each axis
        // rotates pairs of its quarter.
        if !head_dim.is_multiple_of(4) {
            return Err(fields.error(
                fields.key(HEADS),
                ConfigReason::NotDivisible { by: 4 * heads },
            ));
        }
        let layers = fields.count(DEPTH, None)?;
        let (patch_size, merge) = (fields.count(PATCH, None)?, fields.count(MERGE, None)?);
        let window = fields.count(WINDOW, None)?;
        let cell = patch_size * merge;
        if !window.is_multiple_of(cell) {
            return Err(fields.error(fields.key(WINDOW), ConfigReason::NotDivisible { by: cell }));
        }
        if let Some(act) = config.get("hidden_act").and_then(|a| a.as_str())
            && act != "silu"
        {
            return Err(fields.error("hidden_act", ConfigReason::Unsupported));
        }
        let full_attention = full_attention_blocks(&fields, config, layers)?;
        Ok(Self {
            width,
            heads,
            head_dim,
            inter: fields.count(INTER, None)?,
            layers,
            channels: fields.count(CHANNELS, Some(3))?,
            patch_size,
            temporal: fields.count(TEMPORAL, Some(2))?,
            merge,
            window_cells: window / cell,
            full_attention,
            out_width: fields.count(OUT_WIDTH, None)?,
            norm: NormParams::of(NormKind::Rms, 1e-6).map_err(|e| e.for_family(FAMILY))?,
        })
    }

    /// The values one patch flattens to: `channels * temporal * patch^2`.
    pub fn patch_dim(&self) -> usize {
        self.channels * self.temporal * self.patch_size * self.patch_size
    }

    /// The width of a returned embedding row: the text tower's hidden size.
    pub fn out_width(&self) -> usize {
        self.out_width
    }

    /// Patches that merge into one returned row: `merge^2`.
    pub fn merge_unit(&self) -> usize {
        self.merge * self.merge
    }

    /// The side of the merge cell: `merge_unit` is its square.
    pub fn spatial_merge_size(&self) -> usize {
        self.merge
    }

    /// The geometry of an image or video of `grid` patches (`[t, h, w]`, the processor's grid).
    pub fn geometry(&self, grid: [usize; 3]) -> Result<PatchGeometry, GridError> {
        PatchGeometry::new(grid, self.merge, self.window_cells)
    }
}

/// The blocks that attend over the whole frame instead of a window: the list in `vision_config`,
/// each an index of a block.
fn full_attention_blocks(
    fields: &Fields<'_>,
    config: &serde_json::Value,
    layers: usize,
) -> Result<BTreeSet<usize>, ModelError> {
    let list = config
        .get(FULL_ATTENTION)
        .ok_or_else(|| fields.error(FULL_ATTENTION, ConfigReason::Missing))?
        .as_array()
        .ok_or_else(|| fields.error(FULL_ATTENTION, ConfigReason::WrongType))?;
    let mut blocks = BTreeSet::new();
    for index in list {
        let index = index
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .ok_or_else(|| fields.error(FULL_ATTENTION, ConfigReason::WrongType))?;
        if index >= layers {
            return Err(fields.error(FULL_ATTENTION, ConfigReason::Exceeds { max: layers - 1 }));
        }
        blocks.insert(index);
    }
    Ok(blocks)
}

/// Why an image grid has no geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GridError {
    /// A grid with a zero dimension.
    #[error("a grid dimension of {0:?} is zero")]
    Zero([usize; 3]),
    /// A grid whose height or width is not a multiple of the merge size.
    #[error("grid {grid:?}: height and width must be multiples of the merge size {merge}")]
    NotMergeable {
        /// The refused `[t, h, w]` grid.
        grid: [usize; 3],
        /// The spatial merge size.
        merge: usize,
    },
}

/// The grid facts one image's step binds, in window order (see the module docs): the order of the
/// rows the tower returns for it, and the inverse that restores the processor's order.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchGeometry {
    merge_unit: usize,
    /// `cells[i]` is the merge cell (in the processor's order) at window-order position `i`.
    cells: Vec<usize>,
    pos_h: Vec<i32>,
    pos_w: Vec<i32>,
    window: Vec<f32>,
    frame: Vec<f32>,
}

impl PatchGeometry {
    fn new([t, h, w]: [usize; 3], merge: usize, window_cells: usize) -> Result<Self, GridError> {
        if t == 0 || h == 0 || w == 0 {
            return Err(GridError::Zero([t, h, w]));
        }
        if !h.is_multiple_of(merge) || !w.is_multiple_of(merge) {
            return Err(GridError::NotMergeable {
                grid: [t, h, w],
                merge,
            });
        }
        let (rows, cols) = (h / merge, w / merge);
        let merge_unit = merge * merge;
        let cells_per_frame = rows * cols;
        let windows_across = cols.div_ceil(window_cells);
        // Window order: frame by frame, then the windows of the frame in row-major order, each
        // window's cells in row-major order; a window cut by the grid's edge holds fewer cells.
        let mut cells = Vec::with_capacity(t * cells_per_frame);
        let mut window = Vec::with_capacity(t * cells_per_frame * merge_unit);
        let mut next_window = 0usize;
        for frame in 0..t {
            for window_row in 0..rows.div_ceil(window_cells) {
                for window_col in 0..windows_across {
                    for row in window_row * window_cells..rows.min((window_row + 1) * window_cells)
                    {
                        for col in
                            window_col * window_cells..cols.min((window_col + 1) * window_cells)
                        {
                            cells.push(frame * cells_per_frame + row * cols + col);
                            window.extend(std::iter::repeat_n(next_window as f32, merge_unit));
                        }
                    }
                    next_window += 1;
                }
            }
        }
        let frames = (0..t)
            .flat_map(|frame| std::iter::repeat_n(frame as f32, cells_per_frame * merge_unit));
        // A patch's grid position from its place in the processor's order: the cell's row and
        // column, then the patch's offset inside the cell.
        let (mut pos_h, mut pos_w) = (
            Vec::with_capacity(window.len()),
            Vec::with_capacity(window.len()),
        );
        for &cell in &cells {
            let in_frame = cell % cells_per_frame;
            let (row, col) = (in_frame / cols, in_frame % cols);
            for offset in 0..merge_unit {
                pos_h.push((row * merge + offset / merge) as i32);
                pos_w.push((col * merge + offset % merge) as i32);
            }
        }
        Ok(Self {
            merge_unit,
            cells,
            pos_h,
            pos_w,
            window,
            frame: frames.collect(),
        })
    }

    /// Patches of the image: the step's token count.
    pub fn patches(&self) -> usize {
        self.pos_h.len()
    }

    /// Each patch's grid row, in window order: the height-axis rotary position.
    pub fn pos_h(&self) -> &[i32] {
        &self.pos_h
    }

    /// Each patch's grid column, in window order: the width-axis rotary position.
    pub fn pos_w(&self) -> &[i32] {
        &self.pos_w
    }

    /// Each patch's attention window id, in window order; equal ids attend to each other.
    pub fn window(&self) -> &[f32] {
        &self.window
    }

    /// Each patch's frame id, in window order; equal ids attend to each other.
    pub fn frame(&self) -> &[f32] {
        &self.frame
    }

    /// The processor's `patches` (`patch_dim` values each, merge-cell-major) in window order.
    pub fn to_window_order(&self, patches: &[f32], patch_dim: usize) -> Vec<f32> {
        let cell = self.merge_unit * patch_dim;
        debug_assert_eq!(patches.len(), self.cells.len() * cell);
        self.cells
            .iter()
            .flat_map(|&c| &patches[c * cell..(c + 1) * cell])
            .copied()
            .collect()
    }

    /// The tower's merged `rows` (`width` values each, one per cell, in window order) in the
    /// processor's order: the order the prompt's placeholders are in.
    pub fn from_window_order(&self, rows: &[f32], width: usize) -> Vec<f32> {
        debug_assert_eq!(rows.len(), self.cells.len() * width);
        let mut out = vec![0.0; rows.len()];
        for (i, &c) in self.cells.iter().enumerate() {
            out[c * width..(c + 1) * width].copy_from_slice(&rows[i * width..(i + 1) * width]);
        }
        out
    }
}

#[derive(Debug)]
struct Block {
    norm_attn: Weight,
    norm_ffn: Weight,
    qkv: Weight,
    qkv_bias: Weight,
    proj: Weight,
    proj_bias: Weight,
    gate: Weight,
    gate_bias: Weight,
    up: Weight,
    up_bias: Weight,
    down: Weight,
    down_bias: Weight,
}

/// The built Qwen2.5-VL vision tower: config, name-resolved weights, the patch embedding, the
/// transformer blocks and the merger. Owns its weights; holds no per-request state.
#[derive(Debug)]
pub struct Qwen25VlVision {
    params: Qwen25VlVisionParams,
    weights: WeightMap,
    patch: Weight,
    merger_norm: Weight,
    merger_up: Weight,
    merger_up_bias: Weight,
    merger_out: Weight,
    merger_out_bias: Weight,
    blocks: Vec<Block>,
    config: ModelConfig,
}

fn build(raw: &RawConfig<'_>, store: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let p = Qwen25VlVisionParams::from_raw(raw)?;
    let weights = family_weights(FAMILY, store, &[HF_ROWS], p.layers, false)?;
    let weight_error = |e: WeightError| e.for_family(FAMILY);
    let (w, inter) = (p.width, p.inter);
    let merged = w * p.merge_unit();
    let model_weight = |role, shape: &[usize]| Weight::new(&weights, WeightId::model(role), shape);
    let blocks = (0..p.layers)
        .map(|l| {
            let layer =
                |role, shape: &[usize]| Weight::new(&weights, WeightId::layer(l, role), shape);
            Ok(Block {
                norm_attn: layer(WeightRole::Norm(NormRole::Attn), &[w])?,
                norm_ffn: layer(WeightRole::Norm(NormRole::Ffn), &[w])?,
                qkv: layer(WeightRole::Attn(AttnRole::Qkv), &[3 * w, w])?,
                qkv_bias: layer(WeightRole::Attn(AttnRole::QkvBias), &[3 * w])?,
                proj: layer(WeightRole::Attn(AttnRole::O), &[w, w])?,
                proj_bias: layer(WeightRole::Attn(AttnRole::OBias), &[w])?,
                gate: layer(WeightRole::Ffn(FfnRole::Gate), &[inter, w])?,
                gate_bias: layer(WeightRole::Ffn(FfnRole::GateBias), &[inter])?,
                up: layer(WeightRole::Ffn(FfnRole::Up), &[inter, w])?,
                up_bias: layer(WeightRole::Ffn(FfnRole::UpBias), &[inter])?,
                down: layer(WeightRole::Ffn(FfnRole::Down), &[w, inter])?,
                down_bias: layer(WeightRole::Ffn(FfnRole::DownBias), &[w])?,
            })
        })
        .collect::<Result<Vec<_>, WeightError>>()
        .map_err(weight_error)?;
    let rest = || -> Result<_, WeightError> {
        Ok((
            model_weight(
                WeightRole::Patch,
                &[w, p.channels, p.temporal, p.patch_size, p.patch_size],
            )?,
            model_weight(WeightRole::FinalNorm, &[w])?,
            model_weight(WeightRole::Connector, &[merged, merged])?,
            model_weight(WeightRole::ConnectorBias, &[merged])?,
            model_weight(WeightRole::ConnectorOut, &[p.out_width, merged])?,
            model_weight(WeightRole::ConnectorOutBias, &[p.out_width])?,
        ))
    };
    let (patch, merger_norm, merger_up, merger_up_bias, merger_out, merger_out_bias) =
        rest().map_err(weight_error)?;
    let config = ModelConfig {
        family: FAMILY,
        vocab: 0,
        max_positions: MAX_PATCHES,
        eos: BTreeSet::new(),
        bos: None,
        prompt_bos: None,
        output: ModelOutput::Hidden { width: p.out_width },
        prefill_granule: NonZeroUsize::MIN,
        chat: ChatFormat::ChatML,
    };
    Ok(Box::new(Qwen25VlVision {
        params: p,
        weights,
        patch,
        merger_norm,
        merger_up,
        merger_up_bias,
        merger_out,
        merger_out_bias,
        blocks,
        config,
    }))
}

/// The additive attention mask `[rows, 1, patches, patches]` of `ids` (`[rows, patches]` F32): `0`
/// where two patches carry the same id, [`MASKED`] elsewhere.
/// The refusal of a value whose rank is not the one `operation` reads.
fn rank_error(operation: &'static str, expected: usize, found: Vec<usize>) -> TraceError {
    poot_graph_ir::BuilderError::Requirement {
        operation,
        requirement: poot_graph_ir::Requirement::Rank { expected, found },
    }
    .into()
}

fn same_id_mask(b: &Builder, ids: Traced) -> Result<Traced, TraceError> {
    let ids_shape = b.aval(ids)?.shape;
    let [rows, patches] = ids_shape[..] else {
        return Err(rank_error("same_id_mask", 2, ids_shape));
    };
    let square = vec![rows, 1, patches, patches];
    let along_rows = b.broadcast(b.reshape(ids, vec![rows, 1, patches, 1])?, square.clone())?;
    let along_columns = b.broadcast(b.reshape(ids, vec![rows, 1, 1, patches])?, square)?;
    // `a == b` is `a >= b` and `b >= a`.
    let same = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Ge, along_rows, along_columns)?,
        b.binary(BinOp::Ge, along_columns, along_rows)?,
    )?;
    let different = b.binary_scalar(BinOp::Sub, same, Scalar::F32(1.0))?;
    Ok(b.binary_scalar(BinOp::Mul, different, Scalar::F32(-MASKED))?)
}

/// `x` (`[rows, heads, patches, head_dim]`) rotated by the 2-D rotary table: the first quarter of the
/// head's pairs rotate at the patch's grid row, the second at its grid column, the half-split pairing
/// of the reference's `rotate_half`.
fn rotate(b: &Builder, x: Traced, cos: Traced, sin: Traced) -> Result<Traced, TraceError> {
    let d = b.aval(x)?.shape[3];
    let half = d / 2;
    let (x1, x2) = (b.slice(x, 3, 0, half)?, b.slice(x, 3, half, d)?);
    let rotated = b.concat(3, &[b.unary(UnOp::Neg, x2)?, x1])?;
    Ok(b.binary(
        BinOp::Add,
        b.binary(BinOp::Mul, x, cos)?,
        b.binary(BinOp::Mul, rotated, sin)?,
    )?)
}

impl Qwen25VlVision {
    /// The cos and sin tables `[rows, 1, patches, head_dim]` of the patches' grid positions.
    fn rotary(
        &self,
        b: &Builder,
        pos_h: Traced,
        pos_w: Traced,
    ) -> Result<(Traced, Traced), TraceError> {
        let p = &self.params;
        let pos_shape = b.aval(pos_h)?.shape;
        let [rows, patches] = pos_shape[..] else {
            return Err(rank_error("rotary", 2, pos_shape));
        };
        let quarter = p.head_dim / 4;
        let spec = RopeSpec::new(p.head_dim / 2, ROTARY_THETA, &RopeFlavor::Plain)?;
        let positions =
            u32::try_from(patches).map_err(|_| WiringError::PatchesOutOfRange { patches })?;
        let table = |which| {
            let full = b.computed(ComputedConst::Rope {
                table: which,
                positions,
                spec,
            })?;
            let axis =
                |pos| Ok::<_, TraceError>(b.slice(b.gather(full, 0, pos)?, 2, 0, quarter)?);
            let (h, w) = (axis(pos_h)?, axis(pos_w)?);
            Ok::<_, TraceError>(b.reshape(
                b.concat(2, &[h, w, h, w])?,
                vec![rows, 1, patches, p.head_dim],
            )?)
        };
        Ok((table(RopeTable::Cos)?, table(RopeTable::Sin)?))
    }

    fn attention(
        &self,
        b: &Builder,
        block: &Block,
        x: Traced,
        (cos, sin): (Traced, Traced),
        mask: Traced,
    ) -> Result<Traced, TraceError> {
        let p = &self.params;
        let x_shape = b.aval(x)?.shape;
        let [rows, patches, w] = x_shape[..] else {
            return Err(rank_error("attention", 3, x_shape));
        };
        let qkv = linear(b, x, &block.qkv, Some(&block.qkv_bias))?;
        let part = |i: usize| {
            let part = b.slice(qkv, 2, i * w, (i + 1) * w)?;
            let split = b.reshape(part, vec![rows, patches, p.heads, p.head_dim])?;
            Ok::<_, TraceError>(b.transpose(split, vec![0, 2, 1, 3])?)
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let (q, k) = (rotate(b, q, cos, sin)?, rotate(b, k, cos, sin)?);
        let scale = 1.0 / (p.head_dim as f32).sqrt();
        let out = ops::attention_masked(b, q, k, v, 1, scale, mask)?;
        let merged = b.reshape(b.transpose(out, vec![0, 2, 1, 3])?, vec![rows, patches, w])?;
        linear(b, merged, &block.proj, Some(&block.proj_bias))
    }
}

impl Model for Qwen25VlVision {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, phase: Phase, shape: StepShape, b: Builder) -> Result<Graph, TraceError> {
        let p = &self.params;
        if phase != Phase::Encode {
            return Err(TraceError::PhaseUnsupported {
                family: FAMILY,
                phase,
            });
        }
        let refuse = |reason| {
            Err(TraceError::ShapeUnsupported {
                family: FAMILY,
                shape,
                reason,
            })
        };
        let patches = shape.tokens.get();
        if patches > MAX_PATCHES {
            return refuse(ShapeReason::CapacityAboveMax { max: MAX_PATCHES });
        }
        if !patches.is_multiple_of(p.merge_unit()) {
            return refuse(ShapeReason::Granule {
                granule: p.merge_unit(),
            });
        }
        let rows = shape.rows.get();
        let w = p.width;
        let slot =
            |tag, ty: TensorType| Ok::<_, TraceError>(b.slot_named(Slot::Activation, tag, ty)?);
        let pixels = slot(
            PIXELS_SLOT,
            TensorType::f32(vec![rows, patches, p.patch_dim()]),
        )?;
        let ids = vec![rows, patches];
        let pos_h = slot(POS_H_SLOT, TensorType::new(ids.clone(), DType::I32))?;
        let pos_w = slot(POS_W_SLOT, TensorType::new(ids.clone(), DType::I32))?;
        let window = slot(WINDOW_SLOT, TensorType::f32(ids.clone()))?;
        let frame = slot(FRAME_SLOT, TensorType::f32(ids))?;
        let window_mask = same_id_mask(&b, window)?;
        let frame_mask = same_id_mask(&b, frame)?;
        let rotary = self.rotary(&b, pos_h, pos_w)?;

        // The 3-D convolution as a matmul: its stride is its kernel, so each patch is one dot
        // product against the flattened kernel.
        let kernel = b.reshape(self.patch.declare_f32(&b)?, vec![w, p.patch_dim()])?;
        let mut x = b.matmul(pixels, b.transpose(kernel, vec![1, 0])?)?;
        for (l, block) in self.blocks.iter().enumerate() {
            let _scope = b.layer_scope(l);
            let mask = if p.full_attention.contains(&l) {
                frame_mask
            } else {
                window_mask
            };
            let attn = self.attention(
                &b,
                block,
                norm(&b, x, &block.norm_attn, p.norm)?,
                rotary,
                mask,
            )?;
            x = b.binary(BinOp::Add, x, attn)?;
            let h = norm(&b, x, &block.norm_ffn, p.norm)?;
            let gate = linear(&b, h, &block.gate, Some(&block.gate_bias))?;
            let up = linear(&b, h, &block.up, Some(&block.up_bias))?;
            let down = linear(
                &b,
                ops::swiglu(&b, gate, up)?,
                &block.down,
                Some(&block.down_bias),
            )?;
            x = b.binary(BinOp::Add, x, down)?;
        }
        // The merger: normalize each patch, concatenate each cell's patches (consecutive in window
        // order), and two projections around an exact GELU.
        let normed = norm(&b, x, &self.merger_norm, p.norm)?;
        let cells = b.reshape(
            normed,
            vec![rows, patches / p.merge_unit(), w * p.merge_unit()],
        )?;
        let up = linear(&b, cells, &self.merger_up, Some(&self.merger_up_bias))?;
        let out = linear(
            &b,
            ops::gelu_erf(&b, up)?,
            &self.merger_out,
            Some(&self.merger_out_bias),
        )?;
        Ok(b.finish(out)?)
    }
}

/// The fixture's `vision_config`; see [`fixture`].
pub(crate) fn fixture_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen2_5_vl_vision",
        "depth": 4,
        "hidden_act": "silu",
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_heads": 2,
        "in_chans": 3,
        "out_hidden_size": 64,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "window_size": 8,
        "fullatt_block_indexes": [1, 3],
    })
}

/// A tiny tower: 16-wide blocks of two 8-wide heads over 2-pixel patches that merge 2 x 2 into
/// 64-wide rows (the width of the qwen2 fixture's text tower), windows of 2 x 2 cells, and blocks 1
/// and 3 attending over the whole frame.
fn fixture() -> Fixture {
    let config = fixture_config();
    let n = |f: &str| crate::registry::Fixture::count(&config, f);
    let (w, inter, c, t, patch) = (
        n("hidden_size"),
        n("intermediate_size"),
        n("in_chans"),
        n("temporal_patch_size"),
        n("patch_size"),
    );
    let merged = w * n("spatial_merge_size").pow(2);
    let out = n("out_hidden_size");
    let mut shapes = vec![
        (
            "visual.patch_embed.proj.weight".to_string(),
            vec![w, c, t, patch, patch],
        ),
        ("visual.merger.ln_q.weight".to_string(), vec![w]),
        (
            "visual.merger.mlp.0.weight".to_string(),
            vec![merged, merged],
        ),
        ("visual.merger.mlp.0.bias".to_string(), vec![merged]),
        ("visual.merger.mlp.2.weight".to_string(), vec![out, merged]),
        ("visual.merger.mlp.2.bias".to_string(), vec![out]),
    ];
    for l in 0..n("depth") {
        let k = |s: &str| format!("visual.blocks.{l}.{s}");
        shapes.extend([
            (k("norm1.weight"), vec![w]),
            (k("norm2.weight"), vec![w]),
            (k("attn.qkv.weight"), vec![3 * w, w]),
            (k("attn.qkv.bias"), vec![3 * w]),
            (k("attn.proj.weight"), vec![w, w]),
            (k("attn.proj.bias"), vec![w]),
            (k("mlp.gate_proj.weight"), vec![inter, w]),
            (k("mlp.gate_proj.bias"), vec![inter]),
            (k("mlp.up_proj.weight"), vec![inter, w]),
            (k("mlp.up_proj.bias"), vec![inter]),
            (k("mlp.down_proj.weight"), vec![w, inter]),
            (k("mlp.down_proj.bias"), vec![w]),
        ]);
    }
    Fixture::bf16(config, shapes)
}

#[cfg(test)]
mod tests {
    use poot_quant::weights::{DenseWeight, WeightEntry};

    use super::*;
    use crate::components::standard::oracle::{self, assert_matches_f64};
    use crate::components::testing::{build as built, values};
    use crate::model::{KvLayout, LogitRows};

    /// The image the rows below run: two frames of a 6 x 8 patch grid, so 3 x 4 merge cells per frame
    /// against 2 x 2-cell windows (windows cut by the bottom edge), 96 patches in all.
    const GRID: [usize; 3] = [2, 6, 8];

    fn params() -> Qwen25VlVisionParams {
        Qwen25VlVisionParams::from_raw(&fixture().raw()).unwrap()
    }

    fn get(store: &WeightStore, name: &str) -> Vec<f64> {
        values(store.get(name).unwrap_or_else(|| panic!("{name}")))
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// A patch's frame, window (frame and cell block) and grid position, from its place in the
    /// processor's order.
    struct PatchFacts {
        frame: usize,
        window: (usize, usize, usize),
        position: (usize, usize),
    }

    /// `x [rows, k] @ w [out, k]^T + bias`.
    fn dense(x: &[f64], w: &[f64], bias: &[f64]) -> Vec<f64> {
        let (out, k) = (bias.len(), w.len() / bias.len());
        x.chunks(k)
            .flat_map(|row| {
                (0..out).map(move |o| bias[o] + (0..k).map(|i| row[i] * w[o * k + i]).sum::<f64>())
            })
            .collect()
    }

    fn rms_norm(x: &[f64], width: usize, scale: &[f64]) -> Vec<f64> {
        x.chunks(width)
            .flat_map(|row| {
                let mean = row.iter().map(|v| v * v).sum::<f64>() / width as f64;
                let inv = 1.0 / (mean + 1e-6).sqrt();
                row.iter().zip(scale).map(move |(v, s)| v * inv * s)
            })
            .collect()
    }

    fn erf(x: f64) -> f64 {
        let (mut term, mut sum) = (x, x);
        for n in 1..80 {
            term *= -x * x / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    }

    /// The tower on `pixels` (the processor's patches in its own order), written from the reference
    /// modeling code in f64 from the checkpoint's values, with no permutation: each patch's window,
    /// frame and grid position come straight from its place in the processor's order (a frame of
    /// `rows x cols` merge cells, each a `merge x merge` block of consecutive patches). Returns
    /// `[cells, out_width]` in the processor's order.
    fn reference(store: &WeightStore, pixels: &[f32], [t, h, w]: [usize; 3]) -> Vec<f64> {
        let p = params();
        let (merge, wc, hd) = (p.merge, p.window_cells, p.head_dim);
        let width = p.width;
        let (rows, cols) = (h / merge, w / merge);
        let n = t * h * w;
        // Each patch's frame, window and grid position.
        let facts: Vec<PatchFacts> = (0..n)
            .map(|patch| {
                let (cell, offset) = (patch / (merge * merge), patch % (merge * merge));
                let (frame, in_frame) = (cell / (rows * cols), cell % (rows * cols));
                let (cy, cx) = (in_frame / cols, in_frame % cols);
                PatchFacts {
                    frame,
                    window: (frame, cy / wc, cx / wc),
                    position: (cy * merge + offset / merge, cx * merge + offset % merge),
                }
            })
            .collect();
        let v = |s: &str| get(store, &format!("visual.{s}"));
        let mut x = dense(
            &pixels.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            &v("patch_embed.proj.weight"),
            &vec![0.0; width],
        );
        // The rotation of the reference's `apply_rotary_pos_emb_vision`: pair `i` of the head's `hd / 2`
        // rotates dims `i` and `i + hd / 2` by the row position (the first quarter) or the column
        // position (the second) times `theta^(-2 i / (hd / 2))`.
        let angle = |position: (usize, usize), pair: usize| {
            let quarter = hd / 4;
            let (axis, index) = if pair < quarter {
                (position.0, pair)
            } else {
                (position.1, pair - quarter)
            };
            axis as f64 * f64::from(ROTARY_THETA).powf(-2.0 * index as f64 / (hd / 2) as f64)
        };
        let rotate = |head: &[f64], position: (usize, usize)| {
            let mut out = head.to_vec();
            for pair in 0..hd / 2 {
                let (cos, sin) = (angle(position, pair).cos(), angle(position, pair).sin());
                let (a, b) = (head[pair], head[pair + hd / 2]);
                out[pair] = a * cos - b * sin;
                out[pair + hd / 2] = b * cos + a * sin;
            }
            out
        };
        for l in 0..p.layers {
            let k = |s: &str| v(&format!("blocks.{l}.{s}"));
            let normed = rms_norm(&x, width, &k("norm1.weight"));
            let qkv = dense(&normed, &k("attn.qkv.weight"), &k("attn.qkv.bias"));
            let full = p.full_attention.contains(&l);
            let mut merged = vec![0.0; n * width];
            for head in 0..p.heads {
                let at = |patch: usize, part: usize| {
                    let from = patch * 3 * width + part * width + head * hd;
                    qkv[from..from + hd].to_vec()
                };
                let q: Vec<Vec<f64>> = (0..n)
                    .map(|i| rotate(&at(i, 0), facts[i].position))
                    .collect();
                let key: Vec<Vec<f64>> = (0..n)
                    .map(|i| rotate(&at(i, 1), facts[i].position))
                    .collect();
                for i in 0..n {
                    let visible: Vec<usize> = (0..n)
                        .filter(|&j| {
                            if full {
                                facts[i].frame == facts[j].frame
                            } else {
                                facts[i].window == facts[j].window
                            }
                        })
                        .collect();
                    let scores: Vec<f64> = visible
                        .iter()
                        .map(|&j| {
                            (0..hd).map(|d| q[i][d] * key[j][d]).sum::<f64>() / (hd as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exp.iter().sum();
                    for d in 0..hd {
                        merged[i * width + head * hd + d] = visible
                            .iter()
                            .zip(&exp)
                            .map(|(&j, e)| e / sum * at(j, 2)[d])
                            .sum();
                    }
                }
            }
            let attn = dense(&merged, &k("attn.proj.weight"), &k("attn.proj.bias"));
            for (xi, a) in x.iter_mut().zip(&attn) {
                *xi += a;
            }
            let normed = rms_norm(&x, width, &k("norm2.weight"));
            let gate = dense(
                &normed,
                &k("mlp.gate_proj.weight"),
                &k("mlp.gate_proj.bias"),
            );
            let up = dense(&normed, &k("mlp.up_proj.weight"), &k("mlp.up_proj.bias"));
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let down = dense(&act, &k("mlp.down_proj.weight"), &k("mlp.down_proj.bias"));
            for (xi, d) in x.iter_mut().zip(&down) {
                *xi += d;
            }
        }
        let normed = rms_norm(&x, width, &v("merger.ln_q.weight"));
        // The merge concatenates each cell's consecutive patches: the same values, read `4 * width`
        // at a time.
        let up = dense(&normed, &v("merger.mlp.0.weight"), &v("merger.mlp.0.bias"));
        let act: Vec<f64> = up
            .iter()
            .map(|&u| 0.5 * u * (1.0 + erf(u / 2f64.sqrt())))
            .collect();
        dense(&act, &v("merger.mlp.2.weight"), &v("merger.mlp.2.bias"))
    }

    fn encode_shape(rows: usize, patches: usize) -> StepShape {
        StepShape {
            rows: NonZeroUsize::new(rows).unwrap(),
            tokens: NonZeroUsize::new(patches).unwrap(),
            capacity: NonZeroUsize::new(patches).unwrap(),
            kv: KvLayout::Contiguous,
            logits: LogitRows::All,
            hidden: None,
        }
    }

    /// The tower of `fx` on one image of `grid`: the step's merged rows in the processor's order.
    fn run(fx: &Fixture, pixels: &[f32], grid: [usize; 3]) -> Vec<f32> {
        run_over(fx, &fx.store, pixels, grid)
    }

    /// [`run`] with the weights of `store` in place of the fixture's.
    fn run_over(fx: &Fixture, store: &WeightStore, pixels: &[f32], grid: [usize; 3]) -> Vec<f32> {
        let model = built(&ENTRY, &fx.raw(), store);
        let params = Qwen25VlVisionParams::from_raw(&fx.raw()).unwrap();
        let geometry = params.geometry(grid).unwrap();
        let g = model
            .trace(
                Phase::Encode,
                encode_shape(1, geometry.patches()),
                poot_graph_ir::TraceLimits::for_tests(),
            )
            .unwrap();
        let ordered = geometry.to_window_order(pixels, params.patch_dim());
        let (rows, _) = oracle::eval_slots(
            &g,
            store,
            model.weights(),
            &[],
            &[],
            (
                &[
                    ("activation.pos_h", geometry.pos_h()),
                    ("activation.pos_w", geometry.pos_w()),
                ],
                &[
                    ("activation.pixels", &ordered),
                    ("activation.window", geometry.window()),
                    ("activation.frame", geometry.frame()),
                ],
            ),
            &[],
        );
        geometry.from_window_order(&rows, params.out_width())
    }

    fn pixels() -> Vec<f32> {
        oracle::random(GRID.iter().product::<usize>() * params().patch_dim(), 11)
            .into_iter()
            .map(|v| v * 8.0)
            .collect()
    }

    /// SC-003 (reference half): on an image whose windows are cut by the grid's edge and whose two
    /// frames the full-attention blocks keep apart, the tower equals an f64 forward written from the
    /// reference code with no permutation. Mutation: give the windowed blocks the frame mask (drop
    /// the window attention mask); every windowed block then lets patches see the whole frame and
    /// the rows diverge.
    #[test]
    fn the_tower_equals_an_independent_f64_forward() {
        let fx = fixture();
        let pixels = pixels();
        let got = run(&fx, &pixels, GRID);
        let want = reference(&fx.store, &pixels, GRID);
        assert_eq!(got.len(), 24 * 64);
        assert_matches_f64(&got, &want, 1e-4);
    }

    fn bump(store: &WeightStore, key: &str) -> WeightStore {
        let mut out = WeightStore::builder();
        for (name, entry) in store.iter() {
            let entry = if name.as_str() == key {
                let WeightEntry::Dense(dense) = entry else {
                    panic!("the fixture is dense")
                };
                let bytes: Vec<u8> = dense
                    .bytes()
                    .as_slice()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|&pair| {
                        let value = f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16);
                        (((value + 0.3).to_bits() >> 16) as u16).to_le_bytes()
                    })
                    .collect();
                WeightEntry::Dense(
                    DenseWeight::try_new(dense.dtype(), dense.shape().to_vec(), bytes.into())
                        .unwrap(),
                )
            } else {
                entry.clone()
            };
            out.insert(name.clone(), entry).unwrap();
        }
        out.build()
    }

    /// SC-003 (perturbation half): every weight of the tower is read by the output: adding an offset to
    /// any one tensor of the checkpoint changes the merged rows. Mutation: leave the gate bias out of
    /// a block's feed-forward (`Some(&block.gate_bias)` to `None`); that tensor's row goes red.
    #[test]
    fn every_weight_changes_the_output() {
        let fx = fixture();
        let pixels = pixels();
        let base = run(&fx, &pixels, GRID);
        let keys: Vec<String> = fx.store.keys().map(|k| k.as_str().to_string()).collect();
        assert_eq!(keys.len(), 6 + 4 * 12);
        for key in keys {
            let changed = run_over(&fx, &bump(&fx.store, &key), &pixels, GRID);
            let moved = changed
                .iter()
                .zip(&base)
                .filter(|(a, b)| (*a - *b).abs() > 1e-6)
                .count();
            assert!(moved > 0, "the output does not depend on {key}");
        }
    }

    /// Two patches of one window influence each other and two of different windows do not, in a
    /// windowed block: changing the pixels of one patch moves exactly the rows of its window's cells
    /// in a tower whose blocks are all windowed, and the rows of its whole frame (and no other frame)
    /// when a block attends over the frame.
    #[test]
    fn a_patch_reaches_its_window_and_its_frame_by_block_kind() {
        let p = params();
        let pixels = pixels();
        let mut touched = pixels.clone();
        // Patch 0 is the first patch of the first cell of frame 0, window 0.
        for v in &mut touched[..p.patch_dim()] {
            *v += 3.0;
        }
        let moved = |fixture: &Fixture| -> Vec<usize> {
            let a = run(fixture, &pixels, GRID);
            let b = run(fixture, &touched, GRID);
            (0..24)
                .filter(|&cell| (0..64).any(|i| (a[cell * 64 + i] - b[cell * 64 + i]).abs() > 1e-7))
                .collect()
        };
        // All blocks windowed: frame 0's window 0 holds cells (0,0), (0,1), (1,0), (1,1) of a
        // 3 x 4 cell frame: cells 0, 1, 4, 5.
        let mut windowed = fixture();
        windowed.config["fullatt_block_indexes"] = serde_json::json!([]);
        assert_eq!(moved(&windowed), [0, 1, 4, 5]);
        // The default fixture's blocks 1 and 3 see the whole frame: all 12 cells of frame 0.
        assert_eq!(moved(&fixture()), (0..12).collect::<Vec<_>>());
    }

    /// The geometry of a small grid by hand: a 4 x 8 patch grid is 2 x 4 merge cells with 2 x 2-cell
    /// windows, so window order visits cells 0 1 4 5 then 2 3 6 7.
    #[test]
    fn the_geometry_orders_cells_by_window() {
        let g = params().geometry([1, 4, 8]).unwrap();
        assert_eq!(g.cells, [0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(g.patches(), 32);
        assert_eq!(
            g.window(),
            [[0.0; 16], [1.0; 16]].concat(),
            "two windows of four cells"
        );
        assert!(g.frame().iter().all(|&f| f == 0.0));
        // The first cell's four patches sit at rows 0..2 and columns 0..2 (row-major inside the cell),
        // and the second window's first cell is cell 2, at columns 4..6.
        assert_eq!(&g.pos_h()[..4], [0, 0, 1, 1]);
        assert_eq!(&g.pos_w()[..4], [0, 1, 0, 1]);
        assert_eq!(&g.pos_h()[16..20], [0, 0, 1, 1]);
        assert_eq!(&g.pos_w()[16..20], [4, 5, 4, 5]);
        // Window order and back is the identity on rows.
        let rows: Vec<f32> = (0..8 * 3).map(|v| v as f32).collect();
        let reordered: Vec<f32> = g
            .cells
            .iter()
            .flat_map(|&c| rows[c * 3..c * 3 + 3].to_vec())
            .collect();
        assert_eq!(g.from_window_order(&reordered, 3), rows);
        // A grid the merge does not divide, or an empty one, has no geometry.
        assert_eq!(
            params().geometry([1, 3, 4]),
            Err(GridError::NotMergeable {
                grid: [1, 3, 4],
                merge: 2
            })
        );
        assert_eq!(
            params().geometry([0, 2, 2]),
            Err(GridError::Zero([0, 2, 2]))
        );
    }

    /// A phase other than `Encode`, a patch count the merge does not divide and one past the cap are
    /// typed refusals.
    #[test]
    fn a_step_it_cannot_trace_is_a_typed_refusal() {
        let fx = fixture();
        let model = built(&ENTRY, &fx.raw(), &fx.store);
        assert!(matches!(
            model.trace(
                Phase::Prefill,
                encode_shape(1, 16),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::PhaseUnsupported {
                phase: Phase::Prefill,
                ..
            })
        ));
        assert!(matches!(
            model.trace(
                Phase::Encode,
                encode_shape(1, 18),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported {
                reason: ShapeReason::Granule { granule: 4 },
                ..
            })
        ));
        assert!(matches!(
            model.trace(
                Phase::Encode,
                encode_shape(1, MAX_PATCHES + 4),
                poot_graph_ir::TraceLimits::for_tests()
            ),
            Err(TraceError::ShapeUnsupported {
                reason: ShapeReason::CapacityAboveMax { .. },
                ..
            })
        ));
    }

    /// Each config fact the tower reads is checked once, naming its field.
    #[test]
    fn a_config_the_tower_cannot_run_is_refused_naming_the_field() {
        let fx = fixture();
        let refused = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut config = fixture_config();
            edit(&mut config);
            let raw = RawConfig::HfJson {
                config: &config,
                generation: None,
            };
            build(&raw, &fx.store)
                .expect_err("the config is refused")
                .to_string()
        };
        // A window that is not a whole number of merge cells.
        assert!(refused(&|c| c["window_size"] = 9.into()).contains("window_size"));
        // A full-attention block the tower does not have.
        assert!(
            refused(&|c| c["fullatt_block_indexes"] = serde_json::json!([4]))
                .contains("fullatt_block_indexes")
        );
        assert!(
            refused(&|c| {
                c.as_object_mut().unwrap().remove("fullatt_block_indexes");
            })
            .contains("fullatt_block_indexes")
        );
        // A head that cannot split into the two rotary axes.
        assert!(refused(&|c| c["num_heads"] = 8.into()).contains("num_heads"));
        assert!(refused(&|c| c["hidden_size"] = 15.into()).contains("num_heads"));
        // An activation other than SiLU.
        assert!(refused(&|c| c["hidden_act"] = "gelu".into()).contains("hidden_act"));
    }
}

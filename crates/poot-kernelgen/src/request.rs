//! The one generation seam: [`generate`] turns a [`KernelRequest`] into a kernel `Body` and its launch
//! shape for a target.
//!
//! A request is plain data: the planner picks the family, schedule and widths from shapes and
//! [`DeviceCaps`] (policy lives in `poot-graph-plan`), and the generator implements that selection. It
//! does not choose a performance policy. [`generate`] is a pure function of `(request, backend, caps)`, so
//! the request is the kernel's identity: its `Debug` form is the canonical encoding the planner digests
//! into a plan key, and two requests that differ in any parameter (an `f32` is stored as its bits) are
//! different kernels. `Debug` is the canonical encoding only while no request type has a lossy hand-written
//! `Debug` (today only `TileSize`, which prints its `u32` edge and is injective), and an `f32` prints its
//! shortest round-trip text, which tells every value apart except NaN payloads.
//!
//! The launch grid is derived here from the request, in threads (`[x, y, z]`; a caller divides by the
//! body's workgroup size to get dispatch counts), so a body and the grid that shapes it come out of one
//! function and cannot disagree.

use poot_kernel_ir::{BinOp, Body, MathOp, Statement, Ty, UnOp, WmmaDtype, WmmaShape};
use poot_target::{Backend, BufferStorage, DeviceCaps};

use crate::KernelGenError;
use crate::contraction::{DenseLoad, Schedule, dense_contraction};
use crate::data_movement::{
    arg_top_k_dt, broadcast_dt, concat_n_dt, concat2_dt, dyn_update_slice_dt,
    dyn_update_slice_dynamic_dt, gather_axis_index_dt, gather_axis0_index_dt, rope_dt,
    scatter_axis0_dt, scatter_update_chunked_dt, scatter_update_dt, slice_dt, transpose_dt,
};
use crate::elementwise::{
    binary_broadcast_dt_views_grid, binary_broadcast_i32_geu_views_grid,
    binary_broadcast_i32_remu_views_grid, binary_scalar_dt_grid, binary_scalar_i32_geu_grid,
    binary_scalar_i32_grid, binary_scalar_i32_remu_grid, cast_bf16_to_f32, cast_f16_to_f32,
    cast_f32_to_bf16, cast_f32_to_f16, cast_i32_to_f32, erf_dt, math_unary_dt_grid,
    math_unary_dt_views_grid, recip_dt_grid, recip_dt_views_grid, tanh_dt, unary_dt_grid,
    unary_dt_views_grid, unary_i32_clz_grid, unary_i32_clz_views_grid,
};
use crate::error::BodyStage;
use crate::flash::{flash_attention_decode, flash_region_decode, flash_region_prefill};
use crate::fp8::{
    e4m3fn_broadcast_packed, e4m3fn_concat_packed, e4m3fn_dynamic_update_slice_dynamic_packed,
    e4m3fn_dynamic_update_slice_packed, e4m3fn_gather_packed, e4m3fn_packed_to_f32,
    e4m3fn_repack_reshape, e4m3fn_scatter_update_packed, e4m3fn_slice_packed,
    e4m3fn_transpose_packed, f32_to_e4m3fn_packed,
};
use crate::fused::{
    FusedInput, FusedKernel, RowKernel, fused_i32_views_grid, fused_i32_views_grid_pack,
    fused_row_parallel_views, fused_views, pointwise_size, row_size,
};
use crate::gemv::{attn_scores_v_gemv_lds, gemv_lds, indexed_gemv_lds};
use crate::helpers::{BodyBudget, BodyLimits, BodySize, Layout, WeightLayout};
use crate::matmul::{
    indexed_matmul_dt, matmul_batched_bias_dt_grid, matmul_batched_dt_grid, matmul_batched_nk_grid,
};
use crate::packed::{PackedKernelSpec, packed_kernel};
use crate::reduce::reduce_last_dt;
use crate::tiled_region::tiled_region;
use crate::wmma::{matmul_tensorcore, matmul_tensorcore_coopmat};

/// The entry-point name every generated body but the named ones below carries; the shared name the GPU
/// emitters look up.
const ENTRY: &str = "k";

/// The subgroup lane count every matrix-fragment kernel is written for: an NVIDIA warp, an RDNA3 wave32 and
/// the RADV cooperative-matrix subgroup each run one 16x16x16 fragment across 32 lanes.
const FRAGMENT_SUBGROUP_LANES: u32 = 32;

fn product(shape: &[usize]) -> u64 {
    shape
        .iter()
        .fold(1u64, |n, &dim| n.saturating_mul(dim as u64))
}

/// A launch: the total thread grid `[x, y, z]` and the optional read-only `u32` metadata buffer a body
/// takes between its data inputs and its output (param order `[data inputs.., meta, out]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub meta: Option<Vec<u32>>,
}

impl Launch {
    fn flat(threads: usize) -> Self {
        Self::grid([threads as u32, 1, 1])
    }

    fn grid(grid: [u32; 3]) -> Self {
        Self { grid, meta: None }
    }
}

/// What [`generate`] returns: the kernel body and the launch that shapes it.
#[derive(Clone, Debug)]
pub struct Generated {
    pub body: Body,
    pub launch: Launch,
}

/// The serial work a dispatch carries: `threads` invocations that each run `steps_per_thread` dependent
/// steps (a contraction's K loop). The product is the conservative single-dispatch cost the device's
/// `max_dispatch_work` bounds; `Dispatch::work` is a different quantity, the weight a run of dispatches
/// batches into submits by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialWork {
    pub threads: u64,
    pub steps_per_thread: u64,
}

impl SerialWork {
    pub fn total(self) -> u64 {
        self.threads.saturating_mul(self.steps_per_thread)
    }
}

/// What a kernel needs from the device that its `Body` does not state: declared by the generator (or by
/// the imported-kernel manifest) beside the body and launch. The resources the body does state (static
/// LDS, workgroup shape, matrix fragments) are measured from it by [`BodyUse::measure`], so a declaration
/// cannot understate them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelRequirements {
    /// The subgroup lane count a fragment op's collective is written for; `None` when the body uses no
    /// subgroup collective.
    pub subgroup_lanes: Option<u32>,
    /// The serial work of one dispatch; `None` when the kernel's per-thread loop is not modeled.
    pub serial_work: Option<SerialWork>,
}

/// One matrix-fragment shape and operand dtype a body uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FragmentUse {
    pub dtype: WmmaDtype,
    pub shape: WmmaShape,
}

/// The device resources a [`Body`] states, measured from the body itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyUse {
    /// Static workgroup-local memory, in bytes: the sum over declared arrays of element width times
    /// length. There is no alignment or padding model (every measurable element is 2, 4 or 8 bytes and
    /// arrays are laid out back to back), and the sum saturates, so an overflow reads as "too large" and
    /// is refused, never wrapped.
    pub lds_bytes: u64,
    pub workgroup: [u32; 3],
    /// Every distinct matrix fragment, in first-use order.
    pub fragments: Vec<FragmentUse>,
}

/// A workgroup-local array whose element type has no defined width.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("workgroup-local array {array} has element type {elem:?}, which has no static byte width")]
pub struct UnmeasurableLds {
    pub array: usize,
    pub elem: Ty,
}

impl BodyUse {
    /// Measure `body`; refuses an LDS array whose element type has no static width.
    pub fn measure(body: &Body) -> Result<Self, UnmeasurableLds> {
        let mut lds_bytes = 0u64;
        for (array, decl) in body.workgroup_locals.iter().enumerate() {
            let width = match decl.elem_ty {
                Ty::F16 | Ty::BF16 => 2,
                Ty::F32 | Ty::I32 | Ty::U32 => 4,
                Ty::F64 => 8,
                // No static byte width: a new `Ty` variant must choose a side here to compile.
                Ty::Unit
                | Ty::Bool
                | Ty::Usize
                | Ty::Ref { .. }
                | Ty::Slice(_)
                | Ty::Array { .. }
                | Ty::Vec { .. } => {
                    return Err(UnmeasurableLds {
                        array,
                        elem: decl.elem_ty.clone(),
                    });
                }
            };
            lds_bytes = lds_bytes.saturating_add(width * u64::from(decl.len));
        }
        let mut fragments: Vec<FragmentUse> = Vec::new();
        for statement in body.blocks.iter().flat_map(|block| &block.statements) {
            let used = match *statement {
                Statement::WmmaLoad { dtype, shape, .. }
                | Statement::WmmaLoadLds { dtype, shape, .. }
                | Statement::WmmaMma { dtype, shape, .. }
                | Statement::WmmaStore { dtype, shape, .. }
                | Statement::WmmaZero { dtype, shape, .. } => FragmentUse { dtype, shape },
                _ => continue,
            };
            if !fragments.contains(&used) {
                fragments.push(used);
            }
        }
        Ok(Self {
            lds_bytes,
            workgroup: body.workgroup_size,
            fragments,
        })
    }
}

/// The `(x, y)` workgroup grid that lays `total` workgroups out with X filling up to `max_x` and the
/// remainder spilling onto Y. The one fold every 2-D grid in the planner and in [`generate`] uses.
pub fn fold_groups(total: usize, max_x: u32) -> (usize, usize) {
    let x = total.clamp(1, max_x as usize);
    (x, total.div_ceil(x))
}

/// Whether a one-thread-per-element kernel folds past the target's X grid cap onto a 2-D grid of
/// `width`-wide workgroups. The planner decides `two_d` (a policy predicate over the equation and the
/// device); the body's `x_groups` and the launch grid both follow from it here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fold {
    pub numel: usize,
    pub two_d: bool,
    pub width: usize,
}

impl Fold {
    fn groups(self, caps: &DeviceCaps) -> (usize, usize) {
        fold_groups(self.numel.div_ceil(self.width).max(1), caps.max_grid[0])
    }

    fn x_groups(self, caps: &DeviceCaps) -> Option<usize> {
        self.two_d.then(|| self.groups(caps).0)
    }

    fn launch(self, caps: &DeviceCaps) -> Launch {
        if self.two_d {
            let (x, y) = self.groups(caps);
            Launch::grid([(x * self.width) as u32, y as u32, 1])
        } else {
            Launch::flat(self.numel)
        }
    }

    /// A one-thread-per-output contraction `body` launched on this fold with `width`-lane workgroups on the 1-D
    /// and the 2-D grid alike: the contraction owns its whole launch, so the planner never reshapes it (it is
    /// not reshaped). The planner folds onto 2-D before `numel / width` workgroups would pass the X grid cap.
    fn contraction(self, mut body: Body, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
        body.workgroup_size = [self.width as u32, 1, 1];
        with(body, self.launch(caps))
    }
}

/// A shape read through a physical [`Layout`] (a strided view, or the row-major default).
#[derive(Clone, Debug)]
pub struct ViewOperand {
    pub shape: Vec<usize>,
    pub layout: Layout,
}

/// One generation request. See the module docs.
#[derive(Clone, Debug)]
pub enum KernelRequest {
    Elementwise(ElementwiseSpec),
    Pointwise(PointwiseSpec),
    Row(RowSpec),
    Contraction(ContractionSpec),
    Attention(AttentionSpec),
    Movement(MovementSpec),
    Packed(PackedRequest),
    Rope(RopeSpec),
    TopK(TopKSpec),
}

impl KernelRequest {
    /// What this request's kernel needs from the device beyond its body (see [`KernelRequirements`]).
    pub fn requirements(&self) -> KernelRequirements {
        match self {
            Self::Contraction(
                ContractionSpec::TensorCore { .. } | ContractionSpec::Coopmat { .. },
            ) => KernelRequirements {
                subgroup_lanes: Some(FRAGMENT_SUBGROUP_LANES),
                serial_work: None,
            },
            Self::Contraction(ContractionSpec::Serial { shapes, .. }) => KernelRequirements {
                subgroup_lanes: None,
                serial_work: shapes.a_shape.last().map(|&k| SerialWork {
                    threads: product(&shapes.out_shape),
                    steps_per_thread: k as u64,
                }),
            },
            Self::Contraction(ContractionSpec::DenseSerial { shapes, .. }) => KernelRequirements {
                subgroup_lanes: None,
                serial_work: shapes.b_shape.last().map(|&k| SerialWork {
                    threads: product(&shapes.out_shape),
                    steps_per_thread: k as u64,
                }),
            },
            Self::Contraction(ContractionSpec::IndexedSerial { m, k, n, .. }) => {
                KernelRequirements {
                    subgroup_lanes: None,
                    serial_work: Some(SerialWork {
                        threads: (*m as u64).saturating_mul(*n as u64),
                        steps_per_thread: *k as u64,
                    }),
                }
            }
            _ => KernelRequirements::default(),
        }
    }

    /// A human-readable name for this kernel: family, form and, for a dense contraction, its storage, launch and
    /// `K x N` shape. Display only: the plan key is the request's digest, and two shapes that share
    /// one shape-generic body still get two labels, so per-shape device time survives the key.
    pub fn display_label(&self) -> String {
        match self {
            Self::Contraction(ContractionSpec::DenseGemv {
                weight,
                layout,
                schedule,
                k,
                n,
            }) => {
                let launch = match schedule {
                    Schedule::Gemv {
                        width,
                        cols,
                        unroll,
                    } => format!("w{width}c{cols}u{unroll}"),
                    Schedule::Tiled { tile } => format!("tiled{}", tile.get()),
                };
                format!("contraction:dense_gemv:{weight}:{layout:?}:{launch}:{k}x{n}")
            }
            Self::Contraction(spec) => format!("contraction:{}", spec.form()),
            other => other.family().to_string(),
        }
    }

    /// The family name, for diagnostics and the plan summary's `choice` column.
    pub fn family(&self) -> &'static str {
        match self {
            Self::Elementwise(_) => "elementwise",
            Self::Pointwise(_) => "pointwise",
            Self::Row(_) => "row",
            Self::Contraction(_) => "contraction",
            Self::Attention(_) => "attention",
            Self::Movement(_) => "movement",
            Self::Packed(_) => "packed",
            Self::Rope(_) => "rope",
            Self::TopK(_) => "topk",
        }
    }
}

/// A unary kernel's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Basic(UnOp),
    Recip,
    Math(MathOp),
    Tanh,
    Erf,
    /// Leading-zero count of an I32 word.
    ClzI32,
}

/// A binary kernel's operation over exact I32 storage with a literal operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I32Binary {
    GeU,
    RemU,
    Basic(BinOp),
}

/// A two-value binary kernel's operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueBinary {
    /// The element type the operation runs in (`Ty::I32` for an exact-I32 value).
    Basic(BinOp, Ty),
    GeU,
    RemU,
}

/// A dtype cast between two storage types.
#[derive(Clone, Debug)]
pub enum CastSpec {
    F32ToBf16 {
        numel: usize,
    },
    F32ToF16 {
        numel: usize,
    },
    F16ToF32 {
        numel: usize,
    },
    Bf16ToF32 {
        numel: usize,
    },
    I32ToF32 {
        numel: usize,
    },
    /// Pack f32 rows into E4M3FN words; one thread per packed output word.
    F32ToE4m3 {
        out_shape: Vec<usize>,
        row_len: usize,
    },
    /// Widen packed E4M3FN rows to f32; one thread per output element.
    E4m3ToF32 {
        row_len: usize,
        numel: usize,
    },
}

#[derive(Clone, Debug)]
pub enum ElementwiseSpec {
    /// One operand, optionally read through a view of `view.shape` (the output shape).
    Unary {
        op: UnaryOp,
        dt: Ty,
        view: Option<ViewOperand>,
        fold: Fold,
    },
    /// `value op literal` in float arithmetic; `lit_bits` is the literal's `f32` bits.
    ScalarFloat {
        op: BinOp,
        dt: Ty,
        lit_bits: u32,
        fold: Fold,
    },
    /// `value op literal` over exact I32 storage.
    ScalarI32 {
        op: I32Binary,
        lit: i32,
        fold: Fold,
    },
    /// `a op b` with broadcasting, each operand read through its layout.
    Binary {
        op: ValueBinary,
        out_shape: Vec<usize>,
        a: ViewOperand,
        b: ViewOperand,
        fold: Fold,
    },
    Cast(CastSpec),
}

/// Which element form a pointwise region computes in.
#[derive(Clone, Debug)]
pub enum PointwiseForm {
    /// Float arithmetic (`fused_views`), one thread per output element.
    Float,
    /// A single exact-I32 step on the elementwise 2-D fold (the `Select` equation).
    ExactI32 { fold: Fold },
    /// An exact-I32 region; `pack` (empty for none) lists the lanes a row-folded pack writes, one thread
    /// per packed word.
    ExactI32Packed { pack: Vec<FusedInput> },
}

#[derive(Clone, Debug)]
pub struct PointwiseSpec {
    pub out_shape: Vec<usize>,
    pub numel: usize,
    pub leaves: Vec<ViewOperand>,
    pub kernel: FusedKernel,
    pub form: PointwiseForm,
}

#[derive(Clone, Debug)]
pub enum RowSpec {
    /// A row-wise fused region: one workgroup of `width` lanes per row.
    Fused {
        out_shape: Vec<usize>,
        numel: usize,
        leaves: Vec<ViewOperand>,
        kernel: RowKernel,
        width: usize,
    },
    /// A last-axis reduction over `cols`, one thread per output element.
    ReduceLast {
        dt: Ty,
        op: BinOp,
        cols: usize,
        init_bits: u32,
        numel: usize,
    },
}

/// The dtype and shapes of a batched matmul.
#[derive(Clone, Debug)]
pub struct MatmulShapes {
    pub out_shape: Vec<usize>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum ContractionSpec {
    /// The one-thread-per-output reference kernel, with the bias epilogue when `bias`.
    Serial {
        dt: Ty,
        bias: bool,
        shapes: MatmulShapes,
        fold: Fold,
    },
    /// The one-thread-per-output kernel over a weight in checkpoint `[N, K]` order (`shapes.b_shape` is `[N, K]`):
    /// `DenseContraction` where the tiled GEMM is not taken. No bias. The activation and output are f32; `weight`
    /// is the weight's planned storage, `BufferStorage::f32()` or (Card 1007) `BufferStorage::f16_packed()`.
    DenseSerial {
        weight: BufferStorage,
        shapes: MatmulShapes,
        fold: Fold,
    },
    /// Matrix-core fragments (WMMA m16n16k16): one 256-thread block per 16x16 output tile per batch.
    TensorCore { dt: Ty, shapes: MatmulShapes },
    /// RADV cooperative matrix: one 32-invocation subgroup per 16x16 output tile.
    Coopmat { shapes: MatmulShapes },
    /// One range of a watchdog-split decode GEMV: `elems` workgroups of `width` lanes starting at output
    /// element `offset`.
    GemvChunk {
        k: usize,
        n: usize,
        width: usize,
        bias: bool,
        elems: usize,
        offset: usize,
    },
    /// The `DenseContraction` decode GEMV (`M == 1`): `schedule` (a [`Schedule::Gemv`]) over a weight read in
    /// `layout` from `weight` storage (`BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`), the shared Gemv
    /// loop nest with the dense operand load. The body is shape-generic: `K` and `N` ride in its `[K, N]`
    /// metadata; `k` also fixes whether a packed weight row starts on a word boundary. A Gemv over a `[K, N]`
    /// layout is refused.
    DenseGemv {
        weight: BufferStorage,
        layout: WeightLayout,
        schedule: Schedule,
        k: usize,
        n: usize,
    },
    /// Decode attention `scores @ V`: an LDS-parallel reduction over `cap`, `width` lanes per output.
    AttnScoresVGemv {
        cap: usize,
        d: usize,
        width: usize,
        numel: usize,
    },
    /// The synthesized tiled GEMM: `groups` workgroups of `tile * tile` lanes starting at tile `offset`.
    TiledRegion {
        m: usize,
        k: usize,
        n: usize,
        tile: usize,
        offset: usize,
        groups: usize,
    },
    /// [`ContractionSpec::TiledRegion`] reading its weight in checkpoint `[N, K]` order: the `DenseContraction`
    /// prefill GEMM. `weight` as in [`ContractionSpec::DenseSerial`].
    DenseTiled {
        m: usize,
        k: usize,
        n: usize,
        tile: usize,
        offset: usize,
        groups: usize,
        weight: BufferStorage,
    },
    /// The MoE per-expert GEMV: `width` lanes per output element.
    IndexedGemv {
        k: usize,
        n: usize,
        width: usize,
        numel: usize,
    },
    /// The MoE per-expert one-thread-per-output kernel.
    IndexedSerial {
        dt: Ty,
        m: usize,
        k: usize,
        n: usize,
        fold: Fold,
    },
}

impl ContractionSpec {
    /// The contraction form's name, for [`KernelRequest::display_label`].
    fn form(&self) -> &'static str {
        match self {
            Self::Serial { .. } => "serial",
            Self::DenseSerial { .. } => "dense_serial",
            Self::TensorCore { .. } => "tensor_core",
            Self::Coopmat { .. } => "coopmat",
            Self::GemvChunk { .. } => "gemv_chunk",
            Self::DenseGemv { .. } => "dense_gemv",
            Self::AttnScoresVGemv { .. } => "attn_scores_v_gemv",
            Self::TiledRegion { .. } => "tiled_region",
            Self::DenseTiled { .. } => "dense_tiled",
            Self::IndexedGemv { .. } => "indexed_gemv",
            Self::IndexedSerial { .. } => "indexed_serial",
        }
    }
}

#[derive(Clone, Debug)]
pub enum AttentionSpec {
    /// Single-sequence decode with a private output array (head dim above the LDS cap): NVPTX and AMDGCN
    /// only.
    DecodeSingle {
        bsz: usize,
        hq: usize,
        n_rep: usize,
        cap: usize,
        d: usize,
        scale_bits: u32,
        mask_per_head: bool,
    },
    /// LDS-cooperative decode: `bsz * hq` workgroups of `width` lanes.
    RegionDecode {
        bsz: usize,
        hq: usize,
        n_rep: usize,
        cap: usize,
        d: usize,
        scale_bits: u32,
        width: usize,
        mask_per_head: bool,
    },
    /// Prefill, one workgroup per (head, row) on a folded 2-D grid.
    RegionPrefill {
        hq: usize,
        l: usize,
        d: usize,
        n_rep: usize,
        scale_bits: u32,
        mask_per_head: bool,
    },
}

/// A packed-row E4M3FN data-movement kernel; one thread per physical output word.
#[derive(Clone, Debug)]
pub enum E4m3Movement {
    ScatterUpdate {
        base_shape: Vec<usize>,
        src_shape: Vec<usize>,
    },
    Transpose {
        in_shape: Vec<usize>,
        perm: Vec<usize>,
    },
    Slice {
        in_shape: Vec<usize>,
        axis: usize,
        start: usize,
    },
    Concat {
        axis: usize,
        in_shapes: Vec<Vec<usize>>,
    },
    Gather {
        in_shape: Vec<usize>,
        axis: usize,
        index_shape: Vec<usize>,
    },
    DynamicUpdateSlice {
        operand_shape: Vec<usize>,
        update_shape: Vec<usize>,
        axis: usize,
        index: usize,
    },
    DynamicUpdateSliceRuntime {
        operand_shape: Vec<usize>,
        update_shape: Vec<usize>,
        axis: usize,
    },
    Broadcast {
        in_shape: Vec<usize>,
    },
    RepackReshape {
        input_row_len: usize,
        output_row_len: usize,
    },
}

#[derive(Clone, Debug)]
pub enum MovementSpec {
    GatherAxis0 {
        dt: Ty,
        index_dt: Ty,
        rest: usize,
        numel: usize,
    },
    GatherAxis {
        dt: Ty,
        index_dt: Ty,
        inner: usize,
        axis_len: usize,
        idx_numel: usize,
        numel: usize,
    },
    ScatterAxis0 {
        dt: Ty,
        rest: usize,
        numel: usize,
    },
    ScatterUpdate {
        dt: Ty,
        rest: usize,
        numel: usize,
    },
    /// One range of a watchdog-split scatter-update: `groups` workgroups of `width` lanes starting at
    /// output element `offset`.
    ScatterUpdateChunk {
        dt: Ty,
        rest: usize,
        offset: usize,
        groups: usize,
        width: usize,
    },
    Transpose {
        dt: Ty,
        out_shape: Vec<usize>,
        in_shape: Vec<usize>,
        perm: Vec<usize>,
        numel: usize,
    },
    Slice {
        dt: Ty,
        out_shape: Vec<usize>,
        in_shape: Vec<usize>,
        axis: usize,
        start: usize,
        numel: usize,
    },
    Broadcast {
        dt: Ty,
        out_shape: Vec<usize>,
        in_shape: Vec<usize>,
        numel: usize,
    },
    Concat2 {
        dt: Ty,
        out_shape: Vec<usize>,
        axis: usize,
        a_shape: Vec<usize>,
        b_shape: Vec<usize>,
        numel: usize,
    },
    ConcatN {
        dt: Ty,
        out_shape: Vec<usize>,
        axis: usize,
        in_shapes: Vec<Vec<usize>>,
        numel: usize,
    },
    /// A dynamic-update-slice at a baked index.
    UpdateSliceStatic {
        dt: Ty,
        out_shape: Vec<usize>,
        axis: usize,
        index: usize,
        extent: usize,
        numel: usize,
    },
    /// A dynamic-update-slice whose index is read from a buffer. `index_dt` names that buffer's element
    /// type: the body reads either lane the same, but the two are different kernels' identities (an
    /// F32-index and an I32-index update must never share a plan key).
    UpdateSliceRuntime {
        dt: Ty,
        index_dt: Ty,
        out_shape: Vec<usize>,
        axis: usize,
        extent: usize,
        numel: usize,
    },
    /// A packed E4M3FN movement; `out_shape` shapes the grid (one thread per physical output word).
    E4m3 {
        out_shape: Vec<usize>,
        op: E4m3Movement,
    },
}

/// A generated packed-weight kernel. The planner validates and chooses the launch (the schedule's grid,
/// folded and checked against the device's limits with a typed refusal), so it rides in the request.
#[derive(Clone, Debug)]
pub struct PackedRequest {
    pub name: &'static str,
    pub spec: PackedKernelSpec,
    pub grid: [u32; 3],
    pub meta: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct RopeSpec {
    pub dt: Ty,
    pub x_shape: Vec<usize>,
    pub cos_shape: Vec<usize>,
    pub rot: usize,
    pub numel: usize,
}

#[derive(Clone, Debug)]
pub struct TopKSpec {
    pub e: usize,
    pub k: usize,
    pub numel: usize,
}

/// One thread per physical output word of a packed-row E4M3FN output (`rows * ceil(row_len / 4)`).
fn e4m3_row_launch(out_shape: &[usize]) -> Launch {
    let row_len = out_shape.last().copied().unwrap_or(1);
    let rows: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape[..out_shape.len() - 1].iter().product()
    };
    Launch::flat(rows.saturating_mul(row_len.div_ceil(4)))
}

fn slices(shapes: &[Vec<usize>]) -> Vec<&[usize]> {
    shapes.iter().map(Vec::as_slice).collect()
}

fn leaf_shapes(leaves: &[ViewOperand]) -> Vec<&[usize]> {
    leaves.iter().map(|leaf| leaf.shape.as_slice()).collect()
}

fn leaf_layouts(leaves: &[ViewOperand]) -> Vec<Layout> {
    leaves.iter().map(|leaf| leaf.layout.clone()).collect()
}

fn unsupported(request: &'static str, reason: &'static str) -> KernelGenError {
    KernelGenError::Unsupported { request, reason }
}

fn with(body: Body, launch: Launch) -> Result<Generated, KernelGenError> {
    Ok(Generated { body, launch })
}

/// Generate the kernel `req` names for `backend`.
///
/// A pure function of its arguments. A request the target cannot state returns
/// [`KernelGenError::Unsupported`]; a request that violates a generator's own precondition returns that
/// generator's error. Neither panics.
///
/// `limits` caps the body (instructions and locals). A request whose size follows from its recipe (a
/// fused pointwise or row region, where one graph can ask for any number of steps) is refused from its
/// checked conservative size before a generator builds anything; every other body is measured once built
/// and refused rather than returned. Either way the refusal is [`KernelGenError::BodyLimit`].
pub fn generate(
    req: &KernelRequest,
    backend: Backend,
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Generated, KernelGenError> {
    if let Some(size) = recipe_size(req) {
        BodyBudget::new(*limits).reserve(req.family(), BodyStage::Sizing, size)?;
    }
    let generated = build(req, backend, caps)?;
    BodyBudget::new(*limits).reserve(
        req.family(),
        BodyStage::Finished,
        BodySize::of(&generated.body),
    )?;
    Ok(generated)
}

/// The checked size bound of a request whose body grows with its recipe, before it is generated.
fn recipe_size(req: &KernelRequest) -> Option<BodySize> {
    match req {
        KernelRequest::Pointwise(spec) => {
            let pack_lanes = match &spec.form {
                PointwiseForm::ExactI32Packed { pack } => pack.len(),
                PointwiseForm::Float | PointwiseForm::ExactI32 { .. } => 0,
            };
            Some(pointwise_size(
                &spec.kernel,
                spec.out_shape.len(),
                pack_lanes,
            ))
        }
        KernelRequest::Row(RowSpec::Fused {
            out_shape,
            kernel,
            width,
            ..
        }) => Some(row_size(kernel, out_shape.len(), *width)),
        _ => None,
    }
}

fn build(
    req: &KernelRequest,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Generated, KernelGenError> {
    match req {
        KernelRequest::Elementwise(spec) => elementwise(spec, caps),
        KernelRequest::Pointwise(spec) => pointwise(spec, caps),
        KernelRequest::Row(spec) => row(spec, caps),
        KernelRequest::Contraction(spec) => contraction(spec, caps),
        KernelRequest::Attention(spec) => attention(spec, backend, caps),
        KernelRequest::Movement(spec) => movement(spec),
        KernelRequest::Packed(spec) => {
            let body = packed_kernel(spec.name, spec.spec)?;
            Ok(Generated {
                body,
                launch: Launch {
                    grid: spec.grid,
                    meta: Some(spec.meta.clone()),
                },
            })
        }
        KernelRequest::Rope(spec) => {
            let body = rope_dt(
                ENTRY,
                spec.dt.clone(),
                &spec.x_shape,
                &spec.cos_shape,
                spec.rot,
            )?;
            with(body, Launch::flat(spec.numel))
        }
        KernelRequest::TopK(spec) => with(
            arg_top_k_dt(ENTRY, spec.e, spec.k),
            Launch::flat(spec.numel),
        ),
    }
}

fn elementwise(spec: &ElementwiseSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    match spec {
        ElementwiseSpec::Unary { op, dt, view, fold } => {
            let xg = fold.x_groups(caps);
            let dt = dt.clone();
            let body = match (op, view) {
                (UnaryOp::Basic(op), None) => unary_dt_grid(ENTRY, dt, *op, xg),
                (UnaryOp::Basic(op), Some(v)) => {
                    unary_dt_views_grid(ENTRY, dt, *op, &v.shape, &v.layout, xg)
                }
                (UnaryOp::Recip, None) => recip_dt_grid(ENTRY, dt, xg),
                (UnaryOp::Recip, Some(v)) => {
                    recip_dt_views_grid(ENTRY, dt, &v.shape, &v.layout, xg)
                }
                (UnaryOp::Math(op), None) => math_unary_dt_grid(ENTRY, dt, *op, xg),
                (UnaryOp::Math(op), Some(v)) => {
                    math_unary_dt_views_grid(ENTRY, dt, *op, &v.shape, &v.layout, xg)
                }
                (UnaryOp::ClzI32, None) => unary_i32_clz_grid(ENTRY, xg),
                (UnaryOp::ClzI32, Some(v)) => {
                    unary_i32_clz_views_grid(ENTRY, &v.shape, &v.layout, xg)
                }
                (UnaryOp::Tanh, None) => tanh_dt(ENTRY, dt),
                (UnaryOp::Erf, None) => erf_dt(ENTRY, dt),
                (UnaryOp::Tanh | UnaryOp::Erf, Some(_)) => {
                    return Err(unsupported(
                        "unary",
                        "no view-capable body for this transcendental",
                    ));
                }
            };
            with(body, fold.launch(caps))
        }
        ElementwiseSpec::ScalarFloat {
            op,
            dt,
            lit_bits,
            fold,
        } => with(
            binary_scalar_dt_grid(
                ENTRY,
                *op,
                dt.clone(),
                f32::from_bits(*lit_bits),
                fold.x_groups(caps),
            ),
            fold.launch(caps),
        ),
        ElementwiseSpec::ScalarI32 { op, lit, fold } => {
            let xg = fold.x_groups(caps);
            let body = match op {
                I32Binary::GeU => binary_scalar_i32_geu_grid(ENTRY, *lit, xg),
                I32Binary::RemU => binary_scalar_i32_remu_grid(ENTRY, *lit, xg),
                I32Binary::Basic(op) => binary_scalar_i32_grid(ENTRY, *op, *lit, xg),
            };
            with(body, fold.launch(caps))
        }
        ElementwiseSpec::Binary {
            op,
            out_shape,
            a,
            b,
            fold,
        } => {
            let xg = fold.x_groups(caps);
            let body = match op {
                ValueBinary::GeU => binary_broadcast_i32_geu_views_grid(
                    ENTRY, out_shape, &a.shape, &a.layout, &b.shape, &b.layout, xg,
                ),
                ValueBinary::RemU => binary_broadcast_i32_remu_views_grid(
                    ENTRY, out_shape, &a.shape, &a.layout, &b.shape, &b.layout, xg,
                ),
                ValueBinary::Basic(op, dt) => binary_broadcast_dt_views_grid(
                    ENTRY,
                    *op,
                    dt.clone(),
                    out_shape,
                    &a.shape,
                    &a.layout,
                    &b.shape,
                    &b.layout,
                    xg,
                ),
            };
            with(body, fold.launch(caps))
        }
        ElementwiseSpec::Cast(cast) => match cast {
            CastSpec::F32ToBf16 { numel } => with(cast_f32_to_bf16(ENTRY), Launch::flat(*numel)),
            CastSpec::F32ToF16 { numel } => with(cast_f32_to_f16(ENTRY), Launch::flat(*numel)),
            CastSpec::F16ToF32 { numel } => with(cast_f16_to_f32(ENTRY), Launch::flat(*numel)),
            CastSpec::Bf16ToF32 { numel } => with(cast_bf16_to_f32(ENTRY), Launch::flat(*numel)),
            CastSpec::I32ToF32 { numel } => with(cast_i32_to_f32(ENTRY), Launch::flat(*numel)),
            CastSpec::F32ToE4m3 { out_shape, row_len } => with(
                f32_to_e4m3fn_packed(ENTRY, *row_len)?,
                e4m3_row_launch(out_shape),
            ),
            CastSpec::E4m3ToF32 { row_len, numel } => {
                with(e4m3fn_packed_to_f32(ENTRY, *row_len)?, Launch::flat(*numel))
            }
        },
    }
}

fn pointwise(spec: &PointwiseSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    let shapes = leaf_shapes(&spec.leaves);
    let layouts = leaf_layouts(&spec.leaves);
    match &spec.form {
        PointwiseForm::Float => with(
            fused_views(ENTRY, &spec.out_shape, &shapes, &layouts, &spec.kernel)?,
            Launch::flat(spec.numel),
        ),
        PointwiseForm::ExactI32 { fold } => with(
            fused_i32_views_grid(
                ENTRY,
                &spec.out_shape,
                &shapes,
                &layouts,
                &spec.kernel,
                fold.x_groups(caps),
            )?,
            fold.launch(caps),
        ),
        PointwiseForm::ExactI32Packed { pack } => {
            let body = fused_i32_views_grid_pack(
                ENTRY,
                &spec.out_shape,
                &shapes,
                &layouts,
                &spec.kernel,
                None,
                pack,
            )?;
            // A row-folded pack launches one thread per packed lane, not one per output element.
            let threads = if pack.is_empty() {
                spec.numel
            } else {
                (spec.numel / pack.len().max(1)).max(1)
            };
            with(body, Launch::flat(threads))
        }
    }
}

fn row(spec: &RowSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    match spec {
        RowSpec::Fused {
            out_shape,
            numel,
            leaves,
            kernel,
            width,
        } => {
            // One workgroup of `width` lanes per row, the rows folded onto a 2-D grid past the X cap.
            let rows = (numel / kernel.n_cols.max(1)).max(1);
            let (x, y) = fold_groups(rows, caps.max_grid[0]);
            let body = fused_row_parallel_views(
                ENTRY,
                out_shape,
                &leaf_shapes(leaves),
                &leaf_layouts(leaves),
                kernel,
                *width,
                x,
            )?;
            with(body, Launch::grid([(x * width) as u32, y as u32, 1]))
        }
        RowSpec::ReduceLast {
            dt,
            op,
            cols,
            init_bits,
            numel,
        } => with(
            reduce_last_dt(ENTRY, dt.clone(), *op, *cols, f32::from_bits(*init_bits)),
            Launch::flat(*numel),
        ),
    }
}

fn contraction(spec: &ContractionSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    match spec {
        ContractionSpec::Serial {
            dt,
            bias,
            shapes,
            fold,
        } => {
            let build = if *bias {
                matmul_batched_bias_dt_grid
            } else {
                matmul_batched_dt_grid
            };
            let body = build(
                ENTRY,
                dt.clone(),
                &shapes.out_shape,
                &shapes.a_shape,
                &shapes.b_shape,
                fold.x_groups(caps),
            );
            fold.contraction(body, caps)
        }
        ContractionSpec::DenseSerial {
            weight,
            shapes,
            fold,
        } => {
            let body = matmul_batched_nk_grid(
                ENTRY,
                *weight,
                &shapes.out_shape,
                &shapes.a_shape,
                &shapes.b_shape,
                fold.x_groups(caps),
            )?;
            fold.contraction(body, caps)
        }
        ContractionSpec::TensorCore { dt, shapes } => {
            let body = matmul_tensorcore(
                ENTRY,
                dt.clone(),
                &shapes.out_shape,
                &shapes.a_shape,
                &shapes.b_shape,
            )?;
            let r = shapes.out_shape.len();
            let (mm, nn) = (shapes.out_shape[r - 2], shapes.out_shape[r - 1]);
            let batch: usize = shapes.out_shape[..r - 2].iter().product();
            let tiles = (mm / 16) * (nn / 16);
            with(body, Launch::flat(batch * tiles * 256))
        }
        ContractionSpec::Coopmat { shapes } => {
            let body = matmul_tensorcore_coopmat(
                ENTRY,
                &shapes.out_shape,
                &shapes.a_shape,
                &shapes.b_shape,
            )?;
            let r = shapes.out_shape.len();
            let tiles = (shapes.out_shape[r - 2] / 16) * (shapes.out_shape[r - 1] / 16);
            with(body, Launch::flat(tiles * 32))
        }
        ContractionSpec::GemvChunk {
            k,
            n,
            width,
            bias,
            elems,
            offset,
        } => with(
            gemv_lds(
                ENTRY,
                *k,
                *n,
                *width,
                *bias,
                *elems,
                *offset,
                WeightLayout::Kn,
                BufferStorage::f32(),
            )?,
            Launch::flat(elems * width),
        ),
        ContractionSpec::DenseGemv {
            weight,
            layout,
            schedule,
            k,
            n,
        } => {
            let load = DenseLoad {
                weight: *weight,
                layout: *layout,
            };
            let body = dense_contraction(ENTRY, load, *schedule, *k)?;
            let threads = schedule.grid_threads(1, 1, *n);
            with(
                body,
                Launch {
                    grid: [threads as u32, 1, 1],
                    meta: Some(vec![*k as u32, *n as u32]),
                },
            )
        }
        ContractionSpec::AttnScoresVGemv {
            cap,
            d,
            width,
            numel,
        } => {
            let (x, y) = fold_groups(*numel, caps.max_grid[0]);
            with(
                attn_scores_v_gemv_lds(ENTRY, *cap, *d, *width, x),
                Launch::grid([(x * width) as u32, y as u32, 1]),
            )
        }
        ContractionSpec::TiledRegion {
            m,
            k,
            n,
            tile,
            offset,
            groups,
        } => with(
            tiled_region(
                "tiled_region",
                *m,
                *k,
                *n,
                *tile,
                *offset,
                WeightLayout::Kn,
                BufferStorage::f32(),
            )?,
            Launch::flat(groups * tile * tile),
        ),
        ContractionSpec::DenseTiled {
            m,
            k,
            n,
            tile,
            offset,
            groups,
            weight,
        } => with(
            tiled_region(
                "dense_tiled",
                *m,
                *k,
                *n,
                *tile,
                *offset,
                WeightLayout::Nk,
                *weight,
            )?,
            Launch::flat(groups * tile * tile),
        ),
        ContractionSpec::IndexedGemv { k, n, width, numel } => with(
            indexed_gemv_lds(ENTRY, *k, *n, *width),
            Launch::flat(numel * width),
        ),
        ContractionSpec::IndexedSerial { dt, m, k, n, fold } => {
            fold.contraction(indexed_matmul_dt(ENTRY, dt.clone(), *m, *k, *n), caps)
        }
    }
}

fn attention(
    spec: &AttentionSpec,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Generated, KernelGenError> {
    match spec {
        AttentionSpec::DecodeSingle {
            bsz,
            hq,
            n_rep,
            cap,
            d,
            scale_bits,
            mask_per_head,
        } => {
            // The body holds its running output in a private array: unimportable, and it crashes SPIR-V
            // codegen (the wgpu and raw-Vulkan runtimes both go through `Backend::SpirvVulkan`). NVPTX is
            // its intended target; AMDGCN lowers through a different path and is not known to fail.
            if backend == Backend::SpirvVulkan {
                return Err(unsupported(
                    "flash_attention_decode",
                    "a head dim above the LDS cap needs a private output array, which SPIR-V cannot state",
                ));
            }
            with(
                flash_attention_decode(
                    ENTRY,
                    *hq,
                    *n_rep,
                    *cap,
                    *d,
                    f32::from_bits(*scale_bits),
                    *mask_per_head,
                ),
                Launch::flat(bsz * hq),
            )
        }
        AttentionSpec::RegionDecode {
            bsz,
            hq,
            n_rep,
            cap,
            d,
            scale_bits,
            width,
            mask_per_head,
        } => with(
            flash_region_decode(
                ENTRY,
                *bsz,
                *hq,
                *n_rep,
                *cap,
                *d,
                f32::from_bits(*scale_bits),
                *width,
                *mask_per_head,
            ),
            // `bsz * hq` workgroups of `width` lanes.
            Launch::flat(bsz * hq * width),
        ),
        AttentionSpec::RegionPrefill {
            hq,
            l,
            d,
            n_rep,
            scale_bits,
            mask_per_head,
        } => {
            let (x, y) = fold_groups(hq * l, caps.max_grid[0]);
            with(
                flash_region_prefill(
                    ENTRY,
                    *hq,
                    *l,
                    *d,
                    *n_rep,
                    f32::from_bits(*scale_bits),
                    x,
                    *mask_per_head,
                ),
                Launch::grid([x as u32, y as u32, 1]),
            )
        }
    }
}

fn movement(spec: &MovementSpec) -> Result<Generated, KernelGenError> {
    match spec {
        MovementSpec::GatherAxis0 {
            dt,
            index_dt,
            rest,
            numel,
        } => with(
            gather_axis0_index_dt(ENTRY, dt.clone(), index_dt.clone(), *rest),
            Launch::flat(*numel),
        ),
        MovementSpec::GatherAxis {
            dt,
            index_dt,
            inner,
            axis_len,
            idx_numel,
            numel,
        } => with(
            gather_axis_index_dt(
                ENTRY,
                dt.clone(),
                index_dt.clone(),
                *inner,
                *axis_len,
                *idx_numel,
            ),
            Launch::flat(*numel),
        ),
        MovementSpec::ScatterAxis0 { dt, rest, numel } => with(
            scatter_axis0_dt(ENTRY, dt.clone(), *rest),
            Launch::flat(*numel),
        ),
        MovementSpec::ScatterUpdate { dt, rest, numel } => with(
            scatter_update_dt(ENTRY, dt.clone(), *rest),
            Launch::flat(*numel),
        ),
        MovementSpec::ScatterUpdateChunk {
            dt,
            rest,
            offset,
            groups,
            width,
        } => with(
            scatter_update_chunked_dt(ENTRY, dt.clone(), *rest, *offset),
            Launch::flat(groups * width),
        ),
        MovementSpec::Transpose {
            dt,
            out_shape,
            in_shape,
            perm,
            numel,
        } => with(
            transpose_dt(ENTRY, dt.clone(), out_shape, in_shape, perm),
            Launch::flat(*numel),
        ),
        MovementSpec::Slice {
            dt,
            out_shape,
            in_shape,
            axis,
            start,
            numel,
        } => with(
            slice_dt(ENTRY, dt.clone(), out_shape, in_shape, *axis, *start),
            Launch::flat(*numel),
        ),
        MovementSpec::Broadcast {
            dt,
            out_shape,
            in_shape,
            numel,
        } => with(
            broadcast_dt(ENTRY, dt.clone(), out_shape, in_shape),
            Launch::flat(*numel),
        ),
        MovementSpec::Concat2 {
            dt,
            out_shape,
            axis,
            a_shape,
            b_shape,
            numel,
        } => with(
            concat2_dt(ENTRY, dt.clone(), out_shape, *axis, a_shape, b_shape),
            Launch::flat(*numel),
        ),
        MovementSpec::ConcatN {
            dt,
            out_shape,
            axis,
            in_shapes,
            numel,
        } => with(
            concat_n_dt(ENTRY, dt.clone(), out_shape, *axis, &slices(in_shapes))?,
            Launch::flat(*numel),
        ),
        MovementSpec::UpdateSliceStatic {
            dt,
            out_shape,
            axis,
            index,
            extent,
            numel,
        } => with(
            dyn_update_slice_dt(ENTRY, dt.clone(), out_shape, *axis, *index, *extent),
            Launch::flat(*numel),
        ),
        MovementSpec::UpdateSliceRuntime {
            dt,
            index_dt: _,
            out_shape,
            axis,
            extent,
            numel,
        } => with(
            dyn_update_slice_dynamic_dt(ENTRY, dt.clone(), out_shape, *axis, *extent),
            Launch::flat(*numel),
        ),
        MovementSpec::E4m3 { out_shape, op } => {
            let body = match op {
                E4m3Movement::ScatterUpdate {
                    base_shape,
                    src_shape,
                } => e4m3fn_scatter_update_packed(ENTRY, base_shape, src_shape)?,
                E4m3Movement::Transpose { in_shape, perm } => {
                    e4m3fn_transpose_packed(ENTRY, out_shape, in_shape, perm)?
                }
                E4m3Movement::Slice {
                    in_shape,
                    axis,
                    start,
                } => e4m3fn_slice_packed(ENTRY, out_shape, in_shape, *axis, *start)?,
                E4m3Movement::Concat { axis, in_shapes } => {
                    e4m3fn_concat_packed(ENTRY, out_shape, *axis, &slices(in_shapes))?
                }
                E4m3Movement::Gather {
                    in_shape,
                    axis,
                    index_shape,
                } => e4m3fn_gather_packed(ENTRY, out_shape, in_shape, *axis, index_shape)?,
                E4m3Movement::DynamicUpdateSlice {
                    operand_shape,
                    update_shape,
                    axis,
                    index,
                } => e4m3fn_dynamic_update_slice_packed(
                    ENTRY,
                    operand_shape,
                    update_shape,
                    *axis,
                    *index,
                )?,
                E4m3Movement::DynamicUpdateSliceRuntime {
                    operand_shape,
                    update_shape,
                    axis,
                } => e4m3fn_dynamic_update_slice_dynamic_packed(
                    ENTRY,
                    operand_shape,
                    update_shape,
                    *axis,
                )?,
                E4m3Movement::Broadcast { in_shape } => {
                    e4m3fn_broadcast_packed(ENTRY, out_shape, in_shape)?
                }
                E4m3Movement::RepackReshape {
                    input_row_len,
                    output_row_len,
                } => e4m3fn_repack_reshape(ENTRY, *input_row_len, *output_row_len)?,
            };
            with(body, e4m3_row_launch(out_shape))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::error::BodyResource;

    /// Limits no fixture here comes near, so the rows below test what they name and not the cap.
    fn roomy() -> BodyLimits {
        BodyLimits {
            max_instructions: NonZeroUsize::new(1 << 20).unwrap(),
            max_locals: NonZeroUsize::new(1 << 16).unwrap(),
        }
    }

    /// Card 727: a request's display label is built from typed names, never from its `Debug`
    /// text. The dense Gemv names its storage, layout, launch and `K x N`; another contraction its form; any
    /// other request its family. Mutation: label every contraction but the dense Gemv `"contraction"`; the
    /// serial row goes red (`left: "contraction"`, `right: "contraction:serial"`).
    #[test]
    fn display_labels_name_the_family_form_and_dense_gemv_shape() {
        let fold = Fold {
            numel: 6,
            two_d: false,
            width: 256,
        };
        let cases = [
            (
                KernelRequest::Contraction(ContractionSpec::DenseGemv {
                    weight: BufferStorage::bf16_packed(),
                    layout: WeightLayout::Nk,
                    schedule: Schedule::Gemv {
                        width: 256,
                        cols: 4,
                        unroll: 4,
                    },
                    k: 896,
                    n: 4864,
                }),
                format!(
                    "contraction:dense_gemv:{}:Nk:w256c4u4:896x4864",
                    BufferStorage::bf16_packed()
                ),
            ),
            (
                KernelRequest::Contraction(ContractionSpec::Serial {
                    dt: Ty::F32,
                    bias: false,
                    shapes: MatmulShapes {
                        out_shape: vec![2, 3],
                        a_shape: vec![2, 4],
                        b_shape: vec![4, 3],
                    },
                    fold,
                }),
                "contraction:serial".to_string(),
            ),
            (
                KernelRequest::Contraction(ContractionSpec::AttnScoresVGemv {
                    cap: 8,
                    d: 4,
                    width: 128,
                    numel: 4,
                }),
                "contraction:attn_scores_v_gemv".to_string(),
            ),
            (
                KernelRequest::Elementwise(ElementwiseSpec::Unary {
                    op: UnaryOp::Basic(UnOp::Neg),
                    dt: Ty::F32,
                    view: None,
                    fold,
                }),
                "elementwise".to_string(),
            ),
            (
                KernelRequest::TopK(TopKSpec {
                    e: 8,
                    k: 2,
                    numel: 2,
                }),
                "topk".to_string(),
            ),
        ];
        for (request, want) in cases {
            assert_eq!(request.display_label(), want, "{request:?}");
        }
    }

    /// Card 636 SC-002 (dkernel B1): the decode `scores @ V` GEMV launches one workgroup of `width` lanes
    /// per OUTPUT ELEMENT, and the output holds `batch * n` elements, so the request carries `numel =
    /// n * batch` and the grid is derived from it, not from `n` alone. Mutation: derive the grid from
    /// `d` (the per-row width) instead of `numel`; a batch of 3 launches a third of the workgroups and
    /// this row goes red.
    #[test]
    fn the_attn_gemv_grid_covers_every_output_row_of_the_batch() {
        let (d, batch, width) = (32usize, 3usize, 128usize);
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        let generated = generate(
            &KernelRequest::Contraction(ContractionSpec::AttnScoresVGemv {
                cap: 64,
                d,
                width,
                numel: d * batch,
            }),
            Backend::SpirvVulkan,
            &caps,
            &roomy(),
        )
        .expect("the attention GEMV is stateable on SPIR-V");
        assert_eq!(
            generated.launch.grid,
            [(d * batch * width) as u32, 1, 1],
            "one {width}-lane workgroup per output element of all {batch} rows, not per column"
        );
        assert_eq!(generated.launch.meta, None);
    }

    /// Card 636 SC-004: a request the target cannot state returns `Unsupported`, never a panic. The
    /// single-sequence flash decode holds its running output in a private array, which SPIR-V cannot
    /// state; NVPTX and AMDGCN can. Mutation: make the unsupported arm `panic!` instead of returning the
    /// error; this row panics.
    #[test]
    fn a_private_array_flash_decode_is_unsupported_on_spirv_only() {
        let request = KernelRequest::Attention(AttentionSpec::DecodeSingle {
            bsz: 1,
            hq: 2,
            n_rep: 1,
            cap: 8,
            d: 512,
            scale_bits: 0.5f32.to_bits(),
            mask_per_head: false,
        });
        let spirv = DeviceCaps::wgpu_rdna3_igpu();
        match generate(&request, Backend::SpirvVulkan, &spirv, &roomy()) {
            Err(KernelGenError::Unsupported { request, reason }) => {
                assert_eq!(request, "flash_attention_decode");
                assert!(reason.contains("SPIR-V"), "{reason}");
            }
            other => panic!("expected Unsupported on SPIR-V, got {other:?}"),
        }
        let nvptx = DeviceCaps::ptx_default();
        let generated = generate(&request, Backend::Nvptx, &nvptx, &roomy())
            .expect("the same request is stateable on NVPTX");
        assert_eq!(generated.launch.grid, [2, 1, 1], "bsz * hq workgroups");
    }

    /// Finite binary16 bit patterns (zero, subnormal and normal, both signs; the packed decode's admitted
    /// domain excludes exponent `0x1f`) from a fixed xorshift stream, kept small enough that a K-term dot stays
    /// far from f32 overflow.
    fn finite_f16_bits(count: usize, seed: u64) -> Vec<u16> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let bits = (state >> 32) as u16;
                let sign = bits & 0x8000;
                let exponent = ((bits >> 10) & 0x1f) % 18;
                sign | (exponent << 10) | (bits & 0x03ff)
            })
            .collect()
    }

    /// Two binary16 elements per little-endian `u32` word, the last word zero-padded: the
    /// `BufferStorage::f16_packed` lane exactly as the checkpoint's bytes fill it.
    fn pack_f16_words(bits: &[u16]) -> Vec<u32> {
        bits.chunks(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(pair.get(1).copied().unwrap_or(0)) << 16))
            .collect()
    }

    /// Run `request` in the KIR interpreter over `[activation, weight, out]` and return `out`.
    fn interpret(
        request: &KernelRequest,
        weight: poot_kernel_ir::interp::Buffer,
        activation: &[f32],
        out_len: usize,
    ) -> Vec<f32> {
        use poot_kernel_ir::interp::{Buffer, run};
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        let generated =
            generate(request, Backend::SpirvVulkan, &caps, &roomy()).expect("stateable");
        let workgroup = generated.body.workgroup_size;
        let workgroups =
            std::array::from_fn(|axis| generated.launch.grid[axis].div_ceil(workgroup[axis]));
        let mut buffers = vec![Buffer::from_f32s(activation), weight];
        buffers.extend(generated.launch.meta.as_deref().map(Buffer::from_u32s));
        buffers.push(Buffer::from_f32s(&vec![0.0; out_len]));
        run(&generated.body, workgroups, &mut buffers).expect("interpreter");
        buffers.last().unwrap().to_f32s().unwrap()
    }

    /// Card 1007: every `DenseContraction` body (the M == 1 GEMV, the tiled GEMM at M = 4 and 33, with and
    /// without the K-unrolled slab path, and the one-thread-per-output kernel) reads a packed-F16 `[N, K]` weight
    /// to the bit what it reads from an f32 buffer holding the same weight's decoded values, and that f32 body
    /// matches a host dot product. The weights are odd-sized, so the last packed word carries one element and a
    /// zero pad. Mutation: in `WeightLane::read` take the half at `(1 - idx % 2) * 16` (the other element of each
    /// word); the row goes red at its first shape, the GEMV.
    #[test]
    fn dense_contraction_bodies_read_a_packed_f16_weight_as_its_decoded_values() {
        use poot_kernel_ir::interp::Buffer;
        let f16 = BufferStorage::f16_packed();
        let f32_storage = BufferStorage::f32();
        // (m, k, n): GEMV; tiled (K not a slab multiple, then a multiple of the 4 x 8 slab); serial.
        for (m, k, n) in [
            (1usize, 37usize, 5usize),
            (4, 37, 5),
            (33, 64, 7),
            (4, 19, 3),
        ] {
            let bits = finite_f16_bits(n * k, (m * 1000 + k * 10 + n) as u64);
            let decoded: Vec<f32> = bits
                .iter()
                .map(|&b| poot_quant::scalar::f16_to_f32(b))
                .collect();
            let activation: Vec<f32> = finite_f16_bits(m * k, 7 + m as u64)
                .iter()
                .map(|&b| poot_quant::scalar::f16_to_f32(b))
                .collect();
            let request = |weight: BufferStorage| {
                let tile = 8;
                KernelRequest::Contraction(match (m, k) {
                    (1, _) => ContractionSpec::DenseGemv {
                        weight,
                        layout: WeightLayout::Nk,
                        schedule: Schedule::Gemv {
                            width: 16,
                            cols: 2,
                            unroll: 2,
                        },
                        k,
                        n,
                    },
                    (4, 19) => ContractionSpec::DenseSerial {
                        weight,
                        shapes: MatmulShapes {
                            out_shape: vec![m, n],
                            a_shape: vec![m, k],
                            b_shape: vec![n, k],
                        },
                        fold: Fold {
                            numel: m * n,
                            two_d: false,
                            width: 64,
                        },
                    },
                    _ => ContractionSpec::DenseTiled {
                        m,
                        k,
                        n,
                        tile,
                        offset: 0,
                        groups: m.div_ceil(2 * tile) * n.div_ceil(tile),
                        weight,
                    },
                })
            };
            let packed = interpret(
                &request(f16),
                Buffer::from_u32s(&pack_f16_words(&bits)),
                &activation,
                m * n,
            );
            let wide = interpret(
                &request(f32_storage),
                Buffer::from_f32s(&decoded),
                &activation,
                m * n,
            );
            let packed_bits: Vec<u32> = packed.iter().map(|v| v.to_bits()).collect();
            let wide_bits: Vec<u32> = wide.iter().map(|v| v.to_bits()).collect();
            assert_eq!(
                packed_bits, wide_bits,
                "m={m} k={k} n={n}: packed F16 differs from its decoded f32 weight"
            );
            for row in 0..m {
                for col in 0..n {
                    let terms = (0..k).map(|i| {
                        f64::from(activation[row * k + i]) * f64::from(decoded[col * k + i])
                    });
                    let want: f64 = terms.clone().sum();
                    let scale: f64 = terms.map(f64::abs).sum::<f64>().max(1e-30);
                    let got = f64::from(wide[row * n + col]);
                    assert!(
                        (got - want).abs() <= 1e-5 * scale,
                        "m={m} k={k} n={n} out[{row}, {col}] = {got}, host dot {want}"
                    );
                }
            }
        }
    }

    /// Card 1007: a dense-contraction request names the weight's storage, and a storage no generated body reads
    /// (native two-byte F16, exact-I32 words) is a typed `Unsupported`, never a body that reads the buffer at the
    /// wrong width. Mutation: map every non-f32 storage to the packed-F16 lane in `WeightLane::of`; this row goes
    /// red.
    #[test]
    fn a_dense_contraction_over_an_unread_weight_storage_is_unsupported() {
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        for weight in [BufferStorage::f16(), BufferStorage::i32()] {
            let request = KernelRequest::Contraction(ContractionSpec::DenseGemv {
                weight,
                layout: WeightLayout::Nk,
                schedule: Schedule::Gemv {
                    width: 16,
                    cols: 2,
                    unroll: 2,
                },
                k: 8,
                n: 4,
            });
            match generate(&request, Backend::SpirvVulkan, &caps, &roomy()) {
                Err(KernelGenError::Unsupported { request, .. }) => {
                    assert_eq!(request, "dense_contraction")
                }
                Err(other) => panic!("{weight}: expected Unsupported, got {other}"),
                Ok(_) => panic!("{weight}: expected Unsupported, got a generated body"),
            }
        }
    }

    /// Card 727 SC-004: the one-workgroup-per-column Gemv over a `[K, N]` dense weight is
    /// refused when it is generated, for every weight storage, so no plan can carry the stride-`N` body. The
    /// same request over the `[N, K]` weight generates. Mutation: admit the `[K, N]` arm in
    /// `contraction::dense_contraction` (route it to the `[N, K]` body); a body comes back and this row goes red.
    #[test]
    fn a_gemv_over_a_k_by_n_dense_weight_is_refused() {
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        let request = |weight, layout| {
            KernelRequest::Contraction(ContractionSpec::DenseGemv {
                weight,
                layout,
                schedule: Schedule::Gemv {
                    width: 256,
                    cols: 16,
                    unroll: 4,
                },
                k: 896,
                n: 4864,
            })
        };
        for weight in [
            BufferStorage::f32(),
            BufferStorage::f16_packed(),
            BufferStorage::bf16_packed(),
        ] {
            match generate(
                &request(weight, WeightLayout::Kn),
                Backend::SpirvVulkan,
                &caps,
                &roomy(),
            ) {
                Err(KernelGenError::Unsupported { request, reason }) => {
                    assert_eq!(request, "dense_contraction", "{weight}");
                    assert!(
                        reason.contains("stride of N elements"),
                        "{weight}: {reason}"
                    );
                }
                Err(other) => panic!("{weight}: expected the Gemv refusal, got {other}"),
                Ok(generated) => panic!(
                    "{weight}: a [K, N] Gemv generated `{}` instead of being refused",
                    generated.body.name
                ),
            }
            generate(
                &request(weight, WeightLayout::Nk),
                Backend::SpirvVulkan,
                &caps,
                &roomy(),
            )
            .unwrap_or_else(|e| panic!("{weight}: the [N, K] Gemv must generate: {e}"));
        }
    }

    /// `n * k` weight values in `[-1, 1)` with a nonzero contribution in every position, from a seeded xorshift.
    fn weight_values(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let unit = ((s >> 40) as f32) / (1u64 << 24) as f32;
                let v = 2.0 * unit - 1.0;
                if v == 0.0 { 0.5 } else { v }
            })
            .collect()
    }

    /// Card 727 SC-002 (interpreter rehearsal of the device rows): the dense Gemv over an `[N, K]` weight in each
    /// storage lane (f32, packed F16, packed BF16), at the admitted launches (`cols` 4 and 2 at width 256 and
    /// unroll 4), a wide `cols` 64, and a narrow launch whose `unroll` does not divide `K`, equals an f64 host
    /// dot product within ADR-0101 tier 2 on every output. The shapes leave a `K` tail past the last full run, an
    /// `N` that is not a multiple of `cols`, and an odd `K` (packed rows that start mid-word). Mutation: in
    /// `gemv_body` skip the tail runs (the `k0 < K` block) or start every lane one run late; every shape with a
    /// `K` tail goes red.
    #[test]
    fn the_dense_gemv_matches_a_host_dot_product_in_every_lane_and_launch() {
        use poot_kernel_ir::interp::Buffer;
        let launches = [(256u32, 4u32, 4u32), (256, 2, 4), (256, 64, 4), (16, 2, 3)];
        // (k, n): K tail past full runs, N not a multiple of any cols; an odd K; a K below one run.
        let shapes = [(1037usize, 70usize), (515, 33), (3, 5)];
        for (width, cols, unroll) in launches {
            for (k, n) in shapes {
                let f16_bits = finite_f16_bits(n * k, (k * 31 + n) as u64);
                let values: Vec<f32> = f16_bits
                    .iter()
                    .map(|&b| poot_quant::scalar::f16_to_f32(b))
                    .collect();
                let activation = weight_values(k, 17 + k as u64);
                for weight in [
                    BufferStorage::f32(),
                    BufferStorage::f16_packed(),
                    BufferStorage::bf16_packed(),
                ] {
                    // The stored weight and the values it holds (BF16 truncates each value's low bits).
                    let (buffer, held): (Buffer, Vec<f32>) = if weight == BufferStorage::f32() {
                        (Buffer::from_f32s(&values), values.clone())
                    } else if weight == BufferStorage::f16_packed() {
                        (
                            Buffer::from_u32s(&pack_f16_words(&f16_bits)),
                            values.clone(),
                        )
                    } else {
                        let bits: Vec<u16> =
                            values.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                        let held = bits
                            .iter()
                            .map(|&b| f32::from_bits(u32::from(b) << 16))
                            .collect();
                        (Buffer::from_u32s(&pack_f16_words(&bits)), held)
                    };
                    let request = KernelRequest::Contraction(ContractionSpec::DenseGemv {
                        weight,
                        layout: WeightLayout::Nk,
                        schedule: Schedule::Gemv {
                            width,
                            cols,
                            unroll,
                        },
                        k,
                        n,
                    });
                    let got = interpret(&request, buffer, &activation, n);
                    for (col, &value) in got.iter().enumerate() {
                        let terms =
                            (0..k).map(|i| f64::from(activation[i]) * f64::from(held[col * k + i]));
                        let want: f64 = terms.clone().sum();
                        let scale: f64 = terms.map(f64::abs).sum::<f64>().max(1e-30);
                        assert!(
                            (f64::from(value) - want).abs() <= 1e-5 * scale,
                            "{weight} Gemv {{ {width}, {cols}, {unroll} }} k={k} n={n}: \
                             out[{col}] = {value}, host dot {want}"
                        );
                    }
                }
            }
        }
    }

    fn limits(instructions: usize, locals: usize) -> BodyLimits {
        BodyLimits {
            max_instructions: NonZeroUsize::new(instructions).unwrap(),
            max_locals: NonZeroUsize::new(locals).unwrap(),
        }
    }

    /// Card 666 SC-002, the emission boundary: a budget of 8 instructions and 4 locals takes exactly 8
    /// and 4, refuses the 9th instruction and the 5th local, and a refusal records nothing, so the
    /// statements an emitter pushes after each reservation stop at the limit. Mutation: compare with `>=`
    /// instead of `>` in `BodyBudget::reserve`; the exact-fit reservation is refused and this row goes
    /// red. Compare with `> limit + 1`; the 9th is taken and the pushed count reaches 9.
    #[test]
    fn a_body_budget_takes_exactly_its_limits_and_refuses_the_next_reservation() {
        let mut budget = BodyBudget::new(limits(8, 4));
        let mut emitted = Vec::new();
        let refusal = (0..9)
            .find_map(|i| {
                let size = BodySize {
                    instructions: 1,
                    locals: 0,
                };
                match budget.reserve("fixture", BodyStage::Sizing, size) {
                    Ok(()) => {
                        emitted.push(i);
                        None
                    }
                    Err(error) => Some(error),
                }
            })
            .expect("the 9th instruction is refused");
        assert_eq!(emitted.len(), 8, "exactly the limit is emitted");
        assert_eq!(
            refusal,
            KernelGenError::BodyLimit {
                generator: "fixture",
                resource: BodyResource::Instructions,
                stage: BodyStage::Sizing,
                attempted: 9,
                limit: 8,
            }
        );
        budget
            .reserve(
                "fixture",
                BodyStage::Sizing,
                BodySize {
                    instructions: 0,
                    locals: 4,
                },
            )
            .expect("4 locals fit exactly");
        let fifth = budget.reserve(
            "fixture",
            BodyStage::Sizing,
            BodySize {
                instructions: 0,
                locals: 1,
            },
        );
        assert_eq!(
            fifth,
            Err(KernelGenError::BodyLimit {
                generator: "fixture",
                resource: BodyResource::Locals,
                stage: BodyStage::Sizing,
                attempted: 5,
                limit: 4,
            })
        );
        assert_eq!(
            budget.reserved(),
            BodySize {
                instructions: 8,
                locals: 4
            },
            "refusals leave the reserved total where it was"
        );
    }

    /// An f32 fused region of `steps` erf steps over one `[4]` leaf, as the planner would request it.
    fn erf_chain(steps: usize) -> KernelRequest {
        let shape = vec![4usize];
        KernelRequest::Pointwise(PointwiseSpec {
            out_shape: shape.clone(),
            numel: 4,
            leaves: vec![ViewOperand {
                shape: shape.clone(),
                layout: Layout::contiguous(&shape),
            }],
            kernel: FusedKernel {
                n_leaves: 1,
                steps: (0..steps)
                    .map(|s| crate::fused::FusedStep {
                        op: crate::fused::FusedScalarOp::Erf,
                        inputs: vec![if s == 0 {
                            FusedInput::Leaf(0)
                        } else {
                            FusedInput::Step(s - 1)
                        }],
                    })
                    .collect(),
                output: FusedInput::Step(steps - 1),
            },
            form: PointwiseForm::Float,
        })
    }

    /// Card 666 SC-002, production seam: a real fused region through [`generate`] is refused from its
    /// checked size, at the `Sizing` stage, when the cap sits one under the bound, and generates when the
    /// cap equals it. The `Sizing` stage is the proof nothing was built: a generator that ran first
    /// would be caught by the `Finished` measurement instead, with the built body's own count. Mutation:
    /// remove the `recipe_size` reservation in `generate`; the refusal arrives at `Finished` with the
    /// built body's count and the stage assertion fails.
    #[test]
    fn a_fused_region_over_the_body_limits_is_refused_before_generation() {
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        let request = erf_chain(6);
        let bound = recipe_size(&request).expect("a pointwise request is sized from its recipe");
        let fits = generate(
            &request,
            Backend::SpirvVulkan,
            &caps,
            &limits(bound.instructions, bound.locals),
        )
        .expect("a cap equal to the bound admits the region");
        assert!(
            BodySize::of(&fits.body).instructions <= bound.instructions,
            "the bound covers the built body"
        );
        let refused = generate(
            &request,
            Backend::SpirvVulkan,
            &caps,
            &limits(bound.instructions - 1, bound.locals),
        )
        .unwrap_err();
        assert_eq!(
            refused,
            KernelGenError::BodyLimit {
                generator: "pointwise",
                resource: BodyResource::Instructions,
                stage: BodyStage::Sizing,
                attempted: bound.instructions,
                limit: bound.instructions - 1,
            }
        );
        let refused = generate(
            &request,
            Backend::SpirvVulkan,
            &caps,
            &limits(bound.instructions, bound.locals - 1),
        )
        .unwrap_err();
        assert!(
            matches!(
                refused,
                KernelGenError::BodyLimit {
                    resource: BodyResource::Locals,
                    stage: BodyStage::Sizing,
                    ..
                }
            ),
            "the locals cap refuses before generation too: {refused}"
        );
    }

    /// A request whose size does not follow from a recipe is measured once built and refused rather than
    /// returned. Mutation: drop the `Finished` reservation; the over-limit body comes back `Ok`.
    #[test]
    fn a_body_over_the_limits_is_refused_after_generation_when_its_request_cannot_be_sized() {
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        let request = KernelRequest::TopK(TopKSpec {
            e: 8,
            k: 2,
            numel: 4,
        });
        assert!(recipe_size(&request).is_none());
        let built = generate(&request, Backend::SpirvVulkan, &caps, &roomy())
            .expect("generates under roomy limits");
        let size = BodySize::of(&built.body);
        let refused = generate(
            &request,
            Backend::SpirvVulkan,
            &caps,
            &limits(size.instructions - 1, size.locals),
        )
        .unwrap_err();
        assert_eq!(
            refused,
            KernelGenError::BodyLimit {
                generator: "topk",
                resource: BodyResource::Instructions,
                stage: BodyStage::Finished,
                attempted: size.instructions,
                limit: size.instructions - 1,
            }
        );
    }
}

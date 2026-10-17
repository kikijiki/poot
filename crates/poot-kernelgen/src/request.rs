//! The one generation seam: [`generate`] turns a [`KernelRequest`] into a kernel `Body` and its launch
//! shape for a target.
//!
//! A request is plain data: the planner picks the family, schedule and widths from shapes and
//! [`DeviceCaps`] (policy lives in `poot-graph-plan`), and the generator implements that selection. It
//! does not choose a performance policy. [`generate`] is a pure function of `(request, caps)`, so
//! the request is the kernel's identity: its `Debug` form is the canonical encoding the planner digests
//! into a plan key, and two requests that differ in any parameter (an `f32` is stored as its bits) are
//! different kernels. `Debug` is the canonical encoding only while no request type has a lossy hand-written
//! `Debug` (today only `TileSize`, which prints its `u32` edge and is injective), and an `f32` prints its
//! shortest round-trip text, which tells every value apart except NaN payloads.
//!
//! The launch grid is derived here from the request, in threads (`[x, y, z]`; a caller divides by the
//! body's workgroup size to get dispatch counts), so a body and the grid that shapes it come out of one
//! function and cannot disagree.

use poot_kernel_ir::serialize::{Grid, serialize_grid};
use poot_kernel_ir::{
    BinOp, Body, FRAGMENT_COLLECTIVE_LANES, MathOp, Statement, Ty, UnOp, WmmaDtype, WmmaShape,
};
use poot_target::{BufferStorage, DeviceCaps};

use crate::KernelGenError;
use crate::attention::{
    AttentionCombineSpec, AttentionSpec, attention_body, attention_combine_body,
};
use crate::block_top_k::{BlockTopKSpec, block_top_k_body};
use crate::contraction::{Epilogue, Schedule, dense_contraction, indexed_contraction};
use crate::data_movement::{MovementSpec, Storage, arg_top_k_dt, movement_body, rope_dt};
use crate::elementwise::{
    binary_broadcast_dt_views_grid, binary_broadcast_i32_geu_views_grid,
    binary_broadcast_i32_remu_views_grid, binary_scalar_dt_grid, binary_scalar_i32_geu_grid,
    binary_scalar_i32_grid, binary_scalar_i32_remu_grid, cast_bf16_to_f32, cast_f16_to_f32,
    cast_f32_to_bf16, cast_f32_to_f16, cast_i32_to_f32, erf_dt, math_unary_dt_grid,
    math_unary_dt_views_grid, recip_dt_grid, recip_dt_views_grid, tanh_dt, unary_dt_grid,
    unary_dt_views_grid, unary_i32_clz_grid, unary_i32_clz_views_grid,
};
use crate::error::BodyStage;
use crate::fp8::{e4m3fn_packed_to_f32, f32_to_e4m3fn_packed};
use crate::fused::{
    FusedInput, FusedKernel, RowKernel, fused_i32_views_grid, fused_i32_views_grid_pack,
    fused_row_parallel_views, fused_views, pointwise_size, row_size,
};
use crate::gemv::{attn_scores_v_gemv_lds, gemv_lds};
use crate::helpers::{BodyBudget, BodyLimits, BodySize, Layout, WeightLayout};
use crate::matmul::{
    matmul_batched_dt_grid, matmul_batched_epilogue_dt_grid, matmul_batched_nk_grid,
};
use crate::packed::{PackedKernelSpec, packed_contraction, packed_kernel};
use crate::reduce::reduce_last_dt;
use crate::scan::{ScanSpec, scan_body};
use crate::tiled_region::tiled_region;
use crate::wmma::{matmul_tensorcore, matmul_tensorcore_coopmat};
use poot_quant::format::WeightFormat;

/// The entry-point name every generated body but the named ones below carries; the shared name the GPU
/// emitters look up.
const ENTRY: &str = "k";

fn product(shape: &[usize]) -> u64 {
    shape
        .iter()
        .fold(1u64, |n, &dim| n.saturating_mul(dim as u64))
}

/// A launch: the total thread grid `[x, y, z]` and the optional read-only `u32` metadata buffer a body
/// takes between its data inputs and its output (param order `[data inputs.., meta, out]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Total threads `[x, y, z]`; a caller divides by the body's workgroup size to get dispatch counts.
    pub grid: [u32; 3],
    /// Contents of the read-only `u32` metadata buffer, or `None` when the body takes none.
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
    /// The kernel body.
    pub body: Body,
    /// The launch that shapes it.
    pub launch: Launch,
}

/// The serial work a dispatch carries: `threads` invocations that each run `steps_per_thread` dependent
/// steps (a contraction's K loop). The product is the conservative single-dispatch cost the device's
/// `max_dispatch_work` bounds; `Dispatch::work` is a different quantity, the weight a run of dispatches
/// batches into submits by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SerialWork {
    /// Invocations launched.
    pub threads: u64,
    /// Dependent steps each invocation runs (a contraction's `K` loop).
    pub steps_per_thread: u64,
}

impl SerialWork {
    /// `threads * steps_per_thread`, saturating: the conservative cost of one dispatch.
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
    /// The operand dtype of the fragment.
    pub dtype: WmmaDtype,
    /// The fragment's matrix shape.
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
    /// The workgroup size `[x, y, z]` the body declares.
    pub workgroup: [u32; 3],
    /// Every distinct matrix fragment, in first-use order.
    pub fragments: Vec<FragmentUse>,
}

/// A workgroup-local array whose element type has no defined width.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("workgroup-local array {array} has element type {elem:?}, which has no static byte width")]
pub struct UnmeasurableLds {
    /// Index of the workgroup-local array in the body's declarations.
    pub array: usize,
    /// The array's element type.
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
                | Ty::Vec { .. }
                | Ty::Fragment(_) => {
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
    /// Output elements the kernel computes, one thread each.
    pub numel: usize,
    /// Whether the kernel folds onto a 2-D grid; decided by the planner from the device's grid cap.
    pub two_d: bool,
    /// Threads per workgroup.
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
    /// The logical shape read.
    pub shape: Vec<usize>,
    /// The physical layout the shape is read through (strides and offset).
    pub layout: Layout,
}

/// One generation request. See the module docs.
#[derive(Clone, Debug)]
pub enum KernelRequest {
    /// A one- or two-operand per-element kernel, a scalar-literal kernel or a dtype cast.
    Elementwise(ElementwiseSpec),
    /// A fused pointwise region over leaves.
    Pointwise(PointwiseSpec),
    /// A row-wise fused region or a last-axis reduction.
    Row(RowSpec),
    /// A matmul or GEMV in one of its schedules.
    Contraction(ContractionSpec),
    /// The online-softmax attention generator, in its streaming or split-KV stage-one schedule.
    Attention(AttentionSpec),
    /// The combine stage of a split-KV attention ([`AttentionSchedule::SplitKv`]).
    AttentionCombine(AttentionCombineSpec),
    /// A streaming recurrence over the sequence axis (delta rule or SSD).
    Scan(ScanSpec),
    /// A pure index-mapping copy: transpose, slice, broadcast, reshape, gather, concat, update, scatter.
    Movement(MovementSpec),
    /// A packed-weight kernel that is not a contraction.
    Packed(PackedRequest),
    /// Rotary position embedding over the leading `rot` channels of the head dim.
    Rope(RopeSpec),
    /// Batched argtop-k index extraction.
    TopK(TopKSpec),
    /// The per-query block selection of a sparse-attention indexer: a radix select over block scores.
    BlockTopK(BlockTopKSpec),
}

impl KernelRequest {
    /// What this request's kernel needs from the device beyond its body (see [`KernelRequirements`]).
    pub fn requirements(&self) -> KernelRequirements {
        match self {
            Self::Contraction(
                ContractionSpec::TensorCore { .. } | ContractionSpec::Coopmat { .. },
            ) => KernelRequirements {
                subgroup_lanes: Some(FRAGMENT_COLLECTIVE_LANES),
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
            Self::Attention(spec) => KernelRequirements {
                subgroup_lanes: None,
                serial_work: Some(spec.serial_work()),
            },
            Self::AttentionCombine(spec) => KernelRequirements {
                subgroup_lanes: None,
                serial_work: Some(spec.serial_work()),
            },
            Self::Scan(spec) => KernelRequirements {
                subgroup_lanes: None,
                serial_work: Some(spec.serial_work()),
            },
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
                schedule,
                k,
                n,
                epilogue,
            }) => {
                let launch = match schedule {
                    Schedule::Gemv {
                        width,
                        cols,
                        unroll,
                    } => format!("w{width}c{cols}u{unroll}"),
                    Schedule::Tiled { tile } => format!("tiled{}", tile.get()),
                };
                let epilogue = epilogue.as_ref().map_or_else(String::new, |epilogue| {
                    format!(":epilogue{}", epilogue.steps.len())
                });
                format!("contraction:dense_gemv:{weight}:{launch}:{k}x{n}{epilogue}")
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
            Self::AttentionCombine(_) => "attention_combine",
            Self::Scan(_) => "scan",
            Self::Movement(_) => "movement",
            Self::Packed(_) => "packed",
            Self::Rope(_) => "rope",
            Self::TopK(_) => "topk",
            Self::BlockTopK(_) => "block_top_k",
        }
    }
}

/// A unary kernel's operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// A kernel-ir unary operation.
    Basic(UnOp),
    /// Reciprocal `1 / x`.
    Recip,
    /// A kernel-ir math intrinsic.
    Math(MathOp),
    /// Hyperbolic tangent.
    Tanh,
    /// Error function.
    Erf,
    /// Leading-zero count of an I32 word.
    ClzI32,
}

/// A binary kernel's operation over exact I32 storage with a literal operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum I32Binary {
    /// Unsigned `>=` over the I32 words, writing I32 0 or 1.
    GeU,
    /// Unsigned remainder over the I32 words.
    RemU,
    /// A kernel-ir binary operation, in I32 arithmetic.
    Basic(BinOp),
}

/// A two-value binary kernel's operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueBinary {
    /// The element type the operation runs in (`Ty::I32` for an exact-I32 value).
    Basic(BinOp, Ty),
    /// Unsigned `>=` over the I32 words, writing I32 0 or 1.
    GeU,
    /// Unsigned remainder over the I32 words.
    RemU,
}

/// A dtype cast between two storage types.
#[derive(Clone, Debug)]
pub enum CastSpec {
    /// `f32` to `bf16`, one thread per element.
    F32ToBf16 {
        /// Elements converted.
        numel: usize,
    },
    /// `f32` to IEEE half, one thread per element.
    F32ToF16 {
        /// Elements converted.
        numel: usize,
    },
    /// IEEE half to `f32`, one thread per element.
    F16ToF32 {
        /// Elements converted.
        numel: usize,
    },
    /// `bf16` to `f32`, one thread per element.
    Bf16ToF32 {
        /// Elements converted.
        numel: usize,
    },
    /// `i32` to `f32`, one thread per element.
    I32ToF32 {
        /// Elements converted.
        numel: usize,
    },
    /// Pack f32 rows into E4M3FN words; one thread per packed output word.
    F32ToE4m3 {
        /// Logical shape of the `f32` input and the packed output (row-major, the last axis is the row).
        out_shape: Vec<usize>,
        /// Length of a row: the last-axis extent, packed to a whole number of words per row.
        row_len: usize,
    },
    /// Widen packed E4M3FN rows to f32; one thread per output element.
    E4m3ToF32 {
        /// Length of an output row (the last-axis extent); the packed input rows are padded to whole words.
        row_len: usize,
        /// Output elements, one thread each.
        numel: usize,
    },
}

/// A per-element kernel over one or two operands.
#[derive(Clone, Debug)]
pub enum ElementwiseSpec {
    /// One operand, optionally read through a view of `view.shape` (the output shape).
    Unary {
        /// The operation.
        op: UnaryOp,
        /// The element type the operation runs in.
        dt: Ty,
        /// A view the operand is read through, or `None` for a dense row-major operand.
        view: Option<ViewOperand>,
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// `value op literal` in float arithmetic; `lit_bits` is the literal's `f32` bits.
    ScalarFloat {
        /// The binary operation.
        op: BinOp,
        /// The element type the operation runs in.
        dt: Ty,
        /// The literal operand's `f32` bit pattern.
        lit_bits: u32,
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// `value op literal` over exact I32 storage.
    ScalarI32 {
        /// The operation.
        op: I32Binary,
        /// The literal operand.
        lit: i32,
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// `a op b` with broadcasting, each operand read through its layout.
    Binary {
        /// The operation and its element type.
        op: ValueBinary,
        /// The broadcast output shape.
        out_shape: Vec<usize>,
        /// The left operand and the layout it is read through.
        a: ViewOperand,
        /// The right operand and the layout it is read through.
        b: ViewOperand,
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// A dtype cast between two storage types.
    Cast(CastSpec),
}

/// Which element form a pointwise region computes in.
#[derive(Clone, Debug)]
pub enum PointwiseForm {
    /// Float arithmetic (`fused_views`) on the elementwise 2-D fold, one thread per output element.
    Float {
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// A single exact-I32 step on the elementwise 2-D fold (the `Select` equation).
    ExactI32 {
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// An exact-I32 region; `pack` (empty for none) lists the lanes a row-folded pack writes, one thread
    /// per packed word.
    ExactI32Packed {
        /// The lanes one packed word is built from; empty for an unpacked region.
        pack: Vec<FusedInput>,
    },
}

/// A fused pointwise region: its output shape, leaves, steps and the form it computes in.
#[derive(Clone, Debug)]
pub struct PointwiseSpec {
    /// The output's logical shape.
    pub out_shape: Vec<usize>,
    /// Output elements; a packed region launches one thread per packed word of them.
    pub numel: usize,
    /// The leaves, each with its shape and the layout it is read through.
    pub leaves: Vec<ViewOperand>,
    /// The region's steps and which value is written out.
    pub kernel: FusedKernel,
    /// Which element form the region computes in.
    pub form: PointwiseForm,
}

/// A row-wise fused region or a last-axis reduction.
#[derive(Clone, Debug)]
pub enum RowSpec {
    /// A row-wise fused region: one workgroup of `width` lanes per row.
    Fused {
        /// The output's logical shape.
        out_shape: Vec<usize>,
        /// Output elements; the row count is `numel / kernel.n_cols`.
        numel: usize,
        /// The leaves, each with its shape and the layout it is read through.
        leaves: Vec<ViewOperand>,
        /// The region's steps and which value is written out.
        kernel: RowKernel,
        /// Lanes per workgroup, one workgroup per row.
        width: usize,
    },
    /// A last-axis reduction over `cols`, one thread per output element.
    ReduceLast {
        /// The element type reduced.
        dt: Ty,
        /// The reduction operator.
        op: BinOp,
        /// The last-axis extent reduced per output element.
        cols: usize,
        /// The `f32` bit pattern of the reduction's initial value.
        init_bits: u32,
        /// Output elements, one thread each.
        numel: usize,
    },
}

/// The dtype and shapes of a batched matmul.
#[derive(Clone, Debug)]
pub struct MatmulShapes {
    /// The output's shape: batch dimensions then `[M, N]`.
    pub out_shape: Vec<usize>,
    /// The left operand's shape: batch dimensions then `[M, K]`.
    pub a_shape: Vec<usize>,
    /// The right operand's shape: batch dimensions then `[K, N]`, or `[N, K]` for a checkpoint-order weight.
    pub b_shape: Vec<usize>,
}

/// A contraction request: which schedule and which operand storage computes one matmul or GEMV.
#[derive(Clone, Debug)]
pub enum ContractionSpec {
    /// The one-thread-per-output reference kernel, with `epilogue` applied to each output before the store.
    Serial {
        /// The element type of the operands and result.
        dt: Ty,
        /// Pointwise steps applied to each output before the store, if any.
        epilogue: Option<Epilogue>,
        /// The matmul's shapes.
        shapes: MatmulShapes,
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// The one-thread-per-output kernel over a weight in checkpoint `[N, K]` order (`shapes.b_shape` is `[N, K]`):
    /// `DenseContraction` where the tiled GEMM is not taken. The activation and output are f32; `weight` is the
    /// weight's planned storage, `BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`.
    DenseSerial {
        /// The weight's planned storage: `BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`.
        weight: BufferStorage,
        /// The matmul's shapes; `b_shape` is `[N, K]`.
        shapes: MatmulShapes,
        /// How the launch folds past the device's x-grid cap.
        fold: Fold,
    },
    /// Matrix-core fragments (WMMA m16n16k16): one 256-thread block per 16x16 output tile per batch.
    TensorCore {
        /// The element type of the fragments.
        dt: Ty,
        /// The matmul's shapes; the output extents are multiples of 16.
        shapes: MatmulShapes,
    },
    /// RADV cooperative matrix: one 32-invocation subgroup per 16x16 output tile.
    Coopmat {
        /// The matmul's shapes; the output extents are multiples of 16.
        shapes: MatmulShapes,
    },
    /// One range of a watchdog-split decode GEMV: `elems` workgroups of `width` lanes starting at output
    /// element `offset`.
    GemvChunk {
        /// Reduction length (the `[K, N]` weight's rows).
        k: usize,
        /// Output columns of the `[K, N]` weight.
        n: usize,
        /// Lanes per workgroup.
        width: usize,
        /// Workgroups in this range, one per output element.
        elems: usize,
        /// First output element of the range.
        offset: usize,
    },
    /// The `DenseContraction` decode GEMV (`M == 1`): `schedule` (a [`Schedule::Gemv`]) over an `[N, K]` weight
    /// in `weight` storage (`BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`), the shared Gemv loop nest
    /// with the dense operand load, `epilogue` (f32) applied to each column's sum before its store. The body is
    /// shape-generic: `K` and `N` ride in its `[K, N]` metadata; `k` also fixes whether a packed weight row
    /// starts on a word boundary.
    ///
    /// The weight is `[N, K]` by construction: one workgroup per output column over a `[K, N]` weight would read
    /// it at a stride of `N` elements per lane (the pattern card 035 removed), so the variant has no orientation
    /// field and a `[K, N]` Gemv cannot be written down.
    ///
    /// ```
    /// use poot_kernelgen::{ContractionSpec, Schedule};
    /// use poot_target::BufferStorage;
    ///
    /// let _gemv_over_n_by_k = ContractionSpec::DenseGemv {
    ///     weight: BufferStorage::f32(),
    ///     schedule: Schedule::Gemv { width: 256, cols: 4, unroll: 4 },
    ///     k: 896,
    ///     n: 4864,
    ///     epilogue: None,
    /// };
    /// ```
    ///
    /// ```compile_fail,E0559
    /// use poot_kernelgen::{ContractionSpec, Schedule, WeightLayout};
    /// use poot_target::BufferStorage;
    ///
    /// let _gemv_over_k_by_n = ContractionSpec::DenseGemv {
    ///     weight: BufferStorage::f32(),
    ///     layout: WeightLayout::Kn,
    ///     schedule: Schedule::Gemv { width: 256, cols: 4, unroll: 4 },
    ///     k: 896,
    ///     n: 4864,
    ///     epilogue: None,
    /// };
    /// ```
    DenseGemv {
        /// The weight's planned storage: `BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`.
        weight: BufferStorage,
        /// The Gemv loop nest (a [`Schedule::Gemv`]).
        schedule: Schedule,
        /// The reduction length, the weight row length.
        k: usize,
        /// Output columns, the weight's rows.
        n: usize,
        /// Pointwise steps applied to each column's sum before the store, if any.
        epilogue: Option<Epilogue>,
    },
    /// Decode attention `scores @ V`: an LDS-parallel reduction over `cap`, `width` lanes per output.
    AttnScoresVGemv {
        /// Reduction length: the cache capacity.
        cap: usize,
        /// Output width per row.
        d: usize,
        /// Lanes per output, reducing over `cap` through workgroup memory.
        width: usize,
        /// Outputs, one workgroup each.
        numel: usize,
    },
    /// The synthesized tiled GEMM: `groups` workgroups of `tile * tile` lanes starting at tile `offset`, with
    /// `epilogue` (f32) applied to each output element's sum before its store.
    TiledRegion {
        /// Activation rows.
        m: usize,
        /// Reduction length.
        k: usize,
        /// Output columns.
        n: usize,
        /// The tile edge; a workgroup is `tile * tile` lanes.
        tile: usize,
        /// First tile of this range, in the grid's tile order.
        offset: usize,
        /// Workgroups in this range.
        groups: usize,
        /// Pointwise steps applied to each output before the store, if any.
        epilogue: Option<Epilogue>,
    },
    /// [`ContractionSpec::TiledRegion`] reading its weight in checkpoint `[N, K]` order: the `DenseContraction`
    /// prefill GEMM. `weight` as in [`ContractionSpec::DenseSerial`].
    DenseTiled {
        /// Activation rows.
        m: usize,
        /// Reduction length.
        k: usize,
        /// Output columns, the weight's rows.
        n: usize,
        /// The tile edge; a workgroup is `tile * tile` lanes.
        tile: usize,
        /// First tile of this range, in the grid's tile order.
        offset: usize,
        /// Workgroups in this range.
        groups: usize,
        /// The weight's planned storage: `BufferStorage::f32()`, `f16_packed()` or `bf16_packed()`.
        weight: BufferStorage,
        /// Pointwise steps applied to each output before the store, if any.
        epilogue: Option<Epilogue>,
    },
    /// A contraction over a packed weight: the descriptor-driven decode of `format` served to the planner's
    /// schedule ([`PackedContraction`]).
    Packed(PackedContraction),
    /// A contraction over a stacked `[E, K, N]` weight whose output row `m` reads expert slab `ids[m]`
    /// ([`IndexedContraction`]).
    Indexed(IndexedContraction),
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
            Self::Packed(_) => "packed",
            Self::Indexed(_) => "indexed",
        }
    }
}

/// A contraction over a packed weight in `format`: `schedule`'s loop nest over the weight's descriptor-driven
/// decode, with `epilogue` applied to each column's sum before its store. The planner validates and chooses the
/// launch (the schedule's grid, folded and checked against the device's limits with a typed refusal), so `grid`
/// and `meta` ride in the request.
#[derive(Clone, Debug)]
pub struct PackedContraction {
    /// The weight's format, whose descriptor drives the decode.
    pub format: WeightFormat,
    /// How output rows meet weight rows: one weight (block-diagonal or dense), or an expert-stacked table.
    pub rows: PackedRows,
    /// The loop nest the decode is served to.
    pub schedule: Schedule,
    /// Pointwise steps applied to each column's sum before the store, if any.
    pub epilogue: Option<Epilogue>,
    /// Total threads `[x, y, z]` the planner chose for the launch.
    pub grid: [u32; 3],
    /// The metadata buffer's contents (`K`, `N` and further shape facts, by schedule).
    pub meta: Vec<u32>,
}

/// How a [`PackedContraction`]'s output rows meet the packed weight's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedRows {
    /// One weight: the output column is the weight row (`blocks` of them side by side). Params
    /// `(activation, sources.., [epilogue leaves..], metadata, out)`.
    Blocks,
    /// An expert-stacked table: output row `m` reads the slab of expert `ids[m]` in every source, and an id
    /// outside `0..E` makes that row NaN. Params `(activation, ids, sources.., [epilogue
    /// leaves..], metadata, out)`, metadata `[K, N, E, slab_words per source..]`. A [`Schedule::Gemv`] only.
    Indexed,
}

impl PackedRows {
    /// The index of the first source parameter: after the unit param, the activation, and (indexed) the ids.
    pub(crate) const fn leading_params(self) -> u32 {
        match self {
            Self::Blocks => 2,
            Self::Indexed => 3,
        }
    }
}

/// A contraction over a stacked `[E, K, N]` weight, `out[m, n] = sum over k of x[m, k] * W[ids[m], k, n]`: output
/// row `m` reads expert slab `ids[m]`, and an id outside `0..E` makes that row NaN. `schedule` is
/// [`Schedule::Gemv`] (decode) or [`Schedule::Tiled`], which is segmented here: a tile whose rows select several
/// experts walks them. The activation and output are f32, the ids are `I32`, and the weight is read in `weight` storage (f32 or
/// packed BF16); `epilogue` (f32) runs over each output element's sum before its store. The planner chooses the
/// launch (the grid folded and checked against the device's limits) and the metadata `[K, N, E, M, x_groups]`.
#[derive(Clone, Debug)]
pub struct IndexedContraction {
    /// [`Schedule::Gemv`] for decode or [`Schedule::Tiled`] for prefill.
    pub schedule: Schedule,
    /// How each expert slab of the weight is stored: [`WeightLayout::Kn`] `[K, N]` or [`WeightLayout::Nk`] `[N, K]`.
    pub layout: WeightLayout,
    /// The weight buffer's planned storage: `BufferStorage::f32()` or `BufferStorage::bf16_packed()`. Part of the
    /// request, so a body built for one storage is never served for another.
    pub weight: BufferStorage,
    /// Pointwise steps applied to each output element's sum before the store, if any.
    pub epilogue: Option<Epilogue>,
    /// Total threads `[x, y, z]` the planner chose for the launch.
    pub grid: [u32; 3],
    /// The metadata buffer's contents: `[K, N, E, M, x_groups]`.
    pub meta: Vec<u32>,
}

/// A generated packed-weight kernel that is not a contraction (a standalone dequantize or a row gather). The
/// planner validates and chooses the launch, so it rides in the request.
#[derive(Clone, Debug)]
pub struct PackedRequest {
    /// Name the generated body carries.
    pub name: &'static str,
    /// Which packed kernel to generate.
    pub spec: PackedKernelSpec,
    /// Total threads `[x, y, z]` the planner chose for the launch.
    pub grid: [u32; 3],
    /// The metadata buffer's contents.
    pub meta: Vec<u32>,
}

/// Rotary position embedding in half-split (GPT-NeoX) form, over the leading `rot` channels of the head dim.
#[derive(Clone, Debug)]
pub struct RopeSpec {
    /// The storage element type of `x`, `cos`, `sin` and the output.
    pub dt: Ty,
    /// The shape of `x`, which is also the output's.
    pub x_shape: Vec<usize>,
    /// The shape of `cos` and `sin`, broadcast into `x` over the leading axes.
    pub cos_shape: Vec<usize>,
    /// Channels of the head dim that are rotated; the rest pass through.
    pub rot: usize,
    /// Output elements, one thread each.
    pub numel: usize,
}

/// Batched argtop-k index extraction: `rank [rows, e]` to `out [rows, k]`.
#[derive(Clone, Debug)]
pub struct TopKSpec {
    /// Entries per row of `rank`.
    pub e: usize,
    /// Indices extracted per row.
    pub k: usize,
    /// Output elements (`rows * k`), one thread each.
    pub numel: usize,
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

/// Generate the kernel `req` names.
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
    caps: &DeviceCaps,
    limits: &BodyLimits,
) -> Result<Generated, KernelGenError> {
    if let Some(size) = recipe_size(req) {
        BodyBudget::new(*limits).reserve(req.family(), BodyStage::Sizing, size)?;
    }
    let generated = build(req, caps)?;
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
                PointwiseForm::Float { .. } | PointwiseForm::ExactI32 { .. } => 0,
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

fn build(req: &KernelRequest, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    match req {
        KernelRequest::Elementwise(spec) => elementwise(spec, Placement::Grid(caps)),
        KernelRequest::Pointwise(spec) => pointwise(spec, caps),
        KernelRequest::Row(spec) => row(spec, caps),
        KernelRequest::Contraction(spec) => contraction(spec, caps),
        KernelRequest::Attention(spec) => attention(spec, caps),
        KernelRequest::AttentionCombine(spec) => attention_combine(spec, caps),
        KernelRequest::Scan(spec) => scan(spec, caps),
        KernelRequest::BlockTopK(spec) => block_top_k(spec, caps),
        KernelRequest::Movement(spec) => movement(spec, caps),
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

fn elementwise(
    spec: &ElementwiseSpec,
    placement: Placement<'_>,
) -> Result<Generated, KernelGenError> {
    match spec {
        ElementwiseSpec::Unary { op, dt, view, fold } => {
            let xg = placement.x_groups(*fold);
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
            with(body, placement.launch(*fold))
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
                placement.x_groups(*fold),
            ),
            placement.launch(*fold),
        ),
        ElementwiseSpec::ScalarI32 { op, lit, fold } => {
            let xg = placement.x_groups(*fold);
            let body = match op {
                I32Binary::GeU => binary_scalar_i32_geu_grid(ENTRY, *lit, xg),
                I32Binary::RemU => binary_scalar_i32_remu_grid(ENTRY, *lit, xg),
                I32Binary::Basic(op) => binary_scalar_i32_grid(ENTRY, *op, *lit, xg),
            };
            with(body, placement.launch(*fold))
        }
        ElementwiseSpec::Binary {
            op,
            out_shape,
            a,
            b,
            fold,
        } => {
            let xg = placement.x_groups(*fold);
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
            with(body, placement.launch(*fold))
        }
        ElementwiseSpec::Cast(cast) => match cast {
            CastSpec::F32ToBf16 { numel } => with(cast_f32_to_bf16(ENTRY), Launch::flat(*numel)),
            CastSpec::F32ToF16 { numel } => with(cast_f32_to_f16(ENTRY), Launch::flat(*numel)),
            CastSpec::F16ToF32 { numel } => with(cast_f16_to_f32(ENTRY), Launch::flat(*numel)),
            CastSpec::Bf16ToF32 { numel } => with(cast_bf16_to_f32(ENTRY), Launch::flat(*numel)),
            CastSpec::I32ToF32 { numel } => with(cast_i32_to_f32(ENTRY), Launch::flat(*numel)),
            CastSpec::F32ToE4m3 { out_shape, row_len } => with(
                f32_to_e4m3fn_packed(ENTRY, *row_len)?,
                Launch::flat(Storage::BYTE_LANES.units(out_shape)),
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
        PointwiseForm::Float { fold } => with(
            fused_views(
                ENTRY,
                &spec.out_shape,
                &shapes,
                &layouts,
                &spec.kernel,
                fold.x_groups(caps),
            )?,
            fold.launch(caps),
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
            epilogue,
            shapes,
            fold,
        } => {
            let body = match epilogue {
                Some(epilogue) => matmul_batched_epilogue_dt_grid(
                    ENTRY,
                    dt.clone(),
                    &shapes.out_shape,
                    &shapes.a_shape,
                    &shapes.b_shape,
                    epilogue,
                    fold.x_groups(caps),
                )?,
                None => matmul_batched_dt_grid(
                    ENTRY,
                    dt.clone(),
                    &shapes.out_shape,
                    &shapes.a_shape,
                    &shapes.b_shape,
                    fold.x_groups(caps),
                ),
            };
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
            let batch: usize = shapes.out_shape[..r - 2].iter().product();
            let tiles = (shapes.out_shape[r - 2] / 16) * (shapes.out_shape[r - 1] / 16);
            with(body, Launch::flat(batch * tiles * 32))
        }
        ContractionSpec::GemvChunk {
            k,
            n,
            width,
            elems,
            offset,
        } => with(
            gemv_lds(
                ENTRY,
                *k,
                *n,
                *width,
                *elems,
                *offset,
                WeightLayout::Kn,
                BufferStorage::f32(),
            )?,
            Launch::flat(elems * width),
        ),
        ContractionSpec::DenseGemv {
            weight,
            schedule,
            k,
            n,
            epilogue,
        } => {
            let body = dense_contraction(ENTRY, *weight, *schedule, *k, epilogue.as_ref())?;
            let threads = schedule.grid_threads(1, 1, *n);
            with(
                body,
                Launch {
                    grid: [threads as u32, 1, 1],
                    meta: Some(vec![*k as u32, *n as u32]),
                },
            )
        }
        ContractionSpec::Packed(PackedContraction {
            format,
            rows,
            schedule,
            epilogue,
            grid,
            meta,
        }) => {
            let body = packed_contraction(
                "packed_contraction",
                *format,
                *schedule,
                epilogue.as_ref(),
                *rows,
            )?;
            with(
                body,
                Launch {
                    grid: *grid,
                    meta: Some(meta.clone()),
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
            epilogue,
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
                epilogue.as_ref(),
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
            epilogue,
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
                epilogue.as_ref(),
            )?,
            Launch::flat(groups * tile * tile),
        ),
        ContractionSpec::Indexed(IndexedContraction {
            schedule,
            layout,
            weight,
            epilogue,
            grid,
            meta,
        }) => with(
            indexed_contraction(ENTRY, *schedule, *layout, *weight, epilogue.as_ref())?,
            Launch {
                grid: *grid,
                meta: Some(meta.clone()),
            },
        ),
    }
}

/// The 2-D workgroup fold of `workgroups` onto `caps`' grid, refused when the Y extent it needs is past
/// the device's Y cap (a launch the device cannot make is not a launch).
fn fold_within_grid(
    generator: &'static str,
    workgroups: usize,
    caps: &DeviceCaps,
) -> Result<(usize, usize), KernelGenError> {
    let (x, y) = fold_groups(workgroups, caps.max_grid[0]);
    if y > caps.max_grid[1] as usize {
        return Err(KernelGenError::ExceedsBound {
            generator,
            what: "workgroup rows".to_string(),
            value: y,
            bound: caps.max_grid[1] as usize,
        });
    }
    Ok((x, y))
}

fn attention(spec: &AttentionSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    let lanes = spec.schedule.lanes();
    // One workgroup of `lanes` threads per (batch, head, row[, split]), folded onto a 2-D grid past the X cap.
    let (x, y) = fold_within_grid("attention_body", spec.workgroups(), caps)?;
    with(
        attention_body(ENTRY, spec, x)?,
        Launch::grid([(x * lanes) as u32, y as u32, 1]),
    )
}

fn attention_combine(
    spec: &AttentionCombineSpec,
    caps: &DeviceCaps,
) -> Result<Generated, KernelGenError> {
    let (x, y) = fold_within_grid("attention_combine_body", spec.rows, caps)?;
    with(
        attention_combine_body(ENTRY, spec, x)?,
        Launch::grid([(x * spec.lanes) as u32, y as u32, 1]),
    )
}

fn scan(spec: &ScanSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    let (x, y) = fold_groups(spec.workgroups(), caps.max_grid[0]);
    with(
        scan_body(ENTRY, spec, x)?,
        Launch::grid([(x * spec.cols) as u32, y as u32, 1]),
    )
}

fn block_top_k(spec: &BlockTopKSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    let (x, y) = fold_groups(spec.queries, caps.max_grid[0]);
    let lanes = crate::block_top_k::LANES;
    with(
        block_top_k_body(ENTRY, spec, x)?,
        Launch::grid([(x * lanes) as u32, y as u32, 1]),
    )
}

fn movement(spec: &MovementSpec, caps: &DeviceCaps) -> Result<Generated, KernelGenError> {
    with(
        movement_body(ENTRY, spec, spec.fold.x_groups(caps))?,
        spec.fold.launch(caps),
    )
}

/// How an elementwise kernel's threads are placed: on a device's thread grid, or run one after another by a
/// single core.
#[derive(Clone, Copy)]
enum Placement<'a> {
    /// A thread grid bounded by the device's caps (the 2-D fold reads them).
    Grid(&'a DeviceCaps),
    /// One core, no grid cap: the planner never folds a kernel it serializes.
    Serial,
}

impl Placement<'_> {
    fn x_groups(self, fold: Fold) -> Option<usize> {
        match self {
            Placement::Grid(caps) => fold.x_groups(caps),
            Placement::Serial => None,
        }
    }

    fn launch(self, fold: Fold) -> Launch {
        match self {
            Placement::Grid(caps) => fold.launch(caps),
            Placement::Serial => Launch::flat(fold.numel),
        }
    }
}

/// How far an AIE op's F32 result may be from the exact (infinitely precise) result, for finite operands
/// whose result is a finite normal number (overflow, underflow and subnormal results are not promised).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorBound {
    /// The IEEE-754 round-to-nearest-even result, bit for bit.
    Exact,
    /// At most `n * f32::EPSILON * max(|a|, |b|)`. The AIE2p adder does not round to nearest even: its
    /// result is faithful (within one unit in the last place of the result, which is up to two units of the
    /// larger operand), and a cancellation that leaves a small result is exact (Sterbenz). The error is
    /// therefore a bound on the operands, not on the result.
    OperandUlps(u32),
    /// `|got - exact| <= abs + rel * |exact|`.
    Result {
        /// The absolute part of the bound.
        abs: f32,
        /// The relative part of the bound.
        rel: f32,
    },
}

/// The numerical contract of one AIE pointwise op: what its F32 result is promised to be. The AIE2p core has
/// no hardware transcendental unit, so an op is either one the core executes (and rounds in its own way) or
/// a software sequence, and the sequence's accuracy is part of the kernel's identity, not an accident of its
/// lowering. Tier 2 compares a run against the exact result within `bound`, over the inputs in `domain`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericalContract {
    /// The contract's name, stated in the kernel's identity.
    pub id: &'static str,
    /// The error the result may have.
    pub bound: ErrorBound,
    /// The inclusive range of the (first) operand the bound is promised over; the core's software sequences
    /// clamp or lose accuracy outside it. Arithmetic is promised over every finite operand.
    pub domain: (f32, f32),
}

/// A pointwise op an AIE core runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AieOp {
    /// `a <op> b` over F32.
    Binary(BinOp),
    /// `<op>(x)` over F32.
    Unary(UnaryOp),
}

impl AieOp {
    /// The numerical contract the op runs under, or `None` when no lowering is proved for it. An op without a
    /// contract is a typed refusal: it never silently runs under another op's. Each contract was measured on
    /// an XDNA2 core against the exact result (`poot-npu/tests/aie_pointwise.rs`), and the bound has margin
    /// over the worst error seen; an op the real Peano toolchain cannot compile (`sqrt`, `round`, `max`: it
    /// aborts) has none.
    pub fn contract(self) -> Option<NumericalContract> {
        const FINITE: (f32, f32) = (f32::MIN, f32::MAX);
        let exact = |id| NumericalContract {
            id,
            bound: ErrorBound::Exact,
            domain: FINITE,
        };
        let software = |id, domain| NumericalContract {
            id,
            bound: ErrorBound::Result {
                abs: 2e-6,
                rel: 1e-5,
            },
            domain,
        };
        Some(match self {
            AieOp::Binary(BinOp::Add | BinOp::Sub) => NumericalContract {
                id: "f32-add-sub-aie2p-v1",
                bound: ErrorBound::OperandUlps(2),
                domain: FINITE,
            },
            AieOp::Binary(BinOp::Mul) => exact("f32-mul-rne-v1"),
            AieOp::Binary(BinOp::Div) => exact("f32-div-rne-v1"),
            AieOp::Binary(BinOp::Ge) => exact("f32-ge-v1"),
            AieOp::Unary(UnaryOp::Basic(UnOp::Neg)) => exact("f32-neg-v1"),
            AieOp::Unary(UnaryOp::Recip) => exact("f32-recip-rne-v1"),
            // The software sequences (`poot-codegen`'s AIE core lowering): exp clamps its argument to
            // [-88, 88], log is the atanh series, tanh and erf are the devices' shared closed forms.
            AieOp::Unary(UnaryOp::Math(MathOp::Exp)) => {
                software("f32-exp-software-v1", (-87.0, 88.0))
            }
            AieOp::Unary(UnaryOp::Math(MathOp::Log)) => {
                software("f32-log-software-v1", (f32::MIN_POSITIVE, f32::MAX))
            }
            AieOp::Unary(UnaryOp::Tanh) => software("f32-tanh-software-v1", FINITE),
            AieOp::Unary(UnaryOp::Erf) => software("f32-erf-software-v1", FINITE),
            AieOp::Binary(_) | AieOp::Unary(_) => return None,
        })
    }
}

/// The entry symbol of a serialized body (the AIE core object's C symbol).
const AIE_ENTRY: &str = "poot_aie_kernel";

/// The AIE schedule of an ordinary request (POOT-1161): the request the GPU planner builds, generated by the
/// ordinary generator with one-thread workgroups, then its thread grid serialized into loops
/// ([`poot_kernel_ir::serialize::serialize_grid`]) because an AIE core runs one scalar thread. No second
/// generator exists; a request family gains an AIE schedule when its ordinary body has no thread cooperation
/// and the family's ops have a [`NumericalContract`]. The iteration count and the proved access ranges come
/// back in [`SerializedGenerated`].
///
/// Only a dense, equal-shape F32 elementwise unary or binary op with a contract has an AIE schedule today;
/// any other request is a typed [`KernelGenError::Unsupported`] (no emulated grid, no host fallback).
pub fn serialized_schedule(
    req: &KernelRequest,
    limits: &BodyLimits,
) -> Result<SerializedGenerated, KernelGenError> {
    const GENERATOR: &str = "serialized_schedule";
    let KernelRequest::Elementwise(spec) = req else {
        return Err(unsupported(
            req.family(),
            "no AIE schedule for this request family",
        ));
    };
    let (op, out_shape, operands, fold) = match spec {
        ElementwiseSpec::Binary {
            op: ValueBinary::Basic(op, Ty::F32),
            out_shape,
            a,
            b,
            fold,
        } => (AieOp::Binary(*op), out_shape.clone(), vec![a, b], fold),
        ElementwiseSpec::Unary {
            op,
            dt: Ty::F32,
            view: None,
            fold,
        } => (AieOp::Unary(*op), vec![fold.numel], Vec::new(), fold),
        _ => {
            return Err(unsupported(
                GENERATOR,
                "only F32 elementwise unary and binary ops have an AIE schedule",
            ));
        }
    };
    let contract = op.contract().ok_or_else(|| {
        unsupported(
            GENERATOR,
            "the AIE core has no proved lowering and numerical contract for this op",
        )
    })?;
    for operand in &operands {
        if operand.shape != out_shape {
            return Err(unsupported(
                GENERATOR,
                "an operand is broadcast against the output; the AIE shape ABI binds dense equal-shape buffers",
            ));
        }
        if operand.layout != Layout::contiguous(&operand.shape) {
            return Err(unsupported(
                GENERATOR,
                "an operand is a strided view; the AIE shape ABI binds dense row-major buffers",
            ));
        }
    }
    if fold.two_d {
        return Err(unsupported(
            GENERATOR,
            "a 2-D folded launch has no AIE schedule; the core loops the whole tensor",
        ));
    }
    let generated = elementwise(spec, Placement::Serial)?;
    let threads = generated.launch.grid;
    let iterations = threads.iter().map(|&n| u64::from(n)).product::<u64>();
    if iterations != fold.numel as u64 {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "output element count against the launch's u32 thread extent".into(),
            value: fold.numel,
            bound: u32::MAX as usize,
        });
    }
    let mut body = generated.body;
    // Serialization turns every thread into a loop iteration, so the workgroup has one thread. The body is
    // the AIE core object's C symbol.
    body.name = AIE_ENTRY.to_string();
    body.workgroup_size = [1, 1, 1];
    let body =
        serialize_grid(&body, Grid { threads }).map_err(|source| KernelGenError::Serialize {
            generator: GENERATOR,
            source,
        })?;
    BodyBudget::new(*limits).reserve(GENERATOR, BodyStage::Finished, BodySize::of(&body))?;
    let accessed = iterations;
    let dims: Vec<u64> = out_shape.iter().map(|&d| d as u64).collect();
    let writes = std::iter::repeat_n(false, operands.len().max(1)).chain([true]);
    Ok(SerializedGenerated {
        body,
        iterations,
        contract,
        bindings: writes
            .map(|writes| SerializedBinding {
                dims: dims.clone(),
                writes,
                accessed,
            })
            .collect(),
    })
}

/// A sequentially scheduled kernel: its body, how many loop iterations one invocation runs, the numerical
/// contract its op runs under, and the number of leading elements the body reads or writes of each parameter
/// (binding order). The access counts are what the artifact's shape ABI is checked against, so a body cannot
/// be linked against a shape it would read past.
#[derive(Clone, Debug)]
pub struct SerializedGenerated {
    /// The kernel body.
    pub body: Body,
    /// Loop iterations one invocation runs.
    pub iterations: u64,
    /// The numerical contract the op runs under.
    pub contract: NumericalContract,
    /// One entry per body parameter, in binding order.
    pub bindings: Box<[SerializedBinding]>,
}

/// One buffer of a sequential kernel: its logical dimensions, whether the loop writes it, and how many
/// leading elements the loop touches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerializedBinding {
    /// Logical dimensions of the buffer.
    pub dims: Vec<u64>,
    /// Whether the loop writes the buffer.
    pub writes: bool,
    /// Leading elements the loop reads or writes.
    pub accessed: u64,
}

#[cfg(test)]
mod tests {
    use crate::{EpilogueInput, EpilogueLeaf, EpilogueStep, FusedScalarOp};
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
                    schedule: Schedule::Gemv {
                        width: 256,
                        cols: 4,
                        unroll: 4,
                    },
                    k: 896,
                    n: 4864,
                    epilogue: None,
                }),
                format!(
                    "contraction:dense_gemv:{}:w256c4u4:896x4864",
                    BufferStorage::bf16_packed()
                ),
            ),
            (
                KernelRequest::Contraction(ContractionSpec::Serial {
                    dt: Ty::F32,
                    epilogue: None,
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

    /// Finite BF16 bit patterns (both signs, exponents around zero) from a fixed xorshift stream.
    fn finite_bf16_bits(count: usize, seed: u64) -> Vec<u16> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let bits = (state >> 32) as u16;
                let exponent = 0x70 + ((bits >> 7) & 0xff) % 0x18;
                (bits & 0x8000) | (exponent << 7) | (bits & 0x7f)
            })
            .collect()
    }

    /// Run `request` in the KIR interpreter over `[activation, weight, leaf, meta.., out]` and return `out`.
    fn interpret_with_leaf(
        request: &KernelRequest,
        weight: poot_kernel_ir::interp::Buffer,
        activation: &[f32],
        leaf: poot_kernel_ir::interp::Buffer,
        out_len: usize,
    ) -> Vec<f32> {
        use poot_kernel_ir::interp::{Buffer, run};
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        let generated = generate(request, &caps, &roomy()).expect("stateable");
        let workgroup = generated.body.workgroup_size;
        let workgroups =
            std::array::from_fn(|axis| generated.launch.grid[axis].div_ceil(workgroup[axis]));
        let mut buffers = vec![Buffer::from_f32s(activation), weight, leaf];
        buffers.extend(generated.launch.meta.as_deref().map(Buffer::from_u32s));
        buffers.push(Buffer::from_f32s(&vec![0.0; out_len]));
        run(&generated.body, workgroups, &mut buffers).expect("interpreter");
        buffers.last().unwrap().to_f32s().unwrap()
    }

    /// A bias stored BF16 and widened by a `Cast` step reads, through the packed lane of every epilogue body (the
    /// decode GEMV, the one-thread-per-output dense kernel and the `[K, N]` generated kernel), to the bit what the
    /// same bias reads from an f32 buffer holding its widened values. The bias is odd-sized, so the last packed word
    /// carries one element and a zero pad. Mutation: in `Epilogue::leaf_param` declare a packed leaf as an f32
    /// slice; the three bodies stop verifying (or read the wrong words).
    #[test]
    fn epilogue_bodies_read_a_packed_bf16_bias_as_its_widened_values() {
        use poot_kernel_ir::interp::Buffer;
        let (k, n) = (6usize, 5usize);
        let bias_bits = finite_bf16_bits(n, 11);
        let widened: Vec<f32> = bias_bits
            .iter()
            .map(|&b| f32::from_bits(u32::from(b) << 16))
            .collect();
        let words: Vec<u32> = bias_bits
            .chunks(2)
            .map(|pair| u32::from(pair[0]) | (u32::from(pair.get(1).copied().unwrap_or(0)) << 16))
            .collect();
        let weight_values: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.25 - 0.5).collect();
        let activation: Vec<f32> = (0..k).map(|i| (i as f32) * 0.5 - 1.0).collect();
        // `head + bias`, the bias read through `lane`: a Widen step over a packed leaf, or the f32 leaf itself.
        let epilogue = |storage: Option<BufferStorage>| match storage {
            Some(_) => Epilogue {
                dt: Ty::F32,
                out_shape: vec![1, n],
                leaves: vec![EpilogueLeaf {
                    shape: vec![n],
                    storage,
                }],
                steps: vec![
                    EpilogueStep {
                        op: FusedScalarOp::Widen,
                        inputs: vec![EpilogueInput::Leaf(0)],
                    },
                    EpilogueStep {
                        op: FusedScalarOp::Binary(BinOp::Add),
                        inputs: vec![EpilogueInput::Head, EpilogueInput::Step(0)],
                    },
                ],
                output: EpilogueInput::Step(1),
            },
            None => Epilogue {
                dt: Ty::F32,
                out_shape: vec![1, n],
                leaves: vec![EpilogueLeaf {
                    shape: vec![n],
                    storage: None,
                }],
                steps: vec![EpilogueStep {
                    op: FusedScalarOp::Binary(BinOp::Add),
                    inputs: vec![EpilogueInput::Head, EpilogueInput::Leaf(0)],
                }],
                output: EpilogueInput::Step(0),
            },
        };
        let shapes = |b_shape: Vec<usize>| MatmulShapes {
            out_shape: vec![1, n],
            a_shape: vec![1, k],
            b_shape,
        };
        let fold = Fold {
            numel: n,
            two_d: false,
            width: 64,
        };
        let bodies = |storage: Option<BufferStorage>| {
            [
                (
                    ContractionSpec::DenseGemv {
                        weight: BufferStorage::f32(),
                        schedule: Schedule::Gemv {
                            width: 16,
                            cols: 2,
                            unroll: 2,
                        },
                        k,
                        n,
                        epilogue: Some(epilogue(storage)),
                    },
                    weight_values.clone(),
                ),
                (
                    ContractionSpec::DenseTiled {
                        m: 1,
                        k,
                        n,
                        tile: 8,
                        offset: 0,
                        groups: n.div_ceil(8),
                        weight: BufferStorage::f32(),
                        epilogue: Some(epilogue(storage)),
                    },
                    weight_values.clone(),
                ),
                (
                    ContractionSpec::Serial {
                        dt: Ty::F32,
                        epilogue: Some(epilogue(storage)),
                        shapes: shapes(vec![k, n]),
                        fold,
                    },
                    // the same weight, `[K, N]`
                    (0..k * n)
                        .map(|i| weight_values[(i % n) * k + i / n])
                        .collect(),
                ),
            ]
        };
        let packed = bodies(Some(BufferStorage::bf16_packed()));
        let wide = bodies(None);
        for ((packed_spec, weight), (wide_spec, _)) in packed.iter().zip(&wide) {
            let name = packed_spec_name(packed_spec);
            let got = interpret_with_leaf(
                &KernelRequest::Contraction(packed_spec.clone()),
                Buffer::from_f32s(weight),
                &activation,
                Buffer::from_u32s(&words),
                n,
            );
            let want = interpret_with_leaf(
                &KernelRequest::Contraction(wide_spec.clone()),
                Buffer::from_f32s(weight),
                &activation,
                Buffer::from_f32s(&widened),
                n,
            );
            let (got_bits, want_bits): (Vec<u32>, Vec<u32>) = (
                got.iter().map(|v| v.to_bits()).collect(),
                want.iter().map(|v| v.to_bits()).collect(),
            );
            assert_eq!(got_bits, want_bits, "{name}: packed BF16 bias differs");
        }
    }

    fn packed_spec_name(spec: &ContractionSpec) -> &'static str {
        match spec {
            ContractionSpec::DenseGemv { .. } => "dense gemv",
            ContractionSpec::DenseTiled { .. } => "dense tiled",
            _ => "serial",
        }
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
        let generated = generate(request, &caps, &roomy()).expect("stateable");
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
                        schedule: Schedule::Gemv {
                            width: 16,
                            cols: 2,
                            unroll: 2,
                        },
                        k,
                        n,
                        epilogue: None,
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
                        epilogue: None,
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
                schedule: Schedule::Gemv {
                    width: 16,
                    cols: 2,
                    unroll: 2,
                },
                k: 8,
                n: 4,
                epilogue: None,
            });
            match generate(&request, &caps, &roomy()) {
                Err(KernelGenError::Unsupported { request, .. }) => {
                    assert_eq!(request, "dense_contraction")
                }
                Err(other) => panic!("{weight}: expected Unsupported, got {other}"),
                Ok(_) => panic!("{weight}: expected Unsupported, got a generated body"),
            }
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
                        schedule: Schedule::Gemv {
                            width,
                            cols,
                            unroll,
                        },
                        k,
                        n,
                        epilogue: None,
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
                leaf_tys: vec![crate::LeafTy::F32; 1],
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
            form: PointwiseForm::Float {
                fold: Fold {
                    numel: 1,
                    two_d: false,
                    width: 256,
                },
            },
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
        let fits = generate(&request, &caps, &limits(bound.instructions, bound.locals))
            .expect("a cap equal to the bound admits the region");
        assert!(
            BodySize::of(&fits.body).instructions <= bound.instructions,
            "the bound covers the built body"
        );
        let refused = generate(
            &request,
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
        let built = generate(&request, &caps, &roomy()).expect("generates under roomy limits");
        let size = BodySize::of(&built.body);
        let refused =
            generate(&request, &caps, &limits(size.instructions - 1, size.locals)).unwrap_err();
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

    // ---- POOT-1161: the AIE schedule is the ordinary body with its grid serialized -----------------------

    fn dense_binary(op: BinOp, shape: &[usize]) -> KernelRequest {
        let operand = || ViewOperand {
            shape: shape.to_vec(),
            layout: Layout::contiguous(shape),
        };
        KernelRequest::Elementwise(ElementwiseSpec::Binary {
            op: ValueBinary::Basic(op, Ty::F32),
            out_shape: shape.to_vec(),
            a: operand(),
            b: operand(),
            fold: Fold {
                numel: shape.iter().product(),
                two_d: false,
                width: 256,
            },
        })
    }

    fn dense_unary(op: UnaryOp, numel: usize) -> KernelRequest {
        KernelRequest::Elementwise(ElementwiseSpec::Unary {
            op,
            dt: Ty::F32,
            view: None,
            fold: Fold {
                numel,
                two_d: false,
                width: 256,
            },
        })
    }

    /// Every op with a contract, as the request the planner builds for it.
    fn admitted_requests(numel: usize) -> Vec<(AieOp, KernelRequest)> {
        let binary = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Ge]
            .map(|op| (AieOp::Binary(op), dense_binary(op, &[numel])));
        let unary = [
            UnaryOp::Basic(UnOp::Neg),
            UnaryOp::Recip,
            UnaryOp::Math(MathOp::Exp),
            UnaryOp::Math(MathOp::Log),
            UnaryOp::Tanh,
            UnaryOp::Erf,
        ]
        .map(|op| (AieOp::Unary(op), dense_unary(op, numel)));
        binary.into_iter().chain(unary).collect()
    }

    fn aie_limits() -> crate::BodyLimits {
        crate::BodyLimits {
            max_instructions: NonZeroUsize::new(4096).unwrap(),
            max_locals: NonZeroUsize::new(256).unwrap(),
        }
    }

    fn f32_buffers(inputs: usize, numel: usize) -> Vec<poot_kernel_ir::interp::Buffer> {
        use poot_kernel_ir::interp::Buffer;
        let sample = |k: usize| -> Vec<f32> {
            (0..numel)
                .map(|i| 0.3 + 0.17 * ((i * (k + 3)) % 29) as f32)
                .collect()
        };
        (0..inputs)
            .map(|k| Buffer::from_f32s(&sample(k)))
            .chain([Buffer::from_f32s(&vec![f32::NAN; numel])])
            .collect()
    }

    /// Tier 1 on the production bodies: the AIE schedule of each admitted request, run as one thread, writes
    /// exactly what the ordinary GPU body writes when run over its grid. 70 elements is not a whole number of
    /// the GPU body's 64-thread workgroups, so the ordinary body's bounds guard is part of what is compared.
    ///
    /// Mutation: in `serialized_schedule` serialize over `Grid { threads: [1, 1, 1] }`; only the first element
    /// is written and the buffers differ.
    #[test]
    fn the_aie_schedule_computes_what_the_grid_body_computes() {
        use poot_kernel_ir::interp::run;
        const NUMEL: usize = 70;
        let caps = DeviceCaps::wgpu_rdna3_igpu();
        for (op, request) in admitted_requests(NUMEL) {
            let inputs = if matches!(op, AieOp::Unary(_)) { 1 } else { 2 };
            let grid_body = generate(&request, &caps, &aie_limits()).unwrap();
            let [x, ..] = grid_body.launch.grid;
            let [wg, ..] = grid_body.body.workgroup_size;
            let mut want = f32_buffers(inputs, NUMEL);
            run(&grid_body.body, [x.div_ceil(wg), 1, 1], &mut want).unwrap();

            let scheduled = serialized_schedule(&request, &aie_limits()).unwrap();
            assert_eq!(scheduled.body.workgroup_size, [1, 1, 1], "{op:?}");
            let mut got = f32_buffers(inputs, NUMEL);
            run(&scheduled.body, [1, 1, 1], &mut got).unwrap();

            let out = inputs;
            assert!(
                want[out].to_f32s().unwrap().iter().all(|v| !v.is_nan()),
                "{op:?}: the grid body leaves an element unwritten"
            );
            assert_eq!(got, want, "{op:?}");
        }
    }

    /// The schedule states its iteration space, bindings and contract, and its entry symbol is the AIE
    /// core's. Mutation: widen `accessed` past the element count or drop the `writes` flag of the output;
    /// the equalities below fail.
    #[test]
    fn the_aie_schedule_states_its_iteration_space_bindings_and_contract() {
        let binary =
            serialized_schedule(&dense_binary(BinOp::Add, &[2, 3]), &aie_limits()).unwrap();
        assert_eq!(binary.iterations, 6);
        assert_eq!(binary.body.name, AIE_ENTRY);
        assert_eq!(
            binary.contract,
            AieOp::Binary(BinOp::Add).contract().unwrap()
        );
        assert_eq!(
            binary.bindings.as_ref(),
            [false, false, true].map(|writes| SerializedBinding {
                dims: vec![2, 3],
                writes,
                accessed: 6,
            })
        );
        assert!(binary.body.verify().is_ok());

        let unary = serialized_schedule(&dense_unary(UnaryOp::Tanh, 5), &aie_limits()).unwrap();
        assert_eq!(unary.iterations, 5);
        assert_eq!(
            unary.bindings.iter().map(|b| b.writes).collect::<Vec<_>>(),
            [false, true]
        );
        assert!(
            unary
                .bindings
                .iter()
                .all(|b| b.dims == [5] && b.accessed == 5)
        );
    }

    /// An op has an AIE schedule exactly when it has a numerical contract, and an op without one is a typed
    /// refusal that names the missing lowering, never a body under another op's contract.
    /// Mutation: give `sqrt` a contract; the refusal row below turns `Ok`.
    #[test]
    fn an_op_without_a_contract_has_no_aie_schedule() {
        for (op, request) in admitted_requests(4) {
            assert!(op.contract().is_some(), "{op:?}");
            assert!(
                serialized_schedule(&request, &aie_limits()).is_ok(),
                "{op:?}"
            );
        }
        let without = [
            dense_binary(BinOp::Max, &[4]),
            dense_binary(BinOp::BitAnd, &[4]),
            dense_binary(BinOp::Rem, &[4]),
            dense_unary(UnaryOp::Math(MathOp::Sqrt), 4),
            dense_unary(UnaryOp::Math(MathOp::Round), 4),
            dense_unary(UnaryOp::Math(MathOp::Sin), 4),
        ];
        for request in without {
            let error = serialized_schedule(&request, &aie_limits()).unwrap_err();
            let KernelGenError::Unsupported { request, reason } = error else {
                panic!("expected Unsupported, got {error:?}");
            };
            assert_eq!(request, "serialized_schedule");
            assert!(reason.contains("no proved lowering"), "{reason}");
        }
    }

    /// A launch with no threads is the serializer's refusal, surfaced as its own typed variant carrying the
    /// `SerializeError`, not folded into `Unsupported`. Mutation: map the serializer's error back to
    /// `unsupported(..)` in `serialized_schedule`; the match below fails.
    #[test]
    fn a_grid_the_serializer_refuses_is_its_own_typed_error() {
        let error = serialized_schedule(&dense_unary(UnaryOp::Tanh, 0), &aie_limits()).unwrap_err();
        let KernelGenError::Serialize { generator, source } = error else {
            panic!("expected Serialize, got {error:?}");
        };
        assert_eq!(generator, "serialized_schedule");
        assert!(
            matches!(
                source,
                poot_kernel_ir::serialize::SerializeError::EmptyGrid { .. }
            ),
            "{source:?}"
        );
    }

    /// What the AIE shape ABI cannot bind is refused before any body is built: a broadcast or strided
    /// operand, a folded launch, a non-F32 element, a view, another family, and an element count the launch
    /// cannot state. Mutation: delete the operand-shape check, the stride check or the `two_d` check; the
    /// matching row is admitted.
    #[test]
    fn a_request_the_aie_abi_cannot_bind_is_refused() {
        let refused = |request: KernelRequest, needle: &str| {
            let error = serialized_schedule(&request, &aie_limits()).unwrap_err();
            match error {
                KernelGenError::Unsupported { reason, .. } => {
                    assert!(
                        reason.contains(needle),
                        "{reason} does not mention {needle}"
                    )
                }
                other => panic!("expected Unsupported, got {other:?}"),
            }
        };
        let KernelRequest::Elementwise(ElementwiseSpec::Binary {
            op,
            out_shape,
            a,
            b,
            fold,
        }) = dense_binary(BinOp::Add, &[2, 3])
        else {
            unreachable!()
        };
        let binary = |a: ViewOperand, b: ViewOperand, fold: Fold| {
            KernelRequest::Elementwise(ElementwiseSpec::Binary {
                op: op.clone(),
                out_shape: out_shape.clone(),
                a,
                b,
                fold,
            })
        };
        refused(
            binary(
                ViewOperand {
                    shape: vec![1, 3],
                    layout: Layout::contiguous(&[1, 3]),
                },
                b.clone(),
                fold,
            ),
            "broadcast",
        );
        refused(
            binary(
                ViewOperand {
                    shape: vec![2, 3],
                    layout: Layout {
                        strides: vec![1, 2],
                        offset: 0,
                    },
                },
                b.clone(),
                fold,
            ),
            "strided",
        );
        refused(
            binary(
                a.clone(),
                b.clone(),
                Fold {
                    two_d: true,
                    ..fold
                },
            ),
            "2-D folded",
        );
        let ints = KernelRequest::Elementwise(ElementwiseSpec::Binary {
            op: ValueBinary::Basic(BinOp::Add, Ty::I32),
            out_shape: vec![2, 3],
            a: a.clone(),
            b: b.clone(),
            fold,
        });
        refused(ints, "only F32");
        refused(
            KernelRequest::Elementwise(ElementwiseSpec::Unary {
                op: UnaryOp::Tanh,
                dt: Ty::F32,
                view: Some(a),
                fold,
            }),
            "only F32",
        );
        refused(
            KernelRequest::TopK(TopKSpec {
                e: 8,
                k: 2,
                numel: 4,
            }),
            "no AIE schedule",
        );
        let huge = dense_unary(UnaryOp::Tanh, u32::MAX as usize + 1);
        assert!(matches!(
            serialized_schedule(&huge, &aie_limits()),
            Err(KernelGenError::ExceedsBound { .. })
        ));
    }

    /// The finished body is measured against the limits. Mutation: drop the `Finished` reservation in
    /// `serialized_schedule`; the over-limit body comes back `Ok`.
    #[test]
    fn a_serialized_body_over_the_limits_is_refused() {
        let request = dense_binary(BinOp::Add, &[4]);
        let scheduled = serialized_schedule(&request, &aie_limits()).unwrap();
        let size = BodySize::of(&scheduled.body);
        let error =
            serialized_schedule(&request, &limits(size.instructions - 1, size.locals)).unwrap_err();
        assert!(
            matches!(
                error,
                KernelGenError::BodyLimit {
                    stage: BodyStage::Finished,
                    ..
                }
            ),
            "{error:?}"
        );
    }
}

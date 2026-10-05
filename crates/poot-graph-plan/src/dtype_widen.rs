//! Card 235: the mixed-operand-dtype decode `MatMul` planner gap.
//!
//! `plan_eqn`'s `OpKind::MatMul`/`MatMulBias` arms reject any matmul whose operand dtype does not
//! match the output dtype unless a tensor-core arm (`amd_tc` on AmdGcn/RDNA3, `tc` on Nvptx) fires;
//! both require bf16 operands with 16-aligned M/N/K and no batch dims. Decode is always a GEMV
//! (`M == 1`), never 16-aligned, so a decode-time mixed-dtype matmul can never take the tensor-core
//! path. `Runner::load` types every projection weight `DType::BF16` for any natively-bf16 safetensors
//! checkpoint (independent of `to_mixed_bf16`), so a decode matmul's weight is BF16 while the
//! activation/output stay F32, and the planner's guard panics on every backend for such a checkpoint.
//!
//! [`widen_mismatched_matmul_dtypes`] fixes this: a graph transform that inserts an explicit `Cast`
//! widening a mismatched operand back to the output dtype wherever the tensor-core arm would not fire.
//! It mirrors the mixed-bf16 narrowing cast insertion's pass in reverse and reuses the
//! eligibility predicate `plan_eqn` computes ([`matmul_tensor_core_eligible`]) so the two cannot
//! drift: a mismatched matmul is either widened by this pass or left for the tensor-core arm, so
//! `plan_eqn`'s guard should never fire on a graph this pass has run over.
//!
//! This lives in `poot-graph-plan`, not `poot-graph-ir`, because the eligibility check needs
//! [`poot_target::Backend`]/`AmdTensorCore` (arch-specific gating) and `poot-graph-ir` cannot depend on this crate
//! (`poot-graph-plan` already depends on it).
//!
//! Card 1011: a BF16 const is never retyped. Its declared dtype is its stored dtype on every backend, and
//! the lane its bytes upload to (packed `u32` words or native two-byte BF16) is decided by
//! [`bf16_const_is_packed`] and read by the storage plan and the cast planner alike. A reader with no body
//! for that lane refuses at planning; nothing widens the weight on upload.

use poot_graph_ir::graph::{Storage, ValueMeta};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{Eqn, Graph, Operand, ValidationChannel, ValueId};
use poot_target::{Backend, DECODE_GEMV_CHUNK_TRIGGER, DeviceCaps, TensorCoreSupport};
use poot_tensor::DType;

use crate::{GEMV_TILE, numel};
use std::collections::{HashMap, HashSet};

/// Whether `OpKind::MatMul`/`MatMulBias`'s AMD WMMA tensor-core arm (`plan_eqn`'s local `amd_tc`)
/// fires for the given backend, operand dtypes, and output shape. Spec 130 FR-004: RDNA3 only
/// (`TensorCoreSupport::Wmma16x16x16Rdna3`); CDNA and unknown arch fall through. `batch_dims_product` is
/// the product of every leading (non M/N) output dim - the tensor-core arm requires no batch dims.
pub fn matmul_amd_tc_eligible(
    backend: Backend,
    a_dtype: DType,
    b_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    batch_dims_product: usize,
) -> bool {
    let operands_bf16 = a_dtype == DType::BF16 && b_dtype == DType::BF16;
    matches!(backend, Backend::AmdGcn(arch) if arch.tensor_core == TensorCoreSupport::Wmma16x16x16Rdna3)
        && operands_bf16
        && m.is_multiple_of(16)
        && n.is_multiple_of(16)
        && k.is_multiple_of(16)
        && batch_dims_product == 1
}

/// Whether `OpKind::MatMul`/`MatMulBias`'s NVPTX tensor-core arm (`plan_eqn`'s local `tc`) fires for
/// the given backend, operand dtypes, and shape. Spec 025: bf16 operands with 16-aligned M/N/K (batch
/// dims are fine on NVPTX, unlike the AMD arm).
pub fn matmul_nvptx_tc_eligible(
    backend: Backend,
    a_dtype: DType,
    b_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let operands_bf16 = a_dtype == DType::BF16 && b_dtype == DType::BF16;
    backend == Backend::Nvptx
        && operands_bf16
        && m.is_multiple_of(16)
        && n.is_multiple_of(16)
        && k.is_multiple_of(16)
}

/// Whether `OpKind::MatMul`'s SPIR-V cooperative-matrix arm (`plan_eqn`'s local `coopmat`, card 154)
/// fires for the given backend, operand dtypes, and shape. RADV coopmat config 13: F16 operands (RADV
/// has no `VK_KHR_shader_bfloat16`), 16x16x16 Subgroup scope. Unlike the AMD/NVPTX arms, this gate
/// requires `k == 16` exactly (not just 16-aligned) and no batch dims: `matmul_tensorcore_coopmat`'s
/// kernel body has no K-loop, because a coopmat value is one opaque SSA register that cannot be
/// carried across a loop back-edge without a `phi` node the emitter cannot insert (see
/// `poot_kernel_ir::WmmaDtype`), so only a single K-step is implemented. M/N may be multi-tile (each
/// output tile is independent). The accumulator is always F32 (RADV has no f16-accumulate config), so
/// this only fires for a mixed f16-in/f32-out matmul; a pure f16-output matmul would need an extra
/// narrowing store that is not implemented (as in the AMD WMMA arm's F32-only path, see
/// `matmul_amd_tc_eligible`'s caller in `lib.rs`).
pub fn matmul_spirv_coopmat_eligible(
    backend: Backend,
    a_dtype: DType,
    b_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    batch_dims_product: usize,
) -> bool {
    let operands_f16 = a_dtype == DType::F16 && b_dtype == DType::F16;
    backend == Backend::SpirvVulkan
        && operands_f16
        && m.is_multiple_of(16)
        && n.is_multiple_of(16)
        && k == 16
        && batch_dims_product == 1
}

/// True if any tensor-core/coopmat arm fires, i.e. `plan_eqn`'s mixed-operand-dtype guard would not
/// reject this matmul. The single source of truth for `widen_mismatched_matmul_dtypes` and `plan_eqn`,
/// so they cannot disagree about which matmuls need widening.
pub fn matmul_tensor_core_eligible(
    backend: Backend,
    a_dtype: DType,
    b_dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    batch_dims_product: usize,
) -> bool {
    matmul_amd_tc_eligible(backend, a_dtype, b_dtype, m, n, k, batch_dims_product)
        || matmul_nvptx_tc_eligible(backend, a_dtype, b_dtype, m, n, k)
        || matmul_spirv_coopmat_eligible(backend, a_dtype, b_dtype, m, n, k, batch_dims_product)
}

/// Card 1011: whether the device buffer that holds BF16 const `id` on `backend` is the packed `u32` lanes
/// (two little-endian elements per word, [`poot_target::BufferStorage::bf16_packed`]) rather than native
/// two-byte BF16. A BF16 const is never widened: its declared dtype is its stored dtype on every backend,
/// and this decides only the lane its bytes upload to.
///
/// - `SpirvVulkan`: always packed. SPIR-V has no scalar bf16, so a packed reader is the only reader; an
///   operation with no packed reader refuses at planning (the planner's typed `DtypeLowering`).
/// - `AmdGcn`: packed when [`bf16_const_feeds_decode_bf16_gemv`] or [`bf16_const_feeds_only_packed_readers`]
///   holds; otherwise native, the lane the scalar-bf16 matmul bodies read.
/// - `Nvptx`: packed exactly when [`bf16_const_feeds_only_packed_readers`] holds; otherwise native.
///
/// The one decision the storage plan (`value_storage_kind`) and the cast planner share, so the lane a
/// `Cast` kernel reads and the lane its source uploads to cannot drift apart.
pub(crate) fn bf16_const_is_packed<V: ValidationChannel>(
    g: &Graph<V>,
    id: ValueId,
    backend: Backend,
    caps: &DeviceCaps,
) -> bool {
    match backend {
        Backend::SpirvVulkan => true,
        Backend::AmdGcn(_) => {
            bf16_const_feeds_decode_bf16_gemv(g, id, backend, caps)
                || bf16_const_feeds_only_packed_readers(g, id)
        }
        Backend::Nvptx => bf16_const_feeds_only_packed_readers(g, id),
    }
}

/// Whether the decode-GEMV body takes its BF16-weight (packed-u32, in-kernel widen) variant for this
/// matmul, the dtype-driven sibling of [`matmul_tensor_core_eligible`]. Shared by `plan_eqn`'s
/// mixed-dtype guard and GEMV body selection, [`bf16_const_is_packed`] (the weight's packed lane) and
/// [`widen_mismatched_matmul_dtypes`] (skip the widening `Cast`).
///
/// True iff:
/// - `backend` is `SpirvVulkan` or `AmdGcn` (NVPTX keeps the cast path; PTX has two-byte bf16
///   storage),
/// - the operands are the decode projection shape `F32 act x BF16 weight -> F32 out`,
/// - the shape is a decode GEMV (`M == 1`, rank-2 `[K, N]` weight, the tiled grid fits a
///   conservative real-hardware-sized bound; see the body comment),
/// - and on `SpirvVulkan` the output is at or under [`DECODE_GEMV_CHUNK_TRIGGER`], because the
///   over-trigger chunk arm is still `kg::gemv_lds` (f32 weight only).
///
/// Ten scalar/caps arguments mirror [`matmul_tensor_core_eligible`]'s shape-of-call style; splitting a
/// struct would not remove any field the three call sites (plan, storage, widen) each already have.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_decode_gemv_eligible(
    backend: Backend,
    a_dtype: DType,
    b_dtype: DType,
    odt: DType,
    m: usize,
    n: usize,
    k: usize,
    weight_rank: usize,
    out_numel: usize,
    caps: &DeviceCaps,
) -> bool {
    if !matches!(backend, Backend::SpirvVulkan | Backend::AmdGcn(_)) {
        return false;
    }
    if a_dtype != DType::F32 || b_dtype != DType::BF16 || odt != DType::F32 {
        return false;
    }
    if m != 1 || n == 0 || k == 0 || weight_rank != 2 || out_numel == 0 {
        return false;
    }
    // This is a pure graph rewrite deciding only whether the BF16-weight decode GEMV is eligible at
    // all; the real, target-aware dispatch bound is `decode_gemv_plan`'s own `caps.max_grid` split,
    // which runs later and is authoritative regardless of this guess. Reading the same measured
    // `caps.max_grid[0]` here (no baked `65_535` literal survives in planning) keeps this
    // eligibility check from diverging wildly from what `decode_gemv_plan` will actually do on this
    // device.
    let grid_cap = caps.max_grid[0] as usize;
    let nwg = out_numel.div_ceil(GEMV_TILE).max(1);
    let x = nwg.clamp(1, grid_cap);
    if nwg.div_ceil(x) > grid_cap {
        return false;
    }
    if backend == Backend::SpirvVulkan && out_numel as u64 > DECODE_GEMV_CHUNK_TRIGGER {
        return false;
    }
    true
}

/// Shape/dtype projection of [`matmul_bf16_decode_gemv_eligible`] for one eqn of `g` (the weight is
/// `inputs[1]`). `None` when the eqn is not a `MatMul`/`MatMulBias` with the shape the predicate needs.
fn matmul_bf16_decode_gemv_eligible_in<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    eqn: &Eqn,
    caps: &DeviceCaps,
) -> Option<bool> {
    if !matches!(eqn.op, OpKind::MatMul | OpKind::MatMulBias) || eqn.inputs.len() < 2 {
        return None;
    }
    let (Operand::Value(a), Operand::Value(b)) = (&eqn.inputs[0], &eqn.inputs[1]) else {
        return None;
    };
    let out_shape = &g.aval(eqn.out).shape;
    let r = out_shape.len();
    if r < 2 {
        return None;
    }
    let (m, n) = (out_shape[r - 2], out_shape[r - 1]);
    let k = g.aval(*a).shape.last().copied().unwrap_or(0);
    Some(matmul_bf16_decode_gemv_eligible(
        backend,
        g.aval(*a).dtype,
        g.aval(*b).dtype,
        g.aval(eqn.out).dtype,
        m,
        n,
        k,
        g.aval(*b).shape.len(),
        numel(out_shape),
        caps,
    ))
}

/// True if `id` is the weight (`inputs[1]`) of every consumer, and each of those consumers is a
/// decode GEMV that [`matmul_bf16_decode_gemv_eligible`] selects the packed-BF16 body for. A value
/// with no consumer is not eligible (same vacuous-truth rule as
/// [`bf16_const_feeds_only_packed_readers`]).
///
/// The decode GEMV weight is read as raw checkpoint bytes packed two-per-u32 and the bf16 GEMV body widens
/// in-register. Lives here so storage, widen and `plan_eqn` cannot disagree about which consts are
/// physically narrow.
pub fn bf16_const_feeds_decode_bf16_gemv<V: ValidationChannel>(
    g: &Graph<V>,
    id: ValueId,
    backend: Backend,
    caps: &DeviceCaps,
) -> bool {
    if g.liveness_roots().any(|root| root == id) {
        return false;
    }
    let mut consumed = false;
    for eqn in &g.eqns {
        let mut hits = 0usize;
        for operand in &eqn.inputs {
            if matches!(operand, Operand::Value(v) if *v == id) {
                hits += 1;
            }
        }
        if hits == 0 {
            continue;
        }
        consumed = true;
        // The weight must be inputs[1] and the only use of `id` in this eqn (an activation or bias
        // position would need a different upload contract).
        if hits != 1 || !matches!(eqn.inputs.get(1), Some(Operand::Value(v)) if *v == id) {
            return false;
        }
        match matmul_bf16_decode_gemv_eligible_in(g, backend, eqn, caps) {
            Some(true) => {}
            _ => return false,
        }
    }
    consumed
}

/// The operand position at which `op` reads a value as packed BF16 lanes on a typed value walk, if it
/// does. One row per imported kernel that decodes the Card 380 lane layout: the `DenseContraction`
/// weight (card 380), the `DenseRowGather` table (card 381) and the `Cast` to F32 of a norm scale, bias or
/// other elementwise const (card 1011, `packed_bf16_to_f32`). A further reader is a row here plus its
/// kernel, never a second hand-written branch.
///
/// Every other operand of those operations is an ordinary F32 or I32 buffer, so the position matters:
/// a contraction activation or a row-gather index read through this table would be a const uploaded as
/// two-byte words for a kernel that reads four-byte ones.
fn packed_bf16_reader_operand(op: &OpKind) -> Option<usize> {
    match op {
        OpKind::DenseContraction { .. } => Some(1),
        OpKind::DenseRowGather { .. } => Some(0),
        OpKind::Cast { to: DType::F32 } => Some(0),
        _ => None,
    }
}

/// Card 380 and card 381: true if every consumer of `id` reads it through
/// `packed_bf16_reader_operand`, and `id` does not escape the graph. This is the condition under
/// which a BF16 const may stay narrow on a typed value walk: those imported kernels are the only ones
/// there that read BF16, and they read it as packed `u32` lanes holding the checkpoint bytes.
///
/// A value with no consumer is not eligible: "every consumer" must not be vacuously true for a const
/// nothing reads, because the binder would then upload two-byte words no kernel can interpret.
///
/// Lives here with [`bf16_const_is_packed`] for one reason: the uploaded lane and the kernel that reads it
/// must be decided by one predicate, or a kernel silently reads a buffer at the wrong element width.
///
/// Holding here is necessary but not sufficient for a narrow const to be safe: it is only safe on a
/// lane that stores packed BF16, which is the typed value walk on each backend that has one - wgpu's,
/// and ROCm's since card 450 ROCm. Every other wgpu entry point rejects a BF16 value through
/// `crate::validate_wgpu_tensor_execution`.
pub fn bf16_const_feeds_only_packed_readers<V: ValidationChannel>(
    g: &Graph<V>,
    id: ValueId,
) -> bool {
    if g.liveness_roots().any(|root| root == id) {
        return false;
    }
    let mut consumed = false;
    for eqn in &g.eqns {
        for (position, operand) in eqn.inputs.iter().enumerate() {
            if !matches!(operand, Operand::Value(value) if *value == id) {
                continue;
            }
            consumed = true;
            if packed_bf16_reader_operand(&eqn.op) != Some(position) {
                return false;
            }
        }
    }
    consumed
}

/// Card 534a: the graph-level BF16-packed-weight recognition (`fold_dense_contractions`,
/// `fold_dense_bf16_row_gathers`) and non-last-axis reduce legalization (`lower_nonlast_reduces`)
/// every target needs before its own dtype widening, in the one order every caller ran them in before
/// this card: the folds must see the traced `MatMul`/`Transpose` and
/// `Cast`/`Gather` chains before `fuse` or the dtype widening rewrite them, and the reduce legalization
/// must run before `fuse` can absorb a non-last-axis `Reduce` into a region no target plans.
///
/// `compile`'s pipeline runs this between `dce` and `fuse`; the typed exact-value walk (Card 534b) runs
/// it standalone, because the rest of `compile`'s pipeline (fusion, tiling) would reassociate the
/// bit-exact arithmetic that walk's contract requires. One function for every caller closes the R481-003
/// drift: wgpu's copy ran `lower_nonlast_reduces`, ROCm's and PTX's did not, so a MiniMax-M2 decode graph
/// planned on wgpu and refused on ROCm and PTX for a pass gap, not a real capability difference.
pub fn prepare_target_graph<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    caps: &DeviceCaps,
) -> Graph<V> {
    let folded = fold_dense_contractions(g);
    let folded = fold_dense_bf16_row_gathers(&folded);
    let folded = lower_nonlast_reduces(&folded);
    widen_mismatched_matmul_dtypes(&folded, backend, caps)
}

/// A rewritten equation's operand map, keyed by the `ValueId` it reads: which equation index
/// produces it. Local to the two folds below (their own copy of `poot-graph-ir`'s private
/// `packed_consumer_map`/`value_input`/`is_graph_escape`, kept small rather than widening
/// `poot-graph-ir`'s public surface for two callers - Card 534a).
fn packed_consumer_map<V: ValidationChannel>(g: &Graph<V>) -> Vec<Vec<usize>> {
    let mut consumers = vec![Vec::new(); g.values.len()];
    for (equation, eqn) in g.eqns.iter().enumerate() {
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                consumers[*value].push(equation);
            }
        }
    }
    consumers
}

fn is_graph_escape<V: ValidationChannel>(g: &Graph<V>, value: ValueId) -> bool {
    g.liveness_roots().any(|root| root == value)
}

fn value_input(eqn: &Eqn, position: usize) -> Option<ValueId> {
    match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}

/// Card 380 (Card 534a: moved out of `poot-graph-ir`, whose `pub mod transform` let every crate that
/// depends on it - `poot-gpu`, `poot-rocm-gpu`, `poot-ptx-gpu` included - name and sequence this pass
/// itself, defeating SC-001; Card 534a). Rewrite `MatMul(a, Transpose([1, 0], w))`
/// over a weight of an admitted dtype into one [`poot_graph_ir::OpKind::DenseContraction`] that reads `w` in checkpoint
/// `[N, K]` order, so the binder never materializes a widened F32 copy of it, when:
///
/// - the matmul takes exactly two value operands,
/// - operand 1 is produced by `Transpose { perm: [1, 0] }`,
/// - the transposed value has exactly one consumer and is not a graph escape (a validation witness
///   makes it one, so an observed transpose is left alone),
/// - the pre-transpose weight's dtype is in [`poot_graph_ir::DENSE_CONTRACTION_WEIGHT_DTYPES`],
/// - `DenseContraction::infer` agrees with the matmul's own declared output type,
/// - and, for an F16 weight (Card 1007), the weight is a non-escaping const every consumer of which is a
///   transpose this pass folds, so its one packed storage serves every reader.
///
/// Value ids are preserved: the contraction keeps the matmul's own `out` and only the dead transpose
/// equation is dropped, so input, state, and validation bindings stay valid. Idempotent: a folded
/// graph has no `Transpose` left to match.
///
/// `pub(crate)`: reachable only through [`prepare_target_graph`] (Card 534a SC-001).
///
/// ```compile_fail,E0603
/// let g = poot_graph_ir::Graph::default();
/// let _ = poot_graph_plan::fold_dense_contractions(&g);
/// ```
pub(crate) fn fold_dense_contractions<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns.iter().enumerate() {
        producer.insert(eqn.out, index);
    }
    let mut replacements: HashMap<usize, Eqn> = HashMap::new();
    let mut removed = HashSet::new();
    let mut f16_folds = Vec::new();

    for (matmul_index, matmul) in g.eqns.iter().enumerate() {
        if !matches!(matmul.op, OpKind::MatMul) || matmul.inputs.len() != 2 {
            continue;
        }
        let (Some(activation), Some(transposed)) = (value_input(matmul, 0), value_input(matmul, 1))
        else {
            continue;
        };
        let Some(&transpose_index) = producer.get(&transposed) else {
            continue;
        };
        let OpKind::Transpose { perm } = &g.eqns[transpose_index].op else {
            continue;
        };
        if perm.as_slice() != [1, 0]
            || consumers[transposed].len() != 1
            || is_graph_escape(g, transposed)
        {
            continue;
        }
        let Some(weight) = value_input(&g.eqns[transpose_index], 0) else {
            continue;
        };
        let dtype = g.aval(weight).dtype;
        if !poot_graph_ir::DENSE_CONTRACTION_WEIGHT_DTYPES.contains(&dtype) {
            continue;
        }
        let candidate = OpKind::DenseContraction { weight: dtype };
        let candidate_inputs = [g.aval(activation).clone(), g.aval(weight).clone()];
        if candidate.infer(&candidate_inputs).ok().as_ref() != Some(g.aval(matmul.out)) {
            continue;
        }
        removed.insert(transpose_index);
        replacements.insert(
            matmul_index,
            Eqn {
                op: candidate,
                inputs: vec![Operand::Value(activation), Operand::Value(weight)],
                out: matmul.out,
                layer: matmul.layer,
            },
        );
        if dtype == DType::F16 {
            f16_folds.push((matmul_index, transpose_index, weight));
        }
    }

    // Card 1007: an F16 weight is stored packed two elements per `u32` word for the contraction to decode, and
    // a buffer has one storage. So an F16 weight folds only when the fold leaves the contraction its sole
    // reader: a const (a computed value is written natively by its producer), not an escape, and every consumer
    // a transpose folded here. Otherwise each of its compositions stays exactly as traced. The decision is per
    // weight, so dropping one weight's folds never changes another's.
    let unpackable: HashSet<ValueId> = f16_folds
        .iter()
        .map(|&(_, _, weight)| weight)
        .filter(|&weight| {
            g.meta(weight).storage != Storage::Const
                || is_graph_escape(g, weight)
                || consumers[weight]
                    .iter()
                    .any(|consumer| !removed.contains(consumer))
        })
        .collect();
    for &(matmul_index, transpose_index, weight) in &f16_folds {
        if unpackable.contains(&weight) {
            replacements.remove(&matmul_index);
            removed.remove(&transpose_index);
        }
    }

    let eqns = g
        .eqns
        .iter()
        .enumerate()
        .filter_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .or_else(|| (!removed.contains(&index)).then(|| eqn.clone()))
        })
        .collect();
    Graph { eqns, ..g.clone() }
}

/// Card 381 (Card 534a: moved out of `poot-graph-ir`, see [`fold_dense_contractions`]'s doc).
/// Fold `Cast(F32, [Reshape...], Gather { axis: 0 })` over a rank-2 narrow-float table into one
/// [`poot_graph_ir::OpKind::DenseRowGather`] that gathers rows and widens them in a single equation,
/// followed by the same reshapes over F32 values.
///
/// Embedding sibling of [`fold_dense_contractions`]: the only other ways to lower a
/// `[vocab, hidden]` BF16 token-embedding table are to widen it to F32 (multi-GB at real model dims)
/// or leave the gather's output BF16, which on wgpu is representable only as packed `u32` lanes (a
/// kernel producing packed lanes must write two elements per lane). Gathering and widening in one
/// equation avoids both.
///
/// The fold applies only when every condition holds, so it never changes what a graph means:
///
/// - the equation is `Cast { to: F32 }` over a single value operand,
/// - that operand is reached, through zero or more `Reshape` equations, from a `Gather { axis: 0 }`,
/// - every value on that chain has exactly one consumer and is not a graph escape (a validation
///   witness makes it one, so an observed intermediate is left alone),
/// - the table's dtype is in [`poot_graph_ir::DENSE_ROW_GATHER_SOURCE_DTYPES`],
/// - and `infer` agrees, at the gather and at every rebuilt reshape, with the type the folded graph
///   declares.
///
/// The `Cast`'s own value id is preserved (the last rebuilt equation writes it), so input, state and
/// validation bindings stay valid. The BF16 values the chain produced keep their `ValueMeta` entries
/// and lose their producers, as the transpose does in the contraction fold. Idempotent: a folded graph
/// holds no `Cast { to: F32 }` over a `Gather`.
///
/// `pub(crate)`: reachable only through [`prepare_target_graph`] (Card 534a SC-001).
///
/// ```compile_fail,E0603
/// let g = poot_graph_ir::Graph::default();
/// let _ = poot_graph_plan::fold_dense_bf16_row_gathers(&g);
/// ```
pub(crate) fn fold_dense_bf16_row_gathers<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns.iter().enumerate() {
        producer.insert(eqn.out, index);
    }

    let mut out = g.clone();
    let mut replacements: HashMap<usize, Vec<Eqn>> = HashMap::new();
    let mut removed = HashSet::new();

    for (cast_index, cast) in g.eqns.iter().enumerate() {
        if !matches!(cast.op, OpKind::Cast { to: DType::F32 }) || cast.inputs.len() != 1 {
            continue;
        }
        let Some(mut current) = value_input(cast, 0) else {
            continue;
        };
        // Walk back to the gather through movement that commutes with the widening. `Reshape` is the only
        // admitted link: it is a pure relabel of the same elements in the same order, so reshaping decoded
        // rows and decoding reshaped rows are the same values.
        let mut links = Vec::new();
        let chain = loop {
            let Some(&index) = producer.get(&current) else {
                break None;
            };
            if consumers[current].len() != 1 || is_graph_escape(g, current) {
                break None;
            }
            match g.eqns[index].op {
                OpKind::Gather { axis: 0 } => break Some(index),
                OpKind::Reshape { .. } => {
                    let Some(source) = value_input(&g.eqns[index], 0) else {
                        break None;
                    };
                    links.push(index);
                    current = source;
                }
                _ => break None,
            }
        };
        let Some(gather_index) = chain else {
            continue;
        };
        let gather = &g.eqns[gather_index];
        let (Some(table), Some(index)) = (value_input(gather, 0), value_input(gather, 1)) else {
            continue;
        };
        let dtype = g.aval(table).dtype;
        if !poot_graph_ir::DENSE_ROW_GATHER_SOURCE_DTYPES.contains(&dtype) {
            continue;
        }
        let op = OpKind::DenseRowGather { source: dtype };
        let Ok(gathered) = op.infer(&[g.aval(table).clone(), g.aval(index).clone()]) else {
            continue;
        };
        if gathered.shape != g.aval(gather.out).shape {
            continue;
        }

        // Rebuild the chain over F32, innermost first, with the outermost equation writing the cast's own
        // value id. A fresh value is appended for every intermediate the chain still needs.
        let values_before = out.values.len();
        let mut rebuilt = Vec::with_capacity(links.len() + 1);
        let mut inputs = vec![Operand::Value(table), Operand::Value(index)];
        let mut current_op = op;
        let mut current_type = gathered;
        let mut agreed = true;
        for &link in links.iter().rev() {
            let link_op = g.eqns[link].op.clone();
            let Ok(next_type) = link_op.infer(std::slice::from_ref(&current_type)) else {
                agreed = false;
                break;
            };
            if next_type.shape != g.aval(g.eqns[link].out).shape {
                agreed = false;
                break;
            }
            let value = out.values.len();
            out.values
                .push(ValueMeta::new(current_type, Storage::Device, None));
            rebuilt.push(Eqn {
                op: current_op,
                inputs,
                out: value,
                layer: cast.layer,
            });
            inputs = vec![Operand::Value(value)];
            current_op = link_op;
            current_type = next_type;
        }
        if !agreed || &current_type != g.aval(cast.out) {
            // Nothing is committed until every check passes: drop this candidate's appended values, and
            // leave every equation of its chain exactly as it was.
            out.values.truncate(values_before);
            continue;
        }
        rebuilt.push(Eqn {
            op: current_op,
            inputs,
            out: cast.out,
            layer: cast.layer,
        });
        removed.insert(gather_index);
        removed.extend(links);
        replacements.insert(cast_index, rebuilt);
    }

    out.eqns = g
        .eqns
        .iter()
        .enumerate()
        .filter_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .or_else(|| (!removed.contains(&index)).then(|| vec![eqn.clone()]))
        })
        .flatten()
        .collect();
    out
}

/// Card 454 D1 (Card 534a: moved out of `poot-graph-ir`, see [`fold_dense_contractions`]'s doc).
/// Rewrite every [`poot_graph_ir::OpKind::Reduce`] whose axis is not the input's last axis into a
/// [`poot_graph_ir::OpKind::Transpose`] that moves that axis to the end, a last-axis reduce, and (when
/// `keepdim` put the size-1 axis in the wrong place) a [`poot_graph_ir::OpKind::Reshape`] back to the
/// original output shape.
///
/// `plan_eqn`'s `Reduce` arm only plans a last-axis reduce (its sole body is `kg::reduce_last_dt`,
/// whose rows are contiguous), refusing any other axis with a typed `Capability::NonLastAxisReduce`.
/// This pass keeps the IR primitive - no new kernel, no hand-fused region - by expressing the same
/// reduction through ops the planner already plans. Model tracers may still emit a non-last reduce
/// naturally (GLM-5.3-Flash's sparse FFN sums its `top_k` rows on axis 0); the rewrite happens at
/// backend preparation, so the pinned model graph is untouched.
///
/// Value preservation: for each output element the last-axis body folds the same element pairs in the
/// same `0..k` order as the original axis reduce (the transpose only reorders which axis those pairs
/// sit on), so the result is bit-identical, not merely close.
///
/// Properties:
///
/// - Only rewrites when every rebuilt shape agrees with what the graph already declares (checked
///   through `OpKind::infer`); a disagreement leaves the equation untouched and the planner's own
///   typed refusal as the failure.
/// - Preserves the reduce's output value id, so input/state/validation bindings and named outputs
///   survive.
/// - Idempotent: a graph with no non-last reduce (or a lowered one) comes back unchanged.
///
/// Card 534a runs it for every backend, through [`prepare_target_graph`] (`compile`'s pipeline, and
/// the typed exact-value walk): before that card, only the wgpu copy of this preparation ran it, so a
/// MiniMax-M2 or GLM-5.3-Flash MoE gating reduce planned on wgpu and refused on ROCm and PTX
/// even though ROCm and PTX plan `Reduce` through the same `plan_eqn`, not a
/// separate planner.
///
/// `pub(crate)`: reachable only through [`prepare_target_graph`] (Card 534a SC-001).
///
/// ```compile_fail,E0603
/// let g = poot_graph_ir::Graph::default();
/// let _ = poot_graph_plan::lower_nonlast_reduces(&g);
/// ```
pub(crate) fn lower_nonlast_reduces<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let mut out = g.clone();
    let mut replacements: HashMap<usize, Vec<Eqn>> = HashMap::new();

    for (index, eqn) in g.eqns.iter().enumerate() {
        let OpKind::Reduce { op, axis, keepdim } = eqn.op else {
            continue;
        };
        let Some(Operand::Value(input)) = eqn.inputs.first().copied() else {
            continue;
        };
        let input_aval = g.aval(input);
        let rank = input_aval.shape.len();
        if axis + 1 >= rank {
            continue;
        }

        // Move the reduced axis last; the remaining axes keep their relative order, so dropping
        // the moved axis afterwards yields exactly the original output shape.
        let mut perm: Vec<usize> = (0..rank).filter(|&candidate| candidate != axis).collect();
        perm.push(axis);
        let transposed_shape: Vec<usize> = perm.iter().map(|&p| input_aval.shape[p]).collect();
        let transposed_aval = TensorType::new(transposed_shape, input_aval.dtype);
        let Ok(reduced_aval) = OpKind::Reduce {
            op,
            axis: rank - 1,
            keepdim,
        }
        .infer(std::slice::from_ref(&transposed_aval)) else {
            continue;
        };
        let Some(out_aval) = g.values.get(eqn.out).map(|meta| &meta.aval) else {
            continue;
        };

        // Nothing is committed until every check passes: the fresh intermediates are appended to
        // a scratch prefix and dropped again on any disagreement.
        let committed_before = out.values.len();
        let transposed_id = out.values.len();
        out.values
            .push(ValueMeta::new(transposed_aval, Storage::Device, None));

        let mut rebuilt = Vec::with_capacity(3);
        rebuilt.push(Eqn {
            op: OpKind::Transpose { perm },
            inputs: vec![Operand::Value(input)],
            out: transposed_id,
            layer: eqn.layer,
        });

        let agreed = if reduced_aval == *out_aval {
            // keepdim=false: dropping the moved axis leaves the original shape, so the reduce
            // writes the original output id directly.
            rebuilt.push(Eqn {
                op: OpKind::Reduce {
                    op,
                    axis: rank - 1,
                    keepdim,
                },
                inputs: vec![Operand::Value(transposed_id)],
                out: eqn.out,
                layer: eqn.layer,
            });
            true
        } else if reduced_aval.numel() == out_aval.numel() {
            // keepdim=true: the size-1 axis sits where the reduced axis was, not at the end.
            let reduced_id = out.values.len();
            out.values
                .push(ValueMeta::new(reduced_aval, Storage::Device, None));
            rebuilt.push(Eqn {
                op: OpKind::Reduce {
                    op,
                    axis: rank - 1,
                    keepdim,
                },
                inputs: vec![Operand::Value(transposed_id)],
                out: reduced_id,
                layer: eqn.layer,
            });
            rebuilt.push(Eqn {
                op: OpKind::Reshape {
                    shape: out_aval.shape.clone(),
                },
                inputs: vec![Operand::Value(reduced_id)],
                out: eqn.out,
                layer: eqn.layer,
            });
            true
        } else {
            false
        };

        if !agreed {
            out.values.truncate(committed_before);
            continue;
        }
        replacements.insert(index, rebuilt);
    }

    out.eqns = g
        .eqns
        .iter()
        .enumerate()
        .flat_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .unwrap_or_else(|| vec![eqn.clone()])
        })
        .collect();
    out
}

/// Widen a `MatMul`/`MatMulBias`/`Binary` operand whose dtype does not match the eqn's output dtype
/// back to the output dtype via an explicit `Cast`, whenever [`matmul_tensor_core_eligible`] says the
/// tensor-core arm will not fire for `backend` (a `Binary` eqn never has a tensor-core arm, so its
/// mismatches are always widened). Tensor-core-eligible matmuls (e.g. prefill matmuls narrowed by
/// `to_mixed_bf16`, always 16-aligned) are left untouched, since `matmul_tensorcore` reads bf16
/// operands directly and widening them would cost an extra cast and lose the tensor-core dispatch.
///
/// Idempotent and a no-op on any graph with no mismatched-dtype eqn. The pass only inserts `Cast`
/// equations; it never changes a value's declared dtype, and no other pass retypes a const either (Card
/// 1011). A BF16 const stays BF16 on every backend, and the `Cast{BF16 -> F32}` this pass puts in front of
/// it reads the lane the storage plan gave the const ([`bf16_const_is_packed`]).
///
/// Card 235: needed before decode capture on every backend, since decode (`M == 1`) is never
/// 16-aligned and so never takes the tensor-core path; a mismatched-dtype decode matmul is always
/// widened by this pass (except the BF16-weight decode GEMV, which reads the weight packed).
///
/// `Binary` is covered because a bf16 decode graph's RMSNorm `x * gamma` compiles to a plain
/// `Binary(Mul)` against a `Const` weight whose dtype can differ from the activation's; the operand is
/// widened to the eqn's output dtype by an explicit `Cast`.
///
/// `Gather { axis: 0 }` is also covered (first operand only: the table, never the index). A token-embedding
/// lookup is `gather(embed_const, token)`, a raw `Storage::Const` table operand. `plan_eqn`'s `Gather` arm
/// (unlike `MatMul`'s) has no operand/output dtype-mismatch guard: it picks the kernel's element type from the
/// output dtype (`kg::gather_axis0_dt(_, fty(odt), _)`), so a Const whose dtype differs from the output's,
/// read through that kernel, is silent corruption, not a `plan_eqn` rejection (`bf16 GRAPH MISMATCH (8/8)` in
/// `probe_bf16_graph`).
///
/// `Slice` is covered the same way (evidence: `ptx_prefill_check_bf16`'s per-eqn scan): RoPE's
/// `cos`/`sin` tables are `TensorType::f32(...)` unconditionally at trace (like every norm gamma and
/// bias; see `Qwen2Config::proj_dtype`), and `rope_prefill`'s `cos[0..n]`/`sin[0..n]` is a direct
/// `Slice` of that raw Const. `plan_eqn`'s `Slice` arm has the same no-guard shape as `Gather`'s (a
/// zero-cost `Plan::View` or an imported kernel, both keyed off `fty(odt)` alone), with the same
/// silent corruption (`slice ax=0 0..5 ... maxabs=2.59e38 nonfinite=20/320`). The non-axis-0 general
/// `Gather` is out of scope (no known real trace uses a raw Const as its data operand there).
/// `Unary`/`Cast`/`Reduce`/`Transpose`/`Reshape`/`Concat` etc. are not covered: no known real trace
/// feeds them a raw Const directly (always through a Gather/Slice/compute op first); add them the same
/// way if one does.
pub fn widen_mismatched_matmul_dtypes<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    caps: &DeviceCaps,
) -> Graph<V> {
    // Card 273: widen a single-value-operand movement op (`Gather{axis:0}`, `Slice`) whose one operand
    // is directly a mismatched-dtype `Storage::Const`: inserts a Cast on that operand only, so the
    // eqn's output dtype and every downstream consumer are untouched. Both ops can plan to a zero-cost
    // `Plan::View`/imported-kernel path that reads the operand's real device bytes at the eqn's output
    // elem width (`fty(odt)`) with no mismatch guard, so a Const read at an element width its dtype does not
    // have is silent corruption (`probe_bf16_graph` for Gather, `ptx_prefill_check_bf16` for
    // Slice on `rope.cos`/`rope.sin`, both `TensorType::f32(...)` at trace regardless of `proj_dtype`).
    let widen_single_operand = |out: &mut Graph<V>, eqn: &Eqn| -> bool {
        let table_id = match eqn.inputs.first() {
            Some(Operand::Value(v)) => *v,
            _ => return false,
        };
        let odt = g.aval(eqn.out).dtype;
        if g.aval(table_id).dtype == odt {
            return false;
        }
        let aval = TensorType::new(out.aval(table_id).shape.clone(), odt);
        let nv = out.values.len();
        out.values.push(ValueMeta::new(aval, Storage::Device, None));
        out.eqns.push(Eqn {
            op: OpKind::Cast { to: odt },
            inputs: vec![Operand::Value(table_id)],
            out: nv,
            layer: eqn.layer,
        });
        let mut inputs = eqn.inputs.clone();
        inputs[0] = Operand::Value(nv);
        out.eqns.push(Eqn {
            op: eqn.op.clone(),
            inputs,
            out: eqn.out,
            layer: eqn.layer,
        });
        true
    };

    let mut out = g.clone();
    out.eqns.clear();
    for eqn in &g.eqns {
        if matches!(eqn.op, OpKind::Gather { axis: 0 } | OpKind::Slice { .. }) {
            if !widen_single_operand(&mut out, eqn) {
                out.eqns.push(eqn.clone());
            }
            continue;
        }
        let is_matmul = matches!(eqn.op, OpKind::MatMul);
        let is_matmul_bias = matches!(eqn.op, OpKind::MatMulBias);
        let mm = is_matmul || is_matmul_bias;
        let widenable = mm || matches!(eqn.op, OpKind::Binary(_));
        let (a_id, b_id) = match (eqn.inputs.first(), eqn.inputs.get(1)) {
            (Some(Operand::Value(a)), Some(Operand::Value(b))) if widenable => (*a, *b),
            _ => {
                out.eqns.push(eqn.clone());
                continue;
            }
        };
        // Card 273: `MatMulBias`'s bias operand (`eqn.inputs[2]`) is deliberately not widened, even
        // though its declared dtype (F32 after the PTX Const safety transform) never matches a BF16
        // `odt`. `matmul_batched_impl`'s bias kernel parameter is hardcoded `slice_f32` regardless of
        // `dt` (`crates/poot-kernelgen/src/matmul.rs`: bias is an `[N]` f32 vector, unlike `a`/`b`/`c`,
        // which use `slice_dtype(dt, ..)`), so the kernel's contract is "bias is always f32". Widening
        // bias to `odt` produced a 2-byte bf16 bias buffer that the kernel's `slice_f32` param read at
        // 2x the real width (`reldiff` 1.1 at the first q-proj `matmul_bias` in
        // `ptx_prefill_check_bf16`, per `ptx_prefill_compare_bf16`'s CPU-oracle diff). The PTX Const
        // safety transform already leaves the bias Const F32-declared, matching that contract, so only
        // `a`/`b` (which the kernel narrows to `dt`) are widened below. `MatMulBias` never takes a
        // tensor-core arm (the tensor-core path does not carry bias, so it always uses the serial
        // batched kernel), so the `matmul_tensor_core_eligible` skip below applies only to plain
        // `MatMul`.
        let odt = g.aval(eqn.out).dtype;
        let a_dt = g.aval(a_id).dtype;
        let b_dt = g.aval(b_id).dtype;
        if a_dt == odt && b_dt == odt {
            out.eqns.push(eqn.clone());
            continue;
        }
        if is_matmul || is_matmul_bias {
            let out_shape = &g.aval(eqn.out).shape;
            let r = out_shape.len();
            if r < 2 {
                // MatMul/MatMulBias always infer a rank >= 2 output (see OpKind::infer); defensive only.
                out.eqns.push(eqn.clone());
                continue;
            }
            let (m, n) = (out_shape[r - 2], out_shape[r - 1]);
            let k = g.aval(a_id).shape.last().copied().unwrap_or(0);
            let batch = out_shape[..r - 2].iter().product::<usize>();
            if is_matmul && matmul_tensor_core_eligible(backend, a_dt, b_dt, m, n, k, batch) {
                out.eqns.push(eqn.clone());
                continue;
            }
            // The decode bf16 GEMV reads the weight as packed u32 lanes and widens in-register; a
            // widening Cast would materialize an F32 copy and defeat the residency. Same skip shape
            // as the tensor-core arm above (eligibility is the single source of truth with plan_eqn).
            if matmul_bf16_decode_gemv_eligible(
                backend,
                a_dt,
                b_dt,
                g.aval(eqn.out).dtype,
                m,
                n,
                k,
                g.aval(b_id).shape.len(),
                numel(out_shape),
                caps,
            ) {
                out.eqns.push(eqn.clone());
                continue;
            }
        }
        let cast_to = |out: &mut Graph<V>, id: ValueId, dt: DType| -> ValueId {
            if out.aval(id).dtype == dt {
                return id;
            }
            let aval = TensorType::new(out.aval(id).shape.clone(), dt);
            let nv = out.values.len();
            out.values.push(ValueMeta::new(aval, Storage::Device, None));
            out.eqns.push(Eqn {
                op: OpKind::Cast { to: dt },
                inputs: vec![Operand::Value(id)],
                out: nv,
                layer: eqn.layer,
            });
            nv
        };
        let mut inputs = eqn.inputs.clone();
        let na = cast_to(&mut out, a_id, odt);
        inputs[0] = Operand::Value(na);
        let nb = cast_to(&mut out, b_id, odt);
        inputs[1] = Operand::Value(nb);
        out.eqns.push(Eqn {
            op: eqn.op.clone(),
            inputs,
            out: eqn.out,
            layer: eqn.layer,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Builder, Graph, OpKind, Storage, TensorType};
    use poot_target::AmdArch;

    use poot_target::Backend;
    use poot_tensor::DType;

    use super::prepare_target_graph;

    /// `a x w -> f32` over two consts of the given dtypes, `m x 32 x 32`.
    fn const_matmul(m: usize, a_dtype: DType, w_dtype: DType) -> Graph {
        let b = Builder::new();
        let a = b.constant("a", TensorType::new(vec![m, 32], a_dtype));
        let w = b.constant("w", TensorType::new(vec![32, 32], w_dtype));
        let mm = b.matmul(a, w);
        let mut g = b.finish(mm);
        let out = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .expect("matmul eqn")
            .out;
        g.values[out].aval.dtype = DType::F32;
        g
    }

    /// Card 1011: no pass retypes a BF16 const, on any backend or AMD arch family, over the mixed-bf16 and
    /// checkpoint matmuls (a prefill matmul no backend has a packed reader for included): the declared dtype
    /// of every const is the dtype it was traced with, so the executor uploads the stored words.
    ///
    /// Mutation: retype every BF16 const to F32 at the top of `widen_mismatched_matmul_dtypes`; every row
    /// of this test goes red naming the const.
    #[test]
    fn prepare_target_graph_never_retypes_a_bf16_const() {
        let backends = [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::AmdGcn(AmdArch::new("gfx90a", 64)),
            Backend::AmdGcn(AmdArch::new("gfx1030", 32)),
            Backend::AmdGcn(AmdArch::new("gfx1200", 32)),
        ];
        let graphs = [
            (
                "mixed bf16 prefill",
                const_matmul(32, DType::BF16, DType::BF16),
            ),
            (
                "mixed bf16 decode",
                const_matmul(1, DType::BF16, DType::BF16),
            ),
            (
                "bf16 checkpoint prefill",
                const_matmul(32, DType::F32, DType::BF16),
            ),
            (
                "bf16 checkpoint decode",
                const_matmul(1, DType::F32, DType::BF16),
            ),
        ];
        for backend in backends {
            for (label, g) in &graphs {
                let caps = poot_test_util::device_caps::default_caps_for(backend);
                let prepared = prepare_target_graph(g, backend, &caps);
                for id in (0..g.values.len()).filter(|&id| g.meta(id).storage == Storage::Const) {
                    assert_eq!(
                        prepared.aval(id).dtype,
                        g.aval(id).dtype,
                        "{label} on {backend:?}: const v{id} was retyped"
                    );
                }
            }
        }
    }
}

/// Card 278, moved from `poot-ptx-gpu` in Card 534a: the tensor-core choice keeps a BF16 matmul weight
/// narrow and inserts no widening cast.
#[cfg(test)]
mod tensor_core_weight_tests {
    use poot_graph_ir::{Builder, Slot, TensorType};
    use poot_target::Backend;
    use poot_tensor::DType;

    use super::widen_mismatched_matmul_dtypes;
    use crate::{ExactI32StorageAnalysis, Plan};

    #[test]
    fn card278_existing_tensor_core_choice_keeps_eligible_bf16_weight_narrow() {
        let b = Builder::new();
        let x = b.slot(Slot::TokenEmbed, TensorType::new(vec![16, 16], DType::BF16));
        let weight = b.constant("weight", TensorType::new(vec![16, 16], DType::BF16));
        let out = b.matmul(x, weight);
        let g = b.finish(out);

        let caps = poot_test_util::device_caps::default_caps_for(Backend::Nvptx);
        let widened = widen_mismatched_matmul_dtypes(&g, Backend::Nvptx, &caps);
        assert_eq!(widened.aval(weight.id).dtype, DType::BF16);
        assert!(
            !widened
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Cast { .. }))
        );
        let eqn = widened
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::MatMul))
            .expect("matmul remains");
        match crate::plan_eqn_choice_analyzed(
            &ExactI32StorageAnalysis::new(&widened),
            &widened,
            eqn,
            Backend::Nvptx,
            1,
            &std::collections::HashMap::new(),
            &poot_test_util::device_caps::default_caps_for(Backend::Nvptx),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .expect("eligible BF16 matmul plans")
        {
            crate::Planned {
                plan: Plan::Compute { .. },
                choice:
                    crate::KernelChoice::Generated(crate::KernelRequest::Contraction(
                        crate::ContractionSpec::TensorCore { .. },
                    )),
            } => {}
            other => panic!("eligible BF16 matmul chose {:?}", other.choice),
        }
    }
}

/// Card 534a SC-002: `prepare_target_graph` is the one preparation function feeding every target
/// (R481-003's stated fix), so a MiniMax-M2-shaped MoE gating reduce over axis 0 - which the wgpu-only
/// `lower_nonlast_reduces` copy used to leave unlowered for ROCm and PTX - now legalizes identically on
/// every backend.
#[cfg(test)]
mod prepare_target_graph_tests {
    use poot_graph_ir::{Builder, OpKind, RedOp, TensorType};
    use poot_target::AmdArch;
    use poot_target::Backend;
    use poot_tensor::DType;

    use super::prepare_target_graph;

    /// A tiny MoE-router-shaped graph: `top_k`-style expert weights summed on axis 0 (a non-last-axis
    /// reduce), the same shape MiniMax-M2's gating sum traces.
    fn moe_gate_reduce_graph() -> poot_graph_ir::Graph {
        let b = Builder::new();
        let weights = b.constant("expert_weights", TensorType::f32(vec![4, 3]));
        let out = b.reduce(RedOp::Sum, weights, 0, false);
        b.finish(out)
    }

    #[test]
    fn lowers_the_non_last_axis_reduce_identically_on_every_backend() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            let prepared = prepare_target_graph(&moe_gate_reduce_graph(), backend, &caps);
            assert!(
                !prepared
                    .eqns
                    .iter()
                    .any(|eqn| matches!(eqn.op, OpKind::Reduce { axis: 0, .. })),
                "{backend:?}: the axis-0 reduce must be lowered away, not left for the planner to refuse"
            );
            assert!(
                prepared
                    .eqns
                    .iter()
                    .any(|eqn| matches!(eqn.op, OpKind::Transpose { .. })),
                "{backend:?}: `lower_nonlast_reduces` moves the reduced axis to the end with a transpose"
            );
        }
    }

    /// Card 1011: a prefill matmul (`m = 4`, not the decode GEMV) over a BF16 weight no backend folds into a
    /// packed contraction keeps the weight BF16. Nothing retypes it to F32: the widening is the planned
    /// `Cast{BF16 -> F32}` equation in front of the matmul, which a packed or native cast body reads.
    #[test]
    fn a_bf16_weight_stays_bf16_and_is_cast_in_the_graph() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![4, 8], DType::F32));
        let weight = b.constant("weight", TensorType::new(vec![8, 16], DType::BF16));
        let out = b.matmul(x, weight);
        let g = b.finish(out);

        for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            let prepared = prepare_target_graph(&g, backend, &caps);
            assert_eq!(prepared.aval(weight.id).dtype, DType::BF16, "{backend:?}");
            assert!(
                prepared.eqns.iter().any(|eqn| matches!(
                    eqn.op,
                    OpKind::Cast { to: DType::F32 }
                ) && matches!(eqn.inputs.as_slice(), [poot_graph_ir::Operand::Value(v)] if *v == weight.id)),
                "{backend:?}: the weight reaches the matmul through an explicit Cast"
            );
        }
    }
}

/// Cards 380/381, moved from `poot-graph-ir` in Card 534a: `fold_dense_contractions`
/// and `fold_dense_bf16_row_gathers` are `pub(crate)` here now, so their own unit tests moved with them
/// (`poot-graph-ir` can no longer see them either).
#[cfg(test)]
mod fold_dense_bf16_tests {
    use poot_graph_ir::graph::{Eqn, Operand, Storage, ValueMeta};
    use poot_graph_ir::{Builder, Graph, OpKind, Slot, TensorType, ValueId};
    use poot_tensor::DType;

    use super::{fold_dense_bf16_row_gathers, fold_dense_contractions};

    /// The Qwen3.8-27B LM head shape: a BF16 `[N, K]` checkpoint constant, transposed in the graph,
    /// then used as a matmul weight against an F32 activation.
    fn lm_head_shaped_graph(weight_dtype: DType) -> Graph {
        let b = Builder::new();
        let x = b.constant("activation", TensorType::f32(vec![1, 2, 5]));
        let weight = b.constant("lm_head.weight", TensorType::new(vec![7, 5], weight_dtype));
        let logits = poot_graph_ir::ops::linear(&b, x, b.transpose(weight, vec![1, 0]), None);
        b.finish(logits)
    }

    fn const_id(g: &Graph, name: &str) -> ValueId {
        g.values
            .iter()
            .position(|value| value.name.as_deref() == Some(name))
            .expect("named constant")
    }

    /// `Eqn` and `Operand` carry no `PartialEq`, so compare equation lists structurally: the op's name,
    /// its value operands in order, and its output.
    fn eqn_summary(g: &Graph) -> Vec<(String, Vec<ValueId>, ValueId)> {
        g.eqns
            .iter()
            .map(|eqn| {
                let inputs = eqn
                    .inputs
                    .iter()
                    .filter_map(|operand| match operand {
                        Operand::Value(value) => Some(*value),
                        Operand::Lit(_) => None,
                    })
                    .collect();
                (eqn.op.name(), inputs, eqn.out)
            })
            .collect()
    }

    /// Card 380 FR-004. The fold replaces the matmul and drops the transpose, keeps the matmul's own output
    /// value id and type, and points the contraction at the constant in its checkpoint order.
    #[test]
    fn fold_matches_only_the_transposed_bf16_weight_shape() {
        let graph = lm_head_shaped_graph(DType::BF16);
        let folded = fold_dense_contractions(&graph);

        assert_eq!(folded.output, graph.output);
        assert_eq!(folded.aval(folded.output), graph.aval(graph.output));
        assert!(
            !folded
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Transpose { .. })),
            "the transpose is the copy this fold exists to remove"
        );
        let index = folded
            .eqns
            .iter()
            .position(|eqn| matches!(eqn.op, OpKind::DenseContraction { .. }))
            .expect("the LM head folds");
        assert_eq!(
            folded.eqns[index].op,
            OpKind::DenseContraction {
                weight: DType::BF16
            }
        );
        assert_eq!(
            eqn_summary(&folded)[index],
            (
                folded.eqns[index].op.name(),
                vec![
                    const_id(&graph, "activation"),
                    const_id(&graph, "lm_head.weight")
                ],
                graph.output
            ),
            "the contraction reads the activation and the constant in checkpoint order"
        );

        // Idempotent: a folded graph has no transpose left to match.
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&folded)),
            eqn_summary(&folded),
            "the fold must be idempotent"
        );

        // An F32 weight is in the table (Card 645): it folds the same way, into an F32 contraction.
        let f32_weight = fold_dense_contractions(&lm_head_shaped_graph(DType::F32));
        assert!(
            f32_weight
                .eqns
                .iter()
                .any(|eqn| eqn.op == OpKind::DenseContraction { weight: DType::F32 }),
            "an F32 checkpoint weight folds"
        );
        assert!(
            !f32_weight
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
        );

        // An F16 const weight is in the table (Card 1007) and the contraction is its sole reader: it folds.
        let f16_weight = fold_dense_contractions(&lm_head_shaped_graph(DType::F16));
        assert!(
            f16_weight
                .eqns
                .iter()
                .any(|eqn| eqn.op == OpKind::DenseContraction { weight: DType::F16 }),
            "an F16 checkpoint weight folds"
        );
        // An F16 weight that is not a const cannot be stored packed (its producer writes it natively), so the
        // composition is left exactly as traced.
        let computed_f16 = {
            let b = Builder::new();
            let x = b.constant("activation", TensorType::f32(vec![1, 2, 5]));
            let weight = b.slot(Slot::Activation, TensorType::new(vec![7, 5], DType::F16));
            let logits = poot_graph_ir::ops::linear(&b, x, b.transpose(weight, vec![1, 0]), None);
            b.finish(logits)
        };
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&computed_f16)),
            eqn_summary(&computed_f16)
        );

        // The transposed value escapes the graph, so something outside reads it and the fold cannot
        // remove the equation that produces it. Its consumer count is still exactly one.
        let mut escaping = graph.clone();
        let transposed = escaping
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
            .expect("the traced graph transposes")
            .out;
        escaping.output = transposed;
        assert!(escaping.liveness_roots().any(|root| root == transposed));
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&escaping)),
            eqn_summary(&escaping),
            "an escaping transpose must not fold"
        );

        // `infer` must agree with the matmul's OWN declared output type. If it does not, the composition is
        // not the one this operation means, whatever it looks like structurally.
        let mut disagreeing = graph.clone();
        disagreeing.values[graph.output].aval = TensorType::f32(vec![1, 2, 9]);
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&disagreeing)),
            eqn_summary(&disagreeing),
            "a matmul whose declared output disagrees with infer must not fold"
        );

        // A non-swapping permutation reads the weight in a different order, so it must not fold.
        let mut wrong_perm = graph.clone();
        let transpose = wrong_perm
            .eqns
            .iter_mut()
            .find(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
            .expect("the traced graph transposes");
        transpose.op = OpKind::Transpose { perm: vec![0, 1] };
        let out = transpose.out;
        wrong_perm.values[out].aval = TensorType::new(vec![7, 5], DType::BF16);
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&wrong_perm)),
            eqn_summary(&wrong_perm)
        );
    }

    /// Card 645: the F32 contraction the fold makes is the matmul it replaced. `eval` of the folded graph
    /// (the oracle's `DenseContraction` arm) equals `eval` of the traced `Transpose` + `MatMul`, element for
    /// element, on a weight whose `[N, K]` and `[K, N]` readings differ.
    #[test]
    fn folded_f32_contraction_evaluates_as_the_traced_matmul() {
        use poot_eval::{EvalBudget, EvalOptions, Value, eval};
        use poot_tensor::HostTensor;
        use std::collections::HashMap;

        let graph = lm_head_shaped_graph(DType::F32);
        let folded = fold_dense_contractions(&graph);
        assert!(
            folded
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::DenseContraction { .. })),
            "the fixture folds"
        );
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + seed * 13) % 11) as f32 - 5.0)
                .collect()
        };
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        inputs.insert(
            const_id(&graph, "activation"),
            Value::from(HostTensor::f32(vec![1, 2, 5], fill(10, 1))),
        );
        inputs.insert(
            const_id(&graph, "lm_head.weight"),
            Value::from(HostTensor::f32(vec![7, 5], fill(35, 2))),
        );
        let run = |g: &Graph| {
            eval(g, &inputs, EvalOptions::new(EvalBudget::UNBOUNDED))
                .unwrap()
                .output
                .into_host()
                .unwrap()
        };
        let (traced, contracted) = (run(&graph), run(&folded));
        assert_eq!(contracted.shape(), traced.shape());
        assert_eq!(contracted.as_f32().unwrap(), traced.as_f32().unwrap());
    }

    /// Card 380 FR-005. A transposed value something else also reads is a real materialization: folding it
    /// away would leave the other consumer reading a value the graph no longer produces.
    #[test]
    fn fold_declines_a_transpose_with_a_second_consumer() {
        let mut graph = lm_head_shaped_graph(DType::BF16);
        let transposed = graph
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
            .expect("the traced graph transposes")
            .out;
        let widened = graph.values.len();
        graph.values.push(ValueMeta::new(
            TensorType::f32(vec![5, 7]),
            Storage::Device,
            None,
        ));
        graph.eqns.push(Eqn {
            op: OpKind::Cast { to: DType::F32 },
            inputs: vec![Operand::Value(transposed)],
            out: widened,
            layer: None,
        });
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&graph)),
            eqn_summary(&graph)
        );
    }

    /// The Qwen3.8-27B token-embedding shape: a BF16 `[V, R]` checkpoint constant gathered along axis 0 by
    /// an I32 token id, reshaped, then widened to F32.
    fn embedding_shaped_graph(table_dtype: DType) -> Graph {
        let b = Builder::new();
        let table = b.constant(
            "model.language_model.embed_tokens.weight",
            TensorType::new(vec![7, 5], table_dtype),
        );
        let token = b.constant("token", TensorType::new(vec![3], DType::I32));
        let rows = b.gather(table, 0, token);
        let shaped = b.reshape(rows, vec![1, 3, 5]);
        let widened = b.cast(shaped, DType::F32);
        b.finish(widened)
    }

    fn eqn_index(g: &Graph, matches: impl Fn(&Eqn) -> bool) -> usize {
        g.eqns.iter().position(matches).expect("equation")
    }

    /// Card 381 FR-004. The fold replaces the widening cast with one row gather over the checkpoint
    /// constant, drops the gather and the reshape it walked through, and keeps the cast's own output id and
    /// type.
    #[test]
    fn fold_matches_only_a_widened_axis0_gather() {
        let graph = embedding_shaped_graph(DType::BF16);
        let folded = fold_dense_bf16_row_gathers(&graph);

        assert_eq!(folded.output, graph.output);
        assert_eq!(folded.aval(folded.output), graph.aval(graph.output));
        assert!(
            !folded
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Gather { .. })),
            "the BF16 gather is what this fold exists to remove"
        );
        assert!(
            !folded
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Cast { .. })),
            "the widening cast is folded into the row gather"
        );
        let index = eqn_index(&folded, |eqn| {
            matches!(eqn.op, OpKind::DenseRowGather { .. })
        });
        assert_eq!(
            folded.eqns[index].op,
            OpKind::DenseRowGather {
                source: DType::BF16
            }
        );
        assert_eq!(
            eqn_summary(&folded)[index],
            (
                folded.eqns[index].op.name(),
                vec![
                    const_id(&graph, "model.language_model.embed_tokens.weight"),
                    const_id(&graph, "token")
                ],
                folded.eqns[index].out
            ),
            "the row gather reads the checkpoint constant and the token id directly"
        );

        // Idempotent: a folded graph has no widened gather left to match.
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&folded)),
            eqn_summary(&folded),
            "the fold must be idempotent"
        );

        // An F32 table is not in the admitted table, so the composition is left exactly as traced.
        let f32_table = embedding_shaped_graph(DType::F32);
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&f32_table)),
            eqn_summary(&f32_table)
        );

        // A non-axis-0 gather selects along a different axis, so it is a different equation whatever the
        // declared shapes say.
        let mut other_axis = graph.clone();
        let gather = eqn_index(&graph, |eqn| matches!(eqn.op, OpKind::Gather { .. }));
        other_axis.eqns[gather].op = OpKind::Gather { axis: 1 };
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&other_axis)),
            eqn_summary(&other_axis),
            "an axis-1 gather must not fold"
        );

        // `infer` must agree with the cast's OWN declared output type. If it does not, the composition is
        // not the one this operation means, whatever it looks like structurally.
        let mut disagreeing = graph.clone();
        disagreeing.values[graph.output].aval = TensorType::f32(vec![1, 3, 9]);
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&disagreeing)),
            eqn_summary(&disagreeing),
            "a cast whose declared output disagrees with infer must not fold"
        );
    }

    /// Card 381 FR-004. The reshape the fold walks through is rebuilt over F32 values, so the cast's own id
    /// keeps its declared `[1, 3, 5]` shape and the row gather writes the `[3, 5]` block underneath it.
    #[test]
    fn fold_rebuilds_the_reshape_chain_over_f32() {
        let graph = embedding_shaped_graph(DType::BF16);
        let folded = fold_dense_bf16_row_gathers(&graph);

        let gather = eqn_index(&folded, |eqn| {
            matches!(eqn.op, OpKind::DenseRowGather { .. })
        });
        let reshape = eqn_index(&folded, |eqn| matches!(eqn.op, OpKind::Reshape { .. }));
        assert_eq!(
            folded.aval(folded.eqns[gather].out),
            &TensorType::f32(vec![3, 5]),
            "the row gather writes decoded F32 rows, not packed BF16"
        );
        assert_eq!(
            eqn_summary(&folded)[reshape],
            (
                folded.eqns[reshape].op.name(),
                vec![folded.eqns[gather].out],
                graph.output
            ),
            "the rebuilt reshape reads the gathered rows and writes the cast's own id"
        );
        assert_eq!(
            folded.eqns[reshape].op,
            OpKind::Reshape {
                shape: vec![1, 3, 5]
            }
        );
    }

    /// Card 381 FR-005. A gathered value something else also reads is a real materialization: folding it
    /// away would leave the other consumer reading a value the graph no longer produces.
    #[test]
    fn fold_declines_a_gather_with_a_second_consumer() {
        let mut graph = embedding_shaped_graph(DType::BF16);
        let rows = graph.eqns[eqn_index(&graph, |eqn| matches!(eqn.op, OpKind::Gather { .. }))].out;
        let second = graph.values.len();
        graph.values.push(ValueMeta::new(
            TensorType::f32(vec![3, 5]),
            Storage::Device,
            None,
        ));
        graph.eqns.push(Eqn {
            op: OpKind::Cast { to: DType::F32 },
            inputs: vec![Operand::Value(rows)],
            out: second,
            layer: None,
        });
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&graph)),
            eqn_summary(&graph)
        );
    }

    /// Card 381 FR-005. Something outside the graph reads the gathered rows, so the equation that produces
    /// them cannot be removed. Its consumer count is still exactly one.
    #[test]
    fn fold_declines_an_escaping_gather() {
        let mut graph = embedding_shaped_graph(DType::BF16);
        let rows = graph.eqns[eqn_index(&graph, |eqn| matches!(eqn.op, OpKind::Gather { .. }))].out;
        graph.output = rows;
        assert!(graph.liveness_roots().any(|root| root == rows));
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&graph)),
            eqn_summary(&graph),
            "an escaping gather must not fold"
        );
    }

    /// Card 381 FR-005. `Reshape` is the only admitted link. A transpose between the gather and the cast
    /// would be correct to fold through - it relabels the same elements - but no production graph has one,
    /// and the walk admits exactly what production graphs contain.
    #[test]
    fn fold_declines_a_non_reshape_link() {
        let b = Builder::new();
        let table = b.constant(
            "model.language_model.embed_tokens.weight",
            TensorType::new(vec![7, 5], DType::BF16),
        );
        let token = b.constant("token", TensorType::new(vec![3], DType::I32));
        let rows = b.gather(table, 0, token);
        let moved = b.transpose(rows, vec![1, 0]);
        let widened = b.cast(moved, DType::F32);
        let graph = b.finish(widened);
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&graph)),
            eqn_summary(&graph)
        );
    }
}

/// Card 454 D1, moved from `poot-graph-ir` in Card 534a: `lower_nonlast_reduces` is
/// `pub(crate)` here now, so its own unit test moved with it.
#[cfg(test)]
mod lower_nonlast_reduces_tests {
    use poot_graph_ir::graph::Operand;
    use poot_graph_ir::{Builder, OpKind, RedOp, TensorType};

    use super::lower_nonlast_reduces;

    /// Card 454 D1: a reduce whose axis is not last (GLM-5.3-Flash's sparse FFN sums its `top_k`
    /// rows on axis 0 of `[top_k, hidden]`) becomes `Transpose` + a last-axis `Reduce`, keeping the
    /// original output value id and declared shape, and a `keepdim` reduce gets the `Reshape` back
    /// to its declared `[1, hidden]`. A last-axis reduce is left alone, and lowering is idempotent.
    #[test]
    fn lower_nonlast_reduces_moves_a_non_last_reduce_to_the_last_axis() {
        // keepdim = false: [2, 4] --sum axis 0--> [4].
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 4]));
        let reduced = b.reduce(RedOp::Sum, x, 0, false);
        let graph = b.finish(reduced);
        assert_eq!(graph.aval(graph.output).shape, vec![4]);

        let lowered = lower_nonlast_reduces(&graph);
        assert_eq!(lowered.eqns.len(), 2, "transpose + last-axis reduce");
        let OpKind::Transpose { perm } = &lowered.eqns[0].op else {
            panic!(
                "first lowered equation must be the transpose, got {:?}",
                lowered.eqns[0].op
            );
        };
        assert_eq!(perm, &vec![1, 0], "axis 0 moves last");
        let OpKind::Reduce { op, axis, keepdim } = &lowered.eqns[1].op else {
            panic!(
                "second lowered equation must be the reduce, got {:?}",
                lowered.eqns[1].op
            );
        };
        assert_eq!((*op, *axis, *keepdim), (RedOp::Sum, 1, false));
        assert!(
            matches!(&lowered.eqns[1].inputs[..], [Operand::Value(id)] if *id == lowered.eqns[0].out),
            "the reduce reads the transposed value"
        );
        assert_eq!(
            lowered.eqns[1].out, graph.output,
            "the reduce keeps the original output value id"
        );
        assert_eq!(lowered.aval(lowered.eqns[1].out).shape, vec![4]);
        lowered.validate().expect("lowered graph stays valid");

        // Idempotent: nothing non-last is left to rewrite.
        let again = lower_nonlast_reduces(&lowered);
        assert_eq!(again.eqns.len(), lowered.eqns.len());
        assert!(
            again
                .eqns
                .iter()
                .zip(&lowered.eqns)
                .all(|(a, b)| a.op == b.op && a.out == b.out),
            "a lowered graph must come back unchanged"
        );

        // A last-axis reduce is never rewritten.
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 4]));
        let reduced = b.reduce(RedOp::Sum, x, 1, false);
        let last_axis = b.finish(reduced);
        assert_eq!(
            lower_nonlast_reduces(&last_axis).eqns.len(),
            1,
            "a last-axis reduce has nothing to lower"
        );

        // keepdim = true: the size-1 axis stays where the reduced axis was, so a Reshape returns it.
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 4]));
        let reduced = b.reduce(RedOp::Max, x, 0, true);
        let graph = b.finish(reduced);
        assert_eq!(graph.aval(graph.output).shape, vec![1, 4]);

        let lowered = lower_nonlast_reduces(&graph);
        assert_eq!(
            lowered.eqns.len(),
            3,
            "transpose + keepdim reduce + reshape"
        );
        let OpKind::Reduce { op, axis, keepdim } = &lowered.eqns[1].op else {
            panic!(
                "second lowered equation must be the reduce, got {:?}",
                lowered.eqns[1].op
            );
        };
        assert_eq!((*op, *axis, *keepdim), (RedOp::Max, 1, true));
        let OpKind::Reshape { shape } = lowered.eqns[2].op.clone() else {
            panic!(
                "third lowered equation must be the reshape, got {:?}",
                lowered.eqns[2].op
            );
        };
        assert_eq!(shape, vec![1, 4]);
        assert_eq!(lowered.eqns[2].out, graph.output);
        assert_eq!(lowered.aval(lowered.eqns[2].out).shape, vec![1, 4]);
        lowered
            .validate()
            .expect("lowered keepdim graph stays valid");
    }

    /// Card 454 D1: the CPU oracle evaluates `lower_nonlast_reduces`'s output identically to the
    /// original non-last-axis reduce, bit for bit - moved here from `poot-eval`'s own test crate
    /// (Card 534a: the pass itself can no longer be named from outside
    /// `poot-graph-plan`, so its CPU-oracle bit-identity test moved with it; `poot-graph-plan` already
    /// dev-depends on `poot-eval` for exactly this).
    #[test]
    fn lower_nonlast_reduces_is_bit_identical_to_the_original_reduce() {
        use poot_graph_ir::BinOp;
        use std::collections::HashMap;

        /// One non-trivial `[3, 4]` payload: strictly decreasing, no ties, so Sum and Max both depend
        /// on every element (an all-equal payload could not distinguish an ordered reduction from a
        /// permuted one).
        fn payload() -> Vec<f32> {
            (0..12).map(|index| 12.0 - index as f32 * 0.75).collect()
        }

        fn inputs(x: poot_graph_ir::ValueId) -> HashMap<poot_graph_ir::ValueId, poot_eval::Value> {
            HashMap::from([(
                x,
                poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![3, 4], payload())),
            )])
        }

        /// Every `Reduce` in `g` reduces the last axis of its input.
        fn all_reduces_are_last_axis(g: &poot_graph_ir::Graph) -> bool {
            g.eqns.iter().all(|eqn| {
                let OpKind::Reduce { axis, .. } = eqn.op else {
                    return true;
                };
                let Some(Operand::Value(id)) = eqn.inputs.first() else {
                    return true;
                };
                axis + 1 == g.aval(*id).shape.len()
            })
        }

        for (op, keepdim) in [
            (RedOp::Sum, false),
            (RedOp::Sum, true),
            (RedOp::Max, false),
            (RedOp::Max, true),
        ] {
            // Bare axis-0 reduce: [3, 4] -> [4] (or [1, 4] under keepdim).
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![3, 4]));
            let reduced = b.reduce(op, x, 0, keepdim);
            let bare = b.finish(reduced);

            // With a consumer, so a wrong lowered value would surface downstream too.
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![3, 4]));
            let reduced = b.reduce(op, x, 0, keepdim);
            let consumed = b.binary(BinOp::Add, reduced, reduced);
            let graph = b.finish(consumed);

            for original in [bare, graph] {
                let bound = inputs(x.id);
                let base = poot_eval::eval(
                    &original,
                    &bound,
                    poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
                )
                .unwrap()
                .output
                .into_host()
                .unwrap();
                let lowered = lower_nonlast_reduces(&original);
                assert!(
                    all_reduces_are_last_axis(&lowered),
                    "lowered graph still holds a non-last-axis reduce (op={op:?}, keepdim={keepdim})"
                );
                let out = poot_eval::eval(
                    &lowered,
                    &bound,
                    poot_eval::EvalOptions::new(poot_eval::EvalBudget::UNBOUNDED),
                )
                .unwrap()
                .output
                .into_host()
                .unwrap();

                assert_eq!(out.shape(), base.shape(), "op={op:?}, keepdim={keepdim}");
                let bits = |t: &poot_tensor::HostTensor| {
                    t.as_f32()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&out),
                    bits(&base),
                    "lowering changed values (op={op:?}, keepdim={keepdim})"
                );
            }
        }
    }
}

use crate::*;

use std::sync::Arc;

/// Element kind, logical dtype and layout (card 527): the storage contract every runtime's buffer
/// handle now carries. See [`poot_target::BufferStorage`].
pub use poot_target::{BufferStorage, ElementKind, LogicalDType, StorageLayout};

/// One source of `Context::download_segments_u32`: the first `words` 4-byte lanes of `buffer`.
/// `pub(crate)`/test-only: no production caller survives Card 546b's `GpuExecutor` deletion; kept
/// for the runtime-level memory-accounting test (`src/tests.rs`).
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct BufferSegment<'a> {
    pub buffer: &'a DeviceBuffer,
    pub words: usize,
}

/// Destination byte offset of each segment in one concatenated staging buffer, plus the total byte
/// length. Each segment copies from offset 0 of its source and may not request more words than the
/// source holds. `pub(crate)`/test-only alongside [`BufferSegment`]/`download_segments_u32`.
#[cfg(test)]
pub(crate) fn segment_byte_layout(
    segments: impl IntoIterator<Item = (usize, usize)>,
) -> Result<(Vec<u64>, u64), RuntimeError> {
    let mut offsets = Vec::new();
    let mut total = 0u64;
    for (segment, (requested, available)) in segments.into_iter().enumerate() {
        if requested > available {
            return Err(RuntimeError::SegmentLength {
                segment,
                requested,
                available,
            });
        }
        let bytes = u64::try_from(requested)
            .ok()
            .and_then(|words| words.checked_mul(4))
            .ok_or(RuntimeError::SegmentOverflow { segment })?;
        offsets.push(total);
        total = total
            .checked_add(bytes)
            .ok_or(RuntimeError::SegmentOverflow { segment })?;
    }
    Ok((offsets, total))
}

/// A kernel buffer: raw bytes + native element kind (for the length buffer and the schema check, card
/// 608) + whether the kernel writes it (writable buffers are read back into its bytes after dispatch).
///
/// `bytes` and the element count are private, with no setter for either (card 608, SC-004): the only way
/// to build one is through a constructor below, whose element kind and count both come from the same
/// slice, and the only later mutation ([`Self::overwrite_after_readback`]) copies into the existing byte
/// length without resizing, so a `KernelBuffer`'s recorded element count can never desync from its
/// storage.
pub struct KernelBuffer {
    bytes: Vec<u8>,
    element: ElementKind,
    writable: bool,
}

impl KernelBuffer {
    fn new(bytes: Vec<u8>, element: ElementKind, writable: bool) -> Self {
        KernelBuffer {
            bytes,
            element,
            writable,
        }
    }

    pub fn read_only_f32(data: &[f32]) -> Self {
        Self::new(bytemuck::cast_slice(data).to_vec(), ElementKind::F32, false)
    }
    /// A read-only buffer of exact i32 bytes, for a kernel param the body declares as `Slice<i32>` (e.g.
    /// a `PackedDequant`/`PackedContraction` carrier, whose words exceed 2^24 and cannot survive the f32
    /// `Tensor` mirror). The wgpu storage buffer is just bytes; the SPIR-V declares the i32 interpretation.
    pub fn read_only_i32(data: &[i32]) -> Self {
        Self::new(bytemuck::cast_slice(data).to_vec(), ElementKind::I32, false)
    }
    /// A read-only buffer of u32 bytes, for a kernel param the body declares as `Slice<u32>`, e.g. a
    /// shape-generic kernel's runtime dimensions (`dims: &[u32]`). Same native representation as the i32
    /// form (card 608: both are [`ElementKind::I32`], the one 4-byte native word).
    pub fn read_only_u32(data: &[u32]) -> Self {
        Self::new(bytemuck::cast_slice(data).to_vec(), ElementKind::I32, false)
    }
    pub fn write_f32(len: usize) -> Self {
        Self::new(vec![0u8; len * 4], ElementKind::F32, true)
    }
    /// A writable buffer of `len` exact i32 words, for a kernel whose declared output is `&mut [i32]`
    /// (card 551a: `SampleToken`'s `(token, non_finite_index)` pair).
    pub fn write_i32(len: usize) -> Self {
        Self::new(vec![0u8; len * 4], ElementKind::I32, true)
    }
    /// A writable f32 buffer pre-populated with `data` (read back into it after dispatch), for a kernel
    /// that both reads and writes the same binding in place. `pub(crate)` and `#[cfg(test)]`: its only
    /// caller is this crate's own `vec4_probe_tests` (dead-pub has no exemption for a pub item only an
    /// integration test uses).
    #[cfg(test)]
    pub(crate) fn read_write_f32(data: &[f32]) -> Self {
        Self::new(bytemuck::cast_slice(data).to_vec(), ElementKind::F32, true)
    }
    pub fn as_f32(&self) -> &[f32] {
        bytemuck::cast_slice(&self.bytes)
    }
    /// Read this buffer's bytes as exact i32 words (the [`Self::write_i32`] counterpart).
    pub fn as_i32(&self) -> &[i32] {
        bytemuck::cast_slice(&self.bytes)
    }
    /// Raw bytes, for a native element kind (F16) with no matching Rust scalar type in this crate.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The native element kind a dispatch's schema check compares against (card 608).
    pub(crate) fn element(&self) -> ElementKind {
        self.element
    }

    pub(crate) fn elem_count(&self) -> u32 {
        (self.bytes.len() / self.element.byte_width()) as u32
    }

    pub(crate) fn writable(&self) -> bool {
        self.writable
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Overwrite this buffer's bytes with a device readback of the same length (never resizes, so the
    /// recorded element count stays in sync with the storage, card 608 SC-004). Panics if `data`'s length
    /// differs: a mismatched readback is a caller bug in the dispatch it is called from, not a value this
    /// type should silently accept.
    pub(crate) fn overwrite_after_readback(&mut self, data: &[u8]) {
        assert_eq!(
            data.len(),
            self.bytes.len(),
            "KernelBuffer readback length must match its own byte length"
        );
        self.bytes.copy_from_slice(data);
    }
}

pub(crate) fn workgroup_count(threads: u32, wg: u32) -> u32 {
    threads.div_ceil(wg.max(1))
}

/// Raw per-axis workgroup counts for a launch (`threads` / `wg`), before any over-cap fold.
pub(crate) fn raw_groups(threads: [u32; 3], wg: [u32; 3]) -> [u32; 3] {
    [
        workgroup_count(threads[0], wg[0]),
        workgroup_count(threads[1], wg[1]),
        workgroup_count(threads[2], wg[2]),
    ]
}

/// Fold a 1-D grid over the device cap and validate every axis. On success returns the launch grid;
/// on a grid that cannot be folded within the cap returns [`RuntimeError::GridCap`] carrying both
/// the raw grid and the device limit.
pub(crate) fn resolve_groups(
    threads: [u32; 3],
    wg: [u32; 3],
    max_workgroups: u32,
) -> Result<[u32; 3], RuntimeError> {
    let raw = raw_groups(threads, wg);
    let folded = poot_runtime_common::fold_grid(raw, max_workgroups, max_workgroups)
        .ok_or(RuntimeError::GridCap(raw, max_workgroups))?;
    if folded.iter().any(|&g| g == 0 || g > max_workgroups) {
        return Err(RuntimeError::GridCap(folded, max_workgroups));
    }
    Ok(folded)
}

/// The X-thread extent of a folded (or 1-D) launch: `groups[0] * wg[0]`. Shared codegen reconstructs
/// the linear index as `group_y * x_extent + global_id.x` from this value, which the runtime appends
/// to the length buffer at slot `param_count`.
pub(crate) fn x_extent(groups: [u32; 3], wg: [u32; 3]) -> u32 {
    groups[0].saturating_mul(wg[0].max(1))
}

/// A persistent device buffer (stays on the GPU across dispatches). Cloning shares the GPU buffer
/// (`wgpu::Buffer` is Arc-backed); reshape uses this to alias its input.
#[derive(Clone)]
pub struct DeviceBuffer {
    pub(crate) buf: wgpu::Buffer,
    pub(crate) elem_count: u32,
    /// A cheap identity tag distinct on every allocation (see [`next_buffer_id`]) and shared across
    /// clones. Card 156 phase 3: the cached-decode ping-pong KV recovery pointer-compares it against its
    /// two held physical buffers to recover parity / detect a new seed (spec 137 B2), without wgpu-internal
    /// resource ids.
    pub(crate) id: u64,
    /// Element kind, logical dtype and layout, set at allocation (card 527, R471-009/R484-001): before
    /// this field, a wgpu buffer carried no element kind at all (unlike ROCm/PTX's `BufferElement`).
    pub(crate) storage: BufferStorage,
    /// This allocation's role-tagged live-bytes guard (Card 547a: the one memory service). Shared via
    /// `Arc` so every clone aliases the same guard, exactly as every clone aliases the same
    /// `wgpu::Buffer`: the role's live bytes decrement once, when the last clone drops. Never read:
    /// its only purpose is the decrement its `Drop` runs.
    #[allow(dead_code, reason = "held only for its Drop side effect, Card 547a")]
    pub(crate) guard: Arc<poot_runtime_common::AllocGuard>,
}

impl DeviceBuffer {
    /// This buffer's identity tag (see the `id` field doc). Clones of one `DeviceBuffer` (sharing the
    /// `wgpu::Buffer`) compare equal; independently allocated buffers of identical size never do.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The number of native elements this buffer holds (card 608: private field, no setter, mirroring
    /// `poot-vulkan-runtime`'s `DeviceBuffer::elem_count`).
    pub fn elem_count(&self) -> u32 {
        self.elem_count
    }

    /// The native device-word representation this buffer was allocated with (card 527).
    pub fn element(&self) -> ElementKind {
        self.storage.element()
    }

    /// The logical dtype this buffer represents (card 527): may differ from [`Self::element`] for a
    /// packed or widened value, or carry no tensor meaning ([`LogicalDType::RawBytes`]) for a
    /// generic word buffer (metadata, packed rows) whose logical dtype the caller tracks separately.
    pub fn dtype(&self) -> LogicalDType {
        self.storage.dtype()
    }

    /// How [`Self::element`] packs [`Self::dtype`] (card 527).
    pub fn layout(&self) -> StorageLayout {
        self.storage.layout()
    }

    /// The full storage contract (card 527, R471-009/R484-001): element kind, logical dtype and layout
    /// together, set at allocation.
    pub fn storage(&self) -> BufferStorage {
        self.storage
    }

    /// Replace this buffer's recorded storage with `storage` (card 527). The low-level
    /// `Context::upload_*`/`alloc_*` methods tag a buffer by which method was called, which is right for
    /// the common case but wrong when one generic method (e.g. `upload_u32`) serves several distinct
    /// logical payloads (a packed-BF16 const, an E4M3FN packed row, plain dims metadata): the caller,
    /// which knows the plan's real decision for this value, retags the freshly uploaded buffer before
    /// binding it so the handle records what the plan actually decided, not the upload method's generic
    /// default.
    /// `pub(crate)`/test-only: no production caller survives Card 546b's `GpuExecutor` deletion;
    /// kept for the runtime-level storage-check tests (`src/tests.rs`).
    #[cfg(test)]
    pub(crate) fn with_storage(mut self, storage: BufferStorage) -> Self {
        self.storage = storage;
        self
    }
}

/// Validate a copy against this buffer's full storage contract (card 527, R484-001/R471-009): a release
/// check on element kind, logical dtype and layout together, not element kind alone (which a packed or
/// widened value can share with a buffer of a different logical meaning; see
/// [`poot_target::BufferStorage`]). A caller that legitimately changes what a reused buffer holds (e.g.
/// a slot cache hit whose lane flips between the raw-i32 and f32-mirror upload) retags it first via
/// [`DeviceBuffer::with_storage`], so the check here only ever catches a genuine mismatch.
pub(crate) fn checked_storage(
    op: &'static str,
    buf: &DeviceBuffer,
    expected: BufferStorage,
) -> Result<(), RuntimeError> {
    let actual = buf.storage();
    if actual != expected {
        return Err(RuntimeError::RepresentationMismatch {
            op,
            expected,
            actual,
        });
    }
    Ok(())
}

/// Check `ins` then `out` (inputs first, the single output last: the binding order every wgpu entry
/// point uses) against `kernel`'s argument schema (card 608, SC-002): a release check on argument count
/// and each buffer's native element kind, before any submission.
pub(crate) fn check_device_args(
    kernel: &CompiledKernel,
    ins: &[&DeviceBuffer],
    out: &DeviceBuffer,
) -> Result<(), RuntimeError> {
    let elements: Vec<ElementKind> = ins
        .iter()
        .map(|b| b.element())
        .chain(std::iter::once(out.element()))
        .collect();
    poot_runtime_common::check_kernel_args(kernel, &elements)?;
    Ok(())
}

/// [`check_device_args`] for the host-buffer `dispatch`/`launch` entry points, whose `buffers` are
/// already in binding order (inputs then the single output).
pub(crate) fn check_kernel_buffer_args(
    kernel: &CompiledKernel,
    buffers: &[KernelBuffer],
) -> Result<(), RuntimeError> {
    let elements: Vec<ElementKind> = buffers.iter().map(KernelBuffer::element).collect();
    poot_runtime_common::check_kernel_args(kernel, &elements)?;
    Ok(())
}

/// Validate a partial write's range against `buf`'s allocated byte capacity (card 527 review G2): the
/// same shared range check ROCm/PTX already run before every `write_*_at` (`poot_runtime_common::checked_copy_range`),
/// never requesting a whole-buffer match (`write_*_at` writes a sub-range by design). `element_bytes` is
/// the caller's own unit - 4 for the typed f32/i32 writers, 1 for [`Context::write_le_bytes_at`]'s raw
/// byte offset - and the byte capacity is always computed from `buf`'s own recorded element width, not a
/// hardcoded lane size: `write_le_bytes_at` runs against both `f32()`- and `i32()`-tagged buffers (both
/// 4-byte elements today), and this stays correct even for a narrower element width.
pub(crate) fn checked_write_range(
    op: &'static str,
    buf: &DeviceBuffer,
    element_offset: usize,
    elements: usize,
    element_bytes: usize,
) -> Result<(), RuntimeError> {
    let byte_capacity = buf.elem_count as usize * buf.storage().element().byte_width();
    poot_runtime_common::checked_copy_range(
        element_offset,
        elements,
        element_bytes,
        false,
        buf.elem_count,
        byte_capacity,
    )
    .map(|_| ())
    .map_err(|error| match error {
        poot_runtime_common::CopyRangeError::SizeMismatch { .. } => {
            unreachable!("checked_write_range never requests a whole-buffer match")
        }
        poot_runtime_common::CopyRangeError::RangeOverflow => RuntimeError::RangeOverflow {
            op,
            element_offset,
            elements,
        },
        poot_runtime_common::CopyRangeError::RangeOutOfBounds {
            byte_offset,
            byte_end,
            byte_capacity,
        } => RuntimeError::RangeOutOfBounds {
            op,
            byte_offset,
            byte_end,
            byte_capacity,
        },
    })
}

#[cfg(test)]
mod kernel_buffer_tests {
    use super::*;

    /// Card 608 SC-004: every constructor's recorded element count matches its byte storage divided by
    /// its own native element width. `KernelBuffer` has no `elem_count` field to desync at all (card 608
    /// review): the count is computed from `bytes.len()` and `element` on every call, so the mutation
    /// SC-004 names ("expose a setter that writes the count alone") cannot even be expressed against this
    /// representation - there is no field for such a setter to write.
    #[test]
    fn elem_count_is_always_derived_from_bytes_len_and_element_width() {
        let cases: Vec<KernelBuffer> = vec![
            KernelBuffer::read_only_f32(&[1.0, 2.0, 3.0]),
            KernelBuffer::read_only_i32(&[1, 2]),
            KernelBuffer::read_only_u32(&[1, 2, 3, 4]),
            KernelBuffer::write_f32(5),
            KernelBuffer::write_i32(5),
            KernelBuffer::read_write_f32(&[1.0, 2.0]),
        ];
        for kb in &cases {
            assert_eq!(
                kb.elem_count() as usize * kb.element().byte_width(),
                kb.bytes().len(),
                "elem_count * element width must equal the byte storage exactly"
            );
        }
    }

    /// The only post-construction mutation, `overwrite_after_readback`, is length-preserving: it panics
    /// rather than silently resizing (and thus desyncing) the recorded count.
    #[test]
    #[should_panic(expected = "KernelBuffer readback length must match its own byte length")]
    fn overwrite_after_readback_refuses_a_length_change() {
        let mut kb = KernelBuffer::write_f32(4);
        kb.overwrite_after_readback(&[0u8; 8]); // 2 elements' worth of bytes, not 4
    }

    #[test]
    fn overwrite_after_readback_keeps_elem_count_in_sync() {
        let mut kb = KernelBuffer::write_f32(3);
        kb.overwrite_after_readback(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]));
        assert_eq!(kb.elem_count(), 3);
        assert_eq!(kb.as_f32(), &[1.0, 2.0, 3.0]);
    }
}

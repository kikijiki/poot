//! A device buffer's storage contract (card 527, R471-009/R484-001): element kind, logical dtype and
//! layout, set once at allocation from the plan's storage record ([`poot_graph_plan::ValueStorage`],
//! card 526) and read at every bind and copy. Before this, PTX and ROCm each kept a private
//! `BufferElement` copy (native element kind and capacity only, no dtype or layout), and wgpu/Vulkan
//! carried no element kind at all, checking only length, in debug builds only (R471-009). A leaf-crate
//! type ([[sound-architecture-over-convenience]]): `poot-graph-ir`'s `DType` already depends on this
//! crate, so this crate cannot depend back on it, and defines its own dtype vocabulary
//! ([`LogicalDType`]) instead; `poot-graph-plan` converts at the boundary.

/// The physical device-buffer word a copy's byte-range math uses. The one native-element vocabulary
/// every runtime's buffer handle now shares (previously two private `BufferElement` copies, ROCm's and
/// PTX's, with different variant sets and neither shared with wgpu or Vulkan).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ElementKind {
    /// 32-bit IEEE float.
    F32,
    /// 32-bit signed integer.
    I32,
    /// bfloat16.
    Bf16,
    /// 16-bit IEEE float.
    F16,
    /// Opaque bytes with no fixed per-element width the runtime interprets: kernel-argument segments,
    /// length/dims metadata, and the packed-byte payload of an I8 or E4M3FN [`LogicalDType`]. Copies and
    /// binds move raw bytes.
    RawBytes,
}

impl ElementKind {
    /// Bytes one physical unit of this element kind occupies.
    pub const fn byte_width(self) -> usize {
        match self {
            ElementKind::F32 | ElementKind::I32 => 4,
            ElementKind::Bf16 | ElementKind::F16 => 2,
            ElementKind::RawBytes => 1,
        }
    }
}

impl std::fmt::Display for ElementKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The logical dtype a buffer's bytes represent, independent of `poot_tensor::DType` (this crate is a
/// leaf `poot-graph-ir` depends on; see the module doc). May differ from the buffer's native
/// [`ElementKind`]: a packed or widened value (R484-001) shares physical storage with a value of a
/// different logical dtype, e.g. BF16 bytes packed two-per-`u32` word have element kind
/// [`ElementKind::I32`] (canonical, [`BufferStorage::bf16_packed`]) and logical dtype `Bf16`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LogicalDType {
    /// 32-bit IEEE float.
    F32,
    /// bfloat16.
    Bf16,
    /// 16-bit IEEE float.
    F16,
    /// 32-bit signed integer.
    I32,
    /// Signed quantized-KV codes, or another opaque one-byte-per-element carrier.
    I8,
    /// OCP E4M3FN FP8 storage (spec 149).
    E4M3FN,
    /// Not a tensor value: kernel-argument segments, length/dims metadata. Always paired with
    /// [`ElementKind::RawBytes`].
    RawBytes,
}

impl std::fmt::Display for LogicalDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// How a buffer's element kind packs its logical dtype (mirrors `poot_graph_plan::StorageKind`, card
/// 526, at the leaf-crate level, so a buffer handle's own record and the plan's record speak the same
/// vocabulary).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StorageLayout {
    /// One buffer element per logical element, at the element kind's natural width.
    Dense,
    /// Two logical BF16 elements packed per element-kind word.
    Bf16Packed,
    /// Two logical IEEE binary16 elements packed per element-kind word (Card 1007).
    F16Packed,
    /// Four logical E4M3FN elements packed per element-kind word, each row of the last axis padded to a
    /// whole word (POOT-1060): element `c` of a row lives in word `c / 4` at bit `(c % 4) * 8`, and row `r`
    /// starts at word `r * ceil(width / 4)`. The checkpoint's own bytes, `poot_eval::fp8::PortableE4m3FnLayout`'s
    /// device layout, so the generated row gather decodes each byte in-register.
    E4m3Packed,
    /// No typed layout: an arena slot of bytes (POOT-1450). No value is of this storage; every access goes
    /// through a typed view of the buffer ([`BufferStorage::view_elems`]), whose storage is the accessing
    /// value's own. Always paired with [`ElementKind::RawBytes`], so a buffer's native element count is its
    /// byte size.
    Untyped,
}

impl std::fmt::Display for StorageLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageLayout::Dense => write!(f, "dense"),
            StorageLayout::Bf16Packed => write!(f, "bf16-packed"),
            StorageLayout::F16Packed => write!(f, "f16-packed"),
            StorageLayout::E4m3Packed => write!(f, "e4m3-packed"),
            StorageLayout::Untyped => write!(f, "untyped"),
        }
    }
}

/// One device buffer's storage contract: element kind, logical dtype and layout together, so a bind or
/// copy call catches not just a raw representation mismatch (element kind alone - the pre-card-527
/// check) but a same-element, different-meaning mismatch too (R484-001). Two buffers can share an
/// element kind and still not be interchangeable: BF16 packed two-per-word and a widened dense F32
/// buffer can both use element kind `F32`, and a plain dense I32 tensor and a packed-quant I32 payload
/// can both use element kind `I32`; only the full [`BufferStorage`] tells them apart.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferStorage {
    element: ElementKind,
    dtype: LogicalDType,
    layout: StorageLayout,
}

impl BufferStorage {
    /// A storage from its three parts. Prefer the named constructors, which fix the combinations the
    /// executors support.
    pub const fn new(element: ElementKind, dtype: LogicalDType, layout: StorageLayout) -> Self {
        Self {
            element,
            dtype,
            layout,
        }
    }

    /// A dense buffer whose element kind and logical dtype agree (the common case: nothing packed or
    /// widened it).
    pub const fn dense(element: ElementKind, dtype: LogicalDType) -> Self {
        Self::new(element, dtype, StorageLayout::Dense)
    }

    /// BF16 packed two logical elements per `u32` word (card 527): the one canonical element
    /// kind every executor's packed-BF16 upload actually uses (wgpu `upload_u32`, ROCm `upload_i32`),
    /// so the plan's `BufferStorage` and an executor's uploaded handle can be equal for the same value.
    /// Not a caller-chosen parameter: a packed value has exactly one physical representation, decided
    /// here once, not re-decided per call site.
    pub const fn bf16_packed() -> Self {
        Self::new(
            ElementKind::I32,
            LogicalDType::Bf16,
            StorageLayout::Bf16Packed,
        )
    }

    /// IEEE binary16 packed two logical elements per `u32` word (Card 1007), the F16 sibling of
    /// [`Self::bf16_packed`] with the same word: element `i` in word `i / 2` at bit `(i % 2) * 16`, the
    /// checkpoint's own little-endian byte order. A generated body decodes each half in-register, so
    /// the device buffer keeps the checkpoint's two bytes per element on every backend, with neither a
    /// widened f32 copy nor a native two-byte F16 buffer.
    pub const fn f16_packed() -> Self {
        Self::new(
            ElementKind::I32,
            LogicalDType::F16,
            StorageLayout::F16Packed,
        )
    }

    /// E4M3FN packed four elements per `u32` word with rows padded to a whole word (POOT-1060), the
    /// sibling of [`Self::bf16_packed`] on the same word. A buffer of this storage holds a table whose last
    /// axis is the row; its word count depends on that width, so [`Self::device_elems`] takes the shape.
    pub const fn e4m3_packed() -> Self {
        Self::new(
            ElementKind::I32,
            LogicalDType::E4M3FN,
            StorageLayout::E4m3Packed,
        )
    }

    /// Dense F32.
    pub const fn f32() -> Self {
        Self::dense(ElementKind::F32, LogicalDType::F32)
    }
    /// Dense I32.
    pub const fn i32() -> Self {
        Self::dense(ElementKind::I32, LogicalDType::I32)
    }
    /// Dense BF16, one value per 16-bit element.
    pub const fn bf16() -> Self {
        Self::dense(ElementKind::Bf16, LogicalDType::Bf16)
    }
    /// Dense F16.
    pub const fn f16() -> Self {
        Self::dense(ElementKind::F16, LogicalDType::F16)
    }
    /// Opaque bytes with no tensor dtype: kernel-argument segments, length/dims metadata.
    pub const fn raw_bytes() -> Self {
        Self::dense(ElementKind::RawBytes, LogicalDType::RawBytes)
    }

    /// An arena slot of bytes (POOT-1450): a buffer counted in bytes that no value is ever stored *as*.
    /// Values of any device-encoded storage share it over disjoint lifetimes, each reaching it only through
    /// a typed view ([`Self::view_elems`]). Distinct from [`Self::raw_bytes`], which is a typed buffer of
    /// metadata words that admits only a view of itself.
    pub const fn untyped() -> Self {
        Self::new(
            ElementKind::RawBytes,
            LogicalDType::RawBytes,
            StorageLayout::Untyped,
        )
    }

    /// The byte multiple an untyped buffer is planned in: one 32-bit word, which every element width
    /// divides, so a planned slot never refuses a planned view. wgpu and Vulkan bind and copy whole words,
    /// and the copy kernel moves words.
    pub const UNTYPED_ALIGN: usize = 4;

    /// The native element count of a view of storage `view` over a buffer of this storage holding `elems`
    /// native elements (POOT-1450): the storage contract checked per binding, not per allocation.
    ///
    /// A typed buffer admits only a view of its own storage (R484-001: its storage was fixed at
    /// allocation). An untyped buffer admits a view of any storage with a device encoding whose element
    /// width divides the buffer's bytes; the view starts at byte 0, the allocation's base, which every
    /// runtime aligns to at least 256 bytes.
    ///
    /// # Errors
    ///
    /// [`ViewError::Storage`] for another storage over a typed buffer, [`ViewError::Untypable`] for a view
    /// that is itself untyped or has no device encoding, [`ViewError::Misaligned`] when the bytes are not
    /// whole elements of the view.
    pub fn view_elems(self, elems: usize, view: BufferStorage) -> Result<usize, ViewError> {
        if self.layout != StorageLayout::Untyped {
            return if view == self {
                Ok(elems)
            } else {
                Err(ViewError::Storage { buffer: self, view })
            };
        }
        if view.layout == StorageLayout::Untyped || view.device_elems(&[1]).is_none() {
            return Err(ViewError::Untypable { view });
        }
        let bytes = elems * self.element.byte_width();
        let width = view.element.byte_width();
        if !bytes.is_multiple_of(width) {
            return Err(ViewError::Misaligned {
                bytes,
                element: view.element,
            });
        }
        Ok(bytes / width)
    }

    /// The native device-word representation: what a copy's byte-range math must use.
    pub const fn element(&self) -> ElementKind {
        self.element
    }

    /// The logical dtype this buffer represents once unpacked.
    pub const fn dtype(&self) -> LogicalDType {
        self.dtype
    }

    /// How logical elements are arranged in the physical words.
    pub const fn layout(&self) -> StorageLayout {
        self.layout
    }

    /// Native device elements a value of `shape` occupies in this storage (Card 547b): `None` for a lane
    /// no executor device-encodes (dense E4M3FN/RawBytes/I8 - left to the host). The one owner both
    /// `poot-graph-plan` (buffer-plan sizing) and `poot-executor` (bind-time allocation and dispatch
    /// argument lengths) call, so the two can never drift against each other - before this, each crate
    /// kept its own copy of the same formula. Takes the shape, not an element count, because the packed
    /// E4M3FN layout pads each row of the last axis to a whole word (POOT-1060); a 0-D value is one row
    /// of width one.
    pub fn device_elems(&self, shape: &[usize]) -> Option<usize> {
        let numel: usize = shape.iter().product();
        match self.layout {
            StorageLayout::Bf16Packed | StorageLayout::F16Packed => Some(numel.div_ceil(2)),
            StorageLayout::E4m3Packed => {
                let width = shape.last().copied().unwrap_or(1);
                Some(numel.checked_div(width).unwrap_or(0) * width.div_ceil(4))
            }
            StorageLayout::Dense => match self.dtype {
                LogicalDType::E4M3FN | LogicalDType::RawBytes | LogicalDType::I8 => None,
                _ => Some(numel),
            },
            StorageLayout::Untyped => None,
        }
    }
}

/// Why a buffer refuses a typed view ([`BufferStorage::view_elems`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, thiserror::Error)]
pub enum ViewError {
    /// A typed buffer viewed as another storage.
    #[error("a buffer of storage {buffer} cannot be viewed as {view}")]
    Storage {
        /// The buffer's storage, fixed at allocation.
        buffer: BufferStorage,
        /// The storage the view asked for.
        view: BufferStorage,
    },
    /// A view no value can have: untyped itself, or a lane with no device encoding.
    #[error("no binding can view a buffer as {view}")]
    Untypable {
        /// The storage the view asked for.
        view: BufferStorage,
    },
    /// The untyped buffer's bytes are not whole elements of the view.
    #[error("an untyped buffer of {bytes} bytes is not whole {element} elements")]
    Misaligned {
        /// The untyped buffer's byte size.
        bytes: usize,
        /// The view's element kind.
        element: ElementKind,
    },
}

impl std::fmt::Display for BufferStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}, {})", self.dtype, self.element, self.layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_constructors_agree_element_and_dtype() {
        assert_eq!(BufferStorage::f32().element(), ElementKind::F32);
        assert_eq!(BufferStorage::f32().dtype(), LogicalDType::F32);
        assert_eq!(BufferStorage::f32().layout(), StorageLayout::Dense);
    }

    /// Same element kind, different logical dtype and layout: the case a pre-card-527 element-only check
    /// let through (R484-001). `bf16_packed` canonicalizes on `ElementKind::I32` (review F2): the one
    /// word every executor's packed-BF16 upload actually uses, so it collides with a plain dense I32
    /// buffer, not an F32 one.
    #[test]
    fn bf16_packed_shares_element_kind_with_dense_i32() {
        let packed = BufferStorage::bf16_packed();
        let dense = BufferStorage::i32();
        assert_eq!(packed.element(), dense.element());
        assert_ne!(packed.dtype(), dense.dtype());
        assert_ne!(packed.layout(), dense.layout());
        assert_ne!(packed, dense);
        assert_eq!(packed.layout(), StorageLayout::Bf16Packed);
    }

    /// Card 1007: packed F16 shares the packed-BF16 word (and so a dense I32 buffer's element kind) but
    /// not its meaning: a bind comparing full storage records refuses each for the other, and both
    /// occupy one word per two logical elements.
    #[test]
    fn f16_packed_is_its_own_record_on_the_packed_word() {
        let packed = BufferStorage::f16_packed();
        assert_eq!(packed.element(), ElementKind::I32);
        assert_eq!(packed.dtype(), LogicalDType::F16);
        assert_eq!(packed.layout(), StorageLayout::F16Packed);
        assert_ne!(packed, BufferStorage::bf16_packed());
        assert_ne!(packed, BufferStorage::i32());
        assert_ne!(packed, BufferStorage::f16());
        assert_eq!(packed.device_elems(&[7]), Some(4));
        assert_eq!(packed.device_elems(&[2, 4]), Some(4));
    }

    /// POOT-1450: a typed buffer admits only a view of its own storage, whatever the element width: an f32
    /// buffer viewed as i32 (same word) is refused as surely as one viewed as bf16.
    /// Mutation: admit any view on a typed buffer; both refusals go red.
    #[test]
    fn a_typed_buffer_admits_only_its_own_storage() {
        assert_eq!(
            BufferStorage::f32().view_elems(6, BufferStorage::f32()),
            Ok(6)
        );
        for view in [BufferStorage::i32(), BufferStorage::bf16()] {
            assert_eq!(
                BufferStorage::f32().view_elems(6, view),
                Err(ViewError::Storage {
                    buffer: BufferStorage::f32(),
                    view
                })
            );
        }
        assert_eq!(
            BufferStorage::bf16_packed().view_elems(3, BufferStorage::i32()),
            Err(ViewError::Storage {
                buffer: BufferStorage::bf16_packed(),
                view: BufferStorage::i32()
            })
        );
    }

    /// POOT-1450: an untyped buffer admits every device-encoded storage, counted in the view's own
    /// elements, and refuses a view that is not whole elements of its bytes or has no encoding.
    /// Mutation: drop the whole-elements check; the 6-byte f32 view is admitted with 1 element.
    #[test]
    fn an_untyped_buffer_admits_any_encoded_storage_in_whole_elements() {
        let slot = BufferStorage::untyped();
        assert_eq!(slot.element().byte_width(), 1, "elems are bytes");
        assert_eq!(slot.device_elems(&[8]), None, "no value is untyped");
        assert_eq!(slot.view_elems(8, BufferStorage::f32()), Ok(2));
        assert_eq!(slot.view_elems(8, BufferStorage::bf16()), Ok(4));
        assert_eq!(slot.view_elems(8, BufferStorage::bf16_packed()), Ok(2));
        assert_eq!(slot.view_elems(6, BufferStorage::f16()), Ok(3));
        assert_eq!(
            slot.view_elems(6, BufferStorage::f32()),
            Err(ViewError::Misaligned {
                bytes: 6,
                element: ElementKind::F32
            })
        );
        for view in [BufferStorage::untyped(), BufferStorage::raw_bytes()] {
            assert_eq!(slot.view_elems(8, view), Err(ViewError::Untypable { view }));
        }
        assert!(BufferStorage::UNTYPED_ALIGN.is_multiple_of(ElementKind::F32.byte_width()));
    }

    /// POOT-1060: packed E4M3FN is its own record on the packed word, and its word count pads each row of
    /// the last axis to a whole word: a width that is not a multiple of four starts the next row in the
    /// next word. Dense E4M3FN still has no device encoding.
    #[test]
    fn e4m3_packed_pads_each_row_to_a_whole_word() {
        let packed = BufferStorage::e4m3_packed();
        assert_eq!(packed.element(), ElementKind::I32);
        assert_eq!(packed.dtype(), LogicalDType::E4M3FN);
        assert_eq!(packed.layout(), StorageLayout::E4m3Packed);
        assert_ne!(packed, BufferStorage::bf16_packed());
        assert_ne!(packed, BufferStorage::i32());
        assert_eq!(packed.device_elems(&[3, 8]), Some(6));
        assert_eq!(packed.device_elems(&[3, 6]), Some(6));
        assert_eq!(packed.device_elems(&[3, 5]), Some(6));
        assert_eq!(packed.device_elems(&[3, 4]), Some(3));
        assert_eq!(packed.device_elems(&[3, 1]), Some(3));
        assert_eq!(packed.device_elems(&[2, 3, 6]), Some(12));
        assert_eq!(packed.device_elems(&[]), Some(1));
        assert_eq!(packed.device_elems(&[0, 6]), Some(0));
        assert_eq!(
            BufferStorage::dense(ElementKind::RawBytes, LogicalDType::E4M3FN).device_elems(&[4]),
            None
        );
    }
}

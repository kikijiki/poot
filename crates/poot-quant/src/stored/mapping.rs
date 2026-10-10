//! The one `unsafe` in `poot-quant`: a read-only private mapping of one checkpoint file (ADR-0103
//! amendment of 2026-10-09; design: `dmap.md` section 5).
//!
//! # Invariants
//!
//! 1. The file is a regular file (`S_ISREG`) of non-zero length, opened read-only; the mapping is
//!    `PROT_READ` and `MAP_PRIVATE`, so nothing in this process can write the pages (a write faults).
//! 2. Every `&[u8]` into the mapping borrows a [`super::StoredBytes`] whose `Arc<Region>` owns the
//!    [`MappedFile`]; the region is unmapped only by `Drop` of the last `Arc`, so no slice outlives the
//!    mapping.
//! 3. Every view's range is checked against the mapping length at construction
//!    ([`super::StoredBytes::slice`]), so no slice reaches past the mapped length.
//! 4. The one assumption the `unsafe` makes, which no Rust code can enforce: no other process modifies the
//!    file in place while it is mapped. Breaking it is not memory-unsafe in the Rust sense for truncation
//!    (the process dies with `SIGBUS`) and yields changing bytes for an overwrite; poot never derives a
//!    reference, length or index from mapped bytes without checking it, so changed bytes are wrong data,
//!    never wild memory access. A change already made is refused by [`MappedFile::check_unchanged`].
//!
//! `MappedFile` is `Send + Sync` through the auto traits (`memmap2::Mmap` and `File` are); there is no
//! `unsafe impl`.

use std::fs::{File, Metadata};
use std::ops::Range;
use std::os::unix::fs::MetadataExt;

use memmap2::{Advice, Mmap, MmapOptions};

use super::{Access, FileIdentity, MapError, SourceChanged};

/// The largest page size of any Linux target. `madvise` wants a page-aligned start; rounding a start
/// down to a multiple of this is page-aligned on every page size and only widens an advisory hint.
const MAX_PAGE: usize = 64 * 1024;

pub(super) struct MappedFile {
    /// `PROT_READ`, `MAP_PRIVATE`, offset 0, length `identity.len` at map time.
    map: Mmap,
    /// Kept open: [`Self::check_unchanged`] stats this descriptor, never the path.
    file: File,
    identity: FileIdentity,
}

impl FileIdentity {
    fn of(metadata: &Metadata) -> Self {
        let nanos = |seconds: i64, nanoseconds: i64| {
            seconds
                .saturating_mul(1_000_000_000)
                .saturating_add(nanoseconds)
        };
        Self {
            dev: metadata.dev(),
            ino: metadata.ino(),
            len: metadata.size(),
            nlink: metadata.nlink(),
            mtime_ns: nanos(metadata.mtime(), metadata.mtime_nsec()),
            ctime_ns: nanos(metadata.ctime(), metadata.ctime_nsec()),
        }
    }
}

impl MappedFile {
    /// Map all of `file`. The caller handles the zero-length case (an empty mapping is `EINVAL`), so
    /// `Ok(None)` is returned for an empty regular file.
    pub(super) fn new(file: File) -> Result<Option<Self>, MapError> {
        let metadata = file.metadata().map_err(MapError::Io)?;
        if !metadata.file_type().is_file() {
            return Err(MapError::NotRegularFile {
                mode: metadata.mode(),
            });
        }
        let identity = FileIdentity::of(&metadata);
        if identity.len == 0 {
            return Ok(None);
        }
        let len = usize::try_from(identity.len).map_err(|_| {
            MapError::Io(std::io::Error::other(
                "file length exceeds the address space",
            ))
        })?;
        // SAFETY: invariants 1-4 of the module docs. The file is a regular, non-empty file and the
        // mapping is read-only and private, so this process cannot write through it; the mapping is
        // owned by the returned value and unmapped when it drops. The one assumption no code can check,
        // that no other process modifies the file in place while it is mapped, is the documented
        // contract of `StoredBytes` (ADR-0103): a violation changes bytes or ends the process with
        // `SIGBUS`, and is refused by `check_unchanged` once it has happened.
        let map = unsafe { MmapOptions::new().len(len).map_copy_read_only(&file) }
            .map_err(MapError::Io)?;
        Ok(Some(Self {
            map,
            file,
            identity,
        }))
    }

    /// The whole mapping.
    pub(super) fn bytes(&self) -> &[u8] {
        &self.map
    }

    pub(super) fn identity(&self) -> FileIdentity {
        self.identity
    }

    /// `Ok` when the mapped descriptor's identity now equals the one at map time.
    pub(super) fn check_unchanged(&self) -> Result<(), SourceChanged> {
        let now = self
            .file
            .metadata()
            .ok()
            .map(|metadata| FileIdentity::of(&metadata));
        if now.is_some_and(|now| self.identity.same_file_as(&now)) {
            Ok(())
        } else {
            Err(SourceChanged {
                was: self.identity,
                now,
            })
        }
    }

    /// A paging hint over `range` of the mapping. Advisory: a refusal is ignored. Never `DONTNEED`.
    pub(super) fn advise(&self, range: Range<usize>, access: Access) {
        let advice = match access {
            Access::Sequential => Advice::Sequential,
            Access::WillNeed => Advice::WillNeed,
        };
        let start = range.start / MAX_PAGE * MAX_PAGE;
        let _ = self.map.advise_range(advice, start, range.end - start);
    }
}

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
//!    file in place while it is mapped. This is a contract, not a defended boundary. A write to the file
//!    while a `&[u8]` into it is live is a data race, which is undefined behavior by the language's rules
//!    whatever the hardware then does; a truncation ends the process with `SIGBUS`. The content checks of
//!    a packed payload (finite scales) run on the bytes of the one positional read that uploads them
//!    ([`MappedFile::read_at`]), so the bytes checked are the bytes staged. [`MappedFile::check_unchanged`]
//!    runs at the load end and on both sides of the load-time upload's reads, so a modification made
//!    before, or during, that upload is refused after it and the weights are dropped unpublished. The
//!    runtime reads of the mapping (embed rows, expert refills) are not followed by a check, so a
//!    modification racing one is not detected. Replacing a checkpoint by rename never modifies a mapped
//!    inode and is always safe.
//!
//! 5. The load path never touches the mapping: it reads by `pread` ([`MappedFile::read_at`], safe code),
//!    so a truncated file is a typed short-read error, not `SIGBUS`, and no page of the checkpoint is
//!    counted in this process's `RssFile` while it loads. [`MappedFile::evict`] is a
//!    `posix_fadvise(POSIX_FADV_DONTNEED)` on the descriptor: advice about the page cache, which changes no
//!    byte any slice or read observes.
//!
//! `MappedFile` is `Send + Sync` through the auto traits (`memmap2::Mmap` and `File` are); there is no
//! `unsafe impl`.

use std::fs::{File, Metadata};
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::{Mmap, MmapOptions};

use super::{FileIdentity, MapError, SourceChanged};

pub(super) struct MappedFile {
    /// `PROT_READ`, `MAP_PRIVATE`, offset 0, length `identity.len` at map time.
    map: Mmap,
    /// Kept open: [`Self::check_unchanged`] stats this descriptor, never the path.
    file: File,
    identity: FileIdentity,
    /// Bytes returned by [`Self::read_at`] since the map: the load's own read counter.
    read: AtomicU64,
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
            read: AtomicU64::new(0),
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

    /// Fill `out` from the file bytes at `offset` by `pread`, never through the mapping. A file shorter
    /// than `offset + out.len()` (truncated since the map) is an `UnexpectedEof` error.
    pub(super) fn read_at(&self, offset: usize, out: &mut [u8]) -> std::io::Result<()> {
        self.file.read_exact_at(out, offset as u64)?;
        self.read.fetch_add(out.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Bytes [`Self::read_at`] has returned.
    pub(super) fn bytes_read(&self) -> u64 {
        self.read.load(Ordering::Relaxed)
    }

    /// Ask the kernel to drop the page-cache pages of `range` of the file. Advisory: a refusal is
    /// ignored, and the kernel keeps partial pages and pages another process maps.
    pub(super) fn evict(&self, range: Range<usize>) {
        self.advise_cache(range, libc::POSIX_FADV_DONTNEED);
    }

    /// Ask the kernel to start reading `range` of the file into the page cache, without waiting for it.
    /// Advisory: a refusal is ignored.
    pub(super) fn prefetch(&self, range: Range<usize>) {
        self.advise_cache(range, libc::POSIX_FADV_WILLNEED);
    }

    fn advise_cache(&self, range: Range<usize>, advice: libc::c_int) {
        let (Ok(offset), Ok(len)) = (
            libc::off_t::try_from(range.start),
            libc::off_t::try_from(range.len()),
        ) else {
            return;
        };
        // SAFETY: invariant 5 of the module docs. `posix_fadvise` takes a valid open descriptor (the
        // `File` this value owns) and three integers; it reads and writes no memory of this process.
        let _ = unsafe { libc::posix_fadvise(self.file.as_raw_fd(), offset, len, advice) };
    }
}

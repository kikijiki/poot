//! [`StoredBytes`]: the one byte owner for everything read off a checkpoint (ADR-0103 amendment of
//! 2026-10-09; design: `dmap.md`).
//!
//! A `StoredBytes` is a view (a shared region plus a byte range) of one read-only private mapping of a
//! checkpoint file, so a tensor, or a GGUF expert (ADR-0109), is a slice of the mapping and nothing is
//! zero-filled or copied on the host. Bytes that are not a file's (fixtures, in-memory buffers, bytes a
//! loader derives) are an owned heap region of the same type. The region kinds are a private closed enum:
//! no other crate adds an owner with other invariants.
//!
//! Modifying a mapped checkpoint in place is unsupported. A modification already made is refused by
//! [`StoredBytes::check_unchanged`], an `fstat` identity check on the mapped descriptor that loaders and
//! binders run at their boundaries; one racing a read in flight is not detected (a truncation ends the
//! process with `SIGBUS` before any torn value is used). Replacing a checkpoint by rename is always safe.

#[allow(unsafe_code)]
mod mapping;

use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

use mapping::MappedFile;

/// `(st_dev, st_ino, st_size, st_nlink, st_mtime, st_ctime)` of a mapped file's descriptor. `ctime`
/// cannot be set from user space, so a write or truncate through any path changes it; `len` catches
/// every truncation regardless of timestamp granularity.
///
/// Unlinking or renaming over the file also moves its `ctime` (the link count changes) without touching
/// a byte of the inode, and replacing a checkpoint by rename is the supported way to update one. So
/// `ctime` is compared only while the link count is the one recorded at map time ([`Self::same_file_as`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    pub dev: u64,
    pub ino: u64,
    pub len: u64,
    pub nlink: u64,
    /// Nanoseconds since the epoch, saturating outside the range `i64` holds (about 1678 to 2262).
    pub mtime_ns: i64,
    pub ctime_ns: i64,
}

impl FileIdentity {
    /// Whether `now`, observed on the same descriptor, shows the file the identity was recorded from:
    /// same device, inode, length and `mtime`, and an unmoved `ctime` unless the link count changed.
    pub fn same_file_as(&self, now: &Self) -> bool {
        self.dev == now.dev
            && self.ino == now.ino
            && self.len == now.len
            && self.mtime_ns == now.mtime_ns
            && (self.nlink != now.nlink || self.ctime_ns == now.ctime_ns)
    }
}

/// A paging hint over a view ([`StoredBytes::advise`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Sequential,
    WillNeed,
}

/// [`StoredBytes::map`] refused a file.
#[derive(Debug)]
pub enum MapError {
    Io(std::io::Error),
    /// Not a regular file (a directory, FIFO or device): its length is not a stable extent to map.
    NotRegularFile {
        mode: u32,
    },
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "cannot map the file: {error}"),
            Self::NotRegularFile { mode } => {
                write!(
                    f,
                    "cannot map a file that is not a regular file (mode {mode:o})"
                )
            }
        }
    }
}

impl std::error::Error for MapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::NotRegularFile { .. } => None,
        }
    }
}

/// [`StoredBytes::slice`] asked for a range outside the view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceError {
    pub range: Range<usize>,
    pub len: usize,
}

impl fmt::Display for SliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "byte range {}..{} is outside a view of {} bytes",
            self.range.start, self.range.end, self.len
        )
    }
}

impl std::error::Error for SliceError {}

/// A mapped file's identity is no longer the one recorded when it was mapped: the file was modified in
/// place, replaced under the descriptor, or can no longer be stat-ed (`now` is `None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceChanged {
    pub was: FileIdentity,
    pub now: Option<FileIdentity>,
}

impl fmt::Display for SourceChanged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.now {
            Some(now) => write!(
                f,
                "a mapped checkpoint file changed while it was in use (was {:?}, now {now:?})",
                self.was
            ),
            None => write!(
                f,
                "a mapped checkpoint file can no longer be inspected (was {:?})",
                self.was
            ),
        }
    }
}

impl std::error::Error for SourceChanged {}

enum Region {
    /// Fixtures, in-memory sources, and bytes derived by a loader step.
    Heap(Box<[u8]>),
    /// One per checkpoint file.
    Mapped(MappedFile),
}

impl Region {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Heap(bytes) => bytes,
            Self::Mapped(mapped) => mapped.bytes(),
        }
    }
}

/// A view of bytes read off a checkpoint, or derived from it. Cloning is an `Arc` bump, never a byte copy.
#[derive(Clone)]
pub struct StoredBytes {
    region: Arc<Region>,
    /// Within `region.bytes()`; checked at construction.
    range: Range<usize>,
}

impl StoredBytes {
    /// Map all of `file` read-only and private. The `File` is kept for [`Self::check_unchanged`]. A
    /// zero-length file is an empty owned region (mapping length 0 is `EINVAL`), not an error.
    pub fn map(file: File) -> Result<Self, MapError> {
        let Some(mapped) = MappedFile::new(file)? else {
            return Ok(Self::from(Vec::new()));
        };
        let len = mapped.bytes().len();
        Ok(Self {
            region: Arc::new(Region::Mapped(mapped)),
            range: 0..len,
        })
    }

    /// A view of `range` of these bytes (relative to this view): the same region, no copy.
    pub fn slice(&self, range: Range<usize>) -> Result<Self, SliceError> {
        if range.start > range.end || range.end > self.len() {
            return Err(SliceError {
                range,
                len: self.len(),
            });
        }
        Ok(Self {
            region: Arc::clone(&self.region),
            range: self.range.start + range.start..self.range.start + range.end,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.region.bytes()[self.range.clone()]
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }

    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// `Ok` for an owned region; for a mapped one, the file's identity now equals the one at map time.
    pub fn check_unchanged(&self) -> Result<(), SourceChanged> {
        match &*self.region {
            Region::Heap(_) => Ok(()),
            Region::Mapped(mapped) => mapped.check_unchanged(),
        }
    }

    /// A paging hint over this view; a no-op on an owned region. Advisory: failures are ignored.
    pub fn advise(&self, access: Access) {
        if let Region::Mapped(mapped) = &*self.region {
            mapped.advise(self.range.clone(), access);
        }
    }

    /// Whether these bytes are a view of a file mapping (and so can change under the process), not an
    /// owned region.
    pub(crate) fn is_mapped(&self) -> bool {
        matches!(&*self.region, Region::Mapped(_))
    }

    /// Whether two views share one region: one identity check per file, not per tensor.
    pub fn same_region(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.region, &other.region)
    }
}

impl From<Vec<u8>> for StoredBytes {
    fn from(bytes: Vec<u8>) -> Self {
        let region = Region::Heap(bytes.into_boxed_slice());
        let len = region.bytes().len();
        Self {
            region: Arc::new(region),
            range: 0..len,
        }
    }
}

impl PartialEq for StoredBytes {
    /// Content equality; the same region and range is equal without comparing.
    fn eq(&self, other: &Self) -> bool {
        (self.same_region(other) && self.range == other.range)
            || self.as_slice() == other.as_slice()
    }
}

impl Eq for StoredBytes {}

impl fmt::Debug for StoredBytes {
    /// The region kind, a path-free identity and the range; never the bytes.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("StoredBytes");
        match &*self.region {
            Region::Heap(_) => debug.field("region", &"heap"),
            Region::Mapped(mapped) => debug.field("region", &mapped.identity()),
        };
        debug.field("range", &self.range).finish()
    }
}

const _: fn() = || {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<StoredBytes>();
};

#[cfg(test)]
mod tests {
    use super::*;

    /// `slice` is relative to the view it is called on, shares the region, and refuses a range outside it
    /// as the typed `SliceError`. Mutation: check against the region instead of the view; the nested
    /// out-of-range slice is admitted and the row goes red.
    #[test]
    fn slices_are_views_relative_to_their_parent_and_bounded_by_it() {
        let whole = StoredBytes::from(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        let middle = whole.slice(2..6).unwrap();
        assert_eq!(middle.as_slice(), &[2, 3, 4, 5]);
        assert!(middle.same_region(&whole));
        let inner = middle.slice(1..3).unwrap();
        assert_eq!(inner.as_slice(), &[3, 4]);
        assert_eq!(
            middle.slice(3..5),
            Err(SliceError {
                range: 3..5,
                len: 4
            })
        );
        let (start, end) = (3, 2);
        assert!(
            middle.slice(start..end).is_err(),
            "a reversed range is refused"
        );
        assert!(middle.slice(4..4).unwrap().is_empty());
    }

    /// Equality is content, not identity; `Debug` names the range and never the bytes.
    #[test]
    fn equality_is_content_and_debug_hides_the_bytes() {
        let a = StoredBytes::from(vec![9u8, 8, 7]);
        let b = StoredBytes::from(vec![0u8, 9, 8, 7]).slice(1..4).unwrap();
        assert_eq!(a, b);
        assert!(!a.same_region(&b));
        assert_ne!(a, StoredBytes::from(vec![9u8, 8, 6]));
        let shown = format!("{a:?}");
        assert!(shown.contains("heap") && shown.contains("0..3"), "{shown}");
        assert!(!shown.contains('9'), "the bytes are never printed: {shown}");
    }
}

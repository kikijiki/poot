//! [`StoredBytes`]: the one byte owner for everything read off a checkpoint (ADR-0103 amendment of
//! 2026-10-09; design: `dmap.md`).
//!
//! A `StoredBytes` is a view (a shared region plus a byte range) of one read-only private mapping of a
//! checkpoint file, so a tensor, or a GGUF expert (ADR-0109), is a slice of the mapping and nothing is
//! zero-filled or copied on the host. The load reads a view by [`StoredBytes::read_at`] (`pread`) into the
//! device's staging, so the mapping is only touched by the runtime host reads (embed rows, expert refills). Bytes that are not a file's (fixtures, in-memory buffers, bytes a
//! loader derives) are an owned heap region of the same type. The region kinds are a private closed enum:
//! no other crate adds an owner with other invariants.
//!
//! Modifying a mapped checkpoint in place is unsupported, and the unsupported case is undefined behavior,
//! not a defined failure, for a read through the mapping (a write while a slice is live is a data race). The
//! load-time upload reads by [`StoredBytes::read_at`] instead, where a changed file is a changed value or a
//! typed short read, never undefined behavior, and the content checks run on the bytes that read stages (see
//! `mapping`, invariant 4). A modification already made is refused by [`StoredBytes::check_unchanged`], an
//! `fstat` identity check on the mapped descriptor that loaders and binders run at their boundaries. The
//! load-time upload of a weight runs it before and after its reads, so a file modified while that upload
//! read it is refused and the entry is not published; the runtime reads of the mapping (embed rows, expert
//! refills) have no check after them, so a modification racing one is not detected (a truncation ends the
//! process with `SIGBUS`). Replacing a checkpoint by rename is always safe.

#[expect(unsafe_code)]
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
    /// Device number (`st_dev`) the file lives on.
    pub dev: u64,
    /// Inode number (`st_ino`).
    pub ino: u64,
    /// File length in bytes (`st_size`) at observation time.
    pub len: u64,
    /// Hard-link count (`st_nlink`); a change excuses a moved `ctime`.
    pub nlink: u64,
    /// Nanoseconds since the epoch, saturating outside the range `i64` holds (about 1678 to 2262).
    pub mtime_ns: i64,
    /// Inode change time in nanoseconds since the epoch, saturating like `mtime_ns`; user space cannot set it.
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

/// [`StoredBytes::map`] refused a file.
#[derive(Debug)]
pub enum MapError {
    /// Reading the file's metadata or mapping it failed.
    Io(std::io::Error),
    /// Not a regular file (a directory, FIFO or device): its length is not a stable extent to map.
    NotRegularFile {
        /// The file's `st_mode` bits, as returned by `stat`.
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

/// [`StoredBytes::reader`]: a sequential [`std::io::Read`] over a view.
pub struct StoredReader<'a> {
    bytes: &'a StoredBytes,
    position: usize,
}

impl std::io::Read for StoredReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = buf.len().min(self.bytes.len() - self.position);
        self.bytes
            .read_at(self.position..self.position + n, &mut buf[..n])
            .map_err(|error| match error {
                ReadError::Io(error) => error,
                ReadError::Range(error) => std::io::Error::other(error),
            })?;
        self.position += n;
        Ok(n)
    }
}

/// Fill `out`, the bytes `start..start + out.len()` of some source, with `threads` readers at once: `out` is
/// cut into parts of whole MiB, each part is read by `read(range, part)` on its own scoped thread, and the
/// first failure in part order is returned. One thread (or a `threads` of 0 or 1) reads it whole on the
/// calling thread. A reader that panics panics the caller.
pub fn read_split<E: Send>(
    out: &mut [u8],
    start: usize,
    threads: usize,
    read: impl Fn(Range<usize>, &mut [u8]) -> Result<(), E> + Sync,
) -> Result<(), E> {
    if threads <= 1 {
        return read(start..start + out.len(), out);
    }
    let part = out.len().div_ceil(threads).next_multiple_of(1 << 20);
    let read = &read;
    std::thread::scope(|scope| {
        let readers: Vec<_> = out
            .chunks_mut(part)
            .enumerate()
            .map(|(index, piece)| {
                let at = start + index * part;
                scope.spawn(move || read(at..at + piece.len(), piece))
            })
            .collect();
        readers
            .into_iter()
            .try_for_each(|reader| reader.join().expect("a reader does not panic"))
    })
}

/// [`StoredBytes::read_at`] could not deliver a range.
#[derive(Debug)]
pub enum ReadError {
    /// The range is outside the view.
    Range(SliceError),
    /// The file read failed or came back short (truncated since it was mapped).
    Io(std::io::Error),
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range(error) => error.fmt(f),
            Self::Io(error) => write!(f, "cannot read the checkpoint file: {error}"),
        }
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Range(error) => Some(error),
            Self::Io(error) => Some(error),
        }
    }
}

/// [`StoredBytes::slice`] asked for a range outside the view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceError {
    /// The requested byte range, relative to the view.
    pub range: Range<usize>,
    /// Length of the view the range was checked against.
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
    /// The identity recorded when the file was mapped.
    pub was: FileIdentity,
    /// The identity observed now, or `None` when the descriptor can no longer be stat-ed.
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

    /// The view's bytes. For a mapped region the borrow is only as stable as the file; see [`Self::check_unchanged`].
    pub fn as_slice(&self) -> &[u8] {
        &self.region.bytes()[self.range.clone()]
    }

    /// Length of the view in bytes.
    pub fn len(&self) -> usize {
        self.range.len()
    }

    /// Whether the view holds no bytes.
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

    /// Copy `range` of this view (relative to it) into `out` (`out.len() == range.len()`): a positional
    /// read of the file for a mapped region, never a touch of the mapping, so the checkpoint is not
    /// counted in the process's `RssFile`; a plain copy for an owned one. A range outside the view is
    /// [`ReadError::Range`]; a file that no longer holds the range is [`ReadError::Io`].
    pub fn read_at(&self, range: Range<usize>, out: &mut [u8]) -> Result<(), ReadError> {
        assert_eq!(out.len(), range.len(), "the output is the range's length");
        if range.start > range.end || range.end > self.len() {
            return Err(ReadError::Range(SliceError {
                range,
                len: self.len(),
            }));
        }
        let start = self.range.start + range.start;
        match &*self.region {
            Region::Heap(bytes) => {
                out.copy_from_slice(&bytes[start..start + out.len()]);
                Ok(())
            }
            Region::Mapped(mapped) => mapped.read_at(start, out).map_err(ReadError::Io),
        }
    }

    /// The view's bytes as an [`std::io::Read`], by positional reads of the same descriptor (the same inode
    /// the mapping and every later read use), so hashing a mapped checkpoint maps none of it in.
    pub fn reader(&self) -> StoredReader<'_> {
        StoredReader {
            bytes: self,
            position: 0,
        }
    }

    /// Drop the page-cache pages of `range` of this view (relative to it; a range past the end is
    /// clamped) after its bytes were consumed: on a unified-memory device the checkpoint's cache and the
    /// weights it was copied into are the same RAM. Advisory, a no-op on an owned region; a later read of
    /// the range reads the disk again with the same bytes.
    pub fn evict(&self, range: Range<usize>) {
        if let Region::Mapped(mapped) = &*self.region {
            let end = range.end.min(self.len());
            let start = range.start.min(end);
            mapped.evict(self.range.start + start..self.range.start + end);
        }
    }

    /// Ask the kernel to read `range` of this view (relative to its start) into the page cache ahead of a
    /// [`Self::read_at`] of it, so a cold sequential load keeps the disk busy while the reader works on the
    /// bytes before. The range may run past the view's end: it is clamped to the file, and the bytes after a
    /// tensor are the next tensor's. Advisory, non-blocking, a no-op on an owned region.
    pub fn prefetch(&self, range: Range<usize>) {
        if let Region::Mapped(mapped) = &*self.region {
            let file_len = mapped.bytes().len();
            let end = (self.range.start + range.end).min(file_len);
            let start = (self.range.start + range.start).min(end);
            mapped.prefetch(start..end);
        }
    }

    /// Bytes [`Self::read_at`] has read from this view's file since it was mapped (every view of the file
    /// shares the count); `0` for an owned region.
    pub fn bytes_read(&self) -> u64 {
        match &*self.region {
            Region::Heap(_) => 0,
            Region::Mapped(mapped) => mapped.bytes_read(),
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

    /// POOT-1416: `read_at` is a positional file read, not a touch of the mapping: it returns the file's
    /// bytes for any sub-range of any view, counts them once on the file's counter (shared by every view),
    /// refuses a range outside the view as `Range`, and reports a file truncated after the map as an
    /// `UnexpectedEof` I/O error.
    ///
    /// Mutation: make the `Mapped` arm of `read_at` copy from `as_slice` (the mapping); the truncated read
    /// ends the process with `SIGBUS` and this row dies.
    #[test]
    fn read_at_reads_the_file_counts_once_and_reports_a_truncation() {
        let path = poot_test_util::unique_temp_path("read_at.bin");
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i % 253) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let whole = StoredBytes::map(File::open(&path).unwrap()).unwrap();
        let view = whole.slice(100..4000).unwrap();
        let mut out = vec![0u8; 50];
        view.read_at(10..60, &mut out).unwrap();
        assert_eq!(out, bytes[110..160]);
        let mut again = vec![0u8; 5];
        whole.read_at(0..5, &mut again).unwrap();
        assert_eq!(again, bytes[..5]);
        assert_eq!((whole.bytes_read(), view.bytes_read()), (55, 55));
        assert!(matches!(
            view.read_at(3890..3910, &mut [0u8; 20]),
            Err(ReadError::Range(_))
        ));
        assert_eq!(StoredBytes::from(vec![1u8, 2, 3]).bytes_read(), 0);

        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(1000)
            .unwrap();
        match whole.read_at(900..1100, &mut [0u8; 200]) {
            Err(ReadError::Io(error)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
            }
            other => panic!("expected a short read, got {other:?}"),
        }
    }

    /// `reader()` is the view as a sequential `Read`, from the file, ending at the view's end.
    #[test]
    fn the_reader_streams_the_view_and_stops_at_its_end() {
        use std::io::Read;
        let path = poot_test_util::unique_temp_path("reader.bin");
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let whole = StoredBytes::map(File::open(&path).unwrap()).unwrap();
        let mut streamed = Vec::new();
        whole
            .slice(10..990)
            .unwrap()
            .reader()
            .read_to_end(&mut streamed)
            .unwrap();
        assert_eq!(streamed, bytes[10..990]);
    }

    /// `read_split` cuts `out` into whole-MiB parts, hands each its absolute range, and reports the first
    /// failure in part order. The control is one thread, which reads the whole range in one call.
    ///
    /// Mutation: pass `at = start` for every part (forget the part's offset); the filled pattern is wrong
    /// and this row goes red.
    #[test]
    fn read_split_covers_the_range_exactly_and_reports_the_first_failure() {
        let fill = |range: Range<usize>, part: &mut [u8]| -> Result<(), usize> {
            for (offset, byte) in range.clone().zip(part.iter_mut()) {
                *byte = (offset % 251) as u8;
            }
            Ok(())
        };
        let len = 5 << 20;
        let start = 1000;
        let expected: Vec<u8> = (start..start + len).map(|i| (i % 251) as u8).collect();
        for threads in [0, 1, 4, 16] {
            let mut out = vec![0u8; len];
            read_split(&mut out, start, threads, fill).unwrap();
            assert!(out == expected, "{threads} threads filled the wrong bytes");
        }
        // 5 MiB over 4 threads is parts of 2 MiB at 0, 2 and 4: the last two fail, the first of them is
        // the error reported.
        let failing = |range: Range<usize>, _: &mut [u8]| -> Result<(), usize> {
            if range.start >= 2 << 20 {
                Err(range.start)
            } else {
                Ok(())
            }
        };
        let mut out = vec![0u8; len];
        assert_eq!(read_split(&mut out, 0, 4, failing), Err(2 << 20));
    }
}

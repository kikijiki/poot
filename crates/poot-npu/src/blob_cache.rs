//! [`BlobCache`]: compiled graphs kept on disk, so a region the vendor compiler already built loads without
//! compiling again.
//!
//! A blob is only valid for the exact region and compiler that produced it, so a cache entry is keyed by the
//! full identity (POOT-ADR-0112 decision 8): the target part, the vendor compiler's version, the compiler
//! option string, and the region's OpenVINO text and constants (which carry every shape and dtype). The key
//! is stored inside the entry, and a lookup compares it whole, so a hash collision is a miss and never a wrong
//! blob. A digest of the blob is stored beside it and checked on every lookup: the driver does not validate a
//! native blob, so a truncated or corrupted file must be a miss before it reaches the device.

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use poot_runtime_common::VpuGraphArtifact;
use poot_target::VendorCompilerId;

/// The magic that opens every cache file; a different layout would change it.
const MAGIC: &[u8] = b"poot-vpu-blob-v2\n";

/// The length of the blake3 digest stored between the key and the blob.
const DIGEST_LEN: usize = blake3::OUT_LEN;

/// The identity of a compiled region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlobKey {
    text: String,
}

impl BlobKey {
    /// The identity of compiling `artifact` with `compiler` under `options`.
    pub(crate) fn new(
        artifact: &VpuGraphArtifact,
        compiler: VendorCompilerId,
        options: &str,
    ) -> Self {
        let digest = |bytes: &[u8]| blake3::hash(bytes).to_hex();
        let text = format!(
            "{}\n{}\noptions={options}\nxml={}\nweights={}\n",
            artifact.target().canonical(),
            compiler.canonical(),
            digest(artifact.xml()),
            digest(artifact.weights()),
        );
        Self { text }
    }

    fn file_name(&self) -> String {
        format!(
            "{}.blob",
            &blake3::hash(self.text.as_bytes()).to_hex()[..32]
        )
    }
}

/// A directory of compiled blobs, or nothing when caching is off.
pub(crate) struct BlobCache {
    dir: Option<PathBuf>,
}

impl BlobCache {
    pub(crate) fn new(dir: Option<PathBuf>) -> Self {
        Self { dir }
    }

    /// The blob compiled for exactly `key`, if the cache holds one.
    pub(crate) fn get(&self, key: &BlobKey) -> io::Result<Option<Vec<u8>>> {
        let Some(dir) = &self.dir else {
            return Ok(None);
        };
        let bytes = match fs::read(dir.join(key.file_name())) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(decode(key, &bytes).map(<[u8]>::to_vec))
    }

    /// Keep `blob` as the compile of `key`. The entry is written whole under a temporary name and renamed
    /// into place, so a reader never sees half of one.
    pub(crate) fn put(&self, key: &BlobKey, blob: &[u8]) -> io::Result<()> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        fs::create_dir_all(dir)?;
        let name = key.file_name();
        // Unique per write: two devices of one process writing the same key never share a temporary file.
        static WRITES: AtomicU64 = AtomicU64::new(0);
        let temp = dir.join(format!(
            "{name}.{}.{}.tmp",
            std::process::id(),
            WRITES.fetch_add(1, Ordering::Relaxed)
        ));
        let written = (|| {
            let mut file = fs::File::create(&temp)?;
            file.write_all(MAGIC)?;
            file.write_all(&(key.text.len() as u64).to_le_bytes())?;
            file.write_all(key.text.as_bytes())?;
            file.write_all(blake3::hash(blob).as_bytes())?;
            file.write_all(blob)?;
            file.sync_all()?;
            fs::rename(&temp, dir.join(name))
        })();
        if written.is_err() {
            // A failed write leaves no half entry behind; the original error is the one reported.
            let _ = fs::remove_file(&temp);
        }
        written
    }
}

/// The blob inside a cache file, when the file is whole, its stored key is `key` and the blob matches its digest.
fn decode<'a>(key: &BlobKey, bytes: &'a [u8]) -> Option<&'a [u8]> {
    let rest = bytes.strip_prefix(MAGIC)?;
    let (len, rest) = rest.split_first_chunk::<8>()?;
    let len = usize::try_from(u64::from_le_bytes(*len)).ok()?;
    let stored = rest.get(..len)?;
    if stored != key.text.as_bytes() {
        return None;
    }
    let (digest, blob) = rest[len..].split_first_chunk::<DIGEST_LEN>()?;
    (blake3::hash(blob).as_bytes() == digest).then_some(blob)
}

#[cfg(test)]
mod tests {
    use poot_runtime_common::{VpuBinding, VpuLayer};
    use poot_target::{BufferStorage, VpuTargetId};

    use super::*;

    fn key() -> BlobKey {
        let artifact = VpuGraphArtifact::new(
            VpuTargetId::Npu4,
            b"<net/>".to_vec(),
            Vec::new(),
            vec![VpuBinding::new("a", BufferStorage::f16(), vec![4])],
            vec![VpuBinding::new("v1", BufferStorage::f16(), vec![4])],
            vec![VpuLayer::new("v1", 1)],
        )
        .unwrap();
        BlobKey::new(&artifact, VendorCompilerId { major: 8, minor: 3 }, "")
    }

    fn files(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<_> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    /// A failed write leaves no temporary file behind and reports the failure: here the entry's final name is
    /// taken by a directory, so the rename fails after the data was written.
    ///
    /// Mutation: drop the `remove_file` of the temporary in `put`; the `.tmp` stays in the directory.
    #[test]
    fn a_failed_put_removes_its_temporary_file() {
        let dir = std::env::temp_dir().join(format!("poot-blob-put-fail-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = BlobCache::new(Some(dir.clone()));
        let key = key();
        fs::create_dir_all(dir.join(key.file_name())).unwrap();
        assert!(cache.put(&key, b"blob").is_err());
        assert_eq!(
            files(&dir),
            [key.file_name()],
            "only the blocking directory remains"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Concurrent writers of one key never share a temporary file: eight threads put different whole payloads
    /// under the same key and every put succeeds; the cache then holds exactly one entry, and it is one writer's
    /// payload whole, never a mixture.
    ///
    /// Mutation: make the temporary name constant in `put`; writers truncate each other's file or rename a
    /// file another writer already moved, and a put fails or the entry is mixed.
    #[test]
    fn concurrent_puts_of_one_key_leave_one_whole_entry() {
        let dir = std::env::temp_dir().join(format!("poot-blob-put-ok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = BlobCache::new(Some(dir.clone()));
        let key = key();
        std::thread::scope(|scope| {
            for writer in 0..8u8 {
                let (cache, key) = (&cache, &key);
                scope.spawn(move || {
                    for _ in 0..25 {
                        cache
                            .put(key, &vec![writer; 1 << 20])
                            .expect("a put succeeds");
                    }
                });
            }
        });
        assert_eq!(files(&dir), [key.file_name()]);
        let blob = cache.get(&key).unwrap().expect("the entry is whole");
        assert_eq!(blob.len(), 1 << 20);
        assert!(
            blob.iter().all(|&b| b == blob[0]),
            "one writer's payload, not a mixture"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A flipped byte in the blob, or a file cut short, is a miss: the driver does not validate a native blob.
    ///
    /// Mutation: return the blob without comparing its digest in `decode`; the corrupted entry is returned.
    #[test]
    fn a_corrupted_or_truncated_blob_is_a_miss() {
        let dir = std::env::temp_dir().join(format!("poot-blob-corrupt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = BlobCache::new(Some(dir.clone()));
        let key = key();
        cache.put(&key, b"a whole blob").unwrap();
        assert_eq!(
            cache.get(&key).unwrap().as_deref(),
            Some(&b"a whole blob"[..])
        );
        let path = dir.join(key.file_name());
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&path, &bytes).unwrap();
        assert_eq!(cache.get(&key).unwrap(), None, "a flipped byte is a miss");
        bytes.truncate(bytes.len() - 6);
        fs::write(&path, &bytes).unwrap();
        assert_eq!(cache.get(&key).unwrap(), None, "a truncated blob is a miss");
        let _ = fs::remove_dir_all(&dir);
    }
}

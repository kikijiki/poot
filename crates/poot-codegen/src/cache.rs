//! Persistent, epoch-scoped on-disk root for compiled-kernel artifact caches (card 211).
//!
//! The ROCm HSACO disk cache is content-addressed by `fnv1a(Body)` + arch (cards 211/226,
//! `RocmGraphExecutor::artifact_name`); `compile()` shells out to the toolchain only when that path does
//! not exist. This module gives the cache a stable root: `nix develop -c` assigns a fresh `TMPDIR` per
//! invocation, so a cache under `std::env::temp_dir()` was never reused across processes.
//!
//! A stable root would keep stale artifacts across toolchain bumps and emitter edits, and a stale HSACO
//! silently runs the wrong kernel (a correctness bug), so the toolchain/codegen epoch below is mandatory.

use std::collections::HashMap;
use std::io::Read;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use poot_kernel_ir::Body;

use crate::{CompileError, Target, Toolchain, compile};

/// The content fingerprint of a kernel `Body` (a hash of its `Debug` form): the Body half of every
/// [`KernelKey`], shared by the wgpu and PTX caches.
pub fn body_fingerprint(body: &Body) -> u64 {
    poot_runtime_common::fnv1a_bytes(format!("{body:?}").as_bytes())
}

/// The `poot-codegen` source-tree hash, computed at build time by `build.rs` from `src/**/*.rs` and
/// exported as `POOT_CODEGEN_SRC_HASH`. Folding it into the epoch makes an emitter change (e.g. to
/// `src/emit.rs`) invalidate the cache; the `Body`+arch half of the key cannot see such a change.
const CODEGEN_SRC_HASH: &str = env!("POOT_CODEGEN_SRC_HASH");

/// Resolve which compiler binary [`compile`] would invoke for `target`, without running it. `"unset"` when the
/// target's toolchain is not configured: `compile` fails for it, so the epoch only needs a stable value.
fn resolve_tool(target: Target) -> String {
    Toolchain::from_env()
        .binary(target)
        .map_or_else(|_| "unset".to_string(), str::to_owned)
}

/// First line of `<tool> --version`, or `"unknown"` if the tool cannot be run. The epoch needs a value in
/// that case, and "unknown" becoming a real version later is a correct epoch change. Memoized per tool
/// string for the life of the process (a toolchain is not swapped under a running process), so opening a
/// [`KernelCache`] does not spawn the compiler each time.
fn tool_version(tool: &str) -> String {
    static VERSIONS: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(Mutex::default);
    let mut versions = VERSIONS.lock().unwrap_or_else(|e| e.into_inner());
    versions
        .entry(tool.to_owned())
        .or_insert_with(|| query_tool_version(tool))
        .clone()
}

fn query_tool_version(tool: &str) -> String {
    Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// The canonical, newline-joined epoch string for `target` (card 211, "Cache key"). Separate from
/// [`epoch`] so the `EPOCH` sidecar file can hold the readable form.
fn epoch_canonical(target: Target) -> String {
    let tool = resolve_tool(target);
    let tool_version = tool_version(&tool);
    let device_libs = std::env::var("POOT_ROCM_DEVICE_LIBS").unwrap_or_default();
    let flags = target.llc_args().join(" ");
    format!(
        "poot-kernel-cache-v1\ncodegen_src={CODEGEN_SRC_HASH}\ntool={tool}\ntool_version={tool_version}\ndevice_libs={device_libs}\nflags={flags}"
    )
}

/// The toolchain/codegen epoch for `target`: a 16-hex-digit fingerprint of the compiler binary and
/// version, linked device libs, codegen flags, and `poot-codegen`'s own source. Artifacts from a
/// different epoch are never consulted (see [`kernel_cache_root`]).
pub fn epoch(target: Target) -> String {
    format!(
        "{:016x}",
        poot_runtime_common::fnv1a_bytes(epoch_canonical(target).as_bytes())
    )
}

/// `POOT_KERNEL_CACHE=0` (also `off`/`false`, case-insensitive) disables the shared cache, without a
/// rebuild.
fn kill_switch_active() -> bool {
    match std::env::var("POOT_KERNEL_CACHE") {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false"),
        Err(_) => false,
    }
}

/// Base directory for the shared cache (before the `<epoch>/<mcpu>` split), independent of `target`.
/// Order: `$POOT_KERNEL_CACHE_DIR` verbatim, else `$XDG_CACHE_HOME/poot/kernels`, else
/// `$HOME/.cache/poot/kernels`, else `temp_dir()/poot-kernels`.
fn resolve_base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("POOT_KERNEL_CACHE_DIR")
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return PathBuf::from(xdg).join("poot").join("kernels");
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return PathBuf::from(home)
            .join(".cache")
            .join("poot")
            .join("kernels");
    }
    std::env::temp_dir().join("poot-kernels")
}

/// Resolve the on-disk root for a persistent, epoch-scoped kernel artifact cache and ensure it exists.
/// `kind` names the caller's kernel family (e.g. `"rocm"`) so backends sharing a base directory do not
/// collide on the same epoch directory.
///
/// Returns `<base>/<kind>/<epoch>/<mcpu-or-target>`; the content-addressed filename is still
/// `RocmGraphExecutor::artifact_name`'s job. The directory is created (best effort) with a plaintext
/// `EPOCH` sidecar on first creation, so a mismatch can be diagnosed by hand.
///
/// `POOT_KERNEL_CACHE=0`/`off`/`false` bypasses all of this and returns a per-process directory under
/// `temp_dir()` (still a real directory, since `compile()` needs an output path).
///
/// Never hard-fails: if the resolved base cannot be created or written, falls back to
/// `temp_dir()/poot-<kind>` and logs a `tracing::warn!` (`HOME` may be unusable on RunPod images and
/// containers).
pub fn kernel_cache_root(kind: &str, target: Target) -> PathBuf {
    if kill_switch_active() {
        return std::env::temp_dir().join(format!("poot-{kind}-{}", std::process::id()));
    }

    let epoch_hex = epoch(target);
    let arch_component = match target {
        Target::AmdGcn(arch) => arch.mcpu().to_string(),
        other => format!("{other:?}"),
    };
    let base = resolve_base_dir().join(kind);
    let epoch_dir = base.join(&epoch_hex);
    let dir = epoch_dir.join(&arch_component);

    match std::fs::create_dir_all(&dir) {
        Ok(()) => {
            let epoch_file = epoch_dir.join("EPOCH");
            if !epoch_file.exists() {
                // Best effort: a diagnostic sidecar; a failed write must not stop compilation.
                let _ = std::fs::write(&epoch_file, epoch_canonical(target));
            }
            dir
        }
        Err(error) => {
            tracing::warn!(
                ?dir,
                %error,
                "could not create persistent kernel cache dir; falling back to a temp_dir root"
            );
            let fallback = std::env::temp_dir().join(format!("poot-{kind}"));
            let _ = std::fs::create_dir_all(&fallback);
            fallback
        }
    }
}

/// The per-target directory family under the shared base, so backends sharing a base directory do not
/// collide. AMDGCN matches the `kind` the ROCm executor passes to [`kernel_cache_root`] directly.
fn cache_kind(target: Target) -> &'static str {
    match target {
        Target::Nvptx => "ptx",
        Target::SpirvVulkan => "spirv",
        Target::AmdGcn(_) => "rocm",
        Target::AieCore => "aie",
    }
}

/// Every cache entry starts with this magic, then the little-endian [`entry_digest`], then the artifact.
const ENTRY_MAGIC: &[u8; 8] = b"POOTKC01";
const ENTRY_HEADER_LEN: usize = ENTRY_MAGIC.len() + size_of::<u64>();

/// The name of one cache entry: the codegen epoch, the target and the Body fingerprint. Two kernels share
/// a key only if the same toolchain and emitter would produce the same artifact from the same Body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelKey {
    name: String,
    fingerprint: u64,
}

impl KernelKey {
    /// The entry's file name (`k<epoch>-<body fingerprint>.<ext>`).
    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// The Body fingerprint half of the key, as [`body_fingerprint`] computed it.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// The one key function every backend's disk cache uses. Pure: it reads no environment and touches no disk,
/// so the epoch and target it is given are exactly what scopes the entry.
pub fn kernel_entry_key(epoch: &str, target: Target, fingerprint: u64) -> KernelKey {
    KernelKey {
        name: format!("k{epoch}-{fingerprint:016x}.{}", target.output_ext()),
        fingerprint,
    }
}

/// FNV-1a over the key name and the artifact bytes. Binding the name means an entry copied or renamed to
/// another key (say, out of an older epoch's directory) fails its check instead of being loaded.
fn entry_digest(key: &KernelKey, artifact: &[u8]) -> u64 {
    let mut bytes = Vec::with_capacity(key.name.len() + artifact.len());
    bytes.extend_from_slice(key.name.as_bytes());
    bytes.extend_from_slice(artifact);
    poot_runtime_common::fnv1a_bytes(&bytes)
}

/// A compiled artifact is longer than the caller's `max_artifact_bytes`. Raised before the bytes are
/// loaded into host memory, handed to a runtime or published to the cache; an external compiler's own
/// memory and time are outside what this limit bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("kernel artifact of {bytes} bytes exceeds the {limit}-byte limit")]
pub struct ArtifactTooLarge {
    /// The artifact's length, or the first length past the limit when the source was read bounded.
    pub bytes: u64,
    pub limit: u64,
}

impl ArtifactTooLarge {
    /// `Ok` when an artifact of `bytes` fits `limit`, else the refusal.
    fn check(bytes: u64, limit: NonZeroU64) -> Result<(), Self> {
        if bytes > limit.get() {
            return Err(Self {
                bytes,
                limit: limit.get(),
            });
        }
        Ok(())
    }
}

/// Read a whole file of at most `limit` bytes. The length is checked from metadata first and the read
/// itself is bounded one byte past the limit, so a file that grows or lies about its size is still
/// refused without being held in memory.
fn read_bounded(path: &Path, limit: u64) -> std::io::Result<Result<Vec<u8>, ArtifactTooLarge>> {
    let file = std::fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length > limit {
        return Ok(Err(ArtifactTooLarge {
            bytes: length,
            limit,
        }));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Ok(Err(ArtifactTooLarge {
            bytes: bytes.len() as u64,
            limit,
        }));
    }
    Ok(Ok(bytes))
}

/// A kernel artifact and whether this call had to build it (a compile, as opposed to a verified hit).
#[derive(Debug)]
pub struct Artifact {
    pub bytes: Vec<u8>,
    pub built: bool,
}

/// One target's persistent kernel artifact cache, scoped by the codegen [`epoch`] both by directory (an
/// old epoch's directory is never consulted) and by key (an entry carries its epoch in its name and its
/// digest). Every entry is verified on read: a truncated, corrupted or misplaced file is a miss, never a
/// load. Shared by the wgpu and PTX backends; entries are content-addressed, so concurrent
/// processes writing the same key write the same bytes.
#[derive(Debug, Clone)]
pub struct KernelCache {
    dir: PathBuf,
    epoch: String,
    target: Target,
}

impl KernelCache {
    /// The cache under the shared persistent root for `target` (see [`kernel_cache_root`]).
    pub fn open(target: Target) -> Self {
        Self::with_epoch(
            kernel_cache_root(cache_kind(target), target),
            epoch(target),
            target,
        )
    }

    /// The cache rooted at an explicit directory instead of the shared one, keyed by `target`'s epoch.
    pub fn at(dir: impl Into<PathBuf>, target: Target) -> Self {
        Self::with_epoch(dir.into(), epoch(target), target)
    }

    fn with_epoch(dir: PathBuf, epoch: String, target: Target) -> Self {
        Self { dir, epoch, target }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The entry key for `body` under this cache's epoch and target.
    pub fn key(&self, body: &Body) -> KernelKey {
        kernel_entry_key(&self.epoch, self.target, body_fingerprint(body))
    }

    fn entry_path(&self, key: &KernelKey) -> PathBuf {
        self.dir.join(key.as_str())
    }

    /// The verified artifact for `key`, or `None` on a miss. A file that fails the magic or digest check is
    /// a miss (logged), so the caller rebuilds and [`store`](Self::store) replaces it. An entry whose
    /// artifact is longer than `max_artifact_bytes` is [`ArtifactTooLarge`], never read whole: a rebuild
    /// would produce the same oversize artifact, so it is not a miss either.
    pub fn load(
        &self,
        key: &KernelKey,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<Option<Vec<u8>>, ArtifactTooLarge> {
        let path = self.entry_path(key);
        let entry_limit = max_artifact_bytes
            .get()
            .saturating_add(ENTRY_HEADER_LEN as u64);
        let mut bytes = match read_bounded(&path, entry_limit) {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(oversize)) => {
                return Err(ArtifactTooLarge {
                    bytes: oversize.bytes.saturating_sub(ENTRY_HEADER_LEN as u64),
                    limit: max_artifact_bytes.get(),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                tracing::warn!(?path, %error, "kernel cache entry unreadable; treating as a miss");
                return Ok(None);
            }
        };
        let header_ok = bytes.len() >= ENTRY_HEADER_LEN && bytes.starts_with(ENTRY_MAGIC);
        let recorded = header_ok.then(|| {
            u64::from_le_bytes(
                bytes[ENTRY_MAGIC.len()..ENTRY_HEADER_LEN]
                    .try_into()
                    .expect("the header slice is 8 bytes"),
            )
        });
        if recorded
            != Some(entry_digest(
                key,
                &bytes[ENTRY_HEADER_LEN.min(bytes.len())..],
            ))
        {
            tracing::warn!(
                ?path,
                "kernel cache entry failed its digest check; rebuilding"
            );
            return Ok(None);
        }
        bytes.drain(..ENTRY_HEADER_LEN);
        Ok(Some(bytes))
    }

    /// Publish `artifact` under `key`: written to a unique temp file beside the entry, then renamed onto
    /// it, so a concurrent reader never sees a partial file.
    pub fn store(&self, key: &KernelKey, artifact: &[u8]) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let mut entry = Vec::with_capacity(ENTRY_HEADER_LEN + artifact.len());
        entry.extend_from_slice(ENTRY_MAGIC);
        entry.extend_from_slice(&entry_digest(key, artifact).to_le_bytes());
        entry.extend_from_slice(artifact);
        let staged = self.scratch_path(key, "tmp");
        std::fs::write(&staged, &entry)
            .and_then(|()| std::fs::rename(&staged, self.entry_path(key)))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&staged);
            })
    }

    /// `llc` `body` to bytes without touching the cache entries. The scratch artifact is unique per call
    /// and removed before returning, so a failed compile leaves the directory as it found it. The
    /// artifact is read under `max_artifact_bytes`: a longer one is [`ArtifactTooLarge`] and is never held
    /// in memory.
    pub fn compile_body(
        &self,
        body: &Body,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<Vec<u8>, CompileError> {
        std::fs::create_dir_all(&self.dir)?;
        let scratch = self.scratch_path(&self.key(body), "out");
        let compiled = compile(body, self.target, &scratch);
        let bytes = compiled.and_then(|_| {
            read_bounded(&scratch, max_artifact_bytes.get())?.map_err(CompileError::from)
        });
        let _ = std::fs::remove_file(&scratch);
        bytes
    }

    /// The artifact for `key`: a verified hit, else `build`'s output, stored before it is returned. A hit
    /// or a build longer than `max_artifact_bytes` is [`ArtifactTooLarge`] and nothing is published.
    pub fn load_or_build<E: From<std::io::Error> + From<ArtifactTooLarge>>(
        &self,
        key: &KernelKey,
        max_artifact_bytes: NonZeroU64,
        build: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<Artifact, E> {
        if let Some(bytes) = self.load(key, max_artifact_bytes)? {
            return Ok(Artifact {
                bytes,
                built: false,
            });
        }
        let bytes = build()?;
        ArtifactTooLarge::check(bytes.len() as u64, max_artifact_bytes)?;
        self.store(key, &bytes)?;
        Ok(Artifact { bytes, built: true })
    }

    /// The artifact for `body`: a verified hit, else an `llc` compile, stored before it is returned, all
    /// under `max_artifact_bytes`.
    pub fn load_or_compile(
        &self,
        body: &Body,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<Artifact, CompileError> {
        self.load_or_build(&self.key(body), max_artifact_bytes, || {
            self.compile_body(body, max_artifact_bytes)
        })
    }

    /// A path beside the entries that no other call in any process shares.
    fn scratch_path(&self, key: &KernelKey, suffix: &str) -> PathBuf {
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        self.dir.join(format!(
            "{}.{}.{nonce}.{suffix}",
            key.as_str(),
            std::process::id()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AmdArch;
    use std::sync::Mutex;

    // Every test mutates process-global env vars. nextest isolates tests per process; the lock guards
    // against `cargo test`, which runs them as threads in one process.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn unset_all_cache_env() {
        for var in [
            "POOT_KERNEL_CACHE_DIR",
            "XDG_CACHE_HOME",
            "POOT_KERNEL_CACHE",
        ] {
            // SAFETY: single-threaded per nextest process; ENV_LOCK covers `cargo test` threads.
            unsafe { std::env::remove_var(var) };
        }
    }

    fn unique_tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "poot-codegen-cache-test-{name}-{}",
            std::process::id()
        ))
    }

    #[test]
    fn cache_root_prefers_poot_kernel_cache_dir() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let want = unique_tmp("prefers-explicit-dir");
        let _ = std::fs::remove_dir_all(&want);
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &want) };

        let dir = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()));
        assert!(
            dir.starts_with(&want),
            "POOT_KERNEL_CACHE_DIR must take priority over XDG/HOME: got {dir:?}, want under {want:?}"
        );
        assert!(
            dir.exists(),
            "kernel_cache_root must create the directory it returns"
        );

        unsafe { std::env::remove_var("POOT_KERNEL_CACHE_DIR") };
        let _ = std::fs::remove_dir_all(&want);
    }

    #[test]
    fn cache_root_honors_xdg_cache_home() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let want = unique_tmp("xdg-cache-home");
        let _ = std::fs::remove_dir_all(&want);
        unsafe { std::env::set_var("XDG_CACHE_HOME", &want) };

        let dir = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()));
        assert!(
            dir.starts_with(want.join("poot").join("kernels")),
            "must root under $XDG_CACHE_HOME/poot/kernels when POOT_KERNEL_CACHE_DIR is unset: {dir:?}"
        );

        unsafe { std::env::remove_var("XDG_CACHE_HOME") };
        let _ = std::fs::remove_dir_all(&want);
    }

    #[test]
    fn cache_root_falls_back_to_temp_dir_when_root_is_unwritable() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        // A path under a regular file always fails `create_dir_all` with ENOTDIR, without needing root
        // or chmod.
        let blocker = unique_tmp("unwritable-blocker-file");
        std::fs::write(&blocker, b"not a directory").unwrap();
        let impossible = blocker.join("sub").join("dir");
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &impossible) };

        let dir = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()));
        assert!(
            dir.starts_with(std::env::temp_dir()),
            "an unwritable root must fall back to temp_dir(), got {dir:?}"
        );
        assert!(
            !dir.starts_with(&impossible),
            "must not silently succeed under the unwritable path"
        );

        unsafe { std::env::remove_var("POOT_KERNEL_CACHE_DIR") };
        let _ = std::fs::remove_file(&blocker);
    }

    #[test]
    fn kill_switch_returns_a_distinct_per_process_dir() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        unsafe { std::env::set_var("POOT_KERNEL_CACHE", "0") };
        let dir_a = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()));

        unsafe { std::env::set_var("POOT_KERNEL_CACHE", "off") };
        let dir_b = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()));

        // The kill-switch dir is per process, so both calls return the same path: a temp_dir() path
        // distinct from any shared-cache root, stable for one process's own compiles.
        assert!(dir_a.starts_with(std::env::temp_dir()));
        assert_eq!(
            dir_a, dir_b,
            "kill switch must key the fallback dir on pid, not on call count"
        );
        assert!(
            dir_a
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(&std::process::id().to_string()),
            "kill-switch dir must be pid-scoped: {dir_a:?}"
        );

        unsafe { std::env::remove_var("POOT_KERNEL_CACHE") };
    }

    #[test]
    fn epoch_changes_when_amd_clang_path_changes() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        unsafe { std::env::remove_var("POOT_AMD_CLANG") };
        let target = Target::AmdGcn(AmdArch::gfx1151());

        unsafe { std::env::set_var("POOT_AMD_CLANG", "/nix/store/aaaa-clang/bin/clang") };
        let epoch_a = epoch(target);
        unsafe { std::env::set_var("POOT_AMD_CLANG", "/nix/store/bbbb-clang/bin/clang") };
        let epoch_b = epoch(target);

        assert_ne!(
            epoch_a, epoch_b,
            "a toolchain path change (e.g. a flake bump) must change the epoch"
        );
        unsafe { std::env::remove_var("POOT_AMD_CLANG") };
    }

    #[test]
    fn epoch_changes_when_device_libs_change() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        unsafe { std::env::remove_var("POOT_ROCM_DEVICE_LIBS") };
        let target = Target::AmdGcn(AmdArch::gfx1151());

        let epoch_unset = epoch(target);
        unsafe { std::env::set_var("POOT_ROCM_DEVICE_LIBS", "/nix/store/xxxx/amdgcn/bitcode") };
        let epoch_set = epoch(target);

        assert_ne!(
            epoch_unset, epoch_set,
            "a device-libs path change must change the epoch"
        );
        unsafe { std::env::remove_var("POOT_ROCM_DEVICE_LIBS") };
    }

    #[test]
    fn epoch_changes_when_llc_args_change() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        // `llc_args()` for AmdGcn embeds `-mcpu=<mcpu>`, so two archs differ in the folded-in flags.
        let epoch_1151 = epoch(Target::AmdGcn(AmdArch::gfx1151()));
        let epoch_1100 = epoch(Target::AmdGcn(AmdArch::new("gfx1100", 32)));

        assert_ne!(
            epoch_1151, epoch_1100,
            "a codegen-flag change (here, -mcpu) must change the epoch"
        );
    }

    #[test]
    fn codegen_src_hash_is_present_and_nonempty() {
        assert!(
            !CODEGEN_SRC_HASH.is_empty(),
            "build.rs must always export a non-empty POOT_CODEGEN_SRC_HASH"
        );
        assert_eq!(
            CODEGEN_SRC_HASH.len(),
            16,
            "expected a 16-hex-digit fnv1a fingerprint, got {CODEGEN_SRC_HASH:?}"
        );
        assert!(
            u64::from_str_radix(CODEGEN_SRC_HASH, 16).is_ok(),
            "POOT_CODEGEN_SRC_HASH must be valid hex: {CODEGEN_SRC_HASH:?}"
        );
    }

    /// An artifact limit no fixture here comes near.
    const ROOMY: NonZeroU64 = NonZeroU64::new(1 << 20).unwrap();

    fn load(cache: &KernelCache, key: &KernelKey) -> Option<Vec<u8>> {
        cache.load(key, ROOMY).unwrap()
    }

    fn sample_body(name: &str) -> Body {
        poot_kernelgen::unary(name, poot_kernel_ir::UnOp::Neg)
    }

    const TARGETS: [Target; 2] = [Target::SpirvVulkan, Target::Nvptx];

    /// SC-001: an entry written under one codegen epoch is a miss under another, for the wgpu (SPIR-V) and
    /// the PTX key paths, even from the same directory.
    #[test]
    fn entry_from_an_older_epoch_is_a_miss() {
        for target in TARGETS {
            let dir = unique_tmp(&format!("stale-epoch-{target:?}"));
            let _ = std::fs::remove_dir_all(&dir);
            let body = sample_body("stale_epoch");
            let old = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), target);
            let new = KernelCache::with_epoch(dir.clone(), "0000000000000002".into(), target);

            let old_key = old.key(&body);
            old.store(&old_key, b"old artifact").unwrap();
            assert_eq!(load(&old, &old_key).as_deref(), Some(&b"old artifact"[..]));

            assert_eq!(
                load(&new, &new.key(&body)),
                None,
                "{target:?}: a bumped epoch must not load the older epoch's entry"
            );
            assert_ne!(
                old_key,
                new.key(&body),
                "{target:?}: the epoch must be in the key"
            );
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    /// SC-001, end to end: changing an epoch input (the compiler path) moves `open` to a different root and
    /// a different key, so the artifact stored before the change is not consulted.
    #[test]
    fn changing_epoch_inputs_misses_the_shared_cache() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let base = unique_tmp("epoch-inputs");
        let _ = std::fs::remove_dir_all(&base);
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &base) };
        let body = sample_body("epoch_inputs");

        for target in TARGETS {
            unsafe { std::env::set_var("POOT_LLC", "/nix/store/aaaa-llvm/bin/llc") };
            let before = KernelCache::open(target);
            before.store(&before.key(&body), b"artifact").unwrap();
            assert!(load(&before, &before.key(&body)).is_some());

            unsafe { std::env::set_var("POOT_LLC", "/nix/store/bbbb-llvm/bin/llc") };
            let after = KernelCache::open(target);
            assert_ne!(before.dir(), after.dir(), "{target:?}");
            assert_eq!(
                load(&after, &after.key(&body)),
                None,
                "{target:?}: a toolchain change must miss the older epoch's entry"
            );
        }
        unsafe {
            std::env::remove_var("POOT_LLC");
            std::env::remove_var("POOT_KERNEL_CACHE_DIR");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// SC-002: a cached file that no longer matches its digest is a miss and is rebuilt, not loaded.
    #[test]
    fn corrupted_entry_is_rebuilt_not_loaded() {
        let dir = unique_tmp("corrupt-entry");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), Target::Nvptx);
        let key = cache.key(&sample_body("corrupt_entry"));
        cache.store(&key, b"good artifact").unwrap();
        let path = dir.join(key.as_str());

        let mut on_disk = std::fs::read(&path).unwrap();
        *on_disk.last_mut().unwrap() ^= 0xff;
        std::fs::write(&path, &on_disk).unwrap();
        assert_eq!(
            load(&cache, &key),
            None,
            "a flipped payload byte must be a miss"
        );

        let mut builds = 0;
        let rebuilt = cache
            .load_or_build(&key, ROOMY, || {
                builds += 1;
                Ok::<_, CompileError>(b"rebuilt artifact".to_vec())
            })
            .unwrap();
        assert!(rebuilt.built, "a corrupted entry must rebuild");
        assert_eq!(builds, 1);
        assert_eq!(rebuilt.bytes, b"rebuilt artifact");
        assert_eq!(
            load(&cache, &key).as_deref(),
            Some(&b"rebuilt artifact"[..])
        );

        let hit = cache
            .load_or_build(&key, ROOMY, || -> Result<Vec<u8>, CompileError> {
                panic!("a verified hit must not rebuild")
            })
            .unwrap();
        assert!(!hit.built);

        std::fs::write(&path, &on_disk[..ENTRY_HEADER_LEN - 1]).unwrap();
        assert_eq!(
            load(&cache, &key),
            None,
            "a truncated header must be a miss"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An entry copied to another key's name (e.g. out of an older epoch's directory) fails its digest.
    #[test]
    fn entry_moved_to_another_key_is_a_miss() {
        let dir = unique_tmp("moved-entry");
        let _ = std::fs::remove_dir_all(&dir);
        let old = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), Target::Nvptx);
        let new = KernelCache::with_epoch(dir.clone(), "0000000000000002".into(), Target::Nvptx);
        let body = sample_body("moved_entry");
        old.store(&old.key(&body), b"artifact").unwrap();
        std::fs::copy(
            dir.join(old.key(&body).as_str()),
            dir.join(new.key(&body).as_str()),
        )
        .unwrap();
        assert_eq!(load(&new, &new.key(&body)), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// SC-003: the wgpu and PTX key paths fingerprint the same Body identically, and different Bodies
    /// differently; only the epoch and target half of the key may differ between the two.
    #[test]
    fn spirv_and_ptx_keys_share_the_body_fingerprint() {
        let dir = unique_tmp("shared-fingerprint");
        let body = sample_body("shared_fingerprint");
        let spirv = KernelCache::at(&dir, Target::SpirvVulkan).key(&body);
        let ptx = KernelCache::at(&dir, Target::Nvptx).key(&body);

        assert_eq!(spirv.fingerprint(), ptx.fingerprint());
        assert_eq!(spirv.fingerprint(), body_fingerprint(&body));
        assert_ne!(spirv, ptx, "the target must be in the key");
        let other = KernelCache::at(&dir, Target::Nvptx).key(&sample_body("some_other_kernel"));
        assert_ne!(ptx.fingerprint(), other.fingerprint());
    }

    fn limit(bytes: u64) -> NonZeroU64 {
        NonZeroU64::new(bytes).unwrap()
    }

    /// Card 666 SC-002, the artifact charge: an artifact of exactly `max_artifact_bytes` is admitted and
    /// published, one byte more is refused with the typed error and nothing is published, so a later
    /// lookup misses. Mutation: remove the `ArtifactTooLarge::check` in `load_or_build`; the 65-byte
    /// build is stored and returned, and both the refusal and the not-published assertions fail.
    #[test]
    fn an_artifact_over_the_limit_is_refused_before_it_is_published() {
        let dir = unique_tmp("artifact-limit");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), Target::Nvptx);
        let fits = cache.key(&sample_body("artifact_fits"));
        let over = cache.key(&sample_body("artifact_over"));

        let artifact = cache
            .load_or_build(&fits, limit(64), || Ok::<_, CompileError>(vec![7u8; 64]))
            .expect("64 bytes fit a 64-byte limit");
        assert!(artifact.built);
        assert_eq!(load(&cache, &fits).as_deref(), Some(&[7u8; 64][..]));

        let refused = cache
            .load_or_build(&over, limit(64), || Ok::<_, CompileError>(vec![7u8; 65]))
            .unwrap_err();
        assert!(
            matches!(
                refused,
                CompileError::ArtifactTooLarge(ArtifactTooLarge {
                    bytes: 65,
                    limit: 64
                })
            ),
            "65 bytes against a 64-byte limit: {refused}"
        );
        assert_eq!(
            load(&cache, &over),
            None,
            "a refused artifact is never published"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// An oversize entry already on disk is refused from its length, not read whole, and the build is not
    /// retried: it would produce the same artifact. The same entry loads under a limit that admits it.
    /// Mutation: read the entry with no limit (`entry_limit = u64::MAX` in `load`); the 65-byte entry
    /// loads under a 64-byte limit and the first assertion fails.
    #[test]
    fn an_oversize_cached_entry_is_refused_and_not_rebuilt() {
        let dir = unique_tmp("artifact-oversize-entry");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), Target::Nvptx);
        let key = cache.key(&sample_body("artifact_oversize_entry"));
        cache.store(&key, &[9u8; 65]).unwrap();

        assert_eq!(
            cache.load(&key, limit(64)),
            Err(ArtifactTooLarge {
                bytes: 65,
                limit: 64
            })
        );
        let mut builds = 0;
        let refused = cache.load_or_build(&key, limit(64), || {
            builds += 1;
            Ok::<_, CompileError>(vec![9u8; 4])
        });
        assert!(refused.is_err());
        assert_eq!(
            builds, 0,
            "an oversize hit does not fall through to a build"
        );
        assert_eq!(
            cache
                .load(&key, limit(65))
                .unwrap()
                .map(|bytes| bytes.len()),
            Some(65),
            "the same entry loads under a limit that admits it"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The compiler's output is read under the limit: an artifact the toolchain produced longer than
    /// `max_artifact_bytes` is refused and its scratch file is removed. Mutation: read the scratch with
    /// `std::fs::read`; the typed refusal disappears and the compile returns the oversize artifact.
    #[test]
    fn a_compiled_artifact_over_the_limit_is_refused_and_its_scratch_removed() {
        let dir = unique_tmp("artifact-compile-limit");
        let _ = std::fs::remove_dir_all(&dir);
        let cache = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), Target::Nvptx);
        let body = sample_body("artifact_compile_limit");
        let whole = cache
            .compile_body(&body, ROOMY)
            .expect("the sample body compiles");
        let refused = cache
            .compile_body(&body, limit(whole.len() as u64 - 1))
            .unwrap_err();
        assert!(
            matches!(refused, CompileError::ArtifactTooLarge(over) if over.limit == whole.len() as u64 - 1),
            "{refused}"
        );
        let leftovers = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(leftovers, 0, "no scratch file survives a refusal");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scratch_path_parent_equals_artifact_path_parent() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let want = unique_tmp("scratch-parent");
        let _ = std::fs::remove_dir_all(&want);
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &want) };

        let target = Target::AmdGcn(AmdArch::gfx1151());
        let dir = kernel_cache_root("rocm", target);
        let artifact_path = crate::artifact_path(&dir, "some_kernel_abc123", target);
        let (scratch_ll, scratch_tmp) = crate::compile_scratch_paths(&artifact_path, 12345, 0);

        assert_eq!(
            scratch_ll.parent(),
            artifact_path.parent(),
            "scratch .ll must live in the same dir as the published artifact, or the final \
             rename onto it is a cross-filesystem copy and not atomic"
        );
        assert_eq!(
            scratch_tmp.parent(),
            artifact_path.parent(),
            "scratch .tmp must live in the same dir as the published artifact, same reason"
        );

        unsafe { std::env::remove_var("POOT_KERNEL_CACHE_DIR") };
        let _ = std::fs::remove_dir_all(&want);
    }
}

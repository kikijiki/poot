//! Persistent, epoch-scoped on-disk root for compiled-kernel artifact caches (card 211).
//!
//! The ROCm HSACO disk cache is content-addressed by `fnv1a(Body)` + arch (cards 211/226,
//! `RocmGraphExecutor::artifact_name`); `compile()` shells out to the toolchain only when that path does
//! not exist. This module gives the cache a stable root: `nix develop -c` assigns a fresh `TMPDIR` per
//! invocation, so a cache under `std::env::temp_dir()` was never reused across processes.
//!
//! A stable root would keep stale artifacts across toolchain bumps and emitter edits, and a stale HSACO
//! silently runs the wrong kernel (a correctness bug), so the toolchain/codegen epoch below is mandatory.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, SystemTime};

use poot_kernel_ir::Body;

use crate::{CompileError, CompiledArtifact, GpuTarget, Target, Toolchain, capability, compile};

/// The 64-bit content fingerprint of a kernel `Body` (a hash of its `Debug` form). Only the AIE identity
/// text still names a body this way; every byte-artifact kernel is named by its [`KernelId`].
pub(crate) fn body_fingerprint(body: &Body) -> u64 {
    poot_runtime_common::fnv1a_bytes(format!("{body:?}").as_bytes())
}

/// The 128-bit content identity of one compiled kernel: a blake3 hash over the target (its whole `Debug`
/// form, so an AMD arch's wavefront width counts), the target's [`Capability`](crate::Capability) (what
/// shapes a body for it) and the body's `Debug` text. Two bodies with equal text for different targets are
/// different kernels, so they have different ids. The id names the cache entry and rides inside the
/// [`CompiledArtifact`] token, which is how a handle refuses an artifact compiled from another body.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelId([u8; 16]);

impl KernelId {
    /// The id of `body` compiled for `target`.
    pub(crate) fn of(body: &Body, target: Target) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"poot-kernel-id-v2\0");
        hasher.update(format!("{target:?}\0{:?}\0", capability(target)).as_bytes());
        hasher.update(format!("{body:?}").as_bytes());
        Self::truncated(hasher.finalize())
    }

    /// The id of a linked AIE artifact: its core-object and link identity digests, side by side.
    fn of_aie(core_digest: u64, link_digest: u64) -> Self {
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&core_digest.to_be_bytes());
        bytes[8..].copy_from_slice(&link_digest.to_be_bytes());
        Self(bytes)
    }

    fn truncated(hash: blake3::Hash) -> Self {
        Self(
            hash.as_bytes()[..16]
                .try_into()
                .expect("a blake3 hash is 32 bytes"),
        )
    }

    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        Some(Self(bytes.try_into().ok()?))
    }
}

impl std::fmt::Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

impl std::fmt::Debug for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KernelId({self})")
    }
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
/// different epoch are never consulted (see `kernel_cache_root`).
pub fn epoch(target: Target) -> String {
    format!(
        "{:016x}",
        poot_runtime_common::fnv1a_bytes(epoch_canonical(target).as_bytes())
    )
}

/// How many epoch directories besides the current one a kind root keeps, newest first by [`epoch_recency`] (last use).
/// Every emitter edit and toolchain bump mints a directory, and parallel worktrees at different codegen
/// hashes each use their own, so the bound has to cover the lanes that run at once.
const RETAINED_EPOCHS: usize = 8;

/// An epoch directory is never removed while its `EPOCH` sidecar is younger than this, whatever its rank.
/// The sidecar's mtime is the epoch's last use: a cache hit or an adoption at open refreshes it (at most
/// once per [`SIDECAR_REFRESH_INTERVAL`]), so a long-running process keeps its epoch alive against another
/// process's prune, and only an epoch idle for this long is removable.
const EPOCH_MIN_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// The sidecar is only rewritten when older than this, bounding the cost of tracking use to one `stat` per
/// hit and one write per hour per epoch.
const SIDECAR_REFRESH_INTERVAL: Duration = Duration::from_secs(60 * 60);

const EPOCH_SIDECAR: &str = "EPOCH";

/// A compiler scratch file (see [`is_scratch_name`]) is only removed once it is this old. A compile of one
/// kernel takes seconds to minutes, so a file older than this belongs to a process that died.
const SCRATCH_MIN_AGE: Duration = Duration::from_secs(60 * 60);

/// What one prune of a kind root did. Failures are counted and logged, never returned as errors: cleanup
/// is housekeeping and must not fail the compile that triggered it.
#[derive(Debug, Default, PartialEq, Eq)]
struct PruneReport {
    removed_epochs: usize,
    removed_scratch: usize,
    failures: usize,
}

/// Whether `name` is a compiler scratch file: `<stem>.<pid>.<nonce>.<ll|tmp|out>`, as produced by
/// `compile_scratch_paths` (the `.ll` and the temp object) and `KernelCache::scratch_path` (the staged
/// entry and the compile output). Published entries (`k<epoch>-<fingerprint>.<ext>`) never match.
fn is_scratch_name(name: &str) -> bool {
    let mut parts = name.rsplitn(4, '.');
    let (Some(ext), Some(nonce), Some(pid), Some(stem)) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    matches!(ext, "ll" | "tmp" | "out") && digits(nonce) && digits(pid) && !stem.is_empty()
}

/// When epoch directory `dir` was last in use: the mtime of its `EPOCH` sidecar, which creation writes and
/// every cache hit refreshes ([`refresh_epoch_sidecar`]). `None` if it cannot be read, which the caller
/// treats as "keep".
fn epoch_recency(dir: &Path) -> Option<SystemTime> {
    std::fs::metadata(dir.join(EPOCH_SIDECAR))
        .ok()?
        .modified()
        .ok()
}

/// Mark the epoch owning `sidecar` as in use: set the sidecar's mtime to now when it is older than
/// [`SIDECAR_REFRESH_INTERVAL`]. One `stat` per call and one write per interval per epoch. Best effort: a
/// missing sidecar (a cache at an explicit directory) is nothing to refresh, and a failed write is logged
/// and never fails the hit.
fn refresh_epoch_sidecar(sidecar: &Path) {
    let Ok(modified) = std::fs::metadata(sidecar).and_then(|m| m.modified()) else {
        return;
    };
    let now = SystemTime::now();
    if age(now, modified) < SIDECAR_REFRESH_INTERVAL {
        return;
    }
    if let Err(error) = std::fs::File::open(sidecar).and_then(|f| f.set_modified(now)) {
        tracing::warn!(?sidecar, %error, "could not refresh a kernel cache epoch's last-use time");
    }
}

fn age(now: SystemTime, modified: SystemTime) -> Duration {
    now.duration_since(modified).unwrap_or_default()
}

/// Bound the disk a kind root (`<base>/<kind>`) holds: remove epoch directories beyond the newest
/// [`RETAINED_EPOCHS`] that are also older than [`EPOCH_MIN_AGE`], then abandoned scratch inside the
/// directories that remain. `current` is never removed.
///
/// Only a real directory (not a symlink) directly under `root` that carries a regular `EPOCH` sidecar
/// is an epoch; anything else under the root is left alone, and every removal path is `root.join(name)`
/// for a name read from `root` itself. A failed removal is counted and logged and the walk continues.
fn prune_kind_root(root: &Path, current: &str, now: SystemTime) -> PruneReport {
    let mut report = PruneReport::default();
    let Ok(entries) = std::fs::read_dir(root) else {
        return report;
    };
    let mut epochs: Vec<(SystemTime, PathBuf)> = entries
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .filter(|e| e.file_name() != current)
        .map(|e| e.path())
        .filter(|dir| {
            std::fs::symlink_metadata(dir.join(EPOCH_SIDECAR))
                .is_ok_and(|m| m.file_type().is_file())
        })
        .filter_map(|dir| Some((epoch_recency(&dir)?, dir)))
        .collect();
    // Newest first; the name breaks ties so the order is deterministic.
    epochs.sort_by(|a, b| b.cmp(a));

    let mut surviving: Vec<PathBuf> = vec![root.join(current)];
    let mut doomed = Vec::new();
    for (rank, (recency, dir)) in epochs.into_iter().enumerate() {
        if rank >= RETAINED_EPOCHS && age(now, recency) >= EPOCH_MIN_AGE {
            doomed.push(dir);
        } else {
            surviving.push(dir);
        }
    }
    // Oldest first, so a failure part way leaves the newest directories in place.
    for dir in doomed.into_iter().rev() {
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => report.removed_epochs += 1,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                tracing::warn!(?dir, %error, "could not remove an old kernel cache epoch");
                report.failures += 1;
            }
        }
    }
    for epoch_dir in surviving {
        prune_scratch(&epoch_dir, now, &mut report);
    }
    report
}

/// Remove scratch files older than [`SCRATCH_MIN_AGE`] from the `<mcpu>` directories of one epoch.
fn prune_scratch(epoch_dir: &Path, now: SystemTime, report: &mut PruneReport) {
    let Ok(arch_dirs) = std::fs::read_dir(epoch_dir) else {
        return;
    };
    for arch_dir in arch_dirs
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
    {
        let Ok(files) = std::fs::read_dir(arch_dir.path()) else {
            continue;
        };
        for file in files.flatten() {
            let abandoned = file.file_type().is_ok_and(|t| t.is_file())
                && file.file_name().to_str().is_some_and(is_scratch_name)
                && file
                    .metadata()
                    .and_then(|m| m.modified())
                    .is_ok_and(|modified| age(now, modified) >= SCRATCH_MIN_AGE);
            if !abandoned {
                continue;
            }
            match std::fs::remove_file(file.path()) {
                Ok(()) => report.removed_scratch += 1,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    tracing::warn!(path = ?file.path(), %error, "could not remove abandoned kernel scratch");
                    report.failures += 1;
                }
            }
        }
    }
}

/// Prune `root` once per process: `kernel_cache_root` runs for every executor, and one walk per process is
/// enough to bound growth across processes.
fn prune_kind_root_once(root: &Path, current: &str) {
    static PRUNED: LazyLock<Mutex<HashSet<PathBuf>>> = LazyLock::new(Mutex::default);
    let first = PRUNED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(root.to_path_buf());
    if first {
        let report = prune_kind_root(root, current, SystemTime::now());
        if report != PruneReport::default() {
            tracing::info!(?root, ?report, "pruned the kernel cache");
        }
    }
}

/// `POOT_KERNEL_CACHE=0` (also `off`/`false`, case-insensitive) disables the shared cache, without a
/// rebuild.
fn kill_switch_active() -> bool {
    match std::env::var("POOT_KERNEL_CACHE") {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false"),
        Err(_) => false,
    }
}

/// Base directory for the shared cache (before the `<kind>/<epoch>/<mcpu>` split), independent of
/// `target`. Order: `$POOT_KERNEL_CACHE_DIR` verbatim, else `$XDG_CACHE_HOME/poot/kernels`, else
/// `$HOME/.cache/poot/kernels`. `None` when none of them is set: a shared temp directory is not a place
/// to keep executable kernels, so there is no further fallback.
fn resolve_base_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("POOT_KERNEL_CACHE_DIR")
        && !dir.is_empty()
    {
        return Some(PathBuf::from(dir));
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME")
        && !xdg.is_empty()
    {
        return Some(PathBuf::from(xdg).join("poot").join("kernels"));
    }
    if let Ok(home) = std::env::var("HOME")
        && !home.is_empty()
    {
        return Some(
            PathBuf::from(home)
                .join(".cache")
                .join("poot")
                .join("kernels"),
        );
    }
    None
}

/// The user id this process runs as: the owner of its own `/proc/self`. `None` where that cannot be read,
/// and a cache directory is then not trusted.
fn current_uid() -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self").ok().map(|m| m.uid())
}

/// Make `path` a directory only this user can use: created `0700` when absent, and when present a real
/// directory (never a symlink) owned by `uid`, tightened to `0700` if it was wider. Cached kernels are
/// loaded into GPU drivers, so a directory another user can write into is a code-injection point.
fn ensure_private_dir(path: &Path, uid: u32) -> std::io::Result<()> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    let meta = std::fs::symlink_metadata(path)?;
    if !meta.file_type().is_dir() {
        return Err(std::io::Error::other(format!(
            "{path:?} is not a real directory"
        )));
    }
    if meta.uid() != uid {
        return Err(std::io::Error::other(format!(
            "{path:?} is owned by another user"
        )));
    }
    if meta.mode() & 0o077 != 0 {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Resolve the on-disk root for a persistent, epoch-scoped kernel artifact cache and ensure it exists, or
/// `None` when there is no safe place for one. `kind` names the caller's kernel family (e.g. `"rocm"`) so
/// backends sharing a base directory do not collide on the same epoch directory.
///
/// Returns `<base>/<kind>/<epoch>/<mcpu-or-target>`. The `<kind>`, `<epoch>` and `<mcpu>` directories are
/// private to the user (`0700`, owner-checked, never a symlink; see [`ensure_private_dir`]). A plaintext
/// `EPOCH` sidecar is written on first creation, so a mismatch can be diagnosed by hand. The first call per
/// process for a `<kind>` also prunes that root's old epochs and abandoned scratch ([`RETAINED_EPOCHS`],
/// [`EPOCH_MIN_AGE`], [`SCRATCH_MIN_AGE`]), and adopting an existing epoch refreshes its last-use time; a
/// failed removal never fails the call.
///
/// `None` is a disabled cache, not a degraded one: `POOT_KERNEL_CACHE=0`/`off`/`false`, no base directory
/// in the environment, or a base that cannot be created or is not the user's own. The cause is logged with
/// `tracing::warn!`. Nothing is written anywhere else.
fn kernel_cache_root(kind: &str, target: Target) -> Option<PathBuf> {
    if kill_switch_active() {
        return None;
    }
    let Some(base_dir) = resolve_base_dir() else {
        tracing::warn!(
            "no kernel cache directory (POOT_KERNEL_CACHE_DIR, XDG_CACHE_HOME, HOME all unset); the cache is disabled"
        );
        return None;
    };
    let Some(uid) = current_uid() else {
        tracing::warn!("cannot tell which user owns the kernel cache; the cache is disabled");
        return None;
    };

    let epoch_hex = epoch(target);
    let arch_component = match target {
        Target::AmdGcn(arch) => arch.mcpu().to_string(),
        other => format!("{other:?}"),
    };
    let base = base_dir.join(kind);
    let epoch_dir = base.join(&epoch_hex);
    let dir = epoch_dir.join(&arch_component);

    let created = std::fs::create_dir_all(&base_dir).and_then(|()| {
        [&base, &epoch_dir, &dir]
            .into_iter()
            .try_for_each(|path| ensure_private_dir(path, uid))
    });
    if let Err(error) = created {
        tracing::warn!(?dir, %error, "kernel cache directory is unusable; the cache is disabled");
        return None;
    }
    let epoch_file = epoch_dir.join(EPOCH_SIDECAR);
    if epoch_file.exists() {
        refresh_epoch_sidecar(&epoch_file);
    } else {
        // Best effort: a diagnostic sidecar; a failed write must not stop compilation.
        let _ = std::fs::write(&epoch_file, epoch_canonical(target));
    }
    prune_kind_root_once(&base, &epoch_hex);
    Some(dir)
}

/// The per-target directory family under the shared base, so backends sharing a base directory do not
/// collide. AMDGCN matches the `kind` the ROCm executor passes to `kernel_cache_root` directly.
fn cache_kind(target: Target) -> &'static str {
    match target {
        Target::Nvptx => "ptx",
        Target::SpirvVulkan => "spirv",
        Target::SpirvOpenCl => "spirv-opencl",
        Target::AmdGcn(_) => "rocm",
        Target::AieCore => "aie",
    }
}

/// The `poot-codegen` source hash folded into every AIE identity (see [`CODEGEN_SRC_HASH`]).
pub(crate) fn codegen_src_hash() -> &'static str {
    CODEGEN_SRC_HASH
}

/// The entry key of a linked AIE artifact: the digests of its core-object identity and its link identity
/// (both over canonical, length-tagged text), so a changed Body, shape, layout, recipe, platform or
/// toolchain names a different entry. The epoch is not part of the name: every input to the compiler's
/// output is in the identities.
pub(crate) fn aie_entry_key(core_digest: u64, link_digest: u64) -> KernelKey {
    KernelKey {
        name: format!("a{core_digest:016x}{link_digest:016x}.aie"),
        id: KernelId::of_aie(core_digest, link_digest),
    }
}

/// Every cache entry starts with this magic, then the [`KernelId`] the entry was stored for, then the
/// [`entry_digest`], then the artifact.
const ENTRY_MAGIC: &[u8; 8] = b"POOTKC02";
const ENTRY_ID_LEN: usize = size_of::<KernelId>();
const ENTRY_DIGEST_LEN: usize = blake3::OUT_LEN;
const ENTRY_HEADER_LEN: usize = ENTRY_MAGIC.len() + ENTRY_ID_LEN + ENTRY_DIGEST_LEN;

/// The name of one cache entry: the codegen epoch, the target and the kernel's content id. Two kernels
/// share a key only if the same toolchain and emitter would produce the same artifact from the same Body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelKey {
    name: String,
    id: KernelId,
}

impl KernelKey {
    /// The entry's file name (`k<epoch>-<kernel id>.<ext>`).
    pub fn as_str(&self) -> &str {
        &self.name
    }

    /// The content id half of the key: the id of the body the entry is for, on the cache's target.
    pub fn id(&self) -> KernelId {
        self.id
    }
}

/// The one key function every backend's disk cache uses. Pure: it reads no environment and touches no disk,
/// so the epoch and target it is given are exactly what scopes the entry.
fn kernel_entry_key(epoch: &str, target: Target, id: KernelId) -> KernelKey {
    KernelKey {
        name: format!("k{epoch}-{id}.{}", target.output_ext()),
        id,
    }
}

/// blake3 over the key name, the key's id and the artifact bytes. Binding the name means an entry copied or
/// renamed to another key (say, out of an older epoch's directory) fails its check instead of being loaded.
fn entry_digest(key: &KernelKey, artifact: &[u8]) -> [u8; ENTRY_DIGEST_LEN] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(key.name.len() as u64).to_le_bytes());
    hasher.update(key.name.as_bytes());
    hasher.update(&key.id.0);
    hasher.update(artifact);
    *hasher.finalize().as_bytes()
}

/// A compiled artifact is longer than the caller's `max_artifact_bytes`. Raised before the bytes are
/// loaded into host memory, handed to a runtime or published to the cache; an external compiler's own
/// memory and time are outside what this limit bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("kernel artifact of {bytes} bytes exceeds the {limit}-byte limit")]
pub struct ArtifactTooLarge {
    /// The artifact's length, or the first length past the limit when the source was read bounded.
    pub bytes: u64,
    /// The byte limit the artifact was checked against.
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

/// A cache entry's bytes and whether this call had to build them (a build, as opposed to a verified hit).
/// Only the cache's own callers see it: a byte kernel leaves this crate as a [`CompiledArtifact`], which
/// carries the id of the body it was compiled from.
#[derive(Debug)]
pub(crate) struct Artifact {
    pub(crate) bytes: Vec<u8>,
    /// `true` when this call ran the builder, `false` when a verified cache entry was served.
    pub(crate) built: bool,
}

/// One target's persistent kernel artifact cache, scoped by the codegen [`epoch`] both by directory (an
/// old epoch's directory is never consulted) and by key (an entry carries its epoch in its name and its
/// digest). Every entry is verified on read: a truncated, corrupted, misplaced or foreign-body file is a
/// miss, never a load. Shared by the wgpu and PTX backends; entries are content-addressed, so concurrent
/// processes writing the same key write the same bytes.
///
/// A cache may be disabled (`dir()` is `None`): it then keeps nothing, and every
/// kernel is compiled in a private scratch directory that is removed before the call returns.
#[derive(Debug, Clone)]
pub struct KernelCache {
    dir: Option<PathBuf>,
    epoch: String,
    target: Target,
}

impl KernelCache {
    /// The cache under the shared persistent root for `target` (see `kernel_cache_root`), or a disabled
    /// cache when that root cannot be a safe, private directory.
    pub fn open(target: Target) -> Self {
        Self {
            dir: kernel_cache_root(cache_kind(target), target),
            epoch: epoch(target),
            target,
        }
    }

    /// The cache rooted at an explicit directory instead of the shared one, keyed by `target`'s epoch.
    pub fn at(dir: impl Into<PathBuf>, target: Target) -> Self {
        Self::with_epoch(dir.into(), epoch(target), target)
    }

    fn with_epoch(dir: PathBuf, epoch: String, target: Target) -> Self {
        Self {
            dir: Some(dir),
            epoch,
            target,
        }
    }

    /// The directory the cache's entries and scratch files live in, `None` when the cache is disabled (it
    /// keeps nothing, and still compiles).
    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// The entry key for `body` under this cache's epoch and target.
    pub fn key(&self, body: &Body) -> KernelKey {
        kernel_entry_key(&self.epoch, self.target, KernelId::of(body, self.target))
    }

    fn entry_path(dir: &Path, key: &KernelKey) -> PathBuf {
        dir.join(key.as_str())
    }

    /// The verified artifact for `key`, or `None` on a miss (and always on a disabled cache). A file that
    /// fails the magic, id or digest check is a miss (logged), so the caller rebuilds and
    /// [`store`](Self::store) replaces it. The id recorded in the entry must be the id `key` names: an
    /// entry written for another body is never served as this one. An entry whose artifact is longer than
    /// `max_artifact_bytes` is [`ArtifactTooLarge`], never read whole: a rebuild would produce the same
    /// oversize artifact, so it is not a miss either.
    pub(crate) fn load(
        &self,
        key: &KernelKey,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<Option<Vec<u8>>, ArtifactTooLarge> {
        let Some(dir) = &self.dir else {
            return Ok(None);
        };
        let path = Self::entry_path(dir, key);
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
        if bytes.len() < ENTRY_HEADER_LEN || !bytes.starts_with(ENTRY_MAGIC) {
            tracing::warn!(?path, "kernel cache entry has no valid header; rebuilding");
            return Ok(None);
        }
        let id_end = ENTRY_MAGIC.len() + ENTRY_ID_LEN;
        if KernelId::from_bytes(&bytes[ENTRY_MAGIC.len()..id_end]) != Some(key.id) {
            tracing::warn!(
                ?path,
                "kernel cache entry was stored for another kernel; rebuilding"
            );
            return Ok(None);
        }
        if bytes[id_end..ENTRY_HEADER_LEN] != entry_digest(key, &bytes[ENTRY_HEADER_LEN..]) {
            tracing::warn!(
                ?path,
                "kernel cache entry failed its digest check; rebuilding"
            );
            return Ok(None);
        }
        bytes.drain(..ENTRY_HEADER_LEN);
        if let Some(epoch_dir) = dir.parent() {
            refresh_epoch_sidecar(&epoch_dir.join(EPOCH_SIDECAR));
        }
        Ok(Some(bytes))
    }

    /// Publish `artifact` under `key`: written to a unique temp file beside the entry, then renamed onto
    /// it, so a concurrent reader never sees a partial file. A disabled cache publishes nothing.
    pub(crate) fn store(&self, key: &KernelKey, artifact: &[u8]) -> std::io::Result<()> {
        let Some(dir) = &self.dir else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)?;
        let mut entry = Vec::with_capacity(ENTRY_HEADER_LEN + artifact.len());
        entry.extend_from_slice(ENTRY_MAGIC);
        entry.extend_from_slice(&key.id.0);
        entry.extend_from_slice(&entry_digest(key, artifact));
        entry.extend_from_slice(artifact);
        let staged = self.scratch_path(dir, key, "tmp");
        std::fs::write(&staged, &entry)
            .and_then(|()| std::fs::rename(&staged, Self::entry_path(dir, key)))
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&staged);
            })
    }

    /// `llc` `body` to bytes without touching the cache entries. The scratch artifact is unique per call
    /// and removed before returning, so a failed compile leaves the directory as it found it; a disabled
    /// cache compiles in a [`PrivateScratch`] instead. The artifact is read under `max_artifact_bytes`: a
    /// longer one is [`ArtifactTooLarge`] and is never held in memory.
    fn compile_body(
        &self,
        body: &Body,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<Vec<u8>, CompileError> {
        let private;
        let dir = match &self.dir {
            Some(dir) => {
                std::fs::create_dir_all(dir)?;
                dir.as_path()
            }
            None => {
                private = WorkDir::private()?;
                private.path()
            }
        };
        let scratch = self.scratch_path(dir, &self.key(body), "out");
        let compiled = compile(body, self.target, &scratch);
        let bytes = compiled.and_then(|_| {
            read_bounded(&scratch, max_artifact_bytes.get())?.map_err(CompileError::from)
        });
        let _ = std::fs::remove_file(&scratch);
        bytes
    }

    /// The artifact for `key`: a verified hit, else `build`'s output, stored before it is returned. A hit
    /// or a build longer than `max_artifact_bytes` is [`ArtifactTooLarge`] and nothing is published.
    pub(crate) fn load_or_build<E: From<std::io::Error> + From<ArtifactTooLarge>>(
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

    /// The compiled kernel for `body`: a verified hit, else an `llc` compile, stored before it is
    /// returned, all under `max_artifact_bytes`. The result is the only thing [`kernel_handle`] accepts,
    /// and it carries the id of `body` on this cache's target.
    ///
    /// [`kernel_handle`]: crate::kernel_handle
    pub fn load_or_compile(
        &self,
        body: &Body,
        max_artifact_bytes: NonZeroU64,
    ) -> Result<CompiledArtifact, CompileError> {
        let target = GpuTarget::try_from(self.target)?;
        let key = self.key(body);
        let artifact = self.load_or_build(&key, max_artifact_bytes, || {
            self.compile_body(body, max_artifact_bytes)
        })?;
        Ok(CompiledArtifact::new(
            key.id,
            target,
            artifact.bytes,
            artifact.built,
        ))
    }

    /// A private working directory for the build of `key`: beside the entries when the cache is enabled,
    /// else a [`WorkDir::private`] one. Removed when dropped.
    pub(crate) fn work_dir(&self, key: &KernelKey) -> std::io::Result<WorkDir> {
        match &self.dir {
            Some(dir) => {
                let path = self.scratch_path(dir, key, "work");
                std::fs::create_dir_all(&path)?;
                Ok(WorkDir(path))
            }
            None => WorkDir::private(),
        }
    }

    /// A path in `dir` beside the entries that no other call in any process shares.
    fn scratch_path(&self, dir: &Path, key: &KernelKey, suffix: &str) -> PathBuf {
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        dir.join(format!(
            "{}.{}.{nonce}.{suffix}",
            key.as_str(),
            std::process::id()
        ))
    }
}

/// A working directory for one build, removed when dropped.
pub(crate) struct WorkDir(PathBuf);

impl WorkDir {
    /// A fresh directory only this user can enter, for a build whose cache is disabled.
    fn private() -> std::io::Result<Self> {
        use std::os::unix::fs::DirBuilderExt;
        static NONCE: AtomicU64 = AtomicU64::new(0);
        let nonce = NONCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("poot-scratch-{}-{nonce}", std::process::id()));
        // `create` fails on an existing path, symlink included, so the directory is ours.
        std::fs::DirBuilder::new().mode(0o700).create(&path)?;
        Ok(Self(path))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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

        let dir = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()))
            .expect("an explicit, creatable directory is a cache root");
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

        let dir = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()))
            .expect("a creatable XDG cache home is a cache root");
        assert!(
            dir.starts_with(want.join("poot").join("kernels")),
            "must root under $XDG_CACHE_HOME/poot/kernels when POOT_KERNEL_CACHE_DIR is unset: {dir:?}"
        );

        unsafe { std::env::remove_var("XDG_CACHE_HOME") };
        let _ = std::fs::remove_dir_all(&want);
    }

    /// SC-004: a cache directory that cannot be created disables the cache; it does not move it to a shared
    /// temp directory. The kernel still compiles, in a private scratch directory that is gone afterwards, so
    /// nothing is left under `temp_dir()`. Mutation: fall back to `temp_dir()/poot-<kind>` when the
    /// directory cannot be created; the cache reports enabled and a directory is left under the temp dir.
    #[test]
    fn an_unwritable_cache_dir_disables_the_cache_and_still_compiles() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        // A path under a regular file always fails `create_dir_all` with ENOTDIR, without needing root
        // or chmod.
        let blocker = unique_tmp("unwritable-blocker-file");
        std::fs::write(&blocker, b"not a directory").unwrap();
        let impossible = blocker.join("sub").join("dir");
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &impossible) };
        let temp = unique_tmp("unwritable-temp-dir");
        let _ = std::fs::remove_dir_all(&temp);
        std::fs::create_dir_all(&temp).unwrap();
        let outer_temp = std::env::var_os("TMPDIR");
        unsafe { std::env::set_var("TMPDIR", &temp) };

        let cache = KernelCache::open(Target::Nvptx);
        let compiled = cache.load_or_compile(&sample_body("unwritable_cache"), ROOMY);

        match outer_temp {
            Some(outer) => unsafe { std::env::set_var("TMPDIR", outer) },
            None => unsafe { std::env::remove_var("TMPDIR") },
        }
        unsafe { std::env::remove_var("POOT_KERNEL_CACHE_DIR") };
        let left: Vec<_> = std::fs::read_dir(&temp)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::remove_file(&blocker);

        assert_eq!(cache.dir(), None, "an unwritable root disables the cache");
        let compiled = compiled.expect("a disabled cache still compiles");
        assert!(compiled.built() && !compiled.is_empty());
        assert!(
            left.is_empty(),
            "nothing may be left under the temp dir: {left:?}"
        );
    }

    /// With no cache location in the environment at all the cache is disabled, not placed in `temp_dir()`.
    #[test]
    fn no_cache_location_in_the_environment_disables_the_cache() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let home = std::env::var_os("HOME");
        unsafe { std::env::remove_var("HOME") };
        let root = kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151()));
        if let Some(home) = home {
            unsafe { std::env::set_var("HOME", home) };
        }
        assert_eq!(root, None);
    }

    /// A cache directory is private to the user: created `0700`, a wider existing one is tightened, and one
    /// that is a symlink is refused. Mutation: skip the symlink check; the cache opens through the link.
    #[test]
    fn cache_directories_are_private_and_never_symlinks() {
        use std::os::unix::fs::PermissionsExt;
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let base = unique_tmp("private-dirs");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("rocm")).unwrap();
        std::fs::set_permissions(base.join("rocm"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &base) };
        let target = Target::AmdGcn(AmdArch::gfx1151());

        let dir = kernel_cache_root("rocm", target).expect("an existing own directory is usable");
        for private in [
            dir.as_path(),
            dir.parent().unwrap(),
            base.join("rocm").as_path(),
        ] {
            let mode = std::fs::metadata(private).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "{private:?} must be private to the user");
        }

        let elsewhere = unique_tmp("private-dirs-elsewhere");
        let _ = std::fs::remove_dir_all(&elsewhere);
        std::fs::create_dir_all(&elsewhere).unwrap();
        std::os::unix::fs::symlink(&elsewhere, base.join("ptx")).unwrap();
        assert_eq!(
            kernel_cache_root("ptx", Target::Nvptx),
            None,
            "a symlinked kind directory is not the user's own"
        );

        unsafe { std::env::remove_var("POOT_KERNEL_CACHE_DIR") };
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&elsewhere);
    }

    #[test]
    fn the_kill_switch_disables_the_cache() {
        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        for value in ["0", "off", "FALSE"] {
            unsafe { std::env::set_var("POOT_KERNEL_CACHE", value) };
            assert_eq!(
                kernel_cache_root("rocm", Target::AmdGcn(AmdArch::gfx1151())),
                None,
                "POOT_KERNEL_CACHE={value}"
            );
            assert!(
                KernelCache::open(Target::SpirvVulkan).dir().is_none(),
                "POOT_KERNEL_CACHE={value}"
            );
        }
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

    /// A complete, valid kernel (the kernel-ir fixture) renamed to `name`, so each test's key is its own.
    fn sample_body(name: &str) -> Body {
        let mut body = poot_kernel_ir::fixtures::add_kernel();
        body.name = name.to_string();
        body
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

    /// SC-003: one body on two targets is two kernels. The caches share an epoch string and a file
    /// extension, so only the target half of the id tells the Vulkan and OpenCL SPIR-V keys apart; and
    /// different bodies on one target differ. Mutation: leave the target out of `KernelId::of`; the keys
    /// are equal.
    #[test]
    fn one_body_on_two_targets_has_two_keys() {
        let dir = unique_tmp("target-in-key");
        let body = sample_body("target_in_key");
        let epoch = "0000000000000001";
        let vulkan = KernelCache::with_epoch(dir.clone(), epoch.into(), Target::SpirvVulkan);
        let opencl = KernelCache::with_epoch(dir.clone(), epoch.into(), Target::SpirvOpenCl);
        assert_eq!(
            Target::SpirvVulkan.output_ext(),
            Target::SpirvOpenCl.output_ext(),
            "the fixture needs two targets that share an extension"
        );

        assert_ne!(vulkan.key(&body).id(), opencl.key(&body).id());
        assert_ne!(vulkan.key(&body), opencl.key(&body));
        let other = vulkan.key(&sample_body("some_other_kernel"));
        assert_ne!(vulkan.key(&body).id(), other.id());
        let gfx1100 = KernelId::of(&body, Target::AmdGcn(AmdArch::new("gfx1100", 32)));
        let gfx1151 = KernelId::of(&body, Target::AmdGcn(AmdArch::gfx1151()));
        assert_ne!(gfx1100, gfx1151, "an AMD arch is part of the target");
    }

    /// SC-002: an entry whose recorded id is another kernel's is a miss even when its digest is intact (the
    /// digest binds the key, which a file copied between two keys' names can satisfy). Mutation: skip the
    /// id comparison in `load`; the foreign artifact is served.
    #[test]
    fn an_entry_stored_for_another_body_is_a_miss() {
        let dir = unique_tmp("foreign-id");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cache = KernelCache::with_epoch(dir.clone(), "0000000000000001".into(), Target::Nvptx);
        let key = cache.key(&sample_body("wanted"));
        let foreign = cache.key(&sample_body("foreign")).id();
        assert_ne!(foreign, key.id());

        let artifact = b"artifact of the foreign body";
        let mut entry = ENTRY_MAGIC.to_vec();
        entry.extend_from_slice(&foreign.0);
        entry.extend_from_slice(&entry_digest(&key, artifact));
        entry.extend_from_slice(artifact);
        std::fs::write(dir.join(key.as_str()), &entry).unwrap();

        assert_eq!(load(&cache, &key), None);
        let _ = std::fs::remove_dir_all(&dir);
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
        let dir = kernel_cache_root("rocm", target).expect("an explicit, creatable directory");
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

    const DAY: Duration = Duration::from_secs(24 * 60 * 60);

    fn set_mtime(path: &Path, modified: SystemTime) {
        std::fs::File::open(path)
            .and_then(|f| f.set_modified(modified))
            .unwrap();
    }

    /// `<root>/<name>/{EPOCH,gfx1151/entry.hsaco}`, the epoch last changed `age` before `now`. Returns the
    /// `<mcpu>` directory.
    fn make_epoch(root: &Path, name: &str, age: Duration, now: SystemTime) -> PathBuf {
        let epoch_dir = root.join(name);
        let arch = epoch_dir.join("gfx1151");
        std::fs::create_dir_all(&arch).unwrap();
        std::fs::write(epoch_dir.join("EPOCH"), name).unwrap();
        std::fs::write(arch.join("entry.hsaco"), name).unwrap();
        set_mtime(&arch, now - age);
        set_mtime(&epoch_dir.join("EPOCH"), now - age);
        set_mtime(&epoch_dir, now - age);
        arch
    }

    fn write_aged(path: &Path, age: Duration, now: SystemTime) {
        std::fs::write(path, b"scratch").unwrap();
        set_mtime(path, now - age);
    }

    #[test]
    fn scratch_names_are_exactly_what_the_compiler_and_the_cache_write() {
        let artifact = Path::new("/cache/kernel_ab12.hsaco");
        let (ll, tmp) = crate::compile_scratch_paths(artifact, 4242, 7);
        let cache = KernelCache::at("/cache", Target::SpirvVulkan);
        let key = kernel_entry_key("0123456789abcdef", Target::SpirvVulkan, KernelId([9; 16]));
        for scratch in [
            ll,
            tmp,
            cache.scratch_path(Path::new("/cache"), &key, "tmp"),
            cache.scratch_path(Path::new("/cache"), &key, "out"),
        ] {
            let name = scratch.file_name().unwrap().to_str().unwrap();
            assert!(
                is_scratch_name(name),
                "{name} must be recognised as scratch"
            );
        }
        for published in [
            key.as_str(),
            "kernel_ab12.hsaco",
            "EPOCH",
            "notes.tmp",
            "a.b.c.ll",
        ] {
            assert!(
                !is_scratch_name(published),
                "{published} must not be taken for scratch"
            );
        }
    }

    // SC-001
    #[test]
    fn prune_keeps_current_and_newest_epochs_and_only_abandoned_scratch_goes() {
        let root = unique_tmp("prune-retention");
        let _ = std::fs::remove_dir_all(&root);
        let now = SystemTime::now();
        let current = make_epoch(&root, "cur", DAY, now);
        // RETAINED_EPOCHS + 2 older epochs, epoch `i` last used `2 + i` days ago.
        let older: Vec<(String, PathBuf)> = (0..RETAINED_EPOCHS + 2)
            .map(|i| {
                let name = format!("old{i:02}");
                let arch = make_epoch(&root, &name, DAY * (2 + i as u32), now);
                (name, arch)
            })
            .collect();
        write_aged(&current.join("k1-0000000000000001.hsaco"), 5 * DAY, now);
        write_aged(
            &current.join("kernel.hsaco.4242.0.ll"),
            2 * SCRATCH_MIN_AGE,
            now,
        );
        write_aged(
            &current.join("kernel.hsaco.4242.1.tmp"),
            2 * SCRATCH_MIN_AGE,
            now,
        );
        write_aged(
            &current.join("kernel.hsaco.4243.0.tmp"),
            Duration::from_secs(60),
            now,
        );
        write_aged(&current.join("notes.tmp"), 5 * DAY, now);

        let report = prune_kind_root(&root, "cur", now);

        assert_eq!(
            report,
            PruneReport {
                removed_epochs: 2,
                removed_scratch: 2,
                failures: 0
            }
        );
        assert!(root.join("cur").is_dir(), "the current epoch is kept");
        for (i, (name, _)) in older.iter().enumerate() {
            assert_eq!(
                root.join(name).exists(),
                i < RETAINED_EPOCHS,
                "{name}: only the {RETAINED_EPOCHS} newest old epochs survive"
            );
        }
        assert!(
            current.join("kernel.hsaco.4243.0.tmp").exists(),
            "a recent scratch file may belong to a live compiler and must stay"
        );
        assert!(!current.join("kernel.hsaco.4242.0.ll").exists());
        assert!(!current.join("kernel.hsaco.4242.1.tmp").exists());
        assert!(
            current.join("k1-0000000000000001.hsaco").exists(),
            "an old published entry is not scratch"
        );
        assert!(current.join("notes.tmp").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_epoch_used_within_the_day_survives_beyond_the_retained_count() {
        let root = unique_tmp("prune-live-epoch");
        let _ = std::fs::remove_dir_all(&root);
        let now = SystemTime::now();
        make_epoch(&root, "cur", Duration::ZERO, now);
        let names: Vec<String> = (0..RETAINED_EPOCHS + 3)
            .map(|i| format!("old{i:02}"))
            .collect();
        for name in &names {
            make_epoch(&root, name, Duration::from_secs(3600), now);
        }

        let report = prune_kind_root(&root, "cur", now);

        assert_eq!(report, PruneReport::default());
        assert!(names.iter().all(|n| root.join(n).is_dir()));
        let _ = std::fs::remove_dir_all(&root);
    }

    // SC-002
    #[test]
    fn prune_never_leaves_its_kind_root() {
        let base = unique_tmp("prune-containment");
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("rocm");
        let now = SystemTime::now();
        make_epoch(&root, "cur", Duration::ZERO, now);
        for i in 0..RETAINED_EPOCHS {
            make_epoch(&root, &format!("keep{i:02}"), DAY * (2 + i as u32), now);
        }
        // Past retention by name and age, but not epochs of this root.
        let old = DAY * 90;
        make_epoch(&root, "old-real", old, now);
        make_epoch(&base, "old-real", old, now); // `<base>/<epoch>`: the parent of the root
        make_epoch(&base.join("rocm-other"), "old-real", old, now); // a sibling kind
        let outside = unique_tmp("prune-containment-outside");
        let _ = std::fs::remove_dir_all(&outside);
        make_epoch(&outside, "target", old, now);
        std::os::unix::fs::symlink(outside.join("target"), root.join("linked")).unwrap();
        // Epoch-named but without the sidecar: not ours to delete.
        std::fs::create_dir_all(root.join("no-sidecar").join("gfx1151")).unwrap();
        set_mtime(&root.join("no-sidecar"), now - old);

        let report = prune_kind_root(&root, "cur", now);

        assert_eq!(report.removed_epochs, 1, "{report:?}");
        assert!(!root.join("old-real").exists());
        assert!(base.join("old-real/gfx1151/entry.hsaco").exists());
        assert!(
            base.join("rocm-other/old-real/gfx1151/entry.hsaco")
                .exists()
        );
        assert!(outside.join("target/gfx1151/entry.hsaco").exists());
        assert!(root.join("linked").is_symlink());
        assert!(root.join("no-sidecar/gfx1151").is_dir());
        let _ = std::fs::remove_dir_all(&base);
        let _ = std::fs::remove_dir_all(&outside);
    }

    // SC-003
    #[test]
    fn a_removal_failure_does_not_fail_the_cache_that_triggered_the_prune() {
        use std::os::unix::fs::PermissionsExt;

        let _guard = ENV_LOCK.lock().unwrap();
        unset_all_cache_env();
        let base = unique_tmp("prune-failure");
        let _ = std::fs::remove_dir_all(&base);
        let root = base.join("rocm");
        let now = SystemTime::now();
        // The two oldest epochs are removable; the oldest cannot be (its `<mcpu>` directory is read-only,
        // so its entry cannot be unlinked) and is pruned first.
        let stuck = make_epoch(&root, "stuck", 90 * DAY, now);
        make_epoch(&root, "plain", 80 * DAY, now);
        let kept: Vec<PathBuf> = (0..RETAINED_EPOCHS)
            .map(|i| make_epoch(&root, &format!("keep{i:02}"), DAY * (2 + i as u32), now))
            .collect();
        std::fs::set_permissions(&stuck, std::fs::Permissions::from_mode(0o555)).unwrap();
        unsafe { std::env::set_var("POOT_KERNEL_CACHE_DIR", &base) };

        let target = Target::AmdGcn(AmdArch::gfx1151());
        let cache = KernelCache::open(target);

        std::fs::set_permissions(&stuck, std::fs::Permissions::from_mode(0o755)).unwrap();
        unsafe { std::env::remove_var("POOT_KERNEL_CACHE_DIR") };
        assert!(
            stuck.join("entry.hsaco").exists(),
            "the fixture must really make the removal fail (a root user ignores permissions)"
        );
        assert!(
            !root.join("plain").exists(),
            "one failed removal must not stop the next"
        );
        assert!(kept.iter().all(|arch| arch.join("entry.hsaco").exists()));
        let body = sample_body("prune_failure");
        let key = cache.key(&body);
        let built = cache
            .load_or_build(&key, ROOMY, || Ok::<_, CompileError>(b"artifact".to_vec()))
            .unwrap();
        assert!(built.built);
        assert_eq!(
            cache.load(&key, ROOMY).unwrap().as_deref(),
            Some(&b"artifact"[..])
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    // A hit refreshes the epoch's last-use time.
    #[test]
    fn an_idle_epoch_whose_sidecar_a_hit_refreshed_survives_the_prune() {
        let root = unique_tmp("prune-refreshed");
        let _ = std::fs::remove_dir_all(&root);
        let now = SystemTime::now();
        make_epoch(&root, "cur", Duration::ZERO, now);
        for i in 0..RETAINED_EPOCHS {
            make_epoch(&root, &format!("keep{i:02}"), DAY * (2 + i as u32), now);
        }
        let used = make_epoch(&root, "used", 30 * DAY, now);
        make_epoch(&root, "unused", 30 * DAY, now);
        let cache = KernelCache::at(&used, Target::SpirvVulkan);
        let key = cache.key(&sample_body("refreshed"));
        cache.store(&key, b"artifact").unwrap();
        set_mtime(&root.join("used/EPOCH"), now - 30 * DAY);

        assert_eq!(load(&cache, &key).as_deref(), Some(&b"artifact"[..]));
        let report = prune_kind_root(&root, "cur", SystemTime::now());

        assert!(
            root.join("used").is_dir(),
            "a used epoch must survive: {report:?}"
        );
        assert!(!root.join("unused").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_hit_within_the_refresh_interval_does_not_write_the_sidecar() {
        let root = unique_tmp("refresh-interval");
        let _ = std::fs::remove_dir_all(&root);
        let now = SystemTime::now();
        let arch = make_epoch(&root, "used", Duration::ZERO, now);
        let cache = KernelCache::at(&arch, Target::SpirvVulkan);
        let key = cache.key(&sample_body("interval"));
        cache.store(&key, b"artifact").unwrap();
        let sidecar = root.join("used/EPOCH");
        let recent = now - Duration::from_secs(10 * 60);
        set_mtime(&sidecar, recent);

        assert!(load(&cache, &key).is_some());
        let mtime = |p: &Path| std::fs::metadata(p).unwrap().modified().unwrap();
        assert_eq!(
            mtime(&sidecar),
            recent,
            "a hit inside the interval must not write"
        );

        let stale = now - 2 * SIDECAR_REFRESH_INTERVAL;
        set_mtime(&sidecar, stale);
        assert!(load(&cache, &key).is_some());
        assert!(
            mtime(&sidecar) > stale,
            "a hit past the interval must refresh"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}

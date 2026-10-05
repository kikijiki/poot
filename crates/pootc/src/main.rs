//! `pootc` - poot's rustc driver (a kernel is real Rust, lowered from real rustc MIR).
//!
//! It is `rustc` plus a hook. `run_compiler` runs the normal pipeline, so rustc's own static LLVM
//! backend compiles host code into a working binary (no separate codegen-backend dylib is needed on
//! the stock nix toolchain). A `Callbacks::after_analysis` hook then reads MIR via Stable MIR
//! (`rustc_public`), finds the name-mangled `#[kernel]` functions (`poot_kernel_ir::naming::is_kernel`),
//! and imports their MIR into [`poot_kernel_ir::Body`].
//!
//! `poot-codegen` is the back half (Body -> LLVM -> llc -> SPIR-V/PTX); `pootc` supplies the front half
//! (Stable MIR -> Body).
#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_session;
extern crate rustc_span;

mod import;
mod normalize;

use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;
use rustc_public::{CrateDef, rustc_internal};

struct PootDriver {
    failed: bool,
    pending: Option<StagedArtifactSet>,
}

impl Callbacks for PootDriver {
    fn after_analysis(&mut self, _compiler: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        // Bridge into Stable MIR with the `TyCtxt` rustc gave us.
        let result = rustc_internal::run(tcx, stage_crate_artifacts);
        match result {
            Ok(Ok(pending)) => {
                self.pending = pending;
                Compilation::Continue
            }
            Ok(Err(failures)) => {
                for failure in failures {
                    eprintln!("{failure}");
                }
                self.failed = true;
                Compilation::Stop
            }
            Err(error) => {
                eprintln!(
                    "error[pootc::kernel-import]: Stable MIR bridge failed before import: {error:?}"
                );
                self.failed = true;
                Compilation::Stop
            }
        }
    }
}

#[derive(Clone, Copy)]
enum FailureClass {
    KernelImport,
    BackendLowering,
    CachePublication,
}

impl FailureClass {
    fn diagnostic_name(self) -> &'static str {
        match self {
            Self::KernelImport => "pootc::kernel-import",
            Self::BackendLowering => "pootc::backend-lowering",
            Self::CachePublication => "pootc::cache-publication",
        }
    }
}

struct PootFailure {
    class: FailureClass,
    message: String,
}

impl PootFailure {
    fn import(kernel: &str, error: impl fmt::Display) -> Self {
        Self {
            class: FailureClass::KernelImport,
            message: format!("kernel {kernel}: {error}"),
        }
    }

    fn lowering(kernel: &str, backend: &str, error: impl fmt::Display) -> Self {
        Self {
            class: FailureClass::BackendLowering,
            message: format!("kernel {kernel}, backend {backend}: {error}"),
        }
    }

    fn publication(error: impl fmt::Display) -> Self {
        Self {
            class: FailureClass::CachePublication,
            message: error.to_string(),
        }
    }
}

impl fmt::Display for PootFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error[{}]: {}",
            self.class.diagnostic_name(),
            self.message
        )
    }
}

struct ImportedKernel {
    item_name: String,
    source_name: String,
    body: poot_kernel_ir::Body,
}

fn stage_crate_artifacts() -> Result<Option<StagedArtifactSet>, Vec<PootFailure>> {
    let kernels: Vec<_> = rustc_public::all_local_items()
        .into_iter()
        .filter(|item| poot_kernel_ir::naming::is_kernel(&item.name()))
        .collect();
    if kernels.is_empty() {
        eprintln!("pootc: no #[kernel] functions found; host compilation only.");
    }

    if !kernels.is_empty() {
        eprintln!("pootc: found {} kernel(s):", kernels.len());
    }
    let mut imported = Vec::with_capacity(kernels.len());
    let mut failures = Vec::new();
    for item in &kernels {
        match import_kernel(item) {
            Ok(kernel) => imported.push(kernel),
            Err(failure) => failures.push(failure),
        }
    }
    if !failures.is_empty() {
        return Err(failures);
    }

    let Some(out_dir) = std::env::var_os("POOT_KERNEL_OUT").map(PathBuf::from) else {
        if !kernels.is_empty() {
            eprintln!(
                "pootc: import-only success; POOT_KERNEL_OUT is unset, so no artifacts were requested."
            );
        }
        return Ok(None);
    };

    stage_artifact_set(&out_dir, &imported)
        .map(Some)
        .map_err(|failure| vec![failure])
}

/// Import one kernel item. Artifact work starts only after every kernel in the crate imports.
fn import_kernel(item: &rustc_public::CrateItem) -> Result<ImportedKernel, PootFailure> {
    use rustc_public::CrateDef;

    let item_name = item.name();
    let body = import::import_item(item).map_err(|error| PootFailure::import(&item_name, error))?;
    eprintln!(
        "  {item_name}: imported -> Body {{ {} params, {} locals, {} blocks }}",
        body.param_count,
        body.locals.len(),
        body.blocks.len()
    );
    let source_name =
        poot_kernel_ir::naming::source_name_of_path(&item_name).unwrap_or_else(|| "kernel".into());
    Ok(ImportedKernel {
        item_name,
        source_name,
        body,
    })
}

/// Stage the complete requested set. The driver retains it until rustc host code generation succeeds.
fn stage_artifact_set(
    out_dir: &Path,
    kernels: &[ImportedKernel],
) -> Result<StagedArtifactSet, PootFailure> {
    validate_output_directory(out_dir)?;
    validate_unique_source_names(kernels)?;

    let stage = StagingDir::create(out_dir)?;
    let mut files = Vec::with_capacity(kernels.len() * 3 + usize::from(!kernels.is_empty()));
    for kernel in kernels {
        let json_name = format!("{}.kir.json", kernel.source_name);
        let json = serde_json::to_string_pretty(&kernel.body).map_err(|error| {
            PootFailure::publication(format!(
                "could not serialize {json_name} for {}: {error}",
                kernel.item_name
            ))
        })?;
        std::fs::write(stage.path().join(&json_name), json).map_err(|error| {
            PootFailure::publication(format!(
                "could not stage {json_name} in {}: {error}",
                out_dir.display()
            ))
        })?;
        files.push(json_name);

        for (backend, target, extension) in [
            ("spirv-vulkan", poot_codegen::Target::SpirvVulkan, "spv"),
            ("nvptx", poot_codegen::Target::Nvptx, "ptx"),
        ] {
            let artifact_name = format!("{}.{}", kernel.source_name, extension);
            poot_codegen::compile(&kernel.body, target, &stage.path().join(&artifact_name))
                .map_err(|error| PootFailure::lowering(&kernel.item_name, backend, error))?;
            files.push(artifact_name);
        }
        eprintln!(
            "  {}: staged {}.kir.json + {}.spv + {}.ptx",
            kernel.item_name, kernel.source_name, kernel.source_name, kernel.source_name
        );
    }

    if !kernels.is_empty() {
        let cache_name = "poot_cache.rs".to_string();
        std::fs::write(
            stage.path().join(&cache_name),
            render_cache(kernels, &files),
        )
        .map_err(|error| {
            PootFailure::publication(format!(
                "could not stage {cache_name} in {}: {error}",
                out_dir.display()
            ))
        })?;
        // The cache is the success marker and must be the final published file.
        files.push(cache_name);
    }
    Ok(StagedArtifactSet {
        out_dir: out_dir.to_path_buf(),
        stage,
        files,
        kernel_count: kernels.len(),
    })
}

fn validate_output_directory(out_dir: &Path) -> Result<(), PootFailure> {
    if out_dir.as_os_str().is_empty() {
        return Err(PootFailure::publication(
            "POOT_KERNEL_OUT must name an existing directory, not an empty path",
        ));
    }
    let metadata = std::fs::metadata(out_dir).map_err(|error| {
        PootFailure::publication(format!(
            "POOT_KERNEL_OUT {} is not an accessible directory: {error}",
            out_dir.display()
        ))
    })?;
    if !metadata.is_dir() {
        return Err(PootFailure::publication(format!(
            "POOT_KERNEL_OUT {} is not a directory",
            out_dir.display()
        )));
    }
    Ok(())
}

fn validate_unique_source_names(kernels: &[ImportedKernel]) -> Result<(), PootFailure> {
    let mut names = HashSet::with_capacity(kernels.len());
    for kernel in kernels {
        if !names.insert(&kernel.source_name) {
            return Err(PootFailure::publication(format!(
                "duplicate kernel artifact name {:?}; source names must be unique within one crate",
                kernel.source_name
            )));
        }
    }
    Ok(())
}

static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);

struct StagingDir {
    path: PathBuf,
    cleanup: bool,
}

impl StagingDir {
    fn create(out_dir: &Path) -> Result<Self, PootFailure> {
        for _ in 0..16 {
            let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let path = out_dir.join(format!(".pootc-stage-{}-{nonce}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        cleanup: true,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(PootFailure::publication(format!(
                        "could not create a staging directory in {}: {error}",
                        out_dir.display()
                    )));
                }
            }
        }
        Err(PootFailure::publication(format!(
            "could not allocate a unique staging directory in {}",
            out_dir.display()
        )))
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn cleanup(&mut self) -> std::io::Result<()> {
        if self.cleanup {
            std::fs::remove_dir_all(&self.path)?;
            self.cleanup = false;
        }
        Ok(())
    }

    fn preserve_for_recovery(&mut self, out_dir: &Path) -> (PathBuf, Option<String>) {
        for _ in 0..16 {
            let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
            let recovery = out_dir.join(format!(".pootc-recovery-{}-{nonce}", std::process::id()));
            match std::fs::rename(&self.path, &recovery) {
                Ok(()) => {
                    self.path = recovery.clone();
                    self.cleanup = false;
                    return (recovery, None);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    self.cleanup = false;
                    let retained = self.path.clone();
                    return (
                        retained,
                        Some(format!("could not name recovery directory: {error}")),
                    );
                }
            }
        }
        self.cleanup = false;
        (
            self.path.clone(),
            Some("could not allocate a unique recovery directory name".to_string()),
        )
    }
}

impl Drop for StagingDir {
    fn drop(&mut self) {
        if self.cleanup {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

struct StagedArtifactSet {
    out_dir: PathBuf,
    stage: StagingDir,
    files: Vec<String>,
    kernel_count: usize,
}

impl StagedArtifactSet {
    fn publish(mut self) -> Result<(), PootFailure> {
        let out_dir = self.out_dir.clone();
        let files = self.files.clone();
        publish_staged_set(&out_dir, &mut self.stage, &files)?;
        if self.kernel_count == 0 {
            eprintln!(
                "pootc: host-only success cleared prior kernel outputs in {}",
                self.out_dir.display()
            );
        } else {
            eprintln!(
                "pootc: published {} kernel(s) and poot_cache.rs to {}",
                self.kernel_count,
                self.out_dir.display()
            );
        }
        Ok(())
    }
}

/// Replace the prior pootc-owned set with all staged files. The output directory is locked from
/// ownership discovery through backup, publication or rollback, and private-directory cleanup; the OS
/// releases the lock if the process exits. Existing files are moved into the private staging directory
/// first and restored if any publish rename fails. A failed rollback keeps that directory under a
/// recovery name instead of deleting its remaining backups.
fn publish_staged_set(
    out_dir: &Path,
    stage: &mut StagingDir,
    files: &[String],
) -> Result<(), PootFailure> {
    let mut faults = NoPublicationFaults;
    publish_staged_set_with_faults(out_dir, stage, files, &mut faults)
}

fn publish_staged_set_with_faults(
    out_dir: &Path,
    stage: &mut StagingDir,
    files: &[String],
    faults: &mut impl PublicationFaults,
) -> Result<(), PootFailure> {
    let _lock = PublicationLock::acquire(out_dir)?;
    let publication = publish_staged_set_locked(out_dir, stage, files, faults);
    let cleanup = stage.cleanup();
    match (publication, cleanup) {
        (result, Ok(())) => result,
        (Ok(()), Err(error)) => Err(PootFailure::publication(format!(
            "artifact set was published but private transaction directory {} could not be removed: {error}",
            stage.path().display()
        ))),
        (Err(failure), Err(error)) => Err(PootFailure::publication(format!(
            "{}; private transaction directory {} could not be removed: {error}",
            failure.message,
            stage.path().display()
        ))),
    }
}

struct PublicationLock {
    _directory: File,
}

impl PublicationLock {
    fn acquire(out_dir: &Path) -> Result<Self, PootFailure> {
        let directory = File::open(out_dir).map_err(|error| {
            PootFailure::publication(format!(
                "could not open output directory {} for publication locking: {error}",
                out_dir.display()
            ))
        })?;
        directory.lock().map_err(|error| {
            PootFailure::publication(format!(
                "could not lock output directory {} for publication: {error}",
                out_dir.display()
            ))
        })?;
        Ok(Self {
            _directory: directory,
        })
    }
}

trait PublicationFaults {
    fn fail_after_publish(&mut self, _published_count: usize) -> bool {
        false
    }

    fn fail_before_restore(&mut self, _restore_index: usize) -> bool {
        false
    }
}

struct NoPublicationFaults;

impl PublicationFaults for NoPublicationFaults {}

fn publish_staged_set_locked(
    out_dir: &Path,
    stage: &mut StagingDir,
    files: &[String],
    faults: &mut impl PublicationFaults,
) -> Result<(), PootFailure> {
    let prior_owned = read_prior_owned_files(out_dir)?;
    let affected: BTreeSet<_> = prior_owned
        .into_iter()
        .chain(files.iter().cloned())
        .collect();
    for file in &affected {
        let destination = out_dir.join(file);
        match std::fs::symlink_metadata(&destination) {
            Ok(metadata) if metadata.file_type().is_file() || metadata.file_type().is_symlink() => {
            }
            Ok(_) => {
                return Err(PootFailure::publication(format!(
                    "cannot publish {file}: destination {} is not a replaceable file",
                    destination.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(PootFailure::publication(format!(
                    "cannot inspect destination {}: {error}",
                    destination.display()
                )));
            }
        }
    }

    let backup_dir = stage.path().join("backups");
    std::fs::create_dir(&backup_dir).map_err(|error| {
        PootFailure::publication(format!("could not prepare publication rollback: {error}"))
    })?;
    let mut backups = Vec::new();
    for file in &affected {
        let destination = out_dir.join(file);
        if std::fs::symlink_metadata(&destination).is_ok() {
            let backup = backup_dir.join(file);
            if let Err(error) = std::fs::rename(&destination, &backup) {
                return Err(publication_failure(
                    stage,
                    out_dir,
                    &backup_dir,
                    &[],
                    &backups,
                    faults,
                    format!("could not prepare {file} for atomic replacement: {error}"),
                ));
            }
            backups.push(file.clone());
        }
    }

    let mut published = Vec::new();
    for file in files {
        let staged = stage.path().join(file);
        let destination = out_dir.join(file);
        if let Err(error) = std::fs::rename(&staged, &destination) {
            return Err(publication_failure(
                stage,
                out_dir,
                &backup_dir,
                &published,
                &backups,
                faults,
                format!(
                    "could not publish complete artifact set at {} while replacing {file}: {error}",
                    out_dir.display()
                ),
            ));
        }
        published.push(file.clone());
        if faults.fail_after_publish(published.len()) {
            return Err(publication_failure(
                stage,
                out_dir,
                &backup_dir,
                &published,
                &backups,
                faults,
                "injected publication failure after first publish rename".to_string(),
            ));
        }
    }
    Ok(())
}

fn publication_failure(
    stage: &mut StagingDir,
    out_dir: &Path,
    backup_dir: &Path,
    published: &[String],
    backups: &[String],
    faults: &mut impl PublicationFaults,
    cause: String,
) -> PootFailure {
    match rollback_publication(out_dir, backup_dir, published, backups, faults) {
        Ok(()) => PootFailure::publication(cause),
        Err(errors) => {
            let (recovery, naming_error) = stage.preserve_for_recovery(out_dir);
            let naming = naming_error
                .map(|error| format!("; {error}"))
                .unwrap_or_default();
            PootFailure::publication(format!(
                "{cause}; rollback failed: {}; recovery data retained at {}{naming}",
                errors.join(", "),
                recovery.display()
            ))
        }
    }
}

fn rollback_publication(
    out_dir: &Path,
    backup_dir: &Path,
    published: &[String],
    backups: &[String],
    faults: &mut impl PublicationFaults,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for file in published.iter().rev() {
        let destination = out_dir.join(file);
        if let Err(error) = std::fs::remove_file(&destination) {
            errors.push(format!("remove {}: {error}", destination.display()));
        }
    }
    for (index, file) in backups.iter().rev().enumerate() {
        let backup = backup_dir.join(file);
        let destination = out_dir.join(file);
        if faults.fail_before_restore(index) {
            errors.push(format!(
                "restore {}: injected rollback failure",
                destination.display()
            ));
            continue;
        }
        if let Err(error) = std::fs::rename(&backup, &destination) {
            errors.push(format!("restore {}: {error}", destination.display()));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn read_prior_owned_files(out_dir: &Path) -> Result<BTreeSet<String>, PootFailure> {
    let cache = out_dir.join("poot_cache.rs");
    match std::fs::symlink_metadata(&cache) {
        Ok(metadata) if !metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            return Ok(BTreeSet::new());
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(BTreeSet::new());
        }
        Err(error) => {
            return Err(PootFailure::publication(format!(
                "could not inspect prior pootc cache {}: {error}",
                cache.display()
            )));
        }
    }
    let source = match std::fs::read_to_string(&cache) {
        Ok(source) => source,
        Err(error) => {
            return Err(PootFailure::publication(format!(
                "could not read prior pootc cache {}: {error}",
                cache.display()
            )));
        }
    };
    if !source.starts_with("// Generated by pootc.") {
        return Ok(BTreeSet::new());
    }

    let mut files = BTreeSet::from(["poot_cache.rs".to_string()]);
    let mut has_explicit_ownership = false;
    for line in source.lines() {
        let Some(file) = line.strip_prefix("// pootc-owned-artifact: ") else {
            continue;
        };
        validate_owned_filename(&cache, file)?;
        has_explicit_ownership = true;
        files.insert(file.to_string());
    }
    if !has_explicit_ownership {
        for line in source.lines() {
            let Some((_, rest)) = line.split_once("include_bytes!(\"") else {
                continue;
            };
            let Some((spv, _)) = rest.split_once("\")") else {
                continue;
            };
            validate_owned_filename(&cache, spv)?;
            let Some(source_name) = spv.strip_suffix(".spv") else {
                continue;
            };
            files.extend([
                format!("{source_name}.kir.json"),
                spv.to_string(),
                format!("{source_name}.ptx"),
            ]);
        }
    }
    Ok(files)
}

fn validate_owned_filename(cache: &Path, file: &str) -> Result<(), PootFailure> {
    if Path::new(file).file_name().and_then(|name| name.to_str()) == Some(file) {
        Ok(())
    } else {
        Err(PootFailure::publication(format!(
            "prior pootc cache {} records invalid owned artifact {file:?}",
            cache.display()
        )))
    }
}

mod publication_regressions {
    #[cfg(test)]
    mod enabled {
        use super::super::*;

        static TEST_NONCE: AtomicU64 = AtomicU64::new(0);

        struct TestDir(PathBuf);

        impl TestDir {
            fn new(label: &str) -> Self {
                let nonce = TEST_NONCE.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "pootc-publication-{label}-{}-{nonce}",
                    std::process::id()
                ));
                std::fs::create_dir(&path).expect("create publication test directory");
                Self(path)
            }

            fn path(&self) -> &Path {
                &self.0
            }
        }

        impl Drop for TestDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }

        struct InjectedFaults {
            fail_restore: bool,
        }

        impl PublicationFaults for InjectedFaults {
            fn fail_after_publish(&mut self, published_count: usize) -> bool {
                published_count == 1
            }

            fn fail_before_restore(&mut self, restore_index: usize) -> bool {
                self.fail_restore && restore_index == 0
            }
        }

        fn seed_prior_set(out_dir: &Path) -> Vec<(String, Vec<u8>)> {
            let mut prior = Vec::new();
            for extension in ["kir.json", "spv", "ptx"] {
                let name = format!("add.{extension}");
                let bytes = format!("prior {name}").into_bytes();
                std::fs::write(out_dir.join(&name), &bytes).unwrap();
                prior.push((name, bytes));
            }
            let cache = b"// Generated by pootc. test cache\n\
// pootc-owned-artifact: add.kir.json\n\
// pootc-owned-artifact: add.spv\n\
// pootc-owned-artifact: add.ptx\n";
            std::fs::write(out_dir.join("poot_cache.rs"), cache).unwrap();
            prior.push(("poot_cache.rs".to_string(), cache.to_vec()));
            prior
        }

        fn stage_new_set(out_dir: &Path) -> (StagingDir, Vec<String>) {
            let stage = StagingDir::create(out_dir).unwrap_or_else(|failure| panic!("{failure}"));
            let files = ["scale.kir.json", "scale.spv", "scale.ptx", "poot_cache.rs"]
                .map(str::to_string)
                .to_vec();
            for file in &files {
                std::fs::write(stage.path().join(file), format!("new {file}"))
                    .expect("stage synthetic publication file");
            }
            (stage, files)
        }

        fn assert_prior_set(out_dir: &Path, prior: &[(String, Vec<u8>)]) {
            for (name, bytes) in prior {
                assert_eq!(std::fs::read(out_dir.join(name)).unwrap(), *bytes);
            }
            for extension in ["kir.json", "spv", "ptx"] {
                assert!(!out_dir.join(format!("scale.{extension}")).exists());
            }
        }

        #[test]
        fn injected_publish_failure_rolls_back_exactly_and_cleans_private_directory() {
            let out = TestDir::new("rollback-success");
            let prior = seed_prior_set(out.path());
            let (mut stage, files) = stage_new_set(out.path());
            let mut faults = InjectedFaults {
                fail_restore: false,
            };

            let failure =
                publish_staged_set_with_faults(out.path(), &mut stage, &files, &mut faults)
                    .expect_err("injected publication failure must fail");
            assert!(
                failure.message.contains("after first publish rename")
                    && !failure.message.contains("rollback failed")
            );
            assert_prior_set(out.path(), &prior);
            assert_eq!(std::fs::read_dir(out.path()).unwrap().count(), 4);
        }

        #[test]
        fn injected_restore_failure_retains_named_recovery_data() {
            let out = TestDir::new("rollback-failure");
            let prior = seed_prior_set(out.path());
            let prior_cache = prior
                .iter()
                .find(|(name, _)| name == "poot_cache.rs")
                .unwrap()
                .1
                .clone();
            let (mut stage, files) = stage_new_set(out.path());
            let mut faults = InjectedFaults { fail_restore: true };

            let failure =
                publish_staged_set_with_faults(out.path(), &mut stage, &files, &mut faults)
                    .expect_err("injected restore failure must fail");
            assert!(
                failure.message.contains("rollback failed")
                    && failure.message.contains("recovery data retained at")
            );
            let recovery: Vec<_> = std::fs::read_dir(out.path())
                .unwrap()
                .flatten()
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".pootc-recovery-")
                })
                .collect();
            assert_eq!(recovery.len(), 1);
            let recovery = recovery[0].path();
            assert!(failure.message.contains(&recovery.display().to_string()));
            assert_eq!(
                std::fs::read(recovery.join("backups/poot_cache.rs")).unwrap(),
                prior_cache
            );
            assert!(recovery.join("scale.spv").is_file());
            assert!(recovery.join("scale.ptx").is_file());
            assert!(!out.path().join("scale.kir.json").exists());
            assert!(!out.path().join("poot_cache.rs").exists());
            for (name, bytes) in prior.iter().filter(|(name, _)| name != "poot_cache.rs") {
                assert_eq!(std::fs::read(out.path().join(name)).unwrap(), *bytes);
            }
        }
    }
}

/// Render `poot_cache.rs`: a host crate `include!`s it to get each kernel's SPIR-V and compiled `Body`
/// embedded by name. The `Body` (card 608) lets the `#[kernel]` host wrapper build
/// its `CompiledKernel` through `poot_codegen::kernel_handle`, the same verified source every other handle
/// comes from, instead of guessing the schema and trap flag from the Rust signature; a host crate that
/// `include!`s this file therefore needs `poot-kernel-ir` and `serde_json` as dependencies alongside
/// `poot-runtime` and `poot-codegen`.
fn render_cache(kernels: &[ImportedKernel], artifact_files: &[String]) -> String {
    let mut source = String::from(
        "// Generated by pootc. The embedded SPIR-V module + kernel-IR Body cache for this crate's\n\
         // #[kernel]s. A host crate `include!`s this at its crate root; the #[kernel] host wrapper calls\n\
         // `poot_kernel_spv` and `poot_kernel_body`.\n",
    );
    for file in artifact_files {
        source.push_str(&format!("// pootc-owned-artifact: {file}\n"));
    }
    source.push_str("pub static POOT_KERNELS: &[(&str, &[u8])] = &[\n");
    for kernel in kernels {
        source.push_str(&format!(
            "    ({:?}, include_bytes!({:?})),\n",
            kernel.source_name,
            format!("{}.spv", kernel.source_name)
        ));
    }
    source.push_str("];\n\n");
    source.push_str("pub static POOT_KERNEL_BODIES: &[(&str, &str)] = &[\n");
    for kernel in kernels {
        source.push_str(&format!(
            "    ({:?}, include_str!({:?})),\n",
            kernel.source_name,
            format!("{}.kir.json", kernel.source_name)
        ));
    }
    source.push_str("];\n\n");
    source.push_str(
        "/// The SPIR-V for a #[kernel] by source name; panics if absent (a build/cache mismatch).\n\
         pub fn poot_kernel_spv(name: &str) -> &'static [u8] {\n\
        \x20   POOT_KERNELS.iter().find(|(n, _)| *n == name)\n\
        \x20       .unwrap_or_else(|| panic!(\"kernel {name:?} not in the pootc module cache\")).1\n\
         }\n\n\
         /// The compiled kernel-IR `Body` for a #[kernel] by source name (card 608): the host wrapper reads\n\
         /// its `CompiledKernel`'s argument schema and trap flag from this, the same `Body` the device pass\n\
         /// compiled the SPIR-V from, not from the Rust function signature.\n\
         pub fn poot_kernel_body(name: &str) -> ::poot_kernel_ir::Body {\n\
        \x20   let json = POOT_KERNEL_BODIES.iter().find(|(n, _)| *n == name)\n\
        \x20       .unwrap_or_else(|| panic!(\"kernel {name:?} not in the pootc module cache\")).1;\n\
        \x20   ::serde_json::from_str(json)\n\
        \x20       .unwrap_or_else(|e| panic!(\"kernel {name:?}'s cached Body JSON is invalid: {e}\"))\n\
         }\n",
    );
    source
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    // Wrapper mode (`RUSTC_WORKSPACE_WRAPPER`): cargo invokes `pootc <path/to/rustc> <args...>`. pootc is
    // the compiler, so drop the wrapped-compiler path (argv[1] stem == "rustc").
    if args.len() > 1
        && std::path::Path::new(&args[1])
            .file_stem()
            .and_then(|s| s.to_str())
            == Some("rustc")
    {
        args.remove(1);
    }
    // A custom driver is not inside the sysroot, so pass `--sysroot` explicitly.
    if !args.iter().any(|a| a == "--sysroot")
        && let Ok(out) = std::process::Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
        && let Ok(s) = String::from_utf8(out.stdout)
    {
        args.push("--sysroot".into());
        args.push(s.trim().into());
    }
    // GPUs have no panic path: integer overflow wraps in hardware. Disable rustc's debug overflow checks
    // so MIR uses plain wrapping Add/Sub/Mul (the importer cannot honor SubWithOverflow -> Assert).
    if !args
        .iter()
        .any(|a| a.starts_with("-Coverflow-checks") || a.starts_with("overflow-checks"))
    {
        args.push("-C".into());
        args.push("overflow-checks=no".into());
    }
    let mut driver = PootDriver {
        failed: false,
        pending: None,
    };
    rustc_driver::run_compiler(&args, &mut driver);
    if driver.failed {
        std::process::exit(1);
    }
    if let Some(pending) = driver.pending
        && let Err(failure) = pending.publish()
    {
        eprintln!("{failure}");
        std::process::exit(1);
    }
}

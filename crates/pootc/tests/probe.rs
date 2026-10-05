//! Process-level contract tests for the `pootc` rustc driver.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, OnceLock};

static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// `CARGO_MANIFEST_DIR`, read at runtime (not baked via `env!`): the compile-time macro's value is
/// embedded into this test binary's compiled object code at build time, and a shared compile cache (kache)
/// that reuses that object across worktrees by source-content hash would then serve whichever worktree's
/// path happened to compile it first (card 530's build.rs fix; card 543 review). `std::env::var` reads the
/// environment cargo sets fresh for every test-binary invocation, so it is correct regardless of which
/// worktree compiled the binary.
fn manifest_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries"),
        )
    })
}

/// `CARGO_BIN_EXE_pootc`, read at runtime for the same reason as [`manifest_dir`].
fn pootc_exe() -> &'static str {
    static EXE: OnceLock<String> = OnceLock::new();
    EXE.get_or_init(|| {
        std::env::var("CARGO_BIN_EXE_pootc")
            .expect("CARGO_BIN_EXE_pootc must be set by cargo for test binaries")
    })
    .as_str()
}

/// Build `poot-kernel-intrinsics` from this checkout and return its rlib path, memoized for the process
/// (every `pootc_command` needs `--extern poot_kernel_intrinsics=...`; kernel sources depend on it instead
/// of redeclaring their own intrinsic stubs, card 531c). The target directory is keyed by the checkout so
/// no other worktree's build lands in a shared slot target dir (mirrors `macro_kernel.rs`'s
/// `build_proc_macro`).
fn kernel_intrinsics_rlib() -> &'static Path {
    static RLIB: OnceLock<PathBuf> = OnceLock::new();
    RLIB.get_or_init(|| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let manifest_dir = manifest_dir();
        let mut checkout = DefaultHasher::new();
        manifest_dir.hash(&mut checkout);
        let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("kernel-intrinsics-{:016x}", checkout.finish()));
        let output = Command::new(env!("CARGO"))
            .args([
                "build",
                "--locked",
                "--message-format=json-render-diagnostics",
            ])
            .arg("--manifest-path")
            .arg(manifest_dir.join("../poot-kernel-intrinsics/Cargo.toml"))
            .arg("--target-dir")
            .arg(&target_dir)
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .output()
            .expect("run cargo to build poot-kernel-intrinsics");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "building poot-kernel-intrinsics failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        let libraries: Vec<PathBuf> = stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|message| message["reason"] == "compiler-artifact")
            .filter(|message| message["target"]["name"] == "poot_kernel_intrinsics")
            .filter(|message| message["target"]["kind"] == serde_json::json!(["lib"]))
            .flat_map(|message| message["filenames"].as_array().cloned().unwrap_or_default())
            .filter_map(|name| name.as_str().map(PathBuf::from))
            .filter(|path| path.extension().is_some_and(|ext| ext == "rlib"))
            .collect();
        let [library] = libraries.as_slice() else {
            panic!(
                "cargo reported {} rlibs for poot-kernel-intrinsics, wanted exactly one: {libraries:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                libraries.len()
            );
        };
        library.clone()
    })
}

/// `--extern poot_kernel_intrinsics=<rlib>`, appended to every pootc invocation that compiles a kernel
/// source (they all depend on the crate now; card 531c).
fn kernel_intrinsics_extern() -> String {
    format!(
        "poot_kernel_intrinsics={}",
        kernel_intrinsics_rlib().display()
    )
}

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "pootc-probe-{label}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create isolated pootc test directory");
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

fn kernel(stem: &str) -> PathBuf {
    manifest_dir()
        .join("tests/kernels")
        .join(format!("{stem}.rs"))
}

fn run_pootc(
    source: &Path,
    work: &TestDir,
    kernel_out: Option<&Path>,
    llc: Option<&Path>,
) -> Output {
    run_pootc_with(source, work, kernel_out, llc, None, &[])
}

fn run_pootc_with(
    source: &Path,
    work: &TestDir,
    kernel_out: Option<&Path>,
    llc: Option<&Path>,
    host_out: Option<&Path>,
    environment: &[(&str, &str)],
) -> Output {
    pootc_command(source, work, kernel_out, llc, host_out, environment)
        .output()
        .expect("run pootc")
}

fn pootc_command(
    source: &Path,
    work: &TestDir,
    kernel_out: Option<&Path>,
    llc: Option<&Path>,
    host_out: Option<&Path>,
    environment: &[(&str, &str)],
) -> Command {
    let mut command = Command::new(pootc_exe());
    command
        // 2021 (uniform paths): kernel sources `use poot_kernel_intrinsics::...;` without a leading
        // `extern crate` (card 531c); the pre-2018 default edition rejects that as an unresolved import.
        .args(["--edition", "2021", "--crate-type", "lib"])
        .arg(source)
        .arg("-o")
        .arg(host_out.unwrap_or(&work.path().join("host.rlib")))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env_remove("POOT_KERNEL_OUT")
        .env_remove("POOT_LLC")
        .env_remove("POOTC_TEST_FAIL_AFTER_FIRST_PUBLISH")
        .env_remove("POOTC_TEST_FAIL_FIRST_ROLLBACK_RESTORE");
    if let Some(out) = kernel_out {
        command.env("POOT_KERNEL_OUT", out);
    }
    if let Some(llc) = llc {
        command.env("POOT_LLC", llc);
    }
    command.envs(environment.iter().copied());
    command
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn assert_no_success_cache(out: &Path) {
    assert!(
        !out.join("poot_cache.rs").is_file(),
        "failed pootc invocation must not publish a success cache in {}",
        out.display()
    );
}

fn assert_no_kernel_artifacts(out: &Path, source_names: &[&str]) {
    for source_name in source_names {
        for extension in ["kir.json", "spv", "ptx"] {
            let artifact = out.join(format!("{source_name}.{extension}"));
            assert!(
                !artifact.exists(),
                "failed pootc invocation published partial artifact {}",
                artifact.display()
            );
        }
    }
    assert_no_success_cache(out);
    assert_no_internal_directories(out);
}

fn assert_no_internal_directories(out: &Path) {
    let internal: Vec<_> = std::fs::read_dir(out)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with(".pootc-stage-") || name.starts_with(".pootc-recovery-")
        })
        .collect();
    assert!(
        internal.is_empty(),
        "pootc staging/recovery directories leaked: {internal:?}"
    );
}

fn artifact_bytes(out: &Path, source_names: &[&str]) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    for source_name in source_names {
        for extension in ["kir.json", "spv", "ptx"] {
            let name = format!("{source_name}.{extension}");
            files.push((
                name.clone(),
                std::fs::read(out.join(&name)).unwrap_or_else(|error| {
                    panic!("read artifact {}: {error}", out.join(name).display())
                }),
            ));
        }
    }
    files.push((
        "poot_cache.rs".to_string(),
        std::fs::read(out.join("poot_cache.rs")).expect("read poot cache"),
    ));
    files
}

fn assert_artifact_bytes(out: &Path, expected: &[(String, Vec<u8>)]) {
    for (name, bytes) in expected {
        assert_eq!(
            std::fs::read(out.join(name)).unwrap_or_else(|error| {
                panic!("read artifact {}: {error}", out.join(name).display())
            }),
            *bytes,
            "artifact {name} changed"
        );
    }
}

#[test]
fn valid_kernel_import_only_succeeds_without_artifacts() {
    let work = TestDir::new("import-only");
    let output = run_pootc(&kernel("add"), &work, None, None);
    let stderr = stderr(&output);

    assert!(output.status.success(), "stderr:\n{stderr}");
    assert!(
        stderr.contains("found 1 kernel(s)")
            && stderr.contains("__poot_kernel_add: imported -> Body { 3 params")
            && stderr.contains("import-only success"),
        "pootc should discover and import the kernel without requesting artifacts; stderr:\n{stderr}"
    );
    assert_no_kernel_artifacts(work.path(), &["add"]);
}

#[test]
fn rejection_fixtures_exit_nonzero_with_named_diagnostics_and_no_cache() {
    let cases = [
        ("reject", "__poot_kernel_bad", "helper"),
        ("reject_rev", "__poot_kernel_rev", "`.rev()`/`.step_by()`"),
        // Card 531c SC-004 (R468-009): a kernel-local `fn workgroup_barrier()` is not the shared
        // `poot_kernel_intrinsics` declaration, so pootc rejects the call as an ordinary (unsupported)
        // function call rather than silently lowering it as the workgroup-barrier terminator. Mutation:
        // revert the importer to recognize an intrinsic by the callee's last path segment again
        // (`kernel_intrinsic_name` -> `callee_last_seg` in `thread_index_axis`/the workgroup-barrier
        // match in `crates/pootc/src/import.rs`); this fixture then wrongly imports and the row goes red.
        (
            "fake_workgroup_barrier",
            "__poot_kernel_fake_workgroup_barrier",
            "workgroup_barrier",
        ),
    ];
    for (fixture, kernel_name, reason) in cases {
        let work = TestDir::new(fixture);
        let output = run_pootc(&kernel(fixture), &work, Some(work.path()), None);
        let stderr = stderr(&output);

        assert!(
            !output.status.success(),
            "{fixture} must fail the pootc process; stderr:\n{stderr}"
        );
        assert!(
            stderr.contains("error[pootc::kernel-import]")
                && stderr.contains(kernel_name)
                && stderr.contains("unsupported in kernel")
                && stderr.contains(reason),
            "{fixture} should have a named import diagnostic; stderr:\n{stderr}"
        );
        assert_no_success_cache(work.path());
    }
}

#[test]
fn mixed_valid_invalid_module_publishes_nothing() {
    let work = TestDir::new("mixed");
    let output = run_pootc(
        &kernel("mixed_valid_invalid"),
        &work,
        Some(work.path()),
        None,
    );
    let stderr = stderr(&output);

    assert!(!output.status.success(), "stderr:\n{stderr}");
    assert!(
        stderr.contains("error[pootc::kernel-import]")
            && stderr.contains("__poot_kernel_bad_sibling")
            && stderr.contains("unsupported_helper"),
        "invalid sibling should be named; stderr:\n{stderr}"
    );
    assert_no_kernel_artifacts(work.path(), &["good", "bad_sibling"]);
}

#[test]
fn invalid_output_directory_exits_nonzero_without_cache() {
    let work = TestDir::new("invalid-output");
    let not_directory = work.path().join("not-a-directory");
    std::fs::write(&not_directory, "occupied").unwrap();
    let output = run_pootc(&kernel("add"), &work, Some(&not_directory), None);
    let stderr = stderr(&output);

    assert!(!output.status.success(), "stderr:\n{stderr}");
    assert!(
        stderr.contains("error[pootc::cache-publication]")
            && stderr.contains("POOT_KERNEL_OUT")
            && stderr.contains("not a directory"),
        "invalid output path should have a named publication diagnostic; stderr:\n{stderr}"
    );
    assert_eq!(std::fs::read_to_string(&not_directory).unwrap(), "occupied");
}

#[test]
fn backend_lowering_failure_exits_nonzero_and_publishes_nothing() {
    let work = TestDir::new("lowering-failure");
    let missing_llc = work.path().join("llc-does-not-exist");
    let output = run_pootc(&kernel("add"), &work, Some(work.path()), Some(&missing_llc));
    let stderr = stderr(&output);

    assert!(!output.status.success(), "stderr:\n{stderr}");
    assert!(
        stderr.contains("error[pootc::backend-lowering]")
            && stderr.contains("__poot_kernel_add")
            && stderr.contains("backend spirv-vulkan")
            && stderr.contains("No such file or directory"),
        "backend failure should name its class, kernel, and backend; stderr:\n{stderr}"
    );
    assert_no_kernel_artifacts(work.path(), &["add"]);
}

#[test]
fn unusable_cache_destination_keeps_prior_state_and_publishes_no_artifacts() {
    let work = TestDir::new("cache-destination");
    let cache_destination = work.path().join("poot_cache.rs");
    std::fs::create_dir(&cache_destination).unwrap();
    let sentinel = cache_destination.join("sentinel");
    std::fs::write(&sentinel, "prior state").unwrap();
    let output = run_pootc(&kernel("add"), &work, Some(work.path()), None);
    let stderr = stderr(&output);

    assert!(!output.status.success(), "stderr:\n{stderr}");
    assert!(
        stderr.contains("error[pootc::cache-publication]")
            && stderr.contains("poot_cache.rs")
            && stderr.contains("not a replaceable file"),
        "cache publication failure should be named; stderr:\n{stderr}"
    );
    assert_eq!(std::fs::read_to_string(sentinel).unwrap(), "prior state");
    assert_no_kernel_artifacts(work.path(), &["add"]);
}

#[test]
fn requested_artifact_set_succeeds_and_is_complete() {
    let work = TestDir::new("artifact-success");
    let output = run_pootc(&kernel("add"), &work, Some(work.path()), None);
    let stderr = stderr(&output);

    assert!(output.status.success(), "stderr:\n{stderr}");
    for extension in ["kir.json", "spv", "ptx"] {
        assert!(
            work.path().join(format!("add.{extension}")).is_file(),
            "successful requested set is missing add.{extension}"
        );
    }
    let cache = std::fs::read_to_string(work.path().join("poot_cache.rs"))
        .expect("successful requested set has a cache");
    assert!(cache.contains("(\"add\", include_bytes!(\"add.spv\"))"));
    assert!(stderr.contains("published 1 kernel(s) and poot_cache.rs"));
    assert!(cache.contains("// pootc-owned-artifact: add.kir.json"));
}

#[test]
fn host_only_crate_succeeds_without_kernel_cache() {
    let work = TestDir::new("host-only");
    let output = run_pootc(&kernel("host_only"), &work, Some(work.path()), None);
    let stderr = stderr(&output);

    assert!(output.status.success(), "stderr:\n{stderr}");
    assert!(
        stderr.contains("no #[kernel] functions found; host compilation only"),
        "host-only mode should be explicit; stderr:\n{stderr}"
    );
    assert!(work.path().join("host.rlib").is_file());
    assert_no_success_cache(work.path());
}

#[test]
fn late_host_output_failure_publishes_nothing_and_preserves_prior_set_byte_for_byte() {
    let work = TestDir::new("late-host-failure");
    let initial = run_pootc(&kernel("add"), &work, Some(work.path()), None);
    assert!(initial.status.success(), "stderr:\n{}", stderr(&initial));
    let prior = artifact_bytes(work.path(), &["add"]);

    let missing_host_parent = work.path().join("missing-host-parent/host.rlib");
    let failed = run_pootc_with(
        &kernel("scale"),
        &work,
        Some(work.path()),
        None,
        Some(&missing_host_parent),
        &[],
    );
    let failed_stderr = stderr(&failed);

    assert!(!failed.status.success(), "stderr:\n{failed_stderr}");
    assert!(
        failed_stderr.contains("couldn't create") || failed_stderr.contains("No such file"),
        "rustc should fail while writing the late host output; stderr:\n{failed_stderr}"
    );
    assert_artifact_bytes(work.path(), &prior);
    for extension in ["kir.json", "spv", "ptx"] {
        assert!(!work.path().join(format!("scale.{extension}")).exists());
    }
    assert_no_internal_directories(work.path());
}

#[test]
fn successful_host_only_reuse_clears_prior_owned_set() {
    let work = TestDir::new("host-only-reuse");
    let initial = run_pootc(&kernel("add"), &work, Some(work.path()), None);
    assert!(initial.status.success(), "stderr:\n{}", stderr(&initial));

    let host_only = run_pootc(&kernel("host_only"), &work, Some(work.path()), None);
    let host_stderr = stderr(&host_only);
    assert!(host_only.status.success(), "stderr:\n{host_stderr}");
    assert!(host_stderr.contains("host-only success cleared prior kernel outputs"));
    assert_no_kernel_artifacts(work.path(), &["add"]);
}

#[test]
fn successful_reduced_set_reuse_removes_stale_entries_and_files() {
    let work = TestDir::new("reduced-set-reuse");
    let initial = run_pootc(&kernel("two_valid"), &work, Some(work.path()), None);
    assert!(initial.status.success(), "stderr:\n{}", stderr(&initial));
    for source_name in ["add", "scale"] {
        for extension in ["kir.json", "spv", "ptx"] {
            assert!(
                work.path()
                    .join(format!("{source_name}.{extension}"))
                    .is_file()
            );
        }
    }
    let initial_cache = std::fs::read_to_string(work.path().join("poot_cache.rs")).unwrap();
    assert!(initial_cache.contains("add") && initial_cache.contains("scale"));

    let reduced = run_pootc(&kernel("add"), &work, Some(work.path()), None);
    let reduced_stderr = stderr(&reduced);
    assert!(reduced.status.success(), "stderr:\n{reduced_stderr}");
    for extension in ["kir.json", "spv", "ptx"] {
        assert!(work.path().join(format!("add.{extension}")).is_file());
        assert!(!work.path().join(format!("scale.{extension}")).exists());
    }
    let cache = std::fs::read_to_string(work.path().join("poot_cache.rs")).unwrap();
    assert!(cache.contains("(\"add\", include_bytes!(\"add.spv\"))"));
    assert!(
        !cache.contains("scale"),
        "stale cache entry remained:\n{cache}"
    );
    assert_no_internal_directories(work.path());
}

#[test]
fn normal_binary_ignores_removed_publication_fault_environment_variables() {
    let work = TestDir::new("removed-fault-environment");
    let initial = run_pootc(&kernel("add"), &work, Some(work.path()), None);
    assert!(initial.status.success(), "stderr:\n{}", stderr(&initial));

    let scaled = run_pootc_with(
        &kernel("scale"),
        &work,
        Some(work.path()),
        None,
        None,
        &[
            ("POOTC_TEST_FAIL_AFTER_FIRST_PUBLISH", "1"),
            ("POOTC_TEST_FAIL_FIRST_ROLLBACK_RESTORE", "1"),
        ],
    );
    let scaled_stderr = stderr(&scaled);
    assert!(
        scaled.status.success(),
        "removed test variables must not alter the production binary; stderr:\n{scaled_stderr}"
    );
    for extension in ["kir.json", "spv", "ptx"] {
        assert!(!work.path().join(format!("add.{extension}")).exists());
        assert!(work.path().join(format!("scale.{extension}")).is_file());
    }
    let cache = std::fs::read_to_string(work.path().join("poot_cache.rs")).unwrap();
    assert!(cache.contains("(\"scale\", include_bytes!(\"scale.spv\"))"));
    assert!(!cache.contains("add"));
    assert_no_internal_directories(work.path());
}

#[test]
fn concurrent_publishers_never_leave_mixed_artifact_sets() {
    let work = TestDir::new("concurrent-publication");

    for round in 0..12 {
        let out = work.path().join(format!("round-{round}"));
        std::fs::create_dir(&out).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let run = |mut command: Command, barrier: Arc<Barrier>| {
            std::thread::spawn(move || {
                barrier.wait();
                command.output().expect("run concurrent pootc publisher")
            })
        };
        let add_host = work.path().join(format!("add-{round}.rlib"));
        let scale_host = work.path().join(format!("scale-{round}.rlib"));
        let add = run(
            pootc_command(
                &kernel("add"),
                &work,
                Some(&out),
                None,
                Some(&add_host),
                &[],
            ),
            Arc::clone(&barrier),
        );
        let scale = run(
            pootc_command(
                &kernel("scale"),
                &work,
                Some(&out),
                None,
                Some(&scale_host),
                &[],
            ),
            Arc::clone(&barrier),
        );
        barrier.wait();
        let add = add.join().expect("join add publisher");
        let scale = scale.join().expect("join scale publisher");
        assert!(
            add.status.success(),
            "round {round} add stderr:\n{}",
            stderr(&add)
        );
        assert!(
            scale.status.success(),
            "round {round} scale stderr:\n{}",
            stderr(&scale)
        );

        let cache = std::fs::read_to_string(out.join("poot_cache.rs")).unwrap();
        let owns_add = cache.contains("(\"add\", include_bytes!(\"add.spv\"))");
        let owns_scale = cache.contains("(\"scale\", include_bytes!(\"scale.spv\"))");
        assert_ne!(
            owns_add, owns_scale,
            "round {round} cache must describe exactly one winning set:\n{cache}"
        );
        for extension in ["kir.json", "spv", "ptx"] {
            assert_eq!(
                out.join(format!("add.{extension}")).is_file(),
                owns_add,
                "round {round} add files disagree with cache"
            );
            assert_eq!(
                out.join(format!("scale.{extension}")).is_file(),
                owns_scale,
                "round {round} scale files disagree with cache"
            );
        }
        assert_no_internal_directories(&out);
        let unexpected: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .flatten()
            .filter(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name != "poot_cache.rs" && !name.starts_with("add.") && !name.starts_with("scale.")
            })
            .collect();
        assert!(
            unexpected.is_empty(),
            "round {round} left lock/private debris: {unexpected:?}"
        );
    }
}

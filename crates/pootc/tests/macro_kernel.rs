//! Verify the `#[kernel]` attribute end to end: a kernel authored with `#[kernel] fn add(..)` is renamed
//! by the macro into `__poot_kernel_add`, which `pootc` discovers, imports, and compiles to a dispatchable
//! SPIR-V module that runs on the GPU. The proc-macro artifact is wired in via `--extern`.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::env::consts::DLL_EXTENSION;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use poot_kernel_ir::Body;
use poot_runtime::{Context, KernelBuffer};

/// `CARGO_MANIFEST_DIR`, read at runtime (not baked via `env!`): the compile-time macro's value is
/// embedded into this test binary's compiled object code at build time, and a shared compile cache (kache)
/// that reuses that object across worktrees by source-content hash would then serve whichever worktree's
/// path happened to compile it first (card 530's build.rs fix; card 543 review). `std::env::var` reads the
/// environment cargo sets fresh for every test-binary invocation, so it is correct regardless of which
/// worktree compiled the binary.
fn manifest_dir() -> &'static Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR")
                .expect("CARGO_MANIFEST_DIR must be set by cargo for test binaries"),
        )
    })
}

/// `<CARGO_MANIFEST_DIR>/<rel>`, leaked to `&'static str` to match the call-site type these tests need
/// (`concat!(env!(...), ..)`'s old return type); test binaries exit after the process runs, so the leak is
/// bounded and cheap.
fn manifest_path(rel: &str) -> &'static str {
    Box::leak(format!("{}/{rel}", manifest_dir().display()).into_boxed_str())
}

/// `CARGO_BIN_EXE_pootc`, read at runtime for the same reason as [`manifest_dir`].
fn pootc_exe() -> &'static str {
    static EXE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXE.get_or_init(|| {
        std::env::var("CARGO_BIN_EXE_pootc")
            .expect("CARGO_BIN_EXE_pootc must be set by cargo for test binaries")
    })
    .as_str()
}

/// Build `poot-kernel-attr` from this checkout and return the proc-macro library cargo reports for it.
///
/// `device_pass` selects which of the crate's two macro expansions this build links (card 537, ADR-0104
/// decision 5, SC-002): `true` builds `--features device-pass` (the macro renames into the reserved
/// namespace, for `pootc`'s own kernel-extraction pass below); `false` builds the crate's default feature
/// set (the macro emits the host dispatch wrapper). The two builds are separate proc-macro artifacts - the
/// macro itself reads no environment variable to pick between them any more.
///
/// The target directory is keyed by the checkout AND `device_pass`, so no other worktree's build lands in
/// it and the two feature variants never share a fingerprinted build directory: slot target directories are
/// shared, and picking `libpoot_kernel_attr-*` out of a shared `deps/` by mtime can link another worktree's
/// (or the wrong feature set's) stale proc-macro. The path is the `filenames` entry of the package's
/// `compiler-artifact` message; a build that fails, or reports no or several proc-macro libraries, fails the
/// test with cargo's output.
fn build_proc_macro(device_pass: bool) -> PathBuf {
    let manifest_dir = manifest_dir();
    let mut checkout = DefaultHasher::new();
    manifest_dir.hash(&mut checkout);
    device_pass.hash(&mut checkout);
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "kernel-attr-{}-{:016x}",
        if device_pass { "device" } else { "host" },
        checkout.finish()
    ));
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args([
        "build",
        "--locked",
        "--message-format=json-render-diagnostics",
    ])
    .arg("--manifest-path")
    .arg(manifest_dir.join("../poot-kernel-attr/Cargo.toml"))
    .arg("--target-dir")
    .arg(&target_dir)
    // A lint wrapper has no business in a support build.
    .env_remove("RUSTC_WORKSPACE_WRAPPER");
    if device_pass {
        cmd.args(["--features", "device-pass"]);
    }
    let output = cmd.output().expect("run cargo to build poot-kernel-attr");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "building poot-kernel-attr failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );

    let libraries: Vec<PathBuf> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| message["reason"] == "compiler-artifact")
        .filter(|message| message["target"]["name"] == "poot_kernel_attr")
        .filter(|message| message["target"]["kind"] == serde_json::json!(["proc-macro"]))
        .flat_map(|message| message["filenames"].as_array().cloned().unwrap_or_default())
        .filter_map(|name| name.as_str().map(PathBuf::from))
        .filter(|path| path.extension().is_some_and(|ext| ext == DLL_EXTENSION))
        .collect();
    let [library] = libraries.as_slice() else {
        panic!(
            "cargo reported {} proc-macro libraries for poot-kernel-attr, wanted exactly one: {libraries:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            libraries.len()
        );
    };
    library.clone()
}

/// Build `poot-kernel-intrinsics` from this checkout and return its rlib path: `add_macro.rs` calls
/// `poot_kernel_intrinsics::thread_index` (card 531c), which needs `--extern` alongside the proc-macro.
fn build_kernel_intrinsics_rlib() -> PathBuf {
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
}

#[test]
fn kernel_attribute_is_discovered_by_pootc() {
    let exe = PathBuf::from(pootc_exe());
    let pm = build_proc_macro(true); // device pass: the macro emits the renamed device fn
    let intrinsics = build_kernel_intrinsics_rlib();

    let kernel = manifest_path("tests/kernels/add_macro.rs");
    let out_dir = std::env::temp_dir().join("pootc-macro-add");
    std::fs::create_dir_all(&out_dir).unwrap();
    let output = Command::new(&exe)
        .args(["--edition", "2021", "--crate-type", "lib", kernel, "-o"])
        .arg(out_dir.join("add.out"))
        .arg("--extern")
        .arg(format!("poot_kernel_attr={}", pm.display()))
        .arg("--extern")
        .arg(format!("poot_kernel_intrinsics={}", intrinsics.display()))
        .env("POOT_KERNEL_OUT", &out_dir)
        .output()
        .expect("run pootc");
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        stderr.contains("found 1 kernel(s)")
            && stderr.contains("__poot_kernel_add: imported -> Body { 3 params"),
        "the #[kernel] macro should rename add -> __poot_kernel_add for pootc to import; stderr:\n{stderr}"
    );

    // pootc emitted poot_cache.rs + add.spv; read the SPIR-V directly and dispatch it.
    let cache = std::fs::read_to_string(out_dir.join("poot_cache.rs")).expect("cache emitted");
    assert!(
        cache.contains("(\"add\", include_bytes!"),
        "cache should list the add kernel:\n{cache}"
    );
    let spv = std::fs::read(out_dir.join("add.spv")).expect("add.spv emitted");
    // `a[i]`/`b[i]`'s MIR-inserted bounds Asserts check against `a`/`b`'s own lengths, not `c`'s (the
    // explicit `if i < c.len()` guard's slice), so they are not provably redundant and stay live traps
    // (card 531c): the compiled module needs the reserved SpirvVulkan error-word binding.
    let kir_json =
        std::fs::read_to_string(out_dir.join("add.kir.json")).expect("add.kir.json emitted");
    let body: Body = serde_json::from_str(&kir_json).expect("add.kir.json parses");
    let kernel = poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, spv);

    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(a.len()),
    ];
    ctx.dispatch(
        "test",
        &kernel,
        [64, 1, 1],
        [a.len() as u32, 1, 1],
        &mut bufs,
    )
    .expect("dispatch #[kernel]-authored add");
    assert_eq!(bufs[2].as_f32(), &[11.0, 22.0, 33.0, 44.0, 55.0]);
    eprintln!("#[kernel] fn add authored in Rust -> pootc -> SPIR-V -> GPU = a+b");
}

/// Build `poot-runtime`, `poot-codegen` and `poot-kernel-ir` (with its `serde` feature) into an isolated
/// scratch target directory, keyed like `build_proc_macro`'s (see its doc: a shared `deps/` can hand back
/// another worktree's rlib for the same crate name), and return every rlib cargo reports, keyed by crate
/// name. `--tests` also builds their dev-dependencies, since `serde_json` (needed by the generated
/// `poot_kernel_body`) is only a dev-dependency of `poot-runtime`/`poot-kernel-ir`, not a regular one. Card
/// 608 host-pass probe below `--extern`-links its generated wrapper's
/// `poot_runtime`/`poot_codegen`/`poot_kernel_ir`/`serde_json` (transitive) references against these.
fn build_host_pass_rlibs() -> HashMap<String, PathBuf> {
    let manifest_dir = manifest_dir();
    let mut checkout = DefaultHasher::new();
    manifest_dir.hash(&mut checkout);
    let target_dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("kernel-attr-hostpass-{:016x}", checkout.finish()));
    let output = Command::new(env!("CARGO"))
        .args([
            "build",
            "--locked",
            "--tests",
            "--message-format=json-render-diagnostics",
            "-p",
            "poot-runtime",
            "-p",
            "poot-codegen",
            "-p",
            "poot-kernel-ir",
            "--features",
            "poot-kernel-ir/serde",
        ])
        .arg("--target-dir")
        .arg(&target_dir)
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .output()
        .expect("build poot-runtime/poot-codegen/poot-kernel-ir for the host-pass probe");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "building host-pass probe rlibs failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    let mut rlibs = HashMap::new();
    for message in stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|message| message["reason"] == "compiler-artifact")
    {
        let Some(name) = message["target"]["name"].as_str() else {
            continue;
        };
        let Some(files) = message["filenames"].as_array() else {
            continue;
        };
        for file in files {
            if let Some(path) = file.as_str()
                && path.ends_with(".rlib")
            {
                rlibs.insert(name.to_string(), PathBuf::from(path));
            }
        }
    }
    rlibs
}

/// Card 608: `kernel_attribute_is_discovered_by_pootc` only ever runs the
/// **device** pass and then dispatches manually via `poot_codegen::kernel_handle` - it never compiles or
/// calls the generated **host**-pass wrapper. This test does: it runs the device pass on the same
/// `add_macro.rs` (whose `a[i]`/`b[i]` MIR bounds Asserts are live traps, not provably redundant - see the
/// comment above), then compiles a SEPARATE source that `#[kernel]`-authors the same `add` signature and
/// `include!`s the cache pootc just staged, this time built against `poot-kernel-attr`'s default feature
/// set (card 537: no `device-pass` feature) so it emits the dispatch wrapper, and runs the compiled binary
/// against a real GPU. If the wrapper's `CompiledKernel` has the wrong `has_trap` (the bug this finding
/// reported: hardcoded `false` from Rust syntax alone) or the wrong schema, the compiled SPIR-V module's
/// actual bind-group layout does not match what the wrapper binds, and the dispatch fails - it does not
/// silently pass.
#[test]
fn host_pass_wrapper_dispatches_using_the_bodys_real_trap_and_schema() {
    let exe = PathBuf::from(pootc_exe());
    let pm_device = build_proc_macro(true);
    let pm_host = build_proc_macro(false);

    let kernel_src = manifest_path("tests/kernels/add_macro.rs");
    let out_dir = std::env::temp_dir().join("pootc-macro-add-hostpass");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    // Device pass: stages add.spv, add.kir.json and poot_cache.rs, exactly like
    // `kernel_attribute_is_discovered_by_pootc`. This test needs `poot_kernel_intrinsics` extern-linked only
    // for that pass (the device-renamed body still calls `thread_index`); the host-pass source below never
    // does.
    let intrinsics = build_kernel_intrinsics_rlib();
    let device_pass = Command::new(&exe)
        .args(["--edition", "2021", "--crate-type", "lib", kernel_src, "-o"])
        .arg(out_dir.join("add.out"))
        .arg("--extern")
        .arg(format!("poot_kernel_attr={}", pm_device.display()))
        .arg("--extern")
        .arg(format!("poot_kernel_intrinsics={}", intrinsics.display()))
        .env("POOT_KERNEL_OUT", &out_dir)
        .output()
        .expect("run pootc (device pass)");
    assert!(
        device_pass.status.success(),
        "device pass failed: {}",
        String::from_utf8_lossy(&device_pass.stderr)
    );

    let kir_json =
        std::fs::read_to_string(out_dir.join("add.kir.json")).expect("add.kir.json emitted");
    let body: Body = serde_json::from_str(&kir_json).expect("add.kir.json parses");
    assert!(
        body.has_trap(),
        "this test's premise is a trap-needing kernel (card 608); if `add`'s \
         bounds check became provably redundant, swap in a kernel body that still needs one"
    );

    // Host pass: `include!`s the cache the device pass just staged, so `poot_kernel_spv`/`poot_kernel_body`
    // resolve; the fn body is never re-emitted in host-pass mode (see `poot-kernel-attr`'s module doc), so
    // it need not be host-compatible.
    let host_src = out_dir.join("add_host.rs");
    std::fs::write(
        &host_src,
        r#"
use poot_kernel_attr::kernel;

include!("poot_cache.rs");

#[kernel]
pub fn add(a: &[f32], b: &[f32], c: &mut [f32]) {}

fn main() {
    let ctx = match poot_runtime::Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    // Happy path: in-bounds a+b, proving the wrapper's binding order/schema are right.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let mut c = [0.0f32; 5];
    add(&ctx, &a, &b, &mut c).expect("dispatch through the #[kernel] host-pass wrapper");
    assert_eq!(c, [11.0, 22.0, 33.0, 44.0, 55.0], "host-pass #[kernel] add must compute a+b on the GPU");

    // Trap path: a/b shorter than c, so for i in [a.len(), c.len()) the explicit `i < c.len()` guard
    // holds but `i >= a.len()`, firing the MIR-inserted bounds Assert on `a[i]`/`b[i]` - safely, by
    // design (card 531c): the kernel stages a fault instead of touching memory out of bounds. The host
    // only surfaces it when `CompiledKernel::has_trap()` is true, so this is the reliable, safe signal
    // for card 608 exact bug (a hardcoded `has_trap: false`): a wrapper whose
    // handle disagreed with the compiled Body's real trap flag would return `Ok(())` here instead (the
    // fault silently unreported), not a crash.
    let short_a = [1.0f32, 2.0, 3.0];
    let short_b = [10.0f32, 20.0, 30.0];
    let mut wide_c = [0.0f32; 6];
    match add(&ctx, &short_a, &short_b, &mut wide_c) {
        Err(poot_runtime::RuntimeError::KernelAssertFailed { .. }) => {}
        other => panic!(
            "expected the host-pass wrapper to report the compiled Body's real bounds trap, got {other:?}"
        ),
    }
    println!("host-pass #[kernel] add OK");
}
"#,
    )
    .expect("write the host-pass probe source");

    let rlibs = build_host_pass_rlibs();
    let deps_dir = rlibs
        .get("poot_runtime")
        .expect("poot_runtime rlib reported by the host-pass probe build")
        .parent()
        .expect("an rlib path has a parent deps directory")
        .to_path_buf();
    let bin_out = out_dir.join(format!("add_host{}", std::env::consts::EXE_SUFFIX));
    let mut host_pass = Command::new(&exe);
    host_pass
        .args(["--edition", "2021", "--crate-type", "bin", "-o"])
        .arg(&bin_out)
        .arg(&host_src)
        .arg("-L")
        .arg(format!("dependency={}", deps_dir.display()))
        .arg("--extern")
        .arg(format!("poot_kernel_attr={}", pm_host.display()));
    for name in [
        "poot_runtime",
        "poot_codegen",
        "poot_kernel_ir",
        "serde_json",
    ] {
        let rlib = rlibs
            .get(name)
            .unwrap_or_else(|| panic!("{name} rlib reported by the host-pass probe build"));
        host_pass
            .arg("--extern")
            .arg(format!("{name}={}", rlib.display()));
    }
    let host_pass = host_pass.output().expect("run pootc (host pass)");
    assert!(
        host_pass.status.success(),
        "host pass failed to compile the generated wrapper: {}",
        String::from_utf8_lossy(&host_pass.stderr)
    );

    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let run = Command::new(&bin_out)
        .output()
        .expect("run the compiled host-pass probe binary");
    let stdout = String::from_utf8_lossy(&run.stdout);
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        run.status.success(),
        "host-pass probe binary failed: stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    if stdout.contains("no GPU") || stderr.contains("no GPU") {
        eprintln!("no GPU; host-pass dispatch skipped (structural compile/wire-up still verified)");
        return;
    }
    assert!(
        stdout.contains("host-pass #[kernel] add OK"),
        "expected the probe binary to report success; stdout:\n{stdout}"
    );
}

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

//! End-to-end: a kernel authored in ordinary, type/borrow-checked Rust is compiled by `pootc` to a
//! dispatchable SPIR-V module (Stable MIR -> Body -> llc -> .spv, all inside pootc) and dispatched on the
//! GPU (real Rust, real rustc MIR, no hand-built IR). Skips (passes) if no Vulkan adapter.
//! Requires `llc` (nix develop).
//!
//! `manual_clamp` is allowed crate-wide: the gumbel-sampling CPU oracles intentionally mirror the
//! `u.max(a).min(b)` formula in `crates/pootc/tests/kernels/sample_*_gumbel_argmax_batched.rs` line for line
//! (those sources compile through pootc's Stable MIR path and are not clippy-linted), so oracle and kernel
//! can be diffed. `u` is always `(x as f32 + 0.5) / 4294967296.0` for a `u32` `x`, so it is finite and
//! `.clamp()` and `.max().min()` are equivalent here.
#![allow(clippy::manual_clamp)]

use std::process::Command;

use poot_kernel_ir::Body;
use poot_runtime::{CompiledKernel, Context, KernelBuffer};

/// Serialize GPU access within this test binary: the Intel Arc Vulkan driver segfaults when several tests
/// create/use wgpu devices on it concurrently. Poison-tolerant. Hold for the dispatch.
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// `CARGO_MANIFEST_DIR`, read at runtime (not baked via `env!`): the compile-time macro's value is
/// embedded into this test binary's compiled object code at build time, and a shared compile cache (kache)
/// that reuses that object across worktrees by source-content hash would then serve whichever worktree's
/// path happened to compile it first (card 530's build.rs fix; card 543 review). `std::env::var` reads the
/// environment cargo sets fresh for every test-binary invocation, so it is correct regardless of which
/// worktree compiled the binary.
fn manifest_dir() -> &'static std::path::Path {
    static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        std::path::PathBuf::from(
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

/// Build `poot-kernel-intrinsics` from this checkout and return its rlib path, memoized for the process.
/// Every kernel source now depends on it instead of redeclaring its own intrinsic stubs (card 531c), so
/// every `pootc` invocation that compiles one needs `--extern poot_kernel_intrinsics=<rlib>`. The target
/// directory is keyed by the checkout so no other worktree's build lands in a shared slot target dir
/// (mirrors `macro_kernel.rs`'s `build_proc_macro`).
fn kernel_intrinsics_rlib() -> &'static std::path::Path {
    static RLIB: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    RLIB.get_or_init(|| {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let manifest_dir = manifest_dir();
        let mut checkout = DefaultHasher::new();
        manifest_dir.hash(&mut checkout);
        let target_dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
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
        let libraries: Vec<std::path::PathBuf> = stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|message| message["reason"] == "compiler-artifact")
            .filter(|message| message["target"]["name"] == "poot_kernel_intrinsics")
            .filter(|message| message["target"]["kind"] == serde_json::json!(["lib"]))
            .flat_map(|message| message["filenames"].as_array().cloned().unwrap_or_default())
            .filter_map(|name| name.as_str().map(std::path::PathBuf::from))
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
/// source (card 531c).
fn kernel_intrinsics_extern() -> String {
    format!(
        "poot_kernel_intrinsics={}",
        kernel_intrinsics_rlib().display()
    )
}

/// Compile `kernel_file` with pootc (which emits `<src_name>.spv`) and return the dispatchable
/// [`CompiledKernel`] (card 608) plus the imported Body's `param_count` and `has_trap()` (from the
/// emitted `.kir.json`; the third element is `needs_error_word` for tests that assert on it directly,
/// card 531c). Each kernel gets its own out dir.
fn pootc_compile(kernel_file: &str, src_name: &str) -> (CompiledKernel, u32, bool) {
    let exe = pootc_exe();
    let out_dir = std::env::temp_dir().join(format!("pootc-{src_name}"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let status = Command::new(exe)
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            kernel_file,
            "-o",
        ])
        .arg(out_dir.join(format!("{src_name}.out")))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env("POOT_KERNEL_OUT", &out_dir)
        .status()
        .expect("run pootc");
    assert!(status.success(), "pootc failed");
    let json = std::fs::read_to_string(out_dir.join(format!("{src_name}.kir.json")))
        .unwrap_or_else(|_| panic!("pootc should have emitted {src_name}.kir.json"));
    let body: Body = serde_json::from_str(&json).expect("deserialize imported Body");
    let bytes = std::fs::read(out_dir.join(format!("{src_name}.spv")))
        .unwrap_or_else(|_| panic!("pootc should have emitted {src_name}.spv"));
    let param_count = body.param_count;
    let has_trap = body.has_trap();
    let compiled_kernel =
        poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, bytes);
    (compiled_kernel, param_count, has_trap)
}

/// Like [`pootc_compile`] but for the NVPTX backend: compiles `kernel_file` with pootc and returns the
/// imported [`Body`] (from the emitted `.kir.json`) plus the emitted `.ptx` text (pootc stages SPIR-V,
/// PTX and the IR JSON from one invocation). Each kernel gets its own out dir.
fn pootc_compile_nvptx(kernel_file: &str, src_name: &str) -> (Body, String) {
    let exe = pootc_exe();
    let out_dir = std::env::temp_dir().join(format!("pootc-nvptx-{src_name}"));
    std::fs::create_dir_all(&out_dir).unwrap();
    let status = Command::new(exe)
        .args([
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            kernel_file,
            "-o",
        ])
        .arg(out_dir.join(format!("{src_name}.out")))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env("POOT_KERNEL_OUT", &out_dir)
        .status()
        .expect("run pootc");
    assert!(status.success(), "pootc failed");
    let json = std::fs::read_to_string(out_dir.join(format!("{src_name}.kir.json")))
        .unwrap_or_else(|_| panic!("pootc should have emitted {src_name}.kir.json"));
    let body: Body = serde_json::from_str(&json).expect("deserialize imported Body");
    let ptx = std::fs::read_to_string(out_dir.join(format!("{src_name}.ptx")))
        .unwrap_or_else(|_| panic!("pootc should have emitted {src_name}.ptx"));
    (body, ptx)
}

/// Replace the single `Rvalue::BinaryOpNoContract(BinOp::Add, ..)` statement in `body` with a plain
/// `Rvalue::BinaryOp(BinOp::Add, ..)`, as if `no_contract_add`'s importer wiring (card 675) never existed.
/// Panics unless `body` has exactly one such statement (every caller here is the Gumbel-family body, which
/// has exactly one: the perturbed-argmax sum).
fn mutate_away_single_no_contract_add(body: &Body) -> Body {
    let mut mutated = body.clone();
    let mut replaced = 0u32;
    for bb in &mut mutated.blocks {
        for s in &mut bb.statements {
            if let poot_kernel_ir::Statement::Assign(
                _,
                rv @ poot_kernel_ir::Rvalue::BinaryOpNoContract(poot_kernel_ir::BinOp::Add, _, _),
            ) = s
            {
                let poot_kernel_ir::Rvalue::BinaryOpNoContract(op, a, b) = rv.clone() else {
                    unreachable!()
                };
                *rv = poot_kernel_ir::Rvalue::BinaryOp(op, a, b);
                replaced += 1;
            }
        }
    }
    assert_eq!(
        replaced, 1,
        "expected exactly one no-contract Add in the Gumbel body (the perturbed-argmax sum)"
    );
    mutated
}

/// Every kernel source lives under `pootc/kernels/<family>/`, one family per `imported/*.rs` file in
/// `poot-graph-plan`. This does not need to know family names (so a family rename or split, or a later
/// card retiring a whole family, needs no edit here): it finds the one source matching `source` under
/// any family directory.
fn kernel_source_path(source: &str) -> std::path::PathBuf {
    let kernels_dir = std::path::Path::new(manifest_path("kernels"));
    let mut found = Vec::new();
    for family in std::fs::read_dir(kernels_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", kernels_dir.display()))
    {
        let family_dir = family.expect("read_dir entry").path();
        if !family_dir.is_dir() {
            continue;
        }
        let candidate = family_dir.join(format!("{source}.rs"));
        if candidate.is_file() {
            found.push(candidate);
        }
    }
    match found.as_slice() {
        [one] => one.clone(),
        [] => panic!("no pootc/kernels/<family>/{source}.rs found for a MANIFEST entry"),
        many => {
            panic!("{source}.rs exists under more than one kernels/<family>/ directory: {many:?}")
        }
    }
}

/// Regenerate `entry`'s asset from its kernel source (card 559) and return the emitted JSON text.
fn regenerate_asset(entry: &poot_graph_plan::AssetEntry) -> String {
    let exe = pootc_exe();
    let src_path = kernel_source_path(entry.source);
    let out_dir = std::env::temp_dir().join(format!("pootc-assetcheck-{}", entry.source));
    std::fs::create_dir_all(&out_dir).unwrap();
    let status = Command::new(exe)
        .args(["--edition", "2021", "--crate-type", "lib"])
        .arg(&src_path)
        .arg("-o")
        .arg(out_dir.join(format!("{}.out", entry.source)))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env("POOT_KERNEL_OUT", &out_dir)
        .status()
        .expect("run pootc");
    assert!(
        status.success(),
        "pootc failed importing {}",
        src_path.display()
    );
    std::fs::read_to_string(out_dir.join(format!("{}.kir.json", entry.source)))
        .unwrap_or_else(|_| panic!("pootc emitted no {}.kir.json", entry.source))
}

/// Every directory a committed kernel asset can live in today (card 559). Checked even when
/// no `MANIFEST` entry currently names it, so a destination whose every entry was deleted from the
/// manifest - while its directory and files stay on disk - is still caught, not silently skipped
/// because `expected_by_dir` (built from `MANIFEST` alone) never saw it. Grow this list in the same
/// commit that adds a new `AssetDest::Crate(..)` destination to a family file; a `MANIFEST` entry
/// naming a directory this list doesn't know about fails loudly instead (below), so the two can't drift
/// apart unnoticed.
const KNOWN_ASSET_DIRS: &[&str] = &[
    "poot-graph-plan/assets",
    "poot-rocm-gpu/assets",
    "poot-gpu/src/tests/assets",
];

#[test]
fn committed_assets_match_their_kernel_sources() {
    // The imported-kernel assets the engine (or another crate) embeds via `include_str!` are pootc's
    // output for a kernel source under `pootc/kernels/<family>/`. They are committed (so the main build
    // needs no rustc), so a kernel edit that forgets to regenerate the asset goes stale silently.
    //
    // Card 559: `poot_graph_plan::MANIFEST` is the one list - no more hand-kept duplicate here, in the
    // justfile, and in the loaders. This regenerates every entry's asset from its source and checks it
    // byte-for-byte against the committed one (pootc emission is deterministic; SC-002), then checks
    // each destination `assets/` directory holds exactly its manifest entries (SC-001): a committed
    // asset with no manifest entry, or an entry naming a file the directory doesn't have, fails by name.
    let workspace = manifest_path("..");
    let entries: Vec<&poot_graph_plan::AssetEntry> = poot_graph_plan::MANIFEST
        .iter()
        .flat_map(|family| family.iter())
        .collect();

    let mut expected_by_dir: std::collections::BTreeMap<&str, std::collections::BTreeSet<String>> =
        std::collections::BTreeMap::new();
    for entry in &entries {
        let committed_path = format!("{workspace}/{}/{}.kir.json", entry.dest.dir(), entry.name);
        let committed = std::fs::read_to_string(&committed_path)
            .unwrap_or_else(|_| panic!("committed asset missing: {committed_path}"));
        let regen = regenerate_asset(entry);
        // Both must be valid Bodies; pootc emission is deterministic, so the committed bytes must match
        // the regenerated bytes (trailing-newline tolerant).
        let _: Body = serde_json::from_str(&regen).expect("regen Body parses");
        let _: Body = serde_json::from_str(&committed).expect("committed Body parses");
        assert_eq!(
            regen.trim_end(),
            committed.trim_end(),
            "asset {}.kir.json is STALE vs its source - run `just regen-kernel-assets`",
            entry.name
        );
        expected_by_dir
            .entry(entry.dest.dir())
            .or_default()
            .insert(format!("{}.kir.json", entry.name));
    }

    for dir in expected_by_dir.keys() {
        assert!(
            KNOWN_ASSET_DIRS.contains(dir),
            "MANIFEST entry destination {dir} is not in KNOWN_ASSET_DIRS - add it there too"
        );
    }

    let empty = std::collections::BTreeSet::new();
    for &dir in KNOWN_ASSET_DIRS {
        let expected = expected_by_dir.get(dir).unwrap_or(&empty);
        let dir_path = format!("{workspace}/{dir}");
        let actual: std::collections::BTreeSet<String> = match std::fs::read_dir(&dir_path) {
            Ok(read_dir) => read_dir
                .map(|e| {
                    e.expect("read_dir entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .filter(|name| name.ends_with(".kir.json"))
                .collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => std::collections::BTreeSet::new(),
            Err(e) => panic!("read {dir_path}: {e}"),
        };
        assert_eq!(
            actual, *expected,
            "{dir} must hold exactly its MANIFEST entries"
        );
    }

    eprintln!(
        "all {} committed kernel assets match their MANIFEST entries and their kernel sources",
        entries.len()
    );
}

/// `#[ignore]`d: `just regen-kernel-assets` runs this to refresh every committed asset from
/// `poot_graph_plan::MANIFEST` (the write-instead-of-compare mirror of the staleness check above).
#[test]
#[ignore = "writes every committed kernel asset; run via `just regen-kernel-assets`"]
fn regen_kernel_assets() {
    let workspace = manifest_path("..");
    for family in poot_graph_plan::MANIFEST {
        for entry in *family {
            let regen = regenerate_asset(entry);
            let committed_path =
                format!("{workspace}/{}/{}.kir.json", entry.dest.dir(), entry.name);
            std::fs::write(&committed_path, regen)
                .unwrap_or_else(|e| panic!("write {committed_path}: {e}"));
            eprintln!("regenerated {committed_path}");
        }
    }
}

/// Card 461 (filed out of Card 449 D1): a branch condition carrying a two-operand `&&` used to spin the
/// structurizer forever, so `pootc` staged `.kir.json` and never emitted `.spv`. MIR writes
/// `if a && b { .. } else { .. }` with ONE else block shared by both conditions, which puts the inner
/// condition's false arm outside its own dominance region: no merge-redirect can make that selection's
/// join private, so the pass appended one goto-forwarding block per round and never stopped. The fix
/// clones the shared tail for the escaping selection first (`poot-codegen`'s `structurize`, unit-tested
/// by `shared_else_arm_of_a_short_circuit_converges`) and bounds the fixpoint, so a body it still cannot
/// structure returns a typed error instead of hanging the build. The failing body is
/// `tests/kernels/llc_spin_depth4_and.rs`; `packed_e4m3_row_gather.rs` still spells its NaN test as
/// nested `if`s (its committed `.kir.json` would change).
///
/// SC-001/SC-002: a NORMAL test, not `#[ignore]`d. `pootc` must publish `.spv` inside `budget`; on the
/// pre-fix toolchain it never finishes and this test fails with "never finished" when the budget expires
/// (observed red: killed at 60s, `.kir.json` staged, no `.spv`).
#[test]
fn llc_spin_depth4_and_reproducer() {
    let kernel = manifest_path("tests/kernels/llc_spin_depth4_and.rs");
    let exe = pootc_exe();
    let out_dir = std::env::temp_dir().join(format!("pootc-llc-spin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).expect("stage the reproducer out dir");
    let log_path = out_dir.join("pootc.log");
    let log = std::fs::File::create(&log_path).expect("pootc log file");
    let mut child = std::process::Command::new(exe)
        .args(["--edition", "2021", "--crate-type", "lib", kernel, "-o"])
        .arg(out_dir.join("llc_spin_depth4_and.out"))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env("POOT_KERNEL_OUT", &out_dir)
        .stdout(std::process::Stdio::from(
            log.try_clone().expect("clone the log handle"),
        ))
        .stderr(std::process::Stdio::from(log))
        .spawn()
        .expect("spawn pootc");

    // The healthy row-gather siblings lower in single-digit seconds, so a minute is a wide margin rather
    // than a slow machine: pre-fix this budget expires while the single `pootc` process is still spinning.
    let budget = std::time::Duration::from_secs(60);
    let start = std::time::Instant::now();
    let mut finished = None;
    while start.elapsed() < budget {
        match child.try_wait().expect("poll pootc") {
            Some(status) => {
                finished = Some(status);
                break;
            }
            None => std::thread::sleep(std::time::Duration::from_millis(250)),
        }
    }
    let stdout = std::fs::read_to_string(&log_path).unwrap_or_default();
    let Some(status) = finished else {
        let _ = child.kill();
        let _ = child.wait();
        panic!(
            "pootc never finished after {budget:?} - the depth-4 `&&` structurizer spin is back:\n{stdout}"
        );
    };
    assert!(
        status.success(),
        "pootc failed on the depth-4 `&&` body ({status}):\n{stdout}"
    );
    assert!(
        stdout.contains("imported -> Body"),
        "the body must import before it is lowered:\n{stdout}"
    );
    let spv_path = out_dir.join("llc_spin_depth4_and.spv");
    let compiled_kernel = std::fs::read(&spv_path).unwrap_or_else(|error| {
        panic!("pootc must publish llc_spin_depth4_and.spv: {error}\n{stdout}")
    });
    let kir_json = std::fs::read_to_string(out_dir.join("llc_spin_depth4_and.kir.json"))
        .expect("pootc must publish llc_spin_depth4_and.kir.json");
    let body: Body = serde_json::from_str(&kir_json).expect("llc_spin_depth4_and.kir.json parses");
    let compiled_kernel =
        poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, compiled_kernel);

    // llc rebuilds the selection merges from exactly the CFG this pass shapes, so the module has to be
    // strictly valid, not merely tolerated by RADV.
    if Command::new("spirv-val").arg("--version").output().is_err() {
        eprintln!("spirv-val not on PATH; skipping validation");
    } else {
        let v = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&spv_path)
            .output()
            .expect("run spirv-val");
        assert!(
            v.status.success(),
            "the depth-4 `&&` module must validate:\n{}",
            String::from_utf8_lossy(&v.stderr)
        );
    }

    // The structurizer clones the shared else arm for the inner condition; the clone has to compute what
    // the nested-if spelling computes. The bytes below exercise all four outcomes of
    // `exponent == 15 && mantissa == 7`: then/then, then/else (short-circuiting into the shared else),
    // else/short-circuited, and a non-15 exponent.
    let table: Vec<u32> = (0..16u32)
        .map(|row| {
            let b0 = if row % 2 == 0 { 0x7f } else { 0x00 };
            let b1 = if row % 3 == 0 { 0xff } else { 0x11 };
            u32::from_le_bytes([b0, b1, 0x7e, 0x80])
        })
        .collect();
    let index: Vec<i32> = (0..16).collect();
    let mut expected = vec![1.0f32; 64];
    for (i, want) in expected.iter_mut().enumerate() {
        let byte = (table[(index[i / 4]) as usize] >> ((i % 4) * 8)) & 0xff;
        let exponent = (byte >> 3) & 0x0f;
        let mantissa = byte & 0x07;
        if exponent == 15 && mantissa == 7 {
            *want = 0.0;
        }
    }

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            let _ = std::fs::remove_dir_all(&out_dir);
            return;
        }
    };
    let mut bufs = [
        KernelBuffer::read_only_u32(&table),
        KernelBuffer::read_only_i32(&index),
        KernelBuffer::write_f32(expected.len()),
    ];
    ctx.dispatch(
        "llc_spin_depth4_and",
        &compiled_kernel,
        [64, 1, 1],
        [64, 1, 1],
        &mut bufs,
    )
    .expect("dispatch the depth-4 `&&` kernel");
    assert_eq!(
        bufs[2].as_f32(),
        expected.as_slice(),
        "the depth-4 `&&` kernel must match its CPU reference on GPU"
    );
    eprintln!("depth-4 `&&` lowers, validates and runs on GPU = matches CPU");
    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn add_kernel_imported_from_mir_runs_on_gpu() {
    let kernel = manifest_path("tests/kernels/add.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "add");
    assert_eq!(params, 3, "add has 3 slice params");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    // FORK NOTE (card 531c): `add`'s bounds-check Asserts against `a`/`b` are not provably redundant
    // with the `if i < c.len()` guard (different slices), so the compiled module now needs the reserved
    // SpirvVulkan error-word binding (`needs_error_word` is true). `Context::launch` cannot support that
    // (see its doc: no `Body` to check), so this now calls `dispatch` directly with buffers built the
    // same way `launch` builds them, rather than testing the `launch` entry point itself. Flagging for
    // review: this narrows what invariant 8 verifies (no longer exercises `launch`/the `#[kernel]` host
    // wrapper's call shape) - not a decision a mechanical fork should make unilaterally.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [10.0f32, 20.0, 30.0, 40.0, 50.0];
    let mut c = [0.0f32; 5];
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(c.len()),
    ];
    ctx.dispatch(
        "launch",
        &compiled_kernel,
        [64, 1, 1],
        [c.len() as u32, 1, 1],
        &mut bufs,
    )
    .expect("dispatch imported add kernel");
    c.copy_from_slice(bufs[2].as_f32());
    assert_eq!(
        c,
        [11.0, 22.0, 33.0, 44.0, 55.0],
        "imported add kernel must compute c = a + b"
    );
    eprintln!(
        "invariant 8 proven: real Rust add kernel -> pootc -> SPIR-V -> GPU = a+b (via launch)"
    );
}

/// SC-003 (card 531c, R468-007): on SpirvVulkan a failed `Assert` reports through the reserved
/// per-dispatch error word, never a silent continue. `assert_trap.rs`'s `data[idx[i]]` bounds check
/// traps whichever lane's `idx[i]` is out of range for `data`; with every index in range the kernel
/// completes normally. Mutation: silently drop the assert again (e.g. revert the SpirvVulkan arm of
/// `Terminator::Trap` in `crates/poot-codegen/src/emit/core.rs` to a `Goto`/no-op instead of the atomic
/// error-word write); the faulting-index case then wrongly returns `Ok` and this test goes red.
#[test]
fn assert_trap_reports_the_spirv_error_word_on_wgpu() {
    let kernel = manifest_path("kernels/probe/assert_trap.rs");
    let (compiled_kernel, params, needs_error_word) = pootc_compile(kernel, "assert_trap");
    assert_eq!(params, 3, "assert_trap has 3 slice params");
    assert!(
        needs_error_word,
        "assert_trap's data[idx[i]] bounds check is a real Assert; the compiled body must have a trap"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    // Green: every idx in range, the kernel runs to completion and computes data[idx[i]].
    let idx_ok = [3u32, 0, 2, 1];
    let data = [10.0f32, 20.0, 30.0, 40.0];
    let mut ok_bufs = [
        KernelBuffer::read_only_u32(&idx_ok),
        KernelBuffer::read_only_f32(&data),
        KernelBuffer::write_f32(idx_ok.len()),
    ];
    ctx.dispatch(
        "assert_trap_ok",
        &compiled_kernel,
        [64, 1, 1],
        [idx_ok.len() as u32, 1, 1],
        &mut ok_bufs,
    )
    .expect("every idx in range: no assert fires, dispatch must succeed");
    assert_eq!(
        ok_bufs[2].as_f32(),
        [40.0, 10.0, 30.0, 20.0],
        "assert_trap must compute out[i] = data[idx[i]] when every index is in range"
    );

    // Red (data, not code): lane 2's idx is one past data.len(), so data[idx[2]]'s bounds check fails.
    let idx_bad = [3u32, 0, 4, 1];
    let mut bad_bufs = [
        KernelBuffer::read_only_u32(&idx_bad),
        KernelBuffer::read_only_f32(&data),
        KernelBuffer::write_f32(idx_bad.len()),
    ];
    let err = ctx
        .dispatch(
            "assert_trap_fault",
            &compiled_kernel,
            [64, 1, 1],
            [idx_bad.len() as u32, 1, 1],
            &mut bad_bufs,
        )
        .expect_err(
            "idx[2] == 4 is out of range for a 4-element data buffer; the assert must trap",
        );
    let poot_runtime::RuntimeError::KernelAssertFailed { kernel, code } = err else {
        panic!("expected KernelAssertFailed, got {err:?}");
    };
    assert_eq!(kernel, "assert_trap_fault");
    assert_ne!(
        code, 0,
        "a fired trap's code is never the 0 sentinel (no-fault)"
    );
    eprintln!(
        "SC-003 proven: assert_trap's data[idx[i]] bounds check traps on SpirvVulkan via the error word (code {code})"
    );
}

/// SC-002 (card 531c, R468-007): on PTX (NVPTX), a failed `Assert` must trap for real, never silently
/// drop. Same `assert_trap.rs` fixture as SC-003 above; pootc is invoked directly (not via
/// `pootc_compile`) because PTX dispatch needs the entry's mangled name (`body.name`) alongside the
/// `.ptx` text, not just SPIR-V bytes. GREEN: every `idx[i]` in range, the kernel runs to completion and
/// computes `out[i] = data[idx[i]]`. RED (mutation: `idx[2] = data.len()`, one past the end): lane 2's
/// bounds check fails and the NVPTX `Trap` terminator (`llvm.trap()` + `unreachable`, the same lowering
/// path as ROCm, `poot-codegen/src/emit/core.rs`) fires a real PTX `trap;`.
///
/// CUDA's documented behavior for an illegal device-side access is a *sticky* context error: the launch
/// itself always succeeds (it is asynchronous), and the fault surfaces as `Err` from the next
/// synchronizing call (`ctx.synchronize()` below); the context is then unusable, but the process does not
/// abort. This differs from ROCm's default async-error handler, which DOES abort the process for the
/// equivalent hardware exception (`imported_assert_trap_traps_on_rocm` in `poot-rocm-gpu`, empirically
/// confirmed on this box's gfx1151). This box has no NVIDIA device, so the PTX half has not been run on
/// real hardware; the next PTX-pod batch (531c/558a/527) should run
/// `cargo test -p pootc --test import_run -- assert_trap_traps_on_ptx --test-threads=1 --nocapture` and
/// confirm. If a real run instead aborts the process (mirroring ROCm), this test needs the same
/// re-exec-child restructuring `imported_assert_trap_traps_on_rocm` uses.
///
/// Skips cleanly (no assertion failure, no panic) when no NVIDIA device is present. `PtxContext::new()`
/// panics instead when `POOT_REQUIRE_PTX=1` (the project's device-skip convention,
/// `poot_runtime_common`), so a lane that was supposed to run this never reports a skip as a pass.
#[test]
fn assert_trap_traps_on_ptx() {
    let exe = pootc_exe();
    let kernel = manifest_path("kernels/probe/assert_trap.rs");
    let out_dir = std::env::temp_dir().join("pootc-assert-trap-ptx");
    std::fs::create_dir_all(&out_dir).unwrap();
    let status = Command::new(exe)
        .args(["--edition", "2021", "--crate-type", "lib", kernel, "-o"])
        .arg(out_dir.join("assert_trap.out"))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env("POOT_KERNEL_OUT", &out_dir)
        .status()
        .expect("run pootc");
    assert!(status.success());
    let ptx = std::fs::read_to_string(out_dir.join("assert_trap.ptx"))
        .expect("pootc should emit assert_trap.ptx");
    let json = std::fs::read_to_string(out_dir.join("assert_trap.kir.json"))
        .expect("pootc should emit assert_trap.kir.json");
    let body: Body = serde_json::from_str(&json).expect("deserialize imported Body");
    assert_eq!(body.param_count, 3, "assert_trap has 3 slice params");
    assert!(
        body.has_trap(),
        "assert_trap's data[idx[i]] bounds check is a real Assert; the compiled body must have a trap"
    );
    let ptx_kernel =
        poot_codegen::kernel_handle(&body, poot_codegen::Target::Nvptx, ptx.into_bytes());

    let ctx = match poot_ptx_runtime::PtxContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP assert_trap_traps_on_ptx: no PTX device ({e})");
            return;
        }
    };

    // Green: every idx in range.
    let idx_ok = [3i32, 0, 2, 1];
    let data = [10.0f32, 20.0, 30.0, 40.0];
    let idx_buf = ctx.upload_i32(&idx_ok).expect("upload idx");
    let data_buf = ctx.upload_f32(&data).expect("upload data");
    let out_buf = ctx.alloc_f32(idx_ok.len()).expect("alloc out");
    ctx.dispatch_dev(
        "assert_trap_ok",
        &ptx_kernel,
        [idx_ok.len() as u32, 1, 1],
        [idx_ok.len() as u32, 1, 1],
        &[&idx_buf, &data_buf],
        &[idx_buf.elem_count(), data_buf.elem_count()],
        &out_buf,
        out_buf.elem_count(),
    )
    .expect("dispatch is async and should always succeed for valid indices");
    ctx.synchronize()
        .expect("every idx in range: no assert fires, synchronize must succeed");
    let got = ctx.download_f32(&out_buf).expect("download out");
    assert_eq!(
        got,
        [40.0, 10.0, 30.0, 20.0],
        "assert_trap must compute out[i] = data[idx[i]] when every index is in range"
    );
    eprintln!(
        "card 531c SC-002 GREEN: in-range indices ran to completion, out == data[idx] on PTX"
    );

    // Red (data, not code). MUTATION (card 531c SC-002): idx[2] = data.len() (one past the end) forces
    // lane 2's bounds-check Assert to fail.
    let mut idx_bad = idx_ok;
    idx_bad[2] = data.len() as i32;
    let idx_bad_buf = ctx.upload_i32(&idx_bad).expect("upload idx (bad)");
    let out_bad_buf = ctx.alloc_f32(idx_bad.len()).expect("alloc out (bad)");
    ctx.dispatch_dev(
        "assert_trap_fault",
        &ptx_kernel,
        [idx_bad.len() as u32, 1, 1],
        [idx_bad.len() as u32, 1, 1],
        &[&idx_bad_buf, &data_buf],
        &[idx_bad_buf.elem_count(), data_buf.elem_count()],
        &out_bad_buf,
        out_bad_buf.elem_count(),
    )
    .expect("dispatch is async; the launch call itself should still succeed");
    let err = ctx.synchronize().expect_err(
        "idx[2] == 4 is out of range for a 4-element data buffer; the assert must trap",
    );
    eprintln!(
        "card 531c SC-002 RED: out-of-bounds index reported a real device fault on PTX: {err}"
    );
}

#[test]
fn add_kernel_imported_from_mir_emits_ptx() {
    // pootc emits the kernel for both backends; check the .ptx has a valid entry. The NVIDIA run is not
    // exercised here (the Body has the same IR shape as fixtures::add_kernel, which passed PTX parity).
    let exe = pootc_exe();
    let kernel = manifest_path("tests/kernels/add.rs");
    let out_dir = std::env::temp_dir().join("pootc-add-ptx");
    std::fs::create_dir_all(&out_dir).unwrap();
    let status = Command::new(exe)
        .args(["--edition", "2021", "--crate-type", "lib", kernel, "-o"])
        .arg(out_dir.join("add.out"))
        .arg("--extern")
        .arg(kernel_intrinsics_extern())
        .env("POOT_KERNEL_OUT", &out_dir)
        .status()
        .expect("run pootc");
    assert!(status.success());
    let ptx = std::fs::read_to_string(out_dir.join("add.ptx")).expect("pootc should emit add.ptx");
    assert!(
        ptx.contains(".visible .entry"),
        "pootc-emitted add.ptx should have a PTX entry; got:\n{}",
        &ptx[..ptx.len().min(400)]
    );
}

#[test]
fn from_bits_kernel_imported_from_mir_runs_on_gpu() {
    // `f32::from_bits(u32)` -> a `Bitcast` (reinterpret the bits, not an `as` numeric cast): read an f32
    // packed into a u32 buffer (the attention-`scale`-in-the-dims-buffer pattern, spec 055). out = inp * scale
    // where scale's bits are passed as a u32. A numeric cast would give inp * (bits as f32), a huge integer;
    // the bitcast gives inp * 0.25.
    let kernel = manifest_path("tests/kernels/from_bits.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "from_bits");
    assert_eq!(params, 3);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let scale = 0.25f32;
    let bits = [scale.to_bits()];
    let mut bufs = [
        KernelBuffer::read_only_f32(&x),
        KernelBuffer::read_only_u32(&bits),
        KernelBuffer::write_f32(4),
    ];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [4, 1, 1], &mut bufs)
        .expect("dispatch imported from_bits kernel");
    assert_eq!(
        bufs[2].as_f32(),
        &[0.25, 0.5, 0.75, 1.0],
        "from_bits must REINTERPRET the u32 as the f32 0.25, not numeric-cast it"
    );
    eprintln!("imported f32::from_bits kernel runs on GPU = bit-reinterpreted scale");
}

#[test]
fn rowsum_loop_kernel_imported_from_mir_runs_on_gpu() {
    // A `while`-loop serial reduce (per-row sum over COLS=4), the shape of kernelgen::reduce_last: a loop
    // (header SwitchInt + back-edge + accumulator + a named const bound) imports and runs correctly.
    let kernel = manifest_path("tests/kernels/rowsum.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "rowsum");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    // 2 rows x 4 cols: row sums are 1+2+3+4=10 and 10+20+30+40=100.
    let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(2)];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
        .expect("dispatch imported rowsum kernel");
    assert_eq!(
        bufs[1].as_f32(),
        &[10.0, 100.0],
        "imported-from-MIR rowsum kernel must compute per-row sums"
    );
    eprintln!("imported while-loop reduce kernel runs on GPU = per-row sums");
}

#[test]
fn rowsum_for_loop_kernel_imported_from_mir_runs_on_gpu() {
    // The idiomatic `for j in 0..COLS` form. Exercises the for-loop rewrite: the
    // Range/into_iter/Iterator::next/Option desugaring is detected and rewritten to a counter loop.
    let kernel = manifest_path("kernels/probe/rowsum_for.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "rowsum_for");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(2)];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
        .expect("dispatch imported rowsum_for kernel");
    assert_eq!(
        bufs[1].as_f32(),
        &[10.0, 100.0],
        "imported `for j in 0..COLS` kernel must compute per-row sums"
    );
    eprintln!("imported `for j in 0..COLS` kernel runs on GPU = per-row sums");
}

#[test]
fn rowsum_inclusive_for_loop_kernel_imported_from_mir_runs_on_gpu() {
    // card 044: the inclusive `for j in 0..=N` form. `a..=b` lowers to `RangeInclusive::new(a,b)` (a Call, not
    // the exclusive aggregate) and the loop runs `counter <= END`. N=3 gives 4 iterations; a wrong `<` would
    // sum only 3 columns (out[0]=1+2+3=6), so [10,100] proves the boundary.
    let kernel = manifest_path("tests/kernels/rowsum_incl.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "rowsum_incl");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(2)];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
        .expect("dispatch imported rowsum_incl kernel");
    assert_eq!(
        bufs[1].as_f32(),
        &[10.0, 100.0],
        "imported `for j in 0..=N` kernel must sum all N+1 columns (inclusive boundary)"
    );
    eprintln!("imported `for j in 0..=N` (inclusive range) kernel runs on GPU = per-row sums");
}

#[test]
fn for_over_slice_kernel_imported_from_mir_runs_on_gpu() {
    // card 044: `for v in x` (iterate a slice's elements, not an index range). The `slice::Iter` plumbing is
    // detected, the loop var is reused as a `0..x.len()` counter, the bound is `Len(x)`, and the body's `*v`
    // read is rewritten to `x[counter]`. out[r] = sum of all 4 elements = 10 (a wrong bound would undercount).
    let kernel = manifest_path("tests/kernels/for_slice.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "for_slice");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(2)];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
        .expect("dispatch imported for_slice kernel");
    assert_eq!(
        bufs[1].as_f32(),
        &[10.0, 10.0],
        "imported `for v in x` kernel must sum every slice element"
    );
    eprintln!("imported `for v in slice` kernel runs on GPU = full-slice sum");
}

#[test]
fn math_unary_kernel_imported_from_mir_runs_on_gpu() {
    // card 044: a unary float-math method (`x.sqrt()`). The importer maps the method Call to a `MathUnary`
    // rvalue. sqrt of perfect squares is exact, so [4,9,16] -> [2,3,4]. The same path covers exp/sin/cos.
    let kernel = manifest_path("tests/kernels/math_unary.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "math_unary");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [4.0f32, 9.0, 16.0];
    let mut bufs = [KernelBuffer::read_only_f32(&x), KernelBuffer::write_f32(3)];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [3, 1, 1], &mut bufs)
        .expect("dispatch imported math_unary kernel");
    assert_eq!(
        bufs[1].as_f32(),
        &[2.0, 3.0, 4.0],
        "imported `x.sqrt()` kernel must compute square roots"
    );
    eprintln!("imported `x.sqrt()` (MathUnary) kernel runs on GPU = exact roots");
}

#[test]
fn rmsnorm_kernel_imported_from_mir_runs_on_gpu() {
    // card 044: RMSNorm authored in ordinary Rust, imported from MIR, and run on the GPU (sum-of-squares
    // reduction, `D as f32` cast, `.sqrt()`, division, broadcast weight). Verified against a CPU reference
    // with float tolerance (rsqrt is irrational).
    const D: usize = 4;
    let kernel = manifest_path("tests/kernels/rmsnorm.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "rmsnorm");
    assert_eq!(params, 3);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    // two rows of D=4; non-trivial weights so the broadcast multiply is exercised.
    let x = [3.0f32, 4.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0];
    let w = [1.0f32, 2.0, 0.5, 1.5];
    let mut bufs = [
        KernelBuffer::read_only_f32(&x),
        KernelBuffer::read_only_f32(&w),
        KernelBuffer::write_f32(x.len()),
    ];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
        .expect("dispatch imported rmsnorm kernel");

    // CPU reference (same eps as the kernel).
    let eps = 0.00001f32;
    let mut want = vec![0.0f32; x.len()];
    for r in 0..x.len() / D {
        let ss: f32 = (0..D).map(|j| x[r * D + j] * x[r * D + j]).sum();
        let scale = 1.0 / (ss / D as f32 + eps).sqrt();
        for j in 0..D {
            want[r * D + j] = x[r * D + j] * scale * w[j];
        }
    }
    let got = bufs[2].as_f32();
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "rmsnorm[{i}]: got {g}, want {e} (imported kernel != CPU reference)"
        );
    }
    eprintln!(
        "imported RMSNorm kernel runs on GPU = matches CPU reference (a real inference kernel)"
    );
}

#[test]
fn softmax_kernel_imported_from_mir_runs_on_gpu() {
    // card 044: numerically-stable softmax imported from MIR. Exercises `x.max(y)` for the row-max shift,
    // `.exp()`, a `1..D` range, two reductions, and division. Verified against a CPU reference.
    const D: usize = 4;
    let kernel = manifest_path("tests/kernels/softmax.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "softmax");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 10.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&x),
        KernelBuffer::write_f32(x.len()),
    ];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
        .expect("dispatch imported softmax kernel");

    // CPU reference: stable softmax per row.
    let mut want = vec![0.0f32; x.len()];
    for r in 0..x.len() / D {
        let row = &x[r * D..(r + 1) * D];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|v| (v - m).exp()).sum();
        for j in 0..D {
            want[r * D + j] = (row[j] - m).exp() / sum;
        }
    }
    let got = bufs[1].as_f32();
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "softmax[{i}]: got {g}, want {e} (imported kernel != CPU reference)"
        );
    }
    eprintln!("imported stable-softmax kernel runs on GPU = matches CPU reference (uses x.max())");
}

#[test]
fn shape_generic_kernel_runs_any_shape_from_one_body() {
    // card 044: a single imported Body whose row width is a runtime dim (read from `dims: &[u32]`, not a
    // const) runs different shapes: dispatched twice with different `dims` and buffer sizes, the same SPIR-V
    // computes both. kernelgen bakes the dim as a const and generates a Body per shape. u32 dims are passed as
    // i32 bytes (identical bits for small positives).
    let kernel = manifest_path("tests/kernels/rowsum_dyn.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "rowsum_dyn");
    assert_eq!(params, 3);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    // shape A: 2 rows of 4 columns.
    {
        let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let mut bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::read_only_u32(&[4]),
            KernelBuffer::write_f32(2),
        ];
        ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [2, 1, 1], &mut bufs)
            .expect("dispatch shape A");
        assert_eq!(
            bufs[2].as_f32(),
            &[10.0, 100.0],
            "cols=4: per-row sums of 4"
        );
    }
    // shape B: the same compiled_kernel, now 3 rows of 2 columns (dims[0]=2).
    {
        let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0];
        let mut bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::read_only_u32(&[2]),
            KernelBuffer::write_f32(3),
        ];
        ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [3, 1, 1], &mut bufs)
            .expect("dispatch shape B");
        assert_eq!(
            bufs[2].as_f32(),
            &[3.0, 7.0, 30.0],
            "cols=2: per-row sums of 2"
        );
    }
    eprintln!("one imported Body, two shapes (cols=4 and cols=2) - shape-generic kernel on GPU");
}

#[test]
fn shape_generic_rmsnorm_runs_any_hidden_dim_from_one_body() {
    // card 044: the shape-generic capability on a real kernel: RMSNorm whose hidden dim is a runtime
    // `dims[0]`. The same imported Body normalizes two hidden widths (D=4 then D=2), each matching a CPU
    // reference. kernelgen generates a const-baked Body per dim.
    let kernel = manifest_path("kernels/probe/rmsnorm_dyn.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "rmsnorm_dyn");
    assert_eq!(params, 4);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    let rmsnorm_ref = |x: &[f32], w: &[f32], d: usize| -> Vec<f32> {
        let eps = 0.00001f32;
        let mut out = vec![0.0f32; x.len()];
        for r in 0..x.len() / d {
            let ss: f32 = (0..d).map(|j| x[r * d + j] * x[r * d + j]).sum();
            let scale = 1.0 / (ss / d as f32 + eps).sqrt();
            for j in 0..d {
                out[r * d + j] = x[r * d + j] * scale * w[j];
            }
        }
        out
    };
    let run = |d: usize, x: &[f32], w: &[f32]| -> Vec<f32> {
        let mut bufs = [
            KernelBuffer::read_only_f32(x),
            KernelBuffer::read_only_f32(w),
            KernelBuffer::read_only_u32(&[d as u32]),
            KernelBuffer::write_f32(x.len()),
        ];
        let rows = (x.len() / d) as u32;
        ctx.dispatch(
            "test",
            &compiled_kernel,
            [64, 1, 1],
            [rows, 1, 1],
            &mut bufs,
        )
        .expect("dispatch rmsnorm_dyn");
        bufs[3].as_f32().to_vec()
    };

    // hidden dim 4, two rows.
    let x4 = [3.0f32, 4.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0];
    let w4 = [1.0f32, 2.0, 0.5, 1.5];
    // hidden dim 2 (same compiled_kernel), four rows.
    let x2 = [3.0f32, 4.0, 1.0, 2.0, 5.0, 12.0, 0.0, 1.0];
    let w2 = [1.0f32, 0.5];

    for (d, x, w) in [(4usize, &x4[..], &w4[..]), (2usize, &x2[..], &w2[..])] {
        let got = run(d, x, w);
        let want = rmsnorm_ref(x, w, d);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "rmsnorm_dyn d={d} [{i}]: got {g}, want {e}"
            );
        }
    }
    eprintln!(
        "one imported RMSNorm Body, two hidden dims (4 and 2) - shape-generic real kernel on GPU"
    );
}

#[test]
fn imported_kernel_matches_kernelgen_reduce_last() {
    // card 044: an imported-from-Rust kernel is a drop-in for the kernelgen kernel it would supersede.
    // kernelgen's `reduce_last` bakes `cols` as a const (a Body per width); the imported `rowsum_dyn` reads
    // `cols` at runtime, so one Body covers all widths. Both are dispatched on the GPU and their outputs must
    // be identical for each width.
    use poot_codegen::Target;
    use poot_kernel_ir::BinOp;

    let kernel = manifest_path("tests/kernels/rowsum_dyn.rs");
    let (compiled_kernel, _, _needs_error_word) = pootc_compile(kernel, "rowsum_dyn");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let x = [
        1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0,
    ];
    for cols in [4usize, 2] {
        let rows = x.len() / cols;

        // kernelgen's per-shape reduce_last -> SPIR-V.
        let body = poot_test_util::kernel_fixtures::reduce_last("rl", BinOp::Add, cols, 0.0);
        let kg_path = std::env::temp_dir().join(format!("pootc_kg_rl_{cols}.spv"));
        poot_codegen::compile(&body, Target::SpirvVulkan, &kg_path)
            .expect("compile kernelgen reduce_last");
        let kg_spv = std::fs::read(&kg_path).unwrap();
        let kg_kernel = poot_codegen::kernel_handle(&body, Target::SpirvVulkan, kg_spv);
        let mut kg_bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::write_f32(rows),
        ];
        ctx.dispatch(
            "kg",
            &kg_kernel,
            [64, 1, 1],
            [rows as u32, 1, 1],
            &mut kg_bufs,
        )
        .expect("dispatch kernelgen reduce_last");

        // the same imported Body, dispatched with this width as a runtime dim.
        let mut imp_bufs = [
            KernelBuffer::read_only_f32(&x),
            KernelBuffer::read_only_u32(&[cols as u32]),
            KernelBuffer::write_f32(rows),
        ];
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [64, 1, 1],
            [rows as u32, 1, 1],
            &mut imp_bufs,
        )
        .expect("dispatch imported rowsum_dyn");

        assert_eq!(
            kg_bufs[1].as_f32(),
            imp_bufs[2].as_f32(),
            "cols={cols}: imported rowsum_dyn must match kernelgen reduce_last exactly"
        );
    }
    eprintln!(
        "one imported shape-generic Body == kernelgen's per-shape reduce_last family (cols 4 and 2)"
    );
}

#[test]
fn workgroup_index_intrinsics_imported_and_run_on_gpu() {
    // card 044 / spec 054: `local_index()` (lane within the workgroup) and `group_index()` (workgroup id)
    // import to ThreadIndexCall(LocalX/GroupX), distinct from the global `thread_index()`. Dispatched as 2
    // workgroups of 64, lane g writes local*1000 + group, i.e. (g%64)*1000 + (g/64); the test recovers both
    // indices and confirms they are not the global id.
    let kernel = manifest_path("tests/kernels/wg_indices.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "wg_indices");
    assert_eq!(params, 1);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let n = 128usize; // 2 workgroups of 64
    let mut bufs = [KernelBuffer::write_f32(n)];
    ctx.dispatch(
        "test",
        &compiled_kernel,
        [64, 1, 1],
        [n as u32, 1, 1],
        &mut bufs,
    )
    .expect("dispatch wg_indices kernel");
    let out = bufs[0].as_f32();
    for (g, &v) in out.iter().enumerate() {
        let (local, group) = ((v as u32) / 1000, (v as u32) % 1000);
        assert_eq!(
            (local as usize, group as usize),
            (g % 64, g / 64),
            "lane {g}: local_index/group_index must be the within-group lane and workgroup id"
        );
    }
    eprintln!("imported local_index()/group_index() = within-group lane + workgroup id on GPU");
}

#[test]
fn imported_lds_reduction_matches_kernelgen() {
    // card 044 / spec 054 (SC-002): a workgroup-parallel LDS sum-reduction authored in ordinary Rust (64 lanes
    // write to the workgroup-local array, a barrier, lane 0 sums), matching kernelgen's `wg_sum`. The first
    // imported cooperative kernel: barrier + LDS array.
    use poot_codegen::Target;

    let kernel = manifest_path("tests/kernels/wg_reduce.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "wg_reduce");
    assert_eq!(params, 2);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect(); // sum = 2016
    let (wg, threads) = ([64u32, 1, 1], [64u32, 1, 1]); // ONE workgroup of 64 lanes

    let mut imp_bufs = [KernelBuffer::read_only_f32(&a), KernelBuffer::write_f32(1)];
    ctx.dispatch("imp", &compiled_kernel, wg, threads, &mut imp_bufs)
        .expect("dispatch imported LDS reduction");
    assert_eq!(
        imp_bufs[1].as_f32(),
        &[2016.0],
        "imported LDS reduction must sum the 64 lanes"
    );

    // kernelgen's wg_sum, run the same way; must match.
    let kg = poot_test_util::kernel_fixtures::wg_sum("kg_wg_sum", 64);
    let kg_path = std::env::temp_dir().join("pootc_kg_wg_sum.spv");
    poot_codegen::compile(&kg, Target::SpirvVulkan, &kg_path).expect("compile kernelgen wg_sum");
    let kg_spv = std::fs::read(&kg_path).unwrap();
    let kg_kernel = poot_codegen::kernel_handle(&kg, Target::SpirvVulkan, kg_spv);
    let mut kg_bufs = [KernelBuffer::read_only_f32(&a), KernelBuffer::write_f32(1)];
    ctx.dispatch("kg", &kg_kernel, wg, threads, &mut kg_bufs)
        .expect("dispatch kernelgen wg_sum");
    assert_eq!(
        imp_bufs[1].as_f32(),
        kg_bufs[1].as_f32(),
        "imported LDS reduction must match kernelgen wg_sum"
    );
    eprintln!(
        "imported workgroup LDS reduction runs on GPU = matches kernelgen wg_sum (barrier + LDS)"
    );
}

#[test]
fn imported_argmax_matches_host_argmax() {
    // card 044 / card 657: the greedy argmax poot-gpu uses for on-device decode, authored in ordinary Rust
    // and imported from MIR as two kernels, `argmax_partials` (one `(value, index)` partial per workgroup)
    // and `argmax_finalize` (one workgroup reducing the partials), matching the host argmax on the GPU.
    // Exercises multi-array LDS (values + indices), the barrier, a chunked strided scan across workgroups
    // and index tracking. Logits carry an `+ i*1e-4` tiebreak so the argmax is unambiguous; vocab 64 and
    // 5000 (three workgroups of 256 lanes).
    let (partials_kernel, partials_params, _) = pootc_compile(
        manifest_path("tests/kernels/argmax_partials.rs"),
        "argmax_partials",
    );
    let (finalize_kernel, finalize_params, _) = pootc_compile(
        manifest_path("tests/kernels/argmax_finalize.rs"),
        "argmax_finalize",
    );
    assert_eq!((partials_params, finalize_params), (2, 2));

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let wg = [256u32, 1, 1];
    for (vocab, groups) in [(64usize, 1u32), (5000, 3)] {
        // distinct logits with a clear unique max; the i*1e-4 term breaks any integer ties.
        let logits: Vec<f32> = (0..vocab)
            .map(|i| (((i * 48271) % 997) as f32) + (i as f32) * 1e-4)
            .collect();

        let mut stage1 = [
            KernelBuffer::read_only_f32(&logits),
            KernelBuffer::write_f32(2 * groups as usize),
        ];
        ctx.dispatch(
            "imp",
            &partials_kernel,
            wg,
            [groups * 256, 1, 1],
            &mut stage1,
        )
        .expect("dispatch imported argmax_partials");
        let mut stage2 = [
            KernelBuffer::read_only_f32(stage1[1].as_f32()),
            KernelBuffer::write_f32(1),
        ];
        ctx.dispatch("imp", &finalize_kernel, wg, wg, &mut stage2)
            .expect("dispatch imported argmax_finalize");

        // the recovered index is the true argmax.
        let true_argmax = (0..vocab)
            .max_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap())
            .unwrap() as f32;
        assert_eq!(
            stage2[1].as_f32()[0],
            true_argmax,
            "vocab={vocab}: imported argmax index"
        );
    }
    eprintln!("imported two-stage argmax (multi-array LDS) runs on GPU = matches the host argmax");
}

/// SC-005 (deval.md section 8, R472-007): the host's own greedy pick, matching
/// `OpKind::SampleToken { rule: Greedy }`'s oracle exactly - lowest index of the max among finite
/// logits, and the lowest non-finite index (or `-1` if every logit is finite). `token` is forced to `0`
/// when a non-finite logit exists (R-551a-2).
fn host_sample_token_greedy(row: &[f32]) -> (i32, i32) {
    let mut best_idx = -1i32;
    let mut best_val = 0.0f32;
    let mut bad_idx = -1i32;
    for (i, &v) in row.iter().enumerate() {
        if v.is_finite() {
            if best_idx < 0 || v > best_val {
                best_val = v;
                best_idx = i as i32;
            }
        } else if bad_idx < 0 {
            bad_idx = i as i32;
        }
    }
    let token = if bad_idx >= 0 || best_idx < 0 {
        0
    } else {
        best_idx
    };
    (token, bad_idx)
}

#[test]
fn imported_argmax_batched_matches_host_argmax() {
    // card 551a (R472-007, fixing the three bugs the Card 447 consolidation carried): the batched
    // greedy sampler (`argmax_batched.rs`) grids one workgroup per row (`group_index()`) over a flat
    // `[B, vocab]` logits buffer, output `[B, 2]` I32 `(token, non_finite_index)`. Covers: an ordinary
    // row, a row whose logits are all very negative (exercising the old buggy `-1.0e30` sentinel, SC-002),
    // two ties at row 0's max (SC-005: the new tie rule picks the lower index deterministically, unlike
    // the old lane-order-dependent reduce) - a per-lane tie (indices 5 and 5+64, same lane 5 two groups
    // over, already resolved before the cross-lane fold ever sees it) and a cross-lane tie (indices 1
    // and 64, different lanes, the case the fold's tie-break exists for) - and one NaN/+inf/-inf logit
    // each on the batch's last row (SC-007; the same row as the ties only when b=1, so the override
    // can't mask whether the tie-break itself picked the right index).
    let kernel = manifest_path("kernels/sampling/argmax_batched.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "argmax_batched");
    assert_eq!(
        params, 3,
        "argmax_batched has 3 slice params (logits, dims, out)"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    // (batch, vocab): a single row, a small batch, and a vocab not a multiple of 64 (a remainder lane).
    for (b, vocab) in [(1usize, 100usize), (3, 130), (5, 257)] {
        let mut logits = vec![0f32; b * vocab];
        for row in 0..b {
            for i in 0..vocab {
                let base = (((i * 48271 + row * 7919) % 997) as f32) + (i as f32) * 1e-4;
                // odd rows go deep negative: below the old buggy -1.0e30 sentinel (SC-002).
                logits[row * vocab + i] = if row % 2 == 1 { base - 3.3e38 } else { base };
            }
        }
        if vocab > 64 {
            // row 0: force two ties at the row max, SC-005 (the new tie rule picks the lower index
            // deterministically, unlike the old lane-order-dependent reduce). Indices 5 and 5+64 share
            // lane 5 (69 % 64 == 5) across two groups of the strided scan - a per-lane tie, already
            // resolved before the cross-lane fold ever sees it (the lane's own strict `>` scan keeps
            // the first occurrence). Indices 1 and 64 sit in different lanes (64 % 64 == 0, 1 % 64 ==
            // 1) and tie only at the cross-lane fold, the actual case the fold's `ci < bi` tie-break
            // exists for.
            let top = logits[0..vocab]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max)
                + 1.0;
            logits[5] = top; // per-lane tie
            logits[5 + 64] = top;
            logits[1] = top; // cross-lane tie
            logits[64] = top;
        }
        if vocab > 8 {
            // Last row (row 0 only has one row, b=1; otherwise a different row than the ties above,
            // so the non-finite override can't mask whether the tie-break itself picked the right
            // index): one NaN (tail index, V-1, V % 64 != 0 for vocab=100/130/257), one +inf (index 7),
            // one -inf (index 3) - SC-007.
            let nf = (b - 1) * vocab;
            logits[nf + vocab - 1] = f32::NAN;
            logits[nf + 7] = f32::INFINITY;
            logits[nf + 3] = f32::NEG_INFINITY;
        }

        let dims = [vocab as u32];
        let mut bufs = [
            KernelBuffer::read_only_f32(&logits),
            KernelBuffer::read_only_u32(&dims),
            KernelBuffer::write_i32(b * 2),
        ];
        let threads = [64 * b as u32, 1, 1];
        ctx.dispatch(
            "argmax_batched",
            &compiled_kernel,
            [64, 1, 1],
            threads,
            &mut bufs,
        )
        .expect("dispatch batched argmax");
        let got = bufs[2].as_i32().to_vec();

        for row in 0..b {
            let (token, flag) = host_sample_token_greedy(&logits[row * vocab..(row + 1) * vocab]);
            assert_eq!(
                (got[row * 2], got[row * 2 + 1]),
                (token, flag),
                "batch={b} vocab={vocab} row={row}: on-device batched greedy sample (token, \
                 non_finite_index) must match the host oracle"
            );
        }
    }
    eprintln!(
        "batched greedy sample (one workgroup per row) matches the host oracle on ties, the old \
         -1e30 sentinel row and non-finite logits"
    );
}

/// The exact uniform-noise formula `random_uniform.rs` and `OpKind::RandomUniform`'s oracle share
/// (card 551a, R472-007, deval.md section 8.1): murmur3's `fmix32` finalizer over `seed XOR
/// wrapping_mul(idx, 0x9E3779B9)`, then `u = (2*(x>>9)+1) * 2^-24` (an exact I32->F32 conversion below
/// `2^24` and a power-of-two multiply - no rounding artifact is possible, unlike the deleted
/// `(x+0.5)/2^32` formula this replaces).
fn host_hash32(seed: u32, idx: u32) -> u32 {
    let mut x = seed ^ (idx.wrapping_mul(0x9E3779B9));
    x ^= x >> 16;
    x = x.wrapping_mul(0x85EBCA6B);
    x ^= x >> 13;
    x = x.wrapping_mul(0xC2B2AE35);
    x ^= x >> 16;
    x
}

/// `base` is this row's offset into the flat multi-row buffer (`row * cols`): the hash mixes in the
/// GLOBAL flat index `base + i`, not a per-row-reset `i` (deval.md section 8.1's `r*cols+i`, matching
/// `random_uniform.rs` and `poot_eval::ops::sampling::random_uniform`).
fn host_random_uniform_row(seed: u32, base: u32, vocab: usize) -> Vec<f32> {
    (0..vocab as u32)
        .map(|i| {
            let x = host_hash32(seed, base + i);
            let mantissa = (x >> 9) as i32;
            (2.0f32 * (mantissa as f32) + 1.0f32) * 5.9604645e-8f32 // 2^-24
        })
        .collect()
}

/// `-log(-log(u))`: the Gumbel(0,1) transform `poot_graph_ir::ops::sampling::gumbel` composes in-graph
/// (`UnOp::Log`/`UnOp::Neg`), applied host-side here since this file dispatches the kernel directly
/// (bypassing the planner/graph). `random_uniform.rs`'s `u` is always in `[2^-24, 1-2^-24]`, so this is
/// always finite.
fn host_gumbel_noise_row(seed: u32, base: u32, vocab: usize) -> Vec<f32> {
    host_random_uniform_row(seed, base, vocab)
        .into_iter()
        .map(|u| -((-(u.ln())).ln()))
        .collect()
}

#[test]
fn imported_random_uniform_matches_cpu_oracle() {
    // card 551a (SC-001, SC-010): `OpKind::RandomUniform`'s body (`random_uniform.rs`), one thread per
    // output element, must be bit-exact (tier 1, ADR-0101) against the host formula for several
    // (rows, cols) shapes and seeds, including seeds whose hash output exercises every mantissa bit.
    let kernel = manifest_path("kernels/sampling/random_uniform.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "random_uniform");
    assert_eq!(
        params, 3,
        "random_uniform has 3 slice params (seed, dims, out)"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    for (seeds, cols) in [
        (vec![0u32, 1, 0xFFFF_FFFF, 0x9E37_79B9], 17usize),
        (vec![12345, 0xDEAD_BEEF], 257),
        (vec![42], 1),
    ] {
        let rows = seeds.len();
        let seed_words: Vec<i32> = seeds.iter().map(|&s| s as i32).collect();
        let dims = [cols as u32];
        let mut bufs = [
            KernelBuffer::read_only_i32(&seed_words),
            KernelBuffer::read_only_u32(&dims),
            KernelBuffer::write_f32(rows * cols),
        ];
        let threads = [(rows * cols) as u32, 1, 1];
        ctx.dispatch(
            "random_uniform",
            &compiled_kernel,
            [64, 1, 1],
            threads,
            &mut bufs,
        )
        .expect("dispatch random_uniform");
        let got = bufs[2].as_f32().to_vec();

        for (r, &seed) in seeds.iter().enumerate() {
            let want = host_random_uniform_row(seed, (r * cols) as u32, cols);
            assert_eq!(
                &got[r * cols..(r + 1) * cols],
                want.as_slice(),
                "rows={rows} cols={cols} row={r} seed={seed}: on-device RandomUniform must be \
                 bit-exact against the host formula"
            );
            for &u in &got[r * cols..(r + 1) * cols] {
                assert!(
                    u > 0.0 && u < 1.0,
                    "u must stay strictly inside (0,1) so a later log(-log(u)) is finite, got {u}"
                );
            }
        }
    }
    eprintln!("RandomUniform matches the host fmix32+power-of-two-multiply oracle bit-for-bit");
}

/// SC-010 (and the shared body of SC-002/SC-005/SC-006): the oracle for every `SampleToken` Gumbel-family
/// rule (`Gumbel`/`GumbelTopK`/`GumbelTopKTopP`), mirroring `sample_gumbel_argmax_batched.rs` and its two
/// siblings formula for formula - `inv_temp` multiplies (not divides), the row-max/min are value-only
/// reduces with no index (order-independent), the two index-carrying reduces (the lowest non-finite index
/// and the final perturbed argmax) break ties to the lower index. `noise` is precomputed host-side via
/// [`host_gumbel_noise_row`], matching what the graph's `RandomUniform` + `gumbel` composition would
/// produce. `top_k <= 0.0` disables the top-k bisection; `top_p` is `None` for `Gumbel`/`GumbelTopK`.
#[allow(clippy::too_many_arguments)]
fn cpu_gumbel_family_pick(
    logits: &[f32],
    noise: &[f32],
    inv_temp: f32,
    floor_offset: f32,
    noise_scale: f32,
    top_k: f32,
    top_p: Option<f32>,
) -> (i32, i32) {
    let mut max_logit = f32::MIN;
    let mut min_logit = f32::MAX;
    let mut non_finite_idx = -1i32;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() {
            if v > max_logit {
                max_logit = v;
            }
            if v < min_logit {
                min_logit = v;
            }
        } else if non_finite_idx < 0 {
            non_finite_idx = i as i32;
        }
    }

    let mut floor = max_logit + floor_offset;
    if top_k > 0.0 {
        let mut lo = min_logit;
        let mut hi = max_logit;
        for _ in 0..30 {
            let mid = (lo + hi) * 0.5f32;
            // the count is an exact integer (< 2^24 for any test-sized vocab), so summing it in any
            // order (one sequential pass here vs the kernel's 64-way strided partial sums) gives
            // identical f32 bits.
            let mut count = 0.0f32;
            for &v in logits {
                if v >= mid {
                    count += 1.0;
                }
            }
            if count >= top_k {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        if lo > floor {
            floor = lo;
        }
    }
    if let Some(top_p) = top_p {
        let floor1 = floor;
        let quantized_mass = |threshold: f32| -> f32 {
            let mut m = 0.0f32;
            for &v in logits {
                if v >= threshold {
                    let e = ((v - max_logit) * inv_temp).exp();
                    m += (e * 16.0f32).round();
                }
            }
            m
        };
        if top_p <= 0.0 {
            floor = max_logit;
        } else if top_p < 1.0 {
            let z_trunc = quantized_mass(floor1);
            let tp_thresh = (top_p * z_trunc).round();
            let mut tlo = floor1;
            let mut thi = max_logit;
            for _ in 0..30 {
                let mid = (tlo + thi) * 0.5f32;
                if quantized_mass(mid) >= tp_thresh {
                    tlo = mid;
                } else {
                    thi = mid;
                }
            }
            floor = if floor1 > tlo { floor1 } else { tlo };
        }
        // top_p >= 1.0: disabled, floor stays floor1.
    }

    let mut best_idx = -1i32;
    let mut best_val = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if v >= floor {
            let scaled = v * inv_temp;
            let perturbed = scaled + noise_scale * noise[i];
            if best_idx < 0 || perturbed > best_val {
                best_val = perturbed;
                best_idx = i as i32;
            }
        }
    }
    let token = if non_finite_idx >= 0 || best_idx < 0 {
        0
    } else {
        best_idx
    };
    (token, non_finite_idx)
}

#[test]
fn imported_sample_gumbel_argmax_batched_matches_cpu_oracle() {
    // card 551a (R472-007, deval.md section 8): on-device temperature + min-p sampling via the
    // Gumbel-max trick (`sample_gumbel_argmax_batched.rs`), `SampleRule::Gumbel`. `noise` is now a
    // caller-supplied input (no in-kernel hash); `inv_temp` multiplies. The on-device winner (and
    // non-finite index) must be bit-identical to [`cpu_gumbel_family_pick`] for the same (logits, noise,
    // params), across several seeds/temperatures, a min-p case, and a row with a non-finite logit
    // (SC-007: the non-finite flag must still be reported even though this rule has no "kept" sentinel
    // for it).
    let kernel = manifest_path("kernels/sampling/sample_gumbel_argmax_batched.rs");
    let (compiled_kernel, params, _needs_error_word) =
        pootc_compile(kernel, "sample_gumbel_argmax_batched");
    assert_eq!(
        params, 5,
        "sample_gumbel_argmax_batched has 5 slice params (logits, noise, params, dims, out)"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    // (batch, vocab, per-row (seed, temperature, min_p)): several seeds/temperatures plus a min-p row
    // that masks out most of the vocab (min_p=0.9 keeps only tokens within a tight band of the row max).
    #[allow(clippy::type_complexity)]
    let cases: [(usize, usize, &[(u32, f32, f32)]); 3] = [
        (1, 200, &[(12345, 0.8, 0.0)]),
        (
            3,
            337,
            &[(1, 1.0, 0.0), (777, 0.5, 0.0), (0xDEADBEEF, 2.0, 0.0)],
        ),
        (2, 513, &[(42, 1.0, 0.9), (99, 1.3, 0.0)]),
    ];

    for (b, vocab, rows) in cases {
        assert_eq!(rows.len(), b);
        let mut logits = vec![0f32; b * vocab];
        for row in 0..b {
            for i in 0..vocab {
                logits[row * vocab + i] =
                    (((i * 48271 + row * 7919) % 997) as f32) + (i as f32) * 1e-3;
            }
        }
        // row 0 of the first case carries one NaN logit (SC-007), outside any row's deliberately
        // masked min-p band.
        if vocab > 20 {
            logits[10] = f32::NAN;
        }
        let mut noise = vec![0f32; b * vocab];
        let mut params_buf = vec![0f32; b * 3];
        for (row, &(seed, temp, min_p)) in rows.iter().enumerate() {
            noise[row * vocab..(row + 1) * vocab].copy_from_slice(&host_gumbel_noise_row(
                seed,
                (row * vocab) as u32,
                vocab,
            ));
            params_buf[row * 3] = 1.0 / temp; // inv_temp
            params_buf[row * 3 + 1] = if min_p > 0.0 {
                temp * min_p.ln()
            } else {
                -3.4028235e38
            };
            params_buf[row * 3 + 2] = 1.0; // noise_scale
        }
        let dims = [vocab as u32];
        let mut bufs = [
            KernelBuffer::read_only_f32(&logits),
            KernelBuffer::read_only_f32(&noise),
            KernelBuffer::read_only_f32(&params_buf),
            KernelBuffer::read_only_u32(&dims),
            KernelBuffer::write_i32(b * 2),
        ];
        let threads = [64 * b as u32, 1, 1];
        ctx.dispatch(
            "sample_gumbel_argmax_batched",
            &compiled_kernel,
            [64, 1, 1],
            threads,
            &mut bufs,
        )
        .expect("dispatch sample_gumbel_argmax_batched");
        let got = bufs[4].as_i32().to_vec();

        for (row, &(seed, temp, min_p)) in rows.iter().enumerate() {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let row_noise = &noise[row * vocab..(row + 1) * vocab];
            let floor_offset = if min_p > 0.0 {
                temp * min_p.ln()
            } else {
                -3.4028235e38
            };
            let want = cpu_gumbel_family_pick(
                row_logits,
                row_noise,
                1.0 / temp,
                floor_offset,
                1.0,
                0.0,
                None,
            );
            assert_eq!(
                (got[row * 2], got[row * 2 + 1]),
                want,
                "batch={b} vocab={vocab} row={row} seed={seed} temp={temp} min_p={min_p}: on-device \
                 Gumbel-argmax (token, non_finite_index) must match the CPU oracle"
            );
        }
    }
    eprintln!(
        "on-device Gumbel-max temperature/min-p sampling matches the CPU oracle bit-for-bit, noise as \
         an input (no in-kernel hash)"
    );
}

#[test]
fn imported_sample_truncated_gumbel_argmax_batched_matches_cpu_oracle() {
    // card 551a: on-device top-k (+ temperature + min-p) sampling via threshold bisection
    // (`sample_truncated_gumbel_argmax_batched.rs`), `SampleRule::GumbelTopK`. `top_k` is now its own I32
    // operand (not folded into `params`); `noise` is an input. Covers top_k in {1, 8, 100}, a vocab not a
    // multiple of 64 (remainder lanes), top_k=0 (disabled, must match the untruncated rule), and a
    // min-p+top-k interaction (floor = max of the two, both directions).
    let kernel = manifest_path("kernels/sampling/sample_truncated_gumbel_argmax_batched.rs");
    let (compiled_kernel, params, _needs_error_word) =
        pootc_compile(kernel, "sample_truncated_gumbel_argmax_batched");
    assert_eq!(
        params, 6,
        "sample_truncated_gumbel_argmax_batched has 6 slice params (logits, noise, params, top_k, dims, out)"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    // (batch, vocab, per-row (seed, temperature, min_p, top_k)): top_k=0.0 means disabled.
    #[allow(clippy::type_complexity)]
    let cases: [(usize, usize, &[(u32, f32, f32, f32)]); 3] = [
        (1, 200, &[(12345, 0.8, 0.0, 0.0)]),
        (
            3,
            337,
            &[
                (1, 1.0, 0.0, 1.0),
                (777, 0.5, 0.0, 8.0),
                (0xDEADBEEF, 2.0, 0.0, 100.0),
            ],
        ),
        (2, 513, &[(42, 1.0, 0.001, 1.0), (99, 1.3, 0.9999, 100.0)]),
    ];

    for (b, vocab, rows) in cases {
        assert_eq!(rows.len(), b);
        let mut logits = vec![0f32; b * vocab];
        for row in 0..b {
            for i in 0..vocab {
                logits[row * vocab + i] =
                    (((i * 48271 + row * 7919) % 997) as f32) + (i as f32) * 1e-3;
            }
        }
        let mut noise = vec![0f32; b * vocab];
        let mut params_buf = vec![0f32; b * 3];
        let mut top_k_buf = vec![0i32; b];
        for (row, &(seed, temp, min_p, top_k)) in rows.iter().enumerate() {
            noise[row * vocab..(row + 1) * vocab].copy_from_slice(&host_gumbel_noise_row(
                seed,
                (row * vocab) as u32,
                vocab,
            ));
            params_buf[row * 3] = 1.0 / temp;
            params_buf[row * 3 + 1] = if min_p > 0.0 {
                temp * min_p.ln()
            } else {
                -3.4028235e38
            };
            params_buf[row * 3 + 2] = 1.0;
            top_k_buf[row] = top_k as i32;
        }
        let dims = [vocab as u32];
        let mut bufs = [
            KernelBuffer::read_only_f32(&logits),
            KernelBuffer::read_only_f32(&noise),
            KernelBuffer::read_only_f32(&params_buf),
            KernelBuffer::read_only_i32(&top_k_buf),
            KernelBuffer::read_only_u32(&dims),
            KernelBuffer::write_i32(b * 2),
        ];
        let threads = [64 * b as u32, 1, 1];
        ctx.dispatch(
            "sample_truncated_gumbel_argmax_batched",
            &compiled_kernel,
            [64, 1, 1],
            threads,
            &mut bufs,
        )
        .expect("dispatch sample_truncated_gumbel_argmax_batched");
        let got = bufs[5].as_i32().to_vec();

        for (row, &(seed, temp, min_p, top_k)) in rows.iter().enumerate() {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let row_noise = &noise[row * vocab..(row + 1) * vocab];
            let floor_offset = if min_p > 0.0 {
                temp * min_p.ln()
            } else {
                -3.4028235e38
            };
            let want = cpu_gumbel_family_pick(
                row_logits,
                row_noise,
                1.0 / temp,
                floor_offset,
                1.0,
                top_k,
                None,
            );
            assert_eq!(
                (got[row * 2], got[row * 2 + 1]),
                want,
                "batch={b} vocab={vocab} row={row} seed={seed} temp={temp} min_p={min_p} \
                 top_k={top_k}: on-device top-k bisection + Gumbel-argmax must match the CPU oracle"
            );
        }
    }
    eprintln!(
        "on-device top-k threshold-bisection + Gumbel-max sampling matches the CPU oracle bit-for-bit"
    );
}

#[test]
fn imported_sample_topp_gumbel_argmax_batched_matches_cpu_oracle() {
    // card 551a (R-551a-5): on-device top-p (nucleus) sampling via an integer-mass bisection
    // (`sample_topp_gumbel_argmax_batched.rs`), `SampleRule::GumbelTopKTopP`. No scratch buffer and no
    // atomics (unlike the Card 148 original): the bisection is entirely workgroup-local LDS reduction, so
    // this kernel has 6 params, same as the top-k kernel, with `params` carrying a 4th `top_p` column
    // instead of a separate scratch buffer. The quantization scale is 16 (not the old 16384), the largest
    // power of two keeping the row's total quantized mass exact in f32 for realistic vocab sizes (see
    // `sample_topp_gumbel_argmax_batched.rs`); [`cpu_gumbel_family_pick`] uses the identical scale, so
    // this is still an exact-integer-mass comparison, not an approximation.
    let kernel = manifest_path("kernels/sampling/sample_topp_gumbel_argmax_batched.rs");
    let (compiled_kernel, params, _needs_error_word) =
        pootc_compile(kernel, "sample_topp_gumbel_argmax_batched");
    assert_eq!(
        params, 6,
        "sample_topp_gumbel_argmax_batched has 6 slice params (logits, noise, params, top_k, dims, out)"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    // (batch, vocab, per-row (seed, temperature, min_p, top_k, top_p)); top_k/top_p = 0.0/1.0 = disabled.
    // Logits decay by a clear step per index (`peak - step*i`), so the cumulative mass has wide gaps
    // between the top few tokens and the chosen top_p values sit well inside a gap.
    #[allow(clippy::type_complexity)]
    let cases: [(usize, usize, &[(u32, f32, f32, f32, f32)]); 5] = [
        (1, 200, &[(12345, 0.8, 0.0, 0.0, 1.0)]),
        (
            2,
            150,
            &[(555, 0.7, 0.0, 0.0, 0.0), (98765, 1.3, 0.0, 0.0, 0.0)],
        ),
        (
            2,
            337,
            &[(1, 1.0, 0.0, 0.0, 0.9), (777, 1.0, 0.0, 0.0, 0.97)],
        ),
        (
            2,
            257,
            &[(42, 1.0, 0.0, 2.0, 0.97), (99, 1.0, 0.0, 8.0, 0.9)],
        ),
        (1, 128, &[(7, 1.0, 0.5, 0.0, 0.9)]),
    ];

    for (b, vocab, rows) in cases {
        assert_eq!(rows.len(), b);
        let mut logits = vec![0f32; b * vocab];
        for row in 0..b {
            for i in 0..vocab {
                logits[row * vocab + i] = 20.0 - (i as f32);
            }
        }
        let mut noise = vec![0f32; b * vocab];
        let mut params_buf = vec![0f32; b * 4];
        let mut top_k_buf = vec![0i32; b];
        for (row, &(seed, temp, min_p, top_k, top_p)) in rows.iter().enumerate() {
            noise[row * vocab..(row + 1) * vocab].copy_from_slice(&host_gumbel_noise_row(
                seed,
                (row * vocab) as u32,
                vocab,
            ));
            params_buf[row * 4] = 1.0 / temp;
            params_buf[row * 4 + 1] = if min_p > 0.0 {
                temp * min_p.ln()
            } else {
                -3.4028235e38
            };
            params_buf[row * 4 + 2] = 1.0; // noise_scale (top_k rides its own buffer, not params)
            params_buf[row * 4 + 3] = top_p;
            top_k_buf[row] = top_k as i32;
        }
        let dims = [vocab as u32];
        let mut bufs = [
            KernelBuffer::read_only_f32(&logits),
            KernelBuffer::read_only_f32(&noise),
            KernelBuffer::read_only_f32(&params_buf),
            KernelBuffer::read_only_i32(&top_k_buf),
            KernelBuffer::read_only_u32(&dims),
            KernelBuffer::write_i32(b * 2),
        ];
        let threads = [64 * b as u32, 1, 1];
        ctx.dispatch(
            "sample_topp_gumbel_argmax_batched",
            &compiled_kernel,
            [64, 1, 1],
            threads,
            &mut bufs,
        )
        .expect("dispatch sample_topp_gumbel_argmax_batched");
        let got = bufs[5].as_i32().to_vec();

        for (row, &(seed, temp, min_p, top_k, top_p)) in rows.iter().enumerate() {
            let row_logits = &logits[row * vocab..(row + 1) * vocab];
            let row_noise = &noise[row * vocab..(row + 1) * vocab];
            let floor_offset = if min_p > 0.0 {
                temp * min_p.ln()
            } else {
                -3.4028235e38
            };
            let want = cpu_gumbel_family_pick(
                row_logits,
                row_noise,
                1.0 / temp,
                floor_offset,
                1.0,
                top_k,
                Some(top_p),
            );
            assert_eq!(
                (got[row * 2], got[row * 2 + 1]),
                want,
                "batch={b} vocab={vocab} row={row} seed={seed} temp={temp} min_p={min_p} \
                 top_k={top_k} top_p={top_p}: on-device top-p integer-mass bisection + Gumbel-argmax \
                 must match the CPU oracle"
            );
        }
    }
    // SC-006: the degenerate all-filters-coincide case (top_k=1, top_p=0, min_p=1, noise_scale=0).
    // Not literally every logit equal: with ALL logits identical, every filter's floor lands exactly
    // on every logit, so `v >= floor` and `v > floor` both keep either "all" or "none" of them - and
    // "none" falls back to token 0, the SAME index an all-tied row's correct tie-break would pick
    // anyway, which makes the `>=`/`>` boundary unobservable in the final token (checked by hand: a
    // `>= -> >` mutation at the final filter reproduces token 0 either way). Instead: most of the row
    // sits at one low value and exactly two indices (7 and 90, neither index 0) sit at the row max -
    // min_p=1 and top_p<=0 both collapse the floor to exactly that max, so the inclusive filter must
    // keep exactly those two tied survivors (count 2, not vocab, not 0) for the tie-break to correctly
    // pick the lower of them (7); an exclusive filter keeps zero and falls back to 0, a different,
    // observable token.
    {
        let vocab = 128usize;
        let mut logits = vec![0.0f32; vocab];
        logits[7] = 10.0;
        logits[90] = 10.0;
        let noise = vec![0.0f32; vocab];
        let params_buf = [1.0f32, 0.0, 0.0, 0.0]; // inv_temp, floor_offset(min_p=1), noise_scale=0, top_p=0
        let top_k_buf = [1i32];
        let dims = [vocab as u32];
        let mut bufs = [
            KernelBuffer::read_only_f32(&logits),
            KernelBuffer::read_only_f32(&noise),
            KernelBuffer::read_only_f32(&params_buf),
            KernelBuffer::read_only_i32(&top_k_buf),
            KernelBuffer::read_only_u32(&dims),
            KernelBuffer::write_i32(2),
        ];
        ctx.dispatch(
            "sample_topp_gumbel_argmax_batched",
            &compiled_kernel,
            [64, 1, 1],
            [64, 1, 1],
            &mut bufs,
        )
        .expect("dispatch sample_topp_gumbel_argmax_batched (SC-006 degenerate case)");
        let got = bufs[5].as_i32().to_vec();
        let want = cpu_gumbel_family_pick(&logits, &noise, 1.0, 0.0, 0.0, 1.0, Some(0.0));
        assert_eq!(
            (got[0], got[1]),
            want,
            "SC-006 degenerate case (top_k=1, top_p=0, min_p=1, noise_scale=0, two tied survivors at \
             7 and 90): on-device must match the CPU oracle and pick the lower tied index"
        );
        assert_eq!(
            (got[0], got[1]),
            (7, -1),
            "SC-006 degenerate case: the inclusive (>=) filter must keep both tied survivors and the \
             tie-break must pick the lower index (7), not fall back to 0"
        );
    }
    eprintln!(
        "on-device top-p integer-mass bisection + Gumbel-max sampling matches the CPU oracle \
         bit-for-bit, with no scratch buffer and no atomics"
    );
}

/// SC-002 (card 675, ADR 0114 tier 1): `no_contract_add`'s importer wiring reaches NVPTX codegen as the
/// constrained FP intrinsic under `strictfp` (card 628's existing `Rvalue::BinaryOpNoContract` lowering),
/// and the production-compiled PTX for the Gumbel select's perturbed-argmax add carries the PTX-ISA-9.4
/// non-contraction form on both the feeding multiply and the add: an explicit `.rn` rounding modifier
/// (SS9.7.3.3/.4 Notes - an instruction with an explicit rounding modifier "is treated conservatively by
/// the code optimizer", i.e. by `ptxas`'s `-fmad=true` default; a *bare* `mul.f32`/`add.f32` with no
/// modifier is the form `ptxas` is free to contract). No `fma` instruction may appear either way.
///
/// The mutation that can actually fail: `mutate_away_single_no_contract_add` strips the marker (as if card
/// 675's importer wiring never existed) and the body is recompiled through the same `poot-codegen` path.
/// Marked must emit `llvm.experimental.constrained.fadd.f32` + `strictfp`; unmarked must not - a real,
/// provable red/green toggle with no GPU needed. Both still compile to `.rn`-qualified, unfused PTX today:
/// `llc` never attaches LLVM's `contract` fast-math flag to poot-codegen's plain float path (card 628's own
/// `no_contract.rs` established this for its sub-of-two-products shape; confirmed here for the Gumbel
/// formula's single-mul-feeds-add shape, the textbook FMA pattern - a direct local probe before writing
/// this test showed byte-identical `mul.rn.f32`/`add.rn.f32`, no `fma.rn.f32`, for both forms). That is why
/// there is no "unmarked diverges on NVPTX" assertion here (mirroring card 628's own SC-001 framing): the
/// marker is ADR 0114's hedge against a future regression in that `.rn` guarantee, not a divergence this
/// toolchain can reproduce today. Per card 675, this IR/PTX-text contract is now the card's SC-002; SC-001 is the real-hardware bit-exact dispatch, which
/// runs the production asset on a pod and is enforced by the PTX lane -
/// `crates/poot-ptx-gpu/tests/sample_gumbel_no_contract.rs`'s
/// `gumbel_select_nvptx_fma_sensitive_fixture_matches_cpu_oracle` (not here: this crate's test binary has
/// no cargo/rustc on a pod to re-import through `pootc`, so it stays model-free).
///
/// Skips (passes) when `llc` is absent, matching this crate's other NVPTX-text checks.
#[test]
fn gumbel_select_nvptx_add_is_marked_no_contract_and_isa_guaranteed_unfused() {
    if std::process::Command::new("llc")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("llc not on PATH; skipping");
        return;
    }
    let (marked_body, _ptx) = pootc_compile_nvptx(
        manifest_path("kernels/sampling/sample_gumbel_argmax_batched.rs"),
        "sample_gumbel_argmax_batched",
    );
    let no_contract_count = marked_body
        .blocks
        .iter()
        .flat_map(|bb| &bb.statements)
        .filter(|s| {
            matches!(
                s,
                poot_kernel_ir::Statement::Assign(
                    _,
                    poot_kernel_ir::Rvalue::BinaryOpNoContract(poot_kernel_ir::BinOp::Add, _, _)
                )
            )
        })
        .count();
    assert_eq!(
        no_contract_count, 1,
        "the imported Gumbel body should have exactly one no-contract Add (the perturbed-argmax sum); got {no_contract_count}"
    );
    let unmarked_body = mutate_away_single_no_contract_add(&marked_body);

    let dir = std::env::temp_dir().join("pootc-gumbel-nocontract-nvptx");
    std::fs::create_dir_all(&dir).unwrap();
    let mut compiled_ptx_len = std::collections::HashMap::new();
    for (marked, tag, body) in [
        (true, "marked", &marked_body),
        (false, "unmarked", &unmarked_body),
    ] {
        let out = poot_codegen::artifact_path(
            &dir,
            &format!("gumbel_{tag}"),
            poot_codegen::Target::Nvptx,
        );
        let ir = poot_codegen::compile(body, poot_codegen::Target::Nvptx, &out)
            .unwrap_or_else(|e| panic!("Nvptx compile ({tag}) failed: {e}"));
        let has_constrained = ir.contains("llvm.experimental.constrained.fadd.f32");
        let has_strictfp = ir.contains("strictfp");
        assert_eq!(
            has_constrained, marked,
            "{tag}: expected constrained.fadd.f32 present={marked}, IR:\n{ir}"
        );
        assert_eq!(
            has_strictfp, marked,
            "{tag}: expected strictfp present={marked}, IR:\n{ir}"
        );

        let ptx = std::fs::read_to_string(&out).unwrap();
        let lower = ptx.to_lowercase();
        assert!(
            !lower.contains("fma.rn.f32") && !lower.contains("fma.f32"),
            "{tag}: compiled PTX for the Gumbel perturbed-argmax add must have no fma instruction:\n{ptx}"
        );
        assert!(
            ptx.contains("add.rn.f32"),
            "{tag}: compiled PTX lost the explicit .rn on the add - the PTX-ISA-9.4-guaranteed \
             non-contraction form ptxas's -fmad=true default respects:\n{ptx}"
        );
        assert!(
            ptx.contains("mul.rn.f32"),
            "{tag}: compiled PTX lost the explicit .rn on the feeding multiply:\n{ptx}"
        );
        compiled_ptx_len.insert(tag, ptx.len());
    }
    eprintln!(
        "card 675 SC-002: marked ({} bytes) and unmarked ({} bytes) PTX both carry .rn-qualified \
         mul/add with no fma (today's ISA-level guarantee); the marker's own IR-level mutation \
         (constrained.fadd.f32 + strictfp present/absent) is the row that actually toggles red/green",
        compiled_ptx_len["marked"], compiled_ptx_len["unmarked"],
    );
}

#[test]
fn imported_atomic_add_sum_smoke_runs_on_gpu() {
    // card 148 phase 2b: the `atomic_add(buffer, index, value)` imported-kernel intrinsic, an
    // order-independent exact integer reduction. The kernel (`atomic_add_sum_smoke.rs`) has every lane
    // atomically add 1 into the u32 counter at out[0]; with N lanes the counter must equal exactly N, which
    // only holds if the atomic serializes the read-modify-writes. The counter is a u32 uploaded/read back
    // through an f32-typed buffer (raw bytes), so the result is recovered with `.to_bits()`. N=256 with a
    // 64-lane workgroup = 4 workgroups, exercising cross-workgroup global-atomic contention. Proves the
    // imported `Rvalue::GlobalAtomic { Add }` lowers to valid SPIR-V and runs bit-correct on wgpu.
    //
    // Card 559: loads the committed asset through `ImportedKernel::AtomicAddSumSmoke`, not a fresh
    // `pootc_compile` of the source - this is the one real consumer that keeps the manifest entry live.
    let body = poot_graph_plan::ImportedKernel::AtomicAddSumSmoke
        .body()
        .clone();
    assert_eq!(
        body.params().count(),
        1,
        "atomic_add_sum_smoke has 1 slice param (the counter)"
    );
    let out_dir = std::env::temp_dir().join("pootc-atomic-add-sum-smoke");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out_path = poot_codegen::artifact_path(
        &out_dir,
        "atomic_add_sum_smoke",
        poot_codegen::Target::SpirvVulkan,
    );
    poot_codegen::compile(&body, poot_codegen::Target::SpirvVulkan, &out_path)
        .expect("atomic_add_sum_smoke must compile to SPIR-V");
    let spv = std::fs::read(&out_path).unwrap();
    let compiled_kernel =
        poot_codegen::kernel_handle(&body, poot_codegen::Target::SpirvVulkan, spv);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let n = 256u32; // 4 workgroups of 64: cross-workgroup atomic contention
    // the u32 counter rides through an f32 buffer (raw bytes); zero-initialized by write_f32.
    let mut bufs = [KernelBuffer::write_f32(1)];
    ctx.dispatch("test", &compiled_kernel, [64, 1, 1], [n, 1, 1], &mut bufs)
        .expect("dispatch imported atomic_add_sum_smoke kernel");
    let counter = bufs[0].as_f32()[0].to_bits();
    assert_eq!(
        counter, n,
        "the imported atomic-add counter must equal the thread count (no lost updates); got {counter}"
    );
    eprintln!(
        "imported atomic_add(buffer, index, value) intrinsic runs on wgpu = exact cross-workgroup sum {n}"
    );
}

#[test]
fn imported_gather_axis0_matches_kernelgen() {
    // card 044: the axis-0 embedding gather (poot's first kernel each step, never fusable, so always a
    // standalone dispatch). The imported 3-param kernel (row width derived in-kernel as out.len()/index.len())
    // is verified against a CPU reference and kernelgen `gather_axis0_dt` for a few (table_rows, rest,
    // gathered_rows) shapes: the decode case (one gathered row), a multi-row prefill case, and repeated/ragged
    // token ids.
    use poot_codegen::Target;
    use poot_kernel_ir::Ty;

    let kernel = manifest_path("kernels/movement/gather_axis0.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "gather_axis0");
    assert_eq!(params, 3, "gather_axis0: data, index, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (table_rows, rest, rows) in [(1usize, 4usize, 1usize), (5, 8, 1), (5, 3, 4), (10, 7, 3)] {
        let data: Vec<f32> = (0..table_rows * rest)
            .map(|i| ((i % 19) as f32) * 0.1 - 0.6)
            .collect();
        // one f32 token id per gathered row, in 0..table_rows (deterministic, repeats allowed).
        let index: Vec<f32> = (0..rows)
            .map(|r| ((r * 7 + 3) % table_rows) as f32)
            .collect();
        let outs = rows * rest;
        // CPU reference: out[row*rest + r] = data[(index[row] as usize)*rest + r].
        let want: Vec<f32> = (0..outs)
            .map(|i| {
                let (row, r) = (i / rest, i % rest);
                data[(index[row] as usize) * rest + r]
            })
            .collect();

        let mut imp_bufs = [
            KernelBuffer::read_only_f32(&data),
            KernelBuffer::read_only_f32(&index),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut imp_bufs,
        )
        .expect("dispatch imported gather");
        let got = imp_bufs[2].as_f32().to_vec();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "gather[{i}]: {g} vs CPU ref {w} (rows={rows} rest={rest})"
            );
        }

        // kernelgen gather_axis0_dt: same [data, index, out] buffers, same one-thread-per-output grid.
        let kg = poot_test_util::kernel_fixtures::gather_axis0_dt("kg", Ty::F32, rest);
        let kg_path =
            std::env::temp_dir().join(format!("pootc_kg_gather_{table_rows}_{rest}_{rows}.spv"));
        poot_codegen::compile(&kg, Target::SpirvVulkan, &kg_path)
            .expect("compile kernelgen gather");
        let kg_spv = std::fs::read(&kg_path).unwrap();
        let kg_kernel = poot_codegen::kernel_handle(&kg, Target::SpirvVulkan, kg_spv);
        let mut kg_bufs = [
            KernelBuffer::read_only_f32(&data),
            KernelBuffer::read_only_f32(&index),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "kg",
            &kg_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut kg_bufs,
        )
        .expect("dispatch kernelgen gather");
        for (i, (g, kv)) in got.iter().zip(kg_bufs[2].as_f32()).enumerate() {
            assert!(
                (g - kv).abs() < 1e-6,
                "gather[{i}]: imported {g} vs kernelgen {kv}"
            );
        }
    }
    eprintln!(
        "imported gather_axis0 matches kernelgen gather_axis0_dt + CPU (decode + prefill shapes)"
    );
}

#[test]
fn imported_scatter_axis0_matches_kernelgen() {
    // card 044: the axis-0 scatter (write-side inverse of the gather; the MoE expert-routing permutation
    // inversion). The imported 3-param kernel (rest = src.len()/index.len() derived in-kernel) is verified
    // against a CPU reference and kernelgen `scatter_axis0_dt` for a few (rows, rest) shapes, with a reverse
    // permutation index (a real reorder).
    use poot_codegen::Target;
    use poot_kernel_ir::Ty;

    let kernel = manifest_path("kernels/movement/scatter_axis0.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "scatter_axis0");
    assert_eq!(params, 3, "scatter_axis0: src, index, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (rows, rest) in [(1usize, 4usize), (4, 3), (6, 1), (3, 7)] {
        let src: Vec<f32> = (0..rows * rest).map(|i| (i as f32) * 0.5 + 1.0).collect();
        // a reverse permutation: row j -> destination row (rows-1-j).
        let index: Vec<f32> = (0..rows).map(|j| (rows - 1 - j) as f32).collect();
        let outs = rows * rest;
        // CPU reference: out[index[i/rest]*rest + i%rest] = src[i] (i indexes src AND drives the stride math).
        let mut want = vec![0.0f32; outs];
        #[allow(clippy::needless_range_loop)]
        for i in 0..outs {
            let (s, r) = (i / rest, i % rest);
            want[(index[s] as usize) * rest + r] = src[i];
        }

        let mut imp_bufs = [
            KernelBuffer::read_only_f32(&src),
            KernelBuffer::read_only_f32(&index),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut imp_bufs,
        )
        .expect("dispatch imported scatter");
        let got = imp_bufs[2].as_f32().to_vec();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "scatter[{i}]: {g} vs CPU ref {w} (rows={rows} rest={rest})"
            );
        }

        // kernelgen scatter_axis0_dt: same [src, index, out] buffers, same one-thread-per-source grid.
        let kg = poot_kernelgen::scatter_axis0_dt("kg", Ty::F32, rest);
        let kg_path = std::env::temp_dir().join(format!("pootc_kg_scatter_{rows}_{rest}.spv"));
        poot_codegen::compile(&kg, Target::SpirvVulkan, &kg_path)
            .expect("compile kernelgen scatter");
        let kg_spv = std::fs::read(&kg_path).unwrap();
        let kg_kernel = poot_codegen::kernel_handle(&kg, Target::SpirvVulkan, kg_spv);
        let mut kg_bufs = [
            KernelBuffer::read_only_f32(&src),
            KernelBuffer::read_only_f32(&index),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "kg",
            &kg_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut kg_bufs,
        )
        .expect("dispatch kernelgen scatter");
        for (i, (g, kv)) in got.iter().zip(kg_bufs[2].as_f32()).enumerate() {
            assert!(
                (g - kv).abs() < 1e-6,
                "scatter[{i}]: imported {g} vs kernelgen {kv}"
            );
        }
    }
    eprintln!(
        "imported scatter_axis0 matches kernelgen scatter_axis0_dt + CPU (permutation shapes)"
    );
}

#[test]
fn imported_index_remap_matches_kernelgen() {
    // card 044 / card 387: the index-remap copy, kernelgen's shared primitive behind transpose / slice /
    // broadcast. One imported shape-generic kernel (per-dim strides in a metadata buffer) is verified against
    // two things per case: `poot_kernelgen`'s `transpose_dt`/`slice_dt`/`broadcast_dt` on the GPU, and a
    // hand-written oracle (`oracle_transpose`/`oracle_slice`/`oracle_broadcast`, below) computed directly from
    // the semantic definition of each operation via fresh index arithmetic, never building an
    // `(out_stride, out_dim, src_term)` metadata triple and never calling any poot kernel, `poot-kernelgen`
    // function, or graph op.
    //
    // The kernelgen cross-check alone is not independent evidence: `transpose_dt`/`slice_dt`/`broadcast_dt`
    // wrap the same `index_remap_copy` primitive and build strides from
    // `poot_kernelgen::helpers::row_major_strides`, the same row-major formula this test's `rms()` computes. A
    // mistake in the shared convention for expressing transpose/slice/broadcast as an index-remap triple would
    // reproduce on both sides. The oracles never construct a metadata triple, so they cannot inherit it.
    use poot_codegen::Target;
    use poot_kernel_ir::Ty;

    // row-major strides + the meta the imported kernel reads: [rank, src_base, (out_stride,out_dim,src_term)*].
    fn rms(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![1usize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    }
    fn meta(out_shape: &[usize], src_terms: &[usize], src_base: usize) -> Vec<u32> {
        let os = rms(out_shape);
        let mut m = vec![out_shape.len() as u32, src_base as u32];
        for d in 0..out_shape.len() {
            m.push(os[d] as u32);
            m.push(out_shape[d] as u32);
            m.push(src_terms[d] as u32);
        }
        m
    }

    // Independent oracles (card 387). Each decodes/encodes a flat index via a fresh div-mod chain (never a
    // precomputed stride array, never `rms()`) and applies the operation's own semantic definition directly:
    // permute axes, offset-and-copy a range, or replicate along a size-1 axis.
    fn oracle_transpose(input: &[f32], in_shape: &[usize], perm: &[usize]) -> Vec<f32> {
        let rank = in_shape.len();
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let mut out = vec![0.0f32; input.len()];
        for (flat_in, &value) in input.iter().enumerate() {
            let mut in_idx = vec![0usize; rank];
            let mut rem = flat_in;
            for d in (0..rank).rev() {
                in_idx[d] = rem % in_shape[d];
                rem /= in_shape[d];
            }
            let mut flat_out = 0usize;
            for (d, &p) in perm.iter().enumerate() {
                flat_out = flat_out * out_shape[d] + in_idx[p];
            }
            out[flat_out] = value;
        }
        out
    }
    fn oracle_slice(
        input: &[f32],
        in_shape: &[usize],
        axis: usize,
        start: usize,
        len: usize,
    ) -> Vec<f32> {
        let rank = in_shape.len();
        let mut out_shape = in_shape.to_vec();
        out_shape[axis] = len;
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];
        for (flat_out, slot) in out.iter_mut().enumerate() {
            let mut in_idx = vec![0usize; rank];
            let mut rem = flat_out;
            for d in (0..rank).rev() {
                in_idx[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            in_idx[axis] += start;
            let mut flat_in = 0usize;
            for d in 0..rank {
                flat_in = flat_in * in_shape[d] + in_idx[d];
            }
            *slot = input[flat_in];
        }
        out
    }
    fn oracle_broadcast(input: &[f32], in_shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
        let rank = out_shape.len();
        let offset = rank - in_shape.len();
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];
        for (flat_out, slot) in out.iter_mut().enumerate() {
            let mut out_idx = vec![0usize; rank];
            let mut rem = flat_out;
            for d in (0..rank).rev() {
                out_idx[d] = rem % out_shape[d];
                rem /= out_shape[d];
            }
            let mut flat_in = 0usize;
            for (d, &in_dim) in in_shape.iter().enumerate() {
                let od = out_idx[d + offset];
                debug_assert!(
                    in_dim == 1 || in_dim == out_shape[d + offset],
                    "invalid broadcast fixture: in_dim={in_dim} does not broadcast to out_dim={}",
                    out_shape[d + offset]
                );
                let coord = if in_dim == 1 { 0 } else { od };
                flat_in = flat_in * in_dim + coord;
            }
            *slot = input[flat_in];
        }
        out
    }

    let kernel = manifest_path("kernels/movement/index_remap.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "index_remap");
    assert_eq!(params, 3, "index_remap: input, dims, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };

    let dispatch_remap = |ctx: &Context, input: &[f32], m: &[u32], outs: usize| -> Vec<f32> {
        let mut bufs = [
            KernelBuffer::read_only_f32(input),
            KernelBuffer::read_only_u32(m),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch imported index_remap");
        bufs[2].as_f32().to_vec()
    };
    let dispatch_kg =
        |ctx: &Context, kg: &poot_kernel_ir::Body, input: &[f32], outs: usize| -> Vec<f32> {
            let kg_path = std::env::temp_dir().join("pootc_kg_remap.spv");
            poot_codegen::compile(kg, Target::SpirvVulkan, &kg_path)
                .expect("compile kernelgen remap");
            let kg_spv = std::fs::read(&kg_path).unwrap();
            let kg_kernel = poot_codegen::kernel_handle(kg, Target::SpirvVulkan, kg_spv);
            let mut bufs = [
                KernelBuffer::read_only_f32(input),
                KernelBuffer::write_f32(outs),
            ];
            ctx.dispatch("kg", &kg_kernel, [64, 1, 1], [outs as u32, 1, 1], &mut bufs)
                .expect("dispatch kernelgen remap");
            bufs[1].as_f32().to_vec()
        };
    // The reference is either kernelgen or an independent oracle, so the message says "reference"; the label
    // names which.
    let assert_eq_f32 = |label: &str, a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len(), "{label}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() < 1e-6,
                "{label}[{i}]: imported {x} vs reference {y}"
            );
        }
    };

    // (a) TRANSPOSE [2,3,4] with perm [2,0,1] -> out [4,2,3]; src_term[d] = in_stride[perm[d]].
    {
        let in_shape = [2usize, 3, 4];
        let perm = [2usize, 0, 1];
        let out_shape: Vec<usize> = perm.iter().map(|&p| in_shape[p]).collect();
        let in_strides = rms(&in_shape);
        let src_terms: Vec<usize> = (0..out_shape.len()).map(|d| in_strides[perm[d]]).collect();
        let input: Vec<f32> = (0..in_shape.iter().product::<usize>())
            .map(|i| i as f32)
            .collect();
        let outs = out_shape.iter().product::<usize>();
        let imp = dispatch_remap(&ctx, &input, &meta(&out_shape, &src_terms, 0), outs);
        assert_eq_f32(
            "transpose oracle",
            &imp,
            &oracle_transpose(&input, &in_shape, &perm),
        );
        let kg = poot_kernelgen::transpose_dt("kg", Ty::F32, &out_shape, &in_shape, &perm);
        assert_eq_f32("transpose", &imp, &dispatch_kg(&ctx, &kg, &input, outs));
    }
    // (b) SLICE [4,5] axis 1, start 1, len 3 -> out [4,3]; src_term[d] = in_stride[d], base = start*in_stride[axis].
    {
        let in_shape = [4usize, 5];
        let (axis, start, len) = (1usize, 1usize, 3usize);
        let mut out_shape = in_shape.to_vec();
        out_shape[axis] = len;
        let in_strides = rms(&in_shape);
        let src_terms: Vec<usize> = (0..out_shape.len()).map(|d| in_strides[d]).collect();
        let input: Vec<f32> = (0..in_shape.iter().product::<usize>())
            .map(|i| i as f32)
            .collect();
        let outs = out_shape.iter().product::<usize>();
        let base = start * in_strides[axis];
        let imp = dispatch_remap(&ctx, &input, &meta(&out_shape, &src_terms, base), outs);
        assert_eq_f32(
            "slice oracle",
            &imp,
            &oracle_slice(&input, &in_shape, axis, start, len),
        );
        let kg = poot_kernelgen::slice_dt("kg", Ty::F32, &out_shape, &in_shape, axis, start);
        assert_eq_f32("slice", &imp, &dispatch_kg(&ctx, &kg, &input, outs));
    }
    // (c) BROADCAST [3,1] -> [3,4]; src_term = effective stride (0 on the broadcast dim).
    {
        let in_shape = [3usize, 1];
        let out_shape = [3usize, 4];
        // broadcast effective strides (right-aligned): 0 where the operand dim is 1, else its own stride.
        let own = rms(&in_shape);
        let offset = out_shape.len() - in_shape.len();
        let src_terms: Vec<usize> = (0..out_shape.len())
            .map(|d| {
                if d < offset || in_shape[d - offset] == 1 {
                    0
                } else {
                    own[d - offset]
                }
            })
            .collect();
        let input: Vec<f32> = (0..in_shape.iter().product::<usize>())
            .map(|i| i as f32 + 1.0)
            .collect();
        let outs = out_shape.iter().product::<usize>();
        let imp = dispatch_remap(&ctx, &input, &meta(&out_shape, &src_terms, 0), outs);
        assert_eq_f32(
            "broadcast oracle",
            &imp,
            &oracle_broadcast(&input, &in_shape, &out_shape),
        );
        let kg = poot_kernelgen::broadcast_dt("kg", Ty::F32, &out_shape, &in_shape);
        assert_eq_f32("broadcast", &imp, &dispatch_kg(&ctx, &kg, &input, outs));
    }
    // (d) card 387: BROADCAST expanding two axes at once, [1,3,1] -> [2,3,4] (axis 0: 1->2, axis 2: 1->4;
    // axis 1 stays 3->3). Case (c) only expands one axis.
    {
        let in_shape = [1usize, 3, 1];
        let out_shape = [2usize, 3, 4];
        let own = rms(&in_shape);
        let offset = out_shape.len() - in_shape.len();
        let src_terms: Vec<usize> = (0..out_shape.len())
            .map(|d| {
                if d < offset || in_shape[d - offset] == 1 {
                    0
                } else {
                    own[d - offset]
                }
            })
            .collect();
        let input: Vec<f32> = (0..in_shape.iter().product::<usize>())
            .map(|i| i as f32 + 1.0)
            .collect();
        let outs = out_shape.iter().product::<usize>();
        let imp = dispatch_remap(&ctx, &input, &meta(&out_shape, &src_terms, 0), outs);
        assert_eq_f32(
            "multi-axis broadcast oracle",
            &imp,
            &oracle_broadcast(&input, &in_shape, &out_shape),
        );
        let kg = poot_kernelgen::broadcast_dt("kg", Ty::F32, &out_shape, &in_shape);
        assert_eq_f32(
            "multi-axis broadcast",
            &imp,
            &dispatch_kg(&ctx, &kg, &input, outs),
        );
    }
    // (e) card 387: rank-1 SLICE, [10] axis 0 start 3 len 4 -> out [4] (none of (a)-(d) is rank-1).
    {
        let in_shape = [10usize];
        let (axis, start, len) = (0usize, 3usize, 4usize);
        let mut out_shape = in_shape.to_vec();
        out_shape[axis] = len;
        let in_strides = rms(&in_shape);
        let src_terms: Vec<usize> = (0..out_shape.len()).map(|d| in_strides[d]).collect();
        let input: Vec<f32> = (0..in_shape.iter().product::<usize>())
            .map(|i| i as f32 * 0.5 - 1.0)
            .collect();
        let outs = out_shape.iter().product::<usize>();
        let base = start * in_strides[axis];
        let imp = dispatch_remap(&ctx, &input, &meta(&out_shape, &src_terms, base), outs);
        assert_eq_f32(
            "rank-1 slice oracle",
            &imp,
            &oracle_slice(&input, &in_shape, axis, start, len),
        );
        let kg = poot_kernelgen::slice_dt("kg", Ty::F32, &out_shape, &in_shape, axis, start);
        assert_eq_f32("rank-1 slice", &imp, &dispatch_kg(&ctx, &kg, &input, outs));
    }
    eprintln!(
        "imported index_remap matches an independent oracle AND kernelgen transpose_dt + slice_dt + \
         broadcast_dt on the GPU"
    );
}

#[test]
fn imported_dyn_update_slice_matches_kernelgen() {
    // card 044: the dynamic-offset update-slice (poot's contiguous KV-cache write): copy `operand`, but
    // overwrite the `extent`-long slice along `axis` at the runtime offset `index[0]` with `update`. The
    // imported kernel (per-dim strides in a metadata buffer, idx read at runtime) is verified byte-identical to
    // kernelgen `dyn_update_slice_dynamic_dt` and a CPU reference for several (out_shape, axis, extent, idx)
    // cases, including idx=0 (slice at the start), an interior idx, and extent=1 (the single-token KV append).
    use poot_codegen::Target;
    use poot_kernel_ir::Ty;

    fn rms(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![1usize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    }
    // meta: [rank, axis, extent, (out_stride, out_dim, upd_stride) * rank].
    fn meta(out_shape: &[usize], axis: usize, extent: usize) -> Vec<u32> {
        let os = rms(out_shape);
        let mut us = out_shape.to_vec();
        us[axis] = extent;
        let ups = rms(&us);
        let mut m = vec![out_shape.len() as u32, axis as u32, extent as u32];
        for d in 0..out_shape.len() {
            m.push(os[d] as u32);
            m.push(out_shape[d] as u32);
            m.push(ups[d] as u32);
        }
        m
    }

    let kernel = manifest_path("kernels/movement/dyn_update_slice.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "dyn_update_slice");
    assert_eq!(
        params, 5,
        "dyn_update_slice: operand, update, index, dims, out"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (out_shape, axis, extent, idx) in [
        (vec![4usize, 5usize], 0usize, 2usize, 1usize),
        (vec![4, 5], 1, 2, 3),
        (vec![3, 6, 2], 1, 1, 4), // extent=1: the single-token KV append
        (vec![8], 0, 3, 0),       // idx=0: slice at the start
    ] {
        let mut upd_shape = out_shape.clone();
        upd_shape[axis] = extent;
        let outs = out_shape.iter().product::<usize>();
        let upds = upd_shape.iter().product::<usize>();
        let operand: Vec<f32> = (0..outs).map(|i| i as f32 * 0.5).collect();
        let update: Vec<f32> = (0..upds).map(|i| 100.0 + i as f32).collect();
        let index = [idx as f32];

        // CPU reference.
        let os = rms(&out_shape);
        let ups = rms(&upd_shape);
        let mut want = operand.clone();
        for (i, w) in want.iter_mut().enumerate() {
            let coord_axis = (i / os[axis]) % out_shape[axis];
            if coord_axis >= idx && coord_axis < idx + extent {
                let mut acc = 0usize;
                for d in 0..out_shape.len() {
                    let coord_d = (i / os[d]) % out_shape[d];
                    let c = if d == axis { coord_d - idx } else { coord_d };
                    acc += c * ups[d];
                }
                *w = update[acc];
            }
        }

        let mut imp_bufs = [
            KernelBuffer::read_only_f32(&operand),
            KernelBuffer::read_only_f32(&update),
            KernelBuffer::read_only_f32(&index),
            KernelBuffer::read_only_u32(&meta(&out_shape, axis, extent)),
            KernelBuffer::write_f32(outs),
        ];
        // card 159: the kernel folds onto a 2-D grid, baking `WORKGROUP_SIZE = 256`; the dispatch wg must match
        // (the SPIR-V's LocalSize is baked from it). For these small shapes one workgroup suffices.
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [256, 1, 1],
            [256, 1, 1],
            &mut imp_bufs,
        )
        .expect("dispatch imported dyn_update_slice");
        let got = imp_bufs[4].as_f32().to_vec();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "dus[{i}]: {g} vs CPU ref {w} ({out_shape:?} axis={axis} extent={extent} idx={idx})"
            );
        }

        // kernelgen dyn_update_slice_dynamic_dt: same [operand, update, index, out] buffers.
        let kg =
            poot_kernelgen::dyn_update_slice_dynamic_dt("kg", Ty::F32, &out_shape, axis, extent);
        let kg_path = std::env::temp_dir().join(format!("pootc_kg_dus_{axis}_{extent}_{idx}.spv"));
        poot_codegen::compile(&kg, Target::SpirvVulkan, &kg_path).expect("compile kernelgen dus");
        let kg_spv = std::fs::read(&kg_path).unwrap();
        let kg_kernel = poot_codegen::kernel_handle(&kg, Target::SpirvVulkan, kg_spv);
        let mut kg_bufs = [
            KernelBuffer::read_only_f32(&operand),
            KernelBuffer::read_only_f32(&update),
            KernelBuffer::read_only_f32(&index),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "kg",
            &kg_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut kg_bufs,
        )
        .expect("dispatch kernelgen dus");
        for (i, (g, kv)) in got.iter().zip(kg_bufs[3].as_f32()).enumerate() {
            assert!(
                (g - kv).abs() < 1e-6,
                "dus[{i}]: imported {g} vs kernelgen {kv}"
            );
        }
    }
    eprintln!("imported dyn_update_slice matches kernelgen dyn_update_slice_dynamic_dt + CPU");
}

#[test]
fn imported_concat2_matches_kernelgen() {
    // card 044: two-input concat (RoPE rotate_half, the KV-cache append, the MoE row-assembly reduce). The
    // imported branchless kernel (per-dim a/b strides in a metadata buffer) is verified byte-identical to
    // kernelgen `concat2_dt` and a CPU reference for several (a_shape, b_shape, axis) cases, including a
    // last-axis concat (RoPE), a leading-axis concat, and unequal split sizes.
    use poot_codegen::Target;
    use poot_kernel_ir::Ty;

    fn rms(shape: &[usize]) -> Vec<usize> {
        let mut s = vec![1usize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    }
    // meta: [rank, axis, a_axis_len, (out_stride, out_dim, a_stride, b_stride) * rank].
    fn meta(out_shape: &[usize], axis: usize, a_shape: &[usize], b_shape: &[usize]) -> Vec<u32> {
        let os = rms(out_shape);
        let as_ = rms(a_shape);
        let bs = rms(b_shape);
        let mut m = vec![out_shape.len() as u32, axis as u32, a_shape[axis] as u32];
        for d in 0..out_shape.len() {
            m.push(os[d] as u32);
            m.push(out_shape[d] as u32);
            m.push(as_[d] as u32);
            m.push(bs[d] as u32);
        }
        m
    }

    let kernel = manifest_path("kernels/movement/concat2.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "concat2");
    assert_eq!(params, 4, "concat2: a, b, dims, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    // (a_shape, b_shape, axis): last-axis concat (RoPE), leading-axis concat, unequal split.
    let cases: &[(Vec<usize>, Vec<usize>, usize)] = &[
        (vec![2, 3], vec![2, 3], 1),
        (vec![2, 3], vec![2, 5], 1),
        (vec![2, 4], vec![3, 4], 0),
        (vec![2, 3, 4], vec![2, 3, 1], 2),
    ];
    for (a_shape, b_shape, axis) in cases {
        let axis = *axis;
        let mut out_shape = a_shape.clone();
        out_shape[axis] = a_shape[axis] + b_shape[axis];
        let an = a_shape.iter().product::<usize>();
        let bn = b_shape.iter().product::<usize>();
        let outs = out_shape.iter().product::<usize>();
        let av: Vec<f32> = (0..an).map(|i| i as f32).collect();
        let bv: Vec<f32> = (0..bn).map(|i| 100.0 + i as f32).collect();

        // CPU reference.
        let os = rms(&out_shape);
        let as_ = rms(a_shape);
        let bs = rms(b_shape);
        let mut want = vec![0.0f32; outs];
        for (i, w) in want.iter_mut().enumerate() {
            let coord_axis = (i / os[axis]) % out_shape[axis];
            if coord_axis < a_shape[axis] {
                let mut f = 0usize;
                for d in 0..out_shape.len() {
                    f += ((i / os[d]) % out_shape[d]) * as_[d];
                }
                *w = av[f];
            } else {
                let mut f = 0usize;
                for d in 0..out_shape.len() {
                    let c = (i / os[d]) % out_shape[d];
                    let c = if d == axis { c - a_shape[axis] } else { c };
                    f += c * bs[d];
                }
                *w = bv[f];
            }
        }

        let mut imp_bufs = [
            KernelBuffer::read_only_f32(&av),
            KernelBuffer::read_only_f32(&bv),
            KernelBuffer::read_only_u32(&meta(&out_shape, axis, a_shape, b_shape)),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut imp_bufs,
        )
        .expect("dispatch imported concat2");
        let got = imp_bufs[3].as_f32().to_vec();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "concat2[{i}]: {g} vs CPU ref {w} ({a_shape:?}+{b_shape:?} axis={axis})"
            );
        }

        // kernelgen concat2_dt: same [a, b, out] buffers.
        let kg = poot_kernelgen::concat2_dt("kg", Ty::F32, &out_shape, axis, a_shape, b_shape);
        let kg_path = std::env::temp_dir().join(format!("pootc_kg_concat2_{axis}_{an}_{bn}.spv"));
        poot_codegen::compile(&kg, Target::SpirvVulkan, &kg_path)
            .expect("compile kernelgen concat2");
        let kg_spv = std::fs::read(&kg_path).unwrap();
        let kg_kernel = poot_codegen::kernel_handle(&kg, Target::SpirvVulkan, kg_spv);
        let mut kg_bufs = [
            KernelBuffer::read_only_f32(&av),
            KernelBuffer::read_only_f32(&bv),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "kg",
            &kg_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut kg_bufs,
        )
        .expect("dispatch kernelgen concat2");
        for (i, (g, kv)) in got.iter().zip(kg_bufs[2].as_f32()).enumerate() {
            assert!(
                (g - kv).abs() < 1e-6,
                "concat2[{i}]: imported {g} vs kernelgen {kv}"
            );
        }
    }
    eprintln!(
        "imported concat2 matches kernelgen concat2_dt + CPU (last/leading axis, unequal split)"
    );
}

#[test]
fn imported_scatter_update_matches_kernelgen() {
    // card 044: the paged-KV scatter-update (card 059): write `src` rows into a copy of the `base` pool at the
    // positions in `inv` (a source row, or -1 to keep base). The imported branchless kernel (rest =
    // out.len()/inv.len() derived in-kernel) is verified byte-identical to kernelgen `scatter_update_dt` and a
    // CPU reference for a few (pool, n, rest) shapes, with a mix of filled and empty (-1) slots.
    use poot_codegen::Target;
    use poot_kernel_ir::Ty;

    let kernel = manifest_path("kernels/movement/scatter_update.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "scatter_update");
    assert_eq!(params, 4, "scatter_update: base, src, inv, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (pool, n, rest) in [(4usize, 2usize, 3usize), (5, 3, 1), (3, 3, 4), (6, 2, 2)] {
        let base: Vec<f32> = (0..pool * rest).map(|i| i as f32 * 0.25).collect();
        let src: Vec<f32> = (0..n * rest).map(|i| 100.0 + i as f32).collect();
        // inv[row]: a source row in 0..n for some slots, -1 (empty -> keep base) for others.
        let inv: Vec<f32> = (0..pool)
            .map(|row| {
                if row.is_multiple_of(2) && row / 2 < n {
                    (row / 2) as f32
                } else {
                    -1.0
                }
            })
            .collect();
        let outs = pool * rest;
        // CPU reference: out[i] = src[inv[i/rest]*rest + i%rest] if inv>=0 else base[i].
        let want: Vec<f32> = (0..outs)
            .map(|i| {
                let (row, r) = (i / rest, i % rest);
                let j = inv[row] as i32;
                if j >= 0 {
                    src[j as usize * rest + r]
                } else {
                    base[i]
                }
            })
            .collect();

        let mut imp_bufs = [
            KernelBuffer::read_only_f32(&base),
            KernelBuffer::read_only_f32(&src),
            KernelBuffer::read_only_f32(&inv),
            KernelBuffer::write_f32(outs),
        ];
        // card 159: the kernel folds onto a 2-D grid, baking `WORKGROUP_SIZE = 256`; the dispatch wg must match
        // (the SPIR-V's LocalSize is baked from it), so `threads` must cover the `x_groups*256` the kernel derives
        // from `out.len()`. For these small shapes one workgroup of 256 lanes suffices.
        ctx.dispatch(
            "imp",
            &compiled_kernel,
            [256, 1, 1],
            [256, 1, 1],
            &mut imp_bufs,
        )
        .expect("dispatch imported scatter_update");
        let got = imp_bufs[3].as_f32().to_vec();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-6,
                "scatter_update[{i}]: {g} vs CPU ref {w} (pool={pool} n={n} rest={rest})"
            );
        }

        // kernelgen scatter_update_dt: same [base, src, inv, out] buffers.
        let kg = poot_kernelgen::scatter_update_dt("kg", Ty::F32, rest);
        let kg_path = std::env::temp_dir().join(format!("pootc_kg_su_{pool}_{n}_{rest}.spv"));
        poot_codegen::compile(&kg, Target::SpirvVulkan, &kg_path).expect("compile kernelgen su");
        let kg_spv = std::fs::read(&kg_path).unwrap();
        let kg_kernel = poot_codegen::kernel_handle(&kg, Target::SpirvVulkan, kg_spv);
        let mut kg_bufs = [
            KernelBuffer::read_only_f32(&base),
            KernelBuffer::read_only_f32(&src),
            KernelBuffer::read_only_f32(&inv),
            KernelBuffer::write_f32(outs),
        ];
        ctx.dispatch(
            "kg",
            &kg_kernel,
            [64, 1, 1],
            [outs as u32, 1, 1],
            &mut kg_bufs,
        )
        .expect("dispatch kernelgen su");
        for (i, (g, kv)) in got.iter().zip(kg_bufs[3].as_f32()).enumerate() {
            assert!(
                (g - kv).abs() < 1e-6,
                "scatter_update[{i}]: imported {g} vs kernelgen {kv}"
            );
        }
    }
    eprintln!(
        "imported scatter_update matches kernelgen scatter_update_dt + CPU (filled + empty slots)"
    );
}

#[test]
fn imported_tiled_gemm_matches_reference() {
    // card 044: a workgroup-tiled GEMM with an in-loop barrier and 2-D multi-array LDS tiling. One TS x TS
    // output tile per workgroup, 64 lanes, A/B sub-tiles staged through LDS arrays 0/1 with a barrier inside
    // the K-tile loop. Verified vs a CPU reference for several tile-aligned shapes, notably K=16/24 (the
    // K-loop iterates 2-3 times, so the in-loop barrier and loop-carried LDS staging run). A race or a
    // non-convergent barrier would show as wrong or non-deterministic output.
    const TS: usize = 8;
    let kernel = manifest_path("tests/kernels/tiled_gemm.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "tiled_gemm");
    assert_eq!(params, 4, "tiled_gemm: a, b, dims, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (m, k, n) in [
        (8usize, 8usize, 8usize),
        (16, 16, 16),
        (24, 8, 16),
        (32, 16, 8),
        (8, 24, 24),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.7).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect();
        // CPU reference: C[r,c] = sum_j A[r,j] * B[j,c].
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (r, c) = (idx / n, idx % n);
                (0..k).map(|j| a[r * k + j] * b[j * n + c]).sum()
            })
            .collect();
        // one workgroup per TS x TS output tile, 64 lanes each.
        let tiles = (m / TS) * (n / TS);
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_u32(&[m as u32, k as u32, n as u32]),
            KernelBuffer::write_f32(m * n),
        ];
        ctx.dispatch(
            "tg",
            &compiled_kernel,
            [64, 1, 1],
            [(tiles * 64) as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch imported tiled_gemm");
        let got = bufs[3].as_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-3,
                "tiled_gemm {m}x{k}x{n} elem {i} (r={},c={}): {g} vs CPU ref {w}",
                i / n,
                i % n
            );
        }
    }
    eprintln!(
        "imported tiled GEMM (in-loop barrier + 2-D LDS tiling) matches CPU - the last kernelgen kernel type \
         is importable + correct on the GPU"
    );
}

#[test]
fn imported_tiled_gemm_masked_matches_reference() {
    // card 044: the tiled GEMM for ragged shapes (tiled_gemm.rs + branchless bounds masking). The shapes are
    // not multiples of TS=8, so edge tiles stage out-of-bounds elements that must be zeroed by the
    // `* rmask * kmask` masks, and K not a multiple of TS makes the last K-tile partial. Proves the mask is
    // correct and did not break the convergence of the in-loop barrier (a divergent mask would hang or
    // corrupt). Verified vs a CPU A@B reference.
    const TS: usize = 8;
    let kernel = manifest_path("tests/kernels/tiled_gemm_masked.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "tiled_gemm_masked");
    assert_eq!(params, 4, "tiled_gemm_masked: a, b, dims, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (m, k, n) in [
        (7usize, 5usize, 9usize),
        (13, 11, 17),
        (10, 10, 10),
        (1, 3, 2),
        (20, 1, 20),
        (16, 16, 16), // an aligned shape still works through the masked kernel
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.7).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect();
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (r, c) = (idx / n, idx % n);
                (0..k).map(|j| a[r * k + j] * b[j * n + c]).sum()
            })
            .collect();
        // ragged grid: ceil(M/TS) x ceil(N/TS) tiles, 64 lanes each.
        let tiles = m.div_ceil(TS) * n.div_ceil(TS);
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_u32(&[m as u32, k as u32, n as u32]),
            KernelBuffer::write_f32(m * n),
        ];
        ctx.dispatch(
            "tgm",
            &compiled_kernel,
            [64, 1, 1],
            [(tiles * 64) as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch imported tiled_gemm_masked");
        let got = bufs[3].as_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-3,
                "tiled_gemm_masked {m}x{k}x{n} elem {i} (r={},c={}): {g} vs CPU ref {w}",
                i / n,
                i % n
            );
        }
    }
    eprintln!(
        "imported ragged tiled GEMM (branchless bounds masking + in-loop barrier) matches CPU - the mask is \
         correct and convergence-safe"
    );
}

#[test]
fn imported_tiled_gemm_coarsened_matches_reference() {
    // card 044: the thread-coarsened tiled GEMM (kernelgen tiled_gemm_dt's structure): each of the 64 lanes
    // computes two output rows of a 2*TS x TS tile, so the A sub-tile is 128 slots (bigger than the 64-lane
    // workgroup). The first kernel to use `const LDS_SIZE` to size its LDS arrays independently of the lane
    // count. Verified vs a CPU A@B reference over shapes spanning multiple 2*TS row-tiles and ragged shapes
    // where the second row is out of bounds.
    const TS: usize = 8;
    let kernel = manifest_path("kernels/contraction/tiled_gemm_coarsened.rs");
    let (compiled_kernel, params, _needs_error_word) =
        pootc_compile(kernel, "tiled_gemm_coarsened");
    assert_eq!(params, 4, "tiled_gemm_coarsened: a, b, dims, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (m, k, n) in [
        (16usize, 8usize, 8usize), // one 2*TS x TS tile
        (32, 16, 16),              // multiple row-tiles, multi K-tile
        (17, 11, 9),               // ragged: 2 row-tiles, row1 partly out of bounds, partial K-tile
        (7, 5, 9),                 // M < TS: row1 fully out of bounds for every lane
        (33, 10, 20),              // ragged across all dims
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.7).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect();
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (r, c) = (idx / n, idx % n);
                (0..k).map(|j| a[r * k + j] * b[j * n + c]).sum()
            })
            .collect();
        // coarsened grid: ceil(M/(2*TS)) x ceil(N/TS) tiles, 64 lanes each.
        let tiles = m.div_ceil(2 * TS) * n.div_ceil(TS);
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_u32(&[m as u32, k as u32, n as u32, 0]),
            KernelBuffer::write_f32(m * n),
        ];
        ctx.dispatch(
            "tgc",
            &compiled_kernel,
            [64, 1, 1],
            [(tiles * 64) as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch imported tiled_gemm_coarsened");
        let got = bufs[3].as_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-3,
                "tiled_gemm_coarsened {m}x{k}x{n} elem {i} (r={},c={}): {g} vs CPU ref {w}",
                i / n,
                i % n
            );
        }
    }
    eprintln!(
        "imported coarsened tiled GEMM (2 rows/lane, 128-slot LDS via const LDS_SIZE) matches CPU - the LDS \
         array size is now decoupled from the lane count"
    );
}

#[test]
fn imported_tiled_gemm_coarsened_bias_matches_reference() {
    // card 044: the coarsened tiled GEMM with the fused bias epilogue (the q/k/v projections at prefill):
    // tiled_gemm_coarsened.rs + `out[.., col] += bias[col]` on both rows. The Body the planner swaps in for
    // the M>1 bias projection (Plan::ComputeMeta). Verified vs a CPU `A@B + bias` reference.
    const TS: usize = 8;
    let kernel = manifest_path("kernels/contraction/tiled_gemm_coarsened_bias.rs");
    let (compiled_kernel, params, _needs_error_word) =
        pootc_compile(kernel, "tiled_gemm_coarsened_bias");
    assert_eq!(
        params, 5,
        "tiled_gemm_coarsened_bias: a, b, bias, dims, out"
    );

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (m, k, n) in [
        (16usize, 8usize, 8usize),
        (17, 11, 9),
        (7, 5, 9),
        (33, 10, 20),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1 - 0.7).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.07 - 0.3).collect();
        let bias: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.2 - 0.3).collect();
        let want: Vec<f32> = (0..m * n)
            .map(|idx| {
                let (r, c) = (idx / n, idx % n);
                (0..k).map(|j| a[r * k + j] * b[j * n + c]).sum::<f32>() + bias[c]
            })
            .collect();
        let tiles = m.div_ceil(2 * TS) * n.div_ceil(TS);
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_f32(&bias),
            KernelBuffer::read_only_u32(&[m as u32, k as u32, n as u32, 0]),
            KernelBuffer::write_f32(m * n),
        ];
        ctx.dispatch(
            "tgcb",
            &compiled_kernel,
            [64, 1, 1],
            [(tiles * 64) as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch imported tiled_gemm_coarsened_bias");
        let got = bufs[4].as_f32();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-3,
                "tiled_gemm_coarsened_bias {m}x{k}x{n} elem {i}: {g} vs CPU ref {w}"
            );
        }
    }
    eprintln!("imported coarsened tiled GEMM + bias matches CPU A@B + bias");
}

#[test]
fn imported_tiled_gemm_batched_matches_reference() {
    // card 044: the batched-weight tiled GEMM (A[E,M,K] @ B[E,K,N], per-expert/per-head weight): the
    // coarsened kernel plus a batch decode and per-batch A/B/C bases. The Body the planner swaps in for
    // is_batched_tiled_gemm. Verified vs a CPU per-batch `A@B` reference, with ragged shapes so the masking
    // and the batch/tile decode are both exercised.
    const TS: usize = 8;
    let kernel = manifest_path("kernels/contraction/tiled_gemm_batched.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "tiled_gemm_batched");
    assert_eq!(params, 4, "tiled_gemm_batched: a, b, dims, out");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    for (e, m, k, n) in [
        (2usize, 16usize, 8usize, 8usize),
        (3, 17, 11, 9),
        (4, 7, 5, 9),
        (2, 33, 10, 20),
    ] {
        let a: Vec<f32> = (0..e * m * k)
            .map(|i| ((i % 17) as f32) * 0.1 - 0.7)
            .collect();
        let b: Vec<f32> = (0..e * k * n)
            .map(|i| ((i % 11) as f32) * 0.07 - 0.3)
            .collect();
        // per-batch CPU reference: out[bt, r, c] = sum_j a[bt,r,j] * b[bt,j,c].
        let want: Vec<f32> = (0..e * m * n)
            .map(|idx| {
                let bt = idx / (m * n);
                let rem = idx % (m * n);
                let (r, c) = (rem / n, rem % n);
                (0..k)
                    .map(|j| a[bt * m * k + r * k + j] * b[bt * k * n + j * n + c])
                    .sum()
            })
            .collect();
        let tiles = e * m.div_ceil(2 * TS) * n.div_ceil(TS);
        let mut bufs = [
            KernelBuffer::read_only_f32(&a),
            KernelBuffer::read_only_f32(&b),
            KernelBuffer::read_only_u32(&[e as u32, m as u32, k as u32, n as u32]),
            KernelBuffer::write_f32(e * m * n),
        ];
        ctx.dispatch(
            "tgb",
            &compiled_kernel,
            [64, 1, 1],
            [(tiles * 64) as u32, 1, 1],
            &mut bufs,
        )
        .expect("dispatch imported tiled_gemm_batched");
        let got = bufs[3].as_f32();
        for (i, (gv, wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (gv - wv).abs() < 1e-3,
                "tiled_gemm_batched E{e} {m}x{k}x{n} elem {i}: {gv} vs CPU ref {wv}"
            );
        }
    }
    eprintln!("imported batched-weight tiled GEMM (per-batch bases) matches CPU per-expert A@B");
}

#[test]
fn matmul_kernel_imported_from_mir_runs_on_gpu() {
    // The core GEMM primitive (kernelgen::matmul_batched's shape) in ordinary Rust: div/rem index math + a
    // `for k in 0..K` accumulation loop. c[2,3] = a[2,4] @ b[4,3], one thread per output.
    let kernel = manifest_path("tests/kernels/matmul.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "matmul");
    assert_eq!(params, 3);

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let (m, n, k) = (2usize, 3, 4);
    let a: Vec<f32> = (0..m * k).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.5).collect();
    let mut c = vec![0.0f32; m * n];
    // card 531c: matmul's bounds-check Asserts are not provably redundant with any guard, so the
    // compiled module needs the reserved error-word binding `Context::launch` cannot supply.
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_f32(&b),
        KernelBuffer::write_f32(c.len()),
    ];
    ctx.dispatch(
        "launch",
        &compiled_kernel,
        [64, 1, 1],
        [c.len() as u32, 1, 1],
        &mut bufs,
    )
    .expect("dispatch imported matmul kernel");
    c.copy_from_slice(bufs[2].as_f32());
    // CPU reference.
    let mut want = vec![0.0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[r * k + kk] * b[kk * n + col];
            }
            want[r * n + col] = acc;
        }
    }
    assert_eq!(c, want, "imported GEMM kernel must match the CPU matmul");
    eprintln!("imported GEMM kernel (div/rem + for-loop) runs on GPU = a @ b (via launch)");
}

#[test]
fn scale_kernel_imported_from_mir_runs_on_gpu() {
    // `c[i] = a[i] * 2.0`: exercises the f32-literal constant + Mul paths beyond the add kernel.
    let kernel = manifest_path("tests/kernels/scale.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "scale");
    assert_eq!(params, 2, "scale has 2 slice params");

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::write_f32(a.len()),
    ];
    ctx.dispatch(
        "test",
        &compiled_kernel,
        [64, 1, 1],
        [a.len() as u32, 1, 1],
        &mut bufs,
    )
    .expect("dispatch imported scale kernel");
    assert_eq!(
        bufs[1].as_f32(),
        &[2.0, 4.0, 6.0, 8.0, 10.0],
        "imported-from-MIR scale kernel must compute c = a * 2.0"
    );
    eprintln!("imported scale kernel (f32 literal + Mul) runs on GPU = a*2");
}

#[test]
fn nested_selection_emits_valid_spirv() {
    // Regression guard for the control-flow structurization pass (card 100 / spec 058). A selection nested
    // inside the `i < out.len()` bounds guard used to make llc's SPIR-V backend emit invalid SPIR-V ("merge
    // block not structurally dominated") and SIGSEGV RADV, so kernels needed branchless masking. With
    // structurization (each selection gets a dedicated merge block) the nested `if` validates. Requires llc
    // (nix develop) + spirv-val.
    let kernel = manifest_path("tests/kernels/nested_sel.rs");
    let (_spv, params, _needs_error_word) = pootc_compile(kernel, "nested_sel");
    assert_eq!(params, 3, "nested_sel: a, sel, out");
    if Command::new("spirv-val").arg("--version").output().is_err() {
        eprintln!("spirv-val not on PATH; skipping validation");
        return;
    }
    let spv_path = std::env::temp_dir()
        .join("pootc-nested_sel")
        .join("nested_sel.spv");
    let v = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&spv_path)
        .output()
        .expect("run spirv-val");
    assert!(
        v.status.success(),
        "nested-selection SPIR-V must validate after structurization:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
}

#[test]
fn barrier_in_branchy_loop_emits_valid_spirv() {
    // Regression guard for card 100 SC-002: a bounds-guard branch inside a loop that holds a workgroup
    // barrier used to emit invalid SPIR-V and SIGSEGV/hang RADV. Two fixes make it valid: (a) the
    // structurization pass gives the in-loop selection a dedicated merge block, and (b) the SPIR-V
    // barrier-semantics post-process rewrites OpControlBarrier from SequentiallyConsistent (Vulkan-illegal) to
    // AcquireRelease|WorkgroupMemory. Requires llc (nix develop) + spirv-val.
    let kernel = manifest_path("tests/kernels/barrier_branch_loop.rs");
    let (_spv, params, _needs_error_word) = pootc_compile(kernel, "barrier_branch_loop");
    assert_eq!(params, 3, "barrier_branch_loop: a, dims, out");
    if Command::new("spirv-val").arg("--version").output().is_err() {
        eprintln!("spirv-val not on PATH; skipping validation");
        return;
    }
    let spv_path = std::env::temp_dir()
        .join("pootc-barrier_branch_loop")
        .join("barrier_branch_loop.spv");
    let v = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&spv_path)
        .output()
        .expect("run spirv-val");
    assert!(
        v.status.success(),
        "barrier-in-branchy-loop SPIR-V must validate:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
}

#[test]
fn workgroup_barrier_emits_valid_vulkan_spirv() {
    // The SPIR-V barrier-semantics post-process: poot's barrier intrinsic lowers to OpControlBarrier with
    // SequentiallyConsistent semantics, which Vulkan forbids; the post-process rewrites it to
    // AcquireRelease|WorkgroupMemory. Guards that a plain barrier kernel (wg_reduce) passes spirv-val.
    let kernel = manifest_path("tests/kernels/wg_reduce.rs");
    let (_spv, _params, _needs_error_word) = pootc_compile(kernel, "wg_reduce");
    if Command::new("spirv-val").arg("--version").output().is_err() {
        eprintln!("spirv-val not on PATH; skipping validation");
        return;
    }
    let spv_path = std::env::temp_dir()
        .join("pootc-wg_reduce")
        .join("wg_reduce.spv");
    let v = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3"])
        .arg(&spv_path)
        .output()
        .expect("run spirv-val");
    assert!(
        v.status.success(),
        "barrier kernel SPIR-V must validate after the semantics post-process:\n{}",
        String::from_utf8_lossy(&v.stderr)
    );
}

#[test]
fn barrier_in_branchy_loop_runs_on_gpu() {
    // The runtime half of card 100 SC-002: dispatch a kernel with a bounds-guard branch inside a barrier'd
    // loop on RADV and check it against a CPU reference (it used to emit invalid SPIR-V and SIGSEGV the
    // driver). out[lane] = sum over chunks k of (a[k*W + lane] if k*W+lane < len else 0).
    use poot_codegen::Target;
    let _ = Target::SpirvVulkan;

    let kernel = manifest_path("tests/kernels/barrier_branch_loop.rs");
    let (compiled_kernel, params, _needs_error_word) = pootc_compile(kernel, "barrier_branch_loop");
    assert_eq!(params, 3);

    const W: usize = 64;
    let chunks = 4usize;
    let len = 200u32; // last chunk (idx 192..255) is partly out of bounds -> exercises the branch
    let a: Vec<f32> = (0..(chunks * W)).map(|i| i as f32).collect();
    let dims: Vec<u32> = vec![len, chunks as u32];

    // CPU reference.
    let mut expected = vec![0.0f32; W];
    for (lane, e) in expected.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for k in 0..chunks {
            let idx = k * W + lane;
            if (idx as u32) < len {
                acc += a[idx];
            }
        }
        *e = acc;
    }

    let _gpu_guard = gpu_lock();
    let ctx = match Context::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}); skipping the dispatch");
            return;
        }
    };
    let mut bufs = [
        KernelBuffer::read_only_f32(&a),
        KernelBuffer::read_only_u32(&dims),
        KernelBuffer::write_f32(W),
    ];
    ctx.dispatch(
        "barrier_branch_loop",
        &compiled_kernel,
        [W as u32, 1, 1],
        [W as u32, 1, 1],
        &mut bufs,
    )
    .expect("dispatch barrier-in-branchy-loop");
    assert_eq!(
        bufs[2].as_f32(),
        expected.as_slice(),
        "barrier-in-branchy-loop GPU output must match the CPU reference"
    );
    eprintln!("barrier-in-branchy-loop runs on RADV = matches CPU (SC-002 runtime confirmed)");
}

/// Check Peano is present and return the path; else None (the test should skip).
fn peano_llc_or_skip() -> Option<String> {
    match std::env::var("POOT_AIE_LLC") {
        Ok(p)
            if std::process::Command::new(&p)
                .arg("--version")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains("aie2p"))
                .unwrap_or(false) =>
        {
            Some(p)
        }
        _ => {
            eprintln!(
                "POOT_AIE_LLC (Peano llc with aie2p) not set/runnable; skipping AIE import-lower test"
            );
            None
        }
    }
}

/// Import `kernel_file` via pootc, lower the Body to AIE2p via Peano, and write the .o to
/// `~/.cache/pootc-test/aie/{obj_name}`. Returns the ELF bytes. Panics on any failure.
fn lower_kernel_to_aie2p(kernel_file: &str, src_name: &str, obj_name: &str) -> Vec<u8> {
    use poot_codegen::{Target, compile};

    let (_spv, _pc, _needs_error_word) = pootc_compile(kernel_file, src_name);
    let out_dir = std::env::temp_dir().join(format!("pootc-{src_name}"));
    let body: Body = serde_json::from_str(
        &std::fs::read_to_string(out_dir.join(format!("{src_name}.kir.json")))
            .unwrap_or_else(|_| panic!("{src_name}.kir.json missing")),
    )
    .expect("imported Body parses");

    // Sequential kernels must not have a dispatch-id injected.
    assert!(
        !format!("{body:?}").contains("ThreadIndexCall"),
        "sequential kernel `{src_name}` should import without a ThreadIndexCall"
    );

    // Peano runs under steam-run's FHS which only binds $HOME; write artifacts there.
    let aie_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/pootc-test/aie");
    std::fs::create_dir_all(&aie_dir).unwrap();
    let obj = aie_dir.join(obj_name);
    let ir = compile(&body, Target::AieCore, &obj)
        .unwrap_or_else(|e| panic!("AIE-core compile of `{src_name}` failed: {e}"));
    let bytes = std::fs::read(&obj).unwrap();
    assert_eq!(
        &bytes[0..4],
        b"\x7fELF",
        "not an ELF object for {src_name}\n{ir}"
    );
    assert!(
        bytes
            .windows(body.name.len())
            .any(|w| w == body.name.as_bytes()),
        "kernel symbol `{}` not found in the AIE object for {src_name}",
        body.name
    );
    eprintln!(
        "imported-from-MIR sequential kernel `{}` lowered to AIE2p ({} bytes) -> {}",
        body.name,
        bytes.len(),
        obj.display()
    );
    bytes
}

/// For the AIE-core (XDNA2 NPU) target: a sequential Rust `#[kernel]` (`vadd_seq.rs`, a
/// `while i < len` loop, no `thread_index()`) is MIR-imported by pootc to a `Body`, then lowered through
/// poot's emitter to AIE2p machine code by Peano (`llc --march=aie2p`). Unlike the codegen-side tests
/// (hand-built fixture Bodies), this lowers a real rustc-MIR Body (card 089 / spec 057 FR-002).
///
/// Peano is a separate llc fork (`$POOT_AIE_LLC`, an FHS wrapper on NixOS); skips when absent. The on-NPU
/// dispatch (XRT + IRON) is a separate gated step.
#[test]
fn imported_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/vadd_seq.rs");
    lower_kernel_to_aie2p(kernel, "vadd_seq", "vadd_seq.o");
}

/// vmul_seq (c[i] = a[i] * b[i]) lowers to AIE2p via Peano. Produces
/// ~/.cache/pootc-test/aie/vmul_seq.o, input to an IRON vmul xclbin build.
#[test]
fn imported_vmul_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/vmul_seq.rs");
    lower_kernel_to_aie2p(kernel, "vmul_seq", "vmul_seq.o");
}

/// vsub_seq (c[i] = a[i] - b[i]) lowers to AIE2p via Peano. Produces
/// ~/.cache/pootc-test/aie/vsub_seq.o, input to an IRON vsub xclbin build.
#[test]
fn imported_vsub_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/vsub_seq.rs");
    lower_kernel_to_aie2p(kernel, "vsub_seq", "vsub_seq.o");
}

/// Fused 2-op AIE kernel: d[i] = (a[i] + b[i]) * e[i] via a single 3-input sequential loop. Produces
/// ~/.cache/pootc-test/aie/vadd_then_vmul_seq.o for a fused 2-op xclbin build
/// (a 4-ObjectFifo IRON design).
#[test]
fn imported_vadd_then_vmul_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/vadd_then_vmul_seq.rs");
    lower_kernel_to_aie2p(kernel, "vadd_then_vmul_seq", "vadd_then_vmul_seq.o");
}

/// Sequential scalar gemv kernel for the AIE-core: y[m] = sum_k A[m*K+k]*x[k], A flat row-major [M*K],
/// K = x.len(), M = y.len(). Nested while-loop body, no thread_index(). Produces
/// ~/.cache/pootc-test/aie/gemv_seq.o for an IRON gemv xclbin build.
#[test]
fn imported_gemv_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/gemv_seq.rs");
    lower_kernel_to_aie2p(kernel, "gemv_seq", "gemv_seq.o");
}

/// Sequential scalar gemm kernel for the AIE-core: C[m,n] = sum_k A[m*K+k]*B[k*N+n], A [M,K], B [K,N],
/// C [M,N], all row-major.
///
/// Length encoding: the IRON caller passes N (not M*N) as the `c` slice length; the kernel recovers N from
/// c.len(), K = b.len()/N, M = a.len()/K. This keeps A+B as the only two S2MM (host->tile) input channels,
/// matching gemv.
///
/// Produces ~/.cache/pootc-test/aie/gemm_seq.o for an IRON gemm xclbin build.
#[test]
fn imported_gemm_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/gemm_seq.rs");
    lower_kernel_to_aie2p(kernel, "gemm_seq", "gemm_seq.o");
}

/// Unary activation kernel for the AIE-core: y[i] = max(x[i], 0.0) (ReLU). No transcendentals (a
/// conditional move); always links under Peano. Produces ~/.cache/pootc-test/aie/vrelu_seq.o for relu
/// xclbins (an IRON unary-activation build, ReLU).
#[test]
fn imported_vrelu_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/vrelu_seq.rs");
    lower_kernel_to_aie2p(kernel, "vrelu_seq", "vrelu_seq.o");
}

/// Unary activation kernel for the AIE-core: y[i] = x[i] / (1 + exp(-x[i])) (SiLU). Requires __expf from
/// Peano's math library at link time; if the aie2p libm lacks expf this test fails with a link error, in
/// which case only vrelu is usable and silu waits on a math-lib follow-up. Produces
/// ~/.cache/pootc-test/aie/vsilu_seq.o for silu xclbins
/// (an IRON unary-activation build, SiLU).
#[test]
fn imported_vsilu_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/vsilu_seq.rs");
    lower_kernel_to_aie2p(kernel, "vsilu_seq", "vsilu_seq.o");
}

/// Numerically-stable softmax kernel for the AIE-core (XDNA2 NPU): 3-pass sequential (find max, sum
/// exp(x-max), normalize). Requires exp via Peano's math library at link time. Produces
/// ~/.cache/pootc-test/aie/vsoftmax_seq.o for softmax xclbins
/// (an IRON unary-activation build, softmax).
#[test]
fn imported_vsoftmax_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/softmax_seq.rs");
    lower_kernel_to_aie2p(kernel, "vsoftmax_seq", "vsoftmax_seq.o");
}

/// Sequential scalar gemm-NT (B transposed) kernel for the AIE-core: C[i,j] = sum_d A[i*D+d] * B[j*D+d]
/// (= A @ B^T without data movement), A[L,D] and B[L,D] row-major, C[L,L]. Length encoding: IRON passes L
/// (not L*L) as c's length; the kernel recovers L=c.len(), D=a.len()/L. No transcendentals. Produces
/// ~/.cache/pootc-test/aie/gemm_nt_seq.o for an IRON attention xclbin build.
#[test]
fn imported_gemm_nt_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/gemm_nt_seq.rs");
    lower_kernel_to_aie2p(kernel, "gemm_nt_seq", "gemm_nt_seq.o");
}

/// Sequential row-wise softmax kernel for the AIE-core: numerically-stable softmax of each row of a [L,L]
/// score matrix. Length encoding: IRON passes L (not L*L) as y's length; the kernel recovers L=y.len().
/// Requires exp via Peano's math library at link time (as softmax_seq / vsilu_seq). Produces
/// ~/.cache/pootc-test/aie/softmax_row_seq.o for an IRON attention xclbin build.
#[test]
fn imported_softmax_row_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/softmax_row_seq.rs");
    lower_kernel_to_aie2p(kernel, "softmax_row_seq", "softmax_row_seq.o");
}

/// Sequential 2-input RMSNorm kernel for the AIE-core (XDNA2 NPU): y[i] = x[i] * rsqrt(mean(x^2) + eps) *
/// w[i], eps=1e-6. Two passes: sum squares, compute rsqrt(ss/N+eps), scale + weight. 2 inputs (x[N], w[N]),
/// 1 output (y[N]): fits the 2-S2MM IRON binary design. Produces ~/.cache/pootc-test/aie/rmsnorm_seq.o for
/// an IRON rmsnorm xclbin build.
#[test]
fn imported_rmsnorm_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/rmsnorm_seq.rs");
    lower_kernel_to_aie2p(kernel, "rmsnorm_seq", "rmsnorm_seq.o");
}

/// Sequential half-split RoPE kernel for the AIE-core (XDNA2 NPU): y[i] = x[i]*cos[i] - x[i+half]*sin[i]
/// (and symmetric for y[i+half]). 2 inputs (x[D], cs[2D] = [cos, sin] concatenated), 1 output (y[D]). No
/// transcendentals; stack_size=2048. Produces ~/.cache/pootc-test/aie/rope_seq.o for an IRON rope
/// xclbin build.
#[test]
fn imported_rope_sequential_kernel_lowers_to_aie2p() {
    if peano_llc_or_skip().is_none() {
        return;
    }
    let kernel = manifest_path("tests/kernels/rope_seq.rs");
    lower_kernel_to_aie2p(kernel, "rope_seq", "rope_seq.o");
}

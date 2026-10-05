//! Card 408's downstream API boundary.
//!
//! Rustdoc supplies the facade inventory. The privacy rows use standalone Cargo fixtures whose path
//! dependency is this checkout, so Cargo selects and builds the exact artifact under test.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Map, Value};

static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

struct FixtureDir(PathBuf);

impl FixtureDir {
    fn new(label: &str) -> Self {
        loop {
            let id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("poot-card408-{label}-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create Cargo fixture {}: {error}", path.display()),
            }
        }
    }
}

impl Drop for FixtureDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cargo() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

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

fn diagnostics(output: &Output) -> String {
    format!(
        "status: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn rustdoc_json() -> Value {
    let fixture = FixtureDir::new("rustdoc-inventory");
    let manifest = manifest_dir().join("Cargo.toml");
    let target = fixture.0.join("target");
    let output = Command::new(cargo())
        .args([
            "rustdoc",
            "--locked",
            "--offline",
            "--quiet",
            "--manifest-path",
        ])
        .arg(&manifest)
        .args(["--lib", "--target-dir"])
        .arg(&target)
        .args(["--", "-Z", "unstable-options", "--output-format", "json"])
        .output()
        .expect("run Cargo-managed rustdoc inventory");
    assert!(
        output.status.success(),
        "Cargo rustdoc inventory failed:\n{}",
        diagnostics(&output)
    );

    let path = target.join("doc/poot_ptx_gpu.json");
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read rustdoc JSON {}: {error}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("parse rustdoc JSON {}: {error}", path.display()))
}

fn id_key(id: &Value) -> String {
    match id {
        Value::String(id) => id.clone(),
        Value::Number(id) => id.to_string(),
        other => panic!("unexpected rustdoc item id: {other:?}"),
    }
}

fn indexed_item<'a>(index: &'a Map<String, Value>, id: &Value) -> &'a Value {
    let key = id_key(id);
    index
        .get(&key)
        .unwrap_or_else(|| panic!("rustdoc index is missing item {key}"))
}

/// The files that make up Card 408's executor facade. ADR-0099 S1 split the compute-segment,
/// per-replay-input and typed-error surfaces out of `executor.rs` into child modules it owns, so the
/// inventory follows the items, not one file. `src/multi_device.rs` is deliberately absent: the
/// schedule compiler and its `ScheduledCommunication` types live there and were never part of this
/// facade.
const EXECUTOR_FACADE_SOURCES: &[&str] = &[
    "src/multi_device/executor.rs",
    "src/multi_device/segment.rs",
    "src/multi_device/replay_inputs.rs",
    "src/multi_device/error.rs",
];

fn executor_source_item(item: &Value) -> bool {
    item.pointer("/span/filename")
        .and_then(Value::as_str)
        .is_some_and(|filename| {
            EXECUTOR_FACADE_SOURCES
                .iter()
                .any(|source| Path::new(filename).ends_with(Path::new(source)))
        })
}

fn executor_reexport_name(index: &Map<String, Value>, item: &Value) -> Option<String> {
    if executor_source_item(item) {
        return item.get("name")?.as_str().map(str::to_owned);
    }

    let import = item.pointer("/inner/use")?;
    let target = import.get("id").filter(|id| !id.is_null())?;
    let target = indexed_item(index, target);
    executor_source_item(target).then(|| {
        import
            .get("name")
            .and_then(Value::as_str)
            .or_else(|| target.get("name").and_then(Value::as_str))
            .expect("executor re-export has a name")
            .to_owned()
    })
}

fn rustdoc_executor_inventory(document: &Value) -> BTreeSet<String> {
    assert_eq!(
        document.get("includes_private").and_then(Value::as_bool),
        Some(false),
        "facade inventory must come from public-only rustdoc output"
    );
    let index = document
        .get("index")
        .and_then(Value::as_object)
        .expect("rustdoc JSON index");
    let root = indexed_item(index, document.get("root").expect("rustdoc root id"));
    let root_items = root
        .pointer("/inner/module/items")
        .and_then(Value::as_array)
        .expect("rustdoc root module items");
    let multi_device = root_items
        .iter()
        .map(|id| indexed_item(index, id))
        .find(|item| item.get("name").and_then(Value::as_str) == Some("multi_device"))
        .expect("public multi_device module in rustdoc output");
    let public_items = multi_device
        .pointer("/inner/module/items")
        .and_then(Value::as_array)
        .expect("public multi_device items in rustdoc output");

    public_items
        .iter()
        .map(|id| indexed_item(index, id))
        .filter_map(|item| executor_reexport_name(index, item))
        .collect()
}

fn write_downstream_fixture(root: &Path, package: &str, source: &str) -> PathBuf {
    let fixture = root.join(package);
    std::fs::create_dir_all(fixture.join("src"))
        .unwrap_or_else(|error| panic!("create downstream fixture {package}: {error}"));
    let crate_root = manifest_dir();
    let manifest = format!(
        "[package]\nname = {package:?}\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\n\n[dependencies]\npoot-ptx-gpu = {{ path = {crate_root:?} }}\n\n[workspace]\n"
    );
    std::fs::write(fixture.join("Cargo.toml"), manifest)
        .unwrap_or_else(|error| panic!("write downstream manifest {package}: {error}"));
    std::fs::write(fixture.join("src/main.rs"), source)
        .unwrap_or_else(|error| panic!("write downstream source {package}: {error}"));
    fixture.join("Cargo.toml")
}

fn cargo_check_fixture(manifest: &Path, target: &Path) -> Output {
    Command::new(cargo())
        .args(["check", "--offline", "--quiet", "--manifest-path"])
        .arg(manifest)
        .arg("--target-dir")
        .arg(target)
        .output()
        .expect("run Cargo-managed downstream fixture")
}

fn assert_one_private_call(label: &str, method: &str, control_source: &str, private_source: &str) {
    let fixture = FixtureDir::new(label);
    let target = fixture.0.join("target");
    let control_manifest = write_downstream_fixture(
        &fixture.0,
        &format!("card408_{label}_control"),
        control_source,
    );
    let private_manifest = write_downstream_fixture(
        &fixture.0,
        &format!("card408_{label}_private"),
        private_source,
    );

    let control = cargo_check_fixture(&control_manifest, &target);
    assert!(
        control.status.success(),
        "otherwise-valid `{method}` downstream control failed:\n{}",
        diagnostics(&control)
    );

    let private = cargo_check_fixture(&private_manifest, &target);
    let stderr = String::from_utf8_lossy(&private.stderr);
    assert!(
        !private.status.success()
            && stderr.contains(method)
            && (stderr.contains("E0624") || stderr.contains("private")),
        "the single `{method}` call must be the privacy failure; widening only that method must make \
         this Cargo fixture compile:\n{}",
        diagnostics(&private)
    );
}

/// The public facade is exactly this list, read from rustdoc's public-only JSON rather than from
/// source text. A glob re-export (`pub use executor::*`) is one `use` item that names the module, so
/// it shows up here as `executor` and fails the comparison; an added or removed item does the same.
/// Mutation observed red: the explicit re-export list becomes `pub use executor::*;`.
#[test]
fn captured_executor_facade_public_inventory_is_exact() {
    let actual = rustdoc_executor_inventory(&rustdoc_json());
    // ADR-0099 S1 grows this list by exactly the compute-segment capture entry
    // (`PtxDeviceSegment`, `PtxComputeSegment`, `SegmentProgram`, `SegmentCapture`) and the
    // capture-scoped buffer view (`CaptureBufferView`). Everything else stays frozen.
    let expected = [
        "CaptureBufferView",
        "CapturedMultiDeviceError",
        "DeviceCommunicationBuffers",
        "PtxCapturedMultiDeviceExecutor",
        "PtxCommunicationBufferBinding",
        "PtxComputeSegment",
        "PtxDeviceCapture",
        "PtxDeviceSegment",
        "PtxMultiDeviceReplayCounters",
        "PtxMultiDeviceReplayReport",
        "SegmentCapture",
        "SegmentProgram",
        "StaticBufferUpload",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();

    assert_eq!(
        actual, expected,
        "rustdoc-derived public executor facade inventory changed"
    );
}

#[test]
fn captured_move_bytes_is_not_downstream_callable() {
    let control = r#"
        use poot_ptx_gpu::p2p::PtxP2PTransport;

        fn public_control(transport: &PtxP2PTransport) {
            let _ = transport.peerable(0, 1);
        }

        fn main() {}
    "#;
    let one_forbidden_call = r#"
        use poot_ptx_gpu::p2p::PtxP2PTransport;

        fn public_control(transport: &PtxP2PTransport) {
            let _ = transport.peerable(0, 1);
        }

        fn forbidden(transport: &PtxP2PTransport) {
            transport
                .move_bytes_capturable(0, 0, 0, 0, 1, 0, 0, 0, 0)
                .unwrap();
        }

        fn main() {}
    "#;
    assert_one_private_call(
        "move_bytes",
        "move_bytes_capturable",
        control,
        one_forbidden_call,
    );
}

#[test]
fn captured_prepare_is_not_downstream_callable() {
    let control = r#"
        use poot_ptx_gpu::p2p::PtxP2PTransport;

        fn public_control(transport: &PtxP2PTransport) {
            let _ = transport.peerable(0, 1);
        }

        fn main() {}
    "#;
    let one_forbidden_call = r#"
        use poot_ptx_gpu::p2p::PtxP2PTransport;

        fn public_control(transport: &PtxP2PTransport) {
            let _ = transport.peerable(0, 1);
        }

        fn forbidden(transport: &mut PtxP2PTransport) {
            transport.prepare_capturable_peer_access(&[]).unwrap();
        }

        fn main() {}
    "#;
    assert_one_private_call(
        "prepare",
        "prepare_capturable_peer_access",
        control,
        one_forbidden_call,
    );
}

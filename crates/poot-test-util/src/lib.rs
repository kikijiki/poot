//! Test helpers that were copy-pasted across the workspace test modules: deterministic pseudo-random
//! fills and name-derived seeds, plain-Rust reference kernels, and small comparison helpers. Dev-dependency
//! only, plus the optional dependency behind `poot-graph-ir`'s and `poot-eval`'s `test-support` features.
//!
//! This is the rank-1 base: it depends on nothing above `poot-target`. A fixture that builds a graph, a packed
//! payload, a planner input or a kernel body lives in the crate that owns that type, under that crate's
//! `test-support` feature (ADR-0113): `poot_graph_ir::test_support`, `poot_graph_plan::test_support`,
//! `poot_eval::test_support`, `poot_quant::test_support`, `poot_kernel_ir::test_support`,
//! `poot_codegen::test_support` and `poot_runtime_common::device_skip`.

pub mod checkpoints;
pub mod contained_child;
pub mod device_caps;
pub mod qwen_vl;
pub mod smolvlm;

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// A path under the temp dir that no other test process or thread shares: `<temp>/poot-test-<pid>-<n>/<name>`.
///
/// Tests that write a fixture to a fixed name under the temp dir clobber each other whenever two of them run
/// at once. nextest runs every test in its own process, where a counter alone starts at 0 in each, so the
/// process id separates processes and the counter separates threads of one `cargo test` process. The unique
/// part is a parent directory, created here, so `name` (its file name, extension and any check a test makes
/// on it) is unchanged. The returned guard dereferences to the path and removes that parent directory, and
/// everything under it, when it drops: hold it for the fixture's lifetime and remove nothing by hand.
pub fn unique_temp_path(name: impl AsRef<std::ffi::OsStr>) -> UniqueTempPath {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let parent = std::env::temp_dir().join(format!("poot-test-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&parent).expect("create the unique temp parent directory");
    UniqueTempPath {
        path: parent.join(name.as_ref()),
        parent,
    }
}

/// The path [`unique_temp_path`] returns. Dropping it removes the unique parent directory.
#[derive(Debug)]
pub struct UniqueTempPath {
    path: std::path::PathBuf,
    parent: std::path::PathBuf,
}

impl std::ops::Deref for UniqueTempPath {
    type Target = std::path::Path;

    fn deref(&self) -> &std::path::Path {
        &self.path
    }
}

impl AsRef<std::path::Path> for UniqueTempPath {
    fn as_ref(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for UniqueTempPath {
    fn drop(&mut self) {
        // A fixture may already have removed its own file or directory, but the parent is ours alone.
        if let Err(e) = std::fs::remove_dir_all(&self.parent)
            && e.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!("could not remove {}: {e}", self.parent.display());
        }
    }
}

/// The environment variable naming the directory that holds the checkpoints a model test loads.
pub const MODELS_DIR_VARIABLE: &str = "POOT_MODELS_DIR";

/// The environment variable that turns an unset [`MODELS_DIR_VARIABLE`] from a reported skip into a panic.
pub const REQUIRE_MODELS_VARIABLE: &str = "POOT_REQUIRE_MODELS";

/// A checkpoint a test loads: a directory of the models directory, or a path inside one.
///
/// The only way to get one is [`checkpoint!`], which refuses at compile time a checkpoint directory that is not
/// in [`checkpoints::CHECKPOINTS`], so [`model_path`] is never asked for a name that no inventory row backs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    path: &'static str,
}

impl Checkpoint {
    /// The checkpoint at `path`, whose first component must be in [`checkpoints::CHECKPOINTS`]. Panics when it is
    /// not, which [`checkpoint!`] turns into a compile error by evaluating this in a constant.
    #[doc(hidden)]
    pub const fn new(path: &'static str) -> Self {
        assert!(
            checkpoints::is_listed(checkpoints::first_component(path)),
            "the checkpoint directory is not in poot_test_util::checkpoints::CHECKPOINTS: add it to the model \
             inventory first, then to that table"
        );
        Self { path }
    }
}

impl std::fmt::Display for Checkpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.path)
    }
}

/// A [`Checkpoint`] for a string literal: a checkpoint directory of [`checkpoints::CHECKPOINTS`], optionally
/// followed by a path inside it (`"qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"`). The first component is
/// checked during compilation, so a name that is not a model inventory row does not compile.
///
/// ```
/// let checkpoint = poot_test_util::checkpoint!("qwen2.5-0.5b");
/// assert_eq!(checkpoint.to_string(), "qwen2.5-0.5b");
/// ```
///
/// The invented name from Card 519b, which would have skipped forever, is refused:
///
/// ```compile_fail,E0080
/// let _ = poot_test_util::checkpoint!("gemma4-dense-gguf/gemma4-dense.gguf");
/// ```
#[macro_export]
macro_rules! checkpoint {
    ($path:literal) => {
        const { $crate::Checkpoint::new($path) }
    };
}

/// The path of `checkpoint` for a test that loads a model, or `None` after reporting a skip when the test cannot
/// run.
///
/// This is the one place a test finds a checkpoint: there is no default directory, so a run on any machine
/// names its models directory explicitly through `POOT_MODELS_DIR`. With the variable unset the test skips,
/// unless `POOT_REQUIRE_MODELS=1`, which panics naming the variable so a lane that must run checkpoint tests
/// cannot pass by skipping them all. A model missing from the directory skips, with its name printed, unless
/// `POOT_REQUIRE_MODELS=1` too: then it panics naming the model and the directory, so the checkpoint lane
/// (`just test-checkpoint`) fails on a missing model instead of passing on a SKIP.
///
/// ```text
/// let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b")) else {
///     return;
/// };
/// ```
pub fn model_path(checkpoint: Checkpoint) -> Option<std::path::PathBuf> {
    model_path_with(|variable| std::env::var(variable).ok(), checkpoint)
}

/// [`model_path`] with the environment injected, so a test can set one variable without touching the process.
pub fn model_path_with(
    get: impl Fn(&str) -> Option<String>,
    checkpoint: Checkpoint,
) -> Option<std::path::PathBuf> {
    let Some(models_dir) = get(MODELS_DIR_VARIABLE).filter(|dir| !dir.is_empty()) else {
        assert!(
            get(REQUIRE_MODELS_VARIABLE).as_deref() != Some("1"),
            "{MODELS_DIR_VARIABLE} is unset but {REQUIRE_MODELS_VARIABLE}=1: set {MODELS_DIR_VARIABLE} to the \
             directory that holds the checkpoints (model {checkpoint})"
        );
        eprintln!("SKIP: model {checkpoint} needs {MODELS_DIR_VARIABLE}, which is unset");
        return None;
    };
    let path = std::path::Path::new(&models_dir).join(checkpoint.path);
    if !path.exists() {
        assert!(
            get(REQUIRE_MODELS_VARIABLE).as_deref() != Some("1"),
            "model {checkpoint} not found under {MODELS_DIR_VARIABLE}={models_dir} but \
             {REQUIRE_MODELS_VARIABLE}=1: a checkpoint lane fails on a missing model"
        );
        eprintln!("SKIP: model {checkpoint} not found under {MODELS_DIR_VARIABLE}={models_dir}");
        return None;
    }
    Some(path)
}

/// `v` as little-endian bytes.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `n` deterministic pseudo-random values in `[-1, 1)` from an xorshift stream seeded by `seed`.
pub fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Reference `x @ w` for one row: `x` is `in_dim` long, `w` is `[in_dim, out_dim]` row-major.
pub fn linear_ref(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut acc = 0.0f32;
        for i in 0..in_dim {
            acc += x[i] * w[i * out_dim + o];
        }
        y[o] = acc;
    }
    y
}

/// The element pair where `actual` is furthest from `expected`, or the first pair that is a fault.
///
/// `f32::max` returns the non-NaN operand, so a fold of `(a - b).abs()` over an all-NaN output is 0 and passes
/// any tolerance (R482-002). A pair is therefore compared explicitly, in `f64` so an `f32` difference cannot
/// round: equal values (including equal infinities) have error 0, a NaN on either side or a non-finite
/// difference is a fault with an infinite error, and any other pair has the error `error_of` gives. Comparisons
/// follow ADR-0101 decision 4.
struct Worst {
    index: usize,
    actual: f64,
    expected: f64,
    error: f64,
}

impl Worst {
    fn of(
        actual: impl ExactSizeIterator<Item = f64>,
        expected: impl ExactSizeIterator<Item = f64>,
        error_of: impl Fn(f64, f64) -> f64,
    ) -> Self {
        assert_eq!(
            actual.len(),
            expected.len(),
            "actual and expected lengths differ"
        );
        let mut worst = Worst {
            index: 0,
            actual: 0.0,
            expected: 0.0,
            error: 0.0,
        };
        for (index, (a, e)) in actual.zip(expected).enumerate() {
            let error = if a == e { 0.0 } else { error_of(a, e) };
            let error = if error.is_nan() { f64::INFINITY } else { error };
            if error > worst.error {
                worst = Worst {
                    index,
                    actual: a,
                    expected: e,
                    error,
                };
            }
            if error == f64::INFINITY {
                break;
            }
        }
        worst
    }

    fn of_f32(actual: &[f32], expected: &[f32], error_of: impl Fn(f64, f64) -> f64) -> Self {
        assert!(!actual.is_empty(), "{NOTHING_TO_COMPARE}");
        Self::of(
            actual.iter().map(|&v| f64::from(v)),
            expected.iter().map(|&v| f64::from(v)),
            error_of,
        )
    }

    /// The report for a failing comparison of `f64` values.
    fn describe(&self, what: &str) -> String {
        format!(
            "{what} at index {}: actual {} vs expected {} (error {})",
            self.index, self.actual, self.expected, self.error
        )
    }

    /// The report for a failing comparison of `f32` values, printed as `f32` so a value reads as the test wrote
    /// it (`1.1`, not `1.100000023841858`).
    fn describe_f32(&self, what: &str) -> String {
        format!(
            "{what} at index {}: actual {} vs expected {} (error {})",
            self.index, self.actual as f32, self.expected as f32, self.error as f32
        )
    }
}

/// The panic of every comparison of two empty outputs, which would agree with any oracle.
const NOTHING_TO_COMPARE: &str = "nothing to compare: an empty output agrees with any oracle";

fn abs_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs()
}

/// `|actual - expected|` relative to the larger magnitude of the pair, floored at `1e-6`.
fn rel_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / actual.abs().max(expected.abs()).max(1e-6)
}

/// Asserts every element of `actual` is within `tol` (absolute) of `expected`.
///
/// A NaN, or an infinity where `expected` is not the same infinity, fails at any tolerance. The panic names
/// the worst element by index with its actual and expected values.
#[track_caller]
pub fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    let worst = Worst::of_f32(actual, expected, abs_error);
    assert!(
        worst.error <= f64::from(tol),
        "{} exceeds tolerance {tol}",
        worst.describe_f32("worst absolute error")
    );
}

/// Asserts every element of `actual` is within `tol` of `expected`, relative to the larger magnitude of the
/// pair (floored at `1e-6`, so near-zero pairs do not dominate). NaN and infinities fail as in [`assert_close`].
#[track_caller]
pub fn assert_close_rel(actual: &[f32], expected: &[f32], tol: f32) {
    let worst = Worst::of_f32(actual, expected, rel_error);
    assert!(
        worst.error <= f64::from(tol),
        "{} exceeds relative tolerance {tol}",
        worst.describe_f32("worst relative error")
    );
}

/// `|actual - expected|` relative to `max(|expected|, 1)`: absolute below 1, relative above.
fn unit_floor_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

/// `Ok` when every element of `actual` is within `tol * max(|expected|, 1)` of `expected`, else the report of the
/// worst element. The one definition of the tier-2 element bound (ADR-0101): a NaN, or an infinity where
/// `expected` is not the same infinity, is never within it. Panics on a length mismatch or two empty slices.
pub fn check_close_unit_floor(actual: &[f32], expected: &[f32], tol: f32) -> Result<(), String> {
    let worst = Worst::of_f32(actual, expected, unit_floor_error);
    if worst.error <= f64::from(tol) {
        Ok(())
    } else {
        Err(worst.describe_f32("worst error"))
    }
}

/// `|actual - expected|` relative to `1 + |expected|`.
fn scaled_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / (1.0 + expected.abs())
}

/// Asserts every element of `actual` is within `tol * (1 + |expected|)` of `expected`, prefixing the panic with
/// `label`. NaN and infinities fail as in [`assert_close`].
#[track_caller]
pub fn assert_close_scaled(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    let worst = Worst::of_f32(actual, expected, scaled_error);
    assert!(
        worst.error <= f64::from(tol),
        "{label}: {} exceeds tolerance {tol}",
        worst.describe_f32("worst scaled error")
    );
}

/// The largest `|actual - expected| / max(|expected|, 1)`, for tests that bound it against a tolerance. Panics,
/// naming the element, on a NaN or a non-finite difference, so the returned error is always finite.
#[track_caller]
pub fn max_unit_floor_error(actual: &[f32], expected: &[f32]) -> f32 {
    let worst = Worst::of_f32(actual, expected, unit_floor_error);
    assert!(
        worst.error.is_finite(),
        "{}",
        worst.describe_f32("non-finite value")
    );
    worst.error as f32
}

/// The largest absolute error between `actual` and `expected`, for tests that print it or bound it against
/// something other than a fixed tolerance. Panics, naming the element, on a NaN or a non-finite difference,
/// so the returned error is always finite.
#[track_caller]
pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    let worst = Worst::of_f32(actual, expected, abs_error);
    assert!(
        worst.error.is_finite(),
        "{}",
        worst.describe_f32("non-finite value")
    );
    worst.error as f32
}

/// [`max_abs_error`] against an `f64` reference, for tests whose ground truth is computed in double precision.
#[track_caller]
pub fn max_abs_error_f64(actual: &[f32], expected: &[f64]) -> f64 {
    assert!(!actual.is_empty(), "{NOTHING_TO_COMPARE}");
    let worst = Worst::of(
        actual.iter().map(|&v| f64::from(v)),
        expected.iter().copied(),
        abs_error,
    );
    assert!(
        worst.error.is_finite(),
        "{}",
        worst.describe("non-finite value")
    );
    worst.error
}

/// Reference RMS norm of the first `n` elements of `x` with scale `w` and epsilon `eps`.
pub fn rmsnorm_ref(x: &[f32], w: &[f32], n: usize, eps: f32) -> Vec<f32> {
    let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let den = (ms + eps).sqrt();
    (0..n).map(|i| x[i] / den * w[i]).collect()
}

/// Reference half-split rotary embedding of one `d`-wide `row` at position `pos`, from `[pos, d]` cos and
/// sin tables.
pub fn rope_ref(row: &[f32], cos: &[f32], sin: &[f32], pos: usize, d: usize) -> Vec<f32> {
    let c = &cos[pos * d..(pos + 1) * d];
    let s = &sin[pos * d..(pos + 1) * d];
    let mut out = vec![0.0f32; d];
    for i in 0..d / 2 {
        let (a, b) = (row[i], row[i + d / 2]);
        out[i] = a * c[i] - b * s[i];
        out[i + d / 2] = b * c[i + d / 2] + a * s[i + d / 2];
    }
    out
}

/// The 64-bit FNV-1a hash of `name`, the deterministic seed for a test's fill.
pub fn seed_of(name: &str) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    name.bytes().fold(OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

/// One blocking HTTP/1.1 request to `addr`: the response's status code (0 if unparsable) and body.
pub fn http_request(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> anyhow::Result<(u16, String)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp)?;
    let resp = String::from_utf8_lossy(&resp).into_owned();
    let status = resp
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let split = resp.find("\r\n\r\n").map(|i| i + 4).unwrap_or(resp.len());
    Ok((status, resp[split..].to_string()))
}

/// `s` without ANSI escape sequences.
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for c2 in chars.by_ref() {
                if c2.is_ascii_alphabetic() {
                    break;
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Polls `GET /health` on `addr` until it answers 200 or `deadline` passes; whether it answered.
pub fn wait_ready(addr: &str, deadline: Instant) -> bool {
    while Instant::now() < deadline {
        if let Ok((200, _)) = http_request(addr, "GET", "/health", None, Duration::from_secs(5)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_temp_path_guard_removes_its_parent_directory_on_drop() {
        let guard = unique_temp_path("fixture.bin");
        let path = std::path::PathBuf::from(&*guard);
        let parent = path.parent().expect("the path has a parent").to_path_buf();
        assert_eq!(path.file_name().unwrap(), "fixture.bin");
        assert!(
            parent
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("poot-test-"),
            "the parent directory is named poot-test-<pid>-<n>, got {}",
            parent.display()
        );
        std::fs::write(&path, b"payload").unwrap();
        std::fs::create_dir_all(path.with_extension("d")).unwrap();
        assert!(path.exists() && parent.is_dir());

        drop(guard);

        assert!(
            !parent.exists(),
            "dropping the guard must remove {}",
            parent.display()
        );
    }

    #[test]
    fn unique_temp_paths_differ_within_one_process() {
        let (a, b) = (unique_temp_path("x"), unique_temp_path("x"));
        assert_ne!(a.parent(), b.parent());
    }

    /// The panic message of `f`, or a failure if it did not panic.
    fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let payload = std::panic::catch_unwind(f).expect_err("the comparison must panic");
        match payload.downcast::<String>() {
            Ok(message) => *message,
            Err(payload) => (*payload
                .downcast::<&str>()
                .expect("the panic payload is a string"))
            .to_string(),
        }
    }

    /// An environment holding exactly the given variables.
    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_string())
        }
    }

    #[test]
    fn model_path_joins_the_models_directory_when_the_model_exists() {
        let models = unique_temp_path("models");
        std::fs::create_dir_all(models.join("qwen2.5-0.5b")).unwrap();
        let models_dir = models.to_str().unwrap();

        let found = model_path_with(
            env(&[("POOT_MODELS_DIR", models_dir)]),
            checkpoint!("qwen2.5-0.5b"),
        );

        assert_eq!(found, Some(models.join("qwen2.5-0.5b")));
    }

    #[test]
    fn model_path_keeps_the_path_inside_the_checkpoint_directory() {
        let models = unique_temp_path("models");
        std::fs::create_dir_all(models.join("qwen2.5-0.5b-gguf")).unwrap();
        std::fs::write(models.join("qwen2.5-0.5b-gguf/model.gguf"), b"gguf").unwrap();
        let models_dir = models.to_str().unwrap();

        let found = model_path_with(
            env(&[("POOT_MODELS_DIR", models_dir)]),
            checkpoint!("qwen2.5-0.5b-gguf/model.gguf"),
        );

        assert_eq!(found, Some(models.join("qwen2.5-0.5b-gguf/model.gguf")));
    }

    #[test]
    fn model_path_skips_when_the_models_directory_is_unset() {
        // `.` is the models directory itself: it resolves under any directory the resolver might invent.
        // No macro can name it (it is not a table entry), so the test builds the checkpoint directly.
        assert_eq!(model_path_with(env(&[]), Checkpoint { path: "." }), None);
    }

    #[test]
    fn model_path_panics_naming_the_variable_when_unset_and_models_are_required() {
        let message = panic_message(|| {
            model_path_with(
                env(&[("POOT_REQUIRE_MODELS", "1")]),
                checkpoint!("qwen2.5-0.5b"),
            );
        });
        assert!(
            message.contains("POOT_MODELS_DIR is unset")
                && message.contains("POOT_REQUIRE_MODELS=1")
                && message.contains("qwen2.5-0.5b"),
            "the panic names the variable, the requirement and the model, got: {message}"
        );
    }

    #[test]
    fn model_path_panics_naming_the_model_when_it_is_missing_and_models_are_required() {
        let models = unique_temp_path("models");
        std::fs::create_dir_all(&*models).unwrap();
        let models_dir = models.to_str().unwrap().to_string();

        let message = panic_message(|| {
            model_path_with(
                env(&[
                    ("POOT_MODELS_DIR", &models_dir),
                    ("POOT_REQUIRE_MODELS", "1"),
                ]),
                checkpoint!("qwen2.5-0.5b"),
            );
        });

        assert!(
            message.contains("qwen2.5-0.5b")
                && message.contains("POOT_REQUIRE_MODELS=1")
                && message.contains(&models_dir),
            "the panic names the model, the requirement and the directory, got: {message}"
        );
    }

    #[test]
    fn model_path_skips_a_missing_model_when_models_are_not_required() {
        let models = unique_temp_path("models");
        std::fs::create_dir_all(&*models).unwrap();
        let models_dir = models.to_str().unwrap();

        let found = model_path_with(
            env(&[("POOT_MODELS_DIR", models_dir)]),
            checkpoint!("qwen2.5-0.5b"),
        );

        assert_eq!(found, None);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn max_abs_error_rejects_an_all_nan_actual() {
        max_abs_error(&[f32::NAN; 4], &[1.0, -2.0, 3.0, 0.5]);
    }

    #[test]
    #[should_panic(expected = "worst absolute error at index 0: actual NaN")]
    fn assert_close_rejects_an_all_nan_actual_against_a_finite_expected() {
        assert_close(&[f32::NAN; 4], &[1.0, -2.0, 3.0, 0.5], 1e6);
    }

    #[test]
    #[should_panic(expected = "worst relative error at index 2: actual NaN")]
    fn assert_close_rel_rejects_a_nan_after_matching_elements() {
        assert_close_rel(&[1.0, -2.0, f32::NAN], &[1.0, -2.0, 3.0], 1e6);
    }

    #[test]
    #[should_panic(expected = "actual inf vs expected 3")]
    fn assert_close_rejects_an_infinity_against_a_finite_expected() {
        assert_close(&[1.0, f32::INFINITY], &[1.0, 3.0], f32::MAX);
    }

    #[test]
    fn assert_close_accepts_matching_infinities_and_errors_within_tolerance() {
        assert_close(
            &[1.0, f32::NEG_INFINITY, 2.5],
            &[1.001, f32::NEG_INFINITY, 2.5],
            2e-3,
        );
        assert_close_rel(&[100.0, 0.0], &[100.5, 0.0], 6e-3);
    }

    #[test]
    fn assert_close_names_the_worst_element_not_the_first() {
        let message = panic_message(|| {
            assert_close(&[1.5, 2.0, 9.0, 4.25], &[1.0, 2.0, 3.0, 4.0], 0.1);
        });
        assert!(
            message.contains("index 2: actual 9 vs expected 3 (error 6)"),
            "the worst element is index 2, got: {message}"
        );
    }

    #[test]
    fn assert_close_rel_names_the_worst_element_not_the_first() {
        let message = panic_message(|| {
            assert_close_rel(&[1.1, 2.0, 30.0, 4.0], &[1.0, 2.0, 3.0, 4.0], 0.01);
        });
        assert!(
            message.contains("index 2: actual 30 vs expected 3"),
            "the worst element is index 2, got: {message}"
        );
    }

    #[test]
    fn seed_of_is_the_fnv1a_64_hash_of_the_name() {
        // Published FNV-1a 64-bit test vectors.
        assert_eq!(seed_of(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(seed_of("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(seed_of("foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn max_abs_error_returns_the_largest_finite_error() {
        assert_eq!(max_abs_error(&[1.0, 5.0, 2.0], &[1.5, 2.0, 2.0]), 3.0);
    }

    #[test]
    fn max_abs_error_f64_measures_against_a_double_precision_reference() {
        let error = max_abs_error_f64(&[1.0, 2.5], &[1.0, 2.5 + 1e-9]);
        assert!((error - 1e-9).abs() < 1e-12, "got {error}");
    }

    #[test]
    #[should_panic(expected = "non-finite value at index 1: actual NaN")]
    fn max_abs_error_f64_rejects_a_nan_actual() {
        max_abs_error_f64(&[1.0, f32::NAN], &[1.0, 2.0]);
    }

    /// `fold(<seed>, f32::max)`, assembled so this file does not itself contain the pattern it forbids.
    fn fold_with_max(seed: &str) -> String {
        format!("fold({seed}, {}::max)", "f32")
    }

    /// The 1-based lines of `source` that close a zero-seeded `f32::max` fold whose preceding lines take an
    /// `.abs()`. `f32::max` returns the non-NaN operand, so such a fold over an all-NaN output is 0 and passes
    /// any tolerance (R2-L05-002).
    fn nan_blind_folds(source: &str) -> Vec<usize> {
        let closers = ["0.0", "0.0f32"].map(fold_with_max);
        let lines: Vec<&str> = source.lines().collect();
        (0..lines.len())
            .filter(|&i| {
                closers
                    .iter()
                    .any(|closer| lines[i].contains(closer.as_str()))
            })
            .filter(|&i| {
                lines[i.saturating_sub(8)..=i]
                    .iter()
                    .any(|l| l.contains(".abs()"))
            })
            .map(|i| i + 1)
            .collect()
    }

    #[test]
    fn nan_blind_folds_finds_a_fold_over_a_difference_and_nothing_else() {
        let difference = format!(
            "a.iter()\n    .map(|(x, y)| (x - y).abs())\n    .{};",
            fold_with_max("0.0f32")
        );
        assert_eq!(nan_blind_folds(&difference), [3]);
        let relative = format!(
            "v.map(|(x, y)| (x - y).abs() / y.abs().max(1.0))\n    .{}",
            fold_with_max("0.0")
        );
        assert_eq!(nan_blind_folds(&relative), [2]);
        let row_max = format!(
            "let top = scores.iter().copied().{};",
            fold_with_max("f32::NEG_INFINITY")
        );
        assert!(
            nan_blind_folds(&row_max).is_empty(),
            "a row maximum is not a difference"
        );
    }

    /// No file in `crates/` folds `|a - b|` with `f32::max` (R2-L05-002): use `poot_test_util::max_abs_error`,
    /// `max_unit_floor_error` or `assert_close`, which fail on a NaN.
    ///
    /// Mutation: add a zero-seeded `f32::max` fold over `(a - b).abs()` to any `.rs` file under `crates/`; this
    /// goes red naming the file and line.
    #[test]
    fn no_test_folds_a_difference_with_f32_max() {
        fn rust_files(dir: &std::path::Path, found: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    if path.file_name().is_some_and(|name| name != "target") {
                        rust_files(&path, found);
                    }
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    found.push(path);
                }
            }
        }
        let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut files = Vec::new();
        rust_files(&crates, &mut files);
        assert!(
            files.len() > 100,
            "the scan found only {} files",
            files.len()
        );
        let mut offenders = Vec::new();
        for file in files {
            let Ok(source) = std::fs::read_to_string(&file) else {
                continue;
            };
            for line in nan_blind_folds(&source) {
                offenders.push(format!("{}:{line}", file.display()));
            }
        }
        assert!(
            offenders.is_empty(),
            "a NaN-blind fold of a difference (use poot_test_util::max_abs_error): {offenders:#?}"
        );
    }

    #[test]
    #[should_panic(expected = "nan: worst scaled error at index 1: actual NaN")]
    fn assert_close_scaled_rejects_a_nan_and_names_the_label() {
        assert_close_scaled(&[1.0, f32::NAN], &[1.0, 1.0], 1e-3, "nan");
    }

    #[test]
    fn assert_close_scaled_bounds_the_error_by_one_plus_the_expected_magnitude() {
        assert_close_scaled(&[11.0], &[10.0], 0.1, "inside: 1 <= 0.1 * 11");
        let message = panic_message(|| assert_close_scaled(&[11.2], &[10.0], 0.1, "outside"));
        assert!(message.contains("outside: worst scaled error"), "{message}");
    }

    #[test]
    #[should_panic(expected = "nothing to compare")]
    fn assert_close_scaled_rejects_two_empty_outputs() {
        assert_close_scaled(&[], &[], 1.0, "empty");
    }

    #[test]
    fn check_close_unit_floor_refuses_nan_and_off_tolerance_and_accepts_within() {
        check_close_unit_floor(&[1.0, 10.5], &[1.0, 10.0], 0.05).expect("0.5 <= 0.05 * 10");
        let err =
            check_close_unit_floor(&[1.0, 11.0], &[1.0, 10.0], 0.05).expect_err("1.0 > 0.05 * 10");
        assert!(err.contains("index 1"), "{err}");
        let err =
            check_close_unit_floor(&[f32::NAN], &[1.0], f32::MAX).expect_err("NaN never agrees");
        assert!(err.contains("actual NaN"), "{err}");
    }

    #[test]
    #[should_panic(expected = "nothing to compare")]
    fn assert_close_rejects_two_empty_outputs() {
        assert_close(&[], &[], 1.0);
    }

    #[test]
    #[should_panic(expected = "non-finite value at index 1: actual NaN")]
    fn max_unit_floor_error_rejects_a_nan_actual() {
        max_unit_floor_error(&[1.0, f32::NAN], &[1.0, 2.0]);
    }

    #[test]
    fn max_unit_floor_error_is_absolute_below_one_and_relative_above() {
        assert_eq!(max_unit_floor_error(&[0.5, 10.0], &[0.0, 12.0]), 0.5);
        assert_eq!(max_unit_floor_error(&[0.5, 10.0], &[0.25, 20.0]), 0.5);
    }

    #[test]
    #[should_panic(expected = "lengths differ")]
    fn assert_close_rejects_mismatched_lengths() {
        assert_close(&[1.0], &[1.0, 2.0], 1.0);
    }
}

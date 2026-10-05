//! Shared state-directory and process-log management.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};

use crate::clock::{now_iso, now_local_hms};

/// File that `log()` tees into, so `serve` and `logs` can read a run's or build's progress.
static LOG_SINK: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();

/// Machine-wide orchestrator state directory from explicit env values: `$XDG_STATE_HOME/poot-orchestrator`,
/// else `$HOME/.local/state/poot-orchestrator`, else `/tmp/poot-orchestrator`. An empty or non-absolute
/// `XDG_STATE_HOME` is ignored per the XDG spec.
///
/// Deliberately not [`repo_root`]: that resolves via the compile-time `CARGO_MANIFEST_DIR`, giving each
/// git worktree its own state DB. `reap` compares the cloud pod list against this DB, so a split DB hid
/// pods from other worktrees' reap and leaked one for ~8.5h.
pub(crate) fn state_dir_from(xdg_state_home: Option<&str>, home: Option<&str>) -> PathBuf {
    fn abs(v: Option<&str>) -> Option<&str> {
        v.map(str::trim)
            .filter(|v| !v.is_empty() && Path::new(v).is_absolute())
    }
    if let Some(xdg) = abs(xdg_state_home) {
        return Path::new(xdg).join("poot-orchestrator");
    }
    if let Some(home) = abs(home) {
        return Path::new(home)
            .join(".local")
            .join("state")
            .join("poot-orchestrator");
    }
    PathBuf::from("/tmp").join("poot-orchestrator")
}

/// [`state_dir_from`] applied to this process's environment, created if missing.
pub(crate) fn state_dir() -> PathBuf {
    let xdg = std::env::var("XDG_STATE_HOME").ok();
    let home = std::env::var("HOME").ok();
    let d = state_dir_from(xdg.as_deref(), home.as_deref());
    std::fs::create_dir_all(&d).ok();
    d
}

/// `<state_dir>/logs/`: per-run and per-build progress logs (read by the dashboard and `logs`).
pub(crate) fn logs_dir() -> PathBuf {
    let d = state_dir().join("logs");
    std::fs::create_dir_all(&d).ok();
    d
}

/// Tee subsequent `log()` output into `path` (append) on top of stdout.
pub(crate) fn set_log_sink(path: &Path) {
    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok();
    if let Ok(mut g) = LOG_SINK.get_or_init(|| Mutex::new(None)).lock() {
        *g = f;
    }
}

pub(crate) fn log(msg: impl AsRef<str>) {
    let line = format!("[{}] {}", now_local_hms(), msg.as_ref());
    println!("{line}");
    if let Some(cell) = LOG_SINK.get()
        && let Ok(mut g) = cell.lock()
        && let Some(f) = g.as_mut()
    {
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }
}

/// Run `cmd` via `sh -c`, merging stderr into stdout and streaming each line through `log()`. Returns
/// the exit code. Used for the podman build/push.
pub(crate) fn run_streaming(cmd: &str) -> Result<i32> {
    let mut child = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("{cmd} 2>&1"))
        .stdout(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("spawn: {cmd}"))?;
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines() {
            match line {
                Ok(l) => log(l),
                Err(_) => break,
            }
        }
    }
    Ok(child.wait()?.code().unwrap_or(-1))
}

/// A log name is safe iff it is a bare filename (no path separators or parent refs), so requests only
/// reach files inside `logs_dir()`.
pub(crate) fn safe_log_name(name: &str) -> bool {
    !name.is_empty() && !name.contains('/') && !name.contains('\\') && !name.contains("..")
}

/// The last `max_lines` of a log file under `logs_dir()`. `name` is a bare filename (no path parts).
pub(crate) fn read_log_tail(name: &str, max_lines: usize) -> Option<String> {
    if !safe_log_name(name) {
        return None;
    }
    let content = std::fs::read_to_string(logs_dir().join(name)).ok()?;
    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    Some(lines[start..].join("\n"))
}

/// Kind and apparent state of a log, derived from its name and tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogKind {
    /// a sweep run log (`run-*.log`).
    Run,
    /// a container-image build log (`image-*.log`).
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LogState {
    Running,
    Done,
    Failed,
}

impl LogKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            LogKind::Run => "run",
            LogKind::Image => "image",
        }
    }
}

impl LogState {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            LogState::Running => "running",
            LogState::Done => "done",
            LogState::Failed => "failed",
        }
    }
}

/// `image-*.log` is a build; anything else is a sweep run.
pub(crate) fn classify_log_kind(name: &str) -> LogKind {
    if name.starts_with("image-") {
        LogKind::Image
    } else {
        LogKind::Run
    }
}

/// Infer from the log text whether the process is running, done, or failed. A non-zero `*_EXIT=` marker
/// (`SWEEP_EXIT=`, `BUILD_EXIT=`, `EXEC_EXIT=`) is a failure; a completion phrase (`build complete`,
/// `push complete`, ...) with no failure is done; otherwise running.
pub(crate) fn classify_log_state(text: &str) -> LogState {
    let mut state = LogState::Running;
    for raw in text.lines() {
        let line = raw.trim();
        // Exit markers are the strongest signal. The code can sit anywhere on the line, so read the
        // digits after the marker: 0 is clean, anything else is a failure.
        if let Some(state2) = exit_marker_state(line, "SWEEP_EXIT=") {
            state = state2;
            continue;
        }
        if let Some(state2) = exit_marker_state(line, "BUILD_EXIT=") {
            state = state2;
            continue;
        }
        if let Some(state2) = exit_marker_state(line, "EXEC_EXIT=") {
            state = state2;
            continue;
        }
        if line.contains("run failed")
            || line.contains("podman build failed")
            || line.contains("podman push failed")
            || line.contains("image smoke failed")
            || line.contains("sweep FAILED")
        {
            state = LogState::Failed;
            continue;
        }
        // Completion phrases logged at the end of a clean build/push/sweep.
        if line.contains("build complete")
            || line.contains("push complete")
            || line.ends_with("DONE")
            || line.contains("sweep finished")
        {
            state = LogState::Done;
        }
    }
    state
}

/// Done for exit code 0, Failed for any other, None if the marker is absent or has no code.
fn exit_marker_state(line: &str, marker: &str) -> Option<LogState> {
    let after = line.split(marker).nth(1)?.trim_start();
    let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    Some(if digits.trim_start_matches('0').is_empty() {
        LogState::Done
    } else {
        LogState::Failed
    })
}

/// (name, size_bytes, mtime_secs) for each log file, newest first.
pub(crate) fn list_logs() -> Vec<(String, u64, u64)> {
    let mut v: Vec<(String, u64, u64)> = std::fs::read_dir(logs_dir())
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !name.ends_with(".log") {
                return None;
            }
            let md = e.metadata().ok()?;
            let mtime = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            Some((name, md.len(), mtime))
        })
        .collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.2));
    v
}

/// Kind ("run"/"image") and state ("running"/"done"/"failed") of a log from its last lines. None if
/// missing or unsafe.
pub(crate) fn log_kind_and_state(name: &str) -> Option<(&'static str, &'static str)> {
    let tail = read_log_tail(name, 200)?;
    Some((
        classify_log_kind(name).as_str(),
        classify_log_state(&tail).as_str(),
    ))
}

/// Spawn a detached image build by re-invoking this binary's `image` subcommand on its own thread.
/// Logs to `image-<tag>.log` and appends `BUILD_EXIT=<code>`. Returns the log file name.
pub(crate) fn spawn_image_build(dockerfile: &str, tag: &str, push: bool) -> Result<String> {
    let exe = std::env::current_exe().context("locate own binary")?;
    let logname = format!("image-{tag}.log");
    let logpath = logs_dir().join(&logname);
    // Mark the start so a re-trigger does not show a stale prior result.
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&logpath)
    {
        let _ = writeln!(
            f,
            "[{}] === build triggered from dashboard (dockerfile={dockerfile}, tag={tag}, push={push}) ===",
            &now_iso()[11..19]
        );
    }
    let (dockerfile, tag) = (dockerfile.to_string(), tag.to_string());
    std::thread::spawn(move || {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("image")
            .arg("--dockerfile")
            .arg(&dockerfile)
            .arg("--tag")
            .arg(&tag);
        if push {
            cmd.arg("--push");
        }
        let code = cmd.status().ok().and_then(|s| s.code()).unwrap_or(-1);
        // Exit marker for the log classifier.
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&logpath)
        {
            let _ = writeln!(f, "[{}] BUILD_EXIT={code}", &now_iso()[11..19]);
        }
    });
    Ok(logname)
}

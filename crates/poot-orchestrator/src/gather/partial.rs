//! Live results: while a set is being swept, its rows so far are mirrored off the pod into
//! `results/<committed name>.partial/` so the dashboard (which only reads the DB and the results directory) can show
//! the cells as they finish. The harness appends one row per finished cell to the pod's `results.jsonl`.
//!
//! A `.partial` directory is not a result set: the harness tools skip it (`bench.PARTIAL_SUFFIX`), git ignores it,
//! and it is removed when the complete set is gathered.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::ssh::SSH_UNREACHABLE;

use super::{POD_RESULTS, PodFiles, PodSet, canonical_dir, recorded_gpu};

/// The suffix of the directory a set's in-progress rows are mirrored into.
pub(crate) const PARTIAL_SUFFIX: &str = ".partial";

/// Frame markers of [`mirror_command`]'s output, each on a line of its own.
const BEGIN: &str = "@@poot-partial ";
const ENV: &str = "@@poot-env";
const END: &str = "@@poot-end";

/// The in-progress rows of one set as the pod has them.
#[derive(Debug, PartialEq, Eq)]
struct PodPartial {
    dir: String,
    rows: Vec<String>,
    env: Option<String>,
}

/// The shell command that prints, for each of `dirs` that has rows, its `results.jsonl` and `env.json` in frames.
/// Each marker is preceded by a newline, so a file without a final newline cannot swallow it.
pub(crate) fn mirror_command(root: &str, dirs: &[String]) -> String {
    format!(
        "for d in {}; do f={root}/\"$d\"; [ -s \"$f\"/results.jsonl ] || continue; \
         printf '\\n{BEGIN}%s\\n' \"$d\"; cat \"$f\"/results.jsonl; printf '\\n{ENV}\\n'; \
         cat \"$f\"/env.json 2>/dev/null; printf '\\n{END}\\n'; done",
        dirs.join(" ")
    )
}

/// Parse [`mirror_command`]'s output. A row is kept only if it is a whole JSON line, so a cell the pod is writing
/// right now cannot leave a torn row; an `env.json` is kept only if it parses.
fn parse_frames(stdout: &str) -> Vec<PodPartial> {
    let mut sets = Vec::new();
    let mut current: Option<PodPartial> = None;
    let mut env = String::new();
    let mut in_env = false;
    for line in stdout.lines() {
        if let Some(dir) = line.strip_prefix(BEGIN) {
            current = Some(PodPartial {
                dir: dir.trim().to_string(),
                rows: Vec::new(),
                env: None,
            });
            env.clear();
            in_env = false;
        } else if line == ENV {
            in_env = true;
        } else if line == END {
            if let Some(mut set) = current.take() {
                set.env = serde_json::from_str::<serde_json::Value>(&env)
                    .ok()
                    .map(|_| env.clone());
                sets.push(set);
            }
            in_env = false;
        } else if let Some(set) = current.as_mut() {
            if in_env {
                env.push_str(line);
                env.push('\n');
            } else if serde_json::from_str::<serde_json::Value>(line).is_ok() {
                set.rows.push(line.to_string());
            }
        }
    }
    sets
}

/// `<results>/<committed name>.partial` of the set the pod calls `pod_dir`, for a set whose env recorded `gpu`.
fn partial_dir(results: &Path, pod_dir: &str, gpu: Option<&str>) -> Result<PathBuf> {
    Ok(results.join(format!("{}{PARTIAL_SUFFIX}", canonical_dir(pod_dir, gpu)?)))
}

/// Remove the mirror of the set the pod calls `pod_dir`, whichever GPU name it was mirrored under (a mirror made
/// before the pod wrote `env.json` is named `nogpu`).
pub(crate) fn clear_partial(results: &Path, pod_dir: &str, gpu: Option<&str>) -> Result<()> {
    for gpu in [gpu, None] {
        let dir = partial_dir(results, pod_dir, gpu)?;
        if dir.exists() {
            std::fs::remove_dir_all(&dir).with_context(|| format!("remove {}", dir.display()))?;
        }
    }
    Ok(())
}

/// Mirror the rows so far of every one of `dirs` that is not complete locally into its `.partial` directory,
/// replacing the earlier copy, and drop the mirror of a set that is complete here. Returns the sets mirrored.
pub(crate) fn mirror_partials(
    pod: &dyn PodFiles,
    results: &Path,
    dirs: &[String],
) -> Result<Vec<String>> {
    let mut open = Vec::new();
    for dir in dirs {
        let gathered = super::local_set(results, dir)
            .map(|name| results.join(name))
            .or_else(|| {
                super::local_failed_set(results, dir)
                    .map(|name| super::failed_dir(results).join(name))
            });
        match gathered {
            Some(home) => clear_partial(
                results,
                dir,
                recorded_gpu(&home.join("env.json")).as_deref(),
            )?,
            None => open.push(dir.clone()),
        }
    }
    if open.is_empty() {
        return Ok(Vec::new());
    }
    let listing = pod
        .ssh(&mirror_command(POD_RESULTS, &open))
        .context("read the rows so far on the pod")?;
    // ssh exits 255 when it cannot connect, and then says nothing: that is a lost connection, not "no rows yet".
    if listing.code == SSH_UNREACHABLE {
        bail!("the pod is unreachable over ssh");
    }
    let mut mirrored = Vec::new();
    for set in parse_frames(&listing.stdout) {
        if !open.contains(&set.dir) || set.rows.is_empty() {
            continue;
        }
        let gpu = set.env.as_deref().and_then(|env| {
            serde_json::from_str::<serde_json::Value>(env)
                .ok()?
                .get("gpu_name")?
                .as_str()
                .map(String::from)
        });
        // The env arrives after the first cells: the nogpu mirror it supersedes is removed.
        clear_partial(results, &set.dir, None)?;
        let dir = partial_dir(results, &set.dir, gpu.as_deref())?;
        write_replacing(&dir, &set)?;
        mirrored.push(set.dir);
    }
    Ok(mirrored)
}

/// The mirror's state across a sweep's ticks: it logs once when it pauses and once when it resumes, instead of on
/// every tick of an outage.
#[derive(Default)]
pub(crate) struct Mirroring {
    paused: bool,
}

impl Mirroring {
    /// One tick, run after the sweep loop's own poll, whose result is `polled`: a pod that did not answer it (ssh
    /// exit 255) is not asked again (each ask costs ssh's connect timeout, which would delay noticing the pod is
    /// lost). Whether the pod answered is read from `polled` here and nowhere else. Returns the line to log, if the
    /// mirror just paused or resumed.
    pub(crate) fn tick(
        &mut self,
        pod: &dyn PodFiles,
        results: &Path,
        dirs: &[String],
        polled: &crate::ssh::Run,
    ) -> Option<String> {
        let outcome = if polled.code != SSH_UNREACHABLE {
            mirror_partials(pod, results, dirs).map(|_| ())
        } else {
            Err(anyhow::anyhow!("the last poll did not reach the pod"))
        };
        match (outcome, self.paused) {
            (Ok(()), true) => {
                self.paused = false;
                Some("  live results resumed".to_string())
            }
            (Err(e), false) => {
                self.paused = true;
                Some(format!("  live results paused: {e:#}"))
            }
            _ => None,
        }
    }
}

/// Remove the `.partial` directories under `results` that mirror none of `active_pod_dirs` (the sets of the runs
/// still in flight): a run that crashed or was killed leaves its mirror behind, and nothing gathers it. Returns the
/// names removed.
pub(crate) fn remove_orphan_partials(
    results: &Path,
    active_pod_dirs: &[String],
) -> Result<Vec<String>> {
    let live: Vec<String> = active_pod_dirs
        .iter()
        .filter_map(|dir| local_partial(results, dir))
        .collect();
    let mut removed = Vec::new();
    let Ok(entries) = std::fs::read_dir(results) else {
        return Ok(removed);
    };
    for entry in entries {
        let entry = entry?;
        let Ok(name) = entry.file_name().into_string() else {
            continue;
        };
        if name.ends_with(PARTIAL_SUFFIX) && !live.contains(&name) && entry.path().is_dir() {
            std::fs::remove_dir_all(entry.path())
                .with_context(|| format!("remove {}", entry.path().display()))?;
            removed.push(name);
        }
    }
    Ok(removed)
}

/// Write `set`'s files into `dir` so a reader sees the old copy or the new one, never a torn file.
fn write_replacing(dir: &Path, set: &PodPartial) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let write = |name: &str, body: &str| -> Result<()> {
        let tmp = dir.join(format!(".{name}.tmp"));
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, dir.join(name))?;
        Ok(())
    };
    write("results.jsonl", &format!("{}\n", set.rows.join("\n")))?;
    if let Some(env) = &set.env {
        write("env.json", env)?;
    }
    Ok(())
}

/// The mirror of the set the pod calls `pod_dir`, if there is one: its directory name under `results`.
pub(crate) fn local_partial(results: &Path, pod_dir: &str) -> Option<String> {
    let set = PodSet::parse(pod_dir).ok()?;
    std::fs::read_dir(results)
        .ok()?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .find(|name| {
            name.strip_suffix(PARTIAL_SUFFIX).is_some_and(|stem| {
                let gpu = recorded_gpu(&results.join(name).join("env.json"));
                stem == set.canonical(gpu.as_deref())
            })
        })
}

/// The file the run's latest pod-side progress line is kept in, beside the other orchestrator side files.
pub(crate) fn progress_file(results: &Path, run_id: &str) -> PathBuf {
    let (staging, _) = super::side_dirs(results);
    staging
        .with_file_name("progress")
        .join(format!("{run_id}.txt"))
}

/// Record the harness's latest progress line (`isl N done (k/4) ... ETA`) for the dashboard.
pub(crate) fn write_progress(results: &Path, run_id: &str, line: &str) -> Result<()> {
    let file = progress_file(results, run_id);
    std::fs::create_dir_all(file.parent().expect("a progress file has a directory"))?;
    std::fs::write(&file, line)?;
    Ok(())
}

/// The latest progress line of a run, if the run process recorded one.
pub(crate) fn read_progress(results: &Path, run_id: &str) -> Option<String> {
    std::fs::read_to_string(progress_file(results, run_id))
        .ok()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The command runs in a real shell, as it would on the pod, and its frames parse back: a quoting error in
    /// the command string shows here without a pod. Mutation: drop the `printf` newline before a marker and a
    /// results file without a final newline swallows it, so the env frame is lost.
    #[test]
    fn the_mirror_command_runs_in_a_real_shell_and_its_frames_parse() {
        let root = std::env::temp_dir().join(format!("poot-mirror-cmd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let (with_rows, no_rows) = ("run-1-2-m-r1", "run-1-2-m-r2");
        std::fs::create_dir_all(root.join(with_rows)).unwrap();
        std::fs::create_dir_all(root.join(no_rows)).unwrap();
        // No final newline after the last row, as a harness killed mid-write would leave it.
        std::fs::write(
            root.join(with_rows).join("results.jsonl"),
            "{\"status\":\"ok\"}\n{\"status\":\"error\"}",
        )
        .unwrap();
        std::fs::write(
            root.join(with_rows).join("env.json"),
            "{\"gpu_name\":\"X\"}",
        )
        .unwrap();

        let dirs = [with_rows.to_string(), no_rows.to_string()];
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(mirror_command(root.to_str().unwrap(), &dirs))
            .output()
            .unwrap();
        std::fs::remove_dir_all(&root).unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );

        let sets = parse_frames(&String::from_utf8(out.stdout).unwrap());
        assert_eq!(
            sets,
            [PodPartial {
                dir: with_rows.to_string(),
                rows: vec![
                    "{\"status\":\"ok\"}".into(),
                    "{\"status\":\"error\"}".into()
                ],
                env: Some("{\"gpu_name\":\"X\"}\n".into()),
            }]
        );
    }
}

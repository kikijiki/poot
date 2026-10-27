//! Incremental gather: copy each result set off the pod as soon as the harness finishes it, so a pod that
//! disappears mid-sweep loses only the repeat in flight, and a later run can tell which repeats are already here.
//!
//! A set is one `bench run` repeat directory: `env.json`, `results.jsonl`, and `report.md`, which the harness writes
//! last. The pod is asked which of the expected directories have all three; only those are copied, into a staging
//! directory beside `results/`, checked with the harness's own identity rule, and moved into place whole.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub(crate) mod partial;
pub(crate) use partial::{
    Mirroring, clear_partial, local_partial, read_progress, remove_orphan_partials, write_progress,
};

use crate::clock::iso_from_secs;
use crate::heal::{HostFault, LocalFault, SetupFault, host_fault_in, setup_fault_in};
use crate::logging::log;
use crate::ssh;

/// The files that make a result set complete (the harness writes `report.md` after the last row).
const SET_FILES: [&str; 3] = ["env.json", "results.jsonl", "report.md"];

/// The most repeats of one set `--extend-set` looks for when it checks that the set exists (the performance
/// policy's `max_trials`).
pub(crate) const MAX_EXTENDED_REPEATS: u32 = 15;

/// Where the harness writes result sets on the pod.
pub(crate) const POD_RESULTS: &str = "/opt/benchmarks/results";

/// What the incremental gather needs from the pod: a shell and a recursive copy down.
pub(crate) trait PodFiles {
    fn ssh(&self, cmd: &str) -> Result<ssh::Run>;
    fn scp_down(&self, remote: &str, local: &str) -> Result<()>;
}

impl PodFiles for ssh::Endpoint {
    fn ssh(&self, cmd: &str) -> Result<ssh::Run> {
        ssh::Endpoint::ssh(self, cmd)
    }
    fn scp_down(&self, remote: &str, local: &str) -> Result<()> {
        ssh::Endpoint::scp_down(self, remote, local)
    }
}

/// Why a copied set is not committed: the harness's rules for a committed row, one message per violation.
pub(crate) type IdentityCheck<'a> = &'a dyn Fn(&Path) -> Result<Vec<String>>;

/// Outcome of one gather pass.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Gathered {
    /// Sets copied into `results/` by this pass.
    pub(crate) copied: Vec<String>,
    /// Sets copied but refused by the identity check, moved to the rejected directory.
    pub(crate) rejected: Vec<String>,
    /// The sets of `copied` that have no `ok` row, which went to [`failed_dir`] instead of `results/`.
    pub(crate) failed: Vec<String>,
}

/// Does `dir` hold all the files of a complete set?
fn files_complete(dir: &Path) -> bool {
    SET_FILES
        .iter()
        .all(|file| std::fs::metadata(dir.join(file)).is_ok_and(|m| m.is_file() && m.len() > 0))
}

/// A result set as the orchestrator names it on the pod, `run-<epoch>[-<pid>]-<model>-r<N>`: the run that started
/// the set (its epoch second), the model and the repeat. The name a gathered set is committed under is built from
/// these and the GPU the harness recorded, the way `bench.py` names a run directory.
struct PodSet {
    epoch: i64,
    model: String,
    repeat: u32,
}

/// `<model>-r<N>` split into the model and N, if `name` ends in a repeat suffix. A model whose id merely contains
/// "-r" (`llama-rerank`) has none.
fn split_repeat(name: &str) -> Option<(&str, u32)> {
    let (model, n) = name.rsplit_once("-r")?;
    (!n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).then_some(())?;
    Some((model, n.parse().ok()?))
}

/// Would a set of one repeat of the model `id` be mistaken for repeat N of another model? The harness names a
/// one-repeat set without a suffix, so `run-<epoch>-<pid>-<id>` with an id ending in `-r<N>` reads as that repeat.
pub(crate) fn id_ends_in_a_repeat_suffix(id: &str) -> bool {
    split_repeat(id).is_some()
}

impl PodSet {
    /// `run-<epoch>-<pid>-<model>-r<N>`, or `run-<epoch>-<pid>-<model>` for a run of one repeat, which the harness
    /// names without a repeat suffix (that set is repeat 1). An unparseable name is a [`LocalFault`]: the set is on
    /// the pod, this side cannot place it.
    fn parse(dir: &str) -> Result<Self> {
        let parsed = (|| {
            let (epoch, rest) = dir.strip_prefix("run-")?.split_once('-')?;
            // The set id is `run-<epoch>-<pid>`; a bare `run-<epoch>` has no pid to skip.
            let rest = match rest.split_once('-') {
                Some((pid, tail)) if !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()) => {
                    tail
                }
                _ => rest,
            };
            let (model, repeat) = split_repeat(rest).unwrap_or((rest, 1));
            Some(Self {
                epoch: epoch.parse().ok()?,
                model: model.to_string(),
                repeat,
            })
        })();
        parsed.filter(|set| !set.model.is_empty()).ok_or_else(|| {
            LocalFault(format!(
                "set {dir} is not named run-<epoch>-<pid>-<model>[-r<N>]"
            ))
            .into()
        })
    }

    /// The UTC date (`YYYY-MM-DD`) and time (`HHMMSS`) the set started.
    fn started(&self) -> (String, String) {
        let iso = iso_from_secs(self.epoch);
        (iso[..10].to_string(), iso[11..19].replace(':', ""))
    }

    /// `<model>-<time>-r<N>`: the part of the committed name after the date and GPU.
    fn tail(&self) -> String {
        let model = self.model.replace(['/', '.'], "-");
        let (_, time) = self.started();
        format!("{model}-{time}-r{}", self.repeat)
    }

    /// The committed name, `<date>-<gpu>-<model>-<time>-r<N>`, as `bench.py` names a run directory. A set that
    /// recorded no GPU is `nogpu`, as there.
    fn canonical(&self, gpu_name: Option<&str>) -> String {
        let (date, _) = self.started();
        format!("{date}-{}-{}", gpu_slug(gpu_name), self.tail())
    }

    /// Is the directory `name` under `results` this set's committed name? The GPU part is what its own `env.json`
    /// records, so the name must be exactly what gathering it would write: a model whose id is the end of
    /// another's (`4b`, `qwen3-4b`) never matches the other's sets.
    fn is_named(&self, results: &Path, name: &str) -> bool {
        let gpu = recorded_gpu(&results.join(name).join("env.json"));
        name == self.canonical(gpu.as_deref())
    }
}

/// The GPU part of a committed set name: `bench.py`'s slug of `env.json`'s `gpu_name`.
fn gpu_slug(gpu_name: Option<&str>) -> String {
    let lower = gpu_name.unwrap_or("nogpu").to_lowercase();
    let mut slug = String::new();
    for c in lower.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            slug.push(c);
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.replace("nvidia-", "").trim_matches('-').to_string()
}

/// The directory a set that the pod calls `pod_dir` is committed under, given the GPU its env recorded.
pub(crate) fn canonical_dir(pod_dir: &str, gpu_name: Option<&str>) -> Result<String> {
    Ok(PodSet::parse(pod_dir)?.canonical(gpu_name))
}

/// Where sets with no `ok` row are gathered, beside `results/` (git-ignored, `benchmarks/.gitignore`): the
/// harness index holds only runs that measured something, and a set of errors is still worth reading.
pub(crate) fn failed_dir(results: &Path) -> PathBuf {
    results.with_file_name(FAILED_DIR)
}

/// The directory name of [`failed_dir`].
const FAILED_DIR: &str = "failed-results";

/// The directory under `root` holding the complete set the pod calls `pod_dir`, if it is here.
fn find_set(root: &Path, pod_dir: &str) -> Option<String> {
    let set = PodSet::parse(pod_dir).ok()?;
    std::fs::read_dir(root)
        .ok()?
        .filter_map(|entry| entry.ok()?.file_name().into_string().ok())
        .find(|name| set.is_named(root, name) && files_complete(&root.join(name)))
}

/// The directory under `results` holding the complete set the pod calls `pod_dir`, if it is here.
pub(crate) fn local_set(results: &Path, pod_dir: &str) -> Option<String> {
    find_set(results, pod_dir)
}

/// The directory under [`failed_dir`] holding the complete set the pod calls `pod_dir` (a set with no `ok` row).
pub(crate) fn local_failed_set(results: &Path, pod_dir: &str) -> Option<String> {
    find_set(&failed_dir(results), pod_dir)
}

/// Is the set the pod calls `pod_dir` gathered, in `results` or among the failed sets?
pub(crate) fn set_is_complete(results: &Path, pod_dir: &str) -> bool {
    local_set(results, pod_dir).is_some() || local_failed_set(results, pod_dir).is_some()
}

/// Did any row of the set at `dir` measure something? A set with none is gathered as a failed set.
fn has_ok_row(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("results.jsonl")).is_ok_and(|text| {
        text.lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .any(|row| row["status"] == "ok")
    })
}

/// The GPU an `env.json` recorded, if it names one.
fn recorded_gpu(env_json: &Path) -> Option<String> {
    let env: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(env_json).ok()?).ok()?;
    env.get("gpu_name")?.as_str().map(String::from)
}

/// The host fault a copied set records, if any. A set measured on a pod whose GPU was gone says so in its own
/// files: `env.json` lists no GPU, or a row errored with a CUDA marker such as `CUDA_ERROR_NO_DEVICE`. Such a set
/// measured nothing (an external tool that fell back to the CPU measured the wrong device), and the pod that wrote it
/// is broken: it is never gathered, as a result or as a failed set.
fn host_fault_of(set: &Path) -> Option<HostFault> {
    if recorded_gpu(&set.join("env.json")).is_none() {
        return Some(HostFault(
            "no CUDA-capable device is detected: the set's env.json lists no GPU".into(),
        ));
    }
    let rows = std::fs::read_to_string(set.join("results.jsonl")).ok()?;
    rows.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|row| row["status"] == "error")
        .find_map(|row| host_fault_in(row["reason"].as_str()?))
}

/// The setup fault a copied set records, if any: no row measured anything and one says the model is not on the
/// pod. Such a repeat is not a result and not a failed cell, it is the pod missing its setup.
fn setup_fault_of(set: &Path) -> Option<SetupFault> {
    if has_ok_row(set) {
        return None;
    }
    let rows = std::fs::read_to_string(set.join("results.jsonl")).ok()?;
    rows.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|row| row["status"] == "error")
        .find_map(|row| setup_fault_in(row["reason"].as_str()?))
}

/// The shell command that prints the entries of `dirs` that are complete on the pod.
fn complete_on_pod_command(dirs: &[String]) -> String {
    let tests: Vec<String> = SET_FILES
        .iter()
        .map(|file| format!("[ -s {POD_RESULTS}/\"$d\"/{file} ]"))
        .collect();
    format!(
        "for d in {}; do {} && echo \"$d\"; done",
        dirs.join(" "),
        tests.join(" && ")
    )
}

/// Copy every set of `dirs` that is complete on the pod and not yet complete locally into `results`. A set that
/// fails `check` is moved to `rejected` and reported, never left in `results`. A pass that fails to reach the pod
/// is an error; sets already copied stay copied. A set that records a host fault ([`host_fault_of`]) is moved to
/// `rejected` and the pass fails with that [`HostFault`], so the pod is replaced and the repeat runs again; one that
/// records a [`SetupFault`] is treated the same way, and the pod is set up again instead.
pub(crate) fn gather_sets(
    pod: &dyn PodFiles,
    results: &Path,
    staging: &Path,
    rejected: &Path,
    dirs: &[String],
    check: IdentityCheck<'_>,
) -> Result<Gathered> {
    let missing: Vec<String> = dirs
        .iter()
        .filter(|dir| !set_is_complete(results, dir))
        .cloned()
        .collect();
    let mut gathered = Gathered::default();
    if missing.is_empty() {
        return Ok(gathered);
    }
    let listing = pod
        .ssh(&complete_on_pod_command(&missing))
        .context("list the complete sets on the pod")?;
    let ready: Vec<&str> = listing
        .stdout
        .lines()
        .map(str::trim)
        .filter(|line| missing.iter().any(|dir| dir == line))
        .collect();
    for dir in ready {
        let fresh = staging.join(dir);
        // A copy interrupted earlier left a partial directory here; it is never promoted.
        if fresh.exists() {
            std::fs::remove_dir_all(&fresh)?;
        }
        std::fs::create_dir_all(staging)?;
        pod.scp_down(
            &format!("{POD_RESULTS}/{dir}"),
            &format!("{}/", staging.display()),
        )
        .with_context(|| format!("copy set {dir} from the pod"))?;
        if !files_complete(&fresh) {
            bail!("set {dir} copied incompletely; not promoted");
        }
        if let Some(fault) = host_fault_of(&fresh) {
            let kept = quarantine(&fresh, rejected, dir)?;
            log(format!(
                "HOST FAULT in set {dir} (kept in {}): {}",
                kept.display(),
                fault.0
            ));
            return Err(fault.into());
        }
        if let Some(fault) = setup_fault_of(&fresh) {
            let kept = quarantine(&fresh, rejected, dir)?;
            log(format!(
                "SETUP FAULT in set {dir} (kept in {}): {}",
                kept.display(),
                fault.0
            ));
            return Err(fault.into());
        }
        let errors = check(&fresh)?;
        if errors.is_empty() {
            let name = canonical_dir(dir, recorded_gpu(&fresh.join("env.json")).as_deref())?;
            let failed = !has_ok_row(&fresh);
            let root = if failed {
                failed_dir(results)
            } else {
                results.to_path_buf()
            };
            let home = root.join(&name);
            std::fs::create_dir_all(&root)?;
            // Only an incomplete directory can be in the way: a complete set under this name is not "missing".
            if home.exists() {
                std::fs::remove_dir_all(&home)?;
            }
            std::fs::rename(&fresh, &home)?;
            clear_partial(
                results,
                dir,
                recorded_gpu(&home.join("env.json")).as_deref(),
            )?;
            if failed {
                log(format!(
                    "gathered set {dir} as {name}: no row measured anything, kept in {}",
                    root.display()
                ));
                gathered.failed.push(dir.to_string());
            } else {
                log(format!("gathered set {dir} as {name}"));
            }
            gathered.copied.push(dir.to_string());
        } else {
            let kept = quarantine(&fresh, rejected, dir)?;
            log(format!(
                "REJECTED set {dir} (kept in {}): {}",
                kept.display(),
                errors.join("; ")
            ));
            gathered.rejected.push(dir.to_string());
        }
    }
    Ok(gathered)
}

/// Move the copied set `fresh` into `rejected/<dir>`, replacing an earlier copy; returns where it is kept.
fn quarantine(fresh: &Path, rejected: &Path, dir: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(rejected)?;
    let kept = rejected.join(dir);
    if kept.exists() {
        std::fs::remove_dir_all(&kept)?;
    }
    std::fs::rename(fresh, &kept)?;
    Ok(kept)
}

/// The repeat numbers (1-based) of `dirs`, in order, whose set is already complete under `results`.
pub(crate) fn repeats_present(results: &Path, dirs: &[String]) -> Vec<u32> {
    dirs.iter()
        .zip(1u32..)
        .filter(|(dir, _)| set_is_complete(results, dir))
        .map(|(_, n)| n)
        .collect()
}

/// The staging and rejected directories next to `results/`, ignored by git (`benchmarks/.gitignore`).
pub(crate) fn side_dirs(results: &Path) -> (PathBuf, PathBuf) {
    let base = results.parent().map_or_else(
        || PathBuf::from(".orchestrator"),
        |p| p.join(".orchestrator"),
    );
    (base.join("incoming"), base.join("rejected"))
}

/// The harness's rule for a committed row (`result_contract.committed_row_errors`) over every row of a set.
pub(crate) fn harness_identity_errors(harness_dir: &Path, set: &Path) -> Result<Vec<String>> {
    const SCRIPT: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from result_contract import committed_row_errors
for line in open(sys.argv[2] + "/results.jsonl"):
    if line.strip():
        row = json.loads(line)
        for error in committed_row_errors(row):
            print(f"{row.get('model')}/{row.get('scenario')}/{row.get('framework')}: {error}")
"#;
    let out = std::process::Command::new("python3")
        .args(["-I", "-c", SCRIPT])
        .arg(harness_dir)
        .arg(set)
        .output()
        .context("spawn python3 for the identity check")?;
    if !out.status.success() {
        bail!(
            "identity check failed to run: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(String::from)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::partial::{PARTIAL_SUFFIX, mirror_partials};
    use super::*;
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    const GPU: &str = "NVIDIA GeForce RTX 3090";
    /// A pod set name as the orchestrator makes it, and the name it is committed under.
    const R1: &str = "run-1792632197-3202502-qwen3-0.6b-r1";
    const R2: &str = "run-1792632197-3202502-qwen3-0.6b-r2";
    const R3: &str = "run-1792632197-3202502-qwen3-0.6b-r3";
    const C1: &str = "2026-10-22-geforce-rtx-3090-qwen3-0-6b-012317-r1";
    const C2: &str = "2026-10-22-geforce-rtx-3090-qwen3-0-6b-012317-r2";

    /// A scratch directory removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "poot-gather-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A pod whose results directory is a map of set -> files present (name -> contents), changeable between passes.
    struct FakePod {
        sets: RefCell<BTreeMap<String, BTreeMap<String, String>>>,
        copies: RefCell<Vec<String>>,
        /// ssh commands run, and whether the pod answers them (ssh exits 255 when it cannot connect).
        asked: std::cell::Cell<u32>,
        unreachable: std::cell::Cell<bool>,
    }

    impl FakePod {
        fn new() -> Self {
            Self {
                sets: RefCell::new(BTreeMap::new()),
                copies: RefCell::new(Vec::new()),
                asked: std::cell::Cell::new(0),
                unreachable: std::cell::Cell::new(false),
            }
        }
        fn write(&self, dir: &str, files: &[&str]) {
            let mut sets = self.sets.borrow_mut();
            let entry = sets.entry(dir.to_string()).or_default();
            for file in files {
                let body = if *file == "env.json" {
                    format!(r#"{{"gpu_name":"{GPU}"}}"#)
                } else if *file == "results.jsonl" {
                    format!(r#"{{"status":"ok","pod_dir":"{dir}"}}"#)
                } else {
                    format!("{dir}/{file}")
                };
                entry.insert((*file).to_string(), body);
            }
        }
        /// Replace one file of a set on the pod.
        fn put(&self, dir: &str, file: &str, body: &str) {
            self.sets
                .borrow_mut()
                .entry(dir.to_string())
                .or_default()
                .insert(file.to_string(), body.to_string());
        }
        fn finish(&self, dir: &str) {
            self.write(dir, &SET_FILES);
        }
    }

    impl PodFiles for FakePod {
        fn ssh(&self, cmd: &str) -> Result<ssh::Run> {
            self.asked.set(self.asked.get() + 1);
            if self.unreachable.get() {
                return Ok(ssh::Run {
                    code: 255,
                    stdout: String::new(),
                    stderr: "ssh: connect timed out".into(),
                });
            }
            if cmd.contains("@@poot-partial") {
                // The mirror command: a frame for each asked directory that has rows.
                let asked = cmd
                    .strip_prefix("for d in ")
                    .and_then(|r| r.split(';').next())
                    .unwrap_or("");
                let sets = self.sets.borrow();
                let mut stdout = String::new();
                for dir in asked.split_whitespace() {
                    let Some(files) = sets.get(dir).filter(|f| f.contains_key("results.jsonl"))
                    else {
                        continue;
                    };
                    stdout.push_str(&format!(
                        "\n@@poot-partial {dir}\n{}",
                        files["results.jsonl"]
                    ));
                    stdout.push_str(&format!(
                        "\n@@poot-env\n{}",
                        files.get("env.json").map_or("", String::as_str)
                    ));
                    stdout.push_str("\n@@poot-end\n");
                }
                return Ok(ssh::Run {
                    code: 0,
                    stdout,
                    stderr: String::new(),
                });
            }
            // The listing command names the directories it asks about and the files each must hold; answer
            // as the pod's shell would for the files this command tests.
            let asked = cmd
                .strip_prefix("for d in ")
                .and_then(|rest| rest.split(';').next())
                .unwrap_or("");
            let tested: Vec<&str> = cmd
                .split("\"$d\"/")
                .skip(1)
                .filter_map(|part| part.split_whitespace().next())
                .collect();
            let sets = self.sets.borrow();
            let stdout = asked
                .split_whitespace()
                .filter(|dir| {
                    sets.get(*dir)
                        .is_some_and(|files| tested.iter().all(|f| files.contains_key(*f)))
                })
                .map(|dir| format!("{dir}\n"))
                .collect();
            Ok(ssh::Run {
                code: 0,
                stdout,
                stderr: String::new(),
            })
        }
        fn scp_down(&self, remote: &str, local: &str) -> Result<()> {
            let dir = remote.rsplit('/').next().unwrap().to_string();
            self.copies.borrow_mut().push(dir.clone());
            let target = Path::new(local).join(&dir);
            std::fs::create_dir_all(&target)?;
            for (name, body) in &self.sets.borrow()[&dir] {
                std::fs::write(target.join(name), body)?;
            }
            Ok(())
        }
    }

    fn dirs(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_string()).collect()
    }

    fn pass(pod: &FakePod, root: &Path, wanted: &[&str], check: IdentityCheck<'_>) -> Gathered {
        gather_sets(
            pod,
            &root.join("results"),
            &root.join("incoming"),
            &root.join("rejected"),
            &dirs(wanted),
            check,
        )
        .unwrap()
    }

    fn accept(_: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// A set whose `report.md` the harness has not written yet is not copied; it is copied once it is.
    /// Mutation: list a directory on `results.jsonl` alone and the first assertion goes red.
    #[test]
    fn a_partially_written_set_is_never_copied() {
        let tmp = TempDir::new();
        let pod = FakePod::new();
        pod.write(R1, &["env.json", "results.jsonl"]);

        let got = pass(&pod, tmp.path(), &[R1], &accept);
        assert_eq!(got, Gathered::default());
        assert!(pod.copies.borrow().is_empty());
        assert!(!set_is_complete(&tmp.path().join("results"), R1));

        pod.finish(R1);
        let got = pass(&pod, tmp.path(), &[R1], &accept);
        assert_eq!(got.copied, [R1]);
        assert!(set_is_complete(&tmp.path().join("results"), R1));
    }

    /// Repeats finish one by one while the sweep runs; each pass copies the newly finished ones and no
    /// finished one twice, so a pod that vanishes after repeat 2 leaves repeats 1 and 2 local.
    #[test]
    fn finished_repeats_are_copied_while_the_sweep_runs_and_never_twice() {
        let tmp = TempDir::new();
        let pod = FakePod::new();
        let wanted = [R1, R2, R3];

        pod.finish(R1);
        assert_eq!(pass(&pod, tmp.path(), &wanted, &accept).copied, [R1]);
        pod.write(R2, &["env.json"]);
        assert_eq!(
            pass(&pod, tmp.path(), &wanted, &accept).copied,
            Vec::<String>::new()
        );
        pod.finish(R2);
        assert_eq!(pass(&pod, tmp.path(), &wanted, &accept).copied, [R2]);
        assert_eq!(
            pass(&pod, tmp.path(), &wanted, &accept).copied,
            Vec::<String>::new()
        );

        assert_eq!(*pod.copies.borrow(), [R1, R2]);
        assert_eq!(
            repeats_present(&tmp.path().join("results"), &dirs(&wanted)),
            [1, 2]
        );
    }

    /// A set the harness's identity rule refuses is reported and kept out of `results/`.
    #[test]
    fn a_set_that_fails_the_identity_check_is_rejected_not_gathered() {
        let tmp = TempDir::new();
        let pod = FakePod::new();
        pod.finish(R1);
        let refuse = |_: &Path| Ok(vec!["build_sha ends in '-dirty'".to_string()]);

        let got = pass(&pod, tmp.path(), &[R1], &refuse);

        assert_eq!(got.rejected, [R1]);
        assert!(got.copied.is_empty());
        assert!(!set_is_complete(&tmp.path().join("results"), R1));
        assert!(files_complete(&tmp.path().join("rejected").join(R1)));
    }

    /// A copy that died half way leaves a partial staging directory; the next pass replaces it instead of promoting it.
    #[test]
    fn a_stale_partial_staging_copy_is_replaced() {
        let tmp = TempDir::new();
        let pod = FakePod::new();
        pod.finish(R1);
        let stale = tmp.path().join("incoming").join(R1);
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("results.jsonl"), "half").unwrap();

        pass(&pod, tmp.path(), &[R1], &accept);

        let copied =
            std::fs::read_to_string(tmp.path().join("results").join(C1).join("results.jsonl"))
                .unwrap();
        assert_eq!(copied, format!(r#"{{"status":"ok","pod_dir":"{R1}"}}"#));
    }

    /// A gathered set lands under the name every committed set has, `<date>-<gpu>-<model>-<time>-r<N>`: the
    /// date and time are the set's start (its run id's epoch), the GPU is the one its `env.json` recorded.
    /// Mutations: copy under the pod's name, or take the GPU from anywhere but env.json, and this goes red.
    #[test]
    fn a_gathered_set_is_committed_under_the_canonical_name() {
        let tmp = TempDir::new();
        let pod = FakePod::new();
        pod.finish(R1);
        pod.finish(R2);

        pass(&pod, tmp.path(), &[R1, R2], &accept);

        let mut names: Vec<String> = std::fs::read_dir(tmp.path().join("results"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, [C1, C2]);
        assert_eq!(
            local_set(&tmp.path().join("results"), R2).as_deref(),
            Some(C2)
        );
    }

    /// The name is `bench.py`'s own: its slug rules for the GPU and model, and `nogpu` when none was recorded.
    #[test]
    fn the_canonical_name_follows_the_harness_slug_rules() {
        let name = |pod: &str, gpu: Option<&str>| canonical_dir(pod, gpu).unwrap();
        assert_eq!(
            name(
                "run-1792632197-3202502-gemma-3-1b-it-r4",
                Some("NVIDIA H100 80GB HBM3")
            ),
            "2026-10-22-h100-80gb-hbm3-gemma-3-1b-it-012317-r4"
        );
        assert_eq!(name(R3, None), "2026-10-22-nogpu-qwen3-0-6b-012317-r3");
        // An older set id has no pid.
        assert_eq!(
            name(
                "run-1792632197-qwen3-4b-r1",
                Some("AMD Radeon 8060S Graphics")
            ),
            "2026-10-22-amd-radeon-8060s-graphics-qwen3-4b-012317-r1"
        );
        assert!(canonical_dir("s-m-r1", None).is_err());
    }

    fn local_complete_set(results: &Path, name: &str) {
        std::fs::create_dir_all(results.join(name)).unwrap();
        for file in SET_FILES {
            let body = if file == "env.json" {
                format!(r#"{{"gpu_name":"{GPU}"}}"#)
            } else {
                "x".to_string()
            };
            std::fs::write(results.join(name).join(file), body).unwrap();
        }
    }

    /// A set already local is found under its committed name, so a resume skips it; another repeat, model or
    /// start time is a different set.
    #[test]
    fn a_local_set_is_found_by_its_pod_name() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        for name in [C1, "2026-10-22-geforce-rtx-3090-qwen3-4b-012317-r2"] {
            local_complete_set(&results, name);
        }
        assert_eq!(local_set(&results, R1).as_deref(), Some(C1));
        assert_eq!(local_set(&results, R2), None);
        assert_eq!(
            local_set(&results, "run-1792632198-3202502-qwen3-0.6b-r1"),
            None
        );
    }

    /// A model whose id is the end of another's is a different model: `4b` never finds `qwen3-4b`'s sets.
    /// Mutation: match by suffix alone (any GPU part) and the first assertion goes red.
    #[test]
    fn a_model_whose_id_ends_another_models_id_does_not_find_its_sets() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let long = "2026-10-22-geforce-rtx-3090-qwen3-4b-012317-r1";
        let short = "2026-10-22-geforce-rtx-3090-4b-012317-r1";
        let (long_pod, short_pod) = (
            "run-1792632197-3202502-qwen3-4b-r1",
            "run-1792632197-3202502-4b-r1",
        );
        local_complete_set(&results, long);
        assert_eq!(local_set(&results, short_pod), None);
        assert_eq!(local_set(&results, long_pod).as_deref(), Some(long));

        local_complete_set(&results, short);
        assert_eq!(local_set(&results, short_pod).as_deref(), Some(short));
        assert_eq!(local_set(&results, long_pod).as_deref(), Some(long));
    }

    /// The real identity check refuses a dirty poot row and accepts a clean external one.
    #[test]
    fn the_harness_identity_rule_judges_a_set() {
        let tmp = TempDir::new();
        let harness = crate::workflow::repo_root()
            .expect("the checkout the test runs in")
            .join("benchmarks/harness");
        let bad = tmp.path().join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(
            bad.join("results.jsonl"),
            r#"{"framework":"poot","model":"m","scenario":"s","status":"ok","backend":"ptx","build_sha":"0000000000000000000000000000000000000000-dirty"}"#,
        )
        .unwrap();
        let errors = harness_identity_errors(&harness, &bad).unwrap();
        assert!(errors.iter().any(|e| e.contains("-dirty")), "{errors:?}");

        let good = tmp.path().join("good");
        std::fs::create_dir_all(&good).unwrap();
        std::fs::write(
            good.join("results.jsonl"),
            r#"{"framework":"llamacpp","model":"m","scenario":"s","status":"ok","framework_version":"b1"}"#,
        )
        .unwrap();
        assert_eq!(
            harness_identity_errors(&harness, &good).unwrap(),
            Vec::<String>::new()
        );
    }
    const ERR_ROW: &str = r#"{"status":"error","reason":"rc=1: boom"}"#;
    const OK_ROW: &str = r#"{"status":"ok"}"#;

    /// A set whose rows are all errors goes to `failed-results/`, never `results/`; a set with an ok row among its
    /// errors stays in `results/`. Either counts as gathered, so a later pass does not copy it again.
    /// Mutation: gather writes every set into `results/` and the all-error case goes red.
    #[test]
    fn a_set_with_no_ok_row_is_gathered_into_failed_results() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let pod = FakePod::new();
        pod.finish(R1);
        pod.put(R1, "results.jsonl", ERR_ROW);
        pod.finish(R2);
        pod.put(R2, "results.jsonl", &format!("{ERR_ROW}\n{OK_ROW}"));

        let got = pass(&pod, tmp.path(), &[R1, R2], &accept);
        assert_eq!(got.failed, [R1]);
        assert_eq!(got.copied, [R1, R2]);
        assert!(
            failed_dir(&results)
                .join(C1)
                .join("results.jsonl")
                .is_file()
        );
        assert!(!results.join(C1).exists());
        assert!(results.join(C2).join("results.jsonl").is_file());
        assert!(!failed_dir(&results).join(C2).exists());

        assert!(set_is_complete(&results, R1) && set_is_complete(&results, R2));
        assert_eq!(
            local_set(&results, R1),
            None,
            "a failed set is not a result"
        );
        assert_eq!(local_failed_set(&results, R1).as_deref(), Some(C1));
        let again = pass(&pod, tmp.path(), &[R1, R2], &accept);
        assert_eq!(again, Gathered::default());
    }

    fn partials(results: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(results)
            .map(|d| {
                d.filter_map(|e| e.ok()?.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default();
        names.retain(|n| n.ends_with(PARTIAL_SUFFIX));
        names.sort();
        names
    }

    /// While a set is swept its rows so far are mirrored into `<committed name>.partial`, replaced on every pass
    /// as the pod appends cells. Mutation: skip the copy when the mirror exists and the second assertion goes red.
    #[test]
    fn the_rows_so_far_are_mirrored_and_refreshed_while_sweeping() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let pod = FakePod::new();
        pod.put(R1, "results.jsonl", OK_ROW);

        let mirrored = mirror_partials(&pod, &results, &dirs(&[R1, R2])).unwrap();
        assert_eq!(mirrored, [R1]);
        // No env.json yet on the pod: the mirror is named for no GPU.
        assert_eq!(
            partials(&results),
            ["2026-10-22-nogpu-qwen3-0-6b-012317-r1.partial"]
        );

        pod.put(
            R1,
            "results.jsonl",
            &format!("{OK_ROW}\n{ERR_ROW}\n{{\"status\":\"ok\",\"torn"),
        );
        pod.put(R1, "env.json", &format!(r#"{{"gpu_name":"{GPU}"}}"#));
        mirror_partials(&pod, &results, &dirs(&[R1])).unwrap();
        // The env arrived: the mirror is named for the GPU, the nogpu one is gone, and the torn last line is not copied.
        assert_eq!(partials(&results), [format!("{C1}{PARTIAL_SUFFIX}")]);
        let rows =
            std::fs::read_to_string(results.join(format!("{C1}.partial")).join("results.jsonl"))
                .unwrap();
        assert_eq!(rows.lines().collect::<Vec<_>>(), [OK_ROW, ERR_ROW]);
        assert!(
            results
                .join(format!("{C1}.partial"))
                .join("env.json")
                .is_file()
        );
        assert_eq!(
            local_partial(&results, R1).as_deref(),
            Some(format!("{C1}.partial").as_str())
        );
        assert_eq!(local_set(&results, R1), None, "a mirror is not a set");

        // The pod finishes another cell: the same mirror is refreshed, not kept as first copied.
        pod.put(
            R1,
            "results.jsonl",
            &format!("{OK_ROW}\n{ERR_ROW}\n{OK_ROW}"),
        );
        mirror_partials(&pod, &results, &dirs(&[R1])).unwrap();
        let rows =
            std::fs::read_to_string(results.join(format!("{C1}.partial")).join("results.jsonl"))
                .unwrap();
        assert_eq!(rows.lines().count(), 3);
    }

    /// The mirror is replaced by the complete set at gather: nothing `.partial` is left. Mutation: do not clear the
    /// mirror when the set is moved into place and the assertion goes red.
    #[test]
    fn the_mirror_is_removed_when_the_complete_set_is_gathered() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let pod = FakePod::new();
        pod.write(R1, &["env.json", "results.jsonl"]);
        mirror_partials(&pod, &results, &dirs(&[R1])).unwrap();
        assert_eq!(partials(&results).len(), 1);

        pod.finish(R1);
        pass(&pod, tmp.path(), &[R1], &accept);
        assert!(results.join(C1).join("report.md").is_file());
        assert_eq!(partials(&results), Vec::<String>::new());
        // A mirror of a set that is already here (a pass that raced the gather) is dropped by the next mirror pass.
        std::fs::create_dir_all(results.join(format!("{C1}.partial"))).unwrap();
        mirror_partials(&pod, &results, &dirs(&[R1])).unwrap();
        assert_eq!(partials(&results), Vec::<String>::new());
    }

    /// The directories the harness would read as results never include a mirror or the failed sets.
    #[test]
    fn the_failed_and_partial_locations_are_not_in_results() {
        let results = Path::new("benchmarks/results");
        assert_eq!(failed_dir(results), Path::new("benchmarks/failed-results"));
        assert_eq!(PARTIAL_SUFFIX, ".partial");
    }

    /// The harness's progress line is kept per run for the dashboard.
    #[test]
    fn the_progress_line_is_kept_per_run() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        assert_eq!(read_progress(&results, "run-1"), None);
        write_progress(&results, "run-1", "m: isl 128 done (1/4)").unwrap();
        write_progress(&results, "run-1", "m: isl 512 done (2/4)").unwrap();
        assert_eq!(
            read_progress(&results, "run-1").as_deref(),
            Some("m: isl 512 done (2/4)")
        );
        assert_eq!(read_progress(&results, "run-2"), None);
    }
    /// The sweep loop's poll of the log, as `Mirroring::tick` sees it: answered, or ssh exit 255.
    fn poll_answered() -> crate::ssh::Run {
        crate::ssh::Run {
            code: 0,
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    fn poll_without_answer() -> crate::ssh::Run {
        crate::ssh::Run {
            code: crate::ssh::SSH_UNREACHABLE,
            ..poll_answered()
        }
    }

    /// The mirror is not asked of a pod the last poll could not reach, and a pause and a resume are each logged
    /// once, not on every tick of an outage. Mutations: read the poll's exit status as always reached (or ignore it) and the pod is asked while unreachable;
    /// log on every failing tick and the second assertion of each pause goes red.
    #[test]
    fn the_mirror_pauses_once_while_the_pod_is_unreachable_and_resumes() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let pod = FakePod::new();
        pod.put(R1, "results.jsonl", OK_ROW);
        let mut mirroring = Mirroring::default();
        let wanted = dirs(&[R1]);

        // The sweep loop's poll got ssh exit 255: no ask, one line.
        let first = mirroring.tick(&pod, &results, &wanted, &poll_without_answer());
        assert!(first.unwrap().contains("paused"));
        assert_eq!(
            mirroring.tick(&pod, &results, &wanted, &poll_without_answer()),
            None
        );
        assert_eq!(pod.asked.get(), 0, "an unreachable pod is not asked");

        // The poll got through but the mirror's own ask exits 255: still one line for the whole outage.
        pod.unreachable.set(true);
        assert_eq!(
            mirroring.tick(&pod, &results, &wanted, &poll_answered()),
            None,
            "already paused"
        );
        let mut fresh = Mirroring::default();
        assert!(
            fresh
                .tick(&pod, &results, &wanted, &poll_answered())
                .unwrap()
                .contains("unreachable")
        );
        assert_eq!(fresh.tick(&pod, &results, &wanted, &poll_answered()), None);
        assert_eq!(
            partials(&results),
            Vec::<String>::new(),
            "a 255 is not an empty listing"
        );

        // The pod answers again: one resume line, and the rows arrive.
        pod.unreachable.set(false);
        assert!(
            mirroring
                .tick(&pod, &results, &wanted, &poll_answered())
                .unwrap()
                .contains("resumed")
        );
        assert_eq!(
            mirroring.tick(&pod, &results, &wanted, &poll_answered()),
            None
        );
        assert_eq!(partials(&results).len(), 1);
    }

    /// A mirror that no run in flight owns (its run crashed or was killed) is removed; one an active run sweeps,
    /// and every complete set, stay. Mutations: remove every mirror and the first assertion goes red; remove none
    /// and the second does.
    #[test]
    fn an_orphan_mirror_is_removed_and_a_live_one_kept() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let live = format!("{C1}{PARTIAL_SUFFIX}");
        let orphan = format!("{C2}{PARTIAL_SUFFIX}");
        for name in [&live, &orphan] {
            local_complete_set(&results, name);
        }
        local_complete_set(&results, "2026-10-22-geforce-rtx-3090-qwen3-4b-012317-r1");

        let removed = remove_orphan_partials(&results, &dirs(&[R1])).unwrap();
        assert_eq!(removed, std::slice::from_ref(&orphan));
        assert!(
            results.join(&live).is_dir(),
            "the active run's mirror stays"
        );
        assert!(!results.join(&orphan).exists());
        assert!(
            results
                .join("2026-10-22-geforce-rtx-3090-qwen3-4b-012317-r1")
                .is_dir(),
            "a complete set is never touched"
        );
        assert_eq!(remove_orphan_partials(&results, &[]).unwrap(), [live]);
    }
    /// A run of one repeat names its set without a repeat suffix; it is repeat 1 and is committed under the
    /// canonical `-r1` name. Mutation: require the `-r<N>` suffix and the first assertion goes red.
    #[test]
    fn a_one_repeat_set_without_a_suffix_is_repeat_one() {
        const BARE: &str = "run-1792676008-1712166-qwen3.8-flash-next-q2";
        assert_eq!(
            canonical_dir(
                BARE,
                Some("NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
            )
            .unwrap(),
            "2026-10-22-rtx-pro-6000-blackwell-workstation-edition-qwen3-8-flash-next-q2-133328-r1"
        );
        // A model whose id merely contains "-r" is not cut there.
        let set = PodSet::parse("run-1792676008-1712166-llama-rerank").unwrap();
        assert_eq!((set.model.as_str(), set.repeat), ("llama-rerank", 1));
        let set = PodSet::parse("run-1792676008-1712166-llama-rerank-r3").unwrap();
        assert_eq!((set.model.as_str(), set.repeat), ("llama-rerank", 3));
    }

    /// The bare one-repeat set is gathered whole, under the canonical -r1 name, and then counts as present.
    #[test]
    fn a_one_repeat_run_gathers_under_the_canonical_r1_name() {
        const BARE: &str = "run-1792632197-3202502-qwen3-0.6b";
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let pod = FakePod::new();
        pod.finish(BARE);

        let got = pass(&pod, tmp.path(), &[BARE], &accept);

        assert_eq!(got.copied, [BARE]);
        assert!(results.join(C1).join("report.md").is_file());
        assert!(set_is_complete(&results, BARE));
        assert_eq!(repeats_present(&results, &dirs(&[BARE])), [1]);
    }

    /// A set name this side cannot parse is a local fault, not a pod problem: gathering it reports a
    /// `LocalFault` and copies nothing into `results`.
    #[test]
    fn a_set_name_that_cannot_be_parsed_is_a_local_fault() {
        let tmp = TempDir::new();
        let pod = FakePod::new();
        pod.finish("not-a-run-set");
        let error = gather_sets(
            &pod,
            &tmp.path().join("results"),
            &tmp.path().join("incoming"),
            &tmp.path().join("rejected"),
            &dirs(&["not-a-run-set"]),
            &accept,
        );
        // The listing found it complete and the copy came down; placing it is what fails, as a LocalFault.
        let error = error.unwrap_err();
        assert!(error.downcast_ref::<LocalFault>().is_some(), "{error:#}");
        assert!(
            !tmp.path().join("results").exists()
                || partials(&tmp.path().join("results")).is_empty()
        );
    }

    /// The env.json of a pod whose GPU is gone: `bench` captured no GPU name.
    const NO_GPU_ENV: &str = r#"{"gpu_name":null,"gpu_driver":null}"#;
    const NO_DEVICE_ROW: &str = r#"{"framework":"candle","status":"error","reason":"rc=1: stderr: Error: DriverError(CUDA_ERROR_NO_DEVICE, \"no CUDA-capable device is detected\")"}"#;

    /// A finished set the pod wrote after its GPU died.
    fn finish_dead(pod: &FakePod, dir: &str, env: &str) {
        pod.finish(dir);
        pod.put(dir, "env.json", env);
        pod.put(dir, "results.jsonl", &format!("{NO_DEVICE_ROW}\n{OK_ROW}"));
    }

    /// A set from a pod with no GPU (env.json lists none) or whose rows errored with NO_DEVICE is a host fault on
    /// first sight: the pass fails with a `HostFault`, the set reaches neither `results/` nor the failed sets, and
    /// the repeat is not counted. Mutation: drop the `host_fault_of` check in `gather_sets` and the set is
    /// gathered (the llama.cpp row says ok), so the error assertion goes red.
    #[test]
    fn a_set_from_a_pod_with_a_dead_gpu_is_a_host_fault_and_is_not_gathered() {
        for (env, why) in [
            (NO_GPU_ENV, "env.json lists no GPU"),
            (
                r#"{"gpu_name":"NVIDIA GeForce RTX 3090"}"#,
                "NO_DEVICE rows",
            ),
        ] {
            let tmp = TempDir::new();
            let results = tmp.path().join("results");
            let pod = FakePod::new();
            finish_dead(&pod, R1, env);

            let error = gather_sets(
                &pod,
                &results,
                &tmp.path().join("incoming"),
                &tmp.path().join("rejected"),
                &dirs(&[R1]),
                &accept,
            )
            .unwrap_err();

            assert!(
                error.downcast_ref::<HostFault>().is_some(),
                "{why}: {error:#}"
            );
            assert!(!set_is_complete(&results, R1), "{why}");
            assert!(repeats_present(&results, &dirs(&[R1])).is_empty(), "{why}");
            assert!(
                !failed_dir(&results).exists(),
                "{why}: not even a failed set"
            );
            assert!(
                tmp.path().join("rejected").join(R1).is_dir(),
                "{why}: kept as evidence"
            );
        }
    }

    /// A pod's GPU dies after repeat 4 of 6. Repeats 1-4 are gathered; the dead repeat 5 is a host fault, so
    /// supervision replaces the pod at repeat 5 and the new pod runs only 5 and 6. Nothing from the dead state is
    /// gathered and the count of gathered repeats is honest at every step.
    /// Mutation: skip the `host_fault_of` check and repeat 5 is gathered as a result, so the second pod runs only
    /// repeat 6 and `ran` differs.
    #[test]
    fn a_gpu_that_dies_after_repeat_four_replaces_the_pod_at_repeat_five() {
        use crate::heal::{Cursor, Healing, Outcome, Replacement, supervise};
        use std::time::Duration;

        struct Sim<'a> {
            root: &'a Path,
            wanted: Vec<String>,
            pods: u32,
            ran: Vec<Vec<u32>>,
            replaced: Vec<Replacement>,
            gathered_at_failure: Vec<usize>,
        }
        impl Healing for Sim<'_> {
            fn attempt(&mut self, _fresh: bool) -> Result<()> {
                let results = self.root.join("results");
                let first_pod = self.pods == 0;
                self.pods += 1;
                let pod = FakePod::new();
                let todo: Vec<u32> = (1..=6)
                    .filter(|n| !repeats_present(&results, &self.wanted).contains(n))
                    .collect();
                self.ran.push(todo.clone());
                for n in todo {
                    let dir = &self.wanted[n as usize - 1];
                    // The GPU of the first pod is gone from repeat 5 on; the second pod is healthy.
                    if first_pod && n >= 5 {
                        finish_dead(&pod, dir, NO_GPU_ENV);
                    } else {
                        pod.finish(dir);
                    }
                    let wanted = self.wanted.clone();
                    let pass = gather_sets(
                        &pod,
                        &results,
                        &self.root.join("incoming"),
                        &self.root.join("rejected"),
                        &wanted,
                        &accept,
                    );
                    if let Err(e) = pass {
                        self.gathered_at_failure
                            .push(repeats_present(&results, &wanted).len());
                        return Err(e);
                    }
                }
                Ok(())
            }
            fn pod_reachable(&mut self) -> bool {
                true
            }
            fn discard_pod(&mut self, why: Replacement) {
                self.replaced.push(why);
            }
            fn reset_setup(&mut self, _: Option<&str>) -> Result<()> {
                Ok(())
            }
            fn skip_cell(&mut self, _: Option<&str>, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn pause(&mut self, _: Duration) {}
        }

        let tmp = TempDir::new();
        let wanted: Vec<String> = (1..=6)
            .map(|n| format!("run-1792632197-3202502-qwen3-0.6b-r{n}"))
            .collect();
        let mut sim = Sim {
            root: tmp.path(),
            wanted: wanted.clone(),
            pods: 0,
            ran: Vec::new(),
            replaced: Vec::new(),
            gathered_at_failure: Vec::new(),
        };

        let outcome = supervise(&mut sim, &Cursor::default(), None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert_eq!(sim.ran, [vec![1, 2, 3, 4, 5, 6], vec![5, 6]]);
        assert_eq!(sim.replaced, [Replacement::HostFault]);
        assert_eq!(
            sim.gathered_at_failure,
            [4],
            "four valid sets, the fifth not counted"
        );
        let results = tmp.path().join("results");
        assert_eq!(repeats_present(&results, &wanted), [1, 2, 3, 4, 5, 6]);
        assert!(
            !failed_dir(&results).exists(),
            "nothing recorded as a failed set"
        );
        for n in 1..=6 {
            let set = local_set(&results, &wanted[n - 1]).unwrap();
            assert!(!set.contains("nogpu"), "{set}");
        }
    }

    /// The probe: a pod whose shell finds no GPU is a host fault; an unreachable pod is not judged by it.
    /// Mutation: return Ok for a failed listing and the first assertion goes red.
    #[test]
    fn the_gpu_probe_faults_a_pod_that_lists_none_and_ignores_an_unreachable_one() {
        struct Probe(i32);
        impl PodFiles for Probe {
            fn ssh(&self, _: &str) -> Result<ssh::Run> {
                Ok(ssh::Run {
                    code: self.0,
                    stdout: String::new(),
                    stderr: String::new(),
                })
            }
            fn scp_down(&self, _: &str, _: &str) -> Result<()> {
                unreachable!("the probe copies nothing")
            }
        }
        let fault = crate::heal::require_gpu(&Probe(1)).unwrap_err();
        assert!(fault.downcast_ref::<HostFault>().is_some(), "{fault:#}");
        assert!(crate::heal::require_gpu(&Probe(0)).is_ok());
        assert!(crate::heal::require_gpu(&Probe(ssh::SSH_UNREACHABLE)).is_ok());
    }

    const MISSING_MODEL_ROWS: &str = r#"{"framework":"poot","status":"error","reason":"rc=1: stderr: Error: load /root/models/qwen3-4b |  | Caused by: |     0: /root/models/qwen3-4b: No such file or directory (os error 2)"}
{"framework":"llamacpp","status":"error","reason":"invalid runner result: curve must be a non-empty array"}"#;

    /// A finished repeat in which nothing measured and a row says the model directory is absent.
    fn finish_missing_model(pod: &FakePod, dir: &str) {
        pod.finish(dir);
        pod.put(dir, "results.jsonl", MISSING_MODEL_ROWS);
    }

    /// A repeat whose rows all failed because the model is not on the pod is a setup fault on first sight: not a
    /// result, not a failed set, quarantined. A repeat that failed for its own reason ("boom") still is a failed set.
    /// Mutation: drop the `setup_fault_of` check in `gather_sets` and the set lands in failed-results.
    #[test]
    fn a_repeat_whose_model_is_missing_from_the_pod_is_a_setup_fault_not_a_failed_set() {
        let tmp = TempDir::new();
        let results = tmp.path().join("results");
        let pod = FakePod::new();
        finish_missing_model(&pod, R1);

        let error = gather_sets(
            &pod,
            &results,
            &tmp.path().join("incoming"),
            &tmp.path().join("rejected"),
            &dirs(&[R1]),
            &accept,
        )
        .unwrap_err();

        assert!(error.downcast_ref::<SetupFault>().is_some(), "{error:#}");
        assert!(!set_is_complete(&results, R1));
        assert!(!failed_dir(&results).exists());
        assert!(tmp.path().join("rejected").join(R1).is_dir());
    }

    /// The qwen3-4b incident: a pod replaced mid-run has none of the models. The first repeat's sets say so, the
    /// supervisor sets the model up again once, and the repeats then run to the end: no run of failed repeats.
    /// A model whose second setup fails the same way is failed with its reason (see the heal test).
    /// Mutation: `reset_setup` that does not fetch the model leaves the fault in place and the run never ends in
    /// valid sets (all repeats of the third attempt are missing again).
    #[test]
    fn a_pod_replaced_without_its_models_sets_them_up_again_instead_of_failing_every_repeat() {
        use crate::heal::{Cursor, Healing, Outcome, Replacement, supervise};
        use std::cell::Cell;
        use std::time::Duration;

        struct Sim<'a> {
            root: &'a Path,
            wanted: Vec<String>,
            model_on_pod: Cell<bool>,
            ran: Vec<Vec<u32>>,
            resets: u32,
        }
        impl Healing for Sim<'_> {
            fn attempt(&mut self, _fresh: bool) -> Result<()> {
                let results = self.root.join("results");
                let pod = FakePod::new();
                let todo: Vec<u32> = (1..=3)
                    .filter(|n| !repeats_present(&results, &self.wanted).contains(n))
                    .collect();
                self.ran.push(todo.clone());
                for n in todo {
                    let dir = &self.wanted[n as usize - 1];
                    if self.model_on_pod.get() {
                        pod.finish(dir);
                    } else {
                        finish_missing_model(&pod, dir);
                    }
                    gather_sets(
                        &pod,
                        &results,
                        &self.root.join("incoming"),
                        &self.root.join("rejected"),
                        &self.wanted,
                        &accept,
                    )?;
                }
                Ok(())
            }
            fn pod_reachable(&mut self) -> bool {
                true
            }
            fn discard_pod(&mut self, _: Replacement) {}
            fn reset_setup(&mut self, _: Option<&str>) -> Result<()> {
                // The setup runs again: the model is fetched onto the pod.
                self.resets += 1;
                self.model_on_pod.set(true);
                Ok(())
            }
            fn skip_cell(&mut self, _: Option<&str>, _: &str, _: &str) -> Result<()> {
                unreachable!("the second setup works")
            }
            fn pause(&mut self, _: Duration) {}
        }

        let tmp = TempDir::new();
        let wanted: Vec<String> = (1..=3)
            .map(|n| format!("run-1792632197-3202502-qwen3-4b-r{n}"))
            .collect();
        let mut sim = Sim {
            root: tmp.path(),
            wanted: wanted.clone(),
            model_on_pod: Cell::new(false),
            ran: Vec::new(),
            resets: 0,
        };
        let cursor = Cursor::default();
        cursor.set(Some("qwen3-4b"), "sweep");

        let outcome = supervise(&mut sim, &cursor, None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert_eq!(sim.resets, 1, "set up again once");
        assert_eq!(
            sim.ran,
            [vec![1, 2, 3], vec![1, 2, 3]],
            "nothing counted from the first pod"
        );
        let results = tmp.path().join("results");
        assert_eq!(repeats_present(&results, &wanted), [1, 2, 3]);
        assert!(!failed_dir(&results).exists());
    }
}

//! Self-healing supervision of a `run`: infrastructure trouble (a lost pod, an unreachable host, a crashed sweep)
//! replaces the pod and continues with the cells still missing; only the user, or an error that waiting cannot
//! fix, ends a run early.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::Duration;

use anyhow::Result;

use crate::gather::PodFiles;
use crate::logging::log;
use crate::ssh::SSH_UNREACHABLE;

/// How many times in a row the same cell may fail the same way before it is recorded as failed and skipped.
pub(crate) const IDENTICAL_FAILURES_LIMIT: u32 = 3;

/// The pod is gone, or unreachable beyond the grace window: replace it.
#[derive(Debug, thiserror::Error)]
#[error("pod {pod_id} lost: {reason}")]
pub(crate) struct PodLost {
    pub(crate) pod_id: String,
    pub(crate) reason: &'static str,
}

/// A failure on THIS machine (a name the orchestrator cannot parse, a file it cannot write), not the pod's: the pod
/// is fine and may hold the only copy of the results. It is never retried, counted toward the identical failures or
/// answered with a replacement: the run stops, and the pod is kept so the results can be pulled by hand.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct LocalFault(pub(crate) String);

/// The host cannot run CUDA for this image (its driver is older than the image's runtime, or it has no usable GPU).
/// That is the pod's fault, not the work's: the pod is replaced and the failure counts neither as a failed cell nor
/// toward [`IDENTICAL_FAILURES_LIMIT`].
#[derive(Debug, thiserror::Error)]
#[error("host CUDA fault: {0}")]
pub(crate) struct HostFault(pub(crate) String);

/// Error text of a CUDA driver/runtime incompatibility or a host with no usable GPU, lower-cased. A cell's own
/// failure (a model that does not load, an out-of-memory) is none of these.
const HOST_FAULT_MARKERS: [&str; 7] = [
    "cuda driver version is insufficient for cuda runtime version",
    "cudaerrorinsufficientdriver",
    "cuda_error_insufficient_driver",
    "cuinit failed",
    "cuda_error_no_device",
    "no cuda-capable device is detected",
    "unsatisfied condition: cuda>=",
];

/// Is `text` (a failure message, a log tail) a host CUDA fault? Returns the marker it matched.
pub(crate) fn host_fault_in(text: &str) -> Option<HostFault> {
    let lower = text.to_lowercase();
    HOST_FAULT_MARKERS
        .iter()
        .find(|marker| lower.contains(*marker))
        .map(|marker| HostFault((*marker).to_string()))
}

/// The shell command that succeeds when the pod lists a GPU: `nvidia-smi -L` names one, or (AMD pods have no
/// nvidia-smi) `rocminfo` lists a GPU agent.
const GPU_LISTED_COMMAND: &str = "nvidia-smi -L 2>/dev/null | grep -q '^GPU ' || rocminfo 2>/dev/null | grep -q 'Device Type:.*GPU'";

/// The cheap probe run before a sweep starts and while it runs: a pod whose GPU is gone is a host fault now,
/// not after the harness has swept a repeat of errors. A pod that does not answer ssh is not judged here (that is
/// [`PodLost`]'s business).
pub(crate) fn require_gpu(pod: &dyn PodFiles) -> Result<()> {
    let listed = pod.ssh(GPU_LISTED_COMMAND)?;
    if listed.code == SSH_UNREACHABLE || listed.ok() {
        return Ok(());
    }
    Err(HostFault("no CUDA-capable device is detected: the pod lists no GPU".into()).into())
}

/// Why a pod is being replaced; the dashboard says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Replacement {
    /// The pod vanished or stopped answering.
    Lost,
    /// The pod runs but its GPU does not work ([`HostFault`]).
    HostFault,
}

/// A repeat in which no row measured anything and a row says the model's files are not on the pod: the pod was
/// not (fully) set up. Repeating the sweep cannot fix it; running the model's setup again can.
#[derive(Debug, thiserror::Error)]
#[error("setup fault: {0}")]
pub(crate) struct SetupFault(pub(crate) String);

/// Error text of a row whose model is not on the pod, lower-cased. A cell's own failure is none of these.
const SETUP_FAULT_MARKERS: [&str; 2] = ["no such file or directory", "failed to load model"];

/// Is `text` (a row's error) a missing-model error? Returns the marker it matched. A missing file only counts when
/// the text names a path under a `models/` directory (every runner prints the model path it failed to open), so a
/// missing runner binary or harness file is not taken for a missing model.
pub(crate) fn setup_fault_in(text: &str) -> Option<SetupFault> {
    let lower = text.to_lowercase();
    if !lower.contains("models/") {
        return None;
    }
    SETUP_FAULT_MARKERS
        .iter()
        .find(|marker| lower.contains(*marker))
        .map(|marker| SetupFault((*marker).to_string()))
}

/// Waiting cannot fix this (a bad GPU type or cloud value, auth, an unknown image): the run fails now.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct Unrecoverable(pub(crate) String);

/// The user's bound was reached (`--give-up-after-min`, `--max-replacements`): the run stops as cancelled.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct Cancelled(pub(crate) String);

/// Where a pod's work was when it failed: the model and phase the failure belongs to.
#[derive(Default)]
pub(crate) struct Cursor {
    model: RefCell<Option<String>>,
    phase: Cell<&'static str>,
}

impl Cursor {
    pub(crate) fn set(&self, model: Option<&str>, phase: &'static str) {
        *self.model.borrow_mut() = model.map(String::from);
        self.phase.set(phase);
    }
    pub(crate) fn model(&self) -> Option<String> {
        self.model.borrow().clone()
    }
    pub(crate) fn phase(&self) -> &'static str {
        self.phase.get()
    }
}

/// How a supervised run ended without an unrecoverable error.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Outcome {
    Done,
    Cancelled(String),
}

/// What supervision needs from the run it drives.
pub(crate) trait Healing {
    /// Provision (when `fresh_pod`, or when there is no pod), set up, sweep and gather what is missing.
    fn attempt(&mut self, fresh_pod: bool) -> Result<()>;
    /// Is the current pod alive and answering ssh? Waits out transient trouble for the grace window first.
    fn pod_reachable(&mut self) -> bool;
    /// Terminate and forget the current pod.
    fn discard_pod(&mut self, why: Replacement);
    /// Forget that the pod was set up for `model` and stop its sweep, so the next attempt fetches the model again.
    fn reset_setup(&mut self, model: Option<&str>) -> Result<()>;
    /// Record the cell (`model`, `phase`) as failed and skip it from now on.
    fn skip_cell(&mut self, model: Option<&str>, phase: &str, reason: &str) -> Result<()>;
    fn pause(&mut self, d: Duration);
}

/// What identifies a failure for the identical-failure count: where it happened and what kind of failure it was
/// (pod ids and numbers are not part of it).
fn failure_key(cursor: &Cursor, error: &anyhow::Error) -> String {
    let kind = match error.downcast_ref::<PodLost>() {
        Some(lost) => lost.reason.to_string(),
        None => {
            let root = error.root_cause().to_string();
            let line = root.lines().next().unwrap_or("");
            line.chars()
                .filter(|c| !c.is_ascii_digit())
                .take(80)
                .collect()
        }
    };
    format!(
        "{}|{}|{kind}",
        cursor.model().as_deref().unwrap_or("-"),
        cursor.phase()
    )
}

/// Drive `run` to the end. A recoverable failure replaces the pod when it is lost and retries on it when it is
/// not; the same failure of the same cell [`IDENTICAL_FAILURES_LIMIT`] times in a row skips that cell.
pub(crate) fn supervise(
    run: &mut dyn Healing,
    cursor: &Cursor,
    max_replacements: Option<u32>,
) -> Result<Outcome> {
    let mut fresh = false;
    let mut replacements = 0u32;
    let mut last: Option<String> = None;
    let mut same = 0u32;
    let mut setups_redone: Vec<Option<String>> = Vec::new();
    loop {
        let error = match run.attempt(fresh) {
            Ok(()) => return Ok(Outcome::Done),
            Err(e) => e,
        };
        if error.downcast_ref::<Unrecoverable>().is_some() {
            return Err(error);
        }
        if error.downcast_ref::<LocalFault>().is_some() {
            return Err(error);
        }
        if let Some(cancelled) = error.downcast_ref::<Cancelled>() {
            return Ok(Outcome::Cancelled(cancelled.0.clone()));
        }
        // A model missing from the pod: set it up again once, on this pod; a second time the model is failed with the
        // reason rather than sweeping repeat after repeat of errors.
        if let Some(fault) = error.downcast_ref::<SetupFault>() {
            let model = cursor.model();
            if setups_redone.contains(&model) {
                log(format!(
                    "setup fault again after setting up again ({}); the model is failed: {error:#}",
                    fault.0
                ));
                run.skip_cell(model.as_deref(), "setup", &format!("{error:#}"))?;
            } else {
                log(format!(
                    "setup fault ({}); setting the model up again on this pod: {error:#}",
                    fault.0
                ));
                run.reset_setup(model.as_deref())?;
                setups_redone.push(model);
            }
            fresh = false;
            continue;
        }
        // A host that cannot run CUDA is replaced, whatever cell it was on; the cell is not blamed.
        let host_fault = error
            .downcast_ref::<HostFault>()
            .map(|f| f.0.clone())
            .or_else(|| host_fault_in(&format!("{error:#}")).map(|f| f.0));
        if let Some(marker) = host_fault {
            log(format!(
                "host CUDA fault ({marker}); replacing the pod, the cell is not counted as failed: {error:#}"
            ));
            replacements += 1;
            if max_replacements.is_some_and(|max| replacements > max) {
                return Ok(Outcome::Cancelled(format!(
                    "--max-replacements {} reached",
                    max_replacements.unwrap_or(0)
                )));
            }
            run.discard_pod(Replacement::HostFault);
            // The new pod has none of the models: each gets its one setup retry again.
            setups_redone.clear();
            fresh = true;
            continue;
        }
        let key = failure_key(cursor, &error);
        same = if last.as_deref() == Some(key.as_str()) {
            same + 1
        } else {
            1
        };
        last = Some(key);
        log(format!(
            "recoverable failure ({same}/{IDENTICAL_FAILURES_LIMIT} identical): {error:#}"
        ));
        if same >= IDENTICAL_FAILURES_LIMIT {
            run.skip_cell(
                cursor.model().as_deref(),
                cursor.phase(),
                &format!("{error:#}"),
            )?;
            same = 0;
            last = None;
        }
        let lost = error.downcast_ref::<PodLost>().is_some() || !run.pod_reachable();
        if lost {
            replacements += 1;
            if max_replacements.is_some_and(|max| replacements > max) {
                return Ok(Outcome::Cancelled(format!(
                    "--max-replacements {} reached",
                    max_replacements.unwrap_or(0)
                )));
            }
            run.discard_pod(Replacement::Lost);
            setups_redone.clear();
            fresh = true;
        } else {
            // The pod answers: the failure was transient or the cell's own. Try again on it after a pause.
            run.pause(Duration::from_secs(30));
            fresh = false;
        }
    }
}

/// Write `dir` as a complete result set holding one failed row, so a later pod skips the repeat and the set
/// stays complete: the cell failed the same way on [`IDENTICAL_FAILURES_LIMIT`] fresh pods.
pub(crate) fn record_failed_cell(
    results: &Path,
    dir: &str,
    model: &str,
    scenario: &str,
    reason: &str,
) -> Result<()> {
    // Its only row is an error, so it is a failed set (see `gather::failed_dir`).
    let set = crate::gather::failed_dir(results).join(crate::gather::canonical_dir(dir, None)?);
    std::fs::create_dir_all(&set)?;
    let row = serde_json::json!({
        "framework": "poot",
        "model": model,
        "scenario": scenario,
        "status": "error",
        "reason": format!(
            "skipped by the orchestrator after {IDENTICAL_FAILURES_LIMIT} identical failures on fresh pods: {reason}"
        ),
    });
    std::fs::write(set.join("results.jsonl"), format!("{row}\n"))?;
    std::fs::write(
        set.join("env.json"),
        serde_json::json!({ "note": "recorded by the orchestrator, not measured" }).to_string(),
    )?;
    std::fs::write(
        set.join("report.md"),
        format!("{model} / {scenario}: failed, skipped after repeated identical failures.\n"),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A run scripted by the outcome of each attempt; records what supervision asked of it.
    struct Script {
        attempts: VecDeque<Result<()>>,
        reachable: bool,
        calls: Vec<String>,
    }

    impl Script {
        fn new(attempts: Vec<Result<()>>, reachable: bool) -> Self {
            Self {
                attempts: attempts.into(),
                reachable,
                calls: Vec::new(),
            }
        }
    }

    impl Healing for Script {
        fn attempt(&mut self, fresh: bool) -> Result<()> {
            self.calls.push(format!("attempt(fresh={fresh})"));
            self.attempts.pop_front().expect("scripted attempt")
        }
        fn pod_reachable(&mut self) -> bool {
            self.calls.push("probe".into());
            self.reachable
        }
        fn discard_pod(&mut self, why: Replacement) {
            self.calls.push(match why {
                Replacement::Lost => "discard".into(),
                Replacement::HostFault => "discard(host fault)".into(),
            });
        }
        fn skip_cell(&mut self, model: Option<&str>, phase: &str, _reason: &str) -> Result<()> {
            self.calls
                .push(format!("skip({},{phase})", model.unwrap_or("-")));
            Ok(())
        }
        fn pause(&mut self, _d: Duration) {
            self.calls.push("pause".into());
        }
        fn reset_setup(&mut self, model: Option<&str>) -> Result<()> {
            self.calls
                .push(format!("reset_setup({})", model.unwrap_or("-")));
            Ok(())
        }
    }

    fn lost(pod: &str) -> anyhow::Error {
        PodLost {
            pod_id: pod.into(),
            reason: "not RUNNING per the RunPod API",
        }
        .into()
    }

    fn cursor(model: &str, phase: &'static str) -> Cursor {
        let c = Cursor::default();
        c.set(Some(model), phase);
        c
    }

    /// A pod lost mid-sweep is replaced and the run goes on to the end: it is never a failure.
    /// Mutation: return the error instead of replacing (abort on pod loss) and this goes red.
    #[test]
    fn a_lost_pod_is_replaced_and_the_run_completes() {
        let mut run = Script::new(vec![Err(lost("pod-1")), Ok(())], true);

        let outcome = supervise(&mut run, &cursor("m", "sweep"), None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert_eq!(
            run.calls,
            ["attempt(fresh=false)", "discard", "attempt(fresh=true)"]
        );
    }

    /// The same cell failing the same way on three fresh pods is skipped, then the run completes.
    /// Mutation: no cap on identical failures (never skip) and the scripted fourth attempt is never reached.
    #[test]
    fn a_cell_failing_identically_three_times_is_skipped_and_the_run_completes() {
        let mut run = Script::new(
            vec![Err(lost("p1")), Err(lost("p2")), Err(lost("p3")), Ok(())],
            true,
        );

        let outcome = supervise(&mut run, &cursor("m", "sweep"), None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert_eq!(
            run.calls.iter().filter(|c| c.starts_with("skip")).count(),
            1,
            "{:?}",
            run.calls
        );
        assert!(run.calls.contains(&"skip(m,sweep)".to_string()));
        let skip_at = run
            .calls
            .iter()
            .position(|c| c.starts_with("skip"))
            .unwrap();
        let attempts_before = run.calls[..skip_at]
            .iter()
            .filter(|c| c.starts_with("attempt"))
            .count();
        assert_eq!(attempts_before, 3);
    }

    /// Different failures do not add up to the cap.
    #[test]
    fn different_failures_are_not_counted_together() {
        let mut run = Script::new(
            vec![
                Err(lost("p1")),
                Err(anyhow::anyhow!("scp failed")),
                Err(lost("p3")),
                Ok(()),
            ],
            false,
        );

        supervise(&mut run, &cursor("m", "sweep"), None).unwrap();

        assert!(
            !run.calls.iter().any(|c| c.starts_with("skip")),
            "{:?}",
            run.calls
        );
    }

    /// A non-recoverable error ends the run at once, with its reason.
    #[test]
    fn an_unrecoverable_error_fails_at_once() {
        let mut run = Script::new(
            vec![Err(Unrecoverable(
                "create_pod failed (400): bad gpu type".into(),
            )
            .into())],
            true,
        );

        let error = supervise(&mut run, &cursor("-", "provision"), None).unwrap_err();

        assert!(format!("{error:#}").contains("bad gpu type"));
        assert_eq!(run.calls, ["attempt(fresh=false)"]);
    }

    /// A user bound ends the run as cancelled, not failed.
    #[test]
    fn a_users_bound_cancels() {
        let mut run = Script::new(
            vec![Err(Cancelled("gave up after 5 min".into()).into())],
            true,
        );

        let outcome = supervise(&mut run, &cursor("-", "provision"), None).unwrap();

        assert_eq!(outcome, Outcome::Cancelled("gave up after 5 min".into()));
    }

    /// A failure on a pod that still answers retries on it (no replacement); an unreachable pod is replaced.
    #[test]
    fn a_reachable_pod_is_retried_and_an_unreachable_one_replaced() {
        let mut reachable = Script::new(vec![Err(anyhow::anyhow!("ssh hiccup")), Ok(())], true);
        supervise(&mut reachable, &cursor("m", "setup"), None).unwrap();
        assert_eq!(
            reachable.calls,
            [
                "attempt(fresh=false)",
                "probe",
                "pause",
                "attempt(fresh=false)"
            ]
        );

        let mut gone = Script::new(vec![Err(anyhow::anyhow!("ssh hiccup")), Ok(())], false);
        supervise(&mut gone, &cursor("m", "setup"), None).unwrap();
        assert_eq!(
            gone.calls,
            [
                "attempt(fresh=false)",
                "probe",
                "discard",
                "attempt(fresh=true)"
            ]
        );
    }

    /// `--max-replacements` bounds the replacements and ends the run as cancelled.
    #[test]
    fn max_replacements_cancels() {
        let mut run = Script::new(vec![Err(lost("p1")), Err(lost("p2"))], true);

        let outcome = supervise(&mut run, &cursor("m", "sweep"), Some(1)).unwrap();

        assert!(matches!(outcome, Outcome::Cancelled(_)), "{outcome:?}");
    }
    /// Each known driver/runtime incompatibility text is a host fault; an ordinary model-load error and a cell's own
    /// out-of-memory are not. Mutation: drop a marker from `HOST_FAULT_MARKERS` and its row goes red; match
    /// "cuda" alone and the last rows go red.
    #[test]
    fn the_known_cuda_driver_failures_are_host_faults_and_a_model_error_is_not() {
        for text in [
            "RuntimeError: CUDA driver version is insufficient for CUDA runtime version",
            "cudaErrorInsufficientDriver: CUDA driver version is insufficient",
            "CUDA_ERROR_INSUFFICIENT_DRIVER",
            "ptx runtime: cuInit failed: CUDA_ERROR_NO_DEVICE",
            "DriverError(CUDA_ERROR_NO_DEVICE)",
            "RuntimeError: No CUDA-capable device is detected",
            "nvidia-container-cli: requirement error: unsatisfied condition: cuda>=13.0",
        ] {
            assert!(host_fault_in(text).is_some(), "{text}");
        }
        for text in [
            "Error: failed to load model /root/models/x.gguf: invalid magic",
            "CUDA out of memory. Tried to allocate 2.00 GiB",
            "alloc_storage element count 7200034561 exceeds the native u32 kernel-length ABI",
            "cuda 13.0 toolkit found",
            "",
        ] {
            assert!(host_fault_in(text).is_none(), "{text}");
        }
    }

    /// A host that cannot run CUDA replaces the pod and is not a failed cell: four of them in a row do not skip the
    /// cell (the cap is three identical failures), and the run completes. Mutation: count a host fault like any
    /// other failure and the third one skips the cell.
    #[test]
    fn a_host_cuda_fault_replaces_the_pod_without_counting_a_cell_failure() {
        let fault = |n: u32| {
            anyhow::anyhow!(
                "sweep FAILED (SWEEP_EXIT=1); last log lines:\nCUDA driver version is insufficient for CUDA runtime version (pod {n})"
            )
        };
        let mut run = Script::new(
            vec![
                Err(fault(1)),
                Err(fault(2)),
                Err(fault(3)),
                Err(fault(4)),
                Ok(()),
            ],
            true,
        );

        let outcome = supervise(&mut run, &cursor("m", "sweep"), None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert!(
            run.calls.iter().all(|c| !c.starts_with("skip")),
            "a bad host is not the cell's failure: {:?}",
            run.calls
        );
        assert_eq!(
            run.calls
                .iter()
                .filter(|c| *c == "discard(host fault)")
                .count(),
            4
        );
        assert!(
            !run.calls.contains(&"probe".to_string()),
            "no reachability probe: the pod is replaced"
        );
        // The typed error is honoured too, whatever its text says.
        let mut typed = Script::new(vec![Err(HostFault("x".into()).into()), Ok(())], true);
        assert_eq!(
            supervise(&mut typed, &cursor("m", "setup"), None).unwrap(),
            Outcome::Done
        );
        assert_eq!(
            typed.calls,
            [
                "attempt(fresh=false)",
                "discard(host fault)",
                "attempt(fresh=true)"
            ]
        );
    }

    /// A host fault is still a replacement: `--max-replacements` bounds it.
    #[test]
    fn a_host_cuda_fault_counts_toward_max_replacements() {
        let fault = || anyhow::Error::from(HostFault("cuinit failed".into()));
        let mut run = Script::new(vec![Err(fault()), Err(fault()), Ok(())], true);
        let outcome = supervise(&mut run, &cursor("m", "sweep"), Some(1)).unwrap();
        assert!(matches!(outcome, Outcome::Cancelled(_)));
    }
    /// A model missing from the pod is set up again once on the same pod (no replacement, no failure count); the
    /// second time the model is failed with the reason, and the run goes on. Mutation: drop the `setups_redone`
    /// check and the second fault resets forever (the scripted third attempt is never reached).
    #[test]
    fn a_setup_fault_redoes_the_setup_once_and_then_fails_the_model() {
        let fault = || anyhow::Error::from(SetupFault("no such file or directory".into()));
        let mut run = Script::new(vec![Err(fault()), Err(fault()), Ok(())], true);

        let outcome = supervise(&mut run, &cursor("qwen3-4b", "sweep"), None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert_eq!(
            run.calls,
            [
                "attempt(fresh=false)",
                "reset_setup(qwen3-4b)",
                "attempt(fresh=false)",
                "skip(qwen3-4b,setup)",
                "attempt(fresh=false)"
            ]
        );
    }

    /// A replacement pod has none of the models, so a setup fault after it gets its own one retry. Mutation: drop
    /// `setups_redone.clear()` after the replacement and the second fault fails the model at once.
    #[test]
    fn a_replacement_pod_gives_a_model_its_setup_retry_again() {
        let setup = || anyhow::Error::from(SetupFault("no such file or directory".into()));
        let mut run = Script::new(
            vec![Err(setup()), Err(lost("p1")), Err(setup()), Ok(())],
            true,
        );

        supervise(&mut run, &cursor("qwen3-4b", "sweep"), None).unwrap();

        assert!(
            run.calls.iter().all(|c| !c.starts_with("skip")),
            "{:?}",
            run.calls
        );
        assert_eq!(
            run.calls
                .iter()
                .filter(|c| c.starts_with("reset_setup"))
                .count(),
            2
        );
    }

    /// A missing file is a missing model only when it names a path under `models/`: a missing runner binary or
    /// harness file is not. Mutation: drop the `models/` condition in `setup_fault_in` and the second row goes red.
    #[test]
    fn only_a_missing_model_path_is_a_setup_fault() {
        assert!(
            setup_fault_in("load /root/models/qwen3-4b: No such file or directory (os error 2)")
                .is_some()
        );
        assert!(setup_fault_in("FileNotFoundError: [Errno 2] No such file or directory: '/root/models/q/config.json'").is_some());
        assert!(
            setup_fault_in("spawn /root/poot-bench-runner: No such file or directory (os error 2)")
                .is_none()
        );
        assert!(setup_fault_in("CUDA out of memory").is_none());
    }

    /// A failure of this machine (an unparseable set name) stops the run at once: no retry, no replacement, no
    /// probe, no pause, no failure count. Mutation: let it fall into the recoverable path and the discard appears.
    #[test]
    fn a_local_fault_stops_the_run_without_touching_the_pod() {
        let fault = anyhow::Error::from(LocalFault(
            "set x is not named run-<epoch>-<pid>-<model>".into(),
        ));
        let mut run = Script::new(vec![Err(fault)], true);

        let error = supervise(&mut run, &cursor("m", "gather"), None).unwrap_err();

        assert!(error.downcast_ref::<LocalFault>().is_some());
        assert_eq!(
            run.calls,
            ["attempt(fresh=false)"],
            "the pod is neither probed nor replaced"
        );
    }
}

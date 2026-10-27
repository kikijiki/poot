//! Command handlers and the remote benchmark lifecycle.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};

use crate::cli::CloudChoice;
use crate::cli::{ExecArgs, ImageArgs, LogsArgs, RunArgs};
use crate::clock::{Clock, SystemClock, iso_from_secs, iso_in_minutes, new_exec_run_id, now_iso};
use crate::cuda::CudaVersion;
use crate::db::runs::OwnerLiveness;
use crate::db::{Db, ExecOwner, ExecReapReport, PodOwnerPrefix, ReapProtection, Run, WarmExec};
use crate::gather::{self, Gathered};
use crate::hardware;
use crate::heal::{
    self, Cancelled, Cursor, Healing, HostFault, LocalFault, Outcome, PodLost, Replacement,
    SetupFault, Unrecoverable,
};
use crate::image_resolve::{
    DigestResolver, ImageLabels, RegistryV2Resolver, image_min_cuda, pin_image,
};
use crate::logging::{list_logs, log, logs_dir, read_log_tail, run_streaming, set_log_sink};
use crate::manifest::{Model, parse_manifest_models, select_models};
use crate::runpod::{
    CatalogPrice, CloudType, CreateError, CreateSpec, GpuCatalog, Pod, PodNotFound, RunPod,
};
use crate::ssh::{self, Remote, SSH_UNREACHABLE};

/// The RunPod operations the orchestrator uses. Production is [`RunPod`]; tests fake it.
pub(crate) trait PodApi {
    fn create_pod(&self, spec: &CreateSpec<'_>) -> Result<Pod>;
    fn get_pod(&self, id: &str) -> Result<Pod>;
    fn list_pods(&self) -> Result<Vec<Pod>>;
    /// The GPU catalog with each cloud's hourly price, for pricing a cloud before a create.
    fn gpu_catalog(&self) -> Result<GpuCatalog>;
    /// A pod that is already gone counts as deleted.
    fn delete_pod(&self, id: &str) -> Result<()>;
}

impl PodApi for RunPod {
    fn create_pod(&self, spec: &CreateSpec<'_>) -> Result<Pod> {
        RunPod::create_pod(self, spec)
    }
    fn get_pod(&self, id: &str) -> Result<Pod> {
        RunPod::get_pod(self, id)
    }
    fn list_pods(&self) -> Result<Vec<Pod>> {
        RunPod::list_pods(self)
    }
    fn gpu_catalog(&self) -> Result<GpuCatalog> {
        RunPod::gpu_catalog(self)
    }
    fn delete_pod(&self, id: &str) -> Result<()> {
        RunPod::delete_pod(self, id)
    }
}

/// Prepend to on-pod commands that need uv / hf / the poot bin / cuda. A non-interactive ssh shell gets
/// sshd's reset PATH (it ignores the image ENV), so these dirs are missing; restore them. `$PATH` is
/// expanded by the remote shell, not locally.
const POD_PATH: &str =
    "export PATH=\"/root/.local/bin:/root/.cargo/bin:/usr/local/cuda/bin:$PATH\"; ";

/// Which pods to terminate from one cloud snapshot: those named under `owner` (this state
/// database's pods), minus the ones the DB protection snapshot covers. Normal reap excludes recorded pod
/// ids and run-specific names; an unrecorded live claim suppresses deletion for the pass. `force`
/// ignores all protection but still only selects this database's pods.
pub(crate) fn reap_targets(
    pods: &[Pod],
    owner: &PodOwnerPrefix,
    protection: &ReapProtection,
    force: bool,
) -> Vec<String> {
    pods.iter()
        .filter(|p| owner.owns(&p.name))
        .filter(|p| {
            force
                || (!protection.has_unrecorded
                    && !protection.pod_ids.iter().any(|id| id == &p.id)
                    && !protection
                        .run_ids
                        .iter()
                        .any(|run_id| p.name.starts_with(&owner.run_prefix(run_id))))
        })
        .map(|p| p.id.clone())
        .collect()
}

/// Delete a pod through the API. Callers record the pod as gone only after this returns `Ok`.
fn delete_pod_confirmed<P: PodApi + ?Sized>(pods: &P, pod_id: &str) -> Result<()> {
    pods.delete_pod(pod_id).with_context(|| {
        format!(
            "pod {pod_id} was not deleted and may still be billing; retry with `reap` or delete it \
             on the RunPod dashboard"
        )
    })
}

/// The recorded hourly price when it is above `max`. A pod with no recorded price is never over: the cap is not
/// enforced on a price nobody knows.
fn over_price_cap(cost_per_hr: Option<f64>, max: Option<f64>) -> Option<f64> {
    cost_per_hr.filter(|cost| max.is_some_and(|max| *cost > max))
}

/// Whether the kept-warm pod `pod_id` bills above `max`, by the price its `pods` record kept.
fn warm_pod_over_cap(db: &Db, pod_id: &str, max: Option<f64>) -> Result<bool> {
    let cost = db.pod_record(pod_id)?.and_then(|pod| pod.cost_per_hr);
    Ok(over_price_cap(cost, max).is_some())
}

/// Delete a pod, then mark its `pods` record terminated. If the API refuses the delete the record stays
/// live and the error is returned.
fn terminate_recorded_pod<P: PodApi + ?Sized>(db: &Db, pods: &P, pod_id: &str) -> Result<()> {
    delete_pod_confirmed(pods, pod_id)?;
    db.set_pod_status(pod_id, "terminated")
}

#[derive(Debug, PartialEq)]
pub(crate) enum Resume {
    /// reuse this run; its pod is alive at (host, port).
    Reattach(String, u16),
    /// reuse this run; it never provisioned a (live) pod - start the pod phase fresh.
    Reprovision,
    /// the run had a pod that RunPod says is gone; the in-flight sweep is lost - abort it and go on.
    AbortStale,
    /// the pod is booting, or RunPod could not say: it is not gone, so the run and its pod are left alone.
    Leave,
}

/// What RunPod says about a run's recorded pod.
#[derive(Debug, PartialEq)]
pub(crate) enum PodState {
    /// Running with its SSH endpoint published.
    Ready(String, u16),
    /// Running without an endpoint yet, inside its provision window: still booting.
    Booting,
    /// RunPod says 404, or the pod is EXITED/TERMINATED, or it never published an endpoint in its window.
    Gone,
    /// The API call failed for any other reason: nothing is known about the pod.
    Unknown,
}

/// Classify a recorded pod. Only a definite answer makes it [`PodState::Gone`]; a pod that is still inside
/// `within_provision_window` and has no endpoint yet is booting, not gone.
pub(crate) fn pod_state<P: PodApi + ?Sized>(
    api: &P,
    pod_id: &str,
    within_provision_window: bool,
) -> PodState {
    match api.get_pod(pod_id) {
        Err(e) if e.downcast_ref::<PodNotFound>().is_some() => PodState::Gone,
        Err(_) => PodState::Unknown,
        Ok(pod) if matches!(pod.desired_status.as_str(), "EXITED" | "TERMINATED") => PodState::Gone,
        Ok(pod) => match pod.ssh_endpoint() {
            Some((host, port)) => PodState::Ready(host, port),
            None if within_provision_window => PodState::Booting,
            None => PodState::Gone,
        },
    }
}

/// Decide how to handle a resumable run, given what RunPod says about its recorded pod.
pub(crate) fn resume_decision(run: &Run, pod: PodState) -> Resume {
    match (&run.pod_id, pod) {
        (Some(_), PodState::Ready(h, p)) => Resume::Reattach(h, p),
        (Some(_), PodState::Gone) => Resume::AbortStale,
        (Some(_), PodState::Booting | PodState::Unknown) => Resume::Leave,
        (None, _) => Resume::Reprovision,
    }
}

/// Parse one `--env KEY=VAL` argument; the value may contain `=`.
pub(crate) fn parse_env_pair(raw: &str) -> Result<(String, String)> {
    let (k, v) = raw
        .split_once('=')
        .ok_or_else(|| anyhow!("--env expects KEY=VAL, got {raw:?}"))?;
    if k.is_empty() {
        bail!("--env key must be non-empty (got {raw:?})");
    }
    Ok((k.to_string(), v.to_string()))
}

/// Build the RunPod create-time `env` map for `exec`: defaults plus `--env` overrides, with
/// `PUBLIC_KEY` always last so SSH injection cannot be clobbered.
///
/// `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` must be set at container create time; a
/// post-start shell export does not unlock the NVIDIA Vulkan ICD on RunPod.
pub(crate) fn build_exec_create_env(
    pubkey: &str,
    extra: &[String],
) -> Result<HashMap<String, String>> {
    let mut env = HashMap::new();
    env.insert(
        "NVIDIA_DRIVER_CAPABILITIES".to_string(),
        "compute,utility,graphics".to_string(),
    );
    for raw in extra {
        let (k, v) = parse_env_pair(raw)?;
        env.insert(k, v);
    }
    env.insert("PUBLIC_KEY".to_string(), pubkey.trim().to_string());
    Ok(env)
}

pub(crate) fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/")
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}/{rest}");
    }
    p.to_string()
}

/// The repo root, discovered from the current directory via `git rev-parse --show-toplevel`.
///
/// Never `CARGO_MANIFEST_DIR` (compile-time `env!` or runtime `std::env::var`): this is a real deployed
/// production binary (`cargo run` today per `benchmarks/justfile`, but nothing guarantees that stays true),
/// not a test harness cargo always relaunches fresh. A compile-time `env!` bakes this worktree's absolute
/// path into the compiled binary, and a shared compile cache (kache) that reuses that binary across
/// worktrees by source-content hash would then serve whichever worktree's path happened to compile it first
/// (card 530's build.rs fix; card 543 review) - `logging.rs`'s `state_dir` doc records the exact failure
/// shape for a split-identity bug like this. `CARGO_MANIFEST_DIR` the runtime env var is set by `cargo run`,
/// but not by a bare invocation of the built binary, so depending on it would just trade one silent-wrong
/// failure mode for a silent-missing one. `git rev-parse` reflects where the *process* actually runs from
/// (the same technique `cmd_exec`'s sha lookup and `benchmarks/runners/poot/build.rs` already use), and a
/// checkout not found is a typed error - no caller may silently fall back to a baked or relative path.
pub(crate) fn repo_root() -> Result<PathBuf> {
    let root = sh(
        std::process::Command::new("git").args(["rev-parse", "--show-toplevel"]),
        "git rev-parse --show-toplevel (repo root)",
    )
    .context("poot-orchestrator must run from inside the poot git checkout")?;
    Ok(PathBuf::from(root.trim()))
}

pub(crate) fn sh(cmd: &mut std::process::Command, what: &str) -> Result<String> {
    let out = cmd.output().with_context(|| format!("spawn: {what}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "{what} failed (rc={:?})\n{}",
            out.status.code(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Margin on top of `--max-cmd` and `--keep-warm` in the default pod TTL: the upload, patch and setup
/// steps that run before the command's own deadline starts.
const POD_TTL_MARGIN: Duration = Duration::from_secs(30 * 60);

/// Budget for the short fixed remote steps (arming the TTL, patching the binary), which run before the
/// command's own `--max-cmd` deadline starts.
const REMOTE_STEP_BUDGET: Duration = Duration::from_secs(5 * 60);

/// Where the pod-side TTL records the process group of its sleeper, so a later `exec` that adopts the
/// pod can replace it.
const POD_TTL_PID_FILE: &str = "/root/.poot-pod-ttl.pid";

/// How long the pod lives before it removes itself: `--pod-ttl`, else the longest this `exec` can run.
fn pod_ttl(a: &ExecArgs) -> Duration {
    match a.pod_ttl {
        Some(minutes) => Duration::from_secs(minutes * 60),
        None => a.max_cmd + Duration::from_secs(a.keep_warm.unwrap_or(0) * 60) + POD_TTL_MARGIN,
    }
}

/// The remote command that makes the pod remove itself after `ttl`, independent of this machine: a
/// detached sleeper that runs `runpodctl remove pod` with the pod's own credentials (RunPod's REST
/// DELETE refuses the pod's key). An ssh session does not inherit the container's environment, so the
/// sleeper reads the pod id and key from the container's init process. It replaces a sleeper an earlier
/// `exec` left on an adopted pod, and the command fails unless `runpodctl` exists and the new sleeper
/// has started.
fn pod_ttl_command(ttl: Duration) -> String {
    let secs = ttl.as_secs();
    format!(
        "command -v runpodctl >/dev/null || {{ echo 'runpodctl is missing: the pod cannot remove itself' >&2; exit 1; }}; \
         if [ -s {POD_TTL_PID_FILE} ]; then kill -- \"-$(cat {POD_TTL_PID_FILE})\" 2>/dev/null; fi; \
         rm -f {POD_TTL_PID_FILE}; \
         setsid sh -c 'echo $$ > {POD_TTL_PID_FILE}; \
         pod_env() {{ tr \"\\0\" \"\\n\" </proc/1/environ | sed -n \"s/^$1=//p\"; }}; \
         sleep {secs}; \
         RUNPOD_API_KEY=$(pod_env RUNPOD_API_KEY) runpodctl remove pod \"$(pod_env RUNPOD_POD_ID)\"' \
         </dev/null >/root/.poot-pod-ttl.log 2>&1 & \
         for i in 1 2 3 4 5 6 7 8 9 10; do [ -s {POD_TTL_PID_FILE} ] && exit 0; sleep 0.2; done; \
         echo 'the pod TTL sleeper did not start' >&2; exit 1"
    )
}

/// The remote command that points an uploaded binary at the pod's glibc loader and makes it executable.
/// Each step runs only if the one before succeeded, and none hides its errors: a binary that was not
/// patched must fail the exec, not run and die with a misleading ENOENT.
fn patch_command(remote: &str) -> String {
    format!(
        "{{ command -v patchelf >/dev/null || {{ apt-get update -qq && apt-get install -y -qq patchelf; }}; }} \
         && patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 {remote} \
         && chmod +x {remote}"
    )
}

/// Run `cmd` on the remote, killing it at `deadline`. A timeout is the typed [`ssh::DeadlineError`]
/// under the context, so callers can `downcast_ref` it.
fn remote_run<R: Remote + ?Sized>(
    remote: &R,
    cmd: &str,
    deadline: Instant,
    what: &str,
) -> Result<ssh::Run> {
    remote
        .ssh_until(cmd, deadline)
        .with_context(|| format!("remote {what}"))
}

/// [`remote_run`], and fail on a non-zero exit with the command's stderr. Returns stdout.
fn remote_checked<R: Remote + ?Sized>(
    remote: &R,
    cmd: &str,
    deadline: Instant,
    what: &str,
) -> Result<String> {
    let run = remote_run(remote, cmd, deadline, what)?;
    if !run.ok() {
        bail!(
            "remote {what} failed (rc={}): {cmd}\n{}",
            run.code,
            run.stderr.trim()
        );
    }
    Ok(run.stdout)
}

/// A per-run scratch directory: the patched runner binary, the build worktree and the run's ssh
/// known-hosts file live here, so two runs never share a path. Removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Result<Scratch> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let dir =
            std::env::temp_dir().join(format!("poot-orch-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir(&dir).with_context(|| format!("create {}", dir.display()))?;
        Ok(Scratch(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn known_hosts(&self) -> PathBuf {
        self.0.join("known_hosts")
    }

    /// Where [`patchelf_for_pod`] puts the runner that `run` uploads.
    fn runner(&self) -> PathBuf {
        self.0.join("poot-bench-runner-pod")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            log(format!("could not remove {}: {e}", self.0.display()));
        }
    }
}

/// Provision a pod, optionally scp a binary, run a command, tear down. Reuses the bench pod machinery
/// for poot tests on real NVIDIA hardware. SIGINT and SIGTERM tear the run's pod down; the pod also
/// removes itself after its TTL even if this process is gone.
pub(crate) fn cmd_exec(db: &Db, state_db: &str, a: ExecArgs) -> Result<()> {
    let key = runpod_api_key()?;
    let rp = RunPod::new(key.clone())?;
    // Finalize and reap any killed prior exec (stale row, leaked pod) first.
    reap(db, &rp, false)?;
    let ssh_key = expand_tilde(&a.ssh_key);
    let pubkey = std::fs::read_to_string(expand_tilde(&format!("{}.pub", a.ssh_key)))
        .map_err(|e| anyhow!("read {}.pub: {e}", a.ssh_key))?;
    let env = build_exec_create_env(pubkey.trim(), &a.env)?;

    // Track the exec like a sweep: a `runs` row (kind='exec') records the pod at creation and shows the
    // job on the dashboard, plus a tailable log. `resumable_runs` filters kind='sweep', so it is never
    // resumed; reap protects its pod while the row is non-terminal.
    let run_id = new_exec_run_id();
    let logpath = logs_dir().join(format!("{run_id}.log"));
    set_log_sink(&logpath);
    let scratch = Scratch::new("exec")?;
    {
        let state_db = state_db.to_string();
        let run_id = run_id.clone();
        let keep = a.keep;
        let installed = install_signal_teardown(
            move || match (Db::open(&state_db), RunPod::new(key.clone())) {
                (Ok(hdb), Ok(hrp)) => {
                    if let Err(e) = exec_signal_teardown(&hdb, &hrp, &run_id, keep) {
                        log(format!("  {e:#}"));
                    }
                }
                (Err(e), _) | (_, Err(e)) => log(format!(
                    "signal received but run {run_id} could not be torn down and its pod may still \
                     be billing: {e:#}"
                )),
            },
            || std::process::exit(130),
        );
        if let Err(e) = installed {
            log(format!(
                "could not install SIGINT/SIGTERM teardown handler ({e}); a kill will leak this \
                 run's pod until its TTL, `reap` or manual delete"
            ));
        }
    }
    let resolver = RegistryV2Resolver::new()?;
    let ExecPod { pod_id, host, port } = acquire_exec_pod(
        db,
        &rp,
        &resolver,
        &a,
        &run_id,
        &WarmProbe {
            reachable: &|host, port| {
                ssh::Endpoint::new(host.to_string(), port, &ssh_key, scratch.known_hosts())
                    .wait_ssh(WARM_PROBE_TIMEOUT)
                    .is_ok()
            },
            pause: WARM_PROBE_PAUSE,
        },
        &env,
        PROVISION_POLL_INTERVAL,
    )?;
    let remote = ssh::Endpoint::new(host, port, ssh_key, scratch.known_hosts());
    exec_on_pod(db, &rp, &remote, &a, &run_id, &logpath, &pod_id)
}

/// Run `teardown` when the process gets SIGINT or SIGTERM, then `then`. The handler runs on its own
/// thread, so it works while the main thread is blocked in a network call.
fn install_signal_teardown(
    teardown: impl Fn() + Send + 'static,
    then: impl Fn() + Send + 'static,
) -> Result<(), ctrlc::Error> {
    ctrlc::set_handler(move || {
        teardown();
        then();
    })
}

/// What the signal handler does for an `exec`: delete the run's pod (unless `--keep`) and mark the run
/// terminal. The pod id comes from the run row, so a pod created a moment before the signal is found
/// too. A refused delete leaves the run non-terminal, like the normal teardown.
fn exec_signal_teardown<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    run_id: &str,
    keep: bool,
) -> Result<()> {
    match (db.pod_id_for_run(run_id)?, keep) {
        (Some(pod_id), false) => {
            log(format!(
                "signal received: tearing down pod {pod_id} for exec {run_id}"
            ));
            finish_exec_pod(db, pods, run_id, &pod_id, "aborted")
        }
        (Some(pod_id), true) => {
            log(format!(
                "signal received: --keep leaves pod {pod_id} alive (reaped on the next run)"
            ));
            db.set_status(run_id, "aborted")
        }
        (None, _) => {
            log("signal received: no pod to tear down (not yet provisioned)");
            db.set_status(run_id, "aborted")
        }
    }
}

/// Everything `exec` does on the pod once it has one: arm the pod-side TTL, upload, set up, run the
/// command under `--max-cmd`, then tear the pod down (or keep it) and record the run's status.
fn exec_on_pod<P: PodApi + ?Sized, R: Remote + ?Sized>(
    db: &Db,
    pods: &P,
    remote: &R,
    a: &ExecArgs,
    run_id: &str,
    logpath: &Path,
    pod_id: &str,
) -> Result<()> {
    // Run the body, capturing the result so the pod is always torn down (unless --keep).
    // `Ok(bool)` means the command ran over SSH (so the pod's control plane is healthy) and the bool is
    // its exit status; `Err` is a transport/environment failure (`wait_ssh`, `scp_up`, a failed
    // remote step, the `--max-cmd` timeout) before the command's result was known, i.e. a possibly
    // half-broken pod. Card 407.
    let result = exec_body(remote, a, logpath);

    // Cap on keep-warm minutes after a non-zero exit: callers are likelier to abandon a red run than
    // come back and adopt the pod. An unmeasured guess (card 407).
    const KEEP_WARM_RED_CEILING_MINUTES: u64 = 5;

    // Teardown and finalize the tracked status (the log classifier reads the EXEC_EXIT marker).
    // `Ok(_)` (clean or non-zero exit) proves the control plane is healthy, so it is eligible for
    // `--keep-warm`. `Err` always tears the pod down unless `--keep`.
    let mut teardown: Result<()> = Ok(());
    match &result {
        Ok(command_ok) => {
            // Status handling differs per branch on purpose: the keep-warm path never calls `set_status`
            // (`set_warm` tracks the row), while `keep` and teardown set "done"/"failed".
            if let Some(m) = a.keep_warm {
                let minutes = if *command_ok {
                    m
                } else {
                    m.min(KEEP_WARM_RED_CEILING_MINUTES)
                };
                let until = iso_in_minutes(minutes as i64);
                db.set_warm(run_id, &until).ok();
                if *command_ok {
                    log(format!(
                        "--keep-warm: pod {pod_id} kept warm until {until} ({minutes} min); the next exec with image {} adopts it. \
                         The next exec or reap deletes it after that time even if you are still using it by ssh",
                        a.image
                    ));
                } else {
                    log(format!(
                        "--keep-warm: command exited non-zero, but the pod's control plane answered - \
                         pod {pod_id} kept warm until {until} ({minutes} min, capped from {m}); the next \
                         exec with image {} adopts it. The next exec or reap deletes it after that time \
                         even if you are still using it by ssh",
                        a.image
                    ));
                    // Card 437: do not set_status("failed") here. `set_warm` stores warm-ness in `status`,
                    // so that would leave a 'failed' row with `warm_until` in the future, unadoptable and
                    // reapable. The result stays visible via EXEC_EXIT; an expired warm row is finalized
                    // as 'aborted' by `abort_stale_execs`.
                }
            } else if a.keep {
                log(format!(
                    "--keep: leaving pod {pod_id} alive{} (reaped on the next run)",
                    if *command_ok {
                        ""
                    } else {
                        " despite the command's non-zero exit"
                    }
                ));
                db.set_status(run_id, if *command_ok { "done" } else { "failed" })
                    .ok();
            } else {
                log(format!("tearing down pod {pod_id}"));
                teardown = finish_exec_pod(
                    db,
                    pods,
                    run_id,
                    pod_id,
                    if *command_ok { "done" } else { "failed" },
                );
            }
            log(format!("EXEC_EXIT={}", if *command_ok { 0 } else { 1 }));
        }
        Err(e) => {
            if a.keep {
                log(format!(
                    "--keep: leaving pod {pod_id} alive despite failure (reaped on the next run)"
                ));
                db.set_status(run_id, "failed").ok();
            } else {
                log(format!("tearing down pod {pod_id}"));
                teardown = finish_exec_pod(db, pods, run_id, pod_id, "failed");
            }
            log(format!("EXEC_EXIT=1 ({e:#})"));
        }
    }

    // A pod that could not be deleted is the louder failure: it keeps billing.
    if let Err(teardown_error) = teardown {
        log(format!("TEARDOWN_FAILED: {teardown_error:#}"));
        return Err(match result {
            Ok(true) => teardown_error,
            Ok(false) => teardown_error.context("the command also exited non-zero"),
            Err(e) => teardown_error.context(format!("the command also failed: {e:#}")),
        });
    }
    // A non-zero remote command still makes `exec` return `Err` (the CLI exit code).
    match result {
        Ok(true) => Ok(()),
        Ok(false) => Err(anyhow!("command exited non-zero")),
        Err(e) => Err(e),
    }
}

/// Wait for ssh, then arm the pod-side TTL before anything else runs, so every later failure (and this
/// process dying) is bounded by it. Shared by `exec` and the sweep.
fn ready_pod<R: Remote + ?Sized>(remote: &R, ssh_timeout: Duration, ttl: Duration) -> Result<()> {
    remote.wait_ssh(ssh_timeout)?;
    log(format!(
        "pod removes itself after {} min",
        ttl.as_secs() / 60
    ));
    remote_checked(
        remote,
        &pod_ttl_command(ttl),
        Instant::now() + REMOTE_STEP_BUDGET,
        "pod TTL",
    )?;
    Ok(())
}

/// How long a sweep pod lives: every model's repeats at the full `--sweep-timeout-s`, plus the margin
/// for upload and setup.
fn sweep_pod_ttl(a: &RunArgs, models: usize) -> Duration {
    Duration::from_secs(a.sweep_timeout_s) * (a.repeats * models as u32) + POD_TTL_MARGIN
}

/// The remote half of `exec`. Setup and `--cmd` share one `--max-cmd` budget; the steps before it have a
/// fixed one. Returns whether `--cmd` exited zero.
fn exec_body<R: Remote + ?Sized>(remote: &R, a: &ExecArgs, logpath: &Path) -> Result<bool> {
    ready_pod(remote, Duration::from_secs(a.ssh_timeout_s), pod_ttl(a))?;
    if let Some(bin) = &a.bin {
        let base = std::path::Path::new(bin)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "binary".to_string());
        let remote_path = format!("/root/{base}");
        log(format!("scp {bin} -> {remote_path}"));
        remote.scp_up(bin, &remote_path)?;
        remote_checked(
            remote,
            &patch_command(&remote_path),
            Instant::now() + REMOTE_STEP_BUDGET,
            "binary patch",
        )?;
    }
    for up in &a.uploads {
        let (local, remote_path) = up
            .split_once(':')
            .ok_or_else(|| anyhow!("--upload must be local:remote, got {up:?}"))?;
        log(format!("scp {local} -> {remote_path}"));
        remote.scp_up(local, remote_path)?;
    }
    let deadline = Instant::now() + a.max_cmd;
    if let Some(setup) = &a.setup {
        log(format!("setup: {setup}"));
        let out = remote_checked(remote, setup, deadline, "setup")?;
        if !out.trim().is_empty() {
            log(out);
        }
    }
    log(format!("exec: {} (at most {:?})", a.cmd, a.max_cmd));
    let run = remote_run(remote, &a.cmd, deadline, "command")?;
    // Raw stdout goes to our stdout (callers parse the JSON last line) and verbatim into the run log.
    print!("{}", run.stdout);
    if !run.stderr.trim().is_empty() {
        eprint!("{}", run.stderr);
    }
    append_raw(logpath, "=== command stdout ===\n");
    append_raw(logpath, &run.stdout);
    if !run.stderr.trim().is_empty() {
        append_raw(logpath, "=== command stderr ===\n");
        append_raw(logpath, &run.stderr);
    }
    // A non-zero exit is reported through the bool, not `Err`, to stay distinct from transport failures.
    Ok(run.ok())
}

/// How a kept-warm pod is probed before `exec` adopts it.
struct WarmProbe<'a> {
    /// One attempt to reach the pod's ssh endpoint (host, port).
    reachable: &'a dyn Fn(&str, u16) -> bool,
    /// The wait between two attempts.
    pause: Duration,
}

/// Attempts to reach a kept-warm pod before it counts as not answering: one failed probe is a busy host, not a
/// dead one (sshd is slow while a large model loads).
const WARM_PROBES: u32 = 4;
/// Time one attempt waits for ssh, and the pause between attempts: about four minutes of trying in all.
const WARM_PROBE_TIMEOUT: Duration = Duration::from_secs(20);
const WARM_PROBE_PAUSE: Duration = Duration::from_secs(45);

/// How long an unresponsive kept-warm pod is not probed again: every probe round costs minutes, and the pod is
/// not likely to recover within a few more.
const WARM_MISS_MINUTES: i64 = 10;

/// What the probes of a kept-warm pod found.
#[derive(Debug, PartialEq, Eq)]
enum WarmVerdict {
    /// ssh answered: the pod can be adopted.
    Reachable,
    /// RunPod says the pod no longer exists: only its record is left to close.
    Gone,
    /// RunPod lists the pod running but publishes no ssh endpoint for it, so it cannot be adopted. Its record is
    /// closed; the pod itself is left for `reap`.
    NoEndpoint,
    /// RunPod lists the pod but ssh did not answer in any attempt. Whoever holds it may be busy with it, so it
    /// is left alone: never adopted, never deleted, not probed again for [`WARM_MISS_MINUTES`].
    Unresponsive,
}

/// Judge a kept-warm pod. RunPod is asked first and last (a pod it no longer lists is gone whatever ssh says,
/// and needs no delete call); ssh is tried [`WARM_PROBES`] times in between.
fn judge_warm_pod<P: PodApi + ?Sized>(
    pods: &P,
    warm: &WarmExec,
    probe: &WarmProbe<'_>,
) -> WarmVerdict {
    let listed = || match pod_state(pods, &warm.pod_id, true) {
        PodState::Gone => Some(WarmVerdict::Gone),
        PodState::Booting => Some(WarmVerdict::NoEndpoint),
        PodState::Ready(..) | PodState::Unknown => None,
    };
    for attempt in 1..=WARM_PROBES {
        if let Some(verdict) = listed() {
            return verdict;
        }
        if (probe.reachable)(&warm.host, warm.port) {
            return WarmVerdict::Reachable;
        }
        log(format!(
            "  warm pod {} (kept by {}): ssh probe {attempt}/{WARM_PROBES} failed",
            warm.pod_id, warm.run_id
        ));
        if attempt < WARM_PROBES {
            std::thread::sleep(probe.pause);
        }
    }
    listed().unwrap_or(WarmVerdict::Unresponsive)
}

/// Was the kept-warm pod found unresponsive within [`WARM_MISS_MINUTES`]? (Recorded as the pod's status.)
fn missed_recently(db: &Db, pod_id: &str) -> Result<bool> {
    Ok(db.pod_record(pod_id)?.is_some_and(|pod| {
        pod.status == "unresponsive"
            && pod.updated_at > crate::clock::iso_in_minutes(-WARM_MISS_MINUTES)
    }))
}

/// Try to adopt a kept-warm pod (a prior `--keep-warm` exec with the same image); it already has the image
/// pulled. Candidates are tried newest first, so one stuck pod does not hide an older reachable one. Adoption moves
/// the pod claim in one DB transaction, so two concurrent execs cannot take the same pod. A pod past its window is
/// reaped; a pod RunPod no longer lists only has its record closed; a listed pod that does not answer is left
/// alone and not probed again for a while. Pods of running jobs are never candidates.
fn adopt_warm_pod<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    a: &ExecArgs,
    warm: &WarmProbe<'_>,
    image: &str,
    run_id: &str,
    owner: &ExecOwner,
) -> Result<Option<ExecPod>> {
    for w in db.find_warm_execs(image)? {
        if w.warm_until.as_str() <= now_iso().as_str() {
            log(format!(
                "warm pod {} (kept by {}) expired (until {}); reaping it",
                w.pod_id, w.run_id, w.warm_until
            ));
            if db.try_abort_warm_exec(&w)? {
                delete_pod_confirmed(pods, &w.pod_id)?;
            } else {
                log(format!(
                    "warm pod {} changed before expired cleanup; not deleting it",
                    w.pod_id
                ));
            }
            continue;
        }
        if warm_pod_over_cap(db, &w.pod_id, a.max_price_per_hour)? {
            log(format!(
                "warm pod {} is over --max-price-per-hour; not reusing it (it stays until it expires)",
                w.pod_id
            ));
            continue;
        }
        if missed_recently(db, &w.pod_id)? {
            log(format!(
                "warm pod {} (kept by {}) did not answer ssh in the last {WARM_MISS_MINUTES} minutes; not probing it again yet",
                w.pod_id, w.run_id
            ));
            continue;
        }
        match judge_warm_pod(pods, &w, warm) {
            WarmVerdict::Reachable => {
                if db.try_adopt_warm_exec(&w, run_id, owner)? {
                    db.set_pod_status(&w.pod_id, "busy")?;
                    log(format!(
                        "reusing warm pod {} at {}:{} (warm until {}) - skipping provision + image pull",
                        w.pod_id, w.host, w.port, w.warm_until
                    ));
                    if !a.env.is_empty() {
                        log("note: --env is create-time only; ignored when adopting a warm pod");
                    }
                    return Ok(Some(ExecPod {
                        pod_id: w.pod_id,
                        host: w.host,
                        port: w.port,
                    }));
                }
                log(format!("warm pod {} changed before adoption", w.pod_id));
            }
            WarmVerdict::Gone => {
                log(format!(
                    "warm pod {} (kept by {}) no longer exists on RunPod; closing its record",
                    w.pod_id, w.run_id
                ));
                if db.try_abort_warm_exec(&w)? {
                    db.set_pod_status(&w.pod_id, "terminated")?;
                }
            }
            WarmVerdict::NoEndpoint => {
                log(format!(
                    "warm pod {} (kept by {}) is running on RunPod but publishes no ssh endpoint; it cannot be adopted, closing its record (reap takes the pod)",
                    w.pod_id, w.run_id
                ));
                if db.try_abort_warm_exec(&w)? {
                    db.set_pod_status(&w.pod_id, "terminated")?;
                }
            }
            WarmVerdict::Unresponsive => {
                log(format!(
                    "warm pod {} (kept by {}) is listed by RunPod but did not answer ssh {WARM_PROBES} times; \
                     leaving it alone (it is not mine to delete, it expires {})",
                    w.pod_id, w.run_id, w.warm_until
                ));
                db.set_pod_status(&w.pod_id, "unresponsive")?;
            }
        }
    }
    Ok(None)
}

/// The pod an `exec` runs on.
struct ExecPod {
    pod_id: String,
    host: String,
    port: u16,
}

/// Everything `exec` does before it has a pod to run on: pin the image, record the run, then adopt a
/// kept-warm pod or provision one.
///
/// The image tag is resolved to a digest first, before the run row or any pod exists, and the pod is
/// created from the digest reference: a tag that moves during the run cannot change the image. The run
/// row records the digest reference, so only a pod kept warm for that same image is adopted.
#[expect(clippy::too_many_arguments)]
fn acquire_exec_pod<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    resolver: &dyn DigestResolver,
    a: &ExecArgs,
    run_id: &str,
    warm: &WarmProbe<'_>,
    env: &HashMap<String, String>,
    poll_interval: Duration,
) -> Result<ExecPod> {
    let image = match pin_image(resolver, &a.image) {
        Ok(image) => image,
        Err(e) => {
            log(format!("EXEC_EXIT=1 (image {} not pinned: {e})", a.image));
            return Err(anyhow!(e).context(format!("pin image {} to a digest", a.image)));
        }
    };
    log(format!("image {} pinned to {image}", a.image));
    let sha = sh(
        std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]),
        "git rev-parse",
    )
    .map(|s| s.trim().to_string())
    .unwrap_or_else(|_| "exec".into());
    let summary: String = a.cmd.split_whitespace().collect::<Vec<_>>().join(" ");
    let summary = summary.chars().take(120).collect::<String>();
    let exec_run = Run {
        id: run_id.to_string(),
        kind: "exec".into(),
        git_ref: sha,
        scenario: summary,
        repeats: 1,
        models: vec![],
        gpu_types: a.gpu_types.clone(),
        image: image.clone(),
        status: "new".into(),
        pod_id: None,
    };
    let owner = ExecOwner::current().context("claim exec owner identity")?;
    db.create_run_with_owner(&exec_run, Some(&owner))?;
    log(format!("exec {run_id}: {}", a.cmd));

    // Try to adopt a kept-warm pod (a prior `--keep-warm` exec with the same image); it already has the
    // image pulled. An expired or unreachable warm pod is reaped, then we provision fresh. Adoption moves
    // the pod claim in one DB transaction, so two concurrent execs cannot take the same pod.
    let adopted = adopt_warm_pod(db, pods, a, warm, &image, run_id, &owner)?;

    if let Some(pod) = adopted {
        return Ok(pod);
    }
    // Same provisioning as the sweep. The on-create hook records the pod id first,
    // so a crash cannot leak the pod.
    let min_cuda = cuda_floor(a.min_cuda_version, &image);
    let data_center_ids: Vec<&str> = a.data_center.iter().map(String::as_str).collect();
    let owner_prefix = db.pod_owner_prefix()?;
    let cfg = ProvisionCfg {
        gpu_types: &a.gpu_types,
        gpu_count: a.gpu_count,
        cloud: a.cloud,
        max_price_per_hour: a.max_price_per_hour,
        image: &image,
        container_disk_gb: a.container_disk_gb,
        network_volume_id: None,
        data_center_ids: &data_center_ids,
        give_up_after: a.give_up_after_min.map(|m| Duration::from_secs(m * 60)),
        timeout_s: a.provision_timeout_s,
        poll_interval,
        owner: &owner_prefix,
        run_id,
        min_cuda,
    };
    let (pod_id, host, port) = match provision_pod(
        pods,
        &cfg,
        env,
        |pid, gpu, placement| {
            db.create_pod_record(
                pid,
                gpu,
                &placement.cloud.to_string(),
                placement.cost_per_hr,
                &image,
            )?;
            db.set_pod(run_id, pid, None, None)
        },
        |_| {},
        &SystemClock::start(),
    ) {
        Ok(v) => v,
        Err(e) => {
            let status = if e.downcast_ref::<Cancelled>().is_some() {
                "cancelled"
            } else {
                "failed"
            };
            db.set_status(run_id, status).ok();
            log(format!("EXEC_EXIT=1 (provision failed: {e})"));
            return Err(e);
        }
    };
    db.set_pod(run_id, &pod_id, Some(&host), Some(port)).ok();
    db.set_pod_endpoint(&pod_id, &host, port).ok();
    db.assign_pod(&pod_id, run_id).ok();
    db.set_status(run_id, "running").ok();
    log(format!("endpoint up: {host}:{port} (pod {pod_id})"));
    Ok(ExecPod { pod_id, host, port })
}

/// Delete an `exec` pod, then record the run's final status. If the API refuses the delete the run row
/// stays non-terminal, so the pod stays protected as live (and `reap` retries it once the owner is
/// provably dead) instead of being recorded as gone while it still bills.
fn finish_exec_pod<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    run_id: &str,
    pod_id: &str,
    status: &str,
) -> Result<()> {
    delete_pod_confirmed(pods, pod_id)?;
    db.set_pod_status(pod_id, "terminated")?;
    db.set_status(run_id, status)
}

/// Append text verbatim (no timestamp prefix) to a log file, to capture a remote command's raw output.
fn append_raw(path: &Path, text: &str) {
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(text.as_bytes());
        let _ = f.flush();
    }
}

/// Build (and optionally push) the bench image with podman, logging to stdout and
/// `<state_dir>/logs/image-<tag>.log`.
pub(crate) fn cmd_image(a: &ImageArgs) -> Result<()> {
    let image = format!("ghcr.io/{}/poot-bench:{}", a.owner, a.tag);
    set_log_sink(&logs_dir().join(format!("image-{}.log", a.tag)));
    let ctx = repo_root()?.join("benchmarks");
    let mut smoke_done = false;
    if !a.no_build {
        log(format!(
            "building {image}  (dockerfile={}, context={})",
            a.dockerfile,
            ctx.display()
        ));
        // --format docker for compatibility with RunPod's puller.
        let rc = run_streaming(&format!(
            "cd {} && podman build --format docker -f {} -t {} .",
            ctx.display(),
            a.dockerfile,
            image
        ))?;
        if rc != 0 {
            bail!(
                "podman build failed (rc={rc}) - see {}/image-{}.log",
                logs_dir().display(),
                a.tag
            );
        }
        log(format!("build complete: {image}"));
        smoke_image(&image)?;
        smoke_done = true;
    }
    if a.push {
        if !smoke_done {
            smoke_image(&image)?;
        }
        log(format!("pushing {image} (needs `podman login ghcr.io`)"));
        let rc = run_streaming(&format!("podman push {image}"))?;
        if rc != 0 {
            bail!("podman push failed (rc={rc})");
        }
        log(format!("push complete: {image}"));
    }
    if !a.push {
        log("not pushed (pass --push to publish)");
    }
    Ok(())
}

fn smoke_image(image: &str) -> Result<()> {
    log(format!("smoke checking {image}"));
    // Ubuntu's `vulkaninfo --help` prints usage and may return 1, so require the text, not just rc=0.
    let smoke = concat!(
        r#"set -euo pipefail; "#,
        r#"ldconfig -p | grep -q "libvulkan.so.1"; "#,
        r#"command -v vulkaninfo >/dev/null; "#,
        r#"help_rc=0; vulkaninfo --help >/tmp/vulkaninfo-help 2>&1 || help_rc=$?; "#,
        r#"test "$help_rc" -eq 0 -o "$help_rc" -eq 1; grep -q "USAGE" /tmp/vulkaninfo-help; "#,
        r#"/opt/tf-venv/bin/python -c "import transformers, torch; print(\"tf\", transformers.__version__)"; "#,
        "ls /opt/vllm-venv/bin/python ",
        "/opt/llama.cpp/build/bin/llama-bench ",
        "/opt/benchmarks/runners/candle/target/release/bench-candle-runner",
    );
    let rc = run_streaming(&format!("podman run --rm {image} bash -lc '{smoke}'"))?;
    if rc != 0 {
        log(format!("image smoke failed (rc={rc})"));
        bail!("image smoke failed (rc={rc})");
    }
    log(format!("image smoke complete: {image}"));
    Ok(())
}

/// Tail a run/build log, or list available logs when no name is given.
pub(crate) fn cmd_logs(a: &LogsArgs) -> Result<()> {
    let Some(name) = &a.name else {
        for (name, size, mtime) in list_logs() {
            println!("{name}\t{:>8} KB\t(mtime {mtime})", size / 1024);
        }
        return Ok(());
    };
    let tail = read_log_tail(name, a.tail).ok_or_else(|| anyhow!("no such log: {name}"))?;
    println!("{tail}");
    if a.follow {
        let mut seen = tail.len();
        loop {
            std::thread::sleep(Duration::from_secs(2));
            if let Some(full) = read_log_tail(name, usize::MAX)
                && full.len() > seen
            {
                print!("{}", &full[seen..]);
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
                seen = full.len();
            }
        }
    }
    Ok(())
}

/// The RunPod API key, read from `RUNPOD_API_KEY` only: a key given as a command-line argument is
/// visible to every local user in the process list.
pub(crate) fn runpod_api_key() -> Result<String> {
    std::env::var("RUNPOD_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .ok_or_else(|| anyhow!("RUNPOD_API_KEY is not set"))
}

pub(crate) fn cmd_status(db: &Db, state_db: &str) -> Result<()> {
    use std::io::Write as _;

    let runs = db.recent_runs(20)?;
    let pods = db.live_pods()?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let write_result = (|| -> std::io::Result<()> {
        // Print the resolved DB path so a split-state bug is visible.
        writeln!(out, "state db: {state_db}")?;
        for (run, updated) in runs {
            let owner = match db.run_owner(&run.id) {
                Ok(Some(o)) => format!(
                    "pid {} ({})",
                    o.pid,
                    match o.local_liveness() {
                        crate::db::runs::OwnerLiveness::Live => "live",
                        crate::db::runs::OwnerLiveness::Dead => "dead",
                        crate::db::runs::OwnerLiveness::Unknown => "unknown",
                    }
                ),
                _ => "none".to_string(),
            };
            writeln!(
                out,
                "{}  {:11}  {:20}  pod={}  owner={}  models={}  (updated {})",
                run.id,
                run.status,
                run.git_ref,
                run.pod_id.as_deref().unwrap_or("-"),
                owner,
                run.models.join(","),
                updated
            )?;
        }
        for pod in &pods {
            writeln!(out, "{}", pod_status_line(pod))?;
        }
        Ok(())
    })();
    finish_status_output(write_result)
}

/// One live pod as `status` prints it: its cloud and what it bills per hour.
fn pod_status_line(pod: &crate::db::Pod) -> String {
    let price = pod.cost_per_hr.map_or_else(
        || "price unknown".to_string(),
        |cost| format!("{cost:.2} per hour"),
    );
    format!(
        "pod {}  {:11}  {}  {}  {price}",
        pod.id, pod.status, pod.gpu_type, pod.cloud
    )
}

pub(crate) fn finish_status_output(write_result: std::io::Result<()>) -> Result<()> {
    match write_result {
        // A pager or `head` closing stdout is a successful short read, not an
        // orchestrator failure. The standard print macros panic in this case.
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        Err(error) => Err(error).context("write status output"),
        Ok(()) => Ok(()),
    }
}

/// Terminate this state database's orphan pods. Returns how many the API confirmed deleted; a pod it
/// would not delete is an error after every target has been tried.
pub(crate) fn reap<P: PodApi + ?Sized>(db: &Db, rp: &P, force: bool) -> Result<usize> {
    // Finalize killed exec rows first, only once their owner process is provably gone: a concurrent
    // exec may be provisioning in the same DB. `--all` overrides. Split into a helper so the DB half is
    // testable without a RunPod client.
    finalize_stale_claims(db, force)?;
    // Reap is the leak safety net, so a listing failure means orphans were not reaped either.
    let pods = rp.list_pods().context(
        "reap could not list RunPod pods, so it ALSO could not verify or terminate orphan pods right \
         now - any pod still running keeps billing. Check pods directly (RunPod dashboard, or the \
         RunPod API / `list-pods` MCP tool) as soon as credentials/connectivity are restored, and \
         delete any orphan by hand",
    )?;
    // The cloud snapshot must come first: a pod created after it cannot be in `targets`, and a pod
    // already visible is protected by the later DB snapshot (recorded id, run-specific name, or the
    // unrecorded-claim gate). Reading ids before `list_pods` caused the card 149 / 266 incident.
    let protection = if force {
        ReapProtection::default()
    } else {
        db.reap_protection()?
    };
    if protection.has_unrecorded {
        log("kept owned pods protected because a nonterminal run has not recorded its pod id yet");
    }
    let targets = reap_targets(&pods, &db.pod_owner_prefix()?, &protection, force);
    let mut failed = Vec::new();
    for id in &targets {
        log(format!("reaping orphan pod {id}"));
        if let Err(e) = delete_pod_confirmed(rp, id) {
            log(format!("  {e:#}"));
            failed.push(id.as_str());
        }
    }
    if !failed.is_empty() {
        bail!(
            "reap could not delete {} of {} orphan pod(s): {}",
            failed.len(),
            targets.len(),
            failed.join(", ")
        );
    }
    Ok(targets.len())
}

/// Finalize dead-owner claims before the cloud snapshot: killed `exec` runs via `abort_stale_execs`.
/// `force` is `reap --all`. Split from [`reap`] so tests exercise this DB half without a RunPod client.
pub(crate) fn finalize_stale_claims(db: &Db, force: bool) -> Result<ExecReapReport> {
    let report = db.abort_stale_execs(force)?;
    if report.aborted > 0 {
        log(format!(
            "finalized {} stale exec run(s) as aborted",
            report.aborted
        ));
    }
    if report.kept_unknown > 0 {
        log(format!(
            "kept {} exec run(s) protected because owner liveness could not be proven dead; use reap --all only for explicit force cleanup",
            report.kept_unknown
        ));
    }
    Ok(report)
}

pub(crate) fn cmd_run(db: Db, state_db: String, a: RunArgs) -> Result<()> {
    let repo = repo_root()?;
    // models
    let manifest_path = repo.join("benchmarks").join("manifest.toml");
    let manifest = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let all = parse_manifest_models(&manifest)?;
    let models = select_models(&all, a.models.as_deref())?;
    if models.is_empty() {
        bail!("no models selected");
    }
    log(format!(
        "models: {:?}",
        models.iter().map(|m| &m.id).collect::<Vec<_>>()
    ));
    // Refused here, before anything is built or rented (and by `--dry-run`).
    if let Some(set) = a.extend_set.as_deref() {
        extension_set(&results_dir()?, set, &models)?;
    }

    // Build the poot runner locally, always (cheap; a resume picks up local fixes).
    let sha = sh(
        std::process::Command::new("git")
            .args(["rev-parse", "--short", &a.git_ref])
            .current_dir(&repo),
        "git rev-parse",
    )?
    .trim()
    .to_string();
    let scratch = Scratch::new("run")?;
    let poot_bin = match &a.poot_bin {
        // patchelf a prebuilt binary too: a nix-built one would fail execve on the pod (missing nix loader).
        Some(p) => patchelf_for_pod(Path::new(p), &scratch)?,
        None => build_poot_runner(&a.git_ref, &sha, &scratch)?,
    };
    log(format!(
        "poot runner: {} (ref {} = {sha})",
        poot_bin.display(),
        a.git_ref
    ));

    if a.dry_run {
        log("--dry-run: built + validated; not touching RunPod. Done.");
        return Ok(());
    }

    let key = runpod_api_key()?;
    let rp = RunPod::new(key.clone())?;
    // Reap orphans from a previous dead run (also finalizes killed exec rows).
    reap(&db, &rp, false)?;

    // Resume an in-progress run or create a new one.
    let (run, endpoint) = resolve_run(&db, &rp, &a, &models, &sha)?;
    // Tee progress to a log file for the dashboard and `logs`.
    set_log_sink(&logs_dir().join(format!("{}.log", run.id)));

    // Card 230: tear down this run's pod on SIGINT/SIGTERM, then exit. Otherwise a killed sweep leaks
    // its pod: the inline `teardown_pod` below only runs on the run_phases Ok/Err arms, and `reap`
    // excludes a killed sweep's pod because its run row stays non-terminal. `ctrlc` runs the handler on
    // a dedicated thread, so the DB read and network delete are safe. Best-effort; failures fall back to
    // resume / `reap --all` / manual delete. `--keep-pod` is honored. The pod id is read from a fresh Db
    // connection because it is only known after provisioning.
    {
        let state_db = state_db.clone();
        let key = key.clone();
        let run_id = run.id.clone();
        let keep_pod = a.keep_pod;
        if let Err(e) = ctrlc::set_handler(move || {
            log(format!(
                "signal received: finalizing run {run_id} as cancelled"
            ));
            match (Db::open(&state_db), RunPod::new(key.clone())) {
                (Ok(hdb), Ok(hrp)) => {
                    if let Err(e) = finalize_on_signal(&hdb, &hrp, &run_id, keep_pod) {
                        log(format!("  {e:#}"));
                    }
                }
                (Ok(hdb), Err(e)) => {
                    log(format!(
                        "  RunPod client: {e:#}; the pod was not deleted and may still be billing"
                    ));
                    let _ = hdb.set_status(&run_id, "cancelled");
                }
                _ => log("  could not open the state database to finalize the run"),
            }
            std::process::exit(130);
        }) {
            log(format!(
                "could not install SIGINT/SIGTERM teardown handler ({e}); a kill will leak this run's pod \
                 until `reap --force` or manual delete"
            ));
        }
    }

    let mut guard = RunGuard {
        state_db: state_db.clone(),
        key: key.clone(),
        run_id: run.id.clone(),
        keep_pod: a.keep_pod,
        armed: true,
    };
    // Zombies of earlier processes are cleaned up now that this run has its pod or resumed run.
    gc_zombies(&db);
    gc_orphan_partials(&db);
    // Supervise the run: a lost pod is replaced and only the missing cells are run again; the run ends when it
    // is done, when the user bounds it, or on an error waiting cannot fix. Teardown always follows.
    let cursor = Cursor::default();
    let mut live = LiveRun {
        db: &db,
        rp: &rp,
        run: &run,
        models: &models,
        scratch: &scratch,
        a: &a,
        endpoint,
        cursor: &cursor,
        clock: SystemClock::start(),
    };
    let result = heal::supervise(&mut live, &cursor, a.max_replacements);
    // With --keep-pod the pod is released to idle for a later run to adopt; otherwise it is terminated. A failure of
    // this machine (not the pod) keeps the pod too: it may hold the only copy of the results.
    let keep_pod = keep_pod_after(&result, a.keep_pod);
    if keep_pod
        && !a.keep_pod
        && let Some((host, port)) = &live.endpoint
    {
        log(format!(
            "the pod is kept (released to idle) so the results can be pulled by hand; pull the sets with:\n{}",
            ssh::pull_hint(
                &a.ssh_key,
                host,
                *port,
                &format!("{}/run-*", gather::POD_RESULTS),
                "benchmarks/results/"
            )
        ));
    }
    let pid = run.pod_id_now(&db);
    // A run whose supervision ended is only DONE when every requested model has all its valid sets.
    let incomplete = match &result {
        Ok(Outcome::Done) => incomplete_models(&run, &models, &results_dir()?, &db)?,
        _ => None,
    };
    let status = match (&result, &incomplete) {
        (Ok(Outcome::Done), None) => "done".to_string(),
        (Ok(Outcome::Done), Some(missing)) => {
            log(format!("run incomplete: {missing}"));
            format!("incomplete: {missing}")
        }
        (Ok(Outcome::Cancelled(why)), _) => {
            log(format!("run cancelled: {why}"));
            "cancelled".to_string()
        }
        (Err(e), _) => {
            log(format!("run failed: {e:#}"));
            "failed".to_string()
        }
    };
    let finished = finish_run(&db, &rp, &run.id, pid.as_deref(), keep_pod, &status);
    guard.armed = false;
    if let Err(teardown_error) = finished {
        return Err(match result {
            Ok(_) => teardown_error,
            Err(e) => teardown_error.context(format!("the run also failed: {e:#}")),
        });
    }
    match result {
        Ok(Outcome::Done) => match incomplete {
            None => {
                log("DONE");
                Ok(())
            }
            Some(missing) => bail!("run incomplete: {missing}"),
        },
        Ok(Outcome::Cancelled(why)) => bail!("run cancelled: {why}"),
        Err(e) => Err(e),
    }
}

/// The requested models that lack valid sets, as `model valid/requested` joined by commas, or `None` when each
/// has all of its repeats under `results`. A set recorded as failed is not a valid one.
fn incomplete_models(
    run: &Run,
    models: &[Model],
    results: &Path,
    db: &Db,
) -> Result<Option<String>> {
    let mut missing = Vec::new();
    for m in models {
        let dirs = run.model_snapshot_dirs(&sweep_run_id(db, run, &m.id)?);
        let valid = dirs
            .iter()
            .filter(|dir| gather::local_set(results, dir).is_some())
            .count();
        if valid < dirs.len() {
            missing.push(format!("{} {valid}/{}", m.id, dirs.len()));
        }
    }
    Ok((!missing.is_empty()).then(|| missing.join(", ")))
}

/// Does the run's pod outlive it? With `--keep-pod`, and after a failure of this machine ([`LocalFault`]): then the
/// pod is fine and may hold results that were not copied.
fn keep_pod_after(result: &Result<Outcome>, keep_pod: bool) -> bool {
    keep_pod || matches!(result, Err(e) if e.downcast_ref::<LocalFault>().is_some())
}

/// A run that is being interrupted (SIGINT/SIGTERM, a panic, an early return): delete or release its pod and
/// record it `cancelled`. A pod the API will not delete is logged and left to `reap`, which takes it as soon
/// as its run is terminal; the run is `cancelled` either way, never left looking busy.
pub(crate) fn finalize_on_signal<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    run_id: &str,
    keep_pod: bool,
) -> Result<()> {
    if let Some(pid) = db.pod_id_for_run(run_id)? {
        if keep_pod {
            db.release_pod(&pid)?;
        } else if let Err(e) = terminate_recorded_pod(db, pods, &pid) {
            log(format!("  {e:#}"));
        }
    }
    db.set_status(run_id, "cancelled")
}

/// Why a run that ended without finishing is shown greyed on the dashboard.
pub(crate) const ABANDONED_REASON: &str = "owner exited without finalizing";

/// Abandon the unfinished sweeps nobody drives any more: owner provably dead, and nothing to resume onto
/// (no pod, or RunPod says it is gone) or older than the resume window (its pod, if alive, is terminated).
/// A run with a live, unknown or unrecorded owner is never touched, nor a dead-owner run whose pod is alive
/// inside the window (a matching `run` reattaches to it), nor a run whose pod RunPod cannot be asked about.
pub(crate) fn abandon_zombies<P: PodApi + ?Sized>(
    db: &Db,
    pods: Option<&P>,
    now_epoch_s: u64,
    window_s: u64,
    liveness: impl Fn(&ExecOwner) -> OwnerLiveness,
) -> Result<usize> {
    let mut abandoned = 0;
    for (run, owner) in db.unfinished_sweeps()? {
        if !owner
            .as_ref()
            .is_some_and(|o| liveness(o) == OwnerLiveness::Dead)
        {
            continue;
        }
        let too_old = run
            .id
            .strip_prefix("run-")
            .and_then(|rest| rest.split('-').next())
            .and_then(|secs| secs.parse::<u64>().ok())
            .is_some_and(|started| now_epoch_s.saturating_sub(started) > window_s);
        if let Some(pid) = &run.pod_id {
            let Some(pods) = pods else { continue };
            match pod_state(pods, pid, true) {
                PodState::Unknown => continue,
                PodState::Gone => {
                    let _ = db.set_pod_status(pid, "terminated");
                }
                PodState::Ready(..) | PodState::Booting => {
                    if !too_old {
                        continue;
                    }
                    if let Err(e) = terminate_recorded_pod(db, pods, pid) {
                        log(format!("  {e:#}"));
                        continue;
                    }
                }
            }
        }
        db.set_status(&run.id, "abandoned")?;
        log(format!("run {} abandoned: {ABANDONED_REASON}", run.id));
        abandoned += 1;
    }
    Ok(abandoned)
}

/// [`abandon_zombies`] for the CLI: asks RunPod about pods when an API key is set.
pub(crate) fn gc_zombies(db: &Db) {
    let rp = runpod_api_key().ok().and_then(|key| RunPod::new(key).ok());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    if let Err(e) = abandon_zombies(db, rp.as_ref(), now, RESUME_WINDOW_S, |o| {
        o.local_liveness()
    }) {
        log(format!("zombie cleanup: {e:#}"));
    }
}

/// Remove the live-results mirrors (`results/*.partial`) that no run still in flight is sweeping: a run that
/// crashed or was killed leaves its mirror behind and nothing would ever gather or clear it.
fn gc_orphan_partials(db: &Db) {
    let removed = (|| -> Result<Vec<String>> {
        let mut active = Vec::new();
        for id in db.unfinished_run_ids()? {
            active.extend(db.snapshot_dirs(&id)?);
        }
        gather::remove_orphan_partials(&results_dir()?, &active)
    })();
    match removed {
        Ok(names) => names
            .iter()
            .for_each(|name| log(format!("removed orphan live-results mirror {name}"))),
        Err(e) => log(format!("orphan mirrors: {e:#}")),
    }
}

/// Finalizes the run as `cancelled` if `cmd_run` is left any way but the normal end (a panic, an early `?`).
struct RunGuard {
    state_db: String,
    key: String,
    run_id: String,
    keep_pod: bool,
    armed: bool,
}

impl Drop for RunGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let (Ok(db), Ok(rp)) = (Db::open(&self.state_db), RunPod::new(self.key.clone())) {
            let _ = finalize_on_signal(&db, &rp, &self.run_id, self.keep_pod);
        }
    }
}

/// Release (`--keep-pod`) or delete the run's pod, then record the run's final `status`. The run and its
/// pod record change only after the API confirms the deletion: if it refuses, both stay live (the next
/// `run` resumes or aborts the run, and `reap` sees the pod as protected) and the error is returned.
fn finish_run<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    run_id: &str,
    pod_id: Option<&str>,
    keep_pod: bool,
    status: &str,
) -> Result<()> {
    if let Some(pid) = pod_id {
        if keep_pod {
            log(format!(
                "--keep-pod: releasing pod {pid} to idle (a later run can adopt it)"
            ));
            db.release_pod(pid)?;
        } else {
            log(format!("tearing down pod {pid} (run {status})"));
            terminate_recorded_pod(db, pods, pid)?;
        }
    }
    db.set_status(run_id, status)
}

/// Delete the pod of a run whose pod is gone or unreachable, then abort the run. If the API refuses the
/// delete the run stays active with its pod recorded, so the next `run` retries.
fn abort_stale_run<P: PodApi + ?Sized>(db: &Db, pods: &P, run: &Run) -> Result<()> {
    if let Some(pid) = &run.pod_id {
        terminate_recorded_pod(db, pods, pid)?;
        log(format!("  terminated stale pod {pid}"));
    }
    db.set_status(&run.id, "aborted")
}

/// Resolve which run to operate on (resume or new), plus the live SSH endpoint if reattaching.
/// How long after a failed run's start a new run of the same request continues its result set.
const RESUME_WINDOW_S: u64 = 24 * 3600;

/// The set id `--extend-set` names, if every model already has a gathered repeat of it under `results`: extending
/// sets that are not here would silently start new ones.
fn extension_set(results: &Path, set: &str, models: &[Model]) -> Result<String> {
    for model in models {
        // A run of one repeat names its set without the `-r<N>` suffix.
        let bare = format!("{set}-{}", model.id);
        if !gather::set_is_complete(results, &bare)
            && !(1..=gather::MAX_EXTENDED_REPEATS)
                .any(|n| gather::set_is_complete(results, &format!("{bare}-r{n}")))
        {
            bail!(
                "--extend-set {set}: no gathered set of {} under {} (expected <date>-<gpu>-{}-<time>-r<N>)",
                model.id,
                results.display(),
                model.id
            );
        }
    }
    Ok(set.to_string())
}

/// The run a `run` command line asks for, under `id`.
fn requested_run(a: &RunArgs, models: &[Model], sha: &str, id: String) -> Run {
    Run {
        id,
        kind: "sweep".into(),
        git_ref: format!("{}={sha}", a.git_ref),
        scenario: a.scenario.clone(),
        repeats: a.repeats,
        models: models.iter().map(|m| m.id.clone()).collect(),
        gpu_types: a.gpu_types.clone(),
        image: a.image.clone(),
        status: "new".into(),
        pod_id: None,
    }
}

fn resolve_run<P: PodApi + ?Sized>(
    db: &Db,
    rp: &P,
    a: &RunArgs,
    models: &[Model],
    sha: &str,
) -> Result<(Run, Option<(String, u16)>)> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let run = requested_run(a, models, sha, format!("run-{now}-{}", std::process::id()));
    let owner = ExecOwner::current().context("claim run owner identity")?;
    // Another process's run is never touched: only an unfinished run of this same request whose owner process
    // is gone is resumed. A different request, or a live owner, gets this process its own run.
    let provision_window = a.provision_timeout_s.saturating_add(a.ssh_timeout_s);
    let cutoff = iso_from_secs(i64::try_from(now.saturating_sub(provision_window)).unwrap_or(0));
    for candidate in db.resumable_runs(&run, |o| o.local_liveness())? {
        let state = match &candidate.pod_id {
            Some(pid) => {
                let within = db
                    .pod_created_at(pid)?
                    .is_none_or(|created| created > cutoff);
                pod_state(rp, pid, within)
            }
            None => PodState::Unknown,
        };
        match resume_decision(&candidate, state) {
            Resume::Reattach(h, p) => {
                log(format!(
                    "resuming run {} (reattaching to pod {} at {h}:{p})",
                    candidate.id,
                    candidate.pod_id.as_deref().unwrap_or("?")
                ));
                db.set_owner(&candidate.id, &owner)?;
                return Ok((candidate, Some((h, p))));
            }
            Resume::Reprovision => {
                log(format!(
                    "resuming run {} (no live pod yet; will provision)",
                    candidate.id
                ));
                db.set_owner(&candidate.id, &owner)?;
                return Ok((candidate, None));
            }
            Resume::AbortStale => {
                log(format!(
                    "run {}'s pod is gone per RunPod; aborting it",
                    candidate.id
                ));
                abort_stale_run(db, rp, &candidate)?;
            }
            Resume::Leave => log(format!(
                "run {}'s pod is booting or RunPod could not be asked; leaving it alone",
                candidate.id
            )),
        }
    }
    db.create_run_with_owner(&run, Some(&owner))?;
    log(format!("created run {}", run.id));
    // A failed or aborted run of the same request in the last day is the one this resumes: its result sets and
    // this run's are one set (`<set>-<model>-rN`), and the repeats already gathered are not run again.
    // `--extend-set` names the sets instead: a finished run's, to which this run adds repeats.
    let set = match a.extend_set.as_deref() {
        Some(set) => Some(extension_set(&results_dir()?, set, models)?),
        None => db.resumable_set_id(&run, now, RESUME_WINDOW_S)?,
    };
    if let Some(set) = set {
        db.set_set_id(&run.id, &set)?;
        log(format!(
            "continuing result set {set}: repeats already gathered are skipped"
        ));
    }
    // Adopt a live idle pod with the same image instead of provisioning (spec 030); this skips the image
    // pull and model download. If RunPod no longer has it, mark the record terminated and provision.
    let live = db.live_pods()?;
    // `--reuse-pod ID` names the pod instead: one of this database's, alive, and not driven by a live run.
    let wanted = match a.reuse_pod.as_deref() {
        Some(id) => {
            let pod = live.iter().find(|p| p.id == id).ok_or_else(|| {
                Unrecoverable(format!(
                    "--reuse-pod {id}: this state database has no such live pod"
                ))
            })?;
            if let Some(user) = pod.current_run.as_deref().filter(|_| pod.status == "busy") {
                let driven = db
                    .run_owner(user)?
                    .is_none_or(|o| o.local_liveness() != OwnerLiveness::Dead);
                if driven {
                    return Err(Unrecoverable(format!(
                        "--reuse-pod {id}: run {user} is still driving it"
                    ))
                    .into());
                }
            }
            Some(pod)
        }
        None => crate::db::pick_idle_pod(&live, &run.image),
    };
    let wanted = match wanted {
        Some(pod) if over_price_cap(pod.cost_per_hr, a.max_price_per_hour).is_some() => {
            let (cost, max) = (
                pod.cost_per_hr.unwrap_or_default(),
                a.max_price_per_hour.unwrap_or_default(),
            );
            if a.reuse_pod.is_some() {
                return Err(Unrecoverable(format!(
                    "--reuse-pod {}: it bills {cost:.2} per hour, over --max-price-per-hour {max}",
                    pod.id
                ))
                .into());
            }
            log(format!(
                "idle pod {} bills {cost:.2} per hour, over --max-price-per-hour {max}; not adopting it",
                pod.id
            ));
            None
        }
        wanted => wanted,
    };
    if let Some(pod) = wanted {
        match rp.get_pod(&pod.id).ok().and_then(|x| x.ssh_endpoint()) {
            Some((h, p)) => {
                log(format!(
                    "adopting idle pod {} for run {} ({h}:{p})",
                    pod.id, run.id
                ));
                db.set_pod(&run.id, &pod.id, Some(&h), Some(p))?;
                let mut adopted = run;
                adopted.pod_id = Some(pod.id.clone());
                return Ok((adopted, Some((h, p))));
            }
            None if a.reuse_pod.is_some() => {
                return Err(Unrecoverable(format!(
                    "--reuse-pod {}: RunPod has no running pod with an SSH endpoint under that id",
                    pod.id
                ))
                .into());
            }
            None => {
                log(format!("idle pod {} is gone; dropping it", pod.id));
                let _ = db.set_pod_status(&pod.id, "terminated");
            }
        }
    }
    Ok((run, None))
}

fn run_phases(live: &mut LiveRun<'_>) -> Result<()> {
    let (db, rp, run, models, scratch, a, cursor) = (
        live.db,
        live.rp,
        live.run,
        live.models,
        live.scratch,
        live.a,
        live.cursor,
    );
    let (endpoint, clock) = (&mut live.endpoint, &live.clock);
    let ssh_key = expand_tilde(&a.ssh_key);
    // Provision if there is no live endpoint.
    cursor.set(None, "provision");
    if endpoint.is_none() {
        db.set_status(&run.id, "provisioning")?;
        let pubkey = std::fs::read_to_string(expand_tilde(&format!("{}.pub", a.ssh_key)))
            .map_err(|e| Unrecoverable(format!("read SSH public key {}.pub: {e}", a.ssh_key)))?;
        let (pid, host, port) = provision(
            rp,
            a,
            pubkey.trim(),
            &run.id,
            db,
            clock,
            PROVISION_POLL_INTERVAL,
        )?;
        db.set_pod(&run.id, &pid, Some(&host), Some(port))?;
        *endpoint = Some((host, port));
    }
    let pod_id = db.pod_id_for_run(&run.id)?;
    let (host, port) = endpoint.clone().unwrap();
    // This run now holds the pod: record its endpoint and mark it busy (from `provisioning` or `idle`).
    if let Some(pid) = &pod_id {
        let _ = db.set_pod_endpoint(pid, &host, port);
        let _ = db.assign_pod(pid, &run.id);
    }
    let ep = ssh::Endpoint::new(host, port, ssh_key, scratch.known_hosts());
    cursor.set(None, "ready");
    ready_pod(
        &ep,
        Duration::from_secs(a.ssh_timeout_s),
        sweep_pod_ttl(a, models.len()),
    )?;
    let gpu = ep
        .ssh("nvidia-smi --query-gpu=name --format=csv,noheader")
        .map(|r| r.stdout.trim().to_string())
        .unwrap_or_default();
    log(format!("pod GPU: {gpu}"));
    // An extension adds repeats to sets measured elsewhere: on other hardware they would be another population.
    if let Some(set) = a.extend_set.as_deref() {
        hardware::require_same_hardware(&ep, &results_dir()?, set, models)?;
    }

    db.set_status(&run.id, "setup")?;
    setup_pod(db, run, models, &scratch.runner(), &ep, a, cursor)?;

    db.set_status(&run.id, "sweeping")?;
    let watch = Watch {
        pod_id: pod_id.as_deref(),
        cursor,
    };
    sweep(db, rp, run, models, &ep, a, &watch)?;

    db.set_status(&run.id, "gathering")?;
    cursor.set(None, "gather");
    gather(db, run, models, &ep)?;

    postprocess(run, db, a)?;
    Ok(())
}

/// The production [`Healing`]: one `run`'s pod, phases and local result sets.
struct LiveRun<'a> {
    db: &'a Db,
    rp: &'a RunPod,
    run: &'a Run,
    models: &'a [Model],
    scratch: &'a Scratch,
    a: &'a RunArgs,
    endpoint: Option<(String, u16)>,
    cursor: &'a Cursor,
    clock: SystemClock,
}

impl LiveRun<'_> {
    /// Repeats gathered locally and repeats wanted, over all models.
    fn gathered(&self) -> (usize, usize) {
        let results = results_dir().unwrap_or_default();
        self.models.iter().fold((0, 0), |(have, want), m| {
            let dirs = sweep_run_id(self.db, self.run, &m.id)
                .map(|base| self.run.model_snapshot_dirs(&base))
                .unwrap_or_default();
            (
                have + gather::repeats_present(&results, &dirs).len(),
                want + dirs.len(),
            )
        })
    }
}

impl Healing for LiveRun<'_> {
    fn attempt(&mut self, fresh_pod: bool) -> Result<()> {
        if fresh_pod {
            self.endpoint = None;
        }
        run_phases(self)
    }

    fn pod_reachable(&mut self) -> bool {
        let Some((host, port)) = self.endpoint.clone() else {
            return false;
        };
        let Ok(Some(pid)) = self.db.pod_id_for_run(&self.run.id) else {
            return false;
        };
        let ep = ssh::Endpoint::new(
            host,
            port,
            expand_tilde(&self.a.ssh_key),
            self.scratch.known_hosts(),
        );
        let mut wait = Duration::from_secs(10);
        let mut waited = Duration::ZERO;
        loop {
            if matches!(pod_state(self.rp, &pid, false), PodState::Gone) {
                return false;
            }
            if ep.ssh("true").is_ok_and(|r| r.ok()) {
                return true;
            }
            if waited >= SSH_GRACE {
                return false;
            }
            log(format!(
                "  pod {pid} not answering ssh; retrying in {}s",
                wait.as_secs()
            ));
            std::thread::sleep(wait);
            waited += wait;
            wait = (wait * 3 / 2).min(Duration::from_secs(60));
        }
    }

    fn reset_setup(&mut self, model: Option<&str>) -> Result<()> {
        let Some(model) = model else { return Ok(()) };
        self.db.set_model_setup(&self.run.id, model, false)?;
        // The sweep that hit the missing model keeps running on the pod: stop it and drop its log, so the next
        // attempt starts it again after the setup.
        if let Some((host, port)) = self.endpoint.clone() {
            let ep = ssh::Endpoint::new(
                host,
                port,
                expand_tilde(&self.a.ssh_key),
                self.scratch.known_hosts(),
            );
            let _ = ep.ssh(&stop_sweep_command(model));
        }
        Ok(())
    }

    fn discard_pod(&mut self, why: Replacement) {
        // A new pod has none of the models: every model's setup is done again before any repeat.
        for m in self.models {
            if let Err(e) = self.db.set_model_setup(&self.run.id, &m.id, false) {
                log(format!("  could not re-arm the setup of {}: {e:#}", m.id));
            }
        }
        let (have, want) = self.gathered();
        let (what, status) = match why {
            Replacement::Lost => ("lost", "provisioning (replacing a lost pod)"),
            Replacement::HostFault => ("has no working GPU", "provisioning (host fault, replaced)"),
        };
        if let Ok(Some(pid)) = self.db.pod_id_for_run(&self.run.id) {
            log(format!(
                "pod {pid} {what}, provisioning replacement, {have} of {want} repeats already gathered"
            ));
            for _ in 0..3 {
                match terminate_recorded_pod(self.db, self.rp, &pid) {
                    Ok(()) => break,
                    Err(e) => {
                        log(format!("  {e:#}"));
                        std::thread::sleep(Duration::from_secs(10));
                    }
                }
            }
        }
        let _ = self.db.clear_run_pod(&self.run.id);
        self.endpoint = None;
        let _ = self.db.set_status(&self.run.id, status);
    }

    fn skip_cell(&mut self, model: Option<&str>, phase: &str, reason: &str) -> Result<()> {
        let Some(model) = model else {
            return Ok(());
        };
        let results = results_dir()?;
        let dirs = self
            .run
            .model_snapshot_dirs(&sweep_run_id(self.db, self.run, model)?);
        let missing: Vec<&String> = dirs
            .iter()
            .filter(|dir| !gather::set_is_complete(&results, dir))
            .collect();
        // A sweep failure is one repeat's; a setup failure leaves every repeat of the model unrunnable.
        let doomed = if phase == "sweep" {
            &missing[..missing.len().min(1)]
        } else {
            &missing[..]
        };
        for dir in doomed {
            log(format!("recording {dir} as failed: {reason}"));
            heal::record_failed_cell(&results, dir, model, &self.run.scenario, reason)?;
        }
        Ok(())
    }

    fn pause(&mut self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// A detached git worktree of the requested ref, removed when dropped.
struct Worktree {
    repo: PathBuf,
    path: PathBuf,
}

impl Worktree {
    fn add(repo: &Path, path: PathBuf, sha: &str) -> Result<Worktree> {
        sh(
            std::process::Command::new("git")
                .args(["worktree", "add", "--detach"])
                .arg(&path)
                .arg(sha)
                .current_dir(repo),
            "git worktree add",
        )?;
        Ok(Worktree {
            repo: repo.to_path_buf(),
            path,
        })
    }
}

impl Drop for Worktree {
    fn drop(&mut self) {
        if let Err(e) = sh(
            std::process::Command::new("git")
                .args(["worktree", "remove", "--force"])
                .arg(&self.path)
                .current_dir(&self.repo),
            "git worktree remove",
        ) {
            log(format!("{e:#}"));
        }
    }
}

/// Build the runner at `git_ref` and return the patched copy in `scratch`. A ref other than HEAD builds in
/// a throwaway worktree under `scratch` that is removed before this returns.
fn build_poot_runner(git_ref: &str, sha: &str, scratch: &Scratch) -> Result<PathBuf> {
    let repo = repo_root()?;
    let head = sh(
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(&repo),
        "git rev-parse HEAD",
    )?
    .trim()
    .to_string();
    // Build in place for HEAD; otherwise in a clean worktree of the requested ref (cold target, reproducible).
    let worktree = if git_ref == "HEAD" || sha == head {
        None
    } else {
        let wt = scratch.path().join("worktree");
        log(format!(
            "adding worktree for {git_ref} at {} (cold build)",
            wt.display()
        ));
        Some(Worktree::add(&repo, wt, sha)?)
    };
    let build_repo = worktree.as_ref().map_or(repo, |wt| wt.path.clone());

    // The runner generates kernels at runtime, so a plain `cargo build --release` suffices.
    let runner_dir = build_repo.join("benchmarks").join("runners").join("poot");
    log(format!(
        "building the poot bench runner (release) in {}...",
        runner_dir.display()
    ));
    let built = build_runner_artifact("cargo", &runner_dir)?;
    patchelf_for_pod(&built, scratch)
}

const RUNNER_BIN: &str = "poot-bench-runner";

/// Build the runner package in `runner_dir` with `cargo` and return the executable cargo reports having
/// built for it. The path comes from cargo's own JSON messages, never from a guess at the target
/// directory: an exported `CARGO_TARGET_DIR` puts the binary elsewhere, and whatever stale binary sits
/// under `runner_dir/target` must not be uploaded as this build's.
fn build_runner_artifact(cargo: &str, runner_dir: &Path) -> Result<PathBuf> {
    let messages = sh(
        std::process::Command::new(cargo)
            .args(["build", "--release", "--message-format=json"])
            .current_dir(runner_dir),
        "cargo build --release",
    )?;
    runner_executable(&messages, &runner_dir.join("Cargo.toml"))
}

/// The runner executable named by a `cargo build --message-format=json` stream for the package at
/// `manifest`. Refuses when cargo reported none: nothing else is known to come from this build.
fn runner_executable(cargo_messages: &str, manifest: &Path) -> Result<PathBuf> {
    let manifest = manifest
        .canonicalize()
        .with_context(|| format!("resolve {}", manifest.display()))?;
    let mut executable = None;
    for message in cargo_messages
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
    {
        if message["reason"] != "compiler-artifact" || message["target"]["name"] != RUNNER_BIN {
            continue;
        }
        let is_bin = message["target"]["kind"]
            .as_array()
            .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bin"));
        let same_package = message["manifest_path"]
            .as_str()
            .and_then(|path| Path::new(path).canonicalize().ok())
            .is_some_and(|path| path == manifest);
        if let (true, true, Some(path)) = (is_bin, same_package, message["executable"].as_str()) {
            executable = Some(PathBuf::from(path));
        }
    }
    let executable = executable.ok_or_else(|| {
        anyhow!(
            "cargo did not report a `{RUNNER_BIN}` executable for {}; refusing to upload a binary \
             found by path",
            manifest.display()
        )
    })?;
    if !executable.is_file() {
        bail!(
            "cargo reported {} as the runner, but it is not a file",
            executable.display()
        );
    }
    Ok(executable)
}

/// Copy a locally built binary into `scratch` and patchelf its interpreter to the pod's glibc loader.
/// A nix-built binary names the nix loader, absent on the pod, so `execve` fails with ENOENT (surfacing
/// in the harness's Python as a misleading `FileNotFoundError`). Both the build-from-ref path and
/// `--poot-bin` go through here (a `--poot-bin` that skipped it broke run-1781808494). Without
/// `patchelf` there is no binary to upload, so this refuses.
fn patchelf_for_pod(src: &Path, scratch: &Scratch) -> Result<PathBuf> {
    if !src.exists() {
        bail!("poot runner not found: {}", src.display());
    }
    if !which("patchelf") {
        bail!(
            "patchelf not found: the runner would keep the nix loader and fail on the pod with rc 127 \
             (install patchelf, or enter the devshell that provides it)"
        );
    }
    let pod_bin = scratch.runner();
    std::fs::copy(src, &pod_bin)?;
    sh(
        std::process::Command::new("patchelf")
            .args(["--set-interpreter", "/lib64/ld-linux-x86-64.so.2"])
            .arg(&pod_bin),
        "patchelf",
    )?;
    Ok(pod_bin)
}

fn which(name: &str) -> bool {
    std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {name}"))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Provision settings shared by the bench sweep (`run`) and `exec`, both of which call `provision_pod`.
struct ProvisionCfg<'a> {
    pub(crate) gpu_types: &'a str,
    /// GPUs per pod (multi-GPU, e.g. tensor-parallel).
    pub(crate) gpu_count: u32,
    pub(crate) cloud: CloudChoice,
    /// The most a pod may bill per hour (USD); `None` has no cap. A pod RunPod prices above it is deleted at
    /// once and the cloud counts as a capacity miss.
    pub(crate) max_price_per_hour: Option<f64>,
    pub(crate) image: &'a str,
    pub(crate) container_disk_gb: u32,
    pub(crate) network_volume_id: Option<&'a str>,
    /// Data centers to pin the pod to (`dataCenterIds`); empty means no pin.
    pub(crate) data_center_ids: &'a [&'a str],
    /// Stop waiting for capacity after this long; `None` waits as long as it takes.
    pub(crate) give_up_after: Option<Duration>,
    pub(crate) timeout_s: u64,
    /// How often to ask whether the pod's SSH endpoint is published.
    pub(crate) poll_interval: Duration,
    /// The state database's pod prefix and the run the pods belong to; each attempt is one pod.
    pub(crate) owner: &'a PodOwnerPrefix,
    pub(crate) run_id: &'a str,
    /// The lowest host CUDA version a pod may report, or `None` for no filter (the default). RunPod's
    /// `allowedCudaVersions` enum stops at 13.0, so this is checked on the pod's reported `cudaVersion` after the
    /// create, not sent.
    pub(crate) min_cuda: Option<CudaVersion>,
}

/// The interval production waits between endpoint checks.
const PROVISION_POLL_INTERVAL: Duration = Duration::from_secs(8);

/// How long to wait after terminating a pod that reports less CUDA than the floor, before the next attempt.
const BELOW_FLOOR_PAUSE: Duration = Duration::from_secs(10);

/// A pod's reported CUDA version against the floor.
#[derive(Debug, PartialEq, Eq)]
enum CudaVerdict {
    /// No floor, or the pod reports at least the floor.
    Fine,
    /// The pod reports no (readable) CUDA version yet: the floor cannot be judged.
    Unknown,
    /// The pod reports this version, which is below the floor.
    Below(String),
}

/// Read the pod back from the API, log its GPU and reported CUDA version, and judge it against `floor`.
fn pod_cuda_verdict<P: PodApi + ?Sized>(
    pods: &P,
    pod_id: &str,
    gpu: &str,
    floor: Option<CudaVersion>,
) -> CudaVerdict {
    let reported = match pods.get_pod(pod_id) {
        Ok(pod) => pod.reported_cuda(),
        Err(e) => {
            log(format!("  get_pod {pod_id}: {e}"));
            None
        }
    };
    log(format!(
        "  pod {pod_id}: {gpu}, CUDA {}",
        reported.as_deref().unwrap_or("not reported yet")
    ));
    let Some(floor) = floor else {
        return CudaVerdict::Fine;
    };
    let Some(text) = reported else {
        return CudaVerdict::Unknown;
    };
    match text.parse::<CudaVersion>() {
        Ok(found) if found >= floor => CudaVerdict::Fine,
        Ok(_) => CudaVerdict::Below(text),
        Err(e) => {
            log(format!("  {e}; the floor {floor} is not checked"));
            CudaVerdict::Unknown
        }
    }
}

/// Terminate the pod when it reports less CUDA than the floor, record that as a capacity miss and pause briefly.
/// Returns whether the pod was rejected (the caller then goes on to the next attempt).
fn reject_below_floor<P: PodApi + ?Sized>(
    pods: &P,
    pod_id: &str,
    verdict: &CudaVerdict,
    floor: Option<CudaVersion>,
    clock: &dyn Clock,
    last: &mut anyhow::Error,
) -> Result<bool> {
    let (CudaVerdict::Below(found), Some(floor)) = (verdict, floor) else {
        return Ok(false);
    };
    log(format!(
        "  pod {pod_id} reports CUDA {found}, below the floor {floor}: terminating it and trying again"
    ));
    // A pod the API will not delete is still billing: stop here, do not open another.
    delete_pod_confirmed(pods, pod_id)?;
    *last = anyhow!("pod {pod_id} reports CUDA {found}, below the floor {floor}");
    clock.sleep(BELOW_FLOOR_PAUSE);
    Ok(true)
}

/// The host CUDA floor of a run: `--min-cuda-version` when given, else what the image declares through its
/// `org.poot.bench.min-cuda` label, else none. An unreachable registry or an unreadable label is logged and means
/// no floor: the check is an optimisation of where the pod lands, not a precondition.
fn cuda_floor(explicit: Option<CudaVersion>, image: &str) -> Option<CudaVersion> {
    if explicit.is_some() {
        return explicit;
    }
    match RegistryV2Resolver::new() {
        Ok(registry) => floor_for(None, image, &registry),
        Err(e) => {
            log(format!("  image labels unavailable ({e}); no CUDA floor"));
            None
        }
    }
}

/// [`cuda_floor`] with the label source given.
fn floor_for(
    explicit: Option<CudaVersion>,
    image: &str,
    labels: &dyn ImageLabels,
) -> Option<CudaVersion> {
    if explicit.is_some() {
        return explicit;
    }
    match image_min_cuda(labels, image) {
        Ok(Some(floor)) => {
            log(format!("  {image} declares a host CUDA floor of {floor}"));
            Some(floor)
        }
        Ok(None) => None,
        Err(e) => {
            log(format!("  image CUDA floor: {e}; no floor"));
            None
        }
    }
}

/// Poll until a pod publishes its direct-TCP SSH endpoint (`publicIp` + port 22 map).
/// Heartbeat every ~60s: a large image takes minutes to pull.
///
/// Returns `(host, port)` when the endpoint appears, or an error naming `timeout`.
pub(crate) fn wait_for_ssh_endpoint(
    mut get_pod: impl FnMut(&str) -> anyhow::Result<crate::runpod::Pod>,
    pod_id: &str,
    timeout: Duration,
    poll_interval: Duration,
) -> anyhow::Result<(String, u16)> {
    let start = Instant::now();
    let deadline = start + timeout;
    let mut last_beat = Instant::now();
    while Instant::now() < deadline {
        std::thread::sleep(poll_interval);
        match get_pod(pod_id) {
            Ok(cur) => {
                if let Some((h, p)) = cur.ssh_endpoint() {
                    log(format!(
                        "  endpoint up after {}s: {h}:{p}",
                        start.elapsed().as_secs()
                    ));
                    return Ok((h, p));
                }
                if last_beat.elapsed() >= Duration::from_secs(60) {
                    log(format!(
                        "  ...still waiting ({}s): desiredStatus={}, direct SSH endpoint not published yet (machine metadata {}; image pulling / scheduling)",
                        start.elapsed().as_secs(),
                        cur.desired_status,
                        if cur.machine_ready() {
                            "assigned"
                        } else {
                            "not assigned yet"
                        }
                    ));
                    last_beat = Instant::now();
                }
            }
            Err(e) => log(format!("  get_pod: {e}")),
        }
    }
    Err(anyhow!("no SSH endpoint within {}s", timeout.as_secs()))
}

/// Where a created pod landed: the cloud and the hourly price RunPod reported for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Placement {
    pub(crate) cloud: CloudType,
    /// USD per hour; `None` when the create response carried no price.
    pub(crate) cost_per_hr: Option<f64>,
}

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cost_per_hr {
            Some(cost) => write!(f, "{} {cost:.2} per hour", self.cloud),
            None => write!(f, "{} (no price reported)", self.cloud),
        }
    }
}

/// A cloud bills more per hour than `--max-price-per-hour` allows, by the catalog before a create or by the
/// created pod's own price after one. Waiting can fix it (another cloud, or the same one later), so it counts
/// as a capacity miss.
#[derive(Debug, thiserror::Error)]
#[error(
    "price cap: {cloud} bills {price:.2} per hour, over --max-price-per-hour {max} ({})",
    .pod_id.as_ref().map_or_else(|| "catalog price, nothing created".to_string(), |id| format!("pod {id} deleted"))
)]
struct OverPriceCap {
    cloud: CloudType,
    price: f64,
    max: f64,
    /// The pod that was created and deleted; `None` when the catalog refused the cloud before any create.
    pod_id: Option<String>,
}

/// How often a refused delete of an over-cap pod is tried, with 2 s, 4 s, 8 s between the tries.
const OVER_CAP_DELETE_ATTEMPTS: u32 = 4;

/// Delete `pod_id`, retrying with backoff. `false` when the API never confirmed it: the pod may still bill.
fn delete_with_backoff<P: PodApi + ?Sized>(pods: &P, pod_id: &str, clock: &dyn Clock) -> bool {
    for attempt in 1..=OVER_CAP_DELETE_ATTEMPTS {
        match pods.delete_pod(pod_id) {
            Ok(()) => return true,
            Err(e) => log(format!(
                "  delete of pod {pod_id} failed (try {attempt}): {e}"
            )),
        }
        if attempt < OVER_CAP_DELETE_ATTEMPTS {
            clock.sleep(Duration::from_secs(1 << attempt));
        }
    }
    false
}

/// Try once more to delete every pod an earlier delete failed on; the ones still refusing stay listed.
fn retry_strays<P: PodApi + ?Sized>(pods: &P, strays: &mut Vec<String>) {
    strays.retain(|id| match pods.delete_pod(id) {
        Ok(()) => {
            log(format!(
                "  pod {id} deleted after an earlier refused delete"
            ));
            false
        }
        Err(e) => {
            log(format!("  pod {id} is still not deletable: {e}"));
            true
        }
    });
}

/// The hourly price of a freshly created pod. The create response carries it, but when it does not and a cap
/// needs it, one read of the pod does. `None` is "RunPod did not say", never a price.
fn created_pod_price<P: PodApi + ?Sized>(pods: &P, pod: &Pod, capped: bool) -> Option<f64> {
    if pod.cost_per_hr.is_some() || !capped {
        return pod.cost_per_hr;
    }
    match pods.get_pod(&pod.id) {
        Ok(read) => read.cost_per_hr,
        Err(e) => {
            log(format!("  get_pod {} for its price: {e}", pod.id));
            None
        }
    }
}

/// What `--max-price-per-hour` needs besides the config: the catalog (when it could be read), the pods an
/// over-cap delete failed on, and the clock the delete backoff sleeps on.
struct PriceCheck<'a> {
    catalog: Option<&'a GpuCatalog>,
    strays: &'a mut Vec<String>,
    clock: &'a dyn Clock,
}

/// Create the pod of one attempt in the first of `cfg.cloud`'s clouds that accepts it, and return it
/// with where it landed. RunPod has no "any cloud" value, so `--cloud all` is this ordered fallback
/// ([`CloudChoice::tiers`]); the error is the last cloud's refusal.
///
/// Under `--max-price-per-hour`: a `catalog` price above the cap refuses the cloud before any create (and a
/// type the catalog says the cloud does not offer is skipped the same way), and a created pod whose own
/// price is above it is deleted as the backstop. Both are that cloud's capacity miss. A pod whose price
/// RunPod never reports is kept: the cap is not enforced on a price nobody knows. A refused delete of an
/// over-cap pod never aborts the run: it goes on `strays`, which the caller keeps deleting.
fn create_in_first_cloud<P: PodApi + ?Sized>(
    pods: &P,
    cfg: &ProvisionCfg,
    env: &HashMap<String, String>,
    gpu: &str,
    attempt: u32,
    checks: &mut PriceCheck<'_>,
) -> Result<(Pod, Placement)> {
    let mut last = anyhow!("no cloud to rent from");
    let mut refused_for_good = true;
    for &cloud_type in cfg.cloud.tiers() {
        if let (Some(max), Some(catalog)) = (cfg.max_price_per_hour, checks.catalog) {
            match catalog.price(gpu, cloud_type, cfg.gpu_count) {
                CatalogPrice::PerHour(price) if price > max => {
                    let refusal = OverPriceCap {
                        cloud: cloud_type,
                        price,
                        max,
                        pod_id: None,
                    };
                    log(format!("  {gpu} in {cloud_type}: {refusal}"));
                    refused_for_good = false;
                    last = refusal.into();
                    continue;
                }
                CatalogPrice::NotOffered => {
                    log(format!("  {gpu} is not offered in {cloud_type}; skipped"));
                    last = anyhow!("{gpu} is not offered in {cloud_type}");
                    continue;
                }
                CatalogPrice::PerHour(_) | CatalogPrice::Unlisted => {}
            }
        }
        let spec = CreateSpec {
            owner: cfg.owner,
            run_id: cfg.run_id,
            attempt,
            image: cfg.image,
            gpu_type_id: gpu,
            cloud_type,
            gpu_count: cfg.gpu_count,
            container_disk_gb: cfg.container_disk_gb,
            ports: &["22/tcp"],
            env,
            network_volume_id: cfg.network_volume_id,
            data_center_ids: cfg.data_center_ids,
        };
        match pods.create_pod(&spec) {
            Ok(pod) => {
                let placement = Placement {
                    cloud: cloud_type,
                    cost_per_hr: created_pod_price(pods, &pod, cfg.max_price_per_hour.is_some()),
                };
                log(format!("  pod {} created: {placement}", pod.id));
                let (Some(max), Some(price)) = (cfg.max_price_per_hour, placement.cost_per_hr)
                else {
                    return Ok((pod, placement));
                };
                if price <= max {
                    return Ok((pod, placement));
                }
                let refusal = OverPriceCap {
                    cloud: cloud_type,
                    price,
                    max,
                    pod_id: Some(pod.id.clone()),
                };
                if delete_with_backoff(pods, &pod.id, checks.clock) {
                    log(format!("  {refusal}"));
                } else {
                    log(format!(
                        "  WARNING: over-cap pod {} could not be deleted and may still be billing {price:.2} per hour; \
                         it is named under this state database's prefix, so `reap` removes it, and it is retried here",
                        pod.id
                    ));
                    checks.strays.push(pod.id);
                }
                refused_for_good = false;
                last = refusal.into();
            }
            Err(e) => {
                log(format!("  create in {cloud_type} failed: {e}"));
                refused_for_good &= !is_capacity_error(&e);
                last = e;
            }
        }
    }
    if refused_for_good {
        return Err(Unrecoverable(format!("{last}")).into());
    }
    Err(last)
}

/// Can waiting fix this create failure? A refusal that is not a capacity one (bad GPU type, cloud value, auth,
/// image) never can; a transport failure and a capacity refusal can.
fn is_capacity_error(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<CreateError>()
        .is_none_or(CreateError::is_capacity)
}

/// The seconds to wait after the `pass`-th full pass over the GPU types found no capacity: 20 s, then x1.5 per
/// pass, capped at five minutes.
pub(crate) fn capacity_backoff_secs(pass: u32) -> u64 {
    let mut secs = 20.0_f64;
    for _ in 1..pass {
        secs = (secs * 1.5).min(300.0);
    }
    secs.round().min(300.0) as u64
}

/// A little jitter on top of the backoff, up to a tenth of it, so concurrent runs do not retry in step.
fn jittered(base: u64) -> Duration {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::from(d.subsec_nanos()));
    Duration::from_millis(base * 1000 + nanos % (base * 100 + 1))
}

/// Provision one GPU pod and return its SSH endpoint. It cycles over `cfg.gpu_types` (each tried in `cfg.cloud`'s
/// clouds, cheapest first) and, while the only trouble is capacity (a refusal, or a pod that does not boot in time), waits
/// ([`capacity_backoff_secs`]) and cycles again, for as long as it takes or until `cfg.give_up_after`. A refusal
/// that waiting cannot fix fails at once as [`Unrecoverable`]. `on_create` gets the pod id right after creation
/// (before the SSH wait) so callers can persist it and never leak a pod on a crash; a pod that does not boot in
/// time is deleted before the next try. `waiting` is told the pass number whenever it starts waiting.
fn provision_pod<P: PodApi + ?Sized>(
    pods: &P,
    cfg: &ProvisionCfg,
    env: &HashMap<String, String>,
    // Called with (pod_id, gpu_type, placement) right after creation; all are the attempt's actual values.
    on_create: impl Fn(&str, &str, Placement) -> Result<()>,
    waiting: impl Fn(u32),
    clock: &dyn Clock,
) -> Result<(String, String, u16)> {
    let mut strays = Vec::new();
    let result = provision_attempts(pods, cfg, env, on_create, waiting, clock, &mut strays);
    retry_strays(pods, &mut strays);
    for id in &strays {
        log(format!(
            "WARNING: pod {id} was over the price cap and RunPod would not delete it: delete it on the RunPod \
             dashboard or with `reap`"
        ));
    }
    result
}

/// The line logged before a capacity wait. It names the price cap when the last miss was one, so a run that
/// sits waiting on `--max-price-per-hour` says so.
fn wait_line(
    last: &anyhow::Error,
    gpus: &[&str],
    pause: Duration,
    pass: u32,
    waited: Duration,
) -> String {
    let why = match last.downcast_ref::<OverPriceCap>() {
        Some(cap) => format!("price cap (--max-price-per-hour {})", cap.max),
        None => "capacity".to_string(),
    };
    format!(
        "no {why} for any of {}, waiting {}s (pass {pass}, waiting {} min so far)",
        gpus.join(", "),
        pause.as_secs(),
        waited.as_secs() / 60
    )
}

/// The pass loop of [`provision_pod`]; `strays` collects over-cap pods whose delete was refused.
fn provision_attempts<P: PodApi + ?Sized>(
    pods: &P,
    cfg: &ProvisionCfg,
    env: &HashMap<String, String>,
    on_create: impl Fn(&str, &str, Placement) -> Result<()>,
    waiting: impl Fn(u32),
    clock: &dyn Clock,
    strays: &mut Vec<String>,
) -> Result<(String, String, u16)> {
    // The catalog prices a cloud before anything is created. It only matters under a cap, and a catalog that
    // cannot be read leaves the post-create check to enforce the cap.
    let catalog = match cfg.max_price_per_hour {
        Some(_) => match pods.gpu_catalog() {
            Ok(catalog) => Some(catalog),
            Err(e) => {
                log(format!(
                    "gpu catalog unavailable ({e:#}); the price cap is checked on the created pod only"
                ));
                None
            }
        },
        None => None,
    };
    let gpus: Vec<&str> = cfg
        .gpu_types
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if gpus.is_empty() {
        return Err(Unrecoverable("no gpu types given".into()).into());
    }
    let mut attempt = 0u32;
    let mut last = anyhow!("no attempt made");
    for pass in 1u32.. {
        for gpu in &gpus {
            attempt += 1;
            log(format!(
                "provision pass {pass}, try {attempt}: {gpu} ({}) image={}",
                cfg.cloud, cfg.image
            ));
            retry_strays(pods, strays);
            let mut checks = PriceCheck {
                catalog: catalog.as_ref(),
                strays: &mut *strays,
                clock,
            };
            match create_in_first_cloud(pods, cfg, env, gpu, attempt, &mut checks) {
                Ok((pod, placement)) => {
                    // Hand the id to the caller immediately so a crash cannot leak the pod.
                    on_create(&pod.id, gpu, placement)?;
                    // What the API says about the pod it just created: logged on every create, and held
                    // against the CUDA floor when there is one.
                    let mut verdict = pod_cuda_verdict(pods, &pod.id, gpu, cfg.min_cuda);
                    if reject_below_floor(pods, &pod.id, &verdict, cfg.min_cuda, clock, &mut last)?
                    {
                        continue;
                    }
                    log(format!(
                        "  pod {} created; waiting up to {}s for SSH endpoint (large image - be patient)",
                        pod.id, cfg.timeout_s
                    ));
                    match wait_for_ssh_endpoint(
                        |id| pods.get_pod(id),
                        &pod.id,
                        Duration::from_secs(cfg.timeout_s),
                        cfg.poll_interval,
                    ) {
                        Ok((h, p)) => {
                            // The API reports no CUDA version until a host picks the pod up: ask again now that one has.
                            if matches!(verdict, CudaVerdict::Unknown) {
                                verdict = pod_cuda_verdict(pods, &pod.id, gpu, cfg.min_cuda);
                                if reject_below_floor(
                                    pods,
                                    &pod.id,
                                    &verdict,
                                    cfg.min_cuda,
                                    clock,
                                    &mut last,
                                )? {
                                    continue;
                                }
                            }
                            return Ok((pod.id, h, p));
                        }
                        Err(e) => {
                            last = e;
                            log(format!(
                                "  try {attempt} did not boot in time; terminating pod {}",
                                pod.id
                            ));
                            // A pod the API will not delete is still billing: stop here, do not open another.
                            delete_pod_confirmed(pods, &pod.id)?;
                        }
                    }
                }
                Err(e) if e.downcast_ref::<Unrecoverable>().is_some() => return Err(e),
                Err(e) => {
                    log(format!("  create failed: {e}"));
                    last = e;
                }
            }
        }
        let waited = clock.elapsed();
        if cfg.give_up_after.is_some_and(|limit| waited >= limit) {
            return Err(Cancelled(format!(
                "gave up on capacity after {} min and {pass} pass(es) over {}: {last}",
                waited.as_secs() / 60,
                gpus.join(", ")
            ))
            .into());
        }
        let pause = jittered(capacity_backoff_secs(pass));
        log(wait_line(&last, &gpus, pause, pass, waited));
        waiting(pass);
        clock.sleep(pause);
    }
    unreachable!("the pass loop only ends by returning")
}

fn provision<P: PodApi + ?Sized>(
    rp: &P,
    a: &RunArgs,
    pubkey: &str,
    run_id: &str,
    db: &Db,
    clock: &dyn Clock,
    poll_interval: Duration,
) -> Result<(String, String, u16)> {
    let mut env = HashMap::new();
    env.insert("PUBLIC_KEY".to_string(), pubkey.to_string());
    let min_cuda = cuda_floor(a.min_cuda_version, &a.image);
    // Single optional pin from `run --data-center`; empty means the field stays off the body.
    let data_center_ids: Vec<&str> = a.data_center.as_deref().into_iter().collect();
    // The prefix is minted here, before the first create call, so a pod created and then lost to a
    // crash is still named under a prefix `reap` can find.
    let owner = db.pod_owner_prefix()?;
    let cfg = ProvisionCfg {
        gpu_types: &a.gpu_types,
        gpu_count: 1,
        cloud: a.cloud,
        max_price_per_hour: a.max_price_per_hour,
        image: &a.image,
        container_disk_gb: a.container_disk_gb,
        network_volume_id: a.network_volume_id.as_deref(),
        data_center_ids: &data_center_ids,
        give_up_after: a.give_up_after_min.map(|m| Duration::from_secs(m * 60)),
        timeout_s: a.provision_timeout_s,
        poll_interval,
        owner: &owner,
        run_id,
        min_cuda,
    };
    // Persist the pod id right away so reap can find it after a crash, and record it in the `pods` table
    // as `provisioning` (spec 030); run_phases promotes it to `busy`.
    provision_pod(
        rp,
        &cfg,
        &env,
        |id, gpu, placement| {
            db.create_pod_record(
                id,
                gpu,
                &placement.cloud.to_string(),
                placement.cost_per_hr,
                &a.image,
            )?;
            db.set_pod(run_id, id, None, None)
        },
        |pass| {
            let _ = db.set_status(
                run_id,
                &format!("provisioning (waiting for capacity, pass {pass})"),
            );
        },
        clock,
    )
}

/// The python runners are baked into the image too; sync every file of each runner's directory (its `run.py` and
/// the helper modules and data beside it) so a runner fixed after the image build is the one that runs. Only the
/// files directly in a runner directory go up; `__pycache__` and `tests/` are subdirectories and stay behind.
/// (The binary runners, candle and poot, are not scripts.)
fn sync_python_runners<R: Remote + ?Sized>(remote: &R, bench: &Path) -> Result<()> {
    for runner in ["llamacpp", "vllm", "torch"] {
        let dir = bench.join("runners").join(runner);
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let path = entry?.path();
            if path.is_file() {
                files.push(path);
            }
        }
        files.sort();
        for file in files {
            remote.scp_up(
                file.to_str().unwrap(),
                &format!("/opt/benchmarks/runners/{runner}/"),
            )?;
        }
    }
    Ok(())
}

fn setup_pod(
    db: &Db,
    run: &Run,
    models: &[Model],
    poot_bin: &Path,
    ep: &ssh::Endpoint,
    a: &RunArgs,
    cursor: &Cursor,
) -> Result<()> {
    cursor.set(None, "upload");
    log("uploading the poot runner + syncing the current harness...");
    ep.scp_up(poot_bin.to_str().unwrap(), "/root/poot-bench-runner")?;
    ep.ssh_checked("chmod +x /root/poot-bench-runner")?;
    let bench = repo_root()?.join("benchmarks");
    for rel in [
        "run-sweep.sh",
        "runners.toml",
        "manifest.toml",
        "pyproject.toml",
        "uv.lock",
    ] {
        let src = bench.join(rel);
        if src.exists() {
            ep.scp_up(src.to_str().unwrap(), "/opt/benchmarks/")?;
        }
    }
    sync_python_runners(ep, &bench)?;
    for dir in ["harness", "schema", "prompts"] {
        let src = bench.join(dir);
        if src.exists() {
            ep.scp_up(src.to_str().unwrap(), "/opt/benchmarks/")?;
        }
    }
    ep.ssh_checked("chmod +x /opt/benchmarks/run-sweep.sh")?;
    // The sweep detaches via tmux. Baked into the image, but ensure it for older published images.
    ep.ssh("command -v tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux) 2>&1 | tail -2 || true")?;
    // Ensure the `hf` CLI. The slim image has no system pip, so install via uv into /root/.local/bin
    // (needs POD_PATH to find uv).
    ep.ssh(&format!("{POD_PATH} command -v hf >/dev/null 2>&1 || uv tool install --quiet huggingface_hub 2>&1 | tail -2 || true"))?;
    // Forward the local HF token (if any) so gated repos download. scp'd as a file so the secret never
    // lands in a command string that ssh errors would log.
    let hf_token = expand_tilde("~/.cache/huggingface/token");
    if Path::new(&hf_token).exists() {
        ep.ssh("mkdir -p /root/.cache/huggingface")?;
        ep.scp_up(&hf_token, "/root/.cache/huggingface/token")?;
        log("forwarded local HF token (enables gated downloads)");
    }

    for m in models {
        cursor.set(Some(&m.id), "setup");
        let st = db.model_state(&run.id, &m.id)?;
        if st.setup_done {
            log(format!("  {} already set up", m.id));
            continue;
        }
        let base = sweep_run_id(db, run, &m.id)?;
        let dirs = run.model_snapshot_dirs(&base);
        if gather::repeats_present(&results_dir()?, &dirs).len() == dirs.len() {
            log(format!("  {} is fully gathered locally; no setup", m.id));
            continue;
        }
        let repo = m
            .hf_repo
            .as_deref()
            .ok_or_else(|| anyhow!("{} has no hf_repo", m.id))?;
        // Download to the dir the harness reads (local_dir, else id).
        let dst = format!("{}/{}", a.models_root(), m.dir_name());
        // Skip the download if the attached volume already holds the weights.
        let cached = a.network_volume_id.is_some()
            && ep
                .ssh(&format!("{} && echo Y || echo N", cached_probe(m, &dst)))?
                .stdout
                .contains('Y');
        if cached {
            log(format!(
                "  {} present on volume ({dst}); skipping download",
                m.id
            ));
        } else {
            log(format!("  fetching {} ({repo}) -> {dst}", m.id));
            ep.ssh_checked(&download_command(m, repo, &dst))?;
        }
        // GGUF for llama.cpp (best-effort; a failure just makes that cell unsupported).
        if let Some(conv) = preconvert_command(m, &dst)
            && !ep.ssh(&conv)?.ok()
        {
            log(format!(
                "  GGUF convert for {} failed (llama.cpp cell will be skipped)",
                m.id
            ));
        }
        db.set_model_setup(&run.id, &m.id, true)?;
    }
    Ok(())
}

/// The `hf download` of one model into `dst`: its `hf_include` files only, or the whole repo when it names none.
fn download_command(m: &Model, repo: &str, dst: &str) -> String {
    let include: String = m
        .hf_include
        .iter()
        .map(|pattern| format!(" --include '{pattern}'"))
        .collect();
    format!(
        "{POD_PATH} hf download {repo} --local-dir {dst}{include} >/tmp/dl-{}.log 2>&1",
        m.id
    )
}

/// A shell test that `dst` already holds the model: its `weights_file`, else the HF `config.json`.
fn cached_probe(m: &Model, dst: &str) -> String {
    let marker = m.weights_file.as_deref().unwrap_or("config.json");
    format!("test -f {dst}/{marker}")
}

/// The f16 GGUF conversion run on the pod for llama.cpp, for a model that needs one
/// ([`Model::needs_gguf_preconversion`]).
fn preconvert_command(m: &Model, dst: &str) -> Option<String> {
    m.needs_gguf_preconversion().then(|| {
        format!(
            "test -f /opt/llama.cpp/convert_hf_to_gguf.py && /opt/tf-venv/bin/python \
             /opt/llama.cpp/convert_hf_to_gguf.py {dst} --outfile {dst}/{id}-f16.gguf --outtype f16 >/tmp/gguf-{id}.log 2>&1",
            id = m.id
        )
    })
}

/// Is the pod alive per the RunPod API? Needed because `ssh::Endpoint::ssh` returns `Ok` with empty
/// output for an unreachable host, so a pod deleted out-of-band would leave the sweep polling an empty
/// log until the hours-long timeout. `get_pod` errs on 404; a non-`RUNNING` desired status means
/// stopped or terminated. Callers require consecutive failures to ride out API blips.
fn pod_alive<P: PodApi + ?Sized>(rp: &P, pid: &str) -> bool {
    // Only a definite answer (404, EXITED, TERMINATED) is "not alive"; an API blip is not.
    !matches!(pod_state(rp, pid, true), PodState::Gone)
}

/// The `bench run --run-id` of one model's sweep on the pod, and the stem of its result sets: the run's set id
/// (its own id, or the failed run it resumes) and the model.
fn sweep_run_id(db: &Db, run: &Run, model: &str) -> Result<String> {
    Ok(format!("{}-{model}", db.set_id(&run.id)?))
}

/// The local `benchmarks/results` directory.
fn results_dir() -> Result<PathBuf> {
    Ok(repo_root()?.join("benchmarks").join("results"))
}

/// One incremental gather pass over `dirs`: the complete sets on the pod that are not yet local are copied,
/// each checked with the harness's identity rule.
fn gather_pass(ep: &dyn gather::PodFiles, dirs: &[String]) -> Result<Gathered> {
    let results = results_dir()?;
    let harness = repo_root()?.join("benchmarks").join("harness");
    let (staging, rejected) = gather::side_dirs(&results);
    gather::gather_sets(ep, &results, &staging, &rejected, dirs, &|set| {
        gather::harness_identity_errors(&harness, set)
    })
}

/// The shell command that sweeps one model on the pod. Provenance is not passed in: the poot runner embeds its
/// own build sha at build time and reports it, with its backend, on every result row.
/// `POD_PATH` restores uv, the poot bin and cuda that the ssh shell drops.
fn sweep_command(
    models_root: &str,
    model: &str,
    run_id: &str,
    scenario: &str,
    repeats: u32,
    present: &[u32],
) -> String {
    // Repeats whose sets are already local are not run again: the sets of a resumed run are one set.
    let skip = if present.is_empty() {
        String::new()
    } else {
        let list: Vec<String> = present.iter().map(u32::to_string).collect();
        format!(" --skip-repeats {}", list.join(","))
    };
    format!(
        "{POD_PATH} cd /opt/benchmarks && MODELS_DIR={models_root} POOT_BENCH_POOT_BIN=/root/poot-bench-runner \
         ./run-sweep.sh --model {model} --scenario {scenario} --run-id {run_id} --repeats {repeats}{skip} --skip-unsupported"
    )
}

/// The tmux session and the log file of one model's sweep on the pod.
fn sweep_session(model: &str) -> String {
    format!("sweep_{}", model.replace(['.', '-'], "_"))
}

fn sweep_log(model: &str) -> String {
    format!("/tmp/sweep-{model}.log")
}

/// The shell command that stops a model's sweep on the pod and drops its log, so the sweep starts again from scratch.
fn stop_sweep_command(model: &str) -> String {
    format!(
        "tmux kill-session -t {} 2>/dev/null; rm -f {}",
        sweep_session(model),
        sweep_log(model)
    )
}

/// What the sweep loop reports to and reads about the pod it watches.
struct Watch<'a> {
    pod_id: Option<&'a str>,
    cursor: &'a Cursor,
}

fn sweep(
    db: &Db,
    rp: &RunPod,
    run: &Run,
    models: &[Model],
    ep: &ssh::Endpoint,
    a: &RunArgs,
    watch: &Watch<'_>,
) -> Result<()> {
    let (pod_id, cursor) = (watch.pod_id, watch.cursor);
    for m in models {
        cursor.set(Some(&m.id), "sweep");
        let st = db.model_state(&run.id, &m.id)?;
        if st.sweep_done {
            log(format!("  {} sweep already done", m.id));
            continue;
        }
        let base = sweep_run_id(db, run, &m.id)?;
        let dirs = run.model_snapshot_dirs(&base);
        let present = gather::repeats_present(&results_dir()?, &dirs);
        if present.len() == dirs.len() {
            log(format!(
                "  {}: all {} repeat(s) already gathered locally; nothing to sweep",
                m.id,
                dirs.len()
            ));
            db.set_model_sweep(&run.id, &m.id, true, Some(&base))?;
            db.set_model_gathered(&run.id, &m.id)?;
            continue;
        }
        let remaining = u64::try_from(dirs.len() - present.len()).unwrap_or(1);
        let sess = sweep_session(&m.id);
        let logf = sweep_log(&m.id);
        // Done only if the wrapper appended SWEEP_EXIT=0. A non-zero marker is a failed prior attempt and
        // is re-run, not reported as complete.
        let succeeded = ep
            .ssh(&format!(
                "grep -q 'SWEEP_EXIT=0' {logf} 2>/dev/null && echo Y || echo N"
            ))?
            .stdout
            .contains('Y');
        // A live tmux session means a sweep is in flight; resume reattaches.
        let running = ep
            .ssh(&format!(
                "tmux has-session -t {sess} 2>/dev/null && echo Y || echo N"
            ))?
            .stdout
            .contains('Y');
        if !succeeded {
            if !running {
                log(format!("=== sweep: {} (tmux {sess}) ===", m.id));
                let cmd = sweep_command(
                    a.models_root(),
                    &m.id,
                    &base,
                    &run.scenario,
                    run.repeats,
                    &present,
                );
                start_sweep(ep, &sess, &logf, &cmd)?;
            } else {
                log(format!(
                    "=== sweep: {} (reattaching to live tmux {sess}) ===",
                    m.id
                ));
            }
        } else {
            log(format!("  {} sweep already complete (SWEEP_EXIT=0)", m.id));
        }
        // Poll the logfile for progress and completion (skipped if the sweep already succeeded).
        if !succeeded {
            let deadline =
                Instant::now() + Duration::from_secs(a.sweep_timeout_s.saturating_mul(remaining));
            let mut offset = 0usize;
            // Pod-liveness watchdog: probe the RunPod API every PROBE_EVERY iterations (20s each, ~3 min)
            // and abort after MAX_STRIKES consecutive down readings (~9 min), so a deleted pod fails
            // fast while a transient API blip is forgiven.
            const PROBE_EVERY: u32 = 9;
            const MAX_STRIKES: u32 = 3;
            // Copy the sets the harness has finished every GATHER_EVERY iterations (~2 min), so a pod that
            // vanishes loses only the repeat in flight.
            const GATHER_EVERY: u32 = 6;
            let mut iters = 0u32;
            let mut strikes = 0u32;
            let mut unreachable_since: Option<Instant> = None;
            let mut mirroring = gather::Mirroring::default();
            loop {
                if Instant::now() >= deadline {
                    gather_quietly(ep, &dirs)?;
                    return Err(PodLost {
                        pod_id: pod_id.unwrap_or("?").to_string(),
                        reason: "the sweep timed out",
                    }
                    .into());
                }
                std::thread::sleep(Duration::from_secs(20));
                iters += 1;
                if iters.is_multiple_of(GATHER_EVERY) {
                    gather_and_probe(ep, &dirs)?;
                }
                if iters.is_multiple_of(PROBE_EVERY)
                    && let Some(pid) = pod_id
                {
                    if pod_alive(rp, pid) {
                        strikes = 0;
                    } else {
                        strikes += 1;
                        log(format!(
                            "  {}: pod {pid} not alive per RunPod API (strike {strikes}/{MAX_STRIKES})",
                            m.id
                        ));
                        if strikes >= MAX_STRIKES {
                            // The pod is gone: nothing more to copy, and a fault in what it wrote changes nothing.
                            let _ = gather_quietly(ep, &dirs);
                            return Err(PodLost {
                                pod_id: pid.to_string(),
                                reason: "RunPod says it is not running",
                            }
                            .into());
                        }
                    }
                }
                let polled = ep.ssh(&format!("tail -c +{} {logf} 2>/dev/null", offset + 1))?;
                // ssh exits 255 when it cannot connect. A pod that stays unreachable past the grace window is
                // lost; shorter trouble is waited out.
                if polled.code == SSH_UNREACHABLE {
                    let since = *unreachable_since.get_or_insert_with(Instant::now);
                    if since.elapsed() >= SSH_GRACE {
                        let _ = gather_quietly(ep, &dirs);
                        return Err(PodLost {
                            pod_id: pod_id.unwrap_or("?").to_string(),
                            reason: "ssh unreachable beyond the grace window",
                        }
                        .into());
                    }
                    if let Some(line) = mirroring.tick(ep, &results_dir()?, &dirs, &polled) {
                        log(line);
                    }
                    log(format!(
                        "  {}: ssh unreachable for {}s (grace {}s)",
                        m.id,
                        since.elapsed().as_secs(),
                        SSH_GRACE.as_secs()
                    ));
                    continue;
                }
                unreachable_since = None;
                let tail = polled.stdout.as_str();
                for line in tail.lines() {
                    if line.contains("[progress]")
                        || line.contains("[run]")
                        || line.trim_start().starts_with("->")
                    {
                        log(format!("  {}: {}", m.id, line.trim()));
                    }
                    if line.contains("[progress]") {
                        record_progress(&run.id, &m.id, line.trim());
                    }
                }
                offset += tail.len();
                // The cells finished so far, for the dashboard: after the poll, only when the pod answered it.
                if let Some(line) = mirroring.tick(ep, &results_dir()?, &dirs, &polled) {
                    log(line);
                }
                let done = ep
                    .ssh(&format!("grep -q SWEEP_EXIT {logf} && echo Y || echo N"))?
                    .stdout
                    .contains('Y');
                if done {
                    let exit = ep.ssh(&format!("grep SWEEP_EXIT {logf} | tail -1"))?.stdout;
                    let exit = exit.trim();
                    // The wrapper finished; bail on non-zero rather than gather an empty result set.
                    if !exit.contains("SWEEP_EXIT=0") {
                        bail!(
                            "{} sweep FAILED ({exit}); last log lines:\n{}",
                            m.id,
                            ep.ssh(&format!("tail -15 {logf}"))?.stdout.trim()
                        );
                    }
                    log(format!("  {} sweep finished ({exit})", m.id));
                    // A host that cannot run CUDA fails every cell with an error row, and the sweep still exits 0:
                    // read the log before gathering sets of errors, and replace the pod.
                    let tail = ep.ssh(&format!("tail -n 400 {logf} 2>/dev/null"))?.stdout;
                    if let Some(fault) = heal::host_fault_in(&tail) {
                        return Err(fault.into());
                    }
                    gather_quietly(ep, &dirs)?;
                    break;
                }
            }
        }
        // The harness wrote the run directories named by `sweep_run_id` (see `Run::model_snapshot_dirs`).
        db.set_model_sweep(&run.id, &m.id, true, Some(&base))?;
    }
    Ok(())
}

/// How long a pod may stay unreachable over ssh before it counts as lost.
const SSH_GRACE: Duration = Duration::from_secs(10 * 60);

/// Start a model's sweep in a detached tmux session on the pod. A dead GPU is caught here, before a repeat starts,
/// not after the harness has swept a repeat of errors.
fn start_sweep(pod: &dyn gather::PodFiles, sess: &str, logf: &str, cmd: &str) -> Result<()> {
    heal::require_gpu(pod)?;
    // `tee` truncates a stale failed log; pipefail makes SWEEP_EXIT=$? the sweep's exit, not tee's.
    let launch = format!(
        "tmux new-session -d -s {sess} 'set -o pipefail; {cmd} 2>&1 | tee {logf}; echo SWEEP_EXIT=$? >> {logf}'"
    );
    let started = pod.ssh(&launch)?;
    if !started.ok() {
        bail!(
            "remote command failed (rc={}): {launch}\n{}",
            started.code,
            started.stderr.trim()
        );
    }
    Ok(())
}

/// The periodic step of the sweep loop: copy the finished sets, then check the pod still has its GPU.
fn gather_and_probe(pod: &dyn gather::PodFiles, dirs: &[String]) -> Result<()> {
    gather_quietly(pod, dirs)?;
    heal::require_gpu(pod)
}

/// A gather pass during the sweep: a failure is logged, not fatal (the final gather reports what is missing),
/// except a [`HostFault`] in a copied set (the pod's GPU is gone: the sweep ends there and the pod is replaced) or a
/// [`SetupFault`] (the model is not on the pod: the sweep ends there and the model is set up again).
fn gather_quietly(ep: &dyn gather::PodFiles, dirs: &[String]) -> Result<()> {
    match gather_pass(ep, dirs) {
        Err(e)
            if e.downcast_ref::<HostFault>().is_some()
                || e.downcast_ref::<SetupFault>().is_some() =>
        {
            Err(e)
        }
        Err(e) => {
            log(format!("  incremental gather: {e:#}"));
            Ok(())
        }
        Ok(_) => Ok(()),
    }
}

/// Keep the harness's latest progress line where the dashboard can read it.
fn record_progress(run_id: &str, model: &str, line: &str) {
    let written = results_dir()
        .and_then(|results| gather::write_progress(&results, run_id, &format!("{model}: {line}")));
    if let Err(e) = written {
        log(format!("  progress line: {e:#}"));
    }
}

fn gather(db: &Db, run: &Run, models: &[Model], ep: &ssh::Endpoint) -> Result<()> {
    let results = results_dir()?;
    std::fs::create_dir_all(&results)?;
    for m in models {
        let st = db.model_state(&run.id, &m.id)?;
        let Some(base) = st.pod_run_id else { continue };
        if st.gathered {
            continue;
        }
        // Sets the sweep loop already copied are skipped; this pass takes what is left.
        let dirs = run.model_snapshot_dirs(&base);
        let got = gather_pass(ep, &dirs)?;
        let missing: Vec<&String> = dirs
            .iter()
            .filter(|dir| !gather::set_is_complete(&results, dir))
            .collect();
        if !missing.is_empty() {
            bail!(
                "{}: sets {missing:?} are not gathered (rejected by the identity check: {:?})",
                m.id,
                got.rejected
            );
        }
        db.set_model_gathered(&run.id, &m.id)?;
    }
    Ok(())
}

fn postprocess(run: &Run, db: &Db, a: &RunArgs) -> Result<()> {
    let bench = repo_root()?.join("benchmarks");
    log("regenerating results/INDEX.md (the dashboard auto-ingests benchmarks/results/)");
    let _ = sh(
        std::process::Command::new("uv")
            .args(["run", "bench", "index"])
            .current_dir(&bench),
        "bench index",
    );
    if a.commit {
        let results = results_dir()?;
        let rids: Vec<String> = db
            .snapshot_dirs(&run.id)?
            .into_iter()
            .filter_map(|pod_dir| gather::local_set(&results, &pod_dir))
            .collect();
        let repo = repo_root()?;
        let _ = sh(
            std::process::Command::new("git")
                .args(["add", "benchmarks/results/"])
                .current_dir(&repo),
            "git add",
        );
        let msg = format!("bench: sweep snapshot(s) {}", rids.join(", "));
        let _ = sh(
            std::process::Command::new("git")
                .args(["commit", "-q", "-m", &msg])
                .current_dir(&repo),
            "git commit",
        );
        log(format!("committed {} snapshot(s)", rids.len()));
    }
    Ok(())
}

// The pod id may have been set mid-run, so re-read it from the DB.
impl Run {
    pub(crate) fn pod_id_now(&self, db: &Db) -> Option<String> {
        db.pod_id_for_run(&self.id)
            .ok()
            .flatten()
            .or_else(|| self.pod_id.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use crate::image_resolve::ImageResolveError;
    use clap::Parser as _;
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;

    /// A machine that records what is uploaded to it.
    #[derive(Default)]
    struct UploadLog(RefCell<Vec<(String, String)>>);

    impl Remote for UploadLog {
        fn wait_ssh(&self, _timeout: Duration) -> Result<()> {
            Ok(())
        }

        fn scp_up(&self, local: &str, remote: &str) -> Result<()> {
            self.0.borrow_mut().push((local.into(), remote.into()));
            Ok(())
        }

        fn ssh_until(
            &self,
            _cmd: &str,
            _deadline: Instant,
        ) -> Result<ssh::Run, ssh::DeadlineError> {
            unreachable!("syncing the runners runs no command")
        }
    }

    #[test]
    fn python_runner_sync_uploads_helper_modules_beside_run_py() {
        let bench = repo_root().unwrap().join("benchmarks");
        let pod = UploadLog::default();
        sync_python_runners(&pod, &bench).unwrap();
        let uploaded = pod.0.borrow();
        let to_vllm: Vec<&str> = uploaded
            .iter()
            .filter(|(_, remote)| remote == "/opt/benchmarks/runners/vllm/")
            .map(|(local, _)| local.rsplit('/').next().unwrap())
            .collect();
        assert!(to_vllm.contains(&"run.py"), "{to_vllm:?}");
        assert!(to_vllm.contains(&"stream_itl.py"), "{to_vllm:?}");
        for runner in ["llamacpp", "torch"] {
            let dir = format!("/opt/benchmarks/runners/{runner}/");
            assert!(
                uploaded
                    .iter()
                    .any(|(local, remote)| *remote == dir && local.ends_with("/run.py")),
                "{runner} run.py not uploaded"
            );
        }
        assert!(
            uploaded
                .iter()
                .all(|(local, _)| !local.contains("__pycache__") && !local.contains("/tests/")),
            "{uploaded:?}"
        );
    }

    fn exec_run(id: &str) -> Run {
        Run {
            id: id.into(),
            kind: "exec".into(),
            git_ref: "abc".into(),
            scenario: "--test".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA L40S".into(),
            image: "img".into(),
            status: "running".into(),
            pod_id: None,
        }
    }

    /// A non-local owner identity: liveness cannot prove it dead, so normal reap must keep it
    /// protected (mirrors the db.rs unknown-owner fixtures without reaching into /proc).
    fn foreign_owner(pid: i64) -> ExecOwner {
        ExecOwner {
            hostname: "host-elsewhere".into(),
            pid,
            boot_id: "boot-elsewhere".into(),
            start_ticks: format!("{pid}00"),
            pid_ns: "pid:[host-elsewhere]".into(),
        }
    }

    /// `finalize_stale_claims` keeps an exec whose owner cannot be proven dead, and `force`
    /// (`reap --all`) aborts it.
    #[test]
    fn finalize_stale_claims_keeps_an_unprovable_exec_owner_and_force_aborts_it() {
        let db = Db::open(":memory:").unwrap();
        db.create_run_with_owner(&exec_run("exec-foreign"), Some(&foreign_owner(111)))
            .unwrap();

        let report = finalize_stale_claims(&db, false).unwrap();
        assert_eq!((report.aborted, report.kept_unknown), (0, 1));
        assert_eq!(
            db.run_status("exec-foreign").unwrap().as_deref(),
            Some("running")
        );

        let forced = finalize_stale_claims(&db, true).unwrap();
        assert_eq!(forced.aborted, 1);
        assert_eq!(
            db.run_status("exec-foreign").unwrap().as_deref(),
            Some("aborted")
        );
    }

    /// A fake RunPod: a list of live pods, and a switch that makes every delete fail.
    #[derive(Default)]
    struct FakePods {
        /// (id, name, has an SSH endpoint)
        cloud: RefCell<Vec<(String, String, bool)>>,
        refuse_delete: Cell<bool>,
        created: Cell<u32>,
        /// The image each create call asked for, in order.
        images: RefCell<Vec<String>>,
        /// The REST body of every create call, in order: what RunPod would have been sent.
        bodies: RefCell<Vec<serde_json::Value>>,
        /// Makes every SECURE create fail the way RunPod does when that cloud has no capacity.
        refuse_secure: Cell<bool>,
        /// Refuse this many create calls (whatever the cloud) with a capacity error, then accept.
        refuse_creates: Cell<u32>,
        /// Refuse every create call with this HTTP status and body.
        refuse_with: RefCell<Option<(u16, String)>>,
        /// Refuse every SECURE create call with this HTTP status and body (COMMUNITY is unaffected).
        refuse_secure_with: RefCell<Option<(u16, String)>>,
        /// Refuse every COMMUNITY create call with this HTTP status and body (SECURE is unaffected).
        refuse_community_with: RefCell<Option<(u16, String)>>,
        creates: Cell<u32>,
        /// The hourly price a SECURE / COMMUNITY pod reports on create (`None`: no price reported).
        secure_cost: Cell<Option<f64>>,
        community_cost: Cell<Option<f64>>,
        /// What a read of a pod reports as its price (the create response may omit it).
        read_cost: Cell<Option<f64>>,
        /// The GPU catalog `gpu_catalog` answers with; `None` makes the call fail.
        catalog: RefCell<Option<GpuCatalog>>,
        catalog_reads: Cell<u32>,
        /// The CUDA version the API reports for each pod created next, in order (`None`: not reported yet). A
        /// pod created with none queued reports nothing.
        next_cuda: RefCell<std::collections::VecDeque<Option<String>>>,
        /// The version reported for each pod created so far.
        reported_cuda: RefCell<HashMap<String, String>>,
        /// Ids of the pods deleted, in order.
        deleted: RefCell<Vec<String>>,
        /// For the pods created next: a version the API only reports from the third read of the pod on (it has no
        /// host at first), in order; consumed together with `next_cuda`.
        late_cuda: RefCell<std::collections::VecDeque<Option<String>>>,
        late_reported: RefCell<HashMap<String, String>>,
        reads: RefCell<HashMap<String, u32>>,
    }

    impl FakePods {
        fn refusing_deletes() -> Self {
            let pods = Self::default();
            pods.refuse_delete.set(true);
            pods
        }

        fn seed(&self, id: &str, name: &str, ssh: bool) {
            self.cloud.borrow_mut().push((id.into(), name.into(), ssh));
        }

        fn ids(&self) -> Vec<String> {
            self.cloud.borrow().iter().map(|p| p.0.clone()).collect()
        }

        fn to_pod((id, name, ssh): &(String, String, bool)) -> Pod {
            Pod {
                id: id.clone(),
                name: name.clone(),
                desired_status: "RUNNING".into(),
                public_ip: ssh.then(|| "1.2.3.4".to_string()),
                port_mappings: ssh.then(|| HashMap::from([("22".to_string(), 40022)])),
                machine: serde_json::Value::Null,
                cuda_version: None,
                cost_per_hr: None,
            }
        }
    }

    impl PodApi for FakePods {
        fn create_pod(&self, spec: &CreateSpec<'_>) -> Result<Pod> {
            let n = self.created.get() + 1;
            self.created.set(n);
            self.images.borrow_mut().push(spec.image.to_string());
            self.bodies
                .borrow_mut()
                .push(crate::runpod::create_pod_body(spec));
            self.creates.set(self.creates.get() + 1);
            if let Some((status, body)) = self.refuse_with.borrow().clone() {
                return Err(CreateError { status, body }.into());
            }
            if spec.cloud_type == CloudType::Secure
                && let Some((status, body)) = self.refuse_secure_with.borrow().clone()
            {
                return Err(CreateError { status, body }.into());
            }
            if spec.cloud_type == CloudType::Community
                && let Some((status, body)) = self.refuse_community_with.borrow().clone()
            {
                return Err(CreateError { status, body }.into());
            }
            if self.refuse_creates.get() > 0 {
                self.refuse_creates.set(self.refuse_creates.get() - 1);
                return Err(CreateError {
                    status: 500,
                    body: "There are no instances currently available".into(),
                }
                .into());
            }
            if self.refuse_secure.get() && spec.cloud_type == CloudType::Secure {
                return Err(CreateError {
                    status: 500,
                    body: "no SECURE capacity".into(),
                }
                .into());
            }
            let entry = (
                format!("fake-pod-{n}"),
                spec.owner.pod_name(spec.run_id, spec.attempt),
                true,
            );
            let pod = Pod {
                cost_per_hr: match spec.cloud_type {
                    CloudType::Secure => self.secure_cost.get(),
                    CloudType::Community => self.community_cost.get(),
                },
                ..Self::to_pod(&entry)
            };
            if let Some(Some(version)) = self.next_cuda.borrow_mut().pop_front() {
                self.reported_cuda
                    .borrow_mut()
                    .insert(entry.0.clone(), version);
            }
            if let Some(Some(version)) = self.late_cuda.borrow_mut().pop_front() {
                self.late_reported
                    .borrow_mut()
                    .insert(entry.0.clone(), version);
            }
            self.cloud.borrow_mut().push(entry);
            Ok(pod)
        }

        fn get_pod(&self, id: &str) -> Result<Pod> {
            let reads = {
                let mut reads = self.reads.borrow_mut();
                let n = reads.entry(id.to_string()).or_default();
                *n += 1;
                *n
            };
            self.cloud
                .borrow()
                .iter()
                .find(|p| p.0 == id)
                .map(|entry| Pod {
                    cost_per_hr: self.read_cost.get(),
                    cuda_version: self.reported_cuda.borrow().get(id).cloned().or_else(|| {
                        (reads >= 3)
                            .then(|| self.late_reported.borrow().get(id).cloned())
                            .flatten()
                    }),
                    ..Self::to_pod(entry)
                })
                .ok_or_else(|| PodNotFound(id.to_string()).into())
        }

        fn list_pods(&self) -> Result<Vec<Pod>> {
            Ok(self.cloud.borrow().iter().map(Self::to_pod).collect())
        }

        fn gpu_catalog(&self) -> Result<GpuCatalog> {
            self.catalog_reads.set(self.catalog_reads.get() + 1);
            self.catalog
                .borrow()
                .clone()
                .ok_or_else(|| anyhow!("catalog unreachable"))
        }

        fn delete_pod(&self, id: &str) -> Result<()> {
            if self.refuse_delete.get() {
                return Err(anyhow!("delete_pod {id} failed (500): refused"));
            }
            self.cloud.borrow_mut().retain(|p| p.0 != id);
            self.deleted.borrow_mut().push(id.to_string());
            Ok(())
        }
    }

    /// A scratch directory and, inside it, an empty `results` directory. Failed sets live beside `results`
    /// (`gather::failed_dir`), so keeping `results` one level down keeps them out of the shared temp dir, where a
    /// run's leftovers would make the next run see repeats as present.
    fn scratch_results(tag: &str) -> (PathBuf, PathBuf) {
        let root = unique_temp_dir(tag);
        let results = root.join("results");
        std::fs::create_dir_all(&results).unwrap();
        (root, results)
    }

    fn unique_temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "poot-orch-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn provision_cfg<'a>(owner: &'a PodOwnerPrefix, run_id: &'a str) -> ProvisionCfg<'a> {
        ProvisionCfg {
            gpu_types: "NVIDIA L40S",
            gpu_count: 1,
            cloud: CloudChoice::Secure,
            max_price_per_hour: None,
            image: "img",
            container_disk_gb: 10,
            network_volume_id: None,
            data_center_ids: &[],
            give_up_after: None,
            timeout_s: 1,
            poll_interval: Duration::ZERO,
            owner,
            run_id,
            min_cuda: None,
        }
    }

    fn sweep_run(id: &str, status: &str, pod_id: Option<&str>) -> Run {
        Run {
            id: id.into(),
            kind: "sweep".into(),
            git_ref: "HEAD=abc".into(),
            scenario: "decode-curve".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA L40S".into(),
            image: "img".into(),
            status: status.into(),
            pod_id: pod_id.map(String::from),
        }
    }

    /// SC-001: `provision_pod` creates a pod and the process dies before it is recorded (`on_create`
    /// fails). The pod exists in the cloud with no DB record, and `reap_targets` still selects it, by
    /// the state database's prefix. Selecting by the old `poot-bench-` prefix (or any prefix the
    /// created name does not carry) leaves it running.
    #[test]
    fn reap_selects_a_pod_whose_creation_crashed_before_it_was_recorded() {
        let db = Db::open(":memory:").unwrap();
        let owner = db.pod_owner_prefix().unwrap();
        let cloud = FakePods::default();

        let crash = provision_pod(
            &cloud,
            &provision_cfg(&owner, "run-1"),
            &HashMap::new(),
            |_, _, _| bail!("crashed before the pod was recorded"),
            |_| {},
            &FakeClock::default(),
        )
        .unwrap_err();
        assert!(format!("{crash:#}").contains("crashed before"), "{crash:#}");

        let pods = cloud.list_pods().unwrap();
        assert_eq!(pods.len(), 1, "the create call went through");
        assert_eq!(
            pods[0].name,
            owner.pod_name("run-1", 1),
            "named <owner prefix><run id>-a<attempt>"
        );
        assert!(
            db.live_pod_ids().unwrap().is_empty(),
            "the pod was never recorded"
        );

        let targets = reap_targets(&pods, &owner, &db.reap_protection().unwrap(), false);
        assert_eq!(targets, vec![pods[0].id.clone()]);

        assert_eq!(reap(&db, &cloud, false).unwrap(), 1);
        assert!(cloud.ids().is_empty(), "reap deleted the unrecorded pod");
    }

    /// SC-004: a pod that carries another state database's prefix is not selected, by `reap_targets` or
    /// by `reap` (forced or not). Matching every `poot-orch-` prefix selects the second database's pod.
    #[test]
    fn reap_never_selects_another_state_databases_pod() {
        let db_a = Db::open(":memory:").unwrap();
        let db_b = Db::open(":memory:").unwrap();
        let (owner_a, owner_b) = (
            db_a.pod_owner_prefix().unwrap(),
            db_b.pod_owner_prefix().unwrap(),
        );
        assert_ne!(
            owner_a, owner_b,
            "the two databases have their own prefixes"
        );
        let cloud = FakePods::default();
        cloud.seed("pod-a", &owner_a.pod_name("run-1", 1), true);
        cloud.seed("pod-b", &owner_b.pod_name("run-1", 1), true);

        let pods = cloud.list_pods().unwrap();
        for force in [false, true] {
            assert_eq!(
                reap_targets(&pods, &owner_a, &ReapProtection::default(), force),
                vec!["pod-a".to_string()],
                "force={force}: only database A's pod"
            );
        }

        assert_eq!(reap(&db_a, &cloud, true).unwrap(), 1);
        assert_eq!(
            cloud.ids(),
            vec!["pod-b".to_string()],
            "database B's pod survives A's forced reap"
        );
    }

    /// Reap is honest about a pod the API will not delete: every target is tried, the error names the
    /// pod, and the count of deleted pods does not include it.
    #[test]
    fn reap_reports_a_pod_the_api_would_not_delete() {
        let db = Db::open(":memory:").unwrap();
        let owner = db.pod_owner_prefix().unwrap();
        let cloud = FakePods::refusing_deletes();
        cloud.seed("stuck-pod", &owner.pod_name("run-1", 1), true);

        let err = reap(&db, &cloud, false).unwrap_err();
        assert!(
            format!("{err:#}").contains("stuck-pod"),
            "the error names the pod: {err:#}"
        );
        assert_eq!(cloud.ids(), vec!["stuck-pod".to_string()]);
    }

    /// SC-002, site 1 (`exec`): a refused delete returns the error and leaves the run non-terminal, so
    /// the pod stays recorded live and protected. The control shows the same call finishing the run once
    /// the API confirms. `let _ = delete_pod(..)` marks the run done and the pod unprotected.
    #[test]
    fn exec_teardown_failure_keeps_the_pod_recorded_live() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&exec_run("exec-1")).unwrap();
        db.set_pod("exec-1", "pod-1", None, None).unwrap();

        let stuck = FakePods::refusing_deletes();
        stuck.seed("pod-1", "poot-orch-x-exec-1-a1", true);
        let err = finish_exec_pod(&db, &stuck, "exec-1", "pod-1", "done").unwrap_err();
        assert!(format!("{err:#}").contains("pod-1"), "{err:#}");
        assert_eq!(db.run_status("exec-1").unwrap().as_deref(), Some("running"));
        assert_eq!(db.reap_protection().unwrap().pod_ids, vec!["pod-1"]);

        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-exec-1-a1", true);
        finish_exec_pod(&db, &cloud, "exec-1", "pod-1", "done").unwrap();
        assert_eq!(db.run_status("exec-1").unwrap().as_deref(), Some("done"));
        assert!(cloud.ids().is_empty());
    }

    /// SC-002, site 2 (`run` teardown): a refused delete returns the error, the pod record stays `busy`
    /// and the run keeps its phase. The control confirms the same call terminates the pod and finishes
    /// the run. `let _ = delete_pod(..)` records the pod terminated and the run done.
    #[test]
    fn run_teardown_failure_keeps_the_pod_recorded_live() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&sweep_run("run-1", "gathering", None))
            .unwrap();
        db.create_pod_record("pod-1", "L40S", "SECURE", None, "img")
            .unwrap();
        db.set_pod("run-1", "pod-1", None, None).unwrap();
        db.assign_pod("pod-1", "run-1").unwrap();
        let pod_status = |db: &Db| db.live_pods().unwrap().first().map(|p| p.status.clone());

        let stuck = FakePods::refusing_deletes();
        stuck.seed("pod-1", "poot-orch-x-run-1-a1", true);
        let err = finish_run(&db, &stuck, "run-1", Some("pod-1"), false, "done").unwrap_err();
        assert!(format!("{err:#}").contains("pod-1"), "{err:#}");
        assert_eq!(pod_status(&db).as_deref(), Some("busy"));
        assert_eq!(
            db.run_status("run-1").unwrap().as_deref(),
            Some("gathering")
        );

        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-run-1-a1", true);
        finish_run(&db, &cloud, "run-1", Some("pod-1"), false, "done").unwrap();
        assert_eq!(pod_status(&db), None, "terminated pods are not live");
        assert_eq!(db.run_status("run-1").unwrap().as_deref(), Some("done"));
        assert!(cloud.ids().is_empty());
    }

    fn run_args(extra: &[&str]) -> RunArgs {
        let mut argv = vec!["poot-orchestrator", "run"];
        argv.extend_from_slice(extra);
        let Command::Run(args) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("`run` parses to Command::Run");
        };
        args
    }

    fn named_models(ids: &[&str]) -> Vec<Model> {
        let toml: String = ids
            .iter()
            .map(|id| {
                format!("[[model]]\nid = \"{id}\"\nframeworks.poot = {{ precision = \"bf16\" }}\n")
            })
            .collect();
        parse_manifest_models(&toml).unwrap()
    }

    /// An unfinished sweep of exactly what `args` and `models` ask for, owned by `owner`, holding `pod`.
    fn stored_sweep(
        db: &Db,
        id: &str,
        args: &RunArgs,
        models: &[Model],
        owner: &ExecOwner,
        pod: Option<&str>,
    ) {
        let run = requested_run(args, models, "abc", id.into());
        db.create_run_with_owner(
            &Run {
                status: "sweeping".into(),
                ..run
            },
            Some(owner),
        )
        .unwrap();
        if let Some(pod) = pod {
            db.create_pod_record(pod, "L40S", "SECURE", None, "img")
                .unwrap();
            db.set_pod(id, pod, None, None).unwrap();
            db.assign_pod(pod, id).unwrap();
        }
    }

    fn live_owner() -> ExecOwner {
        ExecOwner::current().unwrap()
    }

    /// An owner process that no longer exists: this process's identity with another start time.
    fn dead_owner() -> ExecOwner {
        ExecOwner {
            start_ticks: "1".into(),
            ..ExecOwner::current().unwrap()
        }
    }

    /// SC-002, site 3 (`resolve_run` on a stale run): RunPod says the recorded pod does not exist, so the dead
    /// owner's run is aborted; a refused delete returns the error and neither the run nor the pod record
    /// changes. `let _ = delete_pod(..)` aborts the run and forgets the pod.
    #[test]
    fn stale_run_abort_failure_keeps_the_pod_recorded_live() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        stored_sweep(&db, "run-1", &args, &[], &dead_owner(), Some("pod-1"));

        let stuck = FakePods::refusing_deletes();
        let err = resolve_run(&db, &stuck, &args, &[], "abc").unwrap_err();
        assert!(format!("{err:#}").contains("pod-1"), "{err:#}");
        assert_eq!(db.run_status("run-1").unwrap().as_deref(), Some("sweeping"));
        assert_eq!(
            db.live_pods().unwrap().first().map(|p| p.status.as_str()),
            Some("busy")
        );

        let cloud = FakePods::default();
        let (fresh, endpoint) = resolve_run(&db, &cloud, &args, &[], "abc").unwrap();
        assert_ne!(fresh.id, "run-1", "a new run replaces the aborted one");
        assert_eq!(endpoint, None);
        assert_eq!(db.run_status("run-1").unwrap().as_deref(), Some("aborted"));
        assert!(
            db.live_pods().unwrap().is_empty(),
            "the pod record is terminated"
        );
    }

    /// A pod RunPod reports RUNNING without an SSH endpoint yet is booting, not gone: a second `run` of the same
    /// request leaves the run and its pod alone and starts its own. Mutation: judge a no-endpoint pod gone
    /// (`None if within_provision_window => Gone`) and the pod is deleted and the run aborted.
    #[test]
    fn a_booting_pod_is_not_aborted() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        stored_sweep(&db, "run-1", &args, &[], &dead_owner(), Some("pod-1"));
        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-run-1-a1", false);

        let (fresh, _) = resolve_run(&db, &cloud, &args, &[], "abc").unwrap();

        assert_ne!(fresh.id, "run-1");
        assert_eq!(db.run_status("run-1").unwrap().as_deref(), Some("sweeping"));
        assert_eq!(cloud.ids(), ["pod-1"]);
    }

    /// Two `run`s with different models in one state database each get their own run, and neither touches the
    /// other's pod, booting or not. Mutation: drop the request filter from `resumable_runs` (the old
    /// "the one active run") and the second run attaches to, then aborts, the first's.
    #[test]
    fn concurrent_runs_of_different_models_never_touch_each_other() {
        let db = Db::open(":memory:").unwrap();
        let a_models = named_models(&["a"]);
        let args = run_args(&[]);
        stored_sweep(&db, "run-a", &args, &a_models, &dead_owner(), Some("pod-a"));
        let cloud = FakePods::default();
        cloud.seed("pod-a", "poot-orch-x-run-a-a1", true);

        let (b, endpoint) = resolve_run(&db, &cloud, &args, &named_models(&["b"]), "abc").unwrap();

        assert_ne!(b.id, "run-a");
        assert_eq!(endpoint, None);
        assert_eq!(db.run_status("run-a").unwrap().as_deref(), Some("sweeping"));
        assert_eq!(cloud.ids(), ["pod-a"]);
        assert_eq!(
            db.pod_id_for_run("run-a").unwrap().as_deref(),
            Some("pod-a")
        );
    }

    /// A run whose owner process is alive is never resumed, even by the identical request, and reap never
    /// selects its pod. Mutation: drop the owner filter from `resumable_runs` and the identical request attaches.
    #[test]
    fn a_run_with_a_live_owner_is_not_resumed_and_its_pod_is_not_reaped() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        stored_sweep(&db, "run-1", &args, &[], &live_owner(), Some("pod-1"));
        let cloud = FakePods::default();
        let name = db.pod_owner_prefix().unwrap().pod_name("run-1", 1);
        cloud.seed("pod-1", &name, true);

        let (fresh, _) = resolve_run(&db, &cloud, &args, &[], "abc").unwrap();

        assert_ne!(fresh.id, "run-1");
        let targets = reap_targets(
            &cloud.list_pods().unwrap(),
            &db.pod_owner_prefix().unwrap(),
            &db.reap_protection().unwrap(),
            false,
        );
        assert!(targets.is_empty(), "{targets:?}");
    }

    /// The same request whose owner is dead is resumed: onto its live pod if there is one, else it provisions.
    #[test]
    fn a_matching_request_resumes_a_run_whose_owner_is_dead() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        stored_sweep(&db, "run-1", &args, &[], &dead_owner(), Some("pod-1"));
        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-run-1-a1", true);

        let (resumed, endpoint) = resolve_run(&db, &cloud, &args, &[], "abc").unwrap();

        assert_eq!(resumed.id, "run-1");
        assert!(endpoint.is_some());
        let owner = live_owner();
        let req = requested_run(&args, &[], "abc", "x".into());
        assert!(
            db.resumable_runs(&req, |o| if *o == owner {
                OwnerLiveness::Live
            } else {
                OwnerLiveness::Dead
            })
            .unwrap()
            .is_empty(),
            "the resuming process now owns it"
        );
    }

    /// Another model set, repeat count, GPU list or ref is another request: never resumed.
    #[test]
    fn a_non_matching_request_never_resumes() {
        let args = run_args(&[]);

        for (case, other, models) in [
            ("models", run_args(&[]), named_models(&["b"])),
            (
                "repeats",
                run_args(&["--repeats", "5"]),
                named_models(&["a"]),
            ),
            (
                "gpus",
                run_args(&["--gpu-types", "NVIDIA H200"]),
                named_models(&["a"]),
            ),
            ("ref", run_args(&["--git-ref", "v1"]), named_models(&["a"])),
        ] {
            let db = Db::open(":memory:").unwrap();
            stored_sweep(
                &db,
                "run-1",
                &args,
                &named_models(&["a"]),
                &dead_owner(),
                None,
            );
            let (run, _) = resolve_run(&db, &FakePods::default(), &other, &models, "abc").unwrap();
            assert_ne!(run.id, "run-1", "{case}");
            assert_eq!(
                db.run_status("run-1").unwrap().as_deref(),
                Some("sweeping"),
                "{case}"
            );
        }
    }

    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    /// A mocked registry: it answers every tag with one digest, or refuses, and records what it was asked.
    struct MockRegistry {
        answer: Result<&'static str, ImageResolveError>,
        asked: RefCell<Vec<String>>,
    }

    impl MockRegistry {
        fn resolving_to(digest: &'static str) -> Self {
            Self {
                answer: Ok(digest),
                asked: RefCell::default(),
            }
        }

        fn refusing() -> Self {
            Self {
                answer: Err(ImageResolveError::Unauthorized {
                    context: "GET manifest".into(),
                    detail: "denied".into(),
                }),
                asked: RefCell::default(),
            }
        }
    }

    impl DigestResolver for MockRegistry {
        fn resolve_digest(&self, image_tag: &str) -> Result<String, ImageResolveError> {
            self.asked.borrow_mut().push(image_tag.to_string());
            self.answer.clone().map(str::to_string)
        }
    }

    fn exec_args(image: &str) -> ExecArgs {
        let Command::Exec(args) = Cli::try_parse_from([
            "poot-orchestrator",
            "exec",
            "--cmd",
            "true",
            "--image",
            image,
        ])
        .unwrap()
        .command
        else {
            panic!("`exec` parses to Command::Exec");
        };
        args
    }

    /// SC-002: `exec` resolves its image tag against the registry and creates the pod from the digest
    /// reference; the run row records the digest reference too. Creating the pod from the tag records
    /// `ghcr.io/kikijiki/poot-bench:latest` as the create call's image and this goes red.
    #[test]
    fn exec_resolves_its_image_and_creates_the_pod_from_the_digest() {
        let db = Db::open(":memory:").unwrap();
        let cloud = FakePods::default();
        let registry = MockRegistry::resolving_to(DIGEST);
        let tag = "ghcr.io/kikijiki/poot-bench:latest";
        let by_digest = format!("ghcr.io/kikijiki/poot-bench@{DIGEST}");

        let pod = acquire_exec_pod(
            &db,
            &cloud,
            &registry,
            &exec_args(tag),
            "exec-1",
            &no_warm_probe(),
            &HashMap::new(),
            Duration::ZERO,
        )
        .unwrap();

        assert_eq!(*registry.asked.borrow(), [tag]);
        assert_eq!(
            cloud.images.borrow().as_slice(),
            std::slice::from_ref(&by_digest)
        );
        let runs = db.recent_runs(10).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].0.image, by_digest, "the run records what ran");
        assert_eq!(runs[0].0.pod_id.as_deref(), Some(pod.pod_id.as_str()));
    }

    /// `provision_pod` for `--cloud all` against `cloud` with the given clock; returns where the pod landed.
    fn provision_all_on(
        cloud: &FakePods,
        max_price_per_hour: Option<f64>,
        clock: &FakeClock,
    ) -> Result<Placement> {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cfg = ProvisionCfg {
            cloud: CloudChoice::All,
            max_price_per_hour,
            give_up_after: Some(Duration::ZERO),
            ..provision_cfg(&owner, "run-1")
        };
        let landed = RefCell::new(None);
        provision_pod(
            cloud,
            &cfg,
            &HashMap::new(),
            |_, _, placement| {
                *landed.borrow_mut() = Some(placement);
                Ok(())
            },
            |_| {},
            clock,
        )?;
        Ok(landed.into_inner().expect("on_create ran"))
    }

    fn provision_all(cloud: &FakePods, max_price_per_hour: Option<f64>) -> Result<Placement> {
        provision_all_on(cloud, max_price_per_hour, &FakeClock::default())
    }

    /// The `cloudType` of every create request sent, in order.
    fn sent_clouds(cloud: &FakePods) -> Vec<String> {
        cloud
            .bodies
            .borrow()
            .iter()
            .map(|b| b["cloudType"].as_str().unwrap().to_string())
            .collect()
    }

    /// A catalog pricing `NVIDIA L40S` (the GPU `provision_cfg` rents) per GPU and cloud.
    fn l40s_catalog(community: Option<f64>, secure: Option<f64>) -> GpuCatalog {
        GpuCatalog {
            gpus: vec![crate::runpod::GpuPrice {
                id: "NVIDIA_L40S".into(),
                display_name: "NVIDIA L40S".into(),
                community,
                secure,
            }],
        }
    }

    /// POOT-1326 SC-001: with `--cloud all` COMMUNITY (the cheaper cloud) is asked first, and when it has
    /// stock SECURE is never asked. Mutation: put SECURE first in `CloudChoice::All.tiers()` -> red.
    #[test]
    fn cloud_all_asks_community_first_and_stops_there() {
        let cloud = FakePods::default();
        cloud.community_cost.set(Some(0.22));
        cloud.secure_cost.set(Some(0.50));

        let landed = provision_all(&cloud, None).unwrap();

        assert_eq!(sent_clouds(&cloud), ["COMMUNITY"]);
        assert_eq!(
            landed,
            Placement {
                cloud: CloudType::Community,
                cost_per_hr: Some(0.22)
            }
        );
        assert_eq!(landed.to_string(), "COMMUNITY 0.22 per hour");
    }

    /// POOT-1326 SC-002: `--cloud all` is not a RunPod value (it answers 400 to `cloudType: ALL`): provisioning
    /// sends COMMUNITY, and SECURE only after COMMUNITY has no capacity, and records the cloud and the price
    /// the pod really landed at.
    #[test]
    fn cloud_all_falls_back_from_community_to_secure_on_the_wire() {
        let cloud = FakePods::default();
        cloud.refuse_community_with.replace(Some((
            500,
            "There are no instances currently available".into(),
        )));
        cloud.secure_cost.set(Some(2.49));

        let landed = provision_all(&cloud, None).unwrap();

        assert_eq!(sent_clouds(&cloud), ["COMMUNITY", "SECURE"]);
        assert_eq!(landed.to_string(), "SECURE 2.49 per hour");
        assert_eq!(cloud.ids().len(), 1);
    }

    /// POOT-1326 SC-003: the catalog prices COMMUNITY over the cap, so it is refused BEFORE any create: no
    /// pod is made, billed or deleted, and SECURE (within the cap) is the only create. Mutation: skip the
    /// catalog pre-check in `create_in_first_cloud` -> red (COMMUNITY is created and deleted first).
    #[test]
    fn the_catalog_refuses_an_over_cap_cloud_before_any_create() {
        let cloud = FakePods::default();
        *cloud.catalog.borrow_mut() = Some(l40s_catalog(Some(1.69), Some(0.5)));
        cloud.community_cost.set(Some(1.69));
        cloud.secure_cost.set(Some(0.5));

        let landed = provision_all(&cloud, Some(1.0)).unwrap();

        assert_eq!(
            sent_clouds(&cloud),
            ["SECURE"],
            "COMMUNITY was never created"
        );
        assert!(
            cloud.deleted.borrow().is_empty(),
            "nothing was created to delete"
        );
        assert_eq!(landed.to_string(), "SECURE 0.50 per hour");
        assert_eq!(cloud.catalog_reads.get(), 1);
    }

    /// The catalog price is per GPU: a two-GPU pod is priced at twice it against the cap.
    #[test]
    fn the_catalog_price_is_per_gpu_times_the_gpu_count() {
        let catalog = l40s_catalog(Some(0.6), Some(0.7));
        assert_eq!(
            catalog.price("NVIDIA L40S", CloudType::Community, 2),
            CatalogPrice::PerHour(1.2)
        );
        assert_eq!(
            catalog.price("NVIDIA_L40S", CloudType::Secure, 1),
            CatalogPrice::PerHour(0.7),
            "the catalog id spelling finds the type too"
        );
        assert_eq!(
            catalog.price("NVIDIA H100", CloudType::Secure, 1),
            CatalogPrice::Unlisted
        );
    }

    /// POOT-1326 SC-003: a catalog that omits the GPU type (or cannot be read) says nothing, so the price of
    /// the created pod decides: the dear pod is deleted and the next cloud is tried. A clean answer from the
    /// catalog is not needed for the cap to hold.
    #[test]
    fn without_a_catalog_price_the_created_pod_decides() {
        for catalog in [None, Some(GpuCatalog::default())] {
            let cloud = FakePods::default();
            *cloud.catalog.borrow_mut() = catalog;
            cloud.community_cost.set(Some(1.69));
            cloud.secure_cost.set(Some(0.5));

            let landed = provision_all(&cloud, Some(1.0)).unwrap();

            assert_eq!(sent_clouds(&cloud), ["COMMUNITY", "SECURE"]);
            assert_eq!(landed.cloud, CloudType::Secure);
            assert_eq!(
                *cloud.deleted.borrow(),
                ["fake-pod-1"],
                "the dear pod is gone"
            );
            assert_eq!(cloud.ids(), ["fake-pod-2"]);
        }
    }

    /// POOT-1326 SC-003: every cloud over the cap by the catalog creates nothing and waits (a capacity miss,
    /// not an unrecoverable refusal), here until `give_up_after`, and the reason is the price cap.
    #[test]
    fn a_catalog_over_the_cap_everywhere_creates_nothing_and_waits() {
        let cloud = FakePods::default();
        *cloud.catalog.borrow_mut() = Some(l40s_catalog(Some(1.69), Some(2.49)));

        let err = provision_all(&cloud, Some(1.0)).unwrap_err();

        assert!(err.downcast_ref::<Cancelled>().is_some(), "{err:#}");
        assert!(format!("{err:#}").contains("price cap"), "{err:#}");
        assert_eq!(cloud.created.get(), 0);
    }

    /// POOT-1326 SC-003: when the created pod of every cloud is over the cap nothing is kept.
    #[test]
    fn max_price_over_every_created_pod_keeps_no_pod() {
        let cloud = FakePods::default();
        cloud.community_cost.set(Some(1.69));
        cloud.secure_cost.set(Some(2.49));

        let err = provision_all(&cloud, Some(1.0)).unwrap_err();

        assert!(err.downcast_ref::<Cancelled>().is_some(), "{err:#}");
        assert!(
            format!("{err:#}").contains("over --max-price-per-hour 1"),
            "{err:#}"
        );
        assert!(cloud.ids().is_empty(), "no over-cap pod is left billing");
    }

    /// A create response without a price is read once more for it (a pod read carries `cost`), and that price
    /// decides.
    #[test]
    fn a_create_response_without_a_price_falls_back_to_reading_the_pod() {
        let cloud = FakePods::default();
        cloud.read_cost.set(Some(1.69));

        let err = provision_all(&cloud, Some(1.0)).unwrap_err();
        assert!(format!("{err:#}").contains("price cap"), "{err:#}");
        assert!(cloud.ids().is_empty(), "the read price was over the cap");

        cloud.read_cost.set(Some(0.5));
        let landed = provision_all(&cloud, Some(1.0)).unwrap();
        assert_eq!(landed.cost_per_hr, Some(0.5));
        assert_eq!(cloud.ids().len(), 1);
    }

    /// A pod whose price RunPod never reports is kept, never deleted for it: the cap cannot judge it.
    #[test]
    fn a_pod_with_no_price_anywhere_is_kept() {
        let cloud = FakePods::default();

        let landed = provision_all(&cloud, Some(1.0)).unwrap();

        assert_eq!(landed.cost_per_hr, None);
        assert_eq!(cloud.ids().len(), 1);
        assert!(cloud.deleted.borrow().is_empty());
    }

    /// POOT-1326: an over-cap pod RunPod will not delete does not abort the run: the delete is retried with
    /// backoff (2, 4, 8 s), the run goes on in the next cloud, and the stray stays for `reap`.
    #[test]
    fn a_refused_delete_of_an_over_cap_pod_does_not_abort_the_run() {
        let cloud = FakePods::refusing_deletes();
        cloud.community_cost.set(Some(1.69));
        cloud.secure_cost.set(Some(0.5));
        let clock = FakeClock::default();

        let landed = provision_all_on(&cloud, Some(1.0), &clock).unwrap();

        assert_eq!(landed.cloud, CloudType::Secure);
        assert_eq!(
            cloud.ids(),
            ["fake-pod-1", "fake-pod-2"],
            "the stray is still there"
        );
        assert_eq!(
            *clock.sleeps.borrow(),
            [2, 4, 8].map(Duration::from_secs),
            "backoff between the four delete tries"
        );
    }

    /// POOT-1326 SC-002: a GPU type only one cloud offers is a capacity miss on the other, not an error: the
    /// catalog says COMMUNITY does not offer it, so only SECURE is asked, and when SECURE has no capacity the
    /// run waits (here until `give_up_after`) instead of failing as unrecoverable.
    #[test]
    fn a_type_only_one_cloud_offers_is_a_miss_on_the_other_not_an_error() {
        let cloud = FakePods::default();
        *cloud.catalog.borrow_mut() = Some(l40s_catalog(None, Some(0.5)));

        let landed = provision_all(&cloud, Some(5.0)).unwrap();
        assert_eq!(sent_clouds(&cloud), ["SECURE"]);
        assert_eq!(landed.cloud, CloudType::Secure);

        let none = FakePods::default();
        *none.catalog.borrow_mut() = Some(l40s_catalog(None, Some(0.5)));
        none.refuse_secure.set(true);
        let err = provision_all(&none, Some(5.0)).unwrap_err();
        assert!(err.downcast_ref::<Cancelled>().is_some(), "{err:#}");
        assert!(err.downcast_ref::<Unrecoverable>().is_none(), "{err:#}");
    }

    /// The wait line names the price cap when the last miss was one, and plain capacity otherwise.
    #[test]
    fn the_wait_line_says_price_cap_when_the_last_miss_was_one() {
        let cap: anyhow::Error = OverPriceCap {
            cloud: CloudType::Community,
            price: 1.69,
            max: 1.0,
            pod_id: None,
        }
        .into();
        let line = |last: &anyhow::Error| {
            wait_line(
                last,
                &["NVIDIA L40S"],
                Duration::from_secs(20),
                1,
                Duration::ZERO,
            )
        };
        assert!(
            line(&cap).starts_with("no price cap (--max-price-per-hour 1) for any of NVIDIA L40S"),
            "{}",
            line(&cap)
        );
        assert!(line(&anyhow!("no SECURE capacity")).starts_with("no capacity for any of"));
    }

    /// POOT-1326: a capped `run` end to end through `provision`: the dear COMMUNITY pod is refused, the SECURE
    /// pod lands, and the `pods` record keeps its cloud and price.
    #[test]
    fn a_capped_run_records_the_pod_it_kept_with_its_cloud_and_price() {
        let db = Db::open(":memory:").unwrap();
        let cloud = FakePods::default();
        cloud.community_cost.set(Some(1.69));
        cloud.secure_cost.set(Some(0.5));
        let args = run_args(&["--cloud", "all", "--max-price-per-hour", "1.0"]);

        let (pod_id, _, _) = provision(
            &cloud,
            &args,
            "ssh-ed25519 AAAA",
            "run-1",
            &db,
            &FakeClock::default(),
            Duration::ZERO,
        )
        .unwrap();

        let record = db.pod_record(&pod_id).unwrap().expect("pod recorded");
        assert_eq!(
            (record.cloud.as_str(), record.cost_per_hr),
            ("SECURE", Some(0.5))
        );
        assert_eq!(cloud.ids(), [pod_id]);
    }

    /// POOT-1326: a capped `exec` end to end through `acquire_exec_pod`: the pod is refused above the cap,
    /// the kept one is recorded with its cloud and price and held busy by the exec run, and finishing the
    /// exec marks the record terminated.
    #[test]
    fn a_capped_exec_records_its_pod_cloud_and_price() {
        let db = Db::open(":memory:").unwrap();
        let cloud = FakePods::default();
        cloud.community_cost.set(Some(1.69));
        cloud.secure_cost.set(Some(0.5));
        let a = exec_args_with(&[
            "--cmd",
            "true",
            "--cloud",
            "all",
            "--max-price-per-hour",
            "1.0",
        ]);

        let pod = acquire_exec_pod(
            &db,
            &cloud,
            &MockRegistry::resolving_to(DIGEST),
            &a,
            "exec-1",
            &no_warm_probe(),
            &HashMap::new(),
            Duration::ZERO,
        )
        .unwrap();

        let record = db
            .pod_record(&pod.pod_id)
            .unwrap()
            .expect("exec pod recorded");
        assert_eq!(
            (record.cloud.as_str(), record.cost_per_hr),
            ("SECURE", Some(0.5))
        );
        assert_eq!(record.status, "busy");
        assert_eq!(record.current_run.as_deref(), Some("exec-1"));
        assert_eq!(
            *cloud.deleted.borrow(),
            ["fake-pod-1"],
            "the dear pod was refused"
        );

        finish_exec_pod(&db, &cloud, "exec-1", &pod.pod_id, "done").unwrap();
        assert_eq!(
            db.pod_record(&pod.pod_id).unwrap().unwrap().status,
            "terminated"
        );
    }

    /// POOT-1326: `--max-price-per-hour` also holds for reuse: an idle pod whose recorded price is over the
    /// cap is not adopted (a fresh pod is provisioned), and an explicit `--reuse-pod` of it is refused. A pod
    /// within the cap, or with no recorded price, is adopted. Mutation: skip the adoption cap check in
    /// `resolve_run` -> red.
    #[test]
    fn reuse_cannot_bypass_the_price_cap() {
        let image = run_args(&[]).image;
        let reuse = |cost: Option<f64>, extra: &[&str]| {
            let db = Db::open(":memory:").unwrap();
            db.create_pod_record("pod-warm", "H100", "SECURE", cost, &image)
                .unwrap();
            db.set_pod_endpoint("pod-warm", "1.2.3.4", 40022).unwrap();
            db.set_pod_status("pod-warm", "idle").unwrap();
            let cloud = FakePods::default();
            cloud.seed("pod-warm", "poot-orch-x-old-a1", true);
            let mut argv = vec!["--max-price-per-hour", "1.0"];
            argv.extend_from_slice(extra);
            resolve_run(&db, &cloud, &run_args(&argv), &named_models(&["m"]), "abc")
                .map(|(_, endpoint)| endpoint.is_some())
        };

        assert!(!reuse(Some(2.0), &[]).unwrap(), "over the cap: not adopted");
        let err = reuse(Some(2.0), &["--reuse-pod", "pod-warm"]).unwrap_err();
        assert!(err.downcast_ref::<Unrecoverable>().is_some(), "{err:#}");
        assert!(
            format!("{err:#}").contains("over --max-price-per-hour"),
            "{err:#}"
        );
        assert!(reuse(Some(0.5), &[]).unwrap(), "within the cap: adopted");
        assert!(reuse(None, &[]).unwrap(), "no recorded price: adopted");
    }

    /// POOT-1326: a kept-warm exec pod over the cap is not reused by `exec`: a fresh pod is created and the
    /// warm one is left alone until it expires. Mutation: drop the `warm_pod_over_cap` arm in
    /// `acquire_exec_pod` -> red (the exec tries to reach the warm pod's SSH).
    #[test]
    fn a_warm_exec_pod_over_the_cap_is_not_reused() {
        let image = format!("ghcr.io/kikijiki/poot-bench@{DIGEST}");
        let db = Db::open(":memory:").unwrap();
        let mut warm = exec_run("exec-warm");
        warm.image = image.clone();
        warm.status = "running".into();
        db.create_run_with_owner(&warm, Some(&ExecOwner::current().unwrap()))
            .unwrap();
        db.set_pod("exec-warm", "pod-warm", Some("1.2.3.4"), Some(40000))
            .unwrap();
        db.set_warm("exec-warm", &crate::clock::iso_in_minutes(30))
            .unwrap();
        db.create_pod_record("pod-warm", "H100", "SECURE", Some(2.0), &image)
            .unwrap();
        let cloud = FakePods::default();
        cloud.seed("pod-warm", "poot-orch-x-exec-warm-a1", true);
        let a = exec_args_with(&["--cmd", "true", "--max-price-per-hour", "1.0"]);

        let pod = acquire_exec_pod(
            &db,
            &cloud,
            &MockRegistry::resolving_to(DIGEST),
            &a,
            "exec-1",
            &no_warm_probe(),
            &HashMap::new(),
            Duration::ZERO,
        )
        .unwrap();

        assert_ne!(pod.pod_id, "pod-warm");
        assert_eq!(cloud.created.get(), 1);
        assert!(
            cloud.deleted.borrow().is_empty(),
            "the warm pod is left alone"
        );
    }

    /// A finished run is only DONE when every requested model has all its valid sets: the qwen3-4b run ended
    /// DONE with 0 of 15. The summary names each short model with its valid and requested counts; a set recorded as
    /// failed is not valid. Mutation: count `local_failed_set` as valid and the first model drops out of the line.
    #[test]
    fn a_model_with_no_valid_sets_makes_the_run_incomplete() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&["--repeats", "3"]);
        let models = named_models(&["qwen3-4b", "qwen3-0.6b", "gemma"]);
        let run = Run {
            repeats: 3,
            ..requested_run(&args, &models, "abc", "run-1".into())
        };
        db.create_run_with_owner(&run, Some(&live_owner())).unwrap();
        let (scratch, results) = scratch_results("incomplete");
        for m in &models {
            let dirs = run.model_snapshot_dirs(&sweep_run_id(&db, &run, &m.id).unwrap());
            for (n, dir) in dirs.iter().enumerate() {
                let valid = m.id == "gemma" || (m.id == "qwen3-0.6b" && n == 0);
                if valid {
                    let name = gather::canonical_dir(dir, Some("NVIDIA GeForce RTX 3090")).unwrap();
                    std::fs::create_dir_all(results.join(&name)).unwrap();
                    std::fs::write(
                        results.join(&name).join("env.json"),
                        r#"{"gpu_name":"NVIDIA GeForce RTX 3090"}"#,
                    )
                    .unwrap();
                    for file in ["results.jsonl", "report.md"] {
                        std::fs::write(results.join(&name).join(file), "x").unwrap();
                    }
                } else {
                    heal::record_failed_cell(&results, dir, &m.id, "decode-curve", "setup failed")
                        .unwrap();
                }
            }
        }

        let summary = incomplete_models(&run, &models, &results, &db).unwrap();

        assert_eq!(summary.as_deref(), Some("qwen3-4b 0/3, qwen3-0.6b 1/3"));
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// A `LiveRun` over an in-memory db with `models` set up, holding no pod (so replacing it calls no API).
    fn with_live_run<R>(models: &[&str], f: impl FnOnce(&mut LiveRun<'_>, &Db) -> R) -> R {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        let models = named_models(models);
        let run = requested_run(&args, &models, "abc", "run-1".into());
        db.create_run_with_owner(&run, Some(&live_owner())).unwrap();
        for m in &models {
            db.set_model_setup(&run.id, &m.id, true).unwrap();
        }
        let rp = RunPod::with_base("key".into(), "http://127.0.0.1:1").unwrap();
        let scratch = Scratch::new("live").unwrap();
        let cursor = Cursor::default();
        let mut live = LiveRun {
            db: &db,
            rp: &rp,
            run: &run,
            models: &models,
            scratch: &scratch,
            a: &args,
            endpoint: None,
            cursor: &cursor,
            clock: SystemClock::start(),
        };
        f(&mut live, &db)
    }

    /// POOT-1351: a replaced pod has none of the models, so every model's setup is done again (the qwen3-4b run
    /// kept `setup_done` across the replacement and swept 15 repeats without the model). Mutation: drop the loop
    /// in `LiveRun::discard_pod` and both flags stay set.
    #[test]
    fn replacing_a_pod_rearms_the_setup_of_every_model() {
        with_live_run(&["qwen3-4b", "qwen3-0.6b"], |live, db| {
            live.discard_pod(Replacement::Lost);

            for model in ["qwen3-4b", "qwen3-0.6b"] {
                assert!(
                    !db.model_state("run-1", model).unwrap().setup_done,
                    "{model}"
                );
            }
        });
    }

    /// POOT-1351: the dashboard says why a pod is being replaced; a host fault reads 'host fault, replaced'.
    /// Mutation: map `Replacement::HostFault` to the lost-pod status and the second assertion goes red.
    #[test]
    fn the_stored_status_says_whether_the_pod_was_lost_or_had_a_host_fault() {
        with_live_run(&["m"], |live, db| {
            live.discard_pod(Replacement::HostFault);
            assert_eq!(
                db.run_status("run-1").unwrap().as_deref(),
                Some("provisioning (host fault, replaced)")
            );
            live.discard_pod(Replacement::Lost);
            assert_eq!(
                db.run_status("run-1").unwrap().as_deref(),
                Some("provisioning (replacing a lost pod)")
            );
        });
    }

    /// POOT-1351: setting a model up again clears its flag, so the next attempt fetches it.
    #[test]
    fn resetting_a_models_setup_clears_its_flag() {
        with_live_run(&["a", "b"], |live, db| {
            live.reset_setup(Some("a")).unwrap();

            assert!(!db.model_state("run-1", "a").unwrap().setup_done);
            assert!(
                db.model_state("run-1", "b").unwrap().setup_done,
                "only that model"
            );
        });
    }

    /// Setting a model up again stops that model's sweep session and removes its log: the log's absence is what makes
    /// the next attempt start the sweep (a stale `SWEEP_EXIT=0` would read as done). Mutation: kill another
    /// model's session name and the assertion goes red.
    #[test]
    fn stopping_a_sweep_kills_its_tmux_session_and_drops_its_log() {
        assert_eq!(
            stop_sweep_command("qwen3-0.6b"),
            "tmux kill-session -t sweep_qwen3_0_6b 2>/dev/null; rm -f /tmp/sweep-qwen3-0.6b.log"
        );
    }

    /// A pod that answers the commands it is given, listing a GPU or not, remembering what it was asked.
    struct ScriptedPod {
        has_gpu: bool,
        asked: RefCell<Vec<String>>,
    }

    impl gather::PodFiles for ScriptedPod {
        fn ssh(&self, cmd: &str) -> Result<ssh::Run> {
            self.asked.borrow_mut().push(cmd.to_string());
            let listing = cmd.contains("nvidia-smi -L");
            Ok(ssh::Run {
                code: i32::from(listing && !self.has_gpu),
                stdout: String::new(),
                stderr: String::new(),
            })
        }
        fn scp_down(&self, _: &str, _: &str) -> Result<()> {
            unreachable!("no copy")
        }
    }

    /// POOT-1351: a dead GPU is caught before a repeat starts: the sweep is not launched on a pod that lists no
    /// GPU, and is launched (after the probe) on one that does. Mutation: remove `heal::require_gpu` from
    /// `start_sweep` and the dead pod is handed the tmux launch.
    #[test]
    fn a_sweep_is_not_started_on_a_pod_without_a_gpu() {
        let dead = ScriptedPod {
            has_gpu: false,
            asked: RefCell::new(Vec::new()),
        };
        let error = start_sweep(&dead, "sweep_m", "/tmp/sweep-m.log", "run-sweep").unwrap_err();
        assert!(error.downcast_ref::<HostFault>().is_some(), "{error:#}");
        assert!(
            dead.asked
                .borrow()
                .iter()
                .all(|c| !c.contains("tmux new-session")),
            "{:?}",
            dead.asked
        );

        let live = ScriptedPod {
            has_gpu: true,
            asked: RefCell::new(Vec::new()),
        };
        start_sweep(&live, "sweep_m", "/tmp/sweep-m.log", "run-sweep").unwrap();
        let asked = live.asked.borrow();
        assert!(asked[0].contains("nvidia-smi -L"), "{asked:?}");
        assert!(
            asked[1].contains("tmux new-session -d -s sweep_m"),
            "{asked:?}"
        );
    }

    /// POOT-1351: while a sweep runs, every periodic gather also checks the GPU is still there.
    /// Mutation: drop the probe from `gather_and_probe` and a pod that lost its GPU passes.
    #[test]
    fn the_periodic_gather_step_faults_a_pod_that_lost_its_gpu() {
        let dead = ScriptedPod {
            has_gpu: false,
            asked: RefCell::new(Vec::new()),
        };
        // No set is complete on the fake pod, so the gather itself copies nothing.
        let error = gather_and_probe(&dead, &[]).unwrap_err();
        assert!(error.downcast_ref::<HostFault>().is_some(), "{error:#}");
    }

    /// A db holding exec `exec-warm`'s kept-warm pod `pod-warm` (inside its window) and, when `listed`, RunPod
    /// listing it.
    fn warm_pod_fixture(listed: bool) -> (Db, FakePods, String) {
        let image = format!("ghcr.io/kikijiki/poot-bench@{DIGEST}");
        let db = Db::open(":memory:").unwrap();
        let mut warm = exec_run("exec-warm");
        warm.image = image.clone();
        warm.status = "running".into();
        db.create_run_with_owner(&warm, Some(&ExecOwner::current().unwrap()))
            .unwrap();
        db.set_pod("exec-warm", "pod-warm", Some("1.2.3.4"), Some(40000))
            .unwrap();
        db.set_warm("exec-warm", &crate::clock::iso_in_minutes(30))
            .unwrap();
        db.create_pod_record("pod-warm", "H100", "SECURE", None, &image)
            .unwrap();
        let cloud = FakePods::default();
        if listed {
            cloud.seed("pod-warm", "poot-orch-x-exec-warm-a1", true);
        }
        (db, cloud, image)
    }

    /// `exec` with a warm probe that answers per `answers` (true = ssh answered), counting the attempts.
    fn exec_with_probe(db: &Db, cloud: &FakePods, answers: &[bool]) -> (ExecPod, usize) {
        let attempts = std::cell::Cell::new(0usize);
        let reachable = |_: &str, _: u16| {
            let n = attempts.get();
            attempts.set(n + 1);
            answers.get(n).copied().unwrap_or(false)
        };
        let pod = acquire_exec_pod(
            db,
            cloud,
            &MockRegistry::resolving_to(DIGEST),
            &exec_args_with(&["--cmd", "true"]),
            "exec-1",
            &WarmProbe {
                reachable: &reachable,
                pause: Duration::ZERO,
            },
            &HashMap::new(),
            Duration::ZERO,
        )
        .unwrap();
        (pod, attempts.get())
    }

    /// POOT-1350: another job's warm pod that looks unreachable (RunPod lists it, ssh never answers) is neither
    /// adopted nor deleted: the exec creates its own pod and the warm pod stays warm for its owner's window.
    /// Mutation: delete the pod in the `Unresponsive` arm of `acquire_exec_pod` and `deleted` is not empty.
    #[test]
    fn an_unreachable_looking_warm_pod_of_another_job_is_left_alone() {
        let (db, cloud, _) = warm_pod_fixture(true);

        let (pod, attempts) = exec_with_probe(&db, &cloud, &[]);

        assert_ne!(pod.pod_id, "pod-warm");
        assert_eq!(cloud.created.get(), 1, "its own pod");
        assert_eq!(attempts, 4, "four probes before it counts as not answering");
        assert!(cloud.deleted.borrow().is_empty(), "{:?}", cloud.deleted);
        assert!(
            !db.find_warm_execs(&format!("ghcr.io/kikijiki/poot-bench@{DIGEST}"))
                .unwrap()
                .is_empty(),
            "still kept warm for its owner"
        );
    }

    /// POOT-1350: one failed ssh probe is a busy host, not a dead pod: the next answer adopts it.
    /// Mutation: `WARM_PROBES = 1` -> a fresh pod is created and the warm one is not adopted.
    #[test]
    fn one_failed_ssh_probe_does_not_make_a_warm_pod_unreachable() {
        let (db, cloud, _) = warm_pod_fixture(true);

        let (pod, attempts) = exec_with_probe(&db, &cloud, &[false, true]);

        assert_eq!(pod.pod_id, "pod-warm");
        assert_eq!(attempts, 2);
        assert_eq!(cloud.created.get(), 0);
        assert!(cloud.deleted.borrow().is_empty());
    }

    /// POOT-1350 (the c1337 case): the kept pod is already gone from RunPod. The exec closes the record, issues no
    /// delete for a pod it does not need to touch, and provisions its own. Mutation: delete in the `Gone` arm and
    /// `deleted` lists `pod-warm`.
    #[test]
    fn a_warm_pod_runpod_no_longer_lists_is_closed_without_a_delete() {
        let (db, cloud, image) = warm_pod_fixture(false);

        let (pod, attempts) = exec_with_probe(&db, &cloud, &[]);

        assert_ne!(pod.pod_id, "pod-warm");
        assert_eq!(attempts, 0, "no ssh to a pod RunPod says is gone");
        assert!(cloud.deleted.borrow().is_empty(), "{:?}", cloud.deleted);
        assert!(
            db.find_warm_execs(&image).unwrap().is_empty(),
            "no longer adoptable"
        );
        assert_eq!(
            db.pod_record("pod-warm").unwrap().unwrap().status,
            "terminated"
        );
    }

    /// Another kept-warm exec `run` for the fixture's image, holding `pod` at `host`, kept warm for 30 minutes.
    /// `updated` orders it: the larger sorts newer.
    fn add_warm_pod(db: &Db, cloud: &FakePods, run: &str, pod: &str, host: &str, listed: bool) {
        let image = format!("ghcr.io/kikijiki/poot-bench@{DIGEST}");
        let mut warm = exec_run(run);
        warm.image = image.clone();
        warm.status = "running".into();
        db.create_run_with_owner(&warm, Some(&ExecOwner::current().unwrap()))
            .unwrap();
        db.set_pod(run, pod, Some(host), Some(40000)).unwrap();
        db.set_warm(run, &crate::clock::iso_in_minutes(30)).unwrap();
        db.create_pod_record(pod, "H100", "SECURE", None, &image)
            .unwrap();
        if listed {
            cloud.seed(pod, &format!("poot-orch-x-{run}-a1"), true);
        }
    }

    /// `exec` whose warm probe answers only for the hosts in `reachable`, counting the attempts per host.
    fn exec_probing_hosts(
        db: &Db,
        cloud: &FakePods,
        exec_id: &str,
        reachable: &[&str],
    ) -> (ExecPod, Vec<String>) {
        let probed = RefCell::new(Vec::new());
        let probe = |host: &str, _: u16| {
            probed.borrow_mut().push(host.to_string());
            reachable.contains(&host)
        };
        let pod = acquire_exec_pod(
            db,
            cloud,
            &MockRegistry::resolving_to(DIGEST),
            &exec_args_with(&["--cmd", "true"]),
            exec_id,
            &WarmProbe {
                reachable: &probe,
                pause: Duration::ZERO,
            },
            &HashMap::new(),
            Duration::ZERO,
        )
        .unwrap();
        (pod, probed.into_inner())
    }

    /// Two concurrent execs, one pod each. Exec 1 is RUNNING on a pod RunPod lists whose ssh looks dead; exec 2
    /// provisions its own pod without probing, adopting or deleting exec 1's, and a `reap` keeps both. Mutation:
    /// let `find_warm_execs` return running rows and exec 2 reaches the panicking probe.
    #[test]
    fn a_second_exec_never_touches_the_first_execs_running_pod() {
        let image = format!("ghcr.io/kikijiki/poot-bench@{DIGEST}");
        let db = Db::open(":memory:").unwrap();
        let cloud = FakePods::default();
        let mut first = exec_run("exec-first");
        first.image = image.clone();
        first.status = "running".into();
        db.create_run_with_owner(&first, Some(&ExecOwner::current().unwrap()))
            .unwrap();
        db.set_pod("exec-first", "pod-first", Some("10.0.0.1"), Some(40000))
            .unwrap();
        db.create_pod_record("pod-first", "H100", "SECURE", None, &image)
            .unwrap();
        db.assign_pod("pod-first", "exec-first").unwrap();
        cloud.seed(
            "pod-first",
            &db.pod_owner_prefix().unwrap().pod_name("exec-first", 1),
            true,
        );

        let second = acquire_exec_pod(
            &db,
            &cloud,
            &MockRegistry::resolving_to(DIGEST),
            &exec_args_with(&["--cmd", "true"]),
            "exec-2",
            &no_warm_probe(),
            &HashMap::new(),
            Duration::ZERO,
        )
        .unwrap();
        let reaped = reap(&db, &cloud, false).unwrap();

        assert_ne!(second.pod_id, "pod-first");
        assert_eq!(cloud.created.get(), 1, "exec 2 has its own pod");
        assert_eq!(reaped, 0, "reap keeps both execs' pods");
        assert!(cloud.deleted.borrow().is_empty(), "{:?}", cloud.deleted);
        let row = db.pod_record("pod-first").unwrap().unwrap();
        assert_eq!(
            (row.status.as_str(), row.current_run.as_deref()),
            ("busy", Some("exec-first"))
        );
        assert_eq!(
            db.run_status("exec-first").unwrap().as_deref(),
            Some("running")
        );
    }

    /// One stuck newest warm pod does not hide an older reachable one: the newest is probed and left alone, the
    /// older is adopted. Mutation: take only the first candidate and a fresh pod is created.
    #[test]
    fn a_stuck_newest_warm_pod_does_not_hide_an_older_reachable_one() {
        let (db, cloud, _) = warm_pod_fixture(true);
        add_warm_pod(&db, &cloud, "exec-newer", "pod-stuck", "9.9.9.9", true);
        // `updated_at` has second resolution: make the stuck pod unambiguously the newest.
        db.set_warm("exec-newer", &crate::clock::iso_in_minutes(31))
            .unwrap();

        let (pod, probed) = exec_probing_hosts(&db, &cloud, "exec-1", &["1.2.3.4"]);

        assert_eq!(pod.pod_id, "pod-warm", "the older one is adopted");
        assert_eq!(cloud.created.get(), 0);
        assert!(cloud.deleted.borrow().is_empty());
        assert_eq!(probed.iter().filter(|h| *h == "9.9.9.9").count(), 4);
    }

    /// An unresponsive warm pod is recorded as missed: the next exec does not spend another probe round on it.
    /// Mutation: skip the `missed_recently` check and the second exec probes it four more times.
    #[test]
    fn an_unresponsive_warm_pod_is_not_probed_again_within_the_miss_window() {
        let (db, cloud, _) = warm_pod_fixture(true);

        let (_, first) = exec_probing_hosts(&db, &cloud, "exec-1", &[]);
        let (_, second) = exec_probing_hosts(&db, &cloud, "exec-2", &[]);

        assert_eq!(first.len(), 4);
        assert!(second.is_empty(), "no probes the second time: {second:?}");
        assert_eq!(
            db.pod_record("pod-warm").unwrap().unwrap().status,
            "unresponsive"
        );
        assert!(cloud.deleted.borrow().is_empty());
    }

    /// A pod RunPod lists as running but without an ssh endpoint cannot be adopted; it is closed with its own
    /// reason (it still exists), not as "no longer exists", and not deleted.
    #[test]
    fn a_warm_pod_without_an_ssh_endpoint_is_distinguished_from_a_gone_one() {
        let (db, cloud, _) = warm_pod_fixture(false);
        cloud.seed("pod-warm", "poot-orch-x-exec-warm-a1", false);

        let (pod, probed) = exec_probing_hosts(&db, &cloud, "exec-1", &[]);

        assert_ne!(pod.pod_id, "pod-warm");
        assert!(probed.is_empty(), "no ssh to a pod with no endpoint");
        assert!(cloud.deleted.borrow().is_empty());
        assert_eq!(
            db.pod_record("pod-warm").unwrap().unwrap().status,
            "terminated"
        );
        assert_eq!(
            judge_warm_pod(
                &cloud,
                &WarmExec {
                    run_id: "exec-warm".into(),
                    image: String::new(),
                    pod_id: "pod-warm".into(),
                    host: "1.2.3.4".into(),
                    port: 40000,
                    warm_until: String::new(),
                },
                &no_warm_probe()
            ),
            WarmVerdict::NoEndpoint
        );
    }

    /// `--max-price-per-hour` takes a positive finite number and nothing else.
    #[test]
    fn max_price_flag_parses_positive_numbers_only() {
        let parse = |v: &str| {
            Cli::try_parse_from([
                "poot-orchestrator",
                "exec",
                "--cmd",
                "true",
                "--max-price-per-hour",
                v,
            ])
        };
        let Command::Exec(args) = parse("0.5").unwrap().command else {
            panic!("not exec");
        };
        assert_eq!(args.max_price_per_hour, Some(0.5));
        for bad in ["0", "-1", "abc", "inf", "NaN"] {
            assert!(parse(bad).is_err(), "{bad} must be refused");
        }
    }

    /// A single-cloud choice does not fall back: a refused SECURE stays refused.
    #[test]
    fn cloud_secure_does_not_fall_back_to_community() {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cloud = FakePods::default();
        cloud.refuse_secure.set(true);

        let cfg = ProvisionCfg {
            give_up_after: Some(Duration::ZERO),
            ..provision_cfg(&owner, "run-1")
        };
        let err = provision_pod(
            &cloud,
            &cfg,
            &HashMap::new(),
            |_, _, _| Ok(()),
            |_| {},
            &FakeClock::default(),
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("no SECURE capacity"), "{err:#}");
        assert_eq!(cloud.bodies.borrow().len(), 1);
    }

    /// `--cloud` accepts the three choices in any case and rejects anything else at parse time.
    #[test]
    fn cloud_flag_parses_the_three_choices_and_nothing_else() {
        let parse = |v: &str| {
            Cli::try_parse_from(["poot-orchestrator", "exec", "--cmd", "true", "--cloud", v])
        };
        for (arg, want) in [
            ("ALL", CloudChoice::All),
            ("all", CloudChoice::All),
            ("SECURE", CloudChoice::Secure),
            ("community", CloudChoice::Community),
        ] {
            let Command::Exec(a) = parse(arg).unwrap().command else {
                panic!("exec");
            };
            assert_eq!(a.cloud, want, "{arg}");
        }
        assert!(parse("ANY").is_err());
    }

    /// A registry that will not resolve the tag stops `exec` before the run row or any pod exists.
    #[test]
    fn exec_creates_nothing_when_its_image_cannot_be_pinned() {
        let db = Db::open(":memory:").unwrap();
        let cloud = FakePods::default();

        let err = acquire_exec_pod(
            &db,
            &cloud,
            &MockRegistry::refusing(),
            &exec_args("ghcr.io/kikijiki/poot-bench:latest"),
            "exec-1",
            &no_warm_probe(),
            &HashMap::new(),
            Duration::ZERO,
        )
        .map(|_| ())
        .unwrap_err();

        assert!(format!("{err:#}").contains("denied"), "{err:#}");
        assert_eq!(cloud.created.get(), 0, "no pod was created");
        assert!(db.recent_runs(10).unwrap().is_empty(), "no run row");
    }

    /// The sweep arms the same pod-side TTL, sized from its own budget: every model's repeats at the full
    /// sweep timeout plus the margin. A sweep that skips `ready_pod` leaks its pod when killed.
    #[test]
    fn the_sweep_pod_ttl_covers_its_budget_and_is_armed_first() {
        let Command::Run(a) = Cli::try_parse_from([
            "poot-orchestrator",
            "run",
            "--sweep-timeout-s",
            "600",
            "--repeats",
            "2",
        ])
        .unwrap()
        .command
        else {
            panic!("`run` parses to Command::Run");
        };
        let ttl = sweep_pod_ttl(&a, 3);
        assert_eq!(ttl, Duration::from_secs(600 * 2 * 3) + POD_TTL_MARGIN);

        let pod = LocalPod::new(0);
        ready_pod(&pod, Duration::from_secs(1), ttl).unwrap();
        assert_eq!(pod.commands(), [pod_ttl_command(ttl)]);
    }

    /// The signal handler and the main thread can both delete the pod; the second delete of a pod
    /// RunPod already removed is tolerated and the run still ends terminal.
    #[test]
    fn a_second_delete_after_signal_teardown_is_tolerated() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&exec_run("exec-1")).unwrap();
        db.set_pod("exec-1", "pod-1", None, None).unwrap();
        let pods = SharedPods::default();
        exec_signal_teardown(&db, &pods, "exec-1", false).unwrap();
        finish_exec_pod(&db, &pods, "exec-1", "pod-1", "failed").unwrap();
        assert_eq!(*pods.deleted.lock().unwrap(), ["pod-1", "pod-1"]);
        assert_eq!(db.run_status("exec-1").unwrap().as_deref(), Some("failed"));
    }

    /// A scratch directory and what is in it are gone when it is dropped.
    #[test]
    fn scratch_is_removed_on_drop() {
        let scratch = Scratch::new("drop").unwrap();
        let dir = scratch.path().to_path_buf();
        std::fs::write(scratch.known_hosts(), "x").unwrap();
        assert!(dir.join("known_hosts").exists());
        drop(scratch);
        assert!(!dir.exists());
    }

    /// A warm probe for tests that never reach a warm pod's ssh.
    fn no_warm_probe() -> WarmProbe<'static> {
        fn never_reached(_: &str, _: u16) -> bool {
            panic!("this test never probes a warm pod over ssh")
        }
        WarmProbe {
            reachable: &never_reached,
            pause: Duration::ZERO,
        }
    }

    fn exec_args_with(extra: &[&str]) -> ExecArgs {
        let argv = ["poot-orchestrator", "exec"]
            .into_iter()
            .chain(extra.iter().copied());
        let Command::Exec(args) = Cli::try_parse_from(argv).unwrap().command else {
            panic!("`exec` parses to Command::Exec");
        };
        args
    }

    /// A pod that is a local directory: remote commands run in a local `sh` with `/root/` rewritten to
    /// the sandbox, through the production deadline path ([`ssh::output_until`]). Its `bin/` comes first
    /// on `PATH`. Arming the pod-side TTL is recorded but not run: it would call RunPod.
    struct LocalPod {
        dir: PathBuf,
        commands: std::sync::Mutex<Vec<String>>,
    }

    impl LocalPod {
        fn new(patchelf_exit: i32) -> LocalPod {
            use std::os::unix::fs::PermissionsExt;
            let dir = unique_temp_dir("pod");
            std::fs::create_dir(dir.join("bin")).unwrap();
            let patchelf = dir.join("bin/patchelf");
            std::fs::write(
                &patchelf,
                format!("#!/bin/sh\necho 'patchelf: fake rc {patchelf_exit}' >&2\nexit {patchelf_exit}\n"),
            )
            .unwrap();
            std::fs::set_permissions(&patchelf, std::fs::Permissions::from_mode(0o755)).unwrap();
            LocalPod {
                dir,
                commands: Default::default(),
            }
        }

        fn sandboxed(&self, text: &str) -> String {
            text.replace("/root/", &format!("{}/", self.dir.display()))
        }

        fn commands(&self) -> Vec<String> {
            self.commands.lock().unwrap().clone()
        }
    }

    impl Remote for LocalPod {
        fn wait_ssh(&self, _timeout: Duration) -> Result<()> {
            Ok(())
        }

        fn scp_up(&self, local: &str, remote: &str) -> Result<()> {
            self.commands
                .lock()
                .unwrap()
                .push(format!("scp {local} {remote}"));
            std::fs::copy(local, self.sandboxed(remote))?;
            Ok(())
        }

        fn ssh_until(&self, cmd: &str, deadline: Instant) -> Result<ssh::Run, ssh::DeadlineError> {
            self.commands.lock().unwrap().push(cmd.to_string());
            if cmd.contains(POD_TTL_PID_FILE) {
                return Ok(ssh::Run {
                    code: 0,
                    stdout: String::new(),
                    stderr: String::new(),
                });
            }
            let path = format!(
                "{}:{}",
                self.dir.join("bin").display(),
                std::env::var("PATH").unwrap_or_default()
            );
            let mut c = std::process::Command::new("sh");
            c.arg("-c").arg(self.sandboxed(cmd)).env("PATH", path);
            let out = ssh::output_until(&mut c, deadline)?;
            Ok(ssh::Run {
                code: out.status.code().unwrap_or(-1),
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
            })
        }
    }

    /// A tracked exec run on `pod-1`, in a fresh in-memory state DB and a cloud that holds the pod.
    fn exec_on_fake_pod(a: &ExecArgs, remote: &LocalPod) -> (Result<()>, Db, FakePods) {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&exec_run("exec-1")).unwrap();
        db.set_pod("exec-1", "pod-1", None, None).unwrap();
        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-exec-1-a1", true);
        let logpath = unique_temp_dir("log").join("exec-1.log");
        let result = exec_on_pod(&db, &cloud, remote, a, "exec-1", &logpath, "pod-1");
        (result, db, cloud)
    }

    /// SC-002: a remote command past `--max-cmd` is killed at the deadline with the typed timeout, and
    /// the failed exec still tears the pod down. Giving the command a longer deadline than `--max-cmd`
    /// lets `sleep 20` finish and this goes red.
    #[test]
    fn exec_kills_a_command_that_outlives_max_cmd() {
        let pod = LocalPod::new(0);
        let a = exec_args_with(&["--cmd", "sleep 20", "--max-cmd", "2s"]);
        let started = Instant::now();
        let (result, db, cloud) = exec_on_fake_pod(&a, &pod);
        let elapsed = started.elapsed();

        let err = result.unwrap_err();
        assert_eq!(
            err.downcast_ref::<ssh::DeadlineError>(),
            Some(&ssh::DeadlineError::Timeout),
            "{err:#}"
        );
        assert!(
            elapsed >= Duration::from_secs(2) && elapsed < Duration::from_secs(10),
            "killed at --max-cmd, elapsed {elapsed:?}"
        );
        assert!(cloud.ids().is_empty(), "the pod was torn down");
        assert_eq!(db.run_status("exec-1").unwrap().as_deref(), Some("failed"));
    }

    /// The pod-side TTL is the first remote command, and it is the default window: `--max-cmd` plus
    /// `--keep-warm` plus the margin. Not arming it leaves a dead orchestrator's pod billing.
    #[test]
    fn exec_arms_the_pod_ttl_before_anything_else() {
        let pod = LocalPod::new(0);
        let a = exec_args_with(&["--cmd", "true", "--max-cmd", "10m", "--keep-warm", "20"]);
        let (result, _db, _cloud) = exec_on_fake_pod(&a, &pod);
        result.unwrap();
        let commands = pod.commands();
        let expected = pod_ttl_command(Duration::from_secs((10 + 20 + 30) * 60));
        assert_eq!(commands.first(), Some(&expected), "{commands:?}");
        assert!(expected.contains("sleep 3600;"), "{expected}");
        assert_eq!(
            pod_ttl(&exec_args_with(&["--cmd", "x", "--pod-ttl", "5"])).as_secs(),
            300
        );
    }

    /// SC-005: a patch step that fails fails the exec, and the command never runs. Chaining the steps
    /// with `;` lets the trailing `chmod` succeed and this goes red. The control shows the same exec
    /// running its command once the patch succeeds.
    #[test]
    fn a_failing_remote_patch_step_fails_the_exec() {
        let bin = unique_temp_dir("bin").join("runner");
        std::fs::write(&bin, "not an elf").unwrap();
        let bin = bin.to_str().unwrap();

        let broken = LocalPod::new(3);
        let a = exec_args_with(&["--cmd", "echo ran", "--bin", bin]);
        let (result, _db, _cloud) = exec_on_fake_pod(&a, &broken);
        let err = format!("{:#}", result.unwrap_err());
        assert!(err.contains("binary patch failed (rc=3)"), "{err}");
        assert!(
            !broken.commands().iter().any(|c| c == "echo ran"),
            "the command must not run on an unpatched binary: {:?}",
            broken.commands()
        );

        let sound = LocalPod::new(0);
        let (result, _db, _cloud) = exec_on_fake_pod(&a, &sound);
        result.unwrap();
        assert!(sound.commands().iter().any(|c| c == "echo ran"));
    }

    /// A pod and a DB shared with the signal handler thread.
    #[derive(Clone, Default)]
    struct SharedPods {
        deleted: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl PodApi for SharedPods {
        fn create_pod(&self, _spec: &CreateSpec<'_>) -> Result<Pod> {
            unimplemented!("the pod already exists")
        }
        fn get_pod(&self, _id: &str) -> Result<Pod> {
            unimplemented!("not used")
        }
        fn list_pods(&self) -> Result<Vec<Pod>> {
            unimplemented!("not used")
        }
        fn gpu_catalog(&self) -> Result<GpuCatalog> {
            unimplemented!("not used")
        }
        fn delete_pod(&self, id: &str) -> Result<()> {
            self.deleted.lock().unwrap().push(id.to_string());
            Ok(())
        }
    }

    /// SC-001: a SIGTERM delivered while `exec` is mid-command runs the teardown once: the pod is
    /// deleted and the run row is terminal. Without the installed handler the signal kills this process.
    #[test]
    fn sigterm_during_an_exec_tears_the_pod_down_once() {
        let state_db = unique_temp_dir("sig").join("state.db");
        let state_db = state_db.to_str().unwrap().to_string();
        {
            let db = Db::open(&state_db).unwrap();
            db.create_run(&exec_run("exec-1")).unwrap();
            db.set_pod("exec-1", "pod-1", None, None).unwrap();
        }
        let pods = SharedPods::default();
        let pod = std::sync::Arc::new(LocalPod::new(0));

        let (done_tx, done_rx) = std::sync::mpsc::channel();
        install_signal_teardown(
            {
                let (state_db, pods) = (state_db.clone(), pods.clone());
                move || {
                    let db = Db::open(&state_db).unwrap();
                    exec_signal_teardown(&db, &pods, "exec-1", false).unwrap();
                }
            },
            move || done_tx.send(()).unwrap(),
        )
        .unwrap();

        let exec = {
            let (state_db, pods, pod) = (state_db.clone(), pods.clone(), pod.clone());
            std::thread::spawn(move || {
                let db = Db::open(&state_db).unwrap();
                let a = exec_args_with(&["--cmd", "sleep 20", "--max-cmd", "60s"]);
                let logpath = unique_temp_dir("log").join("exec-1.log");
                let _ = exec_on_pod(&db, &pods, &*pod, &a, "exec-1", &logpath, "pod-1");
            })
        };
        let started = Instant::now();
        while !pod.commands().iter().any(|c| c == "sleep 20") {
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "exec never reached its command"
            );
            std::thread::sleep(Duration::from_millis(20));
        }

        let status = std::process::Command::new("kill")
            .args(["-TERM", &std::process::id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        done_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("the SIGTERM handler ran its teardown");

        assert_eq!(
            *pods.deleted.lock().unwrap(),
            ["pod-1"],
            "terminated exactly once"
        );
        let db = Db::open(&state_db).unwrap();
        assert_eq!(db.run_status("exec-1").unwrap().as_deref(), Some("aborted"));
        drop(exec); // still inside `sleep 20`; the process exit ends it
    }

    /// `--keep` leaves the pod to the next `reap` but still marks the signalled run terminal.
    #[test]
    fn signal_teardown_with_keep_leaves_the_pod_and_ends_the_run() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&exec_run("exec-1")).unwrap();
        db.set_pod("exec-1", "pod-1", None, None).unwrap();
        let pods = SharedPods::default();
        exec_signal_teardown(&db, &pods, "exec-1", true).unwrap();
        assert!(pods.deleted.lock().unwrap().is_empty());
        assert_eq!(db.run_status("exec-1").unwrap().as_deref(), Some("aborted"));
    }

    /// A stand-in `cargo` that insists on `--release --message-format=json` and prints `messages`.
    fn fake_cargo(dir: &Path, messages: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("fake-cargo");
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\n\
                 case \"$*\" in *--release*--message-format=json*) ;; *) echo \"bad args: $*\" >&2; exit 2;; esac\n\
                 cat <<'JSON'\n{messages}\nJSON\n"
            ),
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        script
    }

    fn artifact_message(manifest: &Path, executable: &Path) -> String {
        serde_json::json!({
            "reason": "compiler-artifact",
            "manifest_path": manifest,
            "target": {"name": RUNNER_BIN, "kind": ["bin"]},
            "executable": executable,
            "fresh": false,
        })
        .to_string()
    }

    /// SC-003: a stale runner sits at `runner_dir/target/release/poot-bench-runner` while cargo reports
    /// the binary it built elsewhere (an exported CARGO_TARGET_DIR). The build returns cargo's path.
    /// The fixed-path lookup returns the stale binary.
    #[test]
    fn runner_binary_is_the_artifact_cargo_reports_not_the_stale_target_dir_one() {
        let dir = unique_temp_dir("runner-artifact");
        let runner_dir = dir.join("runner");
        let stale = runner_dir.join("target/release").join(RUNNER_BIN);
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::write(runner_dir.join("Cargo.toml"), "").unwrap();
        std::fs::write(&stale, "stale").unwrap();
        let built = dir.join("shared-target").join(RUNNER_BIN);
        std::fs::create_dir_all(built.parent().unwrap()).unwrap();
        std::fs::write(&built, "fresh").unwrap();
        let cargo = fake_cargo(
            &dir,
            &artifact_message(&runner_dir.join("Cargo.toml"), &built),
        );

        let uploaded = build_runner_artifact(cargo.to_str().unwrap(), &runner_dir).unwrap();
        assert_eq!(uploaded, built);
        assert_eq!(std::fs::read_to_string(&uploaded).unwrap(), "fresh");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// SC-003, refusal: cargo reports no runner executable (or one for a different package) while a
    /// stale binary sits at the old fixed path. The build refuses; nothing is uploaded.
    #[test]
    fn runner_build_refuses_when_cargo_reports_no_artifact_for_this_package() {
        let dir = unique_temp_dir("runner-refuse");
        let runner_dir = dir.join("runner");
        let stale = runner_dir.join("target/release").join(RUNNER_BIN);
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::write(runner_dir.join("Cargo.toml"), "").unwrap();
        std::fs::write(&stale, "stale").unwrap();
        let other_package = dir.join("other");
        std::fs::create_dir_all(&other_package).unwrap();
        std::fs::write(other_package.join("Cargo.toml"), "").unwrap();
        let other_bin = other_package.join(RUNNER_BIN);
        std::fs::write(&other_bin, "someone else's").unwrap();

        for (case, messages) in [
            (
                "no artifact",
                r#"{"reason":"build-finished","success":true}"#.to_string(),
            ),
            (
                "another package's runner",
                artifact_message(&other_package.join("Cargo.toml"), &other_bin),
            ),
        ] {
            let cargo = fake_cargo(&dir, &messages);
            let err = build_runner_artifact(cargo.to_str().unwrap(), &runner_dir).unwrap_err();
            assert!(
                format!("{err:#}").contains("refusing to upload"),
                "{case}: {err:#}"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// SC-001: the sweep command names the model, scenario and repeats, and exports neither provenance variable:
    /// the row's `build_sha` (embedded in the runner at build time) is the one provenance.
    #[test]
    fn the_sweep_command_passes_the_cell_and_repeats_and_exports_no_provenance() {
        let cmd = sweep_command(
            "/root/models",
            "qwen3-0.6b",
            "run-1-qwen3-0.6b",
            "decode-curve",
            5,
            &[],
        );

        assert!(cmd.contains("MODELS_DIR=/root/models "), "{cmd}");
        assert!(cmd.contains("--model qwen3-0.6b "), "{cmd}");
        assert!(cmd.contains("--scenario decode-curve "), "{cmd}");
        assert!(cmd.contains("--run-id run-1-qwen3-0.6b "), "{cmd}");
        assert!(cmd.contains("--repeats 5 "), "{cmd}");
        for provenance in ["POOT_GIT_SHA", "POOT_TARGET"] {
            assert!(!cmd.contains(provenance), "{provenance} in {cmd}");
        }
    }

    /// A new run records the repeats it was asked for.
    #[test]
    fn a_new_run_records_the_repeats_it_was_asked_for() {
        let db = Db::open(":memory:").unwrap();
        let Command::Run(args) =
            Cli::try_parse_from(["poot-orchestrator", "run", "--repeats", "5"])
                .unwrap()
                .command
        else {
            panic!("`run` parses to Command::Run");
        };

        let (run, _) = resolve_run(&db, &FakePods::default(), &args, &[], "abc").unwrap();

        assert_eq!(run.repeats, 5);
    }

    #[test]
    fn repeats_default_to_one_and_must_be_positive() {
        let Command::Run(args) = Cli::try_parse_from(["poot-orchestrator", "run"])
            .unwrap()
            .command
        else {
            panic!("`run` parses to Command::Run");
        };
        assert_eq!(args.repeats, 1);
        let Command::Run(args) =
            Cli::try_parse_from(["poot-orchestrator", "run", "--repeats", "5"])
                .unwrap()
                .command
        else {
            panic!("`run` parses to Command::Run");
        };
        assert_eq!(args.repeats, 5);
        assert!(Cli::try_parse_from(["poot-orchestrator", "run", "--repeats", "0"]).is_err());
    }
    const GGUF_MANIFEST: &str = r#"
[[model]]
id = "big-gguf"
hf_repo = "org/big-GGUF"
local_dir = "big-gguf"
hf_include = ["Q4/*.gguf", "tok.json"]
weights_file = "Q4/big-00001-of-00004.gguf"
frameworks.poot = { precision = "q4_k" }
frameworks.llamacpp = { support = true, precision = "q4_k_xl" }
[[model]]
id = "fp8"
hf_repo = "org/fp8"
frameworks.poot = { precision = "e4m3block128" }
frameworks.llamacpp = { support = false, reason = "no GGUF" }
[[model]]
id = "dense"
hf_repo = "org/dense"
frameworks.poot = { precision = "bf16" }
frameworks.llamacpp = { support = true, precision = "f16" }
"#;

    fn manifest_model(id: &str) -> Model {
        parse_manifest_models(GGUF_MANIFEST)
            .unwrap()
            .into_iter()
            .find(|m| m.id == id)
            .unwrap()
    }

    /// A model that names `hf_include` downloads those files only; one that names none downloads the repo.
    #[test]
    fn the_download_fetches_only_the_included_files() {
        let big = download_command(
            &manifest_model("big-gguf"),
            "org/big-GGUF",
            "/root/models/big-gguf",
        );
        assert!(
            big.contains("hf download org/big-GGUF --local-dir /root/models/big-gguf --include 'Q4/*.gguf' --include 'tok.json' >"),
            "{big}"
        );
        let dense = download_command(&manifest_model("dense"), "org/dense", "/root/models/dense");
        assert!(!dense.contains("--include"), "{dense}");
    }

    /// The volume cache probe looks for the file the runners are handed, not a `config.json` a GGUF repo lacks.
    #[test]
    fn the_cache_probe_looks_for_the_weights_file() {
        assert_eq!(
            cached_probe(&manifest_model("big-gguf"), "/workspace/models/big-gguf"),
            "test -f /workspace/models/big-gguf/Q4/big-00001-of-00004.gguf"
        );
        assert_eq!(
            cached_probe(&manifest_model("dense"), "/workspace/models/dense"),
            "test -f /workspace/models/dense/config.json"
        );
    }

    /// The f16 pre-conversion runs for a safetensors model llama.cpp runs, and for no other: not for a GGUF
    /// download, not for a model the manifest marks llama.cpp-unsupported.
    #[test]
    fn the_f16_gguf_is_made_only_for_a_llamacpp_model_that_downloads_as_safetensors() {
        let dense = preconvert_command(&manifest_model("dense"), "/root/models/dense").unwrap();
        assert!(dense.contains("convert_hf_to_gguf.py /root/models/dense --outfile /root/models/dense/dense-f16.gguf --outtype f16"), "{dense}");
        assert_eq!(preconvert_command(&manifest_model("big-gguf"), "/d"), None);
        assert_eq!(preconvert_command(&manifest_model("fp8"), "/d"), None);
    }

    /// There is no CUDA floor unless one is given: the flag has no default, and it takes a version.
    #[test]
    fn a_run_has_no_cuda_floor_unless_told() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["poot-orchestrator", "run"];
            argv.extend_from_slice(extra);
            let Command::Run(args) = Cli::try_parse_from(argv).unwrap().command else {
                panic!("`run` parses to Command::Run");
            };
            args.min_cuda_version
        };
        assert_eq!(parse(&[]), None);
        assert_eq!(
            parse(&["--min-cuda-version", "13.0"]),
            Some("13.0".parse().unwrap())
        );
        assert!(
            Cli::try_parse_from(["poot-orchestrator", "run", "--min-cuda-version", "thirteen"])
                .is_err()
        );
    }

    fn request(id: &str, status: &str, repeats: u32) -> Run {
        Run {
            models: vec!["qwen3-0.6b".into()],
            repeats,
            ..sweep_run(id, status, None)
        }
    }

    /// A second `run` of the request a failed run made continues its result set: it is named by the failed run,
    /// so the repeats the failed run gathered and the ones this run adds are one `<set>-<model>-rN` set.
    /// Mutation: ignore the status filter in `resumable_set_id` and the "done run" case goes red.
    #[test]
    fn a_rerun_of_a_failed_request_continues_its_result_set() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&request("run-1000", "failed", 5)).unwrap();
        let again = request("run-9000", "new", 5);
        db.create_run(&again).unwrap();

        let set = db.resumable_set_id(&again, 9_000, 24 * 3600).unwrap();

        assert_eq!(set.as_deref(), Some("run-1000"));
        db.set_set_id(&again.id, "run-1000").unwrap();
        assert_eq!(db.set_id("run-9000").unwrap(), "run-1000");
        assert_eq!(db.set_id("run-1000").unwrap(), "run-1000");
    }

    /// Only the same request resumes: another repeat count, an older failure, or a finished run does not.
    #[test]
    fn a_different_or_stale_or_finished_request_is_not_resumed() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&request("run-99500", "failed", 3)).unwrap();
        db.create_run(&request("run-99600", "done", 5)).unwrap();
        db.create_run(&request("run-100", "failed", 5)).unwrap();
        let again = request("run-100000", "new", 5);
        db.create_run(&again).unwrap();

        // run-99500 asked for 3 repeats, run-99600 finished, run-100 is older than the window.
        assert_eq!(db.resumable_set_id(&again, 100_000, 3600).unwrap(), None);
    }

    /// The body of a gathered set's file: its `env.json` names the GPU, which is part of the committed name.
    fn local_body(file: &str) -> &'static str {
        if file == "env.json" {
            r#"{"gpu_name":"NVIDIA GeForce RTX 3090"}"#
        } else {
            "x"
        }
    }

    /// Five repeats gathered under their committed names, `--extend-set` and `--repeats 10`: the run takes the
    /// first run's set id, finds repeats 1-5 local, and its sweep command runs only 6-10. Mutation: ignore
    /// `a.extend_set` in `resolve_run` and the run gets a set id of its own (red on the first assertion).
    #[test]
    fn extending_the_sets_of_a_finished_run_runs_only_the_new_repeats() {
        let tmp = std::env::temp_dir().join(format!("poot-extend-test-{}", std::process::id()));
        let results = tmp.join("results");
        let set = "run-1792632197-3202502";
        for n in 1..=5 {
            let dir = results.join(format!("2026-10-22-geforce-rtx-3090-dense-012317-r{n}"));
            std::fs::create_dir_all(&dir).unwrap();
            for file in ["env.json", "results.jsonl", "report.md"] {
                std::fs::write(dir.join(file), local_body(file)).unwrap();
            }
        }
        let models = [manifest_model("dense")];
        assert_eq!(extension_set(&results, set, &models).unwrap(), set);
        let err = extension_set(&results, set, &[manifest_model("fp8")]).unwrap_err();
        assert!(
            format!("{err:#}").contains("no gathered set of fp8"),
            "{err:#}"
        );
        assert!(extension_set(&results, "run-1-1", &models).is_err());

        let db = Db::open(":memory:").unwrap();
        let args = run_args(&["--repeats", "10", "--extend-set", set]);
        let (run, _) = resolve_run(&db, &FakePods::default(), &args, &[], "abc").unwrap();
        assert_eq!(db.set_id(&run.id).unwrap(), set);

        let dirs = run.model_snapshot_dirs(&format!("{set}-dense"));
        let present = gather::repeats_present(&results, &dirs);
        let cmd = sweep_command(
            "/root/models",
            "dense",
            &format!("{set}-dense"),
            "decode-curve",
            10,
            &present,
        );
        assert_eq!(dirs.len(), 10);
        assert_eq!(present, [1, 2, 3, 4, 5]);
        assert!(
            cmd.contains("--run-id run-1792632197-3202502-dense --repeats 10 --skip-repeats 1,2,3,4,5 --skip-unsupported"),
            "{cmd}"
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    /// The repeats already local are not asked of the new pod: the sweep command skips them, and a model whose
    /// repeats are all local runs nothing. Mutation: drop `--skip-repeats` from `sweep_command` and this goes red.
    #[test]
    fn the_resumed_sweep_skips_the_repeats_already_gathered() {
        let tmp = std::env::temp_dir().join(format!("poot-resume-test-{}", std::process::id()));
        let results = tmp.join("results");
        let dirs: Vec<String> = (1..=5)
            .map(|n| format!("run-1000-qwen3-0.6b-r{n}"))
            .collect();
        // Gathered sets are on disk under their committed names, not the pod's.
        let local = |dir: &String| {
            results.join(gather::canonical_dir(dir, Some("NVIDIA GeForce RTX 3090")).unwrap())
        };
        for dir in &dirs[..3] {
            std::fs::create_dir_all(local(dir)).unwrap();
            for file in ["env.json", "results.jsonl", "report.md"] {
                std::fs::write(local(dir).join(file), local_body(file)).unwrap();
            }
        }
        // r4 is on disk but unfinished (no report.md): it is run again.
        std::fs::create_dir_all(local(&dirs[3])).unwrap();
        std::fs::write(local(&dirs[3]).join("results.jsonl"), "x").unwrap();

        let present = gather::repeats_present(&results, &dirs);
        let cmd = sweep_command(
            "/root/models",
            "qwen3-0.6b",
            "run-1000-qwen3-0.6b",
            "decode-curve",
            5,
            &present,
        );

        assert_eq!(present, [1, 2, 3]);
        assert!(
            cmd.contains("--repeats 5 --skip-repeats 1,2,3 --skip-unsupported"),
            "{cmd}"
        );
        std::fs::remove_dir_all(&tmp).unwrap();
    }
    /// Time that passes only when `sleep` is called, and records every sleep. A runaway loop ends in a panic
    /// instead of hanging the suite.
    #[derive(Default)]
    struct FakeClock {
        now: Cell<Duration>,
        sleeps: RefCell<Vec<Duration>>,
    }

    impl Clock for FakeClock {
        fn elapsed(&self) -> Duration {
            self.now.get()
        }
        fn sleep(&self, d: Duration) {
            assert!(self.sleeps.borrow().len() < 500, "runaway retry loop");
            self.sleeps.borrow_mut().push(d);
            self.now.set(self.now.get() + d);
        }
    }

    fn two_gpu_cfg<'a>(owner: &'a PodOwnerPrefix) -> ProvisionCfg<'a> {
        ProvisionCfg {
            gpu_types: "NVIDIA H100 80GB HBM3,NVIDIA H200",
            ..provision_cfg(owner, "run-1")
        }
    }

    /// Capacity refused for four full passes, then granted: exactly one pod is created, and the waits between
    /// passes grow 20, 30, 45, 68 s (each with up to a tenth of jitter on top). Mutations: no sleep (empty
    /// list), a fixed interval (not growing), a hard cap on passes (the fifth pass never comes) each go red.
    #[test]
    fn capacity_refused_for_several_passes_is_waited_out_with_growing_pauses() {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cloud = FakePods::default();
        cloud.refuse_creates.set(8);
        let clock = FakeClock::default();
        let waiting = RefCell::new(Vec::new());

        provision_pod(
            &cloud,
            &two_gpu_cfg(&owner),
            &HashMap::new(),
            |_, _, _| Ok(()),
            |pass| waiting.borrow_mut().push(pass),
            &clock,
        )
        .unwrap();

        assert_eq!(cloud.ids().len(), 1, "exactly one pod");
        let sleeps: Vec<u64> = clock.sleeps.borrow().iter().map(|d| d.as_secs()).collect();
        assert_eq!(sleeps.len(), 4, "{sleeps:?}");
        for (got, base) in sleeps.iter().zip([20u64, 30, 45, 68]) {
            assert!(*got >= base && *got <= base + base / 10 + 1, "{sleeps:?}");
        }
        assert_eq!(*waiting.borrow(), [1, 2, 3, 4]);
    }

    #[test]
    fn the_backoff_grows_by_half_each_pass_and_stops_at_five_minutes() {
        let got: Vec<u64> = (1..=10).map(capacity_backoff_secs).collect();
        assert_eq!(got, [20, 30, 45, 68, 101, 152, 228, 300, 300, 300]);
    }

    /// A refusal waiting cannot fix (a 4xx that is not about capacity) fails at once with its message, after one
    /// try per cloud and no waiting. Mutation: treat every create error as capacity and the clock panics as runaway.
    #[test]
    fn a_non_capacity_refusal_fails_at_once() {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cloud = FakePods::default();
        *cloud.refuse_with.borrow_mut() = Some((400, "gpuTypeIds: unknown GPU type".into()));
        let clock = FakeClock::default();

        let err = provision_pod(
            &cloud,
            &two_gpu_cfg(&owner),
            &HashMap::new(),
            |_, _, _| Ok(()),
            |_| {},
            &clock,
        )
        .unwrap_err();

        assert!(err.downcast_ref::<Unrecoverable>().is_some(), "{err:#}");
        assert!(format!("{err:#}").contains("unknown GPU type"), "{err:#}");
        assert_eq!(cloud.creates.get(), 1);
        assert!(clock.sleeps.borrow().is_empty());
    }

    /// A 400 saying an image or template is "not available" is a permanent refusal, not a capacity one: it fails at
    /// once. Mutation: restore the broad "not available" text match in `CreateError::is_capacity` and the clock
    /// panics as runaway.
    #[test]
    fn a_400_saying_the_image_is_not_available_fails_at_once() {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cloud = FakePods::default();
        *cloud.refuse_with.borrow_mut() =
            Some((400, "image poot-bench:missing is not available".into()));
        let clock = FakeClock::default();

        let err = provision_pod(
            &cloud,
            &provision_cfg(&owner, "run-1"),
            &HashMap::new(),
            |_, _, _| Ok(()),
            |_| {},
            &clock,
        )
        .unwrap_err();

        assert!(err.downcast_ref::<Unrecoverable>().is_some(), "{err:#}");
        assert!(format!("{err:#}").contains("not available"), "{err:#}");
        assert_eq!(cloud.creates.get(), 1);
        assert!(clock.sleeps.borrow().is_empty());
    }

    /// `--cloud all`: one cloud refusing for good while the other only lacks capacity keeps waiting, then rents,
    /// whichever cloud is which. Mutation: let the last cloud's answer decide (`refused_for_good = !is_capacity_error(&e)`)
    /// and the COMMUNITY-refuses case goes red; let any refusal decide and both do.
    #[test]
    fn one_cloud_refusing_for_good_does_not_end_the_wait_while_the_other_lacks_capacity() {
        for refusing in [CloudType::Secure, CloudType::Community] {
            let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
            let cloud = FakePods::default();
            let refusal = Some((400, "unknown GPU type".to_string()));
            match refusing {
                CloudType::Secure => *cloud.refuse_secure_with.borrow_mut() = refusal,
                CloudType::Community => *cloud.refuse_community_with.borrow_mut() = refusal,
            }
            // The other cloud has no capacity once, then accepts.
            cloud.refuse_creates.set(1);
            let cfg = ProvisionCfg {
                cloud: CloudChoice::All,
                ..provision_cfg(&owner, "run-1")
            };
            let clock = FakeClock::default();

            provision_pod(
                &cloud,
                &cfg,
                &HashMap::new(),
                |_, _, _| Ok(()),
                |_| {},
                &clock,
            )
            .unwrap_or_else(|e| panic!("{refusing}: {e:#}"));

            assert_eq!(clock.sleeps.borrow().len(), 1, "{refusing}");
            assert_eq!(cloud.ids().len(), 1, "{refusing}");
        }
    }

    /// `--cloud all` with every cloud refusing for good fails at once as Unrecoverable.
    #[test]
    fn every_cloud_refusing_for_good_is_unrecoverable() {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cloud = FakePods::default();
        *cloud.refuse_with.borrow_mut() = Some((400, "unknown GPU type".into()));
        let cfg = ProvisionCfg {
            cloud: CloudChoice::All,
            ..provision_cfg(&owner, "run-1")
        };

        let err = provision_pod(
            &cloud,
            &cfg,
            &HashMap::new(),
            |_, _, _| Ok(()),
            |_| {},
            &FakeClock::default(),
        )
        .unwrap_err();

        assert!(err.downcast_ref::<Unrecoverable>().is_some(), "{err:#}");
        assert_eq!(cloud.creates.get(), 2);
    }

    /// `give_up_after` bounds the wait: the run is cancelled with a message listing the GPU types it tried.
    #[test]
    fn giving_up_names_the_gpu_types_tried() {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let cloud = FakePods::default();
        cloud.refuse_creates.set(u32::MAX);
        let clock = FakeClock::default();
        let cfg = ProvisionCfg {
            give_up_after: Some(Duration::from_secs(5 * 60)),
            ..two_gpu_cfg(&owner)
        };

        let err = provision_pod(
            &cloud,
            &cfg,
            &HashMap::new(),
            |_, _, _| Ok(()),
            |_| {},
            &clock,
        )
        .unwrap_err();

        let cancelled = err.downcast_ref::<Cancelled>().expect("cancelled");
        assert!(
            cancelled.0.contains("NVIDIA H100 80GB HBM3, NVIDIA H200"),
            "{}",
            cancelled.0
        );
        assert!(clock.now.get() >= Duration::from_secs(5 * 60));
        assert!(cloud.ids().is_empty());
    }

    /// One `run`'s pod vanishes after repeat 2 of 5: a second pod is provisioned and runs repeats 3-5 only, the
    /// run record is one and never 'failed'. Mutation: `repeats_present` returning nothing re-runs 1-5 on pod two.
    #[test]
    fn a_pod_lost_after_repeat_two_is_replaced_and_only_the_rest_runs() {
        struct Sim<'a> {
            db: &'a Db,
            cloud: &'a FakePods,
            run: Run,
            results: PathBuf,
            owner: PodOwnerPrefix,
            pod: Option<String>,
            ran: Vec<Vec<u32>>,
            statuses: Vec<String>,
        }
        impl Healing for Sim<'_> {
            fn attempt(&mut self, fresh: bool) -> Result<()> {
                if fresh {
                    self.pod = None;
                }
                if self.pod.is_none() {
                    let cfg = ProvisionCfg {
                        run_id: "run-1",
                        ..provision_cfg(&self.owner, "run-1")
                    };
                    let (id, _, _) = provision_pod(
                        self.cloud,
                        &cfg,
                        &HashMap::new(),
                        |id, gpu, placement| {
                            self.db.create_pod_record(
                                id,
                                gpu,
                                &placement.cloud.to_string(),
                                placement.cost_per_hr,
                                "img",
                            )?;
                            self.db.set_pod("run-1", id, None, None)
                        },
                        |_| {},
                        &FakeClock::default(),
                    )?;
                    self.pod = Some(id);
                }
                self.statuses
                    .push(self.db.run_status("run-1")?.unwrap_or_default());
                let dirs = self.run.model_snapshot_dirs("run-1-m");
                let present = gather::repeats_present(&self.results, &dirs);
                let todo: Vec<u32> = (1..=5).filter(|n| !present.contains(n)).collect();
                self.ran.push(todo.clone());
                for n in todo {
                    heal::record_failed_cell(
                        &self.results,
                        &dirs[n as usize - 1],
                        "m",
                        "decode-curve",
                        "stand-in",
                    )?;
                    if n == 2 && self.ran.len() == 1 {
                        // The pod disappears from RunPod right after repeat 2 is gathered.
                        self.cloud.cloud.borrow_mut().clear();
                        return Err(PodLost {
                            pod_id: self.pod.clone().unwrap(),
                            reason: "gone",
                        }
                        .into());
                    }
                }
                Ok(())
            }
            fn pod_reachable(&mut self) -> bool {
                false
            }
            fn reset_setup(&mut self, _: Option<&str>) -> Result<()> {
                Ok(())
            }
            fn discard_pod(&mut self, _: Replacement) {
                let _ = self.db.clear_run_pod("run-1");
            }
            fn skip_cell(&mut self, _: Option<&str>, _: &str, _: &str) -> Result<()> {
                Ok(())
            }
            fn pause(&mut self, _: Duration) {}
        }

        let db = Db::open(":memory:").unwrap();
        let args = run_args(&["--repeats", "5"]);
        let run = Run {
            repeats: 5,
            ..requested_run(&args, &named_models(&["m"]), "abc", "run-1".into())
        };
        db.create_run_with_owner(&run, Some(&live_owner())).unwrap();
        let cloud = FakePods::default();
        let (scratch, results) = scratch_results("heal-results");
        let mut sim = Sim {
            db: &db,
            cloud: &cloud,
            owner: db.pod_owner_prefix().unwrap(),
            run,
            results: results.clone(),
            pod: None,
            ran: Vec::new(),
            statuses: Vec::new(),
        };

        let outcome = heal::supervise(&mut sim, &Cursor::default(), None).unwrap();

        assert_eq!(outcome, Outcome::Done);
        assert_eq!(sim.ran, [vec![1, 2, 3, 4, 5], vec![3, 4, 5]]);
        assert_eq!(cloud.created.get(), 2, "a second pod, once");
        assert_eq!(db.recent_runs(10).unwrap().len(), 1, "one run record");
        assert!(
            sim.statuses.iter().all(|s| s != "failed"),
            "{:?}",
            sim.statuses
        );
        std::fs::remove_dir_all(&scratch).unwrap();
    }

    /// `--reuse-pod` attaches the run to the named live pod and creates none. Mutation: ignore `a.reuse_pod`
    /// (provision anyway) and the pod count is one.
    #[test]
    fn a_reused_pod_is_not_reprovisioned() {
        let db = Db::open(":memory:").unwrap();
        db.create_pod_record("pod-warm", "H100", "SECURE", None, &run_args(&[]).image)
            .unwrap();
        db.set_pod_endpoint("pod-warm", "1.2.3.4", 40022).unwrap();
        let cloud = FakePods::default();
        cloud.seed("pod-warm", "poot-orch-x-old-a1", true);
        let args = run_args(&["--reuse-pod", "pod-warm"]);

        let (run, endpoint) =
            resolve_run(&db, &cloud, &args, &named_models(&["m"]), "abc").unwrap();

        assert_eq!(cloud.created.get(), 0);
        assert_eq!(run.pod_id.as_deref(), Some("pod-warm"));
        assert!(endpoint.is_some());
        let missing = run_args(&["--reuse-pod", "pod-nope"]);
        let err = resolve_run(
            &Db::open(":memory:").unwrap(),
            &cloud,
            &missing,
            &named_models(&["m"]),
            "abc",
        )
        .unwrap_err();
        assert!(err.downcast_ref::<Unrecoverable>().is_some(), "{err:#}");
    }
    /// SIGINT/SIGTERM: the run ends 'cancelled', its pod is deleted and recorded terminated. Mutation: leave the
    /// status alone in `finalize_on_signal` and the run stays 'sweeping' (a zombie on the dashboard).
    #[test]
    fn a_signal_ends_the_run_cancelled_with_its_pod_terminated() {
        let db = Db::open(":memory:").unwrap();
        stored_sweep(
            &db,
            "run-1",
            &run_args(&[]),
            &[],
            &live_owner(),
            Some("pod-1"),
        );
        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-run-1-a1", true);

        finalize_on_signal(&db, &cloud, "run-1", false).unwrap();

        assert_eq!(
            db.run_status("run-1").unwrap().as_deref(),
            Some("cancelled")
        );
        assert!(cloud.ids().is_empty());
        assert!(db.live_pods().unwrap().is_empty());
    }

    /// Even when the API refuses the delete the run is 'cancelled', so `reap` (which spares only unfinished
    /// runs' pods) takes the pod.
    #[test]
    fn a_refused_delete_still_cancels_the_run() {
        let db = Db::open(":memory:").unwrap();
        stored_sweep(
            &db,
            "run-1",
            &run_args(&[]),
            &[],
            &live_owner(),
            Some("pod-1"),
        );
        let stuck = FakePods::refusing_deletes();
        stuck.seed("pod-1", "poot-orch-x-run-1-a1", true);

        finalize_on_signal(&db, &stuck, "run-1", false).unwrap();

        assert_eq!(
            db.run_status("run-1").unwrap().as_deref(),
            Some("cancelled")
        );
    }

    fn zombie_pass(db: &Db, cloud: &FakePods, now: u64) -> usize {
        let owner = live_owner();
        abandon_zombies(db, Some(cloud), now, 24 * 3600, |o| {
            if *o == owner {
                OwnerLiveness::Live
            } else {
                OwnerLiveness::Dead
            }
        })
        .unwrap()
    }

    /// A dead owner's run is abandoned when it has nothing to resume onto (its pod is gone) or is past the
    /// resume window (its live pod is terminated); a dead owner's run with a live pod inside the window stays
    /// resumable; a live owner's run is never touched. Mutation: also abandon live-owner runs (judge
    /// `!= Unknown`) and the live-owner assertions go red.
    #[test]
    fn dead_owner_zombies_are_abandoned_and_live_owners_are_never_touched() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        // Started at epoch 1000; `now` is far past the window for the old ones, close for the fresh one.
        stored_sweep(
            &db,
            "run-1000-1",
            &args,
            &named_models(&["gone"]),
            &dead_owner(),
            Some("pod-gone"),
        );
        stored_sweep(
            &db,
            "run-1000-2",
            &args,
            &named_models(&["old"]),
            &dead_owner(),
            Some("pod-old"),
        );
        stored_sweep(
            &db,
            "run-1000-3",
            &args,
            &named_models(&["live"]),
            &live_owner(),
            Some("pod-live"),
        );
        stored_sweep(
            &db,
            "run-1000-4",
            &args,
            &named_models(&["bare"]),
            &dead_owner(),
            None,
        );
        stored_sweep(
            &db,
            "run-900000-5",
            &args,
            &named_models(&["fresh"]),
            &dead_owner(),
            Some("pod-fresh"),
        );
        let cloud = FakePods::default();
        for pod in ["pod-old", "pod-live", "pod-fresh"] {
            cloud.seed(pod, &format!("poot-orch-x-{pod}-a1"), true);
        }

        let n = zombie_pass(&db, &cloud, 900_100);

        let status = |id: &str| db.run_status(id).unwrap().unwrap();
        assert_eq!(status("run-1000-1"), "abandoned", "its pod is gone");
        assert_eq!(status("run-1000-2"), "abandoned", "past the window");
        assert_eq!(status("run-1000-4"), "abandoned", "no pod to resume onto");
        assert_eq!(
            status("run-1000-3"),
            "sweeping",
            "a live owner is never touched"
        );
        assert_eq!(
            status("run-900000-5"),
            "sweeping",
            "resumable inside the window"
        );
        assert_eq!(n, 3);
        let mut left = cloud.ids();
        left.sort();
        assert_eq!(
            left,
            ["pod-fresh", "pod-live"],
            "only the unowned old pod was terminated"
        );
    }

    /// A capacity wait inside one `run` makes no run rows of its own (the old per-attempt noise): one `run`
    /// invocation is one run. Mutation: create a row per waiting pass and the count is more than one.
    #[test]
    fn a_capacity_wait_creates_exactly_one_run_row() {
        let db = Db::open(":memory:").unwrap();
        let args = run_args(&[]);
        let run = requested_run(&args, &named_models(&["m"]), "abc", "run-1".into());
        db.create_run_with_owner(&run, Some(&live_owner())).unwrap();
        let cloud = FakePods::default();
        cloud.refuse_creates.set(6);
        let clock = FakeClock::default();

        provision(
            &cloud,
            &args,
            "ssh-ed25519 AAAA",
            "run-1",
            &db,
            &clock,
            Duration::ZERO,
        )
        .unwrap();

        assert_eq!(clock.sleeps.borrow().len(), 3, "three passes waited");
        assert_eq!(db.recent_runs(10).unwrap().len(), 1);
        assert_eq!(cloud.ids().len(), 1);
    }
    fn floor_cfg<'a>(owner: &'a PodOwnerPrefix, floor: Option<&str>) -> ProvisionCfg<'a> {
        ProvisionCfg {
            min_cuda: floor.map(|f| f.parse().unwrap()),
            ..provision_cfg(owner, "run-1")
        }
    }

    fn provision_with(cloud: &FakePods, floor: Option<&str>) -> (String, FakeClock) {
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let clock = FakeClock::default();
        let (id, _, _) = provision_pod(
            cloud,
            &floor_cfg(&owner, floor),
            &HashMap::new(),
            |_, _, _| Ok(()),
            |_| {},
            &clock,
        )
        .unwrap();
        (id, clock)
    }

    /// A numeric floor: a 13.2 host passes a 13.0 floor, and 13.10 passes 13.9 (as numbers). Nothing is
    /// terminated. Mutation: compare the version text and 13.10 against 13.9 goes red.
    #[test]
    fn a_pod_at_or_above_the_floor_is_kept_comparing_numbers() {
        for (reported, floor) in [
            ("13.2", "13.0"),
            ("13.0", "13.0"),
            ("13.10", "13.9"),
            ("12.8", "12.8"),
        ] {
            let cloud = FakePods::default();
            cloud
                .next_cuda
                .borrow_mut()
                .push_back(Some(reported.into()));
            let (id, clock) = provision_with(&cloud, Some(floor));
            assert_eq!(id, "fake-pod-1", "{reported} vs {floor}");
            assert!(cloud.deleted.borrow().is_empty(), "{reported} vs {floor}");
            assert!(clock.sleeps.borrow().is_empty());
        }
    }

    /// A pod below the floor is terminated at once and the next attempt continues (a short pause, no capacity
    /// backoff); the second pod, at the floor, is kept. Mutation: keep a below-floor pod and the first assertion goes red.
    #[test]
    fn a_pod_below_the_floor_is_terminated_and_the_next_attempt_continues() {
        let cloud = FakePods::default();
        cloud
            .next_cuda
            .borrow_mut()
            .extend([Some("12.4".to_string()), Some("13.2".to_string())]);

        let (id, clock) = provision_with(&cloud, Some("13.0"));

        assert_eq!(id, "fake-pod-2", "the 12.4 pod is not the one kept");
        assert_eq!(*cloud.deleted.borrow(), ["fake-pod-1"]);
        assert_eq!(cloud.ids(), ["fake-pod-2"]);
        // The short fixed pause comes first; with one GPU type the pass then ends like any capacity miss, so the
        // ordinary backoff follows it.
        let sleeps = clock.sleeps.borrow();
        assert_eq!(sleeps[0], BELOW_FLOOR_PAUSE);
        assert_eq!(sleeps.len(), 2, "{sleeps:?}");
    }

    /// No floor is no filter and no check: a 12.4 pod is kept, and the API is not asked about its CUDA to decide.
    #[test]
    fn no_floor_means_no_filter_and_no_check() {
        let cloud = FakePods::default();
        cloud.next_cuda.borrow_mut().push_back(Some("11.8".into()));
        let (id, _) = provision_with(&cloud, None);
        assert_eq!(id, "fake-pod-1");
        assert!(cloud.deleted.borrow().is_empty());
        // And no CUDA list is ever sent to RunPod, floor or not.
        let cloud = FakePods::default();
        cloud.next_cuda.borrow_mut().push_back(Some("13.2".into()));
        provision_with(&cloud, Some("13.0"));
        assert!(
            cloud
                .bodies
                .borrow()
                .iter()
                .all(|b| b.get("allowedCudaVersions").is_none())
        );
    }

    /// A pod that reports no version when created is judged once it has a host (the endpoint is up): the floor is not
    /// skipped just because the API was early. Here the first pod only reveals 12.4 on a later read.
    /// Mutation: skip the second look and the below-floor pod is kept.
    #[test]
    fn a_pod_reporting_no_version_yet_is_checked_again_once_it_has_a_host() {
        let cloud = FakePods::default();
        cloud.late_cuda.borrow_mut().push_back(Some("12.4".into()));
        cloud
            .next_cuda
            .borrow_mut()
            .extend([None, Some("13.2".to_string())]);

        let (id, _) = provision_with(&cloud, Some("13.0"));

        assert_eq!(id, "fake-pod-2");
        assert_eq!(*cloud.deleted.borrow(), ["fake-pod-1"]);
    }

    /// An image's label is the default floor; the explicit flag wins; an image with no label has none.
    #[test]
    fn an_image_label_sets_the_default_floor_and_the_flag_overrides_it() {
        use crate::image_resolve::{ImageResolveError, MIN_CUDA_LABEL};
        struct Labels(BTreeMap<String, String>);
        impl ImageLabels for Labels {
            fn labels(&self, _: &str) -> Result<BTreeMap<String, String>, ImageResolveError> {
                Ok(self.0.clone())
            }
        }
        let labelled = Labels(BTreeMap::from([(
            MIN_CUDA_LABEL.to_string(),
            "13.0".to_string(),
        )]));
        let bare = Labels(BTreeMap::new());
        let v = |t: &str| Some(t.parse::<CudaVersion>().unwrap());
        assert_eq!(floor_for(None, "img", &labelled), v("13.0"));
        assert_eq!(floor_for(v("12.8"), "img", &labelled), v("12.8"));
        assert_eq!(floor_for(None, "img", &bare), None);
        assert_eq!(floor_for(v("13.2"), "img", &bare), v("13.2"));
    }
    /// A failure of this machine keeps the pod; so does `--keep-pod`; an ordinary failure or a clean run does not.
    /// Mutation: treat a local fault like any failure (return only the flag) and the first case goes red.
    #[test]
    fn a_local_fault_keeps_the_pod_up() {
        let fault: Result<Outcome> = Err(LocalFault("unparseable".into()).into());
        assert!(keep_pod_after(&fault, false));
        let other: Result<Outcome> = Err(anyhow!("boom"));
        assert!(!keep_pod_after(&other, false));
        assert!(!keep_pod_after(&Ok(Outcome::Done), false));
        assert!(keep_pod_after(&Ok(Outcome::Done), true));
    }

    /// `--extend-set` finds the set of a one-repeat run, which has no `-r<N>` in its name.
    #[test]
    fn extending_the_set_of_a_one_repeat_run_is_found() {
        let tmp = unique_temp_dir("extend-one");
        let set = "run-1792632197-3202502";
        let models = vec![manifest_model("dense")];
        assert!(extension_set(&tmp, set, &models).is_err());
        let dir = tmp.join(
            gather::canonical_dir(&format!("{set}-dense"), Some("NVIDIA GeForce RTX 3090"))
                .unwrap(),
        );
        std::fs::create_dir_all(&dir).unwrap();
        for file in ["env.json", "results.jsonl", "report.md"] {
            std::fs::write(dir.join(file), r#"{"gpu_name":"NVIDIA GeForce RTX 3090"}"#).unwrap();
        }
        assert_eq!(extension_set(&tmp, set, &models).unwrap(), set);
    }
}

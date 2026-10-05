//! Command handlers and the remote benchmark lifecycle.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};

use crate::cli::{ExecArgs, ImageArgs, LogsArgs, RunArgs};
use crate::clock::{iso_in_minutes, new_exec_run_id, now_iso};
use crate::db::{Db, ExecOwner, ExecReapReport, PodOwnerPrefix, ReapProtection, Run};
use crate::image_resolve::{DigestResolver, RegistryV2Resolver, pin_image};
use crate::logging::{list_logs, log, logs_dir, read_log_tail, run_streaming, set_log_sink};
use crate::manifest::{Model, parse_manifest_models, select_models};
use crate::runpod::{CreateSpec, Pod, RunPod};
use crate::ssh;

/// The RunPod operations the orchestrator uses. Production is [`RunPod`]; tests fake it.
pub(crate) trait PodApi {
    fn create_pod(&self, spec: &CreateSpec<'_>) -> Result<Pod>;
    fn get_pod(&self, id: &str) -> Result<Pod>;
    fn list_pods(&self) -> Result<Vec<Pod>>;
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
    /// the run had a pod that is gone; the in-flight sweep is lost - abort it and start a new run.
    AbortStale,
}

/// Decide how to handle an existing in-progress run, given whether its recorded pod is alive.
pub(crate) fn resume_decision(run: &Run, pod_alive_endpoint: Option<(String, u16)>) -> Resume {
    match (&run.pod_id, pod_alive_endpoint) {
        (Some(_), Some((h, p))) => Resume::Reattach(h, p),
        (Some(_), None) => Resume::AbortStale,
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

/// Provision a pod, optionally scp a binary, run a command, tear down. Reuses the bench pod machinery
/// for poot tests on real NVIDIA hardware.
pub(crate) fn cmd_exec(db: &Db, a: ExecArgs) -> Result<()> {
    use ssh::Endpoint;
    let rp = RunPod::new(require_key(a.runpod_api_key.clone())?)?;
    // Finalize and reap any killed prior exec (stale row, leaked pod) first.
    reap(db, &rp, false)?;
    let ssh_key = expand_tilde(&a.ssh_key);
    let pubkey = std::fs::read_to_string(expand_tilde(&format!("{}.pub", a.ssh_key)))
        .map_err(|e| anyhow!("read {}.pub: {e}", a.ssh_key))?;
    let env = build_exec_create_env(pubkey.trim(), &a.env)?;

    // Track the exec like a sweep: a `runs` row (kind='exec') records the pod at creation and shows the
    // job on the dashboard, plus a tailable log. `active_run` filters kind='sweep', so it is never
    // resumed; reap protects its pod while the row is non-terminal.
    let run_id = new_exec_run_id();
    let logpath = logs_dir().join(format!("{run_id}.log"));
    set_log_sink(&logpath);
    let resolver = RegistryV2Resolver::new()?;
    let ExecPod { pod_id, host, port } = acquire_exec_pod(
        db,
        &rp,
        &resolver,
        &a,
        &run_id,
        &ssh_key,
        &env,
        PROVISION_POLL_INTERVAL,
    )?;

    // Run the body, capturing the result so the pod is always torn down (unless --keep).
    // `Ok(bool)` means the command ran over SSH (so the pod's control plane is healthy) and the bool is
    // its exit status; `Err` is a transport/environment failure (`wait_ssh`, `scp_up`, `ssh_checked`)
    // before the command's result was known, i.e. a possibly half-broken pod. Card 407.
    let body = || -> Result<bool> {
        let ep = Endpoint::new(host.clone(), port, ssh_key.clone());
        ep.wait_ssh(Duration::from_secs(a.ssh_timeout_s))?;
        if let Some(bin) = &a.bin {
            let base = std::path::Path::new(bin)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "binary".to_string());
            let remote = format!("/root/{base}");
            log(format!("scp {bin} -> {remote}"));
            ep.scp_up(bin, &remote)?;
            ep.ssh_checked(&format!(
                "(command -v patchelf >/dev/null || (apt-get update -qq && apt-get install -y -qq patchelf)) >/dev/null 2>&1; \
                 patchelf --set-interpreter /lib64/ld-linux-x86-64.so.2 {remote} 2>/dev/null; chmod +x {remote}"
            ))?;
        }
        for up in &a.uploads {
            let (local, remote) = up
                .split_once(':')
                .ok_or_else(|| anyhow!("--upload must be local:remote, got {up:?}"))?;
            log(format!("scp {local} -> {remote}"));
            ep.scp_up(local, remote)?;
        }
        if let Some(setup) = &a.setup {
            log(format!("setup: {setup}"));
            let out = ep.ssh_checked(setup)?;
            if !out.trim().is_empty() {
                log(out);
            }
        }
        log(format!("exec: {}", a.cmd));
        let run = ep.ssh(&a.cmd)?;
        // Raw stdout goes to our stdout (callers parse the JSON last line) and verbatim into the run log.
        print!("{}", run.stdout);
        if !run.stderr.trim().is_empty() {
            eprint!("{}", run.stderr);
        }
        append_raw(&logpath, "=== command stdout ===\n");
        append_raw(&logpath, &run.stdout);
        if !run.stderr.trim().is_empty() {
            append_raw(&logpath, "=== command stderr ===\n");
            append_raw(&logpath, &run.stderr);
        }
        // A non-zero exit is reported through the bool, not `Err`, to stay distinct from transport failures.
        Ok(run.ok())
    };
    let result = body();

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
                let until = iso_in_minutes(minutes);
                db.set_warm(&run_id, &until).ok();
                if *command_ok {
                    log(format!(
                        "--keep-warm: pod {pod_id} kept warm until {until} ({minutes} min); the next exec with image {} adopts it",
                        a.image
                    ));
                } else {
                    log(format!(
                        "--keep-warm: command exited non-zero, but the pod's control plane answered - \
                         pod {pod_id} kept warm until {until} ({minutes} min, capped from {m}); the next \
                         exec with image {} adopts it",
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
                db.set_status(&run_id, if *command_ok { "done" } else { "failed" })
                    .ok();
            } else {
                log(format!("tearing down pod {pod_id}"));
                teardown = finish_exec_pod(
                    db,
                    &rp,
                    &run_id,
                    &pod_id,
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
                db.set_status(&run_id, "failed").ok();
            } else {
                log(format!("tearing down pod {pod_id}"));
                teardown = finish_exec_pod(db, &rp, &run_id, &pod_id, "failed");
            }
            log(format!("EXEC_EXIT=1 ({e})"));
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
#[allow(clippy::too_many_arguments)]
fn acquire_exec_pod<P: PodApi + ?Sized>(
    db: &Db,
    pods: &P,
    resolver: &dyn DigestResolver,
    a: &ExecArgs,
    run_id: &str,
    ssh_key: &str,
    env: &HashMap<String, String>,
    poll_interval: Duration,
) -> Result<ExecPod> {
    use ssh::Endpoint;
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
    let adopted: Option<ExecPod> = match db.find_warm_exec(&image)? {
        Some(w) if w.warm_until.as_str() > now_iso().as_str() => {
            let ep = Endpoint::new(w.host.clone(), w.port, ssh_key.to_string());
            if ep.wait_ssh(Duration::from_secs(20)).is_ok() {
                if db.try_adopt_warm_exec(&w, run_id, &owner)? {
                    log(format!(
                        "reusing warm pod {} at {}:{} (warm until {}) - skipping provision + image pull",
                        w.pod_id, w.host, w.port, w.warm_until
                    ));
                    if !a.env.is_empty() {
                        log("note: --env is create-time only; ignored when adopting a warm pod");
                    }
                    Some(ExecPod {
                        pod_id: w.pod_id,
                        host: w.host,
                        port: w.port,
                    })
                } else {
                    log(format!(
                        "warm pod {} changed before adoption; provisioning fresh",
                        w.pod_id
                    ));
                    None
                }
            } else {
                log(format!(
                    "warm pod {} unreachable; reaping it and provisioning fresh",
                    w.pod_id
                ));
                if db.try_abort_warm_exec(&w)? {
                    delete_pod_confirmed(pods, &w.pod_id)?;
                } else {
                    log(format!(
                        "warm pod {} changed before unreachable cleanup; not deleting it",
                        w.pod_id
                    ));
                }
                None
            }
        }
        Some(w) => {
            log(format!(
                "warm pod {} expired (until {}); reaping it and provisioning fresh",
                w.pod_id, w.warm_until
            ));
            if db.try_abort_warm_exec(&w)? {
                delete_pod_confirmed(pods, &w.pod_id)?;
            } else {
                log(format!(
                    "warm pod {} changed before expired cleanup; not deleting it",
                    w.pod_id
                ));
            }
            None
        }
        None => None,
    };

    if let Some(pod) = adopted {
        return Ok(pod);
    }
    // Same provisioning as the sweep. The on-create hook records the pod id first,
    // so a crash cannot leak the pod.
    let allowed_cuda = cuda_floor_to_allowed(a.min_cuda_version.as_deref().unwrap_or(""));
    let data_center_ids: Vec<&str> = a.data_center.iter().map(String::as_str).collect();
    let owner_prefix = db.pod_owner_prefix()?;
    let cfg = ProvisionCfg {
        gpu_types: &a.gpu_types,
        gpu_count: a.gpu_count,
        cloud: &a.cloud,
        image: &image,
        container_disk_gb: a.container_disk_gb,
        network_volume_id: None,
        data_center_ids: &data_center_ids,
        attempts: a.provision_attempts,
        timeout_s: a.provision_timeout_s,
        poll_interval,
        owner: &owner_prefix,
        run_id,
        allowed_cuda_versions: &allowed_cuda,
    };
    let (pod_id, host, port) = match provision_pod(pods, &cfg, env, |pid, _gpu| {
        db.set_pod(run_id, pid, None, None)
    }) {
        Ok(v) => v,
        Err(e) => {
            db.set_status(run_id, "failed").ok();
            log(format!("EXEC_EXIT=1 (provision failed: {e})"));
            return Err(e);
        }
    };
    db.set_pod(run_id, &pod_id, Some(&host), Some(port)).ok();
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

pub(crate) fn require_key(opt: Option<String>) -> Result<String> {
    opt.filter(|k| !k.is_empty())
        .ok_or_else(|| anyhow!("RUNPOD_API_KEY is not set (pass --runpod-api-key or the env var)"))
}

pub(crate) fn cmd_status(db: &Db, state_db: &str) -> Result<()> {
    use std::io::Write as _;

    let runs = db.recent_runs(20)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let write_result = (|| -> std::io::Result<()> {
        // Print the resolved DB path so a split-state bug is visible.
        writeln!(out, "state db: {state_db}")?;
        for (run, updated) in runs {
            writeln!(
                out,
                "{}  {:11}  {:20}  pod={}  models={}  (updated {})",
                run.id,
                run.status,
                run.git_ref,
                run.pod_id.as_deref().unwrap_or("-"),
                run.models.join(","),
                updated
            )?;
        }
        Ok(())
    })();
    finish_status_output(write_result)
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

    // Build the poot runner locally, always (cheap; a resume picks up local fixes).
    let sha = sh(
        std::process::Command::new("git")
            .args(["rev-parse", "--short", &a.git_ref])
            .current_dir(&repo),
        "git rev-parse",
    )?
    .trim()
    .to_string();
    let poot_bin = match &a.poot_bin {
        // patchelf a prebuilt binary too: a nix-built one would fail execve on the pod (missing nix loader).
        Some(p) => patchelf_for_pod(Path::new(p))?,
        None => build_poot_runner(&a.git_ref, &sha)?,
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

    let key = require_key(a.runpod_api_key.clone())?;
    let rp = RunPod::new(key.clone())?;
    // Reap orphans from a previous dead run (also finalizes killed exec rows).
    reap(&db, &rp, false)?;

    // Resume an in-progress run or create a new one.
    let (run, mut endpoint) = resolve_run(&db, &rp, &a, &models, &sha)?;
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
            if !keep_pod && let Ok(hdb) = Db::open(&state_db) {
                if let Ok(Some(pid)) = hdb.pod_id_for_run(&run_id) {
                    log(format!(
                        "signal received: tearing down pod {pid} for run {run_id} (was mid-sweep)"
                    ));
                    match RunPod::new(key.clone()) {
                        Ok(hrp) => {
                            if let Err(e) = terminate_recorded_pod(&hdb, &hrp, &pid) {
                                log(format!("  {e:#}"));
                            }
                        }
                        Err(e) => log(format!(
                            "  pod {pid} was not deleted and may still be billing: RunPod client: {e:#}"
                        )),
                    }
                } else {
                    log("signal received: no pod to tear down (not yet provisioned)");
                }
            }
            std::process::exit(130);
        }) {
            log(format!(
                "could not install SIGINT/SIGTERM teardown handler ({e}); a kill will leak this run's pod \
                 until `reap --force` or manual delete"
            ));
        }
    }

    // Run the phases; always attempt teardown afterwards.
    let result = run_phases(&db, &rp, &run, &models, &poot_bin, &a, &mut endpoint);
    // With --keep-pod the pod is released to idle for a later run to adopt; otherwise it is terminated.
    let pid = run.pod_id_now(&db);
    let status = match &result {
        Ok(()) => "done",
        Err(e) => {
            log(format!("run failed: {e:#}"));
            "failed"
        }
    };
    if let Err(teardown_error) = finish_run(&db, &rp, &run.id, pid.as_deref(), a.keep_pod, status) {
        return Err(match result {
            Ok(()) => teardown_error,
            Err(e) => teardown_error.context(format!("the run also failed: {e:#}")),
        });
    }
    if result.is_ok() {
        log("DONE");
    }
    result
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
fn resolve_run<P: PodApi + ?Sized>(
    db: &Db,
    rp: &P,
    a: &RunArgs,
    models: &[Model],
    sha: &str,
) -> Result<(Run, Option<(String, u16)>)> {
    if let Some(active) = db.active_run()? {
        let alive = match &active.pod_id {
            Some(pid) => rp.get_pod(pid).ok().and_then(|p| p.ssh_endpoint()),
            None => None,
        };
        match resume_decision(&active, alive.clone()) {
            Resume::Reattach(h, p) => {
                log(format!(
                    "resuming run {} (reattaching to pod {} at {h}:{p})",
                    active.id,
                    active.pod_id.as_deref().unwrap_or("?")
                ));
                return Ok((active, Some((h, p))));
            }
            Resume::Reprovision => {
                log(format!(
                    "resuming run {} (no live pod yet; will provision)",
                    active.id
                ));
                return Ok((active, None));
            }
            Resume::AbortStale => {
                log(format!(
                    "run {}'s pod is gone/unreachable; aborting it and starting fresh",
                    active.id
                ));
                // Terminate the stale pod now rather than waiting for reap.
                abort_stale_run(db, rp, &active)?;
            }
        }
    }
    let run = Run {
        id: format!(
            "run-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ),
        kind: "sweep".into(),
        git_ref: format!("{}={sha}", a.git_ref),
        scenario: a.scenario.clone(),
        repeats: a.repeats,
        models: models.iter().map(|m| m.id.clone()).collect(),
        gpu_types: a.gpu_types.clone(),
        image: a.image.clone(),
        status: "new".into(),
        pod_id: None,
    };
    db.create_run(&run)?;
    log(format!("created run {}", run.id));
    // Adopt a live idle pod with the same image instead of provisioning (spec 030); this skips the image
    // pull and model download. If RunPod no longer has it, mark the record terminated and provision.
    let live = db.live_pods()?;
    if let Some(pod) = crate::db::pick_idle_pod(&live, &run.image) {
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
            None => {
                log(format!("idle pod {} is gone; dropping it", pod.id));
                let _ = db.set_pod_status(&pod.id, "terminated");
            }
        }
    }
    Ok((run, None))
}

fn run_phases(
    db: &Db,
    rp: &RunPod,
    run: &Run,
    models: &[Model],
    poot_bin: &Path,
    a: &RunArgs,
    endpoint: &mut Option<(String, u16)>,
) -> Result<()> {
    let ssh_key = expand_tilde(&a.ssh_key);
    // Pod id for the active endpoint: a resumed run carries it on `run`; a fresh one is only written to
    // the DB, so capture it here for the sweep watchdog's liveness probe.
    let mut pod_id = run.pod_id.clone();
    // Provision if there is no live endpoint.
    if endpoint.is_none() {
        db.set_status(&run.id, "provisioning")?;
        let pubkey = std::fs::read_to_string(expand_tilde(&format!("{}.pub", a.ssh_key)))
            .context("read SSH public key")?;
        let (pid, host, port) = provision(rp, a, pubkey.trim(), &run.id, db)?;
        db.set_pod(&run.id, &pid, Some(&host), Some(port))?;
        pod_id = Some(pid);
        *endpoint = Some((host, port));
    }
    let (host, port) = endpoint.clone().unwrap();
    // This run now holds the pod: record its endpoint and mark it busy (from `provisioning` or `idle`).
    if let Some(pid) = &pod_id {
        let _ = db.set_pod_endpoint(pid, &host, port);
        let _ = db.assign_pod(pid, &run.id);
    }
    let ep = ssh::Endpoint::new(host, port, ssh_key);
    ep.wait_ssh(Duration::from_secs(a.ssh_timeout_s))?;
    let gpu = ep
        .ssh("nvidia-smi --query-gpu=name --format=csv,noheader")
        .map(|r| r.stdout.trim().to_string())
        .unwrap_or_default();
    log(format!("pod GPU: {gpu}"));

    db.set_status(&run.id, "setup")?;
    setup_pod(db, run, models, poot_bin, &ep, a)?;

    db.set_status(&run.id, "sweeping")?;
    sweep(db, rp, run, models, &ep, a, pod_id.as_deref())?;

    db.set_status(&run.id, "gathering")?;
    gather(db, run, models, &ep)?;

    postprocess(run, db, a)?;
    Ok(())
}

fn build_poot_runner(git_ref: &str, sha: &str) -> Result<PathBuf> {
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
    let (build_repo, _wt): (PathBuf, Option<PathBuf>) = if git_ref == "HEAD" || sha == head {
        (repo.clone(), None)
    } else {
        let wt = std::env::temp_dir().join(format!("poot-wt-{sha}"));
        if !wt.exists() {
            log(format!(
                "adding worktree for {git_ref} at {} (cold build)",
                wt.display()
            ));
            sh(
                std::process::Command::new("git")
                    .args(["worktree", "add", "--detach", wt.to_str().unwrap(), sha])
                    .current_dir(&repo),
                "git worktree add",
            )?;
        }
        (wt.clone(), Some(wt))
    };

    // The runner generates kernels at runtime, so a plain `cargo build --release` suffices.
    let runner_dir = build_repo.join("benchmarks").join("runners").join("poot");
    log(format!(
        "building the poot bench runner (release) in {}...",
        runner_dir.display()
    ));
    let built = build_runner_artifact("cargo", &runner_dir)?;
    patchelf_for_pod(&built)
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

/// Copy a locally built binary to a temp path and patchelf its interpreter to the pod's glibc loader.
/// A nix-built binary names the nix loader, absent on the pod, so `execve` fails with ENOENT (surfacing
/// in the harness's Python as a misleading `FileNotFoundError`). Both the build-from-ref path and
/// `--poot-bin` go through here (a `--poot-bin` that skipped it broke run-1781808494).
fn patchelf_for_pod(src: &Path) -> Result<PathBuf> {
    if !src.exists() {
        bail!("poot runner not found: {}", src.display());
    }
    let pod_bin = std::env::temp_dir().join("poot-bench-runner-pod");
    std::fs::copy(src, &pod_bin)?;
    if which("patchelf") {
        sh(
            std::process::Command::new("patchelf").args([
                "--set-interpreter",
                "/lib64/ld-linux-x86-64.so.2",
                pod_bin.to_str().unwrap(),
            ]),
            "patchelf",
        )?;
    } else {
        log("WARN: patchelf not found; the pod run may fail with rc 127 (nix interpreter)");
    }
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
    pub(crate) cloud: &'a str,
    pub(crate) image: &'a str,
    pub(crate) container_disk_gb: u32,
    pub(crate) network_volume_id: Option<&'a str>,
    /// Data centers to pin the pod to (`dataCenterIds`); empty means no pin.
    pub(crate) data_center_ids: &'a [&'a str],
    pub(crate) attempts: u32,
    pub(crate) timeout_s: u64,
    /// How often to ask whether the pod's SSH endpoint is published.
    pub(crate) poll_interval: Duration,
    /// The state database's pod prefix and the run the pods belong to; each attempt is one pod.
    pub(crate) owner: &'a PodOwnerPrefix,
    pub(crate) run_id: &'a str,
    /// whitelist of host CUDA versions (RunPod allowedCudaVersions); empty = no filter. Build with
    /// `cuda_floor_to_allowed` from a minimum version.
    pub(crate) allowed_cuda_versions: &'a [String],
}

/// The interval production waits between endpoint checks.
const PROVISION_POLL_INTERVAL: Duration = Duration::from_secs(8);

/// The RunPod `allowedCudaVersions` values >= `floor`, newest first. An empty or unparseable `floor`
/// yields an empty list (no filter). Sending the whole tail lets the pod land on any new-enough host: a
/// floor of 12.8 becomes ["12.8","12.9","13.0"].
pub(crate) fn cuda_floor_to_allowed(floor: &str) -> Vec<String> {
    // RunPod's accepted versions (POST /pods), oldest first, as (major, minor).
    const VALID: &[(u32, u32)] = &[
        (11, 8),
        (12, 0),
        (12, 1),
        (12, 2),
        (12, 3),
        (12, 4),
        (12, 5),
        (12, 6),
        (12, 7),
        (12, 8),
        (12, 9),
        (13, 0),
    ];
    let parse = |s: &str| -> Option<(u32, u32)> {
        let (a, b) = s.trim().split_once('.')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    };
    let Some(min) = parse(floor) else {
        return Vec::new();
    };
    VALID
        .iter()
        .filter(|&&v| v >= min)
        .rev() // newest-first (cosmetic; RunPod treats the list as a set)
        .map(|(a, b)| format!("{a}.{b}"))
        .collect()
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

/// Provision one GPU pod, rotating over `cfg.gpu_types` for up to `cfg.attempts` tries, and return its
/// SSH endpoint. `on_create` gets the pod id right after creation (before the SSH wait) so callers can
/// persist it and never leak a pod on a crash. A timed-out attempt deletes its pod before the next try.
fn provision_pod<P: PodApi + ?Sized>(
    pods: &P,
    cfg: &ProvisionCfg,
    env: &HashMap<String, String>,
    // Called with (pod_id, gpu_type) right after creation; gpu_type is the attempt's actual GPU.
    on_create: impl Fn(&str, &str) -> Result<()>,
) -> Result<(String, String, u16)> {
    let gpus: Vec<&str> = cfg
        .gpu_types
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if gpus.is_empty() {
        bail!("no gpu types given");
    }
    let mut last = anyhow!("no attempts");
    for attempt in 1..=cfg.attempts {
        let gpu = gpus[(attempt as usize - 1) % gpus.len()];
        log(format!(
            "provision {attempt}/{}: {gpu} ({}) image={}",
            cfg.attempts, cfg.cloud, cfg.image
        ));
        let spec = CreateSpec {
            owner: cfg.owner,
            run_id: cfg.run_id,
            attempt,
            image: cfg.image,
            gpu_type_id: gpu,
            cloud_type: cfg.cloud,
            gpu_count: cfg.gpu_count,
            container_disk_gb: cfg.container_disk_gb,
            ports: &["22/tcp"],
            env,
            network_volume_id: cfg.network_volume_id,
            data_center_ids: cfg.data_center_ids,
            allowed_cuda_versions: cfg.allowed_cuda_versions,
        };
        match pods.create_pod(&spec) {
            Ok(pod) => {
                // Hand the id to the caller immediately so a crash cannot leak the pod.
                on_create(&pod.id, gpu)?;
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
                    Ok((h, p)) => return Ok((pod.id, h, p)),
                    Err(e) => {
                        last = e;
                        log(format!(
                            "  attempt {attempt} timed out; terminating pod {}",
                            pod.id
                        ));
                        // A pod the API will not delete is still billing: stop here, do not open another.
                        delete_pod_confirmed(pods, &pod.id)?;
                    }
                }
            }
            Err(e) => {
                log(format!("  create failed: {e}"));
                last = e;
            }
        }
        std::thread::sleep(Duration::from_secs(5));
    }
    Err(anyhow!(
        "could not provision a pod after {} attempts: {last}",
        cfg.attempts
    ))
}

fn provision<P: PodApi + ?Sized>(
    rp: &P,
    a: &RunArgs,
    pubkey: &str,
    run_id: &str,
    db: &Db,
) -> Result<(String, String, u16)> {
    let mut env = HashMap::new();
    env.insert("PUBLIC_KEY".to_string(), pubkey.to_string());
    let allowed_cuda = cuda_floor_to_allowed(a.min_cuda_version.as_deref().unwrap_or(""));
    // Single optional pin from `run --data-center`; empty means the field stays off the body.
    let data_center_ids: Vec<&str> = a.data_center.as_deref().into_iter().collect();
    // The prefix is minted here, before the first create call, so a pod created and then lost to a
    // crash is still named under a prefix `reap` can find.
    let owner = db.pod_owner_prefix()?;
    let cfg = ProvisionCfg {
        gpu_types: &a.gpu_types,
        gpu_count: 1,
        cloud: &a.cloud,
        image: &a.image,
        container_disk_gb: a.container_disk_gb,
        network_volume_id: a.network_volume_id.as_deref(),
        data_center_ids: &data_center_ids,
        attempts: a.provision_attempts,
        timeout_s: a.provision_timeout_s,
        poll_interval: PROVISION_POLL_INTERVAL,
        owner: &owner,
        run_id,
        allowed_cuda_versions: &allowed_cuda,
    };
    // Persist the pod id right away so reap can find it after a crash, and record it in the `pods` table
    // as `provisioning` (spec 030); run_phases promotes it to `busy`.
    provision_pod(rp, &cfg, &env, |id, gpu| {
        db.create_pod_record(id, gpu, &a.cloud, &a.image)?;
        db.set_pod(run_id, id, None, None)
    })
}

fn setup_pod(
    db: &Db,
    run: &Run,
    models: &[Model],
    poot_bin: &Path,
    ep: &ssh::Endpoint,
    a: &RunArgs,
) -> Result<()> {
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
        let st = db.model_state(&run.id, &m.id)?;
        if st.setup_done {
            log(format!("  {} already set up", m.id));
            continue;
        }
        let repo = m
            .hf_repo
            .as_deref()
            .ok_or_else(|| anyhow!("{} has no hf_repo", m.id))?;
        // Download to the dir the harness reads (local_dir, else id).
        let dst = format!("{}/{}", a.models_root(), m.dir_name());
        // Skip the download if the attached volume already holds the weights (config.json present).
        let cached = a.network_volume_id.is_some()
            && ep
                .ssh(&format!("test -f {dst}/config.json && echo Y || echo N"))?
                .stdout
                .contains('Y');
        if cached {
            log(format!(
                "  {} present on volume ({dst}); skipping download",
                m.id
            ));
        } else {
            log(format!("  fetching {} ({repo}) -> {dst}", m.id));
            ep.ssh_checked(&format!(
                "{POD_PATH} hf download {repo} --local-dir {dst} >/tmp/dl-{}.log 2>&1",
                m.id
            ))?;
        }
        // GGUF for llama.cpp (best-effort; a failure just makes that cell unsupported).
        let gguf = format!("{dst}/{}-f16.gguf", m.id);
        let conv = format!(
            "test -f /opt/llama.cpp/convert_hf_to_gguf.py && /opt/tf-venv/bin/python \
             /opt/llama.cpp/convert_hf_to_gguf.py {dst} --outfile {gguf} --outtype f16 >/tmp/gguf-{}.log 2>&1",
            m.id
        );
        if !ep.ssh(&conv)?.ok() {
            log(format!(
                "  GGUF convert for {} failed (llama.cpp cell will be skipped)",
                m.id
            ));
        }
        db.set_model_setup(&run.id, &m.id, true)?;
    }
    Ok(())
}

/// Is the pod alive per the RunPod API? Needed because `ssh::Endpoint::ssh` returns `Ok` with empty
/// output for an unreachable host, so a pod deleted out-of-band would leave the sweep polling an empty
/// log until the hours-long timeout. `get_pod` errs on 404; a non-`RUNNING` desired status means
/// stopped or terminated. Callers require consecutive failures to ride out API blips.
fn pod_alive(rp: &RunPod, pid: &str) -> bool {
    match rp.get_pod(pid) {
        Ok(p) => p.desired_status == "RUNNING",
        Err(_) => false,
    }
}

/// The `bench run --run-id` of one model's sweep on the pod: the run and the model name it.
fn sweep_run_id(run: &Run, model: &str) -> String {
    format!("{}-{model}", run.id)
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
) -> String {
    format!(
        "{POD_PATH} cd /opt/benchmarks && MODELS_DIR={models_root} POOT_BENCH_POOT_BIN=/root/poot-bench-runner \
         ./run-sweep.sh --model {model} --scenario {scenario} --run-id {run_id} --repeats {repeats} --skip-unsupported"
    )
}

fn sweep(
    db: &Db,
    rp: &RunPod,
    run: &Run,
    models: &[Model],
    ep: &ssh::Endpoint,
    a: &RunArgs,
    pod_id: Option<&str>,
) -> Result<()> {
    for m in models {
        let st = db.model_state(&run.id, &m.id)?;
        if st.sweep_done {
            log(format!("  {} sweep already done", m.id));
            continue;
        }
        let sess = format!("sweep_{}", m.id.replace(['.', '-'], "_"));
        let logf = format!("/tmp/sweep-{}.log", m.id);
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
                // `tee` truncates a stale failed log; pipefail makes SWEEP_EXIT=$? the sweep's exit, not tee's.
                let cmd = sweep_command(
                    a.models_root(),
                    &m.id,
                    &sweep_run_id(run, &m.id),
                    &run.scenario,
                    run.repeats,
                );
                ep.ssh_checked(&format!(
                    "tmux new-session -d -s {sess} 'set -o pipefail; {cmd} 2>&1 | tee {logf}; echo SWEEP_EXIT=$? >> {logf}'"
                ))?;
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
            let deadline = Instant::now()
                + Duration::from_secs(a.sweep_timeout_s.saturating_mul(u64::from(run.repeats)));
            let mut offset = 0usize;
            // Pod-liveness watchdog: probe the RunPod API every PROBE_EVERY iterations (20s each, ~3 min)
            // and abort after MAX_STRIKES consecutive down readings (~9 min), so a deleted pod fails
            // fast while a transient API blip is forgiven.
            const PROBE_EVERY: u32 = 9;
            const MAX_STRIKES: u32 = 3;
            let mut iters = 0u32;
            let mut strikes = 0u32;
            loop {
                if Instant::now() >= deadline {
                    bail!("{} sweep exceeded {}s", m.id, a.sweep_timeout_s);
                }
                std::thread::sleep(Duration::from_secs(20));
                iters += 1;
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
                            bail!(
                                "{} sweep aborted: pod {pid} is gone/not RUNNING per the RunPod API; not waiting out the {}s timeout",
                                m.id,
                                a.sweep_timeout_s
                            );
                        }
                    }
                }
                let tail = ep
                    .ssh(&format!("tail -c +{} {logf} 2>/dev/null", offset + 1))?
                    .stdout;
                for line in tail.lines() {
                    if line.contains("[progress]")
                        || line.contains("[run]")
                        || line.trim_start().starts_with("->")
                    {
                        log(format!("  {}: {}", m.id, line.trim()));
                    }
                }
                offset += tail.len();
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
                    break;
                }
            }
        }
        // The harness wrote the run directories named by `sweep_run_id` (see `Run::model_snapshot_dirs`).
        db.set_model_sweep(&run.id, &m.id, true, Some(&sweep_run_id(run, &m.id)))?;
    }
    Ok(())
}

fn gather(db: &Db, run: &Run, models: &[Model], ep: &ssh::Endpoint) -> Result<()> {
    let results = repo_root()?.join("benchmarks").join("results");
    std::fs::create_dir_all(&results)?;
    for m in models {
        let st = db.model_state(&run.id, &m.id)?;
        let Some(base) = st.pod_run_id else { continue };
        if st.gathered {
            continue;
        }
        for dir in run.model_snapshot_dirs(&base) {
            log(format!("copying snapshot {dir} back"));
            ep.scp_down(
                &format!("/opt/benchmarks/results/{dir}"),
                &(results.to_string_lossy().to_string() + "/"),
            )?;
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
        let rids = db.snapshot_dirs(&run.id)?;
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
        db.active_run()
            .ok()
            .flatten()
            .filter(|r| r.id == self.id)
            .and_then(|r| r.pod_id)
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
            }
        }
    }

    impl PodApi for FakePods {
        fn create_pod(&self, spec: &CreateSpec<'_>) -> Result<Pod> {
            let n = self.created.get() + 1;
            self.created.set(n);
            self.images.borrow_mut().push(spec.image.to_string());
            let entry = (
                format!("fake-pod-{n}"),
                spec.owner.pod_name(spec.run_id, spec.attempt),
                true,
            );
            let pod = Self::to_pod(&entry);
            self.cloud.borrow_mut().push(entry);
            Ok(pod)
        }

        fn get_pod(&self, id: &str) -> Result<Pod> {
            self.cloud
                .borrow()
                .iter()
                .find(|p| p.0 == id)
                .map(Self::to_pod)
                .ok_or_else(|| anyhow!("get_pod {id} failed (404)"))
        }

        fn list_pods(&self) -> Result<Vec<Pod>> {
            Ok(self.cloud.borrow().iter().map(Self::to_pod).collect())
        }

        fn delete_pod(&self, id: &str) -> Result<()> {
            if self.refuse_delete.get() {
                return Err(anyhow!("delete_pod {id} failed (500): refused"));
            }
            self.cloud.borrow_mut().retain(|p| p.0 != id);
            Ok(())
        }
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
            cloud: "SECURE",
            image: "img",
            container_disk_gb: 10,
            network_volume_id: None,
            data_center_ids: &[],
            attempts: 1,
            timeout_s: 1,
            poll_interval: Duration::ZERO,
            owner,
            run_id,
            allowed_cuda_versions: &[],
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
            |_, _| bail!("crashed before the pod was recorded"),
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
        db.create_pod_record("pod-1", "L40S", "SECURE", "img")
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

    /// SC-002, site 3 (`resolve_run` on a stale run): the recorded pod exists but has no SSH endpoint,
    /// so the run is judged stale; a refused delete returns the error and neither the run nor the pod
    /// record changes. `let _ = delete_pod(..)` aborts the run and forgets the pod.
    #[test]
    fn stale_run_abort_failure_keeps_the_pod_recorded_live() {
        let db = Db::open(":memory:").unwrap();
        db.create_run(&sweep_run("run-1", "sweeping", None))
            .unwrap();
        db.create_pod_record("pod-1", "L40S", "SECURE", "img")
            .unwrap();
        db.set_pod("run-1", "pod-1", None, None).unwrap();
        db.assign_pod("pod-1", "run-1").unwrap();
        let Command::Run(args) = Cli::try_parse_from(["poot-orchestrator", "run"])
            .unwrap()
            .command
        else {
            panic!("`run` parses to Command::Run");
        };

        let stuck = FakePods::refusing_deletes();
        stuck.seed("pod-1", "poot-orch-x-run-1-a1", false);
        let err = resolve_run(&db, &stuck, &args, &[], "abc").unwrap_err();
        assert!(format!("{err:#}").contains("pod-1"), "{err:#}");
        assert_eq!(db.run_status("run-1").unwrap().as_deref(), Some("sweeping"));
        assert_eq!(
            db.live_pods().unwrap().first().map(|p| p.status.as_str()),
            Some("busy")
        );

        let cloud = FakePods::default();
        cloud.seed("pod-1", "poot-orch-x-run-1-a1", false);
        let (fresh, endpoint) = resolve_run(&db, &cloud, &args, &[], "abc").unwrap();
        assert_ne!(fresh.id, "run-1", "a new run replaces the aborted one");
        assert_eq!(endpoint, None);
        assert_eq!(db.run_status("run-1").unwrap().as_deref(), Some("aborted"));
        assert!(cloud.ids().is_empty());
        assert!(
            db.live_pods().unwrap().is_empty(),
            "the pod record is terminated"
        );
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
            "no-key",
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
            "no-key",
            &HashMap::new(),
            Duration::ZERO,
        )
        .map(|_| ())
        .unwrap_err();

        assert!(format!("{err:#}").contains("denied"), "{err:#}");
        assert_eq!(cloud.created.get(), 0, "no pod was created");
        assert!(db.recent_runs(10).unwrap().is_empty(), "no run row");
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

    /// M2: repeats belongs to the run. A run started with `--repeats 5` and resumed by a plain `run` still
    /// sweeps, gathers and reports five, not the one the resuming command line defaults to.
    #[test]
    fn a_resumed_run_keeps_the_repeats_it_started_with() {
        let db = Db::open(":memory:").unwrap();
        let mut started = sweep_run("run-1", "sweeping", None);
        started.repeats = 5;
        db.create_run(&started).unwrap();
        let Command::Run(args) = Cli::try_parse_from(["poot-orchestrator", "run"])
            .unwrap()
            .command
        else {
            panic!("`run` parses to Command::Run");
        };
        assert_eq!(args.repeats, 1);

        let (resumed, endpoint) =
            resolve_run(&db, &FakePods::default(), &args, &[], "abc").unwrap();

        assert_eq!(resumed.id, "run-1");
        assert_eq!(resumed.repeats, 5);
        assert_eq!(endpoint, None);
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
        assert_eq!(db.active_run().unwrap().unwrap().repeats, 5);
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
}

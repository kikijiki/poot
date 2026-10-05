//! ssh / scp to the pod via the system OpenSSH client. Host-key checking is disabled because every
//! run gets a fresh pod.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};

/// An SSH connection to a pod: public host, the mapped 22/tcp port, and the private key path.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
    pub key: String,
}

/// Result of a remote command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Run {
    pub fn ok(&self) -> bool {
        self.code == 0
    }
}

/// Failure of [`output_until`]: the child was killed at the deadline, or it could not
/// be spawned/reaped. Timeouts are distinguished from other failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeadlineError {
    Timeout,
    Spawn(String),
    Wait(String),
}

impl std::fmt::Display for DeadlineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "command exceeded deadline"),
            Self::Spawn(e) => write!(f, "spawn: {e}"),
            Self::Wait(e) => write!(f, "wait: {e}"),
        }
    }
}

impl std::error::Error for DeadlineError {}

/// Map a finished local child's [`std::process::Output`] onto the [`Run`] seam.
fn run_from_output(out: std::process::Output) -> Run {
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Send `sig` to process group `pgid` (libc `killpg`).
///
/// Shared by [`output_until`] (kill the whole scp/ssh tree at the deadline) and the
/// host-local `HostCommand` terminator — one helper, not a second copy of the extern.
pub fn kill_process_group(pgid: i32, sig: i32) {
    unsafe extern "C" {
        #[link_name = "killpg"]
        fn libc_killpg(pgrp: i32, sig: i32) -> i32;
    }
    unsafe {
        libc_killpg(pgid, sig);
    }
}

/// Spawn `cmd` with piped output and wait with **no deadline** (the plain-wrapper
/// path used by [`Endpoint::ssh`] / [`Endpoint::scp_up`]).
///
/// R4/N1: long exec payloads (workflow.rs main suite, setup, HF download, GGUF
/// conversion) must not be killed by a budget they never opted into. Deadlines belong
/// only on [`output_until`] / the `*_until` variants.
pub fn output_plain(cmd: &mut Command) -> Result<std::process::Output, DeadlineError> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let out = cmd
        .output()
        .map_err(|e| DeadlineError::Spawn(e.to_string()))?;
    Ok(out)
}

/// Spawn `cmd` in its own process group with piped output, reap it, and **kill the
/// whole group at `deadline`**.
///
/// Output is drained on two side threads so a full pipe cannot wedge the child.
/// Killing only the direct child is not enough: `scp` spawns an `ssh` grandchild that
/// inherits stderr, so the reader thread would block on `read_to_end` until that
/// grandchild exits on its own (observed ~45s past the deadline). `process_group(0)`
/// plus `killpg` reaps the local tree with the child.
///
/// **Remote-command caveat:** a local `killpg` does not kill the command running on
/// the pod. Pod teardown / the remote census (FR-012) remains the backstop for
/// anything already exec'd remotely; this only guarantees the local ssh/scp process
/// tree does not outlive the deadline.
///
/// Returns [`DeadlineError::Timeout`] when the kill path was taken.
pub fn output_until(
    cmd: &mut Command,
    deadline: Instant,
) -> Result<std::process::Output, DeadlineError> {
    // Already-expired budget: fail closed before spawn so a child that exits in
    // microseconds cannot win the race against the in-loop deadline check.
    if Instant::now() >= deadline {
        return Err(DeadlineError::Timeout);
    }
    use std::os::unix::process::CommandExt;
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let mut child = cmd
        .spawn()
        .map_err(|e| DeadlineError::Spawn(e.to_string()))?;
    // process_group(0) makes the child's pgid equal to its pid.
    let pgid = child.id() as i32;
    let mut stdout_pipe = child.stdout.take().expect("stdout piped");
    let mut stderr_pipe = child.stderr.take().expect("stderr piped");
    let out_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        let _ = stdout_pipe.read_to_end(&mut buf);
        buf
    });
    let err_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        let _ = stderr_pipe.read_to_end(&mut buf);
        buf
    });
    // Kill the whole local group so a grandchild holding stderr cannot block the joins.
    let kill_group = |child: &mut std::process::Child| {
        kill_process_group(pgid, 9);
        let _ = child.kill();
        let _ = child.wait();
    };
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = out_thread.join().unwrap_or_default();
                let stderr = err_thread.join().unwrap_or_default();
                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    kill_group(&mut child);
                    // Readers exit once the pipes close post-kill; do not leave them hanging.
                    let _ = out_thread.join();
                    let _ = err_thread.join();
                    return Err(DeadlineError::Timeout);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(e) => {
                kill_group(&mut child);
                let _ = out_thread.join();
                let _ = err_thread.join();
                return Err(DeadlineError::Wait(e.to_string()));
            }
        }
    }
}

impl Endpoint {
    pub fn new(host: impl Into<String>, port: u16, key: impl Into<String>) -> Endpoint {
        Endpoint {
            host: host.into(),
            port,
            key: key.into(),
        }
    }

    fn base_opts(&self) -> Vec<String> {
        let mut v: Vec<String> = [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=15",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        v.push("-i".to_string());
        v.push(self.key.clone());
        v
    }

    /// Run `cmd` on the pod, capturing output, killing the ssh child at `deadline`.
    /// Does not fail on a non-zero exit; the caller inspects `code`.
    ///
    /// This is a buffered capture path: nothing streams before the command exits.
    pub fn ssh_until(&self, cmd: &str, deadline: Instant) -> Result<Run, DeadlineError> {
        if Instant::now() >= deadline {
            return Err(DeadlineError::Timeout);
        }
        let mut c = self.ssh_command(cmd);
        let out = output_until(&mut c, deadline)?;
        Ok(run_from_output(out))
    }

    /// Run `cmd` to completion, capturing output. No deadline: long exec payloads
    /// must not be killed by a budget they never opted into (R4/N1).
    /// [`Self::ssh_until`] is the deadline variant.
    pub fn ssh(&self, cmd: &str) -> Result<Run> {
        let mut c = self.ssh_command(cmd);
        let out = output_plain(&mut c).map_err(|e| anyhow!("{e}"))?;
        Ok(run_from_output(out))
    }

    /// Shared ssh argv builder so plain and `_until` options cannot drift.
    fn ssh_command(&self, cmd: &str) -> Command {
        let mut c = Command::new("ssh");
        for o in self.base_opts() {
            c.arg(o);
        }
        c.arg("-p")
            .arg(self.port.to_string())
            .arg(format!("root@{}", self.host))
            .arg(cmd);
        c
    }

    /// Run `cmd`; error if it exits non-zero (with stderr in the message).
    pub fn ssh_checked(&self, cmd: &str) -> Result<String> {
        let r = self.ssh(cmd)?;
        if !r.ok() {
            return Err(anyhow!(
                "remote command failed (rc={}): {cmd}\n{}",
                r.code,
                r.stderr.trim()
            ));
        }
        Ok(r.stdout)
    }

    /// Upload `local` to `remote`. No deadline on the plain path (R4/N1).
    /// `-r` lets directories copy whole and is a no-op for regular files.
    pub fn scp_up(&self, local: &str, remote: &str) -> Result<()> {
        let mut c = self.scp_up_command(local, remote);
        let st = output_plain(&mut c).map_err(|e| anyhow!("{e}"))?;
        if !st.status.success() {
            return Err(anyhow!(
                "scp up {local} -> {remote} failed (rc={:?})",
                st.status.code()
            ));
        }
        Ok(())
    }

    /// Shared scp-up argv builder so plain and `_until` options cannot drift.
    fn scp_up_command(&self, local: &str, remote: &str) -> Command {
        let mut c = Command::new("scp");
        for o in self.base_opts() {
            c.arg(o);
        }
        c.arg("-r")
            .arg("-P")
            .arg(self.port.to_string())
            .arg(local)
            .arg(format!("root@{}:{remote}", self.host));
        c
    }

    pub fn scp_down(&self, remote: &str, local: &str) -> Result<()> {
        let mut c = Command::new("scp");
        for o in self.base_opts() {
            c.arg(o);
        }
        c.arg("-r")
            .arg("-P")
            .arg(self.port.to_string())
            .arg(format!("root@{}:{remote}", self.host))
            .arg(local);
        let st = c.status().map_err(|e| anyhow!("spawn scp: {e}"))?;
        if !st.success() {
            return Err(anyhow!(
                "scp down {remote} -> {local} failed (rc={:?})",
                st.code()
            ));
        }
        Ok(())
    }

    /// Poll until an `echo` over SSH succeeds (PUBLIC_KEY injection lags the endpoint by tens of seconds),
    /// or `timeout` elapses.
    pub fn wait_ssh(&self, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(r) = self.ssh_until("echo ok", deadline)
                && r.ok()
                && r.stdout.contains("ok")
            {
                return Ok(());
            }
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_secs(8));
        }
        Err(anyhow!(
            "SSH never came up at {}:{} within {:?} (PUBLIC_KEY injection failed?)",
            self.host,
            self.port,
            timeout
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// R4/N1: the production plain helper (`output_plain`, used by `Endpoint::ssh` /
    /// `scp_up` with no deadline argument) waits for a child that outlives the
    /// smallest budget any deadline path in this file uses (50ms in the tests below).
    /// If a deadline were reintroduced into `output_plain` (e.g. by routing it through
    /// `output_until` with a short budget), this child would be killed early.
    #[test]
    fn n1_plain_local_child_outlives_short_budget_without_being_killed() {
        let shortest_budget_in_file = Duration::from_millis(50);
        let mut cmd = Command::new("sleep");
        cmd.arg("0.25");
        let started = Instant::now();
        let out = output_plain(&mut cmd).expect("plain path has no deadline");
        let elapsed = started.elapsed();
        assert!(
            out.status.success(),
            "plain child completed: {:?}",
            out.status
        );
        assert!(
            elapsed >= Duration::from_millis(200),
            "N1: output_plain (production plain path) must wait for the full child \
             (>{shortest_budget_in_file:?}), not kill early; elapsed={elapsed:?}"
        );
    }

    /// R4/N1: the explicit deadline variant still kills at a short budget.
    #[test]
    fn n1_deadline_variant_still_kills_at_short_budget() {
        let short_budget = Duration::from_millis(50);
        let mut cmd = Command::new("sleep");
        cmd.arg("0.25");
        let started = Instant::now();
        let err = output_until(&mut cmd, started + short_budget)
            .expect_err("deadline variant must kill at the short budget");
        let elapsed = started.elapsed();
        assert_eq!(err, DeadlineError::Timeout, "got {err:?}");
        assert!(
            elapsed < Duration::from_millis(200),
            "N1: deadline variant killed promptly (budget {short_budget:?}), elapsed={elapsed:?}"
        );
    }

    /// R4/N4: a child that spawns a grandchild holding stderr must not block the
    /// deadline join — kill the whole process group. Without killpg, only `sh` dies
    /// and `sleep` keeps the stderr pipe open until it exits on its own.
    #[test]
    fn n4_deadline_kills_whole_group_when_grandchild_holds_stderr() {
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg("sleep 2 & wait"); // grandchild `sleep` inherits stderr
        let started = Instant::now();
        let err = output_until(&mut cmd, started + Duration::from_millis(200))
            .expect_err("deadline path returns Timeout");
        let elapsed = started.elapsed();
        assert_eq!(err, DeadlineError::Timeout, "got {err:?}");
        assert!(
            elapsed < Duration::from_secs(1),
            "N4: whole group killed at the deadline (grandchild must not hold stderr past ~200ms); \
             elapsed={elapsed:?}"
        );
    }

    /// R3/F4 adapter seam: a real local child (`sleep`) is KILLED at the deadline,
    /// not allowed to run past it. Elapsed must stay far below the child's own runtime.
    #[test]
    fn output_until_kills_local_child_at_deadline() {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let started = Instant::now();
        let err = output_until(&mut cmd, started + Duration::from_millis(200))
            .expect_err("sleep 30 must not finish inside 200ms");
        let elapsed = started.elapsed();
        assert_eq!(err, DeadlineError::Timeout, "got {err:?}");
        assert!(
            elapsed < Duration::from_secs(2),
            "child must be killed at the deadline, not run to completion (elapsed={elapsed:?})"
        );
        // The process is gone: a second kill attempt on a reaped pid is unnecessary;
        // prove by spawning sleep again under a far deadline and seeing Ok quickly path
        // is not confused — the Timeout above is the contract.
    }

    /// An already-expired deadline returns Timeout before any spawn. The child must
    /// be one that would block if actually started, so a missing pre-spawn check
    /// cannot hide behind a near-instant exit race (`true` used to flake under
    /// parallel load when `try_wait` saw exit before the in-loop deadline check).
    #[test]
    fn output_until_timeout_before_child_starts_when_deadline_already_passed() {
        let mut cmd = Command::new("sleep");
        cmd.arg("30");
        let deadline = Instant::now();
        // Give the clock a clear lead: deadline is strictly in the past on entry.
        std::thread::sleep(Duration::from_millis(20));
        let err = output_until(&mut cmd, deadline)
            .expect_err("expired deadline must Timeout without waiting on the child");
        assert_eq!(
            err,
            DeadlineError::Timeout,
            "pre-spawn guard returns Timeout, got {err:?}"
        );
    }
}

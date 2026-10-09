//! Spawn-the-binary tests: a child process that cannot outlive its test, nor hold the test run's fds.
//!
//! A test that spawns `poot-serve` (or any `CARGO_BIN_EXE_*` binary) cleans the child up in a `Drop`.
//! `Drop` does not run when the test binary is SIGKILLed (a nextest timeout, a cgroup kill, the OOM
//! killer), so the child was orphaned: a `poot-serve` ran for 5h23m under `systemd --user` after its
//! test died. The child also inherited every open fd of the test process, among them the
//! `/tmp/poot-gpu.lock` descriptor the lane's `flock` wrapper holds, so the orphan kept the GPU lock
//! held and stalled every later device lane (POOT-1085).
//!
//! [`ContainChild::contained`] closes both holes before the child `exec`s:
//!
//! * `PR_SET_PDEATHSIG(SIGKILL)`: the kernel kills the child when its parent dies, by any cause.
//!   The signal follows the parent *thread* that called `spawn`, so spawn from the thread that owns the
//!   child's lifetime (the test function), not from a short-lived helper thread.
//! * `close_range(3, .., CLOSE_RANGE_CLOEXEC)`: every inherited fd above stderr is closed at `exec`, so
//!   the child never holds a lock, listener or pipe the test process opened.

use std::io;
use std::os::unix::process::CommandExt;
use std::process::Command;

/// `Command` extension: spawn the command as a child that dies with its parent and inherits no fds.
pub trait ContainChild {
    /// Contain the child (see the module docs). Call it before `spawn`, `output` or `status`.
    fn contained(&mut self) -> &mut Self;
}

impl ContainChild for Command {
    fn contained(&mut self) -> &mut Self {
        // SAFETY: the closure runs between `fork` and `exec` and calls only async-signal-safe
        // functions (`getppid`, `prctl`, `syscall`), without allocating.
        unsafe { self.pre_exec(contain_in_child) };
        self
    }
}

fn contain_in_child() -> io::Result<()> {
    // SAFETY: `getppid` has no preconditions.
    let parent = unsafe { libc::getppid() };
    // SAFETY: PR_SET_PDEATHSIG takes a signal number; the remaining arguments are ignored.
    if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // The parent may have died between `fork` and `prctl`, after which no signal will ever arrive.
    // SAFETY: as above.
    if unsafe { libc::getppid() } != parent {
        return Err(io::Error::other(
            "the parent died before the child was armed",
        ));
    }
    // SAFETY: `close_range(first, last, flags)` with CLOSE_RANGE_CLOEXEC only sets FD_CLOEXEC;
    // std's own exec-error pipe is already close-on-exec and keeps reporting a failed `exec`.
    let rc = unsafe {
        libc::syscall(
            libc::SYS_close_range,
            3 as libc::c_uint,
            libc::c_uint::MAX,
            libc::CLOSE_RANGE_CLOEXEC,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

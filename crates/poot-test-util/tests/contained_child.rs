//! POOT-1085 acceptance: a child spawned with [`ContainChild::contained`] dies with a SIGKILLed parent,
//! and holds none of the parent's fds (the `flock` descriptor of `/tmp/poot-gpu.lock` in particular).
//!
//! The parent is this very test binary re-run as `parent_role` (a no-op unless `ROLE_ENV` names the
//! lock file), so the row can SIGKILL a real parent process, which `Drop` cannot survive.

use poot_test_util::contained_child::ContainChild;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const ROLE_ENV: &str = "POOT_CONTAINED_CHILD_PARENT_LOCK";
const BOUND: Duration = Duration::from_secs(10);

/// The parent process: take an exclusive `flock` on a descriptor that is NOT close-on-exec (what the
/// lane's `flock <file> cmd` wrapper leaves open), spawn a long `sleep` child, report its pid, then wait to
/// be killed. Only active when re-run by [`spawn_parent`].
#[test]
fn parent_role() {
    let Ok(lock_path) = std::env::var(ROLE_ENV) else {
        return;
    };
    let lock = File::create(&lock_path).expect("create the lock file");
    let fd = lock.as_raw_fd();
    // SAFETY: `fd` is the open lock file; these calls only change its flags and lock state.
    unsafe {
        assert_eq!(libc::fcntl(fd, libc::F_SETFD, 0), 0, "clear FD_CLOEXEC");
        assert_eq!(
            libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB),
            0,
            "take lock"
        );
    }
    let mut child = Command::new("sleep")
        .arg("600")
        .contained()
        .spawn()
        .expect("spawn sleep");
    println!("contained-child-pid={}", child.id());
    child.wait().expect("wait for the sleeper");
}

struct Parent {
    process: Child,
    child_pid: libc::pid_t,
}

impl Drop for Parent {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
        // SAFETY: signalling a pid we spawned; ESRCH when it is already gone.
        unsafe { libc::kill(self.child_pid, libc::SIGKILL) };
    }
}

fn spawn_parent(lock_path: &Path) -> Parent {
    let mut process = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "parent_role", "--nocapture", "--test-threads=1"])
        .env(ROLE_ENV, lock_path)
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn the parent process");
    let stdout = BufReader::new(process.stdout.take().unwrap());
    let child_pid = stdout
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            line.split_once("contained-child-pid=")?
                .1
                .trim()
                .parse()
                .ok()
        })
        .expect("the parent reports its child's pid");
    Parent { process, child_pid }
}

fn lock_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "poot-contained-child-{name}-{}.lock",
        std::process::id()
    ))
}

/// Whether `pid` is gone; a zombie awaiting a reaper counts as gone.
fn is_gone(pid: libc::pid_t) -> bool {
    match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Err(_) => true,
        Ok(stat) => stat
            .rsplit(')')
            .next()
            .is_some_and(|rest| rest.trim_start().starts_with('Z')),
    }
}

fn wait_until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + BOUND;
    while !done() {
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within {BOUND:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn child_dies_when_its_parent_is_sigkilled() {
    let lock = lock_path("death");
    let mut parent = spawn_parent(&lock);
    assert!(
        !is_gone(parent.child_pid),
        "the child must be running before the kill"
    );
    parent.process.kill().expect("SIGKILL the parent");
    parent.process.wait().unwrap();
    wait_until("the orphaned child's death", || is_gone(parent.child_pid));
    let _ = std::fs::remove_file(lock);
}

#[test]
fn child_holds_no_inherited_lock_descriptor() {
    let lock = lock_path("fd");
    let parent = spawn_parent(&lock);
    let lock_inode = std::fs::metadata(&lock).unwrap();
    let held = std::fs::read_dir(format!("/proc/{}/fd", parent.child_pid))
        .unwrap()
        .filter_map(|entry| std::fs::metadata(entry.ok()?.path()).ok())
        .any(|meta| {
            use std::os::unix::fs::MetadataExt;
            meta.dev() == lock_inode.dev() && meta.ino() == lock_inode.ino()
        });
    assert!(
        !held,
        "the child inherited the parent's lock file descriptor"
    );
    let _ = std::fs::remove_file(lock);
}

#[test]
fn the_lock_is_free_once_the_parent_is_killed() {
    let lock = lock_path("free");
    let mut parent = spawn_parent(&lock);
    parent.process.kill().expect("SIGKILL the parent");
    parent.process.wait().unwrap();
    let file = File::open(&lock).unwrap();
    // SAFETY: `file` is open for the duration of the call.
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(taken, 0, "the lock stayed held after its owner was killed");
    let _ = std::fs::remove_file(lock);
}

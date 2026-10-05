import _thread
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


HARNESS_DIR = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("process_tree_fixture.py")
SIGTERM_DRIVER = Path(__file__).with_name("sigterm_driver.py")
SIGNAL_PRECEDENCE_DRIVER = Path(__file__).with_name("signal_precedence_driver.py")
sys.path.insert(0, str(HARNESS_DIR))

import observer  # noqa: E402
from observer import run_observed  # noqa: E402


# These tests do not depend on wall-clock timing, so they hold on a machine saturated by other work. Two rules:
# - A wait for something that MUST happen (a process to start, a record to appear, a cleanup to finish) has
#   this generous upper bound. It is never reached in a passing run, so it costs nothing, and only a real hang
#   can exhaust it.
# - Something that must EXPIRE (the observed runner's timeout, a blocked sampler's deadline) is triggered by
#   an event that proves the state the test needs, not by a short clock that races the machine.
UPPER_BOUND_S = 120.0


def process_state(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    return stat.rsplit(")", 1)[1].strip().split()[0]


def process_is_running(pid):
    # A terminated orphan may remain a zombie until the host subreaper collects it; it cannot run or hold a
    # pipe. Card 313 makes no portable grandchild-reaping claim.
    return process_state(pid) not in (None, "Z")


def wait_for_pid_record(pid_file, timeout_s=UPPER_BOUND_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            identity = [int(pid) for pid in pid_file.read_text(encoding="ascii").split()]
        except (FileNotFoundError, ValueError):
            identity = []
        if len(identity) == 4:
            return identity
        time.sleep(0.01)
    raise AssertionError(f"fixture did not write a complete four-field PID record: {pid_file}")


def wait_until_not_running(pids, timeout_s=UPPER_BOUND_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(process_is_running(pid) for pid in pids):
            return
        time.sleep(0.02)
    running = [pid for pid in pids if process_is_running(pid)]
    raise AssertionError(f"fixture processes still running: {running}")


def wait_until_absent(pids, timeout_s=UPPER_BOUND_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not any(process_state(pid) is not None for pid in pids):
            return
        time.sleep(0.02)
    present = [pid for pid in pids if process_state(pid) is not None]
    raise AssertionError(f"fixture processes were not reaped: {present}")


def wait_until_exists(path, timeout_s=UPPER_BOUND_S):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if Path(path).exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"marker never appeared: {path}")


def time_out_once_ready(pid_file, *markers):
    """Make the observed runner time out when its process tree is up, not when a short clock says so.

    Replaces the observer's wait for the runner with one that waits for the fixture's complete PID record
    (and any marker files), then reports the timeout the observer's own wait reports. The observer's real
    timeout expiry is covered by test_wait_reports_timeout_for_a_runner_that_does_not_exit.
    """

    def wait_then_time_out(proc, timeout_s, _pending_signals):
        wait_for_pid_record(pid_file)
        for marker in markers:
            wait_until_exists(marker)
        raise subprocess.TimeoutExpired(proc.args, timeout_s)

    return mock.patch.object(observer, "_wait_without_reaping", wait_then_time_out)


class ConnectionsTimeOutWhen:
    """A multiprocessing context whose pipe ends report a poll timeout once `ready()` returns.

    poll() blocks in `ready` (a wait for an event the test needs) and then reports the timeout at once, so a
    wait the observer bounds by a clock expires when the state is reached, however slow the machine is.
    """

    class _Connection:
        def __init__(self, connection, ready):
            self._connection = connection
            self._ready = ready

        def poll(self, timeout=0.0):
            self._ready()
            return False

        def __getattr__(self, name):
            return getattr(self._connection, name)

    def __init__(self, ready, context):
        self._ready = ready
        self._context = context

    def Pipe(self, duplex=True):
        parent, child = self._context.Pipe(duplex)
        return self._Connection(parent, self._ready), child

    def __getattr__(self, name):
        return getattr(self._context, name)


def pipe_reader_threads():
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "poot-observer-pipe-reader" and thread.is_alive()
    ]


def sampler_processes():
    return [
        process
        for process in multiprocessing.active_children()
        if process.name == "poot-observer-sampler" and process.is_alive()
    ]


class ObserverProcessTreeTests(unittest.TestCase):
    def observed_tree(self, mode, pid_file, timeout_s=UPPER_BOUND_S):
        """Observe a fixture process tree. Its own timeout is a bound, never reached; tests that need the
        timeout path wrap the call in time_out_once_ready."""
        return run_observed(
            [sys.executable, str(FIXTURE), mode, "--pid-file", str(pid_file)],
            timeout_s=timeout_s,
            interval_s=0.01,
            termination_grace_s=0.15,
            reader_join_timeout_s=UPPER_BOUND_S,
            sampler_join_timeout_s=UPPER_BOUND_S,
        )

    def observed_success_child(self, pid_file):
        """Observe a runner that succeeds and leaves a descendant holding its stdout open.

        The readers cannot reach EOF (the descendant keeps the pipe), so the observer waits out its reader
        deadline by design. The result line must be read before that wait starts, or a slow reader would
        lose it; so the runner's exit is not reported to the observer until the line has been read.
        """
        result_line_read = threading.Event()
        real_wait = observer._wait_without_reaping

        def wait_until_result_line_is_read(proc, timeout_s, pending_signals):
            real_wait(proc, timeout_s, pending_signals)
            self.assertTrue(result_line_read.wait(UPPER_BOUND_S), "result line was never read")

        def note_result_line(line):
            if '"framework":"fixture"' in line:
                result_line_read.set()

        with mock.patch.object(observer, "_wait_without_reaping", wait_until_result_line_is_read):
            return run_observed(
                [sys.executable, str(FIXTURE), "success-child", "--pid-file", str(pid_file)],
                timeout_s=UPPER_BOUND_S,
                interval_s=0.01,
                reader_join_timeout_s=0.5,
                sampler_join_timeout_s=UPPER_BOUND_S,
                on_line=note_result_line,
            )

    def test_wait_reports_timeout_for_a_runner_that_does_not_exit(self):
        # The runner never exits, so the expiry is the outcome at any load; only how soon is the clock's.
        runner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
        try:
            with self.assertRaises(subprocess.TimeoutExpired):
                observer._wait_without_reaping(runner, 0.05, [])
            self.assertIsNone(runner.poll(), "the wait must not reap or stop the runner")
        finally:
            runner.kill()
            runner.wait(timeout=UPPER_BOUND_S)

    def test_timeout_stops_forked_descendant_and_readers_then_next_cell_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            with time_out_once_ready(pid_file):
                result = self.observed_tree("fork-hold", pid_file)
            identity = wait_for_pid_record(pid_file)
            pids = identity[:2]

        self.assertTrue(result["timed_out"])
        self.assertIsInstance(result["returncode"], int)
        self.assertNotEqual(result["returncode"], 0)
        self.assertEqual(identity[2:], [pids[0], pids[0]])
        self.assertIn("fixture descendant holds stdout open", result["stdout"])
        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertIn(process_state(pids[1]), (None, "Z"))
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

        lines = []
        next_result = run_observed(
            [sys.executable, str(FIXTURE), "success"],
            timeout_s=UPPER_BOUND_S,
            interval_s=0.01,
            reader_join_timeout_s=UPPER_BOUND_S,
            sampler_join_timeout_s=UPPER_BOUND_S,
            on_line=lines.append,
        )
        self.assertFalse(next_result["timed_out"])
        self.assertEqual(next_result["returncode"], 0)
        self.assertIn('"framework":"fixture"', next_result["stdout"])
        self.assertIn("success progress\n", lines)
        self.assertTrue(any('"framework":"fixture"' in line for line in lines))
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_timeout_escalates_when_owned_tree_ignores_sigterm(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            with time_out_once_ready(pid_file):
                result = self.observed_tree("ignore-term", pid_file)
            identity = wait_for_pid_record(pid_file)
            pids = identity[:2]

        self.assertTrue(result["timed_out"])
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertEqual(identity[2:], [pids[0], pids[0]])
        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertIn(process_state(pids[1]), (None, "Z"))
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_ordinary_failure_keeps_status_and_is_not_timeout(self):
        result = run_observed(
            [sys.executable, str(FIXTURE), "fail"],
            timeout_s=UPPER_BOUND_S,
            interval_s=0.01,
            reader_join_timeout_s=UPPER_BOUND_S,
            sampler_join_timeout_s=UPPER_BOUND_S,
        )
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["returncode"], 7)
        self.assertIn("ordinary runner failure", result["stderr"])
        self.assertNotIn("killed after", result["stderr"])
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_keyboard_interrupt_cleans_owned_group_only(self):
        control = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        timer = None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pid_file = Path(tmp) / "pids"

                def interrupt_when_ready():
                    wait_for_pid_record(pid_file)
                    _thread.interrupt_main()

                timer = threading.Thread(target=interrupt_when_ready)
                timer.start()
                with self.assertRaises(KeyboardInterrupt):
                    self.observed_tree("fork-hold", pid_file)
                timer.join(timeout=UPPER_BOUND_S)
                pids = wait_for_pid_record(pid_file)[:2]

            wait_until_not_running(pids)
            self.assertIsNone(control.poll())
            self.assertEqual(pipe_reader_threads(), [])
            self.assertEqual(sampler_processes(), [])
        finally:
            if timer is not None:
                timer.join(timeout=UPPER_BOUND_S)
            if control.poll() is None:
                os.killpg(control.pid, signal.SIGKILL)
            control.wait(timeout=UPPER_BOUND_S)

    def test_sigterm_cleans_owned_group_then_preserves_signal_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            cleanup_marker = Path(tmp) / "cleanup-started"
            driver = subprocess.Popen(
                [sys.executable, str(SIGTERM_DRIVER), str(pid_file), str(cleanup_marker)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                pids = wait_for_pid_record(pid_file)[:2]

                # The complete record proves the child tree is running, and the observer defers SIGTERM from
                # the moment it starts, so the first one needs no settling delay. Its cleanup then signals
                # the process group; the driver writes the marker at that signal and holds the cleanup
                # until the second SIGTERM has arrived, so the second always lands during cleanup.
                driver.terminate()
                wait_until_exists(cleanup_marker)
                driver.terminate()
                self.assertEqual(driver.wait(timeout=UPPER_BOUND_S), -signal.SIGTERM)
                wait_until_not_running(pids)
            finally:
                if driver.poll() is None:
                    driver.kill()
                driver.wait(timeout=UPPER_BOUND_S)

    def test_default_sigterm_precedes_concurrent_cleanup_failure(self):
        driver = subprocess.Popen(
            [sys.executable, str(SIGNAL_PRECEDENCE_DRIVER)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, stderr = driver.communicate(timeout=UPPER_BOUND_S)
            self.assertEqual(driver.returncode, -signal.SIGTERM, stderr)
        finally:
            if driver.poll() is None:
                os.killpg(driver.pid, signal.SIGKILL)
            driver.wait(timeout=UPPER_BOUND_S)

    def test_returning_sigterm_handler_runs_before_cleanup_failure_without_sentinel(self):
        previous_handler = signal.getsignal(signal.SIGTERM)
        handled = []
        finish_sampler = observer._finish_sampler
        wait_without_reaping = observer._wait_without_reaping

        def returning_handler(signum, _frame):
            handled.append(signum)

        def interrupt_then_wait(*args, **kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            return wait_without_reaping(*args, **kwargs)

        def finish_then_fail(*args, **kwargs):
            finish_sampler(*args, **kwargs)
            raise RuntimeError("concurrent cleanup failure")

        signal.signal(signal.SIGTERM, returning_handler)
        try:
            with (
                mock.patch.object(observer, "_wait_without_reaping", interrupt_then_wait),
                mock.patch.object(observer, "_finish_sampler", finish_then_fail),
            ):
                with self.assertRaisesRegex(RuntimeError, "concurrent cleanup failure") as raised:
                    run_observed(
                        [sys.executable, str(FIXTURE), "success"],
                        timeout_s=UPPER_BOUND_S,
                        interval_s=0.01,
                        reader_join_timeout_s=UPPER_BOUND_S,
                        sampler_join_timeout_s=UPPER_BOUND_S,
                    )
            self.assertNotIsInstance(raised.exception, observer._SignalInterruption)
            self.assertEqual(handled, [signal.SIGTERM])
            self.assertIs(signal.getsignal(signal.SIGTERM), returning_handler)
            self.assertEqual(sampler_processes(), [])
        finally:
            signal.signal(signal.SIGTERM, previous_handler)

    def test_setup_window_keyboard_interrupt_cleans_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            original_popen = observer.subprocess.Popen

            def interrupt_immediately_after_popen(*args, **kwargs):
                proc = original_popen(*args, **kwargs)
                wait_for_pid_record(pid_file)
                _thread.interrupt_main()
                return proc

            with mock.patch.object(
                observer.subprocess, "Popen", interrupt_immediately_after_popen
            ):
                with self.assertRaises(KeyboardInterrupt):
                    self.observed_tree("fork-hold", pid_file)
            pids = wait_for_pid_record(pid_file)[:2]

        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_keyboard_interrupt_during_direct_child_reap_is_deferred(self):
        original_reap = observer._OwnedProcessGroup.reap

        def interrupt_during_reap(owned, timeout_s):
            _thread.interrupt_main()
            return original_reap(owned, timeout_s)

        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            with (
                mock.patch.object(observer._OwnedProcessGroup, "reap", interrupt_during_reap),
                time_out_once_ready(pid_file),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    self.observed_tree("fork-hold", pid_file)
            pids = wait_for_pid_record(pid_file)[:2]

        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_repeated_keyboard_interrupt_during_group_cleanup_is_deferred(self):
        original_signal_group = observer._signal_process_group

        def interrupt_after_signal(pgid, signum):
            original_signal_group(pgid, signum)
            _thread.interrupt_main()
            _thread.interrupt_main()

        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            with (
                mock.patch.object(observer, "_signal_process_group", interrupt_after_signal),
                time_out_once_ready(pid_file),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    self.observed_tree("ignore-term", pid_file)
            pids = wait_for_pid_record(pid_file)[:2]

        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_final_group_signal_precedes_reap_and_cleanup_is_idempotent(self):
        events = []

        class FakeProcess:
            pid = 12345
            returncode = None

            def poll(self):
                events.append(("poll", None))
                return self.returncode

            def wait(self, timeout):
                events.append(("wait", timeout))
                self.returncode = -signal.SIGTERM
                return self.returncode

        owned = observer._OwnedProcessGroup(FakeProcess())
        with mock.patch.object(
            observer,
            "_signal_process_group",
            side_effect=lambda _pgid, signum: events.append(("signal", signum)),
        ):
            owned.terminate_group(0)
            owned.reap(1)
            owned.terminate_group(0)
            owned.reap(1)

        self.assertEqual(
            events,
            [("signal", signal.SIGTERM), ("signal", signal.SIGKILL), ("wait", 1)],
        )

    def test_blocking_sampler_is_terminated_and_fails_explicitly(self):
        fake_pynvml = """
class MemoryInfo:
    used = 0

class Utilization:
    gpu = 0

def nvmlInit():
    pass

def nvmlDeviceGetHandleByIndex(_index):
    return object()

def nvmlDeviceGetMemoryInfo(_handle):
    return MemoryInfo()

def nvmlDeviceGetUtilizationRates(_handle):
    return Utilization()

def nvmlDeviceGetPowerUsage(_handle):
    return 0
"""
        fake_psutil = """
import os
from pathlib import Path
import time

class Process:
    def __init__(self, pid):
        self.pid = pid

    def children(self, recursive=True):
        return []

    def cpu_percent(self, interval=None):
        return 0.0

    def is_running(self):
        return True

    def memory_info(self):
        Path(os.environ["POOT_BLOCKING_SAMPLER_MARKER"]).write_text("entered", encoding="ascii")
        while True:
            time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "pynvml.py").write_text(fake_pynvml, encoding="ascii")
            (tmp_path / "psutil.py").write_text(fake_psutil, encoding="ascii")
            marker = tmp_path / "sampler-entered"
            pid_file = tmp_path / "pids"
            sys.path.insert(0, str(tmp_path))
            try:
                finish_sampler = observer._finish_sampler

                def finish_after_short_deadline(sampler, conn, stop, _timeout_s):
                    # The sampler is blocked in memory_info forever (the marker proves it is there), so a
                    # short deadline always expires, at any load. Only the startup wait needs to be long.
                    return finish_sampler(sampler, conn, stop, 0.2)

                with (
                    mock.patch.dict(os.environ, {"POOT_BLOCKING_SAMPLER_MARKER": str(marker)}),
                    mock.patch.object(observer, "_finish_sampler", finish_after_short_deadline),
                    time_out_once_ready(pid_file, marker),
                ):
                    with self.assertRaisesRegex(RuntimeError, "sampler did not finish"):
                        self.observed_tree("fork-hold", pid_file)
                pids = wait_for_pid_record(pid_file)[:2]
                self.assertTrue(marker.exists(), "sampler never entered the blocking metric call")
            finally:
                sys.path.remove(str(tmp_path))

        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])

    def test_blocking_term_resistant_sampler_tool_is_not_orphaned(self):
        fake_pynvml = """
def nvmlInit():
    raise RuntimeError("force nvidia-smi fallback")
"""
        fake_nvidia_smi = """#!/usr/bin/env python3
import os
from pathlib import Path
import signal
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = os.fork()
if child == 0:
    while True:
        time.sleep(60)
record = Path(os.environ["POOT_BLOCKING_TOOL_PID"])
# Atomic: the test treats the record's existence as "both processes are up and named".
partial = record.with_name(record.name + ".partial")
partial.write_text(f"{os.getpid()} {child}", encoding="ascii")
os.replace(partial, record)
while True:
    time.sleep(60)
"""
        tool_pids = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "pynvml.py").write_text(fake_pynvml, encoding="ascii")
            tool = tmp_path / "nvidia-smi"
            tool.write_text(fake_nvidia_smi, encoding="ascii")
            tool.chmod(0o755)
            pid_file = tmp_path / "tool-pid"
            sys.path.insert(0, str(tmp_path))
            try:
                # The sampler never reports ready: its startup blocks in the tool. Its startup wait must
                # expire, but only after the tool is running, so the wait is cut short by that event
                # instead of by a clock the sampler's own spawn can lose to.
                blocking_context = ConnectionsTimeOutWhen(
                    lambda: wait_until_exists(pid_file), multiprocessing.get_context("spawn")
                )
                with (
                    mock.patch.dict(
                        os.environ,
                        {
                            "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
                            "POOT_BLOCKING_TOOL_PID": str(pid_file),
                        },
                    ),
                    mock.patch.object(
                        observer.multiprocessing, "get_context", lambda _method: blocking_context
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "sampler did not initialize"):
                        run_observed(
                            [sys.executable, str(FIXTURE), "success"],
                            timeout_s=UPPER_BOUND_S,
                            interval_s=0.01,
                            reader_join_timeout_s=UPPER_BOUND_S,
                            sampler_join_timeout_s=UPPER_BOUND_S,
                        )
                self.assertTrue(pid_file.exists(), "blocking sampler tool never started")
                tool_pids = [int(pid) for pid in pid_file.read_text(encoding="ascii").split()]
                self.assertEqual(len(tool_pids), 2)
                wait_until_absent(tool_pids)
                self.assertTrue(all(process_state(pid) is None for pid in tool_pids))
                self.assertEqual(sampler_processes(), [])
            finally:
                sys.path.remove(str(tmp_path))
                for pid in tool_pids:
                    if process_is_running(pid):
                        os.kill(pid, signal.SIGKILL)
                if tool_pids:
                    wait_until_not_running(tool_pids)

    def test_sigchld_ignored_success_closes_group_and_restores_handler(self):
        previous_handler = signal.getsignal(signal.SIGCHLD)
        pids = []
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "pids"
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            try:
                result = self.observed_success_child(pid_file)
                identity = wait_for_pid_record(pid_file)
                pids = identity[:2]
                self.assertIs(signal.getsignal(signal.SIGCHLD), signal.SIG_IGN)
            finally:
                signal.signal(signal.SIGCHLD, previous_handler)

        self.assertFalse(result["timed_out"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(identity[2:], [pids[0], pids[0]])
        self.assertIn('"framework":"fixture"', result["stdout"])
        wait_until_not_running(pids)
        self.assertIsNone(process_state(pids[0]), "direct runner child was not reaped")
        self.assertIn(process_state(pids[1]), (None, "Z"))
        self.assertEqual(pipe_reader_threads(), [])
        self.assertEqual(sampler_processes(), [])


if __name__ == "__main__":
    unittest.main()

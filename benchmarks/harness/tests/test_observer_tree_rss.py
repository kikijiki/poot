import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import observer  # noqa: E402

# A parent that spawns a child holding ~200 MB resident and sleeping: the shape of llama-bench under the harness.
CHILD = "b = bytearray(200 * 1024 * 1024); b[::4096] = b'x' * len(b[::4096]); import time; print('ready', flush=True); time.sleep(60)"
PARENT = textwrap.dedent(
    f"""
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", {CHILD!r}], stdout=subprocess.PIPE, text=True)
    child.stdout.readline()
    print("ready", flush=True)
    time.sleep(60)
    """
)


def fake_proc(root, processes):
    """processes: pid -> (ppid, vmrss_kb or None, comm)."""
    for pid, (ppid, rss_kb, comm) in processes.items():
        d = Path(root) / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0\n")
        (d / "status").write_text("Name:\tx\n" + (f"VmRSS:\t{rss_kb} kB\n" if rss_kb is not None else ""))


class ProcTreeRssTests(unittest.TestCase):
    def test_sums_whole_descendant_tree_and_ignores_unrelated_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, {
                10: (1, 100, "python"),
                11: (10, 1000, "llama-bench"),
                12: (11, 10, "worker thread (x) y"),  # comm with spaces and parentheses
                13: (1, 99999, "unrelated"),
                14: (10, None, "kthread-like"),  # no VmRSS line
            })
            (Path(tmp) / "self").mkdir()  # non-numeric entries are skipped
            self.assertEqual(observer._proc_tree_rss_bytes(10, Path(tmp)), 1110 * 1024)

    def test_missing_root_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, {11: (10, 5, "orphan")})
            self.assertIsNone(observer._proc_tree_rss_bytes(10, Path(tmp)))

    def test_process_vanishing_mid_walk_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, {10: (1, 100, "p"), 11: (10, 7, "gone")})
            (Path(tmp) / "11" / "status").unlink()
            self.assertEqual(observer._proc_tree_rss_bytes(10, Path(tmp)), 100 * 1024)

    def test_real_child_allocation_is_counted_without_psutil(self):
        parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            parent_only = observer._proc_vmrss_bytes(parent.pid, Path("/proc"))
            tree = observer._proc_tree_rss_bytes(parent.pid)
            self.assertGreater(tree - parent_only, 150 * 1024 * 1024)
        finally:
            parent.kill()
            parent.wait()


if __name__ == "__main__":
    unittest.main()

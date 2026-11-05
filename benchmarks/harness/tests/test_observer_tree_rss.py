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
    """processes: pid -> (ppid, vmrss_kb or None, comm[, (anon_kb, file_kb, shmem_kb)])."""
    for pid, (ppid, rss_kb, comm, *split) in processes.items():
        d = Path(root) / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0 0\n")
        lines = "Name:\tx\n"
        if rss_kb is not None:
            lines += f"VmRSS:\t{rss_kb} kB\n"
        if split:
            anon, file, shmem = split[0]
            lines += f"RssAnon:\t{anon} kB\nRssFile:\t{file} kB\nRssShmem:\t{shmem} kB\n"
        (d / "status").write_text(lines)


class ProcTreeRssTests(unittest.TestCase):
    def test_resident_set_is_split_by_what_backs_it_over_the_whole_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, {
                10: (1, 300, "python", (100, 150, 50)),
                11: (10, 1000, "llama-bench", (200, 800, 0)),
                13: (1, 99999, "unrelated", (1, 1, 1)),
                14: (10, 7, "old-kernel-without-the-split"),  # VmRSS only: the split reads as zero
            })
            tree = observer._tree_memory(10, Path(tmp))
            self.assertEqual(tree, observer.HostMemory(rss=1307 * 1024, anon=300 * 1024, file=950 * 1024, shmem=50 * 1024))

    def test_real_process_anon_and_file_follow_what_it_touches(self):
        parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            tree = observer._tree_memory(parent.pid)
            # The child's 200 MB bytearray is anonymous, not file-backed.
            self.assertGreater(tree.anon, 150 * 1024 * 1024)
            self.assertEqual(tree.anon + tree.file + tree.shmem, tree.rss)
        finally:
            parent.kill()
            parent.wait()

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
            self.assertEqual(observer._tree_memory(10, Path(tmp)).rss, 1110 * 1024)

    def test_missing_root_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, {11: (10, 5, "orphan")})
            self.assertIsNone(observer._tree_memory(10, Path(tmp)))

    def test_process_vanishing_mid_walk_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, {10: (1, 100, "p"), 11: (10, 7, "gone")})
            (Path(tmp) / "11" / "status").unlink()
            self.assertEqual(observer._tree_memory(10, Path(tmp)).rss, 100 * 1024)

    def test_real_child_allocation_is_counted(self):
        parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(parent.stdout.readline().strip(), "ready")
            parent_only = observer._proc_status_memory(parent.pid, Path("/proc")).rss
            tree = observer._tree_memory(parent.pid).rss
            self.assertGreater(tree - parent_only, 150 * 1024 * 1024)
        finally:
            parent.kill()
            parent.wait()


if __name__ == "__main__":
    unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import host_load  # noqa: E402


def fake_proc(root, busy, cpus, tree):
    """A /proc with `cpus` CPUs and processes {pid: (ticks, ppid)} (a third item True marks a kernel thread).
    `busy` is ignored: the meter reads processes, not the machine total."""
    root = Path(root)
    per = "".join(f"cpu{i} 0 0 0 0 0 0 0 0 0 0\n" for i in range(cpus))
    staged = root / "stat.new"  # replaced atomically: the meter reads while the test rewrites
    staged.write_text(f"cpu  {busy} 0 0 0 0 0 0 0 0 0\n{per}")
    staged.replace(root / "stat")
    (root / "loadavg").write_text("0.10 0.10 0.10 1/100 1\n")
    for pid, (ticks, ppid, *kthread) in tree.items():
        flags = 0x00200000 if kthread and kthread[0] else 0
        fields = ["S", str(ppid), "0", "0", "0", "0", str(flags), "0", "0", "0", "0", str(ticks), "0", "0", "0"]
        staged = root / "p.new"
        staged.write_text(f"{pid} (x y) " + " ".join(fields) + "\n")
        (root / str(pid)).mkdir(exist_ok=True)
        staged.replace(root / str(pid) / "stat")


class Accounting(unittest.TestCase):
    def test_foreign_ticks_count_only_user_processes_outside_the_tree(self):
        tree = {10: (5, 1), 11: (7, 10), 13: (13, 11), 99: (1000, 1), 2: (50, 0, True), 77: (300, 2, True)}
        grown = {10: (9, 1), 11: (7, 10), 13: (20, 11), 99: (1100, 1), 2: (90, 0, True), 77: (900, 2, True),
                 120: (40, 1)}  # 120 was born in the window
        with tempfile.TemporaryDirectory() as before, tempfile.TemporaryDirectory() as after:
            fake_proc(before, 0, 4, tree)
            fake_proc(after, 0, 4, grown)
            ticks = host_load.foreign_ticks(host_load.snapshot(before), host_load.snapshot(after), 10)
        # pid 99 used 100 and pid 120 used 40; the cell's own 10/11/13 and the kernel threads (kworkers) do not count.
        self.assertEqual(ticks, 140)

    def test_cpu_count_reads_the_cpu_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, 0, 8, {})
            self.assertEqual(host_load.cpu_count(tmp), 8)

    def test_foreign_share_is_the_machine_share_not_the_cell_took(self):
        # 32 CPUs for 1 s at 100 ticks/s = 3200 ticks; 320 foreign ticks is 10 %.
        self.assertAlmostEqual(host_load.foreign_share(320, 32, 1.0, 100), 0.10)

    def test_verdict_flags_a_window_over_the_limit(self):
        quiet = host_load.verdict([0.01, 0.02, 0.03])
        loaded = host_load.verdict([0.01, 0.37, 0.03])
        self.assertTrue(quiet["quiet"])
        self.assertFalse(loaded["quiet"])
        self.assertEqual(loaded["windows_over_limit"], 1)
        self.assertEqual(loaded["foreign_share_max"], 0.37)

    def test_no_windows_is_unavailable_not_quiet_by_accident(self):
        self.assertEqual(host_load.verdict([]), {"available": False})

    def test_the_limit_is_half_the_noise_band_over_the_measured_slope(self):
        # +35 % at 100 % foreign share is 0.35 %/pt; half the 10 % band is 5 %: 5 / 0.35 = 14 %, rounded down.
        self.assertEqual(host_load.FOREIGN_SHARE_LIMIT, 0.10)
        self.assertLessEqual(host_load.FOREIGN_SHARE_LIMIT, 0.05 / 0.35)

    def test_wait_quiet_gives_up_after_the_wait(self):
        clock = iter([0.0, 5.0, 10.0, 20.0, 40.0])
        loud = {"quiet": False}
        quiet, waited, last = host_load.wait_quiet(
            15.0, sleep=lambda s: None, now=lambda: next(clock), probe_fn=lambda s: loud)
        self.assertFalse(quiet)
        self.assertGreaterEqual(waited, 15.0)


class Metering(unittest.TestCase):
    def meter(self, foreign_per_step):
        """Run a LoadMeter over a fake /proc where a process outside the metered tree gains `foreign_per_step` ticks
        every 20 ms; the verdict it gave."""
        import threading
        import time
        with tempfile.TemporaryDirectory() as tmp:
            me = host_load.os.getpid()
            fake_proc(tmp, 0, 4, {me: (0, 1), 500: (0, 1)})
            stop = threading.Event()

            def burn():
                ticks = 0
                while not stop.is_set():
                    ticks += foreign_per_step
                    fake_proc(tmp, 0, 4, {me: (0, 1), 500: (ticks, 1)})
                    time.sleep(0.02)

            thread = threading.Thread(target=burn)
            thread.start()
            try:
                with host_load.LoadMeter(proc_root=tmp, interval_s=0.1) as meter:
                    time.sleep(0.6)
            finally:
                stop.set()
                thread.join()
            return meter.result()

    def test_foreign_cpu_outside_the_cell_tree_is_metered(self):
        # 4 CPUs x 0.02 s x 100 ticks = 8 ticks per 20 ms: 8 foreign ticks per step is the whole machine.
        loud = self.meter(8)
        self.assertFalse(loud["quiet"])
        self.assertGreater(loud["foreign_share_max"], 0.5)

    def test_an_idle_host_is_quiet(self):
        self.assertTrue(self.meter(0)["quiet"])


class Probe(unittest.TestCase):
    def test_the_load_average_is_recorded_but_does_not_gate(self):
        # The previous repeat of a sweep leaves its own threads in the 1-minute average; the foreign share decides.
        with tempfile.TemporaryDirectory() as tmp:
            fake_proc(tmp, 0, 4, {})
            (Path(tmp) / "loadavg").write_text("7.50 3.00 1.00 1/100 1\n")
            out = host_load.probe(0.3, proc_root=tmp, interval_s=0.1)
        self.assertTrue(out["quiet"])
        self.assertEqual(out["loadavg_end"], 7.5)


class FakeMeter:
    """Stands in for LoadMeter: each cell attempt gets the next scripted verdict."""
    scripted = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def result(self):
        return dict(FakeMeter.scripted.pop(0))


def load(max_share, quiet):
    return {"available": True, "windows": 10, "foreign_share_mean": 0.01, "foreign_share_max": max_share,
            "windows_over_limit": 0 if quiet else 2, "limit": 0.10, "quiet": quiet, "loadavg_start": 0.1,
            "loadavg_end": 0.2}


class QuietCell(unittest.TestCase):
    def run_cell(self, scripted, pre_quiet=True):
        FakeMeter.scripted = list(scripted)
        runs = []

        def once():
            runs.append(1)
            return {"status": "ok", "decode_tok_s": 20.0}

        with (mock.patch.object(bench.host_load, "wait_quiet",
                                return_value=(pre_quiet, 1.0, {"quiet": pre_quiet, "foreign_share_max": 0.3})),
              mock.patch.object(bench.host_load, "LoadMeter", FakeMeter)):
            return bench.run_quiet_cell(once, 0.0), len(runs)

    def test_a_quiet_cell_is_kept_and_records_its_load(self):
        row, runs = self.run_cell([load(0.01, True)])
        self.assertEqual((runs, row["status"], row["host_load"]["discarded"]), (1, "ok", 0))
        self.assertEqual(row["host_load"]["foreign_share_max"], 0.01)

    def test_a_loaded_window_is_rejected_redone_and_counted(self):
        row, runs = self.run_cell([load(0.38, False), load(0.01, True)])
        self.assertEqual(runs, 2)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["host_load"]["discarded"], 1)
        self.assertEqual(row["host_load"]["discarded_loads"][0]["foreign_share_max"], 0.38)
        self.assertEqual(row["host_load"]["foreign_share_max"], 0.01)  # the kept attempt's load, not the discard's

    def test_a_short_pass_without_evidence_is_discarded_and_redone_for_any_framework(self):
        FakeMeter.scripted = [load(0.01, True), load(0.01, True)]
        rows = iter([
            {"status": "ok", "curve": [{"isl": 128, "iters": 3, "ttft_ms": {"p50": 370, "min": 365, "max": 690}}]},
            {"status": "ok", "curve": [{"isl": 128, "iters": 3, "ttft_ms": {"p50": 370, "min": 365, "max": 372}}]},
        ])
        with (mock.patch.object(bench.host_load, "wait_quiet", return_value=(True, 0.0, {"quiet": True})),
              mock.patch.object(bench.host_load, "LoadMeter", FakeMeter)):
            row = bench.run_quiet_cell(lambda: next(rows), 0.0)
        self.assertEqual(row["host_load"]["discarded"], 1)
        self.assertEqual(row["host_load"]["discarded_loads"][0]["uncovered_short_passes"][0][0], 128)
        self.assertEqual(row["curve"][0]["ttft_ms"]["max"], 372)

    def test_a_cell_that_never_runs_quiet_is_an_error_row_not_a_number(self):
        row, runs = self.run_cell([load(0.4, False)] * (host_load.MAX_DISCARDS + 1))
        self.assertEqual(runs, host_load.MAX_DISCARDS + 1)
        self.assertEqual(row["status"], "error")
        self.assertIn("discarded (host load above 10%", row["reason"])
        self.assertEqual(row["host_load"]["discarded"], host_load.MAX_DISCARDS + 1)

    def test_a_host_that_is_not_quiet_before_the_run_does_not_run_it(self):
        row, runs = self.run_cell([], pre_quiet=False)
        self.assertEqual(runs, 0)
        self.assertEqual(row["status"], "error")
        self.assertIn("host not quiet before the run", row["reason"])

    def test_an_unavailable_meter_is_recorded_not_discarded(self):
        row, runs = self.run_cell([{"available": False, "loadavg_start": None, "loadavg_end": None}])
        self.assertEqual((runs, row["status"]), (1, "ok"))
        self.assertFalse(row["host_load"]["available"])


if __name__ == "__main__":
    unittest.main()

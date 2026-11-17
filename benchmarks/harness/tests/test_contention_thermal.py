"""POOT-1508 and POOT-1523: a repeat records who else used the GPU and CPU and how hot the box was, a contended or
throttled repeat is named by the verdict, a hot start waits outside the locks, and a dirty or mismatched runner is
refused at publication."""

import fcntl
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import device_locks  # noqa: E402
import host_load  # noqa: E402
import perf_page  # noqa: E402
import performance_receipt as receipts  # noqa: E402
import thermal  # noqa: E402
from result_contract import committed_row_errors  # noqa: E402
from test_benchmark_compare import BASELINE, repeats  # noqa: E402
from runner_fixtures import POOT_IDENTITY  # noqa: E402
from test_host_load import fake_proc, run_windows  # noqa: E402


def write_fdinfo(root, pid, engine_ns, client="7"):
    """One DRM client of `pid` that has held the gfx engine for `engine_ns`."""
    fdinfo = Path(root) / str(pid) / "fdinfo"
    fdinfo.mkdir(exist_ok=True)
    staged = fdinfo / "5.new"
    staged.write_text(f"pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client}\ndrm-engine-gfx:\t{engine_ns} ns\n"
                      "drm-engine-capacity-gfx:\t1\n")
    staged.replace(fdinfo / "5")


class FakeGpu:
    """Stands in for the amdgpu card: gpu_busy_percent reads the next value of a script (the last one repeats)."""

    def __init__(self, *series):
        self.series = list(series)

    def util_pct(self):
        return self.series.pop(0) if len(self.series) > 1 else self.series[0]


class PlantedContender(unittest.TestCase):
    """A process that was not there when the repeat started takes the GPU mid-repeat."""

    def repeat(self, contender_gpu_ns_per_step, contender_cpu_per_step=0):
        """Six 100 ms windows; the contender is absent from the first and holds the GPU (and `contender_cpu_per_step`
        ticks per 20 ms) in each later one. Time is a stepped clock, so no scheduler delay can skip a window."""
        with tempfile.TemporaryDirectory() as tmp:
            me = os.getpid()
            fake_proc(tmp, 0, 4, {me: (0, 1)})

            def write(window):
                if window == 0:
                    return  # the contender starts mid-repeat: not there for the first window
                steps = 5 * window  # five 20 ms steps per window, cumulative
                fake_proc(tmp, 0, 4, {me: (0, 1), 500: (contender_cpu_per_step * steps, 1)})
                (Path(tmp) / "500" / "stat").write_text(
                    (Path(tmp) / "500" / "stat").read_text().replace("(x y)", "(obsidian)"))
                write_fdinfo(tmp, 500, contender_gpu_ns_per_step * steps)

            return run_windows(tmp, 6, write, gpu=FakeGpu(3, 4, 90)).result()

    def test_a_gpu_contender_started_mid_repeat_marks_the_repeat_and_is_named(self):
        # 10 ms of engine time per 20 ms step is half of the device.
        out = self.repeat(10_000_000)
        self.assertTrue(out["contended"])
        contention = out["contention"]
        self.assertGreater(contention["gpu_windows_over_limit"], 0)
        self.assertGreater(contention["contenders"]["obsidian"]["gpu_share_max"], host_load.FOREIGN_GPU_SHARE_LIMIT)
        self.assertEqual([p["name"] for p in contention["started"]], ["obsidian"])
        self.assertEqual(contention["top_end"][0]["name"], "obsidian")
        self.assertEqual(contention["gpu_busy_percent"]["before"], 3)
        self.assertEqual(contention["gpu_busy_percent"]["during_max"], 90)
        self.assertEqual(out["quiet"], True)  # the CPU stayed quiet: only the GPU contended

    def test_an_idle_gpu_is_not_contended(self):
        out = self.repeat(0)
        self.assertFalse(out["contended"])
        self.assertEqual(out["contention"]["contenders"], {})

    def test_the_cells_own_gpu_use_is_not_foreign(self):
        with tempfile.TemporaryDirectory() as tmp:
            me = os.getpid()
            fake_proc(tmp, 0, 4, {me: (0, 1)})

            def write(window):
                write_fdinfo(tmp, me, 75_000_000 * (window + 1))  # the cell's own engine time, every window

            meter = run_windows(tmp, 5, write, gpu=FakeGpu(0))
        self.assertEqual(len(meter.gpu_windows), 5)
        self.assertFalse(meter.result()["contended"])

    def test_a_cpu_hog_is_named_with_its_cpu_share(self):
        out = self.repeat(0, contender_cpu_per_step=8)  # all 4 CPUs per 20 ms step
        self.assertTrue(out["contended"])
        self.assertFalse(out["quiet"])
        self.assertGreater(out["contention"]["contenders"]["obsidian"]["cpu_share_max"], host_load.FOREIGN_SHARE_LIMIT)

    def test_the_row_names_its_contenders(self):
        row = {"host_load": {"contended": True, "contention": {"contenders": {
            "obsidian": {"cpu_share_max": 0.02, "gpu_share_max": 0.31}}}}}
        self.assertEqual(host_load.contenders(row), ["obsidian"])
        self.assertEqual(host_load.describe_contention(row), "obsidian (cpu 2%, gpu 31%)")
        self.assertEqual(host_load.contenders({"host_load": {"contended": False}}), [])
        self.assertEqual(host_load.contenders({}), [])


def publishable(rows):
    """Rows of a staged, clean release build, which is what the page's contract admits before it looks at the load."""
    return [{**row, **POOT_IDENTITY, "backend": "ptx", "binary_sha256": "a" * 64, "staged_sha256": "a" * 64}
            for row in rows]


def contended_rows():
    rows = publishable(repeats(BASELINE))
    rows[2] = {**rows[2], "host_load": {"contended": True, "contention": {"contenders": {
        "obsidian": {"cpu_share_max": 0.0, "gpu_share_max": 0.2}}}}}
    return rows


class VerdictNamesTheContender(unittest.TestCase):
    def cell(self, baseline, candidate):
        (cell,) = bench.compare_cells(baseline, candidate, receipts.POLICY)
        return cell

    def test_a_contended_repeat_makes_the_cell_unverified_and_names_it(self):
        cell = self.cell(repeats(BASELINE), contended_rows())
        self.assertEqual(cell["verdict"], receipts.UNVERIFIED)
        self.assertIn("candidate was contended: repeat 3 by obsidian (cpu 0%, gpu 20%)", cell["note"])

    def test_without_the_marker_the_same_repeats_compare(self):
        self.assertEqual(self.cell(repeats(BASELINE), repeats(BASELINE))["verdict"], receipts.WITHIN_NOISE)

    def test_a_throttled_repeat_makes_the_cell_unverified_and_names_the_sensor(self):
        rows = repeats(BASELINE)
        rows[1] = {**rows[1], "thermal": {"throttled": True, "throttled_roles": ["gpu"]}}
        cell = self.cell(repeats(BASELINE), rows)
        self.assertEqual(cell["verdict"], receipts.UNVERIFIED)
        self.assertIn("candidate was throttled: repeat 2 reached the throttle point of the gpu", cell["note"])

    def test_the_performance_page_does_not_chart_a_contended_cell(self):
        reason = perf_page.contract_exclusion(contended_rows(), receipts.POLICY)
        self.assertEqual(reason, "repeat 3 was contended by obsidian (cpu 0%, gpu 20%)")
        throttled = publishable(repeats(BASELINE))
        throttled[0] = {**throttled[0], "thermal": {"throttled": True, "throttled_roles": ["cpu"]}}
        self.assertIn("throttle point of the cpu", perf_page.contract_exclusion(throttled, receipts.POLICY))


class FakeSensors:
    """Stands in for the hwmon source: `script` is the list of readings, the last one repeating."""

    def __init__(self, *readings, driver=None):
        self.readings = list(readings)
        self.sensors = {"gpu": None, "cpu": None}
        self.driver = driver or {}
        self.reads = 0

    def read(self):
        self.reads += 1
        return dict(self.readings.pop(0) if len(self.readings) > 1 else self.readings[0])

    def driver_throttle_c(self, role):
        return self.driver.get(role)


def temps(gpu, cpu, sclk=2755.0):
    return {"gpu_c": gpu, "cpu_c": cpu, "sclk_mhz": sclk}


COOL = temps(50.0, 55.0)
HOT = temps(80.0, 85.0)


class Limits(unittest.TestCase):
    def test_a_limit_is_the_throttle_point_minus_the_margin(self):
        rules = thermal.limits(FakeSensors(COOL, driver={"gpu": 100.0}))
        self.assertEqual(rules["gpu"], {"throttle_c": 100.0, "start_limit_c": 78.0, "basis": "driver"})
        self.assertEqual(rules["cpu"], {"throttle_c": 89.0, "start_limit_c": 67.0, "basis": "observed"})

    def test_the_margin_reaches_the_start_at_which_the_second_pass_verified(self):
        self.assertEqual(thermal.OBSERVED_THROTTLE_C["gpu"] - thermal.MARGIN_C, 65.0)


class StartGate(unittest.TestCase):
    def gate(self, *readings, timeout_s=60.0, slept=None):
        clock = [0.0]
        slept = [] if slept is None else slept

        def sleep(seconds):
            slept.append(seconds)
            clock[0] += seconds

        return thermal.Gate(FakeSensors(*readings), timeout_s=timeout_s, poll_s=5.0, sleep=sleep, now=lambda: clock[0])

    def test_a_cool_box_starts_at_once(self):
        waited = self.gate(COOL).wait()
        self.assertEqual((waited["cooled"], waited["waited_s"], waited["polls"]), (True, 0.0, 1))

    def test_a_planted_hot_start_delays_the_repeat_until_it_cools(self):
        slept = []
        waited = self.gate(HOT, HOT, temps(70.0, 70.0), COOL, slept=slept).wait()
        self.assertTrue(waited["cooled"])
        self.assertEqual(slept, [5.0, 5.0, 5.0])
        self.assertEqual(waited["waited_s"], 15.0)

    def test_a_box_that_never_cools_fails_with_the_typed_reason(self):
        gate = self.gate(HOT, timeout_s=20.0)
        waited = gate.wait()
        self.assertFalse(waited["cooled"])
        self.assertEqual(waited["hot"], ["gpu", "cpu"])
        self.assertTrue(gate.refusal(waited).startswith(f"{thermal.NEVER_COOLED}: after 20 s"))
        self.assertIn("gpu 80 C > 65 C", gate.refusal(waited))

    def test_one_hot_sensor_is_enough_to_wait(self):
        waited = self.gate(temps(50.0, 85.0), timeout_s=10.0).wait()
        self.assertEqual(waited["hot"], ["cpu"])


class RunQuietCellWithTheGate(unittest.TestCase):
    def run_cell(self, gate, lock_paths=(), once=None):
        once = once or (lambda: {"status": "ok", "decode_tok_s": 20.0})
        with (mock.patch.object(bench.host_load, "wait_quiet", return_value=(True, 0.0, {"quiet": True})),
              mock.patch.object(bench.host_load, "LoadMeter", FakeMeter)):
            return bench.run_quiet_cell(once, 0.0, gate, lock_paths)

    def test_a_never_cooling_box_is_an_error_row_with_a_typed_failure_and_the_cell_never_runs(self):
        ran = []
        gate = StartGate().gate(HOT, timeout_s=20.0)
        row = self.run_cell(gate, once=lambda: ran.append(1))
        self.assertEqual((row["status"], row["failure"], ran), ("error", thermal.NEVER_COOLED, []))
        self.assertEqual(row["thermal"]["limits"]["gpu"]["start_limit_c"], 65.0)
        self.assertEqual(row["thermal"]["start"], {"gpu_c": 80.0, "cpu_c": 85.0})

    def test_a_cooled_attempt_records_limits_temperatures_and_waited_time(self):
        gate = StartGate().gate(HOT, COOL)
        row = self.run_cell(gate)
        self.assertEqual(row["status"], "ok")
        record = row["thermal"]
        self.assertEqual(record["waited_s"], 5.0)
        self.assertEqual(record["limits"]["cpu"]["start_limit_c"], 67.0)
        self.assertEqual(record["start"], {"gpu_c": 50.0, "cpu_c": 55.0})
        self.assertEqual(record["end"], {"gpu_c": 50.0, "cpu_c": 55.0})
        self.assertFalse(record["throttled"])

    def test_an_attempt_that_reaches_the_throttle_point_is_marked_not_discarded(self):
        runs = []

        def once():
            runs.append(1)
            return {"status": "ok", "decode_tok_s": 20.0}

        # start cool; the sampler later reads 88 C on the GPU
        gate = StartGate().gate(COOL, COOL, temps(88.0, 60.0))
        row = self.run_cell(gate, once=once)
        self.assertEqual(len(runs), 1)
        self.assertTrue(row["thermal"]["throttled"])
        self.assertEqual(row["thermal"]["throttled_roles"], ["gpu"])
        self.assertEqual(row["thermal"]["peak"]["gpu"], 88.0)

    def test_a_host_without_sensors_is_not_gated_and_says_so(self):
        gate = thermal.Gate(thermal.SysfsThermal("/nonexistent-hwmon"))
        row = self.run_cell(gate)
        self.assertEqual((row["status"], row["thermal"]), ("ok", {"available": False}))

    def test_the_cooldown_holds_no_lock(self):
        """While the gate waits, another process can take the GPU and the heavy lock."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = (str(Path(tmp) / "gpu.lock"), str(Path(tmp) / "heavy.lock"))
            free_during_wait = []
            clock = [0.0]

            def sleep(seconds):
                clock[0] += seconds
                for path in paths:
                    with open(path, "a") as other:
                        try:
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            free_during_wait.append(True)
                        except BlockingIOError:
                            free_during_wait.append(False)

            gate = thermal.Gate(FakeSensors(HOT, HOT, COOL), timeout_s=60.0, poll_s=5.0, sleep=sleep,
                                now=lambda: clock[0])
            held_during_run = []

            def once():
                for path in paths:
                    with open(path, "a") as other:
                        try:
                            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            held_during_run.append(False)
                        except BlockingIOError:
                            held_during_run.append(True)
                return {"status": "ok", "decode_tok_s": 20.0}

            self.run_cell(gate, lock_paths=paths, once=once)
        self.assertEqual(free_during_wait, [True] * 4)  # two sleeps, two locks each
        self.assertEqual(held_during_run, [True, True])  # the repeat itself runs under both

    def test_locks_are_taken_gpu_then_heavy(self):
        self.assertEqual(device_locks.ORDER, (device_locks.GPU_LOCK, device_locks.HEAVY_LOCK))

    def test_a_box_that_heats_up_while_the_locks_were_queued_waits_again(self):
        gate = StartGate().gate(COOL, HOT, COOL)  # cool at the gate, hot when the locks are held, cool again
        row = self.run_cell(gate)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["thermal"]["start"], {"gpu_c": 50.0, "cpu_c": 55.0})  # the repeat did not start hot


class FakeMeter:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def result(self):
        return {"available": True, "quiet": True, "contended": False}


class PrivateRunner(unittest.TestCase):
    def runner(self, directory, body="#!/bin/sh\necho one\n"):
        binary = Path(directory) / "poot-bench-runner"
        binary.write_text(body)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def test_the_sweep_measures_a_copy_the_source_cannot_change(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as stage:
            binary = self.runner(src)
            runners = {"poot": {"bin": str(binary), "cmd": ["{bin}"]}, "vllm": {"bin": "python3", "cmd": ["{bin}"]}}
            staged = bench.stage_runners(runners, stage)
            recorded = bench.binary_sha256(str(binary))
            self.assertEqual(staged["poot"]["staged_sha256"], recorded)
            self.assertTrue(staged["poot"]["bin"].startswith(stage))
            self.assertEqual(staged["vllm"], runners["vllm"])  # an external tool is not copied
            # another lane rebuilds the shared binary
            self.runner(src, "#!/bin/sh\necho two\n")
            self.assertEqual(bench.binary_sha256(staged["poot"]["bin"]), recorded)
            self.assertNotEqual(bench.binary_sha256(str(binary)), recorded)

    def test_the_environment_override_cannot_redirect_the_copy(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as stage:
            binary = self.runner(src)
            runners = {"poot": {"bin_env": "POOT_TEST_RUNNER_BIN", "bin": "unused", "cmd": ["{bin}"]}}
            with mock.patch.dict(os.environ, {"POOT_TEST_RUNNER_BIN": str(binary)}):
                staged = bench.stage_runners(runners, stage)
            self.assertNotIn("bin_env", staged["poot"])
            with mock.patch.dict(os.environ, {"POOT_TEST_RUNNER_BIN": "/elsewhere"}):
                self.assertEqual(bench.resolve_binary(staged["poot"]), staged["poot"]["bin"])


class PublicationRefusesTheRunner(unittest.TestCase):
    def row(self, **kw):
        return {**publishable(repeats(BASELINE))[0], **kw}

    def test_a_staged_clean_row_publishes(self):
        self.assertEqual(committed_row_errors(self.row()), [])

    def test_a_dirty_build_is_refused(self):
        errors = committed_row_errors(self.row(build_sha="1" * 40 + "-dirty"))
        self.assertEqual(len(errors), 1)
        self.assertIn("'-dirty'", errors[0])

    def test_a_binary_whose_checksum_differs_from_the_recorded_build_is_refused(self):
        errors = committed_row_errors(self.row(binary_sha256="b" * 64))
        self.assertEqual(len(errors), 1)
        self.assertIn("is not the recorded build", errors[0])

    def test_the_set_is_not_charted_when_one_repeat_is_mismatched(self):
        rows = [self.row() for _ in range(5)]
        rows[3] = self.row(binary_sha256="b" * 64)
        self.assertIn("is not the recorded build", perf_page.contract_exclusion(rows, receipts.POLICY))


if __name__ == "__main__":
    unittest.main()

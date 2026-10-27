import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import observer  # noqa: E402
from runner_fixtures import run_fixture_runner  # noqa: E402
from test_benchmark_repeats import write_stub_workspace  # noqa: E402

LLAMACPP_RUNNER = bench.BENCH_ROOT / "runners" / "llamacpp" / "run.py"


def llamacpp_runner():
    spec = importlib.util.spec_from_file_location("llamacpp_run", LLAMACPP_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def env_listing(gpu_name):
    return {"gpu_name": gpu_name, "captured_at_utc": "2026-10-23T00:00:00+00:00", "gpu_driver": None}


def run_two_repeats(directory, gpus):
    """`bench run --repeats 2` where the GPU the host lists at each repeat is `gpus[n]`. Returns the exit status
    and the run directories written."""
    manifest, runners = write_stub_workspace(directory)
    results = Path(directory) / "results"
    argv = ["bench", "--manifest", str(manifest), "--runners", str(runners), "run", "--framework", "poot",
            "--models-dir", str(directory), "--results-dir", str(results), "--run-id", "r", "--repeats", "2"]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(bench, "capture_env", side_effect=[env_listing(g) for g in gpus]),
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch("sys.stdout"),
        mock.patch("sys.stderr"),
    ):
        try:
            bench.main()
        except SystemExit as done:
            return done.code, sorted(p.name for p in results.iterdir()) if results.exists() else []
    raise AssertionError("bench.main returned instead of exiting")


class GpuProbeTests(unittest.TestCase):
    """A pod's GPU can die between repeats: the next repeat must not start, so no set exists for it."""

    def test_a_gpu_that_disappears_after_repeat_one_stops_before_repeat_two_writes_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, written = run_two_repeats(tmp, ["NVIDIA GeForce RTX 3090", None])
        self.assertIn("no CUDA-capable device is detected", str(status))
        self.assertEqual(written, ["r-r1"])

    def test_a_host_with_its_gpu_runs_every_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, written = run_two_repeats(tmp, ["NVIDIA GeForce RTX 3090"] * 2)
        self.assertEqual(status, 0)
        self.assertEqual(written, ["r-r1", "r-r2"])


class LlamaCppDeviceTests(unittest.TestCase):
    """llama-bench falls back to the CPU without a usable GPU and still exits 0; a GPU cell must refuse that."""

    def test_a_cpu_only_report_is_a_fallback_and_a_gpu_report_is_not(self):
        runner = llamacpp_runner()
        self.assertIn("not a GPU", runner.cpu_fallback([{"backends": "CPU"}]))
        self.assertIn("not a GPU", runner.cpu_fallback([{"backends": "BLAS,CPU"}]))
        self.assertIn("not a GPU", runner.cpu_fallback([{}]))
        self.assertIsNone(runner.cpu_fallback([{"backends": "CUDA"}]))
        self.assertIsNone(runner.cpu_fallback([{"backends": "CUDA,BLAS"}]))

    def test_the_runner_exits_with_an_error_when_llama_bench_ran_on_the_cpu(self):
        runner = llamacpp_runner()
        rows = [{"backends": "CPU", "n_prompt": 128, "n_gen": 0, "avg_ts": 5.0, "build_commit": "b"}]
        completed = mock.Mock(returncode=0, stdout=json.dumps(rows), stderr="")
        argv = ["run.py", "--gguf", "m.gguf", "--prompt-file", __file__, "--mode", "decode-curve",
                "--isl-list", "128"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(runner, "resolve_gguf", return_value="m.gguf"),
            mock.patch.object(runner.subprocess, "run", return_value=completed),
            mock.patch("sys.stdout") as out,
            self.assertRaises(SystemExit) as exited,
        ):
            runner.main()
        self.assertEqual(exited.exception.code, 1)
        printed = "".join(call.args[0] for call in out.write.call_args_list)
        self.assertIn("not a GPU", json.loads(printed.strip())["error"])

    def test_the_backend_is_the_accelerator_llama_bench_used_in_lower_case(self):
        runner = llamacpp_runner()
        self.assertEqual(runner.reported_backend([{"backends": "CUDA"}]), "cuda")
        self.assertEqual(runner.reported_backend([{"backends": "BLAS,Vulkan"}]), "vulkan")
        with self.assertRaisesRegex(ValueError, "no accelerator backend"):
            runner.reported_backend([{"backends": "CPU"}])

    def test_the_decode_curve_result_names_the_backend_llama_bench_used(self):
        runner = llamacpp_runner()
        rows = [
            {"backends": "CUDA", "n_prompt": 128, "n_gen": 0, "avg_ts": 5000.0, "build_commit": "b"},
            {"backends": "CUDA", "n_prompt": 0, "n_gen": 128, "avg_ts": 50.0, "build_commit": "b"},
        ]
        completed = mock.Mock(returncode=0, stdout=json.dumps(rows), stderr="")
        argv = ["run.py", "--gguf", "m.gguf", "--prompt-file", __file__, "--mode", "decode-curve", "--isl-list", "128"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(runner, "resolve_gguf", return_value="m.gguf"),
            mock.patch.object(runner.subprocess, "run", return_value=completed),
            mock.patch("sys.stdout") as out,
        ):
            runner.main()
        printed = "".join(call.args[0] for call in out.write.call_args_list)
        self.assertEqual(json.loads(printed.strip())["backend"], "cuda")


if __name__ == "__main__":
    unittest.main()

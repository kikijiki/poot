"""The llama.cpp runner states the device memory llama.cpp itself allocated, by class, from the measured tool's own log."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402

MIB = 2**20

# The lines of a `llama-bench -v` run of llama.cpp 0.6.0 (Vulkan, Qwen3-0.6B f16), among the thousands around them.
VULKAN_LOG = """\
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
load_tensors:      Vulkan0 model buffer size =  1137.00 MiB
load_tensors:  Vulkan_Host model buffer size =   296.75 MiB
llama_context: n_ctx                 = 256
llama_context: Vulkan_Host  output buffer size =     0.58 MiB
llama_kv_cache:    Vulkan0 KV buffer size =    28.00 MiB
sched_reserve:    Vulkan0 compute buffer size =    77.25 MiB
sched_reserve: Vulkan_Host compute buffer size =     2.09 MiB
~llama_context:    Vulkan0 compute buffer size is  77.2539 MiB, matches expectation of  77.2539 MiB
"""


def llamacpp_runner():
    spec = importlib.util.spec_from_file_location("llamacpp_run", bench.BENCH_ROOT / "runners" / "llamacpp" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParseMemoryClassesTest(unittest.TestCase):
    def test_a_vulkan_log_gives_the_three_device_classes_and_ignores_host_buffers(self):
        got = llamacpp_runner().parse_memory_classes(VULKAN_LOG)
        self.assertEqual(got, {"weights_bytes": 1137 * MIB, "kv_bytes": 28 * MIB, "workspace_bytes": round(77.25 * MIB)})

    def test_every_device_buffer_is_summed_and_recurrent_state_counts_as_kv(self):
        log = (
            "load_tensors:        CUDA0 model buffer size =  1000.00 MiB\n"
            "load_tensors:        CUDA1 model buffer size =   500.00 MiB\n"
            "load_tensors:          CPU model buffer size =   300.00 MiB\n"
            "load_tensors:   CPU_Mapped model buffer size =   100.00 MiB\n"
            "llama_kv_cache:      CUDA0 KV buffer size =    64.00 MiB\n"
            "llama_kv_cache:      CUDA1 KV buffer size =    32.00 MiB\n"
            "llama_memory_recurrent:      CUDA0 RS buffer size =     8.00 MiB\n"
            "sched_reserve:      CUDA0 compute buffer size =   100.00 MiB\n"
            "sched_reserve:      CUDA1 compute buffer size =    50.00 MiB\n"
            "sched_reserve:  CUDA_Host compute buffer size =    10.00 MiB\n"
        )
        got = llamacpp_runner().parse_memory_classes(log)
        self.assertEqual(got, {"weights_bytes": 1500 * MIB, "kv_bytes": 104 * MIB, "workspace_bytes": 150 * MIB})

    def test_an_older_builds_context_line_is_the_compute_buffer(self):
        log = VULKAN_LOG.replace("sched_reserve:    Vulkan0", "llama_context:    Vulkan0")
        self.assertEqual(llamacpp_runner().parse_memory_classes(log)["workspace_bytes"], round(77.25 * MIB))

    def test_a_log_missing_a_class_gives_none_never_a_partial_breakdown(self):
        runner = llamacpp_runner()
        for dropped in ("load_tensors:      Vulkan0", "llama_kv_cache:    Vulkan0", "sched_reserve:    Vulkan0"):
            log = "".join(line for line in VULKAN_LOG.splitlines(keepends=True) if not line.startswith(dropped))
            self.assertIsNone(runner.parse_memory_classes(log), dropped)
        self.assertIsNone(runner.parse_memory_classes(""))


class RunnerMemoryTest(unittest.TestCase):
    """The runner's whole output with a fake llama-bench: the probe's classes reach the result line, in front of the run."""

    ROWS = [
        {"backends": "Vulkan", "n_prompt": 128, "n_gen": 0, "avg_ts": 5000.0, "samples_ts": [5000.0] * 4, "build_commit": "b"},
        {"backends": "Vulkan", "n_prompt": 0, "n_gen": 8, "avg_ts": 50.0, "samples_ts": [50.0] * 4, "build_commit": "b"},
    ]

    def run_curve(self, probe_stderr, probe_returncode=0):
        runner = llamacpp_runner()
        calls = []

        def fake_run(cmd, **_kwargs):
            calls.append(cmd)
            verbose = "-v" in cmd
            return mock.Mock(returncode=probe_returncode if verbose else 0, stdout=json.dumps(self.ROWS),
                             stderr=probe_stderr if verbose else "")

        argv = ["run.py", "--gguf", "m.gguf", "--prompt-file", __file__, "--mode", "decode-curve",
                "--isl-list", "128,512", "--osl", "8"]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(runner, "resolve_gguf", return_value="m.gguf"),
            mock.patch.object(runner.subprocess, "run", side_effect=fake_run),
            mock.patch("sys.stdout") as out,
        ):
            runner.main()
        printed = "".join(call.args[0] for call in out.write.call_args_list)
        return json.loads(printed.strip()), calls

    def test_the_probe_runs_the_largest_context_first_and_its_classes_reach_the_result(self):
        result, calls = self.run_curve(VULKAN_LOG)
        self.assertEqual(result["memory_classes"]["weights_bytes"], 1137 * MIB)
        probe = calls[0]
        self.assertIn("-v", probe)
        # The deepest test of the run: the context whose KV cache is the largest.
        self.assertEqual(probe[probe.index("-d") + 1], "512")
        self.assertEqual(probe[probe.index("-n") + 1], "8")
        # No measured invocation is verbose: logging must not touch a timed run.
        self.assertTrue(all("-v" not in call for call in calls[1:]))

    def test_an_unparseable_probe_is_a_caveat_and_no_classes(self):
        result, _ = self.run_curve("no buffer lines here")
        self.assertNotIn("memory_classes", result)
        self.assertTrue(any("memory breakdown unavailable" in caveat for caveat in result["caveats"]), result["caveats"])

    def test_a_failed_probe_is_a_caveat_and_leaves_the_run_alone(self):
        result, _ = self.run_curve("", probe_returncode=1)
        self.assertNotIn("memory_classes", result)
        self.assertEqual(len(result["curve"]), 2)
        self.assertTrue(any("exited rc=1" in caveat for caveat in result["caveats"]), result["caveats"])


if __name__ == "__main__":
    unittest.main()

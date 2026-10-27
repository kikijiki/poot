"""A `bench run` turns each runner result into exactly one stored row; a printed receipt is not a result."""

import argparse
import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import observer  # noqa: E402
from runner_fixtures import run_fixture_runner, valid_curve, valid_single  # noqa: E402

# The host a run is on: `bench run` refuses a host that lists no GPU, whatever box the suite runs on.
HOST_WITH_GPU = {"gpu_name": "NVIDIA GeForce RTX 3090", "gpu_driver": None}

MANIFEST = """\
[suite]
warmup_iters = 1
timed_iters = 2

[[scenario]]
id = "single"
gen_tokens = 4

[[scenario]]
id = "curve"
kind = "decode-curve"
isl = [4, 8]
osl = 3

[[model]]
id = "fixture"
local_dir = "fixture"
match_precision = "bf16"
frameworks.poot = { precision = "bf16" }
"""


def runner_script(single, curve, args_log):
    """A runner that prints `single` for the single mode and `curve` when asked for decode-curve, and logs its argv."""
    return (
        "#!/bin/sh\n"
        f'echo "$*" >> {args_log}\n'
        'case " $* " in\n'
        f"  *' --mode decode-curve '*) printf '%s\\n' '{json.dumps(curve)}' ;;\n"
        f"  *) printf '%s\\n' '{json.dumps(single)}' ;;\n"
        "esac\n"
    )


def bench_run(directory, single, curve, manifest=MANIFEST, *, diagnostic=False, skip_repeats=None, repeats=1):
    """`bench run` over a fixture manifest with a stub runner; the rows it stored in results.jsonl."""
    directory = Path(directory)
    stub = directory / "stub-runner"
    stub.write_text(runner_script(single, curve, directory / "args.log"), encoding="ascii")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    (directory / "manifest.toml").write_text(manifest)
    (directory / "runners.toml").write_text(f'[poot]\nbin = "{stub}"\ncmd = ["{{bin}}"]\n')
    args = argparse.Namespace(
        manifest=str(directory / "manifest.toml"),
        runners=str(directory / "runners.toml"),
        model=None,
        framework="poot",
        scenario=None,
        models_dir=str(directory),
        results_dir=str(directory / "results"),
        run_id="stub-run",
        repeats=repeats,
        skip_unsupported=False,
        sample_interval_ms=50.0,
        gen_tokens=None,
        warmup=None,
        iters=None,
        diagnostic=diagnostic,
        skip_repeats=skip_repeats,
    )
    with (
        mock.patch.object(bench, "capture_env", side_effect=lambda manifest: HOST_WITH_GPU),
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": "ptx"}),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        bench.cmd_run(args)
    name = "stub-run" if repeats == 1 else f"stub-run-r{repeats}"  # the last repeat's directory
    run_dir = directory / "results" / bench.DIAGNOSTICS_DIR / name if diagnostic else directory / "results" / name
    lines = (run_dir / "results.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


class StoredRowTests(unittest.TestCase):
    def test_every_mode_writes_one_row_to_results_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = bench_run(tmp, valid_single(), valid_curve())

        self.assertEqual([row["scenario"] for row in rows], ["single", "curve"])
        self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
        single, curve = rows
        self.assertIn("decode_tok_s", single)
        self.assertNotIn("curve", single)
        self.assertEqual([point["isl"] for point in curve["curve"]], [4, 8])

    def test_a_cell_that_raises_is_an_error_row_and_the_sweep_goes_on(self):
        calls = []

        def run_cell(model, scen, fw, *args, **kwargs):
            calls.append(scen["id"])
            if scen["id"] == "single":
                raise RuntimeError("boom")
            return {"framework": fw, "model": model["id"], "scenario": scen["id"], "status": "ok", "precision": "bf16"}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(bench, "run_cell", side_effect=run_cell):
            rows = bench_run(tmp, valid_single(), valid_curve())

        self.assertEqual(calls, ["single", "curve"])
        self.assertEqual([row["status"] for row in rows], ["error", "ok"])
        self.assertIn("boom", rows[0]["reason"])

    def test_skipped_repeats_are_not_run_and_the_others_keep_their_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench_run(tmp, valid_single(), valid_curve(), repeats=3, skip_repeats="1,2")
            written = sorted(p.name for p in (Path(tmp) / "results").iterdir())

        self.assertEqual(written, ["stub-run-r3"])

    def test_a_receipt_that_is_only_printed_is_not_a_result(self):
        # What the removed card-specific flags printed: a JSON line that is no result of the cell's mode.
        receipt = {"card277_bf16_generation": "PASS", "bf16_ptx": True}
        with tempfile.TemporaryDirectory() as tmp:
            rows = bench_run(tmp, receipt, receipt)

        self.assertEqual([row["status"] for row in rows], ["error", "error"])
        for row in rows:
            self.assertIn("invalid runner result", row["reason"])
            self.assertNotIn("decode_tok_s", row)

    def test_a_scenario_chunk_reaches_the_runner_as_its_prefill_chunk_flag(self):
        chunked = MANIFEST.replace("osl = 3\n", "osl = 3\nprefill_chunk = 2\n")
        with tempfile.TemporaryDirectory() as tmp:
            bench_run(tmp, valid_single(), valid_curve(), manifest=chunked)
            curve_argv = [
                line for line in (Path(tmp) / "args.log").read_text().splitlines() if "--mode decode-curve" in line
            ]
        self.assertEqual(len(curve_argv), 1)
        self.assertIn("--prefill-chunk 2", curve_argv[0])

    def test_a_diagnostic_run_times_the_device_and_is_stored_apart(self):
        profiled = valid_single() | {"profiled": True, "device_time": {"kind": "unknown"}}
        with tempfile.TemporaryDirectory() as tmp:
            rows = bench_run(tmp, profiled, valid_curve(), diagnostic=True)
            argv = (Path(tmp) / "args.log").read_text()
            committed = sorted(path.name for path in (Path(tmp) / "results").iterdir())

        self.assertIn("--profile", argv)
        self.assertEqual(committed, [bench.DIAGNOSTICS_DIR])
        self.assertEqual(rows[0]["profiled"], True)
        self.assertEqual(rows[0]["device_time"], {"kind": "unknown"})

    def test_a_run_that_is_not_diagnostic_never_asks_for_device_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = bench_run(tmp, valid_single(), valid_curve())
            argv = (Path(tmp) / "args.log").read_text()

        self.assertNotIn("--profile", argv)
        self.assertEqual(rows[0]["profiled"], False)

    def test_a_curve_without_a_chunk_passes_no_prefill_chunk_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench_run(tmp, valid_single(), valid_curve())
            argv = (Path(tmp) / "args.log").read_text()
        self.assertNotIn("--prefill-chunk", argv)


if __name__ == "__main__":
    unittest.main()

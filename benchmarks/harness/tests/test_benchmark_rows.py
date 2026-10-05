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


def runner_script(single, curve):
    """A runner that prints `single` for the single mode and `curve` when asked for decode-curve."""
    return (
        "#!/bin/sh\n"
        'case " $* " in\n'
        f"  *' --mode decode-curve '*) printf '%s\\n' '{json.dumps(curve)}' ;;\n"
        f"  *) printf '%s\\n' '{json.dumps(single)}' ;;\n"
        "esac\n"
    )


def bench_run(directory, single, curve):
    """`bench run` over the fixture manifest with a stub runner; the rows it stored in results.jsonl."""
    directory = Path(directory)
    stub = directory / "stub-runner"
    stub.write_text(runner_script(single, curve), encoding="ascii")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    (directory / "manifest.toml").write_text(MANIFEST)
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
        repeats=1,
        skip_unsupported=False,
        sample_interval_ms=50.0,
        gen_tokens=None,
        warmup=None,
        iters=None,
    )
    with (
        mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
        mock.patch.dict(os.environ, {"POOT_BENCH_BACKEND": "ptx"}),
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        bench.cmd_run(args)
    lines = (directory / "results" / "stub-run" / "results.jsonl").read_text().splitlines()
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

    def test_a_receipt_that_is_only_printed_is_not_a_result(self):
        # What the removed card-specific flags printed: a JSON line that is no result of the cell's mode.
        receipt = {"card277_bf16_generation": "PASS", "bf16_ptx": True}
        with tempfile.TemporaryDirectory() as tmp:
            rows = bench_run(tmp, receipt, receipt)

        self.assertEqual([row["status"] for row in rows], ["error", "error"])
        for row in rows:
            self.assertIn("invalid runner result", row["reason"])
            self.assertNotIn("decode_tok_s", row)


if __name__ == "__main__":
    unittest.main()

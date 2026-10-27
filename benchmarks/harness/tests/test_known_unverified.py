"""A committed cell may stay unverified only when benchmarks/known-unverified.toml lists it, exactly as measured."""

import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402
import known_unverified  # noqa: E402
import perf_page  # noqa: E402
import performance_receipt as receipts  # noqa: E402
from test_benchmark_compare import BASELINE, DEVICE, DRIVER, HOST_CPU, repeats, result_row  # noqa: E402
from test_perf_page import ROW, write_set  # noqa: E402

# Five repeats with an interquartile range of 10 around a median of 100: the band is 34.54% of the median, far over
# the 10% the manifest tolerates for decode tok/s.
NOISY = [90.0, 95.0, 100.0, 105.0, 110.0]
NOISY_BAND = 0.3454

ENTRY = {
    "model": "m", "scenario": "single", "framework": "poot", "backend": "ptx", "device": DEVICE,
    "host_cpu": HOST_CPU, "driver": DRIVER, "pair": "headline decode_tok_s", "band": NOISY_BAND, "repeats": 5,
    "reason": "three of ten repeats ran slow on a shared host", "follow_up": "POOT-1330",
}


def toml_of(*entries):
    def value(v):
        return json.dumps(v) if isinstance(v, str) else repr(v)

    return "".join(
        "[[cell]]\n" + "".join(f"{k} = {value(v)}\n" for k, v in entry.items()) + "\n" for entry in entries
    )


class KnownUnverifiedGate(unittest.TestCase):
    """`bench compare-committed` over a results directory and a list, the way `just bench-compare` runs it."""

    def run_gate(self, rows, *entries, candidate=()):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            for i, row in enumerate(rows, 1):
                run = results / f"set-r{i}"
                run.mkdir()
                (run / "results.jsonl").write_text(json.dumps(row) + "\n")
            listed = Path(tmp) / "known-unverified.toml"
            listed.write_text(toml_of(*entries))
            return subprocess.run(
                [sys.executable, str(HARNESS_DIR / "bench.py"), "compare-committed", "--results-dir", str(results),
                 "--known-unverified", str(listed), *candidate],
                capture_output=True, text=True, check=False,
            )

    def test_an_unlisted_unverified_cell_is_red_as_before(self):
        done = self.run_gate(repeats(NOISY))

        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertIn("unverified", done.stdout)

    def test_a_listed_unverified_cell_passes_and_says_so(self):
        done = self.run_gate(repeats(NOISY), ENTRY)

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("known unverified", done.stdout)
        self.assertIn("POOT-1330", done.stdout)

    def test_a_listed_cell_that_now_verifies_is_a_stale_entry(self):
        done = self.run_gate(repeats(BASELINE), ENTRY)

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("stale exception", done.stdout)
        self.assertIn("now within noise", done.stdout)

    def test_a_listed_cell_no_committed_row_has_is_a_stale_entry(self):
        done = self.run_gate(repeats(BASELINE, model="other"), {**ENTRY, "model": "gone"})

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("no committed row has this cell", done.stdout)

    def test_an_entry_that_states_another_band_is_stale(self):
        done = self.run_gate(repeats(NOISY), {**ENTRY, "band": 0.2})

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("band 0.2 -> 0.3454", done.stdout)

    def test_an_entry_that_states_another_repeat_count_is_stale(self):
        done = self.run_gate(repeats(NOISY), {**ENTRY, "repeats": 6})

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("repeats 6 -> 5", done.stdout)

    def test_an_entry_does_not_cover_a_cell_with_no_spread_to_state(self):
        done = self.run_gate(repeats(NOISY[:4]), ENTRY)

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("a band cannot cover", done.stdout)

    def test_a_failed_repeat_among_noisy_ones_is_listed_with_its_count(self):
        rows = repeats(NOISY) + [result_row(None, status="error", backend="ptx")]

        done = self.run_gate(rows, {**ENTRY, "failed_repeats": 1})

        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("known unverified", done.stdout)

    def test_a_failed_repeat_the_entry_does_not_state_is_a_stale_entry(self):
        rows = repeats(NOISY) + [result_row(None, status="error", backend="ptx")]

        done = self.run_gate(rows, ENTRY)

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("failed_repeats 0 -> 1", done.stdout)

    def test_failed_repeats_beside_verifying_ones_are_not_covered(self):
        rows = repeats(BASELINE) + [result_row(None, status="error", backend="ptx")]

        done = self.run_gate(rows, {**ENTRY, "failed_repeats": 1})

        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("a band cannot cover", done.stdout)

    def test_a_listed_cell_does_not_hide_another_unlisted_one(self):
        done = self.run_gate(repeats(NOISY) + repeats(NOISY, backend="rocm"), ENTRY)

        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)

    def test_the_list_says_nothing_about_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "cand-r1"
            candidate.mkdir()
            (candidate / "results.jsonl").write_text(json.dumps(repeats(NOISY)[0]) + "\n")
            done = self.run_gate(repeats(NOISY), ENTRY, candidate=("--candidate", str(candidate)))

        self.assertEqual(done.returncode, 2, done.stdout + done.stderr)
        self.assertNotIn("known unverified", done.stdout)


class KnownUnverifiedList(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "known-unverified.toml"
            path.write_text(text)
            return known_unverified.load(path)

    def test_an_entry_is_keyed_like_a_cell(self):
        (key,) = self.load(toml_of(ENTRY))
        self.assertEqual(key, ("m", "single", "poot", "ptx", DEVICE, HOST_CPU, DRIVER))

    def test_an_entry_missing_its_reason_or_follow_up_is_refused(self):
        for field in ("reason", "follow_up", "band", "repeats", "pair"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"lacks {field}"):
                self.load(toml_of({k: v for k, v in ENTRY.items() if k != field}))

    def test_a_follow_up_must_name_a_card(self):
        with self.assertRaisesRegex(ValueError, "follow_up must be a card id"):
            self.load(toml_of({**ENTRY, "follow_up": "later"}))

    def test_fewer_repeats_than_a_band_needs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "repeats must be an integer of at least 5"):
            self.load(toml_of({**ENTRY, "repeats": 3}))

    def test_the_same_cell_listed_twice_is_refused(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            self.load(toml_of(ENTRY, ENTRY))

    def test_the_committed_list_loads(self):
        known_unverified.load(known_unverified.beside(bench.BENCH_ROOT / "results"))


class KnownUnverifiedPage(unittest.TestCase):
    """The Performance page's data for a committed cell the list covers."""

    def derive(self, *entries, **row_overrides):
        """The one host entry (its cells and exclusions) derived from a set of noisy repeats and the listed `entries`."""
        rows = [{**ROW, "decode_tok_s": v, **{k: f(i) for k, f in row_overrides.items()}} for i, v in enumerate(NOISY)]
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            write_set(results, "set", rows)
            listed = Path(tmp) / "known-unverified.toml"
            listed.write_text(toml_of(*entries))
            hosts = {}
            perf_page.collect_runs(results, perf_page.receipts.load_policy(), hosts, listed)
        (host,) = hosts.values()
        return host

    page_entry = {
        "model": "m", "scenario": "decode-128", "framework": "poot", "backend": "wgpu", "device": "Test GPU",
        "host_cpu": "Test CPU", "pair": "headline decode_tok_s", "band": NOISY_BAND, "repeats": 5,
        "reason": "slow outliers on a shared host", "follow_up": "POOT-1330",
    }

    def test_an_unlisted_noisy_cell_is_excluded_not_charted(self):
        hardware = self.derive()

        self.assertEqual(hardware["cells"], [])
        self.assertIn("noise band", hardware["excluded"][0]["reason"])

    def test_a_listed_cell_keeps_its_measured_numbers_and_is_marked_unverified_with_its_band(self):
        (cell,) = self.derive(self.page_entry)["cells"]

        self.assertFalse(cell["verified"])
        self.assertEqual(cell["band"], NOISY_BAND)
        self.assertEqual(cell["metrics"]["decode_tok_s"]["median"], statistics.median(NOISY))
        self.assertEqual(
            cell["unverified"], {"pair": "headline decode_tok_s", "failed_repeats": 0, "reason": "slow outliers on a shared host"},
        )

    def test_a_stale_entry_does_not_unlock_the_cell(self):
        hardware = self.derive({**self.page_entry, "band": 0.2})

        self.assertEqual(hardware["cells"], [])
        self.assertEqual(len(hardware["excluded"]), 1)

    def test_a_listed_cell_still_has_to_keep_the_contract(self):
        hardware = self.derive(self.page_entry, binary_sha256=lambda i: f"{i}" * 64)

        self.assertEqual(hardware["cells"], [])
        self.assertEqual(hardware["excluded"][0]["reason"], "the repeats mix binaries")


class KnownUnverifiedIndex(unittest.TestCase):
    def test_the_index_lists_every_known_unverified_cell_with_its_band(self):
        known = {known_unverified.entry_key(ENTRY): ENTRY}

        index = bench.render_index([], known)

        self.assertIn("## Known-unverified cells", index)
        self.assertIn(f"| {NOISY_BAND:.2%} | 5 | POOT-1330 |", index)

    def test_an_empty_list_adds_no_section(self):
        self.assertNotIn("Known-unverified", bench.render_index([], {}))


if __name__ == "__main__":
    unittest.main()

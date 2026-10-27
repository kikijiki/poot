"""The Performance page's data is derived from the committed evidence, and a set that fails the harness's contract is not charted."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import perf_page  # noqa: E402

ROW = {
    "framework": "poot", "model": "m", "precision": "bf16", "scenario": "decode-128", "binary_sha256": "b" * 64,
    "device": "Test GPU", "backend": "wgpu", "status": "ok", "build_sha": "a" * 40, "build_profile": "release",
    "build_features": "", "rustc_version": "rustc 1", "adapter": "Test GPU", "driver_version": "d 1",
    "checkpoint_digest": "blake3:" + "c" * 64, "compute_dtype": "f32", "profiled": False,
    "prompt_tokens": 5, "gen_tokens": 128, "ttft_ms": 10.0, "tpot_ms": 6.0, "e2e_ms": 780.0, "decode_tok_s": 166.0,
    "peak_rss_bytes": 1000,
}


def write_set(root, name, rows, date="2026-10-21", env=None):
    """One run directory per row, `<name>-r<i>`, the layout `bench record` leaves."""
    for i, row in enumerate(rows, 1):
        run_dir = root / f"{name}-r{i}"
        run_dir.mkdir()
        (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n")
        env = {"cpu": "Test CPU"} if env is None else env
        (run_dir / "env.json").write_text(json.dumps({"captured_at_utc": f"{date}T00:00:00+00:00", **env}))


def repeats(**overrides):
    return [{**ROW, "decode_tok_s": 166.0 + delta, **overrides} for delta in (-1.0, -0.5, 0.0, 0.5, 1.0)]


def write_bank(bank_dir, device=None):
    """A bank of at most one verified kernel receipt measured on `device`."""
    bank_dir = Path(bank_dir)
    entries = []
    if device:
        entries = [{"status": "verified", "case": "case-1", "backend": "wgpu", "receipt": "r.json"}]
        (bank_dir / "r.json").write_text(json.dumps({
            "reference": {"agrees": True},
            "identity": {"device": device, "precision": "f32", "dimensions": {}, "live_context": 1, "driver": "d 1"},
            "cells": {"kernel": {"repeats": [{"status": "ok", "metrics": {perf_page.bank.KERNEL_METRIC: 5.0}}] * 5}},
        }))
    (bank_dir / "receipts.json").write_text(json.dumps({"entries": entries}))
    (bank_dir / "cases.json").write_text(json.dumps({"cases": [{"id": "case-1", "regime": "decode"}]}))


def generate_classes(results_dir, kernel_device=None):
    """Every hardware class derived from `results_dir` and a bank with one receipt on `kernel_device` (or none)."""
    with tempfile.TemporaryDirectory() as bank_dir:
        write_bank(bank_dir, kernel_device)
        return perf_page.generate(results_dir, bank_dir)["hardware"]


def collect_hosts(results_dir):
    """The host entries (cells and exclusions) before a class is formed, one per (device, host CPU, driver)."""
    hosts = {}
    perf_page.collect_runs(Path(results_dir), perf_page.receipts.load_policy(), hosts, Path(results_dir) / "none.toml")
    return hosts


class PerfPageTest(unittest.TestCase):
    def classes(self, sets, kernel_device=None):
        """`sets` maps a set name to (rows, env); returns every hardware class."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, (rows, env) in sets.items():
                write_set(Path(tmp), name, rows, env=env)
            return generate_classes(tmp, kernel_device)

    def derive(self, sets):
        """The cells and exclusions of the one host the `sets` (name -> rows) ran on."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in sets.items():
                write_set(Path(tmp), name, rows)
            (host,) = collect_hosts(tmp).values()
        return host["cells"], host["excluded"]

    def test_committed_json_is_what_the_generator_derives(self):
        self.assertEqual(perf_page.OUTPUT.read_text(), perf_page.render(perf_page.generate()))

    def test_hosts_of_one_gpu_are_one_class_and_one_point_never_a_split_series(self):
        (cls,) = self.classes({
            "set-a": (repeats(), {"cpu": "CPU A"}),
            "set-b": (repeats(decode_tok_s=1.0), {"cpu": "CPU B"}),
        })

        self.assertEqual(cls["label"], "Test GPU")
        self.assertEqual([h["label"] for h in cls["hosts"]], ["CPU A", "CPU B"])
        (point,) = cls["points"]
        self.assertEqual(point["hosts"], 2)
        self.assertNotIn("host", point)

    def test_a_point_is_the_median_of_the_per_host_medians_with_their_spread(self):
        sets = {
            f"set-{i}": ([{**ROW, "decode_tok_s": value + d} for d in (-1.0, -0.5, 0.0, 0.5, 1.0)], {"cpu": f"CPU {i}"})
            for i, value in enumerate((100.0, 120.0, 400.0))
        }
        (cls,) = self.classes(sets)

        (point,) = cls["points"]
        self.assertEqual(point["metrics"]["decode_tok_s"]["median"], 120.0)
        self.assertEqual(point["spread"]["decode_tok_s"], {"min": 100.0, "max": 400.0})
        self.assertEqual((point["hosts"], point["cells"]), (3, {"verified": 3, "unverified": 0}))
        self.assertEqual(point["host_labels"], ["CPU 0", "CPU 1", "CPU 2"])

    NOISY = [90.0, 95.0, 100.0, 105.0, 110.0]

    def classes_with_noisy_host(self, listed):
        """The classes of a clean host A and a noisy host B, with the cell of B on the known-unverified list when `listed`."""
        entry = {
            "model": "m", "scenario": "decode-128", "framework": "poot", "backend": "wgpu", "device": "Test GPU",
            "host_cpu": "CPU B", "driver": None, "pair": "headline decode_tok_s", "band": 0.3454, "repeats": 5,
            "reason": "slow outliers on a shared host", "follow_up": "POOT-1330",
        }
        text = "[[cell]]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in entry.items() if v is not None) if listed else ""
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            write_set(results, "clean", repeats(), env={"cpu": "CPU A"})
            write_set(results, "noisy", [{**ROW, "decode_tok_s": v} for v in self.NOISY], env={"cpu": "CPU B"})
            (Path(tmp) / "known-unverified.toml").write_text(text)
            with tempfile.TemporaryDirectory() as bank_dir:
                write_bank(bank_dir)
                return perf_page.generate(results, bank_dir)["hardware"]

    def test_an_unverified_host_does_not_move_the_median_of_a_verified_class_point(self):
        (cls,) = self.classes_with_noisy_host(listed=True)

        (point,) = cls["points"]
        self.assertTrue(point["verified"])
        self.assertEqual(point["metrics"]["decode_tok_s"]["median"], 166.0)
        self.assertEqual((point["hosts"], point["cells"]), (1, {"verified": 1, "unverified": 1}))
        self.assertEqual(point["host_labels"], ["CPU A"])
        self.assertNotIn("unverified", point)

    def test_a_class_point_with_no_verified_cell_is_unverified_from_all_its_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            results.mkdir()
            for cpu, base in (("CPU A", 100.0), ("CPU B", 200.0)):
                write_set(results, cpu.replace(" ", ""), [{**ROW, "decode_tok_s": base + d} for d in (-10.0, -5.0, 0.0, 5.0, 10.0)], env={"cpu": cpu})
            entries = "".join(
                f'[[cell]]\nmodel = "m"\nscenario = "decode-128"\nframework = "poot"\nbackend = "wgpu"\ndevice = "Test GPU"\n'
                f'host_cpu = "{cpu}"\npair = "headline decode_tok_s"\nband = {band}\nrepeats = 5\nreason = "noisy"\nfollow_up = "POOT-1330"\n\n'
                for cpu, band in (("CPU A", 0.3454), ("CPU B", 0.1727))
            )
            (Path(tmp) / "known-unverified.toml").write_text(entries)
            with tempfile.TemporaryDirectory() as bank_dir:
                write_bank(bank_dir)
                (cls,) = perf_page.generate(results, bank_dir)["hardware"]

        (point,) = cls["points"]
        self.assertFalse(point["verified"])
        self.assertEqual(point["metrics"]["decode_tok_s"]["median"], 150.0)
        self.assertEqual((point["hosts"], point["cells"]), (2, {"verified": 0, "unverified": 2}))
        self.assertEqual(point["unverified"]["reason"], "noisy")

    def test_a_metric_one_host_lacks_comes_from_the_hosts_that_have_it_and_is_absent_when_none_has_it(self):
        sets = {
            "set-a": ([{**ROW, "peak_vram_bytes": 4096}] * 5, {"cpu": "CPU A"}),
            "set-b": ([{**ROW}] * 5, {"cpu": "CPU B"}),
        }
        (cls,) = self.classes(sets)

        (point,) = cls["points"]
        self.assertEqual(point["metrics"]["peak_vram_bytes"]["median"], 4096)
        self.assertEqual(point["spread"]["peak_vram_bytes"], {"min": 4096, "max": 4096})
        self.assertNotIn("energy_j", point["metrics"])
        self.assertNotIn("energy_j", point["spread"])

    def test_the_workstation_and_server_editions_of_a_chip_are_one_class(self):
        (cls,) = self.classes({
            "set-a": (repeats(device="NVIDIA RTX PRO 6000 Blackwell Workstation Edition"), {"cpu": "CPU A"}),
            "set-b": (repeats(device="NVIDIA RTX PRO 6000 Blackwell Server Edition"), {"cpu": "CPU B"}),
        })

        self.assertEqual((cls["label"], cls["id"]), ("NVIDIA RTX PRO 6000 Blackwell", "nvidia-rtx-pro-6000-blackwell"))
        self.assertEqual([p["hosts"] for p in cls["points"]], [2])

    def test_max_q_is_a_different_power_class_and_stays_its_own_class(self):
        classes = self.classes({
            "set-a": (repeats(device="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"), {"cpu": "CPU A"}),
            "set-b": (repeats(device="NVIDIA RTX PRO 6000 Blackwell Workstation Edition"), {"cpu": "CPU B"}),
        })

        self.assertEqual([c["label"] for c in classes], ["NVIDIA RTX PRO 6000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell Max-Q"])

    def test_a_superseded_set_stays_a_row_of_its_own_with_one_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_set(Path(tmp), "old", repeats(), date="2026-10-19", env={"cpu": "CPU A"})
            write_set(Path(tmp), "new", repeats(), date="2026-10-21", env={"cpu": "CPU A"})
            write_set(Path(tmp), "other", repeats(), date="2026-10-20", env={"cpu": "CPU B"})
            (cls,) = generate_classes(tmp)

        self.assertEqual({p["set"]: (p["latest"], p["hosts"]) for p in cls["points"]}, {"old": (False, 1), "other": (False, 1), "new": (True, 1)})

    def test_a_set_another_host_has_since_remeasured_feeds_no_headline_value_of_any_metric(self):
        gib = 2**30
        old = [{**ROW, "peak_rss_bytes": 74 * gib, "decode_tok_s": 10.0 + d / 100} for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        new = [{**ROW, "peak_rss_bytes": 6 * gib, "decode_tok_s": 40.0 + d / 100} for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
        with tempfile.TemporaryDirectory() as tmp:
            write_set(Path(tmp), "old", old, date="2026-10-23", env={"cpu": "CPU A"})
            write_set(Path(tmp), "new", new, date="2026-10-26", env={"cpu": "CPU B"})
            (cls,) = generate_classes(tmp)

        headline = next(p for p in cls["points"] if p["latest"])
        self.assertEqual((headline["set"], headline["hosts"]), ("new", 1))
        self.assertEqual(headline["metrics"]["peak_rss_bytes"]["median"], 6 * gib)
        self.assertEqual(headline["metrics"]["decode_tok_s"]["median"], 40.0)
        self.assertEqual([p["set"] for p in cls["points"] if not p["latest"]], ["old"])

    def test_two_drivers_on_one_cpu_are_two_hosts_labelled_by_driver(self):
        (cls,) = self.classes({
            "set-a": (repeats(), {"cpu": "CPU A", "gpu_driver": "535"}),
            "set-b": (repeats(), {"cpu": "CPU A", "gpu_driver": "550"}),
        })

        self.assertEqual([h["label"] for h in cls["hosts"]], ["CPU A, driver 535", "CPU A, driver 550"])
        self.assertEqual([h["driver"] for h in cls["hosts"]], ["535", "550"])
        self.assertEqual([p["hosts"] for p in cls["points"]], [2])

    def test_a_run_with_no_recorded_host_is_its_own_host_not_merged_into_a_known_one(self):
        (cls,) = self.classes({"set-a": (repeats(), {"cpu": "CPU A"}), "set-b": (repeats(), {})})

        self.assertEqual(sorted(h["label"] for h in cls["hosts"]), ["CPU A", "unknown host"])

    def test_the_same_host_in_two_sets_stays_one_host(self):
        (cls,) = self.classes({"set-a": (repeats(), {"cpu": "CPU A"}), "set-b": (repeats(), {"cpu": "CPU A"})})

        self.assertEqual(len(cls["hosts"]), 1)
        self.assertEqual({p["hosts"] for p in cls["points"]}, {1})

    def test_every_host_kernel_of_a_host_is_shown(self):
        (cls,) = self.classes({
            "set-a": (repeats(), {"cpu": "CPU A", "host_kernel": "7.1"}),
            "set-b": (repeats(), {"cpu": "CPU A", "host_kernel": "7.2"}),
        })

        self.assertEqual(cls["hosts"][0]["kernel"], "7.1, 7.2")

    def test_kernel_receipts_belong_to_their_device_whatever_the_hosts(self):
        (cls,) = self.classes({"set-a": (repeats(), {"cpu": "CPU A"}), "set-b": (repeats(), {"cpu": "CPU B"})}, kernel_device="Test GPU")

        self.assertEqual([k["case"] for k in cls["kernels"]], ["case-1"])

    def test_a_device_with_receipts_and_no_runs_is_not_a_class(self):
        (cls,) = self.classes({"set-a": (repeats(), {"cpu": "CPU A"})}, kernel_device="Other GPU")

        self.assertEqual(cls["label"], "Test GPU")
        self.assertEqual(cls["kernels"], [])

    def test_a_device_with_only_failing_sets_is_not_a_class(self):
        self.assertEqual(self.classes({"set-a": (repeats()[:3], {"cpu": "CPU A"})}), [])

    def test_repeats_split_across_hosts_inside_one_set_are_not_pooled(self):
        # One set whose run directories were recorded on two hosts: 3 + 2 repeats are two short cells, not one
        # band of five.
        with tempfile.TemporaryDirectory() as tmp:
            for i, (row, cpu) in enumerate(zip(repeats(), ["CPU A"] * 3 + ["CPU B"] * 2), 1):
                run_dir = Path(tmp) / f"set-r{i}"
                run_dir.mkdir()
                (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n")
                (run_dir / "env.json").write_text(json.dumps({"captured_at_utc": "2026-10-21T00:00:00+00:00", "cpu": cpu}))
            hosts = list(collect_hosts(tmp).values())

        self.assertEqual([len(h["excluded"]) for h in hosts], [1, 1])
        self.assertEqual([h["cells"] for h in hosts], [[], []])
        self.assertEqual([h["excluded"][0]["reason"] for h in hosts], ["3 repeats, the policy needs 5", "2 repeats, the policy needs 5"])

    def test_a_clean_set_is_a_verified_cell_with_the_median_range_and_band(self):
        cells, excluded = self.derive({"set": repeats()})
        self.assertEqual(excluded, [])
        (cell,) = cells
        self.assertEqual(cell["metrics"]["decode_tok_s"], {"median": 166.0, "min": 165.0, "max": 167.0, "n": 5})
        self.assertTrue(cell["verified"])
        self.assertNotIn("unverified", cell)
        self.assertGreater(cell["band"], 0)
        self.assertNotIn("prefill_tok_s", cell["metrics"])

    def test_a_noisy_set_is_excluded(self):
        # Two slow repeats of five: a slow mode, not an outlier, so the robust band is wide.
        rows = repeats()
        rows[0]["decode_tok_s"] = 100.0
        rows[1]["decode_tok_s"] = 100.0
        cells, excluded = self.derive({"set": rows})
        self.assertEqual(cells, [])
        self.assertIn("noise band", excluded[0]["reason"])

    def test_a_dirty_build_is_excluded(self):
        cells, excluded = self.derive({"set": repeats(build_sha="a" * 40 + "-dirty")})
        self.assertEqual(cells, [])
        self.assertIn("-dirty", excluded[0]["reason"])

    def test_too_few_repeats_are_excluded(self):
        cells, excluded = self.derive({"set": repeats()[:3]})
        self.assertEqual(cells, [])
        self.assertIn("3 repeats", excluded[0]["reason"])

    def test_repeats_from_two_binaries_are_excluded(self):
        rows = repeats()
        rows[0]["binary_sha256"] = "d" * 64
        _, excluded = self.derive({"set": rows})
        self.assertEqual(excluded[0]["reason"], "the repeats mix binaries")

    def test_only_the_newest_set_of_a_cell_is_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_set(Path(tmp), "old", repeats(), date="2026-10-19")
            write_set(Path(tmp), "new", repeats(), date="2026-10-21")
            (cls,) = generate_classes(tmp)
        self.assertEqual({p["set"]: p["latest"] for p in cls["points"]}, {"old": False, "new": True})

    def test_prefill_rate_is_derived_only_for_a_long_prompt(self):
        cells, _ = self.derive({"set": repeats(scenario="prefill-1k", prompt_tokens=1000, ttft_ms=500.0)})
        self.assertEqual(cells[0]["metrics"]["prefill_tok_s"]["median"], 2000.0)

    def curve_rows(self, **overrides):
        def row(delta):
            point = {
                "isl": 512, "prompt_tokens": 512, "decode_tok_s": 100.0 + delta,
                "ttft_ms": {"p50": 256.0, "min": 250.0, "max": 260.0}, "tpot_ms": {"p50": 10.0, "min": 9.0, "max": 11.0},
            }
            return {**ROW, "scenario": "decode-curve", "curve": [point], "decode_tok_s": 100.0 + delta, **overrides}

        return [row(d) for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]

    def test_curve_points_carry_statistics_over_the_repeats(self):
        cells, _ = self.derive({"set": copy.deepcopy(self.curve_rows())})
        (point,) = cells[0]["curve"]
        self.assertEqual(point["decode_tok_s"], {"median": 100.0, "min": 99.0, "max": 101.0, "n": 5})
        self.assertEqual(point["prefill_tok_s"]["median"], 2000.0)

    def test_a_curve_cell_states_its_prefill_rate_at_the_longest_context(self):
        cells, _ = self.derive({"set": self.curve_rows()})
        self.assertEqual(cells[0]["metrics"]["prefill_tok_s"]["median"], 2000.0)
        self.assertEqual(cells[0]["prefill_isl"], 512)

    def points_of(self, sets):
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in sets.items():
                write_set(Path(tmp), name, rows)
            (cls,) = generate_classes(tmp)
        return cls["points"]

    def test_a_single_run_is_one_point_at_its_prompt_length_with_no_scenario(self):
        (point,) = self.points_of({"set": repeats(prompt_tokens=6)})
        self.assertEqual(point["isl"], 6)
        self.assertEqual(point["metrics"]["decode_tok_s"]["median"], 166.0)
        self.assertNotIn("scenario", point)
        self.assertTrue(point["verified"])

    def test_a_sweep_is_one_point_per_context_length_and_a_single_run_joins_the_same_axis(self):
        points = self.points_of({"sweep": self.curve_rows(), "single": repeats(scenario="prefill-1k", prompt_tokens=1022, ttft_ms=500.0)})
        self.assertEqual(sorted((p["set"], p["isl"]) for p in points), [("single", 1022), ("sweep", 512)])

    def test_a_sweeps_memory_sits_on_its_longest_context_and_says_it_is_a_peak_over_the_sweep(self):
        def row(delta):
            points = [
                {"isl": isl, "prompt_tokens": isl, "decode_tok_s": 100.0 + delta,
                 "ttft_ms": {"p50": 10.0, "min": 9.0, "max": 11.0}, "tpot_ms": {"p50": 10.0, "min": 9.0, "max": 11.0}}
                for isl in (128, 512)
            ]
            return {**ROW, "scenario": "decode-curve", "curve": points, "decode_tok_s": 100.0 + delta, "peak_vram_bytes": 4096}

        points = self.points_of({"set": [row(d) for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]})
        by_isl = {p["isl"]: p for p in points}
        self.assertNotIn("peak_vram_bytes", by_isl[128]["metrics"])
        self.assertEqual(by_isl[512]["metrics"]["peak_vram_bytes"]["median"], 4096)
        self.assertTrue(by_isl[512]["over_sweep"])
        self.assertNotIn("over_sweep", by_isl[128])

    def test_memory_is_the_inference_peak_with_the_load_peak_beside_it(self):
        gib = 2**30
        # A 26 GiB load in the first three seconds, then 5-6 GiB at inference; the lifetime peak is the load's.
        rss = [4, 26, 26, 5, 5, 6, 5, 5, 1]
        series = [[t, 0, 0, 0, int(r * gib), 0.0] for t, r in enumerate(rss, 1)]
        rows = repeats(peak_rss_bytes=26 * gib, util_series=series, preparation={"open_ms": 3000})
        (cell,) = self.derive({"set": rows})[0]
        self.assertEqual(cell["memory_scope"], "inference")
        self.assertEqual(cell["metrics"]["peak_rss_bytes"]["median"], 6 * gib)
        self.assertEqual(cell["metrics"]["load_rss_bytes"]["median"], 26 * gib)

    def test_one_repeat_without_a_boundary_keeps_the_lifetime_peak_of_the_whole_cell(self):
        gib = 2**30
        rss = [4, 26, 26, 5, 5, 6, 5, 5, 1]
        series = [[t, 0, 0, 0, int(r * gib), 0.0] for t, r in enumerate(rss, 1)]
        rows = repeats(peak_rss_bytes=26 * gib, util_series=series, preparation={"open_ms": 3000})
        rows[2] = {**rows[2], "util_series": None}
        (cell,) = self.derive({"set": rows})[0]
        self.assertEqual(cell["memory_scope"], "includes_load")
        self.assertEqual(cell["metrics"]["peak_rss_bytes"]["median"], 26 * gib)
        self.assertNotIn("load_rss_bytes", cell["metrics"])

    def test_a_short_generation_run_feeds_no_decode_series(self):
        # prefill-1k generates 32 tokens: its decode rate is not comparable with the 128-token series, its prefill is.
        short = self.points_of({"set": repeats(scenario="prefill-1k", prompt_tokens=1022, gen_tokens=32, ttft_ms=500.0, tpot_ms=9.0, e2e_ms=800.0)})
        (point,) = short
        for metric in ("decode_tok_s", "tpot_ms", "e2e_ms"):
            self.assertNotIn(metric, point["metrics"])
        self.assertEqual(point["metrics"]["prefill_tok_s"]["median"], 2044.0)
        self.assertIn("ttft_ms", point["metrics"])
        self.assertIn("peak_rss_bytes", point["metrics"])

    def test_a_128_token_run_and_a_sweep_feed_the_decode_series(self):
        (single,) = self.points_of({"set": repeats(prompt_tokens=6, gen_tokens=128)})
        self.assertEqual(single["metrics"]["decode_tok_s"]["median"], 166.0)
        sweep = self.points_of({"set": [{**row, "osl": 128} for row in self.curve_rows()]})
        self.assertIn("decode_tok_s", sweep[0]["metrics"])
        self.assertIn("tpot_ms", sweep[0]["metrics"])

    def test_a_prefill_rate_is_stated_from_128_input_tokens(self):
        self.assertEqual(perf_page.prefill_tok_s({"prompt_tokens": 128, "ttft_ms": 64.0}), 2000.0)
        self.assertIsNone(perf_page.prefill_tok_s({"prompt_tokens": 6, "ttft_ms": 5.0}))

    def test_a_series_names_the_framework_and_its_backend(self):
        cells, _ = self.derive({"set": repeats(backend="ptx")})
        self.assertEqual((cells[0]["series"], cells[0]["backend"]), ("poot (PTX)", "ptx"))

    def test_a_baseline_row_that_records_its_backend_keeps_it(self):
        row = {"framework": "llamacpp", "device": "NVIDIA GeForce RTX 3090", "backend": "rocm"}
        self.assertEqual(perf_page.row_backend(row), "rocm")
        self.assertEqual(perf_page.series_label("llamacpp", "rocm"), "llama.cpp (ROCm)")

    def test_a_baseline_row_without_a_backend_is_mapped_by_framework_and_vendor(self):
        mapped = lambda framework, device: perf_page.row_backend({"framework": framework, "device": device})
        self.assertEqual(mapped("llamacpp", "NVIDIA A100-SXM4-80GB"), "cuda")
        self.assertEqual(mapped("vllm", "NVIDIA H200"), "cuda")
        self.assertEqual(mapped("llamacpp", "AMD Radeon 8060S Graphics"), "vulkan")

    def test_a_recorded_unknown_or_empty_backend_is_an_error_not_a_series(self):
        for recorded in ("unknown", ""):
            with self.assertRaisesRegex(ValueError, "records the backend"):
                perf_page.row_backend({"framework": "llamacpp", "device": "NVIDIA H200", "backend": recorded})

    def test_a_baseline_row_with_no_backend_and_no_mapping_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "vllm on 'AMD Radeon"):
            perf_page.row_backend({"framework": "vllm", "device": "AMD Radeon 8060S Graphics"})


if __name__ == "__main__":
    unittest.main()


class PeakVramIsTheRunsOwnTests(unittest.TestCase):
    def test_the_device_use_before_the_run_is_subtracted(self):
        import perf_page
        row = {"peak_vram_bytes": 4000, "vram_baseline_bytes": 1500}
        self.assertEqual(perf_page.row_value(row, "peak_vram_bytes"), 2500)

    def test_without_a_baseline_the_peak_is_kept_and_it_never_goes_negative(self):
        import perf_page
        self.assertEqual(perf_page.row_value({"peak_vram_bytes": 4000}, "peak_vram_bytes"), 4000)
        self.assertEqual(perf_page.row_value({"peak_vram_bytes": 100, "vram_baseline_bytes": 500}, "peak_vram_bytes"), 0)

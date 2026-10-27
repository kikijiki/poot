"""Unit tests of compare.py on synthetic llama.cpp logits files and poot-score reports. Run:
python3 -I scripts/flash-next-correctness/test_compare.py (from the repository root)."""

import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare  # noqa: E402

N_CTX, VOCAB = 8, 12  # scored positions 4..6


def distribution(seed, shift=0.0):
    """A deterministic log-softmax over VOCAB tokens; `shift` perturbs token 3 (the mutation)."""
    logits = [math.sin(seed * 7.3 + t * 1.7) * 3.0 for t in range(VOCAB)]
    logits[3] += shift
    top = max(logits)
    log_sum = top + math.log(sum(math.exp(x - top) for x in logits))
    return [x - log_sum for x in logits]


def logits_file(shift=0.0):
    out = bytearray(compare.MAGIC + struct.pack("<iii", N_CTX, VOCAB, 1))
    out += struct.pack("<%di" % N_CTX, *range(100, 100 + N_CTX))
    for position in range(N_CTX // 2, N_CTX - 1):
        lp = distribution(position, shift)
        floor = min(lp)
        scale = (max(lp) - floor) / 65535.0
        codes = [max(1, round((x - floor) / scale)) for x in lp]
        record = struct.pack("<ff", scale, floor) + struct.pack("<%dH" % VOCAB, *codes)
        out += record + b"\0" * (2 * (2 * ((VOCAB + 1) // 2) + 4) - len(record))
    return bytes(out)


def poot_report(shift=0.0, top=5):
    positions = []
    for position in range(N_CTX // 2, N_CTX - 1):
        lp = distribution(position, shift)
        best = sorted(range(VOCAB), key=lambda t: -lp[t])[:top]
        positions.append({"pos": position, "top": [[t, lp[t]] for t in best]})
    return {"positions": positions, "greedy": {"ids": [1, 2, 3], "text": "abc"}, "adapter": "test"}


class Compare(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def write(self, name, data, mode="wb"):
        path = os.path.join(self.dir.name, name)
        with open(path, mode) as handle:
            handle.write(data)
        return path

    def run_compare(self, cuda_shift, poot_shift):
        cpu = self.write("cpu.bin", logits_file())
        cuda = self.write("cuda.bin", logits_file(cuda_shift))
        poot = self.write("poot.json", json.dumps(poot_report(poot_shift)), "w")
        with contextlib.redirect_stdout(io.StringIO()):
            return compare.main(["--cpu", cpu, "--cuda", cuda, "--poot", poot, "--top", "5", "--out",
                                 os.path.join(self.dir.name, "report.json")])

    def test_reading_back_the_window(self):
        reference = compare.Reference(logits_file())
        self.assertEqual(reference.window(0), list(range(100, 108)))
        self.assertEqual(reference.scored, 3)
        top = reference.top(0, 0, 3)
        expected = distribution(4)
        self.assertEqual(list(top)[0], max(range(VOCAB), key=lambda t: expected[t]))
        self.assertAlmostEqual(top[list(top)[0]], max(expected), places=3)

    def test_an_exact_poot_passes(self):
        self.assertEqual(self.run_compare(0.0, 0.0), 0)

    def test_poot_within_the_reference_noise_passes(self):
        # CPU vs CUDA differ by 0.05 in one logit; poot by the same amount.
        self.assertEqual(self.run_compare(0.05, 0.05), 0)

    def test_a_perturbed_poot_fails_the_kl_and_abs_gates(self):
        self.assertEqual(self.run_compare(0.0, 2.5), 1)
        with open(os.path.join(self.dir.name, "report.json")) as handle:
            report = json.load(handle)
        failed = {f["metric"] for f in report["failures"]}
        self.assertIn("max_abs", failed)
        self.assertIn("mean_kl", failed)

    def test_a_missing_position_is_malformed_not_a_pass(self):
        poot = poot_report()
        poot["positions"].pop()
        cpu = self.write("cpu.bin", logits_file())
        path = self.write("poot.json", json.dumps(poot), "w")
        with contextlib.redirect_stderr(io.StringIO()):
            status = compare.main(["--cpu", cpu, "--cuda", cpu, "--poot", path, "--top", "5"])
        self.assertEqual(status, 2)

    def test_a_nan_never_passes(self):
        limits = {"mean_kl": 1.0, "max_kl": 1.0, "max_abs": 1.0, "top1": 0.5}
        subject = {"mean_kl": float("nan"), "max_kl": 0.0, "max_abs": 0.0, "top1": 1.0, "positions": 3, "dropped": 0}
        self.assertEqual([f[0] for f in compare.judge(subject, limits)], ["mean_kl"])

    def test_top1_gate_resolves_no_finer_than_two_positions(self):
        noise = {"mean_kl": 0.0, "max_kl": 0.0, "max_abs": 0.0, "top1": 1.0, "positions": 23, "dropped": 0}
        floors = {"mean_kl": 1.0, "max_kl": 1.0, "max_abs": 1.0}
        limits = compare.tolerance(noise, 3.0, 0.02, floors)
        one_flip = {"mean_kl": 0.0, "max_kl": 0.0, "max_abs": 0.0, "top1": 22 / 23, "positions": 23, "dropped": 0}
        three_flips = dict(one_flip, top1=20 / 23)
        self.assertEqual(compare.judge(one_flip, limits), [])
        self.assertEqual([f[0] for f in compare.judge(three_flips, limits)], ["top1"])

    def test_each_gate_fails_alone(self):
        limits = {"mean_kl": 1.0, "max_kl": 2.0, "max_abs": 3.0, "top1": 0.5}
        good = {"mean_kl": 0.0, "max_kl": 0.0, "max_abs": 0.0, "top1": 1.0, "positions": 10, "dropped": 0}
        self.assertEqual(compare.judge(good, limits), [])
        for metric, value in (("mean_kl", 1.5), ("max_kl", 2.5), ("max_abs", 3.5), ("top1", 0.4), ("dropped", 3)):
            subject = dict(good, **{metric: value})
            if metric == "dropped":
                subject["positions"] = 7
            self.assertEqual([f[0] for f in compare.judge(subject, limits)], [metric], metric)

    def test_first_divergence(self):
        self.assertIsNone(compare.first_divergence([1, 2], [1, 2]))
        self.assertEqual(compare.first_divergence([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(compare.first_divergence([1, 2], [1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()

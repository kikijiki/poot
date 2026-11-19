"""The website's headline numbers are derived from the committed data, and a changed claim fails (POOT-1630)."""

import copy
import json
import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import headline_claims as hc  # noqa: E402

PERFORMANCE = json.loads(hc.PERFORMANCE.read_text())
LIMITATIONS = json.loads(hc.STATUS.read_text())["limitations"]
GAPS = hc.load_gaps()


def failures(limitations=LIMITATIONS, performance=PERFORMANCE, gaps=GAPS):
    return hc.run(performance, limitations, gaps)[1]


def stated(old, new):
    out = [line.replace(old, new) for line in LIMITATIONS]
    assert out != LIMITATIONS, f"{old!r} is not in the committed limitations"
    return out


class CommittedTextTests(unittest.TestCase):
    def test_the_committed_text_reproduces_except_the_declared_gap(self):
        checks, fails = hc.run(PERFORMANCE, LIMITATIONS, GAPS)
        self.assertEqual(fails, [])
        self.assertEqual([c.claim for c in checks if not c.reproduces], ["first-token-max"])

    def test_the_37_cells_are_the_same_backend_pairs_only(self):
        cells = hc.cells(PERFORMANCE)
        self.assertEqual(len(cells), 37)
        self.assertEqual({c.pair for c in cells}, {"vulkan", "rocm", "cuda"})


class ChangedClaimTests(unittest.TestCase):
    def assert_red(self, limitations, claim):
        fails = failures(limitations)
        self.assertTrue(any(f.startswith(claim) for f in fails), fails)

    def test_a_changed_decode_bound_fails(self):
        self.assert_red(stated("from 1.04x", "from 1.05x"), "decode-low")
        self.assert_red(stated("to 3.6x slower", "to 3.7x slower"), "decode-high")

    def test_a_changed_first_token_count_fails(self):
        self.assert_red(stated("in 26 of 37", "in 25 of 37"), "first-token-behind-cells")
        self.assert_red(stated("in 26 of 37", "in 26 of 36"), "first-token-cells")

    def test_a_changed_gap_number_fails_even_though_it_was_declared(self):
        fails = failures(stated("up to 3.5x slower", "up to 4.0x slower"))
        self.assertTrue(any("review the entry" in f for f in fails), fails)

    def test_the_gap_fails_when_the_prose_is_fixed_to_the_data(self):
        fails = failures(stated("up to 3.5x slower", "up to 4.5x slower"))
        self.assertTrue(any("delete the entry" in f for f in fails), fails)

    def test_a_changed_memory_or_grid_number_fails(self):
        self.assert_red(stated("up to 1.5x", "up to 1.6x"), "memory-max")
        self.assert_red(stated("two machines", "three machines"), "grid-machines")
        self.assert_red(stated("five models", "six models"), "grid-models")

    def test_a_new_number_without_a_derivation_fails(self):
        fails = failures(LIMITATIONS + ["Decodes at 40 tok/s."])
        self.assertTrue(any("no claim derives" in f for f in fails), fails)

    def test_a_rewritten_claim_line_fails(self):
        fails = failures(stated("Behind on first token in", "Behind on prefill in"))
        self.assertTrue(any("exactly one must" in f for f in fails), fails)


class ChangedDataTests(unittest.TestCase):
    def test_a_slower_llamacpp_decode_makes_the_decode_claims_fail(self):
        data = copy.deepcopy(PERFORMANCE)
        for h in data["hardware"]:
            for p in h["points"]:
                if p["framework"] == "llamacpp" and p["latest"] and "decode_tok_s" in p["metrics"]:
                    p["metrics"]["decode_tok_s"]["median"] *= 0.5
        self.assertTrue(any(f.startswith("decode-") for f in failures(performance=data)))

    def test_a_pair_never_crosses_backends(self):
        data = copy.deepcopy(PERFORMANCE)
        for h in data["hardware"]:
            h["points"] = [p for p in h["points"] if not (p["framework"] == "llamacpp" and p["backend"] == "vulkan")]
        self.assertNotIn("vulkan", {c.pair for c in hc.cells(data)})


if __name__ == "__main__":
    unittest.main()

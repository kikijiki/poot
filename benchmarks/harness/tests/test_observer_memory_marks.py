"""The observer cuts its memory samples at the marks the runner prints, on a real process tree."""

import sys
import textwrap
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import observer  # noqa: E402

MB = 1024 * 1024

# A runner with three distinct phases, each holding anonymous memory of a different size: 400 MB while it loads, a
# 700 MB spike while it warms up (a compile, an allocator filling) and 150 MB at steady state. The marks are printed
# on the runner's own clock, as the real runners do.
RUNNER = textwrap.dedent(
    """
    import sys, time

    def mark(event):
        print(f"POOT_BENCH_MARK {event} {time.time():.6f}", file=sys.stderr, flush=True)

    def hold(megabytes, seconds):
        block = bytearray(megabytes * 1024 * 1024)
        block[::4096] = b"x" * len(block[::4096])
        time.sleep(seconds)
        del block

    SKIP_WARMUP_MARK = len(sys.argv) > 1 and sys.argv[1] == "no-warmup-mark"
    hold(400, 0.6)
    mark("load_end")
    hold(700, 0.6)
    if not SKIP_WARMUP_MARK:
        mark("warmup_end")
    hold(150, 0.6)
    print("{}")
    """
)


def observe(*args):
    return observer.run_observed([sys.executable, "-c", RUNNER, *args], interval_s=0.02)


class MemoryMarksTest(unittest.TestCase):
    def test_load_warmup_and_inference_peaks_are_cut_at_the_runners_marks(self):
        got = observe()
        self.assertEqual(got["memory_mark_errors"], [])
        phases = got["memory_phases"]
        self.assertIsNotNone(phases, "both marks were printed")
        # Each part holds only its own allocation (plus the interpreter's own ~20 MB), never the 700 MB warmup spike.
        self.assertGreater(phases["load"]["rss_anon_bytes"], 400 * MB)
        self.assertLess(phases["load"]["rss_anon_bytes"], 600 * MB)
        self.assertGreater(phases["inference"]["rss_anon_bytes"], 150 * MB)
        self.assertLess(phases["inference"]["rss_anon_bytes"], 350 * MB)
        # The lifetime peak is the warmup's, and it is anonymous memory: the whole-run figures say nothing of the parts.
        self.assertGreater(got["peak_rss_anon_bytes"], 700 * MB)
        self.assertGreaterEqual(got["peak_rss_bytes"], got["peak_rss_anon_bytes"])

    def test_a_run_that_never_marks_the_end_of_its_warmup_has_no_exact_split(self):
        got = observe("no-warmup-mark")
        self.assertIsNone(got["memory_phases"])
        self.assertEqual(got["memory_mark_errors"], [])

    def test_a_malformed_mark_is_reported_and_leaves_no_split(self):
        script = 'import sys; print("POOT_BENCH_MARK load-end 1.0", file=sys.stderr, flush=True); print("{}")'
        got = observer.run_observed([sys.executable, "-c", script], interval_s=0.02)
        self.assertIsNone(got["memory_phases"])
        self.assertEqual(len(got["memory_mark_errors"]), 1)
        self.assertIn("load-end", got["memory_mark_errors"][0])

    def test_a_repeated_mark_does_not_move_the_phase(self):
        # A runner that restarts its engine per point prints the marks again; the first of each bounds the part.
        script = textwrap.dedent(
            """
            import sys
            for event, unix_s in (("load_end", 100.0), ("warmup_end", 200.0), ("load_end", 300.0), ("warmup_end", 400.0)):
                print(f"POOT_BENCH_MARK {event} {unix_s}", file=sys.stderr, flush=True)
            print("{}")
            """
        )
        got = observer.run_observed([sys.executable, "-c", script], interval_s=0.02)
        self.assertEqual(got["memory_marks"], {"load_end": 100.0, "warmup_end": 200.0})
        # The marks are in 1970: no tick falls in either part, so there is no split to report.
        self.assertIsNone(got["memory_phases"])


if __name__ == "__main__":
    unittest.main()

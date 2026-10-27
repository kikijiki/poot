"""The vLLM runner measures a fixed generated length: the sampling parameters it builds force it, and its stream
timing yields exactly osl - 1 gaps however vLLM chunks the tokens (POOT-1328)."""

import importlib.util
import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

STREAM = HARNESS_DIR.parent / "runners" / "vllm" / "stream_itl.py"
spec = importlib.util.spec_from_file_location("vllm_stream_itl", STREAM)
stream_itl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream_itl)


def streamed(chunks, tokens):
    """Timing of a stream whose chunks are (seconds, total tokens so far), started at 0."""
    stream = stream_itl.TokenStream(0.0)
    for now, total in chunks:
        stream.observe(now, total)
    return stream.finish(chunks[-1][0], tokens)


class FixedLengthTests(unittest.TestCase):
    def test_the_sampling_parameters_cannot_stop_early(self):
        kwargs = stream_itl.fixed_length_kwargs(127)
        self.assertIs(kwargs["ignore_eos"], True)
        self.assertEqual((kwargs["min_tokens"], kwargs["max_tokens"]), (127, 127))
        self.assertEqual(kwargs["temperature"], 0.0)

    def test_a_first_chunk_with_two_tokens_still_yields_osl_minus_one_gaps(self):
        ttft, e2e, itls = streamed([(0.010, 2), (0.020, 3), (0.040, 5)], 5)
        self.assertEqual(len(itls), 4)
        for got, want in zip(itls, [0.0, 10.0, 10.0, 10.0]):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(ttft, 10.0)
        self.assertAlmostEqual(e2e, 40.0)

    def test_one_token_per_chunk_is_unchanged(self):
        _, _, itls = streamed([(0.010, 1), (0.015, 2), (0.025, 3)], 3)
        self.assertEqual(len(itls), 2)
        for got, want in zip(itls, [5.0, 10.0]):
            self.assertAlmostEqual(got, want)

    def test_a_stream_that_ends_early_names_the_framework_and_the_token(self):
        with self.assertRaisesRegex(stream_itl.StoppedEarly, "vllm generated 126, expected 127 tokens; the stream ended after token 126"):
            streamed([(0.010, 1), (0.020, 126)], 127)


if __name__ == "__main__":
    unittest.main()

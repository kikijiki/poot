"""Unit tests of shards.sh. Run: python3 -I scripts/flash-next-correctness/test_shards.py (from the repository root)."""

import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def first_shard(directory):
    return subprocess.run(
        ["bash", "-c", 'source "$1" && first_shard "$2"', "_", os.path.join(HERE, "shards.sh"), directory],
        capture_output=True,
        text=True,
    )


def touch_all(directory, names):
    for name in names:
        open(os.path.join(directory, name), "w").close()


class FirstShard(unittest.TestCase):
    def test_a_four_shard_set_is_found_by_its_first_shard(self):
        with tempfile.TemporaryDirectory() as d:
            touch_all(d, [f"Qwen3.8-Flash-Next-UD-Q2_K_XL-0000{i}-of-00004.gguf" for i in range(1, 5)])
            r = first_shard(d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), os.path.join(d, "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf"))

    def test_a_three_shard_set_is_found_by_its_first_shard(self):
        # The layout of the manifest's pinned revision (766911a6b736).
        with tempfile.TemporaryDirectory() as d:
            touch_all(d, [f"Qwen3.8-Flash-Next-UD-Q2_K_XL-0000{i}-of-00003.gguf" for i in range(1, 4)])
            r = first_shard(d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), os.path.join(d, "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"))

    def test_a_directory_without_a_first_shard_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            touch_all(d, ["Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00004.gguf"])
            r = first_shard(d)
            self.assertEqual(r.returncode, 1)
            self.assertIn("expected one", r.stderr)

    def test_two_sets_in_one_directory_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as d:
            touch_all(d, ["a-00001-of-00003.gguf", "a-00001-of-00004.gguf"])
            r = first_shard(d)
            self.assertEqual(r.returncode, 1)


if __name__ == "__main__":
    unittest.main()

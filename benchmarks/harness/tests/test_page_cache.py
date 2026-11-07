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
import page_cache  # noqa: E402
import performance_receipt as receipts  # noqa: E402
from runner_fixtures import run_fixture_runner, valid_single  # noqa: E402

GIB = 1 << 30


def write_file(path, size):
    Path(path).write_bytes(b"\0" * size)
    return Path(path)


def install_runner(directory, name):
    path = Path(directory) / name
    path.write_text(f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(valid_single())}'\n", encoding="ascii")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def run_local_cell(models_dir, binary):
    """One poot cell over a local model whose weights are the single file `weights.gguf`."""
    return bench.run_cell(
        model={"id": "fixture", "match_precision": "bf16", "local_dir": "m", "weights_file": "weights.gguf"},
        scen={"id": "single", "gen_tokens": 4},
        fw="poot",
        fw_cfg={"support": True, "precision": "bf16"},
        runners={"poot": {"bin": str(binary), "cmd": ["{bin}"]}},
        models_dir=str(models_dir),
        suite={},
        interval_ms=50,
        device=None,
    )


class WeightFilesTests(unittest.TestCase):
    def test_a_split_set_named_by_its_first_shard_is_every_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            shards = [write_file(Path(tmp) / f"m-0000{i}-of-00003.gguf", 8) for i in (1, 2, 3)]
            self.assertEqual(page_cache.weight_files(shards[0]), shards)

    def test_a_split_set_with_a_shard_missing_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_file(Path(tmp) / "m-00001-of-00003.gguf", 8)
            write_file(Path(tmp) / "m-00003-of-00003.gguf", 8)
            with self.assertRaises(page_cache.WarmRefusal) as refusal:
                page_cache.weight_files(first)
        self.assertEqual(refusal.exception.kind, page_cache.MISSING_SHARD)
        self.assertIn("m-00002-of-00003.gguf", str(refusal.exception))

    def test_a_checkpoint_directory_is_every_file_in_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_file(Path(tmp) / "a.safetensors", 8)
            (Path(tmp) / "sub").mkdir()
            write_file(Path(tmp) / "sub" / "b.safetensors", 8)
            names = [p.name for p in page_cache.weight_files(tmp)]
        self.assertEqual(sorted(names), ["a.safetensors", "b.safetensors"])


class WarmTests(unittest.TestCase):
    def test_warm_reads_every_byte_of_every_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = write_file(Path(tmp) / "m-00001-of-00002.gguf", 3 * page_cache.READ_CHUNK_BYTES // 2)
            write_file(Path(tmp) / "m-00002-of-00002.gguf", 5)
            read = []
            real_open = open

            def counting_open(file, *args, **kwargs):
                stream = real_open(file, *args, **kwargs)
                real_readinto = stream.readinto

                def readinto(buffer):
                    n = real_readinto(buffer)
                    read.append(n)
                    return n

                return mock.Mock(readinto=readinto, fileno=stream.fileno, __enter__=lambda s: s, __exit__=lambda s, *a: stream.close())

            with mock.patch("builtins.open", counting_open):
                state = page_cache.warm(first, available_bytes=GIB)
        self.assertEqual(sum(read), 3 * page_cache.READ_CHUNK_BYTES // 2 + 5)
        self.assertEqual(state, {"state": "warm", "files": 2, "bytes": 3 * page_cache.READ_CHUNK_BYTES // 2 + 5})

    def test_weights_larger_than_the_available_memory_are_refused_with_a_typed_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = write_file(Path(tmp) / "w.gguf", 4096)
            with self.assertRaises(page_cache.WarmRefusal) as refusal:
                page_cache.warm(weights, available_bytes=4095)
        self.assertEqual(refusal.exception.kind, page_cache.EXCEEDS_MEMORY)

    def test_an_unreadable_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(page_cache.WarmRefusal) as refusal:
                page_cache.warm(Path(tmp) / "absent.gguf", available_bytes=GIB)
        self.assertEqual(refusal.exception.kind, page_cache.UNREADABLE)


class CellTests(unittest.TestCase):
    def test_every_engine_starts_from_the_same_cache_whichever_ran_before(self):
        """Each engine leaves a different cache behind (the old runner 79.2 GB, the streamed loader 49.8 GB); a cell
        must start from the same state in either order."""
        left_behind = {"engine-a": "79.2 GB, table warm", "engine-b": "49.8 GB, table cold"}
        for order in (("engine-a", "engine-b"), ("engine-b", "engine-a")):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "m").mkdir()
                write_file(Path(tmp) / "m" / "weights.gguf", 16)
                cache = {"resident": left_behind[order[1]]}
                started_from = {}

                def warm(path):
                    cache["resident"] = "whole weights set"
                    return {"state": "warm", "files": 1, "bytes": 16}

                binaries = {name: install_runner(tmp, name) for name in order}
                current = []

                def run(cmd, **kwargs):
                    started_from[current[0]] = cache["resident"]
                    cache["resident"] = left_behind[current[0]]
                    return run_fixture_runner(cmd, **kwargs)

                with (
                    mock.patch.object(page_cache, "warm", side_effect=warm),
                    mock.patch.object(observer, "run_observed", side_effect=run),
                ):
                    for name in order:
                        current[:] = [name]
                        row = run_local_cell(tmp, binaries[name])
                        self.assertEqual(row["page_cache"]["state"], "warm")
                self.assertEqual(started_from, {"engine-a": "whole weights set", "engine-b": "whole weights set"})

    def test_the_row_records_the_state_and_the_major_faults_the_cell_took(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "m").mkdir()
            write_file(Path(tmp) / "m" / "weights.gguf", 16)
            binary = install_runner(tmp, "runner")
            faults = iter([100, 100 + 515_745])
            with (
                mock.patch.object(page_cache, "children_major_faults", side_effect=lambda: next(faults)),
                mock.patch.object(observer, "run_observed", side_effect=run_fixture_runner),
            ):
                row = run_local_cell(tmp, binary)
        self.assertEqual(row["page_cache"], {"state": "warm", "files": 1, "bytes": 16, "major_faults": 515_745})

    def test_a_cell_whose_weights_cannot_be_warmed_is_refused_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "m").mkdir()
            write_file(Path(tmp) / "m" / "weights.gguf", 16)
            binary = install_runner(tmp, "runner")
            with (
                mock.patch.object(page_cache, "mem_available_bytes", return_value=15),
                mock.patch.object(observer, "run_observed", side_effect=AssertionError("the runner must not start")),
            ):
                row = run_local_cell(tmp, binary)
        self.assertEqual(row["status"], "unsupported")
        self.assertIn(page_cache.EXCEEDS_MEMORY, row["reason"])


class VerdictTests(unittest.TestCase):
    def repeats(self, faults_per_repeat):
        rows = []
        for n, faults in enumerate(faults_per_repeat):
            row = valid_single() | {
                "status": "ok", "model": "fixture", "scenario": "single", "decode_tok_s": 500.0 + n % 3,
                "page_cache": {"state": "warm", "files": 1, "bytes": 16, "major_faults": faults},
                "binary_sha256": "f" * 64,
            }
            rows.append(row)
        return rows

    def cell(self, rows):
        (cell,) = bench.compare_cells(rows, rows, receipts.POLICY)
        return cell

    def test_a_planted_cold_repeat_is_marked(self):
        cell = self.cell(self.repeats([0, 0, 515_745, 0, 0]))
        self.assertEqual(cell["verdict"], receipts.UNVERIFIED)
        self.assertIn("repeat 3 took 515745 major faults", cell["note"])

    def test_warm_repeats_are_not_marked(self):
        cell = self.cell(self.repeats([0, 12, 0, 3, 0]))
        self.assertNotEqual(cell["verdict"], receipts.UNVERIFIED, cell["note"])

    def test_a_row_without_a_recorded_cache_state_is_not_judged(self):
        rows = self.repeats([515_745] * 5)
        for row in rows:
            del row["page_cache"]
        self.assertNotEqual(self.cell(rows)["verdict"], receipts.UNVERIFIED)


if __name__ == "__main__":
    unittest.main()

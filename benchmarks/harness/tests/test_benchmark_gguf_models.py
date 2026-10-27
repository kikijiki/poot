"""A pre-quantized model is fetched file by file and handed to the runners as its first shard, not as a directory."""

import fnmatch
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import bench  # noqa: E402

LLAMACPP_RUNNER = bench.BENCH_ROOT / "runners" / "llamacpp" / "run.py"


def llamacpp_runner():
    spec = importlib.util.spec_from_file_location("llamacpp_run", LLAMACPP_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shard(index, count=4, stem="m-Q4"):
    return f"{stem}-{index:05d}-of-{count:05d}.gguf"


def touch_all(directory, names):
    for name in names:
        (Path(directory) / name).write_bytes(b"")


class ModelPathTests(unittest.TestCase):
    def test_a_checkpoint_directory_is_handed_over_whole(self):
        model = {"id": "m", "local_dir": "m-dir"}
        self.assertEqual(bench.model_path("/models", model), "/models/m-dir")

    def test_a_weights_file_is_handed_over_instead_of_the_directory(self):
        model = {"id": "m", "local_dir": "m-dir", "weights_file": "Q4/" + shard(1)}
        self.assertEqual(bench.model_path("/models", model), "/models/m-dir/Q4/" + shard(1))

    def test_a_model_without_a_local_dir_is_named_by_its_id(self):
        self.assertEqual(bench.model_path("/models", {"id": "org/m", "local_dir": ""}), "org/m")


class ShardSetTests(unittest.TestCase):
    def resolve(self, names):
        with tempfile.TemporaryDirectory() as tmp:
            touch_all(tmp, names)
            return Path(llamacpp_runner().resolve_gguf(tmp, "q4_k_xl")).name

    def test_a_complete_shard_set_resolves_to_its_first_shard(self):
        self.assertEqual(self.resolve([shard(i) for i in (3, 1, 4, 2)]), shard(1))

    def test_a_file_path_is_used_as_given(self):
        self.assertEqual(llamacpp_runner().resolve_gguf("/m/" + shard(1), "q4_k_xl"), "/m/" + shard(1))

    def test_an_incomplete_shard_set_is_not_loaded(self):
        with self.assertRaisesRegex(RuntimeError, "multiple .gguf"):
            self.resolve([shard(1), shard(2), shard(4)])

    def test_two_different_sets_in_one_directory_are_ambiguous(self):
        with self.assertRaisesRegex(RuntimeError, "multiple .gguf"):
            self.resolve([shard(1, 2, "a"), shard(2, 2, "a"), shard(1, 2, "b"), shard(2, 2, "b")])


class ManifestWeightsFileTests(unittest.TestCase):
    def models(self):
        return [m for m in bench.load_toml(bench.BENCH_ROOT / "manifest.toml")["model"] if m.get("weights_file")]

    def test_a_weights_file_is_inside_what_the_download_fetches(self):
        models = self.models()
        self.assertTrue(models, "the manifest names no pre-quantized model")
        for model in models:
            with self.subTest(model=model["id"]):
                self.assertTrue(model.get("local_dir"), "weights_file is relative to local_dir")
                self.assertTrue(
                    any(fnmatch.fnmatch(model["weights_file"], pattern) for pattern in model.get("hf_include", [])),
                    f"{model['weights_file']} is not matched by hf_include {model.get('hf_include')}",
                )

    def test_a_pre_quantized_model_names_the_precision_its_store_holds(self):
        for model in self.models():
            with self.subTest(model=model["id"]):
                self.assertNotIn(model["frameworks"]["poot"]["precision"], ("bf16", "f16", "f32"))


if __name__ == "__main__":
    unittest.main()

"""runners/torch/blackwell-torch.sh swaps the transformers venv's torch for the cu130 build on a Blackwell GPU only."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "runners" / "torch" / "blackwell-torch.sh"


def executable(path, body):
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_script(compute_cap, torch_version):
    """Run the script with a fake nvidia-smi, python (its torch reports `torch_version`) and uv (logs its argv)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        uv_log = tmp / "uv.log"
        executable(tmp / "nvidia-smi", f"echo {compute_cap}\n")
        executable(tmp / "uv", f'echo "$@" >> {uv_log}\n')
        # `python -c <torch check>` exits 0 iff the torch is a cu130 build; the verification heredoc always passes.
        executable(tmp / "python", f'if [ "$1" = -c ]; then case "{torch_version}" in *+cu130) exit 0;; *) exit 1;; esac; fi\ncat >/dev/null\n')
        env = {**os.environ, "PATH": f"{tmp}:{os.environ['PATH']}", "POOT_BENCH_TORCH_PYTHON": str(tmp / "python")}
        done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
        return done, uv_log.read_text() if uv_log.exists() else ""


class BlackwellTorchTest(unittest.TestCase):
    def test_blackwell_with_a_cu124_torch_reinstalls_cu130_keeping_the_transformers_pin(self):
        done, uv = run_script("12.0", "2.6.0+cu124")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("--torch-backend=cu130", uv)
        self.assertIn("transformers==5.0.0", uv)

    def test_blackwell_with_a_cu130_torch_is_left_alone(self):
        done, uv = run_script("12.0", "2.9.0+cu130")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(uv, "")

    def test_ampere_keeps_the_baked_torch(self):
        done, uv = run_script("8.6", "2.6.0+cu124")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(uv, "")


if __name__ == "__main__":
    unittest.main()

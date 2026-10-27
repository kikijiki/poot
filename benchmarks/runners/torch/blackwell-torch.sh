#!/usr/bin/env bash
# Blackwell (sm_120) torch for the transformers venv. The baked /opt/tf-venv ships a cu124 torch, which has no
# sm_120 kernels ("CUDA error: no kernel image is available for execution on the device"); the cu130 build keeps
# sm_80/86 and adds sm_120. On a Blackwell GPU this reinstalls torch from the cu130 index, keeping the
# transformers pin, then proves a CUDA matmul runs. On any other GPU it does nothing. Idempotent. Run on the pod
# by run-sweep.sh before the sweep; it lives beside the runner because the orchestrator syncs runners/torch to
# the pod, not setup/. The torch build lands in each row's framework_version.
set -euo pipefail

PY="${POOT_BENCH_TORCH_PYTHON:-/opt/tf-venv/bin/python}"
export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"

cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')"
major="${cap%%.*}"
if [ -z "$major" ] || [ "$major" -lt 12 ]; then
  echo "blackwell-torch: compute capability '${cap:-unknown}' is not Blackwell; torch left as baked"
  exit 0
fi

if "$PY" -c 'import torch, sys; sys.exit(0 if "+cu130" in torch.__version__ else 1)'; then
  echo "blackwell-torch: torch is already a cu130 build"
else
  echo "blackwell-torch: sm_$cap GPU, reinstalling torch (cu130) in $PY"
  uv pip install --python "$PY" --torch-backend=cu130 --reinstall-package torch "transformers==5.0.0" torch
fi

"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert torch.isfinite((x @ x).float().sum()).item(), "cuda matmul is not finite"
print("blackwell-torch: torch", torch.__version__, "cuda matmul ok on", torch.cuda.get_device_name(0))
PY

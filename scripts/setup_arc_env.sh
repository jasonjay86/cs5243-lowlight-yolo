#!/usr/bin/env bash
# Create the ARC conda env with a torch build that includes Volta (CC 7.0) kernels.
# The default PyPI wheel for torch 2.14.1 is CUDA 13.0, which dropped Volta, so it
# fails at the first kernel launch on the V100S nodes despite is_available() == True.
#
# Usage:  bash scripts/setup_arc_env.sh [env-name]     (default: cs5243-lowlight-cuda)
set -euo pipefail

ENV_NAME="${1:-cs5243-lowlight-cuda}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

module load anaconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] env '$ENV_NAME' already exists, reusing it"
else
  echo "[setup] creating env '$ENV_NAME' from environment.yml"
  conda env create -f environment.yml -n "$ENV_NAME"
fi

PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
echo "[setup] interpreter: $PY"

echo "[setup] installing torch 2.14.1 + cu126 (Volta / CC 7.0 kernels)"
"$PY" -m pip install --force-reinstall --quiet \
  torch==2.14.1 torchvision==0.29.1 \
  --index-url https://download.pytorch.org/whl/cu126

echo "[setup] verifying"
"$PY" - <<'EOF'
import torch
print("  torch      :", torch.__version__)
print("  cuda build :", torch.version.cuda)
print("  available  :", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("  no GPU on this node; rerun the kernel check on a compute node")
    raise SystemExit(0)
y = torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 64, 64, device="cuda"))
print("  conv2d ok  :", tuple(y.shape))
EOF

echo "[setup] done. Activate with: conda activate $ENV_NAME"
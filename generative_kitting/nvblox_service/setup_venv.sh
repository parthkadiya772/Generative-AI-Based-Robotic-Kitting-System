#!/bin/bash
# Create the nvblox service venv in nvblox_service/.venv and check that nvblox runs on the GPU.
# usage: generative_kitting/nvblox_service/setup_venv.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# The v0.0.10 wheel's metadata says 0.0.10.dev1 while its filename says 0.0.10; uv rejects that by default.
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match -r "$HERE/requirements.txt"

"$HERE/run.sh" - <<'EOF'
import torch, nvblox_torch
from nvblox_torch.mapper import Mapper   # loads libnvblox_lib.so (needs run.sh library path)
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("nvblox_torch", nvblox_torch.__version__ if hasattr(nvblox_torch, "__version__") else "?")
EOF

#!/bin/bash
# Run the nvblox venv's python with the CUDA libraries nvblox needs on the loader path.
# libnvblox_lib.so needs NPP (nvidia-npp-cu12) and the CUDA runtime but has no RPATH to the
# pip nvidia/ folders. usage: generative_kitting/nvblox_service/run.sh [python args...]
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE=$(echo "$HERE"/.venv/lib/python3*/site-packages/nvidia)   # any venv Python 3.x
[ -d "$SITE" ] || { echo "nvblox venv missing: run $HERE/setup_venv.sh" >&2; exit 1; }
export LD_LIBRARY_PATH="$SITE/npp/lib:$SITE/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/.venv/bin/python" "$@"

#!/usr/bin/env bash
set -euo pipefail
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cusparse/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python -u evaluate.py "$@"

#!/bin/bash
# N107 QSA: run a python entry point of this directory inside image 24c872759256 on the B70 84:00.0 (VRAM-only, no network).
# usage (from ~/freetoken-exl3 on omarchy):
#   bench/xpu_run.sh 0000:84:00.0 n107-qsa kernels/xpu_bmg/n107-qsa/run_xpu.sh test_qsa_xpu.py [args...]
# env passed through when set: EXL3_QSA_TRITON_CFG EXL3_QSA_TILE_CFG EXL3_QSA_TRITON_FP8 EXL3_QSA_GRF EXL3_QSA_TPW
#   EXL3_QSA_SYCL_CFG EXL3_QSA_SYCL_SCRATCH_MB EXL3_QSA_IMPL (N108)
#   TMO (s, default 1800)
set -euo pipefail
: "${DSV41_XPU_RENDER:?run me through bench/xpu_run.sh}"
[ "${DSV41_XPU_PCI:-}" = "0000:84:00.0" ] || { echo "refuse: DSV41_XPU_PCI=${DSV41_XPU_PCI:-unset}, want 0000:84:00.0"; exit 9; }
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/results/triton-cache"
timeout 20 docker rm -f n107-qsa-test >/dev/null 2>&1 || true
exec timeout "${TMO:-1800}" docker run --rm --network none --name n107-qsa-test \
  --device "$DSV41_XPU_RENDER:$DSV41_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 --memory 16g --memory-swap 16g \
  -e EXL3_QSA_TRITON_CFG -e EXL3_QSA_TILE_CFG -e EXL3_QSA_TRITON_FP8 -e EXL3_QSA_GRF -e EXL3_QSA_TPW -e EXL3_QSA_SYCL_CFG -e EXL3_QSA_SYCL_SCRATCH_MB -e EXL3_QSA_IMPL \
  -e TRITON_CACHE_DIR=/n107qsa/results/triton-cache -e PYTHONUNBUFFERED=1 \
  -v "$D:/n107qsa" -w /n107qsa --entrypoint python3 24c872759256 "$@"

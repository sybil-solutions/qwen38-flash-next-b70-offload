#!/bin/bash
# N112 GDN: run a python entry point of this directory inside image 24c872759256 on the B70 84:00.0 (VRAM-only, no network).
# usage (from ~/freetoken-exl3 on omarchy):
#   bench/xpu_run.sh 0000:84:00.0 n112-gdn kernels/xpu_bmg/n112-gdn/run_xpu.sh test_gdn_xpu.py [args...]
# env passed through when set: EXL3_GDN_SYCL_CFG EXL3_GDN_SYCL_LIB; TMO (s, default 900)
set -euo pipefail
: "${DSV41_XPU_RENDER:?run me through bench/xpu_run.sh}"
[ "${DSV41_XPU_PCI:-}" = "0000:84:00.0" ] || { echo "refuse: DSV41_XPU_PCI=${DSV41_XPU_PCI:-unset}, want 0000:84:00.0"; exit 9; }
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/results/triton-cache"
timeout 20 docker rm -f n112-gdn-test >/dev/null 2>&1 || true
exec timeout "${TMO:-900}" docker run --rm --network none --name n112-gdn-test \
  --device "$DSV41_XPU_RENDER:$DSV41_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 --memory 16g --memory-swap 16g \
  -e EXL3_GDN_SYCL_CFG -e EXL3_GDN_SYCL_LIB \
  -e TRITON_CACHE_DIR=/n112gdn/results/triton-cache -e PYTHONUNBUFFERED=1 \
  -v "$D:/n112gdn" -w /n112gdn --entrypoint python3 24c872759256 "$@"

#!/bin/bash
# N114 decode kernels: run a python entry point of this directory inside image 24c872759256 on the B70 84:00.0
# (VRAM-only, no network). usage (from ~/freetoken-exl3 on omarchy):
#   bench/xpu_run.sh 0000:84:00.0 n114-dec kernels/xpu_bmg/n114-decode/run_xpu.sh test_gdn_dec_xpu.py [args...]
# env passed through when set: EXL3_GDN_DEC_* EXL3_HC_DEC_* EXL3_QSA_DEC_* (see the *_dec.py loaders); TMO (s, default 600)
set -euo pipefail
: "${DSV41_XPU_RENDER:?run me through bench/xpu_run.sh}"
[ "${DSV41_XPU_PCI:-}" = "0000:84:00.0" ] || { echo "refuse: DSV41_XPU_PCI=${DSV41_XPU_PCI:-unset}, want 0000:84:00.0"; exit 9; }
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/results/triton-cache"
timeout 20 docker rm -f n114-dec-test >/dev/null 2>&1 || true
envs=()
for v in $(env | grep -oE '^EXL3_(GDN|HC|QSA)_DEC[A-Z0-9_]*' || true); do envs+=(-e "$v"); done
exec timeout "${TMO:-600}" docker run --rm --network none --name n114-dec-test \
  --device "$DSV41_XPU_RENDER:$DSV41_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 --memory 16g --memory-swap 16g \
  "${envs[@]}" -e TRITON_CACHE_DIR=/n114/results/triton-cache -e PYTHONUNBUFFERED=1 \
  -v "$D:/n114" -v "$D/../n107-hc:/hc:ro" -v "$D/../n107-qsa:/n107qsa:ro" -w /n114 --entrypoint python3 24c872759256 "$@"

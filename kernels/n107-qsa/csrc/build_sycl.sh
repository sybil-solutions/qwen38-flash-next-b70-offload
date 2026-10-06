#!/usr/bin/env bash
# N108: build build/qsa_row_sycl.so (torch.ops.n108qsa: SYCL per-query QSA prompt attention + prefill block selection,
# Strata port, MIT)
# inside image 24c872759256 (no GPU needed), same toolchain/flags as exl3xpu's build_moe.sh minus -ffast-math
# (the kernels rely on exact finite-sentinel compares). From omarchy:
#   docker run --rm --network none --memory 16g --user $(id -u):$(id -g) --name n108-build -v ~/freetoken-exl3/kernels/xpu_bmg/n107-qsa:/n107qsa \
#     -w /n107qsa --entrypoint bash 24c872759256 csrc/build_sycl.sh
# N108_TARGETS: default "intel_gpu_bmg_g31,spir64" (AOT code for the B70 + SPIR-V JIT fallback for other devices).
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
OUT=${N108_OUT:-build/qsa_row_sycl.so}
t0=$(date +%s)
icpx -fsycl -fsycl-targets=${N108_TARGETS:-intel_gpu_bmg_g31,spir64} -O3 -fPIC -std=c++17 -shared \
  -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${N108_FLAGS:-} \
  -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ csrc/qsa_row_sycl.sycl csrc/qsa_select_sycl.sycl -x none -o "$OUT.tmp" \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
mv -f "$OUT.tmp" "$OUT"
echo "built $OUT in $(( $(date +%s) - t0 )) s"
ls -la "$OUT"

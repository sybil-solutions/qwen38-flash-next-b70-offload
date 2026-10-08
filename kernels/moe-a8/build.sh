#!/usr/bin/env bash
# Build _moe_a8.so: the exl3xpu grouped-MoE library with the NVMe-tier kernel patches a2-a8 (device-managed VRAM expert
# cache, RAM-tier residency flags + miss ring, victim write-back, staged prefill copy, SVM prefetch). Source:
# exl3_moe.a8.sycl (trellis-serve exl3_moe.sycl + a2..a8). Runs inside the sglang-exl3-xpu-flashnext image (oneAPI icpx,
# torch XPU); no GPU needed. Device code is spir64 (JIT on first launch), the flags the campaign used for _moe_a8.so.
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
cd "$(dirname "$0")"
OUT=${MOE_OUT:-_moe_a8.so}
T=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel \
  -D_GLIBCXX_USE_CXX11_ABI=1 -I . -I"$T/include" -I"$T/include/torch/csrc/api/include" \
  -x c++ exl3_moe.a8.sycl -x none -o "$OUT.tmp" \
  -L"$T/lib" -Wl,-rpath,"$T/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
mv -f "$OUT.tmp" "$OUT"
ls -la "$OUT"

#!/usr/bin/env bash
# N114: build the decode kernel libraries (torch XPU op libraries, Strata ports, MIT) inside image 24c872759256
# (no GPU needed), same toolchain/flags as n107-qsa / n112-gdn. From omarchy:
#   docker run --rm --network none --memory 16g --user $(id -u):$(id -g) -e HOME=/tmp --name n114-build \
#     -v ~/freetoken-exl3/kernels/xpu_bmg/n114-decode:/n114 -w /n114 --entrypoint bash 24c872759256 csrc/build_sycl.sh [gdn hc qsa]
# default: all three.  build/<name>_dec.so <- csrc/<name>_dec.sycl  (torch.ops.n114<name>.*)
# N114_TARGETS: default "intel_gpu_bmg_g31,spir64" (AOT for the B70 + SPIR-V JIT fallback).
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(gdn hc qsa)
for n in "${names[@]}"; do
  src=csrc/${n}_dec.sycl; out=build/${n}_dec.so
  [ -f "$src" ] || { echo "skip $n: no $src"; continue; }
  t0=$(date +%s)
  icpx -fsycl -fsycl-targets=${N114_TARGETS:-intel_gpu_bmg_g31,spir64} -O3 -fPIC -std=c++17 -shared \
    -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${N114_FLAGS:-} \
    -I$T/include -I$T/include/torch/csrc/api/include \
    -x c++ "$src" -x none -o "$out.tmp" \
    -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu
  mv -f "$out.tmp" "$out"
  echo "built $out in $(( $(date +%s) - t0 )) s"
done
ls -la build/*.so

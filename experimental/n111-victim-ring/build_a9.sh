#!/bin/bash
# N111: build _moe_a9.so (a8 + victim ring) from runs/N111-victim-ring/csrc in image 24c872759256 (CPU only, no GPU).
# usage: build_a9.sh <src.sycl name in csrc> <out .so name>
set -uo pipefail
D=$HOME/freetoken-exl3/runs/N111-victim-ring
SRC=${1:-exl3_moe.sycl}; OUT=${2:-_moe_a9.so}
timeout 1500 docker run --rm --name n111-build --network none --memory 24g --cpuset-cpus 40-47 -v $D:/w \
  --entrypoint bash 24c872759256 -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /w; \
  T=\$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))'); \
  icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel \
  -D_GLIBCXX_USE_CXX11_ABI=1 -I csrc -I\$T/include -I\$T/include/torch/csrc/api/include -x c++ csrc/$SRC -x none -o /w/$OUT \
  -L\$T/lib -Wl,-rpath,\$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu 2>&1 | grep -E 'error|Error' -A3; \
  chown $(id -u):$(id -g) /w/$OUT; ls -la /w/$OUT"

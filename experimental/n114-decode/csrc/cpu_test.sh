#!/usr/bin/env bash
# N114: build + run the CPU-device tests (OpenCL CPU device, no torch, no GPU) inside image 24c872759256. From omarchy:
#   docker run --rm --network none --memory 16g --user $(id -u):$(id -g) -e HOME=/tmp --name n114-cpu \
#     -v ~/freetoken-exl3/kernels/xpu_bmg/n114-decode:/n114 -w /n114 --entrypoint bash 24c872759256 csrc/cpu_test.sh [gdn hc qsa]
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
names=("$@"); [ ${#names[@]} -eq 0 ] && names=(gdn hc qsa)
rc=0
for n in "${names[@]}"; do
  src=csrc/cpu_test_${n}.cpp
  [ -f "$src" ] || { echo "skip $n: no $src"; continue; }
  echo "=== cpu_test_${n}"
  icpx -fsycl -O2 -std=c++17 "$src" -o build/cpu_test_${n} || { rc=1; continue; }
  timeout 1200 build/cpu_test_${n} || rc=1
done
exit $rc

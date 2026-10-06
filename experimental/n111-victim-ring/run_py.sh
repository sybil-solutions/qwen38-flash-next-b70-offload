#!/bin/bash
# N111 generic: bench/xpu_run.sh 0000:84:00.0 n111-<tag> runs/N111-victim-ring/run_py.sh <tag> <script.py> [ENV=VAL...]   (a9 lib)
set -u
D=$HOME/freetoken-exl3/runs/N111-victim-ring; N=$HOME/freetoken-exl3/runs/N104-nvtier
TAG=${1:?tag}; PY=${2:?script}; shift 2
[ "${DSV41_XPU_PCI:-}" = 0000:84:00.0 ] || { echo "refuse"; exit 9; }
start=$(date '+%Y-%m-%d %H:%M:%S')
$D/guard_boot.sh pre-$TAG > $D/guard_pre_$TAG.txt; grep -q GUARD_PASS $D/guard_pre_$TAG.txt || { echo "PRE GUARD FAIL"; cat $D/guard_pre_$TAG.txt; exit 3; }
envs=(); for kv in "$@"; do envs+=(-e "$kv"); done
timeout 870 docker run --rm --name n111-test --device "$DSV41_XPU_RENDER:$DSV41_XPU_RENDER:rwm" -e ZE_AFFINITY_MASK=0 \
  --memory 24g --memory-swap 24g --ulimit memlock=-1:-1 --cpuset-cpus 40-47 --network none \
  -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=1 -e EnableRecoverablePageFaults=1 \
  -v $N/src/exl3xpu:/pkg/exl3xpu:ro -v $D/nvtier.py:/pkg/exl3xpu/nvtier.py:ro -v /mnt/nvx/n104:/s:ro \
  -v $HOME/freetoken-exl3/runs/2026-09-29-K02-traces:/t:ro -v $D:/o -e EXL3_MOE_LIB=${MOELIB:-/o/_moe_a9.so} -e LIBTAG=a9 -e OUT=/o/n111_$TAG.json "${envs[@]}" \
  --entrypoint python3 24c872759256 /o/$PY > $D/log_$TAG.txt 2>&1
rc=$?; echo "docker rc=$rc" >> $D/log_$TAG.txt
timeout 30 docker stop n111-test >/dev/null 2>&1
$D/guard_boot.sh post-$TAG > $D/guard_post_$TAG.txt; tail -1 $D/guard_post_$TAG.txt
exit $rc

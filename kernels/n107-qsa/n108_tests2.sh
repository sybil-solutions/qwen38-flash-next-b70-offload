#!/bin/bash
# N108 test chain v2 (run with the 84:00.0 lock held, via n108_job.sh). Stages in $N108_STAGES (comma list):
#   smoke bf16 time sweep sweep32 select dump
# A stage that crashes (exit other than 0 = pass / 1 = tolerance fails) stops the chain. Per-stage timeout TMO_STAGE.
cd ~/freetoken-exl3
Q=kernels/xpu_bmg/n107-qsa
R=/n107qsa/results
st=",${N108_STAGES:-smoke,time},"
TAG=${N108_TAG:-v2}
run(){ local name=$1; shift; echo "=== $name $(date -Is)"; TMO=${TMO_STAGE:-600} $Q/run_xpu.sh "$@"; local rc=$?; echo "=== $name rc=$rc $(date -Is)"
  [ $rc -le 1 ] || { echo "stage $name crashed (rc=$rc): stopping"; exit $rc; }; }
V=${N108_VARIANTS:-ref,union,sycl_row,sycl_strata,sycl_bcast}
[[ $st == *,smoke,* ]] && run smoke test_qsa_xpu.py --edge --patch-mock --ctx 8192 --rows 2048 --variants $V --iters 3 --out $R/n108_${TAG}_smoke.jsonl
[[ $st == *,bf16,* ]] && run bf16 test_qsa_xpu.py --edge --kv bf16 --ctx 4096 --rows 1024 --variants $V --iters 3 --out $R/n108_${TAG}_bf16.jsonl
[[ $st == *,time,* ]] && run time test_qsa_xpu.py --ctx 8192 32768 --rows 8192 --variants $V --iters 5 --out $R/n108_${TAG}_8k_32k.jsonl
[[ $st == *,sweep,* ]] && run sweep test_qsa_xpu.py --sycl-sweep --ctx 8192 --rows 8192 --iters 3 --out $R/n108_${TAG}_sweep.jsonl
[[ $st == *,sweep32,* ]] && run sweep32 test_qsa_xpu.py --sycl-sweep --ctx 32768 --rows 8192 --iters 3 --out $R/n108_${TAG}_sweep32.jsonl
[[ $st == *,select,* ]] && run select test_qsa_select_xpu.py --patch-test --ctx 8192 32768 --rows 8192 --rpw 4 8 16 --out $R/n108_${TAG}_select.jsonl
[[ $st == *,dump,* ]] && ls $Q/results/dump/qsa_topk_L*.pt >/dev/null 2>&1 && run dump test_qsa_xpu.py --dump '/n107qsa/results/dump/qsa_topk_L*.pt' --variants ref,union,sycl_row --out $R/n108_${TAG}_dump.jsonl
exit 0

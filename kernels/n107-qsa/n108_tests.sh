#!/bin/bash
# N108 test chain (run with the 84:00.0 lock held, via n108_job.sh). Stage list in $N108_STAGES (default all).
# A stage that crashes (exit other than 0 = pass / 1 = tolerance fails) stops the chain.
cd ~/freetoken-exl3
Q=kernels/xpu_bmg/n107-qsa
R=/n107qsa/results
st=${N108_STAGES:-smoke,bf16,time,sweep}
run(){ local name=$1; shift; echo "=== $name $(date -Is)"; TMO=${TMO_STAGE:-600} $Q/run_xpu.sh test_qsa_xpu.py "$@"; local rc=$?; echo "=== $name rc=$rc $(date -Is)"
  [ $rc -le 1 ] || { echo "stage $name crashed (rc=$rc): stopping"; exit $rc; }; }
V=ref,union,sycl_row,sycl_strata,sycl_bcast
[[ $st == *smoke* ]] && run smoke --edge --ctx 8192 --rows 2048 --variants $V --iters 3 --out $R/n108_t0_smoke.jsonl
[[ $st == *bf16* ]] && run bf16 --edge --kv bf16 --ctx 4096 --rows 1024 --variants $V --iters 3 --out $R/n108_t0_bf16.jsonl
[[ $st == *time* ]] && run time --ctx 8192 32768 --rows 8192 --variants $V --iters 5 --out $R/n108_t1_8k_32k.jsonl
[[ $st == *sweep* ]] && run sweep --sycl-sweep --ctx 8192 --rows 8192 --iters 3 --out $R/n108_t2_sweep.jsonl
exit 0

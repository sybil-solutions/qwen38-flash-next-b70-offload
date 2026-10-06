#!/bin/bash
# usage: n104_matrix.sh <srvN> <cells e.g. 8192x1,8192x2> [VAR=VAL ...]   launch on c3 via xpu_run, wait, smoke, matrix
cd ~/freetoken-exl3; R=runs/N104-nvtier/$1; CELLS=$2; shift 2; mkdir -p $R
printf '%s\n' "$@" > $R/env.txt
timeout 60 docker stop -t 20 n104-srv >/dev/null 2>&1; sleep 3
tmux kill-session -t n104srv 2>/dev/null; tmux kill-session -t n104mem 2>/dev/null
tmux new -d -s n104srv "env RAM_GB=8 $* timeout 21600 bench/xpu_run.sh ${N104_PCI:-0000:84:00.0} n104-$(basename $R) runs/N104-nvtier/n104_serve.sh $R"
tmux new -d -s n104mem "while true; do echo \$(date +%T) \$(docker stats --no-stream --format {{.MemUsage}} n104-srv 2>/dev/null); sleep 5; done >> $R/mem.txt"
sleep 40
for i in $(seq 1 400); do grep -q "Uvicorn running" $R/log.txt 2>/dev/null && { echo READY; break; }; docker ps --format "{{.Names}}" | grep -q n104-srv || pgrep -f "xpu_run.sh [0-9a-f:.]* n104-$(basename $R) " >/dev/null || { echo EXITED; grep -v "Ignore import" $R/log.txt | tail -8 | cut -c1-250; exit 1; }; sleep 10; done; sleep 5
grep -E "warm-up|max_running_requests|max_total_num_tokens" $R/log.txt | cut -c22-230
cd runs/N104-nvtier; S=$(basename $R)
timeout 900 python3 smoke_n104.py > $S/smoke.jsonl 2> $S/smoke.err; cut -c1-200 $S/smoke.jsonl; tail -1 $S/smoke.err
timeout 7000 python3 bench_matrix.py $CELLS 2>&1 | tee $S/matrix.jsonl
grep -E "prefill prof" $S/log.txt | tail -6 | cut -c22-330; grep -E "nvtier last" $S/log.txt | tail -2 | cut -c40-330; grep -E "crashed|Killed" $S/log.txt | head -2
sort -k2 -h $S/mem.txt | tail -1

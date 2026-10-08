#!/bin/bash
# usage: n130_phase.sh <tag> <cells|none> <qual 0|1> [VAR=VAL ...]
# One benchmark phase on B70 48:00.0 under flock ~/nvx_bench.lock: launch n130-srv, guard every 60 s, 1 s NVMe/GPU/memory
# monitor, smoke + greedy panel + ref-panel KL (qual=1), matrix (cells), stop, guards before/after.
set -u
cd ~/freetoken-exl3; Q=runs/N130-16g; T=$1; CELLS=$2; QUAL=$3; shift 3; R=$Q/$T; mkdir -p $R
printf '%s\n' "$@" > $R/env.txt
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/driver.log; }
[ -e $Q/TRIPPED ] && { log "global TRIPPED marker present; abort"; exit 3; }
START=$(date '+%F %T')
guard(){ bash $Q/n130_guard.sh "$START" "$1" >> $R/guard.txt 2>&1; local rc=$?; tail -1 $R/guard.txt >> $R/driver.log; [ $rc = 0 ] || { touch $R/TRIPPED $Q/TRIPPED; }; return $rc; }
stopsrv(){ timeout 90 docker stop -t 20 n130-srv >/dev/null 2>&1; for i in $(seq 1 30); do tmux has-session -t n130srv 2>/dev/null || break; sleep 3; done; }
cleanup(){ stopsrv; tmux kill-session -t n130watch 2>/dev/null; tmux kill-session -t n130mon 2>/dev/null; tmux kill-session -t n130mp 2>/dev/null; }
guard pre || { log "TRIPPED before launch"; exit 4; }
log "waiting for ~/nvx_bench.lock"
exec 7>$HOME/nvx_bench.lock; flock 7; log "nvx lock held"
guard pre-locked || { log "TRIPPED"; exit 4; }
tmux has-session -t n130srv 2>/dev/null && { log "n130srv session exists; abort"; exit 5; }
rm -f $R/log.txt
tmux new -d -s n130srv "env $* timeout 14400 bash $Q/n130_xpu_run.sh n130-$T bash $Q/n130_serve.sh $R > $R/xpu_run.out 2>&1"
tmux new -d -s n130watch "bash $Q/n130_watch.sh $R '$START'"
tmux new -d -s n130mon "python3 $Q/n130_mon.py $R/mon.jsonl n130-srv card4"
tmux new -d -s n130mp "sleep 20; bash $Q/n130_memprobe2.sh $R/memprobe.jsonl"
log "launched: $*"
sleep 30; ready=0
for i in $(seq 1 120); do
  grep -q "Uvicorn running" $R/log.txt 2>/dev/null && { ready=1; break; }
  tmux has-session -t n130srv 2>/dev/null || break
  [ -e $R/TRIPPED ] && break
  sleep 5
done
[ $ready = 1 ] || { log "server did not come up"; grep -v "Ignore import" $R/log.txt 2>/dev/null | tail -15 | cut -c1-250 >> $R/driver.log; cleanup; guard post; exit 6; }
sleep 5; log "ready: $(grep -E 'max_total_num_tokens' $R/log.txt | tail -1 | cut -c1-220)"
guard ready || { log TRIPPED; cleanup; exit 7; }
if [ "$QUAL" = 1 ]; then
  timeout 1800 python3 $Q/warm_n130.py > $R/warm.jsonl 2>&1; log "warm-up (not scored) rc=$? $(cut -c1-160 $R/warm.jsonl | tail -1)"
  timeout 1800 python3 $Q/smoke_n130.py > $R/smoke.jsonl 2> $R/smoke.err; log "smoke rc=$?"
  guard smoke || { log TRIPPED; cleanup; exit 7; }
  timeout 3600 python3 $Q/greedy_n130.py kit/reference/qwen3.8-flash-next-exl3-ref-panel.json $R/greedy.json > $R/greedy.log 2>&1; log "greedy rc=$? $(tail -1 $R/greedy.log)"
  guard greedy || { log TRIPPED; cleanup; exit 7; }
  timeout 3600 python3 $Q/score_ref_panel_n130.py --url http://127.0.0.1:30330 --panel kit/reference/qwen3.8-flash-next-exl3-ref-panel.json --out $R/refpanel.json > $R/refpanel.log 2>&1; log "refpanel rc=$? $(tail -1 $R/refpanel.log)"
  guard refpanel || { log TRIPPED; cleanup; exit 7; }
fi
if [ "$CELLS" != none ]; then
  echo "$(date +%s.%N) matrix start" >> $R/phases.txt
  timeout 9000 python3 $Q/bench_matrix_n130.py $CELLS > $R/matrix.jsonl 2> $R/matrix.err; log "matrix rc=$?"
  echo "$(date +%s.%N) matrix end" >> $R/phases.txt
  guard matrix || { log TRIPPED; cleanup; exit 7; }
fi
grep -E "nvtier last" $R/log.txt | tail -2 | cut -c40-400 >> $R/driver.log
grep -E "prefill prof" $R/log.txt | tail -2 | cut -c22-330 >> $R/driver.log
cleanup; log "stopped"
guard post || { log "TRIPPED post"; exit 8; }
log DONE

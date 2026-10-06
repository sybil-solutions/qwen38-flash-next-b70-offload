#!/bin/bash
# N113 driver: one GPU job on the B70 84:00.0 (bench/xpu_run.sh flock; waits for other agents), hard cap 20 min of
# server lifetime (timeout inside the lock). Guard before/after, greedy probe, bench_matrix cells, stop OUR server only.
# usage: n113_run.sh <tag> <cells e.g. 8192x1> VAR=VAL ...      (VARs go to runs/N104-nvtier/n104_serve.sh)
set -u
cd ~/freetoken-exl3; Q=runs/N113-mtp; TAG=$1; CELLS=$2; shift 2; R=$Q/$TAG; mkdir -p $R
OWN=n113-$TAG; LOCKOWN=locks/xpu-0000_84_00_0.owner
printf '%s\n' "$@" > $R/env.txt
START=$(date '+%F %T'); echo "$START" > $R/start_local.txt
log(){ echo "[$(date '+%F %T')] $*" | tee -a $R/driver.log; }
guard(){ bash $Q/n113_guard.sh "$START" "$1" >> $R/guard.txt 2>&1; local rc=$?; tail -2 $R/guard.txt | tee -a $R/driver.log; return $rc; }
mine(){ grep -q "^$OWN " $LOCKOWN 2>/dev/null; }
stopsrv(){
  if mine && timeout 10 docker ps --format '{{.Names}}' | grep -qx n104-srv; then
    log "stopping our n104-srv"; timeout 90 docker stop -t 30 n104-srv >/dev/null 2>&1
  fi
  for i in $(seq 1 36); do tmux has-session -t $OWN 2>/dev/null || break; sleep 5; done
  [ -n "${MEMPID:-}" ] && kill $MEMPID 2>/dev/null
}
guard pre || { log "TRIPPED before launch; abort"; exit 3; }
tmux has-session -t $OWN 2>/dev/null && { log "session $OWN exists; abort"; exit 4; }
rm -f $R/log.txt
tmux new -d -s $OWN "env $* bench/xpu_run.sh 0000:84:00.0 $OWN timeout 1200 runs/N104-nvtier/n104_serve.sh $R > $R/xpu_run.out 2>&1"
log "queued $OWN: $*"
ready=0
for i in $(seq 1 2880); do      # lock queue (other agents) + startup: up to 8 h
  grep -q "Uvicorn running" $R/log.txt 2>/dev/null && { ready=1; break; }
  tmux has-session -t $OWN 2>/dev/null || break
  sleep 10
done
if [ $ready = 1 ]; then
  ( while true; do echo $(date +%T) $(timeout 10 docker stats --no-stream --format '{{.MemUsage}}' n104-srv 2>/dev/null); sleep 5; done >> $R/mem.txt ) &
  MEMPID=$!
fi
T0=$(date +%s)
[ $ready = 1 ] || { log "server did not come up"; grep -v "Ignore import" $R/log.txt 2>/dev/null | grep -E -i "error|Traceback|raise|exl3xpu" | tail -25 | cut -c1-300 >> $R/driver.log; tail -5 $R/xpu_run.out >> $R/driver.log; stopsrv; guard post; exit 5; }
log "ready; $(grep -E 'max_running_requests|max_total_num_tokens' $R/log.txt | tail -1 | cut -c1-260)"
guard ready || { log TRIPPED; stopsrv; exit 6; }
timeout 600 python3 $Q/n113_probe.py $R/probe.jsonl >> $R/driver.log 2>&1; log "probe rc=$?"
guard probe || { log TRIPPED; stopsrv; exit 6; }
left=$(( 1140 - ($(date +%s) - T0) ))
if [ $left -gt 120 ] && [ "$CELLS" != none ]; then
  timeout $left python3 runs/N104-nvtier/bench_matrix.py $CELLS > $R/matrix.jsonl 2>&1; log "matrix rc=$? (budget ${left}s)"
  cut -c1-330 $R/matrix.jsonl | tee -a $R/driver.log
fi
grep -E "nvtier last" $R/log.txt | tail -3 | cut -c40-330 >> $R/driver.log
grep -E -i "accept len|accept_len|spec_accept" $R/log.txt | tail -5 | cut -c1-300 >> $R/driver.log
stopsrv
guard post; rc=$?
log "done rc=$rc; peak mem $(sort -k2 -h $R/mem.txt 2>/dev/null | tail -1)"

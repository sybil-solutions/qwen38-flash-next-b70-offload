#!/bin/bash
# N114 guarded GPU job on the B70 0000:84:00.0 through the bench/xpu_run.sh lock (waits its turn behind other jobs).
#   kernels/xpu_bmg/n114-decode/n114_job.sh <name> <timeout_s <= 600> <command...>   (cwd becomes ~/freetoken-exl3)
# Refuses unless runs/N107/CLEARED exists. With the lock held: guard, run the command under `timeout`, guard again.
# Guard: dsv41 localhost:30141/health == 200; `journalctl -k -b` (since BOOT) has no Completion-Wait timeout,
# Link Down, Card not present, "reboot is needed" or fatal Hardware Error line; no D-state khugepaged / kcompactd.
# Output: n114-decode/results/<name>.out, log n114-decode/results/n114_jobs.log. Exit 3 = guard trip (stop everything).
set -u
cd ~/freetoken-exl3
Q=kernels/xpu_bmg/n114-decode
L=$Q/results/n114_jobs.log
mkdir -p $Q/results
log(){ echo "$(date -Is) $*" | tee -a $L; }
guard(){ # $1 label
  local h bad dst
  h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:30141/health)
  bad=$(timeout 20 journalctl -k -b --no-pager 2>/dev/null | grep -cE "Completion-Wait loop timed out|Link Down|Card not present|reboot is needed|Hardware Error.*([Ff]atal|[Uu]ncorrect)")
  dst=$(ps -eo stat=,comm= | awk '$1 ~ /^D/ && ($2 ~ /khugepaged|kcompactd/)' | wc -l)
  if [ "$h" != 200 ] || [ "$bad" != 0 ] || [ "$dst" != 0 ]; then log "$1 GUARD TRIP health=$h faultlines_since_boot=$bad dstate_khugepaged_kcompactd=$dst"; return 1; fi
  log "$1 guard ok (health 200, 0 fault lines since boot, no D-state khugepaged/kcompactd)"; }
if [ "${1:-}" = "--inner" ]; then
  shift; n=$1 t=$2; shift 2
  guard "$n pre" || exit 3
  log "START $n (lock held, render $DSV41_XPU_RENDER)"
  timeout "$t" "$@" > $Q/results/$n.out 2>&1; rc=$?
  log "END $n rc=$rc"
  docker ps --format '{{.Names}}' | grep -q '^n114-dec-test$' && { log "stopping leftover n114-dec-test"; timeout 30 docker stop -t 10 n114-dec-test >/dev/null 2>&1; }
  guard "$n post" || exit 3
  exit $rc
fi
n=$1; t=$2; shift 2
[ -e runs/N107/CLEARED ] || { echo "refuse: runs/N107/CLEARED missing (box not cleared for GPU work)"; exit 65; }
[ "$t" -le 600 ] || { echo "refuse: timeout $t > 600 s"; exit 64; }
guard "$n queue" || exit 3
log "QUEUED $n (waiting for the 84:00.0 lock)"
exec timeout $((t + 1200)) bench/xpu_run.sh 0000:84:00.0 n114-dec "$Q/n114_job.sh" --inner "$n" "$t" "$@"

#!/bin/bash
# N108 guarded GPU job on the B70 0000:84:00.0 through the bench/xpu_run.sh lock (waits its turn behind other jobs).
#   kernels/xpu_bmg/n107-qsa/n108_job.sh <name> <timeout_s> <command...>     (from anywhere; cwd becomes ~/freetoken-exl3)
# With the lock held: guard (dsv41 /health 200, no kernel fault lines since the job start, AER counters unchanged),
# run the command under `timeout`, guard again. Output: n107-qsa/results/<name>.out, log: n107-qsa/results/n108_jobs.log.
# Exit 3 = guard trip (stop everything and report).
set -u
cd ~/freetoken-exl3
Q=kernels/xpu_bmg/n107-qsa
L=$Q/results/n108_jobs.log
log(){ echo "$(date -Is) $*" | tee -a $L; }
aer(){ for d in 80:03.1 82:00.0 83:01.0 83:02.0 84:00.0 81:00.0; do p=/sys/bus/pci/devices/0000:$d; echo -n "$(awk '/TOTAL/{print $2}' $p/aer_dev_nonfatal 2>/dev/null)/$(awk '/TOTAL/{print $2}' $p/aer_dev_fatal 2>/dev/null) "; done; }
guard(){ # $1 since, $2 aer baseline, $3 label
  local h bad a
  h=$(curl -s -m 5 -o /dev/null -w "%{http_code}" localhost:30141/health)
  bad=$(journalctl -k --since "$1" --no-pager 2>/dev/null | grep -cE "Completion-Wait loop timed out|pciehp.*(Link Down|Card not present|removed)|Hardware Error.*(fatal|ncorrect)")
  a=$(aer)
  if [ "$h" != 200 ] || [ "$bad" != 0 ] || [ "$a" != "$2" ]; then log "$3 GUARD TRIP health=$h faultlines=$bad aer=[$a] base=[$2]"; return 1; fi
  log "$3 guard ok (health 200, 0 fault lines, aer [$a])"; }
if [ "${1:-}" = "--inner" ]; then
  shift; n=$1 t=$2; shift 2
  s=$(date "+%Y-%m-%d %H:%M:%S"); b=$(aer)
  guard "$s" "$b" "$n pre" || exit 3
  log "START $n (lock held, render $DSV41_XPU_RENDER)"
  timeout "$t" "$@" > $Q/results/$n.out 2>&1; rc=$?
  log "END $n rc=$rc"
  docker ps --format '{{.Names}}' | grep -q '^n107-qsa-test$' && { log "stopping leftover n107-qsa-test"; timeout 30 docker stop -t 10 n107-qsa-test >/dev/null 2>&1; }
  guard "$s" "$b" "$n post" || exit 3
  exit $rc
fi
n=$1; t=$2; shift 2
[ "$t" -le 1200 ] || { echo "refuse: timeout $t > 1200 s"; exit 64; }
log "QUEUED $n (waiting for the 84:00.0 lock)"
exec bench/xpu_run.sh 0000:84:00.0 n108-qsa "$Q/n108_job.sh" --inner "$n" "$t" "$@"

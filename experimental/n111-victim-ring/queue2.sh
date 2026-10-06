#!/bin/bash
# N111 queue: after x9 (a9 exact) passes -> x8 (a8 exact, cross-lib) -> s9 (a9 sim + bench). Each its own xpu_run lock turn.
cd ~/freetoken-exl3; Q=runs/N111-victim-ring
while ! grep -q "docker rc" $Q/q_x9.txt 2>/dev/null; do sleep 20; done
grep -q "docker rc=0" $Q/q_x9.txt && grep -q GUARD_PASS $Q/guard_post_x9.txt || { echo "x9 failed/guard - stop $(date -Is)" >> $Q/queue.log; exit 1; }
echo "x9 ok $(date -Is)" >> $Q/queue.log
timeout 14400 bench/xpu_run.sh 0000:84:00.0 n111-x8 $Q/run.sh x8 a8 PHASES=exact > $Q/q_x8.txt 2>&1
echo "x8 rc=$? $(date -Is)" >> $Q/queue.log
grep -q GUARD_PASS $Q/guard_post_x8.txt || { echo "x8 guard fail - stop" >> $Q/queue.log; exit 1; }
timeout 14400 bench/xpu_run.sh 0000:84:00.0 n111-s9 $Q/run.sh s9 a9 PHASES=sim,bench > $Q/q_s9.txt 2>&1
echo "s9 rc=$? $(date -Is)" >> $Q/queue.log

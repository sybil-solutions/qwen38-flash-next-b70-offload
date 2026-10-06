#!/bin/bash
# after the d2h test: a9b exact + sim (ring thread drain) + bench (none/wb/ring/ringT)
cd ~/freetoken-exl3; Q=runs/N111-victim-ring
# (queued directly on the lock)
timeout 21600 bench/xpu_run.sh 0000:84:00.0 n111-s9b $Q/run.sh s9b a9b PHASES=exact,sim,bench SIM_MODES=ring,wb SIM_M=1 > $Q/q_s9b.txt 2>&1
echo "s9b rc=$? $(date -Is)" >> $Q/queue.log

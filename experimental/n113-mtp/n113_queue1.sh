#!/bin/bash
cd ~/freetoken-exl3
C="STAGE=1 NVASYNC=1 HCXPU=1 HOSTBUF=5 RAM_GB=7 MAXTOT=131072 CHUNK=8192 MAXRUN=2 MAMBA=16 GRAPH_BS=1,2 QSAIMPL=sycl_row QSASELECT=sycl"
bash runs/N113-mtp/n113_run.sh s1-spec3 8192x1 $C SLOTS=7400 SPEC=1 SPECSTEPS=3 PLGOVR=$HOME/freetoken-exl3/runs/N113-mtp/sglang_plugin.mtp.py
echo "s1 rc=$?" >> runs/N113-mtp/queue1.done
bash runs/N113-mtp/n113_run.sh b1-base 8192x1 $C SLOTS=8000
echo "b1 rc=$?" >> runs/N113-mtp/queue1.done
echo ALLDONE >> runs/N113-mtp/queue1.done

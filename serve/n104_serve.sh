#!/bin/bash
# n107-qsa: QSATRITON=1 enables kernels/xpu_bmg/n107-qsa (see its README)
# N104 NVMe-tier serve of Qwen3.8-Flash-Next on the free B70 (c3:00.0 only, via bench/xpu_run.sh), host RAM capped.
# usage: bench/xpu_run.sh 0000:c3:00.0 n104-srv runs/N104-nvtier/n104_serve.sh <run dir>
set -uo pipefail
cd ~/freetoken-exl3
node=$DSV41_XPU_RENDER
[ "$DSV41_XPU_PCI" = "${N104_PCI:-0000:84:00.0}" ] || { echo "refuse: not ${N104_PCI:-0000:84:00.0}"; exit 9; }
RUN=${1:?run dir}; mkdir -p "$RUN"
S=$HOME/freetoken-exl3/runs/N104-nvtier
P=/opt/trellis-serve/xpu/exl3xpu
MODEL=turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5
MEM=${MEM:-32g}; PORT=${PORT:-30260}; GRAPH_BS=${GRAPH_BS:-1,2}; GRAPH_BS=${GRAPH_BS//,/ }
timeout 20 docker rm -f n104-srv >/dev/null 2>&1
# N111: VRING=1 -> VRAM victim ring (async write-back): a9 kernel lib + N111 nvtier.py / plugin (NvTierAsyncRing; needs NVASYNC=1).
# VRK = ring slots (default 128 x 1.86 MB), DECPF=1 = svm prefetch of pages landed by decode fills. Unset = unchanged.
NVT=$S/src/exl3xpu/nvtier.py; PLG=$S/sglang_plugin.tier.py; MOELIB=_moe_a8.so; VR_ARGS=()
if [ "${VRING:-0}" = 1 ]; then
  V=$HOME/freetoken-exl3/runs/N111-victim-ring; NVT=$V/nvtier.py; PLG=$V/sglang_plugin.tier.py; MOELIB=_moe_a9.so
  VR_ARGS=(-e EXL3_NVTIER_VICTIM_RING=1 -e EXL3_NVTIER_VRING_K=${VRK:-128} -e EXL3_NVTIER_DECODE_PF=${DECPF:-0})
fi
exec docker run --rm --name n104-srv --device "$node:$node:rwm" -e ZE_AFFINITY_MASK=0 \
  --memory=$MEM --memory-swap=$MEM --shm-size 8g --ulimit memlock=-1:-1 --cpuset-cpus=26-39 \
  -p 127.0.0.1:$PORT:$PORT -e HF_HUB_OFFLINE=1 -e EXL3_MODEL_PATH=/models/$MODEL \
  -e NEOReadDebugKeys=1 -e EnableSharedSystemUsmSupport=1 -e EnableRecoverablePageFaults=1 \
  -e EXL3_MOE_LIB=/n104/$MOELIB -e EXL3_MOE_SLOTS=${SLOTS:-8000} -e EXL3_MOE_CACHE=1 -e EXL3_NVTIER_SVM_PREFETCH=${SVMPF:-1} -e EXL3_NVTIER_STAGE=${STAGE:-0} -e EXL3_NVTIER_NO_VICTIM=${NOVICT:-0} -e EXL3_NVTIER_ASYNC=${NVASYNC:-0} -v $HOME/freetoken-exl3/kernels/xpu_bmg/n107-hc:/hc:ro -e EXL3_HC_DIR=/hc -e EXL3_HC_XPU=${HCXPU:-0} -e EXL3_HC_MIX=${HCMIX:-epilogue} -e EXL3_HC_COMBINE=${HCCOMB:-torch} -e TRITON_CACHE_DIR=/tcache -v $HOME/.cache/n107-triton:/tcache -e EXL3_NVTIER_STAGE_PROF=${STAGEPROF:-0} -e EXL3_NVTIER_STAGE_HOSTBUF=${HOSTBUF:-5} -e EXL3_NVTIER_PREFILL_M=$(( ${MAXRUN:-2} + 1 )) -e EXL3_NVTIER_PROF_EVERY=48 -e EXL3_NVTIER_FILL_THREADS=${FILLTH:-32} \
  -e EXL3_NVTIER=${NVTIER:-1} -e EXL3_NVTIER_STORE=/xfs/qwen_experts.bin -e EXL3_NVTIER_RAM_GB=${RAM_GB:-16} -e EXL3_NVTIER_LAYERS=48 \
  -e EXL3_NVTIER_PRIOR=/n104prior/qwen_expert_prior.json -v $S:/n104prior:ro -e EXL3_NGRAM_TIER=nvme -e SGLANG_EXL3_NGRAM_RAM_GB=${NGRAM_GB:-1} -e SGLANG_EXL3_KV_BITS=5 \
  -e EXL3_QSA_DENSE_BYTES=134217728 -e EXL3_QSA_ROWS=64 -e EXL3_TORCH_MEM_FRACTION=0.95 -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  -v $HOME/models/$MODEL:/models/$MODEL:ro -v /mnt/nvx/n104:/xfs:ro \
  -v $PLG:$P/sglang_plugin.py:ro -v $NVT:$P/nvtier.py:ro \
  -v $S/src/exl3xpu/$MOELIB:/n104/$MOELIB:ro ${VR_ARGS[@]+"${VR_ARGS[@]}"} \
  -v /home/sero/freetoken-exl3/kernels/xpu_bmg/n107-qsa:/n107qsa:ro -v /home/sero/freetoken-exl3/kernels/xpu_bmg/n107-qsa/results:/n107out -v /home/sero/freetoken-exl3/kernels/xpu_bmg/n107-qsa/qsa_xpu_base.py:/opt/trellis-serve/xpu/exl3xpu/qsa_xpu_base.py:ro -v /home/sero/freetoken-exl3/kernels/xpu_bmg/n107-qsa/qsa_xpu_overlay.py:/opt/trellis-serve/xpu/exl3xpu/qsa_xpu.py:ro \
  -e EXL3_QSA_TRITON=${QSATRITON:-0} -e EXL3_QSA_TRITON_VARIANT=${QSAVAR:-row} -e EXL3_QSA_TILE_MAX_CTX=${QSATILECTX:-0} -e EXL3_QSA_TRITON_FP8=${QSAFP8:-bits} \
  ${QSACFG:+-e EXL3_QSA_TRITON_CFG=$QSACFG} ${QSATILECFG:+-e EXL3_QSA_TILE_CFG=$QSATILECFG} -e EXL3_QSA_TRITON_CHECK=${QSACHECK:-0} -e EXL3_QSA_NOSYNC_CHECK=${QSANSCHECK:-0} \
  ${QSAIMPL:+-e EXL3_QSA_IMPL=$QSAIMPL} ${QSASYCLCFG:+-e EXL3_QSA_SYCL_CFG=$QSASYCLCFG} ${QSASYCLMIN:+-e EXL3_QSA_SYCL_MIN_ROWS=$QSASYCLMIN} ${QSASELECT:+-e EXL3_QSA_SELECT=$QSASELECT} ${QSASELCHECK:+-e EXL3_QSA_SELECT_CHECK=$QSASELCHECK} \
  ${QSANOSYNC:+-e EXL3_QSA_NOSYNC=$QSANOSYNC} ${QSAEXPAND:+-e EXL3_QSA_TRITON_EXPAND=$QSAEXPAND} ${QSADUMP:+-e EXL3_QSA_DUMP_DIR=/n107out/dump} ${QSACACHE:+-e TRITON_CACHE_DIR=/n107out/triton-cache-serve} \
  --entrypoint python3 24c872759256 -m sglang.launch_server --model-path /models/$MODEL --quantization exl3 \
  --trust-remote-code --device xpu --host 0.0.0.0 --port $PORT --served-model-name flashnext --disable-shared-experts-fusion \
  --kv-cache-dtype fp8_e4m3 --context-length ${CTX:-65536} --mem-fraction-static ${MEMFRAC:-0.85} --chunked-prefill-size ${CHUNK:-4096} --max-total-tokens ${MAXTOT:-262144} \
  --max-running-requests ${MAXRUN:-2} ${LINATTN:+--linear-attn-backend $LINATTN} ${NORADIX:+--disable-radix-cache} ${PAGE:+--page-size $PAGE} --dtype bfloat16 --max-mamba-cache-size ${MAMBA:-8} ${MSSM:+--mamba-ssm-dtype $MSSM} --cuda-graph-backend-decode full --cuda-graph-bs-decode ${GRAPH_BS:-1 2} \
  > "$RUN/log.txt" 2>&1

# 18. How fast can the B70 process the expert data coming in? (N106 synthesis)

Model: Qwen3.8-Flash-Next EXL3 3.05bpw (turboderp, h5_ng5), SGLang + exl3xpu, NVMe expert tier (N104 store), B70 at 0000:84:00.0.
Synthesis of the N106 map and measure phases; no GPU used for this note. Date 2026-10-06.

Sources (all on omarchy under `~/freetoken-exl3/runs/N106-ingest/`): `model-math/`, `hw-ceilings/`, `code-path/`,
`moe-bench/` (GPU microbenchmarks on 84:00.0), `ngram-bench/` (CPU only), `prefill-profile/` (guard tripped, no data).
"Measured today" server numbers come from the c3 B70 (now retired) with the srv17 config: STAGE=1, RAM tier 7 GB, KV 131072,
chunk 8192, maxrun 4, mamba 32, 8000 VRAM slots. The 84:00.0 microbenchmark reproduces the in-server MoE time (1.24 s vs
1.2 s per 8k chunk), so the per-component costs carry over. Calculator: `runs/N106-ingest/synthesis/ceil.py`.

## Answer

- **Prefill is compute-bound, and not by the experts.** The MoE kernel can take in 33-37 GB/s of distinct expert data at an
  8k chunk, faster than NVMe (26 GB/s) or the copy engine (26.7-28.4 GB/s) can deliver. A chunk needs only 1.1-1.2 s of
  link time, but the GPU spends 8.2-8.6 s on it at best (9.7-12.5 s measured). 7.0-7.4 s of that is non-MoE work: QSA
  attention, GDN, hyper-connections and the dense EXL3 linears.
- **The one ingest cost that does show up in prefill is a runtime bug, not bandwidth.** In one process, side-stream H2D copies
  and compute never overlap; across two processes they overlap almost perfectly (each slows by under 1%). The staged prefill
  most likely pays the 1.1-1.5 s of copies per chunk on top of compute.
- **Decode is part latency-bound, part link-stalled.** Experts already in VRAM cost only 3.0, 5.8 and 10.0 ms per step at
  B=1, 2 and 4. The rest of a 46-136 ms step splits three ways:
  - non-MoE kernels: about 26.5 ms at C1;
  - RAM-tier expert reads that the kernel performs synchronously over SVM: about 12.6, 25 and 48 ms at C1, C2 and C4 (derived);
  - a host bubble of 3.5-4 ms.
- **With scheduling alone (no new kernels), the ceilings are:**
  - prefill: about 950-1,000 tok/s at 8k and 800-830 tok/s at 32k;
  - decode: about 34 tok/s at C1, about 57 tok/s aggregate at C2, and about 83 tok/s aggregate at C4, where the host link caps
    it at today's 64% VRAM hit rate.

## 1. Ceiling table

Each cell reads **measured today / GPU-bound ceiling / limiting component**. "GPU-bound ceiling" means perfect scheduling and
zero I/O stalls, given the per-component costs measured today and today's kernels. Decode is shown as per stream and total.
Prefill is the aggregate (prompt tokens ÷ time until the last first token).

| cell | decode per stream (tok/s) | decode total (tok/s) | prefill (tok/s) |
|---|---|---|---|
| 8k C1 | 21.8 / **~34** / non-MoE decode kernels (26.5 of 29.5 ms per step) | 21.8 / **~34** / same | 655 / **950-1,000** / non-MoE prefill GPU time (7.0-7.4 s of an 8.2-8.6 s chunk) |
| 8k C2 | 13.9 / **~28** (23-31) / non-MoE decode kernels | 27.8 / **~57** (46-62) / non-MoE decode kernels; link cap 81 | 785 / **950-1,000** / non-MoE prefill GPU time (one compute engine, so chunks run back to back) |
| 8k C4 | 7.4 / **~21** (12-21) / host link: RAM-tier expert misses | 29.5 / **~83** (48-83) / host link: 0.32 GB per token at 64% VRAM hit, 26.7 GB/s | 843 / **950-1,000** / non-MoE prefill GPU time |
| 32k C1 | 23.3 / **34-38** / non-MoE decode kernels | 23.3 / **34-38** / same | 687 / **800-830** / non-MoE GPU, with the QSA union growing per chunk (+10.7 s per 32k prompt) |
| 32k C2 | 13.9 / **~28** (23-30) / non-MoE decode kernels | 27.7 / **~57** (46-60) / non-MoE decode kernels; link cap 81 | 721 / **800-830** / same as 32k C1 |

How the ceilings were derived (numbers in `ceil.py`):
- **Prefill at 8k.** The best measured per-chunk wall time is 9.72 s, from 8k C4: 32,768 tokens in 38.9 s. From that I
  remove the in-process serialized copy: 30.9 GB ÷ 28.4 GB/s = 1.09 s, up to the 1.5 s of copy-engine busy time seen in
  the server. MoE stays at 1.24 s and non-MoE is 6.98-7.39 s, so a chunk takes 8.22-8.63 s, which is 949-997 tok/s. This
  agrees with the all-RAM B70 reference of about 1k tok/s. The C1 cell loses another 2.8 s per request (12.5 s vs 9.7 s per
  chunk). That time is per-request overhead, not GPU work for the chunk (see unknowns).
- **Prefill at 32k.** The best measured per-prompt wall is 45.66 s (32k C2). Minus 4 × 1.09-1.5 s of copies, that is
  39.7-41.3 s, or 797-830 tok/s. Using the srv18 GPU wall instead (46.7 s) gives 777-809 tok/s; that is the `ceil32` value in `ceil.py`. Most of the gap to 8k is the QSA union ramp: per-chunk "other" time went
  7.8 → 9.3 → 11.0 → 13.8 s in srv18.
- **Decode, the per-step model.** step = host bubble (3.5-4 ms) + non-MoE + resident MoE (3.0, 5.8 and 10.0 ms at B=1, 2
  and 4, from moe-bench) + RAM-tier SVM reads. The RAM-tier reads are 0.894 GB per token × B × (1 − cross-stream
  sharing) × 36% miss ÷ 25.5 GB/s, which gives 12.6, 24.8 and 47.9 ms.
  - At C1 this leaves 26.5 ms of non-MoE. That fits the C1 step being 10-14 ms slower than the all-RAM reference with nothing
    else to account for it.
  - The ceiling removes the bubble and the SVM stall. For C2 and C4, the central value assumes non-MoE grows by 1-3 ms per
    extra stream. The dense weights are read once per step; each extra stream adds about 231 MB of GDN state traffic plus
    GEMV rows.
  - The low end of each range assumes all of the measured residual is non-MoE GPU work: 37.6 ms at C2 and 73.9 ms at C4.
  - C4 is capped by the host link at about 83 tok/s aggregate unless the VRAM hit rate improves.

Kernel and hardware ceilings, beyond scheduling:

| bound | 8k prefill | decode C1 / C2 / C4 aggregate |
|---|---|---|
| non-MoE at the MoE kernel's efficiency plus memory-bound work at spec bandwidth (model-math) | ~1,570 tok/s | – |
| MoE kernel alone (moe-bench, M=8192, 25.8 ms per layer) | 6,618 tok/s | 330 / 344 / 400 |
| host link alone (30.9 GB per chunk at 26.7 GB/s) | ~7,085 tok/s | 79 / 81 / 83 at a 64% VRAM hit rate |
| VRAM byte floor (608 GB/s spec, not measured on 84:00.0) | – | 124 / 201 / 295 |

## 2. Expert-data ingest: what the GPU consumes vs what the I/O path delivers

| phase | GPU can consume | GPU actually pulls (time-averaged) | supply | verdict |
|---|---|---|---|---|
| prefill, 8k chunk | **33-37 GB/s** of distinct expert data: all 512 experts per layer, 953 MB per layer in 25.8 ms (DPAS kernel; 200-250 GB/s of trellis-decoded blocks internally; plateau about 31-32 TFLOPS, 17% of XMX peak) | 30.9 GB per chunk ÷ 12.5 s = **2.5 GB/s** today at C1; 3.6-3.8 GB/s at the scheduling ceiling; 5.9 GB/s at the 1,570 tok/s kernel ceiling | NVMe RAID0 26 GB/s (1.19 s per chunk); copy engine 26.7 GB/s from anon THP, 28.4 GB/s from pinned USM (1.09-1.16 s per chunk); in-server copy engine busy 1.5 s at about 20 GB/s | **compute-bound**. The link is about 9-12% busy averaged over a chunk. Even during the MoE window alone, link (1.16 s) and kernel (1.24 s) are balanced. I/O would only bind above about 7k tok/s. |
| prefill, 16k chunk | 18-19 GB/s needed: the same experts amortised over twice the rows | – | same | compute-bound, with twice the margin |
| decode | **280-350 GB/s** from VRAM (`moe_forward_cached`, 50-57% of 608 GB/s spec); 3.0, 5.8 and 10.0 ms per step at B=1, 2 and 4 | 0.894 GB of experts per token, of which 0.32 GB comes over the link at a 64% VRAM hit rate: **7.0, 8.8 and 9.0 GB/s** at the measured C1, C2 and C4 rates (26-34% of the link) | the same 26.7 GB/s link; NVMe sees only about 11 picks per step at C2 (srv18: 11.0-11.1 NVMe and 12.2 masked picks per step, about 20 MB per step) | **latency-bound plus link-stalled**. The link is not saturated on average, but RAM-tier picks are read synchronously inside the MoE kernel over SVM: about 12.6 ms of a 45.9 ms step at C1, about 25 of 72 ms at C2 and about 48 of 136 ms at C4 (derived). At C4 the link becomes the binding ceiling (about 83 tok/s aggregate) once the non-MoE decode is fast. |

- **Prefill.** The staged pipeline (NVMe → anon THP → copy engine → VRAM double buffer) has enough bandwidth already: host
  reader-wait is 31 ms per chunk and the measured wait on H2D is 0-4 ms. The only ingest-side loss is the in-process
  copy/compute serialisation measured by moe-bench (copies-first: wall 606 ms vs a serial sum of 605 ms; cross-process:
  MoE +0.6%, H2D −0.7%). That costs about 1.1-1.5 s per chunk, which is 10-12% of the chunk. The stage profiler hides this
  inside "OTHER", because the compute queue is held while the copies run.
- **Decode.** This is the phase where moving data costs speed. The stall is not bandwidth: these bytes cross the link
  on the critical path, one layer at a time, inside the kernel.
- **NVMe masking** (12 picks per step at C2) costs quality, not speed.

## 3. Ranked changes and expected gains

Gains are the new tok/s per cell, starting from today's measured values. Each change is tied to the measured number that
sizes it. Gains from different changes do not simply add up; combined effects are listed where they were computed.

| # | change | sized by (measured) | prefill 8k C1 / C2 / C4 / 32k C1 / 32k C2 | decode total 8k C1 / C2 / C4 / 32k C1 / 32k C2 | confidence |
|---|---|---|---|---|---|
| 1 | **Take RAM-tier expert reads off the decode critical path.** Options: raise the VRAM hit rate (more slots in decode; reuse the prefill staging double buffer and unused KV for slots), or prefetch likely RAM-tier picks into slots with the copy engine from a separate L0 context or process, one layer ahead. | 0.32 GB per token at 64% hit ÷ 25.5 GB/s SVM = 12.6 ms per step at C1, which matches the unexplained +10-14 ms per step vs all-RAM (code-path) | – | 21.8→**30.1** / 27.8→**42.4** / 29.5→**45.6** / 23.3→**33.0** / 27.7→**42.2** (all stalls removed). Each +1% of VRAM hit removes about 0.35 ms per step per stream. | medium: derived; needs the RAM-tier-pick counter |
| 2 | **QSA prefill: replace the dense fp32 union attention with a bf16, fused or truly sparse (2,048-key) path**, and drop the `.item()`, `unique` and `nonzero` syncs | per-chunk "other" ramps 7.8→13.8 s over 4 chunks (+10.7 s per 32k prompt); model-math: 19.8 TFLOP and 0.93 TB of fp32 scores at 8k vs 4.3 TFLOP truly sparse | 8k: 712-780 / 868-971 / 940-1,061 (if 1-2 s of the estimated 2-3.5 s is recovered); 32k: **884 / 942** (ramp only), **1,002 / 1,076** together with #3 | – | high at 32k (measured ramp); low at 8k (estimate) |
| 3 | **True copy/compute overlap in staged prefill**: issue expert H2D from a separate process, a separate SYCL/L0 context, or a copy-only queue that does not hold ccs0 | in-process overlap fully serialised (606 vs 605 ms); cross-process MoE +0.6%, H2D −0.7%; 30.9 GB per chunk at 28.4 GB/s = 1.09 s (1.5 s busy in the server) | **717-744 / 876-917 / 949-997 / 756-786 / 797-831** | – | medium-high: the mechanism is measured, its presence in the server is inferred |
| 4 | **Remove the per-step `torch.xpu.synchronize()` and the ~380 KB of device-to-host reads in `NvTierFast.step()`**: use an event and async reads into pinned buffers | measured "outside" 2.3-3.1 ms plus host work 0.7-1.2 ms per step | – | 23.6-23.9 / 29.2-29.4 / 30.3-30.4 / 25.4-25.7 / 29.1-29.3. **#1 and #4 together: 33.9 / 46.1 / 47.6 / 37.7 / 45.8** | high |
| 5 | **Non-MoE prefill kernels**: GDN via `--linear-attn-backend intel_xpu` (fused `gdn_attention`, never tried), a single-pass grouped RMSNorm and hyper-connection fusion on XPU, `exl3xpu_C.linear` efficiency at M=8192 | 6.98-7.39 s of non-MoE per 8k chunk (78.9 TFLOP at about 7-9 TFLOPS effective, vs 31-32 TFLOPS for the MoE kernel) | pool up to **~1,570** at 8k (model-math); per-component split not measured | decode non-MoE is 26.5 ms per step against an 8 ms byte floor; hyper-connection weights fp16→int8 would cut 0.63 GB per step | low until the events-mode profile runs |
| 6 | **Larger DPAS row blocks (MB=64 or 128)** in the prefill MoE kernel | time scales with the number of 32-row blocks: t ≈ 0.7 ms + blocks × 1.86 MB ÷ ~215 GB/s; projected 1.24 → ~0.7 s per 8k chunk | 682 / 825 / 889 / 717 / 754 | – | medium: a projection from a fitted line |
| 7 | **N-gram prefetch**: call `install_prefill_hints()`, set `eos_token_id`, warm rows when a request arrives; optionally move the table to the RAID0 (about 2x faster) | bench 8k C1 chunk 46 ms; natural text 0.14-0.34 s per cold chunk; worst case 0.45 s; decode 0.45-0.9 ms per step | bench cells 655→657 (others about 0); natural text 8k C1 662-673 | 22.0 / 28.1 / 29.7 (bench) | high (CPU measured) |
| 8 | **Prefill chunk 16384** | expert ingest per token halves (37→19 GB/s needed); MoE 6.8k vs 6.6k tok/s; two copies saved per 32k prompt | 8k: none at C1 (C2/C4 could pack 2 prompts per forward); 32k C1 720-733 with copies still serialised | decode stalls twice as long during prefill | low-medium: QSA union at 16k rows and VRAM headroom unknown |
| 9 | **Interleave decode with prefill**: smaller chunks or decode steps between prefill chunks. There is only one compute engine (ccs0). | srv18 32k C2: one stream's decode stalled 31.1 s (per-stream decode 9.07 vs 13.73 tok/s) | no aggregate gain | per-stream decode of the stalled stream 9.1 → ~13.7; inter-token latency max 31 s → ~1 chunk | high on latency, zero on throughput |
| 10 | **`prefill_min_m`=256 when experts are VRAM-resident or staged** (GEMV for 5-255 rows) | GEMV 1.35-1.8x faster than DPAS at M=8-128 | 0 on these cells (8k chunks are 8,192 rows; decode M ≤ 4 already uses GEMV) | 0 | high; matters for short prompts and future spec-decode verify |

Summary of what scheduling (no new kernels: #1, #3, #4, #7, #9) can still win:
- **prefill:** 8k C1 +45-52%, 8k C4 +13-18%, 32k +11-21%. The fixes identified so far (#3, #7) take 8k C1 only to about
  720-745 tok/s. The rest of the C1 gap is the 2.8 s per-request overhead in unknown 6.
- **decode:** C1 +55%, C2 about +65-105%, C4 up to the link cap of about 83 aggregate, but only with a better VRAM hit rate or
  overlapped prefetch.

Beyond that, the remaining lever is the non-MoE kernels (#2, #5, #6). They take 8k prefill from about 1k to about 1.5k tok/s
before expert ingest matters at all.

## 4. Unknowns that still need measuring

1. **The per-component split of non-MoE prefill (7.0-7.4 s per 8k chunk)** across QSA, GDN, hyper-connections, dense EXL3
   and launch gaps. The events and trace runs (`code-path/n106_serve_prof.sh`, `prefill-profile/pp_*.sh`) never ran because
   omarchy went down during the launch. This is the largest open item; it decides the order of #2 and #5.
2. **Whether the server's staged prefill really serialises the copies.** One check is xe fdinfo on bcs and ccs during a
   STAGE=1 chunk. Another is an A/B of STAGE=1 against a chunk whose experts are all VRAM-resident.
3. **RAM-tier picks per decode step and the true VRAM hit rate at C1, C2 and C4** (a device counter in
   `moe_forward_cached`). This decides whether the extra residual at C2 (+11 ms) and C4 (+47 ms) is link stall or growth in
   non-MoE work, which is why the C2 and C4 decode ceilings span 46-62 and 48-83.
4. **The non-MoE decode split** (26.5 ms per step: GDN decode, QSA decode, hyper-connections, dense GEMV, lm_head). Use trace
   mode with PDEC.
5. Microbenchmarks at M=8192 on 84:00.0: `exl3xpu_C.linear` for the six main shapes, Triton GDN vs `gdn_attention`, QSA
   union at nu of 8k, 16k and 32k (and log the real nu per layer), and the grouped norm.
6. **The per-request overhead at C1**: 12.5 s TTFT vs 9.7 s per chunk at C4, and stage-profile GPU wall of 9.8-11.9 s at C1.
   Also the anomaly of 6.1-6.5 s of "other" at 942 rows, possibly first-of-shape compiles. Both matter for short-prompt
   TTFT.
7. **The 5-cell server matrix rerun on 84:00.0.** Today's server numbers are from c3, which is retired. VRAM bandwidth (608
   GB/s), XMX peak and D2H have not been measured on 84:00.0.
8. **Routing realism.** The bench prompts use a 10-word vocabulary. Expert hit rates, cross-stream sharing and n-gram costs on
   natural text may differ (the n-gram cost is already 4x higher on natural text).
9. **Whether a separate-context copy path (#3) can live inside the SGLang process** (a separate L0 context or copy-only queue)
   or needs a helper process with shared USM/IPC buffers.
10. **Box health.** omarchy went silent at about 15:42 CEST during the prefill-profile launch, after a stuck
    khugepaged/kcompactd THP collapse that blocked page faults on `libtorch_cpu.so`, `libtorch_xpu.so` and `libcommon_clang`.
    The cause is undetermined. It must be rebooted and checked (dsv41-lab health, `journalctl -k -b -1`, containers `n106-pp`,
    tmux `n106pp`) before any further GPU jobs.

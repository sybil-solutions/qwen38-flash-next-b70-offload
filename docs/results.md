# Results

All numbers: one Arc Pro B70 32 GB (PCIe 4.0 x16), AMD EPYC 7443P, 4x Samsung 9100 PRO 1 TB in md RAID0 + XFS
(fio 26 GB/s), SGLang 0.5.20-xpu + exl3xpu. Each stream gets its own prompt; completions run to their natural end
(no output caps). Decode at 2 or 4 streams is the total across streams.

## N130: 16 GB vs 32 GB, same session (2026-10-08)

Card 0000:48:00.0, container pinned to 8 CPUs (40-47), NVMe reads uncapped, another job (an RTX 3090 run) was using
the box at the same time. Raw files: [`results/n130/`](../results/n130/).

### 16 GB (`QWEN_B70_MODE=nvme16`, `--memory 16g --memory-swap 16g`)

RAM tier 2 GB, 2-buffer prefill ring, admission-controlled decode tier, 8 decode-fill queues.

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,139 tok/s | 29.2 | 1 | 131k fp8 (1.5 GiB) | 1 |
| 8k | 1,154 tok/s | 38.7 total (19.3-20.1 per stream) | 2 | 131k fp8 (1.5 GiB) | 1 |
| 8k | 1,301 tok/s | 52.6 total (11.2-13.2 per stream) | 4 | 131k fp8 (1.5 GiB) | 1 |
| 32k | 1,184 tok/s | 30.6 | 1 | 131k fp8 (1.5 GiB) | 1 |
| 32k | 1,332 tok/s | 41.3 total (14.6-20.9 per stream) | 2 | 131k fp8 (1.5 GiB) | 1 |

### 32 GB (`QWEN_B70_MODE=nvme32` = srv23 settings, `--memory 32g`), same-session control

| prefill size | prefill speed | decode speed | concurrency | kv cache | gpu count |
|---|---|---|---|---|---|
| 8k | 1,135 tok/s | 25.6 | 1 | 131k fp8 (1.5 GiB) | 1 |
| 8k | 1,143 tok/s | 30.5 total (15.2-15.9 per stream) | 2 | 131k fp8 (1.5 GiB) | 1 |
| 8k | 1,290 tok/s | 32.4 total (7.5-8.1 per stream) | 4 | 131k fp8 (1.5 GiB) | 1 |
| 32k | 1,193 tok/s | 27.3 | 1 | 131k fp8 (1.5 GiB) | 1 |
| 32k | 1,321 tok/s | 31.3 total (12.6-15.7 per stream) | 2 | 131k fp8 (1.5 GiB) | 1 |

### NVMe reads (md127, % of the 26 GB/s array)

16 GB:

| cell | during prefill | during decode |
|---|---|---|
| 8k C1 | 4.11 GB/s (15.8 %), peak 6.62, 33,476 IOPS | 0.81 GB/s (3.1 %), peak 1.36, 7,054 IOPS |
| 8k C2 | 4.37 GB/s (16.8 %), peak 6.9, 35,647 IOPS | 1.42 GB/s (5.5 %), peak 2.49, 12,311 IOPS |
| 8k C4 | 4.93 GB/s (19.0 %), peak 6.96, 40,218 IOPS | 2.31 GB/s (8.9 %), peak 3.15, 20,025 IOPS |
| 32k C1 | 5.35 GB/s (20.6 %), peak 8.93, 43,586 IOPS | 0.75 GB/s (2.9 %), peak 1.3, 6,536 IOPS |
| 32k C2 | 5.58 GB/s (21.5 %), peak 9.47, 45,502 IOPS | 1.34 GB/s (5.1 %), peak 2.1, 11,577 IOPS |

32 GB:

| cell | during prefill | during decode |
|---|---|---|
| 8k C1 | 4.46 GB/s (17.2 %), peak 6.44, 36,373 IOPS | 0.34 GB/s (1.3 %), peak 0.69, 2,966 IOPS |
| 8k C2 | 4.44 GB/s (17.1 %), peak 6.85, 36,175 IOPS | 0.49 GB/s (1.9 %), peak 0.9, 4,208 IOPS |
| 8k C4 | 4.97 GB/s (19.1 %), peak 6.78, 40,477 IOPS | 0.7 GB/s (2.7 %), peak 1.36, 6,060 IOPS |
| 32k C1 | 5.12 GB/s (19.7 %), peak 8.77, 41,726 IOPS | 0.36 GB/s (1.4 %), peak 1.19, 3,100 IOPS |
| 32k C2 | 5.63 GB/s (21.7 %), peak 9.17, 45,903 IOPS | 0.42 GB/s (1.6 %), peak 1.41, 3,623 IOPS |


Prefill is GPU-bound, not NVMe-bound: per 8k forward the GPU spends 4.2 of 5.4 s in attention / GDN / dense work and
0 ms waiting on the copy engine (H2D 25.7 GB/s); the reader pool waits 30-39 ms in total. A deeper read-ahead ring
(5 buffers instead of 2) does not change prefill speed. Decode reads are small (113 KB requests, the md layer splits
the 1.86 MB records) and stay under 10 % of the array.

### Memory (16 GB cap)

| | GiB |
|---|---|
| process anon (scheduler 6.6, launcher 1.3, detokenizer 1.0) | 9.4-9.5 |
| RAM expert tier (memfd, 2 MiB pages) | 2.0-2.1 |
| prefill host ring (2 x 0.91, shared anonymous) | 1.8 |
| peak anon + shmem during the run | 14.3 |
| swap | 0 |

`memory.current` sits at the cap because of page cache, which the kernel drops as needed. Stopping the container
(`docker stop`) triggers one OOM kill during teardown in both modes: the exiting scheduler faults in part of the sparse
48 GB expert mapping. It happens after the last request and does not affect serving.

### Quality

Neither mode is bit-exact: decode masks expert picks that are in neither VRAM nor RAM yet (weight 0) and fills them in
the background.

| check | 16 GB | 32 GB (srv23) |
|---|---|---|
| reference panel, teacher-forced (prefill path) vs exllamav3, top-1 / KL, fresh server | 0.985 / 0.0030 | 0.984 / 0.0028 |
| same, second pass on the same server | 0.979 / 0.0076 | 0.982 / 0.0085 |
| decode vs prefill on the same server, KL / top-1 (panel items 0-5, greedy to EOS) | 0.055 / 0.938 | 0.021 / 0.963 |
| greedy panel equal to the exllamav3 text | 0 / 8 | 0 / 8 |
| masked picks per decode step during the table | 18-135 | 5-60 |

- The prefill-path deviation (top-1 ~0.985, KL ~0.003, worse on a second pass) is the same at 32 GB, so the 16 GB
  settings do not cause it. It misses the kit band (top-1 >= 0.987, KL <= 0.0013) in both modes.
- The extra decode loss at 16 GB comes from the 2 GB RAM tier (`RAM_GB=2`): more picks are masked. Admission control
  and the decode-fill queues are what make a 2 GB tier work at all; they were not tested separately at 32 GB.
- With the stock tier code, a 2 GB RAM tier collapses: queued fills hold the whole budget, every finished fill is
  evicted before the GPU reads it, ~470 of 480 picks per step are masked and the output is garbage (N130 s16a/s16b).
  This is also what broke the first N110 quality run at 7 GB.
- Right after start the VRAM cache is empty; the first answer can be poor (one cold run ended a code answer after
  4 tokens). Send one or two warm-up requests after start.

Methods: `kit/tools/score_ref_panel.py` (top-20 KL on the reference support) after `/flush_cache`;
[`serve/n130/decode_kl_n130.py`](../serve/n130/decode_kl_n130.py) (greedy decode with top-20 logprobs, flush, then the
same ids teacher-forced through the prefill path). The 32 GB decode check stopped after item 5 on an SGLang error
(`token_ids_logprob` in an overlap decode batch), so both columns use items 0-5.

## 32 GB history (srv20-srv26, 2026-10-06, card at 84:00.0)

srv23, the current default. Each stream gets its own prompt (8,181–8,189 or 32,928–32,932 tokens). Completions run to
their natural end. Decode at 2 or 4 streams is the total across streams.

| prefill length | concurrency | decode tok/s | prefill tok/s | time to first token |
|---|---|---|---|---|
| 8k | 1 | 24.0 | 1,056 | 7.8 s |
| 8k | 2 | 28.7 | 1,092 | 14.9 s |
| 8k | 4 | 26.4 | 1,233 | 13.6–26.6 s |
| 32k | 1 | 23.9 | 1,106 | 29.8 s |
| 32k | 2 | 25.4 | 1,260 | 34.9 / 52.3 s |

#### What each change bought

Each cell is decode tok/s / prefill tok/s. Every row adds one change to the row above.

| run | change | 8k C1 | 8k C2 | 8k C4 | 32k C1 | 32k C2 |
|---|---|---|---|---|---|---|
| srv20 | baseline: staged NVMe prefill | 20.9 / 693 | 25.1 / 786 | 24.1 / 868 | 21.9 / 716 | 22.0 / 734 |
| srv21 | + async decode step (no per-token device sync) | 22.6 / 717 | 26.3 / 784 | 24.7 / 870 | 23.8 / 713 | 22.4 / 741 |
| srv22 | + fused hyper-connection kernels (Triton, `kernels/n107-hc`) | 23.1 / 799 | 27.8 / 883 | 26.8 / 960 | 24.1 / 762 | 23.0 / 752 |
| **srv23** | **+ SYCL sparse attention and selection (`kernels/n107-qsa`)** | **24.0 / 1,056** | **28.7 / 1,092** | **26.4 / 1,233** | **23.9 / 1,106** | **25.4 / 1,260** |
| srv25 | srv23 with the radix cache off (control) | 23.8 / 1,026 | 27.7 / 1,095 | 26.8 / 1,220 | 24.4 / 1,127 | 24.5 / 1,272 |

Raw results are under [`results/`](../results/).

- **Sparse attention:** at 8,192 query rows the SYCL kernel takes 55.9 ms per layer at 8k context and 65.5 ms at 32k.
  The previous union path took 255 and 459 ms. The selection kernels pick the same blocks as the torch path on every
  checked row, both in tests and inside the server.
- **Hyper-connections:** the fused kernels cut the work per 8k forward from 945 ms to 366 ms.
- **Rejected:** SGLang's fused GDN backend (`--linear-attn-backend intel_xpu`). At 4 streams, 3 of 4 outputs ran away to
  ~57k tokens.

### Experimental: no-write-back decode

srv26 = srv23 + `NOVICT=1`. The decode kernel no longer writes VRAM-evicted experts back to the RAM tier. Each
write-back costs ~0.9 ms on the critical path ([`results/N109-svm-vs-usm`](../results/N109-svm-vs-usm/)).

| prefill length | concurrency | decode tok/s | prefill tok/s | token latency p50 |
|---|---|---|---|---|
| 8k | 1 | 34.5 | 1,018 | 27 ms |
| 8k | 2 | 54.8 | 1,013 | 34 ms |
| 8k | 4 | 76.3 | 1,123 | 44 ms |
| 32k | 1 | 34.1 | 1,120 | 28 ms |
| 32k | 2 | 42.9 | 1,136 | 35 ms |

Masked expert picks rise from 4–12 to ~32–38 per decode step, about 7% of picks at 1 stream. That is an
approximation, so it stays off by default until the decode quality gate passes. The fix that keeps exact output (an
async victim ring drained by the copy engine) is in progress.

### Experimental (validated standalone, not yet in the server)

| folder | what | status |
|---|---|---|
| [`experimental/n111-victim-ring`](../experimental/n111-victim-ring/) | decode kernel parks VRAM victims in a VRAM ring (µs) instead of storing to host (~0.9 ms each). Host drains the ring through pinned staging, and only the host sets the RAM-resident flag (`_moe_a9`/`a9b` patches, `VRING=1`) | bit-exact 72/72 calls; stress simulation 0 mismatches; +0.2–0.5 ms/step vs +9–30 ms for today's write-back |
| [`experimental/n112-gdn`](../experimental/n112-gdn/) | SYCL token-serial GDN prefill recurrence ported from Strata (MIT), drop-in for SGLang's `chunk_gated_delta_rule` extend call (`EXL3_GDN_SYCL=1`) | CPU-validated against fp64: output rel 1.66e-3, final state 4e-9, chunk continuation bit-identical. GPU test pending |
| [`experimental/n114-decode`](../experimental/n114-decode/) | fused decode kernels from Strata: GDN step (6–7 → 2 kernels/layer), hyper-connection read (~7 → 3–4/half), decode QSA (~45 → 2/layer), graph-safe | builds; CPU tests pass. GPU test pending |
| [`experimental/n113-mtp`](../experimental/n113-mtp/) | MTP self-speculation (SGLang NEXTN) in tier mode: MTP experts pinned in VRAM, safe verify rows | plan + plugin override written; projected 32–43 tok/s exact at 1 stream. Not run yet |

### Known issues

- **Write-back race in the default tier path.** In a stress simulation (N111), the decode kernel's direct victim
  write-back produced 50–52 mismatched output rows and 8–10 bad RAM pages. The GPU marks a written-back expert
  resident while the host still has that page queued for punching. The victim ring removes the race because only the
  host sets the flag. Until it lands, treat long greedy runs on the default path with care. Runaway generations were
  observed once in the exact config.
- **Hardware.** On the test box, PCIe links are marginal under load. Both B70s dropped off the bus once on 2026-10-06,
  and the box needed a BMC power cycle. None of the measurements above were taken during a fault window.


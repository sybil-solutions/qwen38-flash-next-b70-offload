# qwen38-flash-next-b70-offload

![Qwen3.8-Flash-Next on one Arc Pro B70 with experts on NVMe](docs/banner.svg)

## At a glance

| | |
|---|---|
| Model | Qwen3.8-Flash-Next, EXL3 3.05 bpw ([`turboderp/Qwen3.8-Flash-Next-exl3`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) @ `3.05bpw_h5_ng5`): 48 MoE layers × 512 routed experts, top-10, 45.8 GB of experts, ~32.6 GB n-gram (PLE) table |
| Hardware | 1× Intel Arc Pro B70 32 GB (PCIe Gen4 x16), **host RAM capped at 32 GB**, 4× Samsung 9100 PRO 1 TB in RAID0 (26 GB/s), AMD EPYC 7443P (14 cores used) |
| Decode | 24.0 / 28.7 / 26.4 tok/s at 1 / 2 / 4 streams (8k prompts), 23.9 / 25.4 at 1 / 2 streams (32k prompts) |
| Prefill | 1,056–1,233 tok/s at 8k, 1,106–1,260 tok/s at 32k |
| How | 8,000 hot experts cached in VRAM, a 7 GB RAM expert tier, everything else read from NVMe. Prefill streams each layer's experts NVMe → RAM → copy engine → VRAM, 4 layers ahead |
| Experimental | Decode without the VRAM→RAM victim write-back: **34.5 / 54.8 / 76.3 tok/s** at 1 / 2 / 4 streams. More expert picks are masked; a quality gate is running (see [Experimental](#experimental-no-write-back-decode)) |
| Status | Research build, measured end to end on 2026-10-06 |

Serve **Qwen3.8-Flash-Next** with an OpenAI-compatible API from **one Intel Arc Pro B70** and **32 GB of host RAM**,
with the routed experts on NVMe. SGLang + the [`exl3xpu`](https://github.com/0xSero/exl3xpu) plugin (EXL3 trellis
kernels for Xe2) + an NVMe expert tier:

- **Decode:** a device-managed VRAM expert cache (8,000 slots). Picks that miss VRAM read the RAM tier (memfd, 2 MiB
  pages, read by the GPU through xe SVM); picks in neither are masked and filled from NVMe in the background.
- **Prefill:** big batches stream every non-VRAM expert of a layer from the RAID0 (O_DIRECT, 32 threads) into an
  anonymous THP buffer, then the copy engine moves it into a VRAM double buffer while the previous layer computes.
- **Full-attention layers:** sparse attention with per-query top-2048 selection, using SYCL kernels ported from
  [Strata](https://github.com/Niko1221/Strata) (MIT) and re-tuned for our fp8 KV.

## Measured

srv23, the current default. Each stream gets its own prompt (8,181–8,189 or 32,928–32,932 tokens). Completions run to
their natural end. Decode at 2 or 4 streams is the total across streams.

| prefill length | concurrency | decode tok/s | prefill tok/s | time to first token |
|---|---|---|---|---|
| 8k | 1 | 24.0 | 1,056 | 7.8 s |
| 8k | 2 | 28.7 | 1,092 | 14.9 s |
| 8k | 4 | 26.4 | 1,233 | 13.6–26.6 s |
| 32k | 1 | 23.9 | 1,106 | 29.8 s |
| 32k | 2 | 25.4 | 1,260 | 34.9 / 52.3 s |

### What each change bought

Each cell is decode tok/s / prefill tok/s. Every row adds one change to the row above.

| run | change | 8k C1 | 8k C2 | 8k C4 | 32k C1 | 32k C2 |
|---|---|---|---|---|---|---|
| srv20 | baseline: staged NVMe prefill | 20.9 / 693 | 25.1 / 786 | 24.1 / 868 | 21.9 / 716 | 22.0 / 734 |
| srv21 | + async decode step (no per-token device sync) | 22.6 / 717 | 26.3 / 784 | 24.7 / 870 | 23.8 / 713 | 22.4 / 741 |
| srv22 | + fused hyper-connection kernels (Triton, `kernels/n107-hc`) | 23.1 / 799 | 27.8 / 883 | 26.8 / 960 | 24.1 / 762 | 23.0 / 752 |
| **srv23** | **+ SYCL sparse attention and selection (`kernels/n107-qsa`)** | **24.0 / 1,056** | **28.7 / 1,092** | **26.4 / 1,233** | **23.9 / 1,106** | **25.4 / 1,260** |
| srv25 | srv23 with the radix cache off (control) | 23.8 / 1,026 | 27.7 / 1,095 | 26.8 / 1,220 | 24.4 / 1,127 | 24.5 / 1,272 |

Raw results are under [`results/`](results/).

- **Sparse attention:** at 8,192 query rows the SYCL kernel takes 55.9 ms per layer at 8k context and 65.5 ms at 32k.
  The previous union path took 255 and 459 ms. The selection kernels pick the same blocks as the torch path on every
  checked row, both in tests and inside the server.
- **Hyper-connections:** the fused kernels cut the work per 8k forward from 945 ms to 366 ms.
- **Rejected:** SGLang's fused GDN backend (`--linear-attn-backend intel_xpu`). At 4 streams, 3 of 4 outputs ran away to
  ~57k tokens.

### Experimental: no-write-back decode

srv26 = srv23 + `NOVICT=1`. The decode kernel no longer writes VRAM-evicted experts back to the RAM tier. Each
write-back costs ~0.9 ms on the critical path ([`results/N109-svm-vs-usm`](results/N109-svm-vs-usm/)).

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

## Hardware used

| part | detail |
|---|---|
| GPU | Intel Arc Pro B70 32 GB (Battlemage G31), Gen4 x16. VRAM holds 8,000 expert slots (14.9 GB), fp8 KV cache capped at 131,072 tokens (1.5 GB), mamba state 32 slots (3.8 GB) and 2 × 0.95 GB staging buffers |
| Host RAM | Container capped at 32 GB (`--memory 32g`), peak use ~22 GiB; RAM expert tier 7 GB |
| Expert store | 4× Samsung 9100 PRO 1 TB on an x4x4x4x4 quad M.2 card, md RAID0 (512 KB chunk) + XFS. fio: 26 GB/s at 1.82 MB records, 1.12M 4K IOPS |
| CPU | AMD EPYC 7443P, server pinned to cores 26–39 |

## Run

`serve/n104_serve.sh` launches the server through a GPU lock wrapper (`bench/xpu_run.sh`). It expects:
- the SGLang + exl3xpu image (`24c872759256`);
- the model directory;
- the packed expert store (`qwen_experts.bin`: one 4K-aligned 1,863,680 B record per expert, index `layer*512+e`) at `/mnt/nvx/n104`.

The srv23 configuration:

```bash
STAGE=1 NVASYNC=1 HCXPU=1 HOSTBUF=5 RAM_GB=7 MAXTOT=131072 CHUNK=8192 MAXRUN=4 MAMBA=32 SLOTS=8000 \
GRAPH_BS=1,2,4 QSAIMPL=sycl_row QSASELECT=sycl \
  bench/xpu_run.sh 0000:84:00.0 n104-srv serve/n104_serve.sh runs/srv23
```

The OpenAI-compatible API listens on `127.0.0.1:30260` under the model name `flashnext`. Reproduce the tables with
`serve/n104_matrix.sh <run> 8192x1,8192x2,8192x4,32768x1,32768x2 <VARS...>`. It runs `serve/bench_matrix.py`: distinct
prompts per stream, completions to their natural end, and token timestamps for decode overlap and stalls.

Knobs (`serve/n104_serve.sh`, all default off):

| knob | effect |
|---|---|
| `STAGE=1` | staged prefill (NVMe → THP buffer → copy engine → VRAM) |
| `HOSTBUF` | depth of the host buffer ring (read-ahead = HOSTBUF−1 layers) |
| `NVASYNC=1` | async decode step |
| `HCXPU=1` | Triton hyper-connection kernels |
| `QSAIMPL=sycl_row` | SYCL sparse attention |
| `QSASELECT=sycl` | SYCL selection |
| `NOVICT=1` | no victim write-back (experimental) |

## Layout

| path | what |
|---|---|
| `serve/` | serve script, matrix launcher, benchmark client, smoke test |
| `plugin/` | SGLang exl3xpu plugin with tier mode (`sglang_plugin.tier.py`), `exl3xpu/nvtier.py` (NVMe/RAM/VRAM tier, staged prefill, async step), `exl3xpu/moe_offload.py` |
| `kernels/n107-hc/` | Triton hyper-connection norm/mix/combine for XPU, tests, patch |
| `kernels/n107-qsa/` | SYCL sparse attention and selection (ported from Strata), Triton variants, tests, serve integration |
| `kernels/n107-decode/` | async decode step patch |
| `results/` | `matrix.jsonl` per run, env, smoke outputs, and the SVM-vs-USM decode measurements |
| `research/` | B70 ingest-ceiling analysis and the lessons taken from Strata |

## Credits

- **Model:** Qwen3.8-Flash-Next by the Qwen team.
- **Quantization and kernels:** EXL3 quantization and exllamav3 by turboderp (MIT).
- **Sparse attention:** the SYCL attention and selection kernels are adapted from
  [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors (MIT); the notice is kept in each file.
- **Banner logo:** the Qwen mark comes from [lobe-icons](https://github.com/lobehub/lobe-icons) (MIT).
- **Licenses:** the code in this repository is MIT, and each upstream component keeps its own license.

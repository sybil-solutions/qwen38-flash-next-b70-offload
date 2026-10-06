# N107-qsa: QSA prefill attention on the B70 without the union and without host syncs

Plan item 4 of `kernels/xpu_bmg/n107-PLAN.md` (HOM-264, under cpu-nvme-offload). Written offline on the Mac: no Triton
compile and no GPU run has happened yet. On the Mac, only the torch emulators of the kernels, the sync-removal unit
tests and a mock install were run (results at the end).

## What it replaces

On XPU, `QwenSparseAttnBackend.forward_extend` calls `qsa_sparse_attention(q, k_buf, v_buf, slots, scale)`. exl3xpu sets
that name to `qsa_sparse_attention_union`, which for each of the 12 QSA layers of each prefill chunk:

* runs `token_slots[valid]` (nonzero), `torch.unique` over about 16.8M ids, `int(uni.numel())` and `searchsorted`.
  These are 2-3 host syncs.
* runs dense masked attention over the union `nu` of all selected keys. The scores are bf16 GEMMs, then fp32
  masked_fill, softmax and nan_to_num over [rows, 24, nu]. At nu = 8k that is about 83 GB of fp32 traffic per layer.
  The cost grows linearly with context (N106 analysis, section 1d).

The indexer adds 3 more host syncs per layer: `int(positions.max().item())` in `apply_rope`, once for q and once for
the compressed keys, plus `sequence_lengths.tolist()` in `get_prefill_mqa_inputs`.

Model facts used (from config and code): Hq 24, Hk 2 (G = 12), head_dim 256, scale 1/16. Indexer budget 2048 with
ratio 4 gives top-512 blocks, which expands to **2051 slots per row** (2048 plus up to 3 pending-tail tokens). The KV
pool is fp8_e4m3 without k/v scales.

## Files

| file | what |
|---|---|
| `qsa_sparse_triton.py` | Triton kernels and host wrappers: `qsa_sparse_attention_triton` (per-row, drop-in), `qsa_tile_attention` (block-union), `qsa_compact_row_attention`, `supports()` |
| `qsa_nosync.py` | sync-free `QSAIndexer.apply_rope`, `QSAIndexerMetadata.get_prefill_mqa_inputs`, and a host-context wrapper around `QSAIndexer.forward_cuda` |
| `patch_qsa.py` | env-guarded installer (`EXL3_QSA_TRITON=1`, default off). XPU only, never during graph capture. Falls back to the previous function for unsupported inputs or a compile error |
| `qsa_xpu_overlay.py` | mounted over `exl3xpu/qsa_xpu.py`. The original is mounted as `qsa_xpu_base.py`. Its `install()` runs the original install, then `patch_qsa.install()` |
| `qsa_xpu_base.py` | copy of the image's `exl3xpu/qsa_xpu.py` (from the N106 image dump). `apply_serve_qsa.py` re-extracts it from the image |
| `apply_serve_qsa.py` | adds the mounts and knobs to `n104_serve.sh` / `n106_serve_prof.sh`. Idempotent, keeps a `.pre_n107qsa` backup |
| `run_xpu.sh` | runs a script of this directory in image 24c872759256 on 84:00.0 (`--network none`, `--memory 16g`) |
| `qsa_ref.py` | pure-torch references (verbatim SGLang reference, exl3xpu union, top-k, expander), kernel emulators, input builder |
| `test_qsa_xpu.py` | accuracy and timing at 8k/32k with 8192 rows, launch sweep, sync tests, expander test, mock install, replay of real dumps |

## Kernels

All kernels have the same semantics as SGLang `qsa_sparse_attention_reference`, which is stricter than the union path:

* softmax runs over each row's valid slots (`slot >= 0`) only;
* a row with no valid slot outputs 0;
* GQA uses the `repeat_interleave` mapping (q head h reads kv head h // 12);
* `scale = softmax_scale or D**-0.5`.

Precision: bf16 operands. Scores are fp32, which is exact for bf16 products, unlike the union path's bf16 scores.
Softmax and accumulation are fp32 online softmax in exp2. P is rounded to bf16 for the PV product, as in the union path.

fp8 KV is decoded inside the kernel:

* `EXL3_QSA_TRITON_FP8=bits` (default): integer e4m3fn decode on a `uint8` view. It always compiles.
* `cast`: Triton fp8e4nv to bf16.

**Per-row (`row`, primary, the drop-in).** Grid = (row, kv head). One program loads its row's 12 q heads as an M=16
tile. It walks that row's 2051 slots in `BLOCK_N` chunks, gathering K/V straight from the paged pool by physical slot.
Chunks with no valid slot are skipped, which covers rows near the sequence start.

* Work is O(rows x 2051): it does not depend on context length or on overlap between rows.
* No unique, no union and no host sync.

**Block-union (`tile`, alternative).** It needs the logical indices and `token_slot_table`, so it hooks
`forward_extend`.

1. Rows of one sequence are grouped in tiles of BM=64.
2. `_qsa_bitmap` builds, on device, a bitmap of each row's selected logical positions (`atomic_or`, int32 words,
   R x ceil(S/32)). It also builds a per-(tile, KV block) "any row selected" flag. No unique and no sync.
3. `_qsa_tile_fwd` runs one program per (tile, q head). It walks the tile's causal KV blocks and skips unflagged ones.
   Each kept block of BN=64 contiguous logical positions is loaded once for 64 rows (logical to physical through the
   table). Every (row, key) pair is masked by its selection bit.

**Compact (`compact`).** Per layer, each sequence's visible K/V is copied once to bf16 in logical order (2 KB per token,
64 MB at 32k). The per-row kernel then indexes that copy. This removes the fp8 decode from the inner loop and gives the
gather a contiguous L2-friendly footprint.

### Trade-off: per-row vs block-union on Xe2

| | per-row | block-union (64-row tiles) |
|---|---|---|
| FLOPs | 2051 keys per row; M tile 16 with 12 useful (75% DPAS use) | 64 rows x every key of every flagged 64-token block, including keys the row did not select |
| K/V traffic | each selected key is re-read per (row, kv head): 17 GB/layer fp8 at 8192 rows, mostly L2 hits when the context fits L2 (8k fp8 KV = 8.4 MB) | each kept block is read once per (tile, q head): about 100x less |
| inner loop | gathered rows of 256 B (fp8), dot with M=16; bound by gather/L2 bandwidth and latency, not XMX | contiguous blocks and M=64 DPAS tiles, close to flash-attention efficiency |
| scaling with context | flat: the cost is fixed by the budget | grows with the union of the tile's selections. At 8k (25% density) almost every block is flagged, so about 2x the useful FLOPs. At 32k it is up to about 8x |
| extra | none | bitmap (32 MB at 32k, 8192 rows) + 16.8M atomics + flags |

Per-row is the right default because its cost does not grow with context, which is what the 7.8 -> 13.8 s ramp punishes.
Block-union wins only if per-row gathers turn out slow on Xe2 and the real selections of neighbouring rows overlap
heavily. Real overlap is unknown, so measure it with `QSADUMP=1` and `--dump`. `EXL3_QSA_TRITON_VARIANT=auto` with
`EXL3_QSA_TILE_MAX_CTX=<n>` uses tile up to n tokens of context and per-row beyond.

Why this should not repeat the rejected `EXL3_QSA_TRITON_EXTEND` result (2.5-6.5 s per 4k chunk, torch 2.8): that path
used SGLang's L20 launch table, which gives `BLOCK_N=16, num_warps=1` for more than 1024 queries. The chunked case also
did `.tolist()`, `.item()` and an `index_select` + `torch.cat` of the whole context's K/V per layer. Here the defaults are
`BLOCK_N=64, num_warps=8`, configurable and swept, with fp8 read in place and no cat.

## Sync removal (bit-identical)

* `apply_rope`: SGLang's `_ensure_cos_sin_cache_length(n)` returns when `n < len(cache)` and otherwise appends rows
  (glm53-sglang src_ref `rotary_embedding/base.py`). Existing rows never change.
  * For plain extend batches without multimodal inputs, every position is at most `max(seq_lens_cpu) - 1`, so the
    patch passes that host bound. When the original call would be a no-op, so is this one. Otherwise it appends rows
    the batch never reads.
  * The guard's source is checked once per rotary class. If it is not the known grow-only version, the patch keeps
    `.item()`.
  * Decode, verify, draft and multimodal batches keep the original code.
* `get_prefill_mqa_inputs`: reads `seq_lens_cpu[:n]` instead of `sequence_lengths.tolist()`.
* The union's unique/nonzero/numel go away with the kernels.
* The indexer expander switches from torch argsort to SGLang's own Triton expander (`EXL3_QSA_TRITON_EXPAND=1`). Its
  output is equal for exl3xpu's valid-first top-k layout; `--expand` checks this.

After the patch, a text-only QSA prefill layer has no host sync. `EXL3_QSA_NOSYNC_CHECK=1` re-reads the device values
and asserts they match the host bounds. It syncs, so use it for bring-up only.

## Run on omarchy (B70 84:00.0)

Copy the directory to the box first (from the Mac):

```
rsync -a ~/freetoken-exl3/kernels/xpu_bmg/n107-qsa/ omarchy:~/freetoken-exl3/kernels/xpu_bmg/n107-qsa/
```

Every GPU job goes through the lock wrapper. Run the usual guards (dsv41 /health 200, AER, `journalctl -k`) before and
after. The serve and test jobs both lock 84, so they cannot overlap.

```
cd ~/freetoken-exl3
Q=kernels/xpu_bmg/n107-qsa
# 1. correctness + sync removal + expander (about 5 min)
bench/xpu_run.sh 0000:84:00.0 n107-qsa $Q/run_xpu.sh test_qsa_xpu.py --nosync --expand --patch-mock --edge \
    --ctx 8192 --rows 2048 --variants ref,union,row,row_cast,compact,tile --out /n107qsa/results/t0_smoke.jsonl
# 2. timing at 8k and 32k context, 8192 query rows (fp8 KV like the server)
bench/xpu_run.sh 0000:84:00.0 n107-qsa $Q/run_xpu.sh test_qsa_xpu.py --ctx 8192 32768 --rows 8192 \
    --variants ref,union,row,row_cast,compact,tile --iters 5 --out /n107qsa/results/t1_8k_32k.jsonl
#    same with --sel uniform (worst case for tile)
# 3. launch sweep (row + tile, BLOCK_N / warps / stages / threads_per_warp, about 10-15 min), then re-run 2 with the best:
bench/xpu_run.sh 0000:84:00.0 n107-qsa $Q/run_xpu.sh test_qsa_xpu.py --sweep --ctx 8192 --rows 8192 \
    --out /n107qsa/results/t2_sweep.jsonl
EXL3_QSA_TRITON_CFG=64,8 EXL3_QSA_GRF=large bench/xpu_run.sh 0000:84:00.0 n107-qsa $Q/run_xpu.sh test_qsa_xpu.py ...
```

The CPU check also runs inside the image without the GPU (Triton interpreter):
`docker run --rm --network none -v $PWD/$Q:/n107qsa -w /n107qsa -e TRITON_INTERPRET=1 --entrypoint python3 24c872759256 test_qsa_xpu.py --device cpu --tiny --nosync --patch-mock`.
On the Mac without Triton, add `--emulate`.

Serving (integration):

```
python3 $Q/apply_serve_qsa.py runs/N104-nvtier/n104_serve.sh runs/N106-ingest/code-path/n106_serve_prof.sh
# bring-up: compare the first 4 calls against the union path, assert host bounds, dump real selections
QSATRITON=1 QSACHECK=4 QSANSCHECK=1 QSADUMP=1 STAGE=1 CHUNK=8192 \
  bench/xpu_run.sh 0000:84:00.0 n104-srv runs/N104-nvtier/n104_serve.sh runs/N107-qsa/srv-bringup
grep N107QSA runs/N107-qsa/srv-bringup/log.txt   # "installed ...", "check rows=8192 max_abs=..." (expect <= ~2e-2)
# A/B with the standard table (8k C1/C2/C4, 32k C1/C2), one variable at a time:
QSATRITON=0 ...   vs   QSATRITON=1 [QSAVAR=row|compact|tile|auto QSATILECTX=8192]
# attribution: n106_serve_prof.sh PMODE=events with QSATRITON=1: qsa_attn / qsa_indexer per-category ms per chunk
# real selections offline: test_qsa_xpu.py --dump /n107qsa/results/dump/qsa_topk_L*.pt --variants ref,union,row,tile
```

`QSATRITON=0` (the default after `apply_serve_qsa.py`) leaves the server unchanged. The overlay behaves exactly like
the original module and forwards attribute writes, so the N106 profiler's union wrap still works.

Pass criteria for the test:

* every kernel vs the fp32 reference: max_abs <= 3e-2, mean_abs <= 2e-3, all finite, zero rows exactly 0. The union
  path itself sits at about 1 bf16 ulp; the emulators reach half of that;
* nosync: bit-identical outputs and 0 `.item()`/`.tolist()` calls;
* expander: equal outputs.

## Expected gain (estimates; nothing measured yet)

* **Today (union).** Per the N106 estimate, about 150-200 ms per QSA layer at 8k context and about 4x that at 32k. That
  is about 1.8-2.4 s per 8k chunk at 8k context, growing about 2 s per extra 8k. srv18's per-chunk "other" ramp
  (7.8 -> 9.3 -> 11.0 -> 13.8 s) has that shape. Add about 60 host syncs per forward.
* **Per-row kernel.** About 0.41 TFLOP useful (0.55 issued) and 17 GB of fp8 gathers per layer at 8192 rows,
  independent of context.
  * At 20-30 TFLOPS effective, or gather-bound at 0.5-1.5 TB/s: about **20-45 ms per layer, 0.25-0.55 s per 8k chunk**.
  * It stays flat at 32k, apart from more L2 misses once the context's KV (33.5 MB fp8 at 32k) exceeds L2.
* **Net.** About 1.3-2 s less per 8k chunk at 8k context. Chunk 4 of a 32k prompt drops by about 6-9 s, so the
  per-chunk ramp should flatten. For a 32k prompt that is roughly 47 s to about 28-30 s of prefill (687 to about 1100
  tok/s at 32k C1), and about 8.8 to about 7 s per 8k chunk (+20-30% prefill tok/s at 8k).
  * These assume the N106 attribution of the ramp to the union holds. The `qsa_attn` events in n106 profiling decide it.
* **Not addressed (next target).** The indexer's fp32 einsum scores ([8192, S/4, 4], 1 GB at 32k), `torch.topk(512)`
  over S/4 columns and the int64 expand/l2p. That is about 15-50 ms per layer, growing with context. A fused Triton
  score+top-k kernel would remove most of it, but its selection must stay identical to the fp32 path (ties).

## Risks

* **The Triton code has never been compiled** (no Triton on macOS). Possible issues are the Intel backend's handling of
  `tl.trans` on gathered tiles, `atomic_or`, the fp8e4nv cast (the default `bits` path avoids it), and register
  pressure at D=256 (tile BM=64 holds a 64 KB fp32 accumulator; try `EXL3_QSA_GRF=large` or BM=32).
  * A compile or launch exception on first use logs `N107QSA ... failed` and falls back permanently to the union path.
  * A device-side fault would not be caught. Run test step 1 before serving.
* **First-use compile cost** is about 2-10 s per kernel config. It is one config per (dtype, D, G, BLOCK); rows and
  context are runtime arguments. Warm up the server as in N106 4.2.1.
* **Performance on Xe2 is unknown.** If per-row gathers are slow, try `compact` (bf16, logical order) or `tile`. If all
  are slow, the next step is an ESIMD gather + DPAS kernel in the exl3xpu lib.
* **Sync removal** assumes SGLang's `seq_lens_cpu` is exact for extend batches (SGLang invariant; `QSANSCHECK=1`
  asserts it) and that text-only positions are below `seq_len`. Multimodal batches keep the original path.
* **Numerics.** Outputs differ from the union path by bf16 rounding only. They are closer to SGLang's fp32 reference
  than the union is: scores are fp32, not bf16. The duplicate-slot semantics match the reference; duplicates cannot
  occur with the current expander anyway.
* **Scope.** Not touched: decode/graph replay (the patch is skipped while capturing, and replay uses the captured
  previous path), the speculative paged path, MTP capture (`is_last.nonzero()` when MTP is on), and the
  `hybrid_linear_attn_backend` syncs.

## Offline results (Mac, torch 2.10 CPU, no Triton)

`python3 test_qsa_xpu.py --device cpu --tiny --emulate --nosync --patch-mock`:

* fp8 bit decode equals torch's e4m3fn for all 254 non-NaN codes.
* Emulated per-row and tile kernels vs the fp32 reference: max_abs 3.9e-3 (1 bf16 ulp) and mean_abs 1.1-1.5e-4. That
  is half the union path's error (7.8e-3 / 3.8e-4). Covered cases:
  * S=256/R=128 and S=640/R=200, fp8 and bf16 KV, structured and uniform selection;
  * edge rows: all-invalid rows give exact zeros; holes and shuffled slots;
  * the vectorised reference matches SGLang's per-row loop to bf16 rounding.
* nosync: `apply_rope` (q and compressed-key paths, including cache-growth cases) and `get_prefill_mqa_inputs` (1 and 3
  sequences) are bit-identical, with 1 to 0 host reads each.
* Mock install against fake SGLang modules, all three variants: a 3-sequence chunked batch with prefixes and DP-padded
  rows reproduces the reference, with no fallback calls. Unsupported inputs route to the previous function.

## N108: SYCL per-query kernel (Strata port) - measured on the B70 (2026-10-06, HOM-261)

A SYCL port of Strata's per-query QSA attention (`qsa_decode_attn.dp.cpp`: chunk kernel, log-sum-exp merge, batched
launcher; MIT, attribution in the sources) as a torch XPU op library, plus a faster variant. It also ports the
indexer selection (`qsa_select.dp.cpp`: block scores + radix top-k). Built with icpx in image 24c872759256, with AOT
code for bmg_g31 and a spir64 fallback.

| file | what |
|---|---|
| `csrc/qsa_row_sycl.sycl` | `torch.ops.n108qsa.row_attn` (attention) and `fp8_decode_test` |
| `csrc/qsa_select_sycl.sycl` | `prefill_scores` / `prefill_select` (indexer scores + exact top-k) |
| `csrc/build_sycl.sh` | builds `build/qsa_row_sycl.so` in the image (~70 s, no GPU) |
| `csrc/cpu_test.cpp`, `csrc/cpu_test_select.cpp` | the same kernels on the image's OpenCL CPU device vs an fp64 reference |
| `qsa_sycl.py` | loader, `supports()`, `qsa_sparse_attention_sycl()`, `supports_select()`, `prefill_select()` |
| `test_qsa_select_xpu.py` | selection: current torch path vs SYCL (time, identical sets, the patched SGLang method) |
| `n108_job.sh`, `n108_tests.sh`, `n108_tests2.sh` | guarded lock-wrapped GPU jobs (health, `journalctl -k`, AER before and after) |

### Attention kernel

The kernel reads K/V in place via the physical slots. There is no gather copy, no union and no host sync. It
decodes fp8 e4m3fn in the kernel (bf16/fp16 KV also work). Scores, softmax and accumulation are fp32; the output is
bf16. The semantics match `qsa_sparse_attention_reference`: softmax over valid slots only, and an empty row gives an
exact 0.

* **kernel 1 "strata"**: Strata's structure. Chunks of 64 cells; one sub-group per cell for Q.K (lanes hold dims,
  sub-group reduction); chunk softmax; P.V with one thread per dimension.
  * `cpw` = chunks per work-group. `cpw=1` is Strata's split-K with a merge kernel; `cpw=0` runs all chunks in one
    work-group per (row, kv head) with online softmax and no merge.
* **kernel 2 "bcast"** (the default): Q.K with one lane per cell. Each sub-group's q slice is loaded once per 16
  dims and broadcast lane by lane, so there are no reductions. Chunks are 256 cells.
* `EXL3_QSA_SYCL_CFG=kernel,cpw,sg,grf,fp8`; the default is `2,0,16,128,1`. `fp8=1` decodes through fp16 bits (exact).

Results, ctx x 8192 rows, fp8 KV, structured selection (`results/n108_t1_8k_32k.jsonl`, `n108_t2_sweep.jsonl`,
`n108_j2_sweep32.jsonl`):

| variant | 8K ms/layer | 32K ms/layer | TFLOPS | max_abs vs fp32 ref |
|---|---|---|---|---|
| union_exl3xpu (today) | 255.1 | 459.0 | 1.4 / 0.9 | 1.6e-2 / 3.9e-3 |
| Triton row (N107) | 1704 | 1949 | 0.2 | 1e-3 |
| **sycl_row (bcast, default)** | **55.9** | **65.5** | 6.4 | 2.0e-3 / 9.8e-4 |
| sycl_strata (Strata's launch: split-K 64-cell + merge, sg32, exact fp8) | 111.4 | 128.0 | 3.2 | 2.0e-3 |

* Speedup over union: 4.6x at 8K, 7.0x at 32K. Cost per query-layer: 6.8 us. Strata's own engine measured ~16 us;
  our faithful port measures 13.6 us.
* Projected saving per 8K chunk (12 QSA layers): **2.39 s at 8K context, 4.72 s at 32K**. The saving grows with
  context because the kernel's cost barely does (+17% from 8K to 32K).
* Sweep at 8K (ms): bcast sg16 55.3; bcast sg32 with large GRF 55.8; bcast sg32 59.7; bcast split-K cpw 1/2/4:
  63.9 / 60.3 / 58.6; bcast with exact fp8 decode 64.3; large GRF with sg16 99.9 (worse). Strata kernel:
  cpw=all 76.9, split-K 100.8-110.9. 32K: same ranking, best 63.7.
* Correctness: every configuration passes the n107 tolerances.
  * Edge rows pass: empty rows give exact 0, plus holes and shuffles. bf16 KV passes.
  * The device fp8 decode equals torch for all 254 non-NaN codes, on both routes.
  * The CPU-device test passes all 28 configurations, including the split-K merge with row batches.

### Selection kernels

`prefill_select` computes `relu`-summed indexer scores with fp32 dots of bf16 values, one lane per block. It then
runs an exact radix top-k per row; ties at the threshold go to the lowest index, which is Strata's rule.
Results at 8192 rows, random q/keys (`results/n108_j2_select.jsonl`, `n108_j3_select.jsonl`):

| ctx | current (einsum fp32 + relu-sum + torch.topk, row-chunked) | SYCL select | saved per 8K chunk (12 layers) |
|---|---|---|---|
| 8K (2048 blocks) | 7.54 ms | 1.54 ms (rpw 8) | 0.07 s |
| 32K (8192 blocks) | 49.8 ms | 7.8 ms | 0.50 s |

* Selected sets are identical for 100% of rows at both contexts. The SYCL scores equal the torch logits bit for bit
  on a 512-row sample.
* SGLang's real `QSAIndexer.select_prefill_tokens` (with exl3xpu's top-k) was run against the patched method on a
  2-sequence batch (row_starts > 0, 8000 rows). The outputs give identical token sets on every row.
* No real selection dumps exist yet (`QSADUMP` was never run), so real-selection replays are still pending.

### Enabling (env-guarded; default off)

* `EXL3_QSA_IMPL=sycl_row`: `qsa_sparse_attention` takes the SYCL kernel for XPU, bf16 q, D=256, Hq=12*Hk,
  16-byte-aligned K/V rows and at least `EXL3_QSA_SYCL_MIN_ROWS` (16) rows, outside graph capture.
  * Everything else, and any load or launch exception, falls back to the previous function (union/reference).
  * In this mode the N107 indexer patches (nosync, Triton expander) stay off unless set explicitly.
* `EXL3_QSA_SELECT=sycl`: the indexer prefill selection uses the SYCL kernels. `EXL3_QSA_SELECT_CHECK=N` compares
  the first N calls against the torch path and logs the result.
* Serve scripts: `apply_serve_qsa.py` adds `QSAIMPL`, `QSASYCLCFG`, `QSASYCLMIN`, `QSASELECT` and `QSASELCHECK`.
  The overlay calls `patch_qsa.install()` when any of `EXL3_QSA_TRITON`, `EXL3_QSA_IMPL` or `EXL3_QSA_SELECT` is set.
* Bring-up:
  `QSAIMPL=sycl_row QSASELECT=sycl QSACHECK=4 QSASELCHECK=4 STAGE=1 CHUNK=8192 bench/xpu_run.sh 0000:84:00.0 ...`.
  Then grep `N107QSA` for `installed impl=sycl_row`, the `check ... max_abs` lines and the
  `select check ... identical_rows=1.000000` lines.

Rebuild after editing the kernels (no GPU needed):
`docker run --rm --network none --memory 16g --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/n107qsa -w /n107qsa --entrypoint bash 24c872759256 csrc/build_sycl.sh`

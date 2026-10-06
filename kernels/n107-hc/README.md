# n107-hc: Triton hyper-connection kernels for Qwen3.8-Flash-Next on the B70 (XPU)

N107 plan item 3. Replaces the slow eager and inductor hyper-connection path in SGLang `layers/hyperconnection.py`
(`GatedResidual`, 97 mix + 96 combine per forward) for XPU tensors. Every fused CUDA version (the
`grouped_gemma_rmsnorm` JIT, `hc_mix`, `fused_hc_mix`, `hc_combine`) is gated on `is_cuda` or sm100, so the XPU
currently runs:

- eager fp32 `GroupedGemmaRMSNorm` (about 9 kernels and about 2.9 GB of traffic per call at 8192 rows);
- `torch.compile(_mix_compute)`;
- `torch.compile(_combine_compute)`.

Status: written offline on the Mac and not yet run on XPU. Under the Triton 3.7.1 interpreter (built from source on
the Mac) all kernels are bit-identical to the reference for M in {1,2,4,8} and within 1 ulp for M in {67,256,600}.
See "Offline validation" below.

| file | what |
|---|---|
| `hc_xpu.py` | Triton kernels plus host wrappers: `gemma_rmsnorm_grouped`, `hc_mix(mode=epilogue\|fused\|fused_silu)`, `hc_combine(inject=torch\|fused)`, `*_supported()` |
| `test_hc_xpu.py` | Correctness against the verbatim reference: bit-equality, max abs/rel error, max ulp. Also timing against eager and compiled, an XPU graph-capture test, an end-to-end `patch_hc` test on a real `GatedResidual`, and an optional launch-config sweep. Writes `results_<dev>_<time>.json` next to itself. |
| `patch_hc.py` | Monkeypatch, off unless `EXL3_HC_XPU=1`. Covers XPU tensors only; anything it does not support goes to the original code. |

## Semantics (what is reproduced, and where the rounding happens)

The model config uses `hc_per_branch_norm=True`, which gives `GroupedGemmaRMSNorm(10240, group_size=2560)`: one
10240-wide weight, i.e. a separate weight per branch. The `per_branch=False` layout (`[M, hc, hs]` with a shared
`[hs]` weight) is also supported and tested.

| op | reference that runs on XPU today | kernel |
|---|---|---|
| norm | `xf=x.float(); var=mean(xf^2) over 2560` (torch mean = sum * fp32(1/G)); `y=(xf*rsqrt(var+eps))*(1+w.float())` -> bf16 | One read and one write. Program = (row, branch). Single pass with a masked 4096 block; a loop variant is used when the group exceeds 8192. |
| mix | `t=F.linear(n,Wd)` (bf16 out), `a=silu(t.f32/hc)` (one rounding), `u=F.linear(a,Wu)` (bf16 out), `o=(sum_j sigmoid(u_j)*n_j)/hc` in fp32, one rounding | `epilogue` (default): both GEMMs stay as the same `F.linear` calls (oneDNN); Triton runs silu and the sigmoid*x mean. `fused`: the down GEMM stays in torch; Triton runs the 320->10240 up-projection with `tl.dot` per branch and does the epilogue in registers, rounding `u` to bf16 before the sigmoid. The [M,10240] `u` is never written. |
| combine | `l=F.linear(n,Wi)` (bf16 [M,hc]), `g=2*sigmoid(l.f32/hc)` **stored as bf16**, `o=R+bo*g` in fp32, one rounding | `torch` (default): same `F.linear` for `l`, then one Triton pass (reads R and bo, writes the output). `fused`: `l` is computed in-kernel (fp32 accumulation, rounded to bf16). |

How the rounding points were established: a pure-torch emulation with exactly these roundings is bit-identical
(100.000 %) to `torch.compile(_mix_compute)` and `torch.compile(_combine_compute)` on CPU inductor (torch 2.10).
The generated code shows the realization of `g` as bf16: inductor writes `g` in place into the mm output buffer as
bf16, then the broadcast loop reads it back. The eager, uncompiled functions differ from the compiled ones in about
49 % (mix) and about 27 % (combine) of elements, by 1 ulp. That eager-vs-compiled spread is the test's noise floor.

**Why `epilogue` is the default mix mode.** It keeps both GEMMs on oneDNN, which is well tuned and bit-identical to
the reference GEMMs. The Triton part is a pure streaming epilogue with no DPAS or register-tiling risk. `fused` removes
336 MB of traffic per mix (about 0.6 ms at M=8192), but it depends on Triton-XPU `tl.dot` codegen (2D block loads
of a column-major B, register pressure with two [BM,BN] fp32 tiles), and its GEMM summation order differs from
oneDNN. Switch with `EXL3_HC_MIX=fused` once `test_hc_xpu.py` shows that it passes and is faster on the B70; the test
prints the best mode.

## Run the test on the B70 0000:84:00.0 (host omarchy, from ~/freetoken-exl3)

Follow the N107 guard first: dsv41 `/health` must return 200, and there must be no AER, pciehp or Completion-Wait
lines for 80:..84:.

```bash
cd ~/freetoken-exl3
bench/xpu_run.sh 0000:84:00.0 n107-hc bash -c 'docker run --rm --network none \
  --device $DSV41_XPU_RENDER:$DSV41_XPU_RENDER:rwm -e ZE_AFFINITY_MASK=0 --memory 16g \
  -v $HOME/freetoken-exl3/kernels/xpu_bmg/n107-hc:/w --entrypoint python3 24c872759256 /w/test_hc_xpu.py'
```

- Default M values are {1, 2, 4, 8, 256, 8192}. The run covers correctness, timing (`torch.xpu.Event`, 3 warm-up and
  20 timed iterations), an XPU graph capture and replay at M=1 and M=4 (new values written into the static inputs
  must show up in the replay), and the `patch_hc` end-to-end test on a `GatedResidual` built from the image's sglang.
  It also checks the image's `hyperconnection.py` for drift against the reference copy (10 anchor lines). Expect a
  few minutes, most of it inductor and Triton JIT compiles. Device memory use is about 2 GB.
- Add `--sweep` for a launch-config sweep at M=8192 (norm BLOCK/warps/threads-per-warp, epilogue tiles, up-fused tiles
  with block pointers on and off, combine tiles), then put the winners into the defaults or into `EXL3_HC_*_CFG`.
- Output: a PASS/FAIL line per check, then `=== ALL PASS`. The exit status is 0 only if everything passed. A JSON
  file `results_xpu_<time>.json` is written into the mounted dir; it is created by root inside the container.
- Projection line: `[projection M=8192] per forward (97 mix incl. norm + 96 combine): reference X ms -> n107-hc Y ms`.
- Useful variants: `-e EXL3_HC_TPW=16` (subgroup 16), `-e EXL3_HC_UP_BLOCKPTR=0` (plain-pointer `tl.dot` loads),
  `--ms 1,8192`, `--no-graph`, `--no-patch`.
- CPU or interpreter run inside the image, with no GPU needed:
  `docker run --rm --network none -e TRITON_INTERPRET=1 -v ...:/w --entrypoint python3 24c872759256 /w/test_hc_xpu.py --device cpu`
  This uses M in {1,2,4,8} by default; `--ms 67,256,600` takes about 90 s on an M-series Mac.

### What to read in the output

- `norm grouped vs eager`: expect max_ulp <= 1 and bit_eq >= 99.99 %. Not bit-exact, because the fp32 reduction
  order and the `rsqrt` implementation differ from the torch-xpu reduce kernel.
- `mix[epilogue] vs compiled`: expect close to 100 % bit-equal, since the GEMMs are identical; at most the
  hc-mean summation order differs.
- `mix[fused] vs compiled`: expect >= 99.99 % bit-equal with max_abs well under the floor.
- `combine[torch] vs compiled`: expect 100 % if inductor-XPU realizes `g` as bf16, as CPU inductor does. **If instead
  `combine[torch,round_g=0]` is the bit-exact one on XPU, set `EXL3_HC_ROUND_G=0`.**
- `graph replay vs eager`: must be exact. If it fails, keep `EXL3_HC_XPU=0` for the graph-mode decode server.

## Integration (serve)

1. Add these arguments to `runs/N104-nvtier/n104_serve.sh` (the `docker run` line):
   ```
   -v $HOME/freetoken-exl3/kernels/xpu_bmg/n107-hc:/hc:ro -e EXL3_HC_DIR=/hc -e EXL3_HC_XPU=${HCXPU:-0} \
   -e EXL3_HC_MIX=${HCMIX:-epilogue} -e EXL3_HC_COMBINE=${HCCOMB:-torch} \
   -e TRITON_CACHE_DIR=/tcache -v $HOME/.cache/n107-triton:/tcache
   ```
   The last line is optional. It persists Triton JIT binaries across `--rm` containers.
2. Add the hook to `sglang_plugin.tier.py` `activate()`, after `_cuda_compat_shim()`:
   ```python
   if os.environ.get("EXL3_HC_XPU", "0") == "1":
       try:
           import sys as _sys
           _sys.path.insert(0, os.environ.get("EXL3_HC_DIR", "/hc"))
           import patch_hc
           patch_hc.install(import_now=True)
       except Exception as e:  # pragma: no cover
           logger.warning("exl3xpu: n107-hc not installed (%s)", e)
   ```
   The alternative without a plugin edit is to `import patch_hc` before the model is built (for example from a
   sitecustomize on `PYTHONPATH`). It arms an import hook that patches `sglang.srt.layers.hyperconnection` right after
   it is first imported. `patch_hc.patch_model(model)` re-wires instances that were built earlier.
3. Check that it is live: the server stderr must show `n107-hc: installed (...)`, then
   `n107-hc: norm on XPU via Triton, first shape (8192, 10240)`, `mix[epilogue] ...` and `combine[torch] ...`. It
   must not show `WARNING ... kernel disabled`; a kernel that raises once is disabled for the rest of the process
   and the server falls back to the original path. `patch_hc.STATS` counts calls per path.
4. Correctness gate before speed: `xpu/bench/ref_panel_client.py` against the E003b reference panel, as in X004.
   Then run the standard N107 table (8k C1/C2/C4, 32k C1/C2) with `HCXPU=1` against `HCXPU=0`.

CUDA is untouched: the patched callables only take the Triton path for `x.device.type == "xpu"`, and the CUDA JIT
branches in `mix()` and `combine()` run before them unchanged. On XPU the original `torch.compile` objects are never
called, so there are no inductor compiles or recompiles at serve time (candidate (a) of the code-path map's 1j
anomaly).

## Expected speedup (estimates; the test measures them)

These assume a B70 at about 608 GB/s peak, with Triton streaming kernels reaching about 500 GB/s. Bytes are bf16:
`[8192, 10240]` = 168 MB.

| per call at M=8192 | today (code_path_map 1b) | n107-hc | basis |
|---|---|---|---|
| hc_norm | 7-8 ms | 0.6-0.7 ms | 336 MB, one read and one write |
| mix compute, `epilogue` | 2-3 ms | 2.0-2.6 ms | 2 oneDNN GEMMs (unchanged, 107 GFLOP) + silu + a 378 MB epilogue |
| mix compute, `fused` | | 1.3-2.0 ms | down GEMM + one kernel (reads 168 MB, writes 42 MB, 54 GFLOP on DPAS) |
| combine | 1.5-2 ms | 1.0-1.3 ms | `l` GEMM reads 168 MB + one 378 MB pass |
| **per 8k chunk: 97 x (norm+mix) + 96 x combine** | **1.0-1.3 s** | **~0.40 s (epilogue) / ~0.33 s (fused)** | |

That is a saving of about 0.7-0.9 s per 8k prefill chunk, almost all of it from the norm. The pure-bandwidth floor
for this structure is about 0.22 s (122 GB per forward). Reaching the ~0.3 s goal needs the `fused` mix mode and
oneDNN GEMMs near roofline.

In decode (graph replay, M=1..4), the eager norm's about 9 kernels become 1, about 780 fewer graph nodes per token.
That should save about 1.5-4 ms per token (roughly +3-9 % decode tok/s at 46 ms/token); this has not been measured.

Further fusions are not implemented. The biggest is combine plus the next norm: the `mlp`/`attn` combine output is
re-read immediately by the next mix's norm, so fusing them saves 168 MB per call (about 30 ms per forward plus 96
launches). It needs a cross-module patch (`_prepare_qwen4_exp_mlp` and layer i -> i+1).

## Numerical risks

1. **The norm is not bit-exact against eager.** The fp32 sum-of-squares order and `tl.math.rsqrt` against torch-xpu's
   reduce and `rsqrt` can move a bf16 rounding by 1 ulp (the interpreter shows 99.999 % bit-equal). This is the same
   class of change as any reduction-order change, but greedy outputs can diverge from today's server after many
   tokens. Gate with the reference panel, not with string equality.
2. **`g` rounding in combine.** Matching inductor-XPU depends on inductor realizing `g` as a bf16 buffer. That was
   verified on CPU inductor and is expected on XPU (the producer and consumer have different iteration spaces), but
   it is not verified on XPU. The test reports both variants, and `EXL3_HC_ROUND_G` selects. The difference is at
   most 1 bf16 ulp of `g`.
3. **`fused` mix uses a different GEMM summation order.** The order differs from oneDNN, so `u` can move by 1 bf16
   ulp before the sigmoid. Outputs that are near zero through cancellation across branches then show large ulp
   counts, but the absolute error stays tiny (interpreter: max_abs 5e-4 against a floor of 3e-2 at M=256). The
   `epilogue` mode avoids this entirely.
4. **FMA contraction and fast math.** `R + bo*g` and `sigmoid` may be contracted or use native `exp` on the Intel
   backend. inductor's own Triton code goes through the same compiler, so it should match the compiled reference; it
   will not match eager torch exactly (it does not today either).
5. **hc_count = 5 (MTP role).** Division by hc uses true division like inductor (`/ 4.0` in the generated code). For
   hc=5 that is not a power of two, so 1-ulp differences are possible if inductor-XPU used a reciprocal instead.
6. **Coverage.** The kernels are inference only (no autograd). Inputs must be contiguous, bf16 or fp16, on xpu, with
   weights in the same dtype (the norm weight may be fp32, bf16 or fp16); anything else falls back. Row offsets are
   int64 and block-pointer shapes are int32 (M up to about 2e5 rows at 10240 wide).
7. **The interpreter is not the device.** The Triton 3.7 interpreter truncates fp32->bf16 casts and runs bf16
   `tl.dot` on raw uint16 bits. `test_hc_xpu.py` patches both only under `TRITON_INTERPRET=1`. The interpreter does
   math in numpy fp32 (exact `exp`, no FMA) and validates indexing and semantics only: not XPU codegen, register
   pressure, `threads_per_warp` or performance.
8. **First-call compiles.** Each new (kernel, constexpr) combination is JIT-compiled on first use: norm 1, silu 1,
   epilogue 2 (M<64 and M>=64 tiles), combine 2. That happens in warm-up and graph-capture warm-up, but without a
   persistent `TRITON_CACHE_DIR` every fresh `--rm` container pays it again.

## Offline validation done on the Mac (2026-10-06)

Triton v3.7.1 was built from source with the interpreter only (`scratchpad/tb`). The test ran with
`TRITON_INTERPRET=1 --device cpu` against the image's `hyperconnection.py`, loaded through a stub sglang package
(`hc_mix_triton` stubbed). The references are torch 2.10 CPU inductor and eager. See `interpret_run_mac.log`.

- M=1,2,4,8 at full dims (hc 4, hs 2560, lr 320): every norm, mix (all 3 modes, block pointer and plain pointer) and
  combine (both inject modes) check is 100 % bit-equal, with one 0.02 % 1-ulp exception (mix fused at M=2). The
  `patch_hc` dispatch and fallback were checked on CPU tensors, which correctly take the original path.
- M=67,256,600 (multi-row tiles, masked tails): norm 99.999 % (max 1 ulp), mix 99.998-99.999 %, combine 100 %.
- `combine[round_g=0]` is only 71-78 % bit-equal, which confirms that the bf16 rounding of `g` is real.
